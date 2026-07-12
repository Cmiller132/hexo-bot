"""Lab internals: position featurization (sequence and free-edit), the hooked
forward that exposes attention rows and per-block activation norms, and the
capped lab search.

Runs inside bot worker processes only (imports torch/shrimp at module level;
the web process never imports this module — request validation lives in
``lab_rules``, which is engine-only).

Position modes:

- sequence: the placement list replays through the engine and featurizes via
  the exact serve path (``facts_from_state`` -> ``build_support`` ->
  ``build_features``), identical to what the search sees.
- free-edit: there is no move history, so ``PositionFacts`` is synthesized
  from the raw stones (records in a fixed deterministic order, phase pinned to
  FirstStone, no first_stone) and the three history-derived feature columns —
  own/opp recency and opponent-last-turn — are zeroed after the build. The
  threat/standing-win planes come from ``window_scan`` over the stones, which
  is order-independent, so they are real.

Attention rows use the same forward-hook mechanism as
``scripts/learn_snapshots.py``: a hook on every ``attn_blocks[i].attn``
recomputes q/k with the module's own projections and takes
softmax((q k^T) * scale + bias) — identical math to the module's
'materialized' impl. Hooks are registered, the forward runs once, and the
hooks are removed in a ``finally``, so workers stay stateless between jobs.

Payload bounds: attention rows are emitted only for the single queried cell
(blocks x heads rows, sparse above ``ATTN_FLOOR``); activation norms are one
float per support node per trunk stage (stem + every block).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from shrimp.batching import collate_rows
from shrimp.constants import (
    F_OPP_LAST_TURN,
    F_OPP_RECENCY,
    F_OWN_RECENCY,
    NUM_FEATURES,
    NUM_TOKENS,
)
from shrimp.engine_facts import facts_from_state
from shrimp.features import PHASE_FIRST, PositionFacts, build_features, window_scan
from shrimp.geometry import unpack_action_id
from shrimp.losses import decode_binned_value, decode_moves_left
from shrimp.model import STV_HORIZONS
from shrimp.support import build_support

from .jsonsafe import sanitize_json

# Sparse-attention floor: cells attended below this weight are dropped from
# the payload (the token row is always dense — 8 floats).
ATTN_FLOOR = 1e-3
ROUND_ATTN = 4
ROUND_NORM = 3
ROUND_P = 6

TOP_K = 5

# Free-edit zeroed feature columns (the page's persistent free-mode note and
# the ``zeroed_features`` payload field name them).
FREE_ZEROED = ("own_recency", "opp_recency", "opp_last_turn")
_FREE_ZEROED_COLS = (F_OWN_RECENCY, F_OPP_RECENCY, F_OPP_LAST_TURN)

# Payload feature names, index-aligned with the constants (F_* order 0-14).
FEATURE_NAMES = (
    "own_stone", "opp_stone", "empty", "legal", "phase_second", "first_stone",
    "player_colour", "own_recency", "opp_recency", "opp_hot", "own_hot",
    "dist_to_stone", "opp_last_turn", "opp_win_now", "own_win_now",
)


# ---------------------------------------------------------------------------
# Position building
# ---------------------------------------------------------------------------


def replay_state(cells: list[tuple[int, int]]) -> Any:
    """Engine state after a validated chronological placement list."""
    import hexo_engine as engine
    from hexo_engine.types import AxialCoord, PlacementAction

    state = engine.new_game()
    for q, r in cells:
        engine.apply_action(state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
    return state


def build_sequence_position(cells: list[tuple[int, int]]) -> tuple[PositionFacts, Any, np.ndarray]:
    """(facts, support, features) for a legal placement sequence — the exact
    serve featurization path."""
    facts = facts_from_state(replay_state(cells))
    support = build_support(facts.stones())
    return facts, support, build_features(facts, support)


def build_free_position(
    p0: list[tuple[int, int]], p1: list[tuple[int, int]], to_move: int,
) -> tuple[PositionFacts, Any, np.ndarray]:
    """(facts, support, features) for a free-edit stone set.

    Synthesized history: records carry the stones in a fixed deterministic
    order ((owner, q, r) sorted) purely so the stone/empty planes fill; the
    ordinals are meaningless as a chronology, so the recency and
    opponent-last-turn columns built from them are zeroed afterwards
    (``FREE_ZEROED``). Phase is pinned to FirstStone (the start of the mover's
    turn): phase_second and first_stone are structurally zero.
    """
    ordered = sorted([(q, r, 0) for q, r in p0] + [(q, r, 1) for q, r in p1])
    records = tuple(
        (int(q), int(r), owner, idx) for idx, (q, r, owner) in enumerate(ordered)
    )
    own_hot, opp_hot, own_win, opp_win = window_scan(records, to_move, len(records))
    facts = PositionFacts(
        records=records,
        current_player=int(to_move),
        phase=PHASE_FIRST,
        first_stone=None,
        own_hot=own_hot,
        opp_hot=opp_hot,
        own_win=own_win,
        opp_win=opp_win,
    )
    support = build_support(facts.stones())
    feats = build_features(facts, support)
    feats[:, list(_FREE_ZEROED_COLS)] = 0.0
    return facts, support, feats


# ---------------------------------------------------------------------------
# Hooked forward
# ---------------------------------------------------------------------------


def hooked_forward(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    capture_attention: bool,
    capture_activations: bool,
) -> tuple[dict[str, torch.Tensor], list[np.ndarray], dict[int, np.ndarray]]:
    """Full-head forward with optional attention / activation capture.

    Returns ``(outputs, attn_mats, activations)`` where ``attn_mats[i]`` is the
    (heads, S, S) float32 softmax of attention block i and ``activations`` maps
    ``id(module)`` to that module's captured (nodes,) L2 activation norms.
    Runs under ``torch.enable_grad()`` so ``build_attn_bias`` takes the fp32
    master-table path (the same trick as ``analysis.net_eval``); everything is
    detached before use. Hook registration is paired with removal in a
    ``finally`` so a failed forward never leaves hooks behind.

    Inputs follow the model's device (same contract as
    ``analysis._model_forward``): the batch comes off ``collate_rows`` on the
    CPU and moves to the accelerator when the worker serves there; every
    captured tensor is brought back to the CPU before use.
    """
    device = next(model.parameters()).device
    if device.type != "cpu":
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    handles: list[Any] = []
    attn_mats: list[np.ndarray] = []
    activations: dict[int, np.ndarray] = {}

    def _norms(cells: torch.Tensor) -> np.ndarray:
        # (N, C) -> (N,) float32 L2 norms.
        return torch.linalg.vector_norm(cells.detach().float(), dim=-1).cpu().numpy()

    if capture_attention:
        def make_attn_hook(block_index: int):
            def hook(module, args, output):
                seq, attn_bias = args[0].detach(), args[1].detach()
                b, s, c = seq.shape
                h, d = module.heads, module.head_dim
                with torch.no_grad():
                    q = module.q_proj(seq).reshape(b, s, h, d).transpose(1, 2)
                    k = module.k_proj(seq).reshape(b, s, h, d).transpose(1, 2)
                    scores = (q @ k.transpose(-2, -1)) * module.scale
                    scores = scores + attn_bias.to(scores.dtype)
                    attn = torch.softmax(scores.float(), dim=-1)
                attn_mats.append(attn[0].cpu().numpy())  # (heads, S, S); batch is 1
            return hook

        for i, block in enumerate(model.attn_blocks):
            handles.append(block.attn.register_forward_hook(make_attn_hook(i)))

    if capture_activations:
        def stem_hook(module, args, output):
            # x entering the first block is relu(stem_ln(stem(...))) * mask;
            # batch size 1 has no pad rows, so relu(output) is exactly x.
            activations[id(module)] = _norms(torch.relu(output[0]))

        def conv_hook(module, args, output):
            activations[id(module)] = _norms(output[0])

        def attn_block_hook(module, args, output):
            # AttnBlock outputs the joint [tokens; cells] sequence.
            activations[id(module)] = _norms(output[0, NUM_TOKENS:])

        handles.append(model.stem_ln.register_forward_hook(stem_hook))
        for block in model.conv_blocks:
            handles.append(block.register_forward_hook(conv_hook))
        for block in model.attn_blocks:
            handles.append(block.register_forward_hook(attn_block_hook))

    try:
        with torch.enable_grad():
            out = model.forward(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        out = {k: v.detach() for k, v in out.items()}
    finally:
        for handle in handles:
            handle.remove()
    return out, attn_mats, activations


def _sparse_attention_row(row: np.ndarray) -> dict[str, Any]:
    """One (S,) softmax row -> {'tokens': [8 floats], 'cells': {node: w}}."""
    tokens = [round(float(w), ROUND_ATTN) + 0.0 for w in row[:NUM_TOKENS]]
    cells = {
        str(i): round(float(w), ROUND_ATTN) + 0.0
        for i, w in enumerate(row[NUM_TOKENS:])
        if w >= ATTN_FLOOR
    }
    return {"tokens": tokens, "cells": cells}


def _activation_blocks(model: Any, activations: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    """Per-stage activation payload in trunk execution order: stem, then every
    block of the layout string (conv/attn interleaving preserved)."""
    stages: list[tuple[str, str, Any]] = [("stem", "stem", model.stem_ln)]
    ci = ai = 0
    for kind in model._trunk_layout:
        if kind == "C":
            ci += 1
            stages.append((f"conv{ci}", "conv", model.conv_blocks[ci - 1]))
        else:
            ai += 1
            stages.append((f"attn{ai}", "attn", model.attn_blocks[ai - 1]))
    return [
        {
            "label": label,
            "kind": kind,
            "norms": [round(float(v), ROUND_NORM) for v in activations[id(module)].tolist()],
        }
        for label, kind, module in stages
    ]


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


def _sparse_policy(logits: torch.Tensor, support: Any, floor: float) -> list[dict[str, Any]]:
    """Softmax over the legal prefix -> sparse rows sorted by p descending."""
    legal_count = support.legal_count
    if legal_count <= 0:
        return []
    priors = torch.softmax(logits[:legal_count].float(), dim=0)
    rows = [
        {"q": int(q), "r": int(r), "p": round(float(p), ROUND_P)}
        for (q, r), p in zip(support.legal_coords().tolist(), priors.tolist())
    ]
    rows.sort(key=lambda row: (-row["p"], row["q"], row["r"]))
    return [row for row in rows if row["p"] >= floor]


def eval_payload(
    model: Any,
    facts: PositionFacts,
    support: Any,
    feats: np.ndarray,
    *,
    policy_floor: float,
    attention_cell: tuple[int, int] | None,
    want_activations: bool,
    want_features: bool,
) -> dict[str, Any]:
    """The full lab-eval payload for one prepared position.

    Raises ``ValueError`` (mapped to 422 by the endpoint) when
    ``attention_cell`` is not a support cell of this position.
    """
    attention_node: int | None = None
    if attention_cell is not None:
        attention_node = support.index.get((int(attention_cell[0]), int(attention_cell[1])))
        if attention_node is None:
            raise ValueError(
                f"attention query ({attention_cell[0]}, {attention_cell[1]}) "
                "is not in this position's support set"
            )

    batch = collate_rows([(support, feats)])
    out, attn_mats, activations = hooked_forward(
        model, batch,
        capture_attention=attention_node is not None,
        capture_activations=want_activations,
    )

    value_logits = out["value"][0].reshape(1, -1).float()
    value_dist = torch.softmax(value_logits[0], dim=0)
    policy_rows = _sparse_policy(out["policy"][0], support, policy_floor)
    payload: dict[str, Any] = {
        "to_move": int(facts.current_player),
        "phase": str(facts.phase),
        "ply": int(facts.placements_made),
        "legal_count": int(support.legal_count),
        "support": {
            "coords": support.coords.tolist(),
            "legal_count": int(support.legal_count),
            "stone_count": int(support.stone_count),
            "halo_count": int(support.halo_count),
        },
        "value": round(float(decode_binned_value(value_logits).item()), ROUND_P),
        "value_dist": [round(float(p), 5) for p in value_dist.tolist()],
        "stv": {
            str(h): round(
                float(decode_binned_value(out[f"stvalue_{h}"][0].reshape(1, -1).float()).item()),
                ROUND_P,
            )
            for h in STV_HORIZONS
        },
        "moves_left": round(
            float(decode_moves_left(out["moves_left"][0].reshape(1, -1).float()).item()), 3
        ),
        "policy": policy_rows,
        "opp_policy": _sparse_policy(out["opp_policy"][0], support, policy_floor),
        "soft_policy": _sparse_policy(out["soft_policy"][0], support, policy_floor),
        "top_k": policy_rows[:TOP_K],
    }
    if attention_node is not None:
        heads = int(model.attn_blocks[0].attn.heads)
        payload["attention"] = {
            "query": {
                "q": int(attention_cell[0]), "r": int(attention_cell[1]),
                "node": int(attention_node),
            },
            "blocks": len(attn_mats),
            "heads": heads,
            "floor": ATTN_FLOOR,
            # rows[block][head] = {"tokens": [...], "cells": {node: w}}
            "rows": [
                [
                    _sparse_attention_row(mat[h, NUM_TOKENS + attention_node, :])
                    for h in range(heads)
                ]
                for mat in attn_mats
            ],
        }
    if want_activations:
        payload["activations"] = {"blocks": _activation_blocks(model, activations)}
    if want_features:
        payload["features"] = {
            "names": list(FEATURE_NAMES),
            # planes[f][node], node order = support.coords order.
            "planes": [
                [round(float(v), ROUND_P) for v in feats[:, f].tolist()]
                for f in range(NUM_FEATURES)
            ],
        }
    # Non-finite floats (readouts, value_dist, norms, attention weights) ->
    # JSON null; a raw NaN would 500 at the response encoder.
    return sanitize_json(payload)


# ---------------------------------------------------------------------------
# Lab search
# ---------------------------------------------------------------------------


def search_payload(
    session: Any, evaluator: Any, profile: Any, state: Any, *,
    game_key: int, visits: int, seed: int, decode_action: Any = unpack_action_id,
) -> dict[str, Any]:
    """One as-trained-profile search at the capped budget, greedy selection.

    Same throwaway-tree discipline as ``analysis.searched_eval`` (the key is
    discarded in a ``finally``); additionally reports the raw per-move wire
    weight (``w``) alongside the normalized distribution (``p``). For PUCT
    profiles the weights are visit counts; for Gumbel profiles they carry the
    improved root policy the search produces, so they are floats, not counts.
    """
    try:
        result = profile.search_one(
            session, evaluator, state,
            game_key=game_key, visits=visits, seed=seed, temperature=0.0,
        )
    finally:
        session.discard(game_key)
    ids = np.frombuffer(result["visit_policy_action_ids_bytes"], dtype=np.uint32)
    weights = np.frombuffer(result["visit_policy_weights_bytes"], dtype=np.float32)
    total = float(weights.sum()) or 1.0
    visit_policy = [
        {
            "q": q, "r": r,
            "p": round(float(w) / total, ROUND_P),
            "w": round(float(w), 4),
        }
        for (q, r), w in (
            (decode_action(int(aid)), w) for aid, w in zip(ids.tolist(), weights.tolist())
        )
    ]
    visit_policy.sort(key=lambda row: (-row["p"], row["q"], row["r"]))
    best_q, best_r = decode_action(int(result["action_id"]))
    # NaN/Inf -> null, per the eval_payload contract note.
    return sanitize_json({
        "visits": int(result["visits"]),
        "root_value": round(float(result["root_value"]), ROUND_P),
        "best": {"q": best_q, "r": best_r},
        "visit_policy": visit_policy,
    })
