"""Position analysis: bare-net policy/value readout and a small searched eval.

Runs inside bot worker processes only (imports torch/hexfield at module level;
the web process never imports this module). The net-only path mirrors the
serve featurization exactly: engine state -> PositionFacts -> Support ->
features -> one forward. Policy logits are positional over the support's
legal prefix, so the sparse payload maps slot i to `legal_coords()[i]`.

Note the support radius (HEXFIELD_SUPPORT_RADIUS, 4 for main_7) bounds the
net's legal prefix; cells the engine allows beyond that radius are invisible
to the net by construction — the payload's policy covers what the net sees,
which is also everything the search can expand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from hexfield.batching import collate_rows
from hexfield.engine_facts import facts_from_state
from hexfield.features import build_features
from hexfield.geometry import unpack_action_id
from hexfield.losses import decode_binned_value
from hexfield.support import build_support

TOP_K = 5


def _forward_cpu(model: Any, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Full-head forward with the fp32 relative-position-bias path.

    `build_attn_bias` gathers the bias table in fp16 under no-grad (the GPU
    serve path); running under `enable_grad` takes the fp32 master path
    instead, which is unconditionally safe on CPU. Negligible cost for a
    single position."""
    with torch.enable_grad():
        out = model.forward(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    return {k: v.detach() for k, v in out.items()}


def net_eval(model: Any, state: Any, *, policy_floor: float) -> dict[str, Any]:
    """Bare-net readout for one decision state (no search).

    Returns value (side-to-move POV, [-1, 1]), the sparse legal-cell policy
    (cells with probability >= `policy_floor`, descending), and the top-k
    candidates. The policy is a softmax over the full legal prefix, so the
    dense distribution sums to 1; the floor only trims the reported tail.
    """
    facts = facts_from_state(state)
    support = build_support(facts.stones())
    features = build_features(facts, support)
    batch = collate_rows([(support, features)])
    out = _forward_cpu(model, batch)

    legal_count = support.legal_count
    value = float(decode_binned_value(out["value"][0].reshape(1, -1).float()).item())
    rows: list[dict[str, Any]] = []
    if legal_count > 0:
        priors = torch.softmax(out["policy"][0][:legal_count].float(), dim=0)
        coords = support.legal_coords()
        rows = [
            {"q": int(q), "r": int(r), "p": round(float(p), 6)}
            for (q, r), p in zip(coords.tolist(), priors.tolist())
        ]
        rows.sort(key=lambda row: row["p"], reverse=True)
    return {
        "value": value,
        "legal_count": int(legal_count),
        "policy": [row for row in rows if row["p"] >= policy_floor],
        "top_k": rows[:TOP_K],
    }


def searched_eval(
    session: Any, evaluator: Any, profile: Any, state: Any, *,
    game_key: int, visits: int, seed: int,
) -> dict[str, Any]:
    """Small searched eval (analysis `?search=1`): one as-trained-profile
    search at a capped visit budget, greedy selection. The tree is discarded
    immediately — analysis keys are throwaway, never a live game's."""
    try:
        result = profile.search_one(
            session, evaluator, state,
            game_key=game_key, visits=visits, seed=seed, temperature=0.0,
        )
    finally:
        session.discard(game_key)
    # Wire buffers are native-endian raw u32 ids / f32 weights.
    ids = np.frombuffer(result["visit_policy_action_ids_bytes"], dtype=np.uint32)
    weights = np.frombuffer(result["visit_policy_weights_bytes"], dtype=np.float32)
    total = float(weights.sum()) or 1.0
    visit_policy = [
        {"q": q, "r": r, "p": round(float(w) / total, 6)}
        for (q, r), w in (
            (unpack_action_id(int(aid)), w) for aid, w in zip(ids.tolist(), weights.tolist())
        )
    ]
    visit_policy.sort(key=lambda row: row["p"], reverse=True)
    best_q, best_r = unpack_action_id(int(result["action_id"]))
    return {
        "visits": int(result["visits"]),
        "root_value": round(float(result["root_value"]), 6),
        "best": {"q": best_q, "r": best_r},
        "visit_policy": visit_policy,
    }
