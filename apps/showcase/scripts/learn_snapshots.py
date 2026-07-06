#!/usr/bin/env python3
"""Precomputed snapshot generator for the showcase Learn section.

Bakes real shrimp_main_7 model data into static JSON consumed by the learn
pages (apps/showcase/web/learn/):

  data/attention.json    — real attention rows (softmax(QK^T/sqrt(d) + bias))
                           from the ep58 net, for 4 curated positions x
                           5 attention blocks x 3 heads x ~6 query cells.
  data/checkpoints.json  — net-only policy / value / stv(2) / moves_left for
                           the same positions across 4 training checkpoints
                           (ep2/ep14/ep30/ep58), plus a policy-entropy
                           "sharpening" summary.
  data/eval_history.json — the run's real multistage-eval history: pooled
                           Bradley–Terry ratings (SealBot-anchored), per-epoch
                           match edges, verdicts, sample counts.
  data/features.json     — featurizer ground truth for the network page's
                           feature inspector: 4 positions (quiet / single
                           threat / double threat / win-now) with a per-plane
                           digest {nonzero count, sum, max} of the real
                           (N, 15) feature matrix, so the in-page JS mirror
                           can assert itself against the Python featurizer.
  data/bias_kernels.json — the ep70 net's learned relative-position bias
                           tables, all 5 blocks x 237 rows x 3 heads.
  data/search_compare.json — one real 512-visit search per era on the same
                           position: shrimp_main_7 profile (Gumbel root + SH,
                           ep70) vs shrimp_main_5 profile (PUCT, ep105), via
                           the showcase SearchProfile invocation. Per-move
                           visit counts / Q, root value, chosen move, the
                           gumbel side's candidate logits + priors, and the
                           budget-calibrated SH schedule (verified against
                           the measured visit histogram).
  data/symmetry.json     — the ep70 net's policy + value on the four_threat
                           position (base orientation only; the D6 figure
                           coordinate-transforms it client-side).

Reproducible CLI (run from the repo root, inside the public WSL venv):

  cd /mnt/e/hexo-bot
  PYTHONPATH=packages/shrimp/python:packages/hexo_engine/python \
  SHRIMP_CHANNELS=192 SHRIMP_ATTENTION_HEADS=3 \
  SHRIMP_TRUNK=CCACCACCACCACCA SHRIMP_SUPPORT_RADIUS=4 \
  /root/.venvs/hexo-bot-public/bin/python apps/showcase/scripts/learn_snapshots.py \
      --models-dir /mnt/e/hexo-bot-deploy/models \
      --diagnostics-dir <run_dir>/diagnostics \
      --out apps/showcase/web/learn/data

where <run_dir> is the shrimp_main_7 training run directory (read-only; the
script only parses the *multistage_eval.epoch_*.json diagnostics — the file
prefix before ".multistage_eval" varies with the training-tool version and is
ignored).
The arch env vars above are the published main_7 recipe; the script sets them
itself when unset and refuses to run under a conflicting arch. Output is
deterministic: two runs on the same inputs and --date produce byte-identical
files (CPU forwards, single torch thread, fixed positions, no RNG).

Imports are restricted to stdlib + numpy/torch + shrimp + hexo_engine +
showcase.bots.SearchProfile (the exact as-served search invocation; never
hexo_frontend / hexo_train). The output JSONs contain no filesystem
paths or machine-specific strings; a built-in scrub gate fails the run if a
forbidden substring ever leaks into an output file.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys

# --- published main_7 arch: pin the env BEFORE any shrimp import -------------
_ARCH_ENV = {
    "SHRIMP_CHANNELS": "192",
    "SHRIMP_ATTENTION_HEADS": "3",
    "SHRIMP_TRUNK": "CCACCACCACCACCA",
    "SHRIMP_SUPPORT_RADIUS": "4",
}
for _k, _v in _ARCH_ENV.items():
    os.environ.setdefault(_k, _v)
for _k, _v in _ARCH_ENV.items():
    if os.environ[_k] != _v:
        sys.exit(
            f"{_k}={os.environ[_k]!r} conflicts with the published main_7 "
            f"arch ({_v!r}); unset it or use the documented env."
        )

import numpy as np
import torch

import hexo_engine as engine
from hexo_engine.types import AxialCoord, PlacementAction

from shrimp.batching import collate_rows
from shrimp.constants import BIAS_ROWS, NUM_FEATURES, NUM_TOKENS
from shrimp.engine_facts import facts_from_state
from shrimp.features import build_features
from shrimp.geometry import disk_offsets, hex_dist, on_win_axis, unpack_action_id
from shrimp.losses import decode_binned_value, decode_moves_left
from shrimp.model import ShrimpNet, infer_net_kwargs_from_state_dict
from shrimp.support import build_support

# The showcase server package (for SearchProfile — the as-served invocation).
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server")
    ),
)
from showcase.bots import SearchProfile  # noqa: E402  (needs the path insert)

RUN_LABEL = "shrimp_main_7"
CHECKPOINT_EPOCHS = (2, 14, 30, 58)
ATTENTION_EPOCH = 58
LATEST_EPOCH = 70     # bias kernels / search compare / symmetry checkpoint
PUCT_RUN = "shrimp_main_5"
PUCT_EPOCH = 105
SEARCH_BUDGET = 512   # per-side visit budget for search_compare.json
SEARCH_SEED = 20260705
GUMBEL_MIN_ROUND0_VISITS = 4  # tree.rs GUMBEL_MIN_ROUND0_VISITS (SH calibration)
POLICY_FLOOR = 1e-3   # sparse-policy floor (checkpoints.json, symmetry.json)
ATTN_FLOOR = 1e-3     # sparse-attention floor (attention.json)
ROUND = 4             # decimal places for attention weights
BIAS_DP = 3           # decimal places for bias-table entries
STV_HEAD = "stvalue_2"

# Substrings that must never appear in any output file (case-insensitive).
FORBIDDEN = ("/mnt", "e:", "hexo-bottrainer", "/root", "c:\\", "epicm")


# ==============================================================================
# Curated positions
# ==============================================================================
# Each position is a chronological placement list; the engine assigns owners
# via the fixed Hexo turn structure (P0 places 1 opening stone, then each side
# places 2 per turn). All four sequences are legal, deterministic, and end on
# a non-terminal decision state. double_threat is a strict prefix of
# late_game (the same game, continued).

_QUIET = [
    (0, 0),                # P0 opening
    (1, 1), (0, 2),        # P1
    (2, -1), (1, -2),      # P0
    (-1, 2), (3, 0),       # P1
    (-1, 1), (-2, 0),      # P0
    (2, 2), (-2, 3),       # P1
    (0, -1), (3, -2),      # P0
    (1, 3), (4, -1),       # P1
]

# P1 finishes a split four on the r=1 axis: stones (1,1),(2,1),(3,1),(5,1)
# with the gap at (4,1). Both live 6-windows through the four run over (4,1),
# so P0 (to move) has a single must-play blocking cell.
_FOUR_THREAT = [
    (0, 0),                # P0 opening
    (1, 1), (2, 1),        # P1 starts the line
    (1, 0), (2, 0),        # P0 leans on it from below
    (3, 1), (-1, 4),       # P1 extends; second stone up north
    (0, -2), (2, -2),      # P0 builds own shape
    (0, 4), (1, 3),        # P1 grows the northern group
    (-2, 1), (-1, -1),     # P0
    (5, 1), (2, 3),        # P1 completes the split four (gap (4,1))
]

# Two simultaneous P1 split fours in different directions: the r=1 line
# (gap (4,1)) and the q=-3 column (gap (-3,3)). P0's two placements this
# turn are both forced blocks.
_DOUBLE_THREAT = [
    (0, 0),                # P0 opening
    (1, 1), (2, 1),        # P1 line east
    (1, 0), (2, -1),       # P0
    (-3, 0), (-3, 1),      # P1 opens the western column
    (-1, 0), (0, -2),      # P0
    (3, 1), (-3, 2),       # P1 grows both groups
    (-1, -1), (3, -2),     # P0
    (5, 1), (-3, 4),       # P1 completes both split fours
]

# The same game, continued: P0 blocks both gaps, then a scrappy middlegame
# with several fours forced and answered on both sides. 40 stones; P0 is
# mid-turn (first stone played at (5,0), second pending).
_LATE_GAME = _DOUBLE_THREAT + [
    (4, 1), (-3, 3),       # P0 blocks both gaps
    (4, 2), (3, 3),        # P1 pivots to the (1,-1) diagonal off (5,1)
    (2, 4), (2, 0),        # P0 caps the diagonal, builds the r=0 row
    (3, 0), (2, 2),        # P1 blocks the row four, reinforces the centre
    (2, -2), (3, 2),       # P0 climbs the q=2 column, splits P1's r=2 pair
    (0, 1), (1, 3),        # P1 rebuilds the r=1 four, adds a diagonal trio
    (-1, 1), (4, 0),       # P0 blocks the four, caps the trio
    (0, 4), (5, 3),        # P1 makes a diagonal four (needs (-2,6)/(-1,5))
    (-1, 5), (0, -1),      # P0 blocks it, extends the q=0 column
    (0, 2), (4, 4),        # P1 caps the column, stretches south-east
    (0, -3), (1, -3),      # P0 re-forms the q=0 column four below
    (0, -4), (-2, 2),      # P1 blocks it, links the west group inward
    (5, 0),                # P0 first stone of the current turn
]

# Feature-inspector-only position (features.json; not part of POSITIONS so
# attention.json / checkpoints.json are unchanged). Constructed so every
# feature plane the other three positions leave dead is active at once:
# P0 has a standing win (five stones in the r=0 six-window (0..5, 0), gap
# (4, 0)), P1 has the mirror-image standing win on r=3 (gap (4, 3)), and P0
# is mid-turn — the first stone of the turn went to (1, -1), the second is
# pending — so the phase-second and first-stone planes are set too.
_WIN_NOW = [
    (0, 0),                # P0 opening
    (0, 3), (1, 3),        # P1 starts the r=3 line
    (1, 0), (2, 0),        # P0 starts the r=0 line
    (2, 3), (3, 3),        # P1 extends to four
    (3, 0), (5, 0),        # P0: four, then the split five (gap (4,0))
    (5, 3), (-1, 4),       # P1: the mirror split five (gap (4,3))
    (1, -1),               # P0 first stone of the current turn
]

WIN_NOW_SPEC = {
    "id": "win_now",
    "title": "Standing wins - both sides",
    "moves": _WIN_NOW,
    "description": (
        "Both sides hold a standing win-in-1. P0 has five-of-six on the r=0 "
        "row (gap (4,0)), P1 five-of-six on the r=3 row (gap (4,3)). P0 is "
        "mid-turn (first stone at (1,-1), second pending), so the "
        "phase-second and first-stone planes are set as well. Every hot and "
        "win-now plane is active."
    ),
    "expect": {"opp_hot": 4, "own_hot": 4, "opp_win": 1, "own_win": 1},
    "expect_cells": {"own_win": [(4, 0)], "opp_win": [(4, 3)]},
}

POSITIONS = (
    {
        "id": "quiet_midgame",
        "title": "Quiet middlegame",
        "moves": _QUIET,
        "description": (
            "15 stones, no window with more than three stones of one colour: "
            "both sides are still staking out territory. Baseline for what "
            "attention and policy look like with nothing forced."
        ),
        "expect": {"opp_hot": 0, "own_hot": 0, "opp_win": 0, "own_win": 0},
    },
    {
        "id": "four_threat",
        "title": "Live four - block the gap",
        "moves": _FOUR_THREAT,
        "description": (
            "The opponent just completed a split four on the horizontal axis "
            "(stones at q=1,2,3,5 on r=1 with the gap at (4,1)). Every live "
            "six-window through the four passes over the gap, so the side to "
            "move has exactly one saving cell."
        ),
        "expect": {"opp_hot": 3, "own_win": 0, "opp_win": 0},
    },
    {
        "id": "double_threat",
        "title": "Double threat - two forced blocks",
        "moves": _DOUBLE_THREAT,
        "description": (
            "Two simultaneous opponent split fours in different directions: "
            "one on the r=1 row (gap (4,1)), one on the q=-3 column (gap "
            "(-3,3)). Both of the mover's placements this turn are forced."
        ),
        "expect": {"opp_hot": 6, "own_win": 0, "opp_win": 0},
    },
    {
        "id": "late_game",
        "title": "Late middlegame - 40 stones",
        "moves": _LATE_GAME,
        "description": (
            "The double-threat game continued for 25 more placements: fours "
            "were forced and answered on both sides and the board is dense "
            "with dead windows. The mover is mid-turn (first stone at (5,0), "
            "second pending) and one opponent four-window is still live on "
            "the r=2 row (empties (-1,2) and (1,2))."
        ),
        "expect": {"opp_hot": 2, "opp_win": 0, "own_win": 0},
    },
)


def replay(moves: list[tuple[int, int]]):
    """Apply a placement list through the engine; assert every move legal and
    the final state non-terminal. Returns the engine state."""

    state = engine.new_game()
    for i, (q, r) in enumerate(moves):
        action = PlacementAction(AxialCoord(q=q, r=r))
        if not engine.is_legal_action(state, action):
            raise AssertionError(f"illegal placement {i}: ({q}, {r})")
        result = engine.apply_action(state, action)
        if result.terminal:
            raise AssertionError(f"game ended early at placement {i}: ({q}, {r})")
    if engine.terminal(state) is not None:
        raise AssertionError("curated position is terminal")
    return state


def verify_position(spec: dict, facts, support) -> None:
    """Engine-backed sanity gates for one curated position."""

    expect = spec["expect"]
    got = {
        "opp_hot": len(facts.opp_hot),
        "own_hot": len(facts.own_hot),
        "opp_win": len(facts.opp_win),
        "own_win": len(facts.own_win),
    }
    for key, want in expect.items():
        if got[key] != want:
            raise AssertionError(
                f"{spec['id']}: expected {key}={want}, engine says {got[key]} "
                f"({getattr(facts, key)})"
            )
    for key, cells in spec.get("expect_cells", {}).items():
        engine_cells = sorted(getattr(facts, key))
        if engine_cells != sorted(tuple(c) for c in cells):
            raise AssertionError(
                f"{spec['id']}: expected {key} cells {cells}, engine says "
                f"{engine_cells}"
            )
    # The net's legal prefix (support radius 4) must be a subset of the
    # engine's legal set (radius 8).
    engine_legal = {
        (c.q, c.r) for c in engine.legal_actions(replay(spec["moves"])).coords()
    }
    net_legal = {tuple(c) for c in support.legal_coords().tolist()}
    if not net_legal <= engine_legal:
        raise AssertionError(f"{spec['id']}: support legal cells not all engine-legal")


# ==============================================================================
# Model loading + forwards
# ==============================================================================


def load_net(path: str) -> ShrimpNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    net = ShrimpNet(**infer_net_kwargs_from_state_dict(state_dict))
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    return net


def forward_with_attention(net: ShrimpNet, batch: dict, capture: bool):
    """Full-head forward; optionally captures per-block attention matrices.

    Hook point: a forward hook on every `attn_blocks[i].attn`
    (shrimp.model.RelPosAttention). AttnBlock.forward calls
    `self.attn(self.ln1(seq), attn_bias)`, so the hook's positional inputs are
    exactly the LN'd joint [tokens; cells] sequence and this block's additive
    bias. The hook recomputes q/k with the module's own projections and takes
    softmax((q k^T) * module.scale + bias) — identical math to the module's
    'materialized' impl (sdpa computes the same distribution internally but
    never exposes it). Run under torch.enable_grad() so build_attn_bias takes
    the fp32 master-table path (same trick as showcase analysis.net_eval);
    everything is detached before use.

    Returns (outputs, [per-block (heads, S, S) float32 numpy], S)."""

    rows: list[np.ndarray] = []
    handles = []
    if capture:
        def make_hook(block_index: int):
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
                rows.append(attn[0].numpy())  # (heads, S, S); batch is 1
            return hook

        for i, block in enumerate(net.attn_blocks):
            handles.append(block.attn.register_forward_hook(make_hook(i)))
    try:
        with torch.enable_grad():  # fp32 bias-table path (no-grad casts fp16)
            out = net.forward(
                batch["feats"], batch["nbr"], batch["mask"], batch["coords"]
            )
        out = {k: v.detach() for k, v in out.items()}
    finally:
        for handle in handles:
            handle.remove()
    return out, rows, int(batch["feats"].shape[1])


def net_readout(out: dict, legal_count: int, legal_coords) -> dict:
    """value / stv2 / moves_left / sparse policy / entropy for one row."""

    value = float(decode_binned_value(out["value"][0].reshape(1, -1).float()).item())
    stv = float(decode_binned_value(out[STV_HEAD][0].reshape(1, -1).float()).item())
    ml = float(decode_moves_left(out["moves_left"][0].reshape(1, -1).float()).item())
    priors = torch.softmax(out["policy"][0][:legal_count].float(), dim=0)
    p = priors.numpy().astype(np.float64)
    entropy = float(-(p * np.log(np.clip(p, 1e-30, None))).sum())
    rows = [
        {"q": int(q), "r": int(r), "p": round(float(w), 6)}
        for (q, r), w in zip(legal_coords.tolist(), p.tolist())
    ]
    rows.sort(key=lambda row: (-row["p"], row["q"], row["r"]))
    return {
        "value": round(value, 6),
        "stv2": round(stv, 6),
        "moves_left": round(ml, 3),
        "entropy_nats": round(entropy, 4),
        "top1_p": rows[0]["p"] if rows else 0.0,
        "policy": [row for row in rows if row["p"] >= POLICY_FLOOR],
    }


# ==============================================================================
# Attention row extraction
# ==============================================================================


def pick_queries(support, facts, policy_row: dict) -> list[dict]:
    """~6 deterministic, tactically interesting query cells for one position.

    Roles: last_stone, opening_stone, top_policy (net's best cell),
    threat_cell (first opponent hot cell, when one exists), far_legal (legal
    cell furthest from any stone), halo (first halo-shell cell). Duplicates
    are dropped and backfilled with the next-best policy cells."""

    idx = support.index
    by_policy = [(row["q"], row["r"]) for row in policy_row["policy"]]
    legal, stones, halo = support.segments()

    wanted: list[tuple[str, tuple[int, int]]] = []
    last = facts.records[-1]
    wanted.append(("last_stone", (last[0], last[1])))
    wanted.append(("opening_stone", (facts.records[0][0], facts.records[0][1])))
    if by_policy:
        wanted.append(("top_policy", by_policy[0]))
    if facts.opp_hot:
        wanted.append(("threat_cell", facts.opp_hot[0]))
    if len(legal) > 0:
        dists = support.dist[: support.legal_count]
        far_row = int(np.argmax(dists))
        far = tuple(support.coords[far_row].tolist())
        wanted.append(("far_legal", far))
    if len(halo) > 0:
        wanted.append(("halo", tuple(support.coords[halo[0]].tolist())))

    queries: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for role, cell in wanted:
        if cell in seen:
            continue
        seen.add(cell)
        queries.append({"role": role, "cell": list(cell), "node": idx[cell]})
    for cell in by_policy:  # backfill to 6 with next-best policy cells
        if len(queries) >= 6:
            break
        if cell in seen:
            continue
        seen.add(cell)
        queries.append({"role": "policy_candidate", "cell": list(cell), "node": idx[cell]})
    return queries[:6]


def sparse_attention_row(row: np.ndarray) -> dict:
    """One (S,) softmax row -> {'tokens': [8 floats], 'cells': {node: w}}.

    Asserts the pre-pruning row sums to 1, then prunes cell weights below
    ATTN_FLOOR and quantizes everything to ROUND decimals."""

    total = float(row.sum())
    if not math.isclose(total, 1.0, abs_tol=1e-4):
        raise AssertionError(f"attention row sums to {total}, expected 1")
    tokens = [round(float(w), ROUND) + 0.0 for w in row[:NUM_TOKENS]]
    cells = {
        str(i): round(float(w), ROUND) + 0.0
        for i, w in enumerate(row[NUM_TOKENS:])
        if w >= ATTN_FLOOR
    }
    return {"tokens": tokens, "cells": cells}


# ==============================================================================
# Feature-inspector ground truth (features.json)
# ==============================================================================


def feature_digest(feats: np.ndarray) -> dict:
    """Per-plane {n, sum, max} digest of one (N, 15) feature matrix.

    The digest is the contract the network page's JS featurizer mirror asserts
    itself against: matching nonzero count + sum (4 dp) + max per plane over
    the whole support is a strong full-matrix check in a few bytes."""

    digest = {}
    for i in range(NUM_FEATURES):
        col = feats[:, i].astype(np.float64)
        digest[str(i)] = {
            "n": int(np.count_nonzero(col)),
            "sum": round(float(col.sum()), 4),
            "max": round(float(col.max()) if len(col) else 0.0, 4),
        }
    return digest


# ==============================================================================
# Rel-pos bias kernels (bias_kernels.json)
# ==============================================================================


def bias_kernels_doc(net: ShrimpNet, generated: str) -> dict:
    """Every attention block's learned (237, heads) bias table, plus an
    in-script re-verification of the two observations the network page's
    caption cites (fails the run if a future checkpoint no longer shows them)."""

    offs = disk_offsets(8)
    tables = []
    for block in range(len(net.attn_blocks)):
        t = net.bias_tables[block].detach().cpu().numpy()  # (237, heads)
        if t.shape != (BIAS_ROWS, net.attn_blocks[0].attn.heads):
            raise AssertionError(f"unexpected bias table shape {t.shape}")
        tables.append(
            [[round(float(v), BIAS_DP) + 0.0 for v in t[:, h]] for h in range(t.shape[1])]
        )

    def disk_stats(block: int, head: int) -> dict:
        col = net.bias_tables[block].detach().cpu().numpy()[:217, head]
        by = lambda pred: float(np.mean([col[i] for i, o in enumerate(offs) if pred(o)]))
        return {
            "center": float(col[offs.index((0, 0))]),
            "ring1": by(lambda o: hex_dist(*o) == 1),
            "d4_on": by(lambda o: hex_dist(*o) == 4 and on_win_axis(*o)),
            "d4_off": by(lambda o: hex_dist(*o) == 4 and not on_win_axis(*o)),
            "d6_on": by(lambda o: hex_dist(*o) == 6 and on_win_axis(*o)),
            "d6_off": by(lambda o: hex_dist(*o) == 6 and not on_win_axis(*o)),
        }

    # Caption claims (network.html, bias-kernel figure). block 4 head 1: the
    # query cell and its whole 1-ring are strongly suppressed while the
    # distance-6 on-axis offsets (a winning window's length) get the table's
    # largest positive bump, several times the off-axis mean at the same
    # distance. block 4 head 2: the complement — near-and-on-axis positive
    # (distance-4 on-axis mean well above off-axis).
    s41 = disk_stats(4, 1)
    if not (s41["center"] < -3.0 and s41["ring1"] < -2.0):
        raise AssertionError(f"caption claim broke: b4h1 local suppression {s41}")
    if not (s41["d6_on"] > 1.5 and s41["d6_on"] > 3.0 * s41["d6_off"]):
        raise AssertionError(f"caption claim broke: b4h1 axis-6 bump {s41}")
    s42 = disk_stats(4, 2)
    if not (s42["d4_on"] > 0.6 and s42["d4_on"] > 2.0 * s42["d4_off"]):
        raise AssertionError(f"caption claim broke: b4h2 near-axis bump {s42}")
    print(
        "bias caption checks: b4h1 center %.2f ring1 %.2f d6 on/off %.2f/%.2f · "
        "b4h2 d4 on/off %.2f/%.2f"
        % (s41["center"], s41["ring1"], s41["d6_on"], s41["d6_off"],
           s42["d4_on"], s42["d4_off"])
    )

    return {
        "run": RUN_LABEL,
        "checkpoint": f"ep{LATEST_EPOCH}",
        "generated": generated,
        "blocks": len(net.attn_blocks),
        "heads": net.attn_blocks[0].attn.heads,
        "rows": BIAS_ROWS,
        "disk_radius": 8,
        "ring_min": 9,
        "ring_max": 16,
        "offset_order": "rows 0-216 are the hex-dist<=8 offsets ascending (dq, dr)",
        "row_layout": {
            "disk": [0, 216],
            "ring_on_axis": [217, 224],
            "ring_off_axis": [225, 232],
            "far": 233,
            "cell_token": 234,
            "token_cell": 235,
            "token_token": 236,
        },
        # tables[block][head] = 237 floats (raw additive score units, pre-softmax)
        "tables": tables,
    }


# ==============================================================================
# Gumbel vs PUCT search compare + SH schedule (search_compare.json)
# ==============================================================================


def _sh_rounds_for(m: int) -> int:
    """ceil(log2 m) — tree.rs GumbelRootState::rounds_for."""

    return 0 if m <= 1 else max(1, (m - 1).bit_length())


def _sh_budget_calibrated_m(m: int, budget: int) -> int:
    """tree.rs GumbelRootState::budget_calibrated_m."""

    while m > 2:
        rounds = max(_sh_rounds_for(m), 1)
        if budget // (rounds * m) >= GUMBEL_MIN_ROUND0_VISITS:
            break
        m = (m + 1) // 2
    return m


def _sh_schedule(m_configured: int, legal: int, budget: int) -> dict:
    """The exact SH schedule the engine runs at this budget: calibrated m,
    round count, per-round quotas and cumulative caps (tree.rs
    seed_gumbel_round_caps: quota_r = max(1, floor(budget / (R * |A_r|))))."""

    m = min(max(_sh_budget_calibrated_m(min(m_configured, legal), budget), 1), legal)
    rounds = _sh_rounds_for(m)
    survivors, quotas, cum = [], [], []
    alive, total = m, 0
    for _ in range(rounds):
        per = max(budget // (rounds * alive), 1)
        survivors.append(alive)
        quotas.append(per)
        cum.append((cum[-1] if cum else 0) + per)
        total += alive * per
        alive = (alive + 1) // 2
    return {
        "m_configured": m_configured,
        "m": m,
        "rounds": rounds,
        "min_round0_visits": GUMBEL_MIN_ROUND0_VISITS,
        "survivors_per_round": survivors,
        "round_quota": quotas,
        "cum_cap": cum,
        "scheduled_total": total,
        "leftover": max(budget - total, 0),
    }


def _expected_visit_multiset(sched: dict) -> list[int]:
    """Final per-candidate visit counts implied by the schedule: candidates
    eliminated after round r hold cum_cap[r]; the last survivor holds
    cum_cap[-1] + leftover."""

    counts: list[int] = []
    survivors = sched["survivors_per_round"] + [1]
    for r in range(sched["rounds"]):
        counts.extend([sched["cum_cap"][r]] * (survivors[r] - survivors[r + 1]))
    counts.append(sched["cum_cap"][-1] + sched["leftover"])
    return sorted(counts)


def _f32(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32)


def _u32(buf: bytes) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.uint32)


def run_search_side(
    profile_path: str, net: ShrimpNet, state, *, game_key: int
) -> dict:
    """One as-served single-root search (showcase SearchProfile invocation,
    fixed seed, temperature 0) -> the measured per-move export."""

    from shrimp import _rust
    from shrimp.inference import ShrimpEvaluator

    profile = SearchProfile(profile_path)
    session = _rust.ShrimpMctsSession(max_states=65_536)
    try:
        result = profile.search_one(
            session, ShrimpEvaluator(net, device="cpu"), state,
            game_key=game_key, visits=SEARCH_BUDGET, seed=SEARCH_SEED,
            temperature=0.0,
        )
    finally:
        session.discard(game_key)

    visits = int(result["visits"])
    ids = _u32(result["visit_policy_action_ids_bytes"])
    weights = _f32(result["visit_policy_weights_bytes"])
    qs = _f32(result["visit_policy_q_bytes"])
    counts = np.rint(weights.astype(np.float64) * visits).astype(int)
    if int(counts.sum()) != visits:
        raise AssertionError(
            f"visit histogram does not reconstruct: {counts.sum()} != {visits}"
        )
    prior = {
        int(a): float(w)
        for a, w in zip(
            _u32(result["root_prior_policy_action_ids_bytes"]).tolist(),
            _f32(result["root_prior_policy_weights_bytes"]).tolist(),
        )
    }
    moves = []
    for aid, n, q in zip(ids.tolist(), counts.tolist(), qs.tolist()):
        mq, mr = unpack_action_id(int(aid))
        moves.append(
            {
                "q": mq, "r": mr, "n": int(n),
                "qhat": round(float(q), 4),
                "prior": round(prior.get(int(aid), 0.0), 4),
            }
        )
    moves.sort(key=lambda m: (-m["n"], m["q"], m["r"]))
    cq, cr = unpack_action_id(int(result["action_id"]))
    side = {
        "chosen": [cq, cr],
        "action_selection": str(result["action_selection"]),
        "early_stopped": bool(result["early_stopped"]),
        "root_value": round(float(result["root_value"]), 6),
        "visits": visits,
        "budget": SEARCH_BUDGET,
        "moves_visited": len(moves),
        "moves": moves,
    }
    if "gumbel_policy_count" in result:
        logit_ids = _u32(result["gumbel_policy_action_ids_bytes"]).tolist()
        logits = _f32(result["root_prior_logits_bytes"]).tolist()
        logit_map = dict(zip(logit_ids, logits))
        by_cell = {(m["q"], m["r"]): m for m in moves}
        for aid, logit in logit_map.items():
            cell = unpack_action_id(int(aid))
            if cell in by_cell:
                by_cell[cell]["logit"] = round(float(logit), 4)
    return side


def search_compare_doc(
    spec: dict, facts, support, gumbel_net: ShrimpNet, puct_net: ShrimpNet,
    position_block, generated: str,
) -> dict:
    state = replay(spec["moves"])
    if facts.opp_hot or facts.own_hot or facts.opp_win or facts.own_win:
        raise AssertionError(
            "search-compare position must be threat-free (TSS force-includes "
            "tactical cells beyond the m budget, which would distort the ladder)"
        )
    profiles_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "profiles")
    )
    gumbel = run_search_side(
        os.path.join(profiles_dir, "shrimp_main_7.toml"), gumbel_net, state,
        game_key=1,
    )
    gumbel.update(
        id="gumbel", label="Gumbel root + sequential halving",
        run=RUN_LABEL, checkpoint=f"ep{LATEST_EPOCH}", profile="shrimp_main_7",
    )
    puct = run_search_side(
        os.path.join(profiles_dir, "shrimp_main_5.toml"), puct_net, state,
        game_key=2,
    )
    puct.update(
        id="puct", label="PUCT",
        run=PUCT_RUN, checkpoint=f"ep{PUCT_EPOCH}", profile="shrimp_main_5",
    )

    # SH schedule (reconstructed from the algorithm) must explain the measured
    # gumbel visit histogram exactly — this is what lets the search page's
    # ladder animation replay the rounds from final counts alone.
    sched = _sh_schedule(32, int(support.legal_count), SEARCH_BUDGET)
    measured = sorted(m["n"] for m in gumbel["moves"])
    expected = _expected_visit_multiset(sched)
    if measured != expected:
        raise AssertionError(
            f"measured gumbel visit histogram {measured} does not match the "
            f"SH schedule {expected}; regen needs a caption re-check"
        )
    if any("logit" not in m for m in gumbel["moves"]):
        raise AssertionError("gumbel candidate without an exported logit")
    gumbel["sh"] = sched

    pos = position_block(spec, facts, support)
    pos["support"] = {
        "coords": support.coords.tolist(),
        "legal_count": int(support.legal_count),
        "stone_count": int(support.stone_count),
        "halo_count": int(support.halo_count),
    }
    return {
        "generated": generated,
        "budget": SEARCH_BUDGET,
        "seed": SEARCH_SEED,
        "invocation": (
            "showcase SearchProfile.search_one, temperature 0, CPU, one "
            "single-root search per side; profiles as served by the site"
        ),
        "position": pos,
        "sides": [gumbel, puct],
    }


# ==============================================================================
# D6 symmetry demo (symmetry.json)
# ==============================================================================


def symmetry_doc(
    spec: dict, facts, support, readout: dict, position_block, generated: str
) -> dict:
    """Base-orientation policy/value for the D6 figure. Only orientation 0 is
    evaluated; the training page transforms coordinates client-side for the
    other 11 (the same re-expression training applies to sampled rows)."""

    if not readout["policy"]:
        raise AssertionError("symmetry position has an empty floored policy")
    top = readout["policy"][0]
    pos = position_block(spec, facts, support)
    pos["support"] = {
        "coords": support.coords.tolist(),
        "legal_count": int(support.legal_count),
        "stone_count": int(support.stone_count),
        "halo_count": int(support.halo_count),
    }
    return {
        "run": RUN_LABEL,
        "checkpoint": f"ep{LATEST_EPOCH}",
        "generated": generated,
        "policy_floor": POLICY_FLOOR,
        "d6": (
            "indices 0-5 rotate by 60*i degrees; 6-11 reflect (q,r)->(q,-q-r) "
            "then rotate — geometry.py apply_d6"
        ),
        "position": pos,
        "value": readout["value"],
        "argmax": [top["q"], top["r"]],
        "argmax_p": top["p"],
        "policy": readout["policy"],
    }


# ==============================================================================
# Eval history (real run diagnostics)
# ==============================================================================

_EDGE_FIELDS = (
    "opponent", "role", "kind", "primary", "paired", "games_requested",
    "decided", "winrate", "winrate_ci95", "elo_point",
)
_EDGE_PROVENANCE_FIELDS = (
    "n_pairs", "pentanomial", "pair_winrate", "pair_se", "eval_visits",
)
_RATING_FIELDS = ("label", "elo", "elo_ci95", "se_elo", "is_anchor")


def parse_eval_history(diagnostics_dir: str, generated: str) -> dict:
    """Whitelist-copy the pooled ratings / edges / verdicts out of every
    *multistage_eval.epoch_*.json (the prefix before ".multistage_eval" varies
    with the training-tool version and is ignored). Only labels and numbers
    are copied; paths never leave the run directory."""

    import glob as _glob

    files = sorted(
        _glob.glob(os.path.join(diagnostics_dir, "*multistage_eval.epoch_*.json"))
    )
    if not files:
        raise SystemExit(f"no multistage eval files under --diagnostics-dir")

    epochs = []
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        meta = doc.get("meta", {})
        cfg = meta.get("config", {})
        roster = doc.get("roster", {})
        verdict = doc.get("verdict", {})
        primary = verdict.get("primary", {}) or {}
        ratings = doc.get("ratings", {})
        players = [
            {k: p.get(k) for k in _RATING_FIELDS} for p in ratings.get("players", [])
        ]
        cand_label = meta.get("candidate_label")
        cand = next((p for p in players if p["label"] == cand_label), None)
        edges = []
        for e in doc.get("edges", []):
            row = {k: e.get(k) for k in _EDGE_FIELDS if k in e}
            prov = e.get("provenance", {}) or {}
            for k in _EDGE_PROVENANCE_FIELDS:
                if k in prov:
                    row[k] = prov[k]
            edges.append(row)
        epochs.append(
            {
                "epoch": int(meta.get("candidate_epoch")),
                "candidate": cand_label,
                "champion": (roster.get("champion") or {}).get("label"),
                "verdict": verdict.get("label"),
                "primary": {
                    k: primary.get(k)
                    for k in ("elo_diff", "elo_diff_ci95", "se_elo", "hypothesis")
                    if k in primary
                },
                "candidate_elo": (cand or {}).get("elo"),
                "candidate_elo_ci95": (cand or {}).get("elo_ci95"),
                "candidate_se_elo": (cand or {}).get("se_elo"),
                "games_budget": cfg.get("games_budget"),
                "eval_visits": cfg.get("eval_visits"),
                "edges": edges,
                "ratings": players,
            }
        )
    epochs.sort(key=lambda row: row["epoch"])
    fit = json.load(open(files[-1], "r", encoding="utf-8")).get("ratings", {}).get("fit", {})
    return {
        "run": RUN_LABEL,
        "generated": generated,
        "anchor": "sealbot",
        "notes": {
            "scale": (
                "Pooled Bradley-Terry ratings over every accumulated match edge; "
                "SealBot (a fixed scripted opponent) is pinned at 0 Elo as the "
                "zero-point and its edges are down-weighted out of difference "
                "inference."
            ),
            "candidates": (
                "cand_epN is that epoch's candidate under a fresh label, so its "
                "rating does not pool across epochs; the fixed labels (ep5, "
                "ep30, ...) do pool and tighten over time — use the latest "
                "epoch's ratings table for the compounding strength curve."
            ),
            "pentanomial": (
                "Paired edges play mirrored openings; pentanomial is the "
                "[0, 0.5, 1, 1.5, 2] points-per-pair histogram over n_pairs."
            ),
        },
        "evaluated_epochs": [row["epoch"] for row in epochs],
        "latest_epoch": epochs[-1]["epoch"],
        "bt_fit": {
            k: fit.get(k)
            for k in ("n_edges", "n_players", "converged", "iterations")
            if k in fit
        },
        "epochs": epochs,
    }


# ==============================================================================
# main
# ==============================================================================


def dump(path: str, doc: dict) -> int:
    text = json.dumps(doc, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    low = text.lower()
    for bad in FORBIDDEN:
        if bad in low:
            raise AssertionError(f"forbidden substring {bad!r} in {os.path.basename(path)}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return len(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", required=True,
                    help="directory holding main7_ep{2,14,30,58}.pt inference exports")
    ap.add_argument("--diagnostics-dir", required=True,
                    help="shrimp_main_7 run diagnostics dir (read-only)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "web", "learn", "data"))
    ap.add_argument("--date", default=_dt.date.today().isoformat(),
                    help="'generated' stamp (default: today; pin for byte-identical reruns)")
    args = ap.parse_args()

    torch.set_num_threads(1)  # bit-reproducible CPU reductions
    torch.manual_seed(0)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # --- positions ------------------------------------------------------------
    prepared = []
    for spec in POSITIONS:
        state = replay(spec["moves"])
        facts = facts_from_state(state)
        support = build_support(facts.stones())
        verify_position(spec, facts, support)
        feats = build_features(facts, support)
        batch = collate_rows([(support, feats)])
        prepared.append((spec, facts, support, batch))
        print(
            f"position {spec['id']}: {len(spec['moves'])} stones, "
            f"support={support.num_nodes} (legal={support.legal_count}, "
            f"halo={support.halo_count}), to_move=P{facts.current_player}, "
            f"phase={facts.phase}, opp_hot={list(facts.opp_hot)}"
        )

    def position_block(spec, facts, support) -> dict:
        return {
            "id": spec["id"],
            "title": spec["title"],
            "description": spec["description"],
            "moves": [list(m) for m in spec["moves"]],
            "stones": [[q, r, owner] for q, r, owner, _ in facts.records],
            "to_move": facts.current_player,
            "phase": facts.phase,
            "first_stone": list(facts.first_stone) if facts.first_stone else None,
            "threats": {
                "opp_hot": [list(c) for c in facts.opp_hot],
                "own_hot": [list(c) for c in facts.own_hot],
                "opp_win": [list(c) for c in facts.opp_win],
                "own_win": [list(c) for c in facts.own_win],
            },
        }

    # --- checkpoints.json -------------------------------------------------------
    nets = {}
    for epoch in CHECKPOINT_EPOCHS:
        path = os.path.join(args.models_dir, f"main7_ep{epoch}.pt")
        nets[epoch] = load_net(path)
        print(f"loaded ep{epoch} from {os.path.basename(path)}")

    ckpt_doc = {
        "run": RUN_LABEL,
        "generated": args.date,
        "checkpoints": [
            {"id": f"ep{e}", "epoch": e, "label": f"epoch {e}"} for e in CHECKPOINT_EPOCHS
        ],
        "stv_head": STV_HEAD,
        "policy_floor": POLICY_FLOOR,
        "positions": [],
        "sharpening": {"metric": "policy entropy (nats) over the legal set",
                       "rows": []},
    }
    readouts: dict[tuple[str, int], dict] = {}
    for spec, facts, support, batch in prepared:
        pos = position_block(spec, facts, support)
        pos["legal_count"] = int(support.legal_count)
        pos["per_checkpoint"] = {}
        for epoch in CHECKPOINT_EPOCHS:
            out, _, _ = forward_with_attention(nets[epoch], batch, capture=False)
            ro = net_readout(out, support.legal_count, support.legal_coords())
            readouts[(spec["id"], epoch)] = ro
            pos["per_checkpoint"][f"ep{epoch}"] = ro
        ckpt_doc["positions"].append(pos)
        ckpt_doc["sharpening"]["rows"].append(
            {
                "position": spec["id"],
                "legal_count": int(support.legal_count),
                "max_entropy_nats": round(math.log(max(1, support.legal_count)), 4),
                "entropy": {
                    f"ep{e}": readouts[(spec["id"], e)]["entropy_nats"]
                    for e in CHECKPOINT_EPOCHS
                },
                "top1_p": {
                    f"ep{e}": readouts[(spec["id"], e)]["top1_p"]
                    for e in CHECKPOINT_EPOCHS
                },
            }
        )
    n = dump(os.path.join(out_dir, "checkpoints.json"), ckpt_doc)
    print(f"checkpoints.json: {n/1024:.1f} KB")

    # --- attention.json ---------------------------------------------------------
    attn_net = nets[ATTENTION_EPOCH]
    attn_doc = {
        "run": RUN_LABEL,
        "checkpoint": f"ep{ATTENTION_EPOCH}",
        "generated": args.date,
        "num_tokens": NUM_TOKENS,
        "blocks": len(attn_net.attn_blocks),
        "heads": attn_net.attn_blocks[0].attn.heads,
        "floor": ATTN_FLOOR,
        "positions": [],
    }
    for spec, facts, support, batch in prepared:
        out, mats, s_total = forward_with_attention(attn_net, batch, capture=True)
        if len(mats) != len(attn_net.attn_blocks):
            raise AssertionError("hook captured wrong number of attention blocks")
        ro = readouts[(spec["id"], ATTENTION_EPOCH)]
        queries = pick_queries(support, facts, ro)
        pos = position_block(spec, facts, support)
        pos["support"] = {
            "coords": support.coords.tolist(),
            "legal_count": int(support.legal_count),
            "stone_count": int(support.stone_count),
            "halo_count": int(support.halo_count),
        }
        pos["queries"] = queries
        # attention[block][head][query_index] = {"tokens": [...], "cells": {...}}
        pos["attention"] = [
            [
                [
                    sparse_attention_row(mats[b][h, NUM_TOKENS + qd["node"], :])
                    for qd in queries
                ]
                for h in range(attn_doc["heads"])
            ]
            for b in range(attn_doc["blocks"])
        ]
        attn_doc["positions"].append(pos)
        print(f"attention {spec['id']}: S={s_total + 0} rows={len(queries)*15}")
    n = dump(os.path.join(out_dir, "attention.json"), attn_doc)
    print(f"attention.json: {n/1024:.1f} KB")
    if n >= 900 * 1024:
        raise AssertionError(f"attention.json too large: {n} bytes (budget 900 KB)")

    # --- features.json ------------------------------------------------------------
    # The three tactical tiers reuse curated POSITIONS; win_now is inspector-only.
    feature_specs = [POSITIONS[0], POSITIONS[1], POSITIONS[2], WIN_NOW_SPEC]
    by_id = {spec["id"]: (spec, facts, support) for spec, facts, support, _ in prepared}
    feat_doc = {
        "run": RUN_LABEL,
        "generated": args.date,
        "num_features": NUM_FEATURES,
        "digest_note": (
            "per-plane {n: nonzero cells, sum, max} over the whole (N, 15) "
            "feature matrix in support node order; the page's JS featurizer "
            "mirror recomputes all planes and must reproduce every digest"
        ),
        "positions": [],
    }
    for spec in feature_specs:
        if spec["id"] in by_id:
            _, facts, support = by_id[spec["id"]]
        else:
            state = replay(spec["moves"])
            facts = facts_from_state(state)
            support = build_support(facts.stones())
            verify_position(spec, facts, support)
            print(
                f"position {spec['id']}: {len(spec['moves'])} stones, "
                f"support={support.num_nodes} (legal={support.legal_count}, "
                f"halo={support.halo_count}), to_move=P{facts.current_player}, "
                f"phase={facts.phase}"
            )
        feats = build_features(facts, support)
        pos = position_block(spec, facts, support)
        pos["support"] = {
            "coords": support.coords.tolist(),
            "legal_count": int(support.legal_count),
            "stone_count": int(support.stone_count),
            "halo_count": int(support.halo_count),
        }
        pos["digest"] = feature_digest(feats)
        feat_doc["positions"].append(pos)
    n = dump(os.path.join(out_dir, "features.json"), feat_doc)
    print(f"features.json: {n/1024:.1f} KB")

    # --- bias_kernels.json ----------------------------------------------------------
    latest_net = load_net(os.path.join(args.models_dir, f"main7_ep{LATEST_EPOCH}.pt"))
    print(f"loaded ep{LATEST_EPOCH} from main7_ep{LATEST_EPOCH}.pt")
    n = dump(os.path.join(out_dir, "bias_kernels.json"),
             bias_kernels_doc(latest_net, args.date))
    print(f"bias_kernels.json: {n/1024:.1f} KB")

    # --- search_compare.json ----------------------------------------------------------
    puct_net = load_net(os.path.join(args.models_dir, f"main5_ep{PUCT_EPOCH}.pt"))
    print(f"loaded {PUCT_RUN} ep{PUCT_EPOCH} from main5_ep{PUCT_EPOCH}.pt")
    quiet_spec, quiet_facts, quiet_support = by_id["quiet_midgame"]
    compare = search_compare_doc(
        quiet_spec, quiet_facts, quiet_support, latest_net, puct_net,
        position_block, args.date,
    )
    for side in compare["sides"]:
        print(
            f"search {side['id']}: chosen {tuple(side['chosen'])} "
            f"value {side['root_value']} visits {side['visits']}/{side['budget']} "
            f"moves {side['moves_visited']}"
            + (f" · SH m={side['sh']['m']} rounds={side['sh']['rounds']}"
               if "sh" in side else "")
        )
    n = dump(os.path.join(out_dir, "search_compare.json"), compare)
    print(f"search_compare.json: {n/1024:.1f} KB")

    # --- symmetry.json ----------------------------------------------------------------
    four_spec, four_facts, four_support = by_id["four_threat"]
    four_batch = next(b for s, _, _, b in prepared if s["id"] == "four_threat")
    out, _, _ = forward_with_attention(latest_net, four_batch, capture=False)
    ro70 = net_readout(out, four_support.legal_count, four_support.legal_coords())
    sym_doc = symmetry_doc(
        four_spec, four_facts, four_support, ro70, position_block, args.date
    )
    print(
        f"symmetry four_threat ep{LATEST_EPOCH}: value {sym_doc['value']} "
        f"argmax {tuple(sym_doc['argmax'])} p {sym_doc['argmax_p']}"
    )
    n = dump(os.path.join(out_dir, "symmetry.json"), sym_doc)
    print(f"symmetry.json: {n/1024:.1f} KB")

    # --- eval_history.json --------------------------------------------------------
    hist = parse_eval_history(args.diagnostics_dir, args.date)
    n = dump(os.path.join(out_dir, "eval_history.json"), hist)
    print(f"eval_history.json: {n/1024:.1f} KB")

    total = sum(
        os.path.getsize(os.path.join(out_dir, f))
        for f in (
            "attention.json", "checkpoints.json", "eval_history.json",
            "features.json", "bias_kernels.json", "search_compare.json",
            "symmetry.json",
        )
    )
    print(f"total data size: {total/1024:.1f} KB")
    if total >= 1536 * 1024:
        raise AssertionError(f"data/ exceeds the 1.5 MB budget: {total} bytes")


if __name__ == "__main__":
    main()
