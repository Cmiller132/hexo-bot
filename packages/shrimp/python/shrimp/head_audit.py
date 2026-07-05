"""Moves-left head audit.

Decodes the model's moves-left prediction on stored self-play shards (each
``game_*.npz`` is one game, rows in ply order) and scores it against the
remaining-decisions target carried by the shard. Reuses the expand/collate
path from ``samples.py`` and ``batching.py``; runs one forward pass per chunk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .batching import PAD_QUANTUM, collate_training
from .constants import MOVES_LEFT_CAP
from .losses import decode_moves_left
from .samples import expand_sample
from .shards import read_compact_shard

# Pass thresholds. conv_spearman is the within-game conversion-zone (true
# remaining < 60) rank correlation; full_spearman is over all kept rows; near-end
# MAE is over the [0,5) true-remaining bucket. Monotonicity rate is reported for
# diagnostics only and is not gated.
CONV_SPEARMAN_GATE = 0.50
FULL_SPEARMAN_GATE = 0.40
NEAR_END_MAE_GATE = 25.0  # [0,5) true-remaining decode MAE (decisions)

_BUCKETS = [(0, 5), (5, 20), (20, 60), (60, 120), (120, 10**9)]
_BUCKET_NAMES = ["[0,5)", "[5,20)", "[20,60)", "[60,120)", "[120+)"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


@torch.no_grad()
def _predict_game(model, expanded, device, row_budget_nodes: int = 40_000):
    """Decoded moves-left (decisions) for one game's expanded rows. Rows are
    processed in chunks sized so ``rows * pad_to^2`` stays within
    ``row_budget_nodes * PAD_QUANTUM``. Returns a 1-D array."""
    preds: list[np.ndarray] = []
    i = 0
    n = len(expanded)
    while i < n:
        # Grow the chunk until rows*pad_to^2 would exceed the budget.
        j = i
        max_nodes = 0
        while j < n:
            nodes = expanded[j].support.num_nodes
            cand = max(max_nodes, nodes)
            pad_to = -(-cand // PAD_QUANTUM) * PAD_QUANTUM
            if j > i and (j - i + 1) * pad_to * pad_to > row_budget_nodes * PAD_QUANTUM:
                break
            max_nodes = cand
            j += 1
        chunk = expanded[i:j]
        pad_to = -(-max(r.support.num_nodes for r in chunk) // PAD_QUANTUM) * PAD_QUANTUM
        batch = collate_training(chunk, pad_to=pad_to)
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
        }
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        preds.append(decode_moves_left(out["moves_left"].float()).cpu().numpy())
        i = j
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.float64)


def audit_moves_left_head(model, shard_paths, *, device, max_games: int = 60) -> dict:
    """Audit the moves-left head over up to ``max_games`` shard-games. Returns
    a dict with the aggregate metrics and a ``passed`` verdict."""
    device = torch.device(device)
    model.eval().to(device)

    full_rho, conv_rho, conv_mono, mono_adj = [], [], [], []
    err_all, true_all, pred_all = [], [], []
    n_games = 0
    for path in list(shard_paths)[:max_games]:
        rows = read_compact_shard(Path(path))
        if not rows:
            continue
        expanded = [expand_sample(s, symmetry=0) for s in rows]
        keep = [(e) for e in expanded if float(e.moves_left_mask) > 0.0]
        if len(keep) < 10:
            continue
        true = np.array(
            [(float(e.moves_left) + 1.0) * 0.5 * MOVES_LEFT_CAP for e in keep], dtype=np.float64
        )
        pred = _predict_game(model, keep, device).astype(np.float64)
        if pred.shape[0] != true.shape[0]:
            continue
        # Order rows by true remaining descending (true decreases with ply).
        order = np.argsort(-true)
        true, pred = true[order], pred[order]
        n_games += 1
        true_all.append(true)
        pred_all.append(pred)
        err_all.append(pred - true)
        full_rho.append(_spearman(pred, true))
        m = true < 60
        if m.sum() >= 10:
            conv_rho.append(_spearman(pred[m], true[m]))
            conv_mono.append(float(np.mean(np.diff(pred[m]) < 0)))
        mono_adj.append(float(np.mean(np.diff(pred) < 0)))

    if n_games == 0:
        return {"passed": False, "reason": "no usable games", "n_games": 0}

    true_cat = np.concatenate(true_all)
    err_cat = np.concatenate(err_all)
    buckets = {}
    for (lo, hi), name in zip(_BUCKETS, _BUCKET_NAMES):
        bm = (true_cat >= lo) & (true_cat < hi)
        buckets[name] = (
            {"n": int(bm.sum()), "mae": round(float(np.abs(err_cat[bm]).mean()), 2)}
            if bm.sum()
            else {"n": 0}
        )

    conv_spearman = float(np.mean(conv_rho)) if conv_rho else float("nan")
    full_spearman = float(np.mean(full_rho)) if full_rho else float("nan")
    conv_mono_rate = float(np.mean(conv_mono)) if conv_mono else float("nan")
    near_end_mae = buckets["[0,5)"].get("mae")
    passed = (
        not np.isnan(conv_spearman)
        and conv_spearman >= CONV_SPEARMAN_GATE
        and not np.isnan(full_spearman)
        and full_spearman >= FULL_SPEARMAN_GATE
        and (near_end_mae is None or near_end_mae <= NEAR_END_MAE_GATE)
    )
    return {
        "passed": bool(passed),
        "n_games": n_games,
        "n_positions": int(true_cat.shape[0]),
        "conv_spearman_mean": round(conv_spearman, 3),
        "full_spearman_mean": round(full_spearman, 3),
        "overall_mae": round(float(np.abs(err_cat).mean()), 2),
        "near_end_mae_0_5": near_end_mae,
        "conv_mono_rate_diagnostic": round(conv_mono_rate, 3),  # not gated
        "mono_adj_rate_diagnostic": round(float(np.mean(mono_adj)), 3) if mono_adj else None,
        "buckets": buckets,
        "gates": {
            "conv_spearman_gate": CONV_SPEARMAN_GATE,
            "full_spearman_gate": FULL_SPEARMAN_GATE,
            "near_end_mae_gate": NEAR_END_MAE_GATE,
        },
    }
