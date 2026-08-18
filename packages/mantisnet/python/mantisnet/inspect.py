"""Inspect a model's policy and action values at one move prefix.

Ported from the research repo's ``mantisnet.klent.inspect`` (branch main,
MODEL_REPR_VERSION 7) with the checkpoint-path convenience removed: serving
always holds a loaded model, so loading lives in ``mantisnet.serve`` alone.
"""

from __future__ import annotations

import torch

from .builder import collate_prefixes
from .improve import improved_policy
from .segments import segment_log_softmax


def inspect_position(
    model: torch.nn.Module,
    moves,
    t: int,
    tau: float,
    lam: float,
    mass_floor: float,
    device: str = "cpu",
) -> dict:
    """The model's policy, Q, π′, and v̂ over the legal set at ``moves[:t]``.

    ``tau`` and ``lam`` are required because π′ depends on them. For a recorded
    ply, callers must use the invocation parameters that applied to its
    iteration; the latest ``config.json`` can differ after resume.

    Returns the position's own scalars plus one entry per legal move, in the
    engine order used by telemetry ranks. The model is left in eval mode.
    """
    from . import _rust as hexo_py

    moves = [tuple(m) for m in moves]
    if not 0 <= t <= len(moves):
        raise ValueError(f"ply {t} is outside the {len(moves)}-move line")
    model.eval()

    position = hexo_py.Position.replay(moves[:t])
    if position.is_terminal:
        raise ValueError(f"the position after {t} plies is terminal: nothing to choose")

    batch = collate_prefixes([moves], [t]).to(device)
    with torch.no_grad():
        _s, w, g, cells = model.trunk(batch)
        logits, q_score, q_values = model.cell_heads(w, g, cells, batch, mass_floor)
        logits, q_values = logits.float().cpu(), q_values.float().cpu()
        q_score = q_score.float().cpu()
    offsets = batch.legal_offsets.cpu()
    log_pi = segment_log_softmax(logits, offsets)
    imp = improved_policy(logits, q_score, q_values, offsets, tau, lam)

    legal = position.legal_moves()
    if len(legal) != logits.shape[0]:
        raise RuntimeError(
            f"builder produced {logits.shape[0]} cells for a position with "
            f"{len(legal)} legal moves"
        )
    return {
        "moves": moves[:t],
        "t": t,
        "mover": position.current_player,
        "moves_remaining": position.moves_remaining,
        "stone_count": position.stone_count,
        "legal_count": len(legal),
        "tau": tau,
        "lam": lam,
        "mass_floor": mass_floor,
        "v_hat": float(imp.v_hat[0]),
        "kl": float(imp.kl[0]),
        "norm_entropy": float(imp.norm_entropy[0]),
        # The next recorded move, if the supplied line continues.
        "played": moves[t] if t < len(moves) else None,
        "legal": [
            {
                "move": move,
                "rank": rank,
                "logit": float(logits[rank]),
                "policy": float(log_pi[rank].exp()),
                "q": float(q_values[rank]),
                "q_score": float(q_score[rank]),
                "improved": float(imp.probs[rank]),
            }
            for rank, move in enumerate(legal)
        ],
    }
