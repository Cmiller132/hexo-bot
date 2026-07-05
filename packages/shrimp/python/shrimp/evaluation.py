"""Per-epoch strength evaluation. Runs a multistage strength eval against a
fixed roster of checkpoints and a head-health audit of the moves-left head.
Results are written to diagnostics."""

from __future__ import annotations

import json
import time
from typing import Any

import torch

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from .config import build_divergence_overrides, parse_shrimp_config
from .geometry import unpack_action_id
from .inference import ShrimpEvaluator
from .model import ShrimpNet


def _play_pair(session_a, eval_a, session_b, eval_b, *, visits, c_puct, seed, sp,
               max_plies, opening_plies=8, opening_temperature=1.0,
               divergence_overrides=None, divergence_overrides_b=None):
    """Play one game. A is player0. Returns (winner_int|None, plies).

    The first `opening_plies` moves use temperature sampling (seeded per game);
    after the opening, both sides play greedy (temperature 0). Sampling is
    applied symmetrically to both sides."""

    state = api.new_game()
    sessions = (session_a, session_b)
    evaluators = (eval_a, eval_b)
    # Both seats use the same divergence overrides by default. Passing
    # divergence_overrides_b gives seat B a different search config.
    ov_a = divergence_overrides if divergence_overrides is not None else build_divergence_overrides(sp)
    ov_b = divergence_overrides_b if divergence_overrides_b is not None else ov_a
    overrides_by_mover = (ov_a, ov_b)
    ply = 0
    while ply < max_plies:
        mover = 0 if ply == 0 else (1 if ((ply - 1) // 2) % 2 == 0 else 0)
        session = sessions[mover]
        evaluator = evaluators[mover]
        temperature = opening_temperature if ply < opening_plies else 0.0
        result = session.search(
            [seed], (state,), visits=visits, c_puct=c_puct, temperature=temperature,
            seed=seed * 5003 + ply, evaluator=evaluator,
            virtual_batch_size=8,
            widening_policy_mass=sp.widening_policy_mass,
            widening_max_children=sp.widening_max_children,
            widening_min_children=sp.widening_min_children,
            fpu_reduction=sp.fpu_reduction, tss_enabled=sp.tss_enabled,
            search_parity_mode=sp.search_parity_mode,
            divergence_overrides=overrides_by_mover[mover],
        )[0]
        q, r = unpack_action_id(int(result["action_id"]))
        outcome = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        ply += 1
        if outcome.terminal:
            terminal = api.terminal(state)
            return (0 if str(terminal.winner) == "player0" else 1), ply
    return None, ply


def evaluate_epoch(*, ctx, components, epoch: int) -> dict[str, Any]:
    cfg = parse_shrimp_config(ctx.config.model.config)
    sp = cfg.selfplay
    mse_cfg = cfg.multi_stage_eval
    current = components.model.model
    started = time.time()
    result: dict[str, Any] = {"status": "completed", "epoch": epoch}

    # Multistage strength eval. Plays games_budget paired games against a fixed
    # roster of checkpoints (permanent anchors, a sliding bracket, and a lagged
    # champion) at full_search_visits and eval_virtual_batch_size, pooling every
    # edge into a persisted append-only Bradley-Terry pool. Reports a
    # PROMOTE/REGRESS/INCONCLUSIVE label and writes its own diagnostics plus the
    # pool; it does not gate, halt, or promote. Errors are recorded in the result
    # and do not propagate.
    every = max(int(mse_cfg.every_n_epochs), 1)
    cand_path = ctx.checkpoint_dir / f"epoch_{epoch:06d}.pt"
    if not mse_cfg.enabled:
        result["multistage"] = {"status": "disabled"}
    elif epoch < 2:
        result["multistage"] = {"status": "skipped", "reason": "no opponent yet (epoch<2)"}
    elif epoch % every != 0:
        result["multistage"] = {"status": "skipped", "reason": f"every_n_epochs={every}"}
    elif not cand_path.is_file():
        result["multistage"] = {"status": "skipped", "reason": "candidate checkpoint missing"}
    else:
        try:
            from . import multistage_eval as mse  # lazy import

            # Plays SealBot and all checkpoint opponents in one batched pass with
            # a single shared candidate forward across opponents.
            report = mse.run_multistage_eval_concurrent(
                ctx.output_dir,
                cand_path,
                cfg,
                candidate_epoch=epoch,
                checkpoints_dir=ctx.checkpoint_dir,
                diagnostics_dir=ctx.diagnostics_dir,
                write_diagnostics=True,
            )
            meta = report.get("meta") or {}
            verdict = report.get("verdict") or {}
            result["multistage"] = {
                "status": "completed",
                "verdict": verdict.get("label"),
                "anchor": meta.get("anchor"),
                "elapsed_seconds": meta.get("elapsed_seconds"),
                "diagnostics_path": meta.get("diagnostics_path"),
            }
        except Exception as exc:  # record error; do not propagate
            result["multistage"] = {"status": "error", "error": repr(exc)}
        finally:
            # Restore the live training model to train() before the head audit;
            # the eval runs opponent models in eval().
            current.train()
    result["elapsed_seconds"] = round(time.time() - started, 1)
    # Moves-left head audit. Audits the head on recent shards and records the
    # result as a diagnostic (informational only; it does not gate the lever).
    try:
        from .head_audit import audit_moves_left_head

        shards = sorted(
            ctx.samples_dir.glob("epoch_*/game_*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        head = audit_moves_left_head(current, shards, device=cfg.device, max_games=40)
        result["moves_left_head_audit"] = head
        current.train()  # restore to train(); the audit runs the model in eval()
    except Exception as exc:  # record error; do not propagate
        result["moves_left_head_audit"] = {"error": repr(exc)}
    diag_path = ctx.diagnostics_dir / f"shrimp.evaluation.epoch_{epoch:06d}.json"
    diag_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
