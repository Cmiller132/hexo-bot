"""Per-epoch strength evaluation. Runs a multistage strength eval against a
fixed roster of checkpoints and a head-health audit of the moves-left head.
Results are written to diagnostics."""

from __future__ import annotations

import json
import time
from typing import Any


from .config import ML_AUTO_DISABLED_FLAG, parse_hexfield_config


def evaluate_epoch(*, ctx, components, epoch: int) -> dict[str, Any]:
    cfg = parse_hexfield_config(ctx.config.model.config)
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
    # Moves-left head audit. Audits the head on recent shards. If the audit
    # fails and moves_left_utility is enabled, write the run-dir flag that forces
    # the lever off next epoch (read by build_divergence_overrides); if the audit
    # passes, clear the flag.
    try:
        from .head_audit import audit_moves_left_head

        shards = sorted(
            ctx.samples_dir.glob("epoch_*/game_*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        head = audit_moves_left_head(current, shards, device=cfg.device, max_games=40)
        flag = ctx.diagnostics_dir / ML_AUTO_DISABLED_FLAG
        if head.get("passed"):
            flag.unlink(missing_ok=True)
        elif sp.moves_left_utility:
            flag.write_text(json.dumps({"epoch": epoch, "audit": head}), encoding="utf-8")
        result["moves_left_head_audit"] = head
        result["ml_auto_disabled"] = bool(not head.get("passed") and sp.moves_left_utility)
        current.train()  # restore to train(); the audit runs the model in eval()
    except Exception as exc:  # record error; do not propagate
        result["moves_left_head_audit"] = {"error": repr(exc)}
    diag_path = ctx.diagnostics_dir / f"hexfield.evaluation.epoch_{epoch:06d}.json"
    diag_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
