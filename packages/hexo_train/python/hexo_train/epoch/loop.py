"""Self-play epoch loop orchestration.

This file owns the order inside the repeatable part of training. It does not
perform self-play, sample indexing, symmetry selection, training, or checkpoint
IO directly. Instead, it calls the focused helpers in this package and returns
one `EpochResult` per epoch.

The fixed epoch order is:

1. generate self-play from the current checkpoint;
2. finalize result-dependent samples;
3. select the sample window for training;
4. select deterministic D6 symmetries for that window;
5. train the model for the configured number of passes;
6. save an epoch checkpoint for the next epoch;
7. run model-owned epoch evaluation when the plugin provides it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from hexo_train.checkpoints import save_epoch_checkpoint
from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext

from .samples import finalize_samples, select_training_samples
from .selfplay import generate_selfplay
from .symmetry import select_epoch_symmetries
from .training import train_passes


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Serializable summary of one self-play training epoch.

    The fields intentionally mirror the epoch order so diagnostics and tests can
    show exactly what happened without searching through unrelated state.
    """

    epoch: int
    selfplay: Mapping[str, Any]
    samples: Mapping[str, Any]
    symmetries: Mapping[str, Any]
    training: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    evaluation: Mapping[str, Any]


def run_epochs(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Run every configured epoch and record per-epoch diagnostics.

    Epoch numbers are one-based because they are user-facing: checkpoint files
    and diagnostics use `epoch_000001`, `epoch_000002`, and so on.
    """

    start_epoch = _start_epoch(components)
    results: list[EpochResult] = []
    for epoch in range(start_epoch, ctx.config.loop.epochs + 1):
        # Treat the whole epoch as one diagnostic unit while `EpochResult`
        # preserves the finer-grained outcomes inside it.
        started_at = ctx.diagnostics.start_stage(f"epoch_{epoch:06d}")
        try:
            result = run_epoch(ctx, components, epoch=epoch)
        except Exception as exc:
            ctx.diagnostics.finish_stage(
                stage=f"epoch_{epoch:06d}",
                started_at=started_at,
                status="failed",
                metadata={"error": repr(exc)},
            )
            raise

        ctx.remember_epoch(result)
        results.append(result)
        ctx.diagnostics.finish_stage(
            stage=f"epoch_{epoch:06d}",
            started_at=started_at,
            status="completed",
            metadata={"result": result},
        )

    return {
        "epochs": len(results),
        "start_epoch": start_epoch,
        "target_epoch": ctx.config.loop.epochs,
        "results": tuple(results),
    }


def run_epoch(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    epoch: int,
) -> EpochResult:
    """Run one epoch in the fixed self-play training order.

    The checkpoint saved at the end updates `components.shared.checkpoint_state`
    so the next call to `generate_selfplay()` sees the latest model state.
    """

    selfplay_result = generate_selfplay(ctx, components, epoch=epoch)
    finalize_result = finalize_samples(ctx, components, epoch=epoch)
    sample_result = select_training_samples(ctx, components, epoch=epoch)
    symmetry_result = select_epoch_symmetries(ctx, components, epoch=epoch)
    training_result = train_passes(ctx, components, epoch=epoch)
    checkpoint_result = save_epoch_checkpoint(ctx, components, epoch=epoch)
    evaluation_result = evaluate_epoch(ctx, components, epoch=epoch)

    return EpochResult(
        epoch=epoch,
        selfplay=selfplay_result,
        samples={
            "finalize": finalize_result,
            "selection": sample_result,
        },
        symmetries=symmetry_result,
        training=training_result,
        checkpoint=checkpoint_result,
        evaluation=evaluation_result,
    )


def evaluate_epoch(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Run optional model-owned evaluation after the epoch checkpoint exists."""

    plugin = components.model.plugin
    if hasattr(plugin, "evaluate_epoch"):
        return plugin.evaluate_epoch(ctx=ctx, components=components, epoch=epoch)
    return {
        "status": "skipped",
        "epoch": epoch,
        "reason": "model plugin has no evaluate_epoch hook",
    }


def _start_epoch(components: TrainingComponents) -> int:
    """Derive the first epoch to run from the loaded checkpoint state.

    LOAD-BEARING resume contract: a plugin checkpoint loader (e.g.
    packages/hexfield/python/hexfield/checkpoints.py) returns
    ``{"status": "loaded", "epoch": N}`` for a full resume, and this helper
    fast-forwards the loop to epoch N+1 — this is how a halted run restarts
    from its latest checkpoint. Any other shape (weights-only initialize,
    missing/odd epoch field, no loader) starts from epoch 1. The shape is a
    duck-typed dict convention, not a typed contract; keep both sides in sync.
    """

    state = components.shared.checkpoint_state
    if not isinstance(state, Mapping) or state.get("status") != "loaded":
        return 1
    try:
        loaded_epoch = int(state.get("epoch") or 0)
    except (TypeError, ValueError):
        return 1
    return max(1, loaded_epoch + 1)
