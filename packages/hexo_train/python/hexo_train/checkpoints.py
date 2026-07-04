"""Checkpoint load/save helpers for self-play training.

`hexo_train` owns when checkpoints are loaded and saved. The selected model
plugin owns what a checkpoint contains. These helpers therefore delegate to
model-provided loader/saver objects when they exist, and otherwise write small
placeholder metadata files so the pipeline remains executable during early
development. All four registered plugins ship a real loader/saver (e.g.
packages/dense_cnn_restnet/python/dense_cnn_restnet/checkpoints.py), so the
placeholder branches only fire for the FakePlugin unit tests in
tests/test_training_pipeline_simplification.py.

Cross-module contract: the loader's return dict is stored on
`components.shared.checkpoint_state` and read by epoch/loop.py `_start_epoch`
(`{"status": "loaded", "epoch": N}` drives epoch fast-forward on resume) and
by epoch/selfplay.py via the shared state.

Called by pipeline.py (`load_checkpoint`, `publish_final_model` steps) and
epoch/loop.py (`save_epoch_checkpoint` once per epoch).
"""

from __future__ import annotations

from typing import Any

from .components import TrainingComponents
from .context import RunContext


def load_or_initialize_checkpoint(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Load an existing checkpoint or describe a fresh initialization.

    This runs once before epoch 1. The resulting state is stored on
    `components.shared.checkpoint_state` so self-play generation can see the
    current model reference.
    """

    checkpoint_ref = (
        ctx.config.checkpoint.resume_from
        or ctx.config.checkpoint.initialize_from
    )
    loader = components.model.checkpoint_loader
    if loader is not None:
        state = loader.load(checkpoint_ref, ctx=ctx, components=components)
    else:
        state = {
            "status": "initialized" if checkpoint_ref is None else "referenced",
            "checkpoint_ref": str(checkpoint_ref) if checkpoint_ref else None,
            "note": "Model checkpoint loading is not implemented yet.",
        }

    components.shared.checkpoint_state = state
    return state


def save_epoch_checkpoint(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Save the checkpoint that will feed the next epoch's self-play."""

    return _save_checkpoint(
        ctx,
        components,
        name=f"epoch_{epoch:06d}",
        metadata={"epoch": epoch, "kind": "epoch"},
    )


def save_final_checkpoint(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Save the final model checkpoint for this run.

    Epoch checkpoints are named by epoch number. The final checkpoint uses the
    user-configured `checkpoint.save_name` so external tools have a stable name.
    """

    return _save_checkpoint(
        ctx,
        components,
        name=ctx.config.checkpoint.save_name,
        metadata={"kind": "final"},
    )


def _save_checkpoint(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Shared save path for epoch and final checkpoints.

    The helper updates `components.shared.checkpoint_state` after every save.
    That is the small but important handoff from training to the next self-play
    generation call.
    """

    saver = components.model.checkpoint_saver
    if saver is not None:
        path = saver.save(name=name, ctx=ctx, components=components)
    else:
        path = components.shared.defaults.checkpoint_store.write_placeholder(
            name,
            {
                "model": ctx.config.model.name,
                "note": "Placeholder checkpoint; model saver not wired yet.",
                **metadata,
            },
        )

    result = {"checkpoint_path": str(path), "name": name, **metadata}
    if metadata.get("kind") == "epoch":
        pointer = _publish_epoch_checkpoint_pointer(ctx, result)
        if pointer is not None:
            result["pointer"] = pointer
    components.shared.checkpoint_state = result
    return result


def _publish_epoch_checkpoint_pointer(ctx: RunContext, checkpoint_result: dict[str, Any]) -> dict[str, Any] | None:
    """Write the per-epoch checkpoint pointer file when configured.

    Near-duplicate of artifacts.publish_selfplay_checkpoint_pointer (the final
    variant); the two writers must stay format-identical. Only legacy
    dense_cnn/hexgt configs set `update_checkpoint_pointer = true` — all
    restnet main_* configs disable it, so this returns None on the active run.
    """

    if not ctx.config.selfplay.update_checkpoint_pointer:
        return None
    pointer_path = ctx.config.selfplay.checkpoint_pointer
    if pointer_path is None:
        pointer_path = ctx.output_dir / "selfplay_checkpoint.txt"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(str(checkpoint_result.get("checkpoint_path") or ""), encoding="utf-8")
    return {"status": "updated", "pointer_path": str(pointer_path)}
