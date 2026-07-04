"""Sample lifecycle helpers for self-play epochs.

This file owns the sample-buffer mechanics shared across model families. The
model finalizes model-owned records after self-play, and the trainer selects
the sample window it consumes.

The model owns payload schemas, tensor decoding, replay storage, and window
selection. Every shipped plugin sets `uses_shared_sample_store=False` and owns
its own replay storage: the hexfield lineage writes NPZ shards from its own
self-play and selects windows through its trainer's `select_training_samples`
(packages/hexfield/python/hexfield/{selfplay,trainer}.py). `hexo_train` no
longer ships a shared JSON-chunk sample store.
"""

from __future__ import annotations

from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def finalize_samples(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Let the model finalize result-dependent samples after self-play.

    Self-play decisions can be written as pending samples while the game is in
    progress. The terminal result is only known after the game, so model-owned
    finalizers attach values, weights, or other result-dependent fields here.
    """

    finalizer = components.model.sample_finalizer
    if finalizer is not None:
        # The finalizer owns sample meaning. `hexo_train` only supplies the run
        # context, component handles, and current epoch number.
        return finalizer.finalize(ctx=ctx, components=components, epoch=epoch)
    return {
        "status": "skipped",
        "epoch": epoch,
        "reason": "model finalizes samples during self-play",
    }


def select_training_samples(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Choose the training window for this epoch via the model-owned trainer.

    The trainer owns replay storage and window selection (the hexfield trainer
    builds a KataGo-style shuffle over its mtime-ordered NPZ shard window).
    """

    trainer = components.model.trainer
    if trainer is not None and hasattr(trainer, "select_training_samples"):
        return trainer.select_training_samples(ctx=ctx, components=components, epoch=epoch)

    raise RuntimeError(
        "select_training_samples requires a model trainer that owns replay "
        "storage; hexo_train no longer ships a shared sample store."
    )
