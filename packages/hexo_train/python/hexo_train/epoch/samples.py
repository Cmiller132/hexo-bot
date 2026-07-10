"""Sample lifecycle helpers for self-play epochs.

This file owns the sample-buffer mechanics that are shared across model
families. It opens the store once during run initialization, lets the model
finalize model-owned records after self-play, and builds the sample window that
the trainer will consume.

The model still owns payload schemas and tensor decoding. The shared sample
helpers only handle storage/index/window mechanics.

Production reality: the real model plugins set
`uses_shared_sample_store=False` and own their replay storage (the hexfield
lineage writes NPZ shards from its own selfplay and selects windows through
its trainer's `select_training_samples` — see
packages/hexfield/python/hexfield/{selfplay,trainer}.py). The
`hexo_utils.samples` JSON-chunk store path below is therefore exercised only
by FakePlugin-style pipeline tests, never on a production run.
"""

from __future__ import annotations

from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def prepare_sample_store(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Open or create the model-owned sample store for this run.

    The store is prepared once before epochs begin. Each epoch can then append
    finalized samples and rebuild an index/window over the same store.
    """

    from hexo_utils.samples import open_sample_store

    sample_config = ctx.section("samples")
    sample_store = open_sample_store(
        sample_config.get("path", ctx.samples_dir),
        mode=str(sample_config.get("mode", "append")),
        metadata={"run": ctx.config.run.name},
    )
    components.shared.sample_store = sample_store
    return {"path": str(sample_store.path), "mode": sample_store.mode}


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
    if not components.model.uses_shared_sample_store:
        return {
            "status": "skipped",
            "epoch": epoch,
            "reason": "model finalizes samples during self-play",
        }
    return {
        "status": "skipped",
        "epoch": epoch,
        "reason": "model sample finalizer not wired yet",
    }


def select_training_samples(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Refresh the sample index and choose the training window for this epoch.

    Step by step:

    1. Refresh the index after this epoch's finalizer may have appended data.
    2. Build a bounded or full training window from that index.
    3. Store both handles on `components.shared` for symmetry and training.
    """

    trainer = components.model.trainer
    if trainer is not None and hasattr(trainer, "select_training_samples"):
        # Production path: the model trainer (e.g. hexfield) builds a
        # KataGo-style replay window over its own NPZ shards here; the
        # shared-store code below never runs for it.
        return trainer.select_training_samples(ctx=ctx, components=components, epoch=epoch)

    from hexo_utils.samples import build_sample_window, refresh_sample_index

    sample_index = refresh_sample_index(components.shared.sample_store)
    components.shared.sample_index = sample_index

    # `train_sample_count=None` means "use the full indexed set"; otherwise the
    # sample helper creates a bounded window of that size.
    sample_window = build_sample_window(
        sample_index,
        window_size=ctx.config.samples.train_sample_count,
        seed=ctx.config.run.seed,
    )
    components.shared.sample_window = sample_window

    return {
        "epoch": epoch,
        "sample_count": sample_index.sample_count,
        "window_size": sample_window.window_size,
        "seed": sample_window.seed,
        "store": str(sample_index.store.path),
        "metadata": dict(sample_window.metadata),
    }
