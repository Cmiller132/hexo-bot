"""Epoch-specific D6 symmetry selection.

The root `hexo_train.symmetry` module defines reusable selector types. This
file applies a selector at the right moment in the epoch: after the sample
window exists and before the model trainer decodes samples into tensors.
"""

from __future__ import annotations

from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def select_epoch_symmetries(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Attach deterministic D6 choices to the current epoch sample window.

    The selected symmetries are stored on `components.shared` so model trainers
    can apply the same transform to inputs, masks, policy targets, and any
    model-owned payloads.

    Caveat for the active lineage: the restnet trainer
    (packages/dense_cnn_restnet/python/dense_cnn_restnet/trainer.py) consumes
    only `selection.seed` and re-draws its own per-row D6 symmetries during
    shard expansion, so the per-sample tuple built here (one blake2b per
    visible sample, over windows of hundreds of thousands of rows) is computed
    and discarded, and the `symmetry_count` diagnostic does not describe what
    that trainer actually applied.
    """

    # Model packages may override the selector, but the default selector keeps
    # D6 choices deterministic from run seed, epoch, and sample index.
    selector = (
        components.model.symmetry_selector
        or components.shared.defaults.symmetry_selector
    )
    selection = selector.select_for_window(
        components.shared.sample_window,
        seed=ctx.config.run.seed,
        epoch=epoch,
    )
    components.shared.sample_symmetries = selection
    return {
        "epoch": epoch,
        "symmetry_count": len(selection.symmetries),
        "seed": selection.seed,
        "metadata": dict(selection.metadata),
    }
