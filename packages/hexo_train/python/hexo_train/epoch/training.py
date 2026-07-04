"""Training passes for one self-play epoch.

This file is deliberately small because the model owns the real training work:
decoding samples, applying D6 transforms, batching tensors, computing losses,
and stepping the optimizer. `hexo_train` only decides when training happens and
how many passes are requested.
"""

from __future__ import annotations

from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def train_passes(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Train over the selected sample window for the configured pass count.

    A trainer receives both the sample window and the symmetry selection. That
    keeps the orchestration model-neutral while giving the model enough context
    to produce consistent augmented tensors and targets.
    """

    passes = ctx.config.train.passes_per_epoch
    trainer = components.model.trainer

    if trainer is not None and hasattr(trainer, "train_passes"):
        # The trainer contract is pass-based, not step-based. A model can define
        # whether one pass means full-window iteration, a fixed batch budget, or
        # another architecture-specific training unit.
        return trainer.train_passes(
            passes=passes,
            sample_window=components.shared.sample_window,
            sample_symmetries=components.shared.sample_symmetries,
            ctx=ctx,
            components=components,
            epoch=epoch,
        )

    return {
        "status": "skipped",
        "epoch": epoch,
        "passes": passes,
        "reason": "model trainer not wired yet",
    }
