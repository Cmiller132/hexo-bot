"""Shared training orchestration package.

`hexo_train` owns config loading, run setup, self-play epoch orchestration,
checkpoints, and diagnostics. Model packages still own model architecture,
sample decoding, losses, target semantics, and model-specific training behavior.

The package exports the stable public surface for callers and model packages:
config dataclasses, `RunContext`, `TrainingPipeline`, plugin loading, and the
training-owned D6 selector types.
"""

from __future__ import annotations

from .config import (
    CheckpointConfig,
    LoopConfig,
    SamplesConfig,
    SelfPlayConfig,
    TrainConfig,
    TrainingConfig,
    load_training_config,
)
from .context import RunContext
from .pipeline import TrainingPipeline
from .registry import load_model_plugin
from .symmetry import D6SymmetrySelector, SampleSymmetrySelection

__all__ = [
    "CheckpointConfig",
    "D6SymmetrySelector",
    "LoopConfig",
    "RunContext",
    "SamplesConfig",
    "SampleSymmetrySelection",
    "SelfPlayConfig",
    "TrainConfig",
    "TrainingConfig",
    "TrainingPipeline",
    "load_model_plugin",
    "load_training_config",
]
