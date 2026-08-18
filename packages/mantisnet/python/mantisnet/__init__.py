"""MantisNet: the stone/window graph network for Hexo — inference port.

The serve-side subset of the MantisNet research package (branch main,
MODEL_REPR_VERSION 7): the model, the batch builder over the native encoder,
the closed-form policy improvement, and the position readout. Training,
self-play, and the fit loop stay in the research repo.
"""

from .builder import (
    MODEL_REPR_VERSION,
    Batch,
    collate_positions,
    collate_prefixes,
)
from .improve import ImprovedPolicy, improved_policy
from .inspect import inspect_position
from .model import CRITIC_LOGITS, MantisConfig, MantisNet, strip_legacy_knobs
from .serve import KlentParams, LoadedCheckpoint, PositionRead, load_checkpoint, read_positions

__all__ = [
    "MODEL_REPR_VERSION",
    "Batch",
    "CRITIC_LOGITS",
    "ImprovedPolicy",
    "KlentParams",
    "LoadedCheckpoint",
    "MantisConfig",
    "MantisNet",
    "PositionRead",
    "collate_positions",
    "collate_prefixes",
    "improved_policy",
    "inspect_position",
    "load_checkpoint",
    "read_positions",
    "strip_legacy_knobs",
]
