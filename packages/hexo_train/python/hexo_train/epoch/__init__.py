"""Self-play epoch loop package.

This package contains the repeatable middle of a training run. `loop.py` owns
ordering, while the sibling modules each own one responsibility inside an
epoch: self-play, sample preparation, symmetry selection, and training.
"""

from .loop import EpochResult, run_epoch, run_epochs

__all__ = [
    "EpochResult",
    "run_epoch",
    "run_epochs",
]
