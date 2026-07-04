"""Shared D6 symmetry contracts for Hexo model packages.

Tensor encoding and action masks are model-owned. This namespace only exposes
the symmetry identifiers and mapper protocol used by training sample helpers.
"""

from .symmetry import (
    D6_SIZE,
    IDENTITY_D6,
    ActionSymmetryMapper,
    D6Symmetry,
    transform_action_ids,
)

__all__ = [
    "ActionSymmetryMapper",
    "D6_SIZE",
    "D6Symmetry",
    "IDENTITY_D6",
    "transform_action_ids",
]
