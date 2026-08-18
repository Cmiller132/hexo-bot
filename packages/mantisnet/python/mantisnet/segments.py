"""Segmented reductions over ragged per-position rows.

Flat outputs bounded by a ``(P + 1,)`` CSR offset tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor


def segment_ids(offsets: Tensor) -> Tensor:
    """(P + 1,) CSR offsets to a flat (N,) tensor of position ids."""
    p = offsets.shape[0] - 1
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(torch.arange(p, device=offsets.device), counts)


def segment_sum(values: Tensor, seg: Tensor, p: int) -> Tensor:
    """Per-segment sum of a flat (N,) tensor into (P,)."""
    return values.new_zeros(p).index_add_(0, seg, values)


def segment_max(values: Tensor, seg: Tensor, p: int) -> Tensor:
    """Per-segment maximum of a flat ``(N,)`` tensor into ``(P,)``.

    Every model position has at least one legal cell, so the identity fill is
    never returned.
    """
    out = values.new_full((p,), torch.finfo(values.dtype).min)
    return out.index_reduce_(0, seg, values, "amax", include_self=True)


def segment_log_softmax(values: Tensor, offsets: Tensor) -> Tensor:
    """log-softmax within each segment, numerically shifted per segment."""
    p = offsets.shape[0] - 1
    seg = segment_ids(offsets)
    shifted = values - segment_max(values, seg, p).index_select(0, seg)
    lse = segment_sum(shifted.exp(), seg, p).log()
    return shifted - lse.index_select(0, seg)
