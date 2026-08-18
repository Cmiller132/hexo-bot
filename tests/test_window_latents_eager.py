"""Equivalence detector for the vectorized window-latent eager forwards.

The research repo's CPU path was a literal per-position/per-window loop
oracle; the port replaced it with vectorized fp32 implementations because the
loops took seconds per late-game board on the serve devices. The loop oracle
lives HERE now: these tests re-state it literally and require the vectorized
paths to agree on ragged layouts, empty runs, and the empty batch. Making the
implementation agree with itself would delete the detector — do not import
the oracle from the package.
"""

import math

import pytest
import torch

from mantisnet import window_latents
from mantisnet.window_latents import (
    _eager_broadcast_forward,
    _eager_read_forward,
)


def _oracle_read(q, k, v, offsets, order):
    positions, slots, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    out = torch.zeros((positions, slots, heads, hd), dtype=torch.float32)
    m = torch.full((positions, slots, heads), -float("inf"), dtype=torch.float32)
    l = torch.zeros((positions, slots, heads), dtype=torch.float32)
    for position in range(positions):
        rows = order[offsets[position] : offsets[position + 1]]
        if not rows.numel():
            continue
        for slot in range(slots):
            for head in range(heads):
                scores = (
                    q[position, slot, head].float()[None, :]
                    * k.index_select(0, rows)[:, head].float()
                ).sum(-1) * scale
                row_max = scores.max()
                numer = (scores - row_max).exp()
                denom = numer.sum()
                out[position, slot, head] = (
                    numer[:, None] * v.index_select(0, rows)[:, head].float()
                ).sum(0) / denom
                m[position, slot, head] = row_max
                l[position, slot, head] = denom
    return out, m, l


def _oracle_broadcast(q, k, v, window_pos):
    windows, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    out = torch.empty((windows, heads, hd), dtype=torch.float32)
    m = torch.empty((windows, heads), dtype=torch.float32)
    l = torch.empty((windows, heads), dtype=torch.float32)
    for window in range(windows):
        position = window_pos[window]
        for head in range(heads):
            scores = (
                q[window, head].float()[None, :] * k[position, :, head].float()
            ).sum(-1) * scale
            row_max = scores.max()
            numer = (scores - row_max).exp()
            denom = numer.sum()
            out[window, head] = (
                numer[:, None] * v[position, :, head].float()
            ).sum(0) / denom
            m[window, head] = row_max
            l[window, head] = denom
    return out, m, l


def _layout(window_pos, positions):
    order = torch.argsort(window_pos.to(torch.int32), stable=True)
    sorted_pos = window_pos.index_select(0, order)
    offsets = torch.searchsorted(sorted_pos, torch.arange(positions + 1))
    return offsets, order


SLOTS, HEADS, HD = 4, 4, 32


@pytest.mark.parametrize(
    "counts",
    [
        [3],
        [0],
        [0, 1, 7, 0, 40, 2],
        [17, 17, 17],
        [0, 0, 0],
    ],
)
def test_read_matches_the_loop_oracle(counts):
    torch.manual_seed(20260818)
    positions = len(counts)
    window_pos = torch.repeat_interleave(
        torch.arange(positions), torch.tensor(counts)
    )
    # Shuffle so the layout's stable sort actually reorders something.
    window_pos = window_pos[torch.randperm(window_pos.shape[0])]
    windows = int(window_pos.shape[0])
    offsets, order = _layout(window_pos, positions)
    q = torch.randn(positions, SLOTS, HEADS, HD)
    k = torch.randn(windows, HEADS, HD)
    v = torch.randn(windows, HEADS, HD)

    out, m, l = _eager_read_forward(q, k, v, offsets, order)
    ref_out, ref_m, ref_l = _oracle_read(q, k, v, offsets, order)
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(m, ref_m, rtol=1e-5, atol=1e-6, equal_nan=True)
    torch.testing.assert_close(l, ref_l, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("counts", [[1], [4, 0, 9], [25, 25]])
def test_broadcast_matches_the_loop_oracle(counts):
    torch.manual_seed(20260818)
    positions = len(counts)
    window_pos = torch.repeat_interleave(
        torch.arange(positions), torch.tensor(counts)
    )
    windows = int(window_pos.shape[0])
    q = torch.randn(windows, HEADS, HD)
    k = torch.randn(positions, SLOTS, HEADS, HD)
    v = torch.randn(positions, SLOTS, HEADS, HD)

    out, m, l = _eager_broadcast_forward(q, k, v, window_pos)
    ref_out, ref_m, ref_l = _oracle_broadcast(q, k, v, window_pos)
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(m, ref_m, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(l, ref_l, rtol=1e-5, atol=1e-6)


def test_the_dispatch_ops_take_the_eager_path_on_cpu():
    """read_attention / broadcast_attention on CPU are the eager forwards."""
    torch.manual_seed(7)
    counts = [2, 0, 5]
    positions = len(counts)
    window_pos = torch.repeat_interleave(
        torch.arange(positions), torch.tensor(counts)
    )
    windows = int(window_pos.shape[0])
    offsets, order = _layout(window_pos, positions)
    q_read = torch.randn(positions, SLOTS, HEADS, HD)
    k_flat = torch.randn(windows, HEADS, HD)
    v_flat = torch.randn(windows, HEADS, HD)
    out = window_latents.read_attention(
        q_read, k_flat, v_flat, window_pos, offsets, order
    )
    ref_out, _, _ = _oracle_read(q_read, k_flat, v_flat, offsets, order)
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-6)

    q_b = torch.randn(windows, HEADS, HD)
    k_b = torch.randn(positions, SLOTS, HEADS, HD)
    v_b = torch.randn(positions, SLOTS, HEADS, HD)
    out = window_latents.broadcast_attention(
        q_b, k_b, v_b, window_pos, offsets, order
    )
    ref_out, _, _ = _oracle_broadcast(q_b, k_b, v_b, window_pos)
    torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-6)
