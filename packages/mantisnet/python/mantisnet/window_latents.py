"""Fused ragged attention for the Step 2 window-latent cycle.

The read maps each position's flat window run into its four latent queries;
the broadcast maps every flat window query back over those four latents. CUDA
uses Triton programs that own one complete gradient row and never use atomics.
CPU and XPU forwards use the vectorized fp32 eager paths below — the research
repo's per-position/per-window loop oracles took seconds per late-game board,
which is what made non-CUDA serving unusable. The literal loop oracle lives in
``tests/test_window_latents_eager.py`` as the equivalence detector. Backward
still uses the literal loops: serving runs under ``no_grad`` and never reaches
it.
"""

from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_BLOCK_W = 32
_NUM_WARPS = 1
_FAILED_READ_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_READ_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BROADCAST_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BROADCAST_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}


@torch.library.custom_op("mantisnet::window_latent_layout", mutates_args=())
def window_latent_layout(window_pos: Tensor, positions: int) -> tuple[Tensor, Tensor]:
    """Return position CSR offsets and a stable position-major window order."""
    if window_pos.ndim != 1:
        raise ValueError("window_pos must have shape (N_w,)")
    if positions < 0:
        raise ValueError("positions must be nonnegative")
    if window_pos.numel():
        lo = int(window_pos.min().item())
        hi = int(window_pos.max().item())
        if lo < 0 or hi >= positions:
            raise ValueError("window_pos entries must be in [0, positions)")
    # Stable sorting is also the sorted-layout check: already position-major
    # batches produce the identity order and contiguous kernel accesses.
    order = torch.argsort(window_pos.to(torch.int32), stable=True)
    sorted_pos = window_pos.index_select(0, order)
    steps = torch.arange(positions + 1, device=window_pos.device)
    offsets = torch.searchsorted(sorted_pos, steps)
    return offsets, order


@window_latent_layout.register_fake
def _(window_pos, positions):
    return (
        window_pos.new_empty((positions + 1,), dtype=torch.long),
        window_pos.new_empty(window_pos.shape, dtype=torch.long),
    )


def _validate_layout(offsets: Tensor, order: Tensor, positions: int, windows: int) -> None:
    if offsets.ndim != 1 or offsets.shape[0] != positions + 1:
        raise ValueError("offsets must have shape (P + 1,)")
    if order.ndim != 1 or order.shape[0] != windows:
        raise ValueError("order must have shape (N_w,)")


def _validate_read(q, k, v, window_pos, offsets, order) -> None:
    if q.ndim != 4:
        raise ValueError("read q must have shape (P, slots, heads, head_dim)")
    if k.ndim != 3 or k.shape != v.shape:
        raise ValueError("read k and v must share shape (N_w, heads, head_dim)")
    if q.shape[1] != 4:
        raise ValueError("window-latent attention requires exactly four slots")
    if q.shape[2:] != k.shape[1:]:
        raise ValueError("read q, k, and v must share heads and head_dim")
    if window_pos.ndim != 1 or window_pos.shape[0] != k.shape[0]:
        raise ValueError("window_pos must have one entry per window")
    _validate_layout(offsets, order, q.shape[0], k.shape[0])


def _validate_broadcast(q, k, v, window_pos, offsets, order) -> None:
    if q.ndim != 3:
        raise ValueError("broadcast q must have shape (N_w, heads, head_dim)")
    if k.ndim != 4 or k.shape != v.shape:
        raise ValueError("broadcast k and v must share shape (P, slots, heads, head_dim)")
    if k.shape[1] != 4:
        raise ValueError("window-latent attention requires exactly four slots")
    if q.shape[1:] != k.shape[2:]:
        raise ValueError("broadcast q, k, and v must share heads and head_dim")
    if window_pos.ndim != 1 or window_pos.shape[0] != q.shape[0]:
        raise ValueError("window_pos must have one entry per window")
    _validate_layout(offsets, order, k.shape[0], q.shape[0])


def _eager_read_forward(q, k, v, offsets, order):
    """Vectorized fp32 read: a padded ragged softmax over each position's run.

    Semantics match the literal loop oracle exactly, including the empty-run
    row: zero output, ``m = -inf``, ``l = 0``. The padding gathers each
    position's window rows into a ``(P, W_max)`` table and masks the tail to
    ``-inf`` before the softmax, so no loop ever runs over positions or
    windows.
    """
    positions, slots, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    counts = (offsets[1:] - offsets[:-1]).to(q.device)
    max_w = int(counts.max().item()) if counts.numel() else 0
    if max_w == 0:
        out = torch.zeros(
            (positions, slots, heads, hd), dtype=torch.float32, device=q.device
        )
        m = torch.full(
            (positions, slots, heads), -float("inf"), dtype=torch.float32,
            device=q.device,
        )
        return out, m, torch.zeros_like(m)

    lanes = torch.arange(max_w, device=q.device)
    valid = lanes[None, :] < counts[:, None]  # (P, W)
    ranks = offsets[:-1, None] + torch.minimum(
        lanes[None, :], (counts[:, None] - 1).clamp(min=0)
    )
    rows = order.index_select(0, ranks.reshape(-1)).reshape(positions, max_w)

    k_pad = k.float()[rows]  # (P, W, heads, hd)
    v_pad = v.float()[rows]
    scores = torch.einsum("pshd,pwhd->pshw", q.float(), k_pad) * scale
    scores = scores.masked_fill(~valid[:, None, None, :], -float("inf"))
    m = scores.amax(dim=-1)  # (P, slots, heads); -inf on an empty run
    shift = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    numer = torch.exp(scores - shift.unsqueeze(-1))
    l = numer.sum(dim=-1)  # 0 on an empty run
    denom = torch.where(l > 0, l, torch.ones_like(l))
    out = torch.einsum("pshw,pwhd->pshd", numer, v_pad) / denom.unsqueeze(-1)
    return out, m, l


def _reference_read_backward(q, k, v, offsets, order, out, m, l, grad_out):
    """Literal read backward, recomputing every window-query edge."""
    positions, slots, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    go = grad_out.float()
    for position in range(positions):
        rows = order[offsets[position] : offsets[position + 1]]
        if not rows.numel():
            continue
        k_rows = k.index_select(0, rows).float()
        v_rows = v.index_select(0, rows).float()
        for slot in range(slots):
            for head in range(heads):
                scores = (
                    q[position, slot, head].float()[None, :] * k_rows[:, head]
                ).sum(-1) * scale
                alpha = (scores - m[position, slot, head]).exp() / l[
                    position, slot, head
                ]
                upstream = go[position, slot, head]
                delta = (upstream * out[position, slot, head]).sum()
                dalpha = (upstream[None, :] * v_rows[:, head]).sum(-1)
                dscore = alpha * (dalpha - delta)
                dq[position, slot, head] += (
                    dscore[:, None] * k_rows[:, head]
                ).sum(0) * scale
                dk[rows, head] += dscore[:, None] * q[position, slot, head].float() * scale
                dv[rows, head] += alpha[:, None] * upstream
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def _eager_broadcast_forward(q, k, v, window_pos):
    """Vectorized fp32 broadcast: every window attends over its four slots.

    Semantics match the literal loop oracle exactly. All four slots are
    always live, so the gather is dense and the softmax needs no mask.
    """
    heads = q.shape[1]
    hd = q.shape[2]
    scale = 1.0 / math.sqrt(hd)
    pos = window_pos.long()
    k_w = k.float()[pos]  # (N_w, slots, heads, hd)
    v_w = v.float()[pos]
    scores = torch.einsum("whd,wshd->wsh", q.float(), k_w) * scale
    m = scores.amax(dim=1)  # (N_w, heads)
    numer = torch.exp(scores - m[:, None, :])
    l = numer.sum(dim=1)
    out = torch.einsum("wsh,wshd->whd", numer, v_w) / l[:, :, None]
    return out, m, l


def _reference_broadcast_backward(q, k, v, window_pos, offsets, order, out, m, l, grad_out):
    """Literal broadcast backward, recomputing all four edges per window."""
    windows, heads, hd = q.shape
    positions, slots = k.shape[:2]
    scale = 1.0 / math.sqrt(hd)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    go = grad_out.float()
    for position in range(positions):
        rows = order[offsets[position] : offsets[position + 1]]
        for window in rows:
            for head in range(heads):
                scores = (
                    q[window, head].float()[None, :] * k[position, :, head].float()
                ).sum(-1) * scale
                alpha = (scores - m[window, head]).exp() / l[window, head]
                upstream = go[window, head]
                delta = (upstream * out[window, head]).sum()
                dalpha = (upstream[None, :] * v[position, :, head].float()).sum(-1)
                dscore = alpha * (dalpha - delta)
                dq[window, head] = (
                    dscore[:, None] * k[position, :, head].float()
                ).sum(0) * scale
                dk[position, :, head] += dscore[:, None] * q[window, head].float() * scale
                dv[position, :, head] += alpha[:, None] * upstream
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


if triton is not None:

    @triton.jit
    def _read_forward_kernel(
        q_ptr, k_ptr, v_ptr, offsets_ptr, order_ptr, out_ptr, m_ptr, l_ptr,
        scale, SLOTS: tl.constexpr, HEADS: tl.constexpr, HD: tl.constexpr,
        BLOCK_HD: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        pid = tl.program_id(0)
        position = pid // (SLOTS * HEADS)
        rem = pid % (SLOTS * HEADS)
        slot = rem // HEADS
        head = rem % HEADS
        dims = tl.arange(0, BLOCK_HD)
        live = dims < HD
        q_row = ((position * SLOTS + slot) * HEADS + head) * HD
        query = tl.load(q_ptr + q_row + dims, mask=live, other=0.0).to(tl.float32)
        start = tl.load(offsets_ptr + position)
        end = tl.load(offsets_ptr + position + 1)
        row_max = -float("inf")
        denom = 0.0
        acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
        for lo in tl.range(start, end, BLOCK_W):
            ranks = lo + tl.arange(0, BLOCK_W)
            inside = ranks < end
            rows = tl.load(order_ptr + ranks, mask=inside, other=0)
            keys = tl.load(
                k_ptr + (rows[:, None] * HEADS + head) * HD + dims[None, :],
                mask=inside[:, None] & live[None, :], other=0.0,
            ).to(tl.float32)
            scores = tl.sum(query[None, :] * keys, axis=1) * scale
            scores = tl.where(inside, scores, -float("inf"))
            next_max = tl.maximum(row_max, tl.max(scores, axis=0))
            rescale = tl.exp(row_max - next_max)
            numer = tl.exp(scores - next_max)
            values = tl.load(
                v_ptr + (rows[:, None] * HEADS + head) * HD + dims[None, :],
                mask=inside[:, None] & live[None, :], other=0.0,
            ).to(tl.float32)
            denom = denom * rescale + tl.sum(numer, axis=0)
            acc = acc * rescale + tl.sum(numer[:, None] * values, axis=0)
            row_max = next_max
        result = tl.where(denom > 0.0, acc / denom, 0.0)
        tl.store(out_ptr + q_row + dims, result, mask=live)
        stat = (position * SLOTS + slot) * HEADS + head
        tl.store(m_ptr + stat, row_max)
        tl.store(l_ptr + stat, denom)

    @triton.jit
    def _read_dq_kernel(
        q_ptr, k_ptr, v_ptr, offsets_ptr, order_ptr, out_ptr, m_ptr, l_ptr,
        go_ptr, dq_ptr, scale, SLOTS: tl.constexpr, HEADS: tl.constexpr,
        HD: tl.constexpr, BLOCK_HD: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        pid = tl.program_id(0)
        position = pid // (SLOTS * HEADS)
        rem = pid % (SLOTS * HEADS)
        slot = rem // HEADS
        head = rem % HEADS
        dims = tl.arange(0, BLOCK_HD)
        live = dims < HD
        q_row = ((position * SLOTS + slot) * HEADS + head) * HD
        query = tl.load(q_ptr + q_row + dims, mask=live, other=0.0).to(tl.float32)
        upstream = tl.load(go_ptr + q_row + dims, mask=live, other=0.0).to(tl.float32)
        output = tl.load(out_ptr + q_row + dims, mask=live, other=0.0)
        delta = tl.sum(upstream * output, axis=0)
        stat = (position * SLOTS + slot) * HEADS + head
        row_max = tl.load(m_ptr + stat)
        denom = tl.load(l_ptr + stat)
        start = tl.load(offsets_ptr + position)
        end = tl.load(offsets_ptr + position + 1)
        acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
        for lo in tl.range(start, end, BLOCK_W):
            ranks = lo + tl.arange(0, BLOCK_W)
            inside = ranks < end
            rows = tl.load(order_ptr + ranks, mask=inside, other=0)
            keys = tl.load(
                k_ptr + (rows[:, None] * HEADS + head) * HD + dims[None, :],
                mask=inside[:, None] & live[None, :], other=0.0,
            ).to(tl.float32)
            values = tl.load(
                v_ptr + (rows[:, None] * HEADS + head) * HD + dims[None, :],
                mask=inside[:, None] & live[None, :], other=0.0,
            ).to(tl.float32)
            scores = tl.sum(query[None, :] * keys, axis=1) * scale
            alpha = tl.where(inside, tl.exp(scores - row_max) / denom, 0.0)
            dalpha = tl.sum(upstream[None, :] * values, axis=1)
            dscore = alpha * (dalpha - delta)
            acc += tl.sum(dscore[:, None] * keys, axis=0)
        element = dq_ptr.dtype.element_ty
        tl.store(dq_ptr + q_row + dims, (acc * scale).to(element), mask=live)

    @triton.jit
    def _read_dkdv_kernel(
        q_ptr, k_ptr, v_ptr, window_pos_ptr, out_ptr, m_ptr, l_ptr, go_ptr,
        dk_ptr, dv_ptr, scale, SLOTS: tl.constexpr, HEADS: tl.constexpr,
        HD: tl.constexpr, BLOCK_HD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        window = pid // HEADS
        head = pid % HEADS
        position = tl.load(window_pos_ptr + window)
        dims = tl.arange(0, BLOCK_HD)
        live = dims < HD
        kv_row = (window * HEADS + head) * HD
        key = tl.load(k_ptr + kv_row + dims, mask=live, other=0.0).to(tl.float32)
        value = tl.load(v_ptr + kv_row + dims, mask=live, other=0.0).to(tl.float32)
        acc_k = tl.zeros([BLOCK_HD], dtype=tl.float32)
        acc_v = tl.zeros([BLOCK_HD], dtype=tl.float32)
        for slot in tl.range(0, SLOTS):
            q_row = ((position * SLOTS + slot) * HEADS + head) * HD
            query = tl.load(q_ptr + q_row + dims, mask=live, other=0.0).to(tl.float32)
            upstream = tl.load(go_ptr + q_row + dims, mask=live, other=0.0).to(tl.float32)
            output = tl.load(out_ptr + q_row + dims, mask=live, other=0.0)
            stat = (position * SLOTS + slot) * HEADS + head
            row_max = tl.load(m_ptr + stat)
            denom = tl.load(l_ptr + stat)
            score = tl.sum(query * key, axis=0) * scale
            alpha = tl.exp(score - row_max) / denom
            delta = tl.sum(upstream * output, axis=0)
            dalpha = tl.sum(upstream * value, axis=0)
            dscore = alpha * (dalpha - delta)
            acc_k += dscore * query
            acc_v += alpha * upstream
        element = dk_ptr.dtype.element_ty
        tl.store(dk_ptr + kv_row + dims, (acc_k * scale).to(element), mask=live)
        tl.store(dv_ptr + kv_row + dims, acc_v.to(element), mask=live)

    @triton.jit
    def _broadcast_forward_kernel(
        q_ptr, k_ptr, v_ptr, window_pos_ptr, out_ptr, m_ptr, l_ptr, scale,
        SLOTS: tl.constexpr, HEADS: tl.constexpr, HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        window = pid // HEADS
        head = pid % HEADS
        position = tl.load(window_pos_ptr + window)
        dims = tl.arange(0, BLOCK_HD)
        live = dims < HD
        q_row = (window * HEADS + head) * HD
        query = tl.load(q_ptr + q_row + dims, mask=live, other=0.0).to(tl.float32)
        slots = tl.arange(0, SLOTS)
        latent_rows = ((position * SLOTS + slots) * HEADS + head) * HD
        keys = tl.load(
            k_ptr + latent_rows[:, None] + dims[None, :],
            mask=live[None, :], other=0.0,
        ).to(tl.float32)
        scores = tl.sum(query[None, :] * keys, axis=1) * scale
        row_max = tl.max(scores, axis=0)
        numer = tl.exp(scores - row_max)
        denom = tl.sum(numer, axis=0)
        values = tl.load(
            v_ptr + latent_rows[:, None] + dims[None, :],
            mask=live[None, :], other=0.0,
        ).to(tl.float32)
        result = tl.sum(numer[:, None] * values, axis=0) / denom
        tl.store(out_ptr + q_row + dims, result, mask=live)
        tl.store(m_ptr + window * HEADS + head, row_max)
        tl.store(l_ptr + window * HEADS + head, denom)

    @triton.jit
    def _broadcast_dq_kernel(
        q_ptr, k_ptr, v_ptr, window_pos_ptr, out_ptr, m_ptr, l_ptr, go_ptr,
        dq_ptr, scale, SLOTS: tl.constexpr, HEADS: tl.constexpr,
        HD: tl.constexpr, BLOCK_HD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        window = pid // HEADS
        head = pid % HEADS
        position = tl.load(window_pos_ptr + window)
        dims = tl.arange(0, BLOCK_HD)
        live = dims < HD
        q_row = (window * HEADS + head) * HD
        query = tl.load(q_ptr + q_row + dims, mask=live, other=0.0).to(tl.float32)
        upstream = tl.load(go_ptr + q_row + dims, mask=live, other=0.0).to(tl.float32)
        output = tl.load(out_ptr + q_row + dims, mask=live, other=0.0)
        delta = tl.sum(upstream * output, axis=0)
        slots = tl.arange(0, SLOTS)
        latent_rows = ((position * SLOTS + slots) * HEADS + head) * HD
        keys = tl.load(
            k_ptr + latent_rows[:, None] + dims[None, :],
            mask=live[None, :], other=0.0,
        ).to(tl.float32)
        values = tl.load(
            v_ptr + latent_rows[:, None] + dims[None, :],
            mask=live[None, :], other=0.0,
        ).to(tl.float32)
        scores = tl.sum(query[None, :] * keys, axis=1) * scale
        row_max = tl.load(m_ptr + window * HEADS + head)
        denom = tl.load(l_ptr + window * HEADS + head)
        alpha = tl.exp(scores - row_max) / denom
        dalpha = tl.sum(upstream[None, :] * values, axis=1)
        dscore = alpha * (dalpha - delta)
        grad = tl.sum(dscore[:, None] * keys, axis=0) * scale
        element = dq_ptr.dtype.element_ty
        tl.store(dq_ptr + q_row + dims, grad.to(element), mask=live)

    @triton.jit
    def _broadcast_dkdv_kernel(
        q_ptr, k_ptr, v_ptr, offsets_ptr, order_ptr, out_ptr, m_ptr, l_ptr,
        go_ptr, dk_ptr, dv_ptr, scale, SLOTS: tl.constexpr,
        HEADS: tl.constexpr, HD: tl.constexpr, BLOCK_HD: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        pid = tl.program_id(0)
        position = pid // (SLOTS * HEADS)
        rem = pid % (SLOTS * HEADS)
        slot = rem // HEADS
        head = rem % HEADS
        dims = tl.arange(0, BLOCK_HD)
        live = dims < HD
        latent_row = ((position * SLOTS + slot) * HEADS + head) * HD
        key = tl.load(k_ptr + latent_row + dims, mask=live, other=0.0).to(tl.float32)
        value = tl.load(v_ptr + latent_row + dims, mask=live, other=0.0).to(tl.float32)
        start = tl.load(offsets_ptr + position)
        end = tl.load(offsets_ptr + position + 1)
        acc_k = tl.zeros([BLOCK_HD], dtype=tl.float32)
        acc_v = tl.zeros([BLOCK_HD], dtype=tl.float32)
        for lo in tl.range(start, end, BLOCK_W):
            ranks = lo + tl.arange(0, BLOCK_W)
            inside = ranks < end
            windows = tl.load(order_ptr + ranks, mask=inside, other=0)
            q_rows = (windows[:, None] * HEADS + head) * HD + dims[None, :]
            queries = tl.load(
                q_ptr + q_rows, mask=inside[:, None] & live[None, :], other=0.0
            ).to(tl.float32)
            upstream = tl.load(
                go_ptr + q_rows, mask=inside[:, None] & live[None, :], other=0.0
            ).to(tl.float32)
            outputs = tl.load(
                out_ptr + q_rows, mask=inside[:, None] & live[None, :], other=0.0
            )
            row_max = tl.load(
                m_ptr + windows * HEADS + head, mask=inside, other=0.0
            )
            denom = tl.load(
                l_ptr + windows * HEADS + head, mask=inside, other=1.0
            )
            scores = tl.sum(queries * key[None, :], axis=1) * scale
            alpha = tl.where(inside, tl.exp(scores - row_max) / denom, 0.0)
            delta = tl.sum(upstream * outputs, axis=1)
            dalpha = tl.sum(upstream * value[None, :], axis=1)
            dscore = alpha * (dalpha - delta)
            acc_k += tl.sum(dscore[:, None] * queries, axis=0)
            acc_v += tl.sum(alpha[:, None] * upstream, axis=0)
        element = dk_ptr.dtype.element_ty
        tl.store(dk_ptr + latent_row + dims, (acc_k * scale).to(element), mask=live)
        tl.store(dv_ptr + latent_row + dims, acc_v.to(element), mask=live)


def _shape_key(x: Tensor) -> tuple[object, ...]:
    return (x.device.type, x.device.index, x.dtype, *x.shape[-2:])


def _supported(x: Tensor, windows: int) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[-1] > 0
        and windows > 0
    )


def _launch_read_forward(q, k, v, offsets, order):
    positions, slots, heads, hd = q.shape
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    m = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    _read_forward_kernel[(positions * slots * heads,)](
        q, k, v, offsets, order, out, m, l, 1.0 / math.sqrt(hd),
        SLOTS=slots, HEADS=heads, HD=hd, BLOCK_HD=triton.next_power_of_2(hd),
        BLOCK_W=_BLOCK_W, num_warps=_NUM_WARPS,
    )
    return out, m, l


def _launch_read_backward(q, k, v, window_pos, offsets, order, out, m, l, go):
    positions, slots, heads, hd = q.shape
    block_hd = triton.next_power_of_2(hd)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _read_dq_kernel[(positions * slots * heads,)](
        q, k, v, offsets, order, out, m, l, go, dq, 1.0 / math.sqrt(hd),
        SLOTS=slots, HEADS=heads, HD=hd, BLOCK_HD=block_hd, BLOCK_W=_BLOCK_W,
        num_warps=_NUM_WARPS,
    )
    _read_dkdv_kernel[(k.shape[0] * heads,)](
        q, k, v, window_pos, out, m, l, go, dk, dv, 1.0 / math.sqrt(hd),
        SLOTS=slots, HEADS=heads, HD=hd, BLOCK_HD=block_hd,
        num_warps=_NUM_WARPS,
    )
    return dq, dk, dv


def _launch_broadcast_forward(q, k, v, window_pos):
    windows, heads, hd = q.shape
    slots = k.shape[1]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    m = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    _broadcast_forward_kernel[(windows * heads,)](
        q, k, v, window_pos, out, m, l, 1.0 / math.sqrt(hd),
        SLOTS=slots, HEADS=heads, HD=hd, BLOCK_HD=triton.next_power_of_2(hd),
        num_warps=_NUM_WARPS,
    )
    return out, m, l


def _launch_broadcast_backward(q, k, v, window_pos, offsets, order, out, m, l, go):
    windows, heads, hd = q.shape
    positions, slots = k.shape[:2]
    block_hd = triton.next_power_of_2(hd)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _broadcast_dq_kernel[(windows * heads,)](
        q, k, v, window_pos, out, m, l, go, dq, 1.0 / math.sqrt(hd),
        SLOTS=slots, HEADS=heads, HD=hd, BLOCK_HD=block_hd,
        num_warps=_NUM_WARPS,
    )
    _broadcast_dkdv_kernel[(positions * slots * heads,)](
        q, k, v, offsets, order, out, m, l, go, dk, dv, 1.0 / math.sqrt(hd),
        SLOTS=slots, HEADS=heads, HD=hd, BLOCK_HD=block_hd, BLOCK_W=_BLOCK_W,
        num_warps=_NUM_WARPS,
    )
    return dq, dk, dv


@torch.library.custom_op("mantisnet::window_latent_read", mutates_args=())
def _read_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_pos: Tensor,
    offsets: Tensor,
    order: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_read(q, k, v, window_pos, offsets, order)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if not _supported(q, k.shape[0]):
        return _eager_read_forward(q, k, v, offsets, order)
    key = _shape_key(q)
    if key in _FAILED_READ_SHAPES:
        return _eager_read_forward(q, k, v, offsets, order)
    try:
        return _launch_read_forward(q, k, v, offsets, order)
    except Exception as exc:
        _FAILED_READ_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"window latent read failed for {key}; using reference: {_FAILED_READ_SHAPES[key]}",
            RuntimeWarning, stacklevel=2,
        )
        return _eager_read_forward(q, k, v, offsets, order)


@_read_op.register_fake
def _(q, k, v, window_pos, offsets, order):
    return (
        q.new_empty(q.shape, dtype=torch.float32),
        q.new_empty(q.shape[:-1], dtype=torch.float32),
        q.new_empty(q.shape[:-1], dtype=torch.float32),
    )


@torch.library.custom_op("mantisnet::window_latent_read_backward", mutates_args=())
def _read_backward_op(
    q: Tensor, k: Tensor, v: Tensor, window_pos: Tensor, offsets: Tensor,
    order: Tensor, out: Tensor, m: Tensor, l: Tensor, grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    q, k, v, go = q.contiguous(), k.contiguous(), v.contiguous(), grad_out.contiguous().float()
    if not _supported(q, k.shape[0]):
        return _reference_read_backward(q, k, v, offsets, order, out, m, l, go)
    key = _shape_key(q) + (grad_out.dtype,)
    if key in _FAILED_READ_BACKWARD_SHAPES:
        return _reference_read_backward(q, k, v, offsets, order, out, m, l, go)
    try:
        return _launch_read_backward(q, k, v, window_pos, offsets, order, out, m, l, go)
    except Exception as exc:
        _FAILED_READ_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"window latent read backward failed for {key}; using reference: "
            f"{_FAILED_READ_BACKWARD_SHAPES[key]}", RuntimeWarning, stacklevel=2,
        )
        return _reference_read_backward(q, k, v, offsets, order, out, m, l, go)


@_read_backward_op.register_fake
def _(q, k, v, window_pos, offsets, order, out, m, l, grad_out):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _read_setup_context(ctx, inputs, output) -> None:
    q, k, v, window_pos, offsets, order = inputs
    out, m, l = output
    ctx.save_for_backward(q, k, v, window_pos, offsets, order, out, m, l)


def _read_dispatch_backward(ctx, grad_out, _grad_m, _grad_l):
    dq, dk, dv = _read_backward_op(*ctx.saved_tensors, grad_out)
    return dq, dk, dv, None, None, None


_read_op.register_autograd(_read_dispatch_backward, setup_context=_read_setup_context)


@torch.library.custom_op("mantisnet::window_latent_broadcast", mutates_args=())
def _broadcast_op(
    q: Tensor, k: Tensor, v: Tensor, window_pos: Tensor, offsets: Tensor, order: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    _validate_broadcast(q, k, v, window_pos, offsets, order)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if not _supported(q, q.shape[0]):
        return _eager_broadcast_forward(q, k, v, window_pos)
    key = _shape_key(q)
    if key in _FAILED_BROADCAST_SHAPES:
        return _eager_broadcast_forward(q, k, v, window_pos)
    try:
        return _launch_broadcast_forward(q, k, v, window_pos)
    except Exception as exc:
        _FAILED_BROADCAST_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"window latent broadcast failed for {key}; using reference: "
            f"{_FAILED_BROADCAST_SHAPES[key]}", RuntimeWarning, stacklevel=2,
        )
        return _eager_broadcast_forward(q, k, v, window_pos)


@_broadcast_op.register_fake
def _(q, k, v, window_pos, offsets, order):
    return (
        q.new_empty(q.shape, dtype=torch.float32),
        q.new_empty(q.shape[:-1], dtype=torch.float32),
        q.new_empty(q.shape[:-1], dtype=torch.float32),
    )


@torch.library.custom_op("mantisnet::window_latent_broadcast_backward", mutates_args=())
def _broadcast_backward_op(
    q: Tensor, k: Tensor, v: Tensor, window_pos: Tensor, offsets: Tensor,
    order: Tensor, out: Tensor, m: Tensor, l: Tensor, grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    q, k, v, go = q.contiguous(), k.contiguous(), v.contiguous(), grad_out.contiguous().float()
    if not _supported(q, q.shape[0]):
        return _reference_broadcast_backward(
            q, k, v, window_pos, offsets, order, out, m, l, go
        )
    key = _shape_key(q) + (grad_out.dtype,)
    if key in _FAILED_BROADCAST_BACKWARD_SHAPES:
        return _reference_broadcast_backward(
            q, k, v, window_pos, offsets, order, out, m, l, go
        )
    try:
        return _launch_broadcast_backward(
            q, k, v, window_pos, offsets, order, out, m, l, go
        )
    except Exception as exc:
        _FAILED_BROADCAST_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            f"window latent broadcast backward failed for {key}; using reference: "
            f"{_FAILED_BROADCAST_BACKWARD_SHAPES[key]}", RuntimeWarning, stacklevel=2,
        )
        return _reference_broadcast_backward(
            q, k, v, window_pos, offsets, order, out, m, l, go
        )


@_broadcast_backward_op.register_fake
def _(q, k, v, window_pos, offsets, order, out, m, l, grad_out):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _broadcast_setup_context(ctx, inputs, output) -> None:
    q, k, v, window_pos, offsets, order = inputs
    out, m, l = output
    ctx.save_for_backward(q, k, v, window_pos, offsets, order, out, m, l)


def _broadcast_dispatch_backward(ctx, grad_out, _grad_m, _grad_l):
    dq, dk, dv = _broadcast_backward_op(*ctx.saved_tensors, grad_out)
    return dq, dk, dv, None, None, None


_broadcast_op.register_autograd(
    _broadcast_dispatch_backward, setup_context=_broadcast_setup_context
)


def read_attention(q, k, v, window_pos, offsets, order) -> Tensor:
    """Read flat window rows into four per-position latent queries in fp32."""
    out, _m, _l = _read_op(q, k, v, window_pos, offsets, order)
    return out


def broadcast_attention(q, k, v, window_pos, offsets, order) -> Tensor:
    """Broadcast four per-position latents into flat window queries in fp32."""
    out, _m, _l = _broadcast_op(q, k, v, window_pos, offsets, order)
    return out
