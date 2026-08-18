"""The section 5.1b cell pass: windows exchange state through shared empty cells.

Computes over the decoder incidence (cell, window, class):

    pre[c]  = Σ_e x[window(e)] + e_class[class(e)]     over the cell's entries
    val[c]  = ReLU(pre[c])
    agg[w]  = Σ_e val[cell(e)]                         over the window's entries

with fp32 accumulation, cast back to ``x``'s dtype.  ``relay_tables`` sorts
the incidence into three views (by cell, by window, by class) for contiguous
segment reductions.  The class gradient is sliced across programs because
class runs can be large.  The backward recomputes ``pre`` for the ReLU mask.
CPU: torch composition over the same tables as the parity reference.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


# One program and one warp per output row: segments are short (a cell has at
# most 18 entries, a window at most 5) and fixed geometry keeps symbolic shape
# changes out of Triton's tuning cache.
_NUM_WARPS = 1

# Class runs are the hostile layout — tens of thousands of edges can share a
# class, so ``d_emb`` slices every run across programs, each summing 32-row
# tiles into its own partial. 64 slices keep the worst run to a few hundred
# iterations while the (classes * 64, H) fp32 partial stays a few MB.
_CLASS_SPLITS = 64
_CLASS_BLOCK_E = 32
_CLASS_NUM_WARPS = 4

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}


def relay_tables(
    dec_cell: Tensor,
    dec_window: Tensor,
    dec_class: Tensor,
    n_windows: int,
    n_classes: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Sort one batch's decoder incidence into the three CSR views.

    Returns ``(cell_ptr, edge_window, edge_class, win_ptr, edge_wcell,
    cls_ptr, edge_ccell)``. Cells are relabelled compactly in first-seen
    sorted order — the pass never needs the original cell identity, only the
    run structure — so ``cell_ptr`` spans exactly the covered cells.
    """
    if not (dec_cell.shape == dec_window.shape == dec_class.shape):
        raise ValueError("dec_cell, dec_window, and dec_class must be one length")
    if dec_cell.ndim != 1:
        raise ValueError("decoder incidence arrays must be one-dimensional")
    if dec_window.numel():
        if int(dec_window.min()) < 0 or int(dec_window.max()) >= n_windows:
            raise ValueError(
                f"dec_window entries must lie in [0, {n_windows})"
            )
        if int(dec_class.min()) < 0 or int(dec_class.max()) >= n_classes:
            raise ValueError(f"dec_class entries must lie in [0, {n_classes})")

    order = torch.argsort(dec_cell, stable=True)
    window = dec_window[order]
    cls = dec_class[order]
    _, counts = torch.unique_consecutive(dec_cell[order], return_counts=True)
    cell_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    edge_cell = torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.long), counts
    )

    worder = torch.argsort(window, stable=True)
    win_ptr = torch.searchsorted(
        window[worder], torch.arange(n_windows + 1, dtype=window.dtype)
    )
    edge_wcell = edge_cell[worder]

    corder = torch.argsort(cls, stable=True)
    cls_ptr = torch.searchsorted(
        cls[corder], torch.arange(n_classes + 1, dtype=cls.dtype)
    )
    edge_ccell = edge_cell[corder]
    return cell_ptr, window, cls, win_ptr, edge_wcell, cls_ptr, edge_ccell


if triton is not None:

    @triton.jit
    def _cell_values_kernel(
        x_ptr,
        emb_ptr,
        cell_ptr,
        edge_window,
        edge_class,
        val_ptr,
        stride_xw,
        stride_xh,
        stride_e0,
        stride_e1,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        cell = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        live = offs < H
        start = tl.load(cell_ptr + cell)
        end = tl.load(cell_ptr + cell + 1)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for entry in tl.range(start, end):
            window = tl.load(edge_window + entry)
            acc += tl.load(
                x_ptr + window * stride_xw + offs * stride_xh, mask=live, other=0.0
            ).to(tl.float32)
            cls = tl.load(edge_class + entry)
            acc += tl.load(
                emb_ptr + cls * stride_e0 + offs * stride_e1, mask=live, other=0.0
            ).to(tl.float32)
        acc = tl.maximum(acc, 0.0)
        tl.store(val_ptr + cell * H + offs, acc, mask=live)

    @triton.jit
    def _cell_grad_kernel(
        x_ptr,
        emb_ptr,
        gout_ptr,
        cell_ptr,
        edge_window,
        edge_class,
        dpre_ptr,
        stride_xw,
        stride_xh,
        stride_e0,
        stride_e1,
        stride_gw,
        stride_gh,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        # d_val[c] is the sum of the consuming windows' output gradients — the
        # same run that built pre[c] — so one loop recomputes pre for the ReLU
        # mask and accumulates the gradient together.
        cell = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        live = offs < H
        start = tl.load(cell_ptr + cell)
        end = tl.load(cell_ptr + cell + 1)
        pre = tl.zeros([BLOCK_H], dtype=tl.float32)
        gacc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for entry in tl.range(start, end):
            window = tl.load(edge_window + entry)
            pre += tl.load(
                x_ptr + window * stride_xw + offs * stride_xh, mask=live, other=0.0
            ).to(tl.float32)
            cls = tl.load(edge_class + entry)
            pre += tl.load(
                emb_ptr + cls * stride_e0 + offs * stride_e1, mask=live, other=0.0
            ).to(tl.float32)
            gacc += tl.load(
                gout_ptr + window * stride_gw + offs * stride_gh, mask=live, other=0.0
            ).to(tl.float32)
        dpre = tl.where(pre > 0, gacc, 0.0)
        tl.store(dpre_ptr + cell * H + offs, dpre, mask=live)

    @triton.jit
    def _class_partial_kernel(
        rows_ptr,
        seg_ptr,
        payload,
        partial_ptr,
        SPLITS: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # One slice of one class run: sum the selected fp32 rows of the slice
        # into a partial. Slice bounds and the in-slice order are functions of
        # the tables alone, so the two-stage reduction stays deterministic.
        cls = tl.program_id(0)
        part = tl.program_id(1)
        offs = tl.arange(0, BLOCK_H)
        live = offs < H
        start = tl.load(seg_ptr + cls)
        end = tl.load(seg_ptr + cls + 1)
        per = (end - start + SPLITS - 1) // SPLITS
        lo = start + part * per
        hi = tl.minimum(lo + per, end)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for base in tl.range(lo, hi, BLOCK_E):
            entries = base + tl.arange(0, BLOCK_E)
            inside = entries < hi
            rows = tl.load(payload + entries, mask=inside, other=0)
            acc += tl.sum(
                tl.load(
                    rows_ptr + rows[:, None] * H + offs[None, :],
                    mask=inside[:, None] & live[None, :],
                    other=0.0,
                ),
                axis=0,
            )
        tl.store(partial_ptr + (cls * SPLITS + part) * H + offs, acc, mask=live)

    @triton.jit
    def _segment_sum_kernel(
        rows_ptr,
        seg_ptr,
        payload,
        out_ptr,
        stride_ow,
        stride_oh,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        # Generic run-sum of fp32 rows selected by ``payload``: the window
        # aggregation, ``dx``, and ``d_emb`` are all this shape.
        seg = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        live = offs < H
        start = tl.load(seg_ptr + seg)
        end = tl.load(seg_ptr + seg + 1)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for entry in tl.range(start, end):
            row = tl.load(payload + entry)
            acc += tl.load(rows_ptr + row * H + offs, mask=live, other=0.0)
        element = out_ptr.dtype.element_ty
        tl.store(
            out_ptr + seg * stride_ow + offs * stride_oh, acc.to(element), mask=live
        )


def _edge_cells(cell_ptr: Tensor) -> Tensor:
    counts = cell_ptr[1:] - cell_ptr[:-1]
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=cell_ptr.device), counts
    )


def _reference(
    x: Tensor,
    emb: Tensor,
    cell_ptr: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    win_ptr: Tensor,
    edge_wcell: Tensor,
) -> Tensor:
    """The scatter formulation used by CPU tensors, failed launches, and the
    parity tests. Accumulation is fp32 whatever ``x``'s dtype, matching the
    kernel registers."""
    h = x.shape[1]
    edge_cell = _edge_cells(cell_ptr)
    msg = x.float().index_select(0, edge_window) + emb.float().index_select(
        0, edge_class
    )
    pre = torch.zeros(
        cell_ptr.numel() - 1, h, dtype=torch.float32, device=x.device
    ).index_add_(0, edge_cell, msg)
    val = torch.relu(pre)
    wcounts = win_ptr[1:] - win_ptr[:-1]
    edge_win = torch.repeat_interleave(
        torch.arange(x.shape[0], device=x.device), wcounts
    )
    agg = torch.zeros(
        x.shape[0], h, dtype=torch.float32, device=x.device
    ).index_add_(0, edge_win, val.index_select(0, edge_wcell))
    return agg.to(x.dtype)


def _reference_backward(
    x: Tensor,
    emb: Tensor,
    cell_ptr: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    h = x.shape[1]
    edge_cell = _edge_cells(cell_ptr)
    n_relay = cell_ptr.numel() - 1
    msg = x.float().index_select(0, edge_window) + emb.float().index_select(
        0, edge_class
    )
    pre = torch.zeros(
        n_relay, h, dtype=torch.float32, device=x.device
    ).index_add_(0, edge_cell, msg)
    d_val = torch.zeros(
        n_relay, h, dtype=torch.float32, device=x.device
    ).index_add_(0, edge_cell, grad_out.float().index_select(0, edge_window))
    d_pre = d_val * (pre > 0)
    reached = d_pre.index_select(0, edge_cell)
    dx = torch.zeros(
        x.shape, dtype=torch.float32, device=x.device
    ).index_add_(0, edge_window, reached)
    d_emb = torch.zeros(
        emb.shape, dtype=torch.float32, device=x.device
    ).index_add_(0, edge_class, reached)
    return dx.to(x.dtype), d_emb.to(emb.dtype)


def _validate(
    x: Tensor,
    emb: Tensor,
    cell_ptr: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    win_ptr: Tensor,
    edge_wcell: Tensor,
    cls_ptr: Tensor,
    edge_ccell: Tensor,
) -> None:
    if x.ndim != 2:
        raise ValueError("x must have shape (N_w, H)")
    if emb.ndim != 2 or emb.shape[1] != x.shape[1]:
        raise ValueError("emb must have shape (n_classes, H) matching x")
    edges = edge_window.shape
    if edge_class.shape != edges or edge_wcell.shape != edges or edge_ccell.shape != edges:
        raise ValueError("the three edge views must have one length")
    if win_ptr.numel() != x.shape[0] + 1:
        raise ValueError("win_ptr must have one entry per window plus one")
    if cls_ptr.numel() != emb.shape[0] + 1:
        raise ValueError("cls_ptr must have one entry per class plus one")
    tables = (cell_ptr, edge_window, edge_class, win_ptr, edge_wcell, cls_ptr, edge_ccell)
    if any(t.device != x.device for t in tables) or emb.device != x.device:
        raise ValueError("all cell-pass inputs must be on one device")


def _shape_key(x: Tensor) -> tuple[object, ...]:
    return (x.device.type, x.device.index, x.dtype, x.shape[1], tuple(x.stride()))


def _launch_forward(
    x: Tensor,
    emb: Tensor,
    cell_ptr: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    win_ptr: Tensor,
    edge_wcell: Tensor,
) -> Tensor:
    h = x.shape[1]
    n_relay = cell_ptr.numel() - 1
    block_h = triton.next_power_of_2(h)
    val = torch.empty(n_relay, h, dtype=torch.float32, device=x.device)
    _cell_values_kernel[(n_relay,)](
        x,
        emb,
        cell_ptr,
        edge_window,
        edge_class,
        val,
        *x.stride(),
        *emb.stride(),
        H=h,
        BLOCK_H=block_h,
        num_warps=_NUM_WARPS,
    )
    out = torch.empty_like(x)
    _segment_sum_kernel[(x.shape[0],)](
        val,
        win_ptr,
        edge_wcell,
        out,
        *out.stride(),
        H=h,
        BLOCK_H=block_h,
        num_warps=_NUM_WARPS,
    )
    return out


def _launch_backward(
    x: Tensor,
    emb: Tensor,
    cell_ptr: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    win_ptr: Tensor,
    edge_wcell: Tensor,
    cls_ptr: Tensor,
    edge_ccell: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    h = x.shape[1]
    n_relay = cell_ptr.numel() - 1
    block_h = triton.next_power_of_2(h)
    grad_out = grad_out.contiguous()
    d_pre = torch.empty(n_relay, h, dtype=torch.float32, device=x.device)
    _cell_grad_kernel[(n_relay,)](
        x,
        emb,
        grad_out,
        cell_ptr,
        edge_window,
        edge_class,
        d_pre,
        *x.stride(),
        *emb.stride(),
        *grad_out.stride(),
        H=h,
        BLOCK_H=block_h,
        num_warps=_NUM_WARPS,
    )
    dx = torch.empty_like(x)
    _segment_sum_kernel[(x.shape[0],)](
        d_pre,
        win_ptr,
        edge_wcell,
        dx,
        *dx.stride(),
        H=h,
        BLOCK_H=block_h,
        num_warps=_NUM_WARPS,
    )
    partial = torch.empty(
        emb.shape[0] * _CLASS_SPLITS, h, dtype=torch.float32, device=x.device
    )
    _class_partial_kernel[(emb.shape[0], _CLASS_SPLITS)](
        d_pre,
        cls_ptr,
        edge_ccell,
        partial,
        SPLITS=_CLASS_SPLITS,
        H=h,
        BLOCK_H=block_h,
        BLOCK_E=_CLASS_BLOCK_E,
        num_warps=_CLASS_NUM_WARPS,
    )
    d_emb = partial.view(emb.shape[0], _CLASS_SPLITS, h).sum(dim=1).to(emb.dtype)
    return dx, d_emb


def _supported(x: Tensor, cell_ptr: Tensor) -> bool:
    return (
        triton is not None
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and cell_ptr.numel() > 1
        and x.shape[0] > 0
    )


@torch.library.custom_op("mantisnet::cell_pass", mutates_args=())
def _cell_pass_op(
    x: Tensor,
    emb: Tensor,
    cell_ptr: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    win_ptr: Tensor,
    edge_wcell: Tensor,
    cls_ptr: Tensor,
    edge_ccell: Tensor,
) -> Tensor:
    _validate(
        x, emb, cell_ptr, edge_window, edge_class, win_ptr, edge_wcell, cls_ptr, edge_ccell
    )
    if not _supported(x, cell_ptr):
        return _reference(x, emb, cell_ptr, edge_window, edge_class, win_ptr, edge_wcell)
    key = _shape_key(x)
    if key in _FAILED_SHAPES:
        return _reference(x, emb, cell_ptr, edge_window, edge_class, win_ptr, edge_wcell)
    try:
        return _launch_forward(
            x, emb, cell_ptr, edge_window, edge_class, win_ptr, edge_wcell
        )
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "cell pass failed for "
            f"H={x.shape[1]}, dtype={x.dtype}; scattering instead for this "
            f"shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _reference(x, emb, cell_ptr, edge_window, edge_class, win_ptr, edge_wcell)


@_cell_pass_op.register_fake
def _(
    x: Tensor,
    emb: Tensor,
    cell_ptr: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    win_ptr: Tensor,
    edge_wcell: Tensor,
    cls_ptr: Tensor,
    edge_ccell: Tensor,
) -> Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("mantisnet::cell_pass_backward", mutates_args=())
def _cell_pass_backward_op(
    x: Tensor,
    emb: Tensor,
    cell_ptr: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    win_ptr: Tensor,
    edge_wcell: Tensor,
    cls_ptr: Tensor,
    edge_ccell: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    if not _supported(x, cell_ptr):
        return _reference_backward(x, emb, cell_ptr, edge_window, edge_class, grad_out)
    key = _shape_key(x) + (grad_out.dtype,)
    if key in _FAILED_BACKWARD_SHAPES:
        return _reference_backward(x, emb, cell_ptr, edge_window, edge_class, grad_out)
    try:
        return _launch_backward(
            x,
            emb,
            cell_ptr,
            edge_window,
            edge_class,
            win_ptr,
            edge_wcell,
            cls_ptr,
            edge_ccell,
            grad_out,
        )
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "cell pass backward failed for "
            f"H={x.shape[1]}, dtype={x.dtype}; scattering instead for this "
            f"shape: {_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _reference_backward(x, emb, cell_ptr, edge_window, edge_class, grad_out)


@_cell_pass_backward_op.register_fake
def _(
    x: Tensor,
    emb: Tensor,
    cell_ptr: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    win_ptr: Tensor,
    edge_wcell: Tensor,
    cls_ptr: Tensor,
    edge_ccell: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x), torch.empty_like(emb)


def _setup_context(ctx, inputs, output) -> None:
    (x, emb, cell_ptr, edge_window, edge_class, win_ptr, edge_wcell, cls_ptr, edge_ccell) = inputs
    ctx.save_for_backward(
        x, emb, cell_ptr, edge_window, edge_class, win_ptr, edge_wcell, cls_ptr, edge_ccell
    )


def _dispatch_backward(ctx, grad_out: Tensor):
    dx, d_emb = _cell_pass_backward_op(*ctx.saved_tensors, grad_out)
    return dx, d_emb, None, None, None, None, None, None, None


_cell_pass_op.register_autograd(_dispatch_backward, setup_context=_setup_context)


def cell_pass(
    x: Tensor,
    emb: Tensor,
    cell_ptr: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    win_ptr: Tensor,
    edge_wcell: Tensor,
    cls_ptr: Tensor,
    edge_ccell: Tensor,
) -> Tensor:
    """The (N_w, H) window aggregate of §5.1b over one batch's relay tables."""
    return _cell_pass_op(
        x, emb, cell_ptr, edge_window, edge_class, win_ptr, edge_wcell, cls_ptr, edge_ccell
    )
