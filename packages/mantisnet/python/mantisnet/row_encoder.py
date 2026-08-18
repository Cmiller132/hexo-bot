"""Step 4's fused action-row encoder: the kept-row hidden sum per legal cell.

Each kept action row pairs a window with its post-placement class; the row's
hidden vector is ``relu(pre_w[window] + table[class])`` and a cell's rows sum
in fp32. The row MLP's first layer is folded: ``pre_w`` arrives as one GEMM
over the window rows, and the class side is the table itself, so the per-edge
work is two gathers, an add, a ReLU, and a run reduction. EMPTY rows are not
edges — the caller composes them from per-orbit counts against the shared
empty base.

CUDA: fused run-reduction kernels over the cell-major decoder ordering
(forward), the window-major view (``pre_w`` gradient, ReLU mask recomputed),
and a sliced ``index_add_`` for the class-table gradient. CPU: gather +
``index_add`` parity reference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.library import triton_op, wrap_triton

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_CELL_RUN = 18
_WINDOW_RUN = 6
_RUN_WARPS = 1
_TABLE_GRAD_SLICES = 8


if triton is not None:

    @triton.jit(do_not_specialize_on_alignment=("n_entries",))
    def _row_forward_kernel(
        pre_ptr,
        table_ptr,
        edge_window_ptr,
        edge_class_ptr,
        edge_cell_ptr,
        out_ptr,
        stride_pr: tl.constexpr,
        stride_ph: tl.constexpr,
        stride_tr: tl.constexpr,
        stride_th: tl.constexpr,
        stride_ew: tl.constexpr,
        stride_ec: tl.constexpr,
        stride_ed: tl.constexpr,
        stride_or: tl.constexpr,
        stride_oh: tl.constexpr,
        n_entries,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        RUN_LEN: tl.constexpr,
    ):
        # Each cell-major run head reduces its fused gather/add/ReLU rows,
        # avoiding the literal (E, H) hidden table.
        entry = tl.program_id(0)
        dest = tl.load(edge_cell_ptr + entry * stride_ed)
        previous = tl.load(
            edge_cell_ptr + (entry - 1) * stride_ed,
            mask=entry > 0,
            other=-1,
        )
        run_head = (entry == 0) | (previous != dest)

        offs_h = tl.arange(0, BLOCK_H)
        live_h = offs_h < H
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for relative in tl.static_range(0, RUN_LEN):
            current = entry + relative
            in_bounds = current < n_entries
            current_dest = tl.load(
                edge_cell_ptr + current * stride_ed,
                mask=run_head & in_bounds,
                other=-1,
            )
            same_run = run_head & in_bounds & (current_dest == dest)
            window = tl.load(
                edge_window_ptr + current * stride_ew,
                mask=same_run,
                other=0,
            )
            cls = tl.load(
                edge_class_ptr + current * stride_ec,
                mask=same_run,
                other=0,
            )
            pre = tl.load(
                pre_ptr + window * stride_pr + offs_h * stride_ph,
                mask=same_run & live_h,
                other=0.0,
            ).to(tl.float32)
            table = tl.load(
                table_ptr + cls * stride_tr + offs_h * stride_th,
                mask=same_run & live_h,
                other=0.0,
            ).to(tl.float32)
            acc += tl.maximum(pre + table, 0.0)

        tl.store(
            out_ptr + dest * stride_or + offs_h * stride_oh,
            acc,
            mask=run_head & live_h,
        )

    @triton.jit(do_not_specialize_on_alignment=("n_entries",))
    def _pre_backward_kernel(
        pre_ptr,
        table_ptr,
        edge_window_ptr,
        edge_class_ptr,
        edge_cell_ptr,
        edge_rev_ptr,
        grad_out_ptr,
        grad_pre_ptr,
        stride_pr: tl.constexpr,
        stride_ph: tl.constexpr,
        stride_tr: tl.constexpr,
        stride_th: tl.constexpr,
        stride_ew: tl.constexpr,
        stride_ec: tl.constexpr,
        stride_ed: tl.constexpr,
        stride_er: tl.constexpr,
        stride_gr: tl.constexpr,
        stride_gh: tl.constexpr,
        stride_or: tl.constexpr,
        stride_oh: tl.constexpr,
        n_entries,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        RUN_LEN: tl.constexpr,
    ):
        # ``edge_rev`` exposes bounded window-major runs. Recomputing the
        # ReLU predicate keeps the forward's (E, H) hidden rows out of autograd.
        entry = tl.program_id(0)
        edge = tl.load(edge_rev_ptr + entry * stride_er)
        window = tl.load(edge_window_ptr + edge * stride_ew)
        previous_edge = tl.load(
            edge_rev_ptr + (entry - 1) * stride_er,
            mask=entry > 0,
            other=0,
        )
        previous_window = tl.load(
            edge_window_ptr + previous_edge * stride_ew,
            mask=entry > 0,
            other=-1,
        )
        run_head = (entry == 0) | (previous_window != window)

        offs_h = tl.arange(0, BLOCK_H)
        live_h = offs_h < H
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for relative in tl.static_range(0, RUN_LEN):
            current = entry + relative
            in_bounds = current < n_entries
            current_edge = tl.load(
                edge_rev_ptr + current * stride_er,
                mask=run_head & in_bounds,
                other=0,
            )
            current_window = tl.load(
                edge_window_ptr + current_edge * stride_ew,
                mask=run_head & in_bounds,
                other=-1,
            )
            same_run = run_head & in_bounds & (current_window == window)
            cls = tl.load(
                edge_class_ptr + current_edge * stride_ec,
                mask=same_run,
                other=0,
            )
            cell = tl.load(
                edge_cell_ptr + current_edge * stride_ed,
                mask=same_run,
                other=0,
            )
            pre = tl.load(
                pre_ptr + window * stride_pr + offs_h * stride_ph,
                mask=same_run & live_h,
                other=0.0,
            ).to(tl.float32)
            table = tl.load(
                table_ptr + cls * stride_tr + offs_h * stride_th,
                mask=same_run & live_h,
                other=0.0,
            ).to(tl.float32)
            upstream = tl.load(
                grad_out_ptr + cell * stride_gr + offs_h * stride_gh,
                mask=same_run & live_h,
                other=0.0,
            ).to(tl.float32)
            acc += tl.where(pre + table > 0.0, upstream, 0.0)

        tl.store(
            grad_pre_ptr + window * stride_or + offs_h * stride_oh,
            acc,
            mask=run_head & live_h,
        )


def _encode_reference(
    pre_w: Tensor,
    table: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    edge_cell: Tensor,
    n_cells: int,
) -> Tensor:
    """Literal gather/relu/scatter formulation, fp32 throughout."""
    hidden = F.relu(
        pre_w.float().index_select(0, edge_window)
        + table.float().index_select(0, edge_class)
    )
    out = torch.zeros(
        n_cells, pre_w.shape[1], dtype=torch.float32, device=pre_w.device
    )
    return out.index_add(0, edge_cell, hidden)


def _validate(
    pre_w: Tensor,
    table: Tensor,
    views: tuple[Tensor, ...],
    n_cells: int,
) -> None:
    if pre_w.ndim != 2 or table.ndim != 2 or pre_w.shape[1] != table.shape[1]:
        raise ValueError("pre_w and table must be (N, H) with one width")
    first = views[0]
    if first.ndim != 1 or any(view.shape != first.shape for view in views):
        raise ValueError("the edge views must be one-dimensional and one length")
    if any(view.dtype != torch.int64 for view in views):
        raise ValueError("edge tensors must have dtype int64")
    if (
        any(view.device != pre_w.device for view in views)
        or table.device != pre_w.device
    ):
        raise ValueError("all inputs must be on one device")
    if not pre_w.is_floating_point() or not table.is_floating_point():
        raise ValueError("pre_w and table must have floating-point dtype")
    if n_cells < 0:
        raise ValueError(f"cell count must be nonnegative, got {n_cells}")


def _supported(pre_w: Tensor, table: Tensor, n_cells: int) -> bool:
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    return (
        triton is not None
        and pre_w.is_cuda
        and pre_w.dtype in dtypes
        and table.dtype in dtypes
        and pre_w.shape[1] > 0
        and n_cells > 0
    )


def _launch_forward(
    pre_w: Tensor,
    table: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    edge_cell: Tensor,
    n_cells: int,
) -> Tensor:
    h = pre_w.shape[1]
    n_entries = edge_window.shape[0]
    out = torch.zeros(
        (n_cells, h), dtype=torch.float32, device=pre_w.device
    )
    wrap_triton(_row_forward_kernel)[(n_entries,)](
        pre_w,
        table,
        edge_window,
        edge_class,
        edge_cell,
        out,
        *pre_w.stride(),
        *table.stride(),
        *edge_window.stride(),
        *edge_class.stride(),
        *edge_cell.stride(),
        *out.stride(),
        n_entries,
        H=h,
        BLOCK_H=triton.next_power_of_2(h),
        RUN_LEN=_CELL_RUN,
        num_warps=_RUN_WARPS,
    )
    return out


def _launch_pre_backward(
    pre_w: Tensor,
    table: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    edge_cell: Tensor,
    edge_rev: Tensor,
    grad_out: Tensor,
) -> Tensor:
    h = pre_w.shape[1]
    n_entries = edge_rev.shape[0]
    grad = torch.zeros_like(pre_w)
    wrap_triton(_pre_backward_kernel)[(n_entries,)](
        pre_w,
        table,
        edge_window,
        edge_class,
        edge_cell,
        edge_rev,
        grad_out,
        grad,
        *pre_w.stride(),
        *table.stride(),
        *edge_window.stride(),
        *edge_class.stride(),
        *edge_cell.stride(),
        *edge_rev.stride(),
        *grad_out.stride(),
        *grad.stride(),
        n_entries,
        H=h,
        BLOCK_H=triton.next_power_of_2(h),
        RUN_LEN=_WINDOW_RUN,
        num_warps=_RUN_WARPS,
    )
    return grad


def _table_backward(
    pre_w: Tensor,
    table: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    edge_cell: Tensor,
    grad_out: Tensor,
) -> Tensor:
    # Class runs are unbounded, so fixed slices cap the temporary gathered
    # rows while preserving fp32 accumulation into the whole class table.
    grad = torch.zeros(
        table.shape, dtype=torch.float32, device=table.device
    )
    n = edge_window.shape[0]
    for k in range(_TABLE_GRAD_SLICES):
        piece = slice(
            k * n // _TABLE_GRAD_SLICES,
            (k + 1) * n // _TABLE_GRAD_SLICES,
        )
        windows = edge_window[piece]
        classes = edge_class[piece]
        summed = pre_w.float().index_select(0, windows)
        summed = summed + table.float().index_select(0, classes)
        upstream = grad_out.index_select(0, edge_cell[piece]).float()
        grad.index_add_(0, classes, upstream * (summed > 0.0))
    return grad.to(table.dtype)


if triton is not None:

    @triton_op("mantisnet::action_row_encode", mutates_args={})
    def _encode_rows_op(
        pre_w: Tensor,
        table: Tensor,
        edge_window: Tensor,
        edge_class: Tensor,
        edge_cell: Tensor,
        edge_rev: Tensor,
        n_cells: int,
    ) -> Tensor:
        return _launch_forward(
            pre_w, table, edge_window, edge_class, edge_cell, n_cells
        )

else:  # pragma: no cover - exercised only by installations without Triton
    _encode_rows_op = None


def _setup_context(ctx, inputs, output) -> None:
    del output
    pre_w, table, edge_window, edge_class, edge_cell, edge_rev, _n_cells = inputs
    ctx.save_for_backward(
        pre_w, table, edge_window, edge_class, edge_cell, edge_rev
    )


def _backward(ctx, grad_out: Tensor):
    pre_w, table, edge_window, edge_class, edge_cell, edge_rev = ctx.saved_tensors
    grad_out = grad_out.contiguous()
    grad_pre = _launch_pre_backward(
        pre_w,
        table,
        edge_window,
        edge_class,
        edge_cell,
        edge_rev,
        grad_out,
    )
    grad_table = _table_backward(
        pre_w, table, edge_window, edge_class, edge_cell, grad_out
    )
    return grad_pre, grad_table, None, None, None, None, None


if _encode_rows_op is not None:
    _encode_rows_op.register_autograd(_backward, setup_context=_setup_context)


def encode_rows(
    pre_w: Tensor,
    table: Tensor,
    edge_window: Tensor,
    edge_class: Tensor,
    edge_cell: Tensor,
    edge_rev: Tensor,
    n_cells: int,
) -> Tensor:
    """Sum ``relu(pre_w[window] + table[class])`` rows per cell, fp32.

    ``(edge_window, edge_class, edge_cell)`` are the batch's decoder-order
    views (cell-major runs of at most 18); ``edge_rev`` is the collation-built
    window-major permutation the backward reduces over (runs of at most 6).
    Cells without kept rows are zero, exactly the literal formulation.
    """
    _validate(pre_w, table, (edge_window, edge_class, edge_cell, edge_rev), n_cells)
    # Empty launches are invalid in Triton. The literal path also handles CPU
    # and floating signatures outside the three training dtypes.
    if not _supported(pre_w, table, n_cells) or edge_window.numel() == 0:
        return _encode_reference(
            pre_w, table, edge_window, edge_class, edge_cell, n_cells
        )
    return _encode_rows_op(
        pre_w,
        table,
        edge_window,
        edge_class,
        edge_cell,
        edge_rev,
        n_cells,
    )
