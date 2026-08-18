"""Fused incidence aggregation for the two trunk message-passing passes.

Both directions consume the builder's window-major incidence table and
aggregate projected entity rows.  The class terms run-reduce class-table
rows per destination (``class_row_sum``); ``incidence_plan`` derives the
stone-major view outside autograd.  The stone direction accumulates in fp32.

CUDA: one segment-reduction kernel serving all four directions over
window-major and stone-major orderings.  CPU: gather + ``index_add_`` parity
reference.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


# Run bounds for the orderings.  A window has at most six slots of either
# kind (occupied slots for the trunk incidence, empty cells for the decoder
# incidence), and a stone or cell sits in at most eighteen windows: six span
# offsets on each of the three axes.  The same two bounds therefore serve the
# trunk views, the mixed-scope class sums, and the decoder incidence.
WINDOW_RUN = 6
STONE_RUN = 18
_RUN_WARPS = 1

if triton is not None:

    @triton.jit(do_not_specialize_on_alignment=("n_entries",))
    def _run_reduce_kernel(
        values_ptr,
        gather_ptr,
        run_ptr,
        out_ptr,
        stride_vr: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_ig: tl.constexpr,
        stride_ir: tl.constexpr,
        stride_or: tl.constexpr,
        stride_oh: tl.constexpr,
        n_entries,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        RUN_LEN: tl.constexpr,
    ):
        # ``run_ptr`` holds the sorted destination per entry; a program at a
        # run head gathers its run's ``values`` rows (by ``gather_ptr``) and
        # reduces them into the destination row without materializing the
        # (E, H) message table.
        entry = tl.program_id(0)
        dest = tl.load(run_ptr + entry * stride_ir)
        previous = tl.load(
            run_ptr + (entry - 1) * stride_ir,
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
                run_ptr + current * stride_ir,
                mask=run_head & in_bounds,
                other=-1,
            )
            same_run = run_head & in_bounds & (current_dest == dest)
            source = tl.load(
                gather_ptr + current * stride_ig,
                mask=same_run,
                other=0,
            )
            value = tl.load(
                values_ptr + source * stride_vr + offs_h * stride_vh,
                mask=same_run & live_h,
                other=0.0,
            ).to(tl.float32)
            acc += value

        tl.store(
            out_ptr + dest * stride_or + offs_h * stride_oh,
            acc,
            mask=run_head & live_h,
        )


def _aggregate_reference(
    values: Tensor,
    source_index: Tensor,
    dest_index: Tensor,
    n_dest: int,
    dtype: torch.dtype,
) -> Tensor:
    """Literal gather/scatter formulation for either direction.

    Accumulation is fp32 with one cast at the output boundary, matching the
    run kernel's fp32 accumulator and the stone direction's fp32 contract.
    """
    messages = values.index_select(0, source_index).float()
    out = messages.new_zeros((n_dest, values.shape[1])).index_add_(
        0, dest_index, messages
    )
    return out.to(dtype)


def _validate(
    values: Tensor,
    inc_stone: Tensor,
    inc_window: Tensor,
    run_stone: Tensor,
    run_window: Tensor,
    n_dest: int,
) -> None:
    if values.ndim != 2:
        raise ValueError("values must have shape (N, H)")
    entries = inc_stone.shape
    indices = (inc_stone, inc_window, run_stone, run_window)
    if inc_stone.ndim != 1 or any(index.shape != entries for index in indices):
        raise ValueError("the incidence views must be one length")
    if any(index.dtype != torch.int64 for index in indices):
        raise ValueError("incidence tensors must have dtype int64")
    if any(index.device != values.device for index in indices):
        raise ValueError("all incidence inputs must be on the values device")
    if not values.is_floating_point():
        raise ValueError("values must have floating-point dtype")
    if n_dest < 0:
        raise ValueError(f"destination count must be nonnegative, got {n_dest}")


def _supported(values: Tensor, n_dest: int) -> bool:
    return (
        triton is not None
        and values.is_cuda
        and values.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and values.shape[1] > 0
        and n_dest > 0
    )


def _launch_runs(
    values: Tensor,
    gather: Tensor,
    runs: Tensor,
    n_dest: int,
    run_len: int,
    dtype: torch.dtype,
    fill_zero: bool,
) -> Tensor:
    """One run-reduction launch.  ``fill_zero`` covers destinations that may
    own no entries (a stone whose candidate windows are all dead); window
    destinations are live by construction, so their launches skip the fill."""
    h = values.shape[1]
    n_entries = gather.shape[0]
    empty = torch.zeros if fill_zero else torch.empty
    out = empty((n_dest, h), dtype=dtype, device=values.device)
    wrap_triton(_run_reduce_kernel)[(n_entries,)](
        values,
        gather,
        runs,
        out,
        *values.stride(),
        *gather.stride(),
        *runs.stride(),
        *out.stride(),
        n_entries,
        H=h,
        BLOCK_H=triton.next_power_of_2(h),
        RUN_LEN=run_len,
        num_warps=_RUN_WARPS,
    )
    return out


if triton is not None:

    @triton_op("mantisnet::aggregate_to_windows", mutates_args={})
    def _aggregate_to_windows_op(
        values: Tensor,
        inc_stone: Tensor,
        inc_window: Tensor,
        run_stone: Tensor,
        run_window: Tensor,
        n_windows: int,
    ) -> Tensor:
        return _launch_runs(
            values,
            inc_stone,
            inc_window,
            n_windows,
            WINDOW_RUN,
            values.dtype,
            fill_zero=False,
        )

    @triton_op("mantisnet::aggregate_to_stones", mutates_args={})
    def _aggregate_to_stones_op(
        values: Tensor,
        inc_stone: Tensor,
        inc_window: Tensor,
        run_stone: Tensor,
        run_window: Tensor,
        n_stones: int,
    ) -> Tensor:
        return _launch_runs(
            values,
            run_window,
            run_stone,
            n_stones,
            STONE_RUN,
            torch.float32,
            fill_zero=True,
        )

else:  # pragma: no cover - exercised only by installations without Triton
    _aggregate_to_windows_op = None
    _aggregate_to_stones_op = None


def _setup_context(ctx, inputs, output) -> None:
    del output
    values, inc_stone, inc_window, run_stone, run_window, _n_dest = inputs
    ctx.save_for_backward(inc_stone, inc_window, run_stone, run_window)
    ctx.n_source = values.shape[0]
    ctx.values_dtype = values.dtype


def _backward_to_windows(ctx, grad_out: Tensor):
    # A stone row's gradient sums its entries' upstream window rows — the
    # stone-major run reduction.
    _inc_stone, _inc_window, run_stone, run_window = ctx.saved_tensors
    grad = _launch_runs(
        grad_out.contiguous(),
        run_window,
        run_stone,
        ctx.n_source,
        STONE_RUN,
        ctx.values_dtype,
        fill_zero=True,
    )
    return grad, None, None, None, None, None


def _backward_to_stones(ctx, grad_out: Tensor):
    # A window row's gradient sums its entries' upstream stone rows — the
    # window-major run reduction, on the builder's own ordering.
    inc_stone, inc_window, _run_stone, _run_window = ctx.saved_tensors
    grad = _launch_runs(
        grad_out.contiguous(),
        inc_stone,
        inc_window,
        ctx.n_source,
        WINDOW_RUN,
        ctx.values_dtype,
        fill_zero=False,
    )
    return grad, None, None, None, None, None


if _aggregate_to_windows_op is not None:
    _aggregate_to_windows_op.register_autograd(
        _backward_to_windows, setup_context=_setup_context
    )
    _aggregate_to_stones_op.register_autograd(
        _backward_to_stones, setup_context=_setup_context
    )


def aggregate_to_windows(
    values: Tensor,
    inc_stone: Tensor,
    inc_window: Tensor,
    run_stone: Tensor,
    run_window: Tensor,
    n_windows: int,
) -> Tensor:
    """Sum projected stone rows into live-window rows."""
    _validate(values, inc_stone, inc_window, run_stone, run_window, n_windows)
    # An empty launch is invalid in Triton.  The reference also covers every
    # non-CUDA or unsupported signature.
    if not _supported(values, n_windows) or inc_stone.numel() == 0:
        return _aggregate_reference(
            values, inc_stone, inc_window, n_windows, values.dtype
        )
    return _aggregate_to_windows_op(
        values, inc_stone, inc_window, run_stone, run_window, n_windows
    )


def aggregate_to_stones(
    values: Tensor,
    inc_stone: Tensor,
    inc_window: Tensor,
    run_stone: Tensor,
    run_window: Tensor,
    n_stones: int,
) -> Tensor:
    """Sum projected window rows into stone rows, accumulating in fp32."""
    _validate(values, inc_stone, inc_window, run_stone, run_window, n_stones)
    if not _supported(values, n_stones) or inc_stone.numel() == 0:
        return _aggregate_reference(
            values, inc_window, inc_stone, n_stones, torch.float32
        )
    return _aggregate_to_stones_op(
        values, inc_stone, inc_window, run_stone, run_window, n_stones
    )


# The ternary class vocabularies (726/1458) are too wide
# for the histogram matmul, so its class terms and its decoder aggregation
# reduce rows per run with the same kernel as the trunk aggregations — never
# materializing the (E, H) gather of the literal formulation.

_CLASS_GRAD_SLICES = 8


def _row_sum_reference(
    table: Tensor, gather: Tensor, runs: Tensor, n_dest: int
) -> Tensor:
    """Literal gather/scatter row sum: the pre-kernel formulation."""
    rows = table.index_select(0, gather).float()
    out = torch.zeros(
        n_dest, table.shape[1], dtype=torch.float32, device=table.device
    )
    return out.index_add_(0, runs, rows)


if triton is not None:

    @triton_op("mantisnet::class_row_sum", mutates_args={})
    def _class_row_sum_op(
        weight: Tensor, gather: Tensor, runs: Tensor, n_dest: int, run_len: int
    ) -> Tensor:
        return _launch_runs(
            weight, gather, runs, n_dest, run_len, torch.float32, fill_zero=True
        )

    @triton_op("mantisnet::incidence_row_sum", mutates_args={})
    def _incidence_row_sum_op(
        values: Tensor,
        gather: Tensor,
        runs: Tensor,
        rev_gather: Tensor,
        rev_runs: Tensor,
        n_dest: int,
        run_len: int,
        rev_run_len: int,
    ) -> Tensor:
        return _launch_runs(
            values, gather, runs, n_dest, run_len, torch.float32, fill_zero=True
        )

else:  # pragma: no cover - exercised only by installations without Triton
    _class_row_sum_op = None
    _incidence_row_sum_op = None


def _class_setup_context(ctx, inputs, output) -> None:
    del output
    weight, gather, runs, _n_dest, _run_len = inputs
    ctx.save_for_backward(gather, runs)
    ctx.n_classes = weight.shape[0]
    ctx.weight_dtype = weight.dtype


def _class_row_sum_backward(ctx, grad_out: Tensor):
    # The table gradient reduces by class, and class runs are unbounded — one
    # class can own most of a chunk's entries.  Fixed slices keep the gathered
    # upstream rows a small fraction of the (E, H) table the forward avoids.
    gather, runs = ctx.saved_tensors
    grad_out = grad_out.contiguous()
    grad = torch.zeros(
        ctx.n_classes, grad_out.shape[1], dtype=torch.float32, device=grad_out.device
    )
    n = gather.shape[0]
    for k in range(_CLASS_GRAD_SLICES):
        piece = slice(k * n // _CLASS_GRAD_SLICES, (k + 1) * n // _CLASS_GRAD_SLICES)
        grad.index_add_(
            0, gather[piece], grad_out.index_select(0, runs[piece]).float()
        )
    return grad.to(ctx.weight_dtype), None, None, None, None


def _incidence_setup_context(ctx, inputs, output) -> None:
    del output
    values, _gather, _runs, rev_gather, rev_runs, _n_dest, _run_len, rev_run_len = (
        inputs
    )
    ctx.save_for_backward(rev_gather, rev_runs)
    ctx.n_source = values.shape[0]
    ctx.values_dtype = values.dtype
    ctx.rev_run_len = rev_run_len


def _incidence_row_sum_backward(ctx, grad_out: Tensor):
    # A source row's gradient sums its entries' upstream rows — the reverse
    # ordering's run reduction.
    rev_gather, rev_runs = ctx.saved_tensors
    grad = _launch_runs(
        grad_out.contiguous(),
        rev_gather,
        rev_runs,
        ctx.n_source,
        ctx.rev_run_len,
        ctx.values_dtype,
        fill_zero=True,
    )
    return grad, None, None, None, None, None, None, None


if _class_row_sum_op is not None:
    _class_row_sum_op.register_autograd(
        _class_row_sum_backward, setup_context=_class_setup_context
    )
    _incidence_row_sum_op.register_autograd(
        _incidence_row_sum_backward, setup_context=_incidence_setup_context
    )


def _validate_views(entries: tuple[Tensor, ...], n_dest: int) -> None:
    first = entries[0]
    if first.ndim != 1 or any(view.shape != first.shape for view in entries):
        raise ValueError("the incidence views must be one-dimensional and one length")
    if any(view.dtype != torch.int64 for view in entries):
        raise ValueError("incidence tensors must have dtype int64")
    if n_dest < 0:
        raise ValueError(f"destination count must be nonnegative, got {n_dest}")


def class_row_sum(
    weight: Tensor, gather: Tensor, runs: Tensor, n_dest: int, run_len: int
) -> Tensor:
    """Sum class-table rows into destination rows, accumulating in fp32.

    ``gather`` holds each entry's class row and ``runs`` its destination,
    sorted so equal destinations are adjacent in runs of at most ``run_len``.
    Destinations without entries are zero and the output is fp32 for the
    caller to cast — both exactly the literal zeros/``index_add_`` form.
    """
    _validate_views((gather, runs), n_dest)
    if not _supported(weight, n_dest) or gather.numel() == 0:
        return _row_sum_reference(weight, gather, runs, n_dest)
    return _class_row_sum_op(weight, gather, runs, n_dest, run_len)


def incidence_row_sum(
    values: Tensor,
    gather: Tensor,
    runs: Tensor,
    rev_gather: Tensor,
    rev_runs: Tensor,
    n_dest: int,
    run_len: int,
    rev_run_len: int,
) -> Tensor:
    """Sum gathered ``values`` rows into destination rows, fp32 accumulated.

    ``(gather, runs)`` order the incidence by destination; ``(rev_gather,
    rev_runs)`` hold the same entries ordered by source, which the backward
    run-reduces to build the ``values`` gradient without an (E, H)
    intermediate.  Both orderings bound their runs (``run_len``,
    ``rev_run_len``).  Destinations without entries are zero and the output
    is fp32, exactly the literal zeros/``index_add_`` form.
    """
    _validate_views((gather, runs, rev_gather, rev_runs), n_dest)
    if not _supported(values, n_dest) or gather.numel() == 0:
        return _row_sum_reference(values, gather, runs, n_dest)
    return _incidence_row_sum_op(
        values, gather, runs, rev_gather, rev_runs, n_dest, run_len, rev_run_len
    )


class IncidencePlan(NamedTuple):
    """Per-batch derivations both passes share: the stone-major incidence view."""

    run_stone: Tensor  # (E,) int64: inc_stone, stone-major stable order
    run_window: Tensor  # (E,) int64: inc_window in the same order
    run_class: Tensor  # (E,) int64: inc_class in the same order


def incidence_plan(
    inc_stone: Tensor,
    inc_window: Tensor,
    inc_class: Tensor,
) -> IncidencePlan:
    """The stone-major incidence view both passes share.

    The class terms run-reduce their table rows per destination
    (``class_row_sum``), so the plan carries the class ids alongside the
    reordered incidence.  The plan is data, not activations: it carries no
    gradient, and one derivation per batch serves every block's two passes.
    """
    if inc_class.shape != inc_stone.shape or inc_class.dtype != torch.int64:
        raise ValueError("inc_class must be int64 and one length with inc_stone")
    with torch.no_grad():
        # Stone ids fit int32, and radix passes scale with key width; the
        # stable sort keeps each run in the builder's window-major order.
        order = torch.argsort(inc_stone.to(torch.int32), stable=True)
        return IncidencePlan(
            inc_stone.index_select(0, order),
            inc_window.index_select(0, order),
            inc_class.index_select(0, order),
        )
