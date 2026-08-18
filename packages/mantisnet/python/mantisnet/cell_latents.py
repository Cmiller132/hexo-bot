"""Step 15 cell latents: typed attention over the decoder incidence, and the
per-line window tables.

Every window↔window interaction the game defines — fork, block, shared
line — is mediated by a cell, so pair interaction can factor through cells
at O(incidence) instead of O(pairs). Two mechanisms live here:

- **Cell attention.** Legal cells covered by the decoder incidence carry a
  persistent state vector through the trunk. Per block, a cell updates by
  attention over the windows containing it (at most 18 entries), and each
  window reads back from its at most 5 empty cells. Both directions are
  typed by the 726 joint decoder classes as a per-head score bias; the
  cell-update direction additionally adds class value rows, which is what
  absorbs the §5.1b relay's typed message content. The read-back carries no
  value rows: the cell latent already composed them.

- **Line tables.** Windows grouped per (position, axis, line), every
  intra-line pair an edge at unbounded offset: the colinear vocabulary of
  §5.1c exactly (|offset| − 1 for offsets 1..11), an out-of-reach FAR
  class, and SELF. The views are ``window_pairs.PairTables``, so the line
  pass runs on ``window_pairs.edge_attention`` unchanged.

Both derivations run on-device from batch tensors, as §5.1c's tables do:
the int64 edge views cost more to ship over PCIe than to derive beside the
model, and every block shares one derivation.

Kernels follow the relay's geometry — bounded fan-in, one program per
(row, head), contiguous segment sweeps, class gradients as fixed split
partials — so every reduction is deterministic. CPU and failed launches
fall back to an eager fp32 composition that doubles as the parity
reference.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import torch
from torch import Tensor

from . import window_pairs

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


# Line-pass relation classes: |offset| - 1 for offsets 1..11 (the §5.1c
# colinear vocabulary), FAR for offsets 12 and beyond, SELF loops.
LINE_CLASSES = 13
LINE_FAR = 11
LINE_SELF = 12

# Key packing, as §5.1c: coordinates are i16-bounded, so 17 bits per
# component after an offset is collision-free.
_KOFF = 1 << 16
_KSPAN = 1 << 17

# Segments are at most 18 entries (a cell's windows) or line runs of a few
# dozen, so one 32-edge tile usually covers a run in one iteration.
_BLOCK_E = 32
_NUM_WARPS = 1

# Class runs are the hostile layout — the split-partial shape of the relay
# and §5.1c bias gradients. The class-value gradient gathers full H-wide
# rows per edge, so its edge tile is small to keep the (BLOCK_E, H) load
# tile within register budget.
_CLASS_SPLITS = 64
_CLASS_BLOCK_E = 128
_VCLS_BLOCK_E = 8

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}


class CellTables(NamedTuple):
    """One batch's covered-cell incidence in query-, source-, and class-major
    views, both directions."""

    covered: Tensor  # (N_cov,) legal-cell index of each covered cell
    cell_ptr: Tensor  # (N_cov + 1,) cell-major run starts
    edge_window: Tensor  # (E,) window per entry, cell order (canonical)
    edge_class: Tensor  # (E,) class per entry, cell order
    edge_cell: Tensor  # (E,) covered-cell per entry, cell order
    win_ptr: Tensor  # (N_w + 1,) window-major run starts
    edge_wcell: Tensor  # (E,) covered-cell per entry, window order
    edge_wclass: Tensor  # (E,) class per entry, window order
    edge_wwin: Tensor  # (E,) window per entry, window order
    cls_ptr: Tensor  # (n_classes + 1,) class-major run starts
    cedge_cell: Tensor  # (E,) cell-order entry ids, class order
    cedge_win: Tensor  # (E,) window-order entry ids, class order


def cell_tables(
    dec_cell: Tensor,
    dec_window: Tensor,
    dec_class: Tensor,
    n_windows: int,
    n_classes: int,
    n_cells: int = -1,
) -> CellTables:
    """Sort one batch's decoder incidence into the cell-attention views.

    Covered cells are relabelled compactly in first-seen sorted order, and —
    unlike the relay, which never needs it — the original legal-cell index of
    each covered cell is kept, so refined latents can scatter back into the
    decoder's legal order.
    """
    if not (dec_cell.shape == dec_window.shape == dec_class.shape):
        raise ValueError("dec_cell, dec_window, and dec_class must be one length")
    if dec_cell.ndim != 1:
        raise ValueError("decoder incidence arrays must be one-dimensional")
    if dec_window.numel():
        if int(dec_window.min()) < 0 or int(dec_window.max()) >= n_windows:
            raise ValueError(f"dec_window entries must lie in [0, {n_windows})")
        if int(dec_class.min()) < 0 or int(dec_class.max()) >= n_classes:
            raise ValueError(f"dec_class entries must lie in [0, {n_classes})")
    device = dec_cell.device

    order = torch.argsort(dec_cell, stable=True)
    edge_window = dec_window[order]
    edge_class = dec_class[order]
    covered, counts = torch.unique_consecutive(dec_cell[order], return_counts=True)
    if n_cells < 0:
        cell_ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        edge_cell = torch.repeat_interleave(
            torch.arange(counts.numel(), device=device), counts
        )
    else:
        if dec_cell.numel() and (int(dec_cell.min()) < 0 or int(dec_cell.max()) >= n_cells):
            raise ValueError(f"dec_cell entries must lie in [0, {n_cells})")
        cell_ptr = torch.searchsorted(
            dec_cell[order], torch.arange(n_cells + 1, device=device)
        )
        edge_cell = dec_cell[order]

    worder = torch.argsort(edge_window, stable=True)
    win_ptr = torch.searchsorted(
        edge_window[worder], torch.arange(n_windows + 1, device=device)
    )
    edge_wcell = edge_cell[worder]
    edge_wclass = edge_class[worder]
    edge_wwin = edge_window[worder]

    cedge_cell = torch.argsort(edge_class.to(torch.int32), stable=True)
    cedge_win = torch.argsort(edge_wclass.to(torch.int32), stable=True)
    cls_ptr = torch.searchsorted(
        edge_class[cedge_cell], torch.arange(n_classes + 1, device=device)
    )
    return CellTables(
        covered,
        cell_ptr,
        edge_window,
        edge_class,
        edge_cell,
        win_ptr,
        edge_wcell,
        edge_wclass,
        edge_wwin,
        cls_ptr,
        cedge_cell,
        cedge_win,
    )


@torch.library.custom_op("mantisnet::cell_tables", mutates_args=())
def derive_cell_tables(
    dec_cell: Tensor,
    dec_window: Tensor,
    dec_class: Tensor,
    n_windows: int,
    n_classes: int,
    n_cells: int,
) -> tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
]:
    """``cell_tables`` as an opaque op: the sorts are data-dependent and would
    otherwise break the surrounding graph out of compilation."""
    return tuple(
        cell_tables(dec_cell, dec_window, dec_class, n_windows, n_classes, n_cells)
    )


@derive_cell_tables.register_fake
def _(dec_cell, dec_window, dec_class, n_windows, n_classes, n_cells):
    ctx = torch.library.get_ctx()
    n_cov = ctx.new_dynamic_size()
    e = dec_cell.shape[0]

    def edges():
        return dec_cell.new_empty((e,), dtype=torch.long)

    return (
        dec_cell.new_empty((n_cov,), dtype=torch.long),
        dec_cell.new_empty((n_cov + 1 if n_cells < 0 else n_cells + 1,), dtype=torch.long),
        edges(),
        edges(),
        edges(),
        dec_cell.new_empty((n_windows + 1,), dtype=torch.long),
        edges(),
        edges(),
        edges(),
        dec_cell.new_empty((n_classes + 1,), dtype=torch.long),
        edges(),
        edges(),
    )


def line_tables(window_id: Tensor, window_pos: Tensor) -> window_pairs.PairTables:
    """Every intra-line directed window pair, in §5.1c's three CSR views.

    Same grouping as the §5.1c colinear join — (position, axis, line),
    ordered by position-on-line — without the offset cutoff: within a group
    every ordered pair is an edge. Line classes are symmetric under
    reversal, so the source-major view shares the destination view's arrays
    outright.
    """
    if window_id.ndim != 2 or window_id.shape[1] != 3:
        raise ValueError("window_id must have shape (N_w, 3)")
    if window_pos.shape != window_id.shape[:1]:
        raise ValueError("window_pos must have one entry per window")
    n_w = window_id.shape[0]
    device = window_id.device
    axis, sq, sr = window_id.unbind(1)

    line = torch.where(axis == 0, sr, torch.where(axis == 1, sq, sq + sr))
    pos_on = torch.where(axis == 1, sr, sq)
    group = (window_pos * 4 + axis) * _KSPAN + (line + _KOFF)
    key = group * _KSPAN + (pos_on + _KOFF)
    order = torch.argsort(key)
    sgroup = group[order]
    spos = pos_on[order]

    dsts, srcs, classes = [], [], []
    if n_w:
        slot = torch.arange(n_w, device=device)
        starts = torch.ones(n_w, dtype=torch.bool, device=device)
        starts[1:] = sgroup[1:] != sgroup[:-1]
        run = starts.cumsum(0) - 1
        run_end = run.new_empty(n_w).scatter_reduce_(
            0, run, slot + 1, "amax", include_self=False
        )[run]
        first, second = window_pairs._segment_pairs(run_end - slot - 1, slot)
        offset = spos.index_select(0, second) - spos.index_select(0, first)
        cls = torch.where(
            offset <= LINE_FAR, offset - 1, torch.full_like(offset, LINE_FAR)
        )
        near = order.index_select(0, first)
        far = order.index_select(0, second)
        dsts.append(torch.cat([near, far]))
        srcs.append(torch.cat([far, near]))
        classes.append(torch.cat([cls, cls]))

    loop = torch.arange(n_w, device=device)
    dsts.append(loop)
    srcs.append(loop)
    classes.append(torch.full((n_w,), LINE_SELF, dtype=torch.long, device=device))

    dst = torch.cat(dsts)
    src = torch.cat(srcs)
    cls = torch.cat(classes)
    order = torch.argsort(dst.to(torch.int32), stable=True)
    dst, src, cls = dst[order], src[order], cls[order]
    ptr = torch.searchsorted(dst, torch.arange(n_w + 1, device=device))
    corder = torch.argsort(cls.to(torch.int32), stable=True)
    cptr = torch.searchsorted(
        cls[corder], torch.arange(LINE_CLASSES + 1, device=device)
    )
    return window_pairs.PairTables(ptr, src, cls, ptr, src, cls, cptr, corder)


@torch.library.custom_op("mantisnet::line_tables", mutates_args=())
def derive_line_tables(
    window_id: Tensor, window_pos: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``line_tables``' distinct tensors as an opaque op; the symmetric
    source view is reassembled by the caller."""
    tables = line_tables(window_id, window_pos)
    return (tables.ptr, tables.src, tables.cls, tables.cptr, tables.cedge)


@derive_line_tables.register_fake
def _(window_id, window_pos):
    n_w = window_id.shape[0]
    e = torch.library.get_ctx().new_dynamic_size()

    def edges():
        return window_id.new_empty((e,), dtype=torch.long)

    return (
        window_id.new_empty((n_w + 1,), dtype=torch.long),
        edges(),
        edges(),
        window_id.new_empty((LINE_CLASSES + 1,), dtype=torch.long),
        edges(),
    )


if triton is not None:

    @triton.jit
    def _ca_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        vcls_ptr,
        seg_ptr,
        src_ptr,
        cls_ptr,
        out_ptr,
        m_ptr,
        l_ptr,
        scale,
        stride_bias,
        HAS_VCLS: tl.constexpr,
        HEADS: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # One (query row, head) per program: online softmax over the query's
        # entry run. Head programs of one row are consecutive so their
        # gathers of the same source rows share cache lines. Empty runs
        # (a full window has no empty cells) store zero and l = 0.
        pid = tl.program_id(0)
        w = pid // HEADS
        a = pid % HEADS
        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        row = (w * HEADS + a) * HD
        q_row = tl.load(q_ptr + row + offs, mask=live, other=0.0).to(tl.float32)
        start = tl.load(seg_ptr + w)
        end = tl.load(seg_ptr + w + 1)
        m = -float("inf")
        l = 0.0
        acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
        for lo in tl.range(start, end, BLOCK_E):
            eids = lo + tl.arange(0, BLOCK_E)
            ok = eids < end
            s_idx = tl.load(src_ptr + eids, mask=ok, other=0)
            c_idx = tl.load(cls_ptr + eids, mask=ok, other=0)
            k_tile = tl.load(
                k_ptr + (s_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q_row[None, :] * k_tile, axis=1) * scale
            score += tl.load(
                bias_ptr + a * stride_bias + c_idx, mask=ok, other=0.0
            ).to(tl.float32)
            score = tl.where(ok, score, -float("inf"))
            m_new = tl.maximum(m, tl.max(score, axis=0))
            rescale = tl.exp(m - m_new)
            p = tl.exp(score - m_new)
            l = l * rescale + tl.sum(p, axis=0)
            v_tile = tl.load(
                v_ptr + (s_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            if HAS_VCLS:
                v_tile += tl.load(
                    vcls_ptr + (c_idx[:, None] * HEADS + a) * HD + offs[None, :],
                    mask=ok[:, None] & live[None, :],
                    other=0.0,
                ).to(tl.float32)
            acc = acc * rescale + tl.sum(p[:, None] * v_tile, axis=0)
            m = m_new
        denom = tl.where(l > 0, l, 1.0)
        tl.store(out_ptr + row + offs, acc / denom, mask=live)
        tl.store(m_ptr + w * HEADS + a, m)
        tl.store(l_ptr + w * HEADS + a, l)

    @triton.jit
    def _ca_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        vcls_ptr,
        go_ptr,
        seg_ptr,
        src_ptr,
        cls_ptr,
        m_ptr,
        l_ptr,
        delta_ptr,
        dq_ptr,
        dscore_ptr,
        alpha_ptr,
        scale,
        stride_bias,
        HAS_VCLS: tl.constexpr,
        HEADS: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # Query sweep: recompute alpha from the saved stats, emit per-edge
        # dscore (for the bias gradient) and alpha (for the class value
        # gradient), accumulate dq in registers. Deterministic, no atomics.
        pid = tl.program_id(0)
        w = pid // HEADS
        a = pid % HEADS
        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        row = (w * HEADS + a) * HD
        q_row = tl.load(q_ptr + row + offs, mask=live, other=0.0).to(tl.float32)
        go_row = tl.load(go_ptr + row + offs, mask=live, other=0.0).to(tl.float32)
        m = tl.load(m_ptr + w * HEADS + a)
        l = tl.load(l_ptr + w * HEADS + a)
        l = tl.where(l > 0, l, 1.0)
        delta = tl.load(delta_ptr + w * HEADS + a)
        start = tl.load(seg_ptr + w)
        end = tl.load(seg_ptr + w + 1)
        acc = tl.zeros([BLOCK_HD], dtype=tl.float32)
        for lo in tl.range(start, end, BLOCK_E):
            eids = lo + tl.arange(0, BLOCK_E)
            ok = eids < end
            s_idx = tl.load(src_ptr + eids, mask=ok, other=0)
            c_idx = tl.load(cls_ptr + eids, mask=ok, other=0)
            k_tile = tl.load(
                k_ptr + (s_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(q_row[None, :] * k_tile, axis=1) * scale
            score += tl.load(
                bias_ptr + a * stride_bias + c_idx, mask=ok, other=0.0
            ).to(tl.float32)
            alpha = tl.exp(score - m) / l
            v_tile = tl.load(
                v_ptr + (s_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            if HAS_VCLS:
                v_tile += tl.load(
                    vcls_ptr + (c_idx[:, None] * HEADS + a) * HD + offs[None, :],
                    mask=ok[:, None] & live[None, :],
                    other=0.0,
                ).to(tl.float32)
            dalpha = tl.sum(go_row[None, :] * v_tile, axis=1)
            ds = tl.where(ok, alpha * (dalpha - delta), 0.0)
            tl.store(dscore_ptr + eids * HEADS + a, ds, mask=ok)
            tl.store(
                alpha_ptr + eids * HEADS + a, tl.where(ok, alpha, 0.0), mask=ok
            )
            acc += tl.sum(ds[:, None] * k_tile, axis=0)
        element = dq_ptr.dtype.element_ty
        tl.store(dq_ptr + row + offs, (acc * scale).to(element), mask=live)

    @triton.jit
    def _ca_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        bias_ptr,
        vcls_ptr,
        go_ptr,
        sseg_ptr,
        qid_ptr,
        scls_ptr,
        m_ptr,
        l_ptr,
        delta_ptr,
        dk_ptr,
        dv_ptr,
        scale,
        stride_bias,
        HAS_VCLS: tl.constexpr,
        HEADS: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_HD: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # Source sweep over the source-major view: this program's k and v
        # rows are fixed; the queries' rows and stats are gathered per edge
        # and alpha is recomputed, exactly the §5.1c dkdv discipline.
        pid = tl.program_id(0)
        s = pid // HEADS
        a = pid % HEADS
        offs = tl.arange(0, BLOCK_HD)
        live = offs < HD
        row = (s * HEADS + a) * HD
        k_row = tl.load(k_ptr + row + offs, mask=live, other=0.0).to(tl.float32)
        v_row = tl.load(v_ptr + row + offs, mask=live, other=0.0).to(tl.float32)
        start = tl.load(sseg_ptr + s)
        end = tl.load(sseg_ptr + s + 1)
        acc_k = tl.zeros([BLOCK_HD], dtype=tl.float32)
        acc_v = tl.zeros([BLOCK_HD], dtype=tl.float32)
        for lo in tl.range(start, end, BLOCK_E):
            eids = lo + tl.arange(0, BLOCK_E)
            ok = eids < end
            q_idx = tl.load(qid_ptr + eids, mask=ok, other=0)
            c_idx = tl.load(scls_ptr + eids, mask=ok, other=0)
            q_tile = tl.load(
                q_ptr + (q_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            go_tile = tl.load(
                go_ptr + (q_idx[:, None] * HEADS + a) * HD + offs[None, :],
                mask=ok[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            m_e = tl.load(m_ptr + q_idx * HEADS + a, mask=ok, other=0.0)
            l_e = tl.load(l_ptr + q_idx * HEADS + a, mask=ok, other=1.0)
            l_e = tl.where(l_e > 0, l_e, 1.0)
            delta_e = tl.load(delta_ptr + q_idx * HEADS + a, mask=ok, other=0.0)
            score = tl.sum(q_tile * k_row[None, :], axis=1) * scale
            score += tl.load(
                bias_ptr + a * stride_bias + c_idx, mask=ok, other=0.0
            ).to(tl.float32)
            alpha = tl.where(ok, tl.exp(score - m_e) / l_e, 0.0)
            v_edge = v_row[None, :]
            if HAS_VCLS:
                v_edge = v_edge + tl.load(
                    vcls_ptr + (c_idx[:, None] * HEADS + a) * HD + offs[None, :],
                    mask=ok[:, None] & live[None, :],
                    other=0.0,
                ).to(tl.float32)
            dalpha = tl.sum(go_tile * v_edge, axis=1)
            ds = alpha * (dalpha - delta_e)
            acc_k += tl.sum(ds[:, None] * q_tile, axis=0)
            acc_v += tl.sum(alpha[:, None] * go_tile, axis=0)
        element = dk_ptr.dtype.element_ty
        tl.store(dk_ptr + row + offs, (acc_k * scale).to(element), mask=live)
        tl.store(dv_ptr + row + offs, acc_v.to(element), mask=live)

    @triton.jit
    def _ca_bias_partial_kernel(
        dscore_ptr,
        cptr_ptr,
        cedge_ptr,
        partial_ptr,
        SPLITS: tl.constexpr,
        HEADS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # One fixed slice of one class run summed into its own partial.
        cls = tl.program_id(0)
        part = tl.program_id(1)
        offs = tl.arange(0, BLOCK_H)
        live = offs < HEADS
        start = tl.load(cptr_ptr + cls)
        end = tl.load(cptr_ptr + cls + 1)
        per = (end - start + SPLITS - 1) // SPLITS
        lo = start + part * per
        hi = tl.minimum(lo + per, end)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for base in tl.range(lo, hi, BLOCK_E):
            entries = base + tl.arange(0, BLOCK_E)
            inside = entries < hi
            rows = tl.load(cedge_ptr + entries, mask=inside, other=0)
            acc += tl.sum(
                tl.load(
                    dscore_ptr + rows[:, None] * HEADS + offs[None, :],
                    mask=inside[:, None] & live[None, :],
                    other=0.0,
                ),
                axis=0,
            )
        tl.store(partial_ptr + (cls * SPLITS + part) * HEADS + offs, acc, mask=live)

    @triton.jit
    def _ca_vcls_partial_kernel(
        alpha_ptr,
        go_ptr,
        qid_ptr,
        cptr_ptr,
        cedge_ptr,
        partial_ptr,
        SPLITS: tl.constexpr,
        HEADS: tl.constexpr,
        HD: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        # dvclass[c] = sum over the class run of alpha_e * go[query_e]: the
        # same split-partial shape with a per-edge row gather, alpha
        # broadcast across its head's lanes.
        cls = tl.program_id(0)
        part = tl.program_id(1)
        offs = tl.arange(0, BLOCK_H)
        h_total = HEADS * HD
        live = offs < h_total
        head_of = offs // HD
        start = tl.load(cptr_ptr + cls)
        end = tl.load(cptr_ptr + cls + 1)
        per = (end - start + SPLITS - 1) // SPLITS
        lo = start + part * per
        hi = tl.minimum(lo + per, end)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for base in tl.range(lo, hi, BLOCK_E):
            entries = base + tl.arange(0, BLOCK_E)
            inside = entries < hi
            rows = tl.load(cedge_ptr + entries, mask=inside, other=0)
            q_idx = tl.load(qid_ptr + rows, mask=inside, other=0)
            alpha = tl.load(
                alpha_ptr + rows[:, None] * HEADS + head_of[None, :],
                mask=inside[:, None] & live[None, :],
                other=0.0,
            )
            go_tile = tl.load(
                go_ptr + q_idx[:, None] * h_total + offs[None, :],
                mask=inside[:, None] & live[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(alpha * go_tile, axis=0)
        tl.store(
            partial_ptr + (cls * SPLITS + part) * h_total + offs, acc, mask=live
        )


def _reference_forward(q, k, v, bias, vcls, src, cls, qid):
    """Eager fp32 composition: the parity reference and the CPU path."""
    n_q, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    e = src.shape[0]

    score = bias.t().float().index_select(0, cls)  # (E, heads)
    score += scale * (
        q.float().index_select(0, qid) * k.float().index_select(0, src)
    ).sum(-1)
    m = score.new_full((n_q, heads), torch.finfo(torch.float32).min)
    m.index_reduce_(0, qid, score, "amax", include_self=True)
    alpha = (score - m.index_select(0, qid)).exp_()
    l = score.new_zeros((n_q, heads)).index_add_(0, qid, alpha)

    v_edge = v.float().index_select(0, src)
    if vcls is not None:
        v_edge = v_edge + vcls.float().view(-1, heads, hd).index_select(0, cls)
    out = q.new_zeros((n_q, heads, hd), dtype=torch.float32)
    out.index_add_(0, qid, alpha.unsqueeze(-1) * v_edge)
    out /= torch.where(l > 0, l, torch.ones_like(l)).unsqueeze(-1)
    return out, m, l


def _reference_backward(q, k, v, bias, vcls, src, cls, qid, m, l, delta, go):
    n_q, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)

    score = bias.t().float().index_select(0, cls)
    score += scale * (
        q.float().index_select(0, qid) * k.float().index_select(0, src)
    ).sum(-1)
    alpha = (score - m.index_select(0, qid)).exp_()
    alpha /= torch.where(l > 0, l, torch.ones_like(l)).index_select(0, qid)

    v_edge = v.float().index_select(0, src)
    if vcls is not None:
        v_edge = v_edge + vcls.float().view(-1, heads, hd).index_select(0, cls)
    go_rows = go.index_select(0, qid)
    dalpha = (go_rows * v_edge).sum(-1)
    dscore = alpha * (dalpha - delta.index_select(0, qid))

    dbias = torch.zeros(
        (bias.shape[1], heads), dtype=torch.float32, device=bias.device
    ).index_add_(0, cls, dscore).t()

    weighted = alpha.unsqueeze(-1) * go_rows
    dv = torch.zeros((k.shape[0], heads, hd), dtype=torch.float32, device=q.device)
    dv.index_add_(0, src, weighted)
    dvcls = None
    if vcls is not None:
        dvcls = torch.zeros(
            (vcls.shape[0], heads * hd), dtype=torch.float32, device=q.device
        ).index_add_(0, cls, weighted.reshape(-1, heads * hd))

    dq = torch.zeros((n_q, heads, hd), dtype=torch.float32, device=q.device)
    dq.index_add_(
        0, qid, scale * dscore.unsqueeze(-1) * k.float().index_select(0, src)
    )
    dk = torch.zeros((k.shape[0], heads, hd), dtype=torch.float32, device=q.device)
    dk.index_add_(
        0, src, scale * dscore.unsqueeze(-1) * q.float().index_select(0, qid)
    )
    return (
        dq.to(q.dtype),
        dk.to(k.dtype),
        dv.to(v.dtype),
        dbias.contiguous().to(bias.dtype),
        dvcls.to(vcls.dtype) if dvcls is not None else None,
    )


def _validate(q, k, v, bias, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls):
    if q.ndim != 3 or k.shape != v.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError("q and k/v must share (heads, head_dim)")
    if bias.ndim != 2 or bias.shape[0] != q.shape[1]:
        raise ValueError("bias must have shape (heads, classes)")
    if seg_ptr.numel() != q.shape[0] + 1:
        raise ValueError("seg_ptr must have one entry per query row plus one")
    if sseg_ptr.numel() != k.shape[0] + 1:
        raise ValueError("sseg_ptr must have one entry per source row plus one")
    edges = src.shape
    if any(t.shape != edges for t in (cls, qid, sqid, scls)) or src.ndim != 1:
        raise ValueError("the edge views must have one length")


def _shape_key(x: Tensor) -> tuple[object, ...]:
    return (x.device.type, x.device.index, x.dtype, x.shape[1], x.shape[2])


def _supported(q: Tensor, src: Tensor) -> bool:
    return (
        triton is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and q.shape[0] > 0
        and src.numel() > 0
    )


def _launch_forward(q, k, v, bias, vcls, seg_ptr, src, cls):
    n_q, heads, hd = q.shape
    out = torch.empty((n_q, heads, hd), dtype=torch.float32, device=q.device)
    m = torch.empty((n_q, heads), dtype=torch.float32, device=q.device)
    l = torch.empty((n_q, heads), dtype=torch.float32, device=q.device)
    _ca_forward_kernel[(n_q * heads,)](
        q,
        k,
        v,
        bias,
        vcls if vcls is not None else q,
        seg_ptr,
        src,
        cls,
        out,
        m,
        l,
        1.0 / math.sqrt(hd),
        bias.stride(0),
        HAS_VCLS=vcls is not None,
        HEADS=heads,
        HD=hd,
        BLOCK_HD=triton.next_power_of_2(hd),
        BLOCK_E=_BLOCK_E,
        num_warps=_NUM_WARPS,
    )
    return out, m, l


def _launch_backward(
    q, k, v, bias, vcls, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q, m, l, delta, go
):
    n_q, heads, hd = q.shape
    scale = 1.0 / math.sqrt(hd)
    block_hd = triton.next_power_of_2(hd)
    e = src.shape[0]
    dq = torch.empty_like(q)
    dscore = torch.empty((e, heads), dtype=torch.float32, device=q.device)
    alpha = torch.empty((e, heads), dtype=torch.float32, device=q.device)
    vcls_arg = vcls if vcls is not None else q
    _ca_dq_kernel[(n_q * heads,)](
        q,
        k,
        v,
        bias,
        vcls_arg,
        go,
        seg_ptr,
        src,
        cls,
        m,
        l,
        delta,
        dq,
        dscore,
        alpha,
        scale,
        bias.stride(0),
        HAS_VCLS=vcls is not None,
        HEADS=heads,
        HD=hd,
        BLOCK_HD=block_hd,
        BLOCK_E=_BLOCK_E,
        num_warps=_NUM_WARPS,
    )
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _ca_dkdv_kernel[(k.shape[0] * heads,)](
        q,
        k,
        v,
        bias,
        vcls_arg,
        go,
        sseg_ptr,
        sqid,
        scls,
        m,
        l,
        delta,
        dk,
        dv,
        scale,
        bias.stride(0),
        HAS_VCLS=vcls is not None,
        HEADS=heads,
        HD=hd,
        BLOCK_HD=block_hd,
        BLOCK_E=_BLOCK_E,
        num_warps=_NUM_WARPS,
    )
    classes = bias.shape[1]
    partial = torch.empty(
        (classes * _CLASS_SPLITS, heads), dtype=torch.float32, device=q.device
    )
    _ca_bias_partial_kernel[(classes, _CLASS_SPLITS)](
        dscore,
        cptr,
        cedge_q,
        partial,
        SPLITS=_CLASS_SPLITS,
        HEADS=heads,
        BLOCK_H=triton.next_power_of_2(heads),
        BLOCK_E=_CLASS_BLOCK_E,
        num_warps=_NUM_WARPS,
    )
    dbias = (
        partial.view(classes, _CLASS_SPLITS, heads).sum(dim=1).t().contiguous()
    ).to(bias.dtype)
    dvcls = None
    if vcls is not None:
        h_total = heads * hd
        vpartial = torch.empty(
            (classes * _CLASS_SPLITS, h_total), dtype=torch.float32, device=q.device
        )
        _ca_vcls_partial_kernel[(classes, _CLASS_SPLITS)](
            alpha,
            go,
            qid,
            cptr,
            cedge_q,
            vpartial,
            SPLITS=_CLASS_SPLITS,
            HEADS=heads,
            HD=hd,
            BLOCK_H=triton.next_power_of_2(h_total),
            BLOCK_E=_VCLS_BLOCK_E,
            num_warps=4,
        )
        dvcls = (
            vpartial.view(classes, _CLASS_SPLITS, h_total).sum(dim=1)
        ).to(vcls.dtype)
    return dq, dk, dv, dbias, dvcls


def _forward_impl(q, k, v, bias, vcls, seg_ptr, src, cls, qid):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    if not _supported(q, src):
        return _reference_forward(q, k, v, bias, vcls, src, cls, qid)
    key = _shape_key(q) + (vcls is not None,)
    if key in _FAILED_SHAPES:
        return _reference_forward(q, k, v, bias, vcls, src, cls, qid)
    try:
        return _launch_forward(q, k, v, bias, vcls, seg_ptr, src, cls)
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "cell attention failed for "
            f"heads={q.shape[1]}, hd={q.shape[2]}, dtype={q.dtype}; composing "
            f"instead for this shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _reference_forward(q, k, v, bias, vcls, src, cls, qid)


def _backward_impl(
    q, k, v, bias, vcls, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q, out, m, l, grad_out
):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    go = grad_out.contiguous().float()
    # delta = sum(go * out) per (query, head) — algebraically the segment's
    # sum of alpha * dalpha, so the softmax backward needs no first sweep.
    delta = (go * out).sum(-1)
    if not _supported(q, src):
        return _reference_backward(
            q, k, v, bias, vcls, src, cls, qid, m, l, delta, go
        )
    key = _shape_key(q) + (vcls is not None, grad_out.dtype)
    if key in _FAILED_BACKWARD_SHAPES:
        return _reference_backward(
            q, k, v, bias, vcls, src, cls, qid, m, l, delta, go
        )
    try:
        return _launch_backward(
            q, k, v, bias, vcls, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q, m, l, delta, go
        )
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "cell attention backward failed for "
            f"heads={q.shape[1]}, hd={q.shape[2]}, dtype={q.dtype}; composing "
            f"instead for this shape: {_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _reference_backward(
            q, k, v, bias, vcls, src, cls, qid, m, l, delta, go
        )


@torch.library.custom_op("mantisnet::cell_attention", mutates_args=())
def _ca_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    vcls: Tensor,
    seg_ptr: Tensor,
    src: Tensor,
    cls: Tensor,
    qid: Tensor,
    sseg_ptr: Tensor,
    sqid: Tensor,
    scls: Tensor,
    cptr: Tensor,
    cedge_q: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Typed segment attention with class value rows (the cell-update
    direction)."""
    _validate(q, k, v, bias, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls)
    if vcls.shape != (bias.shape[1], q.shape[1] * q.shape[2]):
        raise ValueError("vcls must have shape (classes, heads * head_dim)")
    return _forward_impl(q, k, v, bias, vcls, seg_ptr, src, cls, qid)


@_ca_op.register_fake
def _(q, k, v, bias, vcls, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q):
    n_q, heads, hd = q.shape
    return (
        q.new_empty((n_q, heads, hd), dtype=torch.float32),
        q.new_empty((n_q, heads), dtype=torch.float32),
        q.new_empty((n_q, heads), dtype=torch.float32),
    )


@torch.library.custom_op("mantisnet::cell_attention_backward", mutates_args=())
def _ca_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    vcls: Tensor,
    seg_ptr: Tensor,
    src: Tensor,
    cls: Tensor,
    qid: Tensor,
    sseg_ptr: Tensor,
    sqid: Tensor,
    scls: Tensor,
    cptr: Tensor,
    cedge_q: Tensor,
    out: Tensor,
    m: Tensor,
    l: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dq, dk, dv, dbias, dvcls = _backward_impl(
        q, k, v, bias, vcls, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q, out, m, l, grad_out
    )
    return dq, dk, dv, dbias, dvcls


@_ca_backward_op.register_fake
def _(q, k, v, bias, vcls, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q, out, m, l, grad_out):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(bias),
        torch.empty_like(vcls),
    )


def _ca_setup(ctx, inputs, output) -> None:
    (q, k, v, bias, vcls, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q) = inputs
    out, m, l = output
    ctx.save_for_backward(
        q, k, v, bias, vcls, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q, out, m, l
    )


def _ca_dispatch(ctx, grad_out, _gm, _gl):
    dq, dk, dv, dbias, dvcls = _ca_backward_op(*ctx.saved_tensors, grad_out)
    return (dq, dk, dv, dbias, dvcls) + (None,) * 9


_ca_op.register_autograd(_ca_dispatch, setup_context=_ca_setup)


@torch.library.custom_op("mantisnet::cell_attention_plain", mutates_args=())
def _ca_plain_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    seg_ptr: Tensor,
    src: Tensor,
    cls: Tensor,
    qid: Tensor,
    sseg_ptr: Tensor,
    sqid: Tensor,
    scls: Tensor,
    cptr: Tensor,
    cedge_q: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Typed segment attention, score bias only (the window read-back)."""
    _validate(q, k, v, bias, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls)
    return _forward_impl(q, k, v, bias, None, seg_ptr, src, cls, qid)


@_ca_plain_op.register_fake
def _(q, k, v, bias, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q):
    n_q, heads, hd = q.shape
    return (
        q.new_empty((n_q, heads, hd), dtype=torch.float32),
        q.new_empty((n_q, heads), dtype=torch.float32),
        q.new_empty((n_q, heads), dtype=torch.float32),
    )


@torch.library.custom_op("mantisnet::cell_attention_plain_backward", mutates_args=())
def _ca_plain_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    seg_ptr: Tensor,
    src: Tensor,
    cls: Tensor,
    qid: Tensor,
    sseg_ptr: Tensor,
    sqid: Tensor,
    scls: Tensor,
    cptr: Tensor,
    cedge_q: Tensor,
    out: Tensor,
    m: Tensor,
    l: Tensor,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dq, dk, dv, dbias, _ = _backward_impl(
        q, k, v, bias, None, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q, out, m, l, grad_out
    )
    return dq, dk, dv, dbias


@_ca_plain_backward_op.register_fake
def _(q, k, v, bias, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q, out, m, l, grad_out):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(bias),
    )


def _ca_plain_setup(ctx, inputs, output) -> None:
    (q, k, v, bias, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q) = inputs
    out, m, l = output
    ctx.save_for_backward(
        q, k, v, bias, seg_ptr, src, cls, qid, sseg_ptr, sqid, scls, cptr, cedge_q, out, m, l
    )


def _ca_plain_dispatch(ctx, grad_out, _gm, _gl):
    dq, dk, dv, dbias = _ca_plain_backward_op(*ctx.saved_tensors, grad_out)
    return (dq, dk, dv, dbias) + (None,) * 9


_ca_plain_op.register_autograd(_ca_plain_dispatch, setup_context=_ca_plain_setup)


def cell_read(
    q: Tensor, k: Tensor, v: Tensor, bias: Tensor, vcls: Tensor, tables: CellTables
) -> Tensor:
    """Cells read their containing windows: fp32 ``(N_cov, heads, head_dim)``.

    ``q`` rows are covered cells; ``k``/``v`` rows are windows; ``vcls`` is
    the ``(726, H)`` class value table added per edge.
    """
    out, _m, _l = _ca_op(
        q,
        k,
        v,
        bias,
        vcls,
        tables.cell_ptr,
        tables.edge_window,
        tables.edge_class,
        tables.edge_cell,
        tables.win_ptr,
        tables.edge_wcell,
        tables.edge_wclass,
        tables.cls_ptr,
        tables.cedge_cell,
    )
    return out


def window_read(
    q: Tensor, k: Tensor, v: Tensor, bias: Tensor, tables: CellTables
) -> Tensor:
    """Windows read their empty cells: fp32 ``(N_w, heads, head_dim)``.

    ``q`` rows are windows; ``k``/``v`` rows are covered cells. A full
    window has no entries and reads zero.
    """
    out, _m, _l = _ca_plain_op(
        q,
        k,
        v,
        bias,
        tables.win_ptr,
        tables.edge_wcell,
        tables.edge_wclass,
        tables.edge_wwin,
        tables.cell_ptr,
        tables.edge_window,
        tables.edge_class,
        tables.cls_ptr,
        tables.cedge_win,
    )
    return out
