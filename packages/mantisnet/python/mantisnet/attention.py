"""Fused coordinate-biased attention for the stone rows."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_PAD_BIAS = -3.0e4

# Fixed launch geometry keeps symbolic shape changes out of Triton's tuning cache.
_BLOCK_M = 64
_BLOCK_N = 64
_NUM_WARPS = 4
_NUM_STAGES = 3

_FAILED_SHAPES: dict[tuple[object, ...], str] = {}
_FAILED_BACKWARD_SHAPES: dict[tuple[object, ...], str] = {}

# A custom-op implementation runs below the Autograd dispatch keys. Restore
# the normal thread-local keysets only for its rare dense-backward fallback.
_DENSE_BACKWARD_INCLUDE_KEYS = torch._C._dispatch_tls_local_include_set()
_DENSE_BACKWARD_EXCLUDE_KEYS = torch._C._dispatch_tls_local_exclude_set()


if triton is not None:

    @triton.jit
    def _fused_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        coords_ptr,
        seq_lens_ptr,
        table_ptr,
        out_ptr,
        lse_ptr,
        stride_qp,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kp,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vp,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_cp,
        stride_ct,
        stride_cc,
        stride_lp,
        stride_th,
        stride_tb,
        stride_op,
        stride_oh,
        stride_ot,
        stride_od,
        stride_lsep,
        stride_lseh,
        stride_lset,
        n_heads,
        n_ctx,
        sm_scale,
        global_rows,
        D_MAX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_ph = tl.program_id(1)
        off_p = off_ph // n_heads
        off_h = off_ph - off_p * n_heads

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        live_len = tl.load(seq_lens_ptr + off_p * stride_lp)
        live_len = tl.minimum(tl.maximum(live_len, 0), n_ctx)
        out_ptrs = (
            out_ptr
            + off_p * stride_op
            + off_h * stride_oh
            + offs_m[:, None] * stride_ot
            + offs_d[None, :] * stride_od
        )
        lse_ptrs = (
            lse_ptr
            + off_p * stride_lsep
            + off_h * stride_lseh
            + offs_m * stride_lset
        )

        # A row whose query tile starts after its live prefix performs no key
        # work. The zeros also make padding deterministic for direct op users.
        if start_m >= live_len:
            zeros = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            tl.store(out_ptrs, zeros, mask=offs_m[:, None] < n_ctx)
            tl.store(lse_ptrs, 0.0, mask=offs_m < n_ctx)
        else:
            q_ptrs = (
                q_ptr
                + off_p * stride_qp
                + off_h * stride_qh
                + offs_m[:, None] * stride_qt
                + offs_d[None, :] * stride_qd
            )
            q_live = offs_m < live_len
            q = tl.load(q_ptrs, mask=q_live[:, None], other=0.0)

            coords_base = coords_ptr + off_p * stride_cp
            q_q = tl.load(
                coords_base + offs_m * stride_ct,
                mask=offs_m < n_ctx,
                other=0,
            )
            q_r = tl.load(
                coords_base + offs_m * stride_ct + stride_cc,
                mask=offs_m < n_ctx,
                other=0,
            )

            m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

            # The dynamic upper bound is the row's live prefix, so no complete
            # key tile beyond it is loaded or multiplied.
            for start_n in tl.range(0, live_len, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + tl.arange(0, BLOCK_N)
                k_live = offs_n < live_len

                k_ptrs = (
                    k_ptr
                    + off_p * stride_kp
                    + off_h * stride_kh
                    + offs_n[:, None] * stride_kt
                    + offs_d[None, :] * stride_kd
                )
                v_ptrs = (
                    v_ptr
                    + off_p * stride_vp
                    + off_h * stride_vh
                    + offs_n[:, None] * stride_vt
                    + offs_d[None, :] * stride_vd
                )
                k = tl.load(k_ptrs, mask=k_live[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=k_live[:, None], other=0.0)

                k_q = tl.load(
                    coords_base + offs_n * stride_ct,
                    mask=k_live,
                    other=0,
                )
                k_r = tl.load(
                    coords_base + offs_n * stride_ct + stride_cc,
                    mask=k_live,
                    other=0,
                )
                dq = q_q[:, None] - k_q[None, :]
                dr = q_r[:, None] - k_r[None, :]
                distance = tl.maximum(
                    tl.abs(dq),
                    tl.maximum(tl.abs(dr), tl.abs(dq + dr)),
                )
                bucket = tl.minimum(tl.maximum(distance, 1), D_MAX) - 1
                on_axis = (dq == 0) | (dr == 0) | (dq + dr == 0)
                bucket = tl.where(on_axis, D_MAX + 3 + bucket, bucket)
                bucket = tl.where(
                    offs_m[:, None] == offs_n[None, :],
                    D_MAX,
                    bucket,
                )
                bucket = tl.where(
                    (offs_m[:, None] < global_rows)
                    | (offs_n[None, :] < global_rows),
                    D_MAX + 1,
                    bucket,
                )
                bucket = tl.where(k_live[None, :], bucket, D_MAX + 2)
                bias = tl.load(
                    table_ptr + off_h * stride_th + bucket * stride_tb,
                    cache_modifier=".ca",
                )

                scores = tl.dot(q, tl.trans(k)) * sm_scale + bias
                m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
                p = tl.math.exp2((scores - m_ij[:, None]) * 1.4426950408889634)
                alpha = tl.math.exp2((m_i - m_ij) * 1.4426950408889634)
                acc *= alpha[:, None]
                acc += tl.dot(p.to(q_ptr.dtype.element_ty), v)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_ij

            out = acc / l_i[:, None]
            out = tl.where(q_live[:, None], out, 0.0)
            tl.store(out_ptrs, out, mask=offs_m[:, None] < n_ctx)
            lse = m_i + tl.log(l_i)
            lse = tl.where(q_live, lse, 0.0)
            tl.store(lse_ptrs, lse, mask=offs_m < n_ctx)


    @triton.jit
    def _fused_attention_delta_kernel(
        out_ptr,
        do_ptr,
        seq_lens_ptr,
        delta_ptr,
        stride_op,
        stride_oh,
        stride_ot,
        stride_od,
        stride_dop,
        stride_doh,
        stride_dot,
        stride_dod,
        stride_lp,
        stride_dp,
        stride_dh,
        stride_dt,
        n_heads,
        n_ctx,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_ph = tl.program_id(1)
        off_p = off_ph // n_heads
        off_h = off_ph - off_p * n_heads

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        live_len = tl.load(seq_lens_ptr + off_p * stride_lp)
        live_len = tl.minimum(tl.maximum(live_len, 0), n_ctx)
        row_live = offs_m < live_len

        out = tl.load(
            out_ptr
            + off_p * stride_op
            + off_h * stride_oh
            + offs_m[:, None] * stride_ot
            + offs_d[None, :] * stride_od,
            mask=row_live[:, None],
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            do_ptr
            + off_p * stride_dop
            + off_h * stride_doh
            + offs_m[:, None] * stride_dot
            + offs_d[None, :] * stride_dod,
            mask=row_live[:, None],
            other=0.0,
        ).to(tl.float32)
        delta = tl.sum(out * do, axis=1)
        delta = tl.where(row_live, delta, 0.0)
        tl.store(
            delta_ptr
            + off_p * stride_dp
            + off_h * stride_dh
            + offs_m * stride_dt,
            delta,
            mask=offs_m < n_ctx,
        )


    @triton.jit
    def _fused_attention_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        coords_ptr,
        seq_lens_ptr,
        table_ptr,
        lse_ptr,
        delta_ptr,
        do_ptr,
        dq_ptr,
        dtable_ptr,
        stride_qp,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kp,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vp,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_cp,
        stride_ct,
        stride_cc,
        stride_lp,
        stride_th,
        stride_tb,
        stride_lsep,
        stride_lseh,
        stride_lset,
        stride_delp,
        stride_delh,
        stride_delt,
        stride_dop,
        stride_doh,
        stride_dot,
        stride_dod,
        stride_dqp,
        stride_dqh,
        stride_dqt,
        stride_dqd,
        stride_dth,
        stride_dtb,
        n_heads,
        n_ctx,
        sm_scale,
        global_rows,
        D_MAX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_BUCKETS: tl.constexpr,
        TABLE_WIDTH: tl.constexpr,
    ):
        start_m = tl.program_id(0) * BLOCK_M
        off_ph = tl.program_id(1)
        off_p = off_ph // n_heads
        off_h = off_ph - off_p * n_heads

        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        live_len = tl.load(seq_lens_ptr + off_p * stride_lp)
        live_len = tl.minimum(tl.maximum(live_len, 0), n_ctx)
        q_live = offs_m < live_len
        dq_ptrs = (
            dq_ptr
            + off_p * stride_dqp
            + off_h * stride_dqh
            + offs_m[:, None] * stride_dqt
            + offs_d[None, :] * stride_dqd
        )

        if start_m >= live_len:
            zeros = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            tl.store(dq_ptrs, zeros, mask=offs_m[:, None] < n_ctx)
        else:
            q = tl.load(
                q_ptr
                + off_p * stride_qp
                + off_h * stride_qh
                + offs_m[:, None] * stride_qt
                + offs_d[None, :] * stride_qd,
                mask=q_live[:, None],
                other=0.0,
            )
            do = tl.load(
                do_ptr
                + off_p * stride_dop
                + off_h * stride_doh
                + offs_m[:, None] * stride_dot
                + offs_d[None, :] * stride_dod,
                mask=q_live[:, None],
                other=0.0,
            )
            lse = tl.load(
                lse_ptr
                + off_p * stride_lsep
                + off_h * stride_lseh
                + offs_m * stride_lset,
                mask=q_live,
                other=0.0,
            )
            delta = tl.load(
                delta_ptr
                + off_p * stride_delp
                + off_h * stride_delh
                + offs_m * stride_delt,
                mask=q_live,
                other=0.0,
            )

            coords_base = coords_ptr + off_p * stride_cp
            q_q = tl.load(
                coords_base + offs_m * stride_ct,
                mask=offs_m < n_ctx,
                other=0,
            )
            q_r = tl.load(
                coords_base + offs_m * stride_ct + stride_cc,
                mask=offs_m < n_ctx,
                other=0,
            )

            dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
            offs_b = tl.arange(0, BLOCK_BUCKETS)
            dtable_acc = tl.zeros([BLOCK_BUCKETS], dtype=tl.float32)

            for start_n in tl.range(0, live_len, BLOCK_N):
                start_n = tl.multiple_of(start_n, BLOCK_N)
                offs_n = start_n + tl.arange(0, BLOCK_N)
                k_live = offs_n < live_len

                k = tl.load(
                    k_ptr
                    + off_p * stride_kp
                    + off_h * stride_kh
                    + offs_n[:, None] * stride_kt
                    + offs_d[None, :] * stride_kd,
                    mask=k_live[:, None],
                    other=0.0,
                )
                v = tl.load(
                    v_ptr
                    + off_p * stride_vp
                    + off_h * stride_vh
                    + offs_n[:, None] * stride_vt
                    + offs_d[None, :] * stride_vd,
                    mask=k_live[:, None],
                    other=0.0,
                )
                k_q = tl.load(
                    coords_base + offs_n * stride_ct,
                    mask=k_live,
                    other=0,
                )
                k_r = tl.load(
                    coords_base + offs_n * stride_ct + stride_cc,
                    mask=k_live,
                    other=0,
                )

                coord_q = q_q[:, None] - k_q[None, :]
                coord_r = q_r[:, None] - k_r[None, :]
                distance = tl.maximum(
                    tl.abs(coord_q),
                    tl.maximum(tl.abs(coord_r), tl.abs(coord_q + coord_r)),
                )
                bucket = tl.minimum(tl.maximum(distance, 1), D_MAX) - 1
                on_axis = (
                    (coord_q == 0)
                    | (coord_r == 0)
                    | (coord_q + coord_r == 0)
                )
                bucket = tl.where(on_axis, D_MAX + 3 + bucket, bucket)
                bucket = tl.where(
                    offs_m[:, None] == offs_n[None, :],
                    D_MAX,
                    bucket,
                )
                bucket = tl.where(
                    (offs_m[:, None] < global_rows)
                    | (offs_n[None, :] < global_rows),
                    D_MAX + 1,
                    bucket,
                )
                bucket = tl.where(k_live[None, :], bucket, D_MAX + 2)
                bias = tl.load(
                    table_ptr + off_h * stride_th + bucket * stride_tb,
                    cache_modifier=".ca",
                )

                scores = tl.dot(q, tl.trans(k)) * sm_scale + bias
                p = tl.math.exp2(
                    (scores - lse[:, None]) * 1.4426950408889634
                )
                pair_live = q_live[:, None] & k_live[None, :]
                p = tl.where(pair_live, p, 0.0)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - delta[:, None])
                dq_acc += tl.dot(ds.to(k_ptr.dtype.element_ty), k) * sm_scale

                for b in tl.static_range(0, TABLE_WIDTH):
                    bucket_grad = tl.sum(
                        tl.sum(tl.where(bucket == b, ds, 0.0), axis=1),
                        axis=0,
                    )
                    dtable_acc += tl.where(offs_b == b, bucket_grad, 0.0)

            dq_acc = tl.where(q_live[:, None], dq_acc, 0.0)
            tl.store(dq_ptrs, dq_acc, mask=offs_m[:, None] < n_ctx)
            tl.atomic_add(
                dtable_ptr
                + off_h * stride_dth
                + offs_b * stride_dtb,
                dtable_acc,
                mask=offs_b < TABLE_WIDTH,
            )


    @triton.jit
    def _fused_attention_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        coords_ptr,
        seq_lens_ptr,
        table_ptr,
        lse_ptr,
        delta_ptr,
        do_ptr,
        dk_ptr,
        dv_ptr,
        stride_qp,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kp,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vp,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_cp,
        stride_ct,
        stride_cc,
        stride_lp,
        stride_th,
        stride_tb,
        stride_lsep,
        stride_lseh,
        stride_lset,
        stride_delp,
        stride_delh,
        stride_delt,
        stride_dop,
        stride_doh,
        stride_dot,
        stride_dod,
        stride_dkp,
        stride_dkh,
        stride_dkt,
        stride_dkd,
        stride_dvp,
        stride_dvh,
        stride_dvt,
        stride_dvd,
        n_heads,
        n_ctx,
        sm_scale,
        global_rows,
        D_MAX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_n = tl.program_id(0) * BLOCK_N
        off_ph = tl.program_id(1)
        off_p = off_ph // n_heads
        off_h = off_ph - off_p * n_heads

        offs_n = start_n + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        live_len = tl.load(seq_lens_ptr + off_p * stride_lp)
        live_len = tl.minimum(tl.maximum(live_len, 0), n_ctx)
        k_live = offs_n < live_len
        dk_ptrs = (
            dk_ptr
            + off_p * stride_dkp
            + off_h * stride_dkh
            + offs_n[:, None] * stride_dkt
            + offs_d[None, :] * stride_dkd
        )
        dv_ptrs = (
            dv_ptr
            + off_p * stride_dvp
            + off_h * stride_dvh
            + offs_n[:, None] * stride_dvt
            + offs_d[None, :] * stride_dvd
        )

        if start_n >= live_len:
            zeros = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            tl.store(dk_ptrs, zeros, mask=offs_n[:, None] < n_ctx)
            tl.store(dv_ptrs, zeros, mask=offs_n[:, None] < n_ctx)
        else:
            k = tl.load(
                k_ptr
                + off_p * stride_kp
                + off_h * stride_kh
                + offs_n[:, None] * stride_kt
                + offs_d[None, :] * stride_kd,
                mask=k_live[:, None],
                other=0.0,
            )
            v = tl.load(
                v_ptr
                + off_p * stride_vp
                + off_h * stride_vh
                + offs_n[:, None] * stride_vt
                + offs_d[None, :] * stride_vd,
                mask=k_live[:, None],
                other=0.0,
            )
            coords_base = coords_ptr + off_p * stride_cp
            k_q = tl.load(
                coords_base + offs_n * stride_ct,
                mask=k_live,
                other=0,
            )
            k_r = tl.load(
                coords_base + offs_n * stride_ct + stride_cc,
                mask=k_live,
                other=0,
            )

            dk_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
            dv_acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

            for start_m in tl.range(0, live_len, BLOCK_M):
                start_m = tl.multiple_of(start_m, BLOCK_M)
                offs_m = start_m + tl.arange(0, BLOCK_M)
                q_live = offs_m < live_len

                q = tl.load(
                    q_ptr
                    + off_p * stride_qp
                    + off_h * stride_qh
                    + offs_m[:, None] * stride_qt
                    + offs_d[None, :] * stride_qd,
                    mask=q_live[:, None],
                    other=0.0,
                )
                do = tl.load(
                    do_ptr
                    + off_p * stride_dop
                    + off_h * stride_doh
                    + offs_m[:, None] * stride_dot
                    + offs_d[None, :] * stride_dod,
                    mask=q_live[:, None],
                    other=0.0,
                )
                lse = tl.load(
                    lse_ptr
                    + off_p * stride_lsep
                    + off_h * stride_lseh
                    + offs_m * stride_lset,
                    mask=q_live,
                    other=0.0,
                )
                delta = tl.load(
                    delta_ptr
                    + off_p * stride_delp
                    + off_h * stride_delh
                    + offs_m * stride_delt,
                    mask=q_live,
                    other=0.0,
                )
                q_q = tl.load(
                    coords_base + offs_m * stride_ct,
                    mask=offs_m < n_ctx,
                    other=0,
                )
                q_r = tl.load(
                    coords_base + offs_m * stride_ct + stride_cc,
                    mask=offs_m < n_ctx,
                    other=0,
                )

                coord_q = q_q[:, None] - k_q[None, :]
                coord_r = q_r[:, None] - k_r[None, :]
                distance = tl.maximum(
                    tl.abs(coord_q),
                    tl.maximum(tl.abs(coord_r), tl.abs(coord_q + coord_r)),
                )
                bucket = tl.minimum(tl.maximum(distance, 1), D_MAX) - 1
                on_axis = (
                    (coord_q == 0)
                    | (coord_r == 0)
                    | (coord_q + coord_r == 0)
                )
                bucket = tl.where(on_axis, D_MAX + 3 + bucket, bucket)
                bucket = tl.where(
                    offs_m[:, None] == offs_n[None, :],
                    D_MAX,
                    bucket,
                )
                bucket = tl.where(
                    (offs_m[:, None] < global_rows)
                    | (offs_n[None, :] < global_rows),
                    D_MAX + 1,
                    bucket,
                )
                bucket = tl.where(k_live[None, :], bucket, D_MAX + 2)
                bias = tl.load(
                    table_ptr + off_h * stride_th + bucket * stride_tb,
                    cache_modifier=".ca",
                )

                scores = tl.dot(q, tl.trans(k)) * sm_scale + bias
                p = tl.math.exp2(
                    (scores - lse[:, None]) * 1.4426950408889634
                )
                pair_live = q_live[:, None] & k_live[None, :]
                p = tl.where(pair_live, p, 0.0)
                dp = tl.dot(do, tl.trans(v))
                ds = p * (dp - delta[:, None])
                dk_acc += (
                    tl.dot(tl.trans(ds.to(q_ptr.dtype.element_ty)), q)
                    * sm_scale
                )
                dv_acc += tl.dot(
                    tl.trans(p.to(do_ptr.dtype.element_ty)),
                    do,
                )

            dk_acc = tl.where(k_live[:, None], dk_acc, 0.0)
            dv_acc = tl.where(k_live[:, None], dv_acc, 0.0)
            tl.store(dk_ptrs, dk_acc, mask=offs_n[:, None] < n_ctx)
            tl.store(dv_ptrs, dv_acc, mask=offs_n[:, None] < n_ctx)


def _bias_table(q: Tensor, dist_bias: Tensor, axis_bias: Tensor) -> Tensor:
    """Cast learned rows once and insert the finite PAD sentinel.

    Layout: distances 1..D_MAX, SELF, TOKEN, PAD, then the on-axis rows
    1..D_MAX that replace the base distance row for aligned pairs.
    """
    if dist_bias.ndim != 2 or dist_bias.shape[1] < 3:
        raise ValueError("dist_bias must have shape (A, d_max + 2) with d_max >= 1")
    d_max = dist_bias.shape[1] - 2
    if axis_bias.shape != (dist_bias.shape[0], d_max):
        raise ValueError("axis_bias must have shape (A, d_max)")
    table = dist_bias.to(q.dtype)
    pad = table.new_full((table.shape[0], 1), _PAD_BIAS)
    return torch.cat((table, pad, axis_bias.to(q.dtype)), dim=1)


def _bucket_index(
    coords: Tensor,
    seq_lens: Tensor,
    t: int,
    d_max: int,
    global_rows: int = 1,
):
    """The (P, T, T) bias-bucket index and (P, T) key validity."""
    dq = coords[:, :, None, 0] - coords[:, None, :, 0]
    dr = coords[:, :, None, 1] - coords[:, None, :, 1]
    distance = torch.maximum(dq.abs(), torch.maximum(dr.abs(), (dq + dr).abs()))
    base = distance.clamp(1, d_max) - 1
    on_axis = (dq == 0) | (dr == 0) | (dq + dr == 0)
    bucket = torch.where(on_axis, d_max + 3 + base, base)

    rows = torch.arange(t, device=coords.device)
    bucket = torch.where(rows[:, None] == rows[None, :], d_max, bucket)
    token = (rows[:, None] < global_rows) | (rows[None, :] < global_rows)
    bucket = torch.where(token, d_max + 1, bucket)
    valid = rows[None, :] < seq_lens[:, None]
    bucket = torch.where(valid[:, None, :], bucket, d_max + 2)
    return bucket, valid


def _apply_reference(q: Tensor, k: Tensor, v: Tensor, mask: Tensor, valid: Tensor) -> Tensor:
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    result = result.masked_fill(~valid[:, None, :, None], 0)

    # Match empty_like(q)'s preserved strides in every dispatch path. The
    # model's following head-to-row transpose can then remain a view.
    out = torch.empty_like(q)
    out.copy_(result)
    return out


def _attention_reference_table(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    d_max: int,
    global_rows: int,
) -> Tensor:
    """The dense formulation used by CPU, failed launches, and recompute."""
    _, _, t, _ = q.shape
    if table.shape[1] != 2 * d_max + 3:
        raise ValueError(
            f"bias table width {table.shape[1]} != 2*d_max+3 = {2 * d_max + 3}"
        )
    bucket, valid = _bucket_index(coords, seq_lens, t, d_max, global_rows)
    mask = table[:, bucket.long()].permute(1, 0, 2, 3)
    return _apply_reference(q, k, v, mask, valid)


def _attention_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    dist_bias: Tensor,
    axis_bias: Tensor,
    global_rows: int = 1,
) -> Tensor:
    """Reference attention with the checkpoint-compatible bias parameter."""
    d_max = dist_bias.shape[1] - 2
    return _attention_reference_table(
        q,
        k,
        v,
        coords,
        seq_lens,
        _bias_table(q, dist_bias, axis_bias),
        d_max,
        global_rows,
    )


def _shape_key(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    d_max: int,
    global_rows: int,
) -> tuple[object, ...]:
    return (
        q.device.type,
        q.device.index,
        q.dtype,
        tuple(q.shape),
        tuple(q.stride()),
        tuple(k.stride()),
        tuple(v.stride()),
        tuple(coords.stride()),
        tuple(seq_lens.stride()),
        tuple(table.shape),
        tuple(table.stride()),
        d_max,
        global_rows,
    )


def _validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    d_max: int,
    global_rows: int,
) -> None:
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same (P, A, T, D) shape")
    p, heads, t, _ = q.shape
    if coords.shape != (p, t, 2) or coords.dtype != torch.int32:
        raise ValueError("coords must be int32 with shape (P, T, 2)")
    if seq_lens.shape != (p,) or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be int32 with shape (P,)")
    if d_max < 1:
        raise ValueError("d_max must be at least 1")
    if global_rows < 1 or global_rows > t:
        raise ValueError(
            f"global_rows must be in [1, {t}], got {global_rows}"
        )
    table_widths = (d_max + 3, 2 * d_max + 3, 3 * d_max + 3)
    if (
        table.ndim != 2
        or table.shape[0] != heads
        or table.shape[1] not in table_widths
    ):
        raise ValueError(
            "bias table must have shape (A, d_max + 3), "
            "(A, 2 * d_max + 3), or (A, 3 * d_max + 3)"
        )
    tensors = (k, v, coords, seq_lens, table)
    if any(x.device != q.device for x in tensors):
        raise ValueError("all attention inputs must be on one device")
    if k.dtype != q.dtype or v.dtype != q.dtype or table.dtype != q.dtype:
        raise ValueError("q, k, v, and the bias table must have one dtype")


def _launch_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    d_max: int,
    global_rows: int,
) -> tuple[Tensor, Tensor]:
    p, heads, t, head_dim = q.shape
    out = torch.empty_like(q)
    lse = torch.empty((p, heads, t), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(t, _BLOCK_M), p * heads)
    _fused_attention_kernel[grid](
        q,
        k,
        v,
        coords,
        seq_lens,
        table,
        out,
        lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *coords.stride(),
        *seq_lens.stride(),
        *table.stride(),
        *out.stride(),
        *lse.stride(),
        heads,
        t,
        1.0 / math.sqrt(head_dim),
        global_rows,
        D_MAX=d_max,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out, lse


def _launch_triton_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    out: Tensor,
    lse: Tensor,
    grad_out: Tensor,
    d_max: int,
    global_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    p, heads, t, head_dim = q.shape
    delta = torch.empty((p, heads, t), dtype=torch.float32, device=q.device)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dtable = torch.zeros(table.shape, dtype=torch.float32, device=table.device)
    row_grid = (triton.cdiv(t, _BLOCK_M), p * heads)
    key_grid = (triton.cdiv(t, _BLOCK_N), p * heads)

    _fused_attention_delta_kernel[row_grid](
        out,
        grad_out,
        seq_lens,
        delta,
        *out.stride(),
        *grad_out.stride(),
        *seq_lens.stride(),
        *delta.stride(),
        heads,
        t,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _fused_attention_dq_kernel[row_grid](
        q,
        k,
        v,
        coords,
        seq_lens,
        table,
        lse,
        delta,
        grad_out,
        dq,
        dtable,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *coords.stride(),
        *seq_lens.stride(),
        *table.stride(),
        *lse.stride(),
        *delta.stride(),
        *grad_out.stride(),
        *dq.stride(),
        *dtable.stride(),
        heads,
        t,
        1.0 / math.sqrt(head_dim),
        global_rows,
        D_MAX=d_max,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_BUCKETS=triton.next_power_of_2(table.shape[1]),
        TABLE_WIDTH=table.shape[1],
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _fused_attention_dkdv_kernel[key_grid](
        q,
        k,
        v,
        coords,
        seq_lens,
        table,
        lse,
        delta,
        grad_out,
        dk,
        dv,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *coords.stride(),
        *seq_lens.stride(),
        *table.stride(),
        *lse.stride(),
        *delta.stride(),
        *grad_out.stride(),
        *dk.stride(),
        *dv.stride(),
        heads,
        t,
        1.0 / math.sqrt(head_dim),
        global_rows,
        D_MAX=d_max,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return dq, dk, dv, dtable.to(table.dtype)


@torch.library.custom_op("mantisnet::fused_attention", mutates_args=())
def _fused_attention_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    d_max: int,
    global_rows: int,
) -> tuple[Tensor, Tensor]:
    _validate_inputs(q, k, v, coords, seq_lens, table, d_max, global_rows)
    supported = (
        triton is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (16, 32, 64)
    )
    if not supported:
        out = _attention_reference_table(
            q, k, v, coords, seq_lens, table, d_max, global_rows
        )
        return out, torch.empty(0, dtype=torch.float32, device=q.device)

    key = _shape_key(q, k, v, coords, seq_lens, table, d_max, global_rows)
    if key in _FAILED_SHAPES:
        out = _attention_reference_table(
            q, k, v, coords, seq_lens, table, d_max, global_rows
        )
        return out, torch.empty(0, dtype=torch.float32, device=q.device)
    try:
        return _launch_triton(
            q, k, v, coords, seq_lens, table, d_max, global_rows
        )
    except Exception as exc:
        _FAILED_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "fused attention failed for "
            f"shape={tuple(q.shape)}, dtype={q.dtype}; using SDPA for this "
            f"shape: {_FAILED_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        out = _attention_reference_table(
            q, k, v, coords, seq_lens, table, d_max, global_rows
        )
        return out, torch.empty(0, dtype=torch.float32, device=q.device)


@_fused_attention_op.register_fake
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    d_max: int,
    global_rows: int,
) -> tuple[Tensor, Tensor]:
    supported = (
        triton is not None
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (16, 32, 64)
    )
    if supported:
        lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty(0, dtype=torch.float32, device=q.device)
    return torch.empty_like(q), lse


def _setup_context(ctx, inputs, output) -> None:
    q, k, v, coords, seq_lens, table, d_max, global_rows = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, coords, seq_lens, table, out, lse)
    ctx.d_max = d_max
    ctx.global_rows = global_rows
    ctx.mark_non_differentiable(lse)


def _backward(ctx, grad_out: Tensor):
    q, k, v, coords, seq_lens, table = ctx.saved_tensors
    t = q.shape[2]
    d_max = ctx.d_max
    global_rows = ctx.global_rows
    bucket, valid = _bucket_index(coords, seq_lens, t, d_max, global_rows)
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        # The dense bias enters the scores additively, so its gradient is the
        # per-pair score gradient. Differentiating the mask directly (instead
        # of the table gather) keeps the scatter out of autograd: reducing
        # P*A*T*T gradients into a table this small serializes on atomics.
        mask = table.detach()[:, bucket.long()].permute(1, 0, 2, 3).requires_grad_(True)
        out = _apply_reference(q_, k_, v_, mask, valid)
        dq, dk, dv, dmask = torch.autograd.grad(
            out,
            (q_, k_, v_, mask),
            grad_out,
        )
    grads = dmask.float()
    dtable = torch.stack(
        [
            (grads * (bucket == b).unsqueeze(1)).sum(dim=(0, 2, 3))
            for b in range(table.shape[1])
        ],
        dim=1,
    ).to(table.dtype)
    return dq, dk, dv, None, None, dtable, None, None


class _DenseBackwardContext:
    def __init__(
        self, saved_tensors: tuple[Tensor, ...], d_max: int, global_rows: int
    ) -> None:
        self.saved_tensors = saved_tensors
        self.d_max = d_max
        self.global_rows = global_rows


def _dense_backward_below_autograd(
    saved_tensors: tuple[Tensor, ...],
    d_max: int,
    global_rows: int,
    grad_out: Tensor,
):
    with torch._C._ForceDispatchKeyGuard(
        _DENSE_BACKWARD_INCLUDE_KEYS, _DENSE_BACKWARD_EXCLUDE_KEYS
    ):
        return _backward(
            _DenseBackwardContext(saved_tensors, d_max, global_rows), grad_out
        )


def _backward_shape_key(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    grad_out: Tensor,
    d_max: int,
    global_rows: int,
) -> tuple[object, ...]:
    return _shape_key(
        q, k, v, coords, seq_lens, table, d_max, global_rows
    ) + (
        grad_out.dtype,
        tuple(grad_out.stride()),
    )


@torch.library.custom_op("mantisnet::fused_attention_backward", mutates_args=())
def _fused_attention_backward_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    out: Tensor,
    lse: Tensor,
    grad_out: Tensor,
    d_max: int,
    global_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_inputs(q, k, v, coords, seq_lens, table, d_max, global_rows)
    saved = (q, k, v, coords, seq_lens, table)
    # The outer autograd formula handles this branch in eager mode. Keep the
    # sentinel check inside the opaque op as well: a compiled graph may have
    # been traced for Triton before a runtime forward launch marks the shape
    # failed and returns the dense sentinel.
    if lse.numel() == 0:
        dq, dk, dv, _, _, dtable, _, _ = _dense_backward_below_autograd(
            saved, d_max, global_rows, grad_out
        )
        return dq, dk, dv, dtable

    key = _backward_shape_key(
        q, k, v, coords, seq_lens, table, grad_out, d_max, global_rows
    )
    if key in _FAILED_BACKWARD_SHAPES:
        dq, dk, dv, _, _, dtable, _, _ = _dense_backward_below_autograd(
            saved, d_max, global_rows, grad_out
        )
        return dq, dk, dv, dtable
    try:
        return _launch_triton_backward(
            q,
            k,
            v,
            coords,
            seq_lens,
            table,
            out,
            lse,
            grad_out,
            d_max,
            global_rows,
        )
    except Exception as exc:
        _FAILED_BACKWARD_SHAPES[key] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "fused attention backward failed for "
            f"shape={tuple(q.shape)}, dtype={q.dtype}; using dense backward "
            f"for this shape: {_FAILED_BACKWARD_SHAPES[key]}",
            RuntimeWarning,
            stacklevel=2,
        )
        dq, dk, dv, _, _, dtable, _, _ = _dense_backward_below_autograd(
            saved, d_max, global_rows, grad_out
        )
        return dq, dk, dv, dtable


@_fused_attention_backward_op.register_fake
def _(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    table: Tensor,
    out: Tensor,
    lse: Tensor,
    grad_out: Tensor,
    d_max: int,
    global_rows: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty(table.shape, dtype=table.dtype, device=table.device),
    )


def _dispatch_backward(ctx, grad_out: Tensor, _grad_lse: Tensor | None):
    q, k, v, coords, seq_lens, table, out, lse = ctx.saved_tensors
    d_max = ctx.d_max
    global_rows = ctx.global_rows
    if lse.numel() == 0:
        return _backward(
            _DenseBackwardContext(
                (q, k, v, coords, seq_lens, table), d_max, global_rows
            ),
            grad_out,
        )
    dq, dk, dv, dtable = _fused_attention_backward_op(
        q, k, v, coords, seq_lens, table, out, lse, grad_out, d_max, global_rows
    )
    return dq, dk, dv, None, None, dtable, None, None


_fused_attention_op.register_autograd(
    _dispatch_backward, setup_context=_setup_context
)


def fused_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    coords: Tensor,
    seq_lens: Tensor,
    dist_bias: Tensor,
    axis_bias: Tensor,
    global_rows: int = 1,
) -> Tensor:
    """Apply fused attention while retaining the checkpoint bias layout."""
    d_max = dist_bias.shape[1] - 2
    out, _ = _fused_attention_op(
        q,
        k,
        v,
        coords,
        seq_lens,
        _bias_table(q, dist_bias, axis_bias),
        d_max,
        global_rows,
    )
    return out
