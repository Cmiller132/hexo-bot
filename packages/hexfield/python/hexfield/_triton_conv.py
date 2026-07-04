"""Fused gather+GEMM Triton kernel for HexNodeConv (serve-only, opt-in).

The reference HexNodeConv materializes a (B, Npad, 7*C) gathered tensor (cat a
zero row, gather, reshape) and feeds it to one GEMM. At serve shapes that
gather write+read is ~60% of the conv cost. This kernel gathers the 7 tap rows
directly into the GEMM's A-tiles (tl.dot, fp32 accumulate), so the expanded
tensor never exists; the missing-neighbour zero row becomes a masked load
(idx == Npad -> 0) and the output row mask is folded into the epilogue.

Exposed as the `hexfield::hex_conv` custom op (with a fake kernel), so the
serve torch.compile(dynamic=True) graph keeps it in-graph as an opaque call —
no graph breaks. Enabled via HEXFIELD_TRITON_CONV=1; model.py routes to it on
the no-grad CUDA path only (there is no backward), for 16-aligned channel
counts (the stem's C_in=15 keeps the reference path).

Numerics: fp16 tap rows x fp16 weights with fp32 accumulation — the same class
as the autocast cuBLAS GEMM it replaces (bias added in fp32, one final fp16
round like the reference's fp16 GEMM output). Output is fp16.
"""

from __future__ import annotations

import os

import torch

try:  # pragma: no cover - triton ships with cuda torch builds
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

# conv+LN kernel tile knobs (bench sweeps; defaults = measured winners at
# c=192, RTX 4070 Ti, 2026-07-03: BM=64/warps=4/stages=2 runs the fused kernel
# 30-37% FASTER than plain conv + eager LN; the first-cut BM=32/warps=8 was
# the worst point in the sweep, ~neutral vs unfused).
_LN_BM = int(os.environ.get("HEXFIELD_CONVLN_BM", "64"))
_LN_WARPS = int(os.environ.get("HEXFIELD_CONVLN_WARPS", "4"))
_LN_STAGES = int(os.environ.get("HEXFIELD_CONVLN_STAGES", "2"))


if HAVE_TRITON:

    @triton.jit
    def _hex_conv_kernel(
        x_ptr, idx_ptr, mask_ptr, w_ptr, bias_ptr, out_ptr,
        B, Npad, C, Cout,
        IS_FP16_IN: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m_offs = pid_m * BM + tl.arange(0, BM)  # rows over B*Npad
        m_valid = m_offs < B * Npad
        b_ids = m_offs // Npad
        n_offs = pid_n * BN + tl.arange(0, BN)  # Cout columns
        n_valid = n_offs < Cout

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for t in tl.static_range(7):
            # Row-local tap index; Npad is the missing/pad sentinel (zero row).
            idx = tl.load(idx_ptr + m_offs * 7 + t, mask=m_valid, other=Npad)
            row_ok = m_valid & (idx < Npad)
            x_row = (b_ids * Npad + idx) * C
            for k0 in tl.range(0, tl.cdiv(C, BK)):
                k_offs = k0 * BK + tl.arange(0, BK)
                k_ok = k_offs < C
                a = tl.load(
                    x_ptr + x_row[:, None] + k_offs[None, :],
                    mask=row_ok[:, None] & k_ok[None, :],
                    other=0.0,
                )
                a16 = a if IS_FP16_IN else a.to(tl.float16)
                w = tl.load(
                    w_ptr + (t * C + k_offs)[:, None] * Cout + n_offs[None, :],
                    mask=k_ok[:, None] & n_valid[None, :],
                    other=0.0,
                )
                acc += tl.dot(a16, w)

        bias = tl.load(bias_ptr + n_offs, mask=n_valid, other=0.0)
        acc += bias[None, :].to(tl.float32)
        rmask = tl.load(mask_ptr + m_offs, mask=m_valid, other=0)
        acc = tl.where(rmask[:, None] > 0, acc, 0.0)
        tl.store(
            out_ptr + m_offs[:, None] * Cout + n_offs[None, :],
            acc.to(tl.float16),
            mask=m_valid[:, None] & n_valid[None, :],
        )

    @torch.library.custom_op("hexfield::hex_conv", mutates_args=())
    def hex_conv(
        x: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        b, npad, c = x.shape
        cout = weight.shape[-1]
        x = x.contiguous()
        gidx = gather_idx.contiguous()
        m8 = mask.to(torch.uint8).contiguous()
        w16 = weight.reshape(7 * c, cout).to(torch.float16).contiguous()
        b32 = bias.to(torch.float32).contiguous()
        out = torch.empty((b, npad, cout), dtype=torch.float16, device=x.device)
        rows = b * npad
        # Small flushes (late-game / singleton groups) need more, smaller
        # programs to keep the SMs fed; big flushes prefer the fatter tile.
        BM = 32 if rows < 32768 else 64
        BN, BK = min(128, cout), 64
        grid = (triton.cdiv(rows, BM), triton.cdiv(cout, BN))
        _hex_conv_kernel[grid](
            x, gidx, m8, w16, b32, out,
            b, npad, c, cout,
            IS_FP16_IN=(x.dtype == torch.float16),
            BM=BM, BN=BN, BK=BK,
            num_warps=4 if BM == 32 else 8, num_stages=3,
        )
        return out

    @hex_conv.register_fake
    def _hex_conv_fake(x, gather_idx, mask, weight, bias):
        return x.new_empty(
            (x.shape[0], x.shape[1], weight.shape[-1]), dtype=torch.float16
        )

    # --- conv + LayerNorm(+ReLU) + row-mask epilogue -----------------------------
    # Same fused gather+GEMM as _hex_conv_kernel, but the program owns the FULL
    # Cout row (BN >= Cout, one N-tile per program), so the ConvBlock's
    # LayerNorm -> (ReLU) -> mask epilogue runs on the fp32 accumulator before
    # the single fp16 store. Kills one full read+write of the (B, Npad, C)
    # activation per conv (the LN kernel's round-trip). LN statistics are fp32
    # over the true Cout columns; numerically the same class as the reference
    # (which LayerNorms the fp16-rounded conv output in fp32).

    @triton.jit
    def _hex_conv_ln_kernel(
        x_ptr, idx_ptr, mask_ptr, w_ptr, bias_ptr, lnw_ptr, lnb_ptr,
        out_ptr,
        B, Npad, C, Cout, eps,
        IS_FP16_IN: tl.constexpr, RELU: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        m_offs = pid_m * BM + tl.arange(0, BM)  # rows over B*Npad
        m_valid = m_offs < B * Npad
        b_ids = m_offs // Npad
        n_offs = tl.arange(0, BN)  # the whole Cout row
        n_valid = n_offs < Cout

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for t in tl.static_range(7):
            idx = tl.load(idx_ptr + m_offs * 7 + t, mask=m_valid, other=Npad)
            row_ok = m_valid & (idx < Npad)
            x_row = (b_ids * Npad + idx) * C
            for k0 in tl.range(0, tl.cdiv(C, BK)):
                k_offs = k0 * BK + tl.arange(0, BK)
                k_ok = k_offs < C
                a = tl.load(
                    x_ptr + x_row[:, None] + k_offs[None, :],
                    mask=row_ok[:, None] & k_ok[None, :],
                    other=0.0,
                )
                a16 = a if IS_FP16_IN else a.to(tl.float16)
                w = tl.load(
                    w_ptr + (t * C + k_offs)[:, None] * Cout + n_offs[None, :],
                    mask=k_ok[:, None] & n_valid[None, :],
                    other=0.0,
                )
                acc += tl.dot(a16, w)

        bias = tl.load(bias_ptr + n_offs, mask=n_valid, other=0.0)
        acc += bias[None, :].to(tl.float32)
        # LayerNorm over the true Cout columns (fp32 stats on the accumulator).
        accm = tl.where(n_valid[None, :], acc, 0.0)
        mean = tl.sum(accm, 1) / Cout
        cent = tl.where(n_valid[None, :], acc - mean[:, None], 0.0)
        var = tl.sum(cent * cent, 1) / Cout
        rstd = tl.math.rsqrt(var + eps)
        lnw = tl.load(lnw_ptr + n_offs, mask=n_valid, other=0.0)
        lnb = tl.load(lnb_ptr + n_offs, mask=n_valid, other=0.0)
        y = cent * rstd[:, None] * lnw[None, :].to(tl.float32) + lnb[None, :].to(
            tl.float32
        )
        if RELU:
            y = tl.maximum(y, 0.0)
        rmask = tl.load(mask_ptr + m_offs, mask=m_valid, other=0)
        y = tl.where(rmask[:, None] > 0, y, 0.0)
        tl.store(
            out_ptr + m_offs[:, None] * Cout + n_offs[None, :],
            y.to(tl.float16),
            mask=m_valid[:, None] & n_valid[None, :],
        )

    @torch.library.custom_op("hexfield::hex_conv_ln", mutates_args=())
    def hex_conv_ln(
        x: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        relu: bool,
    ) -> torch.Tensor:
        b, npad, c = x.shape
        cout = weight.shape[-1]
        x = x.contiguous()
        gidx = gather_idx.contiguous()
        m8 = mask.to(torch.uint8).contiguous()
        w16 = weight.reshape(7 * c, cout).to(torch.float16).contiguous()
        b32 = bias.to(torch.float32).contiguous()
        lnw = ln_weight.to(torch.float32).contiguous()
        lnb = ln_bias.to(torch.float32).contiguous()
        out = torch.empty((b, npad, cout), dtype=torch.float16, device=x.device)
        rows = b * npad
        BN = triton.next_power_of_2(cout)  # whole row per program (LN needs it)
        BM, BK = _LN_BM, 64
        grid = (triton.cdiv(rows, BM),)
        _hex_conv_ln_kernel[grid](
            x, gidx, m8, w16, b32, lnw, lnb, out,
            b, npad, c, cout, eps,
            IS_FP16_IN=(x.dtype == torch.float16), RELU=relu,
            BM=BM, BN=BN, BK=BK,
            num_warps=_LN_WARPS, num_stages=_LN_STAGES,
        )
        return out

    @hex_conv_ln.register_fake
    def _hex_conv_ln_fake(
        x, gather_idx, mask, weight, bias, ln_weight, ln_bias, eps, relu
    ):
        return x.new_empty(
            (x.shape[0], x.shape[1], weight.shape[-1]), dtype=torch.float16
        )

else:  # pragma: no cover
    hex_conv = None
    hex_conv_ln = None
