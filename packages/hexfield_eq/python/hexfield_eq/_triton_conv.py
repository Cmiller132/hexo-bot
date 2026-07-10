"""Fused gather+GEMM Triton kernel for HexNodeConv (serve-only, opt-in).

The reference HexNodeConv materializes a (B, Npad, 7*C) gathered tensor (cat a
zero row, gather, reshape) and feeds it to one GEMM. At serve shapes that
gather write+read is ~60% of the conv cost. This kernel gathers the 7 tap rows
directly into the GEMM's A-tiles (tl.dot, fp32 accumulate), so the expanded
tensor never exists; the missing-neighbour zero row becomes a masked load
(idx == Npad -> 0) and the output row mask is folded into the epilogue.

Exposed as the `hexfield_eq::hex_conv` custom op (with a fake kernel), so the
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
import warnings

import torch
import torch.nn.functional as F

try:  # pragma: no cover - triton ships with cuda torch builds
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

# Triton's CompilationError location moves between versions; import defensively.
# An empty tuple makes isinstance() always False (we still fall back on ANY
# exception — this only tunes the log message).
try:  # pragma: no cover - triton internals vary by version
    from triton.compiler.errors import CompilationError as _TritonCompileError
except Exception:  # pragma: no cover
    try:
        from triton.compiler import CompilationError as _TritonCompileError
    except Exception:
        _TritonCompileError = ()

# conv+LN kernel tile knobs (bench sweeps; defaults = measured winners at
# c=192, RTX 4070 Ti, 2026-07-03: BM=64/warps=4/stages=2 runs the fused kernel
# 30-37% FASTER than plain conv + eager LN; the first-cut BM=32/warps=8 was
# the worst point in the sweep, ~neutral vs unfused).
_LN_BM = int(os.environ.get("HEXFIELD_CONVLN_BM", "64"))
_LN_WARPS = int(os.environ.get("HEXFIELD_CONVLN_WARPS", "4"))
_LN_STAGES = int(os.environ.get("HEXFIELD_CONVLN_STAGES", "2"))


if HAVE_TRITON:

    # --- compile-failure fallback (cross-width eval hardening) --------------------
    # Some channel widths trip a per-arch Triton codegen edge case (observed:
    # c=96 fails to compile _hex_conv_kernel under triton 3.7.0 / torch 2.12,
    # while c=128 compiles fine). The reference gather+GEMM path is numerically
    # equivalent, so on ANY kernel-launch failure we memoize the specializing
    # shape and serve that shape from the reference path forever after — no retry
    # of the (failing, slow) compile on every forward. This lives INSIDE the
    # custom op: under the serve torch.compile(dynamic=True) graph the op is
    # opaque and its Triton compile happens when the op EXECUTES for a new shape,
    # not during dynamo tracing, so this is the only layer that catches it under
    # both eager and compiled serve. Keyed by (C, Cout) — the dims that drive the
    # kernel's tiling/codegen; a shape that compiles (c=128) never enters a set,
    # so its fast path is byte-for-byte unchanged.
    _TRITON_VER = getattr(triton, "__version__", "?")
    _CONV_FAILED: set = set()
    _CONV_LN_FAILED: set = set()

    def _mark_failed(failed: set, kernel: str, key, err: Exception) -> None:
        """Record a failing shape and warn ONCE (called only when key is new)."""
        failed.add(key)
        kind = (
            "compile error"
            if isinstance(err, _TritonCompileError)
            else f"{type(err).__name__}"
        )
        c, cout = key
        warnings.warn(
            f"hexfield: triton {kernel} failed to compile for C={c},Cout={cout} "
            f"({kind}) under triton {_TRITON_VER}; using the reference path for "
            f"this shape.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _conv_ref(x, gather_idx, mask, weight, bias):
        """Reference HexNodeConv (the no-flag path in model.py), fp16 out to match
        the custom op's fake kernel."""
        b, n, c = x.shape
        cout = weight.shape[-1]
        x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
        flat = gather_idx.to(torch.int64).reshape(b, n * 7, 1).expand(-1, -1, c)
        gathered = x_ext.gather(1, flat).reshape(b, n, 7 * c)
        out = gathered @ weight.reshape(7 * c, cout) + bias
        out = out * mask.unsqueeze(-1)
        return out.to(torch.float16)

    def _conv_ln_ref(x, gather_idx, mask, weight, bias, ln_w, ln_b, eps, relu):
        """Reference conv + LayerNorm(+ReLU) + row-mask (the ConvBlock no-flag
        path). LN stats in fp32 on the conv accumulator, matching the fused
        kernel; masked rows are zeroed last (pad rows only, so valid rows match)."""
        b, n, c = x.shape
        cout = weight.shape[-1]
        x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
        flat = gather_idx.to(torch.int64).reshape(b, n * 7, 1).expand(-1, -1, c)
        gathered = x_ext.gather(1, flat).reshape(b, n, 7 * c)
        conv = gathered @ weight.reshape(7 * c, cout) + bias
        y = F.layer_norm(conv.float(), (cout,), ln_w.float(), ln_b.float(), eps)
        if relu:
            y = F.relu(y)
        y = y * mask.unsqueeze(-1)
        return y.to(torch.float16)

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

    @torch.library.custom_op("hexfield_eq::hex_conv", mutates_args=())
    def hex_conv(
        x: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        b, npad, c = x.shape
        cout = weight.shape[-1]
        key = (c, cout)
        if key not in _CONV_FAILED:
            try:
                x = x.contiguous()
                gidx = gather_idx.contiguous()
                m8 = mask.to(torch.uint8).contiguous()
                w16 = weight.reshape(7 * c, cout).to(torch.float16).contiguous()
                b32 = bias.to(torch.float32).contiguous()
                out = torch.empty(
                    (b, npad, cout), dtype=torch.float16, device=x.device
                )
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
            except Exception as err:  # per-arch triton codegen edge case
                _mark_failed(_CONV_FAILED, "hex_conv", key, err)
        return _conv_ref(x, gather_idx, mask, weight, bias)

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

    @torch.library.custom_op("hexfield_eq::hex_conv_ln", mutates_args=())
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
        key = (c, cout)
        if key not in _CONV_LN_FAILED:
            try:
                x = x.contiguous()
                gidx = gather_idx.contiguous()
                m8 = mask.to(torch.uint8).contiguous()
                w16 = weight.reshape(7 * c, cout).to(torch.float16).contiguous()
                b32 = bias.to(torch.float32).contiguous()
                lnw = ln_weight.to(torch.float32).contiguous()
                lnb = ln_bias.to(torch.float32).contiguous()
                out = torch.empty(
                    (b, npad, cout), dtype=torch.float16, device=x.device
                )
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
            except Exception as err:  # per-arch triton codegen edge case
                _mark_failed(_CONV_LN_FAILED, "hex_conv_ln", key, err)
        return _conv_ln_ref(
            x, gather_idx, mask, weight, bias, ln_weight, ln_bias, eps, relu
        )

    @hex_conv_ln.register_fake
    def _hex_conv_ln_fake(
        x, gather_idx, mask, weight, bias, ln_weight, ln_bias, eps, relu
    ):
        return x.new_empty(
            (x.shape[0], x.shape[1], weight.shape[-1]), dtype=torch.float16
        )

    # NOTE: the fp8 (e4m3) conv variants were REMOVED for the equivariant v1
    # (docs/DERIVATION §2.3, "BUGS_FOUND"). Their weight cache was keyed on
    # id(weight), but the tied trunk regenerates a fresh dense weight object
    # every forward (HexNodeConv._materialize), so the cache would miss every
    # forward AND retain a strong ref to each regenerated weight — an unbounded
    # leak. There is no id()-safe re-key while the weight is regenerated per
    # forward, so the whole fp8 path is dropped rather than gated off.

    # --- K1: ray-tap conv + LN(+ReLU) + mask (SPEC_RAYTAP_CONV.md §2.4) ----------
    # The baseline kernels cannot consume ray-tap input (they gather the 7 tap
    # rows from gather_idx). This variant keeps tap 0 (center) as-is and runs
    # an inner k-loop for taps 1..6: 5 masked row loads + FMA per direction
    # (31 rows per query vs 7), consuming the sync-free ray gather index, the
    # per-(side, tap) reach, and the tiled alpha in-kernel — no (B, Npad, 7C)
    # materialization. Numerics class: the 5-term tap input accumulates fp32,
    # rounds once to fp16, then fp16 x fp16 tl.dot with fp32 accumulation (the
    # reference path's class; T5 pins parity at the serve tolerance).

    _CONV_LN_RAYTAP_FAILED: set = set()
    # Per-device cache of the (6,) int32 base ray-gather slot per direction tap
    # (= tap_ray_slot_lut()[:, 0]; slot of k=1 — k walks +1 per step).
    _RT_SLOT_BASE_DEV: dict = {}

    def _rt_slot_base_on(dev) -> torch.Tensor:
        sb = _RT_SLOT_BASE_DEV.get(dev)
        if sb is None:
            from ._raytap import tap_ray_slot_lut

            sb = tap_ray_slot_lut()[:, 0].to(device=dev, dtype=torch.int32)
            _RT_SLOT_BASE_DEV[dev] = sb
        return sb

    def _conv_ln_raytap_ref(
        x, gather_idx, mask, weight, bias, ln_w, ln_b, ray_idx, reach, alpha,
        eps, relu, corb,
    ):
        """Reference ray-tap conv + LN(+ReLU) + mask (the ConvBlock equipped-
        conv serve fallback): the _raytap masked-gather taps + one GEMM, LN
        stats fp32 on the conv output, fp16 store — _conv_ln_ref's numerics
        with the ray-tap gathered tensor."""
        from ._raytap import tap_ray_slot_lut, ray_tap_taps_naive

        b, n, c = x.shape
        cout = weight.shape[-1]
        lut = tap_ray_slot_lut().to(ray_idx.device)
        idx_taps = ray_idx.to(torch.int64)[:, :, lut]
        taps = ray_tap_taps_naive(x, idx_taps, reach, alpha.to(x.dtype), corb)
        gathered = torch.cat([x.unsqueeze(2), taps], dim=2).reshape(b, n, 7 * c)
        conv = gathered @ weight.reshape(7 * c, cout).to(x.dtype) + bias.to(x.dtype)
        y = F.layer_norm(conv.float(), (cout,), ln_w.float(), ln_b.float(), eps)
        if relu:
            y = F.relu(y)
        y = y * mask.unsqueeze(-1)
        return y.to(torch.float16)

    @triton.jit
    def _hex_conv_ln_raytap_kernel(
        x_ptr, idx_ptr, mask_ptr, w_ptr, bias_ptr, lnw_ptr, lnb_ptr,
        rayidx_ptr, reach_ptr, alpha_ptr, slotbase_ptr,
        out_ptr,
        B, Npad, C, Cout, eps,
        IS_FP16_IN: tl.constexpr, RELU: tl.constexpr,
        CORB: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        m_offs = pid_m * BM + tl.arange(0, BM)  # rows over B*Npad
        m_valid = m_offs < B * Npad
        b_ids = m_offs // Npad
        n_offs = tl.arange(0, BN)  # the whole Cout row (LN needs it)
        n_valid = n_offs < Cout

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        # Tap 0 (center): identical to the baseline kernel's t=0 body.
        idx0 = tl.load(idx_ptr + m_offs * 7 + 0, mask=m_valid, other=Npad)
        row0_ok = m_valid & (idx0 < Npad)
        x_row0 = (b_ids * Npad + idx0) * C
        for k0 in tl.range(0, tl.cdiv(C, BK)):
            k_offs = k0 * BK + tl.arange(0, BK)
            k_ok = k_offs < C
            a = tl.load(
                x_ptr + x_row0[:, None] + k_offs[None, :],
                mask=row0_ok[:, None] & k_ok[None, :],
                other=0.0,
            )
            a16 = a if IS_FP16_IN else a.to(tl.float16)
            w = tl.load(
                w_ptr + (0 * C + k_offs)[:, None] * Cout + n_offs[None, :],
                mask=k_ok[:, None] & n_valid[None, :],
                other=0.0,
            )
            acc += tl.dot(a16, w)

        # Taps 1..6 (directions): inner k-loop of 5 masked loads + FMA per
        # direction; per-channel-side visibility from the (B, N, 2, 6) reach.
        HALF: tl.constexpr = CORB // 2
        for t in tl.static_range(6):
            sb = tl.load(slotbase_ptr + t)
            rl_own = tl.load(
                reach_ptr + m_offs * 12 + 0 * 6 + t, mask=m_valid, other=0
            ).to(tl.int32)
            rl_opp = tl.load(
                reach_ptr + m_offs * 12 + 1 * 6 + t, mask=m_valid, other=0
            ).to(tl.int32)
            for k0 in tl.range(0, tl.cdiv(C, BK)):
                k_offs = k0 * BK + tl.arange(0, BK)
                k_ok = k_offs < C
                side_c = (k_offs % CORB) >= HALF  # (BK,) opp-half channels
                acc_tap = tl.zeros((BM, BK), dtype=tl.float32)
                for k in tl.static_range(5):
                    idx = tl.load(
                        rayidx_ptr + m_offs * 32 + sb + k, mask=m_valid, other=Npad
                    )
                    present = m_valid & (idx < Npad)
                    vo = rl_own >= (k + 1)
                    vp = rl_opp >= (k + 1)
                    vis_own = vo.to(tl.float32)
                    vis_opp = vp.to(tl.float32)
                    # Skip the row load when the cell is invisible to BOTH
                    # sides (its contribution is zero either way) — real
                    # raylen truncates most rays, so this sheds most of the
                    # 31-row load bill.
                    a = tl.load(
                        x_ptr + ((b_ids * Npad + idx) * C)[:, None] + k_offs[None, :],
                        mask=(present & (vo | vp))[:, None] & k_ok[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    alpha_k = tl.load(
                        alpha_ptr + k * C + k_offs, mask=k_ok, other=0.0
                    ).to(tl.float32)
                    vis = tl.where(
                        side_c[None, :], vis_opp[:, None], vis_own[:, None]
                    )
                    acc_tap += a * (alpha_k[None, :] * vis)
                a_tap16 = acc_tap.to(tl.float16)
                w = tl.load(
                    w_ptr + ((1 + t) * C + k_offs)[:, None] * Cout + n_offs[None, :],
                    mask=k_ok[:, None] & n_valid[None, :],
                    other=0.0,
                )
                acc += tl.dot(a_tap16, w)

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

    @torch.library.custom_op("hexfield_eq::hex_conv_ln_raytap", mutates_args=())
    def hex_conv_ln_raytap(
        x: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        ray_idx: torch.Tensor,
        reach: torch.Tensor,
        alpha: torch.Tensor,
        eps: float,
        relu: bool,
        corb: int,
    ) -> torch.Tensor:
        """Fused ray-tap conv + LN(+ReLU) + mask (K1). ``ray_idx`` (B, Npad,
        32) int32 sync-free gather index; ``reach`` (B, Npad, 2, 6) u8
        per-(side, tap) visibility reach; ``alpha`` (5, C) the slot-constant
        tiled reach profile; ``corb`` the orbit width whose halves the own/opp
        visibility split rides. Falls back to the reference on any
        compile/launch failure (memoized per (C, Cout))."""

        b, npad, c = x.shape
        cout = weight.shape[-1]
        key = (c, cout)
        if key not in _CONV_LN_RAYTAP_FAILED:
            try:
                xc = x.contiguous()
                gidx = gather_idx.contiguous()
                m8 = mask.to(torch.uint8).contiguous()
                w16 = weight.reshape(7 * c, cout).to(torch.float16).contiguous()
                b32 = bias.to(torch.float32).contiguous()
                lnw = ln_weight.to(torch.float32).contiguous()
                lnb = ln_bias.to(torch.float32).contiguous()
                ridx = ray_idx.contiguous()
                rch = reach.contiguous()
                a16 = alpha.to(torch.float16).contiguous()
                sb = _rt_slot_base_on(x.device)
                out = torch.empty(
                    (b, npad, cout), dtype=torch.float16, device=x.device
                )
                rows = b * npad
                BN = triton.next_power_of_2(cout)
                BM, BK = _LN_BM, 64
                grid = (triton.cdiv(rows, BM),)
                _hex_conv_ln_raytap_kernel[grid](
                    xc, gidx, m8, w16, b32, lnw, lnb,
                    ridx, rch, a16, sb, out,
                    b, npad, c, cout, eps,
                    IS_FP16_IN=(x.dtype == torch.float16), RELU=relu,
                    CORB=corb,
                    BM=BM, BN=BN, BK=BK,
                    num_warps=_LN_WARPS, num_stages=_LN_STAGES,
                )
                return out
            except Exception as err:  # per-arch triton codegen edge case
                _mark_failed(_CONV_LN_RAYTAP_FAILED, "hex_conv_ln_raytap", key, err)
        return _conv_ln_raytap_ref(
            x, gather_idx, mask, weight, bias, ln_weight, ln_bias,
            ray_idx, reach, alpha, eps, relu, corb,
        )

    @hex_conv_ln_raytap.register_fake
    def _hex_conv_ln_raytap_fake(
        x, gather_idx, mask, weight, bias, ln_weight, ln_bias,
        ray_idx, reach, alpha, eps, relu, corb,
    ):
        return x.new_empty(
            (x.shape[0], x.shape[1], weight.shape[-1]), dtype=torch.float16
        )

else:  # pragma: no cover
    hex_conv = None
    hex_conv_ln = None
    hex_conv_ln_raytap = None
