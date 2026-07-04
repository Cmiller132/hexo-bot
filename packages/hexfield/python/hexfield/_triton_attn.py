"""Fused rel-pos-bias attention Triton kernel (serve-only, opt-in).

FlashAttention-2-style online-softmax kernel specialized for the flex-pair
serve path: the per-pair bias-table ROW INDEX (the shared (B, S, S) uint8
`pair` tensor, pad-KEY columns = the appended pad row) is gathered inside the
score loop as ONE 1-byte load + one read of the tiny (BIAS_ROWS + 1, heads)
fp16 table — the same math as the flex score_mod, but in a kernel shaped for
this workload instead of flex's generic lowering:

  * head_dim is a constexpr tile dimension (main_7's d=64 hits the tensor-core
    sweet spot; flex at d=32 ran ~10 TFLOPS on Ada),
  * the key loop is bounded per batch row by `seq_lens` (tokens + last live
    cell), so fully-padded key tiles are never touched. Flex pays full S^2 for
    pad keys (measured: the WASTE_FRACTION raise regressed -43% for exactly
    this reason); here pad tail costs zero.

Skipped key tiles are numerically identical to flex's -3e4-biased pad keys:
exp(-3e4 - m) underflows to exactly 0.0 in fp32, contributing nothing to the
softmax sum. Pad keys INSIDE the seq_lens bound (non-contiguous padding,
partial tiles) still get the pad-row bias via `pair`, so correctness never
depends on padding being contiguous — only tile-skipping efficiency does.

Exposed as the `hexfield::attn_pair` custom op (with a fake kernel) so the
serve torch.compile graph keeps it in-graph as an opaque call. Enabled via
HEXFIELD_TRITON_ATTN=1; model.py routes to it from RelPosAttention.forward on
the no-grad CUDA fp16 path when the block bias is a _FlexPairBias carrying
seq_lens, for head_dim in {32, 64}. Everything else falls through to flex.

Numerics: fp16 QK^T and PV tensor-core dots with fp32 accumulation, fp32
online softmax — the same accumulation class as flex_attention; output fp16.

Tile constants come from env (HEXFIELD_ATTN_BM/BN/WARPS/STAGES, read once at
import) so the bench matrix can sweep them; defaults are the measured winners.
"""

from __future__ import annotations

import math
import os

import torch

try:  # pragma: no cover - triton ships with cuda torch builds
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

_BM = int(os.environ.get("HEXFIELD_ATTN_BM", "64"))
_BN = int(os.environ.get("HEXFIELD_ATTN_BN", "64"))
_WARPS = int(os.environ.get("HEXFIELD_ATTN_WARPS", "4"))
_STAGES = int(os.environ.get("HEXFIELD_ATTN_STAGES", "3"))


if HAVE_TRITON:

    @triton.jit
    def _attn_pair_kernel(
        q_ptr, k_ptr, v_ptr, pair_ptr, tab_ptr, seq_ptr, out_ptr,
        sqb, sqh, sqs,
        skb, skh, sks,
        svb, svh, svs,
        sob, soh, sos,
        spb, sps,
        H, S, scale,
        PAD_ROW: tl.constexpr,
        D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        lo_m = pid_m * BM
        offs_m = lo_m + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        m_ok = offs_m < S
        o_ptrs = out_ptr + b * sob + h * soh + offs_m[:, None] * sos + offs_d[None, :]

        n_keys = tl.load(seq_ptr + b)
        # Whole q tile beyond the live sequence: downstream multiplies attention
        # output by seq_mask, so pad rows only need to be finite. Store zeros.
        if lo_m >= n_keys:
            tl.store(
                o_ptrs, tl.zeros((BM, D), dtype=tl.float16), mask=m_ok[:, None]
            )
            return

        q = tl.load(
            q_ptr + b * sqb + h * sqh + offs_m[:, None] * sqs + offs_d[None, :],
            mask=m_ok[:, None],
            other=0.0,
        )
        m_i = tl.full((BM,), float("-inf"), tl.float32)
        l_i = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, D), tl.float32)
        LOG2E: tl.constexpr = 1.4426950408889634

        for start_n in range(0, n_keys, BN):
            offs_n = start_n + tl.arange(0, BN)
            kv_ok = offs_n < n_keys
            k = tl.load(
                k_ptr + b * skb + h * skh + offs_n[:, None] * sks + offs_d[None, :],
                mask=kv_ok[:, None],
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(k)) * scale
            # Bias row index; out-of-bounds -> pad row -> -3e4 -> exp == 0.
            pr = tl.load(
                pair_ptr + b * spb + offs_m[:, None] * sps + offs_n[None, :],
                mask=m_ok[:, None] & kv_ok[None, :],
                other=PAD_ROW,
            )
            qk += tl.load(tab_ptr + pr.to(tl.int32) * H + h).to(tl.float32)
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2((qk - m_new[:, None]) * LOG2E)
            alpha = tl.math.exp2((m_i - m_new) * LOG2E)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(
                v_ptr + b * svb + h * svh + offs_n[:, None] * svs + offs_d[None, :],
                mask=kv_ok[:, None],
                other=0.0,
            )
            acc += tl.dot(p.to(tl.float16), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        tl.store(o_ptrs, acc.to(tl.float16), mask=m_ok[:, None])

    @torch.library.custom_op("hexfield::attn_pair", mutates_args=())
    def attn_pair(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair: torch.Tensor,
        table2: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        b, h, s, d = q.shape
        # (B,S,H,D).transpose(1,2) views satisfy stride(-1)==1 already; the
        # kernel takes the remaining strides as-is, so no copies here.
        if q.stride(-1) != 1:
            q = q.contiguous()
        if k.stride(-1) != 1:
            k = k.contiguous()
        if v.stride(-1) != 1:
            v = v.contiguous()
        pair = pair.contiguous()
        tab = table2.to(torch.float16).contiguous()
        seq = seq_lens.to(torch.int32).contiguous()
        out = torch.empty((b, h, s, d), dtype=torch.float16, device=q.device)
        grid = (triton.cdiv(s, _BM), b * h)
        _attn_pair_kernel[grid](
            q, k, v, pair, tab, seq, out,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            pair.stride(0), pair.stride(1),
            h, s, 1.0 / math.sqrt(d),
            PAD_ROW=tab.shape[0] - 1,
            D=d, BM=_BM, BN=_BN,
            num_warps=_WARPS, num_stages=_STAGES,
        )
        return out

    @attn_pair.register_fake
    def _attn_pair_fake(q, k, v, pair, table2, seq_lens):
        return q.new_empty(q.shape, dtype=torch.float16)

else:  # pragma: no cover
    attn_pair = None
