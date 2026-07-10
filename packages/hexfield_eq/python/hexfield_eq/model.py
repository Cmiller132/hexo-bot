"""hexfield network: trunk, attention/bias, heads.

Trunk block order C C C A C C C A C C A over variable-N node sets: post-activation
conv residual blocks with LayerNorm, and 3 pre-norm transformer blocks over the
joint sequence [8 summary tokens ; cells], each with its own learned
BIAS_ROWS-row relative-position bias table. HexNodeConv is a gather + one GEMM.

Batch conventions (built by `batching.py`):
- feats  (B, Npad, F) f32; pad rows all-zero
- nbr    (B, Npad, 6) long, row-local; missing/pad -> Npad (the appended
  zero row, giving conv zero-padding semantics)
- mask   (B, Npad) bool, True at real nodes
- coords (B, Npad, 2) long axial (q, r); pad coords arbitrary (not read
  through the bias: pad KEY columns are additively masked and pad QUERY rows
  are re-zeroed after every block)

Convs and attention re-apply the node mask after every parameter-carrying op,
so a row's outputs do not depend on how much padding shares its batch.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .constants import (
    ATTENTION_HEADS,
    BIAS_CELL_TOKEN_ROW,
    BIAS_DISK_RADIUS,
    BIAS_FAR_ROW,
    BIAS_OFF_AXIS_BASE,
    BIAS_ON_AXIS_BASE,
    BIAS_RING_MAX,
    BIAS_RING_MIN,
    BIAS_ROWS,
    BIAS_TOKEN_CELL_ROW,
    BIAS_TOKEN_TOKEN_ROW,
    CHANNELS,
    C_ORBIT,
    FEATURE_VERSION,
    GROUP_ORDER,
    HEAD_DIM,
    MLP_RATIO,
    NUM_FEATURES,
    NUM_TOKENS,
    RAY_BLOCKERS,
    RAY_HEADS,
    RAY_REACH,
    RAYTAP,
    REG_LANE,
    REG_TOK_READ,
    TRUNK_LAYOUT,
    VALUE_BINS,
)
from .geometry import BIAS_FREE_ROWS, bias_orbit_of_row, rel_bias_index
from .support import _SUPPORT_RADIUS

# The full D6 (order-12) regular-representation tie is the default build; the
# equivariant primitives live in equivariant.py and are imported only when the
# tie is active. GROUP_ORDER == 1 is the non-equivariant passthrough (the copied
# dense trunk + the Phase-6 A/B ablation). constants.py already rejects the
# reserved GROUP_ORDER == 6.
EQUIVARIANT = GROUP_ORDER == 12
if EQUIVARIANT:
    from . import equivariant as _eq

# Additive pad-key mask value; finite in fp16.
PAD_KEY_MASK_VALUE = -3.0e4

STV_HORIZONS = (2, 6, 16)

# Invariant-read widening (spec D-S20): the fiber-invariant head reads expand
# the C-fiber by these factors BEFORE the group-pool, so each pooled read block
# is EXPAND*C_ORBIT wide (64 for the value/aux/ml reads at C_ORBIT=16, 32 for
# the per-cell policy reads) instead of the bare C_ORBIT=16 bottleneck.
# Constants, not env knobs: they shape the state dict.
INV_READ_EXPAND = 4
POLICY_READ_EXPAND = 2

# FlexAttention serve path, enabled when HEXFIELD_SERVE_FLEX=1 (default off). The
# no-grad serve forward computes the rel-pos bias inside the attention kernel via
# a score_mod (coords + _cell_bias_lut + bias_table gather + pad-key fill) instead
# of materializing the (B, heads, S, S) bias. Read once at import; the import is
# guarded so a torch without flex still loads.
_SERVE_FLEX = os.environ.get("HEXFIELD_SERVE_FLEX") == "1"
# FlexAttention training path, enabled when HEXFIELD_TRAIN_FLEX=1 (default off).
# The grad-enabled forward uses the same per-block score_mod, but the carrier
# passes the fp32 master bias_tables[i] (not the fp16 serve cast) so the table's
# gradient accumulates in fp32. With both flags off the materialized path runs.
_TRAIN_FLEX = os.environ.get("HEXFIELD_TRAIN_FLEX") == "1"
# Precomputed-pair flex serve variant, HEXFIELD_FLEX_PAIR=1 (default off; only
# applies on the serve-flex no-grad path). The per-pair bias-table ROW INDEX is
# materialized once per forward as a (B, S, S) uint8 tensor shared by all 3
# attention blocks, with the pad-KEY fill folded in as an extra table row
# (row BIAS_ROWS = PAD_KEY_MASK_VALUE). The score_mod then does ONE 1-byte
# gather + a tiny (BIAS_ROWS+1, heads) fp16 table read per score, instead of
# two int64 coord gathers + an int64 LUT gather + branchy region selects. The
# uint8 pair tensor is bounded by the serve pair ceiling (B*S^2 bytes ~ 38 MB),
# 8x smaller than the old materialized fp16 (B, heads, S, S) bias.
_FLEX_PAIR = os.environ.get("HEXFIELD_FLEX_PAIR") == "1"
# Train-side flex-pair, HEXFIELD_TRAIN_FLEX_PAIR=1 (default off; requires
# train-flex). Same precomputed uint8 pair as the serve variant, but the
# appended-pad-row table2 is built from the FP32 master bias_tables[i] (no
# fp16 cast), so the table gradient flows through the table2 gather and
# accumulates in fp32 exactly like the plain train-flex carrier. Gated
# separately from the serve flag so the two paths roll out independently.
_TRAIN_FLEX_PAIR = os.environ.get("HEXFIELD_TRAIN_FLEX_PAIR") == "1"
# Fused gather+GEMM Triton conv, HEXFIELD_TRITON_CONV=1 (default off). Serve
# (no-grad, CUDA) only; the (B, Npad, 7C) gathered tensor is never built. See
# _triton_conv.py. The import is guarded so a torch without triton still loads.
_TRITON_CONV = os.environ.get("HEXFIELD_TRITON_CONV") == "1"
if _TRITON_CONV:
    try:
        from ._triton_conv import hex_conv as _hex_conv_fused
    except Exception:  # pragma: no cover - no triton
        _hex_conv_fused = None
else:
    _hex_conv_fused = None
# Bespoke fused attention, HEXFIELD_TRITON_ATTN=1 (default off; requires the
# flex-pair serve path for the uint8 pair index + appended-pad-row table).
# FA2-style online-softmax kernel with the pair bias gathered in the score
# loop and fully-padded key tiles skipped via per-row seq_lens. Serve
# (no-grad, CUDA, fp16 q/k/v) only; see _triton_attn.py.
_TRITON_ATTN = os.environ.get("HEXFIELD_TRITON_ATTN") == "1"
if _TRITON_ATTN:
    try:
        from ._triton_attn import attn_pair as _attn_pair_fused
    except Exception:  # pragma: no cover - no triton
        _attn_pair_fused = None
else:
    _attn_pair_fused = None
# Gathered ray attention for L blocks, HEXFIELD_EQ_TRITON_RAY=1 (default off;
# spec D-S36/D-S37). Serve (no-grad, CUDA, fp16 q/k/v) only: each query cell
# attends its <= 31 geometric ray cells through a (B, Npad, 32) int32 gather
# index built once per forward (shared by every L block); per-head raylen
# gating and the slot-resolved joint bias run inside the kernel. Any miss
# (grad, CPU, fp32, foreign head_dim) falls through to the flex/materialized
# paths; see _triton_ray.py. The index build is sync-free (sort/searchsorted
# join, 2026-07-09) and CUDA-graph capturable in principle; co-enabling it with
# the graphs serve path still needs its own parity/throughput validation.
_TRITON_RAY = os.environ.get("HEXFIELD_EQ_TRITON_RAY") == "1"
if _TRITON_RAY:
    try:
        from ._triton_ray import build_ray_gather_index as _ray_gather_index_fused
        from ._triton_ray import ray_attn as _ray_attn_fused
        from ._triton_ray import slot_bias_rows as _ray_slot_bias_rows
    except Exception:  # pragma: no cover - no triton/custom-op support
        _ray_attn_fused = None
        _ray_gather_index_fused = None
        _ray_slot_bias_rows = None
else:
    _ray_attn_fused = None
    _ray_gather_index_fused = None
    _ray_slot_bias_rows = None
# Ray-tap conv support (SPEC_RAYTAP_CONV.md §2). Imported LAZILY on the first
# construction of a ray-tap-equipped conv (env HEXFIELD_EQ_RAYTAP or the
# constructor kwarg — a foreign checkpoint can enable it under a default env),
# so the default build's import graph is untouched (live-run isolation, §9.1).
_raytap_mod = None


def _raytap():
    global _raytap_mod
    if _raytap_mod is None:
        from . import _raytap as _m

        _raytap_mod = _m
    return _raytap_mod


# Conv + LayerNorm(+ReLU) + mask epilogue fusion, HEXFIELD_TRITON_CONV_LN=1
# (default off). ConvBlock's post-conv LayerNorm epilogue runs inside the
# fused conv kernel on the fp32 accumulator, saving one full read+write of
# the (B, Npad, C) activation per conv. Serve (no-grad, CUDA) only.
_TRITON_CONV_LN = os.environ.get("HEXFIELD_TRITON_CONV_LN") == "1"
if _TRITON_CONV_LN:
    try:
        from ._triton_conv import hex_conv_ln as _hex_conv_ln_fused
    except Exception:  # pragma: no cover - no triton
        _hex_conv_ln_fused = None
    # K1 (SPEC_RAYTAP_CONV.md §2.4): the ray-tap variant rides the same env
    # gate — equipped convs take it on the fused serve branch when present,
    # else the reference path (labeled reference-path throughput).
    try:
        from ._triton_conv import hex_conv_ln_raytap as _hex_conv_ln_raytap_fused
    except Exception:  # pragma: no cover - no triton
        _hex_conv_ln_raytap_fused = None
else:
    _hex_conv_ln_fused = None
    _hex_conv_ln_raytap_fused = None
# fp8 (e4m3) conv GEMMs were REMOVED for the equivariant v1 (docs/DERIVATION
# §2.3, "BUGS_FOUND"): the fused-conv fp8 weight cache was keyed on id(weight),
# but the tied trunk regenerates a fresh dense weight object every forward
# (HexNodeConv._materialize), so the cache would miss every forward and leak a
# strong ref to each regenerated weight. The custom ops and their cache are gone
# from _triton_conv.py; ConvBlock's serve path uses the plain fp16 conv+LN
# fusion below (its per-channel-affine epilogue, fed the orbit-tied GroupAffineNorm
# affine, is exactly the equivariant group-norm — see ConvBlock.forward).
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    # Inner-compiled flex_attention. _flex_call (below) is torch.compiler.disable'd
    # so the outer torch.compile(dynamic=True) serve graph breaks at the attention
    # and the flex op compiles in its own inner graph.
    #
    # dynamic=False: the score_mod does data-dependent indexing (coords[b, kc],
    # table[row, h]) which inductor cannot lower under dynamic shapes, so flex
    # specializes per distinct (batch, Npad) serve shape.
    _flex_compiled = torch.compile(_flex_attention, dynamic=False)
    # SHAPE-KEYED compile instances (2026-07-03): a single compiled callable
    # holds every (B, S) specialization in ONE dynamo cache, and dynamo resolves
    # hits by linearly scanning guard-sets — with hundreds of live serve shapes
    # that scan runs on every flex call (3 per group, ~20 groups per flush) and
    # dominates the serve submit phase. Keying a separate torch.compile
    # instance per exact q-shape makes the lookup a dict hit with a 1-entry
    # guard chain; the inductor artifact is shared via the code cache, so the
    # per-shape compile cost is unchanged.
    _flex_by_shape: dict = {}

    # Each serve shape gets its own specialization. dynamo's recompile limit is
    # raised so the bounded set of serve shapes (batch <= active_limit, Npad
    # bucketed) each keeps its fused kernel rather than falling back to eager flex.
    try:
        import torch._dynamo as _dynamo

        _dynamo.config.recompile_limit = max(
            getattr(_dynamo.config, "recompile_limit", 8), 512
        )
    except Exception:  # pragma: no cover - older torch
        pass

    # Grad-path flex block sizes (2026-07-03, main_7 bring-up): at d=64 EVERY
    # default flex train config wants 147456B shared memory vs Ada's 101376B
    # limit -> "No valid triton configs" -> dynamo falls back to eager PER
    # MICROBUCKET SHAPE and training runs ~9-10 s/step (measured live,
    # main_7 epochs 1-2). These explicit blocks fit on Ada at every probed
    # shape (B<=48, S<=648) and beat the H100-tuned defaults where those do
    # compile. Serve-path flex is untouched (no-grad calls pass None).
    # HEXFIELD_TRAIN_FLEX_SMALL_BLOCKS=0 reverts.
    if os.environ.get("HEXFIELD_TRAIN_FLEX_SMALL_BLOCKS", "1") == "1":
        _TRAIN_FLEX_KOPTS = {
            "BLOCK_M": 64, "BLOCK_N": 32,
            "BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32,
        }
    else:
        _TRAIN_FLEX_KOPTS = None

    @torch.compiler.disable(recursive=False)
    def _flex_call(q, k, v, score_mod):
        key = (
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],
            q.dtype, q.requires_grad,
        )
        fn = _flex_by_shape.get(key)
        if fn is None:
            fn = torch.compile(_flex_attention, dynamic=False)
            _flex_by_shape[key] = fn
        kopts = _TRAIN_FLEX_KOPTS if q.requires_grad else None
        return fn(q, k, v, score_mod=score_mod, kernel_options=kopts)

except Exception:  # pragma: no cover - older torch without flex
    _flex_attention = None
    _flex_call = None


class _FlexBias:
    """Carrier for the flex attention path. Built once per block in trunk() (each
    block gets its own bias_tables[i]) and passed in place of the materialized
    attn_bias tensor; RelPosAttention.forward detects it and routes to
    flex_attention.

    Holds the raw inputs the score_mod needs (coords, mask, bias table, cell LUT),
    not a pre-built closure. The closure is constructed in RelPosAttention.forward
    (same frame as the flex call) and invoked through the disable'd _flex_call. No
    (B, heads, S, S) tensor is materialized."""

    __slots__ = ("coords", "mask", "table", "lut", "m", "w")

    def __init__(self, coords, mask, table, lut, m) -> None:
        self.coords = coords
        self.mask = mask
        self.table = table
        self.lut = lut
        self.m = m
        self.w = 2 * m + 1

    def make_score_mod(self):
        """Build the flex score_mod closure (called inside RelPosAttention.forward,
        the same frame as the flex_attention call). Computes the additive bias
        build_attn_bias adds (coords + _cell_bias_lut + bias_table gather) plus the
        pad-KEY additive fill (PAD_KEY_MASK_VALUE) folded in via the bool mask."""

        nt = NUM_TOKENS
        coords = self.coords
        mask = self.mask
        table = self.table
        lut = self.lut
        m = self.m
        w = self.w
        pad_fill = PAD_KEY_MASK_VALUE

        def score_mod(score, b, h, q_idx, kv_idx):
            qc = torch.clamp(q_idx - nt, min=0)
            kc = torch.clamp(kv_idx - nt, min=0)
            dq = coords[b, kc, 0] - coords[b, qc, 0]
            dr = coords[b, kc, 1] - coords[b, qc, 1]
            qi = torch.clamp(dq, -m, m) + m
            ri = torch.clamp(dr, -m, m) + m
            cell_idx = lut[qi * w + ri]
            q_tok = q_idx < nt
            k_tok = kv_idx < nt
            row = torch.where(
                q_tok & k_tok,
                torch.full_like(cell_idx, BIAS_TOKEN_TOKEN_ROW),
                torch.where(
                    q_tok & ~k_tok,
                    torch.full_like(cell_idx, BIAS_TOKEN_CELL_ROW),
                    torch.where(
                        ~q_tok & k_tok,
                        torch.full_like(cell_idx, BIAS_CELL_TOKEN_ROW),
                        cell_idx,
                    ),
                ),
            )
            biased = score + table[row, h].to(score.dtype)
            # pad-KEY columns: a cell key (kv_idx >= nt) whose row's mask is False.
            is_pad_key = (kv_idx >= nt) & ~mask[b, kc]
            return torch.where(is_pad_key, biased + pad_fill, biased)

        return score_mod


class _FlexPairBias:
    """Carrier for the precomputed-pair flex serve path (HEXFIELD_FLEX_PAIR=1).

    `pair` (B, S, S) uint8 is the per-pair bias-table row index, built ONCE per
    forward (block-independent) with pad-KEY columns set to the extra pad row;
    `table2` (BIAS_ROWS + 1, heads) fp16 is this block's bias table with the
    PAD_KEY_MASK_VALUE row appended. The score_mod is one gather + one tiny
    table read; no mask/coords/LUT work in-kernel.

    `seq_lens` (B,) int32 = NUM_TOKENS + last-live-cell-index + 1, set only on
    the serve path when the bespoke Triton attention kernel is active — it
    bounds that kernel's key loop so fully-padded key tiles are skipped. None
    on the flex/train paths (the score_mod never reads it)."""

    __slots__ = ("pair", "table2", "seq_lens")

    def __init__(self, pair, table2, seq_lens=None) -> None:
        self.pair = pair
        self.table2 = table2
        self.seq_lens = seq_lens

    def make_score_mod(self):
        pair = self.pair
        table2 = self.table2

        def score_mod(score, b, h, q_idx, kv_idx):
            row = pair[b, q_idx, kv_idx].to(torch.int32)
            return score + table2[row, h].to(score.dtype)

        return score_mod


class _FlexRayBias:
    """Carrier for the ray-attention (L block) flex path. Cells-only: q_idx /
    kv_idx map straight to cells (no token clamping). The score_mod computes the
    joint-tied table bias plus the live-ray test of plan L2 from (coords,
    raylen): pair (i, j) is live for head ``2*coset + side`` iff i == j (self,
    always live) or (dq, dr) is coset-axis-aligned with signed magnitude kk,
    1 <= |kk| <= RAY_REACH and, with blockers on, |kk| <= raylen[i, side, coset,
    sign(kk)]. Dead pairs and pad keys get the additive PAD_KEY_MASK_VALUE.
    The flex-pair / bespoke Triton attention paths never apply to L blocks in
    v1 (Phase L3); this plain carrier serves both the train and serve flex
    modes (fp32 master table on the grad path, fp16 cast on serve)."""

    __slots__ = ("coords", "mask", "raylen", "table", "lut", "m", "w", "blockers")

    def __init__(self, coords, mask, raylen, table, lut, m, blockers) -> None:
        self.coords = coords
        self.mask = mask
        self.raylen = raylen
        self.table = table
        self.lut = lut
        self.m = m
        self.w = 2 * m + 1
        self.blockers = blockers

    def make_score_mod(self):
        coords = self.coords
        mask = self.mask
        raylen = self.raylen
        table = self.table
        lut = self.lut
        m = self.m
        w = self.w
        blockers = self.blockers
        pad_fill = PAD_KEY_MASK_VALUE

        def score_mod(score, b, h, q_idx, kv_idx):
            dq = coords[b, kv_idx, 0] - coords[b, q_idx, 0]
            dr = coords[b, kv_idx, 1] - coords[b, q_idx, 1]
            qi = torch.clamp(dq, -m, m) + m
            ri = torch.clamp(dr, -m, m) + m
            row = lut[qi * w + ri]
            coset = h // 2
            side = h % 2
            kk = torch.where(coset == 1, dr, dq)
            aligned = torch.where(
                coset == 0, dr == 0, torch.where(coset == 1, dq == 0, dq == -dr)
            )
            live = aligned & (kk != 0) & (kk.abs() <= RAY_REACH)
            if blockers:
                slot = side * 6 + coset * 2 + torch.where(kk > 0, 0, 1)
                live = live & (kk.abs() <= raylen[b, q_idx, slot].to(kk.dtype))
            live = live | (q_idx == kv_idx)
            dead = ~live | ~mask[b, kv_idx]
            biased = score + table[row, h].to(score.dtype)
            return torch.where(dead, biased + pad_fill, biased)

        return score_mod


def _cuda_autocast_fp16() -> bool:
    """True when cuda fp16 autocast is active: the serve evaluator's autocast
    mode keeps the LN'd trunk stream fp32, but attention q/k/v (F.linear
    outputs) land fp16, so the gathered ray kernel can still engage."""

    try:
        return bool(
            torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
        )
    except TypeError:  # pragma: no cover - older torch (zero-arg CUDA API)
        return bool(
            torch.is_autocast_enabled()
            and torch.get_autocast_gpu_dtype() == torch.float16
        )


class _RayGatherBias:
    """Carrier for the gathered ray-attention Triton kernel
    (HEXFIELD_EQ_TRITON_RAY=1; spec D-S36/D-S37). Built by trunk() only on the
    no-grad CUDA fp16 serve path at a supported head_dim:

    - ``idx`` (B, N, 32) int32 — the block-independent geometric gather index
      (slot 0 = self, slots 1..30 the fixed (axis, dir, k) ray offsets, the
      sentinel N = absent key), built once per forward;
    - ``slot_bias`` (32, RAY_HEADS) fp16 — this block's expanded ray bias
      table collapsed to one row per slot (each slot's relative offset is
      fixed, so no in-kernel LUT gather);
    - ``raylen`` (B, N, RAYLEN_SLOTS) u8 — the wire buffer, read per head for
      the blocker gating; an empty dummy when ``blockers`` is False (geometric
      rays never read it, D-S16);
    - ``seq_lens`` (B,) int32 — last live cell + 1, bounding the kernel's
      query tiles so fully-padded tiles store zeros and skip every gather.

    RayAttention.forward routes this to the hexfield_eq::ray_attn custom op,
    which itself falls back to a gathered torch reference on any residual miss
    (memoized compile failure, foreign dtype)."""

    __slots__ = ("idx", "slot_bias", "raylen", "seq_lens", "blockers")

    def __init__(self, idx, slot_bias, raylen, seq_lens, blockers) -> None:
        self.idx = idx
        self.slot_bias = slot_bias
        self.raylen = raylen
        self.seq_lens = seq_lens
        self.blockers = blockers


class _RayTapCtx:
    """Per-forward carrier for ray-tap convs (SPEC_RAYTAP_CONV.md §2.5), built
    ONCE by trunk() and shared by every equipped conv:

    - ``idx_taps`` (B, Npad, 6, 5) int64 — row index of the cell at offset
      ``k * DIRECTIONS[t]`` (sentinel Npad = absent -> the zero row), sliced
      from the sync-free ray gather index by the generated tap -> slot LUT;
    - ``reach``   (B, Npad, 2, 6) u8 — per-(side, tap) visibility reach,
      gathered from the raylen wire buffer by the generated raylen-slot LUT;
    - ``ray_idx`` (B, Npad, 32) int32 and ``raylen`` (B, Npad, 12) u8 — the
      raw buffers, for the K1 fused kernel (which does the slot arithmetic
      in-kernel instead of consuming the widened views)."""

    __slots__ = ("idx_taps", "reach", "ray_idx", "raylen")

    def __init__(self, idx_taps, reach, ray_idx, raylen) -> None:
        self.idx_taps = idx_taps
        self.reach = reach
        self.ray_idx = ray_idx
        self.raylen = raylen


class _BiasGather(torch.autograd.Function):
    """table[pair] with a histogram backward.

    Forward is a plain gather. Backward computes the table gradient as a per-head
    bincount over the BIAS_ROWS destination classes rather than a scatter-add."""

    @staticmethod
    def forward(ctx, table: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(pair)
        ctx.rows = table.shape[0]
        return table[pair]

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (pair,) = ctx.saved_tensors
        flat = pair.reshape(-1)
        g = grad.reshape(-1, grad.shape[-1])
        acc = torch.float64 if grad.dtype == torch.float64 else torch.float32
        cols = [
            torch.bincount(flat, weights=g[:, h].to(acc), minlength=ctx.rows)
            for h in range(g.shape[1])
        ]
        return torch.stack(cols, dim=1).to(grad.dtype), None


class HexNodeConv(nn.Module):
    """Direction-typed 7-tap hex convolution: gather (B,N,7,Cin) -> one GEMM.

    Tap 0 = center; taps 1-6 = the fixed direction order D (the rotate60 orbit of
    (1,0)).

    Equivariant build (GROUP_ORDER == 12): the dense ``(7, C_in, C_out)`` weight
    is NOT a free parameter — it is materialized each forward by a pure
    index-gather from small orbit "base" params (docs/DERIVATION §2), so it
    survives the SERVE_HALF deepcopy and CUDA-graph capture and the reference
    GEMM / fp16 Triton kernels below consume it unchanged. Two kinds:
      * ``stem`` (C_in == NUM_FEATURES): typed-lift from the 25-plane input rep
        into the regular fiber (Reynolds projection, §8);
      * ``regular`` (C_in a regular fiber): the tied HexNodeConv, w_base
        ``(7, 12, C_orbit_out, C_orbit_in)`` (§2.3).
    A dense-weight cache keyed on the base-param ``_version`` regenerates the
    frozen-serve weight once rather than per forward.
    """

    def __init__(
        self, in_channels: int, out_channels: int, raytap: bool = False
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.equivariant = EQUIVARIANT
        self.raytap = bool(raytap)
        if not self.equivariant:
            self.weight = nn.Parameter(torch.empty(7, in_channels, out_channels))
            self.bias = nn.Parameter(torch.empty(out_channels))
            # Uniform init with fan_in = 7 * C_in.
            fan_in = 7 * in_channels
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.weight, -bound, bound)
            nn.init.uniform_(self.bias, -bound, bound)
            self._init_raytap()
            return
        # --- equivariant tied conv ---
        if out_channels % GROUP_ORDER != 0:
            raise ValueError(f"out_channels={out_channels} not a regular fiber")
        self.corb_out = out_channels // GROUP_ORDER
        bound = 1.0 / math.sqrt(7 * in_channels)  # fan-in on the orbit basis
        if in_channels == NUM_FEATURES:
            self.kind = "stem"
            # Free stem params in (7, C_out, NF) layout; forward Reynolds-projects
            # onto the equivariant subspace.
            self.w0 = nn.Parameter(torch.empty(7, out_channels, in_channels))
            nn.init.uniform_(self.w0, -bound, bound)
        else:
            self.kind = "regular"
            if in_channels % GROUP_ORDER != 0:
                raise ValueError(f"in_channels={in_channels} not a regular fiber")
            self.corb_in = in_channels // GROUP_ORDER
            self.w_base = nn.Parameter(
                torch.empty(7, GROUP_ORDER, self.corb_out, self.corb_in)
            )
            nn.init.uniform_(self.w_base, -bound, bound)
            self.register_buffer(
                "_conv_gather", _eq.conv_gather_index(), persistent=False
            )
        self.bias_base = nn.Parameter(torch.empty(self.corb_out))
        nn.init.uniform_(self.bias_base, -bound, bound)
        self._cache_v = None
        self._cache_w = None
        self._cache_b = None
        self._init_raytap()

    def _init_raytap(self) -> None:
        """Ray-tap free parameter (spec §2.2): ``alpha (RAY_REACH, C_ORBIT_in)``
        — per-distance, per-orbit-channel, shared across the 6 directions and
        tiled slot-constant over the fiber. Init ``alpha[:, c] = (1, 0, 0, 0,
        0)`` makes the ray-tap conv functionally identical to the baseline
        7-tap conv (init-equivalence, T4). Added ONLY when equipped, so
        RAYTAP=0 keeps the pre-change state-dict key set. Consumes no RNG (the
        shared-param stream stays identical to the unequipped build)."""

        if not self.raytap:
            return
        if self.equivariant and getattr(self, "kind", "") == "stem":
            raise ValueError(
                "the stem conv is always baseline (spec §2.3); raytap=True is "
                "only valid for regular-fiber convs"
            )
        corb_in = (
            self.in_channels // GROUP_ORDER if self.equivariant else self.in_channels
        )
        # Constructor-kwarg twin of the constants.py import-time check (§2.6):
        # the own/opp visibility halves split the orbit index.
        if corb_in % 2 != 0:
            raise ValueError(
                f"ray-tap conv needs an even orbit width (got {corb_in}): the "
                "own/opp visibility halves split the orbit index (spec §2.6)"
            )
        alpha = torch.zeros(RAY_REACH, corb_in)
        alpha[0] = 1.0
        self.alpha = nn.Parameter(alpha)

    def _alpha_full(self) -> torch.Tensor:
        """alpha tiled slot-constant to the full (RAY_REACH, C_in) fiber:
        channel ``c = slot*C_ORBIT + a`` reads ``alpha[:, a]``."""

        return (
            self.alpha.repeat(1, GROUP_ORDER) if self.equivariant else self.alpha
        )

    def _base_param(self) -> torch.Tensor:
        return self.w0 if self.kind == "stem" else self.w_base

    def _gen_weight(self) -> torch.Tensor:
        if self.kind == "stem":
            return _eq.gen_stem_weight(self.w0)
        return _eq.gen_conv_weight(self.w_base, self._conv_gather)

    def _materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(weight (7, C_in, C_out), bias (C_out,)); the passthrough params or
        the generated dense tied weight (cached under no-grad on the base
        ``_version`` so frozen serve regenerates once)."""

        if not self.equivariant:
            return self.weight, self.bias
        grad_on = torch.is_grad_enabled()
        if not grad_on:
            v = (self._base_param()._version, self.bias_base._version)
            if self._cache_w is not None and self._cache_v == v:
                return self._cache_w, self._cache_b
        weight = self._gen_weight()
        bias = self.bias_base.repeat(GROUP_ORDER)  # slot-constant (C_out,)
        if not grad_on:
            self._cache_v = (self._base_param()._version, self.bias_base._version)
            self._cache_w = weight
            self._cache_b = bias
        return weight, bias

    def forward(
        self,
        x: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
        ray_ctx=None,
    ) -> torch.Tensor:
        """x (B, Npad, Cin); gather_idx (B, Npad, 7) with tap 0 = self and
        missing -> Npad; mask (B, Npad) bool. Returns (B, Npad, Cout) with
        pad rows zeroed by the mask. ``ray_ctx`` (a ``_RayTapCtx``, built once
        per forward by trunk()) is required by (and only read by) ray-tap
        convs.
        """

        b, n, c = x.shape
        weight, bias = self._materialize()
        if self.raytap:
            # Ray-tap reference path (spec §2.4): the direction taps consume
            # the visibility-masked alpha-weighted ray aggregates; the center
            # tap and the GEMM against the tied-generated weight are unchanged.
            if ray_ctx is None:
                raise ValueError(
                    "ray-tap conv called without ray_ctx; the trunk builds it "
                    "from coords/mask + the raylen wire input (spec §2.5)"
                )
            taps = _raytap().ray_tap_taps(
                x, ray_ctx.idx_taps, ray_ctx.reach, self._alpha_full(),
                self.alpha.shape[1],
            )
            gathered = torch.cat([x.unsqueeze(2), taps], dim=2).reshape(b, n, 7 * c)
            out = gathered @ weight.reshape(7 * c, self.out_channels) + bias
            return out * mask.unsqueeze(-1)
        # Serve fast path (HEXFIELD_TRITON_CONV): fused gather+GEMM custom op —
        # the (B, Npad, 7C) tensor is never materialized. No-grad CUDA only (no
        # backward); 16-aligned channels only (the stem's C_in=25 falls through).
        if (
            _hex_conv_fused is not None
            and x.is_cuda
            and not torch.is_grad_enabled()
            and c % 16 == 0
            and self.out_channels % 16 == 0
        ):
            # Standalone HexNodeConvs are the stem and the policy/value head
            # convs; trunk ConvBlocks instead take the fused conv+LN path.
            return _hex_conv_fused(x, gather_idx, mask, weight, bias)
        x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)  # zero row at index Npad
        flat = gather_idx.reshape(b, n * 7, 1).expand(-1, -1, c)
        gathered = x_ext.gather(1, flat).reshape(b, n, 7 * c)
        out = gathered @ weight.reshape(7 * c, self.out_channels) + bias
        return out * mask.unsqueeze(-1)


class LayerScale(nn.Module):
    """Per-channel learned residual-branch scale (gamma), init 1e-4.

    Equivariant build: gamma is one C_ORBIT-vector broadcast (tiled) over the 12
    fiber slots so the scale commutes with the slot permutation (docs/DERIVATION
    §3)."""

    def __init__(self, channels: int, init: float = 1e-4) -> None:
        super().__init__()
        self.equivariant = EQUIVARIANT
        width = C_ORBIT if self.equivariant else channels
        self.gamma = nn.Parameter(torch.full((width,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.equivariant:
            return x * self.gamma.repeat(GROUP_ORDER)
        return x * self.gamma


class GroupAffineNorm(nn.Module):
    """Equivariant norm: LayerNorm's symmetric mean/var over the full C fiber
    (invariant under the slot permutation) then an affine tied per C_ORBIT and
    broadcast over the 12 slots (docs/DERIVATION §3). Exposes ``weight``/``bias``
    (C,) tiled views + ``eps`` so the fused Triton conv+LN serve kernel consumes
    it exactly like an ``nn.LayerNorm``."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if channels % C_ORBIT != 0:
            raise ValueError(f"norm channels={channels} not a regular fiber")
        self.channels = channels
        self.groups = channels // C_ORBIT
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(C_ORBIT))
        self.beta = nn.Parameter(torch.zeros(C_ORBIT))

    @property
    def weight(self) -> torch.Tensor:
        return self.gamma.repeat(self.groups)

    @property
    def bias(self) -> torch.Tensor:
        return self.beta.repeat(self.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xhat = F.layer_norm(x, (self.channels,), None, None, self.eps)
        return xhat * self.gamma.repeat(self.groups) + self.beta.repeat(self.groups)


def _make_norm(channels: int) -> nn.Module:
    """LayerNorm(channels) for the passthrough build, GroupAffineNorm for the
    equivariant build."""

    return GroupAffineNorm(channels) if EQUIVARIANT else nn.LayerNorm(channels)


class EquivLinear(nn.Module):
    """Tied 1x1 group-convolution nn.Linear (docs/DERIVATION §2.4): 12 free
    ``C_orbit_out x C_orbit_in`` blocks + a slot-constant bias. Materializes the
    dense ``(C_out, C_in)`` weight each forward (dense cache under no-grad keyed
    on the base ``_version``). Invisible to the Triton attention kernel.

    Optional SERVE-FOLD coset perms (:meth:`set_serve_perms`, set once by an
    equivariant attention/refresh owner): the fixed channel permutation the owner
    applies around attention (docs/DERIVATION §4) is folded into the CACHED dense
    weight so serve does ZERO runtime perm gathers. Output fold (q/k/v):
    ``(W x + b)[out_perm] == W[out_perm] x + b[out_perm]`` — a row permutation, so
    per-element bit-identical. Input fold (out_proj): the runtime
    ``out_proj(out[..., hp_inv])`` re-expresses to ``W[:, in_perm]`` with
    ``in_perm`` the head_perm itself (same products, accumulation reordered —
    allclose, not bit-identical). The fold is applied ONLY on the no-grad cache
    branch, gated on the exact ``not torch.is_grad_enabled()`` condition the
    owner's forward tests as ``folded`` — the two MUST agree or the projection and
    its runtime perm would double-apply / cancel. Default (no perms set) is a
    no-op: most EquivLinear uses (MLP fc1/fc2, token reads, heads) carry none."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels % GROUP_ORDER != 0 or out_channels % GROUP_ORDER != 0:
            raise ValueError("EquivLinear channels must be regular fibers")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.corb_in = in_channels // GROUP_ORDER
        self.corb_out = out_channels // GROUP_ORDER
        self.wb = nn.Parameter(torch.empty(GROUP_ORDER, self.corb_out, self.corb_in))
        self.bias_base = nn.Parameter(torch.zeros(self.corb_out))
        nn.init.trunc_normal_(self.wb, std=0.02)
        self.register_buffer("_gather", _eq.linear_gather_index(), persistent=False)
        # Serve-fold coset perms (set_serve_perms). NON-persistent so the
        # state_dict key set is UNCHANGED (checkpoint-compat + strict-load key
        # checks depend on it) yet they ride the serve-half deepcopy and
        # .to(device) moves; long dtype so Module float-casts (.half()) skip them.
        self.register_buffer("_serve_out_perm", None, persistent=False)
        self.register_buffer("_serve_in_perm", None, persistent=False)
        self._cache_v = None
        self._cache_w = None
        self._cache_b = None

    def set_serve_perms(self, out_perm=None, in_perm=None) -> None:
        """Fold the owner's coset perm (docs/DERIVATION §4) into the serve weight
        cache. ``out_perm`` folds a q/k/v output-row perm (``W[out_perm]``);
        ``in_perm`` folds out_proj's input-column perm as ``W[:, in_perm]`` — the
        head_perm ITSELF, not its inverse (the runtime ``out[..., hp_inv]``
        substitutes to ``sum_k out[k] W[i, hp[k]]``). The perms are static
        geometry, so ordering vs (re-)init of the weights is irrelevant; only the
        stale fold cache is dropped. Cloned so no lru_cache'd perm tensor is
        aliased into a buffer that ``_apply`` may move in place."""

        if out_perm is not None:
            self.register_buffer(
                "_serve_out_perm", out_perm.detach().clone(), persistent=False
            )
        if in_perm is not None:
            self.register_buffer(
                "_serve_in_perm", in_perm.detach().clone(), persistent=False
            )
        self._cache_v = self._cache_w = self._cache_b = None

    def _materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        grad_on = torch.is_grad_enabled()
        if not grad_on:
            v = (self.wb._version, self.bias_base._version)
            if self._cache_w is not None and self._cache_v == v:
                return self._cache_w, self._cache_b
        weight = _eq.gen_linear_weight(self.wb, self._gather)
        bias = self.bias_base.repeat(GROUP_ORDER)
        if not grad_on:
            # Fold the owner's coset perm into the cached weight (docs/DERIVATION
            # §4). This is the ``folded`` half of the owner forward's shared
            # ``not torch.is_grad_enabled()`` gate, so the runtime perm is skipped
            # there exactly when it is baked in here. The grad path returns the
            # unpermuted weight (the owner applies the perm at runtime instead).
            if self._serve_out_perm is not None:
                weight = weight[self._serve_out_perm]
                bias = bias[self._serve_out_perm]
            if self._serve_in_perm is not None:
                weight = weight[:, self._serve_in_perm]
            self._cache_v = (self.wb._version, self.bias_base._version)
            self._cache_w = weight
            self._cache_b = bias
        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self._materialize()
        return F.linear(x, weight, bias)


class ConvBlock(nn.Module):
    """Post-activation residual block (LayerNorm). ``raytap`` (spec §2.3):
    "0" = both convs baseline (current behavior); "conv2" = the second conv
    runs in ray-tap mode; "both" = both convs do."""

    def __init__(self, channels: int, raytap: str = "0") -> None:
        super().__init__()
        self.conv1 = HexNodeConv(channels, channels, raytap=(raytap == "both"))
        self.ln1 = _make_norm(channels)
        self.conv2 = HexNodeConv(
            channels, channels, raytap=(raytap in ("conv2", "both"))
        )
        self.ln2 = _make_norm(channels)
        self.ls = LayerScale(channels)

    def _serve_conv_ln(self, conv, ln, x, gather_idx, mask, ray_ctx, relu):
        """One conv + LN(+ReLU) + mask on the fused serve branch. Baseline
        convs keep the fused Triton conv+LN kernel (byte-identical to the
        pre-ray-tap path). Equipped convs run the ray-tap reference gather-sum
        GEMM followed by the _conv_ln_ref epilogue numerics (fp32 LN stats on
        the conv output, fp16 store) until the K1 fused variant lands; serve
        throughput measured on this path is labeled reference-path (§2.4)."""

        if not conv.raytap:
            w, b = conv._materialize()
            return _hex_conv_ln_fused(
                x, gather_idx, mask, w, b, ln.weight, ln.bias, ln.eps, relu
            )
        if _hex_conv_ln_raytap_fused is not None and ray_ctx is not None:
            # K1 fused variant: in-kernel k-loop over the ray gather index +
            # reach + tiled alpha (spec §2.4); the op itself falls back to the
            # reference on a memoized compile failure.
            w, b = conv._materialize()
            return _hex_conv_ln_raytap_fused(
                x, gather_idx, mask, w, b, ln.weight, ln.bias,
                ray_ctx.ray_idx, ray_ctx.reach, conv._alpha_full(),
                ln.eps, relu, conv.alpha.shape[1],
            )
        out = conv(x, gather_idx, mask, ray_ctx=ray_ctx)
        y = F.layer_norm(
            out.float(), (out.shape[-1],), ln.weight.float(), ln.bias.float(),
            ln.eps,
        )
        if relu:
            y = F.relu(y)
        y = y * mask.unsqueeze(-1)
        return y.to(torch.float16)

    def forward(
        self,
        x: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
        ray_ctx=None,
    ) -> torch.Tensor:
        # Serve fast path (HEXFIELD_TRITON_CONV_LN): conv + LN (+ReLU) + mask
        # in one kernel per conv — the LN round-trip over the activation never
        # happens. Residual + LayerScale + final ReLU stay outside (cheap
        # pointwise; inductor fuses them into one kernel).
        if (
            _hex_conv_ln_fused is not None
            and x.is_cuda
            and not torch.is_grad_enabled()
            and x.shape[-1] % 16 == 0
        ):
            # Materialized (tied-generated) conv weights + the norm's affine.
            # EQUIVARIANCE: the fused kernel applies a per-channel affine
            # (ln.weight[n], ln.bias[n]) in its LN epilogue. In the equivariant
            # build ln1/ln2 are GroupAffineNorm, whose .weight/.bias expose the
            # orbit-tied affine ALREADY expanded to (C,) — gamma/beta.repeat(groups),
            # i.e. weight[slot*C_ORBIT + a] = gamma[a] broadcast over the 12 slots
            # (docs/DERIVATION §3). Feeding that (C,) vector as the fused kernel's
            # per-channel affine makes the epilogue EXACTLY the orbit-tied group-norm,
            # so the fused serve path stays D6-equivariant (the LN mean/var are the
            # same symmetric full-fiber reduction the kernel computes over Cout).
            y = self._serve_conv_ln(
                self.conv1, self.ln1, x, gather_idx, mask, ray_ctx, True
            )
            y = self._serve_conv_ln(
                self.conv2, self.ln2, y, gather_idx, mask, ray_ctx, False
            )
            return F.relu(x + self.ls(y))
        m = mask.unsqueeze(-1)
        y = F.relu(self.ln1(self.conv1(x, gather_idx, mask, ray_ctx=ray_ctx))) * m
        y = self.ln2(self.conv2(y, gather_idx, mask, ray_ctx=ray_ctx)) * m
        return F.relu(x + self.ls(y))


class RelPosAttention(nn.Module):
    """Multi-head self-attention over the joint [tokens ; cells] sequence with
    this block's bias table gathered as an additive mask. Two implementations
    selectable via self.impl: 'sdpa' and 'materialized'."""

    def __init__(self, channels: int, heads: int | None = None) -> None:
        super().__init__()
        # heads defaults to the module-global ATTENTION_HEADS; an explicit value
        # builds the block at a different head count (the dashboard debug worker
        # reconstructs foreign-arch checkpoints this way — its process env is not
        # the run's env).
        self.heads = ATTENTION_HEADS if heads is None else int(heads)
        # head_dim derives from this net's width (channels // heads), so a net built
        # at a non-default width gets the correct per-head dim. At the default width
        # channels == CHANNELS, so head_dim == HEAD_DIM.
        self.head_dim = channels // self.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.equivariant = EQUIVARIANT
        if self.equivariant:
            # Tied group-conv projections (§2.4) + the coset channel permutation
            # (§4) so the (heads=3, head_dim=4*C_ORBIT) reshape lands each head on
            # one win-axis coset's channels.
            self.q_proj = EquivLinear(channels, channels)
            self.k_proj = EquivLinear(channels, channels)
            self.v_proj = EquivLinear(channels, channels)
            self.out_proj = EquivLinear(channels, channels)
            self.register_buffer("_head_perm", _eq.head_perm(), persistent=False)
            self.register_buffer("_head_perm_inv", _eq.head_perm_inv(), persistent=False)
            # Fold the §4 coset perm into the projections' serve weight cache so
            # the no-grad forward skips the q/k/v/out gathers (out_proj folds the
            # INPUT perm as W[:, head_perm]); the grad path keeps the runtime perm.
            self.q_proj.set_serve_perms(out_perm=self._head_perm)
            self.k_proj.set_serve_perms(out_perm=self._head_perm)
            self.v_proj.set_serve_perms(out_perm=self._head_perm)
            self.out_proj.set_serve_perms(in_perm=self._head_perm)
        else:
            self.q_proj = nn.Linear(channels, channels)
            self.k_proj = nn.Linear(channels, channels)
            self.v_proj = nn.Linear(channels, channels)
            self.out_proj = nn.Linear(channels, channels)
        self.impl = "sdpa"

    def forward(self, seq: torch.Tensor, attn_bias) -> torch.Tensor:
        b, s, c = seq.shape
        h, d = self.heads, self.head_dim
        q = self.q_proj(seq)
        k = self.k_proj(seq)
        v = self.v_proj(seq)
        # Serve folds the §4 coset perm into the cached q/k/v/out weights
        # (EquivLinear._materialize) on the SAME ``equivariant and not grad``
        # gate; the two must agree or the perm double-applies / cancels.
        folded = self.equivariant and not torch.is_grad_enabled()
        if self.equivariant and not folded:
            # Reorder slot-major channels into coset-grouped order so head hh of
            # the (heads, head_dim) reshape is win-axis coset hh; bias column hh
            # matches (docs/DERIVATION §4). The per-head dot products and the
            # jointly-tied bias are then equivariant. Applied uniformly so the
            # flex/Triton serve kernels (head-agnostic) work too.
            hp = self._head_perm
            q = q[..., hp]
            k = k[..., hp]
            v = v[..., hp]
        q = q.reshape(b, s, h, d).transpose(1, 2)
        k = k.reshape(b, s, h, d).transpose(1, 2)
        v = v.reshape(b, s, h, d).transpose(1, 2)
        # Each impl branch produces `out` in (B, heads, S, head_dim); the shared
        # tail transposes back, undoes the coset permutation, and applies out_proj.
        # Bespoke fused kernel (HEXFIELD_TRITON_ATTN): flex-pair serve batches
        # route to the FA2-style Triton kernel — pair bias gathered inside the
        # score loop, fully-padded key tiles skipped via seq_lens. No backward;
        # anything it can't take (grad, non-fp16, odd head_dim) falls through
        # to flex below.
        if (
            _attn_pair_fused is not None
            and isinstance(attn_bias, _FlexPairBias)
            and attn_bias.seq_lens is not None
            and not torch.is_grad_enabled()
            and q.is_cuda
            and q.dtype == torch.float16
            and d in (16, 32, 64, 128)
        ):
            out = _attn_pair_fused(
                q, k, v, attn_bias.pair, attn_bias.table2, attn_bias.seq_lens
            )
        # Flex path: the rel-pos bias + pad mask are computed inside the kernel via
        # a score_mod (no materialized (B,heads,S,S) tensor). block_mask is None.
        # The score_mod is built here, in the same frame as the flex call.
        elif isinstance(attn_bias, (_FlexBias, _FlexPairBias)):
            score_mod = attn_bias.make_score_mod()
            out = _flex_call(q, k, v, score_mod)
        else:
            # Match the bias dtype to q under autocast; a dtype mismatch drops sdpa
            # to the math fallback.
            attn_bias = attn_bias.to(q.dtype)
            if self.impl == "sdpa":
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            elif self.impl == "materialized":
                scores = (q @ k.transpose(-2, -1)) * self.scale + attn_bias
                out = torch.softmax(scores, dim=-1) @ v
            else:  # pragma: no cover - config validation
                raise ValueError(f"unknown attention impl: {self.impl}")
        out = out.transpose(1, 2).reshape(b, s, c)
        if self.equivariant and not folded:
            out = out[..., self._head_perm_inv]
        return self.out_proj(out)


class AttnBlock(nn.Module):
    """Pre-norm transformer block (GELU, MLP hidden width MLP_RATIO * channels)."""

    def __init__(self, channels: int, heads: int | None = None) -> None:
        super().__init__()
        self.ln1 = _make_norm(channels)
        self.attn = RelPosAttention(channels, heads)
        self.ln2 = _make_norm(channels)
        # The MLP hidden width is a regular-rep fiber too: MLP_RATIO*C =
        # 12*(MLP_RATIO*C_ORBIT) (docs/DERIVATION §2.4). GELU is per-channel so it
        # commutes with the slot permutation.
        linear = EquivLinear if EQUIVARIANT else nn.Linear
        self.fc1 = linear(channels, MLP_RATIO * channels)
        self.fc2 = linear(MLP_RATIO * channels, channels)
        self.ls_attn = LayerScale(channels)
        self.ls_mlp = LayerScale(channels)

    def forward(
        self, seq: torch.Tensor, attn_bias: torch.Tensor, seq_mask: torch.Tensor
    ) -> torch.Tensor:
        m = seq_mask.unsqueeze(-1)
        seq = seq + self.ls_attn(self.attn(self.ln1(seq), attn_bias) * m)
        seq = seq + self.ls_mlp(self.fc2(F.gelu(self.fc1(self.ln2(seq)))) * m)
        return seq


class RayAttention(nn.Module):
    """Ray-masked self-attention over cells only (docs/PLAN_REGISTER_LANE_RAY_
    ATTENTION.md L3/L4): RAY_HEADS = 6 heads = 3 win-axis cosets x {own, opp}
    orbit-halves. The equivariant build reorders channels via head_perm6 —
    coset-major then orbit-half, so the own/opp split rides the ORBIT index and
    never the K slots (a slot split silently breaks equivariance: K acts simply
    transitively on itself). Passthrough is a plain 6-head masked attention.
    The additive bias + ray mask arrive as a (B, RAY_HEADS, N, N) tensor, a
    _FlexRayBias carrier, or a _RayGatherBias carrier (the gathered Triton
    kernel, HEXFIELD_EQ_TRITON_RAY=1 serve path — spec D-S36/D-S37); impl
    selects 'sdpa' | 'materialized' like RelPosAttention. The A-block flex-pair
    Triton kernel never applies to L blocks."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = RAY_HEADS
        self.head_dim = channels // self.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.equivariant = EQUIVARIANT
        linear = EquivLinear if EQUIVARIANT else nn.Linear
        self.q_proj = linear(channels, channels)
        self.k_proj = linear(channels, channels)
        self.v_proj = linear(channels, channels)
        self.out_proj = linear(channels, channels)
        if EQUIVARIANT:
            self.register_buffer("_head_perm6", _eq.head_perm6(), persistent=False)
            self.register_buffer(
                "_head_perm6_inv", _eq.head_perm6_inv(), persistent=False
            )
            # Fold the §4/L4 6-head coset perm into the serve weight cache; grad
            # keeps the runtime perm (out_proj folds the input perm W[:, hp6]).
            self.q_proj.set_serve_perms(out_perm=self._head_perm6)
            self.k_proj.set_serve_perms(out_perm=self._head_perm6)
            self.v_proj.set_serve_perms(out_perm=self._head_perm6)
            self.out_proj.set_serve_perms(in_perm=self._head_perm6)
        self.impl = "sdpa"

    def forward(self, x: torch.Tensor, attn_bias) -> torch.Tensor:
        b, n, c = x.shape
        h, d = self.heads, self.head_dim
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Same ``equivariant and not grad`` gate as EquivLinear._materialize's
        # fold: serve bakes the perm into the weights, grad applies it here.
        folded = self.equivariant and not torch.is_grad_enabled()
        if self.equivariant and not folded:
            hp = self._head_perm6
            q = q[..., hp]
            k = k[..., hp]
            v = v[..., hp]
        q = q.reshape(b, n, h, d).transpose(1, 2)
        k = k.reshape(b, n, h, d).transpose(1, 2)
        v = v.reshape(b, n, h, d).transpose(1, 2)
        # Bespoke gathered kernel (HEXFIELD_EQ_TRITON_RAY): trunk() builds this
        # carrier only on the no-grad CUDA fp16 serve path at a supported
        # head_dim; the custom op re-checks and serves the gathered torch
        # reference on any residual miss (memoized compile failure, foreign
        # dtype). See _triton_ray.py.
        if _ray_attn_fused is not None and isinstance(attn_bias, _RayGatherBias):
            out = _ray_attn_fused(
                q, k, v,
                attn_bias.idx, attn_bias.slot_bias,
                attn_bias.raylen, attn_bias.seq_lens, attn_bias.blockers,
            )
        elif isinstance(attn_bias, _FlexRayBias):
            out = _flex_call(q, k, v, attn_bias.make_score_mod())
        else:
            attn_bias = attn_bias.to(q.dtype)
            if self.impl == "sdpa":
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            elif self.impl == "materialized":
                scores = (q @ k.transpose(-2, -1)) * self.scale + attn_bias
                out = torch.softmax(scores, dim=-1) @ v
            else:  # pragma: no cover - config validation
                raise ValueError(f"unknown attention impl: {self.impl}")
        out = out.transpose(1, 2).reshape(b, n, c)
        if self.equivariant and not folded:
            out = out[..., self._head_perm6_inv]
        return self.out_proj(out)


class RayAttnBlock(nn.Module):
    """Pre-norm ray-attention block over cells only: residual + MLP structure
    mirrors AttnBlock minus the token rows (tokens interact via the register
    lane and A blocks — plan L3). The masked softmax never sees an empty row:
    the diagonal is always live and every live query holds its own key."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ln1 = _make_norm(channels)
        self.attn = RayAttention(channels)
        self.ln2 = _make_norm(channels)
        linear = EquivLinear if EQUIVARIANT else nn.Linear
        self.fc1 = linear(channels, MLP_RATIO * channels)
        self.fc2 = linear(MLP_RATIO * channels, channels)
        self.ls_attn = LayerScale(channels)
        self.ls_mlp = LayerScale(channels)

    def forward(
        self, x: torch.Tensor, attn_bias, mask: torch.Tensor
    ) -> torch.Tensor:
        m = mask.unsqueeze(-1)
        x = x + self.ls_attn(self.attn(self.ln1(x), attn_bias) * m)
        x = x + self.ls_mlp(self.fc2(F.gelu(self.fc1(self.ln2(x)))) * m)
        return x


# Trunk layout by (conv_blocks, attn_blocks) count, for loaders that rebuild a
# FOREIGN-arch net off its checkpoint (eval anchors, the dashboard debug
# worker): the counts alone don't pin the C/A interleaving, so known lineage
# layouts are mapped explicitly. main_1..main_6 = CCC A CCC A CC A; main_7 =
# CC A x5; main_9 = CC A CC A CC A (6C/3A, the HEXFIELD_TRUNK=CCACCACCA arch).
#
# (6, 3) is SHARED by two DIFFERENT classes: main_9's current-arch net AND the
# frozen legacy-v2 snapshot (also 6C/3A, also CCACCACCA interleaving). This is
# safe because the two are NOT disambiguated by this map — they are told apart
# by their parameter-key SETS at strict-load time in eval_arena._load_checkpoint:
# legacy-v2 carries a single shared `bias_table` + `aux_reduction` and lacks
# `bias_free_tables.{i}` / cell_q / LayerScale (ls_attn/ls_mlp), so a legacy-v2 state
# dict fails the current HexfieldNet's strict load and falls through to
# eval_arena's HexfieldNetV2 fallback regardless of the trunk_layout inferred
# here. A main_9 state dict has the current-arch keys and loads cleanly. Mapping
# (6, 3) is thus NEEDED so a main_9 anchor rebuilt in a foreign-arch process
# (e.g. a main_7 CCAx5 dashboard worker) gets the right CCACCACCA interleaving
# instead of the process-global HEXFIELD_EQ_TRUNK default.
KNOWN_TRUNK_LAYOUTS: dict[tuple[int, int], str] = {
    (8, 3): "CCCACCCACCA",
    (10, 5): "CCACCACCACCACCA",
    (6, 3): "CCACCACCA",  # main_9 (current arch); legacy-v2 same shape, see above
}


def infer_net_kwargs_from_state_dict(sd: dict, meta: dict | None = None) -> dict:
    """HexfieldNet constructor kwargs inferred off a checkpoint state dict.

    The checkpoint ``meta`` (persisted by :meth:`HexfieldNet.arch_meta`) is read
    FIRST — it is the load-bearing self-description for the equivariant build,
    whose tied trunk carries no dense ``stem.weight`` / per-head ``bias_free_tables``
    to shape-infer from. Any field absent from meta falls back to shape inference:
    channels from stem params / tokens, attention_heads from the bias-table column
    count, trunk_layout from the block-count map, ``in_channels`` from
    ``stem.weight.shape[1]`` (passthrough) / ``stem.w0.shape[2]`` (tied). Every
    field is best-effort: anything undeterminable is omitted (the constructor
    falls back to the env-driven module globals).

    ``in_channels`` (the input feature width, NUM_FEATURES) is now RETURNED and
    passed to the constructor, which validates it against this build's
    ``NUM_FEATURES`` (a clear error instead of a silent stem shape mismatch).
    GROUP_ORDER / C_ORBIT remain env/import-time constants — a foreign-arch
    EQUIVARIANT rebuild still needs the matching HEXFIELD_EQ_* env, but the
    checkpoint meta now carries those values (arch_meta) so a loader can detect and
    report a mismatch rather than silently mis-shape the tie."""

    kwargs: dict = {}
    meta = meta or {}
    if meta.get("channels") is not None:
        kwargs["channels"] = int(meta["channels"])
    if meta.get("attention_heads") is not None:
        kwargs["attention_heads"] = int(meta["attention_heads"])
    if meta.get("trunk_layout") is not None:
        kwargs["trunk_layout"] = str(meta["trunk_layout"])
    # Input feature width: meta first (``in_channels`` or its ``feature_width``
    # alias), then the stem shape below.
    if meta.get("in_channels") is not None:
        kwargs["in_channels"] = int(meta["in_channels"])
    elif meta.get("feature_width") is not None:
        kwargs["in_channels"] = int(meta["feature_width"])
    if kwargs.get("channels") is None:
        # stem.weight (7, NF, C) passthrough; stem.w0 (7, C, NF) tied; then the
        # norm/token fallbacks.
        for key, axis in (
            ("stem.weight", 2),
            ("stem.w0", 1),
            ("stem.bias", 0),
            ("stem_ln.weight", 0),
            ("tokens", 1),
        ):
            t = sd.get(key)
            shape = getattr(t, "shape", None)
            if shape is not None and len(shape) > axis:
                kwargs["channels"] = int(shape[axis])
                break
    if kwargs.get("in_channels") is None:
        # in_channels == NF: stem.weight.shape[1] (passthrough) / stem.w0.shape[2]
        # (tied).
        for key, axis in (("stem.weight", 1), ("stem.w0", 2)):
            t = sd.get(key)
            shape = getattr(t, "shape", None)
            if shape is not None and len(shape) > axis:
                kwargs["in_channels"] = int(shape[axis])
                break
    if kwargs.get("attention_heads") is None:
        bt = sd.get("bias_free_tables.0")
        if bt is not None and len(getattr(bt, "shape", ())) == 2:
            kwargs["attention_heads"] = int(bt.shape[1])
    if kwargs.get("trunk_layout") is None:
        conv_ids = {int(k.split(".")[1]) for k in sd if k.startswith("conv_blocks.")}
        attn_ids = {int(k.split(".")[1]) for k in sd if k.startswith("attn_blocks.")}
        if conv_ids and attn_ids:
            layout = KNOWN_TRUNK_LAYOUTS.get((max(conv_ids) + 1, max(attn_ids) + 1))
            if layout is not None:
                kwargs["trunk_layout"] = layout
    # Register lane toggles (Phase R0): meta first; the state-dict key set is
    # affirmative evidence either way (present -> on, absent -> off), so the
    # rebuild is deterministic regardless of the loading process's env.
    if meta.get("reg_lane") is not None:
        kwargs["reg_lane"] = bool(meta["reg_lane"])
    else:
        kwargs["reg_lane"] = any(
            k.startswith(("registers.", "registers_l.")) for k in sd
        )
    if meta.get("reg_tok_read") is not None:
        kwargs["reg_tok_read"] = bool(meta["reg_tok_read"])
    else:
        kwargs["reg_tok_read"] = any(
            k.startswith(("tok_reads.", "tok_reads_l.")) for k in sd
        )
    # Ray-attention mask semantics (Phase L1): meta-only (no state-dict trace —
    # the toggle is a mask-build variant); absent meta keeps the env default.
    if meta.get("ray_blockers") is not None:
        kwargs["ray_blockers"] = bool(meta["ray_blockers"])
    # Ray-tap conv mode (SPEC_RAYTAP_CONV.md §4): meta first (authoritative
    # ternary). Fallback inference distinguishes conv2/both by the presence of
    # `alpha` on FIRST-conv keys — presence of any `alpha` alone is
    # insufficient (both modes carry conv2 alphas).
    if meta.get("raytap") is not None:
        kwargs["raytap"] = str(meta["raytap"])
    else:
        has_conv1_alpha = any(
            k.startswith("conv_blocks.") and k.endswith(".conv1.alpha") for k in sd
        )
        has_conv2_alpha = any(
            k.startswith("conv_blocks.") and k.endswith(".conv2.alpha") for k in sd
        )
        kwargs["raytap"] = (
            "both" if has_conv1_alpha else ("conv2" if has_conv2_alpha else "0")
        )
    return kwargs


class HexfieldNet(nn.Module):
    """The full network: stem, TRUNK_LAYOUT (default C C C A C C C A C C A),
    LN_final, heads."""

    def __init__(
        self,
        channels: int = CHANNELS,
        attention_heads: int | None = None,
        trunk_layout: str | None = None,
        in_channels: int | None = None,
        reg_lane: bool | None = None,
        reg_tok_read: bool | None = None,
        ray_blockers: bool | None = None,
        raytap: str | None = None,
    ) -> None:
        super().__init__()
        # channels/attention_heads/trunk_layout default to the module globals
        # (env-driven, read once at import); explicit values build the net at a
        # different shape. The env path is how every RUN constructs its net; the
        # explicit path exists for cross-arch loaders (the dashboard debug
        # worker infers all three off a checkpoint's state dict and passes them
        # here, so one process can serve main_6 c=128/4-head and main_7
        # c=192/3-head/CCAx5 checkpoints side by side).
        # conv_blocks[i] is the i-th 'C' and attn_blocks[i] the i-th 'A' in
        # layout order.
        #
        # in_channels is the input feature width (== NUM_FEATURES). It is NOT a
        # free shape: the stem lift is import-time bound to this build's
        # NUM_FEATURES rep (25 planes), so an explicit value that disagrees is a
        # hard error rather than a silent stem shape mismatch at load. Loaders
        # that infer it from a checkpoint (infer_net_kwargs_from_state_dict) pass
        # it so a foreign checkpoint built at a different feature width fails
        # loudly with the correct diagnosis.
        if in_channels is not None and int(in_channels) != NUM_FEATURES:
            raise ValueError(
                f"in_channels={int(in_channels)} != this build's feature width "
                f"NUM_FEATURES={NUM_FEATURES}: the stem lift is import-time bound "
                f"to the {NUM_FEATURES}-plane input rep. The checkpoint was built "
                "at a different feature width; rebuild hexfield_eq at a matching "
                "NUM_FEATURES to load it."
            )
        c = channels
        heads = ATTENTION_HEADS if attention_heads is None else int(attention_heads)
        # The kwarg twin of the constants.py import check: under the tied build
        # the A-block head count is structural (heads == the 3 win-axis cosets),
        # so an explicit attention_heads != 3 is an equivariance break, not a
        # shape choice.
        if EQUIVARIANT and heads != 3:
            raise ValueError(
                f"attention_heads={heads} must be 3 for the equivariant build: "
                "the multi-head split is the 3 left cosets of K=stab(Q-axis) "
                "(docs/DERIVATION §4)"
            )
        layout = TRUNK_LAYOUT if trunk_layout is None else str(trunk_layout)
        if not layout or set(layout) - {"C", "A", "L"} or not layout.endswith("A"):
            raise ValueError(f"invalid trunk layout {layout!r}")
        self._trunk_layout = layout
        self._equivariant = EQUIVARIANT
        # Ray-attention blocker toggle (plan L6): a mask-build variant (no
        # state-dict change), env default with an explicit kwarg so meta-first
        # rebuilds reproduce the training-time mask semantics.
        self._ray_blockers = RAY_BLOCKERS if ray_blockers is None else bool(ray_blockers)
        # Ray-tap conv mode (SPEC_RAYTAP_CONV.md §2.3): env default, explicit
        # kwarg for cross-arch loaders (an arch knob — equipped convs carry an
        # `alpha` param, so it changes the state-dict key set and rides
        # arch_meta / infer_net_kwargs_from_state_dict).
        rt_mode = RAYTAP if raytap is None else str(raytap)
        if rt_mode not in ("0", "conv2", "both"):
            raise ValueError(
                f"raytap={rt_mode!r} must be '0', 'conv2', or 'both'"
            )
        self._raytap = rt_mode
        if rt_mode != "0":
            # Constructor-kwarg twin of the constants.py import check (§2.6).
            rt_corb = c // GROUP_ORDER if EQUIVARIANT else c
            if rt_corb % 2 != 0:
                raise ValueError(
                    f"raytap={rt_mode!r} needs an even orbit width (got "
                    f"{rt_corb}): the own/opp visibility halves split the "
                    "orbit index (spec §2.6)"
                )
        n_ray = layout.count("L")
        if n_ray and c % RAY_HEADS != 0:
            raise ValueError(
                f"channels={c} not divisible by RAY_HEADS={RAY_HEADS} "
                "(required for an 'L' trunk layout)"
            )
        if n_ray and EQUIVARIANT and C_ORBIT % 2 != 0:
            raise ValueError(
                f"C_ORBIT={C_ORBIT} must be even for an 'L' layout (the own/opp "
                "sub-head split is along the orbit index; head_dim_L = 2*C_ORBIT)"
            )
        # Register lane toggles (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase
        # R0): env defaults, explicit kwargs for cross-arch loaders (the toggles
        # change the state-dict key set, so meta-first rebuilds pass them here).
        self._reg_lane = REG_LANE if reg_lane is None else bool(reg_lane)
        self._reg_tok_read = (
            REG_TOK_READ if reg_tok_read is None else bool(reg_tok_read)
        )
        if self._reg_tok_read and not self._reg_lane:
            raise ValueError(
                "reg_tok_read=True requires reg_lane=True (the cells <- tokens "
                "read is an arm of the register lane)"
            )
        self.stem = HexNodeConv(NUM_FEATURES, c)
        self.stem_ln = _make_norm(c)
        # The stem and the tied head convs are always baseline (spec §2.3);
        # ray-tap equips trunk ConvBlocks only.
        self.conv_blocks = nn.ModuleList(
            [ConvBlock(c, raytap=rt_mode) for _ in range(layout.count("C"))]
        )
        if rt_mode != "0":
            # Generated tap geometry LUTs (spec §2.5, T7): non-persistent so
            # the state-dict key set is untouched, long dtype so .half()
            # skips them; they ride .to(device) and the serve-half deepcopy.
            rt = _raytap()
            self.register_buffer(
                "_raytap_slot_lut", rt.tap_ray_slot_lut().clone(), persistent=False
            )
            self.register_buffer(
                "_raytap_raylen_slots",
                rt.tap_raylen_slots().clone(),
                persistent=False,
            )
        self.attn_blocks = nn.ModuleList(
            [AttnBlock(c, heads) for _ in range(layout.count("A"))]
        )
        # ray_blocks[i] is the i-th 'L' in layout order; built ONLY for an L
        # layout so C/A layouts keep their state-dict key set.
        if n_ray:
            self.ray_blocks = nn.ModuleList([RayAttnBlock(c) for _ in range(n_ray)])
        n_attn = len(self.attn_blocks)
        # Summary tokens carry the trivial (invariant) subrep in the equivariant
        # build: a learned (NUM_TOKENS, C_ORBIT) broadcast (tiled) over the 12
        # slots (docs/DERIVATION §6). Passthrough keeps the full (NUM_TOKENS, C).
        self.tokens = nn.Parameter(
            torch.empty(NUM_TOKENS, C_ORBIT if EQUIVARIANT else c)
        )
        if EQUIVARIANT:
            # Jointly (row, head)-tied relative-position bias (docs/DERIVATION §5):
            # Phase-2's board-orbit tie with heads left free is NOT equivariant, so
            # the (BIAS_ROWS, heads) table is refined into the joint (row, head)
            # orbit classes. Each block owns a free (n_joint_classes,) param; the
            # expanded (BIAS_ROWS, heads) table every downstream consumer sees is
            # theta[joint_of_row_head]. joint_of_row_head is a persistent buffer
            # (checkpoint self-description).
            joint, n_joint = _eq.joint_bias_lut()
            self._n_joint_classes = int(n_joint)
            self.register_buffer("joint_of_row_head", joint, persistent=True)
            self.bias_free_tables = None
            self.bias_theta = nn.ParameterList(
                [nn.Parameter(torch.zeros(n_joint)) for _ in range(n_attn)]
            )
        else:
            # Orbit-tied relative-position bias (Phase 2): each attention block owns
            # a zero-init FREE (BIAS_FREE_ROWS, heads) table; the expanded
            # (BIAS_ROWS, heads) table every bias builder consumes is
            # free[orbit_of_row], so all rows in a D6 orbit share their bias.
            # orbit_of_row is a persistent buffer (checkpoint self-description).
            self.register_buffer(
                "orbit_of_row",
                torch.as_tensor(bias_orbit_of_row(), dtype=torch.long),
                persistent=True,
            )
            self.bias_theta = None
            self.bias_free_tables = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(BIAS_FREE_ROWS, heads))
                    for _ in range(n_attn)
                ]
            )
        if n_ray:
            # Ray-attention bias (plan L5): the joint (row, head) tie extended
            # by the group-invariant side index — each L block owns a free
            # (n_joint_classes, 2) param indexed [joint_of_row_head[row, coset],
            # side], expanded to the (BIAS_ROWS, RAY_HEADS) table the mask
            # builders consume. The passthrough build mirrors the A-block
            # orbit tie with a free (BIAS_FREE_ROWS, RAY_HEADS) table per block.
            # Both names keep the bias predicates' substrings ("bias_theta" /
            # "bias_free_table") so AdamW no-decay needs no predicate edit.
            # NOTE (spec D-S28): only a small joint-class subset is LIVE — ray
            # offsets have hex-dist <= RAY_REACH along an axis, so ~6 disk
            # classes x 2 sides ever receive gradient; every other class sits
            # behind the additive -3e4 mask (softmax weight underflows to 0 and
            # its grad with it). Dead classes staying at zero-init is expected;
            # do not add init noise or weight decay here expecting it to matter.
            if EQUIVARIANT:
                self.bias_theta_l = nn.ParameterList(
                    [
                        nn.Parameter(torch.zeros(self._n_joint_classes, 2))
                        for _ in range(n_ray)
                    ]
                )
            else:
                self.ray_bias_free_tables = nn.ParameterList(
                    [
                        nn.Parameter(torch.zeros(BIAS_FREE_ROWS, RAY_HEADS))
                        for _ in range(n_ray)
                    ]
                )
            self.register_buffer(
                "_ray_coset_of_head",
                torch.tensor([h // 2 for h in range(RAY_HEADS)], dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_ray_side_of_head",
                torch.tensor([h % 2 for h in range(RAY_HEADS)], dtype=torch.long),
                persistent=False,
            )
        self.ln_final = _make_norm(c)

        # Heads. Policy heads read cells; value/aux read tokens + the masked
        # mean-pool of cells. In the equivariant build a per-cell covariant head is
        # a tied conv then a FIBER-INVARIANT read: expand, group-pool the fiber
        # (mean over the 12 slots), then a plain Linear, so the per-cell logit
        # lands in the trivial rep (docs/DERIVATION §7). The invariant value/aux/ml
        # heads group-pool their (expanded) token + pooled-cell read blocks before
        # the reduction, so the reductions consume invariant vectors.
        # Widened invariant reads (spec D-S20/D-S21): a shared EquivLinear
        # C -> INV_READ_EXPAND*C expansion feeds every group-pooled scalar-head
        # read block (so each block is INV_READ_EXPAND*C_ORBIT wide, not the
        # bare C_ORBIT=16 bottleneck); each per-cell policy head gets its own
        # POLICY_READ_EXPAND*C expansion the same way. The value head reads ALL
        # NUM_TOKENS tokens + the pooled cells + the PRE-ln_final token mean
        # (the register lane's count magnitudes, which ln_final would erase);
        # aux/ml keep their token pairs + pooled + the pre-ln mean.
        head_linear = EquivLinear if EQUIVARIANT else nn.Linear
        read_w = INV_READ_EXPAND * (C_ORBIT if EQUIVARIANT else c)
        pol_w = POLICY_READ_EXPAND * (C_ORBIT if EQUIVARIANT else c)
        red_out = read_w
        self.inv_read = head_linear(c, INV_READ_EXPAND * c)
        self.policy_conv = HexNodeConv(c, c)
        self.policy_expand = head_linear(c, POLICY_READ_EXPAND * c)
        self.policy_head = nn.Linear(pol_w, 1)
        self.opp_policy_conv = HexNodeConv(c, c)
        self.opp_policy_expand = head_linear(c, POLICY_READ_EXPAND * c)
        self.opp_policy_head = nn.Linear(pol_w, 1)
        # Auxiliary soft policy head (train-only): its own conv + Linear, mirroring
        # opp_policy. Initialized by _init_weights.
        self.soft_policy_conv = HexNodeConv(c, c)
        self.soft_policy_expand = head_linear(c, POLICY_READ_EXPAND * c)
        self.soft_policy_head = nn.Linear(pol_w, 1)
        # Per-cell Q head (train-only): emitted in forward() only, not in serve.
        self.cell_q_conv = HexNodeConv(c, c)
        self.cell_q_expand = head_linear(c, POLICY_READ_EXPAND * c)
        self.cell_q_head = nn.Linear(pol_w, VALUE_BINS)
        self.value_reduction = nn.Linear((NUM_TOKENS + 2) * read_w, red_out)
        self.value_head = nn.Linear(red_out, VALUE_BINS)
        self.aux_reduction = nn.Linear(4 * read_w, red_out)
        self.stv_heads = nn.ModuleDict(
            {str(h): nn.Linear(red_out, VALUE_BINS) for h in STV_HORIZONS}
        )
        self.ml_reduction = nn.Linear(4 * read_w, red_out)  # moves_left (tokens 4, 5)
        self.moves_left_head = nn.Linear(red_out, VALUE_BINS)

        # Cell-bias LUT over the (dq, dr) offset domain. Gathered per pair as
        # row = _cell_bias_lut[(clamp(dq,-M,M)+M)*W + (clamp(dr,-M,M)+M)], with
        # M = BIAS_RING_MAX + 1 and W = 2M + 1. For |dq|,|dr| <= M the entry equals
        # rel_bias_index(dq, dr); offsets beyond M map to the far row.
        self._cell_bias_M = BIAS_RING_MAX + 1
        cw = 2 * self._cell_bias_M + 1
        cell_lut = torch.empty(cw * cw, dtype=torch.long)
        for dq in range(-self._cell_bias_M, self._cell_bias_M + 1):
            for dr in range(-self._cell_bias_M, self._cell_bias_M + 1):
                cell_lut[(dq + self._cell_bias_M) * cw + (dr + self._cell_bias_M)] = (
                    rel_bias_index(dq, dr)
                )
        self.register_buffer("_cell_bias_lut", cell_lut, persistent=False)
        # uint8 twin of the LUT for the precomputed-pair path (values <= 233).
        self.register_buffer(
            "_cell_bias_lut_u8", cell_lut.to(torch.uint8), persistent=False
        )

        # Flex flags, read once. serve_flex applies on the no-grad serve path;
        # train_flex on the grad path (fp32-table carrier). With both off,
        # attention uses the materialized build_attn_bias + _BiasGather.
        # flex_pair upgrades the serve-flex score_mod to the precomputed-pair
        # variant (see _FLEX_PAIR above); it is inert without serve_flex.
        self._serve_flex = _SERVE_FLEX and _flex_attention is not None
        self._train_flex = _TRAIN_FLEX and _flex_attention is not None
        self._flex_pair = _FLEX_PAIR
        self._train_flex_pair = _TRAIN_FLEX_PAIR

        self._init_weights()

        # Register lane (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase R0):
        # registers[i] refreshes the tokens at the i-th C block's exit;
        # tok_reads[i] is the cells <- tokens read at its entry. Built ONLY when
        # the toggles are on, so toggle-off keeps the pre-lane state-dict key
        # set. Built AFTER _init_weights: the zero-init out_proj/tok_read must
        # survive the nn.Linear re-init sweep, and the shared-param RNG stream
        # stays identical to the toggle-off build under the same seed. The
        # import is lazy: register.py reuses this module's tied primitives.
        if self._reg_lane:
            from .register import RegisterRefresh, TokenRead

            n_conv = layout.count("C")
            self.registers = nn.ModuleList(
                [RegisterRefresh(c, heads) for _ in range(n_conv)]
            )
            if self._reg_tok_read:
                self.tok_reads = nn.ModuleList(
                    [TokenRead(c) for _ in range(n_conv)]
                )
            # L blocks are non-A blocks too (plan R6): separate ModuleLists so
            # C-block indices stay dense in registers/tok_reads.
            if n_ray:
                self.registers_l = nn.ModuleList(
                    [RegisterRefresh(c, heads) for _ in range(n_ray)]
                )
                if self._reg_tok_read:
                    self.tok_reads_l = nn.ModuleList(
                        [TokenRead(c) for _ in range(n_ray)]
                    )

    def _init_weights(self) -> None:
        """Linears trunc_normal(std=0.02) weight, zero bias; LayerNorm weight 1,
        bias 0; convs keep HexNodeConv's own init; tokens trunc_normal(std=0.02).
        Bias tables are left at their zero init from the ParameterList constructor.
        LayerScale.gamma is neither Linear nor LayerNorm, so the loops below leave
        its init fill untouched."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.tokens, std=0.02)

    def set_attention_impl(self, impl: str) -> None:
        for block in self.attn_blocks:
            block.attn.impl = impl
        for block in getattr(self, "ray_blocks", ()):
            block.attn.impl = impl

    def arch_meta(self) -> dict:
        """Load-bearing architecture self-description for checkpoint meta
        (docs/PLAN §3 Phase 3b). Foreign loaders rebuild the tie from this; the
        geometric LUTs (orbit_of_row / joint_of_row_head) also ride as persistent
        buffers in the state dict."""

        meta = {
            "group_order": GROUP_ORDER,
            "c_orbit": C_ORBIT,
            "channels": int(self.stem.out_channels),
            "in_channels": int(self.stem.in_channels),
            "attention_heads": int(self.attn_blocks[0].attn.heads),
            "trunk_layout": self._trunk_layout,
            "num_tokens": NUM_TOKENS,
            # feature_width is the input feature width (== in_channels); kept as an
            # alias so older readers that only know feature_width still resolve it.
            "feature_width": NUM_FEATURES,
            # The featurizer plane-map version this net was built under (spec
            # §1.1/§4): loaders hard-assert it (same class as support_radius).
            "feature_version": FEATURE_VERSION,
            # Ray-tap conv mode (spec §2.3/§4, ternary, authoritative):
            # load-bearing — equipped convs carry `alpha` params, and
            # conv2-vs-both cannot be told apart by key presence alone.
            "raytap": self._raytap,
            "equivariant": self._equivariant,
            # Register lane toggles (Phase R0): load-bearing — they change the
            # state-dict key set, and KNOWN_TRUNK_LAYOUTS cannot express them.
            "reg_lane": self._reg_lane,
            "reg_tok_read": self._reg_tok_read,
            # The featurizer support radius this net was built/trained under
            # (spec D-S26): a mismatch at load is a silent input-distribution
            # shift, so loaders assert it.
            "support_radius": _SUPPORT_RADIUS,
        }
        if self._equivariant:
            # The bias free_rows reduction the joint (row, head) tie produces.
            meta["bias_reduction"] = "joint_row_head"
            meta["bias_joint_classes"] = self._n_joint_classes
        else:
            meta["bias_reduction"] = "orbit_of_row"
            meta["bias_free_rows"] = BIAS_FREE_ROWS
        if "L" in self._trunk_layout:
            # Ray-attention config (Phase L1): the head count is structural but
            # recorded for self-description; ray_blockers is the mask semantics
            # the net was trained under (plan L6).
            meta["ray_heads"] = RAY_HEADS
            meta["ray_blockers"] = self._ray_blockers
        return meta

    # --- pair index + bias (pair built once per batch; bias built per A block) ---

    def _build_pair(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Block-independent pieces of the bias build, computed once per forward
        and reused by all 3 attention blocks.

        Returns (pair, key_pad):
        - pair (B, S, S) long: the per-pair bias-table row index. S = NUM_TOKENS +
          Npad. Tokens occupy slots 0..NUM_TOKENS-1 with no board position; token
          keys are never masked.
        - key_pad (B, S) bool: True at live keys (the pad-KEY additive fill mask).

        coords (B, Npad, 2) long; mask (B, Npad) bool."""

        b, n, _ = coords.shape
        dq = coords[:, None, :, 0] - coords[:, :, None, 0]  # (B, N, N) key - query
        dr = coords[:, None, :, 1] - coords[:, :, None, 1]
        # LUT gather over the offset domain (see _cell_bias_lut in __init__):
        # clamp + mul-add + gather to the (B, N, N) row indices.
        m = self._cell_bias_M
        w = 2 * m + 1
        qi = dq.clamp(-m, m) + m
        ri = dr.clamp(-m, m) + m
        cell_idx = self._cell_bias_lut[(qi * w + ri).reshape(-1)].reshape(b, n, n)

        s = NUM_TOKENS + n
        pair = coords.new_full((b, s, s), BIAS_TOKEN_TOKEN_ROW)
        pair[:, :NUM_TOKENS, NUM_TOKENS:] = BIAS_TOKEN_CELL_ROW
        pair[:, NUM_TOKENS:, :NUM_TOKENS] = BIAS_CELL_TOKEN_ROW
        pair[:, NUM_TOKENS:, NUM_TOKENS:] = cell_idx

        # Pad-cell KEY columns: additive, finite in fp16; token keys untouched.
        # The mask is added before the attn_mask is materialized so it has
        # stride(-1) == 1 on the key axis; a non-stride-1 attn_mask forces
        # F.scaled_dot_product_attention onto the fp32 math backend instead of the
        # fused fp16 mem-efficient kernel.
        key_pad = torch.cat(
            [mask.new_ones(b, NUM_TOKENS), mask], dim=1
        )  # (B, S) True = live key
        return pair, key_pad

    def _block_bias_table(self, block: int) -> torch.Tensor:
        """Expanded (BIAS_ROWS, heads) relative-position bias table for attention
        block `block`. Equivariant build: gathered from that block's free
        (n_joint_classes,) param by the JOINT (row, head) LUT (joint_of_row_head),
        so the bias ties across the head axis in step with the board orbit
        (docs/DERIVATION §5). Passthrough: the Phase-2 free (BIAS_FREE_ROWS, heads)
        table gathered by orbit_of_row. Either way the index-select carries
        gradients back to the free params and every downstream consumer
        (_BiasGather, the flex carriers, the Triton attn kernel) is unchanged.
        Returns the fp32 master dtype; serve callers cast to fp16 themselves."""

        if self._equivariant:
            return self.bias_theta[block][self.joint_of_row_head]  # (BIAS_ROWS, heads)
        return self.bias_free_tables[block][self.orbit_of_row]

    def build_attn_bias(
        self, pair: torch.Tensor, key_pad: torch.Tensor, block: int
    ) -> torch.Tensor:
        """(B, heads, S, S) additive bias for attention block `block`, using that
        block's own expanded table (_block_bias_table) plus the pad-key mask.

        `pair` (B, S, S) row indices and `key_pad` (B, S) live-key mask come from
        _build_pair (built once per forward, shared across the 3 blocks)."""

        table = self._block_bias_table(block)
        if torch.is_grad_enabled():
            # Grad path: gather in fp32 via _BiasGather (fp32 table gradient). Add
            # the mask in head-last layout (key = dim 2), then one permute+contiguous
            # to (B, heads, Sq, Sk).
            bias = _BiasGather.apply(table, pair)  # (B, Sq, Sk, heads) fp32
            fill = torch.where(key_pad, 0.0, PAD_KEY_MASK_VALUE).to(bias.dtype)
            bias = bias + fill[:, None, :, None]  # broadcast over key axis (dim 2)
            return bias.permute(0, 3, 1, 2).contiguous()  # (B, heads, Sq, Sk)

        # No-grad path: build the (B, heads, S, S) bias in fp16, head-first layout.
        # Indexing the transposed table (heads, ROWS) yields a contiguous
        # (heads, B, Sq, Sk); permute(1,0,2,3) is a stride(-1)==1 view, and the
        # pad-mask add is the single full-tensor materialization.
        bias_t = table.to(torch.float16).t().contiguous()  # (heads, ROWS)
        bias = bias_t[:, pair]                       # (heads, B, Sq, Sk) contiguous
        bias = bias.permute(1, 0, 2, 3)              # (B, heads, Sq, Sk) view, stride(-1)=1
        fill = torch.where(key_pad, 0.0, PAD_KEY_MASK_VALUE).to(bias.dtype)
        return bias + fill[:, None, None, :]         # broadcast over key axis (dim 3)

    def _build_pair_u8(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """(B, S, S) uint8 per-pair bias-table row index for the flex-pair serve
        path: the same rows _build_pair produces, built once per forward and
        shared by all 3 attention blocks, with pad-KEY columns set to the extra
        pad row (BIAS_ROWS). Built through int16/int32 intermediates so no
        (B, S, S) int64 tensor is materialized. Pad QUERY rows carry garbage
        rows exactly like every other bias path (their outputs are re-zeroed
        after each block)."""

        b, n, _ = coords.shape
        c16 = coords.to(torch.int16)
        dq = c16[:, None, :, 0] - c16[:, :, None, 0]  # (B, N, N) key - query
        dr = c16[:, None, :, 1] - c16[:, :, None, 1]
        m = self._cell_bias_M
        w = 2 * m + 1
        qi = (dq.clamp(-m, m) + m).to(torch.int32)
        ri = (dr.clamp(-m, m) + m).to(torch.int32)
        cell = self._cell_bias_lut_u8[(qi * w + ri).reshape(-1)].reshape(b, n, n)

        s = NUM_TOKENS + n
        pair = torch.full(
            (b, s, s), BIAS_TOKEN_TOKEN_ROW, dtype=torch.uint8, device=coords.device
        )
        pair[:, :NUM_TOKENS, NUM_TOKENS:] = BIAS_TOKEN_CELL_ROW
        pair[:, NUM_TOKENS:, :NUM_TOKENS] = BIAS_CELL_TOKEN_ROW
        pair[:, NUM_TOKENS:, NUM_TOKENS:] = cell
        # Pad-cell KEY columns -> the appended pad row; token keys never masked.
        key_dead = torch.cat([mask.new_zeros(b, NUM_TOKENS), ~mask], dim=1)
        return pair.masked_fill(key_dead[:, None, :], BIAS_ROWS)

    def _build_flex_pair_bias(
        self, pair: torch.Tensor, block: int, seq_lens: torch.Tensor | None = None
    ) -> "_FlexPairBias":
        """Flex-pair carrier for attention block `block`: the shared uint8 pair
        index plus this block's fp16 table with the PAD_KEY_MASK_VALUE row
        appended (row BIAS_ROWS), so the pad-KEY additive fill is a plain table
        row instead of an in-kernel mask read. `seq_lens` rides along for the
        bespoke Triton attention kernel (None on the flex path)."""

        table = self._block_bias_table(block).to(torch.float16)
        pad_row = table.new_full((1, table.shape[1]), PAD_KEY_MASK_VALUE)
        return _FlexPairBias(pair, torch.cat([table, pad_row], dim=0), seq_lens)

    def _build_train_flex_pair_bias(
        self, pair: torch.Tensor, block: int
    ) -> "_FlexPairBias":
        """Grad-enabled flex-pair carrier: table2 keeps the FP32 master table
        (no fp16 cast) so the score_mod's table2[row, h] read is differentiable
        and the table gradient accumulates in fp32; the appended pad row is a
        constant (no grad). Same fp32-master rationale as _build_train_flex_bias."""

        table = self._block_bias_table(block)  # fp32 master — NOT cast (see above)
        pad_row = table.detach().new_full((1, table.shape[1]), PAD_KEY_MASK_VALUE)
        return _FlexPairBias(pair, torch.cat([table, pad_row], dim=0))

    def _build_flex_bias(
        self, coords: torch.Tensor, mask: torch.Tensor, block: int
    ) -> "_FlexBias":
        """Serve-flex (no-grad) equivalent of build_attn_bias for attention block
        `block`. Packages the raw tensors the score_mod needs (coords, mask, fp16
        bias table for this block, cell LUT) into a _FlexBias carrier; the closure
        is built in RelPosAttention.forward. No (B, heads, S, S) tensor is
        materialized; block_mask is None (the pad mask is folded into the score).

        The pad-key boundary is read directly from the bool mask, not mask.sum(),
        since a reduction inside the score_mod produces an unbacked symint the
        dynamic-Npad Inductor lowering cannot bind. Table is (ROWS, heads) fp16,
        no transpose; the score_mod indexes table[row, h]."""

        table = self._block_bias_table(block).to(torch.float16)
        return _FlexBias(coords, mask, table, self._cell_bias_lut, self._cell_bias_M)

    def _build_train_flex_bias(
        self, coords: torch.Tensor, mask: torch.Tensor, block: int
    ) -> "_FlexBias":
        """Train-flex (grad-enabled) equivalent of build_attn_bias for attention
        block `block`. Same as _build_flex_bias except it passes the fp32
        expanded table (_block_bias_table) directly (no .to(fp16) cast), so the
        score_mod's `table[row, h].to(score.dtype)` read is differentiable and
        the table gradient accumulates in fp32 back to the 45 free rows. No
        (B, heads, S, S) tensor is materialized; block_mask stays None (pad mask
        folded into the score)."""

        table = self._block_bias_table(block)  # fp32, not cast to fp16 (see above)
        return _FlexBias(coords, mask, table, self._cell_bias_lut, self._cell_bias_M)

    # --- ray-attention bias + mask (L blocks; cells-only) -----------------------

    def _ray_bias_table(self, block: int) -> torch.Tensor:
        """Expanded (BIAS_ROWS, RAY_HEADS) bias table for ray block `block`.
        Equivariant build: the joint (row, head) tie extended by the
        group-invariant side index (plan L5) — gathered from the block's free
        (n_joint_classes, 2) param as theta[joint_of_row_head[row, h6 // 2],
        h6 % 2]. Passthrough: the orbit-tied free (BIAS_FREE_ROWS, RAY_HEADS)
        table gathered by orbit_of_row. Both gathers carry gradients back to
        the free params. Returns the fp32 master dtype."""

        if self._equivariant:
            theta = self.bias_theta_l[block]  # (n_joint_classes, 2)
            joint6 = self.joint_of_row_head[:, self._ray_coset_of_head]
            return theta[joint6, self._ray_side_of_head]  # (BIAS_ROWS, RAY_HEADS)
        return self.ray_bias_free_tables[block][self.orbit_of_row]

    def _ray_live_mask(
        self, dq: torch.Tensor, dr: torch.Tensor, raylen: torch.Tensor | None
    ) -> torch.Tensor:
        """(B, RAY_HEADS, N, N) bool live-ray mask (plan L2): head 2c+s attends
        key j from query i iff i == j (self, always live) or the offset is
        c-axis-aligned with signed magnitude kk, 1 <= |kk| <= RAY_REACH and,
        with blockers on, |kk| <= raylen[i, s, c, sign(kk)]. RAY_BLOCKERS=0
        drops the raylen term (geometric rays — the plan L6 control)."""

        n = dq.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=dq.device).unsqueeze(0)
        aligned = (dr == 0, dq == 0, dq == -dr)
        kks = (dq, dr, dq)
        heads = []
        for h6 in range(RAY_HEADS):
            c, s = h6 // 2, h6 % 2
            kk = kks[c]
            live = aligned[c] & (kk != 0) & (kk.abs() <= RAY_REACH)
            if self._ray_blockers:
                pos = raylen[..., s * 6 + c * 2].to(kk.dtype)  # (B, N) query rows
                neg = raylen[..., s * 6 + c * 2 + 1].to(kk.dtype)
                reach = torch.where(kk > 0, pos[:, :, None], neg[:, :, None])
                live = live & (kk.abs() <= reach)
            heads.append(live | eye)
        return torch.stack(heads, dim=1)

    def _build_ray_bias(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        raylen: torch.Tensor | None,
        block: int,
    ) -> torch.Tensor:
        """(B, RAY_HEADS, N, N) additive bias for ray block `block` — the
        materialized reference path: the expanded table gathered per cells-only
        pair plus PAD_KEY_MASK_VALUE (fp16-finite, plan L5) where the pair is
        off every live ray or the key is padding. The plain table gather
        carries the theta gradient; fp32 on the grad path, cast at use by the
        attention (mirrors build_attn_bias's dtype convention)."""

        b, n, _ = coords.shape
        dq = coords[:, None, :, 0] - coords[:, :, None, 0]  # (B, Nq, Nk) key - query
        dr = coords[:, None, :, 1] - coords[:, :, None, 1]
        m = self._cell_bias_M
        w = 2 * m + 1
        qi = dq.clamp(-m, m) + m
        ri = dr.clamp(-m, m) + m
        rows = self._cell_bias_lut[(qi * w + ri).reshape(-1)].reshape(b, n, n)
        table = self._ray_bias_table(block)
        bias = table[rows].permute(0, 3, 1, 2)  # (B, RAY_HEADS, Nq, Nk)
        live = self._ray_live_mask(dq, dr, raylen)
        dead = ~live | ~mask[:, None, None, :]
        return bias + torch.where(dead, PAD_KEY_MASK_VALUE, 0.0).to(bias.dtype)

    def _build_ray_flex_bias(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        raylen: torch.Tensor | None,
        block: int,
    ) -> "_FlexRayBias":
        """Flex carrier for ray block `block` (both flex modes): fp32 master
        table on the grad path (differentiable table[row, h] read), fp16 cast
        on serve — the same dtype convention as the A-block flex carriers."""

        table = self._ray_bias_table(block)
        if not torch.is_grad_enabled():
            table = table.to(torch.float16)
        return _FlexRayBias(
            coords, mask, raylen, table, self._cell_bias_lut,
            self._cell_bias_M, self._ray_blockers,
        )

    # --- forward ---------------------------------------------------------------

    def trunk(
        self,
        feats: torch.Tensor,
        nbr: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        raylen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (cells (B,Npad,C), tokens (B,T,C), pre_tokens (B,T,C),
        gather_idx): cells/tokens after LN_final, pre_tokens the RAW token
        stream before it (the register lane's count magnitudes — spec D-S21).
        ``raylen`` (B, Npad, RAYLEN_SLOTS) u8 is required by 'L' layouts with
        ray blockers on (batch["raylen"]); C/A layouts ignore it."""

        b, n, _ = feats.shape
        self_idx = torch.arange(n, device=feats.device).reshape(1, n, 1).expand(b, -1, -1)
        gather_idx = torch.cat([self_idx, nbr], dim=2)  # (B, Npad, 7), tap 0 = self

        x = F.relu(self.stem_ln(self.stem(feats, gather_idx, mask))) * mask.unsqueeze(-1)
        # Bias path is chosen per execution mode: serve_flex (no-grad, fp16 carrier)
        # or train_flex (grad, fp32 carrier) use the in-kernel score_mod with no
        # (B, heads, S, S) materialization; with both flags off the materialized
        # build_attn_bias (+ _BiasGather) branch runs. The pair/key_pad index is
        # block-independent and built once here; block_bias(i) returns the per-block
        # object using bias_free_tables[i] via _block_bias_table.
        grad_on = torch.is_grad_enabled()
        serve_flex = self._serve_flex and not grad_on
        train_flex = self._train_flex and grad_on
        flex_pair = serve_flex and self._flex_pair
        train_flex_pair = train_flex and self._train_flex_pair
        flex = serve_flex or train_flex
        if not flex:
            pair, key_pad = self._build_pair(coords, mask)
        elif flex_pair or train_flex_pair:
            # Block-independent, built once and shared by all attention blocks.
            pair_u8 = self._build_pair_u8(coords, mask)
        attn_seq_lens = None
        if flex_pair and _attn_pair_fused is not None:
            # Per-row key count for the bespoke attention kernel: tokens + the
            # LAST live cell + 1 (conservative if padding were ever
            # non-contiguous; pad keys inside the bound still hit the pair
            # tensor's pad row, so this only affects tile-skipping, not math).
            cell_idx = torch.arange(
                1, mask.shape[1] + 1, device=mask.device, dtype=torch.int32
            )
            attn_seq_lens = NUM_TOKENS + (cell_idx * mask).amax(dim=1)

        def block_bias(i: int):
            if train_flex_pair:
                return self._build_train_flex_pair_bias(pair_u8, i)
            if train_flex:
                return self._build_train_flex_bias(coords, mask, i)
            if flex_pair:
                return self._build_flex_pair_bias(pair_u8, i, attn_seq_lens)
            if serve_flex:
                return self._build_flex_bias(coords, mask, i)
            return self.build_attn_bias(pair, key_pad, i)

        if "L" in self._trunk_layout and self._ray_blockers and raylen is None:
            raise ValueError(
                "trunk layout contains 'L' with ray blockers on but no raylen "
                "input; pass batch['raylen'] (B, Npad, RAYLEN_SLOTS) or build "
                "the net with ray_blockers=False (HEXFIELD_EQ_RAY_BLOCKERS=0)"
            )
        # Ray-tap conv context (spec §2.5): built once per forward, shared by
        # every equipped conv. The sync-free ray gather index is enabled
        # whenever ray-tap is on, including L-free layouts (arm A5); ray-tap
        # always consumes the raylen wire (there is no blockers toggle for it).
        ray_ctx = None
        if self._raytap != "0":
            if raylen is None:
                raise ValueError(
                    "ray-tap convs enabled (raytap="
                    f"{self._raytap!r}) but no raylen input; pass "
                    "batch['raylen'] (B, Npad, RAYLEN_SLOTS) u8 (spec §2.5)"
                )
            ray_idx = _raytap().build_ray_gather_index(coords, mask)
            ray_ctx = _RayTapCtx(
                ray_idx[:, :, self._raytap_slot_lut].to(torch.int64),
                raylen[:, :, self._raytap_raylen_slots],
                ray_idx,
                raylen,
            )

        # Gathered ray-attention serve path (HEXFIELD_EQ_TRITON_RAY=1; spec
        # D-S36/D-S37): no-grad CUDA fp16 at a kernel-supported head_dim only.
        # The (B, Npad, 32) geometric gather index is block-independent and
        # built once (sentinel Npad = absent key), alongside the per-row
        # live-key bound for tile skipping; blockers-off nets never read
        # raylen (D-S16), so an empty u8 dummy rides the carrier. Any miss
        # falls through to the flex/materialized paths below.
        ray_gather = (
            _ray_attn_fused is not None
            and "L" in self._trunk_layout
            and not grad_on
            and x.is_cuda
            # serve-half streams are fp16; the autocast serve mode re-upcasts
            # at every LN but its attention q/k/v still land fp16.
            and (x.dtype == torch.float16 or _cuda_autocast_fp16())
            and (x.shape[-1] // RAY_HEADS) in (16, 32, 64, 128)
        )
        if ray_gather:
            ray_idx = _ray_gather_index_fused(coords, mask)
            rg_cell = torch.arange(
                1, mask.shape[1] + 1, device=mask.device, dtype=torch.int32
            )
            ray_seq_lens = (rg_cell * mask).amax(dim=1)
            ray_raylen = (
                raylen
                if self._ray_blockers
                else torch.empty(0, dtype=torch.uint8, device=x.device)
            )
            # Per-device cache of the slot -> bias-table-row LUT (32 rows).
            rg_rows = getattr(self, "_ray_slot_rows_dev", None)
            if rg_rows is None or rg_rows.device != x.device:
                rg_rows = _ray_slot_bias_rows().to(x.device)
                self._ray_slot_rows_dev = rg_rows

        def ray_bias(i: int):
            # Serve fast path (HEXFIELD_EQ_TRITON_RAY): the gathered-kernel
            # carrier; each slot's relative offset is fixed, so this block's
            # expanded table collapses to a (32, RAY_HEADS) fp16 slot bias.
            if ray_gather:
                slot_bias = self._ray_bias_table(i)[rg_rows].to(torch.float16)
                return _RayGatherBias(
                    ray_idx, slot_bias, ray_raylen, ray_seq_lens,
                    self._ray_blockers,
                )
            # Otherwise L blocks ride the plain flex carrier under either flex
            # mode; the flex-pair / A-block Triton attention variants never
            # apply to L. With both flex flags off, the materialized reference
            # bias runs.
            if flex:
                return self._build_ray_flex_bias(coords, mask, raylen, i)
            return self._build_ray_bias(coords, mask, raylen, i)

        seq_mask = torch.cat([mask.new_ones(b, NUM_TOKENS), mask], dim=1)

        # Equivariant tokens are stored as (NUM_TOKENS, C_ORBIT) and broadcast
        # (tiled) over the 12 slots to a slot-constant (invariant) (NUM_TOKENS, C).
        base_tokens = (
            self.tokens.repeat(1, GROUP_ORDER) if self._equivariant else self.tokens
        )
        tokens = base_tokens.unsqueeze(0).expand(b, -1, -1)
        # D8 (spec D-S27): with the register lane on, the loop-carried token
        # stream stays fp32 between A blocks so half-precision serve does not
        # ulp-round late-block count writes away ((B, T, C) — negligible cost;
        # a no-op on the uniform fp32 train path). Projections inside the
        # refresh run in the compute dtype; only the residual carry is fp32.
        if self._reg_lane:
            tokens = tokens.float()
        # Walk the layout string; the op sequence for the default layout is
        # identical to the historical hand-unrolled CCC A CCC A CC A body. After
        # every attention block except the last the joint sequence is split back
        # into (tokens, cells); the last block's output goes to ln_final whole.
        layout = self._trunk_layout
        ci = 0
        ai = 0
        li = 0
        seq = None
        for pos, kind in enumerate(layout):
            if kind == "C":
                if self._reg_tok_read:
                    x = x + self.tok_reads[ci](tokens.to(x.dtype)) * mask.unsqueeze(-1)
                x = self.conv_blocks[ci](x, gather_idx, mask, ray_ctx=ray_ctx)
                if self._reg_lane:
                    tokens = self.registers[ci](tokens, x, mask)
                ci += 1
            elif kind == "L":
                # Cells-only ray attention (plan L3); a non-A block, so the
                # register lane attaches here too (plan R6).
                if self._reg_tok_read:
                    x = x + self.tok_reads_l[li](tokens.to(x.dtype)) * mask.unsqueeze(-1)
                x = self.ray_blocks[li](x, ray_bias(li), mask)
                if self._reg_lane:
                    tokens = self.registers_l[li](tokens, x, mask)
                li += 1
            else:
                seq = self.attn_blocks[ai](
                    torch.cat([tokens.to(x.dtype), x], dim=1), block_bias(ai), seq_mask
                )
                ai += 1
                if pos != len(layout) - 1:
                    tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
                    if self._reg_lane:
                        tokens = tokens.float()  # D8: fp32 carry between A blocks
        # The PRE-ln_final token stream carries the register lane's accumulated
        # count magnitudes (ln_final normalizes them away); the widened invariant
        # reads consume it as an extra block (spec D-S21).
        pre_tokens = seq[:, :NUM_TOKENS]
        seq = self.ln_final(seq)
        tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
        return x * mask.unsqueeze(-1), tokens, pre_tokens, gather_idx

    def _pooled(self, cells: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean of LN_final cell vectors (pad rows excluded)."""

        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(cells.dtype)
        return (cells * mask.unsqueeze(-1)).sum(dim=1) / counts

    def forward(
        self,
        feats: torch.Tensor,
        nbr: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        raylen: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cells, tokens, pre_tokens, gather_idx = self.trunk(
            feats, nbr, mask, coords, raylen
        )
        pooled = self._pooled(cells, mask)
        out = {
            "policy": self._policy_logits(
                self.policy_conv, self.policy_expand, self.policy_head,
                cells, gather_idx, mask,
            ),
            "opp_policy": self._policy_logits(
                self.opp_policy_conv, self.opp_policy_expand, self.opp_policy_head,
                cells, gather_idx, mask,
            ),
            # Auxiliary soft policy (train-only): not in forward_policy_value
            # (serve), like cell_q and opp_policy.
            "soft_policy": self._policy_logits(
                self.soft_policy_conv, self.soft_policy_expand, self.soft_policy_head,
                cells, gather_idx, mask,
            ),
            "value": self.value_head(
                F.relu(
                    self.value_reduction(
                        self._value_input(tokens, range(NUM_TOKENS), pooled, pre_tokens)
                    )
                )
            ),
        }
        out["cell_q"] = self._cell_q_logits(
            self.cell_q_conv, self.cell_q_expand, self.cell_q_head,
            cells, gather_idx, mask,
        )
        aux = F.relu(
            self.aux_reduction(self._value_input(tokens, (2, 3), pooled, pre_tokens))
        )
        for horizon, head in self.stv_heads.items():
            out[f"stvalue_{horizon}"] = head(aux)
        ml = F.relu(
            self.ml_reduction(self._value_input(tokens, (4, 5), pooled, pre_tokens))
        )
        out["moves_left"] = self.moves_left_head(ml)
        return out

    def forward_policy_value(
        self,
        feats: torch.Tensor,
        nbr: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        raylen: torch.Tensor | None = None,
        *,
        request_moves_left: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Serve forward: policy + value always; the aux reduction +
        moves-left top only when requested; opp-policy never (train-only)."""

        cells, tokens, pre_tokens, gather_idx = self.trunk(
            feats, nbr, mask, coords, raylen
        )
        pooled = self._pooled(cells, mask)
        # Head inputs follow the head weights' dtype. A no-op on the uniform
        # paths (train fp32, autocast serve); on the fp16-serve path
        # (HEXFIELD_SERVE_HALF) the evaluator keeps the value/ml tops fp32, so
        # the cast upgrades their inputs and the scalar heads run fp32.
        vin = self._value_input(tokens, range(NUM_TOKENS), pooled, pre_tokens)
        vin = vin.to(self.value_reduction.weight.dtype)
        out = {
            "policy": self._policy_logits(
                self.policy_conv, self.policy_expand, self.policy_head,
                cells, gather_idx, mask,
            ),
            "value": self.value_head(F.relu(self.value_reduction(vin))),
        }
        if request_moves_left:
            mlin = self._value_input(tokens, (4, 5), pooled, pre_tokens)
            mlin = mlin.to(self.ml_reduction.weight.dtype)
            ml = F.relu(self.ml_reduction(mlin))
            out["moves_left"] = self.moves_left_head(ml)
        return out

    def _inv_read(self, v: torch.Tensor) -> torch.Tensor:
        """One widened fiber-invariant read block (spec D-S20): expand the
        C-fiber to INV_READ_EXPAND*C, then group-pool to an invariant
        INV_READ_EXPAND*C_ORBIT vector (passthrough: no pool)."""

        y = self.inv_read(v)
        return _eq.group_pool(y) if EQUIVARIANT else y

    def _value_input(
        self, tokens: torch.Tensor, idx, pooled: torch.Tensor,
        pre_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # Invariant head input (docs/DERIVATION §7 + spec D-S20/D-S21): one
        # widened read block per selected token, one for the pooled-cell fiber,
        # and one for the PRE-ln_final token mean (the register lane's count
        # magnitudes; the mean over tokens of covariant fibers is covariant, so
        # the pooled read is invariant). The value head passes idx =
        # range(NUM_TOKENS); aux/ml pass their pairs.
        blocks = [self._inv_read(tokens[:, i]) for i in idx]
        blocks.append(self._inv_read(pooled))
        blocks.append(self._inv_read(pre_tokens.mean(dim=1).to(tokens.dtype)))
        return torch.cat(blocks, dim=1)

    def _policy_logits(
        self,
        conv: HexNodeConv,
        expand: nn.Module,
        head: nn.Linear,
        cells: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        y = expand(F.relu(conv(cells, gather_idx, mask)))
        if EQUIVARIANT:
            y = _eq.group_pool(y)  # (B, Npad, POLICY_READ_EXPAND*C_ORBIT)
        return head(y).squeeze(-1) * mask  # (B, Npad); pad rows zeroed

    def _cell_q_logits(
        self,
        conv: HexNodeConv,
        expand: nn.Module,
        head: nn.Linear,
        cells: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        y = expand(F.relu(conv(cells, gather_idx, mask)))
        if EQUIVARIANT:
            y = _eq.group_pool(y)  # (B, Npad, POLICY_READ_EXPAND*C_ORBIT)
        return head(y) * mask.unsqueeze(-1)  # (B, Npad, VALUE_BINS); pad rows zeroed
