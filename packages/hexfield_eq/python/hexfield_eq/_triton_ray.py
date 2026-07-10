"""Gathered local-attention Triton kernel for L (ray-attention) blocks
(serve-only, opt-in; spec D-S36/D-S37).

An L-block query cell attends at most 31 distinct cells: itself plus the
<= 30 on-axis neighbours ``{i + k*a : a in {Q, R, QR}, k in +-1..+-RAY_REACH}``
— the 6 heads (3 win-axis cosets x {own, opp}) share those GEOMETRIC cells and
differ only in which axis is live and in the per-side raylen gating. So instead
of an (B, RAY_HEADS, N, N) additive mask (materialized path) or an N^2 flex
score_mod, serve gathers:

  * a (B, Npad, RAY_GATHER_SLOTS=32) int32 index built ONCE per forward from
    coords (block-independent, shared by every L block): slot 0 = self, slots
    1..30 = the fixed (axis, dir, k) offsets, slot 31 unused; value = the key
    cell's row or the sentinel ``Npad`` (= "absent": off-support / pad cell).
    Built by a per-batch coordinate JOIN — sort the live cells' packed axial
    keys, then binary-search (searchsorted) each query cell's neighbour key —
    O(B*N*32 + B*N*log N), no N^2 object anywhere (this deviates from the D-S36
    sketch's `_build_pair_u8` pairwise-delta reuse precisely to keep that
    property). Every allocation shape is a function of coords.shape (B, Npad)
    ALONE, never of the data, so the build performs ZERO device->host syncs; it
    is wrapped in a custom op (opaque to torch.compile) and is now CUDA-graph
    capturable in principle (shape-static given fixed B/Npad). Actually enabling
    it under the graphs serve path is a separate validation and stays gated by
    HEXFIELD_EQ_TRITON_RAY (default off).
  * a per-block (32, RAY_HEADS) fp16 "slot bias": each slot's relative offset
    is FIXED, so the expanded (BIAS_ROWS, RAY_HEADS) ray bias table collapses
    to one row per slot (rows via :func:`slot_bias_rows`); no in-kernel LUT.

The kernel runs one program per (batch, head, query-tile of BM rows): load the
(BM, 32) index tile, gather K/V fp16 (BM, 32, D) for THIS head, apply the
head's liveness (slot 0 always; else axis == coset and, with blockers on,
``k <= raylen[i, side*6 + axis*2 + dir]`` read straight from the u8 wire
buffer), add the slot bias, single-pass fp32 softmax over the 32 slots (no
online rescan — the whole key set is one tile), accumulate V in fp32, store
fp16. Dead slots get the additive PAD_KEY_MASK_VALUE = -3e4 exactly like the
reference paths: exp(-3e4 - m) underflows to 0.0 in fp32, so a masked pair
contributes bit-identically nothing whether it is "present with -3e4" (the
materialized N^2 softmax) or simply absent from the gather. Query tiles beyond
``seq_lens[b]`` (the last live cell + 1) are skipped and store zeros, the
`_triton_attn.py` pad-row convention.

Exposed as the ``hexfield_eq::ray_attn`` / ``hexfield_eq::ray_gather_index``
custom ops (with fake kernels) so the serve torch.compile graph keeps them
in-graph as opaque calls. Enabled via HEXFIELD_EQ_TRITON_RAY=1 (default off);
model.trunk builds the _RayGatherBias carrier only on the no-grad CUDA fp16
serve path at head_dim in {16, 32, 64, 128}, and RayAttention.forward routes
it here; every other combination falls through to the flex/materialized paths.
Inside the op, any kernel launch/compile failure is memoized per head_dim
(`_triton_attn.py` idiom) and served from :func:`_ray_ref`, a gathered pure
-torch reference with the same math (also the CPU test oracle).

Tile constants come from env (HEXFIELD_RAY_BM / HEXFIELD_RAY_WARPS, read once
at import, the HEXFIELD_ATTN_* precedent) so the bench matrix
(scripts/bench_eq_ray_kernel.py) can sweep them.

A second kernel `_ray_attn_kernel_v2` (all-heads-per-program: grid
(cdiv(n, BM), b), the RAY_HEADS heads looped INTERNALLY) shares the (BM, 32)
index tile and slot-liveness metadata across heads so the per-head K/V gathers
of the SAME 32 key rows hit warm L2 lines instead of being re-issued by 6
independent programs. Same numerics as v1. Selected at CALL time by the plain
module global `_USE_V2` (init from HEXFIELD_RAY_V2, DEFAULT OFF), so tests / the
bench flip it at runtime; a v2 compile/launch failure memoizes per head_dim in
`_RAY_V2_FAILED` and falls back to v1 (then the reference). Its tile constants
ride HEXFIELD_RAY_V2_BM / HEXFIELD_RAY_V2_WARPS (default to the v1 values).
"""

from __future__ import annotations

import math
import os
import warnings

import torch

from .constants import RAY_REACH, RAYLEN_SLOTS
from .geometry import rel_bias_index

try:  # pragma: no cover - triton ships with cuda torch builds
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False

# Triton's CompilationError location moves between versions; import defensively.
try:  # pragma: no cover - triton internals vary by version
    from triton.compiler.errors import CompilationError as _TritonCompileError
except Exception:  # pragma: no cover
    try:
        from triton.compiler import CompilationError as _TritonCompileError
    except Exception:
        _TritonCompileError = ()

_BM = int(os.environ.get("HEXFIELD_RAY_BM", "16"))
_WARPS = int(os.environ.get("HEXFIELD_RAY_WARPS", "4"))

# --- kernel variant select (v1 program-per-(batch,head) vs v2 all-heads-per-
# program) -----------------------------------------------------------------------
# v2 (`_ray_attn_kernel_v2`) launches one program per (query-tile, batch) and
# loops the RAY_HEADS heads INTERNALLY, so the (BM, 32) index tile + slot
# liveness metadata are loaded/derived ONCE and the per-head K/V gathers of the
# SAME 32 key rows land close in time (L2 locality across the head loop). It is
# numerically identical to v1 (fp32 scores/softmax over 32 slots, additive slot
# bias, -30000.0 dead fill, fp32 V accumulate, fp16 store).
#
# `_USE_V2` is a PLAIN MODULE GLOBAL read at CALL time inside the ray_attn op, so
# tests / the bench can flip it at runtime (monkeypatch or bare assignment).
# Default OFF (v1) — the deploy decision is made after an idle-GPU bench.
# NOTE: flipping it mid CUDA-graph-capture is NOT supported (the captured graph
# freezes whichever kernel was live at capture time).
_USE_V2 = os.environ.get("HEXFIELD_RAY_V2", "0") == "1"
# v2 tile constants: default to the v1 values so a sweep can widen BM (more heads
# per program amortize a larger query tile better). Read once at import, the
# HEXFIELD_RAY_* precedent; bench sweeps them by re-launching with the env set.
_V2_BM = int(os.environ.get("HEXFIELD_RAY_V2_BM", str(_BM)))
_V2_WARPS = int(os.environ.get("HEXFIELD_RAY_V2_WARPS", str(_WARPS)))

# Additive dead-pair value; MUST match model.PAD_KEY_MASK_VALUE (fp16-finite,
# exp-underflows to exactly 0.0 in fp32). Not imported from model.py: that
# import direction would be a cycle (model.py imports this module).
_DEAD_SCORE = -3.0e4

# --- slot layout ----------------------------------------------------------------
# RAY_GATHER_SLOTS = 32: slot 0 = self; slot 1 + axis*10 + dir*5 + (k-1) for
# axis in {0=Q, 1=R, 2=QR}, dir in {0:+, 1:-}, k in 1..RAY_REACH; slot 31 is
# padding (always the sentinel). The +-axis vector convention matches
# model._ray_live_mask: for head coset c the signed magnitude kk is dq (Q, QR)
# or dr (R), so dir 0 (kk > 0) is +k * axis_vector. The packing arithmetic
# assumes RAY_REACH == 5 (structural: WINDOW_LEN - 1), asserted below.
RAY_GATHER_SLOTS = 32
_AXIS_VECS = ((1, 0), (0, 1), (1, -1))  # Q, R, QR (constants.py order)
if RAY_REACH != 5:  # pragma: no cover - structural constant
    raise AssertionError(
        f"RAY_REACH={RAY_REACH} != 5: the 10-slots-per-axis packing in "
        "_triton_ray.py assumes the WINDOW_LEN-1 reach"
    )


def slot_offset_table() -> torch.Tensor:
    """(RAY_GATHER_SLOTS, 2) long CPU tensor of each slot's (dq, dr) relative
    offset. Slot 0 = (0, 0) (self); slot 31 = (0, 0) too but is never gathered
    (the builder forces it to the sentinel)."""

    off = torch.zeros(RAY_GATHER_SLOTS, 2, dtype=torch.long)
    for c, (aq, ar) in enumerate(_AXIS_VECS):
        for d, sgn in enumerate((1, -1)):
            for k in range(1, RAY_REACH + 1):
                s = 1 + c * 10 + d * 5 + (k - 1)
                off[s, 0] = sgn * k * aq
                off[s, 1] = sgn * k * ar
    return off


def slot_bias_rows() -> torch.Tensor:
    """(RAY_GATHER_SLOTS,) long CPU tensor: the bias-table row of each slot's
    fixed relative offset (rel_bias_index; every ray offset has hex-dist <=
    RAY_REACH, inside the exact disk). Slot 31's row is the self row — it is
    never live, the value is only a safe in-bounds placeholder."""

    off = slot_offset_table()
    return torch.tensor(
        [rel_bias_index(int(off[s, 0]), int(off[s, 1])) for s in range(RAY_GATHER_SLOTS)],
        dtype=torch.long,
    )


# --- gather-index builder ---------------------------------------------------------
#
# The gather index answers, for every query cell i and each of the 32 fixed ray
# offsets, "which row holds the LIVE cell at coords[i] + offset, or the sentinel
# Npad if none?". That is a per-batch coordinate JOIN between the 32 neighbour
# keys of each query cell and the live cells' coords.
#
# The previous build realized the join with a dense (qspan x rspan) scatter grid
# whose ALLOCATION shape (gh*gw+1) was read off the live-coord extents via
# `int(...amin())` / `bool(live.any())` — ~8 device->host syncs per serve
# forward, and the documented reason the CUDA-graphs serve path had to keep this
# kernel off.
#
# Why NOT re-size that grid from a hex-disk span bound (N = 3R^2+3R+1 => span
# 2R+1): the supports here are NOT hex disks. LEGAL_RADIUS = 8 and "Hexo has no
# fixed board bounds" (rust board.rs), so a support is the UNION of radius-
# (SUPPORT_RADIUS+1) axial disks around the stones (support.py). For an elongated
# stone chain — ordinary in a connection game — that union is a long, thin
# "sausage" whose coordinate span grows like O(N/thickness) = O(N), far past the
# O(sqrt N) span of a disk of N cells. A grid sized to the disk span would
# silently alias two live cells (or, behind a guard, crash) on such a legitimate
# board, violating "same output for every valid input"; a grid sized to the true
# O(N) worst-case span is an O(N^2) object. So instead of a bounding-box grid we
# do the join directly with sort + searchsorted: every allocation shape is a
# function of coords.shape (B, N) alone, giving ZERO syncs, correctness for ANY
# board shape, O(B*N*32 + B*N*log N), no N^2 object, and shape-static capture.

# Order-injective axial pack: key = q * _KEY_STRIDE + r. Injective (=> key
# equality is coordinate equality) as long as |r| < _KEY_STRIDE/2; the engine's
# i16 coords (|q|,|r| < 2^15) plus the <= 5-cell ray offset satisfy that with
# room to spare. int64 throughout (max |key| ~ 2^35).
_KEY_STRIDE = 1 << 20
# Pad cells take this key: larger than any real packed key (< ~2^36) so they sort
# past every query key and are NEVER matched (excluded as gatherable keys).
_PAD_KEY = 1 << 40
# Domain guard: engine HexCoord is i16, which is what makes the pack injective.
_COORD_ABS_LIMIT = 1 << 15

# Per-device cache of the (32, 2) int64 slot-offset table. Copying the freshly
# built CPU table to the device every forward is itself a host->device sync (the
# last remaining one after the extents were tensorized); cache it so the steady
# state reads a resident device tensor. Keyed by torch.device.
_SLOT_OFFSETS_DEV: dict = {}


def _slot_offsets_on(dev: torch.device) -> torch.Tensor:
    off = _SLOT_OFFSETS_DEV.get(dev)
    if off is None:
        off = slot_offset_table().to(device=dev, dtype=torch.int64)
        _SLOT_OFFSETS_DEV[dev] = off
    return off


def _build_ray_gather_index_impl(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, N, RAY_GATHER_SLOTS) int32 gather index; sentinel N = absent key.

    Per-batch coordinate join with NO device->host sync: pack each cell's axial
    (q, r) into an order-injective int64 key, sort the LIVE keys (pad cells forced
    to a large sentinel key so they never match and sort to the end), then for
    every (query cell, slot) binary-search the query's neighbour key
    ``coords[i] + slot_offset`` and read back the matching live row — or the
    sentinel ``N`` when no live cell holds that coordinate. Pad rows never enter
    the sorted table, so a pad cell can never be gathered as a key even if its
    garbage coords alias a live cell; pad QUERY rows produce garbage gather rows
    exactly like every other bias path (their outputs are re-zeroed / tile-skipped
    downstream), except slot 0 which is ALWAYS the row itself so no softmax row is
    ever empty. Bit-identical to the old bounding-box-grid build for every valid
    (distinct-live-coord) input, but with no data-dependent allocation shape."""

    b, n, _ = coords.shape
    dev = coords.device
    q = coords[..., 0].to(torch.int64)
    r = coords[..., 1].to(torch.int64)
    live = mask.to(torch.bool)

    # Device-side domain guard (fail loud, don't silently alias): the packed key
    # is injective only while live coords stay in the i16 range the engine
    # guarantees. torch._assert_async does NOT sync on CUDA; on CPU a plain
    # assert with .item() is free. Pad rows are exempt (their coords are unused).
    in_domain = (
        (q.abs() < _COORD_ABS_LIMIT) & (r.abs() < _COORD_ABS_LIMIT)
    ) | ~live
    if coords.is_cuda:
        if hasattr(torch, "_assert_async"):  # pragma: no cover - CUDA-only path
            torch._assert_async(in_domain.all())
    else:
        assert bool(in_domain.all()), (
            "ray gather: a live coord is outside the packable i16 range"
        )

    # Sorted live-key table; pad cells -> sentinel key so they are never matched.
    cell_key = torch.where(live, q * _KEY_STRIDE + r, torch.full_like(q, _PAD_KEY))
    sorted_key, sorted_row = torch.sort(cell_key, dim=1)  # (B, N), (B, N) int64

    # The 32 neighbour keys per query cell (coords + fixed ray offset).
    off = _slot_offsets_on(dev)  # (32, 2) int64, resident on `dev`
    kq = q.unsqueeze(-1) + off[:, 0]  # (B, N, 32)
    kr = r.unsqueeze(-1) + off[:, 1]
    query_key = (kq * _KEY_STRIDE + kr).reshape(b, n * RAY_GATHER_SLOTS)

    # Binary search each neighbour key into the sorted live keys; a hit iff the
    # key at the found slot equals it (clamp guards the past-the-end position).
    pos = torch.searchsorted(sorted_key, query_key).clamp_max_(n - 1)
    hit = sorted_key.gather(1, pos) == query_key
    row = sorted_row.gather(1, pos)
    out = torch.where(
        hit, row.to(torch.int32), torch.full((), n, dtype=torch.int32, device=dev)
    ).reshape(b, n, RAY_GATHER_SLOTS)

    # Slot 0 = self for EVERY row (incl. pad rows: keeps their softmax finite);
    # slot 31 = permanent sentinel.
    out[:, :, 0] = torch.arange(n, device=dev, dtype=torch.int32).unsqueeze(0)
    out[:, :, RAY_GATHER_SLOTS - 1] = n
    return out


@torch.library.custom_op("hexfield_eq::ray_gather_index", mutates_args=())
def build_ray_gather_index(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _build_ray_gather_index_impl(coords, mask)


@build_ray_gather_index.register_fake
def _build_ray_gather_index_fake(coords, mask):
    b, n, _ = coords.shape
    return coords.new_empty((b, n, RAY_GATHER_SLOTS), dtype=torch.int32)


# --- gathered reference (fallback + CPU oracle) ------------------------------------


def _ray_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx: torch.Tensor,
    slot_bias: torch.Tensor,
    raylen: torch.Tensor,
    seq_lens: torch.Tensor,
    blockers: bool,
) -> torch.Tensor:
    """Gathered ray attention in pure torch — the kernel's numerical twin
    (fp32 math, fp16 out, rows >= seq_lens zeroed). Runs on any device/dtype;
    serves compile-failure fallbacks and the CPU parity gates."""

    b, h, n, d = q.shape
    s = RAY_GATHER_SLOTS
    scale = 1.0 / math.sqrt(d)
    idxl = idx.long()
    present = idxl < n
    idxc = torch.where(present, idxl, torch.zeros_like(idxl))
    gi = idxc.view(b, 1, n * s, 1).expand(b, h, n * s, d)
    kg = k.float().gather(2, gi).view(b, h, n, s, d)
    vg = v.float().gather(2, gi).view(b, h, n, s, d)
    scores = (q.float().unsqueeze(3) * kg).sum(-1) * scale  # (B, H, N, S)

    sl = torch.arange(s, device=q.device)
    t = sl - 1
    ax = torch.div(t, 10, rounding_mode="trunc")  # C-style, matches the kernel
    dr_ = torch.div(t % 10, 5, rounding_mode="trunc")
    kk = t % 5 + 1
    geo = (sl >= 1) & (sl <= 30)
    hh = torch.arange(h, device=q.device)
    on_axis = geo[None, :] & (ax[None, :] == (hh[:, None] // 2))  # (H, S)
    if blockers:
        rl_idx = ((hh[:, None] % 2) * 6 + ax[None, :] * 2 + dr_[None, :]).clamp(
            0, RAYLEN_SLOTS - 1
        )
        rl = raylen.long()[:, :, rl_idx.reshape(-1)].view(b, n, h, s)
        rl = rl.permute(0, 2, 1, 3)  # (B, H, N, S)
        reach_ok = kk[None, None, None, :] <= rl
    else:
        reach_ok = torch.ones(1, 1, 1, s, dtype=torch.bool, device=q.device)
    live = present[:, None, :, :] & (
        (sl == 0)[None, None, None, :] | (on_axis[:, None, :][None] & reach_ok)
    )
    bias = slot_bias.float().t()[None, :, None, :]  # (1, H, 1, S)
    scores = torch.where(live, scores + bias, torch.full_like(scores, _DEAD_SCORE))
    attn = torch.softmax(scores, dim=-1)
    out = (attn.unsqueeze(-1) * vg).sum(3)  # (B, H, N, D)
    row_live = (
        torch.arange(n, device=q.device)[None, :] < seq_lens.long()[:, None]
    )  # (B, N)
    out = out * row_live[:, None, :, None]
    return out.to(torch.float16)


# --- the Triton kernel ---------------------------------------------------------------

if HAVE_TRITON:

    _TRITON_VER = getattr(triton, "__version__", "?")
    _RAY_FAILED: set = set()

    def _mark_ray_failed(d: int, err: Exception) -> None:
        """Record a failing head_dim and warn ONCE (called only when d is new)."""
        _RAY_FAILED.add(d)
        kind = (
            "compile error"
            if isinstance(err, _TritonCompileError)
            else f"{type(err).__name__}"
        )
        warnings.warn(
            f"hexfield: triton ray_attn failed for head_dim={d} ({kind}) under "
            f"triton {_TRITON_VER}; using the gathered reference path for this "
            "shape.",
            RuntimeWarning,
            stacklevel=2,
        )

    @triton.jit
    def _ray_attn_kernel(
        q_ptr, k_ptr, v_ptr, idx_ptr, bias_ptr, rl_ptr, seq_ptr, out_ptr,
        sqb, sqh, sqs,
        skb, skh, sks,
        svb, svh, svs,
        sob, soh, sos,
        sib, sis,
        srb, srs,
        H, N, scale,
        BLOCKERS: tl.constexpr,
        D: tl.constexpr, BM: tl.constexpr, S: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        coset = h // 2
        side = h % 2
        lo_m = pid_m * BM
        offs_m = lo_m + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        m_ok = offs_m < N
        o_ptrs = out_ptr + b * sob + h * soh + offs_m[:, None] * sos + offs_d[None, :]

        n_live = tl.load(seq_ptr + b)
        # Whole tile beyond the live rows: downstream re-zeroes pad rows anyway,
        # so store zeros and skip every gather (the _triton_attn convention).
        if lo_m >= n_live:
            tl.store(
                o_ptrs, tl.zeros((BM, D), dtype=tl.float16), mask=m_ok[:, None]
            )
            return
        m_live = offs_m < n_live

        # Static slot metadata (S = 32 slots): slot 0 self, 1..30 geometric,
        # 31 padding. Triton integer division is C-style (trunc toward zero),
        # so slot 0's ax = (0-1)//10 = 0 — harmless, geo masks it out.
        sl = tl.arange(0, S)
        t = sl - 1
        ax = t // 10
        dr_ = (t % 10) // 5
        kk = t % 5 + 1
        geo = (sl >= 1) & (sl <= 30)
        on_axis = geo & (ax == coset)

        # Gather index tile (BM, S); the sentinel N marks absent keys.
        idx = tl.load(
            idx_ptr + b * sib + offs_m[:, None] * sis + sl[None, :],
            mask=m_live[:, None],
            other=N,
        )
        present = idx < N
        idx_c = tl.where(present, idx, 0).to(tl.int64)

        if BLOCKERS:
            # raylen[i, side*6 + ax*2 + dir], u8; loads masked to on-axis slots
            # (off-axis lanes would index out of the 12-slot row).
            rl = tl.load(
                rl_ptr + b * srb + offs_m[:, None] * srs
                + (side * 6 + ax * 2 + dr_)[None, :],
                mask=m_live[:, None] & on_axis[None, :],
                other=0,
            ).to(tl.int32)
            reach_ok = kk[None, :] <= rl
        else:
            # Geometric rays (RAY_BLOCKERS=0): constant reach; every gathered
            # slot satisfies k <= RAY_REACH by construction (import-asserted 5).
            reach_ok = kk[None, :] <= 5
        live = present & ((sl == 0)[None, :] | (on_axis[None, :] & reach_ok))

        q = tl.load(
            q_ptr + b * sqb + h * sqh + offs_m[:, None] * sqs + offs_d[None, :],
            mask=m_live[:, None],
            other=0.0,
        ).to(tl.float32)
        kv_mask = live[:, :, None]
        k = tl.load(
            k_ptr + b * skb + h * skh + idx_c[:, :, None] * sks
            + offs_d[None, None, :],
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[:, None, :] * k, 2) * scale  # (BM, S)
        bias = tl.load(bias_ptr + sl * H + h).to(tl.float32)
        # Dead slots: the additive -3e4 of the reference paths (exp underflows
        # to exactly 0.0 in fp32 — bit-identical softmax weights).
        scores = tl.where(live, scores + bias[None, :], -30000.0)

        m_i = tl.max(scores, 1)
        LOG2E: tl.constexpr = 1.4426950408889634
        p = tl.math.exp2((scores - m_i[:, None]) * LOG2E)
        l_i = tl.sum(p, 1)
        v = tl.load(
            v_ptr + b * svb + h * svh + idx_c[:, :, None] * svs
            + offs_d[None, None, :],
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        acc = tl.sum(p[:, :, None] * v, 1) / l_i[:, None]
        # Dead rows inside a partial tile (all-dead softmax): p is uniform and
        # v is zero, so acc is already exactly 0 — store as-is.
        tl.store(o_ptrs, acc.to(tl.float16), mask=m_ok[:, None])

    # --- v2: all heads per program (grid = (cdiv(n, BM), b)) ----------------------
    _RAY_V2_FAILED: set = set()

    def _mark_ray_v2_failed(d: int, err: Exception) -> None:
        """Record a failing head_dim for the v2 kernel and warn ONCE. A v2 miss
        is NON-fatal: the op falls back to v1 (which memoizes separately in
        `_RAY_FAILED` and itself falls back to the reference)."""
        _RAY_V2_FAILED.add(d)
        kind = (
            "compile error"
            if isinstance(err, _TritonCompileError)
            else f"{type(err).__name__}"
        )
        warnings.warn(
            f"hexfield: triton ray_attn v2 failed for head_dim={d} ({kind}) under "
            f"triton {_TRITON_VER}; falling back to the v1 kernel / reference for "
            "this shape.",
            RuntimeWarning,
            stacklevel=2,
        )

    @triton.jit
    def _ray_attn_kernel_v2(
        q_ptr, k_ptr, v_ptr, idx_ptr, bias_ptr, rl_ptr, seq_ptr, out_ptr,
        sqb, sqh, sqs,
        skb, skh, sks,
        svb, svh, svs,
        sob, soh, sos,
        sib, sis,
        srb, srs,
        N, scale,
        BLOCKERS: tl.constexpr,
        D: tl.constexpr, BM: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    ):
        # One program per (query-tile, batch); the H heads are handled INTERNALLY
        # by the unrolled loop below so the idx tile + slot metadata are loaded /
        # derived ONCE and the per-head K/V gathers of the SAME 32 key rows reuse
        # those registers and hit warm L2 lines. Numerics are bit-for-bit the v1
        # body, just repeated per head.
        pid_m = tl.program_id(0)
        b = tl.program_id(1)
        lo_m = pid_m * BM
        offs_m = lo_m + tl.arange(0, BM)
        offs_d = tl.arange(0, D)
        m_ok = offs_m < N

        n_live = tl.load(seq_ptr + b)
        # Whole tile beyond the live rows: store zeros for EVERY head, skip all
        # gathers (the _triton_attn pad-tile convention, applied per head).
        if lo_m >= n_live:
            zeros = tl.zeros((BM, D), dtype=tl.float16)
            for h in tl.static_range(H):
                o_ptrs = (
                    out_ptr + b * sob + h * soh
                    + offs_m[:, None] * sos + offs_d[None, :]
                )
                tl.store(o_ptrs, zeros, mask=m_ok[:, None])
            return
        m_live = offs_m < n_live

        # --- head-INDEPENDENT metadata (hoisted; computed ONCE) ---
        # Static slot metadata (S = 32): slot 0 self, 1..30 geometric, 31 pad.
        # Triton integer '//' is C-style (trunc), so slot 0's ax=(0-1)//10=0 —
        # harmless, geo masks it out. axis/dir/k and geo do NOT depend on head;
        # only on_axis (via coset) and the raylen side slot do.
        sl = tl.arange(0, S)
        t = sl - 1
        ax = t // 10
        dr_ = (t % 10) // 5
        kk = t % 5 + 1
        geo = (sl >= 1) & (sl <= 30)

        # Gather-index tile (BM, S): the SAME 32 key rows for every head; load once.
        idx = tl.load(
            idx_ptr + b * sib + offs_m[:, None] * sis + sl[None, :],
            mask=m_live[:, None],
            other=N,
        )
        present = idx < N
        idx_c = tl.where(present, idx, 0).to(tl.int64)
        # Base address of this (batch, query-tile)'s raylen rows; the per-head
        # side/axis/dir slot is added inside the loop.
        rl_row = rl_ptr + b * srb + offs_m[:, None] * srs
        is_self = (sl == 0)[None, :]
        LOG2E: tl.constexpr = 1.4426950408889634

        # --- per-head loop: only on_axis / raylen-side / bias-col / q,k,v differ ---
        for h in tl.static_range(H):
            coset = h // 2
            side = h % 2
            on_axis = geo & (ax == coset)
            if BLOCKERS:
                # raylen[i, side*6 + ax*2 + dir], u8; masked to on-axis slots
                # (off-axis lanes would index out of the 12-slot row).
                rl = tl.load(
                    rl_row + (side * 6 + ax * 2 + dr_)[None, :],
                    mask=m_live[:, None] & on_axis[None, :],
                    other=0,
                ).to(tl.int32)
                reach_ok = kk[None, :] <= rl
            else:
                # Geometric rays: constant reach, every gathered slot satisfies
                # k <= RAY_REACH by construction (import-asserted 5).
                reach_ok = kk[None, :] <= 5
            live = present & (is_self | (on_axis[None, :] & reach_ok))

            q = tl.load(
                q_ptr + b * sqb + h * sqh + offs_m[:, None] * sqs + offs_d[None, :],
                mask=m_live[:, None],
                other=0.0,
            ).to(tl.float32)
            kv_mask = live[:, :, None]
            k = tl.load(
                k_ptr + b * skb + h * skh + idx_c[:, :, None] * sks
                + offs_d[None, None, :],
                mask=kv_mask,
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(q[:, None, :] * k, 2) * scale  # (BM, S)
            bias = tl.load(bias_ptr + sl * H + h).to(tl.float32)
            # Dead slots: additive -3e4 (exp underflows to exactly 0.0 in fp32 —
            # bit-identical softmax weights to the reference paths).
            scores = tl.where(live, scores + bias[None, :], -30000.0)

            m_i = tl.max(scores, 1)
            p = tl.math.exp2((scores - m_i[:, None]) * LOG2E)
            l_i = tl.sum(p, 1)
            v = tl.load(
                v_ptr + b * svb + h * svh + idx_c[:, :, None] * svs
                + offs_d[None, None, :],
                mask=kv_mask,
                other=0.0,
            ).to(tl.float32)
            acc = tl.sum(p[:, :, None] * v, 1) / l_i[:, None]
            o_ptrs = (
                out_ptr + b * sob + h * soh
                + offs_m[:, None] * sos + offs_d[None, :]
            )
            # Dead rows inside a partial tile (all-dead softmax): p uniform, v
            # zero, so acc is already exactly 0 — store as-is.
            tl.store(o_ptrs, acc.to(tl.float16), mask=m_ok[:, None])

    def _prep_ray_args(q, k, v, idx, slot_bias, raylen, seq_lens, blockers):
        """Massage the op inputs into the layout both kernels expect (shared by
        v1 and v2): stride(-1)==1 contiguity, fp16 bias, int32 seq_lens, and the
        raylen dummy when blockers are off. Returns the prepared tensors plus the
        raylen (batch, row) strides."""
        if q.stride(-1) != 1:
            q = q.contiguous()
        if k.stride(-1) != 1:
            k = k.contiguous()
        if v.stride(-1) != 1:
            v = v.contiguous()
        idxc = idx.contiguous()
        sb = slot_bias.to(torch.float16).contiguous()
        seq = seq_lens.to(torch.int32).contiguous()
        # Blockers off: raylen is an empty dummy (never dereferenced — BLOCKERS
        # is constexpr); pass seq so the kernel gets a real ptr.
        rl = raylen.contiguous() if blockers else seq
        srb = rl.stride(0) if blockers else 0
        srs = rl.stride(1) if blockers else 0
        return q, k, v, idxc, sb, rl, seq, srb, srs

    def _run_ray_v1(q, k, v, idxc, sb, rl, seq, srb, srs, b, h, n, d, blockers):
        out = torch.empty((b, h, n, d), dtype=torch.float16, device=q.device)
        grid = (triton.cdiv(n, _BM), b * h)
        _ray_attn_kernel[grid](
            q, k, v, idxc, sb, rl, seq, out,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            idxc.stride(0), idxc.stride(1),
            srb, srs,
            h, n, 1.0 / math.sqrt(d),
            BLOCKERS=bool(blockers),
            D=d, BM=_BM, S=RAY_GATHER_SLOTS,
            num_warps=_WARPS,
        )
        return out

    def _run_ray_v2(q, k, v, idxc, sb, rl, seq, srb, srs, b, h, n, d, blockers):
        out = torch.empty((b, h, n, d), dtype=torch.float16, device=q.device)
        grid = (triton.cdiv(n, _V2_BM), b)
        _ray_attn_kernel_v2[grid](
            q, k, v, idxc, sb, rl, seq, out,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            idxc.stride(0), idxc.stride(1),
            srb, srs,
            n, 1.0 / math.sqrt(d),
            BLOCKERS=bool(blockers),
            D=d, BM=_V2_BM, S=RAY_GATHER_SLOTS, H=h,
            num_warps=_V2_WARPS,
        )
        return out


@torch.library.custom_op("hexfield_eq::ray_attn", mutates_args=())
def ray_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx: torch.Tensor,
    slot_bias: torch.Tensor,
    raylen: torch.Tensor,
    seq_lens: torch.Tensor,
    blockers: bool,
) -> torch.Tensor:
    b, h, n, d = q.shape
    if (
        HAVE_TRITON
        and q.is_cuda
        and q.dtype == torch.float16
        and d in (16, 32, 64, 128)
    ):
        # (B,N,H,D).transpose(1,2) views satisfy stride(-1)==1 already; the
        # kernel takes the remaining strides as-is, so no copies here. Both
        # kernels consume the same prepared args.
        prep = _prep_ray_args(q, k, v, idx, slot_bias, raylen, seq_lens, blockers)
        # `_USE_V2` is read HERE (call time), not at import, so a test/bench can
        # flip the module global to steer this dispatch at runtime.
        if _USE_V2 and d not in _RAY_V2_FAILED:
            try:
                return _run_ray_v2(*prep, b, h, n, d, blockers)
            except Exception as err:  # per-arch v2 codegen edge case
                _mark_ray_v2_failed(d, err)  # non-fatal: fall through to v1
        if d not in _RAY_FAILED:
            try:
                return _run_ray_v1(*prep, b, h, n, d, blockers)
            except Exception as err:  # per-arch triton codegen edge case
                _mark_ray_failed(d, err)
    return _ray_ref(q, k, v, idx, slot_bias, raylen, seq_lens, blockers)


@ray_attn.register_fake
def _ray_attn_fake(q, k, v, idx, slot_bias, raylen, seq_lens, blockers):
    return q.new_empty(q.shape, dtype=torch.float16)
