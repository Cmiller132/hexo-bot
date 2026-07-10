"""Gathered ray-attention Triton kernel gate (hexfield_eq; spec D-S36/D-S37).

CPU section (always runs):
  (a) slot-table structure — the 32-slot (axis, dir, k) packing, offset
      uniqueness, and the slot -> bias-row LUT vs rel_bias_index;
  (b) gather-index builder vs model._ray_live_mask — the index's implied
      per-head live set must equal the mask's live set EXACTLY (pad keys
      excluded), blockers on and off, over lone-stone / disk / mixed-size
      padded batches (with an adversarial pad row whose coords alias a live
      cell);
  (c) the hexfield_eq::ray_attn op's gathered math (the _ray_ref fallback the
      op serves on CPU) vs the materialized full-softmax reference at the
      3e-3 serve tolerance — masked pairs absent from the gather are
      mathematically identical to the reference's additive -3e4 (fp32
      exp-underflow to exactly 0);
  (d) the env gate defaults OFF (model module globals are None).

CUDA section (skips on CPU; GPU-POLITE: batch <= 2, Npad <= 256, fp16 only —
the box may be running a prefit; one OOM retry with backoff, then skip):
  (e) the Triton kernel through the RayAttention module vs the fp32
      materialized module path at 3e-3, blockers on/off, lone-stone / disk /
      dense / padded boards;
  (f) full-net serve wiring — trunk() builds the carrier once, routes every L
      block through the op, and matches the unrouted fp16 serve forward.

Runs in the hexgt-build venv via PYTHONPATH=packages/hexfield_eq/python (plus
the shared packages).
"""

from __future__ import annotations

import copy
import os
import random
import time

import pytest
import torch

from hexfield_eq import constants as C
from hexfield_eq._triton_ray import (
    RAY_GATHER_SLOTS,
    _build_ray_gather_index_impl,
    _ray_ref,
    build_ray_gather_index,
    ray_attn,
    slot_bias_rows,
    slot_offset_table,
)
from hexfield_eq.constants import DIRECTIONS
from hexfield_eq.geometry import disk_offsets, rel_bias_index
from hexfield_eq.model import HexfieldNet

RL = C.RAYLEN_SLOTS
KERNEL_ATOL = 3e-3  # the D-S36 parity gate
_AXIS_VECS = ((1, 0), (0, 1), (1, -1))  # Q, R, QR

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="gathered ray kernel is CUDA-only"
)


# --- board fixtures ---------------------------------------------------------------


def _disk_board(radius: int):
    """(cells, n, nbr, coords, mask): one full disk board, batch of 1."""

    cells = disk_offsets(radius)
    n = len(cells)
    cidx = {c: i for i, c in enumerate(cells)}
    nbr = torch.full((1, n, 6), n, dtype=torch.long)
    for i, c in enumerate(cells):
        for d, off in enumerate(DIRECTIONS):
            nb = (c[0] + off[0], c[1] + off[1])
            if nb in cidx:
                nbr[0, i, d] = cidx[nb]
    coords = torch.tensor([[list(c) for c in cells]], dtype=torch.long)
    mask = torch.ones(1, n, dtype=torch.bool)
    return cells, n, nbr, coords, mask


def _padded_batch(r_big: int = 3, r_small: int = 1):
    """(coords, mask): two boards padded to the big board's node count. The
    small board's pad rows carry garbage coords, INCLUDING one row aliasing
    the live center (0, 0) — the builder must never gather a pad cell."""

    big = disk_offsets(r_big)
    small = disk_offsets(r_small)
    n = len(big)
    coords = torch.zeros(2, n, 2, dtype=torch.long)
    mask = torch.zeros(2, n, dtype=torch.bool)
    coords[0] = torch.tensor(big)
    mask[0] = True
    coords[1, : len(small)] = torch.tensor(small)
    mask[1, : len(small)] = True
    for i in range(len(small), n):
        coords[1, i] = torch.tensor([60 + i, -40 - i])  # far garbage
    coords[1, len(small)] = torch.tensor([0, 0])  # adversarial alias of a live cell
    return coords, mask


def _seq_lens(mask: torch.Tensor) -> torch.Tensor:
    n = mask.shape[1]
    ar = torch.arange(1, n + 1, dtype=torch.int32, device=mask.device)
    return (ar * mask).amax(dim=1).to(torch.int32)


def _rand_raylen(mask: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rl = torch.randint(
        0, C.RAY_REACH + 1, (*mask.shape, RL), dtype=torch.uint8, generator=g
    )
    return rl * mask.unsqueeze(-1).cpu()  # pad rows 0 (D-S13)


def _randomize_theta(model: HexfieldNet, seed: int, fp16_exact: bool) -> None:
    """Randomize only the ray bias tables (the default init is all-zero, which
    would make the slot bias vacuous). ``fp16_exact`` rounds the master to
    fp16-representable values so the fp32 reference bias equals the kernel's
    fp16 slot bias bitwise (isolates the gather-vs-full-softmax comparison)."""

    torch.manual_seed(seed)
    tables = (
        model.bias_theta_l if C.GROUP_ORDER == 12 else model.ray_bias_free_tables
    )
    with torch.no_grad():
        for p in tables:
            vals = torch.randn_like(p) * 0.3
            p.copy_(vals.half().float() if fp16_exact else vals)


# --- (a) slot table structure -------------------------------------------------------


def test_slot_table_structure() -> None:
    off = slot_offset_table()
    assert off.shape == (RAY_GATHER_SLOTS, 2)
    assert tuple(off[0].tolist()) == (0, 0)
    seen = set()
    for c, (aq, ar) in enumerate(_AXIS_VECS):
        for d, sgn in enumerate((1, -1)):
            for k in range(1, C.RAY_REACH + 1):
                s = 1 + c * 10 + d * 5 + (k - 1)
                got = tuple(off[s].tolist())
                assert got == (sgn * k * aq, sgn * k * ar), (s, got)
                seen.add(got)
    assert len(seen) == 30, "the 30 geometric offsets must be distinct"

    rows = slot_bias_rows()
    assert rows.shape == (RAY_GATHER_SLOTS,)
    for s in range(31):  # slot 31 is a placeholder, never live
        assert int(rows[s]) == rel_bias_index(int(off[s, 0]), int(off[s, 1])), s


# --- (b) gather index vs _ray_live_mask ---------------------------------------------


def _reconstruct_live(
    idx: torch.Tensor, raylen: torch.Tensor | None, blockers: bool
) -> torch.Tensor:
    """(B, RAY_HEADS, N, N) bool implied by the gather index + the kernel's
    per-head gating rule (a literal transcription of _ray_attn_kernel)."""

    b, n, s = idx.shape
    rec = torch.zeros(b, C.RAY_HEADS, n, n, dtype=torch.bool)
    for bi in range(b):
        for i in range(n):
            for sl in range(s):
                j = int(idx[bi, i, sl])
                if j >= n:
                    continue
                if sl == 0:
                    rec[bi, :, i, j] = True
                    continue
                if sl == 31:
                    continue
                t = sl - 1
                ax, d, k = t // 10, (t % 10) // 5, t % 5 + 1
                for side in range(2):
                    h = 2 * ax + side
                    if blockers:
                        reach = int(raylen[bi, i, side * 6 + ax * 2 + d])
                        if k > reach:
                            continue
                    rec[bi, h, i, j] = True
    return rec


@pytest.mark.parametrize("blockers", [True, False], ids=["blockers", "geometric"])
@pytest.mark.parametrize("board", ["lone", "r2", "r3", "padded"])
def test_gather_index_matches_ray_live_mask(board: str, blockers: bool) -> None:
    if board == "padded":
        coords, mask = _padded_batch()
    else:
        radius = {"lone": 0, "r2": 2, "r3": 3}[board]
        _, _, _, coords, mask = _disk_board(radius)
    b, n = mask.shape
    raylen = _rand_raylen(mask, seed=11) if blockers else None

    model = HexfieldNet(trunk_layout="CLA", ray_blockers=blockers)
    idx = build_ray_gather_index(coords, mask)

    # Structure: dtype/shape, self slot, permanent sentinel slot, and no pad
    # cell ever gathered on a live row (incl. the aliasing pad row).
    assert idx.dtype == torch.int32 and idx.shape == (b, n, RAY_GATHER_SLOTS)
    assert torch.equal(
        idx[:, :, 0], torch.arange(n, dtype=torch.int32).expand(b, n)
    )
    assert (idx[:, :, RAY_GATHER_SLOTS - 1] == n).all()
    for bi in range(b):
        for i in range(n):
            if not mask[bi, i]:
                continue
            row = idx[bi, i]
            gathered = [int(j) for j in row if int(j) < n]
            assert all(bool(mask[bi, j]) for j in gathered), (bi, i)
            assert len(gathered) == len(set(gathered)), "duplicate key gathered"

    # Exact live-set equality with the reference mask (pad KEYS excluded like
    # _build_ray_bias's `dead`), on live QUERY rows.
    dq = coords[:, None, :, 0] - coords[:, :, None, 0]
    dr = coords[:, None, :, 1] - coords[:, :, None, 1]
    ref = model._ray_live_mask(dq, dr, raylen) & mask[:, None, None, :]
    rec = _reconstruct_live(idx, raylen, blockers)
    for bi in range(b):
        live_rows = mask[bi]
        assert torch.equal(
            rec[bi][:, live_rows, :], ref[bi][:, live_rows, :]
        ), (board, blockers, bi)


# --- (b2) sync-free builder: bit-identical to the old bbox-grid build ---------------
#
# The current _build_ray_gather_index_impl does the same coordinate join as the
# pre-rewrite bounding-box scatter grid, but with sort+searchsorted so no
# allocation shape depends on the data (zero device->host syncs). This section
# pins BIT-IDENTICAL output against a faithful copy of that old (synchronizing)
# build across many board shapes, and proves the new build issues no CUDA sync.


def _build_ray_gather_index_old(
    coords: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Faithful copy of the pre-rewrite builder (dense bounding-box scatter grid
    over the LIVE coords' extents, with its ~8 device->host syncs). The oracle
    for the new sort/searchsorted build's exact-parity gate."""

    b, n, _ = coords.shape
    dev = coords.device
    q = coords[..., 0].long()
    r = coords[..., 1].long()
    live = mask.bool()
    big = torch.tensor(1 << 60, dtype=torch.long, device=dev)
    qmin = int(torch.where(live, q, big).amin()) if bool(live.any()) else 0
    rmin = int(torch.where(live, r, big).amin()) if bool(live.any()) else 0
    qmax = int(torch.where(live, q, -big).amax()) if bool(live.any()) else 0
    rmax = int(torch.where(live, r, -big).amax()) if bool(live.any()) else 0
    gh = qmax - qmin + 1
    gw = rmax - rmin + 1
    grid = torch.full((b, gh * gw + 1), n, dtype=torch.int32, device=dev)
    key = (q - qmin) * gw + (r - rmin)
    key_w = torch.where(live, key, torch.full_like(key, gh * gw))
    src = torch.arange(n, device=dev, dtype=torch.int32).unsqueeze(0).expand(b, n)
    grid.scatter_(1, key_w, src)
    off = slot_offset_table().to(dev)
    kq = q.unsqueeze(-1) + off[:, 0]
    kr = r.unsqueeze(-1) + off[:, 1]
    inb = (kq >= qmin) & (kq <= qmax) & (kr >= rmin) & (kr <= rmax)
    kkey = torch.where(inb, (kq - qmin) * gw + (kr - rmin), torch.zeros_like(kq))
    out = torch.where(
        inb,
        grid.gather(1, kkey.reshape(b, -1)).reshape(b, n, RAY_GATHER_SLOTS),
        torch.full((), n, dtype=torch.int32, device=dev),
    )
    out[:, :, 0] = torch.arange(n, device=dev, dtype=torch.int32).unsqueeze(0)
    out[:, :, RAY_GATHER_SLOTS - 1] = n
    return out


def _pack_batch(
    rows: list[list[tuple[int, int]]],
    *,
    npad: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(coords, mask) from per-row LIVE-coord lists (distinct within a row — the
    real support invariant). Live coords occupy each row's prefix; rows are
    padded to a common Npad with bounded garbage pad coords (mask False)."""

    b = len(rows)
    maxn = max((len(rc) for rc in rows), default=0)
    npad = max(npad or 0, maxn, 1)
    g = random.Random(seed)
    coords = torch.zeros(b, npad, 2, dtype=torch.long)
    mask = torch.zeros(b, npad, dtype=torch.bool)
    for bi, rc in enumerate(rows):
        for i, (cq, cr) in enumerate(rc):
            coords[bi, i, 0] = cq
            coords[bi, i, 1] = cr
            mask[bi, i] = True
        for i in range(len(rc), npad):  # garbage pad coords, still i16
            coords[bi, i, 0] = g.randint(-300, 300)
            coords[bi, i, 1] = g.randint(-300, 300)
    return coords, mask


def _parity_cases() -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Diverse (id, coords, mask) inputs for the exact-parity gate."""

    cases: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    rng = random.Random(20260709)

    # Full hex-disk boards, radii 1..8 (centered => coords include negatives).
    for radius in range(1, 9):
        coords, mask = _pack_batch([disk_offsets(radius)], seed=radius)
        cases.append((f"disk_r{radius}", coords, mask))

    # Random subsets of a disk's cells (still within the disk's span), with
    # assorted pad fractions (extra all-pad tail rows).
    for radius in range(2, 9):
        cells = disk_offsets(radius)
        for trial in range(3):
            k = rng.randint(1, len(cells))
            subset = rng.sample(cells, k)
            pad_extra = rng.randint(0, 6)
            coords, mask = _pack_batch(
                [subset], npad=len(subset) + pad_extra, seed=1000 * radius + trial
            )
            cases.append((f"subset_r{radius}_{trial}", coords, mask))

    # Single-cell boards (including off-origin and negative).
    for i, cell in enumerate([(0, 0), (5, -3), (-7, 2)]):
        coords, mask = _pack_batch([[cell]], npad=1 + i, seed=50 + i)
        cases.append((f"single_{i}", coords, mask))

    # Mixed board sizes in one padded batch, a fully-pad batch ROW, and a pad
    # row that ALIASES a live cell (the adversarial fixture).
    coords, mask = _padded_batch(r_big=3, r_small=1)
    cases.append(("padded_alias", coords, mask))
    coords, mask = _pack_batch(
        [disk_offsets(3), disk_offsets(1), []], npad=None, seed=77
    )
    cases.append(("mixed_with_allpad_row", coords, mask))

    # A fully-pad BATCH (no live cell anywhere) — the old build's all-pad
    # extents-fallback branch.
    coords, mask = _pack_batch([[], []], npad=8, seed=88)
    cases.append(("all_pad_batch", coords, mask))

    # Translated boards: shift a disk far from the origin (large / negative
    # coords), exercising the packed-key offset away from 0.
    for i, shift in enumerate([(120, -80), (-140, 90), (200, 200)]):
        cells = [(q + shift[0], r + shift[1]) for q, r in disk_offsets(4)]
        coords, mask = _pack_batch([cells], seed=300 + i)
        cases.append((f"translated_{i}", coords, mask))

    # A multi-row batch mixing distinct translated disks (distinct spans/offsets).
    rows = [
        disk_offsets(2),
        [(q + 40, r - 25) for q, r in disk_offsets(4)],
        [(q - 60, r + 15) for q, r in disk_offsets(1)],
    ]
    coords, mask = _pack_batch(rows, seed=99)
    cases.append(("multi_translated", coords, mask))

    return cases


def test_gather_index_parity_vs_old_reference() -> None:
    """The sync-free build is BIT-IDENTICAL to the old bounding-box-grid build
    for every valid (distinct-live-coord) board shape."""

    for name, coords, mask in _parity_cases():
        new_op = build_ray_gather_index(coords, mask)
        new_impl = _build_ray_gather_index_impl(coords, mask)
        old = _build_ray_gather_index_old(coords, mask)
        assert new_op.dtype == torch.int32 and new_op.shape == old.shape, name
        assert torch.equal(new_op, old), name
        assert torch.equal(new_impl, old), name


@needs_cuda
def test_gather_index_builder_no_cuda_sync() -> None:
    """The builder issues NO device->host synchronization: a call under
    set_sync_debug_mode('error') must not raise. Tiny shapes (radius-4, B=4)
    per the busy-GPU rules; the box may be running a live train job."""

    cells = disk_offsets(4)
    n = len(cells)
    coords = (
        torch.tensor([[list(c) for c in cells]], dtype=torch.long)
        .repeat(4, 1, 1)
    )
    mask = torch.ones(4, n, dtype=torch.bool)

    def run():
        c = coords.cuda()
        m = mask.cuda()
        # Warm up outside the guarded region (first-touch allocation/registration
        # may legitimately sync); then assert the steady-state build is sync-free.
        _build_ray_gather_index_impl(c, m)
        build_ray_gather_index(c, m)
        torch.cuda.synchronize()
        prev = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            idx_impl = _build_ray_gather_index_impl(c, m)
            idx_op = build_ray_gather_index(c, m)
        finally:
            torch.cuda.set_sync_debug_mode(prev)
        # Same result as the (synchronizing) old reference, on device.
        old = _build_ray_gather_index_old(c, m)
        return idx_impl, idx_op, old

    idx_impl, idx_op, old = _cuda_politely(run)
    assert idx_impl.shape == (4, n, RAY_GATHER_SLOTS)
    assert torch.equal(idx_impl, old)
    assert torch.equal(idx_op, old)


# --- (c) gathered op math vs the materialized reference (CPU) -----------------------


@pytest.mark.parametrize("blockers", [True, False], ids=["blockers", "geometric"])
@pytest.mark.parametrize("board", ["lone", "r3"])
def test_ray_attn_op_matches_materialized_cpu(board: str, blockers: bool) -> None:
    radius = {"lone": 0, "r3": 3}[board]
    _, n, _, coords, mask = _disk_board(radius)
    model = HexfieldNet(trunk_layout="CLA", ray_blockers=blockers).eval()
    _randomize_theta(model, seed=3, fp16_exact=True)
    d = C.CHANNELS // C.RAY_HEADS
    torch.manual_seed(13)
    q = torch.randn(1, C.RAY_HEADS, n, d).half()
    k = torch.randn(1, C.RAY_HEADS, n, d).half()
    v = torch.randn(1, C.RAY_HEADS, n, d).half()
    raylen = _rand_raylen(mask, seed=17)
    rl_arg = raylen if blockers else torch.empty(0, dtype=torch.uint8)

    idx = build_ray_gather_index(coords, mask)
    slot_bias = model._ray_bias_table(0)[slot_bias_rows()].to(torch.float16)
    seq = _seq_lens(mask)

    with torch.no_grad():
        got = ray_attn(q, k, v, idx, slot_bias, rl_arg, seq, blockers)
        # Full-softmax reference: EVERY key with the additive -3e4 on dead
        # pairs (the materialized RayAttention path, fp32 math).
        bias = model._build_ray_bias(coords, mask, raylen if blockers else None, 0)
        scores = (q.float() @ k.float().mT) / (d**0.5) + bias
        ref = torch.softmax(scores, dim=-1) @ v.float()

    assert got.dtype == torch.float16
    torch.testing.assert_close(
        got.float()[:, :, mask[0]], ref[:, :, mask[0]], atol=KERNEL_ATOL, rtol=0
    )
    # The op's CPU route IS _ray_ref; pin that equivalence so the CUDA kernel's
    # fallback stays the tested oracle.
    direct = _ray_ref(q, k, v, idx, slot_bias, rl_arg, seq, blockers)
    assert torch.equal(got, direct)


# --- (d) default-off gate -------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("HEXFIELD_EQ_TRITON_RAY") == "1",
    reason="HEXFIELD_EQ_TRITON_RAY explicitly on in this env",
)
def test_kernel_gate_default_off() -> None:
    from hexfield_eq import model as M

    assert M._TRITON_RAY is False
    assert M._ray_attn_fused is None
    assert M._ray_gather_index_fused is None


# --- CUDA section -----------------------------------------------------------------


def _cuda_politely(fn):
    """Run fn(); on OOM (the box may be running a prefit) retry once after a
    backoff, then skip with a 'pending idle window' marker."""

    try:
        return fn()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        time.sleep(20)
        try:
            return fn()
        except torch.cuda.OutOfMemoryError:
            pytest.skip("GPU OOM twice (busy) — kernel parity pending idle window")


def _board_for(board: str):
    if board == "padded":
        coords, mask = _padded_batch()
    else:
        radius = {"lone": 0, "r3": 3, "dense_r8": 8}[board]
        _, _, _, coords, mask = _disk_board(radius)
    return coords, mask


@needs_cuda
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
@pytest.mark.parametrize("blockers", [True, False], ids=["blockers", "geometric"])
@pytest.mark.parametrize("board", ["lone", "r3", "dense_r8", "padded"])
def test_ray_attn_kernel_parity_cuda(
    board: str, blockers: bool, use_v2: bool, monkeypatch
) -> None:
    """Triton kernel through the RayAttention module (fp16, the serve wiring)
    vs the fp32 materialized module path, at the 3e-3 gate. Runs BOTH kernel
    variants against the same oracle by flipping the `_USE_V2` module global.
    Tiny shapes only (batch <= 2, Npad <= 256, fp16) per the busy-GPU rules."""

    import hexfield_eq._triton_ray as TR

    monkeypatch.setattr(TR, "_USE_V2", use_v2)
    coords, mask = _board_for(board)
    b, n = mask.shape
    assert b <= 2 and n <= 256, "GPU-politeness bound"
    model = HexfieldNet(trunk_layout="CLA", ray_blockers=blockers).eval()
    _randomize_theta(model, seed=5, fp16_exact=False)
    attn = model.ray_blocks[0].attn
    raylen = _rand_raylen(mask, seed=23)
    torch.manual_seed(31)
    x16 = torch.randn(b, n, C.CHANNELS).half()

    def run():
        from hexfield_eq import model as M

        dev = "cuda"
        # fp32 materialized reference (deepcopy BEFORE any forward: the
        # dense-weight serve cache must not cross devices — serve-test gotcha).
        attn32 = copy.deepcopy(attn).float().to(dev)
        attn32.impl = "materialized"
        bias = model._build_ray_bias(
            coords, mask, raylen if blockers else None, 0
        ).to(dev)
        with torch.no_grad():
            ref = attn32(x16.float().to(dev), bias)

        attn16 = copy.deepcopy(attn).half().to(dev)
        idx = build_ray_gather_index(coords.to(dev), mask.to(dev))
        slot_bias = (
            model._ray_bias_table(0)[slot_bias_rows()].to(torch.float16).to(dev)
        )
        rl_arg = (
            raylen.to(dev)
            if blockers
            else torch.empty(0, dtype=torch.uint8, device=dev)
        )
        carrier = M._RayGatherBias(
            idx, slot_bias, rl_arg, _seq_lens(mask.to(dev)), blockers
        )
        monkeypatch.setattr(M, "_ray_attn_fused", ray_attn)
        with torch.no_grad():
            got = attn16(x16.to(dev), carrier)
        return got.float().cpu(), ref.float().cpu()

    got, ref = _cuda_politely(run)
    torch.testing.assert_close(
        got[mask], ref[mask], atol=KERNEL_ATOL, rtol=0, msg=f"{board}/{blockers}"
    )


@needs_cuda
@pytest.mark.parametrize("blockers", [True, False], ids=["blockers", "geometric"])
@pytest.mark.parametrize("board", ["r3", "padded"])
def test_ray_attn_kernel_v1_v2_equal_cuda(
    board: str, blockers: bool, monkeypatch
) -> None:
    """v1 (program-per-(batch,head)) and v2 (all-heads-per-program) run the SAME
    math over the SAME inputs, so their fp16 outputs must be (near) bit-identical.
    atol=1e-3 absorbs fp16 store rounding if the head-loop op ordering differs.
    Tiny shapes (radius <= 3 boards, B <= 2, fp16) per the busy-GPU rules. If a
    kernel is unavailable it falls back to the same reference, so this stays a
    valid equality either way."""

    import hexfield_eq._triton_ray as TR

    coords, mask = _board_for(board)
    b, n = mask.shape
    assert b <= 8 and n <= 128, "GPU-politeness bound"
    model = HexfieldNet(trunk_layout="CLA", ray_blockers=blockers).eval()
    _randomize_theta(model, seed=9, fp16_exact=False)
    d = C.CHANNELS // C.RAY_HEADS

    def run():
        dev = "cuda"
        torch.manual_seed(101)
        q = torch.randn(b, C.RAY_HEADS, n, d, device=dev).half()
        k = torch.randn(b, C.RAY_HEADS, n, d, device=dev).half()
        v = torch.randn(b, C.RAY_HEADS, n, d, device=dev).half()
        raylen = _rand_raylen(mask, seed=51).to(dev)
        rl_arg = (
            raylen if blockers
            else torch.empty(0, dtype=torch.uint8, device=dev)
        )
        idx = build_ray_gather_index(coords.to(dev), mask.to(dev))
        slot_bias = (
            model._ray_bias_table(0)[slot_bias_rows()].to(torch.float16).to(dev)
        )
        seq = _seq_lens(mask.to(dev))
        with torch.no_grad():
            monkeypatch.setattr(TR, "_USE_V2", False)
            out_v1 = ray_attn(q, k, v, idx, slot_bias, rl_arg, seq, blockers)
            monkeypatch.setattr(TR, "_USE_V2", True)
            out_v2 = ray_attn(q, k, v, idx, slot_bias, rl_arg, seq, blockers)
        return out_v1.float().cpu(), out_v2.float().cpu()

    out_v1, out_v2 = _cuda_politely(run)
    assert out_v1.dtype == torch.float32 and out_v2.shape == out_v1.shape
    torch.testing.assert_close(
        out_v2, out_v1, atol=1e-3, rtol=0, msg=f"{board}/{blockers}"
    )


@needs_cuda
@pytest.mark.parametrize("mode", ["half", "autocast"])
def test_full_net_serve_wiring_cuda(mode: str, monkeypatch) -> None:
    """trunk() builds the carrier once and routes the L block through the op
    (spy-counted) under BOTH serve fp16 modes — serve-half (fp16 stream) and
    fp16 autocast (fp32 LN'd stream, fp16 q/k/v) — matching the unrouted
    forward on every head."""

    from hexfield_eq import model as M

    _, n, nbr, coords, mask = _disk_board(3)
    model = HexfieldNet(trunk_layout="CLA", ray_blockers=True).eval()
    _randomize_theta(model, seed=7, fp16_exact=False)
    raylen = _rand_raylen(mask, seed=41)
    torch.manual_seed(43)
    feats = torch.randn(1, n, C.NUM_FEATURES)

    def run():
        dev = "cuda"
        net = copy.deepcopy(model).to(dev)
        f = feats.to(dev)
        if mode == "half":
            net = net.half()
            f = f.half()
        args = (f, nbr.to(dev), mask.to(dev), coords.to(dev))
        rl = raylen.to(dev)
        autocast_on = mode == "autocast"

        def forward():
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=autocast_on
            ):
                return net(*args, raylen=rl)

        ref = forward()

        calls = {"n": 0}

        def spy(*a, **kw):
            calls["n"] += 1
            return ray_attn(*a, **kw)

        monkeypatch.setattr(M, "_ray_attn_fused", spy)
        monkeypatch.setattr(M, "_ray_gather_index_fused", build_ray_gather_index)
        monkeypatch.setattr(M, "_ray_slot_bias_rows", slot_bias_rows)
        got = forward()
        return calls["n"], got, ref

    n_calls, got, ref = _cuda_politely(run)
    assert n_calls == 1, "one L block => one kernel call"
    for key in got:
        assert torch.isfinite(got[key]).all(), key
        torch.testing.assert_close(
            got[key].float(), ref[key].float(), atol=5e-3, rtol=0, msg=key
        )
