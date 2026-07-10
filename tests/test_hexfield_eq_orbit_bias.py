"""Phase-2 orbit-tied relative-position bias table (hexfield_eq).

Gate for docs/PLAN_D6_EQUIVARIANT_REWRITE.md Phase 2. Each attention block's
per-row (BIAS_ROWS=237, heads) bias table is replaced by a free
(BIAS_FREE_ROWS=45, heads) table tied over the D6 orbits of the offset lattice;
the expanded 237-row table every bias builder consumes is
``free[orbit_of_row]``. This suite asserts:

  (a) the expanded table equals a per-row table *restricted to be
      orbit-constant* — i.e. rows in one D6 orbit share their bias, there are
      exactly 45 degrees of freedom, and the orbit partition is the genuine D6
      orbit partition (25 disk orbits: 1 center + 12 size-6 + 12 size-12; plus
      16 ring + 1 far + 3 token singletons);
  (b) an equivariance micro-test — with a from-scratch (random) tied bias the
      attention rel-pos bias (the additive term ``build_attn_bias`` contributes
      to the scores) is *exactly* D6-invariant on a probe board, on both the
      no-grad and grad code paths; an untied random table is not (control);
  (c) training a few steps updates all 45 free params of every block — the
      gradient reaches every tied row through the orbit-LUT index-select.

Also checks the param-grouping fix: the plugin's AdamW places every
``bias_free_tables*`` param in the no-decay group (the Phase-2 rename moved them
off the old ``bias_table`` substring, so a stale predicate would silently apply
weight decay).

Runs in the hexgt-build venv via PYTHONPATH=packages/hexfield_eq/python (plus the
shared testkit / opponent packages). CPU-only; the Triton/flex serve kernels are
env-gated off by default so no CUDA is required.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from hexfield_eq import constants as C
from hexfield_eq.geometry import (
    BIAS_FREE_ROWS,
    apply_d6,
    bias_orbit_of_row,
    disk_offsets,
    hex_dist,
    on_win_axis,
    rel_bias_index,
)
from hexfield_eq.model import HexfieldNet

# Phase-2 passthrough gate. Phase 3b makes the equivariant (GROUP_ORDER=12) tie
# the DEFAULT build, where the per-block bias is the jointly (row, head)-tied
# `bias_theta` (docs/DERIVATION §5.3) — it SUPERSEDES this Phase-2 per-head-free
# (45, heads) `bias_free_tables`/`orbit_of_row` table. Run this suite under
# HEXFIELD_EQ_GROUP_ORDER=1 (passthrough); it self-skips under the equivariant
# build.
pytestmark = pytest.mark.skipif(
    C.GROUP_ORDER != 1,
    reason="Phase-2 passthrough bias gate; run with HEXFIELD_EQ_GROUP_ORDER=1",
)

# NOTE: hexfield_eq.plugin is imported lazily inside the predicate test only — it
# pulls in the whole hexo_train pipeline (and hexo_utils), which the gate tests
# (a)/(b)/(c) do not need. Keeping it out of the module top-level means those
# tests collect and run under the minimal PYTHONPATH.

HEADS = C.ATTENTION_HEADS


def _orbit_tensor() -> torch.Tensor:
    return torch.as_tensor(bias_orbit_of_row(), dtype=torch.long)


def _restrict_to_orbit_constant(
    table: torch.Tensor, orbit: torch.Tensor, n_classes: int
) -> torch.Tensor:
    """Project a per-row (BIAS_ROWS, heads) table onto the orbit-constant
    subspace: replace every row by the mean of its D6-orbit class, broadcast
    back to 237 rows. A table is orbit-constant iff it is a fixed point."""

    h = table.shape[1]
    sums = torch.zeros(n_classes, h, dtype=table.dtype).index_add_(0, orbit, table)
    counts = torch.zeros(n_classes, dtype=table.dtype).index_add_(
        0, orbit, torch.ones(orbit.shape[0], dtype=table.dtype)
    )
    means = sums / counts[:, None]
    return means[orbit]


# --- (a) orbit partition + orbit-constant restriction --------------------------


def test_orbit_lut_is_the_d6_partition() -> None:
    orbit = bias_orbit_of_row()
    assert len(orbit) == C.BIAS_ROWS == 237
    # Exactly 45 free classes, contiguously numbered 0..44, all used.
    assert set(orbit) == set(range(45))
    assert BIAS_FREE_ROWS == 45 == max(orbit) + 1

    # Disk rows (0..216) must group EXACTLY as the true D6 orbits, recomputed
    # here independently of the LUT's construction.
    offsets = disk_offsets(C.BIAS_DISK_RADIUS)
    assert len(offsets) == C.BIAS_EXACT_ROWS == 217
    true_orbit_of_offset: dict[tuple[int, int], frozenset] = {}
    for dq, dr in offsets:
        true_orbit_of_offset[(dq, dr)] = frozenset(
            apply_d6(g, dq, dr) for g in range(12)
        )
    for r1, o1 in enumerate(offsets):
        for r2, o2 in enumerate(offsets):
            same_lut = orbit[r1] == orbit[r2]
            same_true = true_orbit_of_offset[o1] == true_orbit_of_offset[o2]
            assert same_lut == same_true, (o1, o2)

    # Orbit-size histogram of the disk classes: 1 center + 12 size-6 + 12 size-12.
    disk_classes = orbit[: C.BIAS_EXACT_ROWS]
    disk_sizes = Counter(Counter(disk_classes).values())
    assert disk_sizes == Counter({6: 12, 12: 12, 1: 1}), disk_sizes
    # The 20 non-disk rows (16 ring + 1 far + 3 token) are each a singleton class.
    tail = orbit[C.BIAS_EXACT_ROWS :]
    assert len(tail) == 20
    assert len(set(tail)) == 20
    # ...and disjoint from the 25 disk classes.
    assert set(tail).isdisjoint(set(disk_classes))


def test_expanded_table_is_orbit_constant_restriction() -> None:
    orbit = _orbit_tensor()
    torch.manual_seed(0)
    free = torch.randn(BIAS_FREE_ROWS, HEADS)
    expanded = free[orbit]  # the model's _block_bias_table construction

    assert expanded.shape == (C.BIAS_ROWS, HEADS)
    # (a1) rows sharing a D6 orbit are byte-identical -> exactly 45 distinct rows.
    for cls in range(BIAS_FREE_ROWS):
        rows = expanded[orbit == cls]
        assert torch.equal(rows, rows[:1].expand_as(rows)), cls
    assert torch.unique(expanded, dim=0).shape[0] == BIAS_FREE_ROWS

    # (a2) the expanded table equals a per-row table restricted to be
    # orbit-constant: it is a fixed point of the orbit-mean projection (up to the
    # fp rounding the mean's sum/divide reintroduces; exact orbit-constancy is
    # asserted bit-for-bit in a1 above).
    projected = _restrict_to_orbit_constant(expanded, orbit, BIAS_FREE_ROWS)
    assert torch.allclose(projected, expanded, atol=1e-6, rtol=1e-6)

    # Control: an arbitrary per-row table is NOT orbit-constant; the projection
    # changes it (so the restriction is a real constraint, not a no-op).
    full = torch.randn(C.BIAS_ROWS, HEADS)
    proj_full = _restrict_to_orbit_constant(full, orbit, BIAS_FREE_ROWS)
    assert not torch.allclose(proj_full, full)
    # And the projection of any table IS orbit-constant (idempotent).
    assert torch.allclose(
        _restrict_to_orbit_constant(proj_full, orbit, BIAS_FREE_ROWS), proj_full
    )


# --- (b) equivariance micro-test -----------------------------------------------


def test_tied_bias_is_exactly_d6_invariant_pure_geometry() -> None:
    """The rel-pos bias table[rel_bias_index(offset)] is exactly equal for an
    offset and its image under every g in D6 — covering disk, ring, far, and
    (trivially) token rows. This is the invariance the orbit tie buys."""

    orbit = _orbit_tensor()
    torch.manual_seed(1)
    free = torch.randn(BIAS_FREE_ROWS, HEADS)
    table = free[orbit]

    saw_ring = saw_far = False
    for dq in range(-18, 19):
        for dr in range(-18, 19):
            base = table[rel_bias_index(dq, dr)]
            d = hex_dist(dq, dr)
            if 9 <= d <= 16:
                saw_ring = True
            elif d >= 17:
                saw_far = True
            for g in range(12):
                gq, gr = apply_d6(g, dq, dr)
                assert torch.equal(table[rel_bias_index(gq, gr)], base), (dq, dr, g)
    assert saw_ring and saw_far  # the sweep actually exercised ring + far rows

    # Control: without the tie (distinct per-row values) invariance breaks.
    torch.manual_seed(2)
    untied = torch.randn(C.BIAS_ROWS, HEADS)
    mismatch = False
    for dq in range(-8, 9):
        for dr in range(-8, 9):
            base = untied[rel_bias_index(dq, dr)]
            for g in range(1, 12):
                gq, gr = apply_d6(g, dq, dr)
                if not torch.equal(untied[rel_bias_index(gq, gr)], base):
                    mismatch = True
    assert mismatch, "untied random table should not be D6-invariant"


def _probe_coords(cells: list[tuple[int, int]], g: int | None = None) -> torch.Tensor:
    if g is not None:
        cells = [apply_d6(g, q, r) for (q, r) in cells]
    return torch.tensor([[list(c) for c in cells]], dtype=torch.long)


def test_build_attn_bias_is_exactly_d6_invariant_on_probe_board() -> None:
    """End-to-end through the model's bias builder: for a from-scratch tied bias,
    ``build_attn_bias`` is element-wise identical for a probe board and each of
    its 12 D6 images (Q/K equivariance is Phase 3; here the rel-pos bias term —
    the D6-controlled part of the attention scores — must be exactly invariant).
    Checked on both the no-grad (fp16) and grad (fp32 _BiasGather) paths."""

    torch.manual_seed(3)
    model = HexfieldNet()
    for p in model.bias_free_tables:
        with torch.no_grad():
            p.copy_(torch.randn_like(p))  # a genuinely nonzero tied table

    # Compact probe board: a radius-3 hex disk (all pairwise offsets have
    # hex_dist <= 6, well inside the exact disk, so no ring/far clamping) with
    # every cell live.
    cells = disk_offsets(3)
    n = len(cells)
    mask = torch.ones(1, n, dtype=torch.bool)
    coords0 = _probe_coords(cells)

    # No-grad path (model.eval + no_grad hits the fp16 head-first branch).
    model.eval()
    for block in range(len(model.attn_blocks)):
        with torch.no_grad():
            pair0, kp0 = model._build_pair(coords0, mask)
            bias0 = model.build_attn_bias(pair0, kp0, block)
            for g in range(12):
                pg, kpg = model._build_pair(_probe_coords(cells, g), mask)
                assert torch.equal(bias0, model.build_attn_bias(pg, kpg, block)), (
                    block,
                    g,
                )

    # Grad path (_BiasGather + fp32 permute/contiguous branch).
    for block in range(len(model.attn_blocks)):
        with torch.enable_grad():
            pair0, kp0 = model._build_pair(coords0, mask)
            bias0 = model.build_attn_bias(pair0, kp0, block)
            for g in range(12):
                pg, kpg = model._build_pair(_probe_coords(cells, g), mask)
                assert torch.equal(bias0, model.build_attn_bias(pg, kpg, block)), (
                    block,
                    g,
                )


# --- (c) gradient reaches all 45 free params -----------------------------------


def _coverage_board() -> torch.Tensor:
    """A board whose pairwise offsets (all measured from the origin cell, which
    is present) realize every one of the 45 free classes: the full radius-8 disk
    (origin -> every disk offset gives all 25 disk orbits), on-axis ring reps
    (dist 9..16), off-axis ring reps (dist 9..16), and a far cell (dist >= 17).
    The 3 token rows are always present in the attention sequence."""

    cells = list(disk_offsets(8))  # 217 offsets, includes (0, 0)
    cells += [(k, 0) for k in range(9, 17)]  # on-axis ring, dist 9..16
    cells += [(7, k) for k in range(2, 10)]  # off-axis ring, dist 9..16
    cells += [(20, 0)]  # far, dist >= 17
    return torch.tensor([[list(c) for c in cells]], dtype=torch.long)


def test_training_steps_reach_all_45_free_params() -> None:
    torch.manual_seed(4)
    model = HexfieldNet()
    coords = _coverage_board()
    n = coords.shape[1]

    # Sanity: the board + token rows really do cover all 45 classes.
    rows = {rel_bias_index(int(q), int(r)) for (q, r) in coords[0].tolist()}
    covered = {int(model.orbit_of_row[r]) for r in rows}
    covered |= {int(model.orbit_of_row[t]) for t in (234, 235, 236)}  # token rows
    assert covered == set(range(45)), sorted(set(range(45)) - covered)

    feats = torch.randn(1, n, C.NUM_FEATURES)
    # No neighbours (index n == the padded zero row): isolates the bias params;
    # attention is dense over the whole sequence regardless.
    nbr = torch.full((1, n, 6), n, dtype=torch.long)
    mask = torch.ones(1, n, dtype=torch.bool)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in model.bias_free_tables]
    reached = [torch.zeros(BIAS_FREE_ROWS, dtype=torch.bool) for _ in before]

    for _ in range(3):
        opt.zero_grad()
        out = model(feats, nbr, mask, coords)
        loss = sum(v.float().pow(2).mean() for v in out.values())
        loss.backward()
        for i, p in enumerate(model.bias_free_tables):
            assert p.grad is not None
            reached[i] |= (p.grad.abs().sum(dim=1) > 0).cpu()
        opt.step()

    for i, p in enumerate(model.bias_free_tables):
        assert bool(reached[i].all()), (
            f"block {i}: only {int(reached[i].sum())}/45 free rows got gradient"
        )
        # A few steps actually moved every free row off its zero init.
        assert bool((p.detach() != before[i]).all()), f"block {i}: some free rows unchanged"


# --- param-grouping fix (no-decay predicate) -----------------------------------


def test_bias_free_tables_are_no_decay_in_plugin_optimizer() -> None:
    try:
        from hexfield_eq.plugin import get_plugin
    except ImportError as exc:  # hexo_train / hexo_utils not on the path
        pytest.skip(f"plugin import chain unavailable: {exc}")

    plugin = get_plugin()
    model = plugin.build_model({}, {})
    overrides = plugin.training_component_overrides(
        defaults=None, config={}, shared=None, model=model
    )
    opt = overrides.optimizer

    free_params = {
        id(p) for n, p in model.named_parameters() if "bias_free_tables" in n
    }
    assert free_params, "no bias_free_tables params found"

    decayed = {
        id(p)
        for grp in opt.param_groups
        if grp["weight_decay"] != 0.0
        for p in grp["params"]
    }
    assert free_params.isdisjoint(decayed), "a bias_free_tables param got weight decay"
    # And they are actually present in some no-decay group (not dropped).
    no_decay = {
        id(p)
        for grp in opt.param_groups
        if grp["weight_decay"] == 0.0
        for p in grp["params"]
    }
    assert free_params <= no_decay
