"""hexfield_eq Rust/Python featurizer parity + D6 equivariance (Phase 1 gate).

docs/PLAN_D6_EQUIVARIANT_REWRITE.md Phase 1. The graded per-axis window planes
(15 -> 25) are computed twice — a Rust featurizer (serve: ``features.rs``;
train: ``replay_expand.rs`` over a ``WindowStore::from_placements`` store) and a
Python oracle (``features.build_features``). This pins them against each other:

  1. SERVE parity — ``_rust.featurize_states`` vs the Python featurizer on
     sampled engine states. Graded/recency planes ride a <= 1e-6 tolerant path;
     the binary/scalar kept planes are exact.
  2. TRAIN parity across ALL 12 D6 — for a synthetic shard window, expand every
     row under each symmetry with both the serial (Python) and rust backends and
     assert the graded feature planes agree (<= 1e-6), with the support graph and
     policy targets exact.
  3. D6 AXIS PERMUTATION — an explicit assertion that a board symmetry g permutes
     the axis-indexed planes: ``own_line[Q]@x -> own_line[sigma_g(Q)]@(g.x)`` (and
     the same for opp_line / own_live / opp_live), with the scalar planes
     invariant. This is the equivariance guarantee the augmentation-free expand
     relies on.

Runs in the hexgt-build venv via PYTHONPATH=packages/hexfield_eq/python (plus the
opponent/testkit packages). CPU-only.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield_eq import constants as C
from hexfield_eq.engine_facts import facts_from_engine
from hexfield_eq.expand_backends import expand_rows
from hexfield_eq.features import build_position
from hexfield_eq.geometry import apply_d6, pack_action_id, unpack_action_id
from hexfield_eq.samples import STV_HORIZONS, HexfieldSampleData
from hexfield_eq.shards import write_compact_shard
from hexfield_eq.window import concat_packed, load_packed_shard

try:
    from hexfield_eq import _rust
except ImportError:  # pragma: no cover
    _rust = None

needs_rust = pytest.mark.skipif(
    _rust is None, reason="hexfield_eq._rust not built (see the Phase-1 build gate)"
)

# The axis-indexed planes as (base, name) groups, each 3 contiguous slots
# ordered by axis [Q, R, QR]: 4 quantities under HEXFIELD_EQ_FEATURE_VERSION=1,
# 10 under version 2 (the liveK planes of SPEC_RAYTAP_CONV.md §1.3). Derived
# from the constants so this suite pins whichever plane map the env selected —
# tests/test_hexfield_eq_feature_v2.py re-runs it in a version-2 child. The
# scalar planes (everything else) are D6-invariant.
_AXIS_QUANTITY_NAMES = (
    "own_line",
    "opp_line",
    "own_live",
    "opp_live",
    "own_live3",
    "opp_live3",
    "own_live4",
    "opp_live4",
    "own_live5",
    "opp_live5",
)
_AXIS_GROUPS = tuple(
    (C.F_OWN_LINE_Q + 3 * q, _AXIS_QUANTITY_NAMES[q])
    for q in range(C.N_AXIS_QUANTITIES)
)
_AXIS_PLANES = tuple(base + a for base, _ in _AXIS_GROUPS for a in range(3))
_SCALAR_PLANES = tuple(p for p in range(C.NUM_FEATURES) if p not in _AXIS_PLANES)

# A crafted game that drives a length-5 P0 line on the Q axis, so the corpus is
# guaranteed to exercise the strong end of the line/fork planes (a purely random
# corpus rarely forms 4+-in-window). Player order after the opening is
# P0, P1,P1, P0,P0, P1,P1, P0,P0, P1 — so indices 0,3,4,7,8 are P0.
_LINE_GAME = [
    (0, 0),  # P0
    (0, 3),  # P1
    (0, 4),  # P1
    (1, 0),  # P0
    (2, 0),  # P0
    (1, 3),  # P1
    (1, 4),  # P1
    (3, 0),  # P0
    (4, 0),  # P0
    (2, 3),  # P1  -> row recorded here sees the 5-in-line P0 wall (opp_line=1.0)
]


# --- state / row generation ---------------------------------------------------


def _random_state(seed: int, plies: int):
    """A non-terminal state from a seeded uniform-random playout."""
    state = api.new_game()
    rng = random.Random(seed)
    for _ in range(plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            break
    return state


def _decision_states():
    """A spread of non-terminal decision states + the crafted line state."""
    states = []
    for seed in range(6):
        for plies in (1, 3, 7, 14, 24, 36):
            st = _random_state(seed * 1000 + plies, plies)
            if api.terminal(st) is None:
                states.append(st)
    # crafted 5-line state: play the whole line game (apply_action mutates the
    # state handle in place), keep the (non-terminal) end.
    st = api.new_game()
    for q, r in _LINE_GAME:
        if api.terminal(st) is not None:
            break
        api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
    if api.terminal(st) is None:
        states.append(st)
    return states


def _rows_from_moves(moves, game_id: str) -> list[HexfieldSampleData]:
    """Play ``moves`` in order, recording one decision row before each move.

    The row's policy target is the played move (weight 1.0), so every row expands
    to a valid legal-set projection. Value/moves_left are left masked-out.

    Rows also carry synthetic opp/gumbel/cell-Q targets so the backend parity
    comparisons exercise real projection + renormalization (f32 accumulation,
    f64 gumbel renormalizer, coverage tracking) rather than all-zero defaults:
    a few legal ids with non-terminating-binary weights, plus (when the board
    has stones) one OFF-legal id — an occupied cell — whose mass must be
    dropped identically by both backends (opp_coverage < 1, gumbel renorm over
    the kept support)."""
    state = api.new_game()
    rows: list[HexfieldSampleData] = []
    for q, r in moves:
        if api.terminal(state) is not None:
            break
        action_id = pack_action_id(q, r)
        legal = sorted(int(a) for a in api.legal_action_ids(state))
        if action_id not in set(legal):
            raise AssertionError(f"crafted move {(q, r)} is not legal")
        facts = facts_from_engine(api.to_python_state(state))
        # Deterministic per-row target support: a small legal slice, rotated by
        # the ply so different rows hit different cells.
        k = min(3, len(legal))
        base = len(rows) % max(1, len(legal) - k + 1)
        tgt_ids = legal[base : base + k]
        weights = (0.6, 0.3, 0.1)[:k]
        # An off-legal id: any occupied cell (absent from the legal set).
        off_legal = (
            (pack_action_id(facts.records[0][0], facts.records[0][1]),)
            if facts.records
            else ()
        )
        gumbel = tuple(zip(tgt_ids, weights)) + tuple((a, 0.2) for a in off_legal)
        opp = tuple(zip(tgt_ids, (0.7, 0.2, 0.1)[:k])) + tuple(
            (a, 0.3) for a in off_legal
        )
        # The shard stores cell-Q aligned to the recorded policy actions, so
        # give the policy a small tail over tgt_ids (all legal) and key the Q
        # targets to those same actions.
        policy = ((int(action_id), 1.0),) + tuple(
            (a, 0.05) for a in tgt_ids if a != int(action_id)
        )
        qpol = ((int(action_id), 0.5),) + tuple(
            zip((a for a in tgt_ids if a != int(action_id)), (-0.25, 0.125, 0.0625))
        )
        rows.append(
            HexfieldSampleData(
                game_id=game_id,
                turn_index=len(facts.records),
                current_player=facts.current_player,
                phase=facts.phase,
                records=facts.records,
                first_stone=facts.first_stone,
                policy=policy,
                opp_policy=opp,
                q_policy=qpol,
                gumbel_policy=gumbel,
                value=0.0,
                moves_left=-1.0,
            )
        )
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    return rows


def _random_moves(seed: int, plies: int) -> list[tuple[int, int]]:
    state = api.new_game()
    rng = random.Random(seed)
    moves: list[tuple[int, int]] = []
    for _ in range(plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        aid = rng.choice(ids)
        q, r = unpack_action_id(aid)
        moves.append((q, r))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            break
    return moves


def _build_window(tmpdir: Path):
    """Write a synthetic shard from a few games and load it as a PackedWindow."""
    samples: list[HexfieldSampleData] = []
    samples += _rows_from_moves(_LINE_GAME, "line")
    for seed in range(5):
        moves = _random_moves(3000 + seed, 34)
        if moves:
            samples += _rows_from_moves(moves, f"rand{seed}")
    assert samples, "no rows generated"
    shard = tmpdir / "epoch_000000" / "game_0.npz"
    write_compact_shard(shard, samples, short_term_value_horizons=STV_HORIZONS)
    return concat_packed([load_packed_shard(shard)])


# --- axis permutation helpers -------------------------------------------------

_AXIS_VECS = ((1, 0), (0, 1), (1, -1))  # Q, R, QR


def _axis_index_of(dq: int, dr: int) -> int:
    """Canonical axis (0=Q,1=R,2=QR) collinear with the (undirected) offset."""
    for idx, (aq, ar) in enumerate(_AXIS_VECS):
        if aq * dr - ar * dq == 0:  # zero cross product => collinear
            return idx
    raise AssertionError(f"offset {(dq, dr)} is not on a win axis")


def _axis_perm(g: int) -> list[int]:
    """sigma_g: the axis a's value lands on axis ``perm[a]`` after applying g."""
    perm = [0, 0, 0]
    for a, (dq, dr) in enumerate(_AXIS_VECS):
        tq, tr = apply_d6(g, dq, dr)
        perm[a] = _axis_index_of(tq, tr)
    return perm


# --- tests --------------------------------------------------------------------


@needs_rust
def test_capabilities_report_plane_count() -> None:
    caps = _rust.capabilities()
    assert caps["model_family"] == "hexfield_eq"
    assert caps["num_features"] == C.NUM_FEATURES == (
        25 if C.FEATURE_VERSION == 1 else 46
    )


@needs_rust
def test_serve_featurizer_parity() -> None:
    states = _decision_states()
    assert len(states) >= 20
    payloads = _rust.featurize_states(states)
    assert len(payloads) == len(states)

    exact_planes = [p for p in range(C.NUM_FEATURES) if p not in (C.F_OWN_RECENCY, C.F_OPP_RECENCY)]
    exact_planes = [p for p in exact_planes if p not in _AXIS_PLANES]
    if C.FEATURE_VERSION == 2:
        # The three float-derived global scalars (spec §1.4) ride the tolerant
        # path with the graded planes; the int-derived planes stay bit-exact.
        exact_planes = [
            p for p in exact_planes
            if p not in (C.F_PLY, C.F_DIST_CENTROID, C.F_SPREAD)
        ]

    saw_line = 0.0
    for state, payload in zip(states, payloads):
        facts = facts_from_engine(api.to_python_state(state))
        sup, feats = build_position(facts)
        n = sup.num_nodes
        assert payload["num_nodes"] == n

        rust = np.frombuffer(payload["feats"], dtype=np.float32).reshape(n, C.NUM_FEATURES)
        diff = np.abs(rust - feats)
        assert diff.max() <= 1e-6, (
            f"feature mismatch: max diff {diff.max()} at "
            f"{np.unravel_index(int(diff.argmax()), diff.shape)}"
        )
        # The binary/scalar kept planes (not recency, not the graded axis planes)
        # are bit-exact; the graded + recency planes ride the tolerant path above.
        assert np.array_equal(rust[:, exact_planes], feats[:, exact_planes])
        saw_line = max(saw_line, float(feats[:, C.F_OWN_LINE_Q : C.F_OPP_LINE_Q + 3].max()))

    # The corpus really exercises the strong end of the line planes (the crafted
    # 5-line state contributes a 1.0), so the parity above is non-trivial.
    assert saw_line >= 0.6, f"line planes never exceeded {saw_line}; corpus too weak"


@needs_rust
def test_expand_all_12_d6_rust_eq_serial(monkeypatch) -> None:
    # Force the serial raylen oracle on (it is gated off under C/A layouts,
    # spec D-S29, while the rust kernel always emits raylen) so the raylen
    # comparison below is apples-to-apples on every layout.
    from hexfield_eq import samples as samples_mod

    monkeypatch.setattr(samples_mod, "_EXPAND_RAYLEN", True)
    with tempfile.TemporaryDirectory() as td:
        window = _build_window(Path(td))
        n = window.n
        assert n > 0

        graded_seen = 0.0
        for sym in range(12):
            d6 = np.full(n, sym, dtype=np.int64)
            rows_s, valid_s = expand_rows(window, None, d6, backend="serial")
            rows_r, valid_r = expand_rows(window, None, d6, backend="rust")
            assert np.array_equal(valid_s, valid_r), f"valid mask differs at sym {sym}"
            for k, (a, b) in enumerate(zip(rows_s, rows_r)):
                assert (a is None) == (b is None), f"None mismatch row {k} sym {sym}"
                if a is None:
                    continue
                # Support graph + policy targets are exact ints/accumulations.
                assert a.support.num_nodes == b.support.num_nodes, f"N row {k} sym {sym}"
                assert a.support.legal_count == b.support.legal_count
                assert np.array_equal(a.support.coords, b.support.coords), f"coords row {k} sym {sym}"
                assert np.array_equal(a.support.nbr, b.support.nbr), f"nbr row {k} sym {sym}"
                assert np.array_equal(a.support.dist, b.support.dist), f"dist row {k} sym {sym}"
                assert np.array_equal(a.policy, b.policy), f"policy row {k} sym {sym}"
                # Auxiliary training targets: both backends promise bit-parity
                # (f32 accumulation in sample order; the gumbel renormalizer
                # accumulates f64 and divides the f32 array by the one-rounding
                # f32 scalar on both sides; cell_q is a scalar assign; raylen is
                # integer; opp_coverage is pure f64 arithmetic) — exact equality.
                assert np.array_equal(a.opp_policy, b.opp_policy), f"opp_policy row {k} sym {sym}"
                assert a.opp_coverage == b.opp_coverage, f"opp_coverage row {k} sym {sym}"
                assert np.array_equal(a.cell_q, b.cell_q), f"cell_q row {k} sym {sym}"
                assert np.array_equal(a.cell_q_mask, b.cell_q_mask), f"cell_q_mask row {k} sym {sym}"
                assert np.array_equal(a.gumbel_policy, b.gumbel_policy), f"gumbel_policy row {k} sym {sym}"
                assert a.gumbel_policy_valid == b.gumbel_policy_valid, f"gumbel_policy_valid row {k} sym {sym}"
                assert np.array_equal(a.raylen, b.raylen), f"raylen row {k} sym {sym}"
                # Graded feature planes: serial (f64->f32) vs rust (f32) can differ
                # by <= 1 ULP on the /6 live planes, so compare on the tolerant path.
                d = np.abs(a.feats - b.feats)
                assert d.max() <= 1e-6, (
                    f"feature mismatch row {k} sym {sym}: max {d.max()} at "
                    f"{np.unravel_index(int(d.argmax()), d.shape)}"
                )
                graded_seen = max(graded_seen, float(a.feats[:, C.F_OWN_LINE_Q :].max()))

        assert graded_seen >= 0.6, f"graded planes never exceeded {graded_seen}"


@needs_rust
def test_d6_axis_permutation() -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _build_window(Path(td))
        n = window.n

        # Identity expansion is the reference; each g maps own_line[a]@x ->
        # own_line[sigma_g(a)]@(g.x). Check on the serial oracle AND the rust
        # kernel so both featurizers' axis indexing is pinned.
        for backend in ("serial", "rust"):
            base_rows, base_valid = expand_rows(
                window, None, np.zeros(n, dtype=np.int64), backend=backend
            )
            checks = 0
            for g in range(12):
                perm = _axis_perm(g)
                d6 = np.full(n, g, dtype=np.int64)
                g_rows, g_valid = expand_rows(window, None, d6, backend=backend)
                for i in range(n):
                    if not base_valid[i] or not g_valid[i]:
                        continue
                    r0, rg = base_rows[i], g_rows[i]
                    # index the g-support by transformed coord.
                    g_index = {
                        (int(q), int(r)): row for row, (q, r) in enumerate(rg.support.coords)
                    }
                    for j, (xq, xr) in enumerate(r0.support.coords):
                        gq, gr = apply_d6(g, int(xq), int(xr))
                        jg = g_index.get((gq, gr))
                        assert jg is not None, f"g.x {(gq, gr)} missing in g-support (sym {g})"
                        # scalar planes are D6-invariant.
                        for p in _SCALAR_PLANES:
                            assert abs(float(r0.feats[j, p]) - float(rg.feats[jg, p])) <= 1e-6, (
                                f"scalar plane {p} not invariant (sym {g}, {backend})"
                            )
                        # axis planes permute by sigma_g: value at axis a in the
                        # identity row sits at axis perm[a] in the g row.
                        for gbase, name in _AXIS_GROUPS:
                            for a in range(3):
                                lhs = float(r0.feats[j, gbase + a])
                                rhs = float(rg.feats[jg, gbase + perm[a]])
                                assert abs(lhs - rhs) <= 1e-6, (
                                    f"{name} axis {a}->{perm[a]} not equivariant under "
                                    f"sym {g} ({backend}): {lhs} != {rhs}"
                                )
                        checks += 1
            assert checks > 0, f"no axis-permutation checks ran ({backend})"
