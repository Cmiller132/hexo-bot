"""Tests for support construction against the engine.

Covered properties:
- Closed-form legality (empty ∧ dist <= LEGAL_RADIUS; opening => {(0,0)})
  equals the engine legal set on random non-terminal states.
- Halo is exactly the HALO_DIST shell; core = stones ∪ legal.
- BFS distances equal brute-force min hex distance.
- Ply 0 support has 7 nodes, 1 legal.
- Node order: [legal | stones | halo], each ascending by packed action id.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from hexfield_testkit import api, sample_decision_states
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.constants import HALO_DIST, LEGAL_RADIUS
from hexfield.geometry import hex_dist, pack_action_id, unpack_action_id
from hexfield.support import (
    SupportContractError,
    assert_decision_support,
    build_support,
)


def _stones(state) -> list[tuple[int, int]]:
    mirror = api.to_python_state(state)
    return [(c.q, c.r) for c, _player in mirror.board.stones]


def _brute_dist(cell: tuple[int, int], stones: list[tuple[int, int]]) -> int:
    return min(hex_dist(cell[0] - q, cell[1] - r) for q, r in stones)


def test_support_matches_engine_on_random_states() -> None:
    states = sample_decision_states(range(8), (1, 2, 5, 9, 14, 23, 36, 51))
    assert len(states) >= 30
    for state in states:
        stones = _stones(state)
        sup = build_support(stones)

        # Closed-form legality equals the engine legal set, in packed-id order.
        engine_ids = sorted(api.legal_action_ids(state))
        sup_ids = [pack_action_id(q, r) for q, r in sup.legal_coords().tolist()]
        assert sup_ids == engine_ids

        # Segment order: each segment ascends by packed id.
        legal_rng, stone_rng, halo_rng = sup.segments()
        for seg in (legal_rng, stone_rng, halo_rng):
            seg_ids = [
                pack_action_id(*sup.coords[i].tolist()) for i in seg
            ]
            assert seg_ids == sorted(seg_ids)

        # Distances: BFS equals brute force; halo is exactly the HALO_DIST shell.
        coords = [tuple(c) for c in sup.coords.tolist()]
        for row, cell in enumerate(coords):
            d = _brute_dist(cell, stones)
            assert sup.dist[row] == d
            in_halo = row in halo_rng
            assert in_halo == (d == HALO_DIST)
            if not in_halo:
                assert d <= LEGAL_RADIUS

        # Core = stones ∪ legal; halo cells are adjacent to core.
        core = set(coords[: sup.legal_count + sup.stone_count])
        assert core == set(coords[: sup.legal_count]) | set(stones)
        for row in halo_rng:
            q, r = coords[row]
            assert any(
                (q + dq, r + dr) in core
                for dq, dr in ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
            )

        # Neighbour table: row-local indices along DIRECTIONS, -1 when absent.
        for row, (q, r) in enumerate(coords):
            for k, (dq, dr) in enumerate(
                ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
            ):
                expected = sup.index.get((q + dq, r + dr), -1)
                assert sup.nbr[row, k] == expected


def test_ply0_support() -> None:
    sup = build_support([])
    assert sup.num_nodes == 7
    assert sup.legal_count == 1
    assert sup.stone_count == 0
    assert sup.halo_count == 6
    assert tuple(sup.coords[0].tolist()) == (0, 0)
    assert sup.dist.tolist() == [0] * 7

    # The opening legal set is exactly the origin.
    state = api.new_game()
    assert sorted(api.legal_action_ids(state)) == [pack_action_id(0, 0)]


def test_scale_anchor_one_stone() -> None:
    # 1 stone: 217-cell core (1 stone + 216 legal) + 54 halo = 271 nodes.
    sup = build_support([(0, 0)])
    assert sup.num_nodes == 271
    assert sup.legal_count == 216
    assert sup.stone_count == 1
    assert sup.halo_count == 54


# --------------------------------------------------------------------------- #
# build_support decision-state contract (opt-in validation hook)
# --------------------------------------------------------------------------- #


def _engine_legal_coords(state) -> set[tuple[int, int]]:
    """The engine's legal set as a coord set."""

    return {unpack_action_id(int(a)) for a in api.legal_action_ids(state)}


def _state_stones(state) -> list[tuple[int, int]]:
    """Stones (occupied cells) of a state, in placement order."""

    mirror = api.to_python_state(state)
    return [(rec.coord.q, rec.coord.r) for rec in mirror.placement_history]


def _play_to_terminal(seed: int, max_plies: int = 400):
    """A genuinely terminal engine state, or None if the playout didn't end."""

    state = api.new_game()
    rng = random.Random(seed)
    for _ in range(max_plies):
        ids = api.legal_action_ids(state)
        if not ids:
            return None
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            return state
    return None


def _first_terminal_state():
    for seed in range(200):
        state = _play_to_terminal(seed)
        if state is not None and api.terminal(state) is not None:
            return state
    return None


def test_default_build_support_matches_no_expected_legal() -> None:
    """``expected_legal=None`` (the default) equals passing it explicitly."""

    stones = [(0, 0), (1, 0), (0, 1), (2, -1), (1, 1), (-1, 2), (3, 0)]
    a = build_support(stones)
    b = build_support(stones, expected_legal=None)
    assert np.array_equal(a.coords, b.coords)
    assert np.array_equal(a.dist, b.dist)
    assert np.array_equal(a.nbr, b.nbr)
    assert (a.legal_count, a.stone_count, a.halo_count) == (
        b.legal_count,
        b.stone_count,
        b.halo_count,
    )


def test_legal_prefix_layout_invariant_on_decision_state() -> None:
    """Legal nodes occupy exactly slots [0, legal_count); stones/halo follow."""

    stones = [(0, 0), (1, 0), (0, 1)]
    sup = build_support(stones)
    legal_rng, stone_rng, halo_rng = sup.segments()
    assert legal_rng == range(0, sup.legal_count)
    # no stone appears in the legal prefix
    legal = {tuple(c) for c in sup.coords[legal_rng].tolist()}
    assert legal.isdisjoint(set(stones))
    # the stones segment is exactly the input stones
    assert {tuple(c) for c in sup.coords[stone_rng].tolist()} == set(stones)
    assert len(halo_rng) == sup.halo_count


def test_ply0_validation_accepts_forced_origin() -> None:
    """Ply 0 forces {(0, 0)}; passing it as expected_legal must not raise."""

    sup = build_support([], expected_legal={(0, 0)})
    assert sup.legal_count == 1
    assert tuple(sup.coords[0]) == (0, 0)


def test_decision_state_validation_passes_against_engine() -> None:
    """On real decision states the closed-form legal set == the engine's."""

    checked = 0
    for seed in range(8):
        state = api.new_game()
        rng = random.Random(seed)
        for _ in range(seed % 5 + 3):
            ids = api.legal_action_ids(state)
            q, r = unpack_action_id(rng.choice(ids))
            api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if api.terminal(state) is not None:
            continue  # decision rows only
        stones = _state_stones(state)
        engine_legal = _engine_legal_coords(state)
        # opt-in validation must pass (no exception) and return the support
        sup = assert_decision_support(stones, engine_legal)
        assert sup.legal_count == len(engine_legal)
        checked += 1
    assert checked > 0, "no decision states sampled"


def test_terminal_state_diverges_and_is_flagged() -> None:
    """On a terminal state the engine legal set is empty while the closed form
    is not. Passing expected_legal raises SupportContractError; the default
    call (no expected_legal) does not."""

    state = _first_terminal_state()
    if state is None:
        pytest.skip("no terminal state reached in 200 random playouts")

    engine_legal = _engine_legal_coords(state)
    assert engine_legal == set(), "terminal engine legal set must be empty"

    stones = _state_stones(state)
    # Without expected_legal, closed-form legality still yields a legal prefix.
    unguarded = build_support(stones)
    assert unguarded.legal_count > 0, "closed-form legal set is non-empty"

    with pytest.raises(SupportContractError):
        build_support(stones, expected_legal=engine_legal)
    with pytest.raises(SupportContractError):
        assert_decision_support(stones, engine_legal)


def test_synthetic_terminal_like_empty_legal_is_flagged() -> None:
    """A non-empty closed-form set against an empty expected_legal raises, and
    the message reports coords in the closed form but not the engine set and
    marks the case TERMINAL."""

    stones = [(0, 0), (1, 0), (0, 1)]
    with pytest.raises(SupportContractError) as exc:
        build_support(stones, expected_legal=[])
    msg = str(exc.value)
    assert "in_closed_form_not_engine" in msg
    assert "TERMINAL" in msg


def test_partial_mismatch_is_flagged_both_directions() -> None:
    """A non-empty mismatch against a non-empty expected_legal raises and
    reports coords missing from and extra to the engine set."""

    stones = [(0, 0), (1, 0), (0, 1)]
    derived = {tuple(c) for c in build_support(stones).legal_coords().tolist()}
    bogus = (derived - {next(iter(derived))}) | {(99, 99)}  # drop one, add one
    with pytest.raises(SupportContractError) as exc:
        build_support(stones, expected_legal=bogus)
    msg = str(exc.value)
    assert "in_engine_not_closed_form" in msg
    assert "in_closed_form_not_engine" in msg
