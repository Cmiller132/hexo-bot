"""Tests comparing hexfield features against the engine.

Covers:
- window_scan output vs the engine WindowStore on sampled states
- own_win_now cells applied to the engine terminate as wins for the side to move
- opp_last_turn plane vs the engine placement records
- record phase/player derived from ordinal position vs engine records
- D6 commutation across all 12 symmetries: build(transform(facts)) equals the
  permuted build(facts)
- ply-0 features
"""

from __future__ import annotations

import numpy as np

from hexfield_testkit import api, sample_decision_states

from hexfield import constants as C
from hexfield.engine_facts import facts_from_engine, player_int
from hexfield.features import (
    PHASE_OPENING,
    PHASE_SECOND,
    build_position,
    record_phase,
    record_player,
    transform_facts,
    window_scan,
)
from hexo_engine.types import AxialCoord, PlacementAction


def _states():
    return sample_decision_states(range(6), (1, 4, 8, 13, 21, 34, 49))


def test_window_scan_matches_engine_window_store() -> None:
    for state in _states():
        mirror = api.to_python_state(state)
        facts = facts_from_engine(mirror)
        derived = window_scan(facts.records, facts.current_player, facts.placements_made)
        assert derived == (facts.own_hot, facts.opp_hot, facts.own_win, facts.opp_win)


def _apply(state, q: int, r: int):
    return api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))


def test_own_win_now_cells_win_immediately() -> None:
    # P0 places five in a row on the Q axis; P1 places non-collinear stones
    # and does not block.
    state = api.new_game()
    for q, r in [
        (0, 0),  # P0 opening
        (0, 4), (4, -4),  # P1
        (1, 0), (2, 0),  # P0
        (-4, 0), (0, -4),  # P1
        (3, 0), (4, 0),  # P0 -> five in a row
        (8, -8), (-8, 4),  # P1
    ]:
        result = _apply(state, q, r)
        assert not result.terminal

    facts = facts_from_engine(api.to_python_state(state))
    assert facts.current_player == 0
    # Both extension cells of the 5-chain are win-in-1 for the side to move.
    assert set(facts.own_win) == {(-1, 0), (5, 0)}
    assert facts.opp_win == ()

    for cell in facts.own_win:
        clone = api.clone_state(state)
        result = _apply(clone, *cell)
        assert result.terminal
        terminal = api.terminal(clone)
        assert terminal is not None
        assert player_int(terminal.winner) == 0

    # The same chain seen from P1's side (one turn earlier), replayed without
    # P1's final turn, appears in opp_win.
    state2 = api.new_game()
    for q, r in [
        (0, 0),
        (0, 4), (4, -4),
        (1, 0), (2, 0),
        (-4, 0), (0, -4),
        (3, 0), (4, 0),
    ]:
        assert not _apply(state2, q, r).terminal
    facts2 = facts_from_engine(api.to_python_state(state2))
    assert facts2.current_player == 1
    assert set(facts2.opp_win) == {(-1, 0), (5, 0)}
    assert facts2.own_win == ()

    # Over sampled states, each own_win cell terminates as a win for the
    # current player when applied.
    for state in _states():
        facts = facts_from_engine(api.to_python_state(state))
        for cell in facts.own_win[:3]:
            clone = api.clone_state(state)
            result = _apply(clone, *cell)
            assert result.terminal
            terminal = api.terminal(clone)
            assert terminal is not None
            assert player_int(terminal.winner) == facts.current_player


def test_record_phase_player_derivation_matches_engine() -> None:
    for state in _states():
        mirror = api.to_python_state(state)
        for ordinal, rec in enumerate(mirror.placement_history):
            # Engine placement_index is 1-based; the most recent record has
            # placement_index == placements_made.
            assert rec.placement_index == ordinal + 1
            assert rec.phase.value == record_phase(ordinal)
            assert player_int(rec.player) == record_player(ordinal)


def test_features_against_engine_state() -> None:
    for state in _states():
        mirror = api.to_python_state(state)
        facts = facts_from_engine(mirror)
        sup, feats = build_position(facts)

        # Stone planes partition with empty; legal is exactly the prefix.
        assert np.allclose(
            feats[:, C.F_OWN_STONE] + feats[:, C.F_OPP_STONE] + feats[:, C.F_EMPTY], 1.0
        )
        assert feats[: sup.legal_count, C.F_LEGAL].min() == 1.0
        assert feats[sup.legal_count :, C.F_LEGAL].max() == 0.0
        assert int(feats[:, C.F_OWN_STONE].sum() + feats[:, C.F_OPP_STONE].sum()) == len(
            facts.records
        )

        # dist_to_stone: stones 0, legal in (0, 1], halo exactly 1.125.
        legal_rng, stone_rng, halo_rng = sup.segments()
        d = feats[:, C.F_DIST_TO_STONE]
        if len(stone_rng):
            assert d[list(stone_rng)].max() == 0.0
        if len(legal_rng):
            assert d[list(legal_rng)].min() > 0.0
            assert d[list(legal_rng)].max() <= 1.0
        assert np.allclose(d[list(halo_rng)], C.HALO_DIST_FEATURE)

        # Constant planes.
        assert np.all(
            feats[:, C.F_PHASE_SECOND] == (1.0 if facts.phase == PHASE_SECOND else 0.0)
        )
        assert np.all(
            feats[:, C.F_PLAYER_COLOUR] == (1.0 if facts.current_player == 0 else 0.0)
        )
        if facts.phase == PHASE_SECOND:
            assert facts.first_stone is not None
            assert feats[sup.index[facts.first_stone], C.F_FIRST_STONE] == 1.0
            assert feats[:, C.F_FIRST_STONE].sum() == 1.0
        else:
            assert feats[:, C.F_FIRST_STONE].max() == 0.0

        # Recency: nonzero exactly at stones; most recent record carries the max.
        recency = feats[:, C.F_OWN_RECENCY] + feats[:, C.F_OPP_RECENCY]
        occupied = feats[:, C.F_OWN_STONE] + feats[:, C.F_OPP_STONE]
        assert np.all((recency > 0) == (occupied > 0))
        last_q, last_r, _o, last_idx = facts.records[-1]
        expected = 1.0 / (1.0 + (facts.placements_made - last_idx))
        assert np.isclose(recency[sup.index[(last_q, last_r)]], expected)

        # opp_last_turn: expected set built by scanning the engine's placement
        # records using their phase / first_stone fields. mirror.last_turn is
        # not used here because it tracks the in-progress turn while the current
        # player is mid-turn.
        marked = {
            tuple(sup.coords[i].tolist())
            for i in np.flatnonzero(feats[:, C.F_OPP_LAST_TURN])
        }
        expected: set[tuple[int, int]] = set()
        opponent = 1 - facts.current_player
        for rec in reversed(mirror.placement_history):
            if player_int(rec.player) != opponent:
                continue
            if rec.phase.value == "SecondStone":
                assert rec.first_stone is not None
                expected = {
                    (rec.first_stone.q, rec.first_stone.r),
                    (rec.coord.q, rec.coord.r),
                }
                break
            if rec.phase.value == "Opening":
                expected = {(rec.coord.q, rec.coord.r)}
                break
        assert marked == expected

        # Hot / win planes mark exactly the facts lists.
        for plane, cells in (
            (C.F_OPP_HOT, facts.opp_hot),
            (C.F_OWN_HOT, facts.own_hot),
            (C.F_OPP_WIN_NOW, facts.opp_win),
            (C.F_OWN_WIN_NOW, facts.own_win),
        ):
            rows = {
                tuple(sup.coords[i].tolist()) for i in np.flatnonzero(feats[:, plane])
            }
            assert rows == set(cells)


def test_d6_commutation_12_of_12() -> None:
    states = sample_decision_states(range(3), (2, 9, 18, 33))
    assert states
    for state in states:
        facts = facts_from_engine(api.to_python_state(state))
        sup, feats = build_position(facts)
        coords = [tuple(c) for c in sup.coords.tolist()]
        for sym in range(12):
            tfacts = transform_facts(facts, sym)
            tsup, tfeats = build_position(tfacts)
            assert tsup.num_nodes == sup.num_nodes
            assert tsup.legal_count == sup.legal_count
            assert tsup.stone_count == sup.stone_count
            from hexfield.geometry import apply_d6

            for row, cell in enumerate(coords):
                trow = tsup.index[apply_d6(sym, *cell)]
                # Segment membership is preserved...
                assert (trow < tsup.legal_count) == (row < sup.legal_count)
                in_stones = sup.legal_count <= row < sup.legal_count + sup.stone_count
                t_in_stones = (
                    tsup.legal_count <= trow < tsup.legal_count + tsup.stone_count
                )
                assert in_stones == t_in_stones
                # ...and every feature value commutes exactly.
                assert np.array_equal(feats[row], tfeats[trow])


def test_ply0_features() -> None:
    state = api.new_game()
    facts = facts_from_engine(api.to_python_state(state))
    sup, feats = build_position(facts)
    assert sup.num_nodes == 7
    assert facts.phase == PHASE_OPENING
    assert feats[0, C.F_LEGAL] == 1.0
    assert feats[:, C.F_LEGAL].sum() == 1.0
    assert np.all(feats[:, C.F_DIST_TO_STONE] == 0.0)
    assert np.all(feats[:, C.F_PLAYER_COLOUR] == 1.0)  # player0 to move
    assert np.all(feats[:, C.F_EMPTY] == 1.0)
    for plane in (
        C.F_OWN_STONE,
        C.F_OPP_STONE,
        C.F_PHASE_SECOND,
        C.F_FIRST_STONE,
        C.F_OWN_RECENCY,
        C.F_OPP_RECENCY,
        C.F_OPP_HOT,
        C.F_OWN_HOT,
        C.F_OPP_LAST_TURN,
        C.F_OPP_WIN_NOW,
        C.F_OWN_WIN_NOW,
    ):
        assert feats[:, plane].max() == 0.0
