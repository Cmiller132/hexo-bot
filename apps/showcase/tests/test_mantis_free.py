"""FreeLabPosition against the engine, and the builder dispatch.

The free position must derive exactly the read surface the engine would
give for a reachable stone set — same legal frontier, same turn scalars —
and `collate_positions` must route it through the reference Python builder
while refusing mixed batches.
"""

from __future__ import annotations

import pytest

_ = pytest.importorskip("mantisnet")

from mantisnet import _rust, collate_positions
from showcase.families.mantis_free import FreeLabPosition

# A fresh-turn position: the origin opening plus one full P1 turn.
MOVES = [(0, 0), (1, 0), (0, 1)]


def _split(position):
    stones = position.stones()
    return (
        [(q, r) for q, r, p in stones if p == 0],
        [(q, r) for q, r, p in stones if p == 1],
    )


def test_free_position_matches_the_engine_on_a_reachable_set():
    real = _rust.Position.replay(MOVES)
    p0, p1 = _split(real)
    free = FreeLabPosition(p0, p1, real.current_player)
    assert free.current_player == real.current_player
    assert free.moves_remaining == real.moves_remaining == 2
    assert free.stone_count == real.stone_count
    assert sorted(free.stones()) == sorted(real.stones())
    assert free.legal_count == real.legal_count
    assert sorted(free.legal_moves()) == sorted(real.legal_moves())
    assert free.is_terminal is False


def test_free_origin_frontier_is_the_radius_disk():
    free = FreeLabPosition([(0, 0)], [], 1)
    r = _rust.LEGAL_RADIUS
    assert free.legal_count == 3 * r * (r + 1)  # disk minus the stone itself
    assert _rust.Position.replay([(0, 0)]).legal_count == free.legal_count


def test_free_empty_board_is_the_forced_opening():
    free = FreeLabPosition([], [], 0)
    assert free.legal_moves() == [(0, 0)]
    assert free.moves_remaining == 1 and free.current_player == 0
    with pytest.raises(ValueError, match="player 0"):
        FreeLabPosition([], [], 1)


def test_free_position_refuses_a_completed_six():
    with pytest.raises(ValueError, match="six"):
        FreeLabPosition([(q, 0) for q in range(6)], [(0, 5), (1, 5)], 1)
    # Five in a line is a decision state, not a terminal one.
    FreeLabPosition([(q, 0) for q in range(5)], [(0, 5), (1, 5)], 1)


def test_free_position_refuses_overlap():
    with pytest.raises(ValueError, match="more than one stone"):
        FreeLabPosition([(0, 0)], [(0, 0)], 0)


def test_collate_dispatch_builds_free_positions_and_refuses_mixes():
    import torch

    real = _rust.Position.replay(MOVES)
    p0, p1 = _split(real)
    free = FreeLabPosition(p0, p1, real.current_player)

    with pytest.raises(ValueError, match="mix"):
        collate_positions([real, free])

    rust_batch = collate_positions([real])
    py_batch = collate_positions([free])
    # Same entity counts and turn scalar; orders may differ (engine scan vs
    # canonical sort), so contents are compared as sets where it matters.
    assert py_batch.n_pos == rust_batch.n_pos == 1
    assert py_batch.moves_idx.tolist() == rust_batch.moves_idx.tolist()
    assert py_batch.stone_own.shape == rust_batch.stone_own.shape
    assert py_batch.window_feat.shape == rust_batch.window_feat.shape
    assert py_batch.n_cells == rust_batch.n_cells
    assert (
        sorted(map(tuple, py_batch.window_id.tolist()))
        == sorted(map(tuple, rust_batch.window_id.tolist()))
    )
    assert torch.equal(
        torch.sort(py_batch.cell_nearest).values,
        torch.sort(rust_batch.cell_nearest).values,
    )
