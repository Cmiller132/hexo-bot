"""The vendored Rust batch builder against the ported Python builder.

Ported from the research repo's ``tests/test_rust_builder.py`` (branch main,
MODEL_REPR_VERSION 7). The Python builder is the reference implementation;
every integral tensor returned by ``mantisnet._rust.build_batch`` must equal
its Python-builder counterpart. This is the correctness gate for the vendored
encoder plus the pyo3 marshaling.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")
_rust = pytest.importorskip("mantisnet._rust")

from mantisnet.builder import (  # noqa: E402
    collate,
    collate_positions,
    collate_prefixes,
    from_position,
)

# P0 wins on the first stone of its turn (T = 11): a run of six completed
# along the Q axis. From the research repo's test_klent_returns.py.
FIRST_STONE_WIN = [
    (0, 0),
    (-8, 8), (-8, 9),
    (1, 0), (2, 0),
    (-8, 10), (-6, 8),
    (3, 0), (4, 0),
    (-6, 9), (-6, 10),
    (5, 0),
]

# Ply depths for the shared position set: both movers, all three turn phases,
# both stones of a turn, and boards from empty to crowded.
PLIES = [0, 1, 2, 3, 5, 9, 12, 21, 34, 60]


def random_moves(plies: int, seed: int) -> list[tuple[int, int]]:
    """A uniformly random legal playout of exactly ``plies`` placements that
    does not end the game, retrying playouts that terminate early."""
    for attempt in range(100):
        rng = random.Random(seed * 1_000_003 + attempt * 1_009 + plies)
        pos = _rust.Position()
        moves: list[tuple[int, int]] = []
        for _ in range(plies):
            pos.advance(*(m := rng.choice(pos.legal_moves())))
            moves.append(m)
        if not pos.is_terminal:
            return moves
    raise AssertionError(f"no non-terminal {plies}-ply playout in 100 seeds")


@pytest.fixture(scope="module")
def move_lists() -> list[list[tuple[int, int]]]:
    return [random_moves(p, seed=7) for p in PLIES] + [random_moves(45, seed=1234)]


@pytest.fixture(scope="module")
def positions(move_lists):
    return [_rust.Position.replay(m) for m in move_lists]


_TENSOR_FIELDS = [
    "stone_own",
    "window_feat",
    "window_id",
    "moves_idx",
    "inc_stone",
    "inc_window",
    "inc_class",
    "stone_slot",
    "coords",
    "attn_valid",
    "window_slot",
    "value_valid",
    "legal_offsets",
    "cell_pos",
    "cell_occupancy",
    "cell_is_legal",
    "cell_nearest",
    "radius_src",
    "radius_dst",
    "radius_orbit",
    "radius_own",
    "radius_on_axis",
    "adjacency_src",
    "adjacency_dst",
    "adjacency_axis",
    "dec_cell",
    "dec_window",
    "dec_class",
    "act_class",
    "act_rev",
    "act_empty",
]


def _assert_equal(rust, python):
    assert (rust.n_pos, rust.max_t, rust.max_w, rust.n_cells) == (
        python.n_pos,
        python.max_t,
        python.max_w,
        python.n_cells,
    )
    for name in _TENSOR_FIELDS:
        a, b = getattr(rust, name), getattr(python, name)
        assert a.dtype == b.dtype, f"{name}: {a.dtype} != {b.dtype}"
        assert torch.equal(a, b), f"{name} differs"


def test_position_batch_parity(positions):
    _assert_equal(
        collate_positions(positions), collate([from_position(p) for p in positions])
    )


def test_single_position_batches_including_ply_zero(positions):
    for pos in [_rust.Position(), positions[-1]]:
        _assert_equal(collate_positions([pos]), collate([from_position(pos)]))


def test_prefix_batch_parity(move_lists):
    games = [m for m in move_lists if m]
    ts = [len(m) for m in games] + [1, len(games[-1]) // 2]
    games = games + [games[-1], games[-1]]
    rust = collate_prefixes(games, ts)
    python = collate(
        [from_position(_rust.Position.replay(list(g[:t]))) for g, t in zip(games, ts)]
    )
    _assert_equal(rust, python)


def test_terminal_position_refused():
    pos = _rust.Position.replay(FIRST_STONE_WIN)
    assert pos.is_terminal
    with pytest.raises(ValueError, match="terminal"):
        collate_positions([pos])
    with pytest.raises(ValueError, match="terminal"):
        collate_prefixes([FIRST_STONE_WIN], [len(FIRST_STONE_WIN)])


def test_bad_prefix_refused():
    with pytest.raises(ValueError, match="exceeds"):
        collate_prefixes([[(0, 0)]], [5])
