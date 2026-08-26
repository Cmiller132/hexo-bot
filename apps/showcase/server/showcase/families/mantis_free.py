"""A history-less MantisNet lab position.

MantisNet's representation is a pure function of the stone set, the side to
move, and ``moves_remaining`` — move history and ply number are deliberately
absent from it (research ``MODEL_SPEC.md`` §3.5), and the encoder reads
stones in canonical order, never play order. A free-edit stone set therefore
evaluates *exactly*, not approximately: this class derives the engine-defined
read surface (stones, the legal frontier, the turn scalars) straight from the
raw stones, and ``mantisnet.builder.from_position`` builds the same graph it
builds for an engine-backed position.

The engine wrapper itself stays replay-only on purpose — a board-shaped
engine constructor would be a rule-bypass hole — so this class lives lab-side
only. It never enters search or the solver; those require an engine state,
which an unreachable position does not have.

Free-edit positions are decision states at the start of a turn: two
placements remaining, or the forced origin opening on an empty board. A stone
set that already completes six in a line is terminal and refused, exactly as
the sequence path refuses terminal positions.
"""

from __future__ import annotations

from mantisnet._rust import LEGAL_RADIUS

_AXES = ((1, 0), (0, 1), (1, -1))
_WIN_LEN = 6

# Every axial offset within LEGAL_RADIUS hex steps, the origin included:
# the legal frontier is the stone set dilated by this disk, minus occupancy.
_DISK = tuple(
    (dq, dr)
    for dq in range(-LEGAL_RADIUS, LEGAL_RADIUS + 1)
    for dr in range(-LEGAL_RADIUS, LEGAL_RADIUS + 1)
    if max(abs(dq), abs(dr), abs(dq + dr)) <= LEGAL_RADIUS
)


class FreeLabPosition:
    """The ``mantisnet._rust.Position`` read surface, from raw stones."""

    def __init__(
        self,
        p0: list[tuple[int, int]],
        p1: list[tuple[int, int]],
        to_move: int,
    ) -> None:
        if to_move not in (0, 1):
            raise ValueError(f"to_move must be 0 or 1, not {to_move!r}")
        owned = {}
        for owner, cells in ((0, p0), (1, p1)):
            for q, r in cells:
                cell = (int(q), int(r))
                if cell in owned:
                    raise ValueError(f"cell {cell} holds more than one stone")
                owned[cell] = owner
        if not owned and to_move != 0:
            raise ValueError("an empty board opens with player 0 at the origin")
        if _completed_six(owned):
            raise ValueError(
                "the stones complete six in a line — the position is terminal, "
                "and the net evaluates decision states only"
            )
        self._stones = sorted((q, r, owner) for (q, r), owner in owned.items())
        if owned:
            frontier = {
                (q + dq, r + dr)
                for (q, r) in owned
                for dq, dr in _DISK
            }
            self._legal = sorted(frontier - owned.keys())
            self.moves_remaining = 2
        else:
            self._legal = [(0, 0)]
            self.moves_remaining = 1
        self.current_player = int(to_move)

    is_terminal = False

    def stones(self) -> list[tuple[int, int, int]]:
        return list(self._stones)

    def legal_moves(self) -> list[tuple[int, int]]:
        return list(self._legal)

    @property
    def stone_count(self) -> int:
        return len(self._stones)

    @property
    def legal_count(self) -> int:
        return len(self._legal)

    def __repr__(self) -> str:
        return (
            f"<FreeLabPosition stones={len(self._stones)} "
            f"to_move={self.current_player}>"
        )


def _completed_six(owned: dict[tuple[int, int], int]) -> bool:
    """Whether any six consecutive cells on one axis share an owner."""
    for (q, r), owner in owned.items():
        for dq, dr in _AXES:
            if owned.get((q - dq, r - dr)) == owner:
                continue  # not the start of this run
            run, nq, nr = 0, q, r
            while owned.get((nq, nr)) == owner:
                run += 1
                nq, nr = nq + dq, nr + dr
            if run >= _WIN_LEN:
                return True
    return False
