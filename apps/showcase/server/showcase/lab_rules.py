"""Lab position validation (web-process side, engine only, no torch).

The lab accepts two position encodings:

- ``actions``: a chronological (q, r) placement list replayed as real Hexo
  turns. The engine is the rules authority; the first illegal placement or a
  terminal mid-sequence state rejects the whole request.
- ``stones``: a free-edit position, one cell list per color. There is no move
  history; the worker synthesizes facts from the raw stones and zeroes the
  history-derived input features (see ``lab.build_free_position``).

Both validators raise ``LabPositionError`` with a client-facing message; the
endpoint maps that to a 422.
"""

from __future__ import annotations

import hexo_engine as engine
from hexo_engine.types import AxialCoord, PlacementAction

# Hard caps, independent of the config limiters: a Hexo game at the showcase
# scale is well under 200 plies, and the featurizer cost grows with the
# support, so oversized requests are rejected before any worker time is spent.
MAX_ACTIONS = 400
MAX_FREE_STONES = 400
MAX_COORD = 4096

# Free-edit stone-count parity: legal Hexo prefixes keep the per-color counts
# within 1 of each other (1-then-2-2 turn structure); the sandbox allows a
# slack of 2 so mid-edit states stay usable, and rejects anything further out.
FREE_COUNT_SLACK = 2


class LabPositionError(ValueError):
    """A lab position failed validation; str(exc) is the client message."""


def _check_coord(q: int, r: int) -> None:
    if abs(q) > MAX_COORD or abs(r) > MAX_COORD:
        raise LabPositionError(f"cell ({q}, {r}) is out of range (|q|, |r| <= {MAX_COORD})")


def validate_actions(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Replay a chronological placement list through the engine.

    Returns the cells unchanged on success. Raises ``LabPositionError`` on an
    illegal placement, a terminal state before the sequence ends, or a
    terminal final state (the net evaluates decision states only).
    """
    if len(cells) > MAX_ACTIONS:
        raise LabPositionError(f"too many placements (max {MAX_ACTIONS})")
    state = engine.new_game()
    for i, (q, r) in enumerate(cells):
        _check_coord(q, r)
        action = PlacementAction(AxialCoord(q=int(q), r=int(r)))
        if not engine.is_legal_action(state, action):
            raise LabPositionError(f"placement {i} at ({q}, {r}) is illegal")
        result = engine.apply_action(state, action)
        if result.terminal and i < len(cells) - 1:
            raise LabPositionError(f"game ends at placement {i} ({q}, {r}); trailing moves are unreachable")
    if engine.terminal(state) is not None:
        raise LabPositionError("position is terminal (six in a line); the net evaluates decision states only")
    return cells


def validate_free_stones(
    p0: list[tuple[int, int]], p1: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Validate a free-edit stone set: bounds, no overlaps, plausible counts."""
    if len(p0) + len(p1) > MAX_FREE_STONES:
        raise LabPositionError(f"too many stones (max {MAX_FREE_STONES})")
    seen: set[tuple[int, int]] = set()
    for cells in (p0, p1):
        for q, r in cells:
            _check_coord(q, r)
            if (q, r) in seen:
                raise LabPositionError(f"cell ({q}, {r}) holds more than one stone")
            seen.add((q, r))
    if abs(len(p0) - len(p1)) > FREE_COUNT_SLACK:
        raise LabPositionError(
            f"stone counts {len(p0)} vs {len(p1)} are not near any legal turn parity "
            f"(difference must be <= {FREE_COUNT_SLACK})"
        )
    return p0, p1


def default_free_to_move(n_p0: int, n_p1: int) -> int:
    """Side to move for a free-edit position when the client does not say.

    In a legal sequence the side with fewer stones is (about to be) on the
    move; equal counts are ambiguous mid-turn states, defaulted to player 0.
    """
    if n_p0 > n_p1:
        return 1
    return 0
