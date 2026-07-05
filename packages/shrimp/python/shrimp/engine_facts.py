"""Convert engine state into PositionFacts.

Hot and standing-win lists are derived from the engine WindowStore mirror
(`mirror.board.windows`). The same lists can be derived without shard data
by `features.window_scan`.
"""

from __future__ import annotations

from hexo_engine import api
from hexo_engine.types import Player, PythonHexoState, TurnPhase

from .constants import HOT_MIN_COUNT, HOT_MIN_PLACEMENTS, WIN_NOW_COUNT, WINDOW_LEN
from .features import AXIS_DELTAS, PositionFacts


def player_int(player: Player) -> int:
    return 0 if player == Player.PLAYER_0 else 1


def facts_from_engine(mirror: PythonHexoState) -> PositionFacts:
    """Facts for the current decision state of a (non-terminal) engine mirror."""

    records = tuple(
        (rec.coord.q, rec.coord.r, player_int(rec.player), rec.placement_index)
        for rec in mirror.placement_history
    )
    current = player_int(mirror.current_player)
    first = (
        (mirror.first_stone.q, mirror.first_stone.r)
        if mirror.phase == TurnPhase.SECOND_STONE and mirror.first_stone is not None
        else None
    )

    own_hot: set[tuple[int, int]] = set()
    opp_hot: set[tuple[int, int]] = set()
    own_win: set[tuple[int, int]] = set()
    opp_win: set[tuple[int, int]] = set()
    for entry in mirror.board.windows.entries:
        p0, p1 = entry.masks
        c0, c1 = bin(p0).count("1"), bin(p1).count("1")
        if c0 > 0 and c1 > 0:
            continue
        count = c0 + c1
        owner = 0 if c0 > 0 else 1
        if count < HOT_MIN_COUNT:
            continue
        dq, dr = AXIS_DELTAS[entry.key.axis]
        union = p0 | p1
        empties = [
            (entry.key.start.q + i * dq, entry.key.start.r + i * dr)
            for i in range(WINDOW_LEN)
            if not (union >> i) & 1
        ]
        if count == WIN_NOW_COUNT:
            (own_win if owner == current else opp_win).update(empties)
        if mirror.placements_made >= HOT_MIN_PLACEMENTS:
            (own_hot if owner == current else opp_hot).update(empties)

    return PositionFacts(
        records=records,
        current_player=current,
        phase=mirror.phase.value,
        first_stone=first,
        own_hot=tuple(sorted(own_hot)),
        opp_hot=tuple(sorted(opp_hot)),
        own_win=tuple(sorted(own_win)),
        opp_win=tuple(sorted(opp_win)),
    )


def facts_from_state(state: "api.HexoState") -> PositionFacts:
    """Convert a HexoState by building its Python mirror, then converting."""

    return facts_from_engine(api.to_python_state(state))
