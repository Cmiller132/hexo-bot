"""Convert engine state into PositionFacts.

The graded per-axis window planes are recomputed from the placement history at
feature-build time (``features.build_features``), so these facts carry only the
placement history, phase, and first stone — no derived hot/standing-win lists.
"""

from __future__ import annotations

from hexo_engine import api
from hexo_engine.types import Player, PythonHexoState, TurnPhase

from .features import PositionFacts


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

    return PositionFacts(
        records=records,
        current_player=current,
        phase=mirror.phase.value,
        first_stone=first,
    )


def facts_from_state(state: "api.HexoState") -> PositionFacts:
    """Convert a HexoState by building its Python mirror, then converting."""

    return facts_from_engine(api.to_python_state(state))
