"""Node features (F = 15) from stored position facts.

Expands feature rows from raw facts (the hexfield_compact_v1 columns): the
placement history (q, r, owner, placement_index), phase, first stone, and the
side-relative hot / standing-win cell lists. Indices 0-10 and 12 follow the
dense_cnn plane semantics; index 11 is distance-to-nearest-stone; 13-14 are
the standing-win planes.

Turn structure is deterministic (1-then-2-2-2...), so each history record's
phase and player are derived from its ordinal position; the shard schema
stores neither.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .constants import (
    F_DIST_TO_STONE,
    F_EMPTY,
    F_FIRST_STONE,
    F_LEGAL,
    F_OPP_HOT,
    F_OPP_LAST_TURN,
    F_OPP_RECENCY,
    F_OPP_STONE,
    F_OPP_WIN_NOW,
    F_OWN_HOT,
    F_OWN_RECENCY,
    F_OWN_STONE,
    F_OWN_WIN_NOW,
    F_PHASE_SECOND,
    F_PLAYER_COLOUR,
    DIST_SCALE,
    HOT_MIN_COUNT,
    HOT_MIN_PLACEMENTS,
    NUM_FEATURES,
    WIN_NOW_COUNT,
    WINDOW_LEN,
)
from .geometry import apply_d6
from .support import Support, build_support

PHASE_OPENING = "Opening"
PHASE_FIRST = "FirstStone"
PHASE_SECOND = "SecondStone"

# Win axes: Q=(1,0), R=(0,1), QR=(1,-1).
AXIS_DELTAS: dict[str, tuple[int, int]] = {"Q": (1, 0), "R": (0, 1), "QR": (1, -1)}


@dataclass(frozen=True)
class PositionFacts:
    """Raw facts for one decision state, side-relative where applicable.

    records: chronological placement history (q, r, owner, placement_index);
    owner is 0/1 (player0/player1). placements_made == len(records). Hot / win
    lists are relative to ``current_player``.
    """

    records: tuple[tuple[int, int, int, int], ...]
    current_player: int
    phase: str
    first_stone: tuple[int, int] | None
    own_hot: tuple[tuple[int, int], ...]
    opp_hot: tuple[tuple[int, int], ...]
    own_win: tuple[tuple[int, int], ...]
    opp_win: tuple[tuple[int, int], ...]

    @property
    def placements_made(self) -> int:
        return len(self.records)

    def stones(self) -> list[tuple[int, int]]:
        return [(q, r) for q, r, _owner, _idx in self.records]


def record_phase(ordinal: int) -> str:
    """Phase of the history record at chronological position ``ordinal``."""

    if ordinal == 0:
        return PHASE_OPENING
    return PHASE_FIRST if (ordinal - 1) % 2 == 0 else PHASE_SECOND


def record_player(ordinal: int) -> int:
    """Player (0/1) of the history record at position ``ordinal``."""

    if ordinal == 0:
        return 0
    return 1 if ((ordinal - 1) // 2) % 2 == 0 else 0


# --- window scan (threat/standing-win logic) ------------------------------------


def window_scan(
    records: tuple[tuple[int, int, int, int], ...],
    current_player: int,
    placements_made: int,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    """(own_hot, opp_hot, own_win, opp_win) cell lists from raw stones.

    Scans single-colour windows only. A window is hot when its stone count is
    >= HOT_MIN_COUNT and placements_made >= HOT_MIN_PLACEMENTS; its empty cells
    are marked hot. A window is a standing win when its stone count ==
    WIN_NOW_COUNT (exactly one empty cell); its empty cell is marked win.
    Hot/win cells are assigned to own_* or opp_* by comparing the window
    owner against ``current_player``.
    """

    owner_at: dict[tuple[int, int], int] = {(q, r): owner for q, r, owner, _ in records}
    own_hot: set[tuple[int, int]] = set()
    opp_hot: set[tuple[int, int]] = set()
    own_win: set[tuple[int, int]] = set()
    opp_win: set[tuple[int, int]] = set()

    seen: set[tuple[int, int, str]] = set()
    for (sq, sr) in owner_at:
        for axis, (dq, dr) in AXIS_DELTAS.items():
            for back in range(WINDOW_LEN):
                start = (sq - back * dq, sr - back * dr)
                key = (start[0], start[1], axis)
                if key in seen:
                    continue
                seen.add(key)
                cells = [
                    (start[0] + i * dq, start[1] + i * dr) for i in range(WINDOW_LEN)
                ]
                counts = [0, 0]
                empties: list[tuple[int, int]] = []
                for cell in cells:
                    owner = owner_at.get(cell)
                    if owner is None:
                        empties.append(cell)
                    else:
                        counts[owner] += 1
                if counts[0] > 0 and counts[1] > 0:
                    continue  # two-coloured window: skipped
                count = counts[0] + counts[1]
                owner = 0 if counts[0] > 0 else 1
                if count == WIN_NOW_COUNT:
                    (own_win if owner == current_player else opp_win).update(empties)
                if count >= HOT_MIN_COUNT and placements_made >= HOT_MIN_PLACEMENTS:
                    (own_hot if owner == current_player else opp_hot).update(empties)

    return (
        tuple(sorted(own_hot)),
        tuple(sorted(opp_hot)),
        tuple(sorted(own_win)),
        tuple(sorted(opp_win)),
    )


# --- feature build ----------------------------------------------------------------


def build_features(facts: PositionFacts, sup: Support) -> np.ndarray:
    """(N, 15) float32 feature matrix in support node order."""

    n = sup.num_nodes
    feats = np.zeros((n, NUM_FEATURES), dtype=np.float32)

    placements_made = facts.placements_made
    for q, r, owner, placement_index in facts.records:
        row = sup.index[(q, r)]
        if owner == facts.current_player:
            feats[row, F_OWN_STONE] = 1.0
            recency_plane = F_OWN_RECENCY
        else:
            feats[row, F_OPP_STONE] = 1.0
            recency_plane = F_OPP_RECENCY
        # age = placements_made - placement_index; recency weight = 1/(1+age).
        age = placements_made - placement_index
        weight = 1.0 / (1.0 + float(age))
        feats[row, recency_plane] = max(feats[row, recency_plane], weight)

    feats[:, F_EMPTY] = 1.0 - feats[:, F_OWN_STONE] - feats[:, F_OPP_STONE]
    feats[: sup.legal_count, F_LEGAL] = 1.0

    if facts.phase == PHASE_SECOND:
        feats[:, F_PHASE_SECOND] = 1.0
        if facts.first_stone is not None:
            feats[sup.index[facts.first_stone], F_FIRST_STONE] = 1.0

    if facts.current_player == 0:
        feats[:, F_PLAYER_COLOUR] = 1.0

    for cell in facts.opp_hot:
        feats[sup.index[cell], F_OPP_HOT] = 1.0
    for cell in facts.own_hot:
        feats[sup.index[cell], F_OWN_HOT] = 1.0
    for cell in facts.opp_win:
        feats[sup.index[cell], F_OPP_WIN_NOW] = 1.0
    for cell in facts.own_win:
        feats[sup.index[cell], F_OWN_WIN_NOW] = 1.0

    feats[:, F_DIST_TO_STONE] = sup.dist.astype(np.float32) / DIST_SCALE

    for cell in _opp_last_turn_cells(facts):
        feats[sup.index[cell], F_OPP_LAST_TURN] = 1.0

    return feats


def _opp_last_turn_cells(facts: PositionFacts) -> list[tuple[int, int]]:
    """Cells of the opponent's most recent full turn.

    Scans history in reverse. The first opponent SecondStone record returns
    both its own cell and the previous record's cell (its first-stone
    companion); an opponent Opening record returns its single cell; opponent
    FirstStone records are skipped."""

    opponent = 1 - facts.current_player
    records = facts.records
    for ordinal in range(len(records) - 1, -1, -1):
        if record_player(ordinal) != opponent:
            continue
        phase = record_phase(ordinal)
        q, r, _owner, _idx = records[ordinal]
        if phase == PHASE_SECOND:
            fq, fr, _o, _i = records[ordinal - 1]
            return [(fq, fr), (q, r)]
        if phase == PHASE_OPENING:
            return [(q, r)]
    return []


def build_position(facts: PositionFacts) -> tuple[Support, np.ndarray]:
    """Support + features for one decision state."""

    sup = build_support(facts.stones())
    return sup, build_features(facts, sup)


# --- D6 augmentation ----------------------------------------------------------------


def transform_facts(facts: PositionFacts, sym: int) -> PositionFacts:
    """Apply D6 transform ``sym`` to every stored coordinate fact.

    Transforms the coordinates in records, first_stone, and the hot/win cell
    lists; all other fields are copied unchanged. Support, node order,
    neighbour table, and features are rebuilt from the transformed facts."""

    def t(cell: tuple[int, int]) -> tuple[int, int]:
        return apply_d6(sym, cell[0], cell[1])

    return replace(
        facts,
        records=tuple(
            (*t((q, r)), owner, idx) for q, r, owner, idx in facts.records
        ),
        first_stone=t(facts.first_stone) if facts.first_stone is not None else None,
        own_hot=tuple(sorted(t(c) for c in facts.own_hot)),
        opp_hot=tuple(sorted(t(c) for c in facts.opp_hot)),
        own_win=tuple(sorted(t(c) for c in facts.own_win)),
        opp_win=tuple(sorted(t(c) for c in facts.opp_win)),
    )
