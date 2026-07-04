"""Lightweight Python types for the engine API boundary.

These objects describe the data Python callers should exchange with
`hexo_engine.api`. The actual rule interpretation stays in Rust; these types are
only handles, identifiers, and transport shapes.

Cross-language contract: `pack_coord_id`/`unpack_coord_id` deliberately
duplicate the packing in rust/src/legal.rs (`pack_coord`/`unpack_coord`) and
the JS re-implementation in hexo_frontend/static/app.js (DBG_COORD_OFFSET).
Packed IDs are persisted in training .npz shards and .hxr game records, so the
three implementations must never diverge; tests/test_hexo_engine_rust_bridge.py
cross-checks this one against `engine.action_id`.

The read-only `Python*` mirror dataclasses are the shape produced by
`api.to_python_state` (consumed mainly by hexo_frontend/dashboard.py); hot
paths bypass them via raw `legal_action_ids` tuples.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, TypeAlias


ActionId = int
LegalActionId = int
_COORD_OFFSET = 1 << 15
_COORD_MIN = -(1 << 15)
_COORD_MAX = (1 << 15) - 1


class Player(StrEnum):
    """Canonical player labels exposed to Python callers."""

    PLAYER_0 = "player0"
    PLAYER_1 = "player1"


class TurnPhase(StrEnum):
    """Rust-like phase of the autoregressive Hexo turn."""

    OPENING = "Opening"
    FIRST_STONE = "FirstStone"
    SECOND_STONE = "SecondStone"


@dataclass(frozen=True, slots=True)
class AxialCoord:
    """Axial hex coordinate passed through the Python API."""

    q: int
    r: int


@dataclass(frozen=True, slots=True)
class PlacementAction:
    """One single-placement action submitted to the Rust engine."""

    coord: AxialCoord


Action: TypeAlias = PlacementAction


class LegalActions(Sequence[PlacementAction]):
    """Deterministic legal action view backed by compact engine action IDs."""

    __slots__ = ("_ids", "_id_set")

    def __init__(self, action_ids: Sequence[int]) -> None:
        self._ids = tuple(action_ids)
        self._id_set = frozenset(self._ids)

    @property
    def action_ids(self) -> tuple[LegalActionId, ...]:
        """Compact deterministic legal action IDs."""

        return self._ids

    def coords(self) -> tuple[AxialCoord, ...]:
        """Legal coordinates without wrapping them in action objects."""

        return tuple(unpack_coord_id(action_id) for action_id in self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[PlacementAction]:
        for action_id in self._ids:
            yield PlacementAction(unpack_coord_id(action_id))

    def __getitem__(self, index: int | slice) -> PlacementAction | tuple[PlacementAction, ...]:
        if isinstance(index, slice):
            return tuple(PlacementAction(unpack_coord_id(action_id)) for action_id in self._ids[index])
        return PlacementAction(unpack_coord_id(self._ids[index]))

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, PlacementAction):
            return False
        return pack_coord_id(item.coord) in self._id_set


def pack_coord_id(coord: AxialCoord) -> LegalActionId:
    """Pack a coordinate the same way the Rust legal move store does.

    Encoding: `((q + 2^15) << 16) | (r + 2^15)`; raw integer order therefore
    matches deterministic signed (q, r) order. Must stay byte-identical to
    rust/src/legal.rs `pack_coord` (IDs are persisted in shards/records).
    Raises ValueError if a component leaves the i16 range.
    """

    q = _checked_coord_component(coord.q)
    r = _checked_coord_component(coord.r)
    return ((q + _COORD_OFFSET) << 16) | (r + _COORD_OFFSET)


# UNUSED(2026-06-12): no references found in packages/tests/scripts (excl.
# scripts/archive); not re-exported by __init__ either. Kept as a debugging
# convenience only.
def format_coord_id(action_id: ActionId) -> str:
    """Format a packed action ID for display only."""

    coord = unpack_coord_id(action_id)
    return f"{coord.q},{coord.r}"


def unpack_coord_id(action_id: LegalActionId) -> AxialCoord:
    """Unpack a compact legal action ID into an axial coordinate."""

    action_id = int(action_id)
    q = (action_id >> 16) - _COORD_OFFSET
    r = (action_id & 0xFFFF) - _COORD_OFFSET
    return AxialCoord(q=q, r=r)


def _checked_coord_component(value: int) -> int:
    value = int(value)
    if value < _COORD_MIN or value > _COORD_MAX:
        raise ValueError(f"coordinate component outside i16 range: {value}")
    return value


# UNUSED(2026-06-12): never constructed anywhere in packages/tests/scripts —
# api.to_python_state assigns a TerminalResult to PythonHexoState.terminal
# instead, so the `PythonTerminal | None` annotation below is wrong at runtime
# (the actual object has winner/reason/metadata, not winner/placements). Kept
# only because it is re-exported by __init__; fixing means either constructing
# this here or retyping the annotation and deleting this class.
@dataclass(frozen=True, slots=True)
class PythonTerminal:
    """Read-only Python mirror of Rust `GameOutcome`."""

    winner: Player
    placements: int


@dataclass(frozen=True, slots=True)
class PythonMoveRecord:
    """Read-only Python mirror of Rust `MoveRecord`."""

    player: Player
    placements: tuple[AxialCoord, ...]


@dataclass(frozen=True, slots=True)
class PythonPlacementRecord:
    """Read-only Python mirror of Rust `PlacementRecord`."""

    player: Player
    coord: AxialCoord
    phase: TurnPhase
    placement_index: int
    first_stone: AxialCoord | None = None


@dataclass(frozen=True, slots=True)
class PythonWindowKey:
    """Read-only Python mirror of Rust `WindowKey`."""

    start: AxialCoord
    axis: str


@dataclass(frozen=True, slots=True)
class PythonWindowEntry:
    """Read-only Python mirror of Rust `WindowEntry`."""

    key: PythonWindowKey
    masks: tuple[int, int]


@dataclass(frozen=True, slots=True)
class PythonWindowStore:
    """Read-only Python mirror of Rust `WindowStore`."""

    entries: tuple[PythonWindowEntry, ...] = ()

    @property
    def len(self) -> int:
        return len(self.entries)

    @property
    def is_empty(self) -> bool:
        return not self.entries


@dataclass(frozen=True, slots=True)
class PythonBoard:
    """Read-only Python mirror of Rust `Board`."""

    stones: tuple[tuple[AxialCoord, Player], ...]
    occupied: tuple[AxialCoord, ...]
    legal: tuple[AxialCoord, ...]
    windows: PythonWindowStore


@dataclass(frozen=True, slots=True)
class PythonHexoState:
    """Read-only Python mirror of Rust `HexoState`.

    Produced only by `api.to_python_state`. CAUTION: despite the annotation,
    `terminal` actually holds a `TerminalResult` at runtime (see the note on
    `PythonTerminal` above) — use `.winner`/`.reason`/`.metadata`.
    """

    board: PythonBoard
    current_player: Player
    phase: TurnPhase
    placements_made: int
    # Annotation is stale: runtime value is TerminalResult | None (api.py).
    terminal: PythonTerminal | None
    last_turn: PythonMoveRecord | None
    placement_history: tuple[PythonPlacementRecord, ...]
    first_stone: AxialCoord | None = None


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Result returned after the engine accepts and applies an action."""

    next_player: Player | None
    terminal: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TerminalResult:
    """Terminal state summary reported by the engine."""

    winner: Player | None
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
