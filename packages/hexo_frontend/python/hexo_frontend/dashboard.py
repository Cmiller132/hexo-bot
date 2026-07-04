"""Pure shaping layer: PythonHexoState mirror -> the browser board payload.

No engine calls and no I/O — input is the read-only ``hexo_engine``
``PythonHexoState`` mirror (stones, legal coords, placement history, window
entries) and output is the JSON-able dict ``static/app.js`` renders as the
board (placements/legal/winner plus a ``tactics`` block re-deriving threat /
immediate-win / must-block facts from the engine's 6-cell windows).

Callers (all in ``web.py``): the live-match payload
(``ManualMatchController._payload_locked``), the recorded-game replay endpoint
(``_training_history``), and the Debug position builder
(``_debug_build_position``). Nothing else imports this module.

Caveat: the client currently reads only ``tactics.threats`` (the Debug board's
threat overlay) — the full per-window masks/cells, ``immediate_wins`` and
``must_blocks`` are computed on every /api/state poll and replay but rendered
nowhere; keep that in mind before adding more derived facts here.

The window semantics (WIN_LENGTH=6, the three axis vectors, per-player bit
masks) mirror hexo_engine's rust/src/tactics.rs window store; this module must
agree with whatever the engine reports through PythonWindowEntry.
"""

from __future__ import annotations

from hexo_engine import AxialCoord, Player, PythonHexoState, PythonPlacementRecord, PythonWindowEntry, TurnPhase


WIN_LENGTH = 6
AXIS_VECTORS = {"Q": (1, 0), "R": (0, 1), "QR": (1, -1)}
WINDOW_MASK = (1 << WIN_LENGTH) - 1


def dashboard_state(state: PythonHexoState) -> dict[str, object]:
    """Translate a Python HexoState mirror into the browser dashboard shape.

    Returns the base board payload (current_player/phase/first_stone/winner/
    placements/legal/tactics); web.py callers update() it with their own
    screen-specific fields (version, players, turn_status, history/debug
    blocks). Coordinates are plain {"q","r"} dicts; players are the strings
    "player0"/"player1" (None for no winner)."""

    legal = [_coord_payload(coord) for coord in state.board.legal]
    legal_coords = {_coord_key(coord) for coord in legal}
    stone_owner = {_coord_key(_coord_payload(coord)): _player(player) for coord, player in state.board.stones}
    placements = [_placement(record) for record in state.placement_history]
    tactics = _dashboard_tactics(state.board.windows.entries, stone_owner, legal_coords)

    return {
        "current_player": _player(state.current_player),
        "phase": _phase(state.phase),
        "first_stone": _coord_payload(state.first_stone) if state.first_stone else None,
        "winner": _player(state.terminal.winner) if state.terminal else None,
        "terminal_reason": "six_in_line" if state.terminal else None,
        "placements": placements,
        "legal": legal,
        "legal_count": len(legal),
        "tactics": tactics,
    }


def _dashboard_tactics(
    entries: tuple[PythonWindowEntry, ...],
    stone_owner: dict[tuple[int, int], str | None],
    legal: set[tuple[int, int]],
) -> dict[str, object]:
    windows = [_window(entry, stone_owner, legal) for entry in entries]
    threats = [window for window in windows if window["is_threat"]]
    winning_windows = [window for window in windows if window["is_win"]]
    immediate_wins = _move_facts(windows, legal, want_win=True)
    must_blocks = _must_blocks(windows, legal)

    return {
        "windows": windows,
        "window_count": len(windows),
        "threats": threats,
        "threat_count": len(threats),
        "winning_windows": winning_windows,
        "immediate_wins": immediate_wins,
        "must_blocks": must_blocks,
        "summary": {
            "active": sum(1 for window in windows if window["is_active"]),
            "blocked": sum(1 for window in windows if window["is_blocked"]),
            "threats": len(threats),
            "wins": len(winning_windows),
            "immediate_wins": len(immediate_wins),
            "must_blocks": len(must_blocks),
        },
    }


def _window(
    entry: PythonWindowEntry,
    stone_owner: dict[tuple[int, int], str | None],
    legal: set[tuple[int, int]],
) -> dict[str, object]:
    start = _coord_payload(entry.key.start)
    axis = entry.key.axis
    p0_mask = int(entry.masks[0])
    p1_mask = int(entry.masks[1])
    p0_count = p0_mask.bit_count()
    p1_count = p1_mask.bit_count()
    active_player = _active_player(p0_count, p1_count)
    threat_player = active_player if active_player and max(p0_count, p1_count) >= 4 else None
    own_count = max(p0_count, p1_count)
    cells = [_add(start, AXIS_VECTORS[axis], index) for index in range(WIN_LENGTH)]
    empty = [cell for cell in cells if _coord_key(cell) not in stone_owner]
    blockable = [cell for cell in empty if _coord_key(cell) in legal]

    return {
        "id": _window_id(start, axis),
        "key": {"start": start, "axis": axis},
        "axis": axis,
        "cells": [
            {"q": cell["q"], "r": cell["r"], "owner": stone_owner.get(_coord_key(cell)), "index": index}
            for index, cell in enumerate(cells)
        ],
        "mask": {
            "player0": p0_mask,
            "player1": p1_mask,
            "occupied": p0_mask | p1_mask,
            "empty": (~(p0_mask | p1_mask)) & WINDOW_MASK,
        },
        "counts": {
            "player0": p0_count,
            "player1": p1_count,
            "empty": len(empty),
            "occupied": p0_count + p1_count,
        },
        "active_player": active_player,
        "threat_player": threat_player,
        "player": threat_player or active_player,
        "own_count": own_count,
        "is_active": active_player is not None,
        "is_blocked": p0_count > 0 and p1_count > 0,
        "is_threat": threat_player is not None,
        "is_win": active_player is not None and own_count >= WIN_LENGTH,
        "severity": "win" if own_count >= WIN_LENGTH else "direct" if own_count == 5 else "threat" if own_count >= 4 else "active",
        "stone_cells": {
            "player0": [_mask_cell(cells, index) for index in range(WIN_LENGTH) if p0_mask & (1 << index)],
            "player1": [_mask_cell(cells, index) for index in range(WIN_LENGTH) if p1_mask & (1 << index)],
        },
        "empty_cells": empty,
        "blockable_cells": blockable,
        "blockable_now": bool(blockable),
    }


def _move_facts(windows: list[dict[str, object]], legal: set[tuple[int, int]], *, want_win: bool) -> list[dict[str, object]]:
    """Legal cells that complete a window for its active player: with
    ``want_win`` a 6th stone (immediate win); without, a 5th (threat builder).
    One fact per (player, cell), carrying every window id it completes."""

    facts: dict[tuple[str, tuple[int, int]], set[str]] = {}
    target = WIN_LENGTH if want_win else WIN_LENGTH - 1
    for window in windows:
        if window["is_blocked"]:
            continue
        player = str(window["active_player"] or "")
        if not player:
            continue
        own_count = int(window["own_count"])
        for empty in window["empty_cells"]:
            coord = _coord(empty)
            key = _coord_key(coord)
            if key in legal and own_count + 1 >= target:
                facts.setdefault((player, key), set()).add(str(window["id"]))
    return [
        {"player": player, "q": coord[0], "r": coord[1], "window_ids": sorted(window_ids)}
        for (player, coord), window_ids in sorted(facts.items())
    ]


def _must_blocks(windows: list[dict[str, object]], legal: set[tuple[int, int]]) -> list[dict[str, object]]:
    """Cells the OPPONENT must take: legal empties of any unblocked 5-stone
    window, attributed to the blocking player (the one who must respond)."""

    blocks: dict[tuple[str, tuple[int, int]], set[str]] = {}
    for window in windows:
        if int(window["own_count"]) != 5 or not window["active_player"]:
            continue
        blocker = "player1" if window["active_player"] == "player0" else "player0"
        for empty in window["empty_cells"]:
            coord = _coord(empty)
            key = _coord_key(coord)
            if key in legal:
                blocks.setdefault((blocker, key), set()).add(str(window["id"]))
    return [
        {"player": player, "q": coord[0], "r": coord[1], "window_ids": sorted(window_ids)}
        for (player, coord), window_ids in sorted(blocks.items())
    ]


def _placement(record: PythonPlacementRecord) -> dict[str, object]:
    coord = _coord_payload(record.coord)
    return {
        "q": coord["q"],
        "r": coord["r"],
        "player": _player(record.player),
        "phase": _phase(record.phase),
        "index": record.placement_index,
    }


def _coord(value: object) -> dict[str, int]:
    if isinstance(value, dict):
        return {"q": int(value.get("q", 0)), "r": int(value.get("r", 0))}
    if isinstance(value, AxialCoord):
        return _coord_payload(value)
    return {"q": 0, "r": 0}


def _coord_payload(coord: AxialCoord) -> dict[str, int]:
    return {"q": coord.q, "r": coord.r}


def _coord_key(coord: dict[str, int]) -> tuple[int, int]:
    return (coord["q"], coord["r"])


def _add(coord: dict[str, int], vector: tuple[int, int], scale: int) -> dict[str, int]:
    return {"q": coord["q"] + vector[0] * scale, "r": coord["r"] + vector[1] * scale}


def _mask_cell(cells: list[dict[str, int]], index: int) -> dict[str, int]:
    return cells[index]


def _window_id(start: dict[str, int], axis: str) -> str:
    return f"{axis}:{start['q']},{start['r']}"


def _player(value: object) -> str | None:
    if value == Player.PLAYER_0 or str(value) in {"player0", "Player0"}:
        return "player0"
    if value == Player.PLAYER_1 or str(value) in {"player1", "Player1"}:
        return "player1"
    return None


def _phase(value: object) -> str:
    if value == TurnPhase.OPENING or str(value) == "Opening":
        return "opening"
    if value == TurnPhase.SECOND_STONE or str(value) == "SecondStone":
        return "second_stone"
    return "first_stone"


def _active_player(p0_count: int, p1_count: int) -> str | None:
    if p0_count and not p1_count:
        return "player0"
    if p1_count and not p0_count:
        return "player1"
    return None
