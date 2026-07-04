"""Centralized adapter for the public hexo_engine API.

The single point where the runner touches `hexo_engine`: `loop.py` and the
modes call only through `HexoEngineAdapter`, never the engine module directly.
scripts/goal_benchmark.py also constructs this adapter for benchmarking.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

import hexo_engine as engine


class HexoEngineAdapter:
    """Small wrapper so the runner never scatters direct engine calls."""

    def metadata(self) -> Mapping[str, Any]:
        return _jsonable(engine.engine_metadata())

    def new_game(self, *, seed: int | None = None, scenario: object | None = None) -> engine.HexoState:
        # NOTE: hexo_engine currently discards both arguments
        # (hexo_engine/rust/src/pybridge.rs new_game: `let _ = seed/scenario`).
        # They are forwarded so the signature stays honest if the engine ever
        # honors them; the seed that matters is the one persisted in the .hxr
        # record header by loop.py.
        return engine.new_game(seed=seed, scenario=scenario)

    def clone_state(self, state: engine.HexoState) -> engine.HexoState:
        return engine.clone_state(state)

    def current_player(self, state: engine.HexoState) -> engine.Player:
        return engine.current_player(state)

    def player_index(self, player: engine.Player) -> int:
        if player == engine.Player.PLAYER_0:
            return 0
        if player == engine.Player.PLAYER_1:
            return 1
        raise ValueError(f"Unknown engine player: {player!r}")

    def player_role(self, player: engine.Player) -> str:
        return str(player)

    def apply_action(self, state: engine.HexoState, action: engine.Action) -> engine.TransitionResult:
        return engine.apply_action(state, action)

    def terminal(self, state: engine.HexoState) -> engine.TerminalResult | None:
        return engine.terminal(state)

    def action_id(self, action: engine.Action) -> int:
        return engine.action_id(action)

    def terminal_payload(self, terminal: object | None) -> Mapping[str, Any] | None:
        """Convert a TerminalResult into a JSON-able dict for GameResult.terminal.

        Shape: {"winner": str|None, "reason": str, "metadata": {...}} —
        loop.py's _terminal_winner/_terminal_placements read this shape.
        """
        if terminal is None:
            return None
        if isinstance(terminal, engine.TerminalResult):
            return {
                "winner": str(terminal.winner) if terminal.winner is not None else None,
                "reason": terminal.reason,
                "metadata": _jsonable(terminal.metadata),
            }
        return _jsonable(terminal)


def _jsonable(value: object) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)
