"""Line-oriented subprocess worker for one SealBot pybind variant.

The two SealBot variants export the same pybind module and class names, so they
cannot both be imported in one Python process. The parent adapter keeps variant
selection safe by running exactly one variant inside this worker process.

Spawned only by _SealBotProcess in hexo_runner/adapters/sealbot.py (never run
by hand). Protocol: emits one ready/error JSON line on startup, then answers
each stdin request line with exactly one stdout JSON line —
{"type": "decide", "state": ...} -> {"ok": true, "moves": [[q, r], ...],
"diagnostics": {...}}; {"type": "close"} ends the loop. stdout must carry
protocol JSON only. Imports game.py + the compiled minimax_cpp extension from
the SealBot checkout passed via --root/--variant.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one SealBot variant as a JSON-line worker.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--variant", required=True, choices=("current", "best"))
    parser.add_argument("--time-limit", required=True, type=float)
    args = parser.parse_args(argv)

    try:
        worker = _Worker(Path(args.root), args.variant, args.time_limit)
    except BaseException as exc:
        _write({"ok": False, "error": _error_payload(exc)})
        return 1

    _write({"ok": True, "type": "ready", "variant": args.variant})

    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            if request.get("type") == "close":
                _write({"ok": True, "type": "closed"})
                return 0
            if request.get("type") != "decide":
                raise ValueError(f"unknown request type: {request.get('type')!r}")
            _write({"ok": True, **worker.decide(request["state"])})
        except BaseException as exc:
            _write({"ok": False, "error": _error_payload(exc)})
    return 0


class _Worker:
    """Holds the imported SealBot variant and one MinimaxBot instance.

    `time_limit` is seconds of think time per get_move call.
    """

    def __init__(self, root: Path, variant: str, time_limit: float) -> None:
        self.root = root.resolve()
        self.variant = variant
        self.variant_dir = (self.root / variant).resolve()
        if not (self.root / "game.py").is_file():
            raise FileNotFoundError(f"SealBot game.py not found under {self.root}")
        if not self.variant_dir.is_dir():
            raise FileNotFoundError(f"SealBot variant directory not found: {self.variant_dir}")

        sys.path.insert(0, str(self.variant_dir))
        sys.path.insert(0, str(self.root))

        from game import HexGame, Player  # type: ignore[import-not-found]
        import minimax_cpp  # type: ignore[import-not-found]

        self._hex_game_type = HexGame
        self._player_type = Player
        self._bot = minimax_cpp.MinimaxBot(float(time_limit))

    def decide(self, state: dict[str, Any]) -> dict[str, Any]:
        """Rebuild a fresh HexGame from the JSON state and ask the bot for a turn.

        The state shape is produced by sealbot.py _state_payload; runner
        "player0"/"player1" map to SealBot Player.A/B. Returns the bot's full
        turn (1-2 moves) — the parent adapter buffers the second stone.
        """
        game = self._hex_game_type()
        player_a = self._player_type.A
        player_b = self._player_type.B
        game.board = {
            (int(stone["q"]), int(stone["r"])): player_a if stone["player"] == "player0" else player_b
            for stone in state.get("stones", ())
        }
        game.current_player = player_a if state.get("current_player") == "player0" else player_b
        game.moves_left_in_turn = int(state.get("moves_left_in_turn", 1))
        game.move_count = int(state.get("placements_made", len(game.board)))
        terminal_winner = state.get("terminal_winner")
        game.game_over = terminal_winner is not None
        game.winner = None if terminal_winner is None else player_a if terminal_winner == "player0" else player_b
        game.winning_cells = []

        moves = self._bot.get_move(game)
        normalized = [[int(q), int(r)] for q, r in moves]
        return {"moves": normalized, "diagnostics": self._diagnostics()}

    def _diagnostics(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "variant": self.variant,
            "pair_moves": _jsonable(getattr(self._bot, "pair_moves", None)),
            "last_depth": _jsonable(getattr(self._bot, "last_depth", None)),
            "last_score": _jsonable(getattr(self._bot, "last_score", None)),
            "nodes": _jsonable(getattr(self._bot, "_nodes", None)),
            "max_depth": _jsonable(getattr(self._bot, "max_depth", None)),
            "time_limit": _jsonable(getattr(self._bot, "time_limit", None)),
        }
        extract_pv = getattr(self._bot, "extract_pv", None)
        if callable(extract_pv):
            try:
                diagnostics["principal_variation"] = _jsonable(extract_pv())
            except Exception as exc:
                diagnostics["principal_variation_error"] = str(exc)
        return diagnostics


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
