"""SealBot adapter for the generic Hexo runner.

Bridges the external C++ minimax baseline bot (a repo-external checkout,
located via $SEALBOT_PATH; see https://github.com/Ramora0/SealBot) into the
RunnerPlayer protocol. The bot runs in a dedicated subprocess (`_sealbot_worker.py` in
this directory) speaking newline-delimited JSON over stdin/stdout, because
the two SealBot variants ("current"/"best") export identical pybind module
names and cannot coexist in one Python process.

Consumers: the shrimp eval harness (packages/shrimp/python/shrimp/
eval_arena.py) and packages/hexo_frontend/python/hexo_frontend/web.py
(Arena opponent + the /api adapters endpoint via discover_sealbot_adapters).
Covered by tests/test_sealbot_adapter.py.
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Iterable, Mapping

import hexo_engine as engine

from ..player import DecisionResult, FinalSummary, GameContext, PlayerIdentity, TransitionEvent, WorkerContext


# --- Configuration and availability discovery -------------------------------

SEALBOT_VARIANTS = ("current", "best")
DEFAULT_SEALBOT_VARIANT = "current"
# Seconds of minimax think time per worker decide() call.
DEFAULT_SEALBOT_TIME_LIMIT = 0.05


class SealBotUnavailableError(RuntimeError):
    """Raised when the external SealBot checkout or worker is unavailable."""


@dataclass(frozen=True, slots=True)
class SealBotConfig:
    """Configuration for one SealBot runner player.

    `path` falls back to the SEALBOT_PATH env var. `time_limit` is the bot's
    per-decision think budget in seconds; `startup_timeout`/`response_timeout`
    are seconds the parent waits on the worker subprocess. `worker_script`
    overrides the bundled _sealbot_worker.py (used by tests to fake the bot).
    """

    path: str | Path | None = None
    variant: str = DEFAULT_SEALBOT_VARIANT
    time_limit: float = DEFAULT_SEALBOT_TIME_LIMIT
    startup_timeout: float = 8.0
    response_timeout: float = 30.0
    worker_script: str | Path | None = None

    @property
    def resolved_path(self) -> Path:
        raw = self.path or os.environ.get("SEALBOT_PATH")
        if not raw:
            raise SealBotUnavailableError("SealBot path is not configured. Set SEALBOT_PATH or pass --sealbot-path.")
        return Path(raw).expanduser().resolve()

    def validate(self) -> None:
        _validate_variant(self.variant)
        _validate_root(self.resolved_path)
        _validate_variant_files(self.resolved_path, self.variant)


# --- RunnerPlayer adapter ----------------------------------------------------


class SealBotPlayer:
    """RunnerPlayer wrapper around an external SealBot minimax variant.

    SealBot answers with a full two-stone turn at once; the runner asks for
    one action at a time, so the second stone is buffered in `_pending_moves`
    and replayed on the next decide() (tagged diagnostics["buffered_move"]).
    The worker subprocess is spawned lazily on the first decide(), not in
    setup_worker.
    """

    def __init__(self, config: SealBotConfig | None = None, *, player_id: str | None = None) -> None:
        self.config = config or SealBotConfig()
        _validate_variant(self.config.variant)
        label = f"SealBot {self.config.variant}"
        self.identity = PlayerIdentity(
            player_id=player_id or f"sealbot-{self.config.variant}",
            label=label,
            metadata={
                "adapter": "sealbot",
                "variant": self.config.variant,
                "time_limit": self.config.time_limit,
            },
        )
        self._worker: _SealBotProcess | None = None
        self._pending_moves: deque[tuple[int, int]] = deque()
        self._pending_diagnostics: dict[str, Any] = {}

    def setup_worker(self, context: WorkerContext) -> None:
        self.config.validate()

    def start_game(self, context: GameContext) -> None:
        self._pending_moves.clear()
        self._pending_diagnostics.clear()

    def decide(self, state: engine.HexoState) -> DecisionResult:
        """Return the next stone, querying the worker only when the buffer is empty.

        Raises ValueError if SealBot returns no moves or an illegal move (the
        buffer is cleared so the game aborts cleanly via the runner loop).
        """
        python_state = engine.to_python_state(state)
        if not self._pending_moves:
            response = self._worker_process().decide(_state_payload(python_state))
            moves = _parse_moves(response.get("moves"))
            if not moves:
                raise ValueError("SealBot returned no moves.")
            self._pending_moves.extend(moves)
            self._pending_diagnostics = dict(response.get("diagnostics") or {})
            diagnostics = dict(self._pending_diagnostics)
        else:
            diagnostics = dict(self._pending_diagnostics)
            diagnostics["buffered_move"] = True

        q, r = self._pending_moves.popleft()
        action = engine.PlacementAction(engine.AxialCoord(q=q, r=r))
        if not engine.is_legal_action(state, action):
            self._pending_moves.clear()
            self._pending_diagnostics.clear()
            raise ValueError(f"SealBot returned illegal move {q},{r}.")
        diagnostics.update(
            {
                "adapter": "sealbot",
                "variant": self.config.variant,
                "buffered_remaining": len(self._pending_moves),
            }
        )
        return DecisionResult(action=action, diagnostics=diagnostics)

    def observe_transition(self, transition: TransitionEvent) -> None:
        return

    def finish_game(self, final_summary: FinalSummary) -> None:
        self._pending_moves.clear()
        self._pending_diagnostics.clear()

    def close(self) -> None:
        self._pending_moves.clear()
        self._pending_diagnostics.clear()
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.close()

    def _worker_process(self) -> "_SealBotProcess":
        if self._worker is None:
            self._worker = _SealBotProcess(self.config)
        return self._worker


def discover_sealbot_adapters(path: str | Path | None = None) -> dict[str, Any]:
    """Return frontend-friendly SealBot availability metadata.

    Served by hexo_frontend/web.py at the /api adapters endpoint. Never
    raises: configuration/installation problems are reported in the payload's
    "error" fields and per-variant "available" flags.
    """

    raw_path = path or os.environ.get("SEALBOT_PATH")
    payload: dict[str, Any] = {
        "configured": bool(raw_path),
        "path": str(Path(raw_path).expanduser().resolve()) if raw_path else None,
        "default_variant": DEFAULT_SEALBOT_VARIANT,
        "variants": [],
    }
    if not raw_path:
        payload["error"] = "SealBot path is not configured. Set SEALBOT_PATH or pass --sealbot-path."
        payload["variants"] = [_variant_status(None, variant) for variant in SEALBOT_VARIANTS]
        return payload

    root = Path(raw_path).expanduser().resolve()
    root_error = _root_error(root)
    variants = []
    for variant in SEALBOT_VARIANTS:
        status = _variant_status(root, variant)
        if root_error is not None:
            status["available"] = False
            status["error"] = root_error
        variants.append(status)
    payload["variants"] = variants
    if root_error is not None:
        payload["error"] = root_error
    return payload


# --- Worker subprocess manager -----------------------------------------------


class _SealBotProcess:
    """Manages one _sealbot_worker.py subprocess over JSON lines.

    Protocol (strictly request-response, no request ids): parent writes one
    {"type": "decide", "state": ...} line to stdin and blocks on the response
    queue fed by a daemon stdout-reader thread; a second thread tails stderr
    into a bounded deque for error reporting. Construction blocks until the
    worker's ready handshake or raises SealBotUnavailableError.
    """

    def __init__(self, config: SealBotConfig) -> None:
        self.config = config
        self.root = config.resolved_path
        self.config.validate()
        self._responses: queue.Queue[dict[str, Any] | BaseException] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=40)
        worker_script = Path(config.worker_script) if config.worker_script else Path(__file__).with_name("_sealbot_worker.py")
        command = [
            sys.executable,
            str(worker_script),
            "--root",
            str(self.root),
            "--variant",
            config.variant,
            "--time-limit",
            str(float(config.time_limit)),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        ready = self._read_response(config.startup_timeout)
        if not ready.get("ok"):
            self.close()
            raise SealBotUnavailableError(_response_error(ready))

    def decide(self, state: Mapping[str, Any]) -> dict[str, Any]:
        self._send({"type": "decide", "state": dict(state)})
        response = self._read_response(self.config.response_timeout)
        if not response.get("ok"):
            raise RuntimeError(_response_error(response))
        return response

    def close(self) -> None:
        """Shut the worker down: polite close request, then terminate, then kill."""
        process = self._process
        if process.poll() is None:
            try:
                self._send({"type": "close"})
            except Exception:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    def _send(self, payload: Mapping[str, Any]) -> None:
        if self._process.poll() is not None:
            raise SealBotUnavailableError(f"SealBot worker exited with code {self._process.returncode}.")
        if self._process.stdin is None:
            raise SealBotUnavailableError("SealBot worker stdin is unavailable.")
        self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _read_response(self, timeout: float) -> dict[str, Any]:
        deadline = monotonic() + timeout
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                stderr = "\n".join(self._stderr)
                detail = f" Stderr:\n{stderr}" if stderr else ""
                raise TimeoutError(f"Timed out waiting for SealBot worker.{detail}")
            try:
                item = self._responses.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self._process.poll() is not None:
                    stderr = "\n".join(self._stderr)
                    detail = f" Stderr:\n{stderr}" if stderr else ""
                    raise SealBotUnavailableError(
                        f"SealBot worker exited with code {self._process.returncode}.{detail}"
                    )
                continue
            if isinstance(item, BaseException):
                raise SealBotUnavailableError(str(item)) from item
            return item

    def _read_stdout(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        try:
            for line in stream:
                if not line.strip():
                    continue
                self._responses.put(json.loads(line))
        except BaseException as exc:
            self._responses.put(exc)

    def _read_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return
        try:
            for line in stream:
                self._stderr.append(line.rstrip())
        except Exception:
            return


# --- Engine-state -> SealBot JSON payload helpers ----------------------------


def _state_payload(state: engine.PythonHexoState) -> dict[str, Any]:
    """Serialize the engine state mirror into the worker's JSON `state` shape.

    Field names/values here are the contract with _sealbot_worker.py's
    _Worker.decide (player0/player1 strings map to SealBot Player.A/B).
    """
    return {
        "current_player": str(state.current_player),
        "phase": str(state.phase),
        "moves_left_in_turn": _moves_left_in_turn(state.phase),
        "placements_made": state.placements_made,
        "terminal_winner": str(state.terminal.winner) if state.terminal else None,
        "stones": [
            {"q": coord.q, "r": coord.r, "player": str(player)}
            for coord, player in state.board.stones
        ],
    }


def _moves_left_in_turn(phase: engine.TurnPhase) -> int:
    # CAUTION: duplicates hexo_engine turn rules (opening stone and the
    # second stone of a turn = 1 placement left, otherwise 2). A rules change
    # in hexo_engine would silently desync the SealBot state payload.
    if phase == engine.TurnPhase.OPENING:
        return 1
    if phase == engine.TurnPhase.SECOND_STONE:
        return 1
    return 2


def _parse_moves(value: object) -> list[tuple[int, int]]:
    """Validate the worker's `moves` payload into [(q, r), ...] axial pairs."""
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        raise ValueError(f"SealBot returned malformed moves: {value!r}")
    moves: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, Iterable) or isinstance(item, (str, bytes, dict)):
            raise ValueError(f"SealBot returned malformed move: {item!r}")
        pair = list(item)
        if len(pair) != 2:
            raise ValueError(f"SealBot returned malformed move: {item!r}")
        moves.append((int(pair[0]), int(pair[1])))
    return moves


# --- Installation validation helpers ------------------------------------------


def _variant_status(root: Path | None, variant: str) -> dict[str, Any]:
    status: dict[str, Any] = {"id": variant, "label": f"SealBot {variant}", "available": False, "error": None}
    if root is None:
        status["error"] = "SealBot path is not configured."
        return status
    variant_dir = root / variant
    if not variant_dir.is_dir():
        status["error"] = f"Variant directory not found: {variant_dir}"
        return status
    if not _extension_candidates(variant_dir):
        status["error"] = f"Compiled minimax_cpp extension not found in {variant_dir}."
        return status
    status["available"] = True
    return status


def _validate_variant(variant: str) -> None:
    if variant not in SEALBOT_VARIANTS:
        raise ValueError(f"Unknown SealBot variant {variant!r}; expected one of {', '.join(SEALBOT_VARIANTS)}.")


def _validate_root(root: Path) -> None:
    error = _root_error(root)
    if error is not None:
        raise SealBotUnavailableError(error)


def _root_error(root: Path) -> str | None:
    if not root.exists():
        return f"SealBot path does not exist: {root}"
    if not (root / "game.py").is_file():
        return f"SealBot game.py not found under {root}"
    return None


def _validate_variant_files(root: Path, variant: str) -> None:
    status = _variant_status(root, variant)
    if not status["available"]:
        raise SealBotUnavailableError(str(status["error"]))


def _extension_candidates(variant_dir: Path) -> list[Path]:
    candidates = []
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidates.extend(variant_dir.glob(f"minimax_cpp*{suffix}"))
    return candidates


def _response_error(response: Mapping[str, Any]) -> str:
    error = response.get("error")
    if isinstance(error, Mapping):
        message = str(error.get("message") or error.get("traceback") or error)
        kind = error.get("type")
        return f"{kind}: {message}" if kind else message
    return str(error or response)
