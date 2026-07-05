"""Stdlib HTTP server for the Hexo training dashboard (three SPA screens).

This is the whole backend of the dashboard: a ``ThreadingHTTPServer`` (no
framework) serving the static bundle in ``static/`` (index.html + app.js +
styles.css) plus the JSON API behind its three screens:

* **Match** (``#match``) — live Arena games through the production runner.
  ``ManualMatchController`` bridges browser clicks to ``hexo_runner`` players
  (manual / SealBot / checkpoint bots, the latter playing via the Debug CPU
  worker) and runs ``hexo_runner.modes.match.run_match`` on a daemon thread.
* **History** (``#history``) — training-run status read directly from run
  directories under ``cwd/runs`` (production cwd is the run mount, NOT this
  repo): ``manifest.json``, ``diagnostics/*.json`` + ``events.jsonl`` written
  by ``hexo_train``/the model packages, ``.hxr`` game records via
  ``hexo_runner.records.HexoRecordFile``, and ``checkpoints/*.pt`` stats.
  Two poll tiers: the full scan (3s memo cache) and ``/api/training/live``
  (~2.5s client poll, 1s micro-cache).
* **Debug** (``#debug``) — single-position model forensics. This module only
  resolves runs/records/checkpoints and shapes payloads; all torch work is
  delegated to the out-of-process CPU worker via ``debug_service`` (this
  process never imports torch, so the GPU and the live poll stay untouched).

Layout (section banners below follow this order): response caches -> Match
controller + player adapters -> HTTP handler/route table -> training-run scan
-> Debug endpoint glue -> artifact/history paging -> per-epoch trends ->
live status -> small file helpers -> server entry points.

Callers: ``python -m hexo_frontend.web`` / the ``hexo-play`` console script
(production: ``scripts/_dashboard_launch.sh``, WSL :8080, cwd at the run
mount); tests call the module functions directly
(tests/test_frontend_training_{artifacts,epoch,live}.py, test_debug_infer.py,
test_hexo_runner_match_mode.py, test_sealbot_adapter.py). Board payload
shaping lives in ``dashboard.py``; the client contract is ``static/app.js``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import statistics
import tempfile
import zlib
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import formatdate
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Condition, Lock, RLock, Thread
from time import monotonic, perf_counter, time as wall_clock
from typing import Any, ClassVar
from urllib.parse import parse_qs, unquote, urlparse

import hexo_engine as engine
from hexo_runner.adapters.sealbot import (
    DEFAULT_SEALBOT_TIME_LIMIT,
    SealBotConfig,
    SealBotPlayer,
    discover_sealbot_adapters,
)
from hexo_runner.modes.match import run_match
from hexo_runner.player import DecisionResult, FinalSummary, PlayerIdentity, TransitionEvent, WorkerContext, GameContext
from hexo_runner.records import GameResult, HexoRecordFile
from hexo_runner.session import GameSpec
from hexo_engine.types import pack_coord_id, unpack_coord_id

from .dashboard import dashboard_state
from . import debug_service


STATIC_ROOT = files("hexo_frontend").joinpath("static")
STATIC_TYPES = {
    "css": "text/css; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "js": "text/javascript; charset=utf-8",
}
ARTIFACT_TYPES = {
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/x-ndjson; charset=utf-8",
    ".png": "image/png",
    ".hxr": "application/octet-stream",
}
ARTIFACT_SUFFIXES = frozenset(ARTIFACT_TYPES)
TRAINING_SCAN_EXCLUDED_DIRS = frozenset({"archive", "quarantine", "__pycache__"})
HISTORY_ALL_RUNS = "__all__"
TRAINING_OVERVIEW_HISTORY_LIMIT = 50
TRAINING_OVERVIEW_ARTIFACT_LIMIT = 50
HISTORY_PAGE_DEFAULT_LIMIT = 400
HISTORY_PAGE_MAX_LIMIT = 500
ARTIFACT_PAGE_DEFAULT_LIMIT = 50
ARTIFACT_PAGE_MAX_LIMIT = 200
BotFactory = Callable[[str, float], object]
CheckpointFactory = Callable[[dict], object]
PLAYER_ROLES = ("player0", "player1")
MANUAL_KIND = "manual"
SEALBOT_PREFIX = "sealbot-"
SERIES_MAX_GAMES = 25
CHECKPOINT_VISITS_MIN = 8
CHECKPOINT_VISITS_MAX = 2048
CHECKPOINT_VISITS_DEFAULT = 256
CHECKPOINT_C_PUCT_MIN = 0.1
CHECKPOINT_C_PUCT_MAX = 10.0
CHECKPOINT_C_PUCT_DEFAULT = 1.5
# Eval-protocol move selection for checkpoint bots (mirrors the trainer's
# [model.config.evaluation] opening_temperature/opening_moves arena defaults):
# the first N plies are sampled from the visit distribution at this temperature
# so repeated games diverge; afterwards play is the strict visit argmax.
CHECKPOINT_OPENING_MOVES = 8
CHECKPOINT_OPENING_TEMPERATURE = 0.6
# In-search opening temperature for the shrimp lineage, whose selection
# happens inside the search (as-trained gumbel profile). 1.0 = the
# multistage-eval arena's opening_temperature, so match-screen games sample
# openings exactly like eval games. The 0.6 above remains the client-side
# visit-sampling knob for the fallback path (a search that does not select
# in-search).
CHECKPOINT_OPENING_TEMPERATURE_IN_SEARCH = 1.0

# --- Transfer efficiency (gzip + caching) -----------------------------------
# The dashboard is often viewed over LAN/VPN, where uncompressed payloads and
# per-request connections are slow. Responses below this size aren't worth
# gzipping (the ~20-byte gzip header/overhead dominates).
GZIP_MIN_BYTES = 600
# The training-run/list scans walk the run tree and open many .hxr files. The
# history screen re-polls every 15s, so memoize the built payload briefly to bound
# that work to at most once per interval regardless of poll/client count.
TRAINING_CACHE_TTL_SECONDS = 3.0
# /api/training/live is the History screen's fast poll tier (~2.5s per client).
# Its payload is only a handful of json/stat reads, but multiple open tabs would
# multiply them; a ~1s micro-cache bounds the disk work to once per second per
# run while staying far fresher than the 3s full-payload cache.
TRAINING_LIVE_CACHE_TTL_SECONDS = 1.0
# /api/training/epochs scans the run's diagnostics dir (a few hundred small
# json files) once per request and merges the selfplay/select/training/eval
# telemetry into one row per epoch. The game-history epoch strip re-polls with
# the 15s screen refresh; cache the built payload by the diagnostics dir's
# newest mtime so an unchanged run costs one dict lookup and a churning run
# (a fresh epoch file just landed) rebuilds immediately.
TRAINING_EPOCHS_CACHE_TTL_SECONDS = 3.0

_static_lock = Lock()
# name -> (mtime, (raw_bytes, gzipped_bytes, etag, last_modified, content_type))
_static_cache: dict[str, tuple[float, tuple[bytes, bytes, str, str, str]]] = {}
_training_cache_lock = Lock()
_training_run_cache: dict[str, tuple[float, dict[str, object]]] = {}
_training_runs_cache: list[tuple[float, dict[str, object]]] = []
# run name -> (monotonic, payload) for /api/training/live (micro-cache, ~1s TTL)
_training_live_cache: dict[str, tuple[float, dict[str, object]]] = {}
# run name -> (diagnostics-dir mtime, payload) for /api/training/epochs. Keyed on
# the directory mtime (not a clock TTL) so it invalidates the instant a new
# per-epoch file lands and is otherwise a free rescan-skip.
_training_epochs_cache: dict[str, tuple[float, dict[str, object]]] = {}
_hxr_history_cache: dict[str, tuple[int, int, list[dict[str, object]]]] = {}
_hxr_count_cache: dict[str, tuple[int, int, int]] = {}


def _strong_etag(data: bytes) -> str:
    return '"' + hashlib.sha1(data).hexdigest()[:20] + '"'


def _static_entry(name: str) -> tuple[bytes, bytes, str, str, str] | None:
    """Return (raw, gzipped, etag, last_modified, content_type) for a static asset.

    Memoized by (name, mtime): the file is read, gzipped, and hashed at most once
    per version, so serving it is just a dict lookup + socket write. Returns None
    when the asset is missing.
    """

    resource = STATIC_ROOT.joinpath(name)
    mtime: float | None
    try:
        mtime = Path(str(resource)).stat().st_mtime
    except (OSError, TypeError, ValueError):
        mtime = None
    if mtime is not None:
        with _static_lock:
            hit = _static_cache.get(name)
            if hit is not None and hit[0] == mtime:
                return hit[1]
    try:
        raw = resource.read_bytes()
    except (FileNotFoundError, IsADirectoryError, OSError):
        return None
    extension = name.rsplit(".", 1)[-1]
    content_type = STATIC_TYPES.get(extension, "application/octet-stream")
    entry = (
        raw,
        gzip.compress(raw, 6),
        _strong_etag(raw),
        formatdate(mtime, usegmt=True) if mtime else formatdate(usegmt=True),
        content_type,
    )
    if mtime is not None:
        with _static_lock:
            _static_cache[name] = (mtime, entry)
    return entry


def _training_run_cached(name: str) -> dict[str, object]:
    now = monotonic()
    with _training_cache_lock:
        hit = _training_run_cache.get(name)
        if hit is not None and now - hit[0] < TRAINING_CACHE_TTL_SECONDS:
            return hit[1]
    payload = _training_run(name)  # heavy scan, kept outside the lock
    with _training_cache_lock:
        _training_run_cache[name] = (monotonic(), payload)
    return payload


def _training_runs_cached() -> dict[str, object]:
    now = monotonic()
    with _training_cache_lock:
        if _training_runs_cache and now - _training_runs_cache[0][0] < TRAINING_CACHE_TTL_SECONDS:
            return _training_runs_cache[0][1]
    payload = _training_runs()
    with _training_cache_lock:
        _training_runs_cache[:] = [(monotonic(), payload)]
    return payload


def _training_live_cached(name: str) -> dict[str, object]:
    """Near-realtime live-status payload for ``GET /api/training/live?run=``.

    Returns ``{"run", "status" (= _training_live_status(run_dir)), "ts"}``.
    _training_live_status only does cheap reads (events.jsonl tail, the watchdog
    jsonl tail, two small json files, and a couple of stat/glob probes) with no
    side effects, but every open History tab polls this every ~2.5s — so the
    built payload is micro-cached per run name for ~1s (module-level
    ``_training_live_cache``, same (monotonic, payload) shape as
    ``_training_runs_cached``) so N clients cost one disk pass per second, not N.
    Unknown runs raise ValueError (-> 400 ``{"error": ...}``) and are never
    cached."""

    now = monotonic()
    with _training_cache_lock:
        hit = _training_live_cache.get(name)
        if hit is not None and now - hit[0] < TRAINING_LIVE_CACHE_TTL_SECONDS:
            return hit[1]
    run_dir = _resolve_run_dir(name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    payload: dict[str, object] = {
        "run": run_dir.name,
        "status": _training_live_status(run_dir),
        "ts": wall_clock(),
    }
    with _training_cache_lock:
        _training_live_cache[name] = (monotonic(), payload)
    return payload


def _training_epochs_cached(name: str) -> dict[str, object]:
    """Per-epoch telemetry payload for ``GET /api/training/epochs?run=`` (the
    game-history epoch strip / inspector detail).

    Cached by the diagnostics dir's newest mtime rather than a clock TTL: an
    unchanged run replays the built payload without touching disk, and a run
    that just wrote a new epoch file rebuilds on the very next request. Unknown
    runs raise ValueError (-> 400 ``{"error": ...}``) and are never cached."""

    run_dir = _resolve_run_dir(name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    stat = _safe_stat(run_dir / "diagnostics")
    mtime = stat.st_mtime if stat is not None else 0.0
    with _training_cache_lock:
        hit = _training_epochs_cache.get(run_dir.name)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    payload = _training_epochs(run_dir)  # heavy scan, kept outside the lock
    with _training_cache_lock:
        _training_epochs_cache[run_dir.name] = (mtime, payload)
    return payload


# ---------------------------------------------------------------------------
# Match screen: ManualMatchController + runner player adapters.
#
# One controller instance per server (bound onto the handler class). It owns a
# daemon match thread running hexo_runner.modes.match.run_match and exposes a
# version-counter long-poll (state) the browser drives. Generation bumping is
# the cancellation mechanism: every player callback raises "manual match reset"
# once the generation moves on, so an abandoned thread dies at its next call.
# ---------------------------------------------------------------------------


class MoveConflict(ValueError):
    """Raised when a browser move arrives while the human cannot act."""


class ManualMatchController:
    """Frontend-owned bridge between HTTP clicks and generic runner players."""

    def __init__(
        self,
        *,
        sealbot_path: str | Path | None = None,
        bot_factory: BotFactory | None = None,
        checkpoint_factory: CheckpointFactory | None = None,
    ) -> None:
        self._condition = Condition(RLock())
        self._sealbot_path = Path(sealbot_path).expanduser().resolve() if sealbot_path else None
        self._bot_factory = bot_factory
        self._checkpoint_factory = checkpoint_factory
        self._thread: Thread | None = None
        self._game_number = 0
        self._generation = 0
        self._cancelled = False
        self._stopped = False
        self._state: engine.HexoState | None = None
        self._python_state: engine.PythonHexoState | None = None
        self._pending_action: engine.Action | None = None
        self._version = 0
        self._result: GameResult | None = None
        self._error: BaseException | None = None
        self._mode = "manual"
        self._game_id = "manual-0-g1"
        self._player_setup: dict[str, dict[str, object]] = {
            "player0": {"kind": MANUAL_KIND},
            "player1": {"kind": MANUAL_KIND},
        }
        self._slot_specs: dict[str, dict[str, object]] = {
            "slot0": {"kind": MANUAL_KIND},
            "slot1": {"kind": MANUAL_KIND},
        }
        self._seat_slots: dict[str, str] = {"player0": "slot0", "player1": "slot1"}
        self._series_games = 1
        self._series_alternate = False
        self._series_current_game = 1
        self._series_finished = False
        self._series_tally: dict[str, int] = {"slot0": 0, "slot1": 0, "draws": 0}
        self._series_results: list[dict[str, object]] = []
        self._bot_time_limit = DEFAULT_SEALBOT_TIME_LIMIT
        self._seed: int | None = None
        self._thinking_player: str | None = None
        self._last_bot_decision: dict[str, object] | None = None
        self._decision_log: list[dict[str, object]] = []
        self._observed_transition: tuple[str, int] | None = None
        self.reset()

    def reset(self, config: dict[str, Any] | None = None) -> dict[str, object]:
        """Start a new match/series from a ``POST /api/new`` body.

        Tears down any running match (generation bump), parses + validates the
        config (bad specs raise ValueError -> 400 BEFORE any thread starts),
        spawns the series thread, blocks until the first state arrives, and
        returns the full state payload (the same shape as ``state()``)."""

        match = self._parse_match_config(config or {})
        self.close()
        with self._condition:
            self._generation += 1
            self._game_number += 1
            self._mode = match["mode"]
            self._slot_specs = {
                "slot0": dict(match["players"]["player0"]),
                "slot1": dict(match["players"]["player1"]),
            }
            self._player_setup = {
                "player0": dict(match["players"]["player0"]),
                "player1": dict(match["players"]["player1"]),
            }
            self._seat_slots = {"player0": "slot0", "player1": "slot1"}
            self._series_games = match["series_games"]
            self._series_alternate = match["series_alternate"]
            self._series_current_game = 1
            self._series_finished = False
            self._series_tally = {"slot0": 0, "slot1": 0, "draws": 0}
            self._series_results = []
            self._bot_time_limit = match["time_limit"]
            self._seed = match["seed"]
            self._game_id = f"{self._mode}-{self._game_number}-g1"
            self._cancelled = False
            self._stopped = False
            self._state = None
            self._python_state = None
            self._pending_action = None
            self._version = 0
            self._result = None
            self._error = None
            self._thinking_player = None
            self._last_bot_decision = None
            self._decision_log = []
            self._observed_transition = None
            self._thread = Thread(target=self._run_series, args=(self._generation,), daemon=True)
            self._thread.start()
            self._wait_for_state_locked()
            return self._payload_locked()

    def stop(self) -> dict[str, object]:
        """Halt the running match/series without joining the match thread.

        A checkpoint search may sit inside the worker for minutes; the thread is a
        daemon and exits at its next controller callback (the generation bump turns
        every later callback into a "manual match reset" error it swallows)."""

        with self._condition:
            try:
                self._wait_for_state_locked()
            except RuntimeError:
                # Stop must succeed even when startup never produced a state
                # (errored/timed-out launch is exactly when a user hits Stop).
                pass
            self._cancelled = True
            self._stopped = True
            self._generation += 1
            # The orphaned thread's finished/failed callbacks raise before they
            # could clear this, so clear it here or it pulses forever.
            self._thinking_player = None
            self._version += 1
            self._condition.notify_all()
            if self._python_state is None:
                return {
                    "stopped": True,
                    "version": self._version,
                    "turn_status": "stopped",
                    "error": self._error_message_locked(),
                }
            return self._payload_locked()

    def adapters(self) -> dict[str, object]:
        return {"sealbot": discover_sealbot_adapters(self._sealbot_path)}

    def state(self, *, since: int | None = None, timeout_ms: int = 0) -> dict[str, object]:
        """Current match payload for ``GET /api/state``.

        With ``since`` + ``timeout_ms`` this is a long-poll: it parks (up to
        30s max) until the internal version counter passes ``since``, so the
        browser sees bot moves the instant they land instead of on a poll
        interval. Returns immediately when the state is already newer."""

        with self._condition:
            self._wait_for_state_locked()
            if since is not None and self._version <= since and timeout_ms > 0:
                deadline = monotonic() + max(0.0, min(timeout_ms, 30000) / 1000.0)
                while self._version <= since and self._error is None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
            return self._payload_locked()

    def submit_move(self, q: int, r: int) -> dict[str, object]:
        """Apply a human click (``POST /api/move``) as the pending action.

        Raises MoveConflict (-> 409) when it is not the human's turn and
        ValueError (-> 400) for an illegal coordinate; otherwise hands the
        action to the parked ``decide`` callback and blocks until the match
        thread has advanced the state."""

        with self._condition:
            self._wait_for_state_locked()
            state = self._state
            if state is None or self._result is not None:
                raise MoveConflict("No move is currently pending.")
            if not self._can_submit_locked():
                raise MoveConflict("It is not the human player's turn.")
            action = engine.PlacementAction(engine.AxialCoord(q=q, r=r))
            if not engine.is_legal_action(state, action):
                raise ValueError(f"{q},{r} is not legal.")

            start_version = self._version
            self._pending_action = action
            self._condition.notify_all()
            while self._version == start_version and self._error is None and self._result is None:
                self._condition.wait(timeout=0.25)
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            return self._payload_locked()

    def close(self) -> None:
        thread = self._thread
        if thread is None:
            return
        with self._condition:
            self._cancelled = True
            self._condition.notify_all()
        thread.join(timeout=5.0)
        # On timeout the thread is most likely blocked inside a long checkpoint
        # search on the worker. Abandon it: it is a daemon, and the generation
        # guard makes every later callback from it a harmless no-op error.
        self._thread = None

    def decide(self, player_index: int, state: engine.HexoState, generation: int) -> DecisionResult:
        """Match-thread side of a manual turn (called by _ManualPlayer.decide):
        publish the pre-move state, then park until submit_move provides the
        pending action or the generation moves on (-> "manual match reset")."""

        with self._condition:
            if self._cancelled or generation != self._generation:
                raise RuntimeError("manual match reset")
            self._set_state_locked(state)
            self._version += 1
            self._condition.notify_all()

            while self._pending_action is None and not self._cancelled and generation == self._generation:
                self._condition.wait()
            if self._cancelled or generation != self._generation:
                raise RuntimeError("manual match reset")

            action = self._pending_action
            self._pending_action = None
            return DecisionResult(action=action, diagnostics={"manual_player": player_index})

    def bot_decision_started(self, player_index: int, state: engine.HexoState, generation: int) -> None:
        with self._condition:
            if self._cancelled or generation != self._generation:
                raise RuntimeError("manual match reset")
            self._set_state_locked(state)
            self._thinking_player = _player_role(player_index)
            self._version += 1
            self._condition.notify_all()

    def bot_decision_finished(
        self, player_index: int, result: DecisionResult, duration_ms: float, generation: int
    ) -> None:
        action = result.action
        role = _player_role(player_index)
        diagnostics = dict(result.diagnostics)
        payload: dict[str, object] = {
            "player": role,
            "duration_ms": round(duration_ms, 3),
            "diagnostics": diagnostics,
        }
        if isinstance(action, engine.PlacementAction):
            payload.update({"q": action.coord.q, "r": action.coord.r})
        with self._condition:
            if generation != self._generation:
                raise RuntimeError("manual match reset")
            # ply = placements BEFORE the move (the state was last set by
            # bot_decision_started, i.e. pre-move), so this is the move index.
            entry: dict[str, object] = {
                "ply": len(self._python_state.placement_history) if self._python_state is not None else None,
                "player": role,
                "duration_ms": round(duration_ms, 3),
                "value": diagnostics.get("root_value"),
                "visits": diagnostics.get("visits"),
                "kind": (self._player_setup.get(role) or {}).get("kind"),
            }
            if isinstance(action, engine.PlacementAction):
                entry.update({"q": action.coord.q, "r": action.coord.r})
            self._decision_log.append(entry)
            self._thinking_player = None
            self._last_bot_decision = payload
            self._version += 1
            self._condition.notify_all()

    def bot_decision_failed(
        self, player_index: int, exc: BaseException, duration_ms: float, generation: int
    ) -> None:
        role = _player_role(player_index)
        with self._condition:
            if generation != self._generation:
                raise RuntimeError("manual match reset")
            self._decision_log.append(
                {
                    "ply": len(self._python_state.placement_history) if self._python_state is not None else None,
                    "player": role,
                    "error": f"{type(exc).__name__}: {exc}",
                    "kind": (self._player_setup.get(role) or {}).get("kind"),
                }
            )
            self._thinking_player = None
            self._last_bot_decision = {
                "player": role,
                "duration_ms": round(duration_ms, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._version += 1
            self._condition.notify_all()

    def observe_transition(self, transition: TransitionEvent, generation: int) -> None:
        with self._condition:
            if generation != self._generation:
                raise RuntimeError("manual match reset")
            key = (transition.game_id, transition.action_index)
            if self._observed_transition == key:
                return
            self._observed_transition = key
            self._set_state_locked(transition.state)
            self._version += 1
            self._condition.notify_all()

    def _run_series(self, generation: int) -> None:
        try:
            for game_index in range(self._series_games):
                swapped = self._series_alternate and game_index % 2 == 1
                seat_slots = {
                    "player0": "slot1" if swapped else "slot0",
                    "player1": "slot0" if swapped else "slot1",
                }
                with self._condition:
                    if self._generation != generation:
                        return
                    self._seat_slots = dict(seat_slots)
                    self._player_setup = {
                        role: dict(self._slot_specs[slot]) for role, slot in seat_slots.items()
                    }
                    self._series_current_game = game_index + 1
                    self._game_id = f"{self._mode}-{self._game_number}-g{game_index + 1}"
                    # Per-game fields only; the finished board state stays visible
                    # until the next game's first decide/observe replaces it.
                    self._result = None
                    self._pending_action = None
                    self._thinking_player = None
                    self._last_bot_decision = None
                    self._decision_log = []
                    self._observed_transition = None
                    players = self._players_for_match(generation)
                    spec = GameSpec(
                        game_id=self._game_id,
                        seed=(self._seed + game_index) if self._seed is not None else None,
                        mode=self._mode,
                    )
                    self._version += 1
                    self._condition.notify_all()
                with tempfile.TemporaryDirectory(prefix="hexo_manual_records_") as tmp:
                    result = run_match(spec, players, tmp)
                with self._condition:
                    if self._generation != generation:
                        return
                    self._result = result
                    self._thinking_player = None
                    winner = str(result.winner) if result.winner is not None else None
                    winner_slot = seat_slots.get(winner) if winner is not None else None
                    if winner_slot is not None:
                        self._series_tally[winner_slot] += 1
                    else:
                        self._series_tally["draws"] += 1
                    self._series_results.append(
                        {
                            "game": game_index + 1,
                            "winner_seat": winner,
                            "winner_slot": winner_slot,
                            "length": int(result.turns),
                        }
                    )
                    aborted = result.abort is not None
                    if not aborted and len(self._series_results) >= self._series_games:
                        self._series_finished = True
                    self._version += 1
                    self._condition.notify_all()
                    if self._cancelled or aborted:
                        return
        except BaseException as exc:
            with self._condition:
                if self._generation != generation:
                    return
                self._error = exc
                self._thinking_player = None
                self._version += 1
                self._condition.notify_all()

    def _players_for_match(self, generation: int) -> tuple[object, object]:
        return (
            self._make_player(0, self._player_setup["player0"], generation),
            self._make_player(1, self._player_setup["player1"], generation),
        )

    def _make_player(self, player_index: int, spec: dict[str, object], generation: int) -> object:
        role = _player_role(player_index)
        kind = str(spec.get("kind") or MANUAL_KIND)
        if kind == MANUAL_KIND:
            return _ManualPlayer(self, player_index, generation, label=f"{_player_label(role)} Manual")

        if kind == "checkpoint":
            if self._checkpoint_factory is not None:
                bot = self._checkpoint_factory(dict(spec))
            else:
                bot = _CheckpointBotPlayer(spec)
            return _ObservedBotPlayer(self, player_index, generation, bot)

        variant = str(spec.get("variant") or "current")
        if self._bot_factory is not None:
            bot = self._bot_factory(variant, self._bot_time_limit)
        else:
            bot = SealBotPlayer(
                SealBotConfig(
                    path=self._sealbot_path,
                    variant=variant,
                    time_limit=self._bot_time_limit,
                )
            )
        return _ObservedBotPlayer(self, player_index, generation, bot)

    def _parse_match_config(self, config: dict[str, Any]) -> dict[str, Any]:
        bot = config.get("bot") if isinstance(config.get("bot"), dict) else {}
        time_limit = float(bot.get("time_limit") or self._bot_time_limit or DEFAULT_SEALBOT_TIME_LIMIT)
        if "time_limit" in config and config["time_limit"] not in {"", None}:
            time_limit = float(config["time_limit"])
        if time_limit <= 0:
            raise ValueError("SealBot time_limit must be positive.")
        seed = config.get("seed")
        players = self._normalize_player_setup(config)
        kinds = {str(spec.get("kind")) for spec in players.values()}
        if kinds == {MANUAL_KIND}:
            mode = "manual"
        elif "checkpoint" in kinds:
            mode = "checkpoint"
        else:
            mode = "sealbot"
        series = config.get("series") if isinstance(config.get("series"), dict) else {}
        games = max(1, min(int(series.get("games") or 1), SERIES_MAX_GAMES))
        return {
            "mode": mode,
            "players": players,
            "time_limit": time_limit,
            "seed": None if seed in {"", None} else int(seed),
            "series_games": games,
            "series_alternate": bool(series.get("alternate") or False),
        }

    def _normalize_player_setup(self, config: dict[str, Any]) -> dict[str, dict[str, object]]:
        raw_players = config.get("players")
        if isinstance(raw_players, dict):
            return {
                "player0": _normalize_player_spec(raw_players.get("player0", MANUAL_KIND)),
                "player1": _normalize_player_spec(raw_players.get("player1", MANUAL_KIND)),
            }

        # Legacy config shape (mode + human_player) from before the players{}
        # spec dict. app.js no longer sends it, but tests/test_sealbot_adapter.py
        # still exercises it; keep until those fixtures move to players{}.
        mode = str(config.get("mode") or "manual")
        if mode not in {"manual", "sealbot"}:
            raise ValueError(f"Unknown match mode: {mode}")
        if mode == "manual":
            return {"player0": {"kind": MANUAL_KIND}, "player1": {"kind": MANUAL_KIND}}

        human_player = str(config.get("human_player") or "player0")
        if human_player not in PLAYER_ROLES:
            raise ValueError("human_player must be player0 or player1.")
        bot = config.get("bot") if isinstance(config.get("bot"), dict) else {}
        variant = str(bot.get("variant") or "current")
        bot_spec = _normalize_player_spec({"kind": "sealbot", "variant": variant})
        return {
            "player0": {"kind": MANUAL_KIND} if human_player == "player0" else bot_spec,
            "player1": {"kind": MANUAL_KIND} if human_player == "player1" else dict(bot_spec),
        }

    def _wait_for_state_locked(self, timeout: float = 5.0) -> None:
        deadline = monotonic() + timeout
        while self._python_state is None and self._error is None:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RuntimeError("Timed out waiting for match state.")
            self._condition.wait(timeout=remaining)
        if self._python_state is None and self._error is not None:
            raise RuntimeError(str(self._error)) from self._error

    def _payload_locked(self) -> dict[str, object]:
        payload = dashboard_state(self._require_state_locked())
        payload.update(
            {
                "version": self._version,
                "game_id": self._game_id,
                "mode": self._mode,
                "players": self._players_payload_locked(),
                "turn_status": self._turn_status_locked(payload),
                "can_submit": self._can_submit_locked(),
                "thinking_player": self._thinking_player,
                "last_bot_decision": self._last_bot_decision,
                "bot_decisions": list(self._decision_log),
                "stopped": self._stopped,
                "series": self._series_payload_locked(),
                "error": self._error_message_locked(),
                "match": {
                    "players": {
                        "player0": dict(self._slot_specs["slot0"]),
                        "player1": dict(self._slot_specs["slot1"]),
                    },
                    "time_limit": self._bot_time_limit,
                    "seed": self._seed,
                },
            }
        )
        return payload

    def _players_payload_locked(self) -> dict[str, dict[str, object]]:
        return {
            role: _player_payload(index, self._player_setup[role])
            for index, role in enumerate(PLAYER_ROLES)
        }

    def _series_payload_locked(self) -> dict[str, object] | None:
        if self._series_games <= 1:
            return None
        return {
            "games": self._series_games,
            "played": len(self._series_results),
            "current_game": self._series_current_game,
            "alternate": self._series_alternate,
            "finished": self._series_finished,
            "tally": dict(self._series_tally),
            "slots": {
                "slot0": _player_payload(0, self._slot_specs["slot0"]),
                "slot1": _player_payload(1, self._slot_specs["slot1"]),
            },
            "seats": dict(self._seat_slots),
            "results": [dict(row) for row in self._series_results],
        }

    def _turn_status_locked(self, payload: dict[str, object]) -> str:
        if self._stopped:
            return "stopped"
        if self._error is not None or (self._result is not None and self._result.abort is not None):
            return "error"
        if self._result is not None or payload.get("winner") is not None:
            return "terminal"
        if self._thinking_player is not None:
            return "bot_thinking"
        current = str(payload.get("current_player") or "")
        return "bot_thinking" if _spec_is_bot(self._player_setup.get(current)) else "human_turn"

    def _can_submit_locked(self) -> bool:
        if self._stopped:
            return False
        if self._state is None or self._result is not None or self._pending_action is not None:
            return False
        if self._thinking_player is not None:
            return False
        if self._python_state is not None and self._python_state.terminal is not None:
            return False
        current = str(engine.current_player(self._state))
        if _spec_is_bot(self._player_setup.get(current)):
            return False
        return True

    def _error_message_locked(self) -> str | None:
        if self._error is not None:
            return str(self._error)
        if self._result is not None and self._result.abort is not None:
            return self._result.abort.message
        return None

    def _set_state_locked(self, state: engine.HexoState) -> None:
        self._state = state
        self._python_state = engine.to_python_state(state)

    def _require_state_locked(self) -> engine.PythonHexoState:
        if self._python_state is None:
            raise RuntimeError("Match state is unavailable.")
        return self._python_state


class _ManualPlayer:
    def __init__(self, controller: ManualMatchController, player_index: int, generation: int, *, label: str) -> None:
        self._controller = controller
        self._player_index = player_index
        self._generation = generation
        self.identity = PlayerIdentity(player_id=f"manual-player-{player_index}", label=label)

    def setup_worker(self, context: WorkerContext) -> None:
        return

    def start_game(self, context: GameContext) -> None:
        return

    def decide(self, state: engine.HexoState) -> DecisionResult:
        return self._controller.decide(self._player_index, state, self._generation)

    def observe_transition(self, transition: TransitionEvent) -> None:
        self._controller.observe_transition(transition, self._generation)

    def finish_game(self, final_summary: FinalSummary) -> None:
        return

    def close(self) -> None:
        return


def _select_visit_action(visit_policy: list[dict[str, object]], ply: int, game_token: str) -> int | None:
    """Eval-protocol move selection from a search's visit rows.

    Mirrors the trainer's SealBot-eval/arena protocol: the first
    ``CHECKPOINT_OPENING_MOVES`` plies are sampled from the visit distribution
    at ``CHECKPOINT_OPENING_TEMPERATURE`` (seeded per game+ply, so repeated
    games diverge while each stays reproducible); afterwards play is the
    strict visit argmax. Returns None when no row carries positive weight."""

    rows = [row for row in visit_policy if float(row.get("w") or 0.0) > 0.0]
    if not rows:
        return None
    if ply >= CHECKPOINT_OPENING_MOVES:
        return int(max(rows, key=lambda row: float(row["w"]))["action_id"])
    inverse = 1.0 / CHECKPOINT_OPENING_TEMPERATURE
    weights = [float(row["w"]) ** inverse for row in rows]
    # str-seeded Random is process-stable (seeded from a hash of the bytes),
    # unlike hash() on strings, so a replayed game samples the same opening.
    rng = random.Random(f"{game_token}|{ply}")
    return int(rng.choices([int(row["action_id"]) for row in rows], weights=weights, k=1)[0])


class _CheckpointBotPlayer:
    """Model player backed by the shared CPU debug worker (debug_service).

    ``search`` mode runs the worker's fresh reproducible MCTS (no root noise)
    and selects the move with the trainer's eval protocol — visit argmax after
    a sampled `CHECKPOINT_OPENING_MOVES`-ply opening (`_select_visit_action`),
    so a match-mode model plays like its SealBot-eval/arena self. ``policy``
    mode takes the prior argmax from one forward pass."""

    def __init__(self, spec: dict[str, object]) -> None:
        self._run = str(spec.get("run") or "")
        self._checkpoint = str(spec.get("checkpoint") or "")
        self._visits = int(spec.get("visits") or CHECKPOINT_VISITS_DEFAULT)
        self._mode = str(spec.get("mode") or "search")
        self._c_puct = float(spec.get("c_puct") or CHECKPOINT_C_PUCT_DEFAULT)
        self._ckpt_path = _debug_resolve_checkpoint(self._run, self._checkpoint)
        self._game_token = f"ckpt|{self._run}|{self._checkpoint}"
        self.identity = PlayerIdentity(
            player_id=f"ckpt-{self._run}-{Path(self._checkpoint).stem}",
            label=_checkpoint_label(self._run, self._checkpoint),
        )

    def setup_worker(self, context: WorkerContext) -> None:
        return

    def start_game(self, context: GameContext) -> None:
        # Per-game token for the opening-sampling RNG: distinct games (and the
        # two seats of one game) draw decorrelated openings, exactly like the
        # eval path's per-(game, move) seeds.
        self._game_token = f"{context.game_id}|{context.seed}|{context.player_index}"

    def decide(self, state: engine.HexoState) -> DecisionResult:
        acts = [pack_coord_id(rec.coord) for rec in engine.to_python_state(state).placement_history]
        # The first move also pays the worker's torch import + model load.
        timeout = min(300.0, max(60.0, self._visits * 0.15))
        if self._mode == "policy":
            signature = _debug_signature("match-policy", self._ckpt_path, acts, None)
            result = _debug_worker().cached(
                signature,
                "analyze",
                timeout=timeout,
                checkpoint=str(self._ckpt_path),
                action_ids=acts,
            )
            policy = result.get("policy") or []
            if not policy:
                raise RuntimeError(f"checkpoint policy returned no legal moves ({self._run}/{self._checkpoint})")
            action_id = int(policy[0]["action_id"])  # rows are sorted p-desc, legal-only
            diagnostics: dict[str, object] = {
                "root_value": result.get("value"),
                "mode": self._mode,
                "run": self._run,
                "checkpoint": self._checkpoint,
                "top_p": policy[0].get("p"),
            }
        else:
            # Eval protocol, mirrored end to end: sampled opening plies at
            # temperature 1 then greedy, with the selection made IN-SEARCH by
            # the run's as-trained profile (gumbel for gumbel-trained runs —
            # the worker reads the run manifest). The seed decorrelates per
            # (game, ply) exactly like the eval path's per-(game, move) seeds;
            # it feeds both the search's stochastic levers and the tempered
            # opening selection, and rides the cache signature so distinct
            # games never share a cached opening move.
            ply = len(acts)
            temperature = (
                CHECKPOINT_OPENING_TEMPERATURE_IN_SEARCH
                if ply < CHECKPOINT_OPENING_MOVES
                else 0.0
            )
            seed = zlib.crc32(f"{self._game_token}|{ply}".encode()) & 0x7FFFFFFF
            signature = _debug_signature(
                f"match-search:{self._visits}:{self._c_puct}:{temperature}:{seed}",
                self._ckpt_path,
                acts,
                None,
            )
            result = _debug_worker().cached(
                signature,
                "search",
                timeout=timeout,
                checkpoint=str(self._ckpt_path),
                action_ids=acts,
                visits=self._visits,
                c_puct=self._c_puct,
                seed=seed,
                temperature=temperature,
            )
            if result.get("selection_in_search"):
                # shrimp: the returned action IS the profile's selection
                # (tempered opening / greedy after) — play it directly.
                action_id = int(result["best_action_id"])
                selection = (
                    "in-search-opening" if temperature > 0.0 else "in-search-argmax"
                )
            else:
                # Other lineages: eval-protocol selection client-side (visit
                # sampling in the opening, argmax after). `best_action_id`
                # stays as the degenerate-rows fallback.
                selected = _select_visit_action(
                    result.get("visit_policy") or [], ply, self._game_token
                )
                action_id = int(result["best_action_id"]) if selected is None else selected
                selection = "opening-sample" if ply < CHECKPOINT_OPENING_MOVES else "argmax"
            diagnostics = {
                "root_value": result.get("root_value"),
                "visits": result.get("visits"),
                "mode": self._mode,
                "run": self._run,
                "checkpoint": self._checkpoint,
                "selection": selection,
                "search_profile": result.get("search_profile"),
            }
        return DecisionResult(
            action=engine.PlacementAction(unpack_coord_id(action_id)),
            diagnostics=diagnostics,
        )

    def observe_transition(self, transition: TransitionEvent) -> None:
        return

    def finish_game(self, final_summary: FinalSummary) -> None:
        return

    def close(self) -> None:
        return


class _ObservedBotPlayer:
    def __init__(self, controller: ManualMatchController, player_index: int, generation: int, delegate: object) -> None:
        self._controller = controller
        self._player_index = player_index
        self._generation = generation
        self._delegate = delegate
        self.identity = delegate.identity

    def setup_worker(self, context: WorkerContext) -> None:
        self._delegate.setup_worker(context)

    def start_game(self, context: GameContext) -> None:
        self._delegate.start_game(context)

    def decide(self, state: engine.HexoState) -> DecisionResult:
        self._controller.bot_decision_started(self._player_index, state, self._generation)
        started = perf_counter()
        try:
            result = self._delegate.decide(state)
        except BaseException as exc:
            self._controller.bot_decision_failed(
                self._player_index, exc, (perf_counter() - started) * 1000.0, self._generation
            )
            raise
        self._controller.bot_decision_finished(
            self._player_index, result, (perf_counter() - started) * 1000.0, self._generation
        )
        return result

    def observe_transition(self, transition: TransitionEvent) -> None:
        self._delegate.observe_transition(transition)
        self._controller.observe_transition(transition, self._generation)

    def finish_game(self, final_summary: FinalSummary) -> None:
        self._delegate.finish_game(final_summary)

    def close(self) -> None:
        self._delegate.close()


def _player_role(player_index: int) -> str:
    return "player0" if player_index == 0 else "player1"


def _player_label(role: str) -> str:
    return "P0" if role == "player0" else "P1"


def _spec_is_bot(spec: dict[str, object] | None) -> bool:
    return spec is not None and spec.get("kind") != MANUAL_KIND


def _normalize_player_spec(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        kind = str(value.get("kind") or value.get("adapter") or value.get("id") or MANUAL_KIND).strip().lower()
        if kind in {"manual", "human"}:
            return {"kind": MANUAL_KIND}
        if kind in {"bot", "sealbot"}:
            variant = str(value.get("variant") or "current")
            if variant not in {"current", "best"}:
                raise ValueError(f"Unknown player kind: sealbot-{variant}")
            return {"kind": "sealbot", "variant": variant}
        if kind == "checkpoint":
            run = str(value.get("run") or "")
            checkpoint = str(value.get("checkpoint") or "")
            # Resolve at parse time so a bad config 400s before any thread starts.
            ckpt_path = _debug_resolve_checkpoint(run, checkpoint)
            # Explicit None/"" checks: `or` would turn a literal 0 into the
            # default instead of clamping it to the spec floor.
            raw_visits = value.get("visits")
            visits = CHECKPOINT_VISITS_DEFAULT if raw_visits in (None, "") else int(raw_visits)
            visits = max(CHECKPOINT_VISITS_MIN, min(visits, CHECKPOINT_VISITS_MAX))
            mode = str(value.get("mode") or "search")
            if mode not in {"search", "policy"}:
                raise ValueError(f"Unknown checkpoint mode: {mode}")
            raw_c_puct = value.get("c_puct")
            c_puct = CHECKPOINT_C_PUCT_DEFAULT if raw_c_puct in (None, "") else float(raw_c_puct)
            c_puct = max(CHECKPOINT_C_PUCT_MIN, min(c_puct, CHECKPOINT_C_PUCT_MAX))
            return {
                "kind": "checkpoint",
                "run": run,
                "checkpoint": ckpt_path.name,
                "visits": visits,
                "mode": mode,
                "c_puct": c_puct,
            }
        return _normalize_player_spec(kind)

    kind = str(value or MANUAL_KIND).strip().lower()
    if kind in {"manual", "human"}:
        return {"kind": MANUAL_KIND}
    if kind in {"bot", "sealbot"}:
        return {"kind": "sealbot", "variant": "current"}
    if kind in {"sealbot-current", "sealbot-best"}:
        return {"kind": "sealbot", "variant": kind.removeprefix(SEALBOT_PREFIX)}
    raise ValueError(f"Unknown player kind: {kind}")


def _checkpoint_label(run: str, checkpoint: str) -> str:
    match = _DEBUG_CKPT_EPOCH_RE.search(checkpoint)
    short = f"e{int(match.group(1))}" if match else Path(checkpoint).stem
    return f"{run} @ {short}"


def _player_payload(player_index: int, spec: dict[str, object]) -> dict[str, object]:
    role = _player_role(player_index)
    kind = str(spec.get("kind") or MANUAL_KIND)
    if kind == MANUAL_KIND:
        return {"role": role, "kind": MANUAL_KIND, "label": "Manual"}
    if kind == "checkpoint":
        run = str(spec.get("run") or "")
        checkpoint = str(spec.get("checkpoint") or "")
        return {
            "role": role,
            "kind": "checkpoint",
            "run": run,
            "checkpoint": checkpoint,
            "visits": spec.get("visits"),
            "mode": spec.get("mode"),
            "label": _checkpoint_label(run, checkpoint),
        }
    variant = str(spec.get("variant") or "current")
    return {
        "role": role,
        "kind": "sealbot",
        "label": f"SealBot {variant}",
        "adapter_id": "sealbot",
        "variant": variant,
    }


# ---------------------------------------------------------------------------
# HTTP handler: the route table for all three screens lives in do_GET/do_POST.
# ---------------------------------------------------------------------------


class HexoPlayHandler(BaseHTTPRequestHandler):
    server_version = "hexo-frontend-play/0.1"
    # HTTP/1.1 keep-alive: reuse one TCP connection for index.html + app.js +
    # styles.css + the API calls instead of a fresh connection per request -- the
    # big win over high-latency LAN/VPN links. Every response sets Content-Length
    # (required for keep-alive); 304s carry no body.
    protocol_version = "HTTP/1.1"
    timeout = 30  # reap idle keep-alive connections so handler threads don't pile up
    controller: ClassVar[ManualMatchController]

    def do_GET(self) -> None:
        """GET route table. All API responses are JSON; errors are
        ``{"error": ...}`` with 400 (deterministic/bad request) or 500
        (debug-worker transport failure, retryable — see the except clauses).

        Match screen:
          /api/state?since=&timeout_ms=     long-poll match payload (controller.state)
          /api/adapters                     SealBot adapter discovery
        History screen (training-run scans; run dirs under cwd/runs):
          /api/training/runs                run list (3s cache)
          /api/training/run?name=           full run overview (trends/health/pages, 3s cache)
          /api/training/live?run=           fast status tier (~1s micro-cache)
          /api/training/epoch?run=&epoch=   epoch-inspector detail (uncached by design)
          /api/training/epochs?run=         per-epoch telemetry strip (selfplay/select/train/eval merge, mtime-cached)
          /api/training/history-page?run=&limit=&cursor=&source=&winner=&sort=&query=&include_total=
                                            paged .hxr game rows (opaque JSON cursor)
          /api/training/history-count?...   filtered game count
          /api/training/artifacts-page?run=&limit=&cursor=&kind=
                                            paged artifact listing (tests-only client today)
          /api/training/history?run=&path=&record=
                                            full replay payload for one recorded game
        Debug screen (positions resolved here, model work in the CPU worker):
          /api/debug/checkpoints?run=       checkpoint list + lineage + worker status
          /api/debug/games?run=&source=     .hxr files available for inspection
          /api/debug/position?run=&path=&record=&ply= (or ?actions=csv)
                                            replayed board payload at a ply
          /api/debug/ckpt_info, /record_row, /game_eval, /trajectory
                                            worker-backed forensics payloads
        Static: / and /index.html (no-store), /static/* (no-cache + ETag).
        """

        try:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/state":
                query = parse_qs(parsed.query)
                since = _query_int(query.get("since", [None])[0])
                timeout_ms = _query_int(query.get("timeout_ms", [None])[0]) or 0
                self._send_json(self.controller.state(since=since, timeout_ms=timeout_ms))
            elif path == "/api/adapters":
                self._send_json(self.controller.adapters())
            elif path == "/api/training/runs":
                self._send_json(_training_runs_cached())
            elif path == "/api/training/run":
                query = parse_qs(parsed.query)
                self._send_json(_training_run_cached(str(query.get("name", [""])[0])))
            elif path == "/api/training/live":
                query = parse_qs(parsed.query)
                self._send_json(_training_live_cached(str(query.get("run", [""])[0])))
            elif path == "/api/training/epoch":
                query = parse_qs(parsed.query)
                self._send_json(
                    _training_epoch(
                        str(query.get("run", [""])[0]),
                        _query_int(query.get("epoch", [None])[0]),
                    )
                )
            elif path == "/api/training/epochs":
                query = parse_qs(parsed.query)
                self._send_json(_training_epochs_cached(str(query.get("run", [""])[0])))
            elif path == "/api/training/history-page":
                query = parse_qs(parsed.query)
                self._send_json(
                    _training_history_page(
                        run_name=str(query.get("run", [""])[0]),
                        limit=_query_limit(
                            query.get("limit", [None])[0],
                            default=HISTORY_PAGE_DEFAULT_LIMIT,
                            maximum=HISTORY_PAGE_MAX_LIMIT,
                        ),
                        cursor=str(query.get("cursor", [""])[0] or ""),
                        source=str(query.get("source", ["all"])[0] or "all"),
                        winner=str(query.get("winner", ["all"])[0] or "all"),
                        sort=str(query.get("sort", ["newest"])[0] or "newest"),
                        query_text=str(query.get("query", [""])[0] or ""),
                        include_total=_query_bool(query.get("include_total", ["1"])[0], default=True),
                    )
                )
            elif path == "/api/training/history-count":
                query = parse_qs(parsed.query)
                self._send_json(
                    _training_history_count(
                        run_name=str(query.get("run", [""])[0]),
                        source=str(query.get("source", ["all"])[0] or "all"),
                        winner=str(query.get("winner", ["all"])[0] or "all"),
                        query_text=str(query.get("query", [""])[0] or ""),
                    )
                )
            elif path == "/api/training/artifacts-page":
                query = parse_qs(parsed.query)
                self._send_json(
                    _training_artifacts_page(
                        run_name=str(query.get("run", [""])[0]),
                        limit=_query_limit(
                            query.get("limit", [None])[0],
                            default=ARTIFACT_PAGE_DEFAULT_LIMIT,
                            maximum=ARTIFACT_PAGE_MAX_LIMIT,
                        ),
                        cursor=str(query.get("cursor", [""])[0] or ""),
                        kind=str(query.get("kind", ["all"])[0] or "all"),
                    )
                )
            elif path == "/api/training/history":
                query = parse_qs(parsed.query)
                self._send_json(
                    _training_history(
                        str(query.get("run", [""])[0]),
                        str(query.get("path", [""])[0]),
                        _query_int(query.get("record", [None])[0]) or 0,
                    )
                )
            elif path == "/api/debug/checkpoints":
                query = parse_qs(parsed.query)
                self._send_json(_debug_checkpoints(str(query.get("run", [""])[0])))
            elif path == "/api/debug/games":
                query = parse_qs(parsed.query)
                self._send_json(
                    _debug_games(
                        str(query.get("run", [""])[0]),
                        str(query.get("source", ["selfplay"])[0] or "selfplay"),
                    )
                )
            elif path == "/api/debug/trajectory":
                query = parse_qs(parsed.query)
                self._send_json(
                    _debug_trajectory(
                        str(query.get("run", [""])[0]),
                        str(query.get("path", [""])[0]),
                        _query_int(query.get("record", [None])[0]) or 0,
                        str(query.get("checkpoint", [""])[0]),
                    )
                )
            elif path == "/api/debug/position":
                query = parse_qs(parsed.query)
                actions_csv = str(query.get("actions", [""])[0] or "")
                if actions_csv:
                    self._send_json(
                        _debug_position_from_actions(
                            str(query.get("run", [""])[0]),
                            actions_csv,
                            _query_int(query.get("ply", [None])[0]) or 0,
                        )
                    )
                else:
                    self._send_json(
                        _debug_position(
                            str(query.get("run", [""])[0]),
                            str(query.get("path", [""])[0]),
                            _query_int(query.get("record", [None])[0]) or 0,
                            _query_int(query.get("ply", [None])[0]) or 0,
                        )
                    )
            elif path == "/api/debug/ckpt_info":
                query = parse_qs(parsed.query)
                self._send_json(
                    _debug_ckpt_info(
                        str(query.get("run", [""])[0]),
                        str(query.get("checkpoint", [""])[0]),
                        _query_int(query.get("radius", [None])[0]),
                    )
                )
            elif path == "/api/debug/record_row":
                query = parse_qs(parsed.query)
                self._send_json(
                    _debug_record_row(
                        str(query.get("run", [""])[0]),
                        str(query.get("path", [""])[0]),
                        _query_int(query.get("record", [None])[0]) or 0,
                        _query_int(query.get("ply", [None])[0]) or 0,
                    )
                )
            elif path == "/api/debug/game_eval":
                query = parse_qs(parsed.query)
                self._send_json(
                    _debug_game_eval(
                        str(query.get("run", [""])[0]),
                        str(query.get("path", [""])[0]),
                        _query_int(query.get("record", [None])[0]) or 0,
                        str(query.get("checkpoint", [""])[0]),
                        _query_int(query.get("start", [None])[0]) or 0,
                        _query_int(query.get("count", [None])[0]) or 16,
                        _query_int(query.get("radius", [None])[0]),
                    )
                )
            elif path == "/" or path == "/index.html":
                self._send_static("index.html")
            elif path.startswith("/static/"):
                self._send_static(unquote(path.removeprefix("/static/")))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except debug_service.DebugWorkerError as exc:
            # Worker infra failure (timeout/dead process — covers DebugWorkerTimeout):
            # 500 = "worker restarted, retry may succeed". DebugRequestError stays a
            # plain RuntimeError below -> 400 = deterministic, do not retry.
            self._send_json(self._error_payload(str(exc)), HTTPStatus.INTERNAL_SERVER_ERROR)
        except (TypeError, ValueError, RuntimeError) as exc:
            self._send_json(self._error_payload(str(exc)), HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        """POST route table (JSON bodies in, JSON out).

        Match screen:
          /api/new          start a match/series; body is the players{}/series
                            spec parsed by _parse_match_config -> full state
          /api/match/stop   halt the running series -> state payload
          /api/move {q,r}   human placement -> 409 MoveConflict when not our turn
        Debug screen (bodies carry run/path/record/ply or action_ids + checkpoint):
          /api/debug/analyze      all model heads for one position (cached)
          /api/debug/search       fresh CPU MCTS (visits/c_puct/seed)
          /api/debug/search_tree  deterministic PUCT debug tree (validated here
                                  so routine UI errors never restart the worker)
        """

        path = urlparse(self.path).path
        try:
            if path == "/api/new":
                self._send_json(self.controller.reset(self._read_json()))
            elif path == "/api/match/stop":
                self._send_json(self.controller.stop())
            elif path == "/api/move":
                body = self._read_json()
                self._send_json(self.controller.submit_move(int(body["q"]), int(body["r"])))
            elif path == "/api/debug/analyze":
                self._send_json(_debug_analyze(self._read_json()))
            elif path == "/api/debug/search":
                self._send_json(_debug_search(self._read_json()))
            elif path == "/api/debug/search_tree":
                self._send_json(_debug_search_tree(self._read_json()))
            elif path == "/api/debug/attention":
                self._send_json(_debug_attention(self._read_json()))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except MoveConflict as exc:
            self._send_json({"error": str(exc), "state": self.controller.state()}, HTTPStatus.CONFLICT)
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json({"error": str(exc), "state": self.controller.state()}, HTTPStatus.BAD_REQUEST)
        except debug_service.DebugWorkerError as exc:
            # Same taxonomy split as do_GET: infra failure -> 500 (retryable).
            self._send_json(self._error_payload(str(exc)), HTTPStatus.INTERNAL_SERVER_ERROR)
        except RuntimeError as exc:
            self._send_json(self._error_payload(str(exc)), HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _accepts_gzip(self) -> bool:
        return "gzip" in (self.headers.get("Accept-Encoding") or "").lower()

    def _send_body(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        cache_control: str | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
        gzip_body: bytes | None = None,
        allow_gzip: bool = True,
    ) -> None:
        """Write one response with conditional-GET (304), gzip, and Content-Length.

        ``gzip_body`` lets callers pass pre-compressed bytes (static assets) so they
        are not re-gzipped per request; otherwise the body is gzipped on the fly when
        the client accepts it and it is large enough to be worth it.
        """

        if etag is not None and status == HTTPStatus.OK:
            inm = self.headers.get("If-None-Match")
            if inm and any(etag == token.strip() for token in inm.split(",")):
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", etag)
                if cache_control:
                    self.send_header("Cache-Control", cache_control)
                self.end_headers()
                return

        encoding: str | None = None
        if allow_gzip and self._accepts_gzip():
            if gzip_body is not None:
                body, encoding = gzip_body, "gzip"
            elif len(body) >= GZIP_MIN_BYTES:
                body, encoding = gzip.compress(body, 6), "gzip"

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if encoding is not None:
            self.send_header("Content-Encoding", encoding)
            self.send_header("Vary", "Accept-Encoding")
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        if etag:
            self.send_header("ETag", etag)
        if last_modified:
            self.send_header("Last-Modified", last_modified)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        # ETag + revalidation lets the history screen's periodic re-poll receive a 304
        # (no body) whenever the run data is unchanged since the last fetch.
        etag = _strong_etag(encoded) if status == HTTPStatus.OK else None
        self._send_body(
            encoded,
            "application/json; charset=utf-8",
            status,
            cache_control="no-cache" if etag else None,
            etag=etag,
        )

    def _error_payload(self, message: str) -> dict[str, object]:
        try:
            return {"error": message, "state": self.controller.state()}
        except Exception:
            return {"error": message}

    def _send_static(self, name: str) -> None:
        if (not name) or ("/" in name) or ("\\" in name) or name.startswith("."):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        entry = _static_entry(name)
        if entry is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        raw, gz, etag, last_modified, content_type = entry
        # Cache policy is deliberately strict so a browser (esp. a phone, which
        # caches HTML aggressively) can NEVER run stale Debug code:
        #   - index.html: no-store -> the HTML document is always re-fetched, so its
        #     ?v= asset references are always current.
        #   - app.js / styles.css: no-cache -> the browser revalidates via ETag on
        #     every load and receives the current bytes the instant the file changes.
        # The asset is served from disk by (name, mtime), IGNORING the ?v= query, so
        # no-cache delivers fresh code even to a client still requesting an OLD ?v=
        # URL from a previously-cached page — exactly the stuck-on-old-app.js case
        # that left mobile on pre-fix Debug navigation. ETag still yields 304s when
        # nothing changed, so revalidation stays cheap.
        if name == "index.html":
            cache_control = "no-store, no-cache, must-revalidate, max-age=0"
        else:
            cache_control = "no-cache"
        self._send_body(
            raw,
            content_type,
            cache_control=cache_control,
            etag=etag,
            last_modified=last_modified,
            gzip_body=gz,
        )


# ---------------------------------------------------------------------------
# History screen: training-run discovery + run overview (the slow scan tier,
# memoized for TRAINING_CACHE_TTL_SECONDS by the *_cached wrappers up top).
# Everything reads run dirs under cwd/runs (production cwd = the run mount);
# the file formats are produced by hexo_train + the model packages.
# ---------------------------------------------------------------------------


def _query_int(value: str | None) -> int | None:
    if value in {"", None}:
        return None
    return int(value)


def _query_bool(value: str | None, *, default: bool) -> bool:
    if value in {"", None}:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _query_limit(value: str | None, *, default: int, maximum: int) -> int:
    raw = _query_int(value)
    if raw is None:
        return default
    return max(1, min(int(raw), maximum))


def _training_roots() -> tuple[Path, ...]:
    cwd = Path.cwd()
    candidates = (cwd / "runs", cwd / "configs" / "runs")
    roots: list[Path] = []
    seen: set[str] = set()
    for root in candidates:
        resolved = str(root.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(root)
    return tuple(roots)


def _training_runs() -> dict[str, object]:
    """Run-list payload for ``GET /api/training/runs``: every dir under the
    training roots that has a ``diagnostics/`` or ``selfplay/`` child, newest
    mtime first, deduped by name across roots (newest wins)."""

    runs_by_name: dict[str, dict[str, object]] = {}
    for root in _training_roots():
        if not root.exists():
            continue
        for path in sorted(
            root.iterdir(),
            key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)),
            reverse=True,
        ):
            if not path.is_dir():
                continue
            # Explicit opt-out: a run dir containing a `.dashboard_hidden` marker
            # file is skipped from the run list. Used to hide a stopped/erroneous
            # run that is kept on disk for its checkpoints/anchors but should not
            # appear in the UI (reversible: delete the marker to unhide).
            if (path / ".dashboard_hidden").exists():
                continue
            diagnostics = path / "diagnostics"
            selfplay = path / "selfplay"
            if not diagnostics.exists() and not selfplay.exists():
                continue
            stat = _safe_stat(path)
            current = {
                "name": path.name,
                "path": str(path),
                "diagnostics": str(diagnostics),
                "selfplay": str(selfplay),
                "modified": stat.st_mtime if stat is not None else 0,
            }
            existing = runs_by_name.get(path.name)
            if existing is None or float(current["modified"]) > float(existing["modified"]):
                runs_by_name[path.name] = current
    runs = sorted(runs_by_name.values(), key=lambda item: float(item["modified"]), reverse=True)
    return {"roots": [str(root) for root in _training_roots()], "runs": runs}


def _training_run(name: str) -> dict[str, object]:
    """Full run-overview payload for ``GET /api/training/run?name=`` (the
    History screen's 15s refresh, behind the 3s memo cache).

    Aggregates: per-epoch trend rows (``epoch_history``), SealBot results
    (``evaluation_history``), ``learning_health``, the live ``status`` block,
    plus first pages of artifacts and game history (the client pages further
    via /history-page and /artifacts-page). Unknown run -> ValueError (400)."""

    run_dir = _resolve_run_dir(name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    diagnostics_by_epoch = _diagnostics_by_epoch(run_dir)
    live_status = _training_live_status(run_dir)
    epoch_history = _epoch_history(run_dir, live_status)
    evaluation_history = _evaluation_history(run_dir)
    multistage_eval_history = _multistage_eval_history(run_dir)
    eval_pool = _eval_pool_summary(run_dir)
    artifacts_page = _training_artifacts_overview_page(
        run_dir,
        limit=TRAINING_OVERVIEW_ARTIFACT_LIMIT,
    )
    history_page = _training_history_page_for_runs(
        [(run_dir.name, run_dir)],
        limit=TRAINING_OVERVIEW_HISTORY_LIMIT,
        cursor="",
        source="all",
        winner="all",
        sort="newest",
        query_text="",
        include_total=False,
        diagnostics_cache={run_dir.name: diagnostics_by_epoch},
        live_status_cache={run_dir.name: live_status},
    )
    histories = list(history_page["items"])
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "artifacts": artifacts_page["items"],
        "artifacts_page": {
            "limit": TRAINING_OVERVIEW_ARTIFACT_LIMIT,
            "next_cursor": artifacts_page["next_cursor"],
            "complete": artifacts_page["complete"],
        },
        "histories": histories,
        "history_page": {
            "limit": TRAINING_OVERVIEW_HISTORY_LIMIT,
            "next_cursor": history_page["next_cursor"],
            "complete": history_page["complete"],
            "history_complete": False,
            "recent_history_count": len(histories),
        },
        "diagnostics_by_epoch": diagnostics_by_epoch,
        "epoch_history": epoch_history,
        "evaluation_history": evaluation_history,
        # Standalone shrimp multi-stage eval (opt-in). [] / None for runs
        # without the new artifacts (non-shrimp lineages, older runs).
        "multistage_eval_history": multistage_eval_history,
        "eval_pool": eval_pool,
        "learning_health": _learning_health(
            epoch_history, evaluation_history, live_status, multistage_eval_history
        ),
        "status": _training_run_status(run_dir, histories, live_status),
    }


def _training_epoch(name: str, epoch: int | None) -> dict[str, object]:
    """Single-epoch detail payload for the History screen's epoch inspector.

    Deliberately uncached: the client fetches once per inspector open and caches
    per (run, epoch), and the expensive part (_epoch_history's .hxr backfill) is
    already memoized by mtime/size via _hxr_base_rows. Do not wire this into a
    polling loop. A known run with no data at ``epoch`` is NOT an error -- the
    envelope comes back with all data fields None."""

    if epoch is None:
        raise ValueError("epoch is required")
    run_dir = _resolve_run_dir(name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    epoch = int(epoch)
    history_row: dict[str, object] | None = None
    prev_row: dict[str, object] | None = None
    for row in _epoch_history(run_dir):  # ascending by epoch
        row_epoch = row.get("epoch")
        if not isinstance(row_epoch, int):
            continue
        if row_epoch == epoch:
            history_row = row
        elif row_epoch < epoch:
            prev_row = row
    evaluation_row = next(
        (row for row in _evaluation_history(run_dir) if row.get("epoch") == epoch),
        None,
    )
    return {
        "run": run_dir.name,
        "epoch": epoch,
        "history": history_row,
        "prev_epoch": prev_row,
        "evaluation": evaluation_row,
        # Per-epoch multi-stage eval row (verdict + headline edges + rating
        # table), or None when this run/epoch has no standalone-eval report.
        "multistage_eval": next(
            (row for row in _multistage_eval_history(run_dir) if row.get("epoch") == epoch),
            None,
        ),
        "diagnostics": _diagnostics_by_epoch(run_dir).get(str(epoch)),
        "selfplay_extras": _selfplay_epoch_extras(run_dir, epoch),
        "manifest": _manifest_model_summary(run_dir),
        "checkpoint": _epoch_checkpoint_stat(run_dir, history_row, epoch),
    }


# ---------------------------------------------------------------------------
# Per-epoch telemetry strip (/api/training/epochs). One request scans the run's
# diagnostics dir once and merges the four shrimp per-epoch artifacts
# (selfplay / select / training / multistage_eval) into one flat record per
# epoch. Every field is guarded: a missing file, a missing key, or a legacy
# (pre-2026-07-03) schema simply yields None on that field — the record is
# never partial-crash, and the client renders "—". Both the legacy and the
# upgraded self-play key names are read, newest-schema winning.
# ---------------------------------------------------------------------------


def _tget(payload: object, key: str) -> object:
    """payload[key] when payload is a dict, else None (never KeyError)."""

    return payload.get(key) if isinstance(payload, dict) else None


def _tfirst(payload: object, *keys: str) -> object:
    """First non-None value among ``keys`` in a dict payload (schema fallback)."""

    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _fraction_of(numerator: object, denominator: object) -> float | None:
    """numerator/denominator as a float in [0, .], or None when either is
    missing / non-numeric / the denominator is zero. Used to synthesise the
    per-move rate fields (fast/full/init, gumbel/lcb) from the raw scheduler
    counters on legacy epochs that predate the emitted ``*_rate`` keys."""

    num = _optional_float(numerator)
    den = _optional_float(denominator)
    if num is None or den is None or den <= 0.0:
        return None
    return num / den


def _epoch_selfplay_record(payload: object) -> dict[str, object]:
    """Flatten one ``shrimp.selfplay.epoch_*.json`` into the strip record's
    self-play block, reading both the upgraded and the legacy key names and
    deriving the per-move rates from the ``scheduler`` counters when the
    producer did not emit the ``*_rate`` fields directly. All-None for a run/
    epoch without a self-play file (``payload`` is None)."""

    sched = payload.get("scheduler") if isinstance(_tget(payload, "scheduler"), dict) else {}
    full = _tget(sched, "full_moves")
    fast = _tget(sched, "fast_moves")
    init = _tget(sched, "init_moves")
    moves_decided = _tfirst(payload, "total_decisions") or _tget(sched, "moves_decided")
    gumbel_moves = _tget(sched, "gumbel_play_moves")
    gumbel_winner = _tget(sched, "gumbel_play_winner_moves")
    gumbel_early = _tget(sched, "gumbel_play_moves_early")
    gumbel_winner_early = _tget(sched, "gumbel_play_winner_early")
    lcb_overrides = _tget(sched, "lcb_overrides")

    wins_by_player = payload.get("wins_by_player") if isinstance(_tget(payload, "wins_by_player"), dict) else None
    p0_share: float | None = None
    if isinstance(wins_by_player, dict):
        w0 = _optional_float(wins_by_player.get("0"))
        w1 = _optional_float(wins_by_player.get("1"))
        if w0 is not None and w1 is not None and (w0 + w1) > 0.0:
            p0_share = w0 / (w0 + w1)

    return {
        "status": _tget(payload, "status"),
        "games_finished": _tfirst(payload, "games_finished", "games_started"),
        "games_started": _tget(payload, "games_started"),
        "truncated_games": _tget(payload, "truncated_games"),
        "rows_written": _tget(payload, "rows_written"),
        "elapsed_seconds": _tget(payload, "elapsed_seconds"),
        "search_visits": _tget(payload, "search_visits"),
        "mean_game_length": _tfirst(payload, "mean_game_length"),
        "game_length_p10": _tget(payload, "game_length_p10"),
        "game_length_p50": _tfirst(payload, "game_length_p50", "mean_game_length"),
        "game_length_p90": _tfirst(payload, "game_length_p90", "p90_game_length"),
        "game_length_max": _tget(payload, "game_length_max"),
        "root_policy_entropy_mean": _tget(payload, "root_policy_entropy_mean"),
        "root_policy_entropy_by_phase": payload.get("root_policy_entropy_by_phase")
        if isinstance(_tget(payload, "root_policy_entropy_by_phase"), dict)
        else None,
        "root_value_mean": _tget(payload, "root_value_mean"),
        "root_value_abs_mean": _tget(payload, "root_value_abs_mean"),
        "root_value_std": _tget(payload, "root_value_std"),
        "root_value_by_phase": payload.get("root_value_by_phase")
        if isinstance(_tget(payload, "root_value_by_phase"), dict)
        else None,
        "decided_fraction": _tget(payload, "decided_fraction"),
        "wins_by_player": wins_by_player,
        "p0_win_share": p0_share,
        # unique_openings is the upgraded {"10","16","20"} object; unique_openings_10ply
        # is the legacy scalar — surface it under the "10" slot so the 10-ply figure
        # is never lost on older epochs.
        "unique_openings": (
            payload.get("unique_openings")
            if isinstance(_tget(payload, "unique_openings"), dict)
            else ({"10": payload.get("unique_openings_10ply")} if _tget(payload, "unique_openings_10ply") is not None else None)
        ),
        "policy_surprise_mean": _tget(payload, "policy_surprise_mean"),
        "policy_surprise_p90": _tget(payload, "policy_surprise_p90"),
        "policy_surprise_max": _tget(payload, "policy_surprise_max"),
        # Per-move rates: prefer the emitted *_rate/​*_fraction fields, else derive
        # from the raw scheduler counters (older epochs carry only the counters).
        "fast_fraction": _tfirst(payload, "fast_fraction") if _tget(payload, "fast_fraction") is not None
        else _fraction_of(fast, moves_decided),
        "full_fraction": _tget(payload, "full_fraction") if _tget(payload, "full_fraction") is not None
        else _fraction_of(full, moves_decided),
        "init_fraction": _tget(payload, "init_fraction") if _tget(payload, "init_fraction") is not None
        else _fraction_of(init, moves_decided),
        "lcb_override_rate": _tget(payload, "lcb_override_rate") if _tget(payload, "lcb_override_rate") is not None
        else _fraction_of(lcb_overrides, moves_decided),
        "gumbel_play_winner_rate": _tget(payload, "gumbel_play_winner_rate") if _tget(payload, "gumbel_play_winner_rate") is not None
        else _fraction_of(gumbel_winner, gumbel_moves),
        "gumbel_play_winner_early_rate": _tget(payload, "gumbel_play_winner_early_rate") if _tget(payload, "gumbel_play_winner_early_rate") is not None
        else _fraction_of(gumbel_winner_early, gumbel_early),
        "scheduler": {
            "full_moves": full,
            "fast_moves": fast,
            "init_moves": init,
            "moves_decided": moves_decided,
            "gumbel_play_moves": gumbel_moves,
            "lcb_overrides": lcb_overrides,
        },
        # Resume/merge provenance: `resumed_skip`/`segments`/`merged_approx` drive
        # the "resumed" / "merged≈" badges. `resumed_existing_games` is the legacy
        # annotation on the zeroed resumed-epoch sample (epoch 13) — surface it so
        # that state reads as resumed too.
        "resumed_skip": _tget(payload, "resumed_skip"),
        "resumed_skip_count": _tget(payload, "resumed_skip_count"),
        "resumed_existing_games": _tget(payload, "resumed_existing_games"),
        "segments": payload.get("segments") if isinstance(_tget(payload, "segments"), list) else None,
        "merged_approx": _tget(payload, "merged_approx"),
    }


def _epoch_select_record(payload: object) -> dict[str, object]:
    """Flatten one ``shrimp.select.epoch_*.json`` into the strip record's
    window-selection block. ``skipped_paths`` is capped to keep the payload
    small; the count comes from ``shards_skipped``. All-None when absent."""

    skipped_paths = payload.get("skipped_paths") if isinstance(_tget(payload, "skipped_paths"), list) else None
    window_span = payload.get("window_epoch_span") if isinstance(_tget(payload, "window_epoch_span"), dict) else None
    return {
        "keep_prob": _tget(payload, "keep_prob"),
        "select_request": _tfirst(payload, "select_request", "desired_rows"),
        "selected_rows": _tfirst(payload, "selected_rows", "window_rows"),
        "window_rows": _tget(payload, "window_rows"),
        "reuse_ratio": _tget(payload, "reuse_ratio"),
        # shards_skipped is the upgraded count; fall back to the length of an
        # emitted skipped_paths list so the warning badge fires on older files too.
        "shards_skipped": _tget(payload, "shards_skipped")
        if _tget(payload, "shards_skipped") is not None
        else (len(skipped_paths) if isinstance(skipped_paths, list) else None),
        "skipped_paths": [str(p) for p in skipped_paths[:20]] if isinstance(skipped_paths, list) else None,
        "window_epoch_span": {
            "min": _tget(window_span, "min"),
            "max": _tget(window_span, "max"),
            "epochs": _tget(window_span, "epochs"),
        } if window_span is not None else None,
        "select_seconds": _tget(payload, "select_seconds"),
    }


def _epoch_training_record(payload: object) -> dict[str, object]:
    """Flatten one ``shrimp.training.epoch_*.json`` into the strip record's
    training block (the loss subset + step/timing keys). ``select_seconds``
    isn't in the training file today, so it stays None unless a future producer
    adds it. All-None when the file is absent."""

    return {
        "loss_policy": _tget(payload, "loss_policy"),
        "loss_soft_policy": _tget(payload, "loss_soft_policy"),
        "loss_value": _tget(payload, "loss_value"),
        "loss_total": _tget(payload, "loss_total"),
        "steps": _tget(payload, "steps"),
        "trained_rows": _tget(payload, "trained_rows"),
        "surprise_weight_mean": _tget(payload, "surprise_weight_mean"),
        "surprise_weight_max": _tget(payload, "surprise_weight_max"),
        "select_seconds": _tget(payload, "select_seconds"),
        # The training file emits `seconds`; the newer schema splits it into
        # select/train. Surface whichever is present as the train elapsed.
        "train_seconds": _tfirst(payload, "train_seconds", "seconds"),
    }


def _epoch_eval_record(row: object) -> dict[str, object] | None:
    """Compact headline-Elo block for the strip from one
    ``_multistage_eval_history`` row (present only at eval epochs). Carries the
    candidate's Elo point + the headline edges (opponent / winrate / Elo). None
    when there is no eval row for the epoch."""

    if not isinstance(row, dict):
        return None
    players = row.get("ratings", {}).get("players") if isinstance(_tget(row, "ratings"), dict) else None
    elo_point: float | None = None
    if isinstance(players, list):
        # The rating table is SealBot-anchored: the candidate is the (single)
        # non-anchor player. Its Elo lives under `elo` (not `rating`).
        candidate = next(
            (p for p in players if isinstance(p, dict) and not p.get("is_anchor")),
            None,
        )
        if candidate is not None:
            elo_point = _optional_float(candidate.get("elo"))
    edges = row.get("edges") if isinstance(_tget(row, "edges"), list) else []
    headline = [
        {
            "opponent": _tget(edge, "opponent"),
            "winrate": _tget(edge, "winrate"),
            "elo_point": _tget(edge, "elo_point"),
            "decided": _tget(edge, "decided"),
        }
        for edge in edges
        if isinstance(edge, dict) and edge.get("headline")
    ]
    return {
        "verdict_label": row.get("verdict_label") or _tget(row.get("verdict"), "label"),
        "elo_point": elo_point,
        "edges": headline,
    }


def _training_epochs(run_dir: Path) -> dict[str, object]:
    """Assemble the per-epoch telemetry strip for ``run_dir``.

    Scans ``diagnostics/`` once for the shrimp self-play / select / training
    per-epoch files, groups them by epoch, and emits one merged record per
    epoch (ascending) carrying the curated self-play / select / training key
    subsets plus a headline-Elo block at the epochs that have a multi-stage
    eval report. Every field degrades to None on a missing file or key, so the
    strip renders for legacy runs (pre-schema-upgrade) and mid-run epochs
    without crashing. Non-shrimp lineages (no ``shrimp.*`` files) yield an
    empty ``epochs`` list."""

    prefix = _diag_prefix(run_dir)
    diagnostics_dir = run_dir / "diagnostics"
    selfplay: dict[int, object] = {}
    select: dict[int, object] = {}
    training: dict[int, object] = {}
    if diagnostics_dir.is_dir():
        for kind, sink in (("selfplay", selfplay), ("select", select), ("training", training)):
            for path in sorted(diagnostics_dir.glob(f"{prefix}.{kind}.epoch_*.json")):
                payload = _read_json_file(path)
                epoch = _coerce_epoch(_tget(payload, "epoch"), path.name)
                if epoch is not None:
                    sink[epoch] = payload
    eval_by_epoch = {
        row.get("epoch"): row
        for row in _multistage_eval_history(run_dir)
        if isinstance(row.get("epoch"), int)
    }

    epochs = sorted(set(selfplay) | set(select) | set(training))
    records: list[dict[str, object]] = []
    for epoch in epochs:
        records.append(
            {
                "epoch": epoch,
                "selfplay": _epoch_selfplay_record(selfplay.get(epoch)),
                "select": _epoch_select_record(select.get(epoch)),
                "training": _epoch_training_record(training.get(epoch)),
                "eval": _epoch_eval_record(eval_by_epoch.get(epoch)),
            }
        )

    # Provisional in-flight record: the currently-running epoch (events.jsonl)
    # has no segment files during self-play, and only some of them afterwards.
    # Mark it (and synthesize it when wholly absent, all sub-blocks all-None)
    # so the strip/inspector can label the epoch "in progress" instead of
    # omitting it until the first segment lands.
    events = _stage_status_from_events(diagnostics_dir / "events.jsonl")
    current = events.get("epoch")
    if (
        isinstance(current, int)
        and str(events.get("status") or "") == "running"
        and not (diagnostics_dir / f"epoch_{current:06d}.json").is_file()
    ):
        record = next((rec for rec in records if rec["epoch"] == current), None)
        if record is None:
            record = {
                "epoch": current,
                "selfplay": _epoch_selfplay_record(None),
                "select": _epoch_select_record(None),
                "training": _epoch_training_record(None),
                "eval": None,
            }
            records.append(record)
            records.sort(key=lambda rec: rec["epoch"])
        record["in_progress"] = True
        live = _read_json_file(diagnostics_dir / f"{prefix}.selfplay.live.json")
        if isinstance(live, dict) and live.get("epoch") == current:
            record["live"] = _selfplay_live_summary(live)
    return {"run": run_dir.name, "epochs": records}


def _selfplay_epoch_extras(run_dir: Path, epoch: int) -> dict[str, object] | None:
    """Curated subset of ``diagnostics/dense_cnn.selfplay.epoch_*.json`` for the
    epoch inspector. Curation IS the size cap: the raw file is ~120KB and carries
    memory-internals (``mcts_diagnostics``, ``scheduler_diagnostics``, the
    384-entry ``npz_writes`` list, ``spill``, ``selfplay_npz_files``) that must
    never pass through. hexgt/hexgnn runs do not produce this file -> None."""

    prefix = _diag_prefix(run_dir)
    payload = _read_json_file(run_dir / "diagnostics" / f"{prefix}.selfplay.epoch_{epoch:06d}.json")
    if not isinstance(payload, dict):
        return None
    passthrough = (
        "scheduler",
        "raw_samples",
        "effective_samples",
        "total_decisions",
        "active_games",
        "mcts_virtual_batch_size",
        "elapsed_seconds",
        "mcts_search_elapsed_seconds",
        # shrimp self-play diagnostics carry these inline (dense_cnn omits them);
        # listed here so the epoch inspector surfaces them when present and stays a
        # no-op for dense_cnn (missing keys are simply skipped below).
        "search_visits",
        "games_started",
        "games_finished",
        "truncated_games",
        "rows_written",
        "full_decisions",
        "mean_game_length",
        "p90_game_length",
        "root_policy_entropy_mean",
        "root_value_mean",
    )
    nested = {
        "temperature_control": ("expected_game_length", "halflife_plies", "halflife_fraction"),
        "pcr": (
            "enabled",
            "full_proportion",
            "fast_visits",
            "full_search_count",
            "fast_search_count",
            "fast_rows_excluded",
        ),
        "policy_init": ("enabled", "fraction", "avg_plies", "max_plies", "temperature", "moves"),
        "root_policy_temperature_control": ("base", "early", "halflife_plies"),
    }
    extras: dict[str, object] = {key: payload[key] for key in passthrough if key in payload}
    for group, keys in nested.items():
        value = payload.get(group)
        if isinstance(value, dict):
            extras[group] = {key: value[key] for key in keys if key in value}
    return extras


def _manifest_model_summary(run_dir: Path) -> dict[str, object] | None:
    """Curated ``model.config`` subset from the run's ``manifest.json`` (arch +
    the exploration knobs under active study). hexgt/hexgnn/dense manifests
    differ in shape, so every level is dict-guarded and missing keys are simply
    omitted -- never KeyError."""

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(model, dict):
        return None
    config = model.get("config") if isinstance(model.get("config"), dict) else {}
    groups = {
        "architecture": (
            "input_channels",
            "channels",
            "blocks_type",
            "attention_heads",
            "short_term_value_horizons",
            "moves_left_head",
        ),
        "selfplay": (
            "search_visits",
            "active_games",
            "c_puct",
            "root_policy_temperature",
            "fpu_reduction",
            "temperature",
        ),
        "evaluation": ("games_per_epoch", "eval_every", "sealbot_variant"),
        "training": ("batch_size", "learning_rate", "train_samples_per_epoch"),
    }
    summary: dict[str, object] = {"model_name": model.get("name")}
    for group, keys in groups.items():
        value = config.get(group)
        if isinstance(value, dict):
            summary[group] = {key: value[key] for key in keys if key in value}
    return summary


def _epoch_checkpoint_stat(
    run_dir: Path,
    history_row: dict[str, object] | None,
    epoch: int,
) -> dict[str, object] | None:
    """Stat the epoch's checkpoint by its history-row name, falling back to the
    canonical ``epoch_{N:06d}.pt``. Tries the run dir first, then the Debug roots
    via _debug_resolve_run_path (CALL only -- covers the HEXO_DEBUG_RUN_ROOT
    worktree/mirror split where the history cwd lacks checkpoints)."""

    ckpt_name = ""
    checkpoint = history_row.get("checkpoint") if isinstance(history_row, dict) else None
    if isinstance(checkpoint, dict):
        raw_name = checkpoint.get("name") or ""
        if not raw_name and checkpoint.get("path"):
            raw_name = Path(str(checkpoint["path"])).name
        ckpt_name = str(raw_name)
    if not ckpt_name:
        ckpt_name = f"epoch_{epoch:06d}.pt"
    path = run_dir / "checkpoints" / ckpt_name
    if not path.is_file():
        resolved = _debug_resolve_run_path(run_dir.name, f"checkpoints/{ckpt_name}")
        if resolved is None or not resolved.is_file():
            return None
        path = resolved
    stat = _safe_stat(path)
    if stat is None:
        return None
    return {"name": ckpt_name, "size": int(stat.st_size), "mtime": stat.st_mtime}


def _training_history(run_name: str, artifact_path: str, record_index: int = 0) -> dict[str, object]:
    """Replay one recorded game (``GET /api/training/history``) into a full
    board payload: the .hxr record's action ids are re-applied through the
    engine and shaped by dashboard.dashboard_state, plus a ``history`` block
    (winner/status/abort) and a ``record_games`` index of the file's games.
    This is the History screen's "Load" view (read-only Match-board shape)."""

    path = _resolve_run_path(run_name, artifact_path)
    if path is None or not path.is_file() or path.suffix.lower() != ".hxr":
        raise ValueError("Unknown game history artifact")
    stat = _safe_stat(path)
    if stat is None or stat.st_size <= 0:
        raise ValueError("Game history artifact is empty")

    with HexoRecordFile.open(path) as record_file:
        players = [_record_player_payload(player) for player in record_file.players]
        records = list(record_file.iter_records())

    if not records:
        raise ValueError("Game history artifact contains no games")
    if record_index < 0 or record_index >= len(records):
        raise ValueError(f"Game history record index out of range: {record_index}")

    record = records[record_index]
    state = engine.new_game(seed=record.seed)
    applied_actions: list[int] = []
    for action_id in record.action_ids:
        action_id = int(action_id)
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(action_id)))
        applied_actions.append(action_id)

    payload = dashboard_state(engine.to_python_state(state))
    payload.update(
        {
            "version": int(stat.st_mtime_ns % 9_000_000_000_000_000),
            "game_id": f"{run_name}:{record.game_id}",
            "mode": "history",
            "players": _players_by_role(players),
            "turn_status": "history",
            "can_submit": False,
            "thinking_player": None,
            "last_bot_decision": None,
            "error": None,
            "match": {
                "players": {item["role"]: item["kind"] for item in players},
                "time_limit": None,
                "seed": record.seed,
            },
            "history": {
                "run": run_name,
                "path": artifact_path,
                "record_index": record_index,
                "record_count": len(records),
                "status": record.status,
                "winner": record.winner,
                "placements": record.placements,
                "action_ids": applied_actions,
                "abort": _abort_payload(record.abort),
            },
            "record_games": [
                {
                    "index": index,
                    "game_id": item.game_id,
                    "status": item.status,
                    "actions": len(item.action_ids),
                    "winner": item.winner,
                }
                for index, item in enumerate(records)
            ],
        }
    )
    return payload


# ---------------------------------------------------------------------------
# Debug tab: position inspection via the CPU inference worker (debug_service).
#
# These endpoints reconstruct a board position from a recorded game (or a raw
# move list) and ask the out-of-process, CPU-only worker what the model thinks —
# policy prior, value distribution, opponent-policy + STV heads, and on-demand
# MCTS. The worker is launched with CUDA_VISIBLE_DEVICES="" so it never contends
# for the training GPU, and results are cached so re-opening a view is instant.
# ---------------------------------------------------------------------------

_DEBUG_CKPT_EPOCH_RE = re.compile(r"epoch_?(\d+)\.pt$")  # hexgt 'epoch000040' + dense 'epoch_000030'
# The STV graft widened the value/STV readout heads at RL epoch 7 (also visible
# as a ~29 MB -> ~31 MB checkpoint-size jump); used only for a display hint.
_DEBUG_GRAFT_EPOCH = 7
_DEBUG_GRAFT_SIZE_BYTES = 30_500_000


def _debug_worker() -> "debug_service.DebugWorker":
    return debug_service.get_worker()


def _debug_training_roots() -> tuple[Path, ...]:
    """Run-dir search roots for the Debug endpoints. Prefers ``HEXO_DEBUG_RUN_ROOT``
    (e.g. the live worktree, which holds the real checkpoints + every .hxr) so the
    Debug tab works even when the dashboard serves its training/history panels
    from a bridge-mirror cwd that has the diagnostics but NOT the checkpoints.
    Falls back to the normal cwd-derived roots when the override is unset, so the
    default single-tree setup is unchanged."""

    roots: list[Path] = []
    seen: set[str] = set()
    override = os.environ.get("HEXO_DEBUG_RUN_ROOT")
    if override:
        base = Path(override).expanduser()
        for candidate in (base / "runs", base):  # accept the tree root or its runs/ dir
            if candidate.is_dir():
                key = str(candidate.resolve())
                if key not in seen:
                    seen.add(key)
                    roots.append(candidate)
    for root in _training_roots():
        key = str(root.resolve())
        if key not in seen:
            seen.add(key)
            roots.append(root)
    return tuple(roots)


def _debug_run_dirs(name: str) -> list[Path]:
    """Every existing run dir named ``name`` across the Debug roots, in priority
    order (override/worktree first, then the cwd roots)."""

    if not name or "/" in name or "\\" in name or name.startswith("."):
        return []
    dirs: list[Path] = []
    for root in _debug_training_roots():
        resolved_root = root.resolve()
        path = (resolved_root / name).resolve()
        if resolved_root != path and resolved_root not in path.parents:
            continue
        if path.is_dir():
            dirs.append(path)
    return dirs


def _debug_resolve_run_dir(name: str) -> Path | None:
    """First existing run dir in Debug-root priority order (worktree before the
    bridge-mirror cwd) — deterministic, so checkpoint/game listing comes from the
    tree that actually has the data rather than whichever has the newest mtime."""

    dirs = _debug_run_dirs(name)
    return dirs[0] if dirs else None


def _debug_resolve_run_path(run_name: str, artifact_path: str) -> Path | None:
    """Resolve an artifact path under whichever Debug run dir actually contains it.

    The dashboard's History tab may serve from a bridge-mirror cwd while the Debug
    endpoints prefer the live worktree, and the two trees can differ (the worktree
    leads on checkpoints; the mirror can momentarily lead on a just-rolled epoch
    .hxr). Trying each run dir in priority order and returning the first that holds
    the file lets a deep-link resolve regardless of which tree has that game."""

    if not artifact_path or artifact_path.startswith(("/", "\\")):
        return None
    fallback: Path | None = None
    for run_dir in _debug_run_dirs(run_name):
        resolved_root = run_dir.resolve()
        path = (run_dir / artifact_path).resolve()
        if resolved_root != path and resolved_root not in path.parents:
            continue
        if fallback is None:
            fallback = path
        if path.exists():
            return path
    return fallback  # confined but absent -> caller raises a clear "unknown artifact"


def _debug_run_lineage(run_dir: Path) -> str | None:
    """Best-effort model lineage for a run, read from its ``manifest.json``.

    Returns the model name (e.g. ``"hexgt"``, ``"dense_cnn_restnet"``,
    ``"hexo_models.dense_cnn"``) or ``None`` if the manifest is absent/unreadable.
    Used only for display hints (the authoritative lineage comes from the loaded
    checkpoint's ``meta`` once a position is analyzed)."""

    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    name = data.get("model", {}).get("name")
    return str(name) if name else None


def _diag_prefix(run_dir: Path) -> str:
    """Diagnostic-file basename prefix for a run's per-epoch self-play / training /
    evaluation JSON (``<prefix>.selfplay.epoch_*.json`` etc.).

    The shrimp lineage writes ``shrimp.*`` files. Derived from the run's
    manifest lineage so the dashboard reads the right files without hard-coding
    the prefix at every glob. A run whose manifest is absent/unreadable or whose
    model name does not identify shrimp (e.g. a hand-staged or imported run
    dir) falls back to the legacy ``dense_cnn`` prefix."""

    lineage = _debug_run_lineage(run_dir)
    if lineage and "shrimp" in lineage.lower():
        return "shrimp"
    return "dense_cnn"


def _debug_detect_radius(run_dir: Path) -> int | None:
    """Detected ``SHRIMP_SUPPORT_RADIUS`` for a run, or ``None`` when it does
    not apply or cannot be read. Source: the latest
    ``shrimp.multistage_eval.epoch_*.json``'s ``featurize_radius`` (the one
    structured on-disk record of the eval process's support radius). ``None`` for
    a non-shrimp run (no support radius applies). Shrimp with no/old eval ->
    ``None``; the caller defaults to 8 and the UI override corrects the rest
    (e.g. an older run whose evals predate the ``featurize_radius``
    annotation)."""

    lineage = _debug_run_lineage(run_dir)
    if not (lineage and "shrimp" in lineage.lower()):
        return None
    diag = run_dir / "diagnostics"
    if not diag.is_dir():
        return None
    try:
        names = sorted(
            (
                e.name
                for e in os.scandir(diag)
                if e.is_file()
                and e.name.startswith("shrimp.multistage_eval.epoch_")
                and e.name.endswith(".json")
            ),
            reverse=True,  # zero-padded epoch -> lexical desc == newest first
        )
    except OSError:
        return None
    for name in names:
        try:
            data = json.loads((diag / name).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        fit = data.get("ratings", {}).get("fit", {})
        if isinstance(fit, dict) and "featurize_radius" in fit:
            return int(fit["featurize_radius"])
        for edge in data.get("edges", []):
            if isinstance(edge, dict) and "featurize_radius" in edge:
                return int(edge["featurize_radius"])
    return None


def _debug_resolve_radius(run_dir: Path | None, body: dict[str, Any]) -> int | None:
    """The ``SHRIMP_SUPPORT_RADIUS`` to run the worker at for this request, or
    ``None`` when radius does not apply (non-shrimp lineage) so the worker is
    not keyed on it. Manual override (body ``radius`` in {4,8}) wins; else
    detected; else 8. Returns ``None`` when the run is not shrimp (detection
    returned ``None`` AND no override) so dense/hexgt stay byte-identical."""

    if run_dir is None:
        return None
    lineage = _debug_run_lineage(run_dir)
    is_shrimp = bool(lineage and "shrimp" in lineage.lower())
    override = body.get("radius")
    if override in (4, 8, "4", "8"):
        ov = int(override)
        # Honor an explicit override only for shrimp runs (the UI shows the
        # control only for shrimp); for non-shrimp, ignore it -> None.
        return ov if is_shrimp else None
    if not is_shrimp:
        return None
    detected = _debug_detect_radius(run_dir)
    return detected if detected is not None else 8


def _debug_checkpoints(run_name: str) -> dict[str, object]:
    run_dir = _debug_resolve_run_dir(run_name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    lineage = _debug_run_lineage(run_dir)
    # The pre/post-STV graft chip is a legacy readout-widening annotation that
    # does not apply to the shrimp lineage, so it is suppressed for shrimp
    # runs to avoid a misleading label (attached only for runs without a
    # shrimp manifest).
    is_graft_lineage = lineage is None or "hexgt" in lineage.lower()
    ckpt_dir = run_dir / "checkpoints"
    items: list[dict[str, object]] = []
    if ckpt_dir.is_dir():
        for entry in os.scandir(ckpt_dir):
            if not entry.is_file() or not entry.name.endswith(".pt"):
                continue
            match = _DEBUG_CKPT_EPOCH_RE.search(entry.name)
            epoch = int(match.group(1)) if match else None
            stat = _safe_stat(Path(entry.path))
            size = int(stat.st_size) if stat else 0
            graft: str | None = None
            if is_graft_lineage:
                if epoch is not None:
                    graft = "post" if epoch >= _DEBUG_GRAFT_EPOCH else "pre"
                elif size:
                    graft = "post" if size > _DEBUG_GRAFT_SIZE_BYTES else "pre"
            items.append(
                {
                    "name": entry.name,
                    "epoch": epoch,
                    "size": size,
                    "mtime": int(stat.st_mtime) if stat else 0,
                    "latest": entry.name.endswith("latest.pt"),
                    "graft": graft,
                }
            )
    items.sort(key=lambda x: (not x["latest"], -(x["epoch"] if x["epoch"] is not None else -1), str(x["name"])))
    return {
        "run": run_name,
        "checkpoints": items,
        "lineage": lineage,
        "support_radius_detected": _debug_detect_radius(run_dir),  # int | None
        "worker": _debug_worker().status(),
    }


def _debug_games(run_name: str, source: str) -> dict[str, object]:
    """List the recorded game files (``.hxr``) available for inspection. Self-play
    files are one-per-epoch; evaluation files live in ``eval*/`` subdirectories."""

    run_dir = _debug_resolve_run_dir(run_name)
    if run_dir is None:
        raise ValueError("Unknown training run")

    def rel(p: Path) -> str:
        return p.relative_to(run_dir).as_posix()

    def hxr_in(directory: Path, recurse: bool) -> list[Path]:
        found: list[Path] = []
        if not directory.is_dir():
            return found
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith(".hxr"):
                found.append(Path(entry.path))
            elif recurse and entry.is_dir():
                for sub in os.scandir(entry.path):
                    if sub.is_file() and sub.name.endswith(".hxr"):
                        found.append(Path(sub.path))
        return found

    files: list[Path] = []
    if source in ("selfplay", "all"):
        files += hxr_in(run_dir / "selfplay", recurse=False)
    if source in ("evaluation", "all"):
        files += hxr_in(run_dir / "evaluation", recurse=True)
        files += hxr_in(run_dir / "eval", recurse=True)

    items = []
    for path in files:
        stat = _safe_stat(path)
        items.append(
            {
                "path": rel(path),
                "name": path.name,
                "size": int(stat.st_size) if stat else 0,
                "mtime": int(stat.st_mtime) if stat else 0,
            }
        )
    items.sort(key=lambda x: str(x["path"]), reverse=True)
    return {"run": run_name, "source": source, "games": items}


def _debug_resolve_checkpoint(run_name: str, checkpoint: str) -> Path:
    name = checkpoint.strip()
    if not name:
        raise ValueError("checkpoint is required")
    if "/" in name or "\\" in name:  # accept a bare filename only, resolve under the run
        name = Path(name).name
    path = _debug_resolve_run_path(run_name, f"checkpoints/{name}")
    if path is None or not path.is_file():
        raise ValueError(f"Unknown checkpoint: {checkpoint}")
    return path


def _debug_open_record(run_name: str, artifact_path: str, record_index: int):
    path = _debug_resolve_run_path(run_name, artifact_path)
    if path is None or not path.is_file() or path.suffix.lower() != ".hxr":
        raise ValueError("Unknown game history artifact")
    with HexoRecordFile.open(path) as record_file:
        players = [_record_player_payload(player) for player in record_file.players]
        records = list(record_file.iter_records())
    if not records:
        raise ValueError("Game history artifact contains no games")
    if record_index < 0 or record_index >= len(records):
        raise ValueError(f"Game history record index out of range: {record_index}")
    return records[record_index], players, records


def _debug_build_position(action_ids: list[int], ply: int, *, seed: object = None) -> dict[str, object]:
    """Replay ``action_ids[:ply]`` into a board-state payload. Coordinates
    (including the last move) are resolved server-side via the engine, so the
    client never re-implements action-id unpacking."""

    total = len(action_ids)
    ply = max(0, min(int(ply), total))
    state = engine.new_game(seed=seed)
    last_coord = None
    for action_id in action_ids[:ply]:
        coord = unpack_coord_id(action_id)
        engine.apply_action(state, engine.PlacementAction(coord))
        last_coord = coord

    payload = dashboard_state(engine.to_python_state(state))
    payload["mode"] = "debug"
    payload["debug"] = {
        "ply": ply,
        "total": total,
        "action_ids": action_ids,
        "last_action_id": action_ids[ply - 1] if ply > 0 else None,
        "last_q": int(last_coord.q) if last_coord is not None else None,
        "last_r": int(last_coord.r) if last_coord is not None else None,
    }
    return payload


def _debug_position(run_name: str, artifact_path: str, record_index: int, ply: int) -> dict[str, object]:
    record, players, records = _debug_open_record(run_name, artifact_path, record_index)
    action_ids = [int(a) for a in record.action_ids]
    payload = _debug_build_position(action_ids, ply, seed=record.seed)
    payload["game_id"] = f"{run_name}:{record.game_id}"
    payload["players"] = _players_by_role(players)
    payload["debug"].update(
        {
            "run": run_name,
            "path": artifact_path,
            "record_index": record_index,
            "record_count": len(records),
            "winner": record.winner,
            "status": record.status,
            "seed": record.seed,
        }
    )
    payload["record_games"] = [
        {
            "index": index,
            "game_id": item.game_id,
            "status": item.status,
            "actions": len(item.action_ids),
            "winner": item.winner,
        }
        for index, item in enumerate(records)
    ]
    return payload


def _debug_position_from_actions(run_name: str, actions_csv: str, ply: int) -> dict[str, object]:
    """Board payload for a pasted/imported action-id list (no .hxr)."""

    action_ids: list[int] = []
    for token in re.split(r"[\s,]+", actions_csv.strip()):
        if not token:
            continue
        try:
            action_ids.append(int(token))
        except ValueError as exc:
            raise ValueError(f"invalid action id: {token!r}") from exc
    if not action_ids:
        raise ValueError("no action ids provided")
    ply = len(action_ids) if ply <= 0 else ply
    payload = _debug_build_position(action_ids, ply)
    payload["game_id"] = f"{run_name}:imported"
    payload["debug"].update({"run": run_name, "imported": True, "winner": None})
    payload["record_games"] = []
    return payload


def _debug_action_prefix(body: dict[str, Any]) -> tuple[str, list[int]]:
    """Resolve (run, action_id prefix) from a debug request body. Either an
    explicit ``action_ids`` list (paste/import) or a recorded game + ``ply``."""

    run = str(body.get("run", ""))
    raw = body.get("action_ids")
    if raw is not None:
        return run, [int(a) for a in raw]
    record, _players, _records = _debug_open_record(run, str(body.get("path", "")), int(body.get("record", 0) or 0))
    full = [int(a) for a in record.action_ids]
    ply = int(body.get("ply", len(full)))
    ply = max(0, min(ply, len(full)))
    return run, full[:ply]


def _debug_signature(
    prefix: str, ckpt_path: Path, action_ids: list[int], n: object, radius: int | None = None
) -> str:
    return json.dumps([prefix, str(ckpt_path), action_ids, n, radius], separators=(",", ":"))


def _debug_analyze(body: dict[str, Any]) -> dict[str, object]:
    run, action_ids = _debug_action_prefix(body)
    ckpt_path = _debug_resolve_checkpoint(run, str(body.get("checkpoint", "")))
    n = body.get("n")
    planes = bool(body.get("planes", False))
    radius = _debug_resolve_radius(_debug_resolve_run_dir(run), body)
    signature = _debug_signature(f"analyze:planes={int(planes)}", ckpt_path, action_ids, n, radius)
    return _debug_worker().cached(
        signature, "analyze", radius=radius, checkpoint=str(ckpt_path), action_ids=action_ids, n=n, planes=planes
    )


def _debug_search(body: dict[str, Any]) -> dict[str, object]:
    run, action_ids = _debug_action_prefix(body)
    ckpt_path = _debug_resolve_checkpoint(run, str(body.get("checkpoint", "")))
    n = body.get("n")
    visits = int(body.get("visits", 512))
    c_puct = float(body.get("c_puct", 1.5))
    seed = int(body.get("seed", 0) or 0)  # B2: forward to the worker (default keeps determinism tests)
    visits = max(1, min(visits, 20_000))  # bound CPU work per request
    radius = _debug_resolve_radius(_debug_resolve_run_dir(run), body)
    signature = _debug_signature(f"search:{visits}:{c_puct}:{seed}", ckpt_path, action_ids, n, radius)
    return _debug_worker().cached(
        signature,
        "search",
        timeout=debug_service.DEFAULT_TIMEOUT,
        radius=radius,
        checkpoint=str(ckpt_path),
        action_ids=action_ids,
        visits=visits,
        c_puct=c_puct,
        seed=seed,
        n=n,
    )


def _debug_search_tree(body: dict[str, Any]) -> dict[str, object]:
    """Pure-Python deterministic PUCT tree for the Tree Explorer (§3.7).

    ``root_actions`` (optional) are appended AFTER the resolved position prefix,
    so a deep subtree can be re-rooted/expanded without re-basing the UI position."""

    run, action_ids = _debug_action_prefix(body)
    ckpt_path = _debug_resolve_checkpoint(run, str(body.get("checkpoint", "")))
    n = body.get("n")
    radius = _debug_resolve_radius(_debug_resolve_run_dir(run), body)
    visits = max(1, min(int(body.get("visits", 512)), 20_000))
    c_puct = float(body.get("c_puct", 1.5))
    seed = int(body.get("seed", 0) or 0)
    max_depth = max(1, min(int(body.get("max_depth", 12)), 40))
    top_k = max(1, min(int(body.get("top_k", 8)), 32))
    min_n = max(0, min(int(body.get("min_n", 2)), 1_000_000))
    action_ids = action_ids + [int(a) for a in (body.get("root_actions") or [])]
    # Validate the position HERE so a routine UI request (e.g. the final ply of
    # a finished game, or a stale root_actions path) becomes a plain 400 and
    # never reaches the worker as an error: the worker stays warm.
    state = engine.new_game()
    for index, action_id in enumerate(action_ids):
        try:
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(int(action_id))))
        except Exception as exc:
            raise ValueError(f"illegal action id {action_id} at ply {index}: {exc}") from exc
    if engine.terminal(state) is not None:
        raise ValueError("position is terminal; nothing to search")
    # py_debug runs one batch-of-1 CPU net forward per visit (~25ms each; spec
    # §4.3 pegs 4096 visits near the 120s budget), so a fixed 120s deadline
    # guarantees a mid-compute worker KILL for any legal >4000-visit request.
    # search_tree alone scales its deadline with the spec'd 1..20000 visit
    # range, capped at 300s so one request cannot hog the single worker lock.
    timeout = min(300.0, max(debug_service.DEFAULT_TIMEOUT, visits * 0.03))
    signature = _debug_signature(
        f"search_tree:{visits}:{c_puct}:{seed}:{max_depth}:{top_k}:{min_n}", ckpt_path, action_ids, n, radius
    )
    return _debug_worker().cached(
        signature,
        "search_tree",
        timeout=timeout,
        radius=radius,
        checkpoint=str(ckpt_path),
        action_ids=action_ids,
        visits=visits,
        c_puct=c_puct,
        seed=seed,
        max_depth=max_depth,
        top_k=top_k,
        min_n=min_n,
        n=n,
    )


def _debug_attention(body: dict[str, Any]) -> dict[str, object]:
    """shrimp interactive attention map for one position (§Model Debug attn).

    Additive route: never alters analyze/search/etc. Resolves the position +
    checkpoint exactly like search/search_tree, clamps block/head/query in the
    server (so bad UI input is a deterministic 400, never a worker kill), and
    validates the position by replay (terminal -> 400). One CPU forward computes
    the full attention internally; the worker slices it to O(N)."""

    run, action_ids = _debug_action_prefix(body)
    ckpt_path = _debug_resolve_checkpoint(run, str(body.get("checkpoint", "")))
    n = body.get("n")
    radius = _debug_resolve_radius(_debug_resolve_run_dir(run), body)

    block = max(0, min(int(body.get("block", 0)), 2))
    raw_head = body.get("head")
    if raw_head in (None, ""):
        head = None
    elif raw_head == "max":
        head = "max"
    else:
        head = max(0, min(int(raw_head), 3))

    q = body.get("query") or {}
    qtype = str(q.get("type", "cell"))
    if qtype not in ("token", "cell"):
        raise ValueError(f"query.type must be 'token' or 'cell', got {qtype!r}")
    qid = int(q.get("id", 0))
    if qtype == "token":
        qid = max(0, min(qid, 7))  # 8 summary tokens; cell ids validated in worker

    # Validate the position HERE (mirror search_tree): a routine UI request on a
    # terminal/illegal position becomes a plain 400 and never reaches the worker.
    state = engine.new_game()
    for index, action_id in enumerate(action_ids):
        try:
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(int(action_id))))
        except Exception as exc:
            raise ValueError(f"illegal action id {action_id} at ply {index}: {exc}") from exc
    if engine.terminal(state) is not None:
        raise ValueError("position is terminal; no attention to inspect")

    # One forward is ~ms even at N~3000, so no visit-style timeout scaling.
    qtag = f"{qtype}:{qid}"
    signature = _debug_signature(
        f"attention:{block}:{('mean' if head is None else head)}:{qtag}",
        ckpt_path,
        action_ids,
        n,
        radius,
    )
    return _debug_worker().cached(
        signature,
        "attention",
        timeout=debug_service.DEFAULT_TIMEOUT,
        radius=radius,
        checkpoint=str(ckpt_path),
        action_ids=action_ids,
        block=block,
        head=head,
        query={"type": qtype, "id": qid},
        n=n,
    )


def _debug_ckpt_info(run_name: str, checkpoint: str, radius: int | None = None) -> dict[str, object]:
    """Checkpoint provenance WITHOUT paying an analyze (§3.8, fixes B3): the
    worker ``info`` op (cached like any result) + a stat on the resolved file.

    ``radius`` (optional UI override, 4|8) selects the worker's support radius so
    ``meta.support_radius`` reflects the toggle; ``None`` -> detected-else-8 for
    shrimp, no radius for other lineages."""

    ckpt_path = _debug_resolve_checkpoint(run_name, checkpoint)
    # info is radius-independent (reads checkpoint meta) BUT it now also surfaces
    # the worker's ACTIVE support radius (meta.support_radius), so run the op in a
    # worker spawned at the run's radius. An explicit query override (radius=4|8)
    # lets the UI's info chip follow the toggle; else detected-else-8 for shrimp.
    override = {"radius": radius} if radius in (4, 8) else {}
    radius = _debug_resolve_radius(_debug_resolve_run_dir(run_name), override)
    signature = _debug_signature("info", ckpt_path, [], None, radius)
    meta = dict(_debug_worker().cached(signature, "info", radius=radius, checkpoint=str(ckpt_path)))
    arch = meta.get("arch")
    if isinstance(arch, dict):  # contract wants a display string, not the raw dict
        meta["arch"] = ", ".join(f"{key}={arch[key]}" for key in sorted(arch)) or None
    stat = _safe_stat(ckpt_path)
    return {
        "run": run_name,
        "checkpoint": ckpt_path.name,
        "size": int(stat.st_size) if stat else 0,
        "mtime": float(stat.st_mtime) if stat else 0.0,
        "meta": meta,
    }


_DEBUG_HXR_EPOCH_RE = re.compile(r"epoch_(\d+)\.hxr$")
_DEBUG_GAME_INDEX_RE = re.compile(r"(\d+)$")  # trailing self-play game index in a game_id


def _debug_resolve_record_npz(run_name: str, artifact_path: str, game_id: object) -> Path | None:
    """Locate the compact ``.npz`` training shard for one self-play .hxr record.

    Shard candidates are ``selfplay/epoch_NNNNNN_game_*.npz`` for the .hxr's
    epoch. Match order: a sidecar ``<shard>.json`` whose ``game_id`` equals the
    record's; then the self-play game index parsed off the record's game_id
    (shards are named ``..._game_{index}.npz`` by that index). Returns None when
    nothing matches. There is deliberately NO record-index fallback: .hxr
    records are written in game FINISH order while shards are named by the
    self-play game index, so ``record_index`` would silently attach a different
    game's shard whenever the orders diverge (they routinely do)."""

    match = _DEBUG_HXR_EPOCH_RE.search(Path(artifact_path).name)
    if match is None:
        return None
    epoch = int(match.group(1))
    hxr_path = _debug_resolve_run_path(run_name, artifact_path)
    if hxr_path is None or not hxr_path.is_file():
        return None
    shard_dir = hxr_path.parent
    prefix = f"epoch_{epoch:06d}_game_"
    candidates = sorted(shard_dir.glob(f"{prefix}*.npz"))
    if not candidates:
        return None
    for shard in candidates:
        sidecar = shard.with_suffix(".json")
        if sidecar.is_file():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            if str(data.get("game_id")) == str(game_id):
                return shard
    index_match = _DEBUG_GAME_INDEX_RE.search(str(game_id or ""))
    if index_match:
        exact = shard_dir / f"{prefix}{int(index_match.group(1)):06d}.npz"
        if exact.is_file():
            return exact
    return None


def _debug_current_player_at(action_ids: list[int], ply: int) -> int:
    """Replay-derived 0/1 side-to-move at ``ply`` (the M8 misalignment guard's
    expectation; parity is NOT assumed — phases can repeat a player)."""

    state = engine.new_game()
    for action_id in action_ids[:ply]:
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(int(action_id))))
    role = getattr(engine.current_player(state), "value", "")
    return 1 if str(role).endswith("1") else 0


_DEBUG_WINNER_INDEX = {"player0": 0, "player1": 1}


def _debug_record_row(run_name: str, artifact_path: str, record_index: int, ply: int) -> dict[str, object]:
    """Recorded .npz training row for one (game, ply) (§3.9). Self-play only."""

    def miss(reason: str) -> dict[str, object]:
        return {"found": False, "reason": reason, "npz": None, "turn_index": None, "row": None}

    if not artifact_path.replace("\\", "/").startswith("selfplay/"):
        return miss("not_selfplay")
    record, _players, _records = _debug_open_record(run_name, artifact_path, record_index)
    npz_path = _debug_resolve_record_npz(run_name, artifact_path, record.game_id)
    if npz_path is None:
        return miss("no_shard")
    action_ids = [int(a) for a in record.action_ids]
    ply = max(0, min(int(ply), len(action_ids)))
    expect_player = _debug_current_player_at(action_ids, ply)

    signature = _debug_signature(f"record_row:{ply}:{expect_player}", npz_path, [], None)
    result = dict(
        _debug_worker().cached(
            signature, "record_row", npz=str(npz_path), turn_index=ply, expect_player=expect_player
        )
    )
    if result.get("npz"):
        result["npz"] = f"selfplay/{npz_path.name}"  # run-relative, like every other path
    row = result.get("row")
    if isinstance(row, dict):
        # Overlay the .hxr-derived facts the compact shard does not persist.
        row = dict(row)  # never mutate the cached copy
        truncated = str(getattr(record, "status", "")) != "completed"
        row["truncated"] = bool(truncated)
        if truncated and not row.get("value_target_reason"):
            row["value_target_reason"] = "max_actions_draw"
        result["row"] = row
    return result


def _debug_game_eval(
    run_name: str,
    artifact_path: str,
    record_index: int,
    checkpoint: str,
    start: int,
    count: int,
    radius: int | None = None,
) -> dict[str, object]:
    """Game Error Sweep chunk (§3.10): the .hxr record is opened once here, the
    worker decodes the matching .npz shard once and joins per ply internally.

    ``radius`` (optional UI override, 4|8) keys the worker's support radius so the
    sweep evaluates at the run's trained radius; ``None`` -> detected-else-8 for
    shrimp, no radius for other lineages."""

    record, _players, _records = _debug_open_record(run_name, artifact_path, record_index)
    action_ids = [int(a) for a in record.action_ids]
    total = len(action_ids)
    ckpt_path = _debug_resolve_checkpoint(run_name, checkpoint)
    override = {"radius": radius} if radius in (4, 8) else {}
    eval_radius = _debug_resolve_radius(_debug_resolve_run_dir(run_name), override)
    start = max(0, min(int(start), total))
    count = max(1, min(int(count), 32))
    plies = list(range(start, min(start + count, total)))
    winner = _DEBUG_WINNER_INDEX.get(str(record.winner)) if record.winner is not None else None
    npz_path = None
    if artifact_path.replace("\\", "/").startswith("selfplay/"):
        npz_path = _debug_resolve_record_npz(run_name, artifact_path, record.game_id)

    signature = _debug_signature(
        f"game_eval:{start}:{count}:{winner}:{npz_path}", ckpt_path, action_ids, None, eval_radius
    )
    raw = _debug_worker().cached(
        signature,
        "game_eval",
        timeout=debug_service.DEFAULT_TIMEOUT,
        radius=eval_radius,
        checkpoint=str(ckpt_path),
        action_ids=action_ids,
        plies=plies,
        npz=str(npz_path) if npz_path else None,
        winner=winner,
    )
    return {
        "run": run_name,
        "path": artifact_path,
        "record": record_index,
        "checkpoint": ckpt_path.name,
        "total": total,
        "start": start,
        "count": count,
        "winner": winner,
        "plies": raw.get("plies", []),
        # v3 cell_q lineages echo the absolute regret blunder threshold so the UI
        # can mark Q-based blunders; null/absent for older (no cell_q) checkpoints.
        "regret_blunder_threshold": raw.get("regret_blunder_threshold"),
    }


def _debug_recorded_trajectory(run_dir: Path, artifact_path: str, game_id: object) -> list[dict[str, object]]:
    """Best-effort recorded root_value per move from ``eval/epoch_*_examples.json``.

    Only self-play example games carry per-move traces, so this returns ``[]`` when
    no matching trace exists. Values are normalized to player-0's perspective."""

    match = re.search(r"epoch_(\d+)", Path(artifact_path).name)
    if match is None:
        return []
    epoch = int(match.group(1))
    examples_path = run_dir / "eval" / f"epoch_{epoch:06d}_examples.json"
    if not examples_path.is_file():
        return []
    try:
        with examples_path.open("r", encoding="utf-8") as handle:
            games = json.load(handle)
    except (OSError, ValueError):
        return []
    trace = None
    for game in games if isinstance(games, list) else []:
        if str(game.get("game_id")) == str(game_id):
            trace = game.get("moves") or []
            break
    if not trace:
        return []
    out = []
    for move in trace:
        rv = move.get("root_value")
        if rv is None:
            continue
        ply = int(move.get("move", 0))
        player0 = str(move.get("player", "player0")).endswith("0")
        out.append({"ply": ply, "root_value": float(rv), "root_value_p0": float(rv) if player0 else -float(rv)})
    return out


def _debug_trajectory(run_name: str, artifact_path: str, record_index: int, checkpoint: str, max_points: int = 160) -> dict[str, object]:
    run_dir = _debug_resolve_run_dir(run_name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    record, _players, _records = _debug_open_record(run_name, artifact_path, record_index)
    action_ids = [int(a) for a in record.action_ids]
    total = len(action_ids)
    ckpt_path = _debug_resolve_checkpoint(run_name, checkpoint)

    # Evaluate plies 0..total, strided so a long game stays bounded (one forward
    # per point). The stride is surfaced so the UI never implies full coverage.
    stride = max(1, -(-(total + 1) // max_points))
    plies = list(range(0, total + 1, stride))
    if plies[-1] != total:
        plies.append(total)
    sequences = [action_ids[:p] for p in plies]

    signature = _debug_signature(f"trajectory:{stride}", ckpt_path, action_ids, max_points)
    raw = _debug_worker().cached(
        signature, "reeval", checkpoint=str(ckpt_path), sequences=sequences, timeout=debug_service.DEFAULT_TIMEOUT
    )
    reeval = []
    for entry in raw.get("values", []):
        cp = int(entry.get("current_player", 0))
        value = float(entry.get("value", 0.0))
        reeval.append({"ply": int(entry["ply"]), "value": value, "current_player": cp,
                       "value_p0": value if cp == 0 else -value})

    return {
        "run": run_name,
        "path": artifact_path,
        "record": record_index,
        "total": total,
        "stride": stride,
        "checkpoint": ckpt_path.name,
        "winner": record.winner,
        "reeval": reeval,
        "recorded": _debug_recorded_trajectory(run_dir, artifact_path, record.game_id),
    }


# ---------------------------------------------------------------------------
# History screen: artifact listing + paged .hxr game history.
#
# Game rows are built from .hxr files via _hxr_base_rows (memoized by
# mtime/size in _hxr_history_cache) and paged with an opaque JSON cursor so
# the newest-first stream stays stable while new epochs roll in.
# ---------------------------------------------------------------------------


def _training_artifacts_page(
    *,
    run_name: str,
    limit: int = ARTIFACT_PAGE_DEFAULT_LIMIT,
    cursor: str = "",
    kind: str = "all",
) -> dict[str, object]:
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None:
        raise ValueError("Unknown training run")
    return _training_artifacts_page_for_run(run_dir, limit=limit, cursor=cursor, kind=kind)


def _training_artifacts_overview_page(run_dir: Path, *, limit: int) -> dict[str, object]:
    paths: list[Path] = []

    def add_direct(root: Path, suffixes: set[str] | frozenset[str] = ARTIFACT_SUFFIXES) -> None:
        try:
            entries = list(os.scandir(root))
        except OSError:
            return
        for entry in entries:
            if entry.is_file() and Path(entry.name).suffix.lower() in suffixes:
                paths.append(Path(entry.path))

    def add_recent_child_dirs(
        root: Path,
        suffixes: set[str] | frozenset[str],
        *,
        max_dirs: int = 2,
    ) -> None:
        try:
            entries = list(os.scandir(root))
        except OSError:
            return
        dirs: list[tuple[float, Path]] = []
        for entry in entries:
            if entry.is_file() and Path(entry.name).suffix.lower() in suffixes:
                paths.append(Path(entry.path))
            elif entry.is_dir():
                try:
                    dirs.append((entry.stat().st_mtime, Path(entry.path)))
                except OSError:
                    continue
        for _, directory in sorted(dirs, reverse=True)[:max_dirs]:
            add_direct(directory, suffixes)

    def add_recursive_limited(
        root: Path,
        suffixes: set[str] | frozenset[str] = ARTIFACT_SUFFIXES,
        *,
        max_files: int = 100,
    ) -> None:
        if not root.is_dir():
            return
        for path in _iter_training_files(root, suffix=None):
            if path.is_file() and path.suffix.lower() in suffixes:
                paths.append(path)
                if len(paths) >= max_files:
                    return

    add_direct(run_dir)
    add_direct(run_dir / "diagnostics")
    add_direct(run_dir / "selfplay", {".hxr"})
    add_recent_child_dirs(run_dir / "evaluation", {".hxr"})
    add_direct(run_dir / "checkpoints")
    add_recursive_limited(run_dir / "bootstrap")

    unique = {str(path.resolve()): path for path in paths}
    paths = list(unique.values())
    paths.sort(key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)), reverse=True)
    selected = paths[: limit + 1]
    return {
        "run": run_dir.name,
        "items": [_artifact_payload(run_dir, path) for path in selected[:limit]],
        "next_cursor": str(limit) if len(selected) > limit else None,
        "complete": len(selected) <= limit,
        "scanned_files": len(paths),
    }


def _training_artifacts_page_for_run(
    run_dir: Path,
    *,
    limit: int,
    cursor: str,
    kind: str,
) -> dict[str, object]:
    offset = max(0, _query_int(cursor) or 0)
    wanted_kind = str(kind or "all").lower()
    paths = [
        path
        for path in _iter_training_files(run_dir)
        if path.is_file()
        and path.suffix.lower() in ARTIFACT_SUFFIXES
        and (wanted_kind == "all" or path.suffix.lower().lstrip(".") == wanted_kind)
    ]
    paths.sort(key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)), reverse=True)
    selected = paths[offset : offset + limit + 1]
    items = [_artifact_payload(run_dir, path) for path in selected[:limit]]
    next_offset = offset + limit
    return {
        "run": run_dir.name,
        "items": items,
        "next_cursor": str(next_offset) if len(selected) > limit else None,
        "complete": len(selected) <= limit,
        "scanned_files": len(paths),
    }


def _artifact_payload(run_dir: Path, path: Path) -> dict[str, object]:
    rel = path.relative_to(run_dir).as_posix()
    stat = _safe_stat(path)
    suffix = path.suffix.lower()
    artifact: dict[str, object] = {
        "path": rel,
        "name": path.name,
        "bytes": stat.st_size if stat is not None else 0,
        "modified": stat.st_mtime if stat is not None else 0,
        "kind": suffix.lstrip(".") or "file",
        "loadable_history": False,
        "history_count": 0,
    }
    if suffix == ".json":
        payload = _read_json_file(path)
        artifact["summary"] = _artifact_summary(payload)
    elif suffix == ".hxr" and _is_loadable_history_path(rel) and stat is not None and stat.st_size > 0:
        rows = _hxr_base_rows(path, run_dir)
        history_count = len(rows)
        artifact["loadable_history"] = history_count > 0
        artifact["history_count"] = history_count
    return artifact


def _training_history_page(
    *,
    run_name: str,
    limit: int = HISTORY_PAGE_DEFAULT_LIMIT,
    cursor: str = "",
    source: str = "all",
    winner: str = "all",
    sort: str = "newest",
    query_text: str = "",
    include_total: bool = True,
) -> dict[str, object]:
    """Paged game-history rows for ``GET /api/training/history-page``.

    ``run_name`` may be HISTORY_ALL_RUNS ("__all__") to merge every run.
    Returns ``{items, next_cursor, complete, total_matches, scanned_files,
    scanned_games, sort}``. newest/oldest stream file-by-file with a
    row-identity cursor (_history_cursor_key); the complete sorts
    (longest/shortest/winner) must materialize all rows first and use a plain
    integer-offset cursor instead."""

    run_infos = _history_run_infos(run_name)
    if not run_infos:
        raise ValueError("Unknown training run")
    return _training_history_page_for_runs(
        run_infos,
        limit=limit,
        cursor=cursor,
        source=source,
        winner=winner,
        sort=sort,
        query_text=query_text,
        include_total=include_total,
    )


def _training_history_count(
    *,
    run_name: str,
    source: str = "all",
    winner: str = "all",
    query_text: str = "",
) -> dict[str, object]:
    run_infos = _history_run_infos(run_name)
    if not run_infos:
        raise ValueError("Unknown training run")
    total_matches, scanned_files, scanned_games = _training_history_count_for_runs(
        run_infos,
        source=source,
        winner=winner,
        query_text=query_text,
    )
    return {
        "total_matches": total_matches,
        "scanned_files": scanned_files,
        "scanned_games": scanned_games,
    }


def _training_history_count_for_runs(
    run_infos: list[tuple[str, Path]],
    *,
    source: str,
    winner: str,
    query_text: str,
) -> tuple[int, int, int]:
    total_matches = 0
    scanned_files = 0
    scanned_games = 0
    can_count_without_rows = _history_filter_matches_all(winner=winner, query_text=query_text)
    diagnostics_cache: dict[str, dict[str, object]] | None = None
    live_status_cache: dict[str, dict[str, object]] | None = None
    if not can_count_without_rows:
        diagnostics_cache = {
            run_name: _diagnostics_by_epoch(run_dir)
            for run_name, run_dir in run_infos
        }
        live_status_cache = {
            run_name: _training_live_status(run_dir)
            for run_name, run_dir in run_infos
        }

    for run_name, run_dir, path, _stat in _history_files_for_runs(run_infos, source=source, reverse=True):
        scanned_files += 1
        if can_count_without_rows:
            record_count = _hxr_record_count(path, run_dir)
            total_matches += record_count
            scanned_games += record_count
            continue
        rows = _history_rows_for_file(
            run_name,
            run_dir,
            path,
            diagnostics_cache=diagnostics_cache,
            live_status_cache=live_status_cache,
            reverse_records=False,
        )
        scanned_games += len(rows)
        total_matches += sum(1 for row in rows if _history_row_matches(row, winner=winner, query_text=query_text))

    return total_matches, scanned_files, scanned_games


def _training_history_page_for_runs(
    run_infos: list[tuple[str, Path]],
    *,
    limit: int,
    cursor: str,
    source: str,
    winner: str,
    sort: str,
    query_text: str,
    include_total: bool = True,
    diagnostics_cache: dict[str, dict[str, object]] | None = None,
    live_status_cache: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    diagnostics_cache = diagnostics_cache or {
        run_name: _diagnostics_by_epoch(run_dir)
        for run_name, run_dir in run_infos
    }
    live_status_cache = live_status_cache or {
        run_name: _training_live_status(run_dir)
        for run_name, run_dir in run_infos
    }
    normalized_sort = sort if sort in {"newest", "oldest", "longest", "shortest", "winner"} else "newest"
    if normalized_sort in {"longest", "shortest", "winner"}:
        rows, scanned_files, scanned_games = _collect_history_rows(
            run_infos,
            source=source,
            diagnostics_cache=diagnostics_cache,
            live_status_cache=live_status_cache,
        )
        rows = [row for row in rows if _history_row_matches(row, winner=winner, query_text=query_text)]
        rows.sort(key=lambda item: _history_complete_sort_key(item, normalized_sort))
        offset = max(0, _query_int(cursor) or 0)
        selected = rows[offset : offset + limit]
        next_offset = offset + limit
        return {
            "items": selected,
            "next_cursor": str(next_offset) if next_offset < len(rows) else None,
            "complete": next_offset >= len(rows),
            "total_matches": len(rows),
            "scanned_files": scanned_files,
            "scanned_games": scanned_games,
            "sort": normalized_sort,
        }

    return _training_history_streaming_page(
        run_infos,
        limit=limit,
        cursor=cursor,
        source=source,
        winner=winner,
        sort=normalized_sort,
        query_text=query_text,
        include_total=include_total,
        diagnostics_cache=diagnostics_cache,
        live_status_cache=live_status_cache,
    )


def _training_history_streaming_page(
    run_infos: list[tuple[str, Path]],
    *,
    limit: int,
    cursor: str,
    source: str,
    winner: str,
    sort: str,
    query_text: str,
    include_total: bool,
    diagnostics_cache: dict[str, dict[str, object]] | None,
    live_status_cache: dict[str, dict[str, object]] | None,
) -> dict[str, object]:
    """Streaming page builder for the newest/oldest sorts: walks the epoch-
    ordered .hxr file list, decoding rows only until the page fills, then
    (when ``include_total`` and the filter is pass-all) counts the remaining
    files by record count alone without building their rows."""

    reverse = sort != "oldest"
    cursor_key = _decode_history_cursor(cursor)
    passed_cursor = cursor_key is None
    selected: list[dict[str, object]] = []
    has_more = False
    total_matches: int | None = 0 if include_total else None
    can_count_without_rows = include_total and _history_filter_matches_all(winner=winner, query_text=query_text)
    scanned_files = 0
    scanned_games = 0

    for run_name, run_dir, path, _stat in _history_files_for_runs(run_infos, source=source, reverse=reverse):
        scanned_files += 1
        if can_count_without_rows and has_more:
            record_count = _hxr_record_count(path, run_dir)
            total_matches = (total_matches or 0) + record_count
            scanned_games += record_count
            continue
        rows = _history_rows_for_file(
            run_name,
            run_dir,
            path,
            diagnostics_cache=diagnostics_cache,
            live_status_cache=live_status_cache,
            reverse_records=reverse,
        )
        scanned_games += len(rows)
        if can_count_without_rows:
            total_matches = (total_matches or 0) + len(rows)
        for row in rows:
            matches = True if can_count_without_rows else _history_row_matches(row, winner=winner, query_text=query_text)
            if include_total and not can_count_without_rows and matches:
                total_matches = (total_matches or 0) + 1
            row_key = _history_cursor_key(row)
            if not passed_cursor:
                if row_key == cursor_key:
                    passed_cursor = True
                continue
            if not matches:
                continue
            if len(selected) >= limit:
                has_more = True
                if include_total:
                    if can_count_without_rows:
                        break
                    continue
                break
            selected.append(row)
        if has_more and not include_total:
            break

    return {
        "items": selected,
        "next_cursor": _encode_history_cursor(_history_cursor_key(selected[-1])) if has_more and selected else None,
        "complete": not has_more,
        "total_matches": total_matches,
        "scanned_files": scanned_files,
        "scanned_games": scanned_games,
        "sort": sort,
    }


def _history_run_infos(run_name: str) -> list[tuple[str, Path]]:
    if run_name == HISTORY_ALL_RUNS:
        infos: list[tuple[str, Path]] = []
        for item in _training_runs()["runs"]:
            resolved = _resolve_run_dir(str(item.get("name") or ""))
            if resolved is not None:
                infos.append((resolved.name, resolved))
        return infos
    run_dir = _resolve_run_dir(run_name)
    return [] if run_dir is None else [(run_dir.name, run_dir)]


def _collect_history_rows(
    run_infos: list[tuple[str, Path]],
    *,
    source: str,
    diagnostics_cache: dict[str, dict[str, object]] | None,
    live_status_cache: dict[str, dict[str, object]] | None,
) -> tuple[list[dict[str, object]], int, int]:
    rows: list[dict[str, object]] = []
    scanned_files = 0
    scanned_games = 0
    for run_name, run_dir, path, _stat in _history_files_for_runs(run_infos, source=source, reverse=True):
        scanned_files += 1
        file_rows = _history_rows_for_file(
            run_name,
            run_dir,
            path,
            diagnostics_cache=diagnostics_cache,
            live_status_cache=live_status_cache,
            reverse_records=False,
        )
        rows.extend(file_rows)
        scanned_games += len(file_rows)
    return rows, scanned_files, scanned_games


def _history_files_for_runs(
    run_infos: list[tuple[str, Path]],
    *,
    source: str,
    reverse: bool,
) -> list[tuple[str, Path, Path, os.stat_result]]:
    files: list[tuple[str, Path, Path, os.stat_result]] = []
    for run_name, run_dir in run_infos:
        for path, stat in _iter_history_artifact_files(run_dir, source=source):
            if stat.st_size <= 0:
                continue
            rel = path.relative_to(run_dir).as_posix()
            if not _is_loadable_history_path(rel):
                continue
            files.append((run_name, run_dir, path, stat))
    files.sort(
        key=lambda item: (
            _epoch_from_artifact_path(item[2].relative_to(item[1]).as_posix()) or 0,
            item[3].st_mtime,
            str(item[0]),
            item[2].relative_to(item[1]).as_posix(),
        ),
        reverse=reverse,
    )
    return files


def _iter_history_artifact_files(
    run_dir: Path,
    *,
    source: str,
) -> list[tuple[Path, os.stat_result]]:
    normalized_source = str(source or "all").lower()
    roots: list[Path] = []
    if normalized_source in {"", "all", "selfplay"}:
        roots.append(run_dir / "selfplay")
    if normalized_source in {"", "all", "evaluation"}:
        roots.append(run_dir / "evaluation")

    files: list[tuple[Path, os.stat_result]] = []
    for root in roots:
        if not root.is_dir():
            continue
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                entries = list(os.scandir(current))
            except OSError:
                continue
            for entry in entries:
                name = entry.name
                if name.startswith(".") or name in TRAINING_SCAN_EXCLUDED_DIRS:
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and name.endswith(".hxr"):
                        files.append((Path(entry.path), entry.stat(follow_symlinks=False)))
                except OSError:
                    continue
    return files


def _history_rows_for_file(
    run_name: str,
    run_dir: Path,
    path: Path,
    *,
    diagnostics_cache: dict[str, dict[str, object]] | None,
    live_status_cache: dict[str, dict[str, object]] | None,
    reverse_records: bool,
) -> list[dict[str, object]]:
    rel = path.relative_to(run_dir).as_posix()
    base_rows = _hxr_base_rows(path, run_dir)
    if reverse_records:
        base_rows = list(reversed(base_rows))
    epoch = _epoch_from_artifact_path(rel)
    source = _history_source(rel)
    diagnostics_by_epoch = (
        diagnostics_cache.get(run_name)
        if diagnostics_cache is not None and run_name in diagnostics_cache
        else _diagnostics_by_epoch(run_dir)
    )
    live_status = (
        live_status_cache.get(run_name)
        if live_status_cache is not None and run_name in live_status_cache
        else _training_live_status(run_dir)
    )
    diagnostics = dict(diagnostics_by_epoch.get(str(epoch), {})) if epoch is not None else {}
    if (
        live_status
        and source == "selfplay"
        and epoch is not None
        and int(live_status.get("current_epoch") or -1) == int(epoch)
        and "selfplay" not in diagnostics
    ):
        diagnostics["live"] = {
            "path": rel,
            "summary": _live_history_diagnostic_summary(live_status),
        }
    brief = _history_diagnostics_brief(diagnostics)
    rows: list[dict[str, object]] = []
    for row in base_rows:
        item = dict(row)
        item["run"] = run_name
        item["diagnostics"] = brief
        rows.append(item)
    return rows


def _candidate_seat_from_game_id(game_id: object) -> str | None:
    """Return which seat ("player0"/"player1") the run's own candidate net held
    in an evaluation game, or None when it cannot be determined.

    Evaluation .hxr games (written by shrimp.eval_arena._write_eval_hxr) carry
    seat-symmetric player labels ("cand_epN/opp · seat 0/1"), so the label alone
    does not identify the candidate. The candidate seat SWAPS per game (CRN
    pairing) and is instead encoded as a "-candP0"/"-candP1" suffix on the
    record's game_id (e.g. "ep65-cand_ep65-vs-ep60-g3-candP1"). Selfplay
    game_ids ("epoch-000066-game-...") carry no such suffix, so this returns None
    for them, which is the correct "no current-model-vs-opponent" answer."""

    text = str(game_id or "")
    if text.endswith("-candP0"):
        return "player0"
    if text.endswith("-candP1"):
        return "player1"
    return None


def _hxr_base_rows(path: Path, run_dir: Path) -> list[dict[str, object]]:
    """Decode one .hxr file into per-game summary rows (game_id, winner,
    length, players, abort, candidate_seat, ...), memoized by (mtime_ns, size)
    in _hxr_history_cache. Returns row COPIES so callers can annotate freely.
    This cache is what makes history paging and the .hxr stat backfills
    (_selfplay_game_stats_from_records) cheap on re-poll."""

    stat = _safe_stat(path)
    if stat is None or stat.st_size <= 0:
        return []
    cache_key = str(path.resolve())
    with _training_cache_lock:
        hit = _hxr_history_cache.get(cache_key)
        if hit is not None and hit[0] == stat.st_mtime_ns and hit[1] == stat.st_size:
            return [dict(row) for row in hit[2]]

    rel = path.relative_to(run_dir).as_posix()
    try:
        with HexoRecordFile.open(path) as record_file:
            players = [_record_player_payload(player) for player in record_file.players]
            records = list(record_file.iter_records())
    except Exception:
        return []

    rows: list[dict[str, object]] = []
    epoch = _epoch_from_artifact_path(rel)
    source = _history_source(rel)
    players_by_role = _players_by_role(players)
    for index, record in enumerate(records):
        length = int(record.placements or len(record.action_ids))
        rows.append(
            {
                "path": rel,
                "record_index": index,
                "game_id": record.game_id,
                "status": record.status,
                "winner": record.winner,
                "winner_label": _winner_label(record.winner),
                "length": length,
                "actions": len(record.action_ids),
                "epoch": epoch,
                "source": source,
                "seed": record.seed,
                "players": players_by_role,
                "candidate_seat": _candidate_seat_from_game_id(record.game_id),
                "modified": stat.st_mtime,
                "modified_ns": stat.st_mtime_ns,
                "bytes": stat.st_size,
                "abort": _abort_payload(record.abort),
            }
        )
    with _training_cache_lock:
        _hxr_history_cache[cache_key] = (stat.st_mtime_ns, stat.st_size, [dict(row) for row in rows])
    return rows


def _hxr_record_count(path: Path, run_dir: Path) -> int:
    stat = _safe_stat(path)
    if stat is None or stat.st_size <= 0:
        return 0
    cache_key = str(path.resolve())
    with _training_cache_lock:
        hit = _hxr_history_cache.get(cache_key)
        if hit is not None and hit[0] == stat.st_mtime_ns and hit[1] == stat.st_size:
            return len(hit[2])
        count_hit = _hxr_count_cache.get(cache_key)
        if count_hit is not None and count_hit[0] == stat.st_mtime_ns and count_hit[1] == stat.st_size:
            return count_hit[2]

    try:
        with HexoRecordFile.open(path) as record_file:
            count = sum(1 for _ in record_file.iter_records())
    except Exception:
        return 0
    with _training_cache_lock:
        _hxr_count_cache[cache_key] = (stat.st_mtime_ns, stat.st_size, count)
    return count


def _is_loadable_history_path(rel: str) -> bool:
    return rel.split("/", 1)[0] in {"selfplay", "evaluation"}


def _history_filter_matches_all(*, winner: str, query_text: str) -> bool:
    return str(winner or "all").lower() in {"", "all"} and not str(query_text or "").strip()


def _history_row_matches(row: dict[str, object], *, winner: str, query_text: str) -> bool:
    normalized_winner = str(winner or "all").lower()
    if normalized_winner == "none":
        if row.get("winner") is not None:
            return False
    elif normalized_winner not in {"", "all"} and row.get("winner") != normalized_winner:
        return False

    query = str(query_text or "").strip().lower()
    if not query:
        return True
    players = row.get("players") if isinstance(row.get("players"), dict) else {}
    diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
    haystack = " ".join(
        str(value)
        for value in (
            row.get("game_id"),
            row.get("run"),
            row.get("path"),
            row.get("status"),
            row.get("source"),
            row.get("epoch"),
            row.get("seed"),
            row.get("winner_label"),
            row.get("length"),
            history_player_label(players.get("player0") if isinstance(players, dict) else None),
            history_player_label(players.get("player1") if isinstance(players, dict) else None),
            json.dumps(diagnostics, sort_keys=True) if diagnostics else "",
        )
        if value is not None
    ).lower()
    return query in haystack


def history_player_label(player: object) -> str:
    if not isinstance(player, dict):
        return "Unknown"
    return str(player.get("label") or player.get("kind") or "Unknown")


def _history_complete_sort_key(row: dict[str, object], sort: str) -> tuple[object, ...]:
    newest = (
        -float(row.get("modified") or 0.0),
        -int(row.get("epoch") or 0),
        str(row.get("run") or ""),
        str(row.get("path") or ""),
        -int(row.get("record_index") or 0),
    )
    if sort == "longest":
        return (-int(row.get("length") or row.get("actions") or 0),) + newest
    if sort == "shortest":
        return (int(row.get("length") or row.get("actions") or 0),) + newest
    if sort == "winner":
        return (str(row.get("winner_label") or _winner_label(row.get("winner"))),) + newest
    return newest


def _history_cursor_key(row: dict[str, object]) -> list[object]:
    return [
        row.get("run"),
        row.get("path"),
        int(row.get("record_index") or 0),
        int(row.get("modified_ns") or 0),
    ]


def _encode_history_cursor(key: list[object]) -> str:
    return json.dumps(key, separators=(",", ":"))


def _decode_history_cursor(cursor: str) -> list[object] | None:
    if not cursor:
        return None
    try:
        value = json.loads(cursor)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def _record_player_payload(player: object) -> dict[str, object]:
    role = str(getattr(player, "role", ""))
    label = getattr(player, "label", None)
    player_id = str(getattr(player, "player_id", role or "player"))
    kind = "manual"
    lowered = player_id.lower()
    if "sealbot" in lowered:
        kind = "sealbot-best" if "best" in lowered else "sealbot-current"
    elif "dense" in lowered:
        kind = "dense-cnn"
    return {
        "role": role,
        "kind": kind,
        "label": str(label or player_id),
        "player_id": player_id,
    }


def _history_diagnostics_brief(diagnostics: dict[str, object]) -> dict[str, object]:
    return {
        label: diagnostics[label]
        for label in ("selfplay", "evaluation")
        if label in diagnostics
    }


# ---------------------------------------------------------------------------
# History screen: per-epoch trend rows. _epoch_history merges every diagnostics
# JSON family (epoch_*.json pipeline results, dense_cnn.selfplay/evaluation
# epoch files, checkpoint stats) into one ascending row list; the *_summary
# helpers below each document their producer's real keys. Output key names are
# the app.js contract and must not change.
# ---------------------------------------------------------------------------


def _iter_training_files(run_dir: Path, *, suffix: str | None = None) -> list[Path]:
    files: list[Path] = []
    for root, dirs, names in os.walk(run_dir):
        dirs[:] = [
            name
            for name in dirs
            if name not in TRAINING_SCAN_EXCLUDED_DIRS and not name.startswith(".")
        ]
        root_path = Path(root)
        for name in names:
            if suffix is not None and not name.endswith(suffix):
                continue
            files.append(root_path / name)
    return files


def _diagnostics_by_epoch(run_dir: Path) -> dict[str, object]:
    by_epoch: dict[str, dict[str, object]] = {}
    diagnostics_dir = run_dir / "diagnostics"
    if not diagnostics_dir.exists():
        return by_epoch
    for path in sorted(diagnostics_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        epoch = _epoch_from_artifact_path(path.name)
        if epoch is None and isinstance(payload, dict) and payload.get("epoch") is not None:
            try:
                epoch = int(payload["epoch"])
            except (TypeError, ValueError):
                epoch = None
        if epoch is None:
            continue
        key = str(epoch)
        by_epoch.setdefault(key, {})
        label = _diagnostic_label(path.name)
        summary = _artifact_summary(payload)
        if summary:
            by_epoch[key][label] = {
                "path": f"diagnostics/{path.name}",
                "summary": summary,
            }
    return by_epoch


def _evaluation_history(run_dir: Path) -> list[dict[str, object]]:
    diagnostics_dir = run_dir / "diagnostics"
    if not diagnostics_dir.exists():
        return []
    rows: list[dict[str, object]] = []
    prefix = _diag_prefix(run_dir)
    for path in sorted(diagnostics_dir.glob(f"{prefix}.evaluation.epoch_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        epoch = _epoch_from_artifact_path(path.name)
        if epoch is None and payload.get("epoch") is not None:
            try:
                epoch = int(payload["epoch"])
            except (TypeError, ValueError):
                epoch = None
        stat = _safe_stat(path)
        rows.append(
            {
                "epoch": epoch,
                "status": payload.get("status"),
                "games": payload.get("games"),
                # shrimp omits `completed`; fall back to total games when it has
                # finished so the eval row still reads as complete.
                "completed": payload.get("completed")
                if payload.get("completed") is not None
                else (payload.get("games") if payload.get("status") == "completed" else None),
                "wins": payload.get("wins"),
                "losses": payload.get("losses"),
                # dense_cnn emits `mean_turns`; shrimp emits `mean_game_length`.
                "mean_turns": payload.get("mean_turns")
                if payload.get("mean_turns") is not None
                else payload.get("mean_game_length"),
                "path": f"diagnostics/{path.name}",
                "modified": stat.st_mtime if stat is not None else 0,
            }
        )
    rows.sort(key=lambda item: int(item.get("epoch") or 0))
    return rows


# Headline edge selector: which descriptive edges to surface as chips. The
# verdict's PRIMARY edge (vs prior champion) plus the two fixed anchors the
# standalone eval always reports against -- SealBot and the BC-prefit base.
_MULTISTAGE_HEADLINE_OPPONENTS = ("sealbot", "bc_prefit", "ep5")


def _multistage_headline_edge(edge: dict[str, object]) -> bool:
    """True for the edges worth promoting to the dashboard headline: the verdict's
    primary edge (vs champion) and the fixed-anchor edges (vs SealBot / BC-prefit)."""

    if edge.get("primary"):
        return True
    opponent = edge.get("opponent")
    return isinstance(opponent, str) and opponent.lower() in _MULTISTAGE_HEADLINE_OPPONENTS


def _multistage_eval_history(run_dir: Path) -> list[dict[str, object]]:
    """Per-epoch rows from the standalone shrimp multi-stage eval reports
    (``diagnostics/shrimp.multistage_eval.epoch_*.json``), ascending by epoch.

    Mirrors ``_evaluation_history``: lineage-gated on the shrimp prefix, reads
    each report via ``_read_json_file`` (graceful on missing/corrupt -> skipped),
    and emits a flat row per epoch carrying the verdict label/block, the
    SealBot-pinned rating table (``ratings.players`` + ``ratings.fit``), the
    headline edges (primary + SealBot/BC-prefit, winrate + CI), and the
    ``sealbot_winrate_ci95`` headline. The newest row is the latest verdict +
    rating table; the full list (each report's ``ratings.players``) is the
    Elo-over-epochs trajectory. Returns ``[]`` for non-shrimp lineages, a
    missing ``diagnostics/`` dir, or when no report files exist (opt-in eval)."""

    if _diag_prefix(run_dir) != "shrimp":
        return []
    diagnostics_dir = run_dir / "diagnostics"
    if not diagnostics_dir.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(diagnostics_dir.glob("shrimp.multistage_eval.epoch_*.json")):
        payload = _read_json_file(path)
        if not isinstance(payload, dict):
            continue
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        epoch = _coerce_epoch(meta.get("candidate_epoch"), path.name)
        ratings = payload.get("ratings") if isinstance(payload.get("ratings"), dict) else {}
        verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
        edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []
        roster = payload.get("roster") if isinstance(payload.get("roster"), dict) else {}
        # ALL edges, enriched for the history screen's eval-detail card. The
        # legacy headline consumers keep working (same keys, `headline` flags
        # the old subset); the new fields add the per-opponent W-L record, the
        # Elo point, the opponent's search profile (puct vs selfplay/gumbel),
        # and the game/visit provenance.
        def _edge_row(edge: dict[str, object]) -> dict[str, object]:
            prov = edge.get("provenance") if isinstance(edge.get("provenance"), dict) else {}
            wins_a = prov.get("physical_wins_a", prov.get("physical_wins_cand"))
            wins_b = prov.get("physical_wins_b", prov.get("physical_wins_sealbot"))
            return {
                "opponent": edge.get("opponent"),
                "role": edge.get("role"),
                "kind": edge.get("kind"),
                "primary": bool(edge.get("primary")),
                "headline": _multistage_headline_edge(edge),
                "decided": edge.get("decided"),
                "winrate": edge.get("winrate"),
                "winrate_ci95": edge.get("winrate_ci95"),
                "elo_point": edge.get("elo_point"),
                "wins_a": wins_a,
                "wins_b": wins_b,
                "eval_visits": prov.get("eval_visits"),
                "n_pairs": prov.get("n_pairs"),
                "opponent_search_profile": prov.get("opponent_search_profile"),
            }

        headline_edges = [
            _edge_row(edge) for edge in edges if isinstance(edge, dict)
        ]
        # Stage-C health: surfaces a PARTIAL eval (e.g. the concurrent
        # checkpoint pass dying on a CUDA error while SealBot already played)
        # instead of silently rendering the surviving edges as a full report.
        stages = payload.get("stages") if isinstance(payload.get("stages"), list) else []
        stage_c = next(
            (s for s in stages if isinstance(s, dict) and s.get("stage") == "C_deep"),
            {},
        )
        stage_health = {
            "status": stage_c.get("status"),
            "opponents_played": stage_c.get("opponents_played"),
            "allocation": stage_c.get("allocation"),
            "multi_checkpoint_error": stage_c.get("multi_checkpoint_error"),
            "sealbot_unavailable": stage_c.get("sealbot_unavailable"),
            "opponent_search_profiles": stage_c.get("opponent_search_profiles"),
        }
        players = ratings.get("players") if isinstance(ratings.get("players"), list) else []
        # Compact roster: the opponent labels/roles actually evaluated this epoch
        # plus the configured permanent anchors, so the frontend can surface a
        # dropped PERMANENT anchor (e.g. bc_prefit gone at ep35) as a muted "not in
        # roster" pill instead of silently dropping it (SEV-2). The anchor allowlist
        # is read from the report's own config -> no hard-coded duplicate.
        opp_list = roster.get("opponents") if isinstance(roster.get("opponents"), list) else []
        roster_opponents = [
            {"label": o.get("label"), "role": o.get("role"), "epoch": o.get("epoch")}
            for o in opp_list
            if isinstance(o, dict)
        ]
        config = meta.get("config") if isinstance(meta.get("config"), dict) else {}
        opp_cfg = config.get("opponents") if isinstance(config.get("opponents"), dict) else {}
        perm = opp_cfg.get("permanent_anchors") if isinstance(opp_cfg.get("permanent_anchors"), list) else []
        permanent_anchors = [
            str(entry[0])
            for entry in perm
            if isinstance(entry, (list, tuple)) and entry
        ]
        stat = _safe_stat(path)
        rows.append(
            {
                "epoch": epoch,
                "candidate_label": meta.get("candidate_label"),
                "anchor": meta.get("anchor") or ratings.get("anchor"),
                "verdict_label": verdict.get("label"),
                "verdict": verdict,
                "roster": {
                    "candidate": roster.get("candidate") if isinstance(roster.get("candidate"), dict) else {},
                    "champion": roster.get("champion") if isinstance(roster.get("champion"), dict) else {},
                    "sealbot": roster.get("sealbot"),
                    "opponents": roster_opponents,
                    "permanent_anchors": permanent_anchors,
                },
                "ratings": {
                    "anchor": ratings.get("anchor"),
                    "players": [p for p in players if isinstance(p, dict)],
                    "fit": ratings.get("fit") if isinstance(ratings.get("fit"), dict) else {},
                },
                "edges": headline_edges,
                "stage_health": stage_health,
                "pure_eval": meta.get("pure_eval"),
                "elapsed_seconds": meta.get("elapsed_seconds"),
                "full_search_visits": config.get("full_search_visits"),
                "sealbot_winrate_ci95": payload.get("sealbot_winrate_ci95"),
                "path": f"diagnostics/{path.name}",
                "modified": stat.st_mtime if stat is not None else 0,
            }
        )
    rows.sort(key=lambda item: int(item.get("epoch") or 0))
    return rows


def _eval_pool_summary(run_dir: Path) -> dict[str, object] | None:
    """Compact view of the rolling SealBot-pinned Bradley-Terry pool
    (``diagnostics/eval_pool.json``) for the Elo-trajectory chart.

    The pool is a single append-only file (not per-epoch), so its edges carry an
    ``epoch`` column spanning every candidate checkpoint. Lineage-gated like the
    other shrimp readers; returns ``None`` for non-shrimp runs and when the
    file is absent/corrupt (``_read_json_file`` -> ``None``). The heavy ``raw``
    block on each edge is dropped to keep the run payload small, EXCEPT the few
    integer ``physical_wins_*`` counts the W-L matrix needs (the top-level
    ``wins_a/wins_b`` are n_eff-weighted, so the true head-to-head must survive)."""

    if _diag_prefix(run_dir) != "shrimp":
        return None
    payload = _read_json_file(run_dir / "diagnostics" / "eval_pool.json")
    if not isinstance(payload, dict):
        return None
    raw_edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []
    edges: list[dict[str, object]] = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        # The heavy raw block (pentanomial, n_eff, virtual_batch_size, ...) is
        # dropped, but the TRUE physical head-to-head counts MUST survive: the
        # top-level wins_a/wins_b can be n_eff-weighted / overdispersion-reweighted
        # (fractional), so the W-L matrix in app.js prefers raw.physical_wins_* to
        # avoid rendering a distorted record (e.g. 3-3 instead of the real 5-5).
        # Keep only those few integer counts as a slim raw block.
        raw = edge.get("raw") if isinstance(edge.get("raw"), dict) else {}
        slim_raw: dict[str, object] = {}
        for key in ("physical_wins_a", "physical_wins_b",
                    "physical_wins_cand", "physical_wins_sealbot"):
            if key in raw:
                slim_raw[key] = raw.get(key)
        compact: dict[str, object] = {
            "epoch": edge.get("epoch"),
            "a": edge.get("a"),
            "b": edge.get("b"),
            "wins_a": edge.get("wins_a"),
            "wins_b": edge.get("wins_b"),
            "weight": edge.get("weight"),
            "kind": edge.get("kind"),
        }
        if slim_raw:
            compact["raw"] = slim_raw
        edges.append(compact)
    return {
        "format": payload.get("format"),
        "version": payload.get("version"),
        "anchor": payload.get("anchor"),
        "edges_total": len(edges),
        "edges": edges,
    }


def _epoch_history(
    run_dir: Path,
    live_status: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """One merged row per epoch (ascending) for the trends charts + epoch
    table: pipeline results from ``diagnostics/epoch_*.json``, self-play and
    evaluation summaries from the ``dense_cnn.*.epoch_*.json`` files (with
    .hxr-derived game-stat backfill), optional policy-target/progress
    overlays, and checkpoint file stats. Rows without a finished pipeline
    result get ``status: "partial"``.

    When ``live_status`` (a ``_training_live_status`` block) reports an
    actively-running epoch whose merged ``epoch_N.json`` has not landed yet,
    that epoch gets a provisional row: ``status: "in_progress"`` plus an
    ``in_progress`` block carrying the live phase/detail (and, during
    self-play, the live games/pos-s counters), so the epoch table can show the
    in-flight epoch instead of nothing for the first ~30 minutes of an epoch."""

    rows: dict[int, dict[str, object]] = {}
    diagnostics_dir = run_dir / "diagnostics"
    prefix = _diag_prefix(run_dir)

    if diagnostics_dir.exists():
        for path in sorted(diagnostics_dir.glob("epoch_*.json")):
            payload = _read_json_file(path)
            if not isinstance(payload, dict):
                continue
            result = payload.get("metadata", {}).get("result") if isinstance(payload.get("metadata"), dict) else None
            if not isinstance(result, dict):
                continue
            epoch = _coerce_epoch(result.get("epoch"), path.name)
            if epoch is None:
                continue
            row = rows.setdefault(epoch, {"epoch": epoch})
            row["status"] = payload.get("status")
            row["elapsed_seconds"] = payload.get("elapsed_seconds")
            _merge_epoch_result(row, result)

        for path in sorted(diagnostics_dir.glob(f"{prefix}.selfplay.epoch_*.json")):
            payload = _read_json_file(path)
            if not isinstance(payload, dict):
                continue
            epoch = _coerce_epoch(payload.get("epoch"), path.name)
            if epoch is None:
                continue
            row = rows.setdefault(epoch, {"epoch": epoch})
            selfplay_summary = _selfplay_epoch_summary(payload)
            # dense_cnn self-play diagnostics omit the game-length + outcome stats
            # hexgnn emits inline; backfill them display-side from the epoch's .hxr.
            _backfill_selfplay_game_stats(run_dir, epoch, selfplay_summary)
            row["selfplay"] = selfplay_summary

        for path in sorted(diagnostics_dir.glob(f"{prefix}.evaluation.epoch_*.json")):
            payload = _read_json_file(path)
            if not isinstance(payload, dict):
                continue
            epoch = _coerce_epoch(payload.get("epoch"), path.name)
            if epoch is None:
                continue
            row = rows.setdefault(epoch, {"epoch": epoch})
            row["evaluation"] = _evaluation_epoch_summary(payload)

    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        for path in sorted(checkpoints_dir.glob("epoch_*.pt")):
            epoch = _coerce_epoch(None, path.name)
            if epoch is None:
                continue
            row = rows.setdefault(epoch, {"epoch": epoch})
            stat = _safe_stat(path)
            row["checkpoint"] = {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": stat.st_size if stat is not None else 0,
                "modified": stat.st_mtime if stat is not None else 0,
            }

    _mark_in_flight_epoch(rows, live_status)

    for row in rows.values():
        if "status" not in row:
            row["status"] = "partial"
        # Per-head/total training loss band. hexgnn/hexgt attach a `buffer` block
        # (parsed from rl_train.log by their dashboard bridge); dense_cnn emits the
        # same numbers in the epoch's `training` block (training.loss +
        # training.loss_components). Surface them through the SAME selfplay.buffer
        # loss band app.js renders, when no producer buffer is present. A skipped/
        # untrained epoch (loss None) yields no buffer, so the band stays empty
        # (graceful) and the main row shows the training status ("skipped").
        training = row.get("training")
        selfplay = row.get("selfplay")
        # _selfplay_epoch_summary always carries a "buffer" key (None for dense_cnn,
        # the bridge dict for hexgnn/hexgt), so guard on falsy — only synthesize when
        # there is no producer buffer; hexgnn's real buffer (truthy) always wins.
        if isinstance(training, dict) and isinstance(selfplay, dict) and not selfplay.get("buffer"):
            loss_buffer = _loss_buffer_from_training(training)
            if loss_buffer:
                selfplay["buffer"] = loss_buffer
    return [rows[key] for key in sorted(rows)]


def _mark_in_flight_epoch(
    rows: dict[int, dict[str, object]],
    live_status: dict[str, object] | None,
) -> None:
    """Synthesize/annotate the provisional row for the CURRENTLY RUNNING epoch.

    Only the live epoch from events.jsonl is eligible, and only while its merged
    ``epoch_N.json`` result has not landed (the row is absent or file-partial) --
    historical partial rows from old crashes are never touched. ``stalled`` runs
    (supervisor down/halted mid-epoch) keep their provisional row so the epoch
    still shows, with the stalled phase making the state honest."""

    if not isinstance(live_status, dict):
        return
    epoch = live_status.get("current_epoch")
    stage_status = str(live_status.get("stage_status") or "")
    if not isinstance(epoch, int) or stage_status not in ("running", "stalled"):
        return
    row = rows.get(epoch)
    if row is not None and "status" in row:
        # A finished pipeline result (completed/failed epoch_N.json) already
        # merged for this epoch -> nothing in flight.
        return
    row = rows.setdefault(epoch, {"epoch": epoch})
    phase = str(live_status.get("sub_phase") or "") or (
        "stalled" if stage_status == "stalled" else "running"
    )
    in_progress: dict[str, object] = {"phase": phase}
    detail = live_status.get("sub_phase_detail")
    if detail is not None:
        in_progress["detail"] = detail
    progress = live_status.get("phase_progress")
    if isinstance(progress, dict):
        in_progress["progress"] = progress
    selfplay_live = live_status.get("selfplay_live")
    if (
        phase == "self-play"
        and isinstance(selfplay_live, dict)
        and selfplay_live.get("epoch") == epoch
    ):
        in_progress["selfplay_live"] = selfplay_live
    row["status"] = "in_progress"
    row["in_progress"] = in_progress


def _loss_buffer_from_training(training: dict[str, object]) -> dict[str, object]:
    """Build the `selfplay.buffer` loss block app.js renders (loss_total / loss_policy
    / loss_value / loss_opp / loss_stvalue_<h>) from a dense_cnn epoch `training`
    result. dense_cnn's trainer returns the weighted total (`loss`) plus the UNWEIGHTED
    per-head components (`loss_components`: policy, value, opp_policy, stvalue_<h>).
    hexgnn/hexgt feed the identical band via their bridge `buffer`; this surfaces the
    dense_cnn lineage's losses through the SAME panel. Returns {} when there is no
    numeric total loss (e.g. a skipped/untrained epoch) so the band degrades gracefully
    rather than breaking."""
    total = _optional_float(training.get("loss"))
    if total is None:
        return {}
    out: dict[str, object] = {"loss_total": total}
    components = training.get("loss_components")
    if isinstance(components, dict):
        # shrimp's per-head components are normalized into this same dict shape by
        # _training_epoch_summary, so the mapping below covers both lineages. The
        # moves_left and cell_q (per-cell action-value) heads are shrimp-only; the
        # epoch card renders both as chips when present.
        for src, dst in (
            ("policy", "loss_policy"),
            ("value", "loss_value"),
            ("opp_policy", "loss_opp"),
            ("moves_left", "loss_moves_left"),
            ("cell_q", "loss_cell_q"),
        ):
            value = _optional_float(components.get(src))
            if value is not None:
                out[dst] = value
        for key, raw in components.items():
            if isinstance(key, str) and key.startswith("stvalue_"):
                value = _optional_float(raw)
                if value is not None:
                    out[f"loss_{key}"] = value  # stvalue_1 -> loss_stvalue_1 (app.js renders stv<h>)
    return out


def _ms_candidate_elo(row: dict[str, object]) -> float | None:
    """The candidate checkpoint's pooled Elo from one multi-stage report row.

    Prefers the verdict's named candidate (``verdict.primary.candidate``); else
    the highest-Elo non-anchor node. Mirrors the frontend ``msCandidatePlayer``
    selection so the health chip and the rating table agree. ``None`` when the
    rating table is absent/degraded."""

    ratings = row.get("ratings") if isinstance(row.get("ratings"), dict) else {}
    players = ratings.get("players") if isinstance(ratings.get("players"), list) else []
    players = [p for p in players if isinstance(p, dict)]
    if not players:
        return None
    verdict = row.get("verdict") if isinstance(row.get("verdict"), dict) else {}
    primary = verdict.get("primary") if isinstance(verdict.get("primary"), dict) else {}
    want = primary.get("candidate")
    if want:
        named = next((p for p in players if p.get("label") == want), None)
        if named is not None:
            return _optional_float(named.get("elo"))
    non_anchor = [p for p in players if not p.get("is_anchor")]
    pool = non_anchor or players
    best = max(pool, key=lambda p: (_optional_float(p.get("elo")) or float("-inf")))
    return _optional_float(best.get("elo"))


def _ms_sealbot_winrate(row: dict[str, object]) -> float | None:
    """The descriptive SealBot zero-point winrate for one report row.

    Prefers the SealBot headline edge's ``winrate``; falls back to the midpoint
    of the report's ``sealbot_winrate_ci95`` headline. ``None`` when SealBot did
    not run (the edge/CI is absent)."""

    edges = row.get("edges") if isinstance(row.get("edges"), list) else []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        opp = str(edge.get("opponent") or edge.get("role") or "").lower()
        if opp == "sealbot":
            wr = _optional_float(edge.get("winrate"))
            if wr is not None:
                return wr
    ci = row.get("sealbot_winrate_ci95")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        lo = _optional_float(ci[0])
        hi = _optional_float(ci[1])
        if lo is not None and hi is not None:
            return (lo + hi) / 2.0
    return None


def _learning_health(
    epoch_history: list[dict[str, object]],
    evaluation_history: list[dict[str, object]],
    live_status: dict[str, object],
    multistage_eval_history: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Heuristic run-health verdict for the History status band: a coarse
    ``status`` ladder (collecting -> ok -> improving / watch -> intervene)
    plus human-readable ``messages``, derived from loss trend, SealBot
    survival/wins, self-play speed, D6 previews, and replay composition.
    Display-side triage only — thresholds here are owner judgment calls, not
    training-side gates.

    For shrimp runs the eval signal is the standalone multi-stage report
    (``multistage_eval_history``: pooled Bradley-Terry verdict + candidate Elo +
    descriptive SealBot zero-point winrate), NOT the all-null wrapper
    ``evaluation_history`` (which only the dense_cnn lineage populates with
    ``mean_turns``/``wins``). When ``multistage_eval_history`` is non-empty the
    eval-health branch is driven from it and the legacy turns-based "no SealBot
    eval yet" / "D6 missing" messages are suppressed."""

    ms_history = [
        row for row in (multistage_eval_history or [])
        if isinstance(row, dict)
    ]
    has_multistage = bool(ms_history)

    completed = [row for row in epoch_history if row.get("status") == "completed"]
    latest = completed[-1] if completed else (epoch_history[-1] if epoch_history else {})
    latest_epoch = int(latest.get("epoch") or 0)
    latest_training = latest.get("training") if isinstance(latest.get("training"), dict) else {}
    first_training = completed[0].get("training") if completed and isinstance(completed[0].get("training"), dict) else {}
    latest_loss = _optional_float(latest_training.get("loss"))
    first_loss = _optional_float(first_training.get("loss"))

    evals = [
        item
        for item in evaluation_history
        if _optional_float(item.get("mean_turns")) is not None
    ]
    latest_eval = evals[-1] if evals else {}
    first_eval = evals[0] if evals else {}
    best_eval = max(evals, key=lambda item: (_optional_float(item.get("mean_turns")) or 0.0, int(item.get("wins") or 0)), default={})
    latest_turns = _optional_float(latest_eval.get("mean_turns"))
    first_turns = _optional_float(first_eval.get("mean_turns"))
    best_turns = _optional_float(best_eval.get("mean_turns"))
    latest_wins = int(latest_eval.get("wins") or 0) if latest_eval else 0
    latest_games = int(latest_eval.get("games") or 0) if latest_eval else 0
    latest_selfplay = latest.get("selfplay") if isinstance(latest.get("selfplay"), dict) else {}
    latest_d6 = latest.get("d6") if isinstance(latest.get("d6"), dict) else {}
    latest_source_summary = (
        latest_training.get("source_summary")
        if isinstance(latest_training.get("source_summary"), dict)
        else {}
    )
    latest_source_counts = (
        latest_source_summary.get("source_counts")
        if isinstance(latest_source_summary.get("source_counts"), dict)
        else {}
    )
    latest_classical_fraction = _source_fraction(latest_source_counts, "classical")
    latest_policy_imitation = (
        latest_training.get("policy_imitation")
        if isinstance(latest_training.get("policy_imitation"), dict)
        else {}
    )
    latest_policy_overall = (
        latest_policy_imitation.get("overall")
        if isinstance(latest_policy_imitation.get("overall"), dict)
        else {}
    )
    latest_policy_top1 = _optional_float(latest_policy_overall.get("top1_accuracy"))
    latest_policy_target_mass = _optional_float(latest_policy_overall.get("mean_target_mass"))

    messages: list[str] = []
    status = "collecting"
    if latest_epoch > 0:
        status = "ok"
    if latest_loss is not None and first_loss is not None and latest_loss < first_loss:
        messages.append(f"Training loss improved from {first_loss:.3f} to {latest_loss:.3f}.")
    elif latest_loss is not None:
        messages.append(f"Latest training loss is {latest_loss:.3f}.")

    # Multi-stage eval fields, populated from the standalone shrimp report when
    # present (so the status-band eval chip renders a real value, not "--").
    latest_verdict: str | None = None
    latest_cand_elo: float | None = None
    latest_sealbot_winrate: float | None = None
    latest_eval_epoch: int | None = None

    if has_multistage:
        # Shrimp lineage: drive eval-health off the multi-stage Bradley-Terry
        # report. NEVER emit the legacy turns-based "no SealBot eval yet" line —
        # the eval has run (one report per evaluated epoch).
        latest_ms = ms_history[-1]
        first_ms = ms_history[0]
        _ep = _optional_float(latest_ms.get("epoch"))
        latest_eval_epoch = int(_ep) if _ep is not None else None
        verdict_block = latest_ms.get("verdict") if isinstance(latest_ms.get("verdict"), dict) else {}
        latest_verdict = (
            latest_ms.get("verdict_label")
            or (verdict_block.get("label") if isinstance(verdict_block, dict) else None)
        )
        latest_verdict = str(latest_verdict).upper() if latest_verdict else None
        latest_cand_elo = _ms_candidate_elo(latest_ms)
        latest_sealbot_winrate = _ms_sealbot_winrate(latest_ms)
        first_sealbot_winrate = _ms_sealbot_winrate(first_ms)

        wr_txt = (
            f" — SealBot winrate {latest_sealbot_winrate * 100.0:.0f}%"
            if latest_sealbot_winrate is not None
            else ""
        )
        elo_txt = (
            f", candidate {latest_cand_elo:+.0f} Elo"
            if latest_cand_elo is not None
            else ""
        )
        ep_txt = f" (epoch {latest_eval_epoch})" if latest_eval_epoch else ""

        if latest_verdict == "PROMOTE":
            status = "improving"
            messages.append(f"Eval verdict PROMOTE{ep_txt}{elo_txt}{wr_txt}.")
        elif latest_verdict == "REGRESS":
            # The ep5 startup REGRESS is a known artifact (weakest net); only the
            # very first eval is treated as a watch rather than an intervene.
            status = "watch" if len(ms_history) <= 1 else "intervene"
            messages.append(f"Eval verdict REGRESS{ep_txt}{elo_txt}{wr_txt}.")
        else:
            # INCONCLUSIVE (the steady state for this resolution floor): read the
            # descriptive SealBot zero-point winrate as the real progress signal.
            if (
                latest_sealbot_winrate is not None
                and first_sealbot_winrate is not None
                and len(ms_history) >= 2
                and latest_sealbot_winrate - first_sealbot_winrate > 0.05
            ):
                status = "improving"
                messages.append(
                    f"Eval INCONCLUSIVE{ep_txt} but SealBot winrate is rising "
                    f"({first_sealbot_winrate * 100.0:.0f}% → {latest_sealbot_winrate * 100.0:.0f}%)"
                    f"{elo_txt}."
                )
            elif latest_sealbot_winrate is not None and latest_sealbot_winrate >= 0.5:
                status = "ok"
                messages.append(
                    f"Eval INCONCLUSIVE{ep_txt}{elo_txt}{wr_txt} (tripwire only, not a fine-edge test)."
                )
            else:
                status = "watch"
                messages.append(
                    f"Eval INCONCLUSIVE{ep_txt}{elo_txt}{wr_txt}; SealBot winrate flat/low."
                )
    elif latest_turns is None:
        status = "collecting"
        messages.append("No SealBot evaluation result yet for the completed epochs.")
    else:
        delta = latest_turns - (first_turns if first_turns is not None else latest_turns)
        if latest_wins > 0:
            status = "improving"
            messages.append(f"Latest SealBot eval has {latest_wins}/{latest_games} wins.")
        elif delta > 3.0:
            status = "improving"
            messages.append(f"SealBot survival improved by {delta:.1f} turns.")
        elif len(evals) >= 2:
            status = "watch"
            messages.append(f"SealBot survival is flat at {latest_turns:.1f} turns.")
        else:
            messages.append(f"Initial SealBot survival is {latest_turns:.1f} turns.")
        if latest_epoch >= 6 and latest_wins == 0 and (best_turns or 0.0) <= 30.0:
            status = "intervene"
            messages.append("Epoch 6+ is still under 30 turns with no wins; inspect games and training targets before continuing blindly.")
        elif status == "watch":
            messages.append("Keep training for now, but inspect previews if this remains flat near epoch 6.")

    exact_128 = abs((_optional_float(latest_selfplay.get("mcts_sims_per_searched_position")) or 0.0) - 128.0) < 1.0e-6
    speed = _optional_float(latest_selfplay.get("search_positions_per_second"))
    if speed is not None and speed >= 128.0 and exact_128:
        messages.append(f"Self-play speed is healthy at {speed:.0f} pos/s with exact 128 sims.")
    elif speed is not None:
        status = "watch" if status != "intervene" else status
        messages.append(f"Self-play speed needs attention: {speed:.0f} pos/s, exact128={exact_128}.")

    d6_mode = str(latest_d6.get("mode") or "")
    d6_preview = latest_d6.get("preview_symmetries") if isinstance(latest_d6.get("preview_symmetries"), list) else []
    if "random_per_training_expansion" in d6_mode or d6_preview:
        messages.append("D6 training augmentation previews are present.")
    elif latest_epoch > 0 and not has_multistage:
        # D6 preview is a dense_cnn-lineage concern; on shrimp runs the absent
        # preview is irrelevant noise, so only flag it for non-multistage runs.
        status = "watch" if status != "intervene" else status
        messages.append("D6 augmentation preview is missing for the latest epoch.")

    if latest_classical_fraction is not None:
        messages.append(f"Training window classical replay is {latest_classical_fraction * 100.0:.0f}%.")
        if latest_epoch >= 7 and latest_classical_fraction < 0.5:
            status = "watch" if status != "intervene" else status
            messages.append("Classical replay is below the bootstrap floor; inspect sample selection.")
    if latest_policy_target_mass is not None and latest_policy_top1 is not None:
        messages.append(f"Policy imitation top-1 is {latest_policy_top1 * 100.0:.0f}% with {latest_policy_target_mass * 100.0:.1f}% target mass.")

    return {
        "status": status,
        "latest_epoch": latest_epoch or None,
        "current_stage": live_status.get("stage"),
        "latest_loss": latest_loss,
        "loss_delta_from_first": (latest_loss - first_loss) if latest_loss is not None and first_loss is not None else None,
        "latest_eval_mean_turns": latest_turns,
        "best_eval_mean_turns": best_turns,
        "eval_delta_from_first": (latest_turns - first_turns) if latest_turns is not None and first_turns is not None else None,
        "latest_eval_wins": latest_wins,
        "latest_eval_games": latest_games,
        # Multi-stage eval (shrimp) — None for dense_cnn lineages. These let the
        # status-band eval chip render a real value instead of "--".
        "latest_verdict": latest_verdict,
        "latest_cand_elo": latest_cand_elo,
        "latest_sealbot_winrate": latest_sealbot_winrate,
        "latest_eval_epoch": latest_eval_epoch,
        "latest_selfplay_pos_s": speed,
        "latest_exact_128": exact_128,
        "latest_classical_fraction": latest_classical_fraction,
        "latest_policy_top1": latest_policy_top1,
        "latest_policy_target_mass": latest_policy_target_mass,
        "d6_preview_symmetries": d6_preview,
        "messages": messages,
    }


def _merge_epoch_result(row: dict[str, object], result: dict[str, object]) -> None:
    if isinstance(result.get("selfplay"), dict):
        row["selfplay"] = _selfplay_epoch_summary(result["selfplay"])
    if isinstance(result.get("training"), dict):
        row["training"] = _training_epoch_summary(result["training"])
    if isinstance(result.get("evaluation"), dict):
        row["evaluation"] = _evaluation_epoch_summary(result["evaluation"])
    if isinstance(result.get("checkpoint"), dict):
        checkpoint = result["checkpoint"]
        row["checkpoint"] = {
            "path": _run_relative_or_value(checkpoint.get("checkpoint_path")),
            "name": checkpoint.get("name"),
        }
    if isinstance(result.get("samples"), dict):
        samples = result["samples"]
        selection = samples.get("selection") if isinstance(samples.get("selection"), dict) else {}
        finalize = samples.get("finalize") if isinstance(samples.get("finalize"), dict) else {}
        row["samples"] = {
            "buffer_count": selection.get("sample_count") or finalize.get("buffer_count"),
            "window_size": selection.get("window_size"),
            "compressed_bytes": finalize.get("compressed_bytes"),
        }
    if isinstance(result.get("symmetries"), dict):
        metadata = result["symmetries"].get("metadata") if isinstance(result["symmetries"].get("metadata"), dict) else {}
        row["d6"] = {
            "mode": metadata.get("mode"),
            "group_size": metadata.get("d6_group_size"),
            "sample_count": metadata.get("sample_count"),
        }


def _selfplay_epoch_summary(payload: dict[str, object]) -> dict[str, object]:
    # Producer: dense_cnn/selfplay.py generate_selfplay_epoch (the summary dict).
    # Real keys: status, epoch, requested_games, games_started, completed_games,
    # truncated_games, games_finished, raw_samples, effective_samples,
    # searched_positions, mcts_simulations, search_visits, selfplay_npz_files,
    # record_path, elapsed_seconds, mcts_search_elapsed_seconds,
    # search_positions_per_second, positions_per_second, active_games,
    # mcts_virtual_batch_size, mcts_diagnostics, npz_writes.
    # Output key names are consumed by app.js and must stay unchanged; only the
    # source key each is populated FROM changes.
    completed_games = payload.get("completed_games")
    truncated_games = payload.get("truncated_games")

    # app.js reads selfplay.games; populate from games_finished, then fall back.
    games = payload.get("games_finished")
    if games is None:
        if completed_games is not None or truncated_games is not None:
            games = (completed_games or 0) + (truncated_games or 0)
        else:
            games = payload.get("games_started")

    # app.js reads selfplay.samples_added; dense_cnn emits effective_samples,
    # shrimp emits rows_written (samples added to the replay buffer).
    samples_added = payload.get("effective_samples")
    if samples_added is None:
        samples_added = payload.get("raw_samples")
    if samples_added is None:
        samples_added = payload.get("rows_written")

    # No producer key for per-searched-position sims; derive when both present.
    # dense_cnn emits searched_positions; shrimp emits total_decisions (decision
    # points evaluated) — the same quantity for this display.
    mcts_simulations = payload.get("mcts_simulations")
    searched_positions = payload.get("searched_positions")
    if searched_positions is None:
        searched_positions = payload.get("total_decisions")
    mcts_sims_per_searched_position: float | None = None
    sims = _optional_float(mcts_simulations)
    searched = _optional_float(searched_positions)
    if sims is not None and searched is not None and searched > 0.0:
        mcts_sims_per_searched_position = sims / searched

    return {
        "status": payload.get("status"),
        "games": games,
        "completed_games": completed_games,
        "truncated_games": truncated_games,
        "winner_counts": None,  # no producer key (separate worker handles display)
        "lengths": None,  # no producer key (separate worker handles display)
        "samples_added": samples_added,
        "searched_positions": searched_positions,
        "mcts_simulations": mcts_simulations,
        "search_positions_per_second": payload.get("search_positions_per_second"),
        "mcts_sims_per_searched_position": mcts_sims_per_searched_position,
        "elapsed_seconds": payload.get("elapsed_seconds"),
        # Game-length stats. hexgnn/hexgt emit these inline; dense_cnn does not, so
        # _epoch_history backfills any None from the epoch's .hxr records (display
        # side). Producer-emitted values pass through unchanged. None values are
        # omitted client-side, so a run with neither stays unaffected.
        # shrimp emits mean_game_length / p90_game_length inline (different names,
        # and p90 rather than p95); fall back to them so its self-play row carries
        # length stats without the .hxr backfill. dense_cnn keys win when present.
        "game_length_mean": payload.get("game_length_mean")
        if payload.get("game_length_mean") is not None
        else payload.get("mean_game_length"),
        "game_length_median": payload.get("game_length_median"),
        "game_length_max": payload.get("game_length_max"),
        "game_length_stdev": payload.get("game_length_stdev"),
        "game_length_p95": payload.get("game_length_p95")
        if payload.get("game_length_p95") is not None
        else payload.get("p90_game_length"),
        # Outcome distribution. Same backfill story as the lengths above — derived
        # from finished .hxr games when the producer omits them.
        "win_p0_fraction": payload.get("win_p0_fraction"),
        "win_p1_fraction": payload.get("win_p1_fraction"),
        "draw_fraction": payload.get("draw_fraction"),
        "decisive_fraction": payload.get("decisive_fraction"),
        # mean_abs_value (mean |value target|) needs the NPZ value labels / self-play
        # internals, not the .hxr — passed through when emitted (hexgnn), else None.
        "mean_abs_value": payload.get("mean_abs_value"),
        # Replay-buffer + per-head training-loss + calibration stats (nested object,
        # None for producers that don't emit it — e.g. dense_cnn runs — so the
        # frontend just omits the detail band). The dashboard bridge attaches this
        # to the published selfplay payload; without this passthrough the per-head
        # Losses group never reaches epochProgressDetail in app.js.
        "buffer": payload.get("buffer"),
    }


def _selfplay_game_stats_from_records(run_dir: Path, epoch: int) -> dict[str, object]:
    """Derive game-length + win-fraction stats for one self-play epoch from its
    ``.hxr`` game records, DISPLAY-SIDE.

    hexgnn/hexgt self-play diagnostics carry these stats inline; dense_cnn's do
    not. The dashboard already reads the same ``.hxr`` for the History panel via
    ``_hxr_base_rows`` (memoized by mtime/size), so this reuse is cheap and adds no
    new file I/O on a warm cache. Returns ``{}`` when the epoch record is absent or
    has no completed games. Only stats that are honestly derivable from finished
    game records are computed here — MCTS-internal diversity (visit/prior entropy,
    candidate counts, opening/move2 entropy, forced-move fraction) and value-target
    stats (mean_abs_value) need self-play internals/NPZ shards and stay absent."""

    path = run_dir / "selfplay" / f"epoch_{epoch:06d}.hxr"
    if not path.is_file():
        return {}
    rows = _hxr_base_rows(path, run_dir)
    completed = [row for row in rows if str(row.get("status")) == "completed"]
    if not completed:
        return {}
    lengths = [
        int(row.get("length") or row.get("actions") or 0)
        for row in completed
    ]
    lengths = [value for value in lengths if value > 0]
    total = len(completed)
    p0_wins = sum(1 for row in completed if row.get("winner") == "player0")
    p1_wins = sum(1 for row in completed if row.get("winner") == "player1")
    draws = total - p0_wins - p1_wins
    stats: dict[str, object] = {
        "win_p0_fraction": p0_wins / total,
        "win_p1_fraction": p1_wins / total,
        "draw_fraction": draws / total,
        "decisive_fraction": (p0_wins + p1_wins) / total,
    }
    if lengths:
        ordered = sorted(lengths)
        idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
        stats.update(
            {
                "game_length_mean": statistics.fmean(lengths),
                "game_length_median": statistics.median(lengths),
                "game_length_max": max(lengths),
                "game_length_stdev": statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
                "game_length_p95": ordered[idx],
            }
        )
    return stats


def _backfill_selfplay_game_stats(run_dir: Path, epoch: int, selfplay: dict[str, object]) -> None:
    """Fill in any game-stat field the self-play diagnostics left ``None`` with the
    ``.hxr``-derived value, in place. Producer-emitted values always win; this only
    populates gaps (so dense_cnn rows gain the stats hexgnn emits natively, while
    hexgnn rows are untouched). Memoized record stats are computed at most once per
    backfilled epoch per request."""

    if not isinstance(selfplay, dict):
        return
    derivable = (
        "game_length_mean",
        "game_length_median",
        "game_length_max",
        "game_length_stdev",
        "game_length_p95",
        "win_p0_fraction",
        "win_p1_fraction",
        "draw_fraction",
        "decisive_fraction",
    )
    if all(selfplay.get(key) is not None for key in derivable):
        return
    derived = _selfplay_game_stats_from_records(run_dir, epoch)
    if not derived:
        return
    for key, value in derived.items():
        if selfplay.get(key) is None:
            selfplay[key] = value


def _training_epoch_summary(payload: dict[str, object]) -> dict[str, object]:
    # Producer: dense_cnn/trainer.py DenseCNNTrainer.train_passes return dict.
    # Real keys: status, epoch, passes, generic_passes_requested, steps, samples,
    # batch_size, loss, loss_components, validation, elapsed_seconds,
    # samples_per_second, train_state. The trainer's return dict DOES carry the
    # unweighted per-head `loss_components` (policy/value/opp_policy/stvalue_*) — pass
    # it through so the per-head Loss band renders (the optional policy_targets overlay
    # still augments source_summary/policy_imitation later in _epoch_history).
    # shrimp's trainer emits the weighted total as `loss_total` and the per-head
    # losses FLAT at top level (loss_policy/loss_value/loss_opp_policy/loss_stvalue_*/
    # loss_moves_left) rather than dense_cnn's `loss` + nested `loss_components`.
    # Normalize to the dense_cnn shape here so every downstream reader (the loss band
    # via _loss_buffer_from_training, _learning_health, app.js) stays unchanged.
    loss = payload.get("loss")
    if loss is None:
        loss = payload.get("loss_total")
    loss_components = payload.get("loss_components")
    if not isinstance(loss_components, dict):
        flat = {
            "policy": payload.get("loss_policy"),
            "value": payload.get("loss_value"),
            "opp_policy": payload.get("loss_opp_policy"),
            "moves_left": payload.get("loss_moves_left"),
            "cell_q": payload.get("loss_cell_q"),
        }
        for key, value in payload.items():
            if isinstance(key, str) and key.startswith("loss_stvalue_"):
                flat[key[len("loss_") :]] = value  # loss_stvalue_2 -> stvalue_2
        flat = {key: value for key, value in flat.items() if value is not None}
        loss_components = flat or None
    return {
        "status": payload.get("status"),
        "loss": loss,
        "loss_components": loss_components,  # per-head (dense_cnn nested / shrimp flat)
        "source_summary": None,  # no producer key (overlaid from policy_targets file)
        "policy_imitation": None,  # no producer key (overlaid from policy_targets file)
        "steps": payload.get("steps"),
        # dense_cnn emits `samples`; shrimp emits `window_rows` (training window size).
        "samples": payload.get("samples")
        if payload.get("samples") is not None
        else payload.get("window_rows"),
        "batch_size": payload.get("batch_size"),
        "samples_per_second": payload.get("samples_per_second"),
        # dense_cnn emits `elapsed_seconds`; shrimp emits `seconds`.
        "elapsed_seconds": payload.get("elapsed_seconds")
        if payload.get("elapsed_seconds") is not None
        else payload.get("seconds"),
    }


def _evaluation_epoch_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "status": payload.get("status"),
        "games": payload.get("games"),
        # shrimp omits `completed`; treat all games as completed once the eval is
        # done so the row reads as finished.
        "completed": payload.get("completed")
        if payload.get("completed") is not None
        else (payload.get("games") if payload.get("status") == "completed" else None),
        "wins": payload.get("wins"),
        "losses": payload.get("losses"),
        # dense_cnn emits `mean_turns`; shrimp emits `mean_game_length`.
        "mean_turns": payload.get("mean_turns")
        if payload.get("mean_turns") is not None
        else payload.get("mean_game_length"),
    }


def _coerce_epoch(value: object, path: str) -> int | None:
    try:
        if value is not None:
            return int(value)
    except (TypeError, ValueError):
        pass
    return _epoch_from_artifact_path(path)


def _source_fraction(source_counts: object, token: str) -> float | None:
    if not isinstance(source_counts, dict):
        return None
    total = 0
    matching = 0
    needle = token.lower()
    for key, value in source_counts.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        total += count
        if needle in str(key).lower():
            matching += count
    return (matching / total) if total > 0 else None


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _run_relative_or_value(value: object) -> object:
    if value is None:
        return None
    try:
        path = Path(str(value))
        if path.parts:
            parts = path.parts
            if "runs" in parts:
                index = parts.index("runs")
                return Path(*parts[index + 2 :]).as_posix()
    except Exception:
        pass
    return value


def _diagnostic_label(name: str) -> str:
    lowered = name.lower()
    if "evaluation" in lowered:
        return "evaluation"
    if "selfplay" in lowered:
        return "selfplay"
    if lowered.startswith("epoch_"):
        return "epoch"
    return Path(name).stem


def _history_source(path: str) -> str:
    parts = Path(path).parts
    if parts:
        return str(parts[0])
    return "history"


def _epoch_from_artifact_path(path: str) -> int | None:
    match = re.search(r"epoch[_-](\d+)", path)
    if not match:
        return None
    return int(match.group(1))


def _winner_label(winner: object | None) -> str:
    if winner == "player0":
        return "P0"
    if winner == "player1":
        return "P1"
    return "None"


def _players_by_role(players: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    by_role = {
        str(player.get("role")): player
        for player in players
        if player.get("role") in PLAYER_ROLES
    }
    for role in PLAYER_ROLES:
        by_role.setdefault(role, {"role": role, "kind": "unknown", "label": role, "player_id": role})
    return by_role


def _abort_payload(abort: object | None) -> object | None:
    if abort is None:
        return None
    return {
        "stage": getattr(abort, "stage", None),
        "exception_type": getattr(abort, "exception_type", None),
        "message": getattr(abort, "message", None),
    }


# ---------------------------------------------------------------------------
# History screen: live status (the fast ~2.5s poll tier behind
# /api/training/live). Cheap reads only: the events.jsonl tail, the watchdog
# jsonl tail, two small JSON files, and a few stat/glob probes.
# ---------------------------------------------------------------------------


def _training_live_status(run_dir: Path) -> dict[str, object]:
    """Live ``status`` block: active stage/epoch from events.jsonl, watchdog +
    calibration + self-play live summaries when their files exist, and the
    derived within-epoch ``sub_phase`` (see _derive_sub_phase). Also embedded
    in the full run payload so both poll tiers share one shape."""

    diagnostics = run_dir / "diagnostics"
    prefix = _diag_prefix(run_dir)
    events = _stage_status_from_events(diagnostics / "events.jsonl")
    watchdog = _read_last_jsonl(diagnostics / "resource_watchdog.jsonl")
    # shrimp does not emit a `<prefix>.performance_calibration.json` (it writes a
    # differently-shaped `calibrate_performance.json`), so the calibration block is
    # omitted for shrimp runs (honest gap, not faked). It DOES emit the live
    # `<prefix>.selfplay.live.json` (shrimp/selfplay.py _write_live, every ~3s
    # during self-play), which feeds the live-progress + sub-phase blocks below.
    calibration = _read_json_file(diagnostics / f"{prefix}.performance_calibration.json")
    selfplay_live = _read_json_file(diagnostics / f"{prefix}.selfplay.live.json")
    training_progress = _latest_training_progress(diagnostics)
    bootstrap_progress = _latest_bootstrap_training_progress(run_dir)
    trainer_command = ""
    if isinstance(watchdog, dict) and isinstance(watchdog.get("trainer"), dict):
        trainer_command = str(watchdog["trainer"].get("command_line") or "")
    status: dict[str, object] = {
        "stage": events.get("stage") or "unknown",
        "stage_status": events.get("status") or "unknown",
        "current_epoch": events.get("epoch"),
        "last_event": events.get("last_event"),
    }
    if "bootstrap_dense_cnn_classical.py" in trainer_command and isinstance(bootstrap_progress, dict):
        training_progress = bootstrap_progress
        status.update(
            {
                "stage": "classical_bootstrap_prefit",
                "stage_status": bootstrap_progress.get("status") or "running",
                "current_epoch": None,
                "bootstrap": {
                    "status": bootstrap_progress.get("status"),
                    "output_dir": bootstrap_progress.get("output_dir"),
                    "path": bootstrap_progress.get("path"),
                },
            }
        )
    if isinstance(watchdog, dict):
        status["watchdog"] = _watchdog_summary(watchdog)
    if isinstance(calibration, dict):
        status["calibration"] = _calibration_summary(calibration)
    if isinstance(selfplay_live, dict):
        status["selfplay_live"] = _selfplay_live_summary(selfplay_live)
    if isinstance(training_progress, dict):
        status["training_progress"] = _training_progress_summary(training_progress)
    # Supervisor lifecycle (supervised runs only: supervisor.log present).
    # A tripped breaker writes supervisor_halted.flag and silently blocks
    # restarts while events.jsonl still ends in an unfinished stage_started,
    # so without this the dashboard says "running" forever over a dead trainer.
    supervisor = _supervisor_summary(run_dir)
    trainer_down = False
    if supervisor is not None:
        status["supervisor"] = supervisor
        trainer_down = bool(supervisor.get("halted")) or supervisor.get("trainer_presumed_up") is False
        if trainer_down and str(status.get("stage_status") or "") == "running":
            status["stage_status"] = "stalled"
    latest_checkpoint = _latest_checkpoint_summary(run_dir)
    if latest_checkpoint is not None:
        status["latest_checkpoint"] = latest_checkpoint
    # Within-epoch sub-phase: skipped when the trainer is known-down so a stale
    # mid-epoch file set cannot claim an ever-"training" run.
    if not trainer_down:
        sub_phase, sub_phase_detail, phase_progress = _derive_sub_phase(
            run_dir,
            diagnostics,
            events,
            selfplay_live if isinstance(selfplay_live, dict) else None,
        )
        if sub_phase is not None:
            status["sub_phase"] = sub_phase
            if sub_phase_detail is not None:
                status["sub_phase_detail"] = sub_phase_detail
            if phase_progress is not None:
                status["phase_progress"] = phase_progress
    return status


def _derive_sub_phase(
    run_dir: Path,
    diagnostics: Path,
    events: dict[str, object],
    selfplay_live: dict[str, object] | None,
) -> tuple[str | None, str | None, dict[str, object] | None]:
    """Derive the active within-epoch sub-phase (self-play / selecting window /
    shuffling / training / evaluating) for the CURRENT live epoch, purely from
    on-disk file signals, as ``(phase, detail, progress)``.

    A dense_cnn epoch runs self-play (~10 min) -> shuffle (~1 min) -> train (~2 min)
    -> SealBot eval (~15-19 min); a shrimp epoch runs self-play (~25-30 min) ->
    window select (~25s) -> train (~5-8 min) -> moves-left audit / multistage eval +
    checkpoint. The run-level ``stage`` stays ``epoch_NNNNNN`` the whole time, so
    without this every non-selfplay minute reads as a stuck "epoch running".

    ``progress`` is an optional ``{"phase", "elapsed_seconds", "typical_seconds"}``
    block (currently the shrimp training pass) that lets the client render a
    progress bar. Returns ``(None, None, None)`` when nothing can be derived (setup
    stages, stopped runs, or models without these file signals) so callers fall
    back to the existing ``stage``/``stage_status`` label. Robust to missing files."""

    # Only derive during an actively-running epoch. Setup stages, finished/stopped
    # runs (no active stage) and non-epoch stages fall through to None, which keeps
    # stopped runs like hexgnn on their existing label.
    if str(events.get("status") or "") != "running":
        return None, None, None
    epoch = events.get("epoch")
    if not isinstance(epoch, int):
        return None, None, None

    prefix = _diag_prefix(run_dir)
    sp_status = str((selfplay_live or {}).get("status") or "")
    sp_epoch = (selfplay_live or {}).get("epoch")
    sp_age = _file_age_seconds(diagnostics / f"{prefix}.selfplay.live.json")

    # SELF-PLAY: live writer still running for this epoch and the file is fresh.
    if (
        sp_status == "running"
        and sp_epoch == epoch
        and sp_age is not None
        and sp_age <= 30.0
    ):
        finished = (selfplay_live or {}).get("games_finished")
        requested = (selfplay_live or {}).get("requested_games")
        detail = None
        progress: dict[str, object] | None = None
        if isinstance(finished, int) and isinstance(requested, int) and requested > 0:
            detail = f"games {finished}/{requested}"
        return "self-play", detail, progress

    # POST-SELF-PLAY. Self-play for this epoch reports completed, but the finished
    # epoch_NNNNNN.json has not been written yet, so the run is somewhere in the
    # epoch tail. The tail differs per lineage.
    epoch_done = (diagnostics / f"epoch_{epoch:06d}.json").is_file()
    if sp_status == "completed" and sp_epoch == epoch and not epoch_done:
        if prefix == "shrimp":
            return _shrimp_post_selfplay_phase(diagnostics, prefix, epoch)
        shuffle_age = _shuffle_dir_age_seconds(run_dir, epoch)
        if shuffle_age is None:
            # Shuffle output for this epoch not on disk yet -> still shuffling.
            return "shuffling", None, None
        # Shuffle done. Training is ~2 min; treat the window right after the shuffle
        # dir appears as training, and everything after as the long SealBot eval.
        if shuffle_age <= 150.0:
            return "training", None, None
        return "evaluating", "SealBot", None

    return None, None, None


def _shrimp_post_selfplay_phase(
    diagnostics: Path,
    prefix: str,
    epoch: int,
) -> tuple[str, str | None, dict[str, object] | None]:
    """shrimp epoch tail from the per-segment diagnostics files, each written
    the moment its phase completes: ``<prefix>.selfplay.epoch_N.json`` (self-play
    done) -> ``.select`` (window selection done, ~25s) -> ``.training`` (training
    pass done, ~5-8 min) -> moves-left audit / multistage eval + checkpoint +
    the merged ``epoch_N.json``. During the training pass the select file's age
    IS the elapsed training time; the typical duration comes from the most
    recent finished epochs' ``train_seconds``, giving the client an honest
    "4.2m elapsed · ~7m typical" progress readout."""

    if (diagnostics / f"{prefix}.training.epoch_{epoch:06d}.json").is_file():
        # Training done; the remainder is the moves-left audit + checkpoint
        # write (seconds), or the multistage eval on eval epochs (minutes).
        return "evaluating", "audit / eval / checkpoint", None
    select_age = _file_age_seconds(diagnostics / f"{prefix}.select.epoch_{epoch:06d}.json")
    if select_age is None:
        # Self-play done but no select output yet: the brief window selection.
        return "selecting window", None, None
    typical = _recent_train_seconds(diagnostics, prefix, epoch)
    detail = f"{select_age / 60.0:.1f}m elapsed"
    if typical is not None:
        detail += f" · ~{typical / 60.0:.0f}m typical"
    progress = {
        "phase": "training",
        "elapsed_seconds": select_age,
        "typical_seconds": typical,
    }
    return "training", detail, progress


def _recent_train_seconds(diagnostics: Path, prefix: str, epoch: int) -> float | None:
    """``train_seconds`` (fallback ``seconds``) from the newest finished
    ``<prefix>.training.epoch_*.json`` within the 5 epochs before ``epoch``,
    or None when no recent epoch has one (fresh runs, schema gaps)."""

    for prev in range(epoch - 1, max(epoch - 6, 0), -1):
        payload = _read_json_file(diagnostics / f"{prefix}.training.epoch_{prev:06d}.json")
        if isinstance(payload, dict):
            seconds = _optional_float(payload.get("train_seconds"))
            if seconds is None:
                seconds = _optional_float(payload.get("seconds"))
            if seconds is not None and seconds > 0.0:
                return seconds
    return None


def _file_age_seconds(path: Path) -> float | None:
    stat = _safe_stat(path)
    if stat is None:
        return None
    return max(0.0, wall_clock() - float(stat.st_mtime))


def _shuffle_dir_age_seconds(run_dir: Path, epoch: int) -> float | None:
    """Seconds since the shuffleddata dir for ``epoch`` was last written, or None if
    no such dir exists yet (shuffle for this epoch has not produced output)."""
    suffix = f"epoch_{epoch:06d}"
    newest_mtime: float | None = None
    try:
        candidates = (run_dir / "shuffleddata").glob(f"*{suffix}")
    except OSError:
        return None
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        stat = _safe_stat(candidate)
        if stat is None:
            continue
        if newest_mtime is None or stat.st_mtime > newest_mtime:
            newest_mtime = stat.st_mtime
    if newest_mtime is None:
        return None
    return max(0.0, wall_clock() - newest_mtime)


def _training_run_status(run_dir: Path, histories: list[dict[str, object]], live_status: dict[str, object]) -> dict[str, object]:
    lengths = [
        int(item.get("length") or item.get("actions") or 0)
        for item in histories
        if int(item.get("length") or item.get("actions") or 0) > 0
    ]
    latest_history = max(
        histories,
        key=lambda item: float(item.get("modified") or 0.0),
        default=None,
    )
    status = dict(live_status)
    p0_wins = sum(1 for item in histories if item.get("winner") == "player0")
    p1_wins = sum(1 for item in histories if item.get("winner") == "player1")
    status["history"] = {
        "games": len(histories),
        "complete": False,
        "scope": "recent",
        "completed": sum(1 for item in histories if item.get("status") == "completed"),
        "aborted": sum(1 for item in histories if item.get("status") != "completed"),
        "p0_wins": p0_wins,
        "p1_wins": p1_wins,
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "avg_length": (sum(lengths) / len(lengths)) if lengths else None,
        "latest_modified": latest_history.get("modified") if latest_history else None,
        "latest_path": latest_history.get("path") if latest_history else None,
    }
    latest_selfplay = max(
        (path for path in (run_dir / "selfplay").glob("*.hxr") if path.is_file()),
        key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)),
        default=None,
    )
    if latest_selfplay is not None:
        stat = _safe_stat(latest_selfplay)
        status["latest_selfplay_record"] = {
            "path": latest_selfplay.relative_to(run_dir).as_posix(),
            "bytes": stat.st_size if stat is not None else 0,
            "modified": stat.st_mtime if stat is not None else 0,
        }
    return status


def _live_history_diagnostic_summary(live_status: dict[str, object]) -> dict[str, object]:
    watchdog = live_status.get("watchdog") if isinstance(live_status.get("watchdog"), dict) else {}
    calibration = live_status.get("calibration") if isinstance(live_status.get("calibration"), dict) else {}
    summary: dict[str, object] = {
        "stage": live_status.get("stage") or "unknown",
        "epoch": live_status.get("current_epoch") or "--",
    }
    if watchdog:
        summary["watchdog"] = watchdog.get("status") or "unknown"
        summary["free_ram_gb"] = watchdog.get("free_ram_gb")
        summary["gpu_free_gb"] = watchdog.get("gpu_free_gb")
        summary["trainer_private_gb"] = watchdog.get("trainer_private_gb")
    if calibration:
        summary["selfplay_pos_s"] = calibration.get("selfplay_pos_s")
        summary["exact_128"] = calibration.get("exact_128")
    return summary


def _latest_training_progress(diagnostics_dir: Path) -> dict[str, object] | None:
    # NOTE: no current producer emits this file; see dense_cnn/selfplay.py (kept for forward-compat / manual drops).
    latest = max(
        diagnostics_dir.glob("dense_cnn.training_progress.epoch_*.json"),
        key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)),
        default=None,
    )
    if latest is None:
        return None
    payload = _read_json_file(latest)
    return payload if isinstance(payload, dict) else None


def _latest_bootstrap_training_progress(run_dir: Path) -> dict[str, object] | None:
    # NOTE: no current producer emits this file; see dense_cnn/selfplay.py (kept for forward-compat / manual drops).
    latest = max(
        (run_dir / "bootstrap").glob("*/diagnostics/dense_cnn.training_progress.epoch_*.json"),
        key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)),
        default=None,
    )
    if latest is None:
        return None
    payload = _read_json_file(latest)
    if not isinstance(payload, dict):
        return None
    payload = dict(payload)
    payload["path"] = latest.relative_to(run_dir).as_posix()
    payload["output_dir"] = latest.parents[1].relative_to(run_dir).as_posix()
    return payload


def _training_progress_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "epoch": payload.get("epoch"),
        "status": payload.get("status"),
        "progress": payload.get("progress"),
        "steps": payload.get("steps"),
        "total_steps": payload.get("total_steps"),
        "samples_seen": payload.get("samples_seen"),
        "samples": payload.get("samples"),
        "passes": payload.get("passes"),
        "loss": payload.get("loss"),
        "samples_per_second": payload.get("samples_per_second"),
        "path": payload.get("path"),
        "output_dir": payload.get("output_dir"),
    }


def _stage_status_from_events(path: Path) -> dict[str, object]:
    """Active stage/epoch from the run's ``diagnostics/events.jsonl`` (written
    by hexo_train's DiagnosticsWriter): a stage_started without its matching
    stage_finished means the stage is still running; otherwise fall back to
    the last event's stage/status."""

    active_stage: str | None = None
    active_epoch: int | None = None
    last_event: dict[str, object] | None = None
    for event in _iter_jsonl(path):
        if not isinstance(event, dict):
            continue
        last_event = event
        name = str(event.get("event") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        stage = str(payload.get("stage") or "")
        if name == "stage_started" and stage:
            active_stage = stage
            active_epoch = _epoch_from_artifact_path(stage)
        elif name == "stage_finished" and stage == active_stage:
            active_stage = None
            active_epoch = None
    return {
        "stage": active_stage or _event_stage(last_event),
        "status": "running" if active_stage else _event_status(last_event),
        "epoch": active_epoch,
        "last_event": last_event,
    }


def _event_stage(event: dict[str, object] | None) -> str | None:
    payload = event.get("payload") if isinstance(event, dict) and isinstance(event.get("payload"), dict) else {}
    stage = payload.get("stage")
    return str(stage) if stage is not None else None


def _event_status(event: dict[str, object] | None) -> str | None:
    payload = event.get("payload") if isinstance(event, dict) and isinstance(event.get("payload"), dict) else {}
    status = payload.get("status")
    return str(status) if status is not None else None


def _watchdog_summary(payload: dict[str, object]) -> dict[str, object]:
    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
    trainer = payload.get("trainer") if isinstance(payload.get("trainer"), dict) else {}
    gpu = payload.get("gpu") if isinstance(payload.get("gpu"), dict) else {}
    return {
        "timestamp": payload.get("timestamp"),
        "status": payload.get("status"),
        "critical": payload.get("critical") or [],
        "free_ram_gb": memory.get("free_ram_gb"),
        "free_virtual_gb": memory.get("free_virtual_gb"),
        "trainer_private_gb": trainer.get("private_gb"),
        "trainer_working_set_gb": trainer.get("working_set_gb"),
        "gpu_free_gb": gpu.get("free_gb"),
        "gpu_used_gb": gpu.get("used_gb"),
        "gpu_utilization_percent": gpu.get("utilization_percent"),
    }


def _calibration_summary(payload: dict[str, object]) -> dict[str, object]:
    return {
        "status": payload.get("status"),
        "device": payload.get("device"),
        "selfplay_pos_s": payload.get("measured_selfplay_positions_per_second"),
        "target_pos_s": payload.get("target_selfplay_positions_per_second"),
        "meets_target": payload.get("meets_target"),
        "exact_128": (
            payload.get("all_searches_exact") is True
            and int(payload.get("selected_mcts_visits") or 0) == 128
        ),
        "selected_inference_batch_size": payload.get("selected_inference_batch_size"),
        "selected_selfplay_batch_size": payload.get("selected_selfplay_batch_size"),
        "selected_training_batch_size": payload.get("selected_training_batch_size"),
        "selected_mcts_virtual_batch_size": payload.get("selected_mcts_virtual_batch_size"),
    }


def _selfplay_live_summary(payload: dict[str, object]) -> dict[str, object]:
    # Self-play writes this file every couple of seconds during an epoch and a
    # final "completed" snapshot at epoch end. "live" means the writer is still
    # running and the file is fresh, so the dashboard can trust the in-progress
    # search-pos/s; a stale "running" file (writer died) falls back to not-live.
    timestamp = payload.get("timestamp")
    age_seconds: float | None = None
    if isinstance(timestamp, (int, float)):
        age_seconds = max(0.0, wall_clock() - float(timestamp))
    status = str(payload.get("status") or "")
    is_live = status == "running" and age_seconds is not None and age_seconds <= 20.0
    return {
        "status": status or "unknown",
        "live": is_live,
        "age_seconds": age_seconds,
        "epoch": payload.get("epoch"),
        "search_pos_s": payload.get("search_positions_per_second"),
        "pos_s": payload.get("positions_per_second"),
        "searched_positions": payload.get("searched_positions"),
        "games_finished": payload.get("games_finished"),
        "requested_games": payload.get("requested_games"),
        "active_games": payload.get("active_games"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
    }


# Supervisor lifecycle lines: "[2026-07-04T18:54:08Z] EXIT pid=74407 code=1 ...".
_SUPERVISOR_LINE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\]\s*(.*)$")
_SUPERVISOR_LOG_TAIL_BYTES = 32768


def _supervisor_ts(stamp: str) -> float | None:
    try:
        return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _supervisor_summary(run_dir: Path) -> dict[str, object] | None:
    """Supervisor lifecycle summary from the run's ``supervisor.log`` tail plus
    the ``supervisor_halted.flag`` breaker file.

    The training supervisor (scripts/supervise.sh) appends one-line lifecycle
    events: LAUNCH / RESUME / EXIT / ``CRASH|`` (crash excerpt lines) /
    RELAUNCH / HALT / ABORT. The trainer is presumed up while the last
    launch/exit-shaped event is a LAUNCH; a tripped breaker writes the halted
    flag and stops relaunching WITHOUT finishing the events.jsonl stage, so
    this block is what keeps a halted run from reading as "running" forever.
    Returns None when the run has no supervisor.log (non-supervised runs), so
    the block is omitted rather than faked."""

    log_path = run_dir / "supervisor.log"
    stat = _safe_stat(log_path)
    if stat is None:
        return None
    try:
        with log_path.open("rb") as handle:
            if stat.st_size > _SUPERVISOR_LOG_TAIL_BYTES:
                handle.seek(-_SUPERVISOR_LOG_TAIL_BYTES, os.SEEK_END)
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None

    now = wall_clock()
    last_launch_ts: float | None = None
    last_exit_ts: float | None = None
    last_event = ""  # last lifecycle-shaping event kind: launch / exit / halt / abort / supervisor-exit
    last_resume: str | None = None
    last_crash_line: str | None = None
    exits_last_hour = 0
    for raw in tail.splitlines():
        match = _SUPERVISOR_LINE_RE.match(raw.strip())
        if match is None:
            continue
        ts = _supervisor_ts(match.group(1))
        body = match.group(2)
        if body.startswith("LAUNCH"):
            last_launch_ts = ts
            last_event = "launch"
        elif body.startswith("EXIT"):
            last_exit_ts = ts
            last_event = "exit"
            if ts is not None and (now - ts) <= 3600.0:
                exits_last_hour += 1
        elif body.startswith("HALT"):
            last_event = "halt"
        elif body.startswith("ABORT"):
            last_event = "abort"
        elif "SUPERVISOR exit" in body:
            last_event = "supervisor-exit"
        elif body.startswith("RESUME from "):
            last_resume = body[len("RESUME from "):].strip()
        elif body.startswith("CRASH|"):
            crash_text = body[len("CRASH|"):].strip()
            if crash_text and not crash_text.startswith("Traceback"):
                last_crash_line = crash_text[:300]
    halted = (run_dir / "supervisor_halted.flag").is_file()
    return {
        "halted": halted,
        "trainer_presumed_up": (not halted) and last_event == "launch",
        "last_launch_age_seconds": (now - last_launch_ts) if last_launch_ts is not None else None,
        "last_exit_age_seconds": (now - last_exit_ts) if last_exit_ts is not None else None,
        "last_resume": last_resume,
        "exits_last_hour": exits_last_hour,
        "last_crash": last_crash_line,
    }


def _latest_checkpoint_summary(run_dir: Path) -> dict[str, object] | None:
    """Newest ``checkpoints/*.pt`` by mtime as ``{name, epoch, modified,
    age_seconds}``, or None when the run has no checkpoints yet. The age is the
    at-a-glance "how stale is the latest weights file" readout: during a healthy
    run it never exceeds roughly one epoch's wall time."""

    checkpoints_dir = run_dir / "checkpoints"
    newest_path: Path | None = None
    newest_mtime = 0.0
    try:
        for path in checkpoints_dir.glob("*.pt"):
            stat = _safe_stat(path)
            if stat is not None and stat.st_mtime > newest_mtime:
                newest_mtime = stat.st_mtime
                newest_path = path
    except OSError:
        return None
    if newest_path is None:
        return None
    return {
        "name": newest_path.name,
        "epoch": _coerce_epoch(None, newest_path.name),
        "modified": newest_mtime,
        "age_seconds": max(0.0, wall_clock() - newest_mtime),
    }


# ---------------------------------------------------------------------------
# Small filesystem/JSON helpers shared across the scan + live tiers. All are
# fail-soft: a missing/corrupt/mid-write file degrades to None/[]/0 rather
# than failing a request (run dirs are being written concurrently by training).
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> list[object]:
    if not path.is_file():
        return []
    records: list[object] = []
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return records
    return records


def _read_last_jsonl(path: Path) -> object | None:
    records = _iter_jsonl(path)
    return records[-1] if records else None


def _safe_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat()
    except OSError:
        return None


def _read_json_file(path: Path, *, retries: int = 1) -> object | None:
    for attempt in range(retries + 1):
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if attempt < retries:
                continue
            return None
        except OSError:
            return None
    return None


def _artifact_summary(payload: object) -> object:
    if not isinstance(payload, dict):
        return None
    keys = (
        "status",
        "epoch",
        "positions_per_second",
        "search_positions_per_second",
        "end_to_end_positions_per_second",
        "mcts_search_elapsed_seconds",
        "samples_added",
        "samples_per_second",
        "measured_selfplay_positions_per_second",
        "selected_inference_batch_size",
        "selected_selfplay_batch_size",
        "selected_mcts_virtual_batch_size",
        "selected_mcts_visits",
        "searched_positions",
        "mcts_simulations",
        "mcts_sims_per_searched_position",
        "meets_target",
        "games",
        "completed",
        "wins",
        "losses",
        "mean_turns",
        "winner",
        "length",
    )
    return {key: payload[key] for key in keys if key in payload}


def _resolve_run_dir(name: str) -> Path | None:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    matches: list[Path] = []
    for root in _training_roots():
        resolved_root = root.resolve()
        path = (resolved_root / name).resolve()
        if resolved_root != path and resolved_root not in path.parents:
            continue
        if path.is_dir():
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda item: (lambda s: s.st_mtime if s is not None else 0)(_safe_stat(item)))


def _resolve_run_path(run_name: str, artifact_path: str) -> Path | None:
    run_dir = _resolve_run_dir(run_name)
    if run_dir is None or not artifact_path or artifact_path.startswith(("/", "\\")):
        return None
    path = (run_dir / artifact_path).resolve()
    if run_dir.resolve() != path and run_dir.resolve() not in path.parents:
        return None
    return path


# ---------------------------------------------------------------------------
# Server entry points: run()/main() (python -m hexo_frontend.web, the
# hexo-play console script, and scripts/_dashboard_launch.sh in production).
# ---------------------------------------------------------------------------


def make_handler(controller: ManualMatchController) -> type[HexoPlayHandler]:
    class BoundHexoPlayHandler(HexoPlayHandler):
        pass

    BoundHexoPlayHandler.controller = controller
    return BoundHexoPlayHandler


def run(host: str = "127.0.0.1", port: int = 8765, *, sealbot_path: str | Path | None = None) -> None:
    controller = ManualMatchController(sealbot_path=sealbot_path)
    server = ThreadingHTTPServer((host, port), make_handler(controller))
    print(f"Hexo frontend match: http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        controller.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the manual Hexo web match.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--sealbot-path", default=None, help="Path to an external SealBot checkout.")
    args = parser.parse_args(argv)
    run(host=args.host, port=args.port, sealbot_path=args.sealbot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
