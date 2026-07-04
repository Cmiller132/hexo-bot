"""Core synchronous runner loop for one game.

Called by hexo_runner/modes/match.py (one game -> one .hxr file),
hexo_runner/modes/batch.py (many games sharing players + one record file per
worker), and scripts/goal_benchmark.py. All engine access goes through
`engine.HexoEngineAdapter`; all per-game persistence goes through the
Rust-backed `HexoRecordFile` writer (hexo_utils, re-exported via
hexo_runner/records). Every player/engine/record call is staged through
`_run_stage` so any failure becomes a structured `AbortRecord` instead of a
raised exception — `run_match_loop` itself never raises for in-game errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from .engine import HexoEngineAdapter
from .player import (
    FinalSummary,
    GameContext,
    RunnerPlayer,
    TransitionEvent,
    WorkerContext,
)
from .records import (
    AbortRecord,
    GameResult,
    GameStatus,
    HexoRecordFile,
)
from .session import GameSpec
from .timing import Timer


StageCall = Callable[[], object]


class RunnerAbort(Exception):
    """Internal control-flow exception carrying structured abort metadata."""

    def __init__(self, abort: AbortRecord) -> None:
        super().__init__(abort.message)
        self.abort = abort


def run_match_loop(
    spec: GameSpec,
    players: tuple[RunnerPlayer, RunnerPlayer],
    record_file: HexoRecordFile,
    *,
    engine_adapter: HexoEngineAdapter | None = None,
    worker_context: WorkerContext | None = None,
    setup_players: bool = True,
    close_players: bool = True,
) -> GameResult:
    """Run one game by mediating between generic players and the engine API.

    The runner owns exactly one authoritative state: `primary_state` below.
    Players never receive that object. On every decision, the runner gives the
    active player only a cloned `HexoState`. The returned action is then
    applied back to the primary state.
    """

    if len(players) != 2:
        raise ValueError("run_match_loop requires exactly two players.")
    if spec.scenario is not None:
        raise ValueError("GameSpec.scenario is not supported by durable .hxr records; use scenario=None.")

    # --- Setup: adapter, timing, worker context, abort bookkeeping ---------
    adapter = engine_adapter or HexoEngineAdapter()
    timer = Timer.start()
    engine_metadata = adapter.metadata()
    worker = worker_context or WorkerContext(worker_id=0, engine_metadata=engine_metadata)
    terminal_payload = None
    abort: AbortRecord | None = None
    status = GameStatus.ABORTED
    primary_state = None
    record_writer = None

    # --- Game body: player setup, new game, decision loop ------------------
    # Any RunnerAbort (or unexpected exception) lands in the handlers below;
    # the game is then finalized as ABORTED rather than propagating.
    try:
        if setup_players:
            for player in players:
                _run_stage(f"player.setup_worker:{player.identity.player_id}", lambda p=player: p.setup_worker(worker))

        primary_state = _run_stage(
            "engine.new_game",
            lambda: adapter.new_game(seed=spec.seed),
        )
        _start_players(spec, players, adapter, engine_metadata)
        record_writer = _run_stage(
            "record_file.begin_game",
            lambda: record_file.begin_game(spec.game_id, seed=spec.seed),
        )

        while adapter.terminal(primary_state) is None:
            if record_writer.action_count >= spec.max_actions:
                raise RunnerAbort(
                    AbortRecord(
                        stage="runner.max_actions",
                        exception_type="MaxActionsExceeded",
                        message=(
                            f"Game {spec.game_id!r} reached max_actions="
                            f"{spec.max_actions} before terminal state."
                        ),
                    )
                )
            current = _run_stage("engine.current_player", lambda: adapter.current_player(primary_state))
            player_index = adapter.player_index(current)
            active_player = players[player_index]
            role = adapter.player_role(current)
            cloned_state = _run_stage("engine.clone_state:decide", lambda: adapter.clone_state(primary_state))
            decision = _run_stage(
                f"player.decide:{active_player.identity.player_id}",
                lambda: active_player.decide(cloned_state),
            )
            action_id = _run_stage("engine.action_id", lambda: adapter.action_id(decision.action))
            transition = _run_stage(
                "engine.apply_action",
                lambda: adapter.apply_action(primary_state, decision.action),
            )
            _run_stage(
                "record_writer.record_action",
                lambda action=decision.action: record_writer.record_action(action),
            )
            terminal = _run_stage("engine.terminal", lambda: adapter.terminal(primary_state))
            terminal_payload = adapter.terminal_payload(terminal)

            for observer in players:
                event = TransitionEvent(
                    game_id=spec.game_id,
                    action_index=record_writer.action_count - 1,
                    player_id=active_player.identity.player_id,
                    player_role=role,
                    action_id=action_id,
                    action=decision.action,
                    transition=transition,
                    terminal=terminal,
                    state=_run_stage("engine.clone_state:observe", lambda: adapter.clone_state(primary_state)),
                )
                _run_stage(
                    f"player.observe_transition:{observer.identity.player_id}",
                    lambda player=observer, transition_event=event: player.observe_transition(transition_event),
                )

        status = GameStatus.COMPLETED
    except RunnerAbort as exc:
        abort = exc.abort
    except Exception as exc:
        abort = AbortRecord(
            stage="runner",
            exception_type=type(exc).__name__,
            message=str(exc),
        )

    # --- Record finalization ------------------------------------------------
    # Always finish the .hxr game entry (completed or aborted), even when the
    # abort happened before begin_game succeeded.
    duration_ms = timer.elapsed_ms()
    record_ref = None
    try:
        if record_writer is None:
            record_writer = record_file.begin_game(spec.game_id, seed=spec.seed)
        if status == GameStatus.COMPLETED:
            record_ref = record_writer.finish_completed(
                _terminal_winner(terminal_payload),
                _terminal_placements(terminal_payload),
            )
        else:
            record_ref = record_writer.finish_aborted(
                abort
                or AbortRecord(
                    stage="runner",
                    exception_type="RuntimeError",
                    message="game aborted before structured abort was set",
                )
            )
    except Exception as exc:
        status = GameStatus.ABORTED
        abort = AbortRecord(
            stage="record_file.finish_game",
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        terminal_payload = None

    # --- Result assembly and player teardown --------------------------------
    # finish_game/close failures are swallowed: the result is already decided
    # and teardown must not turn a completed game into an abort.
    result = GameResult(
        game_id=spec.game_id,
        status=status,
        terminal=terminal_payload if status == GameStatus.COMPLETED else None,
        winner=_terminal_winner(terminal_payload) if status == GameStatus.COMPLETED else None,
        record_ref=record_ref,
        turns=record_writer.action_count if record_writer is not None else 0,
        duration_ms=duration_ms,
        abort=abort,
        metadata={"engine": engine_metadata},
    )

    summary = FinalSummary(game_id=spec.game_id, result=result, metadata=result.metadata)
    for player in players:
        try:
            player.finish_game(summary)
        except Exception:
            pass

    if close_players:
        for player in players:
            try:
                player.close()
            except Exception:
                pass

    return result


def _run_stage(stage: str, func: StageCall) -> object:
    """Run one named player/engine/record call, converting failures to RunnerAbort.

    The `stage` string (e.g. "player.decide:<player_id>") ends up verbatim in
    AbortRecord.stage, which the .hxr writer persists — keep names stable for
    downstream abort triage.
    """
    try:
        return func()
    except RunnerAbort:
        raise
    except Exception as exc:
        raise RunnerAbort(
            AbortRecord(stage=stage, exception_type=type(exc).__name__, message=str(exc))
        ) from exc


def _start_players(
    spec: GameSpec,
    players: Sequence[RunnerPlayer],
    adapter: HexoEngineAdapter,
    engine_metadata: object,
) -> None:
    """Send start_game to both players with their per-game GameContext.

    Seat order is fixed: players[0] is "player0" (moves first), players[1] is
    "player1" — matching HexoEngineAdapter.player_index.
    """
    roles = ("player0", "player1")
    for index, player in enumerate(players):
        context = GameContext(
            game_id=spec.game_id,
            seed=spec.seed,
            player_index=index,
            player_role=roles[index],
            opponent=players[1 - index].identity,
            mode=spec.mode,
            is_evaluation=spec.is_evaluation,
            engine_metadata=engine_metadata,
            metadata=spec.metadata,
        )
        _run_stage(f"player.start_game:{player.identity.player_id}", lambda p=player, ctx=context: p.start_game(ctx))


def _terminal_winner(terminal_payload: object | None) -> object | None:
    if not isinstance(terminal_payload, dict):
        return None
    return terminal_payload.get("winner")


def _terminal_placements(terminal_payload: object | None) -> int:
    if not isinstance(terminal_payload, dict):
        return 0
    metadata = terminal_payload.get("metadata")
    if not isinstance(metadata, dict):
        return 0
    return int(metadata.get("placements", 0))
