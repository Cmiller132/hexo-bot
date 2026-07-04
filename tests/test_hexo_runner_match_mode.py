from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (
    TESTS,
    ROOT / "packages" / "hexo_engine" / "python",
    ROOT / "packages" / "hexo_runner" / "python",
    ROOT / "packages" / "hexo_frontend" / "python",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


WINNING_P0 = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0))
FILLER_P1 = ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2))


class ScriptedPlayer:
    def __init__(self, player_id: str, moves: tuple[tuple[int, int], ...], *, mutate_decision_clone: bool = False) -> None:
        from hexo_runner import PlayerIdentity

        self.identity = PlayerIdentity(player_id=player_id)
        self.base_moves = tuple(moves)
        self.moves: list[tuple[int, int]] = []
        self.mutate_decision_clone = mutate_decision_clone
        self.setup_worker_count = 0
        self.start_game_count = 0
        self.finish_game_count = 0
        self.close_count = 0
        self.observed: list[object] = []

    def setup_worker(self, context: object) -> None:
        self.setup_worker_count += 1

    def start_game(self, context: object) -> None:
        self.start_game_count += 1
        self.moves = list(self.base_moves)

    def decide(self, state: object) -> object:
        from hexo_engine import AxialCoord, PlacementAction, apply_action, legal_actions
        from hexo_runner import DecisionResult

        actions = list(legal_actions(state))
        if self.mutate_decision_clone and actions:
            apply_action(state, actions[0])
        if not self.moves:
            raise RuntimeError("script exhausted")
        q, r = self.moves.pop(0)
        return DecisionResult(action=PlacementAction(AxialCoord(q, r)), diagnostics={"scripted": True})

    def observe_transition(self, transition: object) -> None:
        self.observed.append(transition)

    def finish_game(self, final_summary: object) -> None:
        self.finish_game_count += 1

    def close(self) -> None:
        self.close_count += 1


class IllegalPlayer(ScriptedPlayer):
    def __init__(self, player_id: str) -> None:
        super().__init__(player_id, ())

    def decide(self, state: object) -> object:
        from hexo_engine import AxialCoord, PlacementAction
        from hexo_runner import DecisionResult

        return DecisionResult(action=PlacementAction(AxialCoord(99, 99)))


class ExplodingPlayer(ScriptedPlayer):
    def decide(self, state: object) -> object:
        raise RuntimeError("boom")


class MutatingObserverPlayer(ScriptedPlayer):
    def observe_transition(self, transition: object) -> None:
        from hexo_engine import apply_action, legal_actions

        self.observed.append(transition)
        actions = list(legal_actions(transition.state))
        if actions:
            apply_action(transition.state, actions[0])


class RecordingIllegalObserver(IllegalPlayer):
    def __init__(self, player_id: str) -> None:
        super().__init__(player_id)
        self.python_states: list[object] = []

    def observe_transition(self, transition: object) -> None:
        from hexo_engine import to_python_state

        self.observed.append(transition)
        self.python_states.append(to_python_state(transition.state))


@dataclass
class ScriptedFactory:
    player_id: str
    moves: tuple[tuple[int, int], ...]
    created: int = 0
    instances: list[ScriptedPlayer] = field(default_factory=list)

    def create_player(self) -> ScriptedPlayer:
        self.created += 1
        player = ScriptedPlayer(self.player_id, self.moves)
        self.instances.append(player)
        return player


@dataclass
class ConditionalFactory:
    player_id: str
    moves: tuple[tuple[int, int], ...]
    abort_game_id: str | None = None
    created: int = 0
    instances: list[ScriptedPlayer] = field(default_factory=list)

    def create_player(self) -> ScriptedPlayer:
        self.created += 1
        if self.abort_game_id is None:
            player = ScriptedPlayer(self.player_id, self.moves)
        else:
            player = ConditionalPlayer(self.player_id, self.moves, self.abort_game_id)
        self.instances.append(player)
        return player


class ConditionalPlayer(ScriptedPlayer):
    def __init__(self, player_id: str, moves: tuple[tuple[int, int], ...], abort_game_id: str) -> None:
        super().__init__(player_id, moves)
        self.abort_game_id = abort_game_id
        self.current_game_id = ""

    def start_game(self, context: object) -> None:
        super().start_game(context)
        self.current_game_id = context.game_id

    def decide(self, state: object) -> object:
        if self.current_game_id == self.abort_game_id:
            return IllegalPlayer(self.identity.player_id).decide(state)
        return super().decide(state)


def action_from_record(action_id: int) -> object:
    from hexo_engine import PlacementAction
    from hexo_engine.types import unpack_coord_id

    return PlacementAction(unpack_coord_id(action_id))


def records_from_result(result: object) -> tuple[object, ...]:
    from hexo_runner.records import HexoRecordFile

    with HexoRecordFile.open(result.record_ref["path"]) as record_file:
        return record_file.iter_records()


class RunnerRewriteTests(unittest.TestCase):
    def test_completed_game_writes_compact_replayable_record(self) -> None:
        from hexo_engine import Player, apply_action, new_game, terminal
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import GameStatus, HEXO_RECORD_SCHEMA_VERSION
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(
                GameSpec(game_id="scripted", seed=7),
                (ScriptedPlayer("p0", WINNING_P0), ScriptedPlayer("p1", FILLER_P1)),
                tmp,
            )
            records = records_from_result(result)

        self.assertEqual(result.status, GameStatus.COMPLETED)
        self.assertEqual(result.winner, "player0")
        self.assertEqual(result.turns, 12)
        self.assertEqual(HEXO_RECORD_SCHEMA_VERSION, 1)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.status, "completed")
        self.assertEqual(len(record.action_ids), 12)
        self.assertIsNone(record.abort)

        replay = new_game(seed=record.seed)
        for action_id in record.action_ids:
            apply_action(replay, action_from_record(action_id))
        self.assertEqual(terminal(replay).winner, Player.PLAYER_0)
        self.assertEqual(record.replay().winner, Player.PLAYER_0)

    def test_recorded_scenarios_are_rejected(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "scenario"):
                run_match(
                    GameSpec(game_id="scenario", scenario={"opening": "custom"}),
                    (ScriptedPlayer("p0", WINNING_P0), ScriptedPlayer("p1", FILLER_P1)),
                    tmp,
                )

    def test_record_file_rejects_non_none_scenario(self) -> None:
        from hexo_runner.records import HexoRecordFile

        with tempfile.TemporaryDirectory() as tmp:
            with HexoRecordFile.create(Path(tmp) / "scenario.hxr", {"rules_version": 1, "backend": "test"}, ()) as file:
                with self.assertRaisesRegex(ValueError, "scenarios"):
                    file.begin_game("scenario", scenario={"opening": "custom"})

    def test_player_can_mutate_decision_clone_without_corrupting_primary_state(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_engine.types import unpack_coord_id
        from hexo_runner.records import GameStatus
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(
                GameSpec(game_id="clone-isolation"),
                (ScriptedPlayer("p0", ((0, 0),), mutate_decision_clone=True), IllegalPlayer("p1")),
                tmp,
            )
            records = records_from_result(result)

        self.assertEqual(result.status, GameStatus.ABORTED)
        self.assertEqual(result.turns, 1)
        self.assertEqual(len(records[0].action_ids), 1)
        coord = unpack_coord_id(records[0].action_ids[0])
        self.assertEqual((coord.q, coord.r), (0, 0))

    def test_illegal_action_aborts_loudly_and_writes_aborted_record(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import GameStatus
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(GameSpec(game_id="illegal"), (IllegalPlayer("p0"), ScriptedPlayer("p1", ())), tmp)
            records = records_from_result(result)

        self.assertEqual(result.status, GameStatus.ABORTED)
        self.assertEqual(result.abort.stage, "engine.apply_action")
        self.assertIn("opening placement", result.abort.message)
        record = records[0]
        self.assertEqual(record.status, "aborted")
        self.assertEqual(record.action_ids, ())
        self.assertIsNone(record.winner)
        self.assertEqual(record.abort.stage, "engine.apply_action")

    def test_player_exception_aborts_with_stage_type_and_message(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import GameStatus
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(GameSpec(game_id="explode"), (ExplodingPlayer("p0", ()), ScriptedPlayer("p1", ())), tmp)
            records = records_from_result(result)

        self.assertEqual(result.status, GameStatus.ABORTED)
        self.assertEqual(result.abort.stage, "player.decide:p0")
        self.assertEqual(result.abort.exception_type, "RuntimeError")
        self.assertEqual(result.abort.message, "boom")
        self.assertEqual(records[0].abort.stage, "player.decide:p0")

    def test_max_actions_aborts_before_requesting_next_decision(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import GameStatus
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(
                GameSpec(game_id="max-actions", max_actions=1),
                (ScriptedPlayer("p0", ((0, 0),)), ScriptedPlayer("p1", ((0, 1),))),
                tmp,
            )
            records = records_from_result(result)

        self.assertEqual(result.status, GameStatus.ABORTED)
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.abort.stage, "runner.max_actions")
        self.assertEqual(result.abort.exception_type, "MaxActionsExceeded")
        self.assertIn("max-actions", result.abort.message)
        self.assertIn("max_actions=1", result.abort.message)
        record = records[0]
        self.assertEqual(record.status, "aborted")
        self.assertEqual(len(record.action_ids), 1)
        self.assertEqual(record.abort.stage, "runner.max_actions")

    def test_terminal_move_exactly_at_max_actions_completes(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import GameStatus
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(
                GameSpec(game_id="max-terminal", max_actions=12),
                (ScriptedPlayer("p0", WINNING_P0), ScriptedPlayer("p1", FILLER_P1)),
                tmp,
            )
            records = records_from_result(result)

        self.assertEqual(result.status, GameStatus.COMPLETED)
        self.assertEqual(result.turns, 12)
        self.assertEqual(records[0].status, "completed")
        self.assertIsNone(records[0].abort)

    def test_game_spec_rejects_non_positive_max_actions(self) -> None:
        from hexo_runner.session import GameSpec

        with self.assertRaisesRegex(ValueError, "max_actions"):
            GameSpec(game_id="bad", max_actions=0)

    def test_observers_receive_independent_cloned_states(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.session import GameSpec

        p0 = MutatingObserverPlayer("p0", ((0, 0),))
        p1 = RecordingIllegalObserver("p1")
        with tempfile.TemporaryDirectory() as tmp:
            run_match(GameSpec(game_id="observer-clones"), (p0, p1), tmp)

        self.assertEqual(p1.python_states[0].placements_made, 1)
        self.assertGreater(len(p0.observed), 0)
        self.assertGreater(len(p1.observed), 0)

    def test_hexo_record_file_writes_one_record_per_game(self) -> None:
        from hexo_runner.modes.match import run_match
        from hexo_runner.records import HexoRecordFile
        from hexo_runner.session import GameSpec

        with tempfile.TemporaryDirectory() as tmp:
            result = run_match(
                GameSpec(game_id="hxr"),
                (ScriptedPlayer("p0", WINNING_P0), ScriptedPlayer("p1", FILLER_P1)),
                tmp,
            )
            path = Path(result.record_ref["path"])
            with HexoRecordFile.open(path) as record_file:
                records = record_file.iter_records()

            self.assertEqual(path.suffix, ".hxr")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].game_id, "hxr")
            self.assertEqual(records[0].status, "completed")
            self.assertEqual(len(records[0].action_ids), 12)

    def test_frontend_controller_still_uses_generic_runner(self) -> None:
        from hexo_frontend.web import ManualMatchController

        controller = ManualMatchController()
        try:
            state = controller.state()
            self.assertEqual(state["legal"], [{"q": 0, "r": 0}])

            state = controller.submit_move(0, 0)
            self.assertEqual(len(state["placements"]), 1)
            self.assertEqual(state["current_player"], "player1")

            state = controller.reset()
            self.assertEqual(state["placements"], [])
            self.assertEqual(state["legal"], [{"q": 0, "r": 0}])

            with self.assertRaises(ValueError):
                controller.submit_move(42, 42)
        finally:
            controller.close()


@contextlib.contextmanager
def checkpoint_run(run_name: str = "ckpt_run", checkpoints: tuple[str, ...] = ("epoch_000001.pt", "epoch_000002.pt")):
    """Temp run dir with stub checkpoint files, exposed via HEXO_DEBUG_RUN_ROOT."""

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "runs" / run_name / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        for name in checkpoints:
            (ckpt_dir / name).write_bytes(b"stub")
        previous = os.environ.get("HEXO_DEBUG_RUN_ROOT")
        os.environ["HEXO_DEBUG_RUN_ROOT"] = tmp
        try:
            yield run_name
        finally:
            if previous is None:
                os.environ.pop("HEXO_DEBUG_RUN_ROOT", None)
            else:
                os.environ["HEXO_DEBUG_RUN_ROOT"] = previous


class SeatAwareBot:
    """Scripted stand-in for a checkpoint player: whoever is seated player0 plays
    the winning line, the player1 seat plays filler — so the P0 seat always wins
    and a slot-vs-seat tally mix-up is detectable under seat alternation."""

    def __init__(self, spec: dict) -> None:
        from hexo_runner import PlayerIdentity

        self.spec = dict(spec)
        self.identity = PlayerIdentity(player_id=f"test-ckpt-{spec.get('checkpoint')}")
        self.moves: list[tuple[int, int]] | None = None

    def setup_worker(self, context: object) -> None:
        return

    def start_game(self, context: object) -> None:
        self.moves = None

    def decide(self, state: object) -> object:
        from hexo_engine import AxialCoord, PlacementAction, to_python_state
        from hexo_runner import DecisionResult

        if self.moves is None:
            seated_first = to_python_state(state).placements_made == 0
            self.moves = list(WINNING_P0 if seated_first else FILLER_P1)
        q, r = self.moves.pop(0)
        return DecisionResult(
            action=PlacementAction(AxialCoord(q, r)),
            diagnostics={"root_value": 0.25, "visits": int(self.spec.get("visits") or 0)},
        )

    def observe_transition(self, transition: object) -> None:
        return

    def finish_game(self, final_summary: object) -> None:
        return

    def close(self) -> None:
        return


class MatchBackendTests(unittest.TestCase):
    def test_checkpoint_spec_normalization_defaults_and_clamps(self) -> None:
        from hexo_frontend.web import _normalize_player_spec

        with checkpoint_run() as run:
            spec = _normalize_player_spec({"kind": "checkpoint", "run": run, "checkpoint": "epoch_000001.pt"})
            self.assertEqual(
                spec,
                {
                    "kind": "checkpoint",
                    "run": run,
                    "checkpoint": "epoch_000001.pt",
                    "visits": 256,
                    "mode": "search",
                    "c_puct": 1.5,
                },
            )

            clamped = _normalize_player_spec(
                {
                    "kind": "checkpoint",
                    "run": run,
                    "checkpoint": "epoch_000002.pt",
                    "visits": 99999,
                    "mode": "policy",
                    "c_puct": 99.0,
                }
            )
            self.assertEqual(clamped["visits"], 2048)
            self.assertEqual(clamped["mode"], "policy")
            self.assertEqual(clamped["c_puct"], 10.0)

            low = _normalize_player_spec(
                {"kind": "checkpoint", "run": run, "checkpoint": "epoch_000001.pt", "visits": 1, "c_puct": 0.001}
            )
            self.assertEqual(low["visits"], 8)
            self.assertEqual(low["c_puct"], 0.1)

            with self.assertRaises(ValueError):
                _normalize_player_spec({"kind": "checkpoint", "run": run, "checkpoint": "missing.pt"})
            with self.assertRaises(ValueError):
                _normalize_player_spec(
                    {"kind": "checkpoint", "run": run, "checkpoint": "epoch_000001.pt", "mode": "tree"}
                )

    def test_legacy_player_forms_still_normalize(self) -> None:
        from hexo_frontend.web import _normalize_player_spec

        self.assertEqual(_normalize_player_spec("manual"), {"kind": "manual"})
        self.assertEqual(_normalize_player_spec("human"), {"kind": "manual"})
        self.assertEqual(_normalize_player_spec("bot"), {"kind": "sealbot", "variant": "current"})
        self.assertEqual(_normalize_player_spec("sealbot"), {"kind": "sealbot", "variant": "current"})
        self.assertEqual(_normalize_player_spec("sealbot-best"), {"kind": "sealbot", "variant": "best"})
        self.assertEqual(_normalize_player_spec({"kind": "manual"}), {"kind": "manual"})
        self.assertEqual(
            _normalize_player_spec({"kind": "sealbot", "variant": "best"}),
            {"kind": "sealbot", "variant": "best"},
        )
        self.assertEqual(_normalize_player_spec(None), {"kind": "manual"})
        with self.assertRaises(ValueError):
            _normalize_player_spec("nonsense")
        with self.assertRaises(ValueError):
            _normalize_player_spec({"kind": "sealbot", "variant": "weird"})

    def test_legacy_body_shape_keeps_working(self) -> None:
        from hexo_frontend.web import ManualMatchController

        controller = ManualMatchController(bot_factory=lambda variant, time_limit: SeatAwareBot({"checkpoint": variant}))
        try:
            state = controller.reset(
                {"mode": "sealbot", "human_player": "player1", "bot": {"variant": "best", "time_limit": 0.01}}
            )
            self.assertEqual(state["mode"], "sealbot")
            self.assertEqual(state["players"]["player0"]["kind"], "sealbot")
            self.assertEqual(state["players"]["player0"]["variant"], "best")
            self.assertEqual(state["players"]["player1"]["kind"], "manual")
            self.assertIsNone(state["series"])
        finally:
            controller.close()

    def test_unknown_checkpoint_fails_reset_before_starting(self) -> None:
        from hexo_frontend.web import ManualMatchController

        with checkpoint_run() as run:
            controller = ManualMatchController()
            try:
                with self.assertRaises(ValueError):
                    controller.reset(
                        {
                            "players": {
                                "player0": {"kind": "checkpoint", "run": run, "checkpoint": "missing.pt"},
                                "player1": "manual",
                            }
                        }
                    )
                # The previous (manual) game survives a rejected config.
                state = controller.state()
                self.assertEqual(state["players"]["player0"]["kind"], "manual")
            finally:
                controller.close()

    def test_checkpoint_series_alternates_seats_and_tallies_by_slot(self) -> None:
        from hexo_frontend.web import ManualMatchController

        created_specs: list[dict] = []

        def factory(spec: dict) -> SeatAwareBot:
            created_specs.append(spec)
            return SeatAwareBot(spec)

        with checkpoint_run() as run:
            controller = ManualMatchController(checkpoint_factory=factory)
            try:
                state = controller.reset(
                    {
                        "players": {
                            "player0": {"kind": "checkpoint", "run": run, "checkpoint": "epoch_000001.pt", "visits": 16},
                            "player1": {
                                "kind": "checkpoint",
                                "run": run,
                                "checkpoint": "epoch_000002.pt",
                                "visits": 16,
                                "mode": "policy",
                            },
                        },
                        "series": {"games": 3, "alternate": True},
                    }
                )
                self.assertEqual(state["mode"], "checkpoint")
                player0 = state["players"]["player0"]
                self.assertEqual(player0["kind"], "checkpoint")
                self.assertEqual(player0["run"], run)
                self.assertEqual(player0["checkpoint"], "epoch_000001.pt")
                self.assertEqual(player0["visits"], 16)
                self.assertEqual(player0["label"], f"{run} @ e1")

                controller._thread.join(timeout=20.0)
                self.assertFalse(controller._thread.is_alive())
                state = controller.state()
                self.assertIsNone(state["error"])

                series = state["series"]
                self.assertTrue(series["finished"])
                self.assertEqual(series["games"], 3)
                self.assertEqual(series["played"], 3)
                self.assertTrue(series["alternate"])
                self.assertEqual(len(series["results"]), 3)
                # The player0 SEAT wins every game; alternation moves slot1 into
                # that seat for game 2, so the tally must count by slot.
                self.assertEqual([row["winner_seat"] for row in series["results"]], ["player0"] * 3)
                self.assertEqual([row["winner_slot"] for row in series["results"]], ["slot0", "slot1", "slot0"])
                self.assertEqual(series["tally"], {"slot0": 2, "slot1": 1, "draws": 0})
                self.assertEqual([row["length"] for row in series["results"]], [12, 12, 12])
                self.assertEqual(series["seats"], {"player0": "slot0", "player1": "slot1"})
                self.assertEqual(series["slots"]["slot0"]["checkpoint"], "epoch_000001.pt")
                self.assertEqual(series["slots"]["slot1"]["mode"], "policy")

                decisions = state["bot_decisions"]
                self.assertEqual(len(decisions), 12)  # the final game's log
                self.assertEqual(decisions[0]["ply"], 0)
                self.assertTrue(all(item["value"] == 0.25 for item in decisions))
                self.assertTrue(all(item["kind"] == "checkpoint" for item in decisions))

                self.assertEqual(len(created_specs), 6)  # fresh players per game
                self.assertEqual(created_specs[0]["checkpoint"], "epoch_000001.pt")
            finally:
                controller.close()

    def test_stop_blocks_moves_and_reset_recovers(self) -> None:
        from hexo_frontend.web import ManualMatchController, MoveConflict

        controller = ManualMatchController()
        try:
            controller.submit_move(0, 0)
            state = controller.stop()
            self.assertTrue(state["stopped"])
            self.assertEqual(state["turn_status"], "stopped")
            self.assertFalse(state["can_submit"])
            with self.assertRaises(MoveConflict):
                controller.submit_move(0, 1)

            state = controller.reset()
            self.assertFalse(state["stopped"])
            self.assertEqual(state["placements"], [])
            state = controller.submit_move(0, 0)
            self.assertEqual(len(state["placements"]), 1)
        finally:
            controller.close()

    def test_checkpoint_visit_selection_mimics_eval_protocol(self) -> None:
        from hexo_frontend.web import (
            CHECKPOINT_OPENING_MOVES,
            _select_visit_action,
        )

        rows = [
            {"action_id": 11, "w": 400.0},
            {"action_id": 22, "w": 80.0},
            {"action_id": 33, "w": 20.0},
            {"action_id": 44, "w": 0.0},  # zero-weight: never selectable
        ]

        # Past the opening: strict visit argmax, regardless of row order.
        self.assertEqual(_select_visit_action(list(reversed(rows)), CHECKPOINT_OPENING_MOVES, "g"), 11)
        self.assertEqual(_select_visit_action(rows, CHECKPOINT_OPENING_MOVES + 40, "g"), 11)

        # Opening plies: sampled, reproducible per (game token, ply), always a
        # positive-weight action.
        valid = {11, 22, 33}
        for ply in range(CHECKPOINT_OPENING_MOVES):
            first = _select_visit_action(rows, ply, "game-A|7|0")
            self.assertIn(first, valid)
            self.assertEqual(first, _select_visit_action(rows, ply, "game-A|7|0"))

        # Distinct game tokens decorrelate the opening: across many games the
        # sampled ply-0 move is not always the same action.
        picks = {_select_visit_action(rows, 0, f"game-{i}|7|0") for i in range(64)}
        self.assertGreater(len(picks), 1)

        # A single positive-weight row is always chosen; no rows -> None.
        only = [{"action_id": 5, "w": 3.0}, {"action_id": 6, "w": 0.0}]
        self.assertEqual(_select_visit_action(only, 0, "g"), 5)
        self.assertIsNone(_select_visit_action([], 0, "g"))
        self.assertIsNone(_select_visit_action([{"action_id": 9, "w": 0.0}], 3, "g"))


if __name__ == "__main__":
    unittest.main()
