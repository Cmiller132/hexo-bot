from __future__ import annotations

import importlib.machinery
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from threading import Event


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


class SealBotAdapterTests(unittest.TestCase):
    def test_discovery_reports_available_compiled_variants(self) -> None:
        from hexo_runner.adapters.sealbot import discover_sealbot_adapters

        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_sealbot_root(Path(tmp), variants=("current",))
            payload = discover_sealbot_adapters(root)

        variants = {item["id"]: item for item in payload["variants"]}
        self.assertTrue(payload["configured"])
        self.assertTrue(variants["current"]["available"])
        self.assertFalse(variants["best"]["available"])
        self.assertIn("Compiled minimax_cpp", variants["best"]["error"])

    def test_player_buffers_two_stone_sealbot_turn(self) -> None:
        import hexo_engine as engine
        from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer
        from hexo_runner.player import GameContext, PlayerIdentity, WorkerContext

        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_sealbot_root(Path(tmp))
            worker = _fake_worker(Path(tmp))
            player = SealBotPlayer(
                SealBotConfig(path=root, variant="current", worker_script=worker, startup_timeout=2, response_timeout=2)
            )
            try:
                player.setup_worker(WorkerContext(worker_id=0))
                player.start_game(
                    GameContext(
                        game_id="buffer",
                        seed=None,
                        player_index=1,
                        player_role="player1",
                        opponent=PlayerIdentity("human"),
                    )
                )
                state = engine.new_game()
                first = player.decide(state)
                self.assertEqual((first.action.coord.q, first.action.coord.r), (0, 0))
                engine.apply_action(state, first.action)

                second = player.decide(state)
                self.assertEqual((second.action.coord.q, second.action.coord.r), (0, 1))
                engine.apply_action(state, second.action)

                third = player.decide(state)
                self.assertEqual((third.action.coord.q, third.action.coord.r), (1, 0))
                self.assertTrue(third.diagnostics["buffered_move"])
            finally:
                player.close()

    def test_player_rejects_illegal_worker_move(self) -> None:
        import hexo_engine as engine
        from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer
        from hexo_runner.player import GameContext, PlayerIdentity, WorkerContext

        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_sealbot_root(Path(tmp))
            worker = _fake_worker(Path(tmp), moves="[[99, 99]]")
            player = SealBotPlayer(
                SealBotConfig(path=root, variant="current", worker_script=worker, startup_timeout=2, response_timeout=2)
            )
            try:
                player.setup_worker(WorkerContext(worker_id=0))
                player.start_game(
                    GameContext(
                        game_id="illegal",
                        seed=None,
                        player_index=1,
                        player_role="player1",
                        opponent=PlayerIdentity("human"),
                    )
                )
                with self.assertRaisesRegex(ValueError, "illegal move"):
                    player.decide(engine.new_game())
            finally:
                player.close()


class FrontendSealBotControllerTests(unittest.TestCase):
    def test_human_player0_gets_bot_reply_and_turn_back(self) -> None:
        from hexo_frontend.web import ManualMatchController

        controller = ManualMatchController(bot_factory=lambda variant, limit: AutoLegalBot())
        try:
            state = controller.reset(_sealbot_config("player0"))
            self.assertEqual(state["turn_status"], "human_turn")
            state = controller.submit_move(0, 0)
            state = _wait_for_state(controller, state, lambda item: item["turn_status"] == "human_turn")
            self.assertEqual(state["current_player"], "player0")
            self.assertEqual(len(state["placements"]), 3)
            self.assertIsNotNone(state["last_bot_decision"])
        finally:
            controller.close()

    def test_slot_payload_player0_manual_player1_sealbot(self) -> None:
        from hexo_frontend.web import ManualMatchController

        controller = ManualMatchController(bot_factory=lambda variant, limit: AutoLegalBot())
        try:
            state = controller.reset(_slot_config("manual", "sealbot-current"))
            self.assertEqual(state["players"]["player0"]["kind"], "manual")
            self.assertEqual(state["players"]["player1"]["kind"], "sealbot")
            self.assertEqual(state["players"]["player1"]["variant"], "current")
            self.assertEqual(state["turn_status"], "human_turn")

            state = controller.submit_move(0, 0)
            state = _wait_for_state(controller, state, lambda item: item["turn_status"] == "human_turn")
            self.assertEqual(state["current_player"], "player0")
            self.assertEqual(len(state["placements"]), 3)
        finally:
            controller.close()

    def test_human_player1_starts_after_bot_opening(self) -> None:
        from hexo_frontend.web import ManualMatchController

        controller = ManualMatchController(bot_factory=lambda variant, limit: AutoLegalBot())
        try:
            state = controller.reset(_sealbot_config("player1"))
            state = _wait_for_state(controller, state, lambda item: item["turn_status"] == "human_turn")
            self.assertEqual(state["current_player"], "player1")
            self.assertEqual(len(state["placements"]), 1)
        finally:
            controller.close()

    def test_slot_payload_player0_sealbot_player1_manual_starts_after_opening(self) -> None:
        from hexo_frontend.web import ManualMatchController

        controller = ManualMatchController(bot_factory=lambda variant, limit: AutoLegalBot())
        try:
            state = controller.reset(_slot_config("sealbot-current", "manual"))
            state = _wait_for_state(controller, state, lambda item: item["turn_status"] == "human_turn")
            self.assertEqual(state["players"]["player0"]["kind"], "sealbot")
            self.assertEqual(state["players"]["player0"]["variant"], "current")
            self.assertEqual(state["players"]["player1"]["kind"], "manual")
            self.assertEqual(state["current_player"], "player1")
            self.assertEqual(len(state["placements"]), 1)
        finally:
            controller.close()

    def test_move_during_bot_turn_is_rejected(self) -> None:
        from hexo_frontend.web import ManualMatchController, MoveConflict

        release = Event()
        bot = BlockingBot(release)
        controller = ManualMatchController(bot_factory=lambda variant, limit: bot)
        try:
            state = controller.reset(_sealbot_config("player1"))
            self.assertEqual(state["turn_status"], "bot_thinking")
            with self.assertRaises(MoveConflict):
                controller.submit_move(0, 0)
        finally:
            release.set()
            controller.close()

    def test_bot_exception_is_exposed_as_error_state(self) -> None:
        from hexo_frontend.web import ManualMatchController

        controller = ManualMatchController(bot_factory=lambda variant, limit: ExplodingBot())
        try:
            state = controller.reset(_sealbot_config("player1"))
            state = _wait_for_state(controller, state, lambda item: item["turn_status"] == "error")
            self.assertIn("boom", state["error"])
        finally:
            controller.close()


class AutoLegalBot:
    def __init__(self) -> None:
        from hexo_runner import PlayerIdentity

        self.identity = PlayerIdentity(player_id="auto-bot", label="Auto Bot")

    def setup_worker(self, context: object) -> None:
        return

    def start_game(self, context: object) -> None:
        return

    def decide(self, state: object) -> object:
        import hexo_engine as engine
        from hexo_runner import DecisionResult

        return DecisionResult(action=next(iter(engine.legal_actions(state))), diagnostics={"fake_bot": True})

    def observe_transition(self, transition: object) -> None:
        return

    def finish_game(self, final_summary: object) -> None:
        return

    def close(self) -> None:
        return


class BlockingBot(AutoLegalBot):
    def __init__(self, release: Event) -> None:
        super().__init__()
        self.release = release

    def decide(self, state: object) -> object:
        self.release.wait(timeout=5.0)
        return super().decide(state)


class ExplodingBot(AutoLegalBot):
    def decide(self, state: object) -> object:
        raise RuntimeError("boom")


def _fake_sealbot_root(base: Path, *, variants: tuple[str, ...] = ("current", "best")) -> Path:
    root = base / "SealBot"
    root.mkdir()
    (root / "game.py").write_text("# fake SealBot root\n", encoding="utf-8")
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    for variant in ("current", "best"):
        variant_dir = root / variant
        variant_dir.mkdir()
        if variant in variants:
            (variant_dir / f"minimax_cpp{suffix}").write_bytes(b"")
    return root


def _fake_worker(base: Path, *, moves: str | None = None) -> Path:
    script = base / "fake_worker.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys

            print(json.dumps({{"ok": True, "type": "ready"}}), flush=True)
            for raw in sys.stdin:
                request = json.loads(raw)
                if request.get("type") == "close":
                    print(json.dumps({{"ok": True, "type": "closed"}}), flush=True)
                    break
                state = request["state"]
                if {moves!r} is not None:
                    chosen = json.loads({moves!r})
                elif state.get("phase") == "Opening":
                    chosen = [[0, 0]]
                elif state.get("phase") == "FirstStone":
                    chosen = [[0, 1], [1, 0]]
                else:
                    chosen = [[0, 2]]
                print(json.dumps({{"ok": True, "moves": chosen, "diagnostics": {{"last_depth": 3, "nodes": 12}}}}), flush=True)
            """
        ).strip(),
        encoding="utf-8",
    )
    return script


def _sealbot_config(human_player: str) -> dict[str, object]:
    return {
        "mode": "sealbot",
        "human_player": human_player,
        "bot": {"id": "sealbot", "variant": "current", "time_limit": 0.01},
    }


def _slot_config(player0: str, player1: str) -> dict[str, object]:
    return {"players": {"player0": player0, "player1": player1}, "time_limit": 0.01}


def _wait_for_state(controller: object, state: dict[str, object], predicate: object) -> dict[str, object]:
    deadline = time.monotonic() + 4.0
    current = state
    while time.monotonic() < deadline:
        if predicate(current):
            return current
        current = controller.state(since=int(current["version"]), timeout_ms=500)
    raise AssertionError(f"Timed out waiting for state. Last state: {current}")


if __name__ == "__main__":
    unittest.main()
