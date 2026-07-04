from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for package in ("hexo_engine",):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class RustEngineBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        import hexo_engine as engine

        try:
            engine.engine_metadata()
        except engine.EngineUnavailableError as exc:
            self.skipTest(f"hexo_engine Rust bridge is unavailable: {exc}")

    def test_bridge_metadata_reports_rust_backend(self) -> None:
        import hexo_engine as engine

        metadata = engine.engine_metadata()

        self.assertEqual(metadata["backend"], "rust-pyo3")
        self.assertEqual(metadata["rules_version"], 1)
        self.assertEqual(metadata["state_api_version"], 2)
        self.assertFalse(hasattr(engine, "model1_batch_inputs"))
        self.assertFalse(hasattr(engine, "model1_batched_mcts"))

    def test_legal_actions_are_sorted(self) -> None:
        import hexo_engine as engine

        state = engine.new_game()
        engine.apply_action(state, engine.PlacementAction(engine.AxialCoord(0, 0)))

        coords = [(action.coord.q, action.coord.r) for action in engine.legal_actions(state)]

        self.assertEqual(coords, sorted(coords))

    def test_legal_actions_use_compact_id_view(self) -> None:
        import random

        import hexo_engine as engine
        from hexo_engine.types import pack_coord_id

        state = engine.new_game()
        legal = engine.legal_actions(state)

        self.assertEqual(engine.legal_action_count(state), 1)
        self.assertEqual(legal.action_ids, engine.legal_action_ids(state))
        self.assertEqual(
            engine.action_id(engine.PlacementAction(engine.AxialCoord(0, 0))),
            pack_coord_id(engine.AxialCoord(0, 0)),
        )
        self.assertEqual(random.choice(legal), engine.PlacementAction(engine.AxialCoord(0, 0)))
        self.assertTrue(engine.is_legal_action(state, engine.PlacementAction(engine.AxialCoord(0, 0))))

    def test_clone_mutation_does_not_affect_original(self) -> None:
        import hexo_engine as engine

        state = engine.new_game()
        clone = engine.clone_state(state)
        engine.apply_action(clone, engine.PlacementAction(engine.AxialCoord(0, 0)))

        self.assertEqual(engine.to_python_state(state).placements_made, 0)
        self.assertEqual(engine.to_python_state(clone).placements_made, 1)

    def test_state_api_capsule_is_private_and_read_only(self) -> None:
        import hexo_engine as engine
        import hexo_engine._rust as rust

        state = engine.new_game()
        for q, r in [(0, 0), (1, 0), (0, 1)]:
            engine.apply_action(state, engine.PlacementAction(engine.AxialCoord(q, r)))
        before = engine.to_python_state(state)

        capsule = rust.state_api_capsule()

        self.assertIsNotNone(capsule)
        self.assertFalse(hasattr(rust, "state_api"))
        self.assertFalse(hasattr(rust, "state_hash"))
        self.assertFalse(hasattr(rust, "_clone_state_wire_for_testing"))
        self.assertEqual(engine.to_python_state(state), before)

    def test_illegal_action_does_not_mutate_state(self) -> None:
        import hexo_engine as engine

        state = engine.new_game()

        with self.assertRaises(engine.IllegalActionError):
            engine.apply_action(state, engine.PlacementAction(engine.AxialCoord(99, 99)))

        self.assertEqual(engine.to_python_state(state).placements_made, 0)
        self.assertIsNone(engine.terminal(state))

    def test_python_state_mirror_tracks_terminal_win(self) -> None:
        import hexo_engine as engine

        state = engine.new_game()
        for q, r in [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (2, 0),
            (1, 1),
            (1, 2),
            (3, 0),
            (4, 0),
            (2, 1),
            (2, 2),
            (5, 0),
        ]:
            engine.apply_action(state, engine.PlacementAction(engine.AxialCoord(q, r)))

        terminal = engine.terminal(state)
        mirror = engine.to_python_state(state)

        self.assertEqual(terminal.winner, engine.Player.PLAYER_0)
        self.assertEqual(terminal.reason, "six_in_line")
        self.assertEqual(mirror.terminal.winner, engine.Player.PLAYER_0)
        self.assertEqual(mirror.placements_made, 12)
        self.assertGreater(mirror.board.windows.len, 0)


if __name__ == "__main__":
    unittest.main()
