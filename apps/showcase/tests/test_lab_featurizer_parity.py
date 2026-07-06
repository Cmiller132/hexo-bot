"""Client-featurizer parity: web/learn/lab_features.js must reproduce the
server featurization (shrimp.features / showcase.lab) exactly.

The fixture is generated here from the SERVER side — engine replay +
facts_from_state + build_support + build_features for sequences, and
lab.build_free_position for free-edit positions — then
``lab_featurizer_check.mjs`` recomputes everything with the client mirror
under node and compares to 1e-6.

Node is required for the JS half, so the test SKIPS (loudly) when no ``node``
binary is on PATH — the WSL CI venv has none; run the suite once on a machine
with node (any platform, the check is pure computation) after touching either
featurizer. The fixture-generation path itself always runs, so a server-side
featurizer regression still fails here even without node.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
WEB_LEARN = TESTS_DIR.parents[0] / "web" / "learn"

# Positions chosen to exercise every plane: empty board, the opening single,
# a quiet early sequence, a mid-turn (SecondStone phase + first_stone) state,
# a threat-heavy sequence (hot planes), and free-edit twins including a
# standing-win (win-now planes).
_QUIET = [
    (0, 0),
    (1, 1), (0, 2),
    (2, -1), (1, -2),
    (-1, 2), (3, 0),
    (-1, 1), (-2, 0),
    (2, 2), (-2, 3),
    (0, -1), (3, -2),
    (1, 3), (4, -1),
]

# P1 completes a split four on r=1 (stones q=1,2,3,5; gap (4,1)): opp_hot for
# the mover P0. From apps/showcase/scripts/learn_snapshots.py.
_FOUR_THREAT = [
    (0, 0),
    (1, 1), (2, 1),
    (1, 0), (2, 0),
    (3, 1), (-1, 4),
    (0, -2), (2, -2),
    (0, 4), (1, 3),
    (-2, 1), (-1, -1),
    (5, 1), (2, 3),
]

SEQUENCES = {
    "empty": [],
    "opening": [(0, 0)],
    "early": [(0, 0), (1, 1), (0, 2)],
    # 4 placements: P0 mid-turn, phase SecondStone, first_stone = (2, 0).
    "mid_turn": [(0, 0), (1, 1), (0, 2), (2, 0)],
    "quiet": _QUIET,
    "four_threat": _FOUR_THREAT,
}

FREE = {
    # The 'early' stones as a free position (history planes zeroed).
    "free_early": {"p0": [(0, 0)], "p1": [(1, 1), (0, 2)], "to_move": 0},
    # P0 standing win: five on the r=0 row with the gap at (3, 0) ->
    # own_win_now for P0 to move, opp_win_now with P1 to move.
    "free_win_now": {
        "p0": [(0, 0), (1, 0), (2, 0), (4, 0), (5, 0)],
        "p1": [(0, 1), (1, 1), (2, 1), (3, 1)],
        "to_move": 0,
    },
    "free_win_opp": {
        "p0": [(0, 0), (1, 0), (2, 0), (4, 0), (5, 0)],
        "p1": [(0, 1), (1, 1), (2, 1), (3, 1)],
        "to_move": 1,
    },
    "free_empty": {"p0": [], "p1": [], "to_move": 0},
}


def _support_block(support) -> dict:
    return {
        "coords": support.coords.tolist(),
        "legal_count": int(support.legal_count),
        "stone_count": int(support.stone_count),
        "halo_count": int(support.halo_count),
    }


def _planes(feats) -> list[list[float]]:
    return [[float(v) for v in feats[:, f].tolist()] for f in range(feats.shape[1])]


@pytest.fixture(scope="module")
def fixture_doc() -> dict:
    """Server-side ground truth for every fixture position. Building this
    exercises the server featurizer on its own (engine legality asserted)."""
    import hexo_engine as engine
    from hexo_engine.types import AxialCoord, PlacementAction
    from shrimp.engine_facts import facts_from_state
    from shrimp.features import build_features
    from shrimp.support import build_support

    from showcase.lab import build_free_position

    positions = []
    for pos_id, moves in SEQUENCES.items():
        state = engine.new_game()
        for q, r in moves:
            action = PlacementAction(AxialCoord(q=q, r=r))
            assert engine.is_legal_action(state, action), (pos_id, q, r)
            engine.apply_action(state, action)
        assert engine.terminal(state) is None, pos_id
        facts = facts_from_state(state)
        support = build_support(facts.stones())
        feats = build_features(facts, support)
        positions.append(
            {
                "id": pos_id,
                "mode": "sequence",
                "moves": [list(m) for m in moves],
                "to_move": facts.current_player,
                "phase": facts.phase,
                "support": _support_block(support),
                "planes": _planes(feats),
            }
        )
    for pos_id, spec in FREE.items():
        facts, support, feats = build_free_position(
            [tuple(c) for c in spec["p0"]],
            [tuple(c) for c in spec["p1"]],
            spec["to_move"],
        )
        positions.append(
            {
                "id": pos_id,
                "mode": "free",
                "stones": {"p0": [list(c) for c in spec["p0"]],
                           "p1": [list(c) for c in spec["p1"]]},
                "to_move": spec["to_move"],
                "phase": facts.phase,
                "support": _support_block(support),
                "planes": _planes(feats),
            }
        )
    return {"positions": positions}


def test_fixture_exercises_every_plane(fixture_doc):
    """The chosen positions light up all 15 planes somewhere (so the JS
    comparison cannot vacuously pass on an all-zero column)."""
    import numpy as np

    lit = np.zeros(15, dtype=bool)
    for pos in fixture_doc["positions"]:
        planes = np.asarray(pos["planes"], dtype=np.float64)
        lit |= (np.abs(planes) > 0).any(axis=1)
    assert lit.all(), f"planes never nonzero in any fixture position: {np.where(~lit)[0].tolist()}"


def test_client_featurizer_matches_server(fixture_doc, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip(
            "node not on PATH: the JS half of the featurizer parity check "
            "needs it — run this suite once on a machine with node after "
            "touching lab_features.js or the server featurizer"
        )
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture_doc), encoding="utf-8")
    # No package.json in the repo, so node needs the .mjs suffix to load the
    # module as ESM; check script and module are copied side by side.
    module_path = tmp_path / "lab_features.mjs"
    module_path.write_text(
        (WEB_LEARN / "lab_features.js").read_text(encoding="utf-8"), encoding="utf-8"
    )
    proc = subprocess.run(
        [node, str(TESTS_DIR / "lab_featurizer_check.mjs"),
         str(fixture_path), str(module_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
