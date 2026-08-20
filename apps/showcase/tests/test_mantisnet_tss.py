"""TSS-Gumbel: the MantisNet family's Threat-Space Search layer.

The searches here run against a STUB evaluator — deterministic priors and
values derived from each position's Zobrist hash — so every assertion is about
TSS and the session, never about what a checkpoint happens to think. The stub
is also what the golden vectors below were recorded with.

What is pinned:

* the line mirror is exact — every leaf the session pumps is the position the
  mirrored path replays to, in the showcase engine and in MantisNet's;
* λ¹ reaches the played move (a win-now turn wins, a forced-defence turn
  defends) and hard values reach the leaves;
* a verified deep root win overrides the decision and raises the badge;
* TSS off reproduces the pre-TSS search byte for byte, and TSS on where there
  is nothing to prove decides identically.
"""

from __future__ import annotations

import hashlib

import pytest

_SEED_MASK = (1 << 64) - 1

# --- fixtures (engine placement sequences, P0 opens at the origin) -----------

# hexfield_eq's win-now fixture plus one quiet P0 stone: P0 holds five in a row
# with ONE placement left, so exactly the two completions are win-now moves.
WIN_NOW_ONE_STONE = [
    (0, 0), (0, 8), (2, 7), (1, 0), (2, 0), (4, 6), (6, 5), (3, 0), (4, 0),
    (8, 4), (10, 3), (0, -5),
]
# hexfield_eq's forced-defence fixture: P1 to move against P0's five in a row.
# (-1, 0) and (5, 0) are the only cells that answer it.
FORCED_DEFENCE = [
    (0, 0), (0, 8), (2, 7), (1, 0), (2, 0), (4, 6), (6, 5), (3, 0), (4, 0),
]
# λ¹ says nothing here (no own win-now, no opponent threat) but the deep solver
# proves a win for P0 starting at (5, 4).
DEEP_ROOT_WIN = [
    (0, 0), (0, 8), (2, 7), (1, 0), (2, 0), (4, 6), (6, 5), (3, 0), (0, 4),
    (8, 4), (10, 3), (1, 4), (2, 4), (12, 2), (14, 1), (3, 4),
]
DEEP_ROOT_WIN_MOVE = (5, 4)
# A game's opening as GameSession.create leaves it, and a scattered midgame:
# no >=4 window anywhere in either, so TSS has nothing to say.
OPENING = [(0, 0)]
QUIET_MIDGAME = [
    (0, 0), (3, 1), (1, 4), (-2, 2), (2, -3), (5, 0), (-1, -1), (4, 4), (0, 6),
]


# --- the stub evaluator -----------------------------------------------------


def _stream(seed: int, count: int) -> list[float]:
    """A deterministic float stream in (-4, 4) from a 64-bit seed."""
    values = []
    state = seed & _SEED_MASK
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & _SEED_MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _SEED_MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _SEED_MASK
        z ^= z >> 31
        values.append((z % 8000) / 1000.0 - 4.0)
    return values


class _StubImproved:
    def __init__(self, probs, v_hat) -> None:
        self.probs = probs
        self.v_hat = v_hat


class _StubRead:
    """The slice of `mantisnet.read_positions`' result the driver consumes."""

    def __init__(self, logits, value, probs, v_hat) -> None:
        self.logits = logits
        self.value = value
        self.improved = _StubImproved(probs, v_hat)

    def row(self, index: int) -> int:
        return index


class StubEvaluator:
    """Deterministic per-position priors and values, keyed by Zobrist hash."""

    def read(self, positions):
        import torch

        logits, value, probs, v_hat = [], [], [], []
        for position in positions:
            raw = _stream(int(position.zobrist), position.legal_count)
            row = torch.tensor(raw, dtype=torch.float32)
            logits.append(row)
            probs.append(torch.softmax(row, dim=0))
            scalar = ((int(position.zobrist) >> 33) % 2001 - 1000) / 1000.0
            value.append(scalar)
            v_hat.append(scalar)
        return _StubRead(logits, value, probs, v_hat)


# --- helpers ----------------------------------------------------------------


def _engine_state(moves):
    import hexo_engine as engine
    from hexo_engine.types import AxialCoord, PlacementAction

    state = engine.new_game()
    for q, r in moves:
        engine.apply_action(state, PlacementAction(AxialCoord(int(q), int(r))))
    return state


def _run(moves, *, tss, visits=16, candidates=16, seed=7, temperature=0.0):
    from showcase.families.mantisnet_family import _run_search

    return _run_search(
        StubEvaluator(),
        [(int(q), int(r)) for q, r in moves],
        candidates=candidates,
        visits=visits,
        seed=seed,
        temperature=temperature,
        game_key=11,
        tss=tss,
        telemetry_callback=None,
    )


def _decision(result):
    from hexo_engine.types import unpack_coord_id

    coord = unpack_coord_id(int(result["action_id"]))
    return (int(coord.q), int(coord.r))


def _policy_digest(result) -> str:
    blob = b"".join(
        bytes(result[key])
        for key in (
            "visit_policy_action_ids_bytes",
            "visit_policy_weights_bytes",
            "visit_policy_q_bytes",
        )
    )
    return hashlib.sha256(blob).hexdigest()[:32]


def _random_game(rng, plies):
    """A legal placement sequence of `plies` stones, P0 opening at the origin."""
    from mantisnet import _rust as mn

    position = mn.Position.replay([(0, 0)])
    moves = [(0, 0)]
    while len(moves) < plies and not position.is_terminal:
        legal = position.legal_moves()
        q, r = legal[rng.randrange(len(legal))]
        position.advance(int(q), int(r))
        moves.append((int(q), int(r)))
    return moves


def _player_name(index: int) -> str:
    return "player0" if int(index) == 0 else "player1"


# --- 1. the line mirror is exact --------------------------------------------


def test_mirrored_leaf_paths_are_the_pumped_positions():
    """Every leaf the session pumps is exactly what its mirrored path replays to.

    The whole TSS claim rests on this: a solve is submitted for `root + path`,
    so a path that named a different position than the leaf it was computed for
    would answer the wrong question with a proof.
    """
    import random

    import torch

    import hexo_engine as engine
    from mantisnet import _rust as mn
    from showcase.families.mantisnet_family import (
        _advance_lines, _init_lines, _match_leaves, _prior_argmax,
    )

    rng = random.Random(20260819)
    evaluator = StubEvaluator()
    checked = 0
    for game in range(6):
        moves = _random_game(rng, 6 + 3 * game)
        root = mn.Position.replay(moves)
        if root.is_terminal:
            continue
        search = mn.GumbelSearch(24, 8, 0.0, 1000 + game)
        search.begin(root)
        lines = None
        root_answered = False
        while True:
            decided, leaves = search.pump()
            snap = search.snapshot()
            if lines is None and snap is not None:
                lines = _init_lines(root, snap)
            if decided:
                break
            read = evaluator.read([leaf for _key, leaf in leaves])
            if not root_answered:
                root_answered = True
                for j, (key, _leaf) in enumerate(leaves):
                    priors = torch.softmax(read.logits[read.row(j)], dim=0).tolist()
                    search.resume(key, priors, float(read.value[j]))
                continue
            order = _match_leaves(lines, _advance_lines(lines, snap), leaves)
            for j, (key, leaf) in enumerate(leaves):
                line = lines[order[j]]
                mirror = engine.to_python_state(_engine_state(moves + line.path))
                assert {
                    (int(coord.q), int(coord.r), player.value)
                    for coord, player in mirror.board.stones
                } == {
                    (int(q), int(r), _player_name(p)) for q, r, p in leaf.stones()
                }
                assert int(mirror.placements_made) == int(leaf.stone_count)
                assert len(moves) + len(line.path) == int(leaf.stone_count)
                assert mirror.current_player.value == _player_name(leaf.current_player)
                # phase budget: 2 before a turn's first stone, 1 otherwise
                budget = 2 if mirror.phase.value == "FirstStone" else 1
                assert budget == int(leaf.moves_remaining)
                # ... and MantisNet's own replay of the same path.
                assert int(mn.Position.replay(moves + line.path).zobrist) == int(
                    leaf.zobrist
                )
                checked += 1
                priors = read.improved.probs[read.row(j)].tolist()
                search.resume(key, priors, float(read.improved.v_hat[j]))
                line.policy_rank = _prior_argmax(priors)
                line.evaluated = True
    assert checked >= 50, f"only {checked} leaves exercised"


def test_a_drifted_mirror_is_a_hard_error():
    """An unmatched leaf raises rather than solving some other position."""
    from mantisnet import _rust as mn
    from showcase.families.mantisnet_family import _Line, _match_leaves

    root = mn.Position.replay(QUIET_MIDGAME)
    legal = root.legal_moves()
    mirrored = root.copy()
    mirrored.advance(*legal[0])
    pumped = root.copy()
    pumped.advance(*legal[1])
    with pytest.raises(RuntimeError, match="matches no mirrored line"):
        _match_leaves([_Line(mirrored, [legal[0]])], [0], [(0, pumped)])


# --- 2. λ¹ reaches the played move ------------------------------------------


def test_win_now_turn_plays_a_winning_move():
    import hexo_engine as engine
    from hexo_engine.types import AxialCoord, PlacementAction
    from showcase.families.mantis_tss import TssConfig

    result = _run(WIN_NOW_ONE_STONE, tss=TssConfig())
    played = _decision(result)
    assert played in {(-1, 0), (5, 0)}
    state = _engine_state(WIN_NOW_ONE_STONE)
    engine.apply_action(state, PlacementAction(AxialCoord(*played)))
    assert engine.terminal(state) is not None, (
        "the win-now guard must play a completing cell"
    )
    assert result["tss"]["lambda1_root_guard"] == 1
    assert result["tss"]["root_status"] == "lambda1_win"


def test_forced_defence_turn_plays_a_hitting_cell():
    from showcase.families.mantis_tss import TssConfig

    result = _run(FORCED_DEFENCE, tss=TssConfig())
    # Every other move leaves P0's five in a row alive, so λ¹ refutes it and
    # the guard zeroes it.
    assert _decision(result) in {(-1, 0), (5, 0)}
    assert result["tss"]["lambda1_root_guard"] == 1


def test_lambda1_hard_values_reach_the_leaves():
    """Leaves whose λ¹ is decided answer ±1 rather than the stub's value."""
    from showcase.families.mantis_tss import TssConfig

    result = _run(FORCED_DEFENCE, tss=TssConfig(), visits=32)
    assert result["tss"]["lambda1_leaf_hits"] > 0


# --- 3. the verified deep root solve ----------------------------------------


def test_deep_root_win_overrides_the_decision():
    from mantisnet import _rust as mn
    from showcase.families.mantis_tss import TssConfig

    off = _run(DEEP_ROOT_WIN, tss=TssConfig(enabled=False))
    assert _decision(off) != DEEP_ROOT_WIN_MOVE, (
        "the fixture needs the bare search to pick something else"
    )
    result = _run(DEEP_ROOT_WIN, tss=TssConfig())
    assert _decision(result) == DEEP_ROOT_WIN_MOVE
    assert result["action_selection"] == "tss_deep_root_win"
    assert result["tss"]["root_status"] == "win"
    assert result["tss"]["deep_win"] >= 1
    assert result["tss"]["verify_failed"] == 0
    assert result["root_value"] == pytest.approx(1.0)
    assert DEEP_ROOT_WIN_MOVE in mn.Position.replay(DEEP_ROOT_WIN).legal_moves()
    ids = list(memoryview(bytes(result["visit_policy_action_ids_bytes"])).cast("I"))
    assert int(result["action_id"]) in ids


def test_a_path_off_the_root_is_refused():
    """The probe never silently truncates a replay."""
    from hexfield_eq import _rust as hx

    probe = hx.TssProbe(_engine_state(QUIET_MIDGAME))
    with pytest.raises(ValueError, match="illegal from the search root"):
        probe.lambda1([(0, 0)], [])


# --- 4. parity: TSS off is the pre-TSS search -------------------------------

_GOLDEN_CASES = {
    "deep_root_win/7/16": (DEEP_ROOT_WIN, 7, 16, 16),
    "forced_defence/7/16": (FORCED_DEFENCE, 7, 16, 16),
    "opening/23/32": (OPENING, 23, 32, 16),
    "opening/7/16": (OPENING, 7, 16, 16),
    "quiet/23/32": (QUIET_MIDGAME, 23, 32, 16),
    "quiet/7/16": (QUIET_MIDGAME, 7, 16, 16),
}

# Recorded from the pre-TSS driver (commit 7f27674's mantisnet_family.py) with
# the StubEvaluator above: (action_id, visits, root_value, policy digest), the
# digest being the first 32 hex of sha256 over the three visit_policy buffers.
# Regenerate only against that driver — these pin that TSS off changed nothing.
PRE_TSS_GOLDENS: dict[str, tuple] = {
    "deep_root_win/7/16": (2146992133, 16, 0.201, "731b0f0b27567d10878f763e21ee5ffe"),
    "forced_defence/7/16": (2147188738, 16, 0.412, "2b40e0de46004ecf78d313b0df829302"),
    "opening/23/32": (2147581958, 32, 0.964, "58aac182a28bc47455f56b2c91887f6e"),
    "opening/7/16": (2147385350, 16, 0.868, "d61293394e39778e113fbfff2b3b6836"),
    "quiet/23/32": (2147450876, 32, 0.885, "504507e3b05179cce4e6475b11023188"),
    "quiet/7/16": (2147450876, 16, 0.885, "b88175809d520f27d4d7946af7e3d486"),
}


@pytest.mark.parametrize("case", sorted(_GOLDEN_CASES))
def test_tss_off_reproduces_the_pre_tss_search(case):
    from showcase.families.mantis_tss import TssConfig

    moves, seed, visits, candidates = _GOLDEN_CASES[case]
    result = _run(
        moves, tss=TssConfig(enabled=False), visits=visits,
        candidates=candidates, seed=seed,
    )
    assert (
        int(result["action_id"]),
        int(result["visits"]),
        round(float(result["root_value"]), 6),
        _policy_digest(result),
    ) == PRE_TSS_GOLDENS[case]
    # TSS off adds no keys at all.
    assert "tss" not in result and "action_selection" not in result


@pytest.mark.parametrize(
    "moves", [OPENING, QUIET_MIDGAME], ids=["opening", "quiet_midgame"]
)
def test_tss_on_is_inert_where_there_is_nothing_to_prove(moves):
    """No >=4 window anywhere: the guard is inert and every leaf takes the net."""
    from showcase.families.mantis_tss import TssConfig

    on = _run(moves, tss=TssConfig(), visits=32, seed=23)
    off = _run(moves, tss=TssConfig(enabled=False), visits=32, seed=23)
    assert {
        key: value for key, value in on.items()
        if key not in ("tss", "action_selection")
    } == off
    assert on["tss"]["lambda1_leaf_hits"] == 0
    assert on["tss"]["lambda1_root_guard"] == 0
    assert on["action_selection"] == "gumbel_sh_score"


# --- 5. the real miss: game 34e4cb07 ----------------------------------------

# Placement prefixes from the one game mantis-cellnodes1-it402 lost on the site
# (34e4cb07, sims 128). Literals, so this needs neither the game record nor the
# live checkpoint. At both positions the deep solver proves a win for the side
# to move and the bot played neither, because the ROOT solve ran at the LEAF
# cap of 500 while the proofs need 1577 and 12880 nodes.
LOST_GAME_IDX25 = [
    (0, 0), (1, 2), (0, 1), (3, -1), (2, 2), (2, 0), (2, 6), (1, 6), (4, 1),
    (7, -2), (5, -1), (7, -1), (-2, 8), (-2, 4), (0, 7), (-3, 9), (-4, 10),
    (0, 6), (1, 1), (1, 3), (0, 4), (0, 8), (-1, 5), (-1, 3), (0, 10),
]
LOST_GAME_IDX25_WIN = (1, 7)      # proven in 1577 solver nodes
LOST_GAME_IDX34 = LOST_GAME_IDX25 + [
    (-2, 1), (-1, 1), (3, 1), (-3, 1), (-2, 2), (-2, 3), (-2, 5), (-2, 0),
    (1, 7),
]
LOST_GAME_IDX34_WIN = (-1, 9)     # proven in 12880 solver nodes

_MISSES = [
    (LOST_GAME_IDX25, LOST_GAME_IDX25_WIN),
    (LOST_GAME_IDX34, LOST_GAME_IDX34_WIN),
]


@pytest.mark.parametrize("prefix,proven", _MISSES, ids=["idx25", "idx34"])
def test_the_missed_forced_wins_are_proven_at_the_default_root_cap(prefix, proven):
    from showcase.families.mantis_tss import TssConfig

    off = _run(prefix, tss=TssConfig(enabled=False))
    assert _decision(off) != proven, (
        "the fixture needs the bare search to miss the win, as the live bot did"
    )
    on = _run(prefix, tss=TssConfig())
    assert _decision(on) == proven
    assert on["action_selection"] == "tss_deep_root_win"
    assert on["tss"]["root_status"] == "win"
    assert on["tss"]["root_timeouts"] == 0
    assert on["tss"]["verify_failed"] == 0
    assert on["root_value"] == pytest.approx(1.0)


@pytest.mark.parametrize("prefix,proven", _MISSES, ids=["idx25", "idx34"])
def test_the_leaf_cap_alone_would_not_have_found_them(prefix, proven):
    """Why the root needs its own cap: at 500 nodes neither proof exists."""
    from showcase.families.mantis_tss import TssConfig

    off = _run(prefix, tss=TssConfig(enabled=False))
    capped = _run(prefix, tss=TssConfig(root_node_cap=500))
    assert capped["tss"]["root_status"] == "unknown"
    assert capped["action_selection"] == "gumbel_sh_score"
    assert _decision(capped) == _decision(off) != proven


def test_the_root_budget_is_its_own_clock():
    """A 1 ms root budget drops the solve; the search's own move stands."""
    from showcase.families.mantis_tss import TssConfig

    off = _run(LOST_GAME_IDX34, tss=TssConfig(enabled=False))
    starved = _run(LOST_GAME_IDX34, tss=TssConfig(root_wall_budget_ms=1))
    assert starved["tss"]["root_status"] == "timeout"
    assert starved["tss"]["root_timeouts"] == 1
    assert starved["tss"]["deep_timeouts"] == 0, "a root drop is not a leaf drop"
    assert starved["action_selection"] == "gumbel_sh_score"
    assert _decision(starved) == _decision(off)


# --- 6. the per-game toggle reaches the family ------------------------------


def _mantis_catalogue(path, checkpoint):
    toml = path / "bots.toml"
    toml.write_text(
        f"""sims = [8, 16]

[[checkpoint]]
id = "tiny-mantis"
family = "mantisnet"
checkpoint = "{checkpoint.as_posix()}"
label = "Tiny MantisNet"
run = "showcase_tiny_mantis"
epoch = 0
""",
        encoding="utf-8",
    )
    return toml


def test_worker_turn_honours_the_tss_flag(tiny_mantis_checkpoint, tmp_path):
    """`bot_turn`'s flag reaches the family: TSS on reports counters, off does
    not report them at all."""
    from hexo_engine.types import AxialCoord, pack_coord_id
    from showcase.bots import load_bots_toml
    from test_mantisnet_family import _settings

    toml = _mantis_catalogue(tmp_path, tiny_mantis_checkpoint)
    catalogue = load_bots_toml(toml)
    from showcase.bots import _WorkerRuntime

    runtime = _WorkerRuntime(
        list(catalogue.checkpoints), _settings(toml, tmp_path), device_override="cpu"
    )
    actions = [int(pack_coord_id(AxialCoord(0, 0)))]

    def frames_for(flag):
        frames: list[dict] = []
        runtime.bot_turn(
            bot_slug="tiny-mantis", game_key=5, actions=list(actions), seed=3,
            visits=8, tss_enabled=flag, progress_callback=frames.append,
        )
        return [f for f in frames if f.get("phase") == "complete"]

    on = frames_for(True)
    off = frames_for(False)
    assert on and off
    assert all("tss_stats" not in frame for frame in off)
    for frame in on:
        assert set(frame["tss_stats"]) >= {
            "lambda1_leaf_hits", "deep_attempted", "deep_timeouts",
            "root_status", "root_ms", "total_ms",
        }
        assert frame["action_selection"] in (
            "gumbel_sh_score", "tss_deep_root_win"
        )


def test_create_game_carries_the_tss_flag(tiny_mantis_checkpoint, tmp_path):
    """POST /api/game's `tss` lands on the session the bot turn reads."""
    from fastapi.testclient import TestClient
    from test_showcase_api import create_game, fresh_ip, resign
    from test_mantisnet_family import _settings

    from showcase.app import create_app

    toml = _mantis_catalogue(tmp_path, tiny_mantis_checkpoint)
    with TestClient(create_app(_settings(toml, tmp_path))) as client:
        headers = fresh_ip()
        game = create_game(
            client, headers, checkpoint_id="tiny-mantis", sims=8, tss=False,
        )
        assert client.app.state.sessions[game["id"]].tss_enabled is False
        # ... and the default is on.
        other = create_game(
            client, fresh_ip(), checkpoint_id="tiny-mantis", sims=8,
        )
        assert client.app.state.sessions[other["id"]].tss_enabled is True
        resign(client, game["id"], headers)


# --- 7. config -------------------------------------------------------------


def test_tss_config_refuses_nonsense():
    from showcase.families.mantis_tss import TssConfig

    with pytest.raises(ValueError, match="node_cap"):
        TssConfig(node_cap=0)
    with pytest.raises(ValueError, match="root_node_cap"):
        TssConfig(root_node_cap=0)
    with pytest.raises(ValueError, match="leaf_gate"):
        TssConfig(leaf_gate="sometimes")
    with pytest.raises(ValueError, match="workers"):
        TssConfig(workers=0)
    with pytest.raises(ValueError, match="wall_budget_ms"):
        TssConfig(wall_budget_ms=0)
    with pytest.raises(ValueError, match="root_wall_budget_ms"):
        TssConfig(root_wall_budget_ms=0)
    with pytest.raises(ValueError, match=r"unknown \[tss\] profile keys"):
        TssConfig.from_profile({"node_capp": 100})
    # every knob is settable from a profile, and nothing else is
    assert TssConfig.from_profile({
        "enabled": False, "node_cap": 7, "root_node_cap": 9, "leaf_gate": "all",
        "workers": 2, "wall_budget_ms": 11, "root_wall_budget_ms": 13,
    }) == TssConfig(
        enabled=False, node_cap=7, root_node_cap=9, leaf_gate="all",
        workers=2, wall_budget_ms=11, root_wall_budget_ms=13,
    )
    assert TssConfig.from_profile(None) == TssConfig()
    # the shipped defaults: the root gets 40x the leaf cap and its own clock
    shipped = TssConfig()
    assert (shipped.node_cap, shipped.root_node_cap) == (500, 20_000)
    assert (shipped.wall_budget_ms, shipped.root_wall_budget_ms) == (1500, 3000)
    # the toggle changes `enabled` and nothing else
    assert TssConfig().with_enabled(False) == TssConfig(enabled=False)


# --- 8. the solver endpoint (solve_position) ---------------------------------


@pytest.mark.parametrize("prefix,proven", _MISSES, ids=["idx25", "idx34"])
def test_solve_position_proves_the_missed_wins(prefix, proven):
    from mantisnet import _rust
    from showcase.families.mantis_tss import TssConfig, solve_position

    out = solve_position(prefix, TssConfig(), line_cap=100)
    assert out["status"] == "win"
    assert out["source"] == "deep"
    assert (out["proven"]["q"], out["proven"]["r"]) == proven
    assert out["line"] and (out["line"][0]["q"], out["line"][0]["r"]) == proven
    assert out["nodes"] > 0
    assert out["ms"] > 0
    # The line must replay as legal plies from the solved position, and the
    # walk runs the win down to the board: it ends on the winning six. (The
    # terminal assertion is also the anti-waltz pin — the old walk once
    # marched quiet positions toward infinity and never finished.)
    end = _rust.Position.replay(
        list(prefix) + [(c["q"], c["r"]) for c in out["line"]]
    )
    assert end.is_terminal, "the demonstration line must land the win"
    # The defense is uniquely forced for a leading stretch and never longer
    # than the line itself.
    assert 1 <= out["forced_through"] <= len(out["line"])


def test_solve_position_answers_a_win_now_from_lambda1():
    from showcase.families.mantis_tss import TssConfig, solve_position

    out = solve_position(WIN_NOW_ONE_STONE, TssConfig())
    assert out["status"] == "win"
    assert out["source"] == "lambda1"
    assert (out["proven"]["q"], out["proven"]["r"]) in {(-1, 0), (5, 0)}
    assert out["guard"] is not None
    marked = {
        (row["q"], row["r"]) for row in out["guard"] if row["cls"] == 1
    }
    assert (out["proven"]["q"], out["proven"]["r"]) in marked


def test_solve_position_reports_unknown_within_a_tiny_cap():
    """idx25's proof needs 1577 nodes; a 50-node cap must answer honestly."""
    from showcase.families.mantis_tss import TssConfig, solve_position

    out = solve_position(LOST_GAME_IDX25, TssConfig(root_node_cap=50))
    assert out["status"] == "unknown"
    assert out["source"] == "deep"
    assert out["proven"] is None
    assert out["line"] == []


def test_solve_position_refuses_a_terminal_position():
    from showcase.families.mantis_tss import TssConfig, solve_position

    win = [
        (0, 0), (0, 3), (1, 3), (1, 0), (2, 0), (2, 3),
        (3, 3), (3, 0), (4, 0), (4, 3), (5, 3),
    ]
    with pytest.raises(ValueError, match="terminal"):
        solve_position(win, TssConfig())


# --- cooperative cancel (the zombie-search regression) ------------------------

def _corpus_live_position():
    """0l4291i_live: 63 stones, a deep forced win no quick solve cracks —
    exactly the shape whose abandoned solve used to churn for hours."""
    import json
    from pathlib import Path

    corpus_path = (
        Path(__file__).resolve().parents[1]
        / "web" / "learn" / "data" / "forcing_corpus.json"
    )
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    entry = next(p for p in corpus["positions"] if p["id"] == "0l4291i_live")
    return [(int(q), int(r)) for q, r in entry["moves"]]


def test_deep_solve_cancel_drains_promptly():
    """cancel_deep_solve must end an in-flight solve through the solver's
    budget-exhaustion exits: status unknown (never a value), the real node
    count, and a prompt return instead of churning to the node cap."""
    import threading
    import time

    from hexfield_eq import _rust as hx

    probe = hx.TssProbe(_engine_state(_corpus_live_position()))
    # 20M nodes is minutes of search on this position; the cancel at 0.5s must
    # end it in seconds.
    timer = threading.Timer(0.5, probe.cancel_deep_solve)
    timer.start()
    try:
        started = time.monotonic()
        out = probe.deep_solve([], 20_000_000)
        wall = time.monotonic() - started
    finally:
        timer.cancel()
    assert str(out["status"]) == "unknown"
    assert int(out["nodes"]) > 0
    assert wall < 15.0, f"cancelled solve took {wall:.1f}s to drain"


def test_solve_position_timeout_cancels_and_reports_the_work():
    """A wall-budget timeout now cancels the Rust solve and harvests it: the
    payload stays status "timeout" but carries the real node count, and the
    worker thread is drained rather than left searching."""
    from showcase.families.mantis_tss import TssConfig, solve_position

    out = solve_position(
        _corpus_live_position(),
        TssConfig(root_node_cap=20_000_000, root_wall_budget_ms=1_000),
    )
    assert out["status"] == "timeout"
    assert out["source"] == "deep"
    assert out["proven"] is None
    assert out["line"] == []
    assert out["nodes"] > 0, "a harvested cancel must report its node count"
    # Wall time = budget + drain, nowhere near a full 20M-node search.
    assert out["ms"] < 20_000.0
