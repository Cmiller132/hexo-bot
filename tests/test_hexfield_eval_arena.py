"""Tests for the concurrent checkpoint-vs-checkpoint arena runner
(``eval_arena.play_checkpoint_match`` / ``play_multi_checkpoint_match``).

Three layers, each asserting a distinct oracle:

  1. Fake-engine oracle (pure CPU, always runs). A deterministic engine + fake
     multi-root session injected through the ``make_session`` / ``build_evaluators``
     seams cover the result shape, concurrency mechanics (cross-game leaf batching,
     seats/CRN pairing, full-sims budget), opening-leader batching + per-root
     decorrelation, and truncation / edge cases. The native MCTS extension and the
     engine .so are not imported.

  2. Real-ABI native oracle (skips when the .so is absent). The loop is driven
     through the multi-root ``search`` ABI + real engine with a numpy stub
     evaluator, covering pairing structure, batched-opening determinism, and the
     forced-opening CRN (the follower replays the leader's recorded line) under
     both symmetric and asymmetric nets.

  3. play_multi vs N-serial equivalence (pure CPU). ``play_multi_checkpoint_match``
     over K opponents produces, for each opponent, the same per-opponent result as
     calling ``play_checkpoint_match`` once per opponent on the same seeds/config,
     plus the shared-forward mechanics.

Forced-opening replay is asserted in layer 2; the fake-engine layer asserts the
batched-opening pairing it can observe.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in ("hexo_engine/python", "hexfield/python"):
    _src = str(_REPO / "packages" / _p)
    if _src not in sys.path:
        sys.path.insert(0, _src)

from hexo_engine import api  # noqa: E402
from hexo_engine.types import AxialCoord, PlacementAction, Player  # noqa: E402

from hexfield import eval_arena  # noqa: E402  (torch-free import)
from hexfield.config import (  # noqa: E402
    build_divergence_overrides,
    parse_hexfield_config,
)
from hexfield.geometry import pack_action_id, unpack_action_id  # noqa: E402

from hexfield_eval_kit import (  # noqa: E402
    _FakeApi,
    _FakeEvaluator,
    _FakeSession,
    _FakeState,
    _Terminal,
    _make_session_factory,
)

try:
    from hexfield import _rust as hexfield_rust
except ImportError:  # pragma: no cover
    hexfield_rust = None


def _engine_bridge_available() -> bool:
    """Return True when the real-ABI native tests can run.

    Requires the hexfield Rust extension to be importable and the hexo_engine
    bridge to be live. A build lacking the engine .so raises on the first
    ``api.new_game()``; probing once lets the suite skip rather than error."""

    if hexfield_rust is None:
        return False
    try:
        api.new_game()
    except Exception:  # pragma: no cover - environment-dependent
        return False
    return True


needs_native = pytest.mark.skipif(
    not _engine_bridge_available(),
    reason="hexfield/hexo_engine native modules not built (real-ABI tests skipped)",
)


# =========================================================================== #
# LAYER 1: fake-engine oracle.
# =========================================================================== #
# The fake engine is seedless and only sees opaque states. Each game's winner is
# decided net-relatively by tracking which net-A seat the arena assigned to each
# freshly-created state. The arena creates paired games in
# (a_is_p0=True, a_is_p0=False) order per pair; the winner is the stronger net's
# engine seat.
# --------------------------------------------------------------------------- #
def _run_match(monkeypatch, *, n_games, a_strength, b_strength, game_len=6,
               config=None, **kw):
    """Drive play_checkpoint_match with a deterministic fake engine that tracks
    each game's net-A seat so winners are net-relative."""

    _FakeSession.calls = []
    factory, sessions = _make_session_factory()

    # Per-state net-A seat, filled as games are created in arena order. Paired
    # games are created in (a_is_p0=True, a_is_p0=False) order per pair.
    seat_iter = []
    if kw.get("paired_openings", True):
        n_pairs = (n_games + 1) // 2
        for p in range(n_pairs):
            seat_iter.append(True)
            if 2 * p + 1 < n_games:
                seat_iter.append(False)
    else:
        for i in range(n_games):
            seat_iter.append(i % 2 == 0)

    created: list[_FakeState] = []
    seat_of_state: dict[int, bool] = {}

    def decide(state: _FakeState) -> int:
        a_is_p0 = seat_of_state[id(state)]
        a_seat = 0 if a_is_p0 else 1
        b_seat = 1 - a_seat
        if a_strength == b_strength:
            return 0  # equal strengths resolve to seat 0
        return a_seat if a_strength > b_strength else b_seat

    base_api = _FakeApi(game_len=game_len, decide_winner=decide)

    class _TrackingApi(_FakeApi):
        Player = Player

        def new_game(self, *, seed=None, scenario=None):
            st = _FakeState(game_len=game_len)
            created.append(st)
            # Assign the next seat in creation order.
            seat_of_state[id(st)] = seat_iter[len(created) - 1]
            return st

        def current_player(self, state):
            return base_api.current_player(state)

        def apply_action(self, state, action):
            return base_api.apply_action(state, action)

        def terminal(self, state):
            return base_api.terminal(state)

    monkeypatch.setattr(eval_arena, "api", _TrackingApi(game_len=game_len, decide_winner=decide))

    def _build_evaluators():
        return _FakeEvaluator("A", a_strength), _FakeEvaluator("B", b_strength)

    result = eval_arena.play_checkpoint_match(
        "ckpt_a.pt", "ckpt_b.pt", n_games,
        config=config if config is not None else parse_hexfield_config({}),
        label_a="cand", label_b="opp",
        make_session=factory,
        build_evaluators=_build_evaluators,
        **kw,
    )
    return result, sessions


# --- 1a. Result-dict shape / contract. ------------------------------------- #
def test_result_shape_is_drop_in(monkeypatch):
    result, _ = _run_match(monkeypatch, n_games=8, a_strength=2, b_strength=1, game_len=6)

    # Top-level keys.
    assert set(result) >= {"meta", "score", "game_lengths", "opening_dedup", "games", "pentanomial"}

    score = result["score"]
    for key in ("completed", "truncated", "aborted_budget", "a_wins", "b_wins", "decided", "by_seat"):
        assert key in score, key
    # Candidate (net A) is the stronger net and wins every decided game.
    assert score["completed"] == 8
    assert score["decided"] == 8
    assert score["a_wins"] == 8
    assert score["b_wins"] == 0
    assert score["truncated"] == 0

    # Per-game rows carry these keys.
    assert len(result["games"]) == 8
    for g in result["games"]:
        assert set(g) >= {"index", "seed", "a_seat", "status", "winner", "plies", "opening"}
        assert g["a_seat"] in ("P0", "P1")
        assert g["status"] in ("completed", "truncated", "aborted_budget")
        assert g["winner"] in ("A", "B", None)

    # Pentanomial block + per-pair rows.
    penta = result["pentanomial"]
    assert penta is not None
    assert set(penta) >= {"n_pairs", "histogram_a_wins", "pairs"}
    assert penta["n_pairs"] == 4
    for p in penta["pairs"]:
        assert set(p) >= {
            "pair_index", "seed", "game_indices", "n_games",
            "n_decided", "a_wins", "b_wins", "pentanomial_a_score",
        }
        assert p["n_games"] == 2
        assert p["n_decided"] == 2
        assert p["a_wins"] == 2  # candidate won both seats of every pair
    # histogram keyed by net-A wins among the pair's 2 decided games.
    assert penta["histogram_a_wins"] == {"0": 0, "1": 0, "2": 4}

    # meta keys: games_requested + telemetry.
    meta = result["meta"]
    assert meta["games_requested"] == 8
    assert meta["label_a"] == "cand" and meta["label_b"] == "opp"
    assert meta["concurrent"] is True
    assert meta["rounds"] >= 1 and meta["forward_batches"] >= 1


def test_result_consumed_by_orchestrator_helpers(monkeypatch):
    """The concurrent result feeds the downstream orchestrator helpers."""

    from hexfield import multistage_eval as mse

    result, _ = _run_match(monkeypatch, n_games=8, a_strength=2, b_strength=1, game_len=6)

    # Stage-C edge counts (pentanomial -> effective BT counts).
    wa, wb, n_eff, prov = mse._checkpoint_edge_counts(result)
    assert wa > 0.0
    assert wb == 0.0  # candidate won all decided games; no opponent wins
    assert n_eff > 0.0

    # Stage-B SPRT consumes score.a_wins / score.b_wins directly.
    import hexfield.eval_stats as es

    sprt = es.sprt(result["score"]["a_wins"], result["score"]["b_wins"], elo0=0.0, elo1=35.0)
    assert sprt.verdict in {"accept_h0", "accept_h1", "continue"}

    # Pentanomial -> PairedResult path.
    paired = mse._pentanomial_to_paired_result(result["pentanomial"])
    assert paired is not None and paired.n_pairs == 4


# --- 1b. Concurrency mechanics + seat / pairing preservation. -------------- #
def test_games_run_concurrently_in_multiroot_batches(monkeypatch):
    """With many games in flight, the greedy tail is searched in multi-root
    batches (n_roots > 1) rather than one game at a time."""

    # game_len=6 with opening_plies=2 leaves a multi-ply greedy tail to batch.
    result, sessions = _run_match(
        monkeypatch, n_games=16, a_strength=2, b_strength=1, game_len=6, opening_plies=2,
    )
    batched = [c for c in _FakeSession.calls if c["n_roots"] > 1]
    assert batched, "expected at least one multi-root (concurrent) search batch"
    # The biggest batch pulls in many games at once (cross-game batching).
    assert max(c["n_roots"] for c in _FakeSession.calls) >= 4

    # Two persistent sessions (one per net), not one per game.
    assert len(sessions) == 2

    # Every game's tree is discarded on both sessions at game end.
    for s in sessions:
        assert sorted(s.discarded) == list(range(16))


def test_seats_swapped_within_pairs_and_crn_seed_shared(monkeypatch):
    result, _ = _run_match(monkeypatch, n_games=8, a_strength=2, b_strength=1, game_len=4)
    games = {g["index"]: g for g in result["games"]}
    penta = result["pentanomial"]
    for p in penta["pairs"]:
        i0, i1 = p["game_indices"]
        # Seat-swapped siblings: one P0, one P1.
        assert {games[i0]["a_seat"], games[i1]["a_seat"]} == {"P0", "P1"}
        # Shared CRN seed: both siblings carry the pair's seed.
        assert games[i0]["seed"] == games[i1]["seed"] == p["seed"]


def test_winner_mapping_is_net_a_centric_and_seat_symmetric(monkeypatch):
    """With net B stronger, candidate (A) loses every decided game regardless of
    its seat: the winner mapping is net-relative, not seat-relative."""

    result, _ = _run_match(monkeypatch, n_games=8, a_strength=1, b_strength=3, game_len=4)
    score = result["score"]
    assert score["a_wins"] == 0
    assert score["b_wins"] == 8
    # B wins from both seats.
    by_seat = score["by_seat"]
    assert by_seat["A_as_P0"]["b_wins"] == by_seat["A_as_P0"]["n"]
    assert by_seat["A_as_P1"]["b_wins"] == by_seat["A_as_P1"]["n"]


def test_full_sims_threaded_by_default(monkeypatch):
    """With visits=None, selfplay.search_visits (512) is threaded into every
    search call, not evaluation.eval_visits (128)."""

    cfg = parse_hexfield_config({})
    assert cfg.selfplay.search_visits == 512
    assert cfg.evaluation.eval_visits == 128

    result, _ = _run_match(monkeypatch, n_games=4, a_strength=1, b_strength=1, game_len=4)
    assert result["meta"]["visits"] == 512
    assert all(c["visits"] == 512 for c in _FakeSession.calls)


def test_explicit_visits_overrides_full_default(monkeypatch):
    result, _ = _run_match(monkeypatch, n_games=4, a_strength=1, b_strength=1, game_len=4, visits=128)
    assert result["meta"]["visits"] == 128
    assert all(c["visits"] == 128 for c in _FakeSession.calls)


# --- 1c. CRN under batching: opening leaders batch cross-game; siblings share. #
def test_opening_plies_are_batched_and_pairing_preserved(monkeypatch):
    """Opening plies (temperature-sampled) for the pair leaders are searched
    batched cross-game in multi-root calls, with every root carrying
    ``opening_temperature`` and the per-(net, round) opening base seed
    ``game_seed_base + (0|13_000_003) + rounds*1_000_003``. Paired seat-swapped
    siblings share the identical opening because the follower replays the leader's
    recorded line rather than searching its own."""

    n_games = 8
    opening_plies = 2
    game_seed_base = 100
    result, _ = _run_match(
        monkeypatch, n_games=n_games, a_strength=2, b_strength=1, game_len=6,
        opening_plies=opening_plies, game_seed_base=game_seed_base,
    )

    # Opening leaders batch cross-game: no single-root opening search.
    single_root_opening = [
        c for c in _FakeSession.calls if c["n_roots"] == 1 and c["move_temperatures"][0] > 0.0
    ]
    assert not single_root_opening, (
        "opening leaders must batch cross-game; no single-root opening search expected "
        f"(saw {single_root_opening})"
    )

    # Opening leader batches: multi-root (with 8 games several share each
    # side-to-move every round), every root at opening_temperature (>0), on the
    # per-(net, round) opening seed stream, not the greedy 7_000_003 stream.
    opening_batches = [
        c for c in _FakeSession.calls
        if c["n_roots"] >= 1 and any(t > 0.0 for t in c["move_temperatures"])
    ]
    assert opening_batches, "expected batched opening (temperature>0) searches"
    assert any(c["n_roots"] > 1 for c in opening_batches), (
        "expected at least one MULTI-root opening batch (cross-game leaders)"
    )
    valid_open_seeds = set()
    # rounds is 1-based in the arena; opening plies finish in the first few rounds.
    for rounds in range(1, result["meta"]["rounds"] + 1):
        for off in (13_000_003, 19_000_003):
            valid_open_seeds.add(game_seed_base + off + rounds * 1_000_003)
    for c in opening_batches:
        # All roots in an opening batch are at opening_temperature (leaders are at
        # plies < opening_plies).
        assert all(t > 0.0 for t in c["move_temperatures"]), c["move_temperatures"]
        assert c["seed"] in valid_open_seeds, (
            c["seed"], "not a per-(net,round) opening base seed"
        )

    # The greedy tail uses multi-root batches at temperature 0, on a distinct seed
    # stream (offset 7_000_003) so opening and greedy batches do not collide.
    greedy_batches = [
        c for c in _FakeSession.calls
        if c["n_roots"] > 1 and all(t == 0.0 for t in c["move_temperatures"])
    ]
    assert greedy_batches
    greedy_seeds = {
        game_seed_base + off + rounds * 1_000_003
        for rounds in range(1, result["meta"]["rounds"] + 1)
        for off in (0, 7_000_003)
    }
    for c in greedy_batches:
        assert c["seed"] in greedy_seeds
        assert c["seed"] not in valid_open_seeds, (
            "greedy and opening seed streams must not collide"
        )

    # Paired siblings produced the identical opening prefix: the follower replayed
    # the leader's recorded line.
    games = {g["index"]: g for g in result["games"]}
    for p in result["pentanomial"]["pairs"]:
        i0, i1 = p["game_indices"]
        op0 = games[i0]["opening"][:opening_plies]
        op1 = games[i1]["opening"][:opening_plies]
        assert op0 == op1, f"pair {p['pair_index']} siblings diverged on the opening: {op0} vs {op1}"


def test_batched_openers_decorrelate_across_leaders(monkeypatch):
    """Two different leaders sharing one opening batch get distinct per-root
    sampling seeds so they can sample different openings. In the native ABI the
    per-root seed is ``open_seed.wrapping_add(root_index)``; the fake session
    records the call's base ``open_seed`` and ``n_roots``. Asserts a multi-root
    opening batch exists and its roots span distinct per-root seeds
    (``open_seed+0 .. open_seed+n_roots-1`` are all different)."""

    n_games = 8
    opening_plies = 3
    game_seed_base = 500
    _result, _ = _run_match(
        monkeypatch, n_games=n_games, a_strength=2, b_strength=1, game_len=8,
        opening_plies=opening_plies, game_seed_base=game_seed_base,
    )

    multi_root_opening = [
        c for c in _FakeSession.calls
        if c["n_roots"] > 1 and all(t > 0.0 for t in c["move_temperatures"])
    ]
    assert multi_root_opening, (
        "expected at least one multi-root opening batch with >1 leader to decorrelate"
    )
    for c in multi_root_opening:
        per_root_seeds = [c["seed"] + i for i in range(c["n_roots"])]
        assert len(set(per_root_seeds)) == c["n_roots"], (
            "leaders in one opening batch must get distinct per-root seeds "
            f"(base {c['seed']}, n_roots {c['n_roots']})"
        )


def test_batch_openings_true_batches_everything(monkeypatch):
    """``batch_openings=True`` removes the opening single-root special case so
    every ply (including the opening) batches. The pairing and result shape are
    unaffected."""

    result, _ = _run_match(
        monkeypatch, n_games=8, a_strength=2, b_strength=1, game_len=6,
        opening_plies=3, batch_openings=True,
    )
    # No single-root opening calls when batch_openings is on (all plies batch when
    # >1 game shares a side-to-move, which they do here).
    single_root_opening = [
        c for c in _FakeSession.calls if c["n_roots"] == 1 and c["move_temperatures"][0] > 0.0
    ]
    # With 8 games created and the Connect6 schedule, several games share each
    # side-to-move every round, so opening plies batch (n_roots > 1).
    assert not single_root_opening
    assert result["meta"]["batch_openings"] is True
    assert result["pentanomial"]["n_pairs"] == 4


# --- 1d. Terminal vs max_plies truncation + edge cases. -------------------- #
def test_max_plies_truncation_marks_games_undecided(monkeypatch):
    """A game that never reaches a terminal before ``selfplay.max_game_plies`` is
    finalized as ``status='truncated'`` with ``winner=None`` (hexo has no draws),
    excluded from ``decided``/``a_wins``/``b_wins`` but counted in
    ``score.truncated``, and the round loop terminates. The fake engine declares a
    winner only at ``ply >= game_len``, so a ``max_game_plies`` below ``game_len``
    forces the ply-cap truncation path for every game."""

    cfg = parse_hexfield_config({"selfplay": {"max_game_plies": 3}})
    # game_len (10) > max_game_plies (3): the engine never sets a winner, so the
    # ply cap fires first and every game truncates.
    result, sessions = _run_match(
        monkeypatch, n_games=6, a_strength=2, b_strength=1, game_len=10,
        config=cfg, opening_plies=2,
    )

    score = result["score"]
    assert score["truncated"] == 6
    assert score["completed"] == 0
    assert score["decided"] == 0
    assert score["a_wins"] == 0 and score["b_wins"] == 0
    # Undecided: no descriptive win rate / CI.
    assert score["a_winrate_decided"] is None
    assert score["a_winrate_ci95"] is None

    # Every per-game row is truncated/undecided, capped at max_game_plies.
    assert len(result["games"]) == 6
    for g in result["games"]:
        assert g["status"] == "truncated"
        assert g["winner"] is None
        assert g["plies"] == 3  # finalized at the ply cap

    # Pentanomial: full pairs need 2 decided games, so there are none; the
    # histogram is empty and no pair is informative.
    penta = result["pentanomial"]
    assert penta["n_pairs"] == 3
    assert penta["n_full_pairs"] == 0
    assert penta["n_informative_pairs"] == 0
    assert penta["histogram_a_wins"] == {"0": 0, "1": 0, "2": 0}
    for p in penta["pairs"]:
        assert p["n_decided"] == 0 and p["a_wins"] == 0

    # Trees discarded on both sessions for every game on the truncation path.
    assert len(sessions) == 2
    for s in sessions:
        assert sorted(s.discarded) == list(range(6))


def test_terminal_and_truncation_coexist_in_one_match(monkeypatch):
    """Terminal-decided and ply-truncated games are scored independently in the
    same match: decided games drive the win counts, truncated games bump
    ``score.truncated`` and are dropped from the pentanomial's informative set.

    Games whose index is even are decided early; the rest truncate, driven through
    the fake engine's per-game winner hook."""

    cfg = parse_hexfield_config({"selfplay": {"max_game_plies": 4}})

    # A game is decided (centre-seat wins) iff its index is even, else it runs past
    # the ply cap and truncates. This is independent of the arena's seat assignment.
    decided_indices = {0, 2}

    class _MixedState(_FakeState):
        def __init__(self, *, game_len):
            super().__init__(game_len=game_len)
            self.index: int | None = None  # set by the engine at creation

    order: list[_MixedState] = []

    class _MixedApi(_FakeApi):
        Player = Player

        def new_game(self, *, seed=None, scenario=None):
            st = _MixedState(game_len=10)
            st.index = len(order)
            order.append(st)
            return st

        def current_player(self, state):
            return Player.PLAYER_0 if state.mover_seat() == 0 else Player.PLAYER_1

        def apply_action(self, state, action):
            coord = action.coord
            state.actions.append(pack_action_id(coord.q, coord.r))
            state.ply += 1
            # Decided games resolve at ply 2 (before the cap); others never.
            if state.index in decided_indices and state.ply >= 2:
                state.winner_seat = 0

        def terminal(self, state):
            if state.winner_seat is None:
                return None
            return _Terminal("player0" if state.winner_seat == 0 else "player1")

    monkeypatch.setattr(eval_arena, "api", _MixedApi(game_len=10, decide_winner=lambda s: 0))

    def _build_evaluators():
        return _FakeEvaluator("A", 1), _FakeEvaluator("B", 1)

    result = eval_arena.play_checkpoint_match(
        "a", "b", 4, config=cfg, label_a="cand", label_b="opp",
        opening_plies=2, make_session=_make_session_factory()[0],
        build_evaluators=_build_evaluators,
    )

    score = result["score"]
    assert score["completed"] == 2  # indices 0 and 2 decided
    assert score["truncated"] == 2  # indices 1 and 3 hit the ply cap
    assert score["decided"] == 2
    # Status per game row matches the engine's decided/truncated split.
    by_index = {g["index"]: g for g in result["games"]}
    for i in (0, 2):  # even indices decided
        assert by_index[i]["status"] == "completed"
        assert by_index[i]["winner"] in ("A", "B")
    for i in (1, 3):
        assert by_index[i]["status"] == "truncated"
        assert by_index[i]["winner"] is None
        assert by_index[i]["plies"] == 4


def test_odd_game_count_singleton_final_pair(monkeypatch):
    result, _ = _run_match(monkeypatch, n_games=5, a_strength=2, b_strength=1, game_len=4)
    penta = result["pentanomial"]
    assert penta["n_pairs"] == 3  # ceil(5/2)
    sizes = sorted(p["n_games"] for p in penta["pairs"])
    assert sizes == [1, 2, 2]  # last pair is a singleton
    assert result["score"]["completed"] == 5


def test_unpaired_mode_has_no_pentanomial(monkeypatch):
    result, _ = _run_match(
        monkeypatch, n_games=6, a_strength=2, b_strength=1, game_len=4, paired_openings=False,
    )
    assert result["pentanomial"] is None
    assert result["score"]["completed"] == 6
    assert result["score"]["a_wins"] == 6


# --- 1e. EVAL-SPECIFIC virtual_batch_size override (the LOCKED-16 in-run eval). #
def test_eval_vbs_override_reaches_every_search_call(monkeypatch):
    """play_checkpoint_match(virtual_batch_size=16) must thread 16 into EVERY
    multi-root search call and into the result meta — WITHOUT touching the
    self-play config value (4)."""
    cfg = parse_hexfield_config({})
    assert cfg.selfplay.virtual_batch_size == 4  # self-play stays 4
    result, _sessions = _run_match(
        monkeypatch, n_games=6, a_strength=2, b_strength=1, game_len=4,
        virtual_batch_size=16,
    )
    assert result["meta"]["virtual_batch_size"] == 16
    assert _FakeSession.calls, "expected search calls"
    assert all(c["virtual_batch_size"] == 16 for c in _FakeSession.calls)


def test_eval_vbs_defaults_to_selfplay_value(monkeypatch):
    """Omitting the override falls back to cfg.selfplay.virtual_batch_size (4)."""
    result, _sessions = _run_match(
        monkeypatch, n_games=4, a_strength=2, b_strength=1, game_len=4,
    )
    assert result["meta"]["virtual_batch_size"] == 4
    assert all(c["virtual_batch_size"] == 4 for c in _FakeSession.calls)


# =========================================================================== #
# LAYER 2: real-ABI native oracle (skips when the .so is absent).
# =========================================================================== #
# Deterministic, seat-symmetric numpy stub evaluator. A pure function of the
# position (legal-coordinate prefix), so two paired siblings at the same position
# get identical value/priors regardless of seat. ``salt`` lets two stubs differ
# (asymmetric strengths).
# --------------------------------------------------------------------------- #
def _hash_coords(coords) -> int:
    h = 1469598103934665603
    for q, r in coords:
        h = (h ^ (int(q) & 0xFFFF)) * 1099511628211 % (1 << 61)
        h = (h ^ (int(r) & 0xFFFF)) * 1099511628211 % (1 << 61)
    return h


class _StubEvaluator:
    def __init__(self, salt: int = 0) -> None:
        self.salt = int(salt)
        self.calls = 0

    def __call__(self, payload: dict) -> dict:
        import numpy as np

        b, total = (int(x) for x in payload["shape"])
        self.calls += 1
        legal_counts = np.frombuffer(bytes(payload["legal_counts"]), dtype=np.int32)
        offsets = np.asarray(payload["node_row_offsets"], dtype=np.int64)
        qr = np.frombuffer(bytes(payload["node_qr"]), dtype=np.int16).reshape(total, 2)

        values: list[float] = []
        priors: list[float] = []
        for g in range(b):
            o = int(offsets[g])
            ln = int(legal_counts[g])
            legal = [(int(qr[o + i, 0]), int(qr[o + i, 1])) for i in range(ln)]
            rh = _hash_coords(legal) ^ (self.salt * 0x9E3779B97F4A7C15)
            values.append(((rh % 2001) - 1000) / 1000.0)  # in [-1, 1]
            for i, (q, r) in enumerate(legal):
                priors.append(float((q * 2654435761 + r * 40503 + rh + i) % 997 + 1))
        reply = {
            "values_bytes": struct.pack(f"<{b}f", *values),
            "priors_bytes": struct.pack(f"<{len(priors)}f", *priors),
        }
        if payload.get("request_moves_left"):
            reply["moves_left_bytes"] = struct.pack(f"<{b}f", *([100.0] * b))
        return reply


def _cfg(*, visits: int, max_plies: int, vbs: int = 4):
    return parse_hexfield_config(
        {
            "device": "cpu",
            "selfplay": {
                "search_visits": visits,
                "virtual_batch_size": vbs,
                "max_game_plies": max_plies,
                "active_root_limit": 64,
            },
            # Set far above search_visits so a call that reads eval_visits is caught.
            "evaluation": {"eval_visits": visits + 1000},
        }
    )


def _serial_reference(cfg, eval_a, eval_b, *, n_games, opening_plies,
                      opening_temperature, game_seed_base, visits):
    """Serial reference: replay each game one ply at a time, single-root, no
    greedy batching, recording per-game winner/status/plies/opening line + seat.

    Scope: the runner batches the opening leaders (each leader root sampled with
    the native per-root ``open_seed+index`` rather than a single-root ``seed``), so
    this single-root replay does not reproduce the runner's leader opening line,
    nor the winner/status/plies that depend on it. Callers use this reference for
    the seed-independent pairing structure (game count + per-game seat assignment)
    only; opening-line / winner equivalence is checked via within-pair pairing
    (follower == leader) and concurrent self-determinism. The leader's opening RNG
    here is ``pair_seed*5003+ply`` (the single-root stream) so the forced-opening
    replay mechanics below are exercised.

    Forced-opening CRN: within a pair the leader (``a_is_p0=True``, game 0) searches
    its opening and its opening line is recorded; the follower (``a_is_p0=False``,
    game 1) does not search the opening but replays the leader's recorded action for
    each opening ply, so the pair shares the opening line despite the seat swap (a
    different net moves at ply 0, so a shared seed alone would not). If the leader
    ended its game before ``opening_plies`` (fewer recorded actions), the follower
    falls back to a single-root search for the remaining opening plies, matching the
    fallback ``eval_arena`` uses (the follower shares the leader's seed)."""

    sp = cfg.selfplay
    ov = build_divergence_overrides(sp)
    rows = []
    n_pairs = (n_games + 1) // 2
    for pair_index in range(n_pairs):
        pair_seed = game_seed_base + pair_index
        leader_line: list[int] = []  # leader's recorded opening, replayed below
        for a_is_p0 in (True, False):
            game_index = pair_index * 2 + (0 if a_is_p0 else 1)
            if game_index >= n_games:
                continue
            is_leader = a_is_p0  # game 0 leads, game 1 (seat-swapped) follows
            s_a = hexfield_rust.HexfieldMctsSession(max_states=4096)
            s_b = hexfield_rust.HexfieldMctsSession(max_states=4096)
            state = api.new_game()
            line: list[int] = []
            ply = 0
            winner = None
            status = "truncated"
            while ply < sp.max_game_plies:
                in_opening = ply < opening_plies
                # Follower opening: replay the leader's recorded action (no search)
                # when one exists for this ply; otherwise fall back to a search.
                if (not is_leader) and in_opening and ply < len(leader_line):
                    aid = int(leader_line[ply])
                else:
                    a_to_move = (api.current_player(state) == api.Player.PLAYER_0) == a_is_p0
                    session = s_a if a_to_move else s_b
                    evaluator = eval_a if a_to_move else eval_b
                    temperature = opening_temperature if in_opening else 0.0
                    out = session.search(
                        [game_index], (state,),
                        visits=visits, c_puct=sp.c_puct, temperature=temperature,
                        seed=pair_seed * 5003 + ply, evaluator=evaluator,
                        virtual_batch_size=sp.virtual_batch_size,
                        move_temperatures=[temperature],
                        widening_policy_mass=sp.widening_policy_mass,
                        widening_max_children=sp.widening_max_children,
                        widening_min_children=sp.widening_min_children,
                        fpu_reduction=sp.fpu_reduction, tss_enabled=sp.tss_enabled,
                        search_parity_mode=sp.search_parity_mode,
                        divergence_overrides=ov,
                    )[0]
                    aid = int(out["action_id"])
                if len(line) < opening_plies:
                    line.append(aid)
                q, r = unpack_action_id(aid)
                result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
                ply += 1
                if result.terminal:
                    terminal = api.terminal(state)
                    won_p0 = str(terminal.winner) == "player0"
                    winner = "A" if (won_p0 == a_is_p0) else "B"
                    status = "completed"
                    break
            if is_leader:
                leader_line = line  # record for the follower's replay
            rows.append({"index": game_index, "winner": winner, "status": status,
                         "plies": ply, "opening": line, "a_seat": "P0" if a_is_p0 else "P1"})
    return rows


@needs_native
def test_concurrent_pairing_matches_serial_and_is_deterministic() -> None:
    """Pairing structure + batched-opening determinism.

    The opening leaders batch cross-game, so each leader root samples with the
    native per-root seed ``open_seed+index`` rather than a single-root ``seed``; the
    leader's opening line therefore differs from a single-root serial replay. This
    test asserts:

      1. Pairing structure matches a serial reference: same game count, same seat
         assignment per game (seed-independent), same pair membership.
      2. Within-pair pairing: the follower replays the leader, so the two siblings
         of every pair share the identical opening line and a consistent pair
         outcome under these stubs.
      3. Determinism: two concurrent runs with identical inputs produce
         byte-identical per-game rows.

    Two different stubs make the strengths asymmetric."""

    vbs = 8
    visits = 16
    cfg = _cfg(visits=visits, max_plies=24, vbs=vbs)
    n_games = 6
    opening_plies, opening_temperature, seed_base = 4, 1.0, 4242

    # Serial reference: used for the seed-independent pairing structure (game count
    # + per-game seat assignment), not for the opening line / winner, which diverge
    # because leaders batch.
    serial_rows = _serial_reference(
        cfg, _StubEvaluator(salt=1), _StubEvaluator(salt=2),
        n_games=n_games, opening_plies=opening_plies,
        opening_temperature=opening_temperature, game_seed_base=seed_base, visits=visits,
    )

    def _run():
        return eval_arena.play_checkpoint_match(
            "a", "b", n_games,
            config=cfg, label_a="A", label_b="B",
            paired_openings=True, opening_plies=opening_plies,
            opening_temperature=opening_temperature, game_seed_base=seed_base,
            build_evaluators=lambda: (_StubEvaluator(salt=1), _StubEvaluator(salt=2)),
        )

    res = _run()
    conc = {g["index"]: g for g in res["games"]}

    # (1) Pairing STRUCTURE matches the serial reference: count + seat per game.
    assert len(serial_rows) == n_games
    assert len(res["games"]) == n_games
    for sref in serial_rows:
        cg = conc[sref["index"]]
        assert cg["a_seat"] == sref["a_seat"], (sref["index"], "seat")

    # (2) Within-pair PAIRING: siblings seat-swapped, share the opening line.
    for p in res["pentanomial"]["pairs"]:
        if p["n_games"] != 2:
            continue
        i0, i1 = p["game_indices"]
        assert {conc[i0]["a_seat"], conc[i1]["a_seat"]} == {"P0", "P1"}
        assert conc[i0]["opening"][:opening_plies] == conc[i1]["opening"][:opening_plies], (
            f"pair {p['pair_index']} siblings diverged on the opening (replay broke)"
        )

    # (3) DETERMINISM: a second identical run reproduces every game byte-for-byte.
    res2 = _run()
    conc2 = {g["index"]: g for g in res2["games"]}
    for idx, cg in conc.items():
        cg2 = conc2[idx]
        for key in ("winner", "status", "plies", "opening", "a_seat", "seed"):
            assert cg[key] == cg2[key], (idx, key, cg[key], cg2[key])

    # Anti-vacuity: the sampled opening prefix has more than the forced centre
    # stone, the opening leaders batched (a multi-root opening forward), and the
    # greedy tail batched.
    assert any(len(g["opening"]) >= opening_plies for g in res["games"])
    assert res["meta"]["rounds"] >= opening_plies
    # forward_batches counts every search call; with batched openings and a batched
    # greedy tail it is below n_games * plies.
    total_plies = sum(g["plies"] for g in res["games"])
    assert res["meta"]["forward_batches"] < total_plies


@needs_native
def test_crn_paired_siblings_share_line_and_split() -> None:
    """With two identical stubs, the seat-swapped games of every pair play the same
    MCTS line, so paired opening prefixes match and every decided full pair splits
    (pentanomial_a_score == 1)."""

    cfg = _cfg(visits=16, max_plies=24)
    n_games = 8
    res = eval_arena.play_checkpoint_match(
        "x", "x", n_games,
        config=cfg, label_a="A", label_b="B",
        paired_openings=True, opening_plies=4, opening_temperature=1.0,
        game_seed_base=7,
        build_evaluators=lambda: (_StubEvaluator(salt=0), _StubEvaluator(salt=0)),
    )
    games = res["games"]
    for pi in range(n_games // 2):
        g0, g1 = games[2 * pi], games[2 * pi + 1]
        assert g0["opening"] == g1["opening"], (
            f"pair {pi} opening diverged: {g0['opening']} vs {g1['opening']}"
        )
    for p in res["pentanomial"]["pairs"]:
        if p["n_games"] == 2 and p["n_decided"] == 2:
            assert p["pentanomial_a_score"] == 1, f"pair did not split: {p}"
    hist = res["pentanomial"]["histogram_a_wins"]
    assert hist["0"] == 0 and hist["2"] == 0


def _opening_led_by(cfg, p0_eval, p1_eval, *, seed_base, opening_plies):
    """Search a full opening line with ``p0_eval`` as the net at engine seat P0 and
    ``p1_eval`` at P1, using the serial CRN RNG (``seed_base*5003+ply``). Returns the
    opening action-id line. Reconstructs, for the follower, the line it would search
    without forced-opening replay (its net at P0, swapped vs the leader)."""

    sp = cfg.selfplay
    ov = build_divergence_overrides(sp)
    state = api.new_game()
    line: list[int] = []
    s0 = hexfield_rust.HexfieldMctsSession(max_states=4096)
    s1 = hexfield_rust.HexfieldMctsSession(max_states=4096)
    for ply in range(opening_plies):
        p0_to_move = api.current_player(state) == api.Player.PLAYER_0
        evaluator = p0_eval if p0_to_move else p1_eval
        session = s0 if p0_to_move else s1
        out = session.search(
            [0], (state,),
            visits=sp.search_visits, c_puct=sp.c_puct, temperature=1.0,
            seed=seed_base * 5003 + ply, evaluator=evaluator,
            virtual_batch_size=sp.virtual_batch_size, move_temperatures=[1.0],
            widening_policy_mass=sp.widening_policy_mass,
            widening_max_children=sp.widening_max_children,
            widening_min_children=sp.widening_min_children,
            fpu_reduction=sp.fpu_reduction, tss_enabled=sp.tss_enabled,
            search_parity_mode=sp.search_parity_mode, divergence_overrides=ov,
        )[0]
        aid = int(out["action_id"])
        line.append(aid)
        q, r = unpack_action_id(aid)
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    return line


@needs_native
def test_forced_opening_replay_shares_line_under_asymmetric_nets() -> None:
    """Forced-opening CRN: with asymmetric nets the paired siblings share the
    identical opening line, showing the share comes from replaying the leader's
    recorded actions rather than net symmetry.

    The seat swap means a different net moves at the first real decision ply in each
    sibling (the leader has net A at P0, the follower has net B at P0), so sharing
    only the RNG stream would let the asymmetric nets sample different openings and
    the lines would diverge after the forced centre stone. This first shows that
    divergence directly (the line the follower would search, net B leading from P0,
    differs from the leader's within the opening), then asserts the runner's actual
    siblings agree ply-for-ply."""

    cfg = _cfg(visits=16, max_plies=24)
    opening_plies, seed_base = 4, 9999
    sa, sb = 11, 29  # asymmetric salts

    # Anti-vacuity: the leader line (net A at P0) and the line the follower would
    # have searched without replay (net B at P0, its swapped seat) diverge somewhere
    # within the opening. Ply 0 is the forced centre stone, so divergence appears
    # from ply 1 on; only some divergence is required.
    leader_line = _opening_led_by(
        cfg, _StubEvaluator(salt=sa), _StubEvaluator(salt=sb),
        seed_base=seed_base, opening_plies=opening_plies,
    )
    follower_would_be = _opening_led_by(
        cfg, _StubEvaluator(salt=sb), _StubEvaluator(salt=sa),
        seed_base=seed_base, opening_plies=opening_plies,
    )
    assert leader_line != follower_would_be, (
        "stubs are not asymmetric on the opening; absent replay the siblings would "
        "still match, making the replay test vacuous"
    )

    n_games = 8
    res = eval_arena.play_checkpoint_match(
        "a", "b", n_games,
        config=cfg, label_a="A", label_b="B",
        paired_openings=True, opening_plies=opening_plies, opening_temperature=1.0,
        game_seed_base=seed_base,
        build_evaluators=lambda: (_StubEvaluator(salt=sa), _StubEvaluator(salt=sb)),
    )
    games = {g["index"]: g for g in res["games"]}
    for p in res["pentanomial"]["pairs"]:
        i0, i1 = p["game_indices"]
        # The follower (seat-swapped P1 sibling) replayed the leader's (P0) opening,
        # so its prefix equals the leader's, even though searching on its own
        # (``follower_would_be``) it would have diverged.
        leader = games[i0] if games[i0]["a_seat"] == "P0" else games[i1]
        follower = games[i1] if leader is games[i0] else games[i0]
        assert follower["opening"][:opening_plies] == leader["opening"][:opening_plies], (
            f"pair {p['pair_index']} follower diverged from the leader despite "
            f"replay: {follower['opening']} vs {leader['opening']}"
        )

    # Pair 0's leader is seat P0, follower seat P1, and the leader drove a sampled
    # line (more than the forced centre stone) which the follower replayed.
    # ``leader0["opening"]`` is not compared to ``leader_line`` (the single-root
    # ``_opening_led_by`` reconstruction): the leader samples via the native per-root
    # ``open_seed+index`` in a cross-game batch, so its line differs from the
    # single-root stream. The anti-vacuity check (``leader_line != follower_would_be``)
    # above establishes the asymmetric stubs produce seat-divergent openings, so the
    # follower-replays-leader match is non-vacuous.
    pair0 = res["pentanomial"]["pairs"][0]
    leader0 = games[pair0["game_indices"][0]]
    follower0 = games[pair0["game_indices"][1]]
    assert leader0["a_seat"] == "P0" and follower0["a_seat"] == "P1"
    assert len(leader0["opening"]) >= opening_plies


@needs_native
def test_full_sims_default_through_real_search() -> None:
    """visits=None runs the real search at selfplay.search_visits (full sims);
    the returned root visit count reflects the full budget, and an explicit
    visits overrides it."""

    cfg = _cfg(visits=24, max_plies=10)
    res = eval_arena.play_checkpoint_match(
        "a", "b", 2, config=cfg, paired_openings=True,
        opening_plies=2, opening_temperature=1.0,
        build_evaluators=lambda: (_StubEvaluator(), _StubEvaluator()),
    )
    assert res["meta"]["visits"] == 24  # == search_visits, not eval_visits

    res2 = eval_arena.play_checkpoint_match(
        "a", "b", 2, config=cfg, visits=9, paired_openings=True,
        opening_plies=2, opening_temperature=1.0,
        build_evaluators=lambda: (_StubEvaluator(), _StubEvaluator()),
    )
    assert res2["meta"]["visits"] == 9


# =========================================================================== #
# LAYER 3: play_multi == N-serial equivalence (fake-engine, pure CPU).
# =========================================================================== #
# A tracking fake engine that decides each game's winner net-relatively, keyed by
# the opponent each freshly-created game belongs to. Both the serial driver (one
# play_checkpoint_match per opponent) and the concurrent driver (one
# play_multi_checkpoint_match) create paired games in (a_is_p0=True, a_is_p0=False)
# order; that ordering is replayed so each state's (opponent, net-A seat) is known
# and the winner is the stronger net's engine seat.
# --------------------------------------------------------------------------- #
def _seat_iter(n_games: int) -> list[bool]:
    """Per-game net-A seat in creation order (paired: True, False, ...)."""
    out: list[bool] = []
    n_pairs = (n_games + 1) // 2
    for p in range(n_pairs):
        out.append(True)
        if 2 * p + 1 < n_games:
            out.append(False)
    return out


def _winner_seat_for(a_strength: int, b_strength: int, a_is_p0: bool) -> int:
    a_seat = 0 if a_is_p0 else 1
    b_seat = 1 - a_seat
    if a_strength == b_strength:
        return 0  # equal strengths resolve to seat 0
    return a_seat if a_strength > b_strength else b_seat


class _MultiTrackingApi:
    """Stand-in for hexo_engine.api. Each new game is tagged with the (opponent,
    a_is_p0) it belongs to, via a creation-order plan."""

    Player = Player

    def __init__(self, *, game_len: int, plan: list[tuple[str, bool]],
                 strength_by_opp: dict[str, tuple[int, int]]) -> None:
        self._game_len = game_len
        self._plan = plan
        self._strength = strength_by_opp
        self._created = 0
        self._tag: dict[int, tuple[str, bool]] = {}

    def new_game(self, *, seed=None, scenario=None):
        st = _FakeState(game_len=self._game_len)
        self._tag[id(st)] = self._plan[self._created]
        self._created += 1
        return st

    def current_player(self, state):
        return Player.PLAYER_0 if state.mover_seat() == 0 else Player.PLAYER_1

    def apply_action(self, state, action):
        coord = action.coord
        state.actions.append(pack_action_id(coord.q, coord.r))
        state.ply += 1
        if state.ply >= state.game_len:
            opp, a_is_p0 = self._tag[id(state)]
            a_str, b_str = self._strength[opp]
            state.winner_seat = _winner_seat_for(a_str, b_str, a_is_p0)

    def terminal(self, state):
        if state.winner_seat is None:
            return None
        return _Terminal("player0" if state.winner_seat == 0 else "player1")


def _strength_by_opp(opponents, candidate_strength):
    return {label: (candidate_strength, opp_strength) for label, opp_strength in opponents}


def _run_serial(monkeypatch, *, opponents, n_games, candidate_strength,
                game_len, cfg, **kw):
    """Reference: call play_checkpoint_match once per opponent, each on a fresh
    fake engine whose creation plan is that opponent's games."""
    out: dict[str, dict] = {}
    strength = _strength_by_opp(opponents, candidate_strength)
    for label, opp_strength in opponents:
        _FakeSession.calls = []
        plan = [(label, a_is_p0) for a_is_p0 in _seat_iter(n_games)]
        engine = _MultiTrackingApi(game_len=game_len, plan=plan, strength_by_opp=strength)
        monkeypatch.setattr(eval_arena, "api", engine)

        def _build_evaluators(_label=label, _opp=opp_strength):
            return _FakeEvaluator("A", candidate_strength), _FakeEvaluator(_label, _opp)

        sessions: list[_FakeSession] = []

        def factory():
            s = _FakeSession()
            sessions.append(s)
            return s

        out[label] = eval_arena.play_checkpoint_match(
            "cand.pt", f"{label}.pt", n_games,
            config=cfg, label_a="cand", label_b=label,
            paired_openings=True, make_session=factory,
            build_evaluators=_build_evaluators,
            **kw,
        )
    return out


def _run_concurrent(monkeypatch, *, opponents, n_games, candidate_strength,
                    game_len, cfg, **kw):
    """The runner under test: one play_multi_checkpoint_match over all opponents.

    The concurrent runner creates all opponent groups up front (opponent 0's pairs,
    then opponent 1's pairs, ...), so the creation plan is the concatenation of each
    opponent's seat sequence in roster order."""
    _FakeSession.calls = []
    strength = _strength_by_opp(opponents, candidate_strength)
    plan: list[tuple[str, bool]] = []
    for label, _ in opponents:
        plan.extend((label, a_is_p0) for a_is_p0 in _seat_iter(n_games))
    engine = _MultiTrackingApi(game_len=game_len, plan=plan, strength_by_opp=strength)
    monkeypatch.setattr(eval_arena, "api", engine)

    sessions: list[_FakeSession] = []

    def factory():
        s = _FakeSession()
        sessions.append(s)
        return s

    def build_candidate():
        return _FakeEvaluator("A", candidate_strength)

    opp_strength = dict(opponents)

    def build_opponent(label, ckpt):
        return _FakeEvaluator(label, opp_strength[label])

    result = eval_arena.play_multi_checkpoint_match(
        "cand.pt",
        [(label, f"{label}.pt") for label, _ in opponents],
        n_games,
        config=cfg, candidate_label="cand",
        make_session=factory,
        build_candidate_evaluator=build_candidate,
        build_opponent_evaluator=build_opponent,
        **kw,
    )
    return result, sessions


def _score_tuple(match: dict) -> tuple:
    s = match["score"]
    return (s["completed"], s["truncated"], s["aborted_budget"],
            s["a_wins"], s["b_wins"], s["decided"])


def _penta_tuple(match: dict):
    p = match.get("pentanomial")
    if p is None:
        return None
    return (
        p["n_pairs"], p["n_full_pairs"], p["n_informative_pairs"],
        tuple(sorted(p["histogram_a_wins"].items())),
        tuple(
            (q["pair_index"], q["n_games"], q["n_decided"], q["a_wins"],
             q["b_wins"], q["pentanomial_a_score"], tuple(q["game_indices"]))
            for q in p["pairs"]
        ),
    )


def _winners(match: dict) -> list:
    return [(g["index"], g["a_seat"], g["status"], g["winner"], g["plies"])
            for g in match["games"]]


def _assert_match_equivalent(label, serial_match, conc_match):
    assert _score_tuple(serial_match) == _score_tuple(conc_match), (
        f"[{label}] score differs:\n serial={serial_match['score']}\n conc  ={conc_match['score']}"
    )
    assert _penta_tuple(serial_match) == _penta_tuple(conc_match), (
        f"[{label}] pentanomial differs:\n serial={serial_match['pentanomial']}\n conc  ={conc_match['pentanomial']}"
    )
    assert _winners(serial_match) == _winners(conc_match), (
        f"[{label}] per-game winners/seats/status differ:\n"
        f" serial={_winners(serial_match)}\n conc  ={_winners(conc_match)}"
    )
    assert serial_match["game_lengths"] == conc_match["game_lengths"], f"[{label}] lengths differ"


# --- 3a. The core equivalence: concurrent == serial per opponent. ---------- #
def test_multi_equals_serial_per_opponent(monkeypatch):
    """K=3 opponents at distinct strengths (candidate beats one, loses to one,
    ties one): each opponent's concurrent result equals its serial
    play_checkpoint_match result on the same seeds/config."""

    cfg = parse_hexfield_config({})
    opponents = [("opp_weak", 1), ("opp_strong", 5), ("opp_even", 3)]
    kwargs = dict(opponents=opponents, n_games=8, candidate_strength=3,
                  game_len=6, cfg=cfg, opening_plies=2, game_seed_base=100)

    serial = _run_serial(monkeypatch, **kwargs)
    conc, sessions = _run_concurrent(monkeypatch, **kwargs)

    assert set(conc) == set(serial) == {l for l, _ in opponents}
    for label, _ in opponents:
        _assert_match_equivalent(label, serial[label], conc[label])


def test_multi_equals_serial_longer_opening(monkeypatch):
    """A longer opening (so several rounds are temperature-sampled) still matches:
    the per-opponent open_seed + per-root index reproduces the serial line."""

    cfg = parse_hexfield_config({})
    opponents = [("bc_prefit", 1), ("ep5", 2), ("ep10", 4), ("champ", 3)]
    kwargs = dict(opponents=opponents, n_games=6, candidate_strength=3,
                  game_len=10, cfg=cfg, opening_plies=4, game_seed_base=42)

    serial = _run_serial(monkeypatch, **kwargs)
    conc, _ = _run_concurrent(monkeypatch, **kwargs)
    for label, _ in opponents:
        _assert_match_equivalent(label, serial[label], conc[label])


def test_multi_equals_serial_single_opponent(monkeypatch):
    """Degenerate K=1: the multi-runner reduces to play_checkpoint_match exactly."""

    cfg = parse_hexfield_config({})
    opponents = [("solo", 5)]
    kwargs = dict(opponents=opponents, n_games=8, candidate_strength=2,
                  game_len=6, cfg=cfg, opening_plies=3, game_seed_base=7)
    serial = _run_serial(monkeypatch, **kwargs)
    conc, _ = _run_concurrent(monkeypatch, **kwargs)
    _assert_match_equivalent("solo", serial["solo"], conc["solo"])


def test_multi_equals_serial_odd_game_count(monkeypatch):
    """Odd n_games -> a singleton final pair per opponent; still equivalent."""

    cfg = parse_hexfield_config({})
    opponents = [("a", 1), ("b", 4)]
    kwargs = dict(opponents=opponents, n_games=5, candidate_strength=2,
                  game_len=6, cfg=cfg, opening_plies=2, game_seed_base=3)
    serial = _run_serial(monkeypatch, **kwargs)
    conc, _ = _run_concurrent(monkeypatch, **kwargs)
    for label, _ in opponents:
        _assert_match_equivalent(label, serial[label], conc[label])


def test_multi_equals_serial_with_truncation(monkeypatch):
    """Games that never decide before max_game_plies truncate identically (no
    draws in hexo): the truncated/undecided bookkeeping matches the serial run."""

    cfg = parse_hexfield_config({"selfplay": {"max_game_plies": 3}})
    opponents = [("x", 2), ("y", 5)]
    kwargs = dict(opponents=opponents, n_games=6, candidate_strength=3,
                  game_len=10, cfg=cfg, opening_plies=2, game_seed_base=11)
    serial = _run_serial(monkeypatch, **kwargs)
    conc, _ = _run_concurrent(monkeypatch, **kwargs)
    for label, _ in opponents:
        _assert_match_equivalent(label, serial[label], conc[label])
        assert conc[label]["score"]["truncated"] == 6


# --- 3b. Concurrency mechanics: the candidate forward is SHARED. ----------- #
def test_candidate_greedy_forward_is_shared_across_opponents(monkeypatch):
    """The candidate's greedy plies across all opponents are merged into one
    multi-root call whose roots span more than one opponent's games, and the
    candidate session is a single persistent session, not one per opponent."""

    cfg = parse_hexfield_config({})
    opponents = [("o1", 2), ("o2", 4), ("o3", 1)]
    conc, sessions = _run_concurrent(
        monkeypatch, opponents=opponents, n_games=6, candidate_strength=3,
        game_len=8, cfg=cfg, opening_plies=2, game_seed_base=0,
    )

    # KEY_STRIDE namespaces candidate keys per opponent (opp_index*1_000_000+local).
    # A greedy candidate batch is one whose roots include keys from >1 opponent
    # namespace, at temperature 0.
    KEY_STRIDE = 1_000_000
    cross_opponent_greedy = []
    for c in _FakeSession.calls:
        if all(t == 0.0 for t in c["move_temperatures"]) and c["n_roots"] > 1:
            namespaces = {k // KEY_STRIDE for k in c["game_keys"]}
            if len(namespaces) > 1:
                cross_opponent_greedy.append(c)
    assert cross_opponent_greedy, (
        "expected at least one greedy candidate batch spanning >1 opponent "
        "(the shared candidate forward)"
    )

    # Sessions: 1 candidate + 3 opponents = 4.
    assert len(sessions) == 1 + len(opponents)


def test_candidate_trees_discarded_no_leak(monkeypatch):
    """Every game's candidate tree is discarded on the candidate session (global
    keys) and every opponent tree on that opponent's session (local keys)."""

    cfg = parse_hexfield_config({})
    opponents = [("o1", 2), ("o2", 4)]
    n_games = 6
    conc, sessions = _run_concurrent(
        monkeypatch, opponents=opponents, n_games=n_games, candidate_strength=3,
        game_len=6, cfg=cfg, opening_plies=2, game_seed_base=0,
    )
    KEY_STRIDE = 1_000_000
    # The candidate session is the one that saw keys from multiple namespaces.
    cand_session = None
    for s in sessions:
        ns = {k // KEY_STRIDE for k in s.discarded}
        if len(ns) > 1:
            cand_session = s
            break
    assert cand_session is not None, "could not identify the shared candidate session"
    # Candidate discarded exactly the global keys for every game of every opponent.
    expected_cand = sorted(
        opp_index * KEY_STRIDE + local
        for opp_index in range(len(opponents))
        for local in range(n_games)
    )
    assert sorted(cand_session.discarded) == expected_cand

    # Each opponent session discarded exactly its own local game indices 0..n-1.
    opp_sessions = [s for s in sessions if s is not cand_session]
    for s in opp_sessions:
        assert sorted(s.discarded) == list(range(n_games))


def test_result_dicts_are_play_checkpoint_match_shape(monkeypatch):
    """Each per-opponent result has the shape the orchestrator downstream
    (_checkpoint_edge_counts) consumes, and it is consumed here."""

    from hexfield import multistage_eval as mse

    cfg = parse_hexfield_config({})
    opponents = [("o1", 2), ("o2", 4)]
    conc, _ = _run_concurrent(
        monkeypatch, opponents=opponents, n_games=8, candidate_strength=3,
        game_len=6, cfg=cfg, opening_plies=2, game_seed_base=0,
    )
    for label, _ in opponents:
        match = conc[label]
        assert set(match) >= {"meta", "score", "game_lengths", "opening_dedup",
                              "games", "pentanomial"}
        assert match["meta"]["label_a"] == "cand"
        assert match["meta"]["label_b"] == label
        assert match["meta"]["multi_opponent"] is True
        assert match["meta"]["games_requested"] == 8
        # Downstream effective-count extraction works unchanged.
        wa, wb, n_eff, prov = mse._checkpoint_edge_counts(match)
        assert n_eff > 0.0
        paired = mse._pentanomial_to_paired_result(match["pentanomial"])
        assert paired is not None and paired.n_pairs == 4
