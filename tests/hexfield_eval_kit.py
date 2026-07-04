"""Shared test fixtures for the hexfield eval suites (CPU-only, no GPU/torch/.so).

Contents:

  * Engine/session fakes (used by ``test_hexfield_eval_arena``):
    ``_Terminal``, ``_FakeState``, ``_FakeApi``, ``_FakeEvaluator``,
    ``_FakeSession``, ``_make_session_factory`` — a deterministic Connect6
    engine monkeypatched onto ``eval_arena.api`` plus a fake multi-root session /
    evaluator injected through the ``make_session`` / ``build_evaluators`` seams.
    The native MCTS extension and the engine .so are not imported.

  * Orchestrator stubs (used by ``test_hexfield_eval_orchestrator`` and
    the parts tests): ``_make_run`` (fake run tree), ``_paired_match`` /
    ``_sealbot_match`` (synthetic match-result builders with a pentanomial
    block), ``_StubArena`` (records calls + returns synthetic matches), and
    ``_FakeArena`` (per-opponent scorer + visits capture).

Importing this module touches only ``hexfield.geometry`` and
``hexo_engine.types``; it does not import torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in ("hexo_engine/python", "hexfield/python"):
    _src = str(_REPO / "packages" / _p)
    if _src not in sys.path:
        sys.path.insert(0, _src)

from hexfield.geometry import pack_action_id  # noqa: E402
from hexo_engine.types import Player  # noqa: E402


# =========================================================================== #
# Engine + session fakes for the arena runner.
# =========================================================================== #
class _Terminal:
    def __init__(self, winner_label: str | None) -> None:
        # String label "player0"/"player1", or None. str(self) returns it.
        self.winner = winner_label

    def __str__(self) -> str:  # pragma: no cover
        return str(self.winner)


class _FakeState:
    """Connect6 mover schedule (0, 1,1, 0,0, 1,1, ...) over a fixed game length.

    Seedless: every game starts identical. ``winner_seat`` is set when the game
    reaches ``game_len``.
    """

    def __init__(self, *, game_len: int) -> None:
        self.game_len = game_len
        self.ply = 0
        self.actions: list[int] = []
        self.winner_seat: int | None = None  # 0/1 engine seat, or None until terminal

    def mover_seat(self) -> int:
        return 0 if self.ply == 0 else (1 if ((self.ply - 1) // 2) % 2 == 0 else 0)


class _FakeApi:
    """Stand-in for hexo_engine.api exposing the methods the arena calls."""

    Player = Player

    def __init__(self, *, game_len: int, decide_winner) -> None:
        self._game_len = game_len
        # decide_winner(state) -> engine seat int (0/1) that wins the game.
        self._decide_winner = decide_winner

    def new_game(self, *, seed=None, scenario=None):
        return _FakeState(game_len=self._game_len)

    def current_player(self, state: _FakeState) -> Player:
        return Player.PLAYER_0 if state.mover_seat() == 0 else Player.PLAYER_1

    def apply_action(self, state: _FakeState, action) -> None:
        coord = action.coord
        state.actions.append(pack_action_id(coord.q, coord.r))
        state.ply += 1
        if state.ply >= state.game_len:
            state.winner_seat = self._decide_winner(state)

    def terminal(self, state: _FakeState):
        if state.winner_seat is None:
            return None
        return _Terminal("player0" if state.winner_seat == 0 else "player1")


class _FakeEvaluator:
    def __init__(self, tag: str, strength: int) -> None:
        self.tag = tag
        self.strength = strength


class _FakeSession:
    """Records ``search`` calls and returns one deterministic move per root.

    The chosen move is a pure function of (search RNG seed, position), and does
    not depend on the game index or the evaluator. At temperature 0 the move
    depends on position only (seed-independent); at temperature > 0 it also
    depends on the seed. Only this call/return contract is modeled, not MCTS
    internals.
    """

    # Class-level shared log spanning all sessions, for test inspection.
    calls: list[dict] = []

    def __init__(self) -> None:
        self.discarded: list[int] = []

    @staticmethod
    def _move_for(seed: int, temperature: float, ply: int) -> int:
        # temp 0: position-only. temp > 0: also depends on seed. Coords stay
        # small so pack_action_id does not overflow.
        if temperature > 0.0:
            mix = (seed * 2654435761 + ply * 40503) % 7
        else:
            mix = ply % 7
        q = (mix % 5) - 2
        r = ((mix // 5) % 5) - 2
        return pack_action_id(q, r)

    def search(self, game_keys, states, *, seed, evaluator, move_temperatures, **kw):
        assert len(game_keys) == len(states) == len(move_temperatures)
        _FakeSession.calls.append(
            {
                "n_roots": len(game_keys),
                "game_keys": list(game_keys),
                "seed": seed,
                "evaluator_tag": evaluator.tag,
                "move_temperatures": list(move_temperatures),
                "visits": kw.get("visits"),
                "virtual_batch_size": kw.get("virtual_batch_size"),
            }
        )
        return [
            {"action_id": self._move_for(seed, temp, state.ply)}
            for state, temp in zip(states, move_temperatures)
        ]

    def discard(self, index: int) -> None:
        self.discarded.append(index)


def _make_session_factory():
    """Return (factory, sessions_list); factory() appends a fresh fake session."""
    sessions: list[_FakeSession] = []

    def factory():
        s = _FakeSession()
        sessions.append(s)
        return s

    return factory, sessions


# =========================================================================== #
# Orchestrator stubs: fake run tree + synthetic match-result builders + arenas.
# =========================================================================== #
def _make_run(tmp_path: Path, epochs=(5, 10, 20, 40), *, bc: bool = True) -> Path:
    """A fake run tree ``<tmp>/runs/r/checkpoints/epoch_*.pt`` (+ BC sibling).

    Layout resolved by ``select_opponents``: in-run checkpoints under
    ``<run>/checkpoints`` and the BC prefit in a sibling ``hexfield_bc_1``.
    Files contain the placeholder text "stub"; selection resolves paths and does
    not load checkpoints.
    """

    run = tmp_path / "runs" / "r"
    ckpts = run / "checkpoints"
    ckpts.mkdir(parents=True)
    for epoch in epochs:
        (ckpts / f"epoch_{epoch:06d}.pt").write_text("stub", encoding="utf-8")
    if bc:
        bc_dir = tmp_path / "runs" / "hexfield_bc_1"
        bc_dir.mkdir(parents=True)
        (bc_dir / "checkpoint_epoch2.pt").write_text("stub", encoding="utf-8")
    return run


def _paired_match_from_scores(label_a: str, label_b: str, pair_scores: list[int]) -> dict:
    """A fake ``play_checkpoint_match`` result with a pentanomial block.

    ``pair_scores`` is one net-A score per complete 2-game pair, in {0, 1, 2}
    (net-A wins among the pair's two decided games). The shape matches
    ``eval_arena._build_match_result`` + ``_pentanomial_block``: a ``score`` block
    with net-A-centric ``a_wins``, ``pentanomial.pairs`` rows, and a
    ``histogram_a_wins`` map. Consumed by the orchestrator's
    ``_checkpoint_edge_counts`` / ``_pentanomial_to_paired_result``.
    """

    pairs = []
    a_wins = b_wins = 0
    for i, score in enumerate(pair_scores):
        pairs.append(
            {
                "pair_index": i,
                "seed": 1000 + i,
                "game_indices": [2 * i, 2 * i + 1],
                "n_games": 2,
                "n_decided": 2,
                "a_wins": score,
                "b_wins": 2 - score,
                "pentanomial_a_score": score,
            }
        )
        a_wins += score
        b_wins += 2 - score
    hist = {"0": 0, "1": 0, "2": 0}
    for score in pair_scores:
        hist[str(score)] += 1
    decided = a_wins + b_wins
    return {
        "meta": {"label_a": label_a, "label_b": label_b, "games_requested": 2 * len(pair_scores)},
        "score": {
            "completed": 2 * len(pair_scores),
            "truncated": 0,
            "aborted_budget": 0,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "decided": decided,
            "a_winrate_decided": (a_wins / decided) if decided else None,
        },
        "pentanomial": {
            "n_pairs": len(pair_scores),
            "n_full_pairs": len(pair_scores),
            "pairs": pairs,
            "histogram_a_wins": hist,
        },
    }


def _paired_match(label_a: str, label_b: str, n_pairs: int, a_score: int) -> dict:
    """Fake ``play_checkpoint_match`` result for a fixed per-pair net-A score.

    ``a_score`` is the per-pair net-A score in {0, 1, 2}. It is mixed with a
    split (1) pair on odd indices so the pair-level SE is non-zero. Built on
    ``_paired_match_from_scores``, so the result shape matches that form.
    """

    pair_scores = [a_score if i % 2 == 0 else 1 for i in range(max(n_pairs, 1))]
    return _paired_match_from_scores(label_a, label_b, pair_scores)


def _sealbot_match(label: str, n: int, winrate: float) -> dict:
    """A fake ``play_sealbot_match`` result (unpaired, binomial), net-A-centric."""

    wins = int(round(winrate * n))
    return {
        "meta": {"games_requested": n},
        "score": {"completed": n, "a_wins": wins, "b_wins": n - wins, "decided": n},
    }


class _StubArena:
    """Records calls and returns synthetic matches at a configured strength.

    ``per_score`` is the per-pair net-A score (0/1/2) the candidate scores vs
    every checkpoint opponent; ``sealbot_winrate`` sets the binomial SealBot
    winrate. ``sealbot_raises`` (a callable or None) is invoked at the start of
    ``play_sealbot_match`` when set, allowing a test to raise there.
    """

    def __init__(self, *, per_score: int = 2, sealbot_winrate: float = 0.6,
                 sealbot_raises=None) -> None:
        self._per_score = per_score
        self._sealbot_winrate = sealbot_winrate
        self._sealbot_raises = sealbot_raises
        self.ckpt_calls: list[tuple[str, int]] = []
        self.sealbot_calls: list[int] = []

    def play_checkpoint_match(self, a, b, n, **kw) -> dict:
        self.ckpt_calls.append((kw["label_b"], n))
        return _paired_match(kw["label_a"], kw["label_b"], max(1, n // 2), self._per_score)

    def play_sealbot_match(self, ckpt, n, **kw) -> dict:
        if self._sealbot_raises is not None:
            self._sealbot_raises()
        self.sealbot_calls.append(n)
        return _sealbot_match(kw.get("label", "hexfield"), n, self._sealbot_winrate)


class _FakeArena:
    """Records calls and returns synthetic matches at a configured strength.

    ``ckpt_scorer(label_b, n_pairs) -> per-pair-score pattern`` sets the
    candidate's per-pair scores per opponent; ``sealbot_winrate`` sets the
    binomial SealBot winrate. The ``visits`` kwarg passed to each match is
    captured and echoed into the returned match's ``meta``.
    """

    def __init__(self, *, ckpt_scorer, sealbot_winrate: float = 0.55) -> None:
        self._ckpt_scorer = ckpt_scorer
        self._sealbot_winrate = sealbot_winrate
        self.ckpt_calls: list[tuple[str, int]] = []
        self.sealbot_calls: list[int] = []
        self.ckpt_visits: list[int | None] = []
        self.sealbot_visits: list[int | None] = []

    def play_checkpoint_match(self, a, b, n, **kw) -> dict:
        label_b = kw["label_b"]
        self.ckpt_calls.append((label_b, n))
        self.ckpt_visits.append(kw.get("visits"))
        n_pairs = max(1, n // 2)
        match = _paired_match_from_scores(kw["label_a"], label_b, self._ckpt_scorer(label_b, n_pairs))
        match["meta"]["visits"] = kw.get("visits")  # echo visits into meta
        return match

    def play_sealbot_match(self, ckpt, n, **kw) -> dict:
        self.sealbot_calls.append(n)
        self.sealbot_visits.append(kw.get("visits"))
        match = _sealbot_match(kw.get("label", "hexfield"), n, self._sealbot_winrate)
        match["meta"]["visits"] = kw.get("visits")
        return match


def _scores_for_winrate(target: float, n_pairs: int) -> list[int]:
    """A per-pair {0,1,2} pattern whose mean/2 approximates ``target`` win rate.

    Mixes decisive (2 / 0) and split (1) pairs so the pentanomial has non-zero
    pair-level variance. The rate is approximate.
    """

    if target >= 0.7:
        return [2 if i % 2 == 0 else 1 for i in range(n_pairs)]      # mean/2 ~0.75
    if target <= 0.3:
        return [0 if i % 2 == 0 else 1 for i in range(n_pairs)]      # mean/2 ~0.25
    return [2 if i % 3 == 0 else (0 if i % 3 == 1 else 1) for i in range(n_pairs)]  # mean/2 ~0.5
