"""Shared primitives for the hexfield eval match runners (Layer A engine).

This module holds the reusable, net-A-centric building blocks the four
``eval_arena.play_*`` runners duplicate today: the in-flight ``EvalGame`` record,
the ``finalize`` / ``settle`` / ``record_move`` / ``follower_opening_action``
bookkeeping, the CRN paired-game constructor ``build_paired_games``, and the
single copy of the chunk-at-``root_limit`` multi-root ``session.search`` +
positional zip-scatter driver ``run_hexfield_ply``. Net A is ALWAYS the hexfield
candidate; only net B differs across opponents, so an ``OpponentAdapter`` supplies
net-B move generation while the driver owns everything net-agnostic.

Extracted verbatim from ``eval_arena.py`` (the four ``_Game`` slot sets,
``_finalize`` / ``_settle`` / ``_apply_search`` / ``_replay_action`` /
``_record_move`` / ``_follower_opening_action`` / the paired-game build blocks /
``_run_batch`` / ``_run_hex``), unified to the net-A-centric ``"A"`` / ``"B"``
labels (was ``"hex"`` / ``"strix"`` / ``"sealbot"`` per arena).

Importing this module is torch-free and .so-free: it references ``hexo_engine``
only through the module-level ``api`` name (monkeypatchable in CPU tests) and does
not import the native MCTS extension. The eval invariant is unchanged — nothing
here writes into the run dir.
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from .geometry import pack_action_id, unpack_action_id

# Default search/opening knobs (mirror ``eval_arena``): greedy after a
# temperature-sampled opening. Duplicated here so ``play_eval_match`` has stable
# keyword defaults without importing ``eval_arena`` at module load (that import is
# lazy inside the driver to avoid a circular dependency — ``eval_arena`` imports
# this module). The public ``play_*`` wrappers always pass these explicitly.
DEFAULT_OPENING_PLIES = 8
DEFAULT_OPENING_TEMPERATURE = 1.0


# --------------------------------------------------------------------------- #
# In-flight game record (superset of the four arenas' ``_Game`` __slots__).
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class EvalGame:
    """One in-flight eval game, net-A-centric.

    Superset of the four ``eval_arena`` ``_Game`` slot sets. Net A is always the
    hexfield candidate; net B is the opponent (a second hexfield session, Strix,
    or SealBot). ``winner`` is ``"A"`` / ``"B"`` / ``None`` (the per-arena
    ``"hex"`` / ``"strix"`` / ``"sealbot"`` labels unify to ``"A"`` / ``"B"``).

    Fields:
      * ``index`` — per-game key in net A's (and, for the checkpoint arena, net
        B's) session tree store.
      * ``pair_index`` — CRN pair this game belongs to (``-1`` when unpaired).
      * ``a_is_p0`` — engine seat (player0/player1) net A occupies in this game.
      * ``seed`` — the CRN seed (shared within a pair).
      * ``state`` — the engine game state.
      * ``plies`` — total plies applied (both players).
      * ``a_decisions`` — plies where net A moved (used by arenas that gate the
        opening temperature on net-A move count rather than total plies).
      * ``done`` / ``status`` / ``winner`` — terminal bookkeeping.
      * ``opening`` — the first ``opening_plies`` action ids (the shared line a
        follower replays).
      * ``actions`` — the FULL ordered action-id stream (both players) for the
        replayable ``.hxr`` record.
      * ``is_leader`` / ``leader`` — forced-opening CRN: a pair's LEADER searches
        its opening; the seat-swapped FOLLOWER replays ``leader.opening``. A
        leader's ``leader`` points at itself.
      * ``b_player`` — the per-game net-B mover object (a Strix player / SealBot
        adapter), or ``None`` for the hexfield-vs-hexfield arena.
      * ``opp_index`` / ``cand_key`` — multi-opponent namespacing: the opponent
        group index and the global candidate-session game key.
    """

    index: int
    pair_index: int
    a_is_p0: bool
    seed: int
    state: Any = None
    plies: int = 0
    a_decisions: int = 0
    done: bool = False
    status: str = "truncated"
    winner: str | None = None  # "A" | "B" | None (net-A-centric)
    opening: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    is_leader: bool = True
    leader: "EvalGame | None" = None
    b_player: Any = None
    opp_index: int = -1
    cand_key: int = -1

    def __post_init__(self) -> None:
        # A leader points at itself; build_paired_games repoints followers.
        if self.leader is None:
            self.leader = self

    @property
    def a_role(self) -> Any:
        return api.Player.PLAYER_0 if self.a_is_p0 else api.Player.PLAYER_1

    @property
    def a_role_label(self) -> str:
        return "player0" if self.a_is_p0 else "player1"

    def a_to_move(self) -> bool:
        return api.current_player(self.state) == self.a_role


# --------------------------------------------------------------------------- #
# Terminal / settle / move bookkeeping (net-A-centric, extracted verbatim).
# --------------------------------------------------------------------------- #
def finalize(game: EvalGame, *, budget_hit: bool, on_discard: Callable[[EvalGame], None]) -> None:
    """Close out ``game``: set ``status`` / ``winner`` / ``done`` and discard its
    session tree(s) via ``on_discard``.

    A terminal position resolves to a net-A-centric winner (``"A"`` when net A's
    engine seat won, else ``"B"``; ``None`` for the defensive draw case hexo never
    produces). Non-terminal games are ``aborted_budget`` when a wall-clock budget
    was hit, else ``truncated`` (max-plies). ``on_discard(game)`` performs the
    per-arena session discard(s) (checkpoint discards from BOTH nets' sessions;
    the others discard from one session)."""
    terminal = api.terminal(game.state)
    if terminal is not None:
        game.status = "completed"
        if terminal.winner is None:
            game.winner = None  # hexo has no draws; defensive
        else:
            won_label = str(terminal.winner)  # "player0" / "player1"
            game.winner = "A" if won_label == game.a_role_label else "B"
    elif budget_hit:
        game.status = "aborted_budget"
        game.winner = None
    else:
        game.status = "truncated"
        game.winner = None
    game.done = True
    on_discard(game)


def settle(
    game: EvalGame,
    max_plies: int,
    *,
    budget_hit: bool,
    on_discard: Callable[[EvalGame], None],
) -> bool:
    """Finalize ``game`` if it reached a terminal position or ``max_plies``.

    Returns ``True`` when the game was finalized this call, ``False`` otherwise."""
    if api.terminal(game.state) is not None or game.plies >= max_plies:
        finalize(game, budget_hit=budget_hit, on_discard=on_discard)
        return True
    return False


def record_move(
    game: EvalGame,
    action_id: int,
    *,
    is_a: bool,
    opening_plies: int,
    settle_fn: Callable[[EvalGame], Any],
) -> None:
    """Apply ``action_id`` to ``game`` and do the shared bookkeeping.

    One path for hexfield searches, net-B decisions (packed to an id), and
    follower replays (checkpoint's ``_apply_search`` / ``_replay_action`` and
    strix's ``_record_move`` unified): advance the engine, bump ``plies`` (and
    ``a_decisions`` when net A moved), append to the full move stream, extend the
    ``opening`` line while under ``opening_plies``, then ``settle_fn(game)``.
    ``settle_fn`` is the arena's ``settle`` bound with its ``max_plies`` /
    ``budget_hit`` / ``on_discard``."""
    q, r = unpack_action_id(int(action_id))
    api.apply_action(game.state, PlacementAction(AxialCoord(q=q, r=r)))
    game.plies += 1
    if is_a:
        game.a_decisions += 1
    game.actions.append(int(action_id))
    if len(game.opening) < opening_plies:
        game.opening.append(int(action_id))
    settle_fn(game)


def follower_opening_action(game: EvalGame) -> int | None:
    """The leader's recorded action for the follower ``game``'s current opening
    ply, or ``None`` if the leader has no action there.

    The round order keeps the leader strictly ahead of the follower, so a
    recorded action normally exists; if the leader ended mid-opening it may have
    fewer than ``opening_plies`` actions, in which case the caller falls back to a
    single-root search/decide for the remaining opening plies."""
    line = game.leader.opening
    return line[game.plies] if game.plies < len(line) else None


# --------------------------------------------------------------------------- #
# CRN paired-game construction (extracted verbatim from the four build blocks).
# --------------------------------------------------------------------------- #
def build_paired_games(
    n_games: int,
    game_seed_base: int,
    paired_openings: bool,
    *,
    new_state: Callable[[int], Any],
    b_player_factory: Callable[[int], Any] | None = None,
    side_seed_offset: dict[str, int] | None = None,
) -> tuple[list[EvalGame], dict[int, list[EvalGame]]]:
    """Build the in-flight game set (seats + CRN seeds) and its pair grouping.

    Paired (``paired_openings=True``): games are grouped into
    ``n_pairs = ceil(n_games / 2)`` matched pairs. Both games of a pair share the
    CRN ``pair_seed = game_seed_base + pair_index`` but swap seats: game 0 is the
    LEADER (net A as player0), game 1 the seat-swapped FOLLOWER (net A as player1)
    that replays the leader's opening line. An odd ``n_games`` leaves the last pair
    a singleton (leader only).

    Unpaired: every game gets an independent seed and seats alternate by game
    index. When ``side_seed_offset`` is given (the checkpoint arena's
    ``_SIDE_SEED_OFFSET``) the two seats draw from decorrelated seed streams;
    otherwise (strix / sealbot) the seed is ``game_seed_base + game_index``.

    ``new_state(seed) -> state`` creates the engine state for a game (some arenas
    seed ``new_game``, some do not). ``b_player_factory(seed) -> b_player`` builds
    the per-game net-B mover (Strix player / SealBot adapter) when net B is not a
    hexfield session; ``None`` leaves ``b_player`` unset.

    Returns ``(games, pair_members)``; ``pair_members`` maps ``pair_index`` to the
    list of that pair's games (empty for unpaired runs)."""

    def _make(index: int, pair_index: int, a_is_p0: bool, seed: int) -> EvalGame:
        return EvalGame(
            index=index,
            pair_index=pair_index,
            a_is_p0=a_is_p0,
            seed=seed,
            state=new_state(seed),
            b_player=(b_player_factory(seed) if b_player_factory is not None else None),
        )

    games: list[EvalGame] = []
    pair_members: dict[int, list[EvalGame]] = {}
    if paired_openings:
        n_pairs = (n_games + 1) // 2
        for pair_index in range(n_pairs):
            pair_seed = game_seed_base + pair_index  # shared CRN seed (both seats)
            idx0 = pair_index * 2
            g0 = _make(idx0, pair_index, a_is_p0=True, seed=pair_seed)  # leader
            games.append(g0)
            pair_members.setdefault(pair_index, []).append(g0)
            if idx0 + 1 < n_games:  # odd n_games -> last pair is a singleton
                g1 = _make(idx0 + 1, pair_index, a_is_p0=False, seed=pair_seed)
                g1.is_leader = False
                g1.leader = g0  # g1 replays g0's opening line
                games.append(g1)
                pair_members[pair_index].append(g1)
    else:
        for game_index in range(n_games):
            if side_seed_offset is not None:
                seed = game_seed_base + game_index + (
                    side_seed_offset["b"] if game_index % 2 else side_seed_offset["a"]
                )
            else:
                seed = game_seed_base + game_index
            games.append(_make(game_index, -1, a_is_p0=(game_index % 2 == 0), seed=seed))
    return games, pair_members


# --------------------------------------------------------------------------- #
# The single multi-root hexfield search + positional zip-scatter driver.
# --------------------------------------------------------------------------- #
def run_hexfield_ply(
    session: Any,
    evaluator: Any,
    batch: list[EvalGame],
    *,
    search_kwargs: dict[str, Any],
    root_limit: int,
    overrides: Any,
    seed: int,
    move_temperature_fn: Callable[[EvalGame], float],
    record_fn: Callable[..., None],
    is_a: bool,
    key_fn: Callable[[EvalGame], int] | None = None,
) -> int:
    """One multi-root hexfield ``search`` over ``batch``, chunked at ``root_limit``.

    Every game in ``batch`` has the SAME net to move (net A when ``is_a``, else net
    B via a second hexfield session). Each chunk of up to ``root_limit`` games is
    one multi-root ``session.search`` forward; results come back positionally and
    are scattered with ``zip`` — NEVER re-sorted. ``move_temperature_fn(g)`` gives
    each root its move temperature (opening-temperature while in the opening,
    greedy 0 after). ``record_fn(g, action_id, is_a=is_a)`` applies the move (the
    arena binds ``opening_plies`` / ``settle_fn``). ``key_fn(g)`` supplies the
    session game key (defaults to ``g.index``; the multi-opponent candidate session
    passes ``g.cand_key``). Returns the number of plies applied."""
    if key_fn is None:
        key_fn = lambda g: g.index  # noqa: E731 - trivial default key
    applied = 0
    for start in range(0, len(batch), root_limit):
        chunk = batch[start : start + root_limit]
        move_temperatures = [move_temperature_fn(g) for g in chunk]
        searches = session.search(
            [key_fn(g) for g in chunk],
            tuple(g.state for g in chunk),
            seed=seed,
            evaluator=evaluator,
            move_temperatures=move_temperatures,
            divergence_overrides=overrides,
            **search_kwargs,
        )
        if len(searches) != len(chunk):
            raise RuntimeError(
                f"hexfield eval search returned {len(searches)} results for {len(chunk)} games"
            )
        # Results come back positionally; scatter with zip, never re-sort.
        for g, search in zip(chunk, searches):
            record_fn(g, int(search["action_id"]), is_a=is_a)
            applied += 1
    return applied


# --------------------------------------------------------------------------- #
# Equal-TIME eval budget (ray plan §3 L2/L3, spec D-S35): visit calibration.
# --------------------------------------------------------------------------- #
# Probe-tree session keys, far above any game key (games are keyed by g.index /
# g.cand_key, small ints); every probe tree is discarded after its search.
_TIME_PROBE_KEY_BASE = 1_900_000_000


def visits_for_time_budget(
    base_visits: int,
    measured_ms: float,
    budget_ms: float,
    *,
    min_visits: int = 16,
    max_visits: int | None = None,
) -> int:
    """Visit count that spends ~``budget_ms`` of wall clock per move, given a
    measured ``measured_ms`` for a ``base_visits`` search (linear visits<->time
    model). Clamped to ``[min_visits, max_visits]`` (default max 8x base) so a
    mismeasured probe can neither starve the search nor run away."""
    if max_visits is None:
        max_visits = 8 * int(base_visits)
    if measured_ms <= 0.0 or budget_ms <= 0.0:
        return int(base_visits)
    visits = int(round(base_visits * (budget_ms / measured_ms)))
    return max(int(min_visits), min(visits, int(max_visits)))


def _probe_states(seed: int, plies_list: tuple[int, ...]) -> list:
    """Deterministic probe positions at a few depths (empty board + seeded
    random midgame lines), so the measured ms/move averages over the support
    sizes an eval game actually visits."""
    states = []
    for k, plies in enumerate(plies_list):
        state = api.new_game()
        rng = random.Random(seed * 7919 + k)
        for _ in range(plies):
            ids = api.legal_action_ids(state)
            if not ids:
                break
            q, r = unpack_action_id(int(rng.choice(list(ids))))
            result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
            if getattr(result, "terminal", None):
                break
        if api.terminal(state) is None:
            states.append(state)
    return states


def calibrate_time_budget_visits(
    session: Any,
    evaluator: Any,
    *,
    search_kwargs: dict[str, Any],
    overrides: Any,
    time_ms_per_move: float,
    seed: int = 0,
    probe_plies: tuple[int, ...] = (0, 8, 16),
    min_visits: int = 16,
    max_visits: int | None = None,
    _clock: Callable[[], float] = time.perf_counter,
) -> tuple[int, dict[str, Any]]:
    """Measure this net's single-root search wall time at the configured visit
    budget and return ``(calibrated_visits, info)`` for a ~``time_ms_per_move``
    per-move budget (spec D-S35).

    The Rust ``session.search`` cannot honor wall-clock budgets without deep
    surgery, so this is the sanctioned cheap approximation: one warmup search
    (compile/caches settle) then one timed single-root search per probe
    position; the MEDIAN ms feeds ``visits_for_time_budget``. Probe trees use
    dedicated keys and are discarded, so game trees / CRN seed streams are
    untouched. Under multi-root batching the realized per-move wall time sits
    below the budget for both arms alike — the A/B fairness rides on the RATIO
    of the two arms' measured nps, not the absolute per-move time."""
    base_visits = int(search_kwargs["visits"])
    states = _probe_states(seed, probe_plies)
    if not states:  # pragma: no cover - probe lines can't all be terminal
        return base_visits, {"error": "no probe states", "base_visits": base_visits}
    times_ms: list[float] = []
    for i, state in enumerate([states[0], *states]):  # states[0] twice: warmup
        key = _TIME_PROBE_KEY_BASE + i
        t0 = _clock()
        session.search(
            [key],
            (state,),
            seed=seed + i,
            evaluator=evaluator,
            move_temperatures=[0.0],
            divergence_overrides=overrides,
            **search_kwargs,
        )
        dt_ms = (_clock() - t0) * 1000.0
        session.discard(key)
        if i > 0:  # drop the warmup measurement
            times_ms.append(dt_ms)
    measured_ms = float(statistics.median(times_ms))
    visits = visits_for_time_budget(
        base_visits, measured_ms, float(time_ms_per_move),
        min_visits=min_visits, max_visits=max_visits,
    )
    return visits, {
        "base_visits": base_visits,
        "measured_ms_per_move": round(measured_ms, 3),
        "probe_ms": [round(t, 3) for t in times_ms],
        "time_budget_ms": float(time_ms_per_move),
        "calibrated_visits": visits,
    }


def _resolve_time_budget(cfg: Any, explicit: float | None) -> float:
    """The per-move time budget: an explicit kwarg wins; else the config knob
    ``multi_stage_eval.eval_time_budget_ms`` (0.0 = off)."""
    if explicit is not None:
        return float(explicit)
    section = getattr(cfg, "multi_stage_eval", None)
    return float(getattr(section, "eval_time_budget_ms", 0.0) or 0.0)


# --------------------------------------------------------------------------- #
# Hexfield-search telemetry + a counting session proxy.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _HexTelemetry:
    """Running count of hexfield ``session.search`` calls + their wall time.

    The driver keeps one per hexfield mover so ``forward_batches`` /
    ``mcts_search_elapsed_seconds`` in the result meta match the pre-refactor
    per-call bookkeeping. Net A (the driver) and, for the hexfield-vs-hexfield
    arena, net B (``HexfieldCheckpointAdapter``) each own one; the checkpoint
    ``meta_extra_fn`` sums them into the single combined ``forward_batches`` the
    old runner reported (every hexfield forward, both nets)."""

    forward_batches: int = 0
    mcts_search_elapsed: float = 0.0


class _CountingSession:
    """Wraps a session so each ``search`` call bumps a ``_HexTelemetry``.

    Only ``.search`` is proxied (the sole method ``run_hexfield_ply`` calls);
    ``.discard`` stays on the real session, driven by ``finalize``'s
    ``on_discard``. Times and counts exactly as the pre-refactor ``_run_batch`` /
    ``_run_opening_batch`` / ``_run_single`` did (one forward per chunk)."""

    __slots__ = ("_session", "_telemetry")

    def __init__(self, session: Any, telemetry: _HexTelemetry) -> None:
        self._session = session
        self._telemetry = telemetry

    def search(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        out = self._session.search(*args, **kwargs)
        self._telemetry.mcts_search_elapsed += time.perf_counter() - t0
        self._telemetry.forward_batches += 1
        return out


@dataclass(slots=True)
class MatchTelemetry:
    """Driver-computed runtime telemetry handed to ``meta_extra_fn``.

    ``forward_batches`` / ``mcts_search_elapsed`` are NET A's hexfield search
    counts (the driver's); an arena whose net B is also a hexfield session folds
    the adapter's own counts in via ``meta_extra_fn``. ``eval_visits`` /
    ``virtual_batch_size`` / ``active_root_limit`` are the resolved search budget so
    the caller need not re-resolve it for the meta echo."""

    rounds: int
    forward_batches: int
    mcts_search_elapsed: float
    budget_hit: bool
    elapsed_seconds: float
    eval_visits: int
    virtual_batch_size: int
    active_root_limit: int
    # Equal-time leg only (spec D-S35): net A's calibration record
    # (calibrate_time_budget_visits info dict), None on fixed-visit runs. When
    # present, ``eval_visits`` above is the CALIBRATED count.
    time_calibration: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# One hexfield net's per-round pass (openers batch / followers replay / greedy
# batch). Shared verbatim by the driver (net A) and HexfieldCheckpointAdapter
# (net B) — the two nets ran structurally identical passes in the old runner.
# --------------------------------------------------------------------------- #
def run_hexfield_net_pass(
    to_move: list[EvalGame],
    *,
    is_a: bool,
    session: Any,
    evaluator: Any,
    overrides: Any,
    search_kwargs: dict[str, Any],
    root_limit: int,
    paired_openings: bool,
    batch_openings: bool,
    opening_plies: int,
    opening_temperature: float,
    round_index: int,
    seed_base: int,
    open_offset: int,
    greedy_offset: int,
    record_move: Callable[..., None],
    telemetry: _HexTelemetry,
    key_fn: Callable[[EvalGame], int] | None = None,
    opening_gate_attr: str = "plies",
) -> int:
    """Advance every game in ``to_move`` (all this net's turn) by one hexfield ply.

    Reproduces the pre-refactor per-net round body: in paired mode (and unless
    ``batch_openings``) a pair's LEADER searches its opening batched cross-game
    (all roots pinned to ``opening_temperature``, base seed ``seed_base +
    open_offset + round_index*1_000_003`` with the native per-root ``+index``
    decorrelation), a seat-swapped FOLLOWER replays the leader's recorded opening
    line (single-root fallback ``g.seed*5003 + g.plies`` when the leader ended
    mid-opening), and games past the opening are the greedy batch (temperature 0,
    base seed ``seed_base + greedy_offset + round_index*1_000_003``). Every
    ``session.search`` goes through a ``_CountingSession`` so ``telemetry`` counts
    forwards exactly as the old runner did. Returns the number of plies applied.

    ``opening_gate_attr`` names the ``EvalGame`` field that gates the opening
    window: ``"plies"`` (default — total plies; the paired checkpoint/strix
    arenas) or ``"a_decisions"`` (net-A move count; the unpaired SealBot arena
    counts hexfield decisions, not total plies)."""

    def _in_opening(g: EvalGame) -> bool:
        return getattr(g, opening_gate_attr) < opening_plies

    if paired_openings and not batch_openings:
        openers = [g for g in to_move if _in_opening(g) and g.is_leader]
        followers = [g for g in to_move if _in_opening(g) and not g.is_leader]
        greedy = [g for g in to_move if not _in_opening(g)]
    else:
        openers, followers, greedy = [], [], to_move

    def _temp(g: EvalGame) -> float:
        return opening_temperature if (_in_opening(g) and opening_temperature > 0.0) else 0.0

    counting = _CountingSession(session, telemetry)
    applied = 0
    if openers:
        open_seed = seed_base + open_offset + round_index * 1_000_003
        applied += run_hexfield_ply(
            counting,
            evaluator,
            openers,
            search_kwargs=search_kwargs,
            root_limit=root_limit,
            overrides=overrides,
            seed=open_seed,
            move_temperature_fn=lambda g: opening_temperature,
            record_fn=record_move,
            is_a=is_a,
            key_fn=key_fn,
        )
    for g in followers:
        # Replay the leader's recorded opening action; if the leader ended its
        # game before this ply (no recorded action) fall back to a single-root
        # CRN search so the follower still moves.
        replay = follower_opening_action(g)
        if replay is not None:
            record_move(g, replay, is_a=is_a)
            applied += 1
        else:
            applied += run_hexfield_ply(
                counting,
                evaluator,
                [g],
                search_kwargs=search_kwargs,
                root_limit=root_limit,
                overrides=overrides,
                seed=g.seed * 5003 + g.plies,
                move_temperature_fn=_temp,
                record_fn=record_move,
                is_a=is_a,
                key_fn=key_fn,
            )
    if greedy:
        greedy_seed = seed_base + greedy_offset + round_index * 1_000_003
        applied += run_hexfield_ply(
            counting,
            evaluator,
            greedy,
            search_kwargs=search_kwargs,
            root_limit=root_limit,
            overrides=overrides,
            seed=greedy_seed,
            move_temperature_fn=_temp,
            record_fn=record_move,
            is_a=is_a,
            key_fn=key_fn,
        )
    return applied


# --------------------------------------------------------------------------- #
# Opponent adapter protocol (net-B move generation; the driver owns the rest).
# --------------------------------------------------------------------------- #
@runtime_checkable
class OpponentAdapter(Protocol):
    """Pluggable net-B mover for the central eval driver.

    Net A is always the hexfield candidate (driven by ``run_hexfield_ply``); an
    adapter supplies only net B's move generation. ``start`` builds per-game net-B
    state; ``advance`` moves every game in ``batch`` where net B is to move
    (leaders/greedy — followers are handled by the driver's opening replay) and
    returns the number of plies applied; ``close`` tears down any pools/servers and
    (for Strix) injects telemetry."""

    label: str

    def start(self, games: list[EvalGame]) -> None:
        ...

    def advance(
        self,
        batch: list[EvalGame],
        *,
        round_index: int,
        seed_base: int,
        record_move: Callable[..., None],
        opening_plies: int,
        opening_temperature: float,
    ) -> int:
        ...

    def close(self) -> None:
        ...


# --------------------------------------------------------------------------- #
# The central pentanomial match driver.
# --------------------------------------------------------------------------- #
def play_eval_match(
    candidate_ckpt: str | Any,
    opponent: Any,
    n_games: int,
    *,
    config: Any,
    label_a: str,
    label_b: str,
    meta_extra_fn: Callable[[list[EvalGame], "MatchTelemetry"], dict[str, Any]],
    paired_openings: bool = True,
    visits: int | None = None,
    virtual_batch_size: int | None = None,
    active_root_limit: int | None = None,
    opening_plies: int = DEFAULT_OPENING_PLIES,
    opening_temperature: float = DEFAULT_OPENING_TEMPERATURE,
    batch_openings: bool = False,
    divergence_overrides: dict | None = None,
    diagnostics_dir: str | Any | None = None,
    game_seed_base: int = 0,
    max_wall_seconds: float = 0.0,
    max_states: int = 65_536,
    side_seed_offset: dict[str, int] | None = None,
    net_a_open_offset: int = 13_000_003,
    net_a_greedy_offset: int = 0,
    opening_gate_attr: str = "plies",
    build_candidate_evaluator: Callable[..., Any] | None = None,
    make_session: Callable[..., Any] | None = None,
    time_ms_per_move: float | None = None,
) -> dict[str, Any]:
    """Play the hexfield candidate (net A) vs one ``opponent`` adapter (net B).

    The single, opponent-agnostic pentanomial runner. Net A is ALWAYS the hexfield
    candidate, driven per round through ``run_hexfield_net_pass`` (openers batch /
    followers replay / greedy batch) on its own persistent session/evaluator; the
    ``opponent`` adapter supplies only net-B move generation via ``.advance``. The
    driver owns everything net-agnostic: CRN pairing (``build_paired_games``), the
    round loop and its no-progress / wall-clock guards, ``finalize`` /
    ``settle`` bookkeeping and the per-game tree discards (net A's session plus,
    via the adapter's optional ``on_game_discard``, net B's), the per-game and
    per-pair result rows, and the pentanomial block. Arena-specific meta is
    assembled by ``meta_extra_fn(games, telemetry)`` (it also writes the ``.hxr``
    record and folds in any adapter-side telemetry), then merged with
    ``label_a`` / ``label_b`` by ``eval_arena._build_match_result``.

    ``net_a_open_offset`` / ``net_a_greedy_offset`` seed net A's opening / greedy
    batches (defaults 13M / 0 — the checkpoint & strix arenas; SealBot's net-A
    pass used the 7M greedy offset). ``opening_gate_attr`` names the ``EvalGame``
    field the opening window gates on (``"plies"`` by default; SealBot gates on
    ``"a_decisions"``, the hexfield move count).

    ``time_ms_per_move`` (spec D-S35): a per-move wall-clock budget for the
    equal-time A/B leg. ``None`` reads ``config.multi_stage_eval.
    eval_time_budget_ms`` (default 0.0 = off, byte-identical fixed-visit
    behavior). When > 0, net A's visit budget is re-calibrated from a few
    measured probe searches (``calibrate_time_budget_visits``) so one move
    costs ~that many ms; a hexfield net-B adapter does the same with ITS
    evaluator in ``.start`` (each arm pays its own architecture's latency —
    the point of the leg). Non-hexfield adapters ignore it.

    Injection seams mirror the public runners: ``build_candidate_evaluator`` /
    ``make_session`` skip checkpoint loading / GPU for CPU tests. Budget resolution,
    overrides, and the result builder are reused from ``eval_arena`` (lazy import,
    breaking the eval_arena→eval_driver import cycle). The eval invariant is
    unchanged — nothing here writes into the run dir."""

    # Lazy: eval_arena imports this module, so import it inside the call. Keeps the
    # eval-specific budget/override/loader/result helpers as the single source.
    from . import eval_arena
    from .config import build_eval_search_kwargs, parse_hexfield_config

    cfg = config if config is not None else parse_hexfield_config({})
    sp = cfg.selfplay
    # visits=None defaults to the self-play search budget (sp.search_visits), the
    # vbs / active-root-limit overrides fall back to sp.* too.
    eval_visits, vbs, root_limit = eval_arena._resolve_eval_budget(
        sp, visits=visits, virtual_batch_size=virtual_batch_size, active_root_limit=active_root_limit
    )
    search_kwargs = build_eval_search_kwargs(
        sp, visits=eval_visits, virtual_batch_size=vbs, active_root_limit=root_limit
    )
    # Net A's divergence overrides (an explicit dict is returned as-is).
    ov_a = eval_arena._resolve_eval_overrides(
        sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides
    )

    started = time.perf_counter()

    # ----- Net A: one persistent hexfield session + evaluator. -----
    if build_candidate_evaluator is not None:
        eval_a = build_candidate_evaluator()
    else:
        from .inference import build_serve_evaluator  # lazy: torch only on the GPU path

        eval_a = build_serve_evaluator(
            eval_arena._load_hexfield_net(candidate_ckpt), cfg, role="eval"
        )
    session_a = make_session() if make_session is not None else eval_arena._new_rust_session(max_states)

    # ----- Equal-time leg (spec D-S35): re-calibrate net A's visit budget so a
    # move costs ~time_budget_ms of wall clock. 0.0 (the default) skips this
    # block entirely — the fixed-visit path is untouched.
    time_budget_ms = _resolve_time_budget(cfg, time_ms_per_move)
    calibration_a: dict[str, Any] | None = None
    if time_budget_ms > 0.0:
        eval_visits, calibration_a = calibrate_time_budget_visits(
            session_a,
            eval_a,
            search_kwargs=search_kwargs,
            overrides=ov_a,
            time_ms_per_move=time_budget_ms,
            seed=game_seed_base,
        )
        search_kwargs = dict(search_kwargs, visits=eval_visits)

    # ----- Build the in-flight game set (seats + CRN seeds). Net-B per-game state
    # (a second-session key is keyed by g.index; per-game players when net B is not
    # a hexfield session) comes from the adapter's optional new_state / factory. ---
    new_state = getattr(opponent, "new_state", None)
    if new_state is None:
        new_state = lambda seed: api.new_game()  # noqa: E731 - default engine state
    b_player_factory = getattr(opponent, "b_player_factory", None)
    games, pair_members = build_paired_games(
        n_games,
        game_seed_base,
        paired_openings,
        new_state=new_state,
        b_player_factory=b_player_factory,
        side_seed_offset=side_seed_offset,
    )

    # ----- Net B: hand the built games to the adapter (builds its session/pool). --
    opponent.start(games)
    opp_discard = getattr(opponent, "on_game_discard", None)

    telemetry_a = _HexTelemetry()
    budget_hit = False
    rounds = 0

    def on_discard(g: EvalGame) -> None:
        session_a.discard(g.index)
        if opp_discard is not None:
            opp_discard(g)

    def settle_fn(g: EvalGame) -> None:
        # settle -> finalize only runs during normal play (budget not yet hit);
        # the wall-clock branch below finalizes directly with budget_hit=True.
        settle(g, sp.max_game_plies, budget_hit=False, on_discard=on_discard)

    def record_fn(g: EvalGame, action_id: int, *, is_a: bool) -> None:
        record_move(g, action_id, is_a=is_a, opening_plies=opening_plies, settle_fn=settle_fn)

    # ----- Round loop: each round advances every active game by at least one ply.
    # Net A pass first (openers batched / followers replay / greedy batched), then
    # the adapter's net-B pass (re-gathered, so a game net A just moved can also
    # move this round). A round that makes no progress raises rather than hangs.
    # The loop is wrapped so the adapter is torn down even on a mid-round exception
    # (the strix adapter's close() shuts the daemon batch server + blocked pool
    # workers, which would otherwise hang); ``opponent.close()`` runs exactly once.
    try:
        while True:
            active = [g for g in games if not g.done]
            if not active:
                break
            if max_wall_seconds and (time.perf_counter() - started) > max_wall_seconds:
                budget_hit = True
                for g in active:
                    finalize(g, budget_hit=True, on_discard=on_discard)
                break
            rounds += 1
            plies_this_round = 0
            # (A) net A (hexfield candidate) pass.
            to_move_a = [g for g in active if not g.done and g.a_to_move()]
            plies_this_round += run_hexfield_net_pass(
                to_move_a,
                is_a=True,
                session=session_a,
                evaluator=eval_a,
                overrides=ov_a,
                search_kwargs=search_kwargs,
                root_limit=root_limit,
                paired_openings=paired_openings,
                batch_openings=batch_openings,
                opening_plies=opening_plies,
                opening_temperature=opening_temperature,
                round_index=rounds,
                seed_base=game_seed_base,
                open_offset=net_a_open_offset,
                greedy_offset=net_a_greedy_offset,
                record_move=record_fn,
                telemetry=telemetry_a,
                opening_gate_attr=opening_gate_attr,
            )
            # (B) net B (opponent) pass: re-gather (a game hexfield just moved may
            # now be net-B-to-move -> up to two plies/game/round).
            b_batch = [g for g in games if not g.done and not g.a_to_move()]
            plies_this_round += opponent.advance(
                b_batch,
                round_index=rounds,
                seed_base=game_seed_base,
                record_move=record_fn,
                opening_plies=opening_plies,
                opening_temperature=opening_temperature,
            )
            if plies_this_round == 0:
                raise RuntimeError(
                    "hexfield eval made no progress in a round; aborting to avoid a hang"
                )
    finally:
        # Tear the adapter down (strix reads server telemetry only after close()).
        opponent.close()

    # ----- Per-game result rows + per-pair pentanomial rows (net-A-centric). -----
    game_rows = [
        {
            "index": g.index,
            "seed": g.seed,
            "a_seat": "P0" if g.a_is_p0 else "P1",
            "status": g.status,
            "winner": g.winner,
            "plies": g.plies,
            "opening": list(g.opening),
        }
        for g in games
    ]
    pairs: list[dict[str, Any]] = []
    if paired_openings:
        for pair_index in sorted(pair_members):
            members = pair_members[pair_index]
            decided = [g for g in members if g.status == "completed"]
            a_wins_in_pair = sum(1 for g in decided if g.winner == "A")
            pairs.append(
                {
                    "pair_index": pair_index,
                    "seed": game_seed_base + pair_index,
                    "game_indices": [g.index for g in members],
                    "n_games": len(members),
                    "n_decided": len(decided),
                    "a_wins": a_wins_in_pair,
                    "b_wins": len(decided) - a_wins_in_pair,
                    "pentanomial_a_score": a_wins_in_pair,
                }
            )

    telemetry = MatchTelemetry(
        rounds=rounds,
        forward_batches=telemetry_a.forward_batches,
        mcts_search_elapsed=telemetry_a.mcts_search_elapsed,
        budget_hit=budget_hit,
        elapsed_seconds=time.perf_counter() - started,
        eval_visits=eval_visits,
        virtual_batch_size=vbs,
        active_root_limit=root_limit,
        time_calibration=calibration_a,
    )
    meta_extra = meta_extra_fn(games, telemetry)
    return eval_arena._build_match_result(
        games=game_rows,
        pairs=pairs if paired_openings else None,
        label_a=label_a,
        label_b=label_b,
        meta_extra=meta_extra,
    )


# --------------------------------------------------------------------------- #
# Adapter: net B is a second hexfield session (hexfield-vs-hexfield arena).
# --------------------------------------------------------------------------- #
class HexfieldCheckpointAdapter:
    """Net B is a second hexfield checkpoint: ``.advance`` is a net-B hexfield pass.

    The canonical pentanomial opponent — net B runs the SAME per-round pass net A
    does (``run_hexfield_net_pass`` with ``is_a=False``), on its own persistent
    session/evaluator and its own opening/greedy seed streams (offsets 19M/7M,
    distinct from net A's 13M/0 so an opening and a greedy batch in one round never
    collide). Its hexfield searches accumulate into ``self.telemetry`` so the
    checkpoint ``meta_extra_fn`` can fold them into the single combined
    ``forward_batches``. ``build_paired_games`` keys net B's trees by ``g.index``
    (same as net A), discarded via ``on_game_discard`` when the driver finalizes."""

    def __init__(
        self,
        model_b_ckpt: str | Any,
        *,
        config: Any,
        label: str,
        overrides_b: dict,
        make_session: Callable[..., Any] | None,
        max_states: int,
        visits: int | None,
        virtual_batch_size: int | None,
        active_root_limit: int | None,
        paired_openings: bool,
        batch_openings: bool,
        build_evaluator: Callable[..., Any] | None = None,
        time_ms_per_move: float | None = None,
    ) -> None:
        self.label = label
        self._model_b_ckpt = model_b_ckpt
        self._config = config
        self._overrides_b = overrides_b
        self._make_session = make_session
        self._max_states = max_states
        self._visits = visits
        self._vbs = virtual_batch_size
        self._arl = active_root_limit
        self._paired_openings = paired_openings
        self._batch_openings = batch_openings
        self._build_evaluator = build_evaluator
        self._time_ms_per_move = time_ms_per_move
        # Equal-time leg (spec D-S35): net B's calibration record, None on
        # fixed-visit runs. Filled by ``start`` after its session/evaluator
        # exist; the wrapper's meta_extra_fn may surface it.
        self.time_calibration: dict[str, Any] | None = None
        self.telemetry = _HexTelemetry()
        self._session: Any = None
        self._evaluator: Any = None
        self._search_kwargs: dict[str, Any] | None = None
        self._root_limit: int | None = None

    # Net B shares net A's engine state; the checkpoint arena seeds no opening into
    # ``new_game`` (opening diversity comes from the root temperature sampling).
    def new_state(self, seed: int) -> Any:
        return api.new_game()

    # No per-game net-B player object (net B is a session, keyed by g.index).
    b_player_factory = None

    def start(self, games: list[EvalGame]) -> None:
        from . import eval_arena
        from .config import build_eval_search_kwargs

        sp = self._config.selfplay
        eval_visits, vbs, root_limit = eval_arena._resolve_eval_budget(
            sp, visits=self._visits, virtual_batch_size=self._vbs, active_root_limit=self._arl
        )
        self._search_kwargs = build_eval_search_kwargs(
            sp, visits=eval_visits, virtual_batch_size=vbs, active_root_limit=root_limit
        )
        self._root_limit = root_limit
        if self._build_evaluator is not None:
            self._evaluator = self._build_evaluator()
        else:
            from .inference import build_serve_evaluator  # lazy: torch only on the GPU path

            self._evaluator = build_serve_evaluator(
                eval_arena._load_hexfield_net(self._model_b_ckpt), self._config, role="eval"
            )
        self._session = (
            self._make_session()
            if self._make_session is not None
            else eval_arena._new_rust_session(self._max_states)
        )
        # Equal-time leg (spec D-S35): net B calibrates against ITS OWN
        # evaluator/session, so each arm's visit budget reflects its own
        # architecture's latency (the entire point of the leg). Same probe
        # positions/seed as net A (seed 0 default) — the probes are
        # measurement-only and never touch game trees.
        time_budget_ms = _resolve_time_budget(self._config, self._time_ms_per_move)
        if time_budget_ms > 0.0:
            visits_b, self.time_calibration = calibrate_time_budget_visits(
                self._session,
                self._evaluator,
                search_kwargs=self._search_kwargs,
                overrides=self._overrides_b,
                time_ms_per_move=time_budget_ms,
            )
            self._search_kwargs = dict(self._search_kwargs, visits=visits_b)

    def advance(
        self,
        batch: list[EvalGame],
        *,
        round_index: int,
        seed_base: int,
        record_move: Callable[..., None],
        opening_plies: int,
        opening_temperature: float,
    ) -> int:
        return run_hexfield_net_pass(
            batch,
            is_a=False,
            session=self._session,
            evaluator=self._evaluator,
            overrides=self._overrides_b,
            search_kwargs=self._search_kwargs,
            root_limit=self._root_limit,
            paired_openings=self._paired_openings,
            batch_openings=self._batch_openings,
            opening_plies=opening_plies,
            opening_temperature=opening_temperature,
            round_index=round_index,
            seed_base=seed_base,
            open_offset=19_000_003,
            greedy_offset=7_000_003,
            record_move=record_move,
            telemetry=self.telemetry,
        )

    def on_game_discard(self, game: EvalGame) -> None:
        self._session.discard(game.index)

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Adapter: net B is hexo-strix (a shared GPU batch server + per-game MCTS pool).
# --------------------------------------------------------------------------- #
class StrixAdapter:
    """Net B is hexo-strix, GPU-batched lock-step against the hexfield candidate.

    Strix is the pinned Bradley-Terry anchor. Net B moves are generated by a pool
    of per-game :class:`hexo_strix.mcts_player.StrixMctsPlayer` decisions whose
    leaves coalesce in ONE shared :class:`hexo_strix.batch_server.StrixBatchServer`
    (the lock-step cross-game GPU batching win). ``.advance`` fires every net-B
    to-move game's (read-only) ``decide`` at the pool at once so their leaves block
    simultaneously in the server, then applies the returned actions single-threaded
    after the join (``decide`` never mutates the state, so there are no write
    races). Followers still in their opening replay the leader's recorded line
    (no strix search), exactly as the driver's net-A pass replays for net A.

    The shared server, the per-game player builder, and the thread pool are built
    in ``__init__`` because ``build_paired_games`` calls ``b_player_factory`` for
    every game BEFORE the driver calls ``.start`` — each player must already be
    bound to the live server at construction. ``.close`` shuts the server then the
    pool (idempotent, drains queued requests) and snapshots the server's coalescing
    telemetry (``n_forwards`` / ``n_graphs`` / ``max_seen_batch``) for the wrapper's
    ``meta_extra_fn`` to inject. Net B has no hexfield session, so no per-game tree
    discard hook is exposed (only net A's session is discarded by the driver).

    Injection seams mirror the public runner: ``build_strix_player(game_seed,
    is_leader) -> player`` and ``make_batch_server() -> server`` let a CPU test
    drive the loop with fakes and no GPU / ``hexo_rs`` wheel.
    """

    def __init__(
        self,
        strix_ckpt: str | Any,
        *,
        label: str,
        strix_label: str,
        n_games: int,
        paired_openings: bool,
        strix_noise_plies: int,
        strix_sims: int,
        strix_m_actions: int,
        strix_c_visit: int,
        strix_c_scale: float,
        strix_disable_gumbel_noise: bool,
        strix_device: str,
        strix_linger_s: float,
        strix_max_batch: int,
        strix_pool_threads: int | None,
        build_strix_player: Callable[[int, bool], Any] | None = None,
        make_batch_server: Callable[..., Any] | None = None,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.label = label
        self._paired_openings = paired_openings

        # --- strix (net B): one shared model owned by the cross-game batch server.
        # Lazy imports (mirroring the public runner): hexo_strix stays free of an
        # import-time dependency and never imports hexfield.
        self._strix_ck = None
        if make_batch_server is not None:
            self._server = make_batch_server()
        else:
            from hexo_strix.batch_server import StrixBatchServer
            from hexo_strix.loader import load_strix_checkpoint

            # Load on CPU; the server moves the model to the GPU and owns it there.
            self._strix_ck = load_strix_checkpoint(strix_ckpt, device="cpu")
            self._server = StrixBatchServer(
                self._strix_ck.model,
                device=strix_device,
                max_batch=strix_max_batch,
                linger_s=strix_linger_s,
            )

        if build_strix_player is not None:
            self._build_strix = build_strix_player
        else:
            from hexo_strix.mcts_player import StrixMctsPlayer

            def _build_strix(game_seed: int, is_leader: bool) -> Any:
                # server= routes leaves through the shared server (which owns the
                # model on the GPU); we must NOT pass a device that would .to() the
                # shared model per game (concurrent .to() across threads is a
                # hazard). noise_opening_plies gives strix opening-confined light
                # noise; the tail stays deterministic (disable_gumbel_noise), so
                # pairs share a clean greedy tail and only differ by the seat swap.
                return StrixMctsPlayer(
                    checkpoint=self._strix_ck,
                    identity_id="hexo-strix",
                    sims=strix_sims,
                    m_actions=strix_m_actions,
                    c_visit=strix_c_visit,
                    c_scale=strix_c_scale,
                    disable_gumbel_noise=strix_disable_gumbel_noise,
                    noise_opening_plies=strix_noise_plies,
                    seed=game_seed,
                    label=strix_label,
                    server=self._server,
                )

            self._build_strix = _build_strix

        # Size the pool so ALL strix-to-move games can block in the server at once
        # (else the batcher can only coalesce as many as there are workers).
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(strix_pool_threads) if strix_pool_threads else n_games),
            thread_name_prefix="strix-game",
        )

        # Runtime telemetry surfaced to the wrapper's meta_extra_fn after close().
        self.strix_elapsed = 0.0
        self.strix_forward_batches = 0
        self.strix_leaves = 0
        self.strix_max_seen_batch = 0

    # Both engines share ONE hexo_engine state per game; the public runner seeds it
    # with the CRN seed (api.new_game(seed=...)), so replicate that here.
    def new_state(self, seed: int) -> Any:
        return api.new_game(seed=seed)

    # Per-game strix player, built once and reused every round (holds no engine
    # state; decide() snapshots g.state read-only). is_leader is always True, as in
    # the pre-refactor runner (the CRN seed, not this flag, drives opening noise).
    def b_player_factory(self, seed: int) -> Any:
        return self._build_strix(seed, True)

    def start(self, games: list[EvalGame]) -> None:
        # Players are already built per game via ``b_player_factory`` during
        # ``build_paired_games``; the server/pool were built in ``__init__``.
        return None

    def advance(
        self,
        batch: list[EvalGame],
        *,
        round_index: int,
        seed_base: int,
        record_move: Callable[..., None],
        opening_plies: int,
        opening_temperature: float,
    ) -> int:
        # Paired-mode followers still in their opening replay the leader's recorded
        # line (no strix search); leaders and every game past the opening search.
        search: list[EvalGame] = []
        replay: list[EvalGame] = []
        for g in batch:
            if self._paired_openings and g.plies < opening_plies and not g.is_leader:
                replay.append(g)
            else:
                search.append(g)
        applied = self._run_strix(search, record_move)
        fallback: list[EvalGame] = []
        for g in replay:
            act = follower_opening_action(g)
            if act is not None:
                record_move(g, act, is_a=False)
                applied += 1
            else:  # leader ended mid-opening: decide this follower for real.
                fallback.append(g)
        applied += self._run_strix(fallback, record_move)
        return applied

    def _run_strix(self, batch: list[EvalGame], record_move: Callable[..., None]) -> int:
        """Fire every game in ``batch`` at the strix pool at once so their leaves
        coalesce in the shared batch server; apply single-threaded after the join
        (decide() is read-only on g.state, so no write races). Returns plies
        applied. decide() picks noisy-vs-greedy search by the game ply itself."""
        if not batch:
            return 0
        t0 = time.perf_counter()
        futs = {self._pool.submit(g.b_player.decide, g.state): g for g in batch}
        for fut, g in futs.items():
            decision = fut.result()
            coord = decision.action.coord
            record_move(g, pack_action_id(coord.q, coord.r), is_a=False)
        self.strix_elapsed += time.perf_counter() - t0
        return len(batch)

    def close(self) -> None:
        # Must run even on a mid-run exception, else the daemon batcher / blocked
        # pool workers hang. close() is idempotent + drains queued requests. The
        # server's coalescing telemetry is only final after close().
        try:
            self._server.close()
        finally:
            self._pool.shutdown()
        self.strix_forward_batches = int(getattr(self._server, "n_forwards", 0))
        self.strix_leaves = int(getattr(self._server, "n_graphs", 0))
        self.strix_max_seen_batch = int(getattr(self._server, "max_seen_batch", 0))


# --------------------------------------------------------------------------- #
# Adapter: net B is SealBot (external time-limited minimax) — UNPAIRED.
# --------------------------------------------------------------------------- #
class SealBotAdapter:
    """Net B is SealBot: one persistent minimax worker per game, drained serially.

    SealBot is a down-weighted secondary opponent (Strix is the anchor). Its C++
    minimax is time-limited, so its search depth varies with process load and a
    seat-swapped pair would NOT be a matched comparison — this adapter therefore
    only ever runs under the driver's UNPAIRED mode (``paired_openings=False``,
    ``pairs=None``, ``pentanomial=None``), exactly as the pre-refactor
    ``play_sealbot_match`` did.

    Each game owns its own ``SealBotPlayer`` (the two SealBot variants cannot
    coexist in one process; one worker per game is the established layout). The
    per-game player holds no engine state — ``decide(state)`` reads ``g.state``
    read-only — so it is built once during ``build_paired_games`` and reused every
    round. ``.advance`` drains SealBot's CONSECUTIVE moves per game per turn (the
    hexo mover schedule has same-player runs, so one SealBot turn can be several
    plies) via ``record_move`` (net B, ``is_a=False``); ``.close`` shuts every
    worker (best-effort, mirroring the pre-refactor teardown).

    ``make_opponent() -> player`` is the injection seam (the public runner passes
    a closure that honours the ``build_opponent`` test seam, else builds a real
    ``SealBotPlayer``). Net B has no hexfield session, so no per-game tree discard
    hook is exposed (only net A's session is discarded by the driver).
    """

    def __init__(self, *, label: str, make_opponent: Callable[[], Any]) -> None:
        self.label = label
        self._make_opponent = make_opponent
        self._games: list[EvalGame] = []
        # SealBot decide() wall time, surfaced to the wrapper's meta_extra_fn.
        self.opponent_elapsed = 0.0

    # SealBot games seed their engine state from the CRN seed (matches the
    # pre-refactor per-game ``api.new_game(seed=self.seed)``), overriding the
    # driver's unseeded default new_state.
    def new_state(self, seed: int) -> Any:
        return api.new_game(seed=seed)

    # One SealBotPlayer per game, built during ``build_paired_games`` (the seed is
    # unused — the player holds no engine state).
    def b_player_factory(self, seed: int) -> Any:
        return self._make_opponent()

    def start(self, games: list[EvalGame]) -> None:
        # Players are already built per game via ``b_player_factory``; keep the
        # game list so ``close`` can shut every worker.
        self._games = games

    def advance(
        self,
        batch: list[EvalGame],
        *,
        round_index: int,
        seed_base: int,
        record_move: Callable[..., None],
        opening_plies: int,
        opening_temperature: float,
    ) -> int:
        """Drain SealBot's turn for every game in ``batch``, serially per game.

        A SealBot turn is fully drained (``while not g.done and not g.a_to_move()``)
        so consecutive same-player plies in one turn are all applied this round,
        exactly as the pre-refactor serial SealBot loop did. ``decide`` is read-only
        on ``g.state``; ``record_move`` (net B) applies the packed action, bumps the
        ply / opening bookkeeping, and settles the game."""
        applied = 0
        for g in batch:
            while not g.done and not g.a_to_move():
                t0 = time.perf_counter()
                decision = g.b_player.decide(g.state)
                self.opponent_elapsed += time.perf_counter() - t0
                coord = decision.action.coord
                record_move(g, pack_action_id(coord.q, coord.r), is_a=False)
                applied += 1
        return applied

    def close(self) -> None:
        for g in self._games:
            try:
                g.b_player.close()
            except Exception:
                pass
