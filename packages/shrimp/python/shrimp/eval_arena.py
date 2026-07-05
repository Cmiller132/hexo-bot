"""Game-running layer for the shrimp multi-stage strength evaluation.

This module only plays games and returns structured result dicts; it does not
gate, promote, halt, or mutate a training run. The statistical verdict layer
(SPRT screen, pentanomial/Wilson CIs, the rolling Bradley-Terry pool) lives in
sibling modules and consumes these results.

Runners return the result-dict shape of the standalone arena
(scripts/_wf_h2h2_arena.py): ``meta`` / ``score`` (with ``by_seat``) /
``game_lengths`` / ``opening_dedup`` / per-game rows, plus a pair-level
``pentanomial`` block for paired matches.

play_checkpoint_match(model_a_ckpt, model_b_ckpt, ...)
    Shrimp-vs-shrimp, concurrent and paired. Both players are shrimp nets.
    Each round batches whichever side is to move through that net's
    ``ShrimpMctsSession.search`` multi-root call (cross-game leaf batching):
    two batched forwards per round, one per net. Games are coupled into matched
    pairs via common random numbers (CRN): a shared opening line is played from
    both seats, so a pair is a paired comparison of the two nets on one line.
    See the CRN notes below.

play_multi_checkpoint_match(candidate_ckpt, opponents, ...)
    One candidate (net A) vs many checkpoint opponents in one batched concurrent
    pass. Each round gathers the greedy candidate-to-move games across all
    opponents into one candidate-session multi-root call; each opponent searches
    its own games in its own session.

play_sealbot_match(model_ckpt, ...)
    Shrimp-vs-SealBot. Every game where shrimp is to move is searched
    together in one ``ShrimpMctsSession.search`` multi-root call (cross-game
    leaf batching); SealBot's moves are drained serially per game through the
    hexo_runner SealBot adapter. No CRN pairing: SealBot's minimax depth varies
    under load, so its games are not a matched comparison.

CRN / shared-opening note: ``hexo_engine.api.new_game(seed=...)`` does not
randomize the opening — the engine is deterministic and the first move is the
forced centre stone (api.py docstring). Opening diversity comes from the MCTS
temperature sampling at the root: the first ``opening_plies`` plies sample the
move from the visit distribution using the per-search ``seed``.

CRN under batching: the lockstep ``search`` builds a tree by deterministic PUCT
(no RNG in leaf selection). Its randomness is the final move selection (and, on
Gumbel roots, the Gumbel-Top-k draw). At ``temperature == 0`` (the
greedy tail) move selection is a deterministic argmax / LCB-of-Q (search.rs
select_action_from_policy / select_action_with_lcb), so a batched multi-root
greedy search yields the same move per game regardless of the shared batch seed.
At ``temperature > 0`` (the ``opening_plies`` sampled prefix) the per-root
selection RNG is ``seed.wrapping_add(root_index)``, i.e. each root in a batched
call samples from a distinct stream keyed by its batch position.

The shared opening line within a pair is produced by forced-opening replay: each
pair has a LEADER (game 0) and a seat-swapped FOLLOWER (game 1). Only the leader
searches its opening; the follower replays the leader's recorded opening actions
ply-for-ply (no search). Because a seat swap means a different net moves at
ply 0, a shared seed alone would not share the line, so the pairing depends on
``follower.opening == leader.opening``. Leaders are independent games (distinct
``pair_seed``, no cross-leader CRN), so their opening searches are batched
cross-game into one multi-root ``search`` per round per net, with each leader
root seeded ``open_seed.wrapping_add(root_index)`` for a decorrelated sampling
stream. The greedy tail is batched as well. The rare leader-ended-mid-opening
case falls back to a single-root follower search.

A pair whose two games agree on the winner-by-color is an "even" pentanomial
pair; a split pair is the informative one.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from .config import (
    build_divergence_overrides,
    parse_shrimp_config,
)
from .geometry import pack_action_id, unpack_action_id

# Logger for the eval .hxr writer; a 0-record write is logged as a warning.
_EVAL_LOG = logging.getLogger("shrimp.eval")

# torch / ShrimpEvaluator / ShrimpNet are imported lazily inside the
# checkpoint-loading paths so this module is importable on a CPU-only host
# without torch (the concurrent loop can be unit-tested through the
# ``build_evaluators`` / ``make_session`` seams with a numpy stub evaluator).
# Only ``_load_shrimp_net`` and the non-stub evaluator branches touch them.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model import ShrimpNet

# The native MCTS session lives in the maturin-built extension. Import lazily so
# this module is importable on hosts without the .so (e.g. CPU-only test runners
# that inject a fake session via ``make_session``/``build_opponent``); a real
# session construction without the extension raises a clear error.
try:
    from . import _rust
except ImportError:  # pragma: no cover - exercised only on hosts without the .so
    _rust = None

# Per-side MCTS search-seed offsets used in unpaired mode: the two seats draw
# from decorrelated RNG streams so their opening samples do not mirror each
# other. In paired (CRN) mode these are bypassed (both seat orderings share one
# seed) — see play_checkpoint_match.
_SIDE_SEED_OFFSET = {"a": 0, "b": 500_009_999}

# Default search/opening knobs: greedy after a temperature-sampled opening, no
# Dirichlet noise. ``opening_plies`` is in plies (single stones).
DEFAULT_OPENING_PLIES = 8
DEFAULT_OPENING_TEMPERATURE = 1.0

# Shrimp has no draws. A max_plies truncation is the only non-decisive outcome
# and is reported separately as a "truncated" game, never as a draw.


# --------------------------------------------------------------------------- #
# Shared helpers (result-dict construction; numpy-free, importable on CPU)
# --------------------------------------------------------------------------- #


def _percentile(xs: list[int], q: float) -> int | None:
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _length_stats(xs: list[int]) -> dict[str, Any] | None:
    """Same shape as scripts/_wf_h2h2_arena.py length_stats."""
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 1),
        "median": _percentile(xs, 0.5),
        "p90": _percentile(xs, 0.9),
        "min": min(xs),
        "max": max(xs),
    }


def _opening_dedup(openings: list[tuple[int, ...]]) -> dict[str, Any]:
    """Distinct-opening count + duplicate groups, arena shape. ``openings`` is
    one opening-prefix tuple per game (in game order)."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for game_index, key in enumerate(openings):
        groups.setdefault(key, []).append(game_index)
    dup_groups = {str(v[0]): v for v in groups.values() if len(v) > 1}
    return {
        "n_games": len(openings),
        "distinct_openings": len(groups),
        "duplicate_groups": dup_groups,
    }


def _new_rust_session(max_states: int) -> Any:
    """Construct a native ``ShrimpMctsSession`` (the default session factory).

    Raises a clear error when the maturin-built extension is unavailable rather
    than the bare ``AttributeError`` a ``None`` module would give. Tests that run
    without the .so inject a fake session via ``make_session`` and never hit this.
    """
    if _rust is None:
        raise RuntimeError(
            "shrimp._rust (the MCTS extension) is unavailable; build the .so or "
            "inject a session factory (make_session=) for a CPU-only run"
        )
    return _rust.ShrimpMctsSession(max_states=max_states)


def _load_shrimp_net(checkpoint: str | Path) -> ShrimpNet:
    """Strict-load a shrimp checkpoint into a fresh ShrimpNet.

    Loads ``payload["model"]`` with ``strict=True`` so a value-/moves-left-head
    mismatch raises rather than keeping a random head.
    """
    import torch  # lazy: keep the module importable on CPU hosts without torch

    from .model import ShrimpNet

    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"shrimp checkpoint is not a readable file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"shrimp checkpoint payload has no 'model' state: {path}")
    sd = payload["model"]
    # Build the net at the checkpoint's OWN arch (width, head count, trunk
    # layout) inferred from the state dict, not the process-global env constants,
    # so a checkpoint whose fields are determinable loads at its native shape.
    # Undeterminable fields fall back to the env defaults; a genuine arch
    # mismatch then fails the strict load below with a clear error.
    from .model import infer_net_kwargs_from_state_dict

    net_kwargs = infer_net_kwargs_from_state_dict(sd)
    model = ShrimpNet(**net_kwargs)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        # A checkpoint with a single shared relative-position ``bias_table`` while
        # the current model expects one ``bias_tables.{i}`` per attention block:
        # expand the shared table into per-block copies and retry (strict).
        remapped = None
        if "bias_table" in sd and any(k.startswith("bias_tables.") for k in model.state_dict()):
            remapped = {k: v for k, v in sd.items() if k != "bias_table"}
            for i in range(len(model.bias_tables)):
                remapped[f"bias_tables.{i}"] = sd["bias_table"].clone()
        if remapped is not None:
            try:
                model.load_state_dict(remapped, strict=True)
                model.eval()
                return model
            except RuntimeError:
                pass
        raise RuntimeError(
            f"checkpoint architecture does not match current ShrimpNet: {path} "
            "(only current-arch shrimp checkpoints are supported; a supplied "
            "anchor must share this build's width, head count, and trunk layout)"
        )
    model.eval()
    return model


def _resolve_eval_overrides(
    sp: Any,
    *,
    divergence_overrides: dict | None,
) -> dict:
    """The divergence overrides the arena searches with.

    Default: mirror self-play. An explicit ``divergence_overrides`` takes
    precedence.
    """
    if divergence_overrides is not None:
        return divergence_overrides
    return build_divergence_overrides(sp)


def _write_eval_hxr(
    games,
    diagnostics_dir,
    label_a,
    label_b,
    *,
    kind="checkpoint",
    stats: dict | None = None,
) -> str | None:
    """Write the eval games as a ``.hxr`` record so the dashboard can REPLAY them
    (the History screen's "evaluation" source scans ``<run>/evaluation/*.hxr``).

    Best-effort and fail-soft: any error is swallowed so recording cannot break
    the eval. One file per match at ``<run>/evaluation/epoch_NNNNNN/<a>_vs_<b>.hxr``
    (``<run>`` is the parent of ``diagnostics_dir``). Players are seat-labelled
    (player0/player1); each game id encodes the matchup and which seat the
    candidate held (seats swap per CRN pair). Returns the written path (str) or
    None.

    A 0-record write (every game had falsy ``.actions``) emits a warning and the
    write exception (if any) is logged. If ``stats`` is passed, it is populated
    with ``games_written`` / ``games_skipped`` so the caller can thread the count
    into match meta.
    """

    if stats is not None:
        stats["games_written"] = 0
        stats["games_skipped"] = 0
    if diagnostics_dir is None:
        return None
    try:
        import re
        from pathlib import Path as _P

        from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

        run_dir = _P(diagnostics_dir).parent
        m = re.search(r"(\d+)", str(label_a))
        ep = int(m.group(1)) if m else 0
        rec_dir = run_dir / "evaluation" / f"epoch_{ep:06d}"
        rec_dir.mkdir(parents=True, exist_ok=True)
        safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))
        path = rec_dir / f"{safe(label_a)}_vs_{safe(label_b)}.hxr"
        players = (
            HexoRecordPlayer("seat0", "player0", f"{label_a}/{label_b} · seat 0"),
            HexoRecordPlayer("seat1", "player1", f"{label_a}/{label_b} · seat 1"),
        )
        n = 0
        skipped = 0
        with HexoRecordFile.create(path, api.engine_metadata(), players) as rf:
            for g in games:
                if not getattr(g, "actions", None):
                    skipped += 1
                    continue
                cand_seat = "candP0" if g.a_is_p0 else "candP1"
                # play_checkpoint_match's _Game exposes ``.index``;
                # play_multi_checkpoint_match's _Game exposes ``.local_index``
                # (no ``.index``). Accept either.
                g_index = getattr(g, "index", None)
                if g_index is None:
                    g_index = getattr(g, "local_index", 0)
                writer = rf.begin_game(
                    f"ep{ep}-{label_a}-vs-{label_b}-g{g_index}-{cand_seat}", seed=g.seed
                )
                for aid in g.actions:
                    q, r = unpack_action_id(int(aid))
                    writer.record_action(PlacementAction(AxialCoord(q=q, r=r)))
                if g.winner is None:
                    writer.finish_aborted(
                        AbortRecord(
                            stage="evaluation",
                            exception_type="MaxPliesReached",
                            message="shrimp eval game reached max plies",
                        )
                    )
                else:
                    seat_w = 0 if ((g.winner == "A") == g.a_is_p0) else 1
                    writer.finish_completed(f"player{seat_w}", g.plies)
                n += 1
        if stats is not None:
            stats["games_written"] = n
            stats["games_skipped"] = skipped
        total = len(games) if hasattr(games, "__len__") else (n + skipped)
        if n == 0 and total > 0:
            # A 0-record file is produced when every game had falsy .actions.
            _EVAL_LOG.warning(
                "eval .hxr wrote 0 of %d games (all .actions empty) -> %s",
                total,
                path,
            )
        return str(path) if n else None
    except Exception as exc:  # recording is best-effort; never break the eval
        _EVAL_LOG.warning("eval .hxr write failed: %r", exc)
        return None


# --------------------------------------------------------------------------- #
# (1) Shrimp checkpoint vs shrimp checkpoint — PAIRED (CRN) games
# --------------------------------------------------------------------------- #


def play_checkpoint_match(
    model_a_ckpt: str | Path,
    model_b_ckpt: str | Path,
    n_games: int,
    *,
    config: Any = None,
    label_a: str = "A",
    label_b: str = "B",
    paired_openings: bool = True,
    visits: int | None = None,
    virtual_batch_size: int | None = None,
    opening_plies: int = DEFAULT_OPENING_PLIES,
    opening_temperature: float = DEFAULT_OPENING_TEMPERATURE,
    divergence_overrides_a: dict | None = None,
    divergence_overrides_b: dict | None = None,
    diagnostics_dir: str | Path | None = None,
    max_states: int = 65_536,
    game_seed_base: int = 0,
    max_wall_seconds: float = 0.0,
    active_root_limit: int | None = None,
    batch_openings: bool = False,
    build_evaluators: Callable[..., tuple[Any, Any]] | None = None,
    make_session: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Play model A vs model B concurrently and return a structured pentanomial
    result.

    Both players are shrimp nets. The runner keeps two persistent sessions and
    evaluators (one per net, keyed by game index) and plays all games in lockstep
    rounds. Each round advances every active game by one ply: the games where net
    A is to move are batched through net A's session in one (or, above
    ``active_root_limit``, a few chunked) multi-root ``search`` call, and the
    games where net B is to move likewise through net B's session. The net's move
    is read from the search result and applied to that game's engine state. Search
    runs at full visits; parallelism, not fewer sims, is the throughput lever.

    Pairing (``paired_openings=True``, the default): games are grouped into
    ``n_pairs = ceil(n_games / 2)`` matched pairs. Both games of a pair use the
    same CRN ``pair_seed`` (shared opening-temperature sampling — see the module
    CRN note) but swap seats: game 0 plays A-as-player0, game 1 plays
    B-as-player0. Every in-flight game carries its own ``pair_index`` /
    ``pair_seed`` / ``a_is_p0`` / running ply count. The shared opening line
    within a pair comes from forced-opening replay: only each pair's LEADER
    searches its opening and the seat-swapped FOLLOWER replays the leader's
    recorded actions (no search). Leaders are independent games, so all leaders'
    opening-ply searches are batched cross-game into one multi-root call per round
    per net (each leader root seeded ``open_seed + root_index`` to decorrelate),
    and the greedy tail batches (it is RNG-free at temperature 0) — see the module
    "CRN under batching" note. Each pair yields one pentanomial outcome: how many
    of its two games net A won (0, 1, or 2), plus the seat pattern, feeding the
    downstream pair-level SE (N_pairs units) and the pentanomial->BT mapping.

    Unpaired (``paired_openings=False``): every game gets an independent seed and
    seats alternate by game index (independent-Bernoulli layout).

    Sims: ``visits=None`` defaults to ``cfg.selfplay.search_visits``, not
    ``cfg.evaluation.eval_visits``; an explicit ``visits`` overrides.

    ``active_root_limit`` caps a single multi-root batch (defaults to
    ``cfg.selfplay.active_root_limit``); larger to-move groups are chunked.
    ``batch_openings`` (default False): leaders' opening plies are always batched
    cross-game; this flag changes only the FOLLOWER opening. When False, followers
    replay their leader's recorded opening so the pair shares the opening line.
    When True, the leader/follower split is dropped and every game's opening is
    batched with one decorrelated per-root seed (no replay), which forgoes the
    within-pair shared opening and the paired pentanomial.
    ``build_evaluators`` / ``make_session`` are CPU-test injection seams: given
    them, the runner skips checkpoint loading / GPU and uses the supplied
    (eval_a, eval_b) and session factory. ``build_evaluators`` is called with no
    args and returns ``(eval_a, eval_b)``; ``make_session`` is called with no
    args and returns a fresh session.

    Returns a dict with ``meta`` / ``score`` / ``pentanomial`` / ``game_lengths``
    / ``opening_dedup`` / ``games`` (see module docstring). Win counts in
    ``score`` are net-A-centric ("a_wins"). ``pentanomial.pairs`` is the list the
    pair-level statistics consume; for unpaired runs ``pentanomial`` is ``None``.
    """

    cfg = config if config is not None else parse_shrimp_config({})
    sp = cfg.selfplay
    # visits=None defaults to the self-play search budget, not eval_visits.
    eval_visits = int(visits) if visits is not None else int(sp.search_visits)
    # Eval-only virtual-batch-size override, independent of
    # SelfplayConfig.virtual_batch_size. None -> self-play value.
    vbs = int(virtual_batch_size) if virtual_batch_size is not None else int(sp.virtual_batch_size)
    root_limit = int(active_root_limit) if active_root_limit is not None else int(sp.active_root_limit)
    new_session = make_session if make_session is not None else (
        lambda: _new_rust_session(max_states)
    )

    started = time.perf_counter()
    if build_evaluators is not None:
        # CPU-test seam: skip checkpoint loading + GPU; use the supplied pair.
        eval_a, eval_b = build_evaluators()
    else:
        from .inference import ShrimpEvaluator  # lazy: torch only on the GPU path

        model_a = _load_shrimp_net(model_a_ckpt)
        model_b = _load_shrimp_net(model_b_ckpt)
        eval_a = ShrimpEvaluator(model_a, device=cfg.device)
        eval_b = ShrimpEvaluator(model_b, device=cfg.device)

    # Symmetric divergence overrides by default; an explicit per-net override
    # drives a search-change A/B. The override follows the searching net:
    # ``ov_a`` whenever net A is to move, ``ov_b`` whenever net B is to move,
    # independent of which engine seat each net holds. Each round's two batched
    # searches are single-net, so each carries one net's ov.
    ov_a = _resolve_eval_overrides(sp, divergence_overrides=divergence_overrides_a)
    ov_b = (
        ov_a
        if divergence_overrides_b is None
        else _resolve_eval_overrides(sp, divergence_overrides=divergence_overrides_b)
    )

    # ----- Build the in-flight game set (seats + CRN seeds).
    class _Game:
        __slots__ = (
            "index", "pair_index", "a_is_p0", "seed", "state",
            "plies", "done", "status", "winner", "opening", "actions",
            "is_leader", "leader",
        )

        def __init__(self, index: int, pair_index: int, a_is_p0: bool, seed: int) -> None:
            self.index = index
            self.pair_index = pair_index
            self.a_is_p0 = a_is_p0
            self.seed = seed  # the CRN seed
            self.state = api.new_game()
            self.plies = 0
            self.done = False
            self.status = "truncated"
            self.winner: str | None = None  # "A" | "B" | None (net-A-centric)
            self.opening: list[int] = []
            # FULL ordered move sequence (action_ids, engine move order) so the
            # game is replayable as a .hxr record on the dashboard. Distinct from
            # ``opening`` (only the first ``opening_plies`` actions).
            self.actions: list[int] = []
            # Forced-opening CRN: each pair has a LEADER (game 0, the first seat
            # ordering created) and a FOLLOWER (game 1, the seat-swapped sibling).
            # The leader searches its opening; the follower replays the leader's
            # recorded opening actions ply-for-ply (no search), so the pair shares
            # the opening line. ``leader`` points the follower at its leader so it
            # can read ``leader.opening[ply]``; a leader's ``leader`` is itself.
            self.is_leader = True
            self.leader: _Game = self

        # Engine seat (player0/player1) net A occupies in this game.
        @property
        def a_role(self) -> Any:
            return api.Player.PLAYER_0 if self.a_is_p0 else api.Player.PLAYER_1

        @property
        def a_role_label(self) -> str:
            return "player0" if self.a_is_p0 else "player1"

        def a_to_move(self) -> bool:
            return api.current_player(self.state) == self.a_role

    games: list[_Game] = []
    pair_members: dict[int, list[_Game]] = {}
    if paired_openings:
        n_pairs = (n_games + 1) // 2
        for pair_index in range(n_pairs):
            pair_seed = game_seed_base + pair_index  # shared CRN seed (both seats)
            idx0 = pair_index * 2
            g0 = _Game(idx0, pair_index, a_is_p0=True, seed=pair_seed)
            games.append(g0)
            pair_members.setdefault(pair_index, []).append(g0)
            if idx0 + 1 < n_games:  # odd n_games -> last pair is a singleton
                g1 = _Game(idx0 + 1, pair_index, a_is_p0=False, seed=pair_seed)
                # g1 follows g0: it replays g0's opening line.
                g1.is_leader = False
                g1.leader = g0
                games.append(g1)
                pair_members[pair_index].append(g1)
    else:
        for game_index in range(n_games):
            seed = game_seed_base + game_index + (
                _SIDE_SEED_OFFSET["b"] if game_index % 2 else _SIDE_SEED_OFFSET["a"]
            )
            games.append(_Game(game_index, -1, a_is_p0=(game_index % 2 == 0), seed=seed))

    # ----- Two persistent sessions, one per net, keyed by game index. The Rust
    # per-game-key tree store keeps trees from crossing games, and each game's
    # tree is discarded at end, so trees are never reused across games.
    s_net_a = new_session()
    s_net_b = new_session()
    budget_hit = False
    rounds = 0
    forward_batches = 0
    mcts_search_elapsed = 0.0

    def _finalize(g: _Game) -> None:
        terminal = api.terminal(g.state)
        if terminal is not None:
            g.status = "completed"
            if terminal.winner is None:
                g.winner = None  # hexo has no draws; defensive
            else:
                won_label = str(terminal.winner)  # "player0" / "player1"
                g.winner = "A" if won_label == g.a_role_label else "B"
        elif budget_hit:
            g.status = "aborted_budget"
            g.winner = None
        else:
            g.status = "truncated"
            g.winner = None
        g.done = True
        s_net_a.discard(g.index)
        s_net_b.discard(g.index)

    def _settle(g: _Game) -> None:
        if api.terminal(g.state) is not None or g.plies >= sp.max_game_plies:
            _finalize(g)

    # Common search knobs shared by every batched/single-root call: greedy after
    # a sampled opening, no Dirichlet noise. Per-net override and session are
    # supplied at each call site.
    common = dict(
        visits=eval_visits,
        c_puct=sp.c_puct,
        temperature=0.0,
        virtual_batch_size=vbs,
        active_root_limit=root_limit,
        widening_policy_mass=sp.widening_policy_mass,
        widening_max_children=sp.widening_max_children,
        widening_min_children=sp.widening_min_children,
        fpu_reduction=sp.fpu_reduction,
        tss_enabled=sp.tss_enabled,
        search_parity_mode=sp.search_parity_mode,
    )

    def _net_for(g: _Game) -> str:
        """Which NET is to move in ``g`` ('A' or 'B')."""
        return "A" if g.a_to_move() else "B"

    def _temp(g: _Game) -> float:
        """Opening temperature while ``plies < opening_plies``, then greedy (0)."""
        return opening_temperature if (g.plies < opening_plies and opening_temperature > 0.0) else 0.0

    def _apply_search(g: _Game, search: dict[str, Any]) -> None:
        q, r = unpack_action_id(int(search["action_id"]))
        api.apply_action(g.state, PlacementAction(AxialCoord(q=q, r=r)))
        g.plies += 1
        g.actions.append(int(search["action_id"]))
        if len(g.opening) < opening_plies:
            g.opening.append(int(search["action_id"]))
        _settle(g)

    def _replay_action(g: _Game, action_id: int) -> None:
        """Apply a pre-decided opening action to a follower game without search.
        Same plies/opening/settle bookkeeping as ``_apply_search``; only the move
        source differs (the leader's recorded action, not a fresh search)."""
        q, r = unpack_action_id(int(action_id))
        api.apply_action(g.state, PlacementAction(AxialCoord(q=q, r=r)))
        g.plies += 1
        g.actions.append(int(action_id))
        if len(g.opening) < opening_plies:
            g.opening.append(int(action_id))
        _settle(g)

    def _follower_opening_action(g: _Game) -> int | None:
        """The leader's recorded action for the follower ``g``'s current opening
        ply, or ``None`` if the leader has no action for that ply. The round order
        keeps the leader strictly ahead of the follower, so a recorded action
        normally exists; if the leader's game ended during its own opening it may
        have fewer than ``opening_plies`` actions, in which case the follower
        falls back to a single-root search for the remaining opening plies."""
        line = g.leader.opening
        return line[g.plies] if g.plies < len(line) else None

    def _run_batch(net: str, batch: list[_Game], seed: int) -> int:
        """One multi-root ``search`` for ``batch`` (all to-move for ``net``),
        chunked at ``root_limit``. Returns the number of plies applied."""
        nonlocal mcts_search_elapsed, forward_batches
        session = s_net_a if net == "A" else s_net_b
        evaluator = eval_a if net == "A" else eval_b
        # Every game in the batch has the same net to move: net A uses ov_a, net B
        # uses ov_b. For the default symmetric case ov_a == ov_b.
        overrides = ov_a if net == "A" else ov_b
        applied = 0
        for start in range(0, len(batch), root_limit):
            chunk = batch[start : start + root_limit]
            move_temperatures = [_temp(g) for g in chunk]
            t0 = time.perf_counter()
            searches = session.search(
                [g.index for g in chunk],
                tuple(g.state for g in chunk),
                seed=seed,
                evaluator=evaluator,
                move_temperatures=move_temperatures,
                divergence_overrides=overrides,
                **common,
            )
            mcts_search_elapsed += time.perf_counter() - t0
            forward_batches += 1
            if len(searches) != len(chunk):
                raise RuntimeError(
                    f"shrimp checkpoint eval search returned {len(searches)} "
                    f"results for {len(chunk)} games"
                )
            for g, search in zip(chunk, searches):
                _apply_search(g, search)
                applied += 1
        return applied

    def _run_opening_batch(net: str, batch: list[_Game], seed: int) -> int:
        """One multi-root ``search`` for the opening-ply LEADERS to-move for
        ``net`` (chunked at ``root_limit``). Every root carries
        ``opening_temperature`` so each leader samples its opening move, and the
        native per-root selection seed ``seed.wrapping_add(root_index)`` gives each
        leader its own decorrelated stream (leaders carry no cross-leader CRN).
        Returns the number of plies applied.

        Same structure as ``_run_batch`` except the temperatures are pinned to
        ``opening_temperature`` (these games are all at ``plies < opening_plies``)
        and the base seed is the per-(net, round) opening stream. Each leader's
        sampled action is recorded into ``g.opening`` by ``_apply_search`` so
        followers have a line to replay."""
        nonlocal mcts_search_elapsed, forward_batches
        session = s_net_a if net == "A" else s_net_b
        evaluator = eval_a if net == "A" else eval_b
        overrides = ov_a if net == "A" else ov_b
        applied = 0
        for start in range(0, len(batch), root_limit):
            chunk = batch[start : start + root_limit]
            t0 = time.perf_counter()
            searches = session.search(
                [g.index for g in chunk],
                tuple(g.state for g in chunk),
                seed=seed,
                evaluator=evaluator,
                move_temperatures=[opening_temperature] * len(chunk),
                divergence_overrides=overrides,
                **common,
            )
            mcts_search_elapsed += time.perf_counter() - t0
            forward_batches += 1
            if len(searches) != len(chunk):
                raise RuntimeError(
                    f"shrimp checkpoint eval opening search returned {len(searches)} "
                    f"results for {len(chunk)} games"
                )
            for g, search in zip(chunk, searches):
                _apply_search(g, search)
                applied += 1
        return applied

    def _run_single(g: _Game, net: str) -> None:
        """Single-root ``search`` for one game with seed ``g.seed * 5003 +
        g.plies`` (per-root index 0). Used only as the follower fallback when its
        leader ended before recording an action for this opening ply. The follower
        shares its leader's seed, so this fallback's RNG is
        ``pair_seed * 5003 + ply``."""
        nonlocal mcts_search_elapsed, forward_batches
        session = s_net_a if net == "A" else s_net_b
        evaluator = eval_a if net == "A" else eval_b
        overrides = ov_a if net == "A" else ov_b
        t0 = time.perf_counter()
        searches = session.search(
            [g.index],
            (g.state,),
            seed=g.seed * 5003 + g.plies,
            evaluator=evaluator,
            move_temperatures=[_temp(g)],
            divergence_overrides=overrides,
            **common,
        )
        mcts_search_elapsed += time.perf_counter() - t0
        forward_batches += 1
        _apply_search(g, searches[0])

    # ----- Round loop: each round advances every active game by at least one ply
    # (a game whose seat-to-move flips after net A's batch is also played in net
    # B's recomputed to-move set the same round, so it can advance up to 2 plies).
    # Per round and per net, the opening leaders are batched in one multi-root
    # forward and the greedy to-move games in another; followers replay their
    # leader's recorded opening (no search). ``batch_openings`` collapses the
    # leader/follower distinction (everything goes through ``greedy``). A round
    # that makes no progress raises rather than hangs.
    #
    # Within the opening, a pair's LEADER searches; its seat-swapped FOLLOWER
    # replays the leader's recorded action for that ply, so both games traverse
    # the same opening line. Leaders are independent games (no cross-leader CRN),
    # so all leaders to-move for a net are searched in one multi-root
    # ``_run_opening_batch`` call (each leader root decorrelated by the native
    # per-root ``seed+index`` offset). The round order (net A pass then net B pass)
    # keeps the leader strictly ahead of the follower at every follower move, so
    # the action to replay is already recorded; the rare leader-ended-mid-opening
    # case falls back to a follower single-root search.
    while True:
        active = [g for g in games if not g.done]
        if not active:
            break
        if max_wall_seconds and (time.perf_counter() - started) > max_wall_seconds:
            budget_hit = True
            for g in active:
                _finalize(g)
            break
        rounds += 1
        plies_this_round = 0
        for net in ("A", "B"):
            to_move = [g for g in active if not g.done and _net_for(g) == net]
            if not to_move:
                continue
            # Opening plies: in paired mode (and unless batch_openings) leaders
            # search (batched cross-game) and followers replay the leader's
            # recorded line; games past the opening are batched as greedy.
            if paired_openings and not batch_openings:
                openers = [g for g in to_move if g.plies < opening_plies and g.is_leader]
                followers = [g for g in to_move if g.plies < opening_plies and not g.is_leader]
                greedy = [g for g in to_move if g.plies >= opening_plies]
            else:
                openers = []
                followers = []
                greedy = to_move
            if openers:
                # These leaders are independent games, batched into one multi-root
                # call. The per-(net, round) base seed plus the native per-root
                # ``seed+index`` offset gives each leader its own decorrelated
                # stream. The opening base offsets (13M/19M) are distinct from the
                # greedy offsets (0/7M below) for both nets, so an opening batch and
                # a greedy batch in the same round never share a base seed.
                open_seed = (
                    game_seed_base + (13_000_003 if net == "A" else 19_000_003) + rounds * 1_000_003
                )
                plies_this_round += _run_opening_batch(net, openers, open_seed)
            for g in followers:
                # Replay the leader's recorded opening action; if the leader ended
                # its game before reaching this ply (no recorded action), fall back
                # to a normal single-root CRN search so the follower still moves.
                replay = _follower_opening_action(g)
                if replay is not None:
                    _replay_action(g, replay)
                else:
                    _run_single(g, net)
                plies_this_round += 1
            if greedy:
                # Per-round, per-net batch seed. Greedy plies are temperature 0, so
                # this RNG only tie-breaks; the value is decorrelated per net/round.
                batch_seed = (
                    game_seed_base + (0 if net == "A" else 7_000_003) + rounds * 1_000_003
                )
                plies_this_round += _run_batch(net, greedy, batch_seed)
        if plies_this_round == 0:
            raise RuntimeError(
                "shrimp checkpoint eval made no progress in a round; aborting to avoid a hang"
            )

    # ----- Re-key the in-flight games to the result rows and per-pair rows.
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
                    # Pentanomial class for a 2-game pair: 2/1/0 net-A wins among
                    # the 2 decided games. Singleton/partial pairs report their
                    # decided count so the consumer can weight them correctly.
                    "pentanomial_a_score": a_wins_in_pair,
                }
            )

    # Persist the games as a replayable .hxr (dashboard "evaluation" source).
    # Best-effort: _write_eval_hxr is fully fail-soft.
    _hxr_stats: dict[str, int] = {}
    hxr_path = _write_eval_hxr(games, diagnostics_dir, label_a, label_b, stats=_hxr_stats)

    result = _build_match_result(
        games=game_rows,
        pairs=pairs if paired_openings else None,
        label_a=label_a,
        label_b=label_b,
        meta_extra={
            "kind": "shrimp_vs_shrimp",
            "hxr_record": hxr_path,
            "hxr_games_written": _hxr_stats.get("games_written", 0),
            "ckpt_a": {"label": label_a, "path": str(model_a_ckpt)},
            "ckpt_b": {"label": label_b, "path": str(model_b_ckpt)},
            "games_requested": n_games,
            "visits": eval_visits,
            "virtual_batch_size": vbs,
            "device": cfg.device,
            "paired_openings": paired_openings,
            "opening_plies": opening_plies,
            "opening_temperature": opening_temperature,
            "game_seed_base": game_seed_base,
            "divergence_overrides_a": ov_a,
            "divergence_overrides_b": ov_b,
            "budget_hit": budget_hit,
            # Concurrency telemetry (additive — downstream consumers ignore these).
            "concurrent": True,
            "batch_openings": bool(batch_openings),
            "rounds": rounds,
            "forward_batches": forward_batches,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "mcts_search_elapsed_seconds": round(mcts_search_elapsed, 2),
        },
    )
    return result


# --------------------------------------------------------------------------- #
# (1b) Concurrent multi-opponent checkpoint match — one candidate forward across
#      every opponent's candidate-to-move games per round; each opponent searched
#      in its own session.
# --------------------------------------------------------------------------- #


def play_multi_checkpoint_match(
    candidate_ckpt: str | Path,
    opponents: list[tuple[str, str | Path]],
    n_games_per_opponent: int,
    *,
    config: Any = None,
    candidate_label: str = "cand",
    visits: int | None = None,
    virtual_batch_size: int | None = None,
    opening_plies: int = DEFAULT_OPENING_PLIES,
    opening_temperature: float = DEFAULT_OPENING_TEMPERATURE,
    divergence_overrides_candidate: dict | None = None,
    divergence_overrides_opponent: dict | None = None,
    divergence_overrides_by_opponent: dict[str, dict] | None = None,
    diagnostics_dir: str | Path | None = None,
    max_states: int = 65_536,
    game_seed_base: int = 0,
    max_wall_seconds: float = 0.0,
    active_root_limit: int | None = None,
    build_candidate_evaluator: Callable[..., Any] | None = None,
    build_opponent_evaluator: Callable[..., Any] | None = None,
    make_session: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Play the candidate (always net A) vs many checkpoint opponents in one
    batched concurrent pass and return ``{opponent_label: match_result_dict}``.

    Each opponent's ``match_result_dict`` has the shape
    :func:`play_checkpoint_match` returns (``meta`` / ``score`` /
    ``pentanomial`` / ``game_lengths`` / ``opening_dedup`` / ``games``), consumed
    downstream by ``multistage_eval._checkpoint_edge_counts`` ->
    ``eval_stats.effective_counts`` -> ``BTEdge``.

    Shared candidate forward: the candidate is net A in every game across every
    opponent. It keeps one persistent session and evaluator; each round, the
    greedy candidate-to-move games across all opponents are searched in one
    multi-root candidate-session call. Each opponent keeps its own
    session+evaluator and searches only its own games. Wall-clock is then max over
    opponents, not sum.

    Each opponent group is constructed identically to a standalone
    ``play_checkpoint_match(candidate, opponent_b, n_games, paired_openings=True,
    game_seed_base=game_seed_base, ...)`` — same CRN pairs, same per-pair
    ``pair_seed = game_seed_base + pair_index``, same ``a_is_p0`` seat pattern,
    same leader/follower forced-opening replay. What differs is when the
    candidate's searches fire:

      * Greedy plies (temperature 0): move selection is a deterministic argmax /
        LCB-of-Q (search.rs select_search_action), so every opponent's greedy
        candidate games are merged into one multi-root call. This is the bulk
        of plies. NOTE: under a gumbel profile the search itself is seeded (the
        root Gumbel-Top-k draw mixes the per-call seed, game key, and root
        hash), so greedy moves are deterministic per (seed, key, position) but
        no longer seed-independent as they are for plain PUCT.

      * Opening-leader plies (temperature > 0): the per-root seed
        ``open_seed.wrapping_add(root_index)`` matters, so each opponent's
        candidate opening leaders are searched in their own per-opponent multi-root
        call with that opponent's ``open_seed`` (= ``game_seed_base + 13_000_003 +
        rounds*1_000_003``) and per-group root_index. Followers replay their
        leader's recorded line (no search).

    Game-key namespacing: ``ShrimpMctsSession.search`` keys trees by game_key in
    a HashMap. The candidate session holds trees for games from all opponents at
    once, so each game's candidate-side key is a global ``opp_index * KEY_STRIDE +
    local_index`` (KEY_STRIDE >> any plausible per-opponent game count),
    discarded at game end so candidate trees never collide across opponent groups.
    Each opponent session uses the local per-group index, discarded at game end.

    Also writes one ``.hxr`` eval-game record per opponent under
    ``<run>/evaluation/epoch_N/`` via ``_write_eval_hxr``.

    ``build_candidate_evaluator`` / ``build_opponent_evaluator`` / ``make_session``
    are CPU-test injection seams (no torch/CUDA). ``build_candidate_evaluator()``
    -> candidate evaluator; ``build_opponent_evaluator(label, ckpt_path)`` ->
    that opponent's evaluator; ``make_session()`` -> a fresh session.
    """

    cfg = config if config is not None else parse_shrimp_config({})
    sp = cfg.selfplay
    eval_visits = int(visits) if visits is not None else int(sp.search_visits)
    vbs = int(virtual_batch_size) if virtual_batch_size is not None else int(sp.virtual_batch_size)
    root_limit = int(active_root_limit) if active_root_limit is not None else int(sp.active_root_limit)
    new_session = make_session if make_session is not None else (
        lambda: _new_rust_session(max_states)
    )

    started = time.perf_counter()

    # ----- Evaluators: ONE candidate (net A everywhere) + ONE per opponent. -----
    if build_candidate_evaluator is not None:
        cand_eval = build_candidate_evaluator()
    else:
        from .inference import ShrimpEvaluator  # lazy: torch only on the GPU path

        cand_eval = ShrimpEvaluator(_load_shrimp_net(candidate_ckpt), device=cfg.device)

    # ov follows the searching net: the candidate (net A) searches with
    # ``ov_cand``, an opponent with ``ov_opp``. Symmetric by default.
    ov_cand = _resolve_eval_overrides(
        sp, divergence_overrides=divergence_overrides_candidate
    )
    ov_opp = (
        ov_cand
        if divergence_overrides_opponent is None
        else _resolve_eval_overrides(
            sp, divergence_overrides=divergence_overrides_opponent
        )
    )
    # Per-opponent override map (keyed by opponent label): an entry takes
    # precedence over the shared ``divergence_overrides_opponent``; absent
    # labels fall back to ``ov_opp``. Lets one concurrent pass mix search
    # profiles (e.g. this-run lineage on the self-play profile while foreign
    # anchors keep their original PUCT profile).
    _by_label = divergence_overrides_by_opponent or {}
    ov_by_label = {
        label: (
            _resolve_eval_overrides(
                sp,
                divergence_overrides=_by_label[label],
            )
            if label in _by_label
            else ov_opp
        )
        for label, _ckpt in opponents
    }

    common = dict(
        visits=eval_visits,
        c_puct=sp.c_puct,
        temperature=0.0,
        virtual_batch_size=vbs,
        active_root_limit=root_limit,
        widening_policy_mass=sp.widening_policy_mass,
        widening_max_children=sp.widening_max_children,
        widening_min_children=sp.widening_min_children,
        fpu_reduction=sp.fpu_reduction,
        tss_enabled=sp.tss_enabled,
        search_parity_mode=sp.search_parity_mode,
    )

    # Per-game state. Like play_checkpoint_match._Game but tracks the opponent
    # group and a global candidate-session key so candidate trees never collide.
    class _Game:
        __slots__ = (
            "opp_index", "local_index", "cand_key", "pair_index", "a_is_p0",
            "seed", "state", "plies", "done", "status", "winner", "opening",
            "actions", "is_leader", "leader",
        )

        def __init__(self, opp_index: int, local_index: int, cand_key: int,
                     pair_index: int, a_is_p0: bool, seed: int) -> None:
            self.opp_index = opp_index
            self.local_index = local_index  # key in this opponent's session
            self.cand_key = cand_key        # global key in the candidate session
            self.pair_index = pair_index
            self.a_is_p0 = a_is_p0
            self.seed = seed
            self.state = api.new_game()
            self.plies = 0
            self.done = False
            self.status = "truncated"
            self.winner: str | None = None  # "A" | "B" | None (candidate-centric)
            self.opening: list[int] = []
            self.actions: list[int] = []
            self.is_leader = True
            self.leader: _Game = self

        @property
        def a_role(self) -> Any:
            return api.Player.PLAYER_0 if self.a_is_p0 else api.Player.PLAYER_1

        @property
        def a_role_label(self) -> str:
            return "player0" if self.a_is_p0 else "player1"

        def a_to_move(self) -> bool:
            return api.current_player(self.state) == self.a_role

    # KEY_STRIDE namespaces candidate-session game keys per opponent so two
    # opponents' games never share a candidate tree (n_games_per_opponent << this
    # stride).
    KEY_STRIDE = 1_000_000

    # One opponent group per (label, ckpt). Each group builds the same game layout
    # a standalone play_checkpoint_match would for that opponent.
    class _Group:
        __slots__ = ("opp_index", "label", "ckpt", "session", "evaluator",
                     "games", "pair_members")

        def __init__(self, opp_index, label, ckpt, session, evaluator):
            self.opp_index = opp_index
            self.label = label
            self.ckpt = ckpt
            self.session = session
            self.evaluator = evaluator
            self.games: list[_Game] = []
            self.pair_members: dict[int, list[_Game]] = {}

    groups: list[_Group] = []
    cand_session = new_session()
    for opp_index, (label, ckpt) in enumerate(opponents):
        if build_opponent_evaluator is not None:
            opp_eval = build_opponent_evaluator(label, ckpt)
        else:
            from .inference import ShrimpEvaluator  # lazy: torch only on GPU path

            opp_eval = ShrimpEvaluator(_load_shrimp_net(ckpt), device=cfg.device)
        grp = _Group(opp_index, label, ckpt, new_session(), opp_eval)
        # Build CRN pairs identically to play_checkpoint_match (paired_openings).
        n_pairs = (n_games_per_opponent + 1) // 2
        base = opp_index * KEY_STRIDE
        for pair_index in range(n_pairs):
            pair_seed = game_seed_base + pair_index  # shared CRN seed (both seats)
            idx0 = pair_index * 2
            g0 = _Game(opp_index, idx0, base + idx0, pair_index, a_is_p0=True, seed=pair_seed)
            grp.games.append(g0)
            grp.pair_members.setdefault(pair_index, []).append(g0)
            if idx0 + 1 < n_games_per_opponent:
                g1 = _Game(opp_index, idx0 + 1, base + idx0 + 1, pair_index,
                           a_is_p0=False, seed=pair_seed)
                g1.is_leader = False
                g1.leader = g0
                grp.games.append(g1)
                grp.pair_members[pair_index].append(g1)
        groups.append(grp)

    all_games: list[_Game] = [g for grp in groups for g in grp.games]

    budget_hit = False
    rounds = 0
    forward_batches = 0
    cand_forward_batches = 0
    mcts_search_elapsed = 0.0

    def _opp_session(g: _Game) -> Any:
        return groups[g.opp_index].session

    def _finalize(g: _Game) -> None:
        terminal = api.terminal(g.state)
        if terminal is not None:
            g.status = "completed"
            if terminal.winner is None:
                g.winner = None
            else:
                won_label = str(terminal.winner)  # "player0" / "player1"
                g.winner = "A" if won_label == g.a_role_label else "B"
        elif budget_hit:
            g.status = "aborted_budget"
            g.winner = None
        else:
            g.status = "truncated"
            g.winner = None
        g.done = True
        cand_session.discard(g.cand_key)
        _opp_session(g).discard(g.local_index)

    def _settle(g: _Game) -> None:
        if api.terminal(g.state) is not None or g.plies >= sp.max_game_plies:
            _finalize(g)

    def _temp(g: _Game) -> float:
        return opening_temperature if (g.plies < opening_plies and opening_temperature > 0.0) else 0.0

    def _apply_search(g: _Game, search: dict[str, Any]) -> None:
        q, r = unpack_action_id(int(search["action_id"]))
        api.apply_action(g.state, PlacementAction(AxialCoord(q=q, r=r)))
        g.plies += 1
        g.actions.append(int(search["action_id"]))
        if len(g.opening) < opening_plies:
            g.opening.append(int(search["action_id"]))
        _settle(g)

    def _replay_action(g: _Game, action_id: int) -> None:
        q, r = unpack_action_id(int(action_id))
        api.apply_action(g.state, PlacementAction(AxialCoord(q=q, r=r)))
        g.plies += 1
        g.actions.append(int(action_id))
        if len(g.opening) < opening_plies:
            g.opening.append(int(action_id))
        _settle(g)

    def _follower_opening_action(g: _Game) -> int | None:
        line = g.leader.opening
        return line[g.plies] if g.plies < len(line) else None

    def _candidate_key(g: _Game) -> int:
        return g.cand_key

    def _run_candidate_greedy(batch: list[_Game], seed: int) -> int:
        """One shared candidate-session multi-root search over the greedy
        candidate-to-move games across all opponents (chunked at root_limit). At
        temperature 0 the move is a deterministic seed-independent argmax."""
        nonlocal mcts_search_elapsed, forward_batches, cand_forward_batches
        applied = 0
        for start in range(0, len(batch), root_limit):
            chunk = batch[start : start + root_limit]
            t0 = time.perf_counter()
            searches = cand_session.search(
                [_candidate_key(g) for g in chunk],
                tuple(g.state for g in chunk),
                seed=seed,
                evaluator=cand_eval,
                move_temperatures=[0.0] * len(chunk),
                divergence_overrides=ov_cand,
                **common,
            )
            mcts_search_elapsed += time.perf_counter() - t0
            forward_batches += 1
            cand_forward_batches += 1
            if len(searches) != len(chunk):
                raise RuntimeError(
                    f"shrimp multi-checkpoint candidate greedy search returned "
                    f"{len(searches)} results for {len(chunk)} games"
                )
            for g, search in zip(chunk, searches):
                _apply_search(g, search)
                applied += 1
        return applied

    def _run_candidate_opening(grp: "_Group", openers: list[_Game], seed: int) -> int:
        """Per-opponent candidate opening-leader batch. Uses the candidate session
        and evaluator but this opponent's own ``open_seed`` and per-group
        root_index, so each leader's native ``seed+root_index`` sampling stream is
        keyed per opponent."""
        nonlocal mcts_search_elapsed, forward_batches, cand_forward_batches
        applied = 0
        for start in range(0, len(openers), root_limit):
            chunk = openers[start : start + root_limit]
            t0 = time.perf_counter()
            searches = cand_session.search(
                [_candidate_key(g) for g in chunk],
                tuple(g.state for g in chunk),
                seed=seed,
                evaluator=cand_eval,
                move_temperatures=[opening_temperature] * len(chunk),
                divergence_overrides=ov_cand,
                **common,
            )
            mcts_search_elapsed += time.perf_counter() - t0
            forward_batches += 1
            cand_forward_batches += 1
            if len(searches) != len(chunk):
                raise RuntimeError(
                    f"shrimp multi-checkpoint candidate opening search returned "
                    f"{len(searches)} results for {len(chunk)} games"
                )
            for g, search in zip(chunk, searches):
                _apply_search(g, search)
                applied += 1
        return applied

    def _run_opponent_batch(grp: "_Group", batch: list[_Game], seed: int,
                            *, temperature: float | None) -> int:
        """One multi-root search for the opponent (net B) to-move games in this
        opponent's session (chunked at root_limit). ``temperature`` None -> per-game
        temperature via ``_temp``; a float pins it (opening leaders)."""
        nonlocal mcts_search_elapsed, forward_batches
        applied = 0
        for start in range(0, len(batch), root_limit):
            chunk = batch[start : start + root_limit]
            temps = (
                [temperature] * len(chunk)
                if temperature is not None
                else [_temp(g) for g in chunk]
            )
            t0 = time.perf_counter()
            searches = grp.session.search(
                [g.local_index for g in chunk],
                tuple(g.state for g in chunk),
                seed=seed,
                evaluator=grp.evaluator,
                move_temperatures=temps,
                divergence_overrides=ov_by_label[grp.label],
                **common,
            )
            mcts_search_elapsed += time.perf_counter() - t0
            forward_batches += 1
            if len(searches) != len(chunk):
                raise RuntimeError(
                    f"shrimp multi-checkpoint opponent search returned "
                    f"{len(searches)} results for {len(chunk)} games"
                )
            for g, search in zip(chunk, searches):
                _apply_search(g, search)
                applied += 1
        return applied

    def _run_single(g: _Game, net: str) -> None:
        """Single-root follower fallback (leader ended mid-opening). Uses seed
        ``g.seed * 5003 + g.plies`` and the session/evaluator for ``net``."""
        nonlocal mcts_search_elapsed, forward_batches, cand_forward_batches
        if net == "A":
            session, evaluator, key, ov = cand_session, cand_eval, g.cand_key, ov_cand
        else:
            grp = groups[g.opp_index]
            session, evaluator, key, ov = (
                grp.session, grp.evaluator, g.local_index, ov_by_label[grp.label],
            )
        t0 = time.perf_counter()
        searches = session.search(
            [key],
            (g.state,),
            seed=g.seed * 5003 + g.plies,
            evaluator=evaluator,
            move_temperatures=[_temp(g)],
            divergence_overrides=ov,
            **common,
        )
        mcts_search_elapsed += time.perf_counter() - t0
        forward_batches += 1
        if net == "A":
            cand_forward_batches += 1
        _apply_search(g, searches[0])

    # ----- Round loop. Per round: (1) candidate pass — gather every opponent's
    # candidate-to-move games; search the opening leaders per-opponent (own
    # open_seed) and the greedy games in one shared cross-opponent call; followers
    # replay. (2) opponent pass — each opponent searches its own to-move games in
    # its own session (openers per-opponent open_seed; greedy in one batch). The
    # candidate-first ordering keeps each leader strictly ahead of its follower
    # when the follower replays.
    while True:
        active = [g for g in all_games if not g.done]
        if not active:
            break
        if max_wall_seconds and (time.perf_counter() - started) > max_wall_seconds:
            budget_hit = True
            for g in active:
                _finalize(g)
            break
        rounds += 1
        plies_this_round = 0

        # ---- (1) Candidate pass (net A), shared forward for the greedy tail. ----
        cand_to_move = [g for g in active if not g.done and g.a_to_move()]
        cand_openers_by_opp: dict[int, list[_Game]] = {}
        cand_followers: list[_Game] = []
        cand_greedy: list[_Game] = []
        for g in cand_to_move:
            if g.plies < opening_plies and g.is_leader:
                cand_openers_by_opp.setdefault(g.opp_index, []).append(g)
            elif g.plies < opening_plies and not g.is_leader:
                cand_followers.append(g)
            else:
                cand_greedy.append(g)
        # Opening leaders: per-opponent with that opponent's own open_seed (net A
        # offset 13_000_003).
        for opp_index, openers in cand_openers_by_opp.items():
            open_seed = game_seed_base + 13_000_003 + rounds * 1_000_003
            plies_this_round += _run_candidate_opening(groups[opp_index], openers, open_seed)
        # Followers replay their leader's recorded opening line (no search).
        for g in cand_followers:
            replay = _follower_opening_action(g)
            if replay is not None:
                _replay_action(g, replay)
            else:
                _run_single(g, "A")
            plies_this_round += 1
        # Greedy: one shared candidate forward across all opponents (temp 0 ->
        # seed-independent argmax).
        if cand_greedy:
            cand_seed = game_seed_base + rounds * 1_000_003
            plies_this_round += _run_candidate_greedy(cand_greedy, cand_seed)

        # ---- (2) Opponent pass (net B), each in its own session. ----
        active2 = [g for g in all_games if not g.done]
        for grp in groups:
            to_move = [g for g in active2 if g.opp_index == grp.opp_index and not g.done and not g.a_to_move()]
            if not to_move:
                continue
            openers = [g for g in to_move if g.plies < opening_plies and g.is_leader]
            followers = [g for g in to_move if g.plies < opening_plies and not g.is_leader]
            greedy = [g for g in to_move if g.plies >= opening_plies]
            if openers:
                # Net B opening offset 19_000_003.
                open_seed = game_seed_base + 19_000_003 + rounds * 1_000_003
                plies_this_round += _run_opponent_batch(
                    grp, openers, open_seed, temperature=opening_temperature
                )
            for g in followers:
                replay = _follower_opening_action(g)
                if replay is not None:
                    _replay_action(g, replay)
                else:
                    _run_single(g, "B")
                plies_this_round += 1
            if greedy:
                # Net B greedy offset 7_000_003.
                batch_seed = game_seed_base + 7_000_003 + rounds * 1_000_003
                plies_this_round += _run_opponent_batch(grp, greedy, batch_seed, temperature=None)

        if plies_this_round == 0:
            raise RuntimeError(
                "shrimp multi-checkpoint eval made no progress in a round; "
                "aborting to avoid a hang"
            )

    # ----- Build one result dict per opponent, in play_checkpoint_match's shape. -
    elapsed = round(time.perf_counter() - started, 2)
    results: dict[str, Any] = {}
    for grp in groups:
        game_rows = [
            {
                "index": g.local_index,
                "seed": g.seed,
                "a_seat": "P0" if g.a_is_p0 else "P1",
                "status": g.status,
                "winner": g.winner,
                "plies": g.plies,
                "opening": list(g.opening),
            }
            for g in grp.games
        ]
        pairs: list[dict[str, Any]] = []
        for pair_index in sorted(grp.pair_members):
            members = grp.pair_members[pair_index]
            decided = [g for g in members if g.status == "completed"]
            a_wins_in_pair = sum(1 for g in decided if g.winner == "A")
            pairs.append(
                {
                    "pair_index": pair_index,
                    "seed": game_seed_base + pair_index,
                    "game_indices": [g.local_index for g in members],
                    "n_games": len(members),
                    "n_decided": len(decided),
                    "a_wins": a_wins_in_pair,
                    "b_wins": len(decided) - a_wins_in_pair,
                    "pentanomial_a_score": a_wins_in_pair,
                }
            )
        _hxr_stats: dict[str, int] = {}
        hxr_path = _write_eval_hxr(
            grp.games, diagnostics_dir, candidate_label, grp.label, stats=_hxr_stats
        )
        results[grp.label] = _build_match_result(
            games=game_rows,
            pairs=pairs,
            label_a=candidate_label,
            label_b=grp.label,
            meta_extra={
                "kind": "shrimp_vs_shrimp",
                "hxr_record": hxr_path,
                "hxr_games_written": _hxr_stats.get("games_written", 0),
                "ckpt_a": {"label": candidate_label, "path": str(candidate_ckpt)},
                "ckpt_b": {"label": grp.label, "path": str(grp.ckpt)},
                "games_requested": n_games_per_opponent,
                "visits": eval_visits,
                "virtual_batch_size": vbs,
                "device": cfg.device,
                "paired_openings": True,
                "opening_plies": opening_plies,
                "opening_temperature": opening_temperature,
                "game_seed_base": game_seed_base,
                "divergence_overrides_a": ov_cand,
                "divergence_overrides_b": ov_by_label[grp.label],
                "budget_hit": budget_hit,
                # Concurrency telemetry (additive — downstream consumers ignore).
                "concurrent": True,
                "multi_opponent": True,
                "n_opponents": len(groups),
                "rounds": rounds,
                "forward_batches": forward_batches,
                "candidate_forward_batches": cand_forward_batches,
                "elapsed_seconds": elapsed,
                "mcts_search_elapsed_seconds": round(mcts_search_elapsed, 2),
            },
        )
    return results


# --------------------------------------------------------------------------- #
# (2) Shrimp checkpoint vs SealBot — concurrent, UNPAIRED
# --------------------------------------------------------------------------- #


def play_sealbot_match(
    model_ckpt: str | Path,
    n_games: int,
    *,
    config: Any = None,
    label: str = "shrimp",
    sealbot_variant: str = "best",
    sealbot_time_limit: float = 0.05,
    sealbot_path: str | Path | None = None,
    visits: int | None = None,
    virtual_batch_size: int | None = None,
    opening_plies: int = DEFAULT_OPENING_PLIES,
    opening_temperature: float = DEFAULT_OPENING_TEMPERATURE,
    divergence_overrides: dict | None = None,
    diagnostics_dir: str | Path | None = None,
    max_states: int = 65_536,
    game_seed_base: int = 0,
    active_root_limit: int | None = None,
    max_wall_seconds: float = 0.0,
    build_opponent: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Play shrimp vs SealBot concurrently and return a structured result.

      * One persistent ``ShrimpMctsSession``; every game where shrimp is to
        move is searched together in one ``session.search([keys], (states,),
        ..., evaluator=...)`` multi-root call so the net batches leaves across
        all in-flight games. Search knobs: greedy after a sampled opening, no
        Dirichlet noise.
      * SealBot's turn is drained serially per game through the hexo_runner
        ``SealBotPlayer`` adapter (each game keeps its own worker). The two
        SealBot variants cannot coexist in one process.

    No CRN pairing: SealBot's minimax depth varies under load, so two SealBot
    games are not a matched comparison. This runner produces the
    shrimp-vs-SealBot edge (Wilson CI on the binomial win rate).

    Seats alternate by game index (even -> shrimp is player0). ``build_opponent``
    is an injection seam (tests fake the bot); default builds a real
    ``SealBotPlayer`` per game.
    """

    # Imported lazily so importing this module never requires the SealBot
    # checkout (the checkpoint runner above does not need it).
    from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer

    cfg = config if config is not None else parse_shrimp_config({})
    sp = cfg.selfplay
    # visits=None defaults to the self-play search budget (sp.search_visits), the
    # same default as play_checkpoint_match, not cfg.evaluation.eval_visits. An
    # explicit ``visits`` overrides.
    eval_visits = int(visits) if visits is not None else int(sp.search_visits)
    # Eval-only virtual-batch-size override, independent of
    # SelfplayConfig.virtual_batch_size. None -> self-play value.
    vbs = int(virtual_batch_size) if virtual_batch_size is not None else int(sp.virtual_batch_size)
    root_limit = int(active_root_limit) if active_root_limit is not None else sp.active_root_limit

    overrides = _resolve_eval_overrides(sp, divergence_overrides=divergence_overrides)

    sealbot_config = SealBotConfig(
        path=sealbot_path,
        variant=sealbot_variant,
        time_limit=sealbot_time_limit,
    )
    sealbot_config.validate()  # raises SealBotUnavailableError if the bot is missing

    def _make_opponent() -> Any:
        if build_opponent is not None:
            return build_opponent(sealbot_config)
        return SealBotPlayer(sealbot_config, player_id=f"sealbot-{sealbot_variant}")

    from .inference import ShrimpEvaluator  # lazy: torch only on the GPU path

    started = time.perf_counter()
    model = _load_shrimp_net(model_ckpt)
    evaluator = ShrimpEvaluator(model, device=cfg.device)
    session = _new_rust_session(max_states)

    # One in-flight game.
    class _Game:
        __slots__ = (
            "index", "seed", "hex_is_p0", "state", "opponent",
            "plies", "hex_decisions", "done", "status", "winner", "opening",
            "actions",
        )

        def __init__(self, index: int) -> None:
            self.index = index
            self.seed = game_seed_base + index
            self.hex_is_p0 = index % 2 == 0
            self.state = api.new_game(seed=self.seed)
            self.opponent = _make_opponent()
            self.plies = 0
            self.hex_decisions = 0
            self.done = False
            self.status = "truncated"
            self.winner: str | None = None  # "hex" | "sealbot" | None
            self.opening: list[int] = []
            # Full move stream (BOTH players) for the replayable .hxr record.
            self.actions: list[int] = []

        @property
        def hex_role(self) -> Any:
            return api.Player.PLAYER_0 if self.hex_is_p0 else api.Player.PLAYER_1

        @property
        def hex_role_label(self) -> str:
            return "player0" if self.hex_is_p0 else "player1"

        def hex_to_move(self) -> bool:
            return api.current_player(self.state) == self.hex_role

    games = [_Game(i) for i in range(n_games)]
    search_seed = game_seed_base + 7_000_003
    mcts_search_elapsed = 0.0
    opponent_elapsed = 0.0
    rounds = 0
    forward_batches = 0
    budget_hit = False

    def _finalize(game: _Game) -> None:
        terminal = api.terminal(game.state)
        if terminal is not None:
            game.status = "completed"
            if terminal.winner is None:
                game.winner = None
            else:
                won_label = str(terminal.winner)  # "player0" / "player1"
                game.winner = "hex" if won_label == game.hex_role_label else "sealbot"
        elif budget_hit:
            game.status = "aborted_budget"
            game.winner = None
        else:
            game.status = "truncated"
            game.winner = None
        game.done = True
        session.discard(game.index)

    def _settle(game: _Game) -> bool:
        if api.terminal(game.state) is not None or game.plies >= sp.max_game_plies:
            _finalize(game)
            return True
        return False

    try:
        while True:
            active = [g for g in games if not g.done]
            if not active:
                break
            if max_wall_seconds and (time.perf_counter() - started) > max_wall_seconds:
                budget_hit = True
                for g in active:
                    _finalize(g)
                break
            rounds += 1
            plies_this_round = 0

            # --- Batched shrimp ply across every game where hex is to move. ---
            hex_games = [g for g in active if g.hex_to_move()]
            if hex_games:
                # Cap the multi-root batch at the active-root limit; if more games
                # than the limit are simultaneously hex-to-move, search them in
                # chunks. Each chunk is one multi-root forward.
                for chunk_start in range(0, len(hex_games), root_limit):
                    chunk = hex_games[chunk_start : chunk_start + root_limit]
                    move_temperatures = [
                        opening_temperature
                        if (g.hex_decisions < opening_plies and opening_temperature > 0.0)
                        else 0.0
                        for g in chunk
                    ]
                    t0 = time.perf_counter()
                    searches = session.search(
                        [g.index for g in chunk],
                        tuple(g.state for g in chunk),
                        visits=eval_visits,
                        c_puct=sp.c_puct,
                        temperature=0.0,
                        seed=search_seed + rounds * 1_000_003,
                        evaluator=evaluator,
                        virtual_batch_size=vbs,
                        active_root_limit=root_limit,
                        widening_policy_mass=sp.widening_policy_mass,
                        widening_max_children=sp.widening_max_children,
                        widening_min_children=sp.widening_min_children,
                        fpu_reduction=sp.fpu_reduction,
                        tss_enabled=sp.tss_enabled,
                        search_parity_mode=sp.search_parity_mode,
                        move_temperatures=move_temperatures,
                        divergence_overrides=overrides,
                    )
                    mcts_search_elapsed += time.perf_counter() - t0
                    forward_batches += 1
                    if len(searches) != len(chunk):
                        raise RuntimeError(
                            f"shrimp SealBot eval search returned {len(searches)} "
                            f"results for {len(chunk)} games"
                        )
                    for g, search in zip(chunk, searches):
                        q, r = unpack_action_id(int(search["action_id"]))
                        api.apply_action(g.state, PlacementAction(AxialCoord(q=q, r=r)))
                        g.plies += 1
                        g.hex_decisions += 1
                        g.actions.append(int(search["action_id"]))
                        if len(g.opening) < opening_plies:
                            g.opening.append(int(search["action_id"]))
                        plies_this_round += 1
                        _settle(g)

            # --- SealBot turns, serially per game, fully drained per turn. ---
            for g in active:
                if g.done:
                    continue
                while not g.done and not g.hex_to_move():
                    t0 = time.perf_counter()
                    decision = g.opponent.decide(g.state)
                    opponent_elapsed += time.perf_counter() - t0
                    api.apply_action(g.state, decision.action)
                    g.plies += 1
                    coord = decision.action.coord
                    g.actions.append(pack_action_id(coord.q, coord.r))
                    if len(g.opening) < opening_plies:
                        g.opening.append(pack_action_id(coord.q, coord.r))
                    plies_this_round += 1
                    _settle(g)

            if plies_this_round == 0:
                raise RuntimeError(
                    "shrimp SealBot eval made no progress in a round; aborting to avoid a hang"
                )
    finally:
        for g in games:
            try:
                g.opponent.close()
            except Exception:
                pass

    # Persist the SealBot games as a replayable .hxr (dashboard "evaluation"
    # source). Best-effort / fail-soft. The writer is net-A-centric (expects
    # .a_is_p0 and .winner in {"A","B",None}), so the SealBot _Game (shrimp is
    # net A, winner "hex"/"sealbot") is adapted onto that shape here.
    from types import SimpleNamespace

    _hxr_stats: dict[str, int] = {}
    _hxr_games = [
        SimpleNamespace(
            actions=g.actions,
            a_is_p0=g.hex_is_p0,
            seed=g.seed,
            index=g.index,
            plies=g.plies,
            winner=("A" if g.winner == "hex" else ("B" if g.winner == "sealbot" else None)),
        )
        for g in games
    ]
    hxr_path = _write_eval_hxr(
        _hxr_games, diagnostics_dir, label, f"SealBot {sealbot_variant}", stats=_hxr_stats
    )

    # Re-key game rows to the shrimp-vs-X result shape. _build_match_result is
    # net-A-centric, so map shrimp -> "A", sealbot -> "B".
    game_rows = [
        {
            "index": g.index,
            "seed": g.seed,
            "a_seat": "P0" if g.hex_is_p0 else "P1",
            "status": g.status,
            "winner": (
                "A" if g.winner == "hex" else ("B" if g.winner == "sealbot" else None)
            ),
            "plies": g.plies,
            "opening": list(g.opening),
        }
        for g in games
    ]
    result = _build_match_result(
        games=game_rows,
        pairs=None,  # SealBot games are unpaired
        label_a=label,
        label_b=f"SealBot {sealbot_variant}",
        meta_extra={
            "kind": "shrimp_vs_sealbot",
            "ckpt": {"label": label, "path": str(model_ckpt)},
            "sealbot": {"variant": sealbot_variant, "time_limit": sealbot_time_limit},
            "games_requested": n_games,
            "visits": eval_visits,
            "virtual_batch_size": vbs,
            "device": cfg.device,
            "opening_plies": opening_plies,
            "opening_temperature": opening_temperature,
            "game_seed_base": game_seed_base,
            "divergence_overrides": overrides,
            "budget_hit": budget_hit,
            "rounds": rounds,
            "forward_batches": forward_batches,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "mcts_search_elapsed_seconds": round(mcts_search_elapsed, 2),
            "opponent_elapsed_seconds": round(opponent_elapsed, 2),
            "hxr_record": hxr_path,
            "hxr_games_written": _hxr_stats.get("games_written", 0),
        },
    )
    return result


# --------------------------------------------------------------------------- #
# Result-dict builder (shared by both runners; arena shape + pentanomial)
# --------------------------------------------------------------------------- #


def _build_match_result(
    *,
    games: list[dict[str, Any]],
    pairs: list[dict[str, Any]] | None,
    label_a: str,
    label_b: str,
    meta_extra: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the standalone-arena result dict (meta/score/by-seat/lengths/
    opening-dedup/games) plus a ``pentanomial`` block for paired matches.

    ``games`` rows are net-A-centric: ``winner`` in {"A", "B", None}. No draws in
    hexo, so ``winner is None`` means the game was truncated/aborted (not
    decided), and such games are EXCLUDED from win rates and CIs but reported in
    the status counts.
    """

    completed = [g for g in games if g["status"] == "completed"]
    a_wins = sum(1 for g in completed if g["winner"] == "A")
    b_wins = sum(1 for g in completed if g["winner"] == "B")
    decided = a_wins + b_wins
    lo, hi = _wilson_ci(a_wins, decided)

    p0_games = [g for g in completed if g["a_seat"] == "P0"]
    p1_games = [g for g in completed if g["a_seat"] == "P1"]

    def _seat_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "a_wins": sum(1 for g in rows if g["winner"] == "A"),
            "b_wins": sum(1 for g in rows if g["winner"] == "B"),
        }

    lengths_all = [g["plies"] for g in completed]
    openings = [tuple(g.get("opening") or ()) for g in games]

    score: dict[str, Any] = {
        "completed": len(completed),
        "truncated": sum(1 for g in games if g["status"] == "truncated"),
        "aborted_budget": sum(1 for g in games if g["status"] == "aborted_budget"),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "decided": decided,
        "a_winrate_decided": round(a_wins / decided, 4) if decided else None,
        # 95% Wilson on the binomial win rate, per-game (unit = game). For paired
        # matches, use the pair-level SE in the ``pentanomial`` block instead;
        # this per-game CI does not account for within-pair correlation.
        "a_winrate_ci95": [round(lo, 4), round(hi, 4)] if decided else None,
        "by_seat": {"A_as_P0": _seat_block(p0_games), "A_as_P1": _seat_block(p1_games)},
    }

    result: dict[str, Any] = {
        "meta": {"label_a": label_a, "label_b": label_b, **meta_extra},
        "score": score,
        "game_lengths": {
            "overall": _length_stats(lengths_all),
            "A_as_P0": _length_stats([g["plies"] for g in p0_games]),
            "A_as_P1": _length_stats([g["plies"] for g in p1_games]),
            "a_won": _length_stats([g["plies"] for g in completed if g["winner"] == "A"]),
            "b_won": _length_stats([g["plies"] for g in completed if g["winner"] == "B"]),
        },
        "opening_dedup": _opening_dedup(openings),
        "games": games,
    }
    result["pentanomial"] = _pentanomial_block(pairs) if pairs is not None else None
    return result


def _pentanomial_block(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair-level pentanomial summary and the pair-unit win-rate SE.

    Reports the pentanomial counts (over 2-game pairs, by net-A score in
    {0, 1, 2}) and the pair-level standard error (N_pairs units). The raw
    per-pair scores are surfaced in ``pairs`` for downstream inference.

    For a complete 2-game pair the per-pair net-A score ``s in {0, 1, 2}`` is
    the number of games net A won; the pair-level statistic is ``s / 2 in
    {0, 0.5, 1}`` (an "even" pair = 1 each = 0.5). The mean and standard error of
    that statistic are computed across pairs (each pair one draw). Singleton /
    partial pairs (only at an odd ``n_games`` tail) are reported but, having
    ``n_games < 2``, are excluded from the 0/1/2 histogram; their lone game is
    folded into the pair statistic at ``s / n_decided``.
    """

    full_pairs = [p for p in pairs if p["n_games"] == 2 and p["n_decided"] == 2]
    pent = {0: 0, 1: 0, 2: 0}  # net-A wins among the pair's 2 decided games
    for p in full_pairs:
        pent[p["pentanomial_a_score"]] += 1

    # Pair statistic in [0, 1]: per-pair net-A win fraction over DECIDED games.
    # Pairs with zero decided games (both truncated) carry no information and are
    # dropped from the mean/SE.
    stats = [
        p["a_wins"] / p["n_decided"]
        for p in pairs
        if p["n_decided"] > 0
    ]
    n = len(stats)
    mean = sum(stats) / n if n else None
    if n > 1 and mean is not None:
        var = sum((x - mean) ** 2 for x in stats) / (n - 1)  # sample variance
        se = (var / n) ** 0.5  # SE of the mean, N_pairs units
    else:
        var = None
        se = None

    return {
        "n_pairs": len(pairs),
        "n_full_pairs": len(full_pairs),
        "n_informative_pairs": n,
        # Histogram over full (2-decided-game) pairs, keyed by net-A wins.
        "histogram_a_wins": {"0": pent[0], "1": pent[1], "2": pent[2]},
        "pair_winrate_mean": round(mean, 4) if mean is not None else None,
        # Pair-level SE (N_pairs units), distinct from the per-game Wilson
        # half-width in ``score``.
        "pair_winrate_se": round(se, 4) if se is not None else None,
        "pair_winrate_sample_variance": round(var, 6) if var is not None else None,
        "pairs": pairs,
    }


def _wilson_ci(wins: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval.

    Pass independent counts. For paired matches the unit is the pair, not the
    game — a paired CI uses the pair-level SE in the pentanomial block, not this
    function on per-game counts.
    """
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))
