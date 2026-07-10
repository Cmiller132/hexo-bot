"""Game-running layer for the hexfield multi-stage strength evaluation.

This module only plays games and returns structured result dicts; it does not
gate, promote, halt, or mutate a training run. The statistical verdict layer
(SPRT screen, pentanomial/Wilson CIs, the rolling Bradley-Terry pool) lives in
sibling modules and consumes these results.

Runners return the result-dict shape of the standalone arena
(scripts/_wf_h2h2_arena.py): ``meta`` / ``score`` (with ``by_seat``) /
``game_lengths`` / ``opening_dedup`` / per-game rows, plus a pair-level
``pentanomial`` block for paired matches.

play_checkpoint_match(model_a_ckpt, model_b_ckpt, ...)
    Hexfield-vs-hexfield, concurrent and paired. Both players are hexfield nets.
    Each round batches whichever side is to move through that net's
    ``HexfieldMctsSession.search`` multi-root call (cross-game leaf batching):
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
    Hexfield-vs-SealBot. Every game where hexfield is to move is searched
    together in one ``HexfieldMctsSession.search`` multi-root call (cross-game
    leaf batching); SealBot's moves are drained serially per game through the
    hexo_runner SealBot adapter. No CRN pairing: SealBot's minimax depth varies
    under load, so its games are not a matched comparison.

CRN / shared-opening note: ``hexo_engine.api.new_game(seed=...)`` does not
randomize the opening — the engine is deterministic and the first move is the
forced centre stone (api.py docstring). Opening diversity comes from the MCTS
temperature sampling at the root: the first ``opening_plies`` plies sample the
move from the visit distribution using the per-search ``seed``.

CRN under batching: the lockstep ``search`` builds a tree by deterministic PUCT
(no RNG in leaf selection). Its randomness is (a) optional root Dirichlet noise,
not used in eval, and (b) the final move selection. At ``temperature == 0`` (the
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
    ML_AUTO_DISABLED_FLAG,
    build_divergence_overrides,
    build_eval_search_kwargs,
    parse_hexfield_config,
)
from .eval_driver import (
    EvalGame,
    HexfieldCheckpointAdapter,
    SealBotAdapter,
    StrixAdapter,
    _CountingSession,
    _HexTelemetry,
    finalize,
    follower_opening_action,
    play_eval_match,
    record_move,
    run_hexfield_ply,
    settle,
)
from .geometry import unpack_action_id

# Logger for the eval .hxr writer; a 0-record write is logged as a warning.
_EVAL_LOG = logging.getLogger("hexfield.eval")

# torch / HexfieldEvaluator / HexfieldNet are imported lazily inside the
# checkpoint-loading paths so this module is importable on a CPU-only host
# without torch (the concurrent loop can be unit-tested through the
# ``build_evaluators`` / ``make_session`` seams with a numpy stub evaluator).
# Only ``_load_hexfield_net`` and the non-stub evaluator branches touch them.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model import HexfieldNet

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

# Hexfield has no draws. A max_plies truncation is the only non-decisive outcome
# and is reported separately as a "truncated" game, never as a draw.


# --------------------------------------------------------------------------- #
# Shared helpers (result-dict construction; numpy-free, importable on CPU)
# --------------------------------------------------------------------------- #


def _resolve_eval_budget(
    sp: Any,
    *,
    visits: int | None,
    virtual_batch_size: int | None,
    active_root_limit: int | None,
) -> tuple[int, int, int]:
    """Resolve the (visits, virtual_batch_size, active_root_limit) eval budget.

    The single fallback point for the four arenas' identical budget resolution:
    ``visits=None`` defaults to the self-play search budget (``sp.search_visits``),
    NOT ``cfg.evaluation.eval_visits``; ``virtual_batch_size=None`` and
    ``active_root_limit=None`` default to the corresponding ``sp.*`` fields. An
    explicit value overrides. Preserves the standalone-arena behavior."""
    eval_visits = int(visits) if visits is not None else int(sp.search_visits)
    vbs = int(virtual_batch_size) if virtual_batch_size is not None else int(sp.virtual_batch_size)
    root_limit = int(active_root_limit) if active_root_limit is not None else int(sp.active_root_limit)
    return eval_visits, vbs, root_limit


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
    """Construct a native ``HexfieldMctsSession`` (the default session factory).

    Raises a clear error when the maturin-built extension is unavailable rather
    than the bare ``AttributeError`` a ``None`` module would give. Tests that run
    without the .so inject a fake session via ``make_session`` and never hit this.
    """
    if _rust is None:
        raise RuntimeError(
            "hexfield._rust (the MCTS extension) is unavailable; build the .so or "
            "inject a session factory (make_session=) for a CPU-only run"
        )
    return _rust.HexfieldMctsSession(max_states=max_states)


def _check_eq_arch_matches_import(meta: dict | None, path: Path) -> None:
    """Fail fast when a checkpoint's equivariant arch cannot be rebuilt here.

    GROUP_ORDER / C_ORBIT (and, through C = GROUP_ORDER * C_ORBIT, the trunk
    width) are read from HEXFIELD_EQ_* env ONCE at import and baked into the
    weight tie / head reads; constructor kwargs cannot override them (see
    model.infer_net_kwargs_from_state_dict's docstring). A foreign-arch
    EQUIVARIANT checkpoint therefore fails strict load with an opaque
    size-mismatch — this guard raises the true error instead, mirroring the
    dashboard worker's ``_check_eq_meta_matches_import`` (hexo_frontend
    debug_infer.py). Meta-less (pre-arch_meta) checkpoints skip the guard and
    rely on strict load. Passthrough checkpoints (group_order == 1) keep the
    cross-width rebuild: only group_order itself is enforced for them.
    """
    from .constants import C_ORBIT, CHANNELS, GROUP_ORDER

    meta = meta or {}
    ckpt_group_order = meta.get("group_order")
    checks: list[tuple[str, object, int, str]] = [
        ("group_order", ckpt_group_order, GROUP_ORDER, "HEXFIELD_EQ_GROUP_ORDER"),
    ]
    if ckpt_group_order is not None and int(ckpt_group_order) > 1:
        checks += [
            ("c_orbit", meta.get("c_orbit"), C_ORBIT, "HEXFIELD_EQ_C_ORBIT"),
            ("channels", meta.get("channels"), CHANNELS, "HEXFIELD_EQ_CHANNELS"),
        ]
    for key, want, imported, env_var in checks:
        if want is not None and int(want) != int(imported):
            raise RuntimeError(
                f"hexfield_eq checkpoint {path} was trained with {key}={int(want)} "
                f"but this process imported hexfield_eq with {key}={int(imported)} "
                f"(HEXFIELD_EQ_* env is read once at import); relaunch with "
                f"{env_var}={int(want)} (and the checkpoint's other HEXFIELD_EQ_* "
                "arch env) to evaluate this checkpoint"
            )


def _load_hexfield_net(checkpoint: str | Path) -> HexfieldNet:
    """Strict-load a hexfield checkpoint into a fresh HexfieldNet.

    Loads ``payload["model"]`` with ``strict=True`` so a value-/moves-left-head
    mismatch raises rather than keeping a random head. A checkpoint whose meta
    declares an equivariant arch this process cannot rebuild (import-frozen
    GROUP_ORDER / C_ORBIT / CHANNELS) is rejected up front with a clear
    relaunch instruction instead of an opaque strict-load size mismatch.
    """
    import torch  # lazy: keep the module importable on CPU hosts without torch

    from .model import HexfieldNet

    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"hexfield checkpoint is not a readable file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError(f"hexfield checkpoint payload has no 'model' state: {path}")
    sd = payload["model"]
    # Build the net at the checkpoint's OWN arch (head count, trunk layout —
    # and, for passthrough builds, width), not the process-global env
    # constants, so a foreign-arch anchor loads instead of shape-mismatching.
    # Undeterminable fields fall back to the env defaults.
    from .model import infer_net_kwargs_from_state_dict

    # Meta-first (checklist B2 / spec D-S32): the persisted arch_meta is the
    # authoritative self-description. ray_blockers in particular is META-ONLY
    # (a mask-build variant with no state-dict trace), so an arm-4c
    # (RAY_BLOCKERS=0) checkpoint loaded without meta would silently rebuild
    # with this process's env default (blockers ON) — a semantics flip, not a
    # shape error. Checkpoints without meta fall back to shape inference as
    # before.
    _check_eq_arch_matches_import(payload.get("meta"), path)
    net_kwargs = infer_net_kwargs_from_state_dict(sd, payload.get("meta"))
    model = HexfieldNet(**net_kwargs)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _resolve_eval_overrides(
    sp: Any,
    *,
    diagnostics_dir: str | Path | None,
    divergence_overrides: dict | None,
) -> dict:
    """The divergence overrides the arena searches with.

    Default: mirror self-play, including the heal-gate auto-disable flag
    (``ml_auto_disabled.flag`` in the run's diagnostics dir). An explicit
    ``divergence_overrides`` takes precedence.
    """
    if divergence_overrides is not None:
        return divergence_overrides
    disabled = False
    if diagnostics_dir is not None:
        disabled = (Path(diagnostics_dir) / ML_AUTO_DISABLED_FLAG).exists()
    return build_divergence_overrides(sp, disabled=disabled)


def puct_eval_overrides(sp: Any, *, diagnostics_dir: str | Path | None = None) -> dict:
    """The pre-Gumbel production search profile: the default (self-play) eval
    overrides with the four Gumbel mechanisms forced off.

    Used for foreign anchors that were trained under plain PUCT search — they
    are evaluated with the searcher they were tuned for, while this run's
    lineage keeps the self-play (Gumbel) profile. A no-op difference when the
    run's self-play profile has the Gumbel flags off already."""
    ov = dict(
        _resolve_eval_overrides(
            sp, diagnostics_dir=diagnostics_dir, divergence_overrides=None
        )
    )
    for key in (
        "gumbel_target",
        "gumbel_root",
        "gumbel_sequential_halving",
        "gumbel_nonroot_select",
    ):
        ov[key] = False
    return ov


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
                            message="hexfield eval game reached max plies",
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
# (1) Hexfield checkpoint vs hexfield checkpoint — PAIRED (CRN) games
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
    time_ms_per_move: float | None = None,
) -> dict[str, Any]:
    """Play model A vs model B concurrently and return a structured pentanomial
    result.

    ``time_ms_per_move`` (spec D-S35, the equal-TIME A/B leg): a per-move
    wall-clock budget; ``None`` falls back to the config knob
    ``multi_stage_eval.eval_time_budget_ms`` (0.0 = off). When active, EACH
    net's visit budget is independently re-calibrated from its own measured
    probe-search latency (``eval_driver.calibrate_time_budget_visits``), so an
    architecturally slower net (e.g. an L-layout arm on the flex path) plays
    proportionally fewer visits — wall-clock-fair instead of visit-fair. The
    calibration records land in ``meta.time_calibration_{a,b}``.

    Both players are hexfield nets. The runner keeps two persistent sessions and
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

    cfg = config if config is not None else parse_hexfield_config({})
    sp = cfg.selfplay

    # Symmetric divergence overrides by default; an explicit per-net override
    # drives a search-change A/B. The override follows the searching net: ``ov_a``
    # whenever net A is to move, ``ov_b`` whenever net B is to move, independent of
    # which engine seat each net holds. When ``divergence_overrides_b`` is None,
    # net B shares net A's resolved dict (identity), matching the pre-refactor
    # runner.
    ov_a = _resolve_eval_overrides(
        sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides_a
    )
    ov_b = (
        ov_a
        if divergence_overrides_b is None
        else _resolve_eval_overrides(
            sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides_b
        )
    )

    # Split the single ``build_evaluators() -> (eval_a, eval_b)`` seam into the
    # driver's candidate builder + the adapter's opponent builder, called once
    # (preserves the CPU-test contract). Absent the seam both sides load from the
    # checkpoints on the GPU path.
    if build_evaluators is not None:
        _eval_a, _eval_b = build_evaluators()
        build_candidate_evaluator = lambda: _eval_a  # noqa: E731
        opponent_build_evaluator = lambda: _eval_b  # noqa: E731
    else:
        build_candidate_evaluator = None
        opponent_build_evaluator = None

    # Net B is a second hexfield session with its own opening/greedy seed streams.
    opponent = HexfieldCheckpointAdapter(
        model_b_ckpt,
        config=cfg,
        label=label_b,
        overrides_b=ov_b,
        make_session=make_session,
        max_states=max_states,
        visits=visits,
        virtual_batch_size=virtual_batch_size,
        active_root_limit=active_root_limit,
        paired_openings=paired_openings,
        batch_openings=batch_openings,
        build_evaluator=opponent_build_evaluator,
        time_ms_per_move=time_ms_per_move,
    )

    def _meta_extra(games, tel) -> dict[str, Any]:
        # Persist the games as a replayable .hxr (dashboard "evaluation" source);
        # _write_eval_hxr is fully fail-soft. ``games`` are net-A-centric EvalGame
        # records (.a_is_p0 / .winner in {"A","B",None}), the writer's shape.
        _hxr_stats: dict[str, int] = {}
        hxr_path = _write_eval_hxr(games, diagnostics_dir, label_a, label_b, stats=_hxr_stats)
        # Equal-time leg telemetry (additive; absent on fixed-visit runs):
        # per-net calibration records — net A's from the driver telemetry, net
        # B's from the adapter. "visits" below is net A's PLAYED budget.
        _time_meta = {}
        if tel.time_calibration is not None or opponent.time_calibration is not None:
            _time_meta = {
                "time_calibration_a": tel.time_calibration,
                "time_calibration_b": opponent.time_calibration,
            }
        return {
            "kind": "hexfield_vs_hexfield",
            "hxr_record": hxr_path,
            "hxr_games_written": _hxr_stats.get("games_written", 0),
            "ckpt_a": {"label": label_a, "path": str(model_a_ckpt)},
            "ckpt_b": {"label": label_b, "path": str(model_b_ckpt)},
            "games_requested": n_games,
            "visits": tel.eval_visits,
            **_time_meta,
            "virtual_batch_size": tel.virtual_batch_size,
            "device": cfg.device,
            "paired_openings": paired_openings,
            "opening_plies": opening_plies,
            "opening_temperature": opening_temperature,
            "game_seed_base": game_seed_base,
            "divergence_overrides_a": ov_a,
            "divergence_overrides_b": ov_b,
            "budget_hit": tel.budget_hit,
            # Concurrency telemetry (additive — downstream consumers ignore these).
            "concurrent": True,
            "batch_openings": bool(batch_openings),
            "rounds": tel.rounds,
            # forward_batches / mcts time cover BOTH nets' hexfield searches (net A
            # via the driver, net B via the adapter), matching the old single
            # combined counter.
            "forward_batches": tel.forward_batches + opponent.telemetry.forward_batches,
            "elapsed_seconds": round(tel.elapsed_seconds, 2),
            "mcts_search_elapsed_seconds": round(
                tel.mcts_search_elapsed + opponent.telemetry.mcts_search_elapsed, 2
            ),
        }

    # The driver (eval_driver) references the engine only through ITS module-level
    # ``api`` name; sync it to this module's ``api`` (which tests / the golden
    # harness monkeypatch) for the duration of the call, then restore. This lets
    # the shared driver use the same (possibly faked) engine without a second
    # patch site, with no cross-test leakage.
    from . import eval_driver as _eval_driver

    _saved_api = _eval_driver.api
    _eval_driver.api = api
    try:
        return play_eval_match(
            model_a_ckpt,
            opponent,
            n_games,
            config=cfg,
            label_a=label_a,
            label_b=label_b,
            meta_extra_fn=_meta_extra,
            paired_openings=paired_openings,
            visits=visits,
            virtual_batch_size=virtual_batch_size,
            active_root_limit=active_root_limit,
            opening_plies=opening_plies,
            opening_temperature=opening_temperature,
            batch_openings=batch_openings,
            divergence_overrides=ov_a,
            diagnostics_dir=diagnostics_dir,
            game_seed_base=game_seed_base,
            max_wall_seconds=max_wall_seconds,
            max_states=max_states,
            side_seed_offset=_SIDE_SEED_OFFSET,
            build_candidate_evaluator=build_candidate_evaluator,
            make_session=make_session,
            time_ms_per_move=time_ms_per_move,
        )
    finally:
        _eval_driver.api = _saved_api


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

    Game-key namespacing: ``HexfieldMctsSession.search`` keys trees by game_key in
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

    cfg = config if config is not None else parse_hexfield_config({})
    sp = cfg.selfplay
    eval_visits, vbs, root_limit = _resolve_eval_budget(
        sp, visits=visits, virtual_batch_size=virtual_batch_size, active_root_limit=active_root_limit
    )
    new_session = make_session if make_session is not None else (
        lambda: _new_rust_session(max_states)
    )

    started = time.perf_counter()

    # ----- Evaluators: ONE candidate (net A everywhere) + ONE per opponent. -----
    if build_candidate_evaluator is not None:
        cand_eval = build_candidate_evaluator()
    else:
        from .inference import build_serve_evaluator  # lazy: torch only on the GPU path

        cand_eval = build_serve_evaluator(_load_hexfield_net(candidate_ckpt), cfg, role="eval")

    # ov follows the searching net: the candidate (net A) searches with
    # ``ov_cand``, an opponent with ``ov_opp``. Symmetric by default.
    ov_cand = _resolve_eval_overrides(
        sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides_candidate
    )
    ov_opp = (
        ov_cand
        if divergence_overrides_opponent is None
        else _resolve_eval_overrides(
            sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides_opponent
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
                diagnostics_dir=diagnostics_dir,
                divergence_overrides=_by_label[label],
            )
            if label in _by_label
            else ov_opp
        )
        for label, _ckpt in opponents
    }

    common = build_eval_search_kwargs(
        sp, visits=eval_visits, virtual_batch_size=vbs, active_root_limit=root_limit
    )

    # KEY_STRIDE namespaces candidate-session game keys per opponent so two
    # opponents' games never share a candidate tree (n_games_per_opponent << this
    # stride).
    KEY_STRIDE = 1_000_000

    # One opponent group per (label, ckpt). Each group builds the same game layout
    # a standalone play_checkpoint_match would for that opponent. Games are the
    # shared ``eval_driver.EvalGame`` record: ``index`` is this opponent's session
    # key (the old per-group ``local_index``); ``opp_index`` / ``cand_key`` namespace
    # the candidate session so its trees never collide across opponents.
    class _Group:
        __slots__ = ("opp_index", "label", "ckpt", "session", "evaluator",
                     "games", "pair_members")

        def __init__(self, opp_index, label, ckpt, session, evaluator):
            self.opp_index = opp_index
            self.label = label
            self.ckpt = ckpt
            self.session = session
            self.evaluator = evaluator
            self.games: list[EvalGame] = []
            self.pair_members: dict[int, list[EvalGame]] = {}

    groups: list[_Group] = []
    cand_session = new_session()
    for opp_index, (label, ckpt) in enumerate(opponents):
        if build_opponent_evaluator is not None:
            opp_eval = build_opponent_evaluator(label, ckpt)
        else:
            from .inference import build_serve_evaluator  # lazy: torch only on GPU path

            opp_eval = build_serve_evaluator(_load_hexfield_net(ckpt), cfg, role="eval")
        grp = _Group(opp_index, label, ckpt, new_session(), opp_eval)
        # Build CRN pairs identically to play_checkpoint_match (paired_openings).
        n_pairs = (n_games_per_opponent + 1) // 2
        base = opp_index * KEY_STRIDE
        for pair_index in range(n_pairs):
            pair_seed = game_seed_base + pair_index  # shared CRN seed (both seats)
            idx0 = pair_index * 2
            g0 = EvalGame(
                index=idx0, pair_index=pair_index, a_is_p0=True, seed=pair_seed,
                state=api.new_game(), opp_index=opp_index, cand_key=base + idx0,
            )
            grp.games.append(g0)
            grp.pair_members.setdefault(pair_index, []).append(g0)
            if idx0 + 1 < n_games_per_opponent:
                g1 = EvalGame(
                    index=idx0 + 1, pair_index=pair_index, a_is_p0=False, seed=pair_seed,
                    state=api.new_game(), opp_index=opp_index, cand_key=base + idx0 + 1,
                )
                g1.is_leader = False
                g1.leader = g0
                grp.games.append(g1)
                grp.pair_members[pair_index].append(g1)
        groups.append(grp)

    all_games: list[EvalGame] = [g for grp in groups for g in grp.games]

    # Telemetry: candidate-session searches are counted separately from opponent-
    # session searches so ``candidate_forward_batches`` and the combined
    # ``forward_batches`` both reproduce the pre-refactor per-call counters. Every
    # session is wrapped in a ``_CountingSession`` so ``run_hexfield_ply`` (which
    # does not itself count) records exactly one forward per multi-root chunk, as
    # the old inline search loops did. All opponents share one ``opp_tel`` (their
    # forwards sum into the single combined opponent counter).
    cand_tel = _HexTelemetry()
    opp_tel = _HexTelemetry()
    cand_counting = _CountingSession(cand_session, cand_tel)
    opp_counting = {grp.opp_index: _CountingSession(grp.session, opp_tel) for grp in groups}

    budget_hit = False
    rounds = 0

    def _on_discard(g: EvalGame) -> None:
        # Candidate tree keyed by the global cand_key; opponent tree by the local
        # per-group index (== EvalGame.index). Both discarded at game end so trees
        # never leak across opponent groups.
        cand_session.discard(g.cand_key)
        groups[g.opp_index].session.discard(g.index)

    def _finalize(g: EvalGame) -> None:
        finalize(g, budget_hit=budget_hit, on_discard=_on_discard)

    def _settle(g: EvalGame) -> None:
        settle(g, sp.max_game_plies, budget_hit=budget_hit, on_discard=_on_discard)

    def _temp(g: EvalGame) -> float:
        return opening_temperature if (g.plies < opening_plies and opening_temperature > 0.0) else 0.0

    def _record(g: EvalGame, action_id: int, *, is_a: bool) -> None:
        record_move(g, action_id, is_a=is_a, opening_plies=opening_plies, settle_fn=_settle)

    def _replay_action(g: EvalGame, action_id: int) -> None:
        # Follower opening replay (the leader's recorded action). ``is_a`` follows
        # whose turn it is (candidate pass -> net A, opponent pass -> net B); it only
        # bumps the unused a_decisions counter, so it never affects the result.
        _record(g, action_id, is_a=g.a_to_move())

    def _follower_opening_action(g: EvalGame) -> int | None:
        return follower_opening_action(g)

    def _run_candidate_greedy(batch: list[EvalGame], seed: int) -> int:
        """One shared candidate-session multi-root search over the greedy
        candidate-to-move games across all opponents (chunked at root_limit). At
        temperature 0 the move is a deterministic seed-independent argmax."""
        return run_hexfield_ply(
            cand_counting,
            cand_eval,
            batch,
            search_kwargs=common,
            root_limit=root_limit,
            overrides=ov_cand,
            seed=seed,
            move_temperature_fn=lambda g: 0.0,
            record_fn=_record,
            is_a=True,
            key_fn=lambda g: g.cand_key,
        )

    def _run_candidate_opening(grp: "_Group", openers: list[EvalGame], seed: int) -> int:
        """Per-opponent candidate opening-leader batch. Uses the candidate session
        and evaluator but this opponent's own ``open_seed`` (the caller folds
        ``grp.opp_index`` into the seed), so each opponent group samples a
        distinct opening stream rather than every group replaying identical
        candidate openings."""
        return run_hexfield_ply(
            cand_counting,
            cand_eval,
            openers,
            search_kwargs=common,
            root_limit=root_limit,
            overrides=ov_cand,
            seed=seed,
            move_temperature_fn=lambda g: opening_temperature,
            record_fn=_record,
            is_a=True,
            key_fn=lambda g: g.cand_key,
        )

    def _run_opponent_batch(grp: "_Group", batch: list[EvalGame], seed: int,
                            *, temperature: float | None) -> int:
        """One multi-root search for the opponent (net B) to-move games in this
        opponent's session (chunked at root_limit). ``temperature`` None -> per-game
        temperature via ``_temp``; a float pins it (opening leaders)."""
        temp_fn = (lambda g: temperature) if temperature is not None else _temp
        return run_hexfield_ply(
            opp_counting[grp.opp_index],
            grp.evaluator,
            batch,
            search_kwargs=common,
            root_limit=root_limit,
            overrides=ov_by_label[grp.label],
            seed=seed,
            move_temperature_fn=temp_fn,
            record_fn=_record,
            is_a=False,
            key_fn=lambda g: g.index,
        )

    def _run_single(g: EvalGame, net: str) -> int:
        """Single-root follower fallback (leader ended mid-opening). Uses seed
        ``g.seed * 5003 + g.plies`` and the session/evaluator for ``net``."""
        if net == "A":
            return run_hexfield_ply(
                cand_counting,
                cand_eval,
                [g],
                search_kwargs=common,
                root_limit=root_limit,
                overrides=ov_cand,
                seed=g.seed * 5003 + g.plies,
                move_temperature_fn=_temp,
                record_fn=_record,
                is_a=True,
                key_fn=lambda gg: gg.cand_key,
            )
        grp = groups[g.opp_index]
        return run_hexfield_ply(
            opp_counting[g.opp_index],
            grp.evaluator,
            [g],
            search_kwargs=common,
            root_limit=root_limit,
            overrides=ov_by_label[grp.label],
            seed=g.seed * 5003 + g.plies,
            move_temperature_fn=_temp,
            record_fn=_record,
            is_a=False,
            key_fn=lambda gg: gg.index,
        )

    # The shared eval_driver primitives (EvalGame.a_to_move / finalize / settle /
    # record_move) reference the engine only through eval_driver's module-level
    # ``api``; sync it to this module's ``api`` (tests / the golden harness
    # monkeypatch it) for the duration of the loop, then restore. This lets the
    # cross-opponent loop reuse the shared primitives without a second patch site.
    from . import eval_driver as _eval_driver

    _saved_api = _eval_driver.api
    _eval_driver.api = api
    try:
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
            cand_openers_by_opp: dict[int, list[EvalGame]] = {}
            cand_followers: list[EvalGame] = []
            cand_greedy: list[EvalGame] = []
            for g in cand_to_move:
                if g.plies < opening_plies and g.is_leader:
                    cand_openers_by_opp.setdefault(g.opp_index, []).append(g)
                elif g.plies < opening_plies and not g.is_leader:
                    cand_followers.append(g)
                else:
                    cand_greedy.append(g)
            # Opening leaders: per-opponent with that opponent's own open_seed (net A
            # offset 13_000_003, plus a per-opponent 23_000_009 stride so candidate
            # openings are not correlated across opponent groups).
            for opp_index, openers in cand_openers_by_opp.items():
                open_seed = (
                    game_seed_base + 13_000_003
                    + opp_index * 23_000_009
                    + rounds * 1_000_003
                )
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
                    "hexfield multi-checkpoint eval made no progress in a round; "
                    "aborting to avoid a hang"
                )

        # ----- Build one result dict per opponent, in play_checkpoint_match's shape. -
        # forward_batches / candidate_forward_batches / mcts wall time come from the
        # _CountingSession telemetry (candidate own + all opponents), reproducing the
        # pre-refactor combined counters exactly.
        elapsed = round(time.perf_counter() - started, 2)
        forward_batches = cand_tel.forward_batches + opp_tel.forward_batches
        cand_forward_batches = cand_tel.forward_batches
        mcts_search_elapsed = cand_tel.mcts_search_elapsed + opp_tel.mcts_search_elapsed
        results: dict[str, Any] = {}
        for grp in groups:
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
                        "game_indices": [g.index for g in members],
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
                    "kind": "hexfield_vs_hexfield",
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
    finally:
        _eval_driver.api = _saved_api


# --------------------------------------------------------------------------- #
# (2) Hexfield checkpoint vs SealBot — concurrent, UNPAIRED
# --------------------------------------------------------------------------- #


def play_sealbot_match(
    model_ckpt: str | Path,
    n_games: int,
    *,
    config: Any = None,
    label: str = "hexfield",
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
    """Play hexfield vs SealBot concurrently and return a structured result.

      * One persistent ``HexfieldMctsSession``; every game where hexfield is to
        move is searched together in one ``session.search([keys], (states,),
        ..., evaluator=...)`` multi-root call so the net batches leaves across
        all in-flight games. Search knobs: greedy after a sampled opening, no
        Dirichlet noise.
      * SealBot's turn is drained serially per game through the hexo_runner
        ``SealBotPlayer`` adapter (each game keeps its own worker). The two
        SealBot variants cannot coexist in one process.

    No CRN pairing: SealBot's minimax depth varies under load, so two SealBot
    games are not a matched comparison. This runner produces the
    hexfield-vs-SealBot edge (Wilson CI on the binomial win rate).

    Seats alternate by game index (even -> hexfield is player0). ``build_opponent``
    is an injection seam (tests fake the bot); default builds a real
    ``SealBotPlayer`` per game.

    Thin wrapper over the central ``eval_driver.play_eval_match`` engine in UNPAIRED
    mode (``paired_openings=False``, ``pentanomial=None``): net A is the hexfield
    candidate (driven by the shared multi-root ``session.search`` pass), net B is
    the :class:`SealBotAdapter`. The net-A pass keeps SealBot's 7M greedy seed
    offset and gates the opening temperature on the hexfield move count
    (``a_decisions``, not total plies), so the result dict is byte-identical to the
    pre-refactor runner.
    """

    # Imported lazily so importing this module never requires the SealBot
    # checkout (the checkpoint runner above does not need it).
    from hexo_runner.adapters.sealbot import SealBotConfig, SealBotPlayer

    cfg = config if config is not None else parse_hexfield_config({})
    sp = cfg.selfplay

    # Net A's (and, echoed into meta, the match's) divergence overrides. An
    # explicit dict is returned as-is by _resolve_eval_overrides, so passing this
    # resolved dict on to the driver is idempotent.
    overrides = _resolve_eval_overrides(
        sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides
    )

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

    # Net B is SealBot: one persistent minimax worker per game, drained serially.
    opponent = SealBotAdapter(
        label=f"SealBot {sealbot_variant}", make_opponent=_make_opponent
    )

    def _meta_extra(games: list[Any], tel: Any) -> dict[str, Any]:
        # Persist the games as a replayable .hxr (dashboard "evaluation" source);
        # _write_eval_hxr is fully fail-soft. ``games`` are net-A-centric EvalGame
        # records (.a_is_p0 / .winner in {"A","B",None}), the writer's shape.
        _hxr_stats: dict[str, int] = {}
        hxr_path = _write_eval_hxr(
            games, diagnostics_dir, label, f"SealBot {sealbot_variant}", stats=_hxr_stats
        )
        return {
            "kind": "hexfield_vs_sealbot",
            "ckpt": {"label": label, "path": str(model_ckpt)},
            "sealbot": {"variant": sealbot_variant, "time_limit": sealbot_time_limit},
            "games_requested": n_games,
            "visits": tel.eval_visits,
            "virtual_batch_size": tel.virtual_batch_size,
            "device": cfg.device,
            "opening_plies": opening_plies,
            "opening_temperature": opening_temperature,
            "game_seed_base": game_seed_base,
            "divergence_overrides": overrides,
            "budget_hit": tel.budget_hit,
            "rounds": tel.rounds,
            # forward_batches / mcts time cover ONLY net A's hexfield searches
            # (SealBot decisions go through the minimax worker, not session.search).
            "forward_batches": tel.forward_batches,
            "elapsed_seconds": round(tel.elapsed_seconds, 2),
            "mcts_search_elapsed_seconds": round(tel.mcts_search_elapsed, 2),
            "opponent_elapsed_seconds": round(opponent.opponent_elapsed, 2),
            "hxr_record": hxr_path,
            "hxr_games_written": _hxr_stats.get("games_written", 0),
        }

    # The driver (eval_driver) references the engine only through ITS module-level
    # ``api`` name; sync it to this module's ``api`` (which tests / the golden
    # harness monkeypatch) for the duration of the call, then restore. This lets
    # the shared driver + the SealBotAdapter (new_state / a_to_move) use the same
    # (possibly faked) engine without a second patch site, with no cross-test leak.
    from . import eval_driver as _eval_driver

    _saved_api = _eval_driver.api
    _eval_driver.api = api
    try:
        return play_eval_match(
            model_ckpt,
            opponent,
            n_games,
            config=cfg,
            label_a=label,
            label_b=f"SealBot {sealbot_variant}",
            meta_extra_fn=_meta_extra,
            paired_openings=False,  # SealBot is load-nondeterministic -> unpaired
            visits=visits,
            virtual_batch_size=virtual_batch_size,
            active_root_limit=active_root_limit,
            opening_plies=opening_plies,
            opening_temperature=opening_temperature,
            divergence_overrides=overrides,
            diagnostics_dir=diagnostics_dir,
            game_seed_base=game_seed_base,
            max_wall_seconds=max_wall_seconds,
            max_states=max_states,
            # SealBot's pre-refactor net-A pass used the 7M greedy seed offset and
            # gated the opening on the hexfield move count (a_decisions).
            net_a_greedy_offset=7_000_003,
            opening_gate_attr="a_decisions",
        )
    finally:
        _eval_driver.api = _saved_api


def play_strix_match(
    hexfield_ckpt: str | Path,
    strix_ckpt: str | Path,
    n_games: int,
    *,
    config: Any = None,
    label: str = "hexfield",  # label_a
    strix_label: str = "hexo-strix",  # label_b
    # --- hexfield search knobs (net A) ---
    visits: int | None = None,  # None -> sp.search_visits
    virtual_batch_size: int | None = None,
    opening_plies: int = DEFAULT_OPENING_PLIES,
    opening_temperature: float = DEFAULT_OPENING_TEMPERATURE,
    divergence_overrides: dict | None = None,
    active_root_limit: int | None = None,
    # --- strix search knobs (net B) ---
    strix_sims: int = 256,
    strix_m_actions: int = 16,
    strix_c_visit: int = 50,
    strix_c_scale: float = 1.0,
    strix_disable_gumbel_noise: bool = True,
    # Opening-confined "light noise" for strix (StrixMctsPlayer.noise_opening_plies):
    # None -> opening_plies (strix samples its opening among the top-m candidates,
    # then plays a deterministic greedy tail); 0 -> fully deterministic strix.
    # Confining the noise to the opening keeps the post-opening play reproducible,
    # so strix stays a STABLE anchor, while still diversifying games. The opening
    # noise is seeded from each game's CRN seed, so a fixed game_seed_base replays
    # identically across epochs.
    strix_noise_opening_plies: int | None = None,
    strix_device: str = "cuda",
    strix_linger_s: float = 0.0008,
    strix_max_batch: int = 512,
    strix_pool_threads: int | None = None,  # None -> n_games (see risk R4)
    # --- pairing / bookkeeping ---
    paired_openings: bool = True,
    diagnostics_dir: str | Path | None = None,
    max_states: int = 65_536,
    game_seed_base: int = 0,
    max_wall_seconds: float = 0.0,
    # --- CPU-test injection seams (mirror play_sealbot_match) ---
    build_hexfield_evaluator: Callable[..., Any] | None = None,
    make_session: Callable[..., Any] | None = None,
    build_strix_player: Callable[[int, bool], Any] | None = None,
    make_batch_server: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Play hexfield vs hexo-strix concurrently, GPU-batched on BOTH sides.

    A lock-step multi-game arena that supersedes the thread-per-game
    ``hexo_strix.match_runner.run_threaded_matches`` glue (used by
    ``scripts/_strix_vs_hexfield_eval.py``). It removes that path's two taxes:

      * **hexfield side** is no longer lock-serialised. One driver thread owns a
        single persistent ``HexfieldMctsSession`` and collapses ALL
        hexfield-to-move games into one multi-root ``session.search`` per round
        (one big GPU forward, no lock — only the driver touches the
        non-thread-safe evaluator), chunked at the active-root limit. This is the
        ``play_sealbot_match`` hexfield pass verbatim.
      * **strix side** is fired as a concurrent burst: every strix-to-move game's
        (read-only) ``StrixMctsPlayer.decide`` is submitted to a thread pool at
        once, so their leaves block simultaneously in the shared
        :class:`hexo_strix.batch_server.StrixBatchServer` and coalesce into large
        cross-game GPU forwards. The driver applies each returned action
        single-threaded after the join (``decide`` never mutates the state), so
        there are no write races.

    Both engines share ONE ``hexo_engine`` state per game (``session.search`` and
    ``StrixMctsPlayer.decide`` both take a ``HexoState``), handed to whichever
    side is to move — no per-side state duplication. Net A = hexfield, net B =
    strix throughout, so the returned dict is net-A-centric like
    ``play_sealbot_match``.

    Pairing (``paired_openings=True``, the default): games are grouped into
    ``n_pairs = ceil(n_games / 2)`` CRN pairs sharing a ``pair_seed`` but swapping
    seats — the LEADER (game 0) plays hexfield-as-player0, the seat-swapped
    FOLLOWER (game 1) plays hexfield-as-player1. Only the leader searches its
    opening (hexfield samples via ``opening_temperature``; strix samples via its
    opening-confined Gumbel noise, ``strix_noise_opening_plies``); the follower
    REPLAYS the leader's recorded opening line ply-for-ply (no search on either
    engine), so both games traverse the identical opening board with swapped
    colors. After the opening both engines play a deterministic greedy tail, so a
    pair's only difference is the seat swap — a clean pentanomial CRN pair (net-A
    wins in {0, 1, 2}) that the downstream ``effective_counts`` deflation consumes.
    This works with a (mostly) deterministic strix because the opening variance
    comes from the leader's temperature/noise sampling, which the follower shares.

    The leader/follower forced-opening replay mirrors ``play_checkpoint_match``
    exactly (net A = hexfield pass first, then net B = strix pass; within each
    pass the leader batch runs before the follower replays), so the leader stays
    strictly ahead and the follower's recorded opening action always exists; the
    rare leader-ended-mid-opening case falls back to a single search / decide.

    Unpaired (``paired_openings=False``): every game gets an independent seed and
    seats alternate by game index (even -> hexfield is player0), the
    independent-Bernoulli layout ``play_sealbot_match`` uses for a foreign
    opponent (``pentanomial=None``).

    Injection seams (``build_hexfield_evaluator`` / ``make_session`` /
    ``build_strix_player`` / ``make_batch_server``) let a CPU test drive the loop
    with fakes and no GPU / no ``hexo_rs`` wheel, exactly as ``build_opponent``
    does for ``play_sealbot_match``. ``build_strix_player(game_seed, is_leader)``
    returns a player exposing ``.decide(state) -> DecisionResult``.
    """

    cfg = config if config is not None else parse_hexfield_config({})
    sp = cfg.selfplay
    # Opening-confined light noise window for strix: default to the shared
    # opening_plies so strix samples over the same opening horizon hexfield does.
    strix_noise_plies = (
        int(opening_plies) if strix_noise_opening_plies is None else int(strix_noise_opening_plies)
    )

    # Net A's (and, echoed into meta, the match's) divergence overrides. An
    # explicit dict is returned as-is by _resolve_eval_overrides, so passing this
    # resolved dict on to the driver is idempotent.
    overrides = _resolve_eval_overrides(
        sp, diagnostics_dir=diagnostics_dir, divergence_overrides=divergence_overrides
    )

    # Net B is hexo-strix: a shared GPU batch server + per-game MCTS player pool.
    opponent = StrixAdapter(
        strix_ckpt,
        label=strix_label,
        strix_label=strix_label,
        n_games=n_games,
        paired_openings=paired_openings,
        strix_noise_plies=strix_noise_plies,
        strix_sims=strix_sims,
        strix_m_actions=strix_m_actions,
        strix_c_visit=strix_c_visit,
        strix_c_scale=strix_c_scale,
        strix_disable_gumbel_noise=strix_disable_gumbel_noise,
        strix_device=strix_device,
        strix_linger_s=strix_linger_s,
        strix_max_batch=strix_max_batch,
        strix_pool_threads=strix_pool_threads,
        build_strix_player=build_strix_player,
        make_batch_server=make_batch_server,
    )

    def _meta_extra(games: list[Any], tel: Any) -> dict[str, Any]:
        # Persist the games as a replayable .hxr (dashboard "evaluation" source);
        # _write_eval_hxr is fully fail-soft. ``games`` are net-A-centric EvalGame
        # records (.a_is_p0 / .winner in {"A","B",None}), the writer's shape.
        _hxr_stats: dict[str, int] = {}
        hxr_path = _write_eval_hxr(games, diagnostics_dir, label, strix_label, stats=_hxr_stats)
        return {
            "kind": "hexfield_vs_strix",
            "ckpt": {"label": label, "path": str(hexfield_ckpt)},
            "strix": {
                "label": strix_label,
                "path": str(strix_ckpt),
                "sims": strix_sims,
                "m_actions": strix_m_actions,
                "c_visit": strix_c_visit,
                "c_scale": strix_c_scale,
                "disable_gumbel_noise": strix_disable_gumbel_noise,
                "noise_opening_plies": strix_noise_plies,
                "device": strix_device,
                "linger_s": strix_linger_s,
                "max_batch": strix_max_batch,
            },
            "games_requested": n_games,
            "visits": tel.eval_visits,
            "virtual_batch_size": tel.virtual_batch_size,
            "device": cfg.device,
            "paired_openings": paired_openings,
            "opening_plies": opening_plies,
            "opening_temperature": opening_temperature,
            "game_seed_base": game_seed_base,
            "divergence_overrides": overrides,
            "budget_hit": tel.budget_hit,
            "rounds": tel.rounds,
            # forward_batches / mcts time cover ONLY net A's hexfield searches
            # (strix decisions go through the pool, not session.search).
            "forward_batches": tel.forward_batches,
            "elapsed_seconds": round(tel.elapsed_seconds, 2),
            "mcts_search_elapsed_seconds": round(tel.mcts_search_elapsed, 2),
            "strix_elapsed_seconds": round(opponent.strix_elapsed, 2),
            # Strix batch-server coalescing telemetry (read after close()).
            "strix_forward_batches": opponent.strix_forward_batches,
            "strix_leaves": opponent.strix_leaves,
            "strix_max_batch": opponent.strix_max_seen_batch,
            "hxr_record": hxr_path,
            "hxr_games_written": _hxr_stats.get("games_written", 0),
        }

    # The driver (eval_driver) references the engine only through ITS module-level
    # ``api`` name; sync it to this module's ``api`` (which tests / the golden
    # harness monkeypatch) for the duration of the call, then restore. This lets
    # the shared driver + the StrixAdapter (new_state / a_to_move) use the same
    # (possibly faked) engine without a second patch site, with no cross-test leak.
    from . import eval_driver as _eval_driver

    _saved_api = _eval_driver.api
    _eval_driver.api = api
    try:
        return play_eval_match(
            hexfield_ckpt,
            opponent,
            n_games,
            config=cfg,
            label_a=label,
            label_b=strix_label,
            meta_extra_fn=_meta_extra,
            paired_openings=paired_openings,
            visits=visits,
            virtual_batch_size=virtual_batch_size,
            active_root_limit=active_root_limit,
            opening_plies=opening_plies,
            opening_temperature=opening_temperature,
            divergence_overrides=overrides,
            diagnostics_dir=diagnostics_dir,
            game_seed_base=game_seed_base,
            max_wall_seconds=max_wall_seconds,
            max_states=max_states,
            build_candidate_evaluator=(
                build_hexfield_evaluator if build_hexfield_evaluator is not None else None
            ),
            make_session=make_session,
        )
    finally:
        _eval_driver.api = _saved_api


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
