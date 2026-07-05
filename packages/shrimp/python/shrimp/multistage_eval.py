"""Multi-stage shrimp strength evaluation orchestrator.

Wires the game-running layer (:mod:`shrimp.eval_arena`) to the statistics
layer (:mod:`shrimp.eval_stats`) into a staged protocol. Measures the
candidate's strength against a fixed roster, pools every edge into a persisted,
SealBot-pinned Bradley-Terry rating, and emits a single verdict LABEL
(``PROMOTE`` / ``REGRESS`` / ``INCONCLUSIVE``).

Does not gate, promote, halt, or mutate a training run. The verdict is a
reported string. Config knobs ``eval_gating_enabled`` / ``eval_promotion_enabled``
exist but default OFF and are read nowhere in this module that could alter the
run; :func:`_assert_no_run_mutation` asserts they are False. No checkpoint write,
flag drop, or run-state edit happens anywhere in this module.

STAGE FLOW (``run_multistage_eval``):

  Stage A — bridge / smoke. Resolves the candidate + the full opponent roster
    (SealBot zero-point, permanent anchors, sliding bracket, prior champion) and
    sanity-checks it. No games are played. Never produces a verdict.

  Stage B — SPRT screen (gross-regression triage). Plays paired games vs the
    prior champion, feeding the cumulative decided (w, l) to
    :func:`eval_stats.sprt`. Not a calibrated 5%/5% test: near the indifference
    region the expected-N is ~285 decided games, so a small ``max_games`` cap
    mostly returns ``continue`` -> escalate. Catches a gross regression cheaply
    and early-accepts an obvious improvement; the calibrated measurement is
    Stage C/D.

  Stage C — deep paired eval. Allocates ``games_budget`` across the pairings and
    plays:
      * vs SealBot: concurrent, unpaired (eval_arena.play_sealbot_match) — the
        Wilson-CI zero-point edge.
      * vs every checkpoint opponent (permanent anchors + bracket + prior
        champion): paired with shared openings / common random numbers
        (eval_arena.play_checkpoint_match), scored pentanomially with pair-level
        SEs (eval_stats.pentanomial_summary).

  Stage D — rolling Bradley-Terry pool. Loads ``diagnostics/eval_pool.json``,
    appends THIS epoch's edges, and refits the whole pool with
    :func:`eval_stats.bradley_terry` (SealBot pinned at 0 Elo; SealBot edge
    down-weighted so its non-deterministic depth does not enter difference
    inference at full weight; paired edges pre-deflated to EFFECTIVE counts via
    :func:`eval_stats.effective_counts`). The fit must converge (asserted
    ``max|grad| < bt_grad_tol``) before any covariance is computed.

    RESOLUTION — what compounds and what does not. The primary
    candidate-vs-champion verdict is single-epoch-limited: a fresh candidate node
    enters the pool each epoch (``cand_epN``), so its rating does not accumulate
    information across epochs the way a fixed anchor does. At the per-epoch
    champion-game count the single-epoch ``SE(r_L - r_B)`` is order ~120-140 Elo
    (paired -> effective N below decided, plus the two-rating sqrt(2)), resolving
    ~250-300 Elo — a gross-regression signal, not a fine edge, and it does not
    improve with more epochs. The fixed-anchor descriptive curve does compound:
    the bc_prefit / ep5 / SealBot anchors carry the SAME labels every epoch, so
    their edges pool and their ratings tighten toward ~15-20 Elo over many
    epochs, describing the lineage progress curve rather than the single-epoch
    verdict. SealBot is a down-weighted (0.5) descriptive zero-point that pins
    the rating scale at 0 Elo; it is never the verdict.

VERDICT — one primary hypothesis: candidate ``L`` vs prior champion ``B``, via
the BT difference-CI ``r_L - r_B`` (which carries the shared-anchor ``-2 Cov_LB``
term). Single-epoch-limited (see above) — a gross-regression signal, not a
fine-edge test. Every other opponent edge is descriptive: its Wilson/Elo CI is
reported with no significance verdict. When ``bonferroni_correction`` is set the
per-edge alpha is Bonferroni-split; nothing gates by default.

POOL PERSISTENCE FORMAT (``diagnostics/eval_pool.json``) — see
:func:`_load_pool` / :func:`_save_pool`. A versioned JSON document:

  {
    "format": "shrimp.multistage_eval.pool",
    "version": 1,
    "anchor": "sealbot",                 # the pinned zero-point label
    "edges": [                            # one row PER pairing PER epoch (append-only)
      {
        "epoch": 10,
        "a": "cand_ep10", "b": "ep5",
        "wins_a": 41.0, "wins_b": 23.0,  # EFFECTIVE counts (paired-deflated) for BT
        "weight": 1.0,                    # likelihood weight (<1 down-weights SealBot)
        "kind": "checkpoint",            # "checkpoint" | "sealbot"
        "raw": {...}                      # provenance: physical games, pentanomial, etc.
      }, ...
    ]
  }

Edges are append-only so the pool is a full audit trail; the BT fit consumes the
``(a, b, wins_a, wins_b, weight)`` columns. Re-running the same epoch appends a
fresh batch (idempotency is the caller's concern).

This module imports torch transitively (via eval_arena) and runs GPU search. The
standalone runner (scripts/_shrimp_run_multistage_eval.py) invokes it off the
training loop.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import eval_stats
from .config import ShrimpConfig, MultiStageEvalSection, parse_shrimp_config

# Logger for roster/anchor/SealBot degradation warnings.
_EVAL_LOG = logging.getLogger("shrimp.eval")

# Label of the cross-lineage zero-point in the pool / BT fit. The BT anchor is
# pinned at exactly 0 Elo (eval_stats.bradley_terry(anchor=...)).
SEALBOT_LABEL = "sealbot"

POOL_FORMAT = "shrimp.multistage_eval.pool"
POOL_VERSION = 1

# Diagnostics filename for one epoch's full result (ratings + CIs + verdict).
DIAG_PREFIX = "shrimp.multistage_eval.epoch_"

# Native support radius of opponents listed in ``cfg.opponents.radius8_opponents``.
_RADIUS8_NATIVE = 8


def _live_featurize_radius() -> int:
    """The live process featurizer/support radius (read-only).

    The support radius is a process-global read once at import from
    ``SHRIMP_SUPPORT_RADIUS`` (support._SUPPORT_RADIUS); the in-process eval
    featurizes every opponent at this radius. Read-only; used to detect a
    radius-8-native opponent featurized out-of-distribution.
    """

    try:
        from . import support

        return int(support._SUPPORT_RADIUS)
    except Exception:  # pragma: no cover - support import is normally present
        from .constants import LEGAL_RADIUS

        return int(LEGAL_RADIUS)


def _opponent_featurized_ood(opp_label: str, cfg: MultiStageEvalSection, live_radius: int) -> bool:
    """True when a radius-8-native opponent is featurized at a non-8 radius.

    Only labels listed in ``cfg.opponents.radius8_opponents`` are eligible, and
    only when ``live_radius`` differs from their native radius 8.
    """

    radius8 = set(getattr(cfg.opponents, "radius8_opponents", ()) or ())
    return opp_label in radius8 and live_radius != _RADIUS8_NATIVE


def _eval_visits(cfg: MultiStageEvalSection, full_cfg: ShrimpConfig) -> int:
    """Search visits for this epoch's eval games.

    Resolution:
      * ``cfg.full_search_visits`` if set (an int), else
      * ``full_cfg.selfplay.search_visits``.
    A caller can force a different budget by passing ``visits=`` to the arena
    directly; the orchestrator threads this resolved value.
    """

    if cfg.full_search_visits is not None:
        return int(cfg.full_search_visits)
    return int(full_cfg.selfplay.search_visits)


def _eval_virtual_batch_size(cfg: MultiStageEvalSection, full_cfg: ShrimpConfig) -> int:
    """Eval-only MCTS virtual-batch size for this epoch's eval games.

    Returns ``int(cfg.eval_virtual_batch_size)`` (default 16), threaded into
    every eval search call so eval leaf-parallelism is independent of
    ``SelfplayConfig.virtual_batch_size``. Falls back to the self-play value if
    the field is None.
    """

    v = getattr(cfg, "eval_virtual_batch_size", None)
    if v is not None:
        return int(v)
    return int(full_cfg.selfplay.virtual_batch_size)


# --------------------------------------------------------------------------- #
# Opponent roster resolution (permanent anchors + sliding bracket).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Opponent:
    """One resolved opponent in the roster.

    label:   pool/rating label (stable across epochs so edges compound).
    role:    "sealbot" | "anchor" | "bracket" | "champion". The primary verdict
             compares the candidate to the single ``champion`` opponent; every
             other role is descriptive. SealBot is additionally the pinned
             zero-point and is kept out of difference inference (down-weighted).
    ckpt:    checkpoint path (None for SealBot, which is an external engine).
    epoch:   the opponent's epoch where meaningful (anchors/bracket/champion),
             else None.
    """

    label: str
    role: str
    ckpt: Path | None
    epoch: int | None = None


@dataclass(frozen=True)
class Roster:
    """The full resolved opponent set for one candidate epoch.

    candidate_label / candidate_epoch identify ``L`` (the player under test).
    champion is the prior champion ``B`` (the primary comparison target), or
    None when no prior checkpoint exists (the first eligible epoch), in which
    case there is no primary hypothesis and the verdict is INCONCLUSIVE. sealbot
    is the zero-point (or None when disabled/unavailable). opponents is every
    checkpoint opponent (anchors + bracket + champion, de-duplicated by label)
    that the deep eval plays paired games against.
    """

    candidate_label: str
    candidate_epoch: int | None
    sealbot: Opponent | None
    champion: Opponent | None
    opponents: tuple[Opponent, ...]
    # Permanent anchors that failed to resolve on disk and were dropped from the
    # roster. Each entry: {"label", "raw", "resolved"}. Empty for a fully-resolved
    # roster. Surfaced in _roster_summary (per-epoch JSON) and logged.
    dropped_anchors: tuple[dict, ...] = ()

    def all_labels(self) -> list[str]:
        labels = [self.candidate_label]
        if self.sealbot is not None:
            labels.append(self.sealbot.label)
        labels.extend(o.label for o in self.opponents)
        return labels


def _resolve_anchor_path(run_dir: Path, checkpoints_dir: Path, raw: str) -> Path:
    """Resolve a permanent-anchor path string.

    The config stores anchors as forward-slash strings. A path containing a
    slash (e.g. ``runs/shrimp_bc_1/checkpoint_epoch2.pt``) is resolved against
    several candidate roots and the first where the file exists wins; a bare
    filename (e.g. ``epoch_000005.pt``) is resolved against the run's checkpoints
    dir.

    The run-data tree (where ``epoch_*.pt`` live) and the repo tree that ships
    this package can be separate directories; the same repo-relative anchor may
    exist in only one of them. Candidate roots are tried in order (env override,
    then run-data ancestors, then the repo tree, then the run_dir grandparent)
    and the first yielding an existing file is returned.
    """

    raw = raw.replace("\\", "/")
    p = Path(raw)
    if p.is_absolute():
        return p
    if "/" in raw:
        # Repo-relative (e.g. "runs/shrimp_bc_1/..."). Collect candidate roots
        # and pick the first where the file exists.
        head = raw.split("/", 1)[0]
        roots: list[Path] = []
        # (0) Env-overridable extra search roots: SHRIMP_ANCHOR_ROOTS
        #     (os.pathsep-separated absolute dirs). Tried first, so an operator
        #     override wins over the derived roots below.
        env_roots = os.environ.get("SHRIMP_ANCHOR_ROOTS", "")
        for r in env_roots.split(os.pathsep):
            r = r.strip()
            if r:
                roots.append(Path(r))
        # (a) Walk up from run_dir to each ancestor that holds the leading
        #     component (run-data tree — where epoch_*.pt live).
        anchor_root = run_dir
        for _ in range(6):
            if (anchor_root / head).exists():
                roots.append(anchor_root)
            if anchor_root.parent == anchor_root:
                break
            anchor_root = anchor_root.parent
        # (b) The repo tree that ships this package. This file is
        #     <repo>/packages/shrimp/python/shrimp/multistage_eval.py, so
        #     parents[4] == <repo>. Guarded so a differently-vendored package
        #     omits this root.
        try:
            repo_root = Path(__file__).resolve().parents[4]
            if (repo_root / head).exists():
                roots.append(repo_root)
        except IndexError:
            pass
        # (c) Fallback: run_dir's grandparent (the runs/ container's sibling).
        roots.append(run_dir.parent.parent)
        # First root where the file exists wins; else fall back to the first
        # candidate root so the caller's is_file() skip reports a path.
        for root in roots:
            cand = root / raw
            if cand.is_file():
                return cand
        return (roots[0] / raw) if roots else (run_dir.parent.parent / raw)
    return checkpoints_dir / raw


def _epoch_ckpt(checkpoints_dir: Path, epoch: int) -> Path:
    """Standard in-run checkpoint path (6-digit, matching evaluate_epoch)."""

    return checkpoints_dir / f"epoch_{epoch:06d}.pt"


def _discover_epochs(checkpoints_dir: Path) -> list[int]:
    """All epochs with an ``epoch_NNNNNN.pt`` on disk, ascending."""

    epochs: list[int] = []
    if not checkpoints_dir.is_dir():
        return epochs
    for p in checkpoints_dir.glob("epoch_*.pt"):
        stem = p.stem  # "epoch_000010"
        try:
            epochs.append(int(stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(epochs))


def select_opponents(
    run_dir: str | Path,
    candidate_ckpt: str | Path,
    cfg: MultiStageEvalSection,
    *,
    candidate_epoch: int | None = None,
    candidate_label: str | None = None,
    checkpoints_dir: str | Path | None = None,
) -> Roster:
    """Resolve the opponent roster for ``candidate_ckpt``.

    Roles:
      * SealBot — cross-lineage zero-point, pinned at 0 Elo (role ``sealbot``).
      * Permanent anchors — from ``cfg.opponents.permanent_anchors``. Resolved on
        disk; missing ones are skipped with no error.
      * Sliding bracket — the nearest ``bracket_size`` rungs of
        ``cfg.opponents.log_grid`` strictly below the candidate epoch that exist
        on disk.

    Champion (the primary comparison target): the highest-epoch checkpoint at
    least ``verdict_reference_lag`` epochs below the candidate that exists on
    disk (falling back to the nearest prior when none is old enough). May
    coincide with a bracket rung; de-duplicated by label so it appears once in
    ``opponents`` with role ``champion``.

    Returns a :class:`Roster`. Path resolution only — no checkpoints are loaded
    and no games are played.
    """

    run_dir = Path(run_dir)
    candidate_ckpt = Path(candidate_ckpt)
    ckpt_dir = Path(checkpoints_dir) if checkpoints_dir is not None else (run_dir / "checkpoints")

    cand_epoch = candidate_epoch if candidate_epoch is not None else _infer_epoch(candidate_ckpt)
    cand_label = candidate_label or (
        f"cand_ep{cand_epoch}" if cand_epoch is not None else f"cand_{candidate_ckpt.stem}"
    )

    opp_cfg = cfg.opponents

    # SealBot zero-point (path/availability is validated later by the runner).
    sealbot = (
        Opponent(label=SEALBOT_LABEL, role="sealbot", ckpt=None, epoch=None)
        if opp_cfg.sealbot_enabled
        else None
    )

    # Collect checkpoint opponents into a label-keyed dict so duplicates merge
    # (champion may equal a bracket rung). Insertion order is preserved for a
    # stable, readable roster: anchors, then bracket, then champion.
    collected: dict[str, Opponent] = {}

    # Permanent anchors.
    dropped_anchors: list[dict] = []
    for label, raw in opp_cfg.permanent_anchors:
        path = _resolve_anchor_path(run_dir, ckpt_dir, raw)
        if not path.is_file():
            # An unresolved permanent anchor is recorded and logged, then skipped.
            dropped_anchors.append(
                {"label": str(label), "raw": str(raw), "resolved": str(path)}
            )
            _EVAL_LOG.warning(
                "permanent anchor %r unresolved (%s); dropping from roster",
                label,
                path,
            )
            continue
        collected.setdefault(
            label, Opponent(label=label, role="anchor", ckpt=path, epoch=_infer_epoch(path))
        )

    # Sliding bracket: nearest bracket_size log-grid rungs strictly below cand.
    if cand_epoch is not None:
        below = [g for g in sorted(opp_cfg.log_grid) if g < cand_epoch]
        bracket_rungs = below[-max(int(opp_cfg.bracket_size), 0):] if opp_cfg.bracket_size > 0 else []
        for g in bracket_rungs:
            path = _epoch_ckpt(ckpt_dir, g)
            if not path.is_file():
                continue
            label = f"ep{g}"
            collected.setdefault(label, Opponent(label=label, role="bracket", ckpt=path, epoch=g))

    # Champion (verdict reference): the highest existing epoch at least
    # ``verdict_reference_lag`` epochs below the candidate, falling back to the
    # nearest prior when none is old enough (e.g. the first few epochs). Any
    # closer checkpoint still appears as a descriptive bracket edge and is pooled
    # into the BT fit; only the reported verdict target rests on this reference.
    champion: Opponent | None = None
    if cand_epoch is not None:
        prior_epochs = [e for e in _discover_epochs(ckpt_dir) if e < cand_epoch]
        if prior_epochs:
            lag = max(int(getattr(cfg, "verdict_reference_lag", 0)), 0)
            eligible = [e for e in prior_epochs if e <= cand_epoch - lag] or prior_epochs
            champ_epoch = eligible[-1]
            champ_path = _epoch_ckpt(ckpt_dir, champ_epoch)
            champ_label = f"ep{champ_epoch}"
            # If the champion is already present (e.g. also a bracket rung),
            # reuse its ckpt/epoch but force role "champion".
            existing = collected.get(champ_label)
            champion = Opponent(label=champ_label, role="champion", ckpt=champ_path, epoch=champ_epoch)
            collected[champ_label] = champion if existing is None else Opponent(
                label=champ_label, role="champion", ckpt=existing.ckpt, epoch=existing.epoch
            )
            champion = collected[champ_label]

    return Roster(
        candidate_label=cand_label,
        candidate_epoch=cand_epoch,
        sealbot=sealbot,
        champion=champion,
        opponents=tuple(collected.values()),
        dropped_anchors=tuple(dropped_anchors),
    )


def _infer_epoch(ckpt: Path) -> int | None:
    """Best-effort epoch from a checkpoint filename (``epoch_NNNNNN.pt`` or
    ``checkpoint_epochN.pt``). Returns None if no epoch is encoded."""

    stem = ckpt.stem
    for token in ("epoch_", "checkpoint_epoch", "epoch"):
        if token in stem:
            tail = stem.split(token, 1)[1].lstrip("_")
            digits = ""
            for ch in tail:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                return int(digits)
    return None


# --------------------------------------------------------------------------- #
# Budget allocation across pairings (Stage C).
# --------------------------------------------------------------------------- #
def allocate_budget(
    total_games: int,
    *,
    n_checkpoint_opponents: int,
    has_sealbot: bool,
    sealbot_share: float = 0.25,
) -> dict[str, int]:
    """Split ``total_games`` across the deep-eval pairings.

    SealBot gets ``sealbot_share`` of the budget (floored to 2 when enabled and
    the budget is positive); the remainder is split evenly across the N
    checkpoint opponents and rounded down to an even per-pairing count (paired
    games come two-per-pair, floored to 2). Returns a dict with keys
    :data:`SEALBOT_LABEL` -> SealBot games and ``"per_checkpoint"`` -> games per
    checkpoint opponent; the caller distributes the latter across its opponents.

    Allocation is arithmetic and deterministic; no games are played.

    The per-opponent floor of one CRN pair (2 games) means the budget is
    advisory, not a hard cap, at small N: the physical-game sum may exceed
    ``total_games`` (e.g. budget 4 with 4 opponents -> 2 each = 8 games). A zero
    budget returns all-zeros.
    """

    total = max(int(total_games), 0)
    if total == 0:
        return {SEALBOT_LABEL: 0, "per_checkpoint": 0}
    sealbot_games = 0
    if has_sealbot:
        sealbot_games = int(round(total * max(0.0, min(sealbot_share, 1.0))))
        # Floor SealBot to one pairing's worth so the zero-point edge is played.
        sealbot_games = max(sealbot_games, 2)
    checkpoint_total = max(total - sealbot_games, 0)
    if n_checkpoint_opponents <= 0:
        # No checkpoint opponents (first epoch): everything to SealBot if present.
        return {SEALBOT_LABEL: total if has_sealbot else 0, "per_checkpoint": 0}
    per = checkpoint_total // n_checkpoint_opponents
    if per % 2 == 1:  # keep pairings even (two games per CRN pair)
        per -= 1
    # Floor at one CRN pair so every selected opponent is played.
    per = max(per, 2)
    return {SEALBOT_LABEL: sealbot_games, "per_checkpoint": per}


# --------------------------------------------------------------------------- #
# Pentanomial -> BT effective edge (paired -> effective counts).
# --------------------------------------------------------------------------- #
def _pentanomial_to_paired_result(pentanomial: dict[str, Any]) -> "eval_stats.PairedResult | None":
    """Build an ``eval_stats.PairedResult`` from an eval_arena pentanomial block.

    eval_arena emits a 3-bucket histogram over 2-game pairs keyed by net-A wins
    in {0, 1, 2}. Hexo has no draws, so a complete pair's candidate points are
    {0, 1, 2} out of 2 -> per-pair score {0, 0.5, 1}, mapping to the outer and
    middle buckets of eval_stats' 5-bucket pentanomial (the 0.25 / 0.75 buckets
    stay empty for full pairs; they arise only when a pair is scored on a single
    decided game, folded via :func:`eval_stats.paired_winrate`). Exact per-pair
    scores are used when present; otherwise the histogram is reconstructed.

    Returns None when there are no informative pairs (e.g. an all-truncated or
    empty match), so the caller can skip the edge.
    """

    if not pentanomial:
        return None
    pairs = pentanomial.get("pairs")
    if pairs:
        scores: list[float] = []
        for p in pairs:
            n_dec = p.get("n_decided", 0)
            if n_dec <= 0:
                continue
            scores.append(p["a_wins"] / n_dec)
        if scores:
            return eval_stats.paired_winrate(scores)
    # Fallback: reconstruct from the full-pair histogram (each full pair is 2
    # decided games -> score 0, 0.5, or 1).
    hist = pentanomial.get("histogram_a_wins") or {}
    penta = [
        int(hist.get("0", 0)),  # 0.0
        0,                       # 0.25 (never for full pairs)
        int(hist.get("1", 0)),  # 0.5
        0,                       # 0.75
        int(hist.get("2", 0)),  # 1.0
    ]
    if sum(penta) <= 0:
        return None
    return eval_stats.pentanomial_summary(tuple(penta))


def _checkpoint_edge_counts(match: dict[str, Any]) -> tuple[float, float, float, dict[str, Any]]:
    """Effective ``(wins_cand, wins_opp, n_eff)`` + provenance for a paired match.

    ``play_checkpoint_match`` is net-A-centric with the candidate as net A, so
    ``score.a_wins`` are candidate wins. The BT likelihood consumes effective
    counts (paired games are correlated), so the decided counts are deflated via
    the pentanomial design effect (:func:`eval_stats.effective_counts`). Without
    a pentanomial the raw decided counts are used.

    The returned provenance dict records both the physical and effective counts.
    """

    score = match.get("score") or {}
    raw_a = float(score.get("a_wins", 0))
    raw_b = float(score.get("b_wins", 0))
    # Record the search budget this edge was played at. The pool aggregates edges
    # across epochs by (a, b) label only, so the budget is kept in provenance for
    # auditability.
    eval_visits = (match.get("meta") or {}).get("visits")
    eval_vbs = (match.get("meta") or {}).get("virtual_batch_size")
    paired = _pentanomial_to_paired_result(match.get("pentanomial") or {})
    if paired is not None and paired.n_pairs > 0 and math.isfinite(paired.win_rate):
        w_eff, l_eff, n_eff = eval_stats.effective_counts(paired)
        prov = {
            "physical_wins_a": raw_a,
            "physical_wins_b": raw_b,
            "eval_visits": eval_visits,
            "virtual_batch_size": eval_vbs,
            "n_pairs": paired.n_pairs,
            "pentanomial": list(paired.penta),
            "pair_winrate": round(paired.win_rate, 6),
            "pair_se": round(paired.se, 6) if math.isfinite(paired.se) else None,
            "n_eff": round(n_eff, 4),
            "wins_a_eff": round(w_eff, 4),
            "wins_b_eff": round(l_eff, 4),
        }
        return w_eff, l_eff, n_eff, prov
    # Degenerate fallback: raw decided counts.
    prov = {
        "physical_wins_a": raw_a,
        "physical_wins_b": raw_b,
        "eval_visits": eval_visits,
        "virtual_batch_size": eval_vbs,
        "n_pairs": None,
        "note": "no pentanomial; using raw decided counts",
    }
    return raw_a, raw_b, raw_a + raw_b, prov


def _sealbot_edge(match: dict[str, Any], overdispersion: float) -> tuple[float, float, float, dict[str, Any]]:
    """SealBot edge counts + down-weight factor.

    SealBot games are unpaired (concurrent), so the counts are the raw decided
    (w, l) with the candidate as net A. SealBot's minimax depth varies under GPU
    load, so its edge carries a likelihood ``weight = overdispersion`` (a
    configured factor in (0, 1]) that scales both the gradient and the Fisher
    information of this edge in the BT fit. SealBot is the BT anchor regardless
    of this weight.
    """

    score = match.get("score") or {}
    raw_a = float(score.get("a_wins", 0))
    raw_b = float(score.get("b_wins", 0))
    w = max(0.0, min(float(overdispersion), 1.0))
    prov = {
        "physical_wins_cand": raw_a,
        "physical_wins_sealbot": raw_b,
        "eval_visits": (match.get("meta") or {}).get("visits"),
        "virtual_batch_size": (match.get("meta") or {}).get("virtual_batch_size"),
        "overdispersion_weight": w,
        "note": "unpaired; zero-point edge, down-weighted out of difference inference",
    }
    return raw_a, raw_b, raw_a + raw_b, prov


# --------------------------------------------------------------------------- #
# Rolling pool persistence (Stage D) — append-only edge log.
# --------------------------------------------------------------------------- #
def _pool_path(run_dir: Path, cfg: MultiStageEvalSection) -> Path:
    """Resolve the persisted pool path (config ``pool_path`` is run-relative)."""

    p = Path(cfg.pool_path.replace("\\", "/"))
    return p if p.is_absolute() else (run_dir / p)


def _load_pool(path: Path) -> dict[str, Any]:
    """Load the rolling pool, or return a fresh empty document.

    A missing / unreadable / version-mismatched file yields a fresh pool rather
    than raising (prior edges are then lost, which loosens CIs).
    """

    if not path.is_file():
        return {"format": POOL_FORMAT, "version": POOL_VERSION, "anchor": SEALBOT_LABEL, "edges": []}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"format": POOL_FORMAT, "version": POOL_VERSION, "anchor": SEALBOT_LABEL, "edges": []}
    if not isinstance(doc, dict) or doc.get("format") != POOL_FORMAT:
        return {"format": POOL_FORMAT, "version": POOL_VERSION, "anchor": SEALBOT_LABEL, "edges": []}
    doc.setdefault("anchor", SEALBOT_LABEL)
    doc.setdefault("edges", [])
    if not isinstance(doc["edges"], list):
        doc["edges"] = []
    return doc


def _save_pool(path: Path, doc: dict[str, Any]) -> None:
    """Persist the pool atomically (temp file + replace).

    Writes only the eval pool JSON under the diagnostics tree.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(path)


def _bt_edges_from_pool(doc: dict[str, Any]) -> list[eval_stats.BTEdge]:
    """Project the append-only edge log into BT edges for the fit.

    Each row is already in effective counts (paired-deflated) with its weight.
    Rows sharing an unordered (a, b) pair and weight are aggregated by summing
    counts, so many epochs of the same pairing become one BT edge; distinct
    weights are kept separate, so down-weighted SealBot edges are not merged with
    full-weight edges.
    """

    agg: dict[tuple[str, str, float], list[float]] = {}
    for row in doc.get("edges", []):
        try:
            a = str(row["a"])
            b = str(row["b"])
            wa = float(row["wins_a"])
            wb = float(row["wins_b"])
            weight = float(row.get("weight", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        # Canonicalise direction so (a,b) and (b,a) pool together.
        if a <= b:
            key = (a, b, weight)
            agg.setdefault(key, [0.0, 0.0])
            agg[key][0] += wa
            agg[key][1] += wb
        else:
            key = (b, a, weight)
            agg.setdefault(key, [0.0, 0.0])
            agg[key][0] += wb
            agg[key][1] += wa
    return [
        eval_stats.BTEdge(a=k[0], b=k[1], wins_a=v[0], wins_b=v[1], weight=k[2])
        for k, v in agg.items()
        if (v[0] + v[1]) > 0.0
    ]


# --------------------------------------------------------------------------- #
# No-run-mutation invariant (defensive).
# --------------------------------------------------------------------------- #
def _assert_no_run_mutation(cfg: MultiStageEvalSection) -> None:
    """Assert the gating/promotion knobs are off.

    ``eval_gating_enabled`` / ``eval_promotion_enabled`` are config-readable but
    consumed nowhere in this module: no branch here writes a checkpoint, drops a
    flag, edits run state, or signals the trainer. The assertion is a tripwire so
    a future default flip is caught in tests.
    """

    assert not cfg.eval_gating_enabled, (
        "multistage_eval is PURE EVAL: eval_gating_enabled must be False. The "
        "verdict is a reported label and is wired to nothing that alters the run."
    )
    assert not cfg.eval_promotion_enabled, (
        "multistage_eval is PURE EVAL: eval_promotion_enabled must be False. No "
        "checkpoint/flag/run-state write happens here regardless of the verdict."
    )


# --------------------------------------------------------------------------- #
# The orchestrator.
# --------------------------------------------------------------------------- #
@dataclass
class StageResult:
    """One stage's structured outcome (for the diagnostics JSON)."""

    stage: str
    status: str
    detail: dict[str, Any] = field(default_factory=dict)


def run_multistage_eval(
    run_dir: str | Path,
    candidate_ckpt: str | Path,
    config: ShrimpConfig | MultiStageEvalSection | None = None,
    *,
    candidate_epoch: int | None = None,
    candidate_label: str | None = None,
    checkpoints_dir: str | Path | None = None,
    diagnostics_dir: str | Path | None = None,
    write_diagnostics: bool = True,
    play_checkpoint_match: Callable[..., dict[str, Any]] | None = None,
    play_sealbot_match: Callable[..., dict[str, Any]] | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Run the staged shrimp strength eval and return a verdict-bearing report.

    Parameters
    ----------
    run_dir : the RL run directory (``<run>/checkpoints/epoch_*.pt`` +
        ``<run>/diagnostics/``). Opponent anchors/bracket are resolved relative
        to it (see :func:`select_opponents`).
    candidate_ckpt : the checkpoint under test (player ``L``).
    config : a :class:`ShrimpConfig` (its ``.multi_stage_eval`` is used) or a
        :class:`MultiStageEvalSection` directly, or None for defaults. NB: the
        section defaults to ``enabled=False`` — this function runs regardless of
        that switch (the switch governs whether a *caller* schedules it), but
        respects ``sprt.enabled`` for Stage B.
    candidate_epoch / candidate_label : override the epoch/label inferred from
        the checkpoint filename.
    play_checkpoint_match / play_sealbot_match : injection seams for the two
        eval_arena runners (tests pass fakes to exercise the orchestration on a
        CPU with no GPU/SealBot). Default to the real ``eval_arena`` functions,
        imported lazily so importing THIS module never requires torch.
    write_diagnostics : when True, write the per-epoch diagnostics JSON and the
        updated pool. Set False in tests to keep the run pure.

    Returns a report dict: ``meta`` / ``roster`` / ``stages`` (A-D) /
    ``ratings`` (pooled BT table with CIs) / ``edges`` (this epoch's descriptive
    edges) / ``verdict`` (the primary label). Nothing here gates, promotes, or
    halts; the verdict is a reported string.
    """

    run_dir = Path(run_dir)
    candidate_ckpt = Path(candidate_ckpt)
    cfg = _coerce_section(config)
    _assert_no_run_mutation(cfg)

    diag_dir = Path(diagnostics_dir) if diagnostics_dir is not None else (run_dir / "diagnostics")
    started = now()

    if play_checkpoint_match is None or play_sealbot_match is None:
        # Lazy import: keep torch/CUDA out of import time and out of the pure
        # statistics path; only needed when actually playing games.
        from . import eval_arena as _arena

        if play_checkpoint_match is None:
            play_checkpoint_match = _arena.play_checkpoint_match
        if play_sealbot_match is None:
            play_sealbot_match = _arena.play_sealbot_match

    # ----- Roster (used by Stage A onward). -----
    roster = select_opponents(
        run_dir, candidate_ckpt, cfg,
        candidate_epoch=candidate_epoch, candidate_label=candidate_label,
        checkpoints_dir=checkpoints_dir,
    )
    cand_label = roster.candidate_label

    stages: list[StageResult] = []

    # ===== Stage A — bridge / smoke ===========================================
    stage_a = _stage_a_bridge(roster, candidate_ckpt)
    stages.append(stage_a)

    # The full opponent config we hand to the runners (search/opening knobs).
    full_cfg = config if isinstance(config, ShrimpConfig) else parse_shrimp_config({})

    # ===== Stage B — SPRT screen (gross-regression triage) ====================
    stage_b, sprt_match = _stage_b_sprt(
        cfg, roster, candidate_ckpt, full_cfg,
        play_checkpoint_match=play_checkpoint_match,
        diagnostics_dir=diag_dir,
    )
    stages.append(stage_b)

    # ===== Stage C — deep paired eval =========================================
    stage_c, edges, sealbot_winrate_ci = _stage_c_deep(
        cfg, roster, candidate_ckpt, full_cfg,
        play_checkpoint_match=play_checkpoint_match,
        play_sealbot_match=play_sealbot_match,
        diagnostics_dir=diag_dir,
        reuse_champion_match=sprt_match,
    )
    stages.append(stage_c)

    # ===== Stage D — rolling Bradley-Terry pool ===============================
    # Thread the SealBot-unavailable flag (only when SealBot is config-enabled).
    sealbot_expected_but_unavailable = (
        stage_c.detail.get("sealbot_unavailable") if roster.sealbot is not None else None
    )
    stage_d, ratings, verdict_block, pool_doc = _stage_d_pool(
        cfg, roster, edges, run_dir,
        sealbot_expected_but_unavailable=sealbot_expected_but_unavailable,
    )
    stages.append(stage_d)

    report: dict[str, Any] = {
        "meta": {
            "kind": "shrimp.multistage_eval",
            "run_dir": str(run_dir),
            "candidate_ckpt": str(candidate_ckpt),
            "candidate_label": cand_label,
            "candidate_epoch": roster.candidate_epoch,
            "anchor": SEALBOT_LABEL,
            "config": _config_summary(cfg),
            "elapsed_seconds": round(now() - started, 2),
            # Single-epoch resolution note, derived from the champion games played
            # this epoch.
            "single_epoch_se_elo_note": _resolution_note(cfg, edges, roster),
            "pure_eval": True,
            "gating_enabled": cfg.eval_gating_enabled,
            "promotion_enabled": cfg.eval_promotion_enabled,
        },
        "roster": _roster_summary(roster),
        "stages": [{"stage": s.stage, "status": s.status, **s.detail} for s in stages],
        "ratings": ratings,
        "edges": [e["descriptive"] for e in edges],
        "sealbot_winrate_ci95": sealbot_winrate_ci,
        "verdict": verdict_block,
    }

    if write_diagnostics:
        _save_pool(_pool_path(run_dir, cfg), pool_doc)
        epoch_tag = roster.candidate_epoch if roster.candidate_epoch is not None else 0
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag_path = diag_dir / f"{DIAG_PREFIX}{epoch_tag:06d}.json"
        diag_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["meta"]["diagnostics_path"] = str(diag_path)

    return report


# --------------------------------------------------------------------------- #
# Stage A — bridge / smoke.
# --------------------------------------------------------------------------- #
def _stage_a_bridge(roster: Roster, candidate_ckpt: Path) -> StageResult:
    """Resolve + sanity-check the roster before the expensive stages.

    Checks that the candidate checkpoint exists and the roster has at least the
    zero-point or a comparison opponent. No games are played. Reports what was
    resolved so a misconfigured anchor/bracket is visible before Stage C.
    """

    problems: list[str] = []
    if not candidate_ckpt.is_file():
        problems.append(f"candidate checkpoint missing: {candidate_ckpt}")
    if roster.sealbot is None and not roster.opponents:
        problems.append("no opponents resolved (no SealBot, no anchors, no bracket)")
    if roster.champion is None:
        problems.append("no prior champion (first eligible epoch) -> no primary hypothesis")
    # A dropped permanent anchor is reported here as a problem (it is also in
    # _roster_summary).
    for d in roster.dropped_anchors:
        problems.append(
            f"permanent anchor dropped (unresolved): {d.get('label')} -> {d.get('resolved')}"
        )

    status = "ok" if not problems or (roster.champion is None and len(problems) == 1) else "degraded"
    if not candidate_ckpt.is_file():
        status = "error"
    return StageResult(
        stage="A_bridge",
        status=status,
        detail={
            "candidate_exists": candidate_ckpt.is_file(),
            "sealbot": roster.sealbot.label if roster.sealbot else None,
            "champion": roster.champion.label if roster.champion else None,
            "n_checkpoint_opponents": len(roster.opponents),
            "opponents": [{"label": o.label, "role": o.role, "epoch": o.epoch} for o in roster.opponents],
            "dropped_anchors": [dict(d) for d in roster.dropped_anchors],
            "notes": problems,
        },
    )


# --------------------------------------------------------------------------- #
# Stage B — SPRT screen (gross-regression triage).
# --------------------------------------------------------------------------- #
def _stage_b_sprt(
    cfg: MultiStageEvalSection,
    roster: Roster,
    candidate_ckpt: Path,
    full_cfg: ShrimpConfig,
    *,
    play_checkpoint_match: Callable[..., dict[str, Any]],
    diagnostics_dir: Path,
) -> tuple[StageResult, dict[str, Any] | None]:
    """Play up to ``sprt.max_games`` paired games vs the reference and screen.

    Gross-regression triage only; not a calibrated 5%/5% test of a small edge
    (see eval_stats.sprt). The two simple hypotheses (config defaults
    MultiStageEvalSprt.elo0 / elo1):
      * H0 (``elo0 = 0``): Elo gap ~0 vs the reference.
      * H1 (``elo1 = -large``, default -50): a large negative Elo gap.
        ``winrate_from_elo`` puts ``p1 < 0.5 < p0``, so a loss-dominated record
        drives the LLR up to ``upper`` and accepts H1.

    Plays paired games (the returned match seeds Stage C's pool) and feeds the
    cumulative decided (w, l) to eval_stats.sprt. The triage label is one of:
      * ``accept_h1`` -> "regress_suspected",
      * ``accept_h0`` -> "ok",
      * ``continue``  -> "escalate" (the common near-indifference outcome).
    The screen does not short-circuit the deep eval; Stage C always runs. The
    returned reference match is reused by Stage C.

    The screen target is ``roster.champion``. Returns (StageResult,
    reference_match_or_None); the match is None when SPRT is disabled or there is
    no reference to screen against.
    """

    sprt_cfg = cfg.sprt
    if not sprt_cfg.enabled:
        return StageResult("B_sprt", "skipped", {"reason": "sprt disabled"}), None
    if roster.champion is None or roster.champion.ckpt is None:
        return StageResult("B_sprt", "skipped", {"reason": "no prior champion"}), None

    n = max(int(sprt_cfg.max_games), 0)
    if n <= 0:
        return StageResult("B_sprt", "skipped", {"reason": "max_games=0"}), None

    # Play the SPRT block as one concurrent match, then apply the SPRT LLR
    # accounting on the cumulative decided (w, l). The returned match is reused by
    # Stage C.
    match = play_checkpoint_match(
        str(candidate_ckpt),
        str(roster.champion.ckpt),
        n,
        config=full_cfg,
        label_a=roster.candidate_label,
        label_b=roster.champion.label,
        paired_openings=True,
        visits=_eval_visits(cfg, full_cfg),
        virtual_batch_size=_eval_virtual_batch_size(cfg, full_cfg),
        opening_plies=cfg.opening_plies,
        opening_temperature=cfg.opening_temperature,
        # The champion (a current-arch checkpoint) searches the candidate's own
        # eval profile, symmetric with Stage C's top-up of this same match.
        diagnostics_dir=str(diagnostics_dir),
    )
    score = match.get("score") or {}
    wins = int(score.get("a_wins", 0))
    losses = int(score.get("b_wins", 0))
    sprt_res = eval_stats.sprt(
        wins, losses,
        elo0=sprt_cfg.elo0, elo1=sprt_cfg.elo1,
        alpha=sprt_cfg.alpha, beta=sprt_cfg.beta,
    )
    # Label mapping: accepting H1 (elo1 = -large) means a regression is
    # suspected; accepting H0 (elo0 = 0) means the candidate looks fine (still
    # escalates to Stage C/D). Matches config defaults elo0=0, elo1=-50.
    triage = {
        "accept_h1": "regress_suspected",
        "accept_h0": "ok",
        "continue": "escalate",
    }[sprt_res.verdict]
    return (
        StageResult(
            stage="B_sprt",
            status="completed",
            detail={
                "vs": roster.champion.label,
                "games_requested": n,
                "decided": wins + losses,
                "wins_cand": wins,
                "wins_champion": losses,
                "llr": round(sprt_res.llr, 4),
                "bounds": [round(sprt_res.lower, 4), round(sprt_res.upper, 4)],
                "sprt_verdict": sprt_res.verdict,
                "triage": triage,
                "note": (
                    "GROSS-REGRESSION TRIAGE only — H0=fine (Elo~0), "
                    "H1=grossly regressed (Elo~-50); accept_h1 -> regress_suspected, "
                    "accept_h0 -> ok. NOT a calibrated 5%/5% test; near indifference "
                    "the honest expected-N is ~285 decided games, so a small cap "
                    "mostly returns escalate. Deep eval (Stage C/D) is the "
                    "calibrated measurement and always runs."
                ),
            },
        ),
        match,
    )


# --------------------------------------------------------------------------- #
# Per-opponent play — one opponent -> one edge. Shared by the Stage C loop
# (:func:`_stage_c_deep`) and the resumable parts path (:func:`run_eval_part`).
# --------------------------------------------------------------------------- #
def _build_sealbot_edge_from_match(
    cfg: MultiStageEvalSection,
    roster: Roster,
    sb_match: dict[str, Any],
    sb_games: int,
) -> tuple[dict[str, Any], list[float]]:
    """Build the SealBot (down-weighted) edge dict + win-rate Wilson CI from an
    already-played SealBot match.

    Shared by :func:`_play_sealbot_opponent` and the concurrent path
    (:func:`run_multistage_eval_concurrent`). Returns
    ``(edge, sealbot_winrate_ci)``.
    """

    wa, wb, n_eff, prov = _sealbot_edge(sb_match, cfg.sealbot_overdispersion)
    if isinstance(prov, dict):
        # Which searcher the CANDIDATE side used against SealBot ("selfplay"
        # = the run's own as-trained / Gumbel profile, mirroring the
        # checkpoint-edge vocabulary). Persisted into the pool row for audit.
        prov = {**prov, "candidate_search_profile": "selfplay"}
    decided = int((sb_match.get("score") or {}).get("decided", 0) or 0)
    wins = int((sb_match.get("score") or {}).get("a_wins", 0) or 0)
    lo, hi = eval_stats.wilson_ci(wins, decided) if decided else (0.0, 1.0)
    sealbot_ci = [round(lo, 4), round(hi, 4)]
    edge = {
        "role": "sealbot",
        "opponent": SEALBOT_LABEL,
        "bt": eval_stats.BTEdge(
            a=roster.candidate_label, b=SEALBOT_LABEL,
            wins_a=wa, wins_b=wb, weight=max(0.0, min(cfg.sealbot_overdispersion, 1.0)),
        ),
        "descriptive": {
            "opponent": SEALBOT_LABEL,
            "role": "sealbot",
            "kind": "sealbot",
            "primary": False,
            "paired": False,
            "games_requested": sb_games,
            "decided": decided,
            "winrate": round(wins / decided, 4) if decided else None,
            "winrate_ci95": sealbot_ci,
            "elo_point": _safe_elo(wins / decided) if decided else None,
            "down_weight": round(max(0.0, min(cfg.sealbot_overdispersion, 1.0)), 4),
            "note": (
                "DESCRIPTIVE zero-point. SealBot depth varies under load; "
                "this edge is down-weighted out of difference inference and "
                "only pins the rating scale at 0 Elo."
            ),
            "provenance": prov,
        },
    }
    return edge, sealbot_ci


def _build_checkpoint_edge_from_match(
    roster: Roster,
    opp: Opponent,
    match: dict[str, Any],
    *,
    reused: int = 0,
    cfg: MultiStageEvalSection | None = None,
    opponent_search_profile: str | None = None,
) -> dict[str, Any]:
    """Build one checkpoint opponent's descriptive + BT edge dict from an
    already-played paired match (candidate is net A).

    Shared by :func:`_play_checkpoint_opponent` and
    :func:`run_multistage_eval_concurrent`.
    """

    is_champ = opp.role == "champion"
    # Tag whether this opponent was featurized out-of-distribution (a radius-8
    # native anchor featurized at the live radius). Descriptive only.
    live_radius = _live_featurize_radius()
    featurized_ood = (
        _opponent_featurized_ood(opp.label, cfg, live_radius) if cfg is not None else False
    )
    wa, wb, n_eff, prov = _checkpoint_edge_counts(match)
    # Carry the radius annotation into provenance so it lands in the persisted
    # pool row (_edge_pool_row copies descriptive["provenance"] into raw).
    if isinstance(prov, dict):
        prov = {**prov, "featurized_ood": featurized_ood, "featurize_radius": live_radius}
        # Which searcher the OPPONENT side used ("selfplay" — a current-arch
        # checkpoint mirrors the candidate's Gumbel profile). Persisted into
        # the pool row for audit.
        if opponent_search_profile is not None:
            prov = {**prov, "opponent_search_profile": opponent_search_profile}
    score = match.get("score") or {}
    decided = int(score.get("decided", 0) or 0)
    wins = int(score.get("a_wins", 0) or 0)
    paired = _pentanomial_to_paired_result(match.get("pentanomial") or {})
    if paired is not None and paired.n_pairs > 0 and math.isfinite(paired.se):
        wr_lo, wr_hi = paired.ci()
        elo_lo, elo_hi = eval_stats.elo_ci_from_winrate(paired.win_rate, paired.se)
        winrate = round(paired.win_rate, 4)
    else:
        wr_lo, wr_hi = (eval_stats.wilson_ci(wins, decided) if decided else (0.0, 1.0))
        elo_lo, elo_hi = (_safe_elo(wr_lo), _safe_elo(wr_hi))
        winrate = round(wins / decided, 4) if decided else None
    return {
        "role": opp.role,
        "opponent": opp.label,
        "bt": eval_stats.BTEdge(
            a=roster.candidate_label, b=opp.label, wins_a=wa, wins_b=wb, weight=1.0
        ),
        "descriptive": {
            "opponent": opp.label,
            "role": opp.role,
            "kind": "checkpoint",
            "primary": is_champ,
            "paired": True,
            "decided": decided,
            "winrate": winrate,
            "winrate_ci95": [round(wr_lo, 4), round(wr_hi, 4)],
            "elo_point": _safe_elo(winrate) if winrate is not None else None,
            "elo_ci95_pairlevel": [_round_elo(elo_lo), _round_elo(elo_hi)],
            "reused_sprt_games": reused if is_champ else 0,
            # Radius annotation — flows into eval_pool.json rows (via provenance
            # copy) and per-epoch edges so a reader can flag OOD edges.
            "featurized_ood": featurized_ood,
            "featurize_radius": live_radius,
            "provenance": prov,
            "note": (
                "PRIMARY edge — verdict via the pooled BT difference-CI in "
                "Stage D (carries the Cov_LB term)."
                if is_champ
                else "DESCRIPTIVE edge — Wilson/Elo CIs only, no significance verdict."
            ),
        },
    }


def _play_sealbot_opponent(
    cfg: MultiStageEvalSection,
    roster: Roster,
    candidate_ckpt: Path,
    full_cfg: ShrimpConfig,
    sb_games: int,
    *,
    play_sealbot_match: Callable[..., dict[str, Any]],
    diagnostics_dir: Path,
) -> tuple[dict[str, Any] | None, list[float] | None, str | None]:
    """Play the SealBot zero-point pairing and build its (down-weighted) edge.

    Returns ``(edge_dict | None, sealbot_winrate_ci | None, unavailable | None)``.
    Fail-open: a SealBot runner exception (extension not built, adapter import
    fails, worker dies) is caught here and surfaced as the third tuple element so
    the caller can drop just this edge and continue.
    """

    sb_match: dict[str, Any] | None = None
    try:
        sb_match = play_sealbot_match(
            str(candidate_ckpt),
            sb_games,
            config=full_cfg,
            label=roster.candidate_label,
            sealbot_variant=cfg.opponents.sealbot_variant,
            sealbot_time_limit=cfg.opponents.sealbot_time_limit,
            sealbot_path=cfg.opponents.sealbot_path,
            visits=_eval_visits(cfg, full_cfg),
            virtual_batch_size=_eval_virtual_batch_size(cfg, full_cfg),
            opening_plies=cfg.opening_plies,
            opening_temperature=cfg.opening_temperature,
            # AS-TRAINED SEARCHER: the candidate plays SealBot with its own
            # gumbel eval profile — the same searcher as the lineage edges.
            # The candidate count is budget-calibrated in-tree (init_gumbel_root
            # shrinks gumbel_m down the halving ladder until SH round-0 affords
            # >= GUMBEL_MIN_ROUND0_VISITS look-aheads per candidate), so the
            # SealBot zero-point is measured at full search depth. The
            # candidate_search_profile provenance below records the profile.
            diagnostics_dir=str(diagnostics_dir),
        )
    except Exception as exc:  # noqa: BLE001 — fail-open at the opponent boundary.
        # Broad by intent: a missing hexo_engine raises ModuleNotFoundError, an
        # unbuilt extension or mid-match worker death raises
        # SealBotUnavailableError. These are specific to the SealBot opponent and
        # do not touch the candidate/checkpoint path.
        return None, None, f"{type(exc).__name__}: {exc}"
    if sb_match is None:
        return None, None, None

    edge, sealbot_ci = _build_sealbot_edge_from_match(cfg, roster, sb_match, sb_games)
    return edge, sealbot_ci, None


def _play_checkpoint_opponent(
    cfg: MultiStageEvalSection,
    roster: Roster,
    candidate_ckpt: Path,
    full_cfg: ShrimpConfig,
    opp: Opponent,
    per: int,
    *,
    play_checkpoint_match: Callable[..., dict[str, Any]],
    diagnostics_dir: Path,
    reuse_champion_match: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Play one checkpoint pairing (paired/CRN) and build its descriptive edge.

    Returns the edge dict (candidate as net A) or ``None`` when no match could be
    produced (e.g. zero budget and nothing to reuse). The champion pairing reuses
    ``reuse_champion_match`` (the SPRT block's games) and tops up to ``per``.
    Every opponent is a current-arch shrimp checkpoint, so it plays the
    candidate's own (self-play / Gumbel) eval profile.
    """

    if opp.ckpt is None:
        return None
    # Reuse the SPRT champion match (same opponent, same protocol); top up to the
    # per-pairing budget if needed.
    is_champ = opp.role == "champion"
    match: dict[str, Any] | None = None
    reused = 0
    if is_champ and reuse_champion_match is not None:
        match = reuse_champion_match
        reused = int((reuse_champion_match.get("meta") or {}).get("games_requested", 0) or 0)
    topup = max(per - reused, 0) if is_champ else per
    if topup > 0:
        fresh = play_checkpoint_match(
            str(candidate_ckpt),
            str(opp.ckpt),
            topup,
            config=full_cfg,
            label_a=roster.candidate_label,
            label_b=opp.label,
            paired_openings=True,
            visits=_eval_visits(cfg, full_cfg),
            virtual_batch_size=_eval_virtual_batch_size(cfg, full_cfg),
            opening_plies=cfg.opening_plies,
            opening_temperature=cfg.opening_temperature,
            diagnostics_dir=str(diagnostics_dir),
            # Offset the topup's CRN seeds from the SPRT batch so the added pairs
            # are distinct openings.
            game_seed_base=1_000_000 + (opp.epoch or 0),
        )
        match = _merge_matches(match, fresh) if match is not None else fresh
    if match is None:
        return None
    return _build_checkpoint_edge_from_match(
        roster, opp, match, reused=reused, cfg=cfg,
        opponent_search_profile="selfplay",
    )


# --------------------------------------------------------------------------- #
# Stage C — deep paired eval.
# --------------------------------------------------------------------------- #
def _stage_c_deep(
    cfg: MultiStageEvalSection,
    roster: Roster,
    candidate_ckpt: Path,
    full_cfg: ShrimpConfig,
    *,
    play_checkpoint_match: Callable[..., dict[str, Any]],
    play_sealbot_match: Callable[..., dict[str, Any]],
    diagnostics_dir: Path,
    reuse_champion_match: dict[str, Any] | None,
) -> tuple[StageResult, list[dict[str, Any]], list[float] | None]:
    """Play the budget across all pairings; return per-edge BT inputs + descriptions.

    Each returned edge dict has:
      * ``bt``: an :class:`eval_stats.BTEdge` (effective counts, weight) for the
        pool — candidate as ``a``.
      * ``descriptive``: a JSON-able block with the edge's Wilson/Elo CIs and
        provenance. The primary edge's significance comes from the pooled BT
        difference-CI in Stage D, not a per-edge test.
      * ``role`` / ``opponent``: bookkeeping.

    The SealBot edge (unpaired) carries its down-weight; its win-rate Wilson CI
    is also returned separately. Checkpoint edges are paired and deflated to
    effective counts. Budget per the :func:`allocate_budget` split.
    """

    n_ckpt = len(roster.opponents)
    has_sealbot = roster.sealbot is not None
    alloc = allocate_budget(
        cfg.games_budget, n_checkpoint_opponents=n_ckpt, has_sealbot=has_sealbot,
        sealbot_share=cfg.sealbot_share,
    )

    edges: list[dict[str, Any]] = []
    played: dict[str, Any] = {}
    sealbot_ci: list[float] | None = None
    # Fail-open per opponent: if the SealBot edge cannot be played, drop just that
    # edge, record why here, and continue. The pool then anchors on a checkpoint
    # (see _choose_anchor).
    sealbot_unavailable: str | None = None

    # ----- SealBot zero-point edge (concurrent, unpaired). -----
    if has_sealbot and alloc.get(SEALBOT_LABEL, 0) > 0:
        sb_games = alloc[SEALBOT_LABEL]
        sb_edge, sb_ci, sb_unavail = _play_sealbot_opponent(
            cfg, roster, candidate_ckpt, full_cfg, sb_games,
            play_sealbot_match=play_sealbot_match,
            diagnostics_dir=diagnostics_dir,
        )
        if sb_unavail is not None:
            sealbot_unavailable = sb_unavail
        if sb_edge is not None:
            sealbot_ci = sb_ci
            edges.append(sb_edge)
            # ``played`` feeds only the ``opponents_played`` key list in the stage
            # detail, so presence by label is sufficient.
            played[SEALBOT_LABEL] = {"played": True}

    # ----- Checkpoint pairings (paired, CRN). -----
    per = alloc.get("per_checkpoint", 0)
    for opp in roster.opponents:
        if opp.ckpt is None:
            continue
        edge = _play_checkpoint_opponent(
            cfg, roster, candidate_ckpt, full_cfg, opp, per,
            play_checkpoint_match=play_checkpoint_match,
            diagnostics_dir=diagnostics_dir,
            reuse_champion_match=reuse_champion_match if opp.role == "champion" else None,
        )
        if edge is None:
            continue
        edges.append(edge)
        played[opp.label] = {"played": True}

    # Stage C is "completed" as long as any edge was produced; a dropped SealBot
    # edge does not downgrade the stage. The drop reason is in the detail.
    status = "completed" if edges else "empty"
    detail: dict[str, Any] = {
        "budget": cfg.games_budget,
        "allocation": alloc,
        "n_edges": len(edges),
        "opponents_played": list(played.keys()),
    }
    if sealbot_unavailable is not None:
        detail["sealbot_unavailable"] = sealbot_unavailable
    return (
        StageResult(
            stage="C_deep",
            status=status,
            detail=detail,
        ),
        edges,
        sealbot_ci,
    )


# --------------------------------------------------------------------------- #
# Stage D — rolling Bradley-Terry pool + verdict.
# --------------------------------------------------------------------------- #
def _edge_pool_row(epoch_tag: int, edge: dict[str, Any]) -> dict[str, Any]:
    """Project one Stage-C edge dict into its persisted pool row.

    The pool's append-only row shape, shared by the Stage D append and the
    resumable :func:`run_eval_part` append.
    """

    bt: eval_stats.BTEdge = edge["bt"]
    return {
        "epoch": epoch_tag,
        "a": bt.a,
        "b": bt.b,
        "wins_a": round(float(bt.wins_a), 6),
        "wins_b": round(float(bt.wins_b), 6),
        "weight": round(float(bt.weight), 6),
        "kind": edge["descriptive"].get("kind", "checkpoint"),
        "raw": edge["descriptive"].get("provenance", {}),
    }


def _stage_d_pool(
    cfg: MultiStageEvalSection,
    roster: Roster,
    edges: list[dict[str, Any]],
    run_dir: Path,
    *,
    pool_doc: dict[str, Any] | None = None,
    append: bool = True,
    sealbot_expected_but_unavailable: str | None = None,
) -> tuple[StageResult, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Append this epoch's edges to the pool, refit BT, derive the verdict.

    Returns (StageResult, ratings_table, verdict_block, updated_pool_doc). The BT
    fit (eval_stats.bradley_terry) is CONVERGED (the function asserts
    ``max|grad| < bt_grad_tol`` internally) before the covariance is computed;
    SealBot is the pinned anchor and its edges carry the down-weight already
    baked into each BTEdge.weight.

    The ratings table reports every pooled player's Elo + marginal 95% CI
    (descriptive). The verdict block holds the primary hypothesis: the BT
    difference-CI ``r_candidate - r_champion`` -> PROMOTE / REGRESS /
    INCONCLUSIVE. All else is descriptive.

    Parts path: pass a pre-loaded ``pool_doc`` and ``append=False`` (the parts
    appended their own rows incrementally) for an aggregate fit over the whole
    pool. The default ``pool_doc=None, append=True`` loads the pool, appends this
    epoch's ``edges``, and refits.
    """

    pool_path = _pool_path(run_dir, cfg)
    if pool_doc is None:
        pool_doc = _load_pool(pool_path)

    # Append THIS epoch's edges (effective counts + weight + provenance).
    epoch_tag = roster.candidate_epoch if roster.candidate_epoch is not None else 0
    if append:
        for e in edges:
            pool_doc["edges"].append(_edge_pool_row(epoch_tag, e))

    bt_edges = _bt_edges_from_pool(pool_doc)
    ratings: dict[str, Any] = {"anchor": SEALBOT_LABEL, "players": [], "fit": {}}
    verdict_block: dict[str, Any] = {
        "label": "INCONCLUSIVE",
        "primary": None,
    }

    # The BT fit needs the anchor to appear in an edge. If SealBot is
    # disabled/never played, anchor on the lowest permanent anchor instead so the
    # pool still has a fixed zero-point (it then floats relative to that anchor).
    # Opponents featurized out-of-distribution (a radius-8 native anchor under a
    # non-8 live radius) are excluded from the pinned zero-point (but stay
    # descriptive). Restricted to labels that appear in an edge.
    live_radius = _live_featurize_radius()
    edge_labels = {lbl for e in bt_edges for lbl in (e.a, e.b)}
    ood_labels = {
        o.label
        for o in roster.opponents
        if o.label in edge_labels and _opponent_featurized_ood(o.label, cfg, live_radius)
    }
    anchor_label = _choose_anchor(bt_edges, roster, ood_labels=ood_labels)

    # When SealBot was expected (config-enabled) but unavailable, the anchor
    # re-pins to bc_prefit / the lowest checkpoint, shifting every absolute Elo.
    # Detect that substitution and mark Stage D degraded. A config-disabled
    # SealBot is not flagged (sealbot_expected_but_unavailable is None then).
    sealbot_substituted = bool(
        sealbot_expected_but_unavailable is not None
        and anchor_label is not None
        and anchor_label != SEALBOT_LABEL
    )

    fit = None
    fit_error: str | None = None
    if bt_edges and anchor_label is not None and _anchor_in_edges(anchor_label, bt_edges):
        try:
            fit = eval_stats.bradley_terry(
                bt_edges,
                anchor=anchor_label,
                grad_tol=cfg.bt_grad_tol,
                max_iter=cfg.bt_max_iters,
            )
        except (AssertionError, ValueError, RuntimeError) as exc:
            fit_error = f"{type(exc).__name__}: {exc}"

    if fit is not None:
        for label in sorted(fit.players, key=lambda lbl: -fit.rating(lbl)):
            lo, hi = fit.elo_ci(label)
            ratings["players"].append(
                {
                    "label": label,
                    "elo": round(fit.rating(label), 1),
                    "elo_ci95": [round(lo, 1), round(hi, 1)],
                    "se_elo": round(fit.se(label), 1),
                    "is_anchor": label == anchor_label,
                }
            )
        ratings["fit"] = {
            "anchor": anchor_label,
            "anchor_is_sealbot": anchor_label == SEALBOT_LABEL,
            "max_grad": fit.max_grad,
            "iterations": fit.iterations,
            "converged": fit.converged,
            "n_edges": len(bt_edges),
            "n_players": len(fit.players),
            # Radius-8-native opponents featurized at the live radius. Kept out
            # of the pinned anchor; their cross-lineage edges are not a clean
            # strength signal.
            "ood_opponents": sorted(ood_labels),
            "featurize_radius": live_radius,
            "note": (
                "SealBot pinned at 0 Elo (zero-point); its edges are down-weighted "
                "out of difference inference."
                if anchor_label == SEALBOT_LABEL
                else f"SealBot unavailable; pool anchored on {anchor_label} (floats vs that anchor)."
            ),
        }

        # ----- PRIMARY hypothesis: candidate vs prior champion. -----
        if (
            roster.champion is not None
            and roster.candidate_label in fit.players
            and roster.champion.label in fit.players
        ):
            d = fit.diff(roster.candidate_label, roster.champion.label)
            lo, hi = fit.diff_ci(roster.candidate_label, roster.champion.label)
            se = fit.se_diff(roster.candidate_label, roster.champion.label)
            # One edge gates by default, so alpha is unadjusted unless
            # bonferroni_correction is set.
            alpha = (
                eval_stats.bonferroni_alpha(cfg.primary_alpha, _n_gating_edges(roster))
                if cfg.bonferroni_correction
                else cfg.primary_alpha
            )
            # Verdict from the difference CI, using the configured
            # promote/regress thresholds (both default to 0).
            label = _verdict_with_thresholds(
                lo, hi,
                promote_elo=cfg.promote_elo_threshold,
                regress_elo=cfg.regress_elo_threshold,
            )
            verdict_block = {
                "label": label,
                "primary": {
                    "candidate": roster.candidate_label,
                    "champion": roster.champion.label,
                    "elo_diff": round(d, 1),
                    "elo_diff_ci95": [round(lo, 1), round(hi, 1)],
                    "se_elo": round(se, 1),
                    "promote_threshold_elo": cfg.promote_elo_threshold,
                    "regress_threshold_elo": cfg.regress_elo_threshold,
                    "alpha": alpha,
                    "hypothesis": "r_candidate - r_champion (pooled BT difference, incl. Cov_LB)",
                },
                "note": (
                    "PURE EVAL: this label is reported only and gates nothing. The "
                    "candidate-vs-champion verdict is PERMANENTLY single-epoch-limited "
                    "(a fresh candidate node each epoch never compounds): SE(r_L-r_B) "
                    "~120-140 Elo, resolving ~250-300 Elo — a gross-regression "
                    "tripwire, NOT a fine-edge test, and it does not tighten with more "
                    "epochs. Only the FIXED-anchor DESCRIPTIVE curve (bc_prefit/ep5/"
                    "SealBot) compounds toward ~15-20 Elo over many epochs."
                ),
            }
        else:
            verdict_block["note"] = (
                "No primary hypothesis: prior champion absent from the pool "
                "(first eligible epoch or champion played no edge)."
            )
    else:
        ratings["fit"] = {
            "anchor": anchor_label,
            "converged": False,
            "n_edges": len(bt_edges),
            "error": fit_error or "no usable edges / no anchor in pool",
        }
        verdict_block["note"] = (
            "BT fit unavailable (" + (fit_error or "no anchor edge") + "); verdict INCONCLUSIVE."
        )

    # When any OOD-featurized opponent is in the pool, flag it on the verdict.
    # Descriptive only; does not change the same-lineage primary verdict.
    if ood_labels:
        verdict_block["ood_opponents"] = sorted(ood_labels)
        verdict_block["ood_note"] = (
            "Radius-8-era opponents %s are featurized at radius %d (OOD): they play "
            "weaker than their true strength, inflating the candidate's relative Elo "
            "against them. They are EXCLUDED from the pinned anchor and the cross-"
            "lineage curve is NOT a clean strength signal."
            % (sorted(ood_labels), live_radius)
        )

    # Surface a SealBot substitution as a degraded verdict with machine flags,
    # even when the BT fit converged on the substituted anchor.
    if sealbot_substituted:
        verdict_block["anchor_substituted"] = True
        verdict_block["substituted_from"] = SEALBOT_LABEL
        verdict_block["substituted_to"] = anchor_label
        verdict_block["sealbot_unavailable_reason"] = sealbot_expected_but_unavailable
        verdict_block["degraded"] = True
        verdict_block["degraded_note"] = (
            "SealBot was expected but unavailable (%s); the BT zero-point re-pinned "
            "to %r, shifting every ABSOLUTE Elo. Difference verdicts between same-"
            "lineage nets are unaffected, but absolute placements are NOT calibrated."
            % (sealbot_expected_but_unavailable, anchor_label)
        )
        _EVAL_LOG.warning(
            "SealBot expected but unavailable (%s); anchor substituted to %r — "
            "Stage-D marked degraded",
            sealbot_expected_but_unavailable,
            anchor_label,
        )

    if fit is not None:
        status = "degraded" if sealbot_substituted else "completed"
    else:
        status = "degraded"
    return (
        StageResult(
            stage="D_pool",
            status=status,
            detail={
                "pool_path": str(pool_path),
                "pool_edges_total": len(pool_doc["edges"]),
                "bt_edges_aggregated": len(bt_edges),
                "anchor": anchor_label,
                "converged": bool(fit is not None),
                "sealbot_substituted": sealbot_substituted,
                "sealbot_unavailable_reason": sealbot_expected_but_unavailable,
            },
        ),
        ratings,
        verdict_block,
        pool_doc,
    )


# --------------------------------------------------------------------------- #
# RUN-IN-PARTS — resumable per-opponent chunks.
#
# Runs the eval as a sequence of parts. Each part plays one opponent's games
# (SealBot, or one checkpoint opponent) and appends its edge to the persisted
# pool (diagnostics/eval_pool.json) immediately, so an interruption keeps every
# completed part. A part whose edge for this candidate epoch is already in the
# pool is skipped (resume). After all parts, the aggregate BT fit + verdict runs
# over the whole pool (Stage D, no re-append). Same verdict logic and result
# shape as the monolithic path.
# --------------------------------------------------------------------------- #
def _epoch_edge_exists(
    pool_doc: dict[str, Any], epoch_tag: int, cand_label: str, opp_label: str
) -> bool:
    """True if this candidate epoch's edge vs ``opp_label`` is already pooled.

    The resume predicate. The candidate is net A in a Stage-C edge
    (``BTEdge.a == candidate_label``), so rows are keyed by
    ``(epoch, a==cand_label, b==opp_label)``; the reversed direction is also
    accepted. Matching is exact on the integer epoch tag, so the same opponent in
    a new epoch is a distinct row.
    """

    for row in pool_doc.get("edges", []):
        try:
            if int(row.get("epoch")) != int(epoch_tag):
                continue
        except (TypeError, ValueError):
            continue
        a = str(row.get("a"))
        b = str(row.get("b"))
        if (a == cand_label and b == opp_label) or (a == opp_label and b == cand_label):
            return True
    return False


def _part_opponents(roster: Roster) -> list[Opponent]:
    """Ordered list of parts for a candidate: SealBot first (if enabled), then
    every checkpoint opponent (anchors + bracket + champion), one part each.

    The SealBot part is the roster's ``sealbot`` pseudo-opponent (ckpt None);
    checkpoint parts are the resolved ``roster.opponents`` with a real ckpt. Same
    opponent set as Stage C, run one at a time with persistence between them.
    """

    parts: list[Opponent] = []
    if roster.sealbot is not None:
        parts.append(roster.sealbot)
    parts.extend(o for o in roster.opponents if o.ckpt is not None)
    return parts


def _resolve_part(roster: Roster, part_label: str) -> Opponent | None:
    """Find the part opponent matching ``part_label`` (case-exact), or None."""

    for opp in _part_opponents(roster):
        if opp.label == part_label:
            return opp
    return None


def run_eval_part(
    run_dir: str | Path,
    candidate_ckpt: str | Path,
    part_label: str,
    config: ShrimpConfig | MultiStageEvalSection | None = None,
    *,
    candidate_epoch: int | None = None,
    candidate_label: str | None = None,
    checkpoints_dir: str | Path | None = None,
    diagnostics_dir: str | Path | None = None,
    write_pool: bool = True,
    resume: bool = True,
    pool_doc: dict[str, Any] | None = None,
    reuse_champion_match: dict[str, Any] | None = None,
    play_checkpoint_match: Callable[..., dict[str, Any]] | None = None,
    play_sealbot_match: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Play one part (one opponent's games) and append its edge to the pool.

    Resolves the roster + the same :func:`allocate_budget` split Stage C uses,
    plays the one opponent named by ``part_label`` (``"sealbot"`` or a checkpoint
    label), and on success appends the single edge row to
    ``diagnostics/eval_pool.json`` and persists it atomically. With
    ``resume=True`` a part whose edge for this candidate epoch is already pooled
    is skipped (plays no games). Fail-open: a SealBot part that raises is recorded
    as ``status="unavailable"`` and the pool is left untouched. Writes only the
    pool JSON under diagnostics.

    ``pool_doc`` is an optional in-memory pool accumulator used by both the resume
    check and the append: the orchestrator threads one shared doc across parts in
    a pure (``write_pool=False``) run so the aggregate can fit the in-memory edges
    without touching disk. When ``None`` the pool is loaded from disk. The
    appended row goes into whichever doc is in play; ``write_pool`` additionally
    persists it to disk. The (possibly newly-loaded) doc is returned under
    ``"pool_doc"``.

    Returns a status dict: ``{"part", "status", "epoch", "pool_doc", "edge"?,
    "reason"?, ...}`` where ``status`` is one of ``played`` / ``skipped`` /
    ``unavailable`` / ``empty`` / ``unknown_part``.
    """

    run_dir = Path(run_dir)
    candidate_ckpt = Path(candidate_ckpt)
    cfg = _coerce_section(config)
    _assert_no_run_mutation(cfg)
    full_cfg = config if isinstance(config, ShrimpConfig) else parse_shrimp_config({})
    diag_dir = Path(diagnostics_dir) if diagnostics_dir is not None else (run_dir / "diagnostics")

    if play_checkpoint_match is None or play_sealbot_match is None:
        from . import eval_arena as _arena

        if play_checkpoint_match is None:
            play_checkpoint_match = _arena.play_checkpoint_match
        if play_sealbot_match is None:
            play_sealbot_match = _arena.play_sealbot_match

    roster = select_opponents(
        run_dir, candidate_ckpt, cfg,
        candidate_epoch=candidate_epoch, candidate_label=candidate_label,
        checkpoints_dir=checkpoints_dir,
    )
    cand_label = roster.candidate_label
    epoch_tag = roster.candidate_epoch if roster.candidate_epoch is not None else 0

    opp = _resolve_part(roster, part_label)
    if opp is None:
        return {
            "part": part_label,
            "status": "unknown_part",
            "epoch": epoch_tag,
            "available_parts": [o.label for o in _part_opponents(roster)],
            "reason": f"no part labelled {part_label!r} in the roster",
        }

    pool_path = _pool_path(run_dir, cfg)
    # Use the injected in-memory pool when given; otherwise load from disk. Either
    # way ``pool_doc`` is the accumulator the skip-check and the append act on.
    if pool_doc is None:
        pool_doc = _load_pool(pool_path)

    # Resume: skip a part whose edge for this epoch is already pooled.
    if resume and _epoch_edge_exists(pool_doc, epoch_tag, cand_label, opp.label):
        return {
            "part": opp.label,
            "status": "skipped",
            "epoch": epoch_tag,
            "reason": "edge already in pool",
            "pool_doc": pool_doc,
        }

    alloc = allocate_budget(
        cfg.games_budget,
        n_checkpoint_opponents=len(roster.opponents),
        has_sealbot=roster.sealbot is not None,
        sealbot_share=cfg.sealbot_share,
    )

    # ----- Play the one opponent (the same helpers Stage C uses). -----
    if opp.role == "sealbot":
        sb_games = alloc.get(SEALBOT_LABEL, 0)
        if sb_games <= 0:
            return {"part": opp.label, "status": "empty", "epoch": epoch_tag,
                    "reason": "zero SealBot budget", "pool_doc": pool_doc}
        edge, _sb_ci, unavail = _play_sealbot_opponent(
            cfg, roster, candidate_ckpt, full_cfg, sb_games,
            play_sealbot_match=play_sealbot_match,
            diagnostics_dir=diag_dir,
        )
        if unavail is not None:
            return {"part": opp.label, "status": "unavailable", "epoch": epoch_tag,
                    "reason": unavail, "pool_doc": pool_doc}
    else:
        per = alloc.get("per_checkpoint", 0)
        edge = _play_checkpoint_opponent(
            cfg, roster, candidate_ckpt, full_cfg, opp, per,
            play_checkpoint_match=play_checkpoint_match,
            diagnostics_dir=diag_dir,
            reuse_champion_match=reuse_champion_match if opp.role == "champion" else None,
        )

    if edge is None:
        return {"part": opp.label, "status": "empty", "epoch": epoch_tag,
                "reason": "no informative match produced", "pool_doc": pool_doc}

    # ----- Durably append the single edge row and persist immediately. -----
    row = _edge_pool_row(epoch_tag, edge)
    pool_doc["edges"].append(row)
    if write_pool:
        _save_pool(pool_path, pool_doc)

    return {
        "part": opp.label,
        "status": "played",
        "epoch": epoch_tag,
        "role": opp.role,
        "edge": edge["descriptive"],
        "pool_path": str(pool_path) if write_pool else None,
        "pool_edges_total": len(pool_doc["edges"]),
        "pool_doc": pool_doc,
    }


def aggregate_pool(
    run_dir: str | Path,
    candidate_ckpt: str | Path,
    config: ShrimpConfig | MultiStageEvalSection | None = None,
    *,
    candidate_epoch: int | None = None,
    candidate_label: str | None = None,
    checkpoints_dir: str | Path | None = None,
    diagnostics_dir: str | Path | None = None,
    write_diagnostics: bool = True,
    pool_doc: dict[str, Any] | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Fit the BT pool + emit the verdict over the already-persisted edges.

    The final pass after all parts have run: loads ``diagnostics/eval_pool.json``
    and runs only Stage D over it (no games played, no re-append). Produces the
    same ratings table + verdict block + per-epoch diagnostics JSON
    ``run_multistage_eval`` does. Idempotent: running it repeatedly only reads the
    pool to fit; it never plays or appends.

    ``pool_doc`` injects an in-memory pool to fit instead of the on-disk one. When
    ``None`` the pool is loaded from disk.

    Returns the same report dict shape as :func:`run_multistage_eval` (``meta`` /
    ``roster`` / ``stages`` [only D] / ``ratings`` / ``edges`` / ``verdict``).
    """

    run_dir = Path(run_dir)
    candidate_ckpt = Path(candidate_ckpt)
    cfg = _coerce_section(config)
    _assert_no_run_mutation(cfg)
    diag_dir = Path(diagnostics_dir) if diagnostics_dir is not None else (run_dir / "diagnostics")
    started = now()

    roster = select_opponents(
        run_dir, candidate_ckpt, cfg,
        candidate_epoch=candidate_epoch, candidate_label=candidate_label,
        checkpoints_dir=checkpoints_dir,
    )
    cand_label = roster.candidate_label
    epoch_tag = roster.candidate_epoch if roster.candidate_epoch is not None else 0

    # Fit over the injected in-memory pool if given, else the persisted one — with
    # no re-append (the parts already appended their rows).
    if pool_doc is None:
        pool_doc = _load_pool(_pool_path(run_dir, cfg))
    # No live stage_c_detail in the aggregate path, so SealBot unavailability is
    # inferred from the pool: config-enabled SealBot with no SealBot edge appended
    # for this epoch means the SealBot part did not complete.
    sealbot_expected_but_unavailable = None
    if roster.sealbot is not None and not _epoch_has_sealbot_edge(
        pool_doc, epoch_tag, cand_label
    ):
        sealbot_expected_but_unavailable = (
            "SealBot edge absent from pool for this epoch (part did not complete)"
        )
    stage_d, ratings, verdict_block, pool_doc = _stage_d_pool(
        cfg, roster, [], run_dir, pool_doc=pool_doc, append=False,
        sealbot_expected_but_unavailable=sealbot_expected_but_unavailable,
    )

    # The SealBot win-rate read, recovered from the pooled SealBot edge (if any)
    # for THIS epoch, so the aggregate report carries the same descriptive field.
    sealbot_ci = _sealbot_ci_from_pool(pool_doc, epoch_tag, cand_label)
    # This-epoch descriptive edges, recovered from the pool's provenance rows.
    epoch_edges = _epoch_descriptive_edges(pool_doc, epoch_tag, cand_label, roster)

    report: dict[str, Any] = {
        "meta": {
            "kind": "shrimp.multistage_eval",
            "mode": "aggregate_pool",
            "run_dir": str(run_dir),
            "candidate_ckpt": str(candidate_ckpt),
            "candidate_label": cand_label,
            "candidate_epoch": roster.candidate_epoch,
            "anchor": SEALBOT_LABEL,
            "config": _config_summary(cfg),
            "elapsed_seconds": round(now() - started, 2),
            "single_epoch_se_elo_note": _resolution_note(cfg, epoch_edges, roster),
            "pure_eval": True,
            "gating_enabled": cfg.eval_gating_enabled,
            "promotion_enabled": cfg.eval_promotion_enabled,
        },
        "roster": _roster_summary(roster),
        "stages": [{"stage": stage_d.stage, "status": stage_d.status, **stage_d.detail}],
        "ratings": ratings,
        "edges": [e["descriptive"] for e in epoch_edges],
        "sealbot_winrate_ci95": sealbot_ci,
        "verdict": verdict_block,
    }

    if write_diagnostics:
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag_path = diag_dir / f"{DIAG_PREFIX}{epoch_tag:06d}.json"
        diag_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["meta"]["diagnostics_path"] = str(diag_path)

    return report


def _epoch_has_sealbot_edge(
    pool_doc: dict[str, Any], epoch_tag: int, cand_label: str
) -> bool:
    """True if a SealBot edge for this candidate epoch is present in the pool.

    Used by the parts/aggregate path to infer SealBot unavailability: a
    config-enabled SealBot with no edge appended this epoch means its part did not
    complete.
    """

    for row in pool_doc.get("edges", []):
        try:
            if int(row.get("epoch")) != int(epoch_tag):
                continue
        except (TypeError, ValueError):
            continue
        if row.get("kind") != "sealbot":
            continue
        if str(row.get("a")) == cand_label or str(row.get("b")) == cand_label:
            return True
    return False


def _sealbot_ci_from_pool(
    pool_doc: dict[str, Any], epoch_tag: int, cand_label: str
) -> list[float] | None:
    """Recover this epoch's SealBot win-rate Wilson CI from the pooled edge's
    provenance (the aggregate report surfaces the same descriptive field the
    monolithic path returns). None when no SealBot edge was pooled this epoch."""

    for row in pool_doc.get("edges", []):
        try:
            if int(row.get("epoch")) != int(epoch_tag):
                continue
        except (TypeError, ValueError):
            continue
        if row.get("kind") != "sealbot":
            continue
        if str(row.get("a")) != cand_label and str(row.get("b")) != cand_label:
            continue
        raw = row.get("raw") or {}
        wins = int(raw.get("physical_wins_cand", 0) or 0)
        losses = int(raw.get("physical_wins_sealbot", 0) or 0)
        decided = wins + losses
        if decided <= 0:
            return None
        lo, hi = eval_stats.wilson_ci(wins, decided)
        return [round(lo, 4), round(hi, 4)]
    return None


def _epoch_descriptive_edges(
    pool_doc: dict[str, Any], epoch_tag: int, cand_label: str, roster: Roster
) -> list[dict[str, Any]]:
    """Rebuild this epoch's descriptive edge blocks from the persisted pool rows.

    The aggregate report's ``edges`` list: one descriptive block per opponent
    played this epoch, reconstructed from the row's ``raw`` provenance (physical
    counts, pentanomial, pair winrate/SE, eval_visits). The primary flag is
    re-derived from the roster's champion. Descriptive output only; the verdict
    comes from the pooled BT fit, not these blocks.
    """

    champ_label = roster.champion.label if roster.champion is not None else None
    role_by_label = {o.label: o.role for o in roster.opponents}
    if roster.sealbot is not None:
        role_by_label.setdefault(roster.sealbot.label, "sealbot")

    out: list[dict[str, Any]] = []
    for row in pool_doc.get("edges", []):
        try:
            if int(row.get("epoch")) != int(epoch_tag):
                continue
        except (TypeError, ValueError):
            continue
        a = str(row.get("a"))
        b = str(row.get("b"))
        if a == cand_label:
            opp_label = b
        elif b == cand_label:
            opp_label = a
        else:
            continue
        kind = row.get("kind", "checkpoint")
        raw = row.get("raw") or {}
        role = role_by_label.get(opp_label, "champion" if opp_label == champ_label else "bracket")
        is_champ = opp_label == champ_label
        if kind == "sealbot":
            wins = int(raw.get("physical_wins_cand", 0) or 0)
            losses = int(raw.get("physical_wins_sealbot", 0) or 0)
            decided = wins + losses
            lo, hi = eval_stats.wilson_ci(wins, decided) if decided else (0.0, 1.0)
            desc = {
                "opponent": opp_label,
                "role": "sealbot",
                "kind": "sealbot",
                "primary": False,
                "paired": False,
                "decided": decided,
                "winrate": round(wins / decided, 4) if decided else None,
                "winrate_ci95": [round(lo, 4), round(hi, 4)],
                "elo_point": _safe_elo(wins / decided) if decided else None,
                "down_weight": round(float(row.get("weight", 1.0)), 4),
                "provenance": raw,
                "note": "DESCRIPTIVE zero-point (reconstructed from the pooled edge).",
            }
        else:
            # Reconstruct the pair-level winrate/CI from the stored provenance.
            pr_wr = raw.get("pair_winrate")
            pr_se = raw.get("pair_se")
            wins = int(raw.get("physical_wins_a", 0) or 0)
            losses = int(raw.get("physical_wins_b", 0) or 0)
            decided = wins + losses
            if pr_wr is not None and pr_se is not None and math.isfinite(float(pr_se)):
                wr = float(pr_wr)
                se = float(pr_se)
                wr_lo, wr_hi = max(0.0, wr - 1.96 * se), min(1.0, wr + 1.96 * se)
                elo_lo, elo_hi = eval_stats.elo_ci_from_winrate(wr, se)
                winrate = round(wr, 4)
            else:
                wr_lo, wr_hi = (eval_stats.wilson_ci(wins, decided) if decided else (0.0, 1.0))
                elo_lo, elo_hi = (_safe_elo(wr_lo), _safe_elo(wr_hi))
                winrate = round(wins / decided, 4) if decided else None
            desc = {
                "opponent": opp_label,
                "role": role,
                "kind": "checkpoint",
                "primary": is_champ,
                "paired": True,
                "decided": decided,
                "winrate": winrate,
                "winrate_ci95": [round(wr_lo, 4), round(wr_hi, 4)],
                "elo_point": _safe_elo(winrate) if winrate is not None else None,
                "elo_ci95_pairlevel": [_round_elo(elo_lo), _round_elo(elo_hi)],
                "provenance": raw,
                "note": (
                    "PRIMARY edge — verdict via the pooled BT difference-CI (Stage D)."
                    if is_champ
                    else "DESCRIPTIVE edge — Wilson/Elo CIs only, no significance verdict."
                ),
            }
        out.append({"role": role, "opponent": opp_label, "descriptive": desc})
    return out


def run_multistage_eval_in_parts(
    run_dir: str | Path,
    candidate_ckpt: str | Path,
    config: ShrimpConfig | MultiStageEvalSection | None = None,
    *,
    candidate_epoch: int | None = None,
    candidate_label: str | None = None,
    checkpoints_dir: str | Path | None = None,
    diagnostics_dir: str | Path | None = None,
    write_diagnostics: bool = True,
    resume: bool = True,
    play_checkpoint_match: Callable[..., dict[str, Any]] | None = None,
    play_sealbot_match: Callable[..., dict[str, Any]] | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Run the staged eval as a sequence of resumable per-opponent parts.

    Plays each opponent as an independent part (:func:`run_eval_part`) that
    appends its edge to the pool and persists immediately, then runs the
    aggregate BT fit + verdict (:func:`aggregate_pool`) over the whole pool. An
    interruption leaves every completed part in ``eval_pool.json``; a restart with
    ``resume=True`` skips the parts already pooled for this epoch and plays only
    what remains. A part that fails (e.g. SealBot unavailable) is caught, logged
    in the returned ``parts`` list, and the loop continues (fail-open).

    Scope: the deep eval (Stage C, as per-opponent parts) + the pooled BT
    fit/verdict (Stage D). Does not run the Stage B SPRT triage, so each opponent
    part (including the champion) plays its full ``per`` budget rather than
    reusing SPRT games. The primary verdict (pooled BT difference-CI, candidate vs
    champion) matches the monolithic path in kind; only the champion's physical
    game count differs when SPRT is enabled.

    Returns the aggregate report (same shape as :func:`run_multistage_eval`, with
    ``stages`` holding the Stage D entry) plus an extra ``parts`` list recording
    each part's status.
    """

    run_dir = Path(run_dir)
    candidate_ckpt = Path(candidate_ckpt)
    cfg = _coerce_section(config)
    _assert_no_run_mutation(cfg)

    if play_checkpoint_match is None or play_sealbot_match is None:
        from . import eval_arena as _arena

        if play_checkpoint_match is None:
            play_checkpoint_match = _arena.play_checkpoint_match
        if play_sealbot_match is None:
            play_sealbot_match = _arena.play_sealbot_match

    roster = select_opponents(
        run_dir, candidate_ckpt, cfg,
        candidate_epoch=candidate_epoch, candidate_label=candidate_label,
        checkpoints_dir=checkpoints_dir,
    )

    # write_diagnostics=True: each part loads + persists the on-disk pool, so a
    # restart resumes from what prior runs wrote. write_diagnostics=False: thread
    # one shared in-memory pool across the parts (nothing is written), so the
    # aggregate fits this run's edges without touching disk.
    mem_pool: dict[str, Any] | None = None
    if not write_diagnostics:
        mem_pool = _load_pool(_pool_path(run_dir, cfg))

    part_results: list[dict[str, Any]] = []
    for opp in _part_opponents(roster):
        res = run_eval_part(
            run_dir, candidate_ckpt, opp.label, config,
            candidate_epoch=candidate_epoch, candidate_label=candidate_label,
            checkpoints_dir=checkpoints_dir, diagnostics_dir=diagnostics_dir,
            write_pool=write_diagnostics, resume=resume, pool_doc=mem_pool,
            play_checkpoint_match=play_checkpoint_match,
            play_sealbot_match=play_sealbot_match,
        )
        # Carry the (possibly newly-built) in-memory pool forward to the next part.
        if mem_pool is not None:
            mem_pool = res.get("pool_doc", mem_pool)
        # Drop the doc from the public part record (it is internal plumbing).
        res.pop("pool_doc", None)
        part_results.append(res)

    report = aggregate_pool(
        run_dir, candidate_ckpt, config,
        candidate_epoch=candidate_epoch, candidate_label=candidate_label,
        checkpoints_dir=checkpoints_dir, diagnostics_dir=diagnostics_dir,
        write_diagnostics=write_diagnostics, pool_doc=mem_pool, now=now,
    )
    report["parts"] = part_results
    report["meta"]["mode"] = "run_in_parts"
    return report


# --------------------------------------------------------------------------- #
# CONCURRENT ONE-PASS PATH.
#
# Plays the whole roster in one concurrent pass: SealBot via play_sealbot_match
# (its own concurrent loop) + all checkpoint opponents via the multi-opponent
# runner play_multi_checkpoint_match (one shared candidate forward across every
# opponent), so wall-clock is max over matches rather than the sum the parts path
# incurs. Then the same Stage D (_stage_d_pool / pool append / BT fit / verdict)
# runs over the pool. Fail-soft, no run-state writes.
# --------------------------------------------------------------------------- #
def run_multistage_eval_concurrent(
    run_dir: str | Path,
    candidate_ckpt: str | Path,
    config: ShrimpConfig | MultiStageEvalSection | None = None,
    *,
    candidate_epoch: int | None = None,
    candidate_label: str | None = None,
    checkpoints_dir: str | Path | None = None,
    diagnostics_dir: str | Path | None = None,
    write_diagnostics: bool = True,
    play_multi_checkpoint_match: Callable[..., dict[str, Any]] | None = None,
    play_sealbot_match: Callable[..., dict[str, Any]] | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Run the deep eval (Stage C) as one concurrent pass, then Stage D.

    Resolves the roster + the same :func:`allocate_budget` split (threaded
    ``cfg.sealbot_share``) the other paths use, then plays:
      * SealBot via ``play_sealbot_match`` (its own concurrent, unpaired loop) —
        fail-open: if it raises, the edge is dropped and the pool anchors on a
        checkpoint.
      * Every checkpoint opponent via ``play_multi_checkpoint_match`` in one
        batched pass (one shared candidate forward across all opponents) —
        fail-soft at the whole-batch boundary.
    The pre-played match dicts are turned into edges by the shared edge builders
    (:func:`_build_sealbot_edge_from_match` /
    :func:`_build_checkpoint_edge_from_match`), then Stage D appends + fits +
    verdicts over the pool. Same diagnostics JSON + ``eval_pool.json`` rows as
    :func:`run_multistage_eval`.

    Scope: Stage C (concurrent) + Stage D. No Stage B SPRT triage, so every
    opponent plays its full ``per`` budget.

    Writes only the pool JSON + the per-epoch diagnostics under the diagnostics
    tree.
    """

    run_dir = Path(run_dir)
    candidate_ckpt = Path(candidate_ckpt)
    cfg = _coerce_section(config)
    _assert_no_run_mutation(cfg)
    full_cfg = config if isinstance(config, ShrimpConfig) else parse_shrimp_config({})
    diag_dir = Path(diagnostics_dir) if diagnostics_dir is not None else (run_dir / "diagnostics")
    started = now()

    if play_multi_checkpoint_match is None or play_sealbot_match is None:
        from . import eval_arena as _arena

        if play_multi_checkpoint_match is None:
            play_multi_checkpoint_match = _arena.play_multi_checkpoint_match
        if play_sealbot_match is None:
            play_sealbot_match = _arena.play_sealbot_match

    roster = select_opponents(
        run_dir, candidate_ckpt, cfg,
        candidate_epoch=candidate_epoch, candidate_label=candidate_label,
        checkpoints_dir=checkpoints_dir,
    )
    cand_label = roster.candidate_label
    epoch_tag = roster.candidate_epoch if roster.candidate_epoch is not None else 0

    alloc = allocate_budget(
        cfg.games_budget,
        n_checkpoint_opponents=len(roster.opponents),
        has_sealbot=roster.sealbot is not None,
        sealbot_share=cfg.sealbot_share,
    )

    edges: list[dict[str, Any]] = []
    played: list[str] = []
    sealbot_ci: list[float] | None = None
    sealbot_unavailable: str | None = None
    multi_error: str | None = None

    # ----- SealBot zero-point (separate concurrent runner; fail-open). -----
    if roster.sealbot is not None and alloc.get(SEALBOT_LABEL, 0) > 0:
        sb_games = alloc[SEALBOT_LABEL]
        sb_edge, sb_ci, sb_unavail = _play_sealbot_opponent(
            cfg, roster, candidate_ckpt, full_cfg, sb_games,
            play_sealbot_match=play_sealbot_match,
            diagnostics_dir=diag_dir,
        )
        if sb_unavail is not None:
            sealbot_unavailable = sb_unavail
        if sb_edge is not None:
            sealbot_ci = sb_ci
            edges.append(sb_edge)
            played.append(SEALBOT_LABEL)

    # ----- ALL checkpoint opponents in ONE concurrent multi-opponent pass. -----
    per = alloc.get("per_checkpoint", 0)
    ckpt_opps = [o for o in roster.opponents if o.ckpt is not None]
    if per > 0 and ckpt_opps:
        try:
            matches = play_multi_checkpoint_match(
                str(candidate_ckpt),
                [(o.label, str(o.ckpt)) for o in ckpt_opps],
                per,
                config=full_cfg,
                candidate_label=cand_label,
                visits=_eval_visits(cfg, full_cfg),
                virtual_batch_size=_eval_virtual_batch_size(cfg, full_cfg),
                opening_plies=cfg.opening_plies,
                opening_temperature=cfg.opening_temperature,
                # Every opponent is a current-arch checkpoint that searches the
                # candidate's own (self-play / Gumbel) eval profile.
                diagnostics_dir=str(diag_dir),
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft at the batch boundary.
            multi_error = f"{type(exc).__name__}: {exc}"
            matches = {}
        for opp in ckpt_opps:
            match = matches.get(opp.label)
            if match is None:
                continue
            edges.append(
                _build_checkpoint_edge_from_match(
                    roster, opp, match, cfg=cfg,
                    opponent_search_profile="selfplay",
                )
            )
            played.append(opp.label)

    stage_c_status = "completed" if edges else "empty"
    stage_c_detail: dict[str, Any] = {
        "budget": cfg.games_budget,
        "allocation": alloc,
        "n_edges": len(edges),
        "opponents_played": played,
        "concurrent_one_pass": True,
        # Audit: every checkpoint opponent searches the candidate's own
        # (self-play / Gumbel) eval profile.
        "opponent_search_profiles": {o.label: "selfplay" for o in ckpt_opps},
    }
    if sealbot_unavailable is not None:
        stage_c_detail["sealbot_unavailable"] = sealbot_unavailable
    if multi_error is not None:
        stage_c_detail["multi_checkpoint_error"] = multi_error
    stage_c = StageResult(stage="C_deep", status=stage_c_status, detail=stage_c_detail)

    # ----- Stage D — the same pool append / BT fit / verdict. -----
    # Flag an expected-but-unavailable SealBot so the substituted anchor degrades
    # the verdict. Gated on roster.sealbot so a config-disabled SealBot is not a
    # degradation.
    sealbot_expected_but_unavailable = (
        stage_c_detail.get("sealbot_unavailable") if roster.sealbot is not None else None
    )
    stage_d, ratings, verdict_block, pool_doc = _stage_d_pool(
        cfg, roster, edges, run_dir,
        sealbot_expected_but_unavailable=sealbot_expected_but_unavailable,
    )

    report: dict[str, Any] = {
        "meta": {
            "kind": "shrimp.multistage_eval",
            "mode": "run_concurrent",
            "run_dir": str(run_dir),
            "candidate_ckpt": str(candidate_ckpt),
            "candidate_label": cand_label,
            "candidate_epoch": roster.candidate_epoch,
            "anchor": SEALBOT_LABEL,
            "config": _config_summary(cfg),
            "elapsed_seconds": round(now() - started, 2),
            "single_epoch_se_elo_note": _resolution_note(cfg, edges, roster),
            "pure_eval": True,
            "gating_enabled": cfg.eval_gating_enabled,
            "promotion_enabled": cfg.eval_promotion_enabled,
        },
        "roster": _roster_summary(roster),
        "stages": [
            {"stage": stage_c.stage, "status": stage_c.status, **stage_c.detail},
            {"stage": stage_d.stage, "status": stage_d.status, **stage_d.detail},
        ],
        "ratings": ratings,
        "edges": [e["descriptive"] for e in edges],
        "sealbot_winrate_ci95": sealbot_ci,
        "verdict": verdict_block,
    }

    if write_diagnostics:
        _save_pool(_pool_path(run_dir, cfg), pool_doc)
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag_path = diag_dir / f"{DIAG_PREFIX}{epoch_tag:06d}.json"
        diag_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["meta"]["diagnostics_path"] = str(diag_path)

    return report


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #
def _coerce_section(config: ShrimpConfig | MultiStageEvalSection | None) -> MultiStageEvalSection:
    if config is None:
        return MultiStageEvalSection()
    if isinstance(config, ShrimpConfig):
        return config.multi_stage_eval
    if isinstance(config, MultiStageEvalSection):
        return config
    raise TypeError(f"config must be ShrimpConfig | MultiStageEvalSection | None, got {type(config)}")


def _choose_anchor(
    bt_edges: list[eval_stats.BTEdge],
    roster: Roster,
    *,
    ood_labels: set[str] | None = None,
) -> str | None:
    """Pick the BT zero-point anchor for the pool.

    Returns a label that appears in an edge (so eval_stats.bradley_terry's
    anchor-in-edge guard cannot trip), or None only when there is no non-candidate
    edge. When any checkpoint edge exists the fit is never left free-floating.

    Preference order:
      1. SealBot, when it has an edge (the cross-lineage zero-point).
      2. ``bc_prefit``, then any other permanent anchor with an edge, in
         configured order.
      3. The lowest available checkpoint opponent by epoch (e.g. ep5) with an
         edge.
      4. Any non-candidate player with an edge.

    ``ood_labels`` are featurized-OOD opponents; they are not eligible as the
    pinned zero-point (tiers 2/3 skip them) but still participate as descriptive
    edges. Tier 4 falls back to one only when there is no non-OOD non-candidate
    edge.
    """

    ood = ood_labels or set()
    labels = {lbl for e in bt_edges for lbl in (e.a, e.b)}
    # 1. SealBot zero-point (when it produced an edge).
    if SEALBOT_LABEL in labels:
        return SEALBOT_LABEL
    # 2. bc_prefit first, then any other anchor-role opponent, in configured
    #    order — excluding featurized-OOD anchors.
    for o in roster.opponents:
        if o.role == "anchor" and o.label == "bc_prefit" and o.label in labels and o.label not in ood:
            return o.label
    for o in roster.opponents:
        if o.role == "anchor" and o.label in labels and o.label not in ood:
            return o.label
    # 3. Lowest available checkpoint opponent by epoch. Skip OOD opponents.
    ckpt_opps = sorted(
        (o for o in roster.opponents if o.label in labels and o.epoch is not None and o.label not in ood),
        key=lambda o: o.epoch,
    )
    if ckpt_opps:
        return ckpt_opps[0].label
    # 4. Any non-candidate player with an edge (prefer non-OOD, else accept an
    #    OOD anchor over a free-floating fit).
    for lbl in sorted(labels):
        if lbl != roster.candidate_label and lbl not in ood:
            return lbl
    for lbl in sorted(labels):
        if lbl != roster.candidate_label:
            return lbl
    return None


def _anchor_in_edges(anchor: str, bt_edges: list[eval_stats.BTEdge]) -> bool:
    return any(anchor in (e.a, e.b) for e in bt_edges)


def _n_gating_edges(roster: Roster) -> int:
    """Number of edges that gate (>=1).

    Exactly one edge gates (candidate vs champion), so this is 1 and Bonferroni
    is a no-op.
    """

    return 1


def _verdict_with_thresholds(
    lo: float, hi: float, *, promote_elo: float, regress_elo: float
) -> str:
    """PROMOTE / REGRESS / INCONCLUSIVE from a difference CI with thresholds.

    The whole CI must exceed ``promote_elo`` to PROMOTE and be below
    ``regress_elo`` to REGRESS; otherwise INCONCLUSIVE.
    """

    if lo > promote_elo:
        return "PROMOTE"
    if hi < regress_elo:
        return "REGRESS"
    return "INCONCLUSIVE"


def _merge_matches(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Pool two checkpoint matches vs the same opponent into one result dict.

    Merges only the downstream-consumed fields: ``score`` (counts) and
    ``pentanomial`` (pairs + histogram). Used to combine the reused SPRT champion
    match with a fresh top-up. Both inputs must be net-A-centric on the same
    candidate.
    """

    if a is None:
        return b
    if b is None:
        return a
    sa, sb = a.get("score") or {}, b.get("score") or {}
    merged_score = {
        "completed": int(sa.get("completed", 0)) + int(sb.get("completed", 0)),
        "truncated": int(sa.get("truncated", 0)) + int(sb.get("truncated", 0)),
        "aborted_budget": int(sa.get("aborted_budget", 0)) + int(sb.get("aborted_budget", 0)),
        "a_wins": int(sa.get("a_wins", 0)) + int(sb.get("a_wins", 0)),
        "b_wins": int(sa.get("b_wins", 0)) + int(sb.get("b_wins", 0)),
    }
    merged_score["decided"] = merged_score["a_wins"] + merged_score["b_wins"]
    merged_score["a_winrate_decided"] = (
        round(merged_score["a_wins"] / merged_score["decided"], 4)
        if merged_score["decided"] else None
    )
    # Concatenate pentanomial pairs (the pair-level SE is recomputed downstream
    # from the combined pairs).
    pa = (a.get("pentanomial") or {}).get("pairs") or []
    pb = (b.get("pentanomial") or {}).get("pairs") or []
    combined_pairs = list(pa) + list(pb)
    merged_penta = {"pairs": combined_pairs} if combined_pairs else None
    return {
        "meta": {**(b.get("meta") or {}), "merged_from": 2},
        "score": merged_score,
        "pentanomial": merged_penta,
    }


def _safe_elo(p: float | None) -> float | None:
    if p is None:
        return None
    e = eval_stats.elo_from_winrate(p)
    if not math.isfinite(e):
        return None
    return round(e, 1)


def _round_elo(e: float) -> float | None:
    if e is None or not math.isfinite(e):
        return None
    return round(e, 1)


def _resolution_note(
    cfg: MultiStageEvalSection, edges: list[dict[str, Any]], roster: Roster
) -> str:
    """Single-epoch resolution note, derived from the champion games."""

    champ_decided = 0
    for e in edges:
        if e.get("role") == "champion":
            champ_decided = int(e["descriptive"].get("decided", 0) or 0)
            break
    se = eval_stats.expected_se_elo(champ_decided) if champ_decided else float("inf")
    se_txt = f"{se:.0f}" if math.isfinite(se) else "inf"
    return (
        f"Single-epoch SE(win rate) over {champ_decided} champion games is ~{se_txt} Elo "
        f"(independent-game approx); the PRIMARY difference SE is LARGER still (paired -> "
        f"effective N well below decided, plus the two-rating sqrt(2)) — order ~120-140 "
        f"Elo, resolving ~250-300 Elo. This candidate-vs-champion verdict is PERMANENTLY "
        f"single-epoch-limited (a fresh candidate node each epoch never compounds): it is "
        f"a gross-regression tripwire, not a fine-edge test. Only the FIXED-anchor "
        f"DESCRIPTIVE curve (bc_prefit/ep5/SealBot — same labels every epoch) compounds "
        f"toward the ~15-20 Elo multi-epoch asymptote, which describes the lineage "
        f"progress curve, never the single-epoch verdict."
    )


def _config_summary(cfg: MultiStageEvalSection) -> dict[str, Any]:
    return {
        "games_budget": cfg.games_budget,
        "eval_visits": cfg.eval_visits,
        "full_search_visits": cfg.full_search_visits,
        "eval_virtual_batch_size": cfg.eval_virtual_batch_size,
        "opening_plies": cfg.opening_plies,
        "opening_temperature": cfg.opening_temperature,
        "primary_alpha": cfg.primary_alpha,
        "bonferroni_correction": cfg.bonferroni_correction,
        "sealbot_overdispersion": cfg.sealbot_overdispersion,
        "bt_grad_tol": cfg.bt_grad_tol,
        "bt_max_iters": cfg.bt_max_iters,
        "promote_elo_threshold": cfg.promote_elo_threshold,
        "regress_elo_threshold": cfg.regress_elo_threshold,
        "verdict_reference_lag": cfg.verdict_reference_lag,
        "sprt": {
            "enabled": cfg.sprt.enabled,
            "elo0": cfg.sprt.elo0,
            "elo1": cfg.sprt.elo1,
            "max_games": cfg.sprt.max_games,
        },
        "opponents": {
            "sealbot_enabled": cfg.opponents.sealbot_enabled,
            "sealbot_variant": cfg.opponents.sealbot_variant,
            "permanent_anchors": [list(a) for a in cfg.opponents.permanent_anchors],
            "log_grid": list(cfg.opponents.log_grid),
            "bracket_size": cfg.opponents.bracket_size,
        },
    }


def _roster_summary(roster: Roster) -> dict[str, Any]:
    return {
        "candidate": {"label": roster.candidate_label, "epoch": roster.candidate_epoch},
        "sealbot": roster.sealbot.label if roster.sealbot else None,
        "champion": (
            {"label": roster.champion.label, "epoch": roster.champion.epoch}
            if roster.champion else None
        ),
        "opponents": [
            {"label": o.label, "role": o.role, "epoch": o.epoch, "ckpt": str(o.ckpt) if o.ckpt else None}
            for o in roster.opponents
        ],
        # Permanent anchors dropped because they did not resolve on disk.
        # Recorded in every pipeline's per-epoch JSON (all paths go through here).
        "dropped_anchors": [dict(a) for a in roster.dropped_anchors],
    }
