"""Sample facts, game finalization, and train-time row expansion.

Expansion maps targets from packed action ids onto the row's legal-prefix
slots. Policy mass off the legal set is a hard error for the self policy and a
tracked projection drop (`opp_coverage`) for the opponent policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import MOVES_LEFT_CAP, RAYLEN_SLOTS, RAYTAP, TRUNK_LAYOUT
from .features import PositionFacts, build_position, build_ray_lengths, transform_facts
from .geometry import apply_d6, unpack_action_id
from .support import Support

STV_HORIZONS = (2, 6, 16)

# Ray lengths are consumed by 'L' trunk layouts and by ray-tap convs
# (SPEC_RAYTAP_CONV.md §2.5 — including L-free layouts, arm A5); the per-row
# Python walk (12 rays x <=5 steps per cell, all dict lookups) is pure overhead
# in every prefit/expand worker for arms that consume neither, so the serial
# expand path skips the oracle there (spec D-S29). The Rust expand kernel keeps
# emitting raylen regardless (the model ignores it). Read once at import; the
# parity tests monkeypatch this to force the oracle on under a C/A layout.
_EXPAND_RAYLEN = ("L" in TRUNK_LAYOUT) or (RAYTAP != "0")


@dataclass(frozen=True)
class HexfieldSampleData:
    """One decision row's raw facts + targets (players are ints 0/1)."""

    game_id: str
    turn_index: int
    current_player: int
    phase: str
    records: tuple[tuple[int, int, int, int], ...]  # (q, r, owner, placement_index)
    first_stone: tuple[int, int] | None
    policy: tuple[tuple[int, float], ...]
    opp_policy: tuple[tuple[int, float], ...] = ()
    q_policy: tuple[tuple[int, float], ...] = ()  # (action_id, child Q); parallel to policy
    # Improved-policy target as (action_id, weight) over the searched support.
    # Empty means the row uses the visit target (policy).
    gumbel_policy: tuple[tuple[int, float], ...] = ()
    prior_logit: tuple[tuple[int, float], ...] = ()  # (action_id, root logit)
    value: float = 0.0
    short_term_value: tuple[tuple[int, float], ...] = ()
    moves_left: float = -1.0
    policy_surprise: float = 0.0  # KL(visit ‖ root prior); self policy CE weight
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def facts(self) -> PositionFacts:
        return PositionFacts(
            records=self.records,
            current_player=self.current_player,
            phase=self.phase,
            first_stone=self.first_stone,
        )


def _winner_value(winner: int | None, player: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


def _policy_surprise_kl(action_ids, weights, prior_ids, prior_weights, *, eps: float = 1e-8) -> float:
    """KL(visit ‖ prior); visit normalized to sum 1; prior assumed ~sum 1.

    Returns ``max(0, kl)``, or ``0.0`` if non-finite. Used by ``selfplay.py``
    to record the per-row policy-surprise scalar; the in-loss weight is derived
    at collate from the recorded scalar.
    """
    w = np.asarray(weights, dtype=np.float64)
    s = float(w.sum())
    if s <= 0.0:
        return 0.0
    prior_map = {int(a): float(p) for a, p in zip(prior_ids, prior_weights)}
    kl = 0.0
    for a, tw in zip((int(x) for x in action_ids), (w / s).tolist()):
        if tw <= 0.0:
            continue
        pw = max(prior_map.get(a, 0.0), eps)
        kl += tw * float(np.log((tw + eps) / pw))
    return max(0.0, kl) if np.isfinite(kl) else 0.0


def _future_opponent_policy(
    decisions: Sequence[tuple[int, "HexfieldSampleData", float]],
    index: int,
    player: int,
    *,
    mask_from_fast: bool = False,
) -> tuple[tuple[tuple[int, float], ...], str]:
    """Return the next opponent decision's policy target and a source tag.

    Prefers the opponent decision's improved policy π' when it carries one
    (``future_opponent_gumbel``): under Gumbel Sequential Halving the visit
    histogram is a schedule artifact (equal per-round quotas), so π' is the
    meaningful prediction target for the opp head — mirroring the main-policy
    and soft-policy target selection. Falls back to the visit policy
    (``future_opponent_mcts``) for decisions without one (PUCT search /
    legacy rows).

    Returns an empty policy tagged ``fast_unrecorded_masked`` when
    ``mask_from_fast`` is set and that decision's ``metadata['pcr_full']`` is
    False; ``none`` when no later opponent decision exists.
    """

    for future_player, future_sample, _root_value in decisions[index + 1 :]:
        if future_player != player:
            if mask_from_fast and not future_sample.metadata.get("pcr_full", True):
                return (), "fast_unrecorded_masked"
            if future_sample.gumbel_policy:
                return tuple(future_sample.gumbel_policy), "future_opponent_gumbel"
            return tuple(future_sample.policy), "future_opponent_mcts"
    return (), "none"


def _short_term_value_targets(
    decisions: Sequence[tuple[int, "HexfieldSampleData", float]],
    index: int,
    player: int,
    horizons: Sequence[int],
) -> tuple[tuple[int, float], ...]:
    """Per-horizon EMA of future root values stepped over full turns (even
    decision offsets only), decay (m-1)/(m+1) for horizon m.

    Future root values are taken from the current player's perspective
    (negated on opponent decisions). Returns one (horizon, value) pair per
    horizon, or () when there is no stepped future value.
    """

    future = decisions[index + 1 :]
    perspective = [
        root_value if future_player == player else -root_value
        for future_player, _sample, root_value in future
    ]
    stepped = perspective[1::2]
    if not stepped:
        return ()
    targets: list[tuple[int, float]] = []
    for horizon in horizons:
        decay = (horizon - 1.0) / (horizon + 1.0)
        weighted_sum = 0.0
        weight_total = 0.0
        weight = 1.0
        for value in stepped:
            weighted_sum += weight * value
            weight_total += weight
            weight *= decay
        targets.append((int(horizon), weighted_sum / weight_total))
    return tuple(targets)


def finalize_game_samples(
    pending: Sequence[tuple[int, HexfieldSampleData, float]],
    winner: int | None,
    horizons: Sequence[int] = STV_HORIZONS,
    *,
    truncated: bool = False,
    soft_z_lambda: float = 0.0,
    mask_opp_from_fast: bool = False,
) -> list[HexfieldSampleData]:
    """Assign outcome targets to a finished game's pre-decision samples.

    The value target is ``(1 - soft_z_lambda) * hard_z + soft_z_lambda *
    root_value`` (plain hard_z when ``soft_z_lambda == 0``), where hard_z is
    +1/-1/0 for win/loss/no-winner from the row player's perspective.
    ``soft_z_lambda`` must be in [0, 1].

    For truncated games (``truncated=True``, ``winner=None``) each row gets
    ``metadata['truncated']=True`` and ``moves_left=-1.0`` (the sentinel that
    yields moves_left_mask=0 at expand); ``expand_sample`` zeroes the
    value/stvalue/cell_q masks for these rows, so the value target is not used.
    policy and opp_policy are assigned the same way regardless of
    truncation. Completed games (truncated=False) receive
    ``moves_left = len(decisions) - index - 1``.
    """

    decisions = list(pending)
    horizons = tuple(int(h) for h in horizons)
    lam = float(soft_z_lambda)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"soft_z_lambda must be in [0, 1], got {soft_z_lambda!r}")
    finalized: list[HexfieldSampleData] = []
    for index, (player, sample, root_value) in enumerate(decisions):
        opp_policy, opp_source = _future_opponent_policy(
            decisions, index, player, mask_from_fast=mask_opp_from_fast
        )
        metadata = {
            **dict(sample.metadata),
            "opp_policy_source": opp_source,
            "truncated": bool(truncated),
        }
        hard_z = _winner_value(winner, player)
        value_target = (1.0 - lam) * hard_z + lam * float(root_value) if lam > 0.0 else hard_z
        finalized.append(
            replace(
                sample,
                value=value_target,
                opp_policy=opp_policy,
                short_term_value=_short_term_value_targets(decisions, index, player, horizons),
                moves_left=float(len(decisions) - index - 1) if not truncated else -1.0,
                metadata=metadata,
            )
        )
    return finalized


@dataclass(frozen=True)
class ExpandedRow:
    """One expanded training row (numpy; collated by batching.py)."""

    support: Support
    feats: np.ndarray  # (N, F) f32
    policy: np.ndarray  # (L,) f32 over the legal prefix
    opp_policy: np.ndarray  # (L,) f32; zero row when absent/masked/uncovered
    opp_coverage: float  # kept mass / total mass (1.0 when no target existed)
    value: float
    value_mask: float  # 1.0 for completed games; 0.0 for truncated (no winner)
    stvalue: np.ndarray  # (H,) f32
    stvalue_mask: np.ndarray  # (H,) f32
    moves_left: float  # normalized to [-1, 1]; 0 when masked
    moves_left_mask: float
    cell_q: np.ndarray  # (L,) f32 per-cell Q target over the legal prefix; 0 where absent
    cell_q_mask: np.ndarray  # (L,) f32 presence mask (Q=0.0 is a valid target)
    policy_surprise: float  # KL(visit ‖ prior); collate derives the self-CE weight from it
    # Dense (L,) improved-policy target, renormalized over the kept support: sums
    # to 1 when present, all-zero when absent. An empty array is the default;
    # collate packs an all-zero target for it.
    gumbel_policy: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )  # (L,) f32; sums to 1 over kept support, else all-zero
    gumbel_policy_valid: float = 0.0  # 1.0 when a gumbel target was projected
    prior_logit: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )  # (L,) f32 raw root logits over the legal prefix (0 where absent)
    # Side-relative ray lengths (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase
    # L0), (N, RAYLEN_SLOTS) u8 in support node order, recomputed from the
    # transformed facts like the graded planes. An empty array is the legacy
    # default; collate packs zeros for it.
    raylen: np.ndarray = field(
        default_factory=lambda: np.zeros((0, RAYLEN_SLOTS), dtype=np.uint8)
    )  # (N, RAYLEN_SLOTS) u8


def expand_sample(
    sample: HexfieldSampleData,
    *,
    symmetry: int = 0,
    horizons: Sequence[int] = STV_HORIZONS,
) -> ExpandedRow:
    """Facts -> (support, features, legal-prefix targets) under one D6 draw.

    The ``symmetry`` transform is applied to all stored coordinate facts
    (including policy / opp-policy action ids); support, node order, features,
    and target slots are rebuilt from the transformed facts.
    """

    facts = transform_facts(sample.facts(), symmetry)
    sup, feats = build_position(facts)
    legal_count = sup.legal_count

    policy = np.zeros(legal_count, dtype=np.float32)
    total = 0.0
    for action_id, weight in sample.policy:
        w = float(weight)
        if not np.isfinite(w) or w < 0.0:
            raise ValueError("policy weights must be finite and nonnegative")
        slot = _legal_slot(sup, symmetry, int(action_id))
        if slot is None:
            raise ValueError(
                f"policy target action {action_id} is off the legal set (hard error)"
            )
        policy[slot] += w
        total += w
    # Every recorded row is a full (policy-bearing) row in main_9: fast rows are
    # dropped at the self-play writer and never reach expand. The policy target
    # must therefore always carry positive mass.
    if total <= 0.0:
        raise ValueError("policy target must carry positive mass")

    opp = np.zeros(legal_count, dtype=np.float32)
    opp_total = 0.0
    opp_kept = 0.0
    for action_id, weight in sample.opp_policy:
        w = float(weight)
        if not np.isfinite(w) or w < 0.0:
            raise ValueError("opp policy weights must be finite and nonnegative")
        opp_total += w
        slot = _legal_slot(sup, symmetry, int(action_id))
        if slot is not None:
            opp[slot] += w  # projection onto this row's legal set
            opp_kept += w
    opp_coverage = (opp_kept / opp_total) if opp_total > 0.0 else 1.0

    # Per-cell Q target: scalar child Q projected onto this row's legal set plus
    # a presence mask. Off-legal Q is dropped.
    cell_q = np.zeros(legal_count, dtype=np.float32)
    cell_q_mask = np.zeros(legal_count, dtype=np.float32)
    for action_id, q in sample.q_policy:
        qv = float(q)
        if not np.isfinite(qv) or qv < -1.0 or qv > 1.0:
            raise ValueError("cell_q targets must be finite and in [-1, 1]")
        slot = _legal_slot(sup, symmetry, int(action_id))
        if slot is not None:
            cell_q[slot] = qv  # scalar assign (one action -> one distinct cell)
            cell_q_mask[slot] = 1.0

    # Project the improved-policy target onto this row's legal set and renormalize
    # over the kept (on-legal) support so it sums to 1 when present. When absent,
    # gumbel_policy is all-zero and gumbel_policy_valid is 0.0.
    gumbel_policy = np.zeros(legal_count, dtype=np.float32)
    g_total = 0.0
    for action_id, weight in sample.gumbel_policy:
        w = float(weight)
        if not np.isfinite(w) or w < 0.0:
            raise ValueError("gumbel policy weights must be finite and nonnegative")
        slot = _legal_slot(sup, symmetry, int(action_id))
        if slot is not None:
            gumbel_policy[slot] += w  # projection onto this row's legal set
            g_total += w
    gumbel_policy_valid = 0.0
    if sample.gumbel_policy and g_total > 0.0:
        gumbel_policy /= g_total  # renormalize over the kept support
        gumbel_policy_valid = 1.0
    elif sample.gumbel_policy:
        # A gumbel target existed but no mass landed on-legal: leave gumbel_policy
        # all-zero and gumbel_policy_valid at 0.0 so the row uses the visit target.
        gumbel_policy = np.zeros(legal_count, dtype=np.float32)

    prior_logit = np.zeros(legal_count, dtype=np.float32)
    for action_id, logit in sample.prior_logit:
        lv = float(logit)
        if not np.isfinite(lv):
            raise ValueError("prior_logit values must be finite")
        slot = _legal_slot(sup, symmetry, int(action_id))
        if slot is not None:
            prior_logit[slot] = lv  # scalar assign (one action -> one cell)

    horizons = tuple(int(h) for h in horizons)
    stvalue = np.zeros(len(horizons), dtype=np.float32)
    stvalue_mask = np.zeros(len(horizons), dtype=np.float32)
    horizon_index = {h: i for i, h in enumerate(horizons)}
    for h, v in sample.short_term_value:
        col = horizon_index.get(int(h))
        if col is not None:
            stvalue[col] = float(v)
            stvalue_mask[col] = 1.0

    if float(sample.moves_left) >= 0.0:
        moves_left = 2.0 * min(1.0, float(sample.moves_left) / MOVES_LEFT_CAP) - 1.0
        moves_left_mask = 1.0
    else:
        moves_left = 0.0
        moves_left_mask = 0.0

    # Truncated games (no winner): zero the value, stvalue, and cell_q masks so
    # those heads contribute no loss. moves_left is masked via its -1 sentinel
    # above. Completed rows keep value_mask=1.0 and the presence masks as built.
    truncated = bool(sample.metadata.get("truncated", False))
    if truncated:
        value_mask = 0.0
        stvalue_mask = np.zeros_like(stvalue_mask)
        cell_q_mask = np.zeros_like(cell_q_mask)
    else:
        value_mask = 1.0

    return ExpandedRow(
        support=sup,
        feats=feats,
        policy=policy,
        opp_policy=opp,
        opp_coverage=opp_coverage,
        value=float(sample.value),
        value_mask=value_mask,
        stvalue=stvalue,
        stvalue_mask=stvalue_mask,
        moves_left=moves_left,
        moves_left_mask=moves_left_mask,
        cell_q=cell_q,
        cell_q_mask=cell_q_mask,
        policy_surprise=float(sample.policy_surprise),
        gumbel_policy=gumbel_policy,
        gumbel_policy_valid=gumbel_policy_valid,
        prior_logit=prior_logit,
        raylen=(
            build_ray_lengths(facts, sup)
            if _EXPAND_RAYLEN
            else np.zeros((0, RAYLEN_SLOTS), dtype=np.uint8)
        ),
    )


def _legal_slot(sup: Support, symmetry: int, action_id: int) -> int | None:
    q, r = unpack_action_id(action_id)
    cell = apply_d6(symmetry, q, r)
    slot = sup.index.get(cell)
    if slot is None or slot >= sup.legal_count:
        return None
    return slot
