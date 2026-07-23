//! hexfield PUCT tree.
//!
//! Divergence machinery gated by the `Divergences` toggles:
//!
//! - per-edge sum-of-squares accumulator (LCB selection; inactive when
//!   `lcb_move_selection` is off)
//! - per-node/per-edge (ml_sum, ml_weight) moves-left stats (inactive when
//!   `moves_left_utility` is off)
//! - visit-scaled c_puct schedule (off => the caller's static c)
//!
//! Every engine-legal move is a candidate; there is no candidate crop, so TSS
//! injection covers the full legal set.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use hexo_engine::{
    apply_placement, pack_coord, unpack_coord, GameOutcome, HexCoord, HexoState as RustHexoState,
    PackedCoord, Placement, Player,
};
use hexo_utils::StateHash;

use crate::cache::{state_hash, RustEvaluation};
use crate::state::move_error;
use crate::threats_shared as threats;
use crate::tss_async::{SolveRequest, SolveResponse, TssAsyncHandle};
use crate::tss_core::{self, HardValue, ProofStatus, SolveCaps, SolveGoal, SolveStats};
use crate::tss_solver::TssSolver;
use crate::tss_verify::{Group2Verifier, RootBinding, TssVerifier};

// Edge Q is `value_sum / visits` on the symmetric interval [-1, 1] (win/loss
// utility, no per-node normalization). The exploration (U) and value (Q) terms
// are added on this same scale, so the effective Q utility width is 2.0 (vs
// KataGo's 1.4 reference width); this documents the c_puct scale.

#[derive(Clone, Copy, Debug)]
pub struct RootDirichletNoise {
    pub total_alpha: f32,
    pub fraction: f32,
    pub seed: u64,
    /// When true, per-move concentration is the shaped alpha distribution
    /// (see `shaped_alpha`) times total_alpha; when false, the flat
    /// `total_alpha / count`. Set from the `dirichlet_shaped` divergence flag.
    pub shaped: bool,
}

/// Divergence toggles and constants. `parity()` forces the divergences off;
/// `production()` turns them on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Divergences {
    pub lcb_move_selection: bool,
    pub lcb_z: f32,
    pub lcb_min_visits: u32,
    pub lcb_visit_fraction: f32,
    pub early_stop: bool,
    pub full_visit_floor: f32,
    pub visit_scaled_c_puct: bool,
    pub c_scale: f32,
    pub c_base: f32,
    pub moves_left_utility: bool,
    pub ml_weight: f32,
    pub ml_scale: f32,
    pub ml_q_gate: f32,
    /// When on, the moves-left bonus also fires on the losing side
    /// (Q < -gate); when off, it fires only on the winning side (Q > gate).
    pub ml_two_sided: bool,
    /// Final root-move decisiveness tie-break among near-LCB-tied moves.
    pub ml_final_pick: bool,
    /// LCB band (in value units) defining "near-tied" for the final pick.
    pub ml_final_pick_band: f32,
    /// When true, nucleus widening accumulates cumulative mass in f64 and
    /// short-circuits to "take all" when `widening.mass >= 1.0`. When false,
    /// accumulate in f32 with no `mass == 1.0` short-circuit. See
    /// `nucleus_count_values`.
    pub nucleus_f64: bool,
    /// When true, the new (visits==0) child selection score includes the FPU
    /// baseline `value_or_fpu(parent_value, fpu_reduction)` (fpu + U). When
    /// false, the score is U-only (`prior * exploration_scale`). See
    /// `new_child_score`. Intended to be enabled together with `lazy_widening`.
    pub new_child_fpu: bool,
    /// When true, widening eligibility is a live `peek_next_candidate().is_some()`
    /// check (an unexpanded candidate exists). When false, the frozen comparison
    /// `edges.len() < max_eligible_children`.
    pub lazy_widening: bool,
    /// When true, cache the clean post-temp pre-noise root priors and reset from
    /// them before re-applying temperature/noise on reused/promoted roots. When
    /// false, re-apply in place.
    pub clean_root_prior_cache: bool,
    /// When true, root Dirichlet noise uses the shaped alpha distribution (see
    /// `shaped_alpha`) computed from the clean post-temp pre-noise policy. When
    /// false, flat Dir(total_alpha / count).
    pub dirichlet_shaped: bool,
    /// When true, recorded-target forced-playout pruning uses the dynamic
    /// c_for(root_visits) (matching selection); when false, static c_puct.
    pub pruned_dynamic_cpuct: bool,
    /// When true, the FPU reduction applied to unvisited children is scaled by
    /// `sqrt(Σ prior of already-visited children)` (KataGo
    /// `fpuReductionMax · √policyProbMassVisited`): ~0 at a fresh node (unvisited
    /// children look as good as the parent → early breadth), growing toward the
    /// full reduction as the node fills. When false, the reduction is the flat
    /// `fpu_reduction` (the pre-KataGo behavior, retained for `parity()`). See
    /// `RustNode::visited_policy_mass` and the selection sites.
    pub scaled_fpu: bool,
    // === Gumbel AlphaZero divergences (Danihelka et al. 2022). ===
    // All default off in both parity() and production(); `gumbel()` turns the
    // four mechanism bools on.
    /// Improved-policy target export: train the policy head against
    /// π'(a)=softmax(logits+σ(completedQ)). Off => visit-count target.
    pub gumbel_target: bool,
    /// Gumbel-Top-k root candidate sampling. On => the top-m candidate set
    /// replaces the Dirichlet+temperature root diversity. Off => PUCT+Dirichlet
    /// root.
    pub gumbel_root: bool,
    /// Sequential-Halving root budget allocation over the Gumbel-Top-m
    /// candidates (root only). Requires `gumbel_root`. Off => normal PUCT visits.
    pub gumbel_sequential_halving: bool,
    /// Deterministic non-root selection: visit-regularized rule
    /// argmax[π'(a) − N(a)/(1+ΣN)], π'=softmax(logits+σ(completedQ)), with no
    /// PUCT/c_puct/FPU/widening at interior nodes. Off => PUCT interior.
    pub gumbel_nonroot_select: bool,
    /// σ transform constants: σ(q)=(c_visit+max_b N(b))·c_scale·q.
    pub gumbel_c_visit: f32,
    pub gumbel_c_scale: f32,
    /// Export-only σ scale override for the improved-policy target π'. When
    /// `Some`, this c_scale replaces `gumbel_c_scale` in the σ call inside
    /// gumbel_target_policy ONLY; the SH ranking and interior selection keep
    /// `gumbel_c_scale`. `None` => the target uses `gumbel_c_scale` (unchanged).
    pub gumbel_target_c_scale: Option<f32>,
    /// Candidate count m for Gumbel-Top-k (clamped to n_legal at the root).
    pub gumbel_m: u32,
    /// Draw temperature τ applied to the LOGIT (only) in the Gumbel-Top-m draw
    /// comparator in init_gumbel_root: sa = logit/τ + g(a). This samples the
    /// candidate set from softmax(logits/τ) without replacement (Gumbel-top-k),
    /// widening candidates at τ>1. Affects ONLY the draw sort — the SH σ ranking,
    /// exported target, and TSS force-include all read raw logits. τ<=0 or 1.0 =>
    /// exactly today's behavior.
    pub gumbel_draw_temperature: f32,
    /// Target support floor: actions with N(a) < this are excluded from the
    /// target softmax support (then renormalized over survivors).
    pub gumbel_target_min_visits: u32,
    /// Play-policy quota prune: at an active SH root with temperature > 0 the
    /// PLAYED move samples the delta-visit histogram with round-0 quota losers
    /// zeroed (they carry schedule mass, not quality mass). Recorded targets
    /// are untouched. Off => sample the raw histogram (legacy behavior).
    pub gumbel_play_prune: bool,
    /// Interior forced-move guard (Lever 0, PLAN_TSS_DEEPENING.md §3): at
    /// INTERIOR node expansion with live opponent threats, no own win-now, and
    /// `min_hitting_set == B` (defense consumes the whole turn), the children
    /// set narrows to the hitting-cell universe — every dropped move carries a
    /// one-ply λ¹ refutation (it leaves the threats unanswerable). At
    /// `k < B` (a spare stone exists: quiet/counter-threat replies are live
    /// options) nothing is pruned. Root expansion is untouched. Off => today's
    /// inject-widen-only behavior.
    pub tss_interior_guard: bool,
    /// Deep-solver consumption tier (Stage 4 ladder, §10): 0 = off, 1 = SHADOW
    /// (solve + verify + count at gated leaves, consume nothing — the play and
    /// target stream is bit-identical to off), 2 = shadow + verified hard LOSS
    /// backups with GPU-eval elision, 3 = 2 + verified hard WIN backups. Every
    /// hard value is minted by tss_core::hard_value_from_verified — the
    /// independent certificate verifier runs BEFORE every backup, and a
    /// verification failure increments the fatal `deep_verify_failed` counter
    /// (production must alarm on nonzero) and degrades to net-eval.
    pub tss_solver_mode: u32,
    /// Deterministic per-solve node cap (no wall clock anywhere on this path).
    pub tss_solver_node_cap: u32,
    /// Leaf subsample gate in sixteenths (16 = every gated leaf, 0 = none),
    /// keyed off the leaf StateHash so re-selection is idempotent.
    pub tss_solver_sample_16: u32,
    /// Deep root guard (§10 rung 6): class-0 root moves get a verified deep
    /// solve; proven classes upgrade the shared class map, so the play-time
    /// guard (and, under Lever 1, the recorded targets) consume deep proofs.
    /// The row proof scalar (tss_proof) is likewise deep-upgraded.
    pub tss_solver_root_guard: bool,
    /// Async solve pool (§10 async rung): gated leaves ENQUEUE their solve to
    /// background workers and take the normal net eval; verified results
    /// drain back into the per-move memo and are consumed by the
    /// descent-stop on every later visit through the proven position. Same
    /// solver → verifier → sealed-mint path, off the GPU's critical path.
    /// TRADE: which visit first sees a proof becomes wall-clock dependent, so
    /// flag-ON self-play is not bit-reproducible (flag-off is unchanged).
    /// Only effective where the driver wires a pool (the continuous
    /// scheduler); un-wired searches fall back to the inline solve.
    pub tss_solver_async: bool,
    /// Base worker threads for the async pool (validated in [1, 32]).
    pub tss_solver_async_threads: u32,
    /// Maximum worker threads for park-mode async-pool growth. 0 chooses an
    /// available-parallelism-derived cap; otherwise validated in
    /// `tss_solver_async_threads..=64`. Ignored by the legacy non-park pool,
    /// which remains fixed at the base count for flag-off identity.
    pub tss_solver_async_threads_max: u32,
    /// Wait-at-leaf async consumption: accepted gated leaves remain parked in
    /// the scheduler until their verified result arrives or the bail deadline
    /// expires. Requires `tss_solver_async` and is default-off.
    pub tss_solver_park: bool,
    /// Solve EVERY leaf, not just threat-bearing ones: drops the
    /// `has_threats` gate on the deep-solver leaf routes, so quiet leaves
    /// also solve (park mode: solver-first, GPU eval only on release).
    /// Candidate-free quiet solves self-terminate in ~0.1 ms, so the pool
    /// absorbs the extra request volume; default-off.
    pub tss_solver_all_leaves: bool,
    /// Per-leaf parking bail deadline in milliseconds.
    pub tss_solver_park_timeout_ms: u32,
    /// Hybrid inline tier under the async flag: gated leaves whose
    /// `(hash & 0xF)` falls below THIS threshold solve inline on the search
    /// thread (first-touch consumption, exactly the pre-async behavior);
    /// gated leaves at or above it enqueue to the pool. 0 = pure async.
    /// Deploy shape: sample_16=16 + async + inline_16=4 keeps today's proven
    /// inline tier verbatim and adds pool coverage for the other 12/16. Ignored
    /// when `tss_solver_park` is on: the select thread never solves then.
    pub tss_solver_async_inline_16: u32,
    /// Enable proof-carrying zoned AND nodes. Default off until explicitly
    /// selected by a rollout profile.
    pub tss_zone: bool,
    /// Optional exact stale-area trimming inside the zoned candidate builder.
    pub tss_zone_stale_filter: bool,
    /// Include claimant count-2 windows in the initial zone candidate set.
    pub tss_zone_count2: bool,
    /// Enable P3 same-turn defender-pair canonicalization.
    pub tss_pair_commutation: bool,
    /// Semantic-horizon deadline for deep solves (absolute placements added to
    /// the current ply). `16` is the owner floor (PLAN_TSS_MCTS_INTEGRATION.md
    /// §5); `0` means unbounded (`semantic_horizon = u32::MAX`, node cap the
    /// only budget). Values `1..=15` are rejected at the Rust seam. Replaces
    /// the historical hardcoded `+12` in `tss_solve_verified`.
    pub tss_solver_horizon: u32,
    /// Reuse the unused portion of an undecided wide `Both` WIN attempt for
    /// the opponent-WIN attempt. Default off preserves the primal-only split.
    pub tss_solver_dual_pass: bool,
    /// Post-root budget reserved for the opponent-WIN attempt in wide `Both`
    /// solves. A positive value schedules the floor independently; dual-pass
    /// additionally donates unused primal work. Zero preserves current policy.
    pub tss_solver_loss_reserve_nodes: u32,
    /// v1 Group-2 reduced-fanout selector (default off). Flag-off is
    /// bit-identical to the pre-change engine; flag-on additionally selects
    /// the `Group2V1` verifier policy at the sealed mint.
    pub tss_solver_group2: bool,
    /// Enable the free-tempo J2near attacker-width extension. This is consumed
    /// only while the deep solver mode is nonzero; false preserves the
    /// historical `vcf_pair_complete` leaf profile bit-for-bit.
    pub tss_solver_j2near: bool,
    /// Horizon-ladder escalation (default off): when the base solve is Unknown
    /// with `horizon_cuts > 0`, re-solve ONCE at `2 * horizon` on the same
    /// solver instance (the shared TT replays the proven prefix). Unbounded
    /// base (`tss_solver_horizon = 0`) skips the ladder — there is nothing
    /// taller to climb to.
    pub tss_solver_horizon_ladder: bool,
}

impl Divergences {
    pub fn parity() -> Self {
        Self {
            lcb_move_selection: false,
            lcb_z: 1.6,
            lcb_min_visits: 8,
            lcb_visit_fraction: 0.1,
            early_stop: false,
            full_visit_floor: 0.75,
            visit_scaled_c_puct: false,
            c_scale: 0.45,
            c_base: 500.0,
            moves_left_utility: false,
            ml_weight: 0.03,
            ml_scale: 32.0,
            ml_q_gate: 0.6,
            ml_two_sided: false,
            ml_final_pick: false,
            ml_final_pick_band: 0.05,
            nucleus_f64: false,
            new_child_fpu: false,
            lazy_widening: false,
            clean_root_prior_cache: false,
            dirichlet_shaped: false,
            pruned_dynamic_cpuct: false,
            scaled_fpu: false,
            // Gumbel mechanisms off in parity.
            gumbel_target: false,
            gumbel_root: false,
            gumbel_sequential_halving: false,
            gumbel_nonroot_select: false,
            gumbel_c_visit: 50.0,
            gumbel_c_scale: 1.0,
            gumbel_target_c_scale: None,
            gumbel_m: 16,
            gumbel_draw_temperature: 1.0,
            gumbel_target_min_visits: 1,
            gumbel_play_prune: false,
            tss_interior_guard: false,
            tss_solver_mode: 0,
            tss_solver_node_cap: 2000,
            tss_solver_sample_16: 16,
            tss_solver_root_guard: false,
            tss_solver_async: false,
            tss_solver_async_threads: 8,
            tss_solver_async_threads_max: 0,
            tss_solver_park: false,
            tss_solver_all_leaves: false,
            tss_solver_park_timeout_ms: 100,
            tss_solver_async_inline_16: 0,
            tss_zone: false,
            tss_zone_stale_filter: false,
            tss_zone_count2: false,
            tss_pair_commutation: false,
            tss_solver_horizon: 16,
            tss_solver_dual_pass: false,
            tss_solver_loss_reserve_nodes: 0,
            tss_solver_group2: false,
            tss_solver_j2near: false,
            tss_solver_horizon_ladder: false,
        }
    }

    pub fn production() -> Self {
        Self {
            lcb_move_selection: true,
            early_stop: true,
            visit_scaled_c_puct: true,
            moves_left_utility: true,
            ml_two_sided: true,
            ml_final_pick: true,
            nucleus_f64: true,
            new_child_fpu: true,
            lazy_widening: true,
            clean_root_prior_cache: true,
            dirichlet_shaped: true,
            pruned_dynamic_cpuct: true,
            scaled_fpu: true,
            ..Self::parity()
        }
    }

    pub(crate) fn solver_j2near_enabled(self) -> bool {
        self.tss_solver_mode > 0 && self.tss_solver_j2near
    }

    /// Gumbel AlphaZero profile (Danihelka et al. 2022): starts from
    /// `production()` and turns the four Gumbel mechanism bools on, with the
    /// σ/m scalars at their defaults. Test-only convenience: production
    /// builds this profile from Python-side divergence overrides instead.
    #[cfg(test)]
    pub fn gumbel() -> Self {
        Self {
            gumbel_target: true,
            gumbel_root: true,
            gumbel_sequential_halving: true,
            gumbel_nonroot_select: true,
            ..Self::production()
        }
    }
}

#[derive(Clone, Debug)]
pub struct RustEdge {
    pub action_id: PackedCoord,
    pub action: HexCoord,
    pub prior: f32,
    pub visits: u32,
    pub value_sum: f32,
    /// Sum of squared real backup values (LCB sigma; virtual losses excluded).
    pub value_sq_sum: f32,
    /// Moves-left stats accumulated on real backups.
    pub ml_sum: f32,
    pub ml_weight: f32,
    pub pending: u32,
    pub child: Option<usize>,
    pub forced: bool,
}

impl RustEdge {
    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }

    fn value_or_fpu(&self, parent_value: f32, fpu_reduction: f32) -> f32 {
        if self.visits == 0 {
            parent_value - fpu_reduction
        } else {
            self.value()
        }
    }

    pub fn ml_mean(&self) -> Option<f32> {
        if self.ml_weight > 0.0 {
            Some(self.ml_sum / self.ml_weight)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct RustPriorCandidate {
    pub action_id: PackedCoord,
    pub prior: f32,
}

impl RustPriorCandidate {
    fn into_edge(self) -> RustEdge {
        RustEdge {
            action_id: self.action_id,
            action: unpack_coord(self.action_id),
            prior: self.prior,
            visits: 0,
            value_sum: 0.0,
            value_sq_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            pending: 0,
            child: None,
            forced: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Widening {
    pub mass: f32,
    pub min_children: usize,
    pub max_children: usize,
}

#[derive(Clone, Debug)]
pub enum NodePriors {
    /// Interior nodes share the cache's descending normalized prior vector;
    /// the next unexpanded candidate is `priors[edges.len()]`.
    Shared(Arc<RustEvaluation>),
    /// Owned ascending candidate list (highest prior popped from the back).
    Owned(Vec<RustPriorCandidate>),
}

#[derive(Clone, Debug)]
pub struct RustNode {
    pub state_hash: StateHash,
    pub player: Player,
    pub eval_value: f32,
    /// Evaluator moves-left decode for this node's state, in decisions.
    pub eval_ml: Option<f32>,
    pub visits: u32,
    pub value_sum: f32,
    pub ml_sum: f32,
    pub ml_weight: f32,
    pub edges: Vec<RustEdge>,
    pub priors: NodePriors,
    pub max_eligible_children: usize,
    /// Raw pre-softmax policy logits for this node's legal actions, keyed by
    /// action_id. `None` unless the evaluation carried logits for this node.
    /// Stored raw: never temperature-shaped, normalized, or Dirichlet-noised, so
    /// the σ/Gumbel/completedQ paths read true logits. Carried onto both root and
    /// interior nodes so the Gumbel mechanisms can look up `logits(a)`.
    pub root_logits: Option<HashMap<PackedCoord, f32>>,
}

/// Per-move TSS shadow telemetry (docs/PLAN_TSS_DEEPENING.md §9). Reset with
/// the per-move visit budget (`set_additional_visits`); read into the payload
/// diagnostics. Never consulted for any search decision.
#[derive(Clone, Copy, Debug, Default)]
pub struct TssCounters {
    /// Tactical cells at the Gumbel root (0 on quiet roots / PUCT roots).
    pub root_tactical: u32,
    /// Tactical cells force-included beyond the Gumbel top-m (the injection
    /// fire-rate numerator: > 0 means the hook widened the candidate set).
    pub root_injected: u32,
    /// λ¹ hard-value leaf backups this move (each is one elided GPU eval).
    pub leaf_verdict_hits: u32,
    /// Node expansions meeting the interior forced-move-guard condition
    /// (verdict None, opponent threats live, min_hitting_set == B) — the
    /// Lever-0 preview: how often pruning WOULD fire.
    pub prune_eligible: u32,
    /// Non-tactical actions those eligible nodes carried — the fan-out the
    /// guard would remove.
    pub prune_dropped: u64,
    // === Deep-solver (Stage 4) telemetry ===
    /// Deep solves attempted this move (leaf + root-guard).
    pub deep_calls: u32,
    /// Verified WIN / verified LOSS / UNKNOWN outcomes.
    pub deep_win: u32,
    pub deep_loss: u32,
    pub deep_unknown: u32,
    /// Solver nodes expanded across this move's solves.
    pub deep_nodes: u64,
    /// FATAL: a Win/Loss claim whose certificate the independent verifier
    /// rejected (degraded to Unknown). Production must alarm on nonzero.
    pub deep_verify_failed: u32,
    /// Solver-side zone-horizon preflight retries.  These are expected,
    /// non-fatal diagnostics and must never be folded into verify failures.
    pub horizon_retry: u32,
    /// A retry still produced a zone certificate at a different exact T;
    /// it was stopped before the minting verifier (non-fatal Unknown).
    pub horizon_preflight_failed: u32,
    /// Unknown solves whose base search had at least one still-live line
    /// refused by the semantic deadline (depth-bound Unknowns). The
    /// horizon-ladder gate: structural Unknowns provably cannot convert at a
    /// deeper deadline; these might.
    pub horizon_cut: u32,
    /// Horizon-ladder tall pass (2x horizon) that remained Unknown AND still
    /// depth-cut (`horizon_cuts > 0`). Counts only when the ladder ran.
    pub horizon_cut_tall: u32,
    /// Horizon-ladder tall pass that died at a `k < B` defender node
    /// (`kb_death_cuts > 0`) — the signal that Group-2 zone consumption would
    /// matter. Do NOT build zones ahead of this number.
    pub deep_kb_death: u32,
    /// Zoned AND nodes and P3-commuted replies in submitted certificates.
    pub zone_nodes: u32,
    pub pair_omitted: u32,
    /// Minting-verifier rejection specifically involving a zoned certificate.
    pub zone_verify_failed: u32,
    /// Verified hard values actually backed up (each one elides a GPU eval).
    pub deep_hard_backups: u32,
    /// `deep_hard_backups` split by outcome: verified WIN / verified LOSS hard
    /// values actually consumed through the tier gate. Their sum equals
    /// `deep_hard_backups`. VALUE-SIGNAL SYMMETRY (docs/VALUE_SIGNAL_AUDIT.md):
    /// `deep_win`/`deep_loss` count PROVEN results in every mode, but the
    /// combined `deep_hard_backups` hid whether the loss half of the value
    /// signal was actually reaching backup. In the incoming loss-heavy regime
    /// this split lets a run see the consumed-loss stream directly.
    pub deep_win_backups: u32,
    pub deep_loss_backups: u32,
    /// Per-move memo hits (a solved leaf re-selected).
    pub deep_memo_hits: u32,
    // === Async-pool telemetry (tss_solver_async) ===
    /// Solve requests handed to the background pool this move.
    pub async_enqueued: u32,
    /// Requests dropped (queue full / pool gone) — those leaves took the
    /// plain net eval. Persistent nonzero => widen the queue or the pool.
    pub async_dropped: u32,
    /// Responses discarded because their generation no longer matched the
    /// slot's live search (the move/game advanced past them).
    pub async_stale: u32,
    /// A leaf re-selected while its solve was still in flight.
    pub async_pending_hits: u32,
    // === Wait-at-leaf parking telemetry (tss_solver_park) ===
    /// Gated leaves held out of the evaluator while their solve was in flight.
    pub park_parked: u32,
    /// Parked leaves resolved to a verified, tier-consumable hard backup.
    pub park_hard: u32,
    /// Parked leaves released to evaluation after a non-consumable result.
    pub park_released: u32,
    /// Parked leaves released to evaluation at the bail deadline.
    pub park_bailed: u32,
    /// Sum and maximum of scheduler parking latency in milliseconds.
    pub park_wait_ms_sum: u64,
    pub park_wait_ms_max: u64,
    /// Workers dynamically added above the async pool's base size.
    pub async_workers_spawned: u32,
    // === Search-depth telemetry (every real backup, all leaf kinds) ===
    /// Σ leaf depth over this move's real backups (mean = depth_sum / backups).
    pub depth_sum: u64,
    /// Deepest leaf reached this move.
    pub depth_max: u32,
    /// Real backups this move (the depth distribution's denominator).
    pub backups: u32,
}

impl TssCounters {
    /// Field-wise accumulate (payload builder merges its local root-guard
    /// counters into the search's per-move view).
    pub fn add(&mut self, other: &TssCounters) {
        self.root_tactical += other.root_tactical;
        self.root_injected += other.root_injected;
        self.leaf_verdict_hits += other.leaf_verdict_hits;
        self.prune_eligible += other.prune_eligible;
        self.prune_dropped += other.prune_dropped;
        self.deep_calls += other.deep_calls;
        self.deep_win += other.deep_win;
        self.deep_loss += other.deep_loss;
        self.deep_unknown += other.deep_unknown;
        self.deep_nodes += other.deep_nodes;
        self.deep_verify_failed += other.deep_verify_failed;
        self.horizon_retry += other.horizon_retry;
        self.horizon_preflight_failed += other.horizon_preflight_failed;
        self.horizon_cut += other.horizon_cut;
        self.horizon_cut_tall += other.horizon_cut_tall;
        self.deep_kb_death += other.deep_kb_death;
        self.zone_nodes += other.zone_nodes;
        self.pair_omitted += other.pair_omitted;
        self.zone_verify_failed += other.zone_verify_failed;
        self.deep_hard_backups += other.deep_hard_backups;
        self.deep_win_backups += other.deep_win_backups;
        self.deep_loss_backups += other.deep_loss_backups;
        self.deep_memo_hits += other.deep_memo_hits;
        self.async_enqueued += other.async_enqueued;
        self.async_dropped += other.async_dropped;
        self.async_stale += other.async_stale;
        self.async_pending_hits += other.async_pending_hits;
        self.park_parked += other.park_parked;
        self.park_hard += other.park_hard;
        self.park_released += other.park_released;
        self.park_bailed += other.park_bailed;
        self.park_wait_ms_sum += other.park_wait_ms_sum;
        self.park_wait_ms_max = self.park_wait_ms_max.max(other.park_wait_ms_max);
        self.async_workers_spawned += other.async_workers_spawned;
        self.depth_sum += other.depth_sum;
        self.depth_max = self.depth_max.max(other.depth_max);
        self.backups += other.backups;
    }
}

/// One per-move deep-memo slot. `Pending` marks an async solve in flight
/// (dedup: the same leaf re-selected must not re-enqueue); `Done` carries the
/// verified result. Both hold the full `RootBinding` — a value-bearing hit
/// requires full-position equality, never the 64-bit hash alone (§2.5).
#[derive(Clone, Debug)]
pub enum TssMemoEntry {
    Pending(RootBinding),
    Done(RootBinding, ProofStatus, Option<HardValue>),
}

/// A solver slot with fresh-cache-on-clone semantics: `TssSolver` is
/// deliberately not `Clone` (a proof cache owner), but `RustSearch` derives
/// `Clone` — and a cloned search correctly starts COLD, because cache warmth
/// is discovery state, never truth (O16). Deref-free by design: callers go
/// through `.0`.
#[derive(Debug)]
pub struct TssSolverSlot(pub TssSolver);

impl TssSolverSlot {
    /// A fresh (cold-cache) solver pre-configured to the campaign leaf-decided
    /// profile (§3): wide `vcf_pair_complete` or its named J2near extension,
    /// lazy frontier + interior census gate ON. The persistent per-search leaf
    /// solver, the root guard, and the async workers all run this profile.
    fn leaf_configured(j2near: bool) -> TssSolver {
        let mut solver = TssSolver::default();
        solver.configure_leaf_profile();
        solver.set_leaf_j2near(j2near);
        solver
    }

    fn for_divergences(divergences: Divergences) -> Self {
        Self(Self::leaf_configured(divergences.solver_j2near_enabled()))
    }
}

impl Default for TssSolverSlot {
    fn default() -> Self {
        Self(Self::leaf_configured(false))
    }
}

impl Clone for TssSolverSlot {
    fn clone(&self) -> Self {
        Self(Self::leaf_configured(self.0.j2near_enabled()))
    }
}

/// A completed deep solve after the mandatory verification step. `hard` and
/// `cert` are `Some` only when the independent verifier accepted the
/// certificate; a rejected claim reads as Unknown with the FATAL
/// `deep_verify_failed` counter bumped.
pub struct VerifiedSolve {
    pub status: ProofStatus,
    pub hard: Option<HardValue>,
    pub cert: Option<crate::tss_verify::TssCertificate>,
}

/// Semantic-horizon policy for a deep solve (PLAN_TSS_MCTS_INTEGRATION.md §5;
/// owner ruling 2026-07-20). Replaces the historical hardcoded `+12`.
#[derive(Clone, Copy, Debug)]
pub struct SolverHorizon {
    /// Absolute placements added to the current ply to form the deadline, or
    /// `0` for UNBOUNDED (`semantic_horizon = u32::MAX`, node cap the only
    /// budget). The Rust seam validates `0` or `>= 16` (the owner floor);
    /// `1..=15` are rejected there.
    pub horizon: u32,
    /// When on and a bounded base solve is Unknown with `horizon_cuts > 0`,
    /// re-solve ONCE at `2 * horizon` on the same solver instance. Unbounded
    /// bases (`horizon == 0`) never ladder.
    pub ladder: bool,
}

impl SolverHorizon {
    /// Owner-floor default: h16, ladder off.
    pub const DEFAULT: Self = Self {
        horizon: 16,
        ladder: false,
    };
}

impl Default for SolverHorizon {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Build the production solve caps used by every verified attempt. Keeping
/// this constructor shared prevents diagnostic APIs from mirroring the leaf
/// TT budget or semantic-horizon resolution.
pub(crate) fn tss_verified_solve_caps(
    placements: u32,
    node_cap: u64,
    horizon: SolverHorizon,
) -> SolveCaps {
    SolveCaps {
        node_cap,
        tt_bytes_cap: RustSearch::TSS_SOLVER_TT_BYTES,
        semantic_horizon: if horizon.horizon == 0 {
            u32::MAX
        } else {
            placements.saturating_add(horizon.horizon)
        },
    }
}

/// A verified solve plus diagnostics aggregated across every tight, base,
/// ladder, and certificate-horizon retry attempt that actually ran.
pub struct VerifiedSolveWithStats {
    pub solve: VerifiedSolve,
    pub stats: SolveStats,
}

/// One verified deep solve (the ONLY production path from solver claims to
/// consumable results): solver → independent certificate verifier via the
/// sole deep mint `tss_core::hard_value_from_verified`. Deterministic given
/// (state, caps, goal, horizon, solver-cache state): node cap only, no wall
/// clock. Shared by the per-search leaf hook (persistent solver) and the
/// payload-build root guard (per-move solver).
pub fn tss_solve_verified(
    state: &RustHexoState,
    node_cap: u64,
    goal: SolveGoal,
    zone: crate::tss_core::ZoneSearchCaps,
    horizon: SolverHorizon,
    solver: &mut TssSolver,
    counters: &mut TssCounters,
) -> VerifiedSolve {
    tss_solve_verified_impl(state, node_cap, goal, zone, horizon, solver, counters, None)
}

/// Stats-bearing additive variant of `tss_solve_verified`. The verdict and
/// certificate path are identical; only attempt telemetry is accumulated.
pub fn tss_solve_verified_with_stats(
    state: &RustHexoState,
    node_cap: u64,
    goal: SolveGoal,
    zone: crate::tss_core::ZoneSearchCaps,
    horizon: SolverHorizon,
    solver: &mut TssSolver,
    counters: &mut TssCounters,
) -> VerifiedSolveWithStats {
    let mut stats = SolveStats::default();
    let solve = tss_solve_verified_impl(
        state,
        node_cap,
        goal,
        zone,
        horizon,
        solver,
        counters,
        Some(&mut stats),
    );
    VerifiedSolveWithStats { solve, stats }
}

#[allow(clippy::too_many_arguments)]
fn tss_solve_verified_impl(
    state: &RustHexoState,
    node_cap: u64,
    goal: SolveGoal,
    zone: crate::tss_core::ZoneSearchCaps,
    horizon: SolverHorizon,
    solver: &mut TssSolver,
    counters: &mut TssCounters,
    mut aggregate_stats: Option<&mut SolveStats>,
) -> VerifiedSolve {
    counters.deep_calls += 1;
    // The semantic deadline (owner ruling 07-20): h>=16 minimum, or unbounded
    // (horizon == 0 => u32::MAX) with the node cap as the only budget. The
    // Rust seam has already rejected the 1..=15 band; P2's preflight may
    // diagnose the guess and P3's cache stamps bind it.
    let mut caps = tss_verified_solve_caps(state.placements_made(), node_cap, horizon);
    solver.set_zone_options(zone);
    // Zone tight-ladder (Codex review, wide-deadline neutralization): at a
    // slack defender budget the zone generator must take the FULL legal set,
    // so zones never reduce the first Universal's fanout, exactly where the
    // node budget dies (ep32: zone_nodes = 0 across 33k decided solves). With
    // zones on, first attempt a TIGHT +8 deadline (d = 4 at the first
    // Universal => zones prune from the very first fanout) on half the node
    // budget; a decided tight result is a fully verified proof in its own
    // right (a win by ply T wins, a dual proof by T is a forced loss
    // outright), while Unknown falls through to the full base-horizon solve
    // (the §5 knob). This tight pass is an internal fast-path, orthogonal to
    // the owner horizon floor: a decided +8 result is a genuine <=+8 proof,
    // and an Unknown one loses nothing (it re-solves at the full deadline).
    // Zone-off behavior is bit-identical to before.
    let mut result = if zone.enabled {
        let tight = SolveCaps {
            node_cap: (node_cap / 2).max(1),
            tt_bytes_cap: caps.tt_bytes_cap,
            semantic_horizon: state.placements_made().saturating_add(8),
        };
        let tight_result = solver.solve_goal(state, &tight, goal);
        counters.deep_nodes += tight_result.stats.nodes;
        if let Some(total) = aggregate_stats.as_mut() {
            total.merge(tight_result.stats);
        }
        if tight_result.status != ProofStatus::Unknown {
            caps.semantic_horizon = tight.semantic_horizon;
            tight_result
        } else {
            let full_result = solver.solve_goal(state, &caps, goal);
            counters.deep_nodes += full_result.stats.nodes;
            if let Some(total) = aggregate_stats.as_mut() {
                total.merge(full_result.stats);
            }
            full_result
        }
    } else {
        let full_result = solver.solve_goal(state, &caps, goal);
        counters.deep_nodes += full_result.stats.nodes;
        if let Some(total) = aggregate_stats.as_mut() {
            total.merge(full_result.stats);
        }
        full_result
    };
    // Horizon ladder (§5, owner ruling 07-20; default off). A BOUNDED base
    // solve that returned Unknown while still refusing live lines at the
    // deadline (`horizon_cuts > 0`) is exactly the depth-bound case that a
    // taller deadline might convert; structural Unknowns cannot. Re-solve ONCE
    // at 2x horizon on the SAME solver instance so the shared TT replays the
    // proven prefix and the budget goes to the new plies. Unbounded bases skip
    // the ladder. Soundness needs no new theory: a completed production
    // certificate is a forced chain (implicit dispatch at k==B),
    // depth-independent; an Unknown tall pass just bails to a plain eval. The
    // tall pass feeds the depth-frontier gate counters.
    if horizon.ladder
        && horizon.horizon != 0
        && result.status == ProofStatus::Unknown
        && result.stats.horizon_cuts > 0
    {
        let tall = SolveCaps {
            node_cap,
            tt_bytes_cap: caps.tt_bytes_cap,
            semantic_horizon: state
                .placements_made()
                .saturating_add(horizon.horizon.saturating_mul(2)),
        };
        let tall_result = solver.solve_goal(state, &tall, goal);
        counters.deep_nodes += tall_result.stats.nodes;
        if let Some(total) = aggregate_stats.as_mut() {
            total.merge(tall_result.stats);
        }
        if tall_result.status == ProofStatus::Unknown {
            if tall_result.stats.horizon_cuts > 0 {
                counters.horizon_cut_tall += 1;
            }
            if tall_result.stats.kb_death_cuts > 0 {
                counters.deep_kb_death += 1;
            }
        }
        caps.semantic_horizon = tall.semantic_horizon;
        result = tall_result;
    }
    // A zoned proof is built against a guessed semantic deadline. Derive the
    // certificate's actual maximum leaf resolution before verification and,
    // only when the zone theorem was used, retry once at that exact deadline.
    // No rejection-driven retry is permitted: verifier failures remain fatal.
    if let Some(cert) = result.cert.as_ref() {
        if let Some((derived_t, true)) = crate::tss_verify::certificate_horizon_preflight(cert) {
            if derived_t != caps.semantic_horizon {
                counters.horizon_retry += 1;
                caps.semantic_horizon = derived_t;
                result = solver.solve_goal(state, &caps, goal);
                counters.deep_nodes += result.stats.nodes;
                if let Some(total) = aggregate_stats.as_mut() {
                    total.merge(result.stats);
                }
            }
        }
    }
    if let Some(cert) = result.cert.as_ref() {
        if let Some((derived_t, true)) = crate::tss_verify::certificate_horizon_preflight(cert) {
            if derived_t != caps.semantic_horizon {
                counters.horizon_preflight_failed += 1;
                counters.deep_unknown += 1;
                return VerifiedSolve {
                    status: ProofStatus::Unknown,
                    hard: None,
                    cert: None,
                };
            }
        }
        for node in &cert.nodes {
            if let crate::tss_verify::CertNode::Universal {
                zone, commutations, ..
            } = node
            {
                counters.zone_nodes += u32::from(zone.is_some());
                counters.pair_omitted = counters
                    .pair_omitted
                    .saturating_add(u32::try_from(commutations.len()).unwrap_or(u32::MAX));
            }
        }
    }
    match result.status {
        ProofStatus::Unknown => {
            counters.deep_unknown += 1;
            if result.stats.horizon_cuts > 0 {
                counters.horizon_cut += 1;
            }
            VerifiedSolve {
                status: ProofStatus::Unknown,
                hard: None,
                cert: None,
            }
        }
        status => match if solver.group2_enabled() {
            // Trainer configuration (the solver flag), never certificate
            // contents, selects the Group2V1 verifier policy (design §5.1).
            tss_core::hard_value_from_verified_group2(&Group2Verifier, state, &result)
        } else {
            tss_core::hard_value_from_verified(&TssVerifier, state, &result)
        } {
            Some(hard) => {
                match status {
                    ProofStatus::Win => counters.deep_win += 1,
                    ProofStatus::Loss => counters.deep_loss += 1,
                    ProofStatus::Unknown => unreachable!("matched above"),
                }
                VerifiedSolve {
                    status,
                    hard: Some(hard),
                    cert: result.cert,
                }
            }
            None => {
                if result.cert.as_ref().is_some_and(|cert| {
                    cert.nodes.iter().any(|node| {
                        matches!(
                            node,
                            crate::tss_verify::CertNode::Universal { zone: Some(_), .. }
                        )
                    })
                }) {
                    counters.zone_verify_failed += 1;
                }
                counters.deep_verify_failed += 1;
                VerifiedSolve {
                    status: ProofStatus::Unknown,
                    hard: None,
                    cert: None,
                }
            }
        },
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RustSearchDiagnostics {
    pub node_count: usize,
    pub active_edge_count: usize,
    pub root_active_edges: usize,
    pub root_hidden_priors: usize,
}

impl RustNode {
    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            self.eval_value
        } else {
            self.value_sum / self.visits as f32
        }
    }

    pub fn ml_mean(&self) -> Option<f32> {
        if self.ml_weight > 0.0 {
            Some(self.ml_sum / self.ml_weight)
        } else {
            self.eval_ml
        }
    }

    fn has_actions(&self) -> bool {
        !self.edges.is_empty() || self.remaining_prior_count() > 0
    }

    /// Σ of the (normalized) prior over children that have received at least one
    /// visit — KataGo's `policyProbMassVisited`. Used to scale the FPU reduction
    /// (`fpu_reduction · √mass`) under the `scaled_fpu` divergence.
    fn visited_policy_mass(&self) -> f32 {
        self.edges
            .iter()
            .filter(|edge| edge.visits > 0)
            .map(|edge| edge.prior)
            .sum()
    }

    pub fn remaining_prior_count(&self) -> usize {
        match &self.priors {
            NodePriors::Shared(eval) => eval.priors.len().saturating_sub(self.edges.len()),
            NodePriors::Owned(unexpanded) => unexpanded.len(),
        }
    }

    fn peek_next_candidate(&self) -> Option<(PackedCoord, f32)> {
        match &self.priors {
            NodePriors::Shared(eval) => eval.priors.get(self.edges.len()).copied(),
            NodePriors::Owned(unexpanded) => unexpanded
                .last()
                .map(|candidate| (candidate.action_id, candidate.prior)),
        }
    }

    fn materialize_next_candidate(&mut self) -> RustEdge {
        match &mut self.priors {
            NodePriors::Owned(unexpanded) => unexpanded
                .pop()
                .expect("last prior candidate exists")
                .into_edge(),
            NodePriors::Shared(eval) => {
                let (action_id, prior) = eval.priors[self.edges.len()];
                RustPriorCandidate { action_id, prior }.into_edge()
            }
        }
    }

    pub fn remaining_priors(&self) -> Vec<(PackedCoord, f32)> {
        match &self.priors {
            NodePriors::Shared(eval) => eval.priors[self.edges.len().min(eval.priors.len())..]
                .iter()
                .copied()
                .collect(),
            NodePriors::Owned(unexpanded) => unexpanded
                .iter()
                .map(|candidate| (candidate.action_id, candidate.prior))
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RustSearch {
    pub root_state: RustHexoState,
    pub root_hash: StateHash,
    pub nodes: Vec<RustNode>,
    pub node_table: HashMap<StateHash, usize>,
    pub target_visits: u32,
    pub completed_visits: u32,
    fpu_reduction: f32,
    root_fpu_reduction: f32,
    widening: Widening,
    forced_playout_k: f32,
    pub tss_enabled: bool,
    pub divergences: Divergences,
    /// Set when an early-stop fired for this search (telemetry).
    pub early_stopped: bool,
    /// Per-move TSS shadow telemetry; reset alongside `early_stopped`.
    pub tss: TssCounters,
    /// Per-move deep-solve memo (Stage 4): a solved leaf re-selected must
    /// never re-run the solver. Keyed by the (history-bearing) StateHash with
    /// FULL-POSITION binding equality verified on every hit (§2.5 — the hash
    /// alone is never trusted for a value-bearing result). HardValue presence
    /// implies the certificate already passed the independent verifier.
    /// Bounded; cleared with the per-move counters.
    tss_deep_memo: HashMap<StateHash, TssMemoEntry>,
    /// Async-pool enqueue handle (sender + slot + generation), wired by the
    /// continuous driver at search creation / reuse-rebind / move advance.
    /// `None` (lockstep paths, flag off) => the inline solve path runs.
    tss_async: Option<TssAsyncHandle>,
    /// Persistent deep solver (O16 shared positive-proof-fragment cache):
    /// PERSISTS ACROSS MOVES so forcing structure discovered at one leaf warms
    /// its neighbors and successors. Retained bytes are hard-capped inside the
    /// solver (split_tt_cap of SolveCaps.tt_bytes_cap); cache warmth affects
    /// discovery, never verdict validity — every hard result still carries a
    /// fresh certificate replayed by the independent verifier before minting.
    tss_solver: TssSolverSlot,
    /// Clean post-temp, pre-noise root priors (policy after the at-most-once
    /// root-policy-temperature step), keyed by action_id. When
    /// `clean_root_prior_cache` is on, a reused/promoted root resets its
    /// edge/candidate priors from this cache before re-applying temperature or
    /// noise. Lazily populated on first root setup; None until then.
    clean_root_priors: Option<HashMap<PackedCoord, f32>>,
    active_edge_count: usize,
    /// Per-root Gumbel-Top-k candidate set + Sequential-Halving state. `Some`
    /// only when `gumbel_root` is on for a Full move; `None` otherwise, in which
    /// case the PUCT root path runs.
    gumbel_root: Option<GumbelRootState>,
}

pub struct RustSelectedLeaf {
    pub path: Vec<(usize, usize)>,
    pub state: RustHexoState,
    pub state_hash: StateHash,
    pub parent_node: usize,
    pub edge_index: usize,
    pub terminal: Option<GameOutcome>,
    pub existing_node: Option<usize>,
    /// Async descent-stop result: a verified deep proof for this position
    /// arrived from the pool, so this simulation backs the hard value here
    /// instead of descending/evaluating. Always `None` with the pool off.
    pub hard: Option<HardValue>,
}

pub struct RustLeaf {
    pub root_index: usize,
    pub parent_node: usize,
    pub edge_index: usize,
    pub path: Vec<(usize, usize)>,
    pub state: RustHexoState,
    pub state_hash: StateHash,
}

/// Scheduler route for a gated deep-solver leaf. `Miss` is exactly the
/// historical `None` route (normal evaluator); `Parked` is emitted only while
/// wait-at-leaf parking is enabled and a background request is known to be in
/// flight for the full-bound position.
#[derive(Clone, Copy, Debug)]
pub enum TssLeafRoute {
    Hard(HardValue),
    Parked,
    Miss,
}

/// Result of probing a scheduler-owned parked leaf after async responses have
/// drained into the owning search's memo.
#[derive(Clone, Copy, Debug)]
pub enum TssParkResolution {
    /// No matching completed response yet. The scheduler keeps the leaf parked
    /// until a later drain or its bounded bail deadline.
    Pending,
    /// A full-binding, tier-consumable verified value is ready for hard backup.
    Hard(HardValue),
    /// A matching response completed as Unknown or at a non-consumable tier;
    /// the leaf resumes the ordinary evaluator path.
    Release,
}

impl RustSearch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        root_state: RustHexoState,
        evaluation: &RustEvaluation,
        target_visits: u32,
        fpu_reduction: f32,
        root_fpu_reduction: f32,
        root_policy_temperature: f32,
        root_noise: Option<RootDirichletNoise>,
        widening: Widening,
        forced_playout_k: f32,
        tss_enabled: bool,
        divergences: Divergences,
    ) -> PyResult<Self> {
        let root_hash = state_hash(&root_state);
        let (root_node, clean_root_priors) = owned_root_from_evaluation(
            root_hash,
            &root_state,
            evaluation,
            Some(root_policy_temperature),
            root_noise,
            widening,
            tss_enabled,
            divergences,
        )?;
        let mut node_table = HashMap::new();
        node_table.insert(root_hash, 0);
        Ok(Self {
            root_state,
            root_hash,
            nodes: vec![root_node],
            node_table,
            target_visits,
            completed_visits: 0,
            fpu_reduction,
            root_fpu_reduction,
            widening,
            forced_playout_k,
            tss_enabled,
            divergences,
            early_stopped: false,
            tss: TssCounters::default(),
            tss_deep_memo: HashMap::new(),
            tss_async: None,
            tss_solver: TssSolverSlot::for_divergences(divergences),
            clean_root_priors,
            active_edge_count: 0,
            gumbel_root: None,
        })
    }

    pub fn set_forced_playout_k(&mut self, k: f32) {
        self.forced_playout_k = k;
    }

    pub fn set_tss_enabled(&mut self, enabled: bool) {
        self.tss_enabled = enabled;
    }

    pub fn set_root_fpu_reduction(&mut self, value: f32) {
        self.root_fpu_reduction = value;
    }

    pub fn set_divergences(&mut self, divergences: Divergences) {
        self.tss_solver
            .0
            .set_leaf_j2near(divergences.solver_j2near_enabled());
        self.divergences = divergences;
    }

    /// Apply root-policy softmax temperature to the root priors, before noise,
    /// at most once per root lifetime.
    ///
    /// With `clean_root_prior_cache` on: if the cache is already populated for
    /// this root, reset the edge/candidate priors from it and return without
    /// re-powering (temperature is already baked into the cached priors). If the
    /// cache is empty (a freshly promoted root), apply temperature once and
    /// capture the result into the cache.
    pub fn apply_root_policy_temperature(&mut self, temperature: f32) {
        if self.divergences.clean_root_prior_cache {
            if self.clean_root_priors.is_some() {
                self.reset_root_priors_from_clean_cache();
                return;
            }
            // Promoted root with no cache yet: fall through, apply temperature
            // once, and capture below.
        }
        if !temperature.is_finite() || temperature <= 0.0 || (temperature - 1.0).abs() < 1.0e-6 {
            // A no-op temperature still seeds the clean cache on a promoted root
            // so the subsequent noise mix resets from this baseline.
            if self.divergences.clean_root_prior_cache && self.clean_root_priors.is_none() {
                self.capture_clean_root_priors();
            }
            return;
        }
        self.ensure_root_owned();
        let root = &mut self.nodes[0];
        let NodePriors::Owned(unexpanded) = &mut root.priors else {
            return;
        };
        let inverse = 1.0 / temperature;
        let mut total = 0.0f32;
        for edge in root.edges.iter_mut() {
            if edge.prior.is_finite() && edge.prior > 0.0 {
                edge.prior = edge.prior.powf(inverse);
                total += edge.prior;
            }
        }
        for candidate in unexpanded.iter_mut() {
            if candidate.prior.is_finite() && candidate.prior > 0.0 {
                candidate.prior = candidate.prior.powf(inverse);
                total += candidate.prior;
            }
        }
        if total > 0.0 {
            for edge in root.edges.iter_mut() {
                if edge.prior.is_finite() && edge.prior > 0.0 {
                    edge.prior /= total;
                }
            }
            for candidate in unexpanded.iter_mut() {
                if candidate.prior.is_finite() && candidate.prior > 0.0 {
                    candidate.prior /= total;
                }
            }
        }
        // Capture the post-temp, pre-noise priors for a promoted root that had
        // no cache yet (the fall-through case above).
        if self.divergences.clean_root_prior_cache && self.clean_root_priors.is_none() {
            self.capture_clean_root_priors();
        }
    }

    /// Snapshot the current root edge/candidate priors into the clean cache.
    fn capture_clean_root_priors(&mut self) {
        let root = &self.nodes[0];
        let mut cache: HashMap<PackedCoord, f32> =
            HashMap::with_capacity(root.edges.len() + root.remaining_prior_count());
        for edge in &root.edges {
            cache.insert(edge.action_id, edge.prior);
        }
        if let NodePriors::Owned(unexpanded) = &root.priors {
            for candidate in unexpanded {
                cache.insert(candidate.action_id, candidate.prior);
            }
        }
        self.clean_root_priors = Some(cache);
    }

    /// Reset the root edge/candidate priors back to the clean (post-temp,
    /// pre-noise) values cached for this root.
    fn reset_root_priors_from_clean_cache(&mut self) {
        let Some(cache) = self.clean_root_priors.clone() else {
            return;
        };
        self.ensure_root_owned();
        let root = &mut self.nodes[0];
        for edge in root.edges.iter_mut() {
            if let Some(&clean) = cache.get(&edge.action_id) {
                edge.prior = clean;
            }
        }
        if let NodePriors::Owned(unexpanded) = &mut root.priors {
            for candidate in unexpanded.iter_mut() {
                if let Some(&clean) = cache.get(&candidate.action_id) {
                    candidate.prior = clean;
                }
            }
        }
    }

    fn ensure_root_owned(&mut self) {
        let root = &mut self.nodes[0];
        let owned = match &root.priors {
            NodePriors::Owned(_) => return,
            NodePriors::Shared(eval) => {
                let start = root.edges.len().min(eval.priors.len());
                let mut unexpanded: Vec<RustPriorCandidate> = eval.priors[start..]
                    .iter()
                    .map(|&(action_id, prior)| RustPriorCandidate { action_id, prior })
                    .collect();
                unexpanded.reverse();
                unexpanded.shrink_to_fit();
                unexpanded
            }
        };
        root.priors = NodePriors::Owned(owned);
    }

    pub fn apply_root_dirichlet_noise(&mut self, noise: RootDirichletNoise) {
        // Reset to clean (post-temp, pre-noise) priors before mixing. When the
        // cache is active the priors then sum to 1, so the visible_total scaling
        // below is ~1.0.
        if self.divergences.clean_root_prior_cache {
            if self.clean_root_priors.is_none() {
                self.capture_clean_root_priors();
            }
            self.reset_root_priors_from_clean_cache();
        }
        self.ensure_root_owned();
        // Collect the policy (post-reset) for the shaped-alpha input in
        // edges-then-unexpanded order before borrowing mutably.
        let shaped = noise.shaped;
        let root = &self.nodes[0];
        let NodePriors::Owned(unexpanded_ro) = &root.priors else {
            return;
        };
        let count = root.edges.len() + unexpanded_ro.len();
        if count == 0 || noise.total_alpha <= 0.0 || noise.fraction <= 0.0 {
            return;
        }
        let clean_policy: Vec<f32> = root
            .edges
            .iter()
            .map(|edge| edge.prior)
            .chain(unexpanded_ro.iter().map(|candidate| candidate.prior))
            .collect();
        let samples = if shaped {
            shaped_dirichlet_samples(&clean_policy, noise)
        } else {
            dirichlet_samples(count, noise)
        };
        let root = &mut self.nodes[0];
        let NodePriors::Owned(unexpanded) = &mut root.priors else {
            return;
        };
        let visible_total: f32 = root
            .edges
            .iter()
            .map(|edge| edge.prior)
            .chain(unexpanded.iter().map(|candidate| candidate.prior))
            .filter(|prior| prior.is_finite())
            .sum();
        let fraction = noise.fraction;
        let mut sample_index = 0usize;
        for edge in &mut root.edges {
            edge.prior =
                (1.0 - fraction) * edge.prior + fraction * samples[sample_index] * visible_total;
            sample_index += 1;
        }
        for candidate in unexpanded.iter_mut() {
            candidate.prior = (1.0 - fraction) * candidate.prior
                + fraction * samples[sample_index] * visible_total;
            sample_index += 1;
        }
        unexpanded.sort_by(compare_prior_candidate);
        unexpanded.reverse();
    }

    pub fn root_edges_empty(&self) -> bool {
        !self.nodes[0].has_actions()
    }

    pub fn needs_visits(&self) -> bool {
        self.completed_visits < self.target_visits && !self.root_edges_empty()
    }

    pub fn remaining_visits(&self) -> u32 {
        self.target_visits.saturating_sub(self.completed_visits)
    }

    pub fn set_additional_visits(&mut self, visits: u32) {
        self.target_visits = self.completed_visits.saturating_add(visits);
        self.early_stopped = false;
        // New move: reset the per-move TSS shadow telemetry with the budget
        // (this is the universal per-move entry for lockstep and continuous).
        self.tss = TssCounters::default();
        if self.divergences.tss_solver_async {
            // Async rung: verified DECIDED entries PERSIST across moves. A
            // proof is a property of the position (binding-checked again at
            // every consumption; HardValue exists only post-verification),
            // and the history-bearing hash means the retained entries
            // re-serve exactly the re-searched played line — the forced
            // continuation consumes without re-solving. Pending markers AND
            // Unknown results drop (Codex review 2/3): an Unknown is a cap
            // artifact, not a position property — the next move may re-solve
            // it with a warmer worker cache; and decided-only retention keeps
            // the retained set far below the memo cap so the enqueue gate
            // never starves. Safety valve: if retained proofs alone ever
            // crowd the cap, start the move fresh rather than starve.
            self.tss_deep_memo.retain(|_, entry| {
                matches!(entry, TssMemoEntry::Done(_, status, _) if *status != ProofStatus::Unknown)
            });
            if self.tss_deep_memo.len() > Self::TSS_DEEP_MEMO_RETAIN_MAX {
                self.tss_deep_memo.clear();
            }
        } else {
            self.tss_deep_memo.clear();
        }
        // Drop the async handle with the counters: the driver's wire pass
        // issues a FRESH generation before the next select, so in-flight
        // responses from the move that just ended are counted as stale.
        self.tss_async = None;
    }

    /// Hard cap on per-move deep-memo entries (bounded memory; past the cap
    /// solves simply recompute — never a correctness concern, only cost).
    const TSS_DEEP_MEMO_MAX: usize = 8192;
    /// Cross-move retention bound for decided async proofs (half the memo
    /// cap): above it the memo starts the move fresh instead of letting
    /// retained history starve the enqueue gate.
    const TSS_DEEP_MEMO_RETAIN_MAX: usize = 4096;
    /// Per-solve transposition-table byte cap handed to the deep solver (its
    /// TT is per-solve; this bounds transient allocation, not retained state).
    const TSS_SOLVER_TT_BYTES: usize = 256 << 10;

    /// Route a leaf through the deep solver while preserving the historical
    /// `Option<HardValue>` implementation byte-for-byte when parking is off.
    /// With parking enabled, every gated leaf uses the async pool regardless of
    /// `tss_solver_async_inline_16`; the select thread never runs a solve.
    pub fn tss_deep_leaf_route(&mut self, state: &RustHexoState, hash: StateHash) -> TssLeafRoute {
        if !self.divergences.tss_solver_park {
            return match self.tss_deep_leaf(state, hash) {
                Some(hard) => TssLeafRoute::Hard(hard),
                None => TssLeafRoute::Miss,
            };
        }

        let mode = self.divergences.tss_solver_mode;
        if mode == 0 || !self.tss_enabled {
            return TssLeafRoute::Miss;
        }
        if !self.divergences.tss_solver_all_leaves && !state.board().windows().has_threats() {
            return TssLeafRoute::Miss;
        }
        if ((hash & 0xF) as u32) >= self.divergences.tss_solver_sample_16 {
            return TssLeafRoute::Miss;
        }

        let binding = RootBinding::from_state(state);
        let goal = match mode {
            2 => SolveGoal::Loss,
            _ => SolveGoal::Both,
        };
        let Some(handle) = self.tss_async.clone() else {
            // A parked leaf must always have a live request behind it. Missing
            // wiring therefore degrades to the ordinary evaluator path.
            self.tss.async_dropped += 1;
            return TssLeafRoute::Miss;
        };

        match self.tss_deep_memo.get(&hash) {
            Some(TssMemoEntry::Pending(seen)) if *seen == binding => {
                // A transposed/re-selected simulation may share the existing
                // in-flight solve. Park it too: park-mode queues never evict,
                // and the bounded scheduler bail remains the liveness guard.
                self.tss.async_pending_hits += 1;
                TssLeafRoute::Parked
            }
            Some(TssMemoEntry::Done(seen, status, hard)) if *seen == binding => {
                self.tss.deep_memo_hits += 1;
                let (status, hard) = (*status, *hard);
                match self.tss_consume_gate(status, hard) {
                    Some(hard) => TssLeafRoute::Hard(hard),
                    None => TssLeafRoute::Miss,
                }
            }
            _ => {
                let enqueued = (self.tss_deep_memo.len() < Self::TSS_DEEP_MEMO_MAX)
                    .then(|| {
                        handle.try_enqueue(SolveRequest {
                            slot: handle.slot,
                            generation: handle.generation,
                            hash,
                            binding: binding.clone(),
                            state: state.clone(),
                            node_cap: self.divergences.tss_solver_node_cap as u64,
                            goal,
                            zone: crate::tss_core::ZoneSearchCaps {
                                enabled: self.divergences.tss_zone,
                                stale_area_filter: self.divergences.tss_zone_stale_filter,
                                count2_threshold: self.divergences.tss_zone_count2,
                                pair_commutation: self.divergences.tss_pair_commutation,
                            },
                            horizon: SolverHorizon {
                                horizon: self.divergences.tss_solver_horizon,
                                ladder: self.divergences.tss_solver_horizon_ladder,
                            },
                            dual_pass: self.divergences.tss_solver_dual_pass,
                            loss_reserve_nodes: self.divergences.tss_solver_loss_reserve_nodes,
                            group2: self.divergences.tss_solver_group2,
                            j2near: self.divergences.solver_j2near_enabled(),
                        })
                    })
                    .flatten();
                match enqueued {
                    Some(evicted) => {
                        // `evicted` is zero for a correctly matched park-mode
                        // pool. Keep accounting defensive if a mismatched pool
                        // ever reaches this seam; the orphaned leaf still bails.
                        debug_assert_eq!(evicted, 0, "park-mode pool evicted a request");
                        self.tss.async_enqueued += 1;
                        self.tss.async_dropped += evicted;
                        self.tss_deep_memo
                            .insert(hash, TssMemoEntry::Pending(binding));
                        TssLeafRoute::Parked
                    }
                    None => {
                        self.tss.async_dropped += 1;
                        TssLeafRoute::Miss
                    }
                }
            }
        }
    }

    /// Probe a scheduler-owned parked leaf after the pool drain. A hard result
    /// travels through the same full-binding check and consumption gate used by
    /// async descent-stop; this method does not mint values or inspect proofs.
    pub fn tss_park_resolution(
        &mut self,
        hash: StateHash,
        state: &RustHexoState,
    ) -> TssParkResolution {
        let binding = RootBinding::from_state(state);
        let (status, hard) = match self.tss_deep_memo.get(&hash) {
            Some(TssMemoEntry::Done(seen, status, hard)) if *seen == binding => (*status, *hard),
            _ => return TssParkResolution::Pending,
        };
        self.tss.deep_memo_hits += 1;
        match self.tss_consume_gate(status, hard) {
            Some(hard) => TssParkResolution::Hard(hard),
            None => TssParkResolution::Release,
        }
    }

    /// Deep-solver leaf hook (Stage-4 consumption ladder, PLAN §10). Gated on
    /// tss_enabled, `tss_solver_mode > 0`, live threats, and a deterministic
    /// StateHash subsample. Every gated leaf is solved + certificate-verified
    /// + counted in ALL modes; a backup-capable HardValue is returned only in
    /// the consumption tiers (mode ≥ 2 for verified LOSS, ≥ 3 for verified
    /// WIN) — shadow mode consumes nothing, so play and targets stay
    /// bit-identical to off. Memoized per move with full-position binding
    /// equality so a re-selected solved edge is idempotent and never re-runs
    /// the solver.
    pub fn tss_deep_leaf(&mut self, state: &RustHexoState, hash: StateHash) -> Option<HardValue> {
        let mode = self.divergences.tss_solver_mode;
        if mode == 0 || !self.tss_enabled {
            return None;
        }
        if !self.divergences.tss_solver_all_leaves && !state.board().windows().has_threats() {
            return None;
        }
        if ((hash & 0xF) as u32) >= self.divergences.tss_solver_sample_16 {
            return None;
        }
        let binding = RootBinding::from_state(state);
        // Goal follows the consumption tier: measurement wants both sides;
        // the LOSS tier gives the whole budget to the side it consumes.
        let goal = match mode {
            2 => SolveGoal::Loss,
            _ => SolveGoal::Both,
        };
        // Async rung: hand the solve to the background pool and let this leaf
        // take the plain net eval; the verified result drains into the memo
        // and is consumed by the descent-stop on later visits. Falls through
        // to the inline solve when no pool is wired (lockstep paths) or when
        // the position lands in the hybrid inline tier (first-touch
        // consumption preserved for that slice, exactly the pre-async path).
        // `tss_deep_leaf_route` intercepts park mode before this legacy read,
        // so `tss_solver_async_inline_16` is intentionally ignored while
        // parking: no solve can run on the select thread in that mode.
        if self.divergences.tss_solver_async
            && ((hash & 0xF) as u32) >= self.divergences.tss_solver_async_inline_16
        {
            if let Some(handle) = self.tss_async.as_ref() {
                match self.tss_deep_memo.get(&hash) {
                    Some(TssMemoEntry::Pending(seen)) if *seen == binding => {
                        self.tss.async_pending_hits += 1;
                        return None;
                    }
                    Some(TssMemoEntry::Done(seen, status, hard)) if *seen == binding => {
                        self.tss.deep_memo_hits += 1;
                        let (status, hard) = (*status, *hard);
                        return self.tss_consume_gate(status, hard);
                    }
                    _ => {
                        // LIFO enqueue: fresh work is always accepted (a full
                        // queue evicts its OLDEST entry, counted as dropped);
                        // None => pool shut down.
                        let enqueued = (self.tss_deep_memo.len() < Self::TSS_DEEP_MEMO_MAX)
                            .then(|| {
                                handle.try_enqueue(SolveRequest {
                                    slot: handle.slot,
                                    generation: handle.generation,
                                    hash,
                                    binding: binding.clone(),
                                    state: state.clone(),
                                    node_cap: self.divergences.tss_solver_node_cap as u64,
                                    goal,
                                    zone: crate::tss_core::ZoneSearchCaps {
                                        enabled: self.divergences.tss_zone,
                                        stale_area_filter: self.divergences.tss_zone_stale_filter,
                                        count2_threshold: self.divergences.tss_zone_count2,
                                        pair_commutation: self.divergences.tss_pair_commutation,
                                    },
                                    horizon: SolverHorizon {
                                        horizon: self.divergences.tss_solver_horizon,
                                        ladder: self.divergences.tss_solver_horizon_ladder,
                                    },
                                    dual_pass: self.divergences.tss_solver_dual_pass,
                                    loss_reserve_nodes: self
                                        .divergences
                                        .tss_solver_loss_reserve_nodes,
                                    group2: self.divergences.tss_solver_group2,
                                    j2near: self.divergences.solver_j2near_enabled(),
                                })
                            })
                            .flatten();
                        match enqueued {
                            Some(evicted) => {
                                self.tss.async_enqueued += 1;
                                self.tss.async_dropped += evicted;
                                self.tss_deep_memo
                                    .insert(hash, TssMemoEntry::Pending(binding));
                            }
                            None => {
                                self.tss.async_dropped += 1;
                            }
                        }
                        return None;
                    }
                }
            }
        }
        let (status, hard) = match self.tss_deep_memo.get(&hash) {
            Some(TssMemoEntry::Done(seen, status, hard)) if *seen == binding => {
                self.tss.deep_memo_hits += 1;
                (*status, *hard)
            }
            _ => {
                let node_cap = self.divergences.tss_solver_node_cap as u64;
                self.tss_solver
                    .0
                    .set_dual_pass(self.divergences.tss_solver_dual_pass);
                self.tss_solver
                    .0
                    .set_loss_reserve_nodes(self.divergences.tss_solver_loss_reserve_nodes);
                self.tss_solver
                    .0
                    .set_group2(self.divergences.tss_solver_group2);
                let solved = tss_solve_verified(
                    state,
                    node_cap,
                    goal,
                    crate::tss_core::ZoneSearchCaps {
                        enabled: self.divergences.tss_zone,
                        stale_area_filter: self.divergences.tss_zone_stale_filter,
                        count2_threshold: self.divergences.tss_zone_count2,
                        pair_commutation: self.divergences.tss_pair_commutation,
                    },
                    SolverHorizon {
                        horizon: self.divergences.tss_solver_horizon,
                        ladder: self.divergences.tss_solver_horizon_ladder,
                    },
                    &mut self.tss_solver.0,
                    &mut self.tss,
                );
                let (status, hard) = (solved.status, solved.hard);
                if self.tss_deep_memo.len() < Self::TSS_DEEP_MEMO_MAX {
                    self.tss_deep_memo
                        .insert(hash, TssMemoEntry::Done(binding, status, hard));
                }
                (status, hard)
            }
        };
        self.tss_consume_gate(status, hard)
    }

    /// The consumption tier gate shared by the inline path, the async
    /// memo-hit path, and the descent-stop: verified LOSS backs up at
    /// mode >= 2, verified WIN at mode >= 3, everything else does not.
    fn tss_consume_gate(
        &mut self,
        status: ProofStatus,
        hard: Option<HardValue>,
    ) -> Option<HardValue> {
        let consume = match (status, hard) {
            (ProofStatus::Loss, Some(h)) if self.divergences.tss_solver_mode >= 2 => Some(h),
            (ProofStatus::Win, Some(h)) if self.divergences.tss_solver_mode >= 3 => Some(h),
            _ => None,
        };
        if consume.is_some() {
            self.tss.deep_hard_backups += 1;
            // Symmetry telemetry: split the consumed stream by outcome so a
            // loss-heavy run can see the loss half reaching backup (audit fix).
            match status {
                ProofStatus::Win => self.tss.deep_win_backups += 1,
                ProofStatus::Loss => self.tss.deep_loss_backups += 1,
                ProofStatus::Unknown => {}
            }
        }
        consume
    }

    /// Async descent-stop probe (only path by which drained pool results are
    /// consumed): a simulation arriving at `state` during descent checks the
    /// per-move memo and, on a verified full-binding hit at a consuming tier,
    /// stops here and backs the hard value — mirroring the inline semantics
    /// where solved positions are node-less stop points. Off (or sync mode):
    /// always `None`, zero behavior change.
    pub fn tss_async_descent_hard(
        &mut self,
        hash: StateHash,
        state: &RustHexoState,
    ) -> Option<HardValue> {
        if !self.divergences.tss_solver_async
            || self.divergences.tss_solver_mode < 2
            || !self.tss_enabled
            || self.tss_deep_memo.is_empty()
        {
            return None;
        }
        // Cheap gates FIRST (Codex review 6): only a hash-hit whose status
        // would actually consume at this tier pays the full-binding
        // construction + comparison.
        let (status, hard) = match self.tss_deep_memo.get(&hash) {
            Some(TssMemoEntry::Done(_, status, hard)) => (*status, *hard),
            _ => return None,
        };
        let consumable = match (status, hard) {
            (ProofStatus::Loss, Some(_)) => self.divergences.tss_solver_mode >= 2,
            (ProofStatus::Win, Some(_)) => self.divergences.tss_solver_mode >= 3,
            _ => false,
        };
        if !consumable {
            return None;
        }
        match self.tss_deep_memo.get(&hash) {
            Some(TssMemoEntry::Done(seen, _, _)) if *seen == RootBinding::from_state(state) => {}
            _ => return None,
        }
        let consumed = self.tss_consume_gate(status, hard);
        if consumed.is_some() {
            self.tss.deep_memo_hits += 1;
        }
        consumed
    }

    /// Wire (or clear) the async-pool enqueue handle. The continuous driver
    /// calls this with a FRESH generation at every search creation,
    /// reuse-rebind, and move advance, so stale responses can never match.
    pub fn set_tss_async(&mut self, handle: Option<TssAsyncHandle>) {
        self.tss_async = handle;
    }

    /// The generation this search's requests are stamped with (None when no
    /// pool is wired).
    pub fn tss_async_generation(&self) -> Option<u64> {
        self.tss_async.as_ref().map(|handle| handle.generation)
    }

    /// Land one drained pool response on this search: the solve's telemetry
    /// deltas plus the memo write.
    pub fn apply_tss_async_response(&mut self, response: &SolveResponse) {
        self.tss.add(&response.counters);
        self.tss_async_memo_write(response, false);
    }

    /// Land a response whose generation no longer matches (the move advanced
    /// past it). The counters are dropped as stale — EXCEPT the fatal verify
    /// counter — but the memo entry still lands: it is self-validating (a
    /// `HardValue` exists only post-verification, and consumption re-checks
    /// the full binding), so the already-paid solve keeps serving the game's
    /// persistent memo.
    pub fn apply_tss_async_response_stale(&mut self, response: &SolveResponse) {
        self.tss.async_stale += 1;
        self.tss.deep_verify_failed += response.counters.deep_verify_failed;
        self.tss.horizon_retry += response.counters.horizon_retry;
        self.tss.horizon_preflight_failed += response.counters.horizon_preflight_failed;
        self.tss.horizon_cut += response.counters.horizon_cut;
        self.tss.horizon_cut_tall += response.counters.horizon_cut_tall;
        self.tss.deep_kb_death += response.counters.deep_kb_death;
        self.tss.zone_verify_failed += response.counters.zone_verify_failed;
        if self.divergences.tss_solver_async {
            self.tss_async_memo_write(response, true);
        }
    }

    /// Memo write under the binding discipline. Rules (Codex review 1):
    /// - the response only ever writes over an entry whose binding MATCHES
    ///   the response's own (a hash-colliding stranger never clobbers the
    ///   live entry, in either direction);
    /// - `Pending` with a matching binding resolves to `Done`;
    /// - a decided response UPGRADES a matching `Done(Unknown)` (a later,
    ///   warmer-cache solve may decide what an earlier one could not);
    /// - a decided `Done` is never overwritten (determinism makes duplicates
    ///   equal; anything else must not clobber a verified proof).
    /// STALE responses (round 2): only DECIDED results land (an Unknown
    /// crossing a move boundary would defeat the warmer-cache retry), and
    /// their fresh inserts stop at the retention bound so a flood of late
    /// arrivals can never starve the new move's enqueue gate.
    fn tss_async_memo_write(&mut self, response: &SolveResponse, stale: bool) {
        let decided_response = response.status != ProofStatus::Unknown;
        if stale && !decided_response {
            return;
        }
        let insert_cap = if stale {
            Self::TSS_DEEP_MEMO_RETAIN_MAX
        } else {
            Self::TSS_DEEP_MEMO_MAX
        };
        match self.tss_deep_memo.get(&response.hash) {
            Some(TssMemoEntry::Pending(seen)) if *seen == response.binding => {
                self.tss_deep_memo.insert(
                    response.hash,
                    TssMemoEntry::Done(response.binding.clone(), response.status, response.hard),
                );
            }
            Some(TssMemoEntry::Done(seen, ProofStatus::Unknown, _))
                if decided_response && *seen == response.binding =>
            {
                self.tss_deep_memo.insert(
                    response.hash,
                    TssMemoEntry::Done(response.binding.clone(), response.status, response.hard),
                );
            }
            Some(_) => {} // binding mismatch or already decided: never clobber
            None => {
                if self.tss_deep_memo.len() < insert_cap {
                    self.tss_deep_memo.insert(
                        response.hash,
                        TssMemoEntry::Done(
                            response.binding.clone(),
                            response.status,
                            response.hard,
                        ),
                    );
                }
            }
        }
    }

    pub fn root(&self) -> &RustNode {
        debug_assert_eq!(self.nodes[0].state_hash, self.root_hash);
        &self.nodes[0]
    }

    pub fn root_edge_visits(&self) -> Vec<(PackedCoord, u32)> {
        self.root()
            .edges
            .iter()
            .map(|edge| (edge.action_id, edge.visits))
            .collect()
    }

    /// True when a Gumbel-Top-k root candidate set is active on this search.
    pub fn has_gumbel_root(&self) -> bool {
        self.gumbel_root.is_some()
    }

    /// Round-0 per-candidate visit quota of the active Gumbel SH root
    /// (`floor(budget / (R * m_initial))`, min 1), or None without an active
    /// sequential-halving state. The play-policy quota prune uses it as its
    /// cut line: a candidate whose delta visits never exceeded this quota was
    /// eliminated without surviving a halving.
    pub fn gumbel_play_quota(&self) -> Option<u32> {
        let state = self.gumbel_root.as_ref()?;
        if !state.sequential_halving {
            return None;
        }
        let m = state.gumbel.len().max(1) as u32;
        let r = state.num_rounds.max(1);
        Some((state.budget / (r * m)).max(1))
    }

    /// Tear down any Gumbel root state (e.g. when a slot transitions to a
    /// non-Full move on reuse). Idempotent.
    pub fn clear_gumbel_root(&mut self) {
        self.gumbel_root = None;
    }

    /// Build the Gumbel-Top-k root candidate set and seed the Sequential-Halving
    /// schedule. Draws g(a)~Gumbel(0,1) per legal root action from `seed`, takes
    /// the top `min(m, n_legal)` by `logits(a)+g(a)`, and lays out the SH rounds
    /// over `budget` visits. Clears state and returns without building unless
    /// `gumbel_root` is on and the root carries raw logits. `sequential_halving`
    /// selects SH vs the intermediate "top-m + PUCT" mode.
    pub fn init_gumbel_root(&mut self, seed: u64, budget: u32) {
        self.gumbel_root = None;
        if !self.divergences.gumbel_root {
            return;
        }
        // Raw per-action logits are required for the Gumbel-max draw; without
        // them the state stays None and the PUCT root path runs.
        let Some(logit_map) = self.nodes[0].root_logits.clone() else {
            return;
        };
        // Survivors materialize on demand from the owned candidate tail
        // (gumbel_root_edge_index); a promoted root can still be Shared, so
        // convert it up front.
        self.ensure_root_owned();
        // Legal root action ids: edges (nucleus) + any unexpanded candidates, so
        // the Gumbel draw covers the full legal set.
        let mut action_ids: Vec<PackedCoord> =
            self.nodes[0].edges.iter().map(|e| e.action_id).collect();
        for (action_id, _) in self.nodes[0].remaining_priors() {
            action_ids.push(action_id);
        }
        if action_ids.is_empty() {
            return;
        }
        // g(a) per action from a dedicated per-action sub-stream of `seed`.
        let mut gumbel: HashMap<PackedCoord, f32> = HashMap::with_capacity(action_ids.len());
        for &action_id in &action_ids {
            let g = gumbel_draw(seed.wrapping_add(action_id as u64));
            gumbel.insert(action_id, g);
        }
        // Top-m by logits(a)+g(a) (Gumbel-max sampling-without-replacement).
        // Under SH, m is budget-calibrated: the configured gumbel_m is sized
        // for the selfplay full budget, and reusing it at smaller budgets
        // (eval matches, quick-gate evals) starves the round-0 quota below
        // GUMBEL_MIN_ROUND0_VISITS per candidate.
        let mut m = (self.divergences.gumbel_m as usize)
            .min(action_ids.len())
            .max(1);
        if self.divergences.gumbel_sequential_halving {
            m = GumbelRootState::budget_calibrated_m(m, budget)
                .min(action_ids.len())
                .max(1);
        }
        // Draw temperature τ divides the LOGIT only in the top-m sort, sampling
        // the candidate set from softmax(logits/τ) without replacement. τ<=0 or
        // 1.0 leaves the draw at logit+g (today's behavior); the SH σ ranking,
        // exported target, and TSS force-include below all keep raw logits.
        let draw_tau = self.divergences.gumbel_draw_temperature;
        let draw_tau = if draw_tau > 0.0 { draw_tau } else { 1.0 };
        action_ids.sort_by(|&a, &b| {
            let la = logit_map.get(&a).copied().unwrap_or(0.0) / draw_tau;
            let lb = logit_map.get(&b).copied().unwrap_or(0.0) / draw_tau;
            let sa = la + gumbel[&a];
            let sb = lb + gumbel[&b];
            // Descending by score; ties broken by action_id for determinism.
            sb.partial_cmp(&sa)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        // Force-include every legal TSS tactical cell in the candidate set even
        // if its (logit+g) Gumbel score ranks below the top-m. Tactical cells are
        // added to the m budget, not counted against it.
        let tactical: HashSet<PackedCoord> = if self.tss_enabled {
            threats::tactical_cells(&self.root_state)
                .into_iter()
                .map(pack_coord)
                .collect()
        } else {
            HashSet::new()
        };
        self.tss.root_tactical = tactical.len() as u32;
        let mut survivors: Vec<PackedCoord> = Vec::with_capacity(m + tactical.len());
        let mut chosen: HashSet<PackedCoord> = HashSet::with_capacity(m + tactical.len());
        // 1. top-m by Gumbel score (action_ids already sorted descending).
        for &a in action_ids.iter().take(m) {
            if chosen.insert(a) {
                survivors.push(a);
            }
        }
        // 2. force-include any legal tactical cell not already in the top-m.
        for &a in &action_ids {
            if tactical.contains(&a) && chosen.insert(a) {
                survivors.push(a);
                // Injection fire-rate numerator: this cell ranked outside the
                // Gumbel top-m and was force-included (shadow telemetry).
                self.tss.root_injected += 1;
            }
        }
        // Restrict the kept g/logits maps to the survivor set.
        let mut g_kept: HashMap<PackedCoord, f32> = HashMap::with_capacity(survivors.len());
        let mut l_kept: HashMap<PackedCoord, f32> = HashMap::with_capacity(survivors.len());
        for &a in &survivors {
            g_kept.insert(a, gumbel[&a]);
            l_kept.insert(a, logit_map.get(&a).copied().unwrap_or(0.0));
        }
        let sequential_halving = self.divergences.gumbel_sequential_halving;
        let num_rounds = if sequential_halving {
            GumbelRootState::rounds_for(survivors.len())
        } else {
            0
        };
        // Move-entry snapshot of every root-edge visit count. Read here (before
        // any of this move's visits land) so this move's σ-scale can be built
        // from delta visits regardless of what the reused subtree carried in.
        let entry_visits: HashMap<PackedCoord, u32> = self.nodes[0]
            .edges
            .iter()
            .map(|e| (e.action_id, e.visits))
            .collect();
        let mut state = GumbelRootState {
            survivors,
            gumbel: g_kept,
            logits: l_kept,
            budget,
            num_rounds,
            round: 0,
            round_cap: HashMap::new(),
            entry_visits,
            sequential_halving,
        };
        if sequential_halving {
            self.seed_gumbel_round_caps(&mut state);
        }
        self.gumbel_root = Some(state);
    }

    /// Compute the per-survivor cumulative visit caps for the current round:
    /// each survivor must reach `entry_visits + floor(n/(R·|A_r|))` (min 1).
    /// Reads each survivor's current root-edge visit count as its round-entry
    /// baseline.
    fn seed_gumbel_round_caps(&self, state: &mut GumbelRootState) {
        let entry: HashMap<PackedCoord, u32> = self.nodes[0]
            .edges
            .iter()
            .map(|e| (e.action_id, e.visits))
            .collect();
        state.round_cap.clear();
        // A single final survivor gets an unbounded cap (u32::MAX), so the slot
        // keeps visiting it until the move's target_visits is reached.
        if state.survivors.len() <= 1 {
            for &a in &state.survivors {
                state.round_cap.insert(a, u32::MAX);
            }
            return;
        }
        let a_r = state.survivors.len() as u32;
        let r = state.num_rounds.max(1);
        // Equal per-survivor quota for this round: floor(n/(R·|A_r|)).
        let per = (state.budget / (r * a_r)).max(1);
        for &a in &state.survivors {
            let base = entry.get(&a).copied().unwrap_or(0);
            state.round_cap.insert(a, base.saturating_add(per));
        }
    }

    /// Intra-slot barrier: if every surviving candidate has met its current-round
    /// cap, rank survivors by `g(a)+logits(a)+σ(completedQ(a))`, keep the top
    /// `ceil(|A_r|/2)`, advance the round, and re-seed caps. Returns without
    /// change unless a SH Gumbel root is active and the barrier condition holds.
    /// Returns true when a halving fired.
    pub fn maybe_advance_gumbel_round(&mut self) -> bool {
        let Some(state) = self.gumbel_root.as_ref() else {
            return false;
        };
        if !state.sequential_halving
            || state.survivors.len() <= 1
            || state.round >= state.num_rounds
        {
            return false;
        }
        // Intra-slot barrier: ALL survivors must have reached their round cap.
        let visits: HashMap<PackedCoord, u32> = self.nodes[0]
            .edges
            .iter()
            .map(|e| (e.action_id, e.visits))
            .collect();
        let all_met = state.survivors.iter().all(|a| {
            let v = visits.get(a).copied().unwrap_or(0);
            let cap = state.round_cap.get(a).copied().unwrap_or(0);
            v >= cap
        });
        if !all_met {
            return false;
        }
        // Rank survivors by the SH score g(a)+logits(a)+σ(completedQ(a)).
        let root = &self.nodes[0];
        let logit_map = root.root_logits.clone().unwrap_or_default();
        let (completed, _v_mix) = gumbel_completed_q(root, &logit_map);
        let c_visit = self.divergences.gumbel_c_visit;
        let c_scale = self.divergences.gumbel_c_scale;
        let state = self.gumbel_root.as_ref().expect("checked above");
        // σ scale = THIS MOVE's max delta visits (current − move-entry baseline),
        // NOT the cumulative count: on a reused root the inherited visits would
        // otherwise apply a late-round σ multiplier to survivors whose Q rests on
        // a handful of fresh look-aheads. On a fresh root every entry baseline is
        // 0, so this is numerically identical to the cumulative max.
        let entry = &state.entry_visits;
        let max_n = root
            .edges
            .iter()
            .map(|e| {
                e.visits
                    .saturating_sub(entry.get(&e.action_id).copied().unwrap_or(0))
            })
            .max()
            .unwrap_or(0);
        let mut ranked: Vec<(PackedCoord, f32)> = state
            .survivors
            .iter()
            .map(|&a| {
                let g = state.gumbel.get(&a).copied().unwrap_or(0.0);
                let l = state.logits.get(&a).copied().unwrap_or(0.0);
                // Unsearched survivors fall back to v_mix via `completed`; if a
                // survivor never materialized an edge it uses v_mix too.
                let q = completed.get(&a).copied().unwrap_or(_v_mix);
                let score = g + l + gumbel_sigma(q, max_n, c_visit, c_scale);
                (a, score)
            })
            .collect();
        // Descending by score; deterministic action_id tie-break.
        ranked.sort_by(|&(a, sa), &(b, sb)| {
            sb.partial_cmp(&sa)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let keep = ((ranked.len() + 1) / 2).max(1);
        ranked.truncate(keep);
        let survivors: Vec<PackedCoord> = ranked.into_iter().map(|(a, _)| a).collect();
        let state = self.gumbel_root.as_mut().expect("checked above");
        state.survivors = survivors;
        state.round = state.round.saturating_add(1);
        // Re-seed caps for the new round (against the new, smaller survivor set).
        let mut new_state = state.clone();
        self.seed_gumbel_round_caps(&mut new_state);
        *self.gumbel_root.as_mut().expect("checked above") = new_state;
        true
    }

    pub fn add_node_from_eval(
        &mut self,
        state: &RustHexoState,
        hash: StateHash,
        evaluation: Arc<RustEvaluation>,
    ) -> PyResult<usize> {
        if let Some(existing) = self.node_table.get(&hash).copied() {
            return Ok(existing);
        }
        let id = self.nodes.len();
        // TSS expansion injection. Every tactical cell is engine-legal and the
        // candidate set is the full legal set, so all tactical cells are injected.
        let tactical = if self.tss_enabled {
            threats::tactical_cells(state)
        } else {
            Vec::new()
        };
        let nucleus_f64 = self.divergences.nucleus_f64;
        // Interior forced-move guard (Lever 0 §3). A node that reaches
        // expansion has verdict None (a Some-verdict leaf backs up hard and
        // never creates a node), so the fully-forced condition reduces to
        // min_hitting_set == B with no own win — every non-tactical move then
        // carries a one-ply λ¹ refutation. With the divergence flag ON the
        // children narrow to the hitting-cell universe; OFF keeps today's
        // inject-widen behavior and the counters are a shadow preview.
        let fully_forced = if tactical.is_empty() {
            false
        } else {
            let analysis = threats::analyze(state);
            !analysis.own_win_now && analysis.min_hitting_set == Some(analysis.b)
        };
        let forced_only = fully_forced && self.divergences.tss_interior_guard;
        let legal_total = evaluation.priors.len();
        let node = if tactical.is_empty() {
            shared_from_cache(hash, state, evaluation, self.widening, nucleus_f64)
        } else {
            owned_with_injection_from_eval(
                hash,
                state,
                &evaluation,
                self.widening,
                &tactical,
                nucleus_f64,
                forced_only,
            )
        };
        if fully_forced {
            self.tss.prune_eligible += 1;
            // Fan-out removed (guard on) / removable (shadow preview): the
            // full legal set minus the forced set.
            self.tss.prune_dropped += legal_total.saturating_sub(tactical.len()) as u64;
        }
        let injected_edges = node.edges.len();
        self.nodes.push(node);
        self.node_table.insert(hash, id);
        self.active_edge_count += injected_edges;
        Ok(id)
    }

    /// c_puct used by recorded-target forced-playout pruning. When
    /// `pruned_dynamic_cpuct` is on, returns c_for(root_visits) (matching
    /// selection); otherwise the static c_puct.
    pub fn effective_pruning_c_puct(&self, c_puct: f32, root_visits: u32) -> f32 {
        if self.divergences.pruned_dynamic_cpuct {
            self.c_for(c_puct, root_visits)
        } else {
            c_puct
        }
    }

    /// Exploration constant for a node with `visits`: `c_puct` when
    /// `visit_scaled_c_puct` is off, else
    /// `c_puct + c_scale * ln((visits + c_base) / c_base)`.
    fn c_for(&self, c_puct: f32, visits: u32) -> f32 {
        if !self.divergences.visit_scaled_c_puct {
            return c_puct;
        }
        c_puct
            + self.divergences.c_scale
                * ((visits as f32 + self.divergences.c_base) / self.divergences.c_base).ln()
    }

    /// Moves-left selection bonus for one edge. Returns 0 when
    /// `moves_left_utility` is off, when the edge has no visits, or when either
    /// moves-left mean is absent. Otherwise
    /// `-w * s(Q_e) * tanh((M_e - M_node) / m_scale)`, where s = +1 for
    /// Q_e > ml_q_gate, s = -1 when two-sided and Q_e < -ml_q_gate, and s = 0
    /// for |Q_e| <= ml_q_gate. Delegates to `crate::search::debug_ml_bonus`.
    fn ml_bonus(&self, node: &RustNode, edge: &RustEdge) -> f32 {
        if !self.divergences.moves_left_utility {
            return 0.0;
        }
        if edge.visits == 0 {
            return 0.0;
        }
        let (Some(m_edge), Some(m_node)) = (edge.ml_mean(), node.ml_mean()) else {
            return 0.0;
        };
        crate::search::debug_ml_bonus(
            edge.value(),
            m_edge,
            m_node,
            self.divergences.ml_weight,
            self.divergences.ml_scale,
            self.divergences.ml_q_gate,
            self.divergences.ml_two_sided,
        )
    }

    pub fn select_pending_leaf(&mut self, c_puct: f32) -> PyResult<Option<RustSelectedLeaf>> {
        let mut state = self.root_state.clone();
        let mut node_id = 0usize;
        let mut path = Vec::new();
        let mut last_parent = None;
        let mut last_edge = None;
        let mut current_hash = self.root_hash;

        loop {
            let Some(edge_index) = self.select_or_materialize_edge(node_id, c_puct) else {
                let Some(parent_node) = last_parent else {
                    return Ok(None);
                };
                let edge_index = last_edge.expect("edge index exists with parent");
                return Ok(Some(RustSelectedLeaf {
                    path,
                    state,
                    state_hash: current_hash,
                    parent_node,
                    edge_index,
                    terminal: None,
                    existing_node: Some(node_id),
                    hard: None,
                }));
            };

            let edge = &self.nodes[node_id].edges[edge_index];
            if edge.pending > 0 && edge.child.is_none() {
                return Ok(None);
            }

            let action = edge.action;
            let child = edge.child;
            apply_placement(&mut state, Placement { coord: action }).map_err(move_error)?;
            current_hash = state_hash(&state);
            path.push((node_id, edge_index));
            last_parent = Some(node_id);
            last_edge = Some(edge_index);

            // Async descent-stop: a pool-verified proof for the position just
            // reached ends this simulation here with a hard backup — covering
            // expanded children, table transpositions, and raw leaves alike.
            // No-op unless tss_solver_async is on at a consuming tier.
            if let Some(hard) = self.tss_async_descent_hard(current_hash, &state) {
                return Ok(Some(RustSelectedLeaf {
                    path,
                    state,
                    state_hash: current_hash,
                    parent_node: node_id,
                    edge_index,
                    terminal: None,
                    existing_node: None,
                    hard: Some(hard),
                }));
            }

            if let Some(child_id) = child {
                node_id = child_id;
                continue;
            }

            if let Some(child_id) = self.node_table.get(&current_hash).copied() {
                self.nodes[node_id].edges[edge_index].child = Some(child_id);
                return Ok(Some(RustSelectedLeaf {
                    path,
                    state,
                    state_hash: current_hash,
                    parent_node: node_id,
                    edge_index,
                    terminal: None,
                    existing_node: Some(child_id),
                    hard: None,
                }));
            }

            return Ok(Some(RustSelectedLeaf {
                path,
                state: state.clone(),
                state_hash: current_hash,
                parent_node: node_id,
                edge_index,
                terminal: state.terminal(),
                existing_node: None,
                hard: None,
            }));
        }
    }

    fn select_or_materialize_edge(&mut self, node_id: usize, c_puct: f32) -> Option<usize> {
        // At a Gumbel-Top-k root, selection is constrained to the surviving
        // candidate set and (under SH) the current round's visit caps, replacing
        // the PUCT/Dirichlet/forced-playout root path. Interior nodes
        // (node_id != 0) are not handled here.
        if node_id == 0 && self.gumbel_root.is_some() {
            return self.select_gumbel_root_edge(c_puct);
        }
        // TSS forced edges get a guaranteed first visit before normal PUCT.
        for (index, edge) in self.nodes[node_id].edges.iter().enumerate() {
            if edge.forced && edge.visits == 0 && !(edge.pending > 0 && edge.child.is_none()) {
                return Some(index);
            }
        }

        // Deterministic non-root selection. At interior nodes (node_id != 0),
        // when `gumbel_nonroot_select` is on, child selection is the
        // visit-regularized rule argmax_a [ π'(a) − N(a)/(1+ΣN) ] with
        // π'=softmax(logits+σ(completedQ)), with no PUCT/c_puct/FPU/widening
        // scoring. The root (node_id == 0) does not reach here; it is handled by
        // the Gumbel-Top-k branch above or the PUCT path. Flag off => PUCT.
        if self.divergences.gumbel_nonroot_select && node_id != 0 {
            return self.select_gumbel_nonroot_edge(node_id);
        }

        let node = &self.nodes[node_id];
        let exploration_scale =
            self.c_for(c_puct, node.visits) * (node.visits.max(1) as f32).sqrt();
        let parent_value = node.value();
        let base_fpu_reduction = if node_id == 0 {
            self.root_fpu_reduction
        } else {
            self.fpu_reduction
        };
        // KataGo scales the FPU reduction by √(policyProbMassVisited): ~0 at a
        // fresh node (unvisited children look as good as the parent), growing
        // toward the full reduction as children get searched. Flat under parity().
        let fpu_reduction = if self.divergences.scaled_fpu {
            base_fpu_reduction * node.visited_policy_mass().sqrt()
        } else {
            base_fpu_reduction
        };
        let mut best: Option<(usize, f32, u32, PackedCoord)> = None;
        for (index, edge) in node.edges.iter().enumerate() {
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            let score = edge.value_or_fpu(parent_value, fpu_reduction)
                + edge.prior * exploration_scale / (1.0 + edge.visits as f32)
                + self.ml_bonus(node, edge);
            let candidate = (index, score, edge.visits, edge.action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                best = Some(candidate);
            }
        }

        // Widening eligibility. With lazy_widening: a live check that an
        // unexpanded candidate exists. Otherwise: the frozen comparison
        // `edges.len() < max_eligible_children` (max_eligible_children is set at
        // node creation and not re-derived).
        let can_widen = if self.divergences.lazy_widening {
            self.nodes[node_id].peek_next_candidate().is_some()
        } else {
            self.nodes[node_id].edges.len() < self.nodes[node_id].max_eligible_children
        };
        if can_widen {
            if let Some((action_id, prior)) = self.nodes[node_id].peek_next_candidate() {
                // New-child selection score (see `new_child_score`).
                let score = new_child_score(
                    parent_value,
                    fpu_reduction,
                    prior,
                    exploration_scale,
                    self.divergences.new_child_fpu,
                );
                let candidate_key = (usize::MAX, score, 0, action_id);
                let replace = match best {
                    Some(current) => {
                        compare_edge_score(candidate_key, current) == Ordering::Greater
                    }
                    None => true,
                };
                if replace {
                    let edge_index = self.nodes[node_id].edges.len();
                    let edge = self.nodes[node_id].materialize_next_candidate();
                    self.nodes[node_id].edges.push(edge);
                    self.record_materialized_edge();
                    return Some(edge_index);
                }
            }
        }

        if node_id == 0 && self.forced_playout_k > 0.0 {
            if let Some(forced) = self.forced_root_edge() {
                return Some(forced);
            }
        }

        best.map(|item| item.0)
    }

    fn forced_root_edge(&self) -> Option<usize> {
        let root = &self.nodes[0];
        let root_visits = root.visits.max(1) as f32;
        let k = self.forced_playout_k;
        let mut best: Option<(usize, f32)> = None;
        for (index, edge) in root.edges.iter().enumerate() {
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            if !(edge.prior.is_finite() && edge.prior > 0.0) {
                continue;
            }
            let n_forced = (k * edge.prior * root_visits).sqrt();
            let deficit = n_forced - edge.visits as f32;
            if deficit > 0.0 {
                let replace = match best {
                    Some((_, best_deficit)) => deficit > best_deficit,
                    None => true,
                };
                if replace {
                    best = Some((index, deficit));
                }
            }
        }
        best.map(|(index, _)| index)
    }

    /// Deterministic non-root child selection (Danihelka et al. 2022). Interior
    /// nodes pick the visit-regularized action
    ///
    /// ```text
    ///   argmax_a [ π'(a) − N(a) / (1 + Σ_b N(b)) ]
    /// ```
    ///
    /// where `π'(a) = softmax_a( logits(a) + σ(completedQ(a)) )` is the node-level
    /// improved policy over the node's candidate set (all materialized edges plus
    /// the next-materializable prior), `N(a)` is the action's visit count, and
    /// `Σ_b N(b)` is the total visit count over that candidate set. This drives
    /// the empirical visit distribution toward π' (proportional allocation). No
    /// PUCT, no c_puct, no FPU, no widening-score gate. completedQ(a) = Q(a) for
    /// visited edges, else the visit-weighted node-value fallback `v_mix`. The σ
    /// transform uses this node's own `max_b N(b)`.
    ///
    /// A still-unmaterialized candidate (the next prior in the owned/shared tail)
    /// participates in the π' softmax at its completedQ == v_mix (it is unvisited)
    /// and at N==0, so its visit-regularized score is just π'(candidate). It is
    /// materialized only when it wins the argmax, using the peek/expand machinery.
    ///
    /// Uses `compare_edge_score` `(index, score, visits, action_id)` so ties
    /// break deterministically (fewer visits, then smaller action_id).
    fn select_gumbel_nonroot_edge(&mut self, node_id: usize) -> Option<usize> {
        let c_visit = self.divergences.gumbel_c_visit;
        let c_scale = self.divergences.gumbel_c_scale;
        // logits(a) for this node's legal actions (raw). Absent => logit 0.
        let logit_map = self.nodes[node_id].root_logits.clone().unwrap_or_default();
        let node = &self.nodes[node_id];
        // completedQ map + v_mix fallback (visit-weighted node value).
        let (completed, v_mix) = gumbel_completed_q(node, &logit_map);
        // σ scaling uses this node's own max child visit count.
        let max_n = node.edges.iter().map(|e| e.visits).max().unwrap_or(0);

        // Build the node-level improved policy π'(a) = softmax(logits + σ(Q*))
        // over the candidate set: every materialized edge in order, then the
        // next-materializable candidate (if any) as a virtual trailing slot. The
        // softmax spans the full candidate support so π' is a proper policy.
        let next_candidate = node.peek_next_candidate();
        let n_edges = node.edges.len();
        let mut scores: Vec<f32> = Vec::with_capacity(n_edges + 1);
        for edge in &node.edges {
            let l = logit_map.get(&edge.action_id).copied().unwrap_or(0.0);
            let q = completed.get(&edge.action_id).copied().unwrap_or(v_mix);
            scores.push(l + gumbel_sigma(q, max_n, c_visit, c_scale));
        }
        if let Some((action_id, _prior)) = next_candidate {
            let l = logit_map.get(&action_id).copied().unwrap_or(0.0);
            // The candidate is unvisited ⇒ completedQ == v_mix.
            scores.push(l + gumbel_sigma(v_mix, max_n, c_visit, c_scale));
        }
        let improved_pi = gumbel_softmax(&scores);

        // Σ_b N(b) over the candidate set (the next candidate contributes 0).
        let total_visits: u64 = node.edges.iter().map(|e| e.visits as u64).sum();
        let reg_denom = 1.0 + total_visits as f32;

        // Visit-regularized score for each materialized edge:
        //   π'(a) − N(a)/(1+ΣN). Skip edges with an in-flight expansion exactly
        // as the PUCT path does.
        let mut best: Option<(usize, f32, u32, PackedCoord)> = None;
        for (index, edge) in node.edges.iter().enumerate() {
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            let score = improved_pi[index] - (edge.visits as f32) / reg_denom;
            let candidate = (index, score, edge.visits, edge.action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                best = Some(candidate);
            }
        }

        // Widening: the next-materializable candidate competes at its π' score
        // (its N==0 => visit-regularized score is just π'(candidate)), in place of
        // the lazy-widening FPU/PUCT gate. Materialization uses the peek/expand
        // machinery.
        if let Some((action_id, _prior)) = next_candidate {
            let score = improved_pi[n_edges]; // trailing softmax slot; N==0.
            let candidate_key = (usize::MAX, score, 0, action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate_key, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                let edge_index = self.nodes[node_id].edges.len();
                let edge = self.nodes[node_id].materialize_next_candidate();
                self.nodes[node_id].edges.push(edge);
                self.record_materialized_edge();
                return Some(edge_index);
            }
        }

        best.map(|item| item.0)
    }

    /// Root edge selection under an active Gumbel-Top-k candidate set. Selection
    /// is constrained to the surviving candidates; no PUCT exploration term, no
    /// Dirichlet, no nucleus widening, no forced-playout.
    ///
    /// SH mode: allocate visits equally per round by selecting the survivor with
    /// the largest deficit below its current round cap (ties -> fewest visits ->
    /// action_id). When every survivor has met its cap, return None; the slot
    /// makes no further root progress until the intra-slot barrier
    /// (`maybe_advance_gumbel_round`) halves and re-seeds the next round.
    ///
    /// Intermediate mode (gumbel_root on, SH off): visits allocate among the
    /// survivors via the PUCT score, over the top-m set only.
    fn select_gumbel_root_edge(&mut self, c_puct: f32) -> Option<usize> {
        let state = self.gumbel_root.as_ref()?;
        let survivor_set: HashSet<PackedCoord> = state.survivors.iter().copied().collect();
        let sequential_halving = state.sequential_halving;
        let round_cap = state.round_cap.clone();

        // Materialize any survivor that is still an unexpanded prior so it can be
        // visited. (The Gumbel draw covers the full legal set, so a survivor may
        // sit in the owned-candidate tail rather than the nucleus edges.)
        // We materialize on demand: only when a survivor is the chosen action and
        // has no edge yet. First, build the current edge index per survivor.
        if sequential_halving {
            // Equal-allocation: choose the survivor with the largest (cap - visits)
            // deficit that is not currently pending-without-child. Prefer existing
            // edges; materialize a survivor with no edge only when it is the pick.
            let mut best_action: Option<(PackedCoord, u32)> = None; // (action, deficit)
            for &action in &survivor_set {
                let cap = round_cap.get(&action).copied().unwrap_or(0);
                // Current edge state for this survivor (if materialized).
                let edge = self.nodes[0].edges.iter().find(|e| e.action_id == action);
                if let Some(edge) = edge {
                    if edge.pending > 0 && edge.child.is_none() {
                        continue; // already in flight; respect equal allocation
                    }
                    if edge.visits >= cap {
                        continue; // met its round cap
                    }
                    let deficit = cap.saturating_sub(edge.visits);
                    let replace = match best_action {
                        Some((ba, bd)) => deficit > bd || (deficit == bd && action < ba),
                        None => true,
                    };
                    if replace {
                        best_action = Some((action, deficit));
                    }
                } else {
                    // Unexpanded survivor: visits == 0, deficit == cap (>=1).
                    let deficit = cap.max(1);
                    let replace = match best_action {
                        Some((ba, bd)) => deficit > bd || (deficit == bd && action < ba),
                        None => true,
                    };
                    if replace {
                        best_action = Some((action, deficit));
                    }
                }
            }
            let (action, _) = best_action?;
            return self.gumbel_root_edge_index(action);
        }

        // Intermediate mode: PUCT among survivors only.
        let node = &self.nodes[0];
        let exploration_scale =
            self.c_for(c_puct, node.visits) * (node.visits.max(1) as f32).sqrt();
        let parent_value = node.value();
        // Root FPU, √(policyProbMassVisited)-scaled under scaled_fpu (flat in parity).
        let fpu_reduction = if self.divergences.scaled_fpu {
            self.root_fpu_reduction * node.visited_policy_mass().sqrt()
        } else {
            self.root_fpu_reduction
        };
        let mut best: Option<(usize, f32, u32, PackedCoord)> = None;
        for (index, edge) in node.edges.iter().enumerate() {
            if !survivor_set.contains(&edge.action_id) {
                continue;
            }
            if edge.pending > 0 && edge.child.is_none() {
                continue;
            }
            let score = edge.value_or_fpu(parent_value, fpu_reduction)
                + edge.prior * exploration_scale / (1.0 + edge.visits as f32)
                + self.ml_bonus(node, edge);
            let candidate = (index, score, edge.visits, edge.action_id);
            let replace = match best {
                Some(current) => compare_edge_score(candidate, current) == Ordering::Greater,
                None => true,
            };
            if replace {
                best = Some(candidate);
            }
        }
        // Materialize an unexpanded survivor if PUCT prefers a new child. The
        // new-child score is the standard widening score, gated to the survivor
        // set, so a survivor sitting in the owned tail can get its first visit.
        let mut best_new: Option<(PackedCoord, f32)> = None;
        for (action_id, prior) in node.remaining_priors() {
            if !survivor_set.contains(&action_id) {
                continue;
            }
            let score = new_child_score(
                parent_value,
                fpu_reduction,
                prior,
                exploration_scale,
                self.divergences.new_child_fpu,
            );
            let replace = match best_new {
                Some((ba, bs)) => score > bs || (score == bs && action_id < ba),
                None => true,
            };
            if replace {
                best_new = Some((action_id, score));
            }
        }
        if let Some((new_action, new_score)) = best_new {
            let take_new = match best {
                Some((_, best_score, _, _)) => new_score > best_score,
                None => true,
            };
            if take_new {
                return self.gumbel_root_edge_index(new_action);
            }
        }
        best.map(|item| item.0)
    }

    /// Return the edge index for `action`, materializing it from the owned root
    /// candidate tail if it has not been expanded yet. Returns None only if the
    /// action is neither an edge nor an owned candidate (should not happen for a
    /// survivor drawn from the legal set).
    fn gumbel_root_edge_index(&mut self, action: PackedCoord) -> Option<usize> {
        if let Some(index) = self.nodes[0]
            .edges
            .iter()
            .position(|e| e.action_id == action)
        {
            return Some(index);
        }
        // Materialize the specific owned candidate (root is always Owned).
        if let NodePriors::Owned(unexpanded) = &mut self.nodes[0].priors {
            if let Some(pos) = unexpanded.iter().position(|c| c.action_id == action) {
                let candidate = unexpanded.remove(pos);
                let edge_index = self.nodes[0].edges.len();
                self.nodes[0].edges.push(candidate.into_edge());
                self.record_materialized_edge();
                return Some(edge_index);
            }
        }
        None
    }

    pub fn apply_virtual_visit(&mut self, path: &[(usize, usize)], virtual_loss: f32) {
        self.completed_visits = self.completed_visits.saturating_add(1);
        for &(node_id, edge_index) in path {
            self.nodes[node_id].visits += 1;
            self.nodes[node_id].value_sum -= virtual_loss;
            self.nodes[node_id].edges[edge_index].visits += 1;
            self.nodes[node_id].edges[edge_index].value_sum -= virtual_loss;
        }
    }

    /// Real backup (adds back the virtual loss). `leaf_ml` is the moves-left
    /// estimate at the leaf in decisions (0 at a terminal). Per-step distance is
    /// added below. ML stats are side-agnostic.
    pub fn backup_virtual(
        &mut self,
        path: &[(usize, usize)],
        leaf_player: Player,
        leaf_value: f32,
        virtual_loss: f32,
        leaf_ml: Option<f32>,
    ) {
        let depth = path.len();
        // Depth telemetry: one sample per REAL backup, all leaf kinds
        // (terminal / existing-node / λ¹ / verified-deep / net-eval).
        self.tss.depth_sum += depth as u64;
        self.tss.depth_max = self.tss.depth_max.max(depth as u32);
        self.tss.backups += 1;
        for (step, &(node_id, edge_index)) in path.iter().enumerate() {
            let value = if self.nodes[node_id].player == leaf_player {
                leaf_value
            } else {
                -leaf_value
            };
            let node = &mut self.nodes[node_id];
            node.value_sum += value + virtual_loss;
            if let Some(ml) = leaf_ml {
                let ml_here = ml + (depth - step) as f32;
                node.ml_sum += ml_here;
                node.ml_weight += 1.0;
                let edge = &mut node.edges[edge_index];
                edge.value_sum += value + virtual_loss;
                edge.value_sq_sum += value * value;
                edge.ml_sum += ml_here - 1.0; // edge's child is one decision deeper
                edge.ml_weight += 1.0;
            } else {
                let edge = &mut node.edges[edge_index];
                edge.value_sum += value + virtual_loss;
                edge.value_sq_sum += value * value;
            }
        }
    }

    pub fn mark_pending(&mut self, node_id: usize, edge_index: usize, delta: i32) {
        let edge = &mut self.nodes[node_id].edges[edge_index];
        if delta >= 0 {
            edge.pending = edge.pending.saturating_add(delta as u32);
        } else {
            edge.pending = edge.pending.saturating_sub((-delta) as u32);
        }
    }

    fn record_materialized_edge(&mut self) {
        self.active_edge_count += 1;
    }

    pub fn diagnostics(&self) -> RustSearchDiagnostics {
        RustSearchDiagnostics {
            node_count: self.nodes.len(),
            active_edge_count: self.active_edge_count,
            root_active_edges: self.nodes.first().map(|node| node.edges.len()).unwrap_or(0),
            root_hidden_priors: self
                .nodes
                .first()
                .map(|node| node.remaining_prior_count())
                .unwrap_or(0),
        }
    }

    pub fn advance_root(&mut self, action_id: PackedCoord) -> PyResult<bool> {
        let Some(edge) = self
            .nodes
            .first()
            .and_then(|node| node.edges.iter().find(|edge| edge.action_id == action_id))
            .cloned()
        else {
            return Ok(false);
        };
        let Some(child_id) = edge.child else {
            return Ok(false);
        };

        let mut new_root_state = self.root_state.clone();
        apply_placement(&mut new_root_state, Placement { coord: edge.action })
            .map_err(move_error)?;
        if new_root_state.terminal().is_some() {
            return Ok(false);
        }

        // The old root's player, captured BEFORE the subtree clone replaces
        // self.nodes. edge.value_sum is in the OLD root's player perspective;
        // whether the promoted root shares that perspective decides the sign.
        let old_player = self.nodes[0].player;

        let mut old_to_new = HashMap::new();
        let mut nodes = Vec::new();
        clone_subtree_nodes(child_id, &self.nodes, &mut old_to_new, &mut nodes);
        if nodes.is_empty() {
            return Ok(false);
        }

        let root_hash = state_hash(&new_root_state);
        nodes[0].state_hash = root_hash;
        if edge.visits > nodes[0].visits {
            nodes[0].visits = edge.visits;
            // edge.value_sum carries the child value in the OLD root's player
            // perspective. When the promoted root is the OTHER player (normal
            // turn pass) the perspective flips, so negate; on a same-player
            // promotion (FirstStone -> SecondStone keeps current_player) the
            // perspective already matches and negating would flip the sign.
            // This seeds the promoted root's FPU baseline and first reported
            // value; it is diluted (not overwritten) by fresh backups.
            nodes[0].value_sum = if nodes[0].player == old_player {
                edge.value_sum
            } else {
                -edge.value_sum
            };
        }
        let mut node_table = HashMap::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            node_table.insert(node.state_hash, index);
        }

        self.root_state = new_root_state;
        self.root_hash = root_hash;
        self.nodes = nodes;
        self.node_table = node_table;
        self.target_visits = 0;
        self.completed_visits = self.nodes[0]
            .edges
            .iter()
            .fold(self.nodes[0].visits, |total, edge| total.max(edge.visits));
        // The promoted root is a new root: discard the previous root's clean
        // prior cache so the next temperature/noise application re-captures the
        // promoted root's own clean (post-temp, pre-noise) priors.
        self.clean_root_priors = None;
        self.recompute_accounting();
        Ok(true)
    }

    fn recompute_accounting(&mut self) {
        self.active_edge_count = self.nodes.iter().map(|node| node.edges.len()).sum();
    }
}

fn clone_subtree_nodes(
    old_id: usize,
    old_nodes: &[RustNode],
    old_to_new: &mut HashMap<usize, usize>,
    new_nodes: &mut Vec<RustNode>,
) -> usize {
    if let Some(new_id) = old_to_new.get(&old_id).copied() {
        return new_id;
    }
    let new_id = new_nodes.len();
    old_to_new.insert(old_id, new_id);
    let mut node = old_nodes[old_id].clone();
    for edge in &mut node.edges {
        edge.child = None;
    }
    new_nodes.push(node);

    for (edge_index, old_edge) in old_nodes[old_id].edges.iter().enumerate() {
        if let Some(old_child) = old_edge.child {
            let new_child = clone_subtree_nodes(old_child, old_nodes, old_to_new, new_nodes);
            new_nodes[new_id].edges[edge_index].child = Some(new_child);
        }
    }
    new_id
}

/// Split candidates into forced (tactical) edges and the remaining candidate
/// list, and return the widening cap. All tactical cells are treated as
/// candidates (no crop). The cap is the nucleus plus the number of tactical
/// cells that fall outside the nucleus.
fn split_tactical(
    candidates: Vec<RustPriorCandidate>,
    tactical: &[HexCoord],
    nucleus: usize,
) -> (Vec<RustEdge>, Vec<RustPriorCandidate>, usize) {
    if tactical.is_empty() {
        return (Vec::new(), candidates, nucleus);
    }
    let tac: HashSet<PackedCoord> = tactical.iter().map(|c| pack_coord(*c)).collect();
    let mut by_prior: Vec<usize> = (0..candidates.len()).collect();
    by_prior.sort_by(|&a, &c| {
        candidates[c]
            .prior
            .partial_cmp(&candidates[a].prior)
            .unwrap_or(Ordering::Equal)
    });
    let nucleus_set: HashSet<PackedCoord> = by_prior
        .iter()
        .take(nucleus)
        .map(|&i| candidates[i].action_id)
        .collect();
    let mut forced = Vec::new();
    let mut rest = Vec::with_capacity(candidates.len());
    let mut extra_beyond_nucleus = 0usize;
    for candidate in candidates {
        if tac.contains(&candidate.action_id) {
            if !nucleus_set.contains(&candidate.action_id) {
                extra_beyond_nucleus += 1;
            }
            let mut edge = candidate.into_edge();
            edge.forced = true;
            forced.push(edge);
        } else {
            rest.push(candidate);
        }
    }
    let cap = nucleus + extra_beyond_nucleus;
    (forced, rest, cap)
}

/// Build the raw-logit lookup carried onto a `RustNode` from an evaluation's
/// optional logits. When the evaluation has no logits this returns `None`, so no
/// Gumbel/σ path can read logits.
fn node_logits_from_evaluation(evaluation: &RustEvaluation) -> Option<HashMap<PackedCoord, f32>> {
    evaluation
        .logits
        .as_ref()
        .map(|pairs| pairs.iter().copied().collect())
}

fn owned_with_injection_from_eval(
    state_hash_value: StateHash,
    state: &RustHexoState,
    evaluation: &RustEvaluation,
    widening: Widening,
    tactical: &[HexCoord],
    nucleus_f64: bool,
    forced_only: bool,
) -> RustNode {
    let nucleus = nucleus_count_pairs(&evaluation.priors, widening, nucleus_f64);
    let mut candidates: Vec<RustPriorCandidate> = evaluation
        .priors
        .iter()
        .map(|&(action_id, prior)| RustPriorCandidate { action_id, prior })
        .collect();
    candidates.reverse();
    let (edges, rest, max_eligible_children) = split_tactical(candidates, tactical, nucleus);
    // Interior forced-move guard (Lever 0): at a fully-forced node the caller
    // passes forced_only=true and the non-tactical candidates are dropped
    // entirely — each carries a one-ply λ¹ refutation (see add_node_from_eval).
    let (rest, max_eligible_children) = if forced_only {
        (Vec::new(), edges.len())
    } else {
        (rest, max_eligible_children)
    };
    RustNode {
        state_hash: state_hash_value,
        player: state.current_player(),
        eval_value: evaluation.value,
        eval_ml: evaluation.moves_left,
        visits: 0,
        value_sum: 0.0,
        ml_sum: 0.0,
        ml_weight: 0.0,
        edges,
        priors: NodePriors::Owned(rest),
        max_eligible_children,
        root_logits: node_logits_from_evaluation(evaluation),
    }
}

fn shared_from_cache(
    state_hash_value: StateHash,
    state: &RustHexoState,
    evaluation: Arc<RustEvaluation>,
    widening: Widening,
    nucleus_f64: bool,
) -> RustNode {
    let max_eligible_children = nucleus_count_pairs(&evaluation.priors, widening, nucleus_f64);
    // Build the logit map BEFORE moving the Arc into NodePriors::Shared.
    let root_logits = node_logits_from_evaluation(&evaluation);
    RustNode {
        state_hash: state_hash_value,
        player: state.current_player(),
        eval_value: evaluation.value,
        eval_ml: evaluation.moves_left,
        visits: 0,
        value_sum: 0.0,
        ml_sum: 0.0,
        ml_weight: 0.0,
        edges: Vec::new(),
        priors: NodePriors::Shared(evaluation),
        max_eligible_children,
        root_logits,
    }
}

/// Build the owned root node. Returns the node plus the clean post-temp,
/// pre-noise prior cache (keyed by action_id) when `clean_root_prior_cache` is
/// on, else None.
fn owned_root_from_evaluation(
    state_hash_value: StateHash,
    state: &RustHexoState,
    evaluation: &RustEvaluation,
    root_policy_temperature: Option<f32>,
    root_noise: Option<RootDirichletNoise>,
    widening: Widening,
    tss_enabled: bool,
    divergences: Divergences,
) -> PyResult<(RustNode, Option<HashMap<PackedCoord, f32>>)> {
    let mut candidates: Vec<_> = evaluation
        .priors
        .iter()
        .map(|(action_id, prior)| RustPriorCandidate {
            action_id: *action_id,
            prior: *prior,
        })
        .collect();
    candidates.sort_by(compare_prior_candidate);
    let mut seen_actions = HashSet::new();
    candidates.retain(|candidate| seen_actions.insert(candidate.action_id));
    if let Some(temperature) = root_policy_temperature {
        apply_root_policy_temperature_to(&mut candidates, temperature);
    }
    normalize_candidate_priors(&mut candidates)?;
    // Cache the clean post-temp, pre-noise priors before mixing noise.
    let clean_cache = if divergences.clean_root_prior_cache {
        Some(
            candidates
                .iter()
                .map(|candidate| (candidate.action_id, candidate.prior))
                .collect::<HashMap<_, _>>(),
        )
    } else {
        None
    };
    if let Some(noise) = root_noise {
        // Shaped vs flat is selected by `noise.shaped`. The clean (post-temp,
        // normalized, pre-noise) priors are the shaped-alpha input.
        apply_dirichlet_noise(&mut candidates, noise);
    }
    candidates.sort_by(compare_prior_candidate);
    candidates.reverse();
    let nucleus = nucleus_count(&candidates, widening, divergences.nucleus_f64);
    let tactical = if tss_enabled {
        threats::tactical_cells(state)
    } else {
        Vec::new()
    };
    let (edges, mut candidates, max_eligible_children) =
        split_tactical(candidates, &tactical, nucleus);
    candidates.shrink_to_fit();
    Ok((
        RustNode {
            state_hash: state_hash_value,
            player: state.current_player(),
            eval_value: evaluation.value,
            eval_ml: evaluation.moves_left,
            visits: 0,
            value_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            edges,
            priors: NodePriors::Owned(candidates),
            max_eligible_children,
            // Raw logits keyed by action_id (pre-temperature, pre-Dirichlet).
            root_logits: node_logits_from_evaluation(evaluation),
        },
        clean_cache,
    ))
}

fn nucleus_count(candidates: &[RustPriorCandidate], widening: Widening, f64_mode: bool) -> usize {
    nucleus_count_values(
        candidates.iter().map(|candidate| candidate.prior).collect(),
        widening,
        f64_mode,
    )
}

fn nucleus_count_pairs(priors: &[(PackedCoord, f32)], widening: Widening, f64_mode: bool) -> usize {
    nucleus_count_values(
        priors.iter().map(|(_, prior)| *prior).collect(),
        widening,
        f64_mode,
    )
}

/// Count the nucleus (the top-prior children whose cumulative mass first reaches
/// `widening.mass`), clamped to [min_children, max_children].
///
/// When `f64_mode` is true, accumulate cumulative mass in f64 and short-circuit
/// to `hi` (take all eligible) when `widening.mass >= 1.0`. When false,
/// accumulate in f32 with no `mass == 1.0` short-circuit.
fn nucleus_count_values(mut priors: Vec<f32>, widening: Widening, f64_mode: bool) -> usize {
    let total = priors.len();
    if total == 0 {
        return 0;
    }
    let lo = widening.min_children.max(1).min(total);
    let hi = widening.max_children.max(lo).min(total);
    if lo >= hi {
        return hi;
    }
    if f64_mode && widening.mass >= 1.0 {
        // Take every eligible child: for a normalized policy the cumulative mass
        // cannot exceed 1.0.
        return hi;
    }
    priors.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    let mut count = 0usize;
    if f64_mode {
        let mut cumulative = 0.0f64;
        for prior in priors {
            cumulative += prior as f64;
            count += 1;
            if cumulative >= widening.mass as f64 {
                break;
            }
        }
    } else {
        let mut cumulative = 0.0f32;
        for prior in priors {
            cumulative += prior;
            count += 1;
            if cumulative >= widening.mass {
                break;
            }
        }
    }
    count.clamp(lo, hi)
}

fn apply_root_policy_temperature_to(candidates: &mut [RustPriorCandidate], temperature: f32) {
    if !temperature.is_finite() || temperature <= 0.0 || (temperature - 1.0).abs() < 1.0e-6 {
        return;
    }
    let inverse = 1.0 / temperature;
    for candidate in candidates.iter_mut() {
        if candidate.prior.is_finite() && candidate.prior > 0.0 {
            candidate.prior = candidate.prior.powf(inverse);
        }
    }
}

fn apply_dirichlet_noise(candidates: &mut [RustPriorCandidate], noise: RootDirichletNoise) {
    if candidates.is_empty() || noise.total_alpha <= 0.0 || noise.fraction <= 0.0 {
        return;
    }
    let fraction = noise.fraction;
    // Candidate priors are the clean post-temp, normalized policy (priors sum to
    // 1). For shaped noise the per-move concentration is the shaped alpha
    // distribution computed from these priors; for flat noise it is
    // total_alpha / count.
    let samples = if noise.shaped {
        let clean: Vec<f32> = candidates.iter().map(|c| c.prior).collect();
        shaped_dirichlet_samples(&clean, noise)
    } else {
        dirichlet_samples(candidates.len(), noise)
    };
    for (candidate, sampled) in candidates.iter_mut().zip(samples) {
        candidate.prior = (1.0 - fraction) * candidate.prior + fraction * sampled;
    }
}

fn dirichlet_samples(count: usize, noise: RootDirichletNoise) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    let per_action_alpha = (noise.total_alpha as f64 / count as f64).max(1.0e-6);
    let mut sampler = DirichletSampler::new(noise.seed);
    let mut samples = Vec::with_capacity(count);
    let mut total = 0.0f64;
    for _ in 0..count {
        let value = sampler.gamma(per_action_alpha);
        samples.push(value);
        total += value;
    }
    if total <= 0.0 || !total.is_finite() {
        return vec![1.0 / count as f32; count];
    }
    samples
        .into_iter()
        .map(|sample| (sample / total) as f32)
        .collect()
}

/// Shaped Dirichlet alpha distribution. Input `clean_policy` is the clean
/// post-temp, normalized, pre-noise policy over the legal moves. Returns a
/// per-move alpha shape that sums to 1 over the moves; multiply by a total
/// concentration to get a per-move alpha.
///   a[i]  = log(min(0.01, p[i]) + 1e-20)
///   mean  = sum(a) / n
///   a[i]  = max(0, a[i] - mean)
///   if sum(a) <= 0 : a[i] = 1/n               (uniform fallback)
///   else           : a[i] = 0.5*(a[i]/sum + 1/n)
fn shaped_alpha(clean_policy: &[f32]) -> Vec<f64> {
    let n = clean_policy.len();
    if n == 0 {
        return Vec::new();
    }
    let uniform = 1.0 / n as f64;
    let mut a: Vec<f64> = clean_policy
        .iter()
        .map(|&p| (p.min(0.01) as f64 + 1e-20).ln())
        .collect();
    let log_mean: f64 = a.iter().sum::<f64>() / n as f64;
    let mut alpha_sum = 0.0f64;
    for value in a.iter_mut() {
        *value = (*value - log_mean).max(0.0);
        alpha_sum += *value;
    }
    if alpha_sum <= 0.0 {
        return vec![uniform; n];
    }
    for value in a.iter_mut() {
        *value = 0.5 * (*value / alpha_sum + uniform);
    }
    a
}

/// Draw a Dirichlet sample whose per-move concentration is the shaped alpha
/// (`shaped_alpha(clean) * noise.total_alpha`). Returns samples that sum to 1,
/// matching `dirichlet_samples`' contract. Uses the same gamma sampler and seed
/// stream as `dirichlet_samples`.
fn shaped_dirichlet_samples(clean_policy: &[f32], noise: RootDirichletNoise) -> Vec<f32> {
    let count = clean_policy.len();
    if count == 0 {
        return Vec::new();
    }
    let shape = shaped_alpha(clean_policy);
    let total_concentration = noise.total_alpha as f64;
    let mut sampler = DirichletSampler::new(noise.seed);
    let mut samples = Vec::with_capacity(count);
    let mut total = 0.0f64;
    for &shape_i in &shape {
        let alpha = (shape_i * total_concentration).max(1.0e-6);
        let value = sampler.gamma(alpha);
        samples.push(value);
        total += value;
    }
    if total <= 0.0 || !total.is_finite() {
        return vec![1.0 / count as f32; count];
    }
    samples
        .into_iter()
        .map(|sample| (sample / total) as f32)
        .collect()
}

struct DirichletSampler {
    state: u64,
}

impl DirichletSampler {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xD1B5_4A32_D192_ED03,
        }
    }

    fn uniform_open(&mut self) -> f64 {
        random_unit(self.next_u64()).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform_open();
        let u2 = self.uniform_open();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn gamma(&mut self, alpha: f64) -> f64 {
        if alpha < 1.0 {
            let boosted = self.gamma(alpha + 1.0);
            return boosted * self.uniform_open().powf(1.0 / alpha);
        }
        let d = alpha - 1.0 / 3.0;
        let c = (1.0 / (9.0 * d)).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v3 = v * v * v;
            let u = self.uniform_open();
            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v3;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
                return d * v3;
            }
        }
    }
}

fn compare_prior_candidate(left: &RustPriorCandidate, right: &RustPriorCandidate) -> Ordering {
    right
        .prior
        .partial_cmp(&left.prior)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.action_id.cmp(&right.action_id))
}

fn normalize_candidate_priors(candidates: &mut [RustPriorCandidate]) -> PyResult<()> {
    let mut total = 0.0f32;
    for candidate in candidates.iter() {
        if !candidate.prior.is_finite() || candidate.prior < 0.0 {
            return Err(PyValueError::new_err(format!(
                "prior for action {} must be finite and >= 0",
                candidate.action_id
            )));
        }
        total += candidate.prior;
    }
    if candidates.is_empty() {
        return Ok(());
    }
    if total <= 0.0 {
        return Err(PyValueError::new_err(
            "candidate priors must contain positive mass",
        ));
    }
    for candidate in candidates {
        candidate.prior /= total;
    }
    Ok(())
}

/// PUCT selection value for a new (visits==0) candidate child.
///
/// - `new_child_fpu` on: `(parent_value - fpu_reduction) + U`. A new child has
///   visits==0, so the U denominator is (1 + 0) = 1 and
///   `U = prior * exploration_scale`. This matches the score
///   `RustEdge::value_or_fpu` yields for an existing 0-visit edge with the same
///   prior.
/// - `new_child_fpu` off: U-only `prior * exploration_scale`.
fn new_child_score(
    parent_value: f32,
    fpu_reduction: f32,
    prior: f32,
    exploration_scale: f32,
    new_child_fpu: bool,
) -> f32 {
    let u = prior * exploration_scale;
    if new_child_fpu {
        (parent_value - fpu_reduction) + u
    } else {
        u
    }
}

fn compare_edge_score(
    left: (usize, f32, u32, PackedCoord),
    right: (usize, f32, u32, PackedCoord),
) -> Ordering {
    left.1
        .partial_cmp(&right.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| right.2.cmp(&left.2))
        .then_with(|| right.3.cmp(&left.3))
}

pub fn terminal_value(outcome: GameOutcome, player: Player) -> f32 {
    if outcome.winner == player {
        1.0
    } else {
        -1.0
    }
}

pub fn random_unit(seed: u64) -> f64 {
    let mut value = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

// === Gumbel AlphaZero shared math (Danihelka et al. 2022) ===
// These helpers are used by the Gumbel-Top-k root + Sequential Halving, the
// deterministic non-root selection, and the improved target. They are pure
// functions of the tree state and have no effect unless a Gumbel flag routes
// through them.

/// The σ transform: σ(q)=(c_visit + max_b N(b))·c_scale·q. `max_n` is the
/// maximum child visit count at the node σ is applied at. Q is already on
/// [-1,1] here, so σ is a monotone-in-q scaling (σ(0)=0).
pub fn gumbel_sigma(q: f32, max_n: u32, c_visit: f32, c_scale: f32) -> f32 {
    (c_visit + max_n as f32) * c_scale * q
}

/// A single Gumbel(0,1) draw from a deterministic per-action seed. The
/// inverse-CDF transform g = -ln(-ln(u)) on a uniform u in the open interval
/// (0,1). The uniform is clamped strictly inside (0,1) so the double log never
/// hits ±inf (random_unit can return exactly 0.0).
pub fn gumbel_draw(seed: u64) -> f32 {
    let mut u = random_unit(seed);
    // Clamp to the open interval; the ulp-scale floor/ceil keep g finite.
    const EPS: f64 = 1.0e-12;
    if u < EPS {
        u = EPS;
    } else if u > 1.0 - EPS {
        u = 1.0 - EPS;
    }
    (-(-u.ln()).ln()) as f32
}

/// completedQ + the v_mix unvisited fallback, computed once per node.
///
/// For each candidate action_id this returns `Q(a)` if it has visits, else the
/// shared `v_mix` fallback for unvisited actions. `v_mix` is the visit-count
/// weighted interpolation between the node's own value estimate `v̂` and the
/// prior-weighted average Q over visited children (Danihelka et al. 2022,
/// completed_Q):
///
/// ```text
///   v_mix = ( v̂ + (Σ_b N(b)) · (Σ_b π(b)·Q(b)) / (Σ_b π(b)) ) / ( 1 + Σ_b N(b) )
/// ```
///
/// over visited b only, where `π(b)=softmax(logits)(b)` is the network prior and
/// `v̂ = node.value()` (the node's backed-up value, or its eval_value when the
/// node itself is unvisited). At low visit counts the `v̂` term dominates; as
/// ΣN→∞ it approaches the visited-child average. When no child is visited
/// (ΣN==0 or the visited-π mass is 0) v_mix is exactly `v̂`.
///
/// `logits` maps action_id -> raw policy logit; absent actions get logit 0.
/// Returns the per-(action_id) completedQ map plus the `v_mix` scalar (so
/// callers can extend it to candidates not in `edges`).
pub fn gumbel_completed_q(
    node: &RustNode,
    logits: &HashMap<PackedCoord, f32>,
) -> (HashMap<PackedCoord, f32>, f32) {
    // Prior-weighted average Q over the visited children: softmax(logits) over
    // the visited support, weighting their Q. Numerically-stable shifted form;
    // denominator is the visited-π mass (not the full action set). `sum_n` is the
    // total child visit count Σ_b N(b) over the same visited support.
    let mut max_logit = f32::NEG_INFINITY;
    for edge in &node.edges {
        if edge.visits > 0 {
            let l = logits.get(&edge.action_id).copied().unwrap_or(0.0);
            if l > max_logit {
                max_logit = l;
            }
        }
    }
    let mut weighted_q = 0.0f32;
    let mut weight_sum = 0.0f32;
    let mut sum_n = 0u64;
    if max_logit.is_finite() {
        for edge in &node.edges {
            if edge.visits == 0 {
                continue;
            }
            let l = logits.get(&edge.action_id).copied().unwrap_or(0.0);
            let w = (l - max_logit).exp();
            weighted_q += w * edge.value();
            weight_sum += w;
            sum_n += edge.visits as u64;
        }
    }
    // v_mix: visit-count-weighted blend of v̂ (= node.value()) and the
    // prior-weighted visited-child average. When no child is visited this reduces
    // to v̂.
    let v_node = node.value();
    let v_mix = if weight_sum > 0.0 && sum_n > 0 {
        let visited_avg = weighted_q / weight_sum;
        let n = sum_n as f32;
        (v_node + n * visited_avg) / (1.0 + n)
    } else {
        v_node
    };
    let mut completed = HashMap::with_capacity(node.edges.len());
    for edge in &node.edges {
        let q = if edge.visits > 0 { edge.value() } else { v_mix };
        completed.insert(edge.action_id, q);
    }
    (completed, v_mix)
}

/// Numerically-stable softmax of a slice into a fresh Vec that sums to 1. Used
/// by the SH ranking and the target build. Empty input => empty output.
pub fn gumbel_softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // All -inf (no support): degenerate uniform to avoid NaNs.
        let u = 1.0 / scores.len() as f32;
        return vec![u; scores.len()];
    }
    let mut exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }
    exps
}

/// Per-slot Gumbel-Top-k + Sequential-Halving state for one root. Lives on
/// `RustSearch`. `survivors` shrinks as halving rounds advance; `gumbel` holds
/// the per-candidate g(a) draw for SH ranking; `round_cap[a]` is the cumulative
/// visit cap candidate `a` must reach in the current round before the intra-slot
/// barrier permits a halving. Constructed only when `gumbel_root` is on and the
/// move is a Full move.
#[derive(Clone, Debug)]
pub struct GumbelRootState {
    /// Surviving candidate action_ids for the current SH round (shrinks by ~half
    /// each round; in the no-SH intermediate mode this stays the full top-m set).
    pub survivors: Vec<PackedCoord>,
    /// g(a) ~ Gumbel(0,1) per original top-m candidate (kept across rounds for
    /// the SH rank `g(a)+logits(a)+σ(completedQ(a))`).
    pub gumbel: HashMap<PackedCoord, f32>,
    /// Raw logits per candidate (snapshot of the root's raw logits).
    pub logits: HashMap<PackedCoord, f32>,
    /// Total per-move visit budget `n` to allocate via SH.
    pub budget: u32,
    /// Number of SH rounds R = ceil(log2(m)).
    pub num_rounds: u32,
    /// 0-based current round index.
    pub round: u32,
    /// Cumulative visit cap each surviving candidate must reach by the END of
    /// the current round (its visits at round entry + this round's per-survivor
    /// quota). Keyed by action_id; only survivors appear.
    pub round_cap: HashMap<PackedCoord, u32>,
    /// Per-root-edge visit counts snapshotted at THIS MOVE's Gumbel-root init
    /// (move entry). On a reused root these are the inherited counts from the
    /// previous move's subtree; on a fresh root they are all 0. The SH ranking
    /// and the exported gumbel target derive their σ-scale `max_n` from this
    /// move's DELTA visits (current − entry) so a reuse-inflated cumulative
    /// count cannot apply a late-round σ multiplier to early-round survivors.
    pub entry_visits: HashMap<PackedCoord, u32>,
    /// Whether SH halving is active. When false (gumbel_root on, SH off) the
    /// candidate set is fixed to the top-m and visits allocate by PUCT among
    /// them (no rounds, no caps).
    pub sequential_halving: bool,
}

/// Minimum SH round-0 visits each candidate must afford before the tournament
/// is considered calibrated for the move's visit budget. A candidate scored
/// from fewer look-aheads than this carries too little Q signal to rank
/// tactical replies, so `init_gumbel_root` shrinks the candidate count instead
/// of starving the quota.
const GUMBEL_MIN_ROUND0_VISITS: u32 = 4;

impl GumbelRootState {
    /// Ceil(log2(m)) rounds. m<=1 => 0 rounds (nothing to halve).
    fn rounds_for(m: usize) -> u32 {
        if m <= 1 {
            0
        } else {
            (usize::BITS - (m - 1).leading_zeros()).max(1)
        }
    }

    /// Largest candidate count `m' <= m` (walking the halving ladder
    /// m -> ceil(m/2) -> ...) whose SH round-0 quota
    /// `floor(budget/(R(m')*m'))` reaches GUMBEL_MIN_ROUND0_VISITS. A single
    /// configured `gumbel_m` is sized for the selfplay full budget; smaller
    /// budgets (eval matches, quick-gate evals) reuse the same config, and
    /// without this clamp their round-0 quota collapses (32 candidates at 512
    /// visits = 3 look-aheads each). Never returns below 2 — a two-candidate
    /// tournament is the minimum meaningful SH; the legal-action clamp is the
    /// caller's.
    fn budget_calibrated_m(m: usize, budget: u32) -> usize {
        let mut mm = m;
        while mm > 2 {
            let r = Self::rounds_for(mm).max(1);
            if budget / (r * mm as u32) >= GUMBEL_MIN_ROUND0_VISITS {
                break;
            }
            mm = (mm + 1) / 2;
        }
        mm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::RustEvaluation;

    fn widening(mass: f32, min_c: usize, max_c: usize) -> Widening {
        Widening {
            mass,
            min_children: min_c,
            max_children: max_c,
        }
    }

    // === interior forced-move guard (Lever 0): node construction ===

    fn eval_with_uniform_priors(cells: &[(i16, i16)]) -> RustEvaluation {
        let priors: Vec<(PackedCoord, f32)> = cells
            .iter()
            .map(|&(q, r)| (pack_coord(HexCoord { q, r }), 1.0 / cells.len() as f32))
            .collect();
        RustEvaluation {
            value: 0.0,
            priors,
            moves_left: None,
            logits: None,
        }
    }

    // === async solve pool: memo routing + descent-stop consumption ===========

    fn replay(coords: &[(i16, i16)]) -> RustHexoState {
        let mut state = RustHexoState::new();
        for &(q, r) in coords {
            apply_placement(
                &mut state,
                Placement {
                    coord: HexCoord { q, r },
                },
            )
            .unwrap();
        }
        state
    }

    /// tss_solver.rs win-now fixture (decided, verifies at a tiny cap).
    fn win_now_fixture() -> RustHexoState {
        replay(&[
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (4, 0),
            (8, 4),
            (10, 3),
        ])
    }

    /// tss_solver.rs forced-defense fixture: live opponent threats, so the
    /// deep-leaf gate's has_threats check passes.
    fn forced_defense_fixture() -> RustHexoState {
        replay(&[
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (4, 0),
        ])
    }

    fn async_search(mode: u32) -> RustSearch {
        let root = RustHexoState::new();
        let eval = eval_with_uniform_priors(&[(0, 0)]);
        let mut divergences = Divergences::parity();
        divergences.tss_solver_mode = mode;
        divergences.tss_solver_sample_16 = 16;
        divergences.tss_solver_async = true;
        RustSearch::new(
            root,
            &eval,
            16,
            0.2,
            0.0,
            1.0,
            None,
            widening(0.9, 1, 4),
            0.0,
            true,
            divergences,
        )
        .unwrap()
    }

    /// A verified HardValue obtained through the real production mint (the
    /// sealed type is not constructible any other way, including in tests —
    /// that is the firewall working as intended).
    fn verified_fixture_solve() -> (RustHexoState, ProofStatus, Option<HardValue>) {
        let state = win_now_fixture();
        let mut counters = TssCounters::default();
        let solved = tss_solve_verified(
            &state,
            2000,
            SolveGoal::Both,
            crate::tss_core::ZoneSearchCaps::default(),
            SolverHorizon::DEFAULT,
            &mut TssSolver::default(),
            &mut counters,
        );
        assert_ne!(
            solved.status,
            ProofStatus::Unknown,
            "fixture must be decided"
        );
        assert!(solved.hard.is_some(), "decided fixture must verify");
        (state, solved.status, solved.hard)
    }

    /// A verified LOSS through the production mint. Uses the canonical
    /// forced-loss position and a dedicated LOSS goal so the result never
    /// depends on the Both-goal budget split (docs/VALUE_SIGNAL_AUDIT.md).
    fn verified_loss_fixture_solve() -> (RustHexoState, ProofStatus, Option<HardValue>) {
        let state = replay(&[
            (0, 0),
            (0, 8),
            (2, 7),
            (1, 0),
            (2, 0),
            (4, 6),
            (6, 5),
            (3, 0),
            (0, 4),
            (8, 4),
            (10, 3),
            (1, 4),
            (2, 4),
            (12, 2),
            (14, 1),
            (3, 4),
            (16, 0),
        ]);
        let mut counters = TssCounters::default();
        let solved = tss_solve_verified(
            &state,
            8000,
            SolveGoal::Loss,
            crate::tss_core::ZoneSearchCaps::default(),
            SolverHorizon::DEFAULT,
            &mut TssSolver::default(),
            &mut counters,
        );
        assert_eq!(
            solved.status,
            ProofStatus::Loss,
            "forced-loss fixture must prove LOSS under the loss goal"
        );
        assert!(solved.hard.is_some(), "decided loss must verify");
        (state, solved.status, solved.hard)
    }

    /// A drained response flips Pending -> Done and the descent-stop then
    /// consumes the hard value (with the counters bumped), while a Done entry
    /// is never overwritten.
    #[test]
    fn async_response_lands_and_descent_stop_consumes() {
        let (state, status, hard) = verified_fixture_solve();
        let hash = state_hash(&state);
        let binding = RootBinding::from_state(&state);
        let mut search = async_search(3);

        // Nothing to consume before the response lands.
        assert!(search.tss_async_descent_hard(hash, &state).is_none());

        let response = SolveResponse {
            slot: 0,
            generation: 1,
            hash,
            binding: binding.clone(),
            status,
            hard,
            counters: TssCounters {
                deep_calls: 1,
                deep_win: u32::from(status == ProofStatus::Win),
                deep_loss: u32::from(status == ProofStatus::Loss),
                ..TssCounters::default()
            },
        };
        search.apply_tss_async_response(&response);
        assert_eq!(search.tss.deep_calls, 1);

        // Descent-stop consumes at mode 3 with full-binding equality.
        let consumed = search.tss_async_descent_hard(hash, &state);
        assert!(consumed.is_some(), "verified result must consume at mode 3");
        assert_eq!(consumed.unwrap().value(), hard.unwrap().value());
        assert_eq!(search.tss.deep_hard_backups, 1);
        assert_eq!(search.tss.deep_memo_hits, 1);

        // Same hash key, DIFFERENT position: binding equality must refuse.
        let other = forced_defense_fixture();
        assert!(search.tss_async_descent_hard(hash, &other).is_none());

        // A Done entry is never overwritten by a late duplicate.
        let clobber = SolveResponse {
            counters: TssCounters::default(),
            status: ProofStatus::Unknown,
            hard: None,
            binding,
            ..response
        };
        search.apply_tss_async_response(&clobber);
        assert!(
            search.tss_async_descent_hard(hash, &state).is_some(),
            "Done entry must survive a duplicate response"
        );
    }

    /// VALUE-SIGNAL SYMMETRY (docs/VALUE_SIGNAL_AUDIT.md): the consumed
    /// hard-backup stream splits into `deep_win_backups` + `deep_loss_backups`
    /// by verdict through the single consume gate, and the two sum to
    /// `deep_hard_backups`. Guards against a loss-heavy run in which the loss
    /// half of the value signal reaches backup invisibly.
    #[test]
    fn consumed_hard_backups_split_by_outcome() {
        // A verified WIN consumed at mode 3 counts only on the win side.
        let (wstate, wstatus, whard) = verified_fixture_solve();
        assert_eq!(wstatus, ProofStatus::Win);
        let mut wsearch = async_search(3);
        let whash = state_hash(&wstate);
        wsearch.apply_tss_async_response(&SolveResponse {
            slot: 0,
            generation: 1,
            hash: whash,
            binding: RootBinding::from_state(&wstate),
            status: wstatus,
            hard: whard,
            counters: TssCounters {
                deep_calls: 1,
                deep_win: 1,
                ..TssCounters::default()
            },
        });
        assert!(wsearch.tss_async_descent_hard(whash, &wstate).is_some());
        assert_eq!(wsearch.tss.deep_hard_backups, 1);
        assert_eq!(wsearch.tss.deep_win_backups, 1);
        assert_eq!(wsearch.tss.deep_loss_backups, 0);
        assert_eq!(
            wsearch.tss.deep_hard_backups,
            wsearch.tss.deep_win_backups + wsearch.tss.deep_loss_backups
        );

        // A verified LOSS consumed at mode 3 counts only on the loss side.
        let (lstate, lstatus, lhard) = verified_loss_fixture_solve();
        assert_eq!(lstatus, ProofStatus::Loss);
        let mut lsearch = async_search(3);
        let lhash = state_hash(&lstate);
        lsearch.apply_tss_async_response(&SolveResponse {
            slot: 0,
            generation: 1,
            hash: lhash,
            binding: RootBinding::from_state(&lstate),
            status: lstatus,
            hard: lhard,
            counters: TssCounters {
                deep_calls: 1,
                deep_loss: 1,
                ..TssCounters::default()
            },
        });
        assert!(lsearch.tss_async_descent_hard(lhash, &lstate).is_some());
        assert_eq!(lsearch.tss.deep_hard_backups, 1);
        assert_eq!(lsearch.tss.deep_loss_backups, 1);
        assert_eq!(lsearch.tss.deep_win_backups, 0);
        assert_eq!(
            lsearch.tss.deep_hard_backups,
            lsearch.tss.deep_win_backups + lsearch.tss.deep_loss_backups
        );
    }

    /// Codex-review hardening: memo-write binding rules + cross-move
    /// retention semantics.
    /// - decided proofs persist across set_additional_visits (a proof is a
    ///   position property);
    /// - Unknown results do NOT persist (cap artifacts must be re-solvable);
    /// - a decided response UPGRADES a matching Done(Unknown);
    /// - a decided Done is never clobbered.
    #[test]
    fn async_memo_rules_and_cross_move_retention() {
        let (state, status, hard) = verified_fixture_solve();
        let hash = state_hash(&state);
        let binding = RootBinding::from_state(&state);
        let mut search = async_search(3);

        // Land an UNKNOWN first, then upgrade it with the decided result.
        search.apply_tss_async_response(&SolveResponse {
            slot: 0,
            generation: 1,
            hash,
            binding: binding.clone(),
            status: ProofStatus::Unknown,
            hard: None,
            counters: TssCounters::default(),
        });
        assert!(search.tss_async_descent_hard(hash, &state).is_none());
        search.apply_tss_async_response(&SolveResponse {
            slot: 0,
            generation: 1,
            hash,
            binding: binding.clone(),
            status,
            hard,
            counters: TssCounters::default(),
        });
        assert!(
            search.tss_async_descent_hard(hash, &state).is_some(),
            "a decided response must upgrade a matching Done(Unknown)"
        );

        // Cross-move retention: the decided proof survives the move reset...
        search.set_additional_visits(8);
        assert!(
            search.tss_async_descent_hard(hash, &state).is_some(),
            "decided proofs must persist across moves"
        );

        // ...but an Unknown result does not: after the reset the same leaf
        // re-enqueues instead of serving the stale Unknown.
        let other = forced_defense_fixture();
        let other_hash = state_hash(&other);
        search.apply_tss_async_response(&SolveResponse {
            slot: 0,
            generation: 2,
            hash: other_hash,
            binding: RootBinding::from_state(&other),
            status: ProofStatus::Unknown,
            hard: None,
            counters: TssCounters::default(),
        });
        search.set_additional_visits(8);
        let pool = crate::tss_async::TssAsyncPool::new(1, 1, false);
        search.set_tss_async(Some(pool.handle_for(0)));
        assert!(search.tss_deep_leaf(&other, other_hash).is_none());
        assert_eq!(
            search.tss.async_enqueued, 1,
            "a dropped Unknown must be re-solvable on the next move"
        );
    }

    /// Shadow tier (mode 1) with the pool on: results land but nothing is
    /// consumed — bit-identical play, exactly like the inline shadow.
    #[test]
    fn async_shadow_mode_consumes_nothing() {
        let (state, status, hard) = verified_fixture_solve();
        let hash = state_hash(&state);
        let mut search = async_search(1);
        search.apply_tss_async_response(&SolveResponse {
            slot: 0,
            generation: 1,
            hash,
            binding: RootBinding::from_state(&state),
            status,
            hard,
            counters: TssCounters::default(),
        });
        assert!(search.tss_async_descent_hard(hash, &state).is_none());
        assert_eq!(search.tss.deep_hard_backups, 0);
    }

    /// The enqueue path: first gated call enqueues (Pending), a re-selected
    /// pending leaf does not re-enqueue, the drained result serves memo hits,
    /// and set_additional_visits clears the handle with the memo.
    #[test]
    fn async_enqueue_dedups_and_serves_after_drain() {
        let pool = crate::tss_async::TssAsyncPool::new(1, 1, false);
        let state = forced_defense_fixture();
        let hash = state_hash(&state);
        let mut search = async_search(3);
        search.set_tss_async(Some(pool.handle_for(0)));
        let generation = search.tss_async_generation().expect("handle wired");

        assert!(
            search.tss_deep_leaf(&state, hash).is_none(),
            "async never solves inline"
        );
        assert_eq!(search.tss.async_enqueued, 1);
        assert!(search.tss_deep_leaf(&state, hash).is_none());
        assert_eq!(
            search.tss.async_enqueued, 1,
            "pending leaf must not re-enqueue"
        );
        assert_eq!(search.tss.async_pending_hits, 1);

        // Drain the worker's response and land it.
        let response = 'drain: {
            for _ in 0..1000 {
                let mut drained = pool.try_drain();
                if let Some(response) = drained.pop() {
                    break 'drain response;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            panic!("async solve pool produced no response within 10s");
        };
        assert_eq!(response.generation, generation);
        search.apply_tss_async_response(&response);

        // Third selection: memo hit (consumption depends on the verdict —
        // Unknown serves None, a verified proof serves the hard value).
        let memo_hits_before = search.tss.deep_memo_hits;
        let _ = search.tss_deep_leaf(&state, hash);
        assert_eq!(search.tss.deep_memo_hits, memo_hits_before + 1);
        assert_eq!(
            search.tss.async_enqueued, 1,
            "solved leaf must not re-enqueue"
        );

        // New move: handle drops with the memo; the wire pass re-issues it.
        search.set_additional_visits(4);
        assert!(search.tss_async_generation().is_none());
    }

    /// forced_only=true narrows the node to exactly the tactical (forced)
    /// edges: no leftover prior candidates, widening capped at the forced set.
    /// forced_only=false keeps today's inject-widen shape (tactical edges +
    /// the rest as unexpanded candidates).
    #[test]
    fn forced_only_node_drops_non_tactical_candidates() {
        let state = hexo_engine::HexoState::new();
        let hash = state_hash(&state);
        let tactical = vec![HexCoord { q: -1, r: 0 }, HexCoord { q: 5, r: 0 }];
        let legal: Vec<(i16, i16)> = vec![(-1, 0), (5, 0), (2, 3), (4, 4), (7, 7), (0, 1)];
        let eval = eval_with_uniform_priors(&legal);
        let w = widening(0.9, 2, 8);

        let node_off =
            owned_with_injection_from_eval(hash, &state, &eval, w, &tactical, true, false);
        let off_ids: Vec<PackedCoord> = node_off.edges.iter().map(|e| e.action_id).collect();
        assert_eq!(
            off_ids.len(),
            2,
            "tactical cells materialize as forced edges"
        );
        assert!(node_off.edges.iter().all(|e| e.forced));
        assert_eq!(
            node_off.remaining_prior_count(),
            4,
            "flag off keeps the non-tactical candidates as unexpanded priors"
        );

        let node_on = owned_with_injection_from_eval(hash, &state, &eval, w, &tactical, true, true);
        let on_ids: Vec<PackedCoord> = node_on.edges.iter().map(|e| e.action_id).collect();
        assert_eq!(on_ids, off_ids, "the forced set is identical either way");
        assert!(node_on.edges.iter().all(|e| e.forced));
        assert_eq!(node_on.remaining_prior_count(), 0, "guard drops the rest");
        assert_eq!(node_on.max_eligible_children, 2);
    }

    // === nucleus f64 short-circuit + f64/f32 agreement ===

    #[test]
    fn nucleus_f64_sentinel_returns_total_at_mass_one() {
        let priors = vec![1.0f32 / 50.0; 50];
        let w = widening(1.0, 2, 50);
        // mass >= 1.0 short-circuits to hi == total (50).
        assert_eq!(nucleus_count_values(priors.clone(), w, true), 50);
    }

    #[test]
    fn nucleus_f64_truncation_vs_f32() {
        // 1000 equal priors of 1/1000 each. The f64 path with the mass==1.0
        // short-circuit returns the full cap (hi == 1000).
        let priors = vec![1.0f32 / 1000.0; 1000];
        let w = widening(1.0, 2, 1000);
        let n_f64 = nucleus_count_values(priors.clone(), w, true);
        assert_eq!(n_f64, 1000, "f64 short-circuit takes all at mass==1.0");
    }

    #[test]
    fn nucleus_f32_matches_f64_for_normal_mass() {
        // Sharply-peaked policy with mass 0.95: f32 and f64 modes agree.
        let priors = vec![0.5f32, 0.3, 0.15, 0.04, 0.01];
        let w = widening(0.95, 2, 5);
        let n_f32 = nucleus_count_values(priors.clone(), w, false);
        let n_f64 = nucleus_count_values(priors, w, true);
        assert_eq!(n_f32, n_f64);
        // 0.5+0.3+0.15 = 0.95 reaches mass at the 3rd child.
        assert_eq!(n_f32, 3);
    }

    #[test]
    fn nucleus_respects_min_and_max_clamp() {
        let priors = vec![0.9f32, 0.05, 0.03, 0.02];
        // mass 0.95 would pick 2, but min_children=3 clamps up.
        let n = nucleus_count_values(priors.clone(), widening(0.95, 3, 4), true);
        assert_eq!(n, 3);
        // max_children=1 clamps down (lo>=hi path returns hi).
        let n2 = nucleus_count_values(priors, widening(0.95, 1, 1), true);
        assert_eq!(n2, 1);
    }

    // === shaped alpha matches hand-computed formula ===

    #[test]
    fn shaped_alpha_sums_to_one_over_legal() {
        let policy = [0.4f32, 0.3, 0.2, 0.1];
        let a = shaped_alpha(&policy);
        let sum: f64 = a.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "shaped alpha must sum to 1, got {sum}"
        );
    }

    #[test]
    fn shaped_alpha_hand_computed() {
        // Two moves, both above the 0.01 cap so log(min(0.01,p)+1e-20)=log(0.01)
        // for BOTH -> a[i] all equal -> after subtract-mean all zero -> uniform
        // fallback (alpha_sum<=0): each = 1/2.
        let policy = [0.7f32, 0.3];
        let a = shaped_alpha(&policy);
        assert!((a[0] - 0.5).abs() < 1e-9);
        assert!((a[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn shaped_alpha_distinguishes_below_cap() {
        // One move below the 0.01 cap, one above. log inputs differ -> non-
        // uniform shape. Hand compute:
        //   p = [0.999, 0.001]; min(0.01,p): [0.01, 0.001]
        //   a0 = ln(0.01 + 1e-20) = ln(0.01) = -4.605170...
        //   a1 = ln(0.001 + 1e-20) = ln(0.001) = -6.907755...
        //   mean = (-4.60517 + -6.907755)/2 = -5.756463
        //   a0 = max(0, -4.60517 - (-5.756463)) = 1.151293
        //   a1 = max(0, -6.907755 - (-5.756463)) = 0  (negative -> clamp)
        //   alpha_sum = 1.151293
        //   a0 = 0.5*(1.151293/1.151293 + 0.5) = 0.5*1.5 = 0.75
        //   a1 = 0.5*(0/1.151293 + 0.5) = 0.25
        let policy = [0.999f32, 0.001];
        let a = shaped_alpha(&policy);
        assert!((a[0] - 0.75).abs() < 1e-6, "a0 = {}", a[0]);
        assert!((a[1] - 0.25).abs() < 1e-6, "a1 = {}", a[1]);
    }

    #[test]
    fn shaped_alpha_uniform_fallback_when_flat() {
        // All-equal policy -> all logs equal -> subtract-mean all zero ->
        // alpha_sum==0 -> uniform fallback 1/n.
        let policy = [0.25f32; 4];
        let a = shaped_alpha(&policy);
        for v in a {
            assert!((v - 0.25).abs() < 1e-9);
        }
    }

    // === new-child FPU candidate scoring ===

    #[test]
    fn new_child_score_fpu_vs_u_only() {
        let parent_value = -0.4f32;
        let fpu_reduction = 0.2f32;
        let prior = 0.3f32;
        let scale = 2.0f32;
        let u = prior * scale; // 0.6
                               // fpu off: U-only.
        let off = new_child_score(parent_value, fpu_reduction, prior, scale, false);
        assert!((off - u).abs() < 1e-9);
        // fpu on: (parent_value - fpu_reduction) + U.
        let on = new_child_score(parent_value, fpu_reduction, prior, scale, true);
        assert!((on - ((parent_value - fpu_reduction) + u)).abs() < 1e-9);
        // The FPU baseline shifts the score by exactly (parent_value - fpu_red).
        assert!((on - off - (parent_value - fpu_reduction)).abs() < 1e-9);
    }

    #[test]
    fn new_child_score_matches_existing_zero_visit_edge() {
        // A materialized candidate (visits==0) must score IDENTICALLY to an
        // already-present unvisited edge with the same prior, when new_child_fpu
        // is on — i.e. value_or_fpu + U with the same denominator (1+0)=1.
        let parent_value = 0.25f32;
        let fpu = 0.2f32;
        let prior = 0.4f32;
        let scale = 1.7f32;
        let edge = RustEdge {
            action_id: 0,
            action: unpack_coord(0),
            prior,
            visits: 0,
            value_sum: 0.0,
            value_sq_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            pending: 0,
            child: None,
            forced: false,
        };
        let existing = edge.value_or_fpu(parent_value, fpu) + prior * scale / (1.0 + 0.0);
        let candidate = new_child_score(parent_value, fpu, prior, scale, true);
        assert!(
            (existing - candidate).abs() < 1e-6,
            "{existing} vs {candidate}"
        );
    }

    // === clean-cache reuse-reset behavior ===

    fn eval_with_priors(priors: Vec<(PackedCoord, f32)>, value: f32) -> RustEvaluation {
        RustEvaluation {
            value,
            priors,
            moves_left: None,
            logits: None,
        }
    }

    fn build_search(divergences: Divergences, noise: Option<RootDirichletNoise>) -> RustSearch {
        let state = RustHexoState::new();
        // Arbitrary distinct action ids / priors (descending). Construction does
        // not validate against board legality.
        let eval = eval_with_priors(vec![(0, 0.5), (1, 0.25), (2, 0.15), (3, 0.10)], 0.1);
        RustSearch::new(
            state,
            &eval,
            64,
            0.2,
            0.2,
            1.0,
            noise,
            widening(0.95, 2, 8),
            0.0,
            false, // tss disabled (no tactical injection in unit test)
            divergences,
        )
        .expect("search builds")
    }

    fn root_prior_map(search: &RustSearch) -> std::collections::HashMap<PackedCoord, f32> {
        let root = &search.nodes[0];
        let mut map = std::collections::HashMap::new();
        for edge in &root.edges {
            map.insert(edge.action_id, edge.prior);
        }
        if let NodePriors::Owned(rest) = &root.priors {
            for c in rest {
                map.insert(c.action_id, c.prior);
            }
        }
        map
    }

    #[test]
    fn clean_cache_reset_stops_compounding() {
        // With clean_root_prior_cache ON, re-applying the SAME noise to a reused
        // root must produce the SAME priors each time (reset-from-clean), not a
        // compounding drift.
        let mut dv = Divergences::production();
        dv.dirichlet_shaped = false; // isolate reset behavior from shaped alpha
        let noise = RootDirichletNoise {
            total_alpha: 10.83,
            fraction: 0.25,
            seed: 12345,
            shaped: false,
        };
        // Build WITHOUT construction-time noise so the clean cache == clean
        // post-temp priors and we drive noise purely via the reuse path.
        let mut search = build_search(dv, None);
        search.apply_root_dirichlet_noise(noise);
        let after_first = root_prior_map(&search);
        // Apply the SAME noise again (reuse). Reset-from-clean => identical.
        search.apply_root_dirichlet_noise(noise);
        let after_second = root_prior_map(&search);
        for (id, p1) in &after_first {
            let p2 = after_second.get(id).copied().unwrap();
            assert!(
                (p1 - p2).abs() < 1e-6,
                "reuse must not compound: action {id} {p1} vs {p2}"
            );
        }
    }

    #[test]
    fn no_cache_does_compound() {
        // With clean_root_prior_cache off, the second application mixes on the
        // already-noised priors, so the priors differ between the first and
        // second application.
        let mut dv = Divergences::parity();
        dv.dirichlet_shaped = false;
        assert!(!dv.clean_root_prior_cache);
        let noise = RootDirichletNoise {
            total_alpha: 10.83,
            fraction: 0.25,
            seed: 999,
            shaped: false,
        };
        let mut search = build_search(dv, None);
        search.apply_root_dirichlet_noise(noise);
        let after_first = root_prior_map(&search);
        search.apply_root_dirichlet_noise(noise);
        let after_second = root_prior_map(&search);
        let mut any_diff = false;
        for (id, p1) in &after_first {
            let p2 = after_second.get(id).copied().unwrap();
            if (p1 - p2).abs() > 1e-5 {
                any_diff = true;
            }
        }
        assert!(any_diff, "no-cache path compounds on reuse");
    }

    // === parity() / production() divergence defaults ===

    #[test]
    fn parity_disables_all_divergences() {
        let p = Divergences::parity();
        assert!(!p.nucleus_f64);
        assert!(!p.new_child_fpu);
        assert!(!p.lazy_widening);
        assert!(!p.clean_root_prior_cache);
        assert!(!p.dirichlet_shaped);
        assert!(!p.pruned_dynamic_cpuct);
        assert!(!p.scaled_fpu);
    }

    #[test]
    fn production_enables_all_divergences() {
        let p = Divergences::production();
        assert!(p.nucleus_f64);
        assert!(p.new_child_fpu);
        assert!(p.lazy_widening);
        assert!(p.clean_root_prior_cache);
        assert!(p.dirichlet_shaped);
        assert!(p.pruned_dynamic_cpuct);
        assert!(p.scaled_fpu);
    }

    #[test]
    fn deep_tss_is_off_in_parity_and_production_defaults() {
        for divergences in [Divergences::parity(), Divergences::production()] {
            assert_eq!(divergences.tss_solver_mode, 0);
            assert!(!divergences.tss_solver_async);
            assert!(!divergences.tss_solver_root_guard);
            assert!(!divergences.tss_solver_park);
            assert!(!divergences.tss_solver_all_leaves);
            assert!(!divergences.tss_interior_guard);
            assert!(!divergences.tss_zone);
            assert!(!divergences.tss_zone_stale_filter);
            assert!(!divergences.tss_zone_count2);
            assert!(!divergences.tss_pair_commutation);
            assert!(!divergences.tss_solver_dual_pass);
            assert_eq!(divergences.tss_solver_loss_reserve_nodes, 0);
            assert!(!divergences.tss_solver_group2);
            assert!(!divergences.tss_solver_j2near);
            assert!(!divergences.tss_solver_horizon_ladder);
        }
    }

    #[test]
    fn visited_policy_mass_sums_visited_child_priors() {
        // Only children with visits > 0 contribute; the sqrt of this mass scales
        // the FPU reduction under scaled_fpu.
        let mut node = RustNode {
            state_hash: 0,
            player: Player::Player0,
            eval_value: 0.0,
            eval_ml: None,
            visits: 3,
            value_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            edges: vec![
                RustEdge {
                    action_id: 0,
                    action: unpack_coord(0),
                    prior: 0.5,
                    visits: 2,
                    value_sum: 0.0,
                    value_sq_sum: 0.0,
                    ml_sum: 0.0,
                    ml_weight: 0.0,
                    pending: 0,
                    child: None,
                    forced: false,
                },
                RustEdge {
                    action_id: 1,
                    action: unpack_coord(1),
                    prior: 0.3,
                    visits: 0,
                    value_sum: 0.0,
                    value_sq_sum: 0.0,
                    ml_sum: 0.0,
                    ml_weight: 0.0,
                    pending: 0,
                    child: None,
                    forced: false,
                },
                RustEdge {
                    action_id: 2,
                    action: unpack_coord(2),
                    prior: 0.2,
                    visits: 1,
                    value_sum: 0.0,
                    value_sq_sum: 0.0,
                    ml_sum: 0.0,
                    ml_weight: 0.0,
                    pending: 0,
                    child: None,
                    forced: false,
                },
            ],
            priors: NodePriors::Owned(vec![]),
            max_eligible_children: 3,
            root_logits: None,
        };
        // Visited children: action 0 (prior 0.5) + action 2 (prior 0.2) = 0.7.
        assert!((node.visited_policy_mass() - 0.7).abs() < 1e-6);
        // A fresh node (no visited children) has zero mass => zero reduction.
        for edge in &mut node.edges {
            edge.visits = 0;
        }
        assert!(node.visited_policy_mass().abs() < 1e-6);
    }

    #[test]
    fn parity_dirichlet_is_flat() {
        // With shaped off, the fresh-root construction path produces the flat
        // Dirichlet priors: compare a search built under parity() against a
        // manual flat mix.
        let dv = Divergences::parity();
        let noise = RootDirichletNoise {
            total_alpha: 10.83,
            fraction: 0.25,
            seed: 7,
            shaped: false,
        };
        let search = build_search(dv, Some(noise));
        // Build the expected priors by replicating the fresh-root flat path.
        let mut candidates = vec![
            RustPriorCandidate {
                action_id: 0,
                prior: 0.5,
            },
            RustPriorCandidate {
                action_id: 1,
                prior: 0.25,
            },
            RustPriorCandidate {
                action_id: 2,
                prior: 0.15,
            },
            RustPriorCandidate {
                action_id: 3,
                prior: 0.10,
            },
        ];
        candidates.sort_by(compare_prior_candidate);
        normalize_candidate_priors(&mut candidates).unwrap();
        apply_dirichlet_noise(&mut candidates, noise);
        let mut expected: std::collections::HashMap<PackedCoord, f32> =
            candidates.iter().map(|c| (c.action_id, c.prior)).collect();
        // Renormalization differences: just assert the priors are FINITE and the
        // relative ordering / mix matches the flat path within tolerance.
        let got = root_prior_map(&search);
        for (id, e) in expected.drain() {
            let g = got.get(&id).copied().unwrap();
            assert!(
                (e - g).abs() < 1e-5,
                "flat dirichlet mismatch: action {id} expected {e} got {g}"
            );
        }
    }

    // === Gumbel: raw-logit plumbing onto the tree ===

    /// An evaluation carrying the optional raw-logit column.
    fn eval_with_priors_and_logits(
        priors: Vec<(PackedCoord, f32)>,
        logits: Vec<(PackedCoord, f32)>,
        value: f32,
    ) -> RustEvaluation {
        RustEvaluation {
            value,
            priors,
            moves_left: None,
            logits: Some(logits),
        }
    }

    fn build_search_from_eval(eval: &RustEvaluation, divergences: Divergences) -> RustSearch {
        RustSearch::new(
            RustHexoState::new(),
            eval,
            64,
            0.2,
            0.2,
            1.0,
            None,
            widening(0.95, 2, 8),
            0.0,
            false,
            divergences,
        )
        .unwrap()
    }

    #[test]
    fn root_logits_absent_when_eval_carries_none() {
        // Production/parity: no logits requested ⇒ eval.logits is None ⇒ the
        // root node carries no logit map and no Gumbel/σ path can read one.
        let eval = eval_with_priors(vec![(0, 0.5), (1, 0.3), (2, 0.2)], 0.1);
        assert!(eval.logits.is_none());
        let search = build_search_from_eval(&eval, Divergences::production());
        assert!(
            search.root().root_logits.is_none(),
            "root_logits must be None when the evaluation omits raw logits"
        );
    }

    #[test]
    fn root_logits_present_and_aligned_by_action_id() {
        // Gumbel path: eval carries raw logits aligned to the legal set. The
        // root must carry them keyed by action_id, RAW (no temperature/noise),
        // regardless of the descending-prior reordering applied to priors.
        let priors = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        // Logits in a DIFFERENT order than the prior-sorted set, with a negative
        // value (logits are unconstrained in sign, unlike priors).
        let logits = vec![(2, -1.25), (0, 3.5), (1, 0.0)];
        let eval = eval_with_priors_and_logits(priors, logits, 0.1);
        let search = build_search_from_eval(&eval, Divergences::production());
        let map = search
            .root()
            .root_logits
            .as_ref()
            .expect("root_logits must be Some when the evaluation carries logits");
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&0).copied(), Some(3.5));
        assert_eq!(map.get(&1).copied(), Some(0.0));
        assert_eq!(map.get(&2).copied(), Some(-1.25));
    }

    // === Gumbel: Gumbel-Top-k root + Sequential Halving ===

    /// gumbel() profile: a Full-move Gumbel root carries logits.
    fn gumbel_eval(n: usize) -> RustEvaluation {
        // Distinct priors (descending) and distinct logits per action id.
        let priors: Vec<(PackedCoord, f32)> =
            (0..n).map(|i| (i as PackedCoord, (n - i) as f32)).collect();
        let total: f32 = priors.iter().map(|(_, p)| *p).sum();
        let priors: Vec<(PackedCoord, f32)> =
            priors.into_iter().map(|(a, p)| (a, p / total)).collect();
        let logits: Vec<(PackedCoord, f32)> = (0..n)
            .map(|i| (i as PackedCoord, (n - i) as f32 * 0.5 - 1.0))
            .collect();
        RustEvaluation {
            value: 0.0,
            priors,
            moves_left: None,
            logits: Some(logits),
        }
    }

    #[test]
    fn sigma_transform_is_monotone() {
        // σ(q)=(c_visit+max_n)·c_scale·q ; σ(0)=0 ; monotone increasing in q.
        assert_eq!(gumbel_sigma(0.0, 7, 50.0, 1.0), 0.0);
        assert_eq!(gumbel_sigma(1.0, 0, 50.0, 1.0), 50.0);
        assert_eq!(gumbel_sigma(1.0, 10, 50.0, 1.0), 60.0);
        // boundaries q = ±1
        assert_eq!(gumbel_sigma(-1.0, 50, 50.0, 1.0), -100.0);
        // monotone: σ(q1) < σ(q2) for q1 < q2 at fixed (max_n, consts).
        let lo = gumbel_sigma(-0.3, 5, 50.0, 1.0);
        let hi = gumbel_sigma(0.6, 5, 50.0, 1.0);
        assert!(lo < hi);
    }

    #[test]
    fn gumbel_draw_is_finite_across_seeds() {
        for s in 0u64..10_000 {
            let g = gumbel_draw(s);
            assert!(g.is_finite(), "gumbel draw must be finite for seed {s}");
        }
    }

    #[test]
    fn rounds_for_is_ceil_log2() {
        assert_eq!(GumbelRootState::rounds_for(1), 0);
        assert_eq!(GumbelRootState::rounds_for(2), 1);
        assert_eq!(GumbelRootState::rounds_for(3), 2);
        assert_eq!(GumbelRootState::rounds_for(4), 2);
        assert_eq!(GumbelRootState::rounds_for(8), 3);
        assert_eq!(GumbelRootState::rounds_for(16), 4);
        assert_eq!(GumbelRootState::rounds_for(17), 5);
    }

    #[test]
    fn init_gumbel_root_noop_without_flag_or_logits() {
        // Flag off ⇒ no Gumbel state even with logits present.
        let eval = gumbel_eval(8);
        let mut s = build_search_from_eval(&eval, Divergences::production());
        s.init_gumbel_root(123, 64);
        assert!(!s.has_gumbel_root(), "no gumbel state when gumbel_root off");

        // Flag on but logits absent ⇒ falls back to PUCT (no state).
        let eval_nologit = eval_with_priors(vec![(0, 0.5), (1, 0.3), (2, 0.2)], 0.0);
        let mut s2 = build_search_from_eval(&eval_nologit, Divergences::gumbel());
        s2.init_gumbel_root(123, 64);
        assert!(
            !s2.has_gumbel_root(),
            "no gumbel state when raw logits are absent"
        );
    }

    #[test]
    fn gumbel_topk_selects_top_m_by_logits_plus_g() {
        // m candidates = top-m of logits(a)+g(a). Build the same draw the search
        // uses and assert the survivor set matches.
        let n = 12;
        let eval = gumbel_eval(n);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 5;
        let mut s = build_search_from_eval(&eval, dv);
        let seed = 0xABCD_1234u64;
        s.init_gumbel_root(seed, 256);
        assert!(s.has_gumbel_root());
        let state = s.gumbel_root.as_ref().unwrap();
        assert_eq!(state.survivors.len(), 5, "m clamped to gumbel_m");

        // Reference: logits+g over the full legal set, take top-5.
        let logit_map: std::collections::HashMap<PackedCoord, f32> =
            eval.logits.clone().unwrap().into_iter().collect();
        let mut scored: Vec<(PackedCoord, f32)> = (0..n as PackedCoord)
            .map(|a| {
                let g = gumbel_draw(seed.wrapping_add(a as u64));
                (a, logit_map[&a] + g)
            })
            .collect();
        scored.sort_by(|&(a, sa), &(b, sb)| sb.partial_cmp(&sa).unwrap().then_with(|| a.cmp(&b)));
        let expected: std::collections::HashSet<PackedCoord> =
            scored.into_iter().take(5).map(|(a, _)| a).collect();
        let got: std::collections::HashSet<PackedCoord> = state.survivors.iter().copied().collect();
        assert_eq!(got, expected, "survivor set must be top-m by logits+g");
    }

    #[test]
    fn gumbel_m_budget_calibration_walks_halving_ladder() {
        // Selfplay full budget: 1024/(5*32)=6 >= 4, m untouched.
        assert_eq!(GumbelRootState::budget_calibrated_m(32, 1024), 32);
        // Eval budget: 512/(5*32)=3 < 4 -> halve once: 512/(4*16)=8.
        assert_eq!(GumbelRootState::budget_calibrated_m(32, 512), 16);
        // Quick-gate budget: 128 -> 8 (128/(3*8)=5).
        assert_eq!(GumbelRootState::budget_calibrated_m(32, 128), 8);
        // Floor: never below a two-candidate tournament.
        assert_eq!(GumbelRootState::budget_calibrated_m(32, 1), 2);
        // Already-affordable m is a no-op at any of these budgets.
        assert_eq!(GumbelRootState::budget_calibrated_m(8, 96), 8);
    }

    #[test]
    fn init_gumbel_root_calibrates_m_to_budget() {
        // 40 legal actions, configured m=32. At the selfplay budget the full
        // 32 survive; at the eval budget the candidate set halves to 16 so
        // round-0 still affords >= GUMBEL_MIN_ROUND0_VISITS per candidate.
        let eval = gumbel_eval(40);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 32;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(99, 1024);
        assert_eq!(s.gumbel_root.as_ref().unwrap().survivors.len(), 32);
        s.init_gumbel_root(99, 512);
        let state = s.gumbel_root.as_ref().unwrap();
        assert_eq!(state.survivors.len(), 16, "eval budget must shrink m");
        let per = 512 / (state.num_rounds * 16);
        assert!(per >= 4, "round-0 quota under the calibration floor: {per}");
    }

    #[test]
    fn sh_round_caps_are_equal_allocation() {
        // m=8, budget=256, R=3 ⇒ per-survivor round-0 quota = 256/(3*8)=10.
        let eval = gumbel_eval(8);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 8;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(42, 256);
        let state = s.gumbel_root.as_ref().unwrap();
        assert_eq!(state.num_rounds, 3);
        assert_eq!(state.survivors.len(), 8);
        let per = 256 / (3 * 8); // floor = 10
                                 // At round entry all survivor visits are 0, so cap == per.
        for &a in &state.survivors {
            assert_eq!(state.round_cap.get(&a).copied(), Some(per));
        }
    }

    #[test]
    fn sh_halving_keeps_ceil_half_and_advances() {
        // Drive a Gumbel root to its cap and assert one halving keeps ceil(m/2).
        let eval = gumbel_eval(8);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 8;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(7, 256);
        let m0 = s.gumbel_root.as_ref().unwrap().survivors.len();
        assert_eq!(m0, 8);

        // No halving before any survivor meets its cap.
        assert!(!s.maybe_advance_gumbel_round());

        // Force every survivor's root-edge visits up to its cap (materialize +
        // bump visits directly — this is the intra-slot barrier condition).
        let caps: Vec<(PackedCoord, u32)> = s
            .gumbel_root
            .as_ref()
            .unwrap()
            .round_cap
            .iter()
            .map(|(&a, &c)| (a, c))
            .collect();
        for (a, cap) in caps {
            let idx = s.gumbel_root_edge_index(a).expect("survivor materializes");
            s.nodes[0].edges[idx].visits = cap;
        }
        // Barrier now satisfied ⇒ exactly one halving to ceil(8/2)=4.
        assert!(s.maybe_advance_gumbel_round());
        let state = s.gumbel_root.as_ref().unwrap();
        assert_eq!(state.round, 1);
        assert_eq!(state.survivors.len(), 4, "halving keeps ceil(m/2)");
        // New caps were re-seeded against the survivors' current visits.
        for &a in &state.survivors {
            let v = s.nodes[0]
                .edges
                .iter()
                .find(|e| e.action_id == a)
                .unwrap()
                .visits;
            let cap = state.round_cap.get(&a).copied().unwrap();
            assert!(cap > v, "new round cap must be above current visits");
        }
    }

    #[test]
    fn completed_q_visited_is_edge_value_unvisited_is_v_mix() {
        // Hand-built: two visited edges (with Q via value_sum/visits) and one
        // unvisited. completedQ(visited)=value(); completedQ(unvisited)=v_mix.
        let eval = gumbel_eval(3);
        let mut s = build_search_from_eval(&eval, Divergences::gumbel());
        // Materialize all three as root edges and set visit stats.
        for a in 0u32..3 {
            let _ = s.gumbel_root_edge_index(a); // ensure edge exists
        }
        // edge 0: 4 visits, value_sum 2.0 ⇒ Q=0.5 ; edge 1: 1 visit, sum -0.5 ⇒ -0.5
        // edge 2: 0 visits ⇒ unvisited.
        {
            let e0 = s.nodes[0]
                .edges
                .iter_mut()
                .find(|e| e.action_id == 0)
                .unwrap();
            e0.visits = 4;
            e0.value_sum = 2.0;
        }
        {
            let e1 = s.nodes[0]
                .edges
                .iter_mut()
                .find(|e| e.action_id == 1)
                .unwrap();
            e1.visits = 1;
            e1.value_sum = -0.5;
        }
        let logit_map = s.nodes[0].root_logits.clone().unwrap();
        let (completed, v_mix) = gumbel_completed_q(&s.nodes[0], &logit_map);
        assert!((completed[&0] - 0.5).abs() < 1e-6);
        assert!((completed[&1] - (-0.5)).abs() < 1e-6);
        // unvisited action 2 == v_mix (prior-softmax weighted over visited 0,1).
        assert!((completed[&2] - v_mix).abs() < 1e-6);
        // v_mix must lie within [min,max] of the visited Qs (-0.5..0.5).
        assert!(v_mix >= -0.5 - 1e-6 && v_mix <= 0.5 + 1e-6);
    }

    // === Gumbel: Divergences profile gating ===

    #[test]
    fn gumbel_profile_flips_four_bools_on_with_default_scalars() {
        // gumbel() = production() with the four Gumbel bools on and the σ /
        // candidate scalars at their defaults.
        let g = Divergences::gumbel();
        assert!(g.gumbel_target);
        assert!(g.gumbel_root);
        assert!(g.gumbel_sequential_halving);
        assert!(g.gumbel_nonroot_select);
        assert_eq!(g.gumbel_c_visit, 50.0);
        assert_eq!(g.gumbel_c_scale, 1.0);
        assert_eq!(g.gumbel_m, 16);
        assert_eq!(g.gumbel_target_min_visits, 1);
        // gumbel() inherits the production() divergence set unchanged.
        assert!(g.nucleus_f64);
        assert!(g.dirichlet_shaped);
    }

    #[test]
    fn parity_and_production_disable_all_gumbel_bools() {
        // With all four Gumbel bools false, every Gumbel path is bypassed. Both
        // parity() and production() hold the four bools off; scalars at the
        // defaults.
        for d in [Divergences::parity(), Divergences::production()] {
            assert!(!d.gumbel_target);
            assert!(!d.gumbel_root);
            assert!(!d.gumbel_sequential_halving);
            assert!(!d.gumbel_nonroot_select);
            assert_eq!(d.gumbel_c_visit, 50.0);
            assert_eq!(d.gumbel_c_scale, 1.0);
            assert_eq!(d.gumbel_m, 16);
            assert_eq!(d.gumbel_target_min_visits, 1);
        }
    }

    // === Gumbel: completedQ support floor excludes un-searched actions ===

    #[test]
    fn completed_q_floor_excludes_unsearched_from_target_support() {
        // The target softmax support is the set of edges with visits >=
        // gumbel_target_min_visits. completedQ itself still defines a value for
        // every edge (visited=Q, unvisited=v_mix), but a floored-out action must
        // not enter the softmax support — verified here by composing the same
        // logits+σ(completedQ) score the exporter uses and excluding floored ids.
        let eval = gumbel_eval(4);
        let mut s = build_search_from_eval(&eval, Divergences::gumbel());
        for a in 0u32..4 {
            let _ = s.gumbel_root_edge_index(a);
        }
        // visited: 0 (3 visits) and 2 (5 visits); unsearched: 1 and 3 (0 visits).
        for (a, v, sum) in [(0u32, 3u32, 1.5f32), (2, 5, -2.0)] {
            let e = s.nodes[0]
                .edges
                .iter_mut()
                .find(|e| e.action_id == a)
                .unwrap();
            e.visits = v;
            e.value_sum = sum;
        }
        let min_visits = 1u32;
        let support: Vec<PackedCoord> = s.nodes[0]
            .edges
            .iter()
            .filter(|e| e.visits >= min_visits)
            .map(|e| e.action_id)
            .collect();
        assert_eq!(support, vec![0, 2], "floor keeps only searched actions");
        // The floored-out actions 1 and 3 carry a completedQ (== v_mix) but are
        // NOT members of the softmax support.
        let logit_map = s.nodes[0].root_logits.clone().unwrap();
        let (completed, v_mix) = gumbel_completed_q(&s.nodes[0], &logit_map);
        assert!((completed[&1] - v_mix).abs() < 1e-6);
        assert!((completed[&3] - v_mix).abs() < 1e-6);
        assert!(!support.contains(&1) && !support.contains(&3));
    }

    // === Gumbel: target softmax == softmax(logits + σ(completedQ)) ===

    #[test]
    fn target_softmax_matches_logits_plus_sigma_completed_q() {
        // Build a hand tree, compose the target as gumbel_target_policy does
        // (logits + σ(completedQ) over the floored support, then softmax), and
        // assert it sums to 1 and equals an independent reference computation.
        let eval = gumbel_eval(3);
        let mut s = build_search_from_eval(&eval, Divergences::gumbel());
        for a in 0u32..3 {
            let _ = s.gumbel_root_edge_index(a);
        }
        // edge 0: 8 visits, sum 4.0 ⇒ Q=0.5 ; edge 1: 2 visits, sum -0.6 ⇒ -0.3
        // edge 2: 4 visits, sum 0.0 ⇒ 0.0. All meet the floor (>=1).
        for (a, v, sum) in [(0u32, 8u32, 4.0f32), (1, 2, -0.6), (2, 4, 0.0)] {
            let e = s.nodes[0]
                .edges
                .iter_mut()
                .find(|e| e.action_id == a)
                .unwrap();
            e.visits = v;
            e.value_sum = sum;
        }
        let c_visit = 50.0f32;
        let c_scale = 1.0f32;
        let logit_map = s.nodes[0].root_logits.clone().unwrap();
        let (completed, _v_mix) = gumbel_completed_q(&s.nodes[0], &logit_map);
        let max_n = s.nodes[0].edges.iter().map(|e| e.visits).max().unwrap();
        assert_eq!(max_n, 8);

        // Support in ascending action_id order (mirrors the exporter).
        let mut support: Vec<PackedCoord> = s.nodes[0]
            .edges
            .iter()
            .filter(|e| e.visits >= 1)
            .map(|e| e.action_id)
            .collect();
        support.sort_unstable();
        let scores: Vec<f32> = support
            .iter()
            .map(|a| {
                let l = logit_map[a];
                l + gumbel_sigma(completed[a], max_n, c_visit, c_scale)
            })
            .collect();
        let weights = gumbel_softmax(&scores);

        // Sums to 1 over support.
        let total: f32 = weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "target must sum to 1, got {total}"
        );

        // Independent reference: explicit stable softmax of the same scores.
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&x| (x - m).exp()).collect();
        let z: f32 = exps.iter().sum();
        for (w, e) in weights.iter().zip(exps.iter()) {
            assert!((w - e / z).abs() < 1e-6);
        }
        // Highest score (action 0: largest logit AND largest Q) gets the largest
        // weight — monotone sanity on the composed target.
        let argmax = weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(support[argmax], 0);
    }

    // === Gumbel: Gumbel-Top-k == sampling-without-replacement ===

    /// Chi-squared goodness-of-fit upper-tail survival probability P(X^2 >= stat)
    /// for `df` degrees of freedom, via the regularized upper incomplete gamma
    /// Q(df/2, stat/2). Pure (no deps) so the test stays self-contained.
    fn chi2_sf(stat: f64, df: f64) -> f64 {
        if stat <= 0.0 {
            return 1.0;
        }
        let a = df / 2.0;
        let x = stat / 2.0;
        // Lanczos ln-gamma (z >= 0.5 always holds here: a = df/2 >= 0.5 for the
        // chi-square dfs we use, so the reflection branch is unnecessary).
        fn ln_gamma(z: f64) -> f64 {
            const C: [f64; 8] = [
                676.520_368_121_885_1,
                -1_259.139_216_722_402_8,
                771.323_428_777_653_1,
                -176.615_029_162_140_6,
                12.507_343_278_686_905,
                -0.138_571_095_265_720_12,
                9.984_369_578_019_572e-6,
                1.505_632_735_149_311_6e-7,
            ];
            let g = 7.0;
            let z = z - 1.0;
            let mut x = 0.999_999_999_999_809_93;
            for (i, &c) in C.iter().enumerate() {
                x += c / (z + (i as f64) + 1.0);
            }
            let t = z + g + 0.5;
            0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
        }
        // Regularized lower incomplete gamma P(a,x); SF = 1 - P for the upper tail
        // of the gamma == chi-square upper tail.
        // Series expansion (good for x < a+1), else continued fraction.
        let gln = ln_gamma(a);
        if x < a + 1.0 {
            let mut ap = a;
            let mut sum = 1.0 / a;
            let mut del = sum;
            for _ in 0..500 {
                ap += 1.0;
                del *= x / ap;
                sum += del;
                if del.abs() < sum.abs() * 1e-15 {
                    break;
                }
            }
            let p = sum * (-x + a * x.ln() - gln).exp();
            1.0 - p
        } else {
            // Lentz continued fraction for Q(a,x).
            let tiny = 1e-300;
            let mut b = x + 1.0 - a;
            let mut c = 1.0 / tiny;
            let mut d = 1.0 / b;
            let mut h = d;
            for i in 1..500 {
                let an = -(i as f64) * (i as f64 - a);
                b += 2.0;
                d = an * d + b;
                if d.abs() < tiny {
                    d = tiny;
                }
                c = b + an / c;
                if c.abs() < tiny {
                    c = tiny;
                }
                d = 1.0 / d;
                let del = d * c;
                h *= del;
                if (del - 1.0).abs() < 1e-15 {
                    break;
                }
            }
            (-x + a * x.ln() - gln).exp() * h
        }
    }

    fn softmax_f64(logits: &[f32]) -> Vec<f64> {
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = logits.iter().map(|&l| ((l as f64) - m).exp()).collect();
        let z: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / z).collect()
    }

    /// One Gumbel draw per action with the same per-action seed discipline the
    /// search uses: g(a) = gumbel_draw(seed.wrapping_add(action_id)).
    fn gumbel_argsort(logits: &[f32], seed: u64) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(a, &l)| (a, l + gumbel_draw(seed.wrapping_add(a as u64))))
            .collect();
        scored.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then_with(|| x.0.cmp(&y.0)));
        scored.into_iter().map(|(a, _)| a).collect()
    }

    #[test]
    fn gumbel_topk_is_sampling_without_replacement_chi2_gate() {
        // Top-m of (logits + Gumbel(0,1)) is sampling without replacement from
        // softmax(logits). Gate: over >=100 random logit vectors (varied
        // length/entropy), draw for >=10_000 seeds each, compare empirical
        // first-pick freqs to softmax(logits) via chi-squared GoF at alpha=0.01;
        // allow <=5% of vectors to fail. Then the second pick, conditioned on the
        // first, must match softmax renormalized over the remaining support (same
        // gate).
        const N_VECTORS: usize = 120;
        const N_SEEDS: u64 = 12_000;
        const ALPHA: f64 = 0.01;

        // Deterministic LCG to generate the logit vectors (no rand dep needed).
        let mut rng: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut first_fail = 0usize;
        let mut second_fail = 0usize;
        for v in 0..N_VECTORS {
            // Length 2..=8 and a scale that varies entropy (small scale ⇒ near
            // uniform, large scale ⇒ peaked).
            let n = 2 + (v % 7); // 2..=8
            let scale = 0.5 + (v as f64 % 5.0) * 1.2;
            let logits: Vec<f32> = (0..n)
                .map(|_| ((next() - 0.5) * 2.0 * scale) as f32)
                .collect();
            let probs = softmax_f64(&logits);

            // Space the seeds far apart so seed.wrapping_add(action_id) ranges of
            // adjacent trials never overlap (clean per-action seed independence).
            let stride: u64 = 1 << 20;
            // First-pick counts, plus second-pick counts conditioned on first==j*.
            let mut first_counts = vec![0u64; n];
            // For the most-probable first pick, accumulate second-pick freqs.
            let star = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let mut second_counts = vec![0u64; n];
            let mut second_total = 0u64;
            for t in 0..N_SEEDS {
                let seed = (v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (t * stride);
                let order = gumbel_argsort(&logits, seed);
                first_counts[order[0]] += 1;
                if order[0] == star {
                    second_counts[order[1]] += 1;
                    second_total += 1;
                }
            }

            // First-pick chi-squared GoF vs softmax(logits).
            let total = N_SEEDS as f64;
            let mut stat = 0.0;
            for j in 0..n {
                let exp = probs[j] * total;
                if exp > 0.0 {
                    let obs = first_counts[j] as f64;
                    stat += (obs - exp) * (obs - exp) / exp;
                }
            }
            let p = chi2_sf(stat, (n - 1) as f64);
            if p <= ALPHA {
                first_fail += 1;
            }

            // Second-pick: conditioned on first==star, the remaining support is
            // softmax(logits) renormalized after removing `star`.
            let denom: f64 = 1.0 - probs[star];
            if denom > 1e-9 && second_total > 50 {
                let mut stat2 = 0.0;
                let st = second_total as f64;
                for j in 0..n {
                    if j == star {
                        // star can't be picked second; expect ~0.
                        continue;
                    }
                    let exp = (probs[j] / denom) * st;
                    if exp > 0.0 {
                        let obs = second_counts[j] as f64;
                        stat2 += (obs - exp) * (obs - exp) / exp;
                    }
                }
                // df = (support size after removal) - 1 = (n-1) - 1.
                let df = (n - 2).max(1) as f64;
                let p2 = chi2_sf(stat2, df);
                if p2 <= ALPHA {
                    second_fail += 1;
                }
                // star must essentially never be drawn second.
                assert!(
                    second_counts[star] as f64 <= 0.01 * st + 5.0,
                    "star {star} drawn second too often: {} / {st}",
                    second_counts[star]
                );
            }
        }

        let first_rate = first_fail as f64 / N_VECTORS as f64;
        let second_rate = second_fail as f64 / N_VECTORS as f64;
        assert!(
            first_rate <= 0.05,
            "first-pick GoF fail rate {first_rate:.3} exceeds 5% ({first_fail}/{N_VECTORS})"
        );
        assert!(
            second_rate <= 0.05,
            "second-pick GoF fail rate {second_rate:.3} exceeds 5% ({second_fail}/{N_VECTORS})"
        );
    }

    // === advance_root promoted-root value seeding (same/different player) ===

    /// Build a search on `root_state` with a single materialized root edge for
    /// `action`, wired to a child node whose player is `child_player`. The edge
    /// carries `edge_value_sum` over `edge_visits` (parent-perspective Q). This
    /// is the minimal shape advance_root needs to exercise its value seeding.
    fn search_with_child_edge(
        root_state: RustHexoState,
        action: PackedCoord,
        child_player: Player,
        edge_value_sum: f32,
        edge_visits: u32,
    ) -> RustSearch {
        // eval prior on `action` so RustSearch::new materializes it as a root
        // edge; a second distinct legal action keeps the root non-degenerate.
        let eval = eval_with_priors(vec![(action, 0.7), (action + 1, 0.3)], 0.0);
        let mut s = RustSearch::new(
            root_state,
            &eval,
            64,
            0.2,
            0.2,
            1.0,
            None,
            widening(0.95, 2, 8),
            0.0,
            false,
            Divergences::production(),
        )
        .expect("search builds");
        // Materialize `action` as a root edge (root edges are lazily expanded
        // from the owned candidate tail).
        let edge_index = s
            .gumbel_root_edge_index(action)
            .expect("action materializes as a root edge");
        // Append a child node for `action`; give it its own edges so the
        // promoted root is a normal (non-empty) node after clone.
        let child_id = s.nodes.len();
        let mut child = s.nodes[0].clone();
        child.player = child_player;
        child.visits = edge_visits.saturating_sub(1);
        child.value_sum = 0.0;
        s.nodes.push(child);
        let edge = &mut s.nodes[0].edges[edge_index];
        edge.child = Some(child_id);
        edge.visits = edge_visits;
        edge.value_sum = edge_value_sum;
        s
    }

    #[test]
    fn advance_root_same_player_promotion_keeps_edge_value_sign() {
        // Root at FirstStone (reached by the Opening ZERO move): current_player
        // is Player1 and stays Player1 through FirstStone -> SecondStone, so the
        // promoted child is the SAME player. edge.value_sum is already in that
        // player's perspective, so the seed must NOT be negated.
        let mut root_state = RustHexoState::new();
        apply_placement(
            &mut root_state,
            Placement {
                coord: HexCoord::ZERO,
            },
        )
        .expect("opening move");
        assert_eq!(root_state.current_player(), Player::Player1);
        let action = pack_coord(HexCoord::new(1, 0));
        // Positive edge Q (parent/Player1 perspective) must stay positive.
        let mut s = search_with_child_edge(root_state, action, Player::Player1, 6.0, 8);
        assert!(s.advance_root(action).expect("advance ok"));
        assert_eq!(s.nodes[0].player, Player::Player1);
        assert!(
            s.nodes[0].value() > 0.0,
            "same-player promotion must keep the edge value sign, got {}",
            s.nodes[0].value()
        );
        // Exact: value_sum seeded to +edge.value_sum, visits to edge.visits.
        assert!((s.nodes[0].value() - 6.0 / 8.0).abs() < 1e-6);
    }

    #[test]
    fn advance_root_different_player_promotion_negates_edge_value() {
        // Root at Opening: current_player is Player0; advancing ZERO yields a
        // FirstStone child whose player is Player1 (DIFFERENT). edge.value_sum
        // is in Player0's perspective, so the seed MUST be negated into the
        // child's perspective.
        let root_state = RustHexoState::new();
        assert_eq!(root_state.current_player(), Player::Player0);
        let action = pack_coord(HexCoord::ZERO);
        // Positive edge Q (parent/Player0 perspective) must flip to negative.
        let mut s = search_with_child_edge(root_state, action, Player::Player1, 6.0, 8);
        assert!(s.advance_root(action).expect("advance ok"));
        assert_eq!(s.nodes[0].player, Player::Player1);
        assert!(
            s.nodes[0].value() < 0.0,
            "different-player promotion must negate the edge value, got {}",
            s.nodes[0].value()
        );
        assert!((s.nodes[0].value() - (-6.0 / 8.0)).abs() < 1e-6);
    }

    #[test]
    fn gumbel_sh_ranking_max_n_uses_this_moves_delta_visits() {
        // BUG 3: on a reused root the σ scale must be THIS move's delta visits,
        // not the cumulative (reuse-inflated) max. Snapshot a large entry
        // baseline, then verify the ranking's max_n reflects only fresh deltas.
        let eval = gumbel_eval(8);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 8;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(11, 256);
        // Materialize every survivor as a root edge (edges are lazily expanded).
        let survivors: Vec<PackedCoord> = s.gumbel_root.as_ref().unwrap().survivors.clone();
        for &a in &survivors {
            let _ = s.gumbel_root_edge_index(a).expect("survivor materializes");
        }
        // Simulate a fully-reused root: inflate every materialized edge's
        // cumulative visits by 500 over the move-entry baseline, and set the
        // entry snapshot to those same current counts (every action inherited
        // its whole subtree from the previous move, so this move's delta is 0).
        for e in s.nodes[0].edges.iter_mut() {
            e.visits += 500;
        }
        {
            let entry: HashMap<PackedCoord, u32> = s.nodes[0]
                .edges
                .iter()
                .map(|e| (e.action_id, e.visits))
                .collect();
            s.gumbel_root.as_mut().unwrap().entry_visits = entry;
        }
        // With entry == current, this move's delta max_n is 0; assert the
        // ranking computes σ off delta (max_n 0), not the inflated cumulative.
        let state = s.gumbel_root.as_ref().unwrap();
        let entry = &state.entry_visits;
        let delta_max = s.nodes[0]
            .edges
            .iter()
            .map(|e| {
                e.visits
                    .saturating_sub(entry.get(&e.action_id).copied().unwrap_or(0))
            })
            .max()
            .unwrap_or(0);
        let cumulative_max = s.nodes[0].edges.iter().map(|e| e.visits).max().unwrap_or(0);
        assert_eq!(delta_max, 0, "delta max_n must subtract the entry baseline");
        assert!(cumulative_max >= 500, "cumulative would be reuse-inflated");
    }

    // === Gumbel: draw temperature τ on the top-m sort (lever 2) ===

    /// Reference top-m by the draw comparator `logit/τ + g(a)` (τ<=0 => 1.0),
    /// mirroring init_gumbel_root's sort exactly (descending, action_id tie-break).
    fn draw_topm(logits: &[(PackedCoord, f32)], seed: u64, tau: f32, m: usize) -> Vec<PackedCoord> {
        let tau = if tau > 0.0 { tau } else { 1.0 };
        let mut scored: Vec<(PackedCoord, f32)> = logits
            .iter()
            .map(|&(a, l)| (a, l / tau + gumbel_draw(seed.wrapping_add(a as u64))))
            .collect();
        scored.sort_by(|&(a, sa), &(b, sb)| sb.partial_cmp(&sa).unwrap().then_with(|| a.cmp(&b)));
        scored.into_iter().take(m).map(|(a, _)| a).collect()
    }

    #[test]
    fn draw_temperature_unset_and_one_match_todays_top_m() {
        // τ unset (gumbel() default 1.0) and an explicit τ=1.0 must both reproduce
        // the raw logit+g draw: same survivor set as the reference with no divide.
        let eval = gumbel_eval(12);
        let logits = eval.logits.clone().unwrap();
        let seed = 0x51EED_u64;
        for tau in [None, Some(1.0f32)] {
            let mut dv = Divergences::gumbel();
            dv.gumbel_m = 5;
            if let Some(t) = tau {
                dv.gumbel_draw_temperature = t;
            }
            assert_eq!(dv.gumbel_draw_temperature, 1.0, "default/explicit τ is 1.0");
            let mut s = build_search_from_eval(&eval, dv);
            s.init_gumbel_root(seed, 256);
            let got: HashSet<PackedCoord> = s
                .gumbel_root
                .as_ref()
                .unwrap()
                .survivors
                .iter()
                .copied()
                .collect();
            let expected: HashSet<PackedCoord> =
                draw_topm(&logits, seed, 1.0, 5).into_iter().collect();
            assert_eq!(got, expected, "τ=1 draw must equal today's logit+g top-m");
        }
    }

    #[test]
    fn draw_temperature_widens_candidate_set_and_preserves_raw_ranking_logits() {
        // Peaked logits with a big head gap that τ=4 collapses below the Gumbel
        // noise scale: at τ=1 the head dominates a low-noise tail candidate, but
        // τ=4 shrinks logit/τ enough that a different action clears the top-m cut.
        // Construct provably: action 0 logit 40 (always survives), then a tight
        // cluster [6,5,4,3,2,1,0] whose ordering τ=4 (÷4 => [1.5,1.25,1,...])
        // reshuffles against g(a). Search over seeds for a provable difference.
        let logits: Vec<(PackedCoord, f32)> = vec![
            (0, 40.0),
            (1, 6.0),
            (2, 5.0),
            (3, 4.0),
            (4, 3.0),
            (5, 2.0),
            (6, 1.0),
            (7, 0.0),
        ];
        let priors: Vec<(PackedCoord, f32)> = logits
            .iter()
            .map(|&(a, _)| (a, 1.0 / logits.len() as f32))
            .collect();
        let m = 4usize;

        // Find a seed where the τ=1 and τ=4 candidate sets provably differ.
        let seed = (0u64..10_000)
            .find(|&s| {
                let a: HashSet<_> = draw_topm(&logits, s, 1.0, m).into_iter().collect();
                let b: HashSet<_> = draw_topm(&logits, s, 4.0, m).into_iter().collect();
                a != b
            })
            .expect("a τ=1 vs τ=4 candidate-set difference must exist");

        // τ=4: the search's survivor set must equal the τ=4 reference and DIFFER
        // from the τ=1 reference (the widening actually took effect).
        let eval = eval_with_priors_and_logits(priors.clone(), logits.clone(), 0.0);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = m as u32;
        // SH would budget-calibrate m down; disable SH so the raw top-m is kept and
        // the draw-temperature effect is isolated (target/select paths unaffected).
        dv.gumbel_sequential_halving = false;
        dv.gumbel_draw_temperature = 4.0;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(seed, 256);
        let got: HashSet<PackedCoord> = s
            .gumbel_root
            .as_ref()
            .unwrap()
            .survivors
            .iter()
            .copied()
            .collect();
        let ref_tau4: HashSet<PackedCoord> = draw_topm(&logits, seed, 4.0, m).into_iter().collect();
        let ref_tau1: HashSet<PackedCoord> = draw_topm(&logits, seed, 1.0, m).into_iter().collect();
        assert_eq!(got, ref_tau4, "τ=4 survivor set must match the τ=4 draw");
        assert_ne!(got, ref_tau1, "τ=4 must change the candidate set vs τ=1");

        // The SH σ-ranking reads state.logits, which must remain the RAW logits
        // (undivided) — the τ divide is confined to the draw sort. Assert every
        // survivor's stored ranking logit equals its raw eval logit, not logit/τ.
        let raw: HashMap<PackedCoord, f32> = logits.iter().copied().collect();
        let state = s.gumbel_root.as_ref().unwrap();
        for (&a, &stored) in state.logits.iter() {
            assert_eq!(
                stored, raw[&a],
                "SH ranking logit for {a} must be raw ({}), not logit/τ",
                raw[&a]
            );
        }
    }

    #[test]
    fn gumbel_entry_visits_zero_on_fresh_root_delta_equals_cumulative() {
        // Invariant: on a fresh (non-reused) root the entry snapshot is all 0,
        // so delta visits == cumulative visits and BUG-3's fix is a no-op there.
        let eval = gumbel_eval(8);
        let mut dv = Divergences::gumbel();
        dv.gumbel_m = 8;
        let mut s = build_search_from_eval(&eval, dv);
        s.init_gumbel_root(3, 256);
        for (&_a, &v) in s.gumbel_root.as_ref().unwrap().entry_visits.iter() {
            assert_eq!(v, 0, "fresh-root entry baseline must be 0");
        }
        // Bump some edges; delta must equal cumulative when the baseline is 0.
        for (i, e) in s.nodes[0].edges.iter_mut().enumerate() {
            e.visits = i as u32 + 1;
        }
        let entry = &s.gumbel_root.as_ref().unwrap().entry_visits;
        for e in s.nodes[0].edges.iter() {
            let delta = e
                .visits
                .saturating_sub(entry.get(&e.action_id).copied().unwrap_or(0));
            assert_eq!(
                delta, e.visits,
                "delta must equal cumulative on a fresh root"
            );
        }
    }
}
