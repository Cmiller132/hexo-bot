//! Proof-carrying forced-tree solver.
//!
//! The search fixes a `claimant` player and constructs a winning strategy for
//! that identity.  Nodes owned by the claimant are existential; nodes owned by
//! the other player are universal.  This is deliberately not negamax: a
//! `FirstStone` placement leaves the same player to place `SecondStone`.
//!
//! The implementation is a deterministic, proof-number-ordered depth-first
//! AND/OR proof constructor.  It is equivalent to the proof side of df-pn for
//! the three-valued interface used here: the most promising (lowest initial
//! proof-number) OR child is expanded first, while every child of an AND node
//! must produce a proof.  Failure or any resource exhaustion is `Unknown`; a
//! failed restricted attack is never interpreted as a proof for the opponent.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::mem::size_of;
use std::sync::Arc;

use std::time::Instant;

use hexo_engine::{
    apply_placement, hex_distance, Axis, HexCoord, HexoState as RustHexoState, Placement, Player,
    TurnPhase, WindowKey,
};

use crate::threats_shared as threats;
use crate::tss_core::{
    seed_band_radius, CertVerify, DeepResult, DeepSolve, ProofStatus, SolveCaps, SolveGoal,
    SolveStats, ZoneSearchCaps,
};

use crate::tss_verify::{
    CertCommutation, CertEdge, CertNode, CertNodeId, RootBinding, TssCertificate, TssVerifier,
    ZoneInfo, MAX_CERT_COMMUTATIONS, MAX_CERT_DEPTH, MAX_CERT_EDGES, MAX_CERT_NODES,
    MAX_CERT_ROOT_STONES, MAX_CERT_WITNESSES,
};

/// Solve-level child-ordering policy. The mode is configuration; the actual
/// coordinate weights are position-specific and are consumed by one solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SolveOrdering {
    #[default]
    Off,
    Prior,
}

impl SolveOrdering {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Prior => "prior",
        }
    }
}

#[derive(Clone, Debug, Default)]
struct OrderingHints {
    weights: HashMap<HexCoord, f32>,
}

impl OrderingHints {
    fn from_entries(entries: Vec<(HexCoord, f32)>) -> Self {
        Self {
            weights: entries.into_iter().collect(),
        }
    }

    fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    fn weight(&self, coord: HexCoord) -> Option<f32> {
        self.weights.get(&coord).copied()
    }
}

// Test-only NQ6 census/PN telemetry. The production solver has no field,
// branch, or callable entry point for this collector.

fn interior_census_lb_plies(phase: TurnPhase, census: u8) -> Option<u8> {
    if census > 5 {
        return None;
    }
    let m = match phase {
        TurnPhase::FirstStone if census >= 4 => 6 - census,
        TurnPhase::FirstStone => (7 - census).min(6),
        TurnPhase::SecondStone { .. } if census >= 3 => 6 - census,
        TurnPhase::SecondStone { .. } => (7 - census).min(6),
        TurnPhase::Opening => return None,
    };
    let index = usize::from(m.saturating_sub(1));
    match phase {
        TurnPhase::FirstStone => [1, 2, 5, 6, 9, 10].get(index).copied(),
        TurnPhase::SecondStone { .. } => [1, 4, 5, 8, 9, 12].get(index).copied(),
        TurnPhase::Opening => None,
    }
}

fn interior_census_coordinate_safe(state: &RustHexoState, h_rem: i64) -> bool {
    const SAFE: i64 = 16_383;
    if h_rem < 0 {
        return false;
    }
    let Some(radius) = h_rem.checked_add(1).and_then(|x| x.checked_mul(8)) else {
        return false;
    };
    let Some(limit) = SAFE.checked_sub(radius) else {
        return false;
    };
    state.board().occupied_cells().iter().all(|coord| {
        let q = i64::from(coord.q);
        let r = i64::from(coord.r);
        q.checked_add(r)
            .and_then(|sum| sum.checked_neg())
            .and_then(|s| Some((q.checked_abs()?, r.checked_abs()?, s.checked_abs()?)))
            .is_some_and(|(q_abs, r_abs, s_abs)| q_abs <= limit && r_abs <= limit && s_abs <= limit)
    })
}

#[derive(Clone, Copy, Debug)]
struct InteriorCensusGateEvaluation {
    dismiss: bool,
    nanos: u64,
}

/// Evaluate Contract 8.1/8.2 for one interior claimant-owned bounded WIN arm.
/// `None` means the node is outside the proved/elected scope and no census was
/// scanned. A non-dismissing `Some` is still a measured live evaluation.
fn evaluate_interior_census_gate(
    state: &RustHexoState,
    claimant: Player,
    root_ply: u32,
    semantic_horizon: u32,
) -> Option<InteriorCensusGateEvaluation> {
    if state.is_terminal()
        || state.current_player() != claimant
        || state.placements_made() <= root_ply
        || !matches!(
            state.phase(),
            TurnPhase::FirstStone | TurnPhase::SecondStone { .. }
        )
    {
        return None;
    }

    // Contract 8.2 requires widened, checked absolute-to-relative arithmetic.
    let base_wide = i64::from(state.placements_made());
    let semantic_wide = i64::from(semantic_horizon);
    let h_rem = semantic_wide.checked_sub(base_wide)?;
    if !(0..=8).contains(&h_rem) || !interior_census_coordinate_safe(state, h_rem) {
        return None;
    }

    let started = Instant::now();
    let mut census = 0u8;
    let mut invariant_ok = true;
    for entry in state.board().windows().entries() {
        let ac = entry.count(claimant);
        let dc = entry.count(claimant.other());
        if ac > 5 || dc > 5 {
            invariant_ok = false;
        }
        if ac > 0 && dc == 0 {
            census = census.max(ac);
        }
    }
    let lb_plies = invariant_ok
        .then(|| interior_census_lb_plies(state.phase(), census))
        .flatten();
    let dismiss = lb_plies.is_some_and(|lb| i64::from(lb) > h_rem);
    let nanos = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    Some(InteriorCensusGateEvaluation { dismiss, nanos })
}

/// A second, fixed guard in addition to `SolveCaps::node_cap`.  It bounds stack
/// depth even when a caller supplies an accidentally enormous node cap.
const MAX_SEARCH_DEPTH: usize = MAX_CERT_DEPTH;

/// Conservative allocator-header charge used by the explicit TT accounting.
const ALLOC_OVERHEAD: usize = 32;
/// The direct table never reserves more than this many inline slots.
const MAX_TT_SLOTS: usize = 1 << 20;
/// Expected bytes per slot.  Entries with larger position keys simply make the
/// table stop accepting replacements before the caller's byte cap is crossed.
const TARGET_BYTES_PER_TT_SLOT: usize = 256;
/// Shared entries own certificate fragments and are consequently wider than
/// solve-local entries.  The target only determines direct-table density; the
/// exact retained capacities below remain the authoritative byte accounting.
const TARGET_BYTES_PER_SHARED_TT_SLOT: usize = 512;
/// Internal positive fragments are promoted only while cheap to compact.  A
/// successful attempt root is offered regardless of these two tuning limits.
const MAX_PROMOTED_FRAGMENT_NODES: usize = 128;
const MAX_PROMOTED_FRAGMENT_EDGES: usize = 512;
/// Wide shared fragments may retain at most one eighth of the caller's TT cap.
/// Slots are allocated lazily after a solve, and the next solve subtracts only
/// bytes actually retained, so an empty/cold store leaves the historical wide
/// search cap byte-for-byte intact.
const WIDE_FRAGMENT_CAP_DIVISOR: usize = 8;
/// Bounded independently verified descendants collected from one attempt.
const MAX_WIDE_FRAGMENT_PROMOTIONS: usize = 64;
/// Fragment slots are wider than key-only PN-index entries.
const TARGET_BYTES_PER_PROVEN_FRAGMENT_SLOT: usize = 1024;
/// Solve-local TT sentinel for a fully explored restricted position with no
/// proof in the current wide/depth-bounded attempt. Certificate IDs can never
/// approach this value (`MAX_CERT_NODES` is 100k).
const LOCAL_TT_FAILED: CertNodeId = CertNodeId::MAX;

/// Optional attacker-universe expansions.  The default is deliberately the
/// historical narrow generator so production callers retain byte-identical
/// search behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Round3Flag {
    #[default]
    Off,
    Shadow,
    Consume,
}

impl Round3Flag {
    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Consume => "consume",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WidthOptions {
    vcf_pair_complete: bool,
    free_tempo_j2near: bool,
    quiet_turn_or_edges: Round3Flag,
    ranked_unforced_defender_zone: Round3Flag,
}

#[derive(Clone, Debug)]
pub(crate) struct KReplyKernel {
    pub eligible: bool,
    pub urgent: bool,
    pub cells: Vec<HexCoord>,
}

impl KReplyKernel {
    /// Exact retained view. The common nonurgent case borrows `Legal(P)`
    /// instead of allocating an identical vector.
    pub(crate) fn retained<'a>(&'a self, legal: &'a [HexCoord]) -> &'a [HexCoord] {
        if !self.eligible {
            &[]
        } else if self.urgent {
            &self.cells
        } else {
            legal
        }
    }
}

impl WidthOptions {
    pub(crate) fn vcf_pair_complete() -> Self {
        Self {
            vcf_pair_complete: true,
            free_tempo_j2near: false,
            quiet_turn_or_edges: Round3Flag::Off,
            ranked_unforced_defender_zone: Round3Flag::Off,
        }
    }

    pub(crate) fn vcf_pair_j2near() -> Self {
        Self {
            free_tempo_j2near: true,
            ..Self::vcf_pair_complete()
        }
    }

    pub(crate) fn round3_consume() -> Self {
        Self {
            vcf_pair_complete: true,
            free_tempo_j2near: false,
            quiet_turn_or_edges: Round3Flag::Consume,
            ranked_unforced_defender_zone: Round3Flag::Consume,
        }
    }

    fn consumes_quiet_turns(self) -> bool {
        self.quiet_turn_or_edges == Round3Flag::Consume
    }

    fn consumes_ranked_zone(self) -> bool {
        self.ranked_unforced_defender_zone == Round3Flag::Consume
    }
}

/// Process-environment switches sampled once at the public solve boundary.
/// Keeping the sample separate makes effective-configuration resolution a
/// pure operation that can also back the harness manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SolveRuntimeFlags {
    lazy_frontier: bool,
    interior_census_gate: bool,
    k_reply_consume: bool,
}

/// Fully resolved flags and memory caps used by one `solve_goal` invocation.
/// This is telemetry/configuration data only; the search never mutates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EffectiveSolveConfig {
    pub(crate) vcf_pair_complete: bool,
    pub(crate) free_tempo_j2near: bool,
    pub(crate) dual_pass: bool,
    pub(crate) ordering: &'static str,
    pub(crate) loss_reserve_nodes: u32,
    pub(crate) group2: bool,
    pub(crate) quiet_turn_or_edges: &'static str,
    pub(crate) ranked_unforced_defender_zone: &'static str,
    pub(crate) tt_enabled: bool,
    pub(crate) tt_bytes_cap: usize,
    pub(crate) shared_fragments_enabled: bool,
    pub(crate) fragment_store_cap_bytes: usize,
    pub(crate) lazy_frontier: bool,
    pub(crate) interior_census_gate: bool,
    pub(crate) k_reply_consume: bool,
    uses_wide_pn: bool,
    local_tt_cap: usize,
    shared_tt_cap: usize,
}

/// Reusable proof-carrying solver.  Its shared TT retains only complete,
/// self-contained positive proof fragments; solve-local arena IDs never cross
/// an attempt boundary.
#[derive(Debug)]
pub(crate) struct TssSolver {
    tt_enabled: bool,
    hash_mask: u64,
    shared_tt: SharedProofCache,
    /// Default-off T10/U22 wide proven-fragment path. Read once at construction.
    shared_fragments_enabled: bool,
    fragment_store: ProvenFragmentStore,
    zone: ZoneSearchCaps,
    width: WidthOptions,
    /// Reuse an undecided wide `Both` primal's unspent nodes for the dual
    /// claim. Default-off preserves the historical primal-only wide split.
    dual_pass: bool,
    ordering: SolveOrdering,
    /// Position-specific policy weights. `solve_goal` takes this value at its
    /// boundary, so even early returns leave no hint available to a later root.
    ordering_hints: Option<OrderingHints>,
    /// Hold this many post-root nodes out of the wide `Both` primal for an
    /// opponent-WIN attempt. A positive reserve schedules that attempt even
    /// without the leftover policy; if the primal returns early, an enabled
    /// dual pass upgrades it to every actual leftover node. Zero preserves the
    /// current full-primal allocation.
    loss_reserve_nodes: u32,
    /// Default-off v1 Group-2 reduced-fanout selector (narrow zone path only;
    /// DESIGN_G2_CERT_EXTENSION.md §2.4, task flag `tss_solver_group2`). When
    /// on, eligible unforced defender nodes attempt the exact FHW closure and
    /// emit `UniversalGroup2V1`; any failure falls back to the legacy paths
    /// and, at the attempt boundary, to a clean group2-off re-solve so the
    /// flag can never decide fewer positions than off.
    group2: bool,
    /// Leaf-profile overrides (PLAN_TSS_MCTS_INTEGRATION.md §3). When `Some`,
    /// they replace the per-solve environment reads for the lazy defender
    /// frontier and the interior census gate, so the trainer leaf/root/async
    /// path runs the campaign engine at the leaf-decided config deterministically
    /// (not conditioned on process env). `None` preserves the historical env
    /// behavior for the offline corpus/hunt harnesses.
    force_lazy_frontier: Option<bool>,
    force_interior_census_gate: Option<bool>,
}

impl Default for TssSolver {
    fn default() -> Self {
        let shared_fragments_enabled =
            std::env::var("TSS_SHARED_FRAGMENTS").ok().as_deref() == Some("1");
        Self {
            tt_enabled: true,
            hash_mask: u64::MAX,
            shared_tt: SharedProofCache::new(0, u64::MAX),
            shared_fragments_enabled,
            fragment_store: ProvenFragmentStore::new(0, u64::MAX),
            zone: ZoneSearchCaps::default(),
            width: WidthOptions::default(),
            dual_pass: false,
            ordering: SolveOrdering::Off,
            ordering_hints: None,
            loss_reserve_nodes: 0,
            group2: false,
            force_lazy_frontier: None,
            force_interior_census_gate: None,
        }
    }
}

impl TssSolver {
    /// Set the zone/commutation options for subsequent solves. Changing the
    /// options DROPS the persistent positive-fragment cache: cached fragments
    /// are verified proofs either way, but their node-cost provenance belongs
    /// to the profile that built them — reusing them across an ON→OFF flip
    /// contaminates A/B node counts and conditional determinism (Codex
    /// review, profile isolation). Same-options calls keep the warm cache.
    pub(crate) fn set_zone_options(&mut self, zone: ZoneSearchCaps) {
        if self.zone != zone {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.zone = zone;
    }

    /// Set the attacker-width profile for subsequent solves.  As with zone
    /// options, changing profiles drops cached positive fragments so their
    /// node-cost provenance cannot leak between narrow and wide searches.
    pub(crate) fn set_width_options(&mut self, width: WidthOptions) {
        if self.width != width {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.width = width;
    }

    /// Select the production leaf width profile. False is exactly the
    /// historical `vcf_pair_complete` profile; true selects the named J2near
    /// extension. Profile changes retain the standard cache-isolation rule.
    pub(crate) fn set_leaf_j2near(&mut self, j2near: bool) {
        self.set_width_options(if j2near {
            WidthOptions::vcf_pair_j2near()
        } else {
            WidthOptions::vcf_pair_complete()
        });
    }

    pub(crate) fn j2near_enabled(&self) -> bool {
        self.width.free_tempo_j2near
    }

    pub(crate) fn set_dual_pass(&mut self, dual_pass: bool) {
        if self.dual_pass != dual_pass {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.dual_pass = dual_pass;
    }

    pub(crate) fn set_ordering(&mut self, ordering: SolveOrdering) {
        if self.ordering != ordering {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.ordering = ordering;
    }

    /// Install policy weights for exactly the next `solve_goal`. Hinted
    /// searches are cold on both sides so their order-sensitive fragment
    /// provenance cannot affect either this solve or the following one.
    pub(crate) fn set_ordering_hints(&mut self, hints: Vec<(HexCoord, f32)>) {
        let hints = OrderingHints::from_entries(hints);
        if self.ordering == SolveOrdering::Prior && !hints.is_empty() {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.ordering_hints = Some(hints);
    }

    pub(crate) fn set_loss_reserve_nodes(&mut self, loss_reserve_nodes: u32) {
        if self.loss_reserve_nodes != loss_reserve_nodes {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.loss_reserve_nodes = loss_reserve_nodes;
    }

    /// Enable/disable the v1 Group-2 selector. Changing the option drops the
    /// persistent caches (same profile-isolation rule as the other options:
    /// cached fragments' node-cost provenance must not leak across profiles).
    pub(crate) fn set_group2(&mut self, group2: bool) {
        if self.group2 != group2 {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        self.group2 = group2;
    }

    /// Externally selected verifier policy follows this solver option
    /// (design §5.1: trainer configuration, never certificate contents,
    /// chooses the policy).
    pub(crate) fn group2_enabled(&self) -> bool {
        self.group2
    }

    /// Configure this solver to the campaign leaf-decided profile
    /// (PLAN_TSS_MCTS_INTEGRATION.md §3, HUNT_REPORT_LEAF_SURFACE config D):
    /// wide `vcf_pair_complete` attacker width, the lazy defender frontier ON,
    /// the interior census gate ON, shared fragments OFF and k-reply OFF (the
    /// profile's measured no-value knobs, left at their env/default state). The
    /// lazy/gate forces make the trainer leaf/root/async path deterministic and
    /// independent of process environment. The 256 KiB per-solve TT and the
    /// node cap are supplied per solve by the caller (`SolveCaps`).
    pub(crate) fn configure_leaf_profile(&mut self) {
        self.set_width_options(WidthOptions::vcf_pair_complete());
        self.force_lazy_frontier = Some(true);
        self.force_interior_census_gate = Some(true);
    }

    /// Sample process-global runtime switches once. The returned value is an
    /// explicit input to `effective_solve_config`, keeping resolution itself
    /// deterministic and side-effect free.
    pub(crate) fn sample_runtime_flags(&self) -> SolveRuntimeFlags {
        SolveRuntimeFlags {
            lazy_frontier: std::env::var("TSS_LAZY_FRONTIER").ok().as_deref() == Some("1"),
            interior_census_gate: std::env::var_os("TSS_INTERIOR_CENSUS_GATE")
                .is_some_and(|value| value == "1"),
            k_reply_consume: matches!(std::env::var("TSS_K_REPLY_CONSUME").as_deref(), Ok("1")),
        }
    }

    /// Pure effective-configuration resolver shared by real solves and the
    /// harness manifest. `fragment_store.current_bytes` is projected through
    /// the same reconfiguration rule the solve applies immediately afterward.
    pub(crate) fn effective_solve_config(
        &self,
        caps: &SolveCaps,
        runtime: SolveRuntimeFlags,
    ) -> EffectiveSolveConfig {
        let width = self.width;
        let tt_bytes_cap = if self.tt_enabled {
            caps.tt_bytes_cap
        } else {
            0
        };
        let uses_wide_pn = width.vcf_pair_complete
            && !(width.consumes_quiet_turns() && width.consumes_ranked_zone());
        let fragment_store_cap_bytes = if uses_wide_pn && self.shared_fragments_enabled {
            tt_bytes_cap / WIDE_FRAGMENT_CAP_DIVISOR
        } else {
            0
        };
        let fragment_store_bytes = if self.fragment_store.cap == fragment_store_cap_bytes
            && self.fragment_store.hash_mask == self.hash_mask
        {
            self.fragment_store.current_bytes
        } else {
            0
        };
        let (local_tt_cap, shared_tt_cap) = if uses_wide_pn && self.shared_fragments_enabled {
            (tt_bytes_cap.saturating_sub(fragment_store_bytes), 0)
        } else if width.vcf_pair_complete {
            (tt_bytes_cap, 0)
        } else {
            split_tt_cap(tt_bytes_cap)
        };
        EffectiveSolveConfig {
            vcf_pair_complete: width.vcf_pair_complete,
            free_tempo_j2near: width.free_tempo_j2near,
            dual_pass: self.dual_pass,
            ordering: self.ordering.name(),
            loss_reserve_nodes: self.loss_reserve_nodes,
            group2: self.group2,
            quiet_turn_or_edges: self.width.quiet_turn_or_edges.name(),
            ranked_unforced_defender_zone: self.width.ranked_unforced_defender_zone.name(),
            tt_enabled: self.tt_enabled,
            tt_bytes_cap,
            shared_fragments_enabled: self.shared_fragments_enabled,
            fragment_store_cap_bytes,
            lazy_frontier: self.force_lazy_frontier.unwrap_or(runtime.lazy_frontier),
            interior_census_gate: self
                .force_interior_census_gate
                .unwrap_or(runtime.interior_census_gate),
            k_reply_consume: runtime.k_reply_consume,
            uses_wide_pn,
            local_tt_cap,
            shared_tt_cap,
        }
    }

    /// Solve only for the requested root-perspective side(s).  One-sided modes
    /// receive the entire remaining node budget; the legacy trait entry point
    /// below delegates to `Both`.
    pub(crate) fn solve_goal(
        &mut self,
        state: &RustHexoState,
        caps: &SolveCaps,
        goal: SolveGoal,
    ) -> DeepResult<TssCertificate> {
        let ordering_hints = self.ordering_hints.take().and_then(|hints| {
            (self.ordering == SolveOrdering::Prior && !hints.is_empty()).then_some(hints)
        });
        let hinted = ordering_hints.is_some();
        if hinted {
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        let result = self.solve_goal_inner(state, caps, goal, ordering_hints.as_ref());
        if hinted {
            // Positive fragments are sound across orders, but their warm-hit
            // node costs are not. Drop both persistent stores to guarantee the
            // next unhinted position observes its ordinary cold ordering.
            self.shared_tt.clear();
            self.fragment_store.clear();
        }
        result
    }

    fn solve_goal_inner(
        &mut self,
        state: &RustHexoState,
        caps: &SolveCaps,
        goal: SolveGoal,
        ordering_hints: Option<&OrderingHints>,
    ) -> DeepResult<TssCertificate> {
        let runtime = self.sample_runtime_flags();
        let width = self.width;
        let effective = self.effective_solve_config(caps, runtime);

        self.fragment_store
            .reconfigure(effective.fragment_store_cap_bytes, self.hash_mask);
        debug_assert_eq!(
            effective.local_tt_cap,
            if effective.uses_wide_pn && effective.shared_fragments_enabled {
                effective
                    .tt_bytes_cap
                    .saturating_sub(self.fragment_store.current_bytes)
            } else if effective.vcf_pair_complete {
                effective.tt_bytes_cap
            } else {
                split_tt_cap(effective.tt_bytes_cap).0
            }
        );

        self.shared_tt
            .reconfigure(effective.shared_tt_cap, self.hash_mask);

        let initial_stats = SolveStats {
            peak_tt_bytes: self
                .shared_tt
                .current_bytes
                .saturating_add(self.fragment_store.current_bytes)
                as u64,
            fragment_store_entries: self.fragment_store.entry_count as u64,
            fragment_store_bytes: self.fragment_store.current_bytes as u64,
            ..SolveStats::default()
        };
        if caps.node_cap == 0
            || caps.semantic_horizon < state.placements_made()
            || state.board().len() > MAX_CERT_ROOT_STONES
        {
            return unknown(initial_stats);
        }

        // A root lambda-one/terminal result is both common and symmetric
        // between the primal and dual claims.  Count it as one examined node,
        // but filter it when the caller explicitly requested only the other
        // perspective.
        let mut stats = SolveStats {
            nodes: 1,
            ..initial_stats
        };
        if let Some((claimant, leaf)) = immediate_winner(state, width) {
            if node_resolution(&leaf) > caps.semantic_horizon {
                return unknown(stats);
            }
            let status = status_for_claimant(state.current_player(), claimant);
            if goal_accepts(goal, status) {
                let cert = TssCertificate {
                    root: RootBinding::from_state(state),
                    claimant,
                    root_node: 0,
                    nodes: vec![leaf],
                    semantic_horizon: caps.semantic_horizon,
                };
                return DeepResult {
                    status,
                    cert: Some(cert),
                    stats,
                };
            }
            return unknown(stats);
        }

        let remaining = caps.node_cap - 1;
        let (primal_cap, mut dual_cap) = match goal {
            SolveGoal::Win => (remaining, 0),
            SolveGoal::Loss => (0, remaining),
            // Pair-complete mode is deliberately a restricted VCF WIN search.
            // The default reserve is zero, preserving its full advertised
            // forcing-proof cap. A configured floor is an explicit policy
            // experiment and remains a positive opponent-claim search only;
            // its failure can establish no NO result.
            SolveGoal::Both if width.vcf_pair_complete => {
                wide_both_initial_caps(remaining, effective.loss_reserve_nodes)
            }
            SolveGoal::Both => ((remaining + 1) / 2, remaining / 2),
        };
        let root_player = state.current_player();

        if primal_cap > 0 {
            let attempt = self.prove_for(
                state,
                root_player,
                primal_cap,
                effective.local_tt_cap,
                caps.semantic_horizon,
                self.zone,
                width,
                effective.k_reply_consume,
                effective.interior_census_gate,
                effective.lazy_frontier,
                ordering_hints,
                effective.group2,
            );

            stats.merge(attempt.stats);
            if let Some(cert) = attempt.cert {
                return DeepResult {
                    status: ProofStatus::Win,
                    cert: Some(cert),
                    stats,
                };
            }
            if goal == SolveGoal::Both && effective.vcf_pair_complete && effective.dual_pass {
                dual_cap = caps.node_cap.saturating_sub(stats.nodes);
            }
        }

        if dual_cap > 0 {
            let attempt = self.prove_for(
                state,
                root_player.other(),
                dual_cap,
                effective.local_tt_cap,
                caps.semantic_horizon,
                self.zone,
                width,
                effective.k_reply_consume,
                effective.interior_census_gate,
                effective.lazy_frontier,
                ordering_hints,
                effective.group2,
            );

            stats.merge(attempt.stats);
            if let Some(cert) = attempt.cert {
                return DeepResult {
                    status: ProofStatus::Loss,
                    cert: Some(cert),
                    stats,
                };
            }
        }

        unknown(stats)
    }
}

impl DeepSolve for TssSolver {
    type Cert = TssCertificate;

    fn solve(&mut self, state: &RustHexoState, caps: &SolveCaps) -> DeepResult<Self::Cert> {
        self.solve_goal(state, caps, SolveGoal::Both)
    }
}

impl TssSolver {
    #[allow(clippy::too_many_arguments)]
    fn prove_for(
        &mut self,
        state: &RustHexoState,
        claimant: Player,
        node_cap: u64,
        local_tt_cap: usize,
        semantic_horizon: u32,
        zone: ZoneSearchCaps,
        width: WidthOptions,
        k_reply_consume: bool,
        interior_census_gate: bool,
        lazy_frontier: bool,
        ordering_hints: Option<&OrderingHints>,
        group2: bool,
    ) -> AttemptResult {
        if !width.vcf_pair_complete
            || (width.consumes_quiet_turns() && width.consumes_ranked_zone())
        {
            let zone = if width.consumes_ranked_zone() {
                ZoneSearchCaps {
                    enabled: true,
                    stale_area_filter: false,
                    count2_threshold: true,
                    pair_commutation: false,
                }
            } else {
                zone
            };
            return WidePnSearch::prove_narrow_compat(
                state,
                claimant,
                node_cap,
                local_tt_cap,
                self.hash_mask,
                &mut self.shared_tt,
                semantic_horizon,
                zone,
                width,
                MAX_SEARCH_DEPTH,
                k_reply_consume,
                interior_census_gate,
                group2,
            );
        }
        let depth_cap = wide_search_final_depth(state.placements_made(), semantic_horizon);

        self.prove_for_wide_pn_with_lazy_frontier(
            state,
            claimant,
            node_cap,
            local_tt_cap,
            semantic_horizon,
            depth_cap,
            width,
            interior_census_gate,
            lazy_frontier,
            ordering_hints,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prove_for_wide_pn_with_lazy_frontier(
        &mut self,
        state: &RustHexoState,
        claimant: Player,
        node_cap: u64,
        local_tt_cap: usize,
        semantic_horizon: u32,
        depth_cap: usize,
        width: WidthOptions,
        interior_census_gate: bool,
        lazy_frontier: bool,
        ordering_hints: Option<&OrderingHints>,
    ) -> AttemptResult {
        let fragments_enabled = self.shared_fragments_enabled;
        let shared_bytes = self
            .shared_tt
            .current_bytes
            .saturating_add(self.fragment_store.current_bytes);
        let fragment_store = fragments_enabled.then_some(&self.fragment_store);

        let mut search = WidePnSearch::new_with_width(
            claimant,
            state.placements_made(),
            node_cap,
            local_tt_cap,
            semantic_horizon,
            depth_cap,
            width,
            fragment_store,
        );
        search.interior_census_gate = interior_census_gate;
        search.lazy_frontier = lazy_frontier;
        search.ordering_hints = ordering_hints.cloned();
        let root = search.insert_root(state);

        search.run(state, root);

        let mut stats = SolveStats {
            nodes: search.expansions,
            expansions: search.expansions,
            tt_hits: search.tt_hits,
            tt_entries: (search.by_position.len() + self.shared_tt.entry_count()) as u64,
            peak_tt_bytes: shared_bytes.saturating_add(search.peak_bytes) as u64,
            horizon_cuts: search.horizon_cuts,
            kb_death_cuts: search.kb_death_cuts,

            fragment_lookups: search.fragment_lookups,
            fragment_hits: search.fragment_hits,
            fragment_store_entries: self.fragment_store.entry_count as u64,
            fragment_store_bytes: self.fragment_store.current_bytes as u64,
            interior_gate_evaluations: search.interior_gate_evaluations,
            interior_gate_dismissals: search.interior_gate_dismissals,
            interior_gate_nanos: search.interior_gate_nanos,

            ..SolveStats::default()
        };
        let mut promotions = Vec::new();
        let cert = search.materialize(state, root).and_then(|materialized| {
            let fragment_imports = materialized.fragment_imports;
            let _dag_reuses = materialized.dag_reuses;
            let (nodes, root_node) =
                compact_certificate(&materialized.arena, materialized.root_node)?;
            let mut cert = TssCertificate {
                root: RootBinding::from_state(state),
                claimant,
                root_node,
                nodes,
                semantic_horizon,
            };
            if fragments_enabled {
                rebase_shared_fragment_labels(&mut cert, state)?;
                let claimed = status_for_claimant(state.current_player(), claimant);
                if !TssVerifier.verify(state, &cert, claimed) {
                    return None;
                }
                if let Some(proof) = CachedProof::from_compact(cert.nodes.clone(), cert.root_node) {
                    promotions.push((PositionKey::from_state(state), proof));
                }
            } else {
                rebase_zone_distances(&mut cert, state)?;
            }
            // Count only imports that survive compaction, dominant relabel,
            // and (when enabled) strict final-certificate verification.
            stats.fragment_imports = fragment_imports;
            Some(cert)
        });

        if fragments_enabled {
            let mut promoted_keys = promotions
                .iter()
                .map(|(key, _)| key.clone())
                .collect::<HashSet<_>>();
            for candidate in &search.proven_candidates {
                if promotions.len() >= MAX_WIDE_FRAGMENT_PROMOTIONS {
                    break;
                }
                let key = PositionKey::from_state(&candidate.state);
                if promoted_keys.contains(&key) {
                    continue;
                }
                let Some(materialized) = search.materialize(&candidate.state, candidate.id) else {
                    continue;
                };
                let Some((nodes, root_node)) = compact_certificate_limited(
                    &materialized.arena,
                    materialized.root_node,
                    MAX_PROMOTED_FRAGMENT_NODES,
                    MAX_PROMOTED_FRAGMENT_EDGES,
                ) else {
                    continue;
                };
                let mut cert = TssCertificate {
                    root: RootBinding::from_state(&candidate.state),
                    claimant,
                    root_node,
                    nodes,
                    semantic_horizon,
                };
                if rebase_shared_fragment_labels(&mut cert, &candidate.state).is_none() {
                    continue;
                }
                let claimed = status_for_claimant(candidate.state.current_player(), claimant);
                if !TssVerifier.verify(&candidate.state, &cert, claimed) {
                    continue;
                }
                if let Some(proof) = CachedProof::from_compact(cert.nodes, cert.root_node) {
                    promoted_keys.insert(key.clone());
                    promotions.push((key, proof));
                }
            }
        }

        drop(search);
        // The solved attempt root was queued first; admit it last so a direct-
        // mapped collision with one of its descendants cannot erase the warm
        // repeat entry in the same promotion batch.
        promotions.reverse();
        for (key, proof) in promotions {
            self.fragment_store.insert(key, claimant, proof);
        }
        stats.fragment_store_entries = self.fragment_store.entry_count as u64;
        stats.fragment_store_bytes = self.fragment_store.current_bytes as u64;
        stats.peak_tt_bytes = stats
            .peak_tt_bytes
            .max(self.fragment_store.current_bytes as u64);
        AttemptResult { cert, stats }
    }
}

/// Imported zone fragments may have been built with a larger admissible
/// budget. Their searched set remains sound for the selected proof, but the
/// carried evidence must be relabelled from the assembled certificate itself:
/// exact D14 local budgets and the certificate's exact build horizon.
fn rebase_zone_distances(cert: &mut TssCertificate, root: &RustHexoState) -> Option<()> {
    let mut states = vec![None; cert.nodes.len()];
    let mut stack = vec![(cert.root_node, root.clone())];
    while let Some((id, state)) = stack.pop() {
        let slot = states.get_mut(id as usize)?;
        if let Some(seen) = slot.as_ref() {
            if PositionKey::from_state(seen) != PositionKey::from_state(&state) {
                return None;
            }
            continue;
        }
        *slot = Some(state.clone());
        match cert.nodes.get(id as usize)? {
            CertNode::Choice { mv, child } => {
                let mut next = state;
                let result = apply_placement(&mut next, Placement { coord: *mv }).ok()?;
                if result.outcome.is_some() {
                    return None;
                }
                stack.push((*child, next));
            }
            CertNode::Universal { edges, .. } => {
                for edge in edges {
                    let mut next = state.clone();
                    let result = apply_placement(&mut next, Placement { coord: edge.mv }).ok()?;
                    if result.outcome.is_some() {
                        return None;
                    }
                    stack.push((edge.child, next));
                }
            }
            CertNode::UniversalGroup2V1(node) => {
                for edge in &node.edges {
                    let mut next = state.clone();
                    let result = apply_placement(&mut next, Placement { coord: edge.mv }).ok()?;
                    if result.outcome.is_some() {
                        return None;
                    }
                    stack.push((edge.child, next));
                }
            }
            // Gates are never produced by this solver; fail closed.
            CertNode::FhwGateV1(_) => return None,
            _ => {}
        }
    }
    if states.iter().any(Option::is_none) {
        return None;
    }

    // `compact_certificate` emits nodes in postorder, so every ordinary child
    // precedes its parent. Reconstruct the same D14 recurrence as the
    // independent verifier: factual WIN leaves consume no defender budget,
    // LOSS leaves retain the current turn remainder, Choice passes the budget
    // through, and Universal adds one to the maximum child budget.
    let mut local_budgets = Vec::with_capacity(cert.nodes.len());
    for (index, node) in cert.nodes.iter().enumerate() {
        let local_budget = match node {
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => 0,
            CertNode::Loss { .. } => {
                let state = states.get(index)?.as_ref()?;
                u32::from(threats::placements_remaining(state))
            }
            CertNode::Choice { child, .. } => {
                let child = *child as usize;
                if child >= index {
                    return None;
                }
                *local_budgets.get(child)?
            }
            CertNode::Universal { edges, .. } => {
                let mut maximum = 0u32;
                for edge in edges {
                    let child = edge.child as usize;
                    if child >= index {
                        return None;
                    }
                    maximum = maximum.max(*local_budgets.get(child)?);
                }
                maximum.saturating_add(1)
            }
            CertNode::UniversalGroup2V1(node) => {
                let mut maximum = 0u32;
                for edge in &node.edges {
                    let child = edge.child as usize;
                    if child >= index {
                        return None;
                    }
                    maximum = maximum.max(*local_budgets.get(child)?);
                }
                maximum.saturating_add(1)
            }
            CertNode::FhwGateV1(_) => return None,
        };
        local_budgets.push(local_budget);
    }

    let build_horizon = cert.semantic_horizon;
    for (index, node) in cert.nodes.iter_mut().enumerate() {
        if let CertNode::Universal {
            zone: Some(zone), ..
        } = node
        {
            let Some(&local_budget) = local_budgets.get(index) else {
                return None;
            };
            zone.d = local_budget;
            zone.build_horizon = build_horizon;
        }
    }
    Some(())
}

/// T10/U18 relabelling for an assembled shared DAG. The current Rust grammar
/// serializes D14's local budget but not D15/D16 tables, so the representable
/// max-dominant join is the exact child-max recurrence below. Reachable
/// protected/core obligations remain the union of the final outgoing DAG and
/// are independently reconstructed by `TssVerifier`.
fn rebase_shared_fragment_labels(cert: &mut TssCertificate, root: &RustHexoState) -> Option<u64> {
    fn visit(
        cert: &TssCertificate,
        id: CertNodeId,
        state: &RustHexoState,
        memo: &mut [Option<(PositionKey, u32)>],
        visiting: &mut [bool],
        depth: usize,
    ) -> Option<u32> {
        if depth > MAX_CERT_DEPTH {
            return None;
        }
        let index = id as usize;
        let key = PositionKey::from_state(state);
        if let Some((seen, budget)) = memo.get(index)?.as_ref() {
            return (seen == &key).then_some(*budget);
        }
        if *visiting.get(index)? {
            return None;
        }
        visiting[index] = true;
        let budget = match cert.nodes.get(index)? {
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => 0,
            CertNode::Loss { .. } => u32::from(threats::placements_remaining(state)),
            CertNode::Choice { mv, child } => {
                let mut next = state.clone();
                let result = apply_placement(&mut next, Placement { coord: *mv }).ok()?;
                if result.outcome.is_some() {
                    return None;
                }
                visit(cert, *child, &next, memo, visiting, depth + 1)?
            }
            CertNode::Universal { edges, .. } => {
                let mut maximum = 0u32;
                for edge in edges {
                    let mut next = state.clone();
                    let result = apply_placement(&mut next, Placement { coord: edge.mv }).ok()?;
                    if result.outcome.is_some() {
                        return None;
                    }
                    maximum =
                        maximum.max(visit(cert, edge.child, &next, memo, visiting, depth + 1)?);
                }
                maximum.saturating_add(1)
            }
            // Shared-fragment relabelling never sees extension nodes (they
            // are excluded from fragment promotion); fail closed.
            CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
        };
        visiting[index] = false;
        memo[index] = Some((key, budget));
        Some(budget)
    }

    let mut memo = vec![None; cert.nodes.len()];
    let mut visiting = vec![false; cert.nodes.len()];
    visit(cert, cert.root_node, root, &mut memo, &mut visiting, 0)?;
    if memo.iter().any(Option::is_none) {
        return None;
    }
    let mut relabelled = 0u64;
    for (index, labelled) in memo.into_iter().enumerate() {
        let (_, budget) = labelled?;
        if let CertNode::Universal {
            zone: Some(zone), ..
        } = cert.nodes.get_mut(index)?
        {
            relabelled = relabelled.saturating_add(u64::from(
                zone.d != budget || zone.build_horizon != cert.semantic_horizon,
            ));
            zone.d = budget;
            zone.build_horizon = cert.semantic_horizon;
        }
    }
    Some(relabelled)
}

fn split_tt_cap(total: usize) -> (usize, usize) {
    let shared = total / 2;
    (total - shared, shared)
}

/// Split the post-root wide `Both` allowance while always leaving a nonempty
/// primal allowance when any post-root work is available. This rules out a
/// configuration value silently turning a `Both` solve into loss-only search.
fn wide_both_initial_caps(remaining: u64, loss_reserve_nodes: u32) -> (u64, u64) {
    let reserve = u64::from(loss_reserve_nodes).min(remaining.saturating_sub(1));
    (remaining - reserve, reserve)
}

fn goal_accepts(goal: SolveGoal, status: ProofStatus) -> bool {
    matches!(
        (goal, status),
        (SolveGoal::Both, ProofStatus::Win | ProofStatus::Loss)
            | (SolveGoal::Win, ProofStatus::Win)
            | (SolveGoal::Loss, ProofStatus::Loss)
    )
}

struct AttemptResult {
    cert: Option<TssCertificate>,
    stats: SolveStats,
}

fn unknown<C>(stats: SolveStats) -> DeepResult<C> {
    DeepResult {
        status: ProofStatus::Unknown,
        cert: None,
        stats,
    }
}

fn status_for_claimant(root_player: Player, claimant: Player) -> ProofStatus {
    if root_player == claimant {
        ProofStatus::Win
    } else {
        ProofStatus::Loss
    }
}

/// Return the player proved to win at this node and the corresponding compact
/// leaf.  Lambda-one is intentionally unavailable at Opening (the shared
/// theorem is post-opening), although reachable Opening currently has no
/// threats and therefore cannot produce a verdict anyway.
fn immediate_winner(state: &RustHexoState, width: WidthOptions) -> Option<(Player, CertNode)> {
    if state.is_terminal() {
        return None;
    }
    if matches!(state.phase(), TurnPhase::Opening) {
        return None;
    }
    let analysis = threats::analyze(state);
    let winner = winner_from_analysis(state, &analysis)?;
    typed_lambda_leaf(state, winner, &analysis, width).map(|leaf| (winner, leaf))
}

fn window_key_order(key: WindowKey) -> (u8, i16, i16) {
    (key.axis.index(), key.start.q, key.start.r)
}

const L13_LOSS_WITNESS_CAP_B1: usize = 3;
const L13_LOSS_WITNESS_CAP_B2: usize = 5;

/// Whether no set of at most `budget` cells hits every member of `family`.
/// Connect-6 loss leaves only use budgets one and two; an unsupported budget
/// deliberately returns false so callers cannot emit an unproved witness.
fn family_hitting_exceeds_budget(family: &[Vec<HexCoord>], budget: u8) -> bool {
    if family.is_empty() {
        return false;
    }
    if family.iter().any(Vec::is_empty) {
        return true;
    }

    let mut universe = family.iter().flatten().copied().collect::<Vec<_>>();
    universe.sort_by_key(|coord| (coord.q, coord.r));
    universe.dedup();
    if budget >= 1
        && universe
            .iter()
            .any(|cell| family.iter().all(|set| set.contains(cell)))
    {
        return false;
    }
    if budget >= 2 {
        for left in 0..universe.len() {
            for right in (left + 1)..universe.len() {
                if family
                    .iter()
                    .all(|set| set.contains(&universe[left]) || set.contains(&universe[right]))
                {
                    return false;
                }
            }
        }
    }
    matches!(budget, 1 | 2)
}

/// L13 reverse deletion preserves the earliest canonical family members when
/// either choice is redundant and returns an inclusion-minimal obstruction.
fn inclusion_minimal_loss_obstruction(family: &[Vec<HexCoord>], budget: u8) -> Option<Vec<usize>> {
    let cap = match budget {
        1 => L13_LOSS_WITNESS_CAP_B1,
        2 => L13_LOSS_WITNESS_CAP_B2,
        _ => return None,
    };
    if !family_hitting_exceeds_budget(family, budget) {
        return None;
    }

    let mut kept = (0..family.len()).collect::<Vec<_>>();
    for candidate in (0..family.len()).rev() {
        let trial = kept
            .iter()
            .copied()
            .filter(|index| *index != candidate)
            .map(|index| family[index].clone())
            .collect::<Vec<_>>();
        if family_hitting_exceeds_budget(&trial, budget) {
            kept.retain(|index| *index != candidate);
        }
    }

    debug_assert!(kept.iter().all(|removed| {
        let trial = kept
            .iter()
            .copied()
            .filter(|index| index != removed)
            .map(|index| family[index].clone())
            .collect::<Vec<_>>();
        !family_hitting_exceeds_budget(&trial, budget)
    }));
    if kept.len() > cap {
        return None;
    }
    Some(kept)
}

/// Materialize the sparse obstruction as window identities.  The proved 3/5
/// bounds are checked rather than assumed: an unexpected violation fails
/// closed by declining to materialize a tactical LOSS leaf.
fn sparse_loss_witnesses(
    state: &RustHexoState,
    winner: Player,
    budget: u8,
) -> Option<Vec<WindowKey>> {
    let mut family = state
        .board()
        .windows()
        .threats()
        .filter_map(|(owner, entry)| (owner == winner).then(|| (entry.key(), entry.empty_cells())))
        .collect::<Vec<_>>();
    family.sort_by_key(|(key, _)| window_key_order(*key));
    family.dedup_by_key(|(key, _)| *key);

    let full_sets = family
        .iter()
        .map(|(_, empties)| empties.clone())
        .collect::<Vec<_>>();
    let kept = inclusion_minimal_loss_obstruction(&full_sets, budget)?;
    Some(kept.into_iter().map(|index| family[index].0).collect())
}

fn typed_lambda_leaf(
    state: &RustHexoState,
    winner: Player,
    analysis: &threats::ThreatAnalysis,
    width: WidthOptions,
) -> Option<CertNode> {
    if winner == state.current_player() {
        let mut candidates = state
            .board()
            .windows()
            .entries()
            .filter(|entry| {
                entry.active_player() == Some(winner)
                    && (entry.count(winner) == 5 || (analysis.b == 2 && entry.count(winner) == 4))
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|entry| {
            (
                std::cmp::Reverse(entry.count(winner)),
                window_key_order(entry.key()),
            )
        });
        let witness = candidates.first().copied()?;
        let count = witness.count(winner);
        let extra = if count == 5 { 1 } else { 2 };
        Some(CertNode::Win {
            witness: witness.key(),
            count,
            budget: analysis.b,
            resolution_ply: state.placements_made().saturating_add(extra),
        })
    } else {
        let witnesses = if width.vcf_pair_complete {
            sparse_loss_witnesses(state, winner, analysis.b)?
        } else {
            let mut witnesses = state
                .board()
                .windows()
                .threats()
                .filter_map(|(owner, entry)| (owner == winner).then_some(entry.key()))
                .collect::<Vec<_>>();
            witnesses.sort_by_key(|key| window_key_order(*key));
            witnesses.dedup();
            witnesses
        };
        (!witnesses.is_empty()).then_some(CertNode::Loss {
            witnesses,
            resolution_ply: state
                .placements_made()
                .saturating_add(u32::from(analysis.b))
                .saturating_add(2),
        })
    }
}

fn node_resolution(node: &CertNode) -> u32 {
    match node {
        CertNode::OrCompletion { completion_ply, .. } => *completion_ply,
        CertNode::Win { resolution_ply, .. } | CertNode::Loss { resolution_ply, .. } => {
            *resolution_ply
        }
        CertNode::UniversalGroup2V1(_) => 0,
        // R1: a gate's escape deadline participates in the derived resolution.
        CertNode::FhwGateV1(gate) => gate.proof.escape_resolution_ply,
        CertNode::Choice { .. } | CertNode::Universal { .. } => 0,
    }
}

fn winner_from_analysis(
    state: &RustHexoState,
    analysis: &threats::ThreatAnalysis,
) -> Option<Player> {
    if analysis.own_win_now {
        Some(state.current_player())
    } else if analysis.forced_loss() {
        Some(state.current_player().other())
    } else {
        None
    }
}

const PN_INFINITY: u32 = 1_000_000_000;

/// Default-off live ordering arm for retained attacker-pair children. The
/// numeric values are the public `TSS_ZONE_ORDER` contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ZoneOrderMode {
    #[default]
    Off,
    ZoneBound,
    DStone,
}

impl ZoneOrderMode {
    fn from_env() -> Self {
        match std::env::var("TSS_ZONE_ORDER").ok().as_deref() {
            None | Some("") | Some("0") => Self::Off,
            Some("1") => Self::ZoneBound,
            Some("2") => Self::DStone,
            Some(_) => panic!("TSS_ZONE_ORDER must be one of 0, 1, 2"),
        }
    }

    fn enabled(self) -> bool {
        self != Self::Off
    }

    fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::ZoneBound => "zone_bound",
            Self::DStone => "d_stone",
        }
    }
}

fn zone_order_band_from_env(mode: ZoneOrderMode) -> u32 {
    if !mode.enabled() {
        return 0;
    }
    std::env::var("TSS_ZONE_ORDER_BAND")
        .ok()
        .map(|value| {
            value
                .parse::<u32>()
                .expect("TSS_ZONE_ORDER_BAND must be a nonnegative integer")
        })
        .unwrap_or(0)
}

/// Small conjunctions can exploit ordinary PN re-selection and shared TT work
/// without the multiplicative interleaving measured at the four-way AND
/// frontier. Latch visit-order commitment only once at least four distinct
/// linked proof obligations are live.
const MIN_COMMITTED_UNIVERSAL_OBLIGATIONS: usize = 4;
/// Each placement belongs to at most 18 length-six windows (six starts on
/// each of three axes), so a completed two-stone turn can create at most 36
/// distinct threats.  This geometry bound turns fork degree into a compact,
/// strictly monotone proof prior without a tuned scale.
const MAX_TURN_FORK_DEGREE: u32 = 36;

fn pn_from_fork_degree(fork_degree: usize) -> u32 {
    let fork_degree = u32::try_from(fork_degree)
        .unwrap_or(u32::MAX)
        .min(MAX_TURN_FORK_DEGREE);
    MAX_TURN_FORK_DEGREE + 1 - fork_degree
}

fn dn_from_tau(tau: Option<u8>) -> u32 {
    tau.map(u32::from).unwrap_or(1).max(1)
}

/// No proof can use more placements than either the caller's remaining
/// semantic horizon or the verifier's maximum replay depth.
fn wide_search_final_depth(root_ply: u32, semantic_horizon: u32) -> usize {
    usize::try_from(semantic_horizon.saturating_sub(root_ply))
        .unwrap_or(usize::MAX)
        .min(MAX_SEARCH_DEPTH)
}

/// Advance only to an exact, strictly deeper selected cutoff. Repeated or
/// regressive observations terminate fail-closed instead of spinning.
fn next_wide_stage_depth(
    current_depth: usize,
    encountered_depth: usize,
    final_depth: usize,
) -> Option<usize> {
    let next_depth = encountered_depth.min(final_depth);
    (next_depth > current_depth).then_some(next_depth)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WidePnPrior {
    pn: u32,
    dn: u32,
}

impl WidePnPrior {
    const UNIFORM: Self = Self { pn: 1, dn: 1 };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WidePnKind {
    Choice,
    Universal { implicit_dispatch: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WidePnMove {
    One(HexCoord),
    Pair(HexCoord, HexCoord),
    /// One complete, forced defender turn. This is emitted only by the
    /// wide pair-canonicalization path; unlike `Pair`, it materializes as an
    /// implicit Universal turn with a checked commutation witness.
    DefenderPair(HexCoord, HexCoord),
}

fn compare_hint_weight(left: Option<f32>, right: Option<f32>) -> std::cmp::Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => std::cmp::Ordering::Greater,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn hint_weights_for_move(hints: &OrderingHints, mv: WidePnMove) -> [Option<f32>; 2] {
    let (first, second) = match mv {
        WidePnMove::One(coord) => (hints.weight(coord), None),
        WidePnMove::Pair(first, second) | WidePnMove::DefenderPair(first, second) => {
            (hints.weight(first), hints.weight(second))
        }
    };
    if compare_hint_weight(first, second).is_lt() {
        [second, first]
    } else {
        [first, second]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WidePnChildResult {
    Pending,
    ClaimantCompletion,
    ClaimantTactical,
    Refuted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WidePnStepOutcome {
    Progress,
    DepthCutoff { depth: usize, made_progress: bool },
    Stalled,
}

struct OrderingFeatureContext {
    claimant_stones: Vec<HexCoord>,
}

impl OrderingFeatureContext {
    fn from_state(state: &RustHexoState, claimant: Player, observe_study: bool) -> Self {
        let claimant_stones = state
            .board()
            .occupied_cells()
            .iter()
            .copied()
            .filter(|&coord| state.board().get(coord) == Some(claimant))
            .collect();

        let _ = observe_study;
        Self { claimant_stones }
    }

    fn nearest_claimant_distance(&self, placed: HexCoord) -> u16 {
        self.claimant_stones
            .iter()
            .map(|&stone| hex_distance(placed, stone))
            .min()
            .and_then(|distance| u16::try_from(distance).ok())
            .unwrap_or(u16::MAX)
    }

    fn pair_key(&self, first: HexCoord, second: HexCoord, mode: ZoneOrderMode) -> u16 {
        let first = self.nearest_claimant_distance(first);
        let second = self.nearest_claimant_distance(second);
        match mode {
            ZoneOrderMode::Off => 0,
            ZoneOrderMode::ZoneBound => first.max(second),
            ZoneOrderMode::DStone => first.min(second),
        }
    }
}

#[derive(Clone, Debug)]
struct WidePnChild {
    mv: WidePnMove,
    result: WidePnChildResult,
    entry: Option<usize>,
    /// Exact key retained in lazy mode until the edge links an arena entry.
    /// Defender keys virtually represent the eager entry before selection;
    /// historical attacker-lazy keys remain selection-only.
    future_key: Option<WideFutureKey>,
    /// Static estimates used until the child position is linked. Completed
    /// attacker turns carry both their fork-derived PN and tau-derived DN so
    /// lazy linking cannot erase the principled ordering signal.
    prior: WidePnPrior,
    urgent_block: bool,
    /// Width class of the first placement in an atomic attacker pair.  Zero is
    /// also the neutral value for one-placement and defender children, so the
    /// root-only tier prior cannot perturb their established ordering.
    first_width_tier: u8,
    /// Live R-OS2 distance key. Zero is the inert default used by flag-off and
    /// every non-pair child; selection consults it only when the solve-local
    /// mode is enabled at an attacker Choice.
    zone_order_key: u16,
}

#[derive(Clone, Debug)]
enum WideFutureKey {
    /// Historical attacker lazy edge: the key participates only when selected.
    OnSelection(WidePositionKey),
    /// Attacker-pair selection-only key. The unchanged dense parent body is
    /// shared across siblings and the full key is built only on selection.
    OnSelectionPair {
        template: Arc<WideAttackerPairKeyTemplate>,
        first: HexCoord,
        second: HexCoord,
    },
    /// R-LF1 defender thunk: pre-selection reads virtually observe the eager
    /// entry represented by the deferred key.
    Virtual(WidePositionKey),
}

impl WideFutureKey {
    fn materialize(&self) -> WidePositionKey {
        match self {
            Self::OnSelection(key) | Self::Virtual(key) => key.clone(),
            Self::OnSelectionPair {
                template,
                first,
                second,
            } => template.completed_pair_key(*first, *second),
        }
    }

    fn virtual_key(&self) -> Option<&WidePositionKey> {
        match self {
            Self::Virtual(key) => Some(key),
            Self::OnSelection(_) | Self::OnSelectionPair { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WideChildObligation<'a> {
    Entry(usize),
    FutureKey(&'a WidePositionKey),
}

#[derive(Clone, Copy, Debug)]
struct WideDeferredPosition {
    depth: usize,
    prior: WidePnPrior,
}

fn turn_start_defender_blocks(candidates: &[Candidate]) -> HashSet<HexCoord> {
    candidates
        .iter()
        .filter_map(|candidate| candidate.defender_block.then_some(candidate.coord))
        .collect()
}

fn wide_move_contains_defender_block(mv: WidePnMove, defender_blocks: &HashSet<HexCoord>) -> bool {
    match mv {
        WidePnMove::One(coord) => defender_blocks.contains(&coord),
        WidePnMove::Pair(first, second) | WidePnMove::DefenderPair(first, second) => {
            defender_blocks.contains(&first) || defender_blocks.contains(&second)
        }
    }
}

fn wide_choice_has_urgent_block(children: &[WidePnChild]) -> bool {
    children.iter().any(|child| child.urgent_block)
}

#[derive(Clone, Debug)]
enum WidePnNode {
    Unexpanded,
    ProvenLeaf(CertNode),
    /// Independently verified, exact-key positive proof retained by the
    /// solver-owned cross-solve store. The Arc is immutable for this run.
    ProvenFragment(Arc<ProvenFragment>),
    /// This restricted horizon did not reach a proof. Unlike a genuine
    /// refutation, the node is reopened when the retained search deepens.
    DepthCutoff,
    Refuted,
    Branch {
        kind: WidePnKind,
        children: Vec<WidePnChild>,
    },
}

#[derive(Clone, Debug)]
struct WidePnEntry {
    pn: u32,
    dn: u32,
    /// Immutable initialization restored whenever a staged depth cutoff is
    /// reopened. Recompute may replace the live numbers with child aggregates,
    /// but it never destroys these state-derived priors.
    prior: WidePnPrior,
    node: WidePnNode,
    depth: usize,
    /// Wide-mode visit-order state for an AND node. Once an unresolved
    /// defender obligation is selected, keep driving that same child until it
    /// proves or refutes. This does not participate in PN/DN recomputation or
    /// certificate materialization.
    universal_obligation: Option<usize>,
}

#[derive(Clone, Debug)]
struct WideProvenCandidate {
    id: usize,
    state: RustHexoState,
}

/// Exact, compact key used only by the wide proof-number frontier. Coordinates
/// are zig-zag/varint encoded after sorting `(q,r,owner)` tuples, so equality is
/// collision-free while dense late-game boards do not duplicate a padded
/// `StoneKey` vector in every transposition entry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct WidePositionKey {
    bytes: Box<[u8]>,
}

/// One position's coordinate-sorted occupancy, shared by all prospective
/// pair keys generated from that position. Pair generation can retain many
/// children; sorting the unchanged parent board for every child made dense
/// late-game solves spend substantial time rebuilding identical prefixes.
#[derive(Debug)]
struct WideSortedStones {
    stones: Vec<(i16, i16, u8)>,
    encoded: Vec<u8>,
    offsets: Vec<usize>,
}

/// Shared immutable parent data for selection-only attacker keys. Retained
/// pair edges need only an `Arc` and two coordinates; the exact full key is
/// materialized if and when that edge is selected.
#[derive(Debug)]
struct WideAttackerPairKeyTemplate {
    sorted: WideSortedStones,
    owner: u8,
    next_player: u8,
    placements_after: u32,
}

impl WideAttackerPairKeyTemplate {
    fn from_state(state: &RustHexoState) -> Self {
        debug_assert!(matches!(state.phase(), TurnPhase::FirstStone));
        Self {
            sorted: WideSortedStones::from_state(state),
            owner: player_code(state.current_player()),
            next_player: player_code(state.current_player().other()),
            placements_after: state.placements_made().saturating_add(2),
        }
    }

    fn completed_pair_key(&self, first: HexCoord, second: HexCoord) -> WidePositionKey {
        let mut added = [
            (first.q, first.r, self.owner),
            (second.q, second.r, self.owner),
        ];
        added.sort_unstable();
        let mut encoded = Vec::with_capacity(
            self.sorted
                .stones
                .len()
                .saturating_add(2)
                .saturating_mul(3)
                .saturating_add(12),
        );
        encoded.push(self.next_player);
        push_wide_varint(&mut encoded, self.placements_after);
        encoded.push(1); // TurnPhase::FirstStone after a completed turn.
        encoded.push(0); // Retained Pending pairs are nonterminal.
        self.sorted.append_with_added(&mut encoded, &added);
        WidePositionKey {
            bytes: encoded.into_boxed_slice(),
        }
    }
}

impl WideSortedStones {
    fn from_state(state: &RustHexoState) -> Self {
        let mut stones = state
            .board()
            .occupied_cells()
            .iter()
            .map(|&coord| {
                (
                    coord.q,
                    coord.r,
                    player_code(state.board().get(coord).expect("occupied cell has owner")),
                )
            })
            .collect::<Vec<_>>();
        stones.sort_unstable();
        let mut encoded = Vec::with_capacity(stones.len().saturating_mul(3));
        let mut offsets = Vec::with_capacity(stones.len().saturating_add(1));
        for &(q, r, owner) in &stones {
            offsets.push(encoded.len());
            push_wide_varint(
                &mut encoded,
                zigzag_i16(q).saturating_mul(2) | u32::from(owner),
            );
            push_wide_varint(&mut encoded, zigzag_i16(r));
        }
        offsets.push(encoded.len());
        Self {
            stones,
            encoded,
            offsets,
        }
    }

    /// Append the already-encoded parent occupancy with `added` merged into
    /// the same tuple order used by `WidePositionKey::from_state`.
    fn append_with_added(&self, out: &mut Vec<u8>, added: &[(i16, i16, u8)]) {
        let mut base_start = 0usize;
        for &(q, r, owner) in added {
            // Occupied and newly placed coordinates are disjoint. `<=` also
            // preserves the historical merge's base-before-added tie order.
            let base_end = self.stones.partition_point(|stone| *stone <= (q, r, owner));
            debug_assert!(base_end >= base_start, "added stones must be sorted");
            out.extend_from_slice(&self.encoded[self.offsets[base_start]..self.offsets[base_end]]);
            push_wide_varint(out, zigzag_i16(q).saturating_mul(2) | u32::from(owner));
            push_wide_varint(out, zigzag_i16(r));
            base_start = base_end;
        }
        out.extend_from_slice(&self.encoded[self.offsets[base_start]..]);
    }
}

impl WidePositionKey {
    fn from_state(state: &RustHexoState) -> Self {
        let mut stones = state
            .board()
            .occupied_cells()
            .iter()
            .map(|&coord| {
                (
                    coord.q,
                    coord.r,
                    player_code(state.board().get(coord).expect("occupied cell has owner")),
                )
            })
            .collect::<Vec<_>>();
        stones.sort_unstable();
        let mut encoded = Vec::with_capacity(stones.len().saturating_mul(3).saturating_add(12));
        encoded.push(player_code(state.current_player()));
        push_wide_varint(&mut encoded, state.placements_made());
        match state.phase() {
            TurnPhase::Opening => encoded.push(0),
            TurnPhase::FirstStone => encoded.push(1),
            TurnPhase::SecondStone { first } => {
                encoded.push(2);
                push_wide_varint(&mut encoded, zigzag_i16(first.q));
                push_wide_varint(&mut encoded, zigzag_i16(first.r));
            }
        }
        match state.terminal() {
            None => encoded.push(0),
            Some(outcome) => {
                encoded.push(1 + player_code(outcome.winner));
                push_wide_varint(&mut encoded, outcome.placements);
            }
        }
        for (q, r, owner) in stones {
            push_wide_varint(
                &mut encoded,
                zigzag_i16(q).saturating_mul(2) | u32::from(owner),
            );
            push_wide_varint(&mut encoded, zigzag_i16(r));
        }
        Self {
            bytes: encoded.into_boxed_slice(),
        }
    }

    fn heap_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// The key of the NONTERMINAL claimant FirstStone position reached after
    /// two extra defender placements on `state`, built without touching the
    /// engine. Caller contract (asserted by the defender pair plan before
    /// use): `state` is a forced defender FirstStone node with no live
    /// defender >=4 window, so the pair cannot complete six and the child is
    /// exactly (claimant to move, FirstStone, non-terminal).
    fn for_defender_pair(sorted: &WideSortedStones, claimant: Player, extra: &[HexCoord]) -> Self {
        let defender = claimant.other();
        let owner = player_code(defender);
        let mut added = extra
            .iter()
            .map(|coord| (coord.q, coord.r, owner))
            .collect::<Vec<_>>();
        added.sort_unstable();
        let mut encoded = Vec::with_capacity(
            sorted
                .stones
                .len()
                .saturating_add(added.len())
                .saturating_mul(3)
                .saturating_add(12),
        );
        encoded.push(player_code(claimant));
        push_wide_varint(
            &mut encoded,
            u32::try_from(sorted.stones.len())
                .unwrap_or(u32::MAX)
                .saturating_add(u32::try_from(extra.len()).unwrap_or(0)),
        );
        encoded.push(1); // TurnPhase::FirstStone
        encoded.push(0); // non-terminal
        sorted.append_with_added(&mut encoded, &added);
        Self {
            bytes: encoded.into_boxed_slice(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WideDefenderPair {
    first: HexCoord,
    second: HexCoord,
    final_key: WidePositionKey,
    final_prior: WidePnPrior,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WideDirectedDefenderPair {
    first: HexCoord,
    second: HexCoord,
    final_key: WidePositionKey,
    /// Only the retained raw-low -> raw-high representative pays for the
    /// fork-derived prior. The reverse direction is used solely to validate
    /// exact final-position equality.
    retained_prior: Option<WidePnPrior>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct WideDefenderPairPlan {
    /// The exact K2 kernel, in the ordinary canonical defender order.
    kernel: Vec<HexCoord>,
    /// One raw-coordinate-ordered representative for each symmetric pair.
    pairs: Vec<WideDefenderPair>,
}

/// Derive a complete defender turn at a forced B=2 boundary. The reduction is
/// deliberately all-or-nothing: every K2 first move must be nonterminal, must
/// reach another exact forced boundary with B=1, and every resulting directed
/// pair must have the reverse direction with the identical final position.
/// Any unsupported shape is reported to the caller, which falls back to the
/// ordinary ordered defender expansion rather than dropping a reply.
fn forced_defender_pair_plan_dynamic(
    state: &mut RustHexoState,
    claimant: Player,
) -> Option<WideDefenderPairPlan> {
    if state.current_player() == claimant || !matches!(state.phase(), TurnPhase::FirstStone) {
        return None;
    }
    let root_analysis = threats::analyze(state);
    if root_analysis.b != 2
        || root_analysis.opp_threat_count == 0
        || root_analysis.own_win_now
        || root_analysis.min_hitting_set != Some(2)
    {
        return None;
    }

    let mut kernel = forced_defender_replies(
        state,
        claimant,
        root_analysis.b,
        WidthOptions::vcf_pair_complete(),
    );
    let root_frame = canonical_frame(state);
    kernel.sort_by_key(|coord| canonical_coord_key(root_frame, *coord));
    kernel.dedup();
    if kernel.is_empty() {
        return None;
    }
    let sorted_stones = WideSortedStones::from_state(state);
    let kernel_set = kernel.iter().copied().collect::<HashSet<_>>();

    // One fork scan per plan: a defender pair perturbs the claimant window
    // structure only by two-colouring hit windows, so the plan-root fork
    // degree is a faithful child prior (validated by node-count A/B against
    // the historical per-pair exact scan).
    let shared_fork_pn = pn_from_fork_degree(attacker_fork_degree(state, claimant));

    // Apply/undo on the caller's state instead of cloning the full engine
    // (board + window store) once per kernel cell. Every exit path restores
    // the exact turn-start state.
    let mut directed = Vec::new();
    for &first in &kernel {
        let Ok((first_result, first_delta)) = state.apply_with_delta(Placement { coord: first })
        else {
            return None;
        };
        if first_result.outcome.is_some()
            || state.current_player() == claimant
            || !matches!(state.phase(), TurnPhase::SecondStone { .. })
        {
            state.undo(first_delta);
            return None;
        }

        let analysis = threats::analyze(state);
        if analysis.b != 1
            || analysis.opp_threat_count == 0
            || analysis.own_win_now
            || analysis.min_hitting_set != Some(1)
        {
            state.undo(first_delta);
            return None;
        }
        let mut seconds = forced_defender_replies(
            state,
            claimant,
            analysis.b,
            WidthOptions::vcf_pair_complete(),
        );
        // The plan-root frame is itself D6-covariant, so reusing it keeps the
        // rotation-invariance property while skipping a 12-symmetry stone
        // canonicalization per kernel cell.
        seconds.sort_by_key(|coord| canonical_coord_key(root_frame, *coord));
        seconds.dedup();
        if seconds.is_empty() {
            state.undo(first_delta);
            return None;
        }

        for second in seconds {
            if second == first || !kernel_set.contains(&second) {
                state.undo(first_delta);
                return None;
            }
            // No live defender >=4 window exists at the plan root (checked
            // above via own_win_now), so the pair cannot complete six: the
            // child is exactly (claimant, FirstStone, non-terminal) and its
            // key is constructible without touching the engine. `second` is a
            // kernel threat-window empty, hence always a legal placement.
            let final_key =
                WidePositionKey::for_defender_pair(&sorted_stones, claimant, &[first, second]);
            let retained_prior =
                (raw_coord_key(first) < raw_coord_key(second)).then(|| WidePnPrior {
                    pn: shared_fork_pn,
                    dn: 1,
                });
            directed.push(WideDirectedDefenderPair {
                first,
                second,
                final_key,
                retained_prior,
            });
        }
        state.undo(first_delta);
    }

    let directed_index = directed
        .iter()
        .enumerate()
        .map(|(index, pair)| {
            (
                (raw_coord_key(pair.first), raw_coord_key(pair.second)),
                index,
            )
        })
        .collect::<HashMap<_, _>>();
    if directed_index.len() != directed.len() {
        return None;
    }
    for pair in &directed {
        let reverse = directed_index
            .get(&(raw_coord_key(pair.second), raw_coord_key(pair.first)))
            .and_then(|&index| directed.get(index))?;
        if reverse.final_key != pair.final_key {
            return None;
        }
    }

    let kernel_rank = kernel
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, coord)| (coord, rank))
        .collect::<HashMap<_, _>>();
    let mut pairs = directed
        .into_iter()
        .filter(|pair| raw_coord_key(pair.first) < raw_coord_key(pair.second))
        .map(|pair| {
            Some(WideDefenderPair {
                first: pair.first,
                second: pair.second,
                final_key: pair.final_key,
                final_prior: pair.retained_prior?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    pairs.sort_by_key(|pair| {
        let first_rank = kernel_rank[&pair.first];
        let second_rank = kernel_rank[&pair.second];
        (first_rank.min(second_rank), first_rank.max(second_rank))
    });
    if pairs.is_empty() {
        return None;
    }
    Some(WideDefenderPairPlan { kernel, pairs })
}

/// Stateless K2 plan for the exact rank-two boundary licensed in
/// `RESEARCH_DIVERGENCE_1.md` section 3. The guards are intentionally
/// redundant with the cover construction: if the live opponent family is
/// not nonempty, rank <= 2, tau=2, and free of a defender win-now, the caller
/// falls back to the dynamic reference above.
fn forced_defender_pair_plan_direct(
    state: &RustHexoState,
    claimant: Player,
) -> Option<WideDefenderPairPlan> {
    if state.current_player() == claimant || !matches!(state.phase(), TurnPhase::FirstStone) {
        return None;
    }
    let analysis = threats::analyze(state);
    if analysis.b != 2
        || analysis.opp_threat_count == 0
        || analysis.own_win_now
        || analysis.min_hitting_set != Some(2)
    {
        return None;
    }

    let family = state
        .board()
        .windows()
        .threats()
        .filter_map(|(owner, entry)| (owner == claimant).then(|| entry.empty_cells()))
        .collect::<Vec<_>>();
    if family.is_empty() || family.iter().any(|edge| !(1..=2).contains(&edge.len())) {
        return None;
    }

    // Preserve the reference kernel's canonical order exactly. Membership is
    // the union of the directly enumerated minimum covers; presentation order
    // remains independent of the cover enumeration below.
    let mut kernel = extendable_hit_kernel_for_family(&family, 2);
    let root_frame = canonical_frame(state);
    kernel.sort_by_key(|coord| canonical_coord_key(root_frame, *coord));
    kernel.dedup();
    if kernel.is_empty() || kernel.len() > 4 {
        return None;
    }

    let shared_prior = WidePnPrior {
        pn: pn_from_fork_degree(attacker_fork_degree(state, claimant)),
        dn: 1,
    };
    let sorted_stones = WideSortedStones::from_state(state);
    let mut pairs = Vec::new();
    for left in 0..kernel.len() {
        for right in (left + 1)..kernel.len() {
            let x = kernel[left];
            let y = kernel[right];
            if !family
                .iter()
                .all(|edge| edge.contains(&x) || edge.contains(&y))
            {
                continue;
            }
            let (first, second) = if raw_coord_key(x) < raw_coord_key(y) {
                (x, y)
            } else {
                (y, x)
            };
            pairs.push(WideDefenderPair {
                first,
                second,
                final_key: WidePositionKey::for_defender_pair(
                    &sorted_stones,
                    claimant,
                    &[first, second],
                ),
                final_prior: shared_prior,
            });
        }
    }
    if pairs.is_empty() || pairs.len() > 4 {
        return None;
    }
    let kernel_rank = kernel
        .iter()
        .copied()
        .enumerate()
        .map(|(rank, coord)| (coord, rank))
        .collect::<HashMap<_, _>>();
    pairs.sort_by_key(|pair| {
        let first_rank = kernel_rank[&pair.first];
        let second_rank = kernel_rank[&pair.second];
        (first_rank.min(second_rank), first_rank.max(second_rank))
    });
    Some(WideDefenderPairPlan { kernel, pairs })
}

fn forced_defender_pair_plan(
    state: &mut RustHexoState,
    claimant: Player,
) -> Option<WideDefenderPairPlan> {
    if let Some(direct) = forced_defender_pair_plan_direct(state, claimant) {
        #[cfg(debug_assertions)]
        {
            let reference = forced_defender_pair_plan_dynamic(state, claimant);
            assert_eq!(
                reference.as_ref(),
                Some(&direct),
                "stateless K2 plan diverged from the dynamic reference"
            );
        }
        return Some(direct);
    }
    forced_defender_pair_plan_dynamic(state, claimant)
}

/// Conservative retained-byte charge for one exact-key TT index entry.  PN
/// nodes and their child vectors live in the node-capped search arena and are
/// deliberately excluded from the caller's TT/cache byte ceiling.
fn wide_position_index_bytes(key: &WidePositionKey) -> usize {
    key.heap_bytes()
        .saturating_add(size_of::<(WidePositionKey, usize)>())
        .saturating_add(ALLOC_OVERHEAD)
}

fn zigzag_i16(value: i16) -> u32 {
    let value = i32::from(value);
    u32::try_from((value << 1) ^ (value >> 31)).expect("i16 zig-zag is nonnegative")
}

fn push_wide_varint(out: &mut Vec<u8>, mut value: u32) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Wide VCF search keeps a persistent proof-number frontier.  Unlike the
/// quota-based DFS experiments, expanding a sibling never discards work in an
/// earlier forcing turn. Claimant pairs are represented as one OR edge, so
/// turn-forcing is structural rather than an after-the-fact recursive filter.
struct WidePnSearch<'store> {
    claimant: Player,
    root_ply: u32,
    node_cap: u64,
    tt_bytes_cap: usize,
    semantic_horizon: u32,
    depth_cap: usize,
    /// Final solve depth; staged deepening mutates `depth_cap` below.
    max_depth_cap: usize,
    width: WidthOptions,
    /// Solve-local, read-once R-OS2 ordering configuration. Off takes the
    /// historical selector branch without computing or consulting a key.
    zone_order_mode: ZoneOrderMode,
    zone_order_band: u32,
    /// Optional coordinate priors copied from the owning solve. They are read
    /// only after a node's complete historical child vector has been built.
    ordering_hints: Option<OrderingHints>,
    /// Read once when this solve-local search is created. Default-off keeps the
    /// historical eager defender-frontier admission path byte-for-byte in the
    /// decision logic.
    lazy_frontier: bool,
    /// Still-live lines refused by the semantic-horizon deadline this search
    /// (SolveStats::horizon_cuts), and the defender-to-move subset
    /// (SolveStats::kb_death_cuts).
    horizon_cuts: u64,
    kb_death_cuts: u64,
    interior_census_gate: bool,
    interior_gate_evaluations: u64,
    interior_gate_dismissals: u64,
    interior_gate_nanos: u64,
    expansions: u64,
    tt_hits: u64,
    current_bytes: usize,
    peak_bytes: usize,
    /// Expansion ceiling for one `work` invocation. The production driver
    /// leaves it open (`u64::MAX`); the historical single-expansion `step`
    /// wrapper sets it to `expansions + 1` so the focused stepper tests keep
    /// their one-expansion-per-call contract.
    soft_expansion_cap: u64,

    entries: Vec<WidePnEntry>,
    by_position: HashMap<WidePositionKey, usize>,
    /// Exact prospective identity for lazy defender thunks. This preserves the
    /// first eager admission's prior/depth and lets a selected attacker thunk
    /// recover that transposed state without pre-linking an arena/TT entry.
    deferred_by_position: HashMap<WidePositionKey, WideDeferredPosition>,
    /// Exact window-substructure cache shared by attacker OR generations in
    /// this search. Keys include both occupancy masks, so a position delta
    /// invalidates precisely the windows it changes.
    generation_memo: RefCell<WindowGenerationMemo>,
    fragment_store: Option<&'store ProvenFragmentStore>,
    fragment_lookups: u64,
    fragment_hits: u64,
    proven_candidate_ids: HashSet<usize>,
    proven_candidates: Vec<WideProvenCandidate>,
}

impl<'store> WidePnSearch<'store> {
    /// C1 migration mode: preserve the narrow DFS's recursive expansion order
    /// while entering through the wide engine seam.  Reusing the PN frontier
    /// here would change node counts even when it found the same proof, which
    /// is outside the owner-approved identity contract.
    #[allow(clippy::too_many_arguments)]
    fn prove_narrow_compat(
        state: &RustHexoState,
        claimant: Player,
        node_cap: u64,
        local_tt_cap: usize,
        hash_mask: u64,
        shared_tt: &mut SharedProofCache,
        semantic_horizon: u32,
        zone: ZoneSearchCaps,
        width: WidthOptions,
        depth_cap: usize,
        k_reply_consume: bool,

        interior_census_gate: bool,
        group2: bool,
    ) -> AttemptResult {
        debug_assert!(
            !width.vcf_pair_complete
                || (width.consumes_quiet_turns() && width.consumes_ranked_zone())
        );

        let mut work = state.clone();
        let entry_key = PositionKey::from_state(&work);
        let root_ply = state.placements_made();
        let mut search = NarrowCompatSearch::with_shared(
            node_cap,
            local_tt_cap,
            hash_mask,
            &mut *shared_tt,
            root_ply,
            semantic_horizon,
            zone,
            width,
            depth_cap,
            k_reply_consume,
            interior_census_gate,
            group2,
        );
        let proof = search.prove(&mut work, claimant, root_ply, None);

        debug_assert_eq!(entry_key, PositionKey::from_state(&work));

        let cert = proof.and_then(|root| {
            let (nodes, root_node) = compact_certificate(&search.arena, root)?;
            if search.can_admit_compact(&entry_key, &nodes) {
                if let Some(cached) = CachedProof::from_compact(nodes.clone(), root_node) {
                    search.insert_shared(entry_key.clone(), claimant, cached);
                }
            }
            let mut cert = TssCertificate {
                root: RootBinding::from_state(state),
                claimant,
                root_node,
                nodes,
                semantic_horizon,
            };
            if cert.nodes.iter().any(CertNode::is_group2_extension) {
                // v1 Group-2 finalization: canonical order, strict-tree
                // unfolding, derived scalars, and both digests — then a
                // strict self-verification under the extension policy. Any
                // failure drops the certificate; the clean re-solve below
                // restores flag-off behavior.
                let finalized = crate::tss_verify_group2::finder_finalize_group2(state, &cert)?;
                let claimed = status_for_claimant(state.current_player(), claimant);
                if !crate::tss_core::CertVerify::verify(
                    &crate::tss_verify::Group2Verifier,
                    state,
                    &finalized,
                    claimed,
                ) {
                    return None;
                }
                cert = finalized;
            } else {
                rebase_zone_distances(&mut cert, state)?;
            }
            Some(cert)
        });
        let stats = SolveStats {
            nodes: search.nodes,
            expansions: search.nodes,
            tt_hits: search.tt_hits,
            tt_entries: search.tt_entry_count() as u64,
            peak_tt_bytes: search.peak_tt_bytes as u64,
            tt_evictions: search.tt.replacements,
            tt_admission_rejections: search.tt.refusals,
            interior_gate_evaluations: search.interior_gate_evaluations,
            interior_gate_dismissals: search.interior_gate_dismissals,
            interior_gate_nanos: search.interior_gate_nanos,
            horizon_cuts: search.horizon_cuts,
            kb_death_cuts: search.kb_death_cuts,
            ..SolveStats::default()
        };

        drop(search);
        if group2 && cert.is_none() {
            // Fail-safe re-solve with the selector off: the flag must never
            // decide fewer positions than flag-off. Costs are summed.
            let rerun = Self::prove_narrow_compat(
                state,
                claimant,
                node_cap,
                local_tt_cap,
                hash_mask,
                shared_tt,
                semantic_horizon,
                zone,
                width,
                depth_cap,
                k_reply_consume,
                interior_census_gate,
                false,
            );
            let mut merged = stats;
            merged.merge(rerun.stats);
            return AttemptResult {
                cert: rerun.cert,
                stats: merged,
            };
        }
        AttemptResult { cert, stats }
    }

    fn new(
        claimant: Player,
        root_ply: u32,
        node_cap: u64,
        tt_bytes_cap: usize,
        semantic_horizon: u32,
        depth_cap: usize,
    ) -> Self {
        Self::new_with_width(
            claimant,
            root_ply,
            node_cap,
            tt_bytes_cap,
            semantic_horizon,
            depth_cap,
            WidthOptions::vcf_pair_complete(),
            None,
        )
    }

    fn new_with_width(
        claimant: Player,
        root_ply: u32,
        node_cap: u64,
        tt_bytes_cap: usize,
        semantic_horizon: u32,
        depth_cap: usize,
        width: WidthOptions,
        fragment_store: Option<&'store ProvenFragmentStore>,
    ) -> Self {
        let lazy_frontier = std::env::var("TSS_LAZY_FRONTIER").ok().as_deref() == Some("1");
        let zone_order_mode = ZoneOrderMode::from_env();
        let zone_order_band = zone_order_band_from_env(zone_order_mode);
        Self {
            claimant,
            root_ply,
            node_cap,
            tt_bytes_cap,
            semantic_horizon,
            depth_cap,
            max_depth_cap: depth_cap,
            width,
            zone_order_mode,
            zone_order_band,
            ordering_hints: None,
            lazy_frontier,
            horizon_cuts: 0,
            kb_death_cuts: 0,
            interior_census_gate: false,
            interior_gate_evaluations: 0,
            interior_gate_dismissals: 0,
            interior_gate_nanos: 0,
            expansions: 0,
            tt_hits: 0,
            current_bytes: 0,
            peak_bytes: 0,
            soft_expansion_cap: u64::MAX,

            entries: Vec::new(),
            by_position: HashMap::new(),
            deferred_by_position: HashMap::new(),
            generation_memo: RefCell::new(WindowGenerationMemo::default()),
            fragment_store,
            fragment_lookups: 0,
            fragment_hits: 0,
            proven_candidate_ids: HashSet::new(),
            proven_candidates: Vec::new(),
        }
    }

    fn remember_proven_candidate(&mut self, state: &RustHexoState, id: usize) {
        if self.fragment_store.is_none()
            || self.proven_candidates.len() >= MAX_WIDE_FRAGMENT_PROMOTIONS
            || self.proven_candidate_ids.contains(&id)
            || !matches!(self.entries[id].node, WidePnNode::Branch { .. })
        {
            return;
        }
        self.proven_candidate_ids.insert(id);
        self.proven_candidates.push(WideProvenCandidate {
            id,
            state: state.clone(),
        });
    }

    fn insert_root(&mut self, state: &RustHexoState) -> usize {
        let prior = self.position_prior(state);
        self.insert_position(WidePositionKey::from_state(state), 0, prior)
    }

    fn insert_position(&mut self, key: WidePositionKey, depth: usize, prior: WidePnPrior) -> usize {
        if let Some(&id) = self.by_position.get(&key) {
            self.tt_hits = self.tt_hits.saturating_add(1);
            return id;
        }
        let deferred = self
            .lazy_frontier
            .then(|| self.deferred_by_position.remove(&key))
            .flatten();
        let (depth, prior) = deferred
            .map(|deferred| (deferred.depth, deferred.prior))
            .unwrap_or((depth, prior));

        // The retained PN frontier is the search arena, not the transposition
        // index.  A full (or disabled) TT must only stop indexing new keys;
        // refusing the arena entry would strand the selected Pending edge and
        // make a memory-profile choice alter frontier progress.
        let id = self.entries.len();

        self.entries.push(WidePnEntry {
            pn: prior.pn,
            dn: prior.dn,
            prior,
            node: WidePnNode::Unexpanded,
            depth,
            universal_obligation: None,
        });

        let added = wide_position_index_bytes(&key);
        if self.tt_bytes_cap > 0 && self.current_bytes.saturating_add(added) <= self.tt_bytes_cap {
            self.by_position.insert(key, id);
            self.current_bytes = self.current_bytes.saturating_add(added);
            self.peak_bytes = self.peak_bytes.max(self.current_bytes);
        } else {
        }
        id
    }

    fn defer_position(&mut self, key: &WidePositionKey, depth: usize, prior: WidePnPrior) {
        if self.by_position.contains_key(key) {
            return;
        }
        self.deferred_by_position
            .entry(key.clone())
            .or_insert(WideDeferredPosition { depth, prior });
    }

    fn position_prior(&self, state: &RustHexoState) -> WidePnPrior {
        if state.current_player() == self.claimant {
            let pn = pn_from_fork_degree(attacker_fork_degree(state, self.claimant));

            WidePnPrior { pn, dn: 1 }
        } else {
            let analysis = threats::analyze(state);

            let pn = 1;
            WidePnPrior {
                pn,
                dn: dn_from_tau(analysis.min_hitting_set),
            }
        }
    }

    /// The immutable first-placement width class is a root bootstrap only.
    /// Persisting it below depth zero regresses otherwise closed positions by
    /// overriding accumulated proof-number evidence.
    fn prefer_width_tier_at_depth(&self, depth: usize) -> bool {
        depth == 0
    }

    fn completed_turn_prior(&self, state: &RustHexoState) -> WidePnPrior {
        debug_assert_ne!(state.current_player(), self.claimant);
        let analysis = threats::analyze(state);
        let pn = pn_from_fork_degree(analysis.opp_threat_count);

        WidePnPrior {
            pn,
            dn: dn_from_tau(analysis.min_hitting_set),
        }
    }

    fn run(&mut self, root_state: &RustHexoState, root: usize) {
        let final_depth = self.depth_cap;
        let mut stage_depth = 0usize;

        // The selected PN path discovers the next useful horizon. Every stage
        // shares the caller's one global node cap; there are no scouting quotas.
        loop {
            self.depth_cap = stage_depth;
            self.reopen_depth_cutoffs(stage_depth);
            let is_final = stage_depth == final_depth;
            let selected_cutoff = self.run_until(root_state, root, self.node_cap, !is_final);
            // Transposed parents outside the active recursion also need to see
            // the selected cutoff (or proof) before the stage decision.
            self.refresh_all_bottom_up();

            if self.entries[root].pn == 0 || self.expansions >= self.node_cap || is_final {
                break;
            }
            let Some(encountered_depth) = selected_cutoff else {
                break;
            };
            let Some(next_depth) =
                next_wide_stage_depth(stage_depth, encountered_depth, final_depth)
            else {
                break;
            };
            stage_depth = next_depth;
        }

        self.depth_cap = final_depth;
    }

    fn run_until(
        &mut self,
        root_state: &RustHexoState,
        root: usize,
        expansion_cap: u64,
        deepen_after_selected_cutoff: bool,
    ) -> Option<usize> {
        let mut work = root_state.clone();
        while self.expansions < self.node_cap && self.expansions < expansion_cap {
            self.recompute(root);
            let Some(entry) = self.entries.get(root) else {
                break;
            };
            if entry.pn == 0 || entry.dn == 0 {
                break;
            }
            match self.work(&mut work, root, false, u32::MAX, u32::MAX) {
                WidePnStepOutcome::Progress => {}
                WidePnStepOutcome::DepthCutoff { depth, .. } if deepen_after_selected_cutoff => {
                    return Some(depth);
                }
                WidePnStepOutcome::DepthCutoff {
                    made_progress: true,
                    ..
                } => {}
                WidePnStepOutcome::DepthCutoff {
                    made_progress: false,
                    ..
                } => {
                    break;
                }
                WidePnStepOutcome::Stalled => {
                    break;
                }
            }
        }
        None
    }

    fn reopen_depth_cutoffs(&mut self, depth_cap: usize) {
        let reopened = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(id, entry)| {
                (entry.depth <= depth_cap && matches!(entry.node, WidePnNode::DepthCutoff))
                    .then_some(id)
            })
            .collect::<Vec<_>>();
        for &id in &reopened {
            self.entries[id].node = WidePnNode::Unexpanded;
        }
        if reopened.is_empty() {
            return;
        }

        // Edges always add placements, so a single deepest-first pass
        // propagates the reopened frontier through every (possibly
        // transposed) parent without retaining reverse-parent vectors in each
        // entry.  Refreshing only the cutoff entries leaves an ancestor's
        // cached dn=0 in place and can make the next depth stage stop before
        // doing any work.
        self.refresh_all_bottom_up();
    }

    fn refresh_all_bottom_up(&mut self) {
        let mut ids = (0..self.entries.len()).collect::<Vec<_>>();
        ids.sort_unstable_by_key(|&id| std::cmp::Reverse(self.entries[id].depth));
        for id in ids {
            self.recompute(id);
        }
    }

    fn step(&mut self, state: &mut RustHexoState, id: usize) -> WidePnStepOutcome {
        // Historical single-expansion stepper, preserved for the focused
        // stepper tests. The production driver calls `work` directly with
        // open thresholds and no soft cap.
        self.soft_expansion_cap = self.expansions.saturating_add(1);
        let outcome = self.work(state, id, false, u32::MAX, u32::MAX);
        self.soft_expansion_cap = u64::MAX;
        outcome
    }

    /// Threshold-bounded proof-number descent (df-pn scheduling). The node
    /// keeps driving its selected child while its own numbers stay below the
    /// caller-supplied thresholds, so consecutive expansions land at the
    /// frontier without re-descending from the root. Thresholds bound VISIT
    /// ORDER only: pn/dn recurrences, expansion, refutation marking, and
    /// certificate materialization are untouched, so proofs are unchanged.
    ///
    /// Child thresholds follow the standard df-pn recurrence (min against the
    /// second-best sibling plus one; budget subtraction on the conjunctive
    /// side), floored at the child's current number plus one so a
    /// policy-selected child (urgency, width tier, sequential probe,
    /// commitment) can always make local progress before control unwinds.
    fn work(
        &mut self,
        state: &mut RustHexoState,
        id: usize,
        inherited_commitment: bool,
        pn_threshold: u32,
        dn_threshold: u32,
    ) -> WidePnStepOutcome {
        let mut any_progress = false;
        let mut yielded_universal_children = Vec::new();
        loop {
            if matches!(self.entries[id].node, WidePnNode::DepthCutoff) {
                return WidePnStepOutcome::DepthCutoff {
                    depth: self.entries[id].depth,
                    made_progress: any_progress,
                };
            }
            if matches!(self.entries[id].node, WidePnNode::Unexpanded) {
                let expansion_outcome = self.expand(state, id);

                match expansion_outcome {
                    WidePnStepOutcome::Progress => {
                        any_progress = true;
                        if !matches!(self.entries[id].node, WidePnNode::Branch { .. }) {
                            return WidePnStepOutcome::Progress;
                        }
                    }
                    other => return other,
                }
            }
            self.recompute(id);
            if self.entries[id].pn == 0 || self.entries[id].dn == 0 {
                if self.entries[id].pn == 0 {
                    self.remember_proven_candidate(state, id);
                }
                return if any_progress {
                    WidePnStepOutcome::Progress
                } else {
                    WidePnStepOutcome::Stalled
                };
            }
            if self.entries[id].pn >= pn_threshold || self.entries[id].dn >= dn_threshold {
                // Thresholds crossed: the parent re-decides. Any expansion or
                // refutation made here already counts as progress.

                return WidePnStepOutcome::Progress;
            }
            if self.expansions >= self.node_cap || self.expansions >= self.soft_expansion_cap {
                return if any_progress {
                    WidePnStepOutcome::Progress
                } else {
                    WidePnStepOutcome::Stalled
                };
            }

            let finish_partial_turn = matches!(state.phase(), TurnPhase::SecondStone { .. });
            let urgent_pair = matches!(state.phase(), TurnPhase::FirstStone)
                && matches!(
                    &self.entries[id].node,
                    WidePnNode::Branch {
                        kind: WidePnKind::Choice,
                        children,
                    } if wide_choice_has_urgent_block(children)
                );
            // Sequential probing is a root bootstrap for the two corpus shapes
            // that enter mid-turn or under an urgent block.  Applying it at
            // every descendant discards the proof-number evidence and
            // degenerates into depth-first search inside each forcing branch.
            let sequential_root_probe =
                self.entries[id].depth == 0 && (finish_partial_turn || urgent_pair);
            let prefer_width_tier = self.prefer_width_tier_at_depth(self.entries[id].depth);
            let commitment_domain = inherited_commitment
                || match &self.entries[id].node {
                    WidePnNode::Branch {
                        kind: WidePnKind::Universal { .. },
                        children,
                    } => self.universal_commitment_active(id, children),
                    _ => false,
                };

            let parent_before = (self.entries[id].pn, self.entries[id].dn);
            let selected = self.select_step_child_index_with_commitment(
                id,
                sequential_root_probe,
                prefer_width_tier,
                &yielded_universal_children,
                commitment_domain,
            );
            let Some(child_index) = selected else {
                return if any_progress {
                    WidePnStepOutcome::Progress
                } else {
                    WidePnStepOutcome::Stalled
                };
            };

            let (kind, child, child_pn_threshold, child_dn_threshold, _root_children_unlinked) = {
                let WidePnNode::Branch { kind, children } = &self.entries[id].node else {
                    return WidePnStepOutcome::Stalled;
                };
                let (child_pn, child_dn) = self.child_numbers(&children[child_index]);

                let (child_pn_threshold, child_dn_threshold) = match kind {
                    WidePnKind::Choice => {
                        let mut second_pn = u32::MAX;
                        for (rank, other) in children.iter().enumerate() {
                            if rank != child_index {
                                second_pn = second_pn.min(self.child_numbers(other).0);
                            }
                        }

                        let second_pn_limit = second_pn.saturating_add(1);

                        let child_pn_floor = child_pn.saturating_add(1);

                        let child_dn_floor = child_dn.saturating_add(1);
                        let pn_t = pn_threshold.min(second_pn_limit).max(child_pn_floor);
                        let dn_t = dn_threshold
                            .saturating_sub(self.entries[id].dn.saturating_sub(child_dn))
                            .max(child_dn_floor);
                        (pn_t, dn_t)
                    }
                    WidePnKind::Universal { .. } => {
                        let committed = self.entries[id].universal_obligation == Some(child_index);

                        let child_dn_floor = child_dn.saturating_add(1);
                        let dn_t = if committed {
                            // Commitment domains drive the obligation to a
                            // verdict; sibling DN must not unseat it.
                            dn_threshold.max(child_dn_floor)
                        } else {
                            let mut second_dn = u32::MAX;
                            for (rank, other) in children.iter().enumerate() {
                                if rank != child_index {
                                    second_dn = second_dn.min(self.child_numbers(other).1);
                                }
                            }

                            let second_dn_limit = second_dn.saturating_add(1);
                            dn_threshold.min(second_dn_limit).max(child_dn_floor)
                        };

                        let child_pn_floor = child_pn.saturating_add(1);
                        let pn_t = pn_threshold
                            .saturating_sub(self.entries[id].pn.saturating_sub(child_pn))
                            .max(child_pn_floor);
                        (pn_t, dn_t)
                    }
                };

                (
                    *kind,
                    children[child_index].clone(),
                    child_pn_threshold,
                    child_dn_threshold,
                    children.iter().all(|child| child.entry.is_none()),
                )
            };

            if child.result != WidePnChildResult::Pending {
                self.recompute(id);
                return if any_progress {
                    WidePnStepOutcome::Progress
                } else {
                    WidePnStepOutcome::Stalled
                };
            }

            let outcome = match child.mv {
                WidePnMove::One(coord) => {
                    // Historical attacker edges count first linking as local
                    // progress. A key-bearing defender thunk refines an eager
                    // edge whose arena link already existed, so admission must
                    // not add a progress event that eager never reported.
                    let linked = child.entry.is_none() && matches!(kind, WidePnKind::Choice);

                    let applied = state.apply_with_delta(Placement { coord });

                    let Ok((_result, delta)) = applied else {
                        self.set_child_refuted(id, child_index);
                        self.refresh(id);
                        any_progress = true;
                        continue;
                    };
                    let child_id = child.entry.unwrap_or_else(|| {
                        let depth =
                            usize::try_from(state.placements_made().saturating_sub(self.root_ply))
                                .unwrap_or(usize::MAX);
                        let key = child
                            .future_key
                            .as_ref()
                            .map(WideFutureKey::materialize)
                            .unwrap_or_else(|| WidePositionKey::from_state(state));
                        debug_assert_eq!(key, WidePositionKey::from_state(state));

                        self.insert_position(key, depth, child.prior)
                    });
                    self.set_child_entry(id, child_index, child_id);

                    let outcome = self.work(
                        state,
                        child_id,
                        commitment_domain,
                        child_pn_threshold,
                        child_dn_threshold,
                    );

                    state.undo(delta);

                    match outcome {
                        WidePnStepOutcome::DepthCutoff {
                            depth,
                            made_progress,
                        } => WidePnStepOutcome::DepthCutoff {
                            depth,
                            made_progress: made_progress || linked,
                        },
                        WidePnStepOutcome::Progress => WidePnStepOutcome::Progress,
                        WidePnStepOutcome::Stalled if linked => WidePnStepOutcome::Progress,
                        WidePnStepOutcome::Stalled => WidePnStepOutcome::Stalled,
                    }
                }
                WidePnMove::Pair(first, second) | WidePnMove::DefenderPair(first, second) => {
                    let linked = child.entry.is_none() && matches!(kind, WidePnKind::Choice);

                    let first_applied = state.apply_with_delta(Placement { coord: first });

                    let Ok((_first_result, first_delta)) = first_applied else {
                        self.set_child_refuted(id, child_index);
                        self.refresh(id);
                        any_progress = true;
                        continue;
                    };

                    let second_applied = state.apply_with_delta(Placement { coord: second });

                    let Ok((_second_result, second_delta)) = second_applied else {
                        state.undo(first_delta);

                        self.set_child_refuted(id, child_index);
                        self.refresh(id);
                        any_progress = true;
                        continue;
                    };
                    let child_id = child.entry.unwrap_or_else(|| {
                        let depth =
                            usize::try_from(state.placements_made().saturating_sub(self.root_ply))
                                .unwrap_or(usize::MAX);
                        let key = child
                            .future_key
                            .as_ref()
                            .map(WideFutureKey::materialize)
                            .unwrap_or_else(|| WidePositionKey::from_state(state));
                        debug_assert_eq!(key, WidePositionKey::from_state(state));

                        self.insert_position(key, depth, child.prior)
                    });
                    self.set_child_entry(id, child_index, child_id);

                    let outcome = self.work(
                        state,
                        child_id,
                        commitment_domain,
                        child_pn_threshold,
                        child_dn_threshold,
                    );

                    state.undo(second_delta);
                    state.undo(first_delta);

                    match outcome {
                        WidePnStepOutcome::DepthCutoff {
                            depth,
                            made_progress,
                        } => WidePnStepOutcome::DepthCutoff {
                            depth,
                            made_progress: made_progress || linked,
                        },
                        WidePnStepOutcome::Progress => WidePnStepOutcome::Progress,
                        WidePnStepOutcome::Stalled if linked => WidePnStepOutcome::Progress,
                        WidePnStepOutcome::Stalled => WidePnStepOutcome::Stalled,
                    }
                }
            };
            self.refresh(id);
            let parent_changed = parent_before != (self.entries[id].pn, self.entries[id].dn);
            let outcome = match outcome {
                WidePnStepOutcome::DepthCutoff {
                    depth,
                    made_progress,
                } => WidePnStepOutcome::DepthCutoff {
                    depth,
                    made_progress: made_progress || parent_changed,
                },
                WidePnStepOutcome::Progress => WidePnStepOutcome::Progress,
                WidePnStepOutcome::Stalled if parent_changed => WidePnStepOutcome::Progress,
                WidePnStepOutcome::Stalled => WidePnStepOutcome::Stalled,
            };
            match outcome {
                WidePnStepOutcome::DepthCutoff {
                    depth,
                    made_progress,
                } => {
                    // Depth cutoffs bubble to the stage driver unchanged so
                    // staged deepening keeps its advance-on-selected-cutoff
                    // semantics.
                    return WidePnStepOutcome::DepthCutoff {
                        depth,
                        made_progress: made_progress || any_progress,
                    };
                }
                WidePnStepOutcome::Progress => {
                    any_progress = true;
                }
                WidePnStepOutcome::Stalled => {
                    if matches!(kind, WidePnKind::Universal { .. })
                        && self.entries[id].universal_obligation == Some(child_index)
                        && self.expansions < self.node_cap
                    {
                        yielded_universal_children.push(child_index);
                        continue;
                    }
                    return WidePnStepOutcome::Stalled;
                }
            }
        }
    }

    fn set_child_entry(&mut self, parent: usize, child: usize, entry: usize) {
        if let WidePnNode::Branch { children, .. } = &mut self.entries[parent].node {
            children[child].entry = Some(entry);
            children[child].future_key = None;
        }
    }

    fn set_child_refuted(&mut self, parent: usize, child: usize) {
        if let WidePnNode::Branch { children, .. } = &mut self.entries[parent].node {
            children[child].result = WidePnChildResult::Refuted;
            children[child].future_key = None;
        }
    }

    fn child_numbers(&self, child: &WidePnChild) -> (u32, u32) {
        match child.result {
            WidePnChildResult::ClaimantCompletion | WidePnChildResult::ClaimantTactical => {
                (0, PN_INFINITY)
            }
            WidePnChildResult::Refuted => (PN_INFINITY, 0),
            WidePnChildResult::Pending => self
                .resolved_child_entry(child)
                .and_then(|id| self.entries.get(id))
                .map(|entry| (entry.pn, entry.dn))
                .or_else(|| {
                    child
                        .future_key
                        .as_ref()
                        .and_then(WideFutureKey::virtual_key)
                        .and_then(|key| self.deferred_by_position.get(key))
                        .map(|deferred| (deferred.prior.pn, deferred.prior.dn))
                })
                .unwrap_or((child.prior.pn, child.prior.dn)),
        }
    }

    /// A thunk remains edge-local, but its exact key is also a virtual link to
    /// a transposition admitted through another parent. Every pre-selection
    /// read must observe that live entry just as an eagerly linked edge would.
    fn resolved_child_entry(&self, child: &WidePnChild) -> Option<usize> {
        child.entry.or_else(|| {
            child
                .future_key
                .as_ref()
                .and_then(WideFutureKey::virtual_key)
                .and_then(|key| self.by_position.get(key).copied())
        })
    }

    fn choice_order_pn(&self, child: &WidePnChild) -> u32 {
        self.child_numbers(child).0
    }

    fn select_child_index_with_tier(
        &self,
        kind: WidePnKind,
        children: &[WidePnChild],
        sequential_root_probe: bool,
        prefer_width_tier: bool,
    ) -> Option<usize> {
        if self.children_have_hint(children) {
            // The complete vector has already been stably prior-sorted. Drive
            // its first still-live edge so the external ordering is effective
            // without changing membership or any terminal classification.
            return children
                .iter()
                .enumerate()
                .find_map(|(index, child)| match kind {
                    WidePnKind::Choice if !self.child_is_genuinely_refuted(child) => Some(index),
                    WidePnKind::Universal { .. } if !self.child_is_genuinely_proven(child) => {
                        Some(index)
                    }
                    _ => None,
                });
        }
        if kind != WidePnKind::Choice || !self.zone_order_mode.enabled() {
            return self.select_child_index_baseline(
                kind,
                children,
                sequential_root_probe,
                prefer_width_tier,
            );
        }

        let baseline = self.select_child_index_baseline(
            kind,
            children,
            sequential_root_probe,
            prefer_width_tier,
        )?;
        let baseline_child = &children[baseline];

        if sequential_root_probe {
            let baseline_class = (
                self.child_numbers(baseline_child).0 != 0,
                !baseline_child.urgent_block,
                if prefer_width_tier {
                    baseline_child.first_width_tier
                } else {
                    0
                },
                baseline_child.prior.pn,
            );
            return children
                .iter()
                .enumerate()
                .filter(|(_, child)| !self.child_is_genuinely_refuted(child))
                .filter(|(_, child)| {
                    (
                        self.child_numbers(child).0 != 0,
                        !child.urgent_block,
                        if prefer_width_tier {
                            child.first_width_tier
                        } else {
                            0
                        },
                        child.prior.pn,
                    ) == baseline_class
                })
                .min_by_key(|(rank, child)| (child.zone_order_key, *rank))
                .map(|(index, _)| index);
        }

        // Width tier and immutable fork prior are hard classes. Start with the
        // class selected by the historical policy, then admit only its current
        // PN tie/band. This cannot pull a child across any established class.
        let baseline_width = if prefer_width_tier {
            baseline_child.first_width_tier
        } else {
            0
        };
        let baseline_prior = baseline_child.prior.pn;
        let band_limit = self
            .choice_order_pn(baseline_child)
            .saturating_add(self.zone_order_band);
        children
            .iter()
            .enumerate()
            .filter(|(_, child)| !self.child_is_genuinely_refuted(child))
            .filter(|(_, child)| {
                (if prefer_width_tier {
                    child.first_width_tier
                } else {
                    0
                }) == baseline_width
                    && child.prior.pn == baseline_prior
                    && self.choice_order_pn(child) <= band_limit
            })
            .min_by_key(|(rank, child)| (child.zone_order_key, *rank))
            .map(|(index, _)| index)
    }

    /// Historical selector kept as a separate off-path so R-OS2 cannot alter
    /// default scheduling through a changed tuple or filter.
    fn select_child_index_baseline(
        &self,
        kind: WidePnKind,
        children: &[WidePnChild],
        sequential_root_probe: bool,
        prefer_width_tier: bool,
    ) -> Option<usize> {
        if kind == WidePnKind::Choice && sequential_root_probe {
            return children
                .iter()
                .enumerate()
                .filter(|(_, child)| !self.child_is_genuinely_refuted(child))
                .min_by_key(|(rank, child)| {
                    let tactical = self.child_numbers(child).0 == 0;
                    (
                        !tactical,
                        !child.urgent_block,
                        if prefer_width_tier {
                            child.first_width_tier
                        } else {
                            0
                        },
                        child.prior.pn,
                        *rank,
                    )
                })
                .map(|(index, _)| index);
        }
        if kind == WidePnKind::Choice && prefer_width_tier {
            // A completed proof remains more-proving than every unresolved
            // width class. The tier profile only orders live obligations; it
            // must not postpone an already terminal claimant child.
            if let Some((index, _)) = children.iter().enumerate().find(|(_, child)| {
                !self.child_is_genuinely_refuted(child) && self.choice_order_pn(child) == 0
            }) {
                return Some(index);
            }
        }
        children
            .iter()
            .enumerate()
            .filter(|(_, child)| match kind {
                // A finite sum can saturate at the same sentinel used for a
                // finished child.  Selection must use semantic resolution,
                // not the numeric tie, or an earlier finished child can make
                // an otherwise live frontier report `Stalled`.
                WidePnKind::Choice => !self.child_is_genuinely_refuted(child),
                WidePnKind::Universal { .. } => !self.child_is_genuinely_proven(child),
            })
            .min_by_key(|(_, child)| {
                match kind {
                    // Iterator::min_by_key retains the first equal key, so
                    // canonical generator order is the only normal tie-break.
                    WidePnKind::Choice if prefer_width_tier => (
                        u32::from(child.first_width_tier),
                        self.choice_order_pn(child),
                    ),
                    WidePnKind::Choice => (0, self.choice_order_pn(child)),
                    WidePnKind::Universal { .. } => (self.child_numbers(child).1, 0),
                }
            })
            .map(|(index, _)| index)
    }

    /// Return whether this AND node has the high linked fanout where DN
    /// re-selection compounds obligation interleaving. Exact TT convergence is
    /// counted once, and an unlinked proof obligation postpones commitment.
    /// Linked entries remain part of the node's structural fanout after they
    /// prove so a qualifying Universal stays sequential through its binary
    /// tail instead of changing policy mid-proof.
    fn has_commitment_fanout(&self, children: &[WidePnChild]) -> bool {
        let mut unique = Vec::with_capacity(MIN_COMMITTED_UNIVERSAL_OBLIGATIONS);
        for child in children {
            let WidePnChildResult::Pending = child.result else {
                continue;
            };
            let Some(identity) = self.child_obligation_identity(child) else {
                return false;
            };
            if unique.contains(&identity) {
                continue;
            }
            if unique.len() < MIN_COMMITTED_UNIVERSAL_OBLIGATIONS {
                unique.push(identity);
            }
        }
        unique.len() >= MIN_COMMITTED_UNIVERSAL_OBLIGATIONS
    }

    fn child_obligation_identity<'a>(
        &'a self,
        child: &'a WidePnChild,
    ) -> Option<WideChildObligation<'a>> {
        if let Some(entry) = self
            .resolved_child_entry(child)
            .filter(|&entry| self.entries.get(entry).is_some())
        {
            return Some(WideChildObligation::Entry(entry));
        }
        let key = child.future_key.as_ref()?.virtual_key()?;
        Some(
            self.by_position
                .get(key)
                .copied()
                .map(WideChildObligation::Entry)
                .unwrap_or(WideChildObligation::FutureKey(key)),
        )
    }

    fn same_child_obligation(&self, left: &WidePnChild, right: &WidePnChild) -> bool {
        match (
            self.child_obligation_identity(left),
            self.child_obligation_identity(right),
        ) {
            (Some(left), Some(right)) => left == right,
            _ => false,
        }
    }

    fn universal_commitment_active(&self, id: usize, children: &[WidePnChild]) -> bool {
        self.entries[id]
            .universal_obligation
            .and_then(|index| children.get(index))
            .is_some_and(|child| !self.child_is_genuinely_proven(child))
            || self.has_commitment_fanout(children)
    }

    /// Select one high-fanout Universal obligation without letting changing
    /// DN estimates interleave its siblings. The first selection is exactly
    /// the ordinary lowest-DN/generator-order choice; later selections retain
    /// it until it resolves. `yielded` contains true-stall failures already
    /// tried by the current descent and lets the existing stall path fail over
    /// once per distinct sibling instead of spinning on an unaffordable child.
    fn universal_obligation_index(
        &self,
        id: usize,
        children: &[WidePnChild],
        yielded: &[usize],
    ) -> Option<usize> {
        let selectable = |index: usize, child: &WidePnChild| {
            let yielded_same_entry = yielded.iter().any(|&yielded_index| {
                children
                    .get(yielded_index)
                    .is_some_and(|yielded_child| self.same_child_obligation(child, yielded_child))
            });
            !yielded.contains(&index)
                && !yielded_same_entry
                && !self.child_is_genuinely_proven(child)
        };
        if let Some(index) = self.entries[id].universal_obligation {
            if children
                .get(index)
                .is_some_and(|child| selectable(index, child))
            {
                return Some(index);
            }
        }
        if self.children_have_hint(children) {
            return children
                .iter()
                .enumerate()
                .find(|(index, child)| selectable(*index, child))
                .map(|(index, _)| index);
        }
        children
            .iter()
            .enumerate()
            .filter(|(index, child)| selectable(*index, child))
            .min_by_key(|(_, child)| self.child_numbers(child).1)
            .map(|(index, _)| index)
    }

    fn select_step_child_index(
        &mut self,
        id: usize,
        sequential_root_probe: bool,
        prefer_width_tier: bool,
        yielded: &[usize],
    ) -> Option<usize> {
        self.select_step_child_index_with_commitment(
            id,
            sequential_root_probe,
            prefer_width_tier,
            yielded,
            false,
        )
    }

    fn select_step_child_index_with_commitment(
        &mut self,
        id: usize,
        sequential_root_probe: bool,
        prefer_width_tier: bool,
        yielded: &[usize],
        inherited_commitment: bool,
    ) -> Option<usize> {
        let (selected, universal_commitment) = {
            let WidePnNode::Branch { kind, children } = &self.entries[id].node else {
                return None;
            };
            match kind {
                WidePnKind::Choice => (
                    self.select_child_index_with_tier(
                        *kind,
                        children,
                        sequential_root_probe,
                        prefer_width_tier,
                    ),
                    None,
                ),
                WidePnKind::Universal { .. } => {
                    let commitment =
                        inherited_commitment || self.universal_commitment_active(id, children);
                    let selected = if commitment {
                        self.universal_obligation_index(id, children, yielded)
                    } else {
                        self.select_child_index_with_tier(
                            *kind,
                            children,
                            sequential_root_probe,
                            prefer_width_tier,
                        )
                    };
                    (selected, Some(commitment))
                }
            }
        };
        if let Some(commitment) = universal_commitment {
            self.entries[id].universal_obligation = if commitment { selected } else { None };
        }
        selected
    }

    /// A staged depth cutoff is unresolved, not a disproof. Sequential root
    /// probing must stay committed to that static top child so the caller can
    /// advance the horizon instead of silently moving to a lower-ranked turn.
    fn child_is_genuinely_refuted(&self, child: &WidePnChild) -> bool {
        match child.result {
            WidePnChildResult::Refuted => true,
            WidePnChildResult::Pending => self
                .resolved_child_entry(child)
                .and_then(|id| self.entries.get(id))
                .is_some_and(|entry| {
                    entry.dn == 0 && !matches!(entry.node, WidePnNode::DepthCutoff)
                }),
            WidePnChildResult::ClaimantCompletion | WidePnChildResult::ClaimantTactical => false,
        }
    }

    fn child_is_genuinely_proven(&self, child: &WidePnChild) -> bool {
        match child.result {
            WidePnChildResult::ClaimantCompletion | WidePnChildResult::ClaimantTactical => true,
            WidePnChildResult::Pending => self
                .resolved_child_entry(child)
                .and_then(|id| self.entries.get(id))
                .is_some_and(|entry| entry.pn == 0),
            WidePnChildResult::Refuted => false,
        }
    }

    fn recompute(&mut self, id: usize) -> bool {
        let previous = (self.entries[id].pn, self.entries[id].dn);
        let numbers = match &self.entries[id].node {
            WidePnNode::Unexpanded => {
                let prior = self.entries[id].prior;
                (prior.pn, prior.dn)
            }
            WidePnNode::ProvenLeaf(_) | WidePnNode::ProvenFragment(_) => (0, PN_INFINITY),
            WidePnNode::DepthCutoff | WidePnNode::Refuted => (PN_INFINITY, 0),
            WidePnNode::Branch { kind, children } => match kind {
                WidePnKind::Choice => {
                    let pn = children
                        .iter()
                        .map(|child| self.child_numbers(child).0)
                        .min()
                        .unwrap_or(PN_INFINITY);
                    let dn = children.iter().fold(0u32, |sum, child| {
                        sum.saturating_add(self.child_numbers(child).1)
                            .min(PN_INFINITY)
                    });
                    (pn, dn)
                }
                WidePnKind::Universal { .. } => {
                    let pn = children.iter().fold(0u32, |sum, child| {
                        sum.saturating_add(self.child_numbers(child).0)
                            .min(PN_INFINITY)
                    });
                    let dn = children
                        .iter()
                        .map(|child| self.child_numbers(child).1)
                        .min()
                        .unwrap_or(0);
                    (pn, dn)
                }
            },
        };

        self.entries[id].pn = numbers.0;
        self.entries[id].dn = numbers.1;

        previous != numbers
    }

    fn refresh(&mut self, id: usize) {
        self.recompute(id);
    }

    fn expand(&mut self, state: &mut RustHexoState, id: usize) -> WidePnStepOutcome {
        if self.expansions >= self.node_cap {
            return WidePnStepOutcome::Stalled;
        }
        self.expansions += 1;

        let depth = usize::try_from(state.placements_made().saturating_sub(self.root_ply))
            .unwrap_or(usize::MAX);
        if depth > self.depth_cap {
            self.entries[id].node = WidePnNode::DepthCutoff;
            self.refresh(id);
            return WidePnStepOutcome::DepthCutoff {
                depth,
                made_progress: true,
            };
        }
        if state.placements_made() > self.semantic_horizon {
            // A still-live line the semantic deadline refused (depth-bound, not
            // structural): the horizon-ladder trigger. A defender-to-move node
            // is one where the opponent is still branching (k < B, before the
            // fully-forced boundary) — the Group-2 `deep_kb_death` signal.
            self.horizon_cuts = self.horizon_cuts.saturating_add(1);
            if state.current_player() != self.claimant {
                self.kb_death_cuts = self.kb_death_cuts.saturating_add(1);
            }
            self.entries[id].node = WidePnNode::Refuted;
            self.refresh(id);
            return WidePnStepOutcome::Progress;
        }
        if let Some(store) = self.fragment_store.filter(|store| store.entry_count != 0) {
            self.fragment_lookups = self.fragment_lookups.saturating_add(1);
            let key = PositionKey::from_state(state);
            if let Some(fragment) = store.lookup(&key, self.claimant) {
                let proof = &fragment.proof;
                let root_is_universal = matches!(
                    proof.nodes.get(proof.root_node as usize),
                    Some(CertNode::Universal { .. })
                );
                let compatible = proof.validate().is_some()
                    && proof.resolution_t <= self.semantic_horizon
                    && proof
                        .zone_build_t
                        .is_none_or(|build_t| self.semantic_horizon <= build_t)
                    && depth
                        .checked_add(proof.height)
                        .is_some_and(|height| height <= self.max_depth_cap)
                    // Parent commutation permissions are path-local. A cached
                    // Universal is consumed only at the solve root, whose
                    // verifier context is known to be empty.
                    && (depth == 0 || !root_is_universal);
                if compatible {
                    self.fragment_hits = self.fragment_hits.saturating_add(1);
                    self.entries[id].node = WidePnNode::ProvenFragment(fragment);
                    self.refresh(id);
                    return WidePnStepOutcome::Progress;
                }
            }
        }
        if let Some(outcome) = state.terminal() {
            self.entries[id].node = if outcome.winner == self.claimant {
                WidePnNode::Refuted
            } else {
                WidePnNode::Refuted
            };
            self.refresh(id);
            return WidePnStepOutcome::Progress;
        }
        if !matches!(state.phase(), TurnPhase::Opening) {
            let analysis = threats::analyze(state);
            if let Some(winner) = winner_from_analysis(state, &analysis) {
                if winner == self.claimant {
                    match typed_lambda_leaf(
                        state,
                        winner,
                        &analysis,
                        WidthOptions::vcf_pair_complete(),
                    ) {
                        Some(leaf) if node_resolution(&leaf) <= self.semantic_horizon => {
                            self.entries[id].node = WidePnNode::ProvenLeaf(leaf);
                        }
                        Some(_) => {
                            // A real claimant win whose resolution ply is past
                            // the deadline: a depth-bound refusal, not a
                            // structural one — a horizon cut.
                            self.horizon_cuts = self.horizon_cuts.saturating_add(1);
                            self.entries[id].node = WidePnNode::Refuted;
                        }
                        None => {
                            self.entries[id].node = WidePnNode::Refuted;
                        }
                    }
                } else {
                    self.entries[id].node = WidePnNode::Refuted;
                }
                self.refresh(id);
                return WidePnStepOutcome::Progress;
            }
        }

        if self.interior_census_gate && state.current_player() == self.claimant {
            if let Some(evaluation) = evaluate_interior_census_gate(
                state,
                self.claimant,
                self.root_ply,
                self.semantic_horizon,
            ) {
                self.interior_gate_evaluations = self.interior_gate_evaluations.saturating_add(1);
                self.interior_gate_nanos =
                    self.interior_gate_nanos.saturating_add(evaluation.nanos);
                if evaluation.dismiss {
                    self.interior_gate_dismissals = self.interior_gate_dismissals.saturating_add(1);
                    self.entries[id].node = WidePnNode::Refuted;
                    self.refresh(id);
                    return WidePnStepOutcome::Progress;
                }
            }
        }

        let (kind, mut children) = if state.current_player() == self.claimant {
            (WidePnKind::Choice, self.attack_children(state, depth))
        } else {
            let analysis = threats::analyze(state);
            let implicit_dispatch = !matches!(state.phase(), TurnPhase::Opening)
                && analysis.opp_threat_count > 0
                && !analysis.own_win_now
                && analysis.min_hitting_set == Some(analysis.b);
            if !implicit_dispatch {
                self.entries[id].node = WidePnNode::Refuted;
                self.refresh(id);
                return WidePnStepOutcome::Progress;
            }
            let children = self.defender_boundary_children(state, analysis.b);
            (WidePnKind::Universal { implicit_dispatch }, children)
        };

        self.order_children_by_hints(&mut children);
        children.shrink_to_fit();
        self.entries[id].node = if children.is_empty() {
            WidePnNode::Refuted
        } else {
            WidePnNode::Branch { kind, children }
        };

        self.refresh(id);
        WidePnStepOutcome::Progress
    }

    fn attack_children(&self, state: &mut RustHexoState, depth: usize) -> Vec<WidePnChild> {
        match state.phase() {
            TurnPhase::FirstStone => self.attack_pair_children(state, depth),
            TurnPhase::SecondStone { first } => {
                self.attack_single_children(state, depth, Some(first))
            }
            TurnPhase::Opening => self.attack_single_children(state, depth, None),
        }
    }

    /// Stable-sort a fully generated child vector. This placement makes set
    /// invariance structural: hints cannot participate in generation, gates,
    /// deduplication, legality checks, or pruning. Atomic two-stone edges rank
    /// by their larger hinted weight, then their smaller hinted weight.
    fn order_children_by_hints(&self, children: &mut [WidePnChild]) {
        let Some(hints) = self.ordering_hints.as_ref() else {
            return;
        };
        children.sort_by(|left, right| {
            let left = hint_weights_for_move(hints, left.mv);
            let right = hint_weights_for_move(hints, right.mv);
            compare_hint_weight(right[0], left[0])
                .then_with(|| compare_hint_weight(right[1], left[1]))
        });
    }

    fn children_have_hint(&self, children: &[WidePnChild]) -> bool {
        self.ordering_hints.as_ref().is_some_and(|hints| {
            children
                .iter()
                .any(|child| hint_weights_for_move(hints, child.mv)[0].is_some())
        })
    }

    /// Enumerate complete attacker turns. A first stone is never admitted to
    /// the proof frontier by itself: either it wins immediately, or a retained
    /// pair must pass the new-threat and tight-dispatch forcing checks.
    /// Stateless replacement for the historical apply-and-analyze pair gate;
    /// see `WideTurnGate::evaluate_pair` for the classification contract.
    fn evaluate_wide_pair_at_gate(
        &self,
        gate: &WideTurnGate,
        scratch: &mut PairEvaluationScratch,
        first_windows: Option<&[u32]>,
        first: HexCoord,
        second: HexCoord,
    ) -> Option<(WidePnChildResult, WidePnPrior)> {
        let evaluated =
            gate.evaluate_pair(scratch, first_windows, first, second, self.semantic_horizon);
        let (result, prior) = evaluated?;

        Some((result, prior))
    }

    fn attack_pair_children(&self, state: &mut RustHexoState, _depth: usize) -> Vec<WidePnChild> {
        let gate = {
            let mut memo = self.generation_memo.borrow_mut();
            WideTurnGate::build_memoized(state, self.claimant, &mut memo)
        };

        let eager_pair_keys = false;
        let pair_key_template = (self.lazy_frontier && !eager_pair_keys)
            .then(|| Arc::new(WideAttackerPairKeyTemplate::from_state(state)));

        let observe_ordering_study = false;
        let observe_reveal_prefix = false;

        let ordering_context = if self.zone_order_mode.enabled()
            || observe_ordering_study
            || observe_reveal_prefix
        {
            let context =
                OrderingFeatureContext::from_state(state, self.claimant, observe_ordering_study);

            Some(context)
        } else {
            None
        };

        #[cfg(debug_assertions)]
        {
            let reference = ordered_threat_creating_moves_with_width(
                state,
                self.claimant,
                WidthOptions::vcf_pair_complete(),
            );
            assert_eq!(
                gate.first_candidates, reference,
                "single-scan first-candidate index diverged from the reference"
            );
        }
        let first_candidates = &gate.first_candidates;
        // Freeze urgency at the turn-start position. A block cell can disappear
        // from the second-stone candidate metadata after the other coordinate is
        // played, but the unordered pair still contains that original block.
        let defender_blocks = turn_start_defender_blocks(first_candidates);
        let mut children = Vec::new();
        let mut seen_pairs: HashSet<_, BuildHasherDefault<CoordHasher>> = HashSet::default();
        // No claimant >=4 window exists here (win-now nodes leaf before
        // generation), so a lone first stone can never complete six: the
        // whole double loop is stateless — zero engine applies.
        let mut second_coords: Vec<HexCoord> = Vec::new();
        let mut second_seen: CoordSet = CoordSet::default();
        let mut second_promoted: Vec<(u8, HexCoord)> = Vec::new();
        let mut second_fresh: Vec<HexCoord> = Vec::new();

        let mut pair_allow: Vec<(i16, i16)> = Vec::new();
        let mut pair_counts: Vec<((i16, i16), i16)> = Vec::new();
        let mut defender_allow: Vec<HexCoord> = Vec::new();
        let mut evaluation_scratch = PairEvaluationScratch::default();
        for first_candidate in first_candidates {
            let first_width_tier = wide_candidate_width_tier(first_candidate);
            let first = first_candidate.coord;
            // Exact family-size prefilter (see `pair_prefilter`): pairs it
            // skips are precisely `evaluate_pair` `None`s (empty or
            // single-window families), so children are byte-identical to the
            // unfiltered enumeration.
            let allow_all = gate.pair_prefilter(first, &mut pair_counts, &mut pair_allow);
            // Exact defender tier (see `defender_second_constraint`): with an
            // unhit defender >=4 window, only the <=2 intersection cells can
            // survive `defender_win_now`; an empty intersection rejects the
            // whole first. Skips are exactly `evaluate_pair` `None`s.
            let defender_constrained = gate.defender_second_constraint(first, &mut defender_allow);
            // Whole-first skips: every pair through `first` is provably a
            // `None` evaluation — generation itself is dead work. The
            // reveal-prefix study keeps the historical full enumeration
            // so its per-zone bookkeeping stays complete.
            if !observe_reveal_prefix
                && ((!allow_all && pair_allow.is_empty())
                    || (defender_constrained && defender_allow.is_empty()))
            {
                continue;
            }
            // Windows through `first` (hoisted out of the per-pair
            // evaluation; `evaluate_pair` historically re-looked this up for
            // every second).
            let first_windows = gate.windows_by_cell.get(&first).map(Vec::as_slice);

            {
                let force_reference = false;
                let optimized_growths = if force_reference {
                    {
                        unreachable!()
                    }
                } else {
                    gate.second_candidates(
                        first,
                        first_candidates,
                        &mut second_coords,
                        &mut second_seen,
                        &mut second_promoted,
                        &mut second_fresh,
                    )
                };

                let _ = optimized_growths;
                if self.width.free_tempo_j2near {
                    gate.append_j2near_after_turn_buying_first(
                        state,
                        first,
                        &mut second_coords,
                        &mut second_seen,
                    );
                }
            }

            for (_second_index, &second) in second_coords.iter().enumerate() {
                // Exact skips: outside the family allow set OR the defender
                // intersection the pair is provably a `None` evaluation (the
                // only side effects are the test-gated closure timers).
                if defender_constrained && !defender_allow.contains(&second) {
                    continue;
                }
                if !allow_all && pair_allow.binary_search(&raw_coord_key(second)).is_err() {
                    continue;
                }
                // Stateless classification from the turn-start window
                // snapshot: no engine applies in the pair double loop.

                let evaluated = self.evaluate_wide_pair_at_gate(
                    &gate,
                    &mut evaluation_scratch,
                    first_windows,
                    first,
                    second,
                );

                if let Some((result, prior)) = evaluated {
                    // Deduplicate the two legal orders by their actual
                    // unordered coordinate pair. Candidate membership is not
                    // monotone: a defender-block coordinate can disappear
                    // after the other stone, so coordinate-order pruning can
                    // incorrectly discard the only generated ordering.

                    let inserted = {
                        let first_key = raw_coord_key(first);
                        let second_key = raw_coord_key(second);
                        let pair_key = if first_key <= second_key {
                            (first_key, second_key)
                        } else {
                            (second_key, first_key)
                        };
                        seen_pairs.insert(pair_key)
                    };

                    if !inserted {
                        continue;
                    }

                    let mv = WidePnMove::Pair(first, second);
                    let zone_order_key = if self.zone_order_mode.enabled() {
                        let key = ordering_context
                            .as_ref()
                            .expect("live zone ordering builds a turn-start context")
                            .pair_key(first, second, self.zone_order_mode);

                        key
                    } else {
                        0
                    };
                    children.push(WidePnChild {
                        mv,
                        result,
                        entry: None,
                        future_key: (self.lazy_frontier && result == WidePnChildResult::Pending)
                            .then(|| {
                                let template = pair_key_template
                                    .as_ref()
                                    .expect("lazy pair generation builds a key template");
                                WideFutureKey::OnSelectionPair {
                                    template: Arc::clone(template),
                                    first,
                                    second,
                                }
                            }),
                        prior,
                        urgent_block: wide_move_contains_defender_block(mv, &defender_blocks),
                        first_width_tier,
                        zone_order_key,
                    });
                }
            }
        }

        children
    }

    fn attack_single_children(
        &self,
        state: &mut RustHexoState,
        depth: usize,
        turn_first: Option<HexCoord>,
    ) -> Vec<WidePnChild> {
        let mut candidates = ordered_threat_creating_moves_with_width(
            state,
            self.claimant,
            WidthOptions::vcf_pair_complete(),
        );
        if self.width.free_tempo_j2near
            && turn_first.is_some()
            && partial_turn_is_turn_buying(state, self.claimant)
        {
            let mut seen = candidates
                .iter()
                .map(|candidate| candidate.coord)
                .collect::<HashSet<_>>();
            for coord in j2near_candidates_in_state(state, self.claimant, &mut seen) {
                candidates.push(Candidate {
                    coord,
                    strength: 1,
                    priority_class: 3,
                    child_threats: 0,
                    defender_block: false,
                    pair_start_degree: 0,
                    own_proximity: i16::MAX,
                    created_threats: Vec::new(),
                });
            }
        }

        let mut children = Vec::new();
        for candidate in candidates {
            let Ok((result, delta)) = state.apply_with_delta(Placement {
                coord: candidate.coord,
            }) else {
                continue;
            };
            let completion_ply = self.root_ply.saturating_add(depth as u32).saturating_add(1);
            let (child_result, prior) = if let Some(outcome) = result.outcome {
                if outcome.winner == self.claimant && completion_ply <= self.semantic_horizon {
                    (
                        Some(WidePnChildResult::ClaimantCompletion),
                        WidePnPrior::UNIFORM,
                    )
                } else {
                    (None, WidePnPrior::UNIFORM)
                }
            } else if let Some(first) = turn_first {
                if immediate_winner(state, WidthOptions::vcf_pair_complete()).is_some_and(
                    |(winner, ref leaf)| {
                        winner == self.claimant && node_resolution(leaf) <= self.semantic_horizon
                    },
                ) {
                    (
                        Some(WidePnChildResult::ClaimantTactical),
                        WidePnPrior::UNIFORM,
                    )
                } else {
                    let forcing = (turn_created_claimant_threat(
                        state,
                        self.claimant,
                        first,
                        candidate.coord,
                    ) && turn_forces_small_defender_reply(state, self.claimant))
                    .then_some(WidePnChildResult::Pending);
                    let prior = forcing
                        .is_some()
                        .then(|| self.completed_turn_prior(state))
                        .unwrap_or(WidePnPrior::UNIFORM);
                    (forcing, prior)
                }
            } else {
                (
                    Some(WidePnChildResult::Pending),
                    if state.current_player() == self.claimant {
                        self.position_prior(state)
                    } else {
                        self.completed_turn_prior(state)
                    },
                )
            };
            let future_key = (self.lazy_frontier
                && child_result == Some(WidePnChildResult::Pending))
            .then(|| WideFutureKey::OnSelection(WidePositionKey::from_state(state)));
            state.undo(delta);
            if let Some(result) = child_result {
                children.push(WidePnChild {
                    mv: WidePnMove::One(candidate.coord),
                    result,
                    entry: None,
                    future_key,
                    prior,
                    urgent_block: candidate.defender_block,
                    first_width_tier: 0,
                    zone_order_key: 0,
                });
            }
        }
        children
    }

    fn defender_children(
        &mut self,
        state: &mut RustHexoState,
        defender_budget: u8,
    ) -> Vec<WidePnChild> {
        let mut explicit = forced_defender_replies(
            state,
            self.claimant,
            defender_budget,
            WidthOptions::vcf_pair_complete(),
        );
        let frame = canonical_frame(state);
        explicit.sort_by_key(|coord| canonical_coord_key(frame, *coord));
        let mut children = Vec::with_capacity(explicit.len());
        for coord in explicit {
            let Ok((result, delta)) = state.apply_with_delta(Placement { coord }) else {
                continue;
            };
            let child_result = match result.outcome {
                Some(outcome) if outcome.winner == self.claimant => {
                    WidePnChildResult::ClaimantCompletion
                }
                Some(_) => WidePnChildResult::Refuted,
                None => WidePnChildResult::Pending,
            };
            let prior = (child_result == WidePnChildResult::Pending)
                .then(|| self.position_prior(state))
                .unwrap_or(WidePnPrior::UNIFORM);
            let (entry, future_key) = if child_result == WidePnChildResult::Pending {
                let depth = usize::try_from(state.placements_made().saturating_sub(self.root_ply))
                    .unwrap_or(usize::MAX);
                let key = WidePositionKey::from_state(state);
                if self.lazy_frontier {
                    self.defer_position(&key, depth, prior);
                    (None, Some(WideFutureKey::Virtual(key)))
                } else {
                    (Some(self.insert_position(key, depth, prior)), None)
                }
            } else {
                (None, None)
            };
            state.undo(delta);
            children.push(WidePnChild {
                mv: WidePnMove::One(coord),
                result: child_result,
                entry,
                future_key,
                prior,
                urgent_block: false,
                first_width_tier: 0,
                zone_order_key: 0,
            });
        }
        children
    }

    fn defender_boundary_children(
        &mut self,
        state: &mut RustHexoState,
        defender_budget: u8,
    ) -> Vec<WidePnChild> {
        if defender_budget == 2 && matches!(state.phase(), TurnPhase::FirstStone) {
            if let Some(children) = self.defender_pair_children(state) {
                return children;
            }
        }
        self.defender_children(state, defender_budget)
    }

    fn defender_pair_children(&mut self, state: &mut RustHexoState) -> Option<Vec<WidePnChild>> {
        let plan = forced_defender_pair_plan(state, self.claimant)?;
        let depth = usize::try_from(
            state
                .placements_made()
                .saturating_add(2)
                .saturating_sub(self.root_ply),
        )
        .unwrap_or(usize::MAX);
        Some(
            plan.pairs
                .into_iter()
                .map(|pair| {
                    let final_prior = pair.final_prior;
                    let (entry, future_key) = if self.lazy_frontier {
                        self.defer_position(&pair.final_key, depth, final_prior);
                        (None, Some(WideFutureKey::Virtual(pair.final_key)))
                    } else {
                        (
                            Some(self.insert_position(pair.final_key, depth, final_prior)),
                            None,
                        )
                    };
                    WidePnChild {
                        mv: WidePnMove::DefenderPair(pair.first, pair.second),
                        result: WidePnChildResult::Pending,
                        entry,
                        future_key,
                        prior: final_prior,
                        urgent_block: false,
                        first_width_tier: 0,
                        zone_order_key: 0,
                    }
                })
                .collect(),
        )
    }

    fn materialize(&self, state: &RustHexoState, root: usize) -> Option<WideMaterializedProof> {
        if self.entries.get(root)?.pn != 0 {
            return None;
        }
        let mut work = state.clone();
        let mut builder = WideProofMaterializer {
            search: self,
            arena: Vec::new(),
            edge_count: 0,
            commutation_count: 0,
            witness_count: 0,
            fragment_imports: 0,
            dag_reuses: 0,
            memo: HashMap::new(),
        };
        let root_node = builder.build(&mut work, root)?;
        Some(WideMaterializedProof {
            arena: builder.arena,
            root_node,
            fragment_imports: builder.fragment_imports,
            dag_reuses: builder.dag_reuses,
        })
    }
}

struct WideMaterializedProof {
    arena: Vec<CertNode>,
    root_node: CertNodeId,
    fragment_imports: u64,
    dag_reuses: u64,
}

struct WideProofMaterializer<'search, 'store> {
    search: &'search WidePnSearch<'store>,
    arena: Vec<CertNode>,
    edge_count: usize,
    commutation_count: usize,
    witness_count: usize,
    fragment_imports: u64,
    dag_reuses: u64,
    memo: HashMap<PositionKey, CertNodeId>,
}

impl WideProofMaterializer<'_, '_> {
    fn build(&mut self, state: &mut RustHexoState, id: usize) -> Option<CertNodeId> {
        let key = PositionKey::from_state(state);
        if let Some(&node) = self.memo.get(&key) {
            self.dag_reuses = self.dag_reuses.saturating_add(1);
            return Some(node);
        }
        let entry = self.search.entries.get(id)?;
        if entry.pn != 0 {
            return None;
        }
        let node = match entry.node.clone() {
            WidePnNode::ProvenLeaf(leaf) => self.alloc(leaf, 0)?,
            WidePnNode::ProvenFragment(fragment) => {
                if fragment.claimant != self.search.claimant || fragment.key != key {
                    return None;
                }
                self.import_fragment(state, &fragment.proof)?
            }
            WidePnNode::Branch {
                kind: WidePnKind::Choice,
                children,
            } => {
                let child = children
                    .iter()
                    .find(|child| self.search.child_numbers(child).0 == 0)?
                    .clone();
                self.build_choice(state, &child)?
            }
            WidePnNode::Branch {
                kind: WidePnKind::Universal { implicit_dispatch },
                children,
            } => self.build_universal(state, implicit_dispatch, &children)?,
            WidePnNode::Unexpanded | WidePnNode::DepthCutoff | WidePnNode::Refuted => return None,
        };
        self.memo.insert(key, node);
        Some(node)
    }

    fn import_fragment(
        &mut self,
        state: &RustHexoState,
        proof: &CachedProof,
    ) -> Option<CertNodeId> {
        proof.validate()?;
        let depth =
            usize::try_from(state.placements_made().saturating_sub(self.search.root_ply)).ok()?;
        if proof.resolution_t > self.search.semantic_horizon
            || proof
                .zone_build_t
                .is_some_and(|build_t| self.search.semantic_horizon > build_t)
            || depth.checked_add(proof.height)? > self.search.max_depth_cap
            || self.arena.len().checked_add(proof.nodes.len())? > MAX_CERT_NODES
            || self.edge_count.checked_add(proof.explicit_edges)? > MAX_CERT_EDGES
            || self
                .commutation_count
                .checked_add(proof.commutation_count)?
                > MAX_CERT_COMMUTATIONS
            || self.witness_count.checked_add(proof.witness_count)? > MAX_CERT_WITNESSES
        {
            return None;
        }

        let base = self.arena.len();
        let final_len = base.checked_add(proof.nodes.len())?;
        u32::try_from(final_len).ok()?;
        let mut nodes = proof.nodes.clone();
        for node in &mut nodes {
            remap_node_ids_with_offset(node, base, final_len)?;
        }
        let root = offset_node_id(proof.root_node, base, final_len)?;
        self.arena.append(&mut nodes);
        self.edge_count += proof.explicit_edges;
        self.commutation_count += proof.commutation_count;
        self.witness_count += proof.witness_count;
        self.fragment_imports = self.fragment_imports.saturating_add(1);
        Some(root)
    }

    fn build_choice(
        &mut self,
        state: &mut RustHexoState,
        child: &WidePnChild,
    ) -> Option<CertNodeId> {
        match child.mv {
            WidePnMove::One(coord) => {
                let (result, delta) = state.apply_with_delta(Placement { coord }).ok()?;
                let node = match child.result {
                    WidePnChildResult::ClaimantCompletion => {
                        if result.outcome?.winner != self.search.claimant {
                            state.undo(delta);
                            return None;
                        }
                        let completion = wide_completion_node(
                            state,
                            self.search.claimant,
                            coord,
                            state.placements_made(),
                        );
                        state.undo(delta);
                        self.alloc(completion?, 0)?
                    }
                    WidePnChildResult::ClaimantTactical => {
                        if result.outcome.is_some() {
                            state.undo(delta);
                            return None;
                        }
                        let analysis = threats::analyze(state);
                        let leaf = typed_lambda_leaf(
                            state,
                            self.search.claimant,
                            &analysis,
                            WidthOptions::vcf_pair_complete(),
                        )
                        .filter(|leaf| node_resolution(leaf) <= self.search.semantic_horizon);
                        state.undo(delta);
                        let leaf = self.alloc(leaf?, 0)?;
                        self.alloc(
                            CertNode::Choice {
                                mv: coord,
                                child: leaf,
                            },
                            1,
                        )?
                    }
                    WidePnChildResult::Pending => {
                        let child_id = self.search.resolved_child_entry(child)?;
                        let proof = self.build(state, child_id);
                        state.undo(delta);
                        self.alloc(
                            CertNode::Choice {
                                mv: coord,
                                child: proof?,
                            },
                            1,
                        )?
                    }
                    WidePnChildResult::Refuted => {
                        state.undo(delta);
                        return None;
                    }
                };
                Some(node)
            }
            WidePnMove::Pair(first, second) => {
                let (first_result, first_delta) =
                    state.apply_with_delta(Placement { coord: first }).ok()?;
                if first_result.outcome.is_some() {
                    state.undo(first_delta);
                    return None;
                }
                let (second_result, second_delta) =
                    state.apply_with_delta(Placement { coord: second }).ok()?;
                let node = match child.result {
                    WidePnChildResult::ClaimantCompletion => {
                        if second_result.outcome?.winner != self.search.claimant {
                            state.undo(second_delta);
                            state.undo(first_delta);
                            return None;
                        }
                        let completion = wide_completion_node(
                            state,
                            self.search.claimant,
                            second,
                            state.placements_made(),
                        );
                        state.undo(second_delta);
                        state.undo(first_delta);
                        let completion = self.alloc(completion?, 0)?;
                        self.alloc(
                            CertNode::Choice {
                                mv: first,
                                child: completion,
                            },
                            1,
                        )?
                    }
                    WidePnChildResult::ClaimantTactical => {
                        if second_result.outcome.is_some() {
                            state.undo(second_delta);
                            state.undo(first_delta);
                            return None;
                        }
                        let analysis = threats::analyze(state);
                        let leaf = typed_lambda_leaf(
                            state,
                            self.search.claimant,
                            &analysis,
                            WidthOptions::vcf_pair_complete(),
                        )
                        .filter(|leaf| node_resolution(leaf) <= self.search.semantic_horizon);
                        state.undo(second_delta);
                        state.undo(first_delta);
                        let leaf = self.alloc(leaf?, 0)?;
                        let second_choice = self.alloc(
                            CertNode::Choice {
                                mv: second,
                                child: leaf,
                            },
                            1,
                        )?;
                        self.alloc(
                            CertNode::Choice {
                                mv: first,
                                child: second_choice,
                            },
                            1,
                        )?
                    }
                    WidePnChildResult::Pending => {
                        let proof = self.build(state, self.search.resolved_child_entry(child)?);
                        state.undo(second_delta);
                        state.undo(first_delta);
                        let second_choice = self.alloc(
                            CertNode::Choice {
                                mv: second,
                                child: proof?,
                            },
                            1,
                        )?;
                        self.alloc(
                            CertNode::Choice {
                                mv: first,
                                child: second_choice,
                            },
                            1,
                        )?
                    }
                    WidePnChildResult::Refuted => {
                        state.undo(second_delta);
                        state.undo(first_delta);
                        return None;
                    }
                };
                Some(node)
            }
            WidePnMove::DefenderPair(_, _) => None,
        }
    }

    fn build_universal(
        &mut self,
        state: &mut RustHexoState,
        implicit_dispatch: bool,
        children: &[WidePnChild],
    ) -> Option<CertNodeId> {
        if children
            .first()
            .is_some_and(|child| matches!(child.mv, WidePnMove::DefenderPair(_, _)))
        {
            if !implicit_dispatch
                || children
                    .iter()
                    .any(|child| !matches!(child.mv, WidePnMove::DefenderPair(_, _)))
            {
                return None;
            }
            return self.build_defender_pair_universal(state, children);
        }
        let mut edges = Vec::with_capacity(children.len());
        for child in children {
            if self.search.child_numbers(child).0 != 0 || child.result != WidePnChildResult::Pending
            {
                return None;
            }
            let WidePnMove::One(coord) = child.mv else {
                return None;
            };
            let (_result, delta) = state.apply_with_delta(Placement { coord }).ok()?;
            let proof = self.build(state, self.search.resolved_child_entry(child)?);
            state.undo(delta);
            edges.push(CertEdge {
                mv: coord,
                child: proof?,
            });
        }
        let edge_count = edges.len();
        self.alloc(
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone: None,
                commutations: Vec::new(),
            },
            edge_count,
        )
    }

    fn build_defender_pair_universal(
        &mut self,
        state: &mut RustHexoState,
        children: &[WidePnChild],
    ) -> Option<CertNodeId> {
        let plan = forced_defender_pair_plan(state, self.search.claimant)?;
        if plan.pairs.len() != children.len() {
            return None;
        }

        let mut child_by_pair = HashMap::with_capacity(children.len());
        for child in children {
            if child.result != WidePnChildResult::Pending || self.search.child_numbers(child).0 != 0
            {
                return None;
            }
            let WidePnMove::DefenderPair(first, second) = child.mv else {
                return None;
            };
            if raw_coord_key(first) >= raw_coord_key(second)
                || child_by_pair
                    .insert((raw_coord_key(first), raw_coord_key(second)), child)
                    .is_some()
            {
                return None;
            }
        }

        // Build each unique final-state proof exactly once. The reverse order
        // reaches the same state by construction and is represented below by
        // a checked CertCommutation rather than a second PN obligation.
        let mut proof_by_pair = HashMap::with_capacity(plan.pairs.len());
        for pair in &plan.pairs {
            let pair_key = (raw_coord_key(pair.first), raw_coord_key(pair.second));
            let child = *child_by_pair.get(&pair_key)?;
            let (first_result, first_delta) = state
                .apply_with_delta(Placement { coord: pair.first })
                .ok()?;
            if first_result.outcome.is_some() {
                state.undo(first_delta);
                return None;
            }
            let (second_result, second_delta) =
                match state.apply_with_delta(Placement { coord: pair.second }) {
                    Ok(applied) => applied,
                    Err(_) => {
                        state.undo(first_delta);
                        return None;
                    }
                };
            if second_result.outcome.is_some()
                || WidePositionKey::from_state(state) != pair.final_key
            {
                state.undo(second_delta);
                state.undo(first_delta);
                return None;
            }
            let Some(child_id) = self.search.resolved_child_entry(child) else {
                state.undo(second_delta);
                state.undo(first_delta);
                return None;
            };
            let proof = self.build(state, child_id);
            state.undo(second_delta);
            state.undo(first_delta);
            if proof_by_pair.insert(pair_key, proof?).is_some() {
                return None;
            }
        }

        // Retain the raw-low -> raw-high orientation explicitly. Every
        // raw-high -> raw-low orientation is omitted from that nested
        // Universal and justified by a root-level commutation record.
        let mut nested_by_first = HashMap::with_capacity(plan.kernel.len());
        for &first in &plan.kernel {
            let edges = plan
                .pairs
                .iter()
                .filter(|pair| pair.first == first)
                .map(|pair| {
                    Some(CertEdge {
                        mv: pair.second,
                        child: *proof_by_pair
                            .get(&(raw_coord_key(pair.first), raw_coord_key(pair.second)))?,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            let edge_count = edges.len();
            let node = self.alloc(
                CertNode::Universal {
                    edges,
                    implicit_dispatch: true,
                    zone: None,
                    commutations: Vec::new(),
                },
                edge_count,
            )?;
            if nested_by_first.insert(first, node).is_some() {
                return None;
            }
        }

        let edges = plan
            .kernel
            .iter()
            .map(|&mv| {
                Some(CertEdge {
                    mv,
                    child: *nested_by_first.get(&mv)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let commutations = plan
            .pairs
            .iter()
            .map(|pair| {
                Some(CertCommutation {
                    first: pair.second,
                    omitted_second: pair.first,
                    first_child: *nested_by_first.get(&pair.second)?,
                    mirror_child: *nested_by_first.get(&pair.first)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let edge_count = edges.len();
        self.alloc(
            CertNode::Universal {
                edges,
                implicit_dispatch: true,
                zone: None,
                commutations,
            },
            edge_count,
        )
    }

    fn alloc(&mut self, node: CertNode, added_edges: usize) -> Option<CertNodeId> {
        let added_commutations = match &node {
            CertNode::Universal { commutations, .. } => commutations.len(),
            _ => 0,
        };
        let added_witnesses = match &node {
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => 1,
            CertNode::Loss { witnesses, .. } => witnesses.len(),
            CertNode::Choice { .. }
            | CertNode::Universal { .. }
            | CertNode::UniversalGroup2V1(_) => 0,
            CertNode::FhwGateV1(gate) => gate.proof.threats.len(),
        };
        if self.arena.len() >= MAX_CERT_NODES
            || self.edge_count.saturating_add(added_edges) > MAX_CERT_EDGES
            || self.commutation_count.saturating_add(added_commutations) > MAX_CERT_COMMUTATIONS
            || (self.search.fragment_store.is_some()
                && self.witness_count.saturating_add(added_witnesses) > MAX_CERT_WITNESSES)
        {
            return None;
        }
        let id = u32::try_from(self.arena.len()).ok()?;
        self.arena.push(node);
        self.edge_count += added_edges;
        self.commutation_count += added_commutations;
        self.witness_count += added_witnesses;
        Some(id)
    }
}

fn wide_completion_node(
    state: &RustHexoState,
    claimant: Player,
    coord: HexCoord,
    completion_ply: u32,
) -> Option<CertNode> {
    let mut witnesses = state
        .board()
        .windows()
        .entries()
        .filter(|entry| {
            entry.key().contains(coord)
                && entry.count(claimant) == 6
                && entry.count(claimant.other()) == 0
        })
        .map(|entry| entry.key())
        .collect::<Vec<_>>();
    witnesses.sort_by_key(|key| window_key_order(*key));
    Some(CertNode::OrCompletion {
        mv: coord,
        witness: witnesses.first().copied()?,
        completion_ply,
    })
}

/// Narrow DFS state retained byte-for-byte as the compatibility backend
/// selected by `WidePnSearch::prove_narrow_compat`.
struct NarrowCompatSearch<'a> {
    node_cap: u64,
    nodes: u64,
    tt_hits: u64,
    hit_limit: bool,
    arena: Vec<CertNode>,
    edge_count: usize,
    tt: BoundedTt,
    shared_tt: Option<&'a mut SharedProofCache>,
    peak_tt_bytes: usize,
    /// Absolute placement index at the attempt root.  Structural depth is
    /// derived from the separately threaded ply clock.
    root_ply: u32,
    semantic_horizon: u32,
    clock_is_absolute: bool,
    zone: ZoneSearchCaps,
    width: WidthOptions,
    depth_cap: usize,
    /// Immutable solve-level opt-in. Environment lookup happens in
    /// `TssSolver::solve_goal`, never on the recursive search path.
    k_reply_consume: bool,

    interior_census_gate: bool,
    interior_gate_evaluations: u64,
    interior_gate_dismissals: u64,
    interior_gate_nanos: u64,
    /// Still-live lines the semantic-horizon deadline refused, and the
    /// defender-to-move subset. Mirror of the wide-search counters
    /// (SolveStats::horizon_cuts / kb_death_cuts) for the narrow-compat path.
    horizon_cuts: u64,
    kb_death_cuts: u64,
    /// v1 Group-2 selector opt-in for this attempt.
    group2: bool,
    /// True once any node outside the narrow v1 class (implicit dispatch,
    /// legacy zone, commutation) has been allocated. Later Group-2 attempts
    /// are skipped: the assembled certificate could no longer validate as v1
    /// (class rules 2/3), so trying would only waste budget.
    emitted_dirty: bool,
}

#[derive(Clone, Debug)]
struct PairContext {
    first: HexCoord,
    turn_start_legal: Vec<HexCoord>,
}

/// Exact Q8 reply-survival kernel from the NQ2 proof. Urgency is deliberately
/// scoped to the theorem's nonterminal attacker SecondStone position. Defender
/// windows come from the engine's incrementally maintained exact mirror of all
/// active count-4+ windows; tests bind that mirror to a full `entries()` scan.
fn k_reply_eligible(state: &RustHexoState, claimant: Player) -> bool {
    state.terminal().is_none()
        && state.current_player() == claimant
        && matches!(state.phase(), TurnPhase::SecondStone { .. })
}

pub(crate) fn k_reply_kernel(
    state: &RustHexoState,
    claimant: Player,
    legal: &[HexCoord],
) -> KReplyKernel {
    let eligible = k_reply_eligible(state, claimant);
    let defender = claimant.other();
    let mut defender_windows = Vec::new();
    let mut win_now_windows = Vec::new();
    if eligible {
        // The showcase engine's `threats()` scan yields the same complete
        // `(owner, entry)` multiset as the live engine's incrementally indexed
        // `live_threat_entries()`. Apply Q8's exact owner/count cuts.
        for (owner, entry) in state.board().windows().threats() {
            if owner == defender
                && entry.active_player() == Some(defender)
                && matches!(entry.count(defender), 4 | 5)
            {
                defender_windows.push(entry.key());
            } else if owner == claimant
                && entry.active_player() == Some(claimant)
                && entry.count(claimant) == 5
            {
                win_now_windows.push(entry.key());
            }
        }
    }
    let urgent = !defender_windows.is_empty();
    let cells = if !eligible {
        Vec::new()
    } else if urgent {
        legal
            .iter()
            .copied()
            .filter(|coord| {
                let wins_now = win_now_windows.iter().any(|window| window.contains(*coord));
                wins_now
                    || defender_windows
                        .iter()
                        .all(|window| window.contains(*coord))
            })
            .collect()
    } else {
        // Q8 defines BlockAll_D(P)=Legal(P) for the empty defender-window
        // family. `retained()` returns Legal(P) without copying it.
        Vec::new()
    };
    KReplyKernel {
        eligible,
        urgent,
        cells,
    }
}

impl NarrowCompatSearch<'static> {
    fn new(node_cap: u64, tt_bytes_cap: usize, hash_mask: u64) -> Self {
        let tt = BoundedTt::new(tt_bytes_cap, hash_mask);
        let peak_tt_bytes = tt.current_bytes;
        Self {
            node_cap,
            nodes: 0,
            tt_hits: 0,
            hit_limit: false,
            arena: Vec::new(),
            edge_count: 0,
            tt,
            shared_tt: None,
            peak_tt_bytes,
            root_ply: 0,
            semantic_horizon: u32::MAX,
            clock_is_absolute: false,
            zone: ZoneSearchCaps::default(),
            width: WidthOptions::default(),
            depth_cap: MAX_SEARCH_DEPTH,
            k_reply_consume: false,

            interior_census_gate: false,
            interior_gate_evaluations: 0,
            interior_gate_dismissals: 0,
            interior_gate_nanos: 0,
            horizon_cuts: 0,
            kb_death_cuts: 0,
            group2: false,
            emitted_dirty: false,
        }
    }
}

impl<'a> NarrowCompatSearch<'a> {
    fn with_shared(
        node_cap: u64,
        tt_bytes_cap: usize,
        hash_mask: u64,
        shared_tt: &'a mut SharedProofCache,
        root_ply: u32,
        semantic_horizon: u32,
        zone: ZoneSearchCaps,
        width: WidthOptions,
        depth_cap: usize,
        k_reply_consume: bool,

        interior_census_gate: bool,
        group2: bool,
    ) -> Self {
        let tt = BoundedTt::new(tt_bytes_cap, hash_mask);
        let peak_tt_bytes = tt.current_bytes.saturating_add(shared_tt.current_bytes);
        let shared_tt = (!shared_tt.slots.is_empty()).then_some(shared_tt);
        Self {
            node_cap,
            nodes: 0,
            tt_hits: 0,
            hit_limit: false,
            arena: Vec::new(),
            edge_count: 0,
            tt,
            shared_tt,
            peak_tt_bytes,
            root_ply,
            semantic_horizon,
            clock_is_absolute: true,
            zone,
            width,
            depth_cap,
            k_reply_consume,

            interior_census_gate,
            interior_gate_evaluations: 0,
            interior_gate_dismissals: 0,
            interior_gate_nanos: 0,
            horizon_cuts: 0,
            kb_death_cuts: 0,
            group2,
            emitted_dirty: false,
        }
    }

    fn tt_entry_count(&self) -> usize {
        self.tt.entry_count()
            + self
                .shared_tt
                .as_ref()
                .map(|shared| shared.entry_count())
                .unwrap_or(0)
    }

    fn prove(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
        pair: Option<&PairContext>,
    ) -> Option<CertNodeId> {
        let depth = if self.clock_is_absolute {
            debug_assert_eq!(state.placements_made(), ply);
            usize::try_from(ply.checked_sub(self.root_ply)?).ok()?
        } else {
            ply as usize
        };
        if self.clock_is_absolute && ply > self.semantic_horizon {
            // Deadline refused a still-live line (depth-bound Unknown). A
            // defender-to-move node is pre-forced-boundary (k < B).
            self.horizon_cuts = self.horizon_cuts.saturating_add(1);
            if state.current_player() != claimant {
                self.kb_death_cuts = self.kb_death_cuts.saturating_add(1);
            }
            return None;
        }
        if depth > self.depth_cap {
            if !self.width.vcf_pair_complete {
                self.hit_limit = true;
            }
            return None;
        }
        if self.nodes >= self.node_cap {
            self.hit_limit = true;
            return None;
        }
        self.nodes += 1;

        let pn_init_result = (|| {
            let key = PositionKey::from_state(state);
            if pair.is_none() {
                if let Some(node) = self.tt.lookup(&key, claimant) {
                    if node == LOCAL_TT_FAILED && self.width.vcf_pair_complete {
                        self.tt_hits += 1;
                        return None;
                    }
                    if (node as usize) < self.arena.len() {
                        self.tt_hits += 1;
                        return Some(node);
                    }
                }
                if let Some(node) = self.lookup_shared(&key, claimant, depth) {
                    self.tt.insert(key, claimant, node);
                    self.tt_hits += 1;
                    self.observe_tt_bytes();
                    return Some(node);
                }
            }

            if let Some(outcome) = state.terminal() {
                let _ = outcome;
                // A claimant completion is represented at its parent by the typed
                // OrCompletion leaf; defender-terminal edges are not certifiable.
                return None;
            }

            // Analyze each non-terminal node exactly once.  Universal dispatch
            // consumes this same immutable result instead of repeating the scan.
            let analysis = threats::analyze(state);
            if !matches!(state.phase(), TurnPhase::Opening) {
                if let Some(winner) = winner_from_analysis(state, &analysis) {
                    if winner != claimant {
                        return None;
                    }
                    let leaf = typed_lambda_leaf(state, winner, &analysis, self.width)?;
                    if node_resolution(&leaf) > self.semantic_horizon {
                        self.horizon_cuts = self.horizon_cuts.saturating_add(1);
                        return None;
                    }
                    let node = self.alloc_node(leaf, 0)?;
                    self.remember_proof(key, claimant, node);
                    return Some(node);
                }
            }

            let gate_dismissed = if self.interior_census_gate && state.current_player() == claimant
            {
                evaluate_interior_census_gate(state, claimant, self.root_ply, self.semantic_horizon)
                    .is_some_and(|evaluation| {
                        self.interior_gate_evaluations =
                            self.interior_gate_evaluations.saturating_add(1);
                        self.interior_gate_nanos =
                            self.interior_gate_nanos.saturating_add(evaluation.nanos);
                        if evaluation.dismiss {
                            self.interior_gate_dismissals =
                                self.interior_gate_dismissals.saturating_add(1);
                        }
                        evaluation.dismiss
                    })
            } else {
                false
            };

            let node = if state.current_player() == claimant {
                if gate_dismissed {
                    None
                } else {
                    self.prove_choice(state, claimant, ply, &analysis, pair)
                }
            } else {
                self.prove_universal(state, claimant, ply, &analysis, pair)
            };
            let Some(node) = node else {
                if self.width.vcf_pair_complete && !self.hit_limit && pair.is_none() {
                    self.tt.insert(key, claimant, LOCAL_TT_FAILED);
                    self.observe_tt_bytes();
                }
                return None;
            };
            if pair.is_none() {
                self.remember_proof(key, claimant, node);
            }
            Some(node)
        })();

        pn_init_result
    }

    fn prove_choice(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
        analysis: &threats::ThreatAnalysis,
        pair: Option<&PairContext>,
    ) -> Option<CertNodeId> {
        // Descending line count is the static proof-number initialization:
        // completions before four-builds before three-builds.  The coordinate
        // tie break makes the order independent of WindowStore hash iteration.
        let mut candidates = ordered_threat_creating_moves_with_width(state, claimant, self.width);
        if self.width.vcf_pair_complete {
            if let Some(pair) = pair {
                candidates.retain(|candidate| pair_candidate_allowed(candidate.coord, pair));
            }
        }
        let quiet_priority = candidates
            .iter()
            .enumerate()
            .map(|(rank, candidate)| (candidate.coord, rank))
            .collect::<HashMap<_, _>>();
        let turn_start_candidates = (self.width.vcf_pair_complete
            && pair.is_none()
            && matches!(state.phase(), TurnPhase::FirstStone)
            && threats::placements_remaining(state) == 2)
            .then(|| {
                let mut coords = candidates
                    .iter()
                    .map(|candidate| candidate.coord)
                    .collect::<Vec<_>>();
                coords.sort_by_key(|coord| raw_coord_key(*coord));
                coords
            });
        // Wide mode is a VCF search, not merely a wider unrestricted attack
        // search.  Capture the turn's first coordinate so the completed pair
        // can be rejected unless it created a new count-four (or stronger)
        // claimant window.  This also covers roots entered at SecondStone.
        let turn_first = if self.width.vcf_pair_complete {
            match state.phase() {
                TurnPhase::SecondStone { first } => Some(first),
                _ => None,
            }
        } else {
            None
        };
        for candidate in candidates {
            let Ok((result, delta)) = state.apply_with_delta(Placement {
                coord: candidate.coord,
            }) else {
                continue;
            };
            if result
                .outcome
                .is_some_and(|outcome| outcome.winner == claimant)
            {
                let mut witnesses = state
                    .board()
                    .windows()
                    .entries()
                    .filter(|entry| {
                        entry.key().contains(candidate.coord)
                            && entry.count(claimant) == 6
                            && entry.count(claimant.other()) == 0
                    })
                    .map(|entry| entry.key())
                    .collect::<Vec<_>>();
                witnesses.sort_by_key(|key| window_key_order(*key));
                let witness = witnesses.first().copied();
                state.undo(delta);
                let completion_ply = ply.checked_add(1)?;
                if completion_ply > self.semantic_horizon {
                    self.horizon_cuts = self.horizon_cuts.saturating_add(1);
                    return None;
                }
                return self.alloc_node(
                    CertNode::OrCompletion {
                        mv: candidate.coord,
                        witness: witness?,
                        completion_ply,
                    },
                    0,
                );
            }
            if let Some(first) = turn_first {
                let created = turn_created_claimant_threat(state, claimant, first, candidate.coord);
                if !created || !turn_forces_small_defender_reply(state, claimant) {
                    state.undo(delta);
                    continue;
                }
            }
            let pair_context = turn_start_candidates.as_ref().and_then(|turn_start_legal| {
                (matches!(state.phase(), TurnPhase::SecondStone { .. })).then(|| PairContext {
                    first: candidate.coord,
                    turn_start_legal: turn_start_legal.clone(),
                })
            });
            let child = self.prove(state, claimant, ply.checked_add(1)?, pair_context.as_ref());
            state.undo(delta);

            if let Some(child) = child {
                return self.alloc_node(
                    CertNode::Choice {
                        mv: candidate.coord,
                        child,
                    },
                    1,
                );
            }
            if self.hit_limit {
                return None;
            }
        }
        if self.width.consumes_quiet_turns() {
            let mut complete = Vec::new();
            state.write_legal_moves(&mut complete);

            let eligible = k_reply_eligible(state, claimant);
            // `analysis` was recomputed from this exact current/post-first
            // state immediately before dispatch to `prove_choice`. Because
            // current_player == claimant here, its opponent threat family is
            // exactly T_D(P); no second active-window walk is needed merely
            // to establish the overwhelmingly common nonurgent case.
            let urgent = eligible && analysis.opp_threat_count > 0;
            let observe_k_reply = false;

            let k_reply = (urgent && (self.k_reply_consume || observe_k_reply))
                .then(|| k_reply_kernel(state, claimant, &complete));
            if self.k_reply_consume && urgent {
                let kernel = k_reply
                    .as_ref()
                    .expect("urgent Q8 consumption computes its kernel");
                debug_assert!(kernel.eligible && kernel.urgent);
                complete.clone_from(&kernel.cells);
            }

            if let Some(pair) = pair {
                restrict_pair_candidates(&mut complete, pair);
            }
            let frame = canonical_frame(state);
            complete.sort_by_key(|coord| {
                (
                    quiet_priority.get(coord).copied().unwrap_or(usize::MAX),
                    canonical_coord_key(frame, *coord),
                )
            });
            for coord in complete {
                let Ok((result, delta)) = state.apply_with_delta(Placement { coord }) else {
                    continue;
                };
                let completion_ply = ply.checked_add(1)?;
                if result
                    .outcome
                    .is_some_and(|outcome| outcome.winner == claimant)
                {
                    let completion = (completion_ply <= self.semantic_horizon)
                        .then(|| wide_completion_node(state, claimant, coord, completion_ply))
                        .flatten();
                    state.undo(delta);
                    let node = self.alloc_node(completion?, 0);

                    return node;
                }
                if result.outcome.is_some() {
                    state.undo(delta);
                    continue;
                }
                let pair_context = turn_start_candidates.as_ref().and_then(|turn_start_legal| {
                    matches!(state.phase(), TurnPhase::SecondStone { .. }).then(|| PairContext {
                        first: coord,
                        turn_start_legal: turn_start_legal.clone(),
                    })
                });
                let child = self.prove(state, claimant, completion_ply, pair_context.as_ref());
                state.undo(delta);
                if let Some(child) = child {
                    let node = self.alloc_node(CertNode::Choice { mv: coord, child }, 1);

                    return node;
                }
                if self.hit_limit {
                    return None;
                }
            }
        }
        // Exhausting a restricted attacker set only says that this attack
        // generator found no proof.  It is deliberately not a disproof.
        None
    }

    fn prove_universal(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
        analysis: &threats::ThreatAnalysis,
        pair: Option<&PairContext>,
    ) -> Option<CertNodeId> {
        let implicit_dispatch = !matches!(state.phase(), TurnPhase::Opening)
            && analysis.opp_threat_count > 0
            && !analysis.own_win_now
            && analysis.min_hitting_set == Some(analysis.b);

        // A wide descendant defender is reachable only after a completed
        // forcing attacker turn.  Keep this invariant at the dispatcher as a
        // backstop so an opening/special-phase path can never reintroduce the
        // full-legal fallback that vcf_pair_complete is designed to exclude.
        if self.width.vcf_pair_complete && !implicit_dispatch && !self.width.consumes_ranked_zone()
        {
            return None;
        }

        // v1 Group-2 selector (design §2.4, gate-free sub-class): at an
        // eligible unforced node, run the exact append-only FHW closure and
        // emit `UniversalGroup2V1`. Any failure falls through to the
        // unchanged legacy paths below; children proven during the attempt
        // stay memoized in the local TT, so the fallback re-proves them at
        // hit cost.
        if self.group2
            && !implicit_dispatch
            && !self.emitted_dirty
            && (self.zone.enabled || self.width.consumes_ranked_zone())
            && !matches!(state.phase(), TurnPhase::Opening)
            && group2_finder_preconditions(state, claimant, analysis)
        {
            if let Some(node) = self.prove_universal_group2(state, claimant, ply) {
                return Some(node);
            }
        }

        // At the proved L1 boundary U3 lets the verifier theorem-dismiss the
        // complement without enumerating it.  At spare nodes the default-off
        // U1 generator is consumable only because U2 re-derives the zone.
        let zone = (!implicit_dispatch
            && (self.zone.enabled || self.width.consumes_ranked_zone())
            && !matches!(state.phase(), TurnPhase::Opening))
        .then(|| {
            remaining_defender_placements_for_horizon(state, claimant, self.semantic_horizon).map(
                |d| ZoneInfo {
                    d,
                    build_horizon: self.semantic_horizon,
                },
            )
        })
        .flatten();
        let mut explicit = if implicit_dispatch {
            forced_defender_replies(state, claimant, analysis.b, self.width)
        } else if let Some(zone) = zone {
            zone_initial_candidates(state, claimant, zone.d, self.zone)
        } else {
            let mut all_legal = Vec::new();
            state.write_legal_moves(&mut all_legal);
            if all_legal.is_empty() {
                return None;
            }
            all_legal
        };
        if !implicit_dispatch && zone.is_none() {
            if let Some(pair) = pair {
                restrict_pair_candidates(&mut explicit, pair);
            }
        }

        if implicit_dispatch {
            let frame = canonical_frame(state);
            explicit.sort_by_key(|coord| canonical_coord_key(frame, *coord));
        } else {
            // At spare-budget nodes every legal move remains explicit, but
            // likely defenses are searched first so a refutation can stop the
            // lazy child loop before distant quiet moves are materialized.
            let hitting = hitting_universe(state, claimant);
            let frame = canonical_frame(state);
            explicit.sort_by_key(|coord| {
                let hits = hitting.contains(coord);
                (!hits, canonical_coord_key(frame, *coord))
            });
        }
        if explicit.is_empty() {
            return None;
        }

        let turn_start_legal = ((self.zone.pair_commutation
            || (self.width.vcf_pair_complete && implicit_dispatch))
            && pair.is_none()
            && matches!(state.phase(), TurnPhase::FirstStone)
            && threats::placements_remaining(state) == 2)
            .then(|| {
                let mut legal = Vec::new();
                state.write_legal_moves(&mut legal);
                legal.sort_by_key(|coord| raw_coord_key(*coord));
                legal
            });
        let mut edges = Vec::with_capacity(explicit.len());
        for &mv in &explicit {
            let Ok((result, delta)) = state.apply_with_delta(Placement { coord: mv }) else {
                return None;
            };
            let pair_context = turn_start_legal.as_ref().and_then(|legal| {
                (result.outcome.is_none() && matches!(state.phase(), TurnPhase::SecondStone { .. }))
                    .then(|| PairContext {
                        first: mv,
                        turn_start_legal: legal.clone(),
                    })
            });
            let child = self.prove(state, claimant, ply.checked_add(1)?, pair_context.as_ref());
            state.undo(delta);
            let child = child?; // Unknown poisons the universal claim.
            edges.push(CertEdge { mv, child });
        }

        if let Some(zone) = zone {
            loop {
                let required =
                    zone_certificate_extras(state, claimant, zone.d, &edges, &self.arena)?;
                let mut added = required
                    .into_iter()
                    .filter(|mv| !explicit.contains(mv))
                    .collect::<Vec<_>>();
                if added.is_empty() {
                    break;
                }
                let frame = canonical_frame(state);
                added.sort_by_key(|coord| canonical_coord_key(frame, *coord));
                for mv in added {
                    let Ok((_result, delta)) = state.apply_with_delta(Placement { coord: mv })
                    else {
                        return None;
                    };
                    let child = self.prove(state, claimant, ply.checked_add(1)?, None);
                    state.undo(delta);
                    let child = child?;
                    explicit.push(mv);
                    edges.push(CertEdge { mv, child });
                }
            }
        }

        let commutations = turn_start_legal
            .as_ref()
            .map(|legal| pair_commutations(legal, &edges, &self.arena))
            .unwrap_or_default();
        let explicit_edge_count = edges.len();

        self.alloc_node(
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            },
            explicit_edge_count,
        )
    }

    /// G2-Z1 append-only closure with the exact §3.4 required set: seed with
    /// the current hitting universe (or the least legal cell), prove children,
    /// recompute `Required_FHW` against the frozen children, and repeat until
    /// the explicit set covers it. Emits a placeholder-proof
    /// `UniversalGroup2V1`; scalars and digests are filled by
    /// `finder_finalize_group2` after compaction.
    fn prove_universal_group2(
        &mut self,
        state: &mut RustHexoState,
        claimant: Player,
        ply: u32,
    ) -> Option<CertNodeId> {
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        legal.sort_by_key(|coord| raw_coord_key(*coord));
        if legal.is_empty() {
            return None;
        }
        let in_legal = |mv: HexCoord| {
            legal
                .binary_search_by_key(&raw_coord_key(mv), |c| raw_coord_key(*c))
                .is_ok()
        };
        let mut queue: Vec<HexCoord> = hitting_universe(state, claimant)
            .into_iter()
            .filter(|mv| in_legal(*mv))
            .collect();
        queue.sort_by_key(|coord| raw_coord_key(*coord));
        queue.dedup();
        if queue.is_empty() {
            queue.push(legal[0]);
        }
        let mut edges: Vec<CertEdge> = Vec::new();
        let mut proven: Vec<HexCoord> = Vec::new();
        // The required set is monotone in the frozen child set and bounded by
        // the finite legal set, so this loop terminates.
        loop {
            for mv in std::mem::take(&mut queue) {
                let Ok((result, delta)) = state.apply_with_delta(Placement { coord: mv }) else {
                    return None;
                };
                if result.outcome.is_some() {
                    state.undo(delta);
                    return None;
                }
                let child = self.prove(state, claimant, ply.checked_add(1)?, None);
                state.undo(delta);
                let child = child?;
                edges.push(CertEdge { mv, child });
                proven.push(mv);
            }
            let pairs: Vec<(HexCoord, CertNodeId)> =
                edges.iter().map(|edge| (edge.mv, edge.child)).collect();
            let required = crate::tss_verify_group2::finder_required_fhw(
                state,
                claimant,
                &pairs,
                &self.arena,
            )?;
            let mut missing: Vec<HexCoord> = required
                .into_iter()
                .filter(|mv| in_legal(*mv) && !proven.contains(mv))
                .collect();
            if missing.is_empty() {
                break;
            }
            missing.sort_by_key(|coord| raw_coord_key(*coord));
            missing.dedup();
            queue = missing;
        }
        edges.sort_by_key(|edge| raw_coord_key(edge.mv));
        let edge_count = edges.len();
        self.alloc_node(
            CertNode::UniversalGroup2V1(Box::new(crate::tss_verify::UniversalGroup2NodeV1 {
                edges,
                proof: crate::tss_verify::Group2ZoneV1 {
                    schema_version: 1,
                    authority: crate::tss_verify::Group2AuthorityV1::compiled(),
                    claimed_d14_budget: 0,
                    build_horizon: 0,
                    child_plan_sha256: [0u8; 32],
                    finder_summary_sha256: [0u8; 32],
                },
            })),
            edge_count,
        )
    }

    fn alloc_node(&mut self, node: CertNode, added_edges: usize) -> Option<CertNodeId> {
        if self.arena.len() >= MAX_CERT_NODES
            || self.edge_count.saturating_add(added_edges) > MAX_CERT_EDGES
        {
            self.hit_limit = true;
            return None;
        }
        if let CertNode::Universal {
            implicit_dispatch,
            zone,
            commutations,
            ..
        } = &node
        {
            if *implicit_dispatch || zone.is_some() || !commutations.is_empty() {
                self.emitted_dirty = true;
            }
        }
        let id = u32::try_from(self.arena.len()).ok()?;
        self.arena.push(node);
        self.edge_count += added_edges;
        Some(id)
    }

    fn lookup_shared(
        &mut self,
        key: &PositionKey,
        claimant: Player,
        depth: usize,
    ) -> Option<CertNodeId> {
        let proof = self.shared_tt.as_ref()?.lookup_cloned(key, claimant)?;
        self.import_cached_proof(proof, depth)
    }

    fn remember_proof(&mut self, key: PositionKey, claimant: Player, node: CertNodeId) {
        self.tt.insert(key.clone(), claimant, node);
        // Persistent promotion is aimed at reusable forcing structure.  A
        // factual leaf is cheaper to re-establish than to compact, allocate,
        // and retain, while every non-leaf fragment still owns its leaves.
        // The solve root is offered separately after final compaction.
        let promotes_structure = matches!(
            self.arena.get(node as usize),
            Some(CertNode::Choice { .. } | CertNode::Universal { .. })
        );
        if promotes_structure
            && self
                .shared_tt
                .as_ref()
                .is_some_and(|shared| shared.could_admit_minimal(&key))
        {
            if let Some(proof) = CachedProof::from_arena_limited(
                &self.arena,
                node,
                MAX_PROMOTED_FRAGMENT_NODES,
                MAX_PROMOTED_FRAGMENT_EDGES,
            ) {
                self.insert_shared(key, claimant, proof);
            }
        }
        self.observe_tt_bytes();
    }

    fn insert_shared(&mut self, key: PositionKey, claimant: Player, proof: CachedProof) {
        if let Some(shared) = self.shared_tt.as_deref_mut() {
            shared.insert(key, claimant, proof);
        }
        self.observe_tt_bytes();
    }

    fn can_admit_compact(&self, key: &PositionKey, nodes: &[CertNode]) -> bool {
        self.shared_tt
            .as_ref()
            .is_some_and(|shared| shared.could_admit_compact(key, nodes))
    }

    /// Import is atomic with respect to the live arena: every structural and
    /// resource check happens against the owned clone before any node is
    /// appended.  A fragment that does not fit is merely a cache miss.
    fn import_cached_proof(&mut self, mut proof: CachedProof, depth: usize) -> Option<CertNodeId> {
        proof.validate()?;
        if proof.resolution_t > self.semantic_horizon
            || proof
                .zone_build_t
                .is_some_and(|build_t| self.semantic_horizon > build_t)
            || depth.checked_add(proof.height)? > MAX_SEARCH_DEPTH
            || self.arena.len().checked_add(proof.nodes.len())? > MAX_CERT_NODES
            || self.edge_count.checked_add(proof.explicit_edges)? > MAX_CERT_EDGES
        {
            return None;
        }

        let base = self.arena.len();
        let final_len = base.checked_add(proof.nodes.len())?;
        u32::try_from(final_len).ok()?;
        for node in &mut proof.nodes {
            remap_node_ids_with_offset(node, base, final_len)?;
        }
        let root = offset_node_id(proof.root_node, base, final_len)?;
        self.arena.append(&mut proof.nodes);
        self.edge_count += proof.explicit_edges;
        Some(root)
    }

    fn observe_tt_bytes(&mut self) {
        let shared = self
            .shared_tt
            .as_ref()
            .map(|cache| cache.current_bytes)
            .unwrap_or(0);
        self.peak_tt_bytes = self
            .peak_tt_bytes
            .max(self.tt.current_bytes.saturating_add(shared));
    }
}

fn arena_subtree_contains_zone(arena: &[CertNode], root: CertNodeId) -> bool {
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        match arena.get(id as usize) {
            Some(CertNode::Choice { child, .. }) => stack.push(*child),
            Some(CertNode::Universal { edges, zone, .. }) => {
                if zone.is_some() {
                    return true;
                }
                stack.extend(edges.iter().map(|edge| edge.child));
            }
            Some(_) => {}
            None => return true,
        }
    }
    false
}

fn pair_commutations(
    turn_start_legal: &[HexCoord],
    parent_edges: &[CertEdge],
    arena: &[CertNode],
) -> Vec<CertCommutation> {
    let mut result = Vec::new();
    for first_edge in parent_edges {
        let Some(CertNode::Universal {
            edges: first_replies,
            ..
        }) = arena.get(first_edge.child as usize)
        else {
            continue;
        };
        for &omitted_second in turn_start_legal {
            if raw_coord_key(omitted_second) >= raw_coord_key(first_edge.mv)
                || first_replies.iter().any(|edge| edge.mv == omitted_second)
            {
                continue;
            }
            let Some(mirror_edge) = parent_edges.iter().find(|edge| edge.mv == omitted_second)
            else {
                continue;
            };
            let Some(CertNode::Universal {
                edges: mirror_replies,
                ..
            }) = arena.get(mirror_edge.child as usize)
            else {
                continue;
            };
            if mirror_replies.iter().any(|edge| edge.mv == first_edge.mv) {
                result.push(CertCommutation {
                    first: first_edge.mv,
                    omitted_second,
                    first_child: first_edge.child,
                    mirror_child: mirror_edge.child,
                });
            }
        }
    }
    result.sort_by_key(|item| {
        (
            raw_coord_key(item.first),
            raw_coord_key(item.omitted_second),
        )
    });
    result
}

fn restrict_pair_candidates(candidates: &mut Vec<HexCoord>, pair: &PairContext) {
    candidates.retain(|mv| pair_candidate_allowed(*mv, pair));
}

fn pair_candidate_allowed(mv: HexCoord, pair: &PairContext) -> bool {
    raw_coord_key(mv) > raw_coord_key(pair.first) || !pair.turn_start_legal.contains(&mv)
}

/// Reconstruct membership in the pair-complete attacker universe immediately
/// before `first` was placed.  This lets the proof-number search canonicalize
/// ordinary `(a,b)/(b,a)` pairs without pruning a second coordinate that only
/// became a count-two candidate after `first`.
fn wide_candidate_was_legal_before_first(
    state: &RustHexoState,
    claimant: Player,
    first: HexCoord,
    candidate: HexCoord,
) -> bool {
    debug_assert_eq!(state.board().get(first), Some(claimant));
    debug_assert_eq!(state.board().get(candidate), None);
    for axis in Axis::ALL {
        for offset in 0..6i16 {
            let key = WindowKey {
                start: candidate - axis.vector().scale(offset),
                axis,
            };
            let Some(entry) = state.board().windows().entry(key) else {
                continue;
            };
            let first_in_window = key.contains(first);
            let prior_claimant = entry
                .count(claimant)
                .saturating_sub(u8::from(first_in_window));
            let prior_defender = entry.count(claimant.other());
            if (prior_defender == 0 && prior_claimant >= 2)
                || (prior_claimant == 0 && prior_defender >= 4)
            {
                return true;
            }
        }
    }
    false
}

/// True when the just-completed pair created a claimant count-four-or-stronger
/// window that was not already a threat at turn start.  Any changed window is
/// incident to one of the two placements, so at most 36 O(1) store lookups are
/// needed.  Subtracting both stones reconstructs the pre-turn count exactly.
fn turn_created_claimant_threat(
    state: &RustHexoState,
    claimant: Player,
    first: HexCoord,
    second: HexCoord,
) -> bool {
    let mut inspected = Vec::with_capacity(36);
    for placed in [first, second] {
        for axis in Axis::ALL {
            for offset in 0..6i16 {
                let key = WindowKey {
                    start: placed - axis.vector().scale(offset),
                    axis,
                };
                if inspected.contains(&key) {
                    continue;
                }
                inspected.push(key);
                let Some(entry) = state.board().windows().entry(key) else {
                    continue;
                };
                if entry.active_player() != Some(claimant) || entry.count(claimant) < 4 {
                    continue;
                }
                let prior_count = entry
                    .count(claimant)
                    .saturating_sub(u8::from(key.contains(first)))
                    .saturating_sub(u8::from(key.contains(second)));
                if prior_count < 4 {
                    return true;
                }
            }
        }
    }
    false
}

/// The addendum's forcing discipline requires more than merely leaving a live
/// threat: every ensuing defender placement must stay in the small, verifier-
/// justified hitting dispatcher.  A tactical claimant leaf is already done;
/// otherwise this is exactly the dispatch boundary used by `prove_universal`.
/// Rejecting looser turns only narrows the WIN search.
fn turn_forces_small_defender_reply(state: &RustHexoState, claimant: Player) -> bool {
    let analysis = threats::analyze(state);
    winner_from_analysis(state, &analysis) == Some(claimant)
        || (!matches!(state.phase(), TurnPhase::Opening)
            && analysis.opp_threat_count > 0
            && !analysis.own_win_now
            && analysis.min_hitting_set == Some(analysis.b))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Candidate {
    coord: HexCoord,
    /// Maximum pre-placement stone count among active claimant windows that
    /// this coordinate extends.  Larger means a lower initial proof number.
    strength: u8,
    priority_class: u8,
    child_threats: usize,
    /// This move occupies an empty of an active defender count-four/five
    /// window.  Wide mode must retain such tempo-preserving blocks even when
    /// the cell is not yet in a claimant-owned window.
    defender_block: bool,
    /// Distinct count-two windows through this cell.  In pair-complete mode
    /// this is the primary ordering key within the newly admitted tier.
    pair_start_degree: usize,
    /// Nearest claimant stone, used only to break widened-tier ordering ties.
    own_proximity: i16,
    /// Count-three claimant windows this placement turns into live threats.
    /// Their pre-placement empties let SecondStone reply forcedness be derived
    /// without rescanning or mutating the engine state.
    created_threats: Vec<Vec<HexCoord>>,
}

/// The established wide ordering has exactly two attacker-width classes:
/// narrow candidates (including mandatory defender blocks) and count-two-only
/// pair builds.  Keep this derivation shared by generation and the root-tier
/// selector so the prior cannot invent a third classification.
fn wide_candidate_width_tier(candidate: &Candidate) -> u8 {
    match (candidate.defender_block, candidate.strength) {
        (true, _) | (_, 3..) => 0,
        (_, 2) => 1,
        _ => unreachable!("wide candidates are defender blocks or count>=2"),
    }
}

struct CandidateBatch {
    candidates: Vec<Candidate>,
    claimant_threats: Vec<Vec<HexCoord>>,
    defender_threats: Vec<Vec<HexCoord>>,
}

fn threat_creating_moves_with_threshold(
    state: &RustHexoState,
    claimant: Player,
    minimum_strength: u8,
) -> CandidateBatch {
    assert!(
        minimum_strength >= 2,
        "count-one/r3 attacker width is not supported"
    );
    let mut candidates: Vec<Candidate> = Vec::new();
    // Coordinate-keyed dedup index. Aggregation per encounter is identical to
    // the previous linear `find`; only the lookup cost changes (the final
    // deterministic sort below fixes the output order either way).
    let mut candidate_index: HashMap<HexCoord, usize> = HashMap::new();
    let mut claimant_threats = Vec::new();
    let mut defender_threats = Vec::new();
    for entry in state.board().windows().entries() {
        let Some(owner) = entry.active_player() else {
            continue;
        };
        if entry.count(owner) >= 4 {
            if owner == claimant {
                claimant_threats.push(entry.empty_cells());
            } else {
                defender_threats.push(entry.empty_cells());
            }
        }
        if owner != claimant {
            continue;
        }
        let strength = entry.count(claimant);
        if strength < minimum_strength {
            continue;
        }
        let empties = entry.empty_cells();
        for &coord in &empties {
            let created = (strength == 3).then(|| {
                empties
                    .iter()
                    .copied()
                    .filter(|empty| *empty != coord)
                    .collect::<Vec<_>>()
            });
            if let Some(&slot) = candidate_index.get(&coord) {
                let existing = &mut candidates[slot];
                existing.strength = existing.strength.max(strength);
                if strength == 2 {
                    existing.pair_start_degree += 1;
                }
                if let Some(created) = created {
                    existing.created_threats.push(created);
                }
            } else {
                candidate_index.insert(coord, candidates.len());
                candidates.push(Candidate {
                    coord,
                    strength,
                    priority_class: u8::MAX,
                    child_threats: 0,
                    defender_block: false,
                    pair_start_degree: usize::from(strength == 2),
                    own_proximity: i16::MAX,
                    created_threats: created.into_iter().collect(),
                });
            }
        }
    }
    if minimum_strength == 2 {
        for coord in defender_threats.iter().flatten().copied() {
            if let Some(&slot) = candidate_index.get(&coord) {
                candidates[slot].defender_block = true;
            } else {
                candidate_index.insert(coord, candidates.len());
                candidates.push(Candidate {
                    coord,
                    strength: 0,
                    priority_class: u8::MAX,
                    child_threats: 0,
                    defender_block: true,
                    pair_start_degree: 0,
                    own_proximity: i16::MAX,
                    created_threats: Vec::new(),
                });
            }
        }
    }
    candidates.sort_by_key(|item| (Reverse(item.strength), item.coord.q, item.coord.r));
    CandidateBatch {
        candidates,
        claimant_threats,
        defender_threats,
    }
}

/// Static proof-number initialization derived from WindowStore membership.
/// The candidate set is unchanged.  A count-four extension is an immediate
/// lambda-one proof after a one-stone remainder; otherwise same-turn builds
/// precede replies, and newly created threat-window count orders each class.
fn ordered_threat_creating_moves(state: &RustHexoState, claimant: Player) -> Vec<Candidate> {
    ordered_threat_creating_moves_with_width(state, claimant, WidthOptions::default())
}

fn ordered_threat_creating_moves_with_width(
    state: &RustHexoState,
    claimant: Player,
    width: WidthOptions,
) -> Vec<Candidate> {
    let CandidateBatch {
        mut candidates,
        claimant_threats,
        defender_threats,
    } = if width.vcf_pair_complete {
        threat_creating_moves_with_threshold(state, claimant, 2)
    } else {
        threat_creating_moves_with_threshold(state, claimant, 3)
    };
    // Hoisted once per generation: the claimant stone list only depends on the
    // position, not on the candidate being ranked.
    let claimant_stones: Vec<HexCoord> = if width.vcf_pair_complete {
        state
            .board()
            .occupied_cells()
            .iter()
            .copied()
            .filter(|coord| state.board().get(*coord) == Some(claimant))
            .collect()
    } else {
        Vec::new()
    };
    for candidate in &mut candidates {
        candidate.child_threats = claimant_threats.len() + candidate.created_threats.len();
        if width.vcf_pair_complete && candidate.strength <= 2 {
            candidate.own_proximity = claimant_stones
                .iter()
                .map(|&coord| hex_distance(candidate.coord, coord))
                .min()
                .unwrap_or(i16::MAX);
        }
        candidate.priority_class = if candidate.defender_block && candidate.strength < 4 {
            match state.phase() {
                TurnPhase::FirstStone => 1,
                TurnPhase::SecondStone { .. } => 2,
                TurnPhase::Opening => 3,
            }
        } else {
            match state.phase() {
                TurnPhase::FirstStone if candidate.strength >= 4 => 0,
                TurnPhase::FirstStone => 2,
                TurnPhase::SecondStone { .. } => {
                    post_turn_reply_priority(candidate, &claimant_threats, &defender_threats)
                }
                TurnPhase::Opening => 3,
            }
        };
    }
    if candidates.len() <= 1 {
        return candidates;
    }
    let frame = canonical_frame(state);
    if width.vcf_pair_complete {
        candidates.sort_by_key(|item| {
            let width_tier = wide_candidate_width_tier(item);
            let canonical = canonical_coord_key(frame, item.coord);
            (
                width_tier,
                if width_tier == 0 {
                    item.priority_class
                } else {
                    0
                },
                Reverse(if width_tier == 0 {
                    item.child_threats
                } else {
                    item.pair_start_degree
                }),
                Reverse(if width_tier == 0 { item.strength } else { 0 }),
                if width_tier == 0 && matches!(state.phase(), TurnPhase::SecondStone { .. }) {
                    item.pair_start_degree
                } else {
                    0
                },
                if width_tier == 0 {
                    0
                } else {
                    item.own_proximity
                },
                canonical.0,
                canonical.1,
            )
        });
    } else {
        candidates.sort_by_key(|item| {
            let canonical = canonical_coord_key(frame, item.coord);
            (
                item.priority_class,
                Reverse(item.child_threats),
                Reverse(item.strength),
                canonical.0,
                canonical.1,
            )
        });
    }
    candidates
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct WindowGenerationMemoKey {
    key: WindowKey,
    player0_mask: u8,
    player1_mask: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompactEmpties {
    cells: [HexCoord; 5],
    len: u8,
}

impl CompactEmpties {
    fn from_entry(entry: hexo_engine::WindowEntry) -> Self {
        let mut cells = [HexCoord::ZERO; 5];
        let mut len = 0usize;
        let mask = entry.empty_mask();
        for index in 0..6u8 {
            if mask & (1 << index) != 0 {
                debug_assert!(len < cells.len(), "active window has at least one stone");
                cells[len] = entry.key().coord_at(index);
                len += 1;
            }
        }
        Self {
            cells,
            len: len as u8,
        }
    }

    fn iter(&self) -> std::slice::Iter<'_, HexCoord> {
        self.cells[..self.len as usize].iter()
    }

    fn contains(&self, cell: &HexCoord) -> bool {
        self.iter().any(|candidate| candidate == cell)
    }

    fn as_slice(&self) -> &[HexCoord] {
        &self.cells[..self.len as usize]
    }
}

#[derive(Clone, Copy)]
struct WindowGenerationMemoSlot {
    key: WindowGenerationMemoKey,
    empties: CompactEmpties,
}

struct WindowGenerationMemo {
    slots: Vec<Option<WindowGenerationMemoSlot>>,
    lookups: u64,
    hits: u64,
}

impl Default for WindowGenerationMemo {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            lookups: 0,
            hits: 0,
        }
    }
}

impl WindowGenerationMemo {
    const SLOT_COUNT: usize = 32_768;

    fn empties(&mut self, entry: hexo_engine::WindowEntry) -> CompactEmpties {
        let key = WindowGenerationMemoKey {
            key: entry.key(),
            player0_mask: entry.mask(Player::Player0),
            player1_mask: entry.mask(Player::Player1),
        };
        if self.slots.is_empty() {
            self.slots.resize(Self::SLOT_COUNT, None);
        }
        let axis = u64::from(key.key.axis.index());
        let packed = u64::from(key.key.start.q as u16)
            | (u64::from(key.key.start.r as u16) << 16)
            | (axis << 32)
            | (u64::from(key.player0_mask) << 40)
            | (u64::from(key.player1_mask) << 48);
        let mixed = packed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (packed >> 29);
        let index = mixed as usize & (Self::SLOT_COUNT - 1);
        self.lookups = self.lookups.saturating_add(1);
        if let Some(cached) = self.slots[index].filter(|slot| slot.key == key) {
            self.hits = self.hits.saturating_add(1);
            #[cfg(debug_assertions)]
            {
                debug_assert_eq!(cached.empties, CompactEmpties::from_entry(entry));
            }
            return cached.empties;
        }
        let fresh = CompactEmpties::from_entry(entry);
        self.slots[index] = Some(WindowGenerationMemoSlot {
            key,
            empties: fresh,
        });
        fresh
    }
}

/// One claimant-pure count>=2 window as seen from a candidate empty cell:
/// the immutable turn-start facts needed to evaluate any pair through it.
#[derive(Clone)]
struct WidePairWindow {
    key: WindowKey,
    strength: u8,
    empties: CompactEmpties,
}

/// A post-pair live window has at most two empty cells: every retained family
/// member started at count at least two and gained the two placements needed
/// to reach count four. Two placements touch at most 36 distinct windows, so
/// pair classification needs no heap-backed family or universe containers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PairFamilyMember {
    key: WindowKey,
    cells: [HexCoord; 2],
    len: u8,
}

impl PairFamilyMember {
    const EMPTY: Self = Self {
        key: WindowKey {
            start: HexCoord::ZERO,
            axis: Axis::Q,
        },
        cells: [HexCoord::ZERO; 2],
        len: 0,
    };

    fn after_pair(window: &WidePairWindow, first: HexCoord, second: HexCoord) -> Self {
        let mut out = Self {
            key: window.key,
            ..Self::EMPTY
        };
        for &cell in window.empties.iter() {
            if cell == first || cell == second {
                continue;
            }
            let index = usize::from(out.len);
            debug_assert!(
                index < out.cells.len(),
                "post-pair threat has more than two empties"
            );
            out.cells[index] = cell;
            out.len += 1;
        }
        out
    }

    fn as_slice(&self) -> &[HexCoord] {
        &self.cells[..usize::from(self.len)]
    }

    fn contains(&self, cell: &HexCoord) -> bool {
        self.as_slice().contains(cell)
    }
}

struct PairEvaluationScratch {
    family: [PairFamilyMember; 36],
    family_len: usize,
    universe: [HexCoord; 72],
    universe_len: usize,
}

impl Default for PairEvaluationScratch {
    fn default() -> Self {
        Self {
            family: [PairFamilyMember::EMPTY; 36],
            family_len: 0,
            universe: [HexCoord::ZERO; 72],
            universe_len: 0,
        }
    }
}

impl PairEvaluationScratch {
    fn clear(&mut self) {
        self.family_len = 0;
        self.universe_len = 0;
    }

    fn push(&mut self, member: PairFamilyMember) {
        debug_assert!(
            self.family_len < self.family.len(),
            "pair touches at most 36 windows"
        );
        self.family[self.family_len] = member;
        self.family_len += 1;
    }

    fn family(&self) -> &[PairFamilyMember] {
        &self.family[..self.family_len]
    }

    fn family_mut(&mut self) -> &mut [PairFamilyMember] {
        &mut self.family[..self.family_len]
    }
}

/// Turn-start snapshot for stateless wide pair classification. Valid only at
/// a claimant FirstStone Choice node, where `expand` has already proven that
/// no live claimant >=4 window exists (such nodes become win-now leaves
/// before any pair generation). Consequences used throughout: no pair can
/// complete six this turn (a window would need >=4 prior stones), and the
/// defender's post-pair threat family is exactly the windows through the two
/// placed stones that reach count >=4.
struct WideTurnGate {
    /// Pair-complete first-stone candidates built during the same window scan
    /// as the gate. This avoids immediately rescanning WindowStore and
    /// rebuilding the same per-cell index in `attack_pair_children`.
    first_candidates: Vec<Candidate>,
    /// Every claimant-pure count>=2 window, in board window-entry scan order.
    /// Stored once; the per-cell maps hold indices, so gate construction never
    /// clones a window per empty cell.
    windows: Vec<WidePairWindow>,
    /// Every claimant-pure count-1 window, same order discipline.
    weak_windows: Vec<WidePairWindow>,
    /// For each empty cell: indices into `windows` of the count>=2 windows
    /// holding it (scan order — identical to the historical per-cell clones).
    windows_by_cell: CoordMap<Vec<u32>>,
    /// For each empty cell: indices into `weak_windows` of the count-1
    /// windows holding it. After a first stone in such a window its other
    /// empties join the count>=2 second-ply universe; stored separately so
    /// pair evaluation never scans them.
    weak_windows_by_cell: CoordMap<Vec<u32>>,
    /// Indices into `windows` of every claimant count>=3 window: the windows
    /// that reach count>=4 from one new stone. Drives the exact pair
    /// prefilter (`pair_prefilter`).
    c3_windows: Vec<u32>,
    /// Sorted per-cell incidence counts for `c3_windows`. Pair prefiltering
    /// starts from this index and subtracts only the first-incident windows,
    /// instead of rescanning every count-three window for every first stone.
    c3_degree: Vec<((i16, i16), u8)>,
    c3_max_degree: u8,
    /// Empties of every live defender >=4 window (the hit/block sets).
    defender_threats: Vec<CompactEmpties>,

    /// `placements_made` at turn start.
    start_placements: u32,
}

/// Deterministic low-cost hasher for the tiny, solve-local coordinate set in
/// `second_candidates`. `HexCoord` hashes its two i16 fields; the fallback
/// byte path keeps the implementation correct if that representation changes.
#[derive(Default)]
struct CoordHasher(u64);

impl Hasher for CoordHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0.rotate_left(5) ^ u64::from(byte)).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }

    fn write_i16(&mut self, value: i16) {
        self.0 =
            (self.0.rotate_left(5) ^ u64::from(value as u16)).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

type CoordSet = HashSet<HexCoord, BuildHasherDefault<CoordHasher>>;
type CoordMap<V> = HashMap<HexCoord, V, BuildHasherDefault<CoordHasher>>;

impl WideTurnGate {
    fn build(state: &RustHexoState, claimant: Player) -> Self {
        Self::build_inner(state, claimant, None)
    }

    fn build_memoized(
        state: &RustHexoState,
        claimant: Player,
        memo: &mut WindowGenerationMemo,
    ) -> Self {
        Self::build_inner(state, claimant, Some(memo))
    }

    fn build_inner(
        state: &RustHexoState,
        claimant: Player,
        mut memo: Option<&mut WindowGenerationMemo>,
    ) -> Self {
        debug_assert!(matches!(state.phase(), TurnPhase::FirstStone));
        let mut windows: Vec<WidePairWindow> = Vec::new();
        let mut weak_windows: Vec<WidePairWindow> = Vec::new();
        let mut windows_by_cell: CoordMap<Vec<u32>> = CoordMap::default();
        let mut weak_windows_by_cell: CoordMap<Vec<u32>> = CoordMap::default();
        let mut c3_windows: Vec<u32> = Vec::new();
        let mut defender_threats = Vec::new();
        let mut first_candidates = Vec::<Candidate>::new();
        let mut first_candidate_index = CoordMap::<usize>::default();
        let mut claimant_threat_count = 0usize;

        for entry in state.board().windows().entries() {
            let Some(owner) = entry.active_player() else {
                continue;
            };
            let count = entry.count(owner);
            if owner == claimant {
                if count >= 1 {
                    let window = WidePairWindow {
                        key: entry.key(),
                        strength: count,
                        empties: memo
                            .as_deref_mut()
                            .map(|memo| memo.empties(entry))
                            .unwrap_or_else(|| CompactEmpties::from_entry(entry)),
                    };

                    if count >= 4 {
                        claimant_threat_count += 1;
                    }
                    if count >= 2 {
                        for &coord in window.empties.iter() {
                            let creates_threat = count == 3;
                            #[cfg(debug_assertions)]
                            let created = creates_threat.then(|| {
                                window
                                    .empties
                                    .iter()
                                    .copied()
                                    .filter(|empty| *empty != coord)
                                    .collect::<Vec<_>>()
                            });
                            if let Some(&slot) = first_candidate_index.get(&coord) {
                                let candidate = &mut first_candidates[slot];
                                candidate.strength = candidate.strength.max(count);
                                candidate.child_threats += usize::from(creates_threat);
                                if count == 2 {
                                    candidate.pair_start_degree += 1;
                                }
                                #[cfg(debug_assertions)]
                                if let Some(created) = created {
                                    candidate.created_threats.push(created);
                                }
                            } else {
                                #[cfg(debug_assertions)]
                                let created_threats = created.into_iter().collect();
                                #[cfg(not(debug_assertions))]
                                let created_threats = Vec::<Vec<HexCoord>>::new();
                                first_candidate_index.insert(coord, first_candidates.len());
                                first_candidates.push(Candidate {
                                    coord,
                                    strength: count,
                                    priority_class: u8::MAX,
                                    child_threats: usize::from(creates_threat),
                                    defender_block: false,
                                    pair_start_degree: usize::from(count == 2),
                                    own_proximity: i16::MAX,
                                    created_threats,
                                });
                            }
                        }
                    }
                    let (store, sink) = if count >= 2 {
                        (&mut windows, &mut windows_by_cell)
                    } else {
                        (&mut weak_windows, &mut weak_windows_by_cell)
                    };
                    let index = u32::try_from(store.len()).expect("window index fits u32");
                    if count >= 3 {
                        c3_windows.push(index);
                    }
                    for &cell in window.empties.iter() {
                        sink.entry(cell).or_default().push(index);
                    }
                    store.push(window);
                }
            } else if count >= 4 {
                let empties = memo
                    .as_deref_mut()
                    .map(|memo| memo.empties(entry))
                    .unwrap_or_else(|| CompactEmpties::from_entry(entry));
                for &coord in empties.iter() {
                    if let Some(&slot) = first_candidate_index.get(&coord) {
                        first_candidates[slot].defender_block = true;
                    } else {
                        first_candidate_index.insert(coord, first_candidates.len());
                        first_candidates.push(Candidate {
                            coord,
                            strength: 0,
                            priority_class: u8::MAX,
                            child_threats: 0,
                            defender_block: true,
                            pair_start_degree: 0,
                            own_proximity: i16::MAX,
                            created_threats: Vec::new(),
                        });
                    }
                }
                defender_threats.push(empties);
            }
        }
        let claimant_stones = state
            .board()
            .occupied_cells()
            .iter()
            .copied()
            .filter(|coord| state.board().get(*coord) == Some(claimant))
            .collect::<Vec<_>>();
        for candidate in &mut first_candidates {
            candidate.child_threats += claimant_threat_count;
            if candidate.strength <= 2 {
                candidate.own_proximity = claimant_stones
                    .iter()
                    .map(|&coord| hex_distance(candidate.coord, coord))
                    .min()
                    .unwrap_or(i16::MAX);
            }
            candidate.priority_class = if candidate.defender_block && candidate.strength < 4 {
                1
            } else if candidate.strength >= 4 {
                0
            } else {
                2
            };
        }
        if first_candidates.len() > 1 {
            let frame = canonical_frame(state);
            first_candidates.sort_by_key(|item| {
                let width_tier = wide_candidate_width_tier(item);
                let canonical = canonical_coord_key(frame, item.coord);
                (
                    width_tier,
                    if width_tier == 0 {
                        item.priority_class
                    } else {
                        0
                    },
                    Reverse(if width_tier == 0 {
                        item.child_threats
                    } else {
                        item.pair_start_degree
                    }),
                    Reverse(if width_tier == 0 { item.strength } else { 0 }),
                    0usize,
                    if width_tier == 0 {
                        0
                    } else {
                        item.own_proximity
                    },
                    canonical.0,
                    canonical.1,
                )
            });
        }
        let mut c3_degree_by_cell = CoordMap::<u8>::default();
        for &index in &c3_windows {
            for &cell in windows[index as usize].empties.iter() {
                let degree = c3_degree_by_cell.entry(cell).or_insert(0);
                *degree = degree.saturating_add(1);
            }
        }
        let mut c3_degree = c3_degree_by_cell
            .into_iter()
            .map(|(cell, degree)| (raw_coord_key(cell), degree))
            .collect::<Vec<_>>();
        c3_degree.sort_unstable_by_key(|&(key, _)| key);
        let c3_max_degree = c3_degree
            .iter()
            .map(|&(_, degree)| degree)
            .max()
            .unwrap_or(0);
        Self {
            first_candidates,
            windows,
            weak_windows,
            windows_by_cell,
            weak_windows_by_cell,
            c3_windows,
            c3_degree,
            c3_max_degree,
            defender_threats,

            start_placements: state.placements_made(),
        }
    }

    /// Exact pair prefilter over `evaluate_pair`'s outcome. The pair family
    /// has size k1 + j(second) + c(second), where k1 = count>=3 windows
    /// through `first` (second-independent), j = count-2 windows through both
    /// stones (2+1+1 reaches four), and c = count>=3 windows through `second`
    /// not containing `first`. `evaluate_pair` returns `None` whenever the
    /// family is EMPTY (its first gate) — and also whenever the family is a
    /// SINGLE window, because a lone post-pair window always keeps at least
    /// one empty (no count>=4 window exists at generation, so strength<=3,
    /// and a strength-3 joint window retains 3-2=1 empties), making its
    /// min hitting set exactly 1: neither the `mhs.is_none()` nor the
    /// `mhs == Some(2)` arm fires. Acceptance therefore requires family
    /// size >= 2.
    ///
    /// With k1 >= 2 every second already meets the bound — returns true, no
    /// filtering. Otherwise fills `allow` with the sorted keys of exactly
    /// the seconds with k1 + j + c >= 2: every skipped pair is a proven
    /// `None` evaluation.
    fn pair_prefilter(
        &self,
        first: HexCoord,
        counts: &mut Vec<((i16, i16), i16)>,
        allow: &mut Vec<(i16, i16)>,
    ) -> bool {
        counts.clear();
        allow.clear();
        let mut k1 = 0u8;
        let first_list = self.windows_by_cell.get(&first);
        if let Some(list) = first_list {
            for &index in list {
                if self.windows[index as usize].strength >= 3 {
                    k1 = k1.saturating_add(1);
                }
            }
        }
        if k1 >= 2 {
            return true;
        }
        // A defender-only first has no joint contribution. If no cell is in
        // two count-three windows, every pair leaves a singleton family and
        // is an exact `None`; reject the first before any window scan.
        if first_list.is_none() && self.c3_max_degree < 2 {
            return false;
        }
        counts.extend(
            self.c3_degree
                .iter()
                .map(|&(key, degree)| (key, i16::from(degree))),
        );
        // Joint contributions: count-2 windows through `first` reach four
        // exactly when `second` is one of their other empties. (Count>=3
        // windows through `first` are already in k1 and contribute the same
        // single family entry with or without jointness.)
        if let Some(list) = first_list {
            for &index in list {
                let window = &self.windows[index as usize];
                if window.strength == 2 {
                    for &cell in window.empties.iter() {
                        if cell != first {
                            counts.push((raw_coord_key(cell), 1));
                        }
                    }
                }
            }
        }
        // Remove the globally indexed count-three incidences through `first`:
        // evaluate_pair collects those windows from the first list (the k1
        // term) and deliberately skips them in its second pass.
        if let Some(list) = first_list {
            for &index in list {
                let window = &self.windows[index as usize];
                if window.strength >= 3 {
                    for &cell in window.empties.iter() {
                        counts.push((raw_coord_key(cell), -1));
                    }
                }
            }
        }
        counts.sort_unstable_by_key(|&(key, _)| key);
        let need = i16::from(2u8.saturating_sub(k1));
        let mut cursor = 0usize;
        while cursor < counts.len() {
            let key = counts[cursor].0;
            let mut sum = 0i16;
            while cursor < counts.len() && counts[cursor].0 == key {
                sum += counts[cursor].1;
                cursor += 1;
            }
            if sum >= need {
                allow.push(key);
            }
        }
        false
    }

    /// Exact defender-threat second-stone constraint over `evaluate_pair`'s
    /// `defender_win_now` check. Returns true when at least one live defender
    /// >=4 window misses `first`; `allow` then holds the intersection of
    /// those windows' empties (each has <=2, so the intersection has <=2) —
    /// exactly the seconds that hit every unhit threat. Any other second
    /// leaves an unhit defender window and `evaluate_pair` returns `None`
    /// via `defender_win_now`; an empty intersection therefore rejects every
    /// pair through `first`.
    fn defender_second_constraint(&self, first: HexCoord, allow: &mut Vec<HexCoord>) -> bool {
        allow.clear();
        let mut constrained = false;
        for set in &self.defender_threats {
            if set.contains(&first) {
                continue;
            }
            if !constrained {
                constrained = true;
                allow.extend(set.iter().copied());
            } else {
                allow.retain(|cell| set.contains(cell));
            }
        }
        constrained
    }

    /// Whether the first placement alone already buys the whole defender
    /// turn. At a generated FirstStone node no claimant >=4 window exists, so
    /// the post-first family consists exactly of turn-start count-three
    /// windows through `first`. The claimant stone also answers a defender
    /// win-now window exactly when it belongs to that window's empty set.
    fn first_is_turn_buying(&self, first: HexCoord) -> bool {
        if self
            .defender_threats
            .iter()
            .any(|empties| !empties.contains(&first))
        {
            return false;
        }
        let family = self
            .windows_by_cell
            .get(&first)
            .into_iter()
            .flatten()
            .filter_map(|&index| {
                let window = &self.windows[index as usize];
                (window.strength == 3).then(|| {
                    (
                        window.key,
                        window
                            .empties
                            .iter()
                            .copied()
                            .filter(|&cell| cell != first)
                            .collect::<Vec<_>>(),
                    )
                })
            })
            .collect::<Vec<_>>();
        matches!(wide_family_min_hitting_set(&family), Some(2) | None)
    }

    /// Append J2near cells outside the already-built exact second universe.
    /// This remains stateless: count-one windows after `first` are precisely
    /// (a) turn-start count-one windows not containing `first`, plus (b) the
    /// at most 18 previously empty windows through `first`.
    fn append_j2near_after_turn_buying_first<S: std::hash::BuildHasher>(
        &self,
        state: &RustHexoState,
        first: HexCoord,
        out: &mut Vec<HexCoord>,
        seen: &mut HashSet<HexCoord, S>,
    ) {
        if !self.first_is_turn_buying(first) {
            return;
        }
        let mut support = HashMap::<HexCoord, [u8; 3]>::new();
        for window in &self.weak_windows {
            if window.empties.contains(&first) {
                continue;
            }
            add_window_axis_support(&mut support, window.key.axis, window.empties.as_slice());
        }
        for axis in Axis::ALL {
            for offset in 0..6i16 {
                let key = WindowKey {
                    start: first - axis.vector().scale(offset),
                    axis,
                };
                if state.board().windows().entry(key).is_some() {
                    continue;
                }
                let empties = key
                    .cells()
                    .into_iter()
                    .filter(|&cell| cell != first)
                    .collect::<Vec<_>>();
                add_window_axis_support(&mut support, axis, &empties);
            }
        }
        out.extend(ordered_j2near_cells(state, support, seen));
    }

    /// The second-ply candidate coordinates after the claimant plays `first`,
    /// derived without touching the engine: the strongest continuations are
    /// the other empties of the count>=2 windows through `first` (they join
    /// the tight forcing tier — the round-2 width-sorter property), then the
    /// turn-start candidate list, then the empties of count-1 windows through
    /// `first` (which reach the count-2 build tier only via `first`). This is
    /// a slight SUPERSET of the historical post-apply regeneration (cells
    /// whose defender-block status died with `first` are retained); wider is
    /// WIN-sound and the forcing gate discards non-forcing pairs anyway.
    fn second_candidates(
        &self,
        first: HexCoord,
        turn_start: &[Candidate],
        out: &mut Vec<HexCoord>,
        seen: &mut CoordSet,
        promoted: &mut Vec<(u8, HexCoord)>,
        fresh: &mut Vec<HexCoord>,
    ) -> u64 {
        let capacities = (
            out.capacity(),
            seen.capacity(),
            promoted.capacity(),
            fresh.capacity(),
        );
        out.clear();
        seen.clear();
        promoted.clear();
        fresh.clear();
        seen.insert(first);
        if let Some(list) = self.windows_by_cell.get(&first) {
            for &index in list {
                let window = &self.windows[index as usize];
                for &cell in window.empties.iter() {
                    if cell != first {
                        promoted.push((window.strength, cell));
                    }
                }
            }
            promoted.sort_by_key(|&(strength, cell)| (Reverse(strength), raw_coord_key(cell)));
            for &(_, cell) in promoted.iter() {
                if seen.insert(cell) {
                    out.push(cell);
                }
            }
        }
        for candidate in turn_start {
            if seen.insert(candidate.coord) {
                out.push(candidate.coord);
            }
        }
        if let Some(list) = self.weak_windows_by_cell.get(&first) {
            for &index in list {
                let window = &self.weak_windows[index as usize];
                for &cell in window.empties.iter() {
                    if !seen.contains(&cell) {
                        fresh.push(cell);
                    }
                }
            }
            fresh.sort_by_key(|&cell| raw_coord_key(cell));
            for &cell in fresh.iter() {
                if seen.insert(cell) {
                    out.push(cell);
                }
            }
        }
        u64::from(out.capacity() != capacities.0)
            + u64::from(seen.capacity() != capacities.1)
            + u64::from(promoted.capacity() != capacities.2)
            + u64::from(fresh.capacity() != capacities.3)
    }

    /// Classify the attacker turn (first, second) exactly as the reference
    /// apply-and-analyze path did, without touching the engine state:
    ///
    /// - `None`: the turn creates no claimant >=4 window, or the defender
    ///   keeps a win-now/spare-budget reply — the forcing discipline prunes
    ///   it (`turn_created_claimant_threat` / `turn_forces_small_defender_
    ///   reply` both replicated below).
    /// - `ClaimantTactical`: the defender is 1-ply forced-lost and the sparse
    ///   LOSS leaf materializes within the semantic horizon
    ///   (`immediate_winner` + `typed_lambda_leaf` equivalents).
    /// - `Pending` + prior: a forcing turn searched normally, with the
    ///   `completed_turn_prior` numbers derived from the same family.
    fn evaluate_pair(
        &self,
        scratch: &mut PairEvaluationScratch,
        first_windows: Option<&[u32]>,
        first: HexCoord,
        second: HexCoord,
        semantic_horizon: u32,
    ) -> Option<(WidePnChildResult, WidePnPrior)> {
        scratch.clear();
        if let Some(list) = first_windows {
            for &index in list {
                let window = &self.windows[index as usize];
                let joint = window.empties.contains(&second);
                if window.strength + 1 + u8::from(joint) >= 4 {
                    scratch.push(PairFamilyMember::after_pair(window, first, second));
                }
            }
        }
        if let Some(list) = self.windows_by_cell.get(&second) {
            for &index in list {
                let window = &self.windows[index as usize];
                if window.empties.contains(&first) {
                    continue;
                }
                if window.strength + 1 >= 4 {
                    scratch.push(PairFamilyMember::after_pair(window, first, second));
                }
            }
        }
        if scratch.family_len == 0 {
            return None;
        }
        if self
            .defender_threats
            .iter()
            .any(|set| !set.contains(&first) && !set.contains(&second))
        {
            return None;
        }
        let mhs = compact_pair_family_min_hitting_set(scratch);
        let threat_count = scratch.family_len;
        if mhs.is_none() {
            scratch
                .family_mut()
                .sort_by_key(|member| window_key_order(member.key));
            let resolution = self.start_placements.saturating_add(6);
            if resolution <= semantic_horizon {
                // Tactical LOSS materialization is rare. Preserve the shared
                // L13 oracle byte-for-byte here; the ordinary Pending/None
                // pair path remains entirely stack-backed.
                let full_sets = scratch
                    .family()
                    .iter()
                    .map(|member| member.as_slice().to_vec())
                    .collect::<Vec<_>>();
                if inclusion_minimal_loss_obstruction(&full_sets, 2)
                    .is_some_and(|kept| !kept.is_empty())
                {
                    return Some((WidePnChildResult::ClaimantTactical, WidePnPrior::UNIFORM));
                }
            }
            return Some((
                WidePnChildResult::Pending,
                WidePnPrior {
                    pn: pn_from_fork_degree(threat_count),
                    dn: dn_from_tau(None),
                },
            ));
        }
        if mhs == Some(2) {
            return Some((
                WidePnChildResult::Pending,
                WidePnPrior {
                    pn: pn_from_fork_degree(threat_count),
                    dn: dn_from_tau(Some(2)),
                },
            ));
        }
        None
    }
}

fn add_window_axis_support(
    support: &mut HashMap<HexCoord, [u8; 3]>,
    axis: Axis,
    empties: &[HexCoord],
) {
    for &cell in empties {
        let counts = support.entry(cell).or_default();
        let slot = &mut counts[usize::from(axis.index())];
        *slot = slot.saturating_add(1);
    }
}

fn ordered_j2near_cells<S: std::hash::BuildHasher>(
    state: &RustHexoState,
    support: HashMap<HexCoord, [u8; 3]>,
    seen: &mut HashSet<HexCoord, S>,
) -> Vec<HexCoord> {
    let frame = canonical_frame(state);
    let mut cells = support
        .into_iter()
        .filter_map(|(cell, counts)| {
            let mut ordered = counts;
            ordered.sort_unstable_by(|left, right| right.cmp(left));
            (ordered[1] >= 4 && seen.insert(cell)).then_some((cell, ordered))
        })
        .collect::<Vec<_>>();
    cells.sort_by_key(|&(cell, counts)| {
        (
            Reverse(counts[1]),
            Reverse(counts.iter().copied().map(u16::from).sum::<u16>()),
            canonical_coord_key(frame, cell),
        )
    });
    cells.into_iter().map(|(cell, _)| cell).collect()
}

fn partial_turn_is_turn_buying(state: &RustHexoState, claimant: Player) -> bool {
    let mut family = Vec::<(WindowKey, Vec<HexCoord>)>::new();
    for entry in state.board().windows().entries() {
        let Some(owner) = entry.active_player() else {
            continue;
        };
        let count = entry.count(owner);
        if owner == claimant {
            if count >= 4 {
                family.push((entry.key(), entry.empty_cells()));
            }
        } else if count >= 4 {
            return false;
        }
    }
    matches!(wide_family_min_hitting_set(&family), Some(2) | None)
}

fn j2near_candidates_in_state(
    state: &RustHexoState,
    claimant: Player,
    seen: &mut HashSet<HexCoord>,
) -> Vec<HexCoord> {
    let mut support = HashMap::<HexCoord, [u8; 3]>::new();
    for entry in state.board().windows().entries() {
        if entry.active_player() == Some(claimant) && entry.count(claimant) == 1 {
            add_window_axis_support(&mut support, entry.key().axis, &entry.empty_cells());
        }
    }
    ordered_j2near_cells(state, support, seen)
}

/// Exact replica of the shared threat-analysis minimum hitting set at the
/// defender budget of two, over the stateless post-pair family.
fn wide_family_min_hitting_set(family: &[(WindowKey, Vec<HexCoord>)]) -> Option<u8> {
    if family.is_empty() {
        return Some(0);
    }
    if family.iter().any(|(_, set)| set.is_empty()) {
        return None;
    }
    let mut universe: Vec<HexCoord> = Vec::new();
    for (_, set) in family {
        for &cell in set {
            if !universe.contains(&cell) {
                universe.push(cell);
            }
        }
    }
    for &cell in &universe {
        if family.iter().all(|(_, set)| set.contains(&cell)) {
            return Some(1);
        }
    }
    for left in 0..universe.len() {
        for right in (left + 1)..universe.len() {
            let (x, y) = (universe[left], universe[right]);
            if family
                .iter()
                .all(|(_, set)| set.contains(&x) || set.contains(&y))
            {
                return Some(2);
            }
        }
    }
    None
}

/// Allocation-free equivalent of `wide_family_min_hitting_set` for the exact
/// post-pair bounds encoded by `PairEvaluationScratch`.
fn compact_pair_family_min_hitting_set(scratch: &mut PairEvaluationScratch) -> Option<u8> {
    if scratch.family_len == 0 {
        return Some(0);
    }
    if scratch.family().iter().any(|member| member.len == 0) {
        return None;
    }
    scratch.universe_len = 0;
    for family_index in 0..scratch.family_len {
        let member = scratch.family[family_index];
        for &cell in member.as_slice() {
            if !scratch.universe[..scratch.universe_len].contains(&cell) {
                debug_assert!(scratch.universe_len < scratch.universe.len());
                scratch.universe[scratch.universe_len] = cell;
                scratch.universe_len += 1;
            }
        }
    }
    for universe_index in 0..scratch.universe_len {
        let cell = scratch.universe[universe_index];
        if scratch.family().iter().all(|member| member.contains(&cell)) {
            return Some(1);
        }
    }
    for left in 0..scratch.universe_len {
        for right in (left + 1)..scratch.universe_len {
            let (x, y) = (scratch.universe[left], scratch.universe[right]);
            if scratch
                .family()
                .iter()
                .all(|member| member.contains(&x) || member.contains(&y))
            {
                return Some(2);
            }
        }
    }
    None
}

/// Static fork potential for an unexpanded attacker OR node. Count-three
/// extensions contribute the live threats they expose immediately; count-two
/// pair starts contribute their distinct continuation windows. The best
/// available degree is sufficient for an OR prior and is independent of hash
/// iteration because only the maximum is retained.
///
/// Single window pass. For a candidate cell `x` the wide generator derives
/// `child_threats = T + c3(x)` (T = live claimant >=4 windows, c3 = claimant
/// count-3 windows holding `x` as an empty) and `pair_start_degree = c2(x)`,
/// so the maximum over candidates is `max(T, max_x max(T + c3(x), c2(x)))`
/// whenever any candidate exists: cells appearing only in count>=4 claimant
/// windows or only as defender blocks contribute exactly `T`. Building and
/// sorting the full ranked candidate list for one scalar was the dominant
/// cost of every attacker-node prior.
fn attacker_fork_degree(state: &RustHexoState, claimant: Player) -> usize {
    let mut threat_count = 0usize;
    let mut any_candidate = false;
    let mut degrees: HashMap<HexCoord, (usize, usize)> = HashMap::new();
    for entry in state.board().windows().entries() {
        let Some(owner) = entry.active_player() else {
            continue;
        };
        let count = entry.count(owner);
        if owner == claimant {
            if count >= 4 {
                threat_count += 1;
            }
            if count >= 2 {
                any_candidate = true;
            }
            if count == 3 {
                for cell in entry.empty_cells() {
                    degrees.entry(cell).or_default().0 += 1;
                }
            } else if count == 2 {
                for cell in entry.empty_cells() {
                    degrees.entry(cell).or_default().1 += 1;
                }
            }
        } else if count >= 4 {
            // Defender-threat empties are wide-mode block candidates even
            // when no claimant window holds them.
            any_candidate = true;
        }
    }
    if !any_candidate {
        return 0;
    }
    degrees
        .values()
        .map(|&(c3, c2)| (threat_count + c3).max(c2))
        .max()
        .unwrap_or(0)
        .max(threat_count)
}

/// Reconstruct exactly the child threat-cost class after a claimant's
/// SecondStone at a reachable unresolved search node, without mutating the
/// engine state.  (`prove` removes terminal/lambda-one parents first, so a
/// strength-five completion and its off-path ordering distinctions cannot
/// reach this function.)  Window masks change only in windows containing
/// `coord`: claimant windows gain one bit and defender windows become blocked.
/// The returned classes match the former child `analyze` probe: immediate
/// claimant proof (0), fully forced two-hit reply (1), or a reply with
/// spare/counter-winning budget (3).
fn post_turn_reply_priority(
    candidate: &Candidate,
    claimant_threats: &[Vec<HexCoord>],
    defender_threats: &[Vec<HexCoord>],
) -> u8 {
    // The child is FirstStone (B=2), so any defender count-four/five not
    // blocked by this placement is win-now.  Child analysis gives it
    // precedence over claimant threats.
    if defender_threats
        .iter()
        .any(|empties| !empties.contains(&candidate.coord))
    {
        return 3;
    }
    match min_hitting_set_at_most_two(
        claimant_threats,
        &candidate.created_threats,
        candidate.coord,
    ) {
        None => 0,
        Some(2) => 1,
        Some(_) => 3,
    }
}

fn min_hitting_set_at_most_two(
    existing: &[Vec<HexCoord>],
    created: &[Vec<HexCoord>],
    placed: HexCoord,
) -> Option<u8> {
    if existing.is_empty() && created.is_empty() {
        return Some(0);
    }
    let sets = || existing.iter().chain(created.iter());
    if sets().any(|set| !set.iter().any(|coord| *coord != placed)) {
        return None;
    }
    let mut universe = Vec::new();
    for set in sets() {
        for &coord in set {
            if coord != placed && !universe.contains(&coord) {
                universe.push(coord);
            }
        }
    }
    if universe
        .iter()
        .any(|coord| sets().all(|set| set.contains(coord)))
    {
        return Some(1);
    }
    for left in 0..universe.len() {
        for right in (left + 1)..universe.len() {
            if sets().all(|set| set.contains(&universe[left]) || set.contains(&universe[right])) {
                return Some(2);
            }
        }
    }
    None
}

/// Union of empties of every live claimant threat.  At a defender node this is
/// the L1 hitting-cell universe, not a selected minimal hitting set.
fn hitting_universe(state: &RustHexoState, claimant: Player) -> Vec<HexCoord> {
    let mut cells = Vec::new();
    for (owner, entry) in state.board().windows().threats() {
        if owner == claimant {
            cells.extend(entry.empty_cells());
        }
    }
    cells.sort_by_key(|coord| (coord.q, coord.r));
    cells.dedup();
    cells
}

fn forced_defender_replies(
    state: &RustHexoState,
    claimant: Player,
    defender_budget: u8,
    width: WidthOptions,
) -> Vec<HexCoord> {
    if width.vcf_pair_complete {
        extendable_hit_kernel(state, claimant, defender_budget)
    } else {
        hitting_universe(state, claimant)
    }
}

/// Cells that can occur in a size-`budget` transversal of the claimant's live
/// threat family. At the forced boundary `tau == budget`, every omitted cell
/// leaves the defender without an extendable defense, so T6 permits the wide
/// WIN search to restrict its explicit universal replies to this kernel.
///
/// Connect-6 reaches this boundary only with budgets one and two. The fallback
/// deliberately returns the full hitting universe for any future budget so an
/// unsupported phase can lose performance but never lose a necessary reply.
fn extendable_hit_kernel(state: &RustHexoState, claimant: Player, budget: u8) -> Vec<HexCoord> {
    let family = state
        .board()
        .windows()
        .threats()
        .filter_map(|(owner, entry)| (owner == claimant).then(|| entry.empty_cells()))
        .collect::<Vec<_>>();
    extendable_hit_kernel_for_family(&family, budget)
}

fn extendable_hit_kernel_for_family(family: &[Vec<HexCoord>], budget: u8) -> Vec<HexCoord> {
    let mut universe = family.iter().flatten().copied().collect::<Vec<_>>();
    universe.sort_by_key(|coord| (coord.q, coord.r));
    universe.dedup();
    match budget {
        1 => universe
            .into_iter()
            .filter(|cell| family.iter().all(|threat| threat.contains(cell)))
            .collect(),
        2 => universe
            .iter()
            .copied()
            .filter(|cell| {
                universe.iter().copied().any(|mate| {
                    mate != *cell
                        && family
                            .iter()
                            .all(|threat| threat.contains(cell) || threat.contains(&mate))
                })
            })
            .collect(),
        _ => universe,
    }
}

fn remaining_defender_placements_for_horizon(
    state: &RustHexoState,
    claimant: Player,
    horizon: u32,
) -> Option<u32> {
    // Zone machinery only distinguishes budgets 0..=5 (>= 6 takes the full
    // legal set) and production horizons sit ~12 plies out, so any count past
    // this band signals a corrupted/degenerate horizon — bail (None => no
    // zone, full legal set) instead of walking a `u32::MAX` horizon.
    const DEFENDER_BUDGET_BAIL: u32 = 8;
    let mut ply = state.placements_made();
    if horizon < ply {
        return None;
    }
    let mut player = state.current_player();
    let mut phase = state.phase();
    let mut count = 0u32;
    while ply < horizon {
        if player != claimant {
            count = count.checked_add(1)?;
            if count > DEFENDER_BUDGET_BAIL {
                return None;
            }
        }
        match phase {
            TurnPhase::Opening => {
                player = player.other();
                phase = TurnPhase::FirstStone;
            }
            TurnPhase::FirstStone => {
                phase = TurnPhase::SecondStone {
                    first: HexCoord::ZERO,
                }
            }
            TurnPhase::SecondStone { .. } => {
                player = player.other();
                phase = TurnPhase::FirstStone;
            }
        }
        ply = ply.checked_add(1)?;
    }
    Some(count)
}

fn all_incident_windows_two_coloured(state: &RustHexoState, cell: HexCoord) -> bool {
    for axis in Axis::ALL {
        for offset in 0..6i16 {
            let key = WindowKey {
                start: cell - axis.vector().scale(offset),
                axis,
            };
            let Some(entry) = state
                .board()
                .windows()
                .entries()
                .find(|entry| entry.key() == key)
            else {
                return false;
            };
            if entry.count(Player::Player0) == 0 || entry.count(Player::Player1) == 0 {
                return false;
            }
        }
    }
    true
}

/// Finder-side mirror of the verifier's Group-2 class-rule 4 preconditions
/// (§2.3): defender to move at a nonterminal post-opening node, b in {1,2},
/// no mover win-now (conservative direct window upper bound AND the shared
/// analysis), and the exactly reconstructed k < b. This is a pre-check only:
/// the emitted certificate is still strictly re-verified.
fn group2_finder_preconditions(
    state: &RustHexoState,
    claimant: Player,
    analysis: &threats::ThreatAnalysis,
) -> bool {
    if state.is_terminal() || state.current_player() == claimant || analysis.own_win_now {
        return false;
    }
    let b = threats::placements_remaining(state);
    if !(1..=2).contains(&b) {
        return false;
    }
    let mover = state.current_player();
    let direct_win_upper = state
        .board()
        .windows()
        .entries()
        .any(|entry| entry.count(claimant) == 0 && entry.count(mover).saturating_add(b) >= 6);
    if direct_win_upper {
        return false;
    }
    // Exact k: 0 iff the claimant-threat family is empty; 1 iff every member
    // shares a common cell; else >= 2 (rejecting at both accepted budgets).
    let defender = claimant.other();
    let mut family: Vec<Vec<HexCoord>> = Vec::new();
    for entry in state.board().windows().entries() {
        if entry.count(defender) == 0 && entry.count(claimant) >= 4 {
            let empties = entry.empty_cells();
            if empties.is_empty() {
                return false;
            }
            family.push(empties);
        }
    }
    let k: u8 = if family.is_empty() {
        0
    } else {
        let mut common = family[0].clone();
        for member in &family[1..] {
            common.retain(|cell| member.contains(cell));
            if common.is_empty() {
                break;
            }
        }
        if common.is_empty() {
            2
        } else {
            1
        }
    };
    k < b
}

fn zone_initial_candidates(
    state: &RustHexoState,
    claimant: Player,
    d: u32,
    options: ZoneSearchCaps,
) -> Vec<HexCoord> {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| (coord.q, coord.r));
    if d >= 6 {
        return legal;
    }
    let defender = claimant.other();
    let mut out = hitting_universe(state, claimant);
    for entry in state.board().windows().entries() {
        let attacker_term = entry.active_player() == Some(claimant)
            && entry.count(claimant) >= if options.count2_threshold { 2 } else { 1 };
        let defender_term = entry.active_player() == Some(defender)
            && u32::from(entry.count(defender)) >= 6u32.saturating_sub(d);
        if attacker_term || defender_term {
            out.extend(entry.empty_cells());
        }
    }
    out.sort_by_key(|coord| (coord.q, coord.r));
    out.dedup();
    out.retain(|cell| {
        legal
            .binary_search_by_key(&(cell.q, cell.r), |c| (c.q, c.r))
            .is_ok()
    });
    if options.stale_area_filter {
        out.retain(|cell| !all_incident_windows_two_coloured(state, *cell));
    }
    let hitting = hitting_universe(state, claimant);
    let frame = canonical_frame(state);
    out.sort_by_key(|coord| (!hitting.contains(coord), canonical_coord_key(frame, *coord)));
    out
}

fn arena_core(arena: &[CertNode], root: CertNodeId, out: &mut Vec<HexCoord>) -> Option<()> {
    match arena.get(root as usize)? {
        CertNode::OrCompletion { mv, witness, .. } => {
            out.push(*mv);
            out.extend(witness.cells());
        }
        CertNode::Win { witness, .. } => out.extend(witness.cells()),
        CertNode::Loss { witnesses, .. } => {
            for witness in witnesses {
                out.extend(witness.cells());
            }
        }
        CertNode::Choice { mv, child } => {
            out.push(*mv);
            arena_core(arena, *child, out)?;
        }
        CertNode::Universal { edges, .. } => {
            for edge in edges {
                arena_core(arena, edge.child, out)?;
            }
        }
        CertNode::UniversalGroup2V1(node) => {
            for edge in &node.edges {
                arena_core(arena, edge.child, out)?;
            }
        }
        CertNode::FhwGateV1(gate) => {
            for edge in &gate.representatives {
                arena_core(arena, edge.child, out)?;
            }
        }
    }
    Some(())
}

fn zone_certificate_extras(
    state: &RustHexoState,
    claimant: Player,
    d: u32,
    edges: &[CertEdge],
    arena: &[CertNode],
) -> Option<Vec<HexCoord>> {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| (coord.q, coord.r));
    let mut protected = Vec::new();
    for edge in edges {
        arena_core(arena, edge.child, &mut protected)?;
    }
    let defender = claimant.other();
    for entry in state.board().windows().entries() {
        if entry.active_player() == Some(defender)
            && u32::from(entry.count(defender)).saturating_add(d) >= 6
        {
            protected.extend(entry.empty_cells());
        }
    }
    protected.sort_by_key(|coord| (coord.q, coord.r));
    protected.dedup();
    let stones = state.board().occupied_cells();
    let pending = protected
        .iter()
        .copied()
        .filter(|cell| {
            legal
                .binary_search_by_key(&(cell.q, cell.r), |c| (c.q, c.r))
                .is_err()
                && !stones.contains(cell)
        })
        .collect::<Vec<_>>();
    let mut required = protected
        .iter()
        .copied()
        .filter(|cell| {
            legal
                .binary_search_by_key(&(cell.q, cell.r), |c| (c.q, c.r))
                .is_ok()
        })
        .collect::<Vec<_>>();
    if !pending.is_empty() {
        let radius = seed_band_radius(d);
        required.extend(legal.iter().copied().filter(|cell| {
            pending
                .iter()
                .any(|target| i32::from(hex_distance(*cell, *target)) <= radius)
        }));
    }
    required.sort_by_key(|coord| (coord.q, coord.r));
    required.dedup();
    Some(required)
}

/// Choose the lexicographically least D6 image of the full semantic position.
/// Search ties are compared in this frame, so rotating/reflection-transforming
/// an input cannot change which proof-cost class is expanded first merely due
/// to raw `(q,r)` order.  The TT remains uncanonicalized and still uses exact
/// raw-position equality.
fn canonical_frame(state: &RustHexoState) -> u8 {
    let stone_count = state.board().occupied_cells().len();
    // One owner lookup per stone, not one per stone per symmetry.
    let stones: Vec<(HexCoord, u8)> = state
        .board()
        .occupied_cells()
        .iter()
        .map(|&coord| {
            (
                coord,
                player_code(state.board().get(coord).expect("occupied cell has owner")),
            )
        })
        .collect();
    let mut best_phase: Option<(u8, i32, i32)> = None;
    let mut best_stones = Vec::with_capacity(stone_count);
    let mut candidate_stones = Vec::with_capacity(stone_count);
    let mut best_symmetry = 0;
    for symmetry in 0..12u8 {
        let phase = match state.phase() {
            TurnPhase::Opening => (0, 0, 0),
            TurnPhase::FirstStone => (1, 0, 0),
            TurnPhase::SecondStone { first } => {
                let (q, r) = d6_coord_i32(first, symmetry);
                (2, q, r)
            }
        };
        candidate_stones.clear();
        candidate_stones.extend(stones.iter().map(|&(coord, owner)| {
            let (q, r) = d6_coord_i32(coord, symmetry);
            (q, r, owner)
        }));
        candidate_stones.sort_unstable();
        if best_phase
            .as_ref()
            .is_none_or(|best| (&phase, &candidate_stones) < (best, &best_stones))
        {
            best_phase = Some(phase);
            best_symmetry = symmetry;
            std::mem::swap(&mut best_stones, &mut candidate_stones);
        }
    }
    debug_assert!(best_phase.is_some(), "D6 contains identity");
    best_symmetry
}

fn canonical_coord_key(frame: u8, coord: HexCoord) -> (i32, i32) {
    d6_coord_i32(coord, frame)
}

fn raw_coord_key(coord: HexCoord) -> (i16, i16) {
    (coord.q, coord.r)
}

fn d6_coord_i32(coord: HexCoord, symmetry: u8) -> (i32, i32) {
    let mut q = i32::from(coord.q);
    let mut r = i32::from(coord.r);
    if symmetry >= 6 {
        r = -q - r;
    }
    for _ in 0..(symmetry % 6) {
        (q, r) = (-r, q + r);
    }
    (q, r)
}

// === Full-key transposition table ==========================================

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct KeyStone {
    q: i16,
    r: i16,
    owner: u8,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum KeyPhase {
    Opening,
    FirstStone,
    SecondStone { q: i16, r: i16 },
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct KeyTerminal {
    winner: u8,
    placements: u32,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PositionKey {
    stones: Vec<KeyStone>,
    current_player: u8,
    phase: KeyPhase,
    placements_made: u32,
    terminal: Option<KeyTerminal>,
}

impl PositionKey {
    fn from_state(state: &RustHexoState) -> Self {
        let mut stones: Vec<KeyStone> = state
            .board()
            .occupied_cells()
            .iter()
            .map(|coord| KeyStone {
                q: coord.q,
                r: coord.r,
                owner: player_code(state.board().get(*coord).expect("occupied cell has owner")),
            })
            .collect();
        stones.sort_by_key(|stone| (stone.q, stone.r, stone.owner));
        let phase = match state.phase() {
            TurnPhase::Opening => KeyPhase::Opening,
            TurnPhase::FirstStone => KeyPhase::FirstStone,
            TurnPhase::SecondStone { first } => KeyPhase::SecondStone {
                q: first.q,
                r: first.r,
            },
        };
        let terminal = state.terminal().map(|outcome| KeyTerminal {
            winner: player_code(outcome.winner),
            placements: outcome.placements,
        });
        Self {
            stones,
            current_player: player_code(state.current_player()),
            phase,
            placements_made: state.placements_made(),
            terminal,
        }
    }

    fn stable_hash(&self) -> u64 {
        // FNV-1a is used only for bucket selection.  Equality below, never this
        // 64-bit value, authorizes a proof hit.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for &byte in bytes {
                *hash ^= u64::from(byte);
                *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        feed(&mut hash, &[self.current_player]);
        feed(&mut hash, &self.placements_made.to_le_bytes());
        match self.phase {
            KeyPhase::Opening => feed(&mut hash, &[0]),
            KeyPhase::FirstStone => feed(&mut hash, &[1]),
            KeyPhase::SecondStone { q, r } => {
                feed(&mut hash, &[2]);
                feed(&mut hash, &q.to_le_bytes());
                feed(&mut hash, &r.to_le_bytes());
            }
        }
        match self.terminal {
            None => feed(&mut hash, &[0]),
            Some(terminal) => {
                feed(&mut hash, &[1, terminal.winner]);
                feed(&mut hash, &terminal.placements.to_le_bytes());
            }
        }
        for stone in &self.stones {
            feed(&mut hash, &stone.q.to_le_bytes());
            feed(&mut hash, &stone.r.to_le_bytes());
            feed(&mut hash, &[stone.owner]);
        }
        hash
    }

    fn heap_bytes(&self) -> usize {
        self.stones
            .capacity()
            .saturating_mul(size_of::<KeyStone>())
            .saturating_add(ALLOC_OVERHEAD)
    }
}

fn player_code(player: Player) -> u8 {
    match player {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}

#[derive(Debug)]
struct TtEntry {
    hash: u64,
    key: PositionKey,
    claimant: Player,
    node: CertNodeId,
}

#[derive(Debug)]
struct BoundedTt {
    slots: Vec<Option<TtEntry>>,
    cap: usize,
    current_bytes: usize,
    peak_bytes: usize,
    hash_mask: u64,
    replacements: u64,
    refusals: u64,
}

impl BoundedTt {
    fn new(cap: usize, hash_mask: u64) -> Self {
        let slot_count = (cap / TARGET_BYTES_PER_TT_SLOT).min(MAX_TT_SLOTS);
        if slot_count == 0 {
            return Self {
                slots: Vec::new(),
                cap,
                current_bytes: 0,
                peak_bytes: 0,
                hash_mask,
                replacements: 0,
                refusals: 0,
            };
        }
        let mut slots = Vec::with_capacity(slot_count);
        slots.resize_with(slot_count, || None);
        let base = slots
            .capacity()
            .saturating_mul(size_of::<Option<TtEntry>>())
            .saturating_add(ALLOC_OVERHEAD);
        if base > cap {
            return Self {
                slots: Vec::new(),
                cap,
                current_bytes: 0,
                peak_bytes: 0,
                hash_mask,
                replacements: 0,
                refusals: 0,
            };
        }
        Self {
            slots,
            cap,
            current_bytes: base,
            peak_bytes: base,
            hash_mask,
            replacements: 0,
            refusals: 0,
        }
    }

    fn lookup(&self, key: &PositionKey, claimant: Player) -> Option<CertNodeId> {
        if self.slots.is_empty() {
            return None;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let entry = self.slots[index].as_ref()?;
        (entry.hash == hash && entry.claimant == claimant && entry.key == *key)
            .then_some(entry.node)
    }

    fn entry_count(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    fn insert(&mut self, key: PositionKey, claimant: Player, node: CertNodeId) {
        if self.slots.is_empty() {
            return;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old_heap = self.slots[index]
            .as_ref()
            .map(|entry| entry.key.heap_bytes())
            .unwrap_or(0);
        let new_heap = key.heap_bytes();
        let candidate_bytes = self
            .current_bytes
            .saturating_sub(old_heap)
            .saturating_add(new_heap);
        if candidate_bytes > self.cap {
            self.refusals = self.refusals.saturating_add(1);
            return;
        }
        if self.slots[index]
            .as_ref()
            .is_some_and(|old| old.hash != hash || old.claimant != claimant || old.key != key)
        {
            self.replacements = self.replacements.saturating_add(1);
        }
        self.slots[index] = Some(TtEntry {
            hash,
            key,
            claimant,
            node,
        });
        self.current_bytes = candidate_bytes;
        self.peak_bytes = self.peak_bytes.max(candidate_bytes);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CachedProof {
    nodes: Vec<CertNode>,
    root_node: CertNodeId,
    explicit_edges: usize,
    commutation_count: usize,
    witness_count: usize,
    /// Maximum number of certificate edges below `root_node`.
    height: usize,
    /// Maximum exact resolution label over all contained typed leaves.
    resolution_t: u32,
    /// Minimum zone-build deadline over all contained zoned components.
    zone_build_t: Option<u32>,
}

impl CachedProof {
    fn from_arena_limited(
        arena: &[CertNode],
        root: CertNodeId,
        max_nodes: usize,
        max_edges: usize,
    ) -> Option<Self> {
        let (nodes, root_node) = compact_certificate_limited(arena, root, max_nodes, max_edges)?;
        Self::from_compact(nodes, root_node)
    }

    fn from_compact(nodes: Vec<CertNode>, root_node: CertNodeId) -> Option<Self> {
        if nodes.is_empty() || root_node as usize >= nodes.len() {
            return None;
        }
        let mut heights = vec![0usize; nodes.len()];
        let mut explicit_edges = 0usize;
        let mut commutation_count = 0usize;
        let mut witness_count = 0usize;
        let mut resolution_t = 0u32;
        let mut zone_build_t: Option<u32> = None;
        for (index, node) in nodes.iter().enumerate() {
            resolution_t = resolution_t.max(node_resolution(node));
            match node {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } => {
                    witness_count = witness_count.checked_add(1)?;
                }
                CertNode::Loss { witnesses, .. } => {
                    witness_count = witness_count.checked_add(witnesses.len())?;
                }
                CertNode::Universal { commutations, .. } => {
                    commutation_count = commutation_count.checked_add(commutations.len())?;
                }
                CertNode::Choice { .. } => {}
                // Extension nodes are never admitted to the proof caches;
                // refusing here keeps every cached fragment legacy-shaped.
                CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
            }
            if let CertNode::Universal {
                zone: Some(zone), ..
            } = node
            {
                zone_build_t = Some(
                    zone_build_t.map_or(zone.build_horizon, |old| old.min(zone.build_horizon)),
                );
            }
            heights[index] = match node {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => 0,
                CertNode::Choice { child, .. } => {
                    let child = *child as usize;
                    if child >= index {
                        return None;
                    }
                    heights[child].checked_add(1)?
                }
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    explicit_edges = explicit_edges.checked_add(edges.len())?;
                    let mut height = 0usize;
                    for edge in edges {
                        let child = edge.child as usize;
                        if child >= index {
                            return None;
                        }
                        height = height.max(heights[child].checked_add(1)?);
                    }
                    for item in commutations {
                        for child in [item.first_child, item.mirror_child] {
                            let child = child as usize;
                            if child >= index {
                                return None;
                            }
                            height = height.max(heights[child].checked_add(1)?);
                        }
                    }
                    height
                }
                CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
            };
        }
        let height = heights[root_node as usize];
        let proof = Self {
            nodes,
            root_node,
            explicit_edges,
            commutation_count,
            witness_count,
            height,
            resolution_t,
            zone_build_t,
        };
        proof.validate()?;
        Some(proof)
    }

    fn validate(&self) -> Option<()> {
        if self.nodes.is_empty()
            || self.nodes.len() > MAX_CERT_NODES
            || self.root_node as usize >= self.nodes.len()
            || self.height > MAX_CERT_DEPTH
            || self.explicit_edges > MAX_CERT_EDGES
        {
            return None;
        }
        let rebuilt = Self::from_compact_unchecked_metadata(&self.nodes, self.root_node)?;
        (rebuilt
            == (
                self.explicit_edges,
                self.commutation_count,
                self.witness_count,
                self.height,
                self.resolution_t,
                self.zone_build_t,
            ))
            .then_some(())
    }

    fn from_compact_unchecked_metadata(
        nodes: &[CertNode],
        root_node: CertNodeId,
    ) -> Option<(usize, usize, usize, usize, u32, Option<u32>)> {
        let mut heights = vec![0usize; nodes.len()];
        let mut explicit_edges = 0usize;
        let mut commutation_count = 0usize;
        let mut witness_count = 0usize;
        let mut resolution_t = 0u32;
        let mut zone_build_t: Option<u32> = None;
        for (index, node) in nodes.iter().enumerate() {
            resolution_t = resolution_t.max(node_resolution(node));
            match node {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } => {
                    witness_count = witness_count.checked_add(1)?;
                }
                CertNode::Loss { witnesses, .. } => {
                    witness_count = witness_count.checked_add(witnesses.len())?;
                }
                CertNode::Universal { commutations, .. } => {
                    commutation_count = commutation_count.checked_add(commutations.len())?;
                }
                CertNode::Choice { .. } => {}
                // Extension nodes are never admitted to the proof caches;
                // refusing here keeps every cached fragment legacy-shaped.
                CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
            }
            if let CertNode::Universal {
                zone: Some(zone), ..
            } = node
            {
                zone_build_t = Some(
                    zone_build_t.map_or(zone.build_horizon, |old| old.min(zone.build_horizon)),
                );
            }
            heights[index] = match node {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => 0,
                CertNode::Choice { child, .. } => {
                    let child = *child as usize;
                    if child >= index {
                        return None;
                    }
                    heights[child].checked_add(1)?
                }
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    explicit_edges = explicit_edges.checked_add(edges.len())?;
                    let mut height = 0usize;
                    for edge in edges {
                        let child = edge.child as usize;
                        if child >= index {
                            return None;
                        }
                        height = height.max(heights[child].checked_add(1)?);
                    }
                    for item in commutations {
                        for child in [item.first_child, item.mirror_child] {
                            let child = child as usize;
                            if child >= index {
                                return None;
                            }
                            height = height.max(heights[child].checked_add(1)?);
                        }
                    }
                    height
                }
                CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
            };
        }
        Some((
            explicit_edges,
            commutation_count,
            witness_count,
            heights[root_node as usize],
            resolution_t,
            zone_build_t,
        ))
    }

    fn heap_bytes(&self) -> usize {
        let mut bytes = allocation_bytes(self.nodes.capacity(), size_of::<CertNode>());
        for node in &self.nodes {
            match node {
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    bytes = bytes
                        .saturating_add(allocation_bytes(edges.capacity(), size_of::<CertEdge>()));
                    bytes = bytes.saturating_add(allocation_bytes(
                        commutations.capacity(),
                        size_of::<CertCommutation>(),
                    ));
                }
                // Adaptive LOSS contracts own a witness vector too — omitting
                // it made the cap admission understate real heap (Codex
                // review, cache accounting).
                CertNode::Loss { witnesses, .. } => {
                    bytes = bytes.saturating_add(allocation_bytes(
                        witnesses.capacity(),
                        size_of::<WindowKey>(),
                    ));
                }
                // Complete boxed-v3 accounting (design §2.5): extension nodes
                // never enter the cache (from_compact refuses them), but the
                // charge is exhaustive so a future admission path cannot
                // silently understate heap.
                CertNode::UniversalGroup2V1(node) => {
                    bytes = bytes.saturating_add(group2_node_heap_bytes(node));
                }
                CertNode::FhwGateV1(gate) => {
                    bytes = bytes.saturating_add(fhw_gate_heap_bytes(gate));
                }
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Choice { .. } => {}
            }
        }
        bytes
    }
}

/// Exhaustive heap charge for one boxed `UniversalGroup2NodeV1`.
fn group2_node_heap_bytes(node: &crate::tss_verify::UniversalGroup2NodeV1) -> usize {
    allocation_bytes(1, size_of::<crate::tss_verify::UniversalGroup2NodeV1>())
        .saturating_add(allocation_bytes(
            node.edges.capacity(),
            size_of::<CertEdge>(),
        ))
        .saturating_add(allocation_bytes(
            node.proof.authority.defender_path.len(),
            1,
        ))
        .saturating_add(allocation_bytes(node.proof.authority.fhw_path.len(), 1))
}

/// Exhaustive heap charge for one boxed `FhwGateNodeV1`.
fn fhw_gate_heap_bytes(gate: &crate::tss_verify::FhwGateNodeV1) -> usize {
    let mut bytes = allocation_bytes(1, size_of::<crate::tss_verify::FhwGateNodeV1>())
        .saturating_add(allocation_bytes(
            gate.representatives.capacity(),
            size_of::<CertEdge>(),
        ))
        .saturating_add(allocation_bytes(
            gate.proof.threats.capacity(),
            size_of::<WindowKey>(),
        ))
        .saturating_add(allocation_bytes(
            gate.proof.map.capacity(),
            size_of::<crate::tss_verify::FhwMapV1>(),
        ))
        .saturating_add(allocation_bytes(
            gate.proof.authority.defender_path.len(),
            1,
        ))
        .saturating_add(allocation_bytes(gate.proof.authority.fhw_path.len(), 1));
    for entry in &gate.proof.map {
        bytes = bytes
            .saturating_add(allocation_bytes(
                entry.roles.capacity(),
                size_of::<crate::tss_verify::FhwRoleClaimV1>(),
            ))
            .saturating_add(allocation_bytes(
                entry.windows.capacity(),
                size_of::<crate::tss_verify::FhwWindowClaimV1>(),
            ));
    }
    bytes
}

#[derive(Debug, PartialEq, Eq)]
struct SharedTtEntry {
    hash: u64,
    key: PositionKey,
    claimant: Player,
    proof: CachedProof,
}

impl SharedTtEntry {
    fn heap_bytes(&self) -> usize {
        self.key
            .heap_bytes()
            .saturating_add(self.proof.heap_bytes())
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SharedProofCache {
    slots: Vec<Option<SharedTtEntry>>,
    cap: usize,
    current_bytes: usize,
    peak_bytes: usize,
    hash_mask: u64,
}

impl SharedProofCache {
    fn new(cap: usize, hash_mask: u64) -> Self {
        let slot_count = (cap / TARGET_BYTES_PER_SHARED_TT_SLOT).min(MAX_TT_SLOTS);
        if slot_count == 0 {
            return Self {
                slots: Vec::new(),
                cap,
                current_bytes: 0,
                peak_bytes: 0,
                hash_mask,
            };
        }
        let mut slots = Vec::with_capacity(slot_count);
        slots.resize_with(slot_count, || None);
        let base = allocation_bytes(slots.capacity(), size_of::<Option<SharedTtEntry>>());
        if base > cap {
            return Self {
                slots: Vec::new(),
                cap,
                current_bytes: 0,
                peak_bytes: 0,
                hash_mask,
            };
        }
        Self {
            slots,
            cap,
            current_bytes: base,
            peak_bytes: base,
            hash_mask,
        }
    }

    fn reconfigure(&mut self, cap: usize, hash_mask: u64) {
        if self.cap != cap || self.hash_mask != hash_mask {
            *self = Self::new(cap, hash_mask);
        } else {
            self.peak_bytes = self.current_bytes;
        }
    }

    /// Drop every retained fragment (profile isolation on option changes).
    fn clear(&mut self) {
        *self = Self::new(self.cap, self.hash_mask);
    }

    fn lookup_cloned(&self, key: &PositionKey, claimant: Player) -> Option<CachedProof> {
        if self.slots.is_empty() {
            return None;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let entry = self.slots[index].as_ref()?;
        (entry.hash == hash && entry.claimant == claimant && entry.key == *key)
            .then(|| entry.proof.clone())
    }

    fn entry_count(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    fn could_admit_minimal(&self, key: &PositionKey) -> bool {
        self.could_admit_heap(key, allocation_bytes(1, size_of::<CertNode>()))
    }

    fn could_admit_compact(&self, key: &PositionKey, nodes: &[CertNode]) -> bool {
        let mut proof_heap = allocation_bytes(nodes.len(), size_of::<CertNode>());
        for node in nodes {
            match node {
                CertNode::Universal {
                    edges,
                    commutations,
                    ..
                } => {
                    proof_heap = proof_heap
                        .saturating_add(allocation_bytes(edges.len(), size_of::<CertEdge>()));
                    proof_heap = proof_heap.saturating_add(allocation_bytes(
                        commutations.len(),
                        size_of::<CertCommutation>(),
                    ));
                }
                CertNode::Loss { witnesses, .. } => {
                    proof_heap = proof_heap
                        .saturating_add(allocation_bytes(witnesses.len(), size_of::<WindowKey>()));
                }
                CertNode::UniversalGroup2V1(node) => {
                    proof_heap = proof_heap.saturating_add(group2_node_heap_bytes(node));
                }
                CertNode::FhwGateV1(gate) => {
                    proof_heap = proof_heap.saturating_add(fhw_gate_heap_bytes(gate));
                }
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Choice { .. } => {}
            }
        }
        self.could_admit_heap(key, proof_heap)
    }

    fn could_admit_heap(&self, key: &PositionKey, proof_heap: usize) -> bool {
        if self.slots.is_empty() {
            return false;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old_heap = self.slots[index]
            .as_ref()
            .map(SharedTtEntry::heap_bytes)
            .unwrap_or(0);
        self.current_bytes
            .saturating_sub(old_heap)
            .saturating_add(key.heap_bytes())
            .saturating_add(proof_heap)
            <= self.cap
    }

    fn insert(&mut self, key: PositionKey, claimant: Player, proof: CachedProof) {
        if self.slots.is_empty() || proof.validate().is_none() {
            return;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old_heap = self.slots[index]
            .as_ref()
            .map(SharedTtEntry::heap_bytes)
            .unwrap_or(0);
        let entry = SharedTtEntry {
            hash,
            key,
            claimant,
            proof,
        };
        let candidate_bytes = self
            .current_bytes
            .saturating_sub(old_heap)
            .saturating_add(entry.heap_bytes());
        if candidate_bytes > self.cap {
            return;
        }
        self.slots[index] = Some(entry);
        self.current_bytes = candidate_bytes;
        self.peak_bytes = self.peak_bytes.max(candidate_bytes);
    }
}

/// One immutable positive proof proposition. `key` and `claimant` are part of
/// the owned Arc so every live wide-PN handle can recheck identity without
/// cloning either the key or certificate payload.
#[derive(Debug, PartialEq, Eq)]
struct ProvenFragment {
    key: PositionKey,
    claimant: Player,
    proof: CachedProof,
}

impl ProvenFragment {
    fn heap_bytes(&self) -> usize {
        self.key
            .heap_bytes()
            .saturating_add(self.proof.heap_bytes())
            .saturating_add(size_of::<Self>())
            .saturating_add(ALLOC_OVERHEAD)
    }
}

#[derive(Debug)]
struct ProvenFragmentEntry {
    hash: u64,
    fragment: Arc<ProvenFragment>,
}

#[derive(Debug)]
struct ProvenFragmentStore {
    slots: Vec<Option<ProvenFragmentEntry>>,
    cap: usize,
    current_bytes: usize,
    peak_bytes: usize,
    hash_mask: u64,
    entry_count: usize,
    stored_nodes: usize,
    stored_edges: usize,
    admissions: u64,
    replacements: u64,
    refusals: u64,
}

impl ProvenFragmentStore {
    fn new(cap: usize, hash_mask: u64) -> Self {
        Self {
            // A fresh official-corpus solver is cold. Reserving a full direct
            // table here would steal search TT without enabling a single hit.
            // Allocate it only when a verified proof is actually promoted.
            slots: Vec::new(),
            cap,
            current_bytes: 0,
            peak_bytes: 0,
            hash_mask,
            entry_count: 0,
            stored_nodes: 0,
            stored_edges: 0,
            admissions: 0,
            replacements: 0,
            refusals: 0,
        }
    }

    fn reconfigure(&mut self, cap: usize, hash_mask: u64) {
        if self.cap != cap || self.hash_mask != hash_mask {
            *self = Self::new(cap, hash_mask);
        } else {
            self.peak_bytes = self.current_bytes;
        }
    }

    fn clear(&mut self) {
        *self = Self::new(self.cap, self.hash_mask);
    }

    fn ensure_slots(&mut self) -> bool {
        if !self.slots.is_empty() {
            return true;
        }
        let slot_count = (self.cap / TARGET_BYTES_PER_PROVEN_FRAGMENT_SLOT).min(MAX_TT_SLOTS);
        if slot_count == 0 {
            return false;
        }
        let mut slots = Vec::with_capacity(slot_count);
        slots.resize_with(slot_count, || None);
        let base = allocation_bytes(slots.capacity(), size_of::<Option<ProvenFragmentEntry>>());
        if base > self.cap {
            return false;
        }
        self.slots = slots;
        self.current_bytes = base;
        self.peak_bytes = self.peak_bytes.max(base);
        true
    }

    fn lookup(&self, key: &PositionKey, claimant: Player) -> Option<Arc<ProvenFragment>> {
        if self.slots.is_empty() {
            return None;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let entry = self.slots[index].as_ref()?;
        (entry.hash == hash && entry.fragment.claimant == claimant && entry.fragment.key == *key)
            .then(|| Arc::clone(&entry.fragment))
    }

    fn insert(&mut self, key: PositionKey, claimant: Player, proof: CachedProof) -> bool {
        if proof.validate().is_none() || !self.ensure_slots() {
            self.refusals = self.refusals.saturating_add(1);
            return false;
        }
        let hash = key.stable_hash() & self.hash_mask;
        let index = (hash as usize) % self.slots.len();
        let old = self.slots[index].as_ref();

        // Alternative proof graphs are never structurally unioned. For the
        // identical proposition replace only when the new admissible horizon
        // interval contains the old one (or the interval is identical and the
        // payload is smaller). Resolution/build intervals can otherwise be
        // incomparable, so a lexicographic choice would silently discard
        // useful warm queries. This is cache policy, not a proof-label merge.
        if let Some(old) = old.filter(|old| {
            old.hash == hash && old.fragment.claimant == claimant && old.fragment.key == key
        }) {
            let old_build = old.fragment.proof.zone_build_t.unwrap_or(u32::MAX);
            let new_build = proof.zone_build_t.unwrap_or(u32::MAX);
            let interval_dominates =
                proof.resolution_t <= old.fragment.proof.resolution_t && new_build >= old_build;
            let interval_is_strict =
                proof.resolution_t < old.fragment.proof.resolution_t || new_build > old_build;
            let new_is_better = interval_dominates
                && (interval_is_strict || proof.heap_bytes() < old.fragment.proof.heap_bytes());
            if !new_is_better {
                self.refusals = self.refusals.saturating_add(1);
                return false;
            }
        }

        let fragment = Arc::new(ProvenFragment {
            key,
            claimant,
            proof,
        });
        let new_heap = fragment.heap_bytes();
        let old_heap = old.map(|entry| entry.fragment.heap_bytes()).unwrap_or(0);
        let candidate_bytes = self
            .current_bytes
            .saturating_sub(old_heap)
            .saturating_add(new_heap);
        if candidate_bytes > self.cap {
            self.refusals = self.refusals.saturating_add(1);
            return false;
        }

        if let Some(old) = old {
            self.stored_nodes = self
                .stored_nodes
                .saturating_sub(old.fragment.proof.nodes.len());
            self.stored_edges = self
                .stored_edges
                .saturating_sub(old.fragment.proof.explicit_edges);
            self.replacements = self.replacements.saturating_add(1);
        } else {
            self.entry_count = self.entry_count.saturating_add(1);
        }
        self.stored_nodes = self.stored_nodes.saturating_add(fragment.proof.nodes.len());
        self.stored_edges = self
            .stored_edges
            .saturating_add(fragment.proof.explicit_edges);
        self.slots[index] = Some(ProvenFragmentEntry { hash, fragment });
        self.current_bytes = candidate_bytes;
        self.peak_bytes = self.peak_bytes.max(candidate_bytes);
        self.admissions = self.admissions.saturating_add(1);
        true
    }
}

fn allocation_bytes(capacity: usize, element_size: usize) -> usize {
    if capacity == 0 {
        0
    } else {
        capacity
            .saturating_mul(element_size)
            .saturating_add(ALLOC_OVERHEAD)
    }
}

fn offset_node_id(id: CertNodeId, base: usize, final_len: usize) -> Option<CertNodeId> {
    let index = id as usize;
    let mapped = base.checked_add(index)?;
    (mapped < final_len)
        .then(|| u32::try_from(mapped).ok())
        .flatten()
}

fn remap_node_ids_with_offset(node: &mut CertNode, base: usize, final_len: usize) -> Option<()> {
    match node {
        CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {}
        CertNode::Choice { child, .. } => {
            *child = offset_node_id(*child, base, final_len)?;
        }
        CertNode::Universal {
            edges,
            commutations,
            ..
        } => {
            for edge in edges {
                edge.child = offset_node_id(edge.child, base, final_len)?;
            }
            for item in commutations {
                item.first_child = offset_node_id(item.first_child, base, final_len)?;
                item.mirror_child = offset_node_id(item.mirror_child, base, final_len)?;
            }
        }
        CertNode::UniversalGroup2V1(node) => {
            for edge in &mut node.edges {
                edge.child = offset_node_id(edge.child, base, final_len)?;
            }
        }
        // Gate role rows carry node references of their own; this solver
        // never builds gates, so refuse rather than remap partially.
        CertNode::FhwGateV1(_) => return None,
    }
    Some(())
}

/// Remove abandoned OR branches from the certificate arena and remap every
/// reachable child.  The resulting certificate has no orphan nodes, which the
/// independent verifier requires.
pub(crate) fn compact_certificate(
    arena: &[CertNode],
    root: CertNodeId,
) -> Option<(Vec<CertNode>, CertNodeId)> {
    compact_certificate_limited(arena, root, MAX_CERT_NODES, MAX_CERT_EDGES)
}

fn compact_certificate_limited(
    arena: &[CertNode],
    root: CertNodeId,
    max_nodes: usize,
    max_edges: usize,
) -> Option<(Vec<CertNode>, CertNodeId)> {
    fn copy(
        old: CertNodeId,
        arena: &[CertNode],
        remap: &mut [Option<CertNodeId>],
        visiting: &mut [bool],
        out: &mut Vec<CertNode>,
        edge_count: &mut usize,
        max_nodes: usize,
        max_edges: usize,
    ) -> Option<CertNodeId> {
        let index = old as usize;
        if index >= arena.len() || visiting[index] {
            return None;
        }
        if let Some(mapped) = remap[index] {
            return Some(mapped);
        }
        visiting[index] = true;
        let mapped_node = match &arena[index] {
            CertNode::OrCompletion {
                mv,
                witness,
                completion_ply,
            } => CertNode::OrCompletion {
                mv: *mv,
                witness: *witness,
                completion_ply: *completion_ply,
            },
            CertNode::Win {
                witness,
                count,
                budget,
                resolution_ply,
            } => CertNode::Win {
                witness: *witness,
                count: *count,
                budget: *budget,
                resolution_ply: *resolution_ply,
            },
            CertNode::Loss {
                witnesses,
                resolution_ply,
            } => CertNode::Loss {
                witnesses: witnesses.clone(),
                resolution_ply: *resolution_ply,
            },
            CertNode::Choice { mv, child } => CertNode::Choice {
                mv: *mv,
                child: copy(
                    *child, arena, remap, visiting, out, edge_count, max_nodes, max_edges,
                )?,
            },
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            } => {
                *edge_count = edge_count.checked_add(edges.len())?;
                if *edge_count > max_edges {
                    return None;
                }
                let mut mapped_edges = Vec::with_capacity(edges.len());
                for edge in edges {
                    mapped_edges.push(CertEdge {
                        mv: edge.mv,
                        child: copy(
                            edge.child, arena, remap, visiting, out, edge_count, max_nodes,
                            max_edges,
                        )?,
                    });
                }
                let mut mapped_commutations = Vec::with_capacity(commutations.len());
                for item in commutations {
                    mapped_commutations.push(CertCommutation {
                        first: item.first,
                        omitted_second: item.omitted_second,
                        first_child: copy(
                            item.first_child,
                            arena,
                            remap,
                            visiting,
                            out,
                            edge_count,
                            max_nodes,
                            max_edges,
                        )?,
                        mirror_child: copy(
                            item.mirror_child,
                            arena,
                            remap,
                            visiting,
                            out,
                            edge_count,
                            max_nodes,
                            max_edges,
                        )?,
                    });
                }
                CertNode::Universal {
                    edges: mapped_edges,
                    implicit_dispatch: *implicit_dispatch,
                    zone: zone.clone(),
                    commutations: mapped_commutations,
                }
            }
            CertNode::UniversalGroup2V1(node) => {
                *edge_count = edge_count.checked_add(node.edges.len())?;
                if *edge_count > max_edges {
                    return None;
                }
                let mut mapped_edges = Vec::with_capacity(node.edges.len());
                for edge in &node.edges {
                    mapped_edges.push(CertEdge {
                        mv: edge.mv,
                        child: copy(
                            edge.child, arena, remap, visiting, out, edge_count, max_nodes,
                            max_edges,
                        )?,
                    });
                }
                CertNode::UniversalGroup2V1(Box::new(crate::tss_verify::UniversalGroup2NodeV1 {
                    edges: mapped_edges,
                    proof: node.proof.clone(),
                }))
            }
            // Gate role rows reference arena IDs; the solver never builds
            // gates, so compaction refuses rather than remapping partially.
            CertNode::FhwGateV1(_) => return None,
        };
        visiting[index] = false;
        if out.len() >= max_nodes {
            return None;
        }
        let mapped = u32::try_from(out.len()).ok()?;
        out.push(mapped_node);
        remap[index] = Some(mapped);
        Some(mapped)
    }

    if arena.len() > MAX_CERT_NODES || max_nodes > MAX_CERT_NODES || max_edges > MAX_CERT_EDGES {
        return None;
    }
    let mut remap = vec![None; arena.len()];
    let mut visiting = vec![false; arena.len()];
    let mut nodes = Vec::new();
    let mut edge_count = 0usize;
    let root_node = copy(
        root,
        arena,
        &mut remap,
        &mut visiting,
        &mut nodes,
        &mut edge_count,
        max_nodes,
        max_edges,
    )?;
    Some((nodes, root_node))
}
