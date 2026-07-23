//! tss_core.rs — typed Threat-Space Search results: the soundness seam between
//! proof producers and the search tree (docs/PLAN_TSS_DEEPENING.md §2).
//!
//! The tree is the poison channel: any hard ±1 reaching `backup_virtual`
//! propagates into the soft-policy / cell_q / stvalue training targets with no
//! head involvement. This module therefore types the seam: `HardValue` is the
//! only TSS value `backup_virtual` may receive, its field is private, and the
//! only constructors are the certified producers defined HERE:
//!
//!   1. `solve_leaf_lambda1` — the sound one-turn (λ¹) verdict, a verbatim
//!      wrapper of `threats::analyze().verdict()` (sound post-opening; see
//!      threats_shared.rs header and the design doc §1).
//!   2. `hard_value_from_verified` — deep proofs, minted only inside this
//!      module after an independent certificate verifier accepts the claim
//!      (Stage 4; the `DeepSolve` implementation itself can never mint one).
//!
//! Code outside this module cannot fabricate a `HardValue`; "deep results
//! degrade to net-eval until verified" is structural, not a runtime flag.

use hexo_engine::HexoState as RustHexoState;

use crate::threats_shared as threats;

/// Three-valued solve status. UNKNOWN must propagate — a capped / exhausted /
/// unproven solve is UNKNOWN, never a verdict (§2.4). `Loss` is a claim that
/// the SIDE TO MOVE at the solved state loses; for deep solvers that requires
/// the dual certificate (a proven opponent winning strategy, §2.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofStatus {
    Win,
    Loss,
    Unknown,
}

impl ProofStatus {
    /// The backup value for the side to move, when proven.
    #[inline]
    pub fn value(self) -> Option<f32> {
        match self {
            ProofStatus::Win => Some(1.0),
            ProofStatus::Loss => Some(-1.0),
            ProofStatus::Unknown => None,
        }
    }
}

/// A value certified to enter `backup_virtual` as a hard ±1 for the side to
/// move at the solved state. Sealed: the field is private and the only
/// constructors live in this module (the two certified producers above).
#[derive(Clone, Copy, Debug)]
pub struct HardValue(f32);

impl HardValue {
    /// The certified backup value (±1, side-to-move perspective).
    #[inline]
    pub fn value(self) -> f32 {
        self.0
    }

    #[inline]
    pub fn status(self) -> ProofStatus {
        if self.0 > 0.0 {
            ProofStatus::Win
        } else {
            ProofStatus::Loss
        }
    }
}

/// Certified producer #1 — the sound λ¹ verdict for the side to move.
/// Verbatim wrapper of `threats::analyze().verdict()`: `Some(+1)` proven win
/// within the turn budget, `Some(-1)` proven one-turn forced loss, `None`
/// (no proof) stays `None` — the net evaluates the leaf.
#[inline]
pub fn solve_leaf_lambda1(state: &RustHexoState) -> Option<HardValue> {
    threats::analyze(state).verdict().map(HardValue)
}

/// Typed status view of the λ¹ solve, for consumers that classify rather than
/// back up (the root guard / recorded-target classifier).
#[inline]
pub fn lambda1_status(state: &RustHexoState) -> ProofStatus {
    match threats::analyze(state).verdict() {
        Some(v) if v > 0.0 => ProofStatus::Win,
        Some(_) => ProofStatus::Loss,
        None => ProofStatus::Unknown,
    }
}

// === Deep-solver seam (Stage 3/4; frozen for the delegated build) ===========

/// Deterministic solve budget. No wall clock on any path that can mint a hard
/// value: a timed-out solve is UNKNOWN by construction because it never
/// completes a certificate (§2.6). Caps binding must yield UNKNOWN, never a
/// verdict.
#[derive(Clone, Copy, Debug)]
pub struct SolveCaps {
    /// Maximum solver node expansions for this solve.
    pub node_cap: u64,
    /// Hard ceiling on transposition-table + cache bytes (the WSL host kills
    /// unbounded growth; §11). The solver must account and stay under it.
    pub tt_bytes_cap: usize,
    /// Absolute placement index of the semantic proof deadline.  This is
    /// deliberately distinct from `node_cap` and the structural depth guard:
    /// zone obligations and typed leaf resolutions are statements about game
    /// plies, not about how much search work happened to be affordable.
    pub semantic_horizon: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ZoneSearchCaps {
    pub enabled: bool,
    pub stale_area_filter: bool,
    pub count2_threshold: bool,
    pub pair_commutation: bool,
}

/// Uniform D11/T4 seed-band radius. L9' bounds the first protected occupation
/// chain by `8(B-1)`; `d` is the verifier-checked admissible local B wrapper.
/// Keeping this theorem arithmetic in the shared contract module preserves
/// finder/verifier separation while giving both sides one production value.
#[inline]
pub(crate) fn seed_band_radius(d: u32) -> i32 {
    i32::try_from(d.saturating_sub(1).saturating_mul(8)).unwrap_or(i32::MAX)
}

/// Which root-perspective hard result a caller wants the deep solver to seek.
///
/// This is deliberately separate from [`SolveCaps`] so existing callers using
/// its two-field literal remain source-compatible.  `DeepSolve::solve` keeps
/// the historical [`SolveGoal::Both`] behavior; reusable solver callers may
/// request one side explicitly and give that attempt the whole node budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveGoal {
    Win,
    Loss,
    Both,
}

/// Engine / certificate-schema version stamped into per-run telemetry so
/// minted certificates are attributable to the engine that produced them
/// (PLAN_TSS_MCTS_INTEGRATION.md §3, C1 one-engine principle). `1` = the
/// original narrow Stage-3 trainer solver; `2` = the campaign wide
/// `vcf_pair_complete` engine adopted wholesale in the V0 port (R-FIX1
/// zone-clock fix, lazy defender frontier, interior census gate, incremental
/// defender enumeration, cap-resume, extended-contract zones P0–P3). Bump on
/// any change to the minting engine or the certificate grammar.
pub const TSS_CERT_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, Default)]
pub struct SolveStats {
    pub nodes: u64,
    pub expansions: u64,
    pub tt_hits: u64,
    pub tt_entries: u64,
    pub peak_tt_bytes: u64,
    /// Lines the semantic-horizon deadline refused while still alive (a live
    /// descent, typed-leaf resolution, or completion past the deadline).
    /// Distinguishes depth-bound Unknowns from structural ones ahead of any
    /// horizon-ladder decision. Telemetry only: never a value, never a
    /// backup — a heuristic escalation trigger under contract rule 7.
    pub horizon_cuts: u64,
    /// Subset of `horizon_cuts` that fell at a defender-to-move node (the
    /// opponent still branching, i.e. before the fully-forced `k == B`
    /// boundary). Feeds `deep_kb_death` on the horizon-ladder tall pass: the
    /// signal that Group-2 zone consumption would matter.
    pub kb_death_cuts: u64,
    /// Direct-map slot replacements in the solve-local TT. These are cache
    /// evictions, not proof-semantic events.
    pub tt_evictions: u64,
    /// TT/index insertions refused because the caller's byte cap was full.
    pub tt_admission_rejections: u64,
    /// Exact-key positive-fragment queries made by the wide solver.
    pub fragment_lookups: u64,
    /// Queries that passed full-key, claimant, horizon, and depth checks.
    pub fragment_hits: u64,
    /// Shared fragment roots actually imported into the returned certificate.
    pub fragment_imports: u64,
    /// Resident entries after this solve (telemetry only).
    pub fragment_store_entries: u64,
    /// Resident byte-accounted fragment-store charge after this solve.
    pub fragment_store_bytes: u64,
    pub interior_gate_evaluations: u64,
    pub interior_gate_dismissals: u64,
    pub interior_gate_nanos: u64,
}

impl SolveStats {
    /// Fold one solver attempt into a solve-level aggregate. Additive counters
    /// sum, high-water marks take their maximum, and resident-store gauges
    /// describe the most recently completed attempt.
    pub(crate) fn merge(&mut self, part: Self) {
        self.nodes = self.nodes.saturating_add(part.nodes);
        self.expansions = self.expansions.saturating_add(part.expansions);
        self.tt_hits = self.tt_hits.saturating_add(part.tt_hits);
        self.tt_entries = self.tt_entries.max(part.tt_entries);
        self.peak_tt_bytes = self.peak_tt_bytes.max(part.peak_tt_bytes);
        self.horizon_cuts = self.horizon_cuts.saturating_add(part.horizon_cuts);
        self.kb_death_cuts = self.kb_death_cuts.saturating_add(part.kb_death_cuts);
        self.tt_evictions = self.tt_evictions.saturating_add(part.tt_evictions);
        self.tt_admission_rejections = self
            .tt_admission_rejections
            .saturating_add(part.tt_admission_rejections);
        self.fragment_lookups = self.fragment_lookups.saturating_add(part.fragment_lookups);
        self.fragment_hits = self.fragment_hits.saturating_add(part.fragment_hits);
        self.fragment_imports = self.fragment_imports.saturating_add(part.fragment_imports);
        self.fragment_store_entries = part.fragment_store_entries;
        self.fragment_store_bytes = part.fragment_store_bytes;
        self.interior_gate_evaluations = self
            .interior_gate_evaluations
            .saturating_add(part.interior_gate_evaluations);
        self.interior_gate_dismissals = self
            .interior_gate_dismissals
            .saturating_add(part.interior_gate_dismissals);
        self.interior_gate_nanos = self
            .interior_gate_nanos
            .saturating_add(part.interior_gate_nanos);
    }
}

/// A deep solve's outcome: a typed status, an optional replayable certificate
/// (present for every Win/Loss claim), and diagnostics. The certificate type
/// is solver-defined; the search consumes only `status` — and only via
/// `hard_value_from_verified`, never directly.
pub struct DeepResult<C> {
    pub status: ProofStatus,
    pub cert: Option<C>,
    pub stats: SolveStats,
}

/// The deep-solver interface the Stage-3 delegated build implements
/// (docs/TSS_SOLVER_SPEC.md freezes the semantics: df-pn, exhaustive-with-
/// instant-dispatch AND nodes, threat-creating OR restriction, dual LOSS
/// certificates, UNKNOWN propagation, full-canonical-key cache equality).
pub trait DeepSolve {
    type Cert;
    fn solve(&mut self, state: &RustHexoState, caps: &SolveCaps) -> DeepResult<Self::Cert>;
}

/// The independent certificate verifier (§2.2): replays a certificate against
/// the state and accepts or rejects the claimed status. Implemented as its own
/// module sharing only engine primitives with the solver, so a solver bug is
/// not mirrored in its checker.
pub trait CertVerify {
    type Cert;
    fn verify(&self, state: &RustHexoState, cert: &Self::Cert, claimed: ProofStatus) -> bool;
}

/// Certified producer #2 — deep proofs, minted ONLY here and only after the
/// independent verifier accepts the certificate for this exact state. A
/// rejected or missing certificate yields `None` (the caller must degrade to
/// net-eval AND bump the fatal `verify_failed` telemetry counter).
///
/// The verifier parameter is the CONCRETE `TssVerifier` — not the `CertVerify`
/// trait — so no sibling module can mint a `HardValue` through an
/// always-accepting stand-in (Codex review, mint sealing). The generic
/// trait-driven variant survives as a test-only helper below.
pub fn hard_value_from_verified(
    verifier: &crate::tss_verify::TssVerifier,
    state: &RustHexoState,
    result: &DeepResult<crate::tss_verify::TssCertificate>,
) -> Option<HardValue> {
    hard_value_from_verify_impl(verifier, state, result)
}

/// Sealed sibling mint for the externally selected `Group2V1` verifier policy
/// (design §5.1). The parameter is the CONCRETE `Group2Verifier` for the same
/// reason as above: no stand-in can mint a `HardValue`. Certificates without
/// extension nodes verify byte-identically to the legacy path inside it.
pub fn hard_value_from_verified_group2(
    verifier: &crate::tss_verify::Group2Verifier,
    state: &RustHexoState,
    result: &DeepResult<crate::tss_verify::TssCertificate>,
) -> Option<HardValue> {
    hard_value_from_verify_impl(verifier, state, result)
}

/// Trait-generic mint used by `hard_value_from_verified` and (directly) by
/// tests exercising the accept/reject contract with stub verifiers. Private:
/// production callers cannot name it with a stub verifier.
fn hard_value_from_verify_impl<V, C>(
    verifier: &V,
    state: &RustHexoState,
    result: &DeepResult<C>,
) -> Option<HardValue>
where
    V: CertVerify<Cert = C>,
{
    let value = result.status.value()?;
    let cert = result.cert.as_ref()?;
    if verifier.verify(state, cert, result.status) {
        Some(HardValue(value))
    } else {
        None
    }
}
