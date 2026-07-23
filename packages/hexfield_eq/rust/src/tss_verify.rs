//! Independent verifier and replayable certificate format for deep TSS.
//!
//! This module deliberately does not depend on `tss_solver`.  A certificate is
//! checked by replaying every represented move through `HexoState` and by using
//! only the shared, one-turn lambda-1 analysis for leaves and instant dispatch.

use std::mem::size_of;

use hexo_engine::{
    hex_distance, Axis, GameOutcome, HexCoord, HexoState as RustHexoState, Placement, Player,
    TurnPhase, WindowKey,
};

use crate::threats_shared;
use crate::tss_core::{seed_band_radius, CertVerify, ProofStatus};

/// Maximum number of arena nodes accepted from one certificate.
pub const MAX_CERT_NODES: usize = 100_000;
/// Maximum total number of explicitly represented universal edges.
pub const MAX_CERT_EDGES: usize = 1_000_000;
/// Maximum total witness identities carried by typed leaves. Window keys are
/// compact, but LOSS families are attacker-controlled certificate data.
pub const MAX_CERT_WITNESSES: usize = 1_000_000;
pub const MAX_CERT_COMMUTATIONS: usize = 1_000_000;
/// Maximum replay depth.  This is also a guard against adversarially deep DAGs.
pub const MAX_CERT_DEPTH: usize = 256;
/// Maximum number of root stones encoded in a certificate binding.
pub const MAX_CERT_ROOT_STONES: usize = 1_000_000;
/// Fixed verifier memo ceiling.  The verifier trait has no solve caps, so its
/// replay cache has its own hard byte bound rather than borrowing a solver TT
/// budget.
pub const MAX_VERIFY_MEMO_BYTES: usize = 64 << 20;
/// Number of rotations/reflections in the dihedral symmetry group of the hex.
pub const D6_SYMMETRY_COUNT: u8 = 12;

/// Compact arena index.  IDs always index `TssCertificate::nodes` directly.
pub type CertNodeId = u32;

/// Exact, history-independent binding of a certificate to its root position.
/// `occupancy` is lexicographically sorted by `(q, r)` and `owners[i]` owns
/// `occupancy[i]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RootBinding {
    pub occupancy: Vec<HexCoord>,
    pub owners: Vec<Player>,
    pub current_player: Player,
    pub phase: TurnPhase,
    pub placements_made: u32,
    pub terminal: Option<GameOutcome>,
}

impl RootBinding {
    /// Construct the canonical full-position binding used by certificates.
    pub fn from_state(state: &RustHexoState) -> Self {
        let mut stones: Vec<(HexCoord, Player)> = state
            .board()
            .occupied_cells()
            .iter()
            .copied()
            .map(|coord| {
                let owner = state
                    .board()
                    .get(coord)
                    .expect("occupied_cells and Board::get must agree");
                (coord, owner)
            })
            .collect();
        stones.sort_by_key(|(coord, _)| coord_key(*coord));
        let (occupancy, owners) = stones.into_iter().unzip();
        Self {
            occupancy,
            owners,
            current_player: state.current_player(),
            phase: state.phase(),
            placements_made: state.placements_made(),
            terminal: state.terminal(),
        }
    }
}

/// One explicitly searched move at a universal (opponent) node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertEdge {
    pub mv: HexCoord,
    pub child: CertNodeId,
}

/// P3 same-turn commutation evidence attached to the turn-start Universal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CertCommutation {
    pub first: HexCoord,
    pub omitted_second: HexCoord,
    pub first_child: CertNodeId,
    pub mirror_child: CertNodeId,
}

/// Horizon-dependent data carried by a defender zone node.  `d` is evidence
/// only: the verifier always recomputes the exact remaining defender budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZoneInfo {
    pub d: u32,
    /// Semantic deadline against which this D-dependent zone was built.
    pub build_horizon: u32,
}

/// Raw SHA-256 digest bytes carried by v1 Group-2 records.
pub type Sha256Bytes = [u8; 32];

/// Compiled dual authority binding (design §1.1). Every v1 Group-2 node must
/// carry these exact six fields; any byte mismatch rejects the new class.
pub const G2_DEFENDER_COMMIT: [u8; 20] = [
    0x6d, 0xc0, 0x8d, 0x7a, 0x89, 0xd4, 0x22, 0x52, 0x4f, 0x6d, 0x92, 0xda, 0xdf, 0x66, 0x20, 0x73,
    0xd2, 0x5b, 0x19, 0x63,
];
pub const G2_DEFENDER_PATH: &str = "docs/PROOF_TSS_DEFENDER_ZONES.md";
pub const G2_DEFENDER_SHA256: Sha256Bytes = [
    0x39, 0x19, 0x74, 0x60, 0xD0, 0x68, 0xCE, 0x54, 0x42, 0xBA, 0x0A, 0xFF, 0xC6, 0x87, 0xF1, 0x40,
    0x8D, 0xF3, 0xF2, 0x8E, 0xEE, 0xB2, 0x6C, 0x4D, 0xD7, 0x19, 0x2B, 0x87, 0xA2, 0x02, 0x06, 0x4B,
];
pub const G2_FHW_COMMIT: [u8; 20] = [
    0x99, 0x45, 0xc2, 0x1b, 0xf1, 0x77, 0x05, 0x5a, 0xa4, 0xde, 0x0b, 0xbd, 0x3a, 0xad, 0x15, 0xb9,
    0xcf, 0x24, 0x5e, 0x51,
];
pub const G2_FHW_PATH: &str = "PROOF_TSS_ZONES_FHW.md";
pub const G2_FHW_SHA256: Sha256Bytes = [
    0x16, 0xF7, 0xD6, 0x84, 0xB5, 0xD7, 0x63, 0xE8, 0xB6, 0x73, 0xEC, 0x3A, 0x03, 0xB5, 0x11, 0x0B,
    0x9A, 0xBF, 0x5B, 0xB7, 0xE8, 0x0F, 0xCA, 0x06, 0x3E, 0x62, 0xC8, 0x1A, 0x11, 0x3F, 0x9E, 0xA0,
];
/// Hard cap on authority path length inside a v1 certificate (design §3.5).
pub const MAX_AUTHORITY_PATH: usize = 64;

/// Dual authority binding carried by every new-class node (design §2.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group2AuthorityV1 {
    pub defender_commit: [u8; 20],
    pub defender_path: Box<str>,
    pub defender_sha256: Sha256Bytes,
    pub fhw_commit: [u8; 20],
    pub fhw_path: Box<str>,
    pub fhw_sha256: Sha256Bytes,
}

impl Group2AuthorityV1 {
    /// The compiled dual binding. All six fields are compared byte-for-byte
    /// at verification; the constructor exists so finder and tests emit the
    /// only acceptable value.
    pub fn compiled() -> Self {
        Self {
            defender_commit: G2_DEFENDER_COMMIT,
            defender_path: G2_DEFENDER_PATH.into(),
            defender_sha256: G2_DEFENDER_SHA256,
            fhw_commit: G2_FHW_COMMIT,
            fhw_path: G2_FHW_PATH.into(),
            fhw_sha256: G2_FHW_SHA256,
        }
    }

    pub fn matches_compiled(&self) -> bool {
        self.defender_commit == G2_DEFENDER_COMMIT
            && &*self.defender_path == G2_DEFENDER_PATH
            && self.defender_sha256 == G2_DEFENDER_SHA256
            && self.fhw_commit == G2_FHW_COMMIT
            && &*self.fhw_path == G2_FHW_PATH
            && self.fhw_sha256 == G2_FHW_SHA256
            && self.defender_path.len() <= MAX_AUTHORITY_PATH
            && self.fhw_path.len() <= MAX_AUTHORITY_PATH
    }
}

/// Ordinary reduced-AND evidence record (design §2.2). Stored scalar fields
/// are evidence only; the verifier recomputes every one of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group2ZoneV1 {
    pub schema_version: u16,
    pub authority: Group2AuthorityV1,
    pub claimed_d14_budget: u32,
    pub build_horizon: u32,
    pub child_plan_sha256: Sha256Bytes,
    pub finder_summary_sha256: Sha256Bytes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FhwEdgeClassV1 {
    Exact,
    FrontierCovered,
    NonFrontierCovered,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoleKeyV1 {
    ChoiceMove {
        node: CertNodeId,
        cell: HexCoord,
    },
    OrCompletionMove {
        node: CertNodeId,
        cell: HexCoord,
    },
    LeafEmpty {
        node: CertNodeId,
        witness: WindowKey,
        cell: HexCoord,
    },
    Checkpoint {
        gate: CertNodeId,
        threat: WindowKey,
        cell: HexCoord,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FhwRoleRowV1 {
    ExactOrFcZero,
    NonFcRcZero,
    NonFcCharged,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FhwRoleClaimV1 {
    pub role: RoleKeyV1,
    pub child_f: u32,
    pub row: FhwRoleRowV1,
    pub epsilon: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FhwKappaRowV1 {
    NonDAlive,
    ExactOrFcNonIncident,
    ExactOrFcDirect,
    NonFcTouchedNonIncident,
    NonFcTouchedDirect,
    NonFcEmptyDirect,
    NonFcEmptyNonIncidentQlt6,
    NonFcEmptyNonIncidentWcPass,
    NonFcEmptyNonIncidentWcFail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuardResultV1 {
    NotApplicable,
    Pass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FhwWindowClaimV1 {
    pub window: WindowKey,
    pub child_q: u32,
    pub d_in_window: bool,
    pub s_in_window: bool,
    pub row: FhwKappaRowV1,
    pub kappa: u8,
    pub retained_guard: GuardResultV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FhwMapV1 {
    pub real_reply: HexCoord,
    pub representative: HexCoord,
    pub edge_class: FhwEdgeClassV1,
    pub roles: Vec<FhwRoleClaimV1>,
    pub windows: Vec<FhwWindowClaimV1>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FhwGateProofV1 {
    pub schema_version: u16,
    pub authority: Group2AuthorityV1,
    pub threats: Vec<WindowKey>,
    pub escape_resolution_ply: u32,
    pub map: Vec<FhwMapV1>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniversalGroup2NodeV1 {
    pub edges: Vec<CertEdge>,
    pub proof: Group2ZoneV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FhwGateNodeV1 {
    pub representatives: Vec<CertEdge>,
    pub proof: FhwGateProofV1,
}

/// A proof arena node.  Nodes prove that `TssCertificate::claimant` wins.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CertNode {
    /// Claimant placement that completes its named window immediately.
    OrCompletion {
        mv: HexCoord,
        witness: WindowKey,
        completion_ply: u32,
    },
    /// Claimant-to-move lambda-1 win with exact count/budget evidence.
    Win {
        witness: WindowKey,
        count: u8,
        budget: u8,
        resolution_ply: u32,
    },
    /// Defender-to-move adaptive lambda-1 loss contract.
    Loss {
        witnesses: Vec<WindowKey>,
        resolution_ply: u32,
    },
    /// A claimant move selecting one winning continuation.
    Choice { mv: HexCoord, child: CertNodeId },
    /// All listed opponent moves are replayed. When `implicit_dispatch` is
    /// true, every extendable-hit kernel cell must be represented explicitly
    /// or by a parent-validated same-turn commutation. The unrepresented
    /// complement is individually checked by the debug oracle by applying the
    /// move and invoking lambda-1.
    Universal {
        edges: Vec<CertEdge>,
        implicit_dispatch: bool,
        zone: Option<ZoneInfo>,
        commutations: Vec<CertCommutation>,
    },
    /// v1 Group-2 reduced ordinary AND (design §2.2). Boxed so the legacy
    /// `CertNode` size/alignment is provably unchanged.
    UniversalGroup2V1(Box<UniversalGroup2NodeV1>),
    /// v1 FHW-T3-R forcing gate (design §2.2). Boxed for the same reason.
    FhwGateV1(Box<FhwGateNodeV1>),
}

impl CertNode {
    /// True for the two v1 Group-2 extension variants. A certificate
    /// containing any such node takes the extension verifier under the
    /// `Group2V1` policy and REJECTS under the default legacy-only policy.
    pub fn is_group2_extension(&self) -> bool {
        matches!(
            self,
            CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_)
        )
    }
}

/// True when the certificate contains at least one v1 extension node.
pub(crate) fn certificate_has_group2(cert: &TssCertificate) -> bool {
    cert.nodes.iter().any(CertNode::is_group2_extension)
}

/// Replayable proof that `claimant` wins from the exactly bound root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TssCertificate {
    pub root: RootBinding,
    pub claimant: Player,
    pub root_node: CertNodeId,
    pub nodes: Vec<CertNode>,
    /// Caller-supplied absolute deadline.  The verifier derives the
    /// certificate's actual T as the maximum exact leaf resolution and merely
    /// checks that derived value against this external cap.
    pub semantic_horizon: u32,
}

/// Independent checker for [`TssCertificate`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TssVerifier;

impl CertVerify for TssVerifier {
    type Cert = TssCertificate;

    fn verify(&self, state: &RustHexoState, cert: &Self::Cert, claimed: ProofStatus) -> bool {
        verify_certificate(state, cert, claimed, false)
    }
}

/// Extension-enabled checker (design §5.1 verifier mode `Group2V1`). A
/// certificate without any new node takes the byte-identical legacy path; a
/// certificate containing new nodes is validated by the isolated
/// `tss_verify_group2` module. The default [`TssVerifier`] remains
/// `LegacyOnly` and rejects every new in-memory variant. Certificate contents
/// can never select the policy: the caller (trainer configuration) chooses
/// which concrete verifier to construct.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Group2Verifier;

impl CertVerify for Group2Verifier {
    type Cert = TssCertificate;

    fn verify(&self, state: &RustHexoState, cert: &Self::Cert, claimed: ProofStatus) -> bool {
        if !certificate_has_group2(cert) {
            return verify_certificate(state, cert, claimed, false);
        }
        crate::tss_verify_group2::verify_group2_certificate(state, cert, claimed)
    }
}

fn verify_certificate(
    state: &RustHexoState,
    cert: &TssCertificate,
    claimed: ProofStatus,
    dispatch_oracle: bool,
) -> bool {
    // LegacyOnly policy: the two v1 extension variants are rejected before
    // any other work; every legacy certificate takes the unchanged path below.
    if certificate_has_group2(cert) {
        return false;
    }
    if claimed == ProofStatus::Unknown || cert.root != RootBinding::from_state(state) {
        return false;
    }

    // Win/Loss is from the root side-to-move perspective, while the arena
    // itself is uniformly a winning strategy for the named claimant.
    let expected_claimant = match claimed {
        ProofStatus::Win => state.current_player(),
        ProofStatus::Loss => state.current_player().other(),
        ProofStatus::Unknown => return false,
    };
    if cert.claimant != expected_claimant || !validate_arena(cert) {
        return false;
    }

    let Some(meta) = certificate_metadata(cert) else {
        return false;
    };
    if meta.derived_t > cert.semantic_horizon
        || meta
            .zone_build_t
            .is_some_and(|build_t| meta.derived_t > build_t)
        || (meta.has_zone && state.is_terminal())
    {
        return false;
    }

    let mut replay = state.clone();
    let Some(mut memo) = ReplayMemo::new(cert) else {
        return false;
    };
    verify_node(
        cert,
        cert.root_node,
        &mut replay,
        cert.claimant,
        0,
        &mut memo,
        dispatch_oracle,
        &meta,
        &[],
    )
}

struct CertificateMetadata {
    derived_t: u32,
    has_zone: bool,
    zone_build_t: Option<u32>,
    cores: Vec<Vec<HexCoord>>,
    root_stones: Vec<HexCoord>,
}

fn certificate_metadata(cert: &TssCertificate) -> Option<CertificateMetadata> {
    let mut derived_t = 0u32;
    let mut has_zone = false;
    let mut zone_build_t: Option<u32> = None;
    for node in &cert.nodes {
        match node {
            CertNode::OrCompletion { completion_ply, .. } => {
                derived_t = derived_t.max(*completion_ply);
            }
            CertNode::Win { resolution_ply, .. } | CertNode::Loss { resolution_ply, .. } => {
                derived_t = derived_t.max(*resolution_ply);
            }
            CertNode::Universal { zone, .. } => {
                has_zone |= zone.is_some();
                if let Some(zone) = zone {
                    zone_build_t = Some(
                        zone_build_t.map_or(zone.build_horizon, |old| old.min(zone.build_horizon)),
                    );
                }
            }
            CertNode::Choice { .. } => {}
            CertNode::UniversalGroup2V1(_) => {}
            // R1 (DESIGN_AMENDMENT_R1_R2.md): every gate escape deadline
            // participates in the maximum defining the certificate's derived
            // resolution T. (Gate-bearing certificates are additionally
            // rejected wholesale by the current narrowed class rules, but the
            // derived-T definition is the amended one.)
            CertNode::FhwGateV1(gate) => {
                derived_t = derived_t.max(gate.proof.escape_resolution_ply);
            }
        }
    }
    let mut cores = vec![None; cert.nodes.len()];
    // Depth-bounded like `verify_node`: metadata construction must never
    // out-resource the replay it precedes (a valid acyclic million-node Choice
    // chain would otherwise overflow the stack here before verification could
    // reject it at MAX_CERT_DEPTH).
    fn build(
        cert: &TssCertificate,
        id: CertNodeId,
        memo: &mut [Option<Vec<HexCoord>>],
        depth: usize,
    ) -> Option<Vec<HexCoord>> {
        if depth > MAX_CERT_DEPTH {
            return None;
        }
        if let Some(core) = memo.get(id as usize)?.as_ref() {
            return Some(core.clone());
        }
        let mut core = Vec::new();
        match cert.nodes.get(id as usize)? {
            CertNode::OrCompletion { mv, witness, .. } => {
                core.push(*mv);
                core.extend(witness.cells());
            }
            CertNode::Win { witness, .. } => core.extend(witness.cells()),
            CertNode::Loss { witnesses, .. } => {
                for witness in witnesses {
                    core.extend(witness.cells());
                }
            }
            CertNode::Choice { mv, child } => {
                core.push(*mv);
                core.extend(build(cert, *child, memo, depth + 1)?);
            }
            CertNode::Universal { edges, .. } => {
                for edge in edges {
                    core.extend(build(cert, edge.child, memo, depth + 1)?);
                }
            }
            CertNode::UniversalGroup2V1(node) => {
                for edge in &node.edges {
                    core.extend(build(cert, edge.child, memo, depth + 1)?);
                }
            }
            CertNode::FhwGateV1(gate) => {
                for edge in &gate.representatives {
                    core.extend(build(cert, edge.child, memo, depth + 1)?);
                }
            }
        }
        core.sort_by_key(|coord| coord_key(*coord));
        core.dedup();
        memo[id as usize] = Some(core.clone());
        Some(core)
    }
    build(cert, cert.root_node, &mut cores, 0)?;
    let cores = cores.into_iter().collect::<Option<Vec<_>>>()?;
    Some(CertificateMetadata {
        derived_t,
        has_zone,
        zone_build_t,
        cores,
        root_stones: cert.root.occupancy.clone(),
    })
}

/// Cheap structural preflight used by the solver wrapper before verification.
/// This does not establish truth; it only derives the certificate's exact
/// semantic deadline and whether any AND node used the zone theorem.
pub(crate) fn certificate_horizon_preflight(cert: &TssCertificate) -> Option<(u32, bool)> {
    certificate_metadata(cert).map(|meta| (meta.derived_t, meta.has_zone))
}

/// Metadata view for the Group-2 extension verifier. `derived_t` follows the
/// R1-amended definition (gate escape deadlines participate in the maximum).
pub(crate) struct Group2Metadata {
    pub(crate) derived_t: u32,
}

pub(crate) fn certificate_metadata_for_group2(cert: &TssCertificate) -> Option<Group2Metadata> {
    certificate_metadata(cert).map(|meta| Group2Metadata {
        derived_t: meta.derived_t,
    })
}

/// Arena validation shared with the Group-2 extension verifier (bounds,
/// caps, per-node duplicate edges, acyclicity, full reachability).
pub(crate) fn validate_arena_for_group2(cert: &TssCertificate) -> bool {
    validate_arena(cert)
}

/// A full-position identity used only to make replay of shared DAG nodes both
/// bounded and sound.  A shared arena node may only denote one exact state.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ReplayKey {
    stones: Vec<(i16, i16, u8)>,
    /// Same-position Universal nodes can have different obligations when a
    /// parent P3 commutation supplies an omitted reply. Bind that context into
    /// memo identity so a permissive occurrence cannot discharge a stricter
    /// one.
    allowed_commuted: Vec<(i16, i16)>,
    current_player: u8,
    phase: PhaseKey,
    placements_made: u32,
    terminal: Option<(u8, u32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PhaseKey {
    Opening,
    FirstStone,
    SecondStone(i16, i16),
}

impl ReplayKey {
    fn from_state_with_allowed(state: &RustHexoState, allowed_commuted: &[HexCoord]) -> Self {
        let binding = RootBinding::from_state(state);
        let stones = binding
            .occupancy
            .iter()
            .copied()
            .zip(binding.owners.iter().copied())
            .map(|(c, p)| (c.q, c.r, player_key(p)))
            .collect();
        let mut allowed_commuted = allowed_commuted
            .iter()
            .map(|coord| (coord.q, coord.r))
            .collect::<Vec<_>>();
        allowed_commuted.sort_unstable();
        let phase = match binding.phase {
            TurnPhase::Opening => PhaseKey::Opening,
            TurnPhase::FirstStone => PhaseKey::FirstStone,
            TurnPhase::SecondStone { first } => PhaseKey::SecondStone(first.q, first.r),
        };
        Self {
            stones,
            allowed_commuted,
            current_player: player_key(binding.current_player),
            phase,
            placements_made: binding.placements_made,
            terminal: binding
                .terminal
                .map(|outcome| (player_key(outcome.winner), outcome.placements)),
        }
    }

    fn heap_bytes(&self) -> usize {
        self.stones
            .capacity()
            .saturating_mul(size_of::<(i16, i16, u8)>())
            .saturating_add(
                self.allowed_commuted
                    .capacity()
                    .saturating_mul(size_of::<(i16, i16)>()),
            )
            .saturating_add(usize::from(!self.allowed_commuted.is_empty()).saturating_mul(32))
            .saturating_add(32)
    }
}

struct ReplayMemo {
    /// A node is accepted only when every occurrence reaches the same full
    /// position.  Once verified there, subsequent transposition edges reuse it.
    states: Vec<Option<(ReplayKey, bool)>>,
    shared: Vec<bool>,
    bytes: usize,
}

impl ReplayMemo {
    fn new(cert: &TssCertificate) -> Option<Self> {
        let nodes = cert.nodes.len();
        let mut indegree = vec![0u32; nodes];
        for node in &cert.nodes {
            match node {
                CertNode::Choice { child, .. } => {
                    indegree[*child as usize] = indegree[*child as usize].saturating_add(1);
                }
                CertNode::Universal { edges, .. } => {
                    for edge in edges {
                        indegree[edge.child as usize] =
                            indegree[edge.child as usize].saturating_add(1);
                    }
                }
                CertNode::UniversalGroup2V1(node) => {
                    for edge in &node.edges {
                        indegree[edge.child as usize] =
                            indegree[edge.child as usize].saturating_add(1);
                    }
                }
                CertNode::FhwGateV1(gate) => {
                    for edge in &gate.representatives {
                        indegree[edge.child as usize] =
                            indegree[edge.child as usize].saturating_add(1);
                    }
                }
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {}
            }
        }
        let shared: Vec<bool> = indegree.into_iter().map(|count| count > 1).collect();
        let mut states = Vec::with_capacity(nodes);
        states.resize_with(nodes, || None);
        let bytes = states
            .capacity()
            .checked_mul(size_of::<Option<(ReplayKey, bool)>>())?
            .checked_add(shared.capacity().checked_mul(size_of::<bool>())?)?
            .checked_add(64)?;
        (bytes <= MAX_VERIFY_MEMO_BYTES).then_some(Self {
            states,
            shared,
            bytes,
        })
    }

    fn get(&self, id: CertNodeId) -> Option<&(ReplayKey, bool)> {
        if !self.shared.get(id as usize).copied().unwrap_or(false) {
            return None;
        }
        self.states.get(id as usize)?.as_ref()
    }

    fn is_shared(&self, id: CertNodeId) -> bool {
        self.shared.get(id as usize).copied().unwrap_or(false)
    }

    fn insert(&mut self, id: CertNodeId, key: ReplayKey, result: bool) -> bool {
        if !self.shared.get(id as usize).copied().unwrap_or(false) {
            return true;
        }
        let Some(slot) = self.states.get_mut(id as usize) else {
            return false;
        };
        if slot.is_some() {
            return false;
        }
        let new_bytes = self.bytes.saturating_add(key.heap_bytes());
        if new_bytes > MAX_VERIFY_MEMO_BYTES {
            return false;
        }
        *slot = Some((key, result));
        self.bytes = new_bytes;
        true
    }
}

fn verify_node(
    cert: &TssCertificate,
    id: CertNodeId,
    state: &mut RustHexoState,
    claimant: Player,
    depth: usize,
    memo: &mut ReplayMemo,
    dispatch_oracle: bool,
    meta: &CertificateMetadata,
    allowed_commuted: &[HexCoord],
) -> bool {
    if depth > MAX_CERT_DEPTH {
        return false;
    }

    let replay_key = memo
        .is_shared(id)
        .then(|| ReplayKey::from_state_with_allowed(state, allowed_commuted));
    if let (Some(key), Some((seen, result))) = (replay_key.as_ref(), memo.get(id)) {
        return seen == key && *result;
    }

    // The graph is already known acyclic, so inserting after evaluation cannot
    // recurse back to this node.  Failed nodes are memoized as well.
    let node = &cert.nodes[id as usize];
    let result = match node {
        CertNode::OrCompletion {
            mv,
            witness,
            completion_ply,
        } => verify_or_completion(state, claimant, *mv, *witness, *completion_ply, meta),
        CertNode::Win {
            witness,
            count,
            budget,
            resolution_ply,
        } => verify_win_leaf(
            state,
            claimant,
            *witness,
            *count,
            *budget,
            *resolution_ply,
            meta,
        ),
        CertNode::Loss {
            witnesses,
            resolution_ply,
        } => verify_loss_leaf(state, claimant, witnesses, *resolution_ply, meta),
        CertNode::Choice { mv, child } => {
            state.current_player() == claimant
                && !state.is_terminal()
                && attacker_placement_wf(state, claimant, *mv, meta)
                && with_move(state, *mv, |child_state, outcome| {
                    if outcome.is_some() {
                        return false;
                    }
                    verify_node(
                        cert,
                        *child,
                        child_state,
                        claimant,
                        depth + 1,
                        memo,
                        dispatch_oracle,
                        meta,
                        &[],
                    )
                })
        }
        CertNode::Universal {
            edges,
            implicit_dispatch,
            zone,
            commutations,
        } => verify_universal(
            cert,
            state,
            claimant,
            edges,
            *implicit_dispatch,
            zone.clone(),
            commutations,
            depth,
            memo,
            dispatch_oracle,
            meta,
            id,
            allowed_commuted,
        ),
        // Unreachable on the legacy path (verify_certificate rejects any
        // certificate containing an extension node before replay); kept as an
        // explicit fail-closed arm rather than a wildcard.
        CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => false,
    };

    if let Some(key) = replay_key {
        if !memo.insert(id, key, result) {
            return false;
        }
    }
    result
}

fn window_entry(state: &RustHexoState, key: WindowKey) -> Option<hexo_engine::WindowEntry> {
    state
        .board()
        .windows()
        .entries()
        .find(|entry| entry.key() == key)
}

fn attacker_placement_wf(
    state: &RustHexoState,
    claimant: Player,
    mv: HexCoord,
    meta: &CertificateMetadata,
) -> bool {
    state
        .board()
        .occupied_cells()
        .iter()
        .copied()
        .filter(|stone| state.board().get(*stone) == Some(claimant))
        .chain(meta.root_stones.iter().copied())
        .any(|anchor| hex_distance(anchor, mv) <= 8)
}

fn verify_or_completion(
    state: &mut RustHexoState,
    claimant: Player,
    mv: HexCoord,
    witness: WindowKey,
    completion_ply: u32,
    meta: &CertificateMetadata,
) -> bool {
    state.current_player() == claimant
        && !state.is_terminal()
        && witness.contains(mv)
        && attacker_placement_wf(state, claimant, mv, meta)
        && completion_ply == state.placements_made().saturating_add(1)
        && completion_ply <= meta.derived_t
        && with_move(state, mv, |child_state, outcome| {
            outcome.is_some_and(|outcome| outcome.winner == claimant)
                && window_entry(child_state, witness).is_some_and(|entry| {
                    entry.count(claimant) == 6 && entry.count(claimant.other()) == 0
                })
        })
}

fn verify_win_leaf(
    state: &RustHexoState,
    claimant: Player,
    witness: WindowKey,
    count: u8,
    budget: u8,
    resolution_ply: u32,
    meta: &CertificateMetadata,
) -> bool {
    if state.is_terminal() || state.current_player() != claimant {
        return false;
    }
    let actual_budget = threats_shared::placements_remaining(state);
    let Some(entry) = window_entry(state, witness) else {
        return false;
    };
    let expected_resolution = match count {
        5 => state.placements_made().saturating_add(1),
        4 if actual_budget == 2 => state.placements_made().saturating_add(2),
        _ => return false,
    };
    budget == actual_budget
        && entry.count(claimant) == count
        && entry.count(claimant.other()) == 0
        && resolution_ply == expected_resolution
        && resolution_ply <= meta.derived_t
        && entry
            .empty_cells()
            .into_iter()
            .all(|mv| attacker_placement_wf(state, claimant, mv, meta))
}

fn family_hitting_exceeds(witnesses: &[Vec<HexCoord>], b: u8) -> bool {
    let mut universe = witnesses.iter().flatten().copied().collect::<Vec<_>>();
    universe.sort_by_key(|coord| coord_key(*coord));
    universe.dedup();
    if witnesses.iter().any(Vec::is_empty) {
        return true;
    }
    if b >= 1
        && universe
            .iter()
            .any(|a| witnesses.iter().all(|w| w.contains(a)))
    {
        return false;
    }
    if b >= 2 {
        for (index, a) in universe.iter().enumerate() {
            for b_cell in &universe[index..] {
                if witnesses
                    .iter()
                    .all(|w| w.contains(a) || w.contains(b_cell))
                {
                    return false;
                }
            }
        }
    }
    true
}

fn verify_loss_leaf(
    state: &RustHexoState,
    claimant: Player,
    witnesses: &[WindowKey],
    resolution_ply: u32,
    meta: &CertificateMetadata,
) -> bool {
    if state.is_terminal() || state.current_player() == claimant || witnesses.is_empty() {
        return false;
    }
    let analysis = threats_shared::analyze(state);
    if analysis.own_win_now {
        return false;
    }
    let mut empties = Vec::with_capacity(witnesses.len());
    for &key in witnesses {
        let Some(entry) = window_entry(state, key) else {
            return false;
        };
        if entry.active_player() != Some(claimant) || entry.count(claimant) < 4 {
            return false;
        }
        let cells = entry.empty_cells();
        if !cells
            .iter()
            .copied()
            .all(|mv| attacker_placement_wf(state, claimant, mv, meta))
        {
            return false;
        }
        empties.push(cells);
    }
    let expected = state
        .placements_made()
        .saturating_add(u32::from(analysis.b))
        .saturating_add(2);
    family_hitting_exceeds(&empties, analysis.b)
        && resolution_ply == expected
        && resolution_ply <= meta.derived_t
}

fn validate_commutations(
    cert: &TssCertificate,
    state: &mut RustHexoState,
    edges: &[CertEdge],
    commutations: &[CertCommutation],
) -> Option<Vec<(HexCoord, Vec<HexCoord>)>> {
    if commutations.is_empty() {
        return Some(Vec::new());
    }
    if !matches!(state.phase(), TurnPhase::FirstStone)
        || threats_shared::placements_remaining(state) != 2
    {
        return None;
    }
    let mut grouped: Vec<(HexCoord, Vec<HexCoord>)> = Vec::new();
    let mut seen = Vec::new();
    for item in commutations {
        if coord_key(item.omitted_second) >= coord_key(item.first)
            || seen.contains(&(item.first, item.omitted_second))
        {
            return None;
        }
        seen.push((item.first, item.omitted_second));
        let first_edge = edges.iter().find(|edge| edge.mv == item.first)?;
        let mirror_edge = edges.iter().find(|edge| edge.mv == item.omitted_second)?;
        if first_edge.child != item.first_child || mirror_edge.child != item.mirror_child {
            return None;
        }
        let CertNode::Universal {
            edges: first_replies,
            ..
        } = cert.nodes.get(item.first_child as usize)?
        else {
            return None;
        };
        let CertNode::Universal {
            edges: mirror_replies,
            ..
        } = cert.nodes.get(item.mirror_child as usize)?
        else {
            return None;
        };
        if first_replies
            .iter()
            .any(|edge| edge.mv == item.omitted_second)
            || !mirror_replies.iter().any(|edge| edge.mv == item.first)
        {
            return None;
        }
        for mv in [item.first, item.omitted_second] {
            if !with_move(state, mv, |child, outcome| {
                outcome.is_none() && matches!(child.phase(), TurnPhase::SecondStone { .. })
            }) {
                return None;
            }
        }
        let pair_outcome = |a: HexCoord, b: HexCoord| {
            let mut replay = state.clone();
            let first = replay.apply_with_delta(Placement { coord: a }).ok()?.0;
            if first.outcome.is_some() {
                return None;
            }
            Some(
                replay
                    .apply_with_delta(Placement { coord: b })
                    .ok()?
                    .0
                    .outcome,
            )
        };
        let forward = pair_outcome(item.first, item.omitted_second)?;
        let mirror = pair_outcome(item.omitted_second, item.first)?;
        if forward != mirror {
            return None;
        }
        match grouped.iter_mut().find(|(first, _)| *first == item.first) {
            Some((_, omitted)) => omitted.push(item.omitted_second),
            None => grouped.push((item.first, vec![item.omitted_second])),
        }
    }
    for (_, omitted) in &mut grouped {
        omitted.sort_by_key(|coord| coord_key(*coord));
    }
    Some(grouped)
}

fn verify_universal(
    cert: &TssCertificate,
    state: &mut RustHexoState,
    claimant: Player,
    edges: &[CertEdge],
    implicit_dispatch: bool,
    zone: Option<ZoneInfo>,
    commutations: &[CertCommutation],
    depth: usize,
    memo: &mut ReplayMemo,
    dispatch_oracle: bool,
    meta: &CertificateMetadata,
    node_id: CertNodeId,
    allowed_commuted: &[HexCoord],
) -> bool {
    if state.is_terminal()
        || state.current_player() == claimant
        || threats_shared::analyze(state).own_win_now
    {
        return false;
    }
    // Duplicate explicit moves are rejected rather than silently coalesced.
    // Legality is independently established by the replay below.
    let mut explicit_moves: Vec<HexCoord> = edges.iter().map(|edge| edge.mv).collect();
    explicit_moves.sort_by_key(|coord| coord_key(*coord));
    if explicit_moves.windows(2).any(|pair| pair[0] == pair[1]) {
        return false;
    }
    let mut allowed = allowed_commuted.to_vec();
    allowed.sort_by_key(|coord| coord_key(*coord));
    if allowed.windows(2).any(|pair| pair[0] == pair[1])
        || allowed.iter().any(|mv| explicit_moves.contains(mv))
        || allowed.iter().any(|mv| {
            let mut probe = state.clone();
            probe.apply_with_delta(Placement { coord: *mv }).is_err()
        })
    {
        return false;
    }
    let mut represented = explicit_moves.clone();
    represented.extend(allowed.iter().copied());
    represented.sort_by_key(|coord| coord_key(*coord));
    // Empty nested nodes are meaningful only when a validated parent
    // commutation supplies their entire same-turn obligation.
    if represented.is_empty() || (zone.is_some() && !allowed.is_empty()) {
        return false;
    }

    let boundary = dispatch_boundary(state, claimant);
    let child_commutations = match validate_commutations(cert, state, edges, commutations) {
        Some(value) => value,
        None => return false,
    };
    if implicit_dispatch && boundary.is_none() {
        // In particular, a spare-stone node may never advertise an implicit
        // complement even if this particular certificate happened to list all
        // of its legal moves.
        return false;
    }

    if implicit_dispatch {
        // T6 kernel staple: at a checked post-opening ¬own_win_now, tau=b
        // boundary, only cells extendable to a size-b transversal can retain a
        // live defense. Requiring the independently derived kernel is the
        // complete obligation; certificates may explicitly prove any superset.
        let kernel = boundary.as_ref().expect("checked above");
        if kernel.iter().any(|mv| {
            represented
                .binary_search_by_key(&coord_key(*mv), |c| coord_key(*c))
                .is_err()
        }) {
            return false;
        }
    } else if let Some(zone) = zone {
        if !verify_zone_node(cert, state, claimant, &explicit_moves, zone, meta, node_id) {
            return false;
        }
    } else {
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        legal.sort_by_key(|coord| coord_key(*coord));
        if represented != legal {
            return false;
        }
    }

    for edge in edges {
        if !with_move(state, edge.mv, |child_state, outcome| {
            if outcome.is_some() {
                return false;
            }
            verify_node(
                cert,
                edge.child,
                child_state,
                claimant,
                depth + 1,
                memo,
                dispatch_oracle,
                meta,
                child_commutations
                    .iter()
                    .find(|(first, _)| *first == edge.mv)
                    .map(|(_, omitted)| omitted.as_slice())
                    .unwrap_or(&[]),
            )
        }) {
            return false;
        }
    }

    if implicit_dispatch && dispatch_oracle {
        // Paired debug oracle: validate every omitted nonkernel move with the
        // per-move lambda-1 staple. Production never enters this arm.
        let mut legal = Vec::new();
        state.write_legal_moves(&mut legal);
        let kernel = boundary.as_ref().expect("checked above");
        for mv in legal {
            if represented
                .binary_search_by_key(&coord_key(mv), |c| coord_key(*c))
                .is_ok()
            {
                continue;
            }
            if kernel
                .binary_search_by_key(&coord_key(mv), |c| coord_key(*c))
                .is_ok()
                || !with_move(state, mv, |child_state, outcome| match outcome {
                    Some(outcome) => outcome.winner == claimant,
                    None => lambda1_proves_claimant(child_state, claimant),
                })
            {
                return false;
            }
        }
    }
    true
}

fn remaining_defender_placements(
    state: &RustHexoState,
    claimant: Player,
    horizon: u32,
) -> Option<u32> {
    // A valid zone node exists only for defender budgets 0..=5 (the solver
    // takes the full legal set at d >= 6), so once the count passes that band
    // the exact value can no longer matter — bail rather than walk a
    // corrupted/adversarial horizon (`u32::MAX` would otherwise spin billions
    // of iterations outside every node cap). `None` rejects the node, which
    // is always sound.
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

fn set_contains(sorted: &[HexCoord], coord: HexCoord) -> bool {
    sorted
        .binary_search_by_key(&coord_key(coord), |candidate| coord_key(*candidate))
        .is_ok()
}

#[derive(Clone, Debug)]
struct VerifierZoneSummary {
    local_budget: u32,
    protected: Vec<HexCoord>,
}

/// Independently reconstruct D10's reachable live-role union and D14's local
/// defender clock. No finder-supplied candidate set participates in this pass.
fn verifier_zone_summary(
    cert: &TssCertificate,
    state: &mut RustHexoState,
    node_id: CertNodeId,
    depth: usize,
) -> Option<VerifierZoneSummary> {
    if depth > MAX_CERT_DEPTH {
        return None;
    }
    let node = cert.nodes.get(node_id as usize)?;
    let mut protected = Vec::new();
    let local_budget = match node {
        CertNode::OrCompletion { mv, .. } => {
            protected.push(*mv);
            0
        }
        CertNode::Win { witness, .. } => {
            protected.extend(window_entry(state, *witness)?.empty_cells());
            0
        }
        CertNode::Loss { witnesses, .. } => {
            for witness in witnesses {
                protected.extend(window_entry(state, *witness)?.empty_cells());
            }
            u32::from(threats_shared::placements_remaining(state))
        }
        CertNode::Choice { mv, child } => {
            let (result, delta) = state.apply_with_delta(Placement { coord: *mv }).ok()?;
            if result.outcome.is_some() {
                state.undo(delta);
                return None;
            }
            let child_summary = verifier_zone_summary(cert, state, *child, depth + 1);
            state.undo(delta);
            let child_summary = child_summary?;
            protected.push(*mv);
            protected.extend(child_summary.protected);
            child_summary.local_budget
        }
        CertNode::Universal { edges, .. } => {
            let mut maximum = 0u32;
            for edge in edges {
                let (result, delta) = state.apply_with_delta(Placement { coord: edge.mv }).ok()?;
                if result.outcome.is_some() {
                    state.undo(delta);
                    return None;
                }
                let child_summary = verifier_zone_summary(cert, state, edge.child, depth + 1);
                state.undo(delta);
                let child_summary = child_summary?;
                maximum = maximum.max(child_summary.local_budget);
                protected.extend(child_summary.protected);
            }
            maximum.saturating_add(1)
        }
        // The legacy zone theorem is never combined with extension nodes
        // (narrow-v1 no-mixing rule); fail closed.
        CertNode::UniversalGroup2V1(_) | CertNode::FhwGateV1(_) => return None,
    };
    protected.sort_by_key(|coord| coord_key(*coord));
    protected.dedup();
    Some(VerifierZoneSummary {
        local_budget,
        protected,
    })
}

/// Re-derive the mandatory T4 union from the replayed position alone:
/// Z_dir union Z_seed union Z_touch union Z_virgin, with the deterministic D9
/// nonempty fallback. Current hitting cells are intentionally absent (the
/// revised document makes them an optional heuristic, not a T3/T4 clause).
fn verifier_uniform_zone(
    state: &RustHexoState,
    claimant: Player,
    summary: &VerifierZoneSummary,
) -> Vec<HexCoord> {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| coord_key(*coord));
    let stones = state.board().occupied_cells();

    let mut zone = summary
        .protected
        .iter()
        .copied()
        .filter(|cell| set_contains(&legal, *cell))
        .collect::<Vec<_>>();

    let pending = summary
        .protected
        .iter()
        .copied()
        .filter(|cell| !set_contains(&legal, *cell) && !stones.contains(cell))
        .collect::<Vec<_>>();
    if !pending.is_empty() {
        let radius = seed_band_radius(summary.local_budget);
        zone.extend(legal.iter().copied().filter(|cell| {
            pending
                .iter()
                .any(|target| i32::from(hex_distance(*cell, *target)) <= radius)
        }));
    }

    let defender = claimant.other();
    for entry in state.board().windows().entries() {
        let count = entry.count(defender);
        if entry.active_player() == Some(defender)
            && count >= 1
            && u32::from(count).saturating_add(summary.local_budget) >= 6
        {
            zone.extend(entry.empty_cells());
        }
    }

    // Conservative uniform exposure wrapper: B>=6 searches the whole legal
    // set, a valid superset of Z_virgin. For B<=5 the virgin component is empty.
    if summary.local_budget >= 6 {
        zone.extend(legal.iter().copied());
    }
    zone.sort_by_key(|coord| coord_key(*coord));
    zone.dedup();
    if zone.is_empty() {
        if let Some(&fallback) = legal.first() {
            zone.push(fallback);
        }
    }
    zone
}

fn verify_zone_node(
    cert: &TssCertificate,
    state: &RustHexoState,
    claimant: Player,
    explicit: &[HexCoord],
    zone: ZoneInfo,
    meta: &CertificateMetadata,
    node_id: CertNodeId,
) -> bool {
    let analysis = threats_shared::analyze(state);
    if matches!(state.phase(), TurnPhase::Opening)
        || state.current_player() == claimant
        || analysis.own_win_now
        || analysis.min_hitting_set.is_some_and(|k| k >= analysis.b)
        || explicit.is_empty()
    {
        return false;
    }
    let mut replay = state.clone();
    let Some(summary) = verifier_zone_summary(cert, &mut replay, node_id, 0) else {
        return false;
    };
    if zone.d != summary.local_budget || zone.build_horizon != cert.semantic_horizon {
        return false;
    }

    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| coord_key(*coord));
    if explicit.iter().any(|mv| !set_contains(&legal, *mv)) {
        return false;
    }
    let required = verifier_uniform_zone(state, claimant, &summary);
    let _ = meta; // Metadata remains the independently derived horizon/WF contract.
    let result = required.iter().all(|mv| set_contains(explicit, *mv));

    result
}

/// Return the independently derived extendable-hit kernel exactly when the
/// parent is at the sound instant-dispatch boundary.
fn dispatch_boundary(state: &RustHexoState, claimant: Player) -> Option<Vec<HexCoord>> {
    if matches!(state.phase(), TurnPhase::Opening) {
        return None;
    }
    let analysis = threats_shared::analyze(state);
    if analysis.opp_threat_count == 0
        || analysis.own_win_now
        || analysis.min_hitting_set != Some(analysis.b)
    {
        return None;
    }

    // At a universal node the claimant is the opponent of the mover. Collect
    // its active-window empty sets directly from the engine, independently of
    // any solver candidate list or stored coverage claim.
    let family = state
        .board()
        .windows()
        .threats()
        .filter_map(|(owner, entry)| (owner == claimant).then(|| entry.empty_cells()))
        .collect::<Vec<_>>();
    let kernel = extendable_hit_kernel_for_family(&family, analysis.b);
    (!kernel.is_empty()).then_some(kernel)
}

fn extendable_hit_kernel_for_family(family: &[Vec<HexCoord>], budget: u8) -> Vec<HexCoord> {
    let mut universe = family.iter().flatten().copied().collect::<Vec<_>>();
    universe.sort_by_key(|coord| coord_key(*coord));
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
        // Connect-6 dispatch boundaries only have one or two placements. Keep
        // an independently safe full-universe fallback for future phases.
        _ => universe,
    }
}

fn lambda1_proves_claimant(state: &RustHexoState, claimant: Player) -> bool {
    // `analyze` is a forward, one-turn argument; a terminal fact is represented
    // by `CertNode::Terminal` and must not be reinterpreted as a lambda leaf.
    // Its soundness contract is post-opening, so Opening is rejected even
    // though every reachable Opening state is currently threat-free.
    if state.is_terminal() || matches!(state.phase(), TurnPhase::Opening) {
        return false;
    }
    let Some(verdict) = threats_shared::analyze(state).verdict() else {
        return false;
    };
    let proved_winner = if verdict > 0.0 {
        state.current_player()
    } else {
        state.current_player().other()
    };
    proved_winner == claimant
}

fn with_move(
    state: &mut RustHexoState,
    mv: HexCoord,
    verify_child: impl FnOnce(&mut RustHexoState, Option<GameOutcome>) -> bool,
) -> bool {
    let Ok((result, delta)) = state.apply_with_delta(Placement { coord: mv }) else {
        return false;
    };
    let accepted = verify_child(state, result.outcome);
    state.undo(delta);
    accepted
}

fn validate_arena(cert: &TssCertificate) -> bool {
    if cert.root.occupancy.len() != cert.root.owners.len()
        || cert.root.occupancy.len() > MAX_CERT_ROOT_STONES
        || cert.nodes.is_empty()
        || cert.nodes.len() > MAX_CERT_NODES
        || cert.root_node as usize >= cert.nodes.len()
    {
        return false;
    }

    let mut edge_count = 0usize;
    let mut witness_count = 0usize;
    let mut commutation_count = 0usize;
    for node in &cert.nodes {
        match node {
            CertNode::Choice { child, .. } => {
                if *child as usize >= cert.nodes.len() {
                    return false;
                }
            }
            CertNode::Universal {
                edges,
                commutations,
                ..
            } => {
                edge_count = match edge_count.checked_add(edges.len()) {
                    Some(count) if count <= MAX_CERT_EDGES => count,
                    _ => return false,
                };
                if edges
                    .iter()
                    .any(|edge| edge.child as usize >= cert.nodes.len())
                {
                    return false;
                }
                commutation_count = match commutation_count.checked_add(commutations.len()) {
                    Some(count) if count <= MAX_CERT_COMMUTATIONS => count,
                    _ => return false,
                };
                if commutations.iter().any(|item| {
                    item.first_child as usize >= cert.nodes.len()
                        || item.mirror_child as usize >= cert.nodes.len()
                }) {
                    return false;
                }
                let mut moves: Vec<_> = edges.iter().map(|edge| edge.mv).collect();
                moves.sort_by_key(|coord| coord_key(*coord));
                if moves.windows(2).any(|pair| pair[0] == pair[1]) {
                    return false;
                }
            }
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => {
                witness_count = match witness_count.checked_add(1) {
                    Some(count) if count <= MAX_CERT_WITNESSES => count,
                    _ => return false,
                };
            }
            CertNode::Loss { witnesses, .. } => {
                witness_count = match witness_count.checked_add(witnesses.len()) {
                    Some(count) if count <= MAX_CERT_WITNESSES => count,
                    _ => return false,
                };
            }
            CertNode::UniversalGroup2V1(g2) => {
                edge_count = match edge_count.checked_add(g2.edges.len()) {
                    Some(count) if count <= MAX_CERT_EDGES => count,
                    _ => return false,
                };
                if g2
                    .edges
                    .iter()
                    .any(|edge| edge.child as usize >= cert.nodes.len())
                {
                    return false;
                }
                let mut moves: Vec<_> = g2.edges.iter().map(|edge| edge.mv).collect();
                moves.sort_by_key(|coord| coord_key(*coord));
                if moves.windows(2).any(|pair| pair[0] == pair[1]) {
                    return false;
                }
            }
            CertNode::FhwGateV1(gate) => {
                edge_count = match edge_count.checked_add(gate.representatives.len()) {
                    Some(count) if count <= MAX_CERT_EDGES => count,
                    _ => return false,
                };
                if gate
                    .representatives
                    .iter()
                    .any(|edge| edge.child as usize >= cert.nodes.len())
                {
                    return false;
                }
                witness_count = match witness_count.checked_add(gate.proof.threats.len()) {
                    Some(count) if count <= MAX_CERT_WITNESSES => count,
                    _ => return false,
                };
                let mut moves: Vec<_> = gate.representatives.iter().map(|edge| edge.mv).collect();
                moves.sort_by_key(|coord| coord_key(*coord));
                if moves.windows(2).any(|pair| pair[0] == pair[1]) {
                    return false;
                }
            }
        }
    }

    // Three-colour DFS over the entire arena catches cycles even in components
    // unreachable from the declared root.
    let mut colours = vec![0u8; cert.nodes.len()];
    for start in 0..cert.nodes.len() {
        if colours[start] == 0 && !acyclic_from(cert, start, &mut colours) {
            return false;
        }
    }

    // A separate reachability pass rejects every orphan, including an acyclic
    // but otherwise well-formed component.
    let mut seen = vec![false; cert.nodes.len()];
    let mut stack = vec![cert.root_node as usize];
    while let Some(id) = stack.pop() {
        if seen[id] {
            continue;
        }
        seen[id] = true;
        push_children(&cert.nodes[id], &mut stack);
    }
    seen.into_iter().all(|reachable| reachable)
}

fn acyclic_from(cert: &TssCertificate, start: usize, colours: &mut [u8]) -> bool {
    // `(node, exiting)` avoids verifier call-stack exhaustion on malformed
    // certificates while retaining ordinary three-colour DFS semantics.
    let mut stack = vec![(start, false)];
    while let Some((id, exiting)) = stack.pop() {
        if exiting {
            colours[id] = 2;
            continue;
        }
        match colours[id] {
            1 => return false,
            2 => continue,
            _ => {}
        }
        colours[id] = 1;
        stack.push((id, true));
        let mut children = Vec::new();
        push_children(&cert.nodes[id], &mut children);
        // Reverse only to preserve certificate order in the conceptual DFS.
        for child in children.into_iter().rev() {
            match colours[child] {
                1 => return false,
                0 => stack.push((child, false)),
                _ => {}
            }
        }
    }
    true
}

fn push_children(node: &CertNode, out: &mut Vec<usize>) {
    match node {
        CertNode::Choice { child, .. } => out.push(*child as usize),
        CertNode::Universal {
            edges,
            commutations,
            ..
        } => {
            out.extend(edges.iter().map(|edge| edge.child as usize));
            out.extend(
                commutations
                    .iter()
                    .flat_map(|item| [item.first_child as usize, item.mirror_child as usize]),
            );
        }
        CertNode::UniversalGroup2V1(node) => {
            out.extend(node.edges.iter().map(|edge| edge.child as usize));
        }
        CertNode::FhwGateV1(gate) => {
            out.extend(gate.representatives.iter().map(|edge| edge.child as usize));
        }
        CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {}
    }
}

#[inline]
fn coord_key(coord: HexCoord) -> (i16, i16) {
    (coord.q, coord.r)
}

#[inline]
fn player_key(player: Player) -> u8 {
    match player {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}

/// Apply one of the 12 D6 symmetries to an axial coordinate.
///
/// IDs `0..=5` are rotations by repeated `(-r, q+r)`.  IDs `6..=11`
/// first reflect by `(q, -q-r)` and then apply the corresponding rotation.
/// `None` is returned for an invalid ID or if an intermediate coordinate does
/// not fit in the engine's `i16` representation.
pub fn d6_transform_coord(coord: HexCoord, symmetry: u8) -> Option<HexCoord> {
    if symmetry >= D6_SYMMETRY_COUNT {
        return None;
    }
    let mut q = i32::from(coord.q);
    let mut r = i32::from(coord.r);
    if symmetry >= 6 {
        r = q.checked_neg()?.checked_sub(r)?;
        i16::try_from(r).ok()?;
    }
    for _ in 0..(symmetry % 6) {
        let next_q = r.checked_neg()?;
        let next_r = q.checked_add(r)?;
        i16::try_from(next_q).ok()?;
        i16::try_from(next_r).ok()?;
        q = next_q;
        r = next_r;
    }
    Some(HexCoord {
        q: i16::try_from(q).ok()?,
        r: i16::try_from(r).ok()?,
    })
}

fn d6_transform_role_key(role: &RoleKeyV1, symmetry: u8) -> Option<RoleKeyV1> {
    Some(match role {
        RoleKeyV1::ChoiceMove { node, cell } => RoleKeyV1::ChoiceMove {
            node: *node,
            cell: d6_transform_coord(*cell, symmetry)?,
        },
        RoleKeyV1::OrCompletionMove { node, cell } => RoleKeyV1::OrCompletionMove {
            node: *node,
            cell: d6_transform_coord(*cell, symmetry)?,
        },
        RoleKeyV1::LeafEmpty {
            node,
            witness,
            cell,
        } => RoleKeyV1::LeafEmpty {
            node: *node,
            witness: d6_transform_window(*witness, symmetry)?,
            cell: d6_transform_coord(*cell, symmetry)?,
        },
        RoleKeyV1::Checkpoint { gate, threat, cell } => RoleKeyV1::Checkpoint {
            gate: *gate,
            threat: d6_transform_window(*threat, symmetry)?,
            cell: d6_transform_coord(*cell, symmetry)?,
        },
    })
}

fn d6_transform_window(key: WindowKey, symmetry: u8) -> Option<WindowKey> {
    let first = d6_transform_coord(key.coord_at(0), symmetry)?;
    let second = d6_transform_coord(key.coord_at(1), symmetry)?;
    let dq = i32::from(second.q) - i32::from(first.q);
    let dr = i32::from(second.r) - i32::from(first.r);
    let axis = match (dq, dr) {
        (1, 0) => {
            return Some(WindowKey {
                start: first,
                axis: Axis::Q,
            })
        }
        (0, 1) => {
            return Some(WindowKey {
                start: first,
                axis: Axis::R,
            })
        }
        (1, -1) => {
            return Some(WindowKey {
                start: first,
                axis: Axis::QR,
            })
        }
        (-1, 0) => Axis::Q,
        (0, -1) => Axis::R,
        (-1, 1) => Axis::QR,
        _ => return None,
    };
    Some(WindowKey {
        start: d6_transform_coord(key.coord_at(5), symmetry)?,
        axis,
    })
}

/// Remap every coordinate in a certificate under one D6 symmetry.
/// Arena IDs, player identities, counts, and terminal facts are invariant.
pub fn d6_remap_certificate(cert: &TssCertificate, symmetry: u8) -> Option<TssCertificate> {
    if symmetry >= D6_SYMMETRY_COUNT {
        return None;
    }

    let mut stones: Vec<(HexCoord, Player)> = cert
        .root
        .occupancy
        .iter()
        .copied()
        .zip(cert.root.owners.iter().copied())
        .map(|(coord, owner)| Some((d6_transform_coord(coord, symmetry)?, owner)))
        .collect::<Option<_>>()?;
    if stones.len() != cert.root.occupancy.len()
        || cert.root.occupancy.len() != cert.root.owners.len()
    {
        return None;
    }
    stones.sort_by_key(|(coord, _)| coord_key(*coord));
    if stones.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return None;
    }
    let (occupancy, owners) = stones.into_iter().unzip();
    let phase = match cert.root.phase {
        TurnPhase::Opening => TurnPhase::Opening,
        TurnPhase::FirstStone => TurnPhase::FirstStone,
        TurnPhase::SecondStone { first } => TurnPhase::SecondStone {
            first: d6_transform_coord(first, symmetry)?,
        },
    };
    let root = RootBinding {
        occupancy,
        owners,
        current_player: cert.root.current_player,
        phase,
        placements_made: cert.root.placements_made,
        terminal: cert.root.terminal,
    };

    let nodes = cert
        .nodes
        .iter()
        .map(|node| match node {
            CertNode::OrCompletion {
                mv,
                witness,
                completion_ply,
            } => Some(CertNode::OrCompletion {
                mv: d6_transform_coord(*mv, symmetry)?,
                witness: d6_transform_window(*witness, symmetry)?,
                completion_ply: *completion_ply,
            }),
            CertNode::Win {
                witness,
                count,
                budget,
                resolution_ply,
            } => Some(CertNode::Win {
                witness: d6_transform_window(*witness, symmetry)?,
                count: *count,
                budget: *budget,
                resolution_ply: *resolution_ply,
            }),
            CertNode::Loss {
                witnesses,
                resolution_ply,
            } => Some(CertNode::Loss {
                witnesses: witnesses
                    .iter()
                    .map(|key| d6_transform_window(*key, symmetry))
                    .collect::<Option<_>>()?,
                resolution_ply: *resolution_ply,
            }),
            CertNode::Choice { mv, child } => Some(CertNode::Choice {
                mv: d6_transform_coord(*mv, symmetry)?,
                child: *child,
            }),
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            } => Some(CertNode::Universal {
                edges: edges
                    .iter()
                    .map(|edge| {
                        Some(CertEdge {
                            mv: d6_transform_coord(edge.mv, symmetry)?,
                            child: edge.child,
                        })
                    })
                    .collect::<Option<_>>()?,
                implicit_dispatch: *implicit_dispatch,
                zone: zone.clone(),
                commutations: commutations
                    .iter()
                    .map(|commutation| {
                        Some(CertCommutation {
                            first: d6_transform_coord(commutation.first, symmetry)?,
                            omitted_second: d6_transform_coord(
                                commutation.omitted_second,
                                symmetry,
                            )?,
                            first_child: commutation.first_child,
                            mirror_child: commutation.mirror_child,
                        })
                    })
                    .collect::<Option<_>>()?,
            }),
            CertNode::UniversalGroup2V1(node) => {
                Some(CertNode::UniversalGroup2V1(Box::new(
                    UniversalGroup2NodeV1 {
                        edges: node
                            .edges
                            .iter()
                            .map(|edge| {
                                Some(CertEdge {
                                    mv: d6_transform_coord(edge.mv, symmetry)?,
                                    child: edge.child,
                                })
                            })
                            .collect::<Option<_>>()?,
                        // Authority bytes and stored digests are D6-invariant by
                        // construction (§2.4 lexicographic-min over transforms).
                        proof: node.proof.clone(),
                    },
                )))
            }
            CertNode::FhwGateV1(gate) => Some(CertNode::FhwGateV1(Box::new(FhwGateNodeV1 {
                representatives: gate
                    .representatives
                    .iter()
                    .map(|edge| {
                        Some(CertEdge {
                            mv: d6_transform_coord(edge.mv, symmetry)?,
                            child: edge.child,
                        })
                    })
                    .collect::<Option<_>>()?,
                proof: FhwGateProofV1 {
                    schema_version: gate.proof.schema_version,
                    authority: gate.proof.authority.clone(),
                    threats: gate
                        .proof
                        .threats
                        .iter()
                        .map(|key| d6_transform_window(*key, symmetry))
                        .collect::<Option<_>>()?,
                    escape_resolution_ply: gate.proof.escape_resolution_ply,
                    map: gate
                        .proof
                        .map
                        .iter()
                        .map(|entry| {
                            Some(FhwMapV1 {
                                real_reply: d6_transform_coord(entry.real_reply, symmetry)?,
                                representative: d6_transform_coord(entry.representative, symmetry)?,
                                edge_class: entry.edge_class,
                                roles: entry
                                    .roles
                                    .iter()
                                    .map(|claim| {
                                        Some(FhwRoleClaimV1 {
                                            role: d6_transform_role_key(&claim.role, symmetry)?,
                                            child_f: claim.child_f,
                                            row: claim.row,
                                            epsilon: claim.epsilon,
                                        })
                                    })
                                    .collect::<Option<_>>()?,
                                windows: entry
                                    .windows
                                    .iter()
                                    .map(|claim| {
                                        Some(FhwWindowClaimV1 {
                                            window: d6_transform_window(claim.window, symmetry)?,
                                            child_q: claim.child_q,
                                            d_in_window: claim.d_in_window,
                                            s_in_window: claim.s_in_window,
                                            row: claim.row,
                                            kappa: claim.kappa,
                                            retained_guard: claim.retained_guard,
                                        })
                                    })
                                    .collect::<Option<_>>()?,
                            })
                        })
                        .collect::<Option<_>>()?,
                },
            }))),
        })
        .collect::<Option<_>>()?;

    let mut nodes: Vec<CertNode> = nodes;
    // The v1 extension class mandates canonical (transformed-move sorted)
    // edge order and canonically sorted Loss witness lists; a legacy
    // certificate's stored order is part of its identity and stays
    // untouched.
    if nodes.iter().any(CertNode::is_group2_extension) {
        for node in &mut nodes {
            match node {
                CertNode::Universal { edges, .. } => {
                    edges.sort_by_key(|edge| coord_key(edge.mv));
                }
                CertNode::UniversalGroup2V1(g2) => {
                    g2.edges.sort_by_key(|edge| coord_key(edge.mv));
                }
                CertNode::FhwGateV1(gate) => {
                    gate.representatives.sort_by_key(|edge| coord_key(edge.mv));
                }
                CertNode::Loss { witnesses, .. } => {
                    witnesses.sort_by_key(|key| {
                        let a = key.coord_at(0);
                        let b = key.coord_at(5);
                        let smaller = if (a.q, a.r) <= (b.q, b.r) { a } else { b };
                        (
                            match key.axis {
                                Axis::Q => 0u8,
                                Axis::R => 1,
                                Axis::QR => 2,
                            },
                            smaller.q,
                            smaller.r,
                        )
                    });
                }
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Choice { .. } => {}
            }
        }
    }
    Some(TssCertificate {
        root,
        claimant: cert.claimant,
        root_node: cert.root_node,
        nodes,
        semantic_horizon: cert.semantic_horizon,
    })
}
