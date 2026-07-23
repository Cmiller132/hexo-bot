//! Extension verifier for the v1 Group-2 certificate class
//! (.codex-g2-resolve/DESIGN_G2_CERT_EXTENSION.md + DESIGN_AMENDMENT_R1_R2.md).
//!
//! IMPLEMENTED SUB-CLASS (documented narrowing, see .codex-z/G2_IMPL_NOTES.md):
//! gate-free Group-2 certificates — trees containing `UniversalGroup2V1`
//! ordinary reduced-AND nodes but NO `FhwGateV1` node. Any certificate with a
//! gate node REJECTS. On gate-free trees the §3.2 cut clocks coincide with the
//! full clocks (the gate clauses are the only divergence), so the exact
//! per-role/per-window derivation below is the complete FHW obligation for
//! this sub-class.
//!
//! Design principles enforced throughout:
//! - conservative: every unspecified, ambiguous, or arithmetically-overflowing
//!   situation returns reject, never accept (checked arithmetic on all new
//!   paths; work caps reject);
//! - no acceptance oracle: stored scalars/digests are compared against fresh
//!   derivations from the replayed positions; `threats_shared::analyze` is
//!   used only as an ADDITIONAL rejector, never to accept;
//! - this module never imports `tss_solver`.

use std::collections::HashMap;

use hexo_engine::{
    hex_distance, Axis, HexCoord, HexoState as RustHexoState, Placement, Player, TurnPhase,
    WindowKey,
};

use crate::threats_shared;
use crate::tss_core::ProofStatus;
use crate::tss_verify::{
    certificate_metadata_for_group2, d6_transform_coord, validate_arena_for_group2, CertNode,
    CertNodeId, Group2AuthorityV1, RootBinding, TssCertificate, D6_SYMMETRY_COUNT, MAX_CERT_DEPTH,
};

/// Hard fail-closed work/memory limits (design §3.5). Reaching any limit is
/// rejection of the new certificate, never partial acceptance.
pub(crate) const MAX_G2_ROLES: usize = 1_000_000;
pub(crate) const MAX_G2_WORK_ITEMS: u64 = 10_000_000;

// ---------------------------------------------------------------------------
// SHA-256 (self-contained; no new crate dependency enters the verifier TCB).
// FIPS 180-4. Golden vectors are pinned in the test module below.
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub(crate) struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    total_len: u64,
}

impl Sha256 {
    pub(crate) fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: [0u8; 64],
            buffered: 0,
            total_len: 0,
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }

    pub(crate) fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    pub(crate) fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len.wrapping_mul(8);
        self.update(&[0x80]);
        while self.buffered != 56 {
            self.update(&[0x00]);
        }
        // Manual length append: update() would recount these bytes.
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);
        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

pub(crate) fn sha256(domain: &[u8], payload: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(payload);
    hash.finalize()
}

// ---------------------------------------------------------------------------
// Canonical scalar encoders (design §2.4 scalar grammar).
// ---------------------------------------------------------------------------

fn enc_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn enc_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn enc_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn enc_coord(out: &mut Vec<u8>, coord: HexCoord) {
    out.extend_from_slice(&coord.q.to_le_bytes());
    out.extend_from_slice(&coord.r.to_le_bytes());
}

fn axis_tag(axis: Axis) -> u8 {
    match axis {
        Axis::Q => 0,
        Axis::R => 1,
        Axis::QR => 2,
    }
}

/// Unoriented axis tag, then the numerically lexicographically smaller
/// endpoint (§2.4 window encoding).
fn enc_window(out: &mut Vec<u8>, key: WindowKey) {
    out.push(axis_tag(key.axis));
    let a = key.coord_at(0);
    let b = key.coord_at(5);
    let smaller = if (a.q, a.r) <= (b.q, b.r) { a } else { b };
    enc_coord(out, smaller);
}

fn window_sort_key(key: WindowKey) -> (u8, i16, i16) {
    let a = key.coord_at(0);
    let b = key.coord_at(5);
    let smaller = if (a.q, a.r) <= (b.q, b.r) { a } else { b };
    (axis_tag(key.axis), smaller.q, smaller.r)
}

fn player_tag(player: Player) -> u8 {
    match player {
        Player::Player0 => 0,
        Player::Player1 => 1,
    }
}

fn enc_authority(out: &mut Vec<u8>, authority: &Group2AuthorityV1) {
    out.extend_from_slice(&authority.defender_commit);
    enc_u64(out, authority.defender_path.len() as u64);
    out.extend_from_slice(authority.defender_path.as_bytes());
    out.extend_from_slice(&authority.defender_sha256);
    out.extend_from_slice(&authority.fhw_commit);
    enc_u64(out, authority.fhw_path.len() as u64);
    out.extend_from_slice(authority.fhw_path.as_bytes());
    out.extend_from_slice(&authority.fhw_sha256);
}

// ---------------------------------------------------------------------------
// Internal typed view of the (already structurally validated) certificate.
// ---------------------------------------------------------------------------

type CoordKey = (i16, i16);
type WinId = (u8, i16, i16);

fn coord_key(coord: HexCoord) -> CoordKey {
    (coord.q, coord.r)
}

fn win_id(key: WindowKey) -> WinId {
    (axis_tag(key.axis), key.start.q, key.start.r)
}

/// Identity of a live role: the discharge node plus the carried cell (and the
/// named witness for leaf-empty roles). Matches `RoleKeyV1` minus checkpoint
/// roles, which only exist at gates (excluded from this sub-class).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
enum RoleId {
    ChoiceMove {
        node: CertNodeId,
        cell: CoordKey,
    },
    OrCompletionMove {
        node: CertNodeId,
        cell: CoordKey,
    },
    LeafEmpty {
        node: CertNodeId,
        witness: WinId,
        cell: CoordKey,
    },
}

impl RoleId {
    fn carrier(&self) -> CoordKey {
        match self {
            RoleId::ChoiceMove { cell, .. }
            | RoleId::OrCompletionMove { cell, .. }
            | RoleId::LeafEmpty { cell, .. } => *cell,
        }
    }
}

/// Immutable origin bits for demanded windows (design §2.4).
const SOURCE_TOUCHED: u8 = 0x01;
const SOURCE_VIRGIN: u8 = 0x02;

struct G2Context<'a> {
    cert: &'a TssCertificate,
    claimant: Player,
    /// Exact replayed state per arena node (tree: exactly one occurrence).
    states: Vec<RustHexoState>,
    /// Outgoing (move, child) pairs in stored order.
    children: Vec<Vec<(HexCoord, CertNodeId)>>,
    /// Postorder over the reachable tree (children before parents).
    postorder: Vec<CertNodeId>,
    /// D14 full scalar local budget per node.
    b_local: Vec<u32>,
    /// Subtree maximum exact leaf resolution per node.
    t_sub: Vec<u32>,
    /// Live-role clock map per node (f_cut == r_full on gate-free trees; the
    /// gate clauses of §3.2 are the only divergence, so one derivation serves
    /// both sides of the required `f_cut <= r_full` inequality).
    roles: Vec<HashMap<RoleId, u32>>,
    /// Demanded windows with OR'd source bits per node (fixed point of the
    /// ordinary propagation rules; no gate-local demands in this sub-class).
    demands: Vec<Vec<(WindowKey, u8)>>,
    /// Q_cut == E_full memo per (node, window); design §3.2 recurrence.
    window_clock: HashMap<(CertNodeId, WinId), u32>,
    /// Derived k (0 or 1) at each Group-2 node; absent elsewhere.
    derived_k: HashMap<CertNodeId, u8>,
    /// Derived Required_FHW per Group-2 node (§3.4).
    required: HashMap<CertNodeId, Vec<HexCoord>>,
    /// Fail-closed work counter.
    work: u64,
}

impl<'a> G2Context<'a> {
    fn charge(&mut self, items: u64) -> Option<()> {
        self.work = self.work.checked_add(items)?;
        (self.work <= MAX_G2_WORK_ITEMS).then_some(())
    }
}

fn placements_remaining(state: &RustHexoState) -> u8 {
    match state.phase() {
        TurnPhase::Opening => 1,
        TurnPhase::FirstStone => 2,
        TurnPhase::SecondStone { .. } => 1,
    }
}

/// Direct window-mask reconstruction of "the mover could complete a window
/// this turn": some window holds `count(mover) >= 6 - remaining` stones with
/// zero opponent stones. This deliberately ignores empty-cell legality, so it
/// over-approximates true win-now positions: using it as a REJECTOR is sound
/// and conservative (notes deviation 2).
fn direct_own_win_now_upper(state: &RustHexoState) -> bool {
    let mover = state.current_player();
    let other = mover.other();
    let remaining = placements_remaining(state);
    state
        .board()
        .windows()
        .entries()
        .any(|entry| entry.count(other) == 0 && entry.count(mover).saturating_add(remaining) >= 6)
}

/// Z4 coupling-stability anchor (§3.4): an attacker stone or a root-binding
/// stone within hex distance eight of the placement.
fn anchored(
    state: &RustHexoState,
    claimant: Player,
    root_stones: &[HexCoord],
    mv: HexCoord,
) -> bool {
    state
        .board()
        .occupied_cells()
        .iter()
        .copied()
        .filter(|stone| state.board().get(*stone) == Some(claimant))
        .chain(root_stones.iter().copied())
        .any(|anchor| hex_distance(anchor, mv) <= 8)
}

fn window_has_claimant_stone(state: &RustHexoState, claimant: Player, key: WindowKey) -> bool {
    key.cells()
        .iter()
        .any(|cell| state.board().get(*cell) == Some(claimant))
}

fn window_defender_count(state: &RustHexoState, defender: Player, key: WindowKey) -> u32 {
    key.cells()
        .iter()
        .filter(|cell| state.board().get(**cell) == Some(defender))
        .count() as u32
}

fn window_is_all_empty(state: &RustHexoState, key: WindowKey) -> bool {
    key.cells()
        .iter()
        .all(|cell| state.board().get(*cell).is_none())
}

fn window_empty_cells(state: &RustHexoState, key: WindowKey) -> Vec<HexCoord> {
    key.cells()
        .iter()
        .copied()
        .filter(|cell| state.board().get(*cell).is_none())
        .collect()
}

/// Distance from a cell to a window: minimum over the six window cells.
fn window_distance(cell: HexCoord, key: WindowKey) -> u32 {
    key.cells()
        .iter()
        .map(|w| i32::from(hex_distance(cell, *w)) as u32)
        .min()
        .unwrap_or(u32::MAX)
}

fn sorted_legal_moves(state: &RustHexoState) -> Vec<HexCoord> {
    let mut legal = Vec::new();
    state.write_legal_moves(&mut legal);
    legal.sort_by_key(|coord| coord_key(*coord));
    legal
}

fn set_contains(sorted: &[HexCoord], coord: HexCoord) -> bool {
    sorted
        .binary_search_by_key(&coord_key(coord), |candidate| coord_key(*candidate))
        .is_ok()
}

// ---------------------------------------------------------------------------
// Top-level verification (design §3.1, narrowed as documented).
// ---------------------------------------------------------------------------

pub(crate) fn verify_group2_certificate(
    state: &RustHexoState,
    cert: &TssCertificate,
    claimed: ProofStatus,
) -> bool {
    verify_group2_impl(state, cert, claimed).is_some()
}

fn verify_group2_impl(
    state: &RustHexoState,
    cert: &TssCertificate,
    claimed: ProofStatus,
) -> Option<()> {
    // Unchanged root/status/claimant/arena/horizon checks (§3.1 step 1).
    if claimed == ProofStatus::Unknown || cert.root != RootBinding::from_state(state) {
        return None;
    }
    let expected_claimant = match claimed {
        ProofStatus::Win => state.current_player(),
        ProofStatus::Loss => state.current_player().other(),
        ProofStatus::Unknown => return None,
    };
    if cert.claimant != expected_claimant || !validate_arena_for_group2(cert) {
        return None;
    }
    let meta = certificate_metadata_for_group2(cert)?;
    // R1: derived T includes every gate escape deadline (folded into
    // `certificate_metadata`) and must fit the declared semantic horizon.
    if meta.derived_t > cert.semantic_horizon {
        return None;
    }
    // R1 second clause: every gate's escape deadline must individually fit
    // the declared horizon. Gates are subsequently rejected wholesale by the
    // narrowed class (below), so this is enforced a fortiori; the explicit
    // loop keeps the amended rule present in code.
    for node in &cert.nodes {
        if let CertNode::FhwGateV1(gate) = node {
            if gate.proof.escape_resolution_ply > cert.semantic_horizon {
                return None;
            }
        }
    }
    // NARROWING: FhwGateV1 validation (§3.3) is not implemented in this
    // session. A certificate containing any gate node rejects.
    if cert
        .nodes
        .iter()
        .any(|node| matches!(node, CertNode::FhwGateV1(_)))
    {
        return None;
    }
    // R2: the root position of any certificate containing a new-class node is
    // post-opening — explicit structural rule, not the accidental Z4 vacuity.
    if matches!(cert.root.phase, TurnPhase::Opening) {
        return None;
    }

    // Narrow-v1 structural preflight (§2.3).
    preflight_structure(cert)?;

    // Bind one exact state to every node and run the per-node direct checks.
    let mut ctx = build_context(state, cert)?;

    // Postorder derivations: D14 B, subtree resolution T, live-role clocks.
    derive_budgets_and_roles(&mut ctx)?;

    // Window demand fixed point + Q_cut/E_full evaluation.
    derive_window_demands(&mut ctx)?;

    // Per-Group-2-node class rules, zone coverage, and stored-scalar equality.
    check_group2_nodes(&mut ctx)?;

    // Digest recomputation and comparison (§2.4). Never an acceptance oracle
    // on its own — everything semantic above has already been re-derived —
    // but a mismatch rejects.
    check_digests(&mut ctx)?;

    Some(())
}

/// §2.3 structural rules that need no replay: exact tree shape, no mixing
/// with legacy zone/dispatch/commutation machinery, schema and authority
/// bytes, canonical edge order.
fn preflight_structure(cert: &TssCertificate) -> Option<()> {
    let nodes = cert.nodes.len();
    let mut indegree = vec![0u32; nodes];
    for node in &cert.nodes {
        match node {
            CertNode::Choice { child, .. } => {
                indegree[*child as usize] = indegree[*child as usize].checked_add(1)?;
            }
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            } => {
                // Class rule 2/3: a legacy Universal is admissible only as a
                // plain full-enumeration node (the full-set check itself needs
                // the replayed position and happens in build_context).
                if *implicit_dispatch || zone.is_some() || !commutations.is_empty() {
                    return None;
                }
                let mut previous: Option<CoordKey> = None;
                for edge in edges {
                    let key = coord_key(edge.mv);
                    if previous.is_some_and(|prev| prev >= key) {
                        return None;
                    }
                    previous = Some(key);
                    indegree[edge.child as usize] = indegree[edge.child as usize].checked_add(1)?;
                }
            }
            CertNode::UniversalGroup2V1(g2) => {
                if g2.proof.schema_version != 1 || !g2.proof.authority.matches_compiled() {
                    return None;
                }
                if g2.edges.is_empty() {
                    return None;
                }
                let mut previous: Option<CoordKey> = None;
                for edge in &g2.edges {
                    let key = coord_key(edge.mv);
                    if previous.is_some_and(|prev| prev >= key) {
                        return None;
                    }
                    previous = Some(key);
                    indegree[edge.child as usize] = indegree[edge.child as usize].checked_add(1)?;
                }
            }
            CertNode::FhwGateV1(_) => return None,
            CertNode::Loss { witnesses, .. } => {
                // Canonical sorted-unique witness order inside the new class.
                let mut previous: Option<(u8, i16, i16)> = None;
                for key in witnesses {
                    let sort_key = window_sort_key(*key);
                    if previous.is_some_and(|prev| prev >= sort_key) {
                        return None;
                    }
                    previous = Some(sort_key);
                }
            }
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => {}
        }
    }
    // Exact reachable tree: root indegree zero, every other node exactly one.
    // (Acyclicity and full reachability were established by validate_arena.)
    for (index, count) in indegree.iter().enumerate() {
        let expected = u32::from(index != cert.root_node as usize);
        if *count != expected {
            return None;
        }
    }
    Some(())
}

/// Replay the tree, binding one exact state per node, running every per-node
/// direct D9 check (§3.2) as the class requires.
fn build_context<'a>(root: &RustHexoState, cert: &'a TssCertificate) -> Option<G2Context<'a>> {
    let claimant = cert.claimant;
    let node_count = cert.nodes.len();
    let mut ctx = G2Context {
        cert,
        claimant,
        states: Vec::new(),
        children: vec![Vec::new(); node_count],
        postorder: Vec::with_capacity(node_count),
        b_local: vec![0; node_count],
        t_sub: vec![0; node_count],
        roles: Vec::new(),
        demands: vec![Vec::new(); node_count],
        window_clock: HashMap::new(),
        derived_k: HashMap::new(),
        required: HashMap::new(),
        work: 0,
    };
    // Tree replay: recursion with explicit depth cap. Each node is visited
    // exactly once (indegree checks), so `states` can be dense.
    let mut states: Vec<Option<RustHexoState>> = vec![None; node_count];
    let root_stones = cert.root.occupancy.clone();
    replay_node(
        cert,
        cert.root_node,
        root.clone(),
        claimant,
        &root_stones,
        0,
        &mut states,
        &mut ctx,
    )?;
    let states = states.into_iter().collect::<Option<Vec<_>>>()?;
    ctx.states = states;
    // Postorder (children before parents) over the tree.
    let mut order = Vec::with_capacity(node_count);
    let mut stack = vec![(cert.root_node, false)];
    while let Some((id, exiting)) = stack.pop() {
        if exiting {
            order.push(id);
            continue;
        }
        stack.push((id, true));
        for (_, child) in &ctx.children[id as usize] {
            stack.push((*child, false));
        }
        if stack.len() > node_count.checked_mul(2)?.checked_add(4)? {
            return None;
        }
    }
    if order.len() != node_count {
        return None;
    }
    ctx.postorder = order;
    Some(ctx)
}

#[allow(clippy::too_many_arguments)]
fn replay_node(
    cert: &TssCertificate,
    id: CertNodeId,
    state: RustHexoState,
    claimant: Player,
    root_stones: &[HexCoord],
    depth: usize,
    states: &mut [Option<RustHexoState>],
    ctx: &mut G2Context<'_>,
) -> Option<()> {
    if depth > MAX_CERT_DEPTH {
        return None;
    }
    ctx.charge(1)?;
    if states.get(id as usize)?.is_some() {
        return None; // tree property violated
    }
    match cert.nodes.get(id as usize)? {
        CertNode::OrCompletion {
            mv,
            witness,
            completion_ply,
        } => {
            check_or_completion(
                &state,
                claimant,
                root_stones,
                *mv,
                *witness,
                *completion_ply,
            )?;
            ctx.t_sub[id as usize] = *completion_ply;
        }
        CertNode::Win {
            witness,
            count,
            budget,
            resolution_ply,
        } => {
            check_win_leaf(
                &state,
                claimant,
                root_stones,
                *witness,
                *count,
                *budget,
                *resolution_ply,
            )?;
            ctx.t_sub[id as usize] = *resolution_ply;
        }
        CertNode::Loss {
            witnesses,
            resolution_ply,
        } => {
            let b = check_loss_leaf(&state, claimant, root_stones, witnesses, *resolution_ply)?;
            ctx.b_local[id as usize] = u32::from(b);
            ctx.t_sub[id as usize] = *resolution_ply;
        }
        CertNode::Choice { mv, child } => {
            if state.current_player() != claimant
                || state.is_terminal()
                || !anchored(&state, claimant, root_stones, *mv)
            {
                return None;
            }
            let mut next = state.clone();
            let result = next.apply_with_delta(Placement { coord: *mv }).ok()?.0;
            if result.outcome.is_some() {
                return None;
            }
            ctx.children[id as usize].push((*mv, *child));
            replay_node(
                cert,
                *child,
                next,
                claimant,
                root_stones,
                depth + 1,
                states,
                ctx,
            )?;
        }
        CertNode::Universal { edges, .. } => {
            // Legacy full-enumeration AND inside the new class: defender to
            // move, nonterminal, no win-now, and the edge set must be exactly
            // the full sorted legal set.
            if state.current_player() == claimant
                || state.is_terminal()
                || matches!(state.phase(), TurnPhase::Opening)
                || direct_own_win_now_upper(&state)
                || threats_shared::analyze(&state).own_win_now
            {
                return None;
            }
            let legal = sorted_legal_moves(&state);
            ctx.charge(legal.len() as u64)?;
            let moves: Vec<HexCoord> = edges.iter().map(|edge| edge.mv).collect();
            if moves != legal {
                return None;
            }
            for edge in edges {
                let mut next = state.clone();
                let result = next.apply_with_delta(Placement { coord: edge.mv }).ok()?.0;
                if result.outcome.is_some() {
                    return None;
                }
                ctx.children[id as usize].push((edge.mv, edge.child));
                replay_node(
                    cert,
                    edge.child,
                    next,
                    claimant,
                    root_stones,
                    depth + 1,
                    states,
                    ctx,
                )?;
            }
        }
        CertNode::UniversalGroup2V1(g2) => {
            // Class rule 4 (§2.3): post-opening, defender-to-move,
            // nonterminal, not own_win_now, and an exactly reconstructed
            // current k < b.
            if state.current_player() == claimant
                || state.is_terminal()
                || matches!(state.phase(), TurnPhase::Opening)
                || direct_own_win_now_upper(&state)
                || threats_shared::analyze(&state).own_win_now
            {
                return None;
            }
            let b = placements_remaining(&state);
            if !(1..=2).contains(&b) {
                return None;
            }
            let k = derive_exact_k(&state, claimant, ctx)?;
            if u32::from(k) >= u32::from(b) {
                return None;
            }
            ctx.derived_k.insert(id, k);
            let legal = sorted_legal_moves(&state);
            for edge in &g2.edges {
                if !set_contains(&legal, edge.mv) {
                    return None;
                }
                let mut next = state.clone();
                let result = next.apply_with_delta(Placement { coord: edge.mv }).ok()?.0;
                if result.outcome.is_some() {
                    return None;
                }
                ctx.children[id as usize].push((edge.mv, edge.child));
                replay_node(
                    cert,
                    edge.child,
                    next,
                    claimant,
                    root_stones,
                    depth + 1,
                    states,
                    ctx,
                )?;
            }
        }
        CertNode::FhwGateV1(_) => return None,
    }
    states[id as usize] = Some(state);
    Some(())
}

/// §3.2: enumerate the COMPLETE current claimant-threat family from the
/// replayed board and derive `k = tau` on the only accepted side of the
/// threshold: 0 iff empty, 1 iff every member shares a common cell, else >=2
/// (which rejects at both accepted budgets). Never trusts a capped
/// `min_hitting_set`.
fn derive_exact_k(state: &RustHexoState, claimant: Player, ctx: &mut G2Context<'_>) -> Option<u8> {
    let defender = claimant.other();
    let mut family: Vec<Vec<HexCoord>> = Vec::new();
    for entry in state.board().windows().entries() {
        ctx.charge(1)?;
        if entry.count(defender) == 0 && entry.count(claimant) >= 4 {
            let empties = entry.empty_cells();
            if empties.is_empty() {
                // A filled window is terminal; the node was required
                // nonterminal, so this is corrupt state.
                return None;
            }
            family.push(empties);
        }
    }
    if family.is_empty() {
        return Some(0);
    }
    let mut common = family[0].clone();
    for member in &family[1..] {
        ctx.charge(member.len() as u64)?;
        common.retain(|cell| member.contains(cell));
        if common.is_empty() {
            break;
        }
    }
    Some(if common.is_empty() { 2 } else { 1 })
}

fn check_or_completion(
    state: &RustHexoState,
    claimant: Player,
    root_stones: &[HexCoord],
    mv: HexCoord,
    witness: WindowKey,
    completion_ply: u32,
) -> Option<()> {
    if state.current_player() != claimant
        || state.is_terminal()
        || !witness.contains(mv)
        || !anchored(state, claimant, root_stones, mv)
        || completion_ply != state.placements_made().checked_add(1)?
    {
        return None;
    }
    let mut next = state.clone();
    let result = next.apply_with_delta(Placement { coord: mv }).ok()?.0;
    let outcome = result.outcome?;
    if outcome.winner != claimant {
        return None;
    }
    // Direct D9 mask check: the named window is completely claimant-filled.
    let filled = witness
        .cells()
        .iter()
        .all(|cell| next.board().get(*cell) == Some(claimant));
    filled.then_some(())
}

fn check_win_leaf(
    state: &RustHexoState,
    claimant: Player,
    root_stones: &[HexCoord],
    witness: WindowKey,
    count: u8,
    budget: u8,
    resolution_ply: u32,
) -> Option<()> {
    if state.is_terminal() || state.current_player() != claimant {
        return None;
    }
    let actual_budget = placements_remaining(state);
    let cells = witness.cells();
    let claimant_count = cells
        .iter()
        .filter(|cell| state.board().get(**cell) == Some(claimant))
        .count() as u8;
    let defender_count = cells
        .iter()
        .filter(|cell| state.board().get(**cell) == Some(claimant.other()))
        .count() as u8;
    if defender_count != 0 || claimant_count != count || budget != actual_budget {
        return None;
    }
    let empties: Vec<HexCoord> = cells
        .iter()
        .copied()
        .filter(|cell| state.board().get(*cell).is_none())
        .collect();
    let expected_resolution = match count {
        5 => state.placements_made().checked_add(1)?,
        4 if actual_budget == 2 => state.placements_made().checked_add(2)?,
        _ => return None,
    };
    if resolution_ply != expected_resolution {
        return None;
    }
    if !empties
        .iter()
        .all(|mv| anchored(state, claimant, root_stones, *mv))
    {
        return None;
    }
    // Replay the one/two legal continuations directly (§3.2).
    let mut replay = state.clone();
    let mut final_outcome = None;
    for mv in &empties {
        if final_outcome.is_some() {
            return None;
        }
        let result = replay.apply_with_delta(Placement { coord: *mv }).ok()?.0;
        final_outcome = result.outcome;
    }
    (final_outcome?.winner == claimant).then_some(())
}

/// Direct LOSS-leaf check. Returns the exact defender budget `b`.
fn check_loss_leaf(
    state: &RustHexoState,
    claimant: Player,
    root_stones: &[HexCoord],
    witnesses: &[WindowKey],
    resolution_ply: u32,
) -> Option<u8> {
    if state.is_terminal() || state.current_player() == claimant || witnesses.is_empty() {
        return None;
    }
    if direct_own_win_now_upper(state) || threats_shared::analyze(state).own_win_now {
        return None;
    }
    let b = placements_remaining(state);
    let mut empties = Vec::with_capacity(witnesses.len());
    for &key in witnesses {
        let cells = key.cells();
        let claimant_count = cells
            .iter()
            .filter(|cell| state.board().get(**cell) == Some(claimant))
            .count();
        let defender_count = cells
            .iter()
            .filter(|cell| state.board().get(**cell) == Some(claimant.other()))
            .count();
        if defender_count != 0 || claimant_count < 4 {
            return None;
        }
        let empty: Vec<HexCoord> = cells
            .iter()
            .copied()
            .filter(|cell| state.board().get(*cell).is_none())
            .collect();
        if empty.is_empty()
            || !empty
                .iter()
                .all(|mv| anchored(state, claimant, root_stones, *mv))
        {
            return None;
        }
        empties.push(empty);
    }
    // Exact tau > b at b in {1, 2}: no single cell (b=1) or no pair of cells
    // (b=2) hits every named member.
    if !family_hitting_exceeds(&empties, b) {
        return None;
    }
    let expected = state
        .placements_made()
        .checked_add(u32::from(b))?
        .checked_add(2)?;
    (resolution_ply == expected).then_some(b)
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

/// Postorder pass: D14 full scalar B, subtree resolution T, and the live-role
/// clock map (f_cut == r_full on gate-free trees, §3.2).
fn derive_budgets_and_roles(ctx: &mut G2Context<'_>) -> Option<()> {
    let mut roles: Vec<HashMap<RoleId, u32>> = vec![HashMap::new(); ctx.cert.nodes.len()];
    let mut total_roles = 0usize;
    for &id in &ctx.postorder.clone() {
        let index = id as usize;
        let mut map: HashMap<RoleId, u32> = HashMap::new();
        match &ctx.cert.nodes[index] {
            CertNode::OrCompletion { mv, .. } => {
                map.insert(
                    RoleId::OrCompletionMove {
                        node: id,
                        cell: coord_key(*mv),
                    },
                    0,
                );
                ctx.b_local[index] = 0;
            }
            CertNode::Win { witness, .. } => {
                for cell in window_empty_cells(&ctx.states[index], *witness) {
                    map.insert(
                        RoleId::LeafEmpty {
                            node: id,
                            witness: win_id(*witness),
                            cell: coord_key(cell),
                        },
                        0,
                    );
                }
                ctx.b_local[index] = 0;
            }
            CertNode::Loss { witnesses, .. } => {
                for key in witnesses {
                    for cell in window_empty_cells(&ctx.states[index], *key) {
                        map.insert(
                            RoleId::LeafEmpty {
                                node: id,
                                witness: win_id(*key),
                                cell: coord_key(cell),
                            },
                            0,
                        );
                    }
                }
                // b_local was set during replay (LOSS retains the remainder).
            }
            CertNode::Choice { mv, child } => {
                map.insert(
                    RoleId::ChoiceMove {
                        node: id,
                        cell: coord_key(*mv),
                    },
                    0,
                );
                for (role, f) in &roles[*child as usize] {
                    // Ordinary OR while live: pass-through.
                    if map.insert(*role, *f).is_some() {
                        return None;
                    }
                }
                ctx.b_local[index] = ctx.b_local[*child as usize];
                ctx.t_sub[index] = ctx.t_sub[*child as usize];
            }
            CertNode::Universal { .. } | CertNode::UniversalGroup2V1(_) => {
                let mut maximum_b = 0u32;
                let mut maximum_t = 0u32;
                for (_, child) in &ctx.children[index] {
                    let child_index = *child as usize;
                    maximum_b = maximum_b.max(ctx.b_local[child_index]);
                    maximum_t = maximum_t.max(ctx.t_sub[child_index]);
                    for (role, f) in &roles[child_index] {
                        // Ordinary AND: 1 + child clock. In a tree each role
                        // is reachable below exactly one child, so a repeat
                        // key is a structural corruption.
                        let bumped = f.checked_add(1)?;
                        if map.insert(*role, bumped).is_some() {
                            return None;
                        }
                    }
                }
                ctx.b_local[index] = maximum_b.checked_add(1)?;
                ctx.t_sub[index] = maximum_t;
            }
            CertNode::FhwGateV1(_) => return None,
        }
        ctx.charge(map.len() as u64)?;
        total_roles = total_roles.checked_add(map.len())?;
        if total_roles > MAX_G2_ROLES {
            return None;
        }
        roles[index] = map;
    }
    ctx.roles = roles;
    Some(())
}

/// Compute Q_cut(node, W) (== E_full on gate-free trees) with memoization.
fn window_clock(ctx: &mut G2Context<'_>, id: CertNodeId, key: WindowKey) -> Option<u32> {
    if let Some(value) = ctx.window_clock.get(&(id, win_id(key))) {
        return Some(*value);
    }
    ctx.charge(1)?;
    let index = id as usize;
    let value = if window_has_claimant_stone(&ctx.states[index], ctx.claimant, key) {
        // Non-D-alive: permanence stop (first clause, has precedence).
        0
    } else {
        match &ctx.cert.nodes[index] {
            CertNode::OrCompletion { .. } | CertNode::Win { .. } => 0,
            CertNode::Loss { .. } => ctx.b_local[index],
            CertNode::Choice { mv, child } => {
                if key.contains(*mv) {
                    // OR placement entering W.
                    0
                } else {
                    window_clock(ctx, *child, key)?
                }
            }
            CertNode::Universal { .. } | CertNode::UniversalGroup2V1(_) => {
                let children = ctx.children[index].clone();
                if children.is_empty() {
                    return None;
                }
                let mut maximum = 0u32;
                for (_, child) in children {
                    maximum = maximum.max(window_clock(ctx, child, key)?);
                }
                maximum.checked_add(1)?
            }
            CertNode::FhwGateV1(_) => return None,
        }
    };
    // Q_cut <= E_full <= B: on this gate-free class Q == E by construction;
    // the containment in B is checked here for every evaluated pair.
    if value > ctx.b_local[index] {
        return None;
    }
    ctx.window_clock.insert((id, win_id(key)), value);
    Some(value)
}

/// Seed and propagate the ordinary window demands (§3.2): all D-alive touched
/// windows at every Group-2 node plus the finite all-empty superset when
/// B >= 6; propagation stops below an OR entering W, at typed leaves, and at
/// the first non-D-alive node (recorded there with clock zero).
fn derive_window_demands(ctx: &mut G2Context<'_>) -> Option<()> {
    // Seeds per Group-2 node.
    let mut seeds: Vec<(CertNodeId, WindowKey, u8)> = Vec::new();
    for &id in &ctx.postorder.clone() {
        let index = id as usize;
        if !matches!(ctx.cert.nodes[index], CertNode::UniversalGroup2V1(_)) {
            continue;
        }
        let state = ctx.states[index].clone();
        let defender = ctx.claimant.other();
        for entry in state.board().windows().entries() {
            ctx.charge(1)?;
            if entry.count(ctx.claimant) == 0 && entry.count(defender) >= 1 {
                seeds.push((id, entry.key(), SOURCE_TOUCHED));
            }
        }
        let b = ctx.b_local[index];
        if b >= 6 {
            let radius = 8u32.checked_mul(b.checked_sub(6)?)?;
            let radius = i32::try_from(radius).ok()?;
            let legal = sorted_legal_moves(&state);
            let mut candidate_windows: Vec<WindowKey> = Vec::new();
            for c in &legal {
                // Cells x with dist(x, c) <= radius, then the 18 windows
                // through each x. d(c, W) <= radius iff W holds such a cell.
                // Charge the square-box enumeration before running it so an
                // adversarially large derived budget rejects instead of
                // spinning.
                let side = u64::try_from(radius.checked_mul(2)?.checked_add(1)?).ok()?;
                ctx.charge(side.checked_mul(side)?.checked_mul(19)?)?;
                let mut x_cells = Vec::new();
                for dq in -radius..=radius {
                    for dr in -radius..=radius {
                        let q = i32::from(c.q).checked_add(dq)?;
                        let r = i32::from(c.r).checked_add(dr)?;
                        let cell = HexCoord {
                            q: i16::try_from(q).ok()?,
                            r: i16::try_from(r).ok()?,
                        };
                        if i32::from(hex_distance(*c, cell)) <= radius {
                            x_cells.push(cell);
                        }
                    }
                }
                ctx.charge(x_cells.len() as u64)?;
                for x in x_cells {
                    for axis in Axis::ALL {
                        for offset in 0..6i16 {
                            let start = HexCoord {
                                q: x.q.checked_sub(axis.vector().q.checked_mul(offset)?)?,
                                r: x.r.checked_sub(axis.vector().r.checked_mul(offset)?)?,
                            };
                            candidate_windows.push(WindowKey { start, axis });
                        }
                    }
                }
            }
            candidate_windows.sort_by_key(|key| win_id(*key));
            candidate_windows.dedup_by_key(|key| win_id(*key));
            ctx.charge(candidate_windows.len() as u64)?;
            for key in candidate_windows {
                if window_is_all_empty(&state, key) {
                    seeds.push((id, key, SOURCE_VIRGIN));
                }
            }
        }
    }
    // Top-down propagation. Preorder = reverse postorder.
    let mut incoming: Vec<HashMap<WinId, (WindowKey, u8)>> =
        vec![HashMap::new(); ctx.cert.nodes.len()];
    for (id, key, bits) in seeds {
        let entry = incoming[id as usize].entry(win_id(key)).or_insert((key, 0));
        entry.1 |= bits;
    }
    let preorder: Vec<CertNodeId> = ctx.postorder.iter().rev().copied().collect();
    for &id in &preorder {
        let index = id as usize;
        let rows: Vec<(WindowKey, u8)> = {
            let mut rows: Vec<_> = incoming[index]
                .values()
                .map(|(key, bits)| (*key, *bits))
                .collect();
            rows.sort_by_key(|(key, _)| window_sort_key(*key));
            rows
        };
        ctx.charge(rows.len() as u64)?;
        for (key, bits) in &rows {
            // Evaluate the clock (also enforces the <= B containment).
            let _ = window_clock(ctx, id, *key)?;
            // Propagate downward unless a stop rule applies at this node.
            let non_d_alive = window_has_claimant_stone(&ctx.states[index], ctx.claimant, *key);
            if non_d_alive {
                continue;
            }
            match &ctx.cert.nodes[index] {
                CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {}
                CertNode::Choice { mv, child } => {
                    if !key.contains(*mv) {
                        let entry = incoming[*child as usize]
                            .entry(win_id(*key))
                            .or_insert((*key, 0));
                        entry.1 |= bits;
                    }
                }
                CertNode::Universal { .. } | CertNode::UniversalGroup2V1(_) => {
                    for (_, child) in ctx.children[index].clone() {
                        let entry = incoming[child as usize]
                            .entry(win_id(*key))
                            .or_insert((*key, 0));
                        entry.1 |= bits;
                    }
                }
                CertNode::FhwGateV1(_) => return None,
            }
        }
        ctx.demands[index] = rows;
    }
    Some(())
}

/// §3.4 zone construction plus the stored-scalar equality checks for every
/// Group-2 node.
fn check_group2_nodes(ctx: &mut G2Context<'_>) -> Option<()> {
    for &id in &ctx.postorder.clone() {
        let index = id as usize;
        let CertNode::UniversalGroup2V1(g2) = &ctx.cert.nodes[index] else {
            continue;
        };
        let g2 = g2.clone();
        let state = ctx.states[index].clone();
        // Stored scalars are evidence only; equality with the derivation is
        // mandatory.
        if g2.proof.claimed_d14_budget != ctx.b_local[index]
            || g2.proof.build_horizon != ctx.cert.semantic_horizon
        {
            return None;
        }
        let legal = sorted_legal_moves(&state);
        let stones = state.board().occupied_cells();
        let explicit: Vec<HexCoord> = g2.edges.iter().map(|edge| edge.mv).collect();
        if explicit.is_empty() {
            return None;
        }

        // Z_dir and Z_seed from the live-role clocks.
        let mut required: Vec<HexCoord> = Vec::new();
        let mut carrier_f: HashMap<CoordKey, u32> = HashMap::new();
        for (role, f) in &ctx.roles[index] {
            let carrier = role.carrier();
            let slot = carrier_f.entry(carrier).or_insert(0);
            *slot = (*slot).max(*f);
        }
        ctx.charge(carrier_f.len() as u64)?;
        for (carrier, f) in &carrier_f {
            let cell = HexCoord {
                q: carrier.0,
                r: carrier.1,
            };
            if set_contains(&legal, cell) {
                required.push(cell); // Z_dir
            } else if !stones.contains(&cell) && *f >= 1 {
                // Z_seed: Legal within B_{8(f-1)}(carrier).
                let radius = 8u32.checked_mul(f.checked_sub(1)?)?;
                ctx.charge(legal.len() as u64)?;
                for c in &legal {
                    if i32::from(hex_distance(*c, cell)) as u32 <= radius {
                        required.push(*c);
                    }
                }
            }
        }

        // Z_touch and Z_virgin from the demanded windows at this node.
        let demands = ctx.demands[index].clone();
        for (key, _bits) in &demands {
            let q = window_clock(ctx, id, *key)?;
            let defender_count = window_defender_count(&state, ctx.claimant.other(), *key);
            let claimant_blocked = window_has_claimant_stone(&state, ctx.claimant, *key);
            if !claimant_blocked && defender_count >= 1 && defender_count.checked_add(q)? >= 6 {
                required.extend(window_empty_cells(&state, *key)); // Z_touch
            }
            if !claimant_blocked
                && defender_count == 0
                && window_is_all_empty(&state, *key)
                && q >= 6
            {
                let radius = 8u32.checked_mul(q.checked_sub(6)?)?;
                ctx.charge(legal.len() as u64)?;
                for c in &legal {
                    if window_distance(*c, *key) <= radius {
                        required.push(*c); // Z_virgin
                    }
                }
            }
        }

        required.sort_by_key(|coord| coord_key(*coord));
        required.dedup();
        // Every required coordinate must be legal and covered by an explicit
        // edge. Supersets are valid; empty Required still needs the (already
        // enforced) nonempty explicit set.
        for cell in &required {
            if !set_contains(&legal, *cell) {
                return None;
            }
            if explicit
                .binary_search_by_key(&coord_key(*cell), |c| coord_key(*c))
                .is_err()
            {
                return None;
            }
        }
        ctx.required.insert(id, required);
    }
    Some(())
}

// ---------------------------------------------------------------------------
// §2.4 digest recomputation (gate-free payloads).
// ---------------------------------------------------------------------------

struct TransformTables {
    /// Per transform: preorder IDs after transformed-move edge sorting.
    pre_ids: Vec<Vec<u32>>,
    /// Per transform, per node: sorted outgoing (transformed move, child).
    sorted_children: Vec<Vec<Vec<(HexCoord, CertNodeId)>>>,
}

fn build_transform_tables(ctx: &G2Context<'_>) -> Option<TransformTables> {
    let node_count = ctx.cert.nodes.len();
    let mut pre_ids = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
    let mut sorted_children = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
    for symmetry in 0..D6_SYMMETRY_COUNT {
        let mut per_node = Vec::with_capacity(node_count);
        for children in &ctx.children {
            let mut transformed = children
                .iter()
                .map(|(mv, child)| Some((d6_transform_coord(*mv, symmetry)?, *child)))
                .collect::<Option<Vec<_>>>()?;
            transformed.sort_by_key(|(mv, _)| coord_key(*mv));
            if transformed
                .windows(2)
                .any(|pair| coord_key(pair[0].0) == coord_key(pair[1].0))
            {
                return None;
            }
            per_node.push(transformed);
        }
        // Depth-first preorder from the root, following sorted edges.
        let mut ids = vec![u32::MAX; node_count];
        let mut next_id = 0u32;
        let mut stack = vec![ctx.cert.root_node];
        while let Some(id) = stack.pop() {
            if ids[id as usize] != u32::MAX {
                return None;
            }
            ids[id as usize] = next_id;
            next_id = next_id.checked_add(1)?;
            for (_, child) in per_node[id as usize].iter().rev() {
                stack.push(*child);
            }
        }
        if ids.iter().any(|assigned| *assigned == u32::MAX) {
            return None;
        }
        pre_ids.push(ids);
        sorted_children.push(per_node);
    }
    Some(TransformTables {
        pre_ids,
        sorted_children,
    })
}

fn enc_state_record(out: &mut Vec<u8>, state: &RustHexoState, symmetry: u8) -> Option<()> {
    let mut stones: Vec<(HexCoord, Player)> = state
        .board()
        .occupied_cells()
        .iter()
        .map(|coord| {
            let owner = state.board().get(*coord)?;
            Some((d6_transform_coord(*coord, symmetry)?, owner))
        })
        .collect::<Option<Vec<_>>>()?;
    stones.sort_by_key(|(coord, owner)| (coord.q, coord.r, player_tag(*owner)));
    enc_u64(out, stones.len() as u64);
    for (coord, owner) in &stones {
        enc_coord(out, *coord);
        out.push(player_tag(*owner));
    }
    out.push(player_tag(state.current_player()));
    match state.phase() {
        TurnPhase::Opening => out.push(0),
        TurnPhase::FirstStone => out.push(1),
        TurnPhase::SecondStone { first } => {
            out.push(2);
            enc_coord(out, d6_transform_coord(first, symmetry)?);
        }
    }
    enc_u32(out, state.placements_made());
    match state.terminal() {
        None => out.push(0),
        Some(outcome) => {
            out.push(1);
            out.push(player_tag(outcome.winner));
            enc_u32(out, outcome.placements);
        }
    }
    Some(())
}

fn node_tag(node: &CertNode) -> u8 {
    match node {
        CertNode::OrCompletion { .. } => 0,
        CertNode::Win { .. } => 1,
        CertNode::Loss { .. } => 2,
        CertNode::Choice { .. } => 3,
        CertNode::Universal { .. } => 4,
        CertNode::UniversalGroup2V1(_) => 5,
        CertNode::FhwGateV1(_) => 6,
    }
}

fn transform_window(key: WindowKey, symmetry: u8) -> Option<WindowKey> {
    // Reuse the verifier's D6 window transform through the public coord
    // transform: map both endpoints and rebuild the canonical key.
    let first = d6_transform_coord(key.coord_at(0), symmetry)?;
    let second = d6_transform_coord(key.coord_at(1), symmetry)?;
    let dq = i32::from(second.q) - i32::from(first.q);
    let dr = i32::from(second.r) - i32::from(first.r);
    match (dq, dr) {
        (1, 0) => Some(WindowKey {
            start: first,
            axis: Axis::Q,
        }),
        (0, 1) => Some(WindowKey {
            start: first,
            axis: Axis::R,
        }),
        (1, -1) => Some(WindowKey {
            start: first,
            axis: Axis::QR,
        }),
        (-1, 0) => Some(WindowKey {
            start: d6_transform_coord(key.coord_at(5), symmetry)?,
            axis: Axis::Q,
        }),
        (0, -1) => Some(WindowKey {
            start: d6_transform_coord(key.coord_at(5), symmetry)?,
            axis: Axis::R,
        }),
        (-1, 1) => Some(WindowKey {
            start: d6_transform_coord(key.coord_at(5), symmetry)?,
            axis: Axis::QR,
        }),
        _ => None,
    }
}

/// Local semantic payload (§2.4): the node payload with outgoing edges and
/// child IDs removed; stored digest fields omitted.
fn enc_semantic_local(out: &mut Vec<u8>, node: &CertNode, symmetry: u8) -> Option<()> {
    out.push(node_tag(node));
    match node {
        CertNode::OrCompletion {
            mv,
            witness,
            completion_ply,
        } => {
            enc_coord(out, d6_transform_coord(*mv, symmetry)?);
            enc_window(out, transform_window(*witness, symmetry)?);
            enc_u32(out, *completion_ply);
        }
        CertNode::Win {
            witness,
            count,
            budget,
            resolution_ply,
        } => {
            enc_window(out, transform_window(*witness, symmetry)?);
            out.push(*count);
            out.push(*budget);
            enc_u32(out, *resolution_ply);
        }
        CertNode::Loss {
            witnesses,
            resolution_ply,
        } => {
            let mut keys = witnesses
                .iter()
                .map(|key| transform_window(*key, symmetry))
                .collect::<Option<Vec<_>>>()?;
            keys.sort_by_key(|key| window_sort_key(*key));
            enc_u64(out, keys.len() as u64);
            for key in keys {
                enc_window(out, key);
            }
            enc_u32(out, *resolution_ply);
        }
        CertNode::Choice { .. } => {}
        CertNode::Universal {
            implicit_dispatch,
            zone,
            commutations,
            ..
        } => {
            out.push(u8::from(*implicit_dispatch));
            match zone {
                None => out.push(0),
                Some(zone) => {
                    out.push(1);
                    enc_u32(out, zone.d);
                    enc_u32(out, zone.build_horizon);
                }
            }
            enc_u64(out, commutations.len() as u64);
            // The narrow class rejects commutations before hashing; encode
            // the count anyway so the grammar stays total.
        }
        CertNode::UniversalGroup2V1(g2) => {
            enc_u16(out, g2.proof.schema_version);
            enc_authority(out, &g2.proof.authority);
            enc_u32(out, g2.proof.claimed_d14_budget);
            enc_u32(out, g2.proof.build_horizon);
        }
        CertNode::FhwGateV1(_) => return None,
    }
    Some(())
}

struct DigestTables {
    /// semantic_hash[g][node]
    semantic: Vec<Vec<[u8; 32]>>,
    /// derived_hash[g][node]
    derived: Vec<Vec<[u8; 32]>>,
    transforms: TransformTables,
}

fn build_digest_tables(ctx: &mut G2Context<'_>) -> Option<DigestTables> {
    let transforms = build_transform_tables(ctx)?;
    let node_count = ctx.cert.nodes.len();
    let mut semantic = vec![vec![[0u8; 32]; node_count]; D6_SYMMETRY_COUNT as usize];
    let mut derived = vec![vec![[0u8; 32]; node_count]; D6_SYMMETRY_COUNT as usize];
    let postorder = ctx.postorder.clone();
    for symmetry in 0..D6_SYMMETRY_COUNT {
        let g = symmetry as usize;
        for &id in &postorder {
            let index = id as usize;
            ctx.charge(4)?;
            // Semantic Merkle value.
            let mut payload = Vec::new();
            enc_semantic_local(&mut payload, &ctx.cert.nodes[index], symmetry)?;
            let children = &transforms.sorted_children[g][index];
            enc_u64(&mut payload, children.len() as u64);
            for (mv, child) in children {
                enc_coord(&mut payload, *mv);
                payload.extend_from_slice(&semantic[g][*child as usize]);
            }
            semantic[g][index] = sha256(b"hexo-g2-semantic-node-v1\0", &payload);

            // Derived Merkle value.
            let mut record = Vec::new();
            enc_state_record(&mut record, &ctx.states[index], symmetry)?;
            record.push(node_tag(&ctx.cert.nodes[index]));
            enc_u32(&mut record, ctx.b_local[index]);
            enc_u32(&mut record, ctx.t_sub[index]);
            enc_u32(&mut record, ctx.cert.semantic_horizon);
            // Role rows.
            let mut role_rows: Vec<Vec<u8>> = Vec::with_capacity(ctx.roles[index].len());
            for (role, f) in &ctx.roles[index] {
                let mut row = Vec::new();
                enc_role_key(&mut row, role, symmetry, &transforms.pre_ids[g])?;
                let carrier = role.carrier();
                enc_coord(
                    &mut row,
                    d6_transform_coord(
                        HexCoord {
                            q: carrier.0,
                            r: carrier.1,
                        },
                        symmetry,
                    )?,
                );
                enc_u32(&mut row, *f); // r_full
                enc_u32(&mut row, *f); // f_cut (equal on gate-free trees)
                role_rows.push(row);
            }
            role_rows.sort();
            enc_u64(&mut record, role_rows.len() as u64);
            for row in role_rows {
                record.extend_from_slice(&row);
            }
            // Demand rows.
            let mut demand_rows: Vec<Vec<u8>> = Vec::with_capacity(ctx.demands[index].len());
            for (key, bits) in ctx.demands[index].clone() {
                let clock = window_clock(ctx, id, key)?;
                let mut row = Vec::new();
                enc_window(&mut row, transform_window(key, symmetry)?);
                row.push(bits);
                enc_u32(&mut row, clock); // E_full
                enc_u32(&mut row, clock); // Q_cut (equal on gate-free trees)
                demand_rows.push(row);
            }
            demand_rows.sort();
            enc_u64(&mut record, demand_rows.len() as u64);
            for row in demand_rows {
                record.extend_from_slice(&row);
            }
            // Derived class payload.
            match &ctx.cert.nodes[index] {
                CertNode::UniversalGroup2V1(_) => {
                    record.push(1); // OrdinaryGroup2
                    record.push(*ctx.derived_k.get(&id)?);
                    let mut cells = ctx
                        .required
                        .get(&id)?
                        .iter()
                        .map(|cell| d6_transform_coord(*cell, symmetry))
                        .collect::<Option<Vec<_>>>()?;
                    cells.sort_by_key(|coord| coord_key(*coord));
                    enc_u64(&mut record, cells.len() as u64);
                    for cell in cells {
                        enc_coord(&mut record, cell);
                    }
                }
                CertNode::FhwGateV1(_) => return None,
                _ => record.push(0), // Other
            }
            let mut payload = record;
            enc_u64(&mut payload, children.len() as u64);
            for (mv, child) in children {
                enc_coord(&mut payload, *mv);
                payload.extend_from_slice(&derived[g][*child as usize]);
            }
            derived[g][index] = sha256(b"hexo-g2-derived-node-v1\0", &payload);
        }
    }
    Some(DigestTables {
        semantic,
        derived,
        transforms,
    })
}

fn enc_role_key(out: &mut Vec<u8>, role: &RoleId, symmetry: u8, pre_ids: &[u32]) -> Option<()> {
    match role {
        RoleId::ChoiceMove { node, cell } => {
            out.push(0);
            enc_u32(out, *pre_ids.get(*node as usize)?);
            enc_coord(
                out,
                d6_transform_coord(
                    HexCoord {
                        q: cell.0,
                        r: cell.1,
                    },
                    symmetry,
                )?,
            );
        }
        RoleId::OrCompletionMove { node, cell } => {
            out.push(1);
            enc_u32(out, *pre_ids.get(*node as usize)?);
            enc_coord(
                out,
                d6_transform_coord(
                    HexCoord {
                        q: cell.0,
                        r: cell.1,
                    },
                    symmetry,
                )?,
            );
        }
        RoleId::LeafEmpty {
            node,
            witness,
            cell,
        } => {
            out.push(2);
            enc_u32(out, *pre_ids.get(*node as usize)?);
            let key = WindowKey {
                start: HexCoord {
                    q: witness.1,
                    r: witness.2,
                },
                axis: match witness.0 {
                    0 => Axis::Q,
                    1 => Axis::R,
                    2 => Axis::QR,
                    _ => return None,
                },
            };
            enc_window(out, transform_window(key, symmetry)?);
            enc_coord(
                out,
                d6_transform_coord(
                    HexCoord {
                        q: cell.0,
                        r: cell.1,
                    },
                    symmetry,
                )?,
            );
        }
    }
    Some(())
}

fn lexicographic_min(candidates: Vec<Vec<u8>>) -> Option<Vec<u8>> {
    candidates.into_iter().min()
}

/// Recompute and compare `child_plan_sha256` and `finder_summary_sha256` for
/// every Group-2 node.
fn check_digests(ctx: &mut G2Context<'_>) -> Option<()> {
    let tables = build_digest_tables(ctx)?;
    for &id in &ctx.postorder.clone() {
        let index = id as usize;
        let CertNode::UniversalGroup2V1(g2) = &ctx.cert.nodes[index] else {
            continue;
        };
        // child_plan_sha256.
        let mut plan_preimages = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let g = symmetry as usize;
            let mut preimage = Vec::new();
            enc_u16(&mut preimage, 1);
            enc_state_record(&mut preimage, &ctx.states[index], symmetry)?;
            let children = &tables.transforms.sorted_children[g][index];
            enc_u64(&mut preimage, children.len() as u64);
            for (mv, child) in children {
                enc_coord(&mut preimage, *mv);
                preimage.extend_from_slice(&tables.semantic[g][*child as usize]);
            }
            plan_preimages.push(preimage);
        }
        let child_plan = sha256(
            b"hexo-g2-child-plan-v1\0",
            &lexicographic_min(plan_preimages)?,
        );
        if child_plan != g2.proof.child_plan_sha256 {
            return None;
        }
        // finder_summary_sha256.
        let mut summary_preimages = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let g = symmetry as usize;
            let mut preimage = Vec::new();
            enc_u16(&mut preimage, 1);
            enc_authority(&mut preimage, &g2.proof.authority);
            enc_state_record(&mut preimage, &ctx.states[index], symmetry)?;
            preimage.extend_from_slice(&child_plan);
            preimage.extend_from_slice(&tables.derived[g][index]);
            summary_preimages.push(preimage);
        }
        let summary = sha256(
            b"hexo-g2-summary-v1\0",
            &lexicographic_min(summary_preimages)?,
        );
        if summary != g2.proof.finder_summary_sha256 {
            return None;
        }
    }
    Some(())
}

// ---------------------------------------------------------------------------
// Finder-facing helpers (called from tss_solver; this module never imports
// the solver). Deviation 3 in the notes: finder and verifier share these
// derivations, so the digest comparison detects drift/tampering, not
// correlated implementation bugs.
// ---------------------------------------------------------------------------

/// Exact §3.4 Required_FHW for a candidate Group-2 node under construction:
/// the node's state plus its proven `(move, child)` edges over `arena`.
/// Returns None when the subtree leaves the narrow class (gates, dispatch,
/// zoned or commuted nodes, DAG sharing is fine here — clocks are
/// state-determined) or any derivation fails; the caller must then fall back
/// to the legacy uniform path.
pub(crate) fn finder_required_fhw(
    state: &RustHexoState,
    claimant: Player,
    edges: &[(HexCoord, CertNodeId)],
    arena: &[CertNode],
) -> Option<Vec<HexCoord>> {
    // Build a temporary single-node certificate view: a synthetic Group-2
    // node above the given children. We reuse the verification derivations by
    // materializing the subtree as its own certificate.
    let mut nodes: Vec<CertNode> = Vec::new();
    let mut remap: HashMap<CertNodeId, CertNodeId> = HashMap::new();
    let mut new_edges = Vec::with_capacity(edges.len());
    for (mv, child) in edges {
        let child = copy_subtree(arena, *child, &mut nodes, &mut remap)?;
        new_edges.push(crate::tss_verify::CertEdge { mv: *mv, child });
    }
    new_edges.sort_by_key(|edge| coord_key(edge.mv));
    if new_edges
        .windows(2)
        .any(|pair| coord_key(pair[0].mv) == coord_key(pair[1].mv))
    {
        return None;
    }
    let synthetic =
        CertNode::UniversalGroup2V1(Box::new(crate::tss_verify::UniversalGroup2NodeV1 {
            edges: new_edges,
            proof: crate::tss_verify::Group2ZoneV1 {
                schema_version: 1,
                authority: Group2AuthorityV1::compiled(),
                claimed_d14_budget: 0,
                build_horizon: 0,
                child_plan_sha256: [0u8; 32],
                finder_summary_sha256: [0u8; 32],
            },
        }));
    let root_id = nodes.len() as CertNodeId;
    nodes.push(synthetic);
    let cert = TssCertificate {
        root: RootBinding::from_state(state),
        claimant,
        root_node: root_id,
        nodes,
        semantic_horizon: u32::MAX,
    };
    let mut ctx = build_context(state, &cert)?;
    derive_budgets_and_roles(&mut ctx)?;
    derive_window_demands(&mut ctx)?;
    // check_group2_nodes enforces required ⊆ explicit, which is exactly the
    // closure question; recompute required directly instead.
    compute_required_only(&mut ctx, root_id)
}

/// The §3.4 union for one node without the coverage requirement.
fn compute_required_only(ctx: &mut G2Context<'_>, id: CertNodeId) -> Option<Vec<HexCoord>> {
    let index = id as usize;
    let state = ctx.states[index].clone();
    let legal = sorted_legal_moves(&state);
    let stones = state.board().occupied_cells();
    let mut required: Vec<HexCoord> = Vec::new();
    let mut carrier_f: HashMap<CoordKey, u32> = HashMap::new();
    for (role, f) in &ctx.roles[index] {
        let carrier = role.carrier();
        let slot = carrier_f.entry(carrier).or_insert(0);
        *slot = (*slot).max(*f);
    }
    for (carrier, f) in &carrier_f {
        let cell = HexCoord {
            q: carrier.0,
            r: carrier.1,
        };
        if set_contains(&legal, cell) {
            required.push(cell);
        } else if !stones.contains(&cell) && *f >= 1 {
            let radius = 8u32.checked_mul(f.checked_sub(1)?)?;
            for c in &legal {
                if i32::from(hex_distance(*c, cell)) as u32 <= radius {
                    required.push(*c);
                }
            }
        }
    }
    let demands = ctx.demands[index].clone();
    for (key, _bits) in &demands {
        let q = window_clock(ctx, id, *key)?;
        let defender_count = window_defender_count(&state, ctx.claimant.other(), *key);
        let claimant_blocked = window_has_claimant_stone(&state, ctx.claimant, *key);
        if !claimant_blocked && defender_count >= 1 && defender_count.checked_add(q)? >= 6 {
            required.extend(window_empty_cells(&state, *key));
        }
        if !claimant_blocked && defender_count == 0 && window_is_all_empty(&state, *key) && q >= 6 {
            let radius = 8u32.checked_mul(q.checked_sub(6)?)?;
            for c in &legal {
                if window_distance(*c, *key) <= radius {
                    required.push(*c);
                }
            }
        }
    }
    required.sort_by_key(|coord| coord_key(*coord));
    required.dedup();
    Some(required)
}

/// Copy the subtree below `root` from a solver arena into `nodes`, unfolding
/// DAG sharing into a tree (each occurrence copied). Rejects when the copy
/// leaves the narrow class or exceeds arena caps.
fn copy_subtree(
    arena: &[CertNode],
    root: CertNodeId,
    nodes: &mut Vec<CertNode>,
    _remap: &mut HashMap<CertNodeId, CertNodeId>,
) -> Option<CertNodeId> {
    fn copy(
        arena: &[CertNode],
        id: CertNodeId,
        nodes: &mut Vec<CertNode>,
        depth: usize,
    ) -> Option<CertNodeId> {
        if depth > MAX_CERT_DEPTH || nodes.len() >= crate::tss_verify::MAX_CERT_NODES {
            return None;
        }
        let node = arena.get(id as usize)?;
        let copied = match node {
            CertNode::OrCompletion { .. } | CertNode::Win { .. } | CertNode::Loss { .. } => {
                node.clone()
            }
            CertNode::Choice { mv, child } => CertNode::Choice {
                mv: *mv,
                child: copy(arena, *child, nodes, depth + 1)?,
            },
            CertNode::Universal {
                edges,
                implicit_dispatch,
                zone,
                commutations,
            } => {
                if *implicit_dispatch || zone.is_some() || !commutations.is_empty() {
                    return None;
                }
                let mut new_edges = Vec::with_capacity(edges.len());
                for edge in edges {
                    new_edges.push(crate::tss_verify::CertEdge {
                        mv: edge.mv,
                        child: copy(arena, edge.child, nodes, depth + 1)?,
                    });
                }
                new_edges.sort_by_key(|edge| coord_key(edge.mv));
                CertNode::Universal {
                    edges: new_edges,
                    implicit_dispatch: false,
                    zone: None,
                    commutations: Vec::new(),
                }
            }
            CertNode::UniversalGroup2V1(g2) => {
                let mut new_edges = Vec::with_capacity(g2.edges.len());
                for edge in &g2.edges {
                    new_edges.push(crate::tss_verify::CertEdge {
                        mv: edge.mv,
                        child: copy(arena, edge.child, nodes, depth + 1)?,
                    });
                }
                new_edges.sort_by_key(|edge| coord_key(edge.mv));
                CertNode::UniversalGroup2V1(Box::new(crate::tss_verify::UniversalGroup2NodeV1 {
                    edges: new_edges,
                    proof: g2.proof.clone(),
                }))
            }
            CertNode::FhwGateV1(_) => return None,
        };
        let new_id = u32::try_from(nodes.len()).ok()?;
        nodes.push(copied);
        Some(new_id)
    }
    copy(arena, root, nodes, 0)
}

/// Post-compaction pass used by the finder: given an assembled certificate
/// whose Group-2 nodes carry placeholder scalars/digests, (1) sort every edge
/// and Loss-witness list into canonical order, (2) unfold DAG sharing into a
/// strict tree, (3) fill `claimed_d14_budget`, `build_horizon`, and both
/// digests from the same derivations the verifier replays. Returns None when
/// the certificate cannot be brought into the narrow class.
pub(crate) fn finder_finalize_group2(
    state: &RustHexoState,
    cert: &TssCertificate,
) -> Option<TssCertificate> {
    // Unfold to a strict tree rooted at root_node.
    let mut nodes: Vec<CertNode> = Vec::new();
    let mut remap = HashMap::new();
    let root = copy_subtree(&cert.nodes, cert.root_node, &mut nodes, &mut remap)?;
    // Canonicalize Loss witness order.
    for node in &mut nodes {
        if let CertNode::Loss { witnesses, .. } = node {
            witnesses.sort_by_key(|key| window_sort_key(*key));
            if witnesses
                .windows(2)
                .any(|pair| window_sort_key(pair[0]) == window_sort_key(pair[1]))
            {
                return None;
            }
        }
    }
    let mut out = TssCertificate {
        root: cert.root.clone(),
        claimant: cert.claimant,
        root_node: root,
        nodes,
        semantic_horizon: cert.semantic_horizon,
    };
    // Derive scalars on the unfolded tree.
    let mut ctx = build_context(state, &out)?;
    derive_budgets_and_roles(&mut ctx)?;
    derive_window_demands(&mut ctx)?;
    let b_local = ctx.b_local.clone();
    // Fill claimed scalars first (they enter the semantic hashes).
    for (index, node) in out.nodes.iter_mut().enumerate() {
        if let CertNode::UniversalGroup2V1(g2) = node {
            g2.proof.schema_version = 1;
            g2.proof.authority = Group2AuthorityV1::compiled();
            g2.proof.claimed_d14_budget = b_local[index];
            g2.proof.build_horizon = out.semantic_horizon;
        }
    }
    // Re-derive on the finalized scalar values and fill digests. The derived
    // k / required tables come from the checking pass.
    let mut ctx = build_context(state, &out)?;
    derive_budgets_and_roles(&mut ctx)?;
    derive_window_demands(&mut ctx)?;
    check_group2_nodes(&mut ctx)?;
    let tables = build_digest_tables(&mut ctx)?;
    let mut plans: HashMap<usize, ([u8; 32], [u8; 32])> = HashMap::new();
    for (index, node) in out.nodes.iter().enumerate() {
        let CertNode::UniversalGroup2V1(g2) = node else {
            continue;
        };
        let mut plan_preimages = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let g = symmetry as usize;
            let mut preimage = Vec::new();
            enc_u16(&mut preimage, 1);
            enc_state_record(&mut preimage, &ctx.states[index], symmetry)?;
            let children = &tables.transforms.sorted_children[g][index];
            enc_u64(&mut preimage, children.len() as u64);
            for (mv, child) in children {
                enc_coord(&mut preimage, *mv);
                preimage.extend_from_slice(&tables.semantic[g][*child as usize]);
            }
            plan_preimages.push(preimage);
        }
        let child_plan = sha256(
            b"hexo-g2-child-plan-v1\0",
            &lexicographic_min(plan_preimages)?,
        );
        let mut summary_preimages = Vec::with_capacity(D6_SYMMETRY_COUNT as usize);
        for symmetry in 0..D6_SYMMETRY_COUNT {
            let g = symmetry as usize;
            let mut preimage = Vec::new();
            enc_u16(&mut preimage, 1);
            enc_authority(&mut preimage, &g2.proof.authority);
            enc_state_record(&mut preimage, &ctx.states[index], symmetry)?;
            preimage.extend_from_slice(&child_plan);
            preimage.extend_from_slice(&tables.derived[g][index]);
            summary_preimages.push(preimage);
        }
        let summary = sha256(
            b"hexo-g2-summary-v1\0",
            &lexicographic_min(summary_preimages)?,
        );
        plans.insert(index, (child_plan, summary));
    }
    for (index, node) in out.nodes.iter_mut().enumerate() {
        if let CertNode::UniversalGroup2V1(g2) = node {
            let (plan, summary) = plans.get(&index)?;
            g2.proof.child_plan_sha256 = *plan;
            g2.proof.finder_summary_sha256 = *summary;
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
