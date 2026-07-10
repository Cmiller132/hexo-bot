//! Train-read row expansion kernel (Rust + rayon, runs GIL-free).
//!
//! Per row this performs: a D6 transform of every stored coordinate fact, a
//! depth-(radius+1) multi-source BFS support build, the feature build, and the
//! legal-slot policy projection. Rows run across `rows.par_iter()` under
//! `py.detach` (GIL released); `collect` preserves input order. The stacked
//! result is exposed as zero-copy buffers consumed Python-side.
//!
//! This mirrors the Python expansion chain: `support.py::_build_support`,
//! `features.py::build_features`, `samples.py::expand_sample`/`_legal_slot`, and
//! `geometry.py::apply_d6`. Inputs are the stored `hexfield_compact_v1` facts:
//! the unified placement history, the phase, and the first stone. Legality is
//! derived in closed form (`empty ∧ dist <= radius`); the graded per-axis window
//! planes are recomputed from the D6-transformed placement history (via
//! `WindowStore::from_placements`), not read from the shard. The Rust/Python
//! element-wise parity test across all 12 D6 values lives in
//! `tests/test_hexfield_eq_rust_parity.py`.
//!
//! OFF-LEGAL: when `tolerate_off_legal`, an off-legal SELF policy target flags the
//! row INVALID in the returned `valid` mask (the row is not dropped in-worker);
//! otherwise it is a hard error. The caller filters survivors / permutes /
//! truncates on the main thread, so the survivor set is a function of
//! `(row, d6, radius)`.
//!
//! DETERMINISM: the per-row `d6: i32[n]` vector is pre-drawn on the main thread
//! and passed positionally; the kernel makes no rng call. `par_iter().collect()`
//! preserves input order, so the output is independent of worker count.

use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::ptr;

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rayon::prelude::*;

use hexo_engine::{HexCoord, Player, WindowStore};

use crate::constants::{
    f_opp_fork, f_own_fork, feature_version, num_axis_planes, num_features, DIRECTIONS,
    DIST_SCALE, F_DIST_TO_STONE, F_EMPTY, F_FIRST_STONE, F_LEGAL, F_OPP_LAST_TURN,
    F_OPP_RECENCY, F_OPP_STONE, F_OWN_LINE_Q, F_OWN_RECENCY, F_OWN_STONE, F_PHASE_SECOND,
    F_PLAYER_COLOUR, RAYLEN_SLOTS,
};

// hexfield_compact_v1 phase enum: index 2 == "SecondStone".
const PHASE_SECOND_STONE: u8 = 2;
// moves_left is normalized to [-1, 1] over [0, CAP]. Must match the Python
// constant of the same name.
const MOVES_LEFT_CAP: f32 = 209.0;
// Packed action id = ((q + COORD_OFFSET) << 16) | (r + COORD_OFFSET).
const COORD_OFFSET: i32 = 1 << 15;

// =============================================================================
// Geometry (mirrors geometry.py) — integer math, no floats.
// =============================================================================

/// `geometry.rot60`: (-r, q+r).
#[inline]
fn rot60(q: i32, r: i32) -> (i32, i32) {
    (-r, q + r)
}

/// `geometry.reflect`: (q, -q-r).
#[inline]
fn reflect(q: i32, r: i32) -> (i32, i32) {
    (q, -q - r)
}

/// `geometry.apply_d6`: index 0-5 == rot60^i; 6-11 == rot60^(i-6) ∘ reflect
/// (reflect first when index >= 6, then rotate).
#[inline]
fn apply_d6(index: i32, q: i32, r: i32) -> (i32, i32) {
    let (mut q, mut r) = (q, r);
    let mut idx = index;
    if idx >= 6 {
        let (nq, nr) = reflect(q, r);
        q = nq;
        r = nr;
        idx -= 6;
    }
    for _ in 0..idx {
        let (nq, nr) = rot60(q, r);
        q = nq;
        r = nr;
    }
    (q, r)
}

/// `geometry.unpack_action_id`: inverse of pack_action_id.
#[inline]
fn unpack_action_id(action_id: u32) -> (i32, i32) {
    let q = ((action_id >> 16) & 0xFFFF) as i32 - COORD_OFFSET;
    let r = (action_id & 0xFFFF) as i32 - COORD_OFFSET;
    (q, r)
}

// =============================================================================
// Per-row stored facts (one PackedRowView's worth, copied out of the byte
// buffers on the main thread so workers own their data — no borrow of the npz).
// =============================================================================

struct RowFacts {
    // (q, r, owner, placement_index)
    records: Vec<(i32, i32, u8, u32)>,
    current_player: u8,
    phase: u8,
    first_stone: Option<(i32, i32)>,
    // (action_id, weight)
    policy: Vec<(u32, f32)>,
    q_policy: Vec<(u32, f32)>,
    // Improved-policy target π' and raw root logits, both as (action_id, value)
    // aligned to `policy`. Empty `gumbel_policy` means no target (shard
    // `gumbel_present == 0`): the projection emits an all-zero gumbel_policy
    // with gumbel_policy_valid 0.
    gumbel_policy: Vec<(u32, f32)>,
    prior_logit: Vec<(u32, f32)>,
    opp_policy: Vec<(u32, f32)>,
    policy_surprise: f32,
    value: f32,
    // (H,) stvalue + mask blocks
    stvalue: Vec<f32>,
    stvalue_mask: Vec<f32>,
    moves_left: f32,
    // 1 == completed game (grounded outcome); 0 == truncated (no engine winner).
    // When 0, the value/stvalue/cell_q heads are masked to zero loss.
    outcome_valid: u8,
}

/// One expanded row's flat arrays (mirrors `ExpandedRow`). Invalid rows carry
/// zero-length node/policy vecs and `valid=false`.
struct RowOut {
    valid: bool,
    legal_count: i32,
    stone_count: i32,
    halo_count: i32,
    // node-major: coords (2N), dist (N), nbr (6N), feats (NUM_FEATURES*N),
    // raylen (RAYLEN_SLOTS*N)
    coords: Vec<i32>,
    dist: Vec<i32>,
    nbr: Vec<i32>,
    feats: Vec<f32>,
    raylen: Vec<u8>,
    // legal-prefix targets (legal_count each)
    policy: Vec<f32>,
    opp_policy: Vec<f32>,
    cell_q: Vec<f32>,
    cell_q_mask: Vec<f32>,
    // Dense (legal_count) improved-policy target, renormalized over the kept
    // support and all-zero when absent, plus a presence flag and the dense
    // (legal_count) raw root logits.
    gumbel_policy: Vec<f32>,
    gumbel_policy_valid: f32,
    prior_logit: Vec<f32>,
    policy_surprise: f32,
    // opp_coverage is an f64 ratio of f64-accumulated sums, matching the Python
    // float computation. Other scalars are f32: value widens losslessly;
    // moves_left = 2*min(1, ml/CAP) - 1 is emitted as f32.
    opp_coverage: f64,
    value: f32,
    value_mask: f32,
    stvalue: Vec<f32>,
    stvalue_mask: Vec<f32>,
    moves_left: f32,
    moves_left_mask: f32,
}

// =============================================================================
// Support build (mirrors support.py::_build_support) — closed-form legality.
// =============================================================================

struct Support {
    /// [legal | stones | halo], each ascending by (q, r).
    coords: Vec<(i32, i32)>,
    legal_count: usize,
    stone_count: usize,
    halo_count: usize,
    dist: Vec<i32>,
    index: HashMap<(i32, i32), usize>,
}

impl Support {
    #[inline]
    fn num_nodes(&self) -> usize {
        self.coords.len()
    }
}

/// `support.py::_build_support(stones)`. `stones` is the D6-transformed stone
/// coordinate list; order is irrelevant since it is deduped into a set.
/// `radius`/`halo` correspond to `_SUPPORT_RADIUS`/`_SUPPORT_HALO`.
fn build_support(stones: &[(i32, i32)], radius: i32, halo: i32) -> Support {
    if stones.is_empty() {
        // Ply 0: origin plus its 6 halo neighbours (7 nodes, 1 legal), all dist 0.
        // coords = [(0,0)] + sorted(DIRECTIONS).
        let mut dirs: Vec<(i32, i32)> = DIRECTIONS.iter().map(|&(dq, dr)| (dq as i32, dr as i32)).collect();
        dirs.sort();
        let mut coords = Vec::with_capacity(7);
        coords.push((0, 0));
        coords.extend(dirs);
        let dist = vec![0i32; coords.len()];
        let index = build_index(&coords);
        return Support {
            coords,
            legal_count: 1,
            stone_count: 0,
            halo_count: 6,
            dist,
            index,
        };
    }

    // Multi-source BFS to depth `halo` (== radius+1) from the stones. dist is
    // seeded from the deduped stone set, so duplicates do not double-seed.
    let mut dist_map: HashMap<(i32, i32), i32> = HashMap::with_capacity(stones.len() * 16);
    let mut frontier: std::collections::VecDeque<(i32, i32)> =
        std::collections::VecDeque::with_capacity(stones.len() * 8);
    let mut stone_set: HashMap<(i32, i32), ()> = HashMap::with_capacity(stones.len());
    for &s in stones {
        if stone_set.insert(s, ()).is_none() {
            dist_map.insert(s, 0);
            frontier.push_back(s);
        }
    }
    while let Some(cell) = frontier.pop_front() {
        let d = dist_map[&cell];
        if d == halo {
            continue;
        }
        let (q, r) = cell;
        for &(dq, dr) in &DIRECTIONS {
            let nxt = (q + dq as i32, r + dr as i32);
            if !dist_map.contains_key(&nxt) {
                dist_map.insert(nxt, d + 1);
                frontier.push_back(nxt);
            }
        }
    }

    // legal = empty ∧ dist <= radius (NOT a stone); stones = sorted set;
    // halo = dist == halo. Each segment ascending by (q, r).
    let mut legal: Vec<(i32, i32)> = dist_map
        .iter()
        .filter(|(c, &d)| d <= radius && !stone_set.contains_key(*c))
        .map(|(&c, _)| c)
        .collect();
    legal.sort();
    let mut stones_sorted: Vec<(i32, i32)> = stone_set.keys().copied().collect();
    stones_sorted.sort();
    let mut halo_cells: Vec<(i32, i32)> = dist_map
        .iter()
        .filter(|(_, &d)| d == halo)
        .map(|(&c, _)| c)
        .collect();
    halo_cells.sort();

    let legal_count = legal.len();
    let stone_count = stones_sorted.len();
    let halo_count = halo_cells.len();
    let mut coords = legal;
    coords.extend(stones_sorted);
    coords.extend(halo_cells);
    let dist: Vec<i32> = coords.iter().map(|c| dist_map[c]).collect();
    let index = build_index(&coords);
    Support {
        coords,
        legal_count,
        stone_count,
        halo_count,
        dist,
        index,
    }
}

#[inline]
fn build_index(coords: &[(i32, i32)]) -> HashMap<(i32, i32), usize> {
    let mut index = HashMap::with_capacity(coords.len());
    for (i, &c) in coords.iter().enumerate() {
        index.insert(c, i);
    }
    index
}

/// `support._neighbor_table`: (N,6) row-local neighbour index per DIRECTIONS,
/// -1 when absent. Returned node-major flat (row*6 + k).
fn neighbor_table(coords: &[(i32, i32)], index: &HashMap<(i32, i32), usize>) -> Vec<i32> {
    let n = coords.len();
    let mut nbr = vec![-1i32; n * 6];
    for (row, &(q, r)) in coords.iter().enumerate() {
        for (k, &(dq, dr)) in DIRECTIONS.iter().enumerate() {
            if let Some(&j) = index.get(&(q + dq as i32, r + dr as i32)) {
                nbr[row * 6 + k] = j as i32;
            }
        }
    }
    nbr
}

// =============================================================================
// Phase / player ordinal derivation (features.py::record_phase/record_player).
// =============================================================================

const REC_PHASE_OPENING: u8 = 0;
const REC_PHASE_FIRST: u8 = 1;
const REC_PHASE_SECOND: u8 = 2;

#[inline]
fn record_phase(ordinal: usize) -> u8 {
    if ordinal == 0 {
        return REC_PHASE_OPENING;
    }
    if (ordinal - 1) % 2 == 0 {
        REC_PHASE_FIRST
    } else {
        REC_PHASE_SECOND
    }
}

#[inline]
fn record_player(ordinal: usize) -> i32 {
    if ordinal == 0 {
        return 0;
    }
    if ((ordinal - 1) / 2) % 2 == 0 {
        1
    } else {
        0
    }
}

/// `features._opp_last_turn_cells`: reversed-history scan over the records
/// (which carry the D6-transformed coords already).
fn opp_last_turn_cells(records: &[(i32, i32, u8, u32)], current_player: i32) -> Vec<(i32, i32)> {
    let opponent = 1 - current_player;
    let n = records.len();
    for ordinal in (0..n).rev() {
        if record_player(ordinal) != opponent {
            continue;
        }
        let phase = record_phase(ordinal);
        let (q, r, _o, _i) = records[ordinal];
        if phase == REC_PHASE_SECOND {
            // ordinal-1 is the opponent's first-stone companion (ordinal >= 1
            // here since phase is SecondStone).
            let (fq, fr, _o2, _i2) = records[ordinal - 1];
            return vec![(fq, fr), (q, r)];
        }
        if phase == REC_PHASE_OPENING {
            return vec![(q, r)];
        }
        // FirstStone: skip (mid-turn).
    }
    Vec::new()
}

// =============================================================================
// Feature build (mirrors features.py::build_features) from D6-transformed facts.
// =============================================================================

/// Build the (N*NUM_FEATURES) node-major feature matrix. `records` carries the
/// D6-transformed coords; `first_stone` is likewise transformed. The graded
/// window planes are recomputed here from the transformed placements (via
/// `WindowStore::from_placements`) rather than read from the shard.
///
/// A cell absent from the support is returned as `ExpandErr::Hard` (a clean error
/// return rather than a panic crossing the rayon/FFI boundary). On valid decision
/// states every fact cell is in the support.
fn build_features(
    sup: &Support,
    records: &[(i32, i32, u8, u32)],
    current_player: i32,
    phase: u8,
    first_stone: Option<(i32, i32)>,
) -> Result<Vec<f32>, ExpandErr> {
    let n = sup.num_nodes();
    let nf = num_features();
    let mut feats = vec![0f32; n * nf];
    let placements_made = records.len() as i64;

    let lookup = |sup: &Support, cell: (i32, i32), what: &str| -> Result<usize, ExpandErr> {
        sup.index
            .get(&cell)
            .copied()
            .ok_or_else(|| ExpandErr::Hard(format!("{what} cell {cell:?} missing from support")))
    };

    // Stones + recency. age = placements_made - placement_index;
    // weight = 1/(1+age); max-accumulate.
    for &(q, r, owner, placement_index) in records {
        let row = lookup(sup, (q, r), "stone")?;
        let recency_plane = if owner as i32 == current_player {
            feats[row * nf + F_OWN_STONE] = 1.0;
            F_OWN_RECENCY
        } else {
            feats[row * nf + F_OPP_STONE] = 1.0;
            F_OPP_RECENCY
        };
        let age = placements_made - placement_index as i64;
        // `1.0 / (1.0 + age)` is computed in f64 then cast to f32, matching the
        // Python path which computes in f64 and stores into a float32 array.
        // Computing in f32 directly can differ in the last ULP for non-dyadic
        // ratios (e.g. 1/3).
        let weight = (1.0f64 / (1.0 + age as f64)) as f32;
        let off = row * nf + recency_plane;
        if weight > feats[off] {
            feats[off] = weight;
        }
    }

    // EMPTY = 1 - own - opp; LEGAL on the legal prefix.
    for row in 0..n {
        let own = feats[row * nf + F_OWN_STONE];
        let opp = feats[row * nf + F_OPP_STONE];
        feats[row * nf + F_EMPTY] = 1.0 - own - opp;
    }
    for row in 0..sup.legal_count {
        feats[row * nf + F_LEGAL] = 1.0;
    }

    // Phase-second + first-stone.
    if phase == PHASE_SECOND_STONE {
        for row in 0..n {
            feats[row * nf + F_PHASE_SECOND] = 1.0;
        }
        if let Some(fs) = first_stone {
            let row = lookup(sup, fs, "first_stone")?;
            feats[row * nf + F_FIRST_STONE] = 1.0;
        }
    }

    // Player colour.
    if current_player == 0 {
        for row in 0..n {
            feats[row * nf + F_PLAYER_COLOUR] = 1.0;
        }
    }

    // Graded per-axis window planes (11-24 under version 1, 11-42 under
    // version 2): recompute from the transformed placements. Build the same
    // incremental window store the engine maintains during play, then read the
    // graded features per support cell. `me`/`other` are the current player's
    // own/opp perspective.
    {
        let placements: Vec<(HexCoord, Player)> = records
            .iter()
            .map(|&(q, r, o, _)| {
                (
                    HexCoord { q: q as i16, r: r as i16 },
                    if o == 0 { Player::Player0 } else { Player::Player1 },
                )
            })
            .collect();
        let windows = WindowStore::from_placements(&placements);
        let me = if current_player == 0 { Player::Player0 } else { Player::Player1 };
        let other = me.other();
        let n_axis = num_axis_planes();
        let (own_fork, opp_fork) = (f_own_fork(), f_opp_fork());
        for row in 0..n {
            let (q, r) = sup.coords[row];
            let x = HexCoord { q: q as i16, r: r as i16 };
            // Empty iff neither stone plane was set for this row above.
            let is_empty = feats[row * nf + F_OWN_STONE] == 0.0
                && feats[row * nf + F_OPP_STONE] == 0.0;
            let vals = crate::features::window_feature_row(&windows, x, is_empty, me, other);
            let base = row * nf;
            for k in 0..n_axis {
                feats[base + F_OWN_LINE_Q + k] = vals[k];
            }
            feats[base + own_fork] = vals[30];
            feats[base + opp_fork] = vals[31];
        }
    }

    // dist_to_stone: dist / DIST_SCALE.
    for row in 0..n {
        feats[row * nf + F_DIST_TO_STONE] = sup.dist[row] as f32 / DIST_SCALE;
    }

    // Opponent last full turn.
    for cell in opp_last_turn_cells(records, current_player) {
        let row = lookup(sup, cell, "opp_last_turn")?;
        feats[row * nf + F_OPP_LAST_TURN] = 1.0;
    }

    // Version-2 global scalar planes (spec §1.4), from the transformed records
    // (chronological — the centroid sum order matches the Python oracle).
    if feature_version() == 2 {
        let stones: Vec<(i32, i32)> = records.iter().map(|&(q, r, _, _)| (q, r)).collect();
        crate::features::fill_global_scalars(&mut feats, nf, n, &stones, |row| sup.coords[row]);
    }

    Ok(feats)
}

/// (N * RAYLEN_SLOTS) node-major side-relative ray lengths over the
/// reconstructed board (the train twin of `features::build_ray_lengths`,
/// sharing `features::ray_length_row` — the L1 walk cannot drift between the
/// serve and train paths). `records` carries the D6-transformed coords.
fn build_ray_lengths(
    sup: &Support,
    records: &[(i32, i32, u8, u32)],
    current_player: i32,
) -> Vec<u8> {
    let owner_at: HashMap<(i32, i32), u8> =
        records.iter().map(|&(q, r, o, _)| ((q, r), o)).collect();
    let me = current_player as u8;
    let on_support = |q: i32, r: i32| sup.index.contains_key(&(q, r));
    let owner = |q: i32, r: i32| owner_at.get(&(q, r)).copied();
    let n = sup.num_nodes();
    let mut out = vec![0u8; n * RAYLEN_SLOTS];
    for row in 0..n {
        let (q, r) = sup.coords[row];
        let vals = crate::features::ray_length_row(&on_support, &owner, q, r, me);
        out[row * RAYLEN_SLOTS..(row + 1) * RAYLEN_SLOTS].copy_from_slice(&vals);
    }
    out
}

// =============================================================================
// Policy projection (mirrors samples.py::_legal_slot + expand_sample).
// =============================================================================

/// `samples._legal_slot`: unpack the action id, apply D6, look up the support
/// slot; None when off-support or not in the legal prefix.
#[inline]
fn legal_slot(sup: &Support, sym: i32, action_id: u32) -> Option<usize> {
    let (q, r) = unpack_action_id(action_id);
    let (tq, tr) = apply_d6(sym, q, r);
    match sup.index.get(&(tq, tr)) {
        Some(&slot) if slot < sup.legal_count => Some(slot),
        _ => None,
    }
}

/// Expand one row under symmetry `sym` (mirrors samples.py::expand_sample).
/// Returns `Err(ExpandErr::OffLegal)` for a tolerated off-legal SELF policy
/// target (row flagged invalid) and `Err(ExpandErr::Hard)` for a hard error.
/// Numeric errors (non-finite / negative / zero-mass policy) always hard-error.
fn expand_one(
    facts: &RowFacts,
    sym: i32,
    radius: i32,
    halo: i32,
    horizons_len: usize,
    tolerate_off_legal: bool,
) -> Result<RowOut, ExpandErr> {
    // (1) Transform every stored coordinate fact (see transform_facts).
    let records: Vec<(i32, i32, u8, u32)> = facts
        .records
        .iter()
        .map(|&(q, r, o, p)| {
            let (tq, tr) = apply_d6(sym, q, r);
            (tq, tr, o, p)
        })
        .collect();
    let first_stone = facts.first_stone.map(|(q, r)| apply_d6(sym, q, r));

    // (2) Support from the transformed stones.
    let stones: Vec<(i32, i32)> = records.iter().map(|&(q, r, _, _)| (q, r)).collect();
    let sup = build_support(&stones, radius, halo);
    let legal_count = sup.legal_count;

    // (3) Features + ray lengths (both recomputed from the transformed facts).
    let feats = build_features(
        &sup,
        &records,
        facts.current_player as i32,
        facts.phase,
        first_stone,
    )?;
    let raylen = build_ray_lengths(&sup, &records, facts.current_player as i32);

    // (4) Self policy projection. An off-legal target is a hard error unless
    // `tolerate_off_legal`, in which case it flags the row invalid.
    let mut policy = vec![0f32; legal_count];
    let mut total = 0.0f32;
    for &(action_id, w) in &facts.policy {
        if !w.is_finite() || w < 0.0 {
            return Err(ExpandErr::Hard(
                "policy weights must be finite and nonnegative".to_string(),
            ));
        }
        match legal_slot(&sup, sym, action_id) {
            Some(slot) => {
                policy[slot] += w;
                total += w;
            }
            None => {
                if tolerate_off_legal {
                    return Err(ExpandErr::OffLegal);
                }
                return Err(ExpandErr::Hard(format!(
                    "policy target action {action_id} is off the legal set (hard error)"
                )));
            }
        }
    }
    // Every recorded row is a full (policy-bearing) row in main_9: fast rows are
    // dropped at the self-play writer and never reach expand, so the policy
    // target must always carry positive mass.
    if total <= 0.0 {
        return Err(ExpandErr::Hard(
            "policy target must carry positive mass".to_string(),
        ));
    }

    // (5) Opp policy projection: drop off-legal targets, track coverage. Off-legal
    // opp targets never raise.
    //
    // `opp[slot] += w` accumulates in f32 (matching numpy's in-place float32 add).
    // The coverage scalars `opp_total` / `opp_kept` accumulate in f64, matching
    // the Python float computation of `opp_kept / opp_total`.
    let mut opp = vec![0f32; legal_count];
    let mut opp_total = 0.0f64;
    let mut opp_kept = 0.0f64;
    for &(action_id, w) in &facts.opp_policy {
        if !w.is_finite() || w < 0.0 {
            return Err(ExpandErr::Hard(
                "opp policy weights must be finite and nonnegative".to_string(),
            ));
        }
        opp_total += w as f64;
        if let Some(slot) = legal_slot(&sup, sym, action_id) {
            opp[slot] += w;
            opp_kept += w as f64;
        }
    }
    let opp_coverage: f64 = if opp_total > 0.0 { opp_kept / opp_total } else { 1.0 };

    // (5b) Per-cell Q projection: scalar assign plus presence mask. Off-legal
    // targets are dropped (never raise); q must be finite and in [-1, 1].
    let mut cell_q = vec![0f32; legal_count];
    let mut cell_q_mask = vec![0f32; legal_count];
    for &(action_id, q) in &facts.q_policy {
        if !q.is_finite() || q < -1.0 || q > 1.0 {
            return Err(ExpandErr::Hard(
                "cell_q targets must be finite and in [-1, 1]".to_string(),
            ));
        }
        if let Some(slot) = legal_slot(&sup, sym, action_id) {
            cell_q[slot] = q;        // SCALAR assign (one action -> one distinct cell)
            cell_q_mask[slot] = 1.0;
        }
    }

    // (5c) Improved-policy target π' projection and raw root logits: project the
    // per-action π' weights onto this row's legal set and renormalize over the
    // kept (on-legal) support so the dense target sums to 1 when present. When
    // absent (gumbel_present == 0, i.e. empty facts.gumbel_policy) the target is
    // all-zero with gumbel_policy_valid 0.0. prior_logit is a scalar assign.
    // `gumbel_policy[slot] += w` accumulates in f32 (numpy float32 in-place add);
    // the renormalizer `g_total` accumulates in f64 matching the Python oracle's
    // scalar accumulation (samples.py expand_sample), then the f32 array is
    // divided by the one-rounding `g_total as f32` — the file's f64-with-one-
    // rounding convention, mirroring numpy's float32 /= python-float.
    let mut gumbel_policy = vec![0f32; legal_count];
    let mut g_total = 0.0f64;
    for &(action_id, w) in &facts.gumbel_policy {
        if !w.is_finite() || w < 0.0 {
            return Err(ExpandErr::Hard(
                "gumbel policy weights must be finite and nonnegative".to_string(),
            ));
        }
        if let Some(slot) = legal_slot(&sup, sym, action_id) {
            gumbel_policy[slot] += w;
            g_total += w as f64;
        }
    }
    let gumbel_policy_valid = if !facts.gumbel_policy.is_empty() && g_total > 0.0 {
        let g_total = g_total as f32;
        for w in gumbel_policy.iter_mut() {
            *w /= g_total; // renormalize over the kept support
        }
        1.0f32
    } else {
        // No target, or a target with no mass on-legal: emit an all-zero
        // distribution with valid 0.0.
        for w in gumbel_policy.iter_mut() {
            *w = 0.0;
        }
        0.0f32
    };
    let mut prior_logit = vec![0f32; legal_count];
    for &(action_id, l) in &facts.prior_logit {
        if !l.is_finite() {
            return Err(ExpandErr::Hard(
                "prior_logit values must be finite".to_string(),
            ));
        }
        if let Some(slot) = legal_slot(&sup, sym, action_id) {
            prior_logit[slot] = l; // SCALAR assign (one action -> one cell)
        }
    }

    // (6) STV + moves_left (D6-invariant). Columns with a zero stvalue_mask are
    // re-zeroed so only masked columns carry a value.
    let mut stvalue = facts.stvalue[..horizons_len].to_vec();
    let mut stvalue_mask = facts.stvalue_mask[..horizons_len].to_vec();
    for c in 0..horizons_len {
        if !(stvalue_mask[c] > 0.0) {
            stvalue[c] = 0.0;
        }
    }
    // Computed in f64 then cast to f32, matching the Python `2.0 * min(1.0,
    // moves_left / MOVES_LEFT_CAP) - 1.0` (f64) stored into a float32 tensor.
    // A negative stored moves_left is a sentinel that masks the head (mask 0.0).
    let (moves_left, moves_left_mask) = if facts.moves_left >= 0.0 {
        let m = 2.0f64 * (facts.moves_left as f64 / MOVES_LEFT_CAP as f64).min(1.0) - 1.0;
        (m as f32, 1.0f32)
    } else {
        (0.0f32, 0.0f32)
    };

    // (7) Truncated-game outcome masking. A truncated row (outcome_valid == 0, no
    // engine winner) has no grounded terminal outcome, so value_mask,
    // stvalue_mask, and cell_q_mask are zeroed (gating the value/stvalue/cell_q
    // heads to zero loss) while the policy/opp_policy heads (and moves_left,
    // masked via its sentinel above) train normally. Only the masks are zeroed;
    // the stvalue/cell_q target arrays are left as built. Completed rows
    // (outcome_valid == 1) keep value_mask 1.0 and the presence masks as built.
    let value_mask = if facts.outcome_valid == 0 {
        for c in 0..horizons_len {
            stvalue_mask[c] = 0.0;
        }
        for m in cell_q_mask.iter_mut() {
            *m = 0.0;
        }
        0.0f32
    } else {
        1.0f32
    };
    let nbr = neighbor_table(&sup.coords, &sup.index);
    let mut coords_flat = Vec::with_capacity(sup.num_nodes() * 2);
    for &(q, r) in &sup.coords {
        coords_flat.push(q);
        coords_flat.push(r);
    }

    Ok(RowOut {
        valid: true,
        legal_count: legal_count as i32,
        stone_count: sup.stone_count as i32,
        halo_count: sup.halo_count as i32,
        coords: coords_flat,
        dist: sup.dist,
        nbr,
        feats,
        raylen,
        policy,
        opp_policy: opp,
        cell_q,
        cell_q_mask,
        gumbel_policy,
        gumbel_policy_valid,
        prior_logit,
        policy_surprise: facts.policy_surprise,
        opp_coverage,
        value: facts.value,
        value_mask,
        stvalue,
        stvalue_mask,
        moves_left,
        moves_left_mask,
    })
}

enum ExpandErr {
    /// Off-legal SELF policy target under tolerate_off_legal: flag row invalid.
    OffLegal,
    /// A hard error to surface to Python.
    Hard(String),
}

// =============================================================================
// Zero-copy output buffers (PlaneBuffer ABI; same pattern as serve_pack.rs).
// =============================================================================

macro_rules! out_buffer {
    ($name:ident, $ty:ty) => {
        #[pyclass]
        pub struct $name {
            data: Vec<$ty>,
        }
        #[pymethods]
        impl $name {
            fn __len__(&self) -> usize {
                self.data.len() * std::mem::size_of::<$ty>()
            }
            /// SAFETY: read-only 1-D byte view over `data`, keeping `slf` alive.
            unsafe fn __getbuffer__(
                slf: Bound<'_, Self>,
                view: *mut ffi::Py_buffer,
                flags: c_int,
            ) -> PyResult<()> {
                if view.is_null() {
                    return Err(PyBufferError::new_err("buffer view is null"));
                }
                if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
                    (*view).obj = ptr::null_mut();
                    return Err(PyBufferError::new_err("buffer is read-only"));
                }
                let guard = slf.borrow();
                let data = &guard.data;
                (*view).buf = data.as_ptr() as *mut c_void;
                (*view).len = (data.len() * std::mem::size_of::<$ty>()) as ffi::Py_ssize_t;
                (*view).readonly = 1;
                (*view).itemsize = 1;
                (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                    b"B\0".as_ptr() as *mut c_char
                } else {
                    ptr::null_mut()
                };
                (*view).ndim = 1;
                (*view).shape = ptr::null_mut();
                (*view).strides = ptr::null_mut();
                (*view).suboffsets = ptr::null_mut();
                (*view).internal = ptr::null_mut();
                (*view).obj = slf.clone().into_any().into_ptr();
                Ok(())
            }
            unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
        }
    };
}

out_buffer!(RxF32Buf, f32);
out_buffer!(RxF64Buf, f64);
out_buffer!(RxI32Buf, i32);
out_buffer!(RxI64Buf, i64);
out_buffer!(RxU8Buf, u8);

// =============================================================================
// Column extraction (reinterpret the PackedWindow byte buffers + CSR offsets).
// =============================================================================

/// Reinterpret a `&[u8]` as a typed slice using native endianness (same pattern
/// as serve_pack.rs for the wire buffers). Length-checked against `count`.
fn as_typed<'a, T: Copy>(bytes: &'a [u8], count: usize, name: &str) -> PyResult<&'a [T]> {
    let want = count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| PyValueError::new_err(format!("{name}: length overflow")))?;
    if bytes.len() != want {
        return Err(PyValueError::new_err(format!(
            "{name}: {} bytes, expected {} ({} items)",
            bytes.len(),
            want,
            count
        )));
    }
    // SAFETY: length checked; T is POD with no invalid bit patterns (i16/u8/u16/
    // u32/f32/i64); PyBytes is malloc-aligned and the arrays are contiguous; the
    // source byte buffers are alive for the call.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) })
}

/// Pull a column's bytes from the dict and reinterpret to a typed slice.
fn col_typed<'a, T: Copy>(
    columns: &'a Bound<'_, PyDict>,
    key: &str,
    count: usize,
) -> PyResult<&'a [T]> {
    let item = columns
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("columns missing '{key}'")))?;
    let bytes = item.downcast::<PyBytes>()?.as_bytes();
    // Reinterpret over the borrowed bytes. The Bound keeps the PyBytes alive for
    // the duration of `columns` (the caller holds it), so the slice is valid.
    let typed = as_typed::<T>(bytes, count, key)?;
    // Transmute the lifetime to 'a (tied to `columns`): the PyBytes objects live
    // in the dict for the whole call, so this is sound.
    Ok(unsafe { std::mem::transmute::<&[T], &'a [T]>(typed) })
}

/// Optional variant of [`col_typed`]: returns `None` when the column is absent
/// (e.g. a packed window without the `gumbel_present` / `gumbel_pol_w` /
/// `prior_logit` columns). A present-but-wrong-length column still errors via
/// `as_typed`.
fn col_typed_opt<'a, T: Copy>(
    columns: &'a Bound<'_, PyDict>,
    key: &str,
    count: usize,
) -> PyResult<Option<&'a [T]>> {
    match columns.get_item(key)? {
        Some(item) => {
            let bytes = item.downcast::<PyBytes>()?.as_bytes();
            let typed = as_typed::<T>(bytes, count, key)?;
            Ok(Some(unsafe {
                std::mem::transmute::<&[T], &'a [T]>(typed)
            }))
        }
        None => Ok(None),
    }
}

// =============================================================================
// Entry point — expand a window's rows under their pre-drawn D6 symmetries.
// =============================================================================

/// Expand the rows named by `row_index` (into the packed window columns) under
/// their pre-drawn `d6` symmetries, in parallel and GIL-free.
///
/// `columns` is a dict of the `hexfield_compact_v1` window columns, each value a
/// `bytes` object (LE/native) EXCEPT the offset arrays which arrive as the typed
/// keys below. Required keys (all `bytes` unless noted):
///   scalars[n]:  current_player(u8), phase(u8), value(f32), moves_left(f32),
///                policy_surprise(f32), outcome_valid(u8),
///                first_q(i16), first_r(i16), first_present(u8)
///   blocks[n*H]: stvalue(f32), stvalue_mask(f32)
///   hist CSR:    hist_qr(i16, 2*L), hist_owner(u8, L), hist_pidx(u16, L),
///                hist_off(i64, n+1)
///   pol/opp CSR: pol_act(u32), pol_w(f32), pol_off(i64, n+1);
///                opp_act(u32), opp_w(f32), opp_off(i64, n+1)
/// The graded per-axis window planes are recomputed from `hist_*` inside the
/// kernel, so no hot/win cell CSR columns are read (schema-25).
/// `n` is the window row count (the column length); `row_index: i64[r]` selects
/// the subset to expand (aligned 1:1 with `d6: i32[r]`). `horizons_len` is H,
/// `support_radius` the model radius (== `HEXFIELD_EQ_SUPPORT_RADIUS`).
///
/// Returns a dict of zero-copy buffers + per-(expanded-row) CSR offsets:
///   valid(RxU8Buf[r]), legal_count/stone_count/halo_count(RxI32Buf[r]),
///   node_off(i64[r+1]), pol_off_out(i64[r+1]),
///   coords(RxI32Buf, 2*ΣN), dist(RxI32Buf, ΣN), nbr(RxI32Buf, 6*ΣN),
///   feats(RxF32Buf, NUM_FEATURES*ΣN), raylen(RxU8Buf, RAYLEN_SLOTS*ΣN),
///   policy(RxF32Buf, ΣL), opp_policy(RxF32Buf, ΣL),
///   opp_coverage/value/value_mask/moves_left/moves_left_mask(RxF32Buf[r]),
///   stvalue(RxF32Buf, r*H), stvalue_mask(RxF32Buf, r*H).
#[pyfunction]
#[pyo3(signature = (columns, n, row_index, d6, horizons_len, support_radius, tolerate_off_legal))]
pub fn expand_shard_train<'py>(
    py: Python<'py>,
    columns: &Bound<'py, PyDict>,
    n: usize,
    row_index: Vec<i64>,
    d6: Vec<i32>,
    horizons_len: usize,
    support_radius: i32,
    tolerate_off_legal: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let r = row_index.len();
    if d6.len() != r {
        return Err(PyValueError::new_err(format!(
            "d6 length {} != row_index length {r}",
            d6.len()
        )));
    }
    let halo = support_radius + 1;

    // --- reinterpret the scalar + block columns (length n) -------------------
    let current_player = col_typed::<u8>(columns, "current_player", n)?;
    let phase = col_typed::<u8>(columns, "phase", n)?;
    let value = col_typed::<f32>(columns, "value", n)?;
    let policy_surprise = col_typed::<f32>(columns, "policy_surprise", n)?;
    let moves_left = col_typed::<f32>(columns, "moves_left", n)?;
    let first_q = col_typed::<i16>(columns, "first_q", n)?;
    let first_r = col_typed::<i16>(columns, "first_r", n)?;
    let first_present = col_typed::<u8>(columns, "first_present", n)?;
    // outcome_valid[i] (u8): 1 completed / 0 truncated. When 0, the value/stvalue/
    // cell_q heads are masked to zero loss.
    let outcome_valid = col_typed::<u8>(columns, "outcome_valid", n)?;
    // gumbel_present[i] (u8): 1 means this row carries a π' target. Optional: None
    // for a packed window without the column, in which case every row is treated
    // as absent (visit fallback).
    let gumbel_present = col_typed_opt::<u8>(columns, "gumbel_present", n)?;
    let stvalue = col_typed::<f32>(columns, "stvalue", n * horizons_len)?;
    let stvalue_mask = col_typed::<f32>(columns, "stvalue_mask", n * horizons_len)?;

    // --- CSR offset arrays (length n+1) --------------------------------------
    let hist_off = col_typed::<i64>(columns, "hist_off", n + 1)?;
    let pol_off = col_typed::<i64>(columns, "pol_off", n + 1)?;
    let opp_off = col_typed::<i64>(columns, "opp_off", n + 1)?;

    // --- CSR data arrays (length from the offset tails) ----------------------
    let hist_total = *hist_off.last().unwrap() as usize;
    let pol_total = *pol_off.last().unwrap() as usize;
    let opp_total = *opp_off.last().unwrap() as usize;
    let hist_qr = col_typed::<i16>(columns, "hist_qr", 2 * hist_total)?;
    let hist_owner = col_typed::<u8>(columns, "hist_owner", hist_total)?;
    let hist_pidx = col_typed::<u16>(columns, "hist_pidx", hist_total)?;
    let pol_act = col_typed::<u32>(columns, "pol_act", pol_total)?;
    let pol_w = col_typed::<f32>(columns, "pol_w", pol_total)?;
    let q_pol_q = col_typed::<f32>(columns, "q_pol_q", pol_total)?;
    // π' target on its OWN CSR group (schema v3; support can exceed pol_act).
    // Optional: absent for legacy windows, which instead may carry the v2
    // aligned column below. The raw logit stays aligned to pol_act.
    let gumbel_off = col_typed_opt::<i64>(columns, "gumbel_off", n + 1)?;
    let gumbel_total = gumbel_off.map(|o| *o.last().unwrap() as usize).unwrap_or(0);
    let gumbel_act = col_typed_opt::<u32>(columns, "gumbel_act", gumbel_total)?;
    let gumbel_w = col_typed_opt::<f32>(columns, "gumbel_w", gumbel_total)?;
    // Legacy v2 storage: π' weight aligned to pol_act. Read only when the CSR
    // group is absent.
    let gumbel_pol_w = col_typed_opt::<f32>(columns, "gumbel_pol_w", pol_total)?;
    let prior_logit_col = col_typed_opt::<f32>(columns, "prior_logit", pol_total)?;
    let opp_act = col_typed::<u32>(columns, "opp_act", opp_total)?;
    let opp_w = col_typed::<f32>(columns, "opp_w", opp_total)?;

    // --- materialize per-row facts on the MAIN thread (workers own their data,
    // no borrow of the PyBytes inside par_iter so the kernel is GIL-free) ------
    let mut facts: Vec<RowFacts> = Vec::with_capacity(r);
    for &ri64 in &row_index {
        let i = ri64 as usize;
        if i >= n {
            return Err(PyValueError::new_err(format!(
                "row_index entry {i} out of range for n={n}"
            )));
        }
        let h0 = hist_off[i] as usize;
        let h1 = hist_off[i + 1] as usize;
        let records: Vec<(i32, i32, u8, u32)> = (h0..h1)
            .map(|k| {
                (
                    hist_qr[2 * k] as i32,
                    hist_qr[2 * k + 1] as i32,
                    hist_owner[k],
                    hist_pidx[k] as u32,
                )
            })
            .collect();
        let p0 = pol_off[i] as usize;
        let p1 = pol_off[i + 1] as usize;
        let policy: Vec<(u32, f32)> = (p0..p1).map(|k| (pol_act[k], pol_w[k])).collect();
        let q_policy: Vec<(u32, f32)> = (p0..p1).map(|k| (pol_act[k], q_pol_q[k])).collect();
        // Carry the π' target / logits only when this row is flagged present and
        // the columns exist; otherwise empty (visit fallback).
        let row_gumbel_present = gumbel_present.map(|g| g[i]).unwrap_or(0);
        let (gumbel_policy, prior_logit_facts): (Vec<(u32, f32)>, Vec<(u32, f32)>) =
            if row_gumbel_present != 0 {
                // Prefer the v3 CSR group (π' on its own support); fall back to
                // the v2 pol_act-aligned column for legacy windows.
                let gp: Vec<(u32, f32)> = match (gumbel_off, gumbel_act, gumbel_w) {
                    (Some(goff), Some(gact), Some(gwv)) => {
                        let g0 = goff[i] as usize;
                        let g1 = goff[i + 1] as usize;
                        (g0..g1).map(|k| (gact[k], gwv[k])).collect()
                    }
                    _ => gumbel_pol_w
                        .map(|g| (p0..p1).map(|k| (pol_act[k], g[k])).collect())
                        .unwrap_or_default(),
                };
                let pl = prior_logit_col
                    .map(|l| (p0..p1).map(|k| (pol_act[k], l[k])).collect())
                    .unwrap_or_default();
                (gp, pl)
            } else {
                (Vec::new(), Vec::new())
            };
        let o0 = opp_off[i] as usize;
        let o1 = opp_off[i + 1] as usize;
        let opp_policy: Vec<(u32, f32)> = (o0..o1).map(|k| (opp_act[k], opp_w[k])).collect();
        let first_stone = if first_present[i] == 1 {
            Some((first_q[i] as i32, first_r[i] as i32))
        } else {
            None
        };
        let stv = stvalue[i * horizons_len..(i + 1) * horizons_len].to_vec();
        let stv_mask = stvalue_mask[i * horizons_len..(i + 1) * horizons_len].to_vec();
        facts.push(RowFacts {
            records,
            current_player: current_player[i],
            phase: phase[i],
            first_stone,
            policy,
            q_policy,
            gumbel_policy,
            prior_logit: prior_logit_facts,
            opp_policy,
            policy_surprise: policy_surprise[i],
            value: value[i],
            stvalue: stv,
            stvalue_mask: stv_mask,
            moves_left: moves_left[i],
            outcome_valid: outcome_valid[i],
        });
    }

    // --- expand in parallel under py.detach (GIL released) -------------------
    // par_iter().collect() preserves input order, so the output is independent of
    // worker count. A Hard error in any row aborts the whole call (surfaced at the
    // first offending row below).
    let results: Vec<Result<RowOut, ExpandErr>> = py.detach(|| {
        facts
            .par_iter()
            .zip(d6.par_iter())
            .map(|(f, &sym)| expand_one(f, sym, support_radius, halo, horizons_len, tolerate_off_legal))
            .collect()
    });

    // Surface the first hard error in row order (deterministic message).
    let mut rows: Vec<RowOut> = Vec::with_capacity(r);
    for res in results {
        match res {
            Ok(row) => rows.push(row),
            Err(ExpandErr::OffLegal) => rows.push(RowOut {
                valid: false,
                legal_count: 0,
                stone_count: 0,
                halo_count: 0,
                coords: Vec::new(),
                dist: Vec::new(),
                nbr: Vec::new(),
                feats: Vec::new(),
                raylen: Vec::new(),
                policy: Vec::new(),
                opp_policy: Vec::new(),
                cell_q: Vec::new(),
                cell_q_mask: Vec::new(),
                gumbel_policy: Vec::new(),
                gumbel_policy_valid: 0.0,
                prior_logit: Vec::new(),
                policy_surprise: 0.0,
                opp_coverage: 1.0,
                value: 0.0,
                value_mask: 0.0,
                stvalue: vec![0.0; horizons_len],
                stvalue_mask: vec![0.0; horizons_len],
                moves_left: 0.0,
                moves_left_mask: 0.0,
            }),
            Err(ExpandErr::Hard(msg)) => return Err(PyValueError::new_err(msg)),
        }
    }

    // --- serial order-preserving concat into the flat output buffers ---------
    let total_nodes: usize = rows.iter().map(|x| x.coords.len() / 2).sum();
    let total_legal: usize = rows.iter().map(|x| x.policy.len()).sum();

    let mut valid = Vec::with_capacity(r);
    let mut legal_count = Vec::with_capacity(r);
    let mut stone_count = Vec::with_capacity(r);
    let mut halo_count = Vec::with_capacity(r);
    let mut node_off = Vec::with_capacity(r + 1);
    let mut pol_off_out = Vec::with_capacity(r + 1);
    let mut coords = Vec::with_capacity(total_nodes * 2);
    let mut dist = Vec::with_capacity(total_nodes);
    let mut nbr = Vec::with_capacity(total_nodes * 6);
    let mut feats = Vec::with_capacity(total_nodes * num_features());
    let mut raylen = Vec::with_capacity(total_nodes * RAYLEN_SLOTS);
    let mut policy = Vec::with_capacity(total_legal);
    let mut opp_policy = Vec::with_capacity(total_legal);
    let mut cell_q = Vec::with_capacity(total_legal);
    let mut cell_q_mask = Vec::with_capacity(total_legal);
    // Dense π' and raw logits follow pol_off; the present flag is per-row.
    let mut gumbel_policy = Vec::with_capacity(total_legal);
    let mut prior_logit_out = Vec::with_capacity(total_legal);
    let mut gumbel_policy_valid_out = Vec::with_capacity(r);
    let mut policy_surprise_out = Vec::with_capacity(r);
    let mut opp_coverage: Vec<f64> = Vec::with_capacity(r);
    let mut value_out = Vec::with_capacity(r);
    let mut value_mask_out = Vec::with_capacity(r);
    let mut moves_left_out = Vec::with_capacity(r);
    let mut moves_left_mask = Vec::with_capacity(r);
    let mut stvalue_out = Vec::with_capacity(r * horizons_len);
    let mut stvalue_mask_out = Vec::with_capacity(r * horizons_len);

    node_off.push(0i64);
    pol_off_out.push(0i64);
    for row in &rows {
        valid.push(if row.valid { 1u8 } else { 0u8 });
        legal_count.push(row.legal_count);
        stone_count.push(row.stone_count);
        halo_count.push(row.halo_count);
        coords.extend_from_slice(&row.coords);
        dist.extend_from_slice(&row.dist);
        nbr.extend_from_slice(&row.nbr);
        feats.extend_from_slice(&row.feats);
        raylen.extend_from_slice(&row.raylen);
        policy.extend_from_slice(&row.policy);
        opp_policy.extend_from_slice(&row.opp_policy);
        cell_q.extend_from_slice(&row.cell_q);
        cell_q_mask.extend_from_slice(&row.cell_q_mask);
        gumbel_policy.extend_from_slice(&row.gumbel_policy);
        prior_logit_out.extend_from_slice(&row.prior_logit);
        gumbel_policy_valid_out.push(row.gumbel_policy_valid);
        policy_surprise_out.push(row.policy_surprise);
        opp_coverage.push(row.opp_coverage);
        value_out.push(row.value);
        value_mask_out.push(row.value_mask);
        moves_left_out.push(row.moves_left);
        moves_left_mask.push(row.moves_left_mask);
        stvalue_out.extend_from_slice(&row.stvalue);
        stvalue_mask_out.extend_from_slice(&row.stvalue_mask);
        node_off.push(node_off.last().unwrap() + (row.coords.len() / 2) as i64);
        pol_off_out.push(pol_off_out.last().unwrap() + row.policy.len() as i64);
    }

    let out = PyDict::new(py);
    out.set_item("valid", Py::new(py, RxU8Buf { data: valid })?)?;
    out.set_item("legal_count", Py::new(py, RxI32Buf { data: legal_count })?)?;
    out.set_item("stone_count", Py::new(py, RxI32Buf { data: stone_count })?)?;
    out.set_item("halo_count", Py::new(py, RxI32Buf { data: halo_count })?)?;
    out.set_item("node_off", Py::new(py, RxI64Buf { data: node_off })?)?;
    out.set_item("pol_off", Py::new(py, RxI64Buf { data: pol_off_out })?)?;
    out.set_item("coords", Py::new(py, RxI32Buf { data: coords })?)?;
    out.set_item("dist", Py::new(py, RxI32Buf { data: dist })?)?;
    out.set_item("nbr", Py::new(py, RxI32Buf { data: nbr })?)?;
    out.set_item("feats", Py::new(py, RxF32Buf { data: feats })?)?;
    // Side-relative ray lengths, node-major u8 (RAYLEN_SLOTS per node), sliced
    // by node_off like coords/dist/nbr/feats (Phase L0 wire data).
    out.set_item("raylen", Py::new(py, RxU8Buf { data: raylen })?)?;
    out.set_item("policy", Py::new(py, RxF32Buf { data: policy })?)?;
    out.set_item("opp_policy", Py::new(py, RxF32Buf { data: opp_policy })?)?;
    out.set_item("opp_coverage", Py::new(py, RxF64Buf { data: opp_coverage })?)?;
    out.set_item("value", Py::new(py, RxF32Buf { data: value_out })?)?;
    out.set_item("value_mask", Py::new(py, RxF32Buf { data: value_mask_out })?)?;
    out.set_item("moves_left", Py::new(py, RxF32Buf { data: moves_left_out })?)?;
    out.set_item("moves_left_mask", Py::new(py, RxF32Buf { data: moves_left_mask })?)?;
    out.set_item("stvalue", Py::new(py, RxF32Buf { data: stvalue_out })?)?;
    out.set_item("stvalue_mask", Py::new(py, RxF32Buf { data: stvalue_mask_out })?)?;
    out.set_item("cell_q", Py::new(py, RxF32Buf { data: cell_q })?)?;
    out.set_item("cell_q_mask", Py::new(py, RxF32Buf { data: cell_q_mask })?)?;
    // Dense π' target (follows pol_off), per-row present flag, dense raw logits.
    out.set_item("gumbel_policy", Py::new(py, RxF32Buf { data: gumbel_policy })?)?;
    out.set_item(
        "gumbel_policy_valid",
        Py::new(py, RxF32Buf { data: gumbel_policy_valid_out })?,
    )?;
    out.set_item("prior_logit", Py::new(py, RxF32Buf { data: prior_logit_out })?)?;
    out.set_item("policy_surprise", Py::new(py, RxF32Buf { data: policy_surprise_out })?)?;
    out.set_item("num_rows", r)?;
    // The active plane-map width; the Python consumer asserts it against its
    // own NUM_FEATURES so a stale .so / feature-version desync fails loudly.
    out.set_item("num_features", num_features())?;
    Ok(out)
}
