//! hexfield constants. Corresponding Python values are in python/hexfield/constants.py.

use std::sync::OnceLock;

pub const LEGAL_RADIUS: i32 = 8;
pub const HALO_DIST: i32 = 9;

/// Fixed direction order D: the rotate60 orbit of (1, 0).
pub const DIRECTIONS: [(i16, i16); 6] = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)];

// Node features (F = num_features(): 25 under HEXFIELD_EQ_FEATURE_VERSION=1,
// 46 under version 2 — SPEC_RAYTAP_CONV.md §1.2). Planes 0-10 are the 11 kept
// scalars; the graded per-(cell, axis) window planes start at 11 (4 quantities
// x 3 axes Q/R/QR under version 1; version 2 appends live3/live4/live5 per
// side for 10 quantities, planes 11-40), each quantity 3 contiguous slots so a
// D6 axis-permutation acts on 3-slot blocks. The 2 scalar fork planes follow
// the axis block (23-24 under version 1, RE-INDEXED to 41-42 under version 2),
// and version 2 appends 3 global scalar planes (43-45). The binary hot/win
// planes of the hexfield lineage are retired (see
// docs/PLAN_D6_EQUIVARIANT_REWRITE.md §3).
pub const NUM_FEATURES_V1: usize = 25;
pub const NUM_FEATURES_V2: usize = 46;
pub const F_OWN_STONE: usize = 0;
pub const F_OPP_STONE: usize = 1;
pub const F_EMPTY: usize = 2;
pub const F_LEGAL: usize = 3;
pub const F_PHASE_SECOND: usize = 4;
pub const F_FIRST_STONE: usize = 5;
pub const F_PLAYER_COLOUR: usize = 6;
pub const F_OWN_RECENCY: usize = 7;
pub const F_OPP_RECENCY: usize = 8;
pub const F_DIST_TO_STONE: usize = 9;
pub const F_OPP_LAST_TURN: usize = 10;
// Graded per-axis window planes. Each quantity spans 3 contiguous slots ordered
// by Axis::ALL == [Q, R, QR], so `BASE + Axis::index()` selects the axis plane.
// Rust code writes the whole block via `F_OWN_LINE_Q + k`; the individual
// plane names below are retained (allow(dead_code)) as the named plane-map
// mirror of python/hexfield_eq/constants.py.
pub const F_OWN_LINE_Q: usize = 11;
#[allow(dead_code)]
pub const F_OWN_LINE_R: usize = 12;
#[allow(dead_code)]
pub const F_OWN_LINE_QR: usize = 13;
#[allow(dead_code)]
pub const F_OPP_LINE_Q: usize = 14;
#[allow(dead_code)]
pub const F_OPP_LINE_R: usize = 15;
#[allow(dead_code)]
pub const F_OPP_LINE_QR: usize = 16;
#[allow(dead_code)]
pub const F_OWN_LIVE_Q: usize = 17;
#[allow(dead_code)]
pub const F_OWN_LIVE_R: usize = 18;
#[allow(dead_code)]
pub const F_OWN_LIVE_QR: usize = 19;
#[allow(dead_code)]
pub const F_OPP_LIVE_Q: usize = 20;
#[allow(dead_code)]
pub const F_OPP_LIVE_R: usize = 21;
#[allow(dead_code)]
pub const F_OPP_LIVE_QR: usize = 22;
// Version-dependent planes. The fork planes keep their definition but move
// under version 2 (the spec §1.2 re-index) — consumers go through
// f_own_fork()/f_opp_fork(). The version-2 liveK planes are 23-40 (contiguous
// continuation of the axis block, plane = 11 + q*3 + axis for q = 4..9); the
// version-2 global scalars follow the forks.
pub const F_OWN_FORK_V1: usize = 23;
pub const F_OPP_FORK_V1: usize = 24;
pub const F_OWN_FORK_V2: usize = 41;
pub const F_OPP_FORK_V2: usize = 42;
pub const F_PLY_V2: usize = 43;
pub const F_DIST_CENTROID_V2: usize = 44;
pub const F_SPREAD_V2: usize = 45;

/// HEXFIELD_EQ_FEATURE_VERSION in {1, 2}, default 1 (byte-identical current
/// behavior). Read once (the support_radius() OnceLock pattern). Unlike the
/// clamping radius read, an invalid value PANICS — the Python reader raises a
/// ValueError for the same value, and a silent version fallback would desync
/// the two featurizers' plane maps.
pub fn feature_version() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        use std::env::VarError;
        match std::env::var("HEXFIELD_EQ_FEATURE_VERSION") {
            Err(VarError::NotPresent) => 1,
            Err(err) => panic!("HEXFIELD_EQ_FEATURE_VERSION unreadable: {err}"),
            Ok(s) if s == "1" => 1,
            Ok(s) if s == "2" => 2,
            Ok(s) => panic!("HEXFIELD_EQ_FEATURE_VERSION={s:?} must be '1' or '2'"),
        }
    })
}

/// The active plane-map width (matches python constants.NUM_FEATURES).
pub fn num_features() -> usize {
    if feature_version() == 2 {
        NUM_FEATURES_V2
    } else {
        NUM_FEATURES_V1
    }
}

/// Width of the contiguous graded axis block at plane 11
/// (3 axes x N_AXIS_QUANTITIES: 12 under version 1, 30 under version 2).
pub fn num_axis_planes() -> usize {
    if feature_version() == 2 {
        30
    } else {
        12
    }
}

pub fn f_own_fork() -> usize {
    if feature_version() == 2 {
        F_OWN_FORK_V2
    } else {
        F_OWN_FORK_V1
    }
}

pub fn f_opp_fork() -> usize {
    if feature_version() == 2 {
        F_OPP_FORK_V2
    } else {
        F_OPP_FORK_V1
    }
}

pub const WINDOW_LEN: usize = 6;

// Side-relative ray lengths (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase L0):
// per cell u8[RAYLEN_SLOTS], flat index side*6 + axis*2 + dir with side in
// {own=0, opp=1}, axis in Axis::ALL order [Q, R, QR], dir in {+=0, -=1}.
// Values 0..=RAY_REACH; the reach is the window-6 geometry made exact (a
// length-6 window through x extends at most 5 cells along the axis), not a knob.
pub const RAYLEN_SLOTS: usize = 12;
pub const RAY_REACH: usize = WINDOW_LEN - 1;

// Graded-feature normalizers (match python/hexfield_eq/constants.py):
//   line count / 5 (a clean window holds at most 5 own stones in a decision
//   state — 6 is a played win), live window count / 6 (6 windows per cell per
//   axis), fork axis count / 3 (3 axes). A raw per-axis line count >=
//   FORK_LINE_THRESHOLD marks that axis as forking.
pub const LINE_NORM: f32 = 5.0;
pub const LIVE_NORM: f32 = 6.0;
pub const FORK_NORM: f32 = 3.0;
pub const FORK_LINE_THRESHOLD: u32 = 3;

// Version-2 global-scalar normalizers (spec §1.4, match
// python/hexfield_eq/constants.py). f64 — the ply/centroid/spread pipeline
// computes in f64 with ONE rounding into the f32 plane, mirroring the Python
// oracle's float arithmetic exactly (the recency-dtype convention of
// features.rs / replay_expand.rs).
pub const PLY_NORM: f64 = 96.0;
pub const SPREAD_NORM: f64 = 16.0;

pub const DIST_SCALE: f32 = LEGAL_RADIUS as f32;
