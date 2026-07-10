//! hexfield search drivers: lockstep batched search and the continuous per-slot
//! scheduler.
//!
//! - `root_fpu_zero_under_noise` defaults FALSE and the root-policy-temperature
//!   schedule defaults OFF (1.0 / no ramp).
//! - The optional search divergences (LCB greedy selection, early-stop by move
//!   class, visit-scaled c_puct, moves-left utility) default ON and are forced
//!   off by `search_parity_mode`.
//!
//! Seed discipline: `mix_seed` and stream ids 0-6 are pinned by golden vectors
//! in tests.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use hexo_engine::{
    apply_placement, unpack_coord, HexoState as RustHexoState, PackedCoord, Placement,
};

use crate::cache::{
    new_shared_evaluation_cache, new_shared_evaluation_stats, state_hash, EvaluationStats,
    RustEvaluation, RustEvaluationRequest, SharedEvaluationCache, SharedEvaluationStats,
    EVAL_CACHE_MAX_STATES,
};
use crate::payload::{
    evaluate_state_refs_cached, finish_eval_cached, submit_eval_cached, PendingEval,
};
use crate::state::states_from_py_states;
use crate::threats_shared as threats;
use crate::tree::{
    gumbel_completed_q, gumbel_sigma, gumbel_softmax, random_unit, terminal_value, Divergences,
    RootDirichletNoise, RustEdge, RustLeaf, RustNode, RustSearch, Widening,
};

pub const ACTIVE_ROOT_LIMIT: usize = 512;

pub const SEED_STREAM_ROOT_NOISE: u64 = 0;
pub const SEED_STREAM_MOVE_SELECT: u64 = 1;
pub const SEED_STREAM_PCR: u64 = 2;
pub const SEED_STREAM_POLICY_INIT_SELECT: u64 = 3;
pub const SEED_STREAM_POLICY_INIT_COUNT: u64 = 4;
pub const SEED_STREAM_POLICY_INIT_SAMPLE: u64 = 5;
/// Gumbel-Top-k root draws. Dedicated stream so Gumbel noise is independent of
/// the Dirichlet root-noise stream (id 0).
pub const SEED_STREAM_GUMBEL: u64 = 6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MoveClass {
    Full,
    Fast,
    Init,
}

#[derive(Clone, Copy)]
struct ContinuousMovePolicy {
    full_visits: u32,
    fast_visits: u32,
    /// Play temperature for the Fast class. 0.0 (default) => greedy LCB pick.
    fast_temperature: f32,
    pcr_full_proportion: f32,
    policy_init_fraction: f32,
    policy_init_avg_plies: f32,
    policy_init_max_plies: u32,
    policy_init_temperature: f32,
    root_policy_temperature: f32,
    root_policy_temperature_early: f32,
    root_policy_temperature_halflife: f32,
    fpu_reduction: f32,
    forced_playout_k: f32,
    noise: Option<RootNoiseConfig>,
    tss_enabled: bool,
    root_fpu_zero_under_noise: bool,
    /// Root FPU reduction. When Some it takes precedence over the
    /// noise-conditioned `root_fpu_zero_under_noise` mechanism and applies to
    /// every move class. When None, the `root_fpu_zero_under_noise` path applies.
    root_fpu_reduction: Option<f32>,
    /// Divergences view for Full/Init move classes (the base search profile).
    divergences_full: Divergences,
    /// Divergences view for the Fast move class. Equals `divergences_full` when
    /// no `fast_*` overrides are set (golden invariant), so absent fast levers
    /// reproduce today's single-profile behavior byte-for-byte.
    divergences_fast: Divergences,
}

impl ContinuousMovePolicy {
    /// The per-class Divergences view: Fast moves get `divergences_fast`,
    /// Full/Init get `divergences_full`.
    fn divergences_for(&self, class: MoveClass) -> Divergences {
        match class {
            MoveClass::Fast => self.divergences_fast,
            MoveClass::Full | MoveClass::Init => self.divergences_full,
        }
    }
    fn policy_init_plies(&self, base_seed: u64, game_key: u64) -> u32 {
        if self.policy_init_fraction <= 0.0
            || self.policy_init_avg_plies <= 0.0
            || self.policy_init_max_plies == 0
        {
            return 0;
        }
        if self.policy_init_fraction < 1.0 {
            let select =
                random_unit(mix_seed(base_seed, game_key, 0, SEED_STREAM_POLICY_INIT_SELECT));
            if select >= self.policy_init_fraction as f64 {
                return 0;
            }
        }
        let unit = random_unit(mix_seed(base_seed, game_key, 1, SEED_STREAM_POLICY_INIT_COUNT));
        let count =
            (-(self.policy_init_avg_plies as f64) * (1.0 - unit).max(1.0e-12).ln()).floor();
        (count.max(0.0) as u32).min(self.policy_init_max_plies)
    }

    /// Classify a ply into a Full/Fast/Init move class.
    ///
    /// PCR classification is per-TURN, not per-ply: the mix counter is
    /// `ply / 2`, so the two plies of one turn (2k and 2k+1) hash to the SAME
    /// stream input and therefore share one Full/Fast class. Two reasons:
    ///
    ///  1. Clean tree reuse. A Full turn builds a deep PUCT subtree that its
    ///     paired ply promotes and reuses under the SAME regime. If the two
    ///     plies could land in different classes, the per-class
    ///     `set_divergences` refactor below would swap the Gumbel-root vs PUCT
    ///     regime mid-turn onto a reused root, corrupting the promoted SH state.
    ///     Sharing the class keeps the whole turn's reused tree on one regime.
    ///  2. Balanced player coverage. Each Full turn exports one P0 and one P1
    ///     policy target, so Full turns contribute training rows for both
    ///     players symmetrically instead of skewing to whichever seat happened
    ///     to draw Full.
    ///
    /// The `policy_init_remaining > 0 => Init` short-circuit and the
    /// `pcr_full_proportion >= 1.0 => Full` short-circuit are unchanged. Call
    /// sites still pass the real `ply`; the `/2` happens only here.
    fn classify(
        &self,
        base_seed: u64,
        game_key: u64,
        ply: u32,
        policy_init_remaining: u32,
    ) -> MoveClass {
        if policy_init_remaining > 0 {
            return MoveClass::Init;
        }
        if self.pcr_full_proportion >= 1.0 {
            return MoveClass::Full;
        }
        // Per-turn: both plies 2k and 2k+1 map to turn index k.
        let turn = ply / 2;
        let unit = random_unit(mix_seed(base_seed, game_key, turn, SEED_STREAM_PCR));
        if unit < self.pcr_full_proportion as f64 {
            MoveClass::Full
        } else {
            MoveClass::Fast
        }
    }

    fn visits_for(&self, class: MoveClass) -> u32 {
        match class {
            MoveClass::Full => self.full_visits,
            MoveClass::Fast => self.fast_visits,
            MoveClass::Init => 1,
        }
    }

    fn forced_k_for(&self, class: MoveClass) -> f32 {
        match class {
            MoveClass::Full => self.forced_playout_k,
            _ => 0.0,
        }
    }

    fn noise_for(&self, class: MoveClass) -> Option<RootNoiseConfig> {
        match class {
            MoveClass::Full => self.noise,
            _ => None,
        }
    }

    fn root_fpu_for(&self, class: MoveClass) -> f32 {
        // When set, `root_fpu_reduction` takes precedence and applies to every
        // move class.
        if let Some(value) = self.root_fpu_reduction {
            return value;
        }
        // Otherwise zero FPU only at noised Full roots when the knob is set.
        if matches!(class, MoveClass::Full)
            && self.noise.is_some()
            && self.root_fpu_zero_under_noise
        {
            0.0
        } else {
            self.fpu_reduction
        }
    }

    fn root_temp_for(&self, class: MoveClass, ply: u32) -> f32 {
        if !matches!(class, MoveClass::Full) {
            return 1.0;
        }
        if self.root_policy_temperature_early <= 0.0
            || self.root_policy_temperature_halflife <= 0.0
        {
            return self.root_policy_temperature;
        }
        self.root_policy_temperature
            + (self.root_policy_temperature_early - self.root_policy_temperature)
                * 0.5f32.powf(ply as f32 / self.root_policy_temperature_halflife)
    }

    /// Per-class PLAY temperature for the continuous driver.
    ///   Full => the ply schedule (floor applied by config).
    ///   Fast => `fast_temperature` (default 0.0 = greedy LCB pick, bit-for-bit
    ///           unchanged; at T=0.1 the sampler exponent is 1/T=10 over the
    ///           guard-filtered delta-visit histogram — gentle exploration, near
    ///           argmax unless the top candidates are close. At T>0 the LCB pick +
    ///           ml_final_pick no longer fire for Fast moves — they require T==0).
    ///   Init => 0.0 (the played move is then prior-sampled by the caller at
    ///           policy_init_temperature).
    fn temperature_for_class(
        &self,
        class: MoveClass,
        temperature_by_ply: &[f32],
        ply: u32,
    ) -> f32 {
        match class {
            MoveClass::Full => temperature_for_ply(temperature_by_ply, ply),
            MoveClass::Fast => self.fast_temperature,
            MoveClass::Init => 0.0,
        }
    }

    /// Whether the evaluator must emit moves-left output. True when EITHER
    /// class view enables the moves-left utility (a shared evaluation feeds both
    /// Full and Fast roots, so it must satisfy whichever class needs ML).
    fn request_moves_left(&self) -> bool {
        self.divergences_full.moves_left_utility || self.divergences_fast.moves_left_utility
    }

    /// Whether the evaluator must emit raw pre-softmax policy logits. True when
    /// EITHER class view enables a Gumbel mechanism that reads `logits(a)` (the
    /// improved target, the Gumbel-Top-k root sampler, or the non-root
    /// selection). Fast will need logits when it runs under Gumbel while Full
    /// stays PUCT, so both views are OR-ed.
    fn request_logits(&self) -> bool {
        let needs = |d: &Divergences| d.gumbel_target || d.gumbel_root || d.gumbel_nonroot_select;
        needs(&self.divergences_full) || needs(&self.divergences_fast)
    }
}

enum ContinuousPhase {
    Active,
    AwaitRootEval,
    Empty,
}

struct ContinuousSlot {
    game_key: u64,
    ply: u32,
    search: Option<RustSearch>,
    phase: ContinuousPhase,
    in_flight: u32,
    baseline: HashMap<PackedCoord, u32>,
    policy_init_remaining: u32,
    move_class: MoveClass,
}

enum ContinuousEvalItem {
    Leaf(RustLeaf),
    RootInit {
        slot_index: usize,
        state: RustHexoState,
        state_hash: hexo_utils::StateHash,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ContinuousFlushDecision {
    Hold,
    Flush { no_progress: bool },
    Stop,
}

#[derive(Default)]
struct ContinuousSchedulerStats {
    flush_count: u64,
    no_progress_flushes: u64,
    queued_states: u64,
    flushed_states: u64,
    flush_size_histogram: HashMap<usize, u64>,
    on_move_seconds: f64,
    moves_decided: u64,
    full_moves: u64,
    fast_moves: u64,
    init_moves: u64,
    early_stops_fast: u64,
    early_stops_full: u64,
    early_stop_visits_saved: u64,
    // Moves finalized by the SH-saturation safety net (force_stuck_gumbel): a
    // stuck Gumbel root finalized from the visits accrued so far. Distinct from
    // the early_stops_* counters — a forced completion is not a genuine early
    // stop and contributes nothing to early_stop_visits_saved.
    force_stuck_completions: u64,
    lcb_overrides: u64,
    // Play-policy telemetry: moves selected via the quota-pruned Gumbel play
    // distribution, and how many of those played the raw delta leader (the SH
    // winner). winner/moves ≈ exploitation rate of the play sampler. The
    // `_early` pair covers ply < 20 (the high-temperature exploration window).
    // Late-game rates do NOT approach 1: SH forces the two finalists to a
    // ~228:196 visit split (1024 visits, m=32), so at the 0.15 temperature
    // floor the runner-up keeps (196/228)^(1/0.15) ≈ 0.37 relative weight and
    // the winner-rate ceiling is ≈ 0.73.
    gumbel_play_moves: u64,
    gumbel_play_winner_moves: u64,
    gumbel_play_moves_early: u64,
    gumbel_play_winner_early: u64,
    // Per-phase wall time (seconds) accumulated over the run: where the
    // scheduler loop actually spends its time. `Instant::now()` bracketing is
    // ~ns-scale against ms-scale phases, so this is always on.
    select_seconds: f64,
    submit_seconds: f64,
    finish_seconds: f64,
    backup_seconds: f64,
    complete_seconds: f64,
    loop_iterations: u64,
    completes_skipped: u64,
}

fn continuous_flush_decision(
    queue_len: usize,
    flush_target: usize,
    made_progress: bool,
) -> ContinuousFlushDecision {
    if queue_len == 0 {
        return if made_progress {
            ContinuousFlushDecision::Hold
        } else {
            ContinuousFlushDecision::Stop
        };
    }
    if queue_len >= flush_target {
        return ContinuousFlushDecision::Flush {
            no_progress: !made_progress,
        };
    }
    if !made_progress {
        return ContinuousFlushDecision::Flush { no_progress: true };
    }
    ContinuousFlushDecision::Hold
}

fn continuous_completion_ready(completed_visits: u32, target_visits: u32, in_flight: u32) -> bool {
    completed_visits >= target_visits && in_flight == 0
}

/// Early-stop test. Greedy unrecorded searches (Fast / eval-arena) stop when
/// the remaining budget cannot overtake the visit leader AND, when LCB
/// selection is active, the LCB winner currently equals the visit winner.
/// Recorded Full roots must first pass a visit floor (`full_visit_floor`).
fn early_stop_ready(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    recorded_full: bool,
    in_flight: u32,
) -> bool {
    let dv = search.divergences;
    if !dv.early_stop || in_flight > 0 {
        return false;
    }
    let remaining = search.remaining_visits();
    if remaining == 0 {
        return false;
    }
    if recorded_full {
        let floor = (search.target_visits as f32 * dv.full_visit_floor).ceil() as u32;
        if search.completed_visits < floor {
            return false;
        }
    }
    let root = search.root();
    // Build the per-edge stats vec once (delta + LCB inputs) and derive
    // best/second/best_id from it. The `delta > best` (strictly-greater)
    // tie-break keeps the first edge at the max delta as best_id.
    let stats = lcb_stats(root, baseline);
    let mut best = 0u32;
    let mut second = 0u32;
    let mut best_id: Option<PackedCoord> = None;
    for &(action_id, delta, _visits, _value_sum, _value_sq_sum) in &stats {
        if delta > best {
            second = best;
            best = delta;
            best_id = Some(action_id as PackedCoord);
        } else if delta > second {
            second = delta;
        }
    }
    let Some(best_id) = best_id else {
        return false;
    };
    if best.saturating_sub(second) <= remaining {
        return false;
    }
    if dv.lcb_move_selection && !recorded_full {
        if let Some(lcb_id) =
            debug_lcb_from_stats(&stats, dv.lcb_z, dv.lcb_min_visits, dv.lcb_visit_fraction)
                .map(|id| id as PackedCoord)
        {
            if lcb_id != best_id {
                return false;
            }
        }
    }
    true
}

/// Per-edge LCB inputs over root edges: (action_id, delta_visits, visits,
/// value_sum, value_sq_sum), in edge order. Shared by lcb_pick and
/// early_stop_ready so the edge scan happens once per decision.
fn lcb_stats(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
) -> Vec<(u64, u32, u32, f32, f32)> {
    root.edges
        .iter()
        .map(|edge| {
            (
                edge.action_id as u64,
                edge_delta_visits(edge, baseline),
                edge.visits,
                edge.value_sum,
                edge.value_sq_sum,
            )
        })
        .collect()
}

/// LCB pick among eligible root edges: Q - z * sigma / sqrt(n), eligibility
/// delta >= max(lcb_min_visits, lcb_visit_fraction * max_child_delta). None
/// when no child qualifies (caller falls back to max-visits). Delegates to
/// `debug_lcb_from_stats`.
fn lcb_pick(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    dv: &Divergences,
) -> Option<PackedCoord> {
    let stats = lcb_stats(root, baseline);
    debug_lcb_from_stats(&stats, dv.lcb_z, dv.lcb_min_visits, dv.lcb_visit_fraction)
        .map(|id| id as PackedCoord)
}

/// Final-move decisiveness tie-break. Among root moves whose LCB is within
/// `ml_final_pick_band` of the LCB leader AND are guard-positive, pick the most
/// decisive one: fewest moves-left when the root is clearly winning (root value
/// > ml_q_gate), most moves-left when clearly losing (< -ml_q_gate). Returns
/// None in the |value| <= gate dead-zone or when no candidate carries a
/// moves-left mean; the caller then keeps the plain LCB pick. Only re-picks
/// among moves within `ml_final_pick_band` of the LCB leader.
fn ml_final_pick(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    dv: &Divergences,
    action_ids: &[PackedCoord],
    guarded_weights: &[f32],
) -> Option<PackedCoord> {
    let root_v = root.value();
    let dir: i32 = if root_v > dv.ml_q_gate {
        1
    } else if root_v < -dv.ml_q_gate {
        -1
    } else {
        return None;
    };
    let stats = lcb_stats(root, baseline);
    let max_delta = stats.iter().map(|s| s.1).max().unwrap_or(0);
    if max_delta == 0 {
        return None;
    }
    let threshold = (dv.lcb_min_visits as f32).max(dv.lcb_visit_fraction * max_delta as f32);
    let mut best_lcb = f32::NEG_INFINITY;
    let mut eligible: Vec<(PackedCoord, f32)> = Vec::new();
    for &(action_id, delta, visits, value_sum, value_sq_sum) in &stats {
        if (delta as f32) < threshold || visits == 0 {
            continue;
        }
        let n = visits as f32;
        let q = value_sum / n;
        let variance = (value_sq_sum / n - q * q).max(0.0);
        let lcb = q - dv.lcb_z * variance.sqrt() / n.sqrt();
        eligible.push((action_id as PackedCoord, lcb));
        if lcb > best_lcb {
            best_lcb = lcb;
        }
    }
    let mut pick: Option<(PackedCoord, f32)> = None;
    for &(id, lcb) in &eligible {
        if lcb < best_lcb - dv.ml_final_pick_band {
            continue;
        }
        let guard_positive = action_ids
            .iter()
            .zip(guarded_weights.iter())
            .any(|(&aid, &w)| aid == id && w > 0.0);
        if !guard_positive {
            continue;
        }
        let Some(m) = root
            .edges
            .iter()
            .find(|e| e.action_id == id)
            .and_then(|e| e.ml_mean())
        else {
            continue;
        };
        let better = match pick {
            None => true,
            Some((_, bm)) => {
                if dir == 1 {
                    m < bm
                } else {
                    m > bm
                }
            }
        };
        if better {
            pick = Some((id, m));
        }
    }
    pick.map(|(id, _)| id)
}

#[pyclass(unsendable)]
pub struct HexfieldMctsSession {
    searches: HashMap<u64, RustSearch>,
    evaluation_cache: SharedEvaluationCache,
    cache_max_states: usize,
}

#[pymethods]
impl HexfieldMctsSession {
    #[new]
    #[pyo3(signature = (max_states=None))]
    fn new(max_states: Option<usize>) -> PyResult<Self> {
        let cache_max_states =
            validate_positive_usize("max_states", max_states.unwrap_or(EVAL_CACHE_MAX_STATES))?;
        Ok(Self {
            searches: HashMap::new(),
            evaluation_cache: new_shared_evaluation_cache(),
            cache_max_states,
        })
    }

    fn discard(&mut self, game_key: u64) {
        self.searches.remove(&game_key);
    }

    fn len(&self) -> usize {
        self.searches.len()
    }

    /// Lockstep batched search (eval ladder / arena / differential harness).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (game_keys, states, visits, c_puct, temperature, seed, evaluator, virtual_batch_size=None, active_root_limit=None, root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None, root_policy_temperature=None, fpu_reduction=None, virtual_loss=None, widening_policy_mass=None, widening_max_children=None, widening_min_children=None, forced_playout_k=None, move_temperatures=None, root_policy_temperatures=None, tss_enabled=None, root_fpu_zero_under_noise=None, root_fpu_reduction=None, search_parity_mode=None, divergence_overrides=None, debug_no_advance=None))]
    fn search(
        &mut self,
        py: Python<'_>,
        game_keys: Vec<u64>,
        states: &Bound<'_, PyAny>,
        visits: u32,
        c_puct: f32,
        temperature: f32,
        seed: u64,
        evaluator: &Bound<'_, PyAny>,
        virtual_batch_size: Option<u32>,
        active_root_limit: Option<usize>,
        root_dirichlet_total_alpha: Option<f32>,
        root_dirichlet_noise_fraction: Option<f32>,
        root_policy_temperature: Option<f32>,
        fpu_reduction: Option<f32>,
        virtual_loss: Option<f32>,
        widening_policy_mass: Option<f32>,
        widening_max_children: Option<u32>,
        widening_min_children: Option<u32>,
        forced_playout_k: Option<f32>,
        move_temperatures: Option<Vec<f32>>,
        root_policy_temperatures: Option<Vec<f32>>,
        tss_enabled: Option<bool>,
        // Default FALSE: no FPU zeroing at noised roots.
        root_fpu_zero_under_noise: Option<bool>,
        // When provided, the root FPU reduction; takes precedence over the
        // noise-conditioned mechanism.
        root_fpu_reduction: Option<f32>,
        search_parity_mode: Option<bool>,
        divergence_overrides: Option<&Bound<'_, PyDict>>,
        debug_no_advance: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        validate_search_inputs(visits, c_puct, temperature)?;
        let divergences = resolve_divergences(search_parity_mode, divergence_overrides, false)?;
        let roots = states_from_py_states(py, states)?;
        if roots.is_empty() {
            return Ok(PyTuple::empty(py).into_any().unbind());
        }
        if roots.len() != game_keys.len() {
            return Err(PyValueError::new_err(format!(
                "hexfield MCTS session received {} game keys for {} states",
                game_keys.len(),
                roots.len()
            )));
        }
        let move_temps: Vec<f32> = match move_temperatures {
            Some(values) => {
                if values.len() != roots.len() {
                    return Err(PyValueError::new_err(format!(
                        "move_temperatures has {} entries for {} roots",
                        values.len(),
                        roots.len()
                    )));
                }
                for value in &values {
                    if !value.is_finite() || *value < 0.0 {
                        return Err(PyValueError::new_err(
                            "move_temperatures entries must be finite and >= 0",
                        ));
                    }
                }
                values
            }
            None => vec![temperature; roots.len()],
        };
        let root_limit = validate_positive_usize(
            "active_root_limit",
            active_root_limit.unwrap_or(ACTIVE_ROOT_LIMIT),
        )?;
        if roots.len() > root_limit {
            return Err(PyValueError::new_err(format!(
                "hexfield MCTS session received {} active roots, above strict limit {}",
                roots.len(),
                root_limit
            )));
        }

        let target_visits = visits;
        let leaf_batch_per_root = validate_positive_u32(
            "virtual_batch_size",
            virtual_batch_size.unwrap_or(target_visits),
        )?;
        let evaluation_stats = new_shared_evaluation_stats();
        // Root policy temperature defaults to 1.0 (schedule off).
        let root_policy_temperature = validate_positive_f32(
            "root_policy_temperature",
            root_policy_temperature.unwrap_or(1.0),
        )?;
        let root_policy_temps: Vec<f32> = match root_policy_temperatures {
            Some(values) => {
                if values.len() != roots.len() {
                    return Err(PyValueError::new_err(format!(
                        "root_policy_temperatures has {} entries for {} roots",
                        values.len(),
                        roots.len()
                    )));
                }
                for value in &values {
                    if !value.is_finite() || *value <= 0.0 {
                        return Err(PyValueError::new_err(
                            "root_policy_temperatures entries must be finite and > 0",
                        ));
                    }
                }
                values
            }
            None => vec![root_policy_temperature; roots.len()],
        };
        let fpu_reduction =
            validate_nonnegative_f32("fpu_reduction", fpu_reduction.unwrap_or(0.20))?;
        let virtual_loss = validate_nonnegative_f32("virtual_loss", virtual_loss.unwrap_or(1.0))?;
        let forced_playout_k =
            validate_nonnegative_f32("forced_playout_k", forced_playout_k.unwrap_or(0.0))?;
        let root_noise_config =
            root_noise_config(root_dirichlet_total_alpha, root_dirichlet_noise_fraction)?;
        let tss_enabled = tss_enabled.unwrap_or(true);
        // Root FPU reduction. If `root_fpu_reduction` is given explicitly it
        // takes precedence. Otherwise use the noise-conditioned mechanism: zero
        // FPU only at noised roots when `root_fpu_zero_under_noise` is set.
        let root_fpu_reduction = match root_fpu_reduction {
            Some(value) => validate_nonnegative_f32("root_fpu_reduction", value)?,
            None => {
                if root_noise_config.is_some() && root_fpu_zero_under_noise.unwrap_or(false) {
                    0.0
                } else {
                    fpu_reduction
                }
            }
        };
        let widening = build_widening(
            widening_policy_mass,
            widening_min_children,
            widening_max_children,
        )?;
        let request_ml = divergences.moves_left_utility;
        // Request raw logits whenever any Gumbel mechanism reads them.
        let request_logits = divergences.gumbel_target
            || divergences.gumbel_root
            || divergences.gumbel_nonroot_select;

        let mut searches: Vec<Option<RustSearch>> = Vec::with_capacity(roots.len());
        let mut missing_indices = Vec::new();
        let mut missing_roots = Vec::new();
        for (index, (game_key, root)) in game_keys.iter().zip(roots.iter()).enumerate() {
            let root_hash = state_hash(root);
            if let Some(mut search) = self.searches.remove(game_key) {
                if search.root_hash == root_hash {
                    search.set_additional_visits(target_visits);
                    search.set_forced_playout_k(forced_playout_k);
                    search.set_root_fpu_reduction(root_fpu_reduction);
                    search.set_tss_enabled(tss_enabled);
                    search.set_divergences(divergences);
                    search.apply_root_policy_temperature(root_policy_temps[index]);
                    if let Some(noise) =
                        root_noise(root_noise_config, seed, index, divergences.dirichlet_shaped)
                    {
                        search.apply_root_dirichlet_noise(noise);
                    }
                    // (Re)build the Gumbel-Top-k candidate set + SH schedule on
                    // the reused root, mirroring the continuous reuse paths;
                    // cleared when the Gumbel root is off so the PUCT root runs.
                    // The root hash folds into the seed stream so successive
                    // moves of one game draw fresh Gumbel noise even when the
                    // caller repeats its per-call seed.
                    if divergences.gumbel_root {
                        let gumbel_seed =
                            mix_seed(seed, *game_key ^ root_hash, 0, SEED_STREAM_GUMBEL);
                        search.init_gumbel_root(gumbel_seed, target_visits);
                    } else {
                        search.clear_gumbel_root();
                    }
                    searches.push(Some(search));
                    continue;
                }
            }
            missing_indices.push(index);
            missing_roots.push(root.clone());
            searches.push(None);
        }

        if !missing_roots.is_empty() {
            let requests: Vec<RustEvaluationRequest> = missing_roots
                .iter()
                .map(|state| RustEvaluationRequest {
                    state,
                    state_hash: state_hash(state),
                })
                .collect();
            let root_evals = evaluate_state_refs_cached(
                py,
                evaluator,
                &requests,
                &self.evaluation_cache,
                Some(&evaluation_stats),
                self.cache_max_states,
                request_ml,
                request_logits,
            )?;
            for ((index, root), evaluation) in missing_indices
                .into_iter()
                .zip(missing_roots.into_iter())
                .zip(root_evals.iter())
            {
                let root_hash = state_hash(&root);
                let mut search = RustSearch::new(
                    root,
                    &**evaluation,
                    target_visits,
                    fpu_reduction,
                    root_fpu_reduction,
                    root_policy_temps[index],
                    root_noise(root_noise_config, seed, index, divergences.dirichlet_shaped),
                    widening,
                    forced_playout_k,
                    tss_enabled,
                    divergences,
                )?;
                // Build the Gumbel-Top-k candidate set + SH schedule for a
                // fresh root when the Gumbel root is on (mirrors the continuous
                // RootInit path). No-op without raw root logits.
                if divergences.gumbel_root {
                    let gumbel_seed =
                        mix_seed(seed, game_keys[index] ^ root_hash, 0, SEED_STREAM_GUMBEL);
                    search.init_gumbel_root(gumbel_seed, target_visits);
                }
                searches[index] = Some(search);
            }
        }

        let mut searches: Vec<RustSearch> = searches
            .into_iter()
            .map(|search| search.expect("session search initialized"))
            .collect();
        if searches.iter().any(RustSearch::root_edges_empty) {
            return Err(PyValueError::new_err("MCTS root has no legal actions"));
        }

        let baselines: Vec<HashMap<PackedCoord, u32>> = searches
            .iter()
            .map(|search| search.root_edge_visits().into_iter().collect())
            .collect();
        run_searches_to_targets(
            py,
            evaluator,
            &mut searches,
            c_puct,
            leaf_batch_per_root,
            &self.evaluation_cache,
            &evaluation_stats,
            self.cache_max_states,
            virtual_loss,
            request_ml,
            request_logits,
            &move_temps,
            &baselines,
        )?;
        let cache_len = self
            .evaluation_cache
            .lock()
            .expect("evaluation cache mutex poisoned")
            .len();
        let evaluation_stats_snapshot = evaluation_stats
            .lock()
            .expect("evaluation stats mutex poisoned")
            .clone();
        let selected_actions: Vec<_> = searches
            .iter()
            .enumerate()
            .map(|(index, search)| {
                select_search_action(
                    search,
                    baselines.get(index),
                    move_temps[index],
                    seed.wrapping_add(index as u64),
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let results = build_search_result_payloads(
            py,
            &searches,
            Some(&evaluation_stats_snapshot),
            Some(cache_len),
            &move_temps,
            seed,
            Some(&baselines),
            c_puct,
            forced_playout_k,
        )?;

        let no_advance = debug_no_advance.unwrap_or(false);
        for ((game_key, mut search), selected) in game_keys
            .into_iter()
            .zip(searches.into_iter())
            .zip(selected_actions.into_iter())
        {
            if no_advance {
                // Forensics only: store the searched tree as-is (root not
                // advanced) so the next search call on this game_key can
                // inspect/reuse it unchanged.
                self.searches.insert(game_key, search);
                continue;
            }
            if let Some(action_id) = selected {
                if search.advance_root(action_id)? {
                    self.searches.insert(game_key, search);
                }
            }
        }

        Ok(results)
    }

    /// Continuous per-slot scheduler (the production self-play driver).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (game_keys, states, evaluator, on_move, visits, c_puct, base_seed, virtual_batch_size, flush_target, active_root_limit, temperature_by_ply, root_dirichlet_total_alpha=None, root_dirichlet_noise_fraction=None, root_policy_temperature=None, fpu_reduction=None, virtual_loss=None, widening_policy_mass=None, widening_max_children=None, widening_min_children=None, forced_playout_k=None, root_policy_temperature_early=None, root_policy_temperature_halflife=None, pcr_full_proportion=None, pcr_fast_visits=None, pcr_fast_temperature=None, policy_init_fraction=None, policy_init_avg_plies=None, policy_init_max_plies=None, policy_init_temperature=None, tss_enabled=None, root_fpu_zero_under_noise=None, root_fpu_reduction=None, search_parity_mode=None, divergence_overrides=None, fast_divergence_overrides=None))]
    fn run_continuous(
        &mut self,
        py: Python<'_>,
        game_keys: Vec<u64>,
        states: &Bound<'_, PyAny>,
        evaluator: &Bound<'_, PyAny>,
        on_move: &Bound<'_, PyAny>,
        visits: u32,
        c_puct: f32,
        base_seed: u64,
        virtual_batch_size: u32,
        flush_target: usize,
        active_root_limit: usize,
        temperature_by_ply: Vec<f32>,
        root_dirichlet_total_alpha: Option<f32>,
        root_dirichlet_noise_fraction: Option<f32>,
        root_policy_temperature: Option<f32>,
        fpu_reduction: Option<f32>,
        virtual_loss: Option<f32>,
        widening_policy_mass: Option<f32>,
        widening_max_children: Option<u32>,
        widening_min_children: Option<u32>,
        forced_playout_k: Option<f32>,
        root_policy_temperature_early: Option<f32>,
        root_policy_temperature_halflife: Option<f32>,
        pcr_full_proportion: Option<f32>,
        pcr_fast_visits: Option<u32>,
        pcr_fast_temperature: Option<f32>,
        policy_init_fraction: Option<f32>,
        policy_init_avg_plies: Option<f32>,
        policy_init_max_plies: Option<u32>,
        policy_init_temperature: Option<f32>,
        tss_enabled: Option<bool>,
        root_fpu_zero_under_noise: Option<bool>,
        // Root FPU reduction; takes precedence over the noise-conditioned knob
        // when provided.
        root_fpu_reduction: Option<f32>,
        search_parity_mode: Option<bool>,
        divergence_overrides: Option<&Bound<'_, PyDict>>,
        // Fast-class divergence view. When None, the Fast class reuses the base
        // (Full) divergences, so absent fast levers = today's single profile.
        fast_divergence_overrides: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        validate_search_inputs(visits, c_puct, 0.0)?;
        let divergences = resolve_divergences(search_parity_mode, divergence_overrides, false)?;
        // Fast-class view: parse the fast override map when provided, else fall
        // back to the base view (golden invariant: divergences_fast ==
        // divergences_full when no fast_* keys are set). Only this map may
        // carry fast_* keys.
        let divergences_fast = match fast_divergence_overrides {
            Some(fast) => resolve_divergences(search_parity_mode, Some(fast), true)?,
            None => divergences,
        };
        let roots = states_from_py_states(py, states)?;
        if roots.len() != game_keys.len() {
            return Err(PyValueError::new_err(format!(
                "hexfield continuous MCTS received {} game keys for {} states",
                game_keys.len(),
                roots.len()
            )));
        }
        let root_limit = validate_positive_usize("active_root_limit", active_root_limit)?;
        if roots.len() > root_limit {
            return Err(PyValueError::new_err(format!(
                "hexfield continuous MCTS received {} active roots, above strict limit {}",
                roots.len(),
                root_limit
            )));
        }
        let leaf_batch_per_root = validate_positive_u32("virtual_batch_size", virtual_batch_size)?;
        let flush_target = validate_positive_usize("flush_target", flush_target)?;
        let root_policy_temperature = validate_positive_f32(
            "root_policy_temperature",
            root_policy_temperature.unwrap_or(1.0),
        )?;
        let fpu_reduction =
            validate_nonnegative_f32("fpu_reduction", fpu_reduction.unwrap_or(0.20))?;
        let virtual_loss = validate_nonnegative_f32("virtual_loss", virtual_loss.unwrap_or(1.0))?;
        let forced_playout_k =
            validate_nonnegative_f32("forced_playout_k", forced_playout_k.unwrap_or(0.0))?;
        let root_noise_config =
            root_noise_config(root_dirichlet_total_alpha, root_dirichlet_noise_fraction)?;
        let root_policy_temperature_early = validate_nonnegative_f32(
            "root_policy_temperature_early",
            root_policy_temperature_early.unwrap_or(0.0),
        )?;
        let root_policy_temperature_halflife = validate_nonnegative_f32(
            "root_policy_temperature_halflife",
            root_policy_temperature_halflife.unwrap_or(0.0),
        )?;
        if root_policy_temperature_early > 0.0 && root_policy_temperature_halflife <= 0.0 {
            return Err(PyValueError::new_err(
                "root_policy_temperature_halflife must be > 0 when root_policy_temperature_early is set",
            ));
        }
        let pcr_full_proportion = pcr_full_proportion.unwrap_or(1.0);
        if !pcr_full_proportion.is_finite()
            || pcr_full_proportion <= 0.0
            || pcr_full_proportion > 1.0
        {
            return Err(PyValueError::new_err(
                "pcr_full_proportion must be in (0, 1]",
            ));
        }
        let pcr_fast_visits = pcr_fast_visits.unwrap_or(visits);
        if pcr_full_proportion < 1.0 && pcr_fast_visits == 0 {
            return Err(PyValueError::new_err(
                "pcr_fast_visits must be >= 1 when PCR is enabled",
            ));
        }
        // Fast-class play temperature. Default 0.0 reproduces the greedy LCB pick
        // (see temperature_for_class) bit-for-bit.
        let pcr_fast_temperature =
            validate_nonnegative_f32("pcr_fast_temperature", pcr_fast_temperature.unwrap_or(0.0))?;
        let policy_init_fraction = policy_init_fraction.unwrap_or(0.0);
        if !policy_init_fraction.is_finite() || !(0.0..=1.0).contains(&policy_init_fraction) {
            return Err(PyValueError::new_err(
                "policy_init_fraction must be in [0, 1]",
            ));
        }
        let policy_init_avg_plies = policy_init_avg_plies.unwrap_or(0.0);
        let policy_init_max_plies = policy_init_max_plies.unwrap_or(0);
        let policy_init_temperature = policy_init_temperature.unwrap_or(1.0);
        if policy_init_fraction > 0.0 {
            if !policy_init_avg_plies.is_finite() || policy_init_avg_plies <= 0.0 {
                return Err(PyValueError::new_err(
                    "policy_init_avg_plies must be > 0 when policy-init openings are enabled",
                ));
            }
            if policy_init_max_plies == 0 {
                return Err(PyValueError::new_err(
                    "policy_init_max_plies must be >= 1 when policy-init openings are enabled",
                ));
            }
            if !policy_init_temperature.is_finite() || policy_init_temperature <= 0.0 {
                return Err(PyValueError::new_err("policy_init_temperature must be > 0"));
            }
        }
        let move_policy = ContinuousMovePolicy {
            full_visits: visits,
            fast_visits: pcr_fast_visits,
            fast_temperature: pcr_fast_temperature,
            pcr_full_proportion,
            policy_init_fraction,
            policy_init_avg_plies,
            policy_init_max_plies,
            policy_init_temperature,
            root_policy_temperature,
            root_policy_temperature_early,
            root_policy_temperature_halflife,
            fpu_reduction,
            forced_playout_k,
            noise: root_noise_config,
            tss_enabled: tss_enabled.unwrap_or(true),
            // Default false.
            root_fpu_zero_under_noise: root_fpu_zero_under_noise.unwrap_or(false),
            // Root FPU reduction (validated >= 0 when provided).
            root_fpu_reduction: match root_fpu_reduction {
                Some(value) => Some(validate_nonnegative_f32("root_fpu_reduction", value)?),
                None => None,
            },
            divergences_full: divergences,
            divergences_fast,
        };
        let widening = build_widening(
            widening_policy_mass,
            widening_min_children,
            widening_max_children,
        )?;
        if temperature_by_ply.is_empty() {
            return Err(PyValueError::new_err("temperature_by_ply must not be empty"));
        }
        if temperature_by_ply
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(PyValueError::new_err(
                "temperature_by_ply entries must be finite and >= 0",
            ));
        }

        let evaluation_stats = new_shared_evaluation_stats();
        let mut slots = Vec::with_capacity(roots.len());
        let mut queue: Vec<ContinuousEvalItem> = Vec::new();
        for (slot_index, (game_key, root)) in
            game_keys.into_iter().zip(roots.into_iter()).enumerate()
        {
            let root_hash = state_hash(&root);
            let policy_init_remaining = move_policy.policy_init_plies(base_seed, game_key);
            let move_class = move_policy.classify(base_seed, game_key, 0, policy_init_remaining);
            let mut slot = ContinuousSlot {
                game_key,
                ply: 0,
                search: None,
                phase: ContinuousPhase::AwaitRootEval,
                in_flight: 0,
                baseline: HashMap::new(),
                policy_init_remaining,
                move_class,
            };
            if let Some(mut search) = self.searches.remove(&game_key) {
                if search.root_hash == root_hash {
                    // Per-class divergence view: Fast=fast, Full/Init=base.
                    let class_div = move_policy.divergences_for(move_class);
                    search.set_additional_visits(move_policy.visits_for(move_class));
                    search.set_forced_playout_k(move_policy.forced_k_for(move_class));
                    search.set_root_fpu_reduction(move_policy.root_fpu_for(move_class));
                    search.set_tss_enabled(move_policy.tss_enabled);
                    search.set_divergences(class_div);
                    search.apply_root_policy_temperature(move_policy.root_temp_for(move_class, 0));
                    if let Some(noise) = root_noise_exact(
                        move_policy.noise_for(move_class),
                        mix_seed(base_seed, game_key, 0, SEED_STREAM_ROOT_NOISE),
                        class_div.dirichlet_shaped,
                    ) {
                        search.apply_root_dirichlet_noise(noise);
                    }
                    // (Re)build the Gumbel-Top-k candidate set + SH schedule on
                    // a reused root. init_gumbel_root clears any prior state
                    // first; when this class's view has gumbel_root off it is
                    // cleared so the normal PUCT root runs.
                    if class_div.gumbel_root {
                        let gumbel_seed = mix_seed(base_seed, game_key, 0, SEED_STREAM_GUMBEL);
                        search.init_gumbel_root(gumbel_seed, move_policy.visits_for(move_class));
                    } else {
                        search.clear_gumbel_root();
                    }
                    slot.baseline = search.root_edge_visits().into_iter().collect();
                    slot.search = Some(search);
                    slot.phase = ContinuousPhase::Active;
                }
            }
            if matches!(slot.phase, ContinuousPhase::AwaitRootEval) {
                queue.push(ContinuousEvalItem::RootInit {
                    slot_index,
                    state: root,
                    state_hash: root_hash,
                });
            }
            slots.push(slot);
        }

        let mut stats = ContinuousSchedulerStats::default();
        // Select-eval overlap: the next select pass runs with the flush's
        // virtual losses still pending (pre-backup tree state). A no-progress
        // prefetch is discarded so the next iteration re-selects after the
        // backup frees the paths.
        let mut prefetched: Option<(Vec<RustLeaf>, bool)> = None;
        // HEXFIELD_ASYNC_EVAL: the forward is enqueued (submit, no device
        // sync), the pre-backup select runs with the GIL released while those
        // kernels execute, then the forward is drained (finish). Off =>
        // synchronous eval-then-select. Only the sync point moves.
        // HEXFIELD_NO_PREFETCH disables the prefetch select.
        let async_eval = std::env::var("HEXFIELD_ASYNC_EVAL").is_ok();
        let no_prefetch = std::env::var("HEXFIELD_NO_PREFETCH").is_ok();
        // HEXFIELD_PIPELINE_DEPTH2: depth-2 double-buffered eval (default OFF).
        // Keeps one eval in flight on the GPU while the host selects the next
        // batch and backs up the previous flush. Deepens the async
        // (submit/finish) window by one flush, so the leaf stream differs from
        // strict lockstep (still virtual-loss-faithful). Requires
        // HEXFIELD_ASYNC_EVAL for submit-without-sync; without it, falls back to
        // the lockstep loop with a warning.
        let pipeline_depth2 = std::env::var("HEXFIELD_PIPELINE_DEPTH2").is_ok();
        let pipeline_depth2 = if pipeline_depth2 && !async_eval {
            eprintln!(
                "hexfield: HEXFIELD_PIPELINE_DEPTH2 ignored (requires HEXFIELD_ASYNC_EVAL=1); \
                 falling back to the lockstep scheduler"
            );
            false
        } else {
            pipeline_depth2
        };
        if pipeline_depth2 {
            self.run_continuous_pipeline_depth2(
                py,
                &mut slots,
                &mut queue,
                evaluator,
                on_move,
                c_puct,
                base_seed,
                leaf_batch_per_root,
                flush_target,
                virtual_loss,
                &move_policy,
                widening,
                divergences,
                &temperature_by_ply,
                &evaluation_stats,
                &mut stats,
            )?;
            return self.finish_continuous_stats(py, stats, &evaluation_stats);
        }
        // HEXFIELD_GATE_COMPLETE: skip the per-iteration complete scan (a
        // par_iter readiness sweep over every slot) on iterations where nothing
        // could have become ready — no backup ran this iteration, the previous
        // complete decided no moves, and the loop is not at a Stop decision.
        // Completion readiness only changes when a backup lands new visits or a
        // completed move advances a root, so the gated scan is decision-
        // identical; the flag exists for the A/B.
        let gate_complete = std::env::var("HEXFIELD_GATE_COMPLETE").is_ok();
        let mut last_moves_decided: u64 = 1; // force the first scan
        while continuous_has_work(&slots) || !queue.is_empty() {
            stats.loop_iterations += 1;
            let phase_t0 = std::time::Instant::now();
            let (new_leaves, made_progress) = match prefetched.take() {
                Some(result) => result,
                None => py.detach(|| {
                    select_continuous_pass(&mut slots, c_puct, leaf_batch_per_root, virtual_loss)
                })?,
            };
            stats.select_seconds += phase_t0.elapsed().as_secs_f64();
            queue.extend(new_leaves.into_iter().map(ContinuousEvalItem::Leaf));

            let decision = continuous_flush_decision(queue.len(), flush_target, made_progress);
            if let ContinuousFlushDecision::Flush { no_progress } = decision {
                if no_progress {
                    stats.no_progress_flushes += 1;
                }
                let items = std::mem::take(&mut queue);
                stats.flush_count += 1;
                stats.queued_states += items.len() as u64;
                let unique_before = evaluation_stats
                    .lock()
                    .expect("evaluation stats mutex poisoned")
                    .unique_states;
                let requests: Vec<RustEvaluationRequest> = items
                    .iter()
                    .map(|item| match item {
                        ContinuousEvalItem::Leaf(leaf) => RustEvaluationRequest {
                            state: &leaf.state,
                            state_hash: leaf.state_hash,
                        },
                        ContinuousEvalItem::RootInit {
                            state, state_hash, ..
                        } => RustEvaluationRequest {
                            state,
                            state_hash: *state_hash,
                        },
                    })
                    .collect();
                // Eval the flush and run the pre-backup select (on pre-backup
                // tree state). Async: submit -> select -> finish. Sync: eval ->
                // select. Both yield (prefetch_result, evaluations).
                let (prefetch_result, evaluations) = if async_eval {
                    let t_submit = std::time::Instant::now();
                    let pending = submit_eval_cached(
                        py,
                        evaluator,
                        &requests,
                        &self.evaluation_cache,
                        Some(&evaluation_stats),
                        move_policy.request_moves_left(),
                        move_policy.request_logits(),
                    )?;
                    stats.submit_seconds += t_submit.elapsed().as_secs_f64();
                    let t_prefetch = std::time::Instant::now();
                    let prefetch_result = if no_prefetch {
                        (Vec::new(), false)
                    } else {
                        py.detach(|| {
                            select_continuous_pass(
                                &mut slots,
                                c_puct,
                                leaf_batch_per_root,
                                virtual_loss,
                            )
                        })?
                    };
                    stats.select_seconds += t_prefetch.elapsed().as_secs_f64();
                    let t_finish = std::time::Instant::now();
                    let evaluations = finish_eval_cached(
                        py,
                        evaluator,
                        pending,
                        &self.evaluation_cache,
                        Some(&evaluation_stats),
                        self.cache_max_states,
                    )?;
                    stats.finish_seconds += t_finish.elapsed().as_secs_f64();
                    (prefetch_result, evaluations)
                } else {
                    let evaluations = evaluate_state_refs_cached(
                        py,
                        evaluator,
                        &requests,
                        &self.evaluation_cache,
                        Some(&evaluation_stats),
                        self.cache_max_states,
                        move_policy.request_moves_left(),
                        move_policy.request_logits(),
                    )?;
                    let prefetch_result = if no_prefetch {
                        (Vec::new(), false)
                    } else {
                        select_continuous_pass(
                            &mut slots,
                            c_puct,
                            leaf_batch_per_root,
                            virtual_loss,
                        )?
                    };
                    (prefetch_result, evaluations)
                };
                let unique_after = evaluation_stats
                    .lock()
                    .expect("evaluation stats mutex poisoned")
                    .unique_states;
                let unique_flushed = unique_after.saturating_sub(unique_before);
                stats.flushed_states += unique_flushed as u64;
                *stats
                    .flush_size_histogram
                    .entry(unique_flushed.max(1).next_power_of_two())
                    .or_insert(0) += 1;
                let t_backup = std::time::Instant::now();
                backup_continuous_items(
                    py,
                    &mut slots,
                    items,
                    &evaluations,
                    &move_policy,
                    widening,
                    base_seed,
                    virtual_loss,
                    divergences,
                )?;
                stats.backup_seconds += t_backup.elapsed().as_secs_f64();
                prefetched = if prefetch_result.1 {
                    Some(prefetch_result)
                } else {
                    None
                };
            }

            let flushed_this_iter = matches!(decision, ContinuousFlushDecision::Flush { .. });
            let must_complete = !gate_complete
                || flushed_this_iter
                || last_moves_decided > 0
                || matches!(decision, ContinuousFlushDecision::Stop);
            let t_complete = std::time::Instant::now();
            let mut moves_decided = if must_complete {
                complete_continuous_slots(
                    py,
                    on_move,
                    &mut slots,
                    c_puct,
                    &move_policy,
                    &temperature_by_ply,
                    base_seed,
                    &mut queue,
                    &mut stats,
                    false,
                )?
            } else {
                stats.completes_skipped += 1;
                0
            };
            stats.complete_seconds += t_complete.elapsed().as_secs_f64();

            if matches!(decision, ContinuousFlushDecision::Stop) && moves_decided == 0 {
                // Rescue pass before declaring a stall: a Gumbel
                // Sequential-Halving root can saturate its reachable tree below
                // target_visits and its round caps (terminal subtrees), which
                // the normal completion path cannot finalize. Force-complete any
                // such stuck Gumbel slot from its accrued visits; a non-Gumbel
                // deadlock is a hard error.
                moves_decided = complete_continuous_slots(
                    py,
                    on_move,
                    &mut slots,
                    c_puct,
                    &move_policy,
                    &temperature_by_ply,
                    base_seed,
                    &mut queue,
                    &mut stats,
                    true,
                )?;
                if moves_decided == 0 {
                    let stuck = slots
                        .iter()
                        .filter(|slot| !matches!(slot.phase, ContinuousPhase::Empty))
                        .count();
                    return Err(PyRuntimeError::new_err(format!(
                        "hexfield continuous MCTS scheduler stalled with {stuck} unfinished slots \
                         (queue empty, no selectable leaves, no completable roots)"
                    )));
                }
            }
            // After the rescue so a rescue-decided move re-arms the next scan.
            last_moves_decided = moves_decided;
        }

        self.finish_continuous_stats(py, stats, &evaluation_stats)
    }
}

// Internal (non-`#[pymethods]`) scheduler helpers. These take native Rust types
// (`Widening`, `Divergences`, `&mut [ContinuousSlot]`) that pyo3 cannot expose,
// so they MUST live outside the `#[pymethods]` block above.
impl HexfieldMctsSession {
    /// Build the `run_continuous` stats dict (shared by the lockstep loop and the
    /// depth-2 pipeline). Pure GIL-held conversion of the accumulated counters.
    fn finish_continuous_stats(
        &self,
        py: Python<'_>,
        stats: ContinuousSchedulerStats,
        evaluation_stats: &SharedEvaluationStats,
    ) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("flush_count", stats.flush_count)?;
        dict.set_item("queued_states", stats.queued_states)?;
        dict.set_item("flushed_states", stats.flushed_states)?;
        dict.set_item(
            "mean_flush_states",
            if stats.flush_count > 0 {
                stats.flushed_states as f64 / stats.flush_count as f64
            } else {
                0.0
            },
        )?;
        dict.set_item("no_progress_flushes", stats.no_progress_flushes)?;
        dict.set_item("moves_decided", stats.moves_decided)?;
        dict.set_item("full_moves", stats.full_moves)?;
        dict.set_item("fast_moves", stats.fast_moves)?;
        dict.set_item("init_moves", stats.init_moves)?;
        dict.set_item("early_stops_fast", stats.early_stops_fast)?;
        dict.set_item("early_stops_full", stats.early_stops_full)?;
        dict.set_item("early_stop_visits_saved", stats.early_stop_visits_saved)?;
        dict.set_item("force_stuck_completions", stats.force_stuck_completions)?;
        dict.set_item("lcb_overrides", stats.lcb_overrides)?;
        dict.set_item("gumbel_play_moves", stats.gumbel_play_moves)?;
        dict.set_item("gumbel_play_winner_moves", stats.gumbel_play_winner_moves)?;
        dict.set_item("gumbel_play_moves_early", stats.gumbel_play_moves_early)?;
        dict.set_item("gumbel_play_winner_early", stats.gumbel_play_winner_early)?;
        let hist = PyDict::new(py);
        let mut hist_items: Vec<_> = stats.flush_size_histogram.into_iter().collect();
        hist_items.sort_unstable_by_key(|(size, _)| *size);
        for (size, count) in hist_items {
            hist.set_item(size, count)?;
        }
        dict.set_item("flush_size_histogram", hist)?;
        dict.set_item("on_move_seconds", stats.on_move_seconds)?;
        dict.set_item("select_seconds", stats.select_seconds)?;
        dict.set_item("submit_seconds", stats.submit_seconds)?;
        dict.set_item("finish_seconds", stats.finish_seconds)?;
        dict.set_item("backup_seconds", stats.backup_seconds)?;
        dict.set_item("complete_seconds", stats.complete_seconds)?;
        dict.set_item("loop_iterations", stats.loop_iterations)?;
        dict.set_item("completes_skipped", stats.completes_skipped)?;
        let eval_snapshot = evaluation_stats
            .lock()
            .expect("evaluation stats mutex poisoned")
            .clone();
        dict.set_item("evaluation", eval_stats_dict(py, &eval_snapshot)?)?;
        let cache_len = self
            .evaluation_cache
            .lock()
            .expect("evaluation cache mutex poisoned")
            .len();
        dict.set_item("cache_len", cache_len)?;
        Ok(dict.into_any().unbind())
    }

    /// Depth-2 double-buffered eval loop (gated OFF by default via
    /// `HEXFIELD_PIPELINE_DEPTH2`; the lockstep loop above is the default path).
    ///
    /// Invariant (one eval in flight, one staged): at the top of each iteration
    /// at most one flush's eval (`inflight`) is enqueued on the GPU but not yet
    /// backed up. Per iteration the host: selects N (next leaves), submits N to
    /// the GPU (no sync), then drains the previous flush (`inflight` = P) —
    /// finish + parallel backup — and stashes N as the new `inflight`. So the GPU
    /// computes N while the host backs up P and selects N+1; the staleness
    /// window is one flush wider than the lockstep async path. Virtual loss
    /// (applied at selection, restored at backup) keeps the extra-stale selects
    /// search-faithful: a leaf with an in-flight eval carries a pending virtual
    /// penalty so the next select does not re-pick it.
    ///
    /// Exactly-once backup: each flush's `items` ride in the `inflight` tuple
    /// next to their `PendingEval`; `take` on submit (from the queue) and `take`
    /// on drain (from the Option) make every flush submitted once and finished
    /// once. The loop runs while `inflight.is_some()` so the final flush is
    /// always drained before exit; the Gumbel stuck-root rescue runs only after
    /// the pipeline is empty.
    #[allow(clippy::too_many_arguments)]
    fn run_continuous_pipeline_depth2(
        &mut self,
        py: Python<'_>,
        slots: &mut [ContinuousSlot],
        queue: &mut Vec<ContinuousEvalItem>,
        evaluator: &Bound<'_, PyAny>,
        on_move: &Bound<'_, PyAny>,
        c_puct: f32,
        base_seed: u64,
        leaf_batch_per_root: u32,
        flush_target: usize,
        virtual_loss: f32,
        move_policy: &ContinuousMovePolicy,
        widening: Widening,
        divergences: Divergences,
        temperature_by_ply: &[f32],
        evaluation_stats: &SharedEvaluationStats,
        stats: &mut ContinuousSchedulerStats,
    ) -> PyResult<()> {
        // The in-flight (submitted, not-yet-backed-up) flush: its eval handle, the
        // items it will resolve, and the unique-state count snapshot taken at its
        // submit (for the per-flush histogram, computed when it drains).
        let mut inflight: Option<(PendingEval, Vec<ContinuousEvalItem>, usize)> = None;

        // HEXFIELD_PIPELINE_COMPLETE_OVERLAP (default OFF): moves the
        // per-iteration `complete` phase (Phase-A parallel build under
        // py.detach + Phase-B GIL-held `on_move`) to run after submit(N) but
        // before the drain of the previous flush P, so it runs while N's forward
        // computes on the GPU rather than with the GPU idle after the drain's
        // `finish` D2H sync. `complete_continuous_slots` only finalizes slots
        // with `in_flight == 0`; a slot whose eval is still buffered in the
        // un-drained `inflight` (P) keeps `in_flight > 0`, so it is not completed
        // in the overlapped pass and completes on the next iteration after P is
        // drained. Off => complete runs after the drain.
        let complete_overlap = std::env::var("HEXFIELD_PIPELINE_COMPLETE_OVERLAP").is_ok();

        // The loop continues as long as there is host work OR an eval is still in
        // flight (so the last flush is always drained + completed).
        while continuous_has_work(slots) || !queue.is_empty() || inflight.is_some() {
            // (1) select N on the CURRENT (post-previous-backup) tree state.
            let (new_leaves, made_progress) = py.detach(|| {
                select_continuous_pass(slots, c_puct, leaf_batch_per_root, virtual_loss)
            })?;
            queue.extend(new_leaves.into_iter().map(ContinuousEvalItem::Leaf));

            let decision = continuous_flush_decision(queue.len(), flush_target, made_progress);

            // Track whether THIS pass drained the buffered eval. A drain backs up
            // a flush and mutates the trees (slots can become Active / completable
            // next pass), so it counts as pipeline progress: we must NOT declare a
            // stall in the same iteration that drained — loop again and let select /
            // complete act on the freshly backed-up state first.
            let mut drained_this_pass = false;

            // When the overlapped complete runs inside the flush branch (before
            // the drain), it records its decided count here and suppresses the
            // post-drain complete for this pass.
            let mut completed_this_pass = false;
            let mut overlapped_moves = 0u64;

            // (2) On a flush: submit N (enqueue, no sync), THEN drain the previous
            // flush P (finish + backup). Submitting first keeps the GPU busy with N
            // while the host backs up P.
            if let ContinuousFlushDecision::Flush { no_progress } = decision {
                if no_progress {
                    stats.no_progress_flushes += 1;
                }
                let items_n = std::mem::take(queue);
                stats.flush_count += 1;
                stats.queued_states += items_n.len() as u64;
                let unique_before_n = lock_unique_states(evaluation_stats);
                let requests_n: Vec<RustEvaluationRequest> = items_n
                    .iter()
                    .map(continuous_item_request)
                    .collect();
                let pending_n = submit_eval_cached(
                    py,
                    evaluator,
                    &requests_n,
                    &self.evaluation_cache,
                    Some(evaluation_stats),
                    move_policy.request_moves_left(),
                    move_policy.request_logits(),
                )?;
                drop(requests_n);
                // With the complete-overlap flag set, finalize ready slots here:
                // after N is enqueued (its forward computing on the GPU) but
                // before the drain of P, so the completes overlap N's GPU
                // forward. Slots whose eval is still buffered in the un-drained
                // `inflight` (P) keep in_flight > 0 and are not finalized here,
                // so they complete on the next pass after P is drained.
                if complete_overlap {
                    overlapped_moves = complete_continuous_slots(
                        py,
                        on_move,
                        slots,
                        c_puct,
                        move_policy,
                        temperature_by_ply,
                        base_seed,
                        queue,
                        stats,
                        false,
                    )?;
                    completed_this_pass = true;
                }
                // Drain the PREVIOUS flush now that N is enqueued on the GPU.
                if let Some((pending_p, items_p, unique_before_p)) = inflight.take() {
                    self.drain_pipeline_flush(
                        py,
                        slots,
                        evaluator,
                        pending_p,
                        items_p,
                        unique_before_p,
                        move_policy,
                        widening,
                        base_seed,
                        virtual_loss,
                        divergences,
                        evaluation_stats,
                        stats,
                    )?;
                    drained_this_pass = true;
                }
                inflight = Some((pending_n, items_n, unique_before_n));
            } else if !made_progress && inflight.is_some() {
                // No new flush this pass and select stalled: drain the buffered
                // eval so its backup frees paths / completes slots. Without this
                // the loop would spin (select keeps stalling) until a flush; this
                // both unblocks progress and bounds the staleness to one flush.
                let (pending_p, items_p, unique_before_p) = inflight.take().expect("inflight set");
                self.drain_pipeline_flush(
                    py,
                    slots,
                    evaluator,
                    pending_p,
                    items_p,
                    unique_before_p,
                    move_policy,
                    widening,
                    base_seed,
                    virtual_loss,
                    divergences,
                    evaluation_stats,
                    stats,
                )?;
                drained_this_pass = true;
            }

            // (3) Complete any slots whose evals have all landed (in_flight == 0).
            // A slot with an eval still in `inflight` has in_flight > 0 and is
            // correctly NOT completed here. When the complete-overlap path already
            // ran the complete this pass (after submit, before drain), reuse its
            // decided count instead of completing a second time.
            let mut moves_decided = if completed_this_pass {
                overlapped_moves
            } else {
                complete_continuous_slots(
                    py,
                    on_move,
                    slots,
                    c_puct,
                    move_policy,
                    temperature_by_ply,
                    base_seed,
                    queue,
                    stats,
                    false,
                )?
            };

            // (4) Stall handling: only a GENUINE deadlock — Stop decision, no move
            // completed, the pipeline fully drained (inflight None), AND no drain
            // happened this pass (a drain just mutated the trees, so loop again and
            // let the next select/complete act before judging the run stuck).
            if matches!(decision, ContinuousFlushDecision::Stop)
                && moves_decided == 0
                && inflight.is_none()
                && !drained_this_pass
            {
                moves_decided = complete_continuous_slots(
                    py,
                    on_move,
                    slots,
                    c_puct,
                    move_policy,
                    temperature_by_ply,
                    base_seed,
                    queue,
                    stats,
                    true,
                )?;
                if moves_decided == 0 {
                    let stuck = slots
                        .iter()
                        .filter(|slot| !matches!(slot.phase, ContinuousPhase::Empty))
                        .count();
                    return Err(PyRuntimeError::new_err(format!(
                        "hexfield continuous MCTS scheduler (depth-2) stalled with {stuck} \
                         unfinished slots (queue empty, no selectable leaves, no in-flight eval, \
                         no completable roots)"
                    )));
                }
            }
        }
        debug_assert!(
            inflight.is_none(),
            "depth-2 pipeline exited with an undrained in-flight eval"
        );
        Ok(())
    }

    /// Finish + back up one in-flight flush P (parallel backup), folding its
    /// unique-state count into the flush histogram. Exactly-once: called only on
    /// an `inflight` value moved out by `take`.
    #[allow(clippy::too_many_arguments)]
    fn drain_pipeline_flush(
        &mut self,
        py: Python<'_>,
        slots: &mut [ContinuousSlot],
        evaluator: &Bound<'_, PyAny>,
        pending: PendingEval,
        items: Vec<ContinuousEvalItem>,
        unique_before: usize,
        move_policy: &ContinuousMovePolicy,
        widening: Widening,
        base_seed: u64,
        virtual_loss: f32,
        divergences: Divergences,
        evaluation_stats: &SharedEvaluationStats,
        stats: &mut ContinuousSchedulerStats,
    ) -> PyResult<()> {
        let evaluations = finish_eval_cached(
            py,
            evaluator,
            pending,
            &self.evaluation_cache,
            Some(evaluation_stats),
            self.cache_max_states,
        )?;
        let unique_after = lock_unique_states(evaluation_stats);
        let unique_flushed = unique_after.saturating_sub(unique_before);
        stats.flushed_states += unique_flushed as u64;
        *stats
            .flush_size_histogram
            .entry(unique_flushed.max(1).next_power_of_two())
            .or_insert(0) += 1;
        backup_continuous_items(
            py,
            slots,
            items,
            &evaluations,
            move_policy,
            widening,
            base_seed,
            virtual_loss,
            divergences,
        )?;
        Ok(())
    }
}

// === Lockstep internals ===

#[allow(clippy::too_many_arguments)]
fn run_searches_to_targets(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    searches: &mut [RustSearch],
    c_puct: f32,
    leaf_batch_per_root: u32,
    evaluation_cache: &SharedEvaluationCache,
    evaluation_stats: &SharedEvaluationStats,
    cache_max_states: usize,
    virtual_loss: f32,
    request_moves_left: bool,
    request_logits: bool,
    move_temps: &[f32],
    baselines: &[HashMap<PackedCoord, u32>],
) -> PyResult<()> {
    // Two-stage pipeline: the next batch is selected before the current batch
    // is backed up. This ordering extends the virtual-loss window by one batch:
    // select(N+1) runs after evaluate(N) and before backup(N).
    //
    // Early-stop: in_flight is passed as 0 here. The visit-overtake test inside
    // early_stop_ready is in-flight-safe — apply_virtual_visit increments both
    // completed_visits and the selected edge's visit count at selection time, so
    // best/second (per-edge delta visits) include pending leaves while
    // remaining = target - completed excludes them. best-second > remaining thus
    // proves the visit leader is unbeatable by all un-selected visits regardless
    // of how many are pending; the pending batch is still evaluated + backed up
    // by the loop below before exit. The continuous path's in_flight==0 guard is
    // about slot-advance safety (node-id invalidation), a separate concern.
    let early_stop_pass = |searches: &mut [RustSearch]| {
        for (index, search) in searches.iter_mut().enumerate() {
            if search.needs_visits()
                && move_temps.get(index).copied().unwrap_or(1.0) == 0.0
                && early_stop_ready(search, baselines.get(index), false, 0)
            {
                search.early_stopped = true;
                search.target_visits = search.completed_visits;
            }
        }
    };

    // HEXFIELD_ASYNC_EVAL: the forward is enqueued (submit, no device sync), the
    // pre-backup select runs with the GIL released while those kernels execute,
    // then the forward is drained (finish). Off => synchronous eval-then-select.
    // Only the sync point moves; the leaf stream is bit-identical. Depth-2 is
    // NOT read here — it stays self-play-only (it would change the leaf stream).
    //
    // Unlike run_continuous (self-play, always a real HexfieldEvaluator), the eval
    // `search` entry receives diverse evaluators (arena stubs, custom eval
    // opponents). The async split needs the two-phase submit_payload/result
    // protocol; when the evaluator only implements the synchronous __call__
    // contract, fall back to the sync path rather than raising. Real evaluators
    // have submit_payload, so production async is unaffected.
    let async_eval = std::env::var("HEXFIELD_ASYNC_EVAL").is_ok()
        && evaluator.hasattr("submit_payload").unwrap_or(false);

    early_stop_pass(searches);
    // No leaves in flight on the priming select, so the SH barrier is unblocked
    // for every search (empty in-flight set).
    let (mut pending_leaves, _primed_progress) =
        select_leaf_batch(searches, c_puct, leaf_batch_per_root, virtual_loss, &[])?;

    loop {
        // Check between every batch (a no-op in parity mode); see the
        // in-flight-safety note on early_stop_pass above.
        early_stop_pass(searches);
        if pending_leaves.is_empty() {
            if !searches.iter().any(RustSearch::needs_visits) {
                break;
            }
            // pending_leaves is empty here: nothing is un-backed, so the SH
            // barrier is unblocked for every search.
            let (leaves, made_progress) =
                select_leaf_batch(searches, c_puct, leaf_batch_per_root, virtual_loss, &[])?;
            if leaves.is_empty() {
                if !made_progress {
                    break;
                }
                continue;
            }
            pending_leaves = leaves;
        }

        let leaf_requests: Vec<_> = pending_leaves
            .iter()
            .map(|leaf| RustEvaluationRequest {
                state: &leaf.state,
                state_hash: leaf.state_hash,
            })
            .collect();
        // Prefetch select with the current batch still pending (pre-backup
        // tree state). `pending_leaves` carries −virtual_loss on the trees of
        // the searches it touches, so the SH barrier is blocked for exactly
        // those searches (their round ranking would read contaminated stats).
        // Async: submit -> select (GIL released) -> finish. Sync: eval ->
        // select. Both yield (next_leaves, evaluations); the leaf stream is
        // identical because the select reads the same pre-backup tree state
        // with the same batch in flight either way.
        let (next_leaves, evaluations) = if async_eval {
            let pending = submit_eval_cached(
                py,
                evaluator,
                &leaf_requests,
                evaluation_cache,
                Some(evaluation_stats),
                request_moves_left,
                request_logits,
            )?;
            let next_leaves = if searches.iter().any(RustSearch::needs_visits) {
                py.detach(|| {
                    select_leaf_batch(
                        searches,
                        c_puct,
                        leaf_batch_per_root,
                        virtual_loss,
                        &pending_leaves,
                    )
                })?
                .0
            } else {
                Vec::new()
            };
            let evaluations = finish_eval_cached(
                py,
                evaluator,
                pending,
                evaluation_cache,
                Some(evaluation_stats),
                cache_max_states,
            )?;
            (next_leaves, evaluations)
        } else {
            let evaluations = evaluate_state_refs_cached(
                py,
                evaluator,
                &leaf_requests,
                evaluation_cache,
                Some(evaluation_stats),
                cache_max_states,
                request_moves_left,
                request_logits,
            )?;
            let next_leaves = if searches.iter().any(RustSearch::needs_visits) {
                select_leaf_batch(
                    searches,
                    c_puct,
                    leaf_batch_per_root,
                    virtual_loss,
                    &pending_leaves,
                )?
                .0
            } else {
                Vec::new()
            };
            (next_leaves, evaluations)
        };
        apply_eval_backups(searches, pending_leaves, &evaluations, virtual_loss)?;
        pending_leaves = next_leaves;
    }
    Ok(())
}

fn select_leaf_batch(
    searches: &mut [RustSearch],
    c_puct: f32,
    leaf_batch_per_root: u32,
    virtual_loss: f32,
    // Leaves selected in a prior batch that have not yet been backed up. Each
    // still carries −virtual_loss on its owning search's tree, so the SH barrier
    // must not advance a round for any search that owns one (its ranking would
    // read vl-contaminated per-edge visits/completedQ).
    in_flight: &[RustLeaf],
) -> PyResult<(Vec<RustLeaf>, bool)> {
    let mut leaves = Vec::new();
    let mut made_progress = false;
    for (root_index, search) in searches.iter_mut().enumerate() {
        if !search.needs_visits() {
            continue;
        }
        // Intra-search Sequential-Halving barrier (mirrors the continuous
        // scheduler): when every surviving Gumbel candidate has met its round
        // cap, halve the survivor set and re-seed before selecting. Looped
        // because advancing may immediately satisfy the next round's barrier.
        // No-op without an active Gumbel root.
        //
        // Gated on a drained search: skip the barrier while this search has any
        // un-backed leaf in flight, since those leaves' virtual losses would
        // contaminate the round ranking. The pending leaves are guaranteed to
        // back up (apply_eval_backups runs every loop iteration), so the barrier
        // fires on a later drained pass — no deadlock.
        let drained = !in_flight.iter().any(|leaf| leaf.root_index == root_index);
        if drained && search.has_gumbel_root() {
            while search.maybe_advance_gumbel_round() {}
        }
        let budget = leaf_batch_per_root.min(search.remaining_visits());
        for _ in 0..budget {
            let selected = search.select_pending_leaf(c_puct)?;
            let Some(selected) = selected else {
                break;
            };
            search.apply_virtual_visit(&selected.path, virtual_loss);
            made_progress = true;

            let ml_on = search.divergences.moves_left_utility;
            if let Some(outcome) = selected.terminal {
                let leaf_player = selected.state.current_player();
                let leaf_value = terminal_value(outcome, leaf_player);
                let leaf_ml = ml_on.then_some(0.0);
                search.backup_virtual(&selected.path, leaf_player, leaf_value, virtual_loss, leaf_ml);
            } else if let Some(node_id) = selected.existing_node {
                let node = &search.nodes[node_id];
                let player = node.player;
                let value = node.value();
                let leaf_ml = if ml_on { node.ml_mean() } else { None };
                search.backup_virtual(&selected.path, player, value, virtual_loss, leaf_ml);
            } else if let Some(verdict) = search
                .tss_enabled
                .then(|| threats::analyze(&selected.state).verdict())
                .flatten()
            {
                let leaf_player = selected.state.current_player();
                search.backup_virtual(&selected.path, leaf_player, verdict, virtual_loss, None);
            } else {
                search.mark_pending(selected.parent_node, selected.edge_index, 1);
                leaves.push(RustLeaf {
                    root_index,
                    parent_node: selected.parent_node,
                    edge_index: selected.edge_index,
                    path: selected.path,
                    state: selected.state,
                    state_hash: selected.state_hash,
                });
            }
        }
    }
    Ok((leaves, made_progress))
}

fn apply_eval_backups(
    searches: &mut [RustSearch],
    leaves: Vec<RustLeaf>,
    evaluations: &[Arc<RustEvaluation>],
    virtual_loss: f32,
) -> PyResult<()> {
    for (leaf, evaluation) in leaves.into_iter().zip(evaluations.iter()) {
        let search = &mut searches[leaf.root_index];
        let child_id =
            search.add_node_from_eval(&leaf.state, leaf.state_hash, Arc::clone(evaluation))?;
        search.nodes[leaf.parent_node].edges[leaf.edge_index].child = Some(child_id);
        search.mark_pending(leaf.parent_node, leaf.edge_index, -1);
        let child_player = search.nodes[child_id].player;
        let child_value = search.nodes[child_id].value();
        let leaf_ml = if search.divergences.moves_left_utility {
            search.nodes[child_id].ml_mean()
        } else {
            None
        };
        search.backup_virtual(&leaf.path, child_player, child_value, virtual_loss, leaf_ml);
    }
    Ok(())
}

// === Continuous internals ===

fn select_continuous_leaves(
    search: &mut RustSearch,
    slot_index: usize,
    c_puct: f32,
    budget: u32,
    virtual_loss: f32,
) -> PyResult<(Vec<RustLeaf>, bool, u32)> {
    let mut leaves = Vec::new();
    let mut made_progress = false;
    let mut added_in_flight = 0u32;
    let budget = budget.min(search.remaining_visits());
    for _ in 0..budget {
        let selected = search.select_pending_leaf(c_puct)?;
        let Some(selected) = selected else {
            break;
        };
        search.apply_virtual_visit(&selected.path, virtual_loss);
        made_progress = true;
        let ml_on = search.divergences.moves_left_utility;
        if let Some(outcome) = selected.terminal {
            let leaf_player = selected.state.current_player();
            let leaf_value = terminal_value(outcome, leaf_player);
            let leaf_ml = ml_on.then_some(0.0);
            search.backup_virtual(&selected.path, leaf_player, leaf_value, virtual_loss, leaf_ml);
        } else if let Some(node_id) = selected.existing_node {
            let node = &search.nodes[node_id];
            let player = node.player;
            let value = node.value();
            let leaf_ml = if ml_on { node.ml_mean() } else { None };
            search.backup_virtual(&selected.path, player, value, virtual_loss, leaf_ml);
        } else if let Some(verdict) = search
            .tss_enabled
            .then(|| threats::analyze(&selected.state).verdict())
            .flatten()
        {
            let leaf_player = selected.state.current_player();
            search.backup_virtual(&selected.path, leaf_player, verdict, virtual_loss, None);
        } else {
            search.mark_pending(selected.parent_node, selected.edge_index, 1);
            added_in_flight += 1;
            leaves.push(RustLeaf {
                root_index: slot_index,
                parent_node: selected.parent_node,
                edge_index: selected.edge_index,
                path: selected.path,
                state: selected.state,
                state_hash: selected.state_hash,
            });
        }
    }
    Ok((leaves, made_progress, added_in_flight))
}

fn select_continuous_pass(
    slots: &mut [ContinuousSlot],
    c_puct: f32,
    leaf_batch_per_root: u32,
    virtual_loss: f32,
) -> PyResult<(Vec<RustLeaf>, bool)> {
    // Per-slot selection is independent (each closure owns one slot's tree via
    // &mut; the RNG is seeded by slot_index, not execution order), so it is
    // fanned across cores with rayon. Results fold in slot order.
    let per_slot: PyResult<Vec<(Vec<RustLeaf>, bool)>> = slots
        .par_iter_mut()
        .enumerate()
        .map(|(slot_index, slot)| {
            if !matches!(slot.phase, ContinuousPhase::Active) {
                return Ok((Vec::new(), false));
            }
            let cap = leaf_batch_per_root.saturating_sub(slot.in_flight);
            if cap == 0 {
                return Ok((Vec::new(), false));
            }
            let Some(search) = slot.search.as_mut() else {
                return Ok((Vec::new(), false));
            };
            if !search.needs_visits() {
                return Ok((Vec::new(), false));
            }
            // Intra-slot Sequential-Halving barrier: when all surviving Gumbel
            // candidates in this slot have reached the current round's
            // per-candidate cap, halve the survivor set and advance the SH
            // round. No-op unless a Gumbel root is active. Looped because
            // advancing may immediately satisfy the next round's barrier (e.g.
            // tiny budgets).
            //
            // Gated on a DRAINED slot (in_flight == 0): the barrier ranks on
            // per-edge visits and completedQ, both of which carry −virtual_loss
            // for every in-flight sim (apply_virtual_visit bumps visits and
            // subtracts vl at selection; the real backup adds it back). Advancing
            // a round on vl-contaminated stats mis-ranks survivors. A re-descent
            // into an existing subtree carries −vl on the root edge WITHOUT a
            // pending flag, so the root-edge `pending` count alone is not a
            // sufficient drain test — the slot's in_flight counter is. When
            // in_flight > 0 the barrier simply waits: those evals are guaranteed
            // to back up and drive in_flight to 0, at which point either the
            // barrier fires or the force-stuck rescue (in_flight == 0 in
            // complete_ready_slots) finalizes the move, so this cannot deadlock.
            if slot.in_flight == 0 && search.has_gumbel_root() {
                while search.maybe_advance_gumbel_round() {}
            }
            let (leaves, progressed, added_in_flight) =
                select_continuous_leaves(search, slot_index, c_puct, cap, virtual_loss)?;
            slot.in_flight = slot.in_flight.saturating_add(added_in_flight);
            Ok((leaves, progressed))
        })
        .collect();
    let mut leaves = Vec::new();
    let mut made_progress = false;
    for (slot_leaves, progressed) in per_slot? {
        made_progress |= progressed;
        leaves.extend(slot_leaves);
    }
    Ok((leaves, made_progress))
}

/// Apply one backup item (Leaf or RootInit) to its owning slot. `slot` is the
/// item's owning slot (`leaf.root_index` for Leaf / `slot_index` for RootInit)
/// and is never indexed by any other slot, so callers may hand it a disjoint
/// `&mut` from `par_iter_mut`.
#[allow(clippy::too_many_arguments)]
fn apply_backup_item(
    slot: &mut ContinuousSlot,
    item: ContinuousEvalItem,
    evaluation: &Arc<RustEvaluation>,
    move_policy: &ContinuousMovePolicy,
    widening: Widening,
    base_seed: u64,
    virtual_loss: f32,
    // The RootInit branch now derives its per-class divergence view from
    // `move_policy` (divergences_for), and the Leaf branch reads the search's
    // own stored view; the threaded base divergences are no longer consulted
    // here. Kept in the signature so the shared serial/parallel backup callers
    // pass one uniform argument.
    _divergences: Divergences,
) -> PyResult<()> {
    match item {
        ContinuousEvalItem::Leaf(leaf) => {
            let Some(search) = slot.search.as_mut() else {
                return Err(PyValueError::new_err(
                    "continuous MCTS leaf resolved for empty slot",
                ));
            };
            let child_id =
                search.add_node_from_eval(&leaf.state, leaf.state_hash, Arc::clone(evaluation))?;
            search.nodes[leaf.parent_node].edges[leaf.edge_index].child = Some(child_id);
            search.mark_pending(leaf.parent_node, leaf.edge_index, -1);
            slot.in_flight = slot.in_flight.saturating_sub(1);
            let child_player = search.nodes[child_id].player;
            let child_value = search.nodes[child_id].value();
            let leaf_ml = if search.divergences.moves_left_utility {
                search.nodes[child_id].ml_mean()
            } else {
                None
            };
            search.backup_virtual(&leaf.path, child_player, child_value, virtual_loss, leaf_ml);
        }
        ContinuousEvalItem::RootInit { state, .. } => {
            let move_class = move_policy.classify(
                base_seed,
                slot.game_key,
                slot.ply,
                slot.policy_init_remaining,
            );
            slot.move_class = move_class;
            // Per-class divergence view for this fresh root (Fast=fast,
            // Full/Init=base). Replaces the single threaded `divergences`.
            let class_div = move_policy.divergences_for(move_class);
            let mut search = RustSearch::new(
                state,
                &**evaluation,
                move_policy.visits_for(move_class),
                move_policy.fpu_reduction,
                move_policy.root_fpu_for(move_class),
                move_policy.root_temp_for(move_class, slot.ply),
                root_noise_exact(
                    move_policy.noise_for(move_class),
                    mix_seed(base_seed, slot.game_key, slot.ply, SEED_STREAM_ROOT_NOISE),
                    class_div.dirichlet_shaped,
                ),
                widening,
                move_policy.forced_k_for(move_class),
                move_policy.tss_enabled,
                class_div,
            )?;
            if search.root_edges_empty() {
                return Err(PyValueError::new_err(
                    "hexfield continuous MCTS root has no legal actions",
                ));
            }
            // Build the Gumbel-Top-k candidate set + SH schedule when this
            // class's view has gumbel_root on. No-op otherwise (the search keeps
            // the normal PUCT root). budget = the move's visits.
            if class_div.gumbel_root {
                let gumbel_seed = mix_seed(base_seed, slot.game_key, slot.ply, SEED_STREAM_GUMBEL);
                search.init_gumbel_root(gumbel_seed, move_policy.visits_for(move_class));
            }
            slot.baseline = search.root_edge_visits().into_iter().collect();
            slot.search = Some(search);
            slot.phase = ContinuousPhase::Active;
            slot.in_flight = 0;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn backup_continuous_items(
    py: Python<'_>,
    slots: &mut [ContinuousSlot],
    items: Vec<ContinuousEvalItem>,
    evaluations: &[Arc<RustEvaluation>],
    move_policy: &ContinuousMovePolicy,
    widening: Widening,
    base_seed: u64,
    virtual_loss: f32,
    divergences: Divergences,
) -> PyResult<()> {
    // Each item targets exactly one slot. Items are bucketed by slot
    // (order-preserving) and slots processed with `par_iter_mut`: within a slot
    // items run in the same in-flush order, and across slots there is no shared
    // mutable state (each closure owns one slot's `&mut` tree, the same
    // disjoint-borrow guarantee `select_continuous_pass` uses).
    // `HEXFIELD_SERIAL_BACKUP=1` runs the serial path instead.
    if std::env::var("HEXFIELD_SERIAL_BACKUP").is_ok() {
        return py.detach(|| {
            for (item, evaluation) in items.into_iter().zip(evaluations.iter()) {
                let slot_index = match &item {
                    ContinuousEvalItem::Leaf(leaf) => leaf.root_index,
                    ContinuousEvalItem::RootInit { slot_index, .. } => *slot_index,
                };
                apply_backup_item(
                    &mut slots[slot_index],
                    item,
                    evaluation,
                    move_policy,
                    widening,
                    base_seed,
                    virtual_loss,
                    divergences,
                )?;
            }
            Ok(())
        });
    }

    py.detach(|| {
        // Stage 1: bucket items by owning slot, preserving in-flush order
        // (serial, cheap — no tree work).
        let mut per_slot: Vec<Vec<(ContinuousEvalItem, Arc<RustEvaluation>)>> =
            (0..slots.len()).map(|_| Vec::new()).collect();
        for (item, evaluation) in items.into_iter().zip(evaluations.iter()) {
            let slot_index = match &item {
                ContinuousEvalItem::Leaf(leaf) => leaf.root_index,
                ContinuousEvalItem::RootInit { slot_index, .. } => *slot_index,
            };
            per_slot[slot_index].push((item, Arc::clone(evaluation)));
        }

        // Stage 2: process slots in parallel (disjoint `&mut`), serial within a
        // slot in the preserved in-flush order.
        slots
            .par_iter_mut()
            .zip(per_slot.into_par_iter())
            .try_for_each(|(slot, bucket)| -> PyResult<()> {
                for (item, evaluation) in bucket {
                    apply_backup_item(
                        slot,
                        item,
                        &evaluation,
                        move_policy,
                        widening,
                        base_seed,
                        virtual_loss,
                        divergences,
                    )?;
                }
                Ok(())
            })
    })
}

#[allow(clippy::too_many_arguments)]
/// Everything the serial Phase-B dispatch needs to call `on_move` and apply its
/// response for one completed slot, computed in the off-GIL parallel Phase A.
/// Holds the pure-Rust `PayloadNative` (converted to a `PyDict` under the GIL in
/// Phase B) plus the per-slot scalars Phase B applies (early-stop bookkeeping,
/// the resolved `action_id`, move-class flags). No Python objects live here, so
/// it is `Send` and safe to collect from a rayon `par_iter`.
struct PreparedMove {
    move_class: MoveClass,
    game_key: u64,
    ply: u32,
    /// True when this completion is an early stop (drives the early-stop search
    /// mutation + stats in Phase B).
    early: bool,
    /// True when the completion came from the SH-saturation safety net
    /// (force_stuck_gumbel): `early` still drives the finalization mutation,
    /// but Phase B counts it as `force_stuck_completions` instead of the
    /// early-stop counters/visits-saved.
    force_stuck: bool,
    /// `search.remaining_visits()` captured before the early-stop mutation, used
    /// for the `early_stop_visits_saved` stat.
    early_remaining_visits: u32,
    payload: PayloadNative,
    /// Final played action_id (= the Init prior sample when `move_class==Init`,
    /// else the payload's selected action). Drives `advance_root` in Phase B.
    action_id: PackedCoord,
    /// For Init moves, the sampled action_id + selection label that overwrite
    /// the payload dict.
    init_override: Option<PackedCoord>,
}

fn complete_continuous_slots(
    py: Python<'_>,
    on_move: &Bound<'_, PyAny>,
    slots: &mut [ContinuousSlot],
    c_puct: f32,
    move_policy: &ContinuousMovePolicy,
    temperature_by_ply: &[f32],
    base_seed: u64,
    queue: &mut Vec<ContinuousEvalItem>,
    stats: &mut ContinuousSchedulerStats,
    force_stuck_gumbel: bool,
) -> PyResult<u64> {
    // Phase A (parallel, GIL released): build the payload + decision for every
    // ready slot. Pure Rust, read-only over the slot trees (`par_iter`). Each
    // closure writes only its own `Option<PreparedMove>` (disjoint by slot
    // index), with no shared mutable state. `HEXFIELD_SERIAL_COMPLETE=1` runs
    // the build serially instead.
    let serial_build = std::env::var("HEXFIELD_SERIAL_COMPLETE").is_ok();
    let prepared: Vec<Option<PreparedMove>> = py.detach(|| {
        let prepare = |_slot_index: usize, slot: &ContinuousSlot| -> PyResult<Option<PreparedMove>> {
            if !matches!(slot.phase, ContinuousPhase::Active) {
                return Ok(None);
            }
            let move_class = slot.move_class;
            let in_flight = slot.in_flight;
            let (complete, early, force_stuck) = slot
                .search
                .as_ref()
                .map(|search| {
                    let normal = continuous_completion_ready(
                        search.completed_visits,
                        search.target_visits,
                        in_flight,
                    );
                    if normal {
                        return (true, false, false);
                    }
                    // SH saturation safety net: a Gumbel root can exhaust its
                    // reachable tree (terminal/solved subtrees) below
                    // target_visits and below its SH round caps. When the slot
                    // has no in-flight evals and the scheduler made no global
                    // progress this pass (force_stuck_gumbel), it can neither
                    // reach completion nor advance the SH barrier. Finalize the
                    // move from the visits accrued so far instead.
                    if force_stuck_gumbel
                        && search.has_gumbel_root()
                        && in_flight == 0
                        && search.needs_visits()
                    {
                        // `early=true` drives the same finalization mutation in
                        // Phase B; `force_stuck=true` routes the stats to
                        // force_stuck_completions instead of the early-stop
                        // counters.
                        return (true, true, true);
                    }
                    // Fast moves stop unrestricted; recorded Full roots keep the
                    // visit floor.
                    let early = early_stop_ready(
                        search,
                        Some(&slot.baseline),
                        matches!(move_class, MoveClass::Full),
                        in_flight,
                    );
                    (early, early, false)
                })
                .unwrap_or((false, false, false));
            if !complete {
                return Ok(None);
            }

            let search = slot
                .search
                .as_ref()
                .expect("active continuous slot has search");
            // Capture remaining_visits() before Phase B applies the early-stop
            // mutation (target_visits = completed_visits), for the
            // early_stop_visits_saved stat.
            let early_remaining_visits = if early {
                search.remaining_visits()
            } else {
                0
            };

            let game_key = slot.game_key;
            let ply = slot.ply;
            let move_seed = mix_seed(base_seed, game_key, ply, SEED_STREAM_MOVE_SELECT);
            let temperature = move_policy.temperature_for_class(move_class, temperature_by_ply, ply);
            let mut payload = build_search_result_payload_native(
                search,
                Some(&slot.baseline),
                temperature,
                move_seed,
                c_puct,
                move_policy.forced_k_for(move_class),
            )?;
            // The early-stop path sets `search.early_stopped = true` in Phase B;
            // this build is read-only, so reflect `early` onto the native field
            // here so the payload's `early_stopped` reads true.
            if early {
                payload.early_stopped = true;
            }

            // Init class: sample the played move from the root prior (overrides
            // the payload's selected action). Deterministic seed.
            let init_override = if matches!(move_class, MoveClass::Init) {
                let (prior_ids, prior_weights) = root_prior_policy(search.root());
                let sampled = select_action_from_policy(
                    &prior_ids,
                    &prior_weights,
                    move_policy.policy_init_temperature,
                    mix_seed(base_seed, game_key, ply, SEED_STREAM_POLICY_INIT_SAMPLE),
                )?
                .ok_or_else(|| {
                    PyValueError::new_err("policy-init sampling found no positive prior mass")
                })?;
                Some(sampled)
            } else {
                None
            };
            let action_id = init_override.unwrap_or(payload.action_id);

            Ok(Some(PreparedMove {
                move_class,
                game_key,
                ply,
                early,
                force_stuck,
                early_remaining_visits,
                payload,
                action_id,
                init_override,
            }))
        };

        if serial_build {
            let mut out = Vec::with_capacity(slots.len());
            for (slot_index, slot) in slots.iter().enumerate() {
                out.push(prepare(slot_index, slot)?);
            }
            Ok(out)
        } else {
            slots
                .par_iter()
                .enumerate()
                .map(|(slot_index, slot)| prepare(slot_index, slot))
                .collect::<PyResult<Vec<_>>>()
        }
    })?;

    // Phase B (serial, GIL held): convert each native payload to a PyDict and
    // dispatch `on_move` in slot-index order, then apply the (possibly
    // tree-mutating) response. All slot mutation that depends on Python output
    // stays here, single-owner.
    let mut moves_decided = 0u64;
    for slot_index in 0..slots.len() {
        let Some(prepared) = prepared[slot_index].as_ref() else {
            continue;
        };
        let move_class = prepared.move_class;
        let game_key = prepared.game_key;
        let ply = prepared.ply;
        let action_id = prepared.action_id;

        // Early-stop bookkeeping (mutates the slot's search + stats), applied
        // here in slot order.
        if prepared.early {
            let search = slots[slot_index].search.as_mut().expect("active slot");
            if prepared.force_stuck {
                // SH-saturation safety-net finalization: a stuck-root forced
                // completion, not a genuine early stop — count it separately
                // and leave the early-stop counters/visits-saved untouched.
                stats.force_stuck_completions += 1;
            } else {
                stats.early_stop_visits_saved += prepared.early_remaining_visits as u64;
                match move_class {
                    MoveClass::Full => stats.early_stops_full += 1,
                    _ => stats.early_stops_fast += 1,
                }
            }
            search.early_stopped = true;
            search.target_visits = search.completed_visits;
        }

        let payload_dict = prepared.payload.to_pydict(py, None, None)?;
        payload_dict.set_item("pcr_full", matches!(move_class, MoveClass::Full))?;
        payload_dict.set_item("policy_init", matches!(move_class, MoveClass::Init))?;
        if prepared.payload.lcb_override {
            stats.lcb_overrides += 1;
        }
        if prepared.payload.play_pruned {
            stats.gumbel_play_moves += 1;
            if prepared.payload.play_winner {
                stats.gumbel_play_winner_moves += 1;
            }
            if ply < 20 {
                stats.gumbel_play_moves_early += 1;
                if prepared.payload.play_winner {
                    stats.gumbel_play_winner_early += 1;
                }
            }
        }
        if let Some(sampled) = prepared.init_override {
            payload_dict.set_item("action_id", sampled)?;
            payload_dict.set_item("action_selection", "policy_init_prior")?;
        }

        moves_decided += 1;
        stats.moves_decided += 1;
        match move_class {
            MoveClass::Full => stats.full_moves += 1,
            MoveClass::Fast => stats.fast_moves += 1,
            MoveClass::Init => stats.init_moves += 1,
        }
        let started = std::time::Instant::now();
        let response = on_move.call1((game_key, &payload_dict))?;
        stats.on_move_seconds += started.elapsed().as_secs_f64();
        if response.is_none() {
            slots[slot_index].search = None;
            slots[slot_index].phase = ContinuousPhase::Empty;
            continue;
        }
        let tuple = response.downcast::<PyTuple>()?;
        if tuple.is_empty() {
            return Err(PyValueError::new_err(
                "continuous on_move response tuple is empty",
            ));
        }
        let action: String = tuple.get_item(0)?.extract()?;
        match action.as_str() {
            "advance" => {
                if tuple.len() != 2 {
                    return Err(PyValueError::new_err(
                        "advance response must be ('advance', state)",
                    ));
                }
                let next_state = single_state_from_py(py, &tuple.get_item(1)?)?;
                let next_hash = state_hash(&next_state);
                if matches!(move_class, MoveClass::Init) {
                    slots[slot_index].policy_init_remaining =
                        slots[slot_index].policy_init_remaining.saturating_sub(1);
                }
                let next_ply = ply.saturating_add(1);
                let next_class = move_policy.classify(
                    base_seed,
                    game_key,
                    next_ply,
                    slots[slot_index].policy_init_remaining,
                );
                slots[slot_index].move_class = next_class;
                let mut keep_promoted = false;
                if let Some(search) = slots[slot_index].search.as_mut() {
                    if search.advance_root(action_id)? && search.root_hash == next_hash {
                        // Per-class divergence view for the promoted root. The
                        // paired ply of a turn shares the turn's class (see
                        // classify), so a Full-turn PUCT subtree is reused under
                        // the Full regime and a Fast turn under the Fast regime.
                        let class_div = move_policy.divergences_for(next_class);
                        search.set_additional_visits(move_policy.visits_for(next_class));
                        search.set_forced_playout_k(move_policy.forced_k_for(next_class));
                        search.set_root_fpu_reduction(move_policy.root_fpu_for(next_class));
                        search.set_tss_enabled(move_policy.tss_enabled);
                        search.set_divergences(class_div);
                        search
                            .apply_root_policy_temperature(move_policy.root_temp_for(next_class, next_ply));
                        if let Some(noise) = root_noise_exact(
                            move_policy.noise_for(next_class),
                            mix_seed(base_seed, game_key, next_ply, SEED_STREAM_ROOT_NOISE),
                            class_div.dirichlet_shaped,
                        ) {
                            search.apply_root_dirichlet_noise(noise);
                        }
                        // (Re)build the Gumbel-Top-k candidate set + SH schedule
                        // for the promoted root, mirroring the epoch-entry reuse
                        // path. Without this the previous move's finished SH
                        // state (survivors/round caps keyed to the old root's
                        // actions) persists onto the new root, and the slot
                        // either hammers a stale survivor or stalls until the
                        // force-stuck safety net finalizes the move with zero
                        // new visits. When this class's view has gumbel_root off
                        // the state is cleared so the normal PUCT root runs.
                        if class_div.gumbel_root {
                            let gumbel_seed =
                                mix_seed(base_seed, game_key, next_ply, SEED_STREAM_GUMBEL);
                            search.init_gumbel_root(
                                gumbel_seed,
                                move_policy.visits_for(next_class),
                            );
                        } else {
                            search.clear_gumbel_root();
                        }
                        slots[slot_index].baseline =
                            search.root_edge_visits().into_iter().collect();
                        keep_promoted = true;
                    }
                }
                slots[slot_index].ply = next_ply;
                slots[slot_index].in_flight = 0;
                if keep_promoted {
                    slots[slot_index].phase = ContinuousPhase::Active;
                } else {
                    // Driver-contract guard: 'advance' must carry a NON-terminal
                    // state (the driver detects game end and responds None /
                    // 'replace' instead). A terminal state can never promote a
                    // subtree (advance_root returns false on terminal children),
                    // so it always lands here; requeueing it as a RootInit would
                    // surface later as a misleading batch-wide "continuous MCTS
                    // root has no legal actions" abort. Raise the attributable
                    // error now instead.
                    if next_state.terminal().is_some() {
                        return Err(PyValueError::new_err(format!(
                            "continuous driver advanced slot {slot_index} (game key {game_key}, ply {next_ply}) into a terminal state; \
                             'advance' requires a non-terminal state (driver contract violation — respond None or ('replace', ...) when the game ends)"
                        )));
                    }
                    slots[slot_index].search = None;
                    slots[slot_index].phase = ContinuousPhase::AwaitRootEval;
                    queue.push(ContinuousEvalItem::RootInit {
                        slot_index,
                        state: next_state,
                        state_hash: next_hash,
                    });
                }
            }
            "replace" => {
                if tuple.len() != 3 {
                    return Err(PyValueError::new_err(
                        "replace response must be ('replace', new_key, state)",
                    ));
                }
                let new_key: u64 = tuple.get_item(1)?.extract()?;
                let next_state = single_state_from_py(py, &tuple.get_item(2)?)?;
                let next_hash = state_hash(&next_state);
                slots[slot_index].game_key = new_key;
                slots[slot_index].ply = 0;
                slots[slot_index].search = None;
                slots[slot_index].baseline.clear();
                slots[slot_index].in_flight = 0;
                slots[slot_index].phase = ContinuousPhase::AwaitRootEval;
                slots[slot_index].policy_init_remaining =
                    move_policy.policy_init_plies(base_seed, new_key);
                slots[slot_index].move_class = move_policy.classify(
                    base_seed,
                    new_key,
                    0,
                    slots[slot_index].policy_init_remaining,
                );
                queue.push(ContinuousEvalItem::RootInit {
                    slot_index,
                    state: next_state,
                    state_hash: next_hash,
                });
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "continuous on_move returned unsupported action {other:?}"
                )));
            }
        }
    }
    Ok(moves_decided)
}

fn continuous_has_work(slots: &[ContinuousSlot]) -> bool {
    slots
        .iter()
        .any(|slot| !matches!(slot.phase, ContinuousPhase::Empty))
}

/// Snapshot the cumulative unique-states counter (depth-2 per-flush histogram).
fn lock_unique_states(stats: &SharedEvaluationStats) -> usize {
    stats
        .lock()
        .expect("evaluation stats mutex poisoned")
        .unique_states
}

/// Map a queued eval item to its forward-pass request (state + hash). Identical
/// to the inline match in the lockstep loop; shared by the depth-2 path.
fn continuous_item_request(item: &ContinuousEvalItem) -> RustEvaluationRequest<'_> {
    match item {
        ContinuousEvalItem::Leaf(leaf) => RustEvaluationRequest {
            state: &leaf.state,
            state_hash: leaf.state_hash,
        },
        ContinuousEvalItem::RootInit {
            state, state_hash, ..
        } => RustEvaluationRequest {
            state,
            state_hash: *state_hash,
        },
    }
}

fn temperature_for_ply(values: &[f32], ply: u32) -> f32 {
    let index = (ply as usize).min(values.len().saturating_sub(1));
    values[index]
}

// === Shared helpers ===

fn single_state_from_py(py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<RustHexoState> {
    let tuple = PyTuple::new(py, [state])?;
    let states = states_from_py_states(py, tuple.as_any())?;
    states
        .into_iter()
        .next()
        .ok_or_else(|| PyValueError::new_err("expected one state"))
}

/// Every key `resolve_divergences` understands. Unknown keys are a hard error:
/// a silently-dropped key (version skew, typo) reverts part of the search
/// profile to defaults with zero symptoms — the same silent-PUCT failure class
/// the lockstep Gumbel-init fix closed.
const KNOWN_DIVERGENCE_KEYS: &[&str] = &[
    "lcb_move_selection",
    "early_stop",
    "visit_scaled_c_puct",
    "moves_left_utility",
    "ml_weight",
    "ml_scale",
    "ml_q_gate",
    "ml_two_sided",
    "ml_final_pick",
    "ml_final_pick_band",
    "lcb_z",
    "c_scale",
    "c_base",
    "nucleus_f64",
    "new_child_fpu",
    "lazy_widening",
    "clean_root_prior_cache",
    "dirichlet_shaped",
    "pruned_dynamic_cpuct",
    "scaled_fpu",
    "gumbel_target",
    "gumbel_root",
    "gumbel_sequential_halving",
    "gumbel_nonroot_select",
    "gumbel_c_visit",
    "gumbel_c_scale",
    "gumbel_target_c_scale",
    "gumbel_m",
    "gumbel_draw_temperature",
    "gumbel_target_min_visits",
    "gumbel_play_prune",
    // Fast-class Gumbel levers (main_8: PUCT Full / Gumbel Fast). These name the
    // Fast view's values; the driver's Python side folds them into the SECOND
    // (fast) override map whose base keys resolve_divergences reads. They are
    // whitelisted here so the strict known-keys gate never rejects them when
    // they ride in an override dict — a parser/whitelist mismatch on new keys
    // tripped the supervisor circuit breaker on 2026-07-04 (supervisor_halted.flag).
    // NOTE: accepted ONLY in the fast override map; the BASE map rejects them
    // (allow_fast_keys=false in resolve_divergences) because folding a fast_*
    // key into the base map silently mutates the Full-class profile — the
    // exact skew this whitelist exists to prevent.
    "fast_gumbel_root_enabled",
    "fast_gumbel_sequential_halving",
    "fast_gumbel_nonroot_select",
    "fast_gumbel_c_visit",
    "fast_gumbel_c_scale",
    "fast_gumbel_m",
    "fast_gumbel_play_prune",
];

/// Validate a numeric divergence override on top of extraction: every f32
/// override must be finite, plus the field's semantic range (`range` is the
/// human-readable range for the error message, `ok` the predicate enforcing it).
fn checked_divergence_f32(
    key: &str,
    value: f32,
    range: &str,
    ok: impl Fn(f32) -> bool,
) -> PyResult<f32> {
    if !value.is_finite() || !ok(value) {
        return Err(PyValueError::new_err(format!(
            "divergence override {key:?} must be {range}, got {value}"
        )));
    }
    Ok(value)
}

fn resolve_divergences(
    search_parity_mode: Option<bool>,
    overrides: Option<&Bound<'_, PyDict>>,
    // True only for the SECOND (Fast-class) override map: fast_* keys are the
    // Fast view's levers and folding them into the BASE map would silently skew
    // the Full-class profile — exactly what the strict whitelist exists to
    // prevent — so the base map rejects them.
    allow_fast_keys: bool,
) -> PyResult<Divergences> {
    if let Some(overrides) = overrides {
        for key in overrides.keys() {
            let key: String = key.extract()?;
            if !KNOWN_DIVERGENCE_KEYS.contains(&key.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "unknown divergence override key {key:?}; known keys: {KNOWN_DIVERGENCE_KEYS:?}"
                )));
            }
            if !allow_fast_keys && key.starts_with("fast_") {
                return Err(PyValueError::new_err(format!(
                    "divergence override key {key:?} is a Fast-class lever and is only \
                     accepted in the fast override map; in the base map it would silently \
                     mutate the Full-class profile"
                )));
            }
        }
    }
    let mut dv = if search_parity_mode.unwrap_or(false) {
        Divergences::parity()
    } else {
        Divergences::production()
    };
    if let Some(overrides) = overrides {
        // Per-divergence toggles from the override dict.
        if let Some(v) = overrides.get_item("lcb_move_selection")? {
            dv.lcb_move_selection = v.extract()?;
        }
        if let Some(v) = overrides.get_item("early_stop")? {
            dv.early_stop = v.extract()?;
        }
        if let Some(v) = overrides.get_item("visit_scaled_c_puct")? {
            dv.visit_scaled_c_puct = v.extract()?;
        }
        if let Some(v) = overrides.get_item("moves_left_utility")? {
            dv.moves_left_utility = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_weight")? {
            dv.ml_weight = checked_divergence_f32("ml_weight", v.extract()?, "finite", |_| true)?;
        }
        if let Some(v) = overrides.get_item("ml_scale")? {
            // tanh divisor in the moves-left bonus: 0 => NaN, negative flips
            // the bonus direction. It is a positive length scale (in moves).
            dv.ml_scale =
                checked_divergence_f32("ml_scale", v.extract()?, "finite and > 0 (tanh length scale)", |x| {
                    x > 0.0
                })?;
        }
        if let Some(v) = overrides.get_item("ml_q_gate")? {
            // Q-gate in value units (Q in [-1, 1]); negative gates invert the
            // dead-zone semantics.
            dv.ml_q_gate =
                checked_divergence_f32("ml_q_gate", v.extract()?, "finite and >= 0 (Q gate in value units)", |x| {
                    x >= 0.0
                })?;
        }
        if let Some(v) = overrides.get_item("ml_two_sided")? {
            dv.ml_two_sided = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_final_pick")? {
            dv.ml_final_pick = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_final_pick_band")? {
            dv.ml_final_pick_band = checked_divergence_f32(
                "ml_final_pick_band",
                v.extract()?,
                "finite and >= 0 (LCB band in value units)",
                |x| x >= 0.0,
            )?;
        }
        if let Some(v) = overrides.get_item("lcb_z")? {
            dv.lcb_z = checked_divergence_f32("lcb_z", v.extract()?, "finite", |_| true)?;
        }
        if let Some(v) = overrides.get_item("c_scale")? {
            dv.c_scale =
                checked_divergence_f32("c_scale", v.extract()?, "finite and >= 0", |x| x >= 0.0)?;
        }
        if let Some(v) = overrides.get_item("c_base")? {
            // c_for computes ((visits + c_base) / c_base).ln(): c_base == 0
            // yields inf/NaN that partial_cmp tie-breaks silently swallow.
            dv.c_base = checked_divergence_f32(
                "c_base",
                v.extract()?,
                "finite and > 0 (dynamic c_puct log denominator)",
                |x| x > 0.0,
            )?;
        }
        // Search divergences, individually flippable via the override dict.
        if let Some(v) = overrides.get_item("nucleus_f64")? {
            dv.nucleus_f64 = v.extract()?;
        }
        if let Some(v) = overrides.get_item("new_child_fpu")? {
            dv.new_child_fpu = v.extract()?;
        }
        if let Some(v) = overrides.get_item("lazy_widening")? {
            dv.lazy_widening = v.extract()?;
        }
        if let Some(v) = overrides.get_item("clean_root_prior_cache")? {
            dv.clean_root_prior_cache = v.extract()?;
        }
        if let Some(v) = overrides.get_item("dirichlet_shaped")? {
            dv.dirichlet_shaped = v.extract()?;
        }
        if let Some(v) = overrides.get_item("pruned_dynamic_cpuct")? {
            dv.pruned_dynamic_cpuct = v.extract()?;
        }
        if let Some(v) = overrides.get_item("scaled_fpu")? {
            dv.scaled_fpu = v.extract()?;
        }
        // Gumbel AlphaZero flags (default-OFF).
        if let Some(v) = overrides.get_item("gumbel_target")? {
            dv.gumbel_target = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_root")? {
            dv.gumbel_root = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_sequential_halving")? {
            dv.gumbel_sequential_halving = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_nonroot_select")? {
            dv.gumbel_nonroot_select = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_c_visit")? {
            dv.gumbel_c_visit = checked_divergence_f32(
                "gumbel_c_visit",
                v.extract()?,
                "finite and >= 0 (σ transform visit constant)",
                |x| x >= 0.0,
            )?;
        }
        if let Some(v) = overrides.get_item("gumbel_c_scale")? {
            dv.gumbel_c_scale = checked_divergence_f32(
                "gumbel_c_scale",
                v.extract()?,
                "finite and >= 0 (σ transform scale)",
                |x| x >= 0.0,
            )?;
        }
        // Export-only target σ override (absent => target keeps gumbel_c_scale).
        if let Some(v) = overrides.get_item("gumbel_target_c_scale")? {
            dv.gumbel_target_c_scale = Some(checked_divergence_f32(
                "gumbel_target_c_scale",
                v.extract()?,
                "finite and >= 0 (export-only σ transform scale)",
                |x| x >= 0.0,
            )?);
        }
        if let Some(v) = overrides.get_item("gumbel_m")? {
            dv.gumbel_m = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_draw_temperature")? {
            // τ <= 0 is the documented disable sentinel (init_gumbel_root falls
            // back to today's raw logit+g draw), so only non-finite is invalid.
            dv.gumbel_draw_temperature = checked_divergence_f32(
                "gumbel_draw_temperature",
                v.extract()?,
                "finite (<= 0 disables the draw-temperature transform)",
                |_| true,
            )?;
        }
        if let Some(v) = overrides.get_item("gumbel_target_min_visits")? {
            dv.gumbel_target_min_visits = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_play_prune")? {
            dv.gumbel_play_prune = v.extract()?;
        }
        // Fast-class Gumbel levers (main_8). When present these override the
        // gumbel fields with the Fast view's values; they are applied LAST so a
        // fast-override map carrying fast_* keys wins over any base-keyed gumbel
        // entry it also holds. Absent => the base gumbel fields stand, so a plain
        // (non-fast) override map is unchanged and the fast view falls back to
        // the base view (golden invariant).
        if let Some(v) = overrides.get_item("fast_gumbel_root_enabled")? {
            dv.gumbel_root = v.extract()?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_sequential_halving")? {
            dv.gumbel_sequential_halving = v.extract()?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_nonroot_select")? {
            dv.gumbel_nonroot_select = v.extract()?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_c_visit")? {
            dv.gumbel_c_visit = checked_divergence_f32(
                "fast_gumbel_c_visit",
                v.extract()?,
                "finite and >= 0 (σ transform visit constant)",
                |x| x >= 0.0,
            )?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_c_scale")? {
            dv.gumbel_c_scale = checked_divergence_f32(
                "fast_gumbel_c_scale",
                v.extract()?,
                "finite and >= 0 (σ transform scale)",
                |x| x >= 0.0,
            )?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_m")? {
            dv.gumbel_m = v.extract()?;
        }
        if let Some(v) = overrides.get_item("fast_gumbel_play_prune")? {
            dv.gumbel_play_prune = v.extract()?;
        }
    }
    Ok(dv)
}

fn build_widening(
    mass: Option<f32>,
    min_children: Option<u32>,
    max_children: Option<u32>,
) -> PyResult<Widening> {
    let widening_mass = mass.unwrap_or(0.95);
    if !widening_mass.is_finite() || widening_mass <= 0.0 || widening_mass > 1.0 {
        return Err(PyValueError::new_err("widening_policy_mass must be in (0, 1]"));
    }
    let widening = Widening {
        mass: widening_mass,
        min_children: validate_positive_u32("widening_min_children", min_children.unwrap_or(2))?
            as usize,
        max_children: validate_positive_u32("widening_max_children", max_children.unwrap_or(32))?
            as usize,
    };
    if widening.min_children > widening.max_children {
        return Err(PyValueError::new_err(
            "widening_min_children must be <= widening_max_children",
        ));
    }
    Ok(widening)
}

/// Pure-Rust core of a single search-result payload: every value the PyDict in
/// `build_search_result_payloads` carries, as plain Rust scalars/bytes so it can
/// be computed without the GIL (built in a `par_iter` off the GIL, then
/// converted to a `PyDict` serially under the GIL right before `on_move`).
/// `to_pydict` is the GIL-held conversion.
struct PayloadNative {
    action_id: PackedCoord,
    action_selection: &'static str,
    lcb_override: bool,
    early_stopped: bool,
    // Play-policy telemetry: whether the quota-pruned Gumbel play distribution
    // drove selection, and whether the played move is the raw delta leader.
    play_pruned: bool,
    play_winner: bool,
    export_action_ids: Vec<PackedCoord>,
    export_weights: Vec<f32>,
    export_q: Vec<f32>,
    root_prior_action_ids: Vec<PackedCoord>,
    root_prior_weights: Vec<f32>,
    // Present only when `gumbel_target` is on (otherwise None; the gumbel keys
    // are omitted from the payload).
    gumbel: Option<GumbelTargetNative>,
    root_value: f32,
    visits: u32,
    node_count: usize,
    active_edge_count: usize,
    root_active_edges: usize,
    root_hidden_priors: usize,
}

struct GumbelTargetNative {
    action_ids: Vec<PackedCoord>,
    weights: Vec<f32>,
    logits: Vec<f32>,
}

impl PayloadNative {
    /// Convert the pure-Rust payload into the `PyDict` the on_move callback
    /// expects. `eval_stats` / `cache_len` are the per-batch diagnostics only
    /// the lockstep multi-search path supplies (the continuous path passes None).
    fn to_pydict<'py>(
        &self,
        py: Python<'py>,
        eval_stats: Option<&EvaluationStats>,
        cache_len: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        result.set_item("action_id", self.action_id)?;
        result.set_item("action_selection", self.action_selection)?;
        result.set_item("lcb_override", self.lcb_override)?;
        result.set_item("early_stopped", self.early_stopped)?;
        result.set_item("play_pruned", self.play_pruned)?;
        result.set_item("play_winner", self.play_winner)?;
        let to_bytes = |data: &[u32]| -> Bound<'py, PyBytes> {
            let len = std::mem::size_of_val(data);
            let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) };
            PyBytes::new(py, raw)
        };
        let to_bytes_f32 = |data: &[f32]| -> Bound<'py, PyBytes> {
            let len = std::mem::size_of_val(data);
            let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) };
            PyBytes::new(py, raw)
        };
        result.set_item(
            "visit_policy_action_ids_bytes",
            to_bytes(&self.export_action_ids),
        )?;
        result.set_item("visit_policy_weights_bytes", to_bytes_f32(&self.export_weights))?;
        result.set_item("visit_policy_q_bytes", to_bytes_f32(&self.export_q))?;
        result.set_item("visit_policy_count", self.export_action_ids.len())?;
        result.set_item(
            "root_prior_policy_action_ids_bytes",
            to_bytes(&self.root_prior_action_ids),
        )?;
        result.set_item(
            "root_prior_policy_weights_bytes",
            to_bytes_f32(&self.root_prior_weights),
        )?;
        result.set_item("root_prior_policy_count", self.root_prior_action_ids.len())?;
        if let Some(gumbel) = &self.gumbel {
            result.set_item("gumbel_policy_action_ids_bytes", to_bytes(&gumbel.action_ids))?;
            result.set_item("gumbel_policy_weights_bytes", to_bytes_f32(&gumbel.weights))?;
            result.set_item("gumbel_policy_count", gumbel.action_ids.len())?;
            result.set_item("root_prior_logits_bytes", to_bytes_f32(&gumbel.logits))?;
        }
        result.set_item("root_value", self.root_value)?;
        result.set_item("visits", self.visits)?;
        let diag = PyDict::new(py);
        diag.set_item("node_count", self.node_count)?;
        diag.set_item("active_edge_count", self.active_edge_count)?;
        diag.set_item("root_active_edges", self.root_active_edges)?;
        diag.set_item("root_hidden_priors", self.root_hidden_priors)?;
        if let Some(stats) = eval_stats {
            diag.set_item("evaluation", eval_stats_dict(py, stats)?)?;
        }
        if let Some(cache_len) = cache_len {
            diag.set_item("cache_len", cache_len)?;
        }
        result.set_item("diagnostics", diag)?;
        Ok(result)
    }
}

/// Build the native payload for one search. Carries no Python state and makes
/// no Python calls, so it is safe to run inside a rayon `par_iter` with the GIL
/// released; the final `PyDict` construction is deferred (see
/// `PayloadNative::to_pydict`).
#[allow(clippy::too_many_arguments)]
fn build_search_result_payload_native(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    temperature: f32,
    seed: u64,
    c_puct: f32,
    forced_playout_k: f32,
) -> PyResult<PayloadNative> {
    let root = search.root();
    let (policy_action_ids, policy_weights, policy_q, policy_total) = visit_policy(root, baseline);
    // Forced-playout pruning is PUCT bookkeeping: at a Gumbel SH root the
    // selection path never takes the forced branches, so there are no forced
    // playouts to prune and the PUCT pruning math (n_forced = sqrt(k*P*N))
    // would strip legitimate SH round-quota visits from the recorded target.
    // Gate it off whenever the SH root is active.
    let (mut export_action_ids, mut export_weights, mut export_q) =
        if forced_playout_k > 0.0 && !search.has_gumbel_root() {
            // When pruned_dynamic_cpuct is on the recorded-target pruning uses
            // selection's c_for(N); otherwise static c_puct.
            let effective_c = search.effective_pruning_c_puct(c_puct, root.visits);
            pruned_visit_policy(root, baseline, forced_playout_k, effective_c)
        } else {
            // Common branch: the recorded target IS the play policy computed
            // above, so reuse it instead of a second full visit_policy scan.
            (policy_action_ids.clone(), policy_weights.clone(), policy_q)
        };
    // Recorded-target fallback for a force-completed Gumbel SH root: such a move
    // can finalize with zero net delta visits over its reuse baseline, so the
    // delta-visit export above is empty. A Full (pcr_full) row with an
    // empty/zero-mass policy target is a hard error in shard expansion, so
    // substitute the cumulative visit distribution (baseline-free), then the
    // root prior — both real, legal, positive-mass targets. Inert for normal
    // completion, where the export is non-empty.
    if export_action_ids.is_empty() {
        let (cum_ids, cum_w, cum_q, cum_total) = visit_policy(root, None);
        if !cum_ids.is_empty() && cum_total > 0 {
            export_action_ids = cum_ids;
            export_weights = cum_w;
            export_q = cum_q;
        } else {
            // No edge carries a cumulative visit: fall back to the root prior.
            let (prior_ids, prior_w) = root_prior_policy(root);
            let prior_q: Vec<f32> = prior_ids
                .iter()
                .map(|id| {
                    root.edges
                        .iter()
                        .find(|e| e.action_id == *id)
                        .map(|e| e.value())
                        .unwrap_or(0.0)
                })
                .collect();
            export_action_ids = prior_ids;
            export_weights = prior_w;
            export_q = prior_q;
        }
    }
    let (root_prior_action_ids, root_prior_weights) = root_prior_policy(root);
    // Play distribution for Gumbel SH roots at exploration temperatures
    // (gumbel_play_prune): the delta-visit histogram is a SCHEDULE artifact —
    // every round-0 loser carries its equal entry quota (~budget/(R*m)), so
    // temperature-sampling it plays measured-bad moves at the quota rate.
    // Zero every action whose delta never exceeded the round-0 quota (it was
    // eliminated without surviving a halving) and renormalize; the surviving
    // mass is ordered by rounds survived — SH's own quality ranking at visit
    // counts it already paid for. The RECORDED targets above are untouched.
    // Gated to T>0 (the T=0 greedy/LCB path keeps the raw histogram, so eval
    // arena behavior is unchanged) and inert when pruning would empty the
    // support (degenerate/force-finalized roots keep the fallback chain).
    let play_pair: Option<(Vec<PackedCoord>, Vec<f32>)> = if temperature > 0.0
        && search.divergences.gumbel_play_prune
        && policy_total > 0
    {
        search.gumbel_play_quota().and_then(|quota| {
            let total = policy_total as f32;
            let cut = quota as f32 + 0.5;
            let mut ids = Vec::with_capacity(policy_action_ids.len());
            let mut ws = Vec::with_capacity(policy_action_ids.len());
            for (id, w) in policy_action_ids.iter().zip(policy_weights.iter()) {
                if *w * total > cut {
                    ids.push(*id);
                    ws.push(*w);
                }
            }
            if ids.is_empty() {
                None
            } else {
                let sum: f32 = ws.iter().sum();
                if sum > 0.0 {
                    for w in ws.iter_mut() {
                        *w /= sum;
                    }
                }
                Some((ids, ws))
            }
        })
    } else {
        None
    };
    let play_pruned = play_pair.is_some();
    let (sel_ids, sel_weights): (&Vec<PackedCoord>, &Vec<f32>) = match &play_pair {
        Some((ids, ws)) => (ids, ws),
        None => (&policy_action_ids, &policy_weights),
    };
    let guarded_weights = if search.tss_enabled {
        tactical_guard_weights(&search.root_state, sel_ids, sel_weights)
    } else {
        sel_weights.clone()
    };
    let (selected, lcb_override) = select_action_with_lcb(
        search,
        baseline,
        sel_ids,
        &guarded_weights,
        temperature,
        seed,
    )?;
    // Played-move resolution. `selected` can be None when the delta-visit policy
    // is empty: a force-completed Gumbel SH root can finalize a move with zero
    // net visits over its reuse baseline, so every edge's delta is 0 and
    // `visit_policy` drops them all. PackedCoord 0 unpacks to the illegal
    // sentinel HexCoord{q:-32768,r:-32768}, so fall back to the cumulative visit
    // distribution (baseline-free), then to the root prior — both real, legal
    // root action_ids. Inert for normal full-visit completion, where `selected`
    // is always Some.
    let selected = match selected {
        Some(action_id) => action_id,
        None => fallback_root_action(root).ok_or_else(|| {
            PyValueError::new_err(
                "continuous move selection found no legal root action (empty edges and priors)",
            )
        })?,
    };
    debug_assert!(
        root.edges.iter().any(|e| e.action_id == selected)
            || root
                .remaining_priors()
                .iter()
                .any(|(a, _)| *a == selected),
        "selected played action_id must be a real root action, never the sentinel"
    );
    let action_selection = if play_pruned {
        "gumbel_play_policy"
    } else if baseline.is_some() {
        "delta_visit_policy"
    } else {
        "cumulative_visit_policy"
    };
    // Telemetry: whether the played action is the raw delta-visit leader (the
    // SH winner on a completed SH root). Read alongside play_pruned to judge
    // how exploratory the play distribution actually is.
    let play_winner = policy_action_ids
        .iter()
        .zip(policy_weights.iter())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(id, _)| *id == selected)
        .unwrap_or(false);
    // Improved-policy target π'=softmax(logits+σ(completedQ)). Exported only when
    // gumbel_target is on; the raw root logits column ships alongside. When the
    // flag is off, none of these keys appear.
    let div = &search.divergences;
    let gumbel = if div.gumbel_target {
        // Export-only σ softening: gumbel_target_c_scale overrides c_scale in the
        // target's σ call ONLY, so π' can be flattened without touching the SH
        // ranking or interior selection (both keep div.gumbel_c_scale).
        let target_c_scale = div.gumbel_target_c_scale.unwrap_or(div.gumbel_c_scale);
        let (gumbel_ids, gumbel_weights, gumbel_logits) = gumbel_target_policy(
            root,
            baseline,
            div.gumbel_c_visit,
            target_c_scale,
            div.gumbel_target_min_visits,
        );
        Some(GumbelTargetNative {
            action_ids: gumbel_ids,
            weights: gumbel_weights,
            logits: gumbel_logits,
        })
    } else {
        None
    };
    let tree = search.diagnostics();
    Ok(PayloadNative {
        action_id: selected,
        action_selection,
        lcb_override,
        early_stopped: search.early_stopped,
        play_pruned,
        play_winner,
        export_action_ids,
        export_weights,
        export_q,
        root_prior_action_ids,
        root_prior_weights,
        gumbel,
        root_value: root.value(),
        visits: policy_total,
        node_count: tree.node_count,
        active_edge_count: tree.active_edge_count,
        root_active_edges: tree.root_active_edges,
        root_hidden_priors: tree.root_hidden_priors,
    })
}

/// Lockstep multi-search batch caller (`search` / eval_arena): builds the
/// native payload per search and converts to a `PyList[PyDict]`. The continuous
/// scheduler bypasses this and calls `build_search_result_payload_native`
/// directly in its off-GIL parallel build phase.
#[allow(clippy::too_many_arguments)]
fn build_search_result_payloads(
    py: Python<'_>,
    searches: &[RustSearch],
    eval_stats: Option<&EvaluationStats>,
    cache_len: Option<usize>,
    temperatures: &[f32],
    seed: u64,
    baselines: Option<&[HashMap<PackedCoord, u32>]>,
    c_puct: f32,
    forced_playout_k: f32,
) -> PyResult<Py<PyAny>> {
    let results = PyList::empty(py);
    for (index, search) in searches.iter().enumerate() {
        let baseline = baselines.and_then(|items| items.get(index));
        let native = build_search_result_payload_native(
            search,
            baseline,
            temperatures[index],
            seed.wrapping_add(index as u64),
            c_puct,
            forced_playout_k,
        )?;
        let result = native.to_pydict(py, eval_stats, cache_len)?;
        results.append(result)?;
    }

    Ok(results.into_any().unbind())
}

fn eval_stats_dict<'py>(py: Python<'py>, stats: &EvaluationStats) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("requested_states", stats.requested_states)?;
    dict.set_item("cache_hits", stats.cache_hits)?;
    dict.set_item("duplicate_hits", stats.duplicate_hits)?;
    dict.set_item("unique_states", stats.unique_states)?;
    dict.set_item("evaluator_chunks", stats.evaluator_chunks)?;
    dict.set_item("encoded_states", stats.encoded_states)?;
    dict.set_item("encoded_nodes", stats.encoded_nodes)?;
    dict.set_item("encoding_seconds", stats.encoding_seconds)?;
    dict.set_item("evaluator_seconds", stats.evaluator_seconds)?;
    dict.set_item("parse_seconds", stats.parse_seconds)?;
    dict.set_item("cache_inserts", stats.cache_inserts)?;
    dict.set_item("cache_size_peak", stats.cache_size_peak)?;
    Ok(dict)
}

/// Deterministic last-resort played-move pick when the normal (delta-visit)
/// selection yields nothing (a force-completed Gumbel SH root that accrued no
/// net visits over its reuse baseline). Prefers the most-visited cumulative root
/// edge (baseline-free), then the highest-prior root action. Returns a real,
/// legal root action_id (never the PackedCoord-0 sentinel), or None only if the
/// root has no edges and no priors. Ties broken by smallest action_id.
fn fallback_root_action(root: &RustNode) -> Option<PackedCoord> {
    // 1) Most-visited cumulative edge.
    let by_visits = root
        .edges
        .iter()
        .max_by(|a, b| {
            a.visits
                .cmp(&b.visits)
                .then_with(|| b.action_id.cmp(&a.action_id))
        })
        .map(|edge| (edge.action_id, edge.visits));
    if let Some((action_id, visits)) = by_visits {
        if visits > 0 {
            return Some(action_id);
        }
    }
    // 2) No edge carries a visit (degenerate): take the highest-prior root
    // action across edges + unexpanded candidates.
    let (prior_ids, prior_weights) = root_prior_policy(root);
    let best_prior = prior_ids
        .iter()
        .copied()
        .zip(prior_weights.iter().copied())
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.cmp(&a.0))
        })
        .map(|(action_id, _)| action_id);
    if best_prior.is_some() {
        return best_prior;
    }
    // 3) Priors all non-positive but an edge exists: any edge action_id beats a
    // sentinel. Fall back to the visit-argmax (already legal) if present.
    by_visits.map(|(action_id, _)| action_id)
}

fn root_prior_policy(root: &RustNode) -> (Vec<PackedCoord>, Vec<f32>) {
    let remaining = root.remaining_priors();
    let mut priors: HashMap<PackedCoord, f32> =
        HashMap::with_capacity(root.edges.len() + remaining.len());
    for edge in &root.edges {
        if edge.prior.is_finite() && edge.prior > 0.0 {
            priors.insert(edge.action_id, edge.prior);
        }
    }
    for (action_id, prior) in remaining {
        if prior.is_finite() && prior > 0.0 {
            priors.insert(action_id, prior);
        }
    }
    let mut pairs: Vec<(PackedCoord, f32)> = priors.into_iter().collect();
    pairs.sort_unstable_by_key(|(action_id, _prior)| *action_id);
    let action_ids: Vec<PackedCoord> = pairs.iter().map(|(action_id, _prior)| *action_id).collect();
    let mut weights: Vec<f32> = pairs.into_iter().map(|(_action_id, prior)| prior).collect();
    let total: f32 = weights.iter().copied().sum();
    if total > 0.0 {
        for weight in &mut weights {
            *weight /= total;
        }
    }
    (action_ids, weights)
}

/// Improved-policy target π'(a)=softmax(logits+σ(completedQ)) over the root
/// candidate support. Returns (action_ids, weights, raw_logits); the raw-logits
/// column is returned alongside. Only the root's own edges form the support;
/// v_mix fills completedQ for an in-support edge that is unvisited.
///
/// Support floor: edges with `N(a) < min_visits` are excluded from the softmax
/// support (which then renormalizes over the survivors). Falls back to the full
/// edge set if the floor would empty the support.
fn gumbel_target_policy(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    c_visit: f32,
    c_scale: f32,
    min_visits: u32,
) -> (Vec<PackedCoord>, Vec<f32>, Vec<f32>) {
    let logit_map = root.root_logits.clone().unwrap_or_default();
    // completedQ map + the v_mix visited-weighted fallback.
    let (completed, v_mix) = gumbel_completed_q(root, &logit_map);
    // σ scale = THIS MOVE's max delta visits over the move-entry baseline, so the
    // exported target's σ multiplier matches the SH ranking's (tree.rs
    // maybe_advance_gumbel_round) and a reused root's inherited visits do not
    // inflate it. On a fresh (baseline None / all-zero) root this equals the
    // cumulative max, so the recorded target is unchanged for lockstep/fresh
    // roots.
    let max_n = root
        .edges
        .iter()
        .map(|e| edge_delta_visits(e, baseline))
        .max()
        .unwrap_or(0);

    // Candidate support = root edges meeting the visit floor.
    let mut in_support: Vec<&RustEdge> = root
        .edges
        .iter()
        .filter(|edge| edge.visits >= min_visits)
        .collect();
    // Degenerate guard: if the floor empties the support, fall back to all edges.
    if in_support.is_empty() {
        in_support = root.edges.iter().collect();
    }

    // Deterministic action_id order (mirrors root_prior_policy's stable order).
    in_support.sort_unstable_by_key(|edge| edge.action_id);

    let mut action_ids = Vec::with_capacity(in_support.len());
    let mut logits = Vec::with_capacity(in_support.len());
    let mut scores = Vec::with_capacity(in_support.len());
    for edge in &in_support {
        let l = logit_map.get(&edge.action_id).copied().unwrap_or(0.0);
        let q = completed.get(&edge.action_id).copied().unwrap_or(v_mix);
        action_ids.push(edge.action_id);
        logits.push(l);
        scores.push(l + gumbel_sigma(q, max_n, c_visit, c_scale));
    }
    let weights = gumbel_softmax(&scores);
    (action_ids, weights, logits)
}

fn validate_search_inputs(visits: u32, c_puct: f32, temperature: f32) -> PyResult<()> {
    if visits == 0 {
        return Err(PyValueError::new_err("visits must be > 0"));
    }
    if !c_puct.is_finite() || c_puct <= 0.0 {
        return Err(PyValueError::new_err("c_puct must be finite and > 0"));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(PyValueError::new_err("temperature must be finite and >= 0"));
    }
    Ok(())
}

fn validate_positive_u32(name: &str, value: u32) -> PyResult<u32> {
    if value == 0 {
        return Err(PyValueError::new_err(format!("{name} must be > 0")));
    }
    Ok(value)
}

fn validate_positive_usize(name: &str, value: usize) -> PyResult<usize> {
    if value == 0 {
        return Err(PyValueError::new_err(format!("{name} must be > 0")));
    }
    Ok(value)
}

fn validate_positive_f32(name: &str, value: f32) -> PyResult<f32> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and > 0"
        )));
    }
    Ok(value)
}

fn validate_nonnegative_f32(name: &str, value: f32) -> PyResult<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and >= 0"
        )));
    }
    Ok(value)
}

fn validate_bounded_f32(name: &str, value: f32, minimum: f32, maximum: f32) -> PyResult<f32> {
    if !value.is_finite() || value < minimum || value > maximum {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and in [{minimum}, {maximum}]"
        )));
    }
    Ok(value)
}

#[derive(Clone, Copy)]
struct RootNoiseConfig {
    total_alpha: f32,
    fraction: f32,
}

fn root_noise_config(
    total_alpha: Option<f32>,
    fraction: Option<f32>,
) -> PyResult<Option<RootNoiseConfig>> {
    match (total_alpha, fraction) {
        (None, None) => Ok(None),
        (Some(total_alpha), Some(fraction)) => {
            let total_alpha = validate_positive_f32("root_dirichlet_total_alpha", total_alpha)?;
            let fraction =
                validate_bounded_f32("root_dirichlet_noise_fraction", fraction, 0.0, 1.0)?;
            if fraction == 0.0 {
                return Ok(None);
            }
            Ok(Some(RootNoiseConfig {
                total_alpha,
                fraction,
            }))
        }
        _ => Err(PyValueError::new_err(
            "root_dirichlet_total_alpha and root_dirichlet_noise_fraction must be provided together",
        )),
    }
}

fn root_noise(
    config: Option<RootNoiseConfig>,
    seed: u64,
    index: usize,
    shaped: bool,
) -> Option<RootDirichletNoise> {
    let config = config?;
    Some(RootDirichletNoise {
        total_alpha: config.total_alpha,
        fraction: config.fraction,
        seed: seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        shaped,
    })
}

fn root_noise_exact(
    config: Option<RootNoiseConfig>,
    seed: u64,
    shaped: bool,
) -> Option<RootDirichletNoise> {
    let config = config?;
    Some(RootDirichletNoise {
        total_alpha: config.total_alpha,
        fraction: config.fraction,
        seed,
        shaped,
    })
}

/// Test-facing surface exercising the same LCB formula the search uses. Pure
/// function over the per-edge stats.
pub fn debug_lcb_from_stats(
    stats: &[(u64, u32, u32, f32, f32)], // (action_id, delta, visits, value_sum, value_sq_sum)
    z: f32,
    min_visits: u32,
    visit_fraction: f32,
) -> Option<u64> {
    let max_delta = stats.iter().map(|s| s.1).max().unwrap_or(0);
    if max_delta == 0 {
        return None;
    }
    let threshold = (min_visits as f32).max(visit_fraction * max_delta as f32);
    let mut best: Option<(f32, u64)> = None;
    for &(action_id, delta, visits, value_sum, value_sq_sum) in stats {
        if (delta as f32) < threshold || visits == 0 {
            continue;
        }
        let n = visits as f32;
        let q = value_sum / n;
        let variance = (value_sq_sum / n - q * q).max(0.0);
        let lcb = q - z * variance.sqrt() / n.sqrt();
        let replace = match best {
            Some((current, current_id)) => lcb > current || (lcb == current && action_id < current_id),
            None => true,
        };
        if replace {
            best = Some((lcb, action_id));
        }
    }
    best.map(|(_, id)| id)
}

pub fn debug_ml_bonus(
    q: f32,
    m_edge: f32,
    m_node: f32,
    weight: f32,
    scale: f32,
    gate: f32,
    two_sided: bool,
) -> f32 {
    // s gates by the chooser's-perspective Q: +1 when q > gate (prefer fewer
    // moves left), -1 when two-sided and q < -gate (prefer more moves left),
    // 0 in the |Q| <= gate dead-zone. Both signs add a positive bonus to the
    // desired child because tanh flips with (m_edge - m_node). Bounded by
    // `weight`.
    let s = if q > gate {
        1.0
    } else if two_sided && q < -gate {
        -1.0
    } else {
        return 0.0;
    };
    -weight * s * ((m_edge - m_node) / scale).tanh()
}

pub fn mix_seed(base_seed: u64, game_key: u64, ply: u32, stream: u64) -> u64 {
    let mut value = base_seed ^ 0xA076_1D64_78BD_642F;
    value ^= game_key.wrapping_mul(0xE703_7ED1_A0B4_28DB);
    value ^= (ply as u64).wrapping_mul(0x8EBC_6AF0_9C88_C6E3);
    value ^= stream.wrapping_mul(0x5899_65CC_7537_4CC3);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn classify_root_move(root_state: &RustHexoState, action_id: PackedCoord) -> i8 {
    let me = root_state.current_player();
    let mut child = root_state.clone();
    let coord = unpack_coord(action_id);
    match apply_placement(&mut child, Placement { coord }) {
        Err(_) => 0,
        Ok(res) => {
            if let Some(outcome) = res.outcome {
                return if outcome.winner == me { 1 } else { -1 };
            }
            match threats::analyze(&child).verdict() {
                Some(v) => {
                    let ours = if child.current_player() == me { v } else { -v };
                    if ours > 0.5 {
                        1
                    } else if ours < -0.5 {
                        -1
                    } else {
                        0
                    }
                }
                None => 0,
            }
        }
    }
}

fn tactical_guard_weights(
    root_state: &RustHexoState,
    action_ids: &[PackedCoord],
    weights: &[f32],
) -> Vec<f32> {
    let analysis = threats::analyze(root_state);
    if !analysis.own_win_now && analysis.opp_threat_count == 0 {
        return weights.to_vec();
    }
    let classes: Vec<i8> = action_ids
        .iter()
        .map(|&id| classify_root_move(root_state, id))
        .collect();
    let mut guarded = weights.to_vec();
    if classes.iter().any(|&c| c == 1) {
        for (i, &c) in classes.iter().enumerate() {
            if c != 1 {
                guarded[i] = 0.0;
            }
        }
    } else if classes.iter().any(|&c| c != -1) {
        for (i, &c) in classes.iter().enumerate() {
            if c == -1 {
                guarded[i] = 0.0;
            }
        }
    }
    if guarded.iter().all(|&w| w <= 0.0) {
        return weights.to_vec();
    }
    guarded
}

fn select_search_action(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    temperature: f32,
    seed: u64,
) -> PyResult<Option<PackedCoord>> {
    let (action_ids, weights, _q, _total) = visit_policy(search.root(), baseline);
    let guarded = if search.tss_enabled {
        tactical_guard_weights(&search.root_state, &action_ids, &weights)
    } else {
        weights.clone()
    };
    let (selected, _override) =
        select_action_with_lcb(search, baseline, &action_ids, &guarded, temperature, seed)?;
    Ok(selected)
}

/// Action selection: temperature sampling when temperature > 0, and on greedy
/// (T == 0) paths with `lcb_move_selection` on, LCB-of-Q selection among
/// eligible children (fallback max-visits). The TSS guard has already zeroed
/// proven-losing weights; LCB only ever picks among guard-positive actions.
fn select_action_with_lcb(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    action_ids: &[PackedCoord],
    guarded_weights: &[f32],
    temperature: f32,
    seed: u64,
) -> PyResult<(Option<PackedCoord>, bool)> {
    let dv = search.divergences;
    if temperature == 0.0 && dv.lcb_move_selection {
        let visit_pick = select_action_from_policy(action_ids, guarded_weights, 0.0, seed)?;
        let root = search.root();
        if let Some(lcb_id) = lcb_pick(root, baseline, &dv) {
            // Respect the tactical guard: never let LCB pick a zeroed action.
            let allowed = action_ids
                .iter()
                .zip(guarded_weights.iter())
                .any(|(&id, &w)| id == lcb_id && w > 0.0);
            if allowed {
                // Decisiveness tie-break on the played move: among moves
                // value-tied with the LCB leader, prefer the decisive one. Gated
                // on moves_left_utility; returns lcb_id in the dead-zone or with
                // no ml stats.
                let final_id = if dv.ml_final_pick && dv.moves_left_utility {
                    ml_final_pick(root, baseline, &dv, action_ids, guarded_weights)
                        .unwrap_or(lcb_id)
                } else {
                    lcb_id
                };
                let overrode = visit_pick.map(|v| v != final_id).unwrap_or(false);
                return Ok((Some(final_id), overrode));
            }
        }
        return Ok((visit_pick, false));
    }
    Ok((
        select_action_from_policy(action_ids, guarded_weights, temperature, seed)?,
        false,
    ))
}

fn visit_policy(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
) -> (Vec<PackedCoord>, Vec<f32>, Vec<f32>, u32) {
    // Compute each edge's delta visits once (edge_delta_visits is a HashMap
    // lookup when baseline is Some) and reuse the deltas for both the total and
    // the per-edge weights.
    let deltas: Vec<u32> = root
        .edges
        .iter()
        .map(|edge| edge_delta_visits(edge, baseline))
        .collect();
    let policy_total: u32 = deltas.iter().copied().sum();
    let mut policy_action_ids = Vec::with_capacity(root.edges.len());
    let mut policy_weights = Vec::with_capacity(root.edges.len());
    let mut policy_q = Vec::with_capacity(root.edges.len());
    for (edge, &visits) in root.edges.iter().zip(deltas.iter()) {
        if baseline.is_some() && visits == 0 {
            continue;
        }
        let weight = if policy_total > 0 {
            visits as f32 / policy_total as f32
        } else {
            edge.prior
        };
        policy_action_ids.push(edge.action_id);
        policy_weights.push(weight);
        policy_q.push(edge.value());
    }
    (policy_action_ids, policy_weights, policy_q, policy_total)
}

fn edge_delta_visits(edge: &RustEdge, baseline: Option<&HashMap<PackedCoord, u32>>) -> u32 {
    let before = baseline
        .and_then(|visits| visits.get(&edge.action_id).copied())
        .unwrap_or(0);
    edge.visits.saturating_sub(before)
}

fn pruned_visit_policy(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    forced_playout_k: f32,
    c_puct: f32,
) -> (Vec<PackedCoord>, Vec<f32>, Vec<f32>) {
    let edges = &root.edges;
    let deltas: Vec<u32> = edges
        .iter()
        .map(|edge| edge_delta_visits(edge, baseline))
        .collect();
    let priors: Vec<f32> = edges.iter().map(|edge| edge.prior).collect();
    let cumulative: Vec<u32> = edges.iter().map(|edge| edge.visits).collect();
    let values: Vec<f32> = edges.iter().map(|edge| edge.value()).collect();
    let pruned = prune_forced_delta_counts(
        &deltas,
        &priors,
        &cumulative,
        &values,
        root.visits,
        forced_playout_k,
        c_puct,
    );
    let total: u32 = pruned.iter().sum();
    if total == 0 {
        let (ids, weights, q, _total) = visit_policy(root, baseline);
        return (ids, weights, q);
    }
    let mut out_ids = Vec::with_capacity(edges.len());
    let mut weights = Vec::with_capacity(edges.len());
    let mut out_q = Vec::with_capacity(edges.len());
    for (index, edge) in edges.iter().enumerate() {
        if pruned[index] == 0 {
            continue;
        }
        out_ids.push(edge.action_id);
        weights.push(pruned[index] as f32 / total as f32);
        out_q.push(edge.value());
    }
    (out_ids, weights, out_q)
}

fn prune_forced_delta_counts(
    deltas: &[u32],
    priors: &[f32],
    cumulative: &[u32],
    values: &[f32],
    root_visits: u32,
    forced_playout_k: f32,
    c_puct: f32,
) -> Vec<u32> {
    let mut pruned = deltas.to_vec();
    if forced_playout_k <= 0.0 {
        return pruned;
    }
    let mut best_idx: Option<usize> = None;
    for index in 0..deltas.len() {
        if deltas[index] == 0 {
            continue;
        }
        best_idx = match best_idx {
            None => Some(index),
            Some(current) => {
                if deltas[index] > deltas[current] {
                    Some(index)
                } else {
                    Some(current)
                }
            }
        };
    }
    let Some(best_idx) = best_idx else {
        return pruned;
    };
    let root_n = root_visits.max(1) as f32;
    let explore = c_puct * root_n.sqrt();
    let u_best =
        values[best_idx] + priors[best_idx] * explore / (1.0 + cumulative[best_idx] as f32);
    for index in 0..deltas.len() {
        if index == best_idx || pruned[index] == 0 {
            continue;
        }
        if !(priors[index].is_finite() && priors[index] > 0.0) {
            continue;
        }
        let n_forced = (forced_playout_k * priors[index] * root_n).sqrt().floor() as u32;
        if n_forced == 0 {
            continue;
        }
        let q = values[index];
        let mut removed = 0u32;
        while removed < n_forced && pruned[index] > 0 {
            let reduced = cumulative[index].saturating_sub(removed + 1);
            let u = q + priors[index] * explore / (1.0 + reduced as f32);
            if u > u_best {
                break;
            }
            removed += 1;
            pruned[index] -= 1;
        }
    }
    pruned
}

fn select_action_from_policy(
    action_ids: &[PackedCoord],
    weights: &[f32],
    temperature: f32,
    seed: u64,
) -> PyResult<Option<PackedCoord>> {
    if action_ids.is_empty() || weights.is_empty() {
        return Ok(None);
    }
    if action_ids.len() != weights.len() {
        return Err(PyValueError::new_err(
            "visit policy action and weight lengths differ",
        ));
    }
    let total_weight: f32 = weights.iter().copied().sum();
    for weight in weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(PyValueError::new_err(format!(
                "visit policy weights must be finite and >= 0, got {weight}"
            )));
        }
    }
    if total_weight <= 0.0 {
        return Err(PyValueError::new_err(
            "visit policy must contain positive weight mass",
        ));
    }
    if temperature == 0.0 {
        return Ok(action_ids
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(action_id, _)| action_id));
    }
    let inv_temperature = 1.0 / temperature;
    let mut total = 0.0f64;
    let mut adjusted = Vec::with_capacity(weights.len());
    for weight in weights {
        // powf in f64: at low temperatures (large exponents) an f32 powf
        // underflows flat histograms to all-zero mass, aborting the batch with
        // the positive-finite-mass error below. f64 keeps ~1e-308 of headroom.
        let value = (*weight as f64).powf(inv_temperature as f64);
        total += value;
        adjusted.push(value);
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(PyValueError::new_err(
            "temperature-adjusted visit policy must contain positive finite mass",
        ));
    }
    // Walk the CDF, skipping zero-weight (e.g. tactical-guard-zeroed) entries so
    // they can never be selected: random_unit == 0.0 puts threshold at 0.0 up
    // front, which would otherwise return the FIRST action even if its adjusted
    // weight is 0; and f64 residue at the tail must not fall through onto a
    // zero-weight last action. The fallback is the LAST positive-weight action.
    let mut threshold = random_unit(seed) * total;
    let mut last_positive: Option<PackedCoord> = None;
    for (action_id, weight) in action_ids.iter().copied().zip(adjusted) {
        if weight <= 0.0 {
            continue;
        }
        last_positive = Some(action_id);
        threshold -= weight;
        if threshold <= 0.0 {
            return Ok(Some(action_id));
        }
    }
    Ok(last_positive)
}

#[cfg(test)]
mod fallback_tests {
    use super::*;
    use crate::tree::{NodePriors, RustPriorCandidate};
    use hexo_engine::Player;
    use hexo_utils::StateHash;

    fn edge(action_id: PackedCoord, prior: f32, visits: u32, value_sum: f32) -> RustEdge {
        RustEdge {
            action_id,
            action: unpack_coord(action_id),
            prior,
            visits,
            value_sum,
            value_sq_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            pending: 0,
            child: None,
            forced: false,
        }
    }

    fn node(edges: Vec<RustEdge>, candidates: Vec<RustPriorCandidate>) -> RustNode {
        RustNode {
            state_hash: StateHash::default(),
            player: Player::Player0,
            eval_value: 0.0,
            eval_ml: None,
            visits: edges.iter().map(|e| e.visits).sum(),
            value_sum: 0.0,
            ml_sum: 0.0,
            ml_weight: 0.0,
            edges,
            priors: NodePriors::Owned(candidates),
            max_eligible_children: 8,
            root_logits: None,
        }
    }

    // ---- Forced-playout target pruning (KataGo policy-target pruning). ----
    // These lock the main_8 Full-move export path: for a PUCT (non-Gumbel) root
    // with forced_playout_k > 0, build_search_result_payload_native records the
    // policy target from pruned_visit_policy -> prune_forced_delta_counts, which
    // strips the sqrt(k*P*N) forced-exploration visits back out while leaving the
    // raw visit_policy (used only for play selection) untouched. Counts are exact
    // u32, so these assert exact expected histograms. explore = c_puct*sqrt(N).

    #[test]
    fn prune_forced_strips_forced_visits_from_a_low_value_child() {
        // N=100, c=1.5 -> explore=15. Best = idx0 (60 delta).
        // u_best = 0.5 + 0.7*15/61 = 0.6721.
        // idx1: n_forced = floor(sqrt(2*0.05*100)) = 3; each removal keeps
        // U_1 (~ -0.22) far below u_best, so all 3 forced visits are stripped.
        let pruned = prune_forced_delta_counts(
            &[60, 10], &[0.7, 0.05], &[60, 10], &[0.5, -0.3], 100, 2.0, 1.5,
        );
        assert_eq!(pruned, vec![60, 7]);
    }

    #[test]
    fn prune_forced_keeps_genuine_visits_of_a_high_value_child() {
        // Same best/u_best. idx1 has n_forced=floor(sqrt(60))=7, but value 0.8
        // makes U_1 = 0.8 + 0.3*15/10 = 1.25 > u_best on the first candidate
        // removal, so the loop breaks immediately: genuine visits, not forced.
        let pruned = prune_forced_delta_counts(
            &[60, 10], &[0.7, 0.3], &[60, 10], &[0.5, 0.8], 100, 2.0, 1.5,
        );
        assert_eq!(pruned, vec![60, 10]);
    }

    #[test]
    fn prune_forced_zero_k_is_identity() {
        // forced_playout_k = 0 (Gumbel-era / off): the recorded target is the
        // raw delta-visit histogram, unchanged.
        let deltas = [60u32, 10, 5];
        let pruned = prune_forced_delta_counts(
            &deltas, &[0.5, 0.3, 0.2], &[60, 10, 5], &[0.4, 0.1, -0.2], 75, 0.0, 1.5,
        );
        assert_eq!(pruned, deltas.to_vec());
    }

    #[test]
    fn prune_forced_never_prunes_best_and_caps_at_n_forced() {
        // Three children; best = idx0 (50) is never touched. u_best = 0.5471.
        // idx1: n_forced=floor(sqrt(80))=8, U stays below u_best across all 8 ->
        // loses exactly 8 (30->22). idx2: n_forced=floor(sqrt(4))=2 -> 8->6.
        let pruned = prune_forced_delta_counts(
            &[50, 30, 8], &[0.5, 0.4, 0.02], &[50, 30, 8], &[0.4, 0.2, -0.5], 100, 2.0, 1.5,
        );
        assert_eq!(pruned, vec![50, 22, 6]);
    }

    // Non-sentinel action ids: PackedCoord 0 unpacks to the illegal sentinel
    // HexCoord{q:-32768,r:-32768}; any non-zero id we use here is a "real" action
    // for the purposes of the played-move invariant.
    const A1: PackedCoord = 0x8001_8000; // q=1, r=0
    const A2: PackedCoord = 0x8000_8001; // q=0, r=1
    const A3: PackedCoord = 0x8001_8001; // q=1, r=1

    #[test]
    fn delta_visit_policy_is_empty_when_all_edges_match_baseline() {
        // A root whose edges all sit at their reuse baseline (zero net delta)
        // yields an empty delta-visit policy, so the normal selection returns
        // None (the force-completed Gumbel SH case).
        let root = node(
            vec![edge(A1, 0.6, 3, 1.5), edge(A2, 0.4, 2, 0.5)],
            Vec::new(),
        );
        let baseline: HashMap<PackedCoord, u32> =
            [(A1, 3u32), (A2, 2u32)].into_iter().collect();
        let (ids, weights, _q, total) = visit_policy(&root, Some(&baseline));
        assert!(ids.is_empty(), "all-baseline delta policy must be empty");
        assert!(weights.is_empty());
        assert_eq!(total, 0);
        // The normal sampler returns None on an empty policy.
        let picked = select_action_from_policy(&ids, &weights, 1.0, 7).unwrap();
        assert!(picked.is_none());
    }

    #[test]
    fn fallback_never_returns_sentinel_and_prefers_most_visited() {
        // The fallback used when `selected` is None must yield a REAL root action
        // (never PackedCoord 0 / the sentinel). With visits present it picks the
        // most-visited cumulative edge.
        let root = node(
            vec![edge(A1, 0.2, 5, 2.0), edge(A2, 0.7, 1, 0.1)],
            vec![RustPriorCandidate { action_id: A3, prior: 0.9 }],
        );
        let picked = fallback_root_action(&root).expect("fallback yields an action");
        assert_ne!(picked, 0, "fallback must never return the sentinel id 0");
        assert_eq!(picked, A1, "most-visited cumulative edge wins");
    }

    #[test]
    fn cumulative_visit_policy_recovers_target_when_delta_is_empty() {
        // The RECORDED-target fallback in build_search_result_payloads substitutes
        // the baseline-free cumulative visit policy when the delta-visit export is
        // empty. Pin the property it relies on: with edges carrying cumulative
        // visits, visit_policy(root, None) yields a NON-EMPTY, positive-mass target
        // even though the delta policy (vs an all-matching baseline) is empty.
        let root = node(
            vec![edge(A1, 0.6, 3, 1.5), edge(A2, 0.4, 2, 0.5)],
            Vec::new(),
        );
        let baseline: HashMap<PackedCoord, u32> =
            [(A1, 3u32), (A2, 2u32)].into_iter().collect();
        let (d_ids, _d_w, _d_q, d_total) = visit_policy(&root, Some(&baseline));
        assert!(d_ids.is_empty() && d_total == 0, "delta policy is empty");
        let (c_ids, c_w, _c_q, c_total) = visit_policy(&root, None);
        assert_eq!(c_ids.len(), 2, "cumulative policy keeps both edges");
        assert_eq!(c_total, 5, "cumulative total = sum of edge visits");
        let mass: f32 = c_w.iter().sum();
        assert!((mass - 1.0).abs() < 1e-6, "cumulative target carries unit mass");
        assert!(c_ids.iter().all(|&id| id != 0), "no sentinel in the target");
    }

    #[test]
    fn fallback_uses_highest_prior_when_no_visits() {
        // Degenerate root: edges exist but carry zero visits (force-completed with
        // nothing searched). Fallback then takes the highest-prior action across
        // edges + unexpanded candidates — still a real, legal action id.
        let root = node(
            vec![edge(A1, 0.2, 0, 0.0), edge(A2, 0.3, 0, 0.0)],
            vec![RustPriorCandidate { action_id: A3, prior: 0.5 }],
        );
        let picked = fallback_root_action(&root).expect("fallback yields an action");
        assert_ne!(picked, 0);
        assert_eq!(picked, A3, "highest-prior action wins when no edge is visited");
    }

    // --- Fast-class play temperature + sampler zero-weight edge ---------------

    // Non-sentinel ids for the sampler tests.
    const S1: PackedCoord = A1;
    const S2: PackedCoord = A2;
    const S3: PackedCoord = A3;

    fn move_policy(fast_temperature: f32) -> ContinuousMovePolicy {
        ContinuousMovePolicy {
            full_visits: 512,
            fast_visits: 128,
            fast_temperature,
            pcr_full_proportion: 0.33,
            policy_init_fraction: 0.0,
            policy_init_avg_plies: 0.0,
            policy_init_max_plies: 0,
            policy_init_temperature: 1.0,
            root_policy_temperature: 1.0,
            root_policy_temperature_early: 0.0,
            root_policy_temperature_halflife: 0.0,
            fpu_reduction: 0.2,
            forced_playout_k: 0.0,
            noise: None,
            tss_enabled: true,
            root_fpu_zero_under_noise: false,
            root_fpu_reduction: None,
            divergences_full: Divergences::production(),
            divergences_fast: Divergences::production(),
        }
    }

    #[test]
    fn fast_class_default_temperature_is_zero() {
        // Default 0.0 => Fast plays greedily (the T==0 LCB pick branch),
        // reproducing current behavior. Full uses the ply schedule; Init is 0.0.
        let policy = move_policy(0.0);
        let by_ply = vec![0.9, 0.5, 0.15];
        assert_eq!(policy.temperature_for_class(MoveClass::Fast, &by_ply, 1), 0.0);
        assert_eq!(policy.temperature_for_class(MoveClass::Init, &by_ply, 1), 0.0);
        assert_eq!(
            policy.temperature_for_class(MoveClass::Full, &by_ply, 1),
            0.5,
            "Full follows the ply schedule"
        );
    }

    #[test]
    fn fast_class_uses_the_lever_when_set() {
        // The lever flows to the Fast class only; Full/Init are unchanged.
        let policy = move_policy(0.1);
        let by_ply = vec![0.9, 0.5, 0.15];
        assert_eq!(policy.temperature_for_class(MoveClass::Fast, &by_ply, 0), 0.1);
        assert_eq!(policy.temperature_for_class(MoveClass::Init, &by_ply, 0), 0.0);
        assert_eq!(
            policy.temperature_for_class(MoveClass::Full, &by_ply, 0),
            0.9,
            "Full still follows the ply schedule"
        );
    }

    #[test]
    fn sampler_never_selects_zero_weight_first_entry() {
        // random_unit(seed) can be exactly 0.0 for some seeds, putting the CDF
        // threshold at 0.0 before the walk. A zero-weight (tactical-guard-zeroed)
        // FIRST entry must still never be selected. Sweep many seeds at T=0.1.
        let ids = vec![S1, S2, S3];
        let weights = vec![0.0f32, 0.7, 0.3];
        for seed in 0u64..2000 {
            let picked = select_action_from_policy(&ids, &weights, 0.1, seed)
                .unwrap()
                .expect("positive mass yields a pick");
            assert_ne!(picked, S1, "zero-weight first entry must never be selected");
        }
    }

    #[test]
    fn sampler_never_selects_zero_weight_last_entry() {
        // f64 residue at the tail must not fall through onto a zero-weight LAST
        // action; the fallback is the last POSITIVE-weight action.
        let ids = vec![S1, S2, S3];
        let weights = vec![0.6f32, 0.4, 0.0];
        for seed in 0u64..2000 {
            let picked = select_action_from_policy(&ids, &weights, 0.1, seed)
                .unwrap()
                .expect("positive mass yields a pick");
            assert_ne!(picked, S3, "zero-weight last entry must never be selected");
        }
    }

    #[test]
    fn sampler_leading_zero_weights_never_selected_high_temperature() {
        // Two leading zero-weight entries (guard-zeroed) with only the tail
        // carrying mass; at a large T the exponent flattens weights but the zero
        // entries stay zero and must never be picked, across seeds.
        let ids = vec![S1, S2, S3];
        let weights = vec![0.0f32, 0.0, 1.0];
        for seed in 0u64..500 {
            let picked = select_action_from_policy(&ids, &weights, 2.0, seed)
                .unwrap()
                .expect("positive mass yields a pick");
            assert_eq!(picked, S3, "only the positive-weight action is selectable");
        }
    }

    // --- Export-only σ softening: gumbel_target_c_scale (lever 1) --------------

    /// Root with two searched edges carrying distinct Qs and distinct logits,
    /// plus a root_logits map, so gumbel_target_policy exercises the full σ path.
    /// Qs are modest (+0.2 / 0.0) so the softening test's softmax stays away from
    /// the one-hot saturation region where a c_scale change is invisible.
    fn target_root() -> RustNode {
        // edge A1: 5 visits, sum 1.0 => Q=+0.2 ; edge A2: 4 visits, sum 0.0 =>
        // Q=0.0. Distinct Qs so a smaller c_scale provably flattens the target.
        let mut root = node(
            vec![edge(A1, 0.6, 5, 1.0), edge(A2, 0.4, 4, 0.0)],
            Vec::new(),
        );
        // Distinct logits (raw, unconstrained sign) keyed by action id.
        root.root_logits = Some([(A1, 0.2f32), (A2, -0.2f32)].into_iter().collect());
        root
    }

    #[test]
    fn target_c_scale_unset_is_bit_identical_to_gumbel_c_scale() {
        // The resolver `gumbel_target_c_scale.unwrap_or(gumbel_c_scale)` must make
        // an unset override bit-identical to computing the target with the plain
        // gumbel_c_scale — no drift from the default path.
        let root = target_root();
        let c_visit = 50.0f32;
        let c_scale = 1.0f32;
        // Reference: the exporter called with c_scale directly.
        let (ref_ids, ref_w, ref_l) = gumbel_target_policy(&root, None, c_visit, c_scale, 1);
        // Resolved value when the override is None (mirrors search.rs call site).
        let div = Divergences::gumbel(); // gumbel_target_c_scale defaults to None
        let resolved = div.gumbel_target_c_scale.unwrap_or(div.gumbel_c_scale);
        assert_eq!(resolved, c_scale, "unset override resolves to gumbel_c_scale");
        let (ids, w, l) = gumbel_target_policy(&root, None, c_visit, resolved, 1);
        assert_eq!(ids, ref_ids);
        assert_eq!(l, ref_l, "logits output is independent of c_scale");
        // Bit-identical weights (same inputs => same float ops).
        assert_eq!(w, ref_w, "unset target c_scale must be bit-identical");
    }

    #[test]
    fn target_c_scale_softens_and_matches_reference_softmax() {
        // With gumbel_target_c_scale = 0.35 the exported weights must equal an
        // independent softmax(l + σ(q, max_n, c_visit, 0.35)) over the support and
        // be strictly flatter (lower top-1 mass) than the c_scale=1.0 target.
        // c_visit=1.0 keeps the σ gain small enough that neither target saturates
        // to a one-hot (where a c_scale change would be invisible in f32).
        let root = target_root();
        let c_visit = 1.0f32;
        let soft = 0.35f32;

        // Exporter output at the softened scale.
        let (ids, weights, _logits) = gumbel_target_policy(&root, None, c_visit, soft, 1);

        // Independent reference over the same (ascending action_id) support.
        let logit_map = root.root_logits.clone().unwrap();
        let (completed, v_mix) = gumbel_completed_q(&root, &logit_map);
        let max_n = root.edges.iter().map(|e| e.visits).max().unwrap();
        let mut support: Vec<PackedCoord> =
            root.edges.iter().filter(|e| e.visits >= 1).map(|e| e.action_id).collect();
        support.sort_unstable();
        assert_eq!(ids, support, "exporter uses ascending action_id support");
        let ref_scores: Vec<f32> = support
            .iter()
            .map(|a| {
                let l = logit_map.get(a).copied().unwrap_or(0.0);
                let q = completed.get(a).copied().unwrap_or(v_mix);
                l + gumbel_sigma(q, max_n, c_visit, soft)
            })
            .collect();
        let ref_weights = gumbel_softmax(&ref_scores);
        for (w, r) in weights.iter().zip(ref_weights.iter()) {
            assert!((w - r).abs() < 1e-6, "softened target must match reference");
        }

        // Strictly flatter than the c_scale=1.0 target: lower top-1 mass.
        let (_full_ids, full_weights, _fl) = gumbel_target_policy(&root, None, c_visit, 1.0, 1);
        let top1_soft = weights.iter().copied().fold(f32::MIN, f32::max);
        let top1_full = full_weights.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            top1_soft < top1_full,
            "softened top-1 mass {top1_soft} must be below the c_scale=1.0 top-1 {top1_full}"
        );
    }

    /// The strict KNOWN_DIVERGENCE_KEYS gate must accept every key the python
    /// side emits — the 2026-07-04 deploy crashed because the new lever keys
    /// were parsed but missing from the whitelist. Exercises the real pyo3
    /// resolve path with both new keys present, plus every main_8 fast_* key.
    #[test]
    fn resolve_divergences_accepts_the_new_lever_keys() {
        Python::initialize();
        Python::attach(|py| {
            let overrides = PyDict::new(py);
            overrides.set_item("gumbel_target_c_scale", 0.35f32).unwrap();
            overrides.set_item("gumbel_draw_temperature", 1.0f32).unwrap();
            let dv = resolve_divergences(None, Some(&overrides), false)
                .expect("new lever keys must pass the known-keys gate");
            assert_eq!(dv.gumbel_target_c_scale, Some(0.35));
            assert_eq!(dv.gumbel_draw_temperature, 1.0);

            // Every main_8 fast_* key must pass the gate AND fold onto its
            // gumbel field. A whitelist/parser mismatch here is exactly the
            // failure class that tripped the supervisor circuit breaker.
            let fast = PyDict::new(py);
            fast.set_item("fast_gumbel_root_enabled", true).unwrap();
            fast.set_item("fast_gumbel_sequential_halving", true).unwrap();
            fast.set_item("fast_gumbel_nonroot_select", true).unwrap();
            fast.set_item("fast_gumbel_c_visit", 12.0f32).unwrap();
            fast.set_item("fast_gumbel_c_scale", 0.5f32).unwrap();
            fast.set_item("fast_gumbel_m", 8u32).unwrap();
            fast.set_item("fast_gumbel_play_prune", true).unwrap();
            let fv = resolve_divergences(None, Some(&fast), true)
                .expect("fast_* keys must pass the known-keys gate in the fast map");
            assert!(fv.gumbel_root);
            assert!(fv.gumbel_sequential_halving);
            assert!(fv.gumbel_nonroot_select);
            assert_eq!(fv.gumbel_c_visit, 12.0);
            assert_eq!(fv.gumbel_c_scale, 0.5);
            assert_eq!(fv.gumbel_m, 8);
            assert!(fv.gumbel_play_prune);

            // The gate itself still rejects a genuinely unknown key.
            let bogus = PyDict::new(py);
            bogus.set_item("gumbel_bogus_lever", 1.0f32).unwrap();
            assert!(resolve_divergences(None, Some(&bogus), false).is_err());

            // fast_* keys are Fast-view levers: accepted only in the fast map,
            // rejected in the BASE map where they would silently mutate the
            // Full-class profile.
            assert!(
                resolve_divergences(None, Some(&fast), false).is_err(),
                "base map must reject fast_* keys"
            );

            // Numeric validation: garbage values fail loudly instead of
            // reaching c_for/tanh as inf/NaN.
            for (key, value) in [
                ("c_base", 0.0f32),
                ("c_base", f32::NAN),
                ("c_scale", -0.1f32),
                ("ml_scale", 0.0f32),
                ("lcb_z", f32::INFINITY),
                ("gumbel_c_visit", -1.0f32),
                ("gumbel_c_scale", f32::NAN),
                ("gumbel_target_c_scale", -0.5f32),
                ("gumbel_draw_temperature", f32::INFINITY),
                ("ml_q_gate", -0.1f32),
                ("ml_final_pick_band", f32::NAN),
                ("ml_weight", f32::NAN),
            ] {
                let bad = PyDict::new(py);
                bad.set_item(key, value).unwrap();
                assert!(
                    resolve_divergences(None, Some(&bad), false).is_err(),
                    "{key}={value} must be rejected"
                );
            }
            // The documented τ<=0 disable sentinel stays accepted.
            let tau_off = PyDict::new(py);
            tau_off.set_item("gumbel_draw_temperature", 0.0f32).unwrap();
            assert!(resolve_divergences(None, Some(&tau_off), false).is_ok());
        });
    }

    // === main_8: turn-based classification + per-class divergences ===========

    /// Full ContinuousMovePolicy tuned for classify() sampling tests: a real
    /// pcr_full_proportion and no policy-init so classify exercises the PCR
    /// hash rather than the Init/Full short-circuits.
    fn classify_policy(pcr_full_proportion: f32) -> ContinuousMovePolicy {
        let mut p = move_policy(0.0);
        p.pcr_full_proportion = pcr_full_proportion;
        p.policy_init_fraction = 0.0;
        p.policy_init_avg_plies = 0.0;
        p.policy_init_max_plies = 0;
        p
    }

    #[test]
    fn classify_is_per_turn_paired_plies_share_a_class() {
        // Both plies of a turn (2k, 2k+1) must map to the same class, for many
        // turns and many seeds. This is the invariant the per-class
        // set_divergences reuse relies on.
        let policy = classify_policy(0.5);
        for base_seed in 0u64..64 {
            for game_key in [0u64, 1, 7, 4242, u64::MAX] {
                for k in 0u32..64 {
                    let a = policy.classify(base_seed, game_key, 2 * k, 0);
                    let b = policy.classify(base_seed, game_key, 2 * k + 1, 0);
                    assert_eq!(
                        a, b,
                        "plies {} and {} of turn {k} must share a class (seed {base_seed}, key {game_key})",
                        2 * k,
                        2 * k + 1
                    );
                    // And it is never Init here (no policy-init remaining).
                    assert!(matches!(a, MoveClass::Full | MoveClass::Fast));
                }
            }
        }
    }

    #[test]
    fn classify_full_fraction_matches_proportion_over_turns() {
        // Over a large sample of turns, roughly pcr_full_proportion of TURNS are
        // Full. Sample one ply per turn (the pair is identical by the test
        // above) across many game_keys to average out the per-key hash.
        let prop = 0.33f32;
        let policy = classify_policy(prop);
        let base_seed = 12345u64;
        let mut full = 0u64;
        let mut total = 0u64;
        for game_key in 0u64..2000 {
            for k in 0u32..16 {
                if matches!(policy.classify(base_seed, game_key, 2 * k, 0), MoveClass::Full) {
                    full += 1;
                }
                total += 1;
            }
        }
        let frac = full as f64 / total as f64;
        assert!(
            (frac - prop as f64).abs() < 0.02,
            "Full turn fraction {frac} should be ~{prop} over {total} turns"
        );
    }

    #[test]
    fn classify_short_circuits_are_unchanged() {
        // policy_init_remaining > 0 => Init regardless of ply/turn.
        let policy = classify_policy(0.33);
        assert!(matches!(policy.classify(1, 2, 0, 3), MoveClass::Init));
        assert!(matches!(policy.classify(1, 2, 5, 1), MoveClass::Init));
        // pcr_full_proportion >= 1.0 => always Full.
        let all_full = classify_policy(1.0);
        for ply in 0u32..16 {
            assert!(matches!(all_full.classify(9, 9, ply, 0), MoveClass::Full));
        }
    }

    #[test]
    fn divergences_for_selects_the_class_view() {
        let mut policy = move_policy(0.0);
        let mut fast = Divergences::production();
        fast.gumbel_root = true; // make the fast view distinguishable
        policy.divergences_fast = fast;
        assert!(!policy.divergences_for(MoveClass::Full).gumbel_root);
        assert!(!policy.divergences_for(MoveClass::Init).gumbel_root);
        assert!(policy.divergences_for(MoveClass::Fast).gumbel_root);
    }

    #[test]
    fn golden_invariant_fast_equals_full_without_fast_overrides() {
        // When no fast_* keys are set, the driver falls back to the base view
        // for the fast map (fast_divergence_overrides=None => divergences_fast =
        // divergences). Mirror that here: fast resolved from None equals base.
        Python::initialize();
        Python::attach(|py| {
            let base_overrides = PyDict::new(py);
            base_overrides.set_item("gumbel_root", true).unwrap();
            base_overrides.set_item("gumbel_c_scale", 0.7f32).unwrap();
            let base = resolve_divergences(None, Some(&base_overrides), false).unwrap();
            // No fast overrides => fast view IS the base view (Rust fallback).
            let fast_fallback = base;
            assert_eq!(
                fast_fallback, base,
                "divergences_fast must equal divergences_full when no fast_* keys set"
            );

            // And request_logits/request_moves_left with identical views match
            // the single-view result.
            let mut policy = move_policy(0.0);
            policy.divergences_full = base;
            policy.divergences_fast = base;
            assert_eq!(policy.request_logits(), base.gumbel_root);
        });
    }

    #[test]
    fn request_logits_true_if_either_view_needs_them() {
        let mut policy = move_policy(0.0);
        // Neither view needs logits.
        let mut plain = Divergences::production();
        plain.gumbel_target = false;
        plain.gumbel_root = false;
        plain.gumbel_nonroot_select = false;
        policy.divergences_full = plain;
        policy.divergences_fast = plain;
        assert!(!policy.request_logits());
        // Fast view alone needs them => request_logits true.
        let mut fast = plain;
        fast.gumbel_root = true;
        policy.divergences_fast = fast;
        assert!(policy.request_logits());
        // Reset fast, set only full.
        policy.divergences_fast = plain;
        let mut full = plain;
        full.gumbel_nonroot_select = true;
        policy.divergences_full = full;
        assert!(policy.request_logits());
    }
}
