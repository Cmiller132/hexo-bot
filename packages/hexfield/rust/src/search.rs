//! hexfield search drivers: lockstep batched search and the continuous per-slot
//! scheduler.
//!
//! - The root-policy-temperature schedule defaults OFF (1.0 / no ramp).
//! - The optional search divergences (LCB greedy selection, early-stop by move
//!   class, moves-left utility) default ON and are forced off by
//!   `search_parity_mode`.
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
use crate::payload::{evaluate_state_refs_cached, finish_eval_cached, submit_eval_cached};
use crate::state::states_from_py_states;
use crate::threats_shared as threats;
use crate::tree::{
    gumbel_completed_q, gumbel_sigma, gumbel_softmax, random_unit, terminal_value, Divergences,
    RustEdge, RustLeaf, RustNode, RustSearch, Widening,
};

pub const ACTIVE_ROOT_LIMIT: usize = 512;

/// Stream id 0 is reserved (formerly the root-noise stream); the numbering is
/// pinned by the mix_seed golden-vector contract, so downstream streams keep
/// their historical ids.
pub const SEED_STREAM_ROOT_NOISE: u64 = 0;
pub const SEED_STREAM_MOVE_SELECT: u64 = 1;
pub const SEED_STREAM_PCR: u64 = 2;
pub const SEED_STREAM_POLICY_INIT_SELECT: u64 = 3;
pub const SEED_STREAM_POLICY_INIT_COUNT: u64 = 4;
pub const SEED_STREAM_POLICY_INIT_SAMPLE: u64 = 5;
/// Gumbel-Top-k root draws. Dedicated stream (id 6), independent of the other
/// streams.
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
    tss_enabled: bool,
    /// Root FPU reduction. When Some it applies to every move class; when None
    /// the root falls back to the interior `fpu_reduction`.
    root_fpu_reduction: Option<f32>,
    divergences: Divergences,
}

impl ContinuousMovePolicy {
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
        let unit = random_unit(mix_seed(base_seed, game_key, ply, SEED_STREAM_PCR));
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

    fn root_fpu_for(&self, _class: MoveClass) -> f32 {
        // `root_fpu_reduction`, when set, applies to every move class; otherwise
        // the root falls back to the interior `fpu_reduction`.
        self.root_fpu_reduction.unwrap_or(self.fpu_reduction)
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

    fn request_moves_left(&self) -> bool {
        self.divergences.moves_left_utility
    }

    /// Whether the evaluator must emit raw pre-softmax policy logits. True when
    /// any Gumbel mechanism that reads `logits(a)` is enabled: the improved
    /// target, the Gumbel-Top-k root sampler, or the non-root selection.
    fn request_logits(&self) -> bool {
        self.divergences.gumbel_target
            || self.divergences.gumbel_root
            || self.divergences.gumbel_nonroot_select
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
    #[pyo3(signature = (game_keys, states, visits, c_puct, temperature, seed, evaluator, virtual_batch_size=None, active_root_limit=None, root_policy_temperature=None, fpu_reduction=None, virtual_loss=None, widening_policy_mass=None, widening_max_children=None, widening_min_children=None, move_temperatures=None, root_policy_temperatures=None, tss_enabled=None, root_fpu_reduction=None, search_parity_mode=None, divergence_overrides=None, debug_no_advance=None))]
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
        root_policy_temperature: Option<f32>,
        fpu_reduction: Option<f32>,
        virtual_loss: Option<f32>,
        widening_policy_mass: Option<f32>,
        widening_max_children: Option<u32>,
        widening_min_children: Option<u32>,
        move_temperatures: Option<Vec<f32>>,
        root_policy_temperatures: Option<Vec<f32>>,
        tss_enabled: Option<bool>,
        // When provided, the root FPU reduction; else the interior fpu_reduction.
        root_fpu_reduction: Option<f32>,
        search_parity_mode: Option<bool>,
        divergence_overrides: Option<&Bound<'_, PyDict>>,
        debug_no_advance: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        validate_search_inputs(visits, c_puct, temperature)?;
        let divergences = resolve_divergences(search_parity_mode, divergence_overrides)?;
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
        let tss_enabled = tss_enabled.unwrap_or(true);
        // Root FPU reduction. If `root_fpu_reduction` is given explicitly it
        // applies to the root; otherwise the root uses the interior
        // `fpu_reduction`.
        let root_fpu_reduction = match root_fpu_reduction {
            Some(value) => validate_nonnegative_f32("root_fpu_reduction", value)?,
            None => fpu_reduction,
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
                    search.set_root_fpu_reduction(root_fpu_reduction);
                    search.set_tss_enabled(tss_enabled);
                    search.set_divergences(divergences);
                    search.apply_root_policy_temperature(root_policy_temps[index]);
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
                    widening,
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
        )?;

        let no_advance = debug_no_advance.unwrap_or(false);
        for ((game_key, mut search), selected) in game_keys
            .into_iter()
            .zip(searches.into_iter())
            .zip(selected_actions.into_iter())
        {
            if no_advance {
                // Forensics only: store the searched tree as-is for debug_dump.
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
    #[pyo3(signature = (game_keys, states, evaluator, on_move, visits, c_puct, base_seed, virtual_batch_size, flush_target, active_root_limit, temperature_by_ply, root_policy_temperature=None, fpu_reduction=None, virtual_loss=None, widening_policy_mass=None, widening_max_children=None, widening_min_children=None, root_policy_temperature_early=None, root_policy_temperature_halflife=None, pcr_full_proportion=None, pcr_fast_visits=None, pcr_fast_temperature=None, policy_init_fraction=None, policy_init_avg_plies=None, policy_init_max_plies=None, policy_init_temperature=None, tss_enabled=None, root_fpu_reduction=None, search_parity_mode=None, divergence_overrides=None))]
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
        root_policy_temperature: Option<f32>,
        fpu_reduction: Option<f32>,
        virtual_loss: Option<f32>,
        widening_policy_mass: Option<f32>,
        widening_max_children: Option<u32>,
        widening_min_children: Option<u32>,
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
        // Root FPU reduction; else the interior fpu_reduction.
        root_fpu_reduction: Option<f32>,
        search_parity_mode: Option<bool>,
        divergence_overrides: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        validate_search_inputs(visits, c_puct, 0.0)?;
        let divergences = resolve_divergences(search_parity_mode, divergence_overrides)?;
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
            tss_enabled: tss_enabled.unwrap_or(true),
            // Root FPU reduction (validated >= 0 when provided).
            root_fpu_reduction: match root_fpu_reduction {
                Some(value) => Some(validate_nonnegative_f32("root_fpu_reduction", value)?),
                None => None,
            },
            divergences,
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
                    search.set_additional_visits(move_policy.visits_for(move_class));
                    search.set_root_fpu_reduction(move_policy.root_fpu_for(move_class));
                    search.set_tss_enabled(move_policy.tss_enabled);
                    search.set_divergences(divergences);
                    search.apply_root_policy_temperature(move_policy.root_temp_for(move_class, 0));
                    // (Re)build the Gumbel-Top-k candidate set + SH schedule on
                    // a reused root. init_gumbel_root clears any prior state
                    // first; for a non-Full reuse it is cleared so the normal
                    // PUCT root runs.
                    if divergences.gumbel_root && matches!(move_class, MoveClass::Full) {
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
        let async_eval = std::env::var("HEXFIELD_ASYNC_EVAL").is_ok();
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
                    let prefetch_result = py.detach(|| {
                        select_continuous_pass(&mut slots, c_puct, leaf_batch_per_root, virtual_loss)
                    })?;
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
                    let prefetch_result =
                        select_continuous_pass(&mut slots, c_puct, leaf_batch_per_root, virtual_loss)?;
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

            let t_complete = std::time::Instant::now();
            let mut moves_decided = complete_continuous_slots(
                py,
                on_move,
                &mut slots,
                &move_policy,
                &temperature_by_ply,
                base_seed,
                &mut queue,
                &mut stats,
                false,
            )?;
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
        }

        self.finish_continuous_stats(py, stats, &evaluation_stats)
    }
}

// Internal (non-`#[pymethods]`) scheduler helpers. These take native Rust types
// (`Widening`, `Divergences`, `&mut [ContinuousSlot]`) that pyo3 cannot expose,
// so they MUST live outside the `#[pymethods]` block above.
impl HexfieldMctsSession {
    /// Build the `run_continuous` stats dict. Pure GIL-held conversion of the
    /// accumulated counters.
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
        // Prefetch select with the current batch still pending (pre-backup
        // tree state). `pending_leaves` carries −virtual_loss on the trees of
        // the searches it touches, so the SH barrier is blocked for exactly
        // those searches (their round ranking would read contaminated stats).
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
    divergences: Divergences,
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
            let mut search = RustSearch::new(
                state,
                &**evaluation,
                move_policy.visits_for(move_class),
                move_policy.fpu_reduction,
                move_policy.root_fpu_for(move_class),
                move_policy.root_temp_for(move_class, slot.ply),
                widening,
                move_policy.tss_enabled,
                divergences,
            )?;
            if search.root_edges_empty() {
                return Err(PyValueError::new_err(
                    "hexfield continuous MCTS root has no legal actions",
                ));
            }
            // Build the Gumbel-Top-k candidate set + SH schedule for Full moves
            // when gumbel_root is on. No-op otherwise (the search keeps the
            // normal PUCT root). budget = the move's visits.
            if divergences.gumbel_root && matches!(move_class, MoveClass::Full) {
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
    // index), with no shared mutable state.
    let prepared: Vec<Option<PreparedMove>> = py.detach(|| {
        let prepare = |slot: &ContinuousSlot| -> PyResult<Option<PreparedMove>> {
            if !matches!(slot.phase, ContinuousPhase::Active) {
                return Ok(None);
            }
            let move_class = slot.move_class;
            let in_flight = slot.in_flight;
            let (complete, early) = slot
                .search
                .as_ref()
                .map(|search| {
                    let normal = continuous_completion_ready(
                        search.completed_visits,
                        search.target_visits,
                        in_flight,
                    );
                    if normal {
                        return (true, false);
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
                        return (true, true);
                    }
                    // Fast moves stop unrestricted; recorded Full roots keep the
                    // visit floor.
                    let early = early_stop_ready(
                        search,
                        Some(&slot.baseline),
                        matches!(move_class, MoveClass::Full),
                        in_flight,
                    );
                    (early, early)
                })
                .unwrap_or((false, false));
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
                early_remaining_visits,
                payload,
                action_id,
                init_override,
            }))
        };

        slots
            .par_iter()
            .map(prepare)
            .collect::<PyResult<Vec<_>>>()
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
            stats.early_stop_visits_saved += prepared.early_remaining_visits as u64;
            match move_class {
                MoveClass::Full => stats.early_stops_full += 1,
                _ => stats.early_stops_fast += 1,
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
                        search.set_additional_visits(move_policy.visits_for(next_class));
                        search.set_root_fpu_reduction(move_policy.root_fpu_for(next_class));
                        search.set_tss_enabled(move_policy.tss_enabled);
                        search
                            .apply_root_policy_temperature(move_policy.root_temp_for(next_class, next_ply));
                        // (Re)build the Gumbel-Top-k candidate set + SH schedule
                        // for the promoted root, mirroring the epoch-entry reuse
                        // path. Without this the previous move's finished SH
                        // state (survivors/round caps keyed to the old root's
                        // actions) persists onto the new root, and the slot
                        // either hammers a stale survivor or stalls until the
                        // force-stuck safety net finalizes the move with zero
                        // new visits. Non-Full moves clear the state so the
                        // normal PUCT root runs.
                        if move_policy.divergences.gumbel_root
                            && matches!(next_class, MoveClass::Full)
                        {
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
    "moves_left_utility",
    "ml_weight",
    "ml_scale",
    "ml_q_gate",
    "ml_two_sided",
    "ml_final_pick",
    "ml_final_pick_band",
    "lcb_z",
    "nucleus_f64",
    "new_child_fpu",
    "lazy_widening",
    "clean_root_prior_cache",
    "gumbel_target",
    "gumbel_root",
    "gumbel_sequential_halving",
    "gumbel_nonroot_select",
    "gumbel_c_visit",
    "gumbel_c_scale",
    "gumbel_m",
    "gumbel_target_min_visits",
    "gumbel_play_prune",
];

fn resolve_divergences(
    search_parity_mode: Option<bool>,
    overrides: Option<&Bound<'_, PyDict>>,
) -> PyResult<Divergences> {
    if let Some(overrides) = overrides {
        for key in overrides.keys() {
            let key: String = key.extract()?;
            if !KNOWN_DIVERGENCE_KEYS.contains(&key.as_str()) {
                return Err(PyValueError::new_err(format!(
                    "unknown divergence override key {key:?}; known keys: {KNOWN_DIVERGENCE_KEYS:?}"
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
        if let Some(v) = overrides.get_item("moves_left_utility")? {
            dv.moves_left_utility = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_weight")? {
            dv.ml_weight = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_scale")? {
            dv.ml_scale = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_q_gate")? {
            dv.ml_q_gate = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_two_sided")? {
            dv.ml_two_sided = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_final_pick")? {
            dv.ml_final_pick = v.extract()?;
        }
        if let Some(v) = overrides.get_item("ml_final_pick_band")? {
            dv.ml_final_pick_band = v.extract()?;
        }
        if let Some(v) = overrides.get_item("lcb_z")? {
            dv.lcb_z = v.extract()?;
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
            dv.gumbel_c_visit = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_c_scale")? {
            dv.gumbel_c_scale = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_m")? {
            dv.gumbel_m = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_target_min_visits")? {
            dv.gumbel_target_min_visits = v.extract()?;
        }
        if let Some(v) = overrides.get_item("gumbel_play_prune")? {
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
) -> PyResult<PayloadNative> {
    let root = search.root();
    let (policy_action_ids, policy_weights, _policy_q, policy_total) = visit_policy(root, baseline);
    let (mut export_action_ids, mut export_weights, mut export_q) = {
        let (ids, w, q, _t) = visit_policy(root, baseline);
        (ids, w, q)
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
        let (gumbel_ids, gumbel_weights, gumbel_logits) = gumbel_target_policy(
            root,
            baseline,
            div.gumbel_c_visit,
            div.gumbel_c_scale,
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
) -> PyResult<Py<PyAny>> {
    let results = PyList::empty(py);
    for (index, search) in searches.iter().enumerate() {
        let baseline = baselines.and_then(|items| items.get(index));
        let native = build_search_result_payload_native(
            search,
            baseline,
            temperatures[index],
            seed.wrapping_add(index as u64),
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
        let value = weight.powf(inv_temperature) as f64;
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
            tss_enabled: true,
            root_fpu_reduction: None,
            divergences: Divergences::production(),
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
}
