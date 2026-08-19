//! tss_py.rs — the Threat-Space Search read surface for Python search drivers
//! that own their own tree (the MantisNet Gumbel family, in
//! `apps/showcase/server/showcase/families/mantisnet_family.py`).
//!
//! The hexfield_eq tree calls the TSS producers in Rust, inline. A Python
//! driver cannot, so this module marshals THE SAME functions across the pyo3
//! boundary: λ¹ (`threats_shared::analyze`), the root-move classifier
//! (`search::classify_root_move`), and the verified deep solve
//! (`tree::tss_solve_verified`). No threat semantics are defined here — this
//! file is marshalling only, so a Python caller and the hexfield tree cannot
//! drift apart.
//!
//! One [`TssProbe`] holds the search ROOT, cloned once out of a showcase
//! `hexo_engine` state through the engine's state capsule. Every query names a
//! position by its PLACEMENT PATH from that root and the probe replays the
//! path into a fresh clone, so the queried state carries the real placement
//! history — and therefore the real `TurnPhase::SecondStone { first }`, which
//! the solver reads when it generates attack children. A path that is not
//! legal from the root is an error, never a silently truncated replay.
//!
//! Solver configuration is the live main_5 serve configuration
//! (`configs/hexfield_eq_main_5.toml`): goal Both, dual pass ON, leaf j2near
//! OFF, Group-2 OFF, zero loss reserve, zones OFF, unbounded semantic horizon
//! (the node cap is the only budget). Only the node cap is a caller knob;
//! exposing the rest would be a second solver configuration to keep in step
//! with the first.
//!
//! Both queries release the GIL for the whole computation, so a caller's
//! thread pool overlaps solves with its own model forward.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use hexo_engine::{apply_placement, HexCoord, HexoState as RustHexoState, Placement};

use crate::search::classify_root_move;
use crate::state::state_from_py_state;
use crate::threats_shared as threats;
use crate::tree::{tss_solve_verified, SolverHorizon, TssCounters};
use crate::tss_core::{ProofStatus, SolveGoal, ZoneSearchCaps};
use crate::tss_solver::TssSolver;
use crate::tss_verify::CertNode;

/// Serve-side deep-solve shape, copied from `configs/hexfield_eq_main_5.toml`.
const SERVE_DUAL_PASS: bool = true;
const SERVE_J2NEAR: bool = false;
const SERVE_GROUP2: bool = false;
const SERVE_LOSS_RESERVE_NODES: u32 = 0;
/// `horizon = 0` is UNBOUNDED (`semantic_horizon = u32::MAX`); the node cap is
/// then the only budget, which is what main_5 serves with.
const SERVE_HORIZON: SolverHorizon = SolverHorizon {
    horizon: 0,
    ladder: false,
};

/// Replay a placement path from the search root. Fails loudly on the first
/// illegal placement: a driver whose line mirror has drifted must find out
/// here, not silently solve a different position.
fn replay_path(root: &RustHexoState, path: &[(i16, i16)]) -> Result<RustHexoState, String> {
    let mut state = root.clone();
    for (index, &(q, r)) in path.iter().enumerate() {
        let coord = HexCoord { q, r };
        apply_placement(&mut state, Placement { coord }).map_err(|error| {
            format!(
                "TSS path placement {index} ({q}, {r}) is illegal from the search root: {error}"
            )
        })?;
    }
    Ok(state)
}

/// One λ¹ read, plus the root-guard classes for the caller's move list.
struct Lambda1Read {
    verdict: Option<f32>,
    has_threats: bool,
    own_win_now: bool,
    opp_threat_count: usize,
    b: u8,
    classes: Option<Vec<i8>>,
}

/// One verified deep solve, flattened for Python.
struct DeepRead {
    status: ProofStatus,
    proven_move: Option<(i16, i16)>,
    nodes: u64,
    verify_failed: u32,
}

fn status_name(status: ProofStatus) -> &'static str {
    match status {
        ProofStatus::Win => "win",
        ProofStatus::Loss => "loss",
        ProofStatus::Unknown => "unknown",
    }
}

/// Threat-Space Search over one search root, for a Python-driven tree.
#[pyclass]
pub(crate) struct TssProbe {
    root: RustHexoState,
}

#[pymethods]
impl TssProbe {
    /// Bind to one showcase `hexo_engine` state — the search root. The state
    /// is cloned through the engine capsule, so the probe is unaffected by
    /// later mutation of the Python object.
    #[new]
    fn new(py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            root: state_from_py_state(py, state)?,
        })
    }

    /// λ¹ analysis of `root + path`, plus per-move guard classes for `moves`.
    ///
    /// Keys:
    ///   * `verdict` — `1.0` proven win / `-1.0` proven one-turn forced loss /
    ///     `None` no λ¹ proof, all from the SIDE TO MOVE at that state. This is
    ///     `tss_core::solve_leaf_lambda1`'s value, i.e. the only λ¹ value the
    ///     hexfield tree ever backs up.
    ///   * `has_threats` — the engine's board-level ≥4-window index: the deep
    ///     leaf gate the hexfield leaf ladder uses.
    ///   * `own_win_now`, `opp_threat_count`, `b` — the analysis fields behind
    ///     the verdict (`b` is the placements-left budget for the turn).
    ///   * `move_classes` — `classify_root_move` for each entry of `moves`, in
    ///     the caller's order: `1` win-now, `-1` λ¹-refuted, `0` neither. It is
    ///     `None` exactly when the guard is inert (`not own_win_now` and no
    ///     opponent threats), which is the same short circuit
    ///     `search::tactical_guard_weights` takes — the caller must then use
    ///     its raw weights and pays for no classification pass.
    #[pyo3(signature = (path, moves))]
    fn lambda1<'py>(
        &self,
        py: Python<'py>,
        path: Vec<(i16, i16)>,
        moves: Vec<(i16, i16)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let root = self.root.clone();
        let read = py
            .detach(move || -> Result<Lambda1Read, String> {
                let state = replay_path(&root, &path)?;
                let analysis = threats::analyze(&state);
                // The guard's own short circuit (search::tactical_guard_weights):
                // with no own win and no opponent threat every class is 0 and
                // the weights come back untouched, so the pass is skipped whole.
                let classes = if analysis.own_win_now || analysis.opp_threat_count > 0 {
                    Some(
                        moves
                            .par_iter()
                            .map(|&(q, r)| classify_root_move(&state, HexCoord { q, r }))
                            .collect::<Vec<i8>>(),
                    )
                } else {
                    None
                };
                Ok(Lambda1Read {
                    verdict: analysis.verdict(),
                    has_threats: state.board().windows().has_threats(),
                    own_win_now: analysis.own_win_now,
                    opp_threat_count: analysis.opp_threat_count,
                    b: analysis.b,
                    classes,
                })
            })
            .map_err(PyValueError::new_err)?;
        let dict = PyDict::new(py);
        dict.set_item("verdict", read.verdict)?;
        dict.set_item("has_threats", read.has_threats)?;
        dict.set_item("own_win_now", read.own_win_now)?;
        dict.set_item("opp_threat_count", read.opp_threat_count)?;
        dict.set_item("b", read.b)?;
        dict.set_item("move_classes", read.classes)?;
        Ok(dict)
    }

    /// A verified deep solve of `root + path` under `node_cap` solver nodes.
    ///
    /// Keys:
    ///   * `status` — `"win"` / `"loss"` / `"unknown"`. A Win/Loss is reported
    ///     only after the INDEPENDENT certificate verifier accepted the
    ///     claim (`VerifiedSolve::hard` is `Some`); anything else is
    ///     `"unknown"`, so a capped, exhausted, or rejected solve can never
    ///     become a value.
    ///   * `value` — `1.0` / `-1.0` for a verified win/loss, `None` for
    ///     unknown, from the side to move at the solved state.
    ///   * `move` — for a verified win whose certificate root is a `Choice`,
    ///     the proven move as `(q, r)`; `None` otherwise. (Callers gate the
    ///     solve on an undecided λ¹, and an undecided λ¹ has no own win-now,
    ///     so an immediate-completion certificate root cannot occur there.)
    ///   * `nodes` — solver node expansions across every attempt of this solve.
    ///   * `verify_failed` — FATAL if nonzero: a Win/Loss claim whose
    ///     certificate the verifier rejected. Degraded to `"unknown"` here.
    #[pyo3(signature = (path, node_cap))]
    fn deep_solve<'py>(
        &self,
        py: Python<'py>,
        path: Vec<(i16, i16)>,
        node_cap: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if node_cap == 0 {
            return Err(PyValueError::new_err("TSS node_cap must be at least 1"));
        }
        let root = self.root.clone();
        let read = py
            .detach(move || -> Result<DeepRead, String> {
                let state = replay_path(&root, &path)?;
                let mut solver = TssSolver::default();
                solver.configure_leaf_profile();
                solver.set_leaf_j2near(SERVE_J2NEAR);
                solver.set_dual_pass(SERVE_DUAL_PASS);
                solver.set_loss_reserve_nodes(SERVE_LOSS_RESERVE_NODES);
                solver.set_group2(SERVE_GROUP2);
                let mut counters = TssCounters::default();
                let solved = tss_solve_verified(
                    &state,
                    node_cap,
                    SolveGoal::Both,
                    ZoneSearchCaps::default(),
                    SERVE_HORIZON,
                    &mut solver,
                    &mut counters,
                );
                // `hard` is Some only after the independent verifier accepted
                // the certificate; that — not the solver's own claim — is what
                // makes a status consumable.
                let status = if solved.hard.is_some() {
                    solved.status
                } else {
                    ProofStatus::Unknown
                };
                let proven_move = if status == ProofStatus::Win {
                    solved.cert.as_ref().and_then(|cert| {
                        match cert.nodes.get(cert.root_node as usize) {
                            Some(CertNode::Choice { mv, .. }) => Some((mv.q, mv.r)),
                            _ => None,
                        }
                    })
                } else {
                    None
                };
                Ok(DeepRead {
                    status,
                    proven_move,
                    nodes: counters.deep_nodes,
                    verify_failed: counters.deep_verify_failed,
                })
            })
            .map_err(PyValueError::new_err)?;
        let dict = PyDict::new(py);
        dict.set_item("status", status_name(read.status))?;
        dict.set_item("value", read.status.value())?;
        dict.set_item("move", read.proven_move)?;
        dict.set_item("nodes", read.nodes)?;
        dict.set_item("verify_failed", read.verify_failed)?;
        Ok(dict)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TssProbe>()
}
