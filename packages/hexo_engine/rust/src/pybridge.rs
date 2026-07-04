//! PyO3 bridge for the Hexo rules engine.
//!
//! Compiled (behind the `python` feature) as the extension module
//! `hexo_engine._rust`, wrapped by `python/hexo_engine/api.py`. Two distinct
//! consumers:
//! - Python callers go through the pyfunctions below; rule violations map to
//!   `ValueError`, which api.py re-raises as `IllegalActionError`.
//! - Model accelerator crates (hexo_models/dense_cnn, hexo_models/hexgt,
//!   hexgnn — each rust/src/state.rs) fetch `state_api_capsule()` at batch-MCTS
//!   time to clone live `PyHexoState` objects into owned Rust states via the
//!   C-ABI fn pointers; they must check `version == STATE_API_VERSION` (2) and
//!   fail loudly on mismatch.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::ffi::c_void;

use crate::{
    apply_placement, pack_coord, Axis, GameOutcome, HexCoord, HexoState as RustHexoState,
    MoveError, Placement, Player, TurnPhase,
};

const STATE_API_CAPSULE_NAME: &str = "hexo_engine._rust.state_api";
const STATE_API_VERSION: u32 = 2;
const STATE_API_OK: i32 = 0;
const STATE_API_NULL_ARGUMENT: i32 = -1;
const STATE_API_TYPE_ERROR: i32 = -2;

#[repr(C)]
#[derive(Clone, Copy)]
struct HexoStateApi {
    version: u32,
    clone_state: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> i32,
    free_state: unsafe extern "C" fn(*mut c_void),
}

static STATE_API: HexoStateApi = HexoStateApi {
    version: STATE_API_VERSION,
    clone_state: clone_state_for_capsule,
    free_state: free_state_for_capsule,
};

/// Python-owned opaque handle to a Rust Hexo state.
#[pyclass(name = "HexoState", module = "hexo_engine._rust", skip_from_py_object)]
#[derive(Clone)]
pub struct PyHexoState {
    state: RustHexoState,
}

#[pymethods]
impl PyHexoState {
    fn __repr__(&self) -> String {
        format!(
            "HexoState(placements_made={}, terminal={})",
            self.state.placements_made(),
            self.state.is_terminal()
        )
    }
}

/// Create a fresh game state.
///
/// `seed` and `scenario` are accepted for API-shape stability but DISCARDED:
/// the engine has no randomness and no scenario loader, so every game starts
/// identically. Callers (hexo_runner session plumbing, hexo_frontend web.py)
/// pass `seed` through anyway — do not read that as engine reproducibility
/// control. `GameSpec.scenario` must be None and hexo_runner enforces that.
#[pyfunction(signature = (seed=None, scenario=None))]
pub fn new_game(seed: Option<u64>, scenario: Option<Py<PyAny>>) -> PyHexoState {
    let _ = seed;
    let _ = scenario;
    PyHexoState {
        state: RustHexoState::new(),
    }
}

#[pyfunction]
pub fn clone_state(state: PyRef<'_, PyHexoState>) -> PyHexoState {
    PyHexoState {
        state: state.state.clone(),
    }
}

#[pyfunction]
pub fn current_player(state: PyRef<'_, PyHexoState>) -> &'static str {
    player_label(state.state.current_player())
}

#[pyfunction]
pub fn legal_action_ids(py: Python<'_>, state: PyRef<'_, PyHexoState>) -> PyResult<Py<PyAny>> {
    let mut actions = Vec::with_capacity(state.state.legal_move_count());
    state.state.write_legal_action_ids(&mut actions);
    Ok(PyTuple::new(py, actions)?.into_any().unbind())
}

#[pyfunction]
pub fn legal_action_count(state: PyRef<'_, PyHexoState>) -> usize {
    state.state.legal_move_count()
}

#[pyfunction]
pub fn is_legal_action(state: PyRef<'_, PyHexoState>, q: i16, r: i16) -> bool {
    crate::is_legal_placement(&state.state, HexCoord { q, r }).is_ok()
}

#[pyfunction]
pub fn apply_action(
    py: Python<'_>,
    mut state: PyRefMut<'_, PyHexoState>,
    q: i16,
    r: i16,
) -> PyResult<Py<PyAny>> {
    let result = apply_placement(
        &mut state.state,
        Placement {
            coord: HexCoord { q, r },
        },
    )
    .map_err(move_error)?;

    let dict = PyDict::new(py);
    dict.set_item("terminal", result.outcome.is_some())?;
    dict.set_item(
        "next_player",
        result
            .outcome
            .is_none()
            .then(|| player_label(state.state.current_player())),
    )?;

    let metadata = PyDict::new(py);
    metadata.set_item("placements_made", state.state.placements_made())?;
    dict.set_item("metadata", metadata)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
pub fn terminal(py: Python<'_>, state: PyRef<'_, PyHexoState>) -> PyResult<Option<Py<PyAny>>> {
    outcome_obj(py, state.state.terminal())
}

/// Materialize the full state as nested Python dicts (the `PythonHexoState`
/// mirror shape parsed by api.py `to_python_state`).
///
/// Heavyweight by design: every stone, legal coordinate, and window entry
/// becomes a dict — O(windows) ~ O(18 x placements) per call. Intended for the
/// dashboard/replay layer (hexo_frontend), not for search/selfplay hot paths,
/// which use `legal_action_ids` + the state capsule instead.
#[pyfunction]
pub fn to_python_state(py: Python<'_>, state: PyRef<'_, PyHexoState>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("board", board_obj(py, &state.state)?)?;
    dict.set_item("current_player", player_label(state.state.current_player()))?;
    dict.set_item("phase", phase_label(state.state.phase()))?;
    dict.set_item("placements_made", state.state.placements_made())?;
    dict.set_item("terminal", outcome_obj(py, state.state.terminal())?)?;
    dict.set_item("last_turn", last_turn_obj(py, &state.state)?)?;
    dict.set_item(
        "placement_history",
        placement_history_obj(py, &state.state)?,
    )?;
    dict.set_item("first_stone", first_stone_obj(py, state.state.phase())?)?;
    Ok(dict.into_any().unbind())
}

#[pyfunction]
pub fn action_id(q: i16, r: i16) -> u32 {
    pack_coord(HexCoord { q, r })
}

/// Identity dict (`backend`, `rules_version`, `state_api_version`) embedded in
/// hexo_runner game records and asserted by tests/test_hexo_engine_rust_bridge.py.
#[pyfunction]
pub fn engine_metadata(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("engine_api", true)?;
    dict.set_item("backend", "rust-pyo3")?;
    dict.set_item(
        "rules_version",
        RustHexoState::new().snapshot().rules_version(),
    )?;
    dict.set_item("state_api_version", STATE_API_VERSION)?;
    Ok(dict.into_any().unbind())
}

// --- C-ABI state capsule (FFI entry for the model accelerator crates) ---

/// Expose the `HexoStateApi` fn-pointer table as a PyCapsule named
/// `hexo_engine._rust.state_api` so model crates compiled into a DIFFERENT
/// cdylib (hexo_models) can clone/free Rust states without linking this crate's
/// Python types.
#[pyfunction]
pub fn state_api_capsule(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let name = pyo3::ffi::c_str!("hexo_engine._rust.state_api");
    debug_assert_eq!(name.to_string_lossy(), STATE_API_CAPSULE_NAME);
    let pointer = (&STATE_API as *const HexoStateApi).cast::<c_void>() as *mut c_void;
    let capsule = unsafe { pyo3::ffi::PyCapsule_New(pointer, name.as_ptr(), None) };
    if capsule.is_null() {
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { Py::<PyAny>::from_owned_ptr(py, capsule) })
}

#[pymodule]
pub fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHexoState>()?;
    module.add_function(wrap_pyfunction!(new_game, module)?)?;
    module.add_function(wrap_pyfunction!(clone_state, module)?)?;
    module.add_function(wrap_pyfunction!(current_player, module)?)?;
    module.add_function(wrap_pyfunction!(legal_action_ids, module)?)?;
    module.add_function(wrap_pyfunction!(legal_action_count, module)?)?;
    module.add_function(wrap_pyfunction!(is_legal_action, module)?)?;
    module.add_function(wrap_pyfunction!(apply_action, module)?)?;
    module.add_function(wrap_pyfunction!(terminal, module)?)?;
    module.add_function(wrap_pyfunction!(to_python_state, module)?)?;
    module.add_function(wrap_pyfunction!(action_id, module)?)?;
    module.add_function(wrap_pyfunction!(engine_metadata, module)?)?;
    module.add_function(wrap_pyfunction!(state_api_capsule, module)?)?;
    Ok(())
}

// --- private helpers: error mapping, capsule fns, dict builders ---

fn move_error(error: MoveError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

unsafe extern "C" fn clone_state_for_capsule(object: *mut c_void, out: *mut *mut c_void) -> i32 {
    if object.is_null() || out.is_null() {
        return STATE_API_NULL_ARGUMENT;
    }
    Python::try_attach(|py| {
        let any =
            unsafe { Bound::<PyAny>::from_borrowed_ptr(py, object as *mut pyo3::ffi::PyObject) };
        let Ok(state) = any.extract::<PyRef<'_, PyHexoState>>() else {
            return STATE_API_TYPE_ERROR;
        };
        let cloned = Box::new(state.state.clone());
        unsafe {
            *out = Box::into_raw(cloned).cast::<c_void>();
        }
        STATE_API_OK
    })
    .unwrap_or(STATE_API_TYPE_ERROR)
}

unsafe extern "C" fn free_state_for_capsule(state: *mut c_void) {
    if state.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(state.cast::<RustHexoState>());
    }
}

fn player_label(player: Player) -> &'static str {
    match player {
        Player::Player0 => "player0",
        Player::Player1 => "player1",
    }
}

fn phase_label(phase: TurnPhase) -> &'static str {
    match phase {
        TurnPhase::Opening => "Opening",
        TurnPhase::FirstStone => "FirstStone",
        TurnPhase::SecondStone { .. } => "SecondStone",
    }
}

fn axis_label(axis: Axis) -> &'static str {
    match axis {
        Axis::Q => "Q",
        Axis::R => "R",
        Axis::QR => "QR",
    }
}

fn coord_obj(py: Python<'_>, coord: HexCoord) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("q", coord.q)?;
    dict.set_item("r", coord.r)?;
    Ok(dict.into_any().unbind())
}

fn outcome_obj(py: Python<'_>, outcome: Option<GameOutcome>) -> PyResult<Option<Py<PyAny>>> {
    let Some(outcome) = outcome else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    dict.set_item("winner", player_label(outcome.winner))?;
    dict.set_item("reason", "six_in_line")?;
    let metadata = PyDict::new(py);
    metadata.set_item("placements", outcome.placements)?;
    dict.set_item("metadata", metadata)?;
    Ok(Some(dict.into_any().unbind()))
}

fn board_obj(py: Python<'_>, state: &RustHexoState) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);

    let stones = PyList::empty(py);
    let mut occupied = state.board().occupied_cells().to_vec();
    occupied.sort_by_key(|coord| (coord.q, coord.r));
    for coord in occupied {
        if let Some(player) = state.board().get(coord) {
            let item = PyDict::new(py);
            item.set_item("coord", coord_obj(py, coord)?)?;
            item.set_item("player", player_label(player))?;
            stones.append(item)?;
        }
    }
    dict.set_item("stones", stones)?;

    let occupied_ordered = PyList::empty(py);
    for coord in state.board().occupied_cells() {
        occupied_ordered.append(coord_obj(py, *coord)?)?;
    }
    dict.set_item("occupied", occupied_ordered)?;

    let legal = PyList::empty(py);
    let mut legal_coords = Vec::new();
    state.write_legal_moves(&mut legal_coords);
    for coord in legal_coords {
        legal.append(coord_obj(py, coord)?)?;
    }
    dict.set_item("legal", legal)?;

    dict.set_item("windows", window_entries_obj(py, state)?)?;
    Ok(dict.into_any().unbind())
}

fn window_entries_obj(py: Python<'_>, state: &RustHexoState) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    let mut entries: Vec<_> = state.board().windows().entries().collect();
    entries.sort_by_key(|entry| {
        let key = entry.key();
        (key.axis.index(), key.start.q, key.start.r)
    });

    for entry in entries {
        let key = entry.key();
        let item = PyDict::new(py);
        item.set_item("start", coord_obj(py, key.start)?)?;
        item.set_item("axis", axis_label(key.axis))?;
        item.set_item(
            "masks",
            (entry.mask(Player::Player0), entry.mask(Player::Player1)),
        )?;
        list.append(item)?;
    }
    Ok(list.into_any().unbind())
}

fn last_turn_obj(py: Python<'_>, state: &RustHexoState) -> PyResult<Option<Py<PyAny>>> {
    let Some(record) = state.last_turn() else {
        return Ok(None);
    };
    let dict = PyDict::new(py);
    dict.set_item("player", player_label(record.player))?;
    let placements = PyList::empty(py);
    for coord in &record.placements {
        placements.append(coord_obj(py, *coord)?)?;
    }
    dict.set_item("placements", placements)?;
    Ok(Some(dict.into_any().unbind()))
}

fn placement_history_obj(py: Python<'_>, state: &RustHexoState) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for record in state.placement_history() {
        let item = PyDict::new(py);
        item.set_item("player", player_label(record.player))?;
        item.set_item("coord", coord_obj(py, record.coord)?)?;
        item.set_item("phase", phase_label(record.phase))?;
        item.set_item("placement_index", record.placement_index)?;
        item.set_item("first_stone", first_stone_obj(py, record.phase)?)?;
        list.append(item)?;
    }
    Ok(list.into_any().unbind())
}

fn first_stone_obj(py: Python<'_>, phase: TurnPhase) -> PyResult<Option<Py<PyAny>>> {
    match phase {
        TurnPhase::SecondStone { first } => Ok(Some(coord_obj(py, first)?)),
        TurnPhase::Opening | TurnPhase::FirstStone => Ok(None),
    }
}
