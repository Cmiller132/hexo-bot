//! Narrow PyO3 bridge for Python access to stable Rust utilities.
//!
//! Compiled (behind the `python` feature) into the `hexo_utils._rust`
//! extension module that maturin builds; `python/hexo_utils/records.py`
//! re-exports these classes and
//! `packages/hexo_runner/python/hexo_runner/records/record.py` wraps that for
//! all production callers. Only the `.hxr` codec crosses the boundary --
//! `state_hash` stays Rust-only.
//!
//! Duck-typed runtime contracts (no Python imports at build time):
//! - `parse_players` accepts hexo_runner player objects via `.identity`
//!   (`.player_id`/`.label`) or any object with those attributes directly.
//! - `parse_abort` accepts anything with `.stage`/`.exception_type`/`.message`
//!   -- both this module's `AbortRecord` and hexo_runner's same-named
//!   dataclass satisfy it.
//! - `parse_action_id` takes either a packed u32 action id (the
//!   hexo_engine legal.rs (q,r) packing) or an action object with
//!   `.coord.q`/`.coord.r`.
//! - `PyHexoRecord.replay()` re-runs the stored action ids through the
//!   `hexo_engine` Python module (`new_game`/`PlacementAction`/`apply_action`/
//!   `terminal`).
//!
//! Error taxonomy (`record_error`): IO -> OSError; lifecycle misuse
//! (read-only/closed/finished writer) -> RuntimeError; malformed data ->
//! ValueError.

use std::path::PathBuf;

use hexo_engine::{pack_coord, HexCoord};
use pyo3::exceptions::{PyAttributeError, PyOSError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict, PyModule, PyTuple, PyType};

use crate::records::{
    AbortRecord as RustAbortRecord, HexoRecord as RustHexoRecord, HexoRecordEngineMetadata,
    HexoRecordFile as RustHexoRecordFile, HexoRecordGameWriter as RustHexoRecordGameWriter,
    HexoRecordPlayer as RustHexoRecordPlayer, HexoRecordRef, RecordError, HEXO_RECORD_MAGIC,
    HEXO_RECORD_SCHEMA_VERSION,
};

/// Abort information for fail-loud runner outcomes.
#[pyclass(
    name = "AbortRecord",
    module = "hexo_utils._rust",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyAbortRecord {
    #[pyo3(get)]
    stage: String,
    #[pyo3(get)]
    exception_type: String,
    #[pyo3(get)]
    message: String,
}

#[pymethods]
impl PyAbortRecord {
    #[new]
    fn new(stage: String, exception_type: String, message: String) -> Self {
        Self {
            stage,
            exception_type,
            message,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AbortRecord(stage={:?}, exception_type={:?}, message={:?})",
            self.stage, self.exception_type, self.message
        )
    }
}

impl From<RustAbortRecord> for PyAbortRecord {
    fn from(value: RustAbortRecord) -> Self {
        Self {
            stage: value.stage,
            exception_type: value.exception_type,
            message: value.message,
        }
    }
}

impl From<PyAbortRecord> for RustAbortRecord {
    fn from(value: PyAbortRecord) -> Self {
        Self {
            stage: value.stage,
            exception_type: value.exception_type,
            message: value.message,
        }
    }
}

/// Player identity stored once in a HexoRecordFile header.
#[pyclass(
    name = "HexoRecordPlayer",
    module = "hexo_utils._rust",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyHexoRecordPlayer {
    #[pyo3(get)]
    player_id: String,
    #[pyo3(get)]
    role: String,
    #[pyo3(get)]
    label: Option<String>,
}

#[pymethods]
impl PyHexoRecordPlayer {
    #[new]
    #[pyo3(signature = (player_id, role, label=None))]
    fn new(player_id: String, role: String, label: Option<String>) -> Self {
        Self {
            player_id,
            role,
            label,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HexoRecordPlayer(player_id={:?}, role={:?}, label={:?})",
            self.player_id, self.role, self.label
        )
    }
}

impl From<RustHexoRecordPlayer> for PyHexoRecordPlayer {
    fn from(value: RustHexoRecordPlayer) -> Self {
        Self {
            player_id: value.player_id,
            role: value.role,
            label: value.label,
        }
    }
}

impl From<PyHexoRecordPlayer> for RustHexoRecordPlayer {
    fn from(value: PyHexoRecordPlayer) -> Self {
        Self {
            player_id: value.player_id,
            role: value.role,
            label: value.label,
        }
    }
}

/// Replay-core data for one Hexo game.
#[pyclass(
    name = "HexoRecord",
    module = "hexo_utils._rust",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyHexoRecord {
    inner: RustHexoRecord,
}

#[pymethods]
impl PyHexoRecord {
    #[getter]
    fn game_id(&self) -> &str {
        &self.inner.game_id
    }

    #[getter]
    fn seed(&self) -> Option<i64> {
        self.inner.seed
    }

    #[getter]
    fn status(&self) -> &'static str {
        self.inner.status.as_str()
    }

    #[getter]
    fn action_ids(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.inner.action_ids.iter().copied())?
            .into_any()
            .unbind())
    }

    #[getter]
    fn abort(&self) -> Option<PyAbortRecord> {
        self.inner.abort.clone().map(Into::into)
    }

    #[getter]
    fn winner(&self) -> Option<String> {
        self.inner.winner.clone()
    }

    #[getter]
    fn placements(&self) -> Option<i64> {
        self.inner.placements
    }

    /// Re-run the stored action ids through the live `hexo_engine` Python
    /// module and return its `terminal(...)` result.
    ///
    /// The seed is forwarded to `new_game` for API symmetry, but the engine
    /// currently ignores it (hexo_engine pybridge.rs `new_game`); replay is
    /// deterministic from the action ids alone. Raises if the engine module
    /// is unavailable or a stored action is illegal under current rules.
    fn replay(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let engine = py.import("hexo_engine")?;
        let engine_types = py.import("hexo_engine.types")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("seed", self.inner.seed)?;
        let state = engine.call_method("new_game", (), Some(&kwargs))?;
        let placement_action = engine.getattr("PlacementAction")?;
        for action_id in &self.inner.action_ids {
            let coord = engine_types.call_method1("unpack_coord_id", (*action_id,))?;
            let action = placement_action.call1((coord,))?;
            engine.call_method1("apply_action", (&state, action))?;
        }
        Ok(engine.call_method1("terminal", (state,))?.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "HexoRecord(game_id={:?}, seed={:?}, status={:?}, actions={})",
            self.inner.game_id,
            self.inner.seed,
            self.inner.status.as_str(),
            self.inner.action_ids.len()
        )
    }
}

impl From<RustHexoRecord> for PyHexoRecord {
    fn from(value: RustHexoRecord) -> Self {
        Self { inner: value }
    }
}

/// Reader/writer for the binary Hexo runner record file format.
#[pyclass(
    name = "HexoRecordFile",
    module = "hexo_utils._rust",
    skip_from_py_object
)]
pub struct PyHexoRecordFile {
    inner: RustHexoRecordFile,
}

#[pymethods]
impl PyHexoRecordFile {
    #[classmethod]
    fn create(
        _cls: &Bound<'_, PyType>,
        path: &Bound<'_, PyAny>,
        engine_metadata: &Bound<'_, PyAny>,
        players: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let path = py_path(path)?;
        let engine_metadata = parse_engine_metadata(engine_metadata)?;
        let players = parse_players(players)?;
        Ok(Self {
            inner: RustHexoRecordFile::create(path, engine_metadata, players)
                .map_err(record_error)?,
        })
    }

    #[classmethod]
    fn open(_cls: &Bound<'_, PyType>, path: &Bound<'_, PyAny>) -> PyResult<Self> {
        let path = py_path(path)?;
        Ok(Self {
            inner: RustHexoRecordFile::open(path).map_err(record_error)?,
        })
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path().to_string_lossy().into_owned()
    }

    #[getter]
    fn mode(&self) -> &'static str {
        self.inner.mode().as_str()
    }

    #[getter]
    fn engine_metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("rules_version", self.inner.engine_metadata().rules_version)?;
        dict.set_item("backend", &self.inner.engine_metadata().backend)?;
        Ok(dict.into_any().unbind())
    }

    #[getter]
    fn players(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let players = self
            .inner
            .players()
            .iter()
            .cloned()
            .map(PyHexoRecordPlayer::from);
        Ok(PyTuple::new(py, players)?.into_any().unbind())
    }

    #[pyo3(signature = (game_id, seed=None, **kwargs))]
    fn begin_game(
        &mut self,
        game_id: String,
        seed: Option<i64>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyHexoRecordGameWriter> {
        if let Some(kwargs) = kwargs {
            if kwargs.contains("scenario")? {
                return Err(PyValueError::new_err(
                    "HexoRecordFile scenarios are not supported; use scenario=None.",
                ));
            }
            return Err(PyValueError::new_err(
                "unexpected HexoRecordFile.begin_game keyword argument",
            ));
        }
        Ok(PyHexoRecordGameWriter {
            inner: self.inner.begin_game(game_id, seed).map_err(record_error)?,
        })
    }

    fn iter_records(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let records = self
            .inner
            .iter_records()
            .map_err(record_error)?
            .into_iter()
            .map(PyHexoRecord::from);
        Ok(PyTuple::new(py, records)?.into_any().unbind())
    }

    fn close(&mut self) {
        self.inner.close();
    }

    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        exc_type: &Bound<'_, PyAny>,
        exc: &Bound<'_, PyAny>,
        traceback: &Bound<'_, PyAny>,
    ) -> bool {
        let _ = (exc_type, exc, traceback);
        self.close();
        false
    }

    fn __del__(&mut self) {
        self.close();
    }

    fn __repr__(&self) -> String {
        format!(
            "HexoRecordFile(path={:?}, mode={:?})",
            self.inner.path().to_string_lossy(),
            self.inner.mode().as_str()
        )
    }
}

/// Append-only writer for one game inside a HexoRecordFile.
#[pyclass(
    name = "HexoRecordGameWriter",
    module = "hexo_utils._rust",
    skip_from_py_object
)]
pub struct PyHexoRecordGameWriter {
    inner: RustHexoRecordGameWriter,
}

#[pymethods]
impl PyHexoRecordGameWriter {
    #[getter]
    fn game_id(&self) -> &str {
        self.inner.game_id()
    }

    #[getter]
    fn seed(&self) -> Option<i64> {
        self.inner.seed()
    }

    #[getter]
    fn action_count(&self) -> usize {
        self.inner.action_count()
    }

    fn record_action(&mut self, action: &Bound<'_, PyAny>) -> PyResult<()> {
        let action_id = parse_action_id(action)?;
        self.inner.record_action(action_id).map_err(record_error)
    }

    fn finish_completed(
        &mut self,
        py: Python<'_>,
        winner: &Bound<'_, PyAny>,
        placements: i64,
    ) -> PyResult<Py<PyAny>> {
        let winner = if winner.is_none() {
            None
        } else {
            Some(winner.str()?.to_str()?.to_owned())
        };
        let record_ref = self
            .inner
            .finish_completed(winner, placements)
            .map_err(record_error)?;
        record_ref_obj(py, &record_ref)
    }

    fn finish_aborted(&mut self, py: Python<'_>, abort: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let abort = parse_abort(abort)?;
        let record_ref = self.inner.finish_aborted(abort).map_err(record_error)?;
        record_ref_obj(py, &record_ref)
    }

    fn __repr__(&self) -> String {
        format!(
            "HexoRecordGameWriter(game_id={:?}, actions={})",
            self.inner.game_id(),
            self.inner.action_count()
        )
    }
}

/// Return a tiny capabilities object for smoke tests and packaging checks.
// UNUSED(2026-06-12): no references found in packages/tests/scripts -- every
// caller of a `capabilities()` resolves to a model package's own rust_bridge
// (dense_cnn/hexgt/hexgnn/dense_cnn_restnet); nothing imports
// hexo_utils._rust.capabilities despite the docstring's smoke-test claim.
#[pyfunction]
pub fn capabilities(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("status", "ready")?;
    dict.set_item("records", true)?;
    dict.set_item("selfplay", false)?;
    dict.set_item("samples", false)?;
    dict.set_item("message", "hexo_utils exposes the stable record codec")?;
    Ok(dict.into_any().unbind())
}

/// Register Python-visible functions on a module.
pub fn register_pybridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("HEXO_RECORD_MAGIC", PyBytes::new(py, HEXO_RECORD_MAGIC))?;
    module.add("HEXO_RECORD_SCHEMA_VERSION", HEXO_RECORD_SCHEMA_VERSION)?;
    module.add_class::<PyAbortRecord>()?;
    module.add_class::<PyHexoRecordPlayer>()?;
    module.add_class::<PyHexoRecord>()?;
    module.add_class::<PyHexoRecordFile>()?;
    module.add_class::<PyHexoRecordGameWriter>()?;
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    Ok(())
}

/// Private Python extension module entry point.
#[pymodule]
pub fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_pybridge(module)
}

// --- Duck-typed parsers + error mapping (Python -> Rust boundary helpers) ---

fn py_path(path: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    let os = path.py().import("os")?;
    let value = os.call_method1("fspath", (path,))?;
    Ok(PathBuf::from(value.extract::<String>()?))
}

fn parse_engine_metadata(metadata: &Bound<'_, PyAny>) -> PyResult<HexoRecordEngineMetadata> {
    let rules_version = metadata
        .call_method1("get", ("rules_version", 0_u64))?
        .extract::<u64>()?;
    let backend = metadata
        .call_method1("get", ("backend", ""))?
        .str()?
        .to_str()?
        .to_owned();
    Ok(HexoRecordEngineMetadata::new(rules_version, backend))
}

fn parse_players(players: &Bound<'_, PyAny>) -> PyResult<Vec<RustHexoRecordPlayer>> {
    let roles = ["player0", "player1"];
    let mut out = Vec::new();
    for (index, item) in players.try_iter()?.enumerate() {
        let player = item?;
        let identity = match player.getattr("identity") {
            Ok(identity) => identity,
            Err(error) if error.is_instance_of::<PyAttributeError>(player.py()) => player.clone(),
            Err(error) => return Err(error),
        };
        let player_id = required_attr_string(&identity, "player_id")?;
        let label = optional_attr_string(&identity, "label")?;
        let role = roles
            .get(index)
            .map(|role| (*role).to_owned())
            .unwrap_or_else(|| format!("player{index}"));
        out.push(RustHexoRecordPlayer::new(player_id, role, label));
    }
    Ok(out)
}

fn parse_action_id(action: &Bound<'_, PyAny>) -> PyResult<u32> {
    if let Ok(value) = action.extract::<i64>() {
        return u32::try_from(value).map_err(|_| {
            PyValueError::new_err(format!("packed action id outside u32 range: {value}"))
        });
    }

    let coord = match action.getattr("coord") {
        Ok(coord) => coord,
        Err(error) if error.is_instance_of::<PyAttributeError>(action.py()) => {
            return Err(PyTypeError::new_err(format!(
                "unsupported record action type: {}",
                action.get_type().name()?
            )));
        }
        Err(error) => return Err(error),
    };
    let q = coord.getattr("q")?.extract::<i16>()?;
    let r = coord.getattr("r")?.extract::<i16>()?;
    Ok(pack_coord(HexCoord { q, r }))
}

fn parse_abort(abort: &Bound<'_, PyAny>) -> PyResult<RustAbortRecord> {
    Ok(RustAbortRecord::new(
        required_attr_string(abort, "stage")?,
        required_attr_string(abort, "exception_type")?,
        required_attr_string(abort, "message")?,
    ))
}

fn required_attr_string(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    Ok(obj.getattr(name)?.str()?.to_str()?.to_owned())
}

fn optional_attr_string(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    match obj.getattr(name) {
        Ok(value) if value.is_none() => Ok(None),
        Ok(value) => Ok(Some(value.str()?.to_str()?.to_owned())),
        Err(error) if error.is_instance_of::<PyAttributeError>(obj.py()) => Ok(None),
        Err(error) => Err(error),
    }
}

fn record_ref_obj(py: Python<'_>, record_ref: &HexoRecordRef) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("path", record_ref.path.to_string_lossy().as_ref())?;
    dict.set_item("game_id", &record_ref.game_id)?;
    dict.set_item("status", record_ref.status.as_str())?;
    Ok(dict.into_any().unbind())
}

fn record_error(error: RecordError) -> PyErr {
    match error {
        RecordError::Io(error) => PyOSError::new_err(error.to_string()),
        RecordError::ReadOnlyFile
        | RecordError::WriteOnlyFile
        | RecordError::ClosedFile
        | RecordError::FinishedWriter { .. } => PyRuntimeError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}
