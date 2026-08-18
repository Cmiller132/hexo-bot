//! Python bindings for the vendored MantisNet engine and encoder: the read
//! surface a model builder needs, and nothing that could bypass the rules.
//! Ported from the research repo's `python/hexo-py/src/lib.rs` (branch main,
//! MODEL_REPR_VERSION 7); the module name changes to `mantisnet._rust`, the
//! surface does not.
//!
//! The surface is the input list — stones, legal moves in canonical order,
//! `moves_remaining` — plus `windows_through`, which exists so a builder test
//! can check window enumeration against the engine as an independent oracle.
//! Positions are created empty or by replay, never deserialised: a board-shaped
//! constructor would be a rule-bypass hole, which is the same argument the
//! engine makes for itself.

use hexo_engine as engine;
use hexo_model_mantisnet::{MODEL_REPR_VERSION, encoder};
use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A Hexo position. Wraps `hexo_engine::Position` one-to-one.
#[pyclass]
pub(crate) struct Position {
    pub(crate) inner: engine::Position,
}

/// One window through a cell: `(axis, start_q, start_r, mask_p0, mask_p1)`.
type WindowTuple = (u8, i16, i16, u8, u8);

fn action(q: i16, r: i16) -> engine::Action {
    engine::Action::new(engine::HexCoord::new(q, r))
}

#[pymethods]
impl Position {
    /// The empty position: `P0` to move at the origin.
    #[new]
    fn new() -> Self {
        Self {
            inner: engine::Position::new(),
        }
    }

    /// Replay a placement sequence from the empty board.
    #[staticmethod]
    fn replay(moves: Vec<(i16, i16)>) -> PyResult<Self> {
        let actions: Vec<engine::Action> = moves.iter().map(|&(q, r)| action(q, r)).collect();
        engine::Position::replay(&actions)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Apply one placement. Raises `ValueError` on an illegal move.
    fn advance(&mut self, q: i16, r: i16) -> PyResult<()> {
        self.inner
            .advance(action(q, r))
            .map(|_| ())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// An independent copy.
    fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// Every stone as `(q, r, player)`, in canonical `(q, r)` order.
    fn stones(&self) -> Vec<(i16, i16, u8)> {
        self.inner
            .stones()
            .map(|(c, p)| (c.q, c.r, p.index() as u8))
            .collect()
    }

    /// Legal placements as `(q, r)`, in the engine's canonical order
    /// (`ACTION_ORDER_VERSION`). Empty exactly when the position is terminal.
    fn legal_moves(&self) -> Vec<(i16, i16)> {
        self.inner
            .legal_actions()
            .map(|a| {
                let c = a.coord();
                (c.q, c.r)
            })
            .collect()
    }

    /// The legal placement at `index` in that same order — what a caller
    /// holding a sampled rank wants, without materialising the whole list.
    fn nth_legal(&self, index: usize) -> PyResult<(i16, i16)> {
        self.inner
            .nth_legal(index)
            .map(|a| {
                let c = a.coord();
                (c.q, c.r)
            })
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "legal index {index} out of range ({} legal moves)",
                    self.inner.legal_count()
                ))
            })
    }

    /// The 18 win windows through `(q, r)`: `(axis, start_q, start_r, mask_p0,
    /// mask_p1)`, where bit `k` of a mask is the cell `k` steps from the start
    /// along the axis. Axes are `0 = Q (1,0)`, `1 = R (0,1)`, `2 = QR (1,-1)`.
    ///
    /// This is the engine's own window walk, exposed for the builder-oracle
    /// test. The builder must not call it — it is the independent oracle.
    fn windows_through(&self, q: i16, r: i16) -> PyResult<Vec<WindowTuple>> {
        let c = engine::HexCoord::new(q, r);
        if !c.is_valid() {
            return Err(PyValueError::new_err(format!(
                "coordinate ({q}, {r}) is outside the engine's domain"
            )));
        }
        Ok(self
            .inner
            .windows_through(c)
            .iter()
            .filter(|wr| wr.window.start.is_valid())
            .map(|wr| {
                (
                    wr.window.axis.index() as u8,
                    wr.window.start.q,
                    wr.window.start.r,
                    wr.mask.mask(engine::Player::P0),
                    wr.mask.mask(engine::Player::P1),
                )
            })
            .collect())
    }

    /// Number of legal placements. `0` if and only if terminal.
    #[getter]
    fn legal_count(&self) -> usize {
        self.inner.legal_count()
    }

    /// Whose turn it is: `0` or `1`. Frozen at the winner once terminal.
    #[getter]
    fn current_player(&self) -> u8 {
        self.inner.current_player().index() as u8
    }

    /// Placements the mover still has this turn: `2` before the first stone of
    /// a normal turn, `1` before its second stone or the opening stone.
    #[getter]
    fn moves_remaining(&self) -> u8 {
        match self.inner.phase() {
            engine::TurnPhase::FirstStone => 2,
            engine::TurnPhase::Opening | engine::TurnPhase::SecondStone => 1,
        }
    }

    /// Whether the game is over.
    #[getter]
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    /// The winner (`0` or `1`), or `None` while the game runs.
    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.outcome().map(|o| o.winner.index() as u8)
    }

    /// Total stones placed.
    #[getter]
    fn stone_count(&self) -> u32 {
        self.inner.stone_count()
    }

    /// Incremental Zobrist hash.
    #[getter]
    fn zobrist(&self) -> u64 {
        self.inner.zobrist()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Position stones={} to_move={} terminal={}>",
            self.inner.stone_count(),
            self.inner.current_player().index(),
            self.inner.is_terminal(),
        )
    }
}

/// A collated `RawBatch` as a dict of numpy arrays, keyed by the field names
/// `mantisnet.builder.Batch` uses.
fn raw_to_dict<'py>(py: Python<'py>, raw: encoder::RawBatch) -> PyResult<Bound<'py, PyDict>> {
    let (p, max_t, max_w) = (raw.n_pos, raw.max_t, raw.max_w);
    let d = PyDict::new(py);
    let n_w = raw.window_feat.len();
    let n_cells = raw.cell_pos.len();
    d.set_item("stone_own", PyArray1::from_vec(py, raw.stone_own))?;
    d.set_item("window_feat", PyArray1::from_vec(py, raw.window_feat))?;
    d.set_item(
        "window_id",
        PyArray1::from_vec(py, raw.window_id).reshape([n_w, 3])?,
    )?;
    d.set_item("moves_idx", PyArray1::from_vec(py, raw.moves_idx))?;
    d.set_item("inc_stone", PyArray1::from_vec(py, raw.inc_stone))?;
    d.set_item("inc_window", PyArray1::from_vec(py, raw.inc_window))?;
    d.set_item("inc_class", PyArray1::from_vec(py, raw.inc_class))?;
    d.set_item("stone_slot", PyArray1::from_vec(py, raw.stone_slot))?;
    d.set_item(
        "coords",
        PyArray1::from_vec(py, raw.coords).reshape([p, max_t, 2])?,
    )?;
    d.set_item(
        "attn_valid",
        PyArray1::from_vec(py, raw.attn_valid).reshape([p, max_t])?,
    )?;
    d.set_item("window_slot", PyArray1::from_vec(py, raw.window_slot))?;
    d.set_item(
        "value_valid",
        PyArray1::from_vec(py, raw.value_valid).reshape([p, max_w])?,
    )?;
    d.set_item("legal_offsets", PyArray1::from_vec(py, raw.legal_offsets))?;
    d.set_item("cell_pos", PyArray1::from_vec(py, raw.cell_pos))?;
    d.set_item(
        "cell_occupancy",
        PyArray1::from_vec(py, raw.cell_occupancy),
    )?;
    d.set_item(
        "cell_is_legal",
        PyArray1::from_vec(py, raw.cell_is_legal),
    )?;
    d.set_item("cell_nearest", PyArray1::from_vec(py, raw.cell_nearest))?;
    d.set_item("radius_src", PyArray1::from_vec(py, raw.radius_src))?;
    d.set_item("radius_dst", PyArray1::from_vec(py, raw.radius_dst))?;
    d.set_item("radius_orbit", PyArray1::from_vec(py, raw.radius_orbit))?;
    d.set_item("radius_own", PyArray1::from_vec(py, raw.radius_own))?;
    d.set_item(
        "radius_on_axis",
        PyArray1::from_vec(py, raw.radius_on_axis),
    )?;
    d.set_item(
        "adjacency_src",
        PyArray1::from_vec(py, raw.adjacency_src),
    )?;
    d.set_item(
        "adjacency_dst",
        PyArray1::from_vec(py, raw.adjacency_dst),
    )?;
    d.set_item(
        "adjacency_axis",
        PyArray1::from_vec(py, raw.adjacency_axis),
    )?;
    d.set_item("dec_cell", PyArray1::from_vec(py, raw.dec_cell))?;
    d.set_item("dec_window", PyArray1::from_vec(py, raw.dec_window))?;
    d.set_item("dec_class", PyArray1::from_vec(py, raw.dec_class))?;
    d.set_item("act_class", PyArray1::from_vec(py, raw.act_class))?;
    d.set_item("act_rev", PyArray1::from_vec(py, raw.act_rev))?;
    d.set_item(
        "act_empty",
        PyArray1::from_vec(py, raw.act_empty).reshape([n_cells, 3])?,
    )?;
    Ok(d)
}

/// Build a collated MantisNet batch from positions, in parallel.
///
/// The production twin of `mantisnet.builder`'s Python path, held equal to it
/// field for field by that package's parity tests. Raises `ValueError` on a
/// terminal position. Every nonempty window is represented under the ternary
/// tables.
#[pyfunction]
fn build_batch<'py>(
    py: Python<'py>,
    positions: Vec<PyRef<'py, Position>>,
) -> PyResult<Bound<'py, PyDict>> {
    let owned: Vec<engine::Position> = positions.iter().map(|p| p.inner.clone()).collect();
    let raw = py
        .detach(|| encoder::build_batch(&owned))
        .map_err(PyValueError::new_err)?;
    raw_to_dict(py, raw)
}

/// Replay each game's first `ts[i]` placements and build the batch, in
/// parallel — the fitting path, where a stored position is a move prefix.
#[pyfunction]
fn build_batch_prefixes<'py>(
    py: Python<'py>,
    games: Vec<Vec<(i16, i16)>>,
    ts: Vec<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let raw = py
        .detach(|| encoder::build_batch_prefixes(&games, &ts))
        .map_err(PyValueError::new_err)?;
    raw_to_dict(py, raw)
}

/// The module: `Position`, the batch builders, the decision sessions, and the
/// version constants a checkpoint pins.
#[pymodule]
pub fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Position>()?;
    crate::search_py::register(m)?;
    m.add_function(wrap_pyfunction!(build_batch, m)?)?;
    m.add_function(wrap_pyfunction!(build_batch_prefixes, m)?)?;
    m.add("RULES_VERSION", engine::RULES_VERSION)?;
    m.add("ACTION_ORDER_VERSION", engine::ACTION_ORDER_VERSION)?;
    m.add("LEGAL_RADIUS", engine::LEGAL_RADIUS)?;
    m.add("MODEL_REPR_VERSION", MODEL_REPR_VERSION)?;
    // A host orchestrator opens every seat with the three versions of
    // `CONTAINER_SPEC.md` §3.1's handshake. Two of them are the engine's and
    // already here; the third is the runner's, and a Python orchestrator
    // holding its own copy of that number is the drift this re-export exists
    // to prevent.
    m.add("PROTOCOL_VERSION", hexo_runner::PROTOCOL_VERSION)?;
    Ok(())
}
