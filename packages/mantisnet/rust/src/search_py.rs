//! Python bindings for the vendored Gumbel decision session.
//!
//! Python drives the whole loop — `begin`, then `pump`/`resume` until decided,
//! then `decision` — so leaf evaluation batches happen wherever the caller's
//! model lives, and the session never calls back into Python. `snapshot` is
//! the pull-based telemetry read: an observational copy of the running
//! search's candidates, taken between pumps, that cannot perturb the search
//! (see the `live_snapshot` note in `vendor/search/src/gumbel.rs`).

use std::collections::HashMap;
use std::num::{NonZeroU32, NonZeroUsize};

use hexo_search::gumbel::{GumbelConfig, GumbelSession};
use hexo_search::seam::Evaluation;
use hexo_search::session::{DecisionSession, LeafId, SessionStatus};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use crate::py::Position;

/// A nonblocking Gumbel sequential-halving search over one root position.
///
/// The as-evaled MantisNet search shape: `sims` total deepenings below the
/// root, at most `candidates` Gumbel-top root candidates, root noise scaled
/// by `temperature`.
#[pyclass]
pub(crate) struct GumbelSearch {
    inner: GumbelSession,
    root: Option<hexo_engine::Position>,
    in_flight: HashMap<u64, (LeafId, usize)>,
    next_key: u64,
}

#[pymethods]
impl GumbelSearch {
    #[new]
    #[pyo3(signature = (sims, candidates, temperature, seed))]
    fn new(sims: u32, candidates: usize, temperature: f64, seed: u64) -> PyResult<Self> {
        let simulations = NonZeroU32::new(sims)
            .ok_or_else(|| PyValueError::new_err("sims must be at least 1"))?;
        let candidates = NonZeroUsize::new(candidates)
            .ok_or_else(|| PyValueError::new_err("candidates must be at least 1"))?;
        if !(temperature.is_finite() && temperature >= 0.0) {
            return Err(PyValueError::new_err(
                "temperature must be finite and non-negative",
            ));
        }
        Ok(Self {
            inner: GumbelSession::new(
                GumbelConfig {
                    simulations,
                    candidates,
                    temperature,
                },
                seed,
            ),
            root: None,
            in_flight: HashMap::new(),
            next_key: 0,
        })
    }

    /// Reset onto `position` and discard any previous search.
    fn begin(&mut self, position: &Position) -> PyResult<()> {
        if position.inner.is_terminal() {
            return Err(PyValueError::new_err("cannot search a terminal position"));
        }
        self.in_flight.clear();
        self.inner.begin(&position.inner);
        self.root = Some(position.inner.clone());
        Ok(())
    }

    /// Run until the decision is ready or a wave of leaves needs evaluation.
    ///
    /// Returns `(decided, leaves)` where `leaves` is a list of
    /// `(key, position)` pairs. Every returned leaf must be answered with
    /// `resume(key, priors, value)` before the next `pump` can progress.
    fn pump(&mut self, py: Python<'_>) -> PyResult<(bool, Py<PyList>)> {
        if self.root.is_none() {
            return Err(PyValueError::new_err("pump before begin"));
        }
        let mut pairs: Vec<(LeafId, hexo_engine::Position)> = Vec::new();
        let status = self.inner.pump(&mut |leaf, position| {
            pairs.push((leaf, position.clone()));
        });
        let list = PyList::empty(py);
        for (leaf, position) in pairs {
            let key = self.next_key;
            self.next_key += 1;
            self.in_flight.insert(key, (leaf, position.legal_count()));
            let wrapped = Py::new(py, Position { inner: position })?;
            list.append((key, wrapped))?;
        }
        Ok((matches!(status, SessionStatus::Decided), list.unbind()))
    }

    /// Deliver one leaf's evaluation: `priors` over the leaf position's legal
    /// actions in canonical order, and `value` in `[-1, 1]` from the side to
    /// move at the leaf.
    fn resume(&mut self, key: u64, priors: Vec<f32>, value: f32) -> PyResult<()> {
        let (leaf, legal_count) = self
            .in_flight
            .remove(&key)
            .ok_or_else(|| PyValueError::new_err(format!("leaf key {key} is not in flight")))?;
        if priors.len() != legal_count {
            self.in_flight.insert(key, (leaf, legal_count));
            return Err(PyValueError::new_err(format!(
                "expected {legal_count} priors for leaf {key}, got {}",
                priors.len()
            )));
        }
        if !(value.is_finite() && (-1.0..=1.0).contains(&value)) {
            self.in_flight.insert(key, (leaf, legal_count));
            return Err(PyValueError::new_err(format!(
                "value {value} is outside [-1, 1]"
            )));
        }
        if priors.iter().any(|p| !p.is_finite() || *p < 0.0) {
            self.in_flight.insert(key, (leaf, legal_count));
            return Err(PyValueError::new_err(
                "priors must be finite and non-negative",
            ));
        }
        self.inner.resume(
            leaf,
            Evaluation {
                priors: priors.into_boxed_slice(),
                value,
            },
        );
        Ok(())
    }

    /// The decided move as `(q, r)`, or `None` while the search runs.
    fn decision(&mut self) -> Option<(i16, i16)> {
        self.inner.take_decision().map(|d| {
            let c = d.action.coord();
            (c.q, c.r)
        })
    }

    /// Observational snapshot of the running (or just-decided) search, or
    /// `None` before the root evaluation has been delivered.
    ///
    /// Keys: `actions` (candidate `(q, r)` in Gumbel-top order), `root_ranks`,
    /// `visits`, `values`, `scores`, `survivors` (indices into the candidate
    /// lists), `round`, `rounds`, `completed_visits`, `target_visits`.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(root) = &self.root else {
            return Ok(None);
        };
        let Some(snap) = self.inner.live_snapshot() else {
            return Ok(None);
        };
        let d = PyDict::new(py);
        let actions: Vec<(i16, i16)> = snap
            .lines
            .iter()
            .map(|line| {
                let action = root
                    .nth_legal(line.root_rank)
                    .expect("snapshot rank indexes the root legal set");
                let c = action.coord();
                (c.q, c.r)
            })
            .collect();
        d.set_item("actions", actions)?;
        d.set_item(
            "root_ranks",
            snap.lines.iter().map(|l| l.root_rank).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "visits",
            snap.lines.iter().map(|l| l.visits).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "values",
            snap.lines.iter().map(|l| l.value).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "scores",
            snap.lines.iter().map(|l| l.score).collect::<Vec<_>>(),
        )?;
        d.set_item("survivors", snap.survivors)?;
        d.set_item("round", snap.round)?;
        d.set_item("rounds", snap.rounds)?;
        d.set_item("completed_visits", snap.completed_visits)?;
        d.set_item("target_visits", snap.target_visits)?;
        Ok(Some(d))
    }

    /// Replace the RNG seed. Call at move boundaries, before `begin`.
    fn reseed(&mut self, seed: u64) {
        self.inner.reseed(seed);
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GumbelSearch>()
}
