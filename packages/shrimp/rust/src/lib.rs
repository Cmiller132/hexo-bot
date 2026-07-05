//! shrimp Rust accelerator crate, built as the cdylib `shrimp._rust`.
//!
//! Engine state intake goes through hexo_engine's C-ABI state capsule
//! (state.rs).
//!
//! Surfaces: serve-time support/feature construction, payload assembly, the
//! PUCT tree, and the continuous scheduler.
//!
//! Build: scripts/build_native.sh (maturin, --release).

// Several search-stat fields are write-only (telemetry).
#![allow(dead_code)]

mod constants;
mod features;
mod support;

// Threat-Space Search core (vendored into shrimp; see threats_shared.rs).
mod threats_shared;

#[cfg(feature = "python")]
mod cache;
#[cfg(feature = "python")]
mod payload;
#[cfg(feature = "python")]
mod replay_expand;
#[cfg(feature = "python")]
mod search;
#[cfg(feature = "python")]
mod serve_pack;
#[cfg(feature = "python")]
mod state;
#[cfg(feature = "python")]
mod tree;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyBytes, PyDict, PyList};

#[cfg(feature = "python")]
fn bytes_of<T>(py: Python<'_>, data: &[T]) -> Py<PyBytes> {
    let byte_len = std::mem::size_of_val(data);
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
    PyBytes::new(py, bytes).unbind()
}

#[cfg(feature = "python")]
#[pyfunction]
fn capabilities(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("status", "ready")?;
    dict.set_item("model_family", "shrimp")?;
    dict.set_item("state_source", "direct_engine_state")?;
    dict.set_item("num_features", constants::NUM_FEATURES)?;
    dict.set_item("support_node_order", "legal|stones|halo asc packed id")?;
    Ok(dict.into_any().unbind())
}

/// Serve-time featurization of engine states. Returns one dict per state:
/// coords (i16 q,r pairs), legal/stone/halo counts, dist (i32), nbr
/// (i32 row-local, -1 missing, node-major x 6), feats (f32 node-major x 15).
#[cfg(feature = "python")]
#[pyfunction(signature = (states))]
fn featurize_states(py: Python<'_>, states: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let rust_states = state::states_from_py_states(py, states)?;
    let out = PyList::empty(py);
    for st in &rust_states {
        let sup = support::build_support(st);
        let feats = features::build_features(st, &sup);

        let mut qr: Vec<i16> = Vec::with_capacity(sup.num_nodes() * 2);
        for c in &sup.coords {
            qr.push(c.q);
            qr.push(c.r);
        }
        let mut nbr_flat: Vec<i32> = Vec::with_capacity(sup.num_nodes() * 6);
        for row in &sup.nbr {
            nbr_flat.extend_from_slice(row);
        }

        let dict = PyDict::new(py);
        dict.set_item("num_nodes", sup.num_nodes())?;
        dict.set_item("legal_count", sup.legal_count)?;
        dict.set_item("stone_count", sup.stone_count)?;
        dict.set_item("halo_count", sup.halo_count)?;
        dict.set_item("coords", bytes_of(py, &qr))?;
        dict.set_item("dist", bytes_of(py, &sup.dist))?;
        dict.set_item("nbr", bytes_of(py, &nbr_flat))?;
        dict.set_item("feats", bytes_of(py, &feats))?;
        out.append(dict)?;
    }
    Ok(out.into_any().unbind())
}

/// Deterministic seed mixing from (base_seed, game_key, ply, stream).
#[cfg(feature = "python")]
#[pyfunction]
fn mix_seed(base_seed: u64, game_key: u64, ply: u32, stream: u64) -> u64 {
    search::mix_seed(base_seed, game_key, ply, stream)
}

/// LCB pick over (stats, z, min_visits, visit_fraction); see search::debug_lcb_from_stats.
/// The moves-left utility bonus is exposed below via debug_ml_bonus.
#[cfg(feature = "python")]
#[pyfunction]
fn debug_lcb_pick(
    stats: Vec<(u64, u32, u32, f32, f32)>,
    z: f32,
    min_visits: u32,
    visit_fraction: f32,
) -> Option<u64> {
    search::debug_lcb_from_stats(&stats, z, min_visits, visit_fraction)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (q, m_edge, m_node, weight, scale, gate, two_sided=false))]
fn debug_ml_bonus(
    q: f32,
    m_edge: f32,
    m_node: f32,
    weight: f32,
    scale: f32,
    gate: f32,
    two_sided: bool,
) -> f32 {
    search::debug_ml_bonus(q, m_edge, m_node, weight, scale, gate, two_sided)
}

#[cfg(feature = "python")]
#[pymodule]
pub fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    module.add_function(wrap_pyfunction!(featurize_states, module)?)?;
    module.add_function(wrap_pyfunction!(mix_seed, module)?)?;
    module.add_function(wrap_pyfunction!(debug_lcb_pick, module)?)?;
    module.add_function(wrap_pyfunction!(debug_ml_bonus, module)?)?;
    module.add_class::<search::ShrimpMctsSession>()?;
    // Parallel serve-pack with zero-copy buffers (SHRIMP_RUST_PACK path).
    module.add_function(wrap_pyfunction!(serve_pack::build_serve_groups, module)?)?;
    module.add_class::<serve_pack::F16Buf>()?;
    module.add_class::<serve_pack::I32Buf>()?;
    module.add_class::<serve_pack::U8Buf>()?;
    // rayon GIL-free train-read expand kernel (expand_backend="rust").
    module.add_function(wrap_pyfunction!(replay_expand::expand_shard_train, module)?)?;
    module.add_class::<replay_expand::RxF32Buf>()?;
    module.add_class::<replay_expand::RxF64Buf>()?;
    module.add_class::<replay_expand::RxI32Buf>()?;
    module.add_class::<replay_expand::RxI64Buf>()?;
    module.add_class::<replay_expand::RxU8Buf>()?;
    Ok(())
}
