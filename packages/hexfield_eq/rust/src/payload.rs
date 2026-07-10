//! Evaluator payload ABI + cached batch evaluation.
//!
//! Request: one dict per flush; CSR flat-concat over support nodes; rows
//! pre-sorted by support size DESCENDING (stable by request index) so Python
//! grouping is contiguous slicing; the dedup slot-map restores caller order on
//! reply. Reply keys: values_bytes (f32 x B), priors_bytes (f32 x sum(L_g),
//! positional over each row's legal prefix), the optional moves_left_bytes
//! (f32 x B, decoded decisions in [0, 512]) when `request_moves_left` is set,
//! and the optional priors_logits_bytes (f32 x sum(L_g), raw pre-softmax policy
//! logits, same positional layout as priors_bytes) when `request_logits` is set.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use half::f16;
use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;
use hexo_engine::{pack_coord, HexoState as RustHexoState, PackedCoord};
use hexo_utils::StateHash;

use crate::cache::{
    lock_cache, lock_stats, RustEvaluation, RustEvaluationRequest, SharedEvaluationCache,
    SharedEvaluationStats,
};
use crate::constants::{num_features, RAYLEN_SLOTS};
use crate::features::{build_features, build_ray_lengths};
use crate::support::build_support;

pub const ABI_VERSION: u32 = 1;
pub const NBR_SENTINEL: u16 = 0xFFFF;
/// Maximum number of states per evaluator chunk.
pub const EVAL_CHUNK_STATES: usize = 1024;

/// One featurized request row. Owns all its data (no borrow of the source
/// state), so a built row set can outlive `states`; used by the async
/// submit/finish split, where parsing happens after the borrowing scope ends.
struct Row {
    request_index: usize,
    legal_ids: Vec<PackedCoord>,
    coords_qr: Vec<i16>,
    nbr_local: Vec<u16>,
    feats: Vec<f16>,
    raylen: Vec<u8>,
    num_nodes: usize,
}

/// Featurize each row, then order rows by support size DESCENDING (stable by
/// request index). Rust keeps the per-row sorted legal action ids; they never
/// cross the boundary — priors return positionally over the prefix.
fn featurize_and_sort(states: &[&RustHexoState]) -> PyResult<Vec<Row>> {
    // Featurize rows across rayon workers: build_support (a depth-9 BFS) and
    // build_features are pure functions of &state, so per-row work is
    // independent. par_iter collect preserves request order; the sort below is
    // deterministic. Runs without holding the GIL.
    let mut rows: Vec<Row> = states
        .par_iter()
        .enumerate()
        .map(|(request_index, state)| {
            let sup = build_support(state);
            let feats32 = build_features(state, &sup);
            let raylen = build_ray_lengths(state, &sup);
            let mut feats = vec![f16::ZERO; feats32.len()];
            // SIMD f32->f16, round-to-nearest, element-wise.
            feats.convert_from_f32_slice(&feats32);
            let mut coords_qr = Vec::with_capacity(sup.num_nodes() * 2);
            for c in &sup.coords {
                coords_qr.push(c.q);
                coords_qr.push(c.r);
            }
            let mut nbr_local = Vec::with_capacity(sup.num_nodes() * 6);
            for row in &sup.nbr {
                for &j in row {
                    nbr_local.push(if j < 0 { NBR_SENTINEL } else { j as u16 });
                }
            }
            if sup.num_nodes() > NBR_SENTINEL as usize {
                return Err(PyValueError::new_err(format!(
                    "support of {} nodes exceeds the u16 neighbour wire limit",
                    sup.num_nodes()
                )));
            }
            let legal_ids: Vec<PackedCoord> = sup.coords[..sup.legal_count]
                .iter()
                .map(|&c| pack_coord(c))
                .collect();
            Ok(Row {
                request_index,
                legal_ids,
                coords_qr,
                nbr_local,
                feats,
                raylen,
                num_nodes: sup.num_nodes(),
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    rows.sort_by(|a, b| {
        b.num_nodes
            .cmp(&a.num_nodes)
            .then_with(|| a.request_index.cmp(&b.request_index))
    });
    Ok(rows)
}

/// Pack featurized rows into the wire payload dict and fold in the encode
/// stats. Used by both the sync and async paths.
fn build_chunk_payload<'py>(
    py: Python<'py>,
    rows: &[Row],
    request_moves_left: bool,
    request_logits: bool,
    encoding_started: Instant,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Bound<'py, PyDict>> {
    let total_nodes: usize = rows.iter().map(|r| r.num_nodes).sum();
    let b = rows.len();
    let mut node_feats: Vec<f16> = Vec::with_capacity(total_nodes * num_features());
    let mut node_qr: Vec<i16> = Vec::with_capacity(total_nodes * 2);
    let mut nbr: Vec<u16> = Vec::with_capacity(total_nodes * 6);
    let mut raylen: Vec<u8> = Vec::with_capacity(total_nodes * RAYLEN_SLOTS);
    let mut node_row_offsets: Vec<i64> = Vec::with_capacity(b + 1);
    let mut legal_counts: Vec<i32> = Vec::with_capacity(b);
    node_row_offsets.push(0);
    for row in rows {
        node_feats.extend_from_slice(&row.feats);
        node_qr.extend_from_slice(&row.coords_qr);
        nbr.extend_from_slice(&row.nbr_local);
        raylen.extend_from_slice(&row.raylen);
        legal_counts.push(row.legal_ids.len() as i32);
        node_row_offsets.push(node_row_offsets.last().unwrap() + row.num_nodes as i64);
    }

    fn bytes_of<'py, T>(py: Python<'py>, data: &[T]) -> Bound<'py, PyBytes> {
        let len = std::mem::size_of_val(data);
        let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, len) };
        PyBytes::new(py, raw)
    }

    let payload = PyDict::new(py);
    payload.set_item("abi", ABI_VERSION)?;
    // The featurizer support radius this payload was built under (spec D-S26):
    // the evaluator asserts it against its own build so a serve-side radius
    // desync fails loudly instead of silently shifting the input distribution.
    payload.set_item("support_radius", crate::support::support_radius())?;
    payload.set_item("shape", (b, total_nodes))?;
    payload.set_item("node_feats", bytes_of(py, &node_feats))?;
    payload.set_item("node_qr", bytes_of(py, &node_qr))?;
    payload.set_item("node_row_offsets", node_row_offsets)?;
    payload.set_item("nbr", bytes_of(py, &nbr))?;
    // Side-relative ray lengths (Phase L0), u8 node-major x RAYLEN_SLOTS,
    // CSR-flat like nbr. Additive key: ABI_VERSION stays 1 (the payload is a
    // self-describing dict and no consumer requires the key yet).
    payload.set_item("raylen", bytes_of(py, &raylen))?;
    payload.set_item("legal_counts", bytes_of(py, &legal_counts))?;
    payload.set_item("request_moves_left", request_moves_left)?;
    // When set, requests the evaluator emit raw pre-softmax logits
    // (`priors_logits_bytes`). When unset, no logit column is emitted.
    payload.set_item("request_logits", request_logits)?;
    if let Some(stats) = stats {
        let mut stats = lock_stats(stats);
        stats.evaluator_chunks += 1;
        stats.encoded_states += b;
        stats.encoded_nodes += total_nodes;
        stats.max_chunk_states = stats.max_chunk_states.max(b);
        stats.input_bytes += node_feats.len() * 2 + node_qr.len() * 2 + nbr.len() * 2 + raylen.len();
        stats.encoding_seconds += encoding_started.elapsed().as_secs_f64();
    }
    Ok(payload)
}

/// Parse one evaluator reply (the dict returned by `evaluate_payload`/`result`)
/// against the sorted rows it was built from, restoring caller order.
fn parse_chunk_reply(
    output: &Bound<'_, PyAny>,
    rows: &[Row],
    states_len: usize,
    request_moves_left: bool,
    request_logits: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let parse_started = Instant::now();
    let b = rows.len();
    let values_obj = output
        .get_item("values_bytes")
        .map_err(|_| PyValueError::new_err("hexfield evaluator output missing values_bytes"))?;
    let priors_obj = output
        .get_item("priors_bytes")
        .map_err(|_| PyValueError::new_err("hexfield evaluator output missing priors_bytes"))?;
    let value_bytes = values_obj.downcast::<PyBytes>()?.as_bytes();
    let prior_bytes = priors_obj.downcast::<PyBytes>()?.as_bytes();
    require_exact_bytes("values_bytes", value_bytes.len(), b, 4)?;
    let expected_priors: usize = rows.iter().map(|r| r.legal_ids.len()).sum();
    require_exact_bytes("priors_bytes", prior_bytes.len(), expected_priors, 4)?;
    let moves_left_bytes: Option<Vec<u8>> = if request_moves_left {
        let obj = output.get_item("moves_left_bytes").map_err(|_| {
            PyValueError::new_err(
                "hexfield evaluator output missing moves_left_bytes (request_moves_left was set)",
            )
        })?;
        let bytes = obj.downcast::<PyBytes>()?.as_bytes().to_vec();
        require_exact_bytes("moves_left_bytes", bytes.len(), b, 4)?;
        Some(bytes)
    } else {
        None
    };
    // Optional raw-logit column. Same positional layout as `priors_bytes` (sum
    // of legal prefixes, fp32). When `request_logits` is unset, or when set but
    // the key is absent from the reply, this is None.
    let logits_bytes: Option<Vec<u8>> = if request_logits {
        match output.get_item("priors_logits_bytes") {
            Ok(obj) => {
                let bytes = obj.downcast::<PyBytes>()?.as_bytes().to_vec();
                require_exact_bytes("priors_logits_bytes", bytes.len(), expected_priors, 4)?;
                Some(bytes)
            }
            Err(_) => None,
        }
    } else {
        None
    };
    if let Some(stats) = stats {
        let mut stats = lock_stats(stats);
        stats.value_bytes += value_bytes.len();
        stats.prior_bytes += prior_bytes.len();
    }

    // Parse per (sorted) row across rayon workers, then restore caller order.
    // Each row reads a disjoint prior slice (precomputed prior_offsets) plus one
    // value/moves_left, and finalize_priors is deterministic.
    let mut prior_offsets = Vec::with_capacity(rows.len() + 1);
    let mut running = 0usize;
    prior_offsets.push(0usize);
    for row in rows {
        running += row.legal_ids.len();
        prior_offsets.push(running);
    }
    let parsed: PyResult<Vec<(usize, RustEvaluation)>> = rows
        .par_iter()
        .enumerate()
        .map(|(sorted_index, row)| {
            let value = read_value(value_bytes, sorted_index)?;
            let base = prior_offsets[sorted_index];
            let mut priors = Vec::with_capacity(row.legal_ids.len());
            for (k, &action_id) in row.legal_ids.iter().enumerate() {
                let prior = read_prior(prior_bytes, base + k, sorted_index)?;
                priors.push((action_id, prior));
            }
            finalize_priors(&mut priors, row.legal_ids.len(), sorted_index)?;
            // Build raw logits aligned to the row's legal-id ordering (not the
            // descending-sorted/normalized `priors` order). Each entry pairs an
            // action_id with its logit; ordering is not otherwise significant.
            let logits = match &logits_bytes {
                Some(bytes) => {
                    let mut row_logits = Vec::with_capacity(row.legal_ids.len());
                    for (k, &action_id) in row.legal_ids.iter().enumerate() {
                        let logit = read_logit(bytes, base + k, sorted_index)?;
                        row_logits.push((action_id, logit));
                    }
                    Some(row_logits)
                }
                None => None,
            };
            let moves_left = match &moves_left_bytes {
                Some(bytes) => {
                    let ml = read_f32_required("moves_left_bytes", bytes, sorted_index)?;
                    if !ml.is_finite() || !(0.0..=512.0).contains(&ml) {
                        return Err(PyValueError::new_err(format!(
                            "moves_left_bytes row {sorted_index} must be in [0, 512], got {ml}"
                        )));
                    }
                    Some(ml)
                }
                None => None,
            };
            Ok((
                row.request_index,
                RustEvaluation {
                    value,
                    legal_action_count: row.legal_ids.len(),
                    priors,
                    moves_left,
                    logits,
                },
            ))
        })
        .collect();
    let mut by_request: Vec<Option<RustEvaluation>> = (0..states_len).map(|_| None).collect();
    for (request_index, eval) in parsed? {
        by_request[request_index] = Some(eval);
    }
    if let Some(stats) = stats {
        lock_stats(stats).parse_seconds += parse_started.elapsed().as_secs_f64();
    }
    Ok(by_request
        .into_iter()
        .map(|item| item.expect("every payload row parsed"))
        .collect())
}

fn evaluate_states_chunk(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    request_moves_left: bool,
    request_logits: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let encoding_started = Instant::now();
    let rows = featurize_and_sort(states)?;
    let payload = build_chunk_payload(
        py,
        &rows,
        request_moves_left,
        request_logits,
        encoding_started,
        stats,
    )?;

    let evaluator_started = Instant::now();
    let output = evaluator.call1((payload,))?;
    if let Some(stats) = stats {
        lock_stats(stats).evaluator_seconds += evaluator_started.elapsed().as_secs_f64();
    }
    parse_chunk_reply(
        &output,
        &rows,
        states.len(),
        request_moves_left,
        request_logits,
        stats,
    )
}

/// Async phase 1: featurize and enqueue the forward via `evaluator.submit_payload`
/// (no device sync). Returns the handle plus the row metadata needed to parse
/// the reply later. `finish_states_chunk` drains the result.
fn submit_states_chunk(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    request_moves_left: bool,
    request_logits: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<(Py<PyAny>, Vec<Row>, usize)> {
    let encoding_started = Instant::now();
    let rows = featurize_and_sort(states)?;
    let payload = build_chunk_payload(
        py,
        &rows,
        request_moves_left,
        request_logits,
        encoding_started,
        stats,
    )?;
    let handle = evaluator.call_method1("submit_payload", (payload,))?;
    Ok((handle.unbind(), rows, states.len()))
}

/// Async phase 2: drain a `submit_states_chunk` handle via `evaluator.result`
/// (the device->host sync) and parse it.
fn finish_states_chunk(
    _py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    handle: Py<PyAny>,
    rows: &[Row],
    states_len: usize,
    request_moves_left: bool,
    request_logits: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    let evaluator_started = Instant::now();
    let output = evaluator.call_method1("result", (handle,))?;
    if let Some(stats) = stats {
        lock_stats(stats).evaluator_seconds += evaluator_started.elapsed().as_secs_f64();
    }
    parse_chunk_reply(
        &output,
        rows,
        states_len,
        request_moves_left,
        request_logits,
        stats,
    )
}

fn evaluate_state_refs(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    states: &[&RustHexoState],
    request_moves_left: bool,
    request_logits: bool,
    stats: Option<&SharedEvaluationStats>,
) -> PyResult<Vec<RustEvaluation>> {
    if states.len() > EVAL_CHUNK_STATES {
        let mut evaluations = Vec::with_capacity(states.len());
        for chunk in states.chunks(EVAL_CHUNK_STATES) {
            evaluations.extend(evaluate_states_chunk(
                py,
                evaluator,
                chunk,
                request_moves_left,
                request_logits,
                stats,
            )?);
        }
        return Ok(evaluations);
    }
    evaluate_states_chunk(py, evaluator, states, request_moves_left, request_logits, stats)
}

/// Cache-checked, duplicate-coalescing batch evaluation preserving caller order.
pub fn evaluate_state_refs_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    requests: &[RustEvaluationRequest<'_>],
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    cache_max_states: usize,
    request_moves_left: bool,
    request_logits: bool,
) -> PyResult<Vec<Arc<RustEvaluation>>> {
    let mut result_slots: Vec<Option<Arc<RustEvaluation>>> = vec![None; requests.len()];
    let mut unique_states: Vec<&RustHexoState> = Vec::new();
    let mut unique_keys: Vec<StateHash> = Vec::new();
    let mut unique_index_by_key: HashMap<StateHash, usize> = HashMap::new();
    let mut slot_to_unique: Vec<Option<usize>> = vec![None; requests.len()];
    if let Some(stats) = stats {
        lock_stats(stats).requested_states += requests.len();
    }

    {
        let cached = lock_cache(cache);
        if let Some(stats) = stats {
            let mut stats = lock_stats(stats);
            stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
        }
        for (index, request) in requests.iter().enumerate() {
            let key = request.state_hash;
            if let Some(cached_eval) = cached.get(&key) {
                // A cached eval missing a requested optional field (moves_left
                // or raw logits) cannot serve a request that needs it; treat as
                // a miss so the reply carries the field.
                if (!request_moves_left || cached_eval.moves_left.is_some())
                    && (!request_logits || cached_eval.logits.is_some())
                {
                    result_slots[index] = Some(cached_eval);
                    if let Some(stats) = stats {
                        lock_stats(stats).cache_hits += 1;
                    }
                    continue;
                }
            }
            if unique_index_by_key.contains_key(&key) {
                slot_to_unique[index] = unique_index_by_key.get(&key).copied();
                if let Some(stats) = stats {
                    lock_stats(stats).duplicate_hits += 1;
                }
                continue;
            }
            unique_index_by_key.insert(key, unique_states.len());
            unique_keys.push(key);
            slot_to_unique[index] = Some(unique_states.len());
            unique_states.push(request.state);
        }
    }

    if !unique_states.is_empty() {
        if let Some(stats) = stats {
            lock_stats(stats).unique_states += unique_states.len();
        }
        let unique_evals = evaluate_state_refs(
            py,
            evaluator,
            &unique_states,
            request_moves_left,
            request_logits,
            stats,
        )?;
        let unique_evals: Vec<Arc<RustEvaluation>> = unique_evals
            .into_iter()
            .map(|mut eval| {
                eval.priors.shrink_to_fit();
                Arc::new(eval)
            })
            .collect();
        {
            let mut cached = lock_cache(cache);
            let mut inserted = 0usize;
            for (key, evaluation) in unique_keys.iter().copied().zip(unique_evals.iter()) {
                cached.insert_bounded(key, Arc::clone(evaluation), cache_max_states);
                inserted += 1;
            }
            if let Some(stats) = stats {
                let mut stats = lock_stats(stats);
                stats.cache_inserts += inserted;
                stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
            }
        }
        for (index, unique_index) in slot_to_unique.into_iter().enumerate() {
            if result_slots[index].is_some() {
                continue;
            }
            if let Some(unique_index) = unique_index {
                result_slots[index] = Some(Arc::clone(&unique_evals[unique_index]));
            }
        }
    }

    Ok(result_slots
        .into_iter()
        .map(|item| item.expect("every hexfield evaluation slot must be populated"))
        .collect())
}

/// Insert freshly-evaluated unique evals into the cache and fan them out to the
/// still-empty result slots. Shared by the sync and async cached paths so both
/// produce identical cache state and ordering.
fn integrate_unique_evals(
    unique_evals: Vec<RustEvaluation>,
    unique_keys: &[StateHash],
    slot_to_unique: Vec<Option<usize>>,
    result_slots: &mut [Option<Arc<RustEvaluation>>],
    cache: &SharedEvaluationCache,
    cache_max_states: usize,
    stats: Option<&SharedEvaluationStats>,
) {
    let unique_evals: Vec<Arc<RustEvaluation>> = unique_evals
        .into_iter()
        .map(|mut eval| {
            eval.priors.shrink_to_fit();
            Arc::new(eval)
        })
        .collect();
    {
        let mut cached = lock_cache(cache);
        let mut inserted = 0usize;
        for (key, evaluation) in unique_keys.iter().copied().zip(unique_evals.iter()) {
            cached.insert_bounded(key, Arc::clone(evaluation), cache_max_states);
            inserted += 1;
        }
        if let Some(stats) = stats {
            let mut stats = lock_stats(stats);
            stats.cache_inserts += inserted;
            stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
        }
    }
    for (index, unique_index) in slot_to_unique.into_iter().enumerate() {
        if result_slots[index].is_some() {
            continue;
        }
        if let Some(unique_index) = unique_index {
            result_slots[index] = Some(Arc::clone(&unique_evals[unique_index]));
        }
    }
}

/// GPU work staged by `submit_eval_cached`, completed by `finish_eval_cached`.
enum PendingKind {
    /// Every request was a cache/duplicate hit; no forward to drain.
    None,
    /// One async chunk in flight: drain via `evaluator.result(handle)`.
    Async {
        handle: Py<PyAny>,
        rows: Vec<Row>,
        states_len: usize,
    },
    /// Multi-chunk flush (> EVAL_CHUNK_STATES uniques): evaluated synchronously
    /// at submit time and already parsed.
    Ready(Vec<RustEvaluation>),
}

/// Cache-checked evaluation split into submit/finish phases. Holds the
/// resolved cache hits plus the in-flight GPU work; `finish_eval_cached` drains
/// it. All fields are owned (no borrow of the requests/slots), so the caller
/// may run other work between submit and finish.
pub struct PendingEval {
    result_slots: Vec<Option<Arc<RustEvaluation>>>,
    slot_to_unique: Vec<Option<usize>>,
    unique_keys: Vec<StateHash>,
    request_moves_left: bool,
    request_logits: bool,
    pending: PendingKind,
}

/// Async phase 1 of `evaluate_state_refs_cached`: resolve cache/duplicate hits
/// and enqueue the unique forward (no device sync), returning a `PendingEval`.
/// Call `finish_eval_cached` with the same cache/stats to drain and integrate.
pub fn submit_eval_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    requests: &[RustEvaluationRequest<'_>],
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    request_moves_left: bool,
    request_logits: bool,
) -> PyResult<PendingEval> {
    let mut result_slots: Vec<Option<Arc<RustEvaluation>>> = vec![None; requests.len()];
    let mut unique_states: Vec<&RustHexoState> = Vec::new();
    let mut unique_keys: Vec<StateHash> = Vec::new();
    let mut unique_index_by_key: HashMap<StateHash, usize> = HashMap::new();
    let mut slot_to_unique: Vec<Option<usize>> = vec![None; requests.len()];
    if let Some(stats) = stats {
        lock_stats(stats).requested_states += requests.len();
    }

    {
        let cached = lock_cache(cache);
        if let Some(stats) = stats {
            let mut stats = lock_stats(stats);
            stats.cache_size_peak = stats.cache_size_peak.max(cached.len());
        }
        for (index, request) in requests.iter().enumerate() {
            let key = request.state_hash;
            if let Some(cached_eval) = cached.get(&key) {
                if (!request_moves_left || cached_eval.moves_left.is_some())
                    && (!request_logits || cached_eval.logits.is_some())
                {
                    result_slots[index] = Some(cached_eval);
                    if let Some(stats) = stats {
                        lock_stats(stats).cache_hits += 1;
                    }
                    continue;
                }
            }
            if unique_index_by_key.contains_key(&key) {
                slot_to_unique[index] = unique_index_by_key.get(&key).copied();
                if let Some(stats) = stats {
                    lock_stats(stats).duplicate_hits += 1;
                }
                continue;
            }
            unique_index_by_key.insert(key, unique_states.len());
            unique_keys.push(key);
            slot_to_unique[index] = Some(unique_states.len());
            unique_states.push(request.state);
        }
    }

    let pending = if unique_states.is_empty() {
        PendingKind::None
    } else {
        if let Some(stats) = stats {
            lock_stats(stats).unique_states += unique_states.len();
        }
        if unique_states.len() > EVAL_CHUNK_STATES {
            // Multi-chunk flushes are evaluated synchronously here rather than
            // tracking multiple in-flight handles.
            PendingKind::Ready(evaluate_state_refs(
                py,
                evaluator,
                &unique_states,
                request_moves_left,
                request_logits,
                stats,
            )?)
        } else {
            let (handle, rows, states_len) = submit_states_chunk(
                py,
                evaluator,
                &unique_states,
                request_moves_left,
                request_logits,
                stats,
            )?;
            PendingKind::Async {
                handle,
                rows,
                states_len,
            }
        }
    };

    Ok(PendingEval {
        result_slots,
        slot_to_unique,
        unique_keys,
        request_moves_left,
        request_logits,
        pending,
    })
}

/// Async phase 2: drain the in-flight forward (the device->host sync), insert
/// into the cache, and fan out to the result slots.
pub fn finish_eval_cached(
    py: Python<'_>,
    evaluator: &Bound<'_, PyAny>,
    pending: PendingEval,
    cache: &SharedEvaluationCache,
    stats: Option<&SharedEvaluationStats>,
    cache_max_states: usize,
) -> PyResult<Vec<Arc<RustEvaluation>>> {
    let PendingEval {
        mut result_slots,
        slot_to_unique,
        unique_keys,
        request_moves_left,
        request_logits,
        pending,
    } = pending;

    let unique_evals: Option<Vec<RustEvaluation>> = match pending {
        PendingKind::None => None,
        PendingKind::Ready(evals) => Some(evals),
        PendingKind::Async {
            handle,
            rows,
            states_len,
        } => Some(finish_states_chunk(
            py,
            evaluator,
            handle,
            &rows,
            states_len,
            request_moves_left,
            request_logits,
            stats,
        )?),
    };

    if let Some(unique_evals) = unique_evals {
        integrate_unique_evals(
            unique_evals,
            &unique_keys,
            slot_to_unique,
            &mut result_slots,
            cache,
            cache_max_states,
            stats,
        );
    }

    Ok(result_slots
        .into_iter()
        .map(|item| item.expect("every hexfield evaluation slot must be populated"))
        .collect())
}

/// Validate, descending-sort by prior, and normalize to sum 1. Here
/// `legal_action_count == priors.len()`: the vocabulary is the legal set.
fn finalize_priors(
    priors: &mut Vec<(PackedCoord, f32)>,
    legal_action_count: usize,
    row_index: usize,
) -> PyResult<()> {
    if legal_action_count == 0 {
        if priors.is_empty() {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} priors for terminal row {row_index}",
            priors.len()
        )));
    }
    if priors.is_empty() {
        return Err(PyValueError::new_err(format!(
            "evaluator returned no priors for non-terminal row {row_index}"
        )));
    }
    let mut seen = HashSet::with_capacity(priors.len());
    let mut total = 0.0f32;
    for (action_id, prior) in priors.iter().copied() {
        if !seen.insert(action_id) {
            return Err(PyValueError::new_err(format!(
                "duplicate action {action_id} in row {row_index}"
            )));
        }
        if !prior.is_finite() || prior < 0.0 {
            return Err(PyValueError::new_err(format!(
                "invalid prior {prior} for action {action_id} in row {row_index}"
            )));
        }
        total += prior;
    }
    if total <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "zero total prior mass for row {row_index}"
        )));
    }
    priors.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    for entry in priors.iter_mut() {
        entry.1 /= total;
    }
    Ok(())
}

fn require_exact_bytes(
    name: &str,
    actual_bytes: usize,
    expected_items: usize,
    bytes_per_item: usize,
) -> PyResult<()> {
    let Some(expected_bytes) = expected_items.checked_mul(bytes_per_item) else {
        return Err(PyValueError::new_err(format!(
            "{name} expected byte count overflow"
        )));
    };
    if actual_bytes != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "{name} has {actual_bytes} bytes, expected {expected_bytes}"
        )));
    }
    Ok(())
}

fn read_f32(bytes: &[u8], index: usize) -> Option<f32> {
    let start = index.checked_mul(4)?;
    let chunk = bytes.get(start..start + 4)?;
    Some(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_f32_required(name: &str, bytes: &[u8], index: usize) -> PyResult<f32> {
    read_f32(bytes, index)
        .ok_or_else(|| PyValueError::new_err(format!("{name} missing f32 at item index {index}")))
}

fn read_value(bytes: &[u8], index: usize) -> PyResult<f32> {
    let value = read_f32_required("values_bytes", bytes, index)?;
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        return Err(PyValueError::new_err(format!(
            "values_bytes row {index} must be finite and in [-1, 1], got {value}"
        )));
    }
    Ok(value)
}

fn read_prior(bytes: &[u8], index: usize, row_index: usize) -> PyResult<f32> {
    let value = read_f32_required("priors_bytes", bytes, index)?;
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "priors_bytes row {row_index} entry {index} must be finite and >= 0, got {value}"
        )));
    }
    Ok(value)
}

/// Raw pre-softmax policy logit. Unlike priors, logits are unconstrained in
/// sign (any real value); only finiteness is required.
fn read_logit(bytes: &[u8], index: usize, row_index: usize) -> PyResult<f32> {
    let value = read_f32_required("priors_logits_bytes", bytes, index)?;
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "priors_logits_bytes row {row_index} entry {index} must be finite, got {value}"
        )));
    }
    Ok(value)
}
