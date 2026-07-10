//! Rust parallel serve-pack with zero-copy buffers.
//!
//! `build_serve_groups` takes the CSR-flat request (f16 feats / i16 coords /
//! u16 nbr / u8 raylen) plus the per-row offsets, runs the `plan_groups` boundary planner,
//! and assembles the padded per-group buffers in parallel (rayon), exposing
//! them as read-only zero-copy `#[pyclass]` buffers consumed Python-side via
//! `torch.frombuffer(buf, ...).to(device)`.
//!
//! Pad rules:
//!   - feats pad rows  = 0  (`f16::ZERO`)
//!   - nbr fill        = `pad_to`; `NBR_SENTINEL (0xFFFF)` -> `pad_to`
//!   - mask            = `1` at real nodes, else `0`
//!   - coords          = 0 at pad rows
//!   - raylen pad rows = 0 (a 0-length ray masks everything but the diagonal)
//! Indices are emitted as int32; `payload.rs` asserts
//! `num_nodes <= NBR_SENTINEL`, so every index fits i32. Python casts to int64
//! on the GPU.

use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use half::f16;

use crate::constants::{num_features, RAYLEN_SLOTS};

const NBR_SENTINEL: u16 = 0xFFFF;
// Node-count quantization for pad_to; mirrors inference.py.
const QUANT_NODES: usize = 64;
// Token count added in the (pad_to + NUM_TOKENS)^2 pair-ceiling test.
const NUM_TOKENS: usize = 8;
const PAIR_CEILING: f64 = 3.8e7;
const WASTE_FRACTION: f64 = 0.18;

/// 64-quantized ceiling of `n`: `max(64, div_ceil(n, 64) * 64)`. Returns 64
/// for `n == 0`.
fn ceil_quant(n: usize) -> usize {
    let q = n.div_ceil(QUANT_NODES) * QUANT_NODES;
    QUANT_NODES.max(q)
}

/// Group boundary planner. Rows arrive size-descending. Returns
/// (start, end, pad_to) groups in ascending row order.
pub fn plan_groups(sizes: &[usize]) -> Vec<(usize, usize, usize)> {
    let n = sizes.len();
    let mut groups: Vec<(usize, usize, usize)> = Vec::new();
    let mut start = 0usize;
    while start < n {
        let pad_to = ceil_quant(sizes[start]);
        // floor = pad_to - max(QUANT_NODES, trunc(WASTE_FRACTION * pad_to)).
        // The f64 product is non-negative, so `as usize` truncates toward zero.
        let waste = (WASTE_FRACTION * pad_to as f64) as usize;
        let floor = pad_to - QUANT_NODES.max(waste);
        let mut end = start + 1;
        while end < n {
            // (end - start + 1) is the candidate group size INCLUDING sizes[end].
            let g = (end - start + 1) as f64;
            let pair = g * ((pad_to + NUM_TOKENS) * (pad_to + NUM_TOKENS)) as f64;
            if pair > PAIR_CEILING {
                break;
            }
            if sizes[end] < floor {
                break;
            }
            end += 1;
        }
        groups.push((start, end, pad_to));
        start = end;
    }
    groups
}

// --- Zero-copy buffers --------------------------------------------------------

macro_rules! plane_buffer {
    ($name:ident, $ty:ty) => {
        #[pyclass]
        pub struct $name {
            data: Vec<$ty>,
        }

        #[pymethods]
        impl $name {
            fn __len__(&self) -> usize {
                self.data.len() * std::mem::size_of::<$ty>()
            }

            /// SAFETY: populate the CPython-supplied `Py_buffer` with a read-only
            /// 1-D byte view over `data`, keeping `slf` alive via `view.obj`.
            unsafe fn __getbuffer__(
                slf: Bound<'_, Self>,
                view: *mut ffi::Py_buffer,
                flags: c_int,
            ) -> PyResult<()> {
                if view.is_null() {
                    return Err(PyBufferError::new_err("buffer view is null"));
                }
                if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
                    (*view).obj = ptr::null_mut();
                    return Err(PyBufferError::new_err("buffer is read-only"));
                }
                let guard = slf.borrow();
                let data = &guard.data;
                (*view).buf = data.as_ptr() as *mut c_void;
                (*view).len =
                    (data.len() * std::mem::size_of::<$ty>()) as ffi::Py_ssize_t;
                (*view).readonly = 1;
                (*view).itemsize = 1;
                (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                    b"B\0".as_ptr() as *mut c_char
                } else {
                    ptr::null_mut()
                };
                (*view).ndim = 1;
                (*view).shape = ptr::null_mut();
                (*view).strides = ptr::null_mut();
                (*view).suboffsets = ptr::null_mut();
                (*view).internal = ptr::null_mut();
                (*view).obj = slf.clone().into_any().into_ptr();
                Ok(())
            }

            unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
        }
    };
}

plane_buffer!(F16Buf, f16); // feats:  g*pad_to*NUM_FEATURES
plane_buffer!(I32Buf, i32); // nbr: g*pad_to*6 ; coords: g*pad_to*2
plane_buffer!(U8Buf, u8); //  mask:   g*pad_to (0/1)

/// One assembled group's buffers + shape, ready to hand to Python.
struct GroupBufs {
    start: usize,
    end: usize,
    pad_to: usize,
    g: usize,
    feats: Vec<f16>,
    nbr: Vec<i32>,
    mask: Vec<u8>,
    coords: Vec<i32>,
    raylen: Vec<u8>,
}

/// Assemble all groups' padded buffers in parallel (one worker per group),
/// applying the pad rules documented in the module header.
///
/// `feats` is the CSR-flat f16 node features (total_nodes * NUM_FEATURES),
/// `qr` the i16 coords (total_nodes * 2), `nbr` the u16 row-local neighbours
/// (total_nodes * 6, sentinel 0xFFFF), `offsets` the per-row node offsets
/// (len b+1), `sizes` the per-row node counts (len b). All in size-descending
/// row order.
fn assemble_groups(
    groups: &[(usize, usize, usize)],
    feats: &[f16],
    qr: &[i16],
    nbr: &[u16],
    raylen: &[u8],
    offsets: &[usize],
    sizes: &[usize],
) -> Vec<GroupBufs> {
    let nf = num_features();
    groups
        .par_iter()
        .map(|&(start, end, pad_to)| {
            let g = end - start;
            let pad_i32 = pad_to as i32;

            let mut f = vec![f16::ZERO; g * pad_to * nf];
            let mut nb = vec![pad_i32; g * pad_to * 6];
            let mut m = vec![0u8; g * pad_to];
            let mut c = vec![0i32; g * pad_to * 2];
            let mut rl = vec![0u8; g * pad_to * RAYLEN_SLOTS];

            for k in 0..g {
                let row = start + k;
                let n = sizes[row];
                let o = offsets[row];

                // feats: [k, :n, :] = feats[o:o+n]; rest already f16::ZERO.
                let src = &feats[o * nf..(o + n) * nf];
                let dst = &mut f[k * pad_to * nf..k * pad_to * nf + n * nf];
                dst.copy_from_slice(src);

                // nbr: [k, :n, :] = where(row==SENTINEL, pad_to, row); rest = pad_to.
                let nsrc = &nbr[o * 6..(o + n) * 6];
                let nbase = k * pad_to * 6;
                for (i, &j) in nsrc.iter().enumerate() {
                    nb[nbase + i] = if j == NBR_SENTINEL { pad_i32 } else { j as i32 };
                }

                // mask: [k, :n] = 1; rest already 0.
                for i in 0..n {
                    m[k * pad_to + i] = 1;
                }

                // coords: [k, :n, :] = qr[o:o+n]; rest already 0.
                let qsrc = &qr[o * 2..(o + n) * 2];
                let cbase = k * pad_to * 2;
                for (i, &v) in qsrc.iter().enumerate() {
                    c[cbase + i] = v as i32;
                }

                // raylen: [k, :n, :] = raylen[o:o+n]; rest already 0.
                let rsrc = &raylen[o * RAYLEN_SLOTS..(o + n) * RAYLEN_SLOTS];
                let rbase = k * pad_to * RAYLEN_SLOTS;
                rl[rbase..rbase + n * RAYLEN_SLOTS].copy_from_slice(rsrc);
            }

            GroupBufs {
                start,
                end,
                pad_to,
                g,
                feats: f,
                nbr: nb,
                mask: m,
                coords: c,
                raylen: rl,
            }
        })
        .collect()
}

/// Parse the CSR-flat request, plan the groups, assemble the padded per-group
/// buffers in parallel (GIL released), then build the Python group dict list.
///
/// `feats_bytes`: f16 LE, total_nodes*NUM_FEATURES. `qr_bytes`: i16 LE,
/// total_nodes*2. `nbr_bytes`: u16 LE, total_nodes*6. `raylen_bytes`: u8,
/// total_nodes*RAYLEN_SLOTS. `offsets`: i64 row offsets (len b+1,
/// offsets[-1]==total_nodes). Returns a list of dicts:
/// {start,end,pad_to,g, feats:F16Buf, nbr:I32Buf, mask:U8Buf, coords:I32Buf,
/// raylen:U8Buf}.
#[pyfunction]
pub fn build_serve_groups<'py>(
    py: Python<'py>,
    feats_bytes: &[u8],
    qr_bytes: &[u8],
    nbr_bytes: &[u8],
    raylen_bytes: &[u8],
    offsets: Vec<i64>,
) -> PyResult<Bound<'py, PyList>> {
    if offsets.is_empty() {
        return Err(PyValueError::new_err("offsets must be non-empty (b+1)"));
    }
    let b = offsets.len() - 1;
    let total_nodes = *offsets.last().unwrap() as usize;
    let nf = num_features();

    // Reinterpret the wire bytes as typed slices; assumes native (LE) byte order.
    if feats_bytes.len() != total_nodes * nf * std::mem::size_of::<f16>() {
        return Err(PyValueError::new_err("feats byte count mismatch"));
    }
    if qr_bytes.len() != total_nodes * 2 * std::mem::size_of::<i16>() {
        return Err(PyValueError::new_err("qr byte count mismatch"));
    }
    if nbr_bytes.len() != total_nodes * 6 * std::mem::size_of::<u16>() {
        return Err(PyValueError::new_err("nbr byte count mismatch"));
    }
    if raylen_bytes.len() != total_nodes * RAYLEN_SLOTS {
        return Err(PyValueError::new_err("raylen byte count mismatch"));
    }
    // SAFETY: lengths checked above; f16/i16/u16 are POD with no invalid bit
    // patterns; the source byte buffers are alive for the call and aligned
    // (element align <= 2, contiguous arrays). Copied into owned Vecs.
    let feats: Vec<f16> = unsafe {
        std::slice::from_raw_parts(feats_bytes.as_ptr() as *const f16, total_nodes * nf)
    }
    .to_vec();
    let qr: Vec<i16> = unsafe {
        std::slice::from_raw_parts(qr_bytes.as_ptr() as *const i16, total_nodes * 2)
    }
    .to_vec();
    let nbr: Vec<u16> = unsafe {
        std::slice::from_raw_parts(nbr_bytes.as_ptr() as *const u16, total_nodes * 6)
    }
    .to_vec();
    let raylen: Vec<u8> = raylen_bytes.to_vec();

    let offsets_us: Vec<usize> = offsets.iter().map(|&o| o as usize).collect();
    let sizes: Vec<usize> = (0..b).map(|i| offsets_us[i + 1] - offsets_us[i]).collect();

    let groups = plan_groups(&sizes);

    // Parallel pad with the GIL released.
    let assembled = py.detach(|| {
        assemble_groups(&groups, &feats, &qr, &nbr, &raylen, &offsets_us, &sizes)
    });

    let out = PyList::empty(py);
    for gb in assembled {
        let d = PyDict::new(py);
        d.set_item("start", gb.start)?;
        d.set_item("end", gb.end)?;
        d.set_item("pad_to", gb.pad_to)?;
        d.set_item("g", gb.g)?;
        d.set_item("feats", Py::new(py, F16Buf { data: gb.feats })?)?;
        d.set_item("nbr", Py::new(py, I32Buf { data: gb.nbr })?)?;
        d.set_item("mask", Py::new(py, U8Buf { data: gb.mask })?)?;
        d.set_item("coords", Py::new(py, I32Buf { data: gb.coords })?)?;
        d.set_item("raylen", Py::new(py, U8Buf { data: gb.raylen })?)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Exposes `plan_groups` to Python. Returns the (start, end, pad_to) tuples.
#[pyfunction]
pub fn debug_plan_groups(sizes: Vec<usize>) -> Vec<(usize, usize, usize)> {
    plan_groups(&sizes)
}
