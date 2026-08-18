//! Runtime-independent MantisNet forward boundary.
//!
//! Defines batch and output types; concrete runtime adapters implement
//! [`Forward`] and [`ForwardLoader`].

use crate::encoder::RawBatch;
use std::path::Path;

/// Boxed error type for forward and load operations.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// The two MantisNet cell-head outputs, concatenated by position.
///
/// Both arrays are in engine canonical legal order within each position. Their
/// ragged row boundaries are [`RawBatch::legal_offsets`] from the input batch.
#[derive(Clone, Debug, PartialEq)]
pub struct RawOutputs {
    /// Raw policy logits, one per legal action.
    pub policy_logits: Vec<f32>,
    /// Bounded action values in `[-1, 1]`, one per legal action.
    pub q_values: Vec<f32>,
}

/// One loaded MantisNet module, called once per collated evaluator batch.
///
/// Implementations convert [`RawBatch`] arrays to runtime tensors and return
/// plain Rust vectors.
pub trait Forward: Send {
    /// Run both cell heads for `batch`.
    ///
    /// Implementations must preserve the input's concatenated canonical legal
    /// order. The evaluator validates both output lengths and every value before
    /// using them.
    fn forward(&mut self, batch: &RawBatch) -> Result<RawOutputs, BoxError>;
}

/// Load a MantisNet checkpoint into a [`Forward`] implementation.
pub trait ForwardLoader: Send + Sync {
    /// Load `weights`, including all package/runtime version checks.
    fn load(&self, weights: &Path) -> Result<Box<dyn Forward>, BoxError>;
}
