//! The MantisNet model package.
//!
//! [`encoder`] implements the position representation used by both the PyO3
//! extension and the container package.

mod config;
pub mod encoder;
mod forward;
pub mod improvement;
mod package;
mod seam;
mod select;

pub use forward::{BoxError, Forward, ForwardLoader, RawOutputs};
pub use package::{MantisPackage, WEIGHTS_FILE};

/// Version of the MantisNet position representation.
///
/// Bumped whenever the bytes or index tables produced by [`encoder`] change
/// meaning. Checkpoints are not compatible across versions.
pub const MODEL_REPR_VERSION: u32 = 7;

/// Version of MantisNet package semantics outside the representation.
///
/// This covers KLENT improvement, session choices, diagnostics bytes, and
/// checkpoint metadata independently of [`MODEL_REPR_VERSION`].
pub const PACKAGE_VERSION: u32 = 1;

/// The registry and checkpoint name of this package.
pub const PACKAGE_NAME: &str = "mantisnet";
