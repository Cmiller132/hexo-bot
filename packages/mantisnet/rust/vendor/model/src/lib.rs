//! Model-package interfaces, checkpoint manifests, and evaluator probes.
//!
//! Each crate under `crates/models/<name>/` implements [`ModelPackage`]. The
//! container uses that trait and its name registry without interpreting model
//! architecture.
//!
//! - [`ModelPackage`] defines package lifecycle and session construction.
//! - [`Manifest`] records package identity, compatibility versions, epoch,
//!   package metadata, and probe hash.
//! - [`probe_hash`] hashes exact evaluator outputs over the fixed probe set.
//!
//! ```
//! use hexo_model::{Manifest, probe_positions};
//!
//! // Producers write manifests; consumers validate them before use.
//! let manifest = Manifest::new("mock", 1, 1, 0, 0x0123_4567_89ab_cdef);
//! manifest.validate("mock", 1, 1)?;
//! assert!(manifest.validate("gnn", 1, 1).is_err());
//!
//! // Every fixed probe position is live and has legal actions.
//! for position in probe_positions() {
//!     assert!(!position.is_terminal());
//!     assert!(position.legal_count() > 0);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod error;
pub mod manifest;
pub mod package;
pub mod probe;

pub use error::PackageError;
pub use manifest::{MANIFEST_FILE, Manifest};
pub use package::ModelPackage;
pub use probe::{probe_hash, probe_positions};
