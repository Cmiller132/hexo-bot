//! The model-package interface consumed by the container.

use crate::error::PackageError;
use crate::manifest::Manifest;
use hexo_search::{DecisionSession, Encoder, Evaluator};
use std::path::{Path, PathBuf};

/// A model package exposed to the container.
///
/// The package owns its encoder, evaluator, session policies, diagnostics,
/// checkpoint format, and fitting implementation. All evaluations must follow
/// [`hexo_search::Evaluation`]: priors use canonical legal-action order and
/// value uses the evaluated position's side-to-move perspective.
///
/// Implementations must provide distinct self-play and evaluation session
/// constructors. [`ModelPackage::variant_session`] may reject all names through
/// its default implementation.
///
/// A successful [`ModelPackage::load`] validates common and package-owned
/// metadata, loads the weights, and verifies their probe hash. Callers obtain a
/// fresh evaluator after each load; the validity of older evaluator handles is
/// package-defined.
///
/// Package-created sessions must use distinct initial streams. Drivers may
/// replace those streams through [`DecisionSession::reseed`].
pub trait ModelPackage {
    /// The registry name written into shard headers and checkpoint manifests.
    fn name(&self) -> &'static str;

    /// The package semantics version.
    ///
    /// This version is independent of the encoder byte-layout version.
    fn package_version(&self) -> u32;

    /// The version of the bytes [`ModelPackage::encoder`] writes.
    ///
    /// Implementations increment it whenever the bytes change shape, order, or
    /// meaning. Loading refuses a checkpoint with a different version.
    fn encoder_version(&self) -> u32;

    /// Write an epoch-0 checkpoint into `dir`.
    ///
    /// The manifest must contain the probe hash computed from the written
    /// weights. This operation does not load the checkpoint.
    ///
    /// # Errors
    ///
    /// [`PackageError::Io`] if the directory or its files cannot be written, and
    /// whatever the package's own initialisation can fail with.
    fn init(&self, dir: &Path) -> Result<Manifest, PackageError>;

    /// Load and verify the checkpoint in `dir`.
    ///
    /// Implementations validate the manifest and package metadata, load a
    /// candidate, and compare its computed probe hash with the manifest. On
    /// error, the previously loaded state must remain unchanged.
    ///
    /// # Errors
    ///
    /// [`PackageError::Io`] or [`PackageError::ManifestParse`] if the checkpoint
    /// cannot be read, any of the version variants if it disagrees with this
    /// build, and [`PackageError::ProbeMismatch`] if the weights do not answer
    /// the way the manifest says they do.
    fn load(&mut self, dir: &Path) -> Result<Manifest, PackageError>;

    /// Return the package's encoder.
    ///
    /// The encoder is available before weights are loaded and may be shared
    /// across workers.
    fn encoder(&self) -> Box<dyn Encoder>;

    /// Return an evaluator for the currently loaded weights.
    ///
    /// Callers must request a new evaluator after each successful load.
    ///
    /// # Errors
    ///
    /// [`PackageError::NotLoaded`] before a successful [`ModelPackage::load`].
    fn evaluator(&self) -> Result<Box<dyn Evaluator>, PackageError>;

    /// Return a session using the package's self-play policy.
    ///
    /// # Errors
    ///
    /// [`PackageError::NotLoaded`] if the package needs weights it does not
    /// have, and [`PackageError::InvalidConfig`] if its configured search shape
    /// cannot be built.
    fn self_play_session(&self) -> Result<Box<dyn DecisionSession>, PackageError>;

    /// Return a session using the package's evaluation policy.
    ///
    /// # Errors
    ///
    /// As [`ModelPackage::self_play_session`].
    fn eval_session(&self) -> Result<Box<dyn DecisionSession>, PackageError>;

    /// Return a named session variant for comparisons or benchmark matches.
    ///
    /// Names and syntax are package-defined. The default rejects every name.
    ///
    /// # Errors
    ///
    /// [`PackageError::UnknownVariant`] for a name the package does not define,
    /// which is what the default always returns.
    fn variant_session(&self, name: &str) -> Result<Box<dyn DecisionSession>, PackageError> {
        Err(PackageError::UnknownVariant {
            package: self.name(),
            variant: name.to_owned(),
        })
    }

    /// Consume record shards and write checkpoint `epoch` into `out_dir`.
    ///
    /// The package owns the objective, optimiser, and data pipeline. A
    /// successful implementation must consume at least one game and write a
    /// manifest containing the probe hash of the output weights. Writing does
    /// not load the new checkpoint.
    ///
    /// Packages without container-side fitting return
    /// [`PackageError::Unsupported`].
    ///
    /// # Errors
    ///
    /// [`PackageError::NoTrainingData`] if there were no shards or no games,
    /// [`PackageError::Io`] for a shard or checkpoint the filesystem refuses,
    /// [`PackageError::Failed`] wrapping a package-owned reader or trainer
    /// failure, [`PackageError::NotLoaded`] when required source weights are not
    /// loaded, or [`PackageError::Unsupported`] when fitting is unavailable.
    fn fit(
        &mut self,
        shards: &[PathBuf],
        out_dir: &Path,
        epoch: u32,
    ) -> Result<Manifest, PackageError>;
}
