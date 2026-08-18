//! Model-package failure types.

use std::path::PathBuf;

/// Everything a [`crate::ModelPackage`] can refuse to do.
///
/// Version errors carry expected and found values. [`PackageError::Failed`]
/// preserves a package-owned source error for inspection or downcasting, while
/// [`PackageError::NoTrainingData`] reports empty fitting inputs explicitly.
#[derive(Debug)]
pub enum PackageError {
    /// The filesystem refused an operation.
    Io {
        /// The file or directory it was refused on.
        path: PathBuf,
        /// What the filesystem said.
        source: std::io::Error,
    },
    /// A `manifest.json` is not JSON, or is not a [`crate::Manifest`].
    ///
    /// Missing and unknown fields are parse errors.
    ManifestParse {
        /// The manifest that could not be read.
        path: PathBuf,
        /// What the deserialiser said, with its line and column.
        source: serde_json::Error,
    },
    /// An artefact names a different package than the one running.
    ///
    /// Raised for checkpoint manifests and training shards whose package name
    /// differs from the active package.
    PackageName {
        /// The running package's registry name.
        expected: String,
        /// What the artefact states.
        found: String,
    },
    /// The checkpoint was written by a different version of this package.
    PackageVersion {
        /// What this build is.
        expected: u32,
        /// What the checkpoint states.
        found: u32,
    },
    /// The weights require a different encoder representation version.
    EncoderVersion {
        /// What this build's encoder is.
        expected: u32,
        /// What the checkpoint states.
        found: u32,
    },
    /// The weights were produced under different rules than this build links.
    RulesVersion {
        /// `hexo_engine::RULES_VERSION` as linked.
        expected: u32,
        /// What the checkpoint states.
        found: u32,
    },
    /// The weights require a different canonical action ordering.
    ActionOrderVersion {
        /// `hexo_engine::ACTION_ORDER_VERSION` as linked.
        expected: u32,
        /// What the checkpoint states.
        found: u32,
    },
    /// The weights were produced under a different runner decision and result
    /// model than this build links.
    ProtocolVersion {
        /// `hexo_runner::PROTOCOL_VERSION` as linked.
        expected: u32,
        /// What the checkpoint states.
        found: u32,
    },
    /// Package-owned checkpoint metadata disagrees with this package instance.
    ///
    /// The JSON values are opaque to this crate.
    PackageMetadata {
        /// What the running package requires.
        expected: serde_json::Value,
        /// What the checkpoint states.
        found: serde_json::Value,
    },
    /// The loaded weights' computed probe hash differs from the manifest.
    ProbeMismatch {
        /// What the manifest promised.
        expected: u64,
        /// What the loaded weights actually produced.
        computed: u64,
    },
    /// Something that needs weights was asked for before [`crate::ModelPackage::load`]
    /// succeeded.
    NotLoaded {
        /// The package that has no weights.
        package: &'static str,
    },
    /// The package has no session variant by that name.
    ///
    /// [`crate::ModelPackage::variant_session`] returns this by default.
    UnknownVariant {
        /// The package that was asked.
        package: &'static str,
        /// The name it does not have.
        variant: String,
    },
    /// The package's configuration string is not one it can use.
    ///
    /// The package supplies the configuration-specific description.
    InvalidConfig {
        /// The package that refused.
        package: &'static str,
        /// What is wrong with the string, in the package's own words.
        problem: String,
    },
    /// A weight file is present but does not hold weights this build can read.
    ///
    /// The package supplies the format-specific description. Structured loader
    /// errors may instead use [`PackageError::Failed`].
    MalformedWeights {
        /// The weight file.
        path: PathBuf,
        /// What is wrong with it, in the package's own words.
        problem: String,
    },
    /// A `fit` was handed nothing to fit on.
    ///
    /// Carries both shard and game counts.
    NoTrainingData {
        /// The package whose fit refused.
        package: &'static str,
        /// How many shards it was handed.
        shards: usize,
        /// How many games it found in them.
        games: usize,
    },
    /// The package does not implement this operation.
    Unsupported {
        /// The package that declined.
        package: &'static str,
        /// The operation it does not implement.
        operation: &'static str,
        /// Why the package declines it.
        reason: &'static str,
    },
    /// A package-internal operation failed, carrying the package's own error.
    ///
    /// The boxed source preserves its concrete type for callers that can
    /// downcast it.
    Failed {
        /// The package that failed.
        package: &'static str,
        /// What it was doing, as a gerund clause: `"reading a record shard"`.
        doing: &'static str,
        /// The error it failed with, intact.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl PackageError {
    /// Wrap a package-internal error as [`PackageError::Failed`].
    ///
    /// The source is boxed without converting it to a string.
    pub fn failed<E>(package: &'static str, doing: &'static str, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Failed {
            package,
            doing,
            source: Box::new(source),
        }
    }
}

impl core::fmt::Display for PackageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
            Self::ManifestParse { path, source } => {
                write!(f, "{}: {source}", path.display())
            }
            Self::PackageName { expected, found } => write!(
                f,
                "the artefact was produced by package {found:?}, but this build runs {expected:?}"
            ),
            Self::PackageVersion { expected, found } => write!(
                f,
                "the checkpoint was written by package version {found}, but this build is version \
                 {expected}"
            ),
            Self::EncoderVersion { expected, found } => write!(
                f,
                "the weights were trained against encoder version {found}, but this build's \
                 encoder is version {expected}"
            ),
            Self::RulesVersion { expected, found } => write!(
                f,
                "the checkpoint states rules version {found}, but this build links rules version \
                 {expected}"
            ),
            Self::ActionOrderVersion { expected, found } => write!(
                f,
                "the checkpoint states action-order version {found}, but this build links \
                 action-order version {expected}"
            ),
            Self::ProtocolVersion { expected, found } => write!(
                f,
                "the checkpoint states runner protocol version {found}, but this build links \
                 protocol version {expected}"
            ),
            Self::PackageMetadata { expected, found } => write!(
                f,
                "the checkpoint's package metadata {found} does not match the running package's \
                 required metadata {expected}"
            ),
            Self::ProbeMismatch { expected, computed } => write!(
                f,
                "the loaded weights answer the probe with {computed:#018x}, but the manifest \
                 promises {expected:#018x}; these are not the weights the checkpoint describes"
            ),
            Self::NotLoaded { package } => write!(
                f,
                "{package} has no weights loaded; a checkpoint has to be loaded before anything \
                 can answer"
            ),
            Self::UnknownVariant { package, variant } => {
                write!(f, "{package} has no session variant named {variant:?}")
            }
            Self::InvalidConfig { package, problem } => {
                write!(f, "{package} cannot use this configuration: {problem}")
            }
            Self::MalformedWeights { path, problem } => {
                write!(f, "{}: {problem}", path.display())
            }
            Self::NoTrainingData {
                package,
                shards,
                games,
            } => write!(
                f,
                "{package} was asked to fit on {shards} shard(s) holding {games} game(s); a fit \
                 that consumed nothing would produce weights nothing trained"
            ),
            Self::Unsupported {
                package,
                operation,
                reason,
            } => write!(f, "{package} does not support {operation}: {reason}"),
            Self::Failed {
                package,
                doing,
                source,
            } => write!(f, "{package} failed {doing}: {source}"),
        }
    }
}

impl std::error::Error for PackageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::ManifestParse { source, .. } => Some(source),
            Self::Failed { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}
