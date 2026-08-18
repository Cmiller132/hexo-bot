//! Checkpoint identity, compatibility versions, metadata, and probe hash.

use crate::error::PackageError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The manifest's file name inside a checkpoint directory.
pub const MANIFEST_FILE: &str = "manifest.json";

/// What a checkpoint says about itself.
///
/// The common fields follow `docs/CONTAINER_SPEC.md` §10.
/// [`Manifest::package_metadata`] is opaque to this crate and is written and
/// validated by the package. Architecture-specific data belongs in the
/// package's weight file.
///
/// [`Manifest::validate`] requires every common version to equal the constants
/// linked into the current build. [`Manifest::new`] fills linked-crate versions
/// directly.
///
/// ```
/// use hexo_model::{Manifest, PackageError};
///
/// let dir = tempfile::tempdir()?;
/// let manifest = Manifest::new("mock", 1, 1, 7, 0xfeed_face_dead_beef);
/// manifest.write(dir.path())?;
///
/// let read = Manifest::read(dir.path())?;
/// assert_eq!(read, manifest);
/// read.validate("mock", 1, 1)?;
///
/// // The hash is stored as a fixed-width hexadecimal string.
/// let json = std::fs::read_to_string(dir.path().join("manifest.json"))?;
/// assert!(json.contains("\"0xfeedfacedeadbeef\""));
///
/// // An encoder-version mismatch is reported explicitly.
/// let err = read.validate("mock", 1, 2).expect_err("the encoder moved");
/// assert!(matches!(
///     err,
///     PackageError::EncoderVersion { expected: 2, found: 1 }
/// ));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The registry name of the package that wrote these weights.
    pub package: String,
    /// The package semantics version.
    pub package_version: u32,
    /// The package's encoder version, bumped whenever the bytes its encoder
    /// writes change meaning.
    pub encoder_version: u32,
    /// `hexo_engine::RULES_VERSION` as the writing build linked it.
    pub rules_version: u32,
    /// `hexo_engine::ACTION_ORDER_VERSION` as the writing build linked it.
    pub action_order_version: u32,
    /// `hexo_runner::PROTOCOL_VERSION` as the writing build linked it.
    pub protocol_version: u32,
    /// The checkpoint epoch. [`crate::ModelPackage::init`] writes epoch 0.
    pub epoch: u32,
    /// Package-owned semantic configuration required to interpret the weights.
    ///
    /// This crate neither names nor validates fields inside the JSON value.
    /// Packages with no additional metadata write `{}`.
    pub package_metadata: serde_json::Value,
    /// The probe hash of these weights (`docs/CONTAINER_SPEC.md` §10.2).
    ///
    /// Serialized as a `0x`-prefixed, zero-padded, 16-digit lowercase
    /// hexadecimal string to preserve the full `u64` value across JSON readers.
    #[serde(with = "hex_u64")]
    pub probe_hash: u64,
}

impl Manifest {
    /// Construct a manifest for a checkpoint written by this build.
    ///
    /// Rules, action-order, and protocol versions come from the linked crates.
    #[must_use]
    pub fn new(
        package: &str,
        package_version: u32,
        encoder_version: u32,
        epoch: u32,
        probe_hash: u64,
    ) -> Self {
        Self {
            package: package.to_owned(),
            package_version,
            encoder_version,
            rules_version: hexo_engine::RULES_VERSION,
            action_order_version: hexo_engine::ACTION_ORDER_VERSION,
            protocol_version: hexo_runner::PROTOCOL_VERSION,
            epoch,
            package_metadata: serde_json::json!({}),
            probe_hash,
        }
    }

    /// Attach package-owned semantic metadata to a manifest being written.
    ///
    /// The value is opaque to this crate. A package is responsible for choosing
    /// a stable shape and validating it when loading its checkpoint.
    #[must_use]
    pub fn with_package_metadata(mut self, package_metadata: serde_json::Value) -> Self {
        self.package_metadata = package_metadata;
        self
    }

    /// Write `manifest.json` into an existing `dir`.
    ///
    /// The caller owns checkpoint-directory placement and atomic publication.
    ///
    /// # Errors
    ///
    /// [`PackageError::Io`] if the file cannot be written.
    pub fn write(&self, dir: &Path) -> Result<(), PackageError> {
        let path = dir.join(MANIFEST_FILE);
        let mut json = serde_json::to_string_pretty(self).expect("a manifest serialises");
        json.push('\n');
        std::fs::write(&path, json).map_err(|source| PackageError::Io { path, source })
    }

    /// Read `manifest.json` out of `dir`.
    ///
    /// # Errors
    ///
    /// [`PackageError::Io`] if the file is missing or unreadable, and
    /// [`PackageError::ManifestParse`] if it does not match the manifest schema,
    /// including when it contains unknown fields.
    pub fn read(dir: &Path) -> Result<Self, PackageError> {
        let path = dir.join(MANIFEST_FILE);
        let text = std::fs::read_to_string(&path).map_err(|source| PackageError::Io {
            path: path.clone(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| PackageError::ManifestParse { path, source })
    }

    /// Validate common manifest fields against a package and this build.
    ///
    /// Package identity and versions come from the caller. Linked versions are
    /// compared directly with `hexo_engine::RULES_VERSION`,
    /// `hexo_engine::ACTION_ORDER_VERSION`, and
    /// `hexo_runner::PROTOCOL_VERSION`.
    ///
    /// # Errors
    ///
    /// [`PackageError::PackageName`], [`PackageError::PackageVersion`],
    /// [`PackageError::EncoderVersion`], [`PackageError::RulesVersion`],
    /// [`PackageError::ActionOrderVersion`], or
    /// [`PackageError::ProtocolVersion`], whichever disagrees first. Package
    /// metadata is opaque here and is validated by the owning package after
    /// these common checks.
    pub fn validate(
        &self,
        package_name: &str,
        package_version: u32,
        encoder_version: u32,
    ) -> Result<(), PackageError> {
        if self.package != package_name {
            return Err(PackageError::PackageName {
                expected: package_name.to_owned(),
                found: self.package.clone(),
            });
        }
        if self.package_version != package_version {
            return Err(PackageError::PackageVersion {
                expected: package_version,
                found: self.package_version,
            });
        }
        if self.encoder_version != encoder_version {
            return Err(PackageError::EncoderVersion {
                expected: encoder_version,
                found: self.encoder_version,
            });
        }
        if self.rules_version != hexo_engine::RULES_VERSION {
            return Err(PackageError::RulesVersion {
                expected: hexo_engine::RULES_VERSION,
                found: self.rules_version,
            });
        }
        if self.action_order_version != hexo_engine::ACTION_ORDER_VERSION {
            return Err(PackageError::ActionOrderVersion {
                expected: hexo_engine::ACTION_ORDER_VERSION,
                found: self.action_order_version,
            });
        }
        if self.protocol_version != hexo_runner::PROTOCOL_VERSION {
            return Err(PackageError::ProtocolVersion {
                expected: hexo_runner::PROTOCOL_VERSION,
                found: self.protocol_version,
            });
        }
        Ok(())
    }
}

/// The probe hash's JSON shape: `"0x"` and exactly sixteen lowercase hex digits.
///
/// The reader requires the prefix, width, and hexadecimal syntax exactly.
mod hex_u64 {
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};

    /// Zero-padded to sixteen digits so that every manifest in a tree states the
    /// hash the same width.
    pub(super) fn serialize<S: Serializer>(value: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&format!("{value:#018x}"))
    }

    /// The exact inverse, and nothing else.
    pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        let text = String::deserialize(deserializer)?;
        let digits = text.strip_prefix("0x").ok_or_else(|| {
            D::Error::custom(format!(
                "probe hash {text:?} does not start with `0x`; the format is `0x` and sixteen \
                 lowercase hex digits"
            ))
        })?;
        if digits.len() != 16 || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(D::Error::custom(format!(
                "probe hash {text:?} is not `0x` followed by sixteen hex digits"
            )));
        }
        u64::from_str_radix(digits, 16).map_err(D::Error::custom)
    }
}
