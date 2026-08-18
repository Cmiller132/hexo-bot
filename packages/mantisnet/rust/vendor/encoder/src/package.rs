//! The complete MantisNet package: configuration, checkpoints, and sessions.

use crate::config::{self, Config, Search};
use crate::forward::{Forward, ForwardLoader};
use crate::seam::{MantisEncoder, MantisEvaluator};
use crate::select::{ActingPolicy, MaxVisits};
use crate::{MODEL_REPR_VERSION, PACKAGE_NAME, PACKAGE_VERSION};
use hexo_model::{Manifest, ModelPackage, PackageError, probe_hash};
use hexo_search::{
    DecisionSession, Encoder, Evaluator, GumbelConfig, GumbelSession, MctsSession, PolicySession,
};
use serde_json::json;
use std::cell::Cell;
use std::num::{NonZeroU32, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// The training checkpoint copied into a sealed container checkpoint.
pub const WEIGHTS_FILE: &str = "weights.pt";

/// The evaluation budget chosen for this package.
const EVAL_SIMULATIONS: NonZeroU32 = NonZeroU32::new(32).expect("32 is nonzero");

/// Root candidates under the package's fixed evaluation mode.
const EVAL_CANDIDATES: NonZeroUsize = NonZeroUsize::new(16).expect("16 is nonzero");

/// Probe-verified loaded state.
struct Loaded {
    forward: Arc<Mutex<Box<dyn Forward>>>,
    probe_hash: u64,
}

/// MantisNet as the generic container sees it.
pub struct MantisPackage {
    config: Config,
    loader: Arc<dyn ForwardLoader>,
    loaded: Option<Loaded>,
    next_serial: Cell<u64>,
}

impl MantisPackage {
    /// Build from `tau=F,lambda=F[,source=PATH]` and an injected runtime loader.
    ///
    /// `source` is required only by [`ModelPackage::init`], where a raw Python
    /// training checkpoint is sealed with its container manifest. Normal loads
    /// read `weights.pt` from an already sealed checkpoint directory.
    pub fn from_config(config: &str, loader: Arc<dyn ForwardLoader>) -> Result<Self, PackageError> {
        Ok(Self {
            config: config::parse_config(config)?,
            loader,
            loaded: None,
            next_serial: Cell::new(0),
        })
    }

    fn metadata(&self) -> serde_json::Value {
        // Round-trip through f32 decimal to preserve numeric equality in manifests.
        let tau: f64 = self
            .config
            .tau
            .to_string()
            .parse()
            .expect("a finite f32 decimal is a finite f64");
        let lambda: f64 = self
            .config
            .lambda
            .to_string()
            .parse()
            .expect("a finite f32 decimal is a finite f64");
        json!({
            "lambda": lambda,
            "tau": tau,
        })
    }

    fn load_forward(&self, weights: &Path) -> Result<Box<dyn Forward>, PackageError> {
        self.loader
            .load(weights)
            .map_err(|source| PackageError::Failed {
                package: PACKAGE_NAME,
                doing: "loading the Torch checkpoint",
                source,
            })
    }

    fn candidate(&self, weights: &Path) -> Result<Loaded, PackageError> {
        let forward = Arc::new(Mutex::new(self.load_forward(weights)?));
        let mut evaluator =
            MantisEvaluator::new(Arc::clone(&forward), self.config.tau, self.config.lambda);
        let probe_hash = probe_hash(&MantisEncoder, &mut evaluator);
        Ok(Loaded {
            forward,
            probe_hash,
        })
    }

    fn loaded(&self) -> Result<&Loaded, PackageError> {
        self.loaded.as_ref().ok_or(PackageError::NotLoaded {
            package: PACKAGE_NAME,
        })
    }

    fn seed(&self) -> Result<u64, PackageError> {
        let loaded = self.loaded()?;
        let serial = self.next_serial.get();
        self.next_serial.set(serial.wrapping_add(1));
        Ok(mix(loaded.probe_hash
            ^ serial
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left(23)))
    }

    fn session(
        &self,
        search: Search,
        record_diagnostics: bool,
    ) -> Result<Box<dyn DecisionSession>, PackageError> {
        let seed = self.seed()?;
        Ok(match search {
            Search::Policy => Box::new(PolicySession::new(
                Box::new(ActingPolicy { record_diagnostics }),
                seed,
            )),
            Search::Mcts(config) => Box::new(MctsSession::new(config, Box::new(MaxVisits), seed)),
            Search::Gumbel {
                simulations,
                candidates,
                temperature,
            } => Box::new(GumbelSession::new(
                GumbelConfig {
                    simulations,
                    candidates,
                    temperature,
                },
                seed,
            )),
        })
    }
}

impl ModelPackage for MantisPackage {
    fn name(&self) -> &'static str {
        PACKAGE_NAME
    }

    fn package_version(&self) -> u32 {
        PACKAGE_VERSION
    }

    fn encoder_version(&self) -> u32 {
        MODEL_REPR_VERSION
    }

    fn init(&self, dir: &Path) -> Result<Manifest, PackageError> {
        let source = self
            .config
            .source
            .as_ref()
            .ok_or_else(|| PackageError::InvalidConfig {
                package: PACKAGE_NAME,
                problem: "`init` needs `source=PATH` naming the Python training checkpoint to seal"
                    .to_owned(),
            })?;
        std::fs::create_dir_all(dir).map_err(|source| PackageError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let weights = dir.join(WEIGHTS_FILE);
        std::fs::copy(source, &weights).map_err(|source| PackageError::Io {
            path: weights.clone(),
            source,
        })?;

        // Probe the sealed checkpoint.
        let candidate = self.candidate(&weights)?;
        let manifest = Manifest::new(
            PACKAGE_NAME,
            PACKAGE_VERSION,
            MODEL_REPR_VERSION,
            0,
            candidate.probe_hash,
        )
        .with_package_metadata(self.metadata());
        manifest.write(dir)?;
        Ok(manifest)
    }

    fn load(&mut self, dir: &Path) -> Result<Manifest, PackageError> {
        let manifest = Manifest::read(dir)?;
        manifest.validate(PACKAGE_NAME, PACKAGE_VERSION, MODEL_REPR_VERSION)?;
        let expected = self.metadata();
        if manifest.package_metadata != expected {
            return Err(PackageError::PackageMetadata {
                expected,
                found: manifest.package_metadata,
            });
        }

        let candidate = self.candidate(&dir.join(WEIGHTS_FILE))?;
        if candidate.probe_hash != manifest.probe_hash {
            return Err(PackageError::ProbeMismatch {
                expected: manifest.probe_hash,
                computed: candidate.probe_hash,
            });
        }

        self.loaded = Some(candidate);
        Ok(manifest)
    }

    fn encoder(&self) -> Box<dyn Encoder> {
        Box::new(MantisEncoder)
    }

    fn evaluator(&self) -> Result<Box<dyn Evaluator>, PackageError> {
        Ok(Box::new(MantisEvaluator::new(
            Arc::clone(&self.loaded()?.forward),
            self.config.tau,
            self.config.lambda,
        )))
    }

    fn self_play_session(&self) -> Result<Box<dyn DecisionSession>, PackageError> {
        self.session(Search::Policy, true)
    }

    fn eval_session(&self) -> Result<Box<dyn DecisionSession>, PackageError> {
        self.session(
            Search::Gumbel {
                simulations: EVAL_SIMULATIONS,
                candidates: EVAL_CANDIDATES,
                temperature: 1.0,
            },
            false,
        )
    }

    fn variant_session(&self, name: &str) -> Result<Box<dyn DecisionSession>, PackageError> {
        self.session(config::parse_variant(name)?, false)
    }

    fn fit(
        &mut self,
        _shards: &[PathBuf],
        _out_dir: &Path,
        _epoch: u32,
    ) -> Result<Manifest, PackageError> {
        Err(PackageError::Unsupported {
            package: PACKAGE_NAME,
            operation: "fit",
            reason: "MantisNet trains in `mantisnet.klent.run`; moving that production KLENT \
                     loop into the container requires an owner decision",
        })
    }
}

/// SplitMix64 finalizer used to derive distinct session construction seeds.
const fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
