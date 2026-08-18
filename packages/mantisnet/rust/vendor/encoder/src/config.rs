//! Package and session-variant configuration parsing.

use crate::PACKAGE_NAME;
use hexo_model::PackageError;
use hexo_search::MctsConfig;
use std::{
    num::{NonZeroU32, NonZeroUsize},
    path::PathBuf,
};

/// The model parameters and optional raw-checkpoint source used by `init`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Config {
    /// Reverse-KL weight in the KLENT improvement.
    pub(crate) tau: f32,
    /// Entropy weight in the KLENT improvement.
    pub(crate) lambda: f32,
    /// A raw Python `.pt` checkpoint to seal during `init`.
    pub(crate) source: Option<PathBuf>,
}

/// The search shape a MantisNet session should use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Search {
    /// Sample directly from MantisNet's improved policy.
    Policy,
    /// Search with PUCT under the stated compute limits.
    Mcts(MctsConfig),
    /// Search by Gumbel sequential halving over root candidates.
    Gumbel {
        /// Total number of line-deepening simulations.
        simulations: NonZeroU32,
        /// Maximum number of root candidates.
        candidates: NonZeroUsize,
        /// Root Gumbel scale. Zero is deterministic and one is unscaled.
        temperature: f64,
    },
}

/// Why a session variant could not be parsed.
enum VariantFailure {
    /// The leading word does not name a session shape this package owns.
    UnknownShape,
    /// The leading word is known, but one of its parameters is invalid.
    BadParameters(String),
}

impl VariantFailure {
    fn into_package_error(self, variant: &str) -> PackageError {
        match self {
            Self::UnknownShape => PackageError::UnknownVariant {
                package: PACKAGE_NAME,
                variant: variant.to_owned(),
            },
            Self::BadParameters(problem) => PackageError::InvalidConfig {
                package: PACKAGE_NAME,
                problem,
            },
        }
    }
}

/// Parse `tau=F,lambda=F[,source=PATH]`.
///
/// All keys except `source` are required. Fields may appear in any order.
/// Whitespace is not trimmed, and repeated keys are rejected.
pub(crate) fn parse_config(config: &str) -> Result<Config, PackageError> {
    let mut tau = None;
    let mut lambda = None;
    let mut source = None;

    for field in config.split(',') {
        let Some((key, value)) = field.split_once('=') else {
            return Err(invalid(format!("{field:?} is not a `key=value` pair")));
        };
        match key {
            "tau" => {
                reject_config_repeat(tau.is_some(), key)?;
                tau = Some(non_negative_float("configuration", key, value)?);
            }
            "lambda" => {
                reject_config_repeat(lambda.is_some(), key)?;
                lambda = Some(non_negative_float("configuration", key, value)?);
            }
            "source" => {
                reject_config_repeat(source.is_some(), key)?;
                if value.is_empty() {
                    return Err(invalid(
                        "configuration field \"source\" is empty; it must name a raw `.pt` file"
                            .to_owned(),
                    ));
                }
                source = Some(PathBuf::from(value));
            }
            _ => {
                return Err(invalid(format!(
                    "unknown configuration field {key:?}; expected `tau`, `lambda`, or `source`"
                )));
            }
        }
    }

    let tau = tau.ok_or_else(|| missing_config("tau"))?;
    let lambda = lambda.ok_or_else(|| missing_config("lambda"))?;
    let denominator = tau + lambda;
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(invalid(format!(
            "configuration fields \"tau\" and \"lambda\" sum to {denominator}; their sum must be \
             finite and positive"
        )));
    }

    Ok(Config {
        tau,
        lambda,
        source,
    })
}

/// Parse a session variant string into a [`Search`] configuration.
pub(crate) fn parse_variant(variant: &str) -> Result<Search, PackageError> {
    parse_variant_inner(variant).map_err(|failure| failure.into_package_error(variant))
}

fn parse_variant_inner(variant: &str) -> Result<Search, VariantFailure> {
    let (shape, parameters) = match variant.split_once(':') {
        Some((shape, parameters)) => (shape, Some(parameters)),
        None => (variant, None),
    };
    match shape {
        "policy" => match parameters {
            None => Ok(Search::Policy),
            Some(parameters) => Err(VariantFailure::BadParameters(format!(
                "`policy` takes no parameters, but {parameters:?} follows it"
            ))),
        },
        "mcts" => parameters.map_or_else(
            || {
                Err(VariantFailure::BadParameters(
                    "`mcts` needs `visits`, `inflight`, and `cpuct`".to_owned(),
                ))
            },
            parse_mcts,
        ),
        "gumbel" => parameters.map_or_else(
            || {
                Err(VariantFailure::BadParameters(
                    "`gumbel` needs `sims` and `m`".to_owned(),
                ))
            },
            parse_gumbel,
        ),
        _ => Err(VariantFailure::UnknownShape),
    }
}

fn parse_mcts(parameters: &str) -> Result<Search, VariantFailure> {
    let mut visits = None;
    let mut inflight = None;
    let mut c_puct = None;

    for field in parameters.split(',') {
        let (key, value) = variant_pair("mcts", field)?;
        match key {
            "visits" => {
                reject_variant_repeat(visits.is_some(), "mcts", key)?;
                visits = Some(nonzero_u32("mcts", key, value)?);
            }
            "inflight" => {
                reject_variant_repeat(inflight.is_some(), "mcts", key)?;
                inflight = Some(nonzero_usize("mcts", key, value)?);
            }
            "cpuct" => {
                reject_variant_repeat(c_puct.is_some(), "mcts", key)?;
                c_puct = Some(non_negative_variant_float("mcts", key, value)?);
            }
            _ => {
                return Err(VariantFailure::BadParameters(format!(
                    "unknown `mcts` parameter {key:?}; expected `visits`, `inflight`, or `cpuct`"
                )));
            }
        }
    }

    Ok(Search::Mcts(MctsConfig {
        visits: visits.ok_or_else(|| missing_variant("mcts", "visits"))?,
        max_in_flight: inflight.ok_or_else(|| missing_variant("mcts", "inflight"))?,
        c_puct: c_puct.ok_or_else(|| missing_variant("mcts", "cpuct"))?,
    }))
}

fn parse_gumbel(parameters: &str) -> Result<Search, VariantFailure> {
    let mut simulations = None;
    let mut candidates = None;
    let mut temperature = None;

    for field in parameters.split(',') {
        let (key, value) = variant_pair("gumbel", field)?;
        match key {
            "sims" => {
                reject_variant_repeat(simulations.is_some(), "gumbel", key)?;
                simulations = Some(nonzero_u32("gumbel", key, value)?);
            }
            "m" => {
                reject_variant_repeat(candidates.is_some(), "gumbel", key)?;
                candidates = Some(nonzero_usize("gumbel", key, value)?);
            }
            "temp" => {
                reject_variant_repeat(temperature.is_some(), "gumbel", key)?;
                temperature = Some(non_negative_variant_f64("gumbel", key, value)?);
            }
            _ => {
                return Err(VariantFailure::BadParameters(format!(
                    "unknown `gumbel` parameter {key:?}; expected `sims`, `m`, or `temp`"
                )));
            }
        }
    }

    Ok(Search::Gumbel {
        simulations: simulations.ok_or_else(|| missing_variant("gumbel", "sims"))?,
        candidates: candidates.ok_or_else(|| missing_variant("gumbel", "m"))?,
        temperature: temperature.unwrap_or(1.0),
    })
}

fn variant_pair<'a>(shape: &str, field: &'a str) -> Result<(&'a str, &'a str), VariantFailure> {
    field.split_once('=').ok_or_else(|| {
        VariantFailure::BadParameters(format!(
            "`{shape}` field {field:?} is not a `key=value` pair"
        ))
    })
}

fn reject_config_repeat(already: bool, key: &str) -> Result<(), PackageError> {
    if already {
        return Err(invalid(format!(
            "configuration field {key:?} is stated twice"
        )));
    }
    Ok(())
}

fn reject_variant_repeat(already: bool, shape: &str, key: &str) -> Result<(), VariantFailure> {
    if already {
        return Err(VariantFailure::BadParameters(format!(
            "`{shape}` parameter {key:?} is stated twice"
        )));
    }
    Ok(())
}

fn missing_config(key: &str) -> PackageError {
    invalid(format!(
        "configuration field {key:?} is required and was not stated"
    ))
}

fn missing_variant(shape: &str, key: &str) -> VariantFailure {
    VariantFailure::BadParameters(format!(
        "`{shape}` parameter {key:?} is required and was not stated"
    ))
}

fn non_negative_float(context: &str, key: &str, value: &str) -> Result<f32, PackageError> {
    let parsed = value.parse::<f32>().map_err(|_| {
        invalid(format!(
            "{context} field {key:?} is {value:?}, not a floating-point number"
        ))
    })?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(invalid(format!(
            "{context} field {key:?} is {value:?}; it must be finite and non-negative"
        )));
    }
    Ok(parsed)
}

fn non_negative_variant_float(shape: &str, key: &str, value: &str) -> Result<f32, VariantFailure> {
    let parsed = value.parse::<f32>().map_err(|_| {
        VariantFailure::BadParameters(format!(
            "`{shape}` parameter {key:?} is {value:?}, not a floating-point number"
        ))
    })?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(VariantFailure::BadParameters(format!(
            "`{shape}` parameter {key:?} is {value:?}; it must be finite and non-negative"
        )));
    }
    Ok(parsed)
}

fn non_negative_variant_f64(shape: &str, key: &str, value: &str) -> Result<f64, VariantFailure> {
    let parsed = value.parse::<f64>().map_err(|_| {
        VariantFailure::BadParameters(format!(
            "`{shape}` parameter {key:?} is {value:?}, not a floating-point number"
        ))
    })?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(VariantFailure::BadParameters(format!(
            "`{shape}` parameter {key:?} is {value:?}; it must be finite and non-negative"
        )));
    }
    Ok(parsed)
}

fn nonzero_u32(shape: &str, key: &str, value: &str) -> Result<NonZeroU32, VariantFailure> {
    let parsed = value.parse::<u32>().map_err(|_| {
        VariantFailure::BadParameters(format!(
            "`{shape}` parameter {key:?} is {value:?}, not a non-negative integer"
        ))
    })?;
    NonZeroU32::new(parsed).ok_or_else(|| {
        VariantFailure::BadParameters(format!(
            "`{shape}` parameter {key:?} is zero; it must be at least one"
        ))
    })
}

fn nonzero_usize(shape: &str, key: &str, value: &str) -> Result<NonZeroUsize, VariantFailure> {
    let parsed = value.parse::<usize>().map_err(|_| {
        VariantFailure::BadParameters(format!(
            "`{shape}` parameter {key:?} is {value:?}, not a non-negative integer"
        ))
    })?;
    NonZeroUsize::new(parsed).ok_or_else(|| {
        VariantFailure::BadParameters(format!(
            "`{shape}` parameter {key:?} is zero; it must be at least one"
        ))
    })
}

fn invalid(problem: String) -> PackageError {
    PackageError::InvalidConfig {
        package: PACKAGE_NAME,
        problem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_problem(input: &str) -> String {
        match parse_config(input) {
            Err(PackageError::InvalidConfig { package, problem }) => {
                assert_eq!(package, PACKAGE_NAME);
                problem
            }
            Err(other) => panic!("wrong error for {input:?}: {other:?}"),
            Ok(config) => panic!("{input:?} unexpectedly parsed as {config:?}"),
        }
    }

    fn variant_problem(input: &str) -> String {
        match parse_variant(input) {
            Err(PackageError::InvalidConfig { package, problem }) => {
                assert_eq!(package, PACKAGE_NAME);
                problem
            }
            Err(other) => panic!("wrong error for {input:?}: {other:?}"),
            Ok(search) => panic!("{input:?} unexpectedly parsed as {search:?}"),
        }
    }

    #[test]
    fn config_accepts_either_order_and_an_optional_source() {
        assert_eq!(
            parse_config("tau=0.1,lambda=0.03").expect("required fields"),
            Config {
                tau: 0.1,
                lambda: 0.03,
                source: None,
            }
        );
        assert_eq!(
            parse_config("source=D:\\weights\\checkpoint.pt,lambda=0,tau=1")
                .expect("fields may be reordered"),
            Config {
                tau: 1.0,
                lambda: 0.0,
                source: Some(PathBuf::from("D:\\weights\\checkpoint.pt")),
            }
        );
    }

    #[test]
    fn config_requires_tau_and_lambda() {
        for (input, missing) in [
            ("lambda=0.03", "tau"),
            ("tau=0.1", "lambda"),
            ("source=model.pt", "tau"),
        ] {
            assert!(config_problem(input).contains(missing), "{input:?}");
        }
    }

    #[test]
    fn config_rejects_repeated_and_unknown_fields() {
        assert!(config_problem("tau=0.1,tau=0.2,lambda=0.03").contains("stated twice"));
        assert!(config_problem("tau=0.1,lambda=0.03,lambda=0.04").contains("stated twice"));
        assert!(
            config_problem("tau=0.1,lambda=0.03,source=a.pt,source=b.pt").contains("stated twice")
        );
        assert!(config_problem("tau=0.1,lambda=0.03,temperature=1").contains("temperature"));
    }

    #[test]
    fn config_rejects_missing_pairs_and_empty_source() {
        assert!(config_problem("").contains("key=value"));
        assert!(config_problem("tau=0.1,lambda").contains("key=value"));
        assert!(config_problem("tau=0.1,lambda=0.03,source=").contains("source"));
    }

    #[test]
    fn config_float_fields_are_finite_and_non_negative() {
        for (input, field) in [
            ("tau=no,lambda=1", "tau"),
            ("tau=-1,lambda=1", "tau"),
            ("tau=NaN,lambda=1", "tau"),
            ("tau=inf,lambda=1", "tau"),
            ("tau=1,lambda=no", "lambda"),
            ("tau=1,lambda=-1", "lambda"),
            ("tau=1,lambda=NaN", "lambda"),
            ("tau=1,lambda=inf", "lambda"),
        ] {
            assert!(config_problem(input).contains(field), "{input:?}");
        }
    }

    #[test]
    fn config_requires_a_finite_positive_parameter_sum() {
        assert!(config_problem("tau=0,lambda=0").contains("sum"));
        assert!(config_problem("tau=3e38,lambda=3e38").contains("sum"));
    }

    #[test]
    fn config_does_not_normalize_whitespace() {
        assert!(config_problem("tau =0.1,lambda=0.03").contains("tau "));
        assert!(config_problem("tau= 0.1,lambda=0.03").contains("tau"));
    }

    #[test]
    fn all_variant_shapes_parse_in_any_parameter_order() {
        assert_eq!(parse_variant("policy").expect("policy"), Search::Policy);
        assert_eq!(
            parse_variant("mcts:cpuct=0,inflight=2,visits=32").expect("mcts"),
            Search::Mcts(MctsConfig {
                visits: NonZeroU32::new(32).expect("nonzero"),
                max_in_flight: NonZeroUsize::new(2).expect("nonzero"),
                c_puct: 0.0,
            })
        );
        assert_eq!(
            parse_variant("gumbel:temp=0.25,m=8,sims=32").expect("gumbel"),
            Search::Gumbel {
                simulations: NonZeroU32::new(32).expect("nonzero"),
                candidates: NonZeroUsize::new(8).expect("nonzero"),
                temperature: 0.25,
            }
        );
        assert_eq!(
            parse_variant("gumbel:m=8,sims=32").expect("default temperature"),
            Search::Gumbel {
                simulations: NonZeroU32::new(32).expect("nonzero"),
                candidates: NonZeroUsize::new(8).expect("nonzero"),
                temperature: 1.0,
            },
        );
    }

    #[test]
    fn unknown_shapes_are_unknown_variants() {
        for input in ["greedy", "greedy:x=1", ""] {
            match parse_variant(input) {
                Err(PackageError::UnknownVariant { package, variant }) => {
                    assert_eq!(package, PACKAGE_NAME);
                    assert_eq!(variant, input);
                }
                other => panic!("wrong result for {input:?}: {other:?}"),
            }
        }
    }

    #[test]
    fn malformed_known_shapes_are_invalid_config() {
        for input in ["policy:x=1", "mcts", "mcts:", "gumbel", "gumbel:"] {
            let problem = variant_problem(input);
            assert!(!problem.is_empty(), "{input:?}");
        }
    }

    #[test]
    fn variant_fields_are_required_and_unknown_fields_are_named() {
        for (input, field) in [
            ("mcts:inflight=2,cpuct=1", "visits"),
            ("mcts:visits=32,cpuct=1", "inflight"),
            ("mcts:visits=32,inflight=2", "cpuct"),
            ("gumbel:m=8", "sims"),
            ("gumbel:sims=32", "m"),
            ("mcts:visits=32,inflight=2,cpuct=1,foo=0", "foo"),
            ("gumbel:sims=32,m=8,foo=0", "foo"),
        ] {
            assert!(variant_problem(input).contains(field), "{input:?}");
        }
    }

    #[test]
    fn repeated_variant_fields_are_refused() {
        for input in [
            "mcts:visits=1,visits=2,inflight=1,cpuct=1",
            "mcts:visits=1,inflight=1,inflight=2,cpuct=1",
            "mcts:visits=1,inflight=1,cpuct=1,cpuct=2",
            "gumbel:sims=1,sims=2,m=1",
            "gumbel:sims=1,m=1,m=2",
            "gumbel:sims=1,m=1,temp=1,temp=2",
        ] {
            assert!(variant_problem(input).contains("stated twice"), "{input:?}");
        }
    }

    #[test]
    fn variant_counts_are_positive_integers() {
        for input in [
            "mcts:visits=zero,inflight=1,cpuct=1",
            "mcts:visits=0,inflight=1,cpuct=1",
            "mcts:visits=1,inflight=no,cpuct=1",
            "mcts:visits=1,inflight=0,cpuct=1",
            "gumbel:sims=no,m=1",
            "gumbel:sims=0,m=1",
            "gumbel:sims=1,m=no",
            "gumbel:sims=1,m=0",
        ] {
            let problem = variant_problem(input);
            assert!(
                problem.contains("integer") || problem.contains("at least one"),
                "{input:?}: {problem}"
            );
        }
    }

    #[test]
    fn mcts_cpuct_is_finite_and_non_negative() {
        for input in [
            "mcts:visits=1,inflight=1,cpuct=no",
            "mcts:visits=1,inflight=1,cpuct=-1",
            "mcts:visits=1,inflight=1,cpuct=NaN",
            "mcts:visits=1,inflight=1,cpuct=inf",
        ] {
            assert!(variant_problem(input).contains("cpuct"), "{input:?}");
        }
    }

    #[test]
    fn gumbel_temperature_is_finite_and_non_negative() {
        for input in [
            "gumbel:sims=2,m=1,temp=no",
            "gumbel:sims=2,m=1,temp=-1",
            "gumbel:sims=2,m=1,temp=NaN",
            "gumbel:sims=2,m=1,temp=inf",
            "gumbel:sims=2,m=1,temp=-inf",
        ] {
            assert!(variant_problem(input).contains("temp"), "{input:?}");
        }
        assert_eq!(
            parse_variant("gumbel:sims=2,m=1,temp=0").expect("zero is deterministic"),
            Search::Gumbel {
                simulations: NonZeroU32::new(2).expect("nonzero"),
                candidates: NonZeroUsize::new(1).expect("nonzero"),
                temperature: 0.0,
            },
        );
    }

    #[test]
    fn variant_grammar_does_not_normalize_whitespace() {
        assert!(matches!(
            parse_variant(" policy"),
            Err(PackageError::UnknownVariant { .. })
        ));
        assert!(variant_problem("mcts:visits =1,inflight=1,cpuct=1").contains("visits "));
        assert!(variant_problem("gumbel:sims= 32,m=8").contains("sims"));
    }
}
