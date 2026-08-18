//! KLENT acting, comparison selection, and self-play diagnostics.

use hexo_engine::{Action, Position};
use hexo_search::{Evaluation, SearchOutcome, SelectFromPolicy, SelectFromSearch, SplitMix64};

/// Version of the compact self-play diagnostic payload.
pub(crate) const DIAGNOSTICS_VERSION: u8 = 1;

/// Draw one canonical index from non-negative weights.
fn sample(weights: &[f32], rng: &mut SplitMix64) -> usize {
    let total: f64 = weights.iter().map(|&weight| f64::from(weight)).sum();
    assert!(
        total.is_finite() && total > 0.0,
        "MantisNet acting received {} probabilities totalling {total}",
        weights.len(),
    );
    let mut ticket = rng.next_f64() * total;
    for (index, &weight) in weights.iter().enumerate() {
        ticket -= f64::from(weight);
        if ticket < 0.0 {
            return index;
        }
    }
    weights.len() - 1
}

/// `[version, v_hat: f32-le, entropy(pi_prime): f32-le]`.
fn diagnostics(evaluation: &Evaluation) -> Vec<u8> {
    let entropy = evaluation
        .priors
        .iter()
        .filter(|&&probability| probability > 0.0)
        .map(|&probability| -probability * probability.ln())
        .sum::<f32>();
    let mut bytes = Vec::with_capacity(9);
    bytes.push(DIAGNOSTICS_VERSION);
    bytes.extend_from_slice(&evaluation.value.to_le_bytes());
    bytes.extend_from_slice(&entropy.to_le_bytes());
    bytes
}

/// Policy acting samples `pi_prime`; self-play additionally records diagnostics.
pub(crate) struct ActingPolicy {
    pub(crate) record_diagnostics: bool,
}

impl SelectFromPolicy for ActingPolicy {
    fn select(&mut self, root: &Position, evaluation: &Evaluation, rng: &mut SplitMix64) -> Action {
        root.nth_legal(sample(&evaluation.priors, rng))
            .expect("MantisNet priors use canonical legal order")
    }

    fn diagnostics(&mut self, _root: &Position, evaluation: &Evaluation) -> Option<Vec<u8>> {
        self.record_diagnostics.then(|| diagnostics(evaluation))
    }
}

/// Selects the most-visited root child.
pub(crate) struct MaxVisits;

impl SelectFromSearch for MaxVisits {
    fn select(&mut self, outcome: &SearchOutcome<'_>, _rng: &mut SplitMix64) -> Action {
        outcome
            .children()
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.visits
                    .cmp(&right.visits)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(_, child)| child.action)
            .expect("a live root has a legal child")
    }

    fn diagnostics(&mut self, _outcome: &SearchOutcome<'_>) -> Option<Vec<u8>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_are_versioned_value_and_entropy() {
        let evaluation = Evaluation {
            priors: vec![0.25, 0.75].into_boxed_slice(),
            value: -0.5,
        };
        let bytes = diagnostics(&evaluation);
        assert_eq!(bytes.len(), 9);
        assert_eq!(bytes[0], DIAGNOSTICS_VERSION);
        assert_eq!(
            f32::from_le_bytes(bytes[1..5].try_into().expect("value")),
            -0.5
        );
        let entropy = f32::from_le_bytes(bytes[5..9].try_into().expect("entropy"));
        let expected = -(0.25_f32 * 0.25_f32.ln() + 0.75_f32 * 0.75_f32.ln());
        assert_eq!(entropy, expected);
    }
}
