//! Worker-side encoding and batcher-side MantisNet evaluation.

use crate::encoder;
use crate::forward::{Forward, RawOutputs};
use crate::improvement::improve_policy;
use hexo_engine::Position;
use hexo_search::{EncodedBatch, Encoder, Evaluation, Evaluator};
use std::sync::{Arc, Mutex};

/// The package encoder: one local graph appended in the versioned wire layout.
pub(crate) struct MantisEncoder;

impl Encoder for MantisEncoder {
    fn encode(&self, position: &Position, out: &mut Vec<u8>) {
        encoder::encode_position(position, out);
    }
}

/// Turns raw MantisNet cell heads into the model's improved opinion.
pub(crate) struct MantisEvaluator {
    forward: Arc<Mutex<Box<dyn Forward>>>,
    tau: f32,
    lambda: f32,
}

impl MantisEvaluator {
    pub(crate) fn new(forward: Arc<Mutex<Box<dyn Forward>>>, tau: f32, lambda: f32) -> Self {
        Self {
            forward,
            tau,
            lambda,
        }
    }

    fn answers(&self, batch: &encoder::RawBatch, raw: RawOutputs) -> Vec<Evaluation> {
        let total = batch
            .legal_offsets
            .last()
            .copied()
            .expect("a collated batch always carries its initial legal offset");
        let total = usize::try_from(total).expect("the encoder emits non-negative offsets");
        assert_eq!(
            raw.policy_logits.len(),
            total,
            "MantisNet returned {} policy logits for a batch with {total} legal actions",
            raw.policy_logits.len(),
        );
        assert_eq!(
            raw.q_values.len(),
            total,
            "MantisNet returned {} q values for a batch with {total} legal actions",
            raw.q_values.len(),
        );

        let mut answers = Vec::with_capacity(batch.n_pos);
        for row in batch.legal_offsets.windows(2) {
            let start = usize::try_from(row[0]).expect("the encoder emits non-negative offsets");
            let end = usize::try_from(row[1]).expect("the encoder emits non-negative offsets");
            let improved = improve_policy(
                &raw.policy_logits[start..end],
                &raw.q_values[start..end],
                self.tau,
                self.lambda,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "MantisNet could not improve legal row {start}..{end} with tau={} and \
                     lambda={}: {error}",
                    self.tau, self.lambda,
                )
            });
            answers.push(Evaluation {
                priors: improved.pi_prime.into_boxed_slice(),
                value: improved.v_hat,
            });
        }
        answers
    }
}

impl Evaluator for MantisEvaluator {
    fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>) {
        let typed = encoder::decode_batch(batch.iter()).unwrap_or_else(|error| {
            panic!("MantisNet evaluator received malformed encoded bytes: {error}")
        });
        let raw = self
            .forward
            .lock()
            .expect("the MantisNet forward mutex was poisoned by an earlier failed forward")
            .forward(&typed)
            .unwrap_or_else(|error| panic!("MantisNet forward failed: {error}"));
        let answers = self.answers(&typed, raw);
        assert_eq!(
            answers.len(),
            batch.len(),
            "MantisNet answered {} of {} encoded positions",
            answers.len(),
            batch.len(),
        );
        out.extend(answers);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoxError;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Scripted {
        calls: Arc<AtomicUsize>,
    }

    impl Forward for Scripted {
        fn forward(&mut self, batch: &encoder::RawBatch) -> Result<RawOutputs, BoxError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let cells = usize::try_from(*batch.legal_offsets.last().expect("offset"))
                .expect("non-negative");
            Ok(RawOutputs {
                policy_logits: vec![0.0; cells],
                q_values: vec![0.25; cells],
            })
        }
    }

    #[test]
    fn a_ragged_evaluator_batch_crosses_forward_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let forward: Arc<Mutex<Box<dyn Forward>>> = Arc::new(Mutex::new(Box::new(Scripted {
            calls: Arc::clone(&calls),
        })));
        let mut evaluator = MantisEvaluator::new(forward, 0.1, 0.03);
        let mut encoded = EncodedBatch::new();

        let opening = Position::new();
        let mut opened = Position::new();
        let action = opened.nth_legal(0).expect("the origin");
        opened.advance(action).expect("the opening is legal");
        encoded.push_with(&MantisEncoder, &opening);
        encoded.push_with(&MantisEncoder, &opened);

        let mut answers = Vec::new();
        evaluator.evaluate(&encoded, &mut answers);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(answers.len(), 2);
        assert_eq!(answers[0].priors.len(), opening.legal_count());
        assert_eq!(answers[1].priors.len(), opened.legal_count());
        assert!((answers[0].value - 0.25).abs() <= 1.0e-6);
        assert!((answers[1].value - 0.25).abs() <= 1.0e-6);
    }
}
