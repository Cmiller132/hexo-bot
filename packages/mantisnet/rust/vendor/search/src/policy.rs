//! One root evaluation per move: the policy-only session.

use crate::rng::SplitMix64;
use crate::seam::Evaluation;
use crate::select::SelectFromPolicy;
use crate::session::{DecisionSession, LeafId, SessionStatus};
use hexo_engine::Position;
use hexo_runner::Decision;

/// Where a [`PolicySession`] is in its one-question cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Never begun.
    Fresh,
    /// Begun; the root has not been emitted yet.
    Wanted,
    /// The root is out for evaluation.
    Awaiting(LeafId),
    /// The decision has been authored.
    Decided,
}

/// A seat that asks the network exactly one question per move.
///
/// `begin` copies the position, the first `pump` emits the root, `resume`
/// delivers its evaluation and authors the decision, and the next `pump` reports
/// [`SessionStatus::Decided`].
pub struct PolicySession {
    selector: Box<dyn SelectFromPolicy>,
    rng: SplitMix64,
    root: Position,
    state: State,
    next_serial: u64,
    decision: Option<Decision>,
}

impl PolicySession {
    /// A session that selects with `selector` and samples from `seed`.
    #[must_use]
    pub fn new(selector: Box<dyn SelectFromPolicy>, seed: u64) -> Self {
        Self {
            selector,
            rng: SplitMix64::new(seed),
            root: Position::new(),
            state: State::Fresh,
            next_serial: 0,
            decision: None,
        }
    }
}

impl DecisionSession for PolicySession {
    fn begin(&mut self, position: &Position) {
        assert!(
            !position.is_terminal(),
            "PolicySession::begin on a terminal position; a driver only asks a live position's \
             mover",
        );
        self.root.clone_from(position);
        self.state = State::Wanted;
        self.decision = None;
    }

    fn pump(&mut self, emit: &mut dyn FnMut(LeafId, &Position)) -> SessionStatus {
        match self.state {
            State::Fresh => panic!("PolicySession::pump before begin"),
            State::Wanted => {
                let leaf = LeafId::from_serial(self.next_serial);
                self.next_serial += 1;
                emit(leaf, &self.root);
                self.state = State::Awaiting(leaf);
                SessionStatus::AwaitingEvals { in_flight: 1 }
            }
            State::Awaiting(_) => SessionStatus::AwaitingEvals { in_flight: 1 },
            State::Decided => SessionStatus::Decided,
        }
    }

    fn resume(&mut self, leaf: LeafId, evaluation: Evaluation) {
        let State::Awaiting(wanted) = self.state else {
            panic!(
                "PolicySession::resume with {leaf:?}, but the session has nothing in flight \
                 ({:?})",
                self.state,
            );
        };
        assert_eq!(
            leaf, wanted,
            "PolicySession::resume with unknown {leaf:?}; the leaf in flight is {wanted:?}",
        );
        evaluation.check(self.root.legal_count(), leaf);

        let action = self.selector.select(&self.root, &evaluation, &mut self.rng);
        let diagnostics = self.selector.diagnostics(&self.root, &evaluation);
        self.decision = Some(Decision {
            action,
            zobrist: self.root.zobrist(),
            diagnostics,
        });
        self.state = State::Decided;
    }

    fn take_decision(&mut self) -> Option<Decision> {
        self.decision.take()
    }

    fn reseed(&mut self, seed: u64) {
        self.rng.reseed(seed);
    }
}
