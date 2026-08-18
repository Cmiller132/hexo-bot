//! A seat's search as a nonblocking state machine.

use crate::seam::Evaluation;
use hexo_engine::Position;
use hexo_runner::Decision;

/// Session-scoped handle for one requested leaf evaluation.
///
/// Opaque and never reused within a session, including across
/// [`DecisionSession::begin`] calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LeafId(u64);

impl LeafId {
    /// Mint the id for a session's `serial`-th requested leaf.
    #[inline]
    pub(crate) const fn from_serial(serial: u64) -> Self {
        Self(serial)
    }
}

/// Where a session is between [`DecisionSession::begin`] and its decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionStatus {
    /// The session has evaluations in flight and cannot progress until they are
    /// delivered with [`DecisionSession::resume`].
    ///
    /// `in_flight` is never zero: a session with nothing outstanding and work
    /// left to do keeps working rather than returning.
    AwaitingEvals {
        /// How many leaves are waiting for an answer.
        in_flight: usize,
    },
    /// The decision is ready to take.
    Decided,
}

/// A `Send`, object-safe, nonblocking decision state machine.
///
/// The loop is `begin`, then `pump`/`resume` until [`SessionStatus::Decided`],
/// then `take_decision`. The driver controls session interleaving and evaluator
/// batch size.
pub trait DecisionSession: Send {
    /// Reset onto `position` and discard any previous search.
    ///
    /// The session copies the position and does not retain access to the
    /// caller's mirror or canonical state.
    ///
    /// # Panics
    ///
    /// If `position` is terminal.
    fn begin(&mut self, position: &Position);

    /// Run until the decision is ready, the in-flight cap is reached, or the
    /// visit budget is fully dispatched.
    ///
    /// For each requested leaf, `emit` receives its handle and transient
    /// position. The position is valid only for the callback duration and must
    /// be encoded before the callback returns.
    ///
    /// Calling `pump` again after [`SessionStatus::Decided`] returns `Decided`
    /// and emits nothing.
    ///
    /// # Panics
    ///
    /// If `begin` has never been called.
    fn pump(&mut self, emit: &mut dyn FnMut(LeafId, &Position)) -> SessionStatus;

    /// Deliver one result.
    ///
    /// # Panics
    ///
    /// If `leaf` is not in flight or if the evaluation violates
    /// [`Evaluation`]'s conventions.
    fn resume(&mut self, leaf: LeafId, evaluation: Evaluation);

    /// Take the finished decision, or `None` until the session is
    /// [`SessionStatus::Decided`].
    ///
    /// Taking the decision does not reset the session; [`DecisionSession::begin`]
    /// performs the next reset.
    fn take_decision(&mut self) -> Option<Decision>;

    /// Replace the RNG seed.
    ///
    /// Drivers may call this at game boundaries.
    fn reseed(&mut self, seed: u64);
}
