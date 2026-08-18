//! Package-owned move-selection hooks.
//!
//! This crate defines the policy and search selection contracts but provides no
//! selector implementation.

use crate::rng::SplitMix64;
use crate::seam::Evaluation;
use hexo_engine::{Action, Position};

/// One root child in a completed search.
///
/// Children are in the engine's canonical legal order, so `children()[i]`
/// corresponds to `root.nth_legal(i)` and to prior `i` of the root's own
/// evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Child {
    /// The placement this child plays.
    pub action: Action,
    /// Settled visits below this child; no virtual loss remains.
    pub visits: u32,
    /// Mean backed-up value, from the perspective of the **root's** mover, or
    /// `0.0` for an unvisited child.
    pub mean_value: f64,
    /// The prior the network gave this action at the root.
    pub prior: f32,
}

/// A completed search root and its children.
///
/// A borrowed view, not a snapshot: it exists only for the duration of the
/// selector call. It is publicly constructible so a package can unit-test its
/// own selector against a table of children without standing up a search.
#[derive(Clone, Copy, Debug)]
pub struct SearchOutcome<'a> {
    root: &'a Position,
    children: &'a [Child],
}

impl<'a> SearchOutcome<'a> {
    /// A view over `root` and its `children`, which must be in the canonical
    /// legal order of `root`.
    #[must_use]
    pub const fn new(root: &'a Position, children: &'a [Child]) -> Self {
        Self { root, children }
    }

    /// The position the search started from.
    #[inline]
    #[must_use]
    pub const fn root(&self) -> &'a Position {
        self.root
    }

    /// The root's children, in canonical legal order.
    #[inline]
    #[must_use]
    pub const fn children(&self) -> &'a [Child] {
        self.children
    }

    /// Visits spent below the root, summed over its children.
    ///
    /// Equal to the configured visit budget once the search is done: the root's
    /// own evaluation is not one of them.
    #[must_use]
    pub fn total_visits(&self) -> u32 {
        self.children.iter().map(|c| c.visits).sum()
    }
}

/// Package-owned conversion from a completed tree search to a decision payload.
///
/// Both selection and diagnostics behavior are required.
pub trait SelectFromSearch: Send {
    /// Choose the placement to play.
    ///
    /// `rng` is the session's stream.
    fn select(&mut self, outcome: &SearchOutcome<'_>, rng: &mut SplitMix64) -> Action;

    /// The seat-owned diagnostics for the record, or `None` to record nothing.
    ///
    /// Stored verbatim by `hexo_runner::Game` and never interpreted by anything
    /// in this workspace.
    fn diagnostics(&mut self, outcome: &SearchOutcome<'_>) -> Option<Vec<u8>>;
}

/// Package-owned conversion from one root evaluation to a decision payload.
pub trait SelectFromPolicy: Send {
    /// Choose the placement to play. `evaluation.priors[i]` belongs to
    /// `root.nth_legal(i)`.
    fn select(&mut self, root: &Position, evaluation: &Evaluation, rng: &mut SplitMix64) -> Action;

    /// The seat-owned diagnostics for the record, or `None` to record nothing.
    fn diagnostics(&mut self, root: &Position, evaluation: &Evaluation) -> Option<Vec<u8>>;
}
