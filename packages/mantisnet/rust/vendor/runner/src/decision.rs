//! Seat budgets, decisions, and replies.

use hexo_engine::Action;
use std::time::Duration;

/// What a seat was given to think with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Budget {
    /// No stated limit.
    #[default]
    Unlimited,
    /// A search-node count.
    Nodes(u64),
    /// A tree-visit count.
    Visits(u64),
    /// Wall-clock time.
    Wall(Duration),
}

/// A seat's complete placement decision.
///
/// The seat authors every field. `zobrist` is the hash of the position used for
/// selection, whether canonical or mirrored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    /// Where to place.
    pub action: Action,
    /// The hash of the position the seat chose from.
    pub zobrist: u64,
    /// Opaque, seat-owned bytes, persisted verbatim and never interpreted.
    pub diagnostics: Option<Vec<u8>>,
}

impl Decision {
    /// A placement with no diagnostics.
    #[must_use]
    pub const fn new(action: Action, zobrist: u64) -> Self {
        Self {
            action,
            zobrist,
            diagnostics: None,
        }
    }

    /// Attach seat-owned bytes to this decision.
    #[must_use]
    pub fn with_diagnostics(mut self, bytes: Vec<u8>) -> Self {
        self.diagnostics = Some(bytes);
        self
    }
}

/// Why a seat's turn produced no accepted placement, as the driver reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Failure {
    /// The seat did not answer within its budget.
    Timeout,
    /// The seat's process died, or its transport broke.
    Crashed,
    /// The seat answered, but the answer could not be understood.
    Protocol,
    /// The seat answered from a position that is not the game's, and the driver
    /// could not bring it back into sync.
    Desync {
        /// The canonical hash.
        expected: u64,
        /// The hash the seat attested.
        got: u64,
    },
}

/// Everything a seat can come back with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    /// A placement.
    Place(Decision),
    /// The seat gives up.
    Resign,
    /// The driver could not get a decision.
    Failed(Failure),
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::HexCoord;

    #[test]
    fn the_default_budget_is_unlimited() {
        assert_eq!(Budget::default(), Budget::Unlimited);
    }

    #[test]
    fn diagnostics_are_absent_unless_attached() {
        let a = Action::new(HexCoord::ORIGIN);
        let plain = Decision::new(a, 0x1234);
        assert_eq!(plain.diagnostics, None);
        assert_eq!(plain.zobrist, 0x1234);

        let annotated = Decision::new(a, 0x1234).with_diagnostics(vec![1, 2, 3]);
        assert_eq!(annotated.diagnostics.as_deref(), Some(&[1u8, 2, 3][..]));
    }
}
