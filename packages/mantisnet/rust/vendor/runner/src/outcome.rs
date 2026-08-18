//! Match result and adjudication reason types.

use crate::decision::Failure;
use hexo_engine::{ActionId, MoveError, Player};

/// How a match ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchResult {
    /// A seat won.
    Decisive {
        /// The seat that won.
        winner: Player,
        /// How it won.
        reason: WinReason,
    },
    /// Nobody won, and nothing went wrong.
    Drawn {
        /// Why the game stopped.
        reason: DrawReason,
    },
    /// Nobody won, and the game says nothing about either seat's strength.
    NoContest(NoContest),
}

impl MatchResult {
    /// Whether the match reached a verdict, as opposed to a no-contest.
    ///
    /// Forfeits are contested results; consumers needing board-play outcomes
    /// must also inspect [`WinReason`].
    #[inline]
    #[must_use]
    pub const fn is_contested(self) -> bool {
        !matches!(self, Self::NoContest(_))
    }

    /// The winner, if there was one.
    #[inline]
    #[must_use]
    pub const fn winner(self) -> Option<Player> {
        match self {
            Self::Decisive { winner, .. } => Some(winner),
            _ => None,
        }
    }
}

/// How a seat won.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WinReason {
    /// Six or more of the winner's stones in a row.
    SixInARow,
    /// The other seat resigned.
    Resignation,
    /// The other seat submitted a placement that broke the rules.
    IllegalMove {
        /// The placement, in the record encoding.
        action: ActionId,
        /// The rule violation.
        cause: MoveError,
    },
    /// The other seat did not answer within its budget.
    Timeout,
    /// The other seat died, or its transport broke.
    Crash,
    /// The other seat answered, but unintelligibly.
    Protocol,
    /// The other seat answered from a position that is not the game's.
    Desync {
        /// The canonical hash.
        expected: u64,
        /// The hash the seat attested.
        got: u64,
    },
}

/// Why a game ended without a winner, blamelessly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawReason {
    /// The game reached [`crate::GameSpec::ply_cap`] on a placement that completed
    /// the mover's turn. Turns end at odd placement counts, so a game under an
    /// even cap ends one placement past it.
    PlyCap,
}

/// Why a game produced no usable result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoContest {
    /// The engine could not represent the board a legal placement would produce.
    EngineLimit {
        /// The seat whose placement could not be represented.
        seat: Player,
        /// The refusal.
        error: MoveError,
    },
    /// The driver reported a failure that policy does not charge to either seat.
    SeatFailure {
        /// The seat whose decision failed.
        seat: Player,
        /// What went wrong.
        failure: Failure,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::HexCoord;

    #[test]
    fn only_no_contest_is_uncontested() {
        let decisive = MatchResult::Decisive {
            winner: Player::P0,
            reason: WinReason::SixInARow,
        };
        let drawn = MatchResult::Drawn {
            reason: DrawReason::PlyCap,
        };
        let no_contest = MatchResult::NoContest(NoContest::SeatFailure {
            seat: Player::P0,
            failure: Failure::Protocol,
        });

        assert!(decisive.is_contested());
        assert!(drawn.is_contested());
        assert!(!no_contest.is_contested());
    }

    #[test]
    fn only_a_decisive_result_has_a_winner() {
        assert_eq!(
            MatchResult::Decisive {
                winner: Player::P1,
                reason: WinReason::Resignation,
            }
            .winner(),
            Some(Player::P1)
        );
        assert_eq!(
            MatchResult::Drawn {
                reason: DrawReason::PlyCap
            }
            .winner(),
            None
        );
        assert_eq!(
            MatchResult::NoContest(NoContest::EngineLimit {
                seat: Player::P0,
                error: MoveError::CoordOutOfBounds(HexCoord::ORIGIN),
            })
            .winner(),
            None
        );
    }

    /// A capped game and a crashed game must not compare equal.
    #[test]
    fn a_capped_game_is_distinguishable_from_a_broken_one() {
        let capped = MatchResult::Drawn {
            reason: DrawReason::PlyCap,
        };
        let broken = MatchResult::NoContest(NoContest::SeatFailure {
            seat: Player::P1,
            failure: Failure::Crashed,
        });
        assert_ne!(capped, broken);
        assert!(capped.is_contested());
        assert!(!broken.is_contested());
    }
}
