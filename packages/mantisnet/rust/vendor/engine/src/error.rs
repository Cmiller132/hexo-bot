//! Rejection and integrity error types.

use crate::coord::HexCoord;

/// Why a placement was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveError {
    /// The game is over; no placement is legal.
    TerminalState,
    /// The opening placement must be at [`HexCoord::ORIGIN`].
    IllegalOpening,
    /// The coordinate is outside [`crate::COORD_LIMIT`], yet within
    /// [`crate::LEGAL_RADIUS`] of a stone — a placement the rules allow but the
    /// engine cannot represent. An off-domain cell far from every stone is
    /// [`MoveError::TooFarFromStones`] instead: a rule violation is reported as
    /// one even when the cell is also unrepresentable.
    CoordOutOfBounds(HexCoord),
    /// The cell already holds a stone.
    Occupied(HexCoord),
    /// The cell is empty but further than [`crate::LEGAL_RADIUS`] from every stone.
    TooFarFromStones(HexCoord),
    /// The dense arena would exceed [`crate::MAX_GRID_CELLS`].
    BoardExtentExceeded {
        /// Cells the arena would have needed.
        cells: u64,
    },
}

impl MoveError {
    /// Whether this rejection is a rule violation rather than an engine limit.
    #[inline]
    #[must_use]
    pub const fn is_rule_violation(self) -> bool {
        !matches!(
            self,
            Self::CoordOutOfBounds(_) | Self::BoardExtentExceeded { .. }
        )
    }
}

impl core::fmt::Display for MoveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TerminalState => f.write_str("the game is over"),
            Self::IllegalOpening => f.write_str("the opening placement must be at the origin"),
            Self::CoordOutOfBounds(c) => {
                write!(
                    f,
                    "coordinate ({}, {}) is outside the engine's range",
                    c.q, c.r
                )
            }
            Self::Occupied(c) => write!(f, "({}, {}) already holds a stone", c.q, c.r),
            Self::TooFarFromStones(c) => write!(
                f,
                "({}, {}) is further than the legal radius from every stone",
                c.q, c.r
            ),
            Self::BoardExtentExceeded { cells } => {
                write!(f, "the board arena would need {cells} cells")
            }
        }
    }
}

impl core::error::Error for MoveError {}

/// A placement sequence that stopped being legal partway through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayError {
    /// Index into the replayed slice, counting from zero.
    pub ply: usize,
    /// The placement that was refused.
    pub action: crate::action::Action,
    /// Why it was refused.
    pub cause: MoveError,
}

impl core::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let c = self.action.coord();
        write!(
            f,
            "replay failed at ply {}: ({}, {}): {}",
            self.ply, c.q, c.r, self.cause
        )
    }
}

impl core::error::Error for ReplayError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// A failed [`crate::Position::audit`] check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegrityError {
    /// Which invariant failed.
    pub check: IntegrityCheck,
    /// The cell it failed at, when the check is per-cell.
    pub coord: Option<HexCoord>,
}

impl core::fmt::Display for IntegrityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.coord {
            Some(c) => write!(
                f,
                "integrity check {:?} failed at ({}, {})",
                self.check, c.q, c.r
            ),
            None => write!(f, "integrity check {:?} failed", self.check),
        }
    }
}

impl core::error::Error for IntegrityError {}

/// The invariant that [`crate::Position::audit`] found broken.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IntegrityCheck {
    /// A cell is owned by both players.
    DoubleOwned,
    /// A per-player stone count disagrees with its plane. Also covers the total:
    /// `stone_count` is the sum of the per-player counts.
    StoneCountForPlayer,
    /// The `covered` plane disagrees with a recount of stones within `LEGAL_RADIUS`.
    Coverage,
    /// The derived frontier's population disagrees with the maintained counter.
    FrontierCount,
    /// `hash_cells` disagrees with a from-scratch recomputation.
    Zobrist,
    /// Terminal status disagrees with a brute-force six-in-a-row scan.
    Terminal,
    /// The reported winner is not the owner of the completed window.
    Winner,
    /// Phase or mover disagrees with the closed form of spec §10.2.
    TurnClosedForm,
    /// A stone lies within `LEGAL_RADIUS` of the arena boundary.
    ArenaMargin,
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [MoveError; 6] = [
        MoveError::TerminalState,
        MoveError::IllegalOpening,
        MoveError::CoordOutOfBounds(HexCoord::ORIGIN),
        MoveError::Occupied(HexCoord::ORIGIN),
        MoveError::TooFarFromStones(HexCoord::ORIGIN),
        MoveError::BoardExtentExceeded { cells: 1 },
    ];

    #[test]
    fn rule_violation_classification() {
        assert!(MoveError::TerminalState.is_rule_violation());
        assert!(MoveError::IllegalOpening.is_rule_violation());
        assert!(MoveError::Occupied(HexCoord::ORIGIN).is_rule_violation());
        assert!(MoveError::TooFarFromStones(HexCoord::ORIGIN).is_rule_violation());
        assert!(!MoveError::CoordOutOfBounds(HexCoord::ORIGIN).is_rule_violation());
        assert!(!MoveError::BoardExtentExceeded { cells: 9 }.is_rule_violation());
    }

    #[test]
    fn every_variant_displays_non_empty() {
        for e in ALL {
            let s = alloc_to_string(&e);
            assert!(!s.is_empty(), "{e:?} rendered empty");
        }
    }

    #[test]
    fn integrity_error_displays_with_and_without_a_coord() {
        let with = IntegrityError {
            check: IntegrityCheck::Coverage,
            coord: Some(HexCoord::new(2, -3)),
        };
        let without = IntegrityError {
            check: IntegrityCheck::FrontierCount,
            coord: None,
        };
        assert!(alloc_to_string(&with).contains("2, -3"));
        assert!(alloc_to_string(&without).contains("FrontierCount"));
    }

    fn alloc_to_string(e: &dyn core::fmt::Display) -> String {
        format!("{e}")
    }
}
