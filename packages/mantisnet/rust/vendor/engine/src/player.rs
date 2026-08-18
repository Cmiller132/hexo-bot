//! Who moves, and where they are inside the two-placement turn.

/// One of the two players.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Player {
    /// Player 0. Moves first, at the origin.
    P0 = 0,
    /// Player 1. Moves second; takes plies 1 and 2.
    P1 = 1,
}

impl Player {
    /// The opposing player.
    #[inline]
    #[must_use]
    pub const fn other(self) -> Self {
        match self {
            Self::P0 => Self::P1,
            Self::P1 => Self::P0,
        }
    }

    /// `0` for `P0`, `1` for `P1`.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Where the mover is inside the two-placement turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TurnPhase {
    /// Ply 0 only.
    Opening,
    /// The mover places the first stone of its turn.
    FirstStone,
    /// The mover places the second stone of its turn.
    SecondStone,
}

impl TurnPhase {
    /// Canonical kind index: `Opening = 0`, `FirstStone = 1`, `SecondStone = 2`.
    #[inline]
    #[must_use]
    pub const fn kind_index(self) -> usize {
        match self {
            Self::Opening => 0,
            Self::FirstStone => 1,
            Self::SecondStone => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_other_is_an_involution() {
        assert_eq!(Player::P0.other(), Player::P1);
        assert_eq!(Player::P1.other(), Player::P0);
        assert_eq!(Player::P0.other().other(), Player::P0);
        assert_eq!(Player::P1.other().other(), Player::P1);
    }

    #[test]
    fn player_index_matches_repr() {
        assert_eq!(Player::P0.index(), 0);
        assert_eq!(Player::P1.index(), 1);
        assert_eq!(Player::P0 as u8, 0);
        assert_eq!(Player::P1 as u8, 1);
    }

    #[test]
    fn kind_index_is_the_zobrist_turn_order() {
        assert_eq!(TurnPhase::Opening.kind_index(), 0);
        assert_eq!(TurnPhase::FirstStone.kind_index(), 1);
        assert_eq!(TurnPhase::SecondStone.kind_index(), 2);
    }
}
