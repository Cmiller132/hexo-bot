//! Authoritative Hexo rules and game state.
//!
//! ```
//! use hexo_engine::{Action, HexCoord, Player, Position, TurnPhase};
//!
//! let mut pos = Position::new();
//! assert_eq!(pos.legal_count(), 1); // only the origin
//!
//! pos.advance(Action::new(HexCoord::ORIGIN)).unwrap();
//! assert_eq!(pos.current_player(), Player::P1);
//! assert_eq!(pos.phase(), TurnPhase::FirstStone);
//! assert_eq!(pos.legal_count(), 216); // the radius-8 disk, minus the origin
//!
//! // Legal moves come out in one canonical order: ascending `(q, r)`.
//! let first = pos.legal_actions().next().unwrap();
//! assert_eq!(first.coord(), HexCoord::new(-8, 0));
//!
//! // Rank and selection expose both directions of canonical action ordering.
//! assert_eq!(pos.legal_rank(first), Some(0));
//! assert_eq!(pos.nth_legal(0), Some(first));
//! ```
//!
//! ```
//! use hexo_engine::{Action, HexCoord, Position};
//!
//! // The move list belongs to whoever keeps the record; the position holds no copy.
//! let moves: Vec<Action> = [(0, 0), (1, 0), (2, 0)]
//!     .into_iter()
//!     .map(|(q, r)| Action::new(HexCoord::new(q, r)))
//!     .collect();
//!
//! let mut pos = Position::new();
//! for &a in &moves {
//!     pos.advance(a).unwrap();
//! }
//! assert_eq!(pos.stone_count(), 3);
//!
//! // A record round-trips through `replay`, and so does every prefix of it.
//! assert_eq!(Position::replay(&moves).unwrap(), pos);
//! assert_eq!(Position::replay(&moves[..1]).unwrap().stone_count(), 1);
//! ```
//!
//! ```
//! use hexo_engine::{Action, HexCoord, Position, Search};
//!
//! let mut pos = Position::new();
//! pos.advance(Action::new(HexCoord::ORIGIN)).unwrap();
//! let floor = pos.clone();
//! {
//!     let mut search = Search::new(&mut pos);
//!     search.apply(Action::new(HexCoord::new(1, 0))).unwrap();
//!     assert_eq!(search.depth(), 1);
//! } // Drop unwinds to the floor.
//! assert_eq!(pos, floor);
//! ```

pub mod action;
pub mod coord;
pub mod error;
mod grid;
pub mod player;
pub mod position;
pub mod search;
pub mod window;
mod zobrist;

pub use action::{ACTION_ORDER_VERSION, Action, ActionId};
pub use coord::{Axis, COORD_LIMIT, DISK_CELLS, HexCoord, LEGAL_RADIUS, WINDOW_LEN, hex_distance};
pub use error::{IntegrityCheck, IntegrityError, MoveError, ReplayError};
pub use player::{Player, TurnPhase};
pub use position::{Applied, LegalActions, Outcome, Position, Stones};
pub use search::Search;
pub use window::{WINDOWS_PER_PLACEMENT, Win, Window, WindowMask, WindowRef};

/// Version of the rules and of the Zobrist mixing function.
pub const RULES_VERSION: u32 = 1;

/// Hard ceiling on cells in one dense arena allocation.
pub const MAX_GRID_CELLS: u64 = 1 << 24;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_are_pinned() {
        assert_eq!(RULES_VERSION, 1);
        assert_eq!(ACTION_ORDER_VERSION, 1);
    }

    #[test]
    fn constants_are_self_consistent() {
        assert_eq!(WINDOW_LEN, 6);
        assert_eq!(LEGAL_RADIUS, 8);
        assert_eq!(DISK_CELLS, 3 * 8 * 9 + 1);
        assert_eq!(WINDOWS_PER_PLACEMENT, 18);
        assert_eq!(MAX_GRID_CELLS, 1 << 24);
        assert_eq!(COORD_LIMIT, 16_000);
    }
}
