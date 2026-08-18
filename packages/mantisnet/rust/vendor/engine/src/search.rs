//! The borrow-scoped make/unmake session and the private undo token.

use crate::action::Action;
use crate::error::MoveError;
use crate::player::{Player, TurnPhase};
use crate::position::{Applied, Position};

/// Undo authority for one placement.
#[must_use]
#[derive(Debug)]
pub(crate) struct Undo {
    /// Class-I key for occupancy, cover, frontier, and hash.
    pub(crate) action: Action,
    /// Class II.
    pub(crate) phase_before: TurnPhase,
    /// Class II, and *also the mover* — one source of truth.
    pub(crate) player_before: Player,
    /// Debug-only misuse and drift detector.
    #[cfg(debug_assertions)]
    pub(crate) audit: UndoAudit,
}

/// Values captured before an apply, asserted (never assigned) on undo.
#[cfg(debug_assertions)]
#[derive(Debug)]
pub(crate) struct UndoAudit {
    /// `zobrist()` before the apply.
    pub(crate) zobrist_before: u64,
    /// `zobrist()` after the apply — the LIFO / wrong-position detector.
    pub(crate) zobrist_after: u64,
    /// `hash_cells` before the apply.
    pub(crate) hash_cells_before: u64,
    /// `frontier_cells` before the apply.
    pub(crate) frontier_before: u32,
    /// `stone_count()` before the apply.
    pub(crate) stones_before: u32,
    /// `stone_count_for(mover)` before the apply.
    pub(crate) stones_by_before: u32,
}

#[cfg(debug_assertions)]
impl UndoAudit {
    /// Snapshot the pre-apply values.
    pub(crate) fn capture(p: &Position) -> Self {
        Self {
            zobrist_before: p.zobrist(),
            zobrist_after: 0,
            hash_cells_before: p.hash_cells(),
            frontier_before: p.frontier_cells(),
            stones_before: p.stone_count(),
            stones_by_before: p.stone_count_for(p.current_player()),
        }
    }

    /// Record the post-apply hash.
    pub(crate) fn set_after(&mut self, zobrist_after: u64) {
        self.zobrist_after = zobrist_after;
    }
}

/// Exclusive make/unmake session over a position.
#[derive(Debug)]
pub struct Search<'p> {
    position: &'p mut Position,
    /// The ONLY undo authority in the crate.
    stack: Vec<Undo>,
}

impl<'p> Search<'p> {
    /// Begin a make/unmake session.
    pub fn new(position: &'p mut Position) -> Self {
        Self {
            position,
            stack: Vec::new(),
        }
    }

    /// Read the position at the current depth.
    #[inline]
    #[must_use]
    pub fn position(&self) -> &Position {
        self.position
    }

    /// Plies applied above the floor.
    #[inline]
    #[must_use]
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Whether no plies have been applied above the floor.
    #[inline]
    #[must_use]
    pub fn at_floor(&self) -> bool {
        self.stack.is_empty()
    }

    /// The placements applied above the floor, oldest first.
    pub fn path(&self) -> impl Iterator<Item = Action> + '_ {
        self.stack.iter().map(|u| u.action)
    }

    /// Apply one placement, recording how to reverse it.
    pub fn apply(&mut self, action: Action) -> Result<Applied, MoveError> {
        let (applied, undo) = self.position.apply_raw(action)?;
        self.stack.push(undo);
        Ok(applied)
    }

    /// Reverse the most recent [`Search::apply`], restoring the board, coverage,
    /// frontier, hash, phase, mover, and terminal status exactly.
    pub fn undo(&mut self) -> Option<Action> {
        let u = self.stack.pop()?;
        let action = u.action;
        self.position.undo_raw(u);
        Some(action)
    }

    /// Undo every ply back to the floor.
    pub fn unwind(&mut self) {
        while self.undo().is_some() {}
    }

    /// Move the floor to the current depth: the applied plies become permanent for this
    /// session and can no longer be undone.
    pub fn commit(&mut self) {
        self.stack.clear();
    }
}

impl Drop for Search<'_> {
    /// Unwinds to the floor.
    fn drop(&mut self) {
        self.unwind();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::HexCoord;

    fn act(q: i16, r: i16) -> Action {
        Action::new(HexCoord::new(q, r))
    }

    fn opened() -> Position {
        let mut p = Position::new();
        p.advance(act(0, 0)).expect("opening");
        p
    }

    #[test]
    fn undo_at_the_floor_is_the_identity() {
        let mut pos = opened();
        let before = pos.clone();
        let mut s = Search::new(&mut pos);
        assert!(s.at_floor());
        assert_eq!(s.depth(), 0);
        let z = s.position().zobrist();
        assert_eq!(s.undo(), None);
        assert_eq!(s.position().zobrist(), z);
        assert_eq!(s.undo(), None);
        drop(s);
        assert_eq!(pos, before);
    }

    #[test]
    fn apply_then_undo_returns_to_the_floor() {
        let mut pos = opened();
        let before = pos.clone();
        {
            let mut s = Search::new(&mut pos);
            s.apply(act(1, 0)).expect("legal");
            s.apply(act(2, 0)).expect("legal");
            assert_eq!(s.depth(), 2);
            assert_eq!(s.path().collect::<Vec<_>>(), vec![act(1, 0), act(2, 0)]);
            assert_eq!(s.undo(), Some(act(2, 0)));
            assert_eq!(s.undo(), Some(act(1, 0)));
            assert!(s.at_floor());
        }
        assert_eq!(pos, before);
    }

    #[test]
    fn drop_restores_the_callers_position_after_an_early_return() {
        let mut pos = opened();
        let before = pos.clone();
        fn early(pos: &mut Position) -> Result<(), MoveError> {
            let mut s = Search::new(pos);
            s.apply(act(1, 0))?;
            s.apply(act(2, 0))?;
            s.apply(act(1, 0))?;
            unreachable!("the third apply must fail");
        }
        let err = early(&mut pos).expect_err("must bail");
        assert_eq!(err, MoveError::Occupied(HexCoord::new(1, 0)));
        assert_eq!(pos, before, "Drop did not unwind to the floor");
    }

    #[test]
    fn commit_then_unwind_is_a_no_op() {
        let mut pos = opened();
        let mut s = Search::new(&mut pos);
        s.apply(act(1, 0)).expect("legal");
        s.apply(act(2, 0)).expect("legal");
        let committed = s.position().clone();
        s.commit();
        assert!(s.at_floor());
        assert_eq!(s.depth(), 0);
        s.unwind();
        assert_eq!(s.position(), &committed);
        assert_eq!(s.undo(), None);
        assert_eq!(s.position(), &committed);
        drop(s);
        assert_eq!(pos, committed);
    }

    #[test]
    fn commit_moves_the_floor_and_drop_honours_it() {
        let mut pos = opened();
        let committed;
        {
            let mut s = Search::new(&mut pos);
            s.apply(act(1, 0)).expect("legal");
            s.commit();
            committed = s.position().clone();
            s.apply(act(2, 0)).expect("legal");
            s.apply(act(3, 0)).expect("legal");
        }
        assert_eq!(pos, committed);
    }

    #[test]
    fn failed_apply_leaves_depth_unchanged() {
        let mut pos = opened();
        let mut s = Search::new(&mut pos);
        s.apply(act(1, 0)).expect("legal");
        assert_eq!(s.depth(), 1);
        let snapshot = s.position().clone();
        assert!(s.apply(act(1, 0)).is_err());
        assert_eq!(s.depth(), 1);
        assert_eq!(s.position(), &snapshot);
        assert!(s.apply(act(400, 400)).is_err());
        assert_eq!(s.depth(), 1);
        assert_eq!(s.position(), &snapshot);
    }

    #[test]
    fn unwind_from_depth_restores_exactly() {
        let mut pos = opened();
        let floor = pos.clone();
        let mut s = Search::new(&mut pos);
        for (q, r) in [(1i16, 0i16), (2, 0), (3, 0), (4, 0), (0, 1), (0, 2)] {
            s.apply(act(q, r)).expect("legal");
        }
        assert_eq!(s.depth(), 6);
        s.unwind();
        assert!(s.at_floor());
        assert_eq!(s.position(), &floor);
        s.position().audit().expect("audit after unwind");
    }
}
