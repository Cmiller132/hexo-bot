//! The rule machine: `Position`, its read surface, `advance`, and `audit`.

use crate::action::Action;
#[cfg(debug_assertions)]
use crate::coord::offset;
use crate::coord::{Axis, DISK8, HexCoord, LEGAL_RADIUS, WINDOW_LEN, hex_distance};
use crate::error::{IntegrityCheck, IntegrityError, MoveError, ReplayError};
use crate::grid::Grid;
use crate::player::{Player, TurnPhase};
use crate::search::Undo;
#[cfg(debug_assertions)]
use crate::search::UndoAudit;
use crate::window::{WINDOWS_PER_PLACEMENT, Win, Window, WindowMask, WindowRef};
use crate::zobrist::{TURN_KEY, cell_key};
use core::iter::FusedIterator;

#[cfg(test)]
#[path = "position_tests.rs"]
mod tests;

/// A Hexo position: board, turn phase, mover, hash, terminal status.
#[derive(Clone, Debug)]
pub struct Position {
    grid: Grid,
    phase: TurnPhase,
    current: Player,
    terminal: Option<Outcome>,
    /// XOR of cell keys only; the turn key is applied on read.
    hash_cells: u64,
    stones_by: [u32; 2],
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        // `stone_count()` is the sum of `stones_by`, so it is not compared separately.
        self.stones_by == other.stones_by
            && self.phase == other.phase
            && self.current == other.current
            && self.terminal == other.terminal
            && self.zobrist() == other.zobrist()
            && self.stones().eq(other.stones())
    }
}

impl Eq for Position {}

/// What one placement did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Applied {
    /// The placement that was made.
    pub action: Action,
    /// Who made it. Equals `current_player()` before the call.
    pub mover: Player,
    /// The phase before the placement.
    pub phase_before: TurnPhase,
    /// The phase after. Equals `phase_before` exactly when this placement won.
    pub phase_after: TurnPhase,
    /// `Some` iff this placement completed a six-window.
    pub outcome: Option<Outcome>,
    /// The run this placement completed on each axis, indexed by [`Axis::index`].
    ///
    /// Some entry is `Some` iff `outcome.is_some()`; two can be, when the placement
    /// completes two crossing lines at once. Iterate with `wins.iter().flatten()`.
    pub wins: [Option<Win>; 3],
}

/// How the game ended. Win only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Outcome {
    /// The player who completed a window.
    pub winner: Player,
}

impl Position {
    /// The empty position: `P0` to move, [`TurnPhase::Opening`], no arena allocated.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            grid: Grid::new(),
            phase: TurnPhase::Opening,
            current: Player::P0,
            terminal: None,
            hash_cells: 0,
            stones_by: [0, 0],
        }
    }

    /// Whose turn it is. Frozen at the winner once terminal.
    #[inline]
    #[must_use]
    pub const fn current_player(&self) -> Player {
        self.current
    }

    /// Where the mover is inside its turn. Frozen once terminal.
    #[inline]
    #[must_use]
    pub const fn phase(&self) -> TurnPhase {
        self.phase
    }

    /// The winner, if the game is over.
    #[inline]
    #[must_use]
    pub const fn outcome(&self) -> Option<Outcome> {
        self.terminal
    }

    /// Whether the game is over. Equivalent to `outcome().is_some()`.
    #[inline]
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.terminal.is_some()
    }

    /// `phase.kind_index() * 4 + current.index() * 2 + terminal.is_some()`.
    #[inline]
    const fn turn_slot(&self) -> usize {
        self.phase.kind_index() * 4 + self.current.index() * 2 + self.terminal.is_some() as usize
    }

    /// Incremental Zobrist hash.
    #[inline]
    #[must_use]
    pub const fn zobrist(&self) -> u64 {
        self.hash_cells ^ TURN_KEY[self.turn_slot()]
    }

    /// Maintained geometric frontier population, which is **not**
    /// [`Position::legal_count`].
    #[inline]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) const fn frontier_cells(&self) -> u32 {
        self.grid.frontier_cells()
    }

    /// The stone-only half of the hash, before the turn key is folded in.
    #[inline]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) const fn hash_cells(&self) -> u64 {
        self.hash_cells
    }
}

impl Position {
    /// Owner of `coord`, or `None` if empty.
    #[inline]
    #[must_use]
    pub fn get(&self, coord: HexCoord) -> Option<Player> {
        self.grid.owner(coord)
    }

    /// Whether no stone occupies `coord`. Total, as [`Position::get`].
    #[inline]
    #[must_use]
    pub fn is_empty_cell(&self, coord: HexCoord) -> bool {
        self.grid.is_empty_cell(coord)
    }

    /// Total stones placed.
    #[inline]
    #[must_use]
    pub const fn stone_count(&self) -> u32 {
        self.stones_by[0] + self.stones_by[1]
    }

    /// Stones held by one player.
    #[inline]
    #[must_use]
    pub const fn stone_count_for(&self, player: Player) -> u32 {
        self.stones_by[player.index()]
    }

    /// Every occupied cell with its owner, in canonical `(q, r)` order.
    #[must_use]
    pub fn stones(&self) -> Stones<'_> {
        Stones {
            scan: BitScan::new(&self.grid, ScanPlane::Occupied, self.stone_count() as usize),
        }
    }
}

impl Position {
    /// Rebuild a position by replaying a placement sequence from the empty board.
    pub fn replay(actions: &[Action]) -> Result<Self, ReplayError> {
        let mut pos = Self::new();
        pos.replay_from(actions)?;
        Ok(pos)
    }

    /// Apply a placement sequence to an existing position, continuing from where it
    /// stands.
    pub fn replay_from(&mut self, actions: &[Action]) -> Result<(), ReplayError> {
        for (ply, &action) in actions.iter().enumerate() {
            self.advance(action)
                .map_err(|cause| ReplayError { ply, action, cause })?;
        }
        Ok(())
    }
}

impl Position {
    /// Number of legal placements. `0` if and only if the position is terminal.
    #[must_use]
    pub const fn legal_count(&self) -> usize {
        if self.terminal.is_some() {
            0
        } else if matches!(self.phase, TurnPhase::Opening) {
            1
        } else {
            self.grid.frontier_cells() as usize
        }
    }

    /// Legal placements in canonical order (spec §9). Allocation-free.
    #[must_use]
    pub fn legal_actions(&self) -> LegalActions<'_> {
        let inner = if self.terminal.is_some() {
            LegalInner::Done
        } else if matches!(self.phase, TurnPhase::Opening) {
            LegalInner::Origin
        } else {
            LegalInner::Scan(BitScan::new(
                &self.grid,
                ScanPlane::Frontier,
                self.grid.frontier_cells() as usize,
            ))
        };
        LegalActions { inner }
    }

    /// Where `action` sits in [`Position::legal_actions`] order, or `None` if it is not
    /// legal here.
    #[must_use]
    pub fn legal_rank(&self, action: Action) -> Option<usize> {
        if self.terminal.is_some() {
            return None;
        }
        let c = action.coord();
        match self.phase {
            TurnPhase::Opening => (c == HexCoord::ORIGIN).then_some(0),
            TurnPhase::FirstStone | TurnPhase::SecondStone => self.grid.frontier_rank(c),
        }
    }

    /// The legal placement at `index` in [`Position::legal_actions`] order, or `None`
    /// if `index >= legal_count()`.
    #[must_use]
    pub fn nth_legal(&self, index: usize) -> Option<Action> {
        if self.terminal.is_some() {
            return None;
        }
        match self.phase {
            TurnPhase::Opening => (index == 0).then(|| Action::new(HexCoord::ORIGIN)),
            _ => self.grid.nth_frontier(index).map(Action::new),
        }
    }

    /// Whether `action` is legal right now: phase, occupancy, and radius.
    #[must_use]
    pub fn is_legal(&self, action: Action) -> bool {
        if self.terminal.is_some() {
            return false;
        }
        let c = action.coord();
        match self.phase {
            TurnPhase::Opening => c == HexCoord::ORIGIN,
            TurnPhase::FirstStone | TurnPhase::SecondStone => self.check_placement(c).is_ok(),
        }
    }

    /// Occupancy and radius legality, in precedence order.
    fn check_placement(&self, c: HexCoord) -> Result<(), MoveError> {
        if !c.is_valid() {
            // Off the coordinate domain, which the rules do not know exists. Classify
            // by what the rules would say: within LEGAL_RADIUS of a stone the
            // placement is rule-legal but unrepresentable — an engine limit — and
            // anywhere else it is plain TooFarFromStones. Occupied cannot happen
            // off-domain, because stones exist only on valid cells.
            return Err(if self.off_domain_within_radius(c) {
                MoveError::CoordOutOfBounds(c)
            } else {
                MoveError::TooFarFromStones(c)
            });
        }
        if self.grid.owner(c).is_some() {
            return Err(MoveError::Occupied(c));
        }
        if !self.grid.is_covered(c) {
            return Err(MoveError::TooFarFromStones(c));
        }
        Ok(())
    }

    /// Whether any stone lies within [`LEGAL_RADIUS`] of the off-domain cell `c`.
    /// Cold: reached only for off-domain placements. Probes in `i32`, so the walk
    /// cannot wrap `i16`; a neighbour that does not fit `i16` cannot hold a stone.
    fn off_domain_within_radius(&self, c: HexCoord) -> bool {
        let (q, r) = (i32::from(c.q), i32::from(c.r));
        DISK8.iter().any(|&(dq, dr)| {
            let (nq, nr) = (q + i32::from(dq), r + i32::from(dr));
            i32::from(nq as i16) == nq
                && i32::from(nr as i16) == nr
                && self
                    .grid
                    .owner(HexCoord::new(nq as i16, nr as i16))
                    .is_some()
        })
    }
}

impl Position {
    /// Ownership of the 18 windows through `coord`, in the canonical slot order of spec
    /// §6.3: axis-major (`Q`, `R`, `QR`), then offset `0..6`, where offset `k` means
    /// `coord` sits at bit `k` of the window.
    ///
    /// Near a coordinate-domain face, a returned window's start can be off-domain.
    /// Callers must skip those slots before passing the window to [`Position::window`].
    #[must_use]
    pub fn windows_through(&self, coord: HexCoord) -> [WindowRef; WINDOWS_PER_PLACEMENT] {
        debug_assert!(coord.is_valid());
        let mut out = [WindowRef {
            window: Window {
                start: coord,
                axis: Axis::Q,
            },
            mask: WindowMask::EMPTY,
        }; WINDOWS_PER_PLACEMENT];
        for axis in Axis::ALL {
            // Cell `coord` stepped `i - 5` along the axis. Every cell of every window
            // through `coord` on this axis is one of these eleven, so the whole axis is
            // one gather: bit `m` of slot `k` is the cell at `m - k`.
            let mut line = [None; 2 * WINDOW_LEN - 1];
            for (i, owner) in line.iter_mut().enumerate() {
                *owner = self.get(coord.step(axis, i as i16 - (WINDOW_LEN as i16 - 1)));
            }
            for k in 0..WINDOW_LEN {
                let mut m0 = 0u8;
                let mut m1 = 0u8;
                for m in 0..WINDOW_LEN {
                    match line[m + WINDOW_LEN - 1 - k] {
                        Some(Player::P0) => m0 |= 1 << m,
                        Some(Player::P1) => m1 |= 1 << m,
                        None => {}
                    }
                }
                out[axis.index() * WINDOW_LEN + k] = WindowRef {
                    window: Window {
                        start: coord.step(axis, -(k as i16)),
                        axis,
                    },
                    mask: WindowMask::from_lanes(m0, m1),
                };
            }
        }
        out
    }

    /// Ownership of one specific window.
    #[must_use]
    pub fn window(&self, window: Window) -> WindowMask {
        debug_assert!(window.start.is_valid());
        let mut m0 = 0u8;
        let mut m1 = 0u8;
        for (i, cell) in window.cells().into_iter().enumerate() {
            match self.grid.owner(cell) {
                Some(Player::P0) => m0 |= 1 << i,
                Some(Player::P1) => m1 |= 1 << i,
                None => {}
            }
        }
        WindowMask::from_lanes(m0, m1)
    }
}

/// The only phase transition. Private, called from exactly one site.
#[inline]
const fn advance_turn(before: TurnPhase, current: Player) -> (Player, TurnPhase) {
    match before {
        TurnPhase::Opening => (Player::P1, TurnPhase::FirstStone),
        TurnPhase::FirstStone => (current, TurnPhase::SecondStone),
        TurnPhase::SecondStone => (current.other(), TurnPhase::FirstStone),
    }
}

/// The turn closed form of spec §10.2: `(phase kind index, mover)` implied by
/// the stone count and the terminal bit alone.
pub(crate) const fn turn_closed_form(stones: u32, terminal: bool) -> Option<(usize, Player)> {
    if stones == 0 {
        return if terminal {
            None
        } else {
            Some((0, Player::P0))
        };
    }
    let m = stones - terminal as u32;
    if m == 0 {
        return None;
    }
    let kind = if m.is_multiple_of(2) { 2 } else { 1 };
    let player = if ((m - 1) / 2).is_multiple_of(2) {
        Player::P1
    } else {
        Player::P0
    };
    Some((kind, player))
}

impl Position {
    /// The forward half of a placement (spec §5.4).
    fn place(&mut self, c: HexCoord, p: Player) {
        debug_assert!(self.grid.is_empty_cell(c));
        self.grid.place_stone(c, p);
        self.hash_cells ^= cell_key(c, p);
        self.stones_by[p.index()] += 1;
    }

    /// The exact inverse of [`Position::place`]. Coverage is a pure function of the
    /// stone set — occupancy dilated by the disk — so removing the stone and
    /// recomputing its disk restores every plane exactly.
    fn unplace(&mut self, c: HexCoord, p: Player) {
        self.stones_by[p.index()] -= 1;
        self.hash_cells ^= cell_key(c, p);
        self.grid.unplace_stone(c, p);
    }

    /// The single forward code path, called by [`Position::advance`] and [`Search::apply`].
    pub(crate) fn apply_raw(&mut self, action: Action) -> Result<(Applied, Undo), MoveError> {
        let c = action.coord();

        if self.terminal.is_some() {
            return Err(MoveError::TerminalState);
        }
        match self.phase {
            TurnPhase::Opening => {
                if c != HexCoord::ORIGIN {
                    return Err(MoveError::IllegalOpening);
                }
            }
            // The second stone of a turn takes the same checks as the first: the rule that
            // it may not reuse the first is implied by occupancy. The first stone occupies
            // its cell and stones are permanent, so the reuse placement is already refused
            // as `Occupied`.
            TurnPhase::FirstStone | TurnPhase::SecondStone => self.check_placement(c)?,
        }
        let player_before = self.current;
        let phase_before = self.phase;

        self.grid.reserve_around(c)?;

        #[cfg(debug_assertions)]
        let mut audit = UndoAudit::capture(self);
        self.place(c, player_before);

        // The mover's maximal run through `c` on each axis. `get` is total, so the walk
        // needs no bounds test: it stops at the first cell that is not the mover's, and
        // never steps off one. No run of six existed before this placement — that would
        // have ended the game — so each side extends at most five and `len <= 11`.
        let mut wins: [Option<Win>; 3] = [None; 3];
        for axis in Axis::ALL {
            let mut back = 0u8;
            let mut probe = c.step(axis, -1);
            while self.get(probe) == Some(player_before) {
                back += 1;
                probe = probe.step(axis, -1);
            }
            let mut fwd = 0u8;
            let mut probe = c.step(axis, 1);
            while self.get(probe) == Some(player_before) {
                fwd += 1;
                probe = probe.step(axis, 1);
            }
            let len = back + fwd + 1;
            if len as usize >= WINDOW_LEN {
                wins[axis.index()] = Some(Win {
                    axis,
                    start: c.step(axis, -(back as i16)),
                    len,
                });
            }
        }

        let outcome = if wins.iter().any(Option::is_some) {
            let o = Outcome {
                winner: player_before,
            };
            self.terminal = Some(o);
            Some(o)
        } else {
            let (p, ph) = advance_turn(phase_before, player_before);
            self.current = p;
            self.phase = ph;
            None
        };

        #[cfg(debug_assertions)]
        {
            audit.set_after(self.zobrist());
            self.debug_assert_tier_c(c, player_before, &wins, &audit);
        }

        let applied = Applied {
            action,
            mover: player_before,
            phase_before,
            phase_after: self.phase,
            outcome,
            wins,
        };
        let undo = Undo {
            action,
            phase_before,
            player_before,
            #[cfg(debug_assertions)]
            audit,
        };
        Ok((applied, undo))
    }

    /// Reverse one [`Position::apply_raw`].
    pub(crate) fn undo_raw(&mut self, u: Undo) {
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            self.zobrist(),
            u.audit.zobrist_after,
            "C13: undo applied to the wrong position, or out of LIFO order"
        );

        self.phase = u.phase_before;
        self.current = u.player_before;
        self.terminal = None;
        self.unplace(u.action.coord(), u.player_before);

        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(self.zobrist(), u.audit.zobrist_before, "C14: zobrist");
            debug_assert_eq!(
                self.frontier_cells(),
                u.audit.frontier_before,
                "C14: frontier_cells"
            );
            debug_assert_eq!(self.stone_count(), u.audit.stones_before, "C14: stones");
            self.debug_assert_covered_around(u.action.coord());
            self.debug_assert_turn_closed_form();
            debug_assert_eq!(
                self.legal_count() == 0,
                self.terminal.is_some(),
                "C6: legal_count/terminal disagree"
            );
        }
    }

    /// Advance the position irreversibly by one placement.
    pub fn advance(&mut self, action: Action) -> Result<Applied, MoveError> {
        let (applied, _undo) = self.apply_raw(action)?;
        Ok(applied)
    }
}

#[cfg(debug_assertions)]
impl Position {
    /// C5: the turn closed form.
    fn debug_assert_turn_closed_form(&self) {
        let form = turn_closed_form(self.stone_count(), self.terminal.is_some());
        let (kind, player) = form.expect("C5: unreachable stones/terminal combination");
        debug_assert_eq!(self.phase.kind_index(), kind, "C5: phase kind");
        debug_assert_eq!(self.current, player, "C5: mover");
    }

    /// C1: every in-domain cell of the placed disk reads covered after an apply.
    /// Necessary-direction only; the complete recount is C2, paid on the rarer undo,
    /// and tier A's `audit` closes the sufficient direction at test checkpoints.
    fn debug_assert_disk_covered(&self, c: HexCoord) {
        for d in DISK8 {
            let cell = offset(c, d);
            debug_assert!(
                !cell.is_valid() || self.grid.is_covered(cell),
                "C1: ({}, {}) uncovered inside a placed disk",
                cell.q,
                cell.r
            );
        }
    }

    /// C2: coverage across the undone cell's disk agrees with a stone recount.
    ///
    /// The maintained plane was written by run-OR on apply and by the separable
    /// dilation on undo; this recount walks the `DISK8` offset table and probes the
    /// occupancy directly, so the three formulations are pairwise independent.
    fn debug_assert_covered_around(&self, c: HexCoord) {
        for d in DISK8 {
            let cell = offset(c, d);
            if !cell.is_valid() {
                continue;
            }
            let expect = DISK8.iter().any(|&e| {
                let s = offset(cell, e);
                s.is_valid() && self.grid.owner(s).is_some()
            });
            debug_assert_eq!(
                self.grid.is_covered(cell),
                expect,
                "C2: coverage disagrees with the stone recount at ({}, {})",
                cell.q,
                cell.r
            );
        }
    }

    /// C1, C3, C5, C6, C8, C10, C11, C12 after a successful apply.
    fn debug_assert_tier_c(
        &self,
        c: HexCoord,
        mover: Player,
        wins: &[Option<Win>; 3],
        audit: &UndoAudit,
    ) {
        debug_assert!(self.grid.contains_padded(c), "C8: arena margin");
        debug_assert!(!self.grid.is_double_owned(c), "C3: double-owned cell");
        debug_assert_eq!(self.stone_count(), audit.stones_before + 1, "C10: stones");
        debug_assert_eq!(self.get(c), Some(mover), "C10: owner");
        debug_assert_eq!(
            self.stones_by[mover.index()],
            audit.stones_by_before + 1,
            "C10: stones_by"
        );
        debug_assert_eq!(
            self.hash_cells,
            audit.hash_cells_before ^ cell_key(c, mover),
            "C10: hash_cells"
        );
        debug_assert_eq!(
            self.terminal.is_some(),
            wins.iter().any(Option::is_some),
            "C11: outcome/wins disagree"
        );
        // C11 does not re-derive the transition or the freeze: both are exactly what
        // C5's closed form pins from the stone count and the terminal bit, and a
        // re-derivation through `advance_turn` would check the function against
        // itself. Only the winner's identity is C11's own fact.
        if let Some(o) = self.terminal {
            debug_assert_eq!(o.winner, mover, "C11: winner is not the mover");
        }
        for axis in Axis::ALL {
            // A window whose start is off-domain holds a cell no stone can occupy, so it
            // is never full and is skipped rather than read (§4.4).
            let full = (0..WINDOW_LEN).any(|k| {
                let start = c.step(axis, -(k as i16));
                start.is_valid() && self.window(Window { start, axis }).is_full_for(mover)
            });
            debug_assert_eq!(
                wins[axis.index()].is_some(),
                full,
                "C12: win formulations disagree on {axis:?}"
            );
        }
        self.debug_assert_disk_covered(c);
        self.debug_assert_turn_closed_form();
        debug_assert_eq!(
            self.legal_count() == 0,
            self.terminal.is_some(),
            "C6: legal_count/terminal disagree"
        );
    }
}

/// Which bit plane a [`BitScan`] walks.
#[derive(Clone, Copy, Debug)]
enum ScanPlane {
    /// The legal set.
    Frontier,
    /// The union of both occupancy planes.
    Occupied,
}

/// A canonical-order walk over one bit plane.
#[derive(Clone, Debug)]
struct BitScan<'a> {
    grid: &'a Grid,
    plane: ScanPlane,
    word: usize,
    cur: u64,
    remaining: usize,
}

impl<'a> BitScan<'a> {
    fn new(grid: &'a Grid, plane: ScanPlane, remaining: usize) -> Self {
        let cur = if grid.total_words() == 0 {
            0
        } else {
            Self::word_at(grid, plane, 0)
        };
        Self {
            grid,
            plane,
            word: 0,
            cur,
            remaining,
        }
    }

    fn word_at(grid: &Grid, plane: ScanPlane, i: usize) -> u64 {
        match plane {
            ScanPlane::Frontier => grid.frontier_word(i),
            ScanPlane::Occupied => grid.occupied_word(i),
        }
    }

    /// The next set bit, as the `(word, bit)` slot that holds it.
    #[inline]
    fn next_slot(&mut self) -> Option<(usize, u32)> {
        if self.remaining == 0 {
            #[cfg(debug_assertions)]
            self.debug_assert_plane_exhausted();
            return None;
        }
        loop {
            if self.cur != 0 {
                let b = self.cur.trailing_zeros();
                self.cur &= self.cur - 1;
                self.remaining -= 1;
                return Some((self.word, b));
            }
            self.word += 1;
            if self.word >= self.grid.total_words() {
                debug_assert!(false, "the maintained population count exceeds the plane");
                self.remaining = 0;
                return None;
            }
            self.cur = Self::word_at(self.grid, self.plane, self.word);
        }
    }

    /// The next set bit as a coordinate, for the consumer that does not need the slot.
    #[inline]
    fn next_coord(&mut self) -> Option<HexCoord> {
        let (word, bit) = self.next_slot()?;
        Some(self.grid.coord_of(word, bit))
    }

    /// The plane really is exhausted when `remaining` says so.
    #[cfg(debug_assertions)]
    #[cold]
    fn debug_assert_plane_exhausted(&self) {
        debug_assert_eq!(self.cur, 0, "unyielded bits in the current word");
        debug_assert!(
            (self.word + 1..self.grid.total_words())
                .all(|i| Self::word_at(self.grid, self.plane, i) == 0),
            "the maintained population count is short of the plane"
        );
    }
}

/// Occupied cells with their owners, in canonical `(q, r)` order.
#[derive(Clone, Debug)]
pub struct Stones<'a> {
    scan: BitScan<'a>,
}

impl Iterator for Stones<'_> {
    type Item = (HexCoord, Player);

    fn next(&mut self) -> Option<Self::Item> {
        let (word, bit) = self.scan.next_slot()?;
        let grid = self.scan.grid;
        Some((grid.coord_of(word, bit), grid.owner_at(word, bit)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.scan.remaining, Some(self.scan.remaining))
    }
}

impl ExactSizeIterator for Stones<'_> {}
impl FusedIterator for Stones<'_> {}

/// Legal placements in canonical order (spec §9). Allocation-free.
#[derive(Clone, Debug)]
pub struct LegalActions<'a> {
    inner: LegalInner<'a>,
}

/// The three cases of spec §9, in the order they are tested.
#[derive(Clone, Debug)]
enum LegalInner<'a> {
    /// Terminal: nothing is legal.
    Done,
    /// `Opening`: exactly the origin. The frontier plane is not consulted.
    Origin,
    /// Everything else: the frontier bit scan.
    Scan(BitScan<'a>),
}

impl LegalActions<'_> {
    fn remaining(&self) -> usize {
        match &self.inner {
            LegalInner::Done => 0,
            LegalInner::Origin => 1,
            LegalInner::Scan(s) => s.remaining,
        }
    }
}

impl Iterator for LegalActions<'_> {
    type Item = Action;

    fn next(&mut self) -> Option<Action> {
        match &mut self.inner {
            LegalInner::Done => None,
            LegalInner::Origin => {
                self.inner = LegalInner::Done;
                Some(Action::new(HexCoord::ORIGIN))
            }
            LegalInner::Scan(s) => s.next_coord().map(Action::new),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.remaining();
        (n, Some(n))
    }
}

impl ExactSizeIterator for LegalActions<'_> {}
impl FusedIterator for LegalActions<'_> {}

/// Build an [`IntegrityError`].
#[inline]
const fn fail<T>(check: IntegrityCheck, coord: Option<HexCoord>) -> Result<T, IntegrityError> {
    Err(IntegrityError { check, coord })
}

impl Position {
    /// Recompute every derived structure from the stones alone and compare.
    pub fn audit(&self) -> Result<(), IntegrityError> {
        let g = &self.grid;
        let total = g.total_words();
        let occ0 = g.occ_plane(Player::P0);
        let occ1 = g.occ_plane(Player::P1);

        let pop0: u32 = occ0.iter().map(|w| w.count_ones()).sum();
        let pop1: u32 = occ1.iter().map(|w| w.count_ones()).sum();

        for i in 0..total {
            let both = occ0[i] & occ1[i];
            if both != 0 {
                return fail(
                    IntegrityCheck::DoubleOwned,
                    Some(g.coord_of(i, both.trailing_zeros())),
                );
            }
        }

        if self.stones_by[0] != pop0 || self.stones_by[1] != pop1 {
            return fail(IntegrityCheck::StoneCountForPlayer, None);
        }

        let mut stones: Vec<(HexCoord, Player)> = Vec::with_capacity(self.stone_count() as usize);
        for i in 0..total {
            let mut w = occ0[i] | occ1[i];
            while w != 0 {
                let b = w.trailing_zeros();
                w &= w - 1;
                let owner = if (occ0[i] >> b) & 1 == 1 {
                    Player::P0
                } else {
                    Player::P1
                };
                stones.push((g.coord_of(i, b), owner));
            }
        }

        let pad = LEGAL_RADIUS as i32;
        let (lo_q, hi_q) = (g.origin_q(), g.origin_q() + g.rows() as i32 - 1);
        let (lo_r, hi_r) = (g.origin_r(), g.origin_r() + 64 * g.row_words() as i32 - 1);
        for &(c, _) in &stones {
            let (q, r) = (c.q as i32, c.r as i32);
            if q - pad < lo_q || q + pad > hi_q || r - pad < lo_r || r + pad > hi_r {
                return fail(IntegrityCheck::ArenaMargin, Some(c));
            }
        }

        let mut recount = vec![0u64; total];
        for &(s, _) in &stones {
            for dq in -(pad as i16)..=(pad as i16) {
                for dr in -(pad as i16)..=(pad as i16) {
                    let cell = HexCoord::new(s.q + dq, s.r + dr);
                    if hex_distance(s, cell) > LEGAL_RADIUS || !cell.is_valid() {
                        continue;
                    }
                    let idx = match g.cell_index(cell) {
                        Some(i) => i,
                        None => return fail(IntegrityCheck::ArenaMargin, Some(cell)),
                    };
                    recount[idx / 64] |= 1 << (idx % 64);
                }
            }
        }
        let covered = g.covered_plane();
        for (i, &want) in recount.iter().enumerate() {
            if covered[i] != want {
                let bad = (covered[i] ^ want).trailing_zeros();
                return fail(IntegrityCheck::Coverage, Some(g.coord_of(i, bad)));
            }
        }

        let fpop: u32 = (0..total).map(|i| g.frontier_word(i).count_ones()).sum();
        if fpop != g.frontier_cells() {
            return fail(IntegrityCheck::FrontierCount, None);
        }

        let mut h = 0u64;
        for &(c, p) in &stones {
            h ^= cell_key(c, p);
        }
        if h != self.hash_cells {
            return fail(IntegrityCheck::Zobrist, None);
        }

        let mut winners = [false; 2];
        for &(c, p) in &stones {
            for axis in Axis::ALL {
                for k in 0..WINDOW_LEN {
                    let mut all = true;
                    for m in 0..WINDOW_LEN {
                        let cell = c.step(axis, m as i16 - k as i16);
                        if self.get(cell) != Some(p) {
                            all = false;
                            break;
                        }
                    }
                    if all {
                        winners[p.index()] = true;
                    }
                }
            }
        }
        if (winners[0] || winners[1]) != self.terminal.is_some() {
            return fail(IntegrityCheck::Terminal, None);
        }

        if let Some(o) = self.terminal
            && (!winners[o.winner.index()] || winners[o.winner.other().index()])
        {
            return fail(IntegrityCheck::Winner, None);
        }

        match turn_closed_form(self.stone_count(), self.terminal.is_some()) {
            Some((kind, player)) if kind == self.phase.kind_index() && player == self.current => {}
            _ => return fail(IntegrityCheck::TurnClosedForm, None),
        }

        Ok(())
    }
}
