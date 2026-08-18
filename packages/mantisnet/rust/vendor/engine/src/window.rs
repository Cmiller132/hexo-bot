//! Six-cell window geometry and the exposed win-detection surface.

use crate::coord::{Axis, HexCoord, WINDOW_LEN, hex_distance};
use crate::player::Player;

/// Windows touched by one placement: 3 axes × 6 offsets.
pub const WINDOWS_PER_PLACEMENT: usize = 18;

/// Every bit position inside a window.
const FULL: u8 = 0x3F;

/// Ownership of one six-cell window, as two six-bit masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct WindowMask([u8; 2]);

impl WindowMask {
    /// The empty window.
    pub const EMPTY: Self = Self([0, 0]);

    /// Build a mask from the two per-player lanes.
    #[inline]
    pub(crate) const fn from_lanes(p0: u8, p1: u8) -> Self {
        Self([p0 & FULL, p1 & FULL])
    }

    /// Bit `i` set iff cell `i` of the window holds a stone of `player`. Low six bits.
    #[inline]
    #[must_use]
    pub const fn mask(self, player: Player) -> u8 {
        self.0[player.index()]
    }

    /// Stones `player` holds in this window, `0..=6`.
    #[inline]
    #[must_use]
    pub const fn count(self, player: Player) -> u32 {
        self.0[player.index()].count_ones()
    }

    /// Either player's stones. `mask(P0) | mask(P1)`.
    #[inline]
    #[must_use]
    pub const fn occupied(self) -> u8 {
        self.0[0] | self.0[1]
    }

    /// Complement of [`WindowMask::occupied`] within the low six bits.
    #[inline]
    #[must_use]
    pub const fn empty(self) -> u8 {
        !self.occupied() & FULL
    }

    /// Whether `player` owns all six cells — the win condition for this window.
    #[inline]
    #[must_use]
    pub const fn is_full_for(self, player: Player) -> bool {
        self.0[player.index()] == FULL
    }
}

/// Identity of one six-cell window: its first cell and the axis it runs along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Window {
    /// Cell `0` of the window.
    pub start: HexCoord,
    /// The direction cells `1..6` run in.
    pub axis: Axis,
}

impl Window {
    /// Coordinate of cell `index`.
    #[inline]
    #[must_use]
    pub const fn cell(self, index: usize) -> HexCoord {
        assert!(index < WINDOW_LEN, "window cell index out of range");
        self.start.step(self.axis, index as i16)
    }

    /// All six coordinates, in bit order.
    #[inline]
    #[must_use]
    pub const fn cells(self) -> [HexCoord; WINDOW_LEN] {
        let mut out = [self.start; WINDOW_LEN];
        let mut i = 0;
        while i < WINDOW_LEN {
            out[i] = self.start.step(self.axis, i as i16);
            i += 1;
        }
        out
    }

    /// Which of the six cells `coord` is, or `None` if it is not one of them.
    #[inline]
    #[must_use]
    pub const fn cell_index(self, coord: HexCoord) -> Option<usize> {
        let dq = coord.q as i32 - self.start.q as i32;
        let dr = coord.r as i32 - self.start.r as i32;
        let (off_line, i) = match self.axis {
            Axis::Q => (dr, dq),
            Axis::R => (dq, dr),
            Axis::QR => (dq + dr, dq),
        };
        if off_line != 0 || i < 0 || i >= WINDOW_LEN as i32 {
            return None;
        }
        Some(i as usize)
    }

    /// Whether `coord` is one of this window's six cells.
    #[inline]
    #[must_use]
    pub const fn contains(self, coord: HexCoord) -> bool {
        self.cell_index(coord).is_some()
    }

    /// Whether the two windows share at least one cell. Symmetric.
    #[inline]
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        let mut i = 0;
        while i < WINDOW_LEN {
            if other.contains(self.cell(i)) {
                return true;
            }
            i += 1;
        }
        false
    }

    /// Whether the two windows are disjoint but have a pair of adjacent cells.
    #[inline]
    #[must_use]
    pub const fn touches(self, other: Self) -> bool {
        if self.intersects(other) {
            return false;
        }
        let mine = self.cells();
        let theirs = other.cells();
        let mut i = 0;
        while i < WINDOW_LEN {
            let mut j = 0;
            while j < WINDOW_LEN {
                if hex_distance(mine[i], theirs[j]) == 1 {
                    return true;
                }
                j += 1;
            }
            i += 1;
        }
        false
    }
}

/// A window paired with its current ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WindowRef {
    /// Which window.
    pub window: Window,
    /// Who owns which of its cells.
    pub mask: WindowMask,
}

/// A maximal run of one player's stones along one axis.
///
/// Produced by win detection, so `len >= WINDOW_LEN`. Maximal in both directions: the
/// cells one step before `start` and one step past the end are not that player's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Win {
    /// The axis the run lies along.
    pub axis: Axis,
    /// The first cell of the run — its end furthest back along `axis`.
    pub start: HexCoord,
    /// Cells in the run, `start` stepped `0..len` along `axis`.
    pub len: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mask_algebra() {
        let m = WindowMask::EMPTY;
        assert_eq!(m.mask(Player::P0), 0);
        assert_eq!(m.mask(Player::P1), 0);
        assert_eq!(m.occupied(), 0);
        assert_eq!(m.empty(), FULL);
        assert_eq!(m.count(Player::P0), 0);
        assert!(!m.is_full_for(Player::P0));
        assert_eq!(WindowMask::default(), WindowMask::EMPTY);
    }

    #[test]
    fn mask_accessor_algebra_over_every_disjoint_pair() {
        for a in 0u8..64 {
            for b in 0u8..64 {
                if a & b != 0 {
                    continue;
                }
                let m = WindowMask::from_lanes(a, b);
                assert_eq!(m.mask(Player::P0), a);
                assert_eq!(m.mask(Player::P1), b);
                assert_eq!(m.occupied(), a | b);
                assert_eq!(m.empty(), !(a | b) & FULL);
                assert_eq!(m.count(Player::P0), a.count_ones());
                assert_eq!(m.count(Player::P1), b.count_ones());
                assert_eq!(m.is_full_for(Player::P0), a == FULL);
                assert_eq!(m.is_full_for(Player::P1), b == FULL);
            }
        }
    }

    #[test]
    fn from_lanes_clamps_to_six_bits() {
        let m = WindowMask::from_lanes(0xFF, 0xC0);
        assert_eq!(m.mask(Player::P0), FULL);
        assert_eq!(m.mask(Player::P1), 0);
    }

    #[test]
    fn window_cells_walk_the_axis() {
        for axis in Axis::ALL {
            let w = Window {
                start: HexCoord::new(-4, 6),
                axis,
            };
            let cells = w.cells();
            assert_eq!(cells[0], w.start);
            for (i, &cell) in cells.iter().enumerate() {
                assert_eq!(cell, w.cell(i));
                assert_eq!(
                    crate::coord::hex_distance(w.start, cell),
                    i as u32,
                    "axis {axis:?} index {i}"
                );
            }
        }
    }

    /// Every window whose start lies in a small box, on all three axes.
    fn corpus() -> Vec<Window> {
        let mut out = Vec::new();
        for q in -3..=3 {
            for r in -3..=3 {
                for axis in Axis::ALL {
                    out.push(Window {
                        start: HexCoord::new(q, r),
                        axis,
                    });
                }
            }
        }
        out
    }

    /// The same three relations read off the materialised cell arrays.
    fn brute_contains(w: Window, c: HexCoord) -> bool {
        w.cells().contains(&c)
    }

    fn brute_intersects(a: Window, b: Window) -> bool {
        let (x, y) = (a.cells(), b.cells());
        x.iter().any(|c| y.contains(c))
    }

    fn brute_touches(a: Window, b: Window) -> bool {
        if brute_intersects(a, b) {
            return false;
        }
        let (x, y) = (a.cells(), b.cells());
        x.iter()
            .any(|p| y.iter().any(|q| hex_distance(*p, *q) == 1))
    }

    #[test]
    fn cell_index_inverts_cell() {
        for w in corpus() {
            for i in 0..WINDOW_LEN {
                assert_eq!(w.cell_index(w.cell(i)), Some(i), "{w:?} cell {i}");
            }
        }
    }

    #[test]
    fn contains_agrees_with_a_cell_walk_over_a_whole_neighbourhood() {
        for w in corpus() {
            for q in -9..=9 {
                for r in -9..=9 {
                    let c = HexCoord::new(q, r);
                    assert_eq!(w.contains(c), brute_contains(w, c), "{w:?} vs {c:?}");
                    assert_eq!(w.contains(c), w.cell_index(c).is_some());
                }
            }
        }
    }

    /// A coordinate on the window's line but past either end is not in it, and one step
    /// off the line never is.
    #[test]
    fn contains_rejects_off_line_and_past_the_ends() {
        for axis in Axis::ALL {
            let w = Window {
                start: HexCoord::ORIGIN,
                axis,
            };
            assert!(!w.contains(w.start.step(axis, -1)), "{axis:?} before start");
            assert!(
                !w.contains(w.start.step(axis, WINDOW_LEN as i16)),
                "{axis:?} past end"
            );
            for off in Axis::ALL {
                if off.index() == axis.index() {
                    continue;
                }
                for i in 0..WINDOW_LEN {
                    let beside = w.cell(i).step(off, 1);
                    assert_eq!(
                        w.contains(beside),
                        brute_contains(w, beside),
                        "{axis:?} cell {i} stepped along {off:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn intersects_agrees_with_a_cell_walk_and_is_symmetric() {
        let all = corpus();
        for &a in &all {
            for &b in &all {
                assert_eq!(a.intersects(b), brute_intersects(a, b), "{a:?} vs {b:?}");
                assert_eq!(a.intersects(b), b.intersects(a), "asymmetric {a:?} {b:?}");
            }
        }
    }

    /// A property the cell walk does not state: two windows on the same axis and the
    /// same line overlap exactly when their starts are within six steps.
    #[test]
    fn same_axis_windows_overlap_within_six_steps() {
        for axis in Axis::ALL {
            let a = Window {
                start: HexCoord::ORIGIN,
                axis,
            };
            for k in -8..=8i16 {
                let b = Window {
                    start: HexCoord::ORIGIN.step(axis, k),
                    axis,
                };
                assert_eq!(
                    a.intersects(b),
                    k.abs() < WINDOW_LEN as i16,
                    "{axis:?} offset {k}"
                );
            }
        }
    }

    #[test]
    fn touches_agrees_with_a_cell_walk_and_excludes_overlap() {
        let all = corpus();
        for &a in &all {
            for &b in &all {
                assert_eq!(a.touches(b), brute_touches(a, b), "{a:?} vs {b:?}");
                assert_eq!(a.touches(b), b.touches(a), "asymmetric {a:?} {b:?}");
                assert!(
                    !(a.intersects(b) && a.touches(b)),
                    "{a:?} and {b:?} both overlap and touch"
                );
            }
        }
        assert!(all.iter().any(|&a| all.iter().any(|&b| a.intersects(b))));
        assert!(all.iter().any(|&a| all.iter().any(|&b| a.touches(b))));
        assert!(
            all.iter()
                .any(|&a| all.iter().any(|&b| !a.intersects(b) && !a.touches(b)))
        );
    }

    /// A window always overlaps itself and never touches itself.
    #[test]
    fn a_window_intersects_itself() {
        for w in corpus() {
            assert!(w.intersects(w));
            assert!(!w.touches(w));
        }
    }

    /// Two windows six apart along the same axis are the adjacent-but-disjoint case by
    /// construction: cell 5 of the first neighbours cell 0 of the second.
    #[test]
    fn consecutive_collinear_windows_touch() {
        for axis in Axis::ALL {
            let a = Window {
                start: HexCoord::ORIGIN,
                axis,
            };
            let b = Window {
                start: HexCoord::ORIGIN.step(axis, WINDOW_LEN as i16),
                axis,
            };
            assert!(!a.intersects(b), "{axis:?}");
            assert!(a.touches(b), "{axis:?}");
        }
    }

    #[test]
    #[should_panic(expected = "window cell index out of range")]
    fn window_cell_panics_past_the_end() {
        let w = Window {
            start: HexCoord::ORIGIN,
            axis: Axis::Q,
        };
        let _ = w.cell(WINDOW_LEN);
    }

    #[test]
    fn windows_per_placement_is_three_axes_by_six_offsets() {
        assert_eq!(WINDOWS_PER_PLACEMENT, Axis::ALL.len() * WINDOW_LEN);
    }

    /// A `Win` names its cells by `start` stepped along `axis`, and the run it describes
    /// is the one the winning placement completed.
    #[test]
    fn win_cells_walk_the_axis_from_its_start() {
        for axis in Axis::ALL {
            let win = Win {
                axis,
                start: HexCoord::new(1, -2),
                len: 7,
            };
            let mut cell = win.start;
            for i in 0..win.len {
                assert_eq!(cell, win.start.step(axis, i as i16));
                assert_eq!(hex_distance(win.start, cell), u32::from(i));
                cell = cell.step(axis, 1);
            }
            assert_eq!(hex_distance(win.start, cell), u32::from(win.len));
        }
    }
}
