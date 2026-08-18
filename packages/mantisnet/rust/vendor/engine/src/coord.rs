//! Axial coordinates, the three line axes, hex distance, and the radius-8 disk.

/// Cells in a win window.
pub const WINDOW_LEN: usize = 6;

/// A non-opening placement must lie within this many hex steps of some stone.
pub const LEGAL_RADIUS: u32 = 8;

/// Cells in a radius-[`LEGAL_RADIUS`] hex disk: `3 * 8 * 9 + 1`.
pub const DISK_CELLS: usize = 217;

/// Largest magnitude allowed for any of `q`, `r`, `s` on a placed or queried cell.
pub const COORD_LIMIT: i16 = 16_000;

/// One cell on the unbounded hex board, in axial coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HexCoord {
    /// Axial `q` coordinate. Ordered first by `Ord`.
    pub q: i16,
    /// Axial `r` coordinate. Ordered second by `Ord`.
    pub r: i16,
}

impl HexCoord {
    /// The board centre `(0, 0)`, and the only legal opening placement.
    pub const ORIGIN: Self = Self { q: 0, r: 0 };

    /// Construct an axial coordinate. Total; performs no validation.
    #[inline]
    #[must_use]
    pub const fn new(q: i16, r: i16) -> Self {
        Self { q, r }
    }

    /// The derived cube axis `-q - r`.
    #[inline]
    #[must_use]
    pub const fn s(self) -> i32 {
        -(self.q as i32) - (self.r as i32)
    }

    /// Whether `q`, `r`, and `s` all lie within [`COORD_LIMIT`].
    #[inline]
    #[must_use]
    pub const fn is_valid(self) -> bool {
        let lim = COORD_LIMIT as i32;
        let q = self.q as i32;
        let r = self.r as i32;
        let s = -q - r;
        q >= -lim && q <= lim && r >= -lim && r <= lim && s >= -lim && s <= lim
    }

    /// This coordinate stepped `n` cells along `axis`.
    #[inline]
    #[must_use]
    pub const fn step(self, axis: Axis, n: i16) -> Self {
        debug_assert!(self.is_valid());
        debug_assert!(n >= -8 && n <= 8);
        let v = axis.vector();
        Self {
            q: self.q.wrapping_add(v.q.wrapping_mul(n)),
            r: self.r.wrapping_add(v.r.wrapping_mul(n)),
        }
    }
}

impl core::ops::Add for HexCoord {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            q: self.q.wrapping_add(rhs.q),
            r: self.r.wrapping_add(rhs.r),
        }
    }
}

impl core::ops::Sub for HexCoord {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            q: self.q.wrapping_sub(rhs.q),
            r: self.r.wrapping_sub(rhs.r),
        }
    }
}

/// The three straight-line axes a win window can run along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Axis {
    /// `(1, 0)`.
    Q,
    /// `(0, 1)`.
    R,
    /// `(1, -1)`.
    QR,
}

impl Axis {
    /// All three axes, in canonical order: `Q`, `R`, `QR`.
    pub const ALL: [Self; 3] = [Self::Q, Self::R, Self::QR];

    /// The unit step along this axis.
    #[inline]
    #[must_use]
    pub const fn vector(self) -> HexCoord {
        match self {
            Self::Q => HexCoord::new(1, 0),
            Self::R => HexCoord::new(0, 1),
            Self::QR => HexCoord::new(1, -1),
        }
    }

    /// Canonical index: `Q = 0`, `R = 1`, `QR = 2`.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Q => 0,
            Self::R => 1,
            Self::QR => 2,
        }
    }
}

/// Distance in hex steps between two cells.
#[inline]
#[must_use]
pub const fn hex_distance(a: HexCoord, b: HexCoord) -> u32 {
    let dq = a.q as i32 - b.q as i32;
    let dr = a.r as i32 - b.r as i32;
    let ds = a.s() - b.s();
    let aq = if dq < 0 { -dq } else { dq };
    let ar = if dr < 0 { -dr } else { dr };
    let as_ = if ds < 0 { -ds } else { ds };
    ((aq + ar + as_) / 2) as u32
}

/// Offsets of the radius-[`LEGAL_RADIUS`] disk, `dq`-major and `dr`-minor.
pub(crate) const DISK8: [(i8, i8); DISK_CELLS] = {
    let mut out = [(0i8, 0i8); DISK_CELLS];
    let mut n = 0usize;
    let mut dq = -8i8;
    while dq <= 8 {
        let lo = if -dq - 8 > -8 { -dq - 8 } else { -8 };
        let hi = if -dq + 8 < 8 { -dq + 8 } else { 8 };
        let mut dr = lo;
        while dr <= hi {
            out[n] = (dq, dr);
            n += 1;
            dr += 1;
        }
        dq += 1;
    }
    assert!(n == DISK_CELLS);
    out
};

/// `c` displaced by a `(dq, dr)` offset from [`DISK8`].
#[inline]
#[cfg_attr(not(debug_assertions), allow(dead_code))]
pub(crate) const fn offset(c: HexCoord, d: (i8, i8)) -> HexCoord {
    HexCoord {
        q: c.q.wrapping_add(d.0 as i16),
        r: c.r.wrapping_add(d.1 as i16),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s_is_total_at_i16_extremes() {
        assert_eq!(HexCoord::new(i16::MIN, i16::MIN).s(), 65536);
        assert_eq!(HexCoord::new(i16::MAX, i16::MAX).s(), -65534);
        assert_eq!(HexCoord::new(0, 0).s(), 0);
        assert_eq!(HexCoord::new(3, -5).s(), 2);
    }

    #[test]
    fn hex_distance_is_total_at_i16_extremes() {
        let a = HexCoord::new(i16::MIN, i16::MIN);
        let b = HexCoord::new(i16::MAX, i16::MAX);
        assert_eq!(hex_distance(a, b), hex_distance(b, a));
        assert_eq!(hex_distance(a, a), 0);
        assert_eq!(hex_distance(HexCoord::ORIGIN, HexCoord::new(1, 0)), 1);
        assert_eq!(hex_distance(HexCoord::ORIGIN, HexCoord::new(1, -1)), 1);
        assert_eq!(hex_distance(HexCoord::ORIGIN, HexCoord::new(0, 1)), 1);
        assert_eq!(hex_distance(HexCoord::ORIGIN, HexCoord::new(2, -1)), 2);
        assert_eq!(hex_distance(HexCoord::ORIGIN, HexCoord::new(-3, 0)), 3);
    }

    #[test]
    fn disk8_has_217_distinct_offsets_within_radius_8() {
        assert_eq!(DISK8.len(), DISK_CELLS);
        let mut seen = std::collections::HashSet::new();
        for &(dq, dr) in DISK8.iter() {
            assert!(seen.insert((dq, dr)), "duplicate offset {dq},{dr}");
            let c = HexCoord::new(dq as i16, dr as i16);
            assert!(
                hex_distance(HexCoord::ORIGIN, c) <= LEGAL_RADIUS,
                "offset {dq},{dr} outside radius"
            );
        }
        assert_eq!(seen.len(), DISK_CELLS);
    }

    #[test]
    fn disk8_covers_every_cell_within_radius_8() {
        let mut n = 0;
        for dq in -8i16..=8 {
            for dr in -8i16..=8 {
                let c = HexCoord::new(dq, dr);
                if hex_distance(HexCoord::ORIGIN, c) <= LEGAL_RADIUS {
                    n += 1;
                    assert!(DISK8.contains(&(dq as i8, dr as i8)));
                }
            }
        }
        assert_eq!(n, DISK_CELLS);
    }

    #[test]
    fn disk8_order_is_dq_major_dr_minor() {
        let mut prev = (i8::MIN, i8::MIN);
        for &d in DISK8.iter() {
            assert!(d > prev, "DISK8 not ascending at {d:?} after {prev:?}");
            prev = d;
        }
        assert_eq!(DISK8[0], (-8, 0));
        assert_eq!(DISK8[DISK_CELLS - 1], (8, 0));
        assert!(DISK8.contains(&(0, 0)));
    }

    #[test]
    fn is_valid_boundaries() {
        assert!(HexCoord::new(COORD_LIMIT, 0).is_valid());
        assert!(HexCoord::new(-COORD_LIMIT, 0).is_valid());
        assert!(HexCoord::new(0, COORD_LIMIT).is_valid());
        assert!(!HexCoord::new(COORD_LIMIT + 1, 0).is_valid());
        assert!(!HexCoord::new(0, COORD_LIMIT + 1).is_valid());
        assert!(HexCoord::new(COORD_LIMIT, -COORD_LIMIT).is_valid());
        assert!(!HexCoord::new(COORD_LIMIT, 1).is_valid());
        assert!(!HexCoord::new(-COORD_LIMIT, -1).is_valid());
        assert!(HexCoord::ORIGIN.is_valid());
    }

    #[test]
    fn axis_vectors_and_indices() {
        assert_eq!(Axis::Q.vector(), HexCoord::new(1, 0));
        assert_eq!(Axis::R.vector(), HexCoord::new(0, 1));
        assert_eq!(Axis::QR.vector(), HexCoord::new(1, -1));
        for (i, a) in Axis::ALL.iter().enumerate() {
            assert_eq!(a.index(), i);
            assert_eq!(hex_distance(HexCoord::ORIGIN, a.vector()), 1);
        }
    }

    #[test]
    fn step_walks_the_axis() {
        let c = HexCoord::new(3, -4);
        for axis in Axis::ALL {
            for n in -8i16..=8 {
                let stepped = c.step(axis, n);
                assert_eq!(hex_distance(c, stepped), n.unsigned_abs() as u32);
            }
        }
        assert_eq!(c.step(Axis::Q, 0), c);
    }

    #[test]
    fn add_and_sub_round_trip() {
        let a = HexCoord::new(12, -30);
        let b = HexCoord::new(-5, 7);
        assert_eq!((a + b) - b, a);
        assert_eq!(a + HexCoord::ORIGIN, a);
    }

    #[test]
    fn ord_is_lexicographic_q_then_r() {
        assert!(HexCoord::new(-1, 100) < HexCoord::new(0, -100));
        assert!(HexCoord::new(0, -1) < HexCoord::new(0, 0));
        assert!(HexCoord::new(5, 5) > HexCoord::new(5, 4));
    }
}
