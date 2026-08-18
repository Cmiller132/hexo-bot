//! Crate-private dense recentered arena.

use crate::MAX_GRID_CELLS;
use crate::coord::{COORD_LIMIT, HexCoord, LEGAL_RADIUS};
use crate::error::MoveError;
use crate::player::Player;

/// Rows the arena starts with: `q` spans 32 values.
const MIN_ROWS: usize = 32;

/// Rows a radius-[`LEGAL_RADIUS`] disk spans, and the length of its longest row.
const DISK_ROWS: usize = 2 * LEGAL_RADIUS as usize + 1;

/// Words per row the arena starts with: `r` spans 128 values.
const MIN_ROW_WORDS: usize = 2;

/// Margin, in cells, kept between any written cell and the arena boundary.
const PAD: i32 = LEGAL_RADIUS as i32;

/// Round down to a multiple of 64. Two's complement: `-1 -> -64`, `63 -> 0`.
#[inline]
const fn floor64(x: i32) -> i32 {
    x & !63
}

/// The low `n` bits. Total for `n < 64`.
#[inline]
const fn mask(n: usize) -> u64 {
    debug_assert!(n < 64);
    (1u64 << n) - 1
}

/// Bits `start .. start + n` of `plane`, as a mask whose bit `k` is cell `start + k`.
#[inline]
fn gather_run(plane: &[u64], start: usize, n: usize) -> u64 {
    debug_assert!(n > 0 && n <= DISK_ROWS, "run of {n} cells");
    let (w, sh) = (start >> 6, (start & 63) as u32);
    let mut v = plane[w] >> sh;
    if sh + n as u32 > 64 {
        v |= plane[w + 1] << (64 - sh);
    }
    v & mask(n)
}

/// The dense recentred arena.
#[derive(Clone, Debug)]
pub(crate) struct Grid {
    /// Extent along `q`.
    rows: usize,
    /// `u64` words per row; the extent along `r` is `64 * row_words`.
    row_words: usize,
    /// `q` of row 0.
    origin_q: i32,
    /// `r` of bit 0. Always a multiple of 64.
    origin_r: i32,
    /// Occupancy bit planes, `rows * row_words` words each.
    occ: [Vec<u64>; 2],
    /// Cells within [`LEGAL_RADIUS`] of a stone, `rows * row_words` words. The
    /// occupancy dilated by the radius-8 disk; the frontier is **derived** from it
    /// as `covered & !occupied`, never stored.
    covered: Vec<u64>,
    /// Maintained popcount of the derived frontier.
    frontier_cells: u32,
}

impl Grid {
    /// The empty arena.
    pub(crate) const fn new() -> Self {
        Self {
            rows: 0,
            row_words: 0,
            origin_q: 0,
            origin_r: 0,
            occ: [Vec::new(), Vec::new()],
            covered: Vec::new(),
            frontier_cells: 0,
        }
    }

    /// Rows currently allocated.
    #[inline]
    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    /// `u64` words per row.
    #[inline]
    pub(crate) const fn row_words(&self) -> usize {
        self.row_words
    }

    /// `q` of row 0.
    #[inline]
    pub(crate) const fn origin_q(&self) -> i32 {
        self.origin_q
    }

    /// `r` of bit 0.
    #[inline]
    pub(crate) const fn origin_r(&self) -> i32 {
        self.origin_r
    }

    /// Total words in one bit plane.
    #[inline]
    pub(crate) const fn total_words(&self) -> usize {
        self.rows * self.row_words
    }

    /// One occupancy plane.
    #[inline]
    pub(crate) fn occ_plane(&self, p: Player) -> &[u64] {
        &self.occ[p.index()]
    }

    /// The coverage plane.
    #[inline]
    pub(crate) fn covered_plane(&self) -> &[u64] {
        &self.covered
    }

    /// Word `i` of the derived frontier: covered and not occupied.
    #[inline]
    pub(crate) fn frontier_word(&self, i: usize) -> u64 {
        self.covered[i] & !self.occ[0][i] & !self.occ[1][i]
    }

    /// Maintained frontier population count.
    #[inline]
    pub(crate) const fn frontier_cells(&self) -> u32 {
        self.frontier_cells
    }

    /// Word `i` of the union of both occupancy planes.
    #[inline]
    pub(crate) fn occupied_word(&self, i: usize) -> u64 {
        self.occ[0][i] | self.occ[1][i]
    }

    /// The cell a `(word, bit)` slot addresses.
    #[inline]
    pub(crate) fn coord_of(&self, word: usize, bit: u32) -> HexCoord {
        let row = word / self.row_words;
        let w = word % self.row_words;
        HexCoord::new(
            (self.origin_q + row as i32) as i16,
            (self.origin_r + (w as i32) * 64 + bit as i32) as i16,
        )
    }

    /// `(word index, bit)` of `c`, or `None` if `c` is outside the arena.
    #[inline]
    fn locate(&self, c: HexCoord) -> Option<(usize, u32)> {
        if self.rows == 0 {
            return None;
        }
        let row = c.q as i32 - self.origin_q;
        if row < 0 || row >= self.rows as i32 {
            return None;
        }
        let bit = c.r as i32 - self.origin_r;
        if bit < 0 || bit >= 64 * self.row_words as i32 {
            return None;
        }
        Some((
            row as usize * self.row_words + (bit >> 6) as usize,
            (bit & 63) as u32,
        ))
    }

    /// `locate`, panicking.
    #[inline]
    fn locate_written(&self, c: HexCoord) -> (usize, u32) {
        match self.locate(c) {
            Some(x) => x,
            None => unreachable!("arena write outside the reserved region"),
        }
    }

    /// Owner of `c`, or `None`. Total over every coordinate.
    #[inline]
    pub(crate) fn owner(&self, c: HexCoord) -> Option<Player> {
        let (w, b) = self.locate(c)?;
        if (self.occ[0][w] >> b) & 1 == 1 {
            Some(Player::P0)
        } else if (self.occ[1][w] >> b) & 1 == 1 {
            Some(Player::P1)
        } else {
            None
        }
    }

    /// Owner of the stone at a `(word, bit)` slot of the occupancy planes.
    #[inline]
    pub(crate) fn owner_at(&self, word: usize, bit: u32) -> Player {
        debug_assert!(
            (self.occupied_word(word) >> bit) & 1 == 1,
            "an occupancy slot without a stone"
        );
        if (self.occ[0][word] >> bit) & 1 == 1 {
            Player::P0
        } else {
            Player::P1
        }
    }

    /// Whether both occupancy planes claim `c`. Always false in a sound arena.
    #[inline]
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    pub(crate) fn is_double_owned(&self, c: HexCoord) -> bool {
        match self.locate(c) {
            Some((w, b)) => ((self.occ[0][w] & self.occ[1][w]) >> b) & 1 == 1,
            None => false,
        }
    }

    /// Flat cell index of `c` within the bit planes, or `None` if `c`
    /// is outside the arena.
    #[inline]
    pub(crate) fn cell_index(&self, c: HexCoord) -> Option<usize> {
        let (w, b) = self.locate(c)?;
        Some(w * 64 + b as usize)
    }

    /// How many frontier cells precede `c` in canonical order, or `None` if `c`
    /// is not itself a frontier cell.
    pub(crate) fn frontier_rank(&self, c: HexCoord) -> Option<usize> {
        let (word, bit) = self.locate(c)?;
        if (self.frontier_word(word) >> bit) & 1 == 0 {
            return None;
        }
        let below: u32 = (0..word).map(|i| self.frontier_word(i).count_ones()).sum();
        let within = (self.frontier_word(word) & ((1u64 << bit) - 1)).count_ones();
        Some((below + within) as usize)
    }

    /// The frontier cell at `index` in canonical order, or `None` if `index` is past
    /// the end.
    pub(crate) fn nth_frontier(&self, index: usize) -> Option<HexCoord> {
        let mut remaining = index;
        for word in 0..self.total_words() {
            let bits = self.frontier_word(word);
            let pop = bits.count_ones() as usize;
            if remaining >= pop {
                remaining -= pop;
                continue;
            }
            let mut w = bits;
            for _ in 0..remaining {
                w &= w - 1;
            }
            return Some(self.coord_of(word, w.trailing_zeros()));
        }
        None
    }

    /// Whether `c` holds no stone. Total over every coordinate.
    #[inline]
    pub(crate) fn is_empty_cell(&self, c: HexCoord) -> bool {
        match self.locate(c) {
            Some((w, b)) => ((self.occ[0][w] | self.occ[1][w]) >> b) & 1 == 0,
            None => true,
        }
    }

    /// Whether `c` lies within [`LEGAL_RADIUS`] of some stone. Total: `false`
    /// outside the arena.
    #[inline]
    pub(crate) fn is_covered(&self, c: HexCoord) -> bool {
        match self.locate(c) {
            Some((w, b)) => (self.covered[w] >> b) & 1 == 1,
            None => false,
        }
    }

    /// Whether `c` is a frontier cell: covered and empty. Total: `false` outside
    /// the arena. Derived, never stored.
    #[inline]
    #[cfg(test)]
    pub(crate) fn frontier_bit(&self, c: HexCoord) -> bool {
        match self.locate(c) {
            Some((w, b)) => (self.frontier_word(w) >> b) & 1 == 1,
            None => false,
        }
    }

    /// The radius-[`LEGAL_RADIUS`] disk around `c` as one contiguous cell run per `q`
    /// row: `(first cell index, length)`, rows ascending in `q` and each run ascending
    /// in `r`.
    fn disk_runs(&self, c: HexCoord) -> [(usize, usize); DISK_ROWS] {
        debug_assert!(self.contains_padded(c), "disk outside the reserved region");
        let lim = COORD_LIMIT as i32;
        let rad = LEGAL_RADIUS as i32;
        let (cq, cr) = (c.q as i32, c.r as i32);
        let mut out = [(0usize, 0usize); DISK_ROWS];
        for (i, run) in out.iter_mut().enumerate() {
            let dq = i as i32 - rad;
            let q = cq + dq;
            if q < -lim || q > lim {
                continue;
            }
            let lo = (cr - rad).max(cr - dq - rad);
            let hi = (cr + rad).min(cr - dq + rad);
            let lo = lo.max(-lim).max(-lim - q);
            let hi = hi.min(lim).min(lim - q);
            if lo > hi {
                continue;
            }
            let row = (q - self.origin_q) as usize;
            let bit = (lo - self.origin_r) as usize;
            *run = (row * self.row_words * 64 + bit, (hi - lo + 1) as usize);
        }
        out
    }

    /// Popcount of the derived frontier across the disk runs. Both placement halves
    /// change the frontier only inside the placed cell's disk, so a before/after pair
    /// of these is the whole [`Grid::frontier_cells`] delta.
    fn frontier_pop_runs(&self, runs: &[(usize, usize); DISK_ROWS]) -> u32 {
        let mut pop = 0;
        for &(start, n) in runs {
            if n == 0 {
                continue;
            }
            let f = gather_run(&self.covered, start, n)
                & !gather_run(&self.occ[0], start, n)
                & !gather_run(&self.occ[1], start, n);
            pop += f.count_ones();
        }
        pop
    }

    /// OR `bits` into `covered`, where bit `k` is cell `start + k`. Idempotent.
    #[inline]
    fn cover_run(&mut self, start: usize, n: usize, bits: u64) {
        let (w, sh) = (start >> 6, (start & 63) as u32);
        self.covered[w] |= bits << sh;
        if sh as usize + n > 64 {
            self.covered[w + 1] |= bits >> (64 - sh);
        }
    }

    /// Replace the `n` covered bits at `start` with `bits`.
    #[inline]
    fn store_covered_run(&mut self, start: usize, n: usize, bits: u64) {
        debug_assert_eq!(bits & !mask(n), 0, "bits past the run");
        let (w, sh) = (start >> 6, (start & 63) as u32);
        self.covered[w] = (self.covered[w] & !(mask(n) << sh)) | (bits << sh);
        if sh as usize + n > 64 {
            let spill = 64 - sh;
            self.covered[w + 1] = (self.covered[w + 1] & !(mask(n) >> spill)) | (bits >> spill);
        }
    }

    /// Put `p`'s stone at `c`: occupancy bit, coverage disk, frontier count.
    pub(crate) fn place_stone(&mut self, c: HexCoord, p: Player) {
        let runs = self.disk_runs(c);
        let before = self.frontier_pop_runs(&runs);
        let (w, b) = self.locate_written(c);
        self.occ[p.index()][w] |= 1 << b;
        for &(start, n) in &runs {
            if n > 0 {
                self.cover_run(start, n, mask(n));
            }
        }
        let after = self.frontier_pop_runs(&runs);
        self.frontier_cells = self.frontier_cells - before + after;
    }

    /// The exact inverse of [`Grid::place_stone`]: clear the occupancy bit and
    /// recompute the coverage disk from the stones that remain.
    pub(crate) fn unplace_stone(&mut self, c: HexCoord, p: Player) {
        let runs = self.disk_runs(c);
        let before = self.frontier_pop_runs(&runs);
        let (w, b) = self.locate_written(c);
        debug_assert_eq!((self.occ[p.index()][w] >> b) & 1, 1, "not {p:?}'s stone");
        self.occ[p.index()][w] &= !(1u64 << b);
        self.recompute_covered_disk(c, &runs);
        let after = self.frontier_pop_runs(&runs);
        self.frontier_cells = self.frontier_cells - before + after;
    }

    /// Union occupancy of row `q`, bits `r0 .. r0 + n`, as a window word whose bit `j`
    /// is the cell `(q, r0 + j)`. Total: cells outside the arena read `0`.
    fn occupied_window_row(&self, q: i32, r0: i32, n: usize) -> u64 {
        let row = q - self.origin_q;
        if self.rows == 0 || row < 0 || row >= self.rows as i32 {
            return 0;
        }
        let total_bits = 64 * self.row_words as i32;
        let lo = (r0 - self.origin_r).max(0);
        let hi = (r0 + n as i32 - self.origin_r).min(total_bits);
        if lo >= hi {
            return 0;
        }
        let base = row as usize * self.row_words;
        let (len, w, sh) = ((hi - lo) as usize, (lo >> 6) as usize, (lo & 63) as u32);
        let plane = |i: usize| self.occ[0][base + i] | self.occ[1][base + i];
        let mut v = plane(w) >> sh;
        if sh as usize + len > 64 {
            v |= plane(w + 1) << (64 - sh);
        }
        (v & mask(len)) << (lo - (r0 - self.origin_r))
    }

    /// Recompute `covered` across the disk around `c` from occupancy alone.
    ///
    /// Coverage is occupancy dilated by the radius-[`LEGAL_RADIUS`] hex disk, and the
    /// disk is a zonogon: the Minkowski sum of the segments `0..=8` along `+Q`, `+R`,
    /// and `+QR`, translated by `(-8, 0)`. The dilation therefore factors into three
    /// 1-D dilations, each a log-shift schedule (spans 2, 4, 8, then 9). Removing the
    /// stone at `c` changes coverage only inside `c`'s disk, and any stone covering a
    /// cell of that disk lies within `2 * LEGAL_RADIUS` of `c`, so a 33x33 occupancy
    /// window suffices. Writeback goes through the domain-clipped [`Grid::disk_runs`],
    /// so no out-of-domain cell is ever painted covered.
    fn recompute_covered_disk(&mut self, c: HexCoord, runs: &[(usize, usize); DISK_ROWS]) {
        /// Window rows and bits: `c ± 2 * LEGAL_RADIUS`.
        const W: usize = 4 * LEGAL_RADIUS as usize + 1;
        let rad = LEGAL_RADIUS as usize;
        let (cq, cr) = (c.q as i32, c.r as i32);

        // t[i] bit j = a stone at (c.q - 16 + i, c.r - 16 + j).
        let mut t = [0u64; W];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot =
                self.occupied_window_row(cq - 2 * rad as i32 + i as i32, cr - 2 * rad as i32, W);
        }
        // Dilate by 0..=8 rows of +Q: t[i] |= t[i - d]. Descending, so each pass reads
        // the previous pass's spans, not its own.
        for d in [1usize, 2, 4, 1] {
            for i in (d..W).rev() {
                t[i] |= t[i - d];
            }
        }
        // Dilate by 0..=8 bits of +R: within-word shifts, no row coupling.
        for row in t.iter_mut() {
            let mut v = *row;
            v |= v << 1;
            v |= v << 2;
            v |= v << 4;
            v |= v << 1;
            *row = v;
        }
        // Dilate by 0..=8 steps of +QR: paired (+row, -bit) shifts.
        for d in [1usize, 2, 4, 1] {
            for i in (d..W).rev() {
                t[i] |= t[i - d] >> d;
            }
        }

        // Translate by (-8, 0) and splice: covered row (c.q - 8 + k) = t[16 + k], and
        // the domain-clipped run of that row selects which bits are written.
        for (k, &(start, n)) in runs.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let r = self.origin_r + (start % (self.row_words * 64)) as i32;
            let j0 = r - (cr - 2 * rad as i32);
            debug_assert!(
                (0..=(W as i32 - n as i32)).contains(&j0),
                "run outside the window"
            );
            let bits = (t[2 * rad + k] >> j0) & mask(n);
            self.store_covered_run(start, n, bits);
        }
    }

    /// Whether `[c.q ± PAD] × [c.r ± PAD]` is already inside the arena.
    #[inline]
    pub(crate) fn contains_padded(&self, c: HexCoord) -> bool {
        if self.rows == 0 {
            return false;
        }
        let (cq, cr) = (c.q as i32, c.r as i32);
        cq - PAD >= self.origin_q
            && cq + PAD < self.origin_q + self.rows as i32
            && cr - PAD >= self.origin_r
            && cr + PAD < self.origin_r + 64 * self.row_words as i32
    }

    /// Bounding box of the stones actually on the board, `(lo_q, hi_q, lo_r,
    /// hi_r)`, or `None` when no stone has been placed.
    fn stone_bounds(&self) -> Option<(i32, i32, i32, i32)> {
        let (mut lo_q, mut hi_q) = (i32::MAX, i32::MIN);
        let (mut lo_r, mut hi_r) = (i32::MAX, i32::MIN);
        for row in 0..self.rows {
            let base = row * self.row_words;
            let mut any = false;
            for w in 0..self.row_words {
                let bits = self.occ[0][base + w] | self.occ[1][base + w];
                if bits == 0 {
                    continue;
                }
                any = true;
                let word_r = self.origin_r + (w as i32) * 64;
                lo_r = lo_r.min(word_r + bits.trailing_zeros() as i32);
                hi_r = hi_r.max(word_r + 63 - bits.leading_zeros() as i32);
            }
            if any {
                let q = self.origin_q + row as i32;
                lo_q = lo_q.min(q);
                hi_q = hi_q.max(q);
            }
        }
        if hi_q == i32::MIN {
            None
        } else {
            Some((lo_q, hi_q, lo_r, hi_r))
        }
    }

    /// Grow, if needed, so `[c.q ± PAD] × [c.r ± PAD]` is inside the arena.
    pub(crate) fn reserve_around(&mut self, c: HexCoord) -> Result<(), MoveError> {
        if self.contains_padded(c) {
            return Ok(());
        }
        let (cq, cr) = (c.q as i32, c.r as i32);
        let bounds = self.stone_bounds();

        let (mut lo_q, mut hi_q) = (cq - PAD, cq + PAD);
        let (mut lo_r, mut hi_r) = (cr - PAD, cr + PAD);
        if let Some((sq0, sq1, sr0, sr1)) = bounds {
            lo_q = lo_q.min(sq0 - PAD);
            hi_q = hi_q.max(sq1 + PAD);
            lo_r = lo_r.min(sr0 - PAD);
            hi_r = hi_r.max(sr1 + PAD);
        }
        let need_rows = (hi_q - lo_q + 1) as usize;
        let base_r = floor64(lo_r);
        let need_words = ((hi_r - base_r) as usize / 64) + 1;

        let least_rows = MIN_ROWS.max(need_rows);
        let least_words = MIN_ROW_WORDS.max(need_words);
        let least_cells = least_rows as u64 * least_words as u64 * 64;
        if least_cells > MAX_GRID_CELLS {
            return Err(MoveError::BoardExtentExceeded { cells: least_cells });
        }

        let fits = |rows: usize, words: usize| rows as u64 * words as u64 * 64 <= MAX_GRID_CELLS;
        let bump = |have: usize, need: usize, min: usize| {
            let want = if need > have {
                (2 * have).max(need).next_power_of_two()
            } else {
                have.min(need.next_power_of_two().saturating_mul(4))
            };
            min.max(want)
        };
        let mut new_rows = bump(self.rows, need_rows, MIN_ROWS);
        let mut new_words = bump(self.row_words, need_words, MIN_ROW_WORDS);
        if !fits(new_rows, new_words) {
            new_words = MIN_ROW_WORDS.max(need_words.next_power_of_two());
            if !fits(least_rows, new_words) {
                new_words = least_words;
            }
            let budget = (MAX_GRID_CELLS / (64 * new_words as u64)) as usize;
            new_rows = new_rows.min(budget).max(least_rows);
        }
        debug_assert!(fits(new_rows, new_words), "chosen shape breaks the ceiling");
        debug_assert!(new_rows >= need_rows && new_words >= need_words);

        let new_origin_q = lo_q - ((new_rows - need_rows) / 2) as i32;
        let new_origin_r = base_r - 64 * ((new_words - need_words) / 2) as i32;

        let words = new_rows * new_words;
        let mut occ0 = vec![0u64; words];
        let mut occ1 = vec![0u64; words];
        let mut covered = vec![0u64; words];

        if let Some((sq0, sq1, sr0, sr1)) = bounds {
            let live_lo_q = (sq0 - PAD).max(self.origin_q);
            let live_hi_q = (sq1 + PAD).min(self.origin_q + self.rows as i32 - 1);
            let live_base_r = floor64((sr0 - PAD).max(self.origin_r));
            let live_hi_r = (sr1 + PAD).min(self.origin_r + 64 * self.row_words as i32 - 1);
            let n_rows = (live_hi_q - live_lo_q + 1) as usize;
            let n_words = ((live_hi_r - live_base_r) as usize / 64) + 1;

            debug_assert_eq!((live_base_r - self.origin_r) % 64, 0);
            let src_row0 = (live_lo_q - self.origin_q) as usize;
            let src_word0 = ((live_base_r - self.origin_r) / 64) as usize;
            let dst_row0 = (live_lo_q - new_origin_q) as usize;
            let dst_word0 = ((live_base_r - new_origin_r) / 64) as usize;
            debug_assert!(src_row0 + n_rows <= self.rows && src_word0 + n_words <= self.row_words);
            debug_assert!(dst_row0 + n_rows <= new_rows && dst_word0 + n_words <= new_words);

            for i in 0..n_rows {
                let src = (src_row0 + i) * self.row_words + src_word0;
                let dst = (dst_row0 + i) * new_words + dst_word0;
                occ0[dst..dst + n_words].copy_from_slice(&self.occ[0][src..src + n_words]);
                occ1[dst..dst + n_words].copy_from_slice(&self.occ[1][src..src + n_words]);
                covered[dst..dst + n_words].copy_from_slice(&self.covered[src..src + n_words]);
            }
        }

        self.rows = new_rows;
        self.row_words = new_words;
        self.origin_q = new_origin_q;
        self.origin_r = new_origin_r;
        self.occ = [occ0, occ1];
        self.covered = covered;

        debug_assert!(
            self.contains_padded(c),
            "C9: reserve_around failed to contain the requested cell"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord::{DISK_CELLS, DISK8, offset};

    fn grow(g: &mut Grid, q: i16, r: i16) {
        g.reserve_around(HexCoord::new(q, r)).expect("growth");
    }

    /// Reserve around `(q, r)` and place `P0`'s stone there, the way `Position` does.
    fn place(g: &mut Grid, q: i16, r: i16) {
        grow(g, q, r);
        g.place_stone(HexCoord::new(q, r), Player::P0);
    }

    fn unplace(g: &mut Grid, q: i16, r: i16) {
        g.unplace_stone(HexCoord::new(q, r), Player::P0);
    }

    fn cells(g: &Grid) -> u64 {
        g.rows() as u64 * g.row_words() as u64 * 64
    }

    /// Coverage of `c` recomputed straight off the offset table: any stone within the
    /// disk. Independent of both the run-OR that paints coverage on placement and the
    /// separable dilation that rewrites it on removal.
    fn covered_by_table(g: &Grid, c: HexCoord) -> bool {
        DISK8.iter().any(|&d| {
            let s = offset(c, d);
            s.is_valid() && g.owner(s).is_some()
        })
    }

    /// Every frontier cell in canonical order, computed by a coordinate walk
    /// independent of the word scan used by `frontier_rank` and `nth_frontier`.
    fn frontier_by_brute_force(g: &Grid) -> Vec<HexCoord> {
        let mut out = Vec::new();
        for row in 0..g.rows() {
            for bit in 0..(64 * g.row_words()) {
                let c = HexCoord::new(
                    (g.origin_q() + row as i32) as i16,
                    (g.origin_r() + bit as i32) as i16,
                );
                if g.frontier_bit(c) {
                    out.push(c);
                }
            }
        }
        out
    }

    /// The whole covered plane checked cell for cell against the table recount, and
    /// the frontier counter against a brute popcount. An off-domain cell must never
    /// read covered, however close a stone sits.
    fn assert_coverage_planes(g: &Grid) {
        let mut frontier = 0u32;
        for row in 0..g.rows() {
            for bit in 0..(64 * g.row_words()) {
                let c = HexCoord::new(
                    (g.origin_q() + row as i32) as i16,
                    (g.origin_r() + bit as i32) as i16,
                );
                let want = c.is_valid() && covered_by_table(g, c);
                assert_eq!(g.is_covered(c), want, "covered at ({}, {})", c.q, c.r);
                if want && g.owner(c).is_none() {
                    frontier += 1;
                }
            }
        }
        assert_eq!(frontier, g.frontier_cells(), "frontier counter");
    }

    /// Compare run-OR placement, separable-dilation removal, and offset-table
    /// coverage after each mutation and growth.
    #[test]
    fn coverage_matches_the_stone_recount_through_places_unplaces_and_growth() {
        let mut g = Grid::new();
        let script: [(i16, i16, Player); 8] = [
            (0, 0, Player::P0),
            (0, 63, Player::P1),
            (0, 64, Player::P0),
            (5, -3, Player::P1),
            (-7, 11, Player::P0),
            (12, 120, Player::P1),
            (3, 60, Player::P0),
            (-2, -8, Player::P1),
        ];
        for &(q, r, p) in &script {
            grow(&mut g, q, r);
            g.place_stone(HexCoord::new(q, r), p);
            assert_coverage_planes(&g);
        }
        for &(q, r, p) in &[script[3], script[0], script[7], script[5]] {
            g.unplace_stone(HexCoord::new(q, r), p);
            assert_coverage_planes(&g);
        }
        grow(&mut g, 80, -80);
        assert_coverage_planes(&g);
    }

    #[test]
    fn place_then_unplace_restores_every_plane_exactly() {
        let mut g = Grid::new();
        for &(q, r) in &[(0i16, 0i16), (4, 60), (-5, 7)] {
            place(&mut g, q, r);
        }
        grow(&mut g, 9, 62);
        let (occ0, occ1, covered, fc) = (
            g.occ[0].clone(),
            g.occ[1].clone(),
            g.covered.clone(),
            g.frontier_cells(),
        );
        g.place_stone(HexCoord::new(9, 62), Player::P1);
        assert_ne!(g.covered, covered, "the placement must extend coverage");
        g.unplace_stone(HexCoord::new(9, 62), Player::P1);
        assert_eq!(g.occ[0], occ0);
        assert_eq!(g.occ[1], occ1);
        assert_eq!(g.covered, covered);
        assert_eq!(g.frontier_cells(), fc);
    }

    #[test]
    fn frontier_rank_and_select_are_inverse_over_scattered_stones() {
        let mut g = Grid::new();
        for &(q, r) in &[(0i16, 0i16), (0, 63), (2, 70), (-6, -40), (9, 130)] {
            place(&mut g, q, r);
        }
        let expected = frontier_by_brute_force(&g);
        assert!(
            expected.len() > 400,
            "the disks should spread a real frontier"
        );
        assert_eq!(expected.len() as u32, g.frontier_cells());
        let mut sorted = expected.clone();
        sorted.sort_unstable();
        assert_eq!(expected, sorted, "the brute walk must already be canonical");

        for (i, &c) in expected.iter().enumerate() {
            assert_eq!(g.frontier_rank(c), Some(i), "rank of ({}, {})", c.q, c.r);
            assert_eq!(g.nth_frontier(i), Some(c), "nth_frontier({i})");
        }
        assert_eq!(g.nth_frontier(expected.len()), None);
        assert_eq!(g.nth_frontier(usize::MAX), None);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn frontier_select_rejects_an_index_above_u32_max() {
        let mut g = Grid::new();
        place(&mut g, 0, 0);
        assert_eq!(g.nth_frontier(u32::MAX as usize + 1), None);
    }

    #[test]
    fn frontier_rank_is_none_off_the_frontier() {
        let mut g = Grid::new();
        place(&mut g, 0, 0);
        assert_eq!(g.frontier_rank(HexCoord::ORIGIN), None, "occupied");
        assert_eq!(
            g.frontier_rank(HexCoord::new(9, 0)),
            None,
            "past the radius"
        );
        assert_eq!(
            g.frontier_rank(HexCoord::new(9000, 9000)),
            None,
            "off the arena"
        );
        let empty = Grid::new();
        assert_eq!(empty.frontier_rank(HexCoord::ORIGIN), None);
        assert_eq!(empty.nth_frontier(0), None);
    }

    #[test]
    fn empty_grid_allocates_nothing_and_reads_empty() {
        let g = Grid::new();
        assert_eq!(g.rows(), 0);
        assert_eq!(g.row_words(), 0);
        assert_eq!(g.total_words(), 0);
        assert_eq!(g.frontier_cells(), 0);
        assert!(g.is_empty_cell(HexCoord::ORIGIN));
        assert_eq!(g.owner(HexCoord::new(500, -500)), None);
        assert!(!g.is_covered(HexCoord::ORIGIN));
        assert!(!g.frontier_bit(HexCoord::ORIGIN));
        assert!(g.occ_plane(Player::P0).is_empty());
        assert!(g.covered_plane().is_empty());
    }

    #[test]
    fn first_growth_reaches_the_documented_minimum() {
        let mut g = Grid::new();
        grow(&mut g, 0, 0);
        assert_eq!(g.rows(), MIN_ROWS);
        assert_eq!(g.row_words(), MIN_ROW_WORDS);
        assert_eq!(g.origin_q(), -15);
        assert_eq!(g.origin_r(), -64);
        assert_eq!(g.origin_r() % 64, 0);
        assert!(g.contains_padded(HexCoord::ORIGIN));
    }

    #[test]
    fn growth_in_each_of_four_directions_keeps_origin_r_aligned() {
        for (dq, dr) in [(400i16, 0i16), (-400, 0), (0, 400), (0, -400)] {
            let mut g = Grid::new();
            place(&mut g, 0, 0);
            grow(&mut g, dq, dr);
            assert_eq!(g.origin_r() % 64, 0, "origin_r misaligned for {dq},{dr}");
            assert_eq!(g.owner(HexCoord::ORIGIN), Some(Player::P0));
            assert!(g.is_covered(HexCoord::ORIGIN));
            assert!(g.frontier_bit(HexCoord::new(1, 0)));
            assert!(g.contains_padded(HexCoord::new(dq, dr)));
            assert!(g.contains_padded(HexCoord::ORIGIN));
        }
    }

    #[test]
    fn grown_arena_reads_back_every_written_cell_and_zero_elsewhere() {
        let mut g = Grid::new();
        let written = [(0i16, 0i16), (3, -2), (-5, 7), (9, 9), (-11, -1)];
        for (i, &(q, r)) in written.iter().enumerate() {
            grow(&mut g, q, r);
            g.place_stone(
                HexCoord::new(q, r),
                if i % 2 == 0 { Player::P0 } else { Player::P1 },
            );
        }
        let expect: Vec<(HexCoord, Option<Player>)> = written
            .iter()
            .map(|&(q, r)| {
                let c = HexCoord::new(q, r);
                (c, g.owner(c))
            })
            .collect();
        assert!(
            expect
                .iter()
                .all(|&(c, owner)| owner.is_some() && g.is_covered(c))
        );

        for &(q, r) in &[(300i16, 300i16), (-300, -300), (300, -300), (-300, 300)] {
            grow(&mut g, q, r);
            for &(c, owner) in &expect {
                assert_eq!(g.owner(c), owner, "owner lost at {c:?}");
                assert!(g.is_covered(c), "coverage lost at {c:?}");
            }
            assert!(!g.is_covered(HexCoord::new(q, r)));
            assert!(g.is_empty_cell(HexCoord::new(q, r)));
            assert!(!g.frontier_bit(HexCoord::new(q, r)));
        }
    }

    /// Every cell index a walk of the row runs touches, in order.
    fn cells_by_runs(g: &Grid, c: HexCoord) -> Vec<usize> {
        g.disk_runs(c)
            .into_iter()
            .flat_map(|(start, n)| start..start + n)
            .collect()
    }

    /// The same set read through the `DISK8` offset table and `locate`.
    fn cells_by_table(g: &Grid, c: HexCoord) -> Vec<usize> {
        DISK8
            .iter()
            .map(|&d| offset(c, d))
            .filter(|cell| cell.is_valid())
            .map(|cell| g.cell_index(cell).expect("inside the reserved region"))
            .collect()
    }

    /// The row runs and the `DISK8` table are two independent statements of the same
    /// cell set: placement paints coverage through the runs, and tier C recounts it
    /// through the table.
    #[test]
    fn disk_runs_visit_exactly_the_disk8_cells_in_disk8_order() {
        for &(q, r) in &[(0i16, 0i16), (5, -3), (-7, 11), (40, 40), (-40, 13)] {
            let mut g = Grid::new();
            let c = HexCoord::new(q, r);
            grow(&mut g, q, r);
            let by_runs = cells_by_runs(&g, c);
            assert_eq!(by_runs, cells_by_table(&g, c), "at ({q}, {r})");
            assert_eq!(by_runs.len(), DISK_CELLS, "at ({q}, {r})");
        }
    }

    /// The per-row domain clip must agree cell for cell with `is_valid`, which is only
    /// observable within `LEGAL_RADIUS` of a face.
    #[test]
    fn disk_runs_clip_exactly_what_the_coordinate_domain_excludes() {
        for &(q, r) in &[
            (COORD_LIMIT, -COORD_LIMIT),
            (COORD_LIMIT, 0),
            (0, COORD_LIMIT),
            (-COORD_LIMIT, 0),
        ] {
            let mut g = Grid::new();
            let c = HexCoord::new(q, r);
            assert!(c.is_valid());
            grow(&mut g, q, r);
            let by_runs = cells_by_runs(&g, c);
            assert_eq!(by_runs, cells_by_table(&g, c), "at ({q}, {r})");
            assert!(
                by_runs.len() < DISK_CELLS,
                "({q}, {r}) is on a face; the domain must clip part of its disk"
            );
        }
    }

    /// Placing and removing the same stone restores every plane exactly, at a
    /// coordinate whose disk the domain clips — the case where the paint and the
    /// recompute could disagree about which cells exist.
    #[test]
    fn a_clipped_disk_round_trips() {
        let mut g = Grid::new();
        let c = HexCoord::new(COORD_LIMIT, -COORD_LIMIT + 3);
        grow(&mut g, c.q, c.r);
        g.place_stone(c, Player::P0);
        let covered: u32 = g.covered_plane().iter().map(|w| w.count_ones()).sum();
        assert!(covered > 0 && (covered as usize) < DISK_CELLS);
        assert_eq!(g.frontier_cells(), covered - 1, "the stone is not free");
        assert_coverage_planes(&g);

        g.unplace_stone(c, Player::P0);
        assert_eq!(g.frontier_cells(), 0);
        assert!(g.covered_plane().iter().all(|&w| w == 0));
    }

    #[test]
    fn the_frontier_counter_tracks_the_derived_plane() {
        let mut g = Grid::new();
        for &(q, r) in &[(0i16, 0i16), (1, 1), (0, 63), (14, -2)] {
            place(&mut g, q, r);
            let brute: u32 = (0..g.total_words())
                .map(|i| g.frontier_word(i).count_ones())
                .sum();
            assert_eq!(brute, g.frontier_cells());
        }
        for &(q, r) in &[(1i16, 1i16), (0, 0)] {
            unplace(&mut g, q, r);
            let brute: u32 = (0..g.total_words())
                .map(|i| g.frontier_word(i).count_ones())
                .sum();
            assert_eq!(brute, g.frontier_cells());
        }
    }

    #[test]
    fn max_grid_cells_is_refused_before_allocating() {
        let mut g = Grid::new();
        place(&mut g, 0, 0);
        let before_rows = g.rows();
        let before_words = g.row_words();
        let err = g
            .reserve_around(HexCoord::new(9000, 9000))
            .expect_err("must refuse");
        match err {
            MoveError::BoardExtentExceeded { cells } => assert!(cells > MAX_GRID_CELLS),
            other => panic!("wrong error: {other:?}"),
        }
        assert_eq!(g.rows(), before_rows, "arena mutated on refusal");
        assert_eq!(g.row_words(), before_words, "arena mutated on refusal");
    }

    /// A walk confined to `r = 0` must not increase the arena's `r` extent.
    #[test]
    fn a_q_only_walk_never_widens_r() {
        let mut g = Grid::new();
        place(&mut g, 0, 0);
        let mut q = 0i16;
        for _ in 0..600 {
            q += 8;
            place(&mut g, q, 0);
            assert_eq!(
                g.row_words(),
                MIN_ROW_WORDS,
                "r widened at q = {q} for a walk that never leaves r = 0"
            );
            assert!(cells(&g) <= MAX_GRID_CELLS);
        }
        assert!(g.rows() >= (q as usize) + 2 * PAD as usize);
        assert!(
            cells(&g) <= MAX_GRID_CELLS / 16,
            "{} cells for a one-row game",
            cells(&g)
        );
        for k in 0..=(q / 8) {
            assert_eq!(g.owner(HexCoord::new(k * 8, 0)), Some(Player::P0));
        }
    }

    /// The mirror walk.
    #[test]
    fn q_and_r_walks_reach_the_same_extent() {
        fn walk(along_q: bool) -> usize {
            let mut g = Grid::new();
            place(&mut g, 0, 0);
            let mut n = 0usize;
            for k in 1..2000i16 {
                let (q, r) = if along_q { (k * 8, 0) } else { (0, k * 8) };
                if g.reserve_around(HexCoord::new(q, r)).is_err() {
                    break;
                }
                g.place_stone(HexCoord::new(q, r), Player::P0);
                n += 1;
            }
            n
        }
        assert_eq!(walk(true), 1999, "the q walk hit the arena ceiling");
        assert_eq!(walk(false), 1999, "the r walk hit the arena ceiling");
    }

    /// Extent refusal depends on occupied bounds, not retained arena capacity.
    #[test]
    fn the_ceiling_is_a_function_of_the_stones_not_of_past_growth() {
        let mut inflated = Grid::new();
        place(&mut inflated, 0, 0);
        let mut q = 0i16;
        for _ in 0..400 {
            q += 8;
            place(&mut inflated, q, 0);
        }
        for k in 1..=(q / 8) {
            inflated.unplace_stone(HexCoord::new(k * 8, 0), Player::P0);
        }
        assert!(
            inflated.rows() > 1000,
            "the excursion did not grow the arena"
        );

        let mut fresh = Grid::new();
        place(&mut fresh, 0, 0);

        let (mut q, mut r) = (0i16, 0i16);
        let mut refused = false;
        for step in 0..1400 {
            if step % 2 == 0 {
                q += 8;
            } else {
                r += 8;
            }
            let c = HexCoord::new(q, r);
            let a = inflated.reserve_around(c);
            let b = fresh.reserve_around(c);
            assert_eq!(
                a.is_err(),
                b.is_err(),
                "grown and fresh arenas disagree at ({q}, {r}): {a:?} vs {b:?}"
            );
            if a.is_err() {
                assert_eq!(a, b, "different refusals at ({q}, {r})");
                refused = true;
                break;
            }
            inflated.place_stone(c, Player::P0);
            fresh.place_stone(c, Player::P0);
        }
        assert!(refused, "the diagonal never reached the ceiling");
    }

    #[test]
    fn coord_of_inverts_locate() {
        let mut g = Grid::new();
        grow(&mut g, 5, -70);
        for q in -10i16..=10 {
            for r in -100i16..=0 {
                let c = HexCoord::new(q, r);
                if let Some((w, b)) = g.locate(c) {
                    assert_eq!(g.coord_of(w, b), c);
                }
            }
        }
    }

    #[test]
    fn floor64_rounds_toward_negative_infinity() {
        assert_eq!(floor64(0), 0);
        assert_eq!(floor64(63), 0);
        assert_eq!(floor64(64), 64);
        assert_eq!(floor64(-1), -64);
        assert_eq!(floor64(-64), -64);
        assert_eq!(floor64(-65), -128);
    }

    #[test]
    fn repeated_growth_never_shrinks_or_loses_alignment() {
        let mut g = Grid::new();
        let mut q = 0i16;
        let mut r = 0i16;
        let mut prev_cells = 0u64;
        let mut placed = Vec::new();
        for step in 0..60 {
            place(&mut g, q, r);
            placed.push(HexCoord::new(q, r));
            let cells = cells(&g);
            assert!(cells >= prev_cells, "arena shrank at step {step}");
            assert!(cells <= MAX_GRID_CELLS);
            assert_eq!(g.origin_r() % 64, 0);
            prev_cells = cells;
            q = q.wrapping_add(if step % 2 == 0 { 8 } else { -8 });
            r = r.wrapping_add(if step % 3 == 0 { 8 } else { -8 });
        }
        for c in placed {
            assert_eq!(g.owner(c), Some(Player::P0), "lost the stone at {c:?}");
            assert!(g.contains_padded(c), "{c:?} lost its padding margin");
        }
    }

    /// A dimension the content does not need is handed back when the arena is re-
    /// shaped, so an excursion cannot leave a permanently bloated position.
    #[test]
    fn a_reshape_hands_back_capacity_the_content_no_longer_needs() {
        let mut g = Grid::new();
        place(&mut g, 0, 0);
        let mut q = 0i16;
        for _ in 0..200 {
            q += 8;
            place(&mut g, q, 0);
        }
        let tall = g.rows();
        assert!(
            tall >= 1024,
            "the q walk should have grown rows, got {tall}"
        );
        for k in 1..=(q / 8) {
            g.unplace_stone(HexCoord::new(k * 8, 0), Player::P0);
        }
        grow(&mut g, 0, 400);
        assert!(
            g.rows() < tall,
            "rows stayed at {tall} for a one-stone board"
        );
        assert_eq!(g.owner(HexCoord::ORIGIN), Some(Player::P0));
        assert!(g.contains_padded(HexCoord::ORIGIN));
        assert!(g.contains_padded(HexCoord::new(0, 400)));
    }
}
