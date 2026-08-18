//! SplitMix64 state and sampling helpers.

/// A seeded SplitMix64 stream with 64 bits of state.
///
/// This type is `Clone` but not `Copy`; duplicating a stream must be explicit.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// A stream seeded with `seed`. Every seed is valid, including zero.
    #[inline]
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Restart the stream at `seed`, discarding whatever position it had.
    #[inline]
    pub fn reseed(&mut self, seed: u64) {
        self.state = seed;
    }

    /// The next 64 bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// The next draw in `[0, 1)`, with 53 bits of mantissa.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // 2^-53 times the top 53 bits: exactly representable, and never 1.0.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// A draw in `0..n`, by remainder.
    ///
    /// Remainder reduction has modulo bias. Callers requiring rejection
    /// sampling use [`SplitMix64::next_u64`] directly.
    ///
    /// # Panics
    ///
    /// If `n` is zero.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "SplitMix64::below(0): an empty range has no draw");
        (self.next_u64() % n as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference splitmix64 output for seed 0, which pins the constants.
    #[test]
    fn the_stream_matches_the_reference_vector_for_seed_zero() {
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
    }

    #[test]
    fn two_seeds_produce_different_streams() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn reseeding_restarts_the_stream_exactly() {
        let mut rng = SplitMix64::new(7);
        let first: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        rng.reseed(7);
        let again: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert_eq!(first, again);
        rng.reseed(8);
        let other: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert_ne!(first, other);
    }

    #[test]
    fn a_draw_below_n_stays_inside_the_range() {
        let mut rng = SplitMix64::new(0xdead_beef);
        for _ in 0..1000 {
            assert!(rng.below(7) < 7);
        }
        assert_eq!(rng.below(1), 0);
    }

    #[test]
    fn a_unit_draw_stays_inside_zero_to_one() {
        let mut rng = SplitMix64::new(99);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "{x}");
        }
    }

    #[test]
    #[should_panic(expected = "an empty range has no draw")]
    fn a_draw_below_zero_panics() {
        SplitMix64::new(0).below(0);
    }
}
