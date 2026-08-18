//! The deterministic Zobrist mixing function and the turn-key table.

use crate::coord::HexCoord;
use crate::player::Player;

/// splitmix64 finalizer.
#[inline]
const fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    x
}

/// Domain tag bit for cell keys.
const CELL_DOMAIN: u64 = 1 << 16;

/// Domain tag bit for turn keys.
const TURN_DOMAIN: u64 = 1 << 17;

/// Hash contribution of one stone.
#[inline]
pub(crate) const fn cell_key(c: HexCoord, p: Player) -> u64 {
    mix64(((c.q as u16 as u64) << 48) | ((c.r as u16 as u64) << 32) | CELL_DOMAIN | (p as u64))
}

/// Hash contribution of the turn state.
#[inline]
const fn turn_key_of(slot: usize) -> u64 {
    mix64(TURN_DOMAIN | ((slot as u64) << 1))
}

/// Number of distinct turn slots: 3 phase kinds × 2 players × 2 terminal states.
pub(crate) const TURN_SLOTS: usize = 12;

/// The twelve turn keys, baked at compile time.
pub(crate) const TURN_KEY: [u64; TURN_SLOTS] = {
    let mut t = [0u64; TURN_SLOTS];
    let mut i = 0;
    while i < TURN_SLOTS {
        t[i] = turn_key_of(i);
        i += 1;
    }
    t
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Frozen golden vectors: sixteen `(q, r, player) -> u64` cell keys spanning all
    /// four sign quadrants and both players.
    #[test]
    fn cell_key_golden_vectors() {
        const GOLDEN: [(i16, i16, Player, u64); 16] = [
            (0, 0, Player::P0, 0xceb5_a1e1_5fdb_5cf5),
            (0, 0, Player::P1, 0x1644_d334_51f3_f4fb),
            (1, 0, Player::P0, 0x4d92_5f39_d2d5_6fb7),
            (0, 1, Player::P1, 0x213b_fc08_e153_5d80),
            (-1, 0, Player::P0, 0x1b91_7676_976d_bace),
            (0, -1, Player::P1, 0x99e1_9a11_ed0b_d6ab),
            (7, -3, Player::P0, 0x441f_7808_4c62_46af),
            (-7, 3, Player::P1, 0xc60d_1f43_5a50_c2d3),
            (7, 3, Player::P0, 0x1475_b079_207a_aed2),
            (-7, -3, Player::P1, 0x712f_af77_ac71_ca03),
            (16000, 16000, Player::P0, 0x2919_52a8_7d4e_ecb1),
            (-16000, -16000, Player::P1, 0xb2c4_cee2_330d_6268),
            (16000, -16000, Player::P0, 0xf84f_e82f_43f8_29e3),
            (-16000, 16000, Player::P1, 0xf584_7e80_d7f4_adb5),
            (i16::MIN, i16::MAX, Player::P0, 0xca10_68bb_4d86_4913),
            (i16::MAX, i16::MIN, Player::P1, 0x4d06_3642_0328_1b90),
        ];
        for (q, r, p, expected) in GOLDEN {
            let got = cell_key(HexCoord::new(q, r), p);
            assert_eq!(got, expected, "cell_key(({q}, {r}), {p:?}) = {got:#018x}");
        }
    }

    /// The twelve frozen turn keys.
    #[test]
    fn turn_key_golden_vectors() {
        const GOLDEN: [u64; TURN_SLOTS] = [
            0x323b_8d7c_fce6_4aaa,
            0x2c89_a668_a3e7_e9f7,
            0xddc2_b9b3_c597_f545,
            0x378c_014e_e02c_5f2e,
            0x902a_6273_a474_6cef,
            0x67d3_cf2e_9fd9_4611,
            0x354b_b276_40ed_f6f4,
            0x759e_6f74_91ec_334f,
            0xef1b_625b_3f88_3fd4,
            0x9e33_3049_6503_0378,
            0xfdeb_1016_5eb0_b848,
            0x2b67_9ad5_64f5_0010,
        ];
        assert_eq!(TURN_KEY, GOLDEN);
    }

    #[test]
    fn mix64_is_injective_over_a_sample() {
        let mut seen = std::collections::HashSet::new();
        for i in 0u64..5000 {
            assert!(seen.insert(mix64(i)));
        }
        assert_eq!(mix64(0), 0);
    }

    #[test]
    fn cell_keys_are_distinct_over_a_grid_and_both_players() {
        let mut seen = std::collections::HashSet::new();
        for q in -40i16..=40 {
            for r in -40i16..=40 {
                for p in [Player::P0, Player::P1] {
                    assert!(
                        seen.insert(cell_key(HexCoord::new(q, r), p)),
                        "collision at ({q}, {r}) {p:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn turn_keys_are_distinct_and_disjoint_from_cell_keys() {
        let mut seen = std::collections::HashSet::new();
        for k in TURN_KEY {
            assert!(seen.insert(k));
        }
        for q in -20i16..=20 {
            for r in -20i16..=20 {
                for p in [Player::P0, Player::P1] {
                    assert!(!seen.contains(&cell_key(HexCoord::new(q, r), p)));
                }
            }
        }
    }
}
