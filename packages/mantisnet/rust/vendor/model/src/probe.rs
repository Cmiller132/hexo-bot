//! Frozen evaluator-probe positions and their output hash.
//!
//! The complete probe set is encoded as one batch and evaluated in one call.
//! The hash covers the exact little-endian output bytes in position order. The
//! positions are fixed move-list prefixes with no caller input or RNG.

use hexo_engine::{Action, HexCoord, Position};
use hexo_search::{EncodedBatch, Encoder, Evaluator};

/// FNV-1a's 64-bit offset basis.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a's 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// A packed game whose prefixes supply seven of the probe positions.
///
/// Prefixes cover the opening, turn boundaries, mid-turn states, and mid-game
/// states. The final position leaves both players one placement from a win.
static PACKED: [(i16, i16); 21] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (0, 2),
    (3, 0),
    (1, 1),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 1),
    (1, 3),
    (2, 2),
    (4, 0),
    (4, 1),
    (0, 4),
    (1, 4),
    (5, 1),
    (-1, 0),
    (2, -1),
    (3, -1),
];

/// `P1` on the *first* stone of a turn it can win with two placements: it holds
/// `(1, 0)..(4, 0)` and closes the window with `(5, 0)` and `(6, 0)`.
///
/// Both winning plies have the same mover, so this position distinguishes
/// mover-based value signing from depth-parity signing.
static WIN_IN_TWO: [(i16, i16); 9] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (0, 3),
    (3, 0),
    (4, 0),
    (0, 5),
    (0, 7),
];

/// `P1` on the *second* stone of a turn, one placement from six in a row: it
/// holds `(1, 0)`, `(2, 0)`, `(4, 0)`, `(5, 0)`, `(6, 0)`, and `(3, 0)` closes
/// the window.
static WIN_IN_ONE: [(i16, i16); 10] = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (0, 3),
    (4, 0),
    (5, 0),
    (0, 5),
    (0, 7),
    (6, 0),
];

/// Stones pushed out to the legality radius in every direction, so the frontier
/// is the union of thirteen disks rather than one.
///
/// Its legal count differs from the other probe positions and exercises ragged
/// encodings.
static SCATTERED: [(i16, i16); 13] = [
    (0, 0),
    (6, 0),
    (0, 6),
    (-6, 0),
    (0, -6),
    (6, -6),
    (-6, 6),
    (3, 3),
    (-3, -3),
    (8, 0),
    (0, 8),
    (-8, 0),
    (0, -8),
];

/// The move lists behind [`probe_positions`], in the order they are hashed.
fn probe_games() -> [&'static [(i16, i16)]; 10] {
    [
        // 0 plies: the opening, where the only legal placement is the origin
        // and `legal_count` is 1.
        &PACKED[..0],
        // 1 ply: `P1` on the first stone of the first full turn.
        &PACKED[..1],
        // 2 plies: `P1` one stone into its turn.
        &PACKED[..2],
        // 5 plies: two turns in.
        &PACKED[..5],
        // 9 plies: `P1` to move, two placements from a win.
        &WIN_IN_TWO,
        // 10 plies: `P1` mid-turn, one placement from a win.
        &WIN_IN_ONE,
        // 11 plies: a packed mid-game, `P0` to move.
        &PACKED[..11],
        // 12 plies: the same board one placement later, `P0` mid-turn.
        &PACKED[..12],
        // 13 plies: the widest frontier in the set.
        &SCATTERED,
        // 21 plies: the deepest board, with both sides one placement from a win.
        &PACKED,
    ]
}

/// The probe positions, in the order [`probe_hash`] forwards them.
///
/// Ten positions span plies 0, 1, 2, 5, 9, 10, 11, 12, 13, and 21, including
/// turn boundaries, mid-turn states, near-terminal states, and a wide frontier.
///
/// The set is format-frozen. Changing it changes every probe hash and requires
/// checkpoint regeneration.
///
/// # Panics
///
/// If a fixed move list is illegal or ends in a terminal position.
#[must_use]
pub fn probe_positions() -> Vec<Position> {
    probe_games()
        .iter()
        .map(|moves| {
            let actions: Vec<Action> = moves
                .iter()
                .map(|&(q, r)| Action::new(HexCoord::new(q, r)))
                .collect();
            let position = Position::replay(&actions)
                .unwrap_or_else(|e| panic!("probe game {moves:?} is not a legal game: {e}"));
            assert!(
                !position.is_terminal(),
                "probe game {moves:?} ends in a terminal position, which has no legal actions to \
                 carry priors for",
            );
            position
        })
        .collect()
}

/// The probe hash of the weights `evaluator` is holding.
///
/// Every probe position is encoded into **one** [`EncodedBatch`] and answered by
/// **one** [`Evaluator::evaluate`] call, and the hash folds the exact
/// little-endian bytes of every prior and every value, in order, through FNV-1a.
/// No rounding, bucketing, or summarization is applied.
///
/// # Panics
///
/// If the evaluator returns the wrong number of answers or any answer's prior
/// count differs from its position's `legal_count`.
pub fn probe_hash(encoder: &dyn Encoder, evaluator: &mut dyn Evaluator) -> u64 {
    let positions = probe_positions();
    let mut batch = EncodedBatch::with_capacity(positions.len(), 0);
    for position in &positions {
        batch.push_with(encoder, position);
    }

    let mut answers = Vec::with_capacity(positions.len());
    evaluator.evaluate(&batch, &mut answers);
    assert_eq!(
        answers.len(),
        positions.len(),
        "the evaluator answered {} of {} probe positions; the probe is one whole batch and one \
         forward, so a short answer list is a package that lost part of it",
        answers.len(),
        positions.len(),
    );

    let mut hash = FNV_OFFSET_BASIS;
    for (index, (position, evaluation)) in positions.iter().zip(&answers).enumerate() {
        assert_eq!(
            evaluation.priors.len(),
            position.legal_count(),
            "probe position {index} has {} legal actions but its evaluation carries {} priors; \
             priors are indexed by the engine's canonical legal order",
            position.legal_count(),
            evaluation.priors.len(),
        );
        for prior in &evaluation.priors {
            hash = fold(hash, &prior.to_le_bytes());
        }
        hash = fold(hash, &evaluation.value.to_le_bytes());
    }
    hash
}

/// One FNV-1a step per byte: xor, then multiply.
///
/// The local implementation fixes the hash algorithm and constants as part of
/// the checkpoint format.
fn fold(hash: u64, bytes: &[u8]) -> u64 {
    let mut hash = hash;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fold_matches_the_published_fnv1a_vectors() {
        // These vectors pin the offset basis, prime, and xor-then-multiply order.
        assert_eq!(fold(FNV_OFFSET_BASIS, b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fold(FNV_OFFSET_BASIS, b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fold(FNV_OFFSET_BASIS, b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn folding_in_two_steps_equals_folding_in_one() {
        let split = fold(fold(FNV_OFFSET_BASIS, b"foo"), b"bar");
        assert_eq!(split, fold(FNV_OFFSET_BASIS, b"foobar"));
    }

    #[test]
    fn the_probe_set_is_frozen() {
        // This vector freezes each position's zobrist and legal count,
        // independently of any encoder or evaluator.
        let mut hash = FNV_OFFSET_BASIS;
        for position in probe_positions() {
            hash = fold(hash, &position.zobrist().to_le_bytes());
            hash = fold(hash, &(position.legal_count() as u64).to_le_bytes());
        }
        assert_eq!(hash, 0x656d_6f60_cb31_b861);
    }
}
