//! Position encoding, evaluator outputs, and reusable encoded batches.

use crate::session::LeafId;
use hexo_engine::Position;

/// One network answer for one evaluated position.
///
/// Both fields follow conventions required of every model package:
///
/// - `priors` are in the engine's canonical legal order. Entry `i` belongs to
///   `nth_legal(i)` of the position that was evaluated, and the length must
///   equal that position's `legal_count()`.
/// - `value` is the expected outcome from the perspective of the **side to move
///   at the evaluated position**, not of the seat that is searching and not of
///   `P0`. Consecutive plies may have the same mover, so search signs values by
///   comparing movers rather than by depth parity.
///
/// A session checks both on delivery: a length mismatch or an out-of-range value
/// panics rather than being clamped or padded.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// Prior probabilities over the evaluated position's legal actions, in the
    /// engine's canonical order. Non-negative and finite; not required to sum to
    /// exactly one, since a masked-and-renormalised head cannot promise that.
    pub priors: Box<[f32]>,
    /// Expected outcome in `[-1, 1]` from the perspective of the side to move at
    /// the evaluated position.
    pub value: f32,
}

impl Evaluation {
    /// Validate this answer against [`Evaluation`]'s conventions.
    ///
    /// # Panics
    ///
    /// If the prior count does not match `legal_count`, if any prior is negative
    /// or non-finite, or if the value is outside `[-1, 1]`.
    pub(crate) fn check(&self, legal_count: usize, leaf: LeafId) {
        assert_eq!(
            self.priors.len(),
            legal_count,
            "{leaf:?}: the evaluation carries {} priors but the evaluated position has \
             {legal_count} legal actions. Priors are indexed by the engine's canonical legal \
             order, so a length mismatch means the policy head and the position disagree about \
             the action set",
            self.priors.len(),
        );
        assert!(
            self.value.is_finite() && (-1.0..=1.0).contains(&self.value),
            "{leaf:?}: value {} is outside [-1, 1]; it is the expected outcome from the \
             evaluated position's side to move",
            self.value,
        );
        for (i, &p) in self.priors.iter().enumerate() {
            assert!(
                p.is_finite() && p >= 0.0,
                "{leaf:?}: prior {i} is {p}; priors are probabilities over the canonical legal \
                 order",
            );
        }
    }
}

/// Package-owned: turns a position into bytes.
///
/// Encoding runs inside the leaf callback while the transient position is
/// valid. Only the encoded bytes may be queued after that callback returns.
///
/// Encoders are shared by reference; scratch space belongs in `out`.
pub trait Encoder: Send {
    /// Append the encoding of `position` to `out`.
    ///
    /// `out` is a shared arena holding the items already in the batch. An
    /// implementation appends and never clears, truncates, or reorders it.
    fn encode(&self, position: &Position, out: &mut Vec<u8>);
}

/// Package-owned: answers one whole batch in one call.
///
/// Implementations may own mutable runtime, interpreter, or device state.
pub trait Evaluator: Send {
    /// Append one [`Evaluation`] per item of `batch`, in batch order.
    ///
    /// Implementations append and never clear `out`. They must append exactly
    /// `batch.len()` answers.
    fn evaluate(&mut self, batch: &EncodedBatch, out: &mut Vec<Evaluation>);
}

/// A reusable arena of encoded items: one byte buffer plus the offsets that cut
/// it into items, so assembling a batch costs no per-item allocation.
///
/// Item lengths may differ.
#[derive(Clone, Debug)]
pub struct EncodedBatch {
    data: Vec<u8>,
    /// `len() + 1` entries, starting at zero: item `i` is
    /// `data[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<usize>,
}

impl Default for EncodedBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodedBatch {
    /// An empty batch that has allocated nothing.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
        }
    }

    /// An empty batch reserved for `items` items totalling `bytes` bytes.
    #[must_use]
    pub fn with_capacity(items: usize, bytes: usize) -> Self {
        let mut offsets = Vec::with_capacity(items + 1);
        offsets.push(0);
        Self {
            data: Vec::with_capacity(bytes),
            offsets,
        }
    }

    /// Encode `position` with `encoder` and append it as the next item,
    /// returning its index.
    ///
    /// # Panics
    ///
    /// If the encoder shrinks the arena containing prior items.
    pub fn push_with<E: Encoder + ?Sized>(&mut self, encoder: &E, position: &Position) -> usize {
        let start = self.data.len();
        encoder.encode(position, &mut self.data);
        assert!(
            self.data.len() >= start,
            "the encoder shrank the batch arena from {start} to {} bytes; `out` holds every item \
             already in the batch and must only be appended to",
            self.data.len(),
        );
        self.offsets.push(self.data.len());
        self.offsets.len() - 2
    }

    /// Append one already-encoded item verbatim, returning its index.
    ///
    /// This supports merging worker-encoded items after their source positions
    /// are no longer available.
    pub fn push_bytes(&mut self, item: &[u8]) -> usize {
        self.data.extend_from_slice(item);
        self.offsets.push(self.data.len());
        self.offsets.len() - 2
    }

    /// How many items the batch holds.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Whether the batch holds no items.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bytes of one item.
    ///
    /// # Panics
    ///
    /// If `index` is not an item of this batch.
    #[inline]
    #[must_use]
    pub fn item(&self, index: usize) -> &[u8] {
        assert!(
            index < self.len(),
            "item {index} of a batch holding {} items",
            self.len(),
        );
        &self.data[self.offsets[index]..self.offsets[index + 1]]
    }

    /// Every item, in batch order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u8]> {
        self.offsets.windows(2).map(|w| &self.data[w[0]..w[1]])
    }

    /// The whole arena as one contiguous slice.
    ///
    /// With [`EncodedBatch::offsets`] this is the `values + offsets` pair a
    /// ragged tensor is built from, without walking the items.
    #[inline]
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    /// The item boundaries: `len() + 1` entries starting at zero, where item `i`
    /// spans `offsets()[i]..offsets()[i + 1]`.
    #[inline]
    #[must_use]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Drop every item, keeping the allocation for the next batch.
    pub fn clear(&mut self) {
        self.data.clear();
        self.offsets.clear();
        self.offsets.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_engine::{Action, HexCoord};

    /// Writes the legal count and then one `(q, r)` pair per legal action.
    struct Ragged;

    impl Encoder for Ragged {
        fn encode(&self, position: &Position, out: &mut Vec<u8>) {
            out.extend_from_slice(&(position.legal_count() as u32).to_le_bytes());
            for action in position.legal_actions() {
                out.extend_from_slice(&action.coord().q.to_le_bytes());
                out.extend_from_slice(&action.coord().r.to_le_bytes());
            }
        }
    }

    fn opened() -> Position {
        let mut p = Position::new();
        p.advance(Action::new(HexCoord::ORIGIN)).expect("opening");
        p
    }

    #[test]
    fn a_fresh_batch_is_empty() {
        let batch = EncodedBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.offsets(), &[0]);
        assert!(batch.bytes().is_empty());
        assert_eq!(batch.iter().count(), 0);
    }

    #[test]
    fn items_of_different_lengths_stay_separable() {
        let empty = Position::new();
        let opened = opened();
        let mut batch = EncodedBatch::new();
        assert_eq!(batch.push_with(&Ragged, &empty), 0);
        assert_eq!(batch.push_with(&Ragged, &opened), 1);

        assert_eq!(batch.len(), 2);
        assert_eq!(batch.item(0).len(), 4 + 4);
        assert_eq!(batch.item(1).len(), 4 + 4 * 216);
        assert_ne!(batch.item(0).len(), batch.item(1).len());
        assert_eq!(
            batch.bytes().len(),
            batch.item(0).len() + batch.item(1).len()
        );
        assert_eq!(
            batch.iter().collect::<Vec<_>>(),
            vec![batch.item(0), batch.item(1)]
        );
    }

    #[test]
    fn clearing_a_batch_keeps_its_allocation() {
        let mut batch = EncodedBatch::new();
        batch.push_with(&Ragged, &opened());
        let capacity = batch.data.capacity();
        assert!(capacity > 0);

        batch.clear();
        assert!(batch.is_empty());
        assert_eq!(batch.offsets(), &[0]);
        assert_eq!(batch.data.capacity(), capacity);
    }

    #[test]
    #[should_panic(expected = "item 2 of a batch holding 1 items")]
    fn reading_past_the_last_item_panics() {
        let mut batch = EncodedBatch::new();
        batch.push_with(&Ragged, &opened());
        let _ = batch.item(2);
    }

    #[test]
    #[should_panic(expected = "shrank the batch arena")]
    fn an_encoder_that_clears_the_arena_panics() {
        struct Vandal;
        impl Encoder for Vandal {
            fn encode(&self, _position: &Position, out: &mut Vec<u8>) {
                out.clear();
            }
        }
        let mut batch = EncodedBatch::new();
        batch.push_with(&Ragged, &opened());
        batch.push_with(&Vandal, &opened());
    }
}
