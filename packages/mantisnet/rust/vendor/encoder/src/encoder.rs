//! Batched MantisNet graph encoding and collation.
//!
//! Orderings match the Python builder (`mantisnet.builder`): windows sort by
//! `(q*2^21 + r)*4 + axis`; incidence is window-major then slot; the decoder
//! table is legal-cell-major, then axis, then offset. Parity is checked by
//! `python/mantisnet/tests/test_rust_builder.py`.

use hexo_engine as engine;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::OnceLock;

const QSHIFT: i64 = 1 << 21;
const WIRE_MAGIC: &[u8; 8] = b"MANTIS\x00\x01";
const WIRE_HEADER_LEN: usize = WIRE_MAGIC.len() + 4 + 4 + 7 * 4;
const MAX_ORBIT_RADIUS: i16 = 12;
const ORBIT48_CLASSES: usize = 48;
const CELL_NEAREST_UNREACHED: i64 = engine::LEGAL_RADIUS as i64 + 1;

fn displacement_distance(q: i16, r: i16) -> usize {
    let (q, r) = (i32::from(q), i32::from(r));
    q.abs().max(r.abs()).max((q + r).abs()) as usize
}

fn rotate((q, r): (i16, i16)) -> (i16, i16) {
    (-r, q + r)
}

fn reflect((q, r): (i16, i16)) -> (i16, i16) {
    (r, q)
}

fn canonical_displacement(displacement: (i16, i16)) -> (i16, i16) {
    let mut best = displacement;
    for reflected in [false, true] {
        let mut image = if reflected {
            reflect(displacement)
        } else {
            displacement
        };
        for _ in 0..6 {
            best = best.min(image);
            image = rotate(image);
        }
    }
    best
}

struct OrbitTable {
    grid: Vec<i8>,
}

fn generate_orbit_table() -> OrbitTable {
    let mut representatives = BTreeSet::new();
    for dq in -MAX_ORBIT_RADIUS..=MAX_ORBIT_RADIUS {
        for dr in -MAX_ORBIT_RADIUS..=MAX_ORBIT_RADIUS {
            let distance = displacement_distance(dq, dr);
            if (1..=MAX_ORBIT_RADIUS as usize).contains(&distance) {
                let (cq, cr) = canonical_displacement((dq, dr));
                representatives.insert((distance, cq, cr));
            }
        }
    }
    assert_eq!(representatives.len(), ORBIT48_CLASSES);
    let rank: FxHashMap<_, _> = representatives
        .into_iter()
        .enumerate()
        .map(|(id, (_distance, q, r))| ((q, r), id as i8))
        .collect();
    let side = usize::from((2 * MAX_ORBIT_RADIUS + 1) as u16);
    let mut grid = vec![-1i8; side * side];
    for dq in -MAX_ORBIT_RADIUS..=MAX_ORBIT_RADIUS {
        for dr in -MAX_ORBIT_RADIUS..=MAX_ORBIT_RADIUS {
            let distance = displacement_distance(dq, dr);
            if (1..=MAX_ORBIT_RADIUS as usize).contains(&distance) {
                let row = usize::from((dq + MAX_ORBIT_RADIUS) as u16);
                let col = usize::from((dr + MAX_ORBIT_RADIUS) as u16);
                grid[row * side + col] = rank[&canonical_displacement((dq, dr))];
            }
        }
    }
    OrbitTable { grid }
}

fn orbit_id(dq: i16, dr: i16, radius: usize) -> Result<i64, String> {
    if radius == 0 || radius > MAX_ORBIT_RADIUS as usize {
        return Err(format!(
            "orbit radius must lie in 1..={}, got {radius}",
            MAX_ORBIT_RADIUS
        ));
    }
    let distance = displacement_distance(dq, dr);
    if !(1..=radius).contains(&distance) {
        return Err(format!(
            "displacement ({dq}, {dr}) has distance {distance}, outside 1..={radius}"
        ));
    }
    static TABLE: OnceLock<OrbitTable> = OnceLock::new();
    let side = usize::from((2 * MAX_ORBIT_RADIUS + 1) as u16);
    let row = usize::from((dq + MAX_ORBIT_RADIUS) as u16);
    let col = usize::from((dr + MAX_ORBIT_RADIUS) as u16);
    Ok(i64::from(
        TABLE.get_or_init(generate_orbit_table).grid[row * side + col],
    ))
}

fn structural_axis(dq: i16, dr: i16) -> Option<i64> {
    if dr == 0 {
        Some(0)
    } else if dq == 0 {
        Some(1)
    } else if dq + dr == 0 {
        Some(2)
    } else {
        None
    }
}

/// Number of reversal-canonical nonempty ternary window patterns.
///
/// A slot is empty (0), own (1), or opponent
/// (2): a window is a base-3 pattern over its six slots, digit at `3^k` for
/// slot `k`, mover-relative. 729 patterns fold to 378 orbits under digit
/// reversal (27 palindromes); the empty pattern is unreachable, leaving 377.
pub const TERN_PATTERNS: i64 = 377;

/// Number of ternary joint decoder classes: empty slots of nonempty patterns.
pub const TERN_DEC_CLASSES: i64 = 726;

/// Number of ternary joint incidence classes: occupied slots. With the
/// decoder classes these are the 2184 nonempty-pattern orbits of the joint
/// involution `(pattern, slot) -> (reverse3(pattern), 5 - slot)`; including
/// the empty pattern's three orbits the involution has 2187, asserted in the
/// table constructor.
pub const TERN_OCC_CLASSES: i64 = 1458;

/// Reverse the base-3 digit string of a ternary pattern.
const fn reverse3(p: usize) -> usize {
    let mut rev = 0usize;
    let mut rem = p;
    let mut k = 0;
    while k < 6 {
        rev = rev * 3 + rem % 3;
        rem /= 3;
        k += 1;
    }
    rev
}

/// Rank of each canonical nonempty ternary pattern (0..377); propagated to
/// noncanonical patterns through their reversal; `-1` for the empty pattern.
const TERN_RANK: [i16; 729] = {
    let mut rank = [-1i16; 729];
    let mut next = 0i16;
    let mut orbits = 0i64;
    let mut p = 0usize;
    while p < 729 {
        if reverse3(p) >= p {
            orbits += 1;
            if p > 0 {
                rank[p] = next;
                next += 1;
            }
        }
        p += 1;
    }
    assert!(orbits == 378 && next as i64 == TERN_PATTERNS);
    let mut p = 1usize;
    while p < 729 {
        let rev = reverse3(p);
        if rev < p {
            rank[p] = rank[rev];
        }
        p += 1;
    }
    rank
};

/// The two ternary joint `(pattern, slot)` class tables, indexed
/// `pattern * 6 + slot`.
///
/// One enumeration of the joint involution in ascending `(pattern, slot)`
/// order — 2187 orbits, asserted — re-ranked over each restriction: empty
/// slots of nonempty patterns (`occupied = false`, the decoder table) or
/// occupied slots (`true`, the incidence table). Entries outside the
/// restriction are `-1`.
const fn tern_orbit_table(occupied: bool) -> [i16; 729 * 6] {
    // The shared enumeration: joint orbit ids over every (pattern, slot).
    let mut joint = [-1i32; 729 * 6];
    let mut next = 0i32;
    let mut p = 0usize;
    while p < 729 {
        let rev = reverse3(p);
        let mut s = 0usize;
        while s < 6 {
            if p < rev || (p == rev && s <= 5 - s) {
                joint[p * 6 + s] = next;
                next += 1;
            } else {
                joint[p * 6 + s] = joint[rev * 6 + (5 - s)];
            }
            s += 1;
        }
        p += 1;
    }
    assert!(next == 2187);

    // Re-rank the restriction: each selected orbit's rank is the count of
    // selected orbits with a smaller joint id — an ascending relabel.
    let want = occupied;
    let mut selected = [false; 2187];
    let mut p = 1usize;
    while p < 729 {
        let mut s = 0usize;
        let mut rem = p;
        while s < 6 {
            if rem.is_multiple_of(3) != want {
                selected[joint[p * 6 + s] as usize] = true;
            }
            rem /= 3;
            s += 1;
        }
        p += 1;
    }
    let mut rank_of = [-1i32; 2187];
    let mut count = 0i32;
    let mut orbit = 0usize;
    while orbit < 2187 {
        if selected[orbit] {
            rank_of[orbit] = count;
            count += 1;
        }
        orbit += 1;
    }
    assert!(
        count as i64
            == if want {
                TERN_OCC_CLASSES
            } else {
                TERN_DEC_CLASSES
            }
    );

    let mut table = [-1i16; 729 * 6];
    let mut p = 1usize;
    while p < 729 {
        let mut s = 0usize;
        let mut rem = p;
        while s < 6 {
            if rem.is_multiple_of(3) != want {
                table[p * 6 + s] = rank_of[joint[p * 6 + s] as usize] as i16;
            }
            rem /= 3;
            s += 1;
        }
        p += 1;
    }
    table
}

/// Ternary decoder class of each `(pattern, empty candidate slot)` pair.
const TERN_DEC_CLASS: [i16; 729 * 6] = tern_orbit_table(false);

/// Ternary incidence class of each `(pattern, occupied slot)` pair.
const TERN_OCC_CLASS: [i16; 729 * 6] = tern_orbit_table(true);

/// Powers of three addressing slot digits of a ternary pattern.
const POW3: [u16; 6] = [1, 3, 9, 27, 81, 243];

/// Number of reversal orbits of ternary `(post-placement pattern, own slot)`
/// pairs used by action rows.
pub const TERN_POST1_CLASSES: i64 = 729;

/// Ternary post-placement class of each `(pattern, own slot)` pair.
///
/// Entries are orbit ranks under `(p, s) -> (reverse3(p), 5 - s)`, assigned
/// in ascending `(p, s)` order; slots whose digit is not own are `-1`.
const TERN_POST1_CLASS: [i16; 729 * 6] = {
    let mut table = [-1i16; 729 * 6];
    let mut next = 0i16;
    let mut p = 0usize;
    while p < 729 {
        let rev = reverse3(p);
        let mut s = 0usize;
        while s < 6 {
            if (p / POW3[s] as usize) % 3 == 1 {
                if p < rev || (p == rev && s <= 5 - s) {
                    table[p * 6 + s] = next;
                    next += 1;
                } else {
                    table[p * 6 + s] = table[rev * 6 + (5 - s)];
                }
            }
            s += 1;
        }
        p += 1;
    }
    assert!(next as i64 == TERN_POST1_CLASSES);
    table
};

const ACTION_OWN: i64 = 0;
const ACTION_OPP: i64 = 1;
const ACTION_EMPTY: i64 = 2;
const ACTION_MIXED: i64 = 3;
const ACTION_EMPTY_ORBITS: usize = 3;

fn pack(c: engine::HexCoord) -> i64 {
    c.q as i64 * QSHIFT + c.r as i64
}

/// One position's graph, indices local to the position.
#[derive(Debug, PartialEq, Eq)]
pub struct Graph {
    stone_own: Vec<i64>,
    stone_qr: Vec<[i32; 2]>,
    window_feat: Vec<i64>,
    window_id: Vec<i64>,
    inc_stone: Vec<i64>,
    inc_window: Vec<i64>,
    inc_class: Vec<i64>,
    n_legal: usize,
    cell_qr: Vec<[i32; 2]>,
    cell_occupancy: Vec<i64>,
    cell_is_legal: Vec<i64>,
    cell_nearest: Vec<i64>,
    radius_src: Vec<i64>,
    radius_dst: Vec<i64>,
    radius_orbit: Vec<i64>,
    radius_own: Vec<i64>,
    radius_on_axis: Vec<i64>,
    adjacency_src: Vec<i64>,
    adjacency_dst: Vec<i64>,
    adjacency_axis: Vec<i64>,
    dec_cell: Vec<i64>,
    dec_window: Vec<i64>,
    dec_class: Vec<i64>,
    moves_remaining: u8,
    action_window_index: Vec<i64>,
    action_post1_class: Vec<i64>,
    action_pre_status: Vec<i64>,
}

#[derive(Default)]
struct ActionTables {
    window_index: Vec<i64>,
    post1_class: Vec<i64>,
    pre_status: Vec<i64>,
}

/// Error from decoding a MantisNet wire-format position item.
///
/// Includes the batch item index when produced by [`decode_batch`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WireError {
    item: Option<usize>,
    detail: String,
}

impl WireError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            item: None,
            detail: detail.into(),
        }
    }

    fn at_item(mut self, item: usize) -> Self {
        self.item = Some(item);
        self
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(item) = self.item {
            write!(f, "encoded MantisNet item {item}: {}", self.detail)
        } else {
            write!(f, "encoded MantisNet item: {}", self.detail)
        }
    }
}

impl std::error::Error for WireError {}

#[derive(Clone, Copy)]
struct WireCounts {
    stones: usize,
    windows: usize,
    incidences: usize,
    legal: usize,
    decoder: usize,
    radius: usize,
    adjacency: usize,
}

impl WireCounts {
    fn from_graph(graph: &Graph) -> Self {
        Self {
            stones: graph.stone_own.len(),
            windows: graph.window_feat.len(),
            incidences: graph.inc_stone.len(),
            legal: graph.n_legal,
            decoder: graph.dec_cell.len(),
            radius: graph.radius_src.len(),
            adjacency: graph.adjacency_src.len(),
        }
    }

    fn payload_len(self) -> Option<usize> {
        // Per-stone: 8 (own) + 8 (qr). Per-window: 32 (feat + id triple).
        // Per-incidence and decoder edge: 24 each. Cell features use 24 bytes,
        // radius edges 40, and adjacency edges 24. Each legal cell has 18
        // action rows carrying three i64 values.
        self.stones
            .checked_mul(16)?
            .checked_add(self.windows.checked_mul(32)?)?
            .checked_add(self.incidences.checked_mul(24)?)?
            .checked_add(self.decoder.checked_mul(24)?)?
            .checked_add(self.legal.checked_mul(24)?)?
            .checked_add(self.radius.checked_mul(40)?)?
            .checked_add(self.adjacency.checked_mul(24)?)?
            .checked_add(
                self.legal
                    .checked_mul(engine::WINDOWS_PER_PLACEMENT)?
                    .checked_mul(24)?,
            )
    }
}

struct WireReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> WireReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn take(&mut self, len: usize, field: &'static str) -> Result<&'a [u8], WireError> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or_else(|| WireError::new(format!("{field} length overflows usize")))?;
        let value = self.bytes.get(self.cursor..end).ok_or_else(|| {
            WireError::new(format!(
                "truncated {field}: need {len} bytes, have {}",
                self.bytes.len().saturating_sub(self.cursor)
            ))
        })?;
        self.cursor = end;
        Ok(value)
    }

    fn u8(&mut self, field: &'static str) -> Result<u8, WireError> {
        Ok(self.take(1, field)?[0])
    }

    fn u32(&mut self, field: &'static str) -> Result<u32, WireError> {
        let bytes: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .expect("the reader returned the requested width");
        Ok(u32::from_le_bytes(bytes))
    }

    fn i32(&mut self, field: &'static str) -> Result<i32, WireError> {
        let bytes: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .expect("the reader returned the requested width");
        Ok(i32::from_le_bytes(bytes))
    }

    fn i64(&mut self, field: &'static str) -> Result<i64, WireError> {
        let bytes: [u8; 8] = self
            .take(8, field)?
            .try_into()
            .expect("the reader returned the requested width");
        Ok(i64::from_le_bytes(bytes))
    }
}

fn count_u32(count: usize, field: &'static str) -> u32 {
    u32::try_from(count)
        .unwrap_or_else(|_| panic!("MantisNet {field} count {count} exceeds the wire format"))
}

fn append_i32s(out: &mut Vec<u8>, values: impl IntoIterator<Item = i32>) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn append_i64s(out: &mut Vec<u8>, values: &[i64]) {
    for &value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

/// Append one live position in the versioned worker-to-batcher wire format.
///
/// The bytes already in `out` are left untouched. The appended item is:
///
/// ```text
/// magic[8], MODEL_REPR_VERSION:u32,
/// moves_remaining:u8, reserved_zero[3],
/// stones:u32, windows:u32, incidences:u32, legal:u32, decoder:u32,
/// radius_edges:u32, adjacency_edges:u32,
/// stone_own[stones]:i64,
/// stone_qr[stones][2]:i32,
/// window_feat[windows]:i64,
/// window_id[windows][3]:i64,
/// inc_stone[incidences]:i64, inc_window[incidences]:i64,
/// inc_class[incidences]:i64,
/// dec_cell[decoder]:i64, dec_window[decoder]:i64,
/// dec_class[decoder]:i64,
/// cell_occupancy[legal]:i64, cell_is_legal[legal]:i64,
/// cell_nearest[legal]:i64,
/// radius_src[radius]:i64, radius_dst[radius]:i64, radius_orbit[radius]:i64,
/// radius_own[radius]:i64, radius_on_axis[radius]:i64,
/// adjacency_src[adjacency]:i64, adjacency_dst[adjacency]:i64,
/// adjacency_axis[adjacency]:i64,
/// action_window_index[legal][18]:i64,
/// action_post1_class[legal][18]:i64,
/// action_pre_status[legal][18]:i64
/// ```
///
/// Every integer is little-endian. A terminal position is a caller protocol
/// violation: the engine, not the network, owns terminal outcomes.
pub fn encode_position(position: &engine::Position, out: &mut Vec<u8>) {
    let graph =
        build(position).unwrap_or_else(|why| panic!("MantisNet encoder refuses position: {why}"));
    let counts = WireCounts::from_graph(&graph);
    let encoded_counts = [
        count_u32(counts.stones, "stone"),
        count_u32(counts.windows, "window"),
        count_u32(counts.incidences, "incidence"),
        count_u32(counts.legal, "legal-cell"),
        count_u32(counts.decoder, "decoder"),
        count_u32(counts.radius, "radius edge"),
        count_u32(counts.adjacency, "adjacency edge"),
    ];
    let payload_len = counts
        .payload_len()
        .expect("a buildable MantisNet graph has a representable wire length");
    out.reserve(
        WIRE_HEADER_LEN
            .checked_add(payload_len)
            .expect("a buildable MantisNet graph has a representable wire length"),
    );

    out.extend_from_slice(WIRE_MAGIC);
    out.extend_from_slice(&crate::MODEL_REPR_VERSION.to_le_bytes());
    out.push(graph.moves_remaining);
    out.extend_from_slice(&[0; 3]);
    for count in encoded_counts {
        out.extend_from_slice(&count.to_le_bytes());
    }
    append_i64s(out, &graph.stone_own);
    append_i32s(out, graph.stone_qr.iter().flatten().copied());
    append_i64s(out, &graph.window_feat);
    append_i64s(out, &graph.window_id);
    append_i64s(out, &graph.inc_stone);
    append_i64s(out, &graph.inc_window);
    append_i64s(out, &graph.inc_class);
    append_i64s(out, &graph.dec_cell);
    append_i64s(out, &graph.dec_window);
    append_i64s(out, &graph.dec_class);
    append_i64s(out, &graph.cell_occupancy);
    append_i64s(out, &graph.cell_is_legal);
    append_i64s(out, &graph.cell_nearest);
    append_i64s(out, &graph.radius_src);
    append_i64s(out, &graph.radius_dst);
    append_i64s(out, &graph.radius_orbit);
    append_i64s(out, &graph.radius_own);
    append_i64s(out, &graph.radius_on_axis);
    append_i64s(out, &graph.adjacency_src);
    append_i64s(out, &graph.adjacency_dst);
    append_i64s(out, &graph.adjacency_axis);
    append_i64s(out, &graph.action_window_index);
    append_i64s(out, &graph.action_post1_class);
    append_i64s(out, &graph.action_pre_status);
}

fn read_count(
    reader: &mut WireReader<'_>,
    item_len: usize,
    field: &'static str,
) -> Result<usize, WireError> {
    let count = reader.u32(field)? as usize;
    // Cap every count at the item length to prevent oversized allocations.
    if count > item_len {
        return Err(WireError::new(format!(
            "{field} count {count} exceeds item length {item_len}"
        )));
    }
    Ok(count)
}

fn read_i64_vec(
    reader: &mut WireReader<'_>,
    count: usize,
    field: &'static str,
) -> Result<Vec<i64>, WireError> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.i64(field)?);
    }
    Ok(values)
}

fn invalid_feature(field: &'static str, index: usize, value: i64) -> WireError {
    WireError::new(format!("{field}[{index}] has invalid feature {value}"))
}

fn validate_indices(values: &[i64], upper: usize, field: &'static str) -> Result<(), WireError> {
    let upper_i64 = i64::try_from(upper)
        .map_err(|_| WireError::new(format!("{field} upper bound exceeds i64")))?;
    for (index, &value) in values.iter().enumerate() {
        if value < 0 || value >= upper_i64 {
            return Err(WireError::new(format!(
                "{field}[{index}] index {value} is outside 0..{upper}"
            )));
        }
    }
    Ok(())
}

fn decode_graph(bytes: &[u8]) -> Result<Graph, WireError> {
    let mut reader = WireReader::new(bytes);
    if reader.take(WIRE_MAGIC.len(), "magic")? != WIRE_MAGIC {
        return Err(WireError::new("wrong magic"));
    }
    let version = reader.u32("MODEL_REPR_VERSION")?;
    if version != crate::MODEL_REPR_VERSION {
        return Err(WireError::new(format!(
            "MODEL_REPR_VERSION {version} does not match {}",
            crate::MODEL_REPR_VERSION
        )));
    }
    let moves_remaining = reader.u8("moves_remaining")?;
    if !matches!(moves_remaining, 1 | 2) {
        return Err(WireError::new(format!(
            "moves_remaining must be 1 or 2, got {moves_remaining}"
        )));
    }
    if reader.take(3, "reserved bytes")? != [0; 3] {
        return Err(WireError::new("reserved bytes are nonzero"));
    }

    let counts = WireCounts {
        stones: read_count(&mut reader, bytes.len(), "stones")?,
        windows: read_count(&mut reader, bytes.len(), "windows")?,
        incidences: read_count(&mut reader, bytes.len(), "incidences")?,
        legal: read_count(&mut reader, bytes.len(), "legal cells")?,
        decoder: read_count(&mut reader, bytes.len(), "decoder incidences")?,
        radius: read_count(&mut reader, bytes.len(), "radius edges")?,
        adjacency: read_count(&mut reader, bytes.len(), "adjacency edges")?,
    };
    if counts.legal == 0 {
        return Err(WireError::new("a live position must have a legal cell"));
    }
    let expected_len = WIRE_HEADER_LEN
        .checked_add(
            counts
                .payload_len()
                .ok_or_else(|| WireError::new("payload length overflows usize"))?,
        )
        .ok_or_else(|| WireError::new("item length overflows usize"))?;
    match bytes.len().cmp(&expected_len) {
        std::cmp::Ordering::Less => {
            return Err(WireError::new(format!(
                "truncated payload: header describes {expected_len} bytes, got {}",
                bytes.len()
            )));
        }
        std::cmp::Ordering::Greater => {
            return Err(WireError::new(format!(
                "trailing bytes: header describes {expected_len} bytes, got {}",
                bytes.len()
            )));
        }
        std::cmp::Ordering::Equal => {}
    }

    let stone_own = read_i64_vec(&mut reader, counts.stones, "stone_own")?;
    for (index, &value) in stone_own.iter().enumerate() {
        if !matches!(value, 0 | 1) {
            return Err(invalid_feature("stone_own", index, value));
        }
    }

    let mut stone_qr = Vec::with_capacity(counts.stones);
    for index in 0..counts.stones {
        let q = reader.i32("stone_qr.q")?;
        let r = reader.i32("stone_qr.r")?;
        let q16 = i16::try_from(q)
            .map_err(|_| WireError::new(format!("stone_qr[{index}].q is out of range: {q}")))?;
        let r16 = i16::try_from(r)
            .map_err(|_| WireError::new(format!("stone_qr[{index}].r is out of range: {r}")))?;
        if !engine::HexCoord::new(q16, r16).is_valid() {
            return Err(WireError::new(format!(
                "stone_qr[{index}] is not a valid engine coordinate: ({q}, {r})"
            )));
        }
        stone_qr.push([q, r]);
    }

    let window_feat = read_i64_vec(&mut reader, counts.windows, "window_feat")?;
    for (index, &value) in window_feat.iter().enumerate() {
        if !(0..TERN_PATTERNS).contains(&value) {
            return Err(invalid_feature("window_feat", index, value));
        }
    }

    let window_id = read_i64_vec(
        &mut reader,
        counts
            .windows
            .checked_mul(3)
            .ok_or_else(|| WireError::new("window_id length overflows usize".to_string()))?,
        "window_id",
    )?;
    for (index, chunk) in window_id.chunks_exact(3).enumerate() {
        if !(0..3).contains(&chunk[0]) {
            return Err(invalid_feature("window_id", index, chunk[0]));
        }
    }

    let inc_stone = read_i64_vec(&mut reader, counts.incidences, "inc_stone")?;
    let inc_window = read_i64_vec(&mut reader, counts.incidences, "inc_window")?;
    let inc_class = read_i64_vec(&mut reader, counts.incidences, "inc_class")?;
    validate_indices(&inc_stone, counts.stones, "inc_stone")?;
    validate_indices(&inc_window, counts.windows, "inc_window")?;
    for (index, &value) in inc_class.iter().enumerate() {
        if !(0..TERN_OCC_CLASSES).contains(&value) {
            return Err(invalid_feature("inc_class", index, value));
        }
    }

    let dec_cell = read_i64_vec(&mut reader, counts.decoder, "dec_cell")?;
    let dec_window = read_i64_vec(&mut reader, counts.decoder, "dec_window")?;
    let dec_class = read_i64_vec(&mut reader, counts.decoder, "dec_class")?;
    validate_indices(&dec_cell, counts.legal, "dec_cell")?;
    validate_indices(&dec_window, counts.windows, "dec_window")?;
    for (index, &value) in dec_class.iter().enumerate() {
        if !(0..TERN_DEC_CLASSES).contains(&value) {
            return Err(invalid_feature("dec_class", index, value));
        }
    }

    let cell_occupancy = read_i64_vec(&mut reader, counts.legal, "cell_occupancy")?;
    let cell_is_legal = read_i64_vec(&mut reader, counts.legal, "cell_is_legal")?;
    let cell_nearest = read_i64_vec(&mut reader, counts.legal, "cell_nearest")?;
    for (index, &value) in cell_occupancy.iter().enumerate() {
        if !(0..3).contains(&value) {
            return Err(invalid_feature("cell_occupancy", index, value));
        }
    }
    for (index, &value) in cell_is_legal.iter().enumerate() {
        if !matches!(value, 0 | 1) {
            return Err(invalid_feature("cell_is_legal", index, value));
        }
    }
    for (index, &value) in cell_nearest.iter().enumerate() {
        if !(0..=CELL_NEAREST_UNREACHED).contains(&value) {
            return Err(invalid_feature("cell_nearest", index, value));
        }
    }

    let radius_src = read_i64_vec(&mut reader, counts.radius, "radius_src")?;
    let radius_dst = read_i64_vec(&mut reader, counts.radius, "radius_dst")?;
    let radius_orbit = read_i64_vec(&mut reader, counts.radius, "radius_orbit")?;
    let radius_own = read_i64_vec(&mut reader, counts.radius, "radius_own")?;
    let radius_on_axis = read_i64_vec(&mut reader, counts.radius, "radius_on_axis")?;
    validate_indices(&radius_src, counts.stones, "radius_src")?;
    validate_indices(&radius_dst, counts.legal, "radius_dst")?;
    for (index, &value) in radius_orbit.iter().enumerate() {
        if !(0..ORBIT48_CLASSES as i64).contains(&value) {
            return Err(invalid_feature("radius_orbit", index, value));
        }
    }
    for (field, values) in [
        ("radius_own", radius_own.as_slice()),
        ("radius_on_axis", radius_on_axis.as_slice()),
    ] {
        for (index, &value) in values.iter().enumerate() {
            if !matches!(value, 0 | 1) {
                return Err(invalid_feature(field, index, value));
            }
        }
    }

    let adjacency_src = read_i64_vec(&mut reader, counts.adjacency, "adjacency_src")?;
    let adjacency_dst = read_i64_vec(&mut reader, counts.adjacency, "adjacency_dst")?;
    let adjacency_axis = read_i64_vec(&mut reader, counts.adjacency, "adjacency_axis")?;
    validate_indices(&adjacency_src, counts.legal, "adjacency_src")?;
    validate_indices(&adjacency_dst, counts.legal, "adjacency_dst")?;
    for (index, &value) in adjacency_axis.iter().enumerate() {
        if !(0..3).contains(&value) {
            return Err(invalid_feature("adjacency_axis", index, value));
        }
    }

    let action_count = counts
        .legal
        .checked_mul(engine::WINDOWS_PER_PLACEMENT)
        .ok_or_else(|| WireError::new("action row count overflows usize"))?;
    let action_window_index = read_i64_vec(&mut reader, action_count, "action_window_index")?;
    let action_post1_class = read_i64_vec(&mut reader, action_count, "action_post1_class")?;
    let action_pre_status = read_i64_vec(&mut reader, action_count, "action_pre_status")?;
    for (index, &class) in action_post1_class.iter().enumerate() {
        if !(0..TERN_POST1_CLASSES).contains(&class) {
            return Err(invalid_feature("action_post1_class", index, class));
        }
    }
    let mut edge = 0usize;
    for (flat, (&window, &status)) in action_window_index
        .iter()
        .zip(&action_pre_status)
        .enumerate()
    {
        if !matches!(
            status,
            ACTION_OWN | ACTION_OPP | ACTION_EMPTY | ACTION_MIXED
        ) {
            return Err(invalid_feature("action_pre_status", flat, status));
        }
        let kept = status != ACTION_EMPTY;
        if kept != (window >= 0) || window >= counts.windows as i64 {
            return Err(WireError::new(format!(
                "action_window_index[{flat}]={window} disagrees with status {status}"
            )));
        }
        if kept {
            if edge >= dec_cell.len()
                || dec_cell[edge] as usize != flat / engine::WINDOWS_PER_PLACEMENT
                || dec_window[edge] != window
            {
                return Err(WireError::new(format!(
                    "action row {flat} disagrees with decoder incidence {edge}"
                )));
            }
            edge += 1;
        }
    }
    if edge != dec_cell.len() {
        return Err(WireError::new(format!(
            "{edge} kept action rows do not match {} decoder incidences",
            dec_cell.len()
        )));
    }

    debug_assert_eq!(reader.cursor, bytes.len());
    Ok(Graph {
        stone_own,
        stone_qr,
        window_feat,
        window_id,
        inc_stone,
        inc_window,
        inc_class,
        n_legal: counts.legal,
        cell_qr: vec![],
        cell_occupancy,
        cell_is_legal,
        cell_nearest,
        radius_src,
        radius_dst,
        radius_orbit,
        radius_own,
        radius_on_axis,
        adjacency_src,
        adjacency_dst,
        adjacency_axis,
        dec_cell,
        dec_window,
        dec_class,
        moves_remaining,
        action_window_index,
        action_post1_class,
        action_pre_status,
    })
}

#[allow(clippy::type_complexity)]
fn cell_node_fields(
    stones: &[(engine::HexCoord, engine::Player)],
    stone_own: &[i64],
    legal: &[engine::HexCoord],
) -> Result<
    (
        Vec<[i32; 2]>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
    ),
    String,
> {
    let cell_qr = legal
        .iter()
        .map(|cell| [i32::from(cell.q), i32::from(cell.r)])
        .collect();
    let cell_occupancy = vec![0; legal.len()];
    let cell_is_legal = vec![1; legal.len()];
    let mut cell_nearest = vec![CELL_NEAREST_UNREACHED; legal.len()];
    let mut radius_src = Vec::new();
    let mut radius_dst = Vec::new();
    let mut radius_orbit = Vec::new();
    let mut radius_own = Vec::new();
    let mut radius_on_axis = Vec::new();

    for (destination, cell) in legal.iter().enumerate() {
        for (source, &(stone, _)) in stones.iter().enumerate() {
            let dq = cell.q - stone.q;
            let dr = cell.r - stone.r;
            let distance = displacement_distance(dq, dr);
            cell_nearest[destination] = cell_nearest[destination].min(distance as i64);
            if distance <= engine::LEGAL_RADIUS as usize {
                if distance == 0 {
                    return Err(format!(
                        "legal cell ({}, {}) is occupied by stone {source}",
                        cell.q, cell.r
                    ));
                }
                radius_src.push(source as i64);
                radius_dst.push(destination as i64);
                radius_orbit.push(orbit_id(dq, dr, MAX_ORBIT_RADIUS as usize)?);
                radius_own.push(stone_own[source]);
                radius_on_axis.push(i64::from(structural_axis(dq, dr).is_some()));
            }
        }
        if !stones.is_empty() && cell_nearest[destination] > engine::LEGAL_RADIUS as i64 {
            return Err(format!(
                "legal cell ({}, {}) has no stone within legality radius",
                cell.q, cell.r
            ));
        }
    }

    let cell_index: FxHashMap<_, _> = legal
        .iter()
        .enumerate()
        .map(|(index, &coord)| (pack(coord), index as i64))
        .collect();
    let axes = [(1i16, 0i16), (0, 1), (1, -1)];
    let mut adjacency = Vec::with_capacity(legal.len() * 6);
    for (source, cell) in legal.iter().enumerate() {
        for (axis, &(dq, dr)) in axes.iter().enumerate() {
            for sign in [-1i16, 1] {
                let neighbor = engine::HexCoord::new(cell.q + sign * dq, cell.r + sign * dr);
                if let Some(&destination) = cell_index.get(&pack(neighbor)) {
                    adjacency.push((destination, source as i64, axis as i64));
                }
            }
        }
    }
    adjacency.sort_unstable();
    let mut adjacency_src = Vec::with_capacity(adjacency.len());
    let mut adjacency_dst = Vec::with_capacity(adjacency.len());
    let mut adjacency_axis = Vec::with_capacity(adjacency.len());
    for (destination, source, axis) in adjacency {
        adjacency_src.push(source);
        adjacency_dst.push(destination);
        adjacency_axis.push(axis);
    }
    Ok((
        cell_qr,
        cell_occupancy,
        cell_is_legal,
        cell_nearest,
        radius_src,
        radius_dst,
        radius_orbit,
        radius_own,
        radius_on_axis,
        adjacency_src,
        adjacency_dst,
        adjacency_axis,
    ))
}

/// Build one live position's graph with indices local to that position.
///
/// Returns an error for terminal positions.
///
/// Every nonempty candidate is represented under the ternary tables, and each
/// legal action carries its dense post-placement row tables.
pub fn build(pos: &engine::Position) -> Result<Graph, String> {
    if pos.is_terminal() {
        return Err("terminal position: the builder refuses it".into());
    }
    let mover = pos.current_player();
    let moves_remaining = match pos.phase() {
        engine::TurnPhase::FirstStone => 2,
        engine::TurnPhase::Opening | engine::TurnPhase::SecondStone => 1,
    };

    let stones: Vec<(engine::HexCoord, engine::Player)> = pos.stones().collect();
    let stone_own: Vec<i64> = stones.iter().map(|&(_, p)| (p != mover) as i64).collect();
    let stone_qr: Vec<[i32; 2]> = stones
        .iter()
        .map(|&(c, _)| [c.q as i32, c.r as i32])
        .collect();
    let stone_index: FxHashMap<i64, i64> = stones
        .iter()
        .enumerate()
        .map(|(i, &(c, _))| (pack(c), i as i64))
        .collect();

    let legal: Vec<engine::HexCoord> = pos.legal_actions().map(|a| a.coord()).collect();
    let n_legal = legal.len();
    let (
        cell_qr,
        cell_occupancy,
        cell_is_legal,
        cell_nearest,
        radius_src,
        radius_dst,
        radius_orbit,
        radius_own,
        radius_on_axis,
        adjacency_src,
        adjacency_dst,
        adjacency_axis,
    ) = cell_node_fields(&stones, &stone_own, &legal)?;

    if stones.is_empty() {
        // Ply 0: every action row is an empty insert with no source window.
        let mut actions = ActionTables::default();
        for _ in 0..n_legal {
            for row in 0..engine::WINDOWS_PER_PLACEMENT {
                let slot = row % engine::WINDOW_LEN;
                actions.window_index.push(-1);
                actions
                    .post1_class
                    .push(TERN_POST1_CLASS[POW3[slot] as usize * engine::WINDOW_LEN + slot] as i64);
                actions.pre_status.push(ACTION_EMPTY);
            }
        }
        return Ok(Graph {
            stone_own,
            stone_qr,
            window_feat: vec![],
            window_id: vec![],
            inc_stone: vec![],
            inc_window: vec![],
            inc_class: vec![],
            n_legal,
            cell_qr,
            cell_occupancy,
            cell_is_legal,
            cell_nearest,
            radius_src,
            radius_dst,
            radius_orbit,
            radius_own,
            radius_on_axis,
            adjacency_src,
            adjacency_dst,
            adjacency_axis,
            dec_cell: vec![],
            dec_window: vec![],
            dec_class: vec![],
            moves_remaining,
            action_window_index: actions.window_index,
            action_post1_class: actions.post1_class,
            action_pre_status: actions.pre_status,
        });
    }

    // Candidate windows through every stone, deduplicated and sorted by packed key.
    let mut candidates: Vec<(i64, engine::WindowRef, u8, u8)> =
        Vec::with_capacity(stones.len() * 18);
    for &(c, _) in &stones {
        for wr in pos.windows_through(c) {
            if !wr.window.start.is_valid() {
                continue;
            }
            let key = pack(wr.window.start) * 4 + wr.window.axis.index() as i64;
            candidates.push((
                key,
                wr,
                wr.mask.mask(engine::Player::P0),
                wr.mask.mask(engine::Player::P1),
            ));
        }
    }
    candidates.sort_unstable_by_key(|&(key, ..)| key);
    candidates.dedup_by_key(|&mut (key, ..)| key);

    let mut window_feat = Vec::new();
    let mut window_id = Vec::new();
    let mut live_occ = Vec::new();
    let mut patterns: Vec<u16> = Vec::new();
    let mut live_ref = Vec::new();
    // Sorted for binary-search lookup by the decoder.
    let mut live_keys: Vec<i64> = Vec::new();
    for &(key, wr, m0, m1) in &candidates {
        let (own, opp) = if mover == engine::Player::P0 {
            (m0, m1)
        } else {
            (m1, m0)
        };
        let occ = m0 | m1;
        let mut pattern = 0u16;
        for (k, &place) in POW3.iter().enumerate() {
            let digit = (own >> k & 1) as u16 + 2 * (opp >> k & 1) as u16;
            pattern += digit * place;
        }
        window_feat.push(TERN_RANK[pattern as usize] as i64);
        patterns.push(pattern);
        live_keys.push(key);
        window_id.push(wr.window.axis.index() as i64);
        window_id.push(wr.window.start.q as i64);
        window_id.push(wr.window.start.r as i64);
        live_occ.push(occ);
        live_ref.push(wr);
    }

    // Incidence: window-major, slot-ascending.
    let mut inc_stone = Vec::new();
    let mut inc_window = Vec::new();
    let mut inc_class = Vec::new();
    for (w, (&occ, wr)) in live_occ.iter().zip(&live_ref).enumerate() {
        for k in 0..6 {
            if occ >> k & 1 == 1 {
                let cell = wr.window.cell(k);
                inc_stone.push(stone_index[&pack(cell)]);
                inc_window.push(w as i64);
                inc_class.push(TERN_OCC_CLASS[patterns[w] as usize * 6 + k] as i64);
            }
        }
    }

    // Decoder table: legal-cell-major, then (axis, offset) order. The Step 4
    // action rows ride the same walk: the candidate's mask gives the row's
    // status and pre pattern, the inserted stone is one power-of-three away,
    // and the decoder's own binary search is the kept-window index — no
    // second engine walk. The kept/decoder agreement stays asserted per row.
    let mut dec_cell = Vec::new();
    let mut dec_window = Vec::new();
    let mut dec_class = Vec::new();
    let mut actions = ActionTables::default();
    let rows = n_legal
        .checked_mul(engine::WINDOWS_PER_PLACEMENT)
        .ok_or_else(|| "action row count overflows usize".to_owned())?;
    actions.window_index.reserve(rows);
    actions.post1_class.reserve(rows);
    actions.pre_status.reserve(rows);
    for (j, &cell) in legal.iter().enumerate() {
        for (i, wr) in pos.windows_through(cell).into_iter().enumerate() {
            let slot = i % 6;
            let found = if wr.window.start.is_valid() {
                let key = pack(wr.window.start) * 4 + wr.window.axis.index() as i64;
                live_keys.binary_search(&key).map_or(-1, |w| w as i64)
            } else {
                -1
            };
            if found >= 0 {
                let w = found as usize;
                let class = TERN_DEC_CLASS[patterns[w] as usize * 6 + slot] as i64;
                assert!(
                    class >= 0,
                    "legal cell {cell:?} sits at slot {slot} of a window whose \
                     occupancy {:06b} already fills it",
                    live_occ[w]
                );
                dec_cell.push(j as i64);
                dec_window.push(w as i64);
                dec_class.push(class);
            }
            let m0 = wr.mask.mask(engine::Player::P0);
            let m1 = wr.mask.mask(engine::Player::P1);
            let (own, opp) = if mover == engine::Player::P0 {
                (m0, m1)
            } else {
                (m1, m0)
            };
            if (own | opp) >> slot & 1 != 0 {
                return Err(format!(
                    "legal action {j} at ({}, {}) is occupied in slot {slot}",
                    cell.q, cell.r
                ));
            }
            let status = match (own != 0, opp != 0) {
                (true, false) => ACTION_OWN,
                (false, true) => ACTION_OPP,
                (false, false) => ACTION_EMPTY,
                (true, true) => ACTION_MIXED,
            };
            let mut pre = 0usize;
            for (k, &power) in POW3.iter().enumerate() {
                let digit = ((own >> k) & 1) as usize + 2 * ((opp >> k) & 1) as usize;
                pre += digit * power as usize;
            }
            let class =
                TERN_POST1_CLASS[(pre + POW3[slot] as usize) * engine::WINDOW_LEN + slot] as i64;
            if class < 0 {
                return Err("a post-placement row lost its own stone".to_owned());
            }
            if (found >= 0) != (status != ACTION_EMPTY) {
                return Err("the kept-window set disagrees with the action-row walk".to_owned());
            }
            actions.window_index.push(found);
            actions.post1_class.push(class);
            actions.pre_status.push(status);
        }
    }

    Ok(Graph {
        stone_own,
        stone_qr,
        window_feat,
        window_id,
        inc_stone,
        inc_window,
        inc_class,
        n_legal,
        cell_qr,
        cell_occupancy,
        cell_is_legal,
        cell_nearest,
        radius_src,
        radius_dst,
        radius_orbit,
        radius_own,
        radius_on_axis,
        adjacency_src,
        adjacency_dst,
        adjacency_axis,
        dec_cell,
        dec_window,
        dec_class,
        moves_remaining,
        action_window_index: actions.window_index,
        action_post1_class: actions.post1_class,
        action_pre_status: actions.pre_status,
    })
}

/// Everything `mantisnet.builder.Batch` holds, as flat vectors plus shapes.
#[derive(Debug, PartialEq, Eq)]
pub struct RawBatch {
    /// Number of positions in the batch.
    pub n_pos: usize,
    /// Padded width of each position's `[four state latents; stones]` table.
    pub max_t: usize,
    /// Padded width of each position's `[pooled global context; windows]` table.
    pub max_w: usize,
    /// Stone owner features, relative to each position's mover.
    pub stone_own: Vec<i64>,
    /// Live-window colour and canonical-pattern features.
    pub window_feat: Vec<i64>,
    /// Live-window identities as `(axis, start_q, start_r)` triples, flat in
    /// `(N_w, 3)` row-major layout. Coordinates are position-local; the model
    /// consumes them only through reversal-invariant pair classes.
    pub window_id: Vec<i64>,
    /// `moves_remaining - 1` for each position.
    pub moves_idx: Vec<i64>,
    /// Global stone index for each stone-to-window incidence.
    pub inc_stone: Vec<i64>,
    /// Global window index for each stone-to-window incidence.
    pub inc_window: Vec<i64>,
    /// Reversal-invariant joint occupancy/slot class for each stone-to-window
    /// incidence.
    pub inc_class: Vec<i64>,
    /// Flat padded-table slot occupied by each stone.
    pub stone_slot: Vec<i64>,
    /// Stone coordinates in `(P, max_t, 2)` row-major layout.
    pub coords: Vec<i32>,
    /// Valid rows in the `(P, max_t)` stone-attention table.
    pub attn_valid: Vec<bool>,
    /// Flat padded-table slot occupied by each live window.
    pub window_slot: Vec<i64>,
    /// Valid rows in the `(P, max_w)` value-readout table.
    pub value_valid: Vec<bool>,
    /// CSR offsets delimiting each position's legal cells.
    pub legal_offsets: Vec<i64>,
    /// Position index for each concatenated legal cell.
    pub cell_pos: Vec<i64>,
    /// Mover-relative occupancy of every legal cell (currently all EMPTY).
    pub cell_occupancy: Vec<i64>,
    /// Legality flag of every represented legal cell.
    pub cell_is_legal: Vec<i64>,
    /// Nearest-stone distance bucket; 9 is the stone-free opening sentinel.
    pub cell_nearest: Vec<i64>,
    /// Stone-to-cell radius edges and their invariant fields.
    pub radius_src: Vec<i64>,
    /// Legal-cell destination of each radius edge.
    pub radius_dst: Vec<i64>,
    /// Exact D6 displacement class of each radius edge.
    pub radius_orbit: Vec<i64>,
    /// Mover-relative ownership of each radius-edge source.
    pub radius_own: Vec<i64>,
    /// Whether each radius displacement lies on any structural axis.
    pub radius_on_axis: Vec<i64>,
    /// Directed legal-cell adjacency and its structural axis route.
    pub adjacency_src: Vec<i64>,
    /// Legal-cell destination of each directed adjacency edge.
    pub adjacency_dst: Vec<i64>,
    /// Structural axis of each adjacency edge.
    pub adjacency_axis: Vec<i64>,
    /// Global legal-cell index for each decoder incidence.
    pub dec_cell: Vec<i64>,
    /// Global live-window index for each decoder incidence.
    pub dec_window: Vec<i64>,
    /// Reversal-invariant joint occupancy/slot class for each decoder incidence.
    pub dec_class: Vec<i64>,
    /// Post-placement class for each decoder incidence, in decoder order.
    pub act_class: Vec<i64>,
    /// Stable window-major permutation of the decoder incidences.
    pub act_rev: Vec<i64>,
    /// EMPTY-row counts in `(legal cell, slot orbit)` row-major order.
    pub act_empty: Vec<i64>,
}

fn collate_action_fields(graphs: &[Graph], dec_window: &[i64]) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut act_class = Vec::with_capacity(dec_window.len());
    let mut act_empty = Vec::with_capacity(
        graphs
            .iter()
            .map(|graph| graph.n_legal * ACTION_EMPTY_ORBITS)
            .sum(),
    );

    for graph in graphs {
        let rows = graph.n_legal * engine::WINDOWS_PER_PLACEMENT;
        assert_eq!(
            graph.action_window_index.len(),
            rows,
            "action-row window table has the wrong dense row count"
        );
        assert_eq!(
            graph.action_post1_class.len(),
            rows,
            "action-row class table has the wrong dense row count"
        );
        assert_eq!(
            graph.action_pre_status.len(),
            rows,
            "action-row status table has the wrong dense row count"
        );

        let kept_flat: Vec<usize> = graph
            .action_pre_status
            .iter()
            .enumerate()
            .filter_map(|(flat, &status)| (status != ACTION_EMPTY).then_some(flat))
            .collect();
        assert_eq!(
            kept_flat.len(),
            graph.dec_cell.len(),
            "action-row cells disagree with the decoder walk: kept-row count diverged"
        );
        for (&flat, &cell) in kept_flat.iter().zip(&graph.dec_cell) {
            assert_eq!(
                flat / engine::WINDOWS_PER_PLACEMENT,
                cell as usize,
                "action-row cells disagree with the decoder walk: kept row {flat} maps to a different cell"
            );
        }
        let kept_windows: Vec<i64> = kept_flat
            .iter()
            .map(|&flat| graph.action_window_index[flat])
            .collect();
        assert_eq!(
            kept_windows, graph.dec_window,
            "action-row windows disagree with the decoder walk"
        );
        act_class.extend(kept_flat.iter().map(|&flat| graph.action_post1_class[flat]));

        for cell in 0..graph.n_legal {
            let mut counts = [0i64; ACTION_EMPTY_ORBITS];
            let base = cell * engine::WINDOWS_PER_PLACEMENT;
            for axis in 0..engine::Axis::ALL.len() {
                for slot in 0..engine::WINDOW_LEN {
                    let flat = base + axis * engine::WINDOW_LEN + slot;
                    if graph.action_pre_status[flat] == ACTION_EMPTY {
                        counts[slot.min(engine::WINDOW_LEN - 1 - slot)] += 1;
                    }
                }
            }
            act_empty.extend_from_slice(&counts);
        }
    }

    let mut act_rev: Vec<i64> = (0..dec_window.len() as i64).collect();
    act_rev.sort_by_key(|&edge| dec_window[edge as usize]);
    (act_class, act_rev, act_empty)
}

/// Collate position-local graphs into one globally indexed ragged batch.
pub fn collate(graphs: &[Graph]) -> RawBatch {
    let p = graphs.len();
    let max_t = graphs.iter().map(|g| g.stone_own.len()).max().unwrap_or(0) + 4;
    let max_w = graphs
        .iter()
        .map(|g| g.window_feat.len())
        .max()
        .unwrap_or(0)
        + 1;

    let mut out = RawBatch {
        n_pos: p,
        max_t,
        max_w,
        stone_own: vec![],
        window_feat: vec![],
        window_id: vec![],
        moves_idx: Vec::with_capacity(p),
        inc_stone: vec![],
        inc_window: vec![],
        inc_class: vec![],
        stone_slot: vec![],
        coords: vec![0; p * max_t * 2],
        attn_valid: vec![false; p * max_t],
        window_slot: vec![],
        value_valid: vec![false; p * max_w],
        legal_offsets: Vec::with_capacity(p + 1),
        cell_pos: vec![],
        cell_occupancy: vec![],
        cell_is_legal: vec![],
        cell_nearest: vec![],
        radius_src: vec![],
        radius_dst: vec![],
        radius_orbit: vec![],
        radius_own: vec![],
        radius_on_axis: vec![],
        adjacency_src: vec![],
        adjacency_dst: vec![],
        adjacency_axis: vec![],
        dec_cell: vec![],
        dec_window: vec![],
        dec_class: vec![],
        act_class: vec![],
        act_rev: vec![],
        act_empty: vec![],
    };

    let (mut stone_off, mut win_off, mut cell_off) = (0i64, 0i64, 0i64);
    out.legal_offsets.push(0);
    for (i, g) in graphs.iter().enumerate() {
        let (ns, nw) = (g.stone_own.len(), g.window_feat.len());
        out.stone_own.extend_from_slice(&g.stone_own);
        out.window_feat.extend_from_slice(&g.window_feat);
        out.window_id.extend_from_slice(&g.window_id);
        out.moves_idx.push(g.moves_remaining as i64 - 1);
        out.inc_stone
            .extend(g.inc_stone.iter().map(|&s| s + stone_off));
        out.inc_window
            .extend(g.inc_window.iter().map(|&w| w + win_off));
        out.inc_class.extend_from_slice(&g.inc_class);
        out.stone_slot
            .extend((0..ns).map(|j| (i * max_t + 4 + j) as i64));
        out.window_slot
            .extend((0..nw).map(|j| (i * max_w + 1 + j) as i64));
        out.attn_valid[i * max_t..i * max_t + 4].fill(true);
        out.value_valid[i * max_w] = true;
        for (j, qr) in g.stone_qr.iter().enumerate() {
            out.coords[(i * max_t + 4 + j) * 2] = qr[0];
            out.coords[(i * max_t + 4 + j) * 2 + 1] = qr[1];
            out.attn_valid[i * max_t + 4 + j] = true;
        }
        for j in 0..nw {
            out.value_valid[i * max_w + 1 + j] = true;
        }
        cell_off += g.n_legal as i64;
        out.legal_offsets.push(cell_off);
        out.cell_pos
            .extend(std::iter::repeat_n(i as i64, g.n_legal));
        out.cell_occupancy.extend_from_slice(&g.cell_occupancy);
        out.cell_is_legal.extend_from_slice(&g.cell_is_legal);
        out.cell_nearest.extend_from_slice(&g.cell_nearest);
        out.radius_src
            .extend(g.radius_src.iter().map(|&source| source + stone_off));
        out.radius_dst.extend(
            g.radius_dst
                .iter()
                .map(|&destination| destination + cell_off - g.n_legal as i64),
        );
        out.radius_orbit.extend_from_slice(&g.radius_orbit);
        out.radius_own.extend_from_slice(&g.radius_own);
        out.radius_on_axis.extend_from_slice(&g.radius_on_axis);
        out.adjacency_src.extend(
            g.adjacency_src
                .iter()
                .map(|&source| source + cell_off - g.n_legal as i64),
        );
        out.adjacency_dst.extend(
            g.adjacency_dst
                .iter()
                .map(|&destination| destination + cell_off - g.n_legal as i64),
        );
        out.adjacency_axis.extend_from_slice(&g.adjacency_axis);
        out.dec_cell
            .extend(g.dec_cell.iter().map(|&c| c + cell_off - g.n_legal as i64));
        out.dec_window
            .extend(g.dec_window.iter().map(|&w| w + win_off));
        out.dec_class.extend_from_slice(&g.dec_class);
        stone_off += ns as i64;
        win_off += nw as i64;
    }
    (out.act_class, out.act_rev, out.act_empty) = collate_action_fields(graphs, &out.dec_window);
    out
}

/// Decode worker-produced position items and collate them into one model batch.
///
/// Each input slice must contain exactly one item written by
/// [`encode_position`]. Unknown representation versions, malformed features
/// or indices, truncation, and trailing bytes are all refused.
pub fn decode_batch<'a>(items: impl IntoIterator<Item = &'a [u8]>) -> Result<RawBatch, WireError> {
    let graphs = items
        .into_iter()
        .enumerate()
        .map(|(item, bytes)| decode_graph(bytes).map_err(|error| error.at_item(item)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(collate(&graphs))
}

/// Build every position in parallel, then collate.
pub fn build_batch(positions: &[engine::Position]) -> Result<RawBatch, String> {
    let graphs: Vec<Graph> = positions.par_iter().map(build).collect::<Result<_, _>>()?;
    Ok(collate(&graphs))
}

/// Replay each game's first `t` placements, then build, in parallel.
pub fn build_batch_prefixes(games: &[Vec<(i16, i16)>], ts: &[usize]) -> Result<RawBatch, String> {
    if games.len() != ts.len() {
        return Err("games and ts must have equal length".into());
    }
    let graphs: Vec<Graph> = games
        .par_iter()
        .zip(ts)
        .map(|(moves, &t)| {
            if t > moves.len() {
                return Err(format!(
                    "prefix length {t} exceeds game length {}",
                    moves.len()
                ));
            }
            let actions: Vec<engine::Action> = moves[..t]
                .iter()
                .map(|&(q, r)| engine::Action::new(engine::HexCoord::new(q, r)))
                .collect();
            let pos = engine::Position::replay(&actions).map_err(|e| e.to_string())?;
            build(&pos)
        })
        .collect::<Result<_, _>>()?;
    Ok(collate(&graphs))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_GAMES: &[&[(i16, i16)]] = &[
        &[],
        &[(0, 0)],
        &[(0, 0), (1, 0), (2, 0)],
        &[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
        &[
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (2, 0),
            (0, 3),
            (0, 4),
            (3, 0),
        ],
        &[
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, -1),
            (0, 2),
            (4, -1),
            (-1, 2),
        ],
    ];

    fn replay(moves: &[(i16, i16)]) -> engine::Position {
        let actions: Vec<_> = moves
            .iter()
            .map(|&(q, r)| engine::Action::new(engine::HexCoord::new(q, r)))
            .collect();
        engine::Position::replay(&actions).expect("legal test position")
    }

    fn encoded(position: &engine::Position) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_position(position, &mut bytes);
        bytes
    }

    fn expected_action_row(own: u8, opp: u8, slot: usize) -> (i64, i64) {
        let status = match (own != 0, opp != 0) {
            (true, false) => ACTION_OWN,
            (false, true) => ACTION_OPP,
            (false, false) => ACTION_EMPTY,
            (true, true) => ACTION_MIXED,
        };
        let mut pre = 0usize;
        for (k, &power) in POW3.iter().enumerate() {
            let digit = ((own >> k) & 1) as usize + 2 * ((opp >> k) & 1) as usize;
            pre += digit * power as usize;
        }
        let post = pre + POW3[slot] as usize;
        let class = TERN_POST1_CLASS[post * engine::WINDOW_LEN + slot] as i64;
        (class, status)
    }

    #[test]
    fn action_post1_class_tables_obey_the_reversal_laws() {
        assert_eq!(TERN_POST1_CLASSES, 729);
        assert_eq!(
            TERN_POST1_CLASS.iter().copied().max(),
            Some((TERN_POST1_CLASSES - 1) as i16)
        );
        let mut tern_seen = [false; TERN_POST1_CLASSES as usize];
        for post in 0..729 {
            let reversed = reverse3(post);
            for slot in 0..engine::WINDOW_LEN {
                let class = TERN_POST1_CLASS[post * engine::WINDOW_LEN + slot];
                let mirror =
                    TERN_POST1_CLASS[reversed * engine::WINDOW_LEN + engine::WINDOW_LEN - 1 - slot];
                if (post / POW3[slot] as usize) % 3 == 1 {
                    assert!(class >= 0);
                    assert_eq!(class, mirror);
                    tern_seen[class as usize] = true;
                } else {
                    assert_eq!(class, -1);
                }
            }
        }
        assert!(tern_seen.into_iter().all(|seen| seen));
    }

    #[test]
    fn action_rows_match_successor_windows() {
        let mut rows_checked = 0;
        for &moves in FIXTURE_GAMES {
            let position = replay(moves);
            let mover = position.current_player();
            let graph = build(&position).expect("fixture graph builds");
            let legal: Vec<_> = position
                .legal_actions()
                .map(|action| action.coord())
                .collect();
            let picks: Vec<_> = if legal.len() <= 40 {
                (0..legal.len()).collect()
            } else {
                (0..legal.len()).step_by(7).collect()
            };

            for action in picks {
                let cell = legal[action];
                let mut successor = position.clone();
                successor
                    .advance(engine::Action::new(cell))
                    .expect("selected legal action advances");
                for (row, wr) in successor.windows_through(cell).into_iter().enumerate() {
                    if !wr.window.start.is_valid() {
                        continue;
                    }
                    let slot = row % engine::WINDOW_LEN;
                    let m0 = wr.mask.mask(engine::Player::P0);
                    let m1 = wr.mask.mask(engine::Player::P1);
                    let (own_post, opp_post) = if mover == engine::Player::P0 {
                        (m0, m1)
                    } else {
                        (m1, m0)
                    };
                    assert_eq!(own_post >> slot & 1, 1, "the played stone is missing");
                    let own_pre = own_post & !(1 << slot);
                    let (want_class, want_status) = expected_action_row(own_pre, opp_post, slot);
                    let index = action * engine::WINDOWS_PER_PLACEMENT + row;
                    assert_eq!(graph.action_post1_class[index], want_class);
                    assert_eq!(graph.action_pre_status[index], want_status);

                    let got_window = graph.action_window_index[index];
                    let kept = want_status != ACTION_EMPTY;
                    assert_eq!(got_window >= 0, kept);
                    if got_window >= 0 {
                        let id = got_window as usize * 3;
                        assert_eq!(graph.window_id[id], wr.window.axis.index() as i64);
                        assert_eq!(graph.window_id[id + 1], wr.window.start.q as i64);
                        assert_eq!(graph.window_id[id + 2], wr.window.start.r as i64);
                    }
                    rows_checked += 1;
                }
            }
        }
        assert!(rows_checked > 500);
    }

    #[test]
    fn ply_zero_action_rows_are_empty_inserts() {
        let graph = build(&engine::Position::new()).expect("the opening is live");
        assert_eq!(graph.action_window_index.len(), 3 * engine::WINDOW_LEN);
        assert!(graph.action_window_index.iter().all(|&index| index == -1));
        assert!(
            graph
                .action_pre_status
                .iter()
                .all(|&status| status == ACTION_EMPTY)
        );
        for axis in 0..3 {
            for slot in 0..engine::WINDOW_LEN {
                let row = axis * engine::WINDOW_LEN + slot;
                assert_eq!(
                    graph.action_post1_class[row],
                    expected_action_row(0, 0, slot).0
                );
            }
        }
    }

    #[test]
    fn collated_action_fields_match_the_dense_fixture_rows() {
        let games: Vec<Vec<(i16, i16)>> =
            FIXTURE_GAMES.iter().map(|moves| moves.to_vec()).collect();
        let ts: Vec<usize> = games.iter().map(Vec::len).collect();
        let raw = build_batch_prefixes(&games, &ts).expect("fixture prefixes build");

        let graphs: Vec<Graph> = FIXTURE_GAMES
            .iter()
            .map(|moves| build(&replay(moves)).expect("fixture graph builds"))
            .collect();
        let mut expected_class = Vec::new();
        let mut expected_empty = Vec::new();
        for graph in &graphs {
            for cell in 0..graph.n_legal {
                let mut counts = [0i64; ACTION_EMPTY_ORBITS];
                for axis in 0..engine::Axis::ALL.len() {
                    for slot in 0..engine::WINDOW_LEN {
                        let flat =
                            cell * engine::WINDOWS_PER_PLACEMENT + axis * engine::WINDOW_LEN + slot;
                        if graph.action_pre_status[flat] == ACTION_EMPTY {
                            counts[slot.min(engine::WINDOW_LEN - 1 - slot)] += 1;
                        } else {
                            expected_class.push(graph.action_post1_class[flat]);
                        }
                    }
                }
                expected_empty.extend_from_slice(&counts);
            }
        }
        let mut expected_rev: Vec<i64> = (0..raw.dec_window.len() as i64).collect();
        expected_rev.sort_by_key(|&edge| raw.dec_window[edge as usize]);

        assert_eq!(raw.act_class, expected_class);
        assert_eq!(raw.act_rev, expected_rev);
        assert_eq!(raw.act_empty, expected_empty);
    }

    #[test]
    fn ply_zero_prefix_collates_six_empty_rows_per_slot_orbit() {
        let raw = build_batch_prefixes(&[vec![]], &[0]).expect("the opening is live");
        assert!(raw.act_class.is_empty());
        assert!(raw.act_rev.is_empty());
        assert_eq!(raw.act_empty, [6, 6, 6]);
    }

    #[test]
    fn the_opening_batch_has_only_the_state_latents_and_empty_action_rows() {
        let raw = build_batch(&[engine::Position::new()]).expect("the opening is live");

        assert_eq!(raw.n_pos, 1);
        assert_eq!(raw.max_t, 4);
        assert_eq!(raw.max_w, 1);
        assert!(raw.stone_own.is_empty());
        assert!(raw.window_feat.is_empty());
        assert_eq!(raw.moves_idx, [0]);
        assert!(raw.inc_stone.is_empty());
        assert!(raw.inc_window.is_empty());
        assert!(raw.inc_class.is_empty());
        assert!(raw.stone_slot.is_empty());
        assert_eq!(raw.coords, [0; 8]);
        assert_eq!(raw.attn_valid, [true; 4]);
        assert!(raw.window_slot.is_empty());
        assert_eq!(raw.value_valid, [true]);
        assert_eq!(raw.legal_offsets, [0, 1]);
        assert_eq!(raw.cell_pos, [0]);
        assert!(raw.dec_cell.is_empty());
        assert!(raw.dec_window.is_empty());
        assert!(raw.dec_class.is_empty());
        assert!(raw.act_class.is_empty());
        assert!(raw.act_rev.is_empty());
        assert_eq!(raw.act_empty, [6, 6, 6]);
    }

    #[test]
    fn every_nonempty_candidate_uses_ternary_classes() {
        let position = replay(&[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]);
        let graph = build(&position).expect("live position");

        // Every nonempty candidate window through a stone, deduplicated.
        let mut keys = std::collections::HashSet::new();
        for (c, _) in position.stones() {
            for wr in position.windows_through(c) {
                if wr.window.start.is_valid()
                    && (wr.mask.mask(engine::Player::P0) | wr.mask.mask(engine::Player::P1)) != 0
                {
                    keys.insert(pack(wr.window.start) * 4 + wr.window.axis.index() as i64);
                }
            }
        }
        assert_eq!(graph.window_feat.len(), keys.len());

        for &feat in &graph.window_feat {
            assert!((0..TERN_PATTERNS).contains(&feat));
        }
        for &class in &graph.inc_class {
            assert!((0..TERN_OCC_CLASSES).contains(&class));
        }
        for &class in &graph.dec_class {
            assert!((0..TERN_DEC_CLASSES).contains(&class));
        }
    }

    #[test]
    fn prefix_replay_and_position_build_share_the_same_core() {
        let moves = vec![(0, 0), (1, 0), (2, 0), (0, 1)];
        let actions: Vec<engine::Action> = moves
            .iter()
            .map(|&(q, r)| engine::Action::new(engine::HexCoord::new(q, r)))
            .collect();
        let position = engine::Position::replay(&actions).expect("legal fixture");

        let direct = build_batch(&[position]).expect("live position");
        let replayed =
            build_batch_prefixes(&[moves], &[actions.len()]).expect("legal prefix fixture");
        assert_eq!(direct, replayed);
    }

    #[test]
    fn malformed_prefix_requests_are_refused() {
        let unequal = build_batch_prefixes(&[vec![(0, 0)]], &[]).expect_err("lengths must agree");
        assert_eq!(unequal, "games and ts must have equal length");

        let too_long = build_batch_prefixes(&[vec![(0, 0)]], &[2]).expect_err("prefix is too long");
        assert_eq!(too_long, "prefix length 2 exceeds game length 1");
    }

    #[test]
    fn wire_round_trip_matches_direct_build_for_different_position_shapes() {
        let positions = vec![
            engine::Position::new(),
            replay(&[(0, 0)]),
            replay(&[(0, 0), (1, 0), (2, 0), (0, 1)]),
            replay(&[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]),
        ];
        let items: Vec<Vec<u8>> = positions.iter().map(encoded).collect();

        let decoded =
            decode_batch(items.iter().map(Vec::as_slice)).expect("encoder output is valid");
        let direct = build_batch(&positions).expect("all positions are live");
        assert_eq!(decoded, direct);
    }

    #[test]
    fn wire_encoder_only_appends_to_the_callers_buffer() {
        let prefix = [0x55, 0xaa, 0x13, 0x37];
        let mut bytes = prefix.to_vec();
        encode_position(&replay(&[(0, 0), (1, 0)]), &mut bytes);

        assert_eq!(&bytes[..prefix.len()], prefix);
        decode_batch(std::iter::once(&bytes[prefix.len()..]))
            .expect("the appended suffix is one complete item");
    }

    #[test]
    fn wire_decoder_rejects_magic_version_truncation_and_trailing_bytes() {
        let opening = engine::Position::new();
        let bytes = encoded(&opening);

        let mut wrong_magic = bytes.clone();
        wrong_magic[0] ^= 0xff;
        assert!(
            decode_batch(std::iter::once(wrong_magic.as_slice()))
                .expect_err("magic is part of the contract")
                .to_string()
                .contains("wrong magic")
        );

        let mut wrong_version = bytes.clone();
        wrong_version[WIRE_MAGIC.len()..WIRE_MAGIC.len() + 4]
            .copy_from_slice(&(crate::MODEL_REPR_VERSION + 1).to_le_bytes());
        assert!(
            decode_batch(std::iter::once(wrong_version.as_slice()))
                .expect_err("versions never fall through")
                .to_string()
                .contains("does not match")
        );

        let mut truncated = bytes.clone();
        truncated.pop();
        assert!(
            decode_batch(std::iter::once(truncated.as_slice()))
                .expect_err("truncation is refused")
                .to_string()
                .contains("truncated payload")
        );

        let mut trailing = bytes;
        trailing.push(0);
        assert!(
            decode_batch(std::iter::once(trailing.as_slice()))
                .expect_err("concatenated or dirty items are refused")
                .to_string()
                .contains("trailing bytes")
        );
    }

    #[test]
    fn wire_decoder_validates_counts_and_features_before_collation() {
        let mut huge_legal_count = encoded(&engine::Position::new());
        let legal_count_offset = WIRE_MAGIC.len() + 4 + 4 + 3 * 4;
        huge_legal_count[legal_count_offset..legal_count_offset + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(
            decode_batch(std::iter::once(huge_legal_count.as_slice()))
                .expect_err("item-sized count cap must run before allocation")
                .to_string()
                .contains("exceeds item length")
        );

        let mut invalid_owner = encoded(&replay(&[(0, 0)]));
        invalid_owner[WIRE_HEADER_LEN..WIRE_HEADER_LEN + 8].copy_from_slice(&2i64.to_le_bytes());
        assert!(
            decode_batch(std::iter::once(invalid_owner.as_slice()))
                .expect_err("features have a closed domain")
                .to_string()
                .contains("stone_own[0] has invalid feature 2")
        );
    }

    #[test]
    fn wire_decoder_bounds_the_joint_decoder_class() {
        let position = replay(&[(0, 0)]);
        let graph = build(&position).expect("a one-stone position builds");
        let counts = WireCounts::from_graph(&graph);
        // dec_class follows stone_own, stone_qr, window_feat, window_id, the
        // three incidence arrays, dec_cell, and dec_window.
        let offset = WIRE_HEADER_LEN
            + 8 * counts.stones
            + 8 * counts.stones
            + 8 * counts.windows
            + 8 * 3 * counts.windows
            + 8 * 3 * counts.incidences
            + 8 * 2 * counts.decoder;

        let build_max = graph
            .dec_class
            .iter()
            .copied()
            .max()
            .expect("entries exist");
        assert!((0..TERN_DEC_CLASSES).contains(&build_max));
        for out_of_range in [TERN_DEC_CLASSES, -1] {
            let mut bytes = encoded(&position);
            bytes[offset..offset + 8].copy_from_slice(&out_of_range.to_le_bytes());
            assert!(
                decode_batch(std::iter::once(bytes.as_slice()))
                    .expect_err("the class table has 726 rows and no more")
                    .to_string()
                    .contains(&format!("dec_class[0] has invalid feature {out_of_range}"))
            );
        }
    }

    fn oracle_images(q: i16, r: i16) -> BTreeSet<(i16, i16)> {
        let cube = [q, r, -q - r];
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut images = BTreeSet::new();
        for permutation in permutations {
            for sign in [-1i16, 1] {
                images.insert((sign * cube[permutation[0]], sign * cube[permutation[1]]));
            }
        }
        images
    }

    #[test]
    fn orbit48_matches_an_independent_cube_permutation_oracle() {
        let mut representatives = BTreeSet::new();
        for dq in -12..=12 {
            for dr in -12..=12 {
                let distance = displacement_distance(dq, dr);
                if (1..=12).contains(&distance) {
                    let representative = *oracle_images(dq, dr).iter().min().unwrap();
                    representatives.insert((distance, representative.0, representative.1));
                }
            }
        }
        assert_eq!(representatives.len(), ORBIT48_CLASSES);
        let ranks: FxHashMap<_, _> = representatives
            .into_iter()
            .enumerate()
            .map(|(rank, (_distance, q, r))| ((q, r), rank as i64))
            .collect();
        for dq in -12..=12 {
            for dr in -12..=12 {
                let distance = displacement_distance(dq, dr);
                if (1..=12).contains(&distance) {
                    let representative = *oracle_images(dq, dr).iter().min().unwrap();
                    assert_eq!(orbit_id(dq, dr, 12).unwrap(), ranks[&representative]);
                }
            }
        }
        assert!(orbit_id(1, 0, 13).unwrap_err().contains("1..=12"));
    }

    #[test]
    fn cell_node_edges_match_naive_geometry_and_cover_every_legal_cell() {
        let position = replay(&[(0, 0), (3, 0), (-2, 2), (0, 3)]);
        let graph = build(&position).unwrap();
        let stones: Vec<_> = position.stones().collect();
        let legal: Vec<_> = position
            .legal_actions()
            .map(|action| action.coord())
            .collect();
        let actual: Vec<_> = (0..graph.radius_src.len())
            .map(|edge| {
                (
                    graph.radius_dst[edge],
                    graph.radius_src[edge],
                    graph.radius_orbit[edge],
                    graph.radius_own[edge],
                    graph.radius_on_axis[edge],
                )
            })
            .collect();
        let mut expected = Vec::new();
        for (destination, cell) in legal.iter().enumerate() {
            for (source, &(stone, owner)) in stones.iter().enumerate() {
                let dq = cell.q - stone.q;
                let dr = cell.r - stone.r;
                if (1..=engine::LEGAL_RADIUS as usize).contains(&displacement_distance(dq, dr)) {
                    expected.push((
                        destination as i64,
                        source as i64,
                        orbit_id(dq, dr, 12).unwrap(),
                        i64::from(owner != position.current_player()),
                        i64::from(structural_axis(dq, dr).is_some()),
                    ));
                }
            }
        }
        assert_eq!(actual, expected);
        let destinations: BTreeSet<_> = graph.radius_dst.iter().copied().collect();
        assert_eq!(destinations.len(), legal.len());

        let mut expected_adjacency = Vec::new();
        for (source, a) in legal.iter().enumerate() {
            for (destination, b) in legal.iter().enumerate() {
                let dq = b.q - a.q;
                let dr = b.r - a.r;
                if displacement_distance(dq, dr) == 1 {
                    expected_adjacency.push((
                        destination as i64,
                        source as i64,
                        structural_axis(dq, dr).unwrap(),
                    ));
                }
            }
        }
        expected_adjacency.sort_unstable();
        let actual_adjacency: Vec<_> = graph
            .adjacency_dst
            .iter()
            .copied()
            .zip(graph.adjacency_src.iter().copied())
            .zip(graph.adjacency_axis.iter().copied())
            .map(|((dst, src), axis)| (dst, src, axis))
            .collect();
        assert_eq!(actual_adjacency, expected_adjacency);
    }

    #[test]
    fn fixture_cohort_pins_cell_node_growth() {
        let graphs: Vec<_> = FIXTURE_GAMES
            .iter()
            .map(|moves| build(&replay(moves)).unwrap())
            .collect();
        let totals = graphs.iter().fold([0usize; 4], |mut totals, graph| {
            totals[0] += graph.n_legal;
            totals[1] += graph
                .dec_cell
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len();
            totals[2] += graph.radius_src.len();
            totals[3] += graph.adjacency_src.len();
            totals
        });
        println!(
            "fixture positions={}, legal={}, covered={}, radius_edges={}, adjacency_edges={}",
            graphs.len(),
            totals[0],
            totals[1],
            totals[2],
            totals[3]
        );
        assert_eq!(totals, [1341, 396, 5866, 7366]);
    }
}
