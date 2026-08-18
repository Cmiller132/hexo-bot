# hexo-engine

The authoritative Hexo rules machine. `hexo-engine` defines legal placements,
turn progression, win detection, position hashing, replay, and reversible search
state for a two-player game on an infinite hexagonal board. It has no runtime
dependencies and contains no match policy, persistence, I/O, or model logic.

The game uses axial coordinates `(q, r)` on the hex grid, with three line axes:
Q `(1,0)`, R `(0,1)`, and QR `(1,-1)`. P0 opens at the origin; after that each
player places two stones per turn. A placement is legal if the cell is empty and
within 8 hex steps of an existing stone. Six or more stones in a row along one
axis wins, checked after every placement -- so a turn can end on its first stone.
There are no draws, passes, or captures, and stones are permanent.

## Public surface

The crate root re-exports consumer-facing types flat, so callers write
`use hexo_engine::{Position, Search, Action, Player};`.

```rust
use hexo_engine::{Action, HexCoord, Position};

let mut position = Position::new();
let applied = position.advance(Action::new(HexCoord::ORIGIN))?;
assert_eq!(applied.action.coord(), HexCoord::ORIGIN);
# Ok::<(), hexo_engine::MoveError>(())
```

### Coordinates and geometry (`coord`)

`HexCoord` is a cell on the board in axial coordinates `(q, r)`, with the
derived cube axis `s = -q - r`. `Axis` names the three straight-line directions
(`Q`, `R`, `QR`). `hex_distance` computes the distance in hex steps between any
two cells, using `i32` arithmetic to stay total over the full `i16` coordinate
range.

The coordinate domain is bounded by `COORD_LIMIT = 16_000`: a coordinate is
valid when `|q|`, `|r|`, and `|s|` all lie within this limit. This is a
representation bound, not a game rule.

The radius-8 disk offsets (`DISK8`, crate-private) are the 217 `(dq, dr)` pairs
within `LEGAL_RADIUS` of the origin, stored in `dq`-major, `dr`-minor order.
They define the neighbourhood used for legality and coverage.

### Players and turns (`player`)

`Player` is `P0` or `P1`. `TurnPhase` tracks where the mover is inside the
two-placement turn: `Opening` (ply 0 only, P0 at the origin), `FirstStone`, or
`SecondStone`. The turn pattern is P0; P1 P1; P0 P0; P1 P1; and so on. A win
freezes the mover and phase at whatever state they were in.

### Actions and record encoding (`action`)

`Action` wraps a `HexCoord` as a placement -- the atom of play. It carries no
legality claim; validation happens in `Position::advance` and `Search::apply`.

`ActionId` is the exactly invertible record encoding of a placement, packed as a
`u32`. The packing is order-preserving: unsigned `u32` comparison is exactly
signed lexicographic `(q, r)` comparison, achieved by XOR-biasing each `i16`
coordinate with `0x8000`. `ActionId`, `Action`, and `HexCoord` all share the
same `Ord` and convert freely among each other.

`ACTION_ORDER_VERSION` (currently 1) pins the canonical legal-move ordering.

### Windows and win detection (`window`)

A window is six consecutive cells along one axis, identified by `Window {
start, axis }`. Each placement touches 18 windows (3 axes times 6 offsets).

`WindowMask` holds two six-bit ownership masks (one per player) for a window's
cells. `WindowRef` pairs a `Window` with its `WindowMask`. `Win` describes a
maximal run of one player's stones along one axis, with `start`, `axis`, and
`len` (at least 6, at most 11).

Window masks are derived from the occupancy planes on read, in O(1), by
gathering the 11 cells along each axis through the queried coordinate. No window
state is stored.

Win detection is a separate per-axis run scan from the placed cell outward in
both directions. Six or more in a row wins; there is no overline rule. A
seven-in-a-row reports as one `Win` with `len == 7`, not two overlapping
windows. A placement can win on multiple axes simultaneously -- `Applied::wins`
is `[Option<Win>; 3]` indexed by `Axis::index()`.

### Position (`position`)

`Position` is the sole rule-bearing state type. It holds the board, turn phase,
current player, Zobrist hash accumulator, terminal status, and per-player stone
counts. It stores no placement history; the record-keeper owns the move list,
and `Position::replay` rebuilds state from it.

Construction and replay:
- `Position::new()` -- the empty board, P0 to move, no arena allocated.
- `Position::replay(actions)` -- rebuild from a placement sequence through the
  rule machine. Every replayed placement goes through `advance`, so every
  constructed position is reachable through legal play.
- `Position::replay_from(actions)` -- continue from an existing position.

The single mutation:
- `Position::advance(action)` -- apply one placement. Returns `Applied`
  describing what happened (mover, phase transition, win). Atomic on error: a
  rejected placement leaves the position unchanged.

Turn state: `current_player()`, `phase()`, `outcome()`, `is_terminal()`.

Occupancy: `get(coord)` returns the owner of a cell (total over every
`(i16, i16)` pair -- coordinates outside the arena read as empty). `stones()`
iterates all occupied cells with their owners in canonical `(q, r)` order.
`stone_count()` and `stone_count_for(player)` give counts.

Legal moves: `legal_actions()` yields legal placements in canonical ascending
`(q, r)` order, allocation-free. `legal_count()` gives the count (`0` when
terminal, `1` during `Opening`, otherwise the frontier size). `is_legal()` tests
one action. `legal_rank()` and `nth_legal()` convert between actions and their
canonical indices.

Windows: `windows_through(coord)` returns the 18 `WindowRef` values incident to
a cell. `window(w)` reads ownership of one specific window. Both are total over
the coordinate domain.

Integrity: `audit()` independently recomputes every derived structure from the
stones alone and compares. It is a normal public method, not gated behind a
cargo feature.

`PartialEq` is content-based and ignores arena geometry: two positions with the
same stones, phase, mover, and terminal status are equal. `Position` does not
implement `Hash`; use `zobrist()` as the map key.

### Search session (`search`)

`Search<'p>` is a borrow-scoped make/unmake session over a `&'p mut Position`.
It is the only path to undo.

- `Search::new(position)` -- begin a session. The position's current state
  becomes the undo floor.
- `apply(action)` -- apply one placement, recording how to reverse it.
- `undo()` -- reverse the most recent apply, restoring board, coverage,
  frontier, hash, phase, mover, and terminal status exactly. Returns `None` at
  the floor.
- `unwind()` -- undo every ply back to the floor.
- `commit()` -- move the floor to the current depth; applied plies become
  permanent and can no longer be undone.
- `depth()`, `at_floor()`, `path()` -- session state queries.

Dropping an uncommitted search unwinds to the floor, so a position lent to a
search is returned in its seeded state on every exit path, including `?` and
panic.

The undo stack is internal to `Search`. `Undo` tokens are `pub(crate)`,
unforgeable, non-cloneable, and consumed on use. The borrow prevents aliasing.

### Errors (`error`)

`MoveError` describes why a placement was rejected:
- `TerminalState` -- the game is over.
- `IllegalOpening` -- the opening must be at the origin.
- `CoordOutOfBounds` -- the coordinate is outside `COORD_LIMIT` but within
  `LEGAL_RADIUS` of a stone (a representation limit, not a rule violation).
- `Occupied` -- the cell already holds a stone.
- `TooFarFromStones` -- the cell is further than `LEGAL_RADIUS` from every
  stone.
- `BoardExtentExceeded` -- the arena would exceed `MAX_GRID_CELLS` (a
  representation limit, not a rule violation).

`is_rule_violation()` distinguishes rule violations from engine limits.

`ReplayError` wraps a `MoveError` with the ply index and action that failed.
`IntegrityError` and `IntegrityCheck` describe `audit()` failures.

### Zobrist hashing (`zobrist`)

The Zobrist hash uses a deterministic `splitmix64` mixing function over
explicitly packed keys. Cell keys encode `(q, r, player)` with a domain tag;
turn keys encode `(phase_kind, player, terminal)` into 12 compile-time
constants. All arithmetic is wrapping `u64` -- no float, no RNG, no endianness
dependence.

`hash_cells` accumulates only stone contributions by XOR. The turn key is
applied on read: `zobrist() = hash_cells ^ TURN_KEY[turn_slot()]`. This means
undo restores the turn contribution by restoring `phase` and `current` rather
than by un-XORing the turn key.

The hash is identical across builds, machines, and processes for a given
`RULES_VERSION`.

## Internal representation

The board is a dense recentered arena stored in `Grid` (crate-private). Three
bit planes -- two for occupancy (one per player) and one for coverage (cells
within `LEGAL_RADIUS` of any stone) -- use `q`-major, `r`-minor layout, with
rows of `u64` words.

Cell `(q, r)` maps to row `q - origin_q` and bit `r - origin_r`. All index
arithmetic is in `i32`. `origin_r` is always a multiple of 64, so row copies are
word-aligned.

The legal set is the **derived frontier**: `covered & !occ[0] & !occ[1]`,
composed on read and never stored. One `u32` maintains the frontier population
count. Because the layout is `q`-major, `r`-minor, a bit scan in storage order
produces canonical ascending `(q, r)` enumeration with no sort.

The placement pair (`place_stone` / `unplace_stone`) operates through disk runs
-- 17 contiguous row segments covering the radius-8 neighbourhood. Apply ORs the
coverage disk in word-wide runs. Undo recomputes coverage over the affected disk
using a separable dilation (three 1-D dilations by 9-cell segments via doubling
shifts), operating on a 33x33 window in `u64` words with no allocation.

The growth policy sizes the arena to the padded bounding box of all stones. The
first placement allocates a 32x128 arena. Growth is geometric with independent
per-dimension doubling, capped at 4x the content requirement and bounded by
`MAX_GRID_CELLS = 2^24` (approximately 6 MiB across the three planes). Undo does
not shrink the arena; the next growth reduces oversized dimensions. The arena
shape is a function of the stones, never of allocation history.

Memory scales with the bounding box, not with the number of stones.

## Constants

| Constant | Value | Meaning |
| --- | --- | --- |
| `RULES_VERSION` | 1 | Rule-machine and Zobrist function semantics |
| `ACTION_ORDER_VERSION` | 1 | Canonical legal-move indexing semantics |
| `MAX_GRID_CELLS` | 2^24 | Hard ceiling on dense arena cells |
| `LEGAL_RADIUS` | 8 | Max hex distance from any stone for a legal placement |
| `DISK_CELLS` | 217 | Cells in a radius-8 hex disk |
| `WINDOW_LEN` | 6 | Cells in a win window |
| `COORD_LIMIT` | 16,000 | Largest magnitude for any of `q`, `r`, `s` |
| `WINDOWS_PER_PLACEMENT` | 18 | Windows touched by one placement (3 axes x 6 offsets) |

## Connections

- `crates/hexo-runner` owns match orchestration around one canonical `Position`.
- `crates/hexo-search` consumes `Position`, `Search`, and canonical legal
  ordering.
- `crates/hexo-records` replays stored `ActionId` values through this crate.
- `crates/models/mantisnet` reads stones, windows, and legal actions for its
  encoder.
- `python/hexo-py` exposes a restricted PyO3 read and replay surface.

## Run / test

```sh
cargo test -p hexo-engine
cargo test -p hexo-engine --test golden
cargo test -p hexo-engine --test properties
cargo test -p hexo-engine --test boundary
cargo xtask verify        # all workspace gates
cargo xtask smoke         # extended randomized engine suite
cargo bench -p hexo-engine
```

## Source files

| File | Description |
| --- | --- |
| `lib.rs` | Crate root; module declarations, flat re-exports, and version constants |
| `coord.rs` | `HexCoord`, `Axis`, `hex_distance`, `DISK8`, and coordinate-domain constants |
| `player.rs` | `Player` and `TurnPhase` |
| `action.rs` | `Action`, `ActionId`, and the canonical ordering version |
| `window.rs` | `Window`, `WindowMask`, `WindowRef`, `Win`, and window geometry |
| `position.rs` | `Position`, `Applied`, `Outcome`, `Stones`, `LegalActions`; the rule machine (`advance`, `apply_raw`, `undo_raw`), all read accessors, and `audit` |
| `position_tests.rs` | Unit tests for `position.rs` |
| `search.rs` | `Search<'p>` (borrow-scoped make/unmake session) and the private `Undo` / `UndoAudit` tokens |
| `error.rs` | `MoveError`, `ReplayError`, `IntegrityError`, `IntegrityCheck` |
| `grid.rs` | Crate-private dense recentered arena: two occupancy planes, one coverage plane, the derived frontier, growth policy, and the placement pair |
| `zobrist.rs` | Crate-private deterministic mixing function (`splitmix64`) and the compile-time turn-key table |
