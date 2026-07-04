# hexo_engine

Authoritative rules engine for Hexo, a Gomoku-like 6-in-a-line game on an
unbounded hexagonal grid. Owns board occupancy, the turn/phase machine
(opening stone at origin, then two-stone turns), move legality (radius-8 from
any stone), incremental 6-cell win/threat windows, and the stable packed
action-ID encoding.

**Load-bearing.** The model (`hexfield`), the runner, the trainer, the frontend
dashboard, and most of the test suite drive it. Repo convention: Python callers
come through this package rather than re-implementing game logic.

## Shape

A Rust crate plus a thin typed Python wrapper, distributed via maturin:

- The crate builds as an **rlib** consumed directly by the `hexo_utils` and
  `hexfield` Rust crates (Cargo workspace members; see root `Cargo.toml`).
- With the `python` feature it also builds a **PyO3 extension module**
  `hexo_engine._rust` (`pyproject.toml`: `module-name = "hexo_engine._rust"`,
  `python-source = "python"`).
- The Python package in `python/hexo_engine/` is a thin facade over `_rust`.

The extension (`_rust.cpython-3XX-*.so`, untracked) is built by maturin
(`maturin develop --release`, via `scripts/build_native.sh`); Rust changes take
effect after that rebuild. The bridge is Linux/WSL in practice: importing under
Windows-native Python raises `EngineUnavailableError` lazily.

## Module table

### Rust (`rust/src/`)

| File | Role |
| --- | --- |
| `state.rs` | Heart of the engine: `HexoState`, `TurnPhase` machine (Opening/FirstStone/SecondStone), `apply_placement` / `apply_with_delta` + undo, `snapshot()`/`load_state` replay, large invariant test suite. |
| `board.rs` | Sparse stone storage (AHashMap + insertion-ordered occupied list) with delta/undo; owns the `LegalMoveStore` and `WindowStore`. |
| `tactics.rs` | Incremental 6-cell window tracking (3 axes x 6 offsets = 18 windows per placement), threat (>=4 single-colour) and win detection, plus an O(1) `live_threats` index for the TSS hot path in model crates. |
| `legal.rs` | Incremental legal-move store (`LEGAL_RADIUS = 8` around any stone) and the canonical `pack_coord`/`unpack_coord` u32 action-ID encoding: `((q + 2^15) << 16) \| (r + 2^15)`. ID order is the deterministic legal-action order everywhere. |
| `rules.rs` | `is_legal_placement`: phase validation (origin-only opening, no first-stone reuse, occupancy, radius store). |
| `coord.rs` | Axial `HexCoord` (i16 q,r), `hex_distance`, `coords_within_radius`; re-exported to all downstream crates. |
| `snapshot.rs` | `StateSnapshot` (rules_version = 1 + placement list) for replay via `load_state`; currently used internally (engine_metadata + tests) and reserved for future replay consumers. |
| `error.rs` | `MoveError` / `StateLoadError`; `MoveError` surfaces to Python as `IllegalActionError`. |
| `pybridge.rs` | PyO3 module `hexo_engine._rust`: opaque `PyHexoState` handle plus `new_game`, `clone_state`, `current_player`, `legal_action_ids`, `legal_action_count`, `is_legal_action`, `apply_action`, `terminal`, `to_python_state`, `action_id`, `engine_metadata`, and the C-ABI `state_api_capsule` (version 2: clone_state/free_state fn pointers). |

### Python (`python/hexo_engine/`)

| File | Role |
| --- | --- |
| `__init__.py` | Public surface; re-exports `api`, `errors`, `types`. Consumers do `import hexo_engine`. |
| `api.py` | Thin typed wrappers over the `_rust` functions; converts dict payloads into frozen dataclasses (`TransitionResult`, `TerminalResult`, `PythonHexoState` mirror); raises `EngineUnavailableError` if the extension is missing. |
| `types.py` | Transport types: `Player`/`TurnPhase` StrEnums, `AxialCoord`, `PlacementAction`, lazy `LegalActions` view, `pack_coord_id`/`unpack_coord_id` (must mirror Rust `legal.rs` packing), read-only `Python*` state mirrors. |
| `errors.py` | `HexoEngineError` base, `EngineUnavailableError`, `IllegalActionError`. |

## Design notes

- **Two access tiers by call volume.** Hot paths (selfplay, MCTS) consume raw
  `legal_action_ids` tuples and the C-ABI capsule and never touch the mirror
  layer. `to_python_state()` materializes the full state as Python dicts --
  every stone, legal coord, and window entry (O(18 x placements) window
  entries) -- and is sized for dashboard/replay call volume.
- **Deterministic setup.** `new_game(seed=None, scenario=None)` accepts both
  parameters for API-shape stability across player adapters; the engine has no
  randomness and no scenario loader, so every game starts identically (empty
  board, origin opening).

## Connections to other packages

**Rust rlib consumers** (`hexo_engine.workspace = true` in their Cargo.toml):

- `packages/hexfield` -- uses `HexoState`, `apply_with_delta`+undo,
  `Board::occupied_cells`, and `WindowStore` threat queries for search,
  candidate generation, and tensor featurization.
- `packages/hexo_utils` -- `state_hash.rs` (eval-cache keys) and record replay
  tests.

**C-ABI capsule protocol:** the model accelerator crate does
`py.import("hexo_engine._rust").state_api_capsule()` at batch-MCTS time to
clone live Python `HexoState` objects into owned Rust states
(`STATE_API_VERSION = 2`, `pybridge.rs`). A version mismatch fails loudly at
use time.

**Python consumers:** `hexo_runner` (engine adapter + match loop),
`hexo_frontend` (`web.py`, `debug_infer.py`, `dashboard.py` replay stored
action-ID sequences and render boards from `to_python_state()`), and `hexfield`
(selfplay / evaluation / player / search glue).

**Shared/persisted contracts:**

- The packed action-ID encoding is implemented **twice on purpose** -- Rust
  `legal.rs` and Python `types.py` -- and the frontend JS re-implements the
  same packing (offset 32768), because the IDs are persisted in training
  shards (.npz), `.hxr` game records, and frontend deep links. All
  implementations are required to produce identical IDs;
  `tests/test_hexo_engine_rust_bridge.py` cross-checks `pack_coord_id` against
  `engine.action_id`.
- `engine_metadata()` (`backend=rust-pyo3`, `rules_version=1`,
  `state_api_version=2`) is embedded in runner game records and asserted by
  the bridge test.

## Entry points / how it is exercised

There is no CLI. The runtime entry is the maturin-built extension:

- `import hexo_engine` from any Python consumer.
- `hexo_engine = { workspace = true }` rlib dependency from `hexfield` /
  `hexo_utils`.
- `state_api_capsule()` FFI entry from the `hexfield` accelerator crate.
- Tests: `tests/test_hexo_engine_rust_bridge.py` (bridge contract; skips when
  the .so is unavailable, e.g. Windows-native Python) and `cargo test -p
  hexo_engine` for the Rust invariant suites. Indirectly, nearly every test
  under `tests/` touches it.
- Build: `maturin develop --release` via `scripts/build_native.sh` (see Shape above)
  rebuilds the workspace and refreshes the extension.
