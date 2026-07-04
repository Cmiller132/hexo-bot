# hexo_utils

Shared-utility layer between the engine, runner, trainer, and model packages.
A Rust crate plus a thin Python package, distributed via maturin as `hexo_utils`
with a private PyO3 extension `hexo_utils._rust` (built with `features=["python"]`,
see `pyproject.toml`).

## Subsystems

| Subsystem | Role |
| --- | --- |
| `.hxr` record codec (`rust/src/records.rs` + `pybridge.rs` + `python/hexo_utils/records.py`) | The repo's cross-package game-record format. Every self-play / evaluation / match path writes `.hxr` through it; the dashboard reads it. |
| `state_hash` (`rust/src/state_hash.rs`) | Rust-only. The cache key for the `hexfield` search evaluator (deterministic, placement-order-sensitive position identity). |
| `encoding/` D6 contracts | The D6 symmetry transport contract (`D6_SIZE`, `D6Symmetry`, `ActionSymmetryMapper`, `transform_action_ids`), consumed by `hexo_train.symmetry` for training-time augmentation. |

## The .hxr record format

Owned and specified here (codec core: `rust/src/records.rs`). Wire format, schema v1:
magic `HEXOREC1`, a varint-encoded header (engine metadata, players), then varint/zigzag
per-game payloads (game_id, seed, status, action ids as u32 LE, winner, placements,
optional abort record). Readers reject other schema versions; `HEXO_RECORD_SCHEMA_VERSION`
is bumped on any wire change. API: `HexoRecordFile` reader/writer (`iter_records()` on a
write-mode handle reopens the path for reading) and `HexoRecordGameWriter`, an append-only
per-game writer.

## Module table

| File | Role |
| --- | --- |
| `rust/src/records.rs` | `.hxr` binary codec core (~1000 lines): the wire format above, plus round-trip/corruption unit tests. |
| `rust/src/state_hash.rs` | `hash_state(HexoState) -> u64`: deterministic, placement-order-sensitive state identity (splitmix64-style mixing over placement history + player/phase/terminal) for neural-eval caches. |
| `rust/src/pybridge.rs` | PyO3 bridge behind the `python` feature: `PyHexoRecordFile` / `PyHexoRecordGameWriter` / `PyHexoRecord` / `PyAbortRecord` / `PyHexoRecordPlayer`; duck-typed parsers (players via `.identity`, action ids from int or `.coord.q/.r`); `PyHexoRecord.replay()` re-runs action ids through the `hexo_engine` Python module. Defines the `_rust` pymodule. |
| `rust/src/lib.rs` | Crate root: re-exports records + state_hash; pybridge gated behind the `python` feature. |
| `python/hexo_utils/records.py` | Python facade re-exporting the codec classes from `hexo_utils._rust`. Production callers reach it through `hexo_runner.records`, which wraps this module. |
| `python/hexo_utils/encoding/symmetry.py` | D6 symmetry transport contract: `D6_SIZE=12`, frozen `D6Symmetry`, `ActionSymmetryMapper` Protocol, `transform_action_ids`. Consumed by `hexo_train.symmetry`. |
| `Cargo.toml` / `pyproject.toml` | Workspace crate (rlib + cdylib, depends on `hexo_engine`, pyo3 optional) / maturin config (`module-name = "hexo_utils._rust"`, `python-source = "python"`). |

## Connections to other packages

Inbound (who uses hexo_utils):

- `hexo_runner.records.record` imports `HexoRecordFile` / `HexoRecordGameWriter` / `HexoRecord` / `HexoRecordPlayer` / `AbortRecord` / magic + schema constants from `hexo_utils.records` and re-exports them. All production `.hxr` IO flows through that path: `hexfield` selfplay/evaluation writes records; `hexo_frontend/web.py` reads them for the dashboard.
- Rust: the `hexfield` crate depends on the `hexo_utils` workspace crate for `use hexo_utils::{hash_state, StateHash}` (evaluator cache keys).
- `hexo_train`: `symmetry.py` imports `D6_SIZE` / `D6Symmetry` from `hexo_utils.encoding`.

Outbound (what hexo_utils depends on):

- Rust crate depends on the `hexo_engine` crate (`HexoState`, `HexCoord`, `pack_coord`, `Player`, `TurnPhase`, outcome types) for both `state_hash` and record replay tests.
- `pybridge.rs` `PyHexoRecord.replay()` and the duck-typed parsers form a runtime contract with the `hexo_engine` Python package (`new_game` / `PlacementAction` / `apply_action` / `terminal`, `unpack_coord_id`) and with `hexo_runner` player objects (`.identity.player_id` / `.label`) and AbortRecord-shaped objects (`.stage` / `.exception_type` / `.message`).

Formats owned here: the `.hxr` game-record contract (specified above).

## Entry points / how it gets exercised

- No CLI. Pure library, imported transitively by nearly every Python entry point in the repo via `hexo_runner.records`.
- Build: the PyO3 extension is built into the source tree (`python/hexo_utils/_rust.cpython-3XX-*.so`, untracked) by maturin via `scripts/build_native.sh`; Rust changes take effect after re-running it. On Windows-native Python the extension is absent; importers lazy-guard the import.
- Tests: many tests gate on `pytest.importorskip("hexo_utils._rust")`; `cargo test -p hexo_utils` runs the Rust codec/hash unit suites. Run Python tests from the repo root with the venv active.
