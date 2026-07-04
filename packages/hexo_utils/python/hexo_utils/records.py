"""Python exposure for the Rust-backed `.hxr` record codec.

Pure re-export module: the classes live in `hexo_utils._rust` (compiled from
`packages/hexo_utils/rust/src/records.rs` via `rust/src/pybridge.rs`).
Importing this module raises ImportError when the maturin-built extension is
absent (e.g. Windows Python; only the WSL venv carries the .so).

Callers: production `.hxr` IO flows through
`packages/hexo_runner/python/hexo_runner/records/record.py`, which wraps and
re-exports this module for all model selfplay/evaluation writers and the
hexo_frontend dashboard reader. `scripts/_wf_r4_health.py` and
`analysis/exploration_diversity.py` import `HexoRecordFile` directly for run
audits.

Note: `AbortRecord` here is the PyO3 class; `hexo_runner.records.record`
defines a same-named, same-shaped plain dataclass. The Rust side duck-types
`.stage`/`.exception_type`/`.message`, so either works in `finish_aborted`.
"""

from __future__ import annotations

from ._rust import (
    AbortRecord,
    HEXO_RECORD_MAGIC,
    HEXO_RECORD_SCHEMA_VERSION,
    HexoRecord,
    HexoRecordFile,
    HexoRecordGameWriter,
    HexoRecordPlayer,
)

__all__ = [
    "AbortRecord",
    "HEXO_RECORD_MAGIC",
    "HEXO_RECORD_SCHEMA_VERSION",
    "HexoRecord",
    "HexoRecordFile",
    "HexoRecordGameWriter",
    "HexoRecordPlayer",
]
