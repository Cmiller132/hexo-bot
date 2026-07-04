"""Rust-backed Hexo runner record contracts.

Re-exports the .hxr binary codec from hexo_utils.records (implemented in
packages/hexo_utils/rust/src/records.rs + pybridge.rs). This re-export path
— normally reached via the hexo_runner.records facade — is how ALL production
.hxr IO flows: every model package's selfplay/evaluation writers and the
hexo_frontend dashboard reader import from here rather than hexo_utils.
"""

from __future__ import annotations

from dataclasses import dataclass

from hexo_utils.records import (
    HEXO_RECORD_MAGIC,
    HEXO_RECORD_SCHEMA_VERSION,
    HexoRecord,
    HexoRecordFile,
    HexoRecordGameWriter,
    HexoRecordPlayer,
)


@dataclass(frozen=True, slots=True)
class AbortRecord:
    """Abort information for runner control-plane summaries.

    NOTE: hexo_utils._rust also defines a distinct PyO3 class named
    AbortRecord with the same field shape. The Rust writer's finish_aborted
    accepts THIS dataclass only because pybridge.rs parse_abort duck-types
    the .stage/.exception_type/.message attributes — keep those names stable.
    """

    stage: str
    exception_type: str
    message: str


__all__ = [
    "AbortRecord",
    "HEXO_RECORD_MAGIC",
    "HEXO_RECORD_SCHEMA_VERSION",
    "HexoRecord",
    "HexoRecordFile",
    "HexoRecordGameWriter",
    "HexoRecordPlayer",
]
