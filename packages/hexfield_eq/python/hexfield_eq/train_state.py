"""Persisted trainer state for the replay-buffer train-bucket reuse governor and
window bookkeeping.

Serialized into the checkpoint ``meta`` and restored on resume; an
``initialize_from`` warm start begins with fresh state. Carries no shuffle-output
bookkeeping (the window is shuffled in RAM).

A missing-key or version mismatch loads a fresh state, so checkpoints without a
``train_state`` entry resume without raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# Bump when the persisted schema changes incompatibly; a mismatch loads fresh.
TRAIN_STATE_VERSION = 1


@dataclass
class HexfieldTrainState:
    # Cumulative count of self-play rows seen by the governor; monotonically
    # non-decreasing. Drives bucket accrual (window selection uses the live
    # manifest total instead).
    total_num_data_rows: int = 0
    # Cumulative gradient samples consumed.
    global_step_samples: int = 0
    # First global row index still inside the current window.
    window_start_data_row_idx: int = 0
    # Train-bucket reuse governor. ``level`` is credited by each new self-play row
    # times max_train_bucket_per_new_data and debited by effective_rows at
    # selection time; ``level_at_row`` is the cumulative-row watermark the last
    # accrual was computed against.
    train_bucket_level: float = 0.0
    train_bucket_level_at_row: int = 0
    train_steps_since_last_reload: int = 0
    # Rollback-detection watermarks for the governor. ``last_seen_epoch`` is the
    # highest epoch index a governor accrual has run against; ``last_seen_live_rows``
    # is the live (present, non-monotone) manifest row count at that accrual. When
    # the run regresses below either watermark (resume from an earlier checkpoint,
    # or epoch quarantine drops present rows), ``_update_train_bucket`` rebases the
    # monotone ``train_bucket_level_at_row`` down so crediting resumes instead of
    # freezing. ``-1`` means "never accrued", which never triggers a regression.
    last_seen_epoch: int = -1
    last_seen_live_rows: int = 0
    # Optional no-repeat-files set; empty unless the no-repeat-files option is
    # enabled.
    data_files_used: set[str] = field(default_factory=set)
    version: int = TRAIN_STATE_VERSION

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "HexfieldTrainState":
        """Build from a persisted mapping. ``None``, a non-mapping, or a version
        mismatch returns a fresh state."""
        if not isinstance(raw, Mapping):
            return cls()
        if int(raw.get("version", 0)) != TRAIN_STATE_VERSION:
            return cls()
        return cls(
            total_num_data_rows=int(raw.get("total_num_data_rows", 0)),
            global_step_samples=int(raw.get("global_step_samples", 0)),
            window_start_data_row_idx=int(raw.get("window_start_data_row_idx", 0)),
            train_bucket_level=float(raw.get("train_bucket_level", 0.0)),
            train_bucket_level_at_row=int(raw.get("train_bucket_level_at_row", 0)),
            train_steps_since_last_reload=int(raw.get("train_steps_since_last_reload", 0)),
            last_seen_epoch=int(raw.get("last_seen_epoch", -1)),
            last_seen_live_rows=int(raw.get("last_seen_live_rows", 0)),
            data_files_used=set(str(item) for item in raw.get("data_files_used", ()) or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "total_num_data_rows": int(self.total_num_data_rows),
            "global_step_samples": int(self.global_step_samples),
            "window_start_data_row_idx": int(self.window_start_data_row_idx),
            "train_bucket_level": float(self.train_bucket_level),
            "train_bucket_level_at_row": int(self.train_bucket_level_at_row),
            "train_steps_since_last_reload": int(self.train_steps_since_last_reload),
            "last_seen_epoch": int(self.last_seen_epoch),
            "last_seen_live_rows": int(self.last_seen_live_rows),
            "data_files_used": sorted(self.data_files_used),
        }
