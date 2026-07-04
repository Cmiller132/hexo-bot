"""Small timing helpers for runner records.

Used by `hexo_runner/loop.py` and `hexo_runner/modes/batch.py` to populate
the `duration_ms` fields on GameResult/BatchResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True, slots=True)
class Timer:
    """Monotonic wall-clock stopwatch; `elapsed_ms` returns milliseconds."""

    started: float

    @classmethod
    def start(cls) -> "Timer":
        return cls(started=perf_counter())

    def elapsed_ms(self) -> float:
        return round((perf_counter() - self.started) * 1000.0, 3)
