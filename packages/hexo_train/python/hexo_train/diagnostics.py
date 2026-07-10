"""Diagnostics and run-output helpers for training orchestration.

Diagnostics are intentionally simple files in the run directory:

- `events.jsonl` records an append-only stream of step lifecycle events.
- `<step>.json` records a structured summary for each top-level step or epoch.
- other pipeline files can write additional JSON payloads through `write_json`.

The writer also converts dataclasses and paths into JSON-friendly shapes so
callers can pass normal pipeline objects without manual serialization code.

External readers (file-format contract, no Python import): the dashboard's
live training status (packages/hexo_frontend/python/hexo_frontend/web.py)
tails `events.jsonl`, and model plugins route their own diagnostics — e.g.
hexfield.selfplay.epoch_*.json from the hexfield plugin — through
`write_json` into the same directory, where the dashboard history views read
them. Keep event/file naming stable for those consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json
import time


@dataclass(slots=True)
class StageDiagnostic:
    """Small serializable record for one pipeline step or epoch."""

    stage: str
    status: str
    elapsed_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DiagnosticsWriter:
    """Writes step and run diagnostics to the run output directory."""

    def __init__(self, diagnostics_dir: Path) -> None:
        """Create a writer rooted at one run's diagnostics directory."""

        self.diagnostics_dir = diagnostics_dir
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    def start_stage(self, stage: str) -> float:
        """Record that a step started and return a timer token."""

        self.write_event("stage_started", {"stage": stage})
        return time.perf_counter()

    def finish_stage(
        self,
        *,
        stage: str,
        started_at: float,
        status: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> StageDiagnostic:
        """Record that a step finished and write its JSON summary."""

        diagnostic = StageDiagnostic(
            stage=stage,
            status=status,
            elapsed_seconds=time.perf_counter() - started_at,
            metadata=dict(metadata or {}),
        )
        self.write_json(f"{stage}.json", diagnostic)
        self.write_event("stage_finished", diagnostic)
        return diagnostic

    def write_event(self, name: str, payload: Any) -> None:
        """Append one JSON-lines event to the run event stream."""

        event_path = self.diagnostics_dir / "events.jsonl"
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": name, "payload": _jsonable(payload)}))
            handle.write("\n")

    def write_json(self, name: str, payload: Any) -> Path:
        """Write one pretty JSON payload under the diagnostics directory."""

        path = self.diagnostics_dir / name
        path.write_text(
            json.dumps(_jsonable(payload), indent=2, default=str),
            encoding="utf-8",
        )
        return path


def _jsonable(value: Any) -> Any:
    """Convert common pipeline objects into JSON-compatible values."""

    if hasattr(value, "__dataclass_fields__"):
        return {
            field_name: _jsonable(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
