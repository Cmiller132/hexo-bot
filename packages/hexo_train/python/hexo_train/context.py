"""Run-scoped state for one training invocation.

`RunContext` is deliberately separate from model components. It owns facts
about this run: config, directories, diagnostics, epoch outputs, and shared
artifact locations. Model packages receive the context, but they should not
turn it into a home for tensor semantics or model-specific training logic.

Think of this as the notebook for the run. It records where files live and what
each top-level pipeline step or epoch produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .config import TrainingConfig
from .diagnostics import DiagnosticsWriter


@dataclass(slots=True)
class RunContext:
    """Mutable state shared by the self-play training loop.

    The context is intentionally small: durable paths, the normalized config,
    diagnostics, top-level outputs, and per-epoch outputs. Model-specific state
    belongs in `TrainingComponents.model`.
    """

    config: TrainingConfig
    output_dir: Path
    checkpoint_dir: Path
    diagnostics_dir: Path
    samples_dir: Path
    diagnostics: DiagnosticsWriter
    outputs: dict[str, Any] = field(default_factory=dict)
    epoch_outputs: list[Any] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: TrainingConfig) -> "RunContext":
        """Create run directories and diagnostics from normalized config.

        Directory creation happens once here so the rest of the pipeline can
        assume output, checkpoint, diagnostics, and sample directories exist.
        """

        output_dir = config.run.output_dir
        checkpoint_dir = output_dir / "checkpoints"
        diagnostics_dir = output_dir / "diagnostics"
        samples_dir = output_dir / "samples"
        for directory in (output_dir, checkpoint_dir, diagnostics_dir, samples_dir):
            directory.mkdir(parents=True, exist_ok=True)

        return cls(
            config=config,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            diagnostics_dir=diagnostics_dir,
            samples_dir=samples_dir,
            diagnostics=DiagnosticsWriter(diagnostics_dir),
        )

    def section(self, name: str) -> Mapping[str, Any]:
        """Return a flexible raw top-level config section.

        Typed config fields should be preferred. This helper remains for
        sections that are intentionally still open-ended, such as shared game
        specs or sample-store implementation details.
        """

        value = self.config.raw.get(name, {})
        if isinstance(value, Mapping):
            return value
        return {}

    def remember(self, name: str, result: Any) -> Any:
        """Store a named top-level run result for later steps and diagnostics."""

        self.outputs[name] = result
        return result

    def remember_epoch(self, result: Any) -> Any:
        """Store one epoch result in chronological order."""

        self.epoch_outputs.append(result)
        return result
