"""Run artifact helpers for self-play training.

Artifacts are the durable files that describe or hand off a run: manifests,
placeholder checkpoints, self-play checkpoint pointers, and final summaries.
This module keeps file-writing concerns out of `defaults.py` and `pipeline.py`
so the pipeline can stay focused on ordering.

External file-format contract (no Python import): `manifest.json` written by
`write_run_manifest` is read by the dashboard
(packages/hexo_frontend/python/hexo_frontend/web.py for run lineage/config and
debug_infer.py for architecture detection). Changing its shape changes what
the :8080 dashboard sees for every run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import json

from .components import TrainingComponents
from .context import RunContext


@dataclass(frozen=True, slots=True)
class CheckpointStore:
    """Run-local checkpoint path and placeholder metadata helper.

    Real model packages can override checkpoint writing through their plugin.
    Until then, this tiny store gives the pipeline something deterministic to
    write and test without pretending to serialize model weights. All four
    registered plugins provide a real saver, so `write_placeholder` is reached
    only by FakePlugin tests (tests/test_training_pipeline_simplification.py);
    `path_for` remains generally useful.
    """

    checkpoint_dir: Path

    def path_for(self, name: str) -> Path:
        """Return the path for a named checkpoint in the run checkpoint dir."""

        return self.checkpoint_dir / f"{name}.ckpt"

    def write_placeholder(self, name: str, metadata: Mapping[str, Any]) -> Path:
        """Write a tiny metadata file until model checkpoint IO is implemented."""

        path = self.path_for(name)
        path.write_text(json.dumps(dict(metadata), indent=2), encoding="utf-8")
        return path


def write_run_manifest(ctx: RunContext) -> Path:
    """Write normalized run metadata before long-running epoch work starts.

    The manifest is a quick index for humans and tools: it says which model,
    loop settings, sample settings, and checkpoint settings produced this
    output directory. The dashboard (packages/hexo_frontend/python/
    hexo_frontend/web.py and debug_infer.py) parses `model` for lineage and
    architecture detection — keep the keys stable.
    """

    manifest = {
        "run": asdict(ctx.config.run),
        "model": asdict(ctx.config.model),
        "loop": asdict(ctx.config.loop),
        "selfplay": asdict(ctx.config.selfplay),
        "samples": asdict(ctx.config.samples),
        "train": asdict(ctx.config.train),
        "checkpoint": asdict(ctx.config.checkpoint),
        "output_dir": str(ctx.output_dir),
    }
    path = ctx.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def publish_selfplay_checkpoint_pointer(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Optionally publish the final checkpoint for future self-play workers.

    The pointer is deliberately a small text file. It decouples future self-play
    processes from the exact checkpoint naming scheme inside the run directory.

    Near-duplicate of checkpoints._publish_epoch_checkpoint_pointer (the
    per-epoch variant); keep the two writers format-identical. Only legacy
    dense_cnn/hexgt configs enable the pointer — all restnet main_* configs
    set `update_checkpoint_pointer = false`.
    """

    _ = components
    if not ctx.config.selfplay.update_checkpoint_pointer:
        return {"status": "skipped", "reason": "selfplay pointer update disabled"}

    checkpoint_result = ctx.outputs.get("final_checkpoint", {})
    checkpoint_path = checkpoint_result.get("checkpoint_path")
    pointer_path = ctx.config.selfplay.checkpoint_pointer
    if pointer_path is None:
        pointer_path = ctx.output_dir / "selfplay_checkpoint.txt"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(str(checkpoint_path or ""), encoding="utf-8")
    return {"status": "updated", "pointer_path": str(pointer_path)}


def write_final_diagnostics(
    ctx: RunContext,
    components: TrainingComponents,
) -> dict[str, Any]:
    """Write a final human-readable summary of the training run.

    This does not replace per-step diagnostics. It gathers the top-level facts a
    person usually wants after a run: directories, epoch count, and model plugin
    extras.
    """

    summary = {
        "run": ctx.config.run.name,
        "model": ctx.config.model.name,
        "output_dir": str(ctx.output_dir),
        "checkpoint_dir": str(ctx.checkpoint_dir),
        "samples_dir": str(ctx.samples_dir),
        "epochs": len(ctx.epoch_outputs),
        "outputs": list(ctx.outputs),
        "model_extra": dict(components.model.extra),
    }
    ctx.diagnostics.write_json("run.completed.json", summary)
    return summary
