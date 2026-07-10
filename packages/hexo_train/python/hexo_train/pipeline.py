"""Top-level self-play training pipeline.

This file is intentionally the "map" of a training run. It does not know how
to decode model tensors, run a loss, generate a legal action, or serialize a
real checkpoint. It coordinates the packages that do own those details.

`hexo_train` currently owns one supported training path:

1. initialize run artifacts and shared stores;
2. load or initialize a checkpoint;
3. calibrate model-owned performance settings when available;
4. run self-play training epochs;
5. publish the final checkpoint for future self-play;
6. write diagnostics.

Each top-level step is wrapped by `_run_step()` so failures and successful
results are recorded consistently in the diagnostics directory.

Model packages own tensors, losses, optimizer details, sample decoding, and
checkpoint contents. Game execution happens inside each plugin's
`generate_selfplay` hook (the plugin drives hexo_engine/hexo_runner itself —
e.g. packages/hexfield/python/hexfield/selfplay.py for the active run); this
pipeline never imports the engine or runner directly.

Entry point: cli/train_model.py (`python -m hexo_train.cli.train_model
<config.toml>`), which the WSL supervisor scripts
(scripts/_hexfield_supervise_main1.sh, scripts/_hexfield_eq_supervise_main1.sh)
loop over for the live runs. Tests also call `TrainingPipeline().run()`
directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import (
    publish_selfplay_checkpoint_pointer,
    write_final_diagnostics,
    write_run_manifest,
)
from .checkpoints import load_or_initialize_checkpoint, save_final_checkpoint
from .components import TrainingComponents, build_model_components
from .config import load_training_config
from .context import RunContext
from .defaults import build_shared_components
from .epoch import run_epochs
from .epoch.samples import prepare_sample_store
from .registry import load_model_plugin


PipelineStep = Callable[[RunContext, TrainingComponents], Any]


class TrainingPipeline:
    """Public orchestrator for one config-driven self-play training run."""

    def run(self, config_path: str | Path) -> RunContext:
        """Run training and return the mutable run context.

        Step by step:

        1. Normalize the user config into typed config sections.
        2. Create run directories and the diagnostics writer.
        3. Load the model plugin and let it build model-owned components.
        4. Execute the fixed self-play lifecycle.

        The returned `RunContext` is useful for tests and callers that want to
        inspect artifact paths, top-level outputs, or per-epoch results.
        """

        config = load_training_config(config_path)
        ctx = RunContext.from_config(config)
        plugin = load_model_plugin(config.model)
        shared = build_shared_components(ctx)
        model = build_model_components(plugin=plugin, ctx=ctx, shared=shared)
        components = TrainingComponents(shared=shared, model=model)

        try:
            self._run_step("initialize_run", self._initialize_run, ctx, components)
            self._run_step(
                "load_checkpoint",
                load_or_initialize_checkpoint,
                ctx,
                components,
            )
            self._run_step("calibrate_performance", self._calibrate_performance, ctx, components)
            self._run_step("run_epochs", run_epochs, ctx, components)
            self._run_step("publish_final_model", self._publish_final_model, ctx, components)
            self._run_step("write_diagnostics", write_final_diagnostics, ctx, components)
            return ctx
        finally:
            self._teardown_components(components)

    def _teardown_components(self, components: TrainingComponents) -> None:
        """Release model-owned resources at run end (success or failure).

        Generic: a model trainer that holds external resources (e.g. the
        hexfield trainer's persistent shard-expansion ProcessPool built by
        ``_get_expand_pool``) may expose ``close()``; calling it in a finally
        keeps those resources from leaking past the run. Best-effort.
        """

        trainer = getattr(getattr(components, "model", None), "trainer", None)
        close = getattr(trainer, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - teardown must never mask the real result
                pass

    def _initialize_run(
        self,
        ctx: RunContext,
        components: TrainingComponents,
    ) -> Mapping[str, Any]:
        """Write initial run files and open the sample store.

        This is the only setup step that touches run metadata. It writes the
        normalized config and manifest before epochs start, then opens the
        shared sample store so self-play/finalization can append model-owned
        records through the same handle.
        """

        normalized_config = ctx.diagnostics.write_json(
            "config.normalized.json",
            ctx.config,
        )
        manifest = write_run_manifest(ctx)
        if components.model.uses_shared_sample_store:
            sample_store = prepare_sample_store(ctx, components)
        else:
            sample_store = {
                "status": "skipped",
                "reason": "model owns replay storage",
            }
        return {
            "config": str(normalized_config),
            "manifest": str(manifest),
            "sample_store": sample_store,
        }

    def _publish_final_model(
        self,
        ctx: RunContext,
        components: TrainingComponents,
    ) -> Mapping[str, Any]:
        """Save the final model checkpoint and publish the optional pointer.

        Epoch checkpoints feed the next epoch during this process. The final
        checkpoint is the stable output of the whole run, and the pointer is
        the small handoff file future self-play workers can read.
        """

        checkpoint = save_final_checkpoint(ctx, components)
        ctx.remember("final_checkpoint", checkpoint)
        pointer = publish_selfplay_checkpoint_pointer(ctx, components)
        return {"checkpoint": checkpoint, "pointer": pointer}

    def _calibrate_performance(
        self,
        ctx: RunContext,
        components: TrainingComponents,
    ) -> Mapping[str, Any]:
        """Run optional model-owned performance calibration before epochs."""

        plugin = components.model.plugin
        if hasattr(plugin, "calibrate_performance"):
            return plugin.calibrate_performance(ctx=ctx, components=components)
        return {
            "status": "skipped",
            "reason": "model plugin has no calibrate_performance hook",
        }

    def _run_step(
        self,
        step_name: str,
        step: PipelineStep,
        ctx: RunContext,
        components: TrainingComponents,
    ) -> Any:
        """Run one top-level step with consistent diagnostics.

        The helper records a start event, delegates to the supplied step, stores
        the result on `ctx.outputs`, and writes a JSON summary. If a step raises,
        the failure is recorded before the exception is re-raised.
        """

        diagnostics = ctx.diagnostics
        started_at = diagnostics.start_stage(step_name)
        try:
            result = step(ctx, components)
        except Exception as exc:
            diagnostics.finish_stage(
                stage=step_name,
                started_at=started_at,
                status="failed",
                metadata={"error": repr(exc)},
            )
            raise

        ctx.remember(step_name, result)
        diagnostics.finish_stage(
            stage=step_name,
            started_at=started_at,
            status=self._status_for(result),
            metadata={"result": self._result_metadata(result)},
        )
        return result

    def _status_for(self, result: Any) -> str:
        """Translate a step result payload into a diagnostic status."""

        if isinstance(result, Mapping) and result.get("status") == "skipped":
            return "skipped"
        return "completed"

    def _result_metadata(self, result: Any) -> Mapping[str, Any]:
        """Convert common result objects into compact diagnostic metadata."""

        if result is None:
            return {}
        if isinstance(result, Mapping):
            return result
        if isinstance(result, Path):
            return {"path": str(result)}
        return {"type": type(result).__name__, "repr": repr(result)}
