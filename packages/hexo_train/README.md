# hexo_train

Model-neutral, config-driven training orchestration for Hexo models. Given a
TOML config, it discovers a model plugin (by explicit module path), then runs a
fixed self-play training lifecycle: initialize run artifacts -> load/initialize
checkpoint -> calibrate performance -> N epochs of (selfplay -> finalize ->
sample window -> D6 symmetry -> train passes -> epoch checkpoint -> optional
eval) -> final checkpoint -> diagnostics.

**One shipped plugin: `shrimp`.** The recipe run is launched as
`python -m hexo_train.cli.train_model configs/shrimp_main_7.toml` (via
`scripts/launch_training.sh`); `[model].module = "shrimp.plugin"` selects the
plugin. The package itself is model-neutral — nothing here is shrimp-specific.

## Design: defaults plus plugin overrides

The package separates orchestration (owned here) from model semantics (owned by
plugins). `hexo_train` ships model-neutral defaults -- target helpers, a
deterministic D6 selector, and placeholder checkpoint writers -- and each plugin
returns `ComponentOverrides` to replace the pieces it owns. The Shrimp plugin
supplies its own trainer, checkpoint loader/saver, and replay storage, so on a
real run the epoch ordering, diagnostics, and artifact layout come from this
package while storage, training, and checkpoint IO come from the plugin. The
default implementations are exercised by the package's FakePlugin pipeline
tests.

## Module table

All paths relative to `packages/hexo_train/python/hexo_train/`.

| File | Role |
| --- | --- |
| `cli/train_model.py` | Thin argparse CLI -> `TrainingPipeline().run(config_path)`. The single public command (`python -m hexo_train.cli.train_model` / `hexo-train-model` console script). |
| `pipeline.py` | `TrainingPipeline`: the run "map". Fixed step sequence (initialize_run, load_checkpoint, calibrate_performance, run_epochs, publish_final_model, write_diagnostics), each wrapped in `_run_step` diagnostics. Teardown calls `trainer.close()` when present (e.g. Shrimp's shard-expansion pool). |
| `config.py` | TOML loading and normalization into frozen dataclasses (`ModelConfig`, `RunConfig`, `LoopConfig`, `SelfPlayConfig`, `SamplesConfig`, `TrainConfig`, `CheckpointConfig`, `TrainingConfig`). Resolves paths relative to the config dir. Typed sections cover the orchestration skeleton; `[model.config]` passes through opaquely to the plugin's own config module, and model-neutral extras like `[shared.game]` stay reachable via `TrainingConfig.raw` / `ctx.section()`. Every config in `configs/` is TOML. |
| `registry.py` | Plugin discovery from the explicit module path in `[model].module` (`load_model_plugin` -> `import_module` -> read the module's plugin object). Defines the `ModelPlugin` Protocol covering the construction hooks. |
| `context.py` | `RunContext`: creates `output/`, `checkpoints/`, `diagnostics/` dirs; holds the `DiagnosticsWriter`, `outputs` dict, `epoch_outputs` list; `ctx.section()` exposes raw config sections. |
| `components.py` | Dependency container: `SharedComponents` (mutable run state: window/symmetries/checkpoint_state), `ComponentOverrides` (what a plugin returns), `ModelComponents`; `build_model_components` merges defaults + overrides. Fields are intentionally loosely typed (`Any`); the contract is duck-typed. |
| `defaults.py` | `build_shared_components`: default target helpers, `D6SymmetrySelector`, `CheckpointStore`, game spec from `[shared.game]`. |
| `checkpoints.py` | When checkpoints load/save: `load_or_initialize_checkpoint` (`resume_from`/`initialize_from` -> plugin loader), `save_epoch_checkpoint`/`save_final_checkpoint` (plugin saver, or a placeholder metadata file for plugins without one), per-epoch pointer publish; updates `shared.checkpoint_state` for the next selfplay. |
| `artifacts.py` | Durable run files: `write_run_manifest` (`manifest.json` -- read by the dashboard for lineage/arch), `publish_selfplay_checkpoint_pointer` (`selfplay_checkpoint.txt`), `write_final_diagnostics` (`run.completed.json`), `CheckpointStore` path/placeholder helper. |
| `diagnostics.py` | `DiagnosticsWriter`: append-only `diagnostics/events.jsonl` + per-stage `<step>.json`. The dashboard live-status view tails `events.jsonl`. |
| `symmetry.py` | Training-owned deterministic D6 augmentation selection (blake2b of `seed:epoch:sample-id`); `D6SymmetrySelector`, `SampleSymmetrySelection`. |
| `epoch/loop.py` | `run_epochs`/`run_epoch`: the fixed per-epoch order above; `_start_epoch` resumes from the loader's `{"status": "loaded", "epoch": N}` state (how a resumed run fast-forwards past completed epochs). |
| `epoch/selfplay.py` | `generate_selfplay` dispatch: `plugin.generate_selfplay()` > placeholder payload; the result is stored on `shared.selfplay_result`. |
| `epoch/samples.py` | Sample window per epoch: `finalize_samples` (plugin `sample_finalizer` hook), then `select_training_samples` -- delegates to `trainer.select_training_samples` when the trainer provides it (Shrimp's KataGo-style shuffle over `.npz` shards), else builds a default index/window. |
| `epoch/symmetry.py` | `select_epoch_symmetries`: applies the D6 selector to the current sample window and stores the `SampleSymmetrySelection` on shared state. A trainer may consume the full per-sample tuple or just `selection.seed`; the Shrimp trainer uses the seed and draws its own per-row D6 symmetries during shard expansion. |
| `epoch/training.py` | `train_passes`: calls `trainer.train_passes(passes, sample_window, sample_symmetries, ...)` or returns skipped. |
| `__init__.py` | Re-exports config dataclasses, `RunContext`, `TrainingPipeline`, `load_model_plugin`, D6 selector types. |

## Connections to other packages

Imports OUT (what this package uses):

- `hexo_utils.encoding` (`D6_SIZE`, `D6Symmetry`) for the D6 selector.
- Declares `hexo-engine` and `hexo-runner` as deps; game execution is reached
  through `plugin.generate_selfplay`, which drives them inside the plugin --
  this package itself never imports the engine or runner.

Imports IN (who uses this package):

- The `shrimp` plugin imports `hexo_train.components.ComponentOverrides` and is
  loaded by module path (`[model].module = "shrimp.plugin"` in the configs).

Plugin/trainer contract (duck-typed; hooks dispatched via hasattr checks in
`pipeline.py` and `epoch/*.py`):

- Plugin hooks: `build_model`, `training_component_overrides`,
  `generate_selfplay`, optional `evaluate_epoch` and `calibrate_performance`;
  an optional `sample_finalizer` component (`.finalize()`).
- Trainer hooks: `select_training_samples`, `train_passes`, optional `close()`.
- Checkpoint loader/saver: `loader.load(ref, ...)` returns the state dict
  stored on `shared.checkpoint_state`; `{"status": "loaded", "epoch": N}`
  drives `epoch/loop._start_epoch` to resume at epoch N+1 (keep both sides of
  the dict shape in sync). `saver.save(name=...)` returns the checkpoint path.

File-format contracts (no Python import):

- `manifest.json` (from `artifacts.write_run_manifest`) -- read by
  `hexo_frontend/web.py` and `debug_infer.py` for lineage/arch detection.
- `diagnostics/events.jsonl` -- tailed by the dashboard's live training status.
- The plugin writes `diagnostics/<prefix>.selfplay.epoch_*.json` etc. through
  `ctx.diagnostics.write_json` (the prefix is manifest-derived; Shrimp writes
  `shrimp.*`); the dashboard reads them.

## How the Shrimp selfplay -> replay -> train loop maps onto this package

Per epoch (`epoch/loop.py` order), with the `shrimp` plugin:

1. **Selfplay** -- `plugin.generate_selfplay` runs `shrimp/selfplay.py`
   (continuous scheduler over the Rust search), which writes per-game compact
   `.npz` shards + JSON sidecars under `<run>/selfplay/` and live/epoch
   diagnostics JSON. hexo_train sees only the summary dict.
2. **Replay/sample window** -- `epoch/samples.select_training_samples` delegates
   to the Shrimp trainer's `select_training_samples`, which builds a
   KataGo-style shuffle over the mtime-ordered `.npz` shard window; the plugin
   owns its replay storage.
3. **Train** -- `epoch/training.train_passes` calls the Shrimp trainer's
   `train_passes` (parallel shard expansion with per-row D6, AdamW steps).
4. **Checkpoint + eval** -- `checkpoints.save_epoch_checkpoint` via the plugin
   saver; `plugin.evaluate_epoch` runs the paired-anchor + SealBot evaluation.

## Entry points / how it gets exercised

| Entry | Notes |
| --- | --- |
| `python -m hexo_train.cli.train_model <config>` / `hexo-train-model` | The sole public command. |
| `scripts/launch_training.sh` | Sets the load-bearing Shrimp arch env + perf kernels, then starts the auto-relaunch supervisor (`scripts/supervise.sh`) or one foreground run. |
| `tests/test_training_pipeline_simplification.py` | The package's dedicated test: config normalization, registry, full FakePlugin pipeline run, resume, D6 determinism. Run with `python -m pytest tests/test_training_pipeline_simplification.py -q`. |
