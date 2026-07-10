# hexo_train

Model-neutral, config-driven training orchestration for Hexo models. Given a
TOML config, it discovers a model plugin (via the `hexo_train.models` entry
point group or an explicit module path), then runs a fixed self-play training
lifecycle: initialize run artifacts -> load/initialize checkpoint -> calibrate
performance -> N epochs of (selfplay -> finalize -> sample window -> D6
symmetry -> train passes -> epoch checkpoint -> optional eval) -> final
checkpoint -> diagnostics.

**Status: ACTIVE.** The live runs are the `hexfield` and `hexfield_eq` bots,
launched as `python -m hexo_train.cli.train_model configs/hexfield_main_9.toml`
(and `configs/hexfield_eq_main_1.toml`) by the WSL supervisor scripts. Several
model lineages register plugins against this package; the parked
`dense_cnn_restnet`, `hexo_models` (`dense_cnn`/`hexgt`), and `hexgnn` packages
remain plugin-compatible.

## Design: defaults plus plugin overrides

The package separates orchestration (owned here) from model semantics (owned
by plugins). `hexo_train` ships model-neutral defaults -- a shared
`hexo_utils.samples` store/index/window path, target helpers, a deterministic
D6 selector, and placeholder checkpoint writers -- and each plugin returns
`ComponentOverrides` to replace the pieces it owns. The real model plugins
supply their own trainer, checkpoint loader/saver, and replay storage
(`uses_shared_sample_store=False`), so on production runs the epoch ordering,
diagnostics, and artifact layout come from this package while storage,
training, and checkpoint IO come from the plugin. The default implementations
keep the pipeline runnable without a real model plugin (early bring-up and
smoke tests).

## Module table

All paths relative to `packages/hexo_train/python/hexo_train/`.

| File | Role |
| --- | --- |
| `cli/train_model.py` | Thin argparse CLI -> `TrainingPipeline().run(config_path)`. The single public command (`python -m hexo_train.cli.train_model` / `hexo-train-model` console script). |
| `pipeline.py` | `TrainingPipeline`: the run "map". Fixed step sequence (initialize_run, load_checkpoint, calibrate_performance, run_epochs, publish_final_model, write_diagnostics), each wrapped in `_run_step` diagnostics. Teardown calls `trainer.close()` when present (e.g. the hexfield trainer's shard-expansion process pool). |
| `config.py` | TOML/YAML loading and normalization into frozen dataclasses (`ModelConfig`, `RunConfig`, `LoopConfig`, `SelfPlayConfig`, `SamplesConfig`, `TrainConfig`, `CheckpointConfig`, `TrainingConfig`). Rejects the removed `model_specific`/`stages` fields; resolves paths relative to the config dir. Typed sections cover the orchestration skeleton; `[model.config]` passes through opaquely to the plugin's own config module, and model-neutral extras like `[shared.game]` stay reachable via `TrainingConfig.raw` / `ctx.section()`. Every config in `configs/` is TOML. |
| `registry.py` | Plugin discovery: explicit module (`[model].module`), explicit entry point, or name lookup through the `hexo_train.models` entry point group. Defines the `ModelPlugin` Protocol covering the two construction hooks. |
| `context.py` | `RunContext`: creates `output/`, `checkpoints/`, `diagnostics/`, `samples/` dirs; holds the `DiagnosticsWriter`, `outputs` dict, `epoch_outputs` list; `ctx.section()` exposes raw config sections. |
| `components.py` | Dependency container: `SharedComponents` (mutable run state: sample store/window/symmetries/checkpoint_state), `ComponentOverrides` (what a plugin returns), `ModelComponents`; `build_model_components` merges defaults + overrides. Fields are intentionally loosely typed (`Any`); the contract is duck-typed. |
| `defaults.py` | `build_shared_components`: default target helpers (from `hexo_utils.samples`), `D6SymmetrySelector`, `CheckpointStore`, game spec from `[shared.game]`. |
| `checkpoints.py` | When checkpoints load/save: `load_or_initialize_checkpoint` (`resume_from`/`initialize_from` -> plugin loader), `save_epoch_checkpoint`/`save_final_checkpoint` (plugin saver, or a placeholder metadata file for plugins without one), per-epoch pointer publish; updates `shared.checkpoint_state` for the next selfplay. |
| `artifacts.py` | Durable run files: `write_run_manifest` (`manifest.json` -- read by the dashboard for lineage/arch), `publish_selfplay_checkpoint_pointer` (`selfplay_checkpoint.txt`), `write_final_diagnostics` (`run.completed.json`), `CheckpointStore` path/placeholder helper. |
| `diagnostics.py` | `DiagnosticsWriter`: append-only `diagnostics/events.jsonl` + per-stage `<step>.json`. The dashboard live-status view tails `events.jsonl`. |
| `symmetry.py` | Training-owned deterministic D6 augmentation selection (blake2b of `seed:epoch:sample-id`); `D6SymmetrySelector`, `SampleSymmetrySelection`. |
| `epoch/loop.py` | `run_epochs`/`run_epoch`: the fixed per-epoch order above; `_start_epoch` resumes from the loader's `{"status": "loaded", "epoch": N}` state (drives epoch fast-forward past seeded epochs on resume). |
| `epoch/selfplay.py` | `generate_selfplay` dispatch: `plugin.generate_selfplay()` (implemented by all real plugins) > `plugin.build_selfplay_request()` > placeholder payload; the result is stored on `shared.selfplay_result`. |
| `epoch/samples.py` | Sample window per epoch: `finalize_samples` (plugin `sample_finalizer` hook), then `select_training_samples` -- delegates to `trainer.select_training_samples` when the trainer provides it (e.g. the hexfield KataGo-style replay-window build over NPZ shards), else builds the shared `hexo_utils` index/window. `prepare_sample_store` opens the shared store at run start for plugins that use it. |
| `epoch/symmetry.py` | `select_epoch_symmetries`: applies the D6 selector to the current sample window and stores the `SampleSymmetrySelection` on shared state. A trainer may consume the full per-sample tuple or just `selection.seed`; the hexfield trainer draws its own per-row D6 symmetries from the run seed/epoch at expansion time. |
| `epoch/training.py` | `train_passes`: calls `trainer.train_passes(passes, sample_window, sample_symmetries, ...)` or returns skipped. |
| `__init__.py` | Re-exports config dataclasses, `RunContext`, `TrainingPipeline`, `load_model_plugin`, D6 selector types. |

## Connections to other packages

Imports OUT (what this package uses):

- `hexo_utils.samples` (target helpers, sample store/index/window -- the
  shared-store path), `hexo_utils.encoding` (`D6_SIZE`, `D6Symmetry`).
- Declares `hexo-engine` and `hexo-runner` as deps; game execution is reached
  through `plugin.generate_selfplay`, which drives them inside the plugin --
  this package itself never imports the engine or runner.

Imports IN (who uses this package):

- Model plugins import `hexo_train.components.ComponentOverrides` and register
  under the `hexo_train.models` entry point group:
  - `hexfield_eq` (`packages/hexfield_eq/pyproject.toml`) -- ACTIVE
  - `dense_cnn_restnet` (`packages/dense_cnn_restnet/pyproject.toml`) -- parked
  - `dense_cnn`, `hexgt` (`packages/hexo_models/pyproject.toml`) -- legacy/halted
  - `hexgnn` (`packages/hexgnn/pyproject.toml`) -- parked
- The active bots load by module path (`[model].module = "hexfield.plugin"` in
  `configs/hexfield_main_9.toml`, `"hexfield_eq.plugin"` in
  `configs/hexfield_eq_main_1.toml`), bypassing entry points.

Plugin/trainer contract (duck-typed; hooks dispatched via hasattr checks in
`pipeline.py` and `epoch/*.py`):

- Plugin hooks: `build_model`, `training_component_overrides`,
  `generate_selfplay`, optional `evaluate_epoch` and `calibrate_performance`;
  an optional `sample_finalizer` component (`.finalize()`).
- Trainer hooks: `select_training_samples`, `train_passes`, optional `close()`.
- Checkpoint loader/saver: `loader.load(ref, ...)` returns the state dict
  stored on `shared.checkpoint_state`; `{"status": "loaded", "epoch": N}`
  drives `epoch/loop._start_epoch` to resume at epoch N+1 (load-bearing for
  resume/fast-forward -- keep both sides of the dict shape in sync).
  `saver.save(name=...)` returns the checkpoint path.

File-format contracts (no Python import):

- `manifest.json` (from `artifacts.write_run_manifest`) -- read by
  `hexo_frontend/web.py` and `debug_infer.py` for lineage/arch detection.
- `diagnostics/events.jsonl` -- tailed by the dashboard's live training status.
- Plugins write per-epoch selfplay diagnostics JSON (e.g.
  `diagnostics/hexfield.selfplay.epoch_*.json`) through
  `ctx.diagnostics.write_json`; the dashboard reads them.
- `selfplay_checkpoint.txt` / `data/checkpoints/*_latest.txt` pointer files,
  written per-epoch and at final publish when `update_checkpoint_pointer =
  true` (legacy dense_cnn/hexgt configs; the active hexfield/hexfield_eq
  configs set it false).

## How a plugin's selfplay -> replay -> train loop maps onto this package

Per epoch (epoch/loop.py order), with the active `hexfield` plugin:

1. **Selfplay** -- `plugin.generate_selfplay` runs `hexfield/selfplay.py` (a
   continuous scheduler over the shared Rust MCTS), which writes per-game
   compact NPZ shards + JSON sidecars under `<run>/selfplay/` and live/epoch
   diagnostics JSON. hexo_train sees only the summary dict.
2. **Replay/sample window** -- `epoch/samples.select_training_samples`
   delegates to the hexfield trainer's `select_training_samples`, which builds
   a KataGo-style replay window from an mtime-free `(generation, game_key)`
   manifest over the NPZ shards into an in-RAM packed columnar window; the
   plugin owns its replay storage (`uses_shared_sample_store=False`).
3. **Train** -- `epoch/training.train_passes` calls the hexfield trainer's
   `train_passes` (single-pass shard expansion with pre-drawn per-row D6, AMP
   optimizer steps).
4. **Checkpoint + eval** -- `checkpoints.save_epoch_checkpoint` via the plugin
   saver; `plugin.evaluate_epoch` runs SealBot + checkpoint-opponent
   evaluation games.

## Entry points / how it gets exercised

| Entry | Notes |
| --- | --- |
| `python -m hexo_train.cli.train_model <config>` / `hexo-train-model` | The sole public command. |
| `scripts/_hexfield_supervise_main1.sh` | ACTIVE hexfield supervisor (CONFIG/RUNDIR env overrides select the run; the systemd units `hexfield-supervisor-9`/`-11` launch it). |
| `scripts/_hexfield_eq_supervise_main1.sh` | ACTIVE hexfield_eq supervisor (launched by the `hexfield-eq-supervisor-1` systemd unit). |
| `tests/test_hexfield_eq_smoke.py` and the other `tests/test_hexfield*` suites | Drive `TrainingPipeline`/registry against the real plugins (run under the WSL venv). |
