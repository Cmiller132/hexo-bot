# Shrimp

**The model.** Shrimp is the single neural-network lineage this repo trains: a
variable-geometry hex-lattice net (hex convolutions + attention) plus the Rust
Gumbel/PUCT search that drives it, the featurization that feeds it, and the
self-play / replay / training / evaluation code that grows it. It plugs into the
model-neutral `hexo_train` orchestrator via `shrimp.plugin`.

For the *ideas* (no code, no ML background needed) read
[`docs/shrimp_blueprint.md`](../../docs/shrimp_blueprint.md). For the design
rationale and target contract, [`docs/specs/shrimp_model_spec.md`](../../docs/specs/shrimp_model_spec.md)
and [`docs/specs/shrimp_eval_v2_spec.md`](../../docs/specs/shrimp_eval_v2_spec.md).
For the shipped weights, [`models/MODEL_CARD.md`](../../models/MODEL_CARD.md).

## Shape: Rust + Python

A Rust crate (`rust/src/`) built as the PyO3 extension `shrimp._rust`, plus a
Python package (`python/shrimp/`). **Unlike the other packages, Shrimp is not
pip-installed** — it is imported from its source tree via `PYTHONPATH`, and its
compiled extension is mirrored next to the Python package by
`scripts/build_native.sh`. The launch, prefit, and dashboard scripts set
`PYTHONPATH=packages/shrimp/python` themselves; you set it by hand only for a
manual `python` invocation.

- **Rust** runs the hot loops — featurization and search — where being fast over
  millions of tree steps matters.
- **Python (on a GPU)** runs the network and the training/eval orchestration.

The two halves talk over a strict byte protocol (see `docs/ARCHITECTURE.md` §2).

## Architecture is env-driven and load-bearing

The network geometry is read from the environment **once at import** in
`python/shrimp/constants.py`, not from the TOML config. A checkpoint only loads
into a net built with matching values. The shipped weights use:

```
SHRIMP_CHANNELS=192          # trunk width
SHRIMP_ATTENTION_HEADS=3     # head_dim 64 = 192/3
SHRIMP_TRUNK=CCACCACCACCACCA # C = hex conv block, A = attention block
SHRIMP_SUPPORT_RADIUS=4      # featurize radius (code default is 8)
```

`SUPPORT_RADIUS` is as load-bearing as the rest: the shipped run trained at
radius 4, so loading it at the default 8 silently degrades inference.

## Module table

### Rust (`rust/src/`)

| File | Role |
|---|---|
| `features.rs`, `support.rs` | Build the support set (stones ∪ legal ∪ 1-ring halo) and the per-cell feature tensor; exposed as `_rust.featurize_states`. |
| `search.rs` | `ShrimpMctsSession` and `run_continuous` — the continuous self-play scheduler; Gumbel-Top-m + Sequential Halving on a PUCT tree. |
| `tree.rs` | Tree node storage, PUCT selection, completed-Q backup. |
| `serve_pack.rs` | `build_serve_groups` + the `F16Buf`/`I32Buf`/`U8Buf` packed buffers — the Rust→Python serve wire format. |
| `replay_expand.rs` | `expand_shard_train` — expands compact `.npz` shards to dense training tensors with per-row D6 symmetry. |
| `payload.rs`, `state.rs`, `cache.rs` | Batch payloads, owned Rust state (via the `hexo_engine` C-ABI capsule), and the `state_hash`-keyed eval cache. |
| `threats_shared.rs`, `constants.rs`, `support.rs` | Shared threat queries, tensor/geometry constants, support geometry. |
| `lib.rs` | The `_rust` pymodule: registers the session class, `featurize_states`, the serve buffers, and the replay expander. |

### Python (`python/shrimp/`)

| File | Role |
|---|---|
| `model.py` | `ShrimpNet`: stem → env-driven trunk (`ConvBlock`/attention) → policy/value/aux heads; also the cross-arch loader that infers geometry off a checkpoint's state dict. |
| `constants.py`, `geometry.py`, `features.py`, `support.py` | Env-driven geometry constants and the Python reference featurization (CPU/debug path; the Rust path is production). |
| `_triton_conv.py`, `_triton_attn.py` | Triton kernels for the hex conv and attention, guarded on `x.is_cuda`; the eager PyTorch path is always available on CPU. |
| `inference.py` | `ShrimpEvaluator` — the serve-side half of the wire ABI (`values_bytes`/`priors_bytes`), optional half-precision serve model. |
| `selfplay.py`, `batching.py` | Self-play epoch driver over `ShrimpMctsSession`; per-move levers (temperature, PCR, policy-init). |
| `shards.py`, `samples.py`, `window.py`, `buffer_manifest.py` | Compact `.npz` shard IO, sample construction, and the replay window. |
| `trainer.py`, `losses.py`, `train_state.py` | KataGo-style replay shuffle + AdamW passes; the multi-head loss (policy + 65-bin value + opp-policy + STV + moves-left). |
| `eval_arena.py`, `multistage_eval.py`, `eval_stats.py`, `head_audit.py` | Paired-vs-anchors + unpaired-vs-SealBot evaluation; pentanomial scoring, Bradley-Terry/Elo pool; moves-left head sanity check. |
| `checkpoints.py` | `ShrimpNet` save/load; `resume_from` (model+optimizer+epoch) vs `initialize_from` (weights-only warm start). |
| `expand_backends.py`, `prefit.py`, `engine_facts.py` | Shard-expansion backend selection, BC-prefit helpers, engine fact extraction. |
| `plugin.py` | `ShrimpPlugin` — the `hexo_train` contract: `build_model`, `training_component_overrides`, `generate_selfplay`, `evaluate_epoch`. |

## Connections to other packages

- **hexo_engine** — game truth (legality, terminality, threats) and the C-ABI
  capsule that clones a live `HexoState` into an owned Rust state for search.
- **hexo_utils** — the `.hxr` codec (via `hexo_runner.records`), `state_hash` for
  the eval cache, and the D6 symmetry contract.
- **hexo_runner** — `.hxr` writing and the `SealBotPlayer` used as an eval
  opponent.
- **hexo_train** — imports the plugin by module path
  (`[model].module = "shrimp.plugin"`) and drives it through the epoch loop.
- **hexo_frontend** — the dashboard debug worker imports `ShrimpNet` /
  `debug_infer` to serve checkpoints for the arena and debug screens.

## How it is exercised

- Training: `python -m hexo_train.cli.train_model configs/shrimp_main_7.toml`
  (or a smoke config), via `scripts/launch_training.sh`. See the root README.
- Build the extension: `scripts/build_native.sh` (maturin `--release`).
- Tests: `python -m pytest tests/test_shrimp_*.py -q` from the repo root with
  the venv active and `PYTHONPATH=packages/shrimp/python` set — including the
  Gumbel CPU smoke (`test_shrimp_gumbel_smoke.py`) and the parity harness
  (`test_shrimp_*parity*.py`). Rust side: `cargo test -p shrimp`.
