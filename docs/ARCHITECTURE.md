# Architecture: the Shrimp training system

This document is the study reference for how the whole system fits together —
the layers, the data flow between them, and where every piece lives in the tree.
It assumes you have skimmed [`intro_to_hexo.md`](intro_to_hexo.md) (the game) and,
ideally, [`shrimp_blueprint.md`](shrimp_blueprint.md) (the ideas, no code).
For a plain-language tour of the bot itself — board representation, network,
search, and training — before the code-level detail here, see
[`SHRIMP.md`](SHRIMP.md). Everything named below is a real path in this
repository.

The system is one model lineage — **Shrimp** — wrapped in a model-neutral
orchestration and execution layer. Six packages, in dependency order:

```
hexo_engine   (Rust)   authoritative game rules
hexo_utils    (Rust+Py) .hxr records, state_hash, D6 contract
hexo_runner   (Py)      player contracts, match loop, SealBot adapter
hexo_train    (Py)      config-driven epoch orchestration (model-neutral)
shrimp      (Rust+Py) THE model: net + search + featurization + eval + trainer
hexo_frontend (Py)      web dashboard (arena / history / debug)
```

`hexo_train` knows nothing about Shrimp specifically: it discovers a *plugin*
by module path (`[model].module = "shrimp.plugin"`) and drives it through a
duck-typed contract. Shrimp supplies the model semantics; hexo_train supplies
the run skeleton (epoch ordering, diagnostics, artifact layout).

---

## 1. The layers

### 1.1 Rules engine — `hexo_engine`

A Rust crate (`packages/hexo_engine/rust/src/`) with a thin typed Python facade
(`python/hexo_engine/`), built as the PyO3 extension `hexo_engine._rust`.

- `state.rs` — `HexoState`, the `TurnPhase` machine (Opening → FirstStone →
  SecondStone), `apply_placement`/`apply_with_delta` + undo, snapshot/replay.
- `board.rs` — sparse stone storage (hash map + insertion-ordered occupied list).
- `tactics.rs` — incremental 6-cell window tracking (3 axes × 6 offsets = 18
  windows touched per placement), threat (≥4 single-colour) and win detection.
- `legal.rs` — the incremental legal-move store (`LEGAL_RADIUS = 8`) and the
  canonical packed action-ID encoding `((q + 2^15) << 16) | (r + 2^15)`.
- `rules.rs`, `coord.rs`, `snapshot.rs`, `error.rs` — legality validation, axial
  coordinates + hex distance, replay snapshots, error types.
- `pybridge.rs` — the PyO3 module plus a versioned **C-ABI capsule**
  (`state_api_capsule`, version 2) that lets native crates clone a live Python
  `HexoState` into an owned Rust state without going through Python.

The engine is the single source of truth: everything downstream asks it for
legality, terminality, and threat facts rather than re-deriving them.

### 1.2 Shared contracts — `hexo_utils`

A Rust crate + Python facade (`hexo_utils._rust`) owning three cross-cutting
contracts:

- **`.hxr` record codec** (`rust/src/records.rs`) — the repo's binary
  game-record format (magic `HEXOREC1`, varint/zigzag payloads: game_id, seed,
  status, action IDs, winner, placements, optional abort record). Schema v1;
  readers reject other versions.
- **`state_hash`** (`rust/src/state_hash.rs`) — a deterministic,
  placement-order-sensitive `u64` identity for a position, used as the key for
  the neural-eval cache in search.
- **D6 symmetry contract** (`python/hexo_utils/encoding/symmetry.py`) — the
  order-12 hex symmetry group (`D6_SIZE = 12`), the frozen `D6Symmetry` and
  `ActionSymmetryMapper` protocol, and `transform_action_ids`. Consumed by
  `hexo_train.symmetry` for training-time augmentation.

### 1.3 Featurization — `shrimp` (Rust `features.rs` + Python `features.py`)

Shrimp does not feed the network a fixed board crop. It builds a **support
set** that grows and shrinks with the game:

```
support set = (all stones) + (all legal move cells) + (a one-cell "halo" border)
```

Cells are ordered **legal-first** (`[legal | stones | halo]`), which gives the
*legal-prefix* property: the policy head produces one logit per cell in order, so
it can only ever score legal moves — no masking step, no coverage bugs. Each cell
carries **15 features** (occupancy, turn phase, recency, distance-to-nearest,
threat/standing-win membership for either side).

Featurization is implemented twice for speed and correctness: the fast Rust path
(`rust/src/features.rs`, `support.rs`, exposed as `_rust.featurize_states`) is the
production path; the Python path (`python/shrimp/features.py`, `support.py`,
`geometry.py`) is the reference and the CPU/debug path. The support radius is
**env-driven** (`SHRIMP_SUPPORT_RADIUS`, default 8; the shipped weights use 4).

### 1.4 The network — `shrimp/model.py`

`ShrimpNet` (`python/shrimp/model.py`) is a stem → trunk → three-heads graph
over the support set. **Its geometry is read from the environment at import
time** (`python/shrimp/constants.py`) and is load-bearing — a checkpoint only
loads into a net built with matching values:

| Env var | Shipped value | Meaning |
|---|---|---|
| `SHRIMP_CHANNELS` | 192 | trunk width |
| `SHRIMP_ATTENTION_HEADS` | 3 | attention heads (head_dim 64) |
| `SHRIMP_TRUNK` | `CCACCACCACCACCA` | block order: `C`=hex conv, `A`=attention |
| `SHRIMP_SUPPORT_RADIUS` | 4 | featurize radius |

The trunk alternates two kinds of vision:

- **Hex convolution** (`C`) — each cell combines itself with its 6 hexagonal
  neighbours, with *direction-typed* weights (the Q-neighbour uses different
  learned weights than the R-neighbour, because line direction is the whole
  point of the game). Local pattern recognition.
- **Attention** (`A`) — full all-pairs attention over the support set plus 8
  learned *summary tokens*, with a learned relative-position bias keyed on
  distance/direction. Whole-board strategy in one step.

`LayerNorm` (not BatchNorm) keeps each cell stable independent of board size or
batch composition. The three heads read from the summary tokens plus the mean of
the real cells: **policy** (per-legal-move logits), **value** (a distribution
over 65 bins spanning −1…+1), and **auxiliary** (moves-left + short-term value).

Fast kernels live alongside: `_triton_conv.py` and `_triton_attn.py` are Triton
implementations guarded on `x.is_cuda`, so the eager PyTorch path is always
available on CPU. The env flags that gate them (`SHRIMP_TRITON_*`,
`SHRIMP_FLEX_PAIR`, `SHRIMP_SERVE_HALF`, …) are parity-tested against eager.

### 1.5 Search — `shrimp/rust/src/{search,tree}.rs`

Search is Rust for speed. The entry point is `ShrimpMctsSession` (exposed from
`lib.rs`), and the self-play driver is its `run_continuous` method: a continuous
scheduler that keeps many games in flight and replaces finished games per slot.

The algorithm is **Gumbel AlphaZero** on a PUCT substrate:

- At the **root**, Gumbel-Top-m sampling selects a candidate set and **Sequential
  Halving** allocates the visit budget across them, guaranteeing a
  policy-improving move even at low visit counts.
- At **non-root** nodes, selection uses the completed-Q / σ formulation over the
  same PUCT machinery (`c_puct`, completed-Q backup in `tree.rs`).

The classic AlphaZero exploration knobs — Dirichlet root noise, forced playouts,
visit-scaled c_puct — were removed from the codebase; Gumbel's root sampler and
Sequential Halving are the exploration and low-visit allocation mechanisms in
their place. **Playout Cap Randomization** (PCR) runs the full-visit search on a
fraction of moves and a cheaper search on the rest, so full-search positions
produce the best training targets while games still flow cheaply.

To keep the search fed affordably, the Rust side **batches** deduplicated leaf
positions, **caches** evaluations keyed by `state_hash` (FIFO, bounded), and
**deduplicates** repeats within a batch — see §2 for the wire protocol.

### 1.6 Training loop — `hexo_train` + `shrimp`

`hexo_train` owns the fixed run lifecycle; Shrimp owns what happens inside each
step. Entry is `python -m hexo_train.cli.train_model <config>` →
`TrainingPipeline.run` (`hexo_train/pipeline.py`):

```
initialize run dirs + manifest.json (artifacts.py)
  -> load/initialize checkpoint (checkpoints.py -> shrimp loader)
  -> per-epoch loop (epoch/loop.py):
       selfplay -> finalize samples -> select replay window
                -> D6 symmetry seed -> train passes
                -> epoch checkpoint -> optional eval
  -> final checkpoint + run.completed.json
```

The Shrimp plugin (`python/shrimp/plugin.py`) supplies the hooks:
`build_model`, `training_component_overrides` (trainer + checkpoint loader/saver +
replay storage), `generate_selfplay`, and `evaluate_epoch`. Shrimp owns its own
replay storage (`uses_shared_sample_store` is not used), so on a real run the
epoch ordering and diagnostics come from `hexo_train` while storage, training,
and checkpoint IO come from shrimp.

Inside the plugin:

- **Self-play** — `selfplay.py:generate_selfplay_epoch` builds a
  `ShrimpMctsSession`, runs `run_continuous`, and writes finished games. It
  applies the per-move levers (temperature schedule, PCR full/fast, policy-init
  openings) and records each move's search visit distribution as the policy
  target.
- **Replay window** — finished games become compact `.npz` shards + JSON
  sidecars under `<run>/selfplay/` (`shards.py`, `samples.py`). `trainer.py`
  builds a mtime-ordered KataGo-style shuffle window over them
  (only *completed* games; truncated games are dropped so value labels are real).
- **Training pass** — the Rust `expand_shard_train` (`replay_expand.rs`) expands
  compact shards to dense tensors applying per-row D6 symmetry; `trainer.py` runs
  AdamW steps against `losses.py` (policy + 65-bin value + opponent-policy +
  short-term-value + moves-left).
- **Checkpoints** — `checkpoints.py` writes `ShrimpNet` state (+ optimizer +
  epoch for full checkpoints). `resume_from` restores model+optimizer+epoch;
  `initialize_from` is weights-only (how a BC prefit warm-starts a run).

### 1.7 Evaluation — `shrimp/{eval_arena,multistage_eval,eval_stats}.py`

Strength is measured against *fixed* opponents, because a rising self-play loss
does not mean the model got worse (longer games inflate it). Each eval cadence
plays:

- **Paired games** vs frozen **anchor** checkpoints — two games sharing an
  opening with swapped sides (common random numbers), scored **pentanomially**
  (the five outcomes of a pair) so the correlated pair does not produce a
  falsely-tight confidence interval. A rolling **Bradley-Terry / Elo pool**
  (`eval_stats.py`) combines all results.
- **Unpaired games** vs **SealBot** (optional external C++ minimax, §1.8 of the
  README), whose minimax depth varies with time so CRN pairing does not apply.

`multistage_eval.py` runs the staged SPRT-style screen and writes
`diagnostics/shrimp.multistage_eval.epoch_*.json`. Verdicts
(PROMOTE/REGRESS/INCONCLUSIVE) are **informational only** — nothing gates or
redirects training. A separate `head_audit.py` checks that the moves-left head
correlates with reality and disables it if not.

### 1.8 Dashboard — `hexo_frontend`

A single stdlib `ThreadingHTTPServer` (no web framework) serving one SPA with
three screens. It imports no training code on its HTTP path and never loads torch
in-process — all model inference is delegated to an out-of-process CPU worker.

- **Match** (`#match`) — `ManualMatchController` bridges browser clicks to
  `hexo_runner` players and runs real games through `run_match`. Opponents:
  manual, SealBot, or a checkpoint bot (played via the debug worker).
- **History** (`#history`) — reads run directories under `cwd/runs` read-only:
  `manifest.json`, `diagnostics/*.json` + `events.jsonl`, `.hxr` records,
  `checkpoints/*.pt`. Live status band + per-epoch trends + paged game history.
- **Debug** (`#debug`) — single-position forensics: policy/value/aux heads, a
  fresh MCTS, a pure-Python PUCT *tree explorer*, `.npz` training-row decode, and
  per-ply value sweeps.

The Debug worker chain: `web.py` → `debug_service.py` (spawns and manages the
worker, NDJSON over stdin/stdout, timeouts, LRU cache) → `debug_worker.py`
(per-op dispatch) → `debug_infer.py` (CPU inference, rebuilds `ShrimpNet` from
a checkpoint's state dict). Checkpoint bots in the Match arena also decide moves
through this worker, so a bot in the arena plays exactly like its eval self.

---

## 2. The inner-loop wire protocol

The search (Rust) needs the network's opinion millions of times; crossing the
language boundary one position at a time would be hopeless. The contract:

1. Rust collects a batch of deduplicated leaf positions and packs them, via
   `serve_pack.rs` (`build_serve_groups`, `F16Buf`/`I32Buf`/`U8Buf`), into
   half-precision feature buffers grouped by support size.
2. It calls the Python `ShrimpEvaluator` (`inference.py`), which pads to a
   ceiling, runs `forward_policy_value` on the (optionally half-precision) GPU
   model, and returns:
   - `values_bytes` — f32 × B, clamped to [−1, 1];
   - `priors_bytes` — f32, positional over each group's legal cells;
   - optionally `logits_bytes` when requested.
3. Rust parses those buffers strictly (exact lengths) and backs the values up
   the tree.

The byte layout is a fixed contract in both directions — the parity harness
(`tests/test_shrimp_*parity*.py`) exists to keep the native path and the
reference path byte-identical for the shipped search profile.

---

## 3. Data flow, end to end

```
configs/shrimp_main_7.toml
      |
      v
hexo_train.cli.train_model  ->  TrainingPipeline (hexo_train/pipeline.py)
      |  plugin: hexo_train/registry.py -> shrimp/plugin.py
      v
  per-epoch loop (hexo_train/epoch/loop.py)
      |
      |-- selfplay: shrimp/selfplay.py (continuous scheduler)
      |       |                                   ^
      |       v   f16 feature buffers (serve_pack)| values_bytes /
      |   shrimp._rust.ShrimpMctsSession       | priors_bytes
      |   .run_continuous (search.rs / tree.rs) --> shrimp/inference.py (GPU)
      |       |  state clone via hexo_engine._rust state-api capsule
      |       v
      |   game truth: hexo_engine (Rust rules)
      |       |
      |       +--> <run>/selfplay/*.hxr    (hexo_runner.records / hexo_utils codec)
      |       +--> <run>/selfplay/*.npz + .json  (shards.py / samples.py)
      |       +--> diagnostics/shrimp.selfplay.{live,epoch_N}.json
      |
      |-- replay window: shrimp/trainer.py (KataGo shuffle over .npz shards)
      |
      |-- train: expand shards (replay_expand.rs, per-row D6) + AdamW (losses.py)
      |
      |-- checkpoint: shrimp/checkpoints.py -> <run>/checkpoints/epoch_NNNNNN.pt
      |
      +-- eval: eval_arena.py / multistage_eval.py
              vs anchors (paired, pentanomial) + SealBot (unpaired)
              -> <run>/evaluation/... *.hxr
              -> diagnostics/shrimp.multistage_eval.epoch_N.json

<run dir>  <---- read-only scan ----  hexo_frontend/web.py (:8080)
                                          |              |
                            /api/training/*        /api/debug/* -> debug_service.py
                                  |                                  | NDJSON
                                  v                                  v
                        browser SPA (app.js)              debug_worker.py (CPU torch)
                        #match  #history  #debug           -> debug_infer.py
```

---

## 4. Run directory layout

A run directory is created under `runs/<name>/` by `hexo_train` + the Shrimp
plugin. The dashboard treats a directory as a run if it contains `diagnostics/`
or `selfplay/`, and lists its checkpoints from `checkpoints/`.

| Artifact | Path in run dir | Writer | Reader |
|---|---|---|---|
| Run manifest (lineage, arch, config subset) | `manifest.json` | `hexo_train/artifacts.py` | dashboard, debug worker |
| Event log (step start/finish) | `diagnostics/events.jsonl` | `hexo_train/diagnostics.py` | dashboard live status |
| Self-play epoch summary | `diagnostics/shrimp.selfplay.epoch_*.json` | Shrimp selfplay | dashboard |
| Self-play live progress | `diagnostics/shrimp.selfplay.live.json` | Shrimp selfplay | dashboard `/api/training/live` |
| Eval diagnostics | `diagnostics/shrimp.multistage_eval.epoch_*.json` | Shrimp eval | dashboard |
| Self-play game records | `selfplay/*.hxr` | Shrimp selfplay via `hexo_runner.records` | dashboard, debug worker |
| Compact training shards | `selfplay/*.npz` + `.json` | Shrimp `shards.py` | trainer, debug row decode |
| Checkpoints | `checkpoints/epoch_NNNNNN.pt` | Shrimp `checkpoints.py` | resume, dashboard, arena bots |
| Eval game records | `evaluation/.../*.hxr` | Shrimp eval | dashboard |
| Final marker | `diagnostics/run.completed.json` | `hexo_train/artifacts.py` | humans/scripts |
| Supervisor state | `supervisor.lock`, `supervisor_halted.flag`, `driver.pid`, `_resume_config.toml` | `scripts/supervise.sh` | supervisor |

---

## 5. Cross-cutting contracts and invariants

- **Tensor byte protocol** — §2. Rust ↔ Python must agree on the packed f16
  feature layout and the reply byte layout; the parity harness enforces it.
- **Action-ID packing** `((q + 2^15) << 16) | (r + 2^15)` — implemented in
  `hexo_engine` Rust (`legal.rs`), Python (`types.py`), and the dashboard JS.
  Persisted in `.npz` shards, `.hxr` records, and deep links, so all three must
  produce identical IDs.
- **Env-driven, load-bearing architecture** — the net geometry
  (`SHRIMP_CHANNELS/ATTENTION_HEADS/TRUNK/SUPPORT_RADIUS`) is read once at
  import. A checkpoint only loads into a matching net. The launch/prefit/dashboard
  scripts set the shipped values; a manual load must set them too.
- **Parity kernels** — every fast Triton/Flex kernel is gated on `x.is_cuda` and
  parity-tested against the eager path, so CPU and GPU produce the same decisions
  for the shipped profile.
- **Completed games only** — value labels come from the eventual winner, so
  truncated self-play games are discarded rather than labeled with a guess.

---

For the model's design rationale and target contract, see
[`specs/shrimp_model_spec.md`](specs/shrimp_model_spec.md); for the evaluation
statistics, [`specs/shrimp_eval_v2_spec.md`](specs/shrimp_eval_v2_spec.md);
for the dashboard screens, the `*_screen_v2_spec.md` files under
[`specs/`](specs).
