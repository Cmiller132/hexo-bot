# mantisnet

The MantisNet bot family: the inference side of the MantisNet research
lineage (a stone/window graph network for Hexo, MODEL_REPR_VERSION 7),
vendored and served as a showcase model family alongside `shrimp` and
`hexfield_eq`.

## What this package is

- **`rust/vendor/`** — five crates vendored byte-for-byte from the research
  repo's branch `main`: the rules engine, match runner, decision sessions,
  model-package API, and the MantisNet position encoder. They are renamed
  `mantis-*` (this workspace already ships its own `hexo_engine`); Cargo
  dependency renaming keeps every `use hexo_engine::...` in the vendored
  sources compiling unchanged. One deliberate addition, not in upstream:
  `GumbelSession::live_snapshot` (`vendor/search/src/gumbel.rs`), a
  read-only telemetry accessor.
- **`rust/src/`** — the pyo3 layer (`mantisnet._rust`): `Position`, the
  parallel batch builders, the version constants a checkpoint pins, and
  `GumbelSearch` — the vendored Gumbel sequential-halving session driven
  from Python (`begin` / `pump` / `resume` / `decision`, plus `snapshot` for
  live telemetry).
- **`python/mantisnet/`** — the serve closure of the research Python
  package, unchanged apart from imports and one deliberate rewrite: the
  window-latent custom ops' non-CUDA forwards are vectorized fp32 eager
  paths (`window_latents.py`), replacing the research repo's literal
  per-position/per-window loop oracles, which cost seconds per late-game
  board and made CPU/XPU serving unusable; the loop oracle survives as the
  equivalence detector in `tests/test_window_latents_eager.py`. Also here:
  the batch builder, the closed-form KLENT policy improvement, and the
  position readout. `serve.py` is new: checkpoint loading plus the one
  batched read every serve consumer shares.

## Checkpoint contract

An inference export carries `model` (fp32 state dict), `model_config`
(splats into `MantisConfig`), `versions` (the three semantic versions must
match this build; the recorded torch build string is advisory — a
CUDA-trained export must load on CPU/XPU wheels), and `klent`
(`tau`/`lam`/`mass_floor` — π′ is meaningless under parameters the model
did not train with, so a missing block is an error, never a default).
Architecture rides the checkpoint; nothing is frozen at import time, so any
mix of MantisNet checkpoints can share a worker process.

## Serve search semantics

The showcase family reproduces the research repo's evaluation search
exactly: Gumbel-Top-m root sampling over the bare policy (softmax of raw
logits), deterministic prior-argmax line extension, sequential halving on
line values, where each leaf answers with the improved policy π′ and its
expected value v̂. The Python driver answers the root and leaf pumps with
those quantities; every comparison the session makes is within one root, so
the log-softmax constant cancels and decisions match the as-evaled search.
Live telemetry is pull-based (`snapshot` between evaluation waves): no
extra forward, no callback into the search, an identical evaluator schedule
with the overlay on or off.

## Build / test

Built like the other model packages: `maturin develop --release -m
packages/mantisnet/Cargo.toml` (or `scripts/build_native.sh`, or the
showcase Dockerfiles). Python-side deps are ambient (torch, numpy).

- `cargo test -p mantis-hexo-engine -p mantis-hexo-runner -p
  mantis-hexo-search -p mantis-hexo-model -p mantis-hexo-model-mantisnet`
  — the vendored crates' own unit suites.
- `tests/test_mantisnet_builder_parity.py` — the vendored Rust encoder
  against the ported pure-Python reference builder, field for field: the
  correctness gate for the vendoring plus the pyo3 marshaling.
- `apps/showcase/tests/test_mantisnet_family.py` — the family end to end:
  catalogue, worker runtime, telemetry parity, device self-check, HTTP.
