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

Every human-facing "value" (analysis, per-ply summary, lab, telemetry
root value) is v̂ = E_{π′}[Q], side-to-move POV. The state-value head is
never served: KLENT self-play trains the action-value pathway only, so on
these checkpoints that head is an untrained readout of the trained trunk.

## Threat-Space Search (TSS-Gumbel)

The served family answers the session's questions with proofs where proofs
exist. It is on by default and switchable per game (the "Tactics" control in
game setup; `tss` on `POST /api/game`). The layer lives in
`apps/showcase/server/showcase/families/mantis_tss.py`; every verdict,
classification, and solve comes from `hexfield_eq._rust.TssProbe`, the
marshalling surface over the same Rust functions the hexfield_eq tree calls,
so the showcase holds exactly one definition of Hexo threat semantics.

- **Leaf values.** A leaf whose λ¹ analysis is decided answers ±1 instead of
  v̂. An undecided leaf that passes the deep gate gets a verified deep solve,
  submitted before the wave's forward so the two overlap; a verified win/loss
  answers ±1, an unknown or unfinished solve answers v̂.
- **Priors.** Always the net's, then the λ¹ move guard (zero everything but
  the win-now moves when one exists, else zero the λ¹-refuted replies, and
  fall back to the raw priors if that would zero everything). The session
  extends a line through `prior_argmax`, so the guard is what makes a line
  follow the forced continuation.
- **The root.** The same guard on the root priors, plus one verified deep
  solve running concurrently with the whole search. A proven win overrides
  the decision and reports `action_selection = "tss_deep_root_win"`. The root
  solve carries its own node cap and its own wall clock (20000 nodes / 3000 ms,
  against 500 / 1500 at the leaves): it runs once per move and can replace the
  played move, so it is worth budgeting on a different scale — the three forced
  wins the served bot missed in game 34e4cb07 needed 1577, 1952 and 12880
  solver nodes, and none of them is visible at 500.

The driver mirrors each candidate line's position, so it knows every leaf's
placement path from the root and names solves by that path: the solver reads
the turn phase, and only a true placement history carries it. A pumped leaf
that matches no mirrored line is a hard error, never a guess.

TSS off is the bare search above, byte for byte —
`apps/showcase/tests/test_mantisnet_tss.py` pins that against golden vectors
recorded from the pre-TSS driver, and pins the mirror, the guard, and the
root override.

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
- `apps/showcase/tests/test_mantisnet_tss.py` — TSS-Gumbel: line-mirror
  exactness, the λ¹ guard and leaf values, the verified deep root override,
  and TSS-off parity with the pre-TSS driver.
