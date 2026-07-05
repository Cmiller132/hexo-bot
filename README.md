# hexo-bot

An AlphaZero-style self-play reinforcement-learning system for **Hexo** — a
Connect6-family game played on an *unbounded* hexagonal grid (place stones, win
with six in a row along any of the three hex axes; two stones per turn after the
opening). The repo contains an authoritative **Rust rules engine**, a
**PyTorch + Triton neural model** called *Shrimp*, a config-driven **trainer**,
a **match runner** (with an optional external minimax opponent, SealBot), and a
**web dashboard** for playing the bot and inspecting runs.

**Trained weights are included** (`models/`, via Git LFS) — you can play the bot
in your browser in about ten minutes. This repo is written for people *studying*
the system: the configs and specs are annotated to explain *why*, not just what.

- New to the game? Read [`docs/intro_to_hexo.md`](docs/intro_to_hexo.md).
- Want a plain-language explainer of the bot itself — what Shrimp is, the
  board representation, the network, the search, and how it trains?
  [`docs/SHRIMP.md`](docs/SHRIMP.md).
- Want the ideas end-to-end, no code, no ML background assumed?
  [`docs/shrimp_blueprint.md`](docs/shrimp_blueprint.md).
- Want the architecture and data flow? [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

> **Platform note.** One clean **Linux / WSL** path is documented and supported.
> All commands below are Bash from the repo root. Windows-native Python is
> untested (the compiled Rust extensions and the CUDA/Triton kernels are built
> and exercised under Linux). A GPU is only needed for real training — playing
> the bot and running the CPU smoke train work on a plain CPU.

---

## 1. Quick start — play the bot (~10 minutes)

### 1a. Clone (Git LFS required)

The shipped weights are Git LFS objects. Install LFS **before** cloning or the
`.pt` files come down as tiny pointer stubs.

```bash
# Install the git-lfs binary first (Debian/Ubuntu: sudo apt install git-lfs;
# macOS: brew install git-lfs). `git lfs install` only registers an already-installed git-lfs.
git lfs install
git clone <repo-url> hexo-bot
cd hexo-bot
git lfs pull              # ensure models/*.pt are the real files, not pointers
```

### 1b. Build and install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest

# CPU PyTorch is enough to PLAY the bot (the dashboard bot runs on CPU).
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Build the three native crates (hexo_engine, hexo_utils, shrimp) into the venv.
HEXO_VENV=$PWD/.venv bash scripts/build_native.sh

# Install the pure-Python packages (no build deps; shrimp is imported from-tree).
pip install --no-deps -e packages/hexo_runner -e packages/hexo_train -e packages/hexo_frontend
```

`shrimp` is deliberately **not** pip-installed — it is imported from its source
tree via `PYTHONPATH`. The launch/dashboard scripts set that path themselves; you
only set it by hand for a manual `python` invocation (shown in §2 and §6).

### 1c. Stage the shipped weights as a "run" the dashboard can see

The Match arena picks an opponent by choosing a **run**, then a **checkpoint**
inside that run's `checkpoints/` folder. The dashboard lists a run only if its
directory contains a `diagnostics/` or `selfplay/` subfolder, and lists
checkpoints from `<run>/checkpoints/*.pt`. So drop the shipped inference weights
into a minimal run layout with a one-line manifest that marks it as a Shrimp
run:

```bash
mkdir -p runs/shipped/checkpoints runs/shipped/diagnostics
cp models/shrimp_main7_infer.pt runs/shipped/checkpoints/
printf '{"model": {"name": "shrimp"}}' > runs/shipped/manifest.json
```

(The empty `diagnostics/` folder makes the run appear in the list; the
`manifest.json` marks it as a Shrimp run so the Debug workbench identifies it
correctly. Nothing else is needed to play.)

> **Support radius (important).** The shipped weights were trained at featurize
> radius 4. The Match arena bot inherits its radius from the dashboard process
> environment, and `scripts/dashboard.sh` exports `SHRIMP_SUPPORT_RADIUS=4` for
> you — so launch the dashboard with that script (§1d) rather than a bare
> `python -m hexo_frontend.web`. In the Debug workbench, use the radius control to
> select **4** for these weights.

### 1d. Launch the dashboard and play

```bash
bash scripts/dashboard.sh        # serves http://localhost:8080
```

Open <http://localhost:8080>, go to the **Match** tab, and set up a game:

1. Pick a seat (e.g. **Player 0 = Manual**, **Player 1 = Checkpoint**).
2. For the Checkpoint seat, choose the run **`shipped`** and the checkpoint
   **`shrimp_main7_infer.pt`**.
3. Start the match and click cells to place stones.

The bot plays on the CPU debug worker using the run's as-trained search profile
(Gumbel visit-selection), so it moves like its evaluation self. First move pays a
one-time model-load cost.

> The dashboard is CPU-only and imports no training code on its HTTP path — it is
> safe to run alongside anything. The shipped weights were trained at support
> radius 4; the dashboard worker reads that from the checkpoint/run, so you do
> not set any architecture env vars to play.

---

## 2. Quick start — train

### 2a. Smoke first (CPU, under a minute)

Prove the whole self-play → train → checkpoint loop turns over before committing
a GPU to it. The smoke configs are mini versions of the real recipe with the same
Gumbel search profile and budgets cut to the bone.

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD/packages/shrimp/python

# Tiny architecture for the smoke net (env is LOAD-BEARING — see §7).
export SHRIMP_CHANNELS=32 SHRIMP_ATTENTION_HEADS=4 SHRIMP_TRUNK=CCA SHRIMP_SUPPORT_RADIUS=4

python -m hexo_train.cli.train_model configs/shrimp_smoke_tiny.toml
```

You should get a completed epoch (self-play games, a training pass, a checkpoint)
under `runs/shrimp_smoke_tiny/` in well under a minute. `configs/shrimp_smoke.toml`
is a slightly larger CPU smoke. Neither is a strength run.

### 2b. The real recipe (GPU)

`configs/shrimp_main_7.toml` is the shipped recipe: channels 192, 3 attention
heads, trunk `CCACCACCACCACCA` (~8.1M parameters), 1024 visits with Gumbel-Top-m
+ Sequential Halving. The heavily annotated config header explains every knob.

```bash
# scripts/launch_training.sh sets the LOAD-BEARING architecture env AND the
# parity-gated GPU perf kernels for you, then starts an auto-relaunch supervisor.
scripts/launch_training.sh --foreground     # one attached run
# or:
scripts/launch_training.sh                  # detached supervisor (auto-resume)
```

Install a CUDA build of PyTorch (from <https://pytorch.org>) instead of the CPU
wheel for this. `triton` comes with CUDA PyTorch and powers the fast serve/train
kernels; they are parity-gated and silently no-op on CPU/eager.

**Honest hardware expectations.** The shipped run was trained on a single
**RTX 4070 Ti (12 GB)**. Batch sizes and the in-flight serve depth in the config
are measured against that one card. Self-play inference and training share the
GPU. This is a *slow-cooker* recipe: a day of training buys incremental strength,
not an overnight superhuman bot — the included weights are epoch 18 of a long
run. Expect to leave it running and track strength via the dashboard's arena
evaluation, **not** the self-play loss (rising loss is benign here; see the
blueprint §9).

### 2c. Optional: warm start (behavioral cloning)

Training from a random net is fine but slow to leave the opening. You can
warm-start from a behavioral-cloning prefit on a public human-games corpus:

```bash
pip install huggingface_hub          # needed by the corpus fetcher
scripts/prefit_launch.sh             # fetch corpus -> shards -> BC prefit
```

This runs three stages: [`scripts/fetch_corpus.py`](scripts/fetch_corpus.py)
downloads the [`timmyburn/hexo-bootstrap-corpus`](https://huggingface.co/datasets/timmyburn/hexo-bootstrap-corpus)
dataset, `scripts/bootstrap_from_corpus.py` replays it through the engine into
training shards, and `scripts/prefit.py` trains a BC checkpoint **at the main_7
architecture**. Point `[checkpoint].initialize_from` in the config at a
checkpoint under `runs/shrimp_main_7_prefit/` to consume it. (The prefit
checkpoint itself is not shipped — you regenerate it.)

---

## 3. Building and deploying

The quick starts above get you playing and smoke-training. This section covers
building from source properly and running the system as a long-lived deployment:
a training supervisor, the dashboard, optional systemd units, and refreshing the
shipped weights.

### 3a. Building from source

Three of the six packages carry a Rust crate. `scripts/build_native.sh` builds
all three with `maturin develop --release` into the active venv
(`$HEXO_VENV`, default `.venv` at the repo root):

```bash
HEXO_VENV=$PWD/.venv bash scripts/build_native.sh
```

`--release` is mandatory — a debug build of the featurizer/search crate is
roughly 10x slower. `hexo_engine` and `hexo_utils` resolve from the venv
site-packages after this. `shrimp` is special: it is never pip-installed, it
is imported from its source tree via `PYTHONPATH`, so the script mirrors the
compiled `shrimp/_rust*.so` extension next to the Python package
(`packages/shrimp/python/shrimp/`) after building. If you ever move or clean
that tree, rerun the script.

Rebuild whenever you change anything under `packages/*/rust/` — the Python side
imports the compiled extension, so Rust edits do not take effect until you
rebuild. Pure-Python edits need no rebuild.

The pure-Python packages install once as editable, with no build step:

```bash
pip install --no-deps -e packages/hexo_runner -e packages/hexo_train -e packages/hexo_frontend
```

`--no-deps` is deliberate: these packages pull no third-party dependencies of
their own, and you have already installed torch/numpy/maturin yourself.

### 3b. Running training as a long-lived deployment

`scripts/launch_training.sh` is the front door. It exports the load-bearing
architecture env (channels/heads/trunk/radius — see §7) and the parity-gated GPU
perf kernels, then starts training one of two ways:

```bash
scripts/launch_training.sh --foreground   # one attached process; Ctrl-C stops it
scripts/launch_training.sh                 # detached auto-resume supervisor (default)
```

The detached form launches `scripts/supervise.sh` under `setsid nohup`, so the
run survives the shell you started it from. The supervisor gives you:

- **Crash-restart loop.** It relaunches the trainer after a non-zero exit,
  waiting 3 seconds between attempts.
- **Resume injection.** Before each launch it finds the newest
  `checkpoints/epoch_*.pt` and writes a `_resume_config.toml` with `resume_from`
  pointing at it, so a restart continues from the latest checkpoint (model,
  optimizer, and epoch), not from scratch.
- **Crash breaker.** If the trainer crashes fast (under 300s) three times in a
  row, or more than eight times in an hour, the supervisor writes a halt flag and
  stops relaunching instead of thrashing.
- **Single-instance lock.** A `supervisor.lock` holding the live PID prevents a
  second supervisor from starting on the same run.

Run state lives under `runs/<name>/` (default `runs/shrimp_main_7/`):
`checkpoints/` (the `.pt` files), `selfplay/` (`.npz` shards and `.hxr` records),
`diagnostics/` (per-epoch JSON and the live status), plus the supervisor's own
bookkeeping — `supervisor.log`, `supervisor.lock`, `driver.pid` (the current
trainer PID), and `_resume_config.toml`.

Config, run directory, and venv are env-overridable: `CONFIG`, `RUNDIR`,
`HEXO_VENV`.

**Stopping a run cleanly.** The halt flag,
`runs/shrimp_main_7/supervisor_halted.flag`, is what tells the supervisor not
to (re)start — it is checked when a supervisor boots and prevents a
relaunch-after-crash. But it is not polled mid-run, so creating it alone does not
interrupt training that is already going. To stop a live detached run, write the
halt flag first (so nothing relaunches), then kill the supervisor process group:

```bash
RUNDIR=runs/shrimp_main_7
touch "$RUNDIR/supervisor_halted.flag"          # prevents any relaunch/restart
kill -TERM -"$(cat "$RUNDIR/supervisor.lock")"  # kill the supervisor process group
```

The leading `-` on the PID sends the signal to the whole process group started
by `setsid`, taking the trainer child (`driver.pid`) down with the supervisor.
Delete the halt flag when you want to allow launches again:

```bash
rm runs/shrimp_main_7/supervisor_halted.flag
```

For a `--foreground` run there is no supervisor — just Ctrl-C the attached
process.

### 3c. Deploying the dashboard

`scripts/dashboard.sh` serves the run dashboard (the stdlib HTTP server in
`hexo_frontend.web`) detached. It takes an optional port argument (default 8080),
sets the `PYTHONPATH` for the from-tree packages, and exports
`SHRIMP_SUPPORT_RADIUS=4` so the arena bot matches the shipped weights:

```bash
bash scripts/dashboard.sh          # http://localhost:8080
bash scripts/dashboard.sh 9000     # a different port
```

It is single-instance per port: if one is already listening on that port the
script is a no-op. By default it binds `127.0.0.1` (local only). To expose it on
your LAN, set `HEXO_HOST=0.0.0.0`:

```bash
HEXO_HOST=0.0.0.0 bash scripts/dashboard.sh
```

The dashboard scans `$HEXO_DEBUG_RUN_ROOT` (default the repo root) for runs, and
logs to `dashboard.out.log` under that root. If `SEALBOT_PATH` is set it is
passed through as an arena opponent. To stop the dashboard, kill it by the module
it runs:

```bash
pkill -f 'hexo_frontend.web'
```

### 3d. Optional: systemd units for always-on deployment

For a machine that should run training and the dashboard across reboots, two
convenience units live under `scripts/systemd/`. They are optional — the scripts
above work standalone — and both ship with `/path/to/hexo-bot` placeholders you
must edit before use.

- `hexo-bot-supervisor.service` runs `scripts/supervise.sh`. Edit the four
  `Environment=` paths (`ROOT`, `HEXO_VENV`, `CONFIG`, `RUNDIR`) to your
  checkout, and the `ExecStartPre`/`ExecStart` paths. The architecture and perf
  env are set in the unit and are load-bearing — leave them matching `models/`.
  Uncomment the `SEALBOT_PATH` line to add the reference opponent.
- `hexo-bot-dashboard.service` runs `hexo_frontend.web` on port 8080. Edit the
  `WorkingDirectory`, `HEXO_DEBUG_RUN_ROOT`, the `PYTHONPATH`, and the `ExecStart`
  interpreter path.

Install by copying to the systemd unit directory and enabling:

```bash
# system-wide (as root):
sudo cp scripts/systemd/hexo-bot-supervisor.service /etc/systemd/system/
sudo cp scripts/systemd/hexo-bot-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hexo-bot-supervisor
sudo systemctl enable --now hexo-bot-dashboard
```

To run them as your own user instead, install under `~/.config/systemd/user/`
and use `systemctl --user enable --now ...`.

### 3e. Updating the shipped weights

The shipped inference file is a training checkpoint with the optimizer state
stripped and a small arch-metadata block embedded. To produce a fresh inference
export from a training checkpoint, use `scripts/export_weights.py`:

```bash
export PYTHONPATH=$PWD/packages/shrimp/python   # so --verify can import shrimp
python scripts/export_weights.py \
    runs/shrimp_main_7/checkpoints/epoch_000018.pt \
    --out models/shrimp_main7_infer.pt \
    --run shrimp_main_7 \
    --verify
```

The architecture is taken from `--channels/--heads/--trunk` if given, else from
the `SHRIMP_*` env vars, else inferred from the checkpoint's own weight shapes.
`--verify` reloads the export and instantiates `ShrimpNet` to prove it loads
strict (this needs torch and the Shrimp package importable — build it first, or
set `PYTHONPATH` as above). The exported file keeps the Shrimp-lineage shape,
so it loads through the same dashboard and eval loaders as the original.

---

## 4. Package map

Six packages: three carry a Rust crate (built by `scripts/build_native.sh`);
three are pure Python.

| Package | Role | Language |
|---|---|---|
| [`packages/hexo_engine`](packages/hexo_engine) | Authoritative Hexo rules engine: board, turn phases, legality, incremental 6-cell win/threat windows, packed action-ID encoding | Rust + PyO3 |
| [`packages/hexo_utils`](packages/hexo_utils) | Shared low-level contracts: the `.hxr` game-record codec, a deterministic `state_hash`, and the D6 symmetry transport contract | Rust + Python |
| [`packages/hexo_runner`](packages/hexo_runner) | Model-agnostic game execution: player contracts, the single-game match loop, `.hxr` records, and the SealBot subprocess adapter | Python |
| [`packages/hexo_train`](packages/hexo_train) | Config-driven training orchestration: loads a TOML config, discovers the model plugin, runs the epoch loop (selfplay → train → checkpoint → eval) | Python |
| [`packages/shrimp`](packages/shrimp) | **The model.** PyTorch net (hex convolutions + attention), Rust Gumbel/PUCT search, Triton kernels, self-play/eval, replay, and the training plugin | Rust + Python |
| [`packages/hexo_frontend`](packages/hexo_frontend) | Web dashboard (stdlib HTTP server): Match arena, run History, and a Debug workbench backed by a CPU-only torch worker | Python |

---

## 5. How it works

The learning loop is the classic AlphaZero virtuous cycle: **self-play games**
(a search picks each move) produce a **replay buffer** of positions labeled with
the search's visit distribution and the eventual game outcome; the network is
**trained** to imitate those; the improved network makes the next search sharper;
repeat. A standalone **evaluation** measures real strength against fixed
opponents; its verdict is informational and does not gate a checkpoint.

Where each layer lives, in dependency order:

- **Rules** — `hexo_engine` (Rust) is the single source of truth for the game.
- **Featurization** — `shrimp` builds a variable-size *support set* (stones +
  legal cells + a one-cell halo) with 15 features per cell, ordered legal-first
  so the policy can only ever score legal moves.
- **Search** — `shrimp`'s Rust crate runs Gumbel-Top-m + Sequential Halving on
  a PUCT tree, batching leaf evaluations back to the Python/GPU network.
- **Training** — `hexo_train` drives the epoch loop; `shrimp`'s trainer does
  the KataGo-style replay window, D6 augmentation, and AdamW passes.
- **Evaluation** — `shrimp` plays paired games vs SealBot and frozen anchors,
  scoring a Bradley-Terry/Elo pool with pentanomial confidence intervals.
- **Dashboard** — `hexo_frontend` reads run directories read-only and hosts the
  arena / history / debug screens.

For depth: [`docs/SHRIMP.md`](docs/SHRIMP.md) (a plain-language tour of the
bot), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (system + data flow),
[`docs/shrimp_blueprint.md`](docs/shrimp_blueprint.md) (the ideas, for
beginners), and the specs under [`docs/specs/`](docs/specs)
([model](docs/specs/shrimp_model_spec.md),
[eval](docs/specs/shrimp_eval_v2_spec.md), and the dashboard screen contracts).
The shipped weights are documented in [`models/MODEL_CARD.md`](models/MODEL_CARD.md).

### Why Gumbel, not classic PUCT + Dirichlet

Readers arriving from the AlphaZero paper will look for Dirichlet root noise and
forced playouts and not find them — that is deliberate. This repo ships **only**
the *Gumbel AlphaZero* search path (Danihelka et al., 2022): the root uses
**Gumbel-Top-m sampling + Sequential Halving** to allocate a small visit budget
across candidate moves and to guarantee a policy-improving move even at low
visits, and non-root selection uses the completed-Q / σ formulation. Gumbel's
root sampler *replaces* Dirichlet noise as the exploration source, and Sequential
Halving *replaces* forced playouts as the low-visit visit-allocation mechanism.
The classic exploration knobs (`root_dirichlet_*`, forced-playout `k`,
visit-scaled c_puct) were removed from the codebase entirely, so the configs
expose the Gumbel path and nothing vestigial. The underlying PUCT machinery
(`c_puct`, completed-Q backup) stays — Gumbel is layered on top of it.

---

## 6. Cross-package contracts

A few binding contracts hold the packages together; they are exactly the parts a
student wants to trace:

- **Tensor byte protocol.** The Rust search batches deduplicated leaf positions,
  packs them as half-precision feature tensors over the support set, and hands
  them to the Python evaluator, which returns exact-length `values`/`priors`
  byte buffers that Rust parses strictly. The wire format is a fixed contract in
  both directions (`packages/shrimp/rust/src/{payload,serve_pack}.rs` ↔
  `packages/shrimp/python/shrimp/inference.py`).
- **`.npz` replay shards.** Each finished self-play game is written as a compact
  columnar `.npz` shard plus a JSON sidecar under `<run>/selfplay/`; the trainer
  builds a mtime-ordered KataGo-style replay window over them
  (`packages/shrimp/python/shrimp/{shards,samples,trainer}.py`).
- **`.hxr` game records.** The repo's binary game-record format (magic
  `HEXOREC1`, varint/zigzag payloads) is owned by `hexo_utils` (Rust) and reached
  everywhere through `hexo_runner.records`. Self-play, evaluation, and arena
  games all write `.hxr`; the dashboard reads it.
- **Packed action IDs.** Every cell has a stable `u32` action ID,
  `((q + 2^15) << 16) | (r + 2^15)`, implemented in `hexo_engine` Rust
  (`legal.rs`), mirrored in Python (`types.py`), and again in the dashboard's
  JavaScript — because the IDs are persisted in shards, records, and deep links,
  the three implementations must produce identical IDs.
- **Diagnostics JSON.** The trainer writes per-epoch `diagnostics/*.json` +
  `events.jsonl` + `manifest.json` into each run directory; the dashboard reads
  those read-only to render history and live status.

### Loading the weights in code

To load the shipped weights outside the dashboard, build the net at the shipped
architecture — the geometry constants are read from env **at import time**, so
set them *before* importing `shrimp`:

```bash
export SHRIMP_CHANNELS=192 SHRIMP_ATTENTION_HEADS=3 \
       SHRIMP_TRUNK=CCACCACCACCACCA SHRIMP_SUPPORT_RADIUS=4
export PYTHONPATH=$PWD/packages/shrimp/python
```

```python
import torch
from shrimp.model import ShrimpNet
from shrimp.checkpoints import load_into

net = ShrimpNet()                      # built at the env-driven arch above
payload = torch.load("models/shrimp_main7_infer.pt", map_location="cpu")
load_into(net, payload)
net.eval()
```

`SHRIMP_SUPPORT_RADIUS=4` is load-bearing: the shipped weights were trained at
featurize radius 4, and the code default is 8 — the wrong radius silently
degrades inference. See [`models/MODEL_CARD.md`](models/MODEL_CARD.md) for the
full loading and architecture details.

---

## 7. Architecture is set by environment variables

The network geometry is **not** in the config file — it is read once from the
environment at import time, and it is **load-bearing**: a checkpoint only loads
into a net built with the same values. For the shipped weights:

```
SHRIMP_CHANNELS=192
SHRIMP_ATTENTION_HEADS=3        # head_dim 64 = 192/3, which the fast kernels want
SHRIMP_TRUNK=CCACCACCACCACCA    # C = hex convolution block, A = attention block
SHRIMP_SUPPORT_RADIUS=4
```

`scripts/launch_training.sh`, `scripts/prefit_launch.sh`, `scripts/dashboard.sh`,
and the systemd unit all default these correctly, so the documented flows are
consistent. You only set them by hand for a manual `python` load (§6) or a smoke
run at a smaller arch (§2a).

---

## 8. Optional: SealBot evaluation opponent

[SealBot](https://github.com/Ramora0/SealBot) is an external C++ minimax Hexo
bot. It is optional and disabled by default (`sealbot_enabled = false` in the
config). To use it as a fixed evaluation opponent (or as an arena opponent in the
dashboard):

```bash
git clone https://github.com/Ramora0/SealBot
# build it per its own README, then:
export SEALBOT_PATH=/path/to/SealBot
```

`scripts/dashboard.sh` and the training launcher pass `SEALBOT_PATH` through when
it is set; evaluation fails open (skips the SealBot leg) when it is absent.

---

## 9. Tests

Run the suite from the repo root with the venv active. The native extensions and
torch are required for most tests; those that need them skip cleanly when absent.

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD/packages/shrimp/python
python -m pytest tests/ -q
```

The Rust side has its own unit suites: `cargo test -p hexo_engine`,
`cargo test -p hexo_utils`, `cargo test -p shrimp`. A **parity harness**
(`tests/test_shrimp_*parity*.py` + the Rust `parity()` profile) pins the native
search against a reference implementation — it is both a correctness net and
study material.

---

## License and provenance

MIT — see [`LICENSE`](LICENSE).

This repository was extracted from a larger private research repo and published
as a clean, self-contained snapshot; the single-commit history is intentional.
