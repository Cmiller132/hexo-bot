# Showcase robustness plan

Prioritized plan for the GPU-crash robustness change set: what shipped, how to
deploy and validate it once the host is back, what is deferred, and the
end-user-visible effect. Companion to `ROBUSTNESS_OPS.md` (which documents the
container/ops layer in depth).

Goal, in one line: **use the GPU when it works, fail over to CPU fast when it
doesn't, and never brick the site.**

## 1. What shipped

Five layers, each containing a failure at the lowest level that can handle it.
The first was committed earlier (self-heal); the other four are this change set.

| # | Layer | Files | Contains |
|---|-------|-------|----------|
| 0 | **Worker recycle + move timeout** (already committed) | `bots.py`, `config.py` | A wedged/corrupted-queue worker: killed and replaced while uvicorn stays up. |
| 1 | **Source fix — chunk the no-grad bias gather** | `packages/shrimp/python/shrimp/model.py` | The XPU indexing abort at large S — split into query-axis chunks so no single gather kernel launch is huge. Stops the fault at its origin. |
| 2 | **Runtime GPU→CPU failover** | `bots.py`, `config.py` | Faults that still slip past the source fix: after `gpu_fault_threshold` (default 2) accelerator faults in the recycle window, the shard respawns on CPU and keeps serving. Optional health re-probe promotes it back. |
| 3 | **Container hardening** | `docker-compose.yml`, `docs/ROBUSTNESS_OPS.md` | `init: true` (tini PID-1 reaper) makes the zombie→dockerd→host cascade structurally impossible; `autoheal` sidecar restarts a dead/hung uvicorn. |
| 4 | **Frontend UX** | `web/app.js`, `web/index.html`, `web/style.css` | A slow/cold/failed bot turn: elapsed timer, warm-up note, honest recoverable "game couldn't finish" + "backend busy, try again" messaging instead of a dead "abandoned". |

### How they compose

- **GPU when it works.** Happy path is byte-for-byte unchanged. The source fix
  (layer 1) chunks the gather only when `S > SHRIMP_BIAS_GATHER_CHUNK_THRESHOLD`
  (default 1024); small boards take the exact stock single-gather path. Failover
  (layer 2) is inert while the shard serves cleanly on the accelerator.
- **Fast CPU failover when it doesn't.** If the abort still fires, layer 0
  recycles the wedged worker; layer 2 counts the accelerator fault and, on the
  2nd fault in-window, respawns that shard on CPU (exempt from the poison cap — a
  CPU shard cannot re-wedge, so it cannot respawn-loop). The site keeps answering
  moves, just on CPU for that shard. Optional re-promotion (layer 2) probes GPU
  health in a throwaway subprocess and, after a healthy streak, moves the shard
  back — anti-flap, and the probe can never wedge a serving worker.
- **Never bricks the site.** Layer 3's tini reaper converts the host-outage-class
  failure (unreapable zombies wedging dockerd) into a routine self-healed process
  death; autoheal covers a dead web process. Layer 4 makes every remaining
  user-visible hiccup honest and recoverable rather than a dead end.

Net: a deep/spread board that used to corrupt a worker's SYCL queue and
eventually cascade to a 502 now (a) most likely never faults, and (b) if it does,
degrades that one shard to CPU while the site stays up and the player sees a calm,
recoverable message.

## 2. Human deploy + validation (once the host is back)

Deploy is **gated on host recovery + human review.** Do these in order on the
real Intel Arc XPU container. Nothing below was exercisable on CPU-only WSL, so
this on-hardware pass is mandatory before trusting production behavior.

1. **Recover the host / container first.** Clear the wedged dockerd / zombies
   from the incident (CT restart if needed). DB is on the `showcase-db` volume
   and survives. Do not deploy onto a still-wedged daemon.
2. **Rebuild so the reaper and new code land.** `init: true` needs a container
   *recreate* — a running old container does not pick it up:
   ```sh
   docker compose up -d --build app
   ```
   First `up` also pulls the pinned `willfarrell/autoheal:1.2.0` sidecar.
3. **Mirror any `docker-compose.local.yml` override.** The reaper, the
   `autoheal=true` label, and the sidecar all live in the committed base file and
   apply on any invocation that includes it — **no change is required** unless the
   local override redefines the `app` service's `labels:` (then mirror
   `autoheal: "true"`) or sets `init: false` (must not). See ROBUSTNESS_OPS.md.
4. **Confirm health + autoheal wiring.**
   ```sh
   docker compose ps
   docker inspect --format '{{.State.Health.Status}}' <app-container>   # -> healthy
   docker compose logs --tail=50 autoheal                               # watching app
   ```
5. **XPU re-test the gather fix.** Reconstruct the deep-board stock-vs-patched
   sweep (the sibling agent's `scratchpad/xpu_fixtest.py` was session-local and is
   not committed) and run it *inside the container* against the real XPU. Confirm
   the chunked path completes without the SYCL-queue-corrupting abort across
   S~2700+ up to the deepest board (~4500 nodes). If CHUNK=512 still aborts on
   XPU, lower `SHRIMP_BIAS_GATHER_CHUNK` (256/128) — **env-only, no code change.**
   If no chunk size dodges the defect, the layer-2 CPU failover is the safety net.
6. **Warm the workers.** Load the Play view and let each shard cold-start / serve
   a first move so the pool is warm before inducing faults.
7. **Verify failover by inducing a fault.** Drive a deep/spread board that
   triggers the XPU indexing fault. Confirm: the shard records 2 accelerator
   faults in-window, **respawns on CPU** (not poisoned), and keeps serving moves.
   Then confirm re-promotion (if `SHOWCASE_GPU_REPROBE_S > 0`): the throwaway
   `_gpu_probe_main` resolves xpu, reports healthy, and promotes one downgraded
   shard back after the healthy streak — or reports unhealthy without wedging a
   worker. Check the reprobe spawn cost/frequency (default 120s) is acceptable
   under serving load, and confirm **no zombie accumulation** from the short-lived
   probe subprocesses under the real init situation.
8. **Verify the frontend UX live.** (a) Cold/first-move bot turn shows the elapsed
   clock and the >8s "warming up" note, which clears when the move lands.
   (b) Recycle a worker mid-turn → the finished snapshot renders "game couldn't
   finish" + the inline recover notice with a working New game button (not a dead
   "abandoned"). (c) OS reduced-motion stills the thinking-dot pulse.
9. **Both-net check.** Confirm layer 0 (worker recycle, uvicorn stays up) and
   layer 3 (autoheal, dead uvicorn) both behave, since they cover disjoint cases.

### Conservative first deploy

For a cautious v1, set `SHOWCASE_GPU_REPROBE_S=0` to disable auto re-promotion
(a CPU-downgraded shard then stays on CPU until the next restart, which is
acceptable) until the re-promotion path is observed healthy on real hardware.

## 3. Deferred / follow-up

- **GPU re-promotion is implemented but unproven on hardware.** It was verified
  only against a no-torch mock harness; the real subprocess self-check and
  fault-signature detection cannot run on CPU-only WSL. Treat it as *deferred to
  validation*: ship with `SHOWCASE_GPU_REPROBE_S=0` first, enable after step 7
  passes on XPU. Rationale: re-promotion recycles a shard and depends on a real
  probe — low blast radius but worth watching before trusting it unattended.
- **torch-xpu upgrade.** The chunked gather is a workaround for an Intel Arc XPU
  backend `Indexing.h` defect on an in-bounds gather, not a fix for the defect.
  Track upstream torch-xpu / oneAPI releases; a fixed backend lets us drop the
  chunk threshold back to "never" and remove the workaround. Deferred: no fixed
  release is available now and the workaround is byte-identical on CPU.
- **Blue-green / zero-downtime deploy.** Current deploy is a single-container
  recreate (`up -d --build app`), which drops in-flight games for a few seconds.
  A blue-green swap behind the Cloudflare tunnel would remove that. Deferred:
  out of scope for this incident, and `restart: unless-stopped` + autoheal already
  cover the unattended-recovery need.
- **Reconstruct and commit the XPU repro harness.** `scratchpad/xpu_fixtest.py`
  and `scratchpad/bias_chunk_verify.py` were session-local and are not in the
  repo. Fold a small deep-board repro into a committed on-XPU smoke test so future
  regressions in the gather path are caught, rather than relying on scratch files.
- **`_accel_device` is inferred from the request string, not real hardware.** If
  `device=auto/xpu` but no accelerator exists, the worker falls back to CPU at
  init yet the pool still thinks it is "on the accelerator" — the outcome is
  correct (harmless no-op respawn, reprobe reports unhealthy and never promotes)
  but the logs are slightly misleading. Deferred as a cosmetic/log-clarity item.

## 4. End-user-facing summary

For players, the site should just feel steadier:

- **The bot uses the GPU when it can and quietly switches to CPU when the GPU
  misbehaves.** A move may occasionally take a little longer, but games keep
  working — no site outage.
- **Slow turns are honest.** A thinking move shows an elapsed timer, and if a
  worker is cold you see a brief "taking a little longer than usual (warming up)"
  note that clears as soon as the move lands.
- **Failures are recoverable, not dead ends.** If the backend hiccups mid-game
  you get a clear "the bot backend hiccuped — no fault of yours, start a new game"
  message with a working New game button, instead of a silent "game abandoned".
  A busy backend on your move says "try that move again" rather than throwing an
  error.
- **No gameplay, board, or rules changed** — only resilience and feedback.
