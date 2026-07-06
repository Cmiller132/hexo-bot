# Showcase robustness & ops

How the showcase stack survives an inference-worker crash without taking down
the site or the host, and what an operator does when it misbehaves.

## Background: the incident this hardens against

A torch-xpu indexing kernel faults in the model forward's no-grad
attention-bias gather at large support-set size (deep/spread board, ~2700+
nodes). The indices are provably in-bounds (verified on CPU) — it is an Intel
Arc XPU backend defect, not a bounds bug in our code. The fault corrupts the
worker's SYCL queue, so every later search on that worker hangs.

Two independent failures came out of that:

1. **Wedged worker.** One inference worker's queue is corrupted and its
   searches hang forever.
2. **Host cascade.** The app container runs with **no init / PID-1 reaper**.
   When a child inference process died via the device-side abort it became an
   **unreapable zombie**. Enough zombies wedged `dockerd`, which cascaded to a
   host outage — and, because the container has no published port and rides a
   Cloudflare tunnel, the site went to 502.

The fixes are layered so each failure is contained at the lowest level that can
handle it.

## The layers

### 1. App level — worker recycle + shorter move timeout (server code)

Already committed in `server/showcase/bots.py` + `config.py`. A worker whose
search exceeds the move timeout is treated as wedged, killed, and replaced.
This heals failure (1) — a corrupted-queue worker — **while uvicorn stays up**,
so the site keeps answering. This is the primary, fast path.

### 2. Container level — PID-1 reaper (`init: true`)

`docker-compose.yml` sets `init: true` on the `app` service. Compose injects
**tini** as PID 1, which reaps any child that exits — including a worker killed
by a device-side abort. Zombies can no longer accumulate, so failure (2) — the
dockerd wedge / host cascade — is now **structurally impossible**, regardless
of how the worker dies.

This is the single most important fix: it converts a host-outage-class failure
into a routine, self-healed process death.

### 3. Web level — healthcheck + autoheal auto-recovery

- The `Dockerfile` (and `Dockerfile.xpu`) define a `HEALTHCHECK` that curls
  `http://127.0.0.1:8000/healthz` every 30s (5s timeout, 3 retries, with a
  start-period grace). `/healthz` is served by the single uvicorn web process.
- **Plain Docker does not restart an "unhealthy" container** — the healthcheck
  only flips the status. So the compose stack adds a tiny **`autoheal`
  sidecar** (`willfarrell/autoheal`) that watches healthchecks and restarts any
  container labelled `autoheal=true`. The `app` service carries that label.
- Scope is deliberately narrow: `AUTOHEAL_CONTAINER_LABEL=autoheal` means it
  only ever restarts opted-in containers, never the whole host. The Docker
  socket is mounted **read-only**.

This heals the case the app-level recycle cannot: the **uvicorn web process
itself** dying or hanging (so `/healthz` stops answering).

> **Coverage gap to be aware of.** The web process never imports torch and the
> model workers are a separate pool. If *only* a worker is wedged but uvicorn
> still answers `/healthz`, the container stays **healthy** and autoheal will
> not act — that case is owned by layer 1 (worker recycle). Autoheal is the
> safety net for a dead/hung web process, not a substitute for the recycle.

### Restart policy & resource caps (unchanged)

`restart: unless-stopped` and `mem_limit: 6g` are kept as-is. XPU device binds
(`/dev/dri`), `group_add`, and worker settings are unchanged.

## Operator runbook — "site 502 / worker wedged"

1. **Site returns 502 / games hang.** First check the app is up and healthy:
   ```sh
   docker compose ps
   docker inspect --format '{{.State.Health.Status}}' <app-container>
   ```
2. **App is `healthy` but games hang** → a worker is wedged and the recycle
   should already be replacing it. Watch the logs for the "falling back" /
   worker-recycle messages:
   ```sh
   docker compose logs --tail=100 app
   ```
   If it does not self-recover within a couple of move timeouts, restart just
   the app:
   ```sh
   docker compose restart app
   ```
3. **App is `unhealthy`** → autoheal should restart it within ~15s. If it does
   not (autoheal itself down), restart manually:
   ```sh
   docker compose restart app          # or: docker compose up -d
   ```
4. **`docker` / `dockerd` unresponsive** → this is exactly the cascade the
   `init: true` reaper prevents. On the current (hardened) stack this should no
   longer happen. If you are on an *old* container built before the reaper fix,
   rebuild so tini is PID 1:
   ```sh
   docker compose up -d --build app
   ```
5. **Last resort — host / LXC wedged.** Restart the container/LXC (CT restart).
   The DB is on the `showcase-db` named volume and survives a restart. After
   the host is back, redeploy is **gated on human review** (see repo policy).

## Deploy note for `docker-compose.local.yml`

The live host uses a site-local `docker-compose.local.yml` that is **not in the
repo**. The `init: true` reaper, the `autoheal=true` label on `app`, and the
`autoheal` sidecar service live in the committed `docker-compose.yml`, so they
apply on any invocation that includes it. **No change is required in the local
override** for these to take effect — but if the local override redefines the
`app` service's `labels:`, mirror `autoheal: "true"` there so it is not dropped,
and confirm the override does not set `init: false`.
