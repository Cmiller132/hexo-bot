# Showcase deployment — ops runbook

The stack is two containers (`docker-compose.yml` one directory up): `app`
(the game server, CPU inference by default — see "GPU (XPU) deployment" below
for the Intel Arc variant) and `cloudflared` (a Cloudflare tunnel that
carries all public traffic — no host port is published).

Everything below runs from `apps/showcase/` on the deploy machine.

## First run

1. **Host**: any Linux box/VM/LXC with Docker + the compose plugin
   (LXC needs `nesting=1,keyctl=1`). ~8 cores / 8 GB RAM is comfortable for
   the default caps (`cpus: 7`, `mem_limit: 6g`).

2. **Secrets** — copy the template and fill it in:

   ```bash
   cp .env.example .env && chmod 600 .env
   ```

   - `TUNNEL_TOKEN`: Cloudflare dashboard → Zero Trust → Networks → Tunnels →
     create a tunnel → copy the token from the docker connector command.
     Add a public hostname on the tunnel pointing at `http://app:8000`.
   - `SHOWCASE_IP_SALT`: `openssl rand -hex 32` (stable salt for the hashed
     client IPs stored in the DB).

3. **Models** — `deploy/models/` is mounted read-only at `/models`:

   ```
   deploy/models/
     bots.toml            # catalogue (start from ../bots.example.toml)
     main7_latest.pt      # hexfield inference export(s)
     ...                  # past-epoch entries, immutable filenames
   ```

   `checkpoint` paths inside `bots.toml` resolve relative to the file, so
   entries reference the `.pt` files by bare filename. All checkpoints must
   match the support radius the server runs at
   (`HEXFIELD_SUPPORT_RADIUS=4` in compose); width/heads/trunk are inferred
   per checkpoint from its state dict.

4. **Launch**:

   ```bash
   docker compose up -d --build
   docker compose logs -f app        # wait for "Uvicorn running"
   ```

   The first build compiles the Rust extension crates and downloads CPU
   torch — expect roughly 10–20 minutes cold. Rebuilds reuse the layer cache
   and are much faster.

   For a local smoke test without the tunnel, uncomment the
   `127.0.0.1:8000:8000` ports block in `docker-compose.yml` and hit
   `http://127.0.0.1:8000/healthz`.

## Update (code)

```bash
git pull && docker compose up -d --build
```

(GPU deployments: append the override file to every `docker compose` command —
see below.)

## GPU (XPU) deployment — Intel Arc

Opt-in variant that runs hexfield inference on an Intel GPU via torch's
native XPU backend (`Dockerfile.xpu` + `docker-compose.xpu.yml`). The CPU
stack remains the default; nothing below is required for it. Ship the GPU
variant only if it benchmarks faster than CPU at showcase batch sizes.

### Prerequisites

1. **Host GPU visible in the container's host** (for an LXC: bind `/dev/dri`
   into the container in the Proxmox CT config — on this deployment that is
   already done; the A310 is `card1` + `renderD128`).

2. **Render group gid** — the container's app user needs the gid that owns
   the render node *inside* the LXC:

   ```bash
   stat -c %g /dev/dri/renderD128
   ```

   Put the number in `.env` as `RENDER_GID=<gid>` (see `.env.example`).
   No GPU driver install is needed on the LXC itself beyond the kernel module
   the Proxmox host already loads — the user-mode stack (level-zero loader +
   Intel compute runtime) lives inside the image.

### Launch

```bash
docker compose -f docker-compose.yml -f docker-compose.xpu.yml up -d --build
```

The override swaps the build to `Dockerfile.xpu` (torch `2.12.1+xpu` from
`https://download.pytorch.org/whl/xpu` on an ubuntu-24.04 base with Intel's
level-zero/compute-runtime packages), passes `/dev/dri` through, adds the
render gid, and sets `SHOWCASE_DEVICE=xpu`.

### Confirm the card is seen

```bash
# torch sees the GPU:
docker compose exec app python3 -c \
  "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
# expected: True Intel(R) Arc(TM) A310 Graphics

# lower-level check if that prints False:
docker compose exec app clinfo -l

# the workers picked it up (one line per worker):
docker compose logs app | grep "worker ready"
# expected: device=xpu (requested 'xpu')
```

At startup each worker runs a CPU-vs-XPU parity self-check on a fixed
position (`SHOWCASE_DEVICE_SELFCHECK`, on by default off-cpu). If the check
fails, the worker logs `DEVICE SELF-CHECK FAILED ... FALLING BACK TO CPU` and
serves on cpu — correct moves always win over fast moves. Grep the log for
`self-check` after first launch.

### The SHOWCASE_DEVICE knob

`SHOWCASE_DEVICE` (in `server/showcase/config.py`, settable via compose
`environment:`): `auto` (default; prefers xpu, then cuda, then cpu) | `cpu` |
`xpu` | `cuda`. Explicit accelerator requests fall back to cpu with a logged
warning when unavailable. The override file pins `xpu` so an accidental
CPU fall-through is visible in the logs rather than masked by `auto`.

### Benchmark before shipping

Play a few games / hit `/api/game/{id}/summary` on both stacks and compare
move latency at the deployed visit budgets (or time a scripted game). The
A310 runs hexfield's *eager* fp32 paths (the fast fused kernels are
CUDA-only), so XPU is not automatically a win over 7 modern CPU cores —
measure, then keep whichever is faster. If XPU wins big, raise
`SHOWCASE_MAX_ACTIVE_GAMES`/`SHOWCASE_WORKERS`.

### Reverting to CPU

```bash
docker compose -f docker-compose.yml -f docker-compose.xpu.yml down
docker compose up -d --build
```

## Refresh the model ladder

Drop the new `.pt` into `deploy/models/`, update `bots.toml` if a catalogue
entry changes (the "latest" entry keeps a stable filename so it usually
doesn't), then:

```bash
docker compose restart app
```

Keep past-epoch files immutable — finished games reference the bot identity
they were played against.

## Logs

```bash
docker compose logs -f app          # server log
docker compose logs -f cloudflared  # tunnel connectivity
```

## Database

SQLite lives in the `showcase-db` named volume, mounted at `/data`.

```bash
# shell
docker compose exec app sqlite3 /data/showcase.db

# consistent hot backup (safe while the server is running)
docker compose exec app sqlite3 /data/showcase.db ".backup /data/showcase-$(date +%F).db"

# pull a backup out of the volume onto the host
docker compose cp app:/data/showcase-$(date +%F).db ./
```

Schedule the `.backup` line from cron/systemd on the host for nightly
backups, and prune old copies in `/data` occasionally.

## Kill switch

```bash
docker compose down
```

The tunnel drops with the stack, so the public hostname 502s at Cloudflare's
edge immediately. `docker compose up -d` brings everything back; game state
and history are in the `showcase-db` volume and survive.

## Notes / troubleshooting

- The `app` container runs read-only and non-root: only `/data` (volume),
  `/tmp` (tmpfs) and `/dev/shm` are writable. If a debugging session needs to
  write elsewhere, comment out `read_only: true` temporarily.
- `SHOWCASE_*` knobs (rate limits, worker count, timeouts) are all in
  `server/showcase/config.py`; override them in the compose `environment:`
  block.
- Recommended Cloudflare edge extras: a rate-limiting rule on `/api/*`
  (the app also enforces its own), Bot Fight Mode, caching for static
  assets.
