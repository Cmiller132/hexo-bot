# Shrimp — a Hexo bot — game server

Public-facing FastAPI service: play Hexo against Shrimp checkpoints at
several strengths, review finished games with model analysis, every game
saved to SQLite. Phase 1 of the showcase — the server core; the static web
frontend lands in `web/` as phase 2.

## Layout

```
apps/showcase/
  server/showcase/     python package (the service)
    config.py          SHOWCASE_* env settings
    db.py              SQLite schema (bots/games/analysis_cache + stats views)
    game.py            engine session, turn phases, .hxr record encode/decode,
                       winning-line scan, DB-backed finished snapshots
    bots.py            bots.toml catalogue + worker process pool (search jobs)
    analysis.py        net-only policy/value/stv/moves-left readout, small
                       searched eval, whole-game summary series
    app.py             the API surface, rate limits, idle sweeper
  tests/               pytest suite (self-contained: generates its own tiny
                       checkpoint + bots.toml)
  bots.example.toml    catalogue config example / local-dev default
  requirements.txt     fastapi + uvicorn (on top of the repo's model stack)
```

The server imports `hexo_engine`, `shrimp`, and `hexo_utils` only. Model
inference runs in `SHOWCASE_WORKERS` spawned worker processes, each holding
the full catalogue resident; the web process never imports torch.

## Docker

The production deployment is containerized: a multi-stage `Dockerfile`
(maturin-built extension wheels + CPU torch, serves `web/` statically) and a
`docker-compose.yml` that pairs the server with a Cloudflare tunnel.
Checkpoints are a mounted volume, never baked into the image. Note the build
context is the REPO ROOT (`docker build -f apps/showcase/Dockerfile .`);
compose is already set up that way. First-run walkthrough, model-dir layout,
and the ops runbook (update / logs / backup / kill switch):
[`deploy/README.md`](deploy/README.md).

An opt-in Intel-GPU variant (`Dockerfile.xpu` + `docker-compose.xpu.yml`)
runs inference on torch's native XPU backend; device selection is the
`SHOWCASE_DEVICE` env (`auto | cpu | xpu | cuda`, worker-side, with a startup
CPU-vs-device parity self-check and automatic cpu fallback). See the
"GPU (XPU) deployment" section of [`deploy/README.md`](deploy/README.md).

## Running locally

From the repo root, in an env with the repo's extensions built (see the main
README) plus `pip install -r apps/showcase/requirements.txt`:

```bash
export PYTHONPATH=$PWD/packages/shrimp/python:$PWD/apps/showcase/server
export SHRIMP_SUPPORT_RADIUS=4          # main_7 weights; also the default
python -m pytest apps/showcase/tests -q   # generates tests/data/tiny_bot.pt
# The default catalogue serves the tiny test checkpoint, whose smoke arch must
# come from env (its CCA trunk is not inferable from the state dict):
SHRIMP_CHANNELS=32 SHRIMP_ATTENTION_HEADS=4 SHRIMP_TRUNK=CCA \
  uvicorn showcase.app:app --port 8123
```

The default `SHOWCASE_BOTS_TOML` (`apps/showcase/bots.example.toml`) serves the
tiny random-weight test checkpoint the suite materializes — good for wiring; it
plays legal but weak moves. Point entries at real inference exports (see the
commented example in the toml) for actual strength; those need no arch env —
width/heads/trunk are inferred per checkpoint (production trunk layouts only).
All `SHOWCASE_*` knobs and their defaults are in `server/showcase/config.py`.

`bots.toml` is a CATALOGUE: `[[checkpoint]]` tables (id, checkpoint path,
label, run, epoch, plus optional scalar display metadata such as
`games_trained`) and one global allowed search-budget array
(`sims = [16, 64, 256, 512]` by default). A playable bot is any
(checkpoint, sims) combination, chosen per game. The DB `bots` table gets one
row per PLAYED combination — identity `(slug, weights_sha, visits)`, created
lazily on the first game — so the stats views report per-strength numbers.

## Tests

```bash
PYTHONPATH=packages/shrimp/python:apps/showcase/server \
  python -m pytest apps/showcase/tests -q
```

The suite builds a smoke-size net (env arch `SHRIMP_CHANNELS=32`,
`SHRIMP_ATTENTION_HEADS=4`, `SHRIMP_TRUNK=CCA`,
`SHRIMP_SUPPORT_RADIUS=4`, pinned in `tests/conftest.py`), spins up the real
worker pool once, and drives full games over HTTP.

## API

The frontend track builds against this section.

```
POST /api/game                  {checkpoint_id, sims, human_color} -> state + cookie
GET  /api/game/{id}             state: owner-only while active; PUBLIC once finished
POST /api/game/{id}/move        {q, r}    (cookie-gated)
POST /api/game/{id}/resign                (cookie-gated)
POST /api/game/{id}/nickname    {nickname} after finish (cookie-gated)
GET  /api/game/{id}/analysis    ?ply=N[&search=1]  finished games, cached, public
GET  /api/game/{id}/summary     per-ply value/stv/moves_left series, cached, public
GET  /api/games                 recent finished games feed (public, paginated)
GET  /api/bots                  catalogue metadata (checkpoints + allowed sims)
GET  /api/stats                 win rates / daily activity / hall of fame
GET  /healthz                   liveness
```

Access model: mutating routes always require the session cookie; reading an
ACTIVE game requires it too (403 otherwise). FINISHED games are public by
default — readable by id with no cookie (the feed and shareable URLs depend on
this), served from the live session while it exists and reconstructed from the
DB record afterwards. Public reads of finished games ride the analysis token
bucket; global and per-IP active-game caps plus token buckets (keyed by
`CF-Connecting-IP`, falling back to the peer address) guard game creation,
moves, and analysis. 429 beyond caps.

### POST /api/game

`{"checkpoint_id": "main7-ep75", "sims": 64, "human_color": 0}`. The
checkpoint must be in the catalogue (404 otherwise) and `sims` must be in the
allowed set from `GET /api/bots` (422 otherwise). Returns the game-state
payload below and sets the `showcase_token` httpOnly cookie.

`human_color` is `0` (human moves first, blue), `1` (human moves second,
red), or `"random"` (the server flips a coin); the resolved 0/1 is echoed as
`human_color` in every game-state payload. With the human as player 1 the bot
owns the opening move — the create response arrives as `bot_thinking` and the
client polls until the bot's first stone lands.

### GET /api/bots

```json
{
  "checkpoints": [
    {"id": "main7-ep75", "label": "Shrimp main_7", "run": "shrimp_main_7",
     "epoch": 75, "games_trained": 24000000}
  ],
  "sims": [16, 64, 256, 512]
}
```

Extra scalar keys on a `[[checkpoint]]` table (like `games_trained` above)
pass through verbatim as display metadata.

### Game state (POST /api/game, GET /api/game/{id}, move/resign responses)

```json
{
  "id": "…uuid…",
  "status": "your_turn | bot_thinking | finished",
  "bot": {"checkpoint_id": "main7-ep75", "label": "Shrimp main_7",
          "epoch": 75, "sims": 64},
  "human_color": 0,
  "to_move": 1,
  "phase": "FirstStone",
  "stones_left_this_turn": 2,
  "ply": 7,
  "stones": [{"q": 0, "r": 0, "color": 0}, …],
  "legal": [{"q": 1, "r": 0}, …],
  "last_move": {"q": 2, "r": 1, "color": 1},
  "winning_line": [{"q": 0, "r": 0}, …],
  "result": {"winner": 0, "termination": "six_in_line", "human_result": 1},
  "nickname": null
}
```

- `stones` is in PLACEMENT ORDER (ply order) — the client derives the
  last-two-move marks from its tail. `last_move` equals `stones[-1]`.
- `winning_line` is non-null only when the game finished by `six_in_line`:
  the cells of the completed line, ordered along its axis. Normally exactly
  six cells; a placement that joins two runs can make it longer.
- `legal` is only populated when `status == "your_turn"`; `result` is null
  until the game finishes. DB-served finished games additionally carry
  `finished_at`.

### GET /api/game/{id}/analysis?ply=N[&search=1]

Model readout for the position AFTER ply N (N in 0..ply_count), finished
games only (409 while active), cached per (game, ply):

```json
{
  "game_id": "…", "checkpoint_id": "main7-ep75", "cached": true,
  "ply": 12, "to_move": 0,
  "value": 0.31, "stv": 0.18, "moves_left": 41.5,
  "legal_count": 58,
  "policy": [{"q": 1, "r": 2, "p": 0.412}, …],
  "top_k": [… first 5 of policy …],
  "search": {"visits": 64, "root_value": 0.27, "best": {"q": 1, "r": 2},
             "visit_policy": [{"q": 1, "r": 2, "p": 0.55}, …]},
  "v": 2
}
```

- `value` and `stv` are side-to-move perspective, in [-1, 1]. `stv` is the
  model's shortest-horizon short-term-value head (`stvalue_2`: expected value
  two plies ahead); the value/stv gap reads as imminent swing. `moves_left`
  is the moves-left head decoded to expected remaining plies.
- `search` appears only with `?search=1` (small searched eval at the
  `SHOWCASE_ANALYSIS_VISIT_CAP` budget) and upgrades the cached payload.
- `v` is the payload schema version; bumping it server-side invalidates older
  cached payloads (they are recomputed on first read, never served stale).

### GET /api/game/{id}/summary

Whole-game series for the value/ply chart, finished games only, public.
Computed lazily on the first request — one chunked batched forward over every
position of the game (cheap on CPU: a full game is a few search-batch
equivalents) — then cached in `analysis_cache` (ply = -1 slot):

```json
{
  "game_id": "…", "checkpoint_id": "main7-ep75", "cached": false,
  "ply_count": 34,
  "value": [0.0, …], "stv": [0.0, …], "moves_left": [61.2, …],
  "to_move": [0, 1, 1, …, null],
  "v": 2
}
```

Arrays have `ply_count + 1` entries; index i is the position AFTER ply i
(index 0 = empty board, last index = final position). `value`/`stv` are
side-to-move perspective at each index — use `to_move` (null at a terminal
position) to fold into a fixed-color perspective for charting.

### GET /api/games — public recent-games feed

`?limit=` (default 20, max 50) and `?before=` (opaque cursor). Finished games
only, newest first; keyset pagination on (finished_at, id) so same-second
finishes never skip or duplicate:

```json
{
  "games": [
    {
      "id": "…uuid…",
      "bot": {"checkpoint_id": "main7-ep75", "label": "Shrimp main_7",
              "epoch": 75, "sims": 64},
      "human_color": 0,
      "result": {"winner": 1, "termination": "resign", "human_result": -1},
      "ply_count": 18,
      "finished_at": "2026-07-05T12:34:56+00:00",
      "nickname": null
    }
  ],
  "next": "2026-07-05T12:34:56+00:00~…uuid…"
}
```

`next` is null when there are no further pages; pass it back as `?before=`.

## Search behavior

Bot moves run each checkpoint's as-trained search profile: everything except
the per-game visit budget (the chosen `sims`) is parsed from a profile TOML's
`[model.config.selfplay]` section. Checkpoints without a `search_profile` key
share the global default (`SHOWCASE_SEARCH_CONFIG`, default
`configs/shrimp_main_7.toml`, the gumbel profile). Opening plies are
temperature-sampled like the eval arena so games vary; later moves are
greedy. Analysis `?search=1` runs the position's checkpoint profile at a
capped budget (`SHOWCASE_ANALYSIS_VISIT_CAP`, default 64).

### Per-checkpoint search profiles (legacy PUCT bots)

A `[[checkpoint]]` entry may set `search_profile` to serve that checkpoint
with the search it trained under. Resolution order: a bare name
(`"shrimp_main_5"`) resolves against the built-in profiles dir
(`apps/showcase/profiles/`, shipped in the server image); a path resolves
relative to the bots.toml directory; absolute paths pass through. A missing
profile file fails catalogue load at startup, not the first move. Each worker
parses one `SearchProfile` per unique profile.

Shipped profiles: `shrimp_main_4`, `shrimp_main_5` (PUCT era; the Gumbel
flags default off, selecting the plain PUCT root) and `shrimp_main_7` (the
same values as the global default, for catalogues that name every profile
explicitly). The PUCT profiles are distilled to the knobs this repo's strict
config parser knows; every training-run knob missing from that set (root
Dirichlet noise, forced playouts, the zeroed visit-scaled c_puct term) is
inert under eval/serve conditions, so the distilled profile reproduces the
training repo's eval-arena search exactly.

Two display keys support legacy entries in the picker: `group` (section
heading, e.g. `group = "earlier runs"`; ungrouped entries form the first
section) and `search = "puct"` (renders the PUCT-search tag). Both ride the
normal metadata passthrough of `GET /api/bots`.

Example entry:

```toml
[[checkpoint]]
id = "main5-ep105"
checkpoint = "models/shrimp_main5_infer.pt"
label = "Shrimp main_5"
run = "shrimp_main_5"
epoch = 105
search_profile = "shrimp_main_5"
group = "earlier runs"
search = "puct"
```
