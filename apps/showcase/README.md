# Hexo bot showcase — game server

Public-facing FastAPI service: play Hexo against hexfield checkpoints at
several strengths, review finished games with model analysis, every game
saved to SQLite. Phase 1 of the showcase — the server core; the static web
frontend lands in `web/` as phase 2.

## Layout

```
apps/showcase/
  server/showcase/     python package (the service)
    config.py          SHOWCASE_* env settings
    db.py              SQLite schema (bots/games/analysis_cache + stats views)
    game.py            engine session, turn phases, .hxr record encode/decode
    bots.py            bots.toml ladder + worker process pool (search jobs)
    analysis.py        net-only policy/value readout + small searched eval
    app.py             the API surface, rate limits, idle sweeper
  tests/               pytest suite (self-contained: generates its own tiny
                       checkpoint + bots.toml)
  bots.example.toml    ladder config example / local-dev default
  requirements.txt     fastapi + uvicorn (on top of the repo's model stack)
```

The server imports `hexo_engine`, `hexfield`, and `hexo_utils` only. Model
inference runs in `SHOWCASE_WORKERS` spawned worker processes, each holding
the full ladder resident; the web process never imports torch.

## Running locally

From the repo root, in an env with the repo's extensions built (see the main
README) plus `pip install -r apps/showcase/requirements.txt`:

```bash
export PYTHONPATH=$PWD/packages/hexfield/python:$PWD/apps/showcase/server
export HEXFIELD_SUPPORT_RADIUS=4          # main_7 weights; also the default
python -m pytest apps/showcase/tests -q   # generates tests/data/tiny_bot.pt
# The default ladder serves the tiny test checkpoint, whose smoke arch must
# come from env (its CCA trunk is not inferable from the state dict):
HEXFIELD_CHANNELS=32 HEXFIELD_ATTENTION_HEADS=4 HEXFIELD_TRUNK=CCA \
  uvicorn showcase.app:app --port 8123
```

The default `SHOWCASE_BOTS_TOML` (`apps/showcase/bots.example.toml`) serves the
tiny random-weight test checkpoint the suite materializes — good for wiring; it
plays legal but weak moves. Point entries at real inference exports (see the
commented example in the toml) for actual strength; those need no arch env —
width/heads/trunk are inferred per checkpoint (production trunk layouts only).
All `SHOWCASE_*` knobs and their defaults are in `server/showcase/config.py`.

## Tests

```bash
PYTHONPATH=packages/hexfield/python:apps/showcase/server \
  python -m pytest apps/showcase/tests -q
```

The suite builds a smoke-size net (env arch `HEXFIELD_CHANNELS=32`,
`HEXFIELD_ATTENTION_HEADS=4`, `HEXFIELD_TRUNK=CCA`,
`HEXFIELD_SUPPORT_RADIUS=4`, pinned in `tests/conftest.py`), spins up the real
worker pool once, and drives full games over HTTP.

## API

```
POST /api/game                  {bot_id, human_color}  -> state + session cookie
GET  /api/game/{id}             poll state: your_turn | bot_thinking | finished
POST /api/game/{id}/move        {q, r}    (cookie-gated)
POST /api/game/{id}/resign                (cookie-gated)
POST /api/game/{id}/nickname    {nickname} after finish (cookie-gated)
GET  /api/game/{id}/analysis    ?ply=N[&search=1]  finished games, cached
GET  /api/bots                  ladder metadata
GET  /api/stats                 win rates / daily activity / hall of fame
GET  /healthz                   liveness
```

Abuse control: per-client cookie token on mutating routes; global and per-IP
active-game caps plus token buckets (keyed by `CF-Connecting-IP`, falling back
to the peer address) on game creation, moves, and analysis. 429 beyond caps.

## Search behavior

Bot moves run the as-trained main_7 gumbel search profile: everything except
the per-rung visit budget is parsed from `SHOWCASE_SEARCH_CONFIG` (default
`configs/hexfield_main_7.toml`, section `[model.config.selfplay]`). Opening
plies are temperature-sampled like the eval arena so games vary; later moves
are greedy. Analysis `?search=1` runs the same profile at a capped budget
(`SHOWCASE_ANALYSIS_VISIT_CAP`, default 64).
