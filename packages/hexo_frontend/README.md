# hexo_frontend

Web dashboard for the Hexo RL training project. A single stdlib
`ThreadingHTTPServer` (no web framework) serves one static SPA bundle with
three screens:

- **Match** (`#match`) -- live Arena: manual / SealBot / checkpoint bots play
  real games through `hexo_runner`.
- **History** (`#history`) -- training-run status: live status band (2.5s
  poll), per-epoch trends/table/inspector, paged game history.
- **Debug** (`#debug?run=..&ply=..`) -- single-position model forensics
  (policy/value heads, MCTS search, training-row targets, per-ply sweeps),
  backed by a CPU-only torch worker subprocess so the HTTP server itself never
  imports torch or touches the training GPU.

Launch it with `scripts/dashboard.sh` (serves :8080, cwd at the repo root by
default). Run discovery is cwd-relative by design: it scans `runs/` under the
server's cwd (see Run discovery below).

## Modules

| File | Role |
| --- | --- |
| `python/hexo_frontend/web.py` | The core: HTTP route table (`do_GET`/`do_POST`), `ManualMatchController` (threaded live match/series bridge between browser clicks and `hexo_runner` players, incl. checkpoint bots played via the debug worker), training-run scanning/caching for `/api/training/*`, and glue for the `/api/debug/*` endpoints. Entry points `run()` / `main()`. |
| `python/hexo_frontend/dashboard.py` | Pure shaping layer: `PythonHexoState` mirror -> browser JSON payload (placements, legal moves, winner, window-tactics block). Called only from web.py. |
| `python/hexo_frontend/debug_service.py` | Server-side manager for the Debug worker: lazily spawns `debug_worker` as a child process, serializes requests behind a lock (NDJSON, timeouts, auto-restart on transport failure, LRU result cache). Module singleton via `get_worker()`. |
| `python/hexo_frontend/debug_worker.py` | Child-process main loop: one JSON request per stdin line, dispatches ops (`ping`/`info`/`analyze`/`search`/`search_tree`/`record_row`/`game_eval`/`reeval`) to debug_infer, one JSON response per stdout line. LRU checkpoint cache. Launched by debug_service as `python -m hexo_frontend.debug_worker`. |
| `python/hexo_frontend/debug_infer.py` | CPU-only inference library: rebuilds `ShrimpNet` from a checkpoint's state-dict + run manifest, replays action-id sequences into engine states, and returns a uniform debug schema (priors, distributional value, aux heads, fresh MCTS, pure-Python PUCT debug tree, `.npz` training-row decode, per-ply game-eval sweeps). Only importer is debug_worker (+ tests). |
| `python/hexo_frontend/static/app.js` | ~8.5k-line single-file SPA holding all three screens (`mt*` Match, `hist*` History, `dbg*` Debug prefixes) plus the top diag/error bar and `APP_VERSION`. |
| `python/hexo_frontend/static/index.html` | Single page hosting all three screens; references `styles.css` and `app.js` with the `?v=` cache-bust token. |
| `python/hexo_frontend/static/styles.css` | ~3.2k-line dark-theme stylesheet for all three screens. |
| `python/hexo_frontend/__init__.py` | Version stub only. |
| `pyproject.toml` | Deps `hexo-engine` + `hexo-runner`; registers the `hexo-play` console script -> `hexo_frontend.web:main`. |

## Connections to other packages

**Imports out:**

- `hexo_runner` -- the Match screen plays real games through the production
  runner: `SealBotPlayer`/`SealBotConfig`/`discover_sealbot_adapters`
  (adapters.sealbot), `run_match` (modes.match), the player protocol types
  (`DecisionResult`, `PlayerIdentity`, `WorkerContext`, ...),
  `GameResult`/`HexoRecordFile` (records), `GameSpec` (session).
- `hexo_engine` -- web.py and debug_infer use `HexoState`, `PlacementAction`,
  `is_legal_action`, `to_python_state`, and `pack_coord_id`/`unpack_coord_id`.
  app.js re-implements the same coord<->action-id packing client-side
  (`DBG_COORD_OFFSET` 32768) -- a pinned cross-language contract.
- `shrimp` (lazily, inside the debug worker only): rebuilds `ShrimpNet` from
  the checkpoint state dict + run manifest to serve the arena and debug screens.

**Imports in:** none -- no package imports hexo_frontend. It is exercised by the
browser, by tests, and by `scripts/dashboard.sh` (which puts
`packages/hexo_frontend/python` and `packages/shrimp/python` on `PYTHONPATH`).

**Shared on-disk formats (read-only):** produced by hexo_train + Shrimp:

- `manifest.json` (lineage + architecture), `diagnostics/*.json`
  (`shrimp.selfplay.live.json`, `shrimp.selfplay.epoch_*.json`,
  `shrimp.multistage_eval.epoch_*.json`), `events.jsonl` tails (live status).
- `checkpoints/*.pt` (epoch filename regex `epoch_?NNN.pt`).
- `.hxr` game records via `hexo_runner.records.HexoRecordFile`.
- Self-play per-game `.npz` training-row shards (Targets tab decode).

## Run discovery

Training endpoints scan `runs/` under the server's cwd (`scripts/dashboard.sh`
runs with cwd at the repo root, so `runs/*` shows up automatically). A directory
appears as a run when it contains a `diagnostics/` or `selfplay/` child;
checkpoints are listed from `<run>/checkpoints/*.pt`. Debug endpoints
additionally honor `HEXO_DEBUG_RUN_ROOT`.

## Debug-worker protocol

- web.py never imports torch; all model inference rides the worker subprocess.
- debug_service <-> debug_worker: newline-delimited JSON over stdin/stdout,
  `{id, op, ...}` -> `{id, ok, result|error}`. stdout carries ONLY protocol
  JSON (diagnostics go to stderr).
- Env overrides: `HEXO_DEBUG_WORKER_CMD`, `HEXO_DEBUG_RUN_ROOT` (plus the
  Windows-host WSL launch knobs `HEXO_DEBUG_USE_WSL` / `HEXO_DEBUG_WSL_PYTHON`,
  unused on a native Linux host).
- Error taxonomy: `DebugWorkerError`/timeout = 500 (retryable, transport);
  `DebugRequestError` = 400 (deterministic, bad request).

## Static assets

- index.html is served `no-store`; `app.js`/`styles.css` are served
  `no-cache` + strong ETag (web.py `_send_static`).
- The `?v=` cache-bust token (on both asset references in index.html) and
  `APP_VERSION` in app.js are bumped in lockstep on every static change: the
  token defeats intermediary caches and keeps the on-screen version tag matched
  to the served bundle.

## Entry points / how it gets exercised

- `scripts/dashboard.sh [PORT]` (default 8080) launches it detached with the
  right `PYTHONPATH` and `SEALBOT_PATH` passthrough.
- `python -m hexo_frontend.web --host --port --sealbot-path` / the `hexo-play`
  console script (same `main()`).
- Browser: `/` serves index.html; screens are addressed by URL hash
  (`#match` / `#history` / `#debug?...`).
- HTTP routes: GET `/api/state`, `/api/adapters`,
  `/api/training/{runs,run,live,epoch,epochs,history,history-page,
  history-count,artifacts-page}`, `/api/debug/{checkpoints,games,trajectory,
  position,ckpt_info,record_row,game_eval,attention}`; POST `/api/new`,
  `/api/match/stop`, `/api/move`, `/api/debug/{analyze,search,search_tree}`.
- Tests: `tests/test_hexo_runner_match_mode.py`, `tests/test_sealbot_adapter.py`
  (ManualMatchController / player-spec normalization),
  `tests/test_frontend_training_{artifacts,epoch,live}.py` (training scan
  functions), `tests/test_debug_infer.py` (debug_service + debug_infer + web
  debug endpoints). Run with `python -m pytest tests/<file> -q` from the repo
  root with the venv active.
