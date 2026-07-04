# History Screen v2 — PHASE 2 Spec Addendum (Epoch Inspector + Game Panel + /api/training/epoch)

Status: FINAL. Binding contract for the three phase-2 implementation agents.
Builds ON TOP of the phase-1 spec (`docs/specs/history_screen_v2_spec.md`) — every phase-1
rule (frozen helpers §3, ids, "--" conventions, node --check, no commits/restarts)
still applies. All field names below were verified against `web.py`,
`debug_worker.py`, app.js, and a real run dir (a run under `runs/`) on 2026-06-11.
Where this file and the design draft disagree, THIS FILE wins.

Execution order (sequential): BACKEND agent (P3.*) → FRONTEND agent 1 (P1.*) →
FRONTEND agent 2 (P2.*). Each agent leaves the tree non-broken.

Targets per agent:
- BACKEND: `packages/hexo_frontend/python/hexo_frontend/web.py` (new route + helper
  only) and ONE new test file `tests/test_frontend_training_epoch.py`. HANDS OFF
  `debug_service.py` / `debug_worker.py` / every `_debug_*` helper — the only debug
  helper that may be CALLED (never modified) is `_debug_resolve_run_path` (P3.1
  checkpoint fallback). No new dependencies.
- FRONTEND 1 + 2: `static/app.js`, `static/index.html`, `static/styles.css` only.
  All phase-1 frozen lists apply unchanged. The `dbg*` code is out of bounds.

---

## 0. VERIFIED DATA INVENTORY (normative)

### 0.1 What the client ALREADY has (no fetch needed)
`trainingRunDetails[name]` (the run payload, refreshed every 15s) carries
`epoch_history`, `evaluation_history`, `diagnostics_by_epoch`, `learning_health`,
`status`. THEREFORE (draft correction): the inspector's HEADER / LOSSES / GAME STATS /
BUFFER / CALIBRATION / EVAL groups render INSTANTLY from the run payload with zero
fetches, and per-head deltas are computed CLIENT-SIDE from the previous
`epoch_history` row of the same run. The lazy `/api/training/epoch` fetch only fills
the DIAGNOSTICS and MODEL(config) groups + the checkpoint stat line; the lazy
`ckpt_info` fetch fills the deep model card. Three groups appear progressively; the
panel must never block on a fetch.

Per-epoch row fields (phase-1 spec §1.2 verified still accurate):
- `row.selfplay`: `games, completed_games, truncated_games, samples_added,
  searched_positions, mcts_simulations, search_positions_per_second,
  mcts_sims_per_searched_position, elapsed_seconds, game_length_mean/median/max/
  stdev/p95, win_p0_fraction, win_p1_fraction, draw_fraction, decisive_fraction,
  mean_abs_value, buffer` (any may be null).
- `row.selfplay.buffer` — the UNIFORM per-head loss carrier
  (`loss_total, loss_policy, loss_value, loss_opp, loss_stvalue_<h>`; runs with a real
  pool add `samples, cap, window_epochs, window_span, decay, train_steps, train_batch,
  train_samples_per_epoch, optimism_sum_mean`). Some buffers are loss-only —
  guard every key individually (`asFinite`).
- `row.training`: `status, loss, loss_components, steps, samples, batch_size,
  samples_per_second, elapsed_seconds` (+ optional `source_summary`,
  `policy_imitation`, `progress`).
- `row.evaluation`: `status, games, completed, wins, losses, mean_turns`.
- `row.checkpoint`: `{path:"checkpoints/epoch_000001.pt", bytes, modified}` (from
  the checkpoints-dir scan) OR `{path, name}` (from the epoch result).
- `row.samples`: `{buffer_count, window_size, compressed_bytes}` (optional).
- `row.elapsed_seconds`, `row.status`, `row.d6` (optional).

### 0.2 Checkpoint NAME rule (verified — write into client code as stated)
Canonical checkpoint file for epoch N is `checkpoints/epoch_{N:06d}.pt` (verified on
disk: `epoch_000001.pt` …; `_epoch_history` globs `epoch_*.pt`;
`_debug_checkpoints` parses the same pattern). Client-side derivation for the
`/api/debug/ckpt_info?run=&checkpoint=` call:
1. `payload.checkpoint.name` from `/api/training/epoch` when present (preferred);
2. else `row.checkpoint.name || basename(row.checkpoint.path)` from epoch_history;
3. else the canonical fallback `"epoch_" + String(epoch).padStart(6, "0") + ".pt"`.
`_debug_resolve_checkpoint` accepts a BARE FILENAME (it strips any directories) and
resolves under the DEBUG roots (`HEXO_DEBUG_RUN_ROOT` override first, then the cwd
`runs/` + `configs/runs/`), so ckpt_info works even when the dashboard serves
history from a bridge-mirror cwd that lacks checkpoints. `ckpt_info` loads the model
into the single CPU debug worker on first call (seconds), then is cached server-side
(worker result cache + model LRU) — the pending hint in P1.5 is mandatory.

`/api/debug/ckpt_info` response (verified): `{run, checkpoint, size, mtime, meta}`
with `meta = {lineage, rl_epoch, step, graft, candidate_radius, expanded_value,
expanded_stv, zeroed_feature_cols, load_warnings (list), stv_horizons (list),
has_moves_left, moves_left_cap, param_count, arch (display STRING, pre-flattened
server-side)}`. Any meta key may be null.

### 0.3 Raw diagnostics worth surfacing (the P3 inventory result)
`diagnostics/hexfield.selfplay.epoch_{N:06d}.json` (~120KB; the per-epoch self-play
diagnostics file; the `hexfield.` basename prefix is manifest-derived — see
`_diag_prefix` in `web.py`) carries useful stats NOT in `_selfplay_epoch_summary`:
- `temperature_control` `{expected_game_length, halflife_plies, halflife_fraction}` —
  the game-length EMA driving the temperature schedule (an active tuning lever).
- `pcr` `{enabled, full_proportion, fast_visits, full_search_count,
  fast_search_count, fast_rows_excluded}` — full/fast search split.
- `policy_init` `{enabled, fraction, avg_plies, max_plies, temperature, moves}` —
  opening policy-init usage.
- `root_policy_temperature_control` `{base, early, halflife_plies}` — root
  exploration shape.
- `raw_samples` vs `effective_samples`, `total_decisions`,
  `mcts_virtual_batch_size`, `active_games`, `scheduler`,
  `elapsed_seconds` + `mcts_search_elapsed_seconds` (→ search share of selfplay wall).
EXCLUDED on purpose (too deep / memory-internals / huge): `mcts_diagnostics`,
`scheduler_diagnostics`, `npz_writes` (384-entry list), `spill`,
`selfplay_npz_files`. The endpoint must return ONLY the curated subset (this is the
draft's "size cap": curate, never pass the raw payload through).

`manifest.json` (~4.5KB, per-run constant) `model.config` carries the run's actual
knob values — genuinely useful context when inspecting an epoch (arch + the
exploration levers under active study). Curated subset in §P3.1. Eval-diagnostics
extras (`opponent_elapsed_seconds`, `rounds`, and the per-batch forward/decision
counts) were inventoried and REJECTED: marginal value, would not earn
screen space.

### 0.4 Phase-1 structures being extended (verified, current line refs)
- `renderGameHistoryPage` (app.js ~2553): computes `filtered` → `displayed`
  (epoch-filter) → `selected = selectedHistoryItem(histories, displayed)` →
  `visible`; calls `renderHistStatusBand/Trends/EpochTable(runs)` then writes
  list+detail innerHTML. Phase-2 inserts `renderHistEpochInspector(runs)` after
  `renderHistEpochTable(runs)` and captures `histDisplayedKeys` (P2.1) right after
  `displayed` is computed. The function stays null-safe and callable from
  `renderTraining` at any time.
- `renderHistEpochTable` (~3494): rows are `div.hist-epoch-row` containing the
  `data-hist-epoch` epoch button, cells, `data-hist-epoch-toggle` chevron; the
  delegated listener at ~237 handles toggle then epoch-number branches.
- `setHistEpochFilter(epoch)` (~3389): toggle + re-render. NEVER duplicate; the
  inspector's "filter games" button calls it.
- `histThumbCache`/`loadHistThumb` (~3764): the canonical lazy-fetch +
  patch-in-place pattern (cache states "loading"/"ready"/"error", FIFO cap,
  "patch only if still selected"). P1.5's two fetchers MUST mirror it.
- `handleGameHistoryClick` (~2650): four frozen branches; new branches are APPENDED
  after the `data-history-load` branch and before the final `data-history-key`
  fallthrough.
- Keyboard precedent: `dbgHandleKey` is a document-level keydown guarded by
  `activeScreen !== "debug"` early-return + INPUT/SELECT/TEXTAREA guard + modifier
  guard. The new `histHandleKey` follows the identical shape, guarded on
  `activeScreen !== "history"`, so the two can never both fire.
- 15s refresh: `refreshHistoryIfVisible` re-renders via innerHTML — ALL phase-2 UI
  state lives in module vars (`histInspectEpoch`, caches, `histDisplayedKeys`).
- Cache-bust baseline: index.html currently has `?v=20260611-hist1` at BOTH line 7
  (stylesheet) and line 532 (script).

### 0.5 DOM placement decision (binding)
`#histEpochInspector` is a FULL-WIDTH card between the epoch table and the filters
row — in index.html, directly after `<div id="histEpochTable" …>` and before
`<div class="history-filters" …>`:
`<div id="histEpochInspector" class="hist-epoch-inspector" hidden></div>`
Rationale: it is epoch-scoped (belongs visually with the epoch table that opens it),
full width gives the loss bars + 6 groups room as an internal
`repeat(auto-fill, minmax(240px, 1fr))` grid, the frozen `.history-layout`
list/detail grid is untouched, and mobile needs no special-case (already
full-width — B5 satisfied structurally). NOT a third column, NOT an overlay.

---

## P3 — BACKEND agent

### P3.1 `GET /api/training/epoch?run=<name>&epoch=<int>` (new route + `_training_epoch` helper)
Dispatch: a new `elif path == "/api/training/epoch":` branch in `do_GET` next to the
other `/api/training/*` branches, parsing `run` (str) and `epoch`
(`_query_int`), calling `_training_epoch(run_name, epoch)`. Errors follow the
sibling pattern: raise `ValueError` ("Unknown training run" via `_resolve_run_dir`
returning None; "epoch is required" when the epoch param is missing/non-int) and let
the existing do_GET error handling answer 400 `{"error": ...}` exactly as
`/api/training/run` does. An epoch with NO data is NOT an error (nulls, see below).

`_training_epoch(name: str, epoch: int) -> dict` (place near `_training_run`):
```
{
  "run": <run_dir.name>,
  "epoch": <int>,
  "history":     <the _epoch_history(run_dir) row with row["epoch"] == epoch, else None>,
  "prev_epoch":  <the row with the next-lower epoch number, else None>,
  "evaluation":  <the _evaluation_history(run_dir) row for this epoch, else None>,
  "diagnostics": <_diagnostics_by_epoch(run_dir).get(str(epoch)), else None>,
  "selfplay_extras": <curated dict | None>,
  "manifest":    <curated dict | None>,
  "checkpoint":  {"name": str, "size": int, "mtime": float} | None,
}
```
- `selfplay_extras`: read `diagnostics/<prefix>.selfplay.epoch_{epoch:06d}.json`
  via `_read_json_file` (`<prefix>` from `_diag_prefix(run_dir)`, i.e. `hexfield`);
  None when absent/non-dict. Curate EXACTLY:
  top-level passthrough keys `scheduler, raw_samples, effective_samples,
  total_decisions, active_games, mcts_virtual_batch_size, elapsed_seconds,
  mcts_search_elapsed_seconds`; nested dicts `temperature_control`
  (`expected_game_length, halflife_plies, halflife_fraction`), `pcr` (`enabled,
  full_proportion, fast_visits, full_search_count, fast_search_count,
  fast_rows_excluded`), `policy_init` (`enabled, fraction, avg_plies, max_plies,
  temperature, moves`), `root_policy_temperature_control` (`base, early,
  halflife_plies`) — each nested group included only when the source value is a
  dict, each key only when present. Never include `mcts_diagnostics`,
  `scheduler_diagnostics`, `npz_writes`, `spill`, `selfplay_npz_files` (§0.3).
- `manifest`: read `<run_dir>/manifest.json` (`utf-8-sig`, json, dict-guarded; None
  on any failure). Curate from `data["model"]`:
  `{"model_name": model.name, "architecture": <subset of model.config.architecture:
  input_channels, channels, blocks_type, attention_heads,
  short_term_value_horizons, moves_left_head>, "selfplay": <subset of
  model.config.selfplay: search_visits, active_games, c_puct,
  root_dirichlet_noise_fraction, root_dirichlet_total_alpha,
  root_policy_temperature, fpu_reduction, temperature, forced_playout_k>,
  "evaluation": <subset of model.config.evaluation: games_per_epoch, eval_every,
  sealbot_variant>, "training": <subset of model.config.training: batch_size,
  learning_rate, train_samples_per_epoch>}` — every level dict-guarded, missing
  keys simply omitted (manifest shapes vary; never KeyError).
- `checkpoint`: name = the history row's `checkpoint.name` or
  `Path(checkpoint.path).name` when truthy, else the canonical
  `f"epoch_{epoch:06d}.pt"`. Stat `<run_dir>/checkpoints/<name>`; if absent, fall
  back to `_debug_resolve_run_path(name_of_run, f"checkpoints/{ckpt_name}")` (CALL
  only — covers a HEXO_DEBUG_RUN_ROOT that differs from the repo root, §0.2) and stat that.
  None when no file exists anywhere.
- CACHING DECISION: no server-side cache. Unlike `/api/training/run` (polled every
  15s by every client) this endpoint fires once per inspector open (the client
  caches per `(run, epoch)`, P1.5); `_epoch_history`'s .hxr backfill is already
  memoized by mtime/size via `_hxr_base_rows`. State this in the helper docstring.

### P3.2 (reserved — folded into P3.1; no separate item)

### P3.3 Tests — new file `tests/test_frontend_training_epoch.py` (additive only)
Same style as `tests/test_frontend_training_artifacts.py` (sys.path bootstrap
header, `tmp_path` + `monkeypatch.chdir(tmp_path)`, run dir at
`tmp_path / "runs" / <name>`, direct `web._training_epoch(...)` calls). Minimum
cases:
1. Full epoch: write `diagnostics/epoch_000002.json` (with
   `metadata.result.{epoch,selfplay,training,checkpoint}`),
   `diagnostics/hexfield.selfplay.epoch_000002.json` (including a
   `temperature_control` dict AND an `npz_writes` list + `mcts_diagnostics` dict),
   `diagnostics/hexfield.evaluation.epoch_000002.json`, a previous
   `diagnostics/epoch_000001.json`, `checkpoints/epoch_000002.pt` (a few bytes),
   and a `manifest.json` with `model.name` "hexfield" (so `_diag_prefix` resolves
   the `hexfield.*` diagnostics) and a `model.config` subset. Assert: response shape;
   `history["epoch"] == 2`; `prev_epoch["epoch"] == 1`; `evaluation["epoch"] == 2`;
   `selfplay_extras` contains `temperature_control` but NOT `npz_writes`/
   `mcts_diagnostics`; `manifest["model_name"]`; `checkpoint["name"] ==
   "epoch_000002.pt"` with `size > 0`.
2. Missing epoch: `_training_epoch(run, 99)` returns the run/epoch envelope with
   ALL data fields None (no exception).
3. Unknown run: `pytest.raises(ValueError)`.
4. Checkpoint canonical fallback: epoch row WITHOUT a checkpoint entry but the file
   `checkpoints/epoch_000002.pt` on disk → `checkpoint` populated.
Run the suite from the repo root with the `.venv` active:
`python -m pytest tests/test_frontend_training_epoch.py -q` — at minimum ensure the
new file passes locally and existing `test_frontend_training_artifacts.py` still
passes.

---

## P1 — FRONTEND agent 1: Epoch Inspector

### P1.1 Container + bindings
- index.html: insert `<div id="histEpochInspector" class="hist-epoch-inspector"
  hidden></div>` per §0.5.
- app.js: add `const histEpochInspector = document.getElementById(
  "histEpochInspector");` next to the other hist* const bindings (~line 157).

### P1.2 Module state (next to the other hist* vars, ~line 108)
- `let histInspectEpoch = null;` — `{run, epoch}` | null. Identity is the PAIR
  (draft correction: All-runs mode mixes runs in the epoch table, so a bare epoch
  int is ambiguous). Exactly one inspector at a time by construction.
- `let histEpochInfoCache = new Map();` — key `"<run>::<epoch>"`, value
  `{status: "loading"|"ready"|"error", payload}`; FIFO cap 20 (mirror the
  `histThumbCache` eviction loop). Completed epochs are immutable so entries live
  for the page session; a still-running epoch shows data as of open (acceptable —
  note: its selfplay diagnostics file doesn't exist until the phase completes).
- `let histCkptInfoCache = new Map();` — key `"<run>::<ckptName>"`, same states,
  FIFO cap 12 (the server caches the worker result; this only suppresses refetch
  spam across the 15s re-renders).

### P1.3 Open / close / navigate wiring
- OPEN: in the existing delegated `#histEpochTable` click listener, APPEND a third
  branch (after `data-hist-epoch-toggle` and `data-hist-epoch`, which both
  `return`): `const row = event.target.closest("[data-hist-epoch-inspect]")` → set
  `histInspectEpoch = {run, epoch}` parsed from the attr value `"<run>::<epoch>"`
  (re-clicking the SAME row toggles closed), `renderGameHistoryPage()`. In
  `renderHistEpochTable`, add `data-hist-epoch-inspect="<run>::<epoch>"` + a
  `title` hint to each `.hist-epoch-row` div, and add class `hist-epoch-inspected`
  to the row matching `histInspectEpoch`. The row's existing buttons keep their
  meanings (chevron = expand, epoch number = filter) because their branches return
  first. DRAFT CORRECTION: the draft's "inspect button in the chart hover tooltip"
  is DROPPED — `#histTrendTip` is a `position:fixed` mouse-following tooltip; making
  it clickable would fight the crosshair UX for marginal value. Chart click keeps
  its phase-1 filter-toggle meaning, full stop.
- CLOSE: ✕ button (`data-hist-inspect-close`) in the inspector header; Esc via
  `histHandleKey` (below). Both set `histInspectEpoch = null` +
  `renderGameHistoryPage()`.
- NAVIGATE: header ◀/▶ buttons (`data-hist-inspect-step="-1"|"1"`) move to the
  prev/next row of THE SAME RUN's `epoch_history` (ordered by epoch number),
  disabled at the ends; they update `histInspectEpoch` and re-render WITHOUT
  closing.
- One delegated click listener on `#histEpochInspector` handles close/step/filter
  (`data-hist-epoch` reuses `setHistEpochFilter`) buttons.
- KEYBOARD (P1 installs the handler, P2 extends it): add module function
  `histHandleKey(e)` + one `document.addEventListener("keydown", histHandleKey)` at
  startup. Shape mirrors `dbgHandleKey`: early-return unless
  `activeScreen === "history"`; return when `document.activeElement` matches
  `/^(INPUT|SELECT|TEXTAREA)$/`; return on ctrl/meta/alt. P1 branch: `Escape` →
  if `histInspectEpoch !== null` close it + `preventDefault()`.
- Stale-state guard: in `renderHistEpochInspector`, if `histInspectEpoch` names a
  run/epoch no longer present in the loaded runs' `epoch_history`, render the
  header + a muted "epoch data not loaded" line (do NOT auto-close — the run may
  simply not be re-fetched yet mid-refresh).

### P1.4 `renderHistEpochInspector(runs)` — called from `renderGameHistoryPage`
immediately after `renderHistEpochTable(runs)`; hides (`hidden = true`, clears
innerHTML) when `histInspectEpoch === null`. When open: find the run payload by
name and its `epoch_history` row + previous row (next-lower epoch, same run);
`evaluation_history` row for the epoch as fallback when `row.evaluation` is absent.
Every value through the existing "--" conventions
(`displayValue/formatDecimal/formatRate/formatPercent/asFinite`); reuse
`epochChip()` for chip groups; do NOT modify `epochProgressDetail`.

Groups (omit a group's card entirely when it has no data, EXCEPT the header):
1. HEADER (always): `Epoch N` + run name (when multiple runs loaded) + checkpoint
   tag (`saved` when `row.checkpoint` has truthy path/name, else `pending`) + ◀/▶ +
   `filter games to eN` button (`data-hist-epoch="<epoch>"` →
   `setHistEpochFilter`) + ✕. All buttons ≥44px targets,
   `touch-action: manipulation`.
2. LOSSES — from `row.selfplay.buffer` `loss_*` keys (the uniform carrier, §0.1):
   one horizontal bar per present head — total, policy, value, opp, every
   `loss_stvalue_<h>` sorted by horizon (reuse the `/^loss_stvalue_(\d+)$/` pattern
   from `epochProgressDetail`). Bar width = value / max(present head values),
   numeric value printed on the bar. Δ vs the SAME key in the previous row's buffer
   when both finite: `▼0.012` with class `.hist-delta-down` (green, loss fell) or
   `▲…` `.hist-delta-up` (red); no delta element when no previous value. Group
   omitted when no buffer / no finite loss keys.
3. GAME STATS — from `row.selfplay`: length μ/med/max/σ (`game_length_*`), a mini
   diverging win bar (REUSE the exact `.hist-balance-bar` markup pattern from
   `renderHistEpochTable` with `win_p0_fraction/draw_fraction/win_p1_fraction`),
   chips for `games`, `samples_added`,
   `formatRate(search_positions_per_second, "pos/s")`,
   `mcts_sims_per_searched_position` ("sims/pos"), `elapsed_seconds` ("sp wall").
4. BUFFER — only when `buf.samples` or `buf.cap` is finite (a real pool; loss-only
   buffers hide this group — draft requirement): pool `samples/cap`,
   window `window_epochs [window_span]`, `decay`,
   `train_steps×train_batch = train_samples_per_epoch/ep` (chip formats copied
   from `epochProgressDetail`). Also append `row.samples` chips (`buffer_count`,
   `window_size`, `formatBytes(compressed_bytes)`) when present.
5. CALIBRATION — only when `buf.optimism_sum_mean` finite: value + Δ vs previous.
6. EVAL — `row.evaluation` (fallback: the run's `evaluation_history` row for this
   epoch): `W-L` (`wins`-`losses`), `mean_turns`, win% (`wins/games` via
   formatPercent, guard zero), `completed/games`, `status`.
7. TRAIN — `row.training`: `loss` (3dp), `steps×batch_size`, `samples`,
   `formatRate(samples_per_second,"smp/s")`, `elapsed_seconds`. (Cheap, grounds the
   loss bars; the draft folded this into LOSSES — kept separate for scanability.)
8. DIAGNOSTICS (lazy, from `/api/training/epoch` `selfplay_extras` + `diagnostics`):
   chips for `temperature_control.expected_game_length` ("EMA len") +
   `halflife_plies`; `pcr` full/fast counts + `full_proportion`;
   `policy_init.moves` + `fraction`; `root_policy_temperature_control.base`/`early`;
   search wall share = `mcts_search_elapsed_seconds / elapsed_seconds`
   (formatPercent, both finite); `raw→effective` samples; `mcts_virtual_batch_size`;
   `scheduler`. Placeholder "Loading epoch diagnostics" while pending; on `"error"`
   a single muted line; when the payload has `selfplay_extras: null` (no self-play
   diagnostics file yet, e.g. a running epoch) the group is hidden.
9. MODEL — two layers:
   a. Config (from the SAME `/api/training/epoch` payload `manifest`, instant once
      that one fetch lands): model name, `channels`/`blocks_type`/
      `attention_heads`, STV horizons, moves_left flag, `search_visits`,
      `active_games`, dirichlet `fraction`/`total_alpha`,
      `root_policy_temperature`, `fpu_reduction`, eval `games_per_epoch` every
      `eval_every`.
   b. Checkpoint card (lazy ckpt_info): while pending show the mandatory one-line
      hint "Loading model card (CPU debug worker)…". On ready: `param_count`,
      `lineage`, `rl_epoch`, `step`, `graft`, `stv_horizons`, `moves_left_cap`,
      `formatBytes(size)` + `formatHistoryDate(mtime)`, and `load_warnings` lines
      when non-empty (title attr carries `meta.arch`). When NO checkpoint exists
      (no row.checkpoint AND endpoint `checkpoint` null) → this sub-card is hidden
      entirely, the fetch is never fired, no console spam (B2).

### P1.5 The two lazy fetchers (mirror `loadHistThumb` exactly: dedupe via the
"loading" cache state, FIFO-evict, plain `fetch` + `safeJson`, NEVER touch
`pendingRequest`, patch-in-place only when still relevant)
- `loadHistEpochInfo(run, epoch)` → `GET /api/training/epoch?run=&epoch=` →
  cache `"<run>::<epoch>"`. On resolve, ONLY IF `histInspectEpoch` still equals
  that run/epoch, patch the DIAGNOSTICS + MODEL-config group containers and the
  header checkpoint line via querySelector innerHTML swaps (no full re-render —
  same rationale as the thumbnail: a full re-render mid-async races the 15s
  refresh).
- `loadHistCkptInfo(run, ckptName)` → `GET /api/debug/ckpt_info?run=&checkpoint=` →
  cache `"<run>::<ckptName>"`, patch the model sub-card if still inspected AND the
  derived name unchanged. Fired from `renderHistEpochInspector` only when a
  checkpoint name is derivable (§0.2 rule, implemented as a small
  `histCkptNameForEpoch(row, epoch)` helper).
- Both render functions read the caches synchronously on every re-render, so the
  15s refresh repaints fetched data with zero new requests.

### P1.6 CSS (styles.css, `.hist-*` namespace, copy the phase-1 visual language)
`.hist-epoch-inspector` (panel card: `--panel`-style background, 1px border, 8px
radius, padding); internal `.hist-insp-head` flex row; `.hist-insp-groups` grid
`repeat(auto-fill, minmax(240px, 1fr))`; `.hist-insp-group` +
`.hist-insp-group-title`; `.hist-loss-row` (label / bar / value / delta),
`.hist-loss-bar(-fill)` reusing the `.hist-bar` look; `.hist-delta-down` (green) /
`.hist-delta-up` (red); `.hist-epoch-inspected` row accent (pair with
`.hist-epoch-live` styling, distinct color); buttons ≥44px with
`touch-action: manipulation`. Mobile ≤900px: inspector is already full-width —
just ensure the groups grid collapses to one column and nothing overflows at 390px
(extend the existing ≤900px hist block at ~line 2610).
`node --check app.js` after every edit.

---

## P2 — FRONTEND agent 2: Selected Game panel + keyboard + mobile + bump

### P2.1 Game prev/next navigation
- Module var `let histDisplayedKeys = [];` set inside `renderGameHistoryPage`
  right after `displayed` is computed:
  `histDisplayedKeys = displayed.map(item => historyItemKey(item));`
  (the CURRENT filtered+sorted+epoch-filtered order — exactly what the list
  shows; note `selectedHistoryItem` may fall back to `histories` when `displayed`
  is empty, in which case nav renders disabled).
- Nav row at the TOP of the rewritten `gameHistoryDetailHtml`: buttons
  `#histGamePrev`/`#histGameNext` with `data-hist-game-step="-1"`/`"1"`
  (`disabled` attr at the ends or when the selected key is not in
  `histDisplayedKeys`), plus a `k / n` readout (`indexOf(selectedHistoryKey)+1` /
  `histDisplayedKeys.length`; "-- / n" when not found).
- Click handling: APPEND a `data-hist-game-step` branch to
  `handleGameHistoryClick` (before the final `data-history-key` fallthrough; the
  detail panel already routes through this handler). It computes
  `idx = histDisplayedKeys.indexOf(selectedHistoryKey)`, steps, bounds-checks,
  sets `selectedHistoryKey = histDisplayedKeys[idx + step]`, calls
  `renderGameHistoryPage()` — the EXACT same state+path a row click takes, so the
  lazy thumbnail, selection highlight, and detail render are identical for free.

### P2.2 Detail panel reorganization (`gameHistoryDetailHtml` rewrite — keep every
current FACT, relocate most into the disclosure)
Top-level, in order:
1. NAV ROW (P2.1).
2. OUTCOME LINE (one `.hist-outcome-line`): winner dot (reuse the `.hist-win-dot`
   pattern from `gameHistoryListRow`) + `item.winner_label || winnerLabel(...)`
   with `winnerClass` coloring · length (`item.length || item.actions || 0`) ·
   source · `e{epoch ?? "--"}` · status. Replaces the 4-cell
   `.history-detail-hero`.
3. THUMBNAIL: the phase-1 `.hist-thumb` block UNCHANGED (same cache, same
   `loadHistThumb`, same `.hxr` condition).
4. ACTIONS: Load Replay (existing `data-history-load data-history-run
   data-history-path data-record-index` attrs) + Open in Debug BYTE-EQUIVALENT
   (`data-debug-open data-debug-run data-debug-path data-debug-record`, same
   `.hxr`-suffix condition). DRAFT CORRECTION: the draft said Open-in-Debug only;
   Load Replay STAYS top-level — it is the screen's primary action and burying it
   regresses phase-1 acceptance A7's "Replay behaves exactly as before".
5. DETAILS DISCLOSURE: native `<details class="hist-detail-more">` +
   `<summary>Details</summary>` (summary ≥44px target) holding everything else
   currently in the panel: the `detail-stack` rows (Run, Game, Status, Seed,
   Record, Path, Modified), the Players section, the Diagnostics section, and the
   Abort section (keep the phase-1 object-safe abort rendering verbatim).
   Collapsed by default; open state intentionally NOT persisted — the 15s
   innerHTML refresh collapses it again (explicit, accepted; do not add state).

### P2.3 Keyboard
Extend `histHandleKey` (P1.3) with: `ArrowLeft`/`ArrowRight` → ONLY when
`histInspectEpoch === null` (cheap guard reserving arrows for future inspector
bindings), step the selected game by -1/+1 through `histDisplayedKeys` (same logic
as the P2.1 branch — factor a tiny `histStepGame(step)` used by both),
`preventDefault()` when a step occurred. The existing guards (history screen
active, no input focus, no modifiers) come from P1's handler shell.

### P2.4 Mobile + polish (≤900px, extend the existing hist media block)
- `#histEpochInspector`: full-width card (it is — verify), groups single-column,
  header buttons ≥44px (B5).
- Nav buttons, details summary, action buttons ≥44px with
  `touch-action: manipulation`.
- Verify (DOM reasoning, no server) no horizontal overflow of `.history-page` at
  390px: the inspector grid, loss bars (width %, never fixed px), nav row, and
  outcome line must all wrap or truncate (`text-overflow: ellipsis`).

### P2.5 FINAL STEP — cache-bust
Change BOTH index.html occurrences `?v=20260611-hist1` (line ~7 stylesheet, line
~532 script) to `?v=20260611-hist2`, then run `node --check app.js` one last time.
If ANY later fix touches app.js/styles.css after this, re-bump (-hist3, …) — a
stale ?v= silently serves old code through the WSL proxy.

---

## ACCEPTANCE (binding)

B1. Clicking an epoch-table row (not its chevron/epoch-number buttons) opens
    `#histEpochInspector` with per-head loss bars + green/red deltas vs the
    previous epoch, game stats, and — on runs with a real pool — buffer +
    calibration groups; header ◀/▶ walk that run's epochs without closing;
    re-clicking the row, ✕, or Esc closes; the open inspector and its fetched
    groups survive the 15s refresh (state in `histInspectEpoch` + caches, zero
    refetches on re-render).
B2. The DIAGNOSTICS + MODEL-config groups appear after the single lazy
    `/api/training/epoch` fetch; the deep model card appears after the lazy
    `ckpt_info` fetch with the "CPU debug worker" pending hint, is cached per
    checkpoint client-side, and an absent checkpoint hides the card with no error
    spam and no fetch.
B3. The Selected Game panel is exactly: nav row / outcome line / thumbnail /
    actions (Load Replay + Open in Debug, byte-equivalent attrs) / collapsed
    details disclosure holding path, record, seed, players, diagnostics, abort.
    Prev/next walk the displayed (filtered+sorted+epoch-chip) order, disabled at
    the ends; ArrowLeft/ArrowRight do the same with the focus/modifier/inspector
    guards; Open in Debug still deep-links into the Debug screen exactly as
    before.
B4. `/api/training/epoch` returns the P3.1 contract shape for a real epoch
    (curated `selfplay_extras` without the excluded bulk keys, curated `manifest`,
    `prev_epoch`, stat-backed `checkpoint`), all-null data fields for a missing
    epoch, 400 `{"error":…}` for an unknown run; the new test file passes and ALL
    pre-existing tests still pass (`python -m pytest tests/ -q` from the repo root,
    `.venv` active).
B5. ≤900px: the inspector is a full-width card with single-column groups; every
    new interactive target (inspector buttons, nav buttons, details summary) is
    ≥44px with `touch-action: manipulation`; no horizontal overflow at 390px.
B6. `?v=` bumped to `20260611-hist2` at BOTH index.html occurrences as the very
    last frontend change; `node --check app.js` passes; Match and Debug screens
    untouched; no new fetch call sites beyond `loadHistEpochInfo` +
    `loadHistCkptInfo` (and the phase-1 thumbnail fetch).

## RISKS / NOTES
- `renderGameHistoryPage` runs from `renderTraining` and async guards at any time:
  `renderHistEpochInspector` must early-return null-safely when the container or
  data is missing, and never fire fetches unless the inspector is actually open.
- The endpoint reads `_epoch_history` per call (no cache) — fine at
  once-per-inspector-open; do NOT wire it into any polling loop.
- a run with no `hexfield.selfplay.epoch_*.json` → `selfplay_extras` null →
  DIAGNOSTICS group hidden; a run with a loss-only buffer → BUFFER/CALIBRATION
  hidden; a run mid-epoch has neither file nor checkpoint → header + GAME
  STATS-from-row only. All three shapes must render without throwing.
- Inspector deltas compare the previous epoch OF THE SAME RUN; never mix runs.
- `histDisplayedKeys` is rebuilt every render; the keydown handler must re-read it
  (no captured copies).
