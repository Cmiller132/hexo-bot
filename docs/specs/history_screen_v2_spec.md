# History Screen v2 — "Run Status Dashboard" Implementation Spec

Status: FINAL. Binding contract for the two frontend implementation agents.
Scope: PRESENTATION-ONLY rewrite of the Game History screen. All data fields below were
verified against `web.py` on 2026-06-11; if code and spec disagree on a FIELD NAME, the
field names here win (they were read from the live backend functions).

Targets (the ONLY files that may be modified):
- `packages/hexo_frontend/python/hexo_frontend/static/app.js`
- `packages/hexo_frontend/python/hexo_frontend/static/index.html`
- `packages/hexo_frontend/python/hexo_frontend/static/styles.css`

Hard constraints (do not violate):
- Do NOT touch `web.py`, any backend file, any test file, or the Match/Debug screens'
  HTML/JS/CSS. No frameworks, no build step, hand-rolled SVG only.
- New element ids prefixed `hist`, new CSS classes `.hist-*`.
- `node --check app.js` must pass after EVERY app.js edit. Do NOT git commit. Do NOT
  start/restart any server.
- No new fetch endpoints and no new fetch call sites EXCEPT the one lazy detail-record
  fetch in H9 (`GET /api/training/history?run=&path=&record=`, an endpoint the screen
  already uses via Load Replay).
- Agents run SEQUENTIALLY: agent 1 first (H1-H5), then agent 2 (H6-H11). Agent 1 must
  leave the screen in a non-broken (if visually incomplete) state.

---

## 1. VERIFIED DATA SHAPES (normative — read this, not the design draft)

All of this arrives in the already-fetched run payload `trainingRunDetails[name]`
(`/api/training/run`). No new fetches are needed for regions 1-3.

### 1.1 `run.status` (= `_training_run_status`, a superset of `_training_live_status`)
- `stage` (string id, e.g. `"epoch_000012"`), `stage_status` (`"running"`/...),
  `current_epoch` (int|null), `last_event`.
- `sub_phase` (`"self-play" | "shuffling" | "training" | "evaluating"`, OPTIONAL) and
  `sub_phase_detail` (e.g. `"games 12/40"` or `"SealBot"`). `runStageLabel(status)`
  already folds these in — REUSE IT, do not re-derive.
- `watchdog` (optional): `status`, `critical` (list), `free_ram_gb`, `free_virtual_gb`,
  `trainer_private_gb`, `trainer_working_set_gb`, `gpu_free_gb`, `gpu_used_gb`,
  `gpu_utilization_percent`. NOTE: free values only — there is NO total-RAM/total-GPU
  field, so percentage-fill bars are impossible. Render resources as value chips
  (see H2), not bars. (Draft correction.)
- `calibration` (optional): `selfplay_pos_s`, `target_pos_s`, `meets_target`, `exact_128`.
- `selfplay_live` (optional): `status`, `live` (bool, fresh-file heuristic),
  `age_seconds`, `epoch`, `search_pos_s`, `pos_s`, `searched_positions`,
  `games_finished`, `requested_games`, `active_games`, `elapsed_seconds`.
- `training_progress` (optional): `epoch`, `status`, `progress` (0..1), `steps`,
  `total_steps`, `samples_seen`, `samples`, `passes`, `loss`, `samples_per_second`.
  IMPORTANT (draft correction): web.py marks the producing file
  `<prefix>.training_progress.epoch_*.json` as "no current producer emits this file"
  — `training_progress` is normally ABSENT on current runs (it appears only for the
  classical-bootstrap prefit path / manual drops). The status band's progress bar must
  therefore be conditional, with `selfplay_live` as the primary live signal.
- `history` (recent-window stats): `games`, `completed`, `aborted`, `p0_wins`,
  `p1_wins`, `min_length`, `max_length`, `avg_length`, `latest_modified`, `latest_path`.

Access via the existing `latestRunStatusForHistoryPage()` (keep it).

### 1.2 `run.epoch_history` — list of rows sorted by epoch
Row: `{ epoch, status ("completed"|"partial"|...), elapsed_seconds?, selfplay?,
training?, evaluation?, checkpoint?, samples?, d6? }`
- `selfplay`: `status, games, completed_games, truncated_games, samples_added,
  searched_positions, mcts_simulations, search_positions_per_second,
  mcts_sims_per_searched_position, elapsed_seconds, game_length_mean,
  game_length_median, game_length_max, game_length_stdev, game_length_p95,
  win_p0_fraction, win_p1_fraction, draw_fraction, decisive_fraction, mean_abs_value,
  buffer`. Any of these may be null (some length/fraction stats are backfilled
  server-side from the .hxr records; nulls remain possible) — every chart/cell skips
  null points.
- `selfplay.buffer` (nested object or null): the uniform per-head loss carrier, present
  on Shrimp runs with a training pool. Keys consumed today by
  `epochProgressDetail`: `samples, cap, window_epochs, window_span, decay, train_steps,
  train_batch, train_samples_per_epoch, loss_total, loss_policy, loss_value, loss_opp,
  loss_stvalue_<h>, optimism_sum_mean`. A given run may carry ONLY the `loss_*` keys
  (no pool/optimism), so guard every key individually.
- `training`: `status, loss, loss_components ({policy,value,opp_policy,stvalue_*}),
  source_summary (or null), policy_imitation (or null), steps, samples, batch_size,
  samples_per_second, elapsed_seconds`, optional `progress`.
  DRAFT CORRECTION: per-epoch total loss is `row.training.loss`; per-head losses for
  charting come from `row.selfplay.buffer.loss_policy / .loss_value` (the uniform loss
  carrier — normalized server-side from `training.loss_components`). Do NOT
  read `item.training.loss_total` (does not exist) or `buffer.loss` (does not exist).
- `evaluation`: `status, games, completed, wins, losses, mean_turns`.
- `checkpoint`: `{path, bytes, modified}` (from checkpoints dir) OR `{path, name}`
  (from epoch result) — treat "any truthy path or name" as saved.
- `samples`: `{buffer_count, window_size, compressed_bytes}` (optional).
- `d6`: `{mode, group_size?, sample_count?, preview_count?, preview_symmetries?}`.

### 1.3 `run.evaluation_history` — list sorted by epoch
Row: `{ epoch, status, games, completed, wins, losses, mean_turns, path, modified }`.

### 1.4 `run.learning_health`
`status` ∈ `"collecting" | "ok" | "improving" | "watch" | "intervene"` (FIVE states —
draft listed four; `collecting` must have a pill style too). Fields: `latest_epoch`,
`current_stage`, `latest_loss`, `loss_delta_from_first`, `latest_eval_mean_turns`,
`best_eval_mean_turns`, `eval_delta_from_first`, `latest_eval_wins`,
`latest_eval_games`, `latest_selfplay_pos_s`, `latest_exact_128`,
`latest_classical_fraction`, `latest_policy_top1`, `latest_policy_target_mass`,
`d6_preview_symmetries`, `messages` (string list). Access via existing
`latestLearningHealth(runs)` (keep it).

### 1.5 Game page items (`historyPage.items`, from `/api/training/history-page`)
Each item: `{ run, path, record_index, game_id, status, winner
("player0"|"player1"|null), winner_label, length (int), actions (int), epoch
(int|null), source ("selfplay"|"evaluation"|...), seed, players ({player0,player1:
{role,kind,label,player_id}}), modified, modified_ns, bytes, abort (object|null:
{stage,exception_type,message}), diagnostics (brief) }`.
DRAFT VERIFICATION RESULT: there is NO placements list and NO action_ids on page items
(`placements` in the .hxr record is an int count, already folded into `length`). The
final-position thumbnail therefore REQUIRES the lazy fetch (H9).

### 1.6 Lazy detail fetch (thumbnail only)
`GET /api/training/history?run=<run>&path=<path>&record=<record_index>` returns the full
dashboard payload; the thumbnail needs ONLY `payload.placements`: a list of
`{q, r, player ("player0"|"player1"), phase, index}`. Stones = all entries; last entry =
final move. This endpoint replays the whole game server-side (moderately heavy) — fetch
only when the detail panel shows a `.hxr` item, cache per item key, never in a loop.

---

## 2. LAYOUT (desktop > 900px)

```
.history-page
  .history-page-head            (UNCHANGED: title + #historyRunSelect + #historyRefreshBtn)
  #histStatusBand .hist-status-band     <- Region 1 (replaces #historyOverview + #historyLearningHealth)
  #histHealthDrawer .hist-health-drawer (sibling, hidden unless open)
  #histTrends .hist-trends              <- Region 2 (replaces #historyEvalTrend)
  #histEpochTable .hist-epoch-table     <- Region 3 (replaces #historyEpochProgress)
  .history-filters               (UNCHANGED ids; + #histEpochChip appended at the end)
  .history-layout                (UNCHANGED: #gameHistoryList | #gameHistoryDetail)
```
`#histTrendTip` (the one shared tooltip div) is appended once to `document.body`.

Container divs REMOVED from index.html: `#historyOverview`, `#historyLearningHealth`,
`#historyEvalTrend`, `#historyEpochProgress` (and their classes `history-overview`,
`history-learning-health`, `history-eval-trend`, `history-epoch-progress`).
Containers ADDED: `#histStatusBand`, `#histHealthDrawer`, `#histTrends`,
`#histEpochTable` (agent 1) and `#histEpochChip` inside `.history-filters` (agent 2).

### Final id list (new)
`histStatusBand, histHealthDrawer, histTrends, histTrendTip, histEpochTable,
histEpochChip, histStagePill, histTrainBar, histResources, histHealthPill,
histStatChips` — the last five are rendered INSIDE #histStatusBand by JS (they may be
classes on rendered nodes rather than ids if delegation makes ids unnecessary, except
`histHealthPill` and `histEpochChip` which are interaction targets and MUST keep these
ids/data hooks).

### New module-level state (app.js, near the other history state vars)
- `histEpochFilter = null` (int|null) — client-side epoch display filter.
- `histHealthDrawerOpen = false` — survives the 15s re-render.
- `histExpandedEpochs = new Set()` — keys `"<run>::<epoch>"`, survives re-render.
- `histTrendCharts = {}` — per-render chart data for hover lookup.
- `histThumbCache = new Map()` — key `historyItemKey(item)`, value
  `{status:"loading"|"ready"|"error", placements}`; FIFO-cap at 40 entries.

---

## 3. DO NOT TOUCH (names are exact; behavior and signatures frozen)

Data/paging machinery (may be CALLED, never modified):
`loadTrainingRuns, fetchTrainingRunDetail, loadTrainingRun,
ensureHistorySelectionLoaded, resetHistoryPage, currentHistoryPageKey,
currentHistoryTargets, shouldAutoloadHistoryWindow, historyItemInTargetWindow,
historyWindowBoundaryReached, enterHistoryScreen, loadHistoryPage, loadHistoryCount,
refreshHistoryIfVisible, loadTrainingHistory, loadMoreArtifacts, historyRunsForPage,
historySelectionPendingDetails, historyItemsForPage, syncHistoryRunSelect,
selectedHistoryItem, historyItemKey, filteredHistoryItems, sortedHistoryItems,
compareHistoryNewest, latestRunStatusForHistoryPage`, the state vars
`historyPage, historySelectedRun, historyFilters, historySort, historyVisibleLimit,
selectedHistoryKey, trainingRunDetails, historyDetailsLoading`, and the constants
`HISTORY_PAGE_SIZE, HISTORY_AUTOLOAD_PAGE_SIZE, HISTORY_REFRESH_INTERVAL_MS,
HISTORY_ALL_RUNS`.

Shared helpers (may be CALLED, never modified): `escapeText, escapeAttr, displayValue,
asFinite, firstPresent, formatDecimal, formatRate, formatPercent, formatGib,
formatBytes, formatHistoryDate, center, path, playerColor, HEX, SQRT3, setScreen,
navigateScreen, screenFromHash, debugOpenFromHistory, diagTap, reportError, on, safeJson`.

Match-screen training sidebar (`renderTraining`, `trainingArtifactRow`,
`summaryMetric`, `runStageLabel`, `humanizeStageId`, `historyEpochs`,
`averageHistoryLength`) — used by the Match screen; do not remove or alter. Note
`renderTraining` ends with a call to `renderGameHistoryPage()` — the rewritten
`renderGameHistoryPage` must remain safe to call from there at any time.

Deep link: the `[data-debug-open]` handling in `handleGameHistoryClick` and the
`trainingArtifacts` click delegation (~app.js 196) stay byte-identical in behavior. The
Open in Debug button in the detail panel must keep EXACTLY the attributes
`data-debug-open data-debug-run data-debug-path data-debug-record` and the
`.hxr`-suffix render condition. `handleGameHistoryClick`'s four existing branches
(`data-history-more`, `data-debug-open`, `data-history-load`, `data-history-key`) keep
their current behavior; new branches may be appended.

Existing control ids stay: `#historyRunSelect #historyRefreshBtn #historySearchInput
#historySourceSelect #historyWinnerSelect #historySortSelect #gameHistoryList
#gameHistoryDetail #historyScreen`.

KEPT-AND-REUSED render helpers (do not delete): `epochProgressDetail`, `epochChip`,
`latestLearningHealth`, `learningHealthClass`, `learningHealthLabel`,
`formatTrainingProgress`, `trainingProgressSubtext`, `historyDiagnosticsText`,
`historyPlayerLabel`, `winnerLabel`, `winnerClass`, `classicalReplayFraction`,
`policyTop1`, `historyLength`, `historyWinStats` may be kept if reused (else removable).

---

## 4. WORK ITEMS

### Agent 1 — status band + trends grid (H1-H5)

H1. Container swap + render shell.
  - index.html: replace the four old container divs (lines ~210-213) with
    `#histStatusBand`, `#histHealthDrawer` (hidden by default), `#histTrends`,
    `#histEpochTable` per §2.
  - app.js: replace the four const bindings (`historyOverview`,
    `historyLearningHealth`, `historyEvalTrend`, `historyEpochProgress`, ~lines
    138-148) with `histStatusBand`, `histHealthDrawer`, `histTrends`,
    `histEpochTable`.
  - Rewrite `renderGameHistoryPage`'s shell: same guard pattern (null-safe early
    return), error branch and "no runs" branch clear the new containers; happy path
    calls `renderHistStatusBand(runs)`, `renderHistTrends(runs)`,
    `renderHistEpochTable(runs)` then the games list/detail exactly as before.
    Provide `function renderHistEpochTable(runs) { histEpochTable.innerHTML = ""; }`
    as an explicit stub for agent 2 (comment it `// H6: agent 2`).
  - Acceptance: screen loads with band + trends and an empty epoch-table area; list,
    detail, filters, paging, Load more, Replay, Open in Debug all still work;
    `node --check` passes.

H2. Status band `renderHistStatusBand(runs)` writing `#histStatusBand` +
    `#histHealthDrawer`.
  - Source: `const status = latestRunStatusForHistoryPage()`,
    `const health = latestLearningHealth(runs)`.
  - Stage pill `#histStagePill` (`.hist-pill`): text `runStageLabel(status)`; when
    `status.selfplay_live && status.selfplay_live.live`, prepend a pulsing dot
    (`.hist-live-dot`, CSS animation) and append
    `games_finished/requested · {formatRate(search_pos_s,"pos/s")}`.
  - Inline progress `#histTrainBar` (`.hist-bar` + `.hist-bar-fill`): conditional
    (per §1.1 training_progress is usually absent):
    (a) if `selfplay_live.live` and `requested_games > 0`: fill =
        games_finished/requested, label `selfplay {finished}/{requested}`;
    (b) else if `status.training_progress` present: fill = `progress` (or
        steps/total_steps), label
        `${formatTrainingProgress(tp)} · ${trainingProgressSubtext(tp)}`;
    (c) else render nothing (no empty bar).
  - Resources `#histResources`: two value chips `RAM {formatGib(watchdog.free_ram_gb)}
    free` and `GPU {formatGib(watchdog.gpu_free_gb)} free` (+ `gpu_utilization_percent`
    in the title attr), tinted by `watchdog.status` ("ok" = neutral, anything else or
    non-empty `critical` = warn tint). NO percentage bars (no totals exist). Omit
    chips whose value is null; omit the group when no watchdog.
  - Health pill `#histHealthPill`: label via `learningHealthLabel(health.status)`,
    class `hist-health-{status}` for all FIVE statuses (collecting/ok/improving/
    watch/intervene). It is a `<button type="button">` (44px min target,
    `touch-action: manipulation`). Click toggles `histHealthDrawerOpen` and
    re-renders; `#histHealthDrawer` lists ALL `health.messages` (one line each) when
    open, `hidden` attribute when closed. Wire via one delegated click listener on
    `#histStatusBand` (added once at startup with the `on(...)` helper or direct
    addEventListener guarded for null).
  - Stat chips `#histStatChips` (each `.hist-chip`, value "--" when null):
    `e{latest_epoch}`, `loss {formatDecimal(latest_loss,3)}`,
    `eval {formatDecimal(latest_eval_mean_turns,1)}t` (title: best
    `best_eval_mean_turns` + `latest_eval_wins`/`latest_eval_games`),
    `{formatRate(latest_selfplay_pos_s,"pos/s")}`,
    `P@1 {formatPercent(latest_policy_top1)}`,
    `C {formatPercent(latest_classical_fraction)}`.
  - When `status` and `health` are both null render a single muted "No run status yet"
    line. Everything must degrade to "--" without throwing.

H3. Trends grid `renderHistTrends(runs)` writing `#histTrends`.
  - Series source: when multiple runs are loaded (All runs), chart ONLY the run whose
    `status.history.latest_modified` is newest (same pick as
    `latestRunStatusForHistoryPage`) and show its name as a small caption
    (`.hist-trends-caption`); merging epoch series across runs draws garbage. The
    epoch table (H6) still shows all runs.
  - Charts (each rendered only when its series has >= 2 non-null points; otherwise
    omitted entirely — no empty boxes):
    - T1 Loss: line `training.loss` (total); thin lines
      `selfplay.buffer.loss_policy` and `selfplay.buffer.loss_value` when present.
    - T2 Game length: line `selfplay.game_length_mean`; dots
      `selfplay.game_length_median`; shaded band mean±`selfplay.game_length_stdev`
      (skip band where stdev null).
    - T3 Win balance: stacked 100% area of `selfplay.win_p0_fraction`,
      `selfplay.draw_fraction`, `selfplay.win_p1_fraction` (colors: var(--p0),
      neutral, var(--p1)).
    - T4 SealBot eval (x = `evaluation_history` epochs): line `mean_turns`; dot
      marker on epochs with `wins > 0`; annotate the best epoch (max by
      (mean_turns, wins), the `renderEvaluationTrend` tie-break) with a small label.
    - T5 Selfplay speed: line `selfplay.search_positions_per_second`.
    - T6 Buffer (only when >= 2 epochs have a real pool, i.e.
      `buffer.samples`/`buffer.cap` numeric): line samples/cap fill fraction; and/or
      a second chart for `buffer.optimism_sum_mean` when >= 2 points. (Only runs with
      a real pool carry these; loss-only buffers simply lack them.)
  - Construction: one `<svg>` per chart, fixed `viewBox="0 0 280 110"`, CSS-sized,
    grid `repeat(auto-fill, minmax(230px, 1fr))`. Title text top-left, latest value
    top-right, min/max y labels, a tick + label on the latest epoch. Build the
    polyline/area `d`/`points` strings once per render; store
    `histTrendCharts[chartId] = {epochs, series, x0, dx}` and set
    `data-hist-chart="<chartId>"` on the svg.
  - Hover: ONE delegated `mousemove`/`mouseleave` listener pair on `#histTrends`. On
    move, find the svg under the pointer, map offsetX -> nearest epoch index via
    stored `x0/dx`, move that chart's single crosshair `<line>` (attribute update
    only — no innerHTML), and position/fill the shared `#histTrendTip`
    (position:fixed) with `e{epoch}` + each series label:value. Hide on leave. No
    per-frame allocation beyond the tooltip string.
  - Click: same delegation; clicking a chart maps to nearest epoch and calls
    `setHistEpochFilter(epoch)`. Agent 1 DEFINES
    `function setHistEpochFilter(epoch) { histEpochFilter = (histEpochFilter === epoch ? null : epoch); renderGameHistoryPage(); }`
    (toggle semantics); agent 2 builds the chip UI on top (H7).
  - Cheap: charts re-render only inside `renderGameHistoryPage` (i.e. per data
    refresh / interaction), never on a timer of their own.

H4. CSS for H1-H3: `.hist-status-band` (one wrapping flex row, single viewport row on
    desktop), `.hist-pill`, `.hist-live-dot` (pulse keyframes), `.hist-bar(-fill)`,
    `.hist-chip`, `.hist-health-{collecting,ok,improving,watch,intervene}` (reuse the
    color scheme of the old `.learning-health-panel.{watch,intervene,improving}`
    rules before deleting them), `.hist-health-drawer`, `.hist-trends`,
    `.hist-trend-card`, `.hist-trends-caption`, `.hist-trend-tip`. Dark-theme
    consistent with existing panel styles (`--panel`-style backgrounds, 1px borders,
    8px radii — copy the visual language of `.eval-trend-panel`).

H5. Remove the replaced renderers + dead CSS, then `node --check`.
  - Delete from app.js (verified unreferenced once H1-H3 land):
    `renderHistoryOverview, renderLearningHealth, renderEvaluationTrend, evalTrendRow,
    evalTrendSummary, renderEpochProgress, epochProgressRow, epochProgressSummary,
    historyWinRateText, historyWinRateSubtext, latestDiagnosticSummary` (and
    `historyWinStats` only if H2/H6 ended up not using it). Do NOT delete anything in
    the §3 keep lists.
  - Delete from styles.css: `.history-metric-card` rules, `.learning-health-*` rules,
    `.eval-trend-*` rules, `.epoch-progress-panel/-head/-table/-row/-header` rules and
    the old container rules (`.history-overview`, `.history-learning-health`,
    `.history-eval-trend`, `.history-epoch-progress`), INCLUDING their mentions in the
    media queries around lines 2305-2370. KEEP `.epoch-progress-detail`,
    `.epoch-detail-group`, `.epoch-detail-label`, `.epoch-chip`, `.epoch-chip-total`
    (+ their ~line 2429 media query) — H6 reuses them. Beware combined selectors: when
    a kept selector shares a rule with a deleted one, trim the selector list, don't
    delete the rule.

### Agent 2 — epoch table + games + interactions + mobile (H6-H11)

H6. Epoch table `renderHistEpochTable(runs)` (replace the H1 stub) writing
    `#histEpochTable`.
  - Rows: flatMap `run.epoch_history` across runs tagged with run name, sort by
    (epoch, run), take the LAST 15, display newest first. One dense grid row
    (`.hist-epoch-row`) per epoch: epoch number; selfplay summary
    `{samples_added} smp · len μ{game_length_mean}/{game_length_median} ·
    {formatRate(search_positions_per_second,"pos/s")}`; win-balance inline diverging
    bar (`.hist-balance-bar`: P0 share left in var(--p0), P1 right in var(--p1),
    draws as a muted middle segment; widths from win_p0/draw/win_p1 fractions; omit
    the bar when all null); train `loss {formatDecimal(training.loss,3)}` (else
    "pending"/training.status); eval `{wins}-{losses} · {mean_turns}t` (else
    "pending"); checkpoint tag "saved"/"pending" (per §1.2 checkpoint rule). Null
    segments render "--".
  - Chevron `<button>` (`data-hist-epoch-toggle="<run>::<epoch>"`, 44px target)
    toggles membership in `histExpandedEpochs`; expanded rows append the EXISTING
    `epochProgressDetail(selfplay.buffer)` band (unchanged function). Runs whose
    buffer is loss-only show just the Losses group (epochProgressDetail's
    Buffer group renders "--" chips — acceptable) and runs with no buffer at all
    show nothing — both must not throw. Delegated click listener on
    `#histEpochTable`.
  - Live accent: row gets `.hist-epoch-live` when `status =
    latestRunStatusForHistoryPage()` has `current_epoch === row.epoch` and the row's
    run is that status's run.
  - Clicking the epoch number (`data-hist-epoch="<epoch>"`, a button) calls
    `setHistEpochFilter(epoch)`.

H7. Epoch-filter chip + list integration.
  - index.html: append `<span id="histEpochChip" class="hist-epoch-chip" hidden></span>`
    inside `.history-filters` (after `#historySortSelect`).
  - In `renderGameHistoryPage`: when `histEpochFilter !== null`, fill the chip with
    `Epoch {N} <button data-hist-epoch-clear>×</button>` and unhide; clear button (>=
    44px incl. padding) sets `histEpochFilter = null` and re-renders.
  - Filtering is CLIENT-SIDE DISPLAY ONLY (the server has no epoch param and
    `loadHistoryPage`/`currentHistoryPageKey` are frozen): after the existing
    `filtered` is computed, derive
    `displayed = histEpochFilter === null ? filtered : filtered.filter(i => asFinite(i.epoch) === histEpochFilter)`
    and use `displayed` for the list, the visible slice, and as the `filtered` arg of
    `selectedHistoryItem` (so selection follows the filter). "Load more" keeps
    appending unfiltered pages; the filter re-applies on re-render. Footer/count
    line: when filtering, show `{displayed.length} of {filtered.length} loaded ·
    epoch {N}`; otherwise keep the existing count semantics (totalMatches /
    counting / loaded).
  - Changing run/source/winner/sort/search does NOT auto-clear the chip (it is
    orthogonal); switching runs MAY leave an empty list — the empty-state message
    must then say "No loaded games for epoch N — clear the epoch chip or load more".

H8. Game list rows — rewrite `gameHistoryListRow` to ONE line (`.hist-game-row`,
    min-height 44px):
    winner dot (10px circle, `background: playerColor(item.winner)`, hollow/muted
    when no winner); label `g{record_index} e{epoch ?? "--"}`; source glyph
    (`.hist-src-selfplay` "SP" / `.hist-src-evaluation` "EV" / other "H" — text
    badges, not icons); length number + relative micro-bar (width = length / max
    length among currently displayed rows; compute the max once per render);
    truncated player labels `historyPlayerLabel(p0)} v {p1}`; `formatHistoryDate
    (item.modified)`. Row keeps `data-history-key` (selection) and the row content is
    a full-width button or has the existing `.game-history-select` button semantics;
    keep a compact Replay affordance with the existing `data-history-load
    data-history-run data-history-path data-record-index` attributes. Update/replace
    `.game-history-row` CSS with `.hist-game-row` styles and update the
    `.history-table-head` columns in index.html to match the new single-line layout
    (or hide the old header and render a `.hist-list-head` — implementer's choice,
    but the old 6-column header must not visibly mismatch).

H9. Detail panel — rewrite `gameHistoryDetailHtml` keeping ALL current facts
    (winner/length/epoch/source hero; run, game, status, seed, record, path, modified;
    players; diagnostics blocks; abort) plus:
  - Open in Debug button byte-equivalent per §3 (same attrs, same `.hxr` condition);
    keep Load Replay with its `data-history-load` attrs.
  - Fix the existing abort rendering bug: `item.abort` is an OBJECT — render
    `{stage}: {exception_type}: {message}` fields instead of the current
    `escapeText(item.abort)` "[object Object]".
  - Final-position thumbnail (`.hist-thumb`, ~140px square), only for `.hxr` paths:
    on render, if `histThumbCache` has `ready` placements for
    `historyItemKey(item)`, draw inline; if absent, insert a placeholder
    (`.hist-thumb-loading`) and call `loadHistThumb(item)`:
    plain `fetch("/api/training/history?run=..&path=..&record=..")` + `safeJson`
    (do NOT call `loadTrainingHistory` — it navigates and applies state, and do NOT
    touch `pendingRequest`). On resolve, store `{status:"ready", placements:
    payload.placements || []}` (or `"error"`) in the cache (FIFO-evict past 40) and,
    ONLY IF `selectedHistoryKey` still equals the item key, patch the placeholder
    (querySelector the `.hist-thumb` and set its innerHTML — no full re-render).
    Guard against duplicate in-flight fetches via the `"loading"` cache state.
  - Thumbnail SVG: stones as circles at `center(q, r)` with radius `HEX * 0.62`,
    `fill: playerColor(p.player)`; final placement gets an accent ring; viewBox =
    bounding box of stone centers padded by `2 * HEX`; empty placements -> hide the
    thumb. Pure presentation, no pan/zoom, `pointer-events: none`.

H10. Mobile + polish (<= 900px, extend the existing `@media (max-width: 900px)` block
    or add `.hist-*` rules to the nearest equivalent):
  - Single column: status band wraps to multiple rows; `#histTrends` becomes a
    horizontal scroll strip (`display:flex; overflow-x:auto;
    -webkit-overflow-scrolling:touch; scroll-snap-type:x proximity`, cards
    `min-width: 240px; scroll-snap-align:start`) OR stacked — strip preferred.
  - Epoch table rows allow horizontal scroll within the panel rather than wrapping
    chaos; chevron/epoch buttons, health pill, chip clear, game rows all >= 44px tap
    targets with `touch-action: manipulation`.
  - Hover tooltip is desktop-only; on touch, chart tap = epoch filter (already via
    click). Native form controls untouched.
  - Verify no horizontal overflow of `.history-page` at 390px width (DOM reasoning is
    fine; no server start).

H11. FINAL STEP: cache-bust. In index.html change BOTH `?v=20260611-dbg3` occurrences
    (line ~7 stylesheet, line ~531 script) to `?v=20260611-hist1`, then run
    `node --check app.js` one last time. (If a later fix touches app.js/styles.css
    after this bump, re-bump to -hist2 etc. — a stale ?v= silently serves old code
    through the WSL proxy.)

---

## 5. ACCEPTANCE (binding)

A1. With a live training run selected: status band shows stage pill (with live dot +
    games/pos-s during live selfplay), the conditional progress bar per H2 rules,
    resource chips, and a colored health pill — one viewport row on desktop. Clicking
    the pill toggles the messages drawer; the open state survives the 15s refresh.
A2. With >= 2 epochs: trends grid shows at least T1 loss, T2 game length, T3 win
    balance, T5 speed (T4 when evaluations exist, T6 when a real buffer pool exists);
    hover shows crosshair + per-series values; the latest epoch is visibly ticked.
A3. Clicking epoch N on any chart, an epoch-table epoch button, re-clicking to
    toggle, or the chip's × — filters/unfilters the game list client-side with the
    removable chip visible; Load more keeps working while filtered and newly loaded
    epoch-N games appear.
A4. Epoch table: last 15 epochs, one row each, live row accented; expanding shows
    buffer/losses/calibration chips on runs with a real pool and breaks nothing on
    loss-only runs.
A5. Game rows are one line with winner dot, length micro-bar, source badge; selecting
    fills the detail panel; for .hxr games a final-position thumbnail appears after
    one lazy fetch (cached on reselect); Open in Debug deep-links exactly as before.
A6. At <= 900px: single column, charts in a touch-scrollable strip, all new
    interactive targets >= 44px with touch-action manipulation, native selects.
A7. Match and Debug screens pixel-identical; `node --check app.js` passes; zero new
    fetch call sites except H9's lazy detail fetch; the 15s auto-refresh, run switch,
    filters, sort, search, paging, autoload window, Replay, and artifact-panel
    debug/load buttons all behave exactly as before.

## 6. RISKS / IMPLEMENTER NOTES

- `renderGameHistoryPage` is called from `renderTraining` (Match screen path) and from
  many async guards — it must stay null-safe and idempotent; never assume the History
  screen is visible when it runs.
- Both agents edit `renderGameHistoryPage`; agent 2 must preserve agent 1's band/trends
  call sites and only replace the H1 epoch-table stub + list/detail internals.
- `training_progress` and `selfplay.buffer` pool stats are OPTIONAL; every
  region must render with a loss-only run (no buffer pool, no progress file), a run
  with a full buffer pool, and a stopped run (no live status) without throwing.
- `epoch` can be null on game items (`e--`), `winner` null, `seed` null — keep the
  existing "--" conventions.
- innerHTML re-render wipes DOM state every 15s: drawer/expanded/filter state lives in
  the module vars of §2 only.
- Do not use `<style>` injection or template libs; all CSS in styles.css.
- The thumbnail fetch replays the game server-side; never prefetch it for list rows,
  only for the currently selected detail item.
