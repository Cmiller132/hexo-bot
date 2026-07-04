# Match Screen v2 — "Arena" Implementation Spec

Status: FINAL. Binding contract for the implementation agents.
Scope: FULL overhaul of the Match screen (frontend) PLUS the minimal backend extension
that makes model/checkpoint players and bot-vs-bot series possible. All data fields
below were verified against `web.py` / `debug_worker.py` / `debug_infer.py` /
`dashboard.py`; if code and spec disagree on a FIELD NAME, the names
here win (they were read from the live code). The only model lineage is `hexfield`;
its checkpoints are the only kind that exist.

Design language: copy the Debug screen (`#debugScreen`, `.dbg-*`) — strip/panel/chip
layout, 1px borders, 8px radii, `--panel` backgrounds, delegated listeners, null-safe
renders, ids + `data-*` hooks, keyboard shortcuts with button equivalents.

Targets (the ONLY files that may be modified):
- `packages/hexo_frontend/python/hexo_frontend/web.py`            (backend agent only)
- `tests/test_hexo_runner_match_mode.py`                          (backend agent only)
- `packages/hexo_frontend/python/hexo_frontend/static/app.js`
- `packages/hexo_frontend/python/hexo_frontend/static/index.html`
- `packages/hexo_frontend/python/hexo_frontend/static/styles.css`

Hard constraints (do not violate):
- Do NOT touch `debug_service.py`, `debug_worker.py`, `debug_infer.py`,
  `dashboard.py`, any other backend file, or the History/Debug screens' HTML/JS/CSS
  (exception: the shared-code edits explicitly listed in §6).
- No frameworks, no build step, hand-rolled SVG only.
- New element ids prefixed `mt`, new CSS classes `.mt-*`.
- `node --check app.js` must pass after EVERY app.js edit.
- `python -m py_compile packages/hexo_frontend/python/hexo_frontend/web.py` (any
  python ≥3.11) must pass after every web.py edit.
- Do NOT git commit. Smoke testing uses a separate port (§9), never the running
  dashboard.
- Run tests from the repo root with the `.venv` active:
  `python -m pytest tests/test_hexo_runner_match_mode.py -x -q`.

---

## 1. VERIFIED DATA SHAPES (normative)

### 1.1 Current match payload (`/api/state`, `/api/new`, `/api/move` responses)
From `ManualMatchController._payload_locked` + `dashboard_state`:
- Board: `current_player`, `phase`, `first_stone`, `winner`
  (`"player0"|"player1"|null`), `terminal_reason`, `placements`
  (list of `{q,r,player,phase,index}`), `legal` (list of `{q,r}`), `legal_count`,
  `tactics` (LARGE object — the new frontend IGNORES it; do not remove server-side).
- Match: `version` (int, monotone), `game_id`, `mode`, `players`
  (`{player0:{role,kind,label,...}, player1:{...}}`), `turn_status`
  (`"human_turn"|"bot_thinking"|"terminal"|"error"`), `can_submit` (bool),
  `thinking_player` (`"player0"|"player1"|null`), `last_bot_decision`
  (`{player, duration_ms, q?, r?, diagnostics?, error?}|null`), `error`
  (string|null), `match` (`{players, time_limit, seed}` echo of config).
- Long-poll: `GET /api/state?since=<version>&timeout_ms=<ms≤30000>`.
- `POST /api/move {q,r}` → 409 + `{error, state}` on `MoveConflict`.

### 1.2 Debug worker ops (reused by the checkpoint player; do NOT modify them)
- `search` (debug_infer.search_position → `_search_hexfield`): fresh
  reproducible CPU MCTS, NO root noise. The search runs with the run's AS-TRAINED
  profile (read from the run's manifest — the Gumbel-Top-m + Sequential Halving
  levers and the budget-calibrated `gumbel_m`), and move selection happens
  IN-SEARCH via the eval protocol (sampled opening plies, argmax after); the
  `temperature` param is that in-search selection temperature (0 = the greedy
  debug read). Returns
  `{visits_requested:int, visits:int, root_value:float, best_action_id:int,
  best:{q,r}, visit_policy:[{action_id,q,r,p,w}...], root_prior:[...],
  selection_in_search:true, search_profile:{...}}`.
  `root_value` is side-to-move perspective.
- `analyze` (debug_infer.analyze_position): one forward. Returns
  `{current_player:int, value:float, policy:[{action_id,q,r,p}...] SORTED p-desc,
  opp_policy, value_dist, ...}`. `policy[0].action_id` is the prior argmax over
  LEGAL cells only (rows are built from legal_action_ids).
- Access pattern (web.py): `_debug_worker().cached(signature, op, timeout=..., …)`
  with `signature = _debug_signature(prefix, ckpt_path, action_ids, n)`.
  `DebugWorkerError` = infra failure (worker restarted); `RuntimeError` =
  deterministic request error. Worker is a single CPU process with one lock and a
  3-model LRU — two checkpoint players fit; calls serialize (acceptable).
- Timing: worker `search` uses the Rust MCTS session with batch-1 CPU inference —
  interactive at ≤512 visits (the debug screen's default); `search_tree` (~25ms/visit)
  is the SLOW one and is NOT used here.
- Checkpoint resolution: `_debug_resolve_checkpoint(run_name, name)` → Path under
  `<run>/checkpoints/`, raises ValueError when unknown. Checkpoint lists:
  `GET /api/debug/checkpoints?run=` → `{run, checkpoints:[{name, epoch:int|null,
  size, mtime, latest:bool, graft}], lineage, worker:{...}}` sorted latest-first.
- Run list: `GET /api/training/runs` → `{roots, runs:[{name, path, ...,
  modified}]}` sorted newest-first (already loaded into `trainingRuns` by
  `loadTrainingRuns()`).

### 1.3 Engine facts
- `pack_coord_id(coord)` / `unpack_coord_id(id)` live in `hexo_engine.types`
  (web.py already imports `unpack_coord_id`; ADD `pack_coord_id` to that import).
- `engine.to_python_state(state).placement_history` is the ordered list of
  `PythonPlacementRecord` (`.coord` AxialCoord, `.player`, `.placement_index`) —
  the full action history of the game so far, sufficient to rebuild `action_ids`.
- `GameSpec.mode` is a free string (verified `hexo_runner/session.py:24`); only
  `scenario` must stay None for recorded runs. `max_actions` default 1024.
- JS `dbgPackActionId(q, r)` (app.js:4048) packs the same action-id format; the
  Match screen may CALL it (pure function) for the Open-in-Debug deep link.
- Debug deep link by action list: the debug screen's nav supports an imported
  position via `acts` (`dbgNavigate({run, acts, ply})` → server
  `/api/debug/position?actions=…`). Verified at app.js:4566 and web.py
  `_debug_position_from_actions`.

### 1.4 Frontend machinery that MUST keep working (shared with History screen)
`loadTrainingRuns`, `loadTrainingRun`, `fetchTrainingRunDetail`, `trainingRuns`,
`trainingRun`, `trainingRunDetails`, `trainingLoadError`, the entire History screen
(`renderGameHistoryPage` and everything it calls), the entire Debug screen
(`dbg*`), `setScreen`/`navigateScreen`/`screenFromHash`, the board pipeline
(`buildBoardModel`, `renderBoard`, pan/zoom/pinch, `center`, `path`, `HEX`,
`SQRT3`, `fitBoard`, `zoomBoard*`, `clientToBoardPoint`), polling
(`pollState`, `schedulePoll`, `abortPoll`, `applyState`, `isNewerOrSameState`),
`post`, `safeJson`, `setPending`, `escapeText`, `escapeAttr`, `displayValue`,
`playerColor`, `phaseLabel`, replay machinery (`setReplayIndex`, `resetReplay`,
`toggleReplayPlay`, `stopReplay`, `visiblePlacements`, `viewedPlacementCount`,
`totalPlacements`, `isLiveView`), `renderMoveHistory`/`ensureMoveHistoryEvents`,
`showTip`/`hideTip`/`updateHud`/`cellInfo`.

---

## 2. PRODUCT SCOPE

### Kept (existing behavior, re-skinned)
Board + pan/zoom/fit + hover HUD/tip, turn overlay, replay bar + move-history dock,
manual play (click a legal cell), SealBot players (current/best, time limit), seed,
long-poll live updates, adapter availability display.

### Cut (delete frontend code + UI; server keeps emitting `tactics` untouched)
- ALL tactics UI: Tactical Filters card, Tactical Stats card, Inspect, tactics
  board overlays/badges, windows explorer, cell inspector (§7 delete list).
- The Match screen's "Training Runs" sidebar card (duplicated by the History
  screen). The DATA layer stays (§1.4); only the match-side rendering goes.
- The 10-row "State Summary" panel (replaced by a compact status row).

### New
1. **Per-seat player config**: Manual | SealBot current | SealBot best |
   Checkpoint (run + checkpoint + visits + mode). Any combination — human vs
   model, model vs model, SealBot vs model, human vs human.
2. **Series**: N games (1/3/5/9/15/25), optional seat alternation, running tally
   by configured slot, per-game results, auto-advance to the next game.
3. **Stop**: a real stop control (no auto-restart), safe while a bot is searching.
4. **Bot insight**: per-move decision log (server-side, poll-proof) → value
   sparkline over the game (P0 perspective), per-seat last-move/duration/value.
5. **Open in Debug**: deep-link the current (possibly replay-scrubbed) position
   into the Debug screen with the relevant run/checkpoint preselected.
6. **Keyboard shortcuts** (match screen only, not when an input has focus):
   `←/→` replay step, `Home/End` start/live, `Space` replay play/pause,
   `N` new match/rematch.

---

## 3. BACKEND CONTRACT (web.py — agent B)

### 3.1 Player specs (config in, payload out)
`POST /api/new` body:
```
{ players: { player0: <spec>, player1: <spec> },
  time_limit: float,            # sealbot only
  seed: int|null,
  series: { games: int, alternate: bool } | absent }
```
`<spec>` accepted forms (normalize in ONE function, replacing the string-only
`_normalize_player_kind` — keep accepting every form the old endpoint accepted):
- strings `"manual"|"human"`, `"sealbot"|"bot"`, `"sealbot-current"`, `"sealbot-best"`
- dict `{kind:"manual"}`; `{kind:"sealbot"|"bot", variant:"current"|"best"}`
- NEW dict `{kind:"checkpoint", run:str, checkpoint:str, visits?:int, mode?:str,
  c_puct?:float}` — `visits` clamp 8..2048 default 256; `mode` ∈
  `"search"`(default)|`"policy"`; `c_puct` clamp 0.1..10 default 1.5.
  Resolve `run`+`checkpoint` via `_debug_resolve_checkpoint` AT PARSE TIME so a bad
  config 400s before any thread starts.
Normalized internal spec: a dict `{kind, variant?, run?, checkpoint?, visits?,
mode?, c_puct?}`; `_player_setup` becomes `dict[str, dict]`. Update every
`_is_sealbot_kind(...)` call site (`_turn_status_locked`, `_can_submit_locked`,
`_parse_match_config` mode derivation, `_player_payload`) to spec-dict logic:
"is bot" = `kind != "manual"`.
- `mode` (payload + GameSpec.mode + game_id prefix): `"manual"` if both manual,
  else `"checkpoint"` if any checkpoint seat, else `"sealbot"`.
- `_player_payload` → `{role, kind:"manual"}` |
  `{role, kind:"sealbot", variant, label:"SealBot <variant>", adapter_id:"sealbot"}` |
  `{role, kind:"checkpoint", run, checkpoint, visits, mode, label}` with label
  `"<run> @ <ckpt-short>"` where ckpt-short = `e<epoch>` parsed via
  `_DEBUG_CKPT_EPOCH_RE` else the filename stem (e.g. `latest`).
- Back-compat: legacy `{mode, human_player, bot:{variant}}` body shape and the
  no-body `reset()` (manual/manual) keep working — existing tests rely on it.

### 3.2 `_CheckpointBotPlayer`
New class implementing the runner player protocol (mirror `_ManualPlayer`'s
methods; `setup_worker/observe_transition/finish_game/close` are no-ops;
`start_game` captures a per-game token from `GameContext`
`{game_id, seed, player_index}` for the opening-sampling RNG):
- `identity = PlayerIdentity(player_id=f"ckpt-{run}-{ckpt-stem}", label=<§3.1 label>)`.
- `decide(state)`:
  `acts = [pack_coord_id(rec.coord) for rec in engine.to_python_state(state).placement_history]`;
  mode `"search"` → `_debug_worker().cached(_debug_signature(f"match-search:{visits}:{c_puct}:{temperature}:{seed}", ckpt_path, acts, None), "search", timeout=<below>, checkpoint=str(ckpt_path), action_ids=acts, visits=visits, c_puct=c_puct, seed=<per-(game,ply)>, temperature=<below>)`.
  Selection follows the trainer's `[model.config.evaluation]` arena protocol:
  the first `CHECKPOINT_OPENING_MOVES` (8) plies sample the opening at
  `CHECKPOINT_OPENING_TEMPERATURE_IN_SEARCH` (1.0), argmax afterwards. For the
  hexfield lineage this selection happens IN-SEARCH (the search runs the run's
  as-trained Gumbel profile and returns the tempered/greedy pick as
  `best_action_id`, flagged `selection_in_search:true`), so the bot plays
  `best_action_id` directly. The per-(game, ply) seed is derived from the
  `game_token` (process-stable, decorrelates series games like eval's
  per-(game, move) seeds) and rides the cache signature so distinct games never
  share a cached opening move.
  Diagnostics `{root_value, visits, mode, run, checkpoint,
  selection: "in-search-opening"|"in-search-argmax", search_profile}`;
  mode `"policy"` → worker `analyze` (signature prefix `"match-policy"`), take
  `policy[0]["action_id"]` (rows verified sorted p-desc, legal-only), diagnostics
  `{root_value: <analyze value>, mode, run, checkpoint, top_p: policy[0].p}`.
  Empty `policy` → raise RuntimeError (aborts the game loudly, never silently).
  Return `DecisionResult(action=engine.PlacementAction(unpack_coord_id(aid)), diagnostics=…)`.
- Per-move worker timeout: `min(300.0, max(60.0, visits * 0.15))`.
- Worker errors propagate — `_ObservedBotPlayer` already routes them to
  `bot_decision_failed` → `error` in the payload. Wrap in `_ObservedBotPlayer`
  exactly like SealBot.
- Testability: controller `__init__` gains
  `checkpoint_factory: Callable[[dict], object] | None = None` (mirrors
  `bot_factory`); when set, `_make_player` uses it for checkpoint specs so tests
  inject scripted players and never touch the worker/torch.

### 3.3 Series
- Config: `series.games` clamp 1..25 default 1; `alternate` bool default False.
  Slot semantics: **slot0 = the configured `players.player0` spec; slot1 =
  `players.player1`**. Game g (0-based): seats as configured when
  `not alternate or g % 2 == 0`, else swapped. `_player_setup` must reflect the
  ACTUAL seat assignment of the game in progress (so `players`,
  `turn_status`, `can_submit` stay seat-correct), plus keep a
  `_seat_slots: {player0:"slot0"|"slot1", player1:…}` map per game.
- The match thread becomes a series loop: for each game, fresh players via
  `_make_player`, `GameSpec(game_id=f"{mode}-{n}-g{g+1}",
  seed=(seed+g if seed is not None else None), mode=mode)`, `run_match(...)`;
  on completion update the tally (winner seat → winner slot via `_seat_slots`,
  draw when winner None), append the result row, bump version/notify; continue
  unless cancelled, a game aborted (`result.abort`), or an exception occurred.
  Between games clear per-game fields (`_decision_log`, `_pending_action`,
  `_thinking_player`, `_last_bot_decision`); leave the finished board state
  visible until the next game's first `decide/observe` replaces it.
- Payload `series` (null when games == 1):
  `{games:int, played:int, current_game:int (1-based), alternate:bool,
  finished:bool, tally:{slot0:int, slot1:int, draws:int},
  slots:{slot0:<player_payload-shaped dict>, slot1:…},
  seats:{player0:"slot0"|"slot1", player1:…},
  results:[{game:int, winner_seat:"player0"|"player1"|null,
  winner_slot:"slot0"|"slot1"|null, length:int}]}`.

### 3.4 Stop + abandoned-thread safety
- `POST /api/match/stop` (no body) → `controller.stop()` → payload.
- `stop()`: under the condition, set `_cancelled = True`, set `_stopped = True`,
  bump version, `notify_all`, and return the payload WITHOUT joining (a checkpoint
  search may be minutes inside the worker; the thread is a daemon and exits at its
  next controller callback). Payload gains `stopped: bool`;
  `_can_submit_locked` returns False and `submit_move` raises `MoveConflict`
  when stopped; `_turn_status_locked` returns `"stopped"` when stopped (frontend
  treats unknown statuses as idle).
- Generation guard: controller keeps `self._generation: int`, incremented in
  `reset()` and `stop()`. The series thread captures its generation; every
  controller callback that mutates state (`decide`, `bot_decision_*`,
  `observe_transition`, series bookkeeping) takes/checks the generation and
  raises `RuntimeError("manual match reset")` for a stale one. `close()` keeps
  joining (5s) but on timeout ABANDONS the thread (set `_thread=None`, no raise)
  instead of raising — the generation guard makes the orphan harmless. This
  fixes "Rematch while a 60s checkpoint search runs" (today: RuntimeError).

### 3.5 Decision log
- `self._decision_log: list[dict]`, cleared per game. In `bot_decision_finished`
  append `{ply: len(self._python_state.placement_history) if self._python_state else None,
  player, q, r, duration_ms, value: diagnostics.get("root_value"),
  visits: diagnostics.get("visits"), kind: <seat spec kind>}` (q/r only when the
  action was a placement; `ply` = placements BEFORE the move, i.e. the move index).
  In `bot_decision_failed` append `{ply, player, error: "<type>: <msg>", kind}`.
- Payload `bot_decisions: list` (the current game's log; ≤ `max_actions` entries).

### 3.6 Tests (extend `tests/test_hexo_runner_match_mode.py`; keep every existing
test passing unchanged)
- Checkpoint spec normalization: dict spec → normalized fields, visits/c_puct
  clamps, unknown checkpoint → ValueError at `reset()` (use tmp run dir or assert
  the error type), legacy string/`{mode,human_player,bot}` forms still normalize.
- Scripted `checkpoint_factory` game: ckpt-vs-ckpt series of 3 with
  `alternate=True` — assert tally maps to SLOTS (not seats), `series.results`
  length 3, `seats` flip on game 2, `bot_decisions` non-empty with `value` set,
  payload `players.player0.kind == "checkpoint"` shape.
- Stop: during a manual game, `stop()` → `stopped` True, `submit_move` raises
  `MoveConflict`, `reset()` afterwards starts a fresh game.
- Run via the pytest command in the header; also run
  `tests/test_frontend_training_artifacts.py` (guards the untouched endpoints).

---

## 4. FRONTEND LAYOUT (index.html — replace `#matchScreen`'s contents entirely)

```
main#matchScreen .app-screen .match-screen
  section.mt-grid
    <!-- Region 1: setup strip (mirrors .dbg-ctx) -->
    section#mtSetup .mt-setup
      .mt-seat[data-seat="player0"]
        .mt-seat-head: color dot (var(--p0)) + "P0"
        select#mtKind0  [Manual | SealBot current | SealBot best | Checkpoint]
        .mt-ckpt-cfg[data-seat="player0"] (hidden unless kind=checkpoint):
          select#mtRun0, select#mtCkpt0, input#mtVisits0 (number 8..2048, value 256),
          select#mtSearchMode0 [search|policy]
      .mt-vs "vs"
      .mt-seat[data-seat="player1"] (same, ids suffixed 1, dot var(--p1))
      .mt-opts: label Time input#mtTimeLimit (0.01..30 step 0.01, value 0.05) ·
        label Seed input#mtSeed · label Series select#mtSeriesGames
        [1 game|3|5|9|15|25] · label.mt-check input#mtAlternate[type=checkbox]
        "alternate seats"
      .mt-actions: button#mtStartBtn.primary-action "Start match" ·
        button#mtStopBtn "Stop" · span#mtSetupNote.mt-muted (adapter/worker status)
    <!-- Region 2: board stage -->
    section#mtBoardStage .mt-board-stage
      KEEP byte-for-byte the existing .board-area subtree (#boardArea #boardSvg
      #turnOverlay #turnOverlayTitle #turnOverlaySub #zoomInBtn #zoomOutBtn
      #cellHud #tip) BUT trim .legend to P0/P1/Legal (drop .tactic-legend spans),
      and KEEP the existing .replay-bar subtree (all replay ids) below it.
    <!-- Region 3: side rail -->
    aside#mtSide .mt-side
      section.panel-card#mtStatusCard
        .mt-turn-banner#mtTurnBanner: span#mtTurnDot + strong#mtTurnTitle +
          span#mtTurnSub
        .mt-facts#mtFacts (compact chips: move number, phase, stones, legal count,
          game id, seed — rendered by JS)
      section.panel-card#mtPlayersCard  → div#mtPlayers (rendered by JS)
      section.panel-card#mtInsightCard
        .mt-insight-head: "Insight" + button#mtOpenDebugBtn "Open in Debug"
        div#mtValueChart  (sparkline svg; empty-note when no bot values)
      section.panel-card#mtSeriesCard hidden → div#mtSeries (rendered by JS)
  section.bottom-dock (KEEP: .history-panel .dock-title "Move History" +
    #moveHistory)
```
Remove from index.html: the whole old `.top-grid` side-stack (Match Setup card,
State Summary card, Controls card, SealBot Insight card `#sealbotCard`/`#botPanel`,
Tactical Filters card `#modeSeg #playerSeg #axisSeg #inspectBtn`, Tactical Stats
card `#tacticsPanel`, Training Runs card `#trainingRunSelect #trainingRefreshBtn
#trainingSummary #trainingArtifacts`), `#fitBtn`/`#tacticsBtn` toolbar (zoom
buttons remain; "Fit" becomes `#mtFitBtn` next to zoom in the board view
controls), and the old `.tactic-legend` spans. Everything History/Debug stays
byte-identical.

---

## 5. FRONTEND WORK ITEMS

### Agent F1 — shell (index.html + styles.css + app.js reconciliation)
F1.1 Rewrite `#matchScreen` HTML per §4 (KEEP the board-area + replay-bar +
  bottom-dock subtrees and their ids; add `#mtFitBtn` to `.board-view-controls`).
F1.2 styles.css: add `.mt-grid` (CSS grid: setup strip full-width row; board
  ~1fr + side rail ~340px column; single column ≤ 900px), `.mt-setup`,
  `.mt-seat`, `.mt-ckpt-cfg`, `.mt-opts`, `.mt-actions`, `.mt-turn-banner`,
  `.mt-facts`, `.mt-player-row`, `.mt-chip`, `.mt-value-chart`, `.mt-series-*`,
  `.mt-muted`, `.mt-thinking-dot` (pulse keyframes — copy `.hist-live-dot`'s
  pattern). Visual language: copy `.dbg-ctx`/`.panel-card` rules. All new
  interactive targets ≥ 44px tall on mobile, `touch-action: manipulation`.
F1.3 app.js minimal reconciliation so `node --check` passes AND the page boots
  with zero console errors: bind the new ids null-safely (the `on(...)` helper),
  fix the UNGUARDED `trainingRunSelect.addEventListener` at ~line 179 and the
  `trainingArtifacts.addEventListener` at ~194 (these elements are gone —
  delete both bindings; `syncTrainingRunSelect` body becomes a no-op returning
  early — keep the function, History calls it at 390/424/431),
  `loadTrainingRuns` line ~378: drop the `trainingRunSelect.value` fallback
  (use `(trainingRun && trainingRun.name) || (historyRunSelect && historyRunSelect.value) || ""`).
  Replace every `renderTraining()` call site (≈ lines 403, 409, 425, 440, 672,
  705) with `renderGameHistoryPage()` and delete `renderTraining` +
  `trainingArtifactRow` + the `trainingSummary/trainingArtifacts/
  trainingRunSelect` const bindings. Stub `renderMatchControls`/`render` enough
  that the board still renders and manual play works (full rewrite lands in F2).
  Delete the `#fitBtn/#tacticsBtn` bindings; bind `#mtFitBtn` → `fitBoard`.
F1.4 `node --check`; verify with grep that NO reference to a deleted id remains
  in app.js (`tacticsBtn|inspectBtn|modeSeg|playerSeg|axisSeg|sealbotCard|
  botPanel|tacticsPanel|trainingRunSelect|trainingSummary|trainingArtifacts|
  fitBtn` — `#mtFitBtn` excepted).

### Agent F2 — setup + match flow (app.js)
F2.1 New module state near `matchConfig`:
  `mtSetup = { seats: { player0: {kind:"manual"}, player1: {kind:"sealbot", variant:"current"} },
  time_limit: 0.05, seed: null, series: {games: 1, alternate: false} }`;
  `mtCkptLists = new Map()` (run → checkpoints array, from
  `/api/debug/checkpoints?run=`), `mtCkptListsLoading = new Set()`.
  DELETE the old `matchConfig` + `PLAYER_KIND_LABELS` once nothing reads them.
F2.2 `renderMtSetup()` (idempotent, called from `render()` and on adapter/run
  loads): populate kind selects (SealBot options disabled-with-title when the
  adapter variant is unavailable — reuse `sealbotVariants()`/`sealbotAdapter()`
  logic, keep those helpers); when a seat kind is `checkpoint`, unhide its
  `.mt-ckpt-cfg`, populate `#mtRunX` from `trainingRuns` (lazy
  `loadTrainingRuns()` if empty), `#mtCkptX` from `mtCkptLists` (lazy fetch,
  cache, show "loading…" option meanwhile; default selection = first item =
  newest). `#mtSetupNote`: adapter status line (reuse the old
  `renderAdapterStatus` logic, write into the new element) + "checkpoint bots
  run on the shared CPU debug worker" hint when any seat is checkpoint.
  Inputs never clobbered while focused (`document.activeElement` guard —
  copy the old `renderMatchControls` pattern).
F2.3 `buildMtMatchPayload()` → §3.1 body (seat dicts; omit `series` when
  games==1). `#mtStartBtn` → POST `/api/new` (reuse `post()`), label "Start
  match" / "Rematch" / "Start series" by config; disabled when a selected
  SealBot variant is unavailable or a checkpoint seat has no run/ckpt resolved.
  `#mtStopBtn` → POST `/api/match/stop`; disabled when `state.stopped` or
  terminal-and-no-series-remaining.
F2.4 Seat/series change handlers (delegated `change` listener on `#mtSetup`)
  update `mtSetup` and re-render (no fetch until Start).
F2.5 Status card: `renderMtStatus()` — turn banner (to-play / thinking / winner /
  stopped / error; reuse `turnStatus()` semantics but tolerate `"stopped"`),
  facts chips (move `placementStepLabel()`-equivalent, phase via `phaseLabel`,
  stones, `legal_count`, game_id, seed). Rewrite `turnStatusLabel`/
  `renderTurnOverlay` for the new player model (`players.playerX.kind`,
  checkpoint labels) — keep ids `#turnOverlay*` (board subtree kept).
F2.6 `render()` becomes: board model → `renderBoard` → `renderMtSetup` →
  `renderMtStatus` → `renderMtPlayers` (F3 stub ok) → `renderMtInsight` (F3
  stub ok) → `renderMtSeries` (F3 stub ok) → `renderMoveHistory` →
  `renderReplay`. `buildBoardModel` loses its `tacticMaps` field (return
  placements/legal only); `renderBoard` drops heat/threat/badge overlay calls
  and the `cell-badge` markup. Manual play, replay, pan/zoom must keep working.

### Agent F3 — insight + series + polish (app.js)
F3.1 `renderMtPlayers()`: one `.mt-player-row` per seat — color dot, payload
  `players.playerX.label` (+ kind chip incl. visits/mode for checkpoints),
  pulsing `.mt-thinking-dot` when `thinking_player === seat`, last decision for
  that seat from `bot_decisions` (move, `duration_ms` formatted, `value`
  ±0.000 chip), seat error line when a `bot_decisions` entry for the seat has
  `error` or payload `error` mentions it. Manual seats: "your turn" affordance
  when `can_submit` and it's that seat's turn.
F3.2 `renderMtInsight()`: hand-rolled sparkline svg (viewBox ~"0 0 300 90",
  pattern of `histChartSvg` but simpler): x = `ply`, y = value in [-1,1]
  mapped P0-perspective (`entry.player === "player0" ? v : -v`), one polyline
  per seat (colors var(--p0)/var(--p1)), zero axis line, dot on the latest
  point, y labels +1/0/−1. Series = `bot_decisions` entries with finite
  `value`. < 2 points → `.dbg-empty-note`-style "No bot evaluations yet".
  Hover (single `mousemove`/`mouseleave` pair, attribute-only updates):
  nearest-ply crosshair + `e.g. "ply 42 · P0 +0.31"` title text. Click →
  `setReplayIndex(ply)` (scrub the board to that move).
F3.3 `renderMtSeries()`: hidden when `!state.series`; else tally line
  `"<slot0.label> N — M <slot1.label>"` (+ draws when > 0), progress
  `"game K/G"` (+ "(seats swapped)" when `seats.player0 === "slot1"`),
  result chips per game (✓ winner slot color / `=` draw, title = length), and
  "series finished" state.
F3.4 Open in Debug (`#mtOpenDebugBtn`): build
  `acts = state.placements.slice(0, viewedPlacementCount()).map(p => dbgPackActionId(p.q, p.r))`;
  pick run+ckpt: first checkpoint seat's `{run, checkpoint}` from the payload,
  else `trainingRuns[0].name` + no ckpt. Then
  `dbgNavigate({ run, acts, ply: acts.length, ckptA: ckpt || "" })` +
  `navigateScreen("debug")` (order/exact patch keys: match what
  `debugOpenFromHistory` does at app.js:4172 — read it first). Disabled when
  no placements or no runs.
F3.5 Keyboard shortcuts per §2-New-6: one `keydown` listener; ignore when
  `event.target` is an input/select/textarea or any other screen is active
  (`activeScreen !== "match"`); every shortcut has a button equivalent.
F3.6 Mobile ≤ 900px: setup strip wraps; side rail stacks under the board; chips
  scroll horizontally — extend the existing media query with `.mt-*` rules.

### Agent F4 — cleanup + cache-bust (app.js + styles.css + index.html)
F4.1 Delete (verify each is unreferenced first — grep before delete):
  state `tacticsOn, selectedWindowId, selectedCellKey, tacticFilters,
  tacticsView`; functions `renderCellBadge, renderHeatOverlay,
  renderThreatOverlay, renderTacticsPanel, renderTacticsTabs,
  renderTacticsOverview, renderCellEmptyState, renderCellInspector,
  renderWindowInspector, renderFactSection, renderWindowsExplorer,
  renderWindowGroups, renderWindowGroup, renderWindowCard, renderWindowSlot,
  renderWindowTags, groupedWindows, windowPrioritySort, windowScore,
  windowCountMetric, bindTacticsPanel, buildTacticMaps, emptyTacticMaps,
  visibleWindows, addRole, addHeat, addThreatHeat, heatMax,
  windowMatchesFilters, findWindow, cellDebug, factsForWindow, renderSlot,
  maskRow, maskBits, flag, playerPill, idList, clearTacticSelection, metric,
  renderBotPanel, botMetric, normalizeBotDecision, activeBotVariantLabel,
  adapterErrors, isSealBotMatch, setupHasSealBot, renderControls (fold the
  body-class toggles still needed — `pending`, `replay-mode`, `bot-thinking`,
  `state-error` — into `render()`/`renderMtStatus`), renderMatchControls,
  renderAdapterStatus (logic reused inside renderMtSetup), buildNewMatchPayload,
  playerKindAvailable, syncDefaultVariant, renderStatus (replaced by
  renderMtStatus — CHECK: it also drives the header `#statusText` dot; keep
  that bit), setText + the old info-row updates, matchLabel, playerSlotLabel`
  — KEEP `playerMeta/playerKind/playerKindLabel/playerLabel/playerShort`
  updated for the new payload kinds (used by move history/overlay), KEEP
  `sealbotVariants/sealbotAdapter/hasAvailableSealBotVariant/
  sealbotDefaultVariant`.
F4.2 styles.css: delete `.tactics-*`, `.overlay-controls`, `.seg` rules IF
  match-only (CHECK: `.seg` is ALSO used by the Debug mode bar `.dbg-mode-seg`
  — trim selector lists, never delete shared rules), `.window-*`, `.heat-cell`,
  `.threat-heat`, `.cell-badge`, `.bot-*`, `.sealbot-card`, `.adapter-*` (if
  fully replaced), `.training-card/.training-panel/.training-summary/
  .training-artifacts/.training-toolbar`, `.match-form/.setup-grid/.match-grid/
  .field-block` (if unused after F1 — `.field-block` IS used by the Debug
  screen: keep), `.info/.info-row/.turn-banner` (old state summary) and their
  media-query mentions. Same combined-selector caution as the history spec.
F4.3 index.html: bump BOTH asset refs to `?v=20260611-match1`; app.js: set
  `APP_VERSION = "20260611-match1"`.
F4.4 Final `node --check`; grep-verify zero references to every deleted
  id/class/function; verify the History + Debug screens' HTML blocks are
  byte-identical to before (git diff shows no lines outside `#matchScreen` +
  head asset refs).

---

## 6. DO NOT TOUCH (exact names; behavior frozen)

Backend: every existing endpoint's response shape for History/Debug
(`/api/training/*`, `/api/debug/*`) and `/api/adapters`; `dashboard_state`
(tactics stays in the payload — the new UI just ignores it);
`discover_sealbot_adapters`; `run_match`; record formats.
Frontend: everything in §1.4, the whole `dbg*` module, the whole History module
(`hist*`, `renderGameHistoryPage` and below), `screenFromHash/navigateScreen/
setScreen` semantics, the `#statusText` header dot behavior, `on()` helper,
`__diagBar/__renderDiag/reportError/diagTap`.

---

## 7. ACCEPTANCE (binding)

A1. Manual vs SealBot plays exactly as before (click-to-move, thinking overlay,
    terminal banner) in the new layout; adapter-unavailable degrades to a
    disabled option + note, never a dead Start button with no explanation.
A2. Checkpoint vs Checkpoint: pick two different checkpoints of
    `<your run>` (e.g. e1 vs e2) at low visits; Start; the board
    advances live with the thinking indicator alternating; the value sparkline
    fills in; Stop halts it; Rematch works immediately even mid-search.
A3. Human vs Checkpoint: human moves submit; bot replies; per-seat last
    move/duration/value update; policy mode replies fast (< ~2s/move).
A4. Series of 3+ with alternate seats: tally counts by configured SLOT (a model
    that wins as P0 then as P1 shows 2 for its slot), per-game chips appear,
    series finishes cleanly; payload `seats` flips on even games.
A5. Open in Debug lands on the Debug screen at the SAME position (incl. when
    replay-scrubbed to an earlier ply), with the checkpoint preselected when a
    checkpoint seat exists.
A6. Replay bar, move-history dock, pan/zoom/fit/pinch, hover HUD all work; the
    keyboard shortcuts work and never fire while typing in inputs.
A7. History and Debug screens are pixel-identical; `node --check` passes;
    `py_compile` passes; the pytest commands in §3.6 pass; no console
    errors on load; cache-bust bumped on BOTH index.html refs.
A8. ≤ 900px: single column, setup strip wraps, all new targets ≥ 44px.

## 8. RISKS / IMPLEMENTER NOTES

- `applyState`/`isNewerOrSameState` gate on `version` + `game_id` — series
  advances change game_id mid-poll; verify a new game's version=low payload is
  not discarded as stale (`isSameGame` returns false → accept). Read those
  functions before touching the flow.
- The long-poll returns the LATEST state only; never derive per-move data from
  poll deltas — that is exactly why `bot_decisions` is server-side.
- `renderMoveHistory` keys off `placements.length` + last index as its rebuild
  signature; a new game in a series can have the same length momentarily —
  extend the signature with `game_id` (allowed: it is match-screen code).
- innerHTML re-renders run on every poll tick: focused-input guards mandatory
  on every setup control.
- The debug worker serializes requests: while a checkpoint match runs, Debug
  screen analyses queue behind it. Surface via the `#mtSetupNote` hint, do not
  "fix".
- Two checkpoint players + a user-selected debug ckpt = 3 models = exactly the
  worker LRU size; do not add more concurrent model identities per match.
- SealBot path config: when `--sealbot-path` is unset and no adapter is found,
  sealbot seats must still be selectable-but-disabled with the discovery error
  shown (existing `adapters()` data; do not regress manual or checkpoint play).

## 9. SMOKE PROTOCOL (verification agent)

From the repo root with the `.venv` active, start a server on a spare port that
is NOT the one the dashboard normally uses (8080): `python -m hexo_frontend.web
--port 8901`, with a run present under `runs/` (e.g. `runs/hexfield_smoke`).
Then with curl:
1. `GET /api/state` → 200, `version` present.
2. `POST /api/new` ckpt-vs-ckpt: pick `<your run>` `epoch_000001.pt`
   vs `epoch_000002.pt`, `mode:"policy"`, `visits:16`, `series:{games:1}` →
   200; poll `GET /api/state?since=…&timeout_ms=15000` until `placements`
   grows ≥ 2 (allow ~3 min first-move: torch import + model load).
3. `POST /api/match/stop` → `stopped:true`; `POST /api/move` → 409.
4. `POST /api/new` manual/manual → 200, `series` null, board reset.
5. `GET /` → 200 and contains `?v=20260611-match1`.
Kill the server, report every check with its observed value.
