# Model Debug v2 — Implementation Spec (Single-Position Forensics Workbench)

Status: FINAL. This document is the binding contract between the backend and frontend
implementation agents. Section 3 (API CONTRACT) is normative; if code and spec disagree,
the spec wins until amended here.

Targets:
- `packages/hexo_frontend/python/hexo_frontend/web.py` (routes)
- `packages/hexo_frontend/python/hexo_frontend/debug_service.py` (worker host)
- `packages/hexo_frontend/python/hexo_frontend/debug_worker.py` (worker ops)
- `packages/hexo_frontend/python/hexo_frontend/debug_infer.py` (inference + new Python PUCT)
- `packages/hexo_frontend/python/hexo_frontend/static/{index.html,app.js,styles.css}`

Hard environment constraints (do not violate):
- The dashboard is served by `scripts/dashboard.sh` on :8080 with a single CPU-only
  debug worker. DO NOT start/restart any server as part of this work.
- No frameworks, no build step. Vanilla JS, hand-rolled SVG, one CSS file.
- Worker process model is preserved exactly: lazy subprocess spawn, NDJSON stdin/stdout,
  `CUDA_VISIBLE_DEVICES=""`, single `threading.Lock` (the lock IS the queue), 120s/request
  timeout (ONE exception: `search_tree` runs ~25ms/visit on CPU, so a fixed 120s
  guarantees a mid-compute worker kill — the timeout kills the worker — for any legal
  >4000-visit request; `search_tree` alone scales its deadline with visits, capped at
  300s, see §4.3), one auto-restart, 256-entry result LRU in `debug_service.py`,
  3-checkpoint LRU in the worker. Env knobs `HEXO_DEBUG_WORKER_CMD / HEXO_DEBUG_USE_WSL /
  HEXO_DEBUG_WSL_PYTHON / HEXO_DEBUG_RUN_ROOT` keep working.
- Everything pinned by `tests/test_debug_infer.py` keeps passing unchanged:
  `debug_service._to_wsl`, `debug_infer.load_checkpoint` contract,
  `analyze_position` response keys (incl. `value_swapped`, `optimism`, the
  per-horizon stvalue keys), `search_position` contract (noise-free, bit-deterministic
  across two identical calls with default seed 0, `root_prior` matches analyze priors
  within 1e-5). All response changes in this spec are ADDITIVE.
- Run tests from the repo root with the `.venv` active:
  `python -m pytest tests/test_debug_infer.py -q`.

Known bugs this rewrite MUST fix (verified against code):
- B1. `app.js` (~3905-3908) decodes moves_left with hardcoded `*80`; the real cap is
  `MOVES_LEFT_CAP = 512` (`hexfield.constants.MOVES_LEFT_CAP`). Decode as
  `(v + 1) / 2 * moves_left_cap` where `moves_left_cap` comes from the analyze meta (new
  field, see §3.6).
- B2. `web.py` `_debug_search` (1398-1415) drops the `seed` param even though
  `debug_worker.py:127` accepts it. Forward it (default 0 to preserve determinism tests).
- B3. The worker `info` op (`debug_worker.py:106-108`) has no HTTP route; checkpoint
  provenance currently costs a full analyze. New route §3.8.
- B4. The "optimism probe" UI block (owner-swap Σ with ok/warn coloring) is a misreading
  and does not apply to hexfield: hexfield encodes side-to-move ownership directly in its
  features (own/opp planes), so the owner-swap probe is N/A and `value_swapped`/`optimism`
  come back null. The fields stay in the API (tests pin them) but the UI must NOT present
  them as an optimism metric — the panel is hidden when the fields are null.

Feasibility downgrades applied (silently dropped/changed vs the design draft):
- Pin tray: localStorage (per-run key, free-text notes, JSON export/import, soft cap 24)
  instead of sessionStorage/6-slot.
- Prefetch: client-side only, at most ONE low-priority in-flight request for ply±1,
  aborted the instant the user acts, NEVER issued for checkpoint B (protects the worker's
  3-model LRU). No server-side priority tagging (impossible with the single lock).
- Per-root-child Q for the production `search` op is NOT extractable (the Rust MCTS
  exposes only root aggregates); the existing `w` field
  already IS raw visits. Child-Q data therefore comes exclusively from the new Python
  `search_tree` op's root layer. The `search` endpoint changes only by gaining `seed`.
- INPUTS tab (dense feature planes) is REMOVED in the public release: hexfield is
  support-graph featurized (no 13×41×41 dense planes), so `analyze` always returns
  `input_planes: null`. The tab is a no-op n/a placeholder; do not build a plane viewer.
- "Recorded" trajectory series is REMOVED in the public release: it depended on a
  now-removed lineage that wrote `eval/epoch_*_examples.json`, which hexfield does not
  produce. The Game Error Sweep's reeval series is the replacement for hexfield runs.
- The Python PUCT tree is explicitly labeled "debug search" everywhere in the UI: it
  reuses the exact inference wrappers but will not bit-match Rust tie-breaking.
- Checkpoint Sweep ("when did the model learn this") is pure frontend over the existing
  analyze endpoint + caches; no new route.

---

## 1. LAYOUT

### 1.0 Top-level structure and id scheme

`<main id="debugScreen">` keeps its id (screen routing `#debug`, `setScreen`, body class
`debug-screen-active`, and the `[data-debug-open]` delegation entry points are untouched).
`#debugStatus` (status line) is kept. `#debugDiag` is retired: the live diag bar
is the dynamically created global `#__diag` (reportError/diagTap write only to
it), which supersedes the static div — keeping one would leave inert markup.

Id scheme: every NEW element id starts with `dbg` + RegionAbbrev + Name, camelCase
(e.g. `#dbgCtxRun`, `#dbgPlyStrip`, `#dbgTabHeads`). New CSS classes are `.dbg-*`.
Legacy `debug*` ids are retired except: `#debugScreen`, `#debugStatus`,
`#debugBoardSvg` (the SVG node itself, so existing styles/tests/screenshots keep an
anchor), `#debugBoardHud`.

Desktop grid (CSS grid on `#debugScreen > .dbg-grid`):

```
+--------------------------------------------------------------------------+
| 1 CONTEXT STRIP  #dbgCtx                              (collapsible)      |
+------+------------------------------------+------------------------------+
| 2    | 3 BOARD STAGE  #dbgBoardStage      | 4 INSPECTOR  #dbgInspector   |
| PLY  |   mode bar / legend                |   tab bar #dbgTabs           |
| RAIL |   #debugBoardSvg + #debugBoardHud  |   HEADS|SEARCH|TARGETS|      |
| #dbg |   branch bar #dbgBranchBar         |   COMPARE|INPUTS|CKPT        |
| Ply  |                                    |                              |
| Rail |   (~55% width)                     |   (~30% width)               |
+------+------------------------------------+------------------------------+
| 6 PIN TRAY  #dbgPinTray                               (overlay strip)    |
+--------------------------------------------------------------------------+
| 5 BOTTOM DOCK  #dbgDock                               (collapsible)      |
+--------------------------------------------------------------------------+
```

Mobile (<900px): grid collapses to a single column in order Context → Board → Ply
controls (rail becomes a horizontal strip + slider) → Inspector tabs → Dock; pin tray
becomes a horizontal scroller. All existing mobile hardening is preserved: event
delegation on `#debugScreen` with `click` + `touchend` fallback and the 700ms double-fire
suppressor, `diagTap` echo, `reportError` routing — every new button goes through the
same ACTIONS map.

### 1.1 Region 1 — Context strip `#dbgCtx`

One row: selects `#dbgCtxRun`, `#dbgCtxSource` (selfplay|evaluation), `#dbgCtxFile`,
`#dbgCtxRecord` (with per-record metadata chips "plies · winner · status" rendered from
the `record_games` array the position payload already returns), `#dbgCtxCkptA`,
`#dbgCtxCkptB` (compare slot, default empty = "—"), button `#dbgCtxRefresh`, worker
status dot `#dbgCtxWorkerDot` (green alive / grey unknown / red error; title shows
`cached_results` and mode from /api/debug/checkpoints `worker`), collapse toggle
`#dbgCtxCollapse`. Collapsed state shows only breadcrumb `#dbgCtxCrumb`:
`run · file tail · g<record> · ply N/total [· branch +k] · ckptA [vs ckptB]`.
The breadcrumb is ALWAYS rendered (also when expanded, right-aligned).

### 1.2 Region 2 — Ply rail `#dbgPlyRail` (~56px wide, full height of row 2)

Top to bottom: numeric readout `#dbgPlyReadout` ("N/total"), transport buttons
`#dbgPlyFirst` `#dbgPlyPrev` `#dbgPlyNext` `#dbgPlyLast` (|< < > >|), then the
WRONGNESS STRIP `#dbgPlyStrip`: one absolutely-sized clickable `div.dbg-ply-row` per ply,
height = available/total (min 2px), background color mapped from the Game Error Sweep
(policy KL by default; right 30% of each row tinted by |value error|). Blunder plies
(sign flip or |Δvalue_p0| > 0.5 between consecutive plies) and top-1-mismatch plies get
a marker class (`.dbg-ply-blunder`, `.dbg-ply-miss`). Current ply row gets
`.dbg-ply-cur` + an arrow. Clicking a row = `dbgNavigate({ply})`.
FALLBACK: until a sweep result exists for (ckptA, game), the strip renders flat neutral
rows and a classic `<input type=range>` slider `#dbgPlySlider` is shown under the
transport buttons. A "Sweep" mini-button `#dbgPlySweepBtn` sits at the rail bottom
(same action as `#dbgSweepBtn` in the dock).

### 1.3 Region 3 — Board stage `#dbgBoardStage`

- Mode bar `#dbgModeBar`: segmented control of EXCLUSIVE base heat modes (radio
  semantics, `data-mode` attr): `prior` (1), `visits` (2), `delta` (3, Δ visits−prior
  diverging), `opp` (4), `target` (5, recorded npz visit policy), `mismatch` (6,
  prior−target diverging), `childq` (7, root-child Q from last search_tree),
  plus `none` (0). (The dense-plane heat mode was removed with the INPUTS tab —
  hexfield has no input planes.) Additive toggles (checkboxes, independent of base mode):
  `#dbgTglThreats`, `#dbgTglNumbers`, `#dbgTglLast`, `#dbgTglLegalDim`.
- Legend `#dbgLegend`: one shared color scale with explicit numeric min/max readout
  `#dbgLegendMin`/`#dbgLegendMax`, log-scale toggle `#dbgLegendLog` (key L), opacity
  slider `#dbgOpacity` (0.2–1.0, default 0.9).
- SVG `#debugBoardSvg` inside `.board-area`, hover HUD `#debugBoardHud` upgraded to one
  fixed line: `q,r · prior x.x% · visits x.x% · childQ ±0.xx (N=n) · target x.x%` —
  fed from a per-render `Map(action_id -> metrics)` (no linear `.find` in mousemove).
- Pan/zoom buttons `#dbgZoomIn` `#dbgZoomOut` `#dbgZoomReset` — port the Match board's
  zoom wiring (viewBox scaling + pinch) that the debug board currently lacks.
- Click on an empty legal cell = WHAT-IF INJECTION (appends action to prefix and
  navigates; §2 M4). Click on a tree-PV ghost stone = no-op.
- Branch bar `#dbgBranchBar` (under the board): chip list of the prefix tail —
  recorded-game chips grey (`.dbg-chip-game`), injected chips accent
  (`.dbg-chip-inj`), each injected chip has ✕ = undo-to-before-this-chip; button
  `#dbgBranchReturn` "Return to game" clears `acts`. Hidden when no injected moves.

### 1.4 Region 4 — Inspector `#dbgInspector`, tab bar `#dbgTabs`

Tabs (buttons `data-tab=`): `heads` `search` `targets` `compare` `inputs` `ckpt`.
Panels: `#dbgTabHeads` `#dbgTabSearch` `#dbgTabTargets` `#dbgTabCompare`
`#dbgTabInputs` `#dbgTabCkpt`. Exactly one visible; active tab persisted in the nav
state (`tab` hash key). Every panel header carries a stale dot `.dbg-stale` + refresh
icon: shown whenever the panel's rendered data token != current nav token (§5 F2).

- HEADS `#dbgTabHeads`: value scalar + chip `#dbgValueChip`; 65-bin dist chart
  `#dbgValueDist` with overlaid vertical markers for final z (when game decided) and
  recorded soft-z target (when TARGETS row loaded) + dist-entropy readout; owner-swap
  probe block per B4 (hidden for hexfield — `value_swapped`/`optimism` are null); STV rows `#dbgStvRows`
  (per-horizon centered bipolar bars); moves-left row `#dbgMovesLeft` decoded with
  `meta.moves_left_cap` (B1) and compared to actual remaining plies when on a recorded
  ply; top-moves table `#dbgTopMoves` — columns #, cell, prior%, visits%, Q, Δ(v−p);
  sortable by clicking headers (prior default); Q column filled from last search_tree
  root layer, "—" otherwise; badge `over` on high-prior/low-Q rows, `under` on
  low-prior/high-N rows (thresholds: prior>0.15 & Q < rootQ−0.15; prior<0.03 &
  n/visits>0.10); opp-policy top-k list `#dbgOppList` with the actual recorded reply
  (from record_games actions) highlighted.
- SEARCH `#dbgTabSearch`: controls `#dbgSearchVisits` (1..20000, default 512),
  `#dbgSearchCpuct` (default 1.5), `#dbgSearchSeed` (default 0), buttons
  `#dbgSearchRun` (production-style root search) and `#dbgTreeRun` ("Debug tree");
  summary line (visits, root value, best, Δ vs raw prior argmax); TREE EXPLORER
  `#dbgTree` — expandable nested list, each node row: move cell, N (bar), Q stm /
  Q p0, P%, U, v-at-expansion; children sorted by N desc; PV path highlighted
  root-to-leaf and mirrored on the board as numbered ghost stones; node click =
  preview line on board; per-node "Step into" button = re-base via injection (§2 M11);
  "expand" on a truncated node re-requests with `root_actions`; Q-vs-N scatter
  `#dbgScatter` (x = prior log-scale, y = Q, dot area ∝ N, best + recorded move
  annotated, quadrant labels); visit-ladder table `#dbgLadder` (client-driven: runs
  search at 64/128/256/512/1024 visits, shows best move + root value per rung,
  flags the flip point).
- TARGETS `#dbgTabTargets`: the recorded .npz training row vs current checkpoint
  outputs, two-column with per-row deltas: value target (+ `value_target_reason`),
  STV targets/masks, moves_left target/mask, policy_surprise, search_visits, pcr_full,
  frequency_weight; top-target-moves mini-table. Graceful notice `#dbgTargetsNote`
  ("no training row at this ply" / "eval games have no training rows" /
  "row mismatch — not shown") when `found:false`.
- COMPARE `#dbgTabCompare`: requires ckptB; rows Value A / Value B / Δ, per-horizon STV
  Δ, Top A / Top B / agree, PV divergence (first differing ply of A vs B search_tree
  PVs when both run); button `#dbgCmpHeat` pushes per-cell prior Δ(A−B) to the board
  (diverging) — implemented as a client-computed overlay, not a server mode; toggle
  `#dbgCmpSplit` = A/B SPLIT-BOARD mode: board stage renders two half-size boards
  (`#dbgBoardSvg` + clone `#dbgBoardSvgB`) with a SHARED hover cursor (hovering a cell
  highlights it on both and the HUD shows A and B metrics side by side).
- INPUTS `#dbgTabInputs`: REMOVED in the public release. hexfield is support-graph
  featurized and `analyze` always returns `input_planes: null`, so there are no dense
  feature planes to render. The tab is retained only as an n/a placeholder ("no input
  planes for the hexfield lineage"); do not build a plane viewer.
- CKPT `#dbgTabCkpt`: provenance from GET /api/debug/ckpt_info — readable WITHOUT
  paying an analyze: arch, lineage, rl_epoch, step, graft, STV horizons, moves_left
  presence + cap, expanded heads, zeroed feature cols, load warnings, param count,
  file size, mtime. Shown for ckptA, and ckptB below when set.

### 1.5 Region 5 — Bottom dock `#dbgDock` (full width, collapsible via `#dbgDockToggle`)

Tabs `#dbgDockTabs`: `trajectory` (default) and `ckptsweep`.
- Trajectory `#dbgDockChart`: hand-rolled SVG, shared x-axis = ply. Series: re-eval
  value_p0 (accent), policy KL per ply
  (second y-axis, from sweep), top-1-mismatch ticks, clickable blunder markers
  (jump ply), dashed current-ply cursor, hover crosshair with per-ply readout. Buttons
  `#dbgSweepBtn` ("Sweep game", inline progress `#dbgSweepProgress` "k/total plies"),
  `#dbgTrajBtn` ("Plot" — legacy trajectory fallback when no sweep).
- Checkpoint sweep `#dbgCkptSweep` (SHOULD-HAVE): for the CURRENT position (or a chosen
  pin), evaluate value + top move across all checkpoints of the run (client loop of
  analyze calls, oldest→newest, abortable `#dbgCkptSweepStop`); renders value-vs-epoch
  line + top-move-agreement ticks. Warning note: "loads each checkpoint into the
  worker's 3-model LRU; slow".

### 1.6 Region 6 — Pin tray `#dbgPinTray` (overlay strip above the dock)

Chips: mini-board thumbnail (client-rendered ~80px SVG from placements), value, best
move, checkpoint tag, note tooltip. Click = restore full nav tuple. Buttons per chip:
✕ remove, ✎ note. Tray-level: `#dbgPinAdd` (P), `#dbgPinExport` / `#dbgPinImport`
(JSON download / file input). Persistence: localStorage key `hexoDbgPins:<run>`,
array of `{ts, note, nav:{...full hash state...}, snap:{value, best, ckpt, thumb:[...placements...]}}`,
soft cap 24 (oldest unpinned warning, never silent drop).
Session journal `#dbgJournal` (SHOULD-HAVE): accordion at the bottom of the ply rail
column (collapsed by default) auto-logging every completed analyze:
`{ts, ckpt, ply/branch, value, top move}`, click-to-restore, localStorage key
`hexoDbgJournal:<run>` capped at 200 entries FIFO, "clear" button.

### 1.7 Keyboard shortcuts

Bound via one `keydown` listener on `document`, active only when `#debugScreen` is the
active screen AND `document.activeElement` is not an input/select/textarea AND the
command palette is closed. Every shortcut has a button equivalent (mobile parity).

```
← / →            step ply −1/+1          Home / End      first/last ply
Shift+← / →      prev/next blunder ply   [ / ]           prev/next record
{ / }            prev/next game file     A               analyze now
S                run root search         T               run/toggle debug tree
1..7             heat base modes         0               mode none + clear toggles
Shift+1..7       solo (mode + clear      L               log color scale
                 additive toggles)       U               undo last injected move
G                return to recorded game P               pin current position
C                focus COMPARE tab       K or Ctrl+K     command palette
?                shortcut overlay #dbgHelp
Esc              close palette/overlay
```

Command palette `#dbgPalette` (SHOULD-HAVE): centered input `#dbgPaletteInput` +
fuzzy-filtered list over: game files, records ("g3 · 142 plies · P1 win"), checkpoints,
"ply N" (numeric entry), pins, journal entries, and named actions (Analyze, Sweep,
Tree, Return to game, Export pins…). Enter = navigate/execute. Pure frontend.

---

## 2. FEATURE LIST

### MUST-HAVE

M1. Canonical nav state + URL hash + history.
    Single state object `dbg.nav = {run, src, path, rec, ply, ckptA, ckptB, acts[], tab, mode}`
    serialized to `#debug?run=..&src=..&path=..&rec=..&ply=..&ckptA=..&ckptB=..&acts=12,40,77&tab=heads&mode=prior`.
    One mutation entry point `dbgNavigate(patch, {replace=false})` writes the hash;
    a `hashchange` handler is the ONLY code that applies state → fetch/render.
    Acceptance: when I step plies then press browser Back, I return to the previous
    ply with all panels refilled; when I copy the URL into a new tab, the exact
    position (including injected moves and checkpoint B) is restored; existing
    `[data-debug-open]` buttons in History/Match still land on the right record/ply.

M2. Layout restructure per §1 (context strip, ply rail, board stage, tabbed inspector,
    dock, branch bar) with collapse states and mobile single-column fallback.
    Acceptance: when I load the screen on desktop I see all six regions and no legacy
    two-source-bar layout; when I narrow to <900px every button remains tappable and
    the double-fire suppressor still logs via diagTap; when no sweep has run the ply
    rail shows the slider fallback.

M3. Heatmap mode switcher + shared legend + log toggle + Map-backed hover HUD.
    Acceptance: when I press 2 I see only the visits heat (policy heat gone); the
    legend shows the true min/max of the active mode; when I toggle L a 0.9-prior
    board and a near-flat board become visually distinguishable; hovering any cell
    shows the full one-line HUD without jank on a 150-stone board.

M4. Click-to-play what-if injection + branch bar + undo stack.
    Implementation: full action prefix = recorded actions[0..ply] + injected tail;
    sent via existing `GET /api/debug/position?actions=` then `POST /analyze` with
    `action_ids`. Acceptance: when I click an empty legal cell the move appears as an
    accent chip, the board re-renders the new position, analysis refills, and threat
    overlays still work; when I click ✕ on a chip I return to that prefix; when I
    press G I'm back at the recorded ply; when I press U the last injected move pops.

M5. Δ(visits−prior) board mode + sortable top-moves table with Δ/Q columns + badges.
    Acceptance: when a search result exists and I press 3 I see a diverging overlay
    (promoted vs demoted); when I click the Δ header the table sorts by |Δ|; rows
    where high prior met low Q (after a tree run) show an `over` badge.

M6. moves_left decode fix (B1) + seed forwarding (B2).
    Acceptance: a position ~30 plies from the end shows a moves-left estimate on the
    right order of magnitude (not 80-capped); two searches with seed 7 are identical
    and differ from seed 0 (where ties exist).

M7. Checkpoint info route + CKPT tab (B3).
    Acceptance: when I select a checkpoint and open CKPT I see arch/lineage/epoch/
    graft/warnings within one worker round-trip and WITHOUT an analyze having run.

M8. Recorded training-target panel (.npz `record_row`) + `target` & `mismatch` board
    modes + soft-z marker on the value dist.
    Acceptance: on a self-play full-search ply, TARGETS shows the recorded visit
    policy/value target/value_target_reason/policy_surprise/pcr_full next to the live
    outputs with deltas; pressing 5 paints the recorded visit policy on the board and
    6 paints prior-vs-target mismatch; on an eval game or fast-PCR ply I see the
    graceful "no training row" notice and modes 5/6 render empty with a legend note;
    a row whose turn_index/current_player don't match the replayed position is
    REFUSED (notice, never silently misaligned).

M9. Game Error Sweep + wrongness ply rail + dock overlay + blunder markers.
    One `game_eval` op chunked client-side (16 plies/request, sequential, inline
    progress, abortable). Per ply: reeval value, KL(recorded ‖ prior), top-1 match,
    value error vs z and vs soft-z (when npz row exists). Cached in the service LRU
    and in the client cache. Acceptance: when I click "Sweep game" I see progress
    k/total and can still step plies while it runs; when it finishes the rail is
    colored, Shift+→ jumps to the next blunder, and the dock shows the KL series with
    clickable markers.

M10. MCTS Tree Explorer — NEW pure-Python deterministic PUCT (`search_tree` op).
     Labeled "debug search (Python; may not match engine tie-breaking)". Pruned
     server-side (top_k children by N per node, max_depth, min_n; ≤ ~4000 nodes);
     deeper subtrees load on demand via `root_actions`. Root layer also feeds the Q
     column/scatter/childq board mode. Acceptance: when I click "Debug tree" with 512
     visits I get an expandable tree whose root children's visit shares roughly match
     the production search; the PV is highlighted and drawn on the board as numbered
     ghosts; clicking a deep node previews its line; identical params ⇒ identical tree.

M11. Step-into-node. Acceptance: when I click "Step into" on a tree node the position
     re-bases to that line via the branch bar (each PV move = one injected chip), a
     full analyze runs there, and I can walk back out chip by chip.

M12. Client-side analysis cache. `Map` keyed `run|path|rec|ply|ckpt|acts.join(',')`
     holding {position, analysis, search, tree, record_row}, LRU cap 300 entries.
     Acceptance: revisiting a ply, undoing a chip, or flipping ckptA↔ckptB between two
     already-seen checkpoints re-renders with ZERO network requests; posSeq/anlSeq
     latest-wins tokens still drop stale worker responses.

M13. Keyboard shortcut layer per §1.7 (palette itself is SHOULD-HAVE; the rest MUST).
     Acceptance: when focus is in the visits number input, ← edits the number and does
     NOT step the ply; ? shows the overlay.

M14. Per-panel stale indicators. Acceptance: immediately after I inject a move, every
     inspector panel shows the stale dot until its data for the new prefix arrives.

### SHOULD-HAVE

S1. Command palette (Ctrl+K / K) per §1.7.
    Acceptance: typing "eval" filters to evaluation game files; Enter opens one.
S2. Pinboard with localStorage persistence, notes, JSON export/import (§1.6).
    Acceptance: pins survive a browser restart; export downloads a JSON I can
    re-import on another machine.
S3. Session journal (§1.6). Acceptance: every analyze appends an entry; clicking one
    restores the exact tuple.
S4. Checkpoint Sweep dock tab (§1.5). Acceptance: for a pinned position I see value
    vs epoch and where the top move first became the current one.
S5. A/B split-board with shared hover cursor (§1.4 COMPARE).
    Acceptance: hovering cell (q,r) highlights it on both boards and the HUD shows
    A/B priors side by side.
S6. Overlay ergonomics: Shift+digit solo, 0 clear, opacity slider.
S7. Throttled ply±1 prefetch (client-only, one in-flight, abort-on-action, never for
    ckptB, only when the worker dot is green and no user request is pending).
S8. [ / ] and { / } record/file hotkeys + record metadata chips in `#dbgCtxRecord`
    (data already in `record_games`).
S9. INPUTS tab — REMOVED in the public release (hexfield has no dense input planes;
    `analyze` returns `input_planes: null`). Tab kept only as an n/a placeholder.
S10. Visit-ladder sweep table (client loop over the existing search endpoint).
S11. Trajectory hover crosshair + second KL axis (degenerate without sweep: hidden).

---

## 3. API CONTRACT  (binding)

Conventions: all responses are JSON; errors are `{"error": str}` with HTTP 400/500.
Status mapping is normative for debug routes: `DebugRequestError` (healthy worker,
deterministic failure — do not retry) and request-validation errors → 400;
`DebugWorkerTimeout` / any other `DebugWorkerError` (worker killed/restarted — a retry
may succeed) → 500. Errors are never stored in the 256-entry service result LRU. `run` is the run-dir name, `path` is the run-relative posix path of an .hxr,
`checkpoint` is the checkpoint file name (not path). Field types: int, float, str,
bool, `T|null`, `[T]`. Worker-op signatures are listed with each new endpoint.

### 3.1 GET /api/debug/checkpoints — UNCHANGED
Params: `run`.
Response: `{run:str, checkpoints:[{name:str, epoch:int|null, size:int, mtime:float,
latest:bool, graft:"pre"|"post"|null}], lineage:str|null,
worker:{alive:bool, cached_results:int, mode:"wsl"|"native"}}`

### 3.2 GET /api/debug/games — UNCHANGED
Params: `run`, `source` ("selfplay"|"evaluation"|"all", default selfplay).
Response: `{run:str, source:str, games:[{path:str, name:str, size:int, mtime:float}]}`

### 3.3 GET /api/debug/position — UNCHANGED
Params: `run`,`path`,`record`,`ply` OR `run`,`actions`(csv),`ply`.
Response: exactly the current shape (board payload + `debug:{...}` +
`record_games:[{index:int, game_id:str, status:str, actions:int, winner:int|null}]`).
Frontend additionally consumes `record_games[i].actions/winner/status` for chips and
`record_games[rec].actions` list — NOTE: `actions` here is a COUNT in the current
payload; the recorded next-move highlight instead uses `debug.action_ids` (full id
list already present), so no change is needed.

### 3.4 GET /api/debug/trajectory — UNCHANGED (kept as no-sweep fallback)
Params: `run`,`path`,`record`,`checkpoint`.
Response: current shape `{run,path,record,total:int,stride:int,checkpoint:str,
winner:int|null, reeval:[{ply:int,value:float,current_player:int,value_p0:float}],
recorded:[{ply:int,root_value:float,root_value_p0:float}]}`.

### 3.5 POST /api/debug/search — CHANGED (additive)
Body: `{run:str, checkpoint:str, visits?:int(1..20000,def 512), c_puct?:float(def 1.5),
seed?:int(def 0)  // NEW: forwarded to worker (B2)
, n?:int} + (action_ids:[int] | path,record,ply)`.
Response: UNCHANGED shape: `{visits_requested:int, visits:int, root_value:float,
best_action_id:int, best:{q:int,r:int}, visit_policy:[{action_id:int,q:int,r:int,
p:float,w:int}], root_prior:[{action_id,q,r,p,w}], ply:int}`.
(No per-child Q here — comes from search_tree. `w` is already raw visits.)
Cache signature in debug_service must include the seed.

### 3.6 POST /api/debug/analyze — CHANGED (additive)
Body: current `{run, checkpoint, n?} + (action_ids | path,record,ply)` PLUS optional
`planes?:bool (default false)` — accepted for call-site compatibility; hexfield is
support-graph featurized and returns no planes regardless.
Response: current shape PLUS:
- `meta.moves_left_cap:int|null` — NEW. 512 (`hexfield.constants.MOVES_LEFT_CAP`),
  null when the lineage has no moves_left head. Frontend uses this for B1.
- `input_planes:null` — ALWAYS null in the public release: hexfield emits no dense
  feature planes. (The dense 13×41×41 plane payload was removed with its lineage;
  the field is kept only so the null n/a contract stays stable.)
All existing fields (value, value_swapped, optimism, value_bins, value_dist, policy,
opp_policy, stvalue, moves_left, meta.*, ply) are byte-identical to today.

### 3.7 POST /api/debug/search_tree — NEW
Body: `{run:str, checkpoint:str, visits?:int(1..20000,def 512), c_puct?:float(def 1.5),
seed?:int(def 0), n?:int, max_depth?:int(def 12), top_k?:int(def 8),
min_n?:int(def 2), root_actions?:[int]}  + (action_ids:[int] | path,record,ply)`.
`root_actions` (optional): extra moves appended AFTER the position prefix before
searching — used to re-root/expand a subtree without re-basing the UI position.
Response:
```
{ visits:int, root_value:float,            // side-to-move at the searched root
  best_action_id:int, pv:[int],            // PV as action ids from the searched root
  node_count:int, truncated:bool,          // pruning happened anywhere
  engine:"py_debug",                       // constant; UI must label accordingly
  params:{visits:int,c_puct:float,seed:int,max_depth:int,top_k:int,min_n:int},
  tree: Node }
Node = { action_id:int|null,               // null for root
         q:int|null, r:int|null,           // axial coords (null for root)
         n:int,                            // visit count
         qm:float,                         // mean value, side-to-move AT THAT NODE
         qm_p0:float,                      // same, P0-anchored
         p:float,                          // network prior at parent
         u:float|null,                     // PUCT exploration term at final tree state
         v:float|null,                     // raw net value at expansion (stm), null if unexpanded
         pruned_children:int,              // children removed by top_k/min_n/max_depth
         children:[Node] }                 // sorted n desc
```
Worker op: `search_tree {checkpoint:str, action_ids:[int], visits, c_puct, seed, n,
max_depth, top_k, min_n}` → the response above minus run resolution. Deterministic:
same request ⇒ identical JSON. Hard server cap: serialized node count ≤ 4000 (raise
top_k pruning until satisfied; set truncated=true).

### 3.8 GET /api/debug/ckpt_info — NEW
Params: `run`, `checkpoint`.
Response: `{run:str, checkpoint:str, size:int, mtime:float,
meta:{lineage:str, arch:str|null, rl_epoch:int|null, step:int|null,
graft:"pre"|"post"|null, stv_horizons:[int], has_moves_left:bool,
moves_left_cap:int|null, expanded_value:bool, expanded_stv:[str],
zeroed_feature_cols:[int], load_warnings:[str], param_count:int|null,
candidate_radius:int|null}}`.
Backed by the EXISTING worker `info` op (debug_worker.py:106-108 / `_model_meta`),
extended additively with `param_count` (sum of p.numel()) and `moves_left_cap`;
size/mtime stat'd in web.py. Cached in the service LRU like any op.

### 3.9 GET /api/debug/record_row — NEW
Params: `run`, `path` (the .hxr, e.g. `selfplay/epoch_000123.hxr`), `record` (int),
`ply` (int).
Resolution (web.py): only selfplay paths qualify (`found:false,
reason:"not_selfplay"` otherwise). Shard candidates =
`selfplay/epoch_NNNNNN_game_*.npz` for the .hxr's epoch; the right shard is matched by
`game_id` (read sidecar `.json` `game_id` and compare to the .hxr record's game_id;
fallback: the self-play game index parsed off the record's game_id — shards are named
`..._game_{index}.npz` by that index). There is deliberately NO record-index fallback:
.hxr records are written in game FINISH order while shards are named by the self-play
game index, so `record_index` would silently attach a different game's shard whenever
the orders diverge (they routinely do). Row within the shard: the row with
`turn_index == ply`; the worker MUST verify `row.current_player` equals the
replay-derived current player and return `found:false, reason:"row_mismatch"` on any
disagreement (M8 misalignment guard).
Response:
```
{ found:bool, reason:str|null,            // "no_shard"|"no_row"|"row_mismatch"|"not_selfplay"|"bad_shard"|null
  npz:str|null, turn_index:int|null,
  row: null | {
    current_player:int, phase:str,
    value_target:float, value_target_reason:str|null,
    policy:[{action_id:int,q:int,r:int,p:float}],        // recorded MCTS visit policy, p desc
    opp_policy:[{action_id,q,r,p}]|null,
    opp_policy_source:str|null,
    stvalue:{ "<h>":{target:float, mask:bool} },          // per horizon
    moves_left:{target:float, mask:bool}|null,            // raw decisions remaining; -1 = masked
    policy_surprise:float|null, search_visits:int, pcr_full:bool|null,
    frequency_weight:float, truncated:bool } }
```
`"bad_shard"` = a foreign/partially-written .npz missing expected arrays
(`read_record_row` never raises; the frontend treats it like any other notice reason).
Null semantics: the compact shard format intentionally drops some finalize-time facts,
and the worker MUST NOT fabricate neutral stand-ins for them — `T|null` fields are null
when the shard predates/omits the field, and the frontend renders null as
"not recorded". Concretely: `policy_surprise` and `pcr_full` are NEVER persisted (always
null); `value_target_reason` is null except the server-overlaid `"max_actions_draw"` on
truncated games; `opp_policy_source` is never persisted (null); `moves_left` is the
whole-sub-object-null on shards that predate the moves-left head. `search_visits` stays an int (test-pinned):
the raw visit mass when the shard stores count weights, 0 = unknown when it stores the
normalized visit policy. `frequency_weight` is genuinely recovered from
surprise-materialized row duplication (count of duplicate turn_index rows), not
fabricated. `truncated` is always overlaid by the server from the .hxr record status.
Worker op: `record_row {npz:str(abs path), turn_index:int,
expect_player:int}` → `{found, reason, row}` (numpy + compact_io expand live in the
worker's venv; web.py does path resolution + game_id matching using the sidecar .json,
which is plain json readable directly — sidecar read happens in web.py,
npz decode in the worker).

### 3.10 GET /api/debug/game_eval — NEW (chunked sweep)
Params: `run`, `path`, `record`, `checkpoint`, `start` (int, first ply inclusive),
`count` (int, default 16, max 32).
Semantics: for each ply p in [start, min(start+count, total)): replay prefix
actions[0..p], one forward, value; if a record_row exists for p (same resolution as
3.9, resolved once per request) also KL(recorded_policy ‖ current prior over the
intersection support, recorded as reference), prior-argmax vs recorded chosen move
(actions[p]), value errors.
Response:
```
{ run,path,record:int,checkpoint:str, total:int, start:int, count:int,
  winner:int|null,
  plies:[{ ply:int, current_player:int,
           value:float, value_p0:float,
           kl:float|null,                  // null when no npz row
           top1_match:bool|null,           // prior argmax == recorded move actions[ply]
           value_err_z:float|null,         // value_p0 − z_p0 (winner known)
           value_err_soft:float|null }] }  // value − recorded value_target (stm frame)
```
Worker op: `game_eval {checkpoint, sequences:[{ply:int, action_ids:[int],
recorded:{policy:[[action_id,weight]], value_target:float}|null,
chosen_action:int|null}], n?}` → `{plies:[...as above minus run fields...]}`.
web.py builds the sequences (it already replays .hxr) and attaches recorded rows from
the npz — to avoid double npz decode web.py first calls the `record_row` op per ply?
NO: web.py sends the npz path + ply list and the worker decodes the shard ONCE per
chunk and joins internally: final worker op signature is
`game_eval {checkpoint, action_ids:[int] (full game), plies:[int], npz:str|null,
winner:int|null, n?}`. This is the normative signature.
Caching: service LRU key includes (checkpoint, game digest, start, count) — chunks
are individually cached so re-sweeps after a restart resume cheaply.
Client drives chunks sequentially (never parallel) and may abort between chunks.

### 3.11 Worker protocol summary (after change)
Ops: `ping`, `info` (extended additively), `analyze` (+`planes`), `search` (unchanged;
seed now actually arrives), `reeval` (unchanged, kept for /trajectory), and NEW
`search_tree`, `record_row`, `game_eval`. NDJSON framing, int ids, error surface, and
the 3-checkpoint LRU are untouched.

---

## 4. BACKEND IMPLEMENTATION PLAN

### 4.1 debug_infer.py
- Add `MOVES_LEFT_CAP` plumb-through: the hexfield meta builder gains
  `moves_left_cap` (import from `hexfield.constants`; null when the model has no
  moves_left head) and `param_count`. Do NOT change any existing meta key.
- Add `DebugSearchNode` + `search_tree_position(loaded, action_ids, *, visits, c_puct,
  seed, max_depth, top_k, min_n, n=None) -> dict` — a NEW pure-Python PUCT:
  * Reuses the exact per-position featurize+forward wrappers analyze/search already
    build (one net call per expansion; batch of 1 is fine on CPU at
    ≤20k visits — document that 4096 visits ≈ the practical interactive ceiling).
  * Deterministic by construction: seeded `random.Random(seed)` used ONLY for
    tie-breaking among exactly-equal PUCT scores; neutral FPU/temperature and no
    stochastic root perturbation, so the same request yields an identical tree.
  * Tracks per-node N, W (stm-anchored), P, v-at-expansion; serializer walks the tree
    applying min_n/top_k/max_depth pruning + the 4000-node hard cap, computes final
    U values, PV (max-N path), qm/qm_p0.
  * Terminal handling reuses the same engine-replay legality the import path uses.
- Add `read_record_row(npz_path, turn_index, expect_player) -> dict` using
  compact_io expand helpers; returns the §3.9 row shape; never raises on mismatch —
  returns found:false reasons.
- Add `game_eval_positions(loaded, action_ids, plies, npz_path, winner, n=None)`:
  replays incrementally (reuse one engine walk, do NOT re-replay from scratch per
  ply where the existing analyze path allows it; otherwise accept the rework — this
  is an optimization, not a contract), one forward per ply, joins npz rows by
  turn_index, computes kl/top1/value errors per §3.10. KL definition:
  `sum(t_i * (log t_i − log p_i))` over recorded-support cells, with p floored at 1e-9
  after renormalizing the prior over the recorded support.
- DO NOT touch `load_checkpoint`, `analyze_position` (which accepts the additive
  `planes:bool=False` kwarg but returns `input_planes: null` for hexfield), or
  `search_position` signatures/behavior — all pinned by tests/test_debug_infer.py
  (priors alignment 1e-5, bit-determinism, stvalue keys, optimism field).

### 4.2 debug_worker.py
- Extend the op dispatcher with `search_tree`, `record_row`, `game_eval` mapping 1:1
  to the new debug_infer functions; `analyze` forwards `planes`; `info` result gains
  `param_count`/`moves_left_cap`. Keep the 3-checkpoint LRU; `record_row` needs no
  checkpoint (skip model load). Update the module docstring op list.

### 4.3 debug_service.py
- No structural change. Ensure new ops' cache signatures include ALL params
  (e.g. search seed — fixing the latent staleness where seed wasn't part of the
  request before). Keep CACHE_MAX=256, timeouts as-is (chunk sizes in §3 are sized to
  fit 120s on CPU; game_eval max count=32 ⇒ ≤32 forwards ≈ well under budget) with ONE
  exception: search_tree at ~25ms/visit exceeds 120s for any legal >4000-visit request,
  and a timeout is a mid-compute worker KILL, so web.py scales the search_tree deadline
  as `min(300, max(DEFAULT_TIMEOUT, visits * 0.03))` — capped at 300s to bound how long
  one request can hog the single worker lock. The UI default stays 512 visits.
- `_to_wsl` untouched (test-pinned).

### 4.4 web.py
- Forward `seed` in `_debug_search` (B2) with default 0.
- `_debug_analyze`: accept `planes` bool, pass through.
- New handlers + dispatch entries (mirror the existing style at lines 676-737):
  `GET /api/debug/ckpt_info` (resolve checkpoint path like analyze does, stat size/
  mtime, call worker `info`),
  `GET /api/debug/record_row` (selfplay-path guard, epoch parse, shard candidates
  glob, sidecar-json game_id match with game_id-parsed game-index fallback (no
  record-index fallback — see §3.9), then worker `record_row`),
  `GET /api/debug/game_eval` (open .hxr record once, clamp start/count, resolve npz
  path via the same helper as record_row, call worker `game_eval`, splice run-level
  fields),
  `POST /api/debug/search_tree` (body validation incl. clamps: visits 1..20000,
  max_depth 1..40, top_k 1..32, min_n 0..1e6; position resolution shared with
  analyze/search via the existing action-ids/path-record-ply helper; `root_actions`
  appended to the resolved prefix before dispatch).
- Factor the shared "(action_ids | path/record/ply) -> action prefix" resolution into
  one helper if not already shared, used by analyze/search/search_tree/game_eval.
  Keep route registration inline in the existing dispatch if-chain (no framework).
- Preserve untouched: all non-debug routes, `/static/` serving, cache policy
  (index.html no-store; app.js/styles.css no-cache+ETag; `?v=` ignored server-side),
  `_debug_training_roots` incl. HEXO_DEBUG_RUN_ROOT.

### 4.5 Backend tests (additive, same file style)
Extend tests/test_debug_infer.py (new test functions only, never edit existing ones):
search_tree determinism (two identical calls ⇒ identical serialized tree), root-layer
visit shares vs search_position within tolerance, node-count cap, record_row
found:false reasons on a bogus path, game_eval ply alignment + kl≥0. All gated by the
same importorskip/checkpoint-presence skips.

---

## 5. FRONTEND IMPLEMENTATION PLAN

All three files. Cache-bust: bump the hand-maintained `?v=` token in index.html to
`?v=20260611-dbg2` — it is duplicated in THREE places (app.js tag, styles.css tag,
plus the third occurrence — grep `?v=` in index.html and update all of them in the
same commit). `node --check app.js` must pass after every stage.

### Stage F1 — Layout skeleton (index.html + styles.css)
- Replace the `<main id="debugScreen">` interior (index.html ~258-379) with the §1
  region structure. Every id from the SHARED ID LIST below must exist after F1, even
  if its panel body is an empty placeholder. Tabs/dock/context-collapse work with
  pure class toggling (no JS data dependencies yet). Keep `#debugStatus`,
  `#debugBoardSvg`, `#debugBoardHud` (`#debugDiag` retired per §1.0).
- styles.css: retire the old `.dbgv` block (2435-2610) in place; add `.dbg-grid`
  (grid-template: auto / 56px minmax(0,1fr) minmax(280px,30%) with named areas),
  `.dbg-ply-row` strip styling, `.dbg-chip-*`, `.dbg-tab*`, `.dbg-stale`, dock/pin
  tray, `@media (max-width: 900px)` single-column collapse, palette/help overlays.
- Acceptance: screen renders the full skeleton with dummy text; Match/History screens
  visually unchanged.

### Stage F2 — Core wiring (app.js)
- Replace the `dbg` state object: add `dbg.nav` (M1), `dbgNavigate(patch,{replace})`,
  `hashchange` handler, serializers `dbgNavToHash/dbgHashToNav`. Rewire
  `debugOpenFromHistory`/`pendingDeepLink` to emit a `dbgNavigate` call (keep the
  `[data-debug-open]` delegation untouched). `screenFromHash` keeps treating any
  `#debug...` prefix as the debug screen.
- Sources: context-strip selects populate from /checkpoints + /games (existing
  loaders, retargeted to new ids); record chips from `record_games`.
- Board: port the renderer to the mode-switcher model — one `dbgHeatForMode(mode)`
  returning `{values:Map(action_id->float), scale:'seq'|'div', min,max}`; build the
  hover-HUD metrics Map once per render; event delegation for cell hover/click
  (no per-cell listeners); zoom buttons; click-to-inject (M4) + branch bar render;
  legend + log + opacity.
- Navigation: ply rail transport + slider fallback + client-side instant stepping off
  the existing `gamePlacements` cache; debounced (120ms) position+analyze refetch;
  posSeq/anlSeq latest-wins kept; client cache (M12); stale dots (M14).
- Panels: HEADS tab renderers (value dist with markers, STV, moves-left via
  `meta.moves_left_cap`, sortable top-moves, opp list), CKPT tab (ckpt_info fetch),
  SEARCH tab controls + root-search summary (seed input now honored), legacy
  trajectory plot in the dock. Keyboard layer (M13) minus palette.
- Acceptance: full parity with the old screen (everything it could do, relocated) plus
  M1-M7, M12-M14.

### Stage F3 — Advanced features (app.js + small CSS additions)
- TARGETS tab + record_row fetch + `target`/`mismatch` board modes + soft-z dist
  marker (M8).
- Game Error Sweep driver (sequential chunk loop, abort, progress), rail coloring,
  blunder index for Shift+arrows, dock KL series + markers (M9, S11).
- search_tree: request/serialize, tree explorer render (nested <details>-style rows,
  expand-on-demand via root_actions), PV ghost stones, node preview, step-into (M10,
  M11), Q-vs-N scatter, `childq` mode, Δ-table Q column + badges (M5 completion),
  visit ladder (S10).
- COMPARE tab deep A/B + client prior-Δ overlay + split board (S5).
- Pinboard (S2), journal (S3), palette (S1), checkpoint sweep dock tab (S4),
  prefetch (S7), record/file hotkeys (S8), overlay ergonomics (S6).
  (INPUTS tab / S9 is removed — hexfield has no input planes; keep only the n/a
  placeholder.)

### SHARED ID LIST (created in F1, wired in F2/F3 — binding between stages)
```
#debugScreen #debugStatus #debugBoardSvg #debugBoardHud        (legacy kept; #debugDiag retired, see §1.0)
F2 wires: #dbgCtx #dbgCtxRun #dbgCtxSource #dbgCtxFile #dbgCtxRecord #dbgCtxCkptA
  #dbgCtxCkptB #dbgCtxRefresh #dbgCtxWorkerDot #dbgCtxCrumb #dbgCtxCollapse
  #dbgPlyRail #dbgPlyReadout #dbgPlyFirst #dbgPlyPrev #dbgPlyNext #dbgPlyLast
  #dbgPlyStrip #dbgPlySlider #dbgPlySweepBtn
  #dbgBoardStage #dbgModeBar #dbgPlaneSelect #dbgTglThreats #dbgTglNumbers
  #dbgTglLast #dbgTglLegalDim #dbgLegend #dbgLegendMin #dbgLegendMax #dbgLegendLog
  #dbgOpacity #dbgZoomIn #dbgZoomOut #dbgZoomReset #dbgBranchBar #dbgBranchReturn
  #dbgTabs #dbgTabHeads #dbgTabSearch #dbgTabTargets #dbgTabCompare #dbgTabInputs
  #dbgTabCkpt #dbgValueChip #dbgValueDist #dbgStvRows #dbgMovesLeft #dbgTopMoves
  #dbgOppList #dbgSearchVisits #dbgSearchCpuct #dbgSearchSeed #dbgSearchRun
  #dbgCkptInfo #dbgDock #dbgDockToggle #dbgDockTabs #dbgDockChart #dbgTrajBtn
  #dbgHelp
F3 wires: #dbgTreeRun #dbgTree #dbgScatter #dbgLadder #dbgTargetsNote #dbgCmpHeat
  #dbgCmpSplit #dbgBoardSvgB #dbgPlanes #dbgSweepBtn #dbgSweepProgress #dbgCkptSweep
  #dbgCkptSweepStop #dbgPinTray #dbgPinAdd #dbgPinExport #dbgPinImport #dbgJournal
  #dbgPalette #dbgPaletteInput
```

---

## 6. OUT OF SCOPE / DO-NOT-TOUCH

- Match and History screens, their HTML/JS/CSS, and all non-`/api/debug/*` routes:
  must keep working byte-for-byte. Shared helpers (hex geometry `center/path`,
  `HEX/SQRT3`, `playerColor`, `loadTrainingRuns`, `setScreen`/`navigateScreen`,
  `diagTap`/`reportError`) may be CALLED but not modified.
- No frameworks, no bundler, no build step, no new npm/pip dependencies. Stdlib +
  the repo's `.venv` (torch/numpy/compact_io) only.
- Do not start, restart, or probe the running dashboard (:8080) or its debug worker.
  Verification = `node --check` and `python -m pytest tests/ -q` from the repo root
  with the `.venv` active.
- Do not modify the Rust engine/rust_bridge, the production MCTS, selfplay/training
  code, or anything under the `hexfield` package outside the debug-inference surface
  this spec touches.
- Do not change the worker spawn command, NDJSON protocol framing, env knobs, cache
  policy headers, `/static/` and `/` serving, or `_to_wsl`.
- Do not edit existing test functions; new tests are additive.
- The owner-swap probe stays in the API (test-pinned) but no UI may present it as an
  optimism metric (B4).
