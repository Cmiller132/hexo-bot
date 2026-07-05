// ===========================================================================
// hexo_frontend dashboard SPA — single file, all three screens.
//
// Served by web.py (the stdlib HTTP server, packages/hexo_frontend/python/
// hexo_frontend/web.py); loaded only by static/index.html with a ?v= cache-
// bust token that MUST be bumped together with APP_VERSION below and the
// styles.css ?v= reference. No build step, no modules: declaration order
// matters, and the screens share this one global namespace.
//
// Architecture map (section banners mark the same regions, in file order):
//   1. On-screen diag bar + error trap        (__diagBar/reportError/diagTap)
//   2. Shared module state + constants + DOM  (const HEX ... PLAYER_KIND_LABELS)
//   3. Global event bindings                  (on(), mtStartMatch, on(...) calls)
//   4. Data fetchers                          (loadState/loadAdapters/loadTrainingRuns)
//   5. History data layer                     (resetHistoryPage ... loadTrainingHistory;
//      paged /api/training/history-page, counts, 2.5s /api/training/live poll)
//   6. HTTP helpers + screen routing + match long-poll (post/safeJson/setScreen/pollState)
//   7. Match screen v2 (mt*) rendering        (render/renderMtSetup ... setup strip)
//   8. Board SVG + camera                     (buildBoardModel/renderBoard, pan/zoom/pinch;
//      shared by the Match board and History replay; the Debug board has its own copies)
//   9. Match panels                           (renderMtStatus ... mtHandleKey)
//  10. Shared state/format helpers            (canSubmitMove ... placementStepLabel)
//  11. History screen v2 (hist*) rendering    (clearHistPanels ... formatBytes:
//      status band, trends charts, epoch table, epoch inspector, game list)
//  12. Debug workbench v2 (dbg*)              (big banner below; self-contained,
//      hash-based nav state, /api/debug/* via the CPU worker)
//  13. Screen entry + init                    (enterDebugScreen, init)
//
// Cross-language contracts: dbgPackActionId mirrors hexo_engine.types
// .pack_coord_id (DBG_COORD_OFFSET 32768); all payload field names come from
// web.py's route handlers and the docs/specs/*_v2_spec.md documents.
// ===========================================================================

// ---------------------------------------------------------------------------
// On-screen diagnostics, anchored at the TOP. The Debug tab failed only on the
// owner's real phone (never in headless/desktop), and an earlier BOTTOM-anchored
// banner/version tag was hidden behind the Samsung system nav bar — so errors and
// the version were invisible. This single top bar always shows the running
// version, a live "last tap" echo (so a tap that registers is visible even if its
// effect isn't), and any JS error (uncaught OR surfaced from the Debug code).
const APP_VERSION = "20260617-dbgattn";
function __diagBar() {
  let el = document.getElementById("__diag");
  if (!el) {
    el = document.createElement("div");
    el.id = "__diag";
    el.style.cssText =
      "position:fixed;left:0;right:0;top:0;z-index:2147483647;" +
      "font:11px/1.4 ui-monospace,Menlo,Consolas,monospace;padding:4px 8px;" +
      "white-space:pre-wrap;word-break:break-word;max-height:42vh;overflow:auto;";
    el.addEventListener("click", () => { el.dataset.err = ""; __renderDiag(); });
    (document.body || document.documentElement).appendChild(el);
  }
  return el;
}
function __renderDiag() {
  const el = __diagBar();
  const err = el.dataset.err || "";
  const tap = el.dataset.tap || "";
  el.style.background = err ? "#5a1020" : "rgba(8,20,30,0.92)";
  el.style.color = err ? "#fff" : "#7fe0ff";
  el.style.borderBottom = err ? "2px solid #ff5650" : "1px solid #1d3b50";
  el.textContent = "v" + APP_VERSION + (tap ? "  ·  " + tap : "") + (err ? "\nERROR (tap to clear): " + err : "");
}
function reportError(msg) { try { __diagBar().dataset.err = String(msg); __renderDiag(); } catch (_e) {} }
function diagTap(msg) { try { __diagBar().dataset.tap = String(msg); __renderDiag(); } catch (_e) {} }
function showVersionTag() { try { __renderDiag(); } catch (_e) {} }
window.__appVersion = APP_VERSION;
window.addEventListener("error", e => {
  const m = (e && e.error && (e.error.stack || e.error.message)) ||
    `${e && e.message} @ ${e && e.filename}:${e && e.lineno}:${e && e.colno}`;
  reportError(String(m));
});
window.addEventListener("unhandledrejection", e => {
  const r = e && e.reason;
  reportError("promise rejection: " + ((r && (r.stack || r.message)) || String(r)));
});
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", showVersionTag);
} else {
  showVersionTag();
}

// ===========================================================================
// Shared module state + constants + DOM lookups (all three screens).
// ===========================================================================

const HEX = 19;
const SQRT3 = Math.sqrt(3);
const FIT_MOVE_COUNT = 8;
const FIT_LEGAL_RADIUS = 5;
const HISTORY_ALL_RUNS = "__all__";
const HISTORY_PAGE_SIZE = 50;
const HISTORY_AUTOLOAD_PAGE_SIZE = 500;
const HISTORY_REFRESH_INTERVAL_MS = 15000;
const ARTIFACT_PAGE_SIZE = 50;

let state = null;
let pendingRequest = false;
let requestSeq = 0;
let replayIndex = null;
let replayTimer = null;
let boardBaseView = null;
let boardView = null;
let boardViewDirty = false;
let boardDrag = null;
let suppressBoardClick = false;
const activePointers = new Map(); // pointerId -> {x, y} for pan + pinch gestures
let pinchState = null;
let adapters = null;
let adapterLoadError = null;
let trainingRuns = [];
let trainingRun = null;
let trainingRunDetails = {};
let trainingLoadError = "";
const SCREENS = ["match", "history", "debug"];
let activeScreen = screenFromHash();
let historyFilters = { query: "", source: "all", winner: "all" };
let historySort = "newest";
let historySelectedRun = "";
let historySelectionTouched = false;
let historyVisibleLimit = HISTORY_PAGE_SIZE;
let historyDetailsLoading = false;
let historyRefreshInFlight = false;
let historySearchTimer = null;
let historyPage = {
  items: [],
  nextCursor: null,
  complete: true,
  totalMatches: null,
  loading: false,
  loaded: false,
  requestKey: "",
  countLoading: false,
  countRequestKey: "",
};
let selectedHistoryKey = "";
// History screen v2 presentation state. Lives in module vars (NOT the DOM)
// because the 15s innerHTML refresh wipes all DOM state on every render.
let histEpochFilter = null;          // int|null — client-side, display-only epoch filter
let histHealthDrawerOpen = false;    // learning-health messages drawer toggle
let histExpandedEpochs = new Set();  // "<run>::<epoch>" keys of expanded epoch-table rows (H6)
let histTrendCharts = {};            // chartId -> {epochs, x0, dx, series} for hover lookup
let histThumbCache = new Map();      // historyItemKey -> {status, placements} (H9), FIFO-capped
let histTrendHoverSvg = null;        // svg element owning the visible crosshair
let histTrendTipEl = null;           // the shared #histTrendTip div (appended to body once)
// Epoch Inspector (P1). Identity is the (run, epoch) PAIR — All-runs mode mixes
// runs in the epoch table, so a bare epoch number would be ambiguous.
let histInspectEpoch = null;         // {run, epoch} | null — the open inspector
let histEpochInfoCache = new Map();  // "<run>::<epoch>" -> {status, payload} from /api/training/epoch, FIFO-capped
let histCkptInfoCache = new Map();   // "<run>::<ckptName>" -> {status, payload} from /api/debug/ckpt_info, FIFO-capped
// Per-epoch telemetry strip (schema-upgrade 2026-07-03): one /api/training/epochs
// fetch per run, cached by run name. {status, epochs:[record]} — records merge the
// selfplay/select/training/eval subsets, every field degrading to null. Feeds the
// inspector's Telemetry group + the strip sparklines; missing/legacy fields render
// as "—" and never crash. FIFO-capped like the other lazy caches.
let histEpochsCache = new Map();     // "<run>" -> {status, epochs} from /api/training/epochs
const HIST_EPOCHS_CACHE_MAX = 6;
// Selected-game navigation (P2). The displayed-list key order, rebuilt every
// render — nav buttons and the keydown handler always re-read it.
let histDisplayedKeys = [];          // historyItemKey order of the filtered+sorted+epoch-chip list
// Near-realtime tier (R3-R6): a fast /api/training/live poll that patches ONLY
// the status band between the 15s full refreshes. Visibility-gated (history
// screen active + tab visible) because a hidden tab polling every 2.5s is
// pure waste on both ends.
const HIST_LIVE_POLL_MS = 2500;
const HIST_LIVE_POLL_IDLE_MS = 1000; // cheap no-op cadence while gated off
const HIST_LIVE_POLL_MAX_MS = 30000; // error-backoff ceiling
let histLivePollTimer = null;        // setTimeout id of the next tick (self-rescheduling chain)
let histLivePollDelayMs = HIST_LIVE_POLL_MS; // doubles per fetch error, resets on success
let histLivePollInFlight = false;    // single in-flight guard (a kick must never overlap a fetch)
let histLiveLastRun = "";            // run the last applied live status belongs to
let histLiveLastSig = "";            // JSON signature of the last applied status — unchanged => zero DOM churn
let histLiveLastStatus = null;       // last poll's raw status, for epoch-boundary detection (R5)
let histLiveLastTs = 0;              // Date.now() of the last successful poll (drives #histLiveTick)
let histLiveRefreshAt = 0;           // 2s debounce stamp for the boundary-triggered full refresh
let historyView = false;
let polling = false;
let pollTimer = null;
let pollAbort = null;
let pollFailures = 0;
let lastStatusError = "";
// Match setup state (mt* = match screen v2). Each seat holds a normalized spec
// dict that maps 1:1 onto the /api/new body (§3.1): {kind:"manual"} |
// {kind:"sealbot", variant} | {kind:"checkpoint", run, checkpoint, visits, mode}.
let mtSetup = {
  seats: {
    player0: { kind: "manual" },
    player1: { kind: "sealbot", variant: "current" },
  },
  time_limit: 0.05,
  seed: null,
  series: { games: 1, alternate: false },
};
const mtCkptLists = new Map();        // run name -> checkpoints array (/api/debug/checkpoints, newest-first)
const mtCkptListsLoading = new Set(); // run names with a list fetch in flight
let mtRunsRequested = false;          // a checkpoint seat lazily fired loadTrainingRuns() once

// Kept: historyPlayerLabel (History screen) and playerMeta/playerKindLabel
// still read this map.
const PLAYER_KIND_LABELS = {
  manual: "Manual",
  "sealbot-current": "SealBot current",
  "sealbot-best": "SealBot best",
  checkpoint: "Checkpoint",
  "dense-cnn": "Dense CNN",
  unknown: "Unknown",
};

const svg = document.getElementById("boardSvg");
const boardArea = document.getElementById("boardArea");
const tip = document.getElementById("tip");
const cellHud = document.getElementById("cellHud");
const matchScreen = document.getElementById("matchScreen");
const historyScreen = document.getElementById("historyScreen");
const debugScreen = document.getElementById("debugScreen");
const historyRunSelect = document.getElementById("historyRunSelect");
const historyRefreshBtn = document.getElementById("historyRefreshBtn");
const histStatusBand = document.getElementById("histStatusBand");
const historySearchInput = document.getElementById("historySearchInput");
const historySourceSelect = document.getElementById("historySourceSelect");
const historyWinnerSelect = document.getElementById("historyWinnerSelect");
const historySortSelect = document.getElementById("historySortSelect");
const gameHistoryList = document.getElementById("gameHistoryList");
const gameHistoryDetail = document.getElementById("gameHistoryDetail");
const histHealthDrawer = document.getElementById("histHealthDrawer");
const histTrends = document.getElementById("histTrends");
const histStrength = document.getElementById("histStrength");
const histRatingTable = document.getElementById("histRatingTable");
const histEvalPool = document.getElementById("histEvalPool");
const histEpochTable = document.getElementById("histEpochTable");
const histEpochInspector = document.getElementById("histEpochInspector");
const histEpochChip = document.getElementById("histEpochChip");

// ===========================================================================
// Global event bindings (match setup strip, board, replay controls).
// ===========================================================================

// Null-guarded binding helper: one missing/renamed id should warn and skip,
// not throw and brick the whole script before init() ever runs.
function on(id, evt, fn, opts) {
  const el = document.getElementById(id);
  if (!el) {
    console.warn("missing element #" + id);
    return null;
  }
  el.addEventListener(evt, fn, opts);
  return el;
}

// Start/rematch action — shared by #mtStartBtn and the `N` shortcut (F3.5).
// The blocker guard matches the button's disabled condition so the keyboard
// path can never start a match the button would refuse.
function mtStartMatch() {
  if (pendingRequest || mtStartBlockers().length) return;
  historyView = false;
  clearBoardView();
  resetReplay();
  post("/api/new", buildMtMatchPayload(), { resetReplay: true, clearBoard: true });
}
on("mtStartBtn", "click", mtStartMatch);
on("mtStopBtn", "click", () => post("/api/match/stop", {}));
on("mtOpenDebugBtn", "click", mtOpenDebug);
// F3.2: one mousemove/mouseleave pair for the value sparkline; click scrubs.
on("mtValueChart", "mousemove", mtChartHover);
on("mtValueChart", "mouseleave", mtChartHoverEnd);
on("mtValueChart", "click", mtChartClick);
// F3.5: match-screen keyboard layer (guard shape mirrors histHandleKey).
document.addEventListener("keydown", mtHandleKey);
// F2.4: one delegated listener keeps mtSetup in sync with every setup control
// (seat kind/run/ckpt/visits/mode + time/seed/series). No network call until
// Start — renderMtSetup only does lazy LIST fetches (runs, checkpoints).
on("mtSetup", "change", () => {
  mtReadSetupFromDom();
  renderMtSetup();
});
if (historyRefreshBtn) historyRefreshBtn.addEventListener("click", () => loadTrainingRuns({ preserveHistoryPage: true }));
if (historyRunSelect) historyRunSelect.addEventListener("change", async () => {
  historySelectedRun = historyRunSelect.value || HISTORY_ALL_RUNS;
  historySelectionTouched = true;
  historyVisibleLimit = HISTORY_PAGE_SIZE;
  resetHistoryPage();
  await ensureHistorySelectionLoaded();
  await loadHistoryPage({ reset: true });
  renderGameHistoryPage();
});
document.querySelectorAll("[data-screen]").forEach(button => {
  button.addEventListener("click", () => {
    navigateScreen(button.dataset.screen || "match");
  });
});
if (gameHistoryList) gameHistoryList.addEventListener("click", handleGameHistoryClick);
if (gameHistoryDetail) gameHistoryDetail.addEventListener("click", handleGameHistoryClick);
// History v2 status band: one delegated listener toggles the learning-health
// messages drawer (state in histHealthDrawerOpen so the 15s re-render keeps it).
if (histStatusBand) histStatusBand.addEventListener("click", event => {
  const pill = event.target.closest("#histHealthPill");
  if (!pill) return;
  event.preventDefault();
  histHealthDrawerOpen = !histHealthDrawerOpen;
  renderGameHistoryPage();
});
// History v2 trends: one delegated hover/leave/click trio for every chart.
if (histTrends) {
  histTrends.addEventListener("mousemove", handleHistTrendsMove);
  histTrends.addEventListener("mouseleave", hideHistTrendHover);
  histTrends.addEventListener("click", handleHistTrendsClick);
}
// History v2 epoch table: one delegated listener for the chevron expanders
// (histExpandedEpochs survives the 15s re-render) and the epoch-number
// buttons that toggle the client-side epoch filter.
if (histEpochTable) histEpochTable.addEventListener("click", event => {
  const toggle = event.target.closest("[data-hist-epoch-toggle]");
  if (toggle) {
    event.preventDefault();
    const key = toggle.dataset.histEpochToggle || "";
    if (histExpandedEpochs.has(key)) histExpandedEpochs.delete(key);
    else histExpandedEpochs.add(key);
    renderGameHistoryPage();
    return;
  }
  const epochButton = event.target.closest("[data-hist-epoch]");
  if (epochButton) {
    event.preventDefault();
    const epoch = asFinite(epochButton.dataset.histEpoch);
    if (epoch !== null) setHistEpochFilter(epoch);
    return;
  }
  // P1.3: clicking the row body (not its chevron/epoch buttons, which return
  // above) toggles the epoch inspector for that (run, epoch) pair.
  const inspectRow = event.target.closest("[data-hist-epoch-inspect]");
  if (inspectRow) {
    event.preventDefault();
    const key = String(inspectRow.dataset.histEpochInspect || "");
    const sep = key.lastIndexOf("::");
    if (sep < 0) return;
    const run = key.slice(0, sep);
    const epoch = asFinite(key.slice(sep + 2));
    if (epoch === null) return;
    histInspectEpoch = histInspectEpoch && histInspectEpoch.run === run && histInspectEpoch.epoch === epoch
      ? null
      : { run, epoch };
    renderGameHistoryPage();
  }
});
// P1.3: epoch inspector — one delegated listener for close / prev-next / the
// "filter games" button (which reuses the shared setHistEpochFilter toggle).
if (histEpochInspector) histEpochInspector.addEventListener("click", event => {
  const close = event.target.closest("[data-hist-inspect-close]");
  if (close) {
    event.preventDefault();
    histInspectEpoch = null;
    renderGameHistoryPage();
    return;
  }
  const step = event.target.closest("[data-hist-inspect-step]");
  if (step) {
    event.preventDefault();
    histInspectStep(Number(step.dataset.histInspectStep || 0));
    return;
  }
  const epochButton = event.target.closest("[data-hist-epoch]");
  if (epochButton) {
    event.preventDefault();
    const epoch = asFinite(epochButton.dataset.histEpoch);
    if (epoch !== null) setHistEpochFilter(epoch);
  }
});
// History-screen keyboard layer (P1: Esc closes the inspector; P2 adds the
// game-row arrows). Guard shape mirrors dbgHandleKey so they can never clash.
document.addEventListener("keydown", histHandleKey);
// History v2 epoch-filter chip: the × button clears the display-only filter.
if (histEpochChip) histEpochChip.addEventListener("click", event => {
  const clear = event.target.closest("[data-hist-epoch-clear]");
  if (!clear) return;
  event.preventDefault();
  histEpochFilter = null;
  renderGameHistoryPage();
});
// Evaluation-region deep links: a ladder/W-L "replays" button filters the game
// history list to source=evaluation + that epoch (reuses the source select + the
// shared epoch filter). Lands on real eval replays once the .hxr carry records.
function handleHistEvalReplayClick(event) {
  const link = event.target.closest("[data-hist-eval-epoch]");
  if (!link) return;
  event.preventDefault();
  const epoch = asFinite(link.dataset.histEvalEpoch);
  if (epoch === null) return;
  if (historyFilters.source !== "evaluation") {
    historyFilters.source = "evaluation";
    if (historySourceSelect) historySourceSelect.value = "evaluation";
    historyVisibleLimit = HISTORY_PAGE_SIZE;
    resetHistoryPage();
    loadHistoryPage({ reset: true });
  }
  // Force-set (not toggle) the epoch filter so repeated clicks keep it active.
  histEpochFilter = epoch;
  renderGameHistoryPage();
}
if (histEvalPool) histEvalPool.addEventListener("click", handleHistEvalReplayClick);
// The redesigned #histStrength panel reuses the same eval-replay deep links and
// hosts a single learning-curve chart, so it shares the trends hover trio too.
if (histStrength) {
  histStrength.addEventListener("click", handleHistEvalReplayClick);
  histStrength.addEventListener("mousemove", handleHistTrendsMove);
  histStrength.addEventListener("mouseleave", hideHistTrendHover);
}
if (historySearchInput) historySearchInput.addEventListener("input", event => {
  historyFilters.query = event.target.value || "";
  historyVisibleLimit = HISTORY_PAGE_SIZE;
  resetHistoryPage();
  window.clearTimeout(historySearchTimer);
  historySearchTimer = window.setTimeout(() => loadHistoryPage({ reset: true }), 250);
  renderGameHistoryPage();
});
if (historySourceSelect) historySourceSelect.addEventListener("change", event => {
  historyFilters.source = event.target.value || "all";
  historyVisibleLimit = HISTORY_PAGE_SIZE;
  resetHistoryPage();
  loadHistoryPage({ reset: true });
  renderGameHistoryPage();
});
if (historyWinnerSelect) historyWinnerSelect.addEventListener("change", event => {
  historyFilters.winner = event.target.value || "all";
  historyVisibleLimit = HISTORY_PAGE_SIZE;
  resetHistoryPage();
  loadHistoryPage({ reset: true });
  renderGameHistoryPage();
});
if (historySortSelect) historySortSelect.addEventListener("change", event => {
  historySort = event.target.value || "newest";
  historyVisibleLimit = HISTORY_PAGE_SIZE;
  resetHistoryPage();
  loadHistoryPage({ reset: true });
  renderGameHistoryPage();
});
window.addEventListener("hashchange", () => setScreen(screenFromHash(), { preserveHash: true }));
window.setInterval(refreshHistoryIfVisible, HISTORY_REFRESH_INTERVAL_MS);
// Near-realtime tier (R3/R6). Registered AFTER the setScreen hashchange hook
// above so activeScreen is already updated when the kick checks the gate; the
// visibilitychange kick covers tab hide/show. The tick chain itself starts
// once here and self-guards (see histLivePollTick) — screen entry via the
// initial setScreen at startup needs no extra hook.
document.addEventListener("visibilitychange", histLivePollKick);
window.addEventListener("hashchange", histLivePollKick);
histLiveSchedule(HIST_LIVE_POLL_MS);
window.setInterval(histUpdateLiveTick, 1000);
on("mtFitBtn", "click", fitBoard);
on("zoomInBtn", "click", () => zoomBoardAtCenter(0.82));
on("zoomOutBtn", "click", () => zoomBoardAtCenter(1.22));
on("replayStartBtn", "click", () => setReplayIndex(0));
on("replayPrevBtn", "click", () => setReplayIndex(viewedPlacementCount() - 1));
on("replayPlayBtn", "click", toggleReplayPlay);
on("replayNextBtn", "click", () => setReplayIndex(viewedPlacementCount() + 1));
on("replayLiveBtn", "click", () => setReplayIndex(totalPlacements()));
on("replaySlider", "input", event => setReplayIndex(Number(event.target.value)));
window.addEventListener("resize", () => { if (state) render(); });
boardArea.addEventListener("click", handleBoardClick);
bindBoardViewEvents();

// ===========================================================================
// Data fetchers: match state, SealBot adapters, training-run lists/details.
// loadTrainingRuns/loadTrainingRun serve BOTH the History screen and the
// Match checkpoint seats (mtEnsureCkptList).
// ===========================================================================

async function loadState() {
  historyView = false;
  try {
    const res = await fetch("/api/state");
    const data = await safeJson(res);
    if (res.ok) {
      applyState(data, { resetReplay: true, clearBoard: true });
    } else {
      lastStatusError = (data && data.error) || "State unavailable";
      render();
    }
  } finally {
    schedulePoll(250);
  }
}

async function loadAdapters() {
  try {
    const res = await fetch("/api/adapters");
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Adapter API unavailable");
    adapters = data || {};
    adapterLoadError = null;
    syncDefaultVariant();
  } catch (error) {
    console.warn("loadAdapters: adapter API request failed", error);
    adapters = null;
    adapterLoadError = error && error.message ? error.message : "Adapter API unavailable";
  }
  render();
}

async function loadTrainingRuns(options = {}) {
  const preserveHistoryPage = Boolean(options.preserveHistoryPage);
  const previousHistoryPageKey = activeScreen === "history" ? currentHistoryPageKey() : "";
  const preferred = (trainingRun && trainingRun.name) || (historyRunSelect && historyRunSelect.value) || "";
  try {
    const res = await fetch("/api/training/runs");
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Training runs unavailable");
    trainingRuns = (data && data.runs) || [];
    trainingLoadError = "";
    const selected = trainingRuns.some(run => run.name === preferred) ? preferred : ((trainingRuns[0] && trainingRuns[0].name) || "");
    if (!historySelectionTouched && selected) historySelectedRun = selected;
    if (historySelectedRun !== HISTORY_ALL_RUNS && !trainingRuns.some(run => run.name === historySelectedRun)) {
      historySelectedRun = selected || HISTORY_ALL_RUNS;
    }
    syncTrainingRunSelect(selected);
    syncHistoryRunSelect(historySelectedRun);
    if (trainingRuns.length) {
      await loadTrainingRun(selected, { preserveHistorySelection: true });
      if (activeScreen === "history") {
        const canPreserveHistoryPage = preserveHistoryPage && currentHistoryPageKey() === previousHistoryPageKey;
        if (!canPreserveHistoryPage) resetHistoryPage();
        await ensureHistorySelectionLoaded();
        await loadHistoryPage({ reset: true, preserve: canPreserveHistoryPage });
      }
    }
    else {
      trainingRun = null;
      renderGameHistoryPage();
    }
  } catch (error) {
    trainingLoadError = error && error.message ? error.message : "Training runs unavailable";
    trainingRuns = [];
    trainingRun = null;
    renderGameHistoryPage();
  }
}

async function fetchTrainingRunDetail(name) {
  const res = await fetch(`/api/training/run?name=${encodeURIComponent(name)}`);
  const data = await safeJson(res);
  if (!res.ok) throw new Error((data && data.error) || "Training run unavailable");
  trainingRunDetails[name] = data;
  return data;
}

async function loadTrainingRun(name, options = {}) {
  if (!name) {
    trainingRun = null;
    syncTrainingRunSelect("");
    renderGameHistoryPage();
    return;
  }
  try {
    trainingRun = await fetchTrainingRunDetail(name);
    trainingLoadError = "";
    syncTrainingRunSelect(trainingRun.name || name);
    if (!options.preserveHistorySelection && historySelectedRun !== HISTORY_ALL_RUNS) {
      historySelectedRun = trainingRun.name || name;
      syncHistoryRunSelect(historySelectedRun);
    }
  } catch (error) {
    trainingRun = null;
    trainingLoadError = error && error.message ? error.message : "Training run unavailable";
  }
  renderGameHistoryPage();
}

async function ensureHistorySelectionLoaded() {
  if (!trainingRuns.length) return;
  const names = historySelectedRun === HISTORY_ALL_RUNS
    ? trainingRuns.map(run => run.name)
    : [historySelectedRun];
  const missing = names.filter(name => name && !trainingRunDetails[name]);
  if (!missing.length) return;
  historyDetailsLoading = true;
  renderGameHistoryPage();
  try {
    await Promise.all(missing.map(name => fetchTrainingRunDetail(name)));
    trainingLoadError = "";
  } catch (error) {
    trainingLoadError = error && error.message ? error.message : "Training run unavailable";
  } finally {
    historyDetailsLoading = false;
    renderGameHistoryPage();
  }
}

// ===========================================================================
// History data layer: paged game history (/api/training/history-page with the
// server's opaque cursor), filtered counts, artifact paging, the 2.5s
// /api/training/live status poll (histLive*), and the legacy full-game replay
// load (loadTrainingHistory, still wired to the game "Load" button).
// ===========================================================================

function resetHistoryPage() {
  historyPage = {
    items: [],
    nextCursor: null,
    complete: true,
    totalMatches: null,
    loading: false,
    loaded: false,
    requestKey: "",
    countLoading: false,
    countRequestKey: "",
  };
  selectedHistoryKey = "";
}

function currentHistoryPageKey() {
  return JSON.stringify({
    run: historySelectedRun || HISTORY_ALL_RUNS,
    source: historyFilters.source || "all",
    winner: historyFilters.winner || "all",
    sort: historySort || "newest",
    query: historyFilters.query || "",
  });
}

function currentHistoryTargets() {
  const runs = historyRunsForPage();
  const liveStatus = latestRunStatusForHistoryPage();
  const liveEpoch = asFinite(liveStatus && liveStatus.current_epoch);
  const selfplayEpochsSeen = runs
    .flatMap(run => (run.epoch_history || []).map(item => asFinite(item.epoch)))
    .filter(value => value !== null);
  const latestSelfplayEpoch = selfplayEpochsSeen.length ? Math.max(...selfplayEpochsSeen) : null;
  const currentEpoch = liveEpoch !== null ? liveEpoch : latestSelfplayEpoch;
  const previousEpoch = currentEpoch !== null ? currentEpoch - 1 : null;
  const selfplayEpochs = new Set([currentEpoch, previousEpoch].filter(value => value !== null && value >= 0));
  const evalEpochsSeen = runs
    .flatMap(run => (run.evaluation_history || []).map(item => asFinite(item.epoch)))
    .filter(value => value !== null && (currentEpoch === null || value <= currentEpoch));
  const latestEvalEpoch = evalEpochsSeen.length ? Math.max(...evalEpochsSeen) : null;
  const evaluationEpochs = new Set([latestEvalEpoch].filter(value => value !== null));
  const allTargets = [...selfplayEpochs, ...evaluationEpochs];
  return {
    currentEpoch,
    previousEpoch,
    selfplayEpochs,
    evaluationEpochs,
    minEpoch: allTargets.length ? Math.min(...allTargets) : null,
  };
}

function shouldAutoloadHistoryWindow() {
  return historySort === "newest" &&
    (historyFilters.source || "all") === "all" &&
    (historyFilters.winner || "all") === "all" &&
    !(historyFilters.query || "").trim();
}

function historyItemInTargetWindow(item, targets) {
  const epoch = asFinite(item && item.epoch);
  if (epoch === null) return false;
  const source = String(item && item.source || "history");
  if (source === "selfplay") return targets.selfplayEpochs.has(epoch);
  if (source === "evaluation") return targets.evaluationEpochs.has(epoch);
  return false;
}

function historyWindowBoundaryReached(items, targets) {
  if (targets.minEpoch === null) return true;
  return items.some(item => {
    const epoch = asFinite(item && item.epoch);
    return epoch !== null && epoch < targets.minEpoch;
  });
}

async function enterHistoryScreen() {
  if (!trainingRuns.length) return;
  await ensureHistorySelectionLoaded();
  if (!historyPage.loaded && !historyPage.loading) {
    await loadHistoryPage({ reset: true });
  }
}

async function loadHistoryPage(options = {}) {
  if (!trainingRuns.length || historyPage.loading) return;
  const reset = Boolean(options.reset);
  const append = Boolean(options.append);
  const preserve = Boolean(options.preserve);
  const autoloadWindow = reset && !append && shouldAutoloadHistoryWindow();
  const targets = autoloadWindow ? currentHistoryTargets() : null;
  if (reset && !preserve) {
    historyPage.items = [];
    historyPage.nextCursor = null;
    historyPage.complete = true;
    historyPage.totalMatches = null;
    historyPage.loaded = false;
  }
  if (append && !historyPage.nextCursor) return;

  const requestKey = currentHistoryPageKey();
  historyPage.loading = true;
  historyPage.requestKey = requestKey;
  renderGameHistoryPage();
  try {
    const fetchedItems = [];
    let data = null;
    let cursor = append ? historyPage.nextCursor : "";
    let pageCount = 0;
    do {
      const params = new URLSearchParams({
        run: historySelectedRun || HISTORY_ALL_RUNS,
        limit: String(autoloadWindow ? HISTORY_AUTOLOAD_PAGE_SIZE : HISTORY_PAGE_SIZE),
        source: historyFilters.source || "all",
        winner: historyFilters.winner || "all",
        sort: historySort || "newest",
        query: historyFilters.query || "",
        include_total: "0",
      });
      if (cursor) params.set("cursor", cursor);
      const res = await fetch(`/api/training/history-page?${params.toString()}`);
      data = await safeJson(res);
      if (!res.ok) throw new Error((data && data.error) || "Game histories unavailable");
      fetchedItems.push(...((data && data.items) || []));
      cursor = (data && data.next_cursor) || "";
      pageCount += 1;
    } while (
      autoloadWindow &&
      cursor &&
      pageCount < 8 &&
      !historyWindowBoundaryReached(fetchedItems, targets)
    );
    if (historyPage.requestKey !== requestKey) return;
    const items = autoloadWindow && targets && targets.minEpoch !== null
      ? fetchedItems.filter(item => historyItemInTargetWindow(item, targets))
      : fetchedItems;
    historyPage.items = append ? [...historyPage.items, ...items] : items;
    historyPage.nextCursor = (data && data.next_cursor) || null;
    historyPage.complete = Boolean(data && data.complete);
    if (data && data.total_matches !== null && data.total_matches !== undefined) {
      historyPage.totalMatches = data.total_matches;
    } else if (autoloadWindow) {
      historyPage.totalMatches = items.length;
    } else if (!append && !preserve) {
      historyPage.totalMatches = null;
    }
    historyPage.loaded = true;
    trainingLoadError = "";
    if (!append && !autoloadWindow) loadHistoryCount(requestKey);
  } catch (error) {
    trainingLoadError = error && error.message ? error.message : "Game histories unavailable";
  } finally {
    if (historyPage.requestKey === requestKey) {
      historyPage.loading = false;
      renderGameHistoryPage();
    }
  }
}

async function loadHistoryCount(expectedKey = "") {
  if (!trainingRuns.length) return;
  const requestKey = expectedKey || currentHistoryPageKey();
  if (historyPage.countLoading && historyPage.countRequestKey === requestKey) return;
  historyPage.countLoading = true;
  historyPage.countRequestKey = requestKey;
  renderGameHistoryPage();
  try {
    const params = new URLSearchParams({
      run: historySelectedRun || HISTORY_ALL_RUNS,
      source: historyFilters.source || "all",
      winner: historyFilters.winner || "all",
      query: historyFilters.query || "",
    });
    const res = await fetch(`/api/training/history-count?${params.toString()}`);
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Game count unavailable");
    if (historyPage.countRequestKey !== requestKey || currentHistoryPageKey() !== requestKey) return;
    historyPage.totalMatches = data && data.total_matches !== undefined ? data.total_matches : null;
    trainingLoadError = "";
  } catch (error) {
    console.warn("loadHistoryCount: history count request failed", error);
  } finally {
    if (historyPage.countRequestKey === requestKey) {
      historyPage.countLoading = false;
      renderGameHistoryPage();
    }
  }
}

// UNUSED(2026-06-12): no callers found anywhere in app.js/index.html — its
// callers (renderTraining/trainingArtifactRow) were removed by the Match-v2
// rewrite. Also the only client of GET /api/training/artifacts-page (now
// exercised by tests only). docs/specs/history_screen_v2_spec.md still lists
// it as load-bearing; that list is stale.
async function loadMoreArtifacts() {
  if (!trainingRun || !trainingRun.artifacts_page || !trainingRun.artifacts_page.next_cursor) return;
  try {
    const params = new URLSearchParams({
      run: trainingRun.name,
      limit: String(ARTIFACT_PAGE_SIZE),
      cursor: trainingRun.artifacts_page.next_cursor,
    });
    const res = await fetch(`/api/training/artifacts-page?${params.toString()}`);
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Artifacts unavailable");
    trainingRun.artifacts = [...(trainingRun.artifacts || []), ...((data && data.items) || [])];
    trainingRun.artifacts_page = {
      ...(trainingRun.artifacts_page || {}),
      next_cursor: (data && data.next_cursor) || null,
      complete: Boolean(data && data.complete),
    };
    trainingLoadError = "";
  } catch (error) {
    trainingLoadError = error && error.message ? error.message : "Artifacts unavailable";
  }
  renderGameHistoryPage();
}

async function refreshHistoryIfVisible() {
  if (activeScreen !== "history" || historyRefreshInFlight || pendingRequest) return;
  historyRefreshInFlight = true;
  try {
    await loadTrainingRuns({ preserveHistoryPage: true });
  } finally {
    historyRefreshInFlight = false;
  }
}

// --- Near-realtime status poll (R3-R5): /api/training/live every ~2.5s. ---
// The loop is a permanent self-rescheduling setTimeout chain with a cheap
// inactivity guard rather than an event-stopped interval: setScreen /
// enterHistoryScreen are frozen shared helpers, so there is no seam to hook a
// hard stop into — the guard no-ops (one timer/s, zero fetches) whenever the
// History screen is hidden, and the visibilitychange/hashchange kicks below
// only restore the fast cadence immediately instead of waiting out a backoff.

function histLivePollActive() {
  return activeScreen === "history" && document.visibilityState === "visible";
}

// Poll the same run the status band displays: the selected run, or under
// All runs the newest-modified one (the histTrendRun pick, which is also the
// run latestRunStatusForHistoryPage reports on).
function histLiveRunName() {
  if (historySelectedRun && historySelectedRun !== HISTORY_ALL_RUNS) return historySelectedRun;
  const run = histTrendRun(historyRunsForPage());
  return run && run.name ? String(run.name) : "";
}

function histLiveSchedule(delayMs) {
  window.clearTimeout(histLivePollTimer);
  histLivePollTimer = window.setTimeout(histLivePollTick, delayMs);
}

async function histLivePollTick() {
  if (!histLivePollActive()) {
    histLiveSchedule(HIST_LIVE_POLL_IDLE_MS);
    return;
  }
  if (histLivePollInFlight) {
    histLiveSchedule(histLivePollDelayMs);
    return;
  }
  const runName = histLiveRunName();
  if (!runName) {
    histLiveSchedule(HIST_LIVE_POLL_MS);
    return;
  }
  histLivePollInFlight = true;
  try {
    // Plain fetch on purpose: this tier must never touch pendingRequest or any
    // of the match/history request machinery.
    const res = await fetch(`/api/training/live?run=${encodeURIComponent(runName)}`);
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Live status unavailable");
    histLivePollDelayMs = HIST_LIVE_POLL_MS; // success resets the backoff
    histLiveLastTs = Date.now();
    histApplyLiveStatus(runName, data && data.status);
  } catch (error) {
    // Back off so a down/unreachable backend costs at most one request per 30s.
    histLivePollDelayMs = Math.min(histLivePollDelayMs * 2, HIST_LIVE_POLL_MAX_MS);
  } finally {
    histLivePollInFlight = false;
    histUpdateLiveTick();
    histLiveSchedule(histLivePollDelayMs);
  }
}

// Immediate wake on (re)activation — resets the backoff and polls now instead
// of waiting out the idle/backoff delay. Never overlaps an in-flight fetch.
function histLivePollKick() {
  if (!histLivePollActive() || histLivePollInFlight) return;
  histLivePollDelayMs = HIST_LIVE_POLL_MS;
  histLiveSchedule(0);
}

// R4/R5: apply one poll result. Signature-compare first so an unchanged status
// causes zero DOM churn; otherwise patch the run's cached status in place and
// re-render ONLY the status band (drawer/inspector state lives in module vars
// and the other regions re-render on the 15s full refresh).
function histApplyLiveStatus(runName, status) {
  if (!status || typeof status !== "object") return;
  const sameRun = histLiveLastRun === runName;
  const boundary = sameRun && histLiveEpochBoundary(histLiveLastStatus, status);
  histLiveLastRun = runName;
  histLiveLastStatus = status;
  const sig = JSON.stringify(status);
  if (sameRun && sig === histLiveLastSig) return;
  histLiveLastSig = sig;
  const detail = trainingRunDetails[runName];
  if (detail && detail.status && typeof detail.status === "object") {
    // The cached run.status is _training_run_status = live status + the
    // synthesized "history"/"latest_selfplay_record" blocks. Keep those two
    // from the old object (the band's run pick sorts on history.latest_modified)
    // and take everything else fresh — a plain spread-merge would resurrect
    // stale optional keys like a finished selfplay_live.
    const merged = { ...status };
    if ("history" in detail.status) merged.history = detail.status.history;
    if ("latest_selfplay_record" in detail.status) {
      merged.latest_selfplay_record = detail.status.latest_selfplay_record;
    }
    detail.status = merged;
    renderHistStatusBand(historyRunsForPage());
  }
  if (boundary) histLiveTriggerFullRefresh();
}

// R5: an epoch/stage boundary means new epoch_history/eval rows exist that the
// band patch cannot show — pull one full refresh forward instead of waiting
// for the 15s floor.
function histLiveEpochBoundary(prev, next) {
  if (!prev || !next) return false;
  if (asFinite(prev.current_epoch) !== asFinite(next.current_epoch)) return true;
  const prevSp = prev.selfplay_live ? String(prev.selfplay_live.status || "") : "";
  const nextSp = next.selfplay_live ? String(next.selfplay_live.status || "") : "";
  if (nextSp === "completed" && prevSp && prevSp !== "completed") return true;
  if (String(prev.stage || "") !== String(next.stage || "")) return true;
  // A sub-phase flip (self-play -> selecting window -> training -> evaluating)
  // means the in-flight epoch's provisional row + segment files changed; pull
  // the refresh forward so the epoch table tracks the phase within seconds.
  if (String(prev.sub_phase || "") !== String(next.sub_phase || "")) return true;
  return false;
}

function histLiveTriggerFullRefresh() {
  const now = Date.now();
  if (now - histLiveRefreshAt < 2000) return; // debounce burst transitions
  histLiveRefreshAt = now;
  // A boundary means new per-epoch diagnostics files exist; drop the telemetry
  // strip cache so the inspector re-fetches /api/training/epochs (backend is
  // itself mtime-cached, so this is a cheap rescan at most).
  histEpochsCache.clear();
  // The existing 15s path; its historyRefreshInFlight/pendingRequest guards
  // make this a no-op when a refresh is already running.
  refreshHistoryIfVisible();
}

// R6: "live · updated Ns ago" — textContent only, the band innerHTML is
// otherwise untouched between renders.
function histUpdateLiveTick() {
  if (activeScreen !== "history") return;
  const el = document.getElementById("histLiveTick");
  if (!el) return;
  el.textContent = histLiveLastTs
    ? `live · updated ${Math.max(0, Math.round((Date.now() - histLiveLastTs) / 1000))}s ago`
    : "";
}

async function loadTrainingHistory(runName, artifactPath, recordIndex = 0) {
  if (!runName || !artifactPath) return;
  abortPoll();
  stopReplay();
  setPending(true);
  try {
    const params = new URLSearchParams({ run: runName, path: artifactPath, record: String(recordIndex || 0) });
    const res = await fetch(`/api/training/history?${params.toString()}`);
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "Game history unavailable");
    historyView = true;
    selectedHistoryKey = historyItemKey({ run: runName, path: artifactPath, record_index: recordIndex || 0 });
    lastStatusError = `Loaded ${artifactPath}`;
    applyState(data, { resetReplay: true, clearBoard: true });
    navigateScreen("match");
  } catch (error) {
    lastStatusError = error && error.message ? error.message : "Game history unavailable";
    render();
  } finally {
    setPending(false);
    renderGameHistoryPage();
  }
}

// ===========================================================================
// HTTP helpers + screen routing (#match/#history/#debug hash) + the match
// long-poll loop (pollState drives /api/state?since=&timeout_ms=).
// ===========================================================================

async function post(url, payload, options = {}) {
  if (pendingRequest) return;
  abortPoll();
  const seq = ++requestSeq;
  setPending(true);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await safeJson(res);
    if (seq !== requestSeq) return;
    if (!res.ok) {
      lastStatusError = (data && data.error) || "Request failed";
      if (data && data.state) applyState(data.state, { preserveReplay: true });
      else render();
    } else {
      lastStatusError = "";
      applyState(data, {
        resetReplay: Boolean(options.resetReplay),
        clearBoard: Boolean(options.clearBoard),
        preserveReplay: !options.resetReplay,
      });
    }
  } catch (error) {
    console.error("post: request to " + url + " failed", error);
    if (seq === requestSeq) {
      lastStatusError = "Request failed";
      render();
    }
  } finally {
    if (seq === requestSeq) {
      setPending(false);
      render();
      schedulePoll(250);
    }
  }
}

function setPending(value) {
  pendingRequest = value;
  if (value) stopReplay();
  document.body.classList.toggle("pending", value);
}

async function safeJson(res) {
  try {
    return await res.json();
  } catch (error) {
    console.warn("safeJson: failed to parse response body", error);
    return null;
  }
}

function screenFromHash() {
  const hash = String(window.location.hash || "").replace(/^#\/?/, "");
  // `#debug?run=...` deep links carry nav state after `?` — any `#debug...`
  // prefix still routes to the debug screen (the debug module owns the query).
  const name = hash.split("?")[0];
  return SCREENS.includes(name) ? name : "match";
}

function navigateScreen(screen) {
  setScreen(screen);
  const hash = `#${activeScreen}`;
  if (window.location.hash !== hash) window.location.hash = hash;
}

function setScreen(screen, options = {}) {
  const previousScreen = activeScreen;
  activeScreen = SCREENS.includes(screen) ? screen : "match";
  // Match-screen lifecycle: stop the long-poll and any replay timer when we
  // leave it, and resume polling when we (re)enter it. schedulePoll/pollState
  // also gate on activeScreen === "match", so this can never run while History
  // or Debug is up.
  if (previousScreen === "match" && activeScreen !== "match") {
    abortPoll();
    stopReplay();
    window.clearTimeout(pollTimer);
  } else if (activeScreen === "match" && previousScreen !== "match") {
    schedulePoll(0);
  }
  if (matchScreen) matchScreen.hidden = activeScreen !== "match";
  if (historyScreen) historyScreen.hidden = activeScreen !== "history";
  if (debugScreen) debugScreen.hidden = activeScreen !== "debug";
  document.querySelectorAll("[data-screen]").forEach(button => {
    button.classList.toggle("active", button.dataset.screen === activeScreen);
  });
  document.body.classList.toggle("history-screen-active", activeScreen === "history");
  document.body.classList.toggle("debug-screen-active", activeScreen === "debug");
  if (!options.preserveHash) {
    const hash = `#${activeScreen}`;
    if (window.location.hash && window.location.hash !== hash) window.history.replaceState(null, "", hash);
  }
  renderGameHistoryPage();
  if (activeScreen === "history") enterHistoryScreen();
  if (activeScreen === "debug") enterDebugScreen();
  // Re-render the match view once it is actually visible so layout-dependent
  // work (move-history centering, board fit) runs with real element widths
  // instead of the zero width it would see while the screen was hidden.
  if (activeScreen === "match" && state) render();
}

function syncTrainingRunSelect(selected = "") {
  // The Match screen's Training Runs card is gone (the History screen owns run
  // browsing). History code still calls this on run loads; keep it as a no-op
  // so those call sites stay valid.
  void selected;
}

function syncHistoryRunSelect(selected = historySelectedRun) {
  if (!historyRunSelect) return;
  const runOptions = trainingRuns
    .map(run => `<option value="${escapeAttr(run.name)}">${escapeText(run.name)}</option>`)
    .join("");
  historyRunSelect.innerHTML = trainingRuns.length
    ? `<option value="${HISTORY_ALL_RUNS}">All runs</option>${runOptions}`
    : `<option value="">No runs</option>`;
  const value = selected === HISTORY_ALL_RUNS || trainingRuns.some(run => run.name === selected)
    ? selected
    : HISTORY_ALL_RUNS;
  historySelectedRun = value || HISTORY_ALL_RUNS;
  historyRunSelect.value = historySelectedRun;
}

function applyState(next, options = {}) {
  if (!next || typeof next !== "object") return;
  if (isSameGame(next) && !isNewerOrSameState(next)) return;
  const wasLive = !state || isLiveView();
  const currentVersion = Number(state && state.version);
  const nextVersion = Number(next && next.version);
  if (Number.isFinite(currentVersion) && Number.isFinite(nextVersion) && nextVersion > currentVersion && !next.error) {
    lastStatusError = "";
  }
  state = next;
  if (options.clearBoard) clearBoardView();
  if (options.resetReplay) {
    resetReplay();
  } else if (wasLive && !options.preserveReplay) {
    replayIndex = null;
  } else if (replayIndex !== null) {
    replayIndex = Math.min(replayIndex, totalPlacements());
    if (replayIndex === totalPlacements() && wasLive) replayIndex = null;
  }
  render();
}

function isSameGame(next) {
  if (!state || !next) return true;
  if (!state.game_id || !next.game_id) return true;
  return state.game_id === next.game_id;
}

function isNewerOrSameState(next) {
  const currentVersion = Number(state && state.version);
  const nextVersion = Number(next && next.version);
  if (!Number.isFinite(currentVersion) || !Number.isFinite(nextVersion)) return true;
  return nextVersion >= currentVersion;
}

function schedulePoll(delay = 0) {
  // The match long-poll only runs while the match screen is active and we are
  // not pinned to a static history view.
  if (historyView || activeScreen !== "match") return;
  window.clearTimeout(pollTimer);
  pollTimer = window.setTimeout(pollState, delay);
}

function abortPoll() {
  if (pollAbort) {
    pollAbort.abort();
    pollAbort = null;
  }
  polling = false;
}

async function pollState() {
  if (historyView || activeScreen !== "match") return;
  if (polling || pendingRequest) {
    schedulePoll(600);
    return;
  }
  polling = true;
  const controller = new AbortController();
  pollAbort = controller;
  let failed = false;
  try {
    const params = new URLSearchParams();
    const version = stateVersion();
    if (version !== null) {
      params.set("since", String(version));
      params.set("timeout_ms", "15000");
    }
    const res = await fetch(`/api/state${params.toString() ? "?" + params.toString() : ""}`, { signal: controller.signal });
    const data = await safeJson(res);
    if (res.ok && data) {
      pollFailures = 0;
      if (lastStatusError === "Live update paused") lastStatusError = "";
      applyState(data, { preserveReplay: true });
    }
  } catch (error) {
    if (!controller.signal.aborted) {
      failed = true;
      console.warn("pollState: live state poll failed", error);
      lastStatusError = "Live update paused";
      render();
    }
  } finally {
    if (pollAbort === controller) pollAbort = null;
    polling = false;
    // On consecutive failures, back off exponentially (capped) instead of
    // hammering the server every 300ms; a successful response resets the streak.
    if (failed) {
      pollFailures += 1;
      schedulePoll(Math.min(300 * Math.pow(2, pollFailures), 5000));
    } else {
      schedulePoll(document.hidden ? 2500 : 300);
    }
  }
}

// ===========================================================================
// Match screen v2 (mt*) rendering: top-level render() + the setup strip
// (seat kind/variant/checkpoint pickers, series, start blockers).
// ===========================================================================

// F2.6 — top-level match render. Every step below is null-safe against a
// missing state (first paint runs before /api/state answers): the board is
// simply skipped, and renderMtStatus/renderMoveHistory/renderReplay all
// tolerate state === null.
function render() {
  document.body.classList.toggle("pending", pendingRequest);
  document.body.classList.toggle("replay-mode", Boolean(state) && !isLiveView());
  document.body.classList.toggle("bot-thinking", isBotThinking());
  document.body.classList.toggle("state-error", turnStatus() === "error" || Boolean((state && state.error) || lastStatusError));
  if (state) renderBoard(buildBoardModel());
  renderMtSetup();
  renderMtStatus();
  renderMtPlayers();
  renderMtInsight();
  renderMtSeries();
  renderMoveHistory();
  renderTurnOverlay();
  renderReplay();
  const replayDisabled = totalPlacements() === 0;
  document.querySelectorAll(".replay-buttons button").forEach(button => { button.disabled = replayDisabled; });
  const slider = document.getElementById("replaySlider");
  if (slider) slider.disabled = replayDisabled;
}

// F2.2 — the setup strip. Idempotent and poll-safe: every write into a select
// or input is skipped while that element is focused, so the 300ms poll
// re-render never clobbers a user mid-edit (§8 focused-input guard).
function renderMtSetup() {
  const seats = [["player0", 0], ["player1", 1]];
  const anyCheckpoint = seats.some(([seat]) => mtSeatCfg(seat).kind === "checkpoint");
  for (const [seat, idx] of seats) {
    const cfg = mtSeatCfg(seat);
    const kindEl = document.getElementById(`mtKind${idx}`);
    if (kindEl) {
      for (const option of kindEl.options) {
        if (!option.value.startsWith("sealbot-")) continue;
        const variant = option.value.slice("sealbot-".length);
        const available = mtSealbotVariantAvailable(variant);
        option.disabled = !available;
        option.title = available ? "" : mtSealbotVariantError(variant);
      }
      if (document.activeElement !== kindEl) kindEl.value = mtSeatKindValue(cfg);
    }
    const cfgWrap = document.querySelector(`.mt-ckpt-cfg[data-seat="${seat}"]`);
    if (cfgWrap) cfgWrap.hidden = cfg.kind !== "checkpoint";
    if (cfg.kind !== "checkpoint") continue;

    // Lazy data: the run list once, then per-run checkpoint lists (cached).
    if (!trainingRuns.length && !mtRunsRequested) {
      mtRunsRequested = true;
      loadTrainingRuns().then(() => renderMtSetup());
    }
    if (trainingRuns.length && (!cfg.run || !trainingRuns.some(run => run.name === cfg.run))) {
      cfg.run = trainingRuns[0].name;
      cfg.checkpoint = "";
    }
    const runEl = document.getElementById(`mtRun${idx}`);
    if (runEl && document.activeElement !== runEl) {
      runEl.innerHTML = trainingRuns.length
        ? trainingRuns.map(run => `<option value="${escapeAttr(run.name)}">${escapeText(run.name)}</option>`).join("")
        : `<option value="">${trainingLoadError ? "runs unavailable" : "loading runs…"}</option>`;
      runEl.value = cfg.run || "";
    }
    if (cfg.run) mtEnsureCkptList(cfg.run);
    const list = cfg.run ? mtCkptLists.get(cfg.run) : null;
    if (list && list.length && (!cfg.checkpoint || !list.some(item => item.name === cfg.checkpoint))) {
      cfg.checkpoint = list[0].name; // newest-first per §1.2
    }
    const ckptEl = document.getElementById(`mtCkpt${idx}`);
    if (ckptEl && document.activeElement !== ckptEl) {
      if (!list) {
        ckptEl.innerHTML = `<option value="">loading…</option>`;
      } else if (!list.length) {
        ckptEl.innerHTML = `<option value="">no checkpoints</option>`;
      } else {
        ckptEl.innerHTML = list.map(item => `<option value="${escapeAttr(item.name)}">${escapeText(mtCkptLabel(item))}</option>`).join("");
        ckptEl.value = cfg.checkpoint;
      }
    }
    const visitsEl = document.getElementById(`mtVisits${idx}`);
    if (visitsEl && document.activeElement !== visitsEl) visitsEl.value = String(cfg.visits ?? 256);
    const modeEl = document.getElementById(`mtSearchMode${idx}`);
    if (modeEl && document.activeElement !== modeEl) modeEl.value = cfg.mode === "policy" ? "policy" : "search";
  }

  const timeEl = document.getElementById("mtTimeLimit");
  if (timeEl && document.activeElement !== timeEl) timeEl.value = String(mtSetup.time_limit);
  const seedEl = document.getElementById("mtSeed");
  if (seedEl && document.activeElement !== seedEl) seedEl.value = mtSetup.seed === null ? "" : String(mtSetup.seed);
  const gamesEl = document.getElementById("mtSeriesGames");
  if (gamesEl && document.activeElement !== gamesEl) gamesEl.value = String(mtSetup.series.games);
  const altEl = document.getElementById("mtAlternate");
  if (altEl && document.activeElement !== altEl) altEl.checked = Boolean(mtSetup.series.alternate);

  // Buttons + status note (F2.3).
  const blockers = mtStartBlockers();
  const startBtn = document.getElementById("mtStartBtn");
  if (startBtn) {
    startBtn.textContent = mtSetup.series.games > 1
      ? "Start series"
      : (state && totalPlacements() ? "Rematch" : "Start match");
    startBtn.disabled = pendingRequest || blockers.length > 0;
  }
  const stopBtn = document.getElementById("mtStopBtn");
  if (stopBtn) {
    const seriesRemaining = Boolean(state && state.series && !state.series.finished);
    stopBtn.disabled = pendingRequest || !state || Boolean(state.stopped)
      || (turnStatus() === "terminal" && !seriesRemaining);
  }
  const note = document.getElementById("mtSetupNote");
  if (note) {
    const parts = [...blockers, mtAdapterStatusLine()];
    if (anyCheckpoint) parts.push("Checkpoint bots run on the shared CPU debug worker; Debug analyses queue behind them.");
    note.textContent = parts.filter(Boolean).join(" · ");
  }
}

function mtSeatCfg(seat) {
  return mtSetup.seats[seat] || (mtSetup.seats[seat] = { kind: "manual" });
}

// Seat spec dict -> the #mtKindX select value.
function mtSeatKindValue(cfg) {
  if (cfg.kind === "sealbot") return `sealbot-${cfg.variant || "current"}`;
  return cfg.kind === "checkpoint" ? "checkpoint" : "manual";
}

function mtSealbotVariantAvailable(variant) {
  return sealbotVariants().some(item => item.id === variant && item.available !== false);
}

function mtSealbotVariantError(variant) {
  const known = sealbotVariants().find(item => item.id === variant);
  const adapter = sealbotAdapter();
  return (known && known.error) || (adapter && adapter.error) || adapterLoadError || "SealBot variant unavailable";
}

function mtCkptLabel(item) {
  if (item.epoch !== null && item.epoch !== undefined) return `epoch ${item.epoch}${item.latest ? " (latest)" : ""}`;
  return String(item.name || "").replace(/\.pt$/, "");
}

// Lazy, cached checkpoint-list fetch for one run. Failures cache an empty list
// (surfaced as "run has no checkpoints" by mtStartBlockers) instead of retrying
// on every poll tick.
function mtEnsureCkptList(run) {
  if (!run || mtCkptLists.has(run) || mtCkptListsLoading.has(run)) return;
  mtCkptListsLoading.add(run);
  fetch(`/api/debug/checkpoints?run=${encodeURIComponent(run)}`)
    .then(async res => {
      const data = await safeJson(res);
      if (!res.ok) throw new Error((data && data.error) || "Checkpoint list unavailable");
      mtCkptLists.set(run, (data && data.checkpoints) || []);
    })
    .catch(error => {
      console.warn("mtEnsureCkptList: checkpoint list failed for " + run, error);
      mtCkptLists.set(run, []);
    })
    .finally(() => {
      mtCkptListsLoading.delete(run);
      renderMtSetup();
    });
}

// F2.3 — Start preconditions. Each blocker doubles as the user-facing reason in
// #mtSetupNote, so a disabled Start button always carries an explanation (A1).
function mtStartBlockers() {
  const blockers = [];
  for (const [seat, name] of [["player0", "P0"], ["player1", "P1"]]) {
    const cfg = mtSeatCfg(seat);
    if (cfg.kind === "sealbot" && !mtSealbotVariantAvailable(cfg.variant || "current")) {
      blockers.push(`${name}: SealBot ${cfg.variant || "current"} unavailable`);
    } else if (cfg.kind === "checkpoint") {
      if (!cfg.run) {
        blockers.push(trainingRuns.length ? `${name}: pick a training run` : `${name}: no training runs found`);
      } else if (!cfg.checkpoint) {
        const list = mtCkptLists.get(cfg.run);
        blockers.push(list && !list.length ? `${name}: run has no checkpoints` : `${name}: checkpoint list loading`);
      }
    }
  }
  return blockers;
}

// Adapter availability line (the old adapter-status logic, now one part of
// the setup note).
function mtAdapterStatusLine() {
  const sealbot = sealbotAdapter();
  if (adapterLoadError) return adapterLoadError;
  if (!sealbot) return "Manual play available. SealBot API not detected.";
  if (!sealbot.configured && !hasAvailableSealBotVariant()) return sealbot.error || "SealBot path is not configured.";
  const available = sealbotVariants().filter(variant => variant.available !== false);
  if (!available.length) {
    const firstError = (sealbotVariants().find(variant => variant.error) || {}).error;
    return firstError || sealbot.error || "No SealBot variants are available.";
  }
  return `SealBot ready: ${available.map(variant => variant.label || variant.id).join(", ")}`;
}

// F2.4 — DOM -> mtSetup. Pure reads, so it is always safe to call (the
// focused-input guard only matters in the write direction, in renderMtSetup).
function mtReadSetupFromDom() {
  for (const [seat, idx] of [["player0", 0], ["player1", 1]]) {
    const previous = mtSeatCfg(seat);
    const kindEl = document.getElementById(`mtKind${idx}`);
    const kindValue = kindEl ? kindEl.value : mtSeatKindValue(previous);
    if (kindValue === "checkpoint") {
      const cfg = previous.kind === "checkpoint"
        ? previous
        : { kind: "checkpoint", run: "", checkpoint: "", visits: 256, mode: "search" };
      const runEl = document.getElementById(`mtRun${idx}`);
      const runChanged = Boolean(runEl && runEl.value && runEl.value !== cfg.run);
      if (runChanged) {
        cfg.run = runEl.value;
        cfg.checkpoint = ""; // re-default to the new run's newest checkpoint
      }
      const ckptEl = document.getElementById(`mtCkpt${idx}`);
      // When the run JUST changed, the checkpoint select still holds the OLD
      // run's options (and checkpoint filenames repeat across runs), so only
      // trust its value when the run is unchanged; renderMtSetup repopulates
      // it with the new run's list (newest first) on the next render.
      if (!runChanged && ckptEl && ckptEl.value) cfg.checkpoint = ckptEl.value;
      const visitsEl = document.getElementById(`mtVisits${idx}`);
      const visitsRaw = visitsEl ? String(visitsEl.value).trim() : "";
      if (visitsRaw !== "") {
        const visits = Number(visitsRaw);
        if (Number.isFinite(visits)) cfg.visits = clamp(Math.round(visits), 8, 2048);
      }
      const modeEl = document.getElementById(`mtSearchMode${idx}`);
      if (modeEl) cfg.mode = modeEl.value === "policy" ? "policy" : "search";
      mtSetup.seats[seat] = cfg;
    } else if (kindValue.startsWith("sealbot-")) {
      mtSetup.seats[seat] = { kind: "sealbot", variant: kindValue.slice("sealbot-".length) || "current" };
    } else {
      mtSetup.seats[seat] = { kind: "manual" };
    }
  }
  // Scalar options: only update from elements that exist, so a missing
  // control never silently resets a configured value.
  const timeEl = document.getElementById("mtTimeLimit");
  if (timeEl) {
    const time = Number(timeEl.value);
    mtSetup.time_limit = Number.isFinite(time) && time > 0 ? clamp(time, 0.01, 30) : 0.05;
  }
  const seedEl = document.getElementById("mtSeed");
  if (seedEl) {
    const seedText = String(seedEl.value).trim();
    const seed = Number(seedText);
    mtSetup.seed = seedText !== "" && Number.isFinite(seed) ? Math.trunc(seed) : null;
  }
  const gamesEl = document.getElementById("mtSeriesGames");
  if (gamesEl) {
    const games = Number(gamesEl.value);
    mtSetup.series.games = Number.isFinite(games) && games > 0 ? clamp(Math.round(games), 1, 25) : 1;
  }
  const altEl = document.getElementById("mtAlternate");
  if (altEl) mtSetup.series.alternate = Boolean(altEl.checked);
}

// F2.3 — the §3.1 /api/new body. Re-reads the DOM first so an uncommitted
// input edit (change event not fired yet) is still captured at Start time.
function buildMtMatchPayload() {
  mtReadSetupFromDom();
  const body = {
    players: {
      player0: mtSeatSpecBody("player0"),
      player1: mtSeatSpecBody("player1"),
    },
    time_limit: mtSetup.time_limit,
    seed: mtSetup.seed,
  };
  if (mtSetup.series.games > 1) {
    body.series = { games: mtSetup.series.games, alternate: Boolean(mtSetup.series.alternate) };
  }
  return body;
}

function mtSeatSpecBody(seat) {
  const cfg = mtSeatCfg(seat);
  if (cfg.kind === "sealbot") return { kind: "sealbot", variant: cfg.variant || "current" };
  if (cfg.kind === "checkpoint") {
    return {
      kind: "checkpoint",
      run: cfg.run || "",
      checkpoint: cfg.checkpoint || "",
      visits: clamp(Math.round(Number(cfg.visits) || 256), 8, 2048),
      mode: cfg.mode === "policy" ? "policy" : "search",
    };
  }
  return { kind: "manual" };
}

function sealbotAdapter() {
  if (!adapters) return null;
  return adapters.sealbot || adapters.SealBot || null;
}

function sealbotVariants() {
  const sealbot = sealbotAdapter();
  const raw = sealbot && Array.isArray(sealbot.variants) ? sealbot.variants : [];
  return raw.map(variant => ({
    id: String(variant.id || variant.name || variant.label || ""),
    label: String(variant.label || variant.id || variant.name || "SealBot"),
    available: variant.available !== false,
    error: variant.error || "",
  })).filter(variant => variant.id);
}

function hasAvailableSealBotVariant() {
  return sealbotVariants().some(variant => variant.available !== false);
}

function sealbotDefaultVariant() {
  const sealbot = sealbotAdapter();
  return (sealbot && (sealbot.default_variant || sealbot.defaultVariant)) || (sealbotVariants()[0] && sealbotVariants()[0].id) || "current";
}

// Adapter refresh: a configured-but-unavailable sealbot seat snaps to the
// preferred available variant so Start never silently targets a dead variant.
function syncDefaultVariant() {
  const variants = sealbotVariants();
  const preferred = variants.find(variant => variant.id === sealbotDefaultVariant() && variant.available !== false)
    || variants.find(variant => variant.available !== false);
  if (!preferred) return;
  for (const seat of ["player0", "player1"]) {
    const cfg = mtSeatCfg(seat);
    if (cfg.kind === "sealbot" && !mtSealbotVariantAvailable(cfg.variant || "current")) {
      cfg.variant = preferred.id;
    }
  }
}

// ===========================================================================
// Board SVG + camera: model build, hex rendering, pan/zoom/pinch gestures.
// Shared by the Match board and the History replay view; the Debug board
// keeps its OWN view state (dbg.view) and only reuses the pure geometry
// helpers (center/path/viewForBox/clamp).
// ===========================================================================

function buildBoardModel() {
  const shownPlacements = visiblePlacements();
  const occupied = new Map(shownPlacements.map(p => [`${p.q},${p.r}`, p]));
  const liveLegal = new Map((state.legal || []).map(c => [`${c.q},${c.r}`, c]));
  const legal = isLiveView() ? liveLegal : new Map();
  const cells = new Map();
  for (const [key, cell] of liveLegal) cells.set(key, cell);
  for (const placement of state.placements || []) cells.set(`${placement.q},${placement.r}`, placement);

  let minX = -HEX;
  let maxX = HEX;
  let minY = -HEX;
  let maxY = HEX;
  let focusMinX = Infinity;
  let focusMaxX = -Infinity;
  let focusMinY = Infinity;
  let focusMaxY = -Infinity;
  const data = [];

  for (const [key, cell] of cells) {
    const c = center(cell.q, cell.r);
    minX = Math.min(minX, c.x - HEX * 1.4);
    maxX = Math.max(maxX, c.x + HEX * 1.4);
    minY = Math.min(minY, c.y - HEX * 1.4);
    maxY = Math.max(maxY, c.y + HEX * 1.4);
    if (occupied.has(key)) {
      focusMinX = Math.min(focusMinX, c.x);
      focusMaxX = Math.max(focusMaxX, c.x);
      focusMinY = Math.min(focusMinY, c.y);
      focusMaxY = Math.max(focusMaxY, c.y);
    }
    data.push({ key, q: cell.q, r: cell.r, x: c.x, y: c.y, placement: occupied.get(key), legal: legal.has(key) });
  }

  const hasFocus = Number.isFinite(focusMinX);
  const focusPad = HEX * 7;
  const focus = hasFocus ? {
    minX: Math.max(minX, focusMinX - focusPad),
    maxX: Math.min(maxX, focusMaxX + focusPad),
    minY: Math.max(minY, focusMinY - focusPad),
    maxY: Math.min(maxY, focusMaxY + focusPad),
  } : null;

  const boardBounds = { minX, maxX, minY, maxY };
  const camera = buildCameraBox(shownPlacements, liveLegal, boardBounds);

  return { data, minX, maxX, minY, maxY, focus, camera };
}

function renderBoard(board) {
  const compact = window.innerWidth < 1200;
  const box = board.camera || (compact && board.focus ? board.focus : board);
  const pad = compact ? 44 : 32;
  syncBoardView(viewForBox(box, pad));
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  board.data.sort((a, b) => (a.placement ? 1 : 0) - (b.placement ? 1 : 0));

  let html = "";
  for (const h of board.data) {
    const isStone = Boolean(h.placement);
    const fill = isStone ? playerColor(h.placement.player) : "#101924";
    const stroke = isStone ? "#708296" : "#2c3d50";
    const opacity = isStone ? "1" : h.legal ? "0.86" : "0.62";
    const recentRank = recentPlacementRank(h.placement);
    const recentClass = recentRank === 1 ? "last" : recentRank === 2 ? "previous" : "";
    const cls = (h.legal && !isStone ? "cell legal" : "cell")
      + (recentRank ? ` recent-stone recent-${recentRank}` : "");
    html += `<path class="${cls}" d="${path(h.x, h.y, HEX - 1)}" fill="${fill}" stroke="${stroke}" stroke-width="1" opacity="${opacity}" data-q="${h.q}" data-r="${h.r}"></path>`;
    if (isStone && recentRank) {
      html += `<path class="last-move-outline ${recentClass}" d="${path(h.x, h.y, HEX - 0.5)}"></path>`;
    }
    if (isStone) html += `<text class="stone-label" x="${h.x}" y="${h.y}">${h.placement.index}</text>`;
  }
  svg.innerHTML = html;
  bindBoardEvents();
}

function buildCameraBox(shownPlacements, liveLegal, boardBounds) {
  const coords = [];
  const recent = shownPlacements.slice(-FIT_MOVE_COUNT);
  coords.push(...recent);

  const anchor = coords.length ? coords[coords.length - 1] : shownPlacements[shownPlacements.length - 1];
  if (anchor) {
    for (const cell of liveLegal.values()) {
      if (axialDistance(anchor, cell) <= FIT_LEGAL_RADIUS) coords.push(cell);
    }
  }

  if (!coords.length) return boardBounds;

  const focused = boxForCoords(coords, HEX * 8);
  const maxSpan = HEX * (window.innerWidth < 700 ? 34 : 48);
  if (focused.maxX - focused.minX <= maxSpan && focused.maxY - focused.minY <= maxSpan) return focused;

  const c = center(anchor.q, anchor.r);
  return {
    minX: c.x - maxSpan / 2,
    maxX: c.x + maxSpan / 2,
    minY: c.y - maxSpan / 2,
    maxY: c.y + maxSpan / 2,
  };
}

function boxForCoords(coords, pad) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const coord of coords) {
    const c = center(coord.q, coord.r);
    minX = Math.min(minX, c.x - pad);
    maxX = Math.max(maxX, c.x + pad);
    minY = Math.min(minY, c.y - pad);
    maxY = Math.max(maxY, c.y + pad);
  }
  return { minX, maxX, minY, maxY };
}

function axialDistance(a, b) {
  const dq = a.q - b.q;
  const dr = a.r - b.r;
  return Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr));
}

function bindBoardEvents() {
  svg.querySelectorAll(".cell").forEach(el => {
    el.addEventListener("mousemove", showTip);
    el.addEventListener("mouseleave", hideTip);
  });
}

function handleBoardClick(event) {
  if (event.target.closest(".board-view-controls") || event.target.closest(".legend")) return;
  if (suppressBoardClick || pendingRequest || !isLiveView()) return;
  const el = cellElementFromClick(event);
  if (!el) return;
  if (el.classList.contains("legal")) {
    if (!canSubmitMove()) {
      lastStatusError = isBotThinking() ? "Bot is thinking" : "Move submission is locked";
      renderMtStatus();
      renderTurnOverlay();
      return;
    }
    post("/api/move", { q: Number(el.dataset.q), r: Number(el.dataset.r) });
  }
}

function cellElementFromClick(event) {
  let el = event.target.closest(".cell");
  if (!el) {
    const hit = document.elementFromPoint(event.clientX, event.clientY);
    el = hit && hit.closest(".cell");
  }
  return el && svg.contains(el) ? el : null;
}

function bindBoardViewEvents() {
  boardArea.addEventListener("wheel", event => {
    if (!boardView || event.target.closest(".board-view-controls")) return;
    event.preventDefault();
    const factor = event.deltaY < 0 ? 0.88 : 1.14;
    zoomBoard(factor, clientToBoardPoint(event.clientX, event.clientY));
  }, { passive: false });

  boardArea.addEventListener("pointerdown", event => {
    if (!boardView || pendingRequest || (event.pointerType === "mouse" && event.button !== 0)) return;
    if (event.target.closest(".board-view-controls") || event.target.closest(".legend")) return;
    activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    boardArea.setPointerCapture(event.pointerId);
    boardArea.classList.add("dragging");
    hideTip();
    if (activePointers.size >= 2) {
      beginPinch(); // a second finger upgrades the gesture to pinch-zoom
    } else {
      beginPan(event);
    }
  });

  boardArea.addEventListener("pointermove", event => {
    if (!activePointers.has(event.pointerId)) return;
    event.preventDefault();
    activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pinchState && activePointers.size >= 2) {
      updatePinch();
    } else if (boardDrag && event.pointerId === boardDrag.pointerId) {
      updatePan(event);
    }
  }, { passive: false });

  boardArea.addEventListener("pointerup", endBoardPointer);
  boardArea.addEventListener("pointercancel", endBoardPointer);
}

function boardDragScales() {
  const rect = svg.getBoundingClientRect();
  return {
    scaleX: boardView.width / Math.max(1, rect.width),
    scaleY: boardView.height / Math.max(1, rect.height),
  };
}

function beginPan(event) {
  const { scaleX, scaleY } = boardDragScales();
  boardDrag = {
    pointerId: event.pointerId,
    clientX: event.clientX,
    clientY: event.clientY,
    scaleX,
    scaleY,
    view: { ...boardView },
    moved: false,
  };
}

function updatePan(event) {
  const dx = (event.clientX - boardDrag.clientX) * boardDrag.scaleX;
  const dy = (event.clientY - boardDrag.clientY) * boardDrag.scaleY;
  if (Math.hypot(event.clientX - boardDrag.clientX, event.clientY - boardDrag.clientY) > 4) boardDrag.moved = true;
  boardView = { ...boardDrag.view, x: boardDrag.view.x - dx, y: boardDrag.view.y - dy };
  boardViewDirty = true;
  applyBoardView();
}

function pinchPointers() {
  return [...activePointers.values()].slice(0, 2);
}

function beginPinch() {
  boardDrag = null; // pan and pinch are mutually exclusive
  const [a, b] = pinchPointers();
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  pinchState = {
    rect: svg.getBoundingClientRect(),
    startDist: Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)),
    startView: { ...boardView },
    anchorBoard: clientToBoardPoint(mid.x, mid.y),
  };
  suppressBoardClick = true;
}

// Two-finger pinch: zoom by the change in finger distance, keeping the board
// point that was under the initial midpoint pinned beneath the moving midpoint
// (so the gesture also pans).
function updatePinch() {
  const [a, b] = pinchPointers();
  if (!a || !b) return;
  const dist = Math.max(1, Math.hypot(a.x - b.x, a.y - b.y));
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  const base = boardBaseView || pinchState.startView;
  const nextWidth = clamp(pinchState.startView.width * (pinchState.startDist / dist), base.width * 0.14, base.width * 4.2);
  const scale = nextWidth / pinchState.startView.width;
  const nextHeight = pinchState.startView.height * scale;
  const rect = pinchState.rect;
  const viewScaleX = nextWidth / Math.max(1, rect.width);
  const viewScaleY = nextHeight / Math.max(1, rect.height);
  boardView = {
    width: nextWidth,
    height: nextHeight,
    x: pinchState.anchorBoard.x - (mid.x - rect.left) * viewScaleX,
    y: pinchState.anchorBoard.y - (mid.y - rect.top) * viewScaleY,
  };
  boardViewDirty = true;
  applyBoardView();
}

function endBoardPointer(event) {
  if (!activePointers.has(event.pointerId)) return;
  activePointers.delete(event.pointerId);
  if (boardArea.hasPointerCapture(event.pointerId)) boardArea.releasePointerCapture(event.pointerId);

  const moved = (boardDrag && boardDrag.moved) || Boolean(pinchState);
  if (activePointers.size < 2) pinchState = null;

  if (activePointers.size === 1) {
    // Dropped from pinch to a single finger — resume panning from it.
    const [pointerId, point] = [...activePointers.entries()][0];
    beginPan({ pointerId, clientX: point.x, clientY: point.y });
    boardDrag.moved = true;
  } else if (activePointers.size === 0) {
    boardDrag = null;
    boardArea.classList.remove("dragging");
    if (moved) {
      suppressBoardClick = true;
      window.setTimeout(() => { suppressBoardClick = false; }, 80);
    }
  }
}

function viewForBox(box, pad) {
  return {
    x: box.minX - pad,
    y: box.minY - pad,
    width: box.maxX - box.minX + pad * 2,
    height: box.maxY - box.minY + pad * 2,
  };
}

function syncBoardView(nextBase) {
  boardBaseView = nextBase;
  if (!boardView || !boardViewDirty) boardView = { ...nextBase };
  applyBoardView();
}

function applyBoardView() {
  if (!boardView) return;
  svg.setAttribute("viewBox", `${boardView.x} ${boardView.y} ${boardView.width} ${boardView.height}`);
}

function fitBoard() {
  boardViewDirty = false;
  if (boardBaseView) boardView = { ...boardBaseView };
  render();
}

function clearBoardView() {
  boardBaseView = null;
  boardView = null;
  boardViewDirty = false;
}

function zoomBoardAtCenter(factor) {
  if (!boardView) return;
  zoomBoard(factor, {
    x: boardView.x + boardView.width / 2,
    y: boardView.y + boardView.height / 2,
  });
}

function zoomBoard(factor, anchor) {
  if (!boardView) return;
  const base = boardBaseView || boardView;
  const nextWidth = clamp(boardView.width * factor, base.width * 0.14, base.width * 4.2);
  const scale = nextWidth / boardView.width;
  const nextHeight = boardView.height * scale;
  const point = anchor || {
    x: boardView.x + boardView.width / 2,
    y: boardView.y + boardView.height / 2,
  };
  boardView = {
    x: point.x - (point.x - boardView.x) * scale,
    y: point.y - (point.y - boardView.y) * scale,
    width: nextWidth,
    height: nextHeight,
  };
  boardViewDirty = true;
  applyBoardView();
}

function clientToBoardPoint(clientX, clientY) {
  const matrix = svg.getScreenCTM();
  if (!matrix || !boardView) {
    return {
      x: boardView ? boardView.x + boardView.width / 2 : 0,
      y: boardView ? boardView.y + boardView.height / 2 : 0,
    };
  }
  const point = svg.createSVGPoint();
  point.x = clientX;
  point.y = clientY;
  return point.matrixTransform(matrix.inverse());
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// ===========================================================================
// Match panels: status card, turn banner, move history, replay controls,
// players panel, insight chart, series panel, Open-in-Debug deep link,
// match keyboard shortcuts (mtHandleKey).
// ===========================================================================

// F2.5 — the status card (turn banner + facts chips) plus the header
// #statusText dot line (replaces renderStatus; the header chain is kept
// near-verbatim). Null-safe: the first render can run before /api/state has
// ever answered.
function renderMtStatus() {
  const statusText = document.getElementById("statusText");
  if (!state) {
    document.body.classList.remove("player0-turn", "player1-turn");
    renderMtTurnBanner(null);
    const facts = document.getElementById("mtFacts");
    if (facts) facts.innerHTML = "";
    if (statusText) statusText.textContent = lastStatusError || "Connecting";
    return;
  }
  const total = totalPlacements();
  const viewed = viewedPlacementCount();
  const live = isLiveView();
  const status = turnStatus();
  const active = state.winner || status === "terminal" || status === "stopped" ? null : state.current_player;
  document.body.classList.toggle("player0-turn", active === "player0");
  document.body.classList.toggle("player1-turn", active === "player1");

  renderMtTurnBanner(active);
  renderMtFacts();

  if (!statusText) return;
  if (lastStatusError) {
    statusText.textContent = lastStatusError;
  } else if (!live) {
    statusText.textContent = `Reviewing move ${viewed} / ${total}`;
  } else if (state.winner) {
    statusText.textContent = `${playerLabel(state.winner)} wins by six in line`;
  } else if (state.mode === "history") {
    const history = state.history || {};
    const suffix = history.status ? ` (${history.status})` : "";
    statusText.textContent = `Viewing ${state.game_id || "game history"}${suffix}`;
  } else if (status === "stopped") {
    statusText.textContent = "Match stopped";
  } else if (status === "bot_thinking") {
    statusText.textContent = `${playerShort(active)} ${playerKindLabel(active)} thinking`;
  } else if (status === "starting") {
    statusText.textContent = "Starting match";
  } else if (status === "error" || state.error) {
    statusText.textContent = state.error || "Match error";
  } else if (status === "terminal") {
    statusText.textContent = "Game complete";
  } else {
    statusText.textContent = `${playerShort(active)} ${playerKindLabel(active)} to place - ${placementStepLabel()}`;
  }
}

// Turn banner: to-play / thinking / winner / draw / stopped / error / starting.
// Unknown turn statuses fall through to the idle "to play" branch (§3.4: treat
// unknown statuses as idle).
function renderMtTurnBanner(active) {
  const banner = document.getElementById("mtTurnBanner");
  const title = document.getElementById("mtTurnTitle");
  const sub = document.getElementById("mtTurnSub");
  if (!banner || !title || !sub) return;

  const winner = state ? state.winner : null;
  banner.classList.toggle("p0", active === "player0" || winner === "player0");
  banner.classList.toggle("p1", active === "player1" || winner === "player1");

  if (!state) {
    title.textContent = "Loading";
    sub.textContent = "Waiting for state";
    return;
  }
  const status = turnStatus();
  if (winner) {
    title.textContent = `${playerLabel(winner)} wins`;
    sub.textContent = state.terminal_reason || "Game complete";
    return;
  }
  if (status === "stopped") {
    title.textContent = "Match stopped";
    sub.textContent = "Start a new match to continue";
    return;
  }
  if (status === "error") {
    title.textContent = "Match error";
    sub.textContent = state.error || "See server logs";
    return;
  }
  if (status === "terminal") {
    title.textContent = "Draw";
    sub.textContent = state.terminal_reason || "Game complete";
    return;
  }
  if (status === "bot_thinking") {
    const thinking = state.thinking_player || botPlayer();
    title.textContent = `${playerShort(thinking)} thinking`;
    sub.textContent = `${playerKindLabel(thinking)} is choosing the next placement`;
    return;
  }
  if (status === "starting") {
    title.textContent = "Starting match";
    sub.textContent = "Preparing players";
    return;
  }
  title.textContent = `${playerShort(active)} to play`;
  sub.textContent = `${playerKindLabel(active)} - ${placementStepLabel()}`;
}

// Facts chips (§4 Region 3): move, phase, stones, legal count, game id, seed.
function renderMtFacts() {
  const facts = document.getElementById("mtFacts");
  if (!facts) return;
  const done = Boolean(state.winner) || turnStatus() === "terminal";
  const config = state.match || {};
  const seed = config.seed === null || config.seed === undefined ? "auto" : String(config.seed);
  const chips = [
    ["Move", done ? "Complete" : placementStepLabel()],
    ["Phase", done ? "Complete" : phaseLabel(state.phase)],
    ["Stones", String(totalPlacements())],
    ["Legal", String(state.legal_count ?? (state.legal || []).length)],
    ["Game", String(state.game_id || "game")],
    ["Seed", seed],
  ];
  facts.innerHTML = chips.map(([label, value]) =>
    `<span class="mt-chip"><span class="mt-chip-label">${escapeText(label)}</span>${escapeText(value)}</span>`
  ).join("");
}

// The move list is a bounded, wrapping, vertically-scrolling box. Chips flow and
// wrap inside it, so the element is always exactly its container's width and can
// never push the page sideways — regardless of move count (a 1000-move game just
// scrolls vertically). Chips are only rebuilt when the move list grows, so live
// polling of a long game doesn't churn ~1000 DOM nodes every tick.
let moveHistoryBound = false;
let moveHistoryStructSig = "";

function renderMoveHistory() {
  const history = document.getElementById("moveHistory");
  if (!history) return;
  ensureMoveHistoryEvents(history);
  const placements = (state && state.placements) || [];
  const selected = viewedPlacementCount();

  if (!placements.length) {
    moveHistoryStructSig = "";
    history.classList.remove("has-moves");
    history.innerHTML = `<div class="empty-list">No moves yet</div>`;
    return;
  }

  history.classList.add("has-moves");
  // game_id is part of the rebuild signature (§8): a series can advance to a
  // new game whose placement count momentarily matches the previous game's.
  const structSig = `${(state && state.game_id) || ""}:${placements.length}:${placements[placements.length - 1].index}`;
  if (structSig !== moveHistoryStructSig) {
    moveHistoryStructSig = structSig;
    history.innerHTML = placements.map(p => {
      const cls = p.player === "player0" ? "p0" : "p1";
      return `<button class="history-chip ${cls}" data-move-index="${p.index}">
      <span class="chip-index">${p.index}</span>
      <span class="chip-dot"></span>
      <span class="chip-text">${playerShort(p.player)} (${p.q}, ${p.r})</span>
    </button>`;
    }).join("");
  }

  // Update the selection without a full rebuild, then scroll it into view.
  const previous = history.querySelector(".history-chip.selected");
  if (previous) previous.classList.remove("selected");
  const current = history.querySelector(`.history-chip[data-move-index="${selected}"]`);
  if (current) {
    current.classList.add("selected");
    if (history.clientHeight) {
      const top = current.offsetTop - history.clientHeight / 2 + current.clientHeight / 2;
      history.scrollTop = Math.max(0, top);
    }
  }
}

function ensureMoveHistoryEvents(history) {
  if (moveHistoryBound) return;
  moveHistoryBound = true;
  // Delegated click so chip rebuilds never leave stale per-chip listeners.
  history.addEventListener("click", event => {
    const chip = event.target.closest("[data-move-index]");
    if (!chip) return;
    setReplayIndex(Number(chip.dataset.moveIndex));
  });
}

function renderReplay() {
  const total = totalPlacements();
  const viewed = viewedPlacementCount();
  const slider = document.getElementById("replaySlider");
  slider.max = String(total);
  slider.value = String(viewed);
  document.getElementById("replayLabel").textContent = `${viewed} / ${total}`;
  document.getElementById("replaySub").textContent = replaySubtitle(viewed);
  document.getElementById("replayMidTick").textContent = String(Math.floor(total / 2));
  document.getElementById("replayMaxTick").textContent = String(total);
  document.getElementById("replayPlayBtn").textContent = replayTimer ? "Pause" : "Play";
}

// F2.5 — board overlay for bot-thinking / starting, rewritten for the new
// player model (checkpoint labels come from the payload via playerKindLabel).
// Ids #turnOverlay* are kept (board subtree retained byte-for-byte).
function renderTurnOverlay() {
  const overlay = document.getElementById("turnOverlay");
  if (!overlay) return;
  const title = document.getElementById("turnOverlayTitle");
  const sub = document.getElementById("turnOverlaySub");
  const show = Boolean(state) && isLiveView() && (isBotThinking() || turnStatus() === "starting");
  overlay.hidden = !show;
  if (!show || !title || !sub) return;
  const thinking = state.thinking_player || botPlayer();
  title.textContent = isBotThinking() ? `${playerShort(thinking)} thinking` : "Starting match";
  sub.textContent = isBotThinking()
    ? `${playerKindLabel(thinking)} is choosing the next placement`
    : "Preparing players";
}

// ---- F3.1 — players panel ---------------------------------------------------

// The current game's server-side decision log (§3.5): entries are
// {ply, player, q?, r?, duration_ms, value, visits, kind} on success,
// {ply, player, error, kind} on failure. value is the searched/analyzed
// root_value in SIDE-TO-MOVE perspective (the mover's own view).
function mtBotDecisions() {
  return state && Array.isArray(state.bot_decisions) ? state.bot_decisions : [];
}

// Number(null) is 0 — these guards keep absent fields (null root_value on
// sealbot decisions, null ply) from rendering as fake zeros.
function mtFiniteOrNull(raw) {
  if (raw === null || raw === undefined || raw === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function mtFormatDuration(ms) {
  const value = mtFiniteOrNull(ms);
  if (value === null) return "";
  if (value >= 10000) return `${(value / 1000).toFixed(1)}s`;
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${Math.round(value)} ms`;
}

function mtFormatValue(value) {
  const v = mtFiniteOrNull(value);
  if (v === null) return "";
  return `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;
}

// Kind chip text: checkpoint seats carry their search config; payload meta
// wins, the setup-strip config is the pre-state fallback (same precedence as
// playerKind/playerKindLabel).
function mtSeatKindChip(seat) {
  const meta = playerMeta(seat) || {};
  const kind = playerKind(seat);
  if (kind === "checkpoint") {
    const cfg = mtSeatCfg(seat);
    const mode = meta.mode || cfg.mode || "search";
    const visits = firstFinite(meta.visits, cfg.visits, 256);
    return mode === "policy" ? "checkpoint · policy" : `checkpoint · ${mode} · ${visits}v`;
  }
  if (kind.startsWith("sealbot-")) return `sealbot ${kind.slice("sealbot-".length)}`;
  return kind;
}

// Last finished decision for a seat (the log is append-ordered by ply).
function mtSeatLastDecision(seat, decisions) {
  for (let i = decisions.length - 1; i >= 0; i--) {
    const entry = decisions[i];
    if (entry && entry.player === seat && !entry.error) return entry;
  }
  return null;
}

// Seat error line: a failed decision logged for the seat, else the match-level
// payload error when it names this seat.
function mtSeatError(seat, decisions) {
  for (let i = decisions.length - 1; i >= 0; i--) {
    const entry = decisions[i];
    if (entry && entry.player === seat && entry.error) return String(entry.error);
  }
  const err = state && state.error ? String(state.error) : "";
  if (err && (err.includes(seat) || err.includes(playerShort(seat)))) return err;
  return "";
}

function renderMtPlayers() {
  const el = document.getElementById("mtPlayers");
  if (!el) return;
  const decisions = mtBotDecisions();
  el.innerHTML = ["player0", "player1"].map(seat => {
    const thinking = Boolean(state) && state.thinking_player === seat;
    const yourTurn = playerKind(seat) === "manual"
      && Boolean(state) && state.current_player === seat && canSubmitMove();
    const last = mtSeatLastDecision(seat, decisions);
    const error = mtSeatError(seat, decisions);
    const parts = [
      `<span class="mt-seat-dot" style="background:${playerColor(seat)}"></span>`,
      `<span class="mt-player-name">${escapeText(`${playerShort(seat)} ${playerKindLabel(seat)}`)}</span>`,
      `<span class="mt-chip mt-kind-chip">${escapeText(mtSeatKindChip(seat))}</span>`,
    ];
    if (thinking) parts.push(`<span class="mt-thinking-dot" title="thinking"></span>`);
    if (yourTurn) parts.push(`<span class="mt-your-turn">your turn</span>`);
    if (last) {
      const q = mtFiniteOrNull(last.q);
      const r = mtFiniteOrNull(last.r);
      const move = q !== null && r !== null ? `(${q}, ${r})` : "—";
      const bits = [`<span>last ${escapeText(move)}</span>`];
      const duration = mtFormatDuration(last.duration_ms);
      if (duration) bits.push(`<span>${escapeText(duration)}</span>`);
      const value = mtFormatValue(last.value);
      if (value) bits.push(`<span class="mt-chip mt-value-chip">${escapeText(value)}</span>`);
      parts.push(`<div class="mt-player-last">${bits.join("")}</div>`);
    }
    if (error) parts.push(`<div class="mt-player-err">${escapeText(error)}</div>`);
    return `<div class="mt-player-row" data-seat="${seat}">${parts.join("")}</div>`;
  }).join("");
}

// ---- F3.2 — value sparkline over bot_decisions ------------------------------

const MT_CHART_W = 300;
const MT_CHART_H = 90;
const MT_CHART_X0 = 30;
const MT_CHART_X1 = 292;
const MT_CHART_Y0 = 10;
const MT_CHART_Y1 = 80;

let mtChartSig = "";       // rebuild signature: innerHTML only when the data moved
let mtChartGeom = null;    // {points:[{ply,seat,v,x,y}]} for hover/click hit-testing

// Chart series: decision-log entries with a finite value, mapped to the P0
// perspective (root_value is side-to-move, so P1 entries flip sign).
function mtChartEntries() {
  return mtBotDecisions()
    .map(entry => ({
      ply: mtFiniteOrNull(entry && entry.ply),
      seat: entry && entry.player,
      value: mtFiniteOrNull(entry && entry.value),
    }))
    .filter(entry => entry.ply !== null && entry.value !== null
      && (entry.seat === "player0" || entry.seat === "player1"))
    .map(entry => ({ ply: entry.ply, seat: entry.seat, v: entry.seat === "player0" ? entry.value : -entry.value }));
}

function mtChartSvg(entries) {
  const plies = entries.map(entry => entry.ply);
  const minPly = Math.min(...plies);
  const span = Math.max(1, Math.max(...plies) - minPly);
  const xAt = ply => MT_CHART_X0 + ((ply - minPly) / span) * (MT_CHART_X1 - MT_CHART_X0);
  const yAt = v => {
    const c = Math.max(-1, Math.min(1, v));
    return MT_CHART_Y1 - ((c + 1) / 2) * (MT_CHART_Y1 - MT_CHART_Y0);
  };
  const points = entries
    .map(entry => ({ ply: entry.ply, seat: entry.seat, v: entry.v, x: xAt(entry.ply), y: yAt(entry.v) }))
    .sort((a, b) => a.x - b.x);
  mtChartGeom = { points };
  const zeroY = yAt(0).toFixed(1);
  const parts = [
    `<line class="mt-chart-zero" x1="${MT_CHART_X0}" x2="${MT_CHART_X1}" y1="${zeroY}" y2="${zeroY}"></line>`,
    `<text class="mt-chart-axis" x="4" y="${MT_CHART_Y0 + 3}">+1</text>`,
    `<text class="mt-chart-axis" x="4" y="${Number(zeroY) + 3}">0</text>`,
    `<text class="mt-chart-axis" x="4" y="${MT_CHART_Y1 + 2}">−1</text>`,
  ];
  for (const seat of ["player0", "player1"]) {
    const pts = points.filter(p => p.seat === seat);
    if (!pts.length) continue;
    const cls = seat === "player0" ? "p0" : "p1";
    if (pts.length === 1) {
      parts.push(`<circle class="mt-chart-dot ${cls}" cx="${pts[0].x.toFixed(1)}" cy="${pts[0].y.toFixed(1)}" r="2.4"></circle>`);
    } else {
      const coords = pts.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
      parts.push(`<polyline class="mt-chart-line ${cls}" fill="none" points="${coords}"></polyline>`);
    }
  }
  const latest = points[points.length - 1];
  parts.push(`<circle class="mt-chart-dot ${latest.seat === "player0" ? "p0" : "p1"}" cx="${latest.x.toFixed(1)}" cy="${latest.y.toFixed(1)}" r="3"></circle>`);
  // Hover layer — mousemove updates these two elements only (attribute/text
  // writes), never re-rendering the svg.
  parts.push(`<line id="mtChartCross" class="mt-chart-cross" x1="-10" x2="-10" y1="${MT_CHART_Y0}" y2="${MT_CHART_Y1}" visibility="hidden"></line>`);
  parts.push(`<text id="mtChartHoverText" class="mt-chart-hover" x="${MT_CHART_X1}" y="${MT_CHART_Y0 + 3}" text-anchor="end" visibility="hidden"></text>`);
  return `<svg viewBox="0 0 ${MT_CHART_W} ${MT_CHART_H}" role="img" aria-label="Bot value by ply (P0 perspective)">${parts.join("")}</svg>`;
}

function renderMtInsight() {
  const el = document.getElementById("mtValueChart");
  if (!el) return;
  const button = document.getElementById("mtOpenDebugBtn");
  if (button) {
    // F3.4: disabled when there is no position to export or no run to open it in.
    button.disabled = !state || viewedPlacementCount() === 0
      || !(mtFirstCheckpointSeat() || trainingRuns.length);
  }
  const entries = mtChartEntries();
  const sig = `${(state && state.game_id) || ""}:${entries.length}:${entries.length ? entries[entries.length - 1].ply : ""}`;
  if (sig === mtChartSig) return;
  mtChartSig = sig;
  if (entries.length < 2) {
    mtChartGeom = null;
    el.innerHTML = `<div class="dbg-empty-note">No bot evaluations yet</div>`;
    return;
  }
  el.innerHTML = mtChartSvg(entries);
}

// Nearest-ply hit test in viewBox coordinates (the svg is width:100% scaled).
function mtChartPointAt(event) {
  if (!mtChartGeom || !mtChartGeom.points.length) return null;
  const wrap = document.getElementById("mtValueChart");
  const svgEl = wrap && wrap.querySelector("svg");
  if (!svgEl) return null;
  const rect = svgEl.getBoundingClientRect();
  if (!rect.width) return null;
  const fx = ((event.clientX - rect.left) / rect.width) * MT_CHART_W;
  let best = null;
  for (const p of mtChartGeom.points) {
    if (!best || Math.abs(p.x - fx) < Math.abs(best.x - fx)) best = p;
  }
  return best;
}

function mtChartHover(event) {
  const cross = document.getElementById("mtChartCross");
  const label = document.getElementById("mtChartHoverText");
  if (!cross || !label) return;
  const p = mtChartPointAt(event);
  if (!p) return;
  const x = p.x.toFixed(1);
  cross.setAttribute("x1", x);
  cross.setAttribute("x2", x);
  cross.setAttribute("visibility", "visible");
  label.textContent = `ply ${p.ply} · ${playerShort(p.seat)} ${mtFormatValue(p.v)}`;
  label.setAttribute("visibility", "visible");
}

function mtChartHoverEnd() {
  const cross = document.getElementById("mtChartCross");
  const label = document.getElementById("mtChartHoverText");
  if (cross) cross.setAttribute("visibility", "hidden");
  if (label) label.setAttribute("visibility", "hidden");
}

function mtChartClick(event) {
  const p = mtChartPointAt(event);
  if (p) setReplayIndex(p.ply);  // scrub the board to the position the bot evaluated
}

// ---- F3.3 — series panel ----------------------------------------------------

function renderMtSeries() {
  const card = document.getElementById("mtSeriesCard");
  const el = document.getElementById("mtSeries");
  if (!card || !el) return;
  const series = state && state.series;
  card.hidden = !series;
  if (!series) {
    el.innerHTML = "";
    return;
  }
  const tally = series.tally || {};
  const slots = series.slots || {};
  const slotLabel = slot => (slots[slot] && slots[slot].label) || slot;
  const draws = Number(tally.draws) || 0;
  const tallyHtml = `<div class="mt-series-tally">`
    + `<span class="mt-slot0">${escapeText(slotLabel("slot0"))}</span>`
    + ` ${escapeText(String(tally.slot0 ?? 0))} — ${escapeText(String(tally.slot1 ?? 0))} `
    + `<span class="mt-slot1">${escapeText(slotLabel("slot1"))}</span>`
    + (draws > 0 ? ` · ${escapeText(String(draws))} draw${draws === 1 ? "" : "s"}` : "")
    + `</div>`;
  const swapped = series.seats && series.seats.player0 === "slot1";
  const progress = series.finished
    ? `Series finished · ${series.played ?? series.games ?? "?"} games`
    : `Game ${series.current_game ?? "?"} / ${series.games ?? "?"}${swapped ? " (seats swapped)" : ""}`;
  const chips = (series.results || []).map(result => {
    const winner = result.winner_slot === "slot0" || result.winner_slot === "slot1" ? result.winner_slot : null;
    const title = `Game ${result.game}: ${winner ? `${slotLabel(winner)} wins` : "draw"} · ${result.length} moves`;
    return `<span class="mt-series-chip ${winner || "draw"}" title="${escapeAttr(title)}">${winner ? "✓" : "="}</span>`;
  }).join("");
  el.innerHTML = tallyHtml
    + `<div class="mt-series-progress">${escapeText(progress)}</div>`
    + (chips ? `<div class="mt-series-games">${chips}</div>` : "");
}

// ---- F3.4 — Open in Debug deep link -----------------------------------------

// First checkpoint seat of the RUNNING match (payload meta carries run +
// checkpoint per §3.1) — its model is preselected in the debug screen.
function mtFirstCheckpointSeat() {
  for (const seat of ["player0", "player1"]) {
    const meta = playerMeta(seat);
    if (meta && meta.kind === "checkpoint" && meta.run) return meta;
  }
  return null;
}

// dbgApplyNav routes any nav whose path is empty (or not in the target run's
// game list) through dbgAutoPickGame, which re-navigates with acts:[] and
// would wipe the imported position. So resolve a concrete loadable game path
// for the run BEFORE handing the nav to the debug module. The acts-branch
// position fetch only uses the path for dbgEnsureRecordedActs (record=0,
// ply=0 — the recorded prefix is sliced to length 0), so any record that
// loads at ply 0 works.
async function mtResolveDebugGame(run) {
  if (dbg.nav.run === run && dbg.nav.path) {
    // Warm same-run: keep the already-loaded game (and its src/record).
    return { path: dbg.nav.path, src: dbg.nav.src, rec: dbg.nav.rec };
  }
  const fallback = { path: "", src: "selfplay", rec: 0 };
  let games = [];
  try {
    const res = await fetch(`/api/debug/games?run=${encodeURIComponent(run)}&source=selfplay`);
    if (!res.ok) return fallback;
    const data = await safeJson(res);
    games = (data && data.games) || [];
  } catch (_e) {
    return fallback;
  }
  // Newest-first probe, same reason as dbgAutoPickGame: the newest selfplay
  // file is usually the in-progress epoch with no loadable record yet.
  for (const g of games.slice(0, 6)) {
    try {
      const params = new URLSearchParams({ run, path: g.path, record: "0", ply: "0" });
      const res = await fetch(`/api/debug/position?${params.toString()}`);
      if (res.ok) return { path: g.path, src: "selfplay", rec: 0 };
    } catch (_e) { /* probe the next (older) file */ }
  }
  return fallback;
}

let mtOpenDebugBusy = false;

async function mtOpenDebug() {
  if (!state || mtOpenDebugBusy) return;
  const acts = (state.placements || [])
    .slice(0, viewedPlacementCount())
    .map(p => dbgPackActionId(p.q, p.r));
  const ckptSeat = mtFirstCheckpointSeat();
  const run = (ckptSeat && ckptSeat.run) || (trainingRuns[0] && trainingRuns[0].name) || "";
  const ckpt = (ckptSeat && ckptSeat.checkpoint) || "";
  if (!acts.length || !run) return;
  mtOpenDebugBusy = true;
  try {
    // Switch screens FIRST (snappy, like debugOpenFromHistory). setScreen's
    // replaceState strips any query off the hash, so the full nav hash must be
    // written AFTER this — dbgNavigate below fires its own hashchange, which
    // the global listener routes with preserveHash.
    navigateScreen("debug");
    const game = await mtResolveDebugGame(run);
    // acts is the debug nav's branch-injection list; its position fetch sends
    // recorded[0..ply] + acts, so ply MUST be 0 for the imported list to be
    // the WHOLE position (ply=acts.length would prepend acts.length moves of
    // whatever recorded game the path points at).
    dbgNavigate({ run, src: game.src, path: game.path, rec: game.rec, ply: 0, acts, ckptA: ckpt || "" });
  } finally {
    mtOpenDebugBusy = false;
  }
}

// ---- F3.5 — match-screen keyboard shortcuts ----------------------------------

// Guard shape mirrors histHandleKey/dbgHandleKey: only on the match screen,
// never while typing in a form control, never on modified chords. Every
// shortcut has a button equivalent (replay bar / Start button).
function mtHandleKey(e) {
  if (activeScreen !== "match") return;
  const ae = document.activeElement;
  const tag = ae && ae.tagName;
  if (tag && /^(INPUT|SELECT|TEXTAREA)$/.test(tag)) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key;
  if (k === " " && tag === "BUTTON") return;  // Space activates the focused button
  let handled = true;
  if (k === "ArrowLeft") setReplayIndex(viewedPlacementCount() - 1);
  else if (k === "ArrowRight") setReplayIndex(viewedPlacementCount() + 1);
  else if (k === "Home") setReplayIndex(0);
  else if (k === "End") setReplayIndex(totalPlacements());
  else if (k === " ") toggleReplayPlay();
  else if (k === "n" || k === "N") mtStartMatch();
  else handled = false;
  if (handled) e.preventDefault();
}

// ===========================================================================
// Shared state/format helpers: turn status, player metadata/labels, replay
// position state, board tooltips/HUD, text escaping, value formatting.
// ===========================================================================

function canSubmitMove() {
  if (!state || pendingRequest || !isLiveView() || state.winner || state.stopped) return false;
  const status = turnStatus();
  if (status === "terminal" || status === "stopped") return false;
  if (typeof state.can_submit === "boolean") return state.can_submit;
  return playerKind(state.current_player) === "manual";
}

function turnStatus() {
  if (!state) return "starting";
  if (state.stopped || state.turn_status === "stopped") return "stopped";
  if (state.error) return "error";
  if (state.winner || state.turn_status === "terminal") return "terminal";
  // Fallback for payloads without turn_status: any non-manual seat means a bot
  // match (was isSealBotMatch(); checkpoint seats count as bots too now).
  const anyBot = state.mode !== "manual"
    && ["player0", "player1"].some(player => playerKind(player) !== "manual");
  return state.turn_status || (anyBot ? "human_turn" : "manual_turn");
}

function isBotThinking() {
  return turnStatus() === "bot_thinking";
}

// Any non-manual seat is a bot now (sealbot OR checkpoint).
function botPlayer() {
  if (state && state.thinking_player) return state.thinking_player;
  return ["player0", "player1"].find(player => playerKind(player) !== "manual") || null;
}

function playerMeta(player) {
  if (!state || !state.players) return null;
  if (Array.isArray(state.players)) {
    return state.players.find(item => item.role === player || item.player === player || item.id === player) || null;
  }
  const item = state.players[player];
  if (typeof item === "string") return { role: player, kind: item, label: PLAYER_KIND_LABELS[item] || item };
  if (item && typeof item === "object") return item;
  return null;
}

function playerKind(player) {
  const meta = playerMeta(player);
  if (meta) {
    if (typeof meta.kind === "string") return normalizePlayerKind(meta.kind, meta.variant);
    if (typeof meta.variant === "string") return `sealbot-${meta.variant}`;
  }
  // Pre-state fallback: the seat's configured kind from the setup strip state.
  const cfg = mtSeatCfg(player === "player1" ? "player1" : "player0");
  if (cfg.kind === "sealbot") return `sealbot-${cfg.variant || "current"}`;
  return cfg.kind || "manual";
}

function normalizePlayerKind(kind, variant = "") {
  if (kind === "human") return "manual";
  if (kind === "bot" || kind === "sealbot") return `sealbot-${variant || "current"}`;
  return kind || "manual";
}

// Prefer the server-provided label (checkpoint seats get "<run> @ <ckpt>",
// sealbot seats "SealBot <variant>"); fall back to the static kind map.
function playerKindLabel(player) {
  const meta = playerMeta(player);
  if (meta && typeof meta.label === "string" && meta.label) return meta.label;
  const kind = playerKind(player);
  return PLAYER_KIND_LABELS[kind] || kind;
}

function playerLabel(player) {
  if (!player) return "--";
  const slot = player === "player0" ? "P0" : player === "player1" ? "P1" : player;
  return `${slot} ${playerKindLabel(player)}`;
}

function playerShort(player) {
  if (player === "player0") return "P0";
  if (player === "player1") return "P1";
  return "--";
}

function firstFinite(...values) {
  for (const value of values) {
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }
  return NaN;
}

function playerColor(player) {
  return player === "player0" ? "var(--p0)" : "var(--p1)";
}

function phaseLabel(phase) {
  if (phase === "opening") return "Opening";
  if (phase === "first_stone") return "First stone";
  return "Second stone";
}

function center(q, r) {
  return { x: HEX * SQRT3 * (q + r / 2), y: HEX * 1.5 * r };
}

function path(cx, cy, size) {
  let d = "";
  for (let i = 0; i < 6; i++) {
    const angle = Math.PI / 180 * (60 * i - 30);
    const x = cx + size * Math.cos(angle);
    const y = cy + size * Math.sin(angle);
    d += (i === 0 ? "M" : "L") + x.toFixed(2) + "," + y.toFixed(2);
  }
  return d + "Z";
}

function visiblePlacements() {
  return (state.placements || []).slice(0, viewedPlacementCount());
}

function lastVisiblePlacement(offset = 0) {
  const placements = visiblePlacements();
  return placements[placements.length - 1 - offset] || null;
}

function recentPlacementRank(placement) {
  if (!placement) return 0;
  if (samePlacement(placement, lastVisiblePlacement(0))) return 1;
  if (samePlacement(placement, lastVisiblePlacement(1))) return 2;
  return 0;
}

function samePlacement(a, b) {
  return Boolean(a && b && a.index === b.index);
}

function totalPlacements() {
  return state ? (state.placements || []).length : 0;
}

function stateVersion() {
  const version = Number(state && state.version);
  return Number.isFinite(version) ? version : null;
}

function viewedPlacementCount() {
  const total = totalPlacements();
  if (replayIndex === null) return total;
  return Math.max(0, Math.min(replayIndex, total));
}

function isLiveView() {
  return replayIndex === null || viewedPlacementCount() === totalPlacements();
}

function setReplayIndex(index) {
  stopReplay();
  const total = totalPlacements();
  replayIndex = Math.max(0, Math.min(index, total));
  if (replayIndex === total) replayIndex = null;
  render();
}

function resetReplay() {
  stopReplay();
  replayIndex = null;
}

function toggleReplayPlay() {
  const total = totalPlacements();
  if (!total) return;
  if (replayTimer) {
    stopReplay(true);
    return;
  }
  if (viewedPlacementCount() >= total) replayIndex = 0;
  replayTimer = window.setInterval(() => {
    const next = viewedPlacementCount() + 1;
    if (next >= total) {
      replayIndex = null;
      stopReplay();
    } else {
      replayIndex = next;
    }
    render();
  }, 520);
  render();
}

function stopReplay(renderAfter = false) {
  if (replayTimer) {
    window.clearInterval(replayTimer);
    replayTimer = null;
    if (renderAfter) render();
  }
}

function replaySubtitle(viewed) {
  if (!viewed) return "Opening";
  const placement = (state.placements || [])[viewed - 1];
  if (!placement) return "Live";
  return `${phaseLabel(placement.phase)} - ${playerShort(placement.player)} (${placement.q}, ${placement.r})`;
}

function cellInfo(key) {
  const [q, r] = key.split(",").map(Number);
  const owner = visiblePlacements().find(p => p.q === q && p.r === r);
  const legal = isLiveView() && (state.legal || []).some(c => c.q === q && c.r === r);
  return {
    q,
    r,
    legal,
    owner: owner && owner.player,
    index: owner && owner.index,
  };
}

function showTip(event) {
  if (boardDrag) {
    hideTip();
    return;
  }
  tip.style.display = "block";
  tip.style.left = event.offsetX + 12 + "px";
  tip.style.top = event.offsetY + 12 + "px";
  const key = `${event.target.dataset.q},${event.target.dataset.r}`;
  const info = cellInfo(key);
  updateHud(info);
  tip.textContent = `(${info.q}, ${info.r}) - ${cellStateLabel(info)}`;
}

function hideTip() {
  tip.style.display = "none";
}

function updateHud(info) {
  if (!info) return;
  cellHud.innerHTML = `
    <div><span>Q:</span> <strong>${info.q}</strong> <span>R:</span> <strong>${info.r}</strong></div>
    <div>Cell: ${escapeText(cellStateLabel(info))}</div>
  `;
}

function cellStateLabel(info) {
  if (info.owner) return `${playerShort(info.owner)} stone ${info.index}`;
  if (info.legal) return "legal";
  return "empty";
}

function escapeText(text) {
  return String(text).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function escapeAttr(text) {
  return escapeText(text);
}

// Human-friendly stringification for raw values interpolated into the UI:
// null/undefined/"" become an em dash rather than the literal "null"/"undefined",
// finite numbers pass through, and objects/arrays are JSON-encoded compactly.
function displayValue(value, empty = "—") {
  if (value === null || value === undefined || value === "") return empty;
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : empty;
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch (error) {
      return empty;
    }
  }
  return String(value);
}

function placementStepLabel() {
  if (!state || state.phase === "opening") return "Opening";
  return state.phase === "second_stone" ? "Placement 2 of 2" : "Placement 1 of 2";
}

// ===========================================================================
// History screen v2 (hist*) rendering: status band (+ live patching), trends
// charts (histChartSvg + hover crosshair), epoch table, epoch inspector
// (lazy /api/training/epoch + /api/debug/ckpt_info), paged game list with
// thumbnails, and the per-game detail card.
// ===========================================================================

// Clears every v2 status/trend/epoch container (error + empty branches). The
// drawer keeps its hidden attribute so an empty drawer never reserves layout.
function clearHistPanels() {
  if (histStatusBand) histStatusBand.innerHTML = "";
  if (histHealthDrawer) {
    histHealthDrawer.innerHTML = "";
    histHealthDrawer.hidden = true;
  }
  if (histTrends) histTrends.innerHTML = "";
  if (histStrength) {
    histStrength.innerHTML = "";
    histStrength.hidden = true;
  }
  if (histRatingTable) {
    histRatingTable.innerHTML = "";
    histRatingTable.hidden = true;
  }
  if (histEpochTable) histEpochTable.innerHTML = "";
  // histInspectEpoch survives (the run may simply not be re-fetched yet);
  // only the panel is hidden until data is back.
  if (histEpochInspector) {
    histEpochInspector.innerHTML = "";
    histEpochInspector.hidden = true;
  }
  histTrendCharts = {};
  histTrendHoverSvg = null;
  if (histTrendTipEl) histTrendTipEl.hidden = true;
  // No displayed list in the error/no-runs branches — game nav goes inert.
  histDisplayedKeys = [];
}

function renderGameHistoryPage() {
  if (!histStatusBand || !gameHistoryList || !gameHistoryDetail) return;
  syncHistoryRunSelect(historySelectedRun);
  // The epoch chip lives in .history-filters (outside the cleared panels) so
  // it stays clearable in every branch, including error/no-runs.
  renderHistEpochChip();
  if (trainingLoadError) {
    clearHistPanels();
    gameHistoryList.innerHTML = `<div class="empty-list">${escapeText(trainingLoadError)}</div>`;
    gameHistoryDetail.innerHTML = `<div class="empty-list">No game selected</div>`;
    return;
  }
  const runs = historyRunsForPage();
  const histories = historyItemsForPage(runs);
  if (!runs.length) {
    clearHistPanels();
    const pendingSelection = historyDetailsLoading || historySelectionPendingDetails();
    gameHistoryList.innerHTML = `<div class="empty-list">${pendingSelection ? "Loading game histories" : "No training run selected"}</div>`;
    gameHistoryDetail.innerHTML = `<div class="empty-list">No game selected</div>`;
    return;
  }

  const usingServerPage = historyPage.loaded || historyPage.loading || historyPage.items.length > 0;
  const filtered = usingServerPage ? histories : sortedHistoryItems(filteredHistoryItems(histories));
  // H7: client-side, display-only epoch filter (the server pager has no epoch
  // param — newly loaded pages re-apply the filter on the next render).
  const displayed = histEpochFilter === null
    ? filtered
    : filtered.filter(item => asFinite(item.epoch) === histEpochFilter);
  // P2.1: prev/next + arrow keys walk exactly what the list shows. When the
  // selection falls back to the unfiltered histories (displayed empty), the
  // selected key is absent from this order and nav renders disabled.
  histDisplayedKeys = displayed.map(item => historyItemKey(item));
  const selected = selectedHistoryItem(histories, displayed);
  const visible = usingServerPage ? displayed : displayed.slice(0, historyVisibleLimit);
  renderHistStatusBand(runs);
  // NOTE: renderHistTrends() resets histTrendCharts {} at its start, so the
  // strength panel (which also registers chart geometry there for hover) MUST
  // render AFTER trends, or its registrations get wiped.
  renderHistTrends(runs);
  renderHistStrength(runs);
  renderHistEpochTable(runs);
  renderHistEpochInspector(runs);
  // Length micro-bars are relative to the longest currently displayed row.
  const maxLen = visible.reduce((max, item) => Math.max(max, historyLength(item)), 0);
  const listParts = [];
  if (visible.length) {
    listParts.push(...visible.map(item => gameHistoryListRow(item.run, item, maxLen)));
  } else {
    listParts.push(`<div class="empty-list">${historyPage.loading
      ? "Loading game histories"
      : histEpochFilter !== null
        ? `No loaded games for epoch ${histEpochFilter} — clear the epoch chip or load more`
        : "No games match the current filters"}</div>`);
  }
  // "Load more" keeps paging unfiltered while the epoch filter is on, so it
  // stays visible even when the filter empties the displayed slice.
  if (usingServerPage && historyPage.nextCursor && (visible.length || histEpochFilter !== null)) {
    listParts.push(`<button class="history-list-more" type="button" data-history-more>${historyPage.loading ? "Loading games" : `Load more games (${filtered.length} loaded)`}</button>`);
  } else if (!usingServerPage && displayed.length > visible.length) {
    listParts.push(`<button class="history-list-more" type="button" data-history-more>Show ${Math.min(HISTORY_PAGE_SIZE, displayed.length - visible.length)} more games (${visible.length} of ${displayed.length})</button>`);
  }
  const countLine = histEpochFilter !== null
    ? `${displayed.length} of ${filtered.length} loaded · epoch ${histEpochFilter}`
    : historyPage.totalMatches !== null && historyPage.totalMatches !== undefined
      ? `${filtered.length} of ${historyPage.totalMatches} games`
      : historyPage.countLoading
        ? `${filtered.length} loaded · counting`
        : `${filtered.length} loaded`;
  listParts.push(`<div class="hist-list-count">${escapeText(countLine)}</div>`);
  gameHistoryList.innerHTML = listParts.join("");
  gameHistoryDetail.innerHTML = selected
    ? gameHistoryDetailHtml(selected.run, selected)
    : `<div class="empty-list">No game selected</div>`;
}

// UNUSED(2026-06-12): no callers found in app.js/index.html (orphaned by the
// Match-v2/History-v2 rewrites).
function summaryMetric(key, value) {
  return `<div><span>${escapeText(key)}</span><strong>${escapeText(displayValue(value))}</strong></div>`;
}

function historyRunsForPage() {
  if (historySelectedRun === HISTORY_ALL_RUNS) {
    return trainingRuns
      .map(run => trainingRunDetails[run.name])
      .filter(Boolean);
  }
  const selected = trainingRunDetails[historySelectedRun] ||
    (trainingRun && trainingRun.name === historySelectedRun ? trainingRun : null);
  return selected ? [selected] : [];
}

function historySelectionPendingDetails() {
  if (!trainingRuns.length) return false;
  if (historySelectedRun === HISTORY_ALL_RUNS) {
    return trainingRuns.some(run => run.name && !trainingRunDetails[run.name]);
  }
  return trainingRuns.some(run => run.name === historySelectedRun) && !trainingRunDetails[historySelectedRun];
}

function historyItemsForPage(runs) {
  if (historyPage.loaded || historyPage.loading || historyPage.items.length > 0) {
    return historyPage.items || [];
  }
  return runs.flatMap(run => (run.histories || []).map(item => ({ ...item, run: run.name })));
}

function handleGameHistoryClick(event) {
  const moreButton = event.target.closest("[data-history-more]");
  if (moreButton) {
    event.preventDefault();
    if (historyPage.nextCursor) {
      loadHistoryPage({ append: true });
      return;
    }
    historyVisibleLimit += HISTORY_PAGE_SIZE;
    renderGameHistoryPage();
    return;
  }
  const debugButton = event.target.closest("[data-debug-open]");
  if (debugButton) {
    event.preventDefault();
    debugOpenFromHistory({
      run: debugButton.dataset.debugRun || (trainingRun && trainingRun.name) || "",
      path: debugButton.dataset.debugPath,
      record: Number(debugButton.dataset.debugRecord || 0),
      ply: null,
    });
    return;
  }
  const loadButton = event.target.closest("[data-history-load]");
  if (loadButton) {
    event.preventDefault();
    const runName = loadButton.dataset.historyRun || (trainingRun && trainingRun.name);
    selectedHistoryKey = historyItemKey({
      run: runName,
      path: loadButton.dataset.historyPath,
      record_index: Number(loadButton.dataset.recordIndex || 0),
    });
    renderGameHistoryPage();
    loadTrainingHistory(runName, loadButton.dataset.historyPath, Number(loadButton.dataset.recordIndex || 0));
    return;
  }
  // P2.1: detail-panel prev/next — the same state + render path as a row click.
  const stepButton = event.target.closest("[data-hist-game-step]");
  if (stepButton) {
    event.preventDefault();
    histStepGame(Number(stepButton.dataset.histGameStep || 0));
    return;
  }
  const row = event.target.closest("[data-history-key]");
  if (!row) return;
  event.preventDefault();
  selectedHistoryKey = row.dataset.historyKey || "";
  renderGameHistoryPage();
}

function filteredHistoryItems(histories) {
  const query = historyFilters.query.trim().toLowerCase();
  return histories.filter(item => {
    if (historyFilters.source !== "all" && String(item.source || "history") !== historyFilters.source) return false;
    if (historyFilters.winner === "none" && item.winner) return false;
    if (historyFilters.winner !== "all" && historyFilters.winner !== "none" && item.winner !== historyFilters.winner) return false;
    if (!query) return true;
    const haystack = [
      item.game_id,
      item.run,
      item.path,
      item.status,
      item.source,
      item.epoch,
      item.seed,
      item.winner_label,
      item.length,
      historyPlayerLabel(item.players && item.players.player0),
      historyPlayerLabel(item.players && item.players.player1),
      historyDiagnosticsText(item.diagnostics),
    ].filter(value => value !== undefined && value !== null).join(" ").toLowerCase();
    return haystack.includes(query);
  });
}

function sortedHistoryItems(histories) {
  const items = [...histories];
  const newest = (a, b) => compareHistoryNewest(a, b);
  if (historySort === "longest") {
    return items.sort((a, b) => compareNumber(historyLength(b), historyLength(a)) || newest(a, b));
  }
  if (historySort === "shortest") {
    return items.sort((a, b) => compareNumber(historyLength(a), historyLength(b)) || newest(a, b));
  }
  if (historySort === "oldest") {
    return items.sort((a, b) => -newest(a, b));
  }
  if (historySort === "winner") {
    return items.sort((a, b) => String(a.winner_label || winnerLabel(a.winner)).localeCompare(String(b.winner_label || winnerLabel(b.winner))) || newest(a, b));
  }
  return items.sort(newest);
}

function compareHistoryNewest(a, b) {
  return compareNumber(Number(b.modified || 0), Number(a.modified || 0)) ||
    compareNumber(Number(b.epoch || 0), Number(a.epoch || 0)) ||
    compareNumber(Number(b.record_index || 0), Number(a.record_index || 0));
}

function compareNumber(a, b) {
  const left = Number.isFinite(Number(a)) ? Number(a) : 0;
  const right = Number.isFinite(Number(b)) ? Number(b) : 0;
  return left === right ? 0 : left > right ? 1 : -1;
}

function historyLength(item) {
  return Number(item && (item.length || item.actions || 0)) || 0;
}

function selectedHistoryItem(histories, filtered) {
  const candidates = filtered.length ? filtered : histories;
  let selected = candidates.find(item => historyItemKey(item) === selectedHistoryKey) || null;
  if (!selected && candidates.length) {
    selected = candidates[0];
    selectedHistoryKey = historyItemKey(selected);
  }
  if (!selected) selectedHistoryKey = "";
  return selected;
}

function historyItemKey(item) {
  return `${item && item.run ? item.run : ""}::${item && item.path ? item.path : ""}::${Number(item && item.record_index || 0)}`;
}

// UNUSED(2026-06-12): no callers found in app.js/index.html (orphaned by the
// History-v2 rewrite; the epoch list now comes from histInspRunRows/epoch table).
function historyEpochs(histories) {
  return [...new Set((histories || [])
    .map(item => Number(item.epoch))
    .filter(epoch => Number.isFinite(epoch)))].sort((a, b) => a - b);
}

function latestRunStatusForHistoryPage() {
  const runs = historyRunsForPage();
  return runs
    .map(run => run && run.status)
    .filter(Boolean)
    .sort((a, b) => Number(b.history && b.history.latest_modified || 0) - Number(a.history && a.history.latest_modified || 0))[0] || null;
}

function humanizeStageId(stage) {
  const raw = String(stage || "").trim();
  if (!raw) return "Unknown";
  const lower = raw.toLowerCase();
  const epochMatch = lower.match(/^epoch[_-]?0*(\d+)/);
  if (epochMatch) return `Epoch ${Number(epochMatch[1])}`;
  if (lower.includes("write_diagnostics") || lower.includes("diagnostic")) return "Writing diagnostics";
  if (lower.includes("calibrat")) return "Calibrating";
  if (lower.includes("initialize")) return "Initializing";
  if (lower.includes("load_checkpoint")) return "Loading checkpoint";
  if (lower.includes("publish")) return "Publishing checkpoint";
  if (lower.includes("selfplay")) return "Self-play";
  if (lower.includes("shuffle")) return "Shuffling data";
  if (lower.includes("evaluat")) return "Evaluating";
  if (lower.includes("train")) return "Training";
  // Unknown id: prettify rather than dumping the raw token.
  return raw.replace(/[_-]+/g, " ").replace(/\b\w/g, ch => ch.toUpperCase());
}

function runStageLabel(status) {
  if (!status || typeof status !== "object") return "--";
  const stage = humanizeStageId(status.stage || "unknown");
  // Epoch ids already carry the epoch number; avoid "Epoch 1 · e1".
  const epochNum = asFinite(status.current_epoch);
  const epoch = epochNum !== null && !/^epoch/i.test(String(status.stage || "")) ? ` · Epoch ${epochNum}` : "";
  // Prefer the derived within-epoch sub-phase (self-play / shuffling / training /
  // evaluating) over the generic "running" stage_status, so the long SealBot eval
  // is distinguishable from the rest of an epoch. Falls back to stage_status when
  // no sub-phase is available.
  const subPhase = status.sub_phase ? String(status.sub_phase) : "";
  if (subPhase) {
    const detail = status.sub_phase_detail ? ` ${String(status.sub_phase_detail)}` : "";
    return `${stage}${epoch} · ${subPhase}${detail}`;
  }
  const stageStatus = status.stage_status && status.stage_status !== "unknown"
    ? ` · ${String(status.stage_status).replace(/[_-]+/g, " ")}`
    : "";
  return `${stage}${epoch}${stageStatus}`;
}

// UNUSED(2026-06-12): no callers found in app.js/index.html (orphaned by the
// History-v2 rewrite; game-length stats now come from the server's epoch rows).
function averageHistoryLength(histories) {
  const lengths = (histories || [])
    .map(item => Number(item.length || item.actions || 0))
    .filter(value => Number.isFinite(value) && value > 0);
  return lengths.length ? lengths.reduce((sum, value) => sum + value, 0) / lengths.length : null;
}

function asFinite(value) {
  // Treat null/undefined/"" as missing, not as the numeric 0 that Number()
  // would coerce them to (Number(null) === 0). A real numeric 0 still passes.
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatDecimal(value, digits = 1) {
  const number = asFinite(value);
  if (number === null) return "--";
  return number.toFixed(digits);
}

function formatRate(value, unit) {
  const number = asFinite(value);
  if (number === null) return "--";
  return `${number.toFixed(number >= 100 ? 0 : 1)} ${unit}`;
}

function formatPercent(value, digits = 0) {
  const number = asFinite(value);
  if (number === null) return "--";
  return `${(number * 100).toFixed(digits)}%`;
}

function formatGib(value) {
  const number = asFinite(value);
  if (number === null) return "--";
  // Values are GiB (binary); label honestly to match the /1024 sizing used
  // elsewhere rather than the misleading decimal "GB".
  return `${number.toFixed(1)} GiB`;
}

// Compact age/duration: 42 -> "42s", 310 -> "5.2m", 7300 -> "2.0h".
function formatAge(seconds) {
  const n = asFinite(seconds);
  if (n === null) return "--";
  if (n < 90) return `${Math.round(n)}s`;
  if (n < 5400) return `${(n / 60).toFixed(1)}m`;
  return `${(n / 3600).toFixed(1)}h`;
}

// --- History v2 Region 1: status band (#histStatusBand + #histHealthDrawer). ---
// One wrapping flex row: stage pill (+ live dot during fresh selfplay), an
// optional inline progress bar, watchdog resource chips, learning-health stat
// chips, and the health pill that toggles the messages drawer. Everything
// degrades to "--" / omission when a field is absent (dense_cnn runs lack
// training_progress; stopped runs lack selfplay_live; watchdog is optional).
function histChip(label, value, title = "", extraClass = "") {
  const titleAttr = title ? ` title="${escapeAttr(title)}"` : "";
  const labelHtml = label ? `<i>${escapeText(label)}</i> ` : "";
  return `<span class="hist-chip${extraClass ? ` ${extraClass}` : ""}"${titleAttr}>${labelHtml}${escapeText(value)}</span>`;
}

function renderHistStatusBand(runs) {
  if (!histStatusBand) return;
  const status = latestRunStatusForHistoryPage();
  const health = latestLearningHealth(runs);
  if (!status && !health) {
    histStatusBand.innerHTML = `<div class="hist-band-empty">No run status yet</div>`;
    if (histHealthDrawer) {
      histHealthDrawer.innerHTML = "";
      histHealthDrawer.hidden = true;
    }
    return;
  }
  const parts = [];

  // Stage pill, with pulsing live dot + games/pos-s while selfplay is fresh.
  const live = status && status.selfplay_live && status.selfplay_live.live ? status.selfplay_live : null;
  const liveDot = live ? `<span class="hist-live-dot" aria-hidden="true"></span>` : "";
  const liveSuffix = live
    ? ` · ${Number(live.games_finished || 0)}/${Number(live.requested_games || 0)} · ${formatRate(live.search_pos_s, "pos/s")}`
    : "";
  // The live suffix already carries the games counter, so drop the sub-phase
  // detail (also "games N/M") from the label while self-play is live.
  const labelStatus = live && status && status.sub_phase_detail
    ? { ...status, sub_phase_detail: null }
    : status;
  parts.push(`<span id="histStagePill" class="hist-pill">${liveDot}<span>${escapeText(`${runStageLabel(labelStatus)}${liveSuffix}`)}</span></span>`);

  // Supervisor health. A tripped breaker (halted flag) silently blocks
  // relaunches while events.jsonl still reads "running", so it outranks
  // everything else in the band; a dead trainer (EXIT with no later LAUNCH)
  // is the same signal, softer; a crashy-but-relaunching run gets a warn chip.
  const sup = status && status.supervisor && typeof status.supervisor === "object" ? status.supervisor : null;
  if (sup) {
    const crashTitle = sup.last_crash ? `last crash: ${sup.last_crash}` : "";
    if (sup.halted) {
      parts.push(`<span class="hist-pill hist-pill-halt" title="${escapeAttr(crashTitle || "breaker tripped")}">supervisor HALTED · clear supervisor_halted.flag to resume</span>`);
    } else if (sup.trainer_presumed_up === false) {
      const exitAge = asFinite(sup.last_exit_age_seconds);
      parts.push(`<span class="hist-pill hist-pill-halt" title="${escapeAttr(crashTitle || "no LAUNCH after the last EXIT in supervisor.log")}">trainer down${exitAge !== null ? ` · exited ${formatAge(exitAge)} ago` : ""}</span>`);
    } else if (asFinite(sup.exits_last_hour) !== null && Number(sup.exits_last_hour) > 0) {
      parts.push(histChip("restarts", `${Number(sup.exits_last_hour)} in last hour`, crashTitle, "hist-chip-warn"));
    }
  }

  // Inline progress: live selfplay games are the primary signal; the derived
  // phase progress (shrimp training pass: elapsed vs typical duration) comes
  // next; the training_progress file is normally absent (no current producer
  // emits it), so it is only a fallback. No bar at all otherwise.
  let barFrac = null;
  let barLabel = "";
  if (live && asFinite(live.requested_games) !== null && Number(live.requested_games) > 0) {
    barFrac = Number(live.games_finished || 0) / Number(live.requested_games);
    barLabel = `selfplay ${Number(live.games_finished || 0)}/${Number(live.requested_games)}`;
  } else if (status && status.phase_progress && typeof status.phase_progress === "object"
    && asFinite(status.phase_progress.elapsed_seconds) !== null
    && asFinite(status.phase_progress.typical_seconds) !== null
    && Number(status.phase_progress.typical_seconds) > 0) {
    const pp = status.phase_progress;
    // Typical duration is an estimate; cap the fill so a slow pass reads as
    // "nearly there", never as a false 100%.
    barFrac = Math.min(Number(pp.elapsed_seconds) / Number(pp.typical_seconds), 0.98);
    barLabel = `${pp.phase || "phase"} ${formatAge(Number(pp.elapsed_seconds))} / ~${formatAge(Number(pp.typical_seconds))}`;
  } else if (status && status.training_progress && typeof status.training_progress === "object") {
    const tp = status.training_progress;
    barFrac = asFinite(tp.progress);
    if (barFrac === null) {
      const steps = asFinite(tp.steps);
      const total = asFinite(tp.total_steps);
      barFrac = steps !== null && total !== null && total > 0 ? steps / total : null;
    }
    barLabel = `${formatTrainingProgress(tp)} · ${trainingProgressSubtext(tp)}`;
  }
  if (barFrac !== null) {
    const pct = (Math.max(0, Math.min(1, barFrac)) * 100).toFixed(1);
    parts.push(`<span id="histTrainBar" class="hist-bar" title="${escapeAttr(barLabel)}">`
      + `<span class="hist-bar-fill" style="width:${pct}%"></span>`
      + `<span class="hist-bar-label">${escapeText(barLabel)}</span></span>`);
  }

  // Watchdog resources as value chips (free values only — the watchdog exposes
  // no totals, so percentage bars are impossible). Warn tint on any non-ok
  // status or non-empty critical list.
  const watchdog = status && status.watchdog && typeof status.watchdog === "object" ? status.watchdog : null;
  if (watchdog) {
    const critical = Array.isArray(watchdog.critical) ? watchdog.critical : [];
    const warn = (watchdog.status && watchdog.status !== "ok") || critical.length > 0;
    const warnClass = warn ? "hist-chip-warn" : "";
    const warnTitle = warn ? `watchdog ${watchdog.status || "warn"}${critical.length ? `: ${critical.join(", ")}` : ""}` : "";
    const chips = [];
    if (asFinite(watchdog.free_ram_gb) !== null) {
      chips.push(histChip("RAM", `${formatGib(watchdog.free_ram_gb)} free`, warnTitle, warnClass));
    }
    if (asFinite(watchdog.gpu_free_gb) !== null) {
      const util = asFinite(watchdog.gpu_utilization_percent);
      const gpuTitle = [util !== null ? `GPU util ${util}%` : "", warnTitle].filter(Boolean).join(" · ");
      chips.push(histChip("GPU", `${formatGib(watchdog.gpu_free_gb)} free`, gpuTitle, warnClass));
    }
    if (chips.length) parts.push(`<span id="histResources" class="hist-resources">${chips.join("")}</span>`);
  }

  // Latest checkpoint + age: the at-a-glance "how stale are the weights"
  // readout. Warn tint once the age exceeds ~2h (a healthy epoch is well
  // under an hour, so 2h means the run stopped making checkpoints).
  const ckptInfo = status && status.latest_checkpoint && typeof status.latest_checkpoint === "object" ? status.latest_checkpoint : null;
  if (ckptInfo && ckptInfo.name) {
    const ckptAge = asFinite(ckptInfo.age_seconds);
    const ckptStale = ckptAge !== null && ckptAge > 2 * 3600;
    const ckptEpoch = asFinite(ckptInfo.epoch);
    parts.push(histChip(
      "ckpt",
      `${ckptEpoch !== null ? `e${ckptEpoch}` : String(ckptInfo.name)} · ${ckptAge !== null ? `${formatAge(ckptAge)} ago` : "--"}`,
      `${String(ckptInfo.name)}${ckptStale ? " — no new checkpoint for over 2h" : ""}`,
      ckptStale ? "hist-chip-warn" : "",
    ));
  }

  // Learning-health stat chips ("--" when null so a young run never throws).
  const h = health || {};
  const latestEpoch = asFinite(h.latest_epoch);
  // Eval chip: shrimp runs carry a multi-stage verdict + candidate Elo (the
  // dead `mean_turns` path is all-null), so render the real signal — e.g.
  // "+37 Elo · INCONCLUSIVE" with the SealBot winrate + epoch as the title —
  // instead of "--". dense_cnn lineages still get the legacy turns chip.
  const candElo = asFinite(h.latest_cand_elo);
  const verdictShort = h.latest_verdict ? String(h.latest_verdict).slice(0, 5) : null;  // PROMO/REGRE/INCON
  let evalValue;
  let evalTitle;
  if (candElo !== null || verdictShort) {
    const eloTxt = candElo !== null ? `${candElo > 0 ? "+" : ""}${Math.round(candElo)} Elo` : null;
    evalValue = [eloTxt, verdictShort].filter(Boolean).join(" · ") || "--";
    const sbWr = asFinite(h.latest_sealbot_winrate);
    const evalEp = asFinite(h.latest_eval_epoch);
    evalTitle = [
      h.latest_verdict ? `verdict ${h.latest_verdict}` : "",
      sbWr !== null ? `SealBot ${formatPercent(sbWr)}` : "",
      evalEp !== null ? `eval epoch ${evalEp}` : "",
    ].filter(Boolean).join(" · ");
  } else {
    evalValue = asFinite(h.latest_eval_mean_turns) !== null ? `${formatDecimal(h.latest_eval_mean_turns, 1)}t` : "--";
    evalTitle = [
      asFinite(h.best_eval_mean_turns) !== null ? `best ${formatDecimal(h.best_eval_mean_turns, 1)}t` : "",
      asFinite(h.latest_eval_games) !== null ? `${Number(h.latest_eval_wins || 0)}/${Number(h.latest_eval_games)} wins` : "",
    ].filter(Boolean).join(" · ");
  }
  const statChips = [
    histChip("epoch", latestEpoch !== null ? `e${latestEpoch}` : "--"),
    histChip("loss", formatDecimal(h.latest_loss, 3)),
    histChip("eval", evalValue, evalTitle),
    histChip("speed", formatRate(h.latest_selfplay_pos_s, "pos/s")),
    histChip("P@1", formatPercent(h.latest_policy_top1)),
    histChip("C", formatPercent(h.latest_classical_fraction), "classical replay fraction"),
  ];
  parts.push(`<span id="histStatChips" class="hist-stat-chips">${statChips.join("")}</span>`);

  // Health pill: button toggling the messages drawer; open state survives the
  // 15s re-render via the histHealthDrawerOpen module var.
  const messages = health && Array.isArray(health.messages) ? health.messages : [];
  if (health) {
    const knownStatuses = ["collecting", "ok", "improving", "watch", "intervene"];
    const hs = knownStatuses.includes(health.status) ? health.status : "ok";
    parts.push(`<button id="histHealthPill" type="button" class="hist-pill hist-health-pill hist-health-${hs}"`
      + ` aria-expanded="${histHealthDrawerOpen ? "true" : "false"}" aria-controls="histHealthDrawer"`
      + ` title="Learning health — click for messages">${escapeText(learningHealthLabel(hs))}${messages.length ? ` · ${messages.length}` : ""}</button>`);
  }

  // R6: freshness readout for the fast poll tier. Rendered empty here — the
  // 1s histUpdateLiveTick interval owns its text (textContent only).
  parts.push(`<span id="histLiveTick" class="hist-live-tick"></span>`);

  histStatusBand.innerHTML = parts.join("");
  histUpdateLiveTick();
  if (histHealthDrawer) {
    if (histHealthDrawerOpen && health) {
      histHealthDrawer.innerHTML = messages.length
        ? messages.map(message => `<div class="hist-health-msg">${escapeText(message)}</div>`).join("")
        : `<div class="hist-health-msg">No health messages yet.</div>`;
      histHealthDrawer.hidden = false;
    } else {
      histHealthDrawer.innerHTML = "";
      histHealthDrawer.hidden = true;
    }
  }
}

function formatTrainingProgress(progress) {
  const epochNum = progress ? asFinite(progress.epoch) : null;
  const epoch = epochNum !== null ? `e${epochNum}` : "train";
  const pct = progress && progress.progress !== undefined && progress.progress !== null ? formatPercent(progress.progress) : "--";
  return `${epoch} ${pct}`;
}

function trainingProgressSubtext(progress) {
  if (!progress || typeof progress !== "object") return "Training progress";
  const steps = asFinite(progress.steps) !== null && asFinite(progress.total_steps) !== null
    ? `${progress.steps}/${progress.total_steps} steps`
    : "steps pending";
  const loss = progress.loss !== undefined && progress.loss !== null ? `loss ${formatDecimal(progress.loss, 3)}` : String(progress.status || "training");
  return `${steps} | ${loss}`;
}

function latestLearningHealth(runs) {
  return runs
    .map(run => run && run.learning_health)
    .filter(Boolean)
    .sort((a, b) => Number(b.latest_epoch || 0) - Number(a.latest_epoch || 0))[0] || null;
}

function learningHealthClass(status) {
  if (status === "intervene") return "intervene";
  if (status === "watch") return "watch";
  if (status === "improving") return "improving";
  return "ok";
}

function learningHealthLabel(status) {
  if (status === "intervene") return "Intervene";
  if (status === "watch") return "Watch";
  if (status === "improving") return "Improving";
  if (status === "collecting") return "Collecting";
  return "OK";
}

// --- History v2 Region 2: per-epoch trends grid (#histTrends). ---
// Hand-rolled SVG sparkline charts over run.epoch_history / evaluation_history.
// One fixed viewBox per chart; hover/click are handled by a single delegated
// listener trio on #histTrends using the geometry stored in histTrendCharts.
const HIST_CHART_W = 280;
const HIST_CHART_H = 110;
const HIST_PLOT_X0 = 10;
const HIST_PLOT_X1 = 272;
const HIST_PLOT_Y0 = 22;
const HIST_PLOT_Y1 = 92;

function histSeriesCount(values) {
  return (values || []).filter(value => value !== null && value !== undefined).length;
}

function histLatestNonNull(values) {
  for (let i = (values || []).length - 1; i >= 0; i--) {
    const value = values[i];
    if (value !== null && value !== undefined) return value;
  }
  return null;
}

// Path "d" for a line series; null points lift the pen so gaps stay gaps.
function histPathD(values, xAt, yAt) {
  let d = "";
  let pen = false;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (value === null || value === undefined) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${xAt(i).toFixed(1)} ${yAt(value).toFixed(1)}`;
    pen = true;
  }
  return d;
}

// Shaded band polygons (mean±σ); split into contiguous non-null segments.
function histBandPolys(loVals, hiVals, xAt, yAt, color) {
  const polys = [];
  let seg = [];
  const flush = () => {
    if (seg.length >= 2) {
      const top = seg.map(([i, , h]) => `${xAt(i).toFixed(1)},${yAt(h).toFixed(1)}`).join(" ");
      const bottom = seg.slice().reverse().map(([i, l]) => `${xAt(i).toFixed(1)},${yAt(l).toFixed(1)}`).join(" ");
      polys.push(`<polygon points="${top} ${bottom}" style="fill:${color}" stroke="none"></polygon>`);
    }
    seg = [];
  };
  for (let i = 0; i < loVals.length; i++) {
    const lo = loVals[i];
    const hi = hiVals[i];
    if (lo === null || lo === undefined || hi === null || hi === undefined) {
      flush();
      continue;
    }
    seg.push([i, lo, hi]);
  }
  flush();
  return polys.join("");
}

// Stacked 100% area layers (win balance). layers = [{color, values}] bottom-up;
// values are pre-normalized fractions; null indices break the polygons.
function histStackedPolys(layers, xAt, yAt) {
  const n = layers.length ? layers[0].values.length : 0;
  const cum = new Array(n).fill(0);
  const out = [];
  layers.forEach(layer => {
    let seg = [];
    const flush = () => {
      if (seg.length >= 2) {
        const top = seg.map(([i, , t]) => `${xAt(i).toFixed(1)},${yAt(t).toFixed(1)}`).join(" ");
        const bottom = seg.slice().reverse().map(([i, b]) => `${xAt(i).toFixed(1)},${yAt(b).toFixed(1)}`).join(" ");
        out.push(`<polygon points="${top} ${bottom}" style="fill:${layer.color}" stroke="none"></polygon>`);
      }
      seg = [];
    };
    for (let i = 0; i < n; i++) {
      const value = layer.values[i];
      if (value === null || value === undefined) {
        flush();
        continue;
      }
      seg.push([i, cum[i], cum[i] + value]);
    }
    flush();
    for (let i = 0; i < n; i++) {
      const value = layer.values[i];
      if (value !== null && value !== undefined) cum[i] += value;
    }
  });
  return out.join("");
}

// Generic chart card builder. def: {id, title, epochs, series, band?, stacked?,
// domain?, format?, tip?, annotate?, latest?}. Registers hover geometry in
// histTrendCharts and returns the card HTML ("" when unplottable).
function histChartSvg(def) {
  const epochs = def.epochs || [];
  const n = epochs.length;
  if (n < 2) return "";
  const x0 = HIST_PLOT_X0;
  const dx = (HIST_PLOT_X1 - HIST_PLOT_X0) / (n - 1);
  const xAt = i => x0 + i * dx;
  const fmt = def.format || (value => formatDecimal(value, 2));
  let lo = Infinity;
  let hi = -Infinity;
  const scan = values => (values || []).forEach(value => {
    if (value === null || value === undefined) return;
    if (value < lo) lo = value;
    if (value > hi) hi = value;
  });
  if (def.domain) {
    lo = def.domain[0];
    hi = def.domain[1];
  } else {
    (def.series || []).forEach(series => scan(series.values));
    if (def.band) {
      scan(def.band.lo);
      scan(def.band.hi);
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return "";
  if (hi === lo) {
    const pad = Math.abs(hi) > 1e-9 ? Math.abs(hi) * 0.05 : 1;
    lo -= pad;
    hi += pad;
  }
  const yAt = value => HIST_PLOT_Y1 - ((Math.min(Math.max(value, lo), hi) - lo) / (hi - lo)) * (HIST_PLOT_Y1 - HIST_PLOT_Y0);
  const parts = [];
  if (def.stacked) parts.push(histStackedPolys(def.stacked, xAt, yAt));
  if (def.band) parts.push(histBandPolys(def.band.lo, def.band.hi, xAt, yAt, def.band.color || "rgba(39,215,230,0.12)"));
  (def.series || []).forEach(series => {
    const color = series.color || "var(--accent)";
    if (series.dots) {
      series.values.forEach((value, i) => {
        if (value === null || value === undefined) return;
        parts.push(`<circle cx="${xAt(i).toFixed(1)}" cy="${yAt(value).toFixed(1)}" r="2" style="fill:${color}"></circle>`);
      });
    } else {
      const d = histPathD(series.values, xAt, yAt);
      if (d) parts.push(`<path d="${d}" fill="none" style="stroke:${color}" stroke-width="${series.width || 1.6}" stroke-linejoin="round" stroke-linecap="round"></path>`);
    }
    // Visible per-point dots (cadence signposting): so a sparse every-5-epoch
    // series isn't mistaken for missing data on the dense per-epoch x-axis.
    if (series.points) {
      series.values.forEach((value, i) => {
        if (value === null || value === undefined) return;
        parts.push(`<circle cx="${xAt(i).toFixed(1)}" cy="${yAt(value).toFixed(1)}" r="1.9" style="fill:${color}"></circle>`);
      });
    }
    if (series.markers) {
      series.values.forEach((value, i) => {
        if (!series.markers[i] || value === null || value === undefined) return;
        parts.push(`<circle cx="${xAt(i).toFixed(1)}" cy="${yAt(value).toFixed(1)}" r="2.8" style="fill:${series.markerColor || "#36d399"}"></circle>`);
      });
    }
  });
  if (def.annotate && def.annotate.index >= 0 && def.annotate.index < n) {
    const ax = xAt(def.annotate.index);
    const av = def.annotate.value;
    const ay = av === null || av === undefined
      ? HIST_PLOT_Y0 + 8
      : Math.max(HIST_PLOT_Y0 + 8, yAt(av) - 6);
    const anchor = ax > HIST_CHART_W - 44 ? "end" : ax < 44 ? "start" : "middle";
    parts.push(`<text class="hist-chart-axis" x="${ax.toFixed(1)}" y="${ay.toFixed(1)}" text-anchor="${anchor}">${escapeText(def.annotate.text)}</text>`);
  }
  // In-plot min/max y labels + latest-epoch tick.
  parts.push(`<text class="hist-chart-axis" x="12" y="${HIST_PLOT_Y0 + 8}">${escapeText(fmt(hi))}</text>`);
  parts.push(`<text class="hist-chart-axis" x="12" y="${HIST_PLOT_Y1 - 2}">${escapeText(fmt(lo))}</text>`);
  const lastX = xAt(n - 1).toFixed(1);
  parts.push(`<line class="hist-chart-tick" x1="${lastX}" x2="${lastX}" y1="${HIST_PLOT_Y1}" y2="${HIST_PLOT_Y1 + 4}"></line>`);
  parts.push(`<text class="hist-chart-axis" x="${lastX}" y="${HIST_CHART_H - 3}" text-anchor="end">e${escapeText(epochs[n - 1])}</text>`);
  parts.push(`<text class="hist-chart-title" x="10" y="13">${escapeText(def.title)}</text>`);
  if (def.subtitle) {
    parts.push(`<text class="hist-chart-axis" x="10" y="20">${escapeText(def.subtitle)}</text>`);
  }
  if (def.latest !== undefined && def.latest !== null && def.latest !== "") {
    parts.push(`<text class="hist-chart-latest" x="${HIST_CHART_W - 8}" y="13" text-anchor="end">${escapeText(def.latest)}</text>`);
  }
  parts.push(`<line class="hist-trend-cross" x1="-10" x2="-10" y1="${HIST_PLOT_Y0}" y2="${HIST_PLOT_Y1}" visibility="hidden"></line>`);
  const tipSeries = (def.tip || (def.series || []).map(series => ({ label: series.label, values: series.values })))
    .map(series => ({ label: series.label, values: series.values, fmt: series.fmt || fmt }));
  histTrendCharts[def.id] = { epochs, x0, dx, series: tipSeries };
  return `<div class="hist-trend-card"><svg viewBox="0 0 ${HIST_CHART_W} ${HIST_CHART_H}" data-hist-chart="${escapeAttr(def.id)}" role="img" aria-label="${escapeAttr(def.title)}">${parts.join("")}</svg></div>`;
}

// Charts always come from a single run: merging epoch series across runs
// interleaves epochs into sawtooth garbage. Under "All runs" pick the run
// whose status.history.latest_modified is newest (the same run that
// latestRunStatusForHistoryPage reports on).
function histTrendRun(runs) {
  const withStatus = (runs || []).filter(run => run && run.status);
  if (!withStatus.length) return (runs && runs[0]) || null;
  return withStatus.slice().sort((a, b) =>
    Number(b.status.history && b.status.history.latest_modified || 0) -
    Number(a.status.history && a.status.history.latest_modified || 0))[0];
}

// ---------------------------------------------------------------------------
// Multi-stage eval (shrimp, opt-in): SealBot-pinned Bradley-Terry ratings.
// Every helper degrades to empty/null on absent payload so non-shrimp and
// pre-multistage runs render nothing (the web.py readers return [] / null).
// ---------------------------------------------------------------------------

// The candidate player row in a report's `ratings.players` (Elo-descending,
// SealBot-anchored). Prefer the verdict's named candidate; else the highest
// non-anchor; if every row is an anchor (degenerate), the first player.
function msPlayers(row) {
  const ratings = row && typeof row.ratings === "object" && row.ratings ? row.ratings : {};
  return Array.isArray(ratings.players) ? ratings.players.filter(p => p && typeof p === "object") : [];
}

function msCandidatePlayer(row) {
  const players = msPlayers(row);
  if (!players.length) return null;
  const verdict = row && typeof row.verdict === "object" && row.verdict ? row.verdict : {};
  const primary = verdict && typeof verdict.primary === "object" && verdict.primary ? verdict.primary : {};
  const wantLabel = primary.candidate;
  if (wantLabel) {
    const named = players.find(p => p.label === wantLabel);
    if (named) return named;
  }
  return players.find(p => !p.is_anchor) || players[0];
}

function msAnchorPlayer(row) {
  return msPlayers(row).find(p => p.is_anchor) || null;
}

// The descriptive headline edges in priority order: vs champion (primary), then
// vs SealBot and vs BC-prefit by opponent/role. Returns [] when edges absent.
function msHeadlineEdges(row) {
  const edges = row && Array.isArray(row.edges) ? row.edges.filter(e => e && typeof e === "object") : [];
  if (!edges.length) return [];
  const seen = new Set();
  const picked = [];
  const take = edge => {
    if (!edge || seen.has(edge)) return;
    seen.add(edge);
    picked.push(edge);
  };
  const looksLike = (edge, needle) =>
    String(edge.opponent || edge.role || "").toLowerCase().includes(needle);
  edges.filter(e => e.primary).forEach(take);
  edges.filter(e => looksLike(e, "sealbot")).forEach(take);
  edges.filter(e => looksLike(e, "bc") || looksLike(e, "prefit")).forEach(take);
  return picked.slice(0, 4);
}

// Human label for an edge ("vs SealBot" etc.); falls back to opponent/role.
function msEdgeLabel(edge) {
  const opp = edge && (edge.opponent || edge.role);
  if (!opp) return edge && edge.primary ? "vs champion" : "edge";
  return `vs ${opp}`;
}

// Descriptive SealBot zero-point winrate for one report row: the sealbot edge's
// winrate, else the midpoint of sealbot_winrate_ci95. null when SealBot absent.
function msSealbotWinrate(row) {
  const edges = row && Array.isArray(row.edges) ? row.edges : [];
  const edge = edges.find(e => e && String(e.opponent || e.role || "").toLowerCase() === "sealbot");
  if (edge) {
    const wr = asFinite(edge.winrate);
    if (wr !== null) return wr;
  }
  const ci = Array.isArray(row && row.sealbot_winrate_ci95) ? row.sealbot_winrate_ci95 : null;
  if (ci) {
    const lo = asFinite(ci[0]);
    const hi = asFinite(ci[1]);
    if (lo !== null && hi !== null) return (lo + hi) / 2;
  }
  return null;
}

// Configured permanent anchors that are NOT in this report's roster (SEV-2: the
// bc_prefit drop). Returns the dropped anchor labels, [] when none/absent.
function msDroppedAnchors(row) {
  const roster = row && typeof row.roster === "object" && row.roster ? row.roster : {};
  const perms = Array.isArray(roster.permanent_anchors) ? roster.permanent_anchors : [];
  if (!perms.length) return [];
  const present = new Set(
    (Array.isArray(roster.opponents) ? roster.opponents : [])
      .map(o => o && o.label)
      .filter(Boolean)
  );
  return perms.filter(name => name && !present.has(name));
}

// Cadence subtitle for the eval charts: the multistage eval runs every N epochs
// (inferred from the report spacing) while the x-axis is per-epoch, so signpost
// "evaluated every 5 epochs (7 points)" rather than letting the sparse points
// read as missing data.
function msEvalCadenceSubtitle(msRows) {
  const epochs = (msRows || []).map(r => Number(r.epoch)).filter(e => Number.isFinite(e)).sort((a, b) => a - b);
  if (epochs.length < 2) return "";
  let step = Infinity;
  for (let i = 1; i < epochs.length; i++) step = Math.min(step, epochs[i] - epochs[i - 1]);
  if (!Number.isFinite(step) || step < 1) step = 1;
  return `evaluated every ${step} epoch${step === 1 ? "" : "s"} (${epochs.length} points)`;
}

// "62% [48–74%]" winrate + CI; "--" when the winrate is missing.
function msWinrateText(edge) {
  const wr = asFinite(edge && edge.winrate);
  if (wr === null) return "--";
  const ci = Array.isArray(edge && edge.winrate_ci95) ? edge.winrate_ci95 : null;
  const lo = ci ? asFinite(ci[0]) : null;
  const hi = ci ? asFinite(ci[1]) : null;
  if (lo === null || hi === null) return formatPercent(wr);
  return `${formatPercent(wr)} [${formatPercent(lo)}–${formatPercent(hi)}]`;
}

// "1532 [1500–1564]" Elo + CI for a rating-table row; "--" when Elo missing.
function msEloText(player) {
  const elo = asFinite(player && player.elo);
  if (elo === null) return "--";
  const ci = Array.isArray(player && player.elo_ci95) ? player.elo_ci95 : null;
  const lo = ci ? asFinite(ci[0]) : null;
  const hi = ci ? asFinite(ci[1]) : null;
  if (lo === null || hi === null) return String(Math.round(elo));
  return `${Math.round(elo)} [${Math.round(lo)}–${Math.round(hi)}]`;
}

// Maps a PROMOTE/REGRESS/INCONCLUSIVE label to a status-palette tint class.
function msVerdictClass(label) {
  const norm = String(label || "").toUpperCase();
  if (norm === "PROMOTE") return "hist-health-improving";
  if (norm === "REGRESS") return "hist-health-intervene";
  return "hist-health-watch";
}

// Ascending-by-epoch multistage rows for the trend run; [] when absent.
function msHistoryRows(run) {
  const list = run && Array.isArray(run.multistage_eval_history) ? run.multistage_eval_history : [];
  return list
    .filter(row => row && typeof row === "object" && asFinite(row.epoch) !== null)
    .slice()
    .sort((a, b) => Number(a.epoch) - Number(b.epoch));
}

function renderHistTrends(runs) {
  if (!histTrends) return;
  histTrendCharts = {};
  histTrendHoverSvg = null;
  if (histTrendTipEl) histTrendTipEl.hidden = true;
  const run = histTrendRun(runs);
  const rows = run && Array.isArray(run.epoch_history)
    ? run.epoch_history.filter(row => row && asFinite(row.epoch) !== null).slice().sort((a, b) => Number(a.epoch) - Number(b.epoch))
    : [];
  const epochs = rows.map(row => Number(row.epoch));
  const sp = row => row.selfplay || {};
  const buf = row => (row.selfplay && row.selfplay.buffer) || {};
  const charts = [];
  if (rows.length >= 2) {
    // T1 Loss: total from training.loss; per-head thin lines from the buffer
    // block (uniform across hexgt bridge and the synthesized dense_cnn buffer).
    const lossTotal = rows.map(row => asFinite(row.training && row.training.loss));
    if (histSeriesCount(lossTotal) >= 2) {
      const lossPolicy = rows.map(row => asFinite(buf(row).loss_policy));
      const lossValue = rows.map(row => asFinite(buf(row).loss_value));
      const series = [{ label: "total", color: "var(--accent)", values: lossTotal, width: 1.8 }];
      if (histSeriesCount(lossPolicy) >= 2) series.push({ label: "policy", color: "var(--p0)", values: lossPolicy, width: 1 });
      if (histSeriesCount(lossValue) >= 2) series.push({ label: "value", color: "#ffce56", values: lossValue, width: 1 });
      charts.push(histChartSvg({
        id: "t1",
        title: "Loss",
        epochs,
        series,
        format: value => formatDecimal(value, 3),
        latest: formatDecimal(histLatestNonNull(lossTotal), 3),
      }));
    }
    // T2 Game length: mean line, median dots, mean±σ band.
    const lenMean = rows.map(row => asFinite(sp(row).game_length_mean));
    if (histSeriesCount(lenMean) >= 2) {
      const lenMedian = rows.map(row => asFinite(sp(row).game_length_median));
      const lenStdev = rows.map(row => asFinite(sp(row).game_length_stdev));
      const bandLo = lenMean.map((mean, i) => (mean !== null && lenStdev[i] !== null ? mean - lenStdev[i] : null));
      const bandHi = lenMean.map((mean, i) => (mean !== null && lenStdev[i] !== null ? mean + lenStdev[i] : null));
      const series = [{ label: "mean", color: "var(--accent)", values: lenMean, width: 1.8 }];
      if (histSeriesCount(lenMedian) >= 1) series.push({ label: "median", color: "var(--p0)", values: lenMedian, dots: true });
      charts.push(histChartSvg({
        id: "t2",
        title: "Game length",
        epochs,
        series,
        band: histSeriesCount(bandLo) >= 2 ? { lo: bandLo, hi: bandHi, color: "rgba(39,215,230,0.12)" } : null,
        tip: [
          { label: "mean", values: lenMean, fmt: value => formatDecimal(value, 1) },
          { label: "median", values: lenMedian, fmt: value => formatDecimal(value, 0) },
          { label: "σ", values: lenStdev, fmt: value => formatDecimal(value, 1) },
        ],
        format: value => formatDecimal(value, 0),
        latest: formatDecimal(histLatestNonNull(lenMean), 1),
      }));
    }
    // T3 Win balance: stacked 100% P0 / draw / P1 area.
    const winP0 = rows.map(row => asFinite(sp(row).win_p0_fraction));
    const winP1 = rows.map(row => asFinite(sp(row).win_p1_fraction));
    const drawF = rows.map(row => asFinite(sp(row).draw_fraction));
    const balanced = winP0.map((value, i) => value !== null && winP1[i] !== null);
    if (balanced.filter(Boolean).length >= 2) {
      const norm = i => {
        const total = winP0[i] + (drawF[i] !== null ? drawF[i] : 0) + winP1[i];
        return total > 0 ? total : 1;
      };
      const layerP0 = winP0.map((value, i) => (balanced[i] ? value / norm(i) : null));
      const layerDraw = drawF.map((value, i) => (balanced[i] ? (value !== null ? value : 0) / norm(i) : null));
      const layerP1 = winP1.map((value, i) => (balanced[i] ? value / norm(i) : null));
      charts.push(histChartSvg({
        id: "t3",
        title: "Win balance",
        epochs,
        domain: [0, 1],
        stacked: [
          { label: "P0", color: "var(--p0)", values: layerP0 },
          { label: "draw", color: "rgba(143,155,170,0.45)", values: layerDraw },
          { label: "P1", color: "var(--p1)", values: layerP1 },
        ],
        tip: [
          { label: "P0", values: layerP0 },
          { label: "draw", values: layerDraw },
          { label: "P1", values: layerP1 },
        ],
        format: value => formatPercent(value),
        latest: `P0 ${formatPercent(histLatestNonNull(layerP0))}`,
      }));
    }
    // T5 Selfplay speed.
    const speed = rows.map(row => asFinite(sp(row).search_positions_per_second));
    if (histSeriesCount(speed) >= 2) {
      charts.push(histChartSvg({
        id: "t5",
        title: "Selfplay speed",
        epochs,
        series: [{ label: "pos/s", color: "var(--accent)", values: speed, width: 1.8 }],
        tip: [{ label: "pos/s", values: speed, fmt: value => formatRate(value, "pos/s") }],
        format: value => formatDecimal(value, 0),
        latest: formatRate(histLatestNonNull(speed), "pos/s"),
      }));
    }
    // T6 Buffer pool: only when a real pool exists (hexgt bridge); the
    // synthesized dense_cnn buffer carries only loss_* keys so each key is
    // guarded individually.
    const poolFill = rows.map(row => {
      const samples = asFinite(buf(row).samples);
      const cap = asFinite(buf(row).cap);
      return samples !== null && cap !== null && cap > 0 ? samples / cap : null;
    });
    if (histSeriesCount(poolFill) >= 2) {
      charts.push(histChartSvg({
        id: "t6",
        title: "Buffer fill",
        epochs,
        domain: [0, 1],
        series: [{ label: "fill", color: "var(--accent)", values: poolFill, width: 1.8 }],
        format: value => formatPercent(value),
        latest: formatPercent(histLatestNonNull(poolFill)),
      }));
    }
    const optimism = rows.map(row => asFinite(buf(row).optimism_sum_mean));
    if (histSeriesCount(optimism) >= 2) {
      charts.push(histChartSvg({
        id: "t6b",
        title: "Value optimism",
        epochs,
        series: [{ label: "optimism", color: "#ffce56", values: optimism, width: 1.6 }],
        format: value => formatDecimal(value, 3),
        latest: formatDecimal(histLatestNonNull(optimism), 3),
      }));
    }
  }
  // T4 "SealBot eval" (legacy dense_cnn turns chart) rides evaluation_history,
  // which is ALL-NULL for shrimp (the wrapper JSON only points to the
  // multistage report). Kept ONLY for the dense_cnn lineage that still populates
  // mean_turns; for shrimp the real eval lives in T7/T8 + the Evaluation region
  // (renderHistEvalPool) below, so this never draws (histSeriesCount<2 -> skipped).
  const evals = run && Array.isArray(run.evaluation_history)
    ? run.evaluation_history.filter(row => row && asFinite(row.epoch) !== null).slice().sort((a, b) => Number(a.epoch) - Number(b.epoch))
    : [];
  const evalTurns = evals.map(row => asFinite(row.mean_turns));
  // Only draw for the dense_cnn lineage (no multistage history); on shrimp runs
  // the multistage region owns the eval surface, so suppress the dead chart.
  if (histSeriesCount(evalTurns) >= 2 && !msHistoryRows(run).length) {
    const evalEpochs = evals.map(row => Number(row.epoch));
    const evalWins = evals.map(row => asFinite(row.wins));
    const markers = evals.map(row => Number(row.wins || 0) > 0);
    let bestIdx = 0;
    evals.forEach((row, i) => {
      const best = evals[bestIdx];
      const bestTurns = Number(best.mean_turns || 0);
      const turns = Number(row.mean_turns || 0);
      if (turns > bestTurns || (turns === bestTurns && Number(row.wins || 0) > Number(best.wins || 0))) bestIdx = i;
    });
    const t4 = histChartSvg({
      id: "t4",
      title: "SealBot eval",
      epochs: evalEpochs,
      series: [{ label: "turns", color: "var(--accent)", values: evalTurns, width: 1.8, markers, markerColor: "#36d399" }],
      tip: [
        { label: "turns", values: evalTurns, fmt: value => formatDecimal(value, 1) },
        { label: "wins", values: evalWins, fmt: value => formatDecimal(value, 0) },
      ],
      annotate: { index: bestIdx, value: evalTurns[bestIdx], text: `best e${evalEpochs[bestIdx]}` },
      format: value => formatDecimal(value, 0),
      latest: `${formatDecimal(histLatestNonNull(evalTurns), 1)}t`,
    });
    if (t4) charts.splice(Math.min(3, charts.length), 0, t4);
  }
  // NOTE: the multi-stage eval Elo / win-rate trajectories (formerly T7/T8/T9)
  // now live in the dedicated, compact #histStrength panel (renderHistStrength).
  // The trends grid keeps ONLY the per-epoch selfplay/training charts so the eval
  // surface is not duplicated across two regions.
  const body = charts.filter(Boolean).join("");
  const caption = body && run && (runs || []).length > 1
    ? `<div class="hist-trends-caption">Charts: ${escapeText(run.name || "latest run")}</div>`
    : "";
  histTrends.innerHTML = body ? `${caption}${body}` : "";
}

// ===========================================================================
// #histStrength — the redesigned EVAL / STRENGTH cluster (shrimp opt-in).
// Replaces the old four-part stack (trends eval charts + rating table + giant
// BT-ladder eval pool) with ONE compact, scannable panel in three rows:
//   1. a single "current strength" hero (verdict + candidate Elo vs anchor +
//      SealBot zero-point winrate + eval epoch + Δ-Elo CI tripwire + flags),
//   2. ONE learning curve (candidate Elo vs epoch with its CI band; SealBot
//      winrate overlaid on a right scale via a second descriptive chart),
//   3. ONE opponent panel (top BT-ladder rungs + per-opponent pooled W-L).
// All data comes from the SAME readers the old regions used (multistage_eval_
// history + eval_pool + learning_health); only the presentation is rebuilt.
// Hidden whenever the run carries no multistage_eval_history.
// ---------------------------------------------------------------------------
function renderHistStrength(runs) {
  if (!histStrength) return;
  const run = histTrendRun(runs);
  const msRows = msHistoryRows(run);
  if (!msRows.length) {
    histStrength.innerHTML = "";
    histStrength.hidden = true;
    return;
  }
  const latest = msRows[msRows.length - 1];
  const candPlayer = msCandidatePlayer(latest);
  const anchorPlayer = msAnchorPlayer(latest);
  const anchorLabel = (anchorPlayer && anchorPlayer.label) || latest.anchor || "anchor";
  const verdict = latest && typeof latest.verdict === "object" && latest.verdict ? latest.verdict : {};
  const verdictLabel = latest.verdict_label || verdict.label || null;

  // ---- Row 1: current-strength hero --------------------------------------
  const heroItems = [];
  // Verdict pill leads (PROMOTE / REGRESS / INCONCLUSIVE), status-tinted.
  if (verdictLabel) {
    heroItems.push(`<div class="hist-hero-verdict ${msVerdictClass(verdictLabel)}">`
      + `<span class="hist-hero-k">verdict</span>`
      + `<span class="hist-hero-v">${escapeText(String(verdictLabel).toUpperCase())}</span></div>`);
  }
  // Candidate Elo vs the (named) zero-point anchor — the headline strength number.
  const candElo = candPlayer ? asFinite(candPlayer.elo) : null;
  if (candElo !== null) {
    const sign = candElo > 0 ? "+" : "";
    heroItems.push(`<div class="hist-hero-kpi">`
      + `<span class="hist-hero-k">Elo vs ${escapeText(anchorLabel)}</span>`
      + `<span class="hist-hero-v hist-hero-big">${sign}${Math.round(candElo)}</span>`
      + `<span class="hist-hero-sub">${escapeText(msEloText(candPlayer))}</span></div>`);
  }
  // SealBot zero-point winrate — the descriptive cross-lineage progress signal.
  const sbWr = msSealbotWinrate(latest);
  if (sbWr !== null) {
    heroItems.push(`<div class="hist-hero-kpi">`
      + `<span class="hist-hero-k">SealBot win%</span>`
      + `<span class="hist-hero-v hist-hero-big">${formatPercent(sbWr)}</span>`
      + `<span class="hist-hero-sub">zero-point</span></div>`);
  }
  // Latest eval epoch + cadence.
  heroItems.push(`<div class="hist-hero-kpi">`
    + `<span class="hist-hero-k">eval epoch</span>`
    + `<span class="hist-hero-v hist-hero-big">e${escapeText(latest.epoch)}</span>`
    + `<span class="hist-hero-sub">${escapeText(msEvalCadenceSubtitle(msRows) || `${msRows.length} reports`)}</span></div>`);
  // Δ-Elo 95% CI tripwire (compact horizontal error bar) — keeps a wide
  // INCONCLUSIVE reading as "low resolution", not a regression.
  const primary = verdict && typeof verdict.primary === "object" && verdict.primary ? verdict.primary : null;
  const eloDiff = primary ? asFinite(primary.elo_diff) : null;
  const ci = primary && Array.isArray(primary.elo_diff_ci95) ? primary.elo_diff_ci95 : null;
  const ciLo = ci ? asFinite(ci[0]) : null;
  const ciHi = ci ? asFinite(ci[1]) : null;
  if (ciLo !== null && ciHi !== null) {
    const reach = Math.max(Math.abs(ciLo), Math.abs(ciHi), eloDiff !== null ? Math.abs(eloDiff) : 0, 1);
    const toPct = v => 50 + (Math.max(Math.min(v, reach), -reach) / reach) * 50;
    const left = toPct(ciLo);
    const right = toPct(ciHi);
    const pointPct = eloDiff !== null ? toPct(eloDiff) : 50;
    const diffTxt = eloDiff !== null ? `${eloDiff > 0 ? "+" : ""}${Math.round(eloDiff)}` : "0";
    heroItems.push(`<div class="hist-hero-ci" title="Δ Elo 95% CI [${Math.round(ciLo)}, ${Math.round(ciHi)}] — gross-regression tripwire, not a fine-edge test">`
      + `<span class="hist-hero-k">Δ Elo ${escapeText(diffTxt)} <small>[${Math.round(ciLo)}, ${Math.round(ciHi)}]</small></span>`
      + `<span class="hist-ci-bar"><span class="hist-ci-track"></span>`
      + `<span class="hist-ci-zero" style="left:50%"></span>`
      + `<span class="hist-ci-range" style="left:${left.toFixed(1)}%;width:${Math.max(0, right - left).toFixed(1)}%"></span>`
      + `<span class="hist-ci-point" style="left:${pointPct.toFixed(1)}%"></span></span></div>`);
  }
  // Flags: DEGRADED anchor substitution + OOD opponents + pure-eval / gating.
  const flags = [];
  if (verdict.anchor_substituted === true || verdict.degraded === true) {
    const to = verdict.substituted_to ? ` → ${escapeText(String(verdict.substituted_to))}` : "";
    const note = verdict.degraded_note || verdict.sealbot_unavailable_reason || "anchor substituted; absolute Elo not calibrated";
    flags.push(`<span class="hist-rating-verdict hist-health-intervene" title="${escapeAttr(String(note))}">DEGRADED${to}</span>`);
  }
  const oodOpps = Array.isArray(verdict.ood_opponents) ? verdict.ood_opponents.filter(Boolean) : [];
  if (oodOpps.length) {
    const note = verdict.ood_note || `Opponents ${oodOpps.join(", ")} featurized out-of-distribution (radius mismatch); excluded from the pinned anchor.`;
    flags.push(`<span class="hist-rating-tag hist-rating-tag-muted" title="${escapeAttr(String(note))}">OOD: ${escapeText(oodOpps.join(", "))}</span>`);
  }
  const meta = latest && typeof latest.meta === "object" && latest.meta ? latest.meta : {};
  const pureEval = latest.pure_eval !== undefined ? latest.pure_eval : meta.pure_eval;
  if (pureEval === true) flags.push(`<span class="hist-rating-tag">pure eval</span>`);
  else if (pureEval === false) flags.push(`<span class="hist-rating-tag hist-rating-tag-muted">gating on</span>`);
  // PARTIAL EVAL tripwire: the concurrent checkpoint pass failed (e.g. a CUDA
  // error) after some edges already played — this epoch's ratings come from a
  // fraction of the budget and must not read as a full report.
  const stageHealth = latest && typeof latest.stage_health === "object" && latest.stage_health ? latest.stage_health : {};
  if (stageHealth.multi_checkpoint_error) {
    const played = Array.isArray(stageHealth.opponents_played) && stageHealth.opponents_played.length
      ? `only ${stageHealth.opponents_played.join(", ")} played` : "no opponents played";
    flags.push(`<span class="hist-rating-verdict hist-health-intervene" `
      + `title="${escapeAttr(`checkpoint pass failed — ${played}. ${String(stageHealth.multi_checkpoint_error).split("\n")[0]}`)}">PARTIAL EVAL</span>`);
  }
  msDroppedAnchors(latest).forEach(name => {
    flags.push(`<span class="hist-rating-tag hist-rating-tag-muted" title="permanent anchor missing from this epoch's roster">${escapeText(name)} dropped</span>`);
  });
  if (flags.length) heroItems.push(`<div class="hist-hero-flags">${flags.join("")}</div>`);
  const verdictNote = verdict && verdict.note ? `<div class="hist-hero-note">${escapeText(String(verdict.note))}</div>` : "";
  // Hero is a full-width banner spanning the whole grid (grid-column:1/-1).
  const heroBlock = `<div class="hist-strength-hero"><div class="hist-hero">${heroItems.join("")}</div>${verdictNote}</div>`;

  // ---- Row 2: ONE learning curve (candidate Elo vs epoch + CI band) -------
  const msEpochs = msRows.map(row => Number(row.epoch));
  const msElo = msRows.map(row => asFinite((msCandidatePlayer(row) || {}).elo));
  const charts = [];
  if (histSeriesCount(msElo) >= 2) {
    const ciRows = msRows.map(row => {
      const c = msCandidatePlayer(row);
      return Array.isArray(c && c.elo_ci95) ? c.elo_ci95 : null;
    });
    const bandLo = ciRows.map(c => (c ? asFinite(c[0]) : null));
    const bandHi = ciRows.map(c => (c ? asFinite(c[1]) : null));
    const t = histChartSvg({
      id: "strengthElo",
      title: `Eval Elo (${anchorLabel}=0)`,
      subtitle: msEvalCadenceSubtitle(msRows),
      epochs: msEpochs,
      series: [{ label: "candidate", color: "var(--accent)", values: msElo, width: 1.8, points: true }],
      band: histSeriesCount(bandLo) >= 2 ? { lo: bandLo, hi: bandHi, color: "rgba(39,215,230,0.12)" } : null,
      tip: [
        { label: "Elo", values: msElo, fmt: value => formatDecimal(value, 0) },
        { label: "lo", values: bandLo, fmt: value => formatDecimal(value, 0) },
        { label: "hi", values: bandHi, fmt: value => formatDecimal(value, 0) },
      ],
      format: value => formatDecimal(value, 0),
      latest: `${formatDecimal(histLatestNonNull(msElo), 0)} Elo`,
    });
    if (t) charts.push(t);
  }
  // SealBot zero-point winrate companion — the drift-immune progress signal.
  const sbWrSeries = msRows.map(row => msSealbotWinrate(row));
  if (histSeriesCount(sbWrSeries) >= 2) {
    const sbCi = msRows.map(row => {
      const e = (Array.isArray(row.edges) ? row.edges : []).find(
        x => x && String(x.opponent || x.role || "").toLowerCase() === "sealbot");
      if (e && Array.isArray(e.winrate_ci95)) return e.winrate_ci95;
      return Array.isArray(row.sealbot_winrate_ci95) ? row.sealbot_winrate_ci95 : null;
    });
    const sbLo = sbCi.map(c => (c ? asFinite(c[0]) : null));
    const sbHi = sbCi.map(c => (c ? asFinite(c[1]) : null));
    const sbHalf = msEpochs.map(() => 0.5);
    const t = histChartSvg({
      id: "strengthSealbot",
      title: "SealBot win%",
      subtitle: msEvalCadenceSubtitle(msRows),
      epochs: msEpochs,
      domain: [0, 1],
      series: [
        { label: "win rate", color: "#36d399", values: sbWrSeries, width: 1.8, points: true },
        { label: "50%", color: "var(--muted)", values: sbHalf, width: 1 },
      ],
      band: histSeriesCount(sbLo) >= 2 ? { lo: sbLo, hi: sbHi, color: "rgba(54,211,153,0.12)" } : null,
      tip: [
        { label: "win%", values: sbWrSeries, fmt: value => formatPercent(value) },
        { label: "lo", values: sbLo, fmt: value => formatPercent(value) },
        { label: "hi", values: sbHi, fmt: value => formatPercent(value) },
      ],
      format: value => formatPercent(value),
      latest: formatPercent(histLatestNonNull(sbWrSeries)),
    });
    if (t) charts.push(t);
  }
  // Each curve becomes a peer card in the grid (not a stacked two-row block).
  const curveCards = charts.map(c => `<div class="hist-strength-card">${c}</div>`).join("");

  // ---- Opponent cards: top BT rungs + per-opponent W-L (each its own card) -
  const opponentCards = msStrengthOpponentPanel(run, latest);
  // ---- Latest eval detail: every edge of the newest report ----------------
  const evalDetailCard = msEvalDetailCard(latest);

  const caption = (runs || []).length > 1
    ? `<div class="hist-trends-caption">Strength &amp; eval: ${escapeText((run && run.name) || "latest run")}</div>`
    : `<div class="hist-trends-caption">Strength &amp; eval</div>`;
  // Single responsive grid: full-width hero banner + four equal-weight cards
  // (Elo curve | SealBot curve | BT ladder | latest W-L) that auto-fit columns.
  histStrength.innerHTML = `${caption}<div class="hist-strength-grid">${heroBlock}${curveCards}${evalDetailCard}${opponentCards}</div>`;
  histStrength.hidden = false;
}

// Latest-report eval detail card: one row per EDGE of the newest multistage
// report — opponent (+role/searcher-profile tags), the physical W–L record,
// win% with its 95% CI, and the per-edge Elo point. A PARTIAL banner leads
// when the checkpoint pass failed (stage_health.multi_checkpoint_error) so a
// fraction-of-budget report can never read as a full one. Returns "" when the
// row carries no edges (pre-multistage epochs).
function msEvalDetailCard(latest) {
  const edges = Array.isArray(latest && latest.edges)
    ? latest.edges.filter(e => e && typeof e === "object")
    : [];
  const health = latest && typeof latest.stage_health === "object" && latest.stage_health ? latest.stage_health : {};
  if (!edges.length && !health.multi_checkpoint_error) return "";

  const banners = [];
  if (health.multi_checkpoint_error) {
    const firstLine = String(health.multi_checkpoint_error).split("\n")[0];
    const played = Array.isArray(health.opponents_played) && health.opponents_played.length
      ? health.opponents_played.join(", ") : "none";
    banners.push(`<div class="hist-hero-note" title="${escapeAttr(String(health.multi_checkpoint_error))}">`
      + `<span class="hist-rating-verdict hist-health-intervene">PARTIAL</span> `
      + `checkpoint pass failed (${escapeText(firstLine)}); played: ${escapeText(played)}</div>`);
  }
  if (health.sealbot_unavailable) {
    banners.push(`<div class="hist-hero-note">SealBot unavailable: ${escapeText(String(health.sealbot_unavailable))}</div>`);
  }

  const rows = edges.map(e => {
    const wa = asFinite(e.wins_a);
    const wb = asFinite(e.wins_b);
    const rec = wa !== null && wb !== null ? `${Math.round(wa)}–${Math.round(wb)}` : "—";
    const lead = wa !== null && wb !== null
      ? (wa > wb ? "hist-eval-win" : wa < wb ? "hist-eval-loss" : "hist-eval-draw")
      : "hist-eval-draw";
    const wr = asFinite(e.winrate);
    const ci = Array.isArray(e.winrate_ci95) ? e.winrate_ci95 : null;
    const ciTxt = ci && asFinite(ci[0]) !== null && asFinite(ci[1]) !== null
      ? ` [${formatPercent(asFinite(ci[0]))}–${formatPercent(asFinite(ci[1]))}]` : "";
    const wrTxt = wr !== null ? `${formatPercent(wr)}${ciTxt}` : "";
    const eloPt = asFinite(e.elo_point);
    const eloTxt = eloPt !== null ? `${eloPt > 0 ? "+" : ""}${Math.round(eloPt)}` : "";
    const tags = [];
    if (e.primary) tags.push(`<span class="hist-rating-tag">primary</span>`);
    else if (e.role && e.role !== "checkpoint") tags.push(`<span class="hist-rating-tag hist-rating-tag-muted">${escapeText(String(e.role))}</span>`);
    if (e.opponent_search_profile) {
      tags.push(`<span class="hist-rating-tag hist-rating-tag-muted" `
        + `title="searcher the OPPONENT side used (every opponent, including foreign anchors, is evaluated under the candidate's self-play profile)">`
        + `${escapeText(String(e.opponent_search_profile))}</span>`);
    }
    const games = asFinite(e.decided);
    const gamesTxt = games !== null ? `<span class="hist-hero-sub">${Math.round(games)}g</span>` : "";
    return `<div class="hist-loss-row hist-eval-pair">`
      + `<span class="hist-loss-label" title="${escapeAttr(String(e.opponent || ""))}">${escapeText(String(e.opponent || "?"))}${tags.join("")}</span>`
      + `<span class="hist-eval-record ${lead}" title="physical wins (candidate–opponent)">${rec}</span>`
      + `<span class="hist-hero-sub" title="win rate [95% CI]">${wrTxt}</span>`
      + `<span class="hist-hero-sub" title="per-edge Elo point estimate">${eloTxt}</span>`
      + `${gamesTxt}</div>`;
  }).join("");

  const bits = [];
  const visits = asFinite(latest.full_search_visits);
  if (visits !== null) bits.push(`${Math.round(visits)} visits`);
  const elapsed = asFinite(latest.elapsed_seconds);
  if (elapsed !== null) bits.push(`${Math.round(elapsed)}s`);
  const sub = bits.length ? ` · ${bits.join(" · ")}` : "";
  return `<div class="hist-strength-card"><div class="hist-insp-group">`
    + `<span class="hist-insp-group-title">Eval detail · e${escapeText(latest.epoch)}${escapeText(sub)}</span>`
    + `${banners.join("")}${rows}</div></div>`;
}

// The compact opponent panel for #histStrength: the TOP rungs of the unified BT
// ladder (checkpoints collapsed to one row each, anchors kept) beside the latest
// per-opponent pooled W-L — replacing the old giant full-ladder bar chart + the
// separate W-L table. Pulls from the same run.eval_pool + latest report ratings.
function msStrengthOpponentPanel(run, latest) {
  const pool = run && typeof run.eval_pool === "object" && run.eval_pool ? run.eval_pool : null;
  const edges = pool && Array.isArray(pool.edges) ? pool.edges.filter(e => e && typeof e === "object") : [];
  const players = msPlayers(latest);
  const anchorName = String((pool && pool.anchor) || latest.anchor || "sealbot");

  // --- Unified BT ladder (top rungs) -------------------------------------
  const eloOf = label => { const p = players.find(x => x && x.label === label); return p ? asFinite(p.elo) : null; };
  const ciOf = label => { const p = players.find(x => x && x.label === label); return p && Array.isArray(p.elo_ci95) ? p.elo_ci95 : null; };
  const epOfLabel = label => { const m = /^(?:cand_)?ep(\d+)$/.exec(String(label || "")); return m ? Number(m[1]) : null; };
  const ckptEpochs = new Set();
  players.forEach(p => { const e = epOfLabel(p.label); if (e !== null) ckptEpochs.add(e); });
  const unified = [];
  ckptEpochs.forEach(ep => {
    const candElo = eloOf(`cand_ep${ep}`);
    const oppElo = eloOf(`ep${ep}`);
    const shown = candElo !== null ? candElo : oppElo;
    if (shown === null) return;
    const split = candElo !== null && oppElo !== null ? candElo - oppElo : null;
    unified.push({ key: `ep${ep}`, label: `epoch ${ep}`, elo: shown, ci: ciOf(`cand_ep${ep}`) || ciOf(`ep${ep}`), split, epoch: ep, isAnchor: false });
  });
  const anchors = players
    .filter(p => epOfLabel(p.label) === null)
    .map(p => ({ key: p.label, label: p.label, elo: asFinite(p.elo), ci: Array.isArray(p.elo_ci95) ? p.elo_ci95 : null, split: null, epoch: null, isAnchor: !!p.is_anchor }));
  const ladderItems = unified.concat(anchors)
    .filter(it => it.elo !== null)
    .sort((a, b) => b.elo - a.elo);
  // Compact: top 5 rungs, but always keep the anchor row visible for the 0-point.
  let shown = ladderItems.slice(0, 5);
  const anchorItem = ladderItems.find(it => it.isAnchor);
  if (anchorItem && !shown.includes(anchorItem)) shown = shown.concat(anchorItem);
  let ladderBlock = "";
  if (shown.length) {
    const elos = ladderItems.map(it => it.elo);
    const maxElo = Math.max(...elos);
    const minElo = Math.min(...elos);
    const span = maxElo - minElo;
    const rows = shown.map(it => {
      const pct = span > 0 ? Math.max(4, Math.round(((it.elo - minElo) / span) * 100)) : 50;
      const anchorTag = it.isAnchor ? `<span class="hist-rating-anchor" title="${escapeAttr(anchorName)} anchor (pinned 0 Elo)">⚓</span>` : "";
      const ciTxt = Array.isArray(it.ci) && asFinite(it.ci[0]) !== null && asFinite(it.ci[1]) !== null
        ? ` [${Math.round(asFinite(it.ci[0]))}–${Math.round(asFinite(it.ci[1]))}]` : "";
      const splitTxt = it.split !== null && Math.abs(it.split) >= 1
        ? `<span class="hist-eval-split" title="cand-vs-replay Elo split for the SAME checkpoint (BT single-node artifact)">Δ${it.split > 0 ? "+" : ""}${Math.round(it.split)}</span>` : "";
      const replay = it.epoch !== null ? histEvalReplayLink(it.epoch, "↪") : "<span></span>";
      return `<div class="hist-loss-row hist-eval-ladder-row">`
        + `<span class="hist-loss-label" title="${escapeAttr(it.label)}">${escapeText(it.label)}${anchorTag}${splitTxt}</span>`
        + `<span class="hist-loss-bar"><span class="hist-loss-bar-fill" style="width:${pct}%"></span><span class="hist-loss-val">${Math.round(it.elo)}${escapeText(ciTxt)}</span></span>`
        + `${replay}</div>`;
    }).join("");
    const more = ladderItems.length > shown.length ? `<span class="hist-strength-more">+${ladderItems.length - shown.length} more rungs</span>` : "";
    ladderBlock = `<div class="hist-strength-card"><div class="hist-insp-group"><span class="hist-insp-group-title">BT ladder (top) · anchor ${escapeText(anchorName)} = 0</span>${rows}${more}</div></div>`;
  }

  // --- Per-opponent pooled W-L (latest epoch's matchups) ------------------
  // Group raw head-to-head by unordered pair; show only the LATEST epoch's
  // matchups (the most recent candidate) so the panel stays short.
  let matrixBlock = "";
  if (edges.length) {
    const pairs = new Map();
    edges.forEach(e => {
      const a = String(e.a || ""); const b = String(e.b || "");
      if (!a || !b) return;
      const key = [a, b].join(" ↔ ");
      let rec = pairs.get(key);
      if (!rec) { rec = { a, b, wa: 0, wb: 0, epochs: [], kind: e.kind, weight: e.weight }; pairs.set(key, rec); }
      const raw = e.raw && typeof e.raw === "object" ? e.raw : {};
      const wa = asFinite(raw.physical_wins_a);
      const wb = asFinite(raw.physical_wins_b);
      const wcand = asFinite(raw.physical_wins_cand);
      const wsb = asFinite(raw.physical_wins_sealbot);
      if (wa !== null && wb !== null) { rec.wa += wa; rec.wb += wb; }
      else if (wcand !== null && wsb !== null) { rec.wa += wcand; rec.wb += wsb; }
      else { rec.wa += asFinite(e.wins_a) || 0; rec.wb += asFinite(e.wins_b) || 0; }
      const ep = asFinite(e.epoch);
      if (ep !== null) rec.epochs.push(ep);
    });
    const recs = Array.from(pairs.values());
    const latestEp = recs.reduce((m, r) => Math.max(m, r.epochs.length ? Math.max(...r.epochs) : -Infinity), -Infinity);
    const pairRows = recs
      .filter(rec => Number.isFinite(latestEp) ? (rec.epochs.length && Math.max(...rec.epochs) === latestEp) : true)
      .sort((x, y) => Math.max(...y.epochs, 0) - Math.max(...x.epochs, 0))
      .map(rec => {
        const wa = Math.round(rec.wa); const wb = Math.round(rec.wb);
        const lead = wa > wb ? "hist-eval-win" : wa < wb ? "hist-eval-loss" : "hist-eval-draw";
        const kindBadge = rec.kind === "sealbot" ? `<span class="hist-rating-tag hist-rating-tag-muted">SealBot ×${formatDecimal(rec.weight, 1)}</span>` : "";
        const epLast = rec.epochs.length ? Math.max(...rec.epochs) : null;
        const replay = epLast !== null ? histEvalReplayLink(epLast, "↪") : "";
        return `<div class="hist-loss-row hist-eval-pair">`
          + `<span class="hist-loss-label" title="${escapeAttr(rec.a + " vs " + rec.b)}">${escapeText(rec.a)} vs ${escapeText(rec.b)}</span>`
          + `<span class="hist-eval-record ${lead}">${wa}–${wb}</span>${kindBadge}${replay}</div>`;
      }).join("");
    if (pairRows) {
      const epTxt = Number.isFinite(latestEp) ? ` · e${latestEp}` : "";
      matrixBlock = `<div class="hist-strength-card"><div class="hist-insp-group"><span class="hist-insp-group-title">Latest W–L (pooled head-to-head${epTxt})</span>${pairRows}</div></div>`;
    }
  }

  // Each block is already wrapped as a peer card; the parent grid lays them out.
  return [ladderBlock, matrixBlock].filter(Boolean).join("");
}

// SUPERSEDED by renderHistStrength (the compact #histStrength panel). No longer
// dispatched; #histRatingTable stays hidden/empty. Kept only as reference for the
// verdict-chip / Δ-Elo-CI / rating-row logic now folded into the hero + ladder.
// Standalone "Multi-stage eval" region (shrimp opt-in): the SealBot-pinned Elo
// RATING TABLE from the newest report's ratings.players, the latest PROMOTE/
// REGRESS/INCONCLUSIVE VERDICT (+ note + pure-eval indicator), and the headline
// EDGES (vs SealBot / vs BC-prefit / vs champion winrate + CI). Hidden whenever
// the run carries no multistage_eval_history (non-shrimp + pre-multistage runs).
function renderHistRatingTable(runs) {
  if (!histRatingTable) return;
  const run = histTrendRun(runs);
  const rows = msHistoryRows(run);
  if (!rows.length) {
    histRatingTable.innerHTML = "";
    histRatingTable.hidden = true;
    return;
  }
  const latest = rows[rows.length - 1];
  const players = msPlayers(latest);
  // The pool anchor is dynamic (SealBot when available, else bc_prefit) — read it
  // from the data, never hard-code "SealBot" (the labels were misleading once the
  // anchor fell back to bc_prefit).
  const anchorPlayer = msAnchorPlayer(latest);
  const anchorLabel = (anchorPlayer && anchorPlayer.label) || latest.anchor || "anchor";
  const candPlayer = msCandidatePlayer(latest);

  // --- Verdict chip + note + pure-eval indicator ---
  const verdict = latest && typeof latest.verdict === "object" && latest.verdict ? latest.verdict : {};
  const verdictLabel = latest.verdict_label || verdict.label || null;
  const verdictChips = [];
  // Lead with the candidate's pooled Elo + CI — the headline "how strong is it" number.
  if (candPlayer) verdictChips.push(epochChip("Elo", msEloText(candPlayer)));
  if (verdictLabel) {
    verdictChips.push(`<span class="hist-rating-verdict ${msVerdictClass(verdictLabel)}">${escapeText(String(verdictLabel).toUpperCase())}</span>`);
  }
  // E3: SealBot died mid-eval and the BT zero-point silently re-pinned — surface
  // it as a DEGRADED chip so the (still-emitted) verdict is not over-read; every
  // absolute Elo shifted under it.
  if (verdict.anchor_substituted === true || verdict.degraded === true) {
    const to = verdict.substituted_to ? ` → ${escapeText(String(verdict.substituted_to))}` : "";
    const note = verdict.degraded_note || verdict.sealbot_unavailable_reason || "anchor substituted; absolute Elo not calibrated";
    verdictChips.push(`<span class="hist-rating-verdict hist-health-intervene" title="${escapeAttr(String(note))}">DEGRADED · anchor sub${to}</span>`);
  }
  // E4: radius-8-era opponents featurized OOD at the live radius inflate the
  // candidate's cross-lineage Elo — flag it so the ladder is not read as a clean
  // strength signal.
  const oodOpps = Array.isArray(verdict.ood_opponents) ? verdict.ood_opponents.filter(Boolean) : [];
  if (oodOpps.length) {
    const note = verdict.ood_note || `Opponents ${oodOpps.join(", ")} featurized out-of-distribution (radius mismatch); excluded from the pinned anchor.`;
    verdictChips.push(`<span class="hist-rating-tag hist-rating-tag-muted" title="${escapeAttr(String(note))}">OOD: ${escapeText(oodOpps.join(", "))}</span>`);
  }
  // pure_eval lives on the per-epoch report meta; tolerate it on the row or its meta.
  const meta = latest && typeof latest.meta === "object" && latest.meta ? latest.meta : {};
  const pureEval = latest.pure_eval !== undefined ? latest.pure_eval : meta.pure_eval;
  if (pureEval === true) verdictChips.push(`<span class="hist-rating-tag">pure eval</span>`);
  else if (pureEval === false) verdictChips.push(`<span class="hist-rating-tag hist-rating-tag-muted">gating on</span>`);
  const primary = verdict && typeof verdict.primary === "object" && verdict.primary ? verdict.primary : null;
  const eloDiff = primary ? asFinite(primary.elo_diff) : null;
  if (eloDiff !== null) {
    const sign = eloDiff > 0 ? "+" : "";
    verdictChips.push(epochChip("Δ Elo", `${sign}${Math.round(eloDiff)}`));
  }
  // D3-A: render the Δ-Elo CI as a horizontal error bar so a single wide-CI
  // INCONCLUSIVE (e.g. [-270, +269]) reads as "low resolution", not a real
  // regression. The bar spans a symmetric [-max,+max] window with a 0 reference.
  const ci = primary && Array.isArray(primary.elo_diff_ci95) ? primary.elo_diff_ci95 : null;
  const ciLo = ci ? asFinite(ci[0]) : null;
  const ciHi = ci ? asFinite(ci[1]) : null;
  let ciBar = "";
  if (ciLo !== null && ciHi !== null) {
    const reach = Math.max(Math.abs(ciLo), Math.abs(ciHi), eloDiff !== null ? Math.abs(eloDiff) : 0, 1);
    const toPct = v => 50 + (Math.max(Math.min(v, reach), -reach) / reach) * 50;
    const left = toPct(ciLo);
    const right = toPct(ciHi);
    const pointPct = eloDiff !== null ? toPct(eloDiff) : 50;
    ciBar = `<div class="hist-ci-bar" title="Δ Elo 95% CI [${Math.round(ciLo)}, ${Math.round(ciHi)}]">`
      + `<span class="hist-ci-track"></span>`
      + `<span class="hist-ci-zero" style="left:50%"></span>`
      + `<span class="hist-ci-range" style="left:${left.toFixed(1)}%;width:${Math.max(0, right - left).toFixed(1)}%"></span>`
      + `<span class="hist-ci-point" style="left:${pointPct.toFixed(1)}%"></span>`
      + `</div><div class="hist-ci-caption">Δ Elo 95% CI [${Math.round(ciLo)}, ${Math.round(ciHi)}] — gross-regression tripwire, not a fine-edge test</div>`;
  }
  const verdictNote = verdict && verdict.note ? `<div class="hist-insp-note">${escapeText(String(verdict.note))}</div>` : "";
  const verdictBlock = verdictChips.length || verdictNote || ciBar
    ? `<div class="hist-rating-head"><span class="hist-insp-group-title">Latest verdict · epoch ${escapeText(latest.epoch)}</span><div class="hist-rating-verdict-row">${verdictChips.join("")}</div>${ciBar}${verdictNote}</div>`
    : "";

  // --- Rating table (one row per checkpoint, Elo-descending, anchor marked) ---
  let ratingBlock = "";
  if (players.length) {
    const elos = players.map(p => asFinite(p.elo)).filter(v => v !== null);
    const maxElo = elos.length ? Math.max(...elos) : 0;
    const minElo = elos.length ? Math.min(...elos) : 0;
    const span = maxElo - minElo;
    const ratingRows = players.map(player => {
      const elo = asFinite(player.elo);
      // Bar fills proportionally within the [min,max] window; flat 50% when the
      // pool has a single Elo level (span 0) so the bar still reads.
      const pct = elo === null ? 0 : span > 0 ? Math.max(4, Math.round(((elo - minElo) / span) * 100)) : 50;
      const anchorTag = player.is_anchor ? `<span class="hist-rating-anchor" title="${escapeAttr(anchorLabel)} anchor (pinned 0 Elo)">⚓</span>` : "";
      const label = escapeText(displayValue(player.label, "?"));
      return `<div class="hist-loss-row hist-rating-row">
        <span class="hist-loss-label" title="${escapeAttr(displayValue(player.label, "?"))}">${label}${anchorTag}</span>
        <span class="hist-loss-bar"><span class="hist-loss-bar-fill" style="width:${pct}%"></span><span class="hist-loss-val">${escapeText(msEloText(player))}</span></span>
      </div>`;
    }).join("");
    ratingBlock = `<div class="hist-insp-group"><span class="hist-insp-group-title">Pooled Elo · anchor ${escapeText(anchorLabel)} = 0</span>${ratingRows}</div>`;
  } else {
    // Fit failed (degraded ratings shape) but the report still exists — say so
    // rather than silently dropping the whole panel.
    const fit = latest.ratings && typeof latest.ratings === "object" ? latest.ratings.fit : null;
    const reason = fit && fit.error ? String(fit.error) : "rating fit unavailable";
    ratingBlock = `<div class="hist-insp-group"><span class="hist-insp-group-title">Pooled Elo · anchor ${escapeText(anchorLabel)} = 0</span><div class="hist-insp-note">${escapeText(reason)}</div></div>`;
  }

  // --- Headline edges (vs SealBot / vs BC-prefit / vs champion) ---
  // D3-B: roster-driven. A configured PERMANENT anchor that is absent from this
  // epoch's roster (the live bc_prefit drop, SEV-2) renders a muted "not in
  // roster" pill instead of silently vanishing.
  let edgesBlock = "";
  const headlineEdges = msHeadlineEdges(latest);
  const edgeChips = headlineEdges.map(edge => {
    const decided = asFinite(edge.decided);
    const suffix = decided !== null ? ` (${decided})` : "";
    return epochChip(msEdgeLabel(edge), `${msWinrateText(edge)}${suffix}`);
  });
  msDroppedAnchors(latest).forEach(name => {
    edgeChips.push(`<span class="hist-epoch-chip-stat hist-epoch-chip-muted" title="permanent anchor missing from this epoch's roster">vs ${escapeText(name)} — not in roster</span>`);
  });
  if (edgeChips.length) {
    edgesBlock = `<div class="hist-insp-group"><span class="hist-insp-group-title">Headline edges</span><div class="hist-insp-chips">${edgeChips.join("")}</div></div>`;
  }

  const caption = (runs || []).length > 1
    ? `<div class="hist-trends-caption">Multi-stage eval: ${escapeText(run && run.name || "latest run")}</div>`
    : `<div class="hist-trends-caption">Multi-stage eval</div>`;
  const groups = [verdictBlock, ratingBlock, edgesBlock].filter(Boolean).join("");
  histRatingTable.innerHTML = `${caption}<div class="hist-rating-groups">${groups}</div>`;
  histRatingTable.hidden = false;
}

// Deep-link a ladder/W-L row into the History list filtered to source=evaluation
// + the row's epoch, reusing the existing source select + epoch chip. Eval
// replays (once the .hxr carry records) are then one click away. data-* driven
// via the delegated #histEvalPool listener below.
function histEvalReplayLink(epoch, label) {
  const ep = asFinite(epoch);
  if (ep === null) return "";
  return `<button type="button" class="hist-eval-replay" data-hist-eval-epoch="${escapeAttr(ep)}"`
    + ` title="Show evaluation replays for epoch ${escapeAttr(ep)}">${escapeText(label || "replays")}</button>`;
}

// SUPERSEDED by msStrengthOpponentPanel (folded into #histStrength). No longer
// dispatched; #histEvalPool stays hidden/empty. The compact panel reuses the same
// unified-ladder + pooled-W-L logic, trimmed to the top rungs / latest matchups.
// D2 + D3-C/E/F: the "Evaluation" region consuming the append-only Bradley-Terry
// pool (run.eval_pool, 29 edges with raw head-to-head wins) + the latest report's
// fitted ratings. Three blocks: a BT ladder UNIFIED by checkpoint identity (the
// cand_epN / epN split surfaced as a delta, not two bots), a per-opponent W-L
// matrix from the pool edges, and a per-epoch verdict-history strip. Each row
// deep-links into the History list filtered to source=evaluation + that epoch.
// Hidden when run.eval_pool is null/empty (non-shrimp / older runs).
function renderHistEvalPool(runs) {
  if (!histEvalPool) return;
  const run = histTrendRun(runs);
  const pool = run && typeof run.eval_pool === "object" && run.eval_pool ? run.eval_pool : null;
  const edges = pool && Array.isArray(pool.edges) ? pool.edges.filter(e => e && typeof e === "object") : [];
  const msRows = msHistoryRows(run);
  if (!edges.length || !msRows.length) {
    histEvalPool.innerHTML = "";
    histEvalPool.hidden = true;
    return;
  }
  const latest = msRows[msRows.length - 1];
  const players = msPlayers(latest);
  const anchorName = String(pool.anchor || latest.anchor || "sealbot");

  // --- Block C: BT ladder unified by checkpoint identity ---------------------
  // Each report rates a checkpoint TWICE (cand_epN as the live candidate, epN as
  // a replayed opponent). Collapse them to one "epoch N" row carrying both Elo
  // readings; show the split delta so the ~69-Elo artifact is explained, not
  // read as two distinct bots. Non-checkpoint nodes (sealbot, bc_prefit) keep
  // their own row.
  const eloOf = label => {
    const p = players.find(x => x && x.label === label);
    return p ? asFinite(p.elo) : null;
  };
  const ciOf = label => {
    const p = players.find(x => x && x.label === label);
    return p && Array.isArray(p.elo_ci95) ? p.elo_ci95 : null;
  };
  const epOfLabel = label => {
    const m = /^(?:cand_)?ep(\d+)$/.exec(String(label || ""));
    return m ? Number(m[1]) : null;
  };
  // Gather the distinct checkpoint epochs present in the ladder.
  const ckptEpochs = new Set();
  players.forEach(p => { const e = epOfLabel(p.label); if (e !== null) ckptEpochs.add(e); });
  const unified = [];
  ckptEpochs.forEach(ep => {
    const candElo = eloOf(`cand_ep${ep}`);
    const oppElo = eloOf(`ep${ep}`);
    const elos = [candElo, oppElo].filter(v => v !== null);
    if (!elos.length) return;
    const shown = candElo !== null ? candElo : oppElo;  // prefer the live-candidate reading
    const split = candElo !== null && oppElo !== null ? candElo - oppElo : null;
    unified.push({ epoch: ep, elo: shown, ci: ciOf(`cand_ep${ep}`) || ciOf(`ep${ep}`), split });
  });
  // Non-checkpoint anchors (sealbot pinned 0, bc_prefit, etc.).
  const anchors = players
    .filter(p => epOfLabel(p.label) === null)
    .map(p => ({ label: p.label, elo: asFinite(p.elo), ci: Array.isArray(p.elo_ci95) ? p.elo_ci95 : null, isAnchor: !!p.is_anchor }));
  const ladderItems = unified
    .map(u => ({ key: `ep${u.epoch}`, label: `epoch ${u.epoch}`, elo: u.elo, ci: u.ci, split: u.split, epoch: u.epoch, isAnchor: false }))
    .concat(anchors.map(a => ({ key: a.label, label: a.label, elo: a.elo, ci: a.ci, split: null, epoch: epOfLabel(a.label), isAnchor: a.isAnchor })))
    .filter(it => it.elo !== null)
    .sort((a, b) => b.elo - a.elo);
  let ladderBlock = "";
  if (ladderItems.length) {
    const elos = ladderItems.map(it => it.elo);
    const maxElo = Math.max(...elos);
    const minElo = Math.min(...elos);
    const span = maxElo - minElo;
    const rows = ladderItems.map(it => {
      const pct = span > 0 ? Math.max(4, Math.round(((it.elo - minElo) / span) * 100)) : 50;
      const anchorTag = it.isAnchor ? `<span class="hist-rating-anchor" title="${escapeAttr(anchorName)} anchor (pinned 0 Elo)">⚓</span>` : "";
      const ciTxt = Array.isArray(it.ci) && asFinite(it.ci[0]) !== null && asFinite(it.ci[1]) !== null
        ? ` [${Math.round(asFinite(it.ci[0]))}–${Math.round(asFinite(it.ci[1]))}]` : "";
      const splitTxt = it.split !== null && Math.abs(it.split) >= 1
        ? `<span class="hist-eval-split" title="cand-vs-replay Elo split for the SAME checkpoint (BT single-node artifact)">Δ${it.split > 0 ? "+" : ""}${Math.round(it.split)}</span>` : "";
      const replay = it.epoch !== null ? histEvalReplayLink(it.epoch, "replays") : "";
      return `<div class="hist-loss-row hist-eval-ladder-row">
        <span class="hist-loss-label" title="${escapeAttr(it.label)}">${escapeText(it.label)}${anchorTag}${splitTxt}</span>
        <span class="hist-loss-bar"><span class="hist-loss-bar-fill" style="width:${pct}%"></span><span class="hist-loss-val">${Math.round(it.elo)}${escapeText(ciTxt)}</span></span>
        ${replay || "<span></span>"}
      </div>`;
    }).join("");
    ladderBlock = `<div class="hist-insp-group"><span class="hist-insp-group-title">BT ladder · anchor ${escapeText(anchorName)} = 0 (checkpoints unified)</span>${rows}</div>`;
  }

  // --- Block: per-opponent W-L matrix from the pool's raw head-to-head --------
  // Group edges by the unordered pair (a,b), summing the PHYSICAL win counts from
  // each edge's raw block (the top-level wins_a/wins_b are n_eff-weighted). Latest
  // epoch first; sealbot edges badged + weight shown for the down-weighted ones.
  const pairs = new Map();
  edges.forEach(e => {
    const a = String(e.a || ""); const b = String(e.b || "");
    if (!a || !b) return;
    const key = [a, b].join(" ↔ ");
    let rec = pairs.get(key);
    if (!rec) { rec = { a, b, wa: 0, wb: 0, epochs: [], kind: e.kind, weight: e.weight }; pairs.set(key, rec); }
    const raw = e.raw && typeof e.raw === "object" ? e.raw : {};
    const wa = asFinite(raw.physical_wins_a);
    const wb = asFinite(raw.physical_wins_b);
    const wcand = asFinite(raw.physical_wins_cand);
    const wsb = asFinite(raw.physical_wins_sealbot);
    if (wa !== null && wb !== null) { rec.wa += wa; rec.wb += wb; }
    else if (wcand !== null && wsb !== null) { rec.wa += wcand; rec.wb += wsb; }
    else { rec.wa += asFinite(e.wins_a) || 0; rec.wb += asFinite(e.wins_b) || 0; }
    const ep = asFinite(e.epoch);
    if (ep !== null) rec.epochs.push(ep);
  });
  const pairRows = Array.from(pairs.values())
    .sort((x, y) => Math.max(...y.epochs, 0) - Math.max(...x.epochs, 0))
    .map(rec => {
      const wa = Math.round(rec.wa); const wb = Math.round(rec.wb);
      const lead = wa > wb ? "hist-eval-win" : wa < wb ? "hist-eval-loss" : "hist-eval-draw";
      const kindBadge = rec.kind === "sealbot" ? `<span class="hist-rating-tag hist-rating-tag-muted">SealBot ×${formatDecimal(rec.weight, 1)}</span>` : "";
      const epLast = rec.epochs.length ? Math.max(...rec.epochs) : null;
      const epTxt = epLast !== null ? ` <span class="hist-eval-ep">e${epLast}</span>` : "";
      const replay = epLast !== null ? histEvalReplayLink(epLast, "↪") : "";
      return `<div class="hist-loss-row hist-eval-pair">
        <span class="hist-loss-label" title="${escapeAttr(rec.a + " vs " + rec.b)}">${escapeText(rec.a)} vs ${escapeText(rec.b)}${epTxt}</span>
        <span class="hist-eval-record ${lead}">${wa}–${wb}</span>${kindBadge}${replay}
      </div>`;
    }).join("");
  const matrixBlock = pairRows
    ? `<div class="hist-insp-group"><span class="hist-insp-group-title">Per-opponent W–L (pooled head-to-head)</span>${pairRows}</div>`
    : "";

  // --- Block E: verdict-history strip (ep5 REGRESS, ep10..35 INCONCLUSIVE) ----
  const stripChips = msRows.map(row => {
    const lbl = String(row.verdict_label || (row.verdict || {}).label || "").toUpperCase();
    if (!lbl) return "";
    return `<span class="hist-eval-verdict ${msVerdictClass(lbl)}" title="epoch ${escapeAttr(row.epoch)}: ${escapeText(lbl)}">`
      + `e${escapeText(row.epoch)} ${escapeText(lbl.slice(0, 5))}</span>`;
  }).filter(Boolean).join("");
  const stripBlock = stripChips
    ? `<div class="hist-insp-group"><span class="hist-insp-group-title">Verdict history</span><div class="hist-eval-strip">${stripChips}</div></div>`
    : "";

  const caption = (runs || []).length > 1
    ? `<div class="hist-trends-caption">Evaluation pool: ${escapeText(run && run.name || "latest run")}</div>`
    : `<div class="hist-trends-caption">Evaluation</div>`;
  const blocks = [ladderBlock, matrixBlock, stripBlock].filter(Boolean).join("");
  if (!blocks) {
    histEvalPool.innerHTML = "";
    histEvalPool.hidden = true;
    return;
  }
  histEvalPool.innerHTML = `${caption}<div class="hist-rating-groups">${blocks}</div>`;
  histEvalPool.hidden = false;
}

// Toggle-semantics client-side epoch filter (chart click / epoch-table click /
// chip clear all route through here). Display-only: the server pager has no
// epoch parameter, so filtering happens on already-loaded items (H7).
function setHistEpochFilter(epoch) {
  histEpochFilter = histEpochFilter === epoch ? null : epoch;
  renderGameHistoryPage();
}

// H7: removable "Epoch N ×" chip in .history-filters. Render-only — the state
// lives in histEpochFilter; clicks are handled by the delegated chip listener.
function renderHistEpochChip() {
  if (!histEpochChip) return;
  if (histEpochFilter === null) {
    histEpochChip.innerHTML = "";
    histEpochChip.hidden = true;
    return;
  }
  histEpochChip.innerHTML = `Epoch ${escapeText(histEpochFilter)}`
    + `<button type="button" data-hist-epoch-clear aria-label="Clear epoch filter">×</button>`;
  histEpochChip.hidden = false;
}

function ensureHistTrendTip() {
  if (histTrendTipEl) return histTrendTipEl;
  histTrendTipEl = document.createElement("div");
  histTrendTipEl.id = "histTrendTip";
  histTrendTipEl.className = "hist-trend-tip";
  histTrendTipEl.hidden = true;
  document.body.appendChild(histTrendTipEl);
  return histTrendTipEl;
}

// Pointer -> (chart, nearest epoch index) using the stored x0/dx geometry.
function histChartFromEvent(event) {
  const svgEl = event.target && event.target.closest ? event.target.closest("svg[data-hist-chart]") : null;
  if (!svgEl) return null;
  const chart = histTrendCharts[svgEl.dataset.histChart];
  if (!chart || !chart.epochs.length) return null;
  const rect = svgEl.getBoundingClientRect();
  if (!rect.width) return null;
  const xView = ((event.clientX - rect.left) / rect.width) * HIST_CHART_W;
  const idx = chart.dx > 0
    ? Math.max(0, Math.min(chart.epochs.length - 1, Math.round((xView - chart.x0) / chart.dx)))
    : 0;
  return { svgEl, chart, idx };
}

function hideHistTrendHover() {
  if (histTrendHoverSvg) {
    const cross = histTrendHoverSvg.querySelector(".hist-trend-cross");
    if (cross) cross.setAttribute("visibility", "hidden");
    histTrendHoverSvg = null;
  }
  if (histTrendTipEl) histTrendTipEl.hidden = true;
}

function handleHistTrendsMove(event) {
  const hit = histChartFromEvent(event);
  if (!hit) {
    hideHistTrendHover();
    return;
  }
  const { svgEl, chart, idx } = hit;
  if (histTrendHoverSvg && histTrendHoverSvg !== svgEl) {
    const prev = histTrendHoverSvg.querySelector(".hist-trend-cross");
    if (prev) prev.setAttribute("visibility", "hidden");
  }
  histTrendHoverSvg = svgEl;
  const x = (chart.x0 + idx * chart.dx).toFixed(1);
  const cross = svgEl.querySelector(".hist-trend-cross");
  if (cross) {
    cross.setAttribute("x1", x);
    cross.setAttribute("x2", x);
    cross.setAttribute("visibility", "visible");
  }
  const tipEl = ensureHistTrendTip();
  const lines = [`e${chart.epochs[idx]}`];
  chart.series.forEach(series => {
    const value = series.values[idx];
    lines.push(`${series.label}: ${value === null || value === undefined ? "--" : series.fmt(value)}`);
  });
  tipEl.textContent = lines.join("\n");
  tipEl.hidden = false;
  const pad = 12;
  let left = event.clientX + pad;
  let top = event.clientY + pad;
  if (left + tipEl.offsetWidth + 8 > window.innerWidth) left = Math.max(8, event.clientX - tipEl.offsetWidth - pad);
  if (top + tipEl.offsetHeight + 8 > window.innerHeight) top = Math.max(8, event.clientY - tipEl.offsetHeight - pad);
  tipEl.style.left = `${left}px`;
  tipEl.style.top = `${top}px`;
}

function handleHistTrendsClick(event) {
  const hit = histChartFromEvent(event);
  if (!hit) return;
  event.preventDefault();
  const epoch = asFinite(hit.chart.epochs[hit.idx]);
  if (epoch === null) return;
  setHistEpochFilter(epoch);
}

// --- History v2 Region 3: dense epoch table (#histEpochTable). ---
// Last 15 epochs across the loaded runs (newest first), one dense row each:
// epoch button (filters the games list), selfplay summary, win-balance bar,
// train loss, eval record, checkpoint tag, and a chevron that expands the
// existing epochProgressDetail buffer/losses band. The row whose epoch the
// newest run status is currently executing gets a live accent. Expansion
// state lives in histExpandedEpochs ("<run>::<epoch>") across re-renders.
function renderHistEpochTable(runs) {
  if (!histEpochTable) return;
  const entries = (runs || []).flatMap(run =>
    (run && Array.isArray(run.epoch_history) ? run.epoch_history : [])
      .filter(row => row && asFinite(row.epoch) !== null)
      .map(row => ({ run: (run && run.name) || "", row })));
  if (!entries.length) {
    histEpochTable.innerHTML = "";
    return;
  }
  entries.sort((a, b) =>
    (Number(a.row.epoch) - Number(b.row.epoch)) || String(a.run).localeCompare(String(b.run)));
  const recent = entries.slice(-15).reverse();
  const multiRun = new Set(recent.map(entry => entry.run)).size > 1;
  // Live accent: match the status object back to its run by identity (the
  // run objects come from the same trainingRunDetails cache).
  const liveStatus = latestRunStatusForHistoryPage();
  const liveRunObj = liveStatus ? (runs || []).find(run => run && run.status === liveStatus) : null;
  const liveRun = liveRunObj ? (liveRunObj.name || "") : null;
  const liveEpoch = liveStatus ? asFinite(liveStatus.current_epoch) : null;
  histEpochTable.innerHTML = recent.map(entry => {
    const row = entry.row;
    const epoch = Number(row.epoch);
    const sp = row.selfplay || {};
    const training = row.training || null;
    const evaluation = row.evaluation || null;
    const key = `${entry.run}::${epoch}`;
    const live = liveRun !== null && entry.run === liveRun && liveEpoch === epoch;
    const inspected = !!(histInspectEpoch && histInspectEpoch.run === entry.run && histInspectEpoch.epoch === epoch);
    const expanded = histExpandedEpochs.has(key);
    // Provisional in-flight row (server marks the currently-running epoch):
    // during self-play the live counters replace the absent per-epoch stats;
    // afterwards the phase fills the train/eval cells until real data lands.
    const inProg = row.in_progress && typeof row.in_progress === "object" ? row.in_progress : null;
    const smp = asFinite(sp.samples_added);
    let spText = `${smp !== null ? smp : "--"} smp · len μ${formatDecimal(sp.game_length_mean, 1)}/${formatDecimal(sp.game_length_median, 0)} · ${formatRate(sp.search_positions_per_second, "pos/s")}`;
    if (inProg && inProg.selfplay_live && typeof inProg.selfplay_live === "object") {
      const l = inProg.selfplay_live;
      spText = `self-play ${Number(l.games_finished || 0)}/${Number(l.requested_games || 0)} games · ${formatRate(l.search_pos_s, "pos/s")} · ${formatAge(l.elapsed_seconds)}`;
    }
    // Inline diverging win-balance bar: P0 left, draws middle, P1 right.
    const p0 = asFinite(sp.win_p0_fraction);
    const p1 = asFinite(sp.win_p1_fraction);
    const draw = asFinite(sp.draw_fraction);
    const total = (p0 || 0) + (draw || 0) + (p1 || 0);
    let balance = `<span class="hist-balance-none">--</span>`;
    if ((p0 !== null || draw !== null || p1 !== null) && total > 0) {
      const widthOf = value => (((value || 0) / total) * 100).toFixed(1);
      balance = `<span class="hist-balance-bar" title="${escapeAttr(`P0 ${formatPercent(p0)} · draw ${formatPercent(draw)} · P1 ${formatPercent(p1)}`)}">`
        + `<span class="hist-bal-p0" style="width:${widthOf(p0)}%"></span>`
        + `<span class="hist-bal-draw" style="width:${widthOf(draw)}%"></span>`
        + `<span class="hist-bal-p1" style="width:${widthOf(p1)}%"></span></span>`;
    }
    let trainText = training && asFinite(training.loss) !== null
      ? `loss ${formatDecimal(training.loss, 3)}`
      : (training && training.status) || "pending";
    let trainTitle = trainText;
    const evalReady = evaluation && (asFinite(evaluation.wins) !== null ||
      asFinite(evaluation.losses) !== null || asFinite(evaluation.mean_turns) !== null);
    let evalText = evalReady
      ? `${Number(evaluation.wins || 0)}-${Number(evaluation.losses || 0)} · ${formatDecimal(evaluation.mean_turns, 1)}t`
      : (evaluation && evaluation.status) || "pending";
    if (inProg) {
      if (trainText === "pending" && inProg.phase === "training") {
        trainText = "training…";
        trainTitle = inProg.detail ? `training · ${inProg.detail}` : "training pass running";
      } else if (trainText === "pending" && inProg.phase === "selecting window") {
        trainText = "selecting…";
        trainTitle = "selecting the training window";
      }
      if (evalText === "pending" && inProg.phase === "evaluating") evalText = "evaluating…";
    }
    // §1.2: any truthy checkpoint path OR name counts as saved (the two
    // producer shapes are {path,bytes,modified} and {path,name}).
    const ckpt = row.checkpoint && (row.checkpoint.path || row.checkpoint.name) ? "saved" : "pending";
    const stateTag = inProg
      ? `<span class="hist-epoch-tag hist-epoch-tag-running" title="${escapeAttr(`Epoch ${epoch} in progress — ${inProg.phase || "running"}${inProg.detail ? ` (${inProg.detail})` : ""}; figures are provisional`)}">in progress</span>`
      : `<span class="hist-epoch-tag hist-epoch-tag-${ckpt}">${ckpt}</span>`;
    const buffer = sp.buffer && typeof sp.buffer === "object" ? sp.buffer : null;
    const chevron = buffer
      ? `<button class="hist-epoch-chevron" type="button" data-hist-epoch-toggle="${escapeAttr(key)}" aria-expanded="${expanded ? "true" : "false"}" title="${expanded ? "Hide" : "Show"} buffer and loss detail">${expanded ? "▴" : "▾"}</button>`
      : `<span class="hist-epoch-chevron hist-epoch-chevron-none" aria-hidden="true"></span>`;
    const runTag = multiRun
      ? `<span class="hist-epoch-run" title="${escapeAttr(entry.run)}">${escapeText(entry.run)}</span>`
      : "";
    return `<div class="hist-epoch-row${live ? " hist-epoch-live" : ""}${inProg ? " hist-epoch-inprogress" : ""}${inspected ? " hist-epoch-inspected" : ""}" data-hist-epoch-inspect="${escapeAttr(key)}" title="${escapeAttr(live ? `Currently running epoch — click to inspect` : `Inspect epoch ${epoch}`)}">
      <button class="hist-epoch-num" type="button" data-hist-epoch="${epoch}" title="Filter the games list to epoch ${epoch}">e${epoch}</button>
      ${runTag}
      <span class="hist-epoch-cell hist-epoch-sp" title="${escapeAttr(spText)}">${escapeText(spText)}</span>
      ${balance}
      <span class="hist-epoch-cell hist-epoch-train" title="${escapeAttr(trainTitle)}">${escapeText(trainText)}</span>
      <span class="hist-epoch-cell hist-epoch-eval" title="${escapeAttr(evalText)}">${escapeText(evalText)}</span>
      ${stateTag}
      ${chevron}
    </div>${expanded && buffer ? epochProgressDetail(buffer) : ""}`;
  }).join("");
}

function epochChip(label, value, extraClass = "") {
  return `<span class="epoch-chip ${extraClass}"><i>${escapeText(label)}</i> ${escapeText(value)}</span>`;
}

// Optional full-width detail band beneath an epoch row: a Buffer group, a per-head
// Losses group (total + policy/value/opp + every short-term-value head), and a
// Calibration group (value-head optimism). Rendered only when the producer emits
// the nested `buffer` object (hexgt RL); dense_cnn rows have no buffer, so this
// returns "" and the row is unchanged (additive / dense-safe).
function epochProgressDetail(buf) {
  if (!buf) return "";
  const k = (n) => (asFinite(n) === null ? "--" : `${Math.round(Number(n) / 1000)}k`);
  const windowSpan = buf.window_span ? ` [${buf.window_span}]` : "";
  const bufferChips = [
    epochChip("pool", `${k(buf.samples)}/${k(buf.cap)}`),
    epochChip("window", `${asFinite(buf.window_epochs) ?? "--"}ep${windowSpan}`),
    epochChip("decay", formatDecimal(buf.decay, 2)),
    epochChip("train", `${asFinite(buf.train_steps) ?? "--"}×${asFinite(buf.train_batch) ?? "--"} = ${k(buf.train_samples_per_epoch)}/ep`),
  ].join("");

  // Per-head losses; each head is emitted only when present, so a run missing a
  // head (e.g. pre-deploy epochs without stvalue) just omits that chip.
  const lossChips = [];
  if (asFinite(buf.loss_total) !== null) lossChips.push(epochChip("Σ total", formatDecimal(buf.loss_total, 3), "epoch-chip-total"));
  if (asFinite(buf.loss_policy) !== null) lossChips.push(epochChip("policy", formatDecimal(buf.loss_policy, 3)));
  if (asFinite(buf.loss_value) !== null) lossChips.push(epochChip("value", formatDecimal(buf.loss_value, 3)));
  if (asFinite(buf.loss_opp) !== null) lossChips.push(epochChip("opp", formatDecimal(buf.loss_opp, 3)));
  // Short-term-value heads: every loss_stvalue_<h> the bridge surfaced, by horizon.
  Object.keys(buf)
    .map(key => /^loss_stvalue_(\d+)$/.exec(key))
    .filter(Boolean)
    .sort((a, b) => Number(a[1]) - Number(b[1]))
    .forEach(match => {
      if (asFinite(buf[match[0]]) !== null) lossChips.push(epochChip(`stv${match[1]}`, formatDecimal(buf[match[0]], 3)));
    });
  // Auxiliary shrimp-only heads (emitted only when present).
  if (asFinite(buf.loss_moves_left) !== null) lossChips.push(epochChip("moves", formatDecimal(buf.loss_moves_left, 3)));
  if (asFinite(buf.loss_cell_q) !== null) lossChips.push(epochChip("cellQ", formatDecimal(buf.loss_cell_q, 3)));

  const lossGroup = lossChips.length
    ? `<div class="epoch-detail-group"><span class="epoch-detail-label">Losses</span>${lossChips.join("")}</div>`
    : "";
  // Value-head calibration: optimism_sum_mean (0 = zero-sum-consistent, >0 =
  // optimistic). Emitted by the bridge once the per-epoch calib line is logged.
  const calibGroup = asFinite(buf.optimism_sum_mean) !== null
    ? `<div class="epoch-detail-group"><span class="epoch-detail-label">Calibration</span>${epochChip("optimism", formatDecimal(buf.optimism_sum_mean, 3))}</div>`
    : "";
  return `<div class="epoch-progress-detail">
    <div class="epoch-detail-group"><span class="epoch-detail-label">Buffer</span>${bufferChips}</div>
    ${lossGroup}
    ${calibGroup}
  </div>`;
}

// --- History v2 P1: epoch inspector (#histEpochInspector). ---
// Full-width card between the epoch table and the filters row. The core groups
// (header/losses/game-stats/buffer/calibration/eval/train) render INSTANTLY
// from the run payload already in trainingRunDetails; deltas compare the
// previous epoch row OF THE SAME RUN (never mixed across runs). Two lazy
// fetches (mirroring the loadHistThumb cache pattern) progressively fill the
// DIAGNOSTICS + MODEL groups: /api/training/epoch (curated selfplay extras +
// manifest config + checkpoint stat) and /api/debug/ckpt_info (deep model
// card via the CPU debug worker). All state lives in module vars so the 15s
// innerHTML refresh repaints fetched data with zero new requests.
const HIST_EPOCH_INFO_CACHE_MAX = 20;
const HIST_CKPT_INFO_CACHE_MAX = 12;

// ---- Per-epoch telemetry strip (/api/training/epochs) -----------------------
// One fetch per run, cached by run name (FIFO). The strip records feed the
// inspector's Telemetry group and the sparkline mini-trends. Every field may be
// null (legacy epochs / mid-run) — formatters below render "—" and never throw.

function loadHistEpochs(runName) {
  const cached = histEpochsCache.get(runName);
  if (cached && (cached.status === "loading" || cached.status === "ready")) return;
  histEpochsCache.set(runName, { status: "loading", epochs: [] });
  while (histEpochsCache.size > HIST_EPOCHS_CACHE_MAX) {
    histEpochsCache.delete(histEpochsCache.keys().next().value);
  }
  fetch(`/api/training/epochs?${new URLSearchParams({ run: runName }).toString()}`)
    .then(res => safeJson(res).then(data => ({ res, data })))
    .then(({ res, data }) => {
      if (!res.ok) throw new Error((data && data.error) || "epochs telemetry unavailable");
      const epochs = data && Array.isArray(data.epochs) ? data.epochs : [];
      histEpochsCache.set(runName, { status: "ready", epochs });
      // Repaint the inspector in place if it is still showing this run — the
      // Telemetry group + sparklines fill once the fetch lands (no full render,
      // same rationale as histInspPatchLazy).
      if (histInspectEpoch && histInspectEpoch.run === runName) renderHistEpochInspector(null);
    })
    .catch(error => {
      console.warn("loadHistEpochs failed", error);
      histEpochsCache.set(runName, { status: "error", epochs: [] });
    });
}

// The strip record for one (run, epoch), or null while loading / on miss. Kicks
// the per-run fetch on first miss (deduped by the "loading" cache state).
function histEpochTelemetry(runName, epoch) {
  const entry = histEpochsCache.get(runName);
  if (!entry) { loadHistEpochs(runName); return null; }
  if (entry.status !== "ready") return null;
  return entry.epochs.find(rec => rec && asFinite(rec.epoch) === Number(epoch)) || null;
}

// Compact human count: 21845 -> "21.8k", 950 -> "950", 2_100_000 -> "2.1M".
function formatCount(value) {
  const n = asFinite(value);
  if (n === null) return "--";
  const abs = Math.abs(n);
  if (abs >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e4) return `${(n / 1e3).toFixed(1)}k`;
  if (abs >= 1000) return `${(n / 1e3).toFixed(2)}k`;
  return String(Math.round(n));
}

// Seconds -> "427s" / "32.6m" (compact wall-time).
function formatSeconds(value) {
  const n = asFinite(value);
  if (n === null) return "--";
  if (n >= 600) return `${(n / 60).toFixed(1)}m`;
  return `${Math.round(n)}s`;
}

// Self-contained inline-SVG polyline sparkline (no external deps — the dashboard
// runs offline). `values` may contain nulls (missing epochs); those break the
// line into segments so gaps read as gaps, not zeros. `markIdx` highlights the
// currently-inspected epoch's dot. Returns "" when fewer than 2 finite points.
function histSparkline(values, opts = {}) {
  const w = opts.width || 132;
  const h = opts.height || 26;
  const pad = 3;
  const nums = (values || []).map(asFinite);
  const finite = nums.filter(v => v !== null);
  if (finite.length < 2) return "";
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;
  const n = nums.length;
  const x = i => pad + (n <= 1 ? 0 : (i / (n - 1)) * (w - 2 * pad));
  const y = v => pad + (1 - (v - min) / span) * (h - 2 * pad);
  // Split into contiguous finite runs so nulls leave a gap.
  const segments = [];
  let cur = [];
  nums.forEach((v, i) => {
    if (v === null) { if (cur.length) { segments.push(cur); cur = []; } }
    else cur.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  });
  if (cur.length) segments.push(cur);
  const lines = segments
    .map(pts => `<polyline class="hist-spark-line" fill="none" points="${pts.join(" ")}"></polyline>`)
    .join("");
  let mark = "";
  const mi = asFinite(opts.markIdx);
  if (mi !== null && mi >= 0 && mi < n && nums[mi] !== null) {
    mark = `<circle class="hist-spark-mark" cx="${x(mi).toFixed(1)}" cy="${y(nums[mi]).toFixed(1)}" r="2.4"></circle>`;
  }
  const title = opts.title ? ` title="${escapeAttr(opts.title)}"` : "";
  return `<svg class="hist-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" preserveAspectRatio="none" aria-hidden="true"${title}>${lines}${mark}</svg>`;
}

function histInspRunPayload(runName, runs) {
  const list = runs || historyRunsForPage();
  return (list || []).find(run => run && run.name === runName) || trainingRunDetails[runName] || null;
}

// The inspected run's epoch_history rows, ascending by epoch (◀/▶ walk this).
function histInspRunRows(runName, runs) {
  const run = histInspRunPayload(runName, runs);
  const history = run && Array.isArray(run.epoch_history) ? run.epoch_history : [];
  return history
    .filter(row => row && asFinite(row.epoch) !== null)
    .slice()
    .sort((a, b) => Number(a.epoch) - Number(b.epoch));
}

function histInspectStep(step) {
  if (!histInspectEpoch || !step) return;
  const rows = histInspRunRows(histInspectEpoch.run, null);
  const idx = rows.findIndex(row => Number(row.epoch) === histInspectEpoch.epoch);
  const next = idx >= 0 ? rows[idx + step] : null;
  if (!next) return;
  histInspectEpoch = { run: histInspectEpoch.run, epoch: Number(next.epoch) };
  renderGameHistoryPage();
}

// P2.1/P2.3: step the selected game through histDisplayedKeys (the displayed-
// list order). Shared by the detail-panel prev/next buttons and the arrow
// keys; sets selectedHistoryKey + re-renders — the exact same state + path a
// list-row click takes, so thumbnail/selection/detail behave identically.
// Returns true when a step happened (the key handler preventDefaults on it).
function histStepGame(step) {
  if (!step || !histDisplayedKeys.length) return false;
  const idx = histDisplayedKeys.indexOf(selectedHistoryKey);
  if (idx < 0) return false;
  const next = idx + step;
  if (next < 0 || next >= histDisplayedKeys.length) return false;
  selectedHistoryKey = histDisplayedKeys[next];
  renderGameHistoryPage();
  return true;
}

// History-screen keyboard layer. Mirrors dbgHandleKey's guard shape (screen
// gate, input focus, modifiers) so the two document handlers can never both
// fire. P1 owns the Esc branch; P2 adds the game-row arrows.
function histHandleKey(e) {
  if (activeScreen !== "history") return;
  const ae = document.activeElement;
  if (ae && /^(INPUT|SELECT|TEXTAREA)$/.test(ae.tagName)) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "Escape" && histInspectEpoch !== null) {
    histInspectEpoch = null;
    renderGameHistoryPage();
    e.preventDefault();
    return;
  }
  // Arrows step the selected game ONLY while the epoch inspector is closed —
  // reserved for future inspector bindings when it is open (P2.3 guard).
  if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && histInspectEpoch === null) {
    if (histStepGame(e.key === "ArrowLeft" ? -1 : 1)) e.preventDefault();
  }
}

// §0.2 checkpoint-name rule: endpoint checkpoint.name → row.checkpoint.name ||
// basename(path) → canonical epoch_{N:06d}.pt. Returns null when NO checkpoint
// exists anywhere (no row entry AND the endpoint said null) so the model card
// is hidden and the ckpt_info fetch never fires.
function histCkptNameForEpoch(row, epoch, infoPayload) {
  const fromInfo = infoPayload && infoPayload.checkpoint && infoPayload.checkpoint.name;
  if (fromInfo) return String(fromInfo);
  const ckpt = row && row.checkpoint;
  if (ckpt && (ckpt.name || ckpt.path)) {
    if (ckpt.name) return String(ckpt.name);
    const base = String(ckpt.path || "").split(/[\\/]/).pop();
    if (base) return base;
    return `epoch_${String(epoch).padStart(6, "0")}.pt`;
  }
  return null;
}

// Per-head loss carrier (§0.1): total/policy/value/opp + every stvalue head.
function histLossHeads(buf) {
  if (!buf || typeof buf !== "object") return [];
  const heads = [];
  const push = (key, label) => {
    const value = asFinite(buf[key]);
    if (value !== null) heads.push({ key, label, value });
  };
  push("loss_total", "total");
  push("loss_policy", "policy");
  push("loss_value", "value");
  push("loss_opp", "opp");
  Object.keys(buf)
    .map(key => /^loss_stvalue_(\d+)$/.exec(key))
    .filter(Boolean)
    .sort((a, b) => Number(a[1]) - Number(b[1]))
    .forEach(match => push(match[0], `stv${match[1]}`));
  push("loss_moves_left", "moves");
  push("loss_cell_q", "cellQ");
  return heads;
}

// Green when the metric fell vs the previous same-run epoch, red when it rose
// (losses and optimism both read "down is good"); no element without a prior.
function histDeltaHtml(value, prevValue) {
  if (value === null || prevValue === null) return "";
  const d = value - prevValue;
  if (d < 0) return `<span class="hist-delta hist-delta-down">▼${Math.abs(d).toFixed(3)}</span>`;
  if (d > 0) return `<span class="hist-delta hist-delta-up">▲${d.toFixed(3)}</span>`;
  return `<span class="hist-delta hist-delta-flat">±0.000</span>`;
}

function histInspLossGroup(buf, prevBuf) {
  const heads = histLossHeads(buf);
  if (!heads.length) return "";
  const max = heads.reduce((m, head) => Math.max(m, head.value), 0);
  const rows = heads.map(head => {
    const pct = max > 0 ? Math.max(2, Math.round((head.value / max) * 100)) : 0;
    return `<div class="hist-loss-row">
      <span class="hist-loss-label">${escapeText(head.label)}</span>
      <span class="hist-loss-bar"><span class="hist-loss-bar-fill" style="width:${pct}%"></span><span class="hist-loss-val">${head.value.toFixed(3)}</span></span>
      ${histDeltaHtml(head.value, prevBuf ? asFinite(prevBuf[head.key]) : null)}
    </div>`;
  }).join("");
  return `<div class="hist-insp-group"><span class="hist-insp-group-title">Losses</span>${rows}</div>`;
}

// Same diverging-bar markup as the epoch-table rows; "" when fractions absent.
function histBalanceBarHtml(sp) {
  const p0 = asFinite(sp.win_p0_fraction);
  const p1 = asFinite(sp.win_p1_fraction);
  const draw = asFinite(sp.draw_fraction);
  const total = (p0 || 0) + (draw || 0) + (p1 || 0);
  if ((p0 === null && draw === null && p1 === null) || total <= 0) return "";
  const widthOf = value => (((value || 0) / total) * 100).toFixed(1);
  return `<span class="hist-balance-bar" title="${escapeAttr(`P0 ${formatPercent(p0)} · draw ${formatPercent(draw)} · P1 ${formatPercent(p1)}`)}">`
    + `<span class="hist-bal-p0" style="width:${widthOf(p0)}%"></span>`
    + `<span class="hist-bal-draw" style="width:${widthOf(draw)}%"></span>`
    + `<span class="hist-bal-p1" style="width:${widthOf(p1)}%"></span></span>`;
}

function histInspGameStatsGroup(sp) {
  if (!sp || typeof sp !== "object") return "";
  const chips = [];
  if (asFinite(sp.game_length_mean) !== null) chips.push(epochChip("len μ", formatDecimal(sp.game_length_mean, 1)));
  if (asFinite(sp.game_length_median) !== null) chips.push(epochChip("med", formatDecimal(sp.game_length_median, 0)));
  if (asFinite(sp.game_length_max) !== null) chips.push(epochChip("max", formatDecimal(sp.game_length_max, 0)));
  if (asFinite(sp.game_length_stdev) !== null) chips.push(epochChip("σ", formatDecimal(sp.game_length_stdev, 1)));
  if (asFinite(sp.games) !== null) chips.push(epochChip("games", asFinite(sp.games)));
  if (asFinite(sp.samples_added) !== null) chips.push(epochChip("smp", asFinite(sp.samples_added)));
  if (asFinite(sp.search_positions_per_second) !== null) chips.push(epochChip("speed", formatRate(sp.search_positions_per_second, "pos/s")));
  if (asFinite(sp.mcts_sims_per_searched_position) !== null) chips.push(epochChip("sims/pos", formatDecimal(sp.mcts_sims_per_searched_position, 1)));
  if (asFinite(sp.elapsed_seconds) !== null) chips.push(epochChip("sp wall", `${formatDecimal(sp.elapsed_seconds, 0)}s`));
  const balance = histBalanceBarHtml(sp);
  if (!chips.length && !balance) return "";
  return `<div class="hist-insp-group"><span class="hist-insp-group-title">Game stats</span>${balance}<div class="hist-insp-chips">${chips.join("")}</div></div>`;
}

// Only for runs with a REAL pool (hexgt): dense loss-only buffers hide this.
function histInspBufferGroup(buf, rowSamples) {
  if (!buf || (asFinite(buf.samples) === null && asFinite(buf.cap) === null)) return "";
  const k = (n) => (asFinite(n) === null ? "--" : `${Math.round(Number(n) / 1000)}k`);
  const windowSpan = buf.window_span ? ` [${buf.window_span}]` : "";
  const chips = [
    epochChip("pool", `${k(buf.samples)}/${k(buf.cap)}`),
    epochChip("window", `${asFinite(buf.window_epochs) ?? "--"}ep${windowSpan}`),
    epochChip("decay", formatDecimal(buf.decay, 2)),
    epochChip("train", `${asFinite(buf.train_steps) ?? "--"}×${asFinite(buf.train_batch) ?? "--"} = ${k(buf.train_samples_per_epoch)}/ep`),
  ];
  if (rowSamples && typeof rowSamples === "object") {
    if (asFinite(rowSamples.buffer_count) !== null) chips.push(epochChip("records", asFinite(rowSamples.buffer_count)));
    if (asFinite(rowSamples.window_size) !== null) chips.push(epochChip("win size", asFinite(rowSamples.window_size)));
    if (asFinite(rowSamples.compressed_bytes) !== null) chips.push(epochChip("compressed", formatBytes(rowSamples.compressed_bytes)));
  }
  return `<div class="hist-insp-group"><span class="hist-insp-group-title">Buffer</span><div class="hist-insp-chips">${chips.join("")}</div></div>`;
}

function histInspCalibrationGroup(buf, prevBuf) {
  const value = buf ? asFinite(buf.optimism_sum_mean) : null;
  if (value === null) return "";
  const delta = histDeltaHtml(value, prevBuf ? asFinite(prevBuf.optimism_sum_mean) : null);
  return `<div class="hist-insp-group"><span class="hist-insp-group-title">Calibration</span><div class="hist-insp-chips">${epochChip("optimism", formatDecimal(value, 3))}${delta}</div></div>`;
}

function histInspEvalGroup(row, run, epoch) {
  let ev = row.evaluation || null;
  if (!ev && run && Array.isArray(run.evaluation_history)) {
    ev = run.evaluation_history.find(item => item && asFinite(item.epoch) === epoch) || null;
  }
  if (!ev || typeof ev !== "object") return "";
  const wins = asFinite(ev.wins);
  const losses = asFinite(ev.losses);
  const games = asFinite(ev.games);
  const chips = [];
  if (wins !== null || losses !== null) chips.push(epochChip("W-L", `${wins ?? "--"}-${losses ?? "--"}`));
  if (wins !== null && games !== null && games > 0) chips.push(epochChip("win", formatPercent(wins / games)));
  if (asFinite(ev.mean_turns) !== null) chips.push(epochChip("mean turns", formatDecimal(ev.mean_turns, 1)));
  if (asFinite(ev.completed) !== null || games !== null) chips.push(epochChip("done", `${asFinite(ev.completed) ?? "--"}/${games ?? "--"}`));
  if (ev.status) chips.push(epochChip("status", String(ev.status)));
  if (!chips.length) return "";
  return `<div class="hist-insp-group"><span class="hist-insp-group-title">Evaluation</span><div class="hist-insp-chips">${chips.join("")}</div></div>`;
}

// Per-epoch multi-stage eval (shrimp opt-in): the VERDICT for this epoch plus
// its headline edges. Keys off run.multistage_eval_history by epoch; returns ""
// for every epoch/run lacking a report (so the :empty rule hides the group).
function histInspMultistageGroup(row, run, epoch) {
  const list = run && Array.isArray(run.multistage_eval_history) ? run.multistage_eval_history : [];
  const ms = list.find(item => item && asFinite(item.epoch) === epoch) || null;
  if (!ms) return "";
  const verdict = ms.verdict && typeof ms.verdict === "object" ? ms.verdict : {};
  const verdictLabel = ms.verdict_label || verdict.label || null;
  const parts = [];
  if (verdictLabel) {
    parts.push(`<span class="hist-rating-verdict ${msVerdictClass(verdictLabel)}">${escapeText(String(verdictLabel).toUpperCase())}</span>`);
  }
  const candidate = msCandidatePlayer(ms);
  if (candidate) parts.push(epochChip("Elo", msEloText(candidate)));
  msHeadlineEdges(ms).forEach(edge => parts.push(epochChip(msEdgeLabel(edge), msWinrateText(edge))));
  if (!parts.length) return "";
  const note = verdict.note ? `<div class="hist-insp-note">${escapeText(String(verdict.note))}</div>` : "";
  return `<div class="hist-insp-group"><span class="hist-insp-group-title">Multi-stage eval</span><div class="hist-insp-chips">${parts.join("")}</div>${note}</div>`;
}

function histInspTrainGroup(training) {
  if (!training || typeof training !== "object") return "";
  const chips = [];
  if (asFinite(training.loss) !== null) chips.push(epochChip("loss", formatDecimal(training.loss, 3)));
  if (asFinite(training.steps) !== null || asFinite(training.batch_size) !== null) {
    chips.push(epochChip("steps", `${asFinite(training.steps) ?? "--"}×${asFinite(training.batch_size) ?? "--"}`));
  }
  if (asFinite(training.samples) !== null) chips.push(epochChip("samples", asFinite(training.samples)));
  if (asFinite(training.samples_per_second) !== null) chips.push(epochChip("speed", formatRate(training.samples_per_second, "smp/s")));
  if (asFinite(training.elapsed_seconds) !== null) chips.push(epochChip("wall", `${formatDecimal(training.elapsed_seconds, 0)}s`));
  if (!chips.length && training.status) chips.push(epochChip("status", String(training.status)));
  if (!chips.length) return "";
  return `<div class="hist-insp-group"><span class="hist-insp-group-title">Training</span><div class="hist-insp-chips">${chips.join("")}</div></div>`;
}

// DIAGNOSTICS group inner HTML (also the patch target of loadHistEpochInfo).
// "" hides the group via the :empty rule — hexgt runs and still-running epochs
// have no dense_cnn.selfplay file, so selfplay_extras comes back null.
function histInspDiagInner(entry) {
  if (!entry || entry.status === "loading") {
    return `<span class="hist-insp-group-title">Diagnostics</span><div class="hist-insp-note">Loading epoch diagnostics</div>`;
  }
  if (entry.status === "error") {
    return `<span class="hist-insp-group-title">Diagnostics</span><div class="hist-insp-note">Epoch diagnostics unavailable</div>`;
  }
  const extras = entry.payload && entry.payload.selfplay_extras;
  if (!extras || typeof extras !== "object") return "";
  const chips = [];
  const tc = extras.temperature_control && typeof extras.temperature_control === "object" ? extras.temperature_control : {};
  if (asFinite(tc.expected_game_length) !== null) chips.push(epochChip("EMA len", formatDecimal(tc.expected_game_length, 1)));
  if (asFinite(tc.halflife_plies) !== null) chips.push(epochChip("halflife", `${formatDecimal(tc.halflife_plies, 1)} plies`));
  const pcr = extras.pcr && typeof extras.pcr === "object" ? extras.pcr : {};
  if (asFinite(pcr.full_search_count) !== null || asFinite(pcr.fast_search_count) !== null) {
    chips.push(epochChip("full/fast", `${asFinite(pcr.full_search_count) ?? "--"}/${asFinite(pcr.fast_search_count) ?? "--"}`));
  }
  if (asFinite(pcr.full_proportion) !== null) chips.push(epochChip("full", formatPercent(pcr.full_proportion)));
  const pi = extras.policy_init && typeof extras.policy_init === "object" ? extras.policy_init : {};
  if (asFinite(pi.moves) !== null) chips.push(epochChip("PI moves", asFinite(pi.moves)));
  if (asFinite(pi.fraction) !== null) chips.push(epochChip("PI frac", formatPercent(pi.fraction)));
  const rt = extras.root_policy_temperature_control && typeof extras.root_policy_temperature_control === "object"
    ? extras.root_policy_temperature_control
    : {};
  if (asFinite(rt.base) !== null || asFinite(rt.early) !== null) {
    chips.push(epochChip("root T", `${formatDecimal(rt.base, 2)}/${formatDecimal(rt.early, 2)}`));
  }
  const wall = asFinite(extras.elapsed_seconds);
  const search = asFinite(extras.mcts_search_elapsed_seconds);
  if (wall !== null && wall > 0 && search !== null) chips.push(epochChip("search wall", formatPercent(search / wall)));
  const raw = asFinite(extras.raw_samples);
  const eff = asFinite(extras.effective_samples);
  if (raw !== null || eff !== null) chips.push(epochChip("raw→eff", `${raw ?? "--"}→${eff ?? "--"}`));
  if (asFinite(extras.mcts_virtual_batch_size) !== null) chips.push(epochChip("vbatch", asFinite(extras.mcts_virtual_batch_size)));
  if (extras.scheduler !== undefined && extras.scheduler !== null && extras.scheduler !== "") {
    chips.push(epochChip("sched", String(extras.scheduler)));
  }
  if (!chips.length) return "";
  return `<span class="hist-insp-group-title">Diagnostics</span><div class="hist-insp-chips">${chips.join("")}</div>`;
}

// MODEL-config chips from the manifest subset in the SAME epoch payload.
function histInspConfigInner(entry) {
  if (!entry || entry.status !== "ready") return "";
  const manifest = entry.payload && entry.payload.manifest;
  if (!manifest || typeof manifest !== "object") return "";
  const chips = [];
  if (manifest.model_name) chips.push(epochChip("model", String(manifest.model_name)));
  const arch = manifest.architecture && typeof manifest.architecture === "object" ? manifest.architecture : {};
  if (asFinite(arch.channels) !== null || arch.blocks_type) {
    chips.push(epochChip("arch", `${asFinite(arch.channels) ?? "--"}ch ${arch.blocks_type || ""}`.trim()));
  }
  if (asFinite(arch.attention_heads) !== null) chips.push(epochChip("attn", asFinite(arch.attention_heads)));
  if (Array.isArray(arch.short_term_value_horizons) && arch.short_term_value_horizons.length) {
    chips.push(epochChip("stv", arch.short_term_value_horizons.join(",")));
  }
  if (arch.moves_left_head !== undefined && arch.moves_left_head !== null) {
    chips.push(epochChip("moves-left", arch.moves_left_head ? "on" : "off"));
  }
  const spc = manifest.selfplay && typeof manifest.selfplay === "object" ? manifest.selfplay : {};
  if (asFinite(spc.search_visits) !== null) chips.push(epochChip("visits", asFinite(spc.search_visits)));
  if (asFinite(spc.active_games) !== null) chips.push(epochChip("games", asFinite(spc.active_games)));
  if (asFinite(spc.root_policy_temperature) !== null) chips.push(epochChip("root temp", formatDecimal(spc.root_policy_temperature, 2)));
  if (asFinite(spc.fpu_reduction) !== null) chips.push(epochChip("fpu", formatDecimal(spc.fpu_reduction, 2)));
  const ev = manifest.evaluation && typeof manifest.evaluation === "object" ? manifest.evaluation : {};
  if (asFinite(ev.games_per_epoch) !== null) {
    chips.push(epochChip("eval", `${asFinite(ev.games_per_epoch)}g every ${asFinite(ev.eval_every) ?? "--"}`));
  }
  if (!chips.length) return "";
  return `<div class="hist-insp-chips">${chips.join("")}</div>`;
}

// Deep model card (lazy ckpt_info). The "CPU debug worker" pending hint is
// mandatory: the first call loads the checkpoint into the single worker.
function histInspCkptInner(entry, ckptName) {
  if (!ckptName) return "";
  if (!entry || entry.status === "loading") {
    return `<div class="hist-insp-note">Loading model card (CPU debug worker)…</div>`;
  }
  if (entry.status === "error") return `<div class="hist-insp-note">Model card unavailable</div>`;
  const payload = entry.payload || {};
  const meta = payload.meta && typeof payload.meta === "object" ? payload.meta : {};
  const chips = [];
  const params = asFinite(meta.param_count);
  if (params !== null) chips.push(epochChip("params", params >= 1e6 ? `${(params / 1e6).toFixed(2)}M` : String(params)));
  if (meta.lineage) chips.push(epochChip("lineage", String(meta.lineage)));
  if (asFinite(meta.rl_epoch) !== null) chips.push(epochChip("rl epoch", asFinite(meta.rl_epoch)));
  if (asFinite(meta.step) !== null) chips.push(epochChip("step", asFinite(meta.step)));
  if (meta.graft) chips.push(epochChip("graft", String(meta.graft)));
  if (Array.isArray(meta.stv_horizons) && meta.stv_horizons.length) chips.push(epochChip("stv", meta.stv_horizons.join(",")));
  if (asFinite(meta.moves_left_cap) !== null) chips.push(epochChip("ml cap", asFinite(meta.moves_left_cap)));
  if (asFinite(payload.size) !== null) chips.push(epochChip("file", formatBytes(payload.size)));
  if (payload.mtime) chips.push(epochChip("saved", formatHistoryDate(payload.mtime)));
  const warnings = Array.isArray(meta.load_warnings) ? meta.load_warnings.filter(Boolean) : [];
  const warnHtml = warnings.length
    ? `<div class="hist-insp-warnings">${warnings.map(w => `<div class="hist-insp-warning">${escapeText(String(w))}</div>`).join("")}</div>`
    : "";
  if (!chips.length && !warnHtml) return "";
  const archTitle = meta.arch ? ` title="${escapeAttr(String(meta.arch))}"` : "";
  return `<div class="hist-insp-chips"${archTitle}>${chips.join("")}</div>${warnHtml}`;
}

// Header checkpoint stat line: endpoint {name,size,mtime} when it has landed,
// else the epoch_history row's own {bytes,modified} shape.
function histInspCkptLineInner(row, infoEntry) {
  const fromInfo = infoEntry && infoEntry.status === "ready" && infoEntry.payload && infoEntry.payload.checkpoint;
  if (fromInfo && typeof fromInfo === "object") {
    const bits = [
      fromInfo.name ? String(fromInfo.name) : null,
      asFinite(fromInfo.size) !== null ? formatBytes(fromInfo.size) : null,
      fromInfo.mtime ? formatHistoryDate(fromInfo.mtime) : null,
    ].filter(Boolean);
    if (bits.length) return escapeText(bits.join(" · "));
  }
  const ckpt = row && row.checkpoint;
  if (ckpt && (ckpt.path || ckpt.name)) {
    const bits = [
      ckpt.name ? String(ckpt.name) : String(ckpt.path || "").split(/[\\/]/).pop(),
      asFinite(ckpt.bytes) !== null ? formatBytes(ckpt.bytes) : null,
      ckpt.modified ? formatHistoryDate(ckpt.modified) : null,
    ].filter(Boolean);
    return escapeText(bits.join(" · "));
  }
  return "";
}

// ---- Telemetry group (schema-upgrade strip) ---------------------------------
// Renders the per-epoch telemetry from /api/training/epochs into the inspector's
// detail: state badges, a compact headline chip row, a per-phase entropy+value
// table, and expandable detail chips (unique openings, decided fraction,
// surprise, fast/full/init, lcb/gumbel rates, window span, shards+paths, eval
// Elo edges), plus sparkline mini-trends across epochs. Every value degrades to
// "—"; the whole group returns "" (→ :empty hides it) when the record is null.

function histTelemBadges(sp, sel) {
  const badges = [];
  const resumed = sp.resumed_skip === true || asFinite(sp.resumed_skip_count) > 0 || asFinite(sp.resumed_existing_games) > 0;
  if (resumed) {
    const cnt = asFinite(sp.resumed_skip_count);
    badges.push(`<span class="hist-telem-badge hist-telem-badge-info" title="This epoch resumed existing self-play games (segment merge)">resumed${cnt ? ` ×${cnt}` : ""}</span>`);
  }
  if (sp.merged_approx) {
    badges.push(`<span class="hist-telem-badge hist-telem-badge-info" title="Segment stats merged approximately across a resume boundary">merged≈</span>`);
  }
  const shards = asFinite(sel.shards_skipped);
  if (shards !== null && shards > 0) {
    badges.push(`<span class="hist-telem-badge hist-telem-badge-warn" title="Shards skipped during window selection">skipped shards ${shards}</span>`);
  }
  const trunc = asFinite(sp.truncated_games);
  if (trunc !== null && trunc > 0) {
    badges.push(`<span class="hist-telem-badge hist-telem-badge-warn" title="Games hitting the length cap (truncated, not naturally decided)">truncated ${trunc}</span>`);
  }
  return badges.length ? `<div class="hist-telem-badges">${badges.join("")}</div>` : "";
}

// Per-phase entropy + value table (opening / mid / late). Rows appear only for
// phases the producer emitted; "" when neither by-phase object exists.
function histTelemPhaseTable(sp) {
  const ent = sp.root_policy_entropy_by_phase && typeof sp.root_policy_entropy_by_phase === "object" ? sp.root_policy_entropy_by_phase : {};
  const val = sp.root_value_by_phase && typeof sp.root_value_by_phase === "object" ? sp.root_value_by_phase : {};
  const phases = ["opening", "mid", "late"];
  const present = phases.filter(p => (ent[p] && typeof ent[p] === "object") || (val[p] && typeof val[p] === "object"));
  if (!present.length) return "";
  const cell = obj => {
    if (!obj || typeof obj !== "object") return "<td>—</td>";
    const m = formatDecimal(obj.mean, 2);
    const n = asFinite(obj.n);
    return `<td>${escapeText(m)}${n !== null ? `<span class="hist-telem-n"> n${formatCount(n)}</span>` : ""}</td>`;
  };
  const rows = present.map(p =>
    `<tr><th>${escapeText(p)}</th>${cell(ent[p])}${cell(val[p])}</tr>`).join("");
  return `<table class="hist-telem-phase"><thead><tr><th></th><th>entropy</th><th>value</th></tr></thead><tbody>${rows}</tbody></table>`;
}

// Detail chips: unique openings, decided fraction, surprise, move-mix fractions,
// gumbel/lcb rates, window span, shards + skipped paths.
function histTelemDetailChips(sp, sel, tr) {
  const chips = [];
  const uo = sp.unique_openings && typeof sp.unique_openings === "object" ? sp.unique_openings : null;
  if (uo) {
    const parts = ["10", "16", "20"].map(k => asFinite(uo[k]) !== null ? `${k}:${uo[k]}` : null).filter(Boolean);
    if (parts.length) chips.push(epochChip("openings", parts.join(" ")));
  }
  if (asFinite(sp.decided_fraction) !== null) chips.push(epochChip("decided", formatPercent(sp.decided_fraction)));
  if (asFinite(sp.root_value_abs_mean) !== null) chips.push(epochChip("|value|", formatDecimal(sp.root_value_abs_mean, 3)));
  if (asFinite(sp.root_value_std) !== null) chips.push(epochChip("value σ", formatDecimal(sp.root_value_std, 3)));
  if (asFinite(sp.policy_surprise_mean) !== null) {
    const p90 = asFinite(sp.policy_surprise_p90);
    const mx = asFinite(sp.policy_surprise_max);
    const extra = [p90 !== null ? `p90 ${p90.toFixed(2)}` : null, mx !== null ? `max ${mx.toFixed(2)}` : null].filter(Boolean).join(" · ");
    chips.push(epochChip("surprise", `${formatDecimal(sp.policy_surprise_mean, 2)}${extra ? ` (${extra})` : ""}`));
  }
  const mix = ["fast_fraction", "full_fraction", "init_fraction"]
    .map(k => asFinite(sp[k]) !== null ? `${k[0]}${formatPercent(sp[k])}` : null).filter(Boolean);
  if (mix.length) chips.push(epochChip("f/f/i mix", mix.join(" ")));
  if (asFinite(sp.gumbel_play_winner_rate) !== null) chips.push(epochChip("gumbel win", formatPercent(sp.gumbel_play_winner_rate)));
  if (asFinite(sp.gumbel_play_winner_early_rate) !== null) chips.push(epochChip("gumbel early", formatPercent(sp.gumbel_play_winner_early_rate)));
  if (asFinite(sp.lcb_override_rate) !== null) chips.push(epochChip("lcb ovr", formatPercent(sp.lcb_override_rate)));
  const span = sel.window_epoch_span && typeof sel.window_epoch_span === "object" ? sel.window_epoch_span : null;
  if (span && (asFinite(span.min) !== null || asFinite(span.max) !== null)) {
    chips.push(epochChip("window", `e${asFinite(span.min) ?? "?"}–e${asFinite(span.max) ?? "?"}${asFinite(span.epochs) !== null ? ` (${span.epochs}ep)` : ""}`));
  }
  if (asFinite(sel.keep_prob) !== null) chips.push(epochChip("keep p", formatDecimal(sel.keep_prob, 2)));
  if (asFinite(sel.reuse_ratio) !== null) chips.push(epochChip("reuse", `${formatDecimal(sel.reuse_ratio, 2)}×`));
  if (asFinite(tr.surprise_weight_mean) !== null) {
    chips.push(epochChip("surprise wt", `${formatDecimal(tr.surprise_weight_mean, 2)}${asFinite(tr.surprise_weight_max) !== null ? `/${formatDecimal(tr.surprise_weight_max, 2)}` : ""}`));
  }
  let paths = "";
  const skipped = Array.isArray(sel.skipped_paths) ? sel.skipped_paths.filter(Boolean) : [];
  if (skipped.length) {
    paths = `<details class="hist-telem-paths"><summary>${skipped.length} skipped shard${skipped.length === 1 ? "" : "s"}</summary><ul>${skipped.map(p => `<li>${escapeText(String(p))}</li>`).join("")}</ul></details>`;
  }
  if (!chips.length && !paths) return "";
  return `${chips.length ? `<div class="hist-insp-chips">${chips.join("")}</div>` : ""}${paths}`;
}

// Eval Elo edges block (present only at eval epochs).
function histTelemEvalEdges(ev) {
  if (!ev || typeof ev !== "object") return "";
  const chips = [];
  if (ev.verdict_label) chips.push(`<span class="hist-rating-verdict ${msVerdictClass(ev.verdict_label)}">${escapeText(String(ev.verdict_label).toUpperCase())}</span>`);
  if (asFinite(ev.elo_point) !== null) chips.push(epochChip("Elo", Math.round(Number(ev.elo_point))));
  (Array.isArray(ev.edges) ? ev.edges : []).forEach(edge => {
    if (!edge || typeof edge !== "object") return;
    const wr = asFinite(edge.winrate);
    const elo = asFinite(edge.elo_point);
    const val = [wr !== null ? formatPercent(wr) : null, elo !== null ? `${elo > 0 ? "+" : ""}${Math.round(elo)} Elo` : null].filter(Boolean).join(" · ");
    if (val) chips.push(epochChip(`vs ${edge.opponent || "?"}`, val));
  });
  return chips.length ? `<div class="hist-insp-chips">${chips.join("")}</div>` : "";
}

// Sparkline mini-trends across the run's epochs (entropy, length p50, gumbel-win
// rate, loss_policy), with the inspected epoch marked. "" when the strip has <2
// usable points on any trend.
function histTelemSparklines(records, epoch) {
  if (!Array.isArray(records) || records.length < 2) return "";
  const ordered = records.slice().sort((a, b) => Number(a.epoch) - Number(b.epoch));
  const markIdx = ordered.findIndex(r => Number(r.epoch) === Number(epoch));
  const trends = [
    { label: "entropy", pick: r => asFinite((r.selfplay || {}).root_policy_entropy_mean) },
    { label: "len p50", pick: r => asFinite((r.selfplay || {}).game_length_p50) },
    { label: "gumbel win", pick: r => asFinite((r.selfplay || {}).gumbel_play_winner_rate) },
    { label: "loss pol", pick: r => asFinite((r.training || {}).loss_policy) },
  ];
  const cells = trends.map(t => {
    const values = ordered.map(t.pick);
    const spark = histSparkline(values, { markIdx, title: `${t.label} across epochs ${ordered[0].epoch}–${ordered[ordered.length - 1].epoch}` });
    if (!spark) return "";
    const here = markIdx >= 0 ? values[markIdx] : null;
    return `<div class="hist-telem-spark-cell"><span class="hist-telem-spark-label">${escapeText(t.label)}${here !== null ? ` <b>${here.toFixed(here < 10 ? 2 : 0)}</b>` : ""}</span>${spark}</div>`;
  }).filter(Boolean);
  return cells.length ? `<div class="hist-telem-sparks">${cells.join("")}</div>` : "";
}

function histInspTelemetryGroup(runName, epoch) {
  const entry = histEpochsCache.get(runName);
  if (!entry) { loadHistEpochs(runName); }
  const rec = histEpochTelemetry(runName, epoch);
  const status = entry ? entry.status : "loading";
  if (!rec) {
    if (status === "loading" || !entry) {
      return `<div class="hist-insp-group hist-insp-telem"><span class="hist-insp-group-title">Telemetry</span><div class="hist-insp-note">Loading per-epoch telemetry</div></div>`;
    }
    return "";  // ready-but-no-record (non-shrimp / no diagnostics) → hidden
  }
  const sp = rec.selfplay || {};
  const sel = rec.select || {};
  const tr = rec.training || {};
  // Headline chip row: the compact "at a glance" numbers.
  const head = [];
  const games = asFinite(sp.games_finished);
  const trunc = asFinite(sp.truncated_games);
  if (games !== null) head.push(epochChip("games", `${games}${trunc ? ` (−${trunc})` : ""}`));
  if (asFinite(sp.rows_written) !== null) head.push(epochChip("rows", formatCount(sp.rows_written)));
  if (asFinite(sp.game_length_p50) !== null || asFinite(sp.game_length_p90) !== null) {
    head.push(epochChip("len p50/p90", `${formatDecimal(sp.game_length_p50, 0)}/${formatDecimal(sp.game_length_p90, 0)}`));
  }
  if (asFinite(sp.root_policy_entropy_mean) !== null) head.push(epochChip("entropy", formatDecimal(sp.root_policy_entropy_mean, 2)));
  if (asFinite(sp.gumbel_play_winner_rate) !== null) head.push(epochChip("winner rate", formatPercent(sp.gumbel_play_winner_rate)));
  if (asFinite(sp.p0_win_share) !== null) head.push(epochChip("P0 share", formatPercent(sp.p0_win_share)));
  if (asFinite(tr.loss_policy) !== null || asFinite(tr.loss_value) !== null) {
    head.push(epochChip("loss pol/val", `${formatDecimal(tr.loss_policy, 2)}/${formatDecimal(tr.loss_value, 2)}`));
  }
  if (asFinite(sel.reuse_ratio) !== null) head.push(epochChip("reuse", `${formatDecimal(sel.reuse_ratio, 2)}×`));
  if (asFinite(sp.elapsed_seconds) !== null) head.push(epochChip("sp wall", formatSeconds(sp.elapsed_seconds)));
  if (asFinite(tr.train_seconds) !== null) head.push(epochChip("train wall", formatSeconds(tr.train_seconds)));

  const badges = histTelemBadges(sp, sel);
  const phaseTable = histTelemPhaseTable(sp);
  const detailChips = histTelemDetailChips(sp, sel, tr);
  const evalEdges = histTelemEvalEdges(rec.eval);
  const sparks = histTelemSparklines(entry.epochs, epoch);

  const detailSection = phaseTable || detailChips || evalEdges
    ? `<details class="hist-telem-detail" open><summary>Detail</summary>${phaseTable}${detailChips}${evalEdges ? `<div class="hist-telem-eval-label">Eval edges</div>${evalEdges}` : ""}</details>`
    : "";

  if (!head.length && !badges && !detailSection && !sparks) return "";
  return `<div class="hist-insp-group hist-insp-telem">
    <span class="hist-insp-group-title">Telemetry</span>
    ${badges}
    ${head.length ? `<div class="hist-insp-chips">${head.join("")}</div>` : ""}
    ${sparks}
    ${detailSection}
  </div>`;
}

function renderHistEpochInspector(runs) {
  if (!histEpochInspector) return;
  if (!histInspectEpoch) {
    histEpochInspector.innerHTML = "";
    histEpochInspector.hidden = true;
    return;
  }
  const runName = histInspectEpoch.run;
  const epoch = histInspectEpoch.epoch;
  const run = histInspRunPayload(runName, runs);
  const rows = histInspRunRows(runName, runs);
  const idx = rows.findIndex(row => Number(row.epoch) === epoch);
  const row = idx >= 0 ? rows[idx] : null;
  const prev = idx > 0 ? rows[idx - 1] : null;

  // Lazy epoch payload: fired once per (run, epoch); re-renders read the cache
  // synchronously. Never fetch for a row the run payload doesn't know (stale).
  let info = null;
  if (row) {
    info = histEpochInfoCache.get(`${runName}::${epoch}`) || null;
    if (!info) {
      loadHistEpochInfo(runName, epoch);
      info = histEpochInfoCache.get(`${runName}::${epoch}`) || { status: "loading", payload: null };
    }
  }
  const infoPayload = info && info.status === "ready" ? info.payload : null;

  const ckptSaved = !!(row && row.checkpoint && (row.checkpoint.path || row.checkpoint.name));
  const head = `<div class="hist-insp-head">
    <span class="hist-insp-title">Epoch ${escapeText(epoch)}</span>
    ${(runs || []).length > 1 || !row ? `<span class="hist-insp-run" title="${escapeAttr(runName)}">${escapeText(runName)}</span>` : ""}
    <span class="hist-epoch-tag hist-epoch-tag-${ckptSaved ? "saved" : "pending"}">${ckptSaved ? "saved" : "pending"}</span>
    <span class="hist-insp-ckpt-line" data-hist-insp-ckptline>${histInspCkptLineInner(row, info)}</span>
    <span class="hist-insp-spacer"></span>
    <button class="hist-insp-btn" type="button" data-hist-inspect-step="-1" title="Previous epoch"${prev ? "" : " disabled"}>◀</button>
    <button class="hist-insp-btn" type="button" data-hist-inspect-step="1" title="Next epoch"${idx >= 0 && idx < rows.length - 1 ? "" : " disabled"}>▶</button>
    <button class="hist-insp-btn" type="button" data-hist-epoch="${epoch}" title="Filter the games list to epoch ${epoch}">filter games to e${epoch}</button>
    <button class="hist-insp-btn" type="button" data-hist-inspect-close title="Close inspector (Esc)" aria-label="Close epoch inspector">✕</button>
  </div>`;

  // Stale-state guard: keep the inspector open (the run may simply not be
  // re-fetched yet mid-refresh) but render only the header + a muted line.
  if (!row) {
    histEpochInspector.innerHTML = `${head}<div class="hist-insp-note">Epoch data not loaded for this run yet.</div>`;
    histEpochInspector.hidden = false;
    return;
  }

  const sp = row.selfplay && typeof row.selfplay === "object" ? row.selfplay : {};
  const buf = sp.buffer && typeof sp.buffer === "object" ? sp.buffer : null;
  const prevSp = prev && prev.selfplay && typeof prev.selfplay === "object" ? prev.selfplay : {};
  const prevBuf = prevSp.buffer && typeof prevSp.buffer === "object" ? prevSp.buffer : null;

  const ckptName = histCkptNameForEpoch(row, epoch, infoPayload);
  if (ckptName && !histCkptInfoCache.get(`${runName}::${ckptName}`)) loadHistCkptInfo(runName, ckptName);
  const ckptEntry = ckptName ? histCkptInfoCache.get(`${runName}::${ckptName}`) : null;

  const configInner = histInspConfigInner(info);
  const ckptInner = histInspCkptInner(ckptEntry, ckptName);
  // The model group stays present while the epoch fetch is pending (its config
  // holder fills via patch-in-place); it disappears only once both sources are
  // known-empty.
  const modelGroup = configInner || ckptInner || !info || info.status === "loading"
    ? `<div class="hist-insp-group hist-insp-model"><span class="hist-insp-group-title">Model</span><div data-hist-insp-config>${configInner}</div><div class="hist-insp-ckpt" data-hist-insp-ckpt data-ckpt-name="${escapeAttr(ckptName || "")}">${ckptInner}</div></div>`
    : "";

  const groups = [
    histInspLossGroup(buf, prevBuf),
    histInspGameStatsGroup(sp),
    histInspTelemetryGroup(runName, epoch),
    histInspBufferGroup(buf, row.samples),
    histInspCalibrationGroup(buf, prevBuf),
    histInspEvalGroup(row, run, epoch),
    histInspMultistageGroup(row, run, epoch),
    histInspTrainGroup(row.training),
    `<div class="hist-insp-group hist-insp-diag" data-hist-insp-diag>${histInspDiagInner(info)}</div>`,
    modelGroup,
  ].filter(Boolean).join("");

  histEpochInspector.innerHTML = `${head}<div class="hist-insp-groups">${groups}</div>`;
  histEpochInspector.hidden = false;
}

// Patch-in-place after the epoch fetch resolves (no full re-render — a full
// render mid-async races the 15s refresh, same rationale as loadHistThumb).
function histInspPatchLazy(runName, epoch) {
  if (!histInspectEpoch || histInspectEpoch.run !== runName || histInspectEpoch.epoch !== epoch) return;
  if (!histEpochInspector || histEpochInspector.hidden) return;
  const info = histEpochInfoCache.get(`${runName}::${epoch}`) || null;
  const infoPayload = info && info.status === "ready" ? info.payload : null;
  const row = histInspRunRows(runName, null).find(r => Number(r.epoch) === epoch) || null;
  const diag = histEpochInspector.querySelector("[data-hist-insp-diag]");
  if (diag) diag.innerHTML = histInspDiagInner(info);
  const config = histEpochInspector.querySelector("[data-hist-insp-config]");
  if (config) config.innerHTML = histInspConfigInner(info);
  const line = histEpochInspector.querySelector("[data-hist-insp-ckptline]");
  if (line) line.innerHTML = histInspCkptLineInner(row, info);
  const holder = histEpochInspector.querySelector("[data-hist-insp-ckpt]");
  if (holder) {
    // The checkpoint may only now be derivable (row had no checkpoint entry
    // but the endpoint stat-confirmed the canonical file).
    const ckptName = histCkptNameForEpoch(row, epoch, infoPayload);
    holder.dataset.ckptName = ckptName || "";
    if (ckptName && !histCkptInfoCache.get(`${runName}::${ckptName}`)) loadHistCkptInfo(runName, ckptName);
    holder.innerHTML = histInspCkptInner(ckptName ? histCkptInfoCache.get(`${runName}::${ckptName}`) : null, ckptName);
    const model = histEpochInspector.querySelector(".hist-insp-model");
    if (model && info && info.status !== "loading" && !holder.innerHTML && !(config && config.innerHTML)) {
      model.innerHTML = "";  // both sources known-empty -> :empty hides the card
    }
  }
}

// The two lazy fetchers (P1.5) mirror loadHistThumb exactly: dedupe via the
// "loading" cache state, FIFO-evict, plain fetch + safeJson (NEVER
// pendingRequest), patch in place only while still relevant. Error entries
// stay cached so a re-render never retries a failed fetch in a loop.
async function loadHistEpochInfo(runName, epoch) {
  const key = `${runName}::${epoch}`;
  const cached = histEpochInfoCache.get(key);
  if (cached && (cached.status === "loading" || cached.status === "ready")) return;
  histEpochInfoCache.set(key, { status: "loading", payload: null });
  while (histEpochInfoCache.size > HIST_EPOCH_INFO_CACHE_MAX) {
    histEpochInfoCache.delete(histEpochInfoCache.keys().next().value);
  }
  let entry;
  try {
    const params = new URLSearchParams({ run: runName, epoch: String(epoch) });
    const res = await fetch(`/api/training/epoch?${params.toString()}`);
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "epoch info unavailable");
    entry = { status: "ready", payload: data && typeof data === "object" ? data : {} };
  } catch (error) {
    console.warn("loadHistEpochInfo failed", error);
    entry = { status: "error", payload: null };
  }
  histEpochInfoCache.set(key, entry);
  histInspPatchLazy(runName, epoch);
}

async function loadHistCkptInfo(runName, ckptName) {
  const key = `${runName}::${ckptName}`;
  const cached = histCkptInfoCache.get(key);
  if (cached && (cached.status === "loading" || cached.status === "ready")) return;
  histCkptInfoCache.set(key, { status: "loading", payload: null });
  while (histCkptInfoCache.size > HIST_CKPT_INFO_CACHE_MAX) {
    histCkptInfoCache.delete(histCkptInfoCache.keys().next().value);
  }
  let entry;
  try {
    const params = new URLSearchParams({ run: runName, checkpoint: ckptName });
    const res = await fetch(`/api/debug/ckpt_info?${params.toString()}`);
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "ckpt info unavailable");
    entry = { status: "ready", payload: data && typeof data === "object" ? data : {} };
  } catch (error) {
    console.warn("loadHistCkptInfo failed", error);
    entry = { status: "error", payload: null };
  }
  histCkptInfoCache.set(key, entry);
  // Patch only while still inspected AND the derived name is unchanged.
  if (!histInspectEpoch || histInspectEpoch.run !== runName) return;
  if (!histEpochInspector || histEpochInspector.hidden) return;
  const holder = histEpochInspector.querySelector("[data-hist-insp-ckpt]");
  if (holder && holder.dataset.ckptName === ckptName) holder.innerHTML = histInspCkptInner(entry, ckptName);
}

// H8: one-line game row — winner dot, "g<rec> e<epoch>" label, source badge,
// length with a micro-bar relative to the longest displayed row, truncated
// player labels, modified date, compact Replay. The whole row body is one
// full-width selection button (data-history-key, handled by the existing
// handleGameHistoryClick delegation).
function gameHistoryListRow(runName, item, maxLen) {
  const key = historyItemKey(item);
  const selected = key === selectedHistoryKey;
  const len = historyLength(item);
  const barPct = maxLen > 0 ? Math.max(2, Math.round((len / maxLen) * 100)) : 0;
  const epochNum = asFinite(item.epoch);
  const source = String(item.source || "history");
  const srcClass = source === "selfplay" ? "hist-src-selfplay" : source === "evaluation" ? "hist-src-evaluation" : "hist-src-other";
  const srcText = source === "selfplay" ? "SP" : source === "evaluation" ? "EV" : "H";
  const p0 = historyPlayerLabel(item.players && item.players.player0);
  const p1 = historyPlayerLabel(item.players && item.players.player1);
  const candSeat = historyCandidateSeat(item);
  // Explicit seats ("P0 x · P1 y"); mark the candidate seat with a leading dot
  // so the current model is visible at a glance. Selfplay/no-candidate rows just
  // show plain P0/P1 labels.
  const candMark = `<span class="hist-cand-mark" title="Current model">●</span>`;
  const p0Seat = `<span class="hist-seat${candSeat === "player0" ? " hist-seat-cand" : ""}">${candSeat === "player0" ? candMark : ""}P0 ${escapeText(p0)}</span>`;
  const p1Seat = `<span class="hist-seat${candSeat === "player1" ? " hist-seat-cand" : ""}">${candSeat === "player1" ? candMark : ""}P1 ${escapeText(p1)}</span>`;
  const outcomeBadge = historyOutcomeBadge(item);
  const winnerDot = item.winner
    ? `<span class="hist-win-dot" style="background:${playerColor(item.winner)}" title="${escapeAttr(item.winner_label || winnerLabel(item.winner))}"></span>`
    : `<span class="hist-win-dot hist-win-none" title="No winner"></span>`;
  const rowTitle = `${item.game_id || item.path || ""} · ${runName || "run"} · ${historyDiagnosticsText(item.diagnostics)}`;
  return `<div class="hist-game-row${selected ? " selected" : ""}" data-history-key="${escapeAttr(key)}">
    <button class="hist-game-main" type="button" data-history-key="${escapeAttr(key)}" title="${escapeAttr(rowTitle)}">
      ${winnerDot}
      ${outcomeBadge}
      <span class="hist-game-label">g${escapeText(Number(item.record_index || 0))} e${epochNum !== null ? escapeText(epochNum) : "--"}</span>
      <span class="hist-src-badge ${srcClass}" title="${escapeAttr(source)}">${srcText}</span>
      <span class="hist-game-len"><b>${escapeText(len)}</b><span class="hist-len-bar"><span style="width:${barPct}%"></span></span></span>
      <span class="hist-game-players">${p0Seat} · ${p1Seat}</span>
      <span class="hist-game-date">${escapeText(formatHistoryDate(item.modified))}</span>
    </button>
    <button class="hist-game-replay" type="button" data-history-load data-history-run="${escapeAttr(runName || "")}" data-history-path="${escapeAttr(item.path)}" data-record-index="${escapeAttr(item.record_index || 0)}" title="Load replay on the Match board">Replay</button>
  </div>`;
}

// P2.2: Selected Game panel — nav row / outcome line / thumbnail / actions /
// collapsed details disclosure. The disclosure's open state is intentionally
// NOT persisted: the 15s innerHTML refresh collapses it again.
function gameHistoryDetailHtml(runName, item) {
  const winner = item.winner_label || winnerLabel(item.winner);
  const diagnostics = item.diagnostics || {};
  const p0 = item.players && item.players.player0;
  const p1 = item.players && item.players.player1;
  // P2.1 nav row: position within the displayed-list order; the selected key
  // is absent when selection fell back outside the epoch filter — nav disables.
  const navIdx = histDisplayedKeys.indexOf(selectedHistoryKey);
  const navTotal = histDisplayedKeys.length;
  // H9: final-position thumbnail, .hxr records only (page items carry no
  // placements — they come from one lazy replay fetch, cached per item key).
  let thumb = "";
  if (String(item.path || "").endsWith(".hxr")) {
    const key = historyItemKey(item);
    let entry = histThumbCache.get(key);
    if (!entry) {
      loadHistThumb(item);
      entry = histThumbCache.get(key) || { status: "loading", placements: null };
    }
    thumb = `<div class="hist-thumb">${histThumbInner(entry)}</div>`;
  }
  // item.abort is an object ({stage, exception_type, message}), not a string.
  const abort = item.abort && typeof item.abort === "object" ? item.abort : null;
  const abortText = abort
    ? [abort.stage, abort.exception_type, abort.message].filter(Boolean).map(String).join(": ") || "aborted"
    : (item.abort ? String(item.abort) : "");
  const epochNum = asFinite(item.epoch);
  const candSeat = historyCandidateSeat(item);
  const outcomeBadge = historyOutcomeBadge(item);
  // Explicit P0/P1 identities on the outcome line (also kept in the collapsed
  // Players section below); the candidate seat is marked when identifiable.
  const p0Name = historyPlayerLabel(p0);
  const p1Name = historyPlayerLabel(p1);
  const candMark = `<span class="hist-cand-mark" title="Current model">●</span>`;
  const p0Bit = `<span class="hist-outcome-bit hist-seat${candSeat === "player0" ? " hist-seat-cand" : ""}">${candSeat === "player0" ? candMark : ""}P0 ${escapeText(p0Name)}</span>`;
  const p1Bit = `<span class="hist-outcome-bit hist-seat${candSeat === "player1" ? " hist-seat-cand" : ""}">${candSeat === "player1" ? candMark : ""}P1 ${escapeText(p1Name)}</span>`;
  const winnerDot = item.winner
    ? `<span class="hist-win-dot" style="background:${playerColor(item.winner)}"></span>`
    : `<span class="hist-win-dot hist-win-none"></span>`;
  return `<div class="history-detail-body">
    <div class="hist-game-nav">
      <button id="histGamePrev" class="hist-insp-btn" type="button" data-hist-game-step="-1" title="Previous game (Left arrow)"${navIdx > 0 ? "" : " disabled"}>◀</button>
      <span class="hist-game-nav-pos">${navIdx >= 0 ? navIdx + 1 : "--"} / ${navTotal}</span>
      <button id="histGameNext" class="hist-insp-btn" type="button" data-hist-game-step="1" title="Next game (Right arrow)"${navIdx >= 0 && navIdx < navTotal - 1 ? "" : " disabled"}>▶</button>
    </div>
    <div class="hist-outcome-line">
      ${winnerDot}
      ${outcomeBadge}
      <strong class="${winnerClass(item.winner)}">${escapeText(winner)}</strong>
      <span class="hist-outcome-bit">${escapeText(item.length || item.actions || 0)} moves</span>
      <span class="hist-outcome-bit">${escapeText(item.source || "history")}</span>
      <span class="hist-outcome-bit">e${epochNum !== null ? escapeText(epochNum) : "--"}</span>
      <span class="hist-outcome-bit">${escapeText(item.status || "unknown")}</span>
      ${p0Bit}
      ${p1Bit}
    </div>
    ${thumb}
    <div class="history-detail-actions">
      <button class="primary-action history-replay-btn" type="button" data-history-load data-history-run="${escapeAttr(runName || "")}" data-history-path="${escapeAttr(item.path)}" data-record-index="${escapeAttr(item.record_index || 0)}">Load Replay</button>
      ${String(item.path || "").endsWith(".hxr") ? `<button class="history-debug-btn" type="button" data-debug-open data-debug-run="${escapeAttr(runName || "")}" data-debug-path="${escapeAttr(item.path)}" data-debug-record="${escapeAttr(item.record_index || 0)}">Open in Debug</button>` : ""}
    </div>
    <details class="hist-detail-more">
      <summary>Details</summary>
      <div class="hist-detail-more-body">
        <div class="detail-stack">
          ${detailRow("Run", runName || "Unknown")}
          ${detailRow("Game", item.game_id || "Unknown")}
          ${detailRow("Status", item.status || "unknown")}
          ${detailRow("Seed", item.seed === null || item.seed === undefined ? "--" : item.seed)}
          ${detailRow("Record", Number(item.record_index || 0))}
          ${detailRow("Path", item.path || "—", item.path || "")}
          ${detailRow("Modified", formatHistoryDate(item.modified))}
        </div>
        <div class="history-detail-section">
          <div class="detail-section-title">Players</div>
          <div class="player-detail-grid">
            ${playerDetail("P0", p0)}
            ${playerDetail("P1", p1)}
          </div>
        </div>
        <div class="history-detail-section">
          <div class="detail-section-title">Diagnostics</div>
          ${diagnosticDetailsHtml(diagnostics)}
        </div>
        ${abortText ? `<div class="history-detail-section"><div class="detail-section-title">Abort</div><div class="detail-note">${escapeText(abortText)}</div></div>` : ""}
      </div>
    </details>
  </div>`;
}

// --- H9: final-position thumbnail (lazy, cached, .hxr detail only). ---
const HIST_THUMB_CACHE_MAX = 40;

function histThumbInner(entry) {
  if (entry && entry.status === "ready") return histThumbSvg(entry.placements);
  if (entry && entry.status === "error") return `<div class="hist-thumb-loading">Final position unavailable</div>`;
  return `<div class="hist-thumb-loading">Loading final position</div>`;
}

// Pure-presentation SVG: one circle per placement at the shared hex geometry,
// final move ringed, viewBox = stone bounding box padded by 2*HEX.
function histThumbSvg(placements) {
  const stones = (placements || []).filter(p => p && asFinite(p.q) !== null && asFinite(p.r) !== null);
  if (!stones.length) return "";
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const pts = stones.map(p => {
    const c = center(Number(p.q), Number(p.r));
    if (c.x < minX) minX = c.x;
    if (c.x > maxX) maxX = c.x;
    if (c.y < minY) minY = c.y;
    if (c.y > maxY) maxY = c.y;
    return { x: c.x, y: c.y, player: p.player };
  });
  const pad = 2 * HEX;
  const radius = HEX * 0.62;
  const body = pts.map((pt, i) => {
    const stone = `<circle cx="${pt.x.toFixed(1)}" cy="${pt.y.toFixed(1)}" r="${radius.toFixed(1)}" style="fill:${playerColor(pt.player)}"></circle>`;
    return i === pts.length - 1
      ? `${stone}<circle cx="${pt.x.toFixed(1)}" cy="${pt.y.toFixed(1)}" r="${(radius + 3).toFixed(1)}" fill="none" style="stroke:var(--accent)" stroke-width="2"></circle>`
      : stone;
  }).join("");
  const viewBox = `${(minX - pad).toFixed(1)} ${(minY - pad).toFixed(1)} ${(maxX - minX + 2 * pad).toFixed(1)} ${(maxY - minY + 2 * pad).toFixed(1)}`;
  return `<svg viewBox="${viewBox}" role="img" aria-label="Final position">${body}</svg>`;
}

// One lazy replay fetch per selected .hxr item (the endpoint replays the game
// server-side, so: never for list rows, deduped via the "loading" cache state,
// FIFO-capped, and patched in place only if the item is still selected).
async function loadHistThumb(item) {
  const key = historyItemKey(item);
  const cached = histThumbCache.get(key);
  if (cached && (cached.status === "loading" || cached.status === "ready")) return;
  histThumbCache.set(key, { status: "loading", placements: null });
  while (histThumbCache.size > HIST_THUMB_CACHE_MAX) {
    histThumbCache.delete(histThumbCache.keys().next().value);
  }
  let entry;
  try {
    const params = new URLSearchParams({
      run: item.run || "",
      path: item.path || "",
      record: String(Number(item.record_index || 0)),
    });
    const res = await fetch(`/api/training/history?${params.toString()}`);
    const data = await safeJson(res);
    if (!res.ok) throw new Error((data && data.error) || "history unavailable");
    entry = { status: "ready", placements: Array.isArray(data && data.placements) ? data.placements : [] };
  } catch (error) {
    console.warn("loadHistThumb failed", error);
    entry = { status: "error", placements: null };
  }
  histThumbCache.set(key, entry);
  if (selectedHistoryKey === key && gameHistoryDetail) {
    const holder = gameHistoryDetail.querySelector(".hist-thumb");
    if (holder) holder.innerHTML = histThumbInner(entry);
  }
}

function detailRow(label, value, titleValue) {
  const title = titleValue ? ` title="${escapeAttr(titleValue)}"` : "";
  return `<div class="detail-row"><span>${escapeText(label)}</span><strong${title}>${escapeText(value)}</strong></div>`;
}

function playerDetail(slot, player) {
  const label = historyPlayerLabel(player);
  const kind = player && (player.kind || player.variant || player.id || "");
  return `<div class="player-detail">
    <span>${escapeText(slot)}</span>
    <strong>${escapeText(label)}</strong>
    <small>${escapeText(kind || "unknown")}</small>
  </div>`;
}

function diagnosticDetailsHtml(diagnostics) {
  if (!diagnostics || typeof diagnostics !== "object" || !Object.keys(diagnostics).length) {
    return `<div class="detail-note">No diagnostics attached to this game.</div>`;
  }
  return Object.entries(diagnostics).map(([label, diagnostic]) => {
    const summary = diagnostic && diagnostic.summary ? diagnostic.summary : {};
    const entries = Object.entries(summary);
    return `<div class="diagnostic-block">
      <div class="diagnostic-title">${escapeText(label)}</div>
      <div class="diagnostic-grid">
        ${entries.length ? entries.map(([key, value]) => `<div><span>${escapeText(key)}</span><strong>${escapeText(displayValue(value))}</strong></div>`).join("") : `<div><span>Artifact</span><strong>${escapeText(diagnostic && diagnostic.name || "attached")}</strong></div>`}
      </div>
    </div>`;
  }).join("");
}

function historyPlayerLabel(player) {
  if (!player) return "Unknown";
  return player.label || PLAYER_KIND_LABELS[player.kind] || player.kind || "Unknown";
}

// Which seat ("player0"/"player1") the run's own candidate net held in an eval
// game, or null. Set server-side (web.py _candidate_seat_from_game_id) from the
// record game_id's "-candPN" suffix; only evaluation games carry it. Anything
// other than a known seat string degrades to null (no wrong badge).
function historyCandidateSeat(item) {
  const seat = item && item.candidate_seat;
  return seat === "player0" || seat === "player1" ? seat : null;
}

// "win"/"loss" for the candidate net in an eval game, or null when it cannot be
// decided (not an eval game, no identifiable candidate seat, or no winner).
function historyCandidateOutcome(item) {
  if (!item || String(item.source || "") !== "evaluation") return null;
  const seat = historyCandidateSeat(item);
  if (!seat || !item.winner) return null;
  return item.winner === seat ? "win" : "loss";
}

// Compact color-coded W/L badge for the candidate net (eval games only).
// Returns "" when the outcome is undecidable so callers render nothing extra.
function historyOutcomeBadge(item) {
  const outcome = historyCandidateOutcome(item);
  if (!outcome) return "";
  const seat = historyCandidateSeat(item);
  const oppSeat = seat === "player0" ? "player1" : "player0";
  const oppLabel = historyPlayerLabel(item.players && item.players[oppSeat]);
  const won = outcome === "win";
  const title = `Current model (${seat === "player0" ? "P0" : "P1"}) ${won ? "won" : "lost"} vs ${oppLabel}`;
  return `<span class="hist-cand-badge hist-cand-${won ? "win" : "loss"}" title="${escapeAttr(title)}">${won ? "W" : "L"}</span>`;
}

function winnerLabel(winner) {
  if (winner === "player0") return "P0";
  if (winner === "player1") return "P1";
  return "None";
}

function historyDiagnosticsText(diagnostics) {
  if (!diagnostics || typeof diagnostics !== "object") return "None";
  const evalSummary = diagnostics.evaluation && diagnostics.evaluation.summary;
  if (evalSummary) {
    const parts = [];
    if (asFinite(evalSummary.games) !== null) parts.push(`${evalSummary.games}g`);
    if (asFinite(evalSummary.wins) !== null || asFinite(evalSummary.losses) !== null) parts.push(`${evalSummary.wins || 0}-${evalSummary.losses || 0}`);
    if (asFinite(evalSummary.mean_turns) !== null) parts.push(`${asFinite(evalSummary.mean_turns).toFixed(1)}t`);
    return parts.length ? parts.join(" ") : "Eval";
  }
  const selfplaySummary = diagnostics.selfplay && diagnostics.selfplay.summary;
  if (selfplaySummary) {
    const parts = [];
    if (asFinite(selfplaySummary.samples_added) !== null) parts.push(`${selfplaySummary.samples_added} samples`);
    if (asFinite(selfplaySummary.games) !== null) parts.push(`${selfplaySummary.games}g`);
    if (selfplaySummary.lengths && asFinite(selfplaySummary.lengths.mean) !== null) parts.push(`${asFinite(selfplaySummary.lengths.mean).toFixed(1)}t`);
    if (parts.length) return parts.join(" ");
    if (asFinite(selfplaySummary.searched_positions) !== null) return `${selfplaySummary.searched_positions} pos`;
    return "Selfplay";
  }
  return Object.keys(diagnostics).length ? Object.keys(diagnostics).join(", ") : "None";
}

function winnerClass(winner) {
  if (winner === "player0") return "p0";
  if (winner === "player1") return "p1";
  return "none";
}

function formatHistoryDate(value) {
  if (!value) return "--";
  const raw = Number(value);
  const date = Number.isFinite(raw) ? new Date(raw < 1000000000000 ? raw * 1000 : raw) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function formatBytes(value) {
  const bytes = asFinite(value) || 0;
  // Binary (1024-based) divisions, so use binary unit labels (KiB/MiB).
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

// ===========================================================================
// Debug tab — Model Debug v2 (single-position forensics workbench).
//
// Self-contained from the Match/History screens: its own board SVG, state and
// pan/zoom, reusing only the pure shared helpers (center/path/playerColor/HEX,
// viewForBox/clamp, escape*, formatBytes, loadTrainingRuns, navigateScreen,
// diagTap/reportError). All model outputs come from the CPU inference worker
// via /api/debug/* per scripts/_debug_rewrite_spec.md §3; nothing here touches
// the live-match polling.
//
// Canonical nav state (spec M1): one `dbg.nav` tuple serialized to the URL
// hash (`#debug?run=..&src=..&path=..&rec=..&ply=..&ckptA=..&ckptB=..&acts=..
// &tab=..&mode=..`). `dbgNavigate(patch, {replace})` is the ONLY mutation
// entry point — it writes the hash, and `dbgApplyHash()` (driven by
// `hashchange`) is the only code that applies state → fetch/render. Browser
// Back/Forward therefore step the workbench, and any URL restores the exact
// position including injected what-if moves and checkpoint B.
// ===========================================================================

const DBG_TABS = ["heads", "search", "targets", "compare", "inputs", "ckpt", "attn"];
const DBG_TAB_IDS = {
  heads: "dbgTabHeads",
  search: "dbgTabSearch",
  targets: "dbgTabTargets",
  compare: "dbgTabCompare",
  inputs: "dbgTabInputs",
  ckpt: "dbgTabCkpt",
  attn: "dbgTabAttn",
};
// Base heat modes — index in DBG_MODE_ORDER is the 1..9 keyboard digit.
// "attn" sits at index 7 (digit 8); inserting it shifts "plane" to digit 9.
// "cellq" is the 10th entry — it gets a mode button (click keys off data-mode)
// but NO 1..9 digit hotkey (only the first 9 modes are reachable by digit).
const DBG_MODE_ORDER = ["prior", "visits", "delta", "opp", "target", "mismatch", "childq", "attn", "plane", "cellq"];
const DBG_MODES = ["none"].concat(DBG_MODE_ORDER);
const DBG_CACHE_MAX = 300;  // spec M12: client analysis cache LRU cap
// cell_q regret blunder default (mover POV, regret = bestQ - playedQ). Tiers for
// UI coloring: minor >=0.15 / blunder >=0.30 / catastrophic >=0.60.
const DBG_REGRET_BLUNDER = 0.3;
// Action ids are packed exactly like hexo_engine.types.pack_coord_id (which
// mirrors the Rust legal-move store, so the packing is pinned): the board's
// click-to-inject needs an id for ANY empty legal cell, and the position
// payload's legal list carries only coordinates.
const DBG_COORD_OFFSET = 32768;

const dbg = {
  inited: false,
  loading: false,
  // Canonical nav tuple (spec M1). `ply: null` means "end of game, not yet
  // resolved" — dbgApplyNav replaces it with the real total once known.
  nav: dbgDefaultNav(),
  navApplied: "",      // last hash fully handed to dbgApplyNav (dedupes the two hashchange listeners)
  pendingDeepLink: null,
  // Loaded source lists + their identity (so run/src switches reload them).
  loadedRun: "",
  loadedGamesKey: "",
  games: [],
  checkpoints: [],
  worker: null,        // worker status from /api/debug/checkpoints
  lineage: "",         // run lineage from /api/debug/checkpoints (shrimp|dense_cnn|hexgt|"")
  radiusDetected: null, // support_radius_detected from /api/debug/checkpoints (int|null) — Auto seed
  records: [],         // record_games from the last recorded position payload
  // Current data + the cache key each piece was rendered for (stale dots, M14).
  position: null,
  analysis: null,      // ckptA analyze
  analysisB: null,     // ckptB analyze (COMPARE)
  search: null,        // /api/debug/search result
  searchBusy: false,
  tree: null,          // /api/debug/search_tree result (childq/Q/scatter/PV read it)
  treeBusy: false,
  attn: null,          // /api/debug/attention payload for the current attn cache slot
  attnQuery: null,     // {type:"cell"|"token", id:int} | null — active attention query
  attnBlock: 0,        // attn block index (mirrors nav.attnblk)
  attnNumBlocks: 3,    // arch-dependent (e.g. 3 blocks x 4 heads, or the main_7 5 blocks x 3 heads);
  attnNumHeads: 4,     // refreshed from every attention payload's num_blocks/num_heads
  attnHead: null,      // attn head 0..3 or null=mean (mirrors nav.attnhead)
  attnPending: false,  // an attention fetch is in flight for the current slot
  treeOpen: new Map(), // tree-node path -> explicit expand/collapse (default: PV open)
  treePreview: null,   // {key, ids} — clicked tree node's line, ghosted on the board
  treeGhostsOff: false, // T with a fresh tree toggles its board ghosts (§1.7 "run / toggle")
  ladder: null,        // visit-ladder rows for ladderKey (spec S10)
  ladderBusy: false,
  // Game Error Sweeps (spec M9), keyed run|path|rec|ckptA so flipping checkpoints
  // or games keeps each sweep; sweepRun is the single in-flight chunk loop.
  sweeps: new Map(),
  sweepRun: null,
  trajectory: null,
  trajectoryKey: "",
  cmpHeat: false,      // COMPARE: client-computed prior Δ(A−B) board overlay
  split: false,        // COMPARE: A/B split-board mode (spec S5)
  pins: [],            // pin tray (spec S2), localStorage hexoDbgPins:<run>
  pinsRun: "",
  journal: [],         // session journal (spec S3), localStorage hexoDbgJournal:<run>
  journalRun: "",
  paletteItems: [],    // command palette (spec S1)
  paletteShown: [],
  paletteIdx: 0,
  paletteGames: [],    // all-source game list for the palette (lazy)
  paletteGamesRun: "",
  ckptSweep: null,     // checkpoint sweep dock tab (spec S4)
  prefetchAbort: null, // the ONE in-flight ply±1 prefetch (spec S7)
  keys: { position: "", analysis: "", analysisB: "", search: "", tree: "", attn: "" },
  // Board UI state that intentionally does NOT live in the hash.
  overlays: { threats: false, numbers: true, last: true, legalDim: false },
  logScale: false,
  opacity: 0.9,
  sortKey: "prior",    // top-moves table sort column
  dockTab: "trajectory",
  // cell_q regret blunder marking (M9 augment): "absolute" thresholds the raw
  // regret; "relative" flags the top decile of THIS game's nonzero regrets.
  regretThreshold: 0.3,
  regretMode: "absolute",
  hudMetrics: new Map(),  // per-render `q,r` -> metrics, so the hover HUD never does a linear .find (M3)
  cellIndex: new Map(),   // per-render `q,r` -> cell info for click routing
  // Debug-board pan/zoom (own copies — the Match board owns the globals).
  view: null,
  baseView: null,
  viewDirty: false,
  suppressClick: false,
  // Monotonic request tokens so the LATEST navigation/analysis always wins. The
  // dense_cnn_restnet CPU forward is far slower than hexgt's tiny graph forward,
  // so without these guards a slow/stale position fetch or analyze can resolve
  // AFTER a newer ply click and clobber dbg.position/dbg.analysis — which wedged
  // forward/back stepping and slider scrubbing for that lineage. Each load/analyze
  // claims a token and only commits if it is still the newest.
  posSeq: 0,
  anlSeq: 0,
  anlSeqB: 0,
  attnSeq: 0,
  applySeq: 0,
  // Client caches (spec M12): one LRU keyed run|path|rec|ply|ckpt|acts holding
  // {position, analysis, search, tree, record_row}; revisiting a ply, undoing a
  // chip, or flipping ckptA<->ckptB re-renders with zero network requests.
  cache: new Map(),
  ckptInfo: new Map(),    // dbgCkptInfoKey() `${run}|${ckpt}|${radius}` -> /api/debug/ckpt_info payload (B3/M7)
  // Full ordered stone list + action ids for the current RECORDED game, captured
  // from any loaded position. Lets prev/next/slider re-render the board for any
  // ply INSTANTLY client-side (stones with index <= ply) without waiting on the
  // server position fetch or the slow ResTNet analyze.
  gamePlacements: [],
  gameKey: "",
  gameActs: [],
  gameActsKey: "",
  placementsBackfill: "",  // gameKey already backfilled to the final ply (one-shot)
};

const dbgEl = id => document.getElementById(id);
const debugBoardSvg = dbgEl("debugBoardSvg");

function debugSetStatus(message, kind = "info") {
  // Mirror real errors to the always-on-top banner so they're unmissable on a
  // phone (the small inline status line is easy to miss on a narrow screen).
  if (message && kind === "error") reportError(message);
  const el = dbgEl("debugStatus");
  if (!el) return;
  if (!message) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.className = `debug-status debug-status-${kind}`;
  el.textContent = message;
}

async function debugFetchJson(url, options) {
  const res = await fetch(url, options);
  const data = await safeJson(res);
  if (!res.ok || (data && data.error)) {
    // A 404 on a /api/debug/* route means the running server predates the Debug
    // backend (its routes were loaded at process start). Static assets are read
    // from disk per request, so the tab can appear while the API is missing —
    // say so plainly instead of a cryptic "HTTP 404".
    if (res.status === 404 && url.indexOf("/api/debug/") !== -1) {
      throw new Error("Debug API not found — restart the dashboard server to load the new Debug backend (it must serve the current hexo_frontend from this worktree).");
    }
    throw new Error((data && data.error) || `HTTP ${res.status}`);
  }
  return data;
}

function dbgStub(message) {
  diagTap(message);
  debugSetStatus(message, "info");
}

// ---- action-id packing (mirrors hexo_engine.types.pack/unpack_coord_id) ----

function dbgPackActionId(q, r) {
  return (q + DBG_COORD_OFFSET) * 65536 + (r + DBG_COORD_OFFSET);
}

function dbgUnpackActionId(actionId) {
  const id = Number(actionId);
  return { q: Math.floor(id / 65536) - DBG_COORD_OFFSET, r: (id % 65536) - DBG_COORD_OFFSET };
}

// ---- nav state + URL hash (spec M1) ----------------------------------------

function dbgDefaultNav() {
  return { run: "", src: "selfplay", path: "", rec: 0, ply: null, ckptA: "", ckptB: "", acts: [], tab: "heads", mode: "prior", attnq: "", attnblk: 0, attnhead: "", radius: 0 };
}

function dbgNavToHash(nav) {
  // Deterministic field order so identical navs serialize identically (the
  // dedupe in dbgApplyHash relies on it). `/` and `,` stay readable.
  const enc = v => encodeURIComponent(String(v)).replace(/%2F/gi, "/");
  const parts = [];
  if (nav.run) parts.push("run=" + enc(nav.run));
  if (nav.src && nav.src !== "selfplay") parts.push("src=" + enc(nav.src));
  if (nav.path) parts.push("path=" + enc(nav.path));
  if (nav.rec) parts.push("rec=" + String(nav.rec));
  if (nav.ply != null) parts.push("ply=" + String(nav.ply));
  if (nav.ckptA) parts.push("ckptA=" + enc(nav.ckptA));
  if (nav.ckptB) parts.push("ckptB=" + enc(nav.ckptB));
  if (nav.acts && nav.acts.length) parts.push("acts=" + nav.acts.join(","));
  if (nav.tab && nav.tab !== "heads") parts.push("tab=" + enc(nav.tab));
  if (nav.mode && nav.mode !== "prior") parts.push("mode=" + enc(nav.mode));
  // Attention query/block/head (additive; only when a query has been made).
  if (nav.attnq) parts.push("attnq=" + enc(nav.attnq));
  if (nav.attnblk) parts.push("attnblk=" + String(nav.attnblk));
  if (nav.attnhead !== "" && nav.attnhead != null) parts.push("attnhead=" + enc(String(nav.attnhead)));
  // Manual support-radius override (4|8); omit on Auto (0) so existing deep links
  // are unchanged and the backend default/detection path is exercised.
  if (nav.radius) parts.push("radius=" + String(nav.radius));
  return "#debug" + (parts.length ? "?" + parts.join("&") : "");
}

function dbgHashToNav(hash) {
  const nav = dbgDefaultNav();
  const qi = hash.indexOf("?");
  if (qi === -1) return nav;
  const params = new URLSearchParams(hash.slice(qi + 1));
  nav.run = params.get("run") || "";
  nav.src = params.get("src") === "evaluation" ? "evaluation" : "selfplay";
  nav.path = params.get("path") || "";
  nav.rec = Math.max(0, parseInt(params.get("rec"), 10) || 0);
  const ply = parseInt(params.get("ply"), 10);
  nav.ply = Number.isFinite(ply) ? Math.max(0, ply) : null;
  nav.ckptA = params.get("ckptA") || "";
  nav.ckptB = params.get("ckptB") || "";
  nav.acts = (params.get("acts") || "").split(",").map(s => parseInt(s, 10)).filter(n => Number.isFinite(n));
  const tab = params.get("tab");
  nav.tab = DBG_TABS.includes(tab) ? tab : "heads";
  const mode = params.get("mode");
  nav.mode = DBG_MODES.includes(mode) ? mode : "prior";
  // Attention query/block/head — validated lightly (worker clamps authoritatively).
  const aq = params.get("attnq") || "";
  nav.attnq = /^(cell|token):-?\d+$/.test(aq) ? aq : "";
  // Loose URL clamp only (arch block count is unknown until a payload lands);
  // the worker clamps authoritatively to the loaded model's block count.
  const ablk = parseInt(params.get("attnblk"), 10);
  nav.attnblk = Number.isFinite(ablk) ? Math.max(0, Math.min(ablk, 31)) : 0;
  const ah = params.get("attnhead");
  if (ah == null || ah === "" || ah === "mean") nav.attnhead = "";
  else if (ah === "max") nav.attnhead = "max";
  else {
    const ahn = parseInt(ah, 10);
    nav.attnhead = Number.isFinite(ahn) ? Math.max(0, Math.min(ahn, 3)) : "";
  }
  // Support-radius override: only 4 or 8 are valid; anything else is Auto (0).
  const rad = parseInt(params.get("radius"), 10);
  nav.radius = (rad === 4 || rad === 8) ? rad : 0;
  return nav;
}

// Parse an attnq string ("cell:33152" | "token:3") into {type, id} | null.
function dbgParseAttnQuery(attnq) {
  const m = /^(cell|token):(-?\d+)$/.exec(String(attnq || ""));
  if (!m) return null;
  return { type: m[1], id: Number(m[2]) };
}

function dbgNavigate(patch, { replace = false } = {}) {
  dbgAbortPrefetch();  // S7: any user action kills the speculative ply±1 fetch
  const nav = Object.assign({}, dbg.nav, patch);
  if (patch && patch.acts) nav.acts = patch.acts.slice();
  const hash = dbgNavToHash(nav);
  if (String(window.location.hash || "") === hash) {
    // Explicit re-selection of the identical tuple: clear the dedupe so an apply
    // that failed midway can be retried (re-apply on a cache hit is idempotent).
    dbg.navApplied = "";
    dbgApplyHash();
    return;
  }
  // Optimistic nav update: `location.hash =` fires hashchange ASYNCHRONOUSLY,
  // so without this a rapid key-repeat step would read the stale nav and drop
  // steps. dbgApplyHash re-parses the same tuple from the hash when it lands.
  dbg.nav = nav;
  if (replace) {
    // replaceState fires no hashchange — apply directly (used for default
    // resolution, slider scrubs, and tab/mode flips that shouldn't spam history).
    window.history.replaceState(null, "", hash);
    dbgApplyHash();
  } else {
    window.location.hash = hash;  // hashchange -> dbgApplyHash, the ONLY applier
  }
}

function dbgApplyHash() {
  if (!dbg.inited) return;
  if (dbg.pendingDeepLink) {
    const link = dbg.pendingDeepLink;
    dbg.pendingDeepLink = null;
    dbgNavigate(dbgDeepLinkPatch(link), { replace: true });
    return;
  }
  const hash = String(window.location.hash || "");
  if (!hash.startsWith("#debug")) return;
  if (hash === dbg.navApplied) return;  // both hashchange listeners route here — apply once
  if (hash === "#debug" && dbg.nav.run && dbg.nav.path) {
    // A bare `#debug` (setScreen's replaceState strips the query when the nav
    // buttons are used) restores the in-memory position instead of resetting.
    dbgNavigate(dbg.nav, { replace: true });
    return;
  }
  dbg.navApplied = hash;
  dbg.nav = dbgHashToNav(hash);
  const seq = ++dbg.applySeq;
  dbgApplyNav(dbg.nav, seq).catch(e => {
    if (seq === dbg.applySeq) dbg.navApplied = "";  // failed tuple stays re-appliable
    debugSetStatus(`Debug: ${(e && e.message) || e}`, "error");
    reportError("dbgApplyNav: " + (e && (e.stack || e.message) || e));
  });
}

function dbgDeepLinkPatch(link) {
  // detail from [data-debug-open]: { run, path, record, ply } — ply null = final.
  const patch = {
    run: link.run || dbg.nav.run,
    src: link.path && String(link.path).startsWith("eval") ? "evaluation" : "selfplay",
    path: link.path || "",
    rec: Number(link.record != null ? link.record : link.rec) || 0,
    ply: link.ply != null ? Number(link.ply) : null,
    acts: [],
  };
  if (patch.run !== dbg.nav.run) {
    patch.ckptA = "";  // checkpoints are per-run; let apply re-default
    patch.ckptB = "";
  }
  return patch;
}

function debugOpenFromHistory(detail) {
  // detail: { run, path, record, ply } — kept signature for the untouched
  // [data-debug-open] delegation in History/Match. The deep link is consumed by
  // dbgApplyHash on the hashchange navigateScreen() triggers (or directly from
  // enterDebugScreen when no hashchange is coming).
  dbg.pendingDeepLink = detail;
  navigateScreen("debug");
}

// ---- apply: nav -> sources -> position -> panels ----------------------------

async function dbgApplyNav(nav, seq) {
  // Mirror the attention nav keys into the working state (read by the board heat,
  // the Attention panel, and the fetch trigger). Block/head clamps live in
  // dbgHashToNav; attnhead "" == mean-over-heads.
  dbg.attnQuery = dbgParseAttnQuery(nav.attnq);
  dbg.attnBlock = nav.attnblk || 0;
  dbg.attnHead = nav.attnhead === "" || nav.attnhead == null ? null : (nav.attnhead === "max" ? "max" : Number(nav.attnhead));
  dbgSyncControls();
  dbgRenderCrumb();
  if (!trainingRuns.length) {
    debugSetStatus("Loading runs…");
    try {
      await loadTrainingRuns();
    } catch (_e) { /* fall through with whatever we have */ }
    if (seq !== dbg.applySeq) return;
    debugSetStatus("");
  }
  const runNames = trainingRuns.map(r => r.name);
  if (!runNames.length) {
    debugSetStatus("No training runs found.", "error");
    return;
  }
  if (!nav.run || !runNames.includes(nav.run)) {
    // Prefer the run already selected on History, else the first run.
    const preferred = historySelectedRun && runNames.includes(historySelectedRun) ? historySelectedRun : runNames[0];
    dbgNavigate({ run: preferred, path: "", rec: 0, ply: null, acts: [], ckptA: "", ckptB: "" }, { replace: true });
    return;
  }
  if (dbg.loadedRun !== nav.run) {
    await dbgLoadCheckpoints(nav.run);
    if (seq !== dbg.applySeq) return;
    dbgSyncControls();
  }
  dbgLoadPins();
  dbgLoadJournal();
  if (dbg.checkpoints.length) {
    if (!nav.ckptA || !dbg.checkpoints.some(c => c.name === nav.ckptA)) {
      const latest = dbg.checkpoints.find(c => c.latest) || dbg.checkpoints[0];
      dbgNavigate({ ckptA: latest.name }, { replace: true });
      return;
    }
    if (nav.ckptB && !dbg.checkpoints.some(c => c.name === nav.ckptB)) {
      dbgNavigate({ ckptB: "" }, { replace: true });
      return;
    }
  }
  // Checkpoint provenance is position-independent — ensure it BEFORE the games/
  // position early-returns so CKPT loads even when the source has no games (M7).
  if (nav.tab === "ckpt") dbgEnsureCkptInfo(false);
  const gamesKey = `${nav.run}|${nav.src}`;
  if (dbg.loadedGamesKey !== gamesKey) {
    await dbgLoadGames(nav.run, nav.src);
    if (seq !== dbg.applySeq) return;
    dbgSyncControls();
  }
  if (!dbg.games.length) {
    dbgResetPosition();
    dbgRenderAll();
    debugSetStatus(`No ${nav.src} games in ${nav.run}.`, "error");
    return;
  }
  if (!nav.path || !dbg.games.some(g => g.path === nav.path)) {
    await dbgAutoPickGame(seq);
    return;
  }
  const recorded = dbgRecordedActs();
  if (nav.ply == null) {
    if (recorded) {
      dbgNavigate({ ply: recorded.length }, { replace: true });
      return;
    }
    await dbgResolveEndPly(seq);
    return;
  }
  if (recorded && nav.ply > recorded.length) {
    dbgNavigate({ ply: recorded.length }, { replace: true });
    return;
  }
  const key = dbgCurrentKey();
  const entry = dbg.cache.get(key);
  if (entry && entry.position) {
    // Client cache hit (M12): zero network for the board; panels refill from the
    // same entry, and only missing pieces are fetched.
    dbgCommitEntry(key, entry);
    if (nav.ckptB) {
      const keyB = dbgCacheKey(nav, nav.ckptB);
      const entryB = dbg.cache.get(keyB);
      if (entryB && entryB.analysis) {
        dbg.analysisB = entryB.analysis;
        dbg.keys.analysisB = keyB;
      }
    }
    debugSetStatus("");
    dbgRenderAll();
    if (!entry.analysis || (nav.ckptB && dbg.keys.analysisB !== dbgCacheKey(nav, nav.ckptB))) dbgScheduleFetch(0);
    // Fully cached: dbgFetchCurrent (whose tail prefetches) never runs, so extend
    // the S7 prefetch window here — else steady stepping prefetches alternate plies.
    else dbgMaybePrefetch();
  } else {
    if (!nav.acts.length && recorded && dbg.gamePlacements.length) {
      // INSTANT, analyze-independent step off the client-side placement cache;
      // the debounced fetch below fills in legal cells/tactics + analysis.
      dbgOptimisticStep();
    } else {
      debugSetStatus("Loading position…", "busy");
    }
    dbgRenderAll();
    dbgScheduleFetch(120);  // coalesce rapid steps/slider drags into one fetch
  }
  if (nav.tab === "inputs" || nav.mode === "plane") dbgEnsureInputs(false);
  if (nav.tab === "attn" || nav.mode === "attn") dbgEnsureAttn(false);
  if (dbgWantRecordRow(nav)) dbgEnsureRecordRow(false);
}

// ---- source loading ---------------------------------------------------------

async function dbgLoadCheckpoints(run) {
  try {
    const data = await debugFetchJson(`/api/debug/checkpoints?run=${encodeURIComponent(run)}`);
    dbg.checkpoints = data.checkpoints || [];
    dbg.worker = data.worker || null;
    dbg.lineage = data.lineage || "";
    // int|null detected support radius (shrimp only); seeds the Auto display.
    dbg.radiusDetected = (typeof data.support_radius_detected === "number") ? data.support_radius_detected : null;
    dbgRenderWorkerDot(dbg.worker && dbg.worker.alive ? "ok" : "");
  } catch (e) {
    dbg.checkpoints = [];
    dbg.worker = null;
    dbg.lineage = "";
    dbg.radiusDetected = null;
    dbgRenderWorkerDot("err");
    debugSetStatus(`Checkpoints: ${e.message}`, "error");
  }
  dbg.loadedRun = run;  // set even on error — the Refresh button clears it to retry
}

async function dbgLoadGames(run, src) {
  try {
    const data = await debugFetchJson(`/api/debug/games?run=${encodeURIComponent(run)}&source=${encodeURIComponent(src)}`);
    dbg.games = data.games || [];
  } catch (e) {
    dbg.games = [];
    debugSetStatus(`Games: ${e.message}`, "error");
  }
  dbg.loadedGamesKey = `${run}|${src}`;
}

async function dbgAutoPickGame(seq) {
  // Auto-pick the newest game that actually loads. The newest selfplay file is
  // usually the IN-PROGRESS epoch (still being written by the live run), which
  // has NO complete games yet — probe newest-first and stop at the first game
  // that loads, so the tab always opens on a usable game.
  debugSetStatus("Loading game…", "busy");
  for (const g of dbg.games) {
    try {
      const params = new URLSearchParams({ run: dbg.nav.run, path: g.path, record: "0", ply: "999999" });
      dbgRadiusParam(params);
      const data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
      if (seq !== dbg.applySeq) return;
      const total = data.debug.total;
      const entry = dbgCacheEntry(dbgCacheKey(Object.assign({}, dbg.nav, { path: g.path, rec: 0, ply: total, acts: [] }), dbg.nav.ckptA));
      entry.position = data;
      debugSetStatus("");
      dbgNavigate({ path: g.path, rec: 0, ply: total, acts: [] }, { replace: true });
      return;
    } catch (_e) { /* probe the next (older) file */ }
  }
  if (seq !== dbg.applySeq) return;
  dbgResetPosition();
  dbgRenderAll();
  debugSetStatus(`No loadable ${dbg.nav.src} games in ${dbg.nav.run}.`, "error");
}

async function dbgResolveEndPly(seq) {
  // ply=null means "final position"; ask the server (it clamps) then pin the
  // real total into the hash so the cache key and the URL stay canonical.
  const nav = dbg.nav;
  debugSetStatus("Loading position…", "busy");
  try {
    const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: "999999" });
    dbgRadiusParam(params);
    const data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
    if (seq !== dbg.applySeq) return;
    const total = data.debug.total;
    const entry = dbgCacheEntry(dbgCacheKey(Object.assign({}, nav, { ply: total, acts: [] }), nav.ckptA));
    entry.position = data;
    debugSetStatus("");
    dbgNavigate({ ply: total }, { replace: true });
  } catch (e) {
    if (seq === dbg.applySeq) debugSetStatus(`Position: ${e.message}`, "error");
  }
}

function dbgRefreshSources() {
  dbg.loadedRun = "";
  dbg.loadedGamesKey = "";
  dbg.navApplied = "";
  dbgApplyHash();
}

// ---- cache + keys (spec M12) ------------------------------------------------

function dbgCacheKey(nav, ckpt) {
  // Include the support-radius override so R=4 and R=8 results never collide in
  // the M12 client cache (a 4<->8 toggle must refetch, not serve the wrong set).
  return [nav.run, nav.path, nav.rec, nav.ply, ckpt || "", (nav.acts || []).join(","), nav.radius || 0].join("|");
}

function dbgCurrentKey() {
  return dbgCacheKey(dbg.nav, dbg.nav.ckptA);
}

function dbgCacheEntry(key) {
  let entry = dbg.cache.get(key);
  if (entry) {
    dbg.cache.delete(key);  // LRU bump to most-recently-used
    dbg.cache.set(key, entry);
    return entry;
  }
  entry = {};
  dbg.cache.set(key, entry);
  while (dbg.cache.size > DBG_CACHE_MAX) dbg.cache.delete(dbg.cache.keys().next().value);
  return entry;
}

function dbgRecordRow() {
  // Recorded .npz training row for the current key — fetched in stage F3; the
  // value-dist soft-z marker / target heat read it opportunistically when set.
  const entry = dbg.cache.get(dbgCurrentKey());
  return (entry && entry.record_row) || null;
}

function dbgRecordedActs() {
  const nav = dbg.nav;
  return dbg.gameActsKey === `${nav.run}|${nav.path}|${nav.rec}` && dbg.gameActs.length ? dbg.gameActs : null;
}

function dbgRecordedTotal() {
  // Total plies of the RECORDED game. While branched, dbg.position.debug.total
  // is the branch prefix length (base ply + injected tail), not the game length,
  // so the rail/crumb/clamps must read the recorded action list instead.
  const recorded = dbgRecordedActs();
  if (recorded) return recorded.length;
  if (dbg.position && !dbg.nav.acts.length) return dbg.position.debug.total;
  return null;
}

function dbgFreshData(kind) {
  // Data gated to the CURRENT nav key: the board heat/HUD/Q columns must never
  // paint a previous position's outputs. Panels instead keep their stale data
  // visible with a stale dot until the refill lands (M14).
  // Attention is keyed by the full slot (position + query + block + head) so a
  // stale token/cell/block/head row can never be served (see dbgAttnSlotKey).
  if (kind === "attn") return dbg.keys.attn === dbgAttnSlotKey() ? dbg.attn : null;
  return dbg.keys[kind] === dbgCurrentKey() ? dbg[kind] : null;
}

// Attention cache slot = base position key + query tag + block + head, so
// switching token/cell/block/head/position each maps to a distinct slot. The
// active token switch reuses a slot (all 8 token rows ride in one payload), so
// the qtag for token queries is fixed to "token" — only cell queries vary by id.
function dbgAttnQtag() {
  const q = dbg.attnQuery;
  if (!q) return "none";
  return q.type === "cell" ? `cell:${q.id}` : "token";
}

function dbgAttnSlotKey() {
  const head = dbg.attnHead == null ? "mean" : String(dbg.attnHead);
  return `${dbgCurrentKey()}::attn::${dbgAttnQtag()}::${dbg.attnBlock}::${head}`;
}

function dbgResetPosition() {
  dbg.position = null;
  dbg.analysis = null;
  dbg.analysisB = null;
  dbg.search = null;
  dbg.tree = null;
  dbg.attn = null;
  dbg.records = [];
  dbg.gameActs = [];
  dbg.gameActsKey = "";
  dbg.gamePlacements = [];
  dbg.gameKey = "";
  dbg.keys = { position: "", analysis: "", analysisB: "", search: "", tree: "", attn: "" };
}

function dbgCommitEntry(key, entry) {
  dbgCommitPosition(entry.position, key);
  if (entry.analysis) {
    dbg.analysis = entry.analysis;
    dbg.keys.analysis = key;
  }
  dbg.search = entry.search || null;
  if (entry.search) dbg.keys.search = key;
  dbg.tree = entry.tree || null;
  if (entry.tree) dbg.keys.tree = key;
}

function dbgCommitPosition(data, key) {
  dbg.position = data;
  dbg.keys.position = key;
  const d = data.debug || {};
  if (Array.isArray(data.record_games) && data.record_games.length) dbg.records = data.record_games;
  if (!d.imported && Array.isArray(d.action_ids)) {
    // Harvest the full recorded action list — instant stepping, branch prefixes
    // and the recorded-move highlights all read it client-side.
    dbg.gameActs = d.action_ids;
    dbg.gameActsKey = `${dbg.nav.run}|${dbg.nav.path}|${dbg.nav.rec}`;
  }
  if (!dbg.nav.acts.length) {
    dbgCacheGamePlacements(data);
    // Position payloads carry placements for action_ids[:ply] only, so a deep
    // link to a mid-game ply leaves later stone ownership unknown — and the
    // HEADS recorded-reply highlight needs it. One-shot final-ply backfill.
    if (!d.imported && d.total != null && dbg.gamePlacements.length < d.total) dbgBackfillPlacements(d.total);
  }
}

function dbgBackfillPlacements(total) {
  const nav = dbg.nav;
  const gameKey = `${nav.run}|${nav.path}|${nav.rec}`;
  if (dbg.placementsBackfill === gameKey) return;
  dbg.placementsBackfill = gameKey;
  const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: String(total) });
  dbgRadiusParam(params);
  debugFetchJson(`/api/debug/position?${params.toString()}`).then(data => {
    if (gameKey !== `${dbg.nav.run}|${dbg.nav.path}|${dbg.nav.rec}`) return;  // game changed mid-flight
    // Warm the M12 cache for the final-ply key while the payload is in hand.
    const entry = dbgCacheEntry(dbgCacheKey(Object.assign({}, dbg.nav, { ply: total, acts: [] }), dbg.nav.ckptA));
    if (!entry.position) entry.position = data;
    dbgCacheGamePlacements(data);
    if (dbg.nav.tab === "heads") dbgRenderHeads();
  }).catch(() => {
    if (dbg.placementsBackfill === gameKey) dbg.placementsBackfill = "";  // transient — retry on next commit
  });
}

function dbgCacheGamePlacements(data) {
  // Merge a loaded position's stones into the per-game cache (keyed by index), so
  // the full game's board is known client-side regardless of which ply loaded.
  // Branch (acts) positions are NOT merged — their indexes shift with the base ply.
  const key = `${dbg.nav.run}|${dbg.nav.path}|${dbg.nav.rec}`;
  if (key !== dbg.gameKey) {
    dbg.gameKey = key;
    dbg.gamePlacements = [];
  }
  const byIndex = new Map(dbg.gamePlacements.map(p => [p.index, p]));
  for (const p of (data.placements || [])) if (p && p.index != null) byIndex.set(p.index, p);
  dbg.gamePlacements = [...byIndex.values()].sort((a, b) => a.index - b.index);
}

function debugCurrentPlacements() {
  // Stones to show for the current ply. Prefer the cached full game filtered to
  // `index <= ply` (instant, no round-trip); branch positions use the payload.
  const pos = dbg.position;
  if (!pos) return [];
  if (dbg.nav.acts.length) return pos.placements || [];
  const ply = pos.debug.ply;
  if (dbg.gamePlacements.length) return dbg.gamePlacements.filter(p => p.index <= ply);
  return pos.placements || [];
}

function dbgOptimisticStep() {
  // Clone (never mutate the cached payload) with the new ply; the real position
  // payload + analysis land via the debounced fetch.
  const nav = dbg.nav;
  const pos = dbg.position;
  const acts = dbgRecordedActs() || (pos.debug && pos.debug.action_ids) || [];
  const ply = Math.max(0, Math.min(nav.ply, acts.length));
  const last = ply > 0 ? dbgUnpackActionId(acts[ply - 1]) : null;
  dbg.position = Object.assign({}, pos, {
    debug: Object.assign({}, pos.debug, {
      ply,
      last_action_id: ply > 0 ? acts[ply - 1] : null,
      last_q: last ? last.q : null,
      last_r: last ? last.r : null,
    }),
  });
  dbg.keys.position = "";  // optimistic — the real payload is still loading
  diagTap("ply " + ply + "/" + acts.length);  // visible step feedback on-device
}

// ---- fetch plumbing ----------------------------------------------------------

let dbgFetchTimer = null;

function dbgScheduleFetch(delay) {
  window.clearTimeout(dbgFetchTimer);
  dbgFetchTimer = window.setTimeout(() => {
    dbgFetchCurrent().catch(e => reportError("dbgFetchCurrent: " + (e && (e.stack || e.message) || e)));
  }, delay);
}

async function dbgFetchCurrent() {
  const nav = dbg.nav;
  if (!nav.run || !nav.path || nav.ply == null) return;
  const key = dbgCurrentKey();
  const entry = dbgCacheEntry(key);
  if (!entry.position && nav.ckptB) {
    // The sibling (ckptB) entry holds the same checkpoint-independent position —
    // after a ckptA↔ckptB flip reuse it instead of refetching (M12).
    const sib = dbg.cache.get(dbgCacheKey(nav, nav.ckptB));
    if (sib && sib.position) {
      entry.position = sib.position;
      if (!entry.record_row && sib.record_row) entry.record_row = sib.record_row;
    }
  }
  if (!entry.position) {
    const seq = ++dbg.posSeq;  // claim latest; a newer ply nav supersedes this fetch
    try {
      let data;
      if (nav.acts.length) {
        // What-if branch (M4): full prefix = recorded actions[0..ply] + injected
        // tail, reconstructed by the SERVER via ?actions= (true engine coords +
        // legal cells + tactics, no client action-id unpacking for the board).
        const recorded = await dbgEnsureRecordedActs();
        if (seq !== dbg.posSeq) return;
        const prefix = recorded.slice(0, nav.ply).concat(nav.acts);
        const params = new URLSearchParams({ run: nav.run, actions: prefix.join(","), ply: String(prefix.length) });
        dbgRadiusParam(params);
        data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
      } else {
        const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: String(nav.ply) });
        dbgRadiusParam(params);
        data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
      }
      if (seq !== dbg.posSeq) return;  // superseded by a newer nav — drop stale fetch
      entry.position = data;
    } catch (e) {
      if (seq === dbg.posSeq) debugSetStatus(`Position: ${e.message}`, "error");
      return;
    }
  }
  if (key !== dbgCurrentKey()) return;  // nav moved on while we fetched
  dbgCommitEntry(key, entry);
  debugSetStatus("");
  dbgRenderAll();
  await dbgEnsureAnalysis(key, entry);
  if (dbg.nav.ckptB) await dbgEnsureAnalysisB();
  if (dbgWantRecordRow(dbg.nav)) dbgEnsureRecordRow(false);
  dbgMaybePrefetch();  // S7: speculative ply±1 once the user's own requests settled
}

async function dbgEnsureRecordedActs() {
  const nav = dbg.nav;
  const recorded = dbgRecordedActs();
  if (recorded) return recorded;
  // Deep link straight into a branch: one recorded-position fetch supplies the
  // full action-id list (and lands in the cache for the plain-ply key).
  const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: String(Math.max(0, nav.ply || 0)) });
  dbgRadiusParam(params);
  const data = await debugFetchJson(`/api/debug/position?${params.toString()}`);
  const d = data.debug || {};
  dbg.gameActs = d.action_ids || [];
  dbg.gameActsKey = `${nav.run}|${nav.path}|${nav.rec}`;
  if (Array.isArray(data.record_games) && data.record_games.length) dbg.records = data.record_games;
  const entry = dbgCacheEntry(dbgCacheKey(Object.assign({}, nav, { ply: d.ply, acts: [] }), nav.ckptA));
  if (!entry.position) entry.position = data;
  return dbg.gameActs;
}

async function dbgRequestBody(checkpoint) {
  const nav = dbg.nav;
  const body = { run: nav.run, checkpoint };
  // Manual support-radius override (shrimp only); absent => backend detects/
  // defaults. Covers analyze/search/search_tree/attention bodies built here.
  if (nav.radius) body.radius = nav.radius;
  if (nav.acts.length) {
    // Branch prefix = recorded actions[0..ply] + injected tail. The recorded
    // list may be missing (gameActsKey points at another game after the user
    // browsed elsewhere and the branch came back via a cache hit) — NEVER fall
    // back to an empty prefix: the server would silently analyze a near-empty
    // board while the UI shows the full branch position. Re-fetch instead.
    const recorded = await dbgEnsureRecordedActs();
    body.action_ids = recorded.slice(0, nav.ply).concat(nav.acts);
  } else {
    body.path = nav.path;
    body.record = nav.rec;
    body.ply = nav.ply;
  }
  return body;
}

// Append the manual support-radius override (4|8) to a URLSearchParams for the
// GET endpoints (game_eval / trajectory / ckpt_info / prefetch position+analyze).
// No-op on Auto (0) so the backend detection/default path stays the source of
// truth and non-shrimp runs (radius never set) are byte-identical to today.
function dbgRadiusParam(params) {
  if (dbg.nav.radius) params.set("radius", String(dbg.nav.radius));
  return params;
}

// Detected support radius for the current run when on Auto: prefer the checkpoints
// payload's support_radius_detected (backend-detected from the latest eval), else
// the in-memory eval rows' latest ratings.fit.featurize_radius, else null. Display
// seed only — it never writes nav.radius (Auto must exercise the backend default).
function dbgDetectedRadius() {
  if (typeof dbg.radiusDetected === "number") return dbg.radiusDetected;
  const run = (trainingRuns || []).find(r => r && r.name === dbg.nav.run);
  const rows = run ? msHistoryRows(run) : [];
  for (let i = rows.length - 1; i >= 0; i--) {
    const fit = rows[i] && rows[i].ratings && rows[i].ratings.fit;
    const fr = fit && fit.featurize_radius;
    if (typeof fr === "number") return fr;
  }
  return null;
}

// The support radius to DISPLAY. Truth order: the analyze meta (what the worker
// actually ran at) > the manual override > detection > 8. Lineage-gated: returns
// null for KNOWN non-shrimp runs (no support radius applies) so the UI hides it.
function dbgEffectiveRadius() {
  const lin = dbgAttnLineage();  // "" unknown | shrimp | dense_cnn | hexgt
  if (lin && lin !== "shrimp") return null;
  const a = dbgFreshData("analysis");
  if (a && a.meta && typeof a.meta.support_radius === "number") return a.meta.support_radius;
  if (dbg.nav.radius) return dbg.nav.radius;
  const det = dbgDetectedRadius();
  if (typeof det === "number") return det;
  return 8;
}

// ckpt_info cache key: include the radius override so a 4<->8 toggle refetches
// (its meta.support_radius is radius-dependent). Lineage / has_cell_q are NOT
// radius-dependent, so their readers fall back across radii (see dbgCkptInfoAny).
function dbgCkptInfoKey(name) {
  return `${dbg.nav.run}|${name}|${dbg.nav.radius || 0}`;
}

// Radius-agnostic ckpt_info lookup for the radius-INDEPENDENT fields (lineage,
// has_cell_q): try the active-radius slot first, else any slot for run|name.
function dbgCkptInfoAny(name) {
  const exact = dbg.ckptInfo.get(dbgCkptInfoKey(name));
  if (exact) return exact;
  const pfx = `${dbg.nav.run}|${name}|`;
  for (const [k, v] of dbg.ckptInfo) if (k.startsWith(pfx)) return v;
  return null;
}

async function dbgEnsureAnalysis(key, entry) {
  const nav = dbg.nav;
  if (!nav.ckptA) return;
  if (entry.analysis) {
    if (key === dbgCurrentKey()) {
      dbg.analysis = entry.analysis;
      dbg.keys.analysis = key;
      dbgRenderAll();
    }
    return;
  }
  if (entry.analysisPending) return;  // one in-flight analyze per cache entry
  entry.analysisPending = true;
  const seq = ++dbg.anlSeq;  // claim latest; a slow dense analyze must not clobber a newer ply's
  dbg.loading = true;
  debugSetStatus("Evaluating position on CPU…", "busy");
  try {
    const analysis = await debugFetchJson("/api/debug/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(await dbgRequestBody(nav.ckptA)),
    });
    // Always cache the response — a superseded seq only skips the status/loading
    // bookkeeping, never the data, or a revisited ply could sit permanently
    // un-analyzed (analysisPending is false again and nothing re-triggers).
    entry.analysis = analysis;
    dbgJournalLog(analysis, nav.ckptA, nav);  // S3: auto-log every completed analyze
    if (!dbg.worker || !dbg.worker.alive) {
      // A successful analyze proves the lazily-spawned worker is up — re-read the
      // live status so the dot turns green and the S7 prefetch gate opens.
      debugFetchJson(`/api/debug/checkpoints?run=${encodeURIComponent(nav.run)}`).then(d => {
        dbg.worker = d.worker || null;
        dbgRenderWorkerDot(dbg.worker && dbg.worker.alive ? "ok" : "");
      }).catch(() => {});
    }
    if (key === dbgCurrentKey()) {
      dbg.analysis = analysis;
      dbg.keys.analysis = key;
      if (seq === dbg.anlSeq) debugSetStatus("");
      dbgRenderAll();  // the finally's render is seq-gated — cover the dropped-seq case
    }
  } catch (e) {
    if (seq === dbg.anlSeq) debugSetStatus(`Analyze: ${e.message}`, "error");
  } finally {
    entry.analysisPending = false;
    if (seq === dbg.anlSeq) {
      dbg.loading = false;
      dbgRenderAll();
    }
  }
}

async function dbgEnsureAnalysisB() {
  const nav = dbg.nav;
  if (!nav.ckptB || nav.ply == null) return;
  const keyB = dbgCacheKey(nav, nav.ckptB);
  const entryB = dbgCacheEntry(keyB);
  // position/record_row are checkpoint-independent — share them with the A entry
  // so a later ckptA↔ckptB flip re-renders with zero network requests (M12).
  const entryA = dbg.cache.get(dbgCacheKey(nav, nav.ckptA));
  if (entryA) {
    if (!entryB.position && entryA.position) entryB.position = entryA.position;
    if (!entryB.record_row && entryA.record_row) entryB.record_row = entryA.record_row;
  }
  if (entryB.analysis) {
    dbg.analysisB = entryB.analysis;
    dbg.keys.analysisB = keyB;
    dbgRenderComparePanel();
    dbgUpdateStaleDots();
    return;
  }
  if (entryB.analysisPending) return;
  entryB.analysisPending = true;
  delete entryB.analysisError;  // a retry starts clean
  const seq = ++dbg.anlSeqB;
  debugSetStatus("Evaluating checkpoint B…", "busy");
  try {
    const analysis = await debugFetchJson("/api/debug/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(await dbgRequestBody(nav.ckptB)),
    });
    // Cache even when superseded (see dbgEnsureAnalysis) — only the commit is gated.
    entryB.analysis = analysis;
    dbgJournalLog(analysis, nav.ckptB, nav);
    if (keyB === dbgCacheKey(dbg.nav, dbg.nav.ckptB)) {
      dbg.analysisB = analysis;
      dbg.keys.analysisB = keyB;
      if (seq === dbg.anlSeqB) debugSetStatus("");
    }
  } catch (e) {
    // Record the failure on the entry so the COMPARE panel can show an error
    // state instead of a perpetual "Evaluating checkpoint B…" (nothing retries
    // automatically — only ↻ or the next nav apply).
    entryB.analysisError = e.message;
    if (seq === dbg.anlSeqB) debugSetStatus(`Compare: ${e.message}`, "error");
  } finally {
    entryB.analysisPending = false;
    dbgRenderComparePanel();
    dbgUpdateStaleDots();
  }
}

function dbgAnalyzeNow(force) {
  const nav = dbg.nav;
  if (!nav.path || nav.ply == null) {
    debugSetStatus("No position loaded — pick a game with completed records.", "error");
    return;
  }
  if (force) delete dbgCacheEntry(dbgCurrentKey()).analysis;
  dbgScheduleFetch(0);
}

async function dbgRunSearch() {
  dbgAbortPrefetch();  // S7: an explicit request preempts the speculative ply±1 fetch
  const nav = dbg.nav;
  if (!nav.ckptA || !dbg.position) {
    debugSetStatus("Pick a checkpoint and position before searching.", "error");
    return;
  }
  if (dbg.searchBusy) {
    dbgStub("A search is already running.");
    return;
  }
  const visitsEl = dbgEl("dbgSearchVisits");
  const cpuctEl = dbgEl("dbgSearchCpuct");
  const seedEl = dbgEl("dbgSearchSeed");
  const visits = Math.max(1, Math.min(20000, parseInt(visitsEl && visitsEl.value, 10) || 512));
  const key = dbgCurrentKey();
  const entry = dbgCacheEntry(key);
  dbg.searchBusy = true;
  // #dbgSearchRun is static HTML (not re-rendered from state) — toggle in place.
  const runBtn = dbgEl("dbgSearchRun");
  if (runBtn) runBtn.disabled = true;
  debugSetStatus(`Running ${visits}-visit CPU search…`, "busy");
  try {
    const body = await dbgRequestBody(nav.ckptA);
    body.visits = visits;
    const cPuct = Number(cpuctEl && cpuctEl.value);
    body.c_puct = Number.isFinite(cPuct) && cPuct > 0 ? cPuct : 1.5;
    const seed = parseInt(seedEl && seedEl.value, 10);
    body.seed = Number.isFinite(seed) ? seed : 0;  // B2/M6: the seed is forwarded server-side now
    const result = await debugFetchJson("/api/debug/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    entry.search = result;
    if (key === dbgCurrentKey()) {
      dbg.search = result;
      dbg.keys.search = key;
      debugSetStatus("");
    }
  } catch (e) {
    debugSetStatus(`Search: ${e.message}`, "error");
  } finally {
    dbg.searchBusy = false;
    if (runBtn) runBtn.disabled = false;
    dbgRenderAll();
  }
}

async function dbgEnsureCkptInfo(force) {
  // B3/M7: checkpoint provenance via the worker `info` op — one round-trip,
  // readable WITHOUT paying an analyze.
  const nav = dbg.nav;
  for (const name of [nav.ckptA, nav.ckptB].filter(Boolean)) {
    const key = dbgCkptInfoKey(name);
    if (force) dbg.ckptInfo.delete(key);
    if (dbg.ckptInfo.has(key)) continue;
    dbgRenderCkptPanel();  // show "loading…" while the worker round-trips
    try {
      const params = new URLSearchParams({ run: nav.run, checkpoint: name });
      dbgRadiusParam(params);
      dbg.ckptInfo.set(key, await debugFetchJson(`/api/debug/ckpt_info?${params.toString()}`));
    } catch (e) {
      dbg.ckptInfo.set(key, { error: e.message });
    }
  }
  dbgRenderCkptPanel();
}

async function dbgPlotTrajectory() {
  // Legacy whole-game re-eval plot (kept as the no-sweep fallback per §3.4).
  const nav = dbg.nav;
  if (!nav.run || !nav.path || !nav.ckptA || nav.acts.length) {
    debugSetStatus("Trajectory needs a recorded game + checkpoint (return to the game first).", "error");
    return;
  }
  debugSetStatus("Re-evaluating the whole game on CPU…", "busy");
  try {
    const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), checkpoint: nav.ckptA });
    dbgRadiusParam(params);
    dbg.trajectory = await debugFetchJson(`/api/debug/trajectory?${params.toString()}`);
    dbg.trajectoryKey = `${nav.run}|${nav.path}|${nav.rec}|${nav.ckptA}`;
    debugSetStatus("");
  } catch (e) {
    dbg.trajectory = null;
    debugSetStatus(`Trajectory: ${e.message}`, "error");
  }
  dbgRenderDockChart();
}

// ---- TARGETS: recorded .npz training row (spec M8) ---------------------------------

function dbgWantRecordRow(nav) {
  // The row is only fetched when something actually displays it: the TARGETS
  // tab, the target/mismatch board modes (the HEADS soft-z marker reads it
  // opportunistically once cached, per spec).
  return Boolean(nav.run && nav.path && nav.ply != null
    && (nav.tab === "targets" || nav.mode === "target" || nav.mode === "mismatch"));
}

async function dbgEnsureRecordRow(force) {
  const nav = dbg.nav;
  if (!nav.run || !nav.path || nav.ply == null) return;
  const key = dbgCurrentKey();
  const entry = dbgCacheEntry(key);
  if (force) delete entry.record_row;
  if (!entry.record_row && nav.ckptB) {
    // record_row is checkpoint-independent — reuse the sibling (ckptB) entry's
    // copy after a ckptA↔ckptB flip instead of refetching (M12).
    const sib = dbg.cache.get(dbgCacheKey(nav, nav.ckptB));
    if (sib && sib.record_row) entry.record_row = sib.record_row;
  }
  if (nav.acts.length || nav.src !== "selfplay") {
    // Branch positions / eval games never have training rows — synthesize the
    // graceful miss client-side instead of paying a server round-trip.
    if (!entry.record_row) entry.record_row = { found: false, reason: nav.acts.length ? "branched" : "not_selfplay" };
    dbgRenderTargets();
    return;
  }
  if (entry.record_row || entry.recordRowPending) {
    dbgRenderTargets();
    return;
  }
  entry.recordRowPending = true;
  dbgRenderTargets();
  try {
    const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: String(nav.ply) });
    entry.record_row = await debugFetchJson(`/api/debug/record_row?${params.toString()}`);
  } catch (e) {
    entry.record_row = { found: false, reason: e.message };
  } finally {
    entry.recordRowPending = false;
  }
  if (key === dbgCurrentKey()) {
    dbgRenderTargets();
    dbgRenderHeads();   // soft-z marker on the value distribution
    if (dbg.nav.mode === "target" || dbg.nav.mode === "mismatch") dbgRenderBoard();
  }
}

const DBG_ROW_MISS_REASONS = {
  not_selfplay: "eval games have no training rows",
  branched: "what-if branches have no training rows",
  no_shard: "no .npz training shard found for this game",
  no_row: "no training row at this ply (fast-PCR / not recorded)",
  row_mismatch: "row mismatch — not shown (turn/player disagree with the replay)",
};

function dbgRenderTargets() {
  const note = dbgEl("dbgTargetsNote");
  const body = document.querySelector("#dbgTabTargets .dbg-targets-body");
  if (!note || !body) return;
  const entry = dbg.cache.get(dbgCurrentKey());
  const rr = entry && entry.record_row;
  if (!rr) {
    note.textContent = entry && entry.recordRowPending ? "Loading training row…" : "No training row loaded";
    body.innerHTML = "";
    return;
  }
  if (!rr.found || !rr.row) {
    note.textContent = DBG_ROW_MISS_REASONS[rr.reason] || `no training row (${rr.reason || "unknown"})`;
    body.innerHTML = "";
    return;
  }
  const row = rr.row;
  const a = dbgFreshData("analysis");
  note.textContent = `${rr.npz || ""}${rr.npz ? " · " : ""}turn ${rr.turn_index} · P${row.current_player}${row.phase ? ` · ${row.phase}` : ""}`;
  const num = v => (typeof v === "number" ? v.toFixed(3) : "—");
  // The npz does not persist every field — the backend returns null for those,
  // and null must read as "not recorded", never as a plausible 0.000/yes/"".
  const NOT_REC = `<span class="dbg-muted">not recorded</span>`;
  const d = (recV, liveV) => (typeof recV === "number" && typeof liveV === "number")
    ? `${liveV - recV >= 0 ? "+" : ""}${(liveV - recV).toFixed(3)}` : "—";
  const cols = (label, rec, live, dlt) => `<div class="dbg-target-row"><span class="label">${label}</span><span>${rec}</span><span class="dbg-muted">${live}</span><span>${dlt}</span></div>`;
  const out = [`<div class="dbg-target-row dbg-move-head"><span>field</span><span>recorded</span><span>live ${escapeText(dbgCkptShort(dbg.nav.ckptA))}</span><span>Δ</span></div>`];
  out.push(cols("value target", num(row.value_target), num(a && a.value), d(row.value_target, a && a.value)));
  out.push(cols("reason", row.value_target_reason == null ? NOT_REC : escapeText(row.value_target_reason || "—"), "", ""));
  for (const h of Object.keys(row.stvalue || {}).sort((x, y) => Number(x) - Number(y))) {
    const sv = row.stvalue[h];
    const live = a && a.stvalue && a.stvalue[h] ? a.stvalue[h].scalar : null;
    out.push(cols(`STV+${h}${sv.mask ? "" : " (masked)"}`, sv.target == null ? NOT_REC : num(sv.target), num(live), sv.mask ? d(sv.target, live) : "—"));
  }
  if (row.moves_left) {
    const liveMl = a && a.moves_left && typeof a.moves_left.scalar === "number"
      ? (a.moves_left.scalar + 1) / 2 * ((a.meta && a.meta.moves_left_cap) || 512)
      : null;
    const recMl = row.moves_left.target;
    out.push(cols(`moves left${row.moves_left.mask ? "" : " (masked)"}`,
      typeof recMl === "number" && recMl >= 0 ? recMl.toFixed(0) : (recMl == null ? NOT_REC : "—"),
      liveMl != null ? liveMl.toFixed(0) : "—",
      row.moves_left.mask && typeof recMl === "number" && recMl >= 0 && liveMl != null
        ? `${liveMl - recMl >= 0 ? "+" : ""}${(liveMl - recMl).toFixed(0)}` : "—"));
  }
  out.push(cols("policy surprise", row.policy_surprise == null ? NOT_REC : num(row.policy_surprise), "", ""));
  // 0 is a sentinel: the shard stored a normalized visit policy, so the raw count is unknown.
  out.push(cols("search visits", row.search_visits == null ? NOT_REC
      : (row.search_visits > 0 ? String(row.search_visits) : "unknown (normalized weights)"), "", ""));
  out.push(cols("pcr_full", row.pcr_full == null ? NOT_REC : (row.pcr_full ? "yes" : "no (fast)"), "", ""));
  out.push(cols("frequency wt", row.frequency_weight == null ? NOT_REC : num(row.frequency_weight), "", ""));
  if (row.truncated) out.push(cols("truncated", "yes", "", ""));
  if (row.opp_policy_source) out.push(cols("opp source", escapeText(row.opp_policy_source), "", ""));
  const priorById = new Map(((a && a.policy) || []).map(p => [p.action_id, p.p]));
  const tm = (row.policy || []).slice(0, 8).map((p, i) => {
    const live = priorById.get(p.action_id);
    const dp = live != null ? `${live - p.p >= 0 ? "+" : ""}${((live - p.p) * 100).toFixed(1)}%` : "—";
    return `<div class="dbg-target-row"><span>#${i + 1} ${p.q},${p.r}</span><span>${(p.p * 100).toFixed(1)}%</span><span class="dbg-muted">${live != null ? (live * 100).toFixed(1) + "%" : "—"}</span><span>${dp}</span></div>`;
  }).join("");
  body.innerHTML = out.join("")
    + `<div class="dbg-subhead" style="margin-top:8px">Top target moves <span class="dbg-muted">(recorded · live prior · Δ)</span></div>` + tm;
}

// ---- Game Error Sweep (spec M9) -----------------------------------------------------

function dbgSweepKey(nav) {
  nav = nav || dbg.nav;
  return `${nav.run}|${nav.path}|${nav.rec}|${nav.ckptA}`;
}

function dbgSweepData() {
  const sweep = dbg.sweeps.get(dbgSweepKey());
  return sweep && sweep.plies.length ? sweep : null;
}

function dbgBlunders(sweep) {
  // Q-regret blunders (v3 cell_q) when the sweep carries per-ply regret: marks
  // p.ply — the move WHERE the blunder was committed (sharper than the legacy
  // value-swing marker, which flags the ply AFTER the drop). Falls back to the
  // value-swing rule for older checkpoints with no cell_q (regret all null).
  const out = [];
  const hasRegret = sweep.plies.some(p => p.regret != null);
  if (hasRegret) {
    if (dbg.regretMode === "relative") {
      // Top decile of THIS game's nonzero regrets, floored at 0.10 to suppress
      // clean-game noise.
      const rs = sweep.plies.map(p => p.regret).filter(v => v != null && v > 0).sort((a, b) => a - b);
      const dec = rs.length ? rs[Math.floor(rs.length * 0.9)] : Infinity;
      for (const p of sweep.plies) if (p.regret != null && p.regret >= Math.max(dec, 0.10)) out.push(p.ply);
    } else {
      const thr = dbg.regretThreshold != null ? dbg.regretThreshold : DBG_REGRET_BLUNDER;
      for (const p of sweep.plies) if (p.regret != null && p.regret >= thr) out.push(p.ply);
    }
    return out;
  }
  // Legacy value-swing fallback: value_p0 sign flip or |Δ value_p0| > 0.5 between
  // CONSECUTIVE swept plies; returns the ply AFTER the drop (where to look).
  for (let i = 1; i < sweep.plies.length; i++) {
    const a = sweep.plies[i - 1];
    const b = sweep.plies[i];
    if (b.ply !== a.ply + 1) continue;
    if ((a.value_p0 >= 0) !== (b.value_p0 >= 0) || Math.abs(b.value_p0 - a.value_p0) > 0.5) out.push(b.ply);
  }
  return out;
}

async function dbgRunSweep() {
  dbgAbortPrefetch();  // S7: an explicit request preempts the speculative ply±1 fetch
  const nav = dbg.nav;
  if (!nav.run || !nav.path || !nav.ckptA) {
    debugSetStatus("Sweep needs a recorded game + checkpoint.", "error");
    return;
  }
  const key = dbgSweepKey();
  if (dbg.sweepRun && dbg.sweepRun.key === key) {
    dbg.sweepRun.abort = true;  // the button toggles: click while running = stop
    return;
  }
  if (dbg.sweepRun) dbg.sweepRun.abort = true;  // a sweep of another tuple yields
  const runState = { key, abort: false };
  dbg.sweepRun = runState;
  let sweep = dbg.sweeps.get(key);
  if (!sweep) {
    sweep = { total: null, winner: null, plies: [], byPly: new Map(), version: 0 };
    dbg.sweeps.set(key, sweep);
  }
  const progress = dbgEl("dbgSweepProgress");
  const btns = [dbgEl("dbgSweepBtn"), dbgEl("dbgPlySweepBtn")].filter(Boolean);
  btns.forEach(b => b.classList.add("active"));
  try {
    // Resume cheaply (service LRU caches chunks; the client keeps finished plies).
    let start = 0;
    while (sweep.byPly.has(start)) start++;
    while (!runState.abort && (sweep.total == null || start < sweep.total)) {
      const params = new URLSearchParams({
        run: nav.run, path: nav.path, record: String(nav.rec), checkpoint: nav.ckptA,
        start: String(start), count: "16",
      });
      dbgRadiusParam(params);
      const data = await debugFetchJson(`/api/debug/game_eval?${params.toString()}`);
      sweep.total = data.total;
      sweep.winner = data.winner;
      for (const p of data.plies || []) if (!sweep.byPly.has(p.ply)) sweep.byPly.set(p.ply, p);
      sweep.plies = [...sweep.byPly.values()].sort((x, y) => x.ply - y.ply);
      sweep.version++;
      start = Math.max(start + 1, data.start + (data.plies || []).length);
      while (sweep.byPly.has(start)) start++;
      if (progress) progress.textContent = `${sweep.plies.length}/${sweep.total} plies`;
      if (key === dbgSweepKey()) {
        dbgRenderPlyRail();
        dbgRenderDockChart();
        dbgRenderRegretList();
      }
    }
    if (progress) {
      progress.textContent = runState.abort
        ? `stopped at ${sweep.plies.length}/${sweep.total != null ? sweep.total : "?"}`
        : `${sweep.plies.length}/${sweep.total} plies`;
    }
  } catch (e) {
    debugSetStatus(`Sweep: ${e.message}`, "error");
  } finally {
    if (dbg.sweepRun === runState) dbg.sweepRun = null;
    btns.forEach(b => b.classList.remove("active"));
    if (key === dbgSweepKey()) {
      dbgRenderPlyRail();
      dbgRenderDockChart();
      dbgRenderRegretList();
    }
  }
}

function dbgStepBlunder(dir) {
  const sweep = dbgSweepData();
  if (!sweep) {
    dbgStub("Run a game error sweep first (Sweep button).");
    return;
  }
  const blunders = dbgBlunders(sweep);
  if (!blunders.length) {
    dbgStub("No blunder plies found by the sweep.");
    return;
  }
  const cur = dbg.nav.ply != null ? dbg.nav.ply : 0;
  let next = null;
  if (dir > 0) next = blunders.find(p => p > cur);
  else for (const p of blunders) if (p < cur) next = p;
  if (next == null) {
    dbgStub(dir > 0 ? "No later blunder." : "No earlier blunder.");
    return;
  }
  dbgGotoPly(next);
}

// ---- MCTS Tree Explorer (spec M10/M11) ------------------------------------------

function dbgTreeParams() {
  const visitsEl = dbgEl("dbgSearchVisits");
  const cpuctEl = dbgEl("dbgSearchCpuct");
  const seedEl = dbgEl("dbgSearchSeed");
  const visits = Math.max(1, Math.min(20000, parseInt(visitsEl && visitsEl.value, 10) || 512));
  const cPuct = Number(cpuctEl && cpuctEl.value);
  const seed = parseInt(seedEl && seedEl.value, 10);
  return {
    visits,
    c_puct: Number.isFinite(cPuct) && cPuct > 0 ? cPuct : 1.5,
    seed: Number.isFinite(seed) ? seed : 0,
  };
}

async function dbgRunTree(opts) {
  // opts: {checkpoint} = run for that checkpoint's cache slot (compare B trees);
  // {rootActions, graftPath} = expand-on-demand — re-root the search at a deep
  // node via root_actions and graft the children back onto the rendered tree.
  dbgAbortPrefetch();  // S7: an explicit request preempts the speculative ply±1 fetch
  const nav = dbg.nav;
  const checkpoint = (opts && opts.checkpoint) || nav.ckptA;
  if (!checkpoint || !dbg.position) {
    debugSetStatus("Pick a checkpoint and position before searching.", "error");
    return;
  }
  if (dbg.treeBusy) {
    dbgStub("A debug tree is already running.");
    return;
  }
  const key = dbgCacheKey(nav, checkpoint);
  dbg.treeBusy = true;
  dbgRenderSearchPanel();
  try {
    const body = Object.assign(await dbgRequestBody(checkpoint), dbgTreeParams());
    if (opts && opts.rootActions && opts.rootActions.length) body.root_actions = opts.rootActions;
    debugSetStatus(`Debug tree: ${body.visits} visits on CPU…`, "busy");
    const result = await debugFetchJson("/api/debug/search_tree", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    debugSetStatus("");
    if (opts && opts.graftPath) {
      const host = dbgFreshData("tree");
      const node = host && dbgTreeNodeByPath(host.tree, opts.graftPath);
      if (node) {
        node.children = (result.tree && result.tree.children) || [];
        node.pruned_children = result.tree ? result.tree.pruned_children : 0;
        node.grafted = true;  // re-rooted deeper search — N not comparable to siblings
        dbg.treeOpen.set(opts.graftPath, true);
      }
    } else {
      const entry = dbgCacheEntry(key);
      entry.tree = result;
      if (key === dbgCurrentKey()) {
        dbg.tree = result;
        dbg.keys.tree = key;
        dbg.treeOpen = new Map();
        dbg.treePreview = null;
        dbg.treeGhostsOff = false;  // a fresh run always shows its ghosts
      }
    }
  } catch (e) {
    debugSetStatus(`Debug tree: ${e.message}`, "error");
  } finally {
    dbg.treeBusy = false;
  }
  dbgRenderAll();
}

function dbgTreeNodeByPath(root, pathStr) {
  let node = root;
  for (const part of String(pathStr).split("/")) {
    if (!part) continue;
    node = (node.children || []).find(ch => String(ch.action_id) === part) || null;
    if (!node) return null;
  }
  return node;
}

function dbgTreePathIds(pathStr) {
  return String(pathStr).split("/").map(s => parseInt(s, 10)).filter(n => Number.isFinite(n));
}

function dbgRenderTree() {
  const el = dbgEl("dbgTree");
  if (!el) return;
  const tree = dbgFreshData("tree");
  if (!tree || !tree.tree) {
    el.innerHTML = `<div class="dbg-empty-note">${dbg.treeBusy ? "Debug tree running…" : "Debug tree (Python; may not match engine tie-breaking)"}</div>`;
    return;
  }
  const pvSet = new Set();
  let acc = "";
  for (const id of tree.pv || []) {
    acc = acc ? `${acc}/${id}` : String(id);
    pvSet.add(acc);
  }
  const render = (node, pathStr, maxNSib) => {
    const kids = node.children || [];
    const isPv = pvSet.has(pathStr);
    const open = dbg.treeOpen.has(pathStr) ? dbg.treeOpen.get(pathStr) : isPv;
    const caret = kids.length ? (open ? "▾" : "▸") : "·";
    const expand = node.pruned_children > 0
      ? `<button type="button" class="dbg-tree-expand" data-tree-path="${pathStr}" title="Load pruned children (re-searches this subtree as a new root)">+${node.pruned_children}</button>`
      : "";
    let html = `<div class="dbg-tree-row${isPv ? " pv" : ""}" data-tree-path="${pathStr}" title="N ${node.n} · Q stm ${node.qm.toFixed(3)} · Q p0 ${node.qm_p0.toFixed(3)} · P ${(node.p * 100).toFixed(2)}%${node.u != null ? ` · U ${node.u.toFixed(3)}` : ""}${node.v != null ? ` · v@expand ${node.v.toFixed(3)}` : ""}${node.grafted ? " · grafted (re-rooted search)" : ""}">`
      + `<span class="dbg-tree-caret">${caret}</span>`
      + `<span class="dbg-tree-cell">${node.q},${node.r}</span>`
      + `<span class="dbg-tree-bar"><span style="width:${Math.max(3, 100 * node.n / Math.max(1, maxNSib)).toFixed(0)}%"></span></span>`
      + `<span class="dbg-tree-n">${node.n}</span>`
      + `<span class="dbg-tree-q">${node.qm.toFixed(2)}/${node.qm_p0.toFixed(2)}</span>`
      + `<span class="dbg-tree-p">${(node.p * 100).toFixed(1)}%</span>`
      + `<span class="dbg-tree-u">${node.u != null ? node.u.toFixed(2) : "—"}</span>`
      + `<span class="dbg-tree-v">${node.v != null ? node.v.toFixed(2) : "—"}</span>`
      + `<button type="button" class="dbg-tree-step" data-tree-path="${pathStr}" title="Step into this line (re-bases via what-if injection)">⤓</button>`
      + expand
      + `</div>`;
    if (open && kids.length) {
      const m = Math.max(1, ...kids.map(k => k.n));
      html += `<div class="dbg-tree-node">${kids.map(k => render(k, `${pathStr}/${k.action_id}`, m)).join("")}</div>`;
    }
    return html;
  };
  const roots = tree.tree.children || [];
  const maxN = Math.max(1, ...roots.map(ch => ch.n));
  const pvCells = (tree.pv || []).map(id => {
    const c = dbgUnpackActionId(id);
    return `${c.q},${c.r}`;
  });
  const head = `<div class="dbg-muted" title="debug search (Python; may not match engine tie-breaking)">`
    + `${escapeText(tree.engine || "py_debug")} · ${tree.visits} visits · root ${tree.root_value.toFixed(3)} · ${tree.node_count} nodes${tree.truncated ? " · truncated" : ""}</div>`
    + `<div class="dbg-muted">PV: ${pvCells.join(" → ") || "—"}</div>`
    + `<div class="dbg-tree-cols dbg-move-head"><span></span><span>cell</span><span>N</span><span></span><span>Q stm/p0</span><span>P</span><span>U</span><span>v</span></div>`;
  el.innerHTML = head + roots.map(ch => render(ch, String(ch.action_id), maxN)).join("");
}

function dbgTreeRowClick(pathStr) {
  // Node click = expand/collapse + preview its line on the board (M10).
  const tree = dbgFreshData("tree");
  const node = tree && dbgTreeNodeByPath(tree.tree, pathStr);
  if (!node) return;
  if ((node.children || []).length) {
    const pvSet = new Set();
    let acc = "";
    for (const id of tree.pv || []) {
      acc = acc ? `${acc}/${id}` : String(id);
      pvSet.add(acc);
    }
    const open = dbg.treeOpen.has(pathStr) ? dbg.treeOpen.get(pathStr) : pvSet.has(pathStr);
    dbg.treeOpen.set(pathStr, !open);
  }
  dbg.treePreview = { key: dbgCurrentKey(), ids: dbgTreePathIds(pathStr) };
  dbgRenderTree();
  dbgRenderBoard();
}

function dbgRenderScatter() {
  const el = dbgEl("dbgScatter");
  if (!el) return;
  const tree = dbgFreshData("tree");
  const kids = tree && tree.tree ? (tree.tree.children || []) : [];
  if (!kids.length) {
    el.innerHTML = `<div class="dbg-empty-note">Q vs prior scatter after a tree run</div>`;
    return;
  }
  const W = 340, H = 190, padL = 34, padR = 10, padT = 12, padB = 20;
  const ps = kids.map(k => Math.max(k.p, 1e-4));
  const lo = Math.log(Math.min(...ps));
  const hi = Math.log(Math.max(Math.max(...ps), Math.min(...ps) * 1.0001));
  const xOf = p => padL + (W - padL - padR) * ((Math.log(Math.max(p, 1e-4)) - lo) / Math.max(1e-9, hi - lo));
  const yOf = q => padT + (H - padT - padB) * (1 - (q + 1) / 2);
  const maxN = Math.max(1, ...kids.map(k => k.n));
  const rootQ = tree.tree.qm;
  const recorded = (!dbg.nav.acts.length && dbg.gameActs.length > dbg.nav.ply) ? dbg.gameActs[dbg.nav.ply] : null;
  const midX = padL + (W - padL - padR) / 2;
  let html = `<line x1="${padL}" y1="${yOf(rootQ).toFixed(1)}" x2="${W - padR}" y2="${yOf(rootQ).toFixed(1)}" stroke="#2c3d50" stroke-dasharray="3 3"></line>`
    + `<line x1="${midX.toFixed(1)}" y1="${padT}" x2="${midX.toFixed(1)}" y2="${H - padB}" stroke="#2c3d50" stroke-dasharray="3 3"></line>`
    + `<text x="${padL + 2}" y="${padT + 9}" fill="#5a6b7a" font-size="9">under-read</text>`
    + `<text x="${W - padR - 2}" y="${padT + 9}" fill="#5a6b7a" font-size="9" text-anchor="end">trusted</text>`
    + `<text x="${padL + 2}" y="${H - padB - 3}" fill="#5a6b7a" font-size="9">ignored</text>`
    + `<text x="${W - padR - 2}" y="${H - padB - 3}" fill="#5a6b7a" font-size="9" text-anchor="end">over-read</text>`
    + `<text x="4" y="${yOf(1) + 8}" fill="#5a6b7a" font-size="10">Q+1</text>`
    + `<text x="4" y="${yOf(-1)}" fill="#5a6b7a" font-size="10">−1</text>`
    + `<text x="${W / 2}" y="${H - 4}" fill="#5a6b7a" font-size="10" text-anchor="middle">prior (log)</text>`;
  for (const k of kids) {
    const r2 = 2 + 6 * Math.sqrt(k.n / maxN);
    const isBest = tree.best_action_id === k.action_id;
    const isRec = recorded != null && recorded === k.action_id;
    html += `<circle cx="${xOf(k.p).toFixed(1)}" cy="${yOf(k.qm).toFixed(1)}" r="${r2.toFixed(1)}" fill="${isBest ? "var(--green)" : "var(--accent)"}" opacity="0.75"${isRec ? ` stroke="var(--yellow)" stroke-width="2"` : ""}><title>${k.q},${k.r} · N ${k.n} · Q ${k.qm.toFixed(3)} · P ${(k.p * 100).toFixed(2)}%${isBest ? " · best" : ""}${isRec ? " · recorded move" : ""}</title></circle>`;
    if (isBest || isRec) {
      html += `<text x="${(xOf(k.p) + r2 + 2).toFixed(1)}" y="${(yOf(k.qm) + 3).toFixed(1)}" fill="${isBest ? "var(--green)" : "var(--yellow)"}" font-size="9">${k.q},${k.r}</text>`;
    }
  }
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}">${html}</svg>`;
}

// ---- visit ladder (spec S10) ---------------------------------------------------

async function dbgRunLadder() {
  dbgAbortPrefetch();  // S7: an explicit request preempts the speculative ply±1 fetch
  const nav = dbg.nav;
  if (!nav.ckptA || !dbg.position) {
    debugSetStatus("Pick a checkpoint and position first.", "error");
    return;
  }
  if (dbg.ladderBusy) return;
  const key = dbgCurrentKey();
  dbg.ladderBusy = true;
  dbg.ladder = { key, rungs: [] };
  dbgRenderLadder();
  const params = dbgTreeParams();
  try {
    for (const visits of [64, 128, 256, 512, 1024]) {
      if (dbgCurrentKey() !== key) break;  // user moved on — stop climbing
      debugSetStatus(`Visit ladder: ${visits} visits…`, "busy");
      const body = Object.assign(await dbgRequestBody(nav.ckptA), { visits, c_puct: params.c_puct, seed: params.seed });
      const s = await debugFetchJson("/api/debug/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      dbg.ladder.rungs.push({ visits, best: s.best, best_action_id: s.best_action_id, root_value: s.root_value });
      dbgRenderLadder();
    }
    // Only clear the shared status line while we still own the context — a nav
    // mid-ladder hands it to the newer action's busy message.
    if (dbgCurrentKey() === key) debugSetStatus("");
  } catch (e) {
    debugSetStatus(`Ladder: ${e.message}`, "error");
  } finally {
    dbg.ladderBusy = false;
    dbgRenderLadder();
  }
}

function dbgRenderLadder() {
  const el = dbgEl("dbgLadder");
  if (!el) return;
  const l = dbg.ladder && dbg.ladder.key === dbgCurrentKey() ? dbg.ladder : null;
  const btn = `<button type="button" id="dbgLadderRun" class="dbg-mini-btn"${dbg.ladderBusy ? " disabled" : ""}>${dbg.ladderBusy ? "Running…" : (l && l.rungs.length ? "Re-run" : "Run")}</button>`;
  if (!l || !l.rungs.length) {
    el.innerHTML = `<div class="dbg-flex-head"><span class="dbg-muted">Visit ladder (64 → 1024): best move per budget</span>${btn}</div>`;
    return;
  }
  // Flag EVERY rung where the best move flipped, not just the last one — the
  // early prior-vs-search flips are usually the diagnostic ones.
  const flips = new Set();
  for (let i = 1; i < l.rungs.length; i++) {
    if (l.rungs[i].best_action_id !== l.rungs[i - 1].best_action_id) flips.add(l.rungs[i].visits);
  }
  const rows = l.rungs.map(r => `<div class="dbg-move-row${flips.has(r.visits) ? " dbg-move-best" : ""}"><span></span><span>${r.visits}v</span><span>${r.best ? `${r.best.q},${r.best.r}` : "—"}</span><span>${r.root_value.toFixed(3)}</span><span>${flips.has(r.visits) ? "flip" : ""}</span><span></span></div>`).join("");
  el.innerHTML = `<div class="dbg-flex-head"><span class="dbg-muted">Visit ladder${flips.size ? ` · best flips at ${[...flips].join("/")}v` : " · stable best"}</span>${btn}</div>`
    + `<div class="dbg-move-row dbg-move-head"><span></span><span>visits</span><span>best</span><span>root v</span><span></span><span></span></div>` + rows;
}

// ---- pin tray (spec S2) ----------------------------------------------------------

function dbgPinsStorageKey() {
  return `hexoDbgPins:${dbg.nav.run}`;
}

function dbgLoadPins() {
  if (!dbg.nav.run || dbg.pinsRun === dbg.nav.run) return;
  dbg.pinsRun = dbg.nav.run;
  try {
    const raw = JSON.parse(window.localStorage.getItem(dbgPinsStorageKey()) || "[]");
    dbg.pins = Array.isArray(raw) ? raw : [];
  } catch (_e) {
    dbg.pins = [];
  }
  dbgRenderPinTray();
}

function dbgSavePins() {
  try {
    window.localStorage.setItem(dbgPinsStorageKey(), JSON.stringify(dbg.pins));
  } catch (e) {
    debugSetStatus(`Pins: ${e.message}`, "error");
  }
  dbgRenderPinTray();
}

function dbgCapPins() {
  // Soft cap 24 — drop oldest WITH a warning, never silently (spec §1.6).
  let dropped = 0;
  while (dbg.pins.length > 24) {
    dbg.pins.shift();
    dropped++;
  }
  if (dropped) debugSetStatus(`Pin tray over 24 — ${dropped} oldest pin${dropped > 1 ? "s" : ""} dropped.`, "info");
}

function dbgAddPin() {
  const nav = dbg.nav;
  if (!nav.run || !nav.path || nav.ply == null) {
    debugSetStatus("No position to pin.", "error");
    return;
  }
  dbgLoadPins();
  const a = dbgFreshData("analysis");
  const s = dbgFreshData("search");
  const best = s && s.best
    ? `${s.best.q},${s.best.r}`
    : (a && a.policy && a.policy[0] ? `${a.policy[0].q},${a.policy[0].r}` : null);
  dbg.pins.push({
    ts: Date.now(),
    note: "",
    nav: Object.assign({}, nav, { acts: nav.acts.slice() }),
    snap: {
      value: a ? a.value : null,
      best,
      ckpt: dbgCkptShort(nav.ckptA),
      thumb: debugCurrentPlacements().map(p => ({ q: p.q, r: p.r, player: p.player })),
    },
  });
  dbgCapPins();
  dbgSavePins();
  diagTap("pinned");
}

function dbgPinThumbSvg(thumb) {
  if (!thumb || !thumb.length) return `<svg viewBox="0 0 80 80"></svg>`;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const pts = thumb.map(p => ({ player: p.player, c: center(p.q, p.r) }));
  for (const p of pts) {
    minX = Math.min(minX, p.c.x - HEX);
    maxX = Math.max(maxX, p.c.x + HEX);
    minY = Math.min(minY, p.c.y - HEX);
    maxY = Math.max(maxY, p.c.y + HEX);
  }
  const stones = pts.map(p => `<circle cx="${p.c.x.toFixed(1)}" cy="${p.c.y.toFixed(1)}" r="${(HEX * 0.62).toFixed(1)}" fill="${playerColor(p.player)}"></circle>`).join("");
  return `<svg viewBox="${minX.toFixed(1)} ${minY.toFixed(1)} ${(maxX - minX).toFixed(1)} ${(maxY - minY).toFixed(1)}" preserveAspectRatio="xMidYMid meet">${stones}</svg>`;
}

function dbgRenderPinTray() {
  const chips = document.querySelector("#dbgPinTray .dbg-pin-chips");
  if (!chips) return;
  if (!dbg.pins.length) {
    chips.innerHTML = `<span class="dbg-empty-note">No pinned positions</span>`;
    return;
  }
  chips.innerHTML = dbg.pins.map((pin, i) => {
    const v = pin.snap && pin.snap.value != null ? pin.snap.value.toFixed(2) : "—";
    const branch = pin.nav && pin.nav.acts && pin.nav.acts.length ? `+${pin.nav.acts.length}` : "";
    const label = `ply ${pin.nav ? pin.nav.ply : "?"}${branch} · ${v} · ${(pin.snap && pin.snap.ckpt) || ""}`;
    const tip = pin.note || (pin.snap && pin.snap.best ? `best ${pin.snap.best}` : "pinned position");
    return `<div class="dbg-pin-chip" data-pin="${i}" title="${escapeAttr(tip)}">`
      + dbgPinThumbSvg(pin.snap && pin.snap.thumb)
      + `<span class="dbg-muted">${escapeText(label)}</span>`
      + `<span class="dbg-pin-chip-actions"><button type="button" class="dbg-pin-note dbg-mini-btn" data-pin="${i}" title="Edit note">✎</button><button type="button" class="dbg-pin-x dbg-mini-btn" data-pin="${i}" title="Remove pin">✕</button></span>`
      + `</div>`;
  }).join("");
}

function dbgRestorePin(i) {
  const pin = dbg.pins[i];
  if (!pin || !pin.nav) return;
  diagTap("pin restore");
  dbgNavigate(Object.assign({}, pin.nav, { acts: (pin.nav.acts || []).slice() }));
}

function dbgNotePin(i) {
  dbgLoadPins();
  const pin = dbg.pins[i];
  if (!pin) return;
  const note = window.prompt("Pin note:", pin.note || "");
  if (note == null) return;
  pin.note = note;
  dbgSavePins();
}

function dbgRemovePin(i) {
  dbgLoadPins();
  dbg.pins.splice(i, 1);
  dbgSavePins();
}

function dbgExportPins() {
  dbgLoadPins();
  if (!dbg.pins.length) {
    dbgStub("No pins to export.");
    return;
  }
  const blob = new Blob([JSON.stringify(dbg.pins, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `hexo-debug-pins-${dbg.nav.run || "run"}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function dbgImportPins(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(String(reader.result || "[]"));
      if (!Array.isArray(data)) throw new Error("not a pin array");
      dbgLoadPins();
      dbg.pins = dbg.pins.concat(data.filter(p => p && typeof p === "object"));
      dbgCapPins();
      dbgSavePins();
      debugSetStatus(`Imported ${data.length} pin${data.length === 1 ? "" : "s"}.`, "info");
    } catch (e) {
      debugSetStatus(`Pin import: ${e.message}`, "error");
    }
  };
  reader.readAsText(file);
}

// ---- session journal (spec S3) ------------------------------------------------------

function dbgJournalStorageKey() {
  return `hexoDbgJournal:${dbg.nav.run}`;
}

function dbgLoadJournal() {
  if (!dbg.nav.run || dbg.journalRun === dbg.nav.run) return;
  dbg.journalRun = dbg.nav.run;
  try {
    const raw = JSON.parse(window.localStorage.getItem(dbgJournalStorageKey()) || "[]");
    dbg.journal = Array.isArray(raw) ? raw : [];
  } catch (_e) {
    dbg.journal = [];
  }
  dbgRenderJournal();
}

function dbgJournalLog(analysis, ckpt, nav) {
  // Auto-log every completed analyze: {ts, ckpt, ply/branch, value, top move}.
  if (!nav.run) return;
  dbgLoadJournal();
  const top = analysis.policy && analysis.policy[0];
  dbg.journal.push({
    ts: Date.now(),
    ckpt: dbgCkptShort(ckpt),
    ply: nav.ply,
    branch: nav.acts.length,
    value: analysis.value,
    top: top ? `${top.q},${top.r}` : null,
    nav: Object.assign({}, nav, { acts: nav.acts.slice() }),
  });
  while (dbg.journal.length > 200) dbg.journal.shift();  // FIFO cap (spec §1.6)
  try {
    window.localStorage.setItem(dbgJournalStorageKey(), JSON.stringify(dbg.journal));
  } catch (_e) { /* storage full — journal degrades to in-memory */ }
  dbgRenderJournal();
}

function dbgRenderJournal() {
  const list = document.querySelector("#dbgJournal .dbg-journal-list");
  if (!list) return;
  if (!dbg.journal.length) {
    list.className = "dbg-journal-list dbg-empty-note";
    list.textContent = "No analyses logged yet";
    return;
  }
  list.className = "dbg-journal-list";
  list.innerHTML = dbg.journal.slice(-40).reverse().map((e, idx) => {
    const i = dbg.journal.length - 1 - idx;
    const t = new Date(e.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return `<div class="dbg-palette-item" data-journal="${i}" title="Click to restore this position">${t} · ${escapeText(e.ckpt || "")} · ply ${e.ply}${e.branch ? `+${e.branch}` : ""} · v ${e.value != null ? e.value.toFixed(2) : "—"}${e.top ? ` · ${e.top}` : ""}</div>`;
  }).join("");
}

function dbgClearJournal() {
  dbg.journal = [];
  try {
    window.localStorage.removeItem(dbgJournalStorageKey());
  } catch (_e) { /* ignore */ }
  dbgRenderJournal();
}

// ---- command palette (spec S1) -------------------------------------------------------

function dbgOpenPalette() {
  const pal = dbgEl("dbgPalette");
  const input = dbgEl("dbgPaletteInput");
  if (!pal || !input) return;
  dbg.paletteItems = dbgPaletteAllItems();
  dbg.paletteIdx = 0;
  pal.hidden = false;
  input.value = "";
  dbgRenderPaletteList("");
  dbgEnsurePaletteGames();
  window.setTimeout(() => input.focus(), 0);
}

async function dbgEnsurePaletteGames() {
  // The palette searches BOTH sources (typing "eval" must surface evaluation
  // files even while src=selfplay) — one lazy all-source listing per run.
  if (!dbg.nav.run || dbg.paletteGamesRun === dbg.nav.run) return;
  try {
    const data = await debugFetchJson(`/api/debug/games?run=${encodeURIComponent(dbg.nav.run)}&source=all`);
    dbg.paletteGames = data.games || [];
    dbg.paletteGamesRun = dbg.nav.run;
    dbg.paletteItems = dbgPaletteAllItems();
    const input = dbgEl("dbgPaletteInput");
    const pal = dbgEl("dbgPalette");
    if (pal && !pal.hidden) dbgRenderPaletteList(input ? input.value : "");
  } catch (_e) { /* palette degrades to the current-source game list */ }
}

function dbgPaletteAllItems() {
  const items = [
    { label: "Action: Analyze now (A)", run: () => dbgAnalyzeNow(true) },
    { label: "Action: Run root search (S)", run: () => dbgRunSearch() },
    { label: "Action: Run debug tree (T)", run: () => dbgRunTree() },
    { label: "Action: Sweep game", run: () => dbgRunSweep() },
    { label: "Action: Return to game (G)", run: () => dbgNavigate({ acts: [] }) },
    { label: "Action: Pin current position (P)", run: () => dbgAddPin() },
    { label: "Action: Export pins", run: () => dbgExportPins() },
    { label: "Action: Run visit ladder", run: () => dbgRunLadder() },
    { label: "Action: Checkpoint sweep", run: () => dbgRunCkptSweep() },
  ];
  const games = dbg.paletteGamesRun === dbg.nav.run && dbg.paletteGames.length ? dbg.paletteGames : dbg.games;
  for (const g of games) {
    items.push({
      label: `Game file: ${g.path}`,
      run: () => dbgNavigate({
        src: String(g.path).startsWith("eval") ? "evaluation" : "selfplay",
        path: g.path, rec: 0, ply: null, acts: [],
      }),
    });
  }
  for (const r of dbg.records) {
    items.push({ label: `Record: ${dbgRecordLabel(r)}`, run: () => dbgNavigate({ rec: r.index, ply: null, acts: [] }) });
  }
  for (const c of dbg.checkpoints) {
    items.push({ label: `Checkpoint: ${dbgCkptLabel(c)} (${c.name})`, run: () => dbgNavigate({ ckptA: c.name }) });
  }
  dbg.pins.forEach((pin, i) => {
    items.push({
      label: `Pin: ply ${pin.nav ? pin.nav.ply : "?"} · ${pin.note || (pin.snap && pin.snap.ckpt) || ""}`,
      run: () => dbgRestorePin(i),
    });
  });
  for (const e of dbg.journal.slice(-20)) {
    items.push({
      label: `Journal: ply ${e.ply}${e.branch ? `+${e.branch}` : ""} · ${e.ckpt} · v ${e.value != null ? e.value.toFixed(2) : "—"}`,
      run: () => dbgNavigate(Object.assign({}, e.nav, { acts: ((e.nav && e.nav.acts) || []).slice() })),
    });
  }
  return items;
}

function dbgPaletteFilter(qstr) {
  const q = qstr.trim().toLowerCase();
  const items = [];
  const plyNum = /^(?:ply\s*)?(\d+)$/.exec(q);
  if (plyNum) items.push({ label: `Go to ply ${plyNum[1]}`, run: () => dbgGotoPly(Number(plyNum[1])) });
  if (!q) return items.concat(dbg.paletteItems.slice(0, 12));
  const scored = [];
  for (const it of dbg.paletteItems) {
    const l = it.label.toLowerCase();
    let score = null;
    const sub = l.indexOf(q);
    if (sub !== -1) score = 1000 - sub;       // substring beats subsequence
    else {
      let i = 0;
      for (const ch of l) if (ch === q[i]) i++;
      if (i >= q.length) score = 100 - l.length * 0.1;  // fuzzy subsequence
    }
    if (score != null) scored.push([score, it]);
  }
  scored.sort((x, y) => y[0] - x[0]);
  return items.concat(scored.slice(0, 14).map(s => s[1]));
}

function dbgRenderPaletteList(qstr) {
  const list = document.querySelector("#dbgPalette .dbg-palette-list");
  if (!list) return;
  dbg.paletteShown = dbgPaletteFilter(qstr);
  dbg.paletteIdx = Math.max(0, Math.min(dbg.paletteIdx, dbg.paletteShown.length - 1));
  list.innerHTML = dbg.paletteShown.length
    ? dbg.paletteShown.map((it, i) => `<div class="dbg-palette-item${i === dbg.paletteIdx ? " active" : ""}" data-pal="${i}">${escapeText(it.label)}</div>`).join("")
    : `<div class="dbg-empty-note">No matches</div>`;
}

function dbgPaletteExec(i) {
  const it = dbg.paletteShown && dbg.paletteShown[i];
  const pal = dbgEl("dbgPalette");
  if (pal) pal.hidden = true;
  if (!it) return;
  try {
    it.run();
  } catch (e) {
    reportError("palette: " + (e && (e.stack || e.message) || e));
  }
}

// ---- checkpoint sweep dock tab (spec S4) ----------------------------------------------

async function dbgRunCkptSweep() {
  dbgAbortPrefetch();  // S7: an explicit request preempts the speculative ply±1 fetch
  const nav0 = Object.assign({}, dbg.nav, { acts: dbg.nav.acts.slice() });
  if (!nav0.run || !nav0.path || nav0.ply == null) {
    debugSetStatus("No position for the checkpoint sweep.", "error");
    return;
  }
  if (!dbg.checkpoints.length) {
    debugSetStatus("No checkpoints in this run.", "error");
    return;
  }
  if (dbg.ckptSweep && dbg.ckptSweep.running) {
    dbgStub("Checkpoint sweep already running — Stop aborts it.");
    return;
  }
  // navKey ties the finished curve to the position it swept (ckpt slot empty —
  // the sweep spans checkpoints); the renderer flags it stale after a nav away.
  const sweep = { navKey: dbgCacheKey(nav0, ""), rows: [], running: true, abort: false, refTop: null, refTopCell: "" };
  dbg.ckptSweep = sweep;
  const aCur = dbgFreshData("analysis");
  if (aCur && aCur.policy && aCur.policy[0]) {
    sweep.refTop = aCur.policy[0].action_id;
    sweep.refTopCell = `${aCur.policy[0].q},${aCur.policy[0].r}`;
  }
  const list = dbg.checkpoints.slice()
    .sort((x, y) => (x.epoch != null ? x.epoch : 1e12) - (y.epoch != null ? y.epoch : 1e12));
  const sameNav = () => dbg.nav.run === nav0.run && dbg.nav.path === nav0.path
    && dbg.nav.rec === nav0.rec && dbg.nav.ply === nav0.ply
    && dbg.nav.acts.join(",") === nav0.acts.join(",");
  dbgRenderCkptSweep();
  try {
    // Branch prefix needs the recorded action list — await it (never fall back
    // to an empty prefix, which would sweep the wrong, near-empty position).
    const bodyBase = nav0.acts.length
      ? { run: nav0.run, action_ids: (await dbgEnsureRecordedActs()).slice(0, nav0.ply).concat(nav0.acts) }
      : { run: nav0.run, path: nav0.path, record: nav0.rec, ply: nav0.ply };
    if (nav0.radius) bodyBase.radius = nav0.radius;  // run the sweep at the active radius
    for (const ck of list) {
      if (sweep.abort || !sameNav()) break;
      const key = dbgCacheKey(nav0, ck.name);
      const entry = dbgCacheEntry(key);
      let analysis = entry.analysis;
      if (!analysis) {
        debugSetStatus(`Checkpoint sweep: ${dbgCkptLabel(ck)}…`, "busy");
        analysis = await debugFetchJson("/api/debug/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(Object.assign({ checkpoint: ck.name }, bodyBase)),
        });
        entry.analysis = analysis;  // lands in the M12 cache — flipping ckptA later is free
      }
      const top = analysis.policy && analysis.policy[0];
      sweep.rows.push({
        name: ck.name,
        label: ck.epoch != null ? `e${ck.epoch}` : ck.name.replace(/\.pt$/, ""),
        value: analysis.value,
        top: top ? top.action_id : null,
        topCell: top ? `${top.q},${top.r}` : "",
      });
      dbgRenderCkptSweep();
    }
    if (!sweep.refTop) {
      const own = sweep.rows.find(r => r.name === nav0.ckptA);
      if (own) {
        sweep.refTop = own.top;
        sweep.refTopCell = own.topCell;
      }
    }
    // Clear the shared status line only while we still own the context — a nav
    // mid-sweep hands it to the newer action's busy message.
    if (sweep.abort) debugSetStatus("Checkpoint sweep stopped.");
    else if (sameNav()) debugSetStatus("");
  } catch (e) {
    debugSetStatus(`Checkpoint sweep: ${e.message}`, "error");
  } finally {
    sweep.running = false;
    dbgRenderCkptSweep();
  }
}

function dbgRenderCkptSweep() {
  const el = document.querySelector("#dbgCkptSweep .dbg-ckpt-sweep-chart");
  if (!el) return;
  const cs = dbg.ckptSweep;
  // The curve belongs to the position it swept — once the user navigates away
  // it must say so instead of posing as the current position's history.
  const stale = Boolean(cs && cs.rows.length && cs.navKey !== dbgCacheKey(dbg.nav, ""));
  // Called from dbgRenderAll on every render (the Run button only exists in this
  // markup) — skip the innerHTML swap when nothing observable changed.
  const sig = cs ? `${cs.rows.length}|${cs.running ? 1 : 0}|${cs.refTop}|${cs.refTopCell}|${stale ? 1 : 0}` : "none";
  if (el.__dbgSig === sig) return;
  el.__dbgSig = sig;
  const btn = `<button type="button" id="dbgCkptSweepRun" class="dbg-mini-btn"${cs && cs.running ? " disabled" : ""}>${cs && cs.running ? "Running…" : (cs && cs.rows.length ? "Re-run on current position" : "Run on current position")}</button>`;
  if (!cs || !cs.rows.length) {
    el.innerHTML = `<div class="dbg-flex-head"><span class="dbg-empty-note">${cs && cs.running ? "Evaluating checkpoints…" : "Not run"}</span>${btn}</div>`;
    return;
  }
  const rows = cs.rows;
  const W = 1000, H = 200, padL = 36, padR = 12, padT = 12, padB = 26;
  const xs = i => padL + (W - padL - padR) * (rows.length > 1 ? i / (rows.length - 1) : 0.5);
  const y = v => padT + (H - padT - padB) * (1 - (v + 1) / 2);
  let html = `<line x1="${padL}" y1="${y(0)}" x2="${W - padR}" y2="${y(0)}" stroke="#2c3d50"></line>`
    + `<text x="4" y="${y(1) + 4}" fill="#5a6b7a" font-size="11">+1</text>`
    + `<text x="4" y="${y(-1) + 2}" fill="#5a6b7a" font-size="11">−1</text>`
    + `<path d="${rows.map((r, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)},${y(r.value).toFixed(1)}`).join("")}" fill="none" stroke="${stale ? "#5a6b7a" : "var(--accent)"}" stroke-width="2"></path>`;
  const labelEvery = Math.max(1, Math.ceil(rows.length / 12));
  rows.forEach((r, i) => {
    const agree = cs.refTop != null && r.top === cs.refTop;
    html += `<circle cx="${xs(i).toFixed(1)}" cy="${y(r.value).toFixed(1)}" r="3.5" fill="${agree ? "var(--green)" : (stale ? "#5a6b7a" : "var(--accent)")}"><title>${r.label} · v ${r.value.toFixed(3)} · top ${r.topCell || "—"}${agree ? " (= current top)" : ""}</title></circle>`;
    if (agree) html += `<line x1="${xs(i).toFixed(1)}" y1="${H - padB + 3}" x2="${xs(i).toFixed(1)}" y2="${H - padB + 10}" stroke="var(--green)" stroke-width="2"></line>`;
    if (i % labelEvery === 0) html += `<text x="${xs(i).toFixed(1)}" y="${H - 4}" fill="#5a6b7a" font-size="10" text-anchor="middle">${r.label}</text>`;
  });
  const headTxt = stale
    ? `<span class="dbg-muted">stale — swept a different position; Re-run for current</span>`
    : `<span class="dbg-muted">value (stm) per checkpoint · green tick = top move matches current${cs.refTopCell ? ` (${cs.refTopCell})` : ""}</span>`;
  el.innerHTML = `<div class="dbg-flex-head">${headTxt}${btn}</div>`
    + `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${html}</svg>`;
}

// ---- ply±1 prefetch (spec S7) ---------------------------------------------------------

function dbgAbortPrefetch() {
  if (dbg.prefetchAbort) {
    dbg.prefetchAbort.abort();
    dbg.prefetchAbort = null;
  }
}

function dbgMaybePrefetch() {
  // Client-side only, at most ONE low-priority in-flight request for ply±1,
  // never for checkpoint B, only when the worker dot is green and the user has
  // nothing pending. Aborted the instant the user acts (dbgNavigate).
  const nav = dbg.nav;
  if (dbg.prefetchAbort || dbg.loading || nav.acts.length || nav.ply == null) return;
  if (!dbg.worker || !dbg.worker.alive || !nav.ckptA || !nav.path) return;
  const total = dbgRecordedTotal();
  let target = null;
  for (const d of [1, -1]) {
    const p = nav.ply + d;
    if (p < 0 || (total != null && p > total)) continue;
    const key = dbgCacheKey(Object.assign({}, nav, { ply: p }), nav.ckptA);
    const entry = dbg.cache.get(key);
    if (!entry || !entry.position || !entry.analysis) {
      target = { ply: p, key };
      break;
    }
  }
  if (!target) return;
  const ctl = new AbortController();
  dbg.prefetchAbort = ctl;
  (async () => {
    try {
      const entry = dbgCacheEntry(target.key);
      if (!entry.position) {
        const params = new URLSearchParams({ run: nav.run, path: nav.path, record: String(nav.rec), ply: String(target.ply) });
        dbgRadiusParam(params);
        const res = await fetch(`/api/debug/position?${params.toString()}`, { signal: ctl.signal });
        const data = await safeJson(res);
        if (!res.ok || !data || data.error) return;
        entry.position = data;
      }
      if (!entry.analysis && !entry.analysisPending) {
        entry.analysisPending = true;
        try {
          const body = { run: nav.run, checkpoint: nav.ckptA, path: nav.path, record: nav.rec, ply: target.ply };
          if (nav.radius) body.radius = nav.radius;
          const res = await fetch("/api/debug/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: ctl.signal,
          });
          const data = await safeJson(res);
          if (res.ok && data && !data.error) entry.analysis = data;
        } finally {
          entry.analysisPending = false;
        }
      }
    } catch (_e) { /* aborted or failed — prefetch is best-effort */ }
    finally {
      if (dbg.prefetchAbort === ctl) dbg.prefetchAbort = null;
    }
  })();
}

// ---- INPUTS tab: featurizer input planes -----------------------
// The shrimp lineage is graph-featurized (support-set node tokens), so it has
// no dense per-cell input planes: analyze returns input_planes: null and this
// tab renders an n/a note. The plane-rendering path below is retained for any
// checkpoint that DOES return dense planes.

async function dbgEnsureInputs(force) {
  const nav = dbg.nav;
  if (!nav.run || !nav.path || nav.ply == null || !nav.ckptA) return;
  const key = dbgCurrentKey();
  const entry = dbgCacheEntry(key);
  if (force) delete entry.planes;
  if (entry.planes !== undefined || entry.planesPending) {
    dbgRenderInputs();
    return;
  }
  entry.planesPending = true;
  dbgRenderInputs();
  try {
    const body = Object.assign(await dbgRequestBody(nav.ckptA), { planes: true });
    const data = await debugFetchJson("/api/debug/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    entry.planes = data.input_planes || null;  // null = graph-featurized lineage (e.g. shrimp)
    if (!entry.analysis) entry.analysis = data;  // a planes analyze is a full analyze
  } catch (e) {
    debugSetStatus(`Inputs: ${e.message}`, "error");
  } finally {
    entry.planesPending = false;
  }
  if (key === dbgCurrentKey()) {
    dbgRenderInputs();
    dbgSyncPlaneSelect();
    if (dbg.nav.mode === "plane") dbgRenderBoard();
  }
}

function dbgRenderInputs() {
  const el = dbgEl("dbgPlanes");
  if (!el) return;
  const entry = dbg.cache.get(dbgCurrentKey());
  if (entry && entry.planesPending) {
    el.__dbgPlanesSig = "";
    el.innerHTML = `<div class="dbg-empty-note">Loading input planes…</div>`;
    return;
  }
  const planes = entry ? entry.planes : undefined;
  if (planes === undefined) {
    el.__dbgPlanesSig = "";
    el.innerHTML = `<div class="dbg-empty-note">Featurizer input planes load on first open (n/a for graph-featurized lineages)</div>`;
    return;
  }
  if (planes === null) {
    el.__dbgPlanesSig = "";
    el.innerHTML = `<div class="dbg-empty-note">n/a for the shrimp lineage (graph featurizer — no input planes)</div>`;
    return;
  }
  const dimH = (planes.shape && planes.shape[1]) || 41;
  const dimW = (planes.shape && planes.shape[2]) || 41;
  const sig = `${dbgCurrentKey()}|${(planes.names || []).length}`;
  if (el.__dbgPlanesSig === sig) return;  // canvases already drawn for this key
  el.__dbgPlanesSig = sig;
  el.innerHTML = (planes.names || []).map((name, i) =>
    `<div class="dbg-plane-cell"><canvas data-plane="${i}" width="${dimW}" height="${dimH}"></canvas>${escapeText(name)}</div>`).join("");
  el.querySelectorAll("canvas[data-plane]").forEach(cv => {
    const i = Number(cv.dataset.plane);
    const data = (planes.data && planes.data[i]) || [];
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(dimW, dimH);
    let maxV = 0;
    for (const v of data) maxV = Math.max(maxV, Math.abs(v));
    maxV = maxV || 1;
    for (let px = 0; px < data.length && px < dimW * dimH; px++) {
      const t = Math.max(-1, Math.min(1, data[px] / maxV));
      const o = px * 4;
      img.data[o] = t < 0 ? 255 : 39;       // negative -> p1 red, positive -> accent
      img.data[o + 1] = t < 0 ? 86 : 215;
      img.data[o + 2] = t < 0 ? 80 : 230;
      img.data[o + 3] = Math.round(255 * Math.abs(t));
    }
    ctx.putImageData(img, 0, 0);
  });
}

function dbgSyncPlaneSelect() {
  const sel = dbgEl("dbgPlaneSelect");
  if (!sel) return;
  const entry = dbg.cache.get(dbgCurrentKey());
  const names = (entry && entry.planes && entry.planes.names) || [];
  const html = names.map((nm, i) => `<option value="${i}">${escapeText(nm)}</option>`).join("");
  if (sel.__dbgSig !== html) {
    const prev = sel.value;
    sel.__dbgSig = html;
    sel.innerHTML = html || `<option value="0">plane 0</option>`;
    sel.value = prev;
    if (!sel.value) sel.value = "0";
  }
}

// ---- ATTENTION tab + board mode (shrimp lineage only) ----------------------

// Best-known lineage for the loaded ckptA. Prefer the analyze meta (present after
// any analyze), then the CKPT-tab provenance (loaded independently). Used ONLY to
// disable the Attn mode/tab for dense/hexgt — the worker is the authority and
// still returns found:false for non-shrimp, so a wrong guess only delays a note.
function dbgAttnLineage() {
  const a = dbgFreshData("analysis");
  if (a && a.meta && a.meta.lineage) return String(a.meta.lineage);
  const info = dbgCkptInfoAny(dbg.nav.ckptA);
  if (info && info.meta && info.meta.lineage) return String(info.meta.lineage);
  return "";
}

function dbgAttnLineageOk() {
  const lin = dbgAttnLineage();
  // Empty (unknown) is allowed — let the worker decide; only KNOWN non-shrimp
  // lineages are inert so we never fetch for dense/hexgt.
  return lin === "" || lin === "shrimp";
}

function dbgHasCellQ() {
  // Lineage-level feature gate for the per-cell Q head (v3). Detected from the
  // model meta (state-dict presence of cell_q_head), surfaced as meta.has_cell_q
  // on the committed analyze payload first, then the cached ckpt_info. Unknown
  // (no meta yet) stays false so the Q button is disabled until we know.
  const a = dbgFreshData("analysis");
  if (a && a.meta && typeof a.meta.has_cell_q === "boolean") return a.meta.has_cell_q;
  const info = dbgCkptInfoAny(dbg.nav.ckptA);
  if (info && info.meta && typeof info.meta.has_cell_q === "boolean") return info.meta.has_cell_q;
  return false;
}

// The attn slot tail (everything after the position key) — used to key the
// per-entry attn payload cache so revisiting a query/block/head is free.
function dbgAttnSlotTail() {
  const head = dbg.attnHead == null ? "mean" : String(dbg.attnHead);
  return `${dbgAttnQtag()}::${dbg.attnBlock}::${head}`;
}

async function dbgEnsureAttn(force) {
  const nav = dbg.nav;
  if (nav.tab !== "attn" && nav.mode !== "attn") return;  // only when relevant
  if (!dbg.attnQuery) {
    dbgRenderAttn();
    return;
  }
  if (!dbgAttnLineageOk()) {
    dbgRenderAttn();  // inert for known dense/hexgt — render the n/a note, no fetch
    return;
  }
  if (!nav.run || !nav.path || nav.ply == null || !nav.ckptA) {
    dbgRenderAttn();
    return;
  }
  const slotKey = dbgAttnSlotKey();
  const entry = dbgCacheEntry(dbgCurrentKey());
  if (!entry.attnSlots) entry.attnSlots = new Map();
  const tail = dbgAttnSlotTail();
  if (force) entry.attnSlots.delete(tail);
  const cached = entry.attnSlots.get(tail);
  if (cached !== undefined) {
    // Cache hit — commit to the live slot and paint.
    dbg.attn = cached;
    dbg.keys.attn = slotKey;
    dbgRenderAttn();
    if (nav.mode === "attn") dbgRenderBoard();
    return;
  }
  dbgFetchAttn(slotKey, tail, entry);
}

async function dbgFetchAttn(slotKey, tail, entry) {
  const seq = ++dbg.attnSeq;  // latest-wins (sibling to anlSeq/posSeq)
  dbg.attnPending = true;
  dbgRenderAttn();
  try {
    const nav = dbg.nav;
    const body = Object.assign(await dbgRequestBody(nav.ckptA), {
      block: dbg.attnBlock,
      head: dbg.attnHead,  // null = mean over heads
      query: dbg.attnQuery,
      n: null,
    });
    const data = await debugFetchJson("/api/debug/attention", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (seq !== dbg.attnSeq) return;  // superseded by a newer selection — drop
    entry.attnSlots.set(tail, data);
    dbg.attn = data;
    dbg.keys.attn = slotKey;
  } catch (e) {
    if (seq === dbg.attnSeq) debugSetStatus(`Attention: ${e.message}`, "error");
  } finally {
    if (seq === dbg.attnSeq) dbg.attnPending = false;
  }
  if (slotKey === dbgAttnSlotKey()) {
    dbgRenderAttn();
    if (dbg.nav.mode === "attn") dbgRenderBoard();
  }
}

function dbgAttnFmtPct(v) {
  return `${((Number(v) || 0) * 100).toFixed(1)}%`;
}

// Resolve the active query's row from the cached payload (no refetch when
// switching among the 8 token chips — all rows ride in token_queries).
function dbgAttnActiveRow(attn) {
  if (!attn || !attn.found || !dbg.attnQuery) return null;
  if (dbg.attnQuery.type === "token") {
    const id = Math.max(0, Math.min(dbg.attnQuery.id, (attn.token_queries || []).length - 1));
    return (attn.token_queries || [])[id] || null;
  }
  return attn.cell_query || null;  // null until the cell-query payload lands
}

// Rebuild the block/head <select> options to the loaded model's arch (payload
// num_blocks/num_heads): e.g. a 3-block x 4-head net, or the main_7 5-block x
// 3-head net. Idempotent — only touches the DOM when the counts change.
function dbgSyncAttnArch(attn) {
  if (!attn) return;
  const nb = Number(attn.num_blocks) || 0;
  const nh = Number(attn.num_heads) || 0;
  if (nb > 0 && nb !== dbg.attnNumBlocks) {
    dbg.attnNumBlocks = nb;
    const sel = dbgEl("dbgAttnBlockSel");
    if (sel) {
      sel.innerHTML = "";
      for (let i = 0; i < nb; i++) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = String(i);
        sel.appendChild(opt);
      }
    }
  }
  if (nh > 0 && nh !== dbg.attnNumHeads) {
    dbg.attnNumHeads = nh;
    const sel = dbgEl("dbgAttnHeadSel");
    if (sel) {
      sel.innerHTML = "";
      const mean = document.createElement("option");
      mean.value = "";
      mean.textContent = "mean";
      sel.appendChild(mean);
      const mx = document.createElement("option");
      mx.value = "max";
      mx.textContent = "max";
      sel.appendChild(mx);
      for (let i = 0; i < nh; i++) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = String(i);
        sel.appendChild(opt);
      }
    }
  }
}

function dbgRenderAttn() {
  const body = dbgEl("dbgAttnBody");
  if (!body) return;
  dbgSyncAttnArch(dbgFreshData("attn"));
  const blockSel = dbgEl("dbgAttnBlockSel");
  const headSel = dbgEl("dbgAttnHeadSel");
  if (blockSel) blockSel.value = String(dbg.attnBlock);
  if (headSel) headSel.value = dbg.attnHead == null ? "" : String(dbg.attnHead);
  const tokensEl = dbgEl("dbgAttnTokens");
  const barsEl = dbgEl("dbgAttnTokenBars");
  const incomingEl = dbgEl("dbgAttnIncoming");
  const readoutEl = dbgEl("dbgAttnQueryReadout");
  const noteEl = dbgEl("dbgAttnNote");
  const hide = el => { if (el) el.hidden = true; };

  // Known non-shrimp lineage — inert n/a state, hide the interactive blocks.
  if (!dbgAttnLineageOk()) {
    hide(tokensEl); hide(barsEl); hide(incomingEl); hide(readoutEl);
    if (noteEl) {
      noteEl.hidden = false;
      noteEl.textContent = "Attention map is available for the shrimp lineage only (n/a here).";
    }
    return;
  }

  const attn = dbgFreshData("attn");

  // Always render the 8 token chips when we have a payload (cheap; lets the user
  // switch tokens with no refetch). Active chip reflects dbg.attnQuery.
  if (tokensEl) {
    const numTokens = attn && attn.found ? attn.num_tokens : 8;
    const activeTok = dbg.attnQuery && dbg.attnQuery.type === "token" ? dbg.attnQuery.id : -1;
    let html = "";
    for (let t = 0; t < numTokens; t++) {
      const cls = "dbg-attn-chip" + (t === activeTok ? " active" : "");
      html += `<button type="button" class="${cls}" data-attn-token="${t}">T${t}</button>`;
    }
    if (tokensEl.__dbgSig !== html) {
      tokensEl.__dbgSig = html;
      tokensEl.innerHTML = html;
    }
    tokensEl.hidden = false;
  }

  if (readoutEl) {
    readoutEl.hidden = false;
    const blk = attn && attn.found ? attn.block : dbg.attnBlock;
    const hd = attn && attn.found ? attn.head : dbg.attnHead;
    const where = `block ${blk}${hd == null ? " · mean heads" : (hd === "max" ? " · max heads" : " · head " + hd)}`;
    if (!dbg.attnQuery) readoutEl.textContent = `No query — ${where}`;
    else if (dbg.attnQuery.type === "token") readoutEl.textContent = `Query: token T${dbg.attnQuery.id} — ${where}`;
    else readoutEl.textContent = `Query: cell #${dbg.attnQuery.id} — ${where}`;
  }

  const row = dbgAttnActiveRow(attn);

  // Token bars: the active query's attention onto the 8 summary tokens.
  if (barsEl) {
    if (row && row.attn_over_tokens) {
      const maxV = Math.max(1e-9, ...row.attn_over_tokens);
      let html = `<div class="dbg-attn-bars-title">onto tokens</div>`;
      row.attn_over_tokens.forEach((v, t) => {
        const w = Math.max(0, Math.min(100, (v / maxV) * 100));
        html += `<div class="dbg-attn-bar-row"><span class="dbg-attn-bar-lab">T${t}</span>`
          + `<span class="dbg-attn-bar-track"><span class="dbg-attn-bar-fill" style="width:${w.toFixed(1)}%"></span></span>`
          + `<span class="dbg-attn-bar-val">${dbgAttnFmtPct(v)}</span></div>`;
      });
      barsEl.innerHTML = html;
      barsEl.hidden = false;
    } else {
      barsEl.innerHTML = "";
      barsEl.hidden = true;
    }
  }

  // Incoming readout: for a cell query, how much each token attends TO this cell.
  if (incomingEl) {
    const inc = attn && attn.found ? attn.incoming_token_to_cell : null;
    if (dbg.attnQuery && dbg.attnQuery.type === "cell" && inc) {
      let html = `<div class="dbg-attn-bars-title">incoming (tokens &#8594; this cell)</div>`;
      inc.forEach((v, t) => {
        html += `<span class="dbg-attn-inc-chip">T${t} ${dbgAttnFmtPct(v)}</span>`;
      });
      incomingEl.innerHTML = html;
      incomingEl.hidden = false;
    } else {
      incomingEl.innerHTML = "";
      incomingEl.hidden = true;
    }
  }

  if (noteEl) {
    let msg = "";
    if (dbg.attnPending) msg = "Loading attention…";
    else if (!dbg.attnQuery) msg = "Select a board cell (in Attn mode) or a token chip above.";
    else if (!attn) msg = "Loading attention…";
    else if (!attn.found) msg = attn.reason === "lineage_na"
      ? "Attention map is available for the shrimp lineage only (n/a here)."
      : (attn.reason === "bad_query" ? "That cell is not a support cell here — pick another." : "No attention data.");
    else if (dbg.attnQuery.type === "cell" && !attn.cell_query) msg = "Loading cell attention…";
    noteEl.hidden = !msg;
    noteEl.textContent = msg;
  }
}

// ---- COMPARE helpers (spec S5 + client Δ overlay) ---------------------------------------

function dbgAnalysisBFresh() {
  const nav = dbg.nav;
  if (!nav.ckptB) return null;
  return dbg.keys.analysisB === dbgCacheKey(nav, nav.ckptB) ? dbg.analysisB : null;
}

function dbgClearCmpHeat() {
  if (!dbg.cmpHeat) return;
  dbg.cmpHeat = false;
  const btn = dbgEl("dbgCmpHeat");
  if (btn) btn.classList.remove("active");
}

function dbgToggleCmpHeat() {
  if (!dbg.nav.ckptB) {
    dbgStub("Select checkpoint B in the context strip first.");
    return;
  }
  dbg.cmpHeat = !dbg.cmpHeat;
  const btn = dbgEl("dbgCmpHeat");
  if (btn) btn.classList.toggle("active", dbg.cmpHeat);
  dbgRenderBoard();
}

function dbgToggleSplit(on) {
  if (on && !dbg.nav.ckptB) {
    dbgStub("Select checkpoint B in the context strip first.");
    const cb = dbgEl("dbgCmpSplit");
    if (cb) cb.checked = false;
    return;
  }
  dbg.split = on;
  const wrap = document.querySelector(".dbg-board-wrap");
  if (wrap) wrap.classList.toggle("dbg-split", on);
  const svgB = dbgEl("dbgBoardSvgB");
  if (svgB) svgB.hidden = !on;
  const cb = dbgEl("dbgCmpSplit");
  if (cb) cb.checked = on;
  dbgRenderBoard();
}

function dbgRenderBoardB() {
  // Half-size B board for the A/B split (S5): same cells (dbg.cellIndex carries
  // the A render's geometry), heat = checkpoint B's prior, shared view/zoom.
  const svgB = dbgEl("dbgBoardSvgB");
  if (!svgB || !dbg.split) return;
  const pos = dbg.position;
  if (!pos) {
    svgB.innerHTML = "";
    return;
  }
  if (dbg.view) svgB.setAttribute("viewBox", `${dbg.view.x} ${dbg.view.y} ${dbg.view.width} ${dbg.view.height}`);
  svgB.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const b = dbgAnalysisBFresh();
  const heatB = new Map();
  let maxP = 0;
  if (b) {
    for (const row of b.policy || []) {
      heatB.set(`${row.q},${row.r}`, row.p);
      maxP = Math.max(maxP, row.p);
    }
  }
  let html = "";
  for (const cell of dbg.cellIndex.values()) {
    if (cell.x == null) continue;
    const isStone = Boolean(cell.placement);
    const fill = isStone ? playerColor(cell.placement.player) : "#101924";
    html += `<path class="dbg-cell" d="${path(cell.x, cell.y, HEX - 1)}" fill="${fill}" stroke="${isStone ? "#708296" : "#2c3d50"}" stroke-width="1" opacity="${isStone ? "1" : "0.7"}" data-q="${cell.q}" data-r="${cell.r}"></path>`;
    const p = heatB.get(cell.key);
    if (!isStone && p != null && maxP > 0) {
      let t = p / maxP;
      if (dbg.logScale) t = Math.log1p(99 * t) / Math.log1p(99);
      if (t > 0.015) html += `<path d="${path(cell.x, cell.y, HEX - 2)}" fill="var(--accent)" opacity="${(dbg.opacity * (0.10 + 0.90 * t)).toFixed(3)}" pointer-events="none"></path>`;
    }
  }
  if (!b && dbg.view) {
    html += `<text x="${(dbg.view.x + 8).toFixed(1)}" y="${(dbg.view.y + 18).toFixed(1)}" fill="#5a6b7a" font-size="12">B analysis pending…</text>`;
  }
  svgB.innerHTML = html;
}

// ---- rendering ---------------------------------------------------------------

function dbgRenderAll() {
  try {
    dbgSyncControls();
    dbgRenderCrumb();
    dbgRenderPlyRail();
    dbgRenderBoard();
    dbgRenderBranchBar();
    dbgRenderHeads();
    dbgRenderSearchPanel();
    dbgRenderTargets();
    dbgRenderComparePanel();
    dbgRenderCkptPanel();
    if (dbg.nav.tab === "inputs") {
      dbgRenderInputs();
      dbgSyncPlaneSelect();
    }
    dbgRenderAttn();
    dbgRenderDockChart();
    dbgRenderRegretList();  // worst-plies-by-regret tab + its visibility gate
    dbgRenderCkptSweep();  // keeps the Run button present before any sweep ran
    dbgUpdateStaleDots();
  } catch (e) {
    reportError("dbgRenderAll: " + (e && (e.stack || e.message) || e));
  }
}

function dbgSyncSelect(id, html, value) {
  const sel = dbgEl(id);
  if (!sel) return;
  if (sel.__dbgSig !== html) {
    sel.__dbgSig = html;
    sel.innerHTML = html;
  }
  sel.value = value;
}

function dbgCkptLabel(c) {
  const label = c.epoch != null ? `epoch ${c.epoch}` : c.name.replace(/\.pt$/, "");
  return c.graft ? `${label} · ${c.graft}` : label;
}

function dbgCkptShort(name) {
  const ck = dbg.checkpoints.find(c => c.name === name);
  return ck && ck.epoch != null ? `e${ck.epoch}` : String(name).replace(/\.pt$/, "");
}

function dbgRecordLabel(r) {
  // Per-record metadata chips: "g3 · 142mv · P1 · running".
  const win = r.winner ? ` · ${String(r.winner).replace("player", "P")}` : "";
  const status = r.status && r.status !== "final" ? ` · ${r.status}` : "";
  return `g${r.index} · ${r.actions}mv${win}${status}`;
}

function dbgSyncControls() {
  const nav = dbg.nav;
  const runHtml = trainingRuns.length
    ? trainingRuns.map(r => `<option value="${escapeAttr(r.name)}">${escapeText(r.name)}</option>`).join("")
    : `<option value="">No runs</option>`;
  dbgSyncSelect("dbgCtxRun", runHtml, nav.run);
  const srcSel = dbgEl("dbgCtxSource");
  if (srcSel) srcSel.value = nav.src;
  const fileHtml = dbg.games.length
    ? dbg.games.map(g => `<option value="${escapeAttr(g.path)}">${escapeText(g.path)}</option>`).join("")
    : `<option value="">No ${escapeText(nav.src)} games</option>`;
  dbgSyncSelect("dbgCtxFile", fileHtml, nav.path);
  const recHtml = dbg.records.length
    ? dbg.records.map(r => `<option value="${r.index}">${escapeText(dbgRecordLabel(r))}</option>`).join("")
    : `<option value="0">g0</option>`;
  dbgSyncSelect("dbgCtxRecord", recHtml, String(nav.rec));
  const ckptOptions = dbg.checkpoints.map(c => `<option value="${escapeAttr(c.name)}">${escapeText(dbgCkptLabel(c))}</option>`).join("");
  dbgSyncSelect("dbgCtxCkptA", ckptOptions || `<option value="">—</option>`, nav.ckptA);
  dbgSyncSelect("dbgCtxCkptB", `<option value="">—</option>` + ckptOptions, nav.ckptB);
  // Support-radius override (shrimp only). The control is hidden for KNOWN
  // dense/hexgt lineages (no support radius); unknown stays visible until the
  // analyze/ckpt_info meta resolves the lineage.
  const radSel = dbgEl("dbgCtxRadius");
  if (radSel) radSel.value = String(nav.radius || 0);
  const radField = dbgEl("dbgCtxRadiusField");
  if (radField) {
    const lin = dbgAttnLineage();
    radField.hidden = Boolean(lin) && lin !== "shrimp";
  }
  document.querySelectorAll("#dbgTabs [data-tab]").forEach(b => b.classList.toggle("active", b.dataset.tab === nav.tab));
  for (const tab of DBG_TABS) {
    const panel = dbgEl(DBG_TAB_IDS[tab]);
    if (panel) panel.classList.toggle("active", tab === nav.tab);
  }
  document.querySelectorAll("#dbgModeBar [data-mode]").forEach(b => b.classList.toggle("active", b.dataset.mode === nav.mode));
  const plane = dbgEl("dbgPlaneSelect");
  if (plane) plane.hidden = nav.mode !== "plane";
  // Attention is shrimp-only: disable the Attn MODE button + the Attention TAB
  // button for known dense/hexgt lineages (re-enable for shrimp/unknown).
  const attnOk = dbgAttnLineageOk();
  const attnModeBtn = document.querySelector('#dbgModeBar [data-mode="attn"]');
  if (attnModeBtn) {
    attnModeBtn.disabled = !attnOk;
    attnModeBtn.classList.toggle("dbg-disabled", !attnOk);
  }
  const attnTabBtn = dbgEl("dbgTabAttnBtn");
  if (attnTabBtn) {
    attnTabBtn.disabled = !attnOk;
    attnTabBtn.classList.toggle("dbg-disabled", !attnOk);
  }
  // Per-cell Q head is v3-shrimp-only: disable the Q MODE button for lineages
  // / checkpoints without a cell_q head (meta.has_cell_q false). A position with
  // no legal cells (terminal) keeps the button enabled — dbgHeatForMode's note
  // covers it without a hard disable.
  const cellqOk = dbgHasCellQ();
  const cellqModeBtn = document.querySelector('#dbgModeBar [data-mode="cellq"]');
  if (cellqModeBtn) {
    cellqModeBtn.disabled = !cellqOk;
    cellqModeBtn.classList.toggle("dbg-disabled", !cellqOk);
  }
}

function dbgRenderCrumb() {
  const el = dbgEl("dbgCtxCrumb");
  if (!el) return;
  const nav = dbg.nav;
  if (!nav.run) {
    el.textContent = "no position loaded";
    return;
  }
  const parts = [nav.run];
  if (nav.path) {
    parts.push(String(nav.path).split("/").pop());
    parts.push(`g${nav.rec}`);
  }
  const total = dbgRecordedTotal();
  if (nav.ply != null) parts.push(`ply ${nav.ply}${total != null ? "/" + total : ""}`);
  if (nav.acts.length) parts.push(`branch +${nav.acts.length}`);
  if (nav.ckptA) parts.push(dbgCkptShort(nav.ckptA) + (nav.ckptB ? ` vs ${dbgCkptShort(nav.ckptB)}` : ""));
  // Active support radius (shrimp only). Prefer the analyze meta (what the
  // worker actually ran at); annotate "(auto)" when no manual override is set.
  const effR = dbgEffectiveRadius();
  if (effR != null) parts.push(`R=${effR}${nav.radius ? "" : " (auto)"}`);
  // Candidate vs engine-legal counts: the R-restriction is obvious when they diverge.
  const a = dbgFreshData("analysis");
  if (a && typeof a.candidate_count === "number" && typeof a.legal_count === "number") {
    parts.push(`cand ${a.candidate_count}/legal ${a.legal_count}`);
  }
  el.textContent = parts.join(" · ");
}

function dbgRenderWorkerDot(state2) {
  const el = dbgEl("dbgCtxWorkerDot");
  if (!el) return;
  el.classList.toggle("ok", state2 === "ok");
  el.classList.toggle("err", state2 === "err");
  const w = dbg.worker;
  el.title = w
    ? `Debug worker: ${w.alive ? "alive" : "idle"} · ${w.cached_results} cached · ${w.mode}`
    : (state2 === "err" ? "Debug worker: error" : "Debug worker: unknown");
}

function dbgRenderPlyRail() {
  const pos = dbg.position;
  const totalRec = dbgRecordedTotal();
  const total = totalRec != null ? totalRec : (pos ? pos.debug.total : 0);
  const ply = dbg.nav.ply != null ? Math.min(dbg.nav.ply, total || dbg.nav.ply) : (pos ? pos.debug.ply : 0);
  const readout = dbgEl("dbgPlyReadout");
  if (readout) readout.textContent = `${ply}/${total}`;
  const slider = dbgEl("dbgPlySlider");
  if (slider) {
    slider.max = String(total);
    slider.value = String(ply);
    slider.disabled = !pos;
  }
  const sweep = dbgSweepData();
  // <=900px the rail is a horizontal 22px strip — rows must be laid out with
  // left/width (a ~150-ply game packed vertically into 22px is ~0.15px/row:
  // invisible and untappable).
  const horiz = Boolean(window.matchMedia && window.matchMedia("(max-width: 900px)").matches);
  // The classic slider is the no-sweep fallback (spec §1.2); once the wrongness
  // strip is colored it takes over as the scrubber — except on mobile, where the
  // thin strip rows are too small to be the only scrubber, so keep the slider.
  const sliderWrap = document.querySelector(".dbg-ply-slider-wrap");
  if (sliderWrap) sliderWrap.hidden = Boolean(sweep) && !horiz;
  const strip = dbgEl("dbgPlyStrip");
  if (!strip) return;
  // Flat neutral rows until a game error sweep colors them (M9): background =
  // policy KL, right 30% tinted by |value error|, blunder/top-1-miss markers.
  const sig = `${total}|${horiz ? "h" : "v"}|${sweep ? `${dbgSweepKey()}#${sweep.version}` : 0}`;
  if (strip.__dbgSig !== sig) {
    strip.__dbgSig = sig;
    let maxKl = 0;
    const blunders = sweep ? new Set(dbgBlunders(sweep)) : null;
    if (sweep) for (const p of sweep.plies) if (p.kl != null) maxKl = Math.max(maxKl, p.kl);
    let html = "";
    // One row per navigable position 0..total (matches nav.ply); sweep rows
    // exist for plies 0..total-1 only (the post-final position is never
    // evaluated), so the bottom row stays neutral.
    for (let i = 0; total > 0 && i <= total; i++) {
      const off = (i / (total + 1)) * 100;
      const row = sweep ? sweep.byPly.get(i) : null;  // sweep row = position AT ply i
      let cls = "dbg-ply-row";
      let style = "";
      let inner = "";
      let title = `ply ${i}`;
      if (row) {
        if (row.kl != null && maxKl > 0) {
          const t = Math.min(1, row.kl / maxKl);
          style = `background:rgba(255,99,90,${(0.05 + 0.65 * t).toFixed(3)});`;
          title += ` · KL ${row.kl.toFixed(3)}`;
        }
        const verr = row.value_err_z != null ? Math.abs(row.value_err_z)
          : (row.value_err_soft != null ? Math.abs(row.value_err_soft) : null);
        if (verr != null) {
          inner = `<span class="dbg-ply-verr" style="background:rgba(245,197,66,${Math.min(0.85, verr / 2).toFixed(3)})"></span>`;
          title += ` · |v err| ${verr.toFixed(2)}`;
        }
        if (blunders && blunders.has(i)) {
          cls += " dbg-ply-blunder";
          title += " · blunder";
        }
        if (row.top1_match === false) {
          cls += " dbg-ply-miss";
          title += " · top-1 miss";
        }
        // Per-cell Q regret (v3): a left-edge magnitude bar, tinted red, so the
        // rail shows where the mover left value on the table even below the
        // blunder threshold. No-op on older checkpoints (regret null).
        if (row.regret != null && row.regret > 0.02) {
          inner += `<span class="dbg-ply-regret" style="background:rgba(255,99,90,${Math.min(0.9, row.regret).toFixed(3)})"></span>`;
          title += ` · regret ${row.regret.toFixed(2)}`;
        }
        if (row.missed_near_win === true) title += " · missed near-win";
      }
      const geom = horiz
        ? `left:${off.toFixed(3)}%;width:${(100 / (total + 1)).toFixed(3)}%;`
        : `top:${off.toFixed(3)}%;height:${(100 / (total + 1)).toFixed(3)}%;`;
      html += `<div class="${cls}" data-ply="${i}" style="${geom}${style}" title="${title}">${inner}</div>`;
    }
    strip.innerHTML = html;
  }
  strip.querySelectorAll(".dbg-ply-cur").forEach(rowEl => rowEl.classList.remove("dbg-ply-cur"));
  const cur = strip.querySelector(`[data-ply="${ply}"]`);
  if (cur) cur.classList.add("dbg-ply-cur");
}

// ---- board heat (spec M3/M5) ---------------------------------------------------

function dbgHeatForMode(mode) {
  // One heat source per render: {values: Map("q,r" -> number), scale, min, max,
  // note}. Modes without their data yet return an empty map + a legend note.
  const out = { mode, values: new Map(), scale: "seq", min: 0, max: 0, note: "" };
  const a = dbgFreshData("analysis");
  const s = dbgFreshData("search");
  const tree = dbgFreshData("tree");
  if (dbg.cmpHeat) {
    // COMPARE Δ overlay: client-computed per-cell prior Δ(A−B), not a server
    // mode — it overrides the base mode while toggled (spec §1.4).
    out.mode = "cmp";
    out.scale = "div";
    const b = dbgAnalysisBFresh();
    if (!a || !b) out.note = "A/B analyses pending";
    else {
      const bPrior = new Map((b.policy || []).map(row => [`${row.q},${row.r}`, row.p]));
      for (const row of a.policy || []) {
        const key = `${row.q},${row.r}`;
        out.values.set(key, row.p - (bPrior.get(key) || 0));
      }
      for (const [key, p] of bPrior) if (!out.values.has(key)) out.values.set(key, -p);
      out.note = "prior Δ(A−B)";
    }
    return dbgHeatFinish(out);
  }
  if (mode === "none") return dbgHeatFinish(out);
  if (mode === "prior") {
    if (!a) out.note = "analyze pending";
    else for (const row of a.policy || []) out.values.set(`${row.q},${row.r}`, row.p);
  } else if (mode === "visits") {
    if (!s) out.note = "no search yet";
    else for (const row of s.visit_policy || []) out.values.set(`${row.q},${row.r}`, row.p);
  } else if (mode === "delta") {
    if (!s) out.note = "no search yet";
    else {
      out.scale = "div";  // promoted (visits > prior) vs demoted, diverging
      const prior = new Map((s.root_prior || []).map(row => [`${row.q},${row.r}`, row.p]));
      for (const row of s.visit_policy || []) {
        const key = `${row.q},${row.r}`;
        out.values.set(key, row.p - (prior.get(key) || 0));
      }
      for (const [key, p] of prior) if (!out.values.has(key)) out.values.set(key, -p);
    }
  } else if (mode === "opp") {
    if (!a) out.note = "analyze pending";
    else if (!a.opp_policy || !a.opp_policy.length) out.note = "no opp head";
    else for (const row of a.opp_policy) out.values.set(`${row.q},${row.r}`, row.p);
  } else if (mode === "target") {
    const rr = dbgRecordRow();
    if (!rr || !rr.found || !rr.row) out.note = "no training row";
    else for (const row of rr.row.policy || []) out.values.set(`${row.q},${row.r}`, row.p);
  } else if (mode === "mismatch") {
    const rr = dbgRecordRow();
    if (!a) out.note = "analyze pending";
    else if (!rr || !rr.found || !rr.row) out.note = "no training row";
    else {
      out.scale = "div";
      const target = new Map((rr.row.policy || []).map(row => [`${row.q},${row.r}`, row.p]));
      for (const row of a.policy || []) {
        const key = `${row.q},${row.r}`;
        out.values.set(key, row.p - (target.get(key) || 0));
      }
      for (const [key, p] of target) if (!out.values.has(key)) out.values.set(key, -p);
    }
  } else if (mode === "childq") {
    if (!tree || !tree.tree) out.note = "run a debug tree";
    else {
      out.scale = "div";
      for (const ch of tree.tree.children || []) out.values.set(`${ch.q},${ch.r}`, ch.qm);
    }
  } else if (mode === "attn") {
    if (!dbgAttnLineageOk()) {
      out.note = "n/a for this lineage";
      return dbgHeatFinish(out);
    }
    const attn = dbgFreshData("attn");
    if (!attn || !attn.found) {
      out.note = attn && attn.reason === "lineage_na"
        ? "n/a for this lineage"
        : (dbg.attnQuery ? "loading attention…" : "click a board cell or a token chip");
      return dbgHeatFinish(out);
    }
    const row = dbgAttnActiveRow(attn);
    if (!row) {
      out.note = "loading attention…";
      return dbgHeatFinish(out);
    }
    out.scale = "seq";  // attention is a [0,1] distribution — sequential scale
    const cells = attn.cells || [];
    for (let j = 0; j < cells.length; j++) {
      const c = cells[j];
      const v = row.attn_over_cells[j];
      if (v) out.values.set(`${c.q},${c.r}`, v);  // same "q,r" Map key contract as every mode
    }
    const headLabel = attn.head == null ? "mean" : (attn.head === "max" ? "max" : "h" + attn.head);
    out.note = dbg.attnQuery && dbg.attnQuery.type === "token"
      ? `T${dbg.attnQuery.id} attends (block ${attn.block} ${headLabel})`
      : `cell ${row.q},${row.r} attends (block ${attn.block} ${headLabel})`;
  } else if (mode === "plane") {
    const entry = dbg.cache.get(dbgCurrentKey());
    const planes = entry ? entry.planes : undefined;
    if (entry && entry.planesPending) out.note = "loading planes…";
    else if (planes === undefined) out.note = "open INPUTS once to load planes";
    else if (planes === null) out.note = "n/a for the shrimp lineage";
    else {
      out.scale = "div";
      const sel = dbgEl("dbgPlaneSelect");
      const idx = sel ? Number(sel.value) || 0 : 0;
      const data = (planes.data && planes.data[idx]) || [];
      const dim = (planes.shape && planes.shape[1]) || 41;
      const half = Math.floor(dim / 2);
      // The dense 41x41 crop is anchored on the ROUNDED MEAN of the placed
      // stones (geometry.crop_center / encoding.rs model1_crop_center), NOT on
      // axial (0,0) — projecting without the center shifts every cell by the
      // centroid offset on any non-origin-centered position.
      const center = dbgPlaneCropCenter(planes);
      const pos = dbg.position;
      const coords = pos ? (pos.legal || []).concat(debugCurrentPlacements()) : [];
      for (const c of coords) {
        const qi = c.q - center.q + half;
        const ri = c.r - center.r + half;
        if (qi < 0 || ri < 0 || qi >= dim || ri >= dim) continue;
        const v = data[ri * dim + qi];
        if (v) out.values.set(`${c.q},${c.r}`, v);
      }
      out.note = (planes.names && planes.names[idx]) || "";
    }
  } else if (mode === "cellq") {
    // Per-cell Q head (v3): decoded scalar Q ∈ [-1,1] from the mover's POV, fed
    // from the analyze payload (cell_q rows {action_id,q,r,qv}, sorted qv desc).
    // Diverging scale: +green good for the mover, −red bad. Older checkpoints
    // (no cell_q head) leave cell_q null — note, no paint.
    if (!a) out.note = "analyze pending";
    else if (!a.cell_q) out.note = "no Q head (older checkpoint)";
    else {
      out.scale = "div";
      for (const r of a.cell_q) out.values.set(`${r.q},${r.r}`, r.qv);
      out.note = "per-cell Q (mover POV)";
    }
  }
  return dbgHeatFinish(out);
}

function dbgRoundHalfEven(x) {
  // Banker's rounding — matches Python round() (geometry.crop_center) and the
  // Rust encoder's ties-to-even (encoding.rs model1_crop_center).
  const f = Math.floor(x);
  if (x - f !== 0.5) return Math.round(x);
  return f % 2 === 0 ? f : f + 1;
}

function dbgPlaneCropCenter(planes) {
  // Crop anchor for projecting board axial coords into the 41x41 input planes.
  // Prefer the server-reported center (input_planes.center = [q, r], additive
  // §3.6 field); fall back to recomputing the stone-centroid client-side so the
  // overlay stays correct against older analyze payloads.
  if (planes && Array.isArray(planes.center) && planes.center.length === 2) {
    return { q: Math.trunc(Number(planes.center[0]) || 0), r: Math.trunc(Number(planes.center[1]) || 0) };
  }
  const stones = debugCurrentPlacements();
  if (!stones.length) return { q: 0, r: 0 };
  let sq = 0;
  let sr = 0;
  for (const s of stones) {
    sq += s.q;
    sr += s.r;
  }
  return { q: dbgRoundHalfEven(sq / stones.length), r: dbgRoundHalfEven(sr / stones.length) };
}

function dbgHeatFinish(out) {
  let min = Infinity;
  let max = -Infinity;
  for (const v of out.values.values()) {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  if (!out.values.size) {
    min = 0;
    max = 0;
  }
  out.min = min;
  out.max = max;
  return out;
}

function dbgHeatNorm(heat, v) {
  // Log toggle (key L) compresses ~two decades so a 0.9-prior board and a
  // near-flat board stay visually distinguishable.
  if (heat.scale === "div") {
    const m = Math.max(Math.abs(heat.min), Math.abs(heat.max)) || 1;
    let t = Math.max(-1, Math.min(1, v / m));
    if (dbg.logScale) t = Math.sign(t) * (Math.log1p(99 * Math.abs(t)) / Math.log1p(99));
    return t;
  }
  const m = heat.max > 0 ? heat.max : 1;
  let t = Math.max(0, Math.min(1, v / m));
  if (dbg.logScale) t = Math.log1p(99 * t) / Math.log1p(99);
  return t;
}

function dbgFormatHeatValue(mode, v) {
  if (mode === "childq") return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
  if (mode === "cellq") return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
  if (mode === "plane") return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
  const pct = v * 100;
  const sign = (mode === "delta" || mode === "mismatch" || mode === "cmp") && v >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}

function dbgRenderLegend(heat) {
  const minEl = dbgEl("dbgLegendMin");
  const maxEl = dbgEl("dbgLegendMax");
  const scaleEl = document.querySelector("#dbgLegend .dbg-legend-scale");
  if (!minEl || !maxEl) return;
  if (!heat || !heat.values.size) {
    minEl.textContent = "—";
    maxEl.textContent = heat && heat.note ? heat.note : "—";
    if (scaleEl) {
      scaleEl.style.background = "";
      scaleEl.title = heat && heat.note ? heat.note : "";
    }
    return;
  }
  minEl.textContent = dbgFormatHeatValue(heat.mode, heat.min);
  maxEl.textContent = dbgFormatHeatValue(heat.mode, heat.max);
  if (scaleEl) {
    scaleEl.title = heat.note || "";  // cmp/plane keep their label even with data
    scaleEl.style.background = heat.scale === "div"
      ? "linear-gradient(90deg, var(--p1), rgba(16,25,36,0.9), var(--green))"
      : "";  // default CSS gradient covers the sequential scale
  }
}

function dbgBuildHudMetrics() {
  // One Map per render feeds the hover HUD — no linear .find in mousemove (M3).
  const metrics = new Map();
  const get = (q, r) => {
    const key = `${q},${r}`;
    let m = metrics.get(key);
    if (!m) {
      m = {};
      metrics.set(key, m);
    }
    return m;
  };
  const a = dbgFreshData("analysis");
  const s = dbgFreshData("search");
  const tree = dbgFreshData("tree");
  if (a) {
    for (const row of a.policy || []) get(row.q, row.r).prior = row.p;
    for (const row of a.opp_policy || []) get(row.q, row.r).opp = row.p;
    for (const row of a.cell_q || []) get(row.q, row.r).cellQ = row.qv;  // per-cell Q head
  }
  if (s) for (const row of s.visit_policy || []) get(row.q, row.r).visits = row.p;
  if (tree && tree.tree) {
    for (const ch of tree.tree.children || []) {
      const m = get(ch.q, ch.r);
      m.childQ = ch.qm;
      m.childN = ch.n;
    }
  }
  const rr = dbgRecordRow();
  if (rr && rr.found && rr.row) for (const row of rr.row.policy || []) get(row.q, row.r).target = row.p;
  const b = dbgAnalysisBFresh();  // split board / cmp overlay HUD column (S5)
  if (b) for (const row of b.policy || []) get(row.q, row.r).priorB = row.p;
  // Attention overlay value per cell (incl. stones) so the hover HUD reads them.
  const attn = dbgFreshData("attn");
  if (attn && attn.found) {
    const arow = dbgAttnActiveRow(attn);
    if (arow) {
      const acells = attn.cells || [];
      for (let j = 0; j < acells.length; j++) {
        const av = arow.attn_over_cells[j];
        if (av != null) get(acells[j].q, acells[j].r).attn = av;
      }
    }
  }
  return metrics;
}

// Modes whose heat map carries values for OCCUPIED cells (every support node has a
// value), not just legal/empty cells — their overlay must render ON stones too.
const DBG_NODE_LEVEL_MODES = new Set(["attn", "plane"]);

function dbgRenderBoard() {
  if (!debugBoardSvg) return;
  if (!dbg.nav.ckptB && (dbg.cmpHeat || dbg.split)) {
    // Compare modes die with checkpoint B (no re-render here — we ARE rendering).
    dbgClearCmpHeat();
    dbg.split = false;
    const wrap = document.querySelector(".dbg-board-wrap");
    if (wrap) wrap.classList.remove("dbg-split");
    const svgB = dbgEl("dbgBoardSvgB");
    if (svgB) svgB.hidden = true;
    const cb = dbgEl("dbgCmpSplit");
    if (cb) cb.checked = false;
  }
  const heat = dbgHeatForMode(dbg.nav.mode);
  dbgRenderLegend(heat);
  dbg.hudMetrics = dbgBuildHudMetrics();
  const pos = dbg.position;
  if (!pos) {
    debugBoardSvg.innerHTML = "";
    dbg.cellIndex = new Map();
    return;
  }

  const cells = new Map();
  const addCell = (q, r, extra) => {
    const key = `${q},${r}`;
    const existing = cells.get(key) || { q, r, key };
    cells.set(key, Object.assign(existing, extra));
  };
  for (const c of (pos.legal || [])) addCell(c.q, c.r, { legal: true });
  const placements = debugCurrentPlacements();
  for (const p of placements) addCell(p.q, p.r, { placement: p });
  // Candidate cells from the CURRENT key's analysis (so heat shows even off the
  // legal set); a stale analysis must not add phantom cells to a new position.
  const freshA = dbgFreshData("analysis");
  if (freshA) for (const row of freshA.policy || []) addCell(row.q, row.r, { candidate: true });
  for (const key of heat.values.keys()) {
    const [q, r] = key.split(",").map(Number);
    addCell(q, r, {});
  }
  dbg.cellIndex = cells;

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const data = [];
  for (const cell of cells.values()) {
    const c = center(cell.q, cell.r);
    minX = Math.min(minX, c.x - HEX * 1.6);
    maxX = Math.max(maxX, c.x + HEX * 1.6);
    minY = Math.min(minY, c.y - HEX * 1.6);
    maxY = Math.max(maxY, c.y + HEX * 1.6);
    data.push(Object.assign(cell, { x: c.x, y: c.y }));
  }
  if (!Number.isFinite(minX)) {
    debugBoardSvg.innerHTML = "";
    return;
  }
  // Pan/zoom-aware viewBox: the bounds only reset the camera when the user
  // hasn't zoomed (viewDirty), exactly like the Match board.
  dbgSyncView({ x: minX, y: minY, width: maxX - minX, height: maxY - minY });
  debugBoardSvg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const ov = dbg.overlays;
  const nav = dbg.nav;
  // Last move = the highest-index stone shown at this ply (derived client-side
  // so the highlight tracks instantly on step); server coords as fallback.
  const lastStone = placements.length ? placements[placements.length - 1] : null;
  const lastCoord = lastStone
    ? { q: lastStone.q, r: lastStone.r }
    : ((pos.debug.last_q != null && pos.debug.last_r != null) ? { q: pos.debug.last_q, r: pos.debug.last_r } : null);
  // Highlight the cell/stone currently chosen as the ATTENTION query so you can
  // see which node you are inspecting (works on stones, which the heat now shows).
  let attnQueryKey = null;
  if (dbg.nav.mode === "attn") {
    const attnD = dbgFreshData("attn");
    if (attnD && attnD.found && attnD.cell_query) attnQueryKey = `${attnD.cell_query.q},${attnD.cell_query.r}`;
  }
  // Q-head mode: ring the Q-best legal cell (cell_q is sorted qv desc, so [0] is
  // the best). The played move keeps its own accent "last" ring (below) — the
  // two rings together show played-vs-best at a glance.
  let qBestKey = null;
  if (dbg.nav.mode === "cellq") {
    const aQ = dbgFreshData("analysis");
    if (aQ && aQ.cell_q && aQ.cell_q.length) qBestKey = `${aQ.cell_q[0].q},${aQ.cell_q[0].r}`;
  }
  data.sort((a, b) => (a.placement ? 1 : 0) - (b.placement ? 1 : 0));
  let html = "";

  // Threat windows (count >= 4) drawn under the cells as colored connectors.
  if (ov.threats && pos.tactics && Array.isArray(pos.tactics.threats)) {
    for (const w of pos.tactics.threats) {
      const pts = (w.cells || []).map(c => center(c.q, c.r));
      if (pts.length < 2) continue;
      const owner = (w.threat_player || w.active_player || "");
      const color = owner.endsWith("1") ? "var(--p1)" : "var(--p0)";
      const d = pts.map((c, i) => `${i ? "L" : "M"}${c.x.toFixed(1)},${c.y.toFixed(1)}`).join("");
      html += `<path d="${d}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" opacity="0.5" pointer-events="none"></path>`;
    }
  }
  for (const h of data) {
    const isStone = Boolean(h.placement);
    const attnStone = isStone && heat.mode === "attn";
    // In attention mode a stone is TRANSPARENT with a player-coloured OUTLINE (P0 blue
    // / P1 red) so you can still see whose stone it is; the attention overlay below
    // fills it with the player colour at opacity == attention, so zero attention shows
    // just the outline (no fill) and tint grows in only where the query attends.
    const stoneTint = attnStone ? (h.placement.player === "player0" ? "var(--p0)" : "var(--p1)") : null;
    // MODEL-BLIND cells: engine-legal (R=8) yet OUTSIDE the model's restricted
    // candidate set (radius < d <= 8) — moves the model was never trained to see.
    // Gate on freshA so a cell is never mislabeled before the current analyze lands;
    // at R=8 candidate==legal so `hidden` is always false (full-legal-set runs unchanged).
    const hidden = Boolean(freshA) && h.legal && !h.candidate && !isStone;
    const fill = isStone ? (attnStone ? "transparent" : playerColor(h.placement.player)) : "#101924";
    const stroke = attnStone ? stoneTint : (isStone ? "#708296" : (hidden ? "var(--warn, #d9a23b)" : "#2c3d50"));
    const opacity = isStone ? "1" : (hidden ? "0.3" : (h.legal || h.candidate ? "0.7" : (ov.legalDim ? "0.18" : "0.45")));
    const dash = hidden ? ` stroke-dasharray="3 2"` : "";
    html += `<path class="dbg-cell${hidden ? " dbg-cell-hidden" : ""}" d="${path(h.x, h.y, HEX - 1)}" fill="${fill}" stroke="${stroke}" stroke-width="${attnStone ? "1.6" : "1"}"${dash} opacity="${opacity}" data-q="${h.q}" data-r="${h.r}"></path>`;
    // Node-level modes (attn/plane) carry values on stones too. Stones get an
    // opacity-scaled FILL tinted by player (P0 blue / P1 red); legal/empty cells
    // get GREEN so the three are distinct at a glance. At very low values a stone
    // shows a thin player-coloured OUTLINE (no fill) so it never fades out.
    // Policy modes never key a stone and keep their accent (seq) / signed (div) colours.
    if ((!isStone || DBG_NODE_LEVEL_MODES.has(heat.mode)) && heat.values.has(h.key)) {
      const t = dbgHeatNorm(heat, heat.values.get(h.key));
      const mag = Math.abs(t);
      if (mag > 0.015) {
        const nodeLevel = DBG_NODE_LEVEL_MODES.has(heat.mode);
        // Node-level (attn): opacity scales straight from 0 (no min floor) so an
        // un-attended cell shows NO colour; policy modes keep the original 0.10 floor.
        const hop = (dbg.opacity * (nodeLevel ? mag : (0.10 + 0.90 * mag))).toFixed(3);
        if (isStone) {
          const tint = h.placement.player === "player0" ? "var(--p0)" : "var(--p1)";
          html += `<path d="${path(h.x, h.y, HEX - 2)}" fill="${tint}" opacity="${hop}" pointer-events="none"></path>`;
        } else {
          const hcolor = nodeLevel ? "var(--green)" : (heat.scale === "div" ? (t >= 0 ? "var(--green)" : "var(--p1)") : "var(--accent)");
          html += `<path d="${path(h.x, h.y, HEX - 2)}" fill="${hcolor}" opacity="${hop}" pointer-events="none"></path>`;
        }
      }
    }
    if (attnQueryKey && h.key === attnQueryKey) {
      html += `<path d="${path(h.x, h.y, HEX - 1)}" fill="none" stroke="#fff" stroke-width="2.4" opacity="0.95" pointer-events="none"></path>`;
    }
    if (qBestKey && h.key === qBestKey) {
      html += `<path d="${path(h.x, h.y, HEX - 1)}" fill="none" stroke="var(--green)" stroke-width="2.4" opacity="0.95" pointer-events="none"></path>`;
    }
    if (ov.last && lastCoord && h.q === lastCoord.q && h.r === lastCoord.r) {
      html += `<path class="dbg-last" d="${path(h.x, h.y, HEX - 0.5)}" fill="none" stroke="var(--accent)" stroke-width="2.2" pointer-events="none"></path>`;
    }
    if (isStone && nav.acts.length && h.placement.index > nav.ply) {
      // Injected what-if stone — dashed accent ring matches its branch chip.
      html += `<path d="${path(h.x, h.y, HEX - 3)}" fill="none" stroke="var(--accent)" stroke-width="1.6" stroke-dasharray="4 3" pointer-events="none"></path>`;
    }
    if (isStone && ov.numbers) {
      html += `<text class="dbg-stone-label" x="${h.x}" y="${h.y}" text-anchor="middle" dominant-baseline="central">${h.placement.index}</text>`;
    }
  }
  // Tree PV (or a clicked node's preview line) as NUMBERED GHOST stones (M10).
  // Neutral accent — future-move ownership would need engine turn logic, so no
  // player-color guessing. style beats the .dbg-ghost-stone pointer-events:none
  // so a ghost click is a no-op (it never reaches the .dbg-cell underneath).
  const treeFresh = dbgFreshData("tree");
  const preview = dbg.treePreview && dbg.treePreview.key === dbgCurrentKey() ? dbg.treePreview.ids : null;
  const ghostLine = dbg.treeGhostsOff ? [] : (preview || (treeFresh && treeFresh.pv) || []);
  ghostLine.forEach((id, i) => {
    const g = dbgUnpackActionId(id);
    const c = center(g.q, g.r);
    html += `<g class="dbg-ghost-stone" style="pointer-events:all">`
      + `<path d="${path(c.x, c.y, HEX - 2.5)}" fill="#101924" stroke="var(--accent)" stroke-width="1.6" opacity="0.85"></path>`
      + `<text class="dbg-stone-label" x="${c.x}" y="${c.y}" text-anchor="middle" dominant-baseline="central" style="fill:var(--accent)">${i + 1}</text>`
      + `</g>`;
  });
  debugBoardSvg.innerHTML = html;
  if (dbg.split) dbgRenderBoardB();  // A/B split keeps the B half in lockstep (S5)
  // Hover + click are DELEGATED (bound once in debugBindEvents) — no per-cell
  // listeners on a 1000+ cell board.
}

function dbgHoverCell(q, r) {
  const hud = dbgEl("debugBoardHud");
  if (!hud) return;
  if (dbg.split) {
    // Shared hover cursor (S5): highlight the cell on BOTH half-boards.
    for (const svg of [debugBoardSvg, dbgEl("dbgBoardSvgB")]) {
      if (!svg) continue;
      svg.querySelectorAll(".dbg-hl").forEach(elx => elx.classList.remove("dbg-hl"));
      const cell = svg.querySelector(`.dbg-cell[data-q="${q}"][data-r="${r}"]`);
      if (cell) cell.classList.add("dbg-hl");
    }
  }
  const m = dbg.hudMetrics.get(`${q},${r}`) || {};
  const pct = v => (v != null ? `${(v * 100).toFixed(1)}%` : "—");
  const childQ = m.childQ != null ? `${m.childQ >= 0 ? "+" : ""}${m.childQ.toFixed(2)} (N=${m.childN})` : "—";
  let line = `${q},${r} · prior ${pct(m.prior)} · visits ${pct(m.visits)} · childQ ${childQ} · target ${pct(m.target)}`;
  if (m.cellQ != null) line += ` · Q ${m.cellQ >= 0 ? "+" : ""}${m.cellQ.toFixed(2)}`;
  if (m.attn != null) line += ` · attn ${pct(m.attn)}`;
  if ((dbg.split || dbg.cmpHeat) && dbg.nav.ckptB) line += ` · B prior ${pct(m.priorB)}`;
  hud.innerHTML = `<div>${escapeText(line)}</div>`;
}

function dbgRenderBranchBar() {
  const bar = dbgEl("dbgBranchBar");
  if (!bar) return;
  const acts = dbg.nav.acts;
  bar.hidden = !acts.length;
  const chips = bar.querySelector(".dbg-branch-chips");
  if (!chips) return;
  if (!acts.length) {
    chips.innerHTML = "";
    return;
  }
  let html = `<span class="dbg-chip dbg-chip-game" title="Recorded-game prefix">game @ ply ${dbg.nav.ply}</span>`;
  acts.forEach((id, i) => {
    const c = dbgUnpackActionId(id);
    html += `<span class="dbg-chip dbg-chip-inj">+${c.q},${c.r}<button class="dbg-chip-x" type="button" data-act-cut="${i}" title="Undo to before this move">✕</button></span>`;
  });
  chips.innerHTML = html;
}

// ---- HEADS panel ---------------------------------------------------------------

function dbgRenderHeads() {
  const panel = dbgEl("dbgTabHeads");
  if (!panel) return;
  const a = dbg.analysis;
  const scalarEl = panel.querySelector(".dbg-value-scalar strong");
  const chip = dbgEl("dbgValueChip");
  const distEl = dbgEl("dbgValueDist");
  const swapBlock = panel.querySelector(".dbg-ownerswap");
  const swapEl = panel.querySelector(".dbg-ownerswap-body");
  const stvEl = dbgEl("dbgStvRows");
  const mlEl = dbgEl("dbgMovesLeft");
  const movesEl = dbgEl("dbgTopMoves");
  const oppEl = dbgEl("dbgOppList");
  if (!a) {
    const note = dbg.loading ? "Evaluating…" : "No analysis yet";
    if (scalarEl) {
      scalarEl.textContent = "—";
      scalarEl.className = "";
    }
    if (chip) {
      chip.textContent = "—";
      chip.className = "dbg-chip";
      chip.title = "";
    }
    if (distEl) distEl.innerHTML = `<div class="dbg-empty-note">${note}</div>`;
    // Default-hidden pre-analysis: the owner-swap probe does not apply to the
    // shrimp lineage (value_swapped is null), so the analyze commit unhides
    // this block only when value_swapped is actually present.
    if (swapBlock) swapBlock.hidden = true;
    if (swapEl) swapEl.innerHTML = `<span class="dbg-muted">—</span>`;
    if (stvEl) stvEl.innerHTML = `<div class="dbg-empty-note">Short-term value horizons</div>`;
    if (mlEl) {
      mlEl.className = "dbg-moves-left dbg-empty-note";
      mlEl.textContent = "Moves-left: —";
    }
    if (movesEl) movesEl.innerHTML = `<div class="dbg-empty-note">${note}</div>`;
    if (oppEl) oppEl.innerHTML = `<div class="dbg-empty-note">Opponent policy top-k</div>`;
    return;
  }
  if (scalarEl) {
    scalarEl.textContent = a.value.toFixed(4);
    scalarEl.className = a.value >= 0 ? "pos" : "neg";
  }
  if (chip) {
    chip.textContent = `${a.value >= 0 ? "+" : ""}${a.value.toFixed(3)} stm`;
    chip.className = `dbg-chip ${a.value >= 0 ? "pos" : "neg"}`;
    chip.title = `to play P${a.current_player} (${a.current_role || ""}) · candidates ${a.candidate_count} · legal ${a.legal_count}`;
  }
  if (distEl) distEl.innerHTML = dbgValueDistHtml(a);
  // B4: the owner-swap P0/P1 probe is off-distribution-confounded (FirstStone
  // parity) and must NOT be presented as optimism — no ok/warn coloring.
  if (swapBlock) swapBlock.hidden = typeof a.value_swapped !== "number";
  if (swapEl) {
    if (typeof a.value_swapped === "number") {
      const sum = a.optimism != null ? a.optimism : (a.value + a.value_swapped);
      swapEl.innerHTML = `
        <div class="info-row"><span class="label">Side to move</span><span class="value">${a.value.toFixed(3)}</span></div>
        <div class="info-row"><span class="label">Owner-swapped</span><span class="value">${a.value_swapped.toFixed(3)}</span></div>
        <div class="info-row" title="v_self + v_swapped; off-distribution (FirstStone parity) — not an optimism metric"><span class="label">Σ</span><span class="value">${sum >= 0 ? "+" : ""}${sum.toFixed(3)}</span></div>`;
    } else {
      swapEl.innerHTML = `<span class="dbg-muted">n/a for this lineage</span>`;
    }
  }
  if (stvEl) {
    const stvRows = Object.keys(a.stvalue || {}).sort((x, y) => Number(x) - Number(y)).map(h => {
      const s = a.stvalue[h].scalar;
      return `<div class="dbg-stv-row"><span class="label">STV+${h}</span><span class="dbg-bar-track"><span class="dbg-bar-fill ${s >= 0 ? "pos" : "neg"}" style="width:${(Math.abs(s) * 50).toFixed(1)}%;${s >= 0 ? "left:50%" : "right:50%"}"></span></span><span class="value">${s.toFixed(3)}</span></div>`;
    }).join("");
    stvEl.innerHTML = stvRows || `<div class="dbg-empty-note">No STV heads</div>`;
  }
  if (mlEl) {
    if (a.moves_left && typeof a.moves_left.scalar === "number") {
      // B1 fix: decode with the REAL cap from analyze meta (512 for dense),
      // not the old hardcoded *80.
      const cap = a.meta && a.meta.moves_left_cap != null ? a.meta.moves_left_cap : 512;
      const remaining = Math.max(0, (a.moves_left.scalar + 1) / 2 * cap);
      const pos = dbg.position;
      const actual = (pos && !dbg.nav.acts.length && !pos.debug.imported && pos.debug.winner)
        ? ` · actual ${pos.debug.total - Math.min(dbg.nav.ply != null ? dbg.nav.ply : pos.debug.ply, pos.debug.total)}`
        : "";
      mlEl.className = "dbg-moves-left";
      mlEl.textContent = `Moves-left: ~${remaining.toFixed(0)} (cap ${cap})${actual}`;
    } else {
      mlEl.className = "dbg-moves-left dbg-empty-note";
      mlEl.textContent = "Moves-left: — (no head in this lineage)";
    }
  }
  if (movesEl) movesEl.innerHTML = dbgTopMovesHtml(a);
  if (oppEl) oppEl.innerHTML = dbgOppListHtml(a);
}

function dbgValueDistHtml(a) {
  const dist = a.value_dist || [];
  const bins = a.value_bins || [];
  if (!dist.length) return `<div class="dbg-empty-note">No value distribution</div>`;
  const maxP = dist.reduce((m, x) => Math.max(m, x), 0) || 1;
  let entropy = 0;
  const bars = dist.map((p, i) => {
    if (p > 0) entropy -= p * Math.log(p);
    const h = Math.max(1, (p / maxP) * 100);
    const c = bins[i] != null ? bins[i] : 0;
    return `<span class="dbg-vbar ${c >= 0 ? "pos" : "neg"}" style="height:${h.toFixed(1)}%" title="v=${c.toFixed(3)} p=${(p * 100).toFixed(1)}%"></span>`;
  }).join("");
  let markers = "";
  const pos = dbg.position;
  if (pos && !dbg.nav.acts.length && pos.debug.winner && pos.current_player) {
    const z = pos.debug.winner === pos.current_player ? 1 : -1;
    markers += `<span class="dbg-vdist-marker z" style="left:${(((z + 1) / 2) * 100).toFixed(1)}%" title="final z (stm) = ${z}"></span>`;
  }
  const rr = dbgRecordRow();  // soft-z target marker once TARGETS rows load (F3)
  if (rr && rr.found && rr.row && typeof rr.row.value_target === "number") {
    markers += `<span class="dbg-vdist-marker soft" style="left:${(((rr.row.value_target + 1) / 2) * 100).toFixed(1)}%" title="recorded soft-z target ${rr.row.value_target.toFixed(3)}"></span>`;
  }
  return `
    <div class="dbg-vdist" aria-label="65-bin value distribution">${bars}${markers}</div>
    <div class="dbg-vdist-axis"><span>loss −1</span><span>0</span><span>+1 win</span></div>
    <div class="dbg-muted">dist entropy ${entropy.toFixed(2)} nats</div>`;
}

function dbgTopMovesHtml(a) {
  // Search/tree columns are key-gated: a stale ply's visits/Q must never line
  // up against this analysis' prior rows (the action ids would misalign).
  const s = dbgFreshData("search");
  const tree = dbgFreshData("tree");
  const visitsById = new Map();
  if (s) for (const row of s.visit_policy || []) visitsById.set(row.action_id, row.p);
  const treeById = new Map();
  let rootQ = null;
  if (tree && tree.tree) {
    rootQ = tree.tree.qm;
    for (const ch of tree.tree.children || []) treeById.set(ch.action_id, ch);
  }
  const recorded = (!dbg.nav.acts.length && dbg.gameActs.length > dbg.nav.ply) ? dbg.gameActs[dbg.nav.ply] : null;
  // Row candidates = top-12 priors UNION search-promoted moves (fresh visit
  // share > 3%) UNION fresh tree root children: a low-prior move the search
  // promoted must get a row (the `under` badge and visits/Q/Δ sorts depend on
  // it). a.policy is the complete legal set, so prior/coords always resolve.
  const priorById = new Map((a.policy || []).map(p => [p.action_id, p]));
  const ids = [];
  const seen = new Set();
  const add = id => {
    if (priorById.has(id) && !seen.has(id)) {
      seen.add(id);
      ids.push(id);
    }
  };
  for (const p of (a.policy || []).slice(0, 12)) add(p.action_id);
  if (s) for (const row of s.visit_policy || []) if (row.p > 0.03) add(row.action_id);
  for (const [id, node] of treeById) if (node.n > 0) add(id);
  const rows = ids.map(id => {
    const p = priorById.get(id);
    const visits = visitsById.has(id) ? visitsById.get(id) : null;
    const node = treeById.get(id);
    return {
      action_id: id,
      q: p.q,
      r: p.r,
      prior: p.p,
      visits,
      qm: node ? node.qm : null,  // Q column fills from the search_tree root layer (F3)
      delta: visits != null ? visits - p.p : null,
    };
  });
  const sortKey = dbg.sortKey;
  const sortVal = row => {
    const v = sortKey === "delta" ? (row.delta != null ? Math.abs(row.delta) : null) : row[sortKey];
    return v == null ? -Infinity : v;
  };
  rows.sort((x, y) => sortVal(y) - sortVal(x));
  rows.length = Math.min(rows.length, 16);
  const arrow = k => (sortKey === k ? " ▾" : "");
  const head = `<div class="dbg-move-row dbg-move-head"><span>#</span><span>cell</span><span class="sortable" data-dbg-sort="prior">prior${arrow("prior")}</span><span class="sortable" data-dbg-sort="visits">visits${arrow("visits")}</span><span class="sortable" data-dbg-sort="qm">Q${arrow("qm")}</span><span class="sortable" data-dbg-sort="delta">Δ${arrow("delta")}</span></div>`;
  const body = rows.map((row, i) => {
    const isBest = s && s.best_action_id === row.action_id;
    const isRec = recorded != null && recorded === row.action_id;
    let badge = "";
    if (row.qm != null && rootQ != null && row.prior > 0.15 && row.qm < rootQ - 0.15) {
      badge = ` <span class="dbg-badge dbg-badge-over" title="high prior met low Q">over</span>`;
    } else if (row.visits != null && row.prior < 0.03 && row.visits > 0.10) {
      badge = ` <span class="dbg-badge dbg-badge-under" title="low prior earned high visits">under</span>`;
    }
    return `<div class="dbg-move-row${isBest ? " dbg-move-best" : ""}"${isRec ? ` title="recorded move"` : ""}><span>${i + 1}</span><span>${row.q},${row.r}${isRec ? " ●" : ""}${badge}</span><span>${(row.prior * 100).toFixed(1)}%</span><span>${row.visits != null ? (row.visits * 100).toFixed(1) + "%" : "—"}</span><span>${row.qm != null ? row.qm.toFixed(2) : "—"}</span><span>${row.delta != null ? (row.delta >= 0 ? "+" : "") + (row.delta * 100).toFixed(1) + "%" : "—"}</span></div>`;
  }).join("");
  return head + body;
}

function dbgRecordedReplyActionId() {
  // The actual recorded reply by the OPPONENT (Hexo turns place two stones, so
  // scan forward for the first recorded action by the other player).
  const nav = dbg.nav;
  if (nav.acts.length || !dbg.position || dbg.position.debug.imported) return null;
  const acts = dbgRecordedActs();
  const stm = dbg.position.current_player;
  if (!acts || !stm) return null;
  for (let j = nav.ply; j < acts.length && j < nav.ply + 4; j++) {
    const pl = dbg.gamePlacements.find(p => p.index === j + 1);
    if (pl && pl.player && pl.player !== stm) return acts[j];
  }
  return null;
}

function dbgOppListHtml(a) {
  const opp = a.opp_policy;
  if (!opp || !opp.length) return `<div class="dbg-empty-note">No opponent-policy head</div>`;
  const reply = dbgRecordedReplyActionId();
  const rows = opp.slice(0, 8).map(row => {
    const hit = reply != null && row.action_id === reply;
    return `<div class="dbg-move-row${hit ? " dbg-move-best" : ""}"${hit ? ` title="actual recorded reply"` : ""}><span></span><span>${row.q},${row.r}${hit ? " ●" : ""}</span><span>${(row.p * 100).toFixed(1)}%</span><span></span><span></span><span></span></div>`;
  }).join("");
  return `<div class="dbg-subhead">Opponent policy</div>${rows}`;
}

// ---- SEARCH / COMPARE / CKPT panels ---------------------------------------------

function dbgRenderSearchPanel() {
  const panel = dbgEl("dbgTabSearch");
  if (!panel) return;
  dbgRenderTree();
  dbgRenderScatter();
  dbgRenderLadder();
  const sum = panel.querySelector(".dbg-search-summary");
  if (!sum) return;
  const s = dbg.search;
  if (!s) {
    sum.className = "dbg-search-summary dbg-empty-note";
    sum.textContent = "No search yet";
    return;
  }
  sum.className = "dbg-search-summary";
  // Key AGREEMENT (not freshness): Δ-vs-raw and the argmax compare are only
  // meaningful when search and analysis describe the same position. Both stale
  // to the SAME key stays visible (the M14 dot already flags it).
  const a = dbg.keys.analysis === dbg.keys.search ? dbg.analysis : null;
  const priorTop = a && a.policy && a.policy[0];
  const agree = priorTop && s.best && priorTop.q === s.best.q && priorTop.r === s.best.r;
  const delta = a ? s.root_value - a.value : null;
  sum.innerHTML = `
    <span>visits <strong>${s.visits}</strong></span>
    <span>root value <strong class="${s.root_value >= 0 ? "pos" : "neg"}">${s.root_value.toFixed(3)}</strong></span>
    <span>best <strong>${s.best ? `${s.best.q},${s.best.r}` : "—"}</strong></span>
    ${priorTop ? `<span class="dbg-muted">prior argmax ${priorTop.q},${priorTop.r} ${agree ? "=" : "≠"}</span>` : ""}
    ${delta != null ? `<span class="dbg-muted">Δ vs raw ${delta >= 0 ? "+" : ""}${delta.toFixed(3)}</span>` : ""}`;
}

function dbgRenderComparePanel() {
  const body = document.querySelector("#dbgTabCompare .dbg-cmp-body");
  if (!body) return;
  if (!dbg.nav.ckptB) {
    body.innerHTML = `<div class="dbg-empty-note">Select checkpoint B in the context strip to compare</div>`;
    return;
  }
  const a = dbg.analysis;
  const b = dbg.analysisB;
  if (!b) {
    // Three-way: in flight / failed / not started — a failed B analyze must
    // surface as an error, not sit on "Evaluating…" forever.
    const entryB0 = dbg.cache.get(dbgCacheKey(dbg.nav, dbg.nav.ckptB));
    if (entryB0 && entryB0.analysisPending) {
      body.innerHTML = `<div class="dbg-empty-note">Evaluating checkpoint B…</div>`;
    } else if (entryB0 && entryB0.analysisError) {
      body.innerHTML = `<div class="dbg-empty-note">B analyze failed: ${escapeText(entryB0.analysisError)} — use ↻ to retry</div>`;
    } else {
      body.innerHTML = `<div class="dbg-empty-note">Checkpoint B not evaluated yet</div>`;
    }
    return;
  }
  const topA = a && a.policy && a.policy[0];
  const topB = b.policy && b.policy[0];
  const dv = a ? b.value - a.value : null;
  const rows = [
    ["Value A", a ? a.value.toFixed(3) : "—"],
    ["Value B", b.value.toFixed(3)],
    ["Δ (B−A)", dv != null ? `${dv >= 0 ? "+" : ""}${dv.toFixed(3)}` : "—"],
  ];
  if (a) {
    for (const h of Object.keys(a.stvalue || {}).sort((x, y) => Number(x) - Number(y))) {
      const sb = b.stvalue && b.stvalue[h];
      if (!sb) continue;
      const d = sb.scalar - a.stvalue[h].scalar;
      rows.push([`STV+${h} Δ`, `${d >= 0 ? "+" : ""}${d.toFixed(3)}`]);
    }
  }
  rows.push(["Top A", topA ? `${topA.q},${topA.r} (${(topA.p * 100).toFixed(0)}%)` : "—"]);
  rows.push(["Top B", topB ? `${topB.q},${topB.r} (${(topB.p * 100).toFixed(0)}%)` : "—"]);
  rows.push(["Agree", topA && topB ? (topA.action_id === topB.action_id ? "yes" : "no") : "—"]);
  // PV divergence: first differing ply of the A vs B search_tree PVs (when both run).
  const treeA = dbgFreshData("tree");
  const entryB = dbg.cache.get(dbgCacheKey(dbg.nav, dbg.nav.ckptB));
  const treeB = entryB && entryB.tree;
  let pvRow;
  if (treeA && treeB) {
    const pa = treeA.pv || [];
    const pb = treeB.pv || [];
    let i = 0;
    while (i < pa.length && i < pb.length && pa[i] === pb[i]) i++;
    if (i >= pa.length && i >= pb.length) pvRow = "identical PVs";
    else {
      const ca = pa[i] != null ? dbgUnpackActionId(pa[i]) : null;
      const cb = pb[i] != null ? dbgUnpackActionId(pb[i]) : null;
      pvRow = `ply +${i + 1}: A ${ca ? `${ca.q},${ca.r}` : "(end)"} vs B ${cb ? `${cb.q},${cb.r}` : "(end)"}`;
    }
  } else {
    pvRow = `needs debug trees (A ${treeA ? "✓" : "—"} · B ${treeB ? "✓" : "—"})`;
  }
  rows.push(["PV divergence", pvRow]);
  body.innerHTML = rows.map(([k, v]) => `<div class="info-row"><span class="label">${k}</span><span class="value">${v}</span></div>`).join("")
    + (treeB ? "" : `<button type="button" id="dbgCmpTreeB" class="dbg-mini-btn" style="margin-top:6px"${dbg.treeBusy ? " disabled" : ""}>Run B debug tree</button>`);
}

function dbgRenderCkptPanel() {
  const el = dbgEl("dbgCkptInfo");
  if (!el) return;
  if (!dbg.nav.ckptA) {
    el.innerHTML = `<div class="dbg-empty-note">No checkpoint selected</div>`;
    return;
  }
  let html = dbgCkptInfoHtml(dbg.nav.ckptA);
  if (dbg.nav.ckptB) {
    html += `<div class="dbg-subhead" style="margin-top:10px">Checkpoint B</div>` + dbgCkptInfoHtml(dbg.nav.ckptB);
  }
  el.innerHTML = html;
}

function dbgCkptInfoHtml(name) {
  const info = dbg.ckptInfo.get(dbgCkptInfoKey(name));
  const ck = dbg.checkpoints.find(c => c.name === name);
  const rows = [["Checkpoint", escapeText(name)]];
  if (ck && ck.epoch != null) rows.push(["Epoch", String(ck.epoch)]);
  if (!info) {
    rows.push(["Provenance", dbg.nav.tab === "ckpt" ? "loading…" : "open this tab to load"]);
  } else if (info.error) {
    rows.push(["Error", escapeText(info.error)]);
  } else {
    const m = info.meta || {};
    if (m.lineage) rows.push(["Lineage", escapeText(m.lineage)]);
    // §3.8 contract: meta.arch is a flattened "key=val, …" display STRING (web.py
    // converts the worker's dict). Parse blocks_type out of it for the Trunk row;
    // stay tolerant of a raw dict in case an older/other backend returns one.
    const archStr = m.arch && typeof m.arch === "object"
      ? Object.keys(m.arch).sort().map(k => `${k}=${m.arch[k]}`).join(", ")
      : (m.arch ? String(m.arch) : "");
    const trunkMatch = archStr.match(/(?:^|,\s*)blocks_type=([^,]+)/);
    if (trunkMatch) rows.push(["Trunk", escapeText(trunkMatch[1].trim())]);
    if (archStr) rows.push(["Arch", escapeText(archStr)]);
    if (m.rl_epoch != null) rows.push(["RL epoch", String(m.rl_epoch)]);
    if (m.step != null) rows.push(["Step", String(m.step)]);
    if (m.graft) rows.push(["Graft", m.graft === "pre" ? "pre (≤e6, expanded)" : "post (≥e7)"]);
    if (m.stv_horizons && m.stv_horizons.length) rows.push(["STV horizons", m.stv_horizons.join(", ")]);
    rows.push(["Moves-left", m.has_moves_left ? `yes (cap ${m.moves_left_cap != null ? m.moves_left_cap : "?"})` : "no"]);
    if (m.expanded_value) rows.push(["Expanded value", "yes"]);
    if (m.expanded_stv && m.expanded_stv.length) rows.push(["Expanded STV", `${m.expanded_stv.length} heads`]);
    if (m.zeroed_feature_cols && m.zeroed_feature_cols.length) rows.push(["Zeroed cols", m.zeroed_feature_cols.join(", ")]);
    if (m.candidate_radius != null) rows.push(["Cand. radius", String(m.candidate_radius)]);
    // Active SHRIMP support radius the worker process actually ran this info op
    // at (OWNER C meta.support_radius); null for non-shrimp lineages.
    if (m.support_radius != null) rows.push(["Support radius", String(m.support_radius)]);
    if (m.param_count != null) rows.push(["Params", `${(m.param_count / 1e6).toFixed(2)}M`]);
    if (info.size != null) rows.push(["File", formatBytes(info.size)]);
    if (info.mtime != null) rows.push(["Modified", new Date(info.mtime * 1000).toLocaleString()]);
    if (m.load_warnings && m.load_warnings.length) rows.push(["Warnings", escapeText(m.load_warnings.join("; "))]);
  }
  return rows.map(([k, v]) => `<div class="info-row"><span class="label">${k}</span><span class="value">${v}</span></div>`).join("");
}

// ---- bottom dock ------------------------------------------------------------------

function dbgSetDockTab(tab) {
  dbg.dockTab = tab;
  document.querySelectorAll("#dbgDockTabs [data-dock-tab]").forEach(b => b.classList.toggle("active", b.dataset.dockTab === tab));
  const chart = dbgEl("dbgDockChart");
  if (chart) chart.classList.toggle("active", tab === "trajectory");
  const sweep = dbgEl("dbgCkptSweep");
  if (sweep) sweep.classList.toggle("active", tab === "ckptsweep");
  const regret = dbgEl("dbgRegretList");
  if (regret) regret.classList.toggle("active", tab === "regret");
  if (tab === "regret") dbgRenderRegretList();
}

function dbgRenderDockChart() {
  // Trajectory dock (M9/S11): sweep-fed when a Game Error Sweep exists for the
  // current (game, ckptA) — value_p0 + KL second axis + mismatch ticks +
  // clickable blunder markers; otherwise the legacy /trajectory plot.
  const el = dbgEl("dbgDockChart");
  if (!el) return;
  const nav = dbg.nav;
  const sweep = dbgSweepData();
  const t = dbg.trajectoryKey === `${nav.run}|${nav.path}|${nav.rec}|${nav.ckptA}` ? dbg.trajectory : null;
  const reeval = sweep ? sweep.plies : ((t && t.reeval) || []);
  if (!reeval.length) {
    el.innerHTML = `<div class="dbg-empty-note">Sweep the game (or Plot) to fill the per-ply value / KL chart</div>`;
    el.__dbgChart = null;
    return;
  }
  const W = 1000, H = 220, padL = 36, padR = 34, padT = 14, padB = 22;
  const total = (sweep && sweep.total) || (t && t.total) || (reeval[reeval.length - 1].ply + 1) || 1;
  const x = ply => padL + (W - padL - padR) * (total ? ply / total : 0);
  const y = v => padT + (H - padT - padB) * (1 - (v + 1) / 2);  // v in [-1,1] -> top=+1
  const linePath = (pts, key) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.ply).toFixed(1)},${y(p[key]).toFixed(1)}`).join("");

  let html = "";
  html += `<line x1="${padL}" y1="${y(0)}" x2="${W - padR}" y2="${y(0)}" stroke="#2c3d50" stroke-width="1"></line>`;
  html += `<text x="4" y="${y(1) + 4}" fill="#5a6b7a" font-size="11">+1</text>`;
  html += `<text x="6" y="${y(0) + 4}" fill="#5a6b7a" font-size="11">0</text>`;
  html += `<text x="4" y="${y(-1) + 2}" fill="#5a6b7a" font-size="11">−1</text>`;
  if (nav.ply != null) {
    const px = x(Math.min(nav.ply, total));
    html += `<line x1="${px}" y1="${padT}" x2="${px}" y2="${H - padB}" stroke="var(--accent)" stroke-width="1" opacity="0.4" stroke-dasharray="3 3"></line>`;
  }
  if (t && t.recorded && t.recorded.length) {
    html += `<path d="${linePath(t.recorded, "root_value_p0")}" fill="none" stroke="var(--yellow)" stroke-width="1.6" opacity="0.85"></path>`;
  }
  // Policy-KL series on a second (right) axis — sweep only (S11; hidden without).
  let maxKl = 0;
  const klPts = sweep ? sweep.plies.filter(p => p.kl != null) : [];
  for (const p of klPts) maxKl = Math.max(maxKl, p.kl);
  if (klPts.length && maxKl > 0) {
    const yk = v => padT + (H - padT - padB) * (1 - v / maxKl);
    html += `<path d="${klPts.map((p, i) => `${i ? "L" : "M"}${x(p.ply).toFixed(1)},${yk(p.kl).toFixed(1)}`).join("")}" fill="none" stroke="#b07ce8" stroke-width="1.4" opacity="0.8"></path>`;
    html += `<text x="${W - padR + 3}" y="${padT + 8}" fill="#b07ce8" font-size="10">KL ${maxKl.toFixed(2)}</text>`;
    html += `<text x="${W - padR + 3}" y="${H - padB}" fill="#b07ce8" font-size="10">0</text>`;
  }
  // Per-ply Q-regret series on the right axis (v3 cell_q; hidden without it).
  // regret >= 0; scaled against the game's max regret so the curve fills the
  // panel. The KL series already owns the right-axis label slot, so the regret
  // label sits just below it.
  let maxRegret = 0;
  const regretPts = sweep ? sweep.plies.filter(p => p.regret != null) : [];
  const hasRegret = regretPts.length > 0;
  for (const p of regretPts) maxRegret = Math.max(maxRegret, p.regret);
  if (hasRegret && maxRegret > 0) {
    const yr = v => padT + (H - padT - padB) * (1 - v / maxRegret);
    html += `<path d="${regretPts.map((p, i) => `${i ? "L" : "M"}${x(p.ply).toFixed(1)},${yr(p.regret).toFixed(1)}`).join("")}" fill="none" stroke="var(--p1)" stroke-width="1.4" opacity="0.7"></path>`;
    html += `<text x="${W - padR + 3}" y="${padT + 20}" fill="var(--p1)" font-size="10">reg ${maxRegret.toFixed(2)}</text>`;
  }
  html += `<path d="${linePath(reeval, "value_p0")}" fill="none" stroke="var(--accent)" stroke-width="2"></path>`;
  if (sweep) {
    for (const p of sweep.plies) {
      if (p.top1_match === false) {
        html += `<line x1="${x(p.ply).toFixed(1)}" y1="${H - padB}" x2="${x(p.ply).toFixed(1)}" y2="${H - padB - 6}" stroke="var(--yellow)" stroke-width="1.5" opacity="0.8"></line>`;
      }
      // Diagnostic 2: policy↔Q disagreement — Q-best cell differs from the played
      // move, gated on regret>=0.10 so pure Q-ties don't tick. Distinct color
      // (purple) from the yellow top-1-miss tick, drawn just above it.
      if (p.q_best_match === false && p.regret != null && p.regret >= 0.10) {
        html += `<line x1="${x(p.ply).toFixed(1)}" y1="${H - padB - 7}" x2="${x(p.ply).toFixed(1)}" y2="${H - padB - 13}" stroke="#b07ce8" stroke-width="1.5" opacity="0.85"></line>`;
      }
      // Diagnostic 3: missed near-win — a legal Q≈+1 cell the mover passed up.
      if (p.missed_near_win === true) {
        const sx = x(p.ply);
        const sy = y(p.value_p0) - 11;
        html += `<text data-dbg-jump="${p.ply}" x="${sx.toFixed(1)}" y="${sy.toFixed(1)}" text-anchor="middle" font-size="13" fill="var(--green)" style="cursor:pointer"><title>missed near-win ply ${p.ply} — click to jump</title>★</text>`;
      }
    }
    for (const bp of dbgBlunders(sweep)) {
      const row = sweep.byPly.get(bp);
      if (!row) continue;
      html += `<circle data-dbg-jump="${bp}" cx="${x(bp).toFixed(1)}" cy="${y(row.value_p0).toFixed(1)}" r="5" fill="var(--p1)" opacity="0.85" style="cursor:pointer"><title>blunder ply ${bp}${row.regret != null ? ` · regret ${row.regret.toFixed(2)}` : ""} — click to jump</title></circle>`;
    }
  }
  html += `<line class="dbg-dock-cross" x1="0" y1="${padT}" x2="0" y2="${H - padB}" stroke="#dfe9f3" stroke-width="1" opacity="0" pointer-events="none"></line>`;
  const legend = sweep
    ? (hasRegret
        ? "value_p0 accent · regret red (right) · KL purple · blunder=regret ply red dot · P↔Q tick purple · ★missed win · click to jump"
        : "value_p0 accent · KL purple (right axis) · top-1 miss ticks yellow · blunders red (click to jump)")
    : (t && t.stride > 1 ? `(every ${t.stride} plies · re-eval accent, recorded yellow)` : "re-eval accent · recorded yellow");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${html}</svg>`
    + `<div class="dbg-dock-readout dbg-muted">${legend}</div>`;
  const byPly = new Map();
  for (const p of reeval) byPly.set(p.ply, p);
  el.__dbgChart = { W, padL, padR, total, byPly, legend };
}

function dbgRegretAvailable() {
  // The regret tab/list (and missed-win badge) only make sense once a sweep of
  // the current (game, ckptA) carries per-ply regret (v3 cell_q). Older
  // checkpoints leave every regret null — the tab/panel stay hidden.
  const sweep = dbgSweepData();
  return Boolean(sweep && sweep.plies.some(p => p.regret != null));
}

function dbgRenderRegretList() {
  // Diagnostic 1: worst plies by Q-regret — client-side over the sweep rows
  // (filter regret!=null, sort desc, top 8). Each row jumps to that ply via the
  // global [data-dbg-jump] handler. Also surfaces a missed-near-win count badge.
  const el = dbgEl("dbgRegretList");
  if (!el) return;
  // Toggle the dock tab button + this panel's visibility on data availability.
  const avail = dbgRegretAvailable();
  const tabBtn = document.querySelector('#dbgDockTabs [data-dock-tab="regret"]');
  if (tabBtn) tabBtn.hidden = !avail;
  if (!avail) {
    // A non-regret lineage that had the tab open falls back to the trajectory
    // dock so the user never stares at an empty regret panel.
    if (dbg.dockTab === "regret") dbgSetDockTab("trajectory");
    el.innerHTML = "";
    return;
  }
  const sweep = dbgSweepData();
  const nav = dbg.nav;
  const actsOk = dbg.gameActsKey === `${nav.run}|${nav.path}|${nav.rec}` && dbg.gameActs.length;
  const coordStr = aid => {
    if (aid == null) return "—";
    const c = dbgUnpackActionId(aid);
    return `${c.q},${c.r}`;
  };
  const missed = sweep.plies.filter(p => p.missed_near_win === true);
  const rows = sweep.plies
    .filter(p => p.regret != null)
    .sort((a, b) => b.regret - a.regret)
    .slice(0, 8);
  let html = `<div class="dbg-subhead">Worst plies by Q-regret`
    + (missed.length ? ` <span class="dbg-chip dbg-chip-game" title="legal Q≈+1 cells the mover passed up">${missed.length} missed win${missed.length > 1 ? "s" : ""}</span>` : "")
    + `</div>`;
  if (!rows.length || rows[0].regret <= 0) {
    html += `<div class="dbg-empty-note">No regret recorded yet — sweep the game.</div>`;
    el.innerHTML = html;
    return;
  }
  for (const p of rows) {
    const playedAid = actsOk && dbg.gameActs.length > p.ply ? dbg.gameActs[p.ply] : null;
    const playedCoord = coordStr(playedAid);
    const bestCoord = coordStr(p.q_best_aid);
    const pq = p.played_q != null ? `${p.played_q >= 0 ? "+" : ""}${p.played_q.toFixed(2)}` : "—";
    const bq = p.best_q != null ? `${p.best_q >= 0 ? "+" : ""}${p.best_q.toFixed(2)}` : "—";
    const flags = (p.missed_near_win === true ? " ★" : "")
      + (p.q_best_match === false ? " ⚐" : "");
    html += `<div class="dbg-target-row" data-dbg-jump="${p.ply}" style="cursor:pointer" title="jump to ply ${p.ply}">`
      + `<span>ply ${p.ply}${flags}</span>`
      + `<span class="dbg-ply-regret-val">regret ${p.regret.toFixed(2)}</span>`
      + `<span class="dbg-muted">Q ${pq}→${bq}</span>`
      + `<span class="dbg-muted">${playedCoord} → ${bestCoord}</span>`
      + `</div>`;
  }
  el.innerHTML = html;
}

// ---- stale dots (spec M14) -----------------------------------------------------

function dbgSetStale(panelId, stale) {
  const panel = dbgEl(panelId);
  if (!panel) return;
  const dot = panel.querySelector(".dbg-stale");
  if (dot) dot.hidden = !stale;
}

function dbgUpdateStaleDots() {
  const nav = dbg.nav;
  const ready = nav.run && nav.path && nav.ply != null;
  const cur = ready ? dbgCurrentKey() : "";
  const curB = ready && nav.ckptB ? dbgCacheKey(nav, nav.ckptB) : "";
  const posStale = Boolean(dbg.position) && dbg.keys.position !== cur;
  dbgSetStale("dbgTabHeads", Boolean(dbg.analysis) && (dbg.keys.analysis !== cur || posStale));
  dbgSetStale("dbgTabSearch", (Boolean(dbg.search) && dbg.keys.search !== cur) || (Boolean(dbg.tree) && dbg.keys.tree !== cur));
  dbgSetStale("dbgTabTargets", posStale);  // targets render per-key, so only a stale position misleads
  dbgSetStale("dbgTabCompare", Boolean(nav.ckptB) && Boolean(dbg.analysisB) && dbg.keys.analysisB !== curB);
  dbgSetStale("dbgTabInputs", posStale);
  dbgSetStale("dbgTabCkpt", false);     // checkpoint provenance is position-independent
  dbgSetStale("dbgTabAttn", Boolean(dbg.attn) && dbg.keys.attn !== dbgAttnSlotKey());
}

function dbgRefreshPanel(panelId) {
  if (panelId === "dbgTabHeads") dbgAnalyzeNow(true);
  else if (panelId === "dbgTabSearch") dbgRunSearch();
  else if (panelId === "dbgTabCompare") {
    if (!dbg.nav.ckptB) {
      dbgStub("Select checkpoint B in the context strip first.");
      return;
    }
    const entryB = dbgCacheEntry(dbgCacheKey(dbg.nav, dbg.nav.ckptB));
    delete entryB.analysis;
    delete entryB.analysisError;
    dbgEnsureAnalysisB();
  } else if (panelId === "dbgTabCkpt") dbgEnsureCkptInfo(true);
  else if (panelId === "dbgTabTargets") dbgEnsureRecordRow(true);
  else if (panelId === "dbgTabInputs") dbgEnsureInputs(true);
  else if (panelId === "dbgTabAttn") dbgEnsureAttn(true);
}

// ---- board view (pan/zoom, ported from the Match board on debug-local state) ----

function dbgApplyView() {
  if (dbg.view && debugBoardSvg) {
    debugBoardSvg.setAttribute("viewBox", `${dbg.view.x} ${dbg.view.y} ${dbg.view.width} ${dbg.view.height}`);
    const svgB = dbg.split ? dbgEl("dbgBoardSvgB") : null;
    if (svgB) svgB.setAttribute("viewBox", `${dbg.view.x} ${dbg.view.y} ${dbg.view.width} ${dbg.view.height}`);
  }
}

function dbgSyncView(base) {
  dbg.baseView = base;
  if (!dbg.view || !dbg.viewDirty) dbg.view = { ...base };
  dbgApplyView();
}

function dbgZoom(factor, anchor) {
  if (!dbg.view) return;
  const base = dbg.baseView || dbg.view;
  const nextWidth = clamp(dbg.view.width * factor, base.width * 0.14, base.width * 4.2);
  const scale = nextWidth / dbg.view.width;
  const nextHeight = dbg.view.height * scale;
  const point = anchor || {
    x: dbg.view.x + dbg.view.width / 2,
    y: dbg.view.y + dbg.view.height / 2,
  };
  dbg.view = {
    x: point.x - (point.x - dbg.view.x) * scale,
    y: point.y - (point.y - dbg.view.y) * scale,
    width: nextWidth,
    height: nextHeight,
  };
  dbg.viewDirty = true;
  dbgApplyView();
}

function dbgZoomAtCenter(factor) {
  if (!dbg.view) return;
  dbgZoom(factor, {
    x: dbg.view.x + dbg.view.width / 2,
    y: dbg.view.y + dbg.view.height / 2,
  });
}

function dbgZoomReset() {
  dbg.viewDirty = false;
  if (dbg.baseView) dbg.view = { ...dbg.baseView };
  dbgApplyView();
}

function dbgClientToBoard(clientX, clientY) {
  const matrix = debugBoardSvg && debugBoardSvg.getScreenCTM();
  if (!matrix || !dbg.view) {
    return {
      x: dbg.view ? dbg.view.x + dbg.view.width / 2 : 0,
      y: dbg.view ? dbg.view.y + dbg.view.height / 2 : 0,
    };
  }
  const point = debugBoardSvg.createSVGPoint();
  point.x = clientX;
  point.y = clientY;
  return point.matrixTransform(matrix.inverse());
}

// ---- navigation helpers -------------------------------------------------------

function dbgStepPly(delta) {
  // If a press does nothing, say WHY (the usual cause is the selected game
  // failing to load) instead of silently returning.
  if (!dbg.position && dbg.nav.ply == null) {
    debugSetStatus("No position loaded — pick a game with completed records.", "error");
    return;
  }
  const total = dbgRecordedTotal();
  const cur = dbg.nav.ply != null ? dbg.nav.ply : (total || 0);
  dbgGotoPly(cur + delta);
}

function dbgGotoPly(ply, { replace = false } = {}) {
  const total = dbgRecordedTotal();
  const clamped = Math.max(0, total != null ? Math.min(ply, total) : ply);
  if (clamped === dbg.nav.ply && !dbg.nav.acts.length) return;
  // Explicit ply navigation returns to the RECORDED game: re-basing the injected
  // tail onto a different prefix is never meaningful (the injected cells may be
  // occupied/illegal there) — chips, U and G are the way back out of a branch.
  dbgNavigate({ ply: clamped, acts: [] }, { replace });
}

function dbgStepRecord(delta) {
  if (!dbg.records.length) return;
  const next = Math.max(0, Math.min(dbg.nav.rec + delta, dbg.records.length - 1));
  if (next !== dbg.nav.rec) dbgNavigate({ rec: next, ply: null, acts: [] });
}

function dbgStepFile(delta) {
  if (!dbg.games.length) return;
  const idx = dbg.games.findIndex(g => g.path === dbg.nav.path);
  const next = Math.max(0, Math.min((idx === -1 ? 0 : idx) + delta, dbg.games.length - 1));
  const g = dbg.games[next];
  if (g && g.path !== dbg.nav.path) dbgNavigate({ path: g.path, rec: 0, ply: null, acts: [] });
}

function dbgInjectCell(q, r) {
  // WHAT-IF INJECTION (spec M4): clicking an empty legal cell appends its action
  // to the branch prefix; the server replays + re-analyzes the new position.
  if (dbg.suppressClick) return;
  const cell = dbg.cellIndex.get(`${q},${r}`);
  if (!cell || cell.placement || !dbg.position) return;
  if (!cell.legal) {
    debugSetStatus(`${q},${r} is not a legal cell here.`, "error");
    return;
  }
  diagTap(`inject ${q},${r}`);
  dbgNavigate({ acts: dbg.nav.acts.concat([dbgPackActionId(q, r)]) });
}

function dbgClearToggles() {
  dbg.overlays = { threats: false, numbers: false, last: false, legalDim: false };
  for (const [id, key] of [["dbgTglThreats", "threats"], ["dbgTglNumbers", "numbers"], ["dbgTglLast", "last"], ["dbgTglLegalDim", "legalDim"]]) {
    const el = dbgEl(id);
    if (el) el.checked = dbg.overlays[key];
  }
  // Repaint here: the follow-up dbgNavigate is a same-hash no-paint when the
  // base mode is unchanged (Shift+digit solo / 0 with the mode already set).
  dbgRenderBoard();
}

function dbgToggleLog() {
  dbg.logScale = !dbg.logScale;
  const el = dbgEl("dbgLegendLog");
  if (el) el.checked = dbg.logScale;
  dbgRenderBoard();
}

function dbgToggleCollapse(id) {
  const el = dbgEl(id);
  if (el) el.classList.toggle("dbg-collapsed");
}

// ---- keyboard layer (spec §1.7 / M13) -------------------------------------------

function dbgHandleKey(e) {
  if (activeScreen !== "debug") return;
  const help = dbgEl("dbgHelp");
  const palette = dbgEl("dbgPalette");
  if (e.key === "Escape") {
    if (palette && !palette.hidden) {
      palette.hidden = true;
      e.preventDefault();
      return;
    }
    if (help && !help.hidden) {
      help.hidden = true;
      e.preventDefault();
      return;
    }
    return;
  }
  if (palette && !palette.hidden) return;  // palette owns the keyboard while open
  const ae = document.activeElement;
  if (ae && /^(INPUT|SELECT|TEXTAREA)$/.test(ae.tagName)) return;  // M13: arrows edit the input, never the ply
  const nav = dbg.nav;
  const k = e.key;
  // Never hijack browser/system chords (Ctrl+C copy, Ctrl+A select-all, …);
  // Ctrl+K is the one modified shortcut the spec assigns (command palette).
  if ((e.ctrlKey || e.metaKey || e.altKey) && k.toLowerCase() !== "k") return;
  let handled = true;
  if (k === "?") {
    if (help) help.hidden = !help.hidden;
  } else if (k.toLowerCase() === "k" && !e.altKey && !e.metaKey) {
    dbgOpenPalette();
  } else if (e.code && e.code.startsWith("Digit") && !e.ctrlKey && !e.altKey && !e.metaKey) {
    const num = Number(e.code.slice(5));
    if (num === 0) {
      dbgClearToggles();  // 0 = mode none + clear additive toggles
      dbgClearCmpHeat();
      dbgNavigate({ mode: "none" }, { replace: true });
    } else if (num >= 1 && num <= DBG_MODE_ORDER.length) {
      if (e.shiftKey) dbgClearToggles();  // Shift+digit = solo
      dbgClearCmpHeat();
      dbgNavigate({ mode: DBG_MODE_ORDER[num - 1] }, { replace: true });
    } else {
      handled = false;
    }
  } else if (k === "ArrowLeft" || k === "ArrowRight") {
    if (e.shiftKey) dbgStepBlunder(k === "ArrowLeft" ? -1 : 1);
    else dbgStepPly(k === "ArrowLeft" ? -1 : 1);
  } else if (k === "Home") {
    dbgGotoPly(0);
  } else if (k === "End") {
    dbgGotoPly(1e9);
  } else if (k === "[") {
    dbgStepRecord(-1);
  } else if (k === "]") {
    dbgStepRecord(1);
  } else if (k === "{") {
    dbgStepFile(-1);
  } else if (k === "}") {
    dbgStepFile(1);
  } else if (k === "a" || k === "A") {
    dbgAnalyzeNow(true);
  } else if (k === "s" || k === "S") {
    dbgRunSearch();
  } else if (k === "t" || k === "T") {
    // §1.7 "T  run / toggle debug tree": with a fresh tree for this key, toggle
    // its PV ghosts; otherwise run (dbgRunTree stubs while one is in flight).
    if (!dbg.treeBusy && dbgFreshData("tree")) {
      dbg.treeGhostsOff = !dbg.treeGhostsOff;
      dbg.treePreview = null;
      dbgRenderBoard();
    } else {
      dbgRunTree();
    }
  } else if (k === "l" || k === "L") {
    dbgToggleLog();
  } else if (k === "u" || k === "U") {
    if (nav.acts.length) dbgNavigate({ acts: nav.acts.slice(0, -1) });
  } else if (k === "g" || k === "G") {
    if (nav.acts.length) dbgNavigate({ acts: [] });
  } else if (k === "p" || k === "P") {
    dbgAddPin();
  } else if (k === "c" || k === "C") {
    dbgNavigate({ tab: "compare" }, { replace: true });
  } else if (k === "i" || k === "I") {
    dbgNavigate({ tab: "inputs" }, { replace: true });
  } else {
    handled = false;
  }
  if (handled) e.preventDefault();
}

// ---- events --------------------------------------------------------------------

function debugBindEvents() {
  const root = dbgEl("debugScreen");
  if (!root || root.__dbgDelegated) return;  // bind once on the stable ancestor
  root.__dbgDelegated = true;

  // The ply strip renders row geometry for the current orientation (vertical
  // rail on desktop, horizontal 22px strip <=900px) — rebuild it on breakpoint
  // crossings so the inline top/height vs left/width styles stay correct.
  if (window.matchMedia) {
    const mq = window.matchMedia("(max-width: 900px)");
    const onOrientation = () => {
      try {
        dbgRenderPlyRail();
      } catch (e) {
        reportError("dbgRenderPlyRail: " + (e && (e.stack || e.message) || e));
      }
    };
    if (mq.addEventListener) mq.addEventListener("change", onOrientation);
    else if (mq.addListener) mq.addListener(onOrientation);
  }

  // BUTTONS via event DELEGATION on #debugScreen (which is never rebuilt), plus a
  // touchend fallback. Per-button click listeners failed on the owner's phone:
  // controls trapped in a fixed-height overflow:hidden panel micro-scrolled under
  // the finger, so the browser classified the tap as a scroll and never promoted
  // touchend -> click. Delegation + an explicit touchend handler guarantees the
  // action fires on tap; everything routes through reportError so a throw is
  // visible on-device.
  const ACTIONS = {
    dbgPlyFirst: { fn: () => dbgGotoPly(0), tap: "|< first" },
    dbgPlyPrev: { fn: () => dbgStepPly(-1), tap: "< prev" },
    dbgPlyNext: { fn: () => dbgStepPly(1), tap: "> next" },
    dbgPlyLast: { fn: () => dbgGotoPly(1e9), tap: ">| last" },
    dbgPlySweepBtn: { fn: () => dbgRunSweep(), tap: "sweep" },
    dbgSweepBtn: { fn: () => dbgRunSweep(), tap: "sweep" },
    dbgTrajBtn: { fn: () => dbgPlotTrajectory(), tap: "plot" },
    dbgSearchRun: { fn: () => dbgRunSearch(), tap: "search" },
    dbgTreeRun: { fn: () => dbgRunTree(), tap: "tree" },
    dbgLadderRun: { fn: () => dbgRunLadder(), tap: "ladder" },
    dbgCmpTreeB: { fn: () => dbgRunTree({ checkpoint: dbg.nav.ckptB }), tap: "tree B" },
    dbgCkptSweepRun: { fn: () => dbgRunCkptSweep(), tap: "ckpt sweep" },
    dbgCtxRefresh: { fn: () => dbgRefreshSources(), tap: "refresh" },
    dbgCtxCollapse: { fn: () => dbgToggleCollapse("dbgCtx"), tap: "ctx collapse" },
    dbgDockToggle: { fn: () => dbgToggleCollapse("dbgDock"), tap: "dock collapse" },
    dbgZoomIn: { fn: () => dbgZoomAtCenter(0.82), tap: "zoom in" },
    dbgZoomOut: { fn: () => dbgZoomAtCenter(1.22), tap: "zoom out" },
    dbgZoomReset: { fn: () => dbgZoomReset(), tap: "zoom reset" },
    dbgBranchReturn: { fn: () => dbgNavigate({ acts: [] }), tap: "return to game" },
    dbgPinAdd: { fn: () => dbgAddPin(), tap: "pin" },
    dbgPinExport: { fn: () => dbgExportPins(), tap: "pin export" },
    dbgPaletteBtn: { fn: () => dbgOpenPalette(), tap: "palette" },
    dbgHelpBtn: { fn: () => { const h = dbgEl("dbgHelp"); if (h) h.hidden = !h.hidden; }, tap: "help" },
    dbgBlunderPrev: { fn: () => dbgStepBlunder(-1), tap: "blunder prev" },
    dbgBlunderNext: { fn: () => dbgStepBlunder(1), tap: "blunder next" },
    dbgCmpHeat: { fn: () => dbgToggleCmpHeat(), tap: "cmp heat" },
    dbgCkptSweepStop: {
      fn: () => {
        if (dbg.ckptSweep && dbg.ckptSweep.running) dbg.ckptSweep.abort = true;
        else dbgStub("No checkpoint sweep running.");
      },
      tap: "ckpt sweep stop",
    },
  };
  const fire = ev => {
    const t = ev.target;
    if (!t || !t.closest) return;
    const idBtn = t.closest("button[id]");
    const act = idBtn && ACTIONS[idBtn.id];
    if (act) {
      ev.preventDefault();
      diagTap(act.tap);
      try {
        act.fn();
      } catch (e) {
        reportError("dbg " + idBtn.id + ": " + (e && (e.stack || e.message) || e));
      }
      return;
    }
    const chipX = t.closest(".dbg-chip-x");
    if (chipX) {
      ev.preventDefault();
      diagTap("undo chip");
      dbgNavigate({ acts: dbg.nav.acts.slice(0, Number(chipX.dataset.actCut) || 0) });
      return;
    }
    const treeStep = t.closest(".dbg-tree-step");
    if (treeStep) {
      ev.preventDefault();
      diagTap("tree step-into");
      // M11: re-base via injection — each line move becomes a branch chip.
      dbgNavigate({ acts: dbg.nav.acts.concat(dbgTreePathIds(treeStep.dataset.treePath)) });
      return;
    }
    const treeExpand = t.closest(".dbg-tree-expand");
    if (treeExpand) {
      ev.preventDefault();
      diagTap("tree expand");
      const p = treeExpand.dataset.treePath;
      dbgRunTree({ rootActions: dbgTreePathIds(p), graftPath: p });
      return;
    }
    const treeRow = t.closest(".dbg-tree-row");
    if (treeRow) {
      ev.preventDefault();
      dbgTreeRowClick(treeRow.dataset.treePath);
      return;
    }
    const jump = t.closest("[data-dbg-jump]");
    if (jump) {
      ev.preventDefault();
      diagTap("jump ply " + jump.dataset.dbgJump);
      dbgGotoPly(Number(jump.dataset.dbgJump));
      return;
    }
    const pinX = t.closest(".dbg-pin-x");
    if (pinX) {
      ev.preventDefault();
      dbgRemovePin(Number(pinX.dataset.pin));
      return;
    }
    const pinNote = t.closest(".dbg-pin-note");
    if (pinNote) {
      ev.preventDefault();
      dbgNotePin(Number(pinNote.dataset.pin));
      return;
    }
    const pinChip = t.closest(".dbg-pin-chip");
    if (pinChip) {
      ev.preventDefault();
      dbgRestorePin(Number(pinChip.dataset.pin));
      return;
    }
    const journalRow = t.closest("[data-journal]");
    if (journalRow) {
      ev.preventDefault();
      const e2 = dbg.journal[Number(journalRow.dataset.journal)];
      if (e2 && e2.nav) dbgNavigate(Object.assign({}, e2.nav, { acts: ((e2.nav && e2.nav.acts) || []).slice() }));
      return;
    }
    const palItem = t.closest("[data-pal]");
    if (palItem) {
      ev.preventDefault();
      dbgPaletteExec(Number(palItem.dataset.pal));
      return;
    }
    const refresh = t.closest(".dbg-panel-refresh");
    if (refresh) {
      ev.preventDefault();
      const panel = refresh.closest(".dbg-tab-panel");
      if (panel) dbgRefreshPanel(panel.id);
      return;
    }
    const modeBtn = t.closest("#dbgModeBar [data-mode]");
    if (modeBtn) {
      ev.preventDefault();
      diagTap("mode " + modeBtn.dataset.mode);
      // Parity with the 0 key (§1.7): None = mode none + clear additive toggles.
      if (modeBtn.dataset.mode === "none") dbgClearToggles();
      dbgClearCmpHeat();  // picking a base mode dismisses the cmp Δ overlay
      dbgNavigate({ mode: modeBtn.dataset.mode }, { replace: true });
      return;
    }
    const tabBtn = t.closest("#dbgTabs [data-tab]");
    if (tabBtn) {
      ev.preventDefault();
      diagTap("tab " + tabBtn.dataset.tab);
      dbgNavigate({ tab: tabBtn.dataset.tab }, { replace: true });
      return;
    }
    const dockTab = t.closest("#dbgDockTabs [data-dock-tab]");
    if (dockTab) {
      ev.preventDefault();
      dbgSetDockTab(dockTab.dataset.dockTab);
      return;
    }
    const journalClear = t.closest(".dbg-journal-clear");
    if (journalClear) {
      ev.preventDefault();
      diagTap("journal clear");
      dbgClearJournal();
      return;
    }
    const sortBtn = t.closest("[data-dbg-sort]");
    if (sortBtn) {
      ev.preventDefault();
      dbg.sortKey = sortBtn.dataset.dbgSort;
      dbgRenderHeads();
      return;
    }
    const plyRow = t.closest(".dbg-ply-row");
    if (plyRow) {
      ev.preventDefault();
      diagTap("strip ply " + plyRow.dataset.ply);
      dbgGotoPly(Number(plyRow.dataset.ply));
      return;
    }
    const attnChip = t.closest("#dbgAttnTokens [data-attn-token]");
    if (attnChip) {
      ev.preventDefault();
      const id = Number(attnChip.dataset.attnToken) || 0;
      diagTap("attn token T" + id);
      // Switching tokens reuses the cached payload (all 8 rows ride in it) — no refetch.
      dbgNavigate({ attnq: `token:${id}` }, { replace: true });
      return;
    }
    let cellEl = t.closest(".dbg-cell");
    if (!cellEl && ev.clientX != null) {
      // Desktop (Firefox esp.): the pan/zoom pointer-capture retargets the click
      // off the cell onto the board container, so ev.target is not a .dbg-cell.
      // Recover the cell by hit-testing the pointer position (heat overlays are
      // pointer-events:none, so this lands on the underlying .dbg-cell).
      const hit = document.elementFromPoint(ev.clientX, ev.clientY);
      cellEl = hit && hit.closest ? hit.closest(".dbg-cell") : null;
    }
    if (cellEl) {
      ev.preventDefault();
      const cq = Number(cellEl.dataset.q);
      const cr = Number(cellEl.dataset.r);
      if (dbg.nav.mode === "attn") {
        // ATTN MODE: a board click sets the cell query (no what-if injection).
        const id = dbgPackActionId(cq, cr);
        diagTap("attn cell " + cq + "," + cr);
        dbgNavigate({ attnq: `cell:${id}` }, { replace: true });
        return;
      }
      dbgInjectCell(cq, cr);
      return;
    }
    if (t.classList && t.classList.contains("dbg-overlay")) {
      ev.preventDefault();
      t.hidden = true;  // click outside the card closes help/palette
    }
  };
  let lastTouchFire = 0;
  root.addEventListener("touchend", ev => {
    lastTouchFire = Date.now();
    fire(ev);
  }, { passive: false });
  root.addEventListener("click", ev => {
    if (Date.now() - lastTouchFire < 700) return;  // touchend already handled this tap
    fire(ev);
  });

  // Hover HUD: ONE delegated mousemove on the SVG + the per-render metrics Map.
  if (debugBoardSvg) {
    debugBoardSvg.addEventListener("mousemove", ev => {
      const cellEl = ev.target && ev.target.closest && ev.target.closest(".dbg-cell");
      if (cellEl) dbgHoverCell(Number(cellEl.dataset.q), Number(cellEl.dataset.r));
    });
    debugBoardSvg.addEventListener("mouseleave", () => {
      const hud = dbgEl("debugBoardHud");
      if (hud) hud.innerHTML = `<div>Hover a cell</div>`;
    });
  }

  // Pan / pinch / wheel on the debug board area (debug-local state only).
  const area = debugBoardSvg && debugBoardSvg.closest(".board-area");
  if (area && !area.__dbgViewBound) {
    area.__dbgViewBound = true;
    const pointers = new Map();
    let drag = null;
    let pinch = null;
    const beginPan = ev => {
      const rect = debugBoardSvg.getBoundingClientRect();
      drag = {
        pointerId: ev.pointerId,
        clientX: ev.clientX,
        clientY: ev.clientY,
        scaleX: dbg.view.width / Math.max(1, rect.width),
        scaleY: dbg.view.height / Math.max(1, rect.height),
        view: { ...dbg.view },
        moved: false,
      };
    };
    area.addEventListener("wheel", ev => {
      if (!dbg.view || ev.target.closest(".board-view-controls")) return;
      ev.preventDefault();
      dbgZoom(ev.deltaY < 0 ? 0.88 : 1.14, dbgClientToBoard(ev.clientX, ev.clientY));
    }, { passive: false });
    area.addEventListener("pointerdown", ev => {
      if (!dbg.view || (ev.pointerType === "mouse" && ev.button !== 0)) return;
      if (ev.target.closest(".board-view-controls")) return;
      pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      area.setPointerCapture(ev.pointerId);
      if (pointers.size >= 2) {
        drag = null;  // a second finger upgrades the gesture to pinch-zoom
        const pts = [...pointers.values()].slice(0, 2);
        const mid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
        pinch = {
          rect: debugBoardSvg.getBoundingClientRect(),
          startDist: Math.max(1, Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y)),
          startView: { ...dbg.view },
          anchorBoard: dbgClientToBoard(mid.x, mid.y),
        };
        dbg.suppressClick = true;
      } else {
        beginPan(ev);
      }
    });
    area.addEventListener("pointermove", ev => {
      if (!pointers.has(ev.pointerId)) return;
      ev.preventDefault();
      pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
      if (pinch && pointers.size >= 2) {
        const pts = [...pointers.values()].slice(0, 2);
        const dist = Math.max(1, Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y));
        const mid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
        const base = dbg.baseView || pinch.startView;
        const nextWidth = clamp(pinch.startView.width * (pinch.startDist / dist), base.width * 0.14, base.width * 4.2);
        const nextHeight = pinch.startView.height * (nextWidth / pinch.startView.width);
        const rect = pinch.rect;
        dbg.view = {
          width: nextWidth,
          height: nextHeight,
          x: pinch.anchorBoard.x - (mid.x - rect.left) * (nextWidth / Math.max(1, rect.width)),
          y: pinch.anchorBoard.y - (mid.y - rect.top) * (nextHeight / Math.max(1, rect.height)),
        };
        dbg.viewDirty = true;
        dbgApplyView();
      } else if (drag && ev.pointerId === drag.pointerId) {
        const dx = (ev.clientX - drag.clientX) * drag.scaleX;
        const dy = (ev.clientY - drag.clientY) * drag.scaleY;
        if (Math.hypot(ev.clientX - drag.clientX, ev.clientY - drag.clientY) > 4) drag.moved = true;
        dbg.view = { ...drag.view, x: drag.view.x - dx, y: drag.view.y - dy };
        dbg.viewDirty = true;
        dbgApplyView();
      }
    }, { passive: false });
    const endPointer = ev => {
      if (!pointers.has(ev.pointerId)) return;
      pointers.delete(ev.pointerId);
      if (area.hasPointerCapture(ev.pointerId)) area.releasePointerCapture(ev.pointerId);
      const moved = (drag && drag.moved) || Boolean(pinch);
      if (pointers.size < 2) pinch = null;
      if (pointers.size === 1) {
        // Dropped from pinch to a single finger — resume panning from it.
        const [pointerId, point] = [...pointers.entries()][0];
        beginPan({ pointerId, clientX: point.x, clientY: point.y });
        drag.moved = true;
      } else if (pointers.size === 0) {
        drag = null;
        if (moved) {
          dbg.suppressClick = true;
          window.setTimeout(() => {
            dbg.suppressClick = false;
          }, 80);
        } else {
          dbg.suppressClick = false;
        }
      }
    };
    area.addEventListener("pointerup", endPointer);
    area.addEventListener("pointercancel", endPointer);
  }

  // Selects / sliders / checkboxes keep direct change/input listeners.
  const on = (id, ev, fn) => {
    const el = dbgEl(id);
    if (el) el.addEventListener(ev, fn);
  };
  on("dbgCtxRun", "change", e => dbgNavigate({ run: e.target.value, path: "", rec: 0, ply: null, acts: [], ckptA: "", ckptB: "" }));
  on("dbgCtxSource", "change", e => dbgNavigate({ src: e.target.value, path: "", rec: 0, ply: null, acts: [] }));
  on("dbgCtxFile", "change", e => dbgNavigate({ path: e.target.value, rec: 0, ply: null, acts: [] }));
  on("dbgCtxRecord", "change", e => dbgNavigate({ rec: Number(e.target.value) || 0, ply: null, acts: [] }));
  on("dbgCtxCkptA", "change", e => dbgNavigate({ ckptA: e.target.value }));
  on("dbgCtxCkptB", "change", e => dbgNavigate({ ckptB: e.target.value }));
  // Support-radius override: 0=Auto (backend detects/defaults), 4|8 force a worker
  // respawn at that radius. Pushed (not replace) so Back undoes the toggle.
  on("dbgCtxRadius", "change", e => dbgNavigate({ radius: Number(e.target.value) || 0 }));
  // Slider: first input of a drag PUSHES (so Back returns to the pre-drag ply),
  // the rest REPLACE (no history spam); change ends the drag.
  let sliderDragging = false;
  on("dbgPlySlider", "input", e => {
    diagTap("slide " + e.target.value);
    dbgGotoPly(Number(e.target.value), { replace: sliderDragging });
    sliderDragging = true;
  });
  on("dbgPlySlider", "change", e => {
    sliderDragging = false;
    dbgGotoPly(Number(e.target.value), { replace: true });
  });
  on("dbgTglThreats", "change", e => {
    dbg.overlays.threats = e.target.checked;
    dbgRenderBoard();
  });
  on("dbgTglNumbers", "change", e => {
    dbg.overlays.numbers = e.target.checked;
    dbgRenderBoard();
  });
  on("dbgTglLast", "change", e => {
    dbg.overlays.last = e.target.checked;
    dbgRenderBoard();
  });
  on("dbgTglLegalDim", "change", e => {
    dbg.overlays.legalDim = e.target.checked;
    dbgRenderBoard();
  });
  on("dbgLegendLog", "change", e => {
    dbg.logScale = e.target.checked;
    dbgRenderBoard();
  });
  on("dbgOpacity", "input", e => {
    dbg.opacity = Number(e.target.value) || 0.9;
    dbgRenderBoard();
  });
  on("dbgPlaneSelect", "change", () => dbgRenderBoard());
  // Attention block/head selectors — a new block/head changes the cache signature
  // and refetches (the worker recomputes that block's/head's row).
  on("dbgAttnBlockSel", "change", e => {
    const maxBlk = Math.max(0, (dbg.attnNumBlocks || 3) - 1);
    const blk = Math.max(0, Math.min(Number(e.target.value) || 0, maxBlk));
    dbgNavigate({ attnblk: blk }, { replace: true });
  });
  on("dbgAttnHeadSel", "change", e => {
    const v = e.target.value;
    const maxHead = Math.max(0, (dbg.attnNumHeads || 4) - 1);
    const next = v === "" ? "" : (v === "max" ? "max" : Math.max(0, Math.min(Number(v) || 0, maxHead)));
    dbgNavigate({ attnhead: next }, { replace: true });
  });
  on("dbgCmpSplit", "change", e => dbgToggleSplit(e.target.checked));
  on("dbgPinImport", "change", e => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    dbgImportPins(file);
  });

  // Command palette (S1): the input owns the keyboard while open.
  on("dbgPaletteInput", "input", e => dbgRenderPaletteList(e.target.value));
  on("dbgPaletteInput", "keydown", e => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const n = (dbg.paletteShown || []).length;
      if (n) {
        dbg.paletteIdx = (dbg.paletteIdx + (e.key === "ArrowDown" ? 1 : n - 1)) % n;
        dbgRenderPaletteList(e.target.value);
      }
    } else if (e.key === "Enter") {
      e.preventDefault();
      dbgPaletteExec(dbg.paletteIdx);
    }
  });

  // Split-board (S5) shared hover: the B half reuses the same delegated handler.
  const svgB = dbgEl("dbgBoardSvgB");
  if (svgB) {
    svgB.addEventListener("mousemove", ev => {
      const cellEl = ev.target && ev.target.closest && ev.target.closest(".dbg-cell");
      if (cellEl) dbgHoverCell(Number(cellEl.dataset.q), Number(cellEl.dataset.r));
    });
  }

  // Trajectory dock hover crosshair (S11): per-render chart state on the
  // container; attribute updates only — no re-render in mousemove.
  const dock = dbgEl("dbgDockChart");
  if (dock) {
    dock.addEventListener("mousemove", ev => {
      const st = dock.__dbgChart;
      const svg = dock.querySelector("svg");
      if (!st || !svg) return;
      const rect = svg.getBoundingClientRect();
      const xv = (ev.clientX - rect.left) / Math.max(1, rect.width) * st.W;
      const ply = Math.max(0, Math.min(st.total, Math.round((xv - st.padL) / Math.max(1, st.W - st.padL - st.padR) * st.total)));
      const cross = svg.querySelector(".dbg-dock-cross");
      if (cross) {
        cross.setAttribute("x1", xv.toFixed(1));
        cross.setAttribute("x2", xv.toFixed(1));
        cross.setAttribute("opacity", "0.35");
      }
      const out = dock.querySelector(".dbg-dock-readout");
      const row = st.byPly.get(ply);
      if (out) {
        out.textContent = row
          ? `ply ${ply} · v_p0 ${row.value_p0 != null ? row.value_p0.toFixed(3) : "—"}`
            + `${row.kl != null ? ` · KL ${row.kl.toFixed(3)}` : ""}`
            + `${row.top1_match != null ? ` · top1 ${row.top1_match ? "✓" : "✗"}` : ""}`
            + `${row.value_err_z != null ? ` · err_z ${row.value_err_z >= 0 ? "+" : ""}${row.value_err_z.toFixed(2)}` : ""}`
            + `${row.regret != null ? ` · regret ${row.regret.toFixed(2)} (Q ${row.played_q != null ? row.played_q.toFixed(2) : "—"}→${row.best_q != null ? row.best_q.toFixed(2) : "—"})` : ""}`
          : `ply ${ply}`;
      }
    });
    dock.addEventListener("mouseleave", () => {
      const cross = dock.querySelector(".dbg-dock-cross");
      if (cross) cross.setAttribute("opacity", "0");
      const out = dock.querySelector(".dbg-dock-readout");
      if (out && dock.__dbgChart) out.textContent = dock.__dbgChart.legend || "";
    });
  }

  document.addEventListener("keydown", dbgHandleKey);
}

// ===========================================================================
// Screen entry + app bootstrap (enterDebugScreen, the debug hashchange
// listener, init).
// ===========================================================================

async function enterDebugScreen() {
  showVersionTag();
  if (!dbg.inited) {
    dbg.inited = true;
    // Bind handlers in their OWN try so a later failure can never leave the nav
    // buttons unwired, surfaced to the on-screen banner (the real-phone failure
    // mode we otherwise can't see).
    try {
      debugBindEvents();
    } catch (e) {
      reportError("debugBindEvents: " + (e && (e.stack || e.message) || e));
    }
  }
  if (dbg.pendingDeepLink) {
    // navigateScreen() is mid-flight: when it is about to WRITE `#debug` the
    // resulting hashchange routes through dbgApplyHash (which consumes the deep
    // link); when the hash already IS `#debug...` no event will come — apply now.
    if (String(window.location.hash || "").startsWith("#debug")) dbgApplyHash();
    return;
  }
  dbgApplyHash();
}

// Debug-internal hash navigation (ply steps, injections, checkpoint flips). The
// global listener above (setScreen) handles screen routing; this one applies the
// nav state. dbgApplyHash dedupes, so the double delivery is harmless.
window.addEventListener("hashchange", () => {
  if (screenFromHash() === "debug") dbgApplyHash();
});

async function init() {
  setScreen(activeScreen, { preserveHash: true });
  await Promise.allSettled([loadAdapters(), loadState(), loadTrainingRuns()]);
  render();
}

init();
