/* app.js — showcase frontend wiring: play controller, analysis controller,
 * games feed, hash routing (#game/{id}), toasts and network banner.
 */

/* The ?v= pins on these imports (and on the asset links in index.html) exist
 * so one page version always runs one bundle version: html, js, and css are
 * cached independently by browsers/CDN, and a skewed pair (new index.html +
 * stale app.js, or the reverse) is exactly the "buttons do nothing" class of
 * field bug. Bump ALL of them together whenever any of the five files
 * changes incompatibly. */
import * as api from "./api.js?v=10";
import { buildCkptList, groupCheckpoints, latestCheckpoint, defaultCheckpoint } from "./checkpoints.js?v=10";
import { createBoard, findWin, key } from "./board.js?v=6";

"use strict";

const $ = id => document.getElementById(id);
const sleep = ms => new Promise(res => setTimeout(res, ms));
const clamp = (lo, hi, v) => Math.max(lo, Math.min(hi, v));

/* overlay tone families — deliberately NOT the stone colors: paler and
 * hue-shifted so a tinted empty cell can never be mistaken for a stone */
const H0 = "#9fd0ff", H1 = "#ffb4aa";
const H0R = "#d7ebff", H1R = "#ffddd6";

const fmtV = v => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);
const fmtCell = (q, r) => q + "," + r;
const turnOf = ply => (ply <= 1 ? 1 : Math.floor(ply / 2) + 1);

/* Engine turn structure: ply 1 is color 0's forced opening single, then each
 * color places two. Color of the player who made 1-based ply i: */
const plyColor = i => (i === 1 ? 0 : (Math.floor((i - 2) / 2) % 2 === 0 ? 1 : 0));
/* Color to move after p plies: */
const moverAfter = p => plyColor(p + 1);

function fmtAgo(iso) {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 90) return "now";
  if (s < 5400) return Math.round(s / 60) + "m";
  if (s < 129600) return Math.round(s / 3600) + "h";
  return Math.round(s / 86400) + "d";
}

// ---- toasts + network banner -------------------------------------------------

const toastWrap = $("toastWrap");
const recentToasts = new Map();

function toast(msg, isErr = false) {
  const last = recentToasts.get(msg);
  if (last && Date.now() - last < 4000) return; // dedupe bursts
  recentToasts.set(msg, Date.now());
  const el = document.createElement("div");
  el.className = "toast" + (isErr ? " err" : "");
  el.textContent = msg;
  toastWrap.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

api.onNetChange(down => { $("netBanner").hidden = !down; });
window.addEventListener("offline", () => { $("netBanner").hidden = false; });
window.addEventListener("online", () => { $("netBanner").hidden = true; });

function apiFail(e, fallback) {
  if (e.network) return; // banner covers it
  if (e.status === 429) toast(e.message || "the server is at capacity — try again in a minute", true);
  else toast(fallback || e.message || "request failed", true);
}

// ---- tabs / views --------------------------------------------------------------

function activateView(name) {
  document.querySelectorAll(".tab").forEach(x => {
    const on = x.dataset.view === name;
    x.classList.toggle("active", on);
    x.setAttribute("aria-selected", on);
  });
  document.querySelectorAll(".view").forEach(v => {
    v.classList.toggle("active", v.id === "view-" + name);
  });
  if (name !== "analysis") stopAuto(); // leaving analysis pauses autoplay
  if (name === "analysis") refreshFeed();
}
document.querySelectorAll(".tab").forEach(t => {
  t.addEventListener("click", () => {
    activateView(t.dataset.view);
    if (t.dataset.view !== "analysis" && location.hash) {
      history.replaceState(null, "", location.pathname + location.search);
    }
  });
});

// ================================ PLAY =========================================

const play = {
  id: null, snap: null, moves: [], staged: null,
  label: "", sims: 0, polling: false, lastPlaceT: 0, creating: false,
  // bot-turn elapsed ticker: thinkSince is when the current bot_thinking spell
  // began (ms epoch), thinkTimer the 1s interval that repaints the elapsed read
  thinkSince: 0, thinkTimer: null,
};

const statusEl = $("playStatus"), statusText = $("statusText");
const resignBtn = $("resignBtn"), analyzeBtn = $("analyzeBtn");
const nickForm = $("nickForm"), nickInput = $("nickInput"), nickMsg = $("nickMsg");
const placeChip = $("placeChip"), playTag = $("playTag"), cursorPos = $("cursorPos");
const thinkNote = $("thinkNote");
const playAlert = $("playAlert"), playAlertMsg = $("playAlertMsg");

/* Past this many seconds of one bot turn, add a calm "warming up" note — a
 * cold first move JITs for ~30s, so a long wait is expected, not broken. */
const WARMUP_NOTE_AFTER_S = 8;

const playBoard = createBoard($("playBoard"), {
  onCellClick: onPlayCell,
  onHover: (q, r) => { cursorPos.textContent = q === null ? "—" : fmtCell(q, r); },
  ghostAllowed: () => !!play.snap && play.snap.status === "your_turn",
  onPanStart: () => playBoard.hideHoverGhost(),
  canReset: t => {
    // never reset off the back of a placement double-click / stage-commit tap
    if (Date.now() - play.lastPlaceT < 500) return false;
    if (t && t.tagName === "polygon" && t.classList &&
        t.classList.contains("cell") && !t.classList.contains("occ")) return false;
    return true;
  },
});

function setStatus(txt, cls) {
  statusEl.className = "status " + cls;
  statusText.textContent = txt;
}

// ---- bot-turn feedback: elapsed timer + warm-up note ------------------------

/* Repaint the "thinking" status with elapsed seconds, and past the threshold
 * show a reassuring warm-up note. Called every second while bot_thinking. */
function paintThinking() {
  const s = Math.max(0, Math.round((Date.now() - play.thinkSince) / 1000));
  // keep the status class ("think") the caller set; only refresh the text
  statusText.textContent = `${play.label} thinking… ${s}s`;
  if (s >= WARMUP_NOTE_AFTER_S) {
    thinkNote.textContent = "taking a little longer than usual (warming up)";
    thinkNote.hidden = false;
  }
}

/* Enter the thinking spell: start (or keep) the once-per-second ticker. Idempotent
 * across repeated bot_thinking snapshots so the elapsed clock is continuous. */
function startThinking() {
  if (!play.thinkSince) play.thinkSince = Date.now();
  paintThinking();
  if (play.thinkTimer) return;
  play.thinkTimer = setInterval(paintThinking, 1000);
}

/* Leave the thinking spell: stop the ticker and clear the warm-up note. Safe to
 * call when not thinking (move landed, game over, new game, view switch). */
function stopThinking() {
  if (play.thinkTimer) { clearInterval(play.thinkTimer); play.thinkTimer = null; }
  play.thinkSince = 0;
  thinkNote.hidden = true;
  thinkNote.textContent = "";
}

// ---- recoverable bot-failure notice -----------------------------------------

/* Show a friendly, honest heads-up for a bot-backend hiccup with a way forward.
 * The button label already reads "New game" in the abandoned/finished state, so
 * the notice just explains what happened and points at it. Cleared on any fresh
 * snapshot ingest so a recovered game never carries a stale alert. */
function showPlayAlert(msg) {
  playAlertMsg.textContent = msg;
  playAlert.hidden = false;
}
function clearPlayAlert() {
  playAlert.hidden = true;
  playAlertMsg.textContent = "";
}

function clearStage() {
  play.staged = null;
  playBoard.clearStage();
  placeChip.classList.remove("show");
}

function stageAt(q, r) {
  play.staged = { q, r };
  playBoard.stage(q, r);
  placeChip.classList.add("show");
}

const legalHas = (snap, q, r) =>
  snap.legal && snap.legal.some(c => c.q === q && c.r === r);

/* Per-ply move history from a snapshot. The server's `stones` is in placement
 * order (the client contract), so it IS the move list — verified against
 * last_move; if a build ever sends them sorted instead, fall back to
 * diff-merging new stones, pinning the within-turn order by last_move. */
function orderedStones(snap) {
  const st = snap.stones;
  if (!Array.isArray(st) || !st.length) return st || [];
  const lm = snap.last_move;
  if (lm && (st[st.length - 1].q !== lm.q || st[st.length - 1].r !== lm.r)) return null;
  return st;
}

function ingestMoves(snap) {
  const ordered = orderedStones(snap);
  if (ordered) {
    play.moves = ordered.map(s => ({ q: s.q, r: s.r, color: s.color }));
    return;
  }
  const seen = new Set(play.moves.map(m => key(m.q, m.r)));
  const added = (snap.stones || []).filter(s => !seen.has(key(s.q, s.r)));
  if (!added.length) return;
  const lm = snap.last_move;
  added.sort((a, b) =>
    (lm && a.q === lm.q && a.r === lm.r ? 1 : 0) - (lm && b.q === lm.q && b.r === lm.r ? 1 : 0));
  for (const s of added) play.moves.push({ q: s.q, r: s.r, color: s.color });
}

/* meta-grid "you" cell: blue/red per the RESOLVED human_color (the server
 * echoes it in every snapshot; with "random" it is decided at create time) */
function showYouColor(hc) {
  const el = $("youColor");
  if (!el) return; // cached pre-play-as index.html: cell has no id yet
  el.textContent = hc === 1 ? "red" : "blue";
  el.className = "n " + (hc === 1 ? "is-p1" : "is-p0");
}

function ingestPlay(snap) {
  play.snap = snap;
  ingestMoves(snap);
  if (Number.isInteger(snap.human_color)) showYouColor(snap.human_color);
  const finished = snap.status === "finished";
  const term = finished && snap.result ? snap.result.termination : null;
  const winCells = term === "six_in_line"
    ? (Array.isArray(snap.winning_line) && snap.winning_line.length
        ? snap.winning_line : findWin(play.moves))
    : null;
  playBoard.setStones(play.moves, winCells);
  playBoard.setLegal(snap.status === "your_turn" ? snap.legal : null);
  $("plyCount").textContent = snap.ply;
  $("turnCount").textContent = turnOf(snap.ply);

  if (play.staged && play.moves.some(m => m.q === play.staged.q && m.r === play.staged.r)) {
    clearStage(); // the bot took the staged cell
  }
  // a fresh snapshot means the last request landed — drop any recoverable
  // failure notice; the branches below re-raise it if this snapshot is itself
  // a failure state.
  clearPlayAlert();
  if (snap.status === "your_turn") {
    stopThinking();
    const left = snap.stones_left_this_turn;
    setStatus(`your move · ${left} stone${left === 1 ? "" : "s"}`, "you");
  } else if (snap.status === "bot_thinking") {
    setStatus(`${play.label} thinking…`, "think");
    startThinking(); // owns the status text from here: elapsed + warm-up note
    clearStage();
  } else if (finished) {
    stopThinking();
    clearStage();
    const winner = snap.result ? snap.result.winner : null;
    if (term === "six_in_line") {
      setStatus(winner === snap.human_color
        ? "you win · six-in-a-row"
        : `${play.label} wins · six-in-a-row`, "over");
    } else if (term === "resign") {
      setStatus(`resigned · ${play.label} wins`, "over");
    } else if (term === "timeout") {
      // idle sweep: the game sat untouched too long. Not a bot fault.
      setStatus("game timed out · idle too long", "over");
    } else {
      // termination null == the bot turn couldn't complete (backend hiccup /
      // fail-fast timeout). Say so honestly and point at the New game button,
      // which `resignBtn` already shows in this finished state.
      setStatus("game couldn't finish", "warn");
      showPlayAlert("the bot backend hiccuped and couldn't finish its move — no fault of yours. Start a new game to keep playing.");
    }
  }
  resignBtn.textContent = finished || !play.id ? "New game" : "Resign";
  analyzeBtn.hidden = !finished;
  // the nickname prompt appears ONLY once a game has finished
  if (nickForm.hidden && finished) {
    nickForm.hidden = false;
    nickInput.value = snap.nickname || "";
    nickMsg.textContent = "";
    nickMsg.className = "nick-msg";
  }
}

function startPoll() {
  if (play.polling) return;
  play.polling = true;
  const id = play.id;
  (async () => {
    let delay = 600; // ~600ms while the bot thinks, backing off gently
    while (play.id === id && play.snap && play.snap.status === "bot_thinking") {
      await sleep(delay);
      if (play.id !== id) break;
      try {
        ingestPlay(await api.getGame(id));
        delay = Math.min(delay * 1.25, 3000);
      } catch (e) {
        if (e.status === 404) { stopThinking(); setStatus("game expired on the server", "over"); break; }
        delay = Math.min(delay * 2, 8000); // 5xx/network: keep polling, slower
      }
    }
    play.polling = false;
  })();
}

async function tryPlace(q, r) {
  const snap = play.snap;
  if (!play.id || !snap || snap.status !== "your_turn") return;
  if (!legalHas(snap, q, r)) {
    toast("play within reach of the stones");
    return;
  }
  play.lastPlaceT = Date.now();
  playBoard.hideHoverGhost();
  try {
    ingestPlay(await api.postMove(play.id, q, r));
    if (play.snap.status === "bot_thinking") startPoll();
    if (play.snap.status === "finished") refreshFeed(true);
  } catch (e) {
    if (e.status === 409) {
      try { ingestPlay(await api.getGame(play.id)); } catch (_) {}
    } else if (e.status === 422) {
      toast(e.message || "illegal move", true);
    } else if (e.status === 503) {
      // bot backend busy / fail-fast timeout: the move didn't take, so the
      // position is unchanged and the same move can simply be retried. Resync
      // from the server (authoritative) and surface a calm, recoverable notice.
      try { ingestPlay(await api.getGame(play.id)); } catch (_) {}
      if (play.snap && play.snap.status === "your_turn") {
        showPlayAlert("the bot backend is busy right now — your move didn't take. Try that move again in a moment.");
      }
    } else apiFail(e, "move failed — try again");
  }
}

function onPlayCell(q, r, ptrType) {
  const snap = play.snap;
  if (!play.id || !snap || snap.status !== "your_turn") {
    if (!play.id && !play.creating) toast("press new game to start");
    return;
  }
  if (ptrType === "touch") {
    // touch: first tap stages a persistent ghost + confirm chip, second commits
    if (play.staged && play.staged.q === q && play.staged.r === r) {
      clearStage();
      tryPlace(q, r);
    } else if (legalHas(snap, q, r)) {
      stageAt(q, r);
    }
    return;
  }
  clearStage();
  tryPlace(q, r); // mouse: direct place
}

placeChip.addEventListener("click", () => {
  if (!play.staged) return;
  const s = play.staged;
  clearStage();
  tryPlace(s.q, s.r);
});

async function newGame() {
  if (!botsNorm || play.creating) return;
  play.creating = true;
  resignBtn.disabled = true;
  stopThinking(); // reset the elapsed clock so the new game's first turn is fresh
  clearPlayAlert();
  try {
    const snap = await api.createGame({
      ...botsNorm.payloadFor(sel.ckpt, sel.sims),
      human_color: sel.color,
    });
    play.id = snap.id;
    play.moves = [];
    play.staged = null;
    play.label = (snap.bot && snap.bot.label) || sel.ckptLabel;
    play.sims = (snap.bot && (snap.bot.sims ?? snap.bot.visits)) || sel.sims;
    playTag.textContent = `field · ∞ · vs ${play.label} · ${play.sims} sims`;
    nickForm.hidden = true;
    analyzeBtn.hidden = true;
    ingestPlay(snap);
    playBoard.resetView();
    if (snap.status === "bot_thinking") startPoll();
  } catch (e) {
    apiFail(e, "couldn't start a game");
  } finally {
    play.creating = false;
    resignBtn.disabled = false;
  }
}

resignBtn.addEventListener("click", async () => {
  if (play.id && play.snap && play.snap.status !== "finished") {
    try {
      ingestPlay(await api.resign(play.id));
      refreshFeed(true);
    } catch (e) { apiFail(e, "resign failed"); }
  } else {
    newGame();
  }
});

nickForm.addEventListener("submit", async e => {
  e.preventDefault();
  const v = nickInput.value.trim();
  if (!v || !play.id) return;
  try {
    const out = await api.setNickname(play.id, v);
    nickMsg.textContent = `saved — ${out.nickname}`;
    nickMsg.className = "nick-msg ok";
    refreshFeed(true);
  } catch (e2) {
    nickMsg.textContent = e2.message || "couldn't save nickname";
    nickMsg.className = "nick-msg err"; // charset errors surface inline
  }
});

analyzeBtn.addEventListener("click", () => {
  if (!play.id || !play.snap || play.snap.status !== "finished") return;
  const res = play.snap.result || {};
  openAnalysis({
    id: play.id,
    moves: play.moves.slice(),
    label: play.label,
    sims: play.sims,
    ckpt: play.snap.bot && play.snap.bot.checkpoint_id,
    termination: res.termination,
    winner: res.winner,
    winLine: Array.isArray(play.snap.winning_line) ? play.snap.winning_line : null,
  });
});

// ---- bot pickers ---------------------------------------------------------------

let botsNorm = null;
// color: 0 (first, blue) | 1 (second, red) | "random" — default preserves the
// old always-first behavior
const sel = { ckpt: null, ckptLabel: "", sims: 0, color: 0 };

/* The picker itself (grouping, featured/show-all filter, tags, default pick)
 * lives in the shared checkpoints.js so play, analysis and the lab never drift.
 * When "show all" is off, only featured checkpoints (plus the current pick)
 * show; the rest wait behind the toggle. */
let showAllCkpts = false;

/* Re-render both checkpoint lists after the shared "show all" toggle flips. */
function renderCkptLists() {
  const play = $("showAllCkpt"); if (play) play.checked = showAllCkpts;
  const ana = $("showAllAnaCkpt"); if (ana) ana.checked = showAllCkpts;
  if ($("ckptList") && botsNorm) {
    buildCkptList($("ckptList"), botsNorm.checkpoints, { selectedId: sel.ckpt, showAll: showAllCkpts });
  }
  renderAnaCkpts();
}

function renderPickers() {
  const chk = $("showAllCkpt"); if (chk) chk.checked = showAllCkpts;
  buildCkptList($("ckptList"), botsNorm.checkpoints, { selectedId: sel.ckpt, showAll: showAllCkpts });
  const seg = $("simSeg");
  seg.textContent = "";
  for (const s of botsNorm.sims) {
    const b = document.createElement("button");
    b.dataset.sims = s;
    b.textContent = s;
    b.className = s === sel.sims ? "sel" : "";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", s === sel.sims);
    seg.appendChild(b);
  }
}

$("ckptList").addEventListener("click", e => {
  const b = e.target.closest(".bot");
  if (!b) return;
  sel.ckpt = b.dataset.ckpt;
  const c = botsNorm.checkpoints.find(x => x.id === sel.ckpt);
  sel.ckptLabel = c ? c.label : sel.ckpt;
  document.querySelectorAll("#ckptList .bot").forEach(x => {
    const on = x === b;
    x.classList.toggle("sel", on);
    x.setAttribute("aria-checked", on);
  });
});
for (const id of ["showAllCkpt", "showAllAnaCkpt"]) {
  const el = $(id);
  if (el) el.addEventListener("change", e => {
    showAllCkpts = e.target.checked;
    renderCkptLists();
  });
}
$("simSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  sel.sims = +b.dataset.sims;
  document.querySelectorAll("#simSeg button").forEach(x => {
    const on = x === b;
    x.classList.toggle("sel", on);
    x.setAttribute("aria-checked", on);
  });
});
/* Guarded: users can hold a cached pre-colorSeg index.html against this
 * app.js (the assets are cached independently). A null here must not abort
 * module evaluation — everything below (analysis controls, boot()) would
 * silently die with it. Such users just keep the default color. */
const colorSegEl = $("colorSeg");
if (colorSegEl) colorSegEl.addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  sel.color = b.dataset.color === "random" ? "random" : +b.dataset.color;
  colorSegEl.querySelectorAll("button").forEach(x => {
    const on = x === b;
    x.classList.toggle("sel", on);
    x.setAttribute("aria-checked", on);
  });
});

async function loadBots() {
  for (;;) {
    try {
      botsNorm = await api.getBots();
      break;
    } catch (_) {
      await sleep(4000);
    }
  }
  const def = defaultCheckpoint(botsNorm.checkpoints);
  sel.ckpt = def ? def.id : null;
  sel.ckptLabel = def ? def.label : "";
  sel.sims = botsNorm.sims[botsNorm.sims.length - 1] || 0;
  renderPickers();
  setStatus("pick an opponent · new game", "over");
}

// ============================== ANALYSIS =======================================

const ana = {
  id: null, moves: [], n: 0, ply: 1,
  termination: null, winner: null, winLine: null,
  cache: new Map(), pending: new Map(), rlUntil: 0,
  summary: null, summaryState: "idle", // idle | loading | done | missing
  summaryTimer: null, summaryTries: 0,
  opa: 0.85, autoTimer: null,
  // ckpt: catalogue checkpoint analyzing this game; defaultCkpt: the game's
  // own bot. Equal (or null ckpt) means the server default — the param is
  // omitted, which also keeps requests valid on pre-selector servers.
  ckpt: null, defaultCkpt: null, botLabel: "",
};

const anaCursor = $("anaCursor"), anaTag = $("anaTag");
const valNow = $("valNow"), valWho = $("valWho"), stvNow = $("stvNow"), mlNow = $("mlNow");
const plyNow = $("plyNow"), plyMax = $("plyMax"), plyRange = $("plyRange");
const moveListEl = $("moveList"), chartNote = $("chartNote");
let curHeat = new Map();

const anaBoard = createBoard($("anaBoard"), {
  onHover: (q, r) => {
    if (q === null) { anaCursor.textContent = "—"; return; }
    const p = curHeat.get(key(q, r));
    anaCursor.textContent = p !== undefined
      ? `${fmtCell(q, r)} · ${(p * 100).toFixed(1)}%`
      : fmtCell(q, r);
  },
});

// ---- value chart ----------------------------------------------------------------

const chart = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const svg = $("valueChart");
  const W = 280, H = 104, L = 16, R = 8, T = 10, B = 14;
  let n = 1;
  const chX = i => L + (n <= 1 ? 0 : i * (W - L - R) / (n - 1));
  const chY = v => T + (1 - (v + 1) / 2) * (H - T - B);
  const ln = (x1, y1, x2, y2, cls) => {
    const e = document.createElementNS(NS, "line");
    e.setAttribute("x1", x1); e.setAttribute("y1", y1);
    e.setAttribute("x2", x2); e.setAttribute("y2", y2);
    e.setAttribute("class", cls);
    svg.appendChild(e);
    return e;
  };
  const lab = (x, y, s, cls) => {
    const e = document.createElementNS(NS, "text");
    e.setAttribute("x", x); e.setAttribute("y", y);
    e.setAttribute("class", "ch-lab" + (cls ? " " + cls : ""));
    e.textContent = s;
    svg.appendChild(e);
  };
  for (const v of [1, 0.5, -0.5, -1]) ln(L, chY(v), W - R, chY(v), "ch-grid");
  ln(L, chY(0), W - R, chY(0), "ch-zero");
  lab(2, chY(1) + 2.5, "+1", "p0"); lab(5, chY(0) + 2.5, "0"); lab(2, chY(-1) + 2.5, "−1", "p1");
  const rule = ln(chX(0), chY(1), chX(0), chY(-1), "ch-rule");
  const pl = document.createElementNS(NS, "polyline");
  pl.setAttribute("class", "ch-line");
  svg.appendChild(pl);
  const dot = document.createElementNS(NS, "circle");
  dot.setAttribute("r", 2.6);
  dot.setAttribute("class", "ch-dot");
  dot.style.display = "none";
  svg.appendChild(dot);
  let values = null; // blue-POV, index = ply-1

  function setData(vals) {
    values = vals;
    n = Math.max(1, vals ? vals.length : ana.n || 1);
    pl.setAttribute("points", vals
      ? vals.map((v, i) => chX(i).toFixed(1) + "," + chY(v).toFixed(1)).join(" ")
      : "");
    setPlyMark(ana.ply);
  }
  function setPlyMark(p) {
    const x = chX(p - 1);
    rule.setAttribute("x1", x); rule.setAttribute("x2", x);
    const v = values ? values[p - 1] : undefined;
    dot.style.display = v === undefined || v === null ? "none" : "";
    if (v !== undefined && v !== null) {
      dot.setAttribute("cx", x);
      dot.setAttribute("cy", chY(v));
    }
  }
  let dragging = false;
  const seek = e => {
    const r2 = svg.getBoundingClientRect();
    const px = (e.clientX - r2.left) / r2.width * W;
    if (ana.n) setPly(Math.round((px - L) / ((W - L - R) / Math.max(1, ana.n - 1))) + 1);
  };
  svg.addEventListener("pointerdown", e => {
    if (!ana.id) return;
    e.preventDefault();
    dragging = true;
    try { svg.setPointerCapture(e.pointerId); } catch (_) {}
    seek(e);
  });
  svg.addEventListener("pointermove", e => { if (dragging) seek(e); });
  svg.addEventListener("pointerup", () => { dragging = false; });
  svg.addEventListener("pointercancel", () => { dragging = false; });
  return { setData, setPlyMark };
})();

// ---- summary (per-ply value/stv/moves_left) -------------------------------------

/* Server contract: parallel arrays {value, stv, moves_left, to_move} with
 * ply_count + 1 entries — index i is the position AFTER ply i (entry 0 is the
 * empty board; to_move is null at terminal positions). Value/stv are
 * side-to-move POV. Also accepts length-n arrays (plies 1..n) and per-ply
 * object rows, defensively. */
function normalizeSummary(raw, n) {
  if (!raw || typeof raw !== "object") return null;
  let seq = Array.isArray(raw) ? raw : null;
  for (const k of ["plies", "per_ply", "summary"]) {
    if (!seq && Array.isArray(raw[k])) seq = raw[k];
  }
  let values, stv, ml, toMove;
  if (seq && seq.length && typeof seq[0] === "object" && seq[0] !== null) {
    values = seq.map(x => x.value);
    stv = seq.map(x => x.stv ?? x.short_term_value);
    ml = seq.map(x => x.moves_left ?? x.ml);
    toMove = seq.map(x => x.to_move);
  } else if (Array.isArray(raw.value ?? raw.values)) {
    values = raw.value ?? raw.values;
    stv = Array.isArray(raw.stv) ? raw.stv : null;
    ml = Array.isArray(raw.moves_left) ? raw.moves_left : null;
    toMove = Array.isArray(raw.to_move) ? raw.to_move : null;
  }
  if (!values || !values.length) return null;
  const off = values.length === n + 1 ? 0 : -1;
  const at = (arr, p) => {
    const v = arr ? arr[p + off] : undefined;
    return Number.isFinite(v) ? v : null; // null/NaN/±Inf -> "no data"
  };
  return {
    value: p => at(values, p),
    stv: p => at(stv, p),
    ml: p => at(ml, p),
    mover: p => { const m = at(toMove, p); return m === null ? moverAfter(p) : m; },
  };
}

/* Bounded summary retry: transient failures (5xx / network / 429, after
 * api.js's own short GET retries) reschedule loadSummary with exponential
 * backoff, then give up. Permanent answers (404: endpoint missing; 409: the
 * game's own net left the catalogue) never retry. */
const SUMMARY_RETRY_MAX = 4;
const summaryRetryDelay = tries => 3000 * 2 ** tries; // 3s, 6s, 12s, 24s

function cancelSummaryRetry() {
  if (ana.summaryTimer) clearTimeout(ana.summaryTimer);
  ana.summaryTimer = null;
  ana.summaryTries = 0;
}

async function loadSummary() {
  if (!ana.id || ana.summaryState === "loading" || ana.summaryState === "done") return;
  if (ana.summaryTimer) { clearTimeout(ana.summaryTimer); ana.summaryTimer = null; }
  ana.summaryState = "loading";
  chartNote.hidden = false;
  chartNote.textContent = "loading value trace…";
  const id = ana.id, ck = ana.ckpt;
  try {
    const raw = await api.getSummary(id, anaCkptParam());
    if (ana.id !== id || ana.ckpt !== ck) return;
    ana.summaryTries = 0;
    ana.summary = normalizeSummary(raw, ana.n);
    ana.summaryState = ana.summary ? "done" : "missing";
    if (ana.summary) {
      // chart is blue-POV: flip side-to-move values by the mover's color; at
      // a six_in_line terminal ply there is no mover — sign toward the winner
      const vals = [];
      for (let p = 1; p <= ana.n; p++) {
        const v = ana.summary.value(p);
        if (v === null) { vals.push(0); continue; }
        if (p === ana.n && ana.termination === "six_in_line" && ana.winner !== null) {
          vals.push(ana.winner === 0 ? Math.abs(v) : -Math.abs(v));
        } else {
          vals.push(ana.summary.mover(p) === 0 ? v : -v);
        }
      }
      chart.setData(vals);
      chartNote.hidden = true;
      updateValueRead(ana.ply);
    } else {
      chartNote.textContent = "unexpected summary shape from server";
    }
  } catch (e) {
    if (ana.id !== id || ana.ckpt !== ck) return;
    chartNote.hidden = false;
    if (e.status === 404) {
      ana.summaryState = "missing";
      chartNote.textContent = "per-ply summary not available on this server build";
    } else if (e.status === 409) {
      // this game's own net can't analyze it (left the catalogue) — only a
      // different checkpoint pick can change the answer, so don't retry
      ana.summaryState = "missing";
      chartNote.textContent = "no value trace under this game's own net — pick a checkpoint above";
    } else if (ana.summaryTries < SUMMARY_RETRY_MAX) {
      ana.summaryState = "idle";
      chartNote.textContent = "value trace unavailable — retrying…";
      const delay = summaryRetryDelay(ana.summaryTries++);
      ana.summaryTimer = setTimeout(() => {
        ana.summaryTimer = null;
        if (ana.id === id && ana.ckpt === ck && ana.summaryState === "idle") loadSummary();
      }, delay);
    } else {
      ana.summaryState = "idle"; // a net switch or reopen retries afresh
      chartNote.textContent = "value trace unavailable";
    }
  }
}

// ---- per-ply analysis fetch ------------------------------------------------------

function ensureAnalysis(p) {
  if (ana.cache.has(p)) return Promise.resolve(ana.cache.get(p));
  if (Date.now() < ana.rlUntil) return Promise.resolve(null);
  if (ana.pending.has(p)) return ana.pending.get(p);
  const id = ana.id, ck = ana.ckpt;
  const prom = api.getAnalysis(id, p, anaCkptParam())
    .then(pl => {
      ana.pending.delete(p);
      if (ana.id !== id || ana.ckpt !== ck) return null; // game/net switched mid-flight
      ana.cache.set(p, pl);
      return pl;
    })
    .catch(e => {
      ana.pending.delete(p);
      if (e.status === 429) {
        ana.rlUntil = Date.now() + 15000;
        toast("analysis rate-limited — overlay paused briefly");
      } else if (e.status === 503) {
        // bot pool busy/timed out: brief pause so scrub/autoplay doesn't hammer
        ana.rlUntil = Date.now() + 5000;
        toast("analysis backend busy — retrying shortly");
      }
      return null;
    });
  ana.pending.set(p, prom);
  return prom;
}

function renderHeat(payload, p) {
  curHeat = new Map();
  if (!payload || !Array.isArray(payload.policy)) { anaBoard.clearHeat(); return; }
  for (const h of payload.policy) curHeat.set(key(h.q, h.r), h.p);
  const mover = Number.isInteger(payload.to_move) ? payload.to_move : moverAfter(p);
  anaBoard.setHeat(payload.policy, mover === 0 ? H0 : H1, mover === 0 ? H0R : H1R, ana.opa);
}

// ---- analysis checkpoint selector -------------------------------------------------

/* The checkpoint_id to send with analysis/summary requests: only an explicit
 * non-default pick rides the wire (see the ana state comment). */
const anaCkptParam = () =>
  (ana.ckpt && ana.ckpt !== ana.defaultCkpt ? ana.ckpt : null);

function anaCkptLabel() {
  if (ana.ckpt && botsNorm) {
    const c = botsNorm.checkpoints.find(x => x.id === ana.ckpt);
    if (c) return c.label;
  }
  return ana.botLabel || "net";
}

function updateAnaFine() {
  $("anaFine").textContent = `policy priors · ${anaCkptLabel()} · no search`;
}

function renderAnaCkpts() {
  const list = $("anaCkptList");
  if (!list || !botsNorm) return; // cached pre-selector index.html
  const chk = $("showAllAnaCkpt"); if (chk) chk.checked = showAllCkpts;
  buildCkptList(list, botsNorm.checkpoints, { selectedId: ana.ckpt, showAll: showAllCkpts });
}

/* Switch the analyzing net: drop all per-net state, then refetch the summary
 * and the current position under the new checkpoint. The scrubber position
 * (and a running autoplay) are kept. */
function setAnalysisCkpt(id) {
  if (!ana.id || id === ana.ckpt) return;
  ana.ckpt = id;
  document.querySelectorAll("#anaCkptList .bot").forEach(x => {
    const on = x.dataset.ckpt === id;
    x.classList.toggle("sel", on);
    x.setAttribute("aria-checked", on);
  });
  ana.cache = new Map();
  ana.pending = new Map();
  ana.rlUntil = 0;
  ana.summary = null;
  cancelSummaryRetry();
  ana.summaryState = "idle";
  chart.setData(null);
  updateAnaFine();
  loadSummary();
  setPly(ana.ply, true);
}

const anaCkptListEl = $("anaCkptList");
if (anaCkptListEl) anaCkptListEl.addEventListener("click", e => {
  const b = e.target.closest(".bot");
  if (b) setAnalysisCkpt(b.dataset.ckpt);
});

// ---- value readout ----------------------------------------------------------------

function setBig(el, v, cls) {
  if (v === null) {
    el.textContent = "—";
    el.className = cls;
    return;
  }
  el.textContent = fmtV(v);
  el.className = cls + " " + (Math.abs(v) < 0.08 ? "" : (v >= 0 ? "pos" : "neg"));
}

function updateValueRead(p, payload) {
  const terminal = p === ana.n && ana.termination === "six_in_line";
  payload = payload || ana.cache.get(p) || null;
  const mover = payload && Number.isInteger(payload.to_move) ? payload.to_move : moverAfter(p);
  const flip = v => (v === null || v === undefined ? null : (mover === 0 ? v : -v));
  let v = payload && typeof payload.value === "number" ? flip(payload.value) : null;
  let stv = payload && typeof (payload.stv ?? payload.short_term_value) === "number"
    ? flip(payload.stv ?? payload.short_term_value) : null;
  let ml = payload && typeof (payload.moves_left ?? payload.ml) === "number"
    ? (payload.moves_left ?? payload.ml) : null;
  if (ana.summary) {
    const sFlip = x => (x === null ? null : (ana.summary.mover(p) === 0 ? x : -x));
    if (v === null) v = sFlip(ana.summary.value(p));
    if (stv === null) stv = sFlip(ana.summary.stv(p));
    if (ml === null) ml = ana.summary.ml(p);
  }
  if (terminal) {
    // no side to move at a terminal position: sign readouts toward the winner
    if (ana.winner !== null) {
      v = v === null ? (ana.winner === 0 ? 1 : -1) : (ana.winner === 0 ? Math.abs(v) : -Math.abs(v));
      if (stv !== null) stv = ana.winner === 0 ? Math.abs(stv) : -Math.abs(stv);
    }
    valWho.textContent = ana.winner === 0 ? "blue wins" : ana.winner === 1 ? "red wins" : "game over";
    if (ml === null) ml = 0;
  } else {
    valWho.textContent = v === null ? "" : (Math.abs(v) < 0.08 ? "even" : (v >= 0 ? "blue better" : "red better"));
  }
  setBig(valNow, v, "value-big");
  setBig(stvNow, stv, "hz-v");
  mlNow.textContent = ml === null ? "—" : (terminal ? "0" : "~" + Math.max(0, Math.round(ml)));
}

// ---- ply navigation ----------------------------------------------------------------

function setPly(p, fromAuto) {
  if (!ana.id) return;
  if (!fromAuto) stopAuto(); // any manual navigation pauses autoplay
  p = clamp(1, ana.n, p);
  ana.ply = p;
  plyNow.textContent = p;
  plyRange.value = p;
  const stones = ana.moves.slice(0, p);
  const terminal = p === ana.n && ana.termination === "six_in_line";
  const winCells = terminal ? (ana.winLine || findWin(stones)) : null;
  anaBoard.setStones(stones, winCells);
  chart.setPlyMark(p);
  document.querySelector('[data-step="first"]').disabled = p <= 1;
  document.querySelector('[data-step="prev"]').disabled = p <= 1;
  document.querySelector('[data-step="next"]').disabled = p >= ana.n;
  document.querySelector('[data-step="last"]').disabled = p >= ana.n;

  // move-list highlight + keep it in view
  let cur = null;
  moveListEl.querySelectorAll(".mv-c").forEach(b => {
    const on = +b.dataset.ply === p;
    b.classList.toggle("cur", on);
    if (on) cur = b;
  });
  if (cur) {
    const ot = cur.offsetTop, oh = cur.offsetHeight;
    if (ot < moveListEl.scrollTop) moveListEl.scrollTop = ot - 6;
    else if (ot + oh > moveListEl.scrollTop + moveListEl.clientHeight) {
      moveListEl.scrollTop = ot + oh - moveListEl.clientHeight + 6;
    }
  }

  updateValueRead(p);
  anaBoard.clearHeat();
  curHeat = new Map();
  if (!winCells) {
    ensureAnalysis(p).then(pl => {
      if (!pl || ana.ply !== p) return;
      renderHeat(pl, p);
      updateValueRead(p, pl);
    });
  }
}

document.querySelectorAll(".step-btn[data-step]").forEach(b => {
  b.addEventListener("click", () => {
    const s = b.dataset.step;
    setPly(s === "first" ? 1 : s === "prev" ? ana.ply - 1 : s === "next" ? ana.ply + 1 : ana.n);
  });
});
plyRange.addEventListener("input", function () { setPly(+this.value); });

const autoBtn = $("autoBtn");
function stopAuto() {
  if (!ana.autoTimer) return;
  clearInterval(ana.autoTimer);
  ana.autoTimer = null;
  autoBtn.innerHTML = "&#9654;";
  autoBtn.classList.remove("on");
}
function startAuto() {
  if (ana.autoTimer || !ana.id) return;
  if (ana.ply >= ana.n) setPly(1, true); // pressed play at the end: run again
  ana.autoTimer = setInterval(() => {
    setPly(ana.ply + 1, true);
    if (ana.ply >= ana.n) stopAuto();
  }, 1000);
  autoBtn.textContent = "❚❚";
  autoBtn.classList.add("on");
}
autoBtn.addEventListener("click", () => { ana.autoTimer ? stopAuto() : startAuto(); });

// arrow keys step through plies, only while the analysis view is active
document.addEventListener("keydown", e => {
  if (!$("view-analysis").classList.contains("active")) return;
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); setPly(ana.ply - 1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); setPly(ana.ply + 1); }
});

$("ovlRange").addEventListener("input", function () {
  ana.opa = +this.value / 100;
  const pl = ana.cache.get(ana.ply);
  const terminal = ana.ply === ana.n && ana.termination === "six_in_line";
  if (pl && !terminal) renderHeat(pl, ana.ply);
});

// ---- opening a game in analysis ---------------------------------------------------

function buildMoveList() {
  moveListEl.textContent = "";
  const frag = document.createDocumentFragment();
  let t = 1, p = 1;
  while (p <= ana.n) {
    const li = document.createElement("li");
    li.className = "mv";
    const tn = document.createElement("span");
    tn.className = "mv-t";
    tn.textContent = t;
    li.appendChild(tn);
    const count = p === 1 ? 1 : Math.min(2, ana.n - p + 1);
    for (let j = 0; j < count; j++) {
      const pp = p + j, m = ana.moves[pp - 1];
      const b = document.createElement("button");
      b.className = "mv-c p" + m.color;
      b.dataset.ply = pp;
      b.textContent = fmtCell(m.q, m.r);
      li.appendChild(b);
    }
    frag.appendChild(li);
    p += count;
    t++;
  }
  moveListEl.appendChild(frag);
}
moveListEl.addEventListener("click", e => {
  const b = e.target.closest(".mv-c");
  if (b) setPly(+b.dataset.ply);
});

function openAnalysis(rec) {
  stopAuto();
  ana.id = rec.id;
  ana.moves = rec.moves;
  ana.n = rec.moves.length;
  ana.termination = rec.termination || null;
  ana.winner = rec.winner ?? null;
  ana.winLine = rec.winLine || null;
  ana.cache = new Map();
  ana.pending = new Map();
  ana.rlUntil = 0;
  ana.summary = null;
  cancelSummaryRetry();
  ana.summaryState = "idle";
  ana.botLabel = rec.label || "";
  // selector default = the game's own bot. When that checkpoint has left the
  // catalogue the server default 409s, so start on the newest catalogue net
  // instead (the server lets any current checkpoint analyze a retired-bot
  // game). Without bot info or a loaded catalogue, leave the server default.
  ana.defaultCkpt = rec.ckpt || null;
  ana.ckpt = null;
  if (rec.ckpt && botsNorm) {
    if (botsNorm.checkpoints.some(c => c.id === rec.ckpt)) {
      ana.ckpt = rec.ckpt;
    } else {
      const latest = latestCheckpoint(groupCheckpoints(botsNorm.checkpoints));
      ana.ckpt = latest ? latest.id : null;
    }
  }
  renderAnaCkpts();
  plyRange.max = Math.max(1, ana.n);
  plyMax.textContent = ana.n;
  const vs = rec.label ? `vs ${rec.label}·${rec.sims || "?"}` : "game";
  anaTag.textContent = `game ${String(rec.id).slice(0, 8)} · ${vs}`;
  updateAnaFine();
  buildMoveList();
  chart.setData(null);
  const copyBtn = $("copyLink");
  copyBtn.disabled = false;
  const labBtn = $("openLabBtn");
  if (labBtn) labBtn.disabled = false;
  activateView("analysis");
  history.replaceState(null, "", "#game/" + rec.id);
  markFeedSelection();
  if (ana.n) setPly(ana.n, true);
  anaBoard.resetView();
  loadSummary(); // lazy per-game: first analysis open fetches the value trace
}

/* Load a (finished) game from the server by id. The snapshot's `stones` is
 * the placement-ordered move list (client contract); our own just-played
 * game's local record covers the unlikely ordered-stones miss. */
async function loadServerGame(id) {
  try {
    const snap = await api.getGame(id);
    let moves = orderedStones(snap);
    if (moves === null) {
      if (play.id === id && play.moves.length) moves = play.moves.slice();
      else {
        toast("this server build doesn't expose move order yet", true);
        return;
      }
    }
    const norm = moves.map((m, i) => ({
      q: m.q, r: m.r, color: m.color ?? plyColor(i + 1),
    }));
    const res = snap.result || {};
    openAnalysis({
      id,
      moves: norm,
      label: snap.bot && snap.bot.label,
      sims: snap.bot && (snap.bot.sims ?? snap.bot.visits),
      ckpt: snap.bot && snap.bot.checkpoint_id,
      termination: res.termination,
      winner: res.winner,
      winLine: Array.isArray(snap.winning_line) ? snap.winning_line : null,
    });
  } catch (e) {
    if (e.status === 404) toast("game not found (it may have expired)", true);
    else apiFail(e, "couldn't load that game");
  }
}

// ---- games feed --------------------------------------------------------------------

let feedStamp = 0;

function normFeedItem(g) {
  const humanColor = g.human_color ?? 0;
  let winner = null;
  if (g.result && typeof g.result === "object") winner = g.result.winner ?? null;
  else if (Number.isInteger(g.winner)) winner = g.winner;
  else if (g.result === 1 || g.human_result === 1) winner = humanColor;
  else if (g.result === -1 || g.human_result === -1) winner = 1 - humanColor;
  // Result from the human's perspective: +1 human beat the bot, -1 the bot won,
  // 0/null a draw or no decision. Prefer the server's explicit human_result;
  // fall back to comparing the winner against the human's color.
  let humanResult = null;
  if (g.result && typeof g.result === "object" && Number.isInteger(g.result.human_result)) {
    humanResult = g.result.human_result;
  } else if (Number.isInteger(g.human_result)) {
    humanResult = g.human_result;
  } else if (winner !== null && Number.isInteger(humanColor)) {
    humanResult = winner === humanColor ? 1 : -1;
  }
  return {
    id: g.id ?? g.game_id,
    label: (g.bot && g.bot.label) || g.bot_label || g.label || "?",
    sims: (g.bot && (g.bot.sims ?? g.bot.visits)) || g.visits || g.sims || "",
    nickname: g.nickname || null,
    plies: g.ply_count ?? g.plies ?? g.ply ?? null,
    finished: g.finished_at ?? g.finished ?? null,
    winner,
    humanResult,
  };
}

function renderFeed(items) {
  const list = $("gameList");
  list.textContent = "";
  for (const g of items) {
    const b = document.createElement("button");
    b.className = "grow" + (g.id === ana.id ? " sel" : "");
    b.dataset.id = g.id;
    // Outcome from the HUMAN's perspective (not the winning player color): the
    // human can play either color, so a P1-human win must read "human won", not
    // "red won". humanResult already accounts for human_color server-side.
    const cls = g.humanResult === 1 ? "gh-human" : g.humanResult === -1 ? "gh-bot" : "ghx";
    const oc = g.humanResult === 1 ? { t: "human won", c: "win" }
      : g.humanResult === -1 ? { t: "bot won", c: "loss" }
      : g.winner === null ? null : { t: "draw", c: "draw" };
    b.innerHTML =
      `<svg class="g-glyph" width="10" height="11" viewBox="-5.5 -5.5 11 11" aria-hidden="true">` +
      `<polygon class="${cls}" points="4.33,-2.5 4.33,2.5 0,5 -4.33,2.5 -4.33,-2.5 0,-5"/></svg>` +
      `<span class="g-vs"></span><span class="g-outcome"></span>` +
      `<span class="g-nick"></span><span class="g-meta"></span>`;
    b.querySelector(".g-vs").textContent = `vs ${g.label}·${g.sims}`;
    const ocEl = b.querySelector(".g-outcome");
    if (oc) { ocEl.textContent = oc.t; ocEl.classList.add(oc.c); }
    else ocEl.remove();
    b.querySelector(".g-nick").textContent = g.nickname || "";
    b.querySelector(".g-meta").textContent =
      [g.plies !== null ? g.plies + " ply" : "", fmtAgo(g.finished)].filter(Boolean).join(" · ");
    list.appendChild(b);
  }
}

function markFeedSelection() {
  document.querySelectorAll("#gameList .grow").forEach(x => {
    x.classList.toggle("sel", x.dataset.id === String(ana.id));
  });
}

async function refreshFeed(force) {
  if (!force && Date.now() - feedStamp < 30000) return;
  feedStamp = Date.now();
  const note = $("feedNote");
  try {
    const raw = await api.getGamesFeed();
    const seq = Array.isArray(raw) ? raw : (raw && Array.isArray(raw.games) ? raw.games : []);
    renderFeed(seq.map(normFeedItem).filter(g => g.id));
    note.hidden = seq.length > 0;
    note.textContent = "no finished games yet";
  } catch (e) {
    if (e.status === 404) {
      note.hidden = false;
      note.textContent = "public feed not available on this server build";
    } else if (!e.network) {
      note.hidden = false;
      note.textContent = "couldn't load the games feed";
    }
    feedStamp = 0; // allow a quick retry
  }
}

$("gameList").addEventListener("click", e => {
  const b = e.target.closest(".grow");
  if (!b) return;
  if (b.dataset.id === String(ana.id)) return;
  loadServerGame(b.dataset.id);
});

// ---- copy link ---------------------------------------------------------------------

const copyBtn = $("copyLink");
let copyT = null;

function copyText(s) {
  const fallback = () => {
    const ta = document.createElement("textarea");
    ta.value = s;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (_) {}
    ta.remove();
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(s).catch(fallback);
  } else fallback();
}

copyBtn.addEventListener("click", () => {
  if (!ana.id) return;
  copyText(location.origin + location.pathname + "#game/" + ana.id);
  copyBtn.textContent = "copied";
  copyBtn.classList.add("done");
  clearTimeout(copyT);
  copyT = setTimeout(() => {
    copyBtn.textContent = "copy link";
    copyBtn.classList.remove("done");
  }, 1200);
});

// Open the current analysis game, at the ply on screen, in the lab sandbox.
// The lab imports it via ?game=<id>&ply=<n> (see lab.js importFromGame).
const openLabBtn = $("openLabBtn");
if (openLabBtn) openLabBtn.addEventListener("click", () => {
  if (!ana.id) return;
  const url = new URL("learn/lab.html", location.href);
  url.search = "?game=" + encodeURIComponent(ana.id) + "&ply=" + ana.ply;
  window.open(url.href, "_blank", "noopener");
});

// ---- routing + boot ----------------------------------------------------------------

function onHashChange() {
  const m = location.hash.match(/^#game\/([A-Za-z0-9-]{1,64})$/);
  if (!m) return;
  if (String(ana.id) === m[1]) {
    activateView("analysis");
    return;
  }
  loadServerGame(m[1]); // openAnalysis activates the view on success
}
window.addEventListener("hashchange", onHashChange);

(async function boot() {
  await loadBots();
  onHashChange();
})();
