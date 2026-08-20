/* lab.js — the lab page controller: shared position editor (legal-sequence /
 * free-edit) on one board, six inspection modules reading that position.
 *
 * Data flow per module:
 *   features     client-computed (lab_features.js mirrors the server featurizer)
 *   net eval     POST /api/lab/eval                       (worker forward)
 *   heads        POST /api/lab/eval                       (the mantis klent block)
 *   attention    POST /api/lab/eval wants.attention_query (hooked forward)
 *   activations  POST /api/lab/eval wants.activations     (hooked forward)
 *   search       POST /api/lab/search + /api/lab/solve    (real search, solver)
 *
 * In legal-sequence mode the sandbox holds a set of LINES. The board is the
 * active line up to a ply cursor, so stepping back and playing a different
 * move forks a new line and leaves the old one on the strip.
 *
 * board.js is reused for rendering/pan/zoom; scene.js folds and paints the
 * search telemetry the bot-move replay shows. The small helpers duplicated
 * from app.js (toasts, checkpoint grouping, copy, the deep-look phrasing)
 * stay here by design — the lab must not edit the play bundle.
 */

import { S, axialX, axialY, createBoard, findWin, hexPts, key } from "../board.js?v=14";
import { buildModelPicker, defaultCheckpoint, normalizeCheckpoints } from "../checkpoints.js?v=13";
import { newScene, foldLiveFrame, sceneVerdict, paintSceneTo } from "../scene.js?v=1";
import * as LF from "./lab_features.js?v=1";

"use strict";

const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const clamp = (lo, hi, v) => Math.max(lo, Math.min(hi, v));
const sleep = ms => new Promise(res => setTimeout(res, ms));
const fmtV = v => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);
const fmtCell = (q, r) => q + "," + r;
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";

/* overlay tone families (same choices as app.js): pale + hue-shifted so a
 * tinted cell never passes for a stone */
const H0 = "#9fd0ff", H1 = "#ffb4aa";
const H0R = "#d7ebff", H1R = "#ffddd6";
const ACCENT = "#e8e2d6";
/* Overlay tints outside the player frame: the Δ map's promoted/demoted pair
 * and the long-game mass, deliberately neither side's color. */
const GAP_POS = "#8fd6a8", GAP_NEG = "#e0aa5e";
const LONG_TINT = "#b3a1e0";
const OV_OPA = 0.85;

/* Autoplay stops here. A Hexo game ends long before 200 plies; the cap is the
 * backstop for a position the bots shuffle in forever. */
const AUTOPLAY_CAP = 200;

const motionQuery = window.matchMedia
  ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
const reducedMotion = () => !!(motionQuery && motionQuery.matches);

// ---- toasts -------------------------------------------------------------------

const toastWrap = $("toastWrap");
const recentToasts = new Map();
function toast(msg, isErr = false) {
  const last = recentToasts.get(msg);
  if (last && Date.now() - last < 4000) return;
  recentToasts.set(msg, Date.now());
  const el = document.createElement("div");
  el.className = "toast" + (isErr ? " err" : "");
  el.textContent = msg;
  toastWrap.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// ---- api ----------------------------------------------------------------------

async function requestJson(path, body) {
  let resp;
  try {
    resp = await fetch(path, {
      method: body !== undefined ? "POST" : "GET",
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      credentials: "same-origin",
    });
  } catch (_) {
    throw { status: 0, message: "network error" };
  }
  let data = null;
  try { data = await resp.json(); } catch (_) { /* non-JSON body */ }
  if (!resp.ok) {
    const detail = data && (data.detail || data.message);
    throw { status: resp.status, message: typeof detail === "string" ? detail : `HTTP ${resp.status}` };
  }
  return data;
}

// ---- state ----------------------------------------------------------------------

const state = {
  mode: "sequence",              // "sequence" | "free"
  lines: [[]],                   // sequence variations, each [[q, r], ...]
  lineNotes: [null],             // per-line saved solve summary (parallel to lines)
  lineCursors: [0],              // per-line remembered cursor (parallel to lines)
  lineIdx: 0,                    // the line the board shows
  cursor: 0,                     // ply cursor into that line
  free: { p0: [], p1: [], toMove: 0 },
  freeDirty: false,              // free stones diverged from the sequence
  brush: 0,                      // 0 | 1 | "erase"
  undo: [],                      // free-mode snapshots
  staged: null,                  // touch two-tap staging
  module: "features",
  feature: "support",            // "support" | feature name
  ovl: "policy",                 // board overlay: policy | q | pi2 | long | gap
  ckpt: null,
  ckptLabel: "",
  ckptFamily: "shrimp",
  featureNames: null,
  bots: null,                    // normalized /api/bots payload
  attnCell: null,                // [q, r] attention query
  attnBlock: 0,
  attnHead: 0,
  actStage: 0,
  sims: 64,
  rlUntil: 0,                    // rate-limit backoff deadline
  evalCache: new Map(),          // request key -> payload promise
  searchCache: new Map(),
  lastPlaceT: 0,
  /* Bot play. `run` is the cancel token every search and replay carries: any
   * navigation the user drives bumps it, which orphans the frames in flight
   * instead of letting them paint over a position they do not describe.
   * `autoRun` does the same for the autoplay loop itself. */
  bot: { run: 0, autoRun: 0, busy: false, auto: false, plies: 0 },
};

/* The winner when the position on the board is already decided, else null.
 * renderPosition() is the one place that reads the board, so it owns this. */
let termWinner = null;
/* Per-cell critic read (the mantis `klent` block) for the hover text and the
 * board overlays; null for a family that serves none. */
let labKlent = null;

// ---- board ----------------------------------------------------------------------

const cursorPos = $("cursorPos"), placeChip = $("placeChip");

const board = createBoard($("labBoard"), {
  onCellClick: onBoardCell,
  onHover: onBoardHover,
  ghostAllowed: () => state.mode === "sequence" && state.module !== "attention",
  onPanStart: () => board.hideHoverGhost(),
  canReset: t => {
    if (Date.now() - state.lastPlaceT < 500) return false;
    if (t && t.tagName === "polygon" && t.classList &&
        t.classList.contains("cell") && !t.classList.contains("occ")) return false;
    return true;
  },
});

/* Overlay layer for module visuals, above the stones (board draw order is
 * grid, heat, stones, marks, ghost). */
const overlayG = document.createElementNS(NS, "g");
overlayG.setAttribute("class", "lab-ov");
board.svg.insertBefore(overlayG, board.svg.querySelector(".marksg"));

function clearOverlay() { overlayG.textContent = ""; }

function ovPoly(q, r, rad, cls) {
  const el = document.createElementNS(NS, "polygon");
  el.setAttribute("points", hexPts(axialX(q, r), axialY(r), rad));
  el.setAttribute("class", cls);
  overlayG.appendChild(el);
  return el;
}

/* rows: [{q, r, v}] with v >= 0; fills scale to the row max. */
function paintFill(rows, color, maxOpa = 0.6) {
  let max = 0;
  for (const row of rows) max = Math.max(max, row.v);
  if (max <= 0) return;
  for (const row of rows) {
    if (row.v <= 0) continue;
    const el = ovPoly(row.q, row.r, S * 0.975, "ov-fill");
    el.setAttribute("fill", color);
    el.setAttribute("opacity", (maxOpa * (0.1 + 0.9 * row.v / max)).toFixed(3));
  }
}

const paintHalo = cells => { for (const [q, r] of cells) ovPoly(q, r, S * 0.9, "ov-halo"); };

function paintRing(q, r, color, cls = "ov-ring") {
  const el = ovPoly(q, r, S * 0.86, cls);
  if (color) el.setAttribute("stroke", color);
}

/* Direct clicks on ANY grid polygon (board.js's onCellClick filters occupied
 * cells): attention query picking and free-edit erase/recolor need them. The
 * board's capture-phase drag suppressor stops propagation before this fires
 * on a drag's trailing click. */
board.svg.addEventListener("click", e => {
  const t = e.target;
  if (t.tagName !== "polygon" || t.dataset.q === undefined) return;
  const q = +t.dataset.q, r = +t.dataset.r;
  if (state.module === "attention") { pickAttnQuery(q, r); return; }
  if (state.mode === "free" && t.classList.contains("occ")) freeTouchOccupied(q, r);
});

// ---- the line model (legal-sequence mode) -----------------------------------------

const activeLine = () => state.lines[state.lineIdx];
/* The position the board shows: the active line up to the ply cursor. */
const curMoves = () => activeLine().slice(0, state.cursor);

/* Drop every variation and start again from one line, cursor at its end.
 * Presets, shared links, pasted move lists and game imports all land here. */
function setLine(moves) {
  state.lines = [moves.slice()];
  state.lineNotes = [null];
  state.lineCursors = [moves.length];
  state.lineIdx = 0;
  state.cursor = moves.length;
}

/* Play `q, r` at the cursor of the active line. Replaying the move the line
 * already holds steps forward; any other move forks a new line from the
 * shared prefix and makes it active, which leaves the old line untouched. */
function playMove(q, r) {
  const line = activeLine();
  const next = line[state.cursor];
  if (next && next[0] === q && next[1] === r) {
    state.cursor++;
  } else if (state.cursor < line.length) {
    // the old line remembers the fork point, so switching back lands there
    state.lineCursors[state.lineIdx] = state.cursor;
    state.lines.push(line.slice(0, state.cursor).concat([[q, r]]));
    state.lineNotes.push(null);
    state.lineIdx = state.lines.length - 1;
    state.cursor = activeLine().length;
    state.lineCursors.push(state.cursor);
    toast(`forked line ${state.lineIdx + 1}`);
  } else {
    line.push([q, r]);
    state.cursor = line.length;
  }
  positionChanged();
}

function stepCursor(delta) {
  const next = clamp(0, activeLine().length, state.cursor + delta);
  if (next === state.cursor) return;
  cancelBot();
  state.cursor = next;
  positionChanged();
}

/* Each line keeps its own cursor: switching lines restores the ply you left
 * that line on, so two lines compare side by side at their own positions. */
function selectLine(i) {
  if (i === state.lineIdx || !state.lines[i]) return;
  cancelBot();
  state.lineCursors[state.lineIdx] = state.cursor;
  state.lineIdx = i;
  const remembered = state.lineCursors[i];
  state.cursor = clamp(0, state.lines[i].length,
    Number.isInteger(remembered) ? remembered : state.lines[i].length);
  positionChanged();
}

function deleteLine(i) {
  if (state.lines.length < 2) {
    toast("the sandbox keeps at least one line");
    return;
  }
  cancelBot();
  const wasActive = i === state.lineIdx;
  state.lines.splice(i, 1);
  state.lineNotes.splice(i, 1);
  state.lineCursors.splice(i, 1);
  if (state.lineIdx > i || state.lineIdx >= state.lines.length) state.lineIdx--;
  if (wasActive) {
    const remembered = state.lineCursors[state.lineIdx];
    state.cursor = clamp(0, activeLine().length,
      Number.isInteger(remembered) ? remembered : activeLine().length);
  } else {
    // the line on the board did not change — the cursor stays put
    state.cursor = clamp(0, activeLine().length, state.cursor);
  }
  positionChanged();
}

/* Replay a move list under the engine rules. `ok` is false when a placement
 * is illegal or when play continues after six in a line. `winner` names the
 * color whose final placement won, and is null while the game is live. */
function replaySequence(moves) {
  const staged = [];
  for (const [q, r] of moves) {
    if (!LF.isLegalPlacement(staged, q, r)) return { ok: false, winner: null };
    staged.push([q, r]);
    const stones = staged.map(([a, b], i) => ({ q: a, r: b, color: LF.recordPlayer(i) }));
    if (findWin(stones)) {
      return staged.length === moves.length
        ? { ok: true, winner: LF.recordPlayer(staged.length - 1) }
        : { ok: false, winner: null };
    }
  }
  return { ok: true, winner: null };
}

/* A legal replay that is still live — the shape presets and game imports must
 * land on, because a decided position has nothing left for the net to read. */
function validSequence(moves) {
  const rep = replaySequence(moves);
  return rep.ok && rep.winner === null;
}

// ---- position accessors -----------------------------------------------------------

/* Current stones as [{q, r, color}] in a stable render order. */
function stoneList() {
  if (state.mode === "sequence") {
    return curMoves().map(([q, r], i) => ({ q, r, color: LF.recordPlayer(i) }));
  }
  return state.free.p0.map(([q, r]) => ({ q, r, color: 0 }))
    .concat(state.free.p1.map(([q, r]) => ({ q, r, color: 1 })));
}

function currentToMove() {
  if (state.mode === "sequence") return LF.recordPlayer(state.cursor);
  return state.free.toMove;
}

function currentFacts() {
  return state.mode === "sequence"
    ? LF.factsFromSequence(curMoves())
    : LF.factsFromFree(state.free.p0, state.free.p1, state.free.toMove);
}

/* Server body for the current position ({actions} or {stones} + to_move). */
function positionBody() {
  if (state.mode === "sequence") {
    return { actions: curMoves().map(([q, r]) => ({ q, r })) };
  }
  return {
    stones: {
      p0: state.free.p0.map(([q, r]) => ({ q, r })),
      p1: state.free.p1.map(([q, r]) => ({ q, r })),
    },
    to_move: state.free.toMove,
  };
}

function posKey() {
  return state.mode === "sequence"
    ? "s:" + curMoves().map(m => m.join(",")).join(";")
    : "f:" + state.free.p0.map(m => m.join(",")).join(";") +
      "|" + state.free.p1.map(m => m.join(",")).join(";") + "|" + state.free.toMove;
}

// ---- editor: sequence mode ---------------------------------------------------------

function trySequencePlace(q, r) {
  if (state.bot.busy) {
    toast("the bot is moving — wait for the stone to land");
    return;
  }
  if (termWinner !== null) {
    toast("the game is over — step back to keep exploring");
    return;
  }
  if (!LF.isLegalPlacement(curMoves(), q, r)) {
    toast(state.cursor ? "play within reach of the stones" : "the opening stone is forced to 0,0");
    return;
  }
  cancelBot();
  state.lastPlaceT = Date.now();
  playMove(q, r);
}

// ---- editor: free mode ---------------------------------------------------------------

function pushFreeUndo() {
  state.undo.push(JSON.stringify(state.free));
  if (state.undo.length > 200) state.undo.shift();
}

function freeCounts() { return [state.free.p0.length, state.free.p1.length]; }

function freePlace(q, r) {
  if (state.brush === "erase") return; // nothing to erase on an empty cell
  const [c0, c1] = freeCounts();
  const n0 = c0 + (state.brush === 0 ? 1 : 0), n1 = c1 + (state.brush === 1 ? 1 : 0);
  if (Math.abs(n0 - n1) > 2) {
    toast("stone counts must stay within 2 of each other", true);
    return;
  }
  pushFreeUndo();
  state.lastPlaceT = Date.now();
  (state.brush === 0 ? state.free.p0 : state.free.p1).push([q, r]);
  state.freeDirty = true;
  positionChanged();
}

function freeTouchOccupied(q, r) {
  const inP0 = state.free.p0.findIndex(([a, b]) => a === q && b === r);
  const inP1 = state.free.p1.findIndex(([a, b]) => a === q && b === r);
  if (inP0 < 0 && inP1 < 0) return;
  if (state.brush === "erase") {
    // remove the stone, but only if the counts stay within the envelope
    const n0 = state.free.p0.length - (inP0 >= 0 ? 1 : 0);
    const n1 = state.free.p1.length - (inP1 >= 0 ? 1 : 0);
    if (Math.abs(n0 - n1) > 2) {
      toast("stone counts must stay within 2 of each other", true);
      return;
    }
    pushFreeUndo();
    if (inP0 >= 0) state.free.p0.splice(inP0, 1);
    else state.free.p1.splice(inP1, 1);
  } else if ((state.brush === 0 && inP1 >= 0) || (state.brush === 1 && inP0 >= 0)) {
    // recolor to the brush color (delta stays within the envelope)
    const n0 = state.free.p0.length + (state.brush === 0 ? 1 : -1);
    const n1 = state.free.p1.length + (state.brush === 1 ? 1 : -1);
    if (Math.abs(n0 - n1) > 2) {
      toast("stone counts must stay within 2 of each other", true);
      return;
    }
    pushFreeUndo();
    if (inP1 >= 0) { state.free.p1.splice(inP1, 1); state.free.p0.push([q, r]); }
    else { state.free.p0.splice(inP0, 1); state.free.p1.push([q, r]); }
  } else {
    return; // same-color click: nothing to do
  }
  state.freeDirty = true;
  positionChanged();
}

// ---- board click routing (empty cells; touch stages, mouse places) --------------------

function commitCell(q, r) {
  if (state.mode === "sequence") trySequencePlace(q, r);
  else freePlace(q, r);
}

function clearStage() {
  state.staged = null;
  board.clearStage();
  placeChip.classList.remove("show");
}

function onBoardCell(q, r, ptrType) {
  if (state.module === "attention") return; // svg listener picks the query
  if (ptrType === "touch") {
    if (state.staged && state.staged.q === q && state.staged.r === r) {
      clearStage();
      commitCell(q, r);
    } else if (state.mode === "free" ||
               (termWinner === null && LF.isLegalPlacement(curMoves(), q, r))) {
      state.staged = { q, r };
      board.stage(q, r);
      placeChip.classList.add("show");
    }
    return;
  }
  clearStage();
  commitCell(q, r);
}

placeChip.addEventListener("click", () => {
  if (!state.staged) return;
  const s = state.staged;
  clearStage();
  commitCell(s.q, s.r);
});

/* The cursor readout names whatever the board currently encodes for the cell:
 * the coordinate alone in the modules that paint their own map, plus the
 * critic read wherever the eval/heads overlays are on the board. */
function cursorText(q, r) {
  const parts = [fmtCell(q, r)];
  const cell = labKlent && labKlent.get(key(q, r));
  if (cell && (state.module === "eval" || state.module === "heads")) {
    if (state.ovl === "pi2" || state.module === "heads") {
      parts.push(`π′ ${pct(cell.improved)} · π ${pct(cell.prior)}`);
    } else {
      parts.push(`π ${pct(cell.prior)}`);
    }
    parts.push(`Q ${fmtV(cell.q)}`);
    parts.push(
      `win ${pct(cell.win, 0)} · loss ${pct(cell.loss, 0)} · long ${pct(cell.long, 0)}`,
    );
  }
  return parts.join(" · ");
}

function onBoardHover(q, r) {
  cursorPos.textContent = q === null ? "—" : cursorText(q, r);
}

// ---- editor controls --------------------------------------------------------------

function segSelect(seg, match) {
  seg.querySelectorAll("button").forEach(b => {
    const on = match(b);
    b.classList.toggle("sel", on);
    b.setAttribute(b.getAttribute("role") === "tab" ? "aria-selected" : "aria-checked", on);
  });
}

$("modeSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b || b.dataset.mode === state.mode) return;
  setMode(b.dataset.mode);
});

function setMode(mode) {
  if (mode === state.mode) return;
  cancelBot();
  if (mode === "free") {
    // carry the sequence position into the editable stone set
    state.free = {
      p0: stoneList().filter(s => s.color === 0).map(s => [s.q, s.r]),
      p1: stoneList().filter(s => s.color === 1).map(s => [s.q, s.r]),
      toMove: currentToMove(),
    };
    state.freeDirty = false;
    state.undo = [];
  } else {
    if (state.freeDirty) toast("free edits dropped — restored the last legal sequence");
  }
  state.mode = mode;
  syncModeUI();
  positionChanged();
}

$("brushSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  state.brush = b.dataset.brush === "erase" ? "erase" : +b.dataset.brush;
  segSelect($("brushSeg"), x => x === b);
});

$("tmSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  state.free.toMove = +b.dataset.tm;
  state.freeDirty = true;
  segSelect($("tmSeg"), x => x === b);
  positionChanged();
});

/* Free-edit undo. Sequence mode has no undo button: the step pair owns the
 * cursor, and stepping back never loses the line's moves. */
$("undoBtn").addEventListener("click", () => {
  const snap = state.undo.pop();
  if (!snap) return;
  state.free = JSON.parse(snap);
  segSelect($("tmSeg"), b => +b.dataset.tm === state.free.toMove);
  positionChanged();
});

$("stepBack").addEventListener("click", () => stepCursor(-1));
$("stepFwd").addEventListener("click", () => stepCursor(1));

/* The timeline slider scrubs the cursor through the active line. The board
 * follows every input; the module refresh rides its usual debounce. */
$("plySlider").addEventListener("input", function () {
  const next = clamp(0, activeLine().length, Math.round(+this.value));
  if (next === state.cursor) return;
  cancelBot();
  state.cursor = next;
  positionChanged();
});

/* Arrow keys step the cursor wherever the page has focus, except inside a
 * text control (the slider handles its own arrows through the input event). */
document.addEventListener("keydown", e => {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" ||
            t.tagName === "SELECT" || t.isContentEditable)) return;
  if (state.mode !== "sequence") return;
  // "Left"/"Right" are the pre-standard names some engines still send
  if (e.key === "ArrowLeft" || e.key === "Left") { e.preventDefault(); stepCursor(-1); }
  else if (e.key === "ArrowRight" || e.key === "Right") { e.preventDefault(); stepCursor(1); }
});

$("clearBtn").addEventListener("click", () => {
  cancelBot();
  if (state.mode === "sequence") setLine([]);
  else { pushFreeUndo(); state.free = { p0: [], p1: [], toMove: 0 }; state.freeDirty = true; }
  $("presetSel").value = "";
  positionChanged();
});

$("lineChips").addEventListener("click", e => {
  const pick = e.target.closest(".ls-pick");
  if (pick) { selectLine(+pick.dataset.i); return; }
  const del = e.target.closest(".ls-del");
  if (del) deleteLine(+del.dataset.i);
});

/* One chip per variation: its number, its length, and a delete control. The
 * active chip carries the cursor read, so the strip says both which line is
 * on the board and how far into it the board is. */
function renderLineStrip() {
  $("plyRead").textContent = state.cursor + "/" + activeLine().length;
  $("stepBack").disabled = state.cursor === 0;
  $("stepFwd").disabled = state.cursor >= activeLine().length;
  const slider = $("plySlider");
  slider.max = activeLine().length;
  slider.value = state.cursor;
  slider.disabled = activeLine().length === 0;
  // The active line's saved solve summary rides under the strip, so a
  // reviewed line explains itself whenever it is on the board.
  const note = state.lineNotes[state.lineIdx] || "";
  const noteEl = $("lineNote");
  noteEl.textContent = note;
  noteEl.hidden = !note;
  const chips = $("lineChips");
  chips.textContent = "";
  state.lines.forEach((line, i) => {
    const chip = document.createElement("span");
    chip.className = "ls-chip" + (i === state.lineIdx ? " sel" : "")
      + (state.lineNotes[i] ? " ls-solved" : "");
    const pick = document.createElement("button");
    pick.className = "ls-pick";
    pick.dataset.i = i;
    pick.textContent = `line ${i + 1} · ${line.length}`;
    pick.title = state.lineNotes[i] || `Show line ${i + 1}`;
    const del = document.createElement("button");
    del.className = "ls-del";
    del.dataset.i = i;
    del.textContent = "×";
    del.title = `Delete line ${i + 1}`;
    del.setAttribute("aria-label", `Delete line ${i + 1}`);
    del.disabled = state.lines.length < 2;
    chip.append(pick, del);
    chips.appendChild(chip);
  });
}

// ---- per-line value graph -------------------------------------------------------

/* The analysis page's value chart, retold for the active line: one point per
 * ply (index 0 is the line's start), blue-POV v̂ from the shared eval cache,
 * filled progressively. Click or drag the chart to move the cursor. */
const valChart = (() => {
  const svg = $("labValChart");
  const W = 560, H = 96, L = 16, R = 8, T = 8, B = 12;
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
  lab(2, chY(1) + 2.5, "+1", "p0");
  lab(5, chY(0) + 2.5, "0");
  lab(2, chY(-1) + 2.5, "−1", "p1");
  const rule = ln(chX(0), chY(1), chX(0), chY(-1), "ch-rule");
  const path = document.createElementNS(NS, "path");
  path.setAttribute("class", "ch-line");
  svg.appendChild(path);
  const ptsG = document.createElementNS(NS, "g");
  svg.appendChild(ptsG);
  const dot = document.createElementNS(NS, "circle");
  dot.setAttribute("r", 2.6);
  dot.setAttribute("class", "ch-dot");
  dot.style.display = "none";
  svg.appendChild(dot);
  let values = null;

  /* Known values draw as connected runs; a point with no known neighbor gets
   * its own marker, so a partially filled trace still reads. */
  function redraw() {
    ptsG.textContent = "";
    let d = "";
    if (values) {
      const known = j => j >= 0 && j < values.length &&
        values[j] !== null && values[j] !== undefined;
      for (let i = 0; i < values.length; i++) {
        if (!known(i)) continue;
        d += (known(i - 1) ? " L" : " M") +
          chX(i).toFixed(1) + "," + chY(values[i]).toFixed(1);
        if (!known(i - 1) && !known(i + 1)) {
          const c = document.createElementNS(NS, "circle");
          c.setAttribute("cx", chX(i).toFixed(1));
          c.setAttribute("cy", chY(values[i]).toFixed(1));
          c.setAttribute("r", 1.6);
          c.setAttribute("class", "ch-pt");
          ptsG.appendChild(c);
        }
      }
    }
    path.setAttribute("d", d.trim());
  }
  function setData(vals) {
    values = vals;
    n = Math.max(1, vals ? vals.length : 1);
    redraw();
    setCursor(state.cursor);
  }
  function setCursor(c) {
    const x = chX(clamp(0, n - 1, c));
    rule.setAttribute("x1", x);
    rule.setAttribute("x2", x);
    const v = values ? values[c] : undefined;
    dot.style.display = v === undefined || v === null ? "none" : "";
    if (v !== undefined && v !== null) {
      dot.setAttribute("cx", x);
      dot.setAttribute("cy", chY(v));
    }
  }
  let dragging = false;
  const seek = e => {
    const rc = svg.getBoundingClientRect();
    const px = (e.clientX - rc.left) / rc.width * W;
    const next = clamp(0, activeLine().length,
      Math.round((px - L) / ((W - L - R) / Math.max(1, n - 1))));
    if (next === state.cursor) return;
    cancelBot();
    state.cursor = next;
    positionChanged();
  };
  svg.addEventListener("pointerdown", e => {
    if (state.mode !== "sequence") return;
    e.preventDefault();
    dragging = true;
    try { svg.setPointerCapture(e.pointerId); } catch (_) {}
    seek(e);
  });
  svg.addEventListener("pointermove", e => { if (dragging) seek(e); });
  svg.addEventListener("pointerup", () => { dragging = false; });
  svg.addEventListener("pointercancel", () => { dragging = false; });
  return { setData, setCursor };
})();

let traceRun = 0;
let traceKey = null;
let traceVals = null;

/* Keep the graph in step with the page. Cursor moves only move the rule; a
 * line, checkpoint or mode change starts one sequential fetch pass. Autoplay
 * defers the trace until the loop stops (one eval per ply would ride under
 * every replay). */
function syncValueGraph() {
  const wrap = $("valGraph");
  if (state.mode !== "sequence" || !state.ckpt) {
    wrap.hidden = true;
    traceKey = null;
    traceRun++;
    return;
  }
  wrap.hidden = false;
  const line = activeLine();
  const k = state.ckpt + "|" + state.lineIdx + "|" +
    line.map(m => m.join(",")).join(";");
  if (k === traceKey) {
    valChart.setCursor(state.cursor);
    return;
  }
  if (state.bot.auto || state.bot.busy) {
    // stale on purpose: the post-loop refresh re-enters with the final line
    traceKey = null;
    valChart.setData(new Array(line.length + 1).fill(null));
    return;
  }
  traceKey = k;
  traceVals = new Array(line.length + 1).fill(null);
  valChart.setData(traceVals);
  traceLine(line.slice(), traceVals, ++traceRun);
}

async function traceLine(line, vals, run) {
  // A decided line ends at certainty: the last point is the winner's ±1 and
  // is never fetched (the eval endpoint reads live positions only).
  let last = line.length;
  const rep = replaySequence(line);
  if (rep.winner !== null) {
    vals[line.length] = rep.winner === 0 ? 1 : -1;
    last = line.length - 1;
    valChart.setData(vals);
  }
  for (let p = 0; p <= last; p++) {
    if (run !== traceRun) return;
    let payload;
    try {
      payload = await fetchEvalAt(line.slice(0, p));
    } catch (e) {
      if (e.status === 429) return; // the next line/checkpoint change retraces
      continue;                     // transient: leave the gap, keep going
    }
    if (run !== traceRun) return;
    vals[p] = typeof payload.value === "number"
      ? (payload.to_move === 0 ? payload.value : -payload.value) : null;
    valChart.setData(vals);
  }
}

// ---- share link ----------------------------------------------------------------------

/* The link carries the ACTIVE line up to the cursor — the position on the
 * board. Variations are session-local: the schema is the one old links use. */
function shareHash() {
  if (state.mode === "sequence") {
    return "#m=" + curMoves().map(m => m.join(",")).join(";");
  }
  return "#f0=" + state.free.p0.map(m => m.join(",")).join(";") +
         "&f1=" + state.free.p1.map(m => m.join(",")).join(";") +
         "&tm=" + state.free.toMove;
}

function parseCells(text) {
  if (!text) return [];
  const out = [];
  for (const part of text.split(";")) {
    const m = /^(-?\d+),(-?\d+)$/.exec(part);
    if (!m) return null;
    out.push([+m[1], +m[2]]);
  }
  return out;
}

/* #m=... (sequence) / #f0=...&f1=...&tm=... (free). Returns true if applied.
 * An optional &pv=... rides #m=: a proven continuation (the analysis screen's
 * solver hands positions off this way) that loads as a second line, active
 * and forked at the position, so stepping forward walks the proof. */
function applyHash(hash) {
  if (!hash || hash.length < 2) return false;
  const params = new URLSearchParams(hash.slice(1));
  if (params.has("m")) {
    const moves = parseCells(params.get("m"));
    // A shared link may end on the move that won: the board shows the result
    // and the modules report the position as over.
    if (moves === null || !replaySequence(moves).ok) {
      toast("lab link position was not a legal sequence", true);
      return false;
    }
    state.mode = "sequence";
    setLine(moves);
    const pv = params.has("pv") ? parseCells(params.get("pv")) : null;
    if (pv && pv.length) {
      const full = moves.concat(pv);
      if (replaySequence(full).ok) {
        state.lines.push(full);
        state.lineNotes.push("proof line handed off from the analysis solver");
        state.lineCursors.push(moves.length);
        state.lineIdx = 1;
        state.cursor = moves.length;
      } else {
        toast("lab link proof line was not legal — loaded the position only", true);
      }
    }
    return true;
  }
  if (params.has("f0") || params.has("f1")) {
    const p0 = parseCells(params.get("f0") || "");
    const p1 = parseCells(params.get("f1") || "");
    if (p0 === null || p1 === null) return false;
    const seen = new Set();
    for (const [q, r] of p0.concat(p1)) {
      if (seen.has(key(q, r))) return false;
      seen.add(key(q, r));
    }
    if (Math.abs(p0.length - p1.length) > 2) return false;
    const tm = params.get("tm");
    state.mode = "free";
    state.free = {
      p0, p1,
      toMove: tm === "0" || tm === "1" ? +tm : LF.defaultFreeToMove(p0.length, p1.length),
    };
    return true;
  }
  return false;
}

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

let copyT = null;
$("copyLab").addEventListener("click", () => {
  copyText(location.origin + location.pathname + shareHash());
  const b = $("copyLab");
  b.textContent = "copied";
  b.classList.add("done");
  clearTimeout(copyT);
  copyT = setTimeout(() => { b.textContent = "copy lab link"; b.classList.remove("done"); }, 1200);
});

// ---- presets (learn/data contract; degrade to empty-board-only) -----------------------

async function loadPresets() {
  // Curated tactical positions from the learn/data snapshots (JSON contract:
  // doc.positions[].{id, title, moves}). Both files carry position lists;
  // merge them, first occurrence of an id wins. Any fetch failure degrades to
  // empty-board-only — the page stands alone.
  const positions = [];
  const seen = new Set();
  for (const file of ["data/checkpoints.json", "data/features.json"]) {
    let doc = null;
    try { doc = await requestJson(file); } catch (_) { continue; }
    for (const pos of (doc && Array.isArray(doc.positions) ? doc.positions : [])) {
      if (pos && typeof pos.id === "string" && !seen.has(pos.id)) {
        seen.add(pos.id);
        positions.push(pos);
      }
    }
  }
  const sel = $("presetSel");
  for (const pos of positions) {
    if (!pos || typeof pos.id !== "string" || !Array.isArray(pos.moves)) continue;
    const moves = pos.moves.map(m => (Array.isArray(m) && m.length === 2 ? [+m[0], +m[1]] : null));
    if (moves.some(m => m === null) || !validSequence(moves)) continue;
    const opt = document.createElement("option");
    opt.value = pos.id;
    opt.textContent = typeof pos.title === "string" ? pos.title : pos.id;
    opt.dataset.moves = JSON.stringify(moves);
    sel.appendChild(opt);
  }

  // The community forcing corpus rides the same dropdown under its own group.
  // Its entries extend the contract: a sequence may carry `line` (the
  // submitter's recorded solve continuation, pre-loaded as a second line in
  // the strip) and an out-of-parity puzzle carries `free` instead of moves.
  let corpus = null;
  try { corpus = await requestJson("data/forcing_corpus.json"); } catch (_) { /* page stands alone */ }
  const entries = corpus && Array.isArray(corpus.positions) ? corpus.positions : [];
  const group = document.createElement("optgroup");
  group.label = "Forcing corpus";
  for (const pos of entries) {
    if (!pos || typeof pos.id !== "string" || seen.has(pos.id)) continue;
    seen.add(pos.id);
    const opt = document.createElement("option");
    opt.value = "corpus:" + pos.id;
    opt.textContent = typeof pos.title === "string" ? pos.title : pos.id;
    if (Array.isArray(pos.moves)) {
      const moves = pos.moves.map(m => (Array.isArray(m) && m.length === 2 ? [+m[0], +m[1]] : null));
      if (moves.some(m => m === null) || !validSequence(moves)) continue;
      opt.dataset.moves = JSON.stringify(moves);
      if (Array.isArray(pos.line) && pos.line.length) {
        const line = pos.line.map(m => (Array.isArray(m) && m.length === 2 ? [+m[0], +m[1]] : null));
        if (!line.some(m => m === null) && replaySequence(moves.concat(line)).ok) {
          opt.dataset.line = JSON.stringify(line);
        }
      }
    } else if (pos.free && Array.isArray(pos.free.p0) && Array.isArray(pos.free.p1)) {
      opt.dataset.free = JSON.stringify({
        p0: pos.free.p0.map(m => [+m[0], +m[1]]),
        p1: pos.free.p1.map(m => [+m[0], +m[1]]),
        toMove: pos.free.to_move === 1 ? 1 : 0,
      });
    } else {
      continue;
    }
    group.appendChild(opt);
  }
  if (group.children.length) sel.appendChild(group);
}

/* Land a free-edit stone set (an out-of-parity import: only free mode can
 * hold it). Shared by the corpus dropdown and the community-site import. */
function loadFreePosition(p0, p1, toMove) {
  cancelBot();
  state.mode = "free";
  state.free = { p0, p1, toMove: toMove === 1 ? 1 : 0 };
  state.freeDirty = false;
  state.undo = [];
  syncModeUI();
  positionChanged();
  board.resetView();
}

$("presetSel").addEventListener("change", function () {
  const opt = this.selectedOptions[0];
  cancelBot();
  if (opt && opt.dataset.free) {
    const f = JSON.parse(opt.dataset.free);
    loadFreePosition(f.p0, f.p1, f.toMove);
    return;
  }
  const moves = opt && opt.dataset.moves ? JSON.parse(opt.dataset.moves) : [];
  if (state.mode !== "sequence") setMode("sequence");
  setLine(moves);
  if (opt && opt.dataset.line) {
    // The recorded solve continuation as a second, deletable line; the
    // dropdown still lands on the puzzle position itself.
    state.lines.push(moves.concat(JSON.parse(opt.dataset.line)));
    state.lineNotes.push(null);
    state.lineCursors.push(moves.length);
  }
  positionChanged();
  board.resetView();
});

// ---- paste a move list ------------------------------------------------------------------

const pastePanel = $("pastePanel"), gamesPanel = $("gamesPanel");

/* One panel at a time under the strip: the toggles share the space. */
function togglePanel(panel, btn) {
  const show = panel.hidden;
  for (const [p, b] of [[pastePanel, $("pasteToggle")], [gamesPanel, $("gamesToggle")]]) {
    p.hidden = true;
    b.setAttribute("aria-expanded", "false");
    b.classList.remove("done");
  }
  panel.hidden = !show;
  btn.setAttribute("aria-expanded", show ? "true" : "false");
  btn.classList.toggle("done", show);
  return show;
}

$("pasteToggle").addEventListener("click", () => {
  if (togglePanel(pastePanel, $("pasteToggle"))) $("pasteBox").focus();
});

/* HTTX, the community hexo notation: a `version[1];` header, then numbered
 * two-stone turns `n. [q,r][q,r];`. The first player's opening stone at the
 * origin is implicit, so the flattened record is the origin plus the turns in
 * order. The final turn may carry one cell (a mid-turn export). Returns null
 * when the text is not HTTX, {err} on a malformed document, {moves} on
 * success — the engine replay stays the legality authority. */
function parseHttx(raw) {
  const vm = /version\s*\[\s*(\d+)\s*\]/i.exec(raw);
  if (!vm) return null;
  if (vm[1] !== "1") return { err: `unsupported httx version ${vm[1]}` };
  const turnRe = /(\d+)\s*\.((?:\s*\[\s*-?\d+\s*,\s*-?\d+\s*\])+)\s*;/g;
  const turns = [];
  let m;
  while ((m = turnRe.exec(raw)) !== null) {
    if (+m[1] !== turns.length + 1) {
      return { err: `httx turns must run 1..n — turn ${turns.length + 1} is numbered ${m[1]}` };
    }
    const cells = [];
    const cellRe = /\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]/g;
    let c;
    while ((c = cellRe.exec(m[2])) !== null) cells.push([+c[1], +c[2]]);
    turns.push(cells);
  }
  if (!turns.length) return { err: "httx header but no numbered turns" };
  for (let i = 0; i < turns.length; i++) {
    if (turns[i].length !== 2 && !(i === turns.length - 1 && turns[i].length === 1)) {
      return { err: `httx turn ${i + 1} must place two stones (the final turn may place one)` };
    }
  }
  // Every bracketed cell in the text must have landed in a turn — a turn the
  // regex skipped (say, a missing semicolon) must not silently vanish.
  const cellCount = (raw.match(/\[\s*-?\d+\s*,\s*-?\d+\s*\]/g) || []).length;
  const parsed = turns.reduce((n, t) => n + t.length, 0);
  if (parsed !== cellCount) {
    return { err: `httx has ${cellCount} cells but only ${parsed} parsed — check the turn separators` };
  }
  return { moves: [[0, 0]].concat(turns.flat()) };
}

/* A hexo.did.science link: their sandbox and finished-game pages. The site
 * sends no CORS headers, so the position comes through this server's
 * normalizing proxy (see lab_import.py). */
const DID_SCIENCE_RE = /hexo\.did\.science\/(sandbox|games)\/([A-Za-z0-9-]{1,64})/;

async function importFromDidScience(match) {
  const kind = match[1] === "sandbox" ? "sandbox" : "game";
  const btn = $("pasteApply");
  btn.disabled = true;
  toast("importing from hexo.did.science…");
  let payload;
  try {
    payload = await requestJson(
      `/api/lab/import/didscience?kind=${kind}&id=${encodeURIComponent(match[2])}`);
  } catch (e) {
    toast(e.message || "hexo.did.science import failed", true);
    return;
  } finally {
    btn.disabled = false;
  }
  if (payload.stones) {
    loadFreePosition(payload.stones.p0, payload.stones.p1, payload.to_move);
    toast(`loaded ${payload.name} (free edit — the position is out of turn parity)`);
    return;
  }
  const moves = payload.moves.map(m => [+m[0], +m[1]]);
  if (!replaySequence(moves).ok) {
    toast("the imported record did not replay as a legal sequence", true);
    return;
  }
  cancelBot();
  if (state.mode !== "sequence") setMode("sequence");
  setLine(moves);
  positionChanged();
  board.resetView();
  toast(`loaded ${payload.name} · ${moves.length} plies`);
}

$("pasteApply").addEventListener("click", () => {
  // Takes a bare move list, a copied lab link (the same list behind a "#m="
  // prefix), an HTTX document, or a hexo.did.science sandbox/game link.
  const raw = $("pasteBox").value;
  const did = DID_SCIENCE_RE.exec(raw);
  if (did) {
    importFromDidScience(did);
    return;
  }
  let moves;
  const httx = parseHttx(raw);
  if (httx) {
    if (httx.err) {
      toast(httx.err, true);
      return;
    }
    moves = httx.moves;
  } else {
    const linked = /#m=([^&\s]*)/.exec(raw);
    const text = (linked ? linked[1] : raw).replace(/\s+/g, "");
    if (!text) {
      toast("paste a move list first", true);
      return;
    }
    moves = parseCells(text);
    if (moves === null) {
      toast("moves must read q,r;q,r — one cell per pair — or httx", true);
      return;
    }
  }
  if (!replaySequence(moves).ok) {
    toast("that move list is not a legal sequence", true);
    return;
  }
  cancelBot();
  if (state.mode !== "sequence") setMode("sequence");
  setLine(moves);
  positionChanged();
  board.resetView();
  toast(`loaded ${moves.length} plies as line 1`);
});

// ---- game import (?game=<id>&ply=<n>, and the picker) -------------------------------------

$("gamesToggle").addEventListener("click", () => {
  if (togglePanel(gamesPanel, $("gamesToggle")) && !$("labGameList").children.length) {
    loadGames();
  }
});

$("gamesReload").addEventListener("click", loadGames);

/* The public feed, most recent first. The rows carry the id and the ply
 * count; the ply field picks where in the game the sandbox opens. */
async function loadGames() {
  setStatus("gamesStatus", "loading…");
  let raw;
  try {
    raw = await requestJson("/api/games");
  } catch (e) {
    setStatus("gamesStatus", e.status === 404
      ? "this server build serves no public feed"
      : (e.message || "could not load the games feed"), true);
    return;
  }
  const rows = Array.isArray(raw) ? raw : (raw && Array.isArray(raw.games) ? raw.games : []);
  const list = $("labGameList");
  list.textContent = "";
  for (const g of rows.slice(0, 20)) {
    const id = g.id ?? g.game_id;
    if (!id) continue;
    const plies = g.ply_count ?? g.plies ?? null;
    const b = document.createElement("button");
    b.className = "game-row";
    b.dataset.id = id;
    if (plies !== null) b.dataset.plies = plies;
    b.innerHTML = `<span class="gr-n"></span><span class="gr-m"></span>`;
    b.querySelector(".gr-n").textContent = g.nickname || "anonymous";
    b.querySelector(".gr-m").textContent =
      `vs ${(g.bot && g.bot.label) || g.bot_label || g.label || "?"}` +
      (plies === null ? "" : ` · ${plies} ply`);
    list.appendChild(b);
  }
  setStatus("gamesStatus", list.children.length ? "" : "no finished games yet");
}

$("labGameList").addEventListener("click", e => {
  const row = e.target.closest(".game-row");
  if (!row) return;
  const typed = $("gamePly").value.trim();
  // Empty ply field = the game's last ply, which is what the field's
  // placeholder says. A finished game trims back to its last live position.
  const ply = typed === ""
    ? (row.dataset.plies ? +row.dataset.plies : NaN)
    : parseInt(typed, 10);
  importFromGame(row.dataset.id, ply);
});

async function importFromGame(id, ply) {
  try {
    const snap = await requestJson(`/api/game/${encodeURIComponent(id)}`);
    const st = snap.stones;
    if (!Array.isArray(st)) throw { message: "no stones in game payload" };
    const lm = snap.last_move;
    if (st.length && lm && (st[st.length - 1].q !== lm.q || st[st.length - 1].r !== lm.r)) {
      throw { message: "server did not send placement order" };
    }
    const upto = Number.isFinite(ply) ? clamp(0, st.length, ply) : st.length;
    let moves = st.slice(0, upto).map(s => [s.q, s.r]);
    // A finished game's final placement may be terminal; trim to a decision state.
    while (moves.length && !validSequence(moves)) moves = moves.slice(0, -1);
    cancelBot();
    if (state.mode !== "sequence") { state.mode = "sequence"; syncModeUI(); }
    setLine(moves);
    positionChanged();
    board.resetView();
    toast(`loaded game ${String(id).slice(0, 8)} at ply ${moves.length}`);
  } catch (e) {
    toast(e.message === "network error" ? "couldn't reach the server" : "couldn't load that game", true);
  }
}

// ---- checkpoints (the shared picker; see ../checkpoints.js) ------------------------------

function renderCkpts() {
  buildModelPicker($("ckptList"), state.bots, {
    selectedId: state.ckpt,
    onSelect: (id, c) => {
      if (id === state.ckpt) return;
      cancelBot();
      state.ckpt = id;
      state.ckptLabel = c ? c.label : id;
      state.ckptFamily = c ? c.family : "shrimp";
      state.feature = "support";
      state.featureNames = null;
      labKlent = null;
      buildFeatList();
      refreshModule();
    },
  });
}

async function loadBots(preferredId = null) {
  try {
    state.bots = normalizeCheckpoints(await requestJson("/api/bots"));
  } catch (_) {
    state.bots = [];
    setStatus("evalStatus", "server unreachable — live modules unavailable", true);
    return;
  }
  const def = state.bots.find(c => c.id === preferredId) || defaultCheckpoint(state.bots);
  state.ckpt = def ? def.id : null;
  state.ckptLabel = def ? def.label : "";
  state.ckptFamily = def ? def.family : "shrimp";
  buildFeatList();
  renderCkpts();
}

// ---- module switching --------------------------------------------------------------------

$("modSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b || b.dataset.mod === state.module) return;
  cancelBot();
  state.module = b.dataset.mod;
  segSelect($("modSeg"), x => x.dataset.mod === state.module);
  document.querySelectorAll(".mod").forEach(m => {
    m.classList.toggle("active", m.id === "mod-" + state.module);
  });
  syncOvlUI();
  refreshModule();
});

/* The status line each module reports into. features computes in the page and
 * says everything in the readout, so it has none. */
const MOD_STATUS = {
  eval: "evalStatus", heads: "headsStatus", attention: "attnStatus",
  activations: "actStatus", search: "searchStatus",
};

function setStatus(id, msg, isErr = false) {
  const el = $(id);
  el.textContent = msg || "";
  el.className = "mod-status" + (isErr ? " err" : "");
}

function setModuleStatus(msg, isErr = false) {
  const id = MOD_STATUS[state.module];
  if (id) setStatus(id, msg, isErr);
}

// ---- server eval (cached, debounced, rate-limit aware) -------------------------------------

function wantsKey(wants) {
  return (wants.attention_query ? "a" + wants.attention_query.q + "," + wants.attention_query.r : "") +
         (wants.activations ? "|act" : "") + (wants.features ? "|feat" : "");
}

function cachedEval(k, body) {
  if (!state.ckpt) return Promise.reject({ status: 0, message: "no checkpoint catalogue" });
  if (Date.now() < state.rlUntil) {
    return Promise.reject({ status: 429, message: "rate-limited — try again shortly" });
  }
  if (state.evalCache.has(k)) return state.evalCache.get(k);
  const prom = requestJson("/api/lab/eval", body).catch(e => {
    state.evalCache.delete(k);
    if (e.status === 429) state.rlUntil = Date.now() + 15000;
    throw e;
  });
  state.evalCache.set(k, prom);
  // Large enough for a long line's whole value trace plus the module reads.
  if (state.evalCache.size > 240) {
    state.evalCache.delete(state.evalCache.keys().next().value);
  }
  return prom;
}

function fetchEval(wants = {}) {
  const k = state.ckpt + "|" + posKey() + "|" + wantsKey(wants);
  const body = { checkpoint_id: state.ckpt, ...positionBody() };
  if (wants.attention_query || wants.activations || wants.features) body.wants = wants;
  return cachedEval(k, body);
}

/* Eval one prefix of a line — the value graph's fetch. The cache key matches
 * fetchEval's for the same position, so the graph, the modules and the step
 * cursor all share one entry per position. */
function fetchEvalAt(moves) {
  const k = state.ckpt + "|s:" + moves.map(m => m.join(",")).join(";") + "|";
  return cachedEval(k, {
    checkpoint_id: state.ckpt,
    actions: moves.map(([q, r]) => ({ q, r })),
  });
}

/* Everything the board carries beyond the stones: module overlays, the heat
 * layer the eval/heads maps and the search replay share, and the solver's
 * proof-line previews. All of it describes one position, so all of it comes
 * off together the moment that position changes. */
function clearBoardPaint() {
  clearOverlay();
  board.resetLiveSearch();
  board.clearPreviewStones();
}

let refreshT = null;
function scheduleRefresh() {
  clearTimeout(refreshT);
  refreshT = setTimeout(() => { refreshT = null; refreshModule(); }, 350);
}

/* Run a waiting refresh now. A tool that takes the board over — the solver,
 * a bot move — calls this first, so the panel it paints over is never one
 * position behind. */
function flushRefresh() {
  if (refreshT === null) return;
  clearTimeout(refreshT);
  refreshT = null;
  refreshModule();
}

/* An async module render is stale when the position moved on, another module
 * took the panel, or the bot took the board. */
const staleRender = (k, mod) =>
  posKey() !== k || state.module !== mod || state.bot.busy;

function positionChanged() {
  clearStage();
  clearBoardPaint();
  labKlent = null;
  renderPosition();
  syncValueGraph();
  history.replaceState(null, "", location.pathname + location.search + shareHash());
  // drop an attention query that left the support
  if (state.attnCell) {
    const sup = LF.buildSupport(stoneList().map(s => [s.q, s.r]));
    if (!sup.index.has(key(state.attnCell[0], state.attnCell[1]))) state.attnCell = null;
  }
  scheduleRefresh();
}

// ---- shared position rendering ---------------------------------------------------------------

function renderPosition() {
  const stones = stoneList();
  // Six in a line ends the sequence. The color comes from the winning run
  // itself rather than the turn order, so the outline and the name agree.
  const winCells = state.mode === "sequence" ? findWin(stones) : null;
  const winStone = winCells
    ? stones.find(s => s.q === winCells[0].q && s.r === winCells[0].r) : null;
  termWinner = winStone ? winStone.color : null;
  board.setStones(stones, winCells);
  board.setLegal(
    state.mode === "sequence" && state.module !== "attention" && termWinner === null
      ? LF.legalCells(curMoves()).map(([q, r]) => ({ q, r }))
      : null,
  );
  const facts = currentFacts();
  $("mgStones").textContent = stones.length;
  const mg = $("mgToMove");
  if (termWinner !== null) {
    mg.textContent = "over";
    mg.className = "n";
  } else {
    const tm = facts.currentPlayer;
    mg.textContent = tm === 0 ? "blue" : "red";
    mg.className = "n " + (tm === 0 ? "is-p0" : "is-p1");
  }
  $("mgPhase").textContent =
    facts.phase === "Opening" ? "opening" :
    facts.phase === "SecondStone" ? "2nd stone" : "1st stone";
  if (state.mode === "sequence") renderLineStrip();
  syncPlayUI();
}

// ---- module: features (client-side) ------------------------------------------------------------

function buildFeatList(names = LF.FEATURE_NAMES, zeroedNames = LF.FREE_ZEROED) {
  const list = $("featList");
  list.textContent = "";
  const mk = (val, label, cls = "") => {
    const b = document.createElement("button");
    b.dataset.feat = val;
    b.innerHTML = label;
    if (cls) b.className = cls;
    list.appendChild(b);
  };
  mk("support", "support set");
  for (const name of names) {
    const zeroed = zeroedNames.includes(name);
    mk(name, name.replace(/_/g, " ") + (zeroed ? ' <span class="fz" title="zeroed in free edit">&deg;</span>' : ""));
  }
  segFeat();
}

function segFeat() {
  $("featList").querySelectorAll("button").forEach(b => {
    b.classList.toggle("sel", b.dataset.feat === state.feature);
  });
}

$("featList").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  state.feature = b.dataset.feat;
  segFeat();
  renderFeatures();
});

function renderFeatures() {
  if (state.ckptFamily === "hexfield_eq") {
    const k = posKey();
    setReadout("features", "computing family-specific input planes...");
    fetchEval({ features: true }).then(payload => {
      if (staleRender(k, "features")) return;
      renderServerFeatures(payload);
    }).catch(e => {
      if (staleRender(k, "features")) return;
      setReadout("features", e.message || "feature fetch failed");
    });
    return;
  }
  clearOverlay();
  const facts = currentFacts();
  const sup = LF.buildSupport(facts.records.map(rec => [rec.q, rec.r]));
  const halo = sup.coords.slice(sup.legalCount + sup.stoneCount);
  const mover = facts.currentPlayer;
  if (state.feature === "support") {
    paintFill(
      sup.coords.slice(0, sup.legalCount).map(([q, r]) => ({ q, r, v: 1 })),
      ACCENT, 0.18,
    );
    paintHalo(halo);
    setReadout("support set",
      `${sup.coords.length} nodes · ${sup.legalCount} legal + ${sup.stoneCount} stones + ` +
      `${sup.haloCount} halo · legal = empty cells within distance 4 of a stone`);
    return;
  }
  const f = LF.FEATURE_NAMES.indexOf(state.feature);
  const planes = LF.buildFeatures(facts, sup);
  const vals = planes[f];
  const own = state.feature.startsWith("own");
  const opp = state.feature.startsWith("opp");
  const color = own ? (mover === 0 ? H0 : H1) : opp ? (mover === 0 ? H1 : H0) : ACCENT;
  paintFill(sup.coords.map(([q, r], i) => ({ q, r, v: vals[i] })), color, 0.55);
  paintHalo(halo);
  const nonzero = vals.filter(v => v > 0).length;
  const max = Math.max(0, ...vals);
  const zeroNote = facts.free && LF.FREE_ZEROED.includes(state.feature)
    ? " · zeroed in free edit" : "";
  setReadout(
    "feature · " + state.feature.replace(/_/g, " "),
    `${nonzero} of ${vals.length} cells nonzero · max ${max.toFixed(3)}` +
    ` · own/opp are relative to the side to move (${mover === 0 ? "blue" : "red"})` +
    zeroNote,
  );
}

function renderServerFeatures(payload) {
  clearOverlay();
  const feat = payload.features;
  if (!feat || !Array.isArray(feat.names) || !Array.isArray(feat.planes)) {
    setReadout("features", "this family did not return feature planes");
    return;
  }
  const sig = feat.names.join("|");
  if (state.featureNames !== sig) {
    state.featureNames = sig;
    if (state.feature !== "support" && !feat.names.includes(state.feature)) {
      state.feature = "support";
    }
    buildFeatList(feat.names, payload.zeroed_features || LF.FREE_ZEROED);
  }
  const sup = payload.support;
  const coords = sup.coords;
  const halo = coords.slice(sup.legal_count + sup.stone_count);
  if (state.feature === "support") {
    paintFill(
      coords.slice(0, sup.legal_count).map(([q, r]) => ({ q, r, v: 1 })),
      ACCENT, 0.18,
    );
    paintHalo(halo);
    setReadout(
      "support set",
      `${coords.length} nodes Â· ${sup.legal_count} legal + ${sup.stone_count} stones + ` +
      `${sup.halo_count} halo Â· ${feat.names.length} ${state.ckptFamily} input planes`,
    );
    return;
  }
  const f = feat.names.indexOf(state.feature);
  if (f < 0) return;
  const vals = feat.planes[f];
  const mover = payload.to_move;
  const own = state.feature.startsWith("own");
  const opp = state.feature.startsWith("opp");
  const color = own ? (mover === 0 ? H0 : H1) : opp ? (mover === 0 ? H1 : H0) : ACCENT;
  paintFill(coords.map(([q, r], i) => ({ q, r, v: vals[i] })), color, 0.55);
  paintHalo(halo);
  const finite = vals.filter(v => typeof v === "number");
  const nonzero = finite.filter(v => v > 0).length;
  const max = Math.max(0, ...finite);
  const zeroNote = payload.mode === "free" &&
    (payload.zeroed_features || []).includes(state.feature) ? " Â· zeroed in free edit" : "";
  setReadout(
    "feature Â· " + state.feature.replace(/_/g, " "),
    `${nonzero} of ${vals.length} cells nonzero Â· max ${max.toFixed(3)} Â· ` +
    `own/opp are relative to ${mover === 0 ? "blue" : "red"}${zeroNote}`,
  );
}

function setReadout(k, t) {
  $("roK").textContent = k;
  $("roT").textContent = t;
}

// ---- module: net eval ---------------------------------------------------------------------------

let evalHead = "policy";
$("headSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  evalHead = b.dataset.head;
  segSelect($("headSeg"), x => x === b);
  renderEval();
});

function setBig(el, v, cls) {
  if (v === null || v === undefined) {
    el.textContent = "—";
    el.className = cls;
    return;
  }
  el.textContent = fmtV(v);
  el.className = cls + " " + (Math.abs(v) < 0.08 ? "" : v >= 0 ? "pos" : "neg");
}

function renderDist(dist, value) {
  const svg = $("distChart");
  svg.textContent = "";
  // A family that serves no bin distribution loses the whole section: an
  // empty chart under a "65 bins" heading would read as a broken readout.
  $("distSec").hidden = !dist;
  if (!dist) return;
  const W = 300, H = 72, B = 12, T = 4;
  const max = Math.max(...dist, 1e-9);
  const bw = W / dist.length;
  const curBin = Math.round((value + 1) * (dist.length - 1) / 2);
  for (let i = 0; i < dist.length; i++) {
    const h = (H - T - B) * dist[i] / max;
    const el = document.createElementNS(NS, "rect");
    el.setAttribute("x", (i * bw + 0.5).toFixed(2));
    el.setAttribute("y", (H - B - h).toFixed(2));
    el.setAttribute("width", Math.max(0.5, bw - 1).toFixed(2));
    el.setAttribute("height", Math.max(0, h).toFixed(2));
    el.setAttribute("class", "db" + (i === curBin ? " cur" : ""));
    svg.appendChild(el);
  }
  const ax = document.createElementNS(NS, "line");
  ax.setAttribute("x1", 0); ax.setAttribute("y1", H - B);
  ax.setAttribute("x2", W); ax.setAttribute("y2", H - B);
  ax.setAttribute("class", "ax");
  svg.appendChild(ax);
  for (const [frac, label] of [[0, "−1"], [0.5, "0"], [1, "+1"]]) {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", frac === 0 ? 1 : frac === 1 ? W - 12 : W / 2 - 3);
    t.setAttribute("y", H - 2);
    t.textContent = label;
    svg.appendChild(t);
  }
}

function renderEvalPayload(payload) {
  const tm = payload.to_move;
  const flip = v => (typeof v !== "number" ? null : tm === 0 ? v : -v);
  const v = flip(payload.value);
  setBig($("valNow"), v, "value-big");
  $("valWho").textContent =
    v === null ? "" : Math.abs(v) < 0.08 ? "even" : v >= 0 ? "blue better" : "red better";
  const stv = payload.stv || {};
  setBig($("stv2"), flip(stv["2"]), "hz-v");
  setBig($("stv6"), flip(stv["6"]), "hz-v");
  setBig($("stv16"), flip(stv["16"]), "hz-v");
  // A family that serves no moves-left head says so: "~0" would read as a
  // prediction the net never made.
  $("mlNow").textContent = typeof payload.moves_left === "number"
    ? "~" + Math.max(0, Math.round(payload.moves_left)) : "—";
  renderDist(payload.value_dist, payload.value);

  const rows = payload[evalHead] || [];
  const headMover = evalHead === "opp_policy" ? 1 - tm : tm;
  paintOverlay(payload, rows, headMover);

  const list = $("topList");
  list.textContent = "";
  for (const row of rows.slice(0, 5)) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="rk-c"></span><span class="rk-v"></span>`;
    li.querySelector(".rk-c").textContent = fmtCell(row.q, row.r);
    li.querySelector(".rk-v").textContent = pct(row.p);
    list.appendChild(li);
  }
  setReadout(
    "net eval · " + state.ckptLabel,
    // In policy mode the board carries the head the selector picked, so the
    // readout names that head rather than the overlay mode.
    `${state.ovl === "policy" ? evalHead.replace("_", " ") : OVL_READ[state.ovl]}` +
    ` over ${payload.legal_count} legal cells · ` +
    `value ${v === null ? "—" : fmtV(v)} (blue POV) · no search`,
  );
}

function renderEval() {
  setStatus("evalStatus", "computing…");
  const k = posKey();
  fetchEval({}).then(payload => {
    if (staleRender(k, "eval")) return;
    setStatus("evalStatus", "");
    renderEvalPayload(payload);
  }).catch(e => {
    if (staleRender(k, "eval")) return;
    setStatus("evalStatus", e.message || "eval failed", true);
  });
}

// ---- board overlays for the eval and heads modules -----------------------------------------

/* The mantis `klent` block as a map for the hover text and the overlays.
 * Every value stays side-to-move POV as served; the painters flip to the
 * site's blue/red frame where the mode calls for it. */
function klentCells(payload) {
  const k = payload && payload.klent;
  if (!k || !Array.isArray(k.coords) || !Array.isArray(k.q)) return null;
  const arr = (name, i) => {
    const v = Array.isArray(k[name]) ? k[name][i] : null;
    return Number.isFinite(v) ? v : 0;
  };
  const map = new Map();
  k.coords.forEach((c, i) => {
    if (!Array.isArray(c) || !Number.isFinite(c[0]) || !Number.isFinite(c[1])) return;
    map.set(key(c[0], c[1]), {
      qc: c[0], rc: c[1],
      q: arr("q", i), prior: arr("prior", i), improved: arr("improved", i),
      win: arr("win", i), loss: arr("loss", i), long: arr("long", i),
    });
  });
  return map.size ? map : null;
}

const OVL_READ = {
  policy: "policy priors", q: "critic Q per move", pi2: "improved policy π′",
  long: "long-game mass", gap: "critic against policy",
};

function syncOvlUI() {
  const shown = state.module === "eval" || state.module === "heads";
  $("ovlStrip").hidden = !shown;
  $("ovlSeg").querySelectorAll("button").forEach(b => {
    const on = b.dataset.ovl === state.ovl;
    b.classList.toggle("sel", on);
    b.setAttribute("aria-checked", on ? "true" : "false");
    // Every mode past policy reads the critic block, so a family that serves
    // none disables them rather than painting the prior under another name.
    b.disabled = b.dataset.ovl !== "policy" && !labKlent;
  });
}

$("ovlSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b || b.disabled || b.dataset.ovl === state.ovl) return;
  state.ovl = b.dataset.ovl;
  syncOvlUI();
  refreshModule();
});

/* Paint the board for the module in front. `policyRows` is the distribution
 * that module calls its policy (the eval module's head selector picks one),
 * `policyMover` the side it belongs to. Every other mode reads the critic. */
function paintOverlay(payload, policyRows, policyMover) {
  labKlent = klentCells(payload);
  if (!labKlent && state.ovl !== "policy") state.ovl = "policy";
  syncOvlUI();
  clearOverlay();
  const mover = payload.to_move;
  const tint = mover === 0 ? H0 : H1, ring = mover === 0 ? H0R : H1R;
  if (state.ovl === "policy" || !labKlent) {
    const rows = Array.isArray(policyRows) ? policyRows : [];
    if (rows.length) {
      board.setHeat(rows, policyMover === 0 ? H0 : H1,
                    policyMover === 0 ? H0R : H1R, OV_OPA);
    } else {
      board.clearHeat();
    }
    return;
  }
  const cells = [...labKlent.values()];
  if (state.ovl === "pi2") {
    board.setHeat(
      cells.map(c => ({ q: c.qc, r: c.rc, p: c.improved })).sort((a, b) => b.p - a.p),
      tint, ring, OV_OPA,
    );
    return;
  }
  if (state.ovl === "long") {
    board.setHeat(
      cells.map(c => ({ q: c.qc, r: c.rc, p: c.long })).sort((a, b) => b.p - a.p),
      LONG_TINT, LONG_TINT, OV_OPA,
    );
    return;
  }
  if (state.ovl === "q") {
    // Q is win mass minus loss mass, so the scale is absolute — a faint board
    // IS the critic seeing a close game. Blue/red is the site's value frame;
    // the ring marks the best cell for the side to move.
    const f = mover === 0 ? 1 : -1;
    let best = null;
    for (const c of cells) if (!best || c.q > best.q) best = c;
    board.setSignedHeat(
      cells.map(c => ({ q: c.qc, r: c.rc, v: f * c.q })), H0, H1, OV_OPA,
      best ? { q: best.qc, r: best.rc } : null, ring,
    );
    return;
  }
  // gap: where the critic bends the policy. π′ − π per cell, normalized to
  // the largest shift on the board so the map reads at any sharpness.
  let scale = 0;
  for (const c of cells) scale = Math.max(scale, Math.abs(c.improved - c.prior));
  board.setSignedHeat(
    cells.map(c => ({
      q: c.qc, r: c.rc, v: scale > 0 ? (c.improved - c.prior) / scale : 0,
    })),
    GAP_POS, GAP_NEG, OV_OPA, null, ring,
  );
}

// ---- module: heads (the served policy, critic and improved policy) --------------------------

const HEAD_ROWS = 24;

/* Ring one cell for a moment. The table rows use it to point at the board;
 * the next module paint clears the overlay layer it lives on. */
function flashCell(q, r) {
  const el = ovPoly(q, r, S * 0.86, "ov-flash");
  setTimeout(() => el.remove(), 1400);
}

$("headRows").addEventListener("click", e => {
  const tr = e.target.closest("tr[data-q]");
  if (tr) flashCell(+tr.dataset.q, +tr.dataset.r);
});

function renderHeadsPayload(payload) {
  const tm = payload.to_move;
  const v = typeof payload.value === "number" ? (tm === 0 ? payload.value : -payload.value) : null;
  setBig($("hdValue"), v, "hs-v");
  paintOverlay(payload, payload.policy || [], tm);
  const kl = payload.klent || {};
  $("hdKl").textContent = Number.isFinite(kl.kl) ? kl.kl.toFixed(3) : "—";
  $("hdEnt").textContent = Number.isFinite(kl.norm_entropy)
    ? kl.norm_entropy.toFixed(3) : "—";
  const body = $("headRows");
  body.textContent = "";
  if (!labKlent) {
    $("headTable").hidden = true;
    $("headMore").textContent = "";
    $("headNote").hidden = false;
    setReadout("heads · " + state.ckptLabel,
      "this family serves a policy and a value only. it has no critic read.");
    return;
  }
  $("headNote").hidden = true;
  $("headTable").hidden = false;
  const cells = [...labKlent.values()].sort((a, b) => b.improved - a.improved);
  for (const c of cells.slice(0, HEAD_ROWS)) {
    const tr = document.createElement("tr");
    tr.dataset.q = c.qc;
    tr.dataset.r = c.rc;
    tr.innerHTML =
      `<td class="hd-c"></td><td></td><td></td><td class="hd-q"></td>` +
      `<td class="hd-bar"><span class="wl w"></span><span class="wl l"></span>` +
      `<span class="wl g"></span></td>`;
    const td = tr.children;
    td[0].textContent = fmtCell(c.qc, c.rc);
    td[1].textContent = pct(c.prior);
    td[2].textContent = pct(c.improved);
    td[3].textContent = fmtV(c.q);
    // Sign in the mover's frame, color in the site's: a move that is good for
    // red carries red, whichever side is to move.
    td[3].classList.add((tm === 0 ? c.q : -c.q) >= 0 ? "is-p0" : "is-p1");
    const bar = td[4].children;
    bar[0].style.width = pct(c.win);
    bar[1].style.width = pct(c.loss);
    bar[2].style.width = pct(c.long);
    td[4].title = `win ${pct(c.win, 0)} · loss ${pct(c.loss, 0)} · long ${pct(c.long, 0)}`;
    body.appendChild(tr);
  }
  const rest = cells.length - HEAD_ROWS;
  $("headMore").textContent = rest > 0 ? `… ${rest} more cells` : "";
  setReadout(
    "heads · " + state.ckptLabel,
    `${OVL_READ[state.ovl]} over ${cells.length} legal cells · ` +
    `KL(π′‖π) ${Number.isFinite(kl.kl) ? kl.kl.toFixed(3) : "—"} · ` +
    "point to a cell for its full read",
  );
}

function renderHeads() {
  setStatus("headsStatus", "computing…");
  const k = posKey();
  fetchEval({}).then(payload => {
    if (staleRender(k, "heads")) return;
    setStatus("headsStatus", "");
    renderHeadsPayload(payload);
  }).catch(e => {
    if (staleRender(k, "heads")) return;
    setStatus("headsStatus", e.message || "eval failed", true);
  });
}

// ---- module: attention ----------------------------------------------------------------------------

function pickAttnQuery(q, r) {
  const sup = LF.buildSupport(stoneList().map(s => [s.q, s.r]));
  if (!sup.index.has(key(q, r))) {
    toast("pick a support cell (a stone, a legal cell, or the halo)");
    return;
  }
  state.attnCell = [q, r];
  renderAttention();
}

function buildAttnSegs(blocks, heads) {
  const blockSeg = $("blockSeg");
  if (blockSeg.children.length !== blocks) {
    blockSeg.textContent = "";
    for (let i = 0; i < blocks; i++) {
      const b = document.createElement("button");
      b.dataset.i = i;
      b.textContent = "A" + (i + 1);
      b.setAttribute("role", "radio");
      blockSeg.appendChild(b);
    }
    state.attnBlock = Math.min(state.attnBlock, blocks - 1);
  }
  const headSeg = $("headSegAttn");
  if (headSeg.children.length !== heads) {
    headSeg.textContent = "";
    for (let i = 0; i < heads; i++) {
      const b = document.createElement("button");
      b.dataset.i = i;
      b.textContent = "h" + i;
      b.setAttribute("role", "radio");
      headSeg.appendChild(b);
    }
    state.attnHead = Math.min(state.attnHead, heads - 1);
  }
  segSelect(blockSeg, b => +b.dataset.i === state.attnBlock);
  segSelect(headSeg, b => +b.dataset.i === state.attnHead);
}

$("blockSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  state.attnBlock = +b.dataset.i;
  renderAttention();
});
$("headSegAttn").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  state.attnHead = +b.dataset.i;
  renderAttention();
});

function renderTokens(tokens) {
  const svg = $("tokChart");
  svg.textContent = "";
  if (!tokens) return;
  const W = 300, H = 56, B = 14, T = 4;
  const max = Math.max(...tokens, 1e-9);
  const bw = W / tokens.length;
  tokens.forEach((w, i) => {
    const h = (H - T - B) * w / max;
    const el = document.createElementNS(NS, "rect");
    el.setAttribute("x", (i * bw + 4).toFixed(2));
    el.setAttribute("y", (H - B - h).toFixed(2));
    el.setAttribute("width", (bw - 8).toFixed(2));
    el.setAttribute("height", Math.max(0, h).toFixed(2));
    el.setAttribute("class", "tb");
    svg.appendChild(el);
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", (i * bw + bw / 2 - 4).toFixed(2));
    t.setAttribute("y", H - 3);
    t.textContent = "t" + i;
    svg.appendChild(t);
  });
  const ax = document.createElementNS(NS, "line");
  ax.setAttribute("x1", 0); ax.setAttribute("y1", H - B);
  ax.setAttribute("x2", W); ax.setAttribute("y2", H - B);
  ax.setAttribute("class", "ax");
  svg.appendChild(ax);
}

function renderAttention() {
  clearOverlay();
  if (state.ckptFamily === "hexfield_eq") {
    renderTokens(null);
    $("attnList").textContent = "";
    fetchEval({}).then(payload => {
      if (state.module !== "attention") return;
      const marker = payload.attention;
      const reason = marker && marker.available === false
        ? marker.reason : "attention visualization unavailable for this family";
      setStatus("attnStatus", reason);
      setReadout("attention unavailable", reason);
    }).catch(e => {
      if (state.module === "attention") {
        setStatus("attnStatus", e.message || "attention fetch failed", true);
      }
    });
    return;
  }
  if (!state.attnCell) {
    renderTokens(null);
    $("attnList").textContent = "";
    setStatus("attnStatus", "");
    setReadout("attention", "tap any support cell on the board to set the query.");
    return;
  }
  const [q, r] = state.attnCell;
  paintRing(q, r, null, "ov-query");
  setStatus("attnStatus", "computing…");
  const k = posKey();
  fetchEval({ attention_query: { q, r } }).then(payload => {
    if (staleRender(k, "attention")) return;
    if (!state.attnCell || state.attnCell[0] !== q || state.attnCell[1] !== r) return;
    setStatus("attnStatus", "");
    const attn = payload.attention;
    if (!attn || attn.available === false) {
      const reason = attn && attn.reason ? attn.reason : "attention unavailable";
      setStatus("attnStatus", reason);
      setReadout("attention unavailable", reason);
      return;
    }
    buildAttnSegs(attn.blocks, attn.heads);
    const row = attn.rows[state.attnBlock][state.attnHead];
    const coords = payload.support.coords;
    const cells = Object.entries(row.cells)
      .map(([node, w]) => ({ q: coords[+node][0], r: coords[+node][1], v: w }))
      .sort((a, b) => b.v - a.v);
    clearOverlay();
    paintFill(cells, ACCENT, 0.7);
    paintRing(q, r, null, "ov-query");
    renderTokens(row.tokens);
    const list = $("attnList");
    list.textContent = "";
    for (const c of cells.slice(0, 8)) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="rk-c"></span><span class="rk-v"></span>`;
      li.querySelector(".rk-c").textContent = fmtCell(c.q, c.r);
      li.querySelector(".rk-v").textContent = (c.v * 100).toFixed(1) + "%";
      list.appendChild(li);
    }
    const tokenMass = row.tokens.reduce((a, b) => a + b, 0);
    setReadout(
      `attention · block ${state.attnBlock + 1} head ${state.attnHead} · query ${fmtCell(q, r)}`,
      `${cells.length} cells above ${attn.floor} · ` +
      `${(tokenMass * 100).toFixed(1)}% of the row on the ${row.tokens.length} summary token${row.tokens.length === 1 ? "" : "s"}`,
    );
  }).catch(e => {
    if (staleRender(k, "attention")) return;
    setStatus("attnStatus", e.message || "attention fetch failed", true);
  });
}

// ---- module: activations -----------------------------------------------------------------------------

let actPayload = null; // last activations payload for the current position

$("actRange").addEventListener("input", function () {
  state.actStage = +this.value;
  renderActStage();
});

function renderActStage() {
  if (!actPayload || state.module !== "activations") return;
  const blocks = actPayload.activations.blocks;
  const stage = blocks[clamp(0, blocks.length - 1, state.actStage)];
  $("actLabel").textContent = stage.label;
  clearOverlay();
  const coords = actPayload.support.coords;
  const rows = stage.norms.map((v, i) => ({ q: coords[i][0], r: coords[i][1], v }));
  paintFill(rows, ACCENT, 0.65);
  const ranked = rows.slice().sort((a, b) => b.v - a.v);
  const list = $("actList");
  list.textContent = "";
  for (const c of ranked.slice(0, 6)) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="rk-c"></span><span class="rk-v"></span>`;
    li.querySelector(".rk-c").textContent = fmtCell(c.q, c.r);
    li.querySelector(".rk-v").textContent = c.v.toFixed(2);
    list.appendChild(li);
  }
  const max = ranked.length ? ranked[0].v : 0;
  const mean = rows.length ? rows.reduce((a, b) => a + b.v, 0) / rows.length : 0;
  setReadout(
    `activation flow · ${stage.label} (${state.actStage + 1}/${blocks.length})`,
    `per-cell L2 norm · max ${max.toFixed(2)} · mean ${mean.toFixed(2)} · ` +
    `${stage.kind === "attn" ? "attention block output" : stage.kind === "conv" ? "conv block output" : "stem output"}`,
  );
}

function renderActivations() {
  setStatus("actStatus", "computing…");
  const k = posKey();
  fetchEval({ activations: true }).then(payload => {
    if (staleRender(k, "activations")) return;
    if (!payload.activations || payload.activations.available === false) {
      const reason = payload.activations && payload.activations.reason
        ? payload.activations.reason : "activation visualization unavailable";
      setStatus("actStatus", reason);
      setReadout("activations unavailable", reason);
      return;
    }
    setStatus("actStatus", "");
    actPayload = payload;
    const n = payload.activations.blocks.length;
    const range = $("actRange");
    range.max = n - 1;
    state.actStage = clamp(0, n - 1, state.actStage);
    range.value = state.actStage;
    renderActStage();
  }).catch(e => {
    if (staleRender(k, "activations")) return;
    setStatus("actStatus", e.message || "activations fetch failed", true);
  });
}

// ---- module: search -----------------------------------------------------------------------------------

$("simsSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  state.sims = +b.dataset.sims;
  segSelect($("simsSeg"), x => x === b);
  syncPlayUI();
});

$("searchBtn").addEventListener("click", runSearch);

function renderSearchPayload(payload) {
  const tm = currentToMove();
  const v = tm === 0 ? payload.root_value : -payload.root_value;
  setBig($("searchVal"), v, "value-big");
  $("searchCap").textContent =
    `${payload.visits} visits · best ${fmtCell(payload.best.q, payload.best.r)}`;
  clearOverlay();
  const rows = payload.visit_policy.map(h => ({ q: h.q, r: h.r, v: h.p }));
  paintFill(rows, tm === 0 ? H0 : H1, 0.62);
  paintRing(payload.best.q, payload.best.r, tm === 0 ? H0R : H1R);
  const list = $("visitList");
  list.textContent = "";
  for (const row of payload.visit_policy.slice(0, 8)) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="rk-c"></span><span class="rk-v"></span>`;
    li.querySelector(".rk-c").textContent = fmtCell(row.q, row.r);
    li.querySelector(".rk-v").textContent = (row.p * 100).toFixed(1) + "%";
    li.title = "search weight " + row.w;
    list.appendChild(li);
  }
  setReadout(
    `search · ${payload.sims} visits · ${state.ckptLabel}`,
    `searched value ${fmtV(v)} (blue POV) · chosen move ${fmtCell(payload.best.q, payload.best.r)} · ` +
    `as-trained profile, greedy selection`,
  );
}

function runSearch() {
  if (state.mode !== "sequence" || termWinner !== null || state.bot.busy) return;
  if (!state.ckpt) { setStatus("searchStatus", "no checkpoint catalogue", true); return; }
  if (Date.now() < state.rlUntil) {
    setStatus("searchStatus", "rate-limited — try again shortly", true);
    return;
  }
  const k = state.ckpt + "|" + posKey() + "|" + state.sims;
  const cached = state.searchCache.get(k);
  if (cached) { renderSearchPayload(cached); setStatus("searchStatus", "cached"); return; }
  setStatus("searchStatus", `searching · ${state.sims} visits…`);
  $("searchBtn").disabled = true;
  const pk = posKey();
  requestJson("/api/lab/search", {
    checkpoint_id: state.ckpt,
    sims: state.sims,
    ...positionBody(),
  }).then(payload => {
    state.searchCache.set(k, payload);
    if (state.searchCache.size > 40) {
      state.searchCache.delete(state.searchCache.keys().next().value);
    }
    if (staleRender(pk, "search")) return;
    setStatus("searchStatus", "");
    renderSearchPayload(payload);
  }).catch(e => {
    if (e.status === 429) state.rlUntil = Date.now() + 15000;
    if (staleRender(pk, "search")) return;
    setStatus("searchStatus", e.message || "search failed", true);
  }).finally(() => {
    syncPlayUI();
  });
}

function renderSearchModule() {
  const free = state.mode !== "sequence";
  syncPlayUI();
  if (free) {
    clearOverlay();
    setStatus("searchStatus", "search needs a legal sequence — free-edit positions cannot be replayed", true);
    setReadout("search", "switch to legal-sequence mode to run a search.");
    return;
  }
  const k = state.ckpt + "|" + posKey() + "|" + state.sims;
  const cached = state.searchCache.get(k);
  if (cached) { renderSearchPayload(cached); setStatus("searchStatus", "cached"); }
  else {
    clearOverlay();
    setBig($("searchVal"), null, "value-big");
    $("searchCap").textContent = "";
    $("visitList").textContent = "";
    setStatus("searchStatus", "");
    setReadout("search", `press run search for a real ${state.sims}-visit search on this position.`);
  }
}

// ---- the forced-win solver ---------------------------------------------------------------------------

$("solveBtn").addEventListener("click", runSolve);

/* λ¹ guard classes over the root's legal moves, under the proof line: a
 * refuted move takes a dark veil, a move that wins at once takes a ring. */
function paintGuard(guard) {
  if (!Array.isArray(guard)) return;
  for (const g of guard) {
    if (!Number.isFinite(g.q) || !Number.isFinite(g.r)) continue;
    if (g.cls === -1) ovPoly(g.q, g.r, S * 0.975, "ov-veil");
    else if (g.cls === 1) paintRing(g.q, g.r, null, "ov-guard");
  }
}

/* The verdict for the side to move, plus the proof line as preview stones.
 * Line colors follow the engine's two-stone turn structure from this ply:
 * the line's i-th move is placement record `ply + i`, and the proven first
 * move pulses. */
function renderSolveResult(res, mover, ply, budgets) {
  const side = mover === 0 ? "blue" : "red";
  const nodes = Number.isFinite(res.nodes) && res.nodes > 0
    ? ` · ${res.nodes.toLocaleString()} nodes` : "";
  const ms = Number.isFinite(res.ms) ? ` · ${(res.ms / 1000).toFixed(1)}s` : "";
  clearOverlay();
  board.clearPreviewStones();
  paintGuard(res.guard);
  if (res.status === "win") {
    const walked = Array.isArray(res.line) ? res.line : [];
    const line = walked.length ? walked : (res.proven ? [res.proven] : []);
    board.setPreviewStones(line.map((c, i) => ({
      q: c.q, r: c.r, color: LF.recordPlayer(ply + i), tss: i === 0,
    })));
    const ft = Number.isFinite(res.forced_through) ? res.forced_through : 0;
    const lineRead = line.length > 1
      ? ` · ${line.length}-ply line (${ft >= line.length
          ? "forced to the end" : `defense forced through ${ft}`})`
      : "";
    setStatus("searchStatus", "");
    setReadout("solver · forced win",
      `${side} wins by force — proven${lineRead}${nodes}${ms}`);
    // The proof becomes a line in the strip, forked at the solve ply, so the
    // step controls walk it, and it carries the solve summary for review.
    // The board keeps the verdict paint until the user navigates. A line the
    // strip already holds is selected, not duplicated.
    if (line.length) {
      const forked = curMoves().concat(line.map(c => [c.q, c.r]));
      let idx = state.lines.findIndex(L => L.length === forked.length
        && L.every((m, i) => m[0] === forked[i][0] && m[1] === forked[i][1]));
      if (idx === -1) {
        state.lines.push(forked);
        state.lineNotes.push(null);
        // the proof line opens at the solve ply — stepping forward walks it
        state.lineCursors.push(state.cursor);
        idx = state.lines.length - 1;
      }
      if (idx !== state.lineIdx) state.lineCursors[state.lineIdx] = state.cursor;
      const budgetRead = budgets
        ? ` · budgets ${budgets.node_cap.toLocaleString()} nodes / ${(budgets.budget_ms / 1000)}s / ${budgets.line_cap} ply`
        : "";
      state.lineNotes[idx] =
        `solved at ply ${ply}: ${side} wins by force — ${line.length}-ply line, `
        + `${ft >= line.length ? "forced to the end" : `defense forced through ${ft}`}`
        + `${nodes}${ms}${budgetRead}`;
      state.lineIdx = idx;
      renderLineStrip();
      toast(`proof line is line ${idx + 1} — step forward to walk it`);
    }
  } else if (res.status === "loss") {
    setStatus("searchStatus", "");
    setReadout("solver · lost",
      `${side} is lost here — forced threats beat every reply${nodes}${ms}`);
  } else if (res.status === "timeout") {
    setStatus("searchStatus", "");
    setReadout("solver · no answer",
      `the solver hit its clock before an answer${nodes}${ms}`);
  } else if (res.mem_stopped) {
    const peak = Number.isFinite(res.mem_peak_mb) && res.mem_peak_mb > 0
      ? ` at ${Math.round(res.mem_peak_mb)}MB` : "";
    setStatus("searchStatus", "");
    setReadout("solver · no answer",
      `the solver hit its memory ceiling${peak} before an answer${nodes}${ms}`);
  } else {
    setStatus("searchStatus", "");
    setReadout("solver · no forced win",
      `no forced win found for ${side}${nodes}${ms}`);
  }
}

/* The solver budgets, straight from the inputs. Out-of-range values are an
 * error here — the server would 422 them anyway, never clamp. */
function solverBudgets() {
  const nodes = Math.round(+$("slvNodes").value);
  const secs = +$("slvSecs").value;
  const line = Math.round(+$("slvLine").value);
  if (!(nodes >= 1000 && nodes <= 100000000)) return { err: "solver nodes must be 1000 to 100000000" };
  if (!(secs >= 0.25 && secs <= 600)) return { err: "solver time must be 0.25 to 600 seconds" };
  if (!(line >= 2 && line <= 100)) return { err: "proof line length must be 2 to 100" };
  return { node_cap: nodes, budget_ms: Math.round(secs * 1000), line_cap: line };
}

async function runSolve() {
  if (state.mode !== "sequence" || termWinner !== null || state.bot.busy) return;
  if (!state.ckpt) { setStatus("searchStatus", "no checkpoint catalogue", true); return; }
  if (Date.now() < state.rlUntil) {
    setStatus("searchStatus", "rate-limited — try again shortly", true);
    return;
  }
  const budgets = solverBudgets();
  if (budgets.err) {
    setStatus("searchStatus", budgets.err, true);
    return;
  }
  flushRefresh();
  const run = ++state.bot.run;
  state.bot.busy = true;
  syncPlayUI();
  clearBoardPaint();
  setStatus("searchStatus", "solver running · threat-space proof search");
  const moves = curMoves();
  const mover = currentToMove();
  try {
    const res = await requestJson("/api/lab/solve", {
      checkpoint_id: state.ckpt,
      actions: moves.map(([q, r]) => ({ q, r })),
      ...budgets,
    });
    if (state.bot.run !== run) return;
    renderSolveResult(res, mover, moves.length, budgets);
  } catch (e) {
    if (e.status === 429) state.rlUntil = Date.now() + 15000;
    if (state.bot.run !== run) return;
    setStatus("searchStatus", e.status === 429
      ? "solver rate-limited — wait a moment"
      : (e.message || "solver unavailable"), true);
  } finally {
    if (state.bot.run === run) {
      state.bot.busy = false;
      syncPlayUI();
    }
  }
}

// ---- bot play ------------------------------------------------------------------------------------------

const botPhaseEl = $("botPhase");
function setBotPhase(text) {
  botPhaseEl.textContent = text || "";
}

/* Orphan whatever the bot has in flight and stop autoplay. Every navigation
 * the user drives calls this: the frames on screen describe the position that
 * was on the board when the search ran, not the one replacing it. */
function cancelBot() {
  if (!state.bot.busy && !state.bot.auto) {
    setBotPhase("");
    return;
  }
  state.bot.run++;
  state.bot.busy = false;
  state.bot.auto = false;
  setBotPhase("");
  syncPlayUI();
  scheduleRefresh();
}

const canBotPlay = () =>
  state.mode === "sequence" && !!state.ckpt && termWinner === null &&
  !state.bot.busy && Date.now() >= state.rlUntil;

function syncPlayUI() {
  const seq = state.mode === "sequence";
  const idle = !state.bot.busy;
  $("botMoveBtn").disabled = !seq || !state.ckpt || termWinner !== null || !idle;
  $("autoBtn").disabled = !seq || !state.ckpt || (termWinner !== null && !state.bot.auto);
  $("autoBtn").textContent = state.bot.auto ? "stop" : "autoplay";
  $("autoBtn").classList.toggle("done", state.bot.auto);
  $("searchBtn").disabled = !seq || termWinner !== null || !idle;
  $("solveBtn").disabled = !seq || termWinner !== null || !idle;
}

/* The move the search settled on: the decision frame's action, or the payload
 * `best` for a family that returns no frames. */
function chosenMove(res) {
  const frames = Array.isArray(res.frames) ? res.frames : [];
  for (let i = frames.length - 1; i >= 0; i--) {
    const a = frames[i].action;
    if (a && Number.isFinite(a.q) && Number.isFinite(a.r)) return { q: +a.q, r: +a.r };
  }
  const best = res.best;
  return best && Number.isFinite(best.q) && Number.isFinite(best.r)
    ? { q: +best.q, r: +best.r } : null;
}

/* Same phrasing as the analysis deep look, so one search reads the same way
 * on both pages. */
function botPhaseText(scene, kind) {
  if (kind === "bare_policy") return "policy priors";
  if (kind === "candidate_set") return `${scene.cands.size} candidates drawn`;
  if (kind === "search_round") {
    const round = Number.isInteger(scene.round) ? scene.round + 1 : "?";
    const rounds = Number.isInteger(scene.rounds) ? scene.rounds : "?";
    let alive = 0;
    for (const cand of scene.cands.values()) if (!cand.cut) alive++;
    const visitRead = Number.isFinite(scene.visits) && Number.isFinite(scene.target)
      ? ` · ${scene.visits}/${scene.target} visits` : "";
    return `round ${round}/${rounds} · ${alive} of ${scene.cands.size} alive${visitRead}`;
  }
  const value = Number.isFinite(scene.rootValue) ? ` · value ${fmtV(scene.rootValue)}` : "";
  const verdict = sceneVerdict(scene);
  return `search complete${value}${verdict ? ` · ${verdict}` : ""}`;
}

/* Replay the collected telemetry frames on the lab board at a fixed cadence —
 * the live viewer's language, paced for watching. A family that sends no
 * frames paints the searched distribution directly. */
async function replayBotSearch(res, mover, run) {
  const frames = Array.isArray(res.frames) ? res.frames : null;
  if (!frames || !frames.length) {
    const rows = Array.isArray(res.visit_policy) ? res.visit_policy : [];
    if (rows.length) {
      board.setHeat(rows, mover === 0 ? H0 : H1, mover === 0 ? H0R : H1R, OV_OPA);
    }
    setBotPhase(`${res.visits} visits · value ${fmtV(res.root_value)}`);
    return;
  }
  const scene = newScene("lab");
  const reduced = reducedMotion();
  for (const event of frames) {
    if (state.bot.run !== run) return;
    foldLiveFrame(scene, event);
    if (reduced) continue;
    paintSceneTo(scene, board, mover);
    setBotPhase(botPhaseText(scene, event.kind));
    if (event.kind === "bare_policy") await sleep(420);
    else if (event.kind === "candidate_set") await sleep(280);
    else if (event.kind === "search_round") await sleep(130);
  }
  if (state.bot.run !== run) return;
  if (reduced) paintSceneTo(scene, board, mover);
  setBotPhase(botPhaseText(scene, "search_complete"));
}

/* One bot move: search the position, replay the search on the board, then
 * play the search's own choice into the active line. Returns true when a
 * stone landed. Bot searches are never cached — each one is a fresh search on
 * the position in front of it. */
async function botMove() {
  if (!canBotPlay()) {
    if (Date.now() < state.rlUntil) toast("rate-limited — try again shortly", true);
    return false;
  }
  // Autoplay does not refresh the modules between moves: that would fire one
  // eval per ply for a panel the replay is covering. The loop refreshes when
  // it stops.
  if (state.bot.auto) { clearTimeout(refreshT); refreshT = null; }
  else flushRefresh();
  const run = ++state.bot.run;
  state.bot.busy = true;
  syncPlayUI();
  clearBoardPaint();
  const moves = curMoves();
  const mover = currentToMove();
  setBotPhase(`searching · ${state.sims} visits`);
  try {
    const res = await requestJson("/api/lab/search", {
      checkpoint_id: state.ckpt,
      sims: state.sims,
      frames: true,
      actions: moves.map(([q, r]) => ({ q, r })),
    });
    if (state.bot.run !== run) return false;
    await replayBotSearch(res, mover, run);
    if (state.bot.run !== run) return false;
    const move = chosenMove(res);
    if (!move) {
      toast("the search returned no move", true);
      return false;
    }
    state.bot.busy = false;   // the placement is the lab's own edit
    playMove(move.q, move.r);
    if (termWinner !== null) {
      setBotPhase(`${termWinner === 0 ? "blue" : "red"} wins — six in a line`);
    }
    return true;
  } catch (e) {
    if (e.status === 429) state.rlUntil = Date.now() + 15000;
    if (state.bot.run !== run) return false;
    setBotPhase("");
    toast(e.status === 429
      ? "search rate-limited — wait a moment"
      : (e.message || "the bot could not search"), true);
    return false;
  } finally {
    if (state.bot.run === run) {
      state.bot.busy = false;
      syncPlayUI();
      if (!state.bot.auto) scheduleRefresh();
    }
  }
}

/* Autoplay: bot moves until the game ends, the user stops it, an error
 * arrives, or the ply cap is reached. Each move is fully awaited and fully
 * replayed, so the loop never runs two searches at once. */
async function autoplay() {
  const token = ++state.bot.autoRun;
  state.bot.auto = true;
  state.bot.plies = 0;
  syncPlayUI();
  while (state.bot.auto && state.bot.autoRun === token && termWinner === null) {
    if (state.bot.plies >= AUTOPLAY_CAP) {
      toast(`autoplay stopped after ${AUTOPLAY_CAP} plies`);
      break;
    }
    if (!await botMove()) break;
    state.bot.plies++;
  }
  if (state.bot.autoRun !== token) return;   // a newer run owns the controls
  state.bot.auto = false;
  syncPlayUI();
  if (termWinner !== null) {
    setReadout("game over",
      `${termWinner === 0 ? "blue" : "red"} has six in a line. step back to keep exploring.`);
  }
  scheduleRefresh();
}

$("botMoveBtn").addEventListener("click", () => { botMove(); });

$("autoBtn").addEventListener("click", () => {
  if (state.bot.auto) cancelBot();
  else autoplay();
});

// ---- module dispatch -------------------------------------------------------------------------------------

function refreshModule() {
  // A bot replay owns the board while it runs; the placement that ends it
  // re-arms this through positionChanged().
  if (state.bot.busy) return;
  renderPosition(); // legal ghost depends on the active module
  syncValueGraph();
  // Modules paint different layers — the heat for the eval/heads maps, the
  // overlay group for the rest — so each refresh starts from a bare board
  // rather than under the last module's picture.
  clearBoardPaint();
  if (termWinner !== null) {
    labKlent = null;
    syncOvlUI();
    setModuleStatus("the game is over — nothing left to evaluate");
    setReadout("game over",
      `${termWinner === 0 ? "blue" : "red"} has six in a line. step back to keep exploring.`);
    return;
  }
  setModuleStatus("");
  switch (state.module) {
    case "features": renderFeatures(); break;
    case "eval": renderEval(); break;
    case "heads": renderHeads(); break;
    case "attention": renderAttention(); break;
    case "activations": renderActivations(); break;
    case "search": renderSearchModule(); break;
  }
}

// ---- boot -------------------------------------------------------------------------------------------------

/* Sync the mode controls to state.mode (after applyHash or setMode). */
function syncModeUI() {
  segSelect($("modeSeg"), b => b.dataset.mode === state.mode);
  const free = state.mode === "free";
  $("brushSeg").hidden = !free;
  $("tmSeg").hidden = !free;
  $("freeNote").hidden = !free;
  $("stepPair").hidden = free;
  $("lineStrip").hidden = free;
  $("undoBtn").hidden = !free;
  if (free) segSelect($("tmSeg"), b => +b.dataset.tm === state.free.toMove);
  syncPlayUI();
}

/* A lab link opened while the page is already loaded only changes the hash —
 * no reload — so hashchange re-applies it. positionChanged() writes the same
 * hash back via replaceState, which does not re-fire the event. */
window.addEventListener("hashchange", () => {
  if (applyHash(location.hash)) {
    cancelBot();
    state.freeDirty = false;
    state.undo = [];
    syncModeUI();
    positionChanged();
    board.resetView();
  }
});

(async function boot() {
  buildFeatList();
  const params = new URLSearchParams(location.search);
  const applied = applyHash(location.hash);
  syncModeUI();
  syncOvlUI();
  renderPosition();
  refreshModule();
  await loadBots(params.get("checkpoint_id"));
  syncPlayUI();   // the bot, search and solver controls need a checkpoint
  syncValueGraph();   // the graph needs one too
  loadPresets();
  if (!applied && params.has("game")) {
    importFromGame(params.get("game"), parseInt(params.get("ply") ?? "", 10));
  }
})();
