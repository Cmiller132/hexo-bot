/* lab.js — the lab page controller: shared position editor (legal-sequence /
 * free-edit) on one board, five inspection modules reading that position.
 *
 * Data flow per module:
 *   features     client-computed (lab_features.js mirrors the server featurizer)
 *   net eval     POST /api/lab/eval                       (worker forward)
 *   attention    POST /api/lab/eval wants.attention_query (hooked forward)
 *   activations  POST /api/lab/eval wants.activations     (hooked forward)
 *   search       POST /api/lab/search                     (real capped search)
 *
 * board.js is reused for rendering/pan/zoom; the small helpers duplicated
 * from app.js (toasts, checkpoint grouping, copy) stay here by design — the
 * lab must not edit the play bundle.
 */

import { S, axialX, axialY, createBoard, findWin, hexPts, key } from "../board.js?v=6";
import * as LF from "./lab_features.js?v=1";

"use strict";

const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const clamp = (lo, hi, v) => Math.max(lo, Math.min(hi, v));
const fmtV = v => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);
const fmtCell = (q, r) => q + "," + r;

/* overlay tone families (same choices as app.js): pale + hue-shifted so a
 * tinted cell never passes for a stone */
const H0 = "#9fd0ff", H1 = "#ffb4aa";
const H0R = "#d7ebff", H1R = "#ffddd6";
const ACCENT = "#e8e2d6";

const SEARCH_BUDGETS = [16, 64, 256];

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
  moves: [],                     // sequence history, [[q, r], ...]
  free: { p0: [], p1: [], toMove: 0 },
  freeDirty: false,              // free stones diverged from `moves`
  brush: 0,                      // 0 | 1 | "erase"
  undo: [],                      // free-mode snapshots
  staged: null,                  // touch two-tap staging
  module: "features",
  feature: "support",            // "support" | feature name
  ckpt: null,
  ckptLabel: "",
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
};

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

// ---- position accessors -----------------------------------------------------------

/* Current stones as [{q, r, color}] in a stable render order. */
function stoneList() {
  if (state.mode === "sequence") {
    return state.moves.map(([q, r], i) => ({ q, r, color: LF.recordPlayer(i) }));
  }
  return state.free.p0.map(([q, r]) => ({ q, r, color: 0 }))
    .concat(state.free.p1.map(([q, r]) => ({ q, r, color: 1 })));
}

function currentToMove() {
  if (state.mode === "sequence") return LF.recordPlayer(state.moves.length);
  return state.free.toMove;
}

function currentFacts() {
  return state.mode === "sequence"
    ? LF.factsFromSequence(state.moves)
    : LF.factsFromFree(state.free.p0, state.free.p1, state.free.toMove);
}

/* Server body for the current position ({actions} or {stones} + to_move). */
function positionBody() {
  if (state.mode === "sequence") {
    return { actions: state.moves.map(([q, r]) => ({ q, r })) };
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
    ? "s:" + state.moves.map(m => m.join(",")).join(";")
    : "f:" + state.free.p0.map(m => m.join(",")).join(";") +
      "|" + state.free.p1.map(m => m.join(",")).join(";") + "|" + state.free.toMove;
}

// ---- editor: sequence mode ---------------------------------------------------------

function trySequencePlace(q, r) {
  if (!LF.isLegalPlacement(state.moves, q, r)) {
    toast(state.moves.length ? "play within reach of the stones" : "the opening stone is forced to 0,0");
    return;
  }
  const color = LF.recordPlayer(state.moves.length);
  const next = stoneList().concat([{ q, r, color }]);
  if (findWin(next)) {
    toast("that completes six in a line — the net evaluates live positions only", true);
    return;
  }
  state.lastPlaceT = Date.now();
  state.moves.push([q, r]);
  positionChanged();
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
    pushFreeUndo();
    if (inP0 >= 0) state.free.p0.splice(inP0, 1);
    else state.free.p1.splice(inP1, 1);
  } else if ((state.brush === 0 && inP1 >= 0) || (state.brush === 1 && inP0 >= 0)) {
    // recolor to the brush color (parity check applies)
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
               (LF.isLegalPlacement(state.moves, q, r))) {
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

function onBoardHover(q, r) {
  cursorPos.textContent = q === null ? "—" : fmtCell(q, r);
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

$("undoBtn").addEventListener("click", () => {
  if (state.mode === "sequence") {
    if (!state.moves.length) return;
    state.moves.pop();
  } else {
    const snap = state.undo.pop();
    if (!snap) return;
    state.free = JSON.parse(snap);
    segSelect($("tmSeg"), b => +b.dataset.tm === state.free.toMove);
  }
  positionChanged();
});

$("clearBtn").addEventListener("click", () => {
  if (state.mode === "sequence") state.moves = [];
  else { pushFreeUndo(); state.free = { p0: [], p1: [], toMove: 0 }; state.freeDirty = true; }
  $("presetSel").value = "";
  positionChanged();
});

// ---- share link ----------------------------------------------------------------------

function shareHash() {
  if (state.mode === "sequence") {
    return "#m=" + state.moves.map(m => m.join(",")).join(";");
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

/* #m=... (sequence) / #f0=...&f1=...&tm=... (free). Returns true if applied. */
function applyHash(hash) {
  if (!hash || hash.length < 2) return false;
  const params = new URLSearchParams(hash.slice(1));
  if (params.has("m")) {
    const moves = parseCells(params.get("m"));
    if (moves === null || !validSequence(moves)) {
      toast("lab link position was not a legal sequence", true);
      return false;
    }
    state.mode = "sequence";
    state.moves = moves;
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

function validSequence(moves) {
  const staged = [];
  for (const [q, r] of moves) {
    if (!LF.isLegalPlacement(staged, q, r)) return false;
    staged.push([q, r]);
    if (findWin(staged.map(([a, b], i) => ({ q: a, r: b, color: LF.recordPlayer(i) })))) return false;
  }
  return true;
}

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
}

$("presetSel").addEventListener("change", function () {
  const opt = this.selectedOptions[0];
  const moves = opt && opt.dataset.moves ? JSON.parse(opt.dataset.moves) : [];
  if (state.mode !== "sequence") setMode("sequence");
  state.moves = moves;
  positionChanged();
  board.resetView();
});

// ---- game import (?game=<id>&ply=<n>) ---------------------------------------------------

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
    state.mode = "sequence";
    state.moves = moves;
    positionChanged();
    board.resetView();
    toast(`loaded game ${String(id).slice(0, 8)} at ply ${moves.length}`);
  } catch (e) {
    toast(e.message === "network error" ? "couldn't reach the server" : "couldn't load that game", true);
  }
}

// ---- checkpoints (duplicated minimally from app.js: grouping + latest tag) ---------------

function normalizeBots(raw) {
  const list = raw && Array.isArray(raw.checkpoints) ? raw.checkpoints : [];
  return list.map(c => ({
    id: String(c.id),
    label: String(c.label ?? c.id),
    group: typeof c.group === "string" ? c.group : "",
    search: typeof c.search === "string" ? c.search : "",
    meta: c.run ? String(c.run) : "",
  }));
}

function groupedCheckpoints(checkpoints) {
  const order = [], byName = new Map();
  for (const c of checkpoints) {
    const g = c.group || "";
    if (!byName.has(g)) { byName.set(g, []); order.push(g); }
    byName.get(g).push(c);
  }
  const at = order.indexOf("");
  if (at > 0) { order.splice(at, 1); order.unshift(""); }
  return order.map(name => ({ name, items: byName.get(name) }));
}

function renderCkpts() {
  const list = $("ckptList");
  list.textContent = "";
  const groups = groupedCheckpoints(state.bots);
  const first = groups.length ? groups[0].items : [];
  const latest = first[first.length - 1] || null;
  for (const g of groups) {
    if (g.name) {
      const h = document.createElement("div");
      h.className = "bot-group";
      h.textContent = g.name;
      list.appendChild(h);
    }
    for (const c of g.items) {
      const b = document.createElement("button");
      b.className = "bot" + (c.id === state.ckpt ? " sel" : "");
      b.dataset.ckpt = c.id;
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", c.id === state.ckpt);
      const tags = [];
      if (latest && c.id === latest.id) tags.push('<span class="tag">latest</span>');
      if (c.search === "puct") tags.push('<span class="tag puct">PUCT search</span>');
      const meta = [c.meta, ...tags].filter(Boolean).join(" · ");
      b.innerHTML = `<span class="bot-row"><span class="bot-name"></span>` +
        `<span class="bot-meta">${meta}</span></span>`;
      b.querySelector(".bot-name").textContent = c.label;
      list.appendChild(b);
    }
  }
}

$("ckptList").addEventListener("click", e => {
  const b = e.target.closest(".bot");
  if (!b || b.dataset.ckpt === state.ckpt) return;
  state.ckpt = b.dataset.ckpt;
  const c = state.bots.find(x => x.id === state.ckpt);
  state.ckptLabel = c ? c.label : state.ckpt;
  renderCkpts();
  refreshModule();
});

async function loadBots() {
  try {
    state.bots = normalizeBots(await requestJson("/api/bots"));
  } catch (_) {
    state.bots = [];
    setStatus("evalStatus", "server unreachable — live modules unavailable", true);
    return;
  }
  const groups = groupedCheckpoints(state.bots);
  const first = groups.length ? groups[0].items : [];
  const latest = first[first.length - 1] || null;
  state.ckpt = latest ? latest.id : null;
  state.ckptLabel = latest ? latest.label : "";
  renderCkpts();
}

// ---- module switching --------------------------------------------------------------------

$("modSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b || b.dataset.mod === state.module) return;
  state.module = b.dataset.mod;
  segSelect($("modSeg"), x => x.dataset.mod === state.module);
  document.querySelectorAll(".mod").forEach(m => {
    m.classList.toggle("active", m.id === "mod-" + state.module);
  });
  refreshModule();
});

function setStatus(id, msg, isErr = false) {
  const el = $(id);
  el.textContent = msg || "";
  el.className = "mod-status" + (isErr ? " err" : "");
}

// ---- server eval (cached, debounced, rate-limit aware) -------------------------------------

function wantsKey(wants) {
  return (wants.attention_query ? "a" + wants.attention_query.q + "," + wants.attention_query.r : "") +
         (wants.activations ? "|act" : "");
}

function fetchEval(wants = {}) {
  if (!state.ckpt) return Promise.reject({ status: 0, message: "no checkpoint catalogue" });
  if (Date.now() < state.rlUntil) {
    return Promise.reject({ status: 429, message: "rate-limited — try again shortly" });
  }
  const k = state.ckpt + "|" + posKey() + "|" + wantsKey(wants);
  if (state.evalCache.has(k)) return state.evalCache.get(k);
  const body = { checkpoint_id: state.ckpt, ...positionBody() };
  if (wants.attention_query || wants.activations) body.wants = wants;
  const prom = requestJson("/api/lab/eval", body).catch(e => {
    state.evalCache.delete(k);
    if (e.status === 429) state.rlUntil = Date.now() + 15000;
    throw e;
  });
  state.evalCache.set(k, prom);
  if (state.evalCache.size > 120) {
    state.evalCache.delete(state.evalCache.keys().next().value);
  }
  return prom;
}

let refreshT = null;
function positionChanged() {
  clearStage();
  renderPosition();
  history.replaceState(null, "", location.pathname + location.search + shareHash());
  // drop an attention query that left the support
  if (state.attnCell) {
    const sup = LF.buildSupport(stoneList().map(s => [s.q, s.r]));
    if (!sup.index.has(key(state.attnCell[0], state.attnCell[1]))) state.attnCell = null;
  }
  clearTimeout(refreshT);
  refreshT = setTimeout(refreshModule, 350);
}

// ---- shared position rendering ---------------------------------------------------------------

function renderPosition() {
  const stones = stoneList();
  board.setStones(stones, null);
  board.setLegal(
    state.mode === "sequence" && state.module !== "attention"
      ? LF.legalCells(state.moves).map(([q, r]) => ({ q, r }))
      : null,
  );
  const facts = currentFacts();
  $("mgStones").textContent = stones.length;
  const tm = facts.currentPlayer;
  const mg = $("mgToMove");
  mg.textContent = tm === 0 ? "blue" : "red";
  mg.className = "n " + (tm === 0 ? "is-p0" : "is-p1");
  $("mgPhase").textContent =
    facts.phase === "Opening" ? "opening" :
    facts.phase === "SecondStone" ? "2nd stone" : "1st stone";
}

// ---- module: features (client-side) ------------------------------------------------------------

function buildFeatList() {
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
  for (const name of LF.FEATURE_NAMES) {
    const zeroed = LF.FREE_ZEROED.includes(name);
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
  const flip = v => (v === null || v === undefined ? null : tm === 0 ? v : -v);
  const v = flip(payload.value);
  setBig($("valNow"), v, "value-big");
  $("valWho").textContent =
    v === null ? "" : Math.abs(v) < 0.08 ? "even" : v >= 0 ? "blue better" : "red better";
  setBig($("stv2"), flip(payload.stv["2"]), "hz-v");
  setBig($("stv6"), flip(payload.stv["6"]), "hz-v");
  setBig($("stv16"), flip(payload.stv["16"]), "hz-v");
  $("mlNow").textContent = "~" + Math.max(0, Math.round(payload.moves_left));
  renderDist(payload.value_dist, payload.value);

  const rows = payload[evalHead] || [];
  clearOverlay();
  const headMover = evalHead === "opp_policy" ? 1 - tm : tm;
  paintFill(rows.map(h => ({ q: h.q, r: h.r, v: h.p })), headMover === 0 ? H0 : H1, 0.62);
  if (rows.length) paintRing(rows[0].q, rows[0].r, headMover === 0 ? H0R : H1R);

  const list = $("topList");
  list.textContent = "";
  for (const row of rows.slice(0, 5)) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="rk-c"></span><span class="rk-v"></span>`;
    li.querySelector(".rk-c").textContent = fmtCell(row.q, row.r);
    li.querySelector(".rk-v").textContent = (row.p * 100).toFixed(1) + "%";
    list.appendChild(li);
  }
  setReadout(
    "net eval · " + state.ckptLabel,
    `${evalHead.replace("_", " ")} over ${payload.legal_count} legal cells · ` +
    `value ${v === null ? "—" : fmtV(v)} (blue POV) · no search`,
  );
}

function renderEval() {
  setStatus("evalStatus", "computing…");
  const k = posKey();
  fetchEval({}).then(payload => {
    if (posKey() !== k || state.module !== "eval") return;
    setStatus("evalStatus", "");
    renderEvalPayload(payload);
  }).catch(e => {
    if (posKey() !== k || state.module !== "eval") return;
    setStatus("evalStatus", e.message || "eval failed", true);
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
    if (posKey() !== k || state.module !== "attention") return;
    if (!state.attnCell || state.attnCell[0] !== q || state.attnCell[1] !== r) return;
    setStatus("attnStatus", "");
    const attn = payload.attention;
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
      `${(tokenMass * 100).toFixed(1)}% of the row on the 8 summary tokens`,
    );
  }).catch(e => {
    if (posKey() !== k || state.module !== "attention") return;
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
    if (posKey() !== k || state.module !== "activations") return;
    setStatus("actStatus", "");
    actPayload = payload;
    const n = payload.activations.blocks.length;
    const range = $("actRange");
    range.max = n - 1;
    state.actStage = clamp(0, n - 1, state.actStage);
    range.value = state.actStage;
    renderActStage();
  }).catch(e => {
    if (posKey() !== k || state.module !== "activations") return;
    setStatus("actStatus", e.message || "activations fetch failed", true);
  });
}

// ---- module: search -----------------------------------------------------------------------------------

$("simsSeg").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  state.sims = +b.dataset.sims;
  segSelect($("simsSeg"), x => x === b);
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
  if (state.mode !== "sequence") return;
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
    if (posKey() !== pk || state.module !== "search") return;
    setStatus("searchStatus", "");
    renderSearchPayload(payload);
  }).catch(e => {
    if (e.status === 429) state.rlUntil = Date.now() + 15000;
    if (posKey() !== pk || state.module !== "search") return;
    setStatus("searchStatus", e.message || "search failed", true);
  }).finally(() => {
    $("searchBtn").disabled = state.mode !== "sequence";
  });
}

function renderSearchModule() {
  const free = state.mode !== "sequence";
  $("searchBtn").disabled = free;
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

// ---- module dispatch -------------------------------------------------------------------------------------

function refreshModule() {
  renderPosition(); // legal ghost depends on the active module
  switch (state.module) {
    case "features": renderFeatures(); break;
    case "eval": renderEval(); break;
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
  if (free) segSelect($("tmSeg"), b => +b.dataset.tm === state.free.toMove);
}

/* A lab link opened while the page is already loaded only changes the hash —
 * no reload — so hashchange re-applies it. positionChanged() writes the same
 * hash back via replaceState, which does not re-fire the event. */
window.addEventListener("hashchange", () => {
  if (applyHash(location.hash)) {
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
  if (applied) syncModeUI();
  renderPosition();
  refreshModule();
  await loadBots();
  loadPresets();
  if (!applied && params.has("game")) {
    importFromGame(params.get("game"), parseInt(params.get("ply") ?? "", 10));
  }
})();
