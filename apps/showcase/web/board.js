/* board.js — the virtualized infinite hex board.
 *
 * Pure view: renders exactly the tiles intersecting the current viewBox (plus
 * a one-ring margin), pooling polygons keyed by "q,r". Game legality, state,
 * and network live in app.js. Pan is unbounded; zoom is clamped so the worst
 * case visible tile count stays bounded (~2000 < 2500 budget at max zoom-out).
 */

export const S = 26;
const SQ3 = Math.sqrt(3);
const DRAW = 0.975;
const NS = "http://www.w3.org/2000/svg";

const COL_W = S * SQ3;  // horizontal tile pitch
const ROW_H = S * 1.5;  // vertical tile pitch

// Home framing = the mockup's r9 hexagon extents, centered wherever the
// stones are. Zoom clamps are fixed multiples of this width: at the 2x
// zoom-out clamp the visible range is ~44 rows x ~43 cols (< 2000 tiles).
const HOME_W = 2 * COL_W * 10;
const HOME_H = 2 * (ROW_H * 9 + S * 1.7);
const ASPECT = HOME_H / HOME_W;
const W_MIN = HOME_W / 4;
const W_MAX = HOME_W * 2;
const TILE_BUDGET = 2500;

export const key = (q, r) => q + "," + r;
export const axialX = (q, r) => COL_W * (q + r / 2);
export const axialY = r => ROW_H * r;

export function hexPts(cx, cy, rad) {
  const p = [];
  for (let i = 0; i < 6; i++) {
    const a = Math.PI / 180 * (60 * i - 30);
    p.push((cx + rad * Math.cos(a)).toFixed(2) + "," + (cy + rad * Math.sin(a)).toFixed(2));
  }
  return p.join(" ");
}

const WIN_DIRS = [[1, 0], [0, 1], [1, -1]];

/* Client-side six-in-a-row detection: fallback for servers that don't send
 * winning_line in the terminal state. Order-independent over stones. */
export function findWin(stones) {
  const col = new Map();
  for (const m of stones) col.set(key(m.q, m.r), m.color);
  for (const m of stones) {
    for (const [dq, dr] of WIN_DIRS) {
      if (col.get(key(m.q - dq, m.r - dr)) === m.color) continue; // not the run start
      const run = [];
      let q = m.q, r = m.r;
      while (col.get(key(q, r)) === m.color) { run.push({ q, r }); q += dq; r += dr; }
      if (run.length >= 6) return run.slice(0, 6);
    }
  }
  return null;
}

/* Union boundary of the winning tiles: for each cell keep the hex edges not
 * shared with another winning cell, then chain the directed segments into one
 * closed loop. Edge i (vertex i -> i+1) faces axial neighbor EDGE_NB[i]. */
const EDGE_NB = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]];

function outlinePath(cells) {
  const set = new Set(cells.map(c => key(c.q, c.r)));
  const segs = [];
  for (const c of cells) {
    const cx = axialX(c.q, c.r), cy = axialY(c.r);
    const vx = [];
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 180 * (60 * i - 30);
      vx.push([cx + S * Math.cos(a), cy + S * Math.sin(a)]);
    }
    for (let e = 0; e < 6; e++) {
      if (set.has(key(c.q + EDGE_NB[e][0], c.r + EDGE_NB[e][1]))) continue;
      segs.push([vx[e], vx[(e + 1) % 6]]);
    }
  }
  const pk = p => Math.round(p[0] * 10) + "|" + Math.round(p[1] * 10);
  const byStart = new Map(segs.map(s => [pk(s[0]), s]));
  let cur = segs[0];
  const startK = pk(cur[0]), pts = [];
  for (let n = 0; n < segs.length; n++) {
    pts.push(cur[0]);
    const nk = pk(cur[1]);
    if (nk === startK) break;
    cur = byStart.get(nk);
    if (!cur) break;
  }
  return "M" + pts.map(p => p[0].toFixed(1) + " " + p[1].toFixed(1)).join("L") + "Z";
}

/* opts:
 *   onCellClick(q, r, pointerType)  tap/click on an empty tile
 *   onHover(q, r | null)            cursor readout
 *   ghostAllowed()                  gate for the hover ghost (default true)
 *   canReset(target)                gate for dblclick/double-tap view reset
 *   onPanStart()                    hide transient UI when a pan/pinch begins
 */
export function createBoard(svg, opts = {}) {
  const frame = svg.parentNode;
  let vb = { x: -HOME_W / 2, y: -HOME_H / 2, w: HOME_W, h: HOME_H };

  const mk = cls => {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", cls);
    svg.appendChild(g);
    return g;
  };
  // draw order: grid (tiles + tengen), heat, stones ABOVE heat, marks, ghost
  const grid = mk("gridg"), heat = mk("heatg"), stones = mk("stonesg"),
        marks = mk("marksg"), ghostG = mk("ghostg");
  const tilesG = document.createElementNS(NS, "g");
  grid.appendChild(tilesG);
  const tengen = document.createElementNS(NS, "circle");
  tengen.setAttribute("cx", 0); tengen.setAttribute("cy", 0);
  tengen.setAttribute("r", 1.8); tengen.setAttribute("class", "tengen");
  grid.appendChild(tengen);

  // ---- tile virtualization -------------------------------------------------
  const tiles = new Map();   // "q,r" -> polygon in tilesG
  const pool = [];
  const occupied = new Set();
  let lastSig = "";
  let scheduled = false;

  function acquireTile(q, r) {
    const el = pool.pop() || document.createElementNS(NS, "polygon");
    el.setAttribute("points", hexPts(axialX(q, r), axialY(r), S * DRAW));
    el.setAttribute("class", occupied.has(key(q, r)) ? "cell occ" : "cell");
    el.dataset.q = q;
    el.dataset.r = r;
    tilesG.appendChild(el);
    return el;
  }

  function updateVirtual() {
    // Tile visible iff its center is within half a tile of the viewBox, then
    // one extra ring of margin so mid-pan reveals are already rendered.
    const lo = (vb.x - COL_W / 2) / COL_W - 1;
    const hi = (vb.x + vb.w + COL_W / 2) / COL_W + 1;
    const rMin = Math.floor((vb.y - S) / ROW_H) - 1;
    const rMax = Math.ceil((vb.y + vb.h + S) / ROW_H) + 1;
    // Signature quantized at half-column steps: per-row q ranges are
    // ceil(lo - r/2)..floor(hi - r/2), which only change when lo/hi cross a
    // half-integer — so an unchanged signature means an unchanged tile set.
    const sig = rMin + "|" + rMax + "|" + Math.floor(lo * 2) + "|" + Math.floor(hi * 2);
    if (sig === lastSig) return;
    lastSig = sig;

    for (const [k, el] of tiles) {
      const q = +el.dataset.q, r = +el.dataset.r;
      if (r < rMin || r > rMax || q < Math.ceil(lo - r / 2) || q > Math.floor(hi - r / 2)) {
        tiles.delete(k);
        el.remove();
        if (pool.length < TILE_BUDGET) pool.push(el);
      }
    }
    let count = tiles.size;
    for (let r = rMin; r <= rMax; r++) {
      const qMin = Math.ceil(lo - r / 2), qMax = Math.floor(hi - r / 2);
      for (let q = qMin; q <= qMax; q++) {
        const k = key(q, r);
        if (tiles.has(k)) continue;
        if (++count > TILE_BUDGET) return; // safety net; unreachable within clamps
        tiles.set(k, acquireTile(q, r));
      }
    }
  }

  // rAF-throttled while visible; rAF is starved in hidden documents, so fall
  // back to a short timeout there (keeps tiles fresh across tab switches)
  function scheduleVirtual() {
    if (scheduled) return;
    scheduled = true;
    const run = () => { scheduled = false; updateVirtual(); };
    if (document.hidden) setTimeout(run, 50);
    else requestAnimationFrame(run);
  }

  // ---- viewBox / home ------------------------------------------------------
  let stoneList = [];

  const clampW = w => Math.max(W_MIN, Math.min(W_MAX, w));

  /* Home = the standard framing centered on the stones' bounding box (the
   * origin before any stone exists), zoomed out just enough to fit. */
  function computeHome() {
    if (!stoneList.length) return { x: -HOME_W / 2, y: -HOME_H / 2, w: HOME_W, h: HOME_H };
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const m of stoneList) {
      const x = axialX(m.q, m.r), y = axialY(m.r);
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    const pad = 4 * S;
    const w = clampW(Math.max(HOME_W, x1 - x0 + 2 * pad, (y1 - y0 + 2 * pad) / ASPECT));
    const h = w * ASPECT;
    return { x: (x0 + x1) / 2 - w / 2, y: (y0 + y1) / 2 - h / 2, w, h };
  }

  const resetBtn = document.createElement("button");
  resetBtn.className = "reset-view";
  resetBtn.textContent = "reset view";
  frame.appendChild(resetBtn);

  function offHome() {
    const home = computeHome();
    return Math.abs(vb.x - home.x) > 0.5 || Math.abs(vb.y - home.y) > 0.5 ||
           Math.abs(vb.w - home.w) > 0.5;
  }

  function apply() {
    svg.setAttribute("viewBox",
      vb.x.toFixed(2) + " " + vb.y.toFixed(2) + " " + vb.w.toFixed(2) + " " + vb.h.toFixed(2));
    resetBtn.classList.toggle("show", offHome());
    scheduleVirtual();
  }

  function goHome() {
    vb = computeHome();
    apply();
  }

  function world(cx, cy) {
    const r = svg.getBoundingClientRect();
    return [vb.x + (cx - r.left) / r.width * vb.w, vb.y + (cy - r.top) / r.height * vb.h];
  }

  // ---- pan / zoom ----------------------------------------------------------
  // wheel: zoom about the cursor; drag: pan with a 5px threshold preserving
  // clicks; pinch: zoom about the midpoint; dblclick / double-tap: home.
  const ptrs = {};
  let panStart = null, pinch = null, panning = false;
  let suppress = false; // swallow the click that would follow a drag
  let lastTap = { t: 0, x: 0, y: 0 };
  let lastPtrType = "mouse";
  const n = () => Object.keys(ptrs).length;

  svg.addEventListener("wheel", e => {
    e.preventDefault();
    const w2 = clampW(vb.w * (e.deltaY < 0 ? 1 / 1.15 : 1.15));
    if (w2 === vb.w) return;
    const p = world(e.clientX, e.clientY), k = w2 / vb.w;
    vb.x = p[0] - (p[0] - vb.x) * k;
    vb.y = p[1] - (p[1] - vb.y) * k;
    vb.w = w2; vb.h = w2 * ASPECT;
    apply();
  }, { passive: false });

  svg.addEventListener("pointerdown", e => {
    lastPtrType = e.pointerType || "mouse";
    if (e.pointerType === "mouse" && e.button !== 0) return;
    ptrs[e.pointerId] = { x: e.clientX, y: e.clientY };
    if (n() === 1) {
      panStart = { cx: e.clientX, cy: e.clientY, vx: vb.x, vy: vb.y, vw: vb.w, vh: vb.h };
      panning = false; suppress = false;
    } else if (n() === 2) {
      panStart = null; panning = false; suppress = true;
      if (opts.onPanStart) opts.onPanStart();
      const ids = Object.keys(ptrs), a = ptrs[ids[0]], c = ptrs[ids[1]];
      pinch = { d: Math.hypot(a.x - c.x, a.y - c.y) || 1, vw: vb.w,
                world: world((a.x + c.x) / 2, (a.y + c.y) / 2) };
      try { svg.setPointerCapture(+ids[0]); svg.setPointerCapture(+ids[1]); } catch (_) {}
    }
  });

  svg.addEventListener("pointermove", e => {
    const p = ptrs[e.pointerId];
    if (!p) return;
    p.x = e.clientX; p.y = e.clientY;
    if (n() === 2 && pinch) {
      const ids = Object.keys(ptrs), a = ptrs[ids[0]], c = ptrs[ids[1]];
      const d = Math.hypot(a.x - c.x, a.y - c.y) || 1;
      const w2 = clampW(pinch.vw * pinch.d / d);
      const r = svg.getBoundingClientRect(), mx = (a.x + c.x) / 2, my = (a.y + c.y) / 2;
      vb.w = w2; vb.h = w2 * ASPECT;
      vb.x = pinch.world[0] - (mx - r.left) / r.width * vb.w;
      vb.y = pinch.world[1] - (my - r.top) / r.height * vb.h;
      apply();
    } else if (n() === 1 && panStart) {
      const dx = e.clientX - panStart.cx, dy = e.clientY - panStart.cy;
      if (!panning && Math.hypot(dx, dy) > 5) {
        panning = true; suppress = true;
        svg.classList.add("panning");
        if (opts.onPanStart) opts.onPanStart();
        try { svg.setPointerCapture(e.pointerId); } catch (_) {}
      }
      if (panning) {
        const r2 = svg.getBoundingClientRect();
        vb.x = panStart.vx - dx / r2.width * panStart.vw;
        vb.y = panStart.vy - dy / r2.height * panStart.vh;
        apply();
      }
    }
  });

  function end(e) {
    if (!ptrs[e.pointerId]) return;
    delete ptrs[e.pointerId];
    if (n() < 2) pinch = null;
    if (n() === 1) {
      // pinch released down to one finger: re-baseline the pan
      const id = Object.keys(ptrs)[0], p = ptrs[id];
      panStart = { cx: p.x, cy: p.y, vx: vb.x, vy: vb.y, vw: vb.w, vh: vb.h };
      panning = false;
    } else if (n() === 0) {
      svg.classList.remove("panning");
      if (e.pointerType === "touch" && !suppress) {
        // manual double-tap detect (350ms / 25px window)
        const now = Date.now();
        if (now - lastTap.t < 350 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 25
            && (!opts.canReset || opts.canReset(e.target))) {
          goHome(); suppress = true; lastTap.t = 0;
        } else lastTap = { t: now, x: e.clientX, y: e.clientY };
      }
      panStart = null; panning = false;
    }
  }
  window.addEventListener("pointerup", end);
  window.addEventListener("pointercancel", end);

  // capture-phase: a drag's trailing click never reaches the cells
  svg.addEventListener("click", e => {
    if (suppress) { e.stopPropagation(); e.preventDefault(); }
  }, true);
  svg.addEventListener("dblclick", e => {
    e.preventDefault();
    if (suppress) return; // the "double click" was really two drags
    if (!opts.canReset || opts.canReset(e.target)) goHome();
  });
  resetBtn.addEventListener("click", goHome);

  // ---- ghost + hover + click delegation ------------------------------------
  let legalSet = null; // Set of "q,r" the engine allows, or null (no ghost)

  const hoverGhost = document.createElementNS(NS, "polygon");
  hoverGhost.setAttribute("class", "ghost");
  hoverGhost.style.display = "none";
  ghostG.appendChild(hoverGhost);

  const stagedGhost = document.createElementNS(NS, "polygon");
  stagedGhost.setAttribute("class", "ghost staged");
  stagedGhost.style.display = "none";
  ghostG.appendChild(stagedGhost);

  grid.addEventListener("pointerover", e => {
    const t = e.target;
    if (t.tagName !== "polygon") {
      hoverGhost.style.display = "none";
      if (opts.onHover) opts.onHover(null, null);
      return;
    }
    const q = +t.dataset.q, r = +t.dataset.r;
    if (opts.onHover) opts.onHover(q, r);
    const allowed = legalSet && legalSet.has(key(q, r)) &&
                    (!opts.ghostAllowed || opts.ghostAllowed());
    if (!allowed) { hoverGhost.style.display = "none"; return; }
    hoverGhost.setAttribute("points", hexPts(axialX(q, r), axialY(r), S * 0.8));
    hoverGhost.style.display = "";
  });
  grid.addEventListener("pointerleave", () => {
    hoverGhost.style.display = "none";
    if (opts.onHover) opts.onHover(null, null);
  });
  grid.addEventListener("click", e => {
    const t = e.target;
    if (t.tagName !== "polygon" || t.classList.contains("occ")) return;
    if (opts.onCellClick) opts.onCellClick(+t.dataset.q, +t.dataset.r, lastPtrType);
  });

  // ---- public rendering API -------------------------------------------------
  function setStones(moves, winCells) {
    stoneList = moves;
    stones.textContent = "";
    marks.textContent = "";
    occupied.clear();
    for (const m of moves) {
      occupied.add(key(m.q, m.r));
      const el = document.createElementNS(NS, "polygon");
      el.setAttribute("points", hexPts(axialX(m.q, m.r), axialY(m.r), S * 0.8));
      el.setAttribute("class", "stone " + (m.color === 0 ? "s0" : "s1"));
      stones.appendChild(el);
    }
    for (const [k, el] of tiles) el.classList.toggle("occ", occupied.has(k));
    resetBtn.classList.toggle("show", offHome());
    if (winCells && winCells.length) {
      // terminal: frame the winning six; supersedes last-move marks
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", outlinePath(winCells));
      path.setAttribute("class", "winoutline");
      marks.appendChild(path);
      return;
    }
    // hexo plays two stones per turn: mark the last two, most recent strongest
    for (let i = Math.max(0, moves.length - 2); i < moves.length; i++) {
      const m = moves[i];
      const dot = document.createElementNS(NS, "circle");
      dot.setAttribute("cx", axialX(m.q, m.r));
      dot.setAttribute("cy", axialY(m.r));
      dot.setAttribute("r", i === moves.length - 1 ? 2.5 : 2);
      dot.setAttribute("class", i === moves.length - 1 ? "lastdot" : "lastdot prev");
      marks.appendChild(dot);
    }
  }

  function setLegal(cells) {
    legalSet = cells ? new Set(cells.map(c => key(c.q, c.r))) : null;
    if (!legalSet) hoverGhost.style.display = "none";
  }

  /* Flat 2D policy map: full-tile fills UNDER the solid stones. Tint comes
   * from pale hue-shifted families (never the stone colors) so a fully-lit
   * overlay cell can't pass for a translucent stone; best cell gets a ring. */
  function setHeat(rows, tint, ringTint, opa) {
    heat.textContent = "";
    if (!rows || !rows.length || opa <= 0) return;
    const maxW = rows[0].p || 1;
    for (const h of rows) {
      const el = document.createElementNS(NS, "polygon");
      el.setAttribute("points", hexPts(axialX(h.q, h.r), axialY(h.r), S * DRAW));
      el.setAttribute("class", "heatcell");
      el.setAttribute("fill", tint);
      el.setAttribute("opacity", (opa * (0.12 + 0.88 * h.p / maxW)).toFixed(3));
      heat.appendChild(el);
    }
    const best = rows[0];
    const ring = document.createElementNS(NS, "polygon");
    ring.setAttribute("points", hexPts(axialX(best.q, best.r), axialY(best.r), S * 0.86));
    ring.setAttribute("class", "heattop");
    ring.setAttribute("stroke", ringTint);
    ring.setAttribute("opacity", Math.min(1, opa + 0.1).toFixed(3));
    heat.appendChild(ring);
  }

  const clearHeat = () => { heat.textContent = ""; };

  function stage(q, r) {
    stagedGhost.setAttribute("points", hexPts(axialX(q, r), axialY(r), S * 0.8));
    stagedGhost.style.display = "";
  }
  const clearStage = () => { stagedGhost.style.display = "none"; };
  const hideHoverGhost = () => { hoverGhost.style.display = "none"; };

  apply();
  updateVirtual();

  return {
    svg, setStones, setLegal, setHeat, clearHeat,
    stage, clearStage, hideHoverGhost,
    resetView: goHome,
    tileCount: () => tiles.size,
  };
}
