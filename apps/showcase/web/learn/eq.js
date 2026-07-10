/* eq.js — shared library for the hexfield_eq "how it works" demos.
 *
 * Two halves:
 *
 * 1. A faithful JS port of the hexfield_eq featurizer geometry —
 *    support BFS (packages/hexfield_eq/python/hexfield_eq/support.py),
 *    graded window scan and ray-length walk (features.py). The demo pages
 *    compute windows/rays live as the user places stones; the semantics here
 *    are transcriptions of the Python oracle (which is itself parity-tested
 *    against the Rust production featurizer). Validated against
 *    data/eq_walkthrough.json (exported from the real featurizer) in
 *    scripts/export_eq_learn_data.py's companion check.
 *
 * 2. Small SVG hex-board helpers shared by the figure demos (same axial→pixel
 *    mapping as ../board.js, sized for inline figures).
 */

/* ---------- hex geometry ---------- */

// Canonical direction order — constants.py DIRECTIONS (rot60 cycle order).
export const DIRS = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]];

// Win axes in canonical order [Q, R, QR] — features.py AXIS_DELTAS.
export const AXES = [[1, 0], [0, 1], [1, -1]];
export const AXIS_NAMES = ["Q", "R", "QR"];

export const WINDOW_LEN = 6;
export const RAY_REACH = 5;      // constants.py RAY_REACH = WINDOW_LEN - 1
export const SUPPORT_RADIUS = 4; // production HEXFIELD_EQ_SUPPORT_RADIUS
export const LEGAL_RADIUS = 8;   // rules legality radius (dist norm base)
export const LINE_NORM = 5, LIVE_NORM = 6, FORK_NORM = 3, FORK_LINE_THRESHOLD = 3;

export const hexDist = (q, r) => (Math.abs(q) + Math.abs(r) + Math.abs(q + r)) / 2;
export const key = (q, r) => q + "," + r;

/* ---------- support BFS (support.py _build_support) ---------- */

// stones: array of [q, r]. radius: support radius (default production 4).
// Returns { coords, legalCount, stoneCount, haloCount, dist, index }
// with coords ordered [legal | stones | halo], each segment ascending (q, r).
export function buildSupport(stones, radius = SUPPORT_RADIUS) {
  const haloDist = radius + 1;
  if (!stones.length) {
    const ordered = [[0, 0]].concat(
      DIRS.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]));
    const index = new Map(ordered.map((c, i) => [key(c[0], c[1]), i]));
    return { coords: ordered, legalCount: 1, stoneCount: 0, haloCount: 6,
             dist: ordered.map(() => 0), index };
  }
  const stoneSet = new Set(stones.map(([q, r]) => key(q, r)));
  const dist = new Map();
  const frontier = [];
  for (const [q, r] of stones) {
    const k = key(q, r);
    if (!dist.has(k)) { dist.set(k, 0); frontier.push([q, r]); }
  }
  let head = 0;
  while (head < frontier.length) {
    const [q, r] = frontier[head++];
    const d = dist.get(key(q, r));
    if (d === haloDist) continue;
    for (const [dq, dr] of DIRS) {
      const k = key(q + dq, r + dr);
      if (!dist.has(k)) { dist.set(k, d + 1); frontier.push([q + dq, r + dr]); }
    }
  }
  const cells = [];
  for (const [k, d] of dist) {
    const [q, r] = k.split(",").map(Number);
    cells.push([q, r, d]);
  }
  const asc = (a, b) => a[0] - b[0] || a[1] - b[1];
  const legal = cells.filter(([q, r, d]) => d <= radius && !stoneSet.has(key(q, r)))
    .sort(asc);
  const stonesSorted = cells.filter(([q, r]) => stoneSet.has(key(q, r))).sort(asc);
  const halo = cells.filter(([, , d]) => d === haloDist).sort(asc);
  const ordered = legal.concat(stonesSorted, halo).map(([q, r]) => [q, r]);
  const index = new Map(ordered.map((c, i) => [key(c[0], c[1]), i]));
  return {
    coords: ordered,
    legalCount: legal.length,
    stoneCount: stonesSorted.length,
    haloCount: halo.length,
    dist: legal.concat(stonesSorted, halo).map(([, , d]) => d),
    index,
  };
}

/* ---------- window scan (features.py window_features_for_cell) ---------- */

// ownerAt: Map key(q,r) -> 0|1. Returns per-axis detail for demos plus the
// aggregate quantities the feature planes hold.
// Aggregates match the oracle exactly: line = max clean count, live = clean
// window count, liveK = clean windows holding >= K side stones.
export function scanWindows(ownerAt, xq, xr, me) {
  const other = 1 - me;
  const axes = [];
  const ownLineRaw = [0, 0, 0];
  const oppLineRaw = [0, 0, 0];
  for (let ai = 0; ai < 3; ai++) {
    const [dq, dr] = AXES[ai];
    const windows = [];
    let ownMax = 0, oppMax = 0, ownLive = 0, oppLive = 0;
    const ownLiveK = [0, 0, 0], oppLiveK = [0, 0, 0]; // >=3, >=4, >=5
    for (let offset = 0; offset < WINDOW_LEN; offset++) {
      const sq = xq - dq * offset, sr = xr - dr * offset;
      const cells = [];
      let ownC = 0, oppC = 0;
      for (let i = 0; i < WINDOW_LEN; i++) {
        const cq = sq + dq * i, cr = sr + dr * i;
        const owner = ownerAt.get(key(cq, cr));
        if (owner === me) ownC++;
        else if (owner === other) oppC++;
        cells.push([cq, cr]);
      }
      const cleanOwn = oppC === 0, cleanOpp = ownC === 0;
      if (cleanOwn) {
        ownLive++;
        if (ownC >= 3) ownLiveK[0]++;
        if (ownC >= 4) ownLiveK[1]++;
        if (ownC >= 5) ownLiveK[2]++;
        if (ownC > ownMax) ownMax = ownC;
      }
      if (cleanOpp) {
        oppLive++;
        if (oppC >= 3) oppLiveK[0]++;
        if (oppC >= 4) oppLiveK[1]++;
        if (oppC >= 5) oppLiveK[2]++;
        if (oppC > oppMax) oppMax = oppC;
      }
      windows.push({ cells, ownC, oppC, cleanOwn, cleanOpp });
    }
    ownLineRaw[ai] = ownMax; oppLineRaw[ai] = oppMax;
    axes.push({ name: AXIS_NAMES[ai], delta: AXES[ai], windows,
                ownLine: ownMax, oppLine: oppMax, ownLive, oppLive,
                ownLiveK, oppLiveK });
  }
  const forkOf = raw => raw.filter(c => c >= FORK_LINE_THRESHOLD).length;
  return { axes, ownFork: forkOf(ownLineRaw), oppFork: forkOf(oppLineRaw) };
}

/* ---------- ray walk (features.py ray_lengths_for_cell) ---------- */

// supportIndex: Map (from buildSupport().index) — membership test.
// Returns 12 entries in flat order side*6 + axis*2 + dir (side 0 = to-move),
// each { side, axis, dir, sign, length, cells, blockedBy } where cells are the
// walked coords (up to the stop) and blockedBy is "edge" | "blocker" | null.
export function walkRays(ownerAt, supportIndex, xq, xr, me) {
  const out = [];
  for (let side = 0; side < 2; side++) {
    const anti = side === 0 ? 1 - me : me;
    for (let ai = 0; ai < 3; ai++) {
      const [dq, dr] = AXES[ai];
      for (const [di, sign] of [[0, 1], [1, -1]]) {
        let length = 0, blockedBy = null;
        const cells = [];
        for (let j = 1; j <= RAY_REACH; j++) {
          const cq = xq + sign * dq * j, cr = xr + sign * dr * j;
          if (!supportIndex.has(key(cq, cr))) { blockedBy = "edge"; break; }
          cells.push([cq, cr]);
          length = j;
          if (ownerAt.get(key(cq, cr)) === anti) { blockedBy = "blocker"; break; }
        }
        out.push({ side, axis: ai, dir: di, sign, length, cells, blockedBy });
      }
    }
  }
  return out;
}

/* ---------- D6 coordinate action (geometry.py rot60 / reflect) ---------- */

export const rot60 = (q, r) => [-r, q + r];
export const reflect = (q, r) => [q, -q - r];

// Element g in 0..11: rotation by g*60° for g<6; for g>=6 reflect FIRST,
// then rotate (g-6) times — the exact composition of geometry.py apply_d6,
// matching the exported coord_mats/slot_perms element indexing.
export function applyD6(g, q, r) {
  if (g >= 6) { [q, r] = reflect(q, r); g -= 6; }
  for (let i = 0; i < g; i++) [q, r] = rot60(q, r);
  return [q, r];
}

/* ---------- SVG board helpers (same mapping as ../board.js) ---------- */

export const SQ3 = Math.sqrt(3);
export const axialX = (q, r, s) => s * SQ3 * (q + r / 2);
export const axialY = (q, r, s) => s * 1.5 * r;

export function hexPts(cx, cy, s) {
  const pts = [];
  for (let i = 0; i < 6; i++) {
    const a = Math.PI / 180 * (60 * i - 30);
    pts.push((cx + s * Math.cos(a)).toFixed(2) + "," + (cy + s * Math.sin(a)).toFixed(2));
  }
  return pts.join(" ");
}

const SVG_NS = "http://www.w3.org/2000/svg";
export const svgEl = (tag, attrs = {}) => {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

// Render a static figure board into an <svg>: draws every cell in `cells`
// (array of [q, r]) as a polygon, fits the viewBox, and returns a lookup of
// polygon elements plus a per-cell overlay group. Pages layer their demo
// state (stones, highlights, rays) on top via the returned API.
export function figBoard(svg, cells, opts = {}) {
  const s = opts.cellSize || 16;
  const pad = opts.pad || (s * 1.4);
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (const [q, r] of cells) {
    const x = axialX(q, r, s), y = axialY(q, r, s);
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  svg.setAttribute("viewBox",
    `${(minX - pad).toFixed(1)} ${(minY - pad).toFixed(1)} ` +
    `${(maxX - minX + 2 * pad).toFixed(1)} ${(maxY - minY + 2 * pad).toFixed(1)}`);
  const cellLayer = svgEl("g");
  const overlay = svgEl("g");
  const stoneLayer = svgEl("g");
  svg.appendChild(cellLayer);
  svg.appendChild(overlay);
  svg.appendChild(stoneLayer);
  const polys = new Map();
  for (const [q, r] of cells) {
    const cx = axialX(q, r, s), cy = axialY(q, r, s);
    const p = svgEl("polygon", {
      points: hexPts(cx, cy, s * 0.94),
      "data-q": q, "data-r": r,
    });
    cellLayer.appendChild(p);
    polys.set(key(q, r), p);
  }
  const stones = new Map();
  return {
    s, svg, polys, overlay,
    center: (q, r) => [axialX(q, r, s), axialY(q, r, s)],
    setStone(q, r, owner) {
      const k = key(q, r);
      let c = stones.get(k);
      if (owner == null) { if (c) { c.remove(); stones.delete(k); } return; }
      if (!c) {
        c = svgEl("circle", { cx: axialX(q, r, s), cy: axialY(q, r, s), r: s * 0.52 });
        stoneLayer.appendChild(c);
        stones.set(k, c);
      }
      c.setAttribute("class", "stone p" + owner);
    },
    clearStones() { for (const c of stones.values()) c.remove(); stones.clear(); },
    clearOverlay() { while (overlay.firstChild) overlay.firstChild.remove(); },
  };
}

// Convenience: the disk of cells with hex distance <= radius from origin.
export function disk(radius) {
  const out = [];
  for (let q = -radius; q <= radius; q++)
    for (let r = -radius; r <= radius; r++)
      if (hexDist(q, r) <= radius) out.push([q, r]);
  return out;
}

/* ---------- data ---------- */

export const loadData = name =>
  fetch("data/" + name).then(r => {
    if (!r.ok) throw new Error(name + ": " + r.status);
    return r.json();
  });
