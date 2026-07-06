/* lab_features.js — client-side mirror of Shrimp's input featurization.
 *
 * Pure module (no DOM, no imports) so the same file runs in the browser and
 * under node for the parity test (tests/lab_featurizer_check.mjs compares its
 * output to the server featurizer, packages/shrimp/python/shrimp/features.py,
 * to 1e-6). The mirrored pieces, with their Python sources:
 *
 *   - turn structure (1 then 2-2-2...)          features.record_phase/record_player
 *   - engine legality (empty && dist <= 8)      support.py docstring / engine legal.rs
 *   - support set [legal | stones | halo]       support.build_support, radius 4
 *     (SHRIMP_SUPPORT_RADIUS=4 is the published main_7 recipe; the model sees
 *     legal cells within hex-dist 4 of a stone, halo = the dist-5 shell)
 *   - window scan (hot / standing-win cells)    features.window_scan
 *   - the 15 feature planes                     features.build_features
 *
 * Free-edit positions zero the three history-derived planes (own/opp recency,
 * opp last turn), matching the server's build_free_position.
 */

export const LEGAL_RADIUS = 8;        // engine legality radius
export const SUPPORT_RADIUS = 4;      // model support radius (main_7 recipe)
export const SUPPORT_HALO = SUPPORT_RADIUS + 1;
export const DIST_SCALE = LEGAL_RADIUS;

export const NUM_FEATURES = 15;
export const F = {
  OWN_STONE: 0, OPP_STONE: 1, EMPTY: 2, LEGAL: 3, PHASE_SECOND: 4,
  FIRST_STONE: 5, PLAYER_COLOUR: 6, OWN_RECENCY: 7, OPP_RECENCY: 8,
  OPP_HOT: 9, OWN_HOT: 10, DIST_TO_STONE: 11, OPP_LAST_TURN: 12,
  OPP_WIN_NOW: 13, OWN_WIN_NOW: 14,
};
export const FEATURE_NAMES = [
  "own_stone", "opp_stone", "empty", "legal", "phase_second", "first_stone",
  "player_colour", "own_recency", "opp_recency", "opp_hot", "own_hot",
  "dist_to_stone", "opp_last_turn", "opp_win_now", "own_win_now",
];
export const FREE_ZEROED = ["own_recency", "opp_recency", "opp_last_turn"];

const WINDOW_LEN = 6;
const HOT_MIN_COUNT = 4;
const WIN_NOW_COUNT = 5;
const HOT_MIN_PLACEMENTS = 7;

// Fixed direction order (constants.DIRECTIONS: the rotate60 orbit of (1,0)).
const DIRECTIONS = [[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]];
// Win axes (features.AXIS_DELTAS): Q, R, QR.
const AXES = [[1, 0], [0, 1], [1, -1]];

const key = (q, r) => q + "," + r;

export function hexDist(q1, r1, q2, r2) {
  const dq = q1 - q2, dr = r1 - r2;
  return (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2;
}

/* ---- turn structure (features.record_phase / record_player) -------------- */

export function recordPhase(ordinal) {
  if (ordinal === 0) return "Opening";
  return (ordinal - 1) % 2 === 0 ? "FirstStone" : "SecondStone";
}

export function recordPlayer(ordinal) {
  if (ordinal === 0) return 0;
  return Math.floor((ordinal - 1) / 2) % 2 === 0 ? 1 : 0;
}

/* ---- engine legality (editor rules) --------------------------------------- */

/* Placement legality on the position after `moves` (array of [q, r]):
 * the opening is forced to the origin; afterwards any empty cell within
 * hex-dist LEGAL_RADIUS of a stone is legal. */
export function isLegalPlacement(moves, q, r) {
  if (moves.length === 0) return q === 0 && r === 0;
  let empty = true, near = false;
  for (const [sq, sr] of moves) {
    if (sq === q && sr === r) { empty = false; break; }
    if (hexDist(sq, sr, q, r) <= LEGAL_RADIUS) near = true;
  }
  return empty && near;
}

/* The full legal set as [q, r] pairs (for the board's hover ghost): a BFS of
 * depth LEGAL_RADIUS from the stones. Empty board => the forced origin. */
export function legalCells(moves) {
  if (moves.length === 0) return [[0, 0]];
  const dist = bfsDist(moves, LEGAL_RADIUS);
  const stoneSet = new Set(moves.map(([q, r]) => key(q, r)));
  const out = [];
  for (const [k, d] of dist) {
    if (d <= LEGAL_RADIUS && !stoneSet.has(k)) {
      const [q, r] = k.split(",").map(Number);
      out.push([q, r]);
    }
  }
  return out;
}

/* ---- support set (support.build_support at SUPPORT_RADIUS) ---------------- */

function bfsDist(stones, depth) {
  const dist = new Map();
  let frontier = [];
  for (const [q, r] of stones) {
    const k = key(q, r);
    if (!dist.has(k)) { dist.set(k, 0); frontier.push([q, r]); }
  }
  for (let d = 0; d < depth; d++) {
    const next = [];
    for (const [q, r] of frontier) {
      for (const [dq, dr] of DIRECTIONS) {
        const nq = q + dq, nr = r + dr, k = key(nq, nr);
        if (!dist.has(k)) { dist.set(k, d + 1); next.push([nq, nr]); }
      }
    }
    frontier = next;
  }
  return dist;
}

const lexQR = (a, b) => a[0] - b[0] || a[1] - b[1];

/* Support in node order [legal | stones | halo], each segment ascending by
 * signed (q, r). Returns {coords, legalCount, stoneCount, haloCount, dist,
 * index} — index maps "q,r" -> row. Empty stone list => origin + 6 halo
 * neighbours, dist all 0 (matching support._build_support). */
export function buildSupport(stones) {
  if (stones.length === 0) {
    const ordered = [[0, 0]].concat(DIRECTIONS.slice().sort(lexQR));
    const index = new Map(ordered.map((c, i) => [key(c[0], c[1]), i]));
    return {
      coords: ordered, legalCount: 1, stoneCount: 0, haloCount: 6,
      dist: new Array(7).fill(0), index,
    };
  }
  const dist = bfsDist(stones, SUPPORT_HALO);
  const stoneSet = new Set(stones.map(([q, r]) => key(q, r)));
  const legal = [], halo = [];
  for (const [k, d] of dist) {
    if (stoneSet.has(k)) continue;
    const cell = k.split(",").map(Number);
    if (d <= SUPPORT_RADIUS) legal.push(cell);
    else if (d === SUPPORT_HALO) halo.push(cell);
  }
  legal.sort(lexQR);
  halo.sort(lexQR);
  const stonesSorted = stones.slice().sort(lexQR);
  const coords = legal.concat(stonesSorted, halo);
  const index = new Map(coords.map((c, i) => [key(c[0], c[1]), i]));
  return {
    coords,
    legalCount: legal.length,
    stoneCount: stonesSorted.length,
    haloCount: halo.length,
    dist: coords.map(([q, r]) => dist.get(key(q, r))),
    index,
  };
}

/* ---- window scan (features.window_scan) ----------------------------------- */

/* records: [{q, r, owner, idx}]. Returns four Sets of "q,r" cells. */
export function windowScan(records, currentPlayer, placementsMade) {
  const ownerAt = new Map(records.map(rec => [key(rec.q, rec.r), rec.owner]));
  const ownHot = new Set(), oppHot = new Set(), ownWin = new Set(), oppWin = new Set();
  const seen = new Set();
  for (const [k] of ownerAt) {
    const [sq, sr] = k.split(",").map(Number);
    for (let axis = 0; axis < 3; axis++) {
      const [dq, dr] = AXES[axis];
      for (let back = 0; back < WINDOW_LEN; back++) {
        const q0 = sq - back * dq, r0 = sr - back * dr;
        const wk = q0 + "," + r0 + "," + axis;
        if (seen.has(wk)) continue;
        seen.add(wk);
        let c0 = 0, c1 = 0;
        const empties = [];
        for (let i = 0; i < WINDOW_LEN; i++) {
          const cell = key(q0 + i * dq, r0 + i * dr);
          const owner = ownerAt.get(cell);
          if (owner === undefined) empties.push(cell);
          else if (owner === 0) c0++;
          else c1++;
        }
        if (c0 > 0 && c1 > 0) continue; // two-coloured window: skipped
        const count = c0 + c1;
        const owner = c0 > 0 ? 0 : 1;
        if (count === WIN_NOW_COUNT) {
          const set = owner === currentPlayer ? ownWin : oppWin;
          for (const cell of empties) set.add(cell);
        }
        if (count >= HOT_MIN_COUNT && placementsMade >= HOT_MIN_PLACEMENTS) {
          const set = owner === currentPlayer ? ownHot : oppHot;
          for (const cell of empties) set.add(cell);
        }
      }
    }
  }
  return { ownHot, oppHot, ownWin, oppWin };
}

/* ---- facts ------------------------------------------------------------------ */

/* Facts for a legal placement sequence (moves: [[q, r], ...] in order). Owner
 * and phase per record come from the deterministic turn structure. */
export function factsFromSequence(moves) {
  const n = moves.length;
  // placement_index is 1-based on the engine mirror (the newest stone's
  // recency age n - idx is 0, weight 1).
  const records = moves.map(([q, r], i) => ({ q, r, owner: recordPlayer(i), idx: i + 1 }));
  const currentPlayer = recordPlayer(n);
  const phase = n === 0 ? "Opening" : recordPhase(n);
  const firstStone = phase === "SecondStone" ? [moves[n - 1][0], moves[n - 1][1]] : null;
  return {
    records, currentPlayer, phase, firstStone,
    placementsMade: n,
    ...windowScan(records, currentPlayer, n),
    free: false,
  };
}

/* Facts for a free-edit position (p0/p1: [[q, r], ...]). No real history:
 * phase is pinned to FirstStone and the history planes are zeroed in
 * buildFeatures (matching the server's build_free_position). */
export function factsFromFree(p0, p1, toMove) {
  const ordered = p0.map(([q, r]) => [q, r, 0])
    .concat(p1.map(([q, r]) => [q, r, 1]))
    .sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const records = ordered.map(([q, r, owner], i) => ({ q, r, owner, idx: i }));
  return {
    records, currentPlayer: toMove, phase: "FirstStone", firstStone: null,
    placementsMade: records.length,
    ...windowScan(records, toMove, records.length),
    free: true,
  };
}

/* Default side to move for a free-edit position (lab_rules.default_free_to_move). */
export const defaultFreeToMove = (n0, n1) => (n0 > n1 ? 1 : 0);

/* ---- feature build (features.build_features) ------------------------------- */

/* Returns planes[NUM_FEATURES][N] (feature-major, matching the server's
 * wants.features payload), node order = sup.coords order. */
export function buildFeatures(facts, sup) {
  const n = sup.coords.length;
  const planes = [];
  for (let f = 0; f < NUM_FEATURES; f++) planes.push(new Float32Array(n));

  const made = facts.placementsMade;
  for (const rec of facts.records) {
    const row = sup.index.get(key(rec.q, rec.r));
    const own = rec.owner === facts.currentPlayer;
    planes[own ? F.OWN_STONE : F.OPP_STONE][row] = 1.0;
    const recPlane = planes[own ? F.OWN_RECENCY : F.OPP_RECENCY];
    const weight = 1.0 / (1.0 + (made - rec.idx));
    recPlane[row] = Math.max(recPlane[row], weight);
  }
  for (let i = 0; i < n; i++) {
    planes[F.EMPTY][i] = 1.0 - planes[F.OWN_STONE][i] - planes[F.OPP_STONE][i];
  }
  for (let i = 0; i < sup.legalCount; i++) planes[F.LEGAL][i] = 1.0;

  if (facts.phase === "SecondStone") {
    planes[F.PHASE_SECOND].fill(1.0);
    if (facts.firstStone) {
      planes[F.FIRST_STONE][sup.index.get(key(facts.firstStone[0], facts.firstStone[1]))] = 1.0;
    }
  }
  if (facts.currentPlayer === 0) planes[F.PLAYER_COLOUR].fill(1.0);

  const mark = (cells, plane) => {
    for (const cell of cells) {
      const row = sup.index.get(cell);
      if (row !== undefined) planes[plane][row] = 1.0;
    }
  };
  mark(facts.oppHot, F.OPP_HOT);
  mark(facts.ownHot, F.OWN_HOT);
  mark(facts.oppWin, F.OPP_WIN_NOW);
  mark(facts.ownWin, F.OWN_WIN_NOW);

  for (let i = 0; i < n; i++) planes[F.DIST_TO_STONE][i] = sup.dist[i] / DIST_SCALE;

  if (!facts.free) {
    for (const cell of oppLastTurnCells(facts)) {
      planes[F.OPP_LAST_TURN][sup.index.get(cell)] = 1.0;
    }
  } else {
    // Free-edit: zero the history-derived planes (recency was built from the
    // synthesized ordinals above; wipe it to match the server).
    planes[F.OWN_RECENCY].fill(0.0);
    planes[F.OPP_RECENCY].fill(0.0);
    // opp_last_turn is never built in free mode.
  }
  return planes.map(p => Array.from(p));
}

/* features._opp_last_turn_cells: the opponent's most recent full turn. */
function oppLastTurnCells(facts) {
  const opponent = 1 - facts.currentPlayer;
  const records = facts.records;
  for (let ordinal = records.length - 1; ordinal >= 0; ordinal--) {
    if (recordPlayer(ordinal) !== opponent) continue;
    const phase = recordPhase(ordinal);
    const rec = records[ordinal];
    if (phase === "SecondStone") {
      const first = records[ordinal - 1];
      return [key(first.q, first.r), key(rec.q, rec.r)];
    }
    if (phase === "Opening") return [key(rec.q, rec.r)];
  }
  return [];
}
