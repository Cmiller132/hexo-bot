/* scene.js — the live-search scene: fold telemetry frames, judge them, paint
 * them onto any board. Shared by the play view (streaming SSE frames as they
 * arrive), the analysis deep look, and the lab (replaying collected frames).
 *
 * One scene per searched stone, updated in place the moment a frame is folded
 * and drawn in the same call. Every frame folds idempotently, so a replayed or
 * duplicated frame redraws the same picture instead of re-animating it.
 *
 * The scene keeps the two things the old overlay conflated apart. `base` is
 * the net's policy over every legal cell: painted once, never re-scaled,
 * never narrowed — it stays on the board for the whole search. `cands` is
 * everything the search itself does — candidacy, visits, running Q, the
 * ranking, elimination — and renders as border marks OVER the base. */

import { key } from "./board.js?v=14";

/* overlay tone families — deliberately NOT the stone colors: paler and
 * hue-shifted so a tinted empty cell can never be mistaken for a stone */
export const H0 = "#9fd0ff", H1 = "#ffb4aa";
export const H0R = "#d7ebff", H1R = "#ffddd6";

export function coordKey(row) {
  return row && Number.isFinite(row.q) && Number.isFinite(row.r)
    ? key(row.q, row.r) : null;
}

export function newScene(stoneKey) {
  return {
    stoneKey,
    base: new Map(),      // key -> {q,r,p}: the prior, held for the readout
    baseScope: "",        // "" unpainted, "board" full prior, "rest" non-candidates
    cands: new Map(),     // key -> {q,r,visits,value,score,cut}
    cand0: 0,             // candidate count at the draw; sets the arc scale
    round: null, rounds: null, visits: null, target: null,
    chosen: null, complete: false, rootValue: null,
    tss: false, lcbOverride: false, playPruned: false,
  };
}

/* Fold one frame into the scene. */
export function foldLiveFrame(scene, event) {
  for (const field of ["round", "rounds", "visits"]) {
    if (Number.isInteger(event[field])) scene[field] = event[field];
  }
  if (Number.isInteger(event.target_visits)) scene.target = event.target_visits;

  const rows = (Array.isArray(event.policy) ? event.policy : [])
    .filter(row => coordKey(row) !== null);
  if (event.kind === "bare_policy") {
    for (const row of rows) {
      scene.base.set(key(row.q, row.r), {
        q: Number(row.q), r: Number(row.r),
        p: Number.isFinite(row.p) ? Math.max(0, Number(row.p)) : 0,
      });
    }
    return;
  }

  for (const row of rows) {
    const k = key(row.q, row.r);
    let cand = scene.cands.get(k);
    if (!cand) {
      cand = {
        q: Number(row.q), r: Number(row.r),
        visits: 0, value: null, score: null, cut: false,
      };
      scene.cands.set(k, cand);
    }
    const rowVisits = Number.isFinite(row.visits) ? Number(row.visits) : null;
    if (rowVisits !== null && rowVisits > cand.visits) cand.visits = rowVisits;
    // A frame that reports visit counts also reports a placeholder 0.0 Q for
    // lines nothing has visited yet; only a visited line's Q is real. Frames
    // without visit counts (the post-search replay of the non-telemetry
    // families) carry only real values.
    if (Number.isFinite(row.value) && (rowVisits === null || rowVisits > 0)) {
      cand.value = Number(row.value);
    }
    if (Number.isFinite(row.p)) cand.score = Math.max(0, Number(row.p));
  }
  if (event.kind === "candidate_set") {
    scene.cand0 = Math.max(scene.cand0, scene.cands.size);
  }

  // The survivor set is the halving's own report of who is still in the
  // running. Elimination is one-way within a search: a line a cut dropped
  // never comes back, so a stale re-listed frame cannot relight it.
  const survivors = Array.isArray(event.survivors) ? event.survivors : [];
  if (survivors.length) {
    const keep = new Set(survivors.map(coordKey).filter(Boolean));
    for (const [k, cand] of scene.cands) {
      if (!keep.has(k)) cand.cut = true;
    }
  }

  if (event.kind === "search_complete") {
    scene.complete = true;
    scene.chosen = coordKey(event.action)
      ? { q: Number(event.action.q), r: Number(event.action.r) } : null;
    scene.rootValue = Number.isFinite(event.root_value)
      ? Number(event.root_value) : null;
    scene.tss = event.tss === true;
    scene.lcbOverride = event.lcb_override === true;
    scene.playPruned = event.play_pruned === true;
  }
}

/* The line the search currently favours: the top-scored candidate still in
 * the running. Mid-search it takes the solid ring; on the decision frame it
 * becomes the disagreement marker if the bot played somewhere else. */
export function sceneLeader(scene) {
  let leader = null;
  for (const cand of scene.cands.values()) {
    if (cand.cut || !Number.isFinite(cand.score)) continue;
    if (!leader || cand.score > leader.score ||
        (cand.score === leader.score &&
          (cand.q < leader.q || (cand.q === leader.q && cand.r < leader.r)))) {
      leader = cand;
    }
  }
  return leader ? { q: leader.q, r: leader.r, score: leader.score } : null;
}

/* One short clause naming why the played move is the played move, for the
 * decisions the marks alone would not explain. Near-ties get no clause: the
 * board already draws them equally, so there is nothing to explain. A
 * certificate always says so: it means the search result was discarded
 * outright. */
export function sceneVerdict(scene) {
  if (scene.tss) return "proven win";
  const played = coordKey(scene.chosen);
  const leader = sceneLeader(scene);
  if (!played || !leader || key(leader.q, leader.r) === played) return "";
  const playedCand = scene.cands.get(played);
  if (playedCand && Number.isFinite(playedCand.score) &&
      playedCand.score >= leader.score * 0.9) return "";
  if (scene.lcbOverride) return "value pick";
  if (scene.playPruned) return "sampled";
  return "tie-break";
}

/* Draw the whole scene onto `board` for `botColor` as the searching side.
 * The base is handed to the board at most once per scope -- outside the
 * candidate set the policy fill is constant by design, so there is nothing
 * to re-paint. The marks re-derive every time from candidate state:
 *
 *   fill  the search's CURRENT ranking, the only fill on candidate cells
 *         (the base withdraws there): the favourite is always the visibly
 *         brightest cell, and a cut line's fill sinks with its mark to a
 *         faint remnant.
 *   arc   effort. v/(v+share) of the outline, where share = target/candidates
 *         is one line's fair slice of the budget -- so a half-filled outline
 *         reads "got its fair share" and a full one "got far more", at any
 *         budget, for any family.
 *   tick  judgment, on the site-wide value convention (blue = P0 favoured,
 *         red = P1) -- the engine reports Q from the searching side's view,
 *         so the bot's colour flips it to the absolute frame.
 *   cut   elimination; the mark dims and freezes but stays on the board. */
export function paintSceneTo(scene, board, botColor) {
  const tint = botColor === 0 ? H0 : H1;
  const ringTint = botColor === 0 ? H0R : H1R;
  // The base is the prior over every legal cell until the candidate set is
  // known, then the prior over every NON-candidate cell: on the candidate
  // cells the mark's live ranking fill is the only fill, so "brightest cell"
  // can never mean "had a strong prior once" when the search has moved on.
  // Both paints share the full-board prior maximum as the brightness basis,
  // so withdrawing the candidates never re-scales the rest of the board.
  const baseScope = scene.cands.size ? "rest" : "board";
  if (scene.base.size && scene.baseScope !== baseScope) {
    scene.baseScope = baseScope;
    let maxP = 0;
    for (const row of scene.base.values()) maxP = Math.max(maxP, row.p);
    const rows = [...scene.base.values()].filter(row =>
      baseScope === "board" || !scene.cands.has(key(row.q, row.r)));
    board.setLiveBase(rows, tint, 0.8, maxP);
  }
  if (!scene.cands.size) return;
  const share = Math.max(
    1, (scene.target || 0) / Math.max(1, scene.cand0 || scene.cands.size)
  );
  let maxScore = 0;
  for (const cand of scene.cands.values()) {
    if (!cand.cut && Number.isFinite(cand.score)) {
      maxScore = Math.max(maxScore, cand.score);
    }
  }
  const marks = [];
  for (const [, cand] of scene.cands) {
    marks.push({
      q: cand.q, r: cand.r,
      fill: maxScore > 0 && Number.isFinite(cand.score)
        ? Math.min(1, Math.max(0, cand.score) / maxScore) : 0,
      arc: cand.visits > 0 ? cand.visits / (cand.visits + share) : 0,
      tick: Number.isFinite(cand.value)
        ? (botColor === 0 ? cand.value : -cand.value) : null,
      cut: cand.cut,
    });
  }
  board.setSearchMarks(marks, tint, ringTint, scene.chosen, sceneLeader(scene));
}
