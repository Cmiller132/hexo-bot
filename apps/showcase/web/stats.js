/* stats.js — the Stats tab: totals strip, ELO leaderboard, human-winrate-by-bot
 * chart, and a server-filtered game history table.
 *
 * app.js owns tab activation and the "open a game in analysis by id" flow; it
 * calls initStats() once with the hooks it needs (openGame, catalogue) and
 * refreshStats() on each activation. Everything else — fetching /api/stats and
 * /api/games, rendering, and wiring the filter/sort/pagination controls — lives
 * here so the Play/Analysis controllers stay untouched.
 *
 * Fetch is a thin local helper (api.js's request() is module-private); the
 * shapes are the frozen /api/stats + /api/games contract.
 */

import { groupByModel } from "./checkpoints.js?v=11";

const $ = id => document.getElementById(id);

/* ---- fetch helpers (contract-shaped, best-effort) ------------------------- */

async function getJSON(path) {
  const resp = await fetch(path, { credentials: "same-origin" });
  let data = null;
  try { data = await resp.json(); } catch (_) { /* non-JSON body */ }
  if (!resp.ok) {
    const err = new Error((data && (data.detail || data.message)) || `HTTP ${resp.status}`);
    err.status = resp.status;
    throw err;
  }
  return data;
}

const PAGE = 25; // history rows per page

/* Injected by app.js: openGame(id) reuses the existing analysis load-by-id
 * flow; getCatalogue() returns the normalized /api/bots payload (checkpoints +
 * sims) read live, so a late bots load still populates the filter selects. */
let deps = { openGame: () => {}, getCatalogue: () => null };

/* Cache the last /api/stats payload so re-activation is cheap; a fresh fetch is
 * still issued on each activation (games finish between visits). */
const state = {
  loadedControls: false,
  offset: 0,
  total: 0,
  lastCount: 0,
  querying: false,
};

export function initStats(hooks) {
  deps = { ...deps, ...hooks };
  wireControls();
}

/* Called by app.js on every activation of the stats tab. */
export function refreshStats() {
  ensureFilterOptions();
  loadStats();
  queryHistory(true); // reset to page 0 on (re)entry
}

/* ---- /api/stats: totals + leaderboard + winrate chart --------------------- */

async function loadStats() {
  try {
    const s = await getJSON("/api/stats");
    renderTotals(s.totals || {});
    renderLeaderboard(s.leaderboard || {});
    renderWinrateChart(s.leaderboard && s.leaderboard.bots ? s.leaderboard.bots : []);
    $("lbNote").hidden = true;
    $("wrNote").hidden = true;
  } catch (e) {
    if (e.status === 404) {
      note("lbNote", "leaderboard not available on this server build");
      note("wrNote", "winrate chart not available on this server build");
    } else {
      note("lbNote", "couldn't load the leaderboard");
      note("wrNote", "couldn't load the winrate chart");
    }
  }
}

function note(id, msg) {
  const el = $(id);
  if (!el) return;
  el.hidden = false;
  el.textContent = msg;
}

function renderTotals(t) {
  const set = (id, v) => { const el = $(id); if (el) el.textContent = Number.isFinite(v) ? v : "—"; };
  set("totGames", t.games);
  set("totPlayers", t.players);
  set("totBots", t.bots);
  set("totDraws", t.draws);
}

// ---- leaderboard ------------------------------------------------------------

/* Bar width is the rating scaled within the column's own [min,max] span, with a
 * floor so even the lowest-rated row shows a sliver. Ratings cluster near 1000,
 * so a min-anchored scale reads far better than a 0-anchored one. */
function barScale(ratings) {
  const finite = ratings.filter(Number.isFinite);
  if (!finite.length) return () => 0;
  const min = Math.min(...finite), max = Math.max(...finite);
  const span = max - min;
  return r => {
    if (!Number.isFinite(r)) return 0.06;
    if (span <= 0) return 1;
    return 0.12 + 0.88 * ((r - min) / span);
  };
}

function pct(x) { return (Math.max(0, Math.min(1, x)) * 100).toFixed(1) + "%"; }

function renderLeaderboard(lb) {
  renderLbColumn($("lbPlayers"), (lb.players || []), "player");
  renderLbColumn($("lbBots"), (lb.bots || []), "bot");
}

function renderLbColumn(host, rows, kind) {
  if (!host) return;
  host.textContent = "";
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "lb-empty";
    empty.textContent = kind === "player" ? "no rated players yet" : "no rated bots yet";
    host.appendChild(empty);
    return;
  }
  const scale = barScale(rows.map(r => r.rating));
  rows.forEach((r, i) => {
    const row = document.createElement("div");
    row.className = "lb-row" + (kind === "bot" ? " bot" : "");

    const bar = document.createElement("div");
    bar.className = "lb-bar";
    bar.style.width = pct(scale(r.rating));
    row.appendChild(bar);

    const body = document.createElement("div");
    body.className = "lb-body";
    const rank = document.createElement("span");
    rank.className = "lb-rank";
    rank.textContent = "#" + (i + 1);
    const name = document.createElement("span");
    name.className = "lb-name";
    if (kind === "player") {
      if (!r.nickname) {
        const anon = document.createElement("span");
        anon.className = "anon";
        anon.textContent = r.display || "(anonymous)";
        name.appendChild(anon);
      } else {
        name.textContent = r.display || r.nickname;
      }
    } else {
      name.textContent = `${r.checkpoint_id} · ${r.sims} sims`;
    }
    const rating = document.createElement("span");
    rating.className = "lb-rating";
    rating.textContent = r.rating;
    body.appendChild(rank);
    body.appendChild(name);
    body.appendChild(rating);
    row.appendChild(body);

    const sub = document.createElement("div");
    sub.className = "lb-sub";
    const games = document.createElement("span");
    games.textContent = `${r.games} game${r.games === 1 ? "" : "s"}`;
    sub.appendChild(games);
    const wr = document.createElement("span");
    if (kind === "player") {
      wr.textContent = `${pct(r.winrate)} win`;
    } else {
      // bot column: show the human's winrate against this bot (how beatable it is)
      wr.textContent = `${pct(r.human_winrate)} human win`;
    }
    sub.appendChild(wr);
    if (r.provisional) {
      const prov = document.createElement("span");
      prov.className = "lb-prov";
      prov.textContent = "provisional";
      sub.appendChild(prov);
    }
    row.appendChild(sub);
    host.appendChild(row);
  });
}

// ---- winrate-by-bot chart ---------------------------------------------------

/* Order bots the way the pickers do: checkpoints strongest-first (by run
 * ordinal), and within a checkpoint by sims ascending. Reuses groupByModel's
 * run ordering so this chart never drifts from the play/analysis pickers. */
function orderBots(bots) {
  // synthesize checkpoint entries so groupByModel can order the runs for us
  const seen = new Map(); // run -> order index
  const runOrder = [];
  const byRun = new Map();
  for (const b of bots) {
    const run = b.run || "";
    if (!byRun.has(run)) { byRun.set(run, []); runOrder.push({ id: run + "@x", run }); }
    byRun.get(run).push(b);
  }
  const models = groupByModel(runOrder); // strongest-first run ordering
  models.forEach((m, i) => seen.set(m.run, i));
  const out = [];
  for (const m of models) {
    const group = (byRun.get(m.run) || []).slice()
      .sort((a, b) => (a.sims - b.sims) || (a.epoch - b.epoch));
    out.push(...group);
  }
  // any bots whose run groupByModel didn't place (shouldn't happen) trail behind
  for (const b of bots) if (!seen.has(b.run || "")) out.push(b);
  return out;
}

function renderWinrateChart(bots) {
  const svg = $("wrChart");
  if (!svg) return;
  const NS = "http://www.w3.org/2000/svg";
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const rated = orderBots(bots.filter(b => Number.isFinite(b.games) && b.games > 0));
  if (!rated.length) {
    note("wrNote", "no games played yet");
    return;
  }
  $("wrNote").hidden = true;

  const rowH = 26, gap = 8, padL = 128, padR = 44, padT = 8, padB = 8;
  const barMax = 360; // logical bar-track width
  const W = padL + barMax + padR;
  const H = padT + padB + rated.length * rowH - gap;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const mk = (tag, attrs, cls) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (cls) e.setAttribute("class", cls);
    return e;
  };

  // 50% reference line
  const midX = padL + barMax * 0.5;
  svg.appendChild(mk("line", { x1: midX, y1: padT, x2: midX, y2: H - padB }, "wr-axis"));

  rated.forEach((b, i) => {
    const y = padT + i * rowH;
    const wr = Number.isFinite(b.human_winrate) ? b.human_winrate : 0;
    // human winrate: blue bar (--p0) since it's the human/player-1 palette
    svg.appendChild(mk("rect", { x: padL, y: y + 3, width: barMax, height: rowH - gap - 3 }, "wr-track"));
    svg.appendChild(mk("rect", {
      x: padL, y: y + 3, width: Math.max(1, barMax * wr), height: rowH - gap - 3,
    }, "wr-bar0"));

    const lab = mk("text", { x: padL - 8, y: y + (rowH - gap) / 2 + 3, "text-anchor": "end" }, "wr-lab");
    lab.textContent = `${b.checkpoint_id}·${b.sims}`;
    svg.appendChild(lab);

    const val = mk("text", { x: padL + barMax + 6, y: y + (rowH - gap) / 2 + 3 }, "wr-val");
    val.textContent = pct(wr);
    svg.appendChild(val);
  });
}

/* ---- filter controls -------------------------------------------------------- */

/* Populate the checkpoint + sims <select>s from the /api/bots catalogue passed
 * in by app.js. Done once (the catalogue is stable for a session). */
function ensureFilterOptions() {
  if (state.loadedControls) return;
  const cat = deps.getCatalogue();
  if (!cat) return;
  const ck = $("fCkpt"), sm = $("fSims");
  if (ck && Array.isArray(cat.checkpoints)) {
    // strongest-first, matching the pickers
    const models = groupByModel(cat.checkpoints);
    for (const m of models) {
      for (const c of m.items) {
        const opt = document.createElement("option");
        opt.value = c.id;
        opt.textContent = c.label;
        ck.appendChild(opt);
      }
    }
  }
  if (sm && Array.isArray(cat.sims)) {
    for (const s of cat.sims) {
      const opt = document.createElement("option");
      opt.value = String(s);
      opt.textContent = `${s} sims`;
      sm.appendChild(opt);
    }
  }
  state.loadedControls = true;
}

function wireControls() {
  const reset = () => queryHistory(true); // any filter/sort change resets to page 0
  ["fNick", "fCkpt", "fSims", "fResult", "fSort"].forEach(id => {
    const el = $(id);
    if (!el) return;
    const ev = el.tagName === "INPUT" ? "input" : "change";
    let t = null;
    el.addEventListener(ev, () => {
      if (ev === "input") { clearTimeout(t); t = setTimeout(reset, 300); } // debounce typing
      else reset();
    });
  });
  const prev = $("histPrev"), next = $("histNext");
  if (prev) prev.addEventListener("click", () => {
    if (state.offset <= 0) return;
    state.offset = Math.max(0, state.offset - PAGE);
    queryHistory(false);
  });
  if (next) next.addEventListener("click", () => {
    if (state.offset + PAGE >= state.total) return;
    state.offset += PAGE;
    queryHistory(false);
  });
}

/* ---- /api/games (filtered) ------------------------------------------------- */

function historyParams() {
  const p = new URLSearchParams();
  const nick = ($("fNick") && $("fNick").value.trim()) || "";
  const ckpt = ($("fCkpt") && $("fCkpt").value) || "";
  const sims = ($("fSims") && $("fSims").value) || "";
  const result = ($("fResult") && $("fResult").value) || "";
  const sort = ($("fSort") && $("fSort").value) || "recent";
  if (nick) p.set("nickname", nick);
  if (ckpt) p.set("checkpoint_id", ckpt);
  if (sims) p.set("sims", sims);
  if (result) p.set("result", result);
  p.set("sort", sort);
  p.set("limit", String(PAGE));
  p.set("offset", String(state.offset));
  return p;
}

async function queryHistory(resetPage) {
  if (resetPage) state.offset = 0;
  if (state.querying) return;
  state.querying = true;
  try {
    const data = await getJSON("/api/games?" + historyParams().toString());
    const games = Array.isArray(data.games) ? data.games : [];
    state.total = Number.isFinite(data.total) ? data.total : games.length;
    renderHistory(games);
    updatePager(games.length);
  } catch (e) {
    renderHistory([]);
    const empty = $("histEmpty");
    if (empty) {
      empty.hidden = false;
      empty.textContent = e.status === 404
        ? "game history not available on this server build"
        : "couldn't load game history";
    }
    $("histCount").textContent = "";
    $("histPrev").disabled = true;
    $("histNext").disabled = true;
  } finally {
    state.querying = false;
  }
}

function fmtDate(iso) {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const d = new Date(t);
  const pad = n => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function fmtDur(s) {
  if (!Number.isFinite(s) || s < 0) return "—";
  if (s < 60) return Math.round(s) + "s";
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  return `${m}m ${pad2(sec)}s`;
}
function pad2(n) { return String(n).padStart(2, "0"); }

/* Human-perspective outcome from a /api/games item. The contract's result block
 * carries winner + human_result; fall back to comparing winner vs human_color. */
function outcomeOf(g) {
  const hc = g.human_color ?? 0;
  const res = g.result || {};
  let hr = Number.isInteger(res.human_result) ? res.human_result : null;
  const winner = res.winner ?? null;
  if (hr === null && winner !== null && Number.isInteger(hc)) hr = winner === hc ? 1 : -1;
  if (hr === 1) return { t: "human won", c: "win", winner };
  if (hr === -1) return { t: "bot won", c: "loss", winner };
  return { t: "draw", c: "draw", winner };
}

function renderHistory(games) {
  const body = $("histBody");
  const empty = $("histEmpty");
  if (!body) return;
  body.textContent = "";
  if (!games.length) {
    if (empty) { empty.hidden = false; empty.textContent = "no games match these filters"; }
    return;
  }
  if (empty) empty.hidden = true;

  const NS = "http://www.w3.org/2000/svg";
  const frag = document.createDocumentFragment();
  for (const g of games) {
    const oc = outcomeOf(g);
    const tr = document.createElement("tr");
    tr.className = "hist-row";
    tr.dataset.id = g.id;

    const bot = g.bot || {};
    const nick = g.nickname || "";

    // date
    const tdDate = document.createElement("td");
    tdDate.textContent = fmtDate(g.finished_at);
    tr.appendChild(tdDate);

    // player (winner-tinted hex glyph + name)
    const tdPlayer = document.createElement("td");
    tdPlayer.className = "ht-player";
    const cls = oc.winner === 0 ? "gh0" : oc.winner === 1 ? "gh1" : "ghx";
    const glyph = document.createElementNS(NS, "svg");
    glyph.setAttribute("class", "hist-glyph");
    glyph.setAttribute("width", "10");
    glyph.setAttribute("height", "11");
    glyph.setAttribute("viewBox", "-5.5 -5.5 11 11");
    glyph.setAttribute("aria-hidden", "true");
    const poly = document.createElementNS(NS, "polygon");
    poly.setAttribute("class", cls);
    poly.setAttribute("points", "4.33,-2.5 4.33,2.5 0,5 -4.33,2.5 -4.33,-2.5 0,-5");
    glyph.appendChild(poly);
    tdPlayer.appendChild(glyph);
    const nameSpan = document.createElement("span");
    if (nick) nameSpan.textContent = nick;
    else { nameSpan.className = "anon"; nameSpan.textContent = "(anonymous)"; }
    tdPlayer.appendChild(nameSpan);
    tr.appendChild(tdPlayer);

    // opponent + sims
    const tdOpp = document.createElement("td");
    tdOpp.textContent = `${bot.label || bot.checkpoint_id || "?"} · ${bot.sims ?? bot.visits ?? "?"} sims`;
    tr.appendChild(tdOpp);

    // result
    const tdRes = document.createElement("td");
    tdRes.className = "ht-res " + oc.c;
    tdRes.textContent = oc.t;
    tr.appendChild(tdRes);

    // plies
    const tdPly = document.createElement("td");
    tdPly.textContent = g.ply_count ?? "—";
    tr.appendChild(tdPly);

    // duration
    const tdDur = document.createElement("td");
    tdDur.textContent = fmtDur(g.duration_s);
    tr.appendChild(tdDur);

    frag.appendChild(tr);
  }
  body.appendChild(frag);
}

$("histBody") && $("histBody").addEventListener("click", e => {
  const tr = e.target.closest(".hist-row");
  if (!tr || !tr.dataset.id) return;
  deps.openGame(tr.dataset.id); // reuse the analysis load-by-id flow
});

function updatePager(pageLen) {
  const count = $("histCount"), prev = $("histPrev"), next = $("histNext");
  const from = state.total === 0 ? 0 : state.offset + 1;
  const to = state.offset + pageLen;
  if (count) count.textContent = state.total ? `${from}–${to} of ${state.total}` : "";
  if (prev) prev.disabled = state.offset <= 0;
  if (next) next.disabled = state.offset + PAGE >= state.total;
}
