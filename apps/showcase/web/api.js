/* api.js — thin fetch layer over the showcase server.
 *
 * The server contract is mid-flight (catalogue x sims selection, public game
 * feed, per-ply summary are landing in parallel), so everything here
 * feature-detects: GETs retry 5xx/network failures with backoff, and the bots
 * payload is normalized from either the old flat-ladder shape or the new
 * {checkpoints, sims} shape.
 */

export class ApiError extends Error {
  constructor(status, message, network = false) {
    super(message);
    this.status = status;
    this.network = network;
  }
}

const netListeners = new Set();
let netDown = false;

export const onNetChange = cb => { netListeners.add(cb); };

function setNet(down) {
  if (down === netDown) return;
  netDown = down;
  for (const cb of netListeners) cb(down);
}

async function request(path, { method = "GET", body, retries = method === "GET" ? 2 : 0 } = {}) {
  for (let attempt = 0; ; attempt++) {
    let resp;
    try {
      resp = await fetch(path, {
        method,
        headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        credentials: "same-origin",
      });
    } catch (err) {
      if (attempt < retries) {
        await new Promise(res => setTimeout(res, 500 * 2 ** attempt));
        continue;
      }
      setNet(true);
      throw new ApiError(0, "network error", true);
    }
    setNet(false);
    if (resp.status >= 500 && attempt < retries) {
      await new Promise(res => setTimeout(res, 500 * 2 ** attempt));
      continue;
    }
    let data = null;
    try { data = await resp.json(); } catch (_) { /* non-JSON error body */ }
    if (!resp.ok) {
      const detail = data && (data.detail || data.message);
      throw new ApiError(resp.status, typeof detail === "string" ? detail : `HTTP ${resp.status}`);
    }
    return data;
  }
}

// ---- bots: normalize either ladder shape ------------------------------------
//
// New shape:  {checkpoints: [{id, label, ...}], sims: [16, 64, ...]}
//             POST /api/game {checkpoint_id, sims}
// Old shape:  [{id, label, visits, run, epoch}, ...]  (one entry per rung)
//             POST /api/game {bot_id}; we group rungs by (run, epoch) into
//             checkpoints and map (checkpoint, sims) back to the closest rung.

export async function getBots() {
  const raw = await request("/api/bots");
  if (raw && !Array.isArray(raw) && Array.isArray(raw.checkpoints)) {
    const checkpoints = raw.checkpoints.map(c => {
      // extra scalar keys beyond the fixed ones are display metadata
      // (e.g. games = "3.4M games"); fall back to the run name
      const extras = Object.entries(c)
        .filter(([k, v]) => !["id", "label", "run", "epoch"].includes(k) &&
                            (typeof v === "string" || typeof v === "number"))
        .map(([, v]) => String(v));
      return {
        id: String(c.id ?? c.checkpoint_id),
        label: String(c.label ?? c.id ?? c.checkpoint_id),
        meta: extras.length ? extras.join(" · ") : (c.run ? String(c.run) : ""),
      };
    });
    const sims = (raw.sims || []).map(Number);
    return {
      checkpoints,
      sims,
      payloadFor: (ckptId, simCount) => ({ checkpoint_id: ckptId, sims: simCount }),
    };
  }
  const entries = Array.isArray(raw) ? raw : [];
  const groups = new Map(); // "run@epoch" -> {label, meta, rungs: Map(visits -> bot id)}
  for (const e of entries) {
    const gk = `${e.run ?? ""}@${e.epoch ?? e.id}`;
    if (!groups.has(gk)) {
      groups.set(gk, {
        id: gk,
        label: e.epoch !== undefined ? `ep ${e.epoch}` : String(e.label ?? e.id),
        meta: e.run ? String(e.run) : "",
        rungs: new Map(),
      });
    }
    groups.get(gk).rungs.set(Number(e.visits), String(e.id));
  }
  const checkpoints = [...groups.values()];
  const sims = [...new Set(entries.map(e => Number(e.visits)))].sort((a, b) => a - b);
  return {
    checkpoints,
    sims,
    payloadFor: (ckptId, simCount) => {
      const group = groups.get(ckptId);
      let botId = group && group.rungs.get(simCount);
      if (!botId && group) {
        // no exact rung for this sims count: take the closest one
        const best = [...group.rungs.keys()]
          .sort((a, b) => Math.abs(a - simCount) - Math.abs(b - simCount))[0];
        botId = group.rungs.get(best);
      }
      // superset payload: old servers read bot_id, new ones checkpoint_id+sims
      return { bot_id: botId, checkpoint_id: ckptId, sims: simCount };
    },
  };
}

// ---- games ------------------------------------------------------------------

export const createGame = payload =>
  request("/api/game", { method: "POST", body: { human_color: 0, ...payload } });

export const getGame = id => request(`/api/game/${id}`);

export const postMove = (id, q, r) =>
  request(`/api/game/${id}/move`, { method: "POST", body: { q, r } });

export const resign = id => request(`/api/game/${id}/resign`, { method: "POST" });

export const setNickname = (id, nickname) =>
  request(`/api/game/${id}/nickname`, { method: "POST", body: { nickname } });

export const getGamesFeed = () => request("/api/games");

export const getAnalysis = (id, ply) =>
  request(`/api/game/${id}/analysis?ply=${ply}`);

export const getSummary = id => request(`/api/game/${id}/summary`);
