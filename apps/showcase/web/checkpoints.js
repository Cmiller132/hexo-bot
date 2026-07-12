/* checkpoints.js — the ONE checkpoint picker, shared by the play picker and
 * analysis selector (app.js) and the lab (learn/lab.js). The catalogue is
 * grouped into MODELS (by `run`): each model is a selectable row, and the
 * selected model reveals a native <select> of every epoch/checkpoint it holds.
 * The effective selection is always a single checkpoint id (e.g. "main7-ep70")
 * — that is what every consumer passes to the backend. Grouping, ordering, the
 * per-model default pick, params formatting and the Strongest tag live here so
 * the three pickers never drift.
 *
 * Catalogue flags (from bots.toml, passed through /api/bots):
 *   run        the model a checkpoint belongs to (e.g. "shrimp_main_7")
 *   epoch      the training epoch (dropdown label + default tiebreak)
 *   strongest  "Strongest" tag; also the model default when no `default`
 *   featured   default-pick fallback under `default`/`strongest`
 *   default    the pre-selected checkpoint for its model
 *   params     exact weight count, formatted for the meta line
 */

// 8128812 -> "8.13M params", 1656453 -> "1.66M params", <1M -> a "K" suffix.
function fmtParams(n) {
  const s = n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : Math.round(n / 1e3) + "K";
  return s + " params";
}

/* Normalize a raw /api/bots checkpoint list into the picker's shape. Structural
 * flags (run/epoch/search/featured/strongest/default/params) are lifted out; any
 * other scalar key plus the formatted params make up the display `meta` line. */
export function normalizeCheckpoints(raw) {
  const list = raw && Array.isArray(raw.checkpoints) ? raw.checkpoints : [];
  const reserved = ["id", "label", "run", "epoch", "group", "family", "search",
                    "params", "featured", "strongest", "default"];
  return list.map(c => {
    const extras = Object.entries(c)
      .filter(([k, v]) => !reserved.includes(k) &&
                          (typeof v === "string" || typeof v === "number"))
      .map(([, v]) => String(v));
    const parts = [];
    if (typeof c.params === "number" && c.params > 0) parts.push(fmtParams(c.params));
    parts.push(...extras);
    return {
      id: String(c.id ?? c.checkpoint_id),
      label: String(c.label ?? c.id ?? c.checkpoint_id),
      run: typeof c.run === "string" ? c.run : "",
      family: typeof c.family === "string" ? c.family : "shrimp",
      epoch: c.epoch,
      params: typeof c.params === "number" ? c.params : null,
      search: typeof c.search === "string" ? c.search : "",
      featured: !!c.featured,
      strongest: !!c.strongest,
      isDefault: !!c.default,
      meta: parts.length ? parts.join(" · ") : (c.run ? String(c.run) : ""),
    };
  });
}

/* Trailing integer of a run key ("shrimp_main_7" -> 7), or -Infinity when there
 * is none, so runs sort strongest-first (highest number first) and unlabeled
 * runs fall to the end. */
function runOrdinal(run) {
  const m = /(\d+)\s*$/.exec(run || "");
  return m ? parseInt(m[1], 10) : -Infinity;
}

/* A clean model label from a run key: "shrimp_main_7" -> "main 7", stripping a
 * leading "shrimp" and collapsing underscores to spaces. Falls back to the raw
 * run (or "model") when the shape is unfamiliar. */
export function modelLabel(run) {
  if (!run) return "model";
  const cleaned = String(run)
    .replace(/^(?:shrimp|hexfield[_-]?eq)[_\s-]*/i, "")
    .replace(/[_-]+/g, " ")
    .trim();
  return cleaned || String(run);
}

/* Numeric-descending compare on epoch, non-numeric last. */
function epochValue(c) {
  const n = Number(c.epoch);
  return Number.isFinite(n) ? n : -Infinity;
}

/* The model default checkpoint: `default`, else `strongest`, else `featured`,
 * else the highest epoch (last item as a final tiebreak). */
function modelDefault(items) {
  return items.find(c => c.isDefault)
    || items.find(c => c.strongest)
    || items.find(c => c.featured)
    || items.slice().sort((a, b) => epochValue(b) - epochValue(a))[0]
    || items[items.length - 1]
    || null;
}

/* Group the flat catalogue into models keyed by `run`, ordered strongest-first
 * (main7, main6, main5, main4). Each model carries its checkpoints (epoch-
 * descending), the per-model default checkpoint, a display label, a params count
 * and whether it holds the strongest checkpoint. */
export function groupByModel(checkpoints) {
  const order = [], byRun = new Map();
  for (const c of checkpoints) {
    const r = c.run || "";
    if (!byRun.has(r)) { byRun.set(r, []); order.push(r); }
    byRun.get(r).push(c);
  }
  const models = order.map(run => {
    const items = byRun.get(run).slice().sort((a, b) => epochValue(b) - epochValue(a));
    const def = modelDefault(items);
    const params = items.find(c => c.params)?.params ?? null;
    return {
      run,
      label: modelLabel(run),
      items,
      params,
      defaultId: def ? def.id : (items[0] ? items[0].id : null),
      strongest: items.some(c => c.strongest),
    };
  });
  models.sort((a, b) => runOrdinal(b.run) - runOrdinal(a.run));
  return models;
}

/* The model a given checkpoint id belongs to, and that checkpoint. */
export function findModelOf(models, id) {
  for (const m of models) {
    const c = m.items.find(x => x.id === id);
    if (c) return { model: m, checkpoint: c };
  }
  return { model: null, checkpoint: null };
}

/* The "newest" checkpoint overall: the strongest model's default checkpoint.
 * (Used by the analysis fallback when a retired game's own net has left the
 * catalogue.) */
export function latestCheckpoint(checkpoints) {
  if (!Array.isArray(checkpoints) || !checkpoints.length) return null;
  const models = groupByModel(checkpoints);
  if (!models.length) return null;
  const top = models[0];
  return top.items.find(c => c.id === top.defaultId) || top.items[0] || null;
}

/* The global default pick: an explicit `default` checkpoint if any, else the
 * strongest model's default checkpoint. */
export function defaultCheckpoint(checkpoints) {
  if (!Array.isArray(checkpoints) || !checkpoints.length) return null;
  return checkpoints.find(c => c.isDefault) || latestCheckpoint(checkpoints);
}

/* Render the model-grouped picker into `listEl`. Model rows are radio-like
 * buttons; the selected model additionally shows a native <select> of every one
 * of its epochs (a static label when it has a single checkpoint). All events are
 * wired internally: selecting a model resolves to that model's default, choosing
 * an epoch resolves to that checkpoint, and `onSelect(id, checkpoint)` fires with
 * the effective checkpoint id.
 *
 *   buildModelPicker(listEl, checkpoints, { selectedId, onSelect })
 */
export function buildModelPicker(listEl, checkpoints, { selectedId = null, onSelect = null } = {}) {
  listEl.textContent = "";
  const models = groupByModel(checkpoints);
  const sel = findModelOf(models, selectedId);
  const selModel = sel.model || (models[0] || null);

  const fire = (id) => {
    if (typeof onSelect !== "function") return;
    const found = findModelOf(models, id);
    onSelect(id, found.checkpoint);
  };

  for (const m of models) {
    const on = m === selModel;
    const row = document.createElement("div");
    row.className = "model" + (on ? " sel" : "");

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "model-head";
    btn.dataset.run = m.run;
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", String(on));
    const tags = [];
    if (m.strongest) tags.push('<span class="tag strong">Strongest</span>');
    const metaBits = [];
    if (m.params) metaBits.push(fmtParams(m.params));
    const meta = [...metaBits, ...tags].join(" · ");
    btn.innerHTML = `<span class="bot-row"><span class="bot-name"></span>` +
      `<span class="bot-meta">${meta}</span></span>`;
    btn.querySelector(".bot-name").textContent = m.label;
    btn.addEventListener("click", () => {
      if (m === selModel) return; // already selected — keep its epoch pick
      buildModelPicker(listEl, checkpoints, { selectedId: m.defaultId, onSelect });
      fire(m.defaultId);
    });
    row.appendChild(btn);

    if (on) {
      // effective checkpoint within this model (the selection, else default)
      const curId = (sel.checkpoint && sel.model === m) ? sel.checkpoint.id : m.defaultId;
      row.appendChild(buildEpochControl(m, curId, listEl, checkpoints, onSelect, fire));
    }
    listEl.appendChild(row);
  }
}

/* The per-model epoch control: a native <select> when the model has more than
 * one checkpoint, a static label when it has exactly one. The Strongest tag
 * rides along on the recommended epoch. */
function buildEpochControl(model, curId, listEl, checkpoints, onSelect, fire) {
  const wrap = document.createElement("div");
  wrap.className = "model-epoch";

  if (model.items.length <= 1) {
    const only = model.items[0];
    const span = document.createElement("span");
    span.className = "epoch-static";
    const bits = [only ? only.label : ""];
    if (only && only.strongest) bits.push("Strongest");
    span.textContent = bits.filter(Boolean).join(" · ");
    wrap.appendChild(span);
    return wrap;
  }

  const select = document.createElement("select");
  select.className = "epoch-select";
  select.setAttribute("aria-label", model.label + " checkpoint");
  for (const c of model.items) {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.label + (c.strongest ? " · Strongest" : "");
    if (c.id === curId) opt.selected = true;
    select.appendChild(opt);
  }
  select.addEventListener("change", () => {
    // re-render so the row reflects the new epoch, then notify
    buildModelPicker(listEl, checkpoints, { selectedId: select.value, onSelect });
    fire(select.value);
  });
  wrap.appendChild(select);
  return wrap;
}
