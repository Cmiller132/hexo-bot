/* checkpoints.js — the ONE checkpoint picker, shared by the play picker and
 * analysis selector (app.js) and the lab (learn/lab.js). Catalogue display
 * rules (grouping, the featured/"show all" filter, the Strongest/latest/PUCT
 * tags, params formatting) live here so the three pickers never drift.
 *
 * Catalogue flags (from bots.toml, passed through /api/bots):
 *   group      picker grouping ("" = the default/current-ladder group, first)
 *   featured   shown in the default view; the rest wait behind "show all"
 *   strongest  "Strongest" tag
 *   default    the pre-selected checkpoint
 *   search     "puct" → a legacy-search tag
 *   params     exact weight count, formatted for the meta line
 */

// 8128812 -> "8.13M params", 1656453 -> "1.66M params", <1M -> a "K" suffix.
function fmtParams(n) {
  const s = n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : Math.round(n / 1e3) + "K";
  return s + " params";
}

/* Normalize a raw /api/bots checkpoint list into the picker's shape. Structural
 * flags (group/search/featured/strongest/default) are lifted out; any other
 * scalar key plus the formatted params make up the display `meta` line. */
export function normalizeCheckpoints(raw) {
  const list = raw && Array.isArray(raw.checkpoints) ? raw.checkpoints : [];
  const reserved = ["id", "label", "run", "epoch", "group", "search",
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
      group: typeof c.group === "string" ? c.group : "",
      search: typeof c.search === "string" ? c.search : "",
      featured: !!c.featured,
      strongest: !!c.strongest,
      isDefault: !!c.default,
      meta: parts.length ? parts.join(" · ") : (c.run ? String(c.run) : ""),
    };
  });
}

/* Group by the `group` key: the ungrouped ("") group is the default ladder and
 * comes first; named groups follow in first-appearance order; catalogue order
 * is preserved within each group. */
export function groupCheckpoints(checkpoints) {
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

/* The "latest" tag: the last entry of the FIRST group (newest ladder rung). */
export function latestCheckpoint(groups) {
  const items = groups.length ? groups[0].items : [];
  return items[items.length - 1] || null;
}

/* The default pick: the `default` checkpoint, else the first `strongest`, else
 * the newest ladder rung. */
export function defaultCheckpoint(checkpoints) {
  return (checkpoints.find(c => c.isDefault)
    || checkpoints.find(c => c.strongest)
    || latestCheckpoint(groupCheckpoints(checkpoints)) || null);
}

/* Render the grouped checkpoint list into `listEl`. When `showAll` is off, only
 * featured checkpoints (plus the current selection, so it is never hidden) are
 * listed. Buttons carry `.bot` + `data-ckpt`; the caller wires clicks. */
export function buildCkptList(listEl, checkpoints, { selectedId = null, showAll = false } = {}) {
  listEl.textContent = "";
  const groups = groupCheckpoints(checkpoints);
  const latest = latestCheckpoint(groups);
  for (const g of groups) {
    const items = g.items.filter(c => showAll || c.featured || c.id === selectedId);
    if (!items.length) continue;
    if (g.name) {
      const h = document.createElement("div");
      h.className = "bot-group";
      h.textContent = g.name;
      listEl.appendChild(h);
    }
    for (const c of items) {
      const b = document.createElement("button");
      b.className = "bot" + (c.id === selectedId ? " sel" : "");
      b.dataset.ckpt = c.id;
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", c.id === selectedId);
      const tags = [];
      if (c.strongest) tags.push('<span class="tag strong">Strongest</span>');
      if (latest && c.id === latest.id) tags.push('<span class="tag">latest</span>');
      if (c.search === "puct") tags.push('<span class="tag puct">PUCT search</span>');
      const meta = [c.meta, ...tags].filter(Boolean).join(" · ");
      b.innerHTML = `<span class="bot-row"><span class="bot-name"></span>` +
        `<span class="bot-meta">${meta}</span></span>`;
      b.querySelector(".bot-name").textContent = c.label;
      listEl.appendChild(b);
    }
  }
}
