# Hexfield showcase family-coupling review

Scope: `apps/showcase/server/showcase/*.py`, `apps/showcase/web/**`, and
`apps/showcase/scripts/**`. Status describes behavior when the selected bot is
`family = "hexfield_eq"`. This inventory was written before the implementation
changes, as requested; entries describe the fix made by this change (or the
reason a coupling remains).

## Server

| Status | Coupling point | Finding and disposition |
| --- | --- | --- |
| BROKEN | `server/showcase/families/hexfield_eq_family.py`: `net_eval`, `summary_eval`, `_forward_rows` | STV was returned as `None`. Hexfield trains horizons 2/6/16, but `forward_policy_value(..., request_moves_left=True)` only returns policy/value/moves-left in the current model implementation. Analysis now deliberately uses the full-head `forward(...)`, decodes every `stvalue_<h>` with `decode_binned_value`, and preserves the serve-only forward for device self-check/gameplay. The per-position Analysis compatibility field remains the shortest horizon scalar while Lab exposes the complete horizon map. |
| WORKS | `server/showcase/analysis.py` | This module is the shrimp analysis implementation and intentionally retains its imports, shapes, shortest-horizon Analysis value, batching, and search wire decoder. Worker dispatch calls family hooks for net, searched, and summary evaluation, so hexfield analysis does not import or route through this shrimp module. |
| BROKEN | `server/showcase/lab.py` | The complete Lab implementation was shrimp-specific (module-level shrimp imports, radius-4 support, 15 planes, shrimp heads, hooks, and shrimp action decoding). It remains the shrimp implementation to avoid changing shrimp behavior, while worker dispatch moves behind new `ModelFamily.lab_eval_payload` and `lab_search_payload` hooks. The shared replay/search result mechanics accept a family decoder. |
| BROKEN | `server/showcase/bots.py`: `_WorkerRuntime.lab_eval` / `lab_search` | These called `lab.py` directly for every family. They now delegate position building, eval payload creation, search, and action decoding to the selected bot's family adapter. |
| BROKEN | `server/showcase/families/base.py` | The protocol had analysis hooks but no Lab boundary. It now declares Lab eval/search hooks for both registered families. |
| WORKS | `server/showcase/families/shrimp_family.py` | Existing gameplay/analysis dispatch was correct. New Lab hooks are thin calls into the unchanged shrimp Lab implementation, retaining its payload and numerical behavior. Heavy imports remain method-local. |
| BROKEN | `server/showcase/families/hexfield_eq_family.py`: Lab | No hexfield Lab path existed. The adapter now builds real sequence/free-edit `PositionFacts`, checkpoint-metadata support geometry, checkpoint-version feature planes and ray lengths; returns correct value, all STVs, moves-left, legal-prefix policies, support, and named planes; and uses the family search/session/profile/action decoder. Free edit zeroes the three history-derived planes. Architecture-specific attention and activation payloads explicitly return `{available:false, reason:...}` instead of shrimp-shaped data. |
| WORKS | `server/showcase/app.py`: Analysis/Summary/Lab HTTP routes | Routes select a catalogue checkpoint and submit generic worker jobs. `/api/bots` already publishes `family`. No model import occurs in the web process. Analysis cache version is bumped because the hexfield STV schema/value changes. Branding in the FastAPI title is product copy, not inference coupling. |
| WORKS | `server/showcase/lab_rules.py` | Request validation uses only engine legality and family-neutral coordinate/count bounds. Free-edit restrictions are valid for both featurizers. |
| WORKS | `server/showcase/matchapi.py` | Confirmed family-agnostic: it drives `GameSession` and the already-family-dispatched pool. It never imports a model package or decodes model actions. |
| WORKS | `server/showcase/game.py`, `db.py`, `elo.py`, `jsonsafe.py` | Engine/session, persistence, rating, and JSON sanitation have no neural-architecture assumptions. |
| WORKS | `server/showcase/config.py` | The default search config is shrimp-named for backward-compatible deployments, while each checkpoint may supply `search_profile`; the family adapter parses it. No hexfield checkpoint is forced through shrimp when configured with its profile. |
| WORKS | `server/showcase/device.py` | Family self-check dispatch is already adapter-based. Shrimp batching/warmup helpers are legacy defaults used only when no family is supplied or by `ShrimpFamily`; `HexfieldEqFamily.warmup` is a no-op and never calls them. Hexfield self-check remains on `forward_policy_value`, so analysis full-head work cannot accidentally change gameplay parity. Method-local imports keep the web process torch-free. |
| WORKS | `server/showcase/__init__.py` | The default `SHRIMP_SUPPORT_RADIUS=4` is a shrimp import-time compatibility setting. It does not select a family or configure `HEXFIELD_EQ_*`; hexfield architecture metadata is applied by its adapter before loading. Product/module prose remains shrimp-branded but does not alter behavior. |

## Web frontend

| Status | Coupling point | Finding and disposition |
| --- | --- | --- |
| WORKS | `web/app.js`: Analysis tab | Analysis and summary consume generic value/STV/moves-left/policy shapes and checkpoint ids. The shortest-horizon scalar contract remains valid for both families, and searched coordinates arrive already decoded. The “open in Lab” link now carries the active analysis checkpoint so a hexfield analysis cannot silently reopen under the default shrimp net (or vice versa). |
| DEGRADES | `web/checkpoints.js` | `/api/bots` supplied `family`, but normalization did not retain it as a field; hexfield run labels also kept the raw `hexfield_eq` prefix. Normalization now preserves `family`, and model labels cleanly strip either known family prefix. Picker selection behavior is unchanged. |
| BROKEN | `web/learn/lab.js`, `lab_features.js`, `lab.html` | The live Lab always rendered the client-side shrimp radius-4 support and 15 planes, assumed all three auxiliary policy heads, assumed a 65-bin distribution, and dereferenced shrimp attention/activation payloads. It now selects behavior from the chosen checkpoint family. Shrimp keeps the existing client featurizer. Hexfield requests real server feature planes/support, renders their returned names/counts, hides unsupported policy-head controls, and handles machine-readable unavailable markers for attention/activations without throwing. Static copy is made family-neutral/dynamic where it described shrimp-only dimensions. `lab_features.js` remains the shrimp oracle, used only for shrimp and engine editing. |
| WORKS | `web/learn/eq.js` and the non-Lab learn pages/data | These are the static hexfield_eq architecture explainer and already describe the 46-plane equivariant model. They are not selected by playable checkpoint and do not feed `/api/game` or `/api/lab`; no runtime family dispatch is appropriate. |
| WORKS | `web/api.js`, `board.js`, `stats.js`, `hexo_match_client.py`, `bot-api.md` | Transport, board geometry, stats, and external-match clients are family-neutral. |
| DEGRADES | `web/index.html`, `style.css`, top-level titles/copy | Several labels call the product/bot "shrimp". This is branding rather than architecture or payload coupling; changing it would alter the established showcase identity and is deferred. It does not misroute a hexfield checkpoint. |

## Scripts

| Status | Coupling point | Finding and disposition |
| --- | --- | --- |
| DEGRADES | `scripts/learn_snapshots.py` | The script is intentionally a shrimp_main_7 snapshot baker: it imports shrimp, assumes 15 features/8 tokens/shrimp hooks, and emits static legacy Lab/explainer fixtures. Per task scope it is not part of the playable-bot path and is deferred rather than generalized. Its module documentation already identifies the fixed source run. No server or live Lab endpoint imports it. |

## Explicitly deferred

- Faithful hexfield attention matrices and per-block activation norms. Hexfield
  has coordinate/pair and ray attention plus an equivariant/ray-tap trunk, but
  these are not shape-compatible with shrimp's visualization. The correct
  interim contract is an explicit unavailable marker; a bespoke visualization
  can be added later without presenting misleading tensors now.
- Static shrimp snapshot generation and shrimp product branding, for the scope
  reasons above.
