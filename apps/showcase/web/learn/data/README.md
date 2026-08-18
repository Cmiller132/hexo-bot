# Learn-section data

Static JSON consumed by the learn pages and the lab.

Shared conventions: coordinates are axial `[q, r]` integers; `owner` /
`to_move` are `0` (player 0, moves first) or `1`; a position's `moves` list is
the chronological placement history under the fixed turn structure (1 opening
stone, then 2 per turn) — replaying it reproduces the board exactly.

## MantisNet demo data (the "how it works" pages)

Both files come from one real 64-sim showcase game against the served
`mantis-cellnodes1-it402` checkpoint. Opening noise is disabled site-wide
(temperature 0), so replaying the recorded moves against the same checkpoint
reproduces the game — and this data — exactly.

### mantis_walkthrough.json (~66 KB)

The ply-11 position (11 stones, player 0 to place the first stone of a turn)
through the exact vendored builder and served weights. Generated **inside the
production container** by `apps/showcase/scripts/mantis_learn_walkthrough.py`
(see its docstring); regenerate with that script, do not hand-edit.

| field | meaning |
|---|---|
| `params`, `config`, `klent` | read from the loaded checkpoint: 5,196,965 parameters, the architecture knobs, and the as-trained τ/λ/mass-floor |
| `moves`, `stones`, `to_move`, `moves_remaining` | the position |
| `windows` | all 154 live windows: axis, start, 6 cells, mover-relative digits, canonical pattern class |
| `cells` | all 420 legal cells in engine order: covered flag, nearest-stone distance, and the real head outputs — `prior`, `pi_prime`, `q_value`, `q_score` |
| `v_hat`, `state_head_untrained` | the served value v̂ and, for contrast, the untrained state-value head's readout |
| `focus` | the exhibit cell (−2, 6): its 192-class radius edges (orbit/own/axis per in-range stone) and its 18 action rows (post-placement classes, EMPTY flags) |

The in-page JS transcription of the pattern/orbit vocabularies (`../mantis.js`)
is validated against this file at page load; a drift disables the affected
figures.

### mantis_search.json (~52 KB)

The live-search SSE frames of two searches of the same game, exactly as the
live viewer received them: `overturn` (ply 11 — the prior's favourite is cut
and the eventual game-winning cell is chosen from a 7% prior) and `win_race`
(ply 15 — the last two survivors are both winning-line placements). Plus the
full move list, the result, the winning line, and the exact halving schedules
of the site's four budgets. Captured with a scripted client (create a game
with `?watch_search=1`, keep the flag on every move, record
`/api/game/{id}/search-stream`).

## Lab data (the shrimp sandbox, reached from the analysis view)

Baked from the real `shrimp_main_7` run by the retired
`apps/showcase/scripts/learn_snapshots.py` flow; see the git history of this
README for the full schema notes.

- `checkpoints.json` (~12 KB) — checkpoint catalogue metadata (`ep2`/`ep14`/
  `ep30`/`ep70`) + the four preset positions with per-checkpoint net readouts.
  The lab reads its preset-position dropdown from this file.
- `features.json` (~11 KB) — the four sample positions with per-plane digests
  of shrimp's 15-plane featurizer; the lab's client featurizer
  (`../lab_features.js`) asserts these digests.
