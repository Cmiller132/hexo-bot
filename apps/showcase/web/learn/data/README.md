# Learn-section data

Static JSON consumed by the learn pages and the lab. Two generations coexist.

Shared conventions: coordinates are axial `[q, r]` integers; `owner` /
`to_move` are `0` (player 0, moves first) or `1`; a position's `moves` /
`records` list is the chronological placement history under the fixed turn
structure (1 opening stone, then 2 per turn) — replaying it reproduces the
board exactly.

## hexfield_eq demo data (the "how it works" pages)

Exported from the hexfield_eq package itself by
`scripts/export_eq_learn_data.py` (run from the repo root with
`scripts/prefit_env/hexfield_eq_raytap_a5.env` sourced). Regenerate with that
script; do not hand-edit.

### eq_group_tables.json (~10 KB)

The D6 group action the symmetry demos render.

| field | meaning |
|---|---|
| `group_order` | 12 |
| `coord_mats[g]` | 2×2 integer matrix: element `g`'s action on axial `(q, r)` (columns = images of the basis vectors); `g` 0–5 rotations, 6–11 reflections |
| `mult`, `inv` | 12×12 composition table and inverses |
| `slot_perms[g]` | the regular representation: under `g`, slot `h`'s channels move to slot `slot_perms[g][h]` |
| `tap_perms[g]` | length-7 conv-tap permutation (tap 0 = center, fixed; taps 1–6 = the six directions) |
| `axis_perms[g]` | where each win axis {Q, R, QR} lands |
| `cosets`, `coset_of_slot` | the 3 cosets of the order-4 Q-axis-preserving subgroup — the head partition |
| `bias.disk` | `[dq, dr, row]` for every exact offset (hex-dist ≤ 8, 217 rows) |
| `bias.joint_classes` | (237, 3) tied-class id per (row, head); `bias.free_values` = 81 |
| `conv_tie_classes` | (7, 12, 12) free-block class per (tap, slot-in, slot-out); `conv_free_blocks` = 84 |
| `linear_tie_classes` | (12, 12) 1×1 tie classes; `linear_free_blocks` = 12 |
| `constants` | production architecture numbers (channels 192 = 12×16, trunk CCACCACA, 46 features + 12 raylen, 65 bins, ...) |

### eq_walkthrough.json (~105 KB)

The explainer's §6 position (`docs/explainer_assets/walkthrough_position.json`,
a real ply-40 self-play decision) through the production featurizer.

| field | meaning |
|---|---|
| `records` | 40 × `[q, r, owner, placement_index]` — all that is ever stored |
| `current_player`, `phase`, `first_stone` | decision state (`SecondStone`) |
| `n_nodes`, `legal_count`, `stone_count`, `halo_count` | 402 = 288 + 40 + 74, node order `[legal | stones | halo]` |
| `coords`, `nbr`, `dist` | (402, 2) axial coords, (402, 6) neighbour rows (−1 absent), BFS distance |
| `feats` | (402, 46) feature planes, 4 dp |
| `raylen` | (402, 12) ray lengths, flat order `side*6 + axis*2 + dir` |
| `model_params` | 627343 — HexfieldNet under the production env |

The interactive featurizer in `../eq.js` (support BFS, window scan, ray walk)
is a transcription of the Python oracle (`hexfield_eq/features.py`,
`support.py`) and is validated against this file.

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
