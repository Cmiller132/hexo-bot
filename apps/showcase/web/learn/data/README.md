# Learn-section snapshot data

Static JSON baked from the real `shrimp_main_7` run by
`apps/showcase/scripts/learn_snapshots.py` (env + command in its header).
Regenerate with that script; do not hand-edit. All files are minified JSON,
ASCII-only, no filesystem paths.

Shared conventions

- Coordinates are axial `[q, r]` integers. `owner` / `to_move` are `0`
  (player 0, moves first) or `1`.
- A position's `moves` is the chronological placement list; owners follow the
  fixed turn structure (P0 places 1 opening stone, then each side places 2 per
  turn). Replaying `moves` reproduces the board exactly.
- Every value/stv number is side-to-move POV in `[-1, 1]`; `moves_left` is
  expected remaining placements.

## Position block (shared by attention.json and checkpoints.json)

Each entry of `positions[]` carries:

| field | type | meaning |
|---|---|---|
| `id` | str | stable slug: `quiet_midgame`, `four_threat`, `double_threat`, `late_game` |
| `title`, `description` | str | display copy (why the position is interesting) |
| `moves` | `[[q,r],...]` | chronological placements (see conventions) |
| `stones` | `[[q,r,owner],...]` | final board, chronological |
| `to_move` | 0/1 | side to move |
| `phase` | str | `Opening` / `FirstStone` / `SecondStone` (mid-turn) |
| `first_stone` | `[q,r]` or null | the pending turn's first stone when `phase == "SecondStone"` |
| `threats.opp_hot` etc. | `[[q,r],...]` | empty cells of live 4+-stone windows, relative to `to_move` (`opp_*` = threat against the mover; `*_win` = win-in-1 cells) |

## attention.json (~216 KB)

Real attention rows from the `ep58` net.

| field | type | meaning |
|---|---|---|
| `run`, `checkpoint`, `generated` | str | provenance stamps (`ep58`, ISO date) |
| `num_tokens` | int | 8 learned summary tokens prefixed to the cell sequence |
| `blocks`, `heads` | int | 5 attention blocks (depth order), 3 heads each |
| `floor` | float | cell weights below this (1e-3) were pruned |
| `positions[]` | | position block plus the fields below |
| `positions[].support.coords` | `[[q,r],...]` | node order used by all cell indices; segments `[legal \| stones \| halo]` |
| `positions[].support.legal_count/stone_count/halo_count` | int | segment sizes (row `i` is legal iff `i < legal_count`) |
| `positions[].queries[]` | | the ~6 query cells: `{role, cell:[q,r], node}` where `node` indexes `support.coords`. Roles: `last_stone`, `opening_stone`, `top_policy`, `threat_cell`, `far_legal`, `halo`, `policy_candidate` (backfill) |
| `positions[].attention[b][h][qi]` | | attention row of query `queries[qi]` in block `b`, head `h` |
| `...attention[b][h][qi].tokens` | `[8 floats]` | weight on each summary token (always all 8, 4 dp) |
| `...attention[b][h][qi].cells` | `{"<node>": w}` | sparse cell weights >= `floor`, keys index `support.coords`, 4 dp |

Each full row (tokens + all cells, pre-pruning) is a softmax and sums to 1
(asserted at generation); the stored sparse row sums to ~1 minus pruned mass.

## checkpoints.json (~12 KB)

The same four positions read by four checkpoints of the same run.

| field | type | meaning |
|---|---|---|
| `checkpoints[]` | | `{id: "ep2", epoch, label}` for ep2/ep14/ep30/ep58 |
| `stv_head` | str | which short-term head `stv2` is (`stvalue_2` = value 2 plies ahead) |
| `policy_floor` | float | sparse-policy floor (1e-3) |
| `positions[]` | | position block plus `legal_count` and `per_checkpoint` |
| `positions[].per_checkpoint.epN` | | net-only readout: |
| `.value`, `.stv2` | float | binned-value expectations, mover POV, `[-1,1]` |
| `.moves_left` | float | expected remaining placements |
| `.entropy_nats` | float | policy entropy over the full legal softmax |
| `.top1_p` | float | top move's probability |
| `.policy` | `[{q,r,p},...]` | legal-cell softmax, descending, floored at `policy_floor` (the dense distribution sums to 1; only the tail is trimmed) |
| `sharpening.rows[]` | | per position: `{position, legal_count, max_entropy_nats, entropy: {epN: H}, top1_p: {epN: p}}` — the policy-sharpening summary |

## eval_history.json (~39 KB)

Parsed from the run's real multistage-eval diagnostics.

| field | type | meaning |
|---|---|---|
| `run`, `generated`, `anchor` | str | run label, ISO date, rating zero-point (`sealbot`) |
| `notes` | obj | free text — rating scale, candidate-label caveat, pentanomial definition |
| `evaluated_epochs` | `[int]` | epochs with eval data (5, 10, then every 5 up to `latest_epoch`; 15 has none) |
| `latest_epoch` | int | last evaluated epoch present |
| `bt_fit` | obj | latest pooled fit stats: `n_edges`, `n_players`, `converged`, `iterations` |
| `epochs[]` | | one entry per evaluated epoch: |
| `.epoch`, `.candidate`, `.champion` | | e.g. `5`, `"cand_ep5"`, `"ep4"` |
| `.verdict` | str | `PROMOTE` / `INCONCLUSIVE` / ... (descriptive; run is pure-eval) |
| `.primary` | obj | candidate-vs-champion pooled BT difference — `elo_diff`, `elo_diff_ci95`, `se_elo`, `hypothesis` |
| `.candidate_elo`, `.candidate_elo_ci95`, `.candidate_se_elo` | | the candidate's pooled rating at that epoch |
| `.games_budget`, `.eval_visits` | int | sample-size context |
| `.edges[]` | | this epoch's matches: `{opponent, role, kind, primary, paired, games_requested, decided, winrate, winrate_ci95, elo_point}` plus for paired checkpoint edges `{n_pairs, pentanomial, pair_winrate, pair_se, eval_visits}` |
| `.ratings[]` | | full pooled rating table after this epoch: `{label, elo, elo_ci95, se_elo, is_anchor}` |

Plot the compounding strength curve from the LATEST epoch's `ratings` (labels
`ep5`, `ep10`, ...); the per-epoch `candidate_elo` values are fresh-label
single-epoch estimates with wide CIs (see `notes.candidates`).
