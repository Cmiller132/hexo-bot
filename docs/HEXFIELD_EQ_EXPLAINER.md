# hexfield_eq: how the model works

This document explains the **hexfield_eq** bot (`packages/hexfield_eq`) for
other Hexo bot developers. It assumes you know the game (two stones per turn,
six-in-a-row on an unbounded hex grid, forced center opening) and the shape of
the AlphaZero loop (self-play → train the net to imitate search → stronger
search → repeat). Everything else is explained from scratch.

Every number and mechanism below describes the **production configuration** —
the single build that self-plays, trains, and evaluates — as one fixed system.

| | |
|---|---|
| Trunk width | 192 channels = 12 fiber slots × 16 orbit channels |
| Attention heads | 3 × 64 (structural — one head per win axis) |
| Trunk layout | `C C A C C A C A` — 5 conv blocks, 3 attention blocks |
| Input | 46 feature planes + 12 ray lengths per cell |
| Summary tokens | 6 |
| Value output | distribution over 65 bins on [−1, 1] |
| Support radius | 4 |
| Learnable parameters | 627,343 |

Blocks like this mark mechanisms worth building an interactive visualization
around:

> **Demo idea —** *rendered like this.*

---

## 1. System overview

The package is half Python, half Rust. Rust owns the tree search, the
production featurizer, and the training-data expander; Python owns the network,
the trainer, and the evaluation stack. One epoch of the loop:

1. **Self-play** — 256 concurrent games; searched moves get 256 visits of
   Gumbel AlphaZero search (§7); all leaf positions across all games are
   batched to the GPU through an asynchronous serve pipeline (§8).
2. **Record** — every searched decision writes one training row: the raw facts
   of the position plus the search's improved policy and value targets (§9.1).
3. **Train** — a sliding window over recent games is expanded back into tensors
   and the network takes AdamW steps against nine loss heads (§9.4).
4. **Evaluate** — every 5 epochs the new checkpoint plays a fixed opponent
   roster under a paired, pentanomially-scored protocol (§10). Evaluation
   observes; it never gates or promotes.

---

## 2. The support set

The network does not see a padded grid. It sees a **support set**: exactly the
cells that matter for the current position, rebuilt per position.

- **Stones** — every occupied cell.
- **Legal cells** — every empty cell within hex-distance 4 of a stone (the
  support radius; the rules themselves allow distance 8).
- **Halo** — the one-cell shell just outside, so a convolution centered on an
  edge cell still has neighbours to read.

One multi-source BFS outward from the stones produces all three layers plus
each cell's distance-to-nearest-stone in a single pass. Early positions have a
few dozen nodes, crowded ones a few hundred.

Nodes are stored in a fixed order — **legal cells, then stones, then halo**,
each segment sorted by packed coordinate id. Legal-first gives the
**legal-prefix property**: the policy head emits one logit per node in order,
so the first `legal_count` outputs *are* the legal moves. Downstream consumers
never match coordinates against a rules oracle — taking the first `legal_count`
logits is the mask, a slice that cannot drift out of sync with the node order.

To the network a position is five aligned arrays: `feats (N × 46)`,
`coords (N × 2)` axial coordinates, `nbr (N × 6)` neighbour row-indices along
the six hex directions (missing → a sentinel that reads as a zero row),
`raylen (N × 12)` ray lengths (§3.3), and a boolean node mask (used for batch
padding).

---

## 3. Input features

The featurizer is implemented twice — Rust for production, Python as the
reference oracle — kept identical by exact-parity tests. Stone and threat
planes are **side-to-move relative** ("own" = the player about to move); one
colour plane carries the absolute side.

### 3.1 Base scalar planes (0–10)

| # | Plane | Meaning |
|---|---|---|
| 0/1 | own-stone / opp-stone | occupancy one-hot |
| 2 | empty | 1 − stones |
| 3 | legal | 1 on the legal prefix |
| 4 | phase-is-second-stone | this move is the turn's second placement |
| 5 | first-stone | where this turn's first stone landed |
| 6 | player-colour | 1 everywhere when player 0 is to move |
| 7/8 | own/opp recency | `1/(1+age)` of each stone's placement |
| 9 | distance-to-stone | BFS distance / 8 (halo reads 0.625) |
| 10 | opp-last-turn | the opponent's most recent full turn (1–2 cells) |

### 3.2 Graded window planes (11–42)

A **window** is one of the 6 possible length-6 alignments through a cell along
one win axis (Q, R, or QR). A window is **clean for a side** if it contains no
enemy stones — it could still become that side's six-in-a-row. For every
support cell and every axis, the featurizer scans all 6 windows and emits:

- `own_line`, `opp_line` — the *maximum* stone count among that side's clean
  windows (/5): how advanced the best line through this cell is.
- `own_live`, `opp_live` — the *count* of clean windows (/6): how open this
  cell is on this axis.
- `own_live3/4/5`, `opp_live3/4/5` — the count of clean windows already holding
  ≥3, ≥4, ≥5 side stones (/6). Graded threat multiplicity: `live5` is per-cell
  standing-win multiplicity, `live4` counts distinct win-in-two threats.

10 quantities × 3 axes = **30 axis-indexed planes**, laid out with each
quantity's three axis values contiguous (the layout the symmetry machinery of
§4.3 acts on). Two **fork scalars** (41/42) follow: the number of axes (of 3)
whose raw line count reaches 3, per side.

Windows that run off the support count as clean-and-empty: an absent cell
contributes no stones. Edge cells therefore read slightly more open than they
are. The convention is player-symmetric and identical in both featurizers.

### 3.3 Global scalars (43–45) and ray lengths

Three board-level scalars broadcast to every node: **ply** (placements made,
clamped at 96, /96), **distance-to-centroid** (the node's distance from the
stone centroid, normalized by twice the spread, capped at 1), and **spread**
(the stones' maximum distance from their centroid, clamped at 16, /16). All
three are symmetry-invariant.

Separately, every node carries **12 ray lengths** (`raylen`): for each
side × axis × direction, how far a straight walk from the cell stays
*attendable*. The walk rules: a cell off the support stops the walk; an enemy
stone is included (the blocker itself is visible) and then stops it; own
stones and empty cells pass through. Values run 0–5 — the reach of a length-6
window. Ray lengths feed the convolutions' ray taps (§5.2), not the input
planes.

> **Demo idea —** *hover a cell: draw its 18 windows (6 per axis), colored
> clean-for-own / clean-for-opp / dead, with the line/live/liveK numbers
> updating as stones are placed. A second layer draws the 12 rays with their
> blocker-truncated lengths.*

---

## 4. The symmetry constraint

The defining property of the network: it is **exactly equivariant** under all
12 symmetries of the hexagonal lattice.

### 4.1 The 12 symmetries

The hex lattice about the origin has the dihedral symmetry group **D6**: six
rotations by multiples of 60° and six reflections. In axial coordinates the
generators are

```
rot60:   (q, r) → (−r, q + r)
reflect: (q, r) → (q, −q − r)
```

Two induced actions matter to the architecture:

- the three **win axes** {Q, R, QR} permute among themselves — a 60° rotation
  3-cycles them, a reflection transposes two and fixes one;
- the six **directions** around a cell permute — rotations advance the
  6-cycle, reflections reverse it.

### 4.2 The guarantee

For every symmetry `g` and every position:

```
f(g·board) = g·f(board)
```

The policy over the transformed board is exactly the transformed policy; the
value is exactly unchanged. Consequences: the training pipeline applies no
symmetry augmentation (the property holds by construction, not statistically);
symmetric positions evaluate identically up to floating-point rounding; and no
stored parameter is spent representing a rotated copy of a pattern another
parameter already represents.

The rest of this section is the constraint each layer type satisfies to make
that equation hold.

### 4.3 Typed input

Under a symmetry, the 16 scalar planes (0–10, 41–45) stay in place — their
values move with the cells but the planes don't trade roles. The 30 axis
planes do trade roles: rotating the board 60° maps the "own_line along Q"
plane onto "own_line along R". Each quantity's {Q, R, QR} triple carries the
3-element axis permutation. The input representation is therefore
16 fixed planes + 10 permuting triples.

### 4.4 The fiber: 12 slots × 16 channels

Each cell's 192-channel trunk vector is organized as **12 fiber slots of 16
orbit channels** — one slot per group element. A symmetry `g` acts on the
trunk's feature field in two coupled ways:

1. cells move to their transformed coordinates, and
2. inside every cell's vector, the 12 slots **permute** (slot `h`'s 16 numbers
   move to slot `g·h`).

This is the *regular representation* of D6. Because the action is a pure
permutation — no sign flips, no mixing — every weight constraint below reduces
to an index-gather. The 16 orbit channels are the free learned capacity; the
12 slots are its symmetry-indexed copies.

> **Demo idea —** *one cell's 192-vector drawn as a 12×16 grid of colored
> cells. Applying rot60 visibly cycles the 12 rows; a reflection produces a
> different row shuffle. This is the single hardest idea in the model and the
> one a picture teaches fastest.*

### 4.5 Weight tying

**Convolutions.** The 7-tap hex conv (center + 6 direction taps, each tap its
own 192×192 matrix — direction-typed, because line direction is the content of
the game) must satisfy, for every `g` and tap `t`:

```
W[g·t] = M(g) · W[t] · M(g)⁻¹
```

where `g·t` is the permuted tap and `M(g)` the slot permutation. Solving this
leaves **84 free 16×16 blocks** per conv — the module stores exactly those
(as `w_base`) and materializes the dense `(7, 192, 192)` weight each forward
through a precomputed index-gather. The GPU kernels consume the dense weight
and are unaware of the tie. The conv bias is slot-constant: one 16-vector
broadcast over the 12 slots.

**1×1 linears** (attention projections, MLPs, token reads) are the center-tap
special case: **12 free 16×16 blocks** each, materialized the same way.

**Norms.** LayerNorm's mean/variance over all 192 channels is already
permutation-invariant; the affine is tied — one (γ, β) pair per orbit channel,
broadcast over the slots. LayerScale's per-channel scale is tied identically.

Under inference (no gradients) the materialized dense weights are cached and
regenerate only when the underlying parameters change, so the tie costs
essentially nothing at serve time.

> **Demo idea —** *a dense 7-tap conv weight drawn as 7 tiles of 12×12 blocks.
> Toggle the constraint: blocks snap into tied copies, a counter drops from
> 1008 free blocks to 84, and hovering one free block highlights its 11 tied
> copies across the tiles.*

### 4.6 Heads = win axes

The attention reshape splits 192 channels into `heads × head_dim`. Because the
slots permute under symmetry, a head's channels land in *another head's*
channels — consistent only if the head partition follows the group structure.
The partition used is the three cosets of the order-4 subgroup that preserves
the Q-axis: **3 heads of dimension 64, one per win axis**, permuted by the
symmetries exactly as the axes are. The head count is structural — the build
rejects any other value.

### 4.7 The relative-position bias

Attention adds a learned bias per query/key pair, indexed by their coordinate
offset through a fixed row map: **217 rows** for every exact offset within
distance 8, **8 + 8 ring rows** for distances 9–16 split into on-win-axis and
off-axis, **1 far row**, and **3 token rows** (token↔cell, cell↔token,
token↔token) — 237 rows, 3 head columns.

The symmetry constraint ties this table **jointly across (offset-row, head)
pairs**: a symmetry moves the offset to a new row *and* the head to a new head
simultaneously, so the free parameters are the orbits of that joint action —
**81 free values per attention block**, gathered into the full 237×3 table
each forward. Head diversity survives exactly where geometry permits: an
offset on no reflection axis keeps 3 independent head values, a win-axis
offset keeps 2 (its own head free, the other two shared), and the ring, far,
and token rows keep 1 (all heads equal).

> **Demo idea —** *the 237×3 table drawn as a grid colored by tied class.
> Click any entry: every entry constrained equal to it lights up, sweeping
> across both rows and columns. Pair it with the offset disk so clicking an
> offset highlights its orbit on the board.*

### 4.8 Tokens, reads, and the stem

- **Summary tokens** (6 global scratchpad vectors riding in the attention
  sequence) carry the invariant type: a learned `6 × 16` parameter broadcast
  identically across all 12 slots, so symmetries fix them.
- **Output reads.** A policy logit must be an invariant scalar attached to a
  moving cell; the value must be fully invariant. Both use **group-pooling**:
  average the 12 slots, leaving an invariant vector of one value per orbit
  channel. To avoid a 16-wide bottleneck, reads are widened first by a tied
  expansion — ×4 before pooling for the scalar heads (64-wide pooled blocks),
  ×2 for the per-cell heads (32-wide).
- **The stem** maps the typed 46-plane input into the regular fiber. Scalar
  planes may only write slot-constant weight columns; each axis triple lifts
  along the coset structure. Implemented as a free weight passed through a
  *Reynolds projection* — averaging over all 12 symmetry-conjugates — which
  projects onto the constraint-satisfying subspace.

---

## 5. The network, block by block

`HexfieldNet`: stem conv + norm → **C C A C C A C A** → final norm → heads.
All blocks run at width 192 over the variable-length node set; the node mask
is re-applied after every parameter-carrying op so batch padding never leaks
into live nodes.

### 5.1 Stem

A tied 7-tap conv from 46 planes to 192 channels (§4.8), then group-norm and
ReLU. Output: the `(N, 192)` trunk stream, structured as 12 slots × 16.

### 5.2 Convolution blocks — ray-tap convs

Each C block is a residual block of two convolutions with group-norm after
each, ReLU between and after, and a LayerScale (init 1e-4) on the residual
branch.

Each convolution is a **ray-tap conv**. Its center tap reads the cell itself,
as in a plain hex conv. Each of its 6 direction taps reads a
**visibility-masked, distance-weighted sum of up to 5 cells** down that
direction:

```
tap_d(i)[c] = Σ_{k=1..5}  α[k, c] · [k ≤ reach_side(c)(i, d)] · x(i + k·d)[c]
```

- **α** is a `(5 × 16)` parameter per conv — per-distance, per-orbit-channel —
  shared across all 6 directions (the symmetry constraint of §4.5 requires
  the sharing) and broadcast over the 12 slots.
- **reach** is the ray length of §3.3 for the matching side, axis, and
  direction: the walk sees up to and including an enemy blocker, and never
  past the support edge.
- **Sides ride the channel split**: within each 16-channel orbit, the first 8
  channels use own-side visibility and the last 8 opp-side. (The split lives
  on the orbit index because symmetry fixes orbit channels while permuting
  slots — a slot split would violate §4.4.)

The 6 aggregates plus the center tap then enter the same 7-tap GEMM as a plain
hex conv. Net effect: every conv layer has a 5-cell, blocker-aware receptive
field along all three win axes in both directions, for ~80 extra parameters
per conv. The stem and the head convs (§5.5) are plain 7-tap convs — only the
10 trunk convs carry ray taps.

> **Demo idea —** *pick a cell and a direction; slide stones onto the ray and
> watch the visible window shrink at an enemy stone (which stays lit) and end
> at the support edge. Show α as a per-distance bar chart weighting the
> visible cells, split own/opp by channel half.*

### 5.3 Attention blocks

Pre-norm transformer blocks over the joint sequence **[6 tokens ; N cells]**:
3 heads × 64 (§4.6), the jointly-tied relative-position bias (§4.7) added to
every score, additive masking of padded keys, then a GELU MLP at 2× width;
LayerScale on both branches. One attention block gives every cell a view of
the entire board and of the tokens in a single hop, with the bias supplying
exact geometry to distance 8 and coarser geometry beyond.

### 5.4 Register refresh — the counting channel

Softmax attention computes weighted *averages*, which cannot represent "how
many": three simultaneous live-4 threats and one look alike through a softmax.
At the **exit of every C block**, a register refresh updates the 6 tokens from
the cells with **sigmoid gates and an unnormalized sum**:

```
tokens += out_proj( (Σ_cells sigmoid(q·kᵀ/√64 + gate_bias) · v) · sum_scale )
```

Each token learns a pattern query `q` against cell keys; the sigmoid-gated sum
literally counts matching cells. `gate_bias` is a learned per-token threshold
(init −2.5, so a token's background gate starts near 0.08 rather than
integrating the whole board); `sum_scale` (init 1/32, learned) keeps updates
O(1)–O(10) at typical match counts; `out_proj` starts near zero so the lane is
a numeric no-op at initialization while its gradients stay live. The token
stream is carried in fp32 on the half-precision serve path so late-block count
increments survive rounding.

Because the final LayerNorm normalizes magnitudes away, the scalar heads also
read the **pre-final-norm token mean** as one extra input block — the raw
count magnitudes reach the value estimate directly.

> **Demo idea —** *place N copies of a threat pattern and plot one token's
> summed update against N: a straight line, next to a softmax-attention
> baseline that stays flat. Then show the gate: cells whose sigmoid exceeds
> ~0.5 light up as "counted".*

### 5.5 Heads

Per-cell heads (each: plain 7-tap conv → ×2 tied expansion → group-pool →
linear on the 32-wide invariant read):

| Head | Output | Used at |
|---|---|---|
| policy | 1 logit per node; the legal prefix is the move distribution | search |
| opp_policy | predicted reply distribution of the opponent's next decision | training |
| soft_policy | policy at temperature 2 (flattened target) | training |
| cell_q | a 65-bin value distribution per legal cell | training |

Scalar heads (each reads 64-wide group-pooled ×4 blocks: its tokens + the
masked mean of all cells + the pre-norm token mean):

| Head | Reads | Output |
|---|---|---|
| value | all 6 tokens + 2 pooled blocks (512-wide input) | 65 bins on [−1, 1], decoded as softmax expectation |
| stvalue 2/6/16 | tokens 2–3 | value at 3 short horizons, 65 bins each |
| moves_left | tokens 4–5 | decisions remaining, 65 bins mapped to [0, 209] |

Inference computes policy + value (+ moves_left when the search requests it);
the other heads exist to shape the trunk during training.

### 5.6 Parameters

| Component | Params |
|---|---|
| 5 conv blocks (tied 7-tap ×2, ray-tap α, norms) | 216,400 |
| Heads (reductions + 65-bin tops) | 212,233 |
| 3 attention blocks (tied QKVO + MLP) | 74,352 |
| Register lane (5 refreshes) | 62,115 |
| Stem (46 → 192 typed lift) | 61,840 |
| Bias tables (3 × 81 joint classes) | 243 |
| Tokens + trunk norms | 160 |
| **Total** | **627,343** |

An untied trunk of the same width would store ~4.3 M trunk parameters alone.
The ×12 tie yields a ~7× smaller model that computes like a full 192-channel
network — the dense weights exist at forward time, generated from the free
blocks.

---

## 6. One position, end to end

A concrete trace of a real position from the run's own self-play history: the
ply-40 decision of a recorded game (40 stones down, player 0 placing the
second stone of their turn; the game ran 119 decisions in total). All numbers
below come from running that position through the production featurizer. Each
stage is one frame of the pipeline. The position itself — records, support
layers, and shapes — is exported to
`docs/explainer_assets/walkthrough_position.json` so a renderer can draw it
directly.

**1 · Facts.** The position is 40 records of `(q, r, owner, placement_index)`
plus phase and first-stone. This is all that is ever stored or transmitted —
everything below is derived.

**2 · Support.** BFS from the 40 stones: **N = 402 nodes** = 288 legal + 40
stones + 74 halo, in [legal | stones | halo] order. `coords (402, 2)`,
`nbr (402, 6)`.

**3 · Features.** The featurizer scans 402 cells × 3 axes × 6 windows =
**7,236 windows** and walks 402 × 12 rays, producing `feats (402, 46)` (about
one third of the entries are non-zero) and `raylen (402, 12)`.

**4 · Batching.** During search this position rides in a group with other
leaf positions, padded to a shared node count — here **Npad = 448** (quantized
to 64) — giving batch tensors `(B, 448, 46)`, `(B, 448, 6)`, `(B, 448, 2)`,
`(B, 448, 12)` and a `(B, 448)` mask.

**5 · Stem.** `(402, 46) → (402, 192)`: each cell now carries a 12-slot × 16
fiber.

**6 · Trunk.** The stream passes C C A C C A C A:

- each **C** block: two ray-tap convs (each gathering 7 taps per cell — the
  center plus 6 five-cell ray aggregates) with residual; then the register
  refresh updates the 6 tokens from all 402 cells;
- each **A** block: the sequence `[6 tokens ; 402 cells]` = **408 rows**
  self-attends with a `(3, 408, 408)` bias gathered from that block's 237-row
  table; tokens and cells exchange information globally.

Shapes never change: cells stay `(402, 192)`, tokens `(6, 192)`.

**7 · Heads.** After the final norm: the policy conv+read produces `(402,)`
logits, of which the **first 288 are the legal moves** — softmaxed, they are
the search prior. The value read (all 6 tokens + pooled cells + pre-norm token
mean → 512-wide) produces 65 bin logits, decoded to a scalar in [−1, 1] by
softmax expectation.

**8 · Search.** The prior and value enter the tree at this leaf. At the root,
priors seed the 16 Gumbel candidates; values back up the path that led here.
After 256 visits the search emits the improved policy π′ as the training
target and samples the move to play (§7).

> **Demo idea —** *this section as a scroll-through storyboard: each stage one
> animated panel over the same 402-node position, with live tensor-shape
> captions. The support build, the window scan, the fiber, one conv gather,
> one attention hop, and the legal-prefix slice of the policy.*

---

## 7. The search

The search is **Gumbel AlphaZero** (Danihelka et al., 2022), implemented in
Rust; Python is only called to evaluate leaf batches. Self-play searches 256
visits per decision.

### 7.1 Tree structure

A tree edge is a single stone, not a turn. Each node stores its side-to-move;
value backup flips sign by comparing node player to leaf player, so the two
consecutive same-player plies of a Hexo turn back up correctly without any
turn-level special-casing. The engine checks for six-in-a-row after every
stone, so a win on the first stone of a turn is a terminal node whose second
stone never exists. Terminal nodes back up exact ±1.

### 7.2 Root selection

One Gumbel noise value `g(a)` is drawn per legal root action; the top
**m = 16** actions by `g(a) + logits(a)` become candidates. The tactical-safety
scan (§7.4) force-includes any threat-participating cells that missed the cut,
so the set can exceed 16. **Sequential Halving** then spends the visit budget:
candidates get equal visits per round, the worse half is eliminated each
round (4 rounds at m=16), ranked by

```
g(a) + logits(a) + σ(completedQ(a)),      σ(q) = (50 + max_b N(b)) · q
```

`completedQ` is the child's mean value where visited, and a visit-weighted
blend of the node value and the visited children's prior-weighted average
where not.

### 7.3 Interior selection

Below the root, selection is deterministic: visit the action maximizing

```
π′(a) − N(a) / (1 + Σ_b N(b))
```

where `π′ = softmax(logits + σ(completedQ))` over the node's candidates — the
action whose visit share lags its improved-policy share the most. Leaves are
selected up to 96 at a time per game under a virtual-loss of 1.0, evaluated as
a batch, then backed up.

No noise is injected anywhere in the search — no Dirichlet noise, no forced
playouts. Exploration comes from the root Gumbel draw, from play-time
temperature, and from openings: 35% of games begin with a few plies (up to 8)
sampled directly from the raw prior at temperature 1.4 without search; those
plies produce no training rows.

### 7.4 Choosing the move

The recorded **training target** is π′ at the root — the improved policy — not
the visit histogram (under Sequential Halving, visit counts are a schedule
artifact). The raw root logits are recorded alongside it.

The **played move** is sampled from the root visit distribution under a
temperature schedule `T(ply) = max(0.15, 0.5^(ply/20))` (root prior
temperature 1.05), with candidates eliminated in the first halving round
removed from the sampling distribution. Two guards run before selection: an
immediate own win is played unconditionally, and moves that ignore an
opponent's standing threat are zeroed (both from the engine's threat scan).
At temperature 0 — evaluation games — selection is a greedy
lower-confidence-bound pick on Q (`q − 1.6·σ/√n`) with a decisiveness
tie-break from the moves-left head.

### 7.5 The continuous scheduler

256 games run concurrently against one GPU. Games drop pending leaves into a
shared queue; at ~1024 pending (or when nothing can progress) the batch
flushes to the evaluator. Duplicate positions are evaluated once per batch and
cached across the run (262,144-state cache). Finished games are replaced
immediately. Games cap at 256 plies; there is no resignation.

> **Demo idea —** *Sequential Halving animated: 16 candidate bars, per-round
> visit quotas filling, half the bars greying out each round, the σ-adjusted
> ranking re-sorting live. End by contrasting the final visit histogram with
> π′ — the two distributions differ, and the recorded one is π′.*

---

## 8. Serving the network

The serve pipeline is what lets one GPU feed 256 concurrent searches.

**The wire.** Rust featurizes leaves in parallel and ships one payload per
flush: feature rows for all positions concatenated (f16, sorted
largest-position-first), coordinates, neighbour tables, ray lengths, and
per-row legal counts. Python replies with values, priors (softmaxed over each
row's legal prefix), raw logits, and moves-left when requested.

**Asynchronous evaluation.** Submission and result-draining are separate
calls: the engine submits a payload (GPU work enqueues, no sync) and continues
selecting the next batch of leaves while the forward runs; it drains the
result with a single device sync later.

**Padding groups.** Rows are packed into groups sharing one padded node count,
quantized to multiples of 64, bounded by a padding-waste cap (18%) and a
`B·Npad²` ceiling — bounded wasted math in exchange for few, fixed-shape
launches.

**Fixed shapes, cached weights.** The forward runs in fp16 (the value and
moves-left tops in fp32) with whole-group forwards replayed as pre-captured
GPU graphs keyed on the bucketed shape. The tied model costs nothing extra
here: materialized dense weights are cached against the parameters' version,
and the head-coset channel permutations of §4.6 are folded into the cached
projection weights, so inference performs zero runtime permutes.

Sustained throughput on the production GPU is ~21 evaluated positions/second
at full self-play settings.

> **Demo idea —** *the batching pipeline: differently-sized positions stream
> in, sort by size, pack into fixed-height groups (Npad bars) with visible
> padding waste, and fire to the GPU as the search continues selecting on the
> left — the two-phase overlap is the point.*

---

## 9. Training

### 9.1 Training rows

Every searched decision stores one row: the placement history (the position is
reconstructed from it — feature planes are never stored), plus targets:

- **π′** (the policy target) and the visit distribution (fallback/diagnostic);
- the raw root logits and per-child Q values;
- the **game outcome z** (±1 from the row player's perspective);
- **short-term values**: exponentially-weighted averages of future root values
  at horizons 2, 6, 16 decisions;
- **moves-left**: decisions remaining;
- the **next opponent decision's policy** (for the opp_policy head);
- **policy surprise**: `KL(visit ‖ prior)` — how much the search disagreed
  with the raw net.

One shard per game, written atomically (data file first, sidecar as commit
marker). Games that hit the 256-ply cap without a winner still train the
policy heads; their outcome-dependent targets (value, short-term value,
cell_q, moves-left) are masked.

### 9.2 The replay window

Each epoch trains on up to 56,000 rows drawn from a tapered window over recent
games: the window grows sublinearly with total rows generated (power-law
taper, exponent 0.65, minimum 7,000 rows), takes newest shards first, and
down-samples uniformly toward a 210,000-row keep target. Ordering derives from
game keys, not file timestamps, so a resumed run selects identically. A
**reuse governor** persisted in the checkpoint credits 8 training-row-units
per newly generated self-play row and throttles training when the buffer goes
stale — the trainer cannot over-fit a lagging buffer.

### 9.3 Expansion

Rows are expanded back into tensors in Rust: rebuild the support from the
placement history, recompute all 46 planes and 12 ray lengths, and project the
policy targets onto the legal prefix. No symmetry transform is applied (§4.2).

### 9.4 The losses

Nine heads, summed with fixed weights:

| Head | Target | Loss | Weight |
|---|---|---|---|
| policy | π′ over the legal prefix | soft CE, surprise-weighted | 1.0 |
| value | outcome z | CE vs two-hot 65-bin target | 1.0 |
| soft_policy | π′ at temperature 2 | soft CE | 0.5 |
| opp_policy | next opponent π′ | soft CE | 0.25 |
| stvalue 2, 6, 16 | future-value EMA | 65-bin CE, masked | 0.1 each |
| moves_left | decisions remaining | 65-bin CE, masked | 0.2 |
| cell_q | per-child search Q | 65-bin CE per legal cell | 0.1 |

- **Surprise weighting**: row `i`'s policy CE weight is
  `0.5 + 0.5·n·sᵢ/Σs` (clamped at 8), `s` the stored surprise KL — gradient
  concentrates on positions the net mispredicts.
- **Structural legality**: the policy CE is computed over each row's legal
  prefix directly; target mass off the prefix is a hard error, not a masked
  zero.
- Scalar targets become two-hot distributions over the 65 bins (mass split
  between the two adjacent bin centers), and predictions decode as softmax
  expectations.

### 9.5 Optimization

AdamW at 5e-4, cosine-decayed to 1e-4 over 60 epochs with a 400-step linear
warmup; weight decay 1e-4 on matrix weights only. Forward/backward in fp16
autocast with fp32 loss arithmetic. Gradient clipping is adaptive: 1.75× an
EMA of the pre-clip gradient norm (a static 5.0 during the EMA's warmup) —
the tied trunk concentrates gradient through its 84-block gathers, so its
global norm runs hotter than an untied net's and a fixed clip would
miscalibrate. Batches of 256 rows are split into micro-buckets under a
`B·Npad²` budget with step-global loss denominators, so the gradient is
independent of the split (up to floating-point reassociation).

### 9.6 Initialization

The run warm-starts from a 2,400-step behavioral-cloning prefit: the same
architecture briefly trained to imitate a fixed game corpus, loaded weights-only
(fresh optimizer). This skips the slowest opening-competence epochs of
self-play.

---

## 10. Evaluation

Strength is measured directly, every 5 epochs, from a 192-game budget per
evaluation. Games run at 512 visits, greedy after an 8-ply temperature-1.0
opening.

### 10.1 Paired games, pentanomial scoring

Net-vs-net games run in **pairs**: both games share one sampled opening line
(the leader searches its opening, the follower replays it) and swap seats, so
a pair differs only by who holds which side. Hexo has no draws, so a completed
pair scores {0, 1, 2} candidate wins; a pair with one truncated game scores
{0.5, 1.5}. The **pair is the unit of statistics** — a five-bucket
("pentanomial") histogram whose pair-level variance absorbs the correlation
that shared openings induce, keeping confidence intervals honest.

### 10.2 The roster

| Opponent | What it is | Games | Protocol | Role |
|---|---|---|---|---|
| **Strix** | third-party GNN bot (HeXONet) under its own Gumbel MCTS, 512 sims, fixed checkpoint | 64 (32 pairs) | paired | the pinned 0-Elo anchor |
| **SealBot** | external C++ minimax, 50 ms/decision | 32 | unpaired, binomial | cross-lineage calibrator |
| **Self-bracket** | own checkpoints at grid epochs {5, 10, 20, 40, 80, 160}, nearest 2 | ~96 shared | paired | progress curve |
| **Champion** | own most recent checkpoint ≥5 epochs back | with brackets | paired | per-epoch verdict target |

SealBot is unpaired because its time-limited depth varies with machine load —
a seat-swapped pair would not be a matched comparison — and it enters the
rating pool at half weight for the same reason.

### 10.3 Reading the numbers

All results feed a rolling Bradley–Terry pool anchored at Strix = 0 Elo. The
candidate-vs-champion verdict is single-epoch by construction (~±130 Elo
standard error) — a gross-regression tripwire that never tightens. The
fixed-anchor curves (Strix, SealBot, brackets — the same label every epoch)
pool information across epochs and are the actual progress meter.

Evaluation cannot gate, promote, or halt training (asserted in the eval path).
The one feedback loop is the **head audit**: every epoch the moves-left head
is scored against recent games (rank-correlation and near-end error gates);
on failure, the search lever that consumes it is disabled until the head
recovers.

---

## 11. Numbers cheat-sheet

Geometry & features — legality radius 8 · support radius 4 · halo = radius+1 ·
46 planes = 11 scalars + 30 axis + 2 fork + 3 global · window length 6 ·
norms: line /5, live /6, fork /3, ply /96, spread /16 · raylen 12 slots,
reach 5.

Model — width 192 · 12 slots × 16 orbit channels · 3 heads × 64 · trunk
`CCACCACA` (5C + 3A) · 6 tokens · MLP ×2 · bias 237 rows → 81 free values per
A block · 84 free blocks per conv, 12 per 1×1 · value/aux bins 65 · moves-left
cap 209 · LayerScale init 1e-4 · reads widened ×4 (scalar) / ×2 (per-cell) ·
ray-tap α (5×16) per trunk conv · register: gate bias −2.5, sum scale 1/32 ·
parameters 627,343.

Search — 256 visits (512 in eval) · Gumbel m 16, c_visit 50, c_scale 1.0 ·
σ(q) = (50 + max visits)·q · temperature 0.5^(ply/20), floor 0.15 · root prior
T 1.05 · LCB z 1.6 · prior-sampled openings: 35% of games, ≤8 plies, T 1.4 ·
256 concurrent games · 96 leaves/game/pass, virtual loss 1.0 · flush ~1024 ·
eval cache 262,144 · game cap 256 plies.

Training — 256-row batches · ≤56k rows/epoch · window: min 7k, keep 210k,
taper exponent 0.65 · reuse credit 8/row · AdamW 5e-4 → cosine 1e-4 over 60
epochs, warmup 400 steps · weight decay 1e-4 · adaptive clip 1.75× EMA ·
surprise weight 50% uniform, cap 8× · loss weights 1.0 / 1.0 / 0.5 / 0.25 /
0.1×3 / 0.2 / 0.1 · warm start: 2,400-step BC prefit.

Evaluation — every 5 epochs · 192 games = 64 Strix + 32 SealBot + 96 self ·
512 visits · opening 8 plies at T 1.0 · Strix weight 1.0 (anchor), SealBot 0.5 ·
champion lag 5 epochs · no gating.

---

## 12. Where things live

- **Geometry & symmetry** — `python/hexfield_eq/geometry.py` (D6, bias-row
  indexing), `equivariant.py` (group tables, weight-generation gathers, head
  permutations, joint bias classes, group-pool).
- **Board & features** — `support.py`, `features.py` (Python oracle); Rust
  production twins in `rust/src/features.rs` / `support.rs`; plane indices and
  shape constants in `constants.py`.
- **Model** — `model.py` (`HexfieldNet`, `HexNodeConv`, `EquivLinear`,
  attention + bias builders, heads), `register.py` (register refresh),
  `_raytap.py` (ray-tap aggregation).
- **Search** — `rust/src/search.rs` (drivers, scheduler, targets),
  `rust/src/tree.rs` (selection, Sequential Halving, completed-Q); the serve
  wire in `rust/src/payload.rs`.
- **Serving** — `inference.py` (grouping, async evaluation, decode),
  `batching.py`.
- **Training** — `selfplay.py`, `samples.py` / `shards.py` /
  `buffer_manifest.py` (rows and storage), `window.py`, `trainer.py`,
  `losses.py`, `prefit.py`, `config.py`.
- **Evaluation** — `eval_driver.py` (paired-match engine), `eval_arena.py`,
  `eval_stats.py` (pentanomial/Bradley–Terry math), `multistage_eval.py`
  (roster, budget, pool), `head_audit.py`, `evaluation.py`.
- **Deep references** — `docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md` (the
  full mathematics behind §4), `SPEC_RAYTAP_CONV.md` (features and ray-tap,
  exact definitions).
- **The production run** — `configs/hexfield_eq_main_2.toml` +
  `scripts/prefit_env/hexfield_eq_raytap_a5.env`.
