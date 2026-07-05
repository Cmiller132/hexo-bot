# Shrimp: what it is and how it works

This is the plain-language explainer for Shrimp — the neural bot at the centre
of this repository. The deep design documents
([`shrimp_blueprint.md`](shrimp_blueprint.md) for the ideas and
[`specs/shrimp_model_spec.md`](specs/shrimp_model_spec.md) for the exact
contracts) go further, but this document is meant to be read straight through by
someone who has never seen the code. It stays truthful to the implementation;
where it simplifies, it simplifies without lying.

## What Shrimp is

Shrimp is an AlphaZero-style game-playing agent for Hexo, the Connect6-family
game played on an unbounded hexagonal grid. It has two parts that work together.
The first is a neural network that looks at a board position and outputs two
things: a *policy* (how good each legal move looks) and a *value* (who it thinks
is winning). The second is a Monte Carlo tree search that uses the network to
look ahead — it plays out promising lines in its head, guided by the network's
policy and value, and returns a move that is stronger than the raw network
opinion. The network is trained entirely by self-play: Shrimp plays games
against itself, the search produces better move choices than the bare network
would, and the network is then trained to imitate the search. Repeating that
loop is what makes it improve. There is no human game database in the loop
(though an optional warm start uses one to skip the slowest early learning).

## The board representation

Hexo has no fixed board. Stones are placed on an unbounded hexagonal grid
addressed by axial coordinates `(q, r)`, and a game could in principle wander
anywhere. A neural network needs a finite, well-defined set of cells to look at,
so Shrimp does not feed it a fixed-size board crop. Instead it builds a
**support set** that grows and shrinks with the game.

The support set is defined in three layers:

- **Stones** — every cell that currently holds a stone.
- **Legal cells** — every empty cell close enough to a stone to be a legal move.
  In Hexo, a move is legal if the cell is empty and within hex-distance
  `LEGAL_RADIUS` (8) of some stone; the very first move is forced to the origin.
- **Halo** — a one-cell border just outside that region, so the convolutions
  have somewhere to reach when they look at a legal cell on the edge.

The support is the union of those three. A single breadth-first search outward
from the stones produces all of it in one pass, along with each cell's distance
to the nearest stone. Because the support is exactly the cells that matter, an
early position is a few dozen cells and a crowded late position is a few hundred
— the network only ever pays for the board that exists.

The cells are stored in a fixed order: **legal cells first, then stones, then
halo**, each segment sorted by a canonical packed cell id. Ordering legal cells
first gives a useful guarantee called the *legal-prefix property*: the policy
head emits one number per cell in order, so the first `legal_count` outputs are
exactly the legal moves. The policy can only ever score legal moves — there is
no separate masking step that could drift out of sync with the rules.

`SHRIMP_SUPPORT_RADIUS` is a knob that shrinks the support. It restricts the
legal-cell layer to cells within that hex-distance of a stone (default is the
full `LEGAL_RADIUS` of 8; the shipped weights use 4). A smaller radius means a
smaller support and cheaper evaluation. It does not change the network's weight
shapes — the feature count is fixed — but it does change what the network sees,
so a checkpoint must be run at the radius it was trained with or its play
degrades. The featurizer is implemented twice, once in Rust (the fast production
path, `rust/src/features.rs` and `support.rs`) and once in Python (the reference
and CPU path, `features.py` and `support.py`); a parity harness keeps them
identical.

Each cell in the support carries **15 features**. They are, by index:

- `0` own-stone, `1` opponent-stone, `2` empty — one-hot occupancy from the
  perspective of the player to move.
- `3` legal — 1 on the legal-cell prefix, 0 elsewhere.
- `4` phase-is-second-stone — 1 everywhere when the player is placing their
  second stone of the turn.
- `5` first-stone — marks the cell where this turn's first stone landed (only
  meaningful during the second-stone phase).
- `6` player-colour — 1 everywhere when the player to move is player 0.
- `7` own-recency, `8` opponent-recency — how recently each stone was placed,
  weighted `1/(1+age)` so newer stones read hotter.
- `9` opponent-hot, `10` own-hot — cells belonging to a *threat*: an empty cell
  in a six-long single-colour line that already holds four or more stones (only
  once at least seven stones are on the board).
- `11` distance-to-nearest-stone — the BFS distance, scaled so stones read 0,
  legal cells fall in `(0, 1]`, and the halo reads `1.125`.
- `12` opponent-last-turn — the one or two cells the opponent placed on their
  most recent full turn.
- `13` opponent-win-now, `14` own-win-now — empty cells that complete a
  five-in-a-six line, i.e. a standing win-in-one for that side.

Features 0-12 follow the position's occupancy and history; 13-14 are the
standing-win planes. The hot and win planes come from scanning every six-cell
single-colour window on the three hex axes, which is exactly the threat logic
the game's tactics use.

## The network

The network is called `ShrimpNet` and lives in `model.py`. It is a graph-style
network over the support cells: a *stem* that lifts the 15 input features to the
trunk width, a *trunk* of alternating block types, and a set of output *heads*.

Two block types make up the trunk:

- **Convolution blocks (`C`)** — a hex convolution where each cell combines
  itself with its six hexagonal neighbours. The weights are *direction-typed*:
  the neighbour in the `q` direction uses different learned weights than the
  neighbour in the `r` direction, because line direction is the entire point of
  a connection game. This is local pattern recognition — information spreads one
  ring of neighbours per block. Each `C` in the layout is a residual block
  containing two such convolutions with LayerNorm.
- **Attention blocks (`A`)** — full all-pairs self-attention over the support
  cells plus 8 learned *summary tokens* (global scratchpads that every cell can
  read and write). A learned relative-position bias, keyed on the offset and
  direction between two cells, tells attention about board geometry. This is
  whole-board reasoning in a single step: a threat forming on the far side is
  visible immediately rather than after many convolution rings.

Three environment variables set the geometry, read once when the package is
imported:

- `SHRIMP_CHANNELS` — the trunk width (shipped: 192).
- `SHRIMP_ATTENTION_HEADS` — the number of attention heads (shipped: 3, which
  puts the per-head dimension at 192/3 = 64, the size the fast attention kernels
  want).
- `SHRIMP_TRUNK` — the block order as a string of `C` and `A` (shipped:
  `CCACCACCACCACCA`, i.e. two convs then an attention block, five times over).

These are load-bearing: a checkpoint only loads into a network built with the
same values, because they determine every weight shape. The launch, prefit, and
dashboard scripts set the shipped values for you.

After the trunk and a final LayerNorm, several heads read the result. The cell
vectors feed the policy heads; the summary tokens plus a masked mean of the cell
vectors feed the scalar heads.

- **Policy** — one logit per cell, over the legal prefix. This is the move
  distribution and the head that matters at play time.
- **Value** — a distribution over 65 bins spanning −1 to +1 (a distributional
  value, not a single scalar), representing who is winning.
- **Moves-left** — a distribution over how many moves remain until the game
  ends, used to prefer decisive finishes when already winning.
- **Short-term value** — value at a few short horizons (2, 6, 16 turns ahead).
- **Opponent-policy**, **soft-policy**, and a **per-cell Q** head — auxiliary
  training-only heads that sharpen learning. They are computed during training
  but the serve-time forward pass skips them.

At the shipped configuration the network has **8,128,812 parameters (~8.1M)**.

## The search

The network alone would play reasonably but shallowly. The search is what turns
a snap policy-and-value judgment into a considered move.

**What the tree search does, in plain words.** Starting from the current
position (the *root*), the search repeatedly walks down the tree of possible
moves, always favouring moves that look good (high policy) or have paid off in
earlier walks (high value), until it reaches a position it has not evaluated
yet. It asks the network for that new position's policy and value, adds it to
the tree, and propagates the value back up the path it took so parent nodes
learn from it. Each such walk is one *visit*. After a fixed budget of visits
(1024 in the shipped recipe), the move that got the most attention is the one
the search believes in.

**Gumbel-Top-m root selection and Sequential Halving.** Shrimp does not use
the classic AlphaZero root exploration (Dirichlet noise + forced playouts) —
those knobs were removed from the codebase entirely. It uses *Gumbel AlphaZero*
(Danihelka et al., 2022) instead. At the root it draws a small candidate set of
moves — `gumbel_m` of them (32 in the shipped recipe) — by adding Gumbel random
noise to the policy logits and taking the top `m`. It then runs **Sequential
Halving**: it splits the visit budget into rounds, giving every candidate an
equal share, then keeps only the best half and repeats, so visits concentrate on
the survivors. This has two benefits. It needs only a handful of candidates
rather than exploring every legal move, and it comes with a guarantee that the
move it finally plays is at least as good as the raw policy would have chosen —
a *policy improvement* — even when the visit budget is small. The resulting
improved distribution over root moves is exactly what the network is trained to
imitate (see below).

**The PUCT substrate.** Underneath Gumbel, the tree itself is a standard PUCT
tree (`c_puct`, completed-Q value backups) in `tree.rs`. Non-root node selection
uses the Gumbel completed-Q / σ formulation over that same machinery. Gumbel is
layered on top of PUCT; it does not replace it.

**Threat-space search (TSS).** Hexo is sudden-death: two independent threats
your opponent can't both block loses the game in a turn or two. To make sure the
search never overlooks a tactical shot, TSS force-includes every engine-legal
*tactical cell* (a cell that participates in a live threat) into the root
candidate set, even if its Gumbel score would have ranked it below the top `m`.
These forced cells are added to the budget rather than counted against it, and
each gets a guaranteed first visit. TSS is on by default (`tss_enabled = true`).

**The continuous batched scheduler.** Asking the network one position at a time
would waste the GPU. The production self-play driver is `run_continuous` in
`search.rs`: it keeps many games in flight at once (192 active games in the
shipped recipe), and every game that reaches a not-yet-evaluated leaf drops that
position into a shared queue. When the queue is large enough — or no game can
make further progress without an answer — the whole batch is sent to the network
in one call, and the results are distributed back to the games that were
waiting. Evaluations are deduplicated within a batch and cached by a position
hash, so identical positions across games are computed once. When a game
finishes, its slot is immediately filled by a new game. This is what keeps a
single consumer GPU busy while hundreds of searches run.

To spend visits where they matter, self-play also uses **Playout Cap
Randomization** (PCR): a fraction of moves (`pcr_full_proportion`, 0.33) get the
full 1024-visit search and become training targets, while the rest get a cheap
192-visit search just to keep the game moving. Full-search positions produce the
best targets; fast moves keep games flowing cheaply.

## The training loop

Training is the AlphaZero virtuous cycle, driven by the model-neutral
`hexo_train` orchestrator calling into the Shrimp plugin.

1. **Self-play** generates games. The continuous scheduler plays hundreds of
   games at once. For each full-search move it records the search's improved
   move distribution — the Gumbel target when `policy_target = "gumbel"` (the
   shipped setting), otherwise the raw visit counts — as the policy label for
   that position, and the eventual game result as the value label.
2. **Replay window.** Each finished game is written as a compact `.npz` shard
   plus a JSON sidecar under `<run>/selfplay/`. The trainer builds a
   modification-time-ordered, KataGo-style shuffle window over recent shards, so
   training draws from a moving window of recent games rather than only the
   latest. Only completed games are used — truncated games are dropped, so every
   value label reflects a real outcome rather than a guess.
3. **Training pass.** Each training row is expanded to dense tensors (in Rust,
   applying a random D6 hex-symmetry transform for augmentation) and the network
   takes AdamW steps. The loss is supervised: the policy head toward the
   search's move distribution, the value head toward the game outcome, plus the
   auxiliary heads (opponent-policy, short-term value, moves-left, per-cell Q,
   soft-policy) toward their respective targets.
4. **Checkpoint and eval.** A checkpoint is written each epoch. On a cadence
   (`eval_every`), the new checkpoint is measured against fixed opponents. Note
   that in this repo the eval verdict is **informational**: it reports strength
   but does not automatically gate, promote, or roll back the checkpoint. (A
   separate audit can disable the moves-left head if it stops correlating with
   reality.)

**Optional warm start.** Training from a random network works but is slow to
leave the opening. You can instead warm-start from a *behavioral-cloning* prefit:
`scripts/prefit_launch.sh` downloads a public human-games corpus
([`timmyburn/hexo-bootstrap-corpus`](https://huggingface.co/datasets/timmyburn/hexo-bootstrap-corpus)),
replays it through the engine into training shards, and trains a network at the
main architecture to imitate those human moves. Pointing
`checkpoint.initialize_from` at that prefit seeds the self-play run with an
opening-competent network. The warm start is optional and its checkpoint is not
shipped — you regenerate it.

## Evaluation

Strength cannot be read off the self-play loss, because longer games inflate the
loss without meaning the model got worse. So strength is measured directly, by
playing games against fixed opponents held in an **eval pool**.

- **Anchors** are frozen past checkpoints (from this run or earlier runs). The
  candidate plays them in **paired** games: two games that share an opening with
  the sides swapped, using common random numbers, so the pair is scored
  *pentanomially* (over the five possible pair outcomes) rather than as two
  independent games — this keeps the confidence interval honest.
- **SealBot** is an optional external C++ minimax bot. It is disabled by default;
  enabling it adds an unpaired reference opponent whose minimax depth varies with
  time (so the paired common-random-numbers trick does not apply to it).

All results feed a rolling Bradley-Terry / Elo pool, so the ratings express each
checkpoint's strength relative to the others in the pool rather than an absolute
scale. Because the shipped snapshot is early in training and evaluated on small
samples, its numbers (see [`../models/MODEL_CARD.md`](../models/MODEL_CARD.md))
should be read as a directional early-training signal, not a converged rating.

## Where things live

- Board representation and features — `packages/shrimp/python/shrimp/support.py`
  and `features.py` (reference), `packages/shrimp/rust/src/support.rs` and
  `features.rs` (production); feature indices and geometry constants in
  `constants.py`.
- The network — `packages/shrimp/python/shrimp/model.py`; the env-driven
  geometry constants in `constants.py`; the fast GPU kernels in `_triton_conv.py`
  and `_triton_attn.py`.
- The search — `packages/shrimp/rust/src/search.rs` (the drivers, including
  `run_continuous`) and `tree.rs` (the PUCT tree, Gumbel, and TSS injection).
- Self-play and training targets — `packages/shrimp/python/shrimp/selfplay.py`;
  the trainer and replay window in `trainer.py`; the shard format in `shards.py`
  and `samples.py`; the losses in `losses.py`.
- The training recipe — `configs/shrimp_main_7.toml` (heavily annotated).
- Evaluation — `packages/shrimp/python/shrimp/eval_arena.py`,
  `multistage_eval.py`, and `eval_stats.py`.
- The shipped weights and their provenance — `models/MODEL_CARD.md`.

For the ideas with no code, read [`shrimp_blueprint.md`](shrimp_blueprint.md);
for the system-wide data flow, [`ARCHITECTURE.md`](ARCHITECTURE.md); for the
exact model and target contracts, [`specs/shrimp_model_spec.md`](specs/shrimp_model_spec.md).
