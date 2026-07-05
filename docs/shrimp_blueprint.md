# Shrimp: A Beginner's Blueprint
### How a Computer Learns to Play Hexo

> **Audience:** This document assumes you know **nothing** about the board game *Hexo* and **nothing** about machine learning. By the end you should have a clear mental model of how the whole system works, end to end — not the code, but the *ideas*.
>
> **The whole thing in one sentence:** *Shrimp is a program that gets good at a board game by playing millions of games against itself — a "search" process looks ahead and finds good moves, a neural network learns to imitate that search, and the smarter network makes the next round of search even better. Repeat forever.*

**Roadmap of this document:**
1. The rules of the game (Hexo)
2. How a board is stored as coordinates
3. Turning a board into numbers a network can read
4. What the neural network is and what it predicts
5. Inside the network: local vision + global vision
6. Looking ahead: tree search
7. The engine room: how the two halves of the program talk
8. Learning by self-play
9. Measuring whether it's actually getting stronger
10. End-to-end walkthrough of a single move
11. The big ideas, and common beginner confusions
12. Appendices: glossary + numbers cheat-sheet

---

## 1. The Game of Hexo

**Hexo is "Connect-6 on a hexagonal grid."** Two players take turns placing stones on a grid of hexagons, and the first to line up **six of their own stones in a straight row wins.**

> ⚠️ **Don't confuse Hexo with the classic game "Hex."** The classic game called *Hex* is about connecting two sides of a board. **Hexo is different** — it is about getting **six-in-a-row**, like the games Gomoku or Connect-6, just played on a hex grid. Whenever this document says "Hexo," think *six-in-a-row*.

Key rules:

- **Two players** alternate turns. Player 0 goes first.
- **A hexagonal grid has exactly three straight-line directions** (call them the three *axes*), unlike a square grid which has four (horizontal, vertical, and two diagonals). So there are only three ways to make a winning line.
- **The board is effectively unbounded.** It is not a fixed 19×19 square — it grows outward as stones are placed. Because there's always more room, **games never end in a draw.**
- **The opening is forced:** Player 0's very first stone *must* go on the center cell, the origin.
- **After the opening, every turn places *two* stones.** You place one stone, then a second stone somewhere else. (This two-stones-per-turn rule is the game's *balancing mechanism* — see the analogy below.)
- **A win is checked after *each individual stone*.** If your first stone of the turn already makes six-in-a-row, you win immediately and never place the second stone.
- **Legality rule:** you may only place a stone on an **empty** cell that is **within distance 8 of some existing stone.** You can't play way out in empty space far from the action.

**Why two stones per turn?** Think of it as a small catch-up bonus that keeps the game fair. If each player placed only one stone per turn, the first player would have a permanent one-stone lead. By giving each turn two stones (after the lone opening stone), the initiative stays balanced.

**Why Hexo is tense ("sudden death").** Every stone you place sits inside many possible six-in-a-row lines at once. Your opponent can build several *threats* (lines that are almost complete) simultaneously, but you only get **two stones per turn to defend them all.** If your opponent creates two independent threats you can't both block, you lose. So a single careless move can lose the game within two or three turns. This makes Hexo sharp and tactical.

**Two useful threat words used throughout:**
- A **threat** is a six-cell line that already holds **4 or 5** of one player's stones — completable soon.
- A **standing win** ("win-in-one") is a line with **exactly 5** stones and one empty cell — the opponent *must* block that empty cell immediately or lose.

---

## 2. How the Board Is Stored as Coordinates

Before a computer can reason about the board, it needs a way to *name* every cell.

- Each hexagon is addressed by a pair of numbers **(q, r)** called **axial coordinates**. (A third value, `s = -q - r`, is implied — it's just bookkeeping for the hex geometry.)
- **Distance between two cells** uses a hex version of "king moves on a chessboard": `distance = max(|Δq|, |Δr|, |Δq+Δr|)`. This is the "within distance 8" used by the legality rule.
- The **three winning directions** are named **Q**, **R**, and **QR**.
- Every cell can also be squeezed into a single number called an **action ID** (it packs the two coordinates together). This lets moves be named, sorted, and sent between programs compactly. Crucially, the fast search engine (written in Rust) and the network code (written in Python) **pack coordinates in exactly the same way**, so they always agree on which cell is which.
- The hex board has **12 symmetries** (6 rotations × a mirror flip) that leave the game strategically unchanged. This fact gets reused later to multiply the training data for free (Section 8).

You don't need to memorize any of this — just hold onto the idea that *a position is a sparse list of "this player owns cell (q, r)" facts*, plus whose turn it is and how far into the turn they are.

---

## 3. From a Board to Numbers the Network Can Read

A neural network can't look at a picture of hexagons; it eats **numbers**. So each board position is converted into a list of cells, where every cell is described by a small fact-sheet of numbers. Two big ideas make this efficient.

### Idea 1: Only look at the part of the board that matters — the "support set."

The board is unbounded, but almost all of it is empty space far from any stone, where nothing is happening. So Shrimp only feeds the network a **support set** of relevant cells:

```
support set = (all stones)  +  (all legal move cells)  +  (a one-cell "halo" border)
```

- **Stones** — where the pieces are.
- **Legal cells** — the empty cells you could actually play on (within distance 8 of a stone).
- **Halo** — a one-cell-thick ring just *outside* that region.

> 🔍 **Analogy:** Draw a circle around the battlefield and ignore the empty wilderness beyond it. The **halo** is like the margin on a sheet of paper — it exists only so the cells at the edge have neighbors to look at. **You can never *move* onto a halo cell.**

A beautiful consequence: the input **grows and shrinks with the game** instead of being a fixed-size crop. Early on, the support set is tiny; in a sprawling endgame, it's large.

### Idea 2: Make illegal moves *structurally impossible*.

The cells are listed in a fixed order: **legal cells first, then stones, then halo.** Because the "which move?" output of the network (the *policy*, Section 4) is produced cell-by-cell, and the legal cells come first, **the network can only ever score legal moves.** There's no need to filter out illegal moves afterward — they simply aren't in the scorable region. This is called the **legal-prefix** property.

### Each cell's 15-number fact sheet.

Every cell in the support set gets **15 features** — 15 numbers describing it. In plain terms they answer questions like:

- Is this **my** stone? The **opponent's**? **Empty**? **Legal** to play on?
- What **phase** of the turn are we in? Was this the **first stone** just placed?
- How **recently** were nearby stones placed?
- How **far** is the nearest stone?
- Is this cell part of a **threat** (4-in-a-row) or a **standing win** (5-in-a-row) for either side?

That's the entire input: a variable-length list of cells, each carrying 15 numbers, plus a list of which cells neighbor which (each hex cell has up to **6 neighbors**).

### A practical wrinkle: padding.

Computers like fixed-size blocks of numbers, but our support sets vary in size. So shorter lists are **padded** with extra all-zero "dummy" rows up to a standard length. These dummy rows are carefully made **inert** — the network re-applies a mask after every step so the padding never influences the real cells' answers. (Think of it as blank filler at the end of a form: present, but ignored.)

---

## 4. The Neural Network: What It Is and What It Predicts

**What is a neural network?** It's a big mathematical function with millions of internal tunable knobs ("parameters"). You feed numbers in, predictions come out, and you improve it by showing it examples and nudging the knobs so its predictions get closer to the right answers. Shrimp's network has about **1.23 million** parameters — *small* by modern AI standards.

The network has a shared body (the **trunk**, Section 5) that feeds **three output modules called heads**:

| Head | Plain-English question | Output |
|---|---|---|
| **Policy** | "Which move looks good?" | A promising-ness score for each *legal* move |
| **Value** | "Who is winning right now?" | A number from **−1** (I'm losing) to **+1** (I'm winning), 0 = even |
| **Auxiliary** | "How long until the game ends, and how do things look a few moves out?" | A **moves-left** estimate and a few **short-term value** estimates |

Two things often surprise beginners:

- **The value is *not* a probability.** It's a "winning-ness" score for the player to move: +1 means surely winning, −1 means surely losing. And it's not predicted as a single number — the network predicts a **distribution over 65 "bins"** spanning −1 to +1 (it says how likely each bucket is, then averages). Predicting a spread instead of one number turns out to be more stable and informative.

- **The policy only covers legal moves**, thanks to the legal-prefix property from Section 3. No move-masking step is needed.

---

## 5. Inside the Trunk: Local Vision and Global Vision

The trunk's job is to transform the 15-number fact sheets into rich internal descriptions before the heads read them off. It does this by **stacking 9 layers** that alternate between two complementary kinds of "vision":

```
Layer pattern:   C  C  C  A  C  C  A  C  A
                 └──local──┘     (C = convolution,  A = attention)
```

Internally, each cell's description is widened from 15 numbers to **96 numbers** (called "channels") that the layers refine.

### Convolution = *local* vision ("talk to your neighbors")

A **convolution** here is simple: **each cell looks at itself plus its 6 hexagonal neighbors (7 cells total) and combines what they "see"** into an updated description. Stack a few of these and information spreads outward cell by cell, letting the network recognize local patterns — like "four of my stones in a row with an open end" (a threat).

A clever detail: the convolution is **direction-typed.** The neighbor in the "Q direction" is treated with *different* learned weights than the neighbor in the "R direction." This matters because, in a six-in-a-row game, *which direction a line runs* is the whole point.

> 🔍 **Analogy:** Convolution is like everyone in a stadium briefly comparing notes with the six people seated immediately around them. Do that a few times and information ripples across sections — but it's fundamentally *local*.

### Attention = *global* vision ("everyone sees everyone")

Local vision alone can't grasp whole-board strategy — a threat forming on the far side of the board needs to be noticed *now*, but convolution only spreads information one ring of neighbors per layer. That's what **attention** provides.

In an attention layer, **every cell looks at every other cell at once, in both directions** — it is *full, all-pairs* attention. A stone on one edge of the field directly reads a stone on the opposite edge in a single step; stones talk to stones, legal cells, and everything else without relaying through anything. (The only thing excluded is the inert padding from Section 3.) A small learned lookup table (the **relative-position bias**, 237 entries) doesn't *cut off* any pair — it just *nudges* how much weight each pair gets, based on their **distance and direction** apart. Pairs within distance 8 get an exact, finely-tuned bias; farther pairs share coarser "near-ring" and "far" settings, but they still attend.

On top of this all-to-all attention, Shrimp adds **8 "summary tokens"** — 8 learned slots that aren't tied to any cell and join the *same* attention sequence. They act as whole-board scratchpads: they read from every cell and every cell can read from them, so they accumulate big-picture summaries that the value and "who's winning / how long left" heads later read off. **Importantly, the tokens are *extra* global participants, not a bottleneck** — cells already communicate directly with each other; the tokens just give the network dedicated places to stash board-wide conclusions.

> 🔍 **Analogy:** Attention is like every player in the stadium being able to glance directly at every other player at once (closer/aligned players draw more of their attention). The 8 summary tokens are like 8 shared scoreboards everyone can read and write — handy for board-wide tallies, but players still see each other directly, not only through the scoreboards.

### A note on "LayerNorm"

Because board sizes vary, the network uses a stabilizing step called **LayerNorm** (rather than the more common BatchNorm). The only thing you need to know: it keeps each cell's numbers in a sensible range **independently of how big the board is or what else is in the batch** — like a per-row thermostat. This is part of why variable-size boards "just work."

Finally, the three heads read from **the 8 summary tokens plus an average of all the real cells** — i.e., from both the big-picture observers and the overall state of the board.

---

## 6. Looking Ahead: Tree Search (MCTS)

Here's a key point beginners often miss:

> **The neural network does *not* pick the move by itself.** The network gives a fast *opinion*. A separate **search** process tests that opinion by simulating many possible futures, and *the search* decides the move.

The search method is called **Monte Carlo Tree Search (MCTS)**, guided by a rule called **PUCT**.

### The search tree

Picture a branching diagram:
- The **root** is the current board.
- Each **branch** is a candidate move.
- Each **leaf** is a position the search hasn't explored yet.

The search grows this tree one step at a time. Each step is one **simulation** (also called a **visit**), and it has three parts:

1. **Selection.** Starting at the root, walk down the tree, at each point choosing the most promising branch according to the **PUCT rule** (below), until you reach a leaf.
2. **Expansion.** Ask the **neural network** to evaluate that new leaf — getting its policy (priors for the next moves) and its value (who's winning here).
3. **Backup.** Carry that value back up the path you walked, updating the running average for every branch along the way, so future simulations are better informed.

### The PUCT rule, in words

When choosing which branch to descend, PUCT balances three desires:

- **Trust the network** — prefer moves the policy head rated highly (the *prior*).
- **Trust experience** — prefer moves that have actually *led to good positions* so far (the average value, called **Q**).
- **Stay curious** — give a bonus to moves that haven't been tried much yet (the *exploration* term).

Early in the search, the network's prior dominates (we have little experience). As visits accumulate, real observed results take over. The result is a search that concentrates effort on genuinely promising lines while still checking surprises.

### Picking the move, and the search budget

After spending its budget of simulations, the search picks the move to actually play — usually the **most-visited** branch (the one it spent the most effort confirming), sometimes a slightly safer "high-value-and-confident" pick.

The default budget is **512 visits per move.** But Shrimp uses a money-saving trick called **Playout Cap Randomization (PCR)**: it runs the **full 512-visit** search on only about **33%** of moves, and a cheaper **128-visit** search on the rest. The full-search positions produce the best training targets; the fast ones keep games flowing cheaply. (This is the meaning of the shorthand "*512 visits / PCR 128 @ 33%*.")

A few small production refinements run on top — a safer "lower-confidence-bound" move pick, early-stopping when the answer can't change, exploration that scales with visit count, and a tiny "finish faster when winning" nudge from the moves-left head — all individually toggleable.

> 🔍 **One more concept — "virtual loss":** to run many simulations in parallel without them all piling onto the same branch, the search temporarily marks a branch as "busy" while a simulation is using it (a temporary traffic-jam penalty), then removes the mark on backup.

---

## 7. The Engine Room: How the Two Halves Talk

Shrimp is split across **two programming languages**, each chosen for what it's good at:

- **Rust** runs the **search** — it needs to be blazingly fast and run millions of tree steps.
- **Python (on a GPU)** runs the **neural network** — GPUs are built for the heavy math of neural networks.

The challenge: the search constantly needs the network's opinions, but calling across this boundary one position at a time would be hopelessly slow. Three tricks fix that:

- **Batching.** Instead of asking about one position at a time, Rust **collects many positions and sends them together** in one big call. GPUs are far more efficient evaluating a batch than single positions.
- **Caching.** If the *same* position comes up again (which happens constantly during search), the stored answer is **reused** instead of recomputed. Shrimp keeps up to **262,144** recent evaluations, discarding the oldest first (a simple "first-in, first-out" rule).
- **Deduplication.** If a single batch contains the same position twice, it's evaluated only once and the answer shared.

To move data quickly, positions are packed using **half-precision** numbers (smaller, faster to ship) and padded to standard sizes (like rounding up to a standard envelope size so the mailroom can handle everything uniformly). The reply back to Rust is a small, fixed contract: **policy scores + a value** (plus one optional extra).

> **Why this section matters:** the network being "smart" is useless if you can't run it millions of times affordably. **Speed here comes from engineering — batching, caching, dedup — not from a bigger network.** The network itself is deliberately small.

---

## 8. Learning by Self-Play

Now we connect the network and the search into the loop that actually *learns*. This is the heart of the whole system.

### The virtuous cycle

> **Search is smarter than the raw network** (it looked ahead!), so we train the network to **imitate the search**. The improved network then makes the *next* search smarter, which produces even better training targets, and so on.

```
   self-play games (search picks moves)
            │  produces training examples
            ▼
   train the network to imitate the search
            │  produces a better network
            ▼
   better network → smarter search → better games  ──┐
            ▲                                          │
            └──────────────────────────────────────────┘
```

### What gets recorded

The program plays games **against itself**. For every move, it records:

- **The board position** (as features).
- **The search's visit distribution** — how the search spread its 512 visits across the candidate moves. This becomes the **policy target**: "learn to want what the search wanted."
- Later, when the game ends, **who won.** This becomes the **value target**: every position from the game is labeled **+1** if the player to move there eventually won, **−1** if they lost.

Extra targets are also recorded — short-term value (how things look 1/5/10/20 moves ahead), moves-left, and the opponent's policy — to give the network richer signals to learn from.

> **Only *completed* games are kept.** If a game is cut off early, its value labels would be guesses, which would poison learning — so truncated games are thrown away.

### The replay window

Training uses only the **most recent ~100,000 positions** (a rolling "replay window"). Older data is from a weaker version of the network and would hold the model back, so it ages out.

### The training step

Periodically the trainer:
1. Pulls a batch of recent positions (grouping similar-sized boards together for efficiency).
2. **Augments** each one with a random hex symmetry (one of the 12 rotations/reflections) — free extra data that teaches the model all directions are equivalent.
3. Runs the network (**forward pass**), measures how wrong it was (the **loss**), and nudges the parameters to do better (**AdamW**, a standard optimizer).

The **loss** is a weighted sum of how wrong each head was:

```
total loss = 1.0 × policy      (match the search's move distribution)
           + 1.0 × value       (match the eventual game outcome)
           + 0.25 × opponent-policy
           + 0.1  × short-term-value
           + 0.1  × moves-left
```

The value head is trained with **cross-entropy over its 65 bins** (matching a distribution), not a single-number error. Everything is averaged **per game-position**, not per cell — a deliberate choice so that big boards (many cells) don't drown out small ones.

---

## 9. Measuring Whether It's Actually Getting Stronger

Measuring progress is **surprisingly hard and easy to get wrong.**

> ⚠️ **The biggest trap: a *rising* training loss does NOT mean the model is getting worse.** As the model improves, its self-play games get *longer and more competitive*, which naturally inflates the loss number even as real strength climbs. **Never judge strength by the self-play loss.**

So how is strength measured honestly?

- **Play against fixed opponents.** New checkpoints play **arena games** against *fixed* reference opponents: a traditional non-neural bot called **SealBot** (pinned at "0 Elo" as the origin of the rating scale) and a set of frozen past versions ("anchors"). Because the opponents don't move, improvements are visible.
- **Paired games to cancel luck.** Games are played in **pairs that share the same opening but swap who plays first.** This cancels out the luck of getting an easy opening (a technique called *common random numbers*).
- **Honest statistics.** Because the two games in a pair are *correlated* (not independent coin flips), naive confidence intervals would look falsely precise. Shrimp uses **pentanomial scoring** (counting the five possible outcomes of a pair) to report **honest, appropriately-wide** error bars, and a **Bradley-Terry rating pool** that combines *all games ever played* into ever-tightening **Elo** estimates.
- **Verdicts are informational only.** The evaluator may print **PROMOTE / REGRESS / INCONCLUSIVE**, but these labels **never stop or redirect training** — there's no automated gate yanking the wheel.
- **Head audit.** A separate sanity check confirms the auxiliary **moves-left** prediction actually correlates with reality; if that head goes haywire, it's automatically switched off so it can't mislead the search.

---

## 10. End-to-End Walkthrough: One Move, Start to Finish

Let's trace a single move through everything above. It's the middle of a self-play game, and it's our turn.

1. **Board.** The engine holds the current position: which stones belong to whom, the turn phase, and which lines are already threats. We're in the "first stone" phase of our turn.
2. **Support + features (Section 3).** The system sweeps outward from the stones to build the **support set** — stones, legal cells, and the one-ring halo — orders them as *[legal | stones | halo]*, and computes the **15 features** for each cell.
3. **Search asks the network (Sections 6–7).** The search wants to evaluate this position. Rust packs the support set into a compact batch and ships it to the Python network — but first checks the **cache** in case this exact position was already evaluated.
4. **Network forward pass (Section 5).** Python pads the board to a standard shape, moves it to the GPU, and runs the trunk (`C C C A C C A C A`): local hex convolutions spot nearby patterns, then full all-pairs attention lets every cell read every other cell directly (with 8 summary tokens as shared whole-board scratchpads). The heads return a **policy** (scores for our legal moves), a **value** (say **+0.55**, moderately winning), and a **moves-left** estimate. The answer is cached.
5. **Search looks ahead (Section 6).** Using those priors and value, the search runs ~512 simulations, each walking down the tree by **PUCT**, evaluating new leaves (more batched network calls), and backing results up the path.
6. **Pick the move.** The search now has a visit count per candidate. It plays the best one (here, the most-visited move, nudged to win a touch faster by the moves-left bonus). We place the stone; the engine checks for six-in-a-row (none yet) and moves us to the "second stone" phase to place our second stone the same way.
7. **Record for learning (Section 8).** This position is saved with its **search visit distribution** (policy target) and root value.
8. **Game ends, labels applied.** When the game finishes, every saved position is labeled with the **final winner** (+1/−1) plus the extra targets. Because the game *completed*, it's written to disk.
9. **Training (Section 8).** Later, the trainer pulls the recent ~100k positions, applies a random symmetry to each, batches them, and runs forward + backward passes, nudging the ~1.23M parameters to better match the search and the outcomes.
10. **The loop closes.** The improved network makes the next search sharper → better targets → an even better network. Periodically, the evaluator (Section 9) plays paired games against SealBot and the anchors to report, honestly, whether real strength went up.

---

## 11. The Big Ideas, and Common Confusions

### The through-lines (the ideas that tie everything together)

1. **The virtuous cycle is the whole point.** Search beats the raw network → the network learns to imitate search → smarter search → repeat. Every other piece is machinery serving this loop.
2. **Work on only the relevant part of an infinite board.** The support set lets the system *scale with the game* instead of cropping to a fixed size — and that one choice ripples everywhere (LayerNorm, padding, batching, per-position loss averaging).
3. **Make illegality structurally impossible, not patched afterward.** The legal-first ordering means the policy can only ever score legal moves. No masking, no wasted capacity, no coverage bugs.
4. **Two scales of vision, combined.** Local hex convolutions see immediate patterns and threats (a cell sees only its 6 neighbors, range growing one ring per layer); full all-pairs attention lets every cell read every other cell directly for whole-board strategy, with 8 summary tokens as shared global scratchpads. Neither alone is enough.
5. **Honesty under noise.** Per-position loss averaging, paired games with swapped sides, pentanomial confidence intervals, discarding truncated games — the design repeatedly refuses to fool itself.
6. **Speed comes from engineering, not a bigger brain.** Batching, caching, deduplication, half-precision wire format, fixed padded shapes, and the Rust/Python split make millions of evaluations affordable — while the network stays small (~1.23M parameters).

### Common beginner confusions (and how to think about them)

- **"Hexo = the game Hex."** No — Hexo is **six-in-a-row** on a hex grid, *not* the connect-the-sides game called Hex.
- **"The network picks the move."** No — the network gives an *opinion*; the **search** decides, by testing that opinion over hundreds of simulated futures.
- **"Two stones per turn means double power."** It's a **balancing** mechanism, and it's *why* the game is sudden-death: you have only two stones to defend *all* threats.
- **"Value is a probability."** It's a winning-ness score from −1 to +1 for the side to move, predicted as 65 bins — not a single probability.
- **"Rising training loss = getting worse."** A documented trap — longer games inflate the loss while strength still rises. The **arena evaluation**, not the loss, is the real strength signal.
- **"Halo cells are playable."** No — the halo exists only to give convolutions neighbors. You can never move there.
- **"More search is free."** Each of the ~512 visits is a network evaluation; that's exactly *why* batching and caching exist.
- **"The PROMOTE/REGRESS verdict controls training."** It doesn't — it's informational and never halts or redirects the run.

---

## Appendix A. Glossary

**Game terms**
- **Hexo** — the board game: six-in-a-row on a hexagonal grid (Connect-6 on hex). *Not* the classic game "Hex."
- **Stone** — one piece a player places on a cell.
- **Cell** — one hexagon, addressed by coordinates (q, r).
- **Axial coordinates (q, r)** — the two-number address of a cell; a third axis `s = −q−r` is implied.
- **Hex distance** — steps between two cells: `max(|Δq|, |Δr|, |Δq+Δr|)`.
- **Win axis** — one of the three straight-line directions (Q, R, QR) along which six-in-a-row counts.
- **Origin** — the center cell where Player 0 must place the first stone.
- **Phase** — where you are in a turn: Opening, FirstStone, or SecondStone.
- **Legal move** — an empty cell within hex-distance 8 of any stone.
- **Threat** — a six-cell line holding 4+ of one player's stones (completable soon).
- **Standing win / win-in-one** — a line with exactly 5 stones and one empty cell; must be blocked at once.
- **Action ID** — a single number that uniquely names a cell by packing its (q, r).
- **D6 symmetry** — the 12 rotations/reflections of the board that leave the game unchanged.

**Representation terms**
- **Support set** — the cells the network looks at: stones + legal cells + halo.
- **Halo** — the one-cell border just outside the relevant region; gives convolutions neighbors; never playable.
- **Feature** — one number describing a fact about a cell; Shrimp uses 15 per cell.
- **Legal-prefix property** — listing legal cells first so the policy can only score legal moves.
- **Padding** — inert all-zero filler rows so variable-size boards fit a fixed shape.

**Network terms**
- **Neural network** — a large adjustable function tuned by examples to turn inputs into predictions.
- **Parameter** — one tunable number inside the network (Shrimp has ~1.23 million).
- **Trunk** — the shared body of the network that feeds the output heads.
- **Head** — an output module: policy, value, or auxiliary.
- **Policy** — per-move promising-ness scores.
- **Value** — estimate of who's winning, from −1 to +1.
- **Bins (distributional output)** — predicting a probability for each of 65 buckets across −1…+1 instead of one number.
- **Channels** — the width of each cell's internal description (96 here).
- **Convolution (hex, 7-tap, direction-typed)** — each cell combines its own features with its 6 neighbors', with per-direction weights.
- **Attention** — a mechanism letting cells and the 8 summary tokens share information across the whole board.
- **Summary token** — one of 8 learned "roving observers" gathering the big picture.
- **Relative-position bias** — a 237-row learned lookup adjusting attention by distance/direction.
- **LayerNorm** — normalization that stabilizes each cell independent of board size.

**Search terms**
- **MCTS** — Monte Carlo Tree Search: looking ahead by building a tree of futures, guided by the network.
- **Tree / root / leaf** — the search structure; root = now, leaf = an unexplored position at the edge.
- **Simulation / visit** — one trip down the tree; the search budget is counted in these.
- **PUCT** — the branch-choosing rule balancing prior, observed value, and exploration.
- **Prior** — the network's policy probability for a move, biasing search.
- **Q value** — the average observed value of a move over the simulations that tried it.
- **Backup** — pushing a leaf's evaluation back up the tree.
- **Virtual loss** — a temporary "busy" penalty so parallel simulations don't collide.
- **PCR (Playout Cap Randomization)** — full 512-visit search on ~33% of moves, cheap 128-visit search on the rest.
- **Moves-left head** — a prediction of how many moves remain; a minor search tie-breaker.

**Learning terms**
- **Self-play** — the model playing games against itself to generate training data.
- **Replay window** — the rolling set of ~100k most-recent positions used for training.
- **Shard** — one saved file holding a finished game's positions and targets.
- **Target** — the desired output the network learns to match (policy target = search visits; value target = game outcome).
- **Loss** — a number measuring how wrong predictions are; training minimizes it.
- **Cross-entropy** — the loss used to match a predicted distribution to a target distribution.
- **AdamW** — the optimizer that nudges parameters to reduce the loss.
- **Augmentation (D6)** — feeding rotated/reflected copies so all directions are treated equally.

**Engine / evaluation terms**
- **Batching** — grouping many positions into one network call for speed.
- **Cache** — stored network answers reused when a position recurs (FIFO, up to 262,144 entries).
- **Half-precision (f16)** — smaller, faster numbers used to ship data between Rust and Python.
- **Elo** — a number rating relative strength; higher is stronger.
- **SealBot** — a traditional non-neural bot pinned at 0 Elo as the rating origin.
- **Anchor** — a frozen past checkpoint used as a stable reference opponent.
- **Paired games (common random numbers)** — two games with the same opening but swapped sides, to cancel luck.
- **Bradley-Terry pool** — a rating model fitting one Elo per player from all results.
- **Pentanomial scoring** — counting a pair's five possible outcomes for honest confidence intervals.
- **Verdict (PROMOTE / REGRESS / INCONCLUSIVE)** — the evaluator's informational label; never alters training.
- **Head audit** — a check that the moves-left head correlates with reality, auto-disabling it if not.

---

## Appendix B. Key Numbers Cheat-Sheet

| Number | Meaning |
|---|---|
| **8** | Legality radius — new stones must be within hex-distance 8 of a stone |
| **6** | Length of a winning line (six-in-a-row); also the number of neighbors per hex cell |
| **3** | Winning directions (axes) on a hex grid |
| **4 / 5** | Stones in a line that make a *threat* / a *standing win* |
| **15** | Input features per cell |
| **96** | Internal channels (width of each cell's description in the trunk) |
| **9** | Trunk layers, arranged `C C C A C C A C A` |
| **8** | Summary tokens (roving global observers) |
| **237** | Rows in the relative-position bias table |
| **65** | Value bins (the −1…+1 distribution) |
| **~1.23M** | Total network parameters (small by modern standards) |
| **512** | Default search visits per move |
| **128 @ 33%** | PCR: full 512-visit search on ~33% of moves, 128-visit search on the rest |
| **~100,000** | Positions in the training replay window |
| **262,144** | Maximum cached evaluations (FIFO) |
| **128 (64 pairs)** | Evaluation games per epoch, paired with swapped sides |
| **0 Elo** | SealBot, the fixed origin of the rating scale |

---

*This blueprint is a conceptual companion to the technical specifications in `docs/specs/shrimp_model_spec.md` (model design & rationale) and `docs/specs/shrimp_eval_v2_spec.md` (evaluation), and to `docs/ARCHITECTURE.md` (where each piece lives in the code).*

> **Note on numbers.** This blueprint teaches the *ideas* with a concrete example
> configuration — a 96-channel, 9-layer `C C C A C C A C A` net (~1.23M
> parameters) at 512 visits. The **shipped** weights in `models/` are a larger
> instance of the *same architecture*: 192 channels, 3 attention heads, trunk
> `CCACCACCACCACCA` (~8.1M parameters) at 1024 visits. The mechanisms are
> identical; only the sizes differ. See `models/MODEL_CARD.md` for the shipped
> configuration.
