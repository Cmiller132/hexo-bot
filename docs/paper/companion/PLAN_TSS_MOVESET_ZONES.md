# TSS solver move-set trimming — locality theorems, the r=2 question, and relevance zones

> **Provenance.** 2026-07-13. Companion to `PLAN_TSS_DEEPENING.md` §6 (Lever 3,
> the df-pn solver's AND-node move sets). Owner question: the solver's defender
> enumeration at `k < B` nodes is "every legal move" (= every empty within hex
> distance 8 of any stone, `LEGAL_RADIUS = 8`); can it be trimmed to radius 2
> around existing stones, given that a turn places at most 2 stones so no
> window below count 4 can complete within one turn? Also: moves in no active
> window are dead — prune them.
>
> This document went through one full adversarial cycle: Claude drafted the
> theory + counterexample, a Codex `gpt-5.6-sol` ultra pass attacked every
> claim against the Rust sources, and a CPU solver (mini-model cross-validated
> against `hexo_engine`) adjudicated the constructions. Codex refuted the
> first junction counterexample and the zone theorem's bookkeeping, and
> contributed a new gap class (G3) plus the repair for the construction; the
> solver experiments confirmed both the refutation and the repairs. Verdict
> table in §11.
>
> **Verdict in one line.** Per-node r=2 is provably complete for every
> *single-window* defender purpose (hits, win-now completions, killing a given
> window, advancing a given line — Theorems 1–4), and the owner's one-turn
> tempo argument is the correct core of those proofs; but it is **unsound as a
> blanket defender restriction** because of *multi-window* cells: attacker
> junction pre-blocks (G1) and defender counterforks (G3) live at distance
> 3–4 and can each be the unique game-saving move (concrete positions below,
> machine-checked). A certificate-relative "relevance zone" that would let a
> verifier certify trimmed searches is outlined (§8) but its first version was
> refuted on two counts and it is **not yet proven** — so until the Stage-3
> proof-carrying spec closes it, move-set trimming may ship only as a search
> *heuristic* whose proofs are re-verified against the full legal move set
> (§10).

## 1. Ground truth (from source, 2026-07-13)

| Fact | Source |
|---|---|
| Windows are 6 collinear cells on 3 axes; each cell belongs to 18 windows | `hexo_engine/rust/src/tactics.rs:14-17` |
| Win = one window fully filled by one colour, checked per placement, game ends immediately | `tactics.rs:206-208` (`is_win_for`), `state.rs:309` |
| Threat = single-colour ("active") window with count ≥ 4 | `tactics.rs:189-198` |
| Stones are permanent; a two-coloured window is dead forever | placement-only rules; no capture |
| Turn budget `B`: 2 at FirstStone, 1 at Opening/SecondStone; per-node = per-placement | `threats_shared.rs:48-53` |
| Legal placement = empty cell within hex distance 8 of any stone | `legal.rs:18`, `rules.rs` |
| λ¹: `own_win_now` = own count-5 (any B) or count-4 (B=2 only); forced loss ⇔ `min_hitting_set > B`; k ≤ 2 exhaustive | `threats_shared.rs:81-178` |
| The opening permanently occupies the origin | `rules.rs:16-23` |

Definitions. `dist` = hex distance = max(|dq|, |dr|, |dq+dr|). A window is
**alive for P** if it contains no opponent stones (all-empty windows are alive
for both). A window is **dead** if it contains both colours. A cell is
**dead** if all 18 windows through it are dead. The **hitting universe** at a
defender node = every empty of every opponent ≥ 4 window. **Per-node
r-universe** = empties within distance r of *any* stone currently on the
board (re-evaluated at every node, so it grows as a line is played out).
Constructed positions below are stated in coordinates relative to their
focal cell; embed on the real board by translating away from the occupied
origin (Codex reachability note).

## 2. Lemma 0 — the packing bound  *(Codex: CONFIRMED, tight)*

*In a length-6 window with `s ≥ 1` stones, every empty cell of the window is
within distance `6 − s` of some stone of the same window.*

Proof. Index the window cells 0–5. Let empty cell `e` have nearest in-window
stone at distance `d`. The `d` cells from `e` toward that stone (inclusive of
`e`, exclusive of the stone) are all empty and all in the window, so
`d ≤ 6 − s`. Tightness: stones at offsets `6−s … 5`, empty at offset 0. ∎

Instances: count-5 → empties within 1; count-4 → 2; count-3 → 3; count-2 → 4;
count-1 → 5.

## 3. Theorem 1 — single-turn tactics live inside r=2  *(Codex: CONFIRMED, current-node scope)*

*Every win-now completion and every hitting cell is within distance 2 of an
existing stone.*

Proof. A window first completed within one turn (≤ B ≤ 2 placements) had
≥ 4 same-colour stones before the turn; its empties are within `6 − 4 = 2` of
those stones (Lemma 0). Hitting cells are empties of opponent ≥ 4 windows —
same bound. ∎

This is the owner's observation — "no window below 4 can complete in one
turn" — doing real work: *all* immediate tactics, both colours, are r ≤ 2. The
`k == B` fully-forced defender set is therefore always inside r=2, as is λ¹'s
tactical set (the comment at `threats_shared.rs:34-38` says ≤ 5; the tight
bound is 2). Scope (Codex): this bounds *current* tactics only — it says
nothing about prophylactic placements against future windows.

## 4. Theorem 2 — threat creation lives inside r=3, not r=2  *(Codex: CONFIRMED, tight)*

*Any placement creating a ≥ 4 alive window is within distance 3 of an existing
own stone; distance 3 is achieved (stones at window offsets {3,4,5}, placement
at offset 0).*

Proof. Creating count ≥ 4 needs a window already holding ≥ 3 own stones and no
opponent stones; Lemma 0 with s = 3. ∎

Corollary (OR side). The attacker generator must use r=3 or be window-driven
(empties of own-alive count ≥ 3 windows). r=2 silently loses real threat
moves — safe by direction (missed wins → UNKNOWN) but weaker for no reason.
Both machine-checked counterexample positions below are *triggered* by
distance-3 moves.

## 5. Theorem 3 — killing a given window costs distance 1  *(Codex: CONFIRMED, qualified)*

*Any alive window containing ≥ 1 stone and ≥ 1 empty can be two-coloured by an
**opposing** placement at distance 1 from one of its stones (non-terminal
states; the placement is legal by `LEGAL_RADIUS ≥ 1`).*

Proof. The cells are collinear; a stone–empty boundary pair exists whenever
both sets are non-empty within 6 consecutive cells. ∎

So killing any *single* window never requires leaving r=2. What can require
distance 3–4 is killing (or creating) **several windows with one stone** — the
gaps in §7.

## 6. Lemma 4 — isolated-window completion dominance  *(replaces the refuted Theorem 4)*

The draft claimed: a defender able to complete a window W against a fixed
attacker continuation can complete it with every placement adjacent to his
prior stones in W. Codex refuted the exact statement (two order holes beyond
the conceded one): reordering can complete a *crossing* window earlier
(per-placement win checks make the game end on a different move — strategically
fine, but it falsifies "completes W"), and order changes the legality
frontier (`LEGAL_RADIUS` reach depends on which stones exist when). The
salvageable, and sufficient, statement:

*Lemma 4. Let W be alive for the defender with ≥ 1 defender stone. Against an
attacker continuation that is fixed and never places in W, if the defender can
force a win inside W in `m` further placements, he can force a win at least as
early using only placements inside W, each adjacent to a defender stone
already in W — provided his placements outside W are not load-bearing for
legality of the in-W cells (automatic: in-W cells are within distance 5 of
W's own stones, hence legal throughout).*

Proof sketch. Restrict to the in-W fills of the original plan; order them
boundary-first (Theorem 3 induction). Completion when the last empty fills;
any crossing-window early win only helps. ∎

What Lemma 4 does **not** cover — and the draft wrongly claimed it did —
is counter-threat *fork* creation: making two or more count-4s with one stone
is cell-specific and cannot be reordered around. That is G3.

## 7. The three genuine gaps in bare per-node r=2

Each gap is a family of cells at distance 3–4 that can be the unique
game-relevant move. G1 and G3 are machine-checked below.

### 7.1 Gap G1 — attacker junction / efficient multi-window prophylaxis

One defender stone kills every alive opponent window through its cell.
Killing one window at a time is always available inside r=2 (Theorem 3), but
killing **two or more per stone** may require the *junction* cell where
several low-count attacker routes cross — and a junction of count-c routes
can sit at distance `6 − c` (3–4 for c = 3–2) from every stone (Lemma 0
tightness). When the defender's spare rate (`B − k` per turn) is slower than
the attack's route production, junction efficiency is the only defense, and
r=2 deletes it → false WIN.

Machine-checked construction (v2 — the v1 draft, without the caps, was
refuted by Codex and the refutation was confirmed by the solver: the attacker
answers a defender junction stone with *outward* arm extensions whose shifted
windows avoid the junction). Four count-3 attacker routes crossing at j, arms
at offsets ±{3,4,5} along the Q and R axes; **defender caps at offset ±6 on
each arm** so every shifted arm window contains j or a cap; a single-window
count-4 attacker pin far away (stones on both endpoints of its window, so
k = 1 and the defender keeps exactly one spare placement); scattered
pairwise-non-cowindow defender stones for material balance. Defender to move,
B = 2:

```
                     (0,-6) cap D
                     (0,-3)(0,-4)(0,-5) A
                           |
 cap D  (-3,0)(-4,0)(-5,0) A   j   A (3,0)(4,0)(5,0)  cap D      pin A: count-4,
 at(-6,0)                  |                                     single window,
                     (0,3)(0,4)(0,5) A                           k=1, far away
                     (0,6) cap D
```

- r=2 defender: pin-hit + spare. Any r=2 spare kills at most one route (route
  windows pairwise share only j). Attacker plays j (a distance-3
  threat-creating move, Theorem 2 tightness): the ≥ 3 surviving routes become
  count-4s with pairwise-disjoint empty pairs → `min_hitting_set ≥ 3 > B` →
  λ¹ forced loss.
- Defender plays j (distance 3, outside r=2): all four routes and, via the
  caps, all their outward shifts are dead; the attacker's remaining pressure
  (pin-line extensions) is single-window, k ≤ 2 per turn, and the defender
  holds.

Solver results in §9 (B2). Junction cells always lie inside empties of
*opponent*-alive windows of count ≥ 1, which is what makes them enumerable.

### 7.2 Gap G2 — remote virgin seeding

A defender campaign in fresh territory needs 6 placements in one window (win)
or 4 (first fork/threat). Per-node r=2 reaches any cell eventually by chaining
(+2 per placement) but chaining burns tempo; a direct distant seed can be
strictly faster. By the race arithmetic of §8 a virgin campaign is relevant
only against certificates that leave the defender ≥ 4–6 free placements
before the horizon — no fixed radius handles it; only the horizon-accounting
term can.

### 7.3 Gap G3 — defender counterforks  *(contributed by the Codex review)*

The defender-side mirror of G1: a single stone that creates **multiple**
count-4 counter-threats at once. Creating *one* counter-threat is available at
distance 1 (Theorem 3 / Lemma 4), but a *fork* needs the specific crossing
cell of ≥ 2 own count-3 lines, which sits at distance 3 (Lemma 0). A triple
fork makes `min_hitting_set = 3 > B` — the **attacker** is λ¹ forced-lost on
the spot, so the fork can dominate every r=2 alternative whenever slower
defenses lose.

Machine-checked geometry (experiment G3): defender arms `Q: (8,0)(9,0)(10,0)`,
`R: (5,3)(5,4)(5,5)`, `QR: (8,-3)(9,-4)(10,-5)` — three count-3 alive windows
whose unique common cell is `f = (5,0)`, at distance exactly 3 from every
stone. Playing f yields three defender count-4 windows with pairwise-disjoint
empty pairs `{(6,0),(7,0)}, {(5,1),(5,2)}, {(6,-1),(7,-2)}` →
`min_hit = None` → attacker forced-lost. f is outside r=2, inside r=3.

## 8. The relevance-zone program (certificate-relative)

> **Superseded by the formal proof document.** The program sketched in this
> section and §8.1–8.2 has since been executed:
> **`docs/PROOF_TSS_DEFENDER_ZONES.md`** contains the formal framework, the
> main dismissal-soundness theorem (T3, with the horizon-scaled frontier
> guard and adaptive leaf contracts), the n-relevance closure, the pairing
> impossibility theorem, the Erdős–Selfridge potential layer, the
> domination-pattern layer, mechanized validation results, and a
> four-round hostile review log. That document is normative for the
> Stage-3 build; this section remains as the design rationale.

### 8.0 Original sketch (historical) — status at time of writing: NOT YET PROVEN

The target design: at an AND node of a WIN certificate, dismissed defender
replies are covered wholesale by a *blanket lemma* — "the attacker continues
the certificate as if the stone were not there" — and a verifier re-checks the
blanket's side conditions per certificate. The draft's first formalization
(zone = hitting ∪ certificate core ∪ a `6 − F` window-count term, F = quiet
defender placements before the horizon) was **refuted by the Codex pass on
two counts**, both now understood:

1. **Forks force without completing.** A dismissed reply can create ≥ 1
   count-4 (or a G3 multi-fork) that invalidates the certificate's timing
   without ever completing a window — the F-term only tracked completions.
   Any repaired zone needs a fork term; the conservative static version is
   *empties of defender-alive windows with count ≥ 3* (distance ≤ 3, cheap
   from the window store).
2. **Forced hits can feed a defender window.** The F-accounting assumed only
   quiet placements contribute to a remote defender window W. False: cells
   can lie in an attacker threat window *and* a defender-alive window
   simultaneously, and Codex exhibited a seed/trigger arrangement where four
   consecutive *forced hits* trace out W. The repaired completion condition
   must be per-window: `s(W) + F + H_W ≥ 6`, where `H_W` counts certificate
   hit cells inside W (enumerable at verify time) — or, conservatively, use
   *all* defender placements before the horizon.

Also required (Codex): T and F defined by absolute placement indices with
strict `p < T` (wins are checked per placement; there is no "same turn" tie),
and phase-aware own-win priority (`own_win_now` beats forced-loss at the same
node, `threats_shared.rs:81-89`).

What survives of the program: the blanket-lemma *shape* is right; the
certificate core catches G1 (junctions are future certificate windows); the
`6 − F` arithmetic is confirmed as far as it goes (Codex verdict 8); F = 0
(fully forcing certificates) still collapses the zone to the hitting
universe — the owner's forced-tree thesis remains a theorem *in that regime*,
since with zero spare placements there are no quiet stones to account for at
all. The full parameterized statement, with the fork term, per-window `H_W`,
and DAG-induction bookkeeping, is exactly the kind of lemma the Stage-3
proof-carrying spec (owner decision: delegated to Codex) must deliver with
the solver — it is *not* assumed anywhere until then.

### 8.1 The proof program (owner question: is sufficiency provable, and how)

Unbounded sufficiency of any static set is **provably impossible** (G2: with
enough spare tempo the defender can seed anywhere). The provable statement is
horizon-parameterized — "sufficient within n placements" — and the
conservative version turns out to be *simpler* than the refuted original:

**Target theorem (conservative D-version).** At an AND node of a WIN
certificate, let `D` = the number of defender placements at plies strictly
before the attacker's winning placement (all of them — forced or quiet; this
single change subsumes *both* Codex holes: forced-hit contributions are
counted, and forks matter only as completion accelerators, which
placement-counting already bounds). Then dismissing defender reply c is sound
if:
- **(A) node condition:** every defender-alive window W on the board has
  `s(W) + D < 6` — no defender completion is possible before the horizon no
  matter how all D placements are spent. Where (A) fails, the failing
  windows' empties enter the zone (this *is* the `6 − D` window-count term).
- **(B) cell condition:** c is core-disjoint — it touches no window any
  remaining certificate verdict or placement depends on.

**Why this is now feasible to prove.** The exhaustiveness worry (the G3
lesson: strategy-level taxonomies miss cases) disappears by working at the
*causal-channel* level: a placed stone affects the game **only** through (i)
occupancy of its cell, (ii) the 18 window masks it increments, (iii) the
legality frontier it extends. Wins, threats, and λ¹ verdicts are functions of
window masks alone, legality is a function of stone positions — so the
channel list is mechanically complete, not a judgment call. The proof then
needs four half-page lemmas (mask permanence/monotonicity; completion
counting `6 − s`; λ¹-verdict transfer under core-disjoint extra stones;
legality monotonicity + occupancy-conflict exclusion via the core) and one
induction over the certificate DAG on remaining horizon.

**Coverage corollary.** The §10 static set (r=3 ∪ opponent-active-window
empties ∪ hitting) contains the `6 − D` term for **D ≤ 3** (count ≥ 3 windows
have empties within distance 3), so it is sufficient — modulo the
cert-enumerable core — for every certificate that finishes within 3 defender
placements (≈ 7–8 plies). Deeper certificates widen the window-count term per
the F-table; at D ≥ 6 the zone honestly degrades to all-legal. The payoff
concentrates exactly where the forced-tree thesis lives: fully-forcing
certificates keep D-effects nil at any depth. A sharper second theorem
(`s(W) + F + H_W < 6`, charging forced hits per window) restores tight zones
for deep certificates at the cost of per-branch bookkeeping.

**Robustness stack (defense in depth), in order:**
1. The theorem above, written and adversarially reviewed (Stage-3
   proof-carrying spec; Codex builds to it).
2. **Bounded machine verification** — the owner's "proven within n
   placements", made literal: (a) the base lemmas are *local* (per-window
   facts), so exhaustively enumerate all window-neighborhood configurations
   up to D6 symmetry and check them by machine; (b) extend
   `scripts/_tss_moveset_zone_experiments.py` into a model checker that, from
   large curated + random position sets, exhaustively verifies to horizon n
   that no zone-omitted defender move changes any game value (the B2 harness
   already does exactly this for single positions).
3. Runtime enforcement forever: the verifier re-checks conditions (A)/(B) on
   every certificate — a proof gap can downgrade results to UNKNOWN but can
   never back a false value into the tree (§2 of the main plan).

### 8.2 Toolkit for further trimming and increasing-n proofs (owner direction)

Goal: trim below the §10 static set and prove sufficiency for increasing
horizons n, without game-tree brute force. Four established techniques apply;
they compose, and each carries its own proof obligation style:

1. **Domination / inferior-cell patterns** (precedent: Hayward et al.'s Hex
   solver theory; Kishimoto–Müller relevance zones in tsume-go df-pn).
   Formalize n-domination: reply b is n-dominated by a if every continuation
   after b is simulated after a with outcome ≥ within n placements. Each
   pattern is a *local* lemma over a bounded neighborhood, provable by
   exhaustive case enumeration up to D6 symmetry (four-color-theorem
   methodology — finite local case checking inside a proof is rigorous; it is
   not tree search). Non-dominated-only generation subtracts from every Z_n.
2. **Potential functions** (Erdős–Selfridge 1973; Beck's biased
   generalizations). Adapt the biased blocking theorem to hex-6 / B=2: a
   window-weighted potential Φ = Σ over attacker-alive windows of
   2^(−empties) (suitably biased for double placement) with the theorem "Φ
   below threshold ⇒ greedy-Φ defense (always inside touched windows ⊆ the
   §10 set) blocks forever." Gives horizon-UNBOUNDED sufficiency in its
   domain, checkable in O(threats) per node.
3. **Pairing strategies** (Hales–Jewett; Zetters' 8-in-a-row draw). Attempt a
   periodic pairing for length-6 windows on the 3 hex axes: every window
   contains a dedicated pair; defender answers each stone with its partner.
   Existence is open for this geometry; verifying a candidate periodic
   pattern is finite and mechanical. Prize: defender set of size 1 in
   pairing-intact regions + potential late-game adjudication.
4. **Inductive closure ("n-relevance" fixed point)** — the compute-and-prove
   backbone. Semantic n-irrelevance (occupying the cell changes no game
   value within n placements) is over-approximated by a syntactic closure:
   R₀ = win-now ∪ hitting; R_{k+1} adds cells touching windows that can
   become decisive within k more placements (count ≥ 6 − available, both
   colours, plus hitting-set-overload interactions). The §8.1 channel lemmas
   prove by induction on n that the closure contains all semantically
   relevant cells. Computable per node as a DP over the window store,
   O(windows × n); shrinks automatically in forced regions. The §10 static
   set is ≈ R₂; the closure generalizes it with the proof attached.

Composition: Z_n from the closure (4), minus domination patterns (1), capped
by the potential-safe region (2), collapsed where a pairing holds (3); the
runtime certificate verifier remains the fail-safe under all of it. Build
order: Phase A core theorem + closure (spec spine); Phase B pattern library
(Codex grinding under proof-carrying rules, patterns machine-enumerated);
Phase C potential threshold + pairing attempt; Phase D mechanized local-case
checkers, optional Lean formalization of the core (finite combinatorics —
feasible) yielding a formally-derived certificate checker.

## 9. Experiments (mini-model cross-validated against `hexo_engine`)

Script: `scripts/_tss_moveset_zone_experiments.py` (promote into `tests/`
with the Stage-3 work). The mini-model reproduces engine legality and terminal
detection move-for-move on random games (V). 3-valued depth-limited AND/OR
solver; attacker restricted to threat-creating moves (under-generation-safe:
its exhaustion yields UNKNOWN, never LOSS); defender exhaustive over the
universe variant under test, with λ¹ dispatch; a defender-restricted non-loss
soundly implies a FULL non-loss (restricting the defender is
defender-pessimal).

| Exp | Result |
|---|---|
| V | mini-model ≡ engine on random games (legal sets + win detection) |
| A | 400 random positions: max dist(win-now/hitting cell → nearest stone) = **2**; max dist(threat-creating cell) = **3** — Theorems 1–2 tight in practice |
| C | mean defender-universe sizes: FULL (legal) ≈ **302**, r=3 ≈ **121**, r=2 ≈ **80** |
| G3 | Codex fork geometry confirmed: f at distance 3, outside r=2; after f the attacker faces 3 disjoint count-4s, `min_hit = None` (λ¹ forced-lost) |
| B2 v1 (uncapped) | Codex's refutation of the v1 construction confirmed by direct line replay: after defender [pin-hit, j], attacker (6,0)+(−6,0) leaves the defender facing 6 threats, `min_hit = None` — v1 was lost in every variant |
| B2 v2 (capped) | **r=2 unsound, machine-verified.** Root solve, defender restricted to per-node r=2: ATTACKER WINS — complete refutation of every r=2 defense (2,374 nodes, no caps). All 217 [pin-hit, r2-spare] turn combinations individually proven lost. The excluded distance-3 junction move j: [pin-hit, j] not refuted through a deeper horizon (depth 8, no caps); by defender-pessimality this lower-bounds the unrestricted defender |
| B1 | 109 random attackable positions, R2-WIN vs FULL: **0 genuine divergences** (the single flagged case was node-cap noise; re-adjudicated at 2M nodes, both variants prove the same WIN, uncapped). The unsound cells are adversarially rare — they essentially never arise in random play, only in engineered route/fork geometry |

Methodology notes: fork-degree move ordering (junction-style multi-window
moves first) collapsed the r=2 refutation from >400k nodes (cap-bound) to
2,374 — a direct preview of why the real solver's `H∪C` ordering matters.
Survival probes use a defender-side short-circuit (first unrefuted child →
not-proven-lost), sound for that claim only, never for LOSS proofs. Patch
bounds must cover the attacker's continuation region (an early run
artificially "saved" the defender by clipping pin-line extensions at the
patch edge — worth remembering as a solver-testing trap).

## 10. Recommended solver/verifier posture (revised after the review)

1. **Sound today, no further proof needed:** the `k == B` hitting-universe
   restriction (λ¹, already the plan's instant dispatch); *attacker*-side
   trimming of any kind (under-generation only costs missed wins → UNKNOWN);
   the window-driven attacker generator per Theorem 2 (r=3 equivalent).
2. **Search-time heuristic (allowed to be wrong):** defender candidate
   generation (owner-proposed variant, strictly stronger than the draft's):
   **per-node r=3 ∪ every empty of every opponent-active window (count ≥ 1)**.
   Measured size ≈ 147 cells vs ≈ 302 legal (the window term is
   line-structured, adding only ~24 cells over r=3). Provable coverage — call
   it the *immediate-tactics lemma*: this set contains every defender move
   with immediate tactical effect:
   - all hits and win-now completions (Theorem 1, ⊆ r=2);
   - every block or pre-block of *any existing* attacker structure, at any
     planning depth — by definition of the window term (a pre-block only has
     value against windows the attacker has touched; virgin windows enter the
     set at the node after he touches them). This closes G1 outright.
   - every immediate own counter-threat and every immediate own **fork**: a
     fork cell has count-3 in each forked window, hence lies within distance
     3 of own stones (Lemma 0). This closes G3.
   What provably remains outside are only **≥ 2-tempo quiet plans**: G2
   virgin seeding, and own-line *shaping* (e.g. a distance-4 setup stone in
   an own count-2 window arranging next turn's fork — no counterexample
   constructed yet, but after G3, absence of construction is not evidence of
   absence). Those are exactly the domain of the certificate tempo term (§8),
   so this set is the natural *static* part of the repaired zone.
3. **Trust boundary:** until the repaired zone theorem is proven in the
   Stage-3 spec, a "proof" found under trimmed defender sets is **not** a
   proof. Either the verifier re-checks every AND node against the full
   legal set (with λ¹ instant dispatch doing the bulk for free — feasible for
   the fully-forcing F=0 regime, which needs no zone at all), or the result is
   consumed only as ordering/UNKNOWN-tier signal (§2 of the main plan). This
   keeps the plan's soundness contract intact with zero new trust
   assumptions.
4. Stage-0 addendum: log zone-vs-legal set sizes and the F/H_W distributions
   at `k < B` nodes so Stage 3 sizes node caps from measured fan-outs.

## 11. Codex adversarial review — verdict table (ultra pass, 2026-07-13)

| # | Claim | Verdict |
|---|---|---|
| 1 | Lemma 0 packing bound | CONFIRMED (tight) |
| 2 | Theorem 1 (tactics ⊆ r2) | CONFIRMED, current-node scope |
| 3 | Theorem 2 (threat creation ⊆ r3, tight) | CONFIRMED |
| 4 | Theorem 3 (boundary blocking) | CONFIRMED with ownership/terminal qualifications |
| 5 | Draft Theorem 4 (adjacent completion) + counter-threat corollary | REFUTED (early-terminality + legality-frontier order holes; fork corollary false) → weakened to Lemma 4; G3 added |
| 6 | Draft §7.1 junction counterexample | REFUTED as written (outward extensions beat j; origin-reachability nit) → repaired with arm caps; solver-confirmed |
| 7 | Zone theorem F-accounting | REFUTED (forks force without completing; forced hits can feed a window) → repair outlined §8, unproven |
| 8 | `6 − F` completion arithmetic / no same-turn tie | CONFIRMED (strict per-placement indices) |

Overall (Codex): "move-set trimming may remain a search heuristic only; the
current zone check cannot certify its omissions" — adopted as §10.

## 12. Open items

- Prove the repaired zone theorem (fork term, per-window H_W, DAG induction,
  strict ply indices) inside the Stage-3 proof-carrying spec; until then §10.3
  is the law.
- Dead-cell rule: sound as stated for inertness (a stone in only two-coloured
  windows joins and blocks nothing); the legality-frontier side effect
  (`LEGAL_RADIUS` reach through a dead stone) needs the same formal treatment
  before dead cells are *dropped* rather than deprioritized.
- Promote `scripts/_tss_moveset_zone_experiments.py` into the test tree with
  the Stage-3 work; re-run B1 at scale as a CI property test once the solver
  exists.
