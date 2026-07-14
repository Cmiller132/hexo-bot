# Soundness of defender move-set restriction in Hexo threat search — formal proofs

> **Provenance.** 2026-07-13. The proof-layer companion to
> `PLAN_TSS_MOVESET_ZONES.md` (results survey) and the intended normative
> reference for the Stage-3 solver/verifier build (`PLAN_TSS_DEEPENING.md` §6).
> Produced by an adversarial workflow: Claude drafted the framework and core
> proofs; Codex `gpt-5.6-sol` ultra passes independently derived the
> potential-function and domination components (§10, §11) and adversarially
> reviewed the whole; the mini-model solver (validated against `hexo_engine`)
> machine-checked the finite claims. Review log in §13.
> Round 5 revised the normative statements on 2026-07-14 to adopt all twelve
> reviewed tightenings; the Round 6 hostile confirmation pass ruled every
> adoption applied correctly (§13).
>
> **Status legend.** Every numbered claim carries one of:
> **[PROVEN]** complete proof below; **[PROVEN-MECH]** proof relies on a
> stated finite case enumeration that has been machine-executed;
> **[CONDITIONAL]** proven modulo an explicitly named claim; **[CONJ]**
> conjecture, obstruction stated.

---

## 0. Scope and the theorem being built

The df-pn solver's AND (defender) nodes must consider, in principle, every
legal reply (every empty within hex distance 8 of any stone — hundreds of
cells). The goal is a *certified* restriction: a set Z of replies per node
such that dismissing every reply outside Z is sound — each dismissal carries
a proof — parameterized by the horizon n (number of placements) of the
certificate being verified. Unbounded static sufficiency is impossible
(§9 of the survey doc, gap G2), so all results are horizon-parameterized;
this is a feature: certificates know their own horizon.

Everything below concerns positions after the Opening placement (the origin
is occupied; budgets are 2/1 per D4).

---

## 1. The game, formally

**D1 (board, cells, distance).** Cells are the integer axial lattice
ℤ² ∋ x = (q, r). Hex distance d(x, y) = max(|Δq|, |Δr|, |Δq + Δr|).

**D2 (axes and windows).** Axis vectors 𝔞 ∈ {(1,0), (0,1), (1,−1)}. A
*window* is a set W = {s + i·𝔞 : 0 ≤ i ≤ 5} for a start cell s and axis 𝔞.
Every window has exactly 6 cells; every cell lies in exactly 18 windows
(3 axes × 6 offsets). Two distinct cells of a window differ by a nonzero
multiple of its axis vector; hence **any two cells contained in a common
window are collinear along that window's axis, at distance ≤ 5** (fact F1).

**D3 (positions).** A position is P = (σ, m, b) with σ a finite partial map
cells ⇀ {A, D} (stones; permanent), m ∈ {A, D} the mover, b ∈ {1, 2} the
mover's remaining placements this turn. Stones(P) = dom σ. For a window W:
cntₓ(W, P) = |{c ∈ W : σ(c) = X}|.

**D4 (legality and transitions).** Cell c is *legal* in P iff c ∉ dom σ and
∃ s ∈ dom σ: d(c, s) ≤ 8. The mover places on a legal cell c, giving
σ' = σ ∪ {c ↦ m}. If some window W has cnt_m(W) = 6 after the placement, the
game **terminates immediately** with m the winner (per-placement checking —
wins can occur on the first of a turn's two placements). Otherwise the next
position is (σ', m, 1) if b = 2, else (σ', m̄, 2).

**D5 (window states).** W is *alive for X* iff cnt_X̄(W) = 0 (all-empty
windows are alive for both). W is *dead* iff cnt_A(W) ≥ 1 and cnt_D(W) ≥ 1.
W is an *X-threat* iff alive for X and cnt_X(W) ≥ 4.

**D6 (λ¹ predicates, as implemented).** For mover m with budget b:
`own_win_now(P)` iff ∃ m-alive W with cnt_m(W) = 5, or cnt_m(W) = 4 ∧ b = 2.
`hit(P)` = the multiset of empty-cell sets of the m̄-threat windows.
`mhs(P)` = minimum number of cells meeting every set in `hit(P)` (exhaustive
for values ≤ 2). λ¹ verdict: WIN if `own_win_now`; LOSS if ¬own_win_now and
mhs > b; else undecided. (Source: `threats_shared.rs`; soundness of these
verdicts post-opening is established in the main plan and assumed here.)

**D7 (plies and horizon).** A *ply* is one placement. Ply indices are
absolute along a play. For a position with mover m, the *defender-placement
count before ply T*, written 𝔇(P, T), is the number of plies strictly
before T at which D places, determined by (m, b) and alternation.

**D8 (dominance direction).** Outcomes are evaluated for the attacker A:
WIN ≻ nonWIN. "Sound dismissal" of defender reply c at node N means: if the
certificate proves A wins against all searched replies, then A wins against
c as well.

---

## 2. Elementary geometry [all PROVEN]

**L1 (packing bound).** In a window with k ≥ 1 stones of one colour and no
stones of the other, every empty cell of the window is within distance 6 − k
of a same-window stone; the bound is tight.
*Proof.* Index cells 0..5 along the axis. If empty e has nearest in-window
stone at (axis-)distance δ, the δ cells strictly between (plus e) are empty
and in the window, so δ ≤ #empties = 6 − k. By F1 axis-distance equals hex
distance here. Tightness: stones at offsets 6−k..5, e at 0. ∎

**L2 (boundary pair).** Any window containing ≥ 1 stone and ≥ 1 empty has an
empty at distance 1 from a same-window stone. *Proof.* The 6 cells are
consecutive on a line; a nonempty proper subset has a boundary. ∎

**T1 (single-turn tactical locality).** Every placement that completes a win
within the current turn, and every empty of every threat window, lies within
distance 2 of an existing stone. *Proof.* Completion within ≤ 2 placements
requires a window with ≥ 4 own stones already; L1 with k = 4. Threat windows
have k ≥ 4 by D5. ∎

**T2 (threat-creation locality).** A placement creating an X-threat lies
within distance 3 of an X stone; tight. *Proof.* The window held ≥ 3 X
stones pre-placement; L1 with k = 3. Tightness: stones at offsets 3,4,5,
placement at 0. ∎

---

## 3. The channel decomposition [PROVEN]

**L3 (difference accounting).** Let P' = P ± one stone at cell x. Then every
game-relevant predicate difference between P and P' is confined to:
(C1) *occupancy* of x (availability of x as a move; ownership of x);
(C2) the masks of the **18 windows containing x** (all window-mask–derived
predicates: win, alive/dead/threat, own_win_now, hit, mhs);
(C3) the *legality frontier*: the set {c : d(c, x) ≤ 8} ∖ dom σ gains or
loses legality justification through x.
*Proof.* By D4 the transition relation is a function of (dom σ at the placed
cell, the legality predicate, the window masks for termination); by D5–D6
every tactical predicate is a function of window masks; by D4 legality is a
function of stone positions; window masks change only for windows containing
x; legality justifications change only within distance 8 of x. There are no
other state components. ∎

L3 is the discipline that every transfer argument below obeys: an argument
is complete exactly when it has accounted for C1, C2, C3. (The survey doc
§6 records how arguments that skipped C3 or per-placement termination were
refuted; the channel list is mechanical, read off D4–D6, not a judgment.)

---

## 4. Permanence, transfer, and counting lemmas

**L4 (permanence). [PROVEN]** Window masks are monotone nondecreasing along
any play; dead windows remain dead; a window alive for X at ply t is alive
for X at every earlier ply. *Proof.* Stones are never removed (D3/D4). ∎

**L5 (inert-extension transfer). [PROVEN — restated per review R1]**
Let P' = P + X where X is a finite set of *defender* stones, and let 𝒲 be a
set of windows, C a set of cells, with (i) no x ∈ X lies in any W ∈ 𝒲,
(ii) X ∩ C = ∅. Then:
(a) every cell of C empty in P is empty in P';
(b) every W ∈ 𝒲 has identical masks in P and P';
(c) **Legal(P) ∖ X ⊆ Legal(P')** (legality justification is monotone in
    stones; occupancy removes exactly the X-cells); in particular every
    cell of C legal in P is legal in P' by (ii);
(d) **A-complete windows of P' are exactly the A-complete windows of P that
    are X-free.** (A-complete requires cnt_D = 0, so windows meeting X are
    A-blocked in P'; on X-free windows masks agree.) The direction used
    below is always: an A-completion event on a window known to be X-free
    transfers from P to P', and every A-completion in P' was available in P.
*Proof.* Each item from L3's channels: (a) C1+(ii); (b) C2+(i); (c) C3
monotone plus C1; (d) as computed. ∎

**L6 (deletion monotonicity + WF-legality). [PROVEN]** Let P″ = P − Y where
Y is a finite set of *defender* stones of P. Then:
(a) every A-threat window of P is an A-threat window of P″ (deleting D
    stones can only revive A-windows), so mhs(P″) ≥ mhs(P) for the defender
    and λ¹ LOSS verdicts for the defender transfer from P to P″;
(b) A-complete windows of P are A-complete in P″;
(c) a cell empty in P is empty or *newly empty* in P″ (never newly
    occupied);
(d) legality may **shrink**. Therefore deletion arguments require:
**WF-legality:** every attacker placement relied upon is within distance 8
of a stone that is not in Y (in our use: within distance 3 of an *attacker*
stone — automatic for threat-creating moves by T2 — or a root stone).
*Proof.* (a),(b): masks lose only D counts. (c): D3. (d): D4; the stated
condition restores C3 for the relied-upon cells. ∎

**L7 (completion counting). [PROVEN]** If window W has cnt_D(W, P) = s and
the defender makes at most D placements at plies < T, then in no play does
W become D-complete before ply T unless s + D ≥ 6. *Proof.* D-completion
requires 6 D-stones in W; each is either present at P (≤ s) or placed by one
of the < T defender plies (≤ D). ∎

---

## 5. Zone-carrying certificates

**D9 (certificate, machine-checkable form — round-5 form).** A
*certificate* 𝒞 for "A wins from P₀" is a finite rooted tree of nodes, each
labelled by an exact D3 position and mover/budget. The root position is P₀
and is nonterminal. Every node has a path-derived ply clock (root ply plus
depth), recomputed by the verifier; "leaf-ply" is the index of the last
placement on the path to a leaf. Every edge is the exact D4-successor of its
labelled placement, every placement is legal, and no edge is
defender-terminal. Every maximal node has one of the following types.

- **OR-COMPLETION leaf** (A to place): one designated placement, its
  A-complete witness window, and its completion ply.
- **WIN leaf** (A to place): a verifier-checked `own_win_now` position,
  with named A-alive count-5 witness windows (any budget) or count-4
  witnesses at budget 2, and a resolution within the current attacker turn.
- **LOSS leaf** (D to place, budget b): a named A-threat family 𝒯 whose
  empty-set family has transversal number τ(𝒯) > b, together with the
  verifier-checked defender `¬own_win_now` fact. For every complete
  nonterminal remainder H of the defender turn (exactly b placements unless
  the game ends earlier), some W_H ∈ 𝒯 has E(W_H,P_L) ∩ H = ∅, and A then
  fills W_H's at most two empties during the following turn. The declared
  worst-case resolution is leaf-ply + b + 2. By L13 the verifier may require
  |𝒯| ≤ 3 for b = 1 and |𝒯| ≤ 6 for b = 2.

An internal **OR node** has one designated A-placement and its exact child.
An internal **AND node** N has a nonempty searched set S(N) of legal replies
and one exact child per reply; all other real-legal replies carry the
dismissal claim. The verifier retains the internal defender
`¬own_win_now` check as a diagnostic. Under D11's completion zone it is
derivable (L14), rather than an independent hypothesis of T3; it remains a
mandatory LOSS-leaf check and an explicit T6 premise. Every terminal
resolution must be finite. The global horizon T is the maximum declared
resolution over all root-to-leaf paths. D18/T10 extend this grammar to
finite consistently labelled certificate DAGs.

**D14 (admissible local defender budget).** The exact local budget is

  B(OR-COMPLETION) = B(WIN) = 0;       B(LOSS,b) = b;
  B(OR) = B(child);                    B(AND) = 1 + max_C B(C).

Its unit is defender placements: the two placements of a D turn are two D4
edges, and a LOSS leaf contributes its remaining b placements. Exact maxima
are optimal for pruning but are not required. A verifier may accept any
nonnegative integral labelling B satisfying B(L) ≥ b at every LOSS leaf,
B(N) ≥ B(C) on every OR edge, and B(N) ≥ 1 + B(C) on every AND edge.
The old global quantity 𝔇(P_N,T) is the always-admissible special case and
is generally looser.

**D10 (compressed obligations; replaces full-window core).** Write
E(W,P) := W ∖ Stones(P). A *live obligation role* below N is either:

1. a future designated certificate attacker placement, expressly including
   an OR-COMPLETION placement; or
2. a role (L,W,y), for a named WIN/LOSS witness W at leaf L and a cell
   y ∈ E(W,P_L). These witness-empty roles also represent every possible
   leaf continuation placement, including the adaptive LOSS choice.

Let Ω(N) be all such roles at reachable descendants of N and define

  Obl(𝒞,N) = Prot(N) := { y : some live role in Ω(N) is carried by y }.

A role remains live through its deadline check and is discharged immediately
afterward. Thus an OR-placement role is protected immediately before Step O,
and a witness-empty role is protected at leaf entry. Full witness windows
are not obligations. Reachable-descendant nesting gives Prot(M) ⊆ Prot(N)
whenever M descends from N.

**D15 (cell-specific deadlines).** Each role ρ has a deadline: its designated
A-placement, or leaf entry for a WIN/LOSS witness-empty role. At a node N
from which ρ is reachable, r_N(ρ) is an integral upper bound on defender
placements before that deadline. Exact ranks count the maximum number of
AND edges on a reachable path. The verifier checks r = 0 at the deadline,
r_N(ρ) ≥ r_C(ρ) across an OR edge, and r_N(ρ) ≥ 1 + r_C(ρ) across an AND
edge while the role remains live. For a cell with several roles,

  r_N(y) := max { r_N(ρ) : ρ ∈ Ω(N) is carried by y }.

The role is discharged only after its r = 0 deadline check. A seed band is
formed only at an internal AND node with r_N(y) ≥ 1; no negative radius is
formed at r = 0. In particular LOSS witness empties require protection only
through leaf entry: the adaptive contract controls the remaining b
placements. Defender-completion windows use D14 or D16 instead, never these
role ranks. The uniform admissible choice r_N(ρ) = B(N) before the deadline
recovers the coarser local-budget band.

**D16 (per-window defender exposure).** For each window W, E_Q^D(W) is the
maximum number of future defender placements before the certificate attacker
wins or first places in W. Its exact recurrence is

  E_Q^D(W) = 0                         at WIN/OR-COMPLETION;
  E_Q^D(W) = b                         at LOSS with defender budget b;
  E_Q^D(W) = 0                         at an OR whose move enters W;
  E_Q^D(W) = E_C^D(W)                 at any other ordinary OR;
  E_Q^D(W) = 1 + max_C E_C^D(W)       at an internal AND.

If W is already non-D-alive, set E_Q^D(W) = 0. "The attacker enters W"
means that W becomes permanently non-D-alive; it need not become D5-dead.
Permanence (L4) is the needed property. Each overlapping window has its own
clock. The designated attacker-in-W stopping move is a D10 obligation, so
the obligation zone separately keeps that stop playable in the real game.

**L10 (short-range attacker-obligation containment). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* In a certificate whose attacker
placements are all threat-creating, an attacker placement that is A's k-th
future placement from N lies, for k ≤ 3, in a window containing at least
3 − (k − 1) ≥ 1 current attacker stones of P_N. Thus its D10 role lies in an
A-touched window. *Proof.* Immediately before the placement its window has
at least three A-stones. At most k − 1 are earlier future placements, leaving
at least 3 − (k − 1) current stones. ∎ From the fourth future placement on,
setup cells can lie in currently virgin windows and must be supplied by the
certificate obligation set.

**L11 (local-clock and nesting facts). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* If a descendant M is reached from N
after k defender placements, then:

  B(M) + k ≤ B(N),
  cnt_D(W,P_M) + B(M) ≤ cnt_D(W,P_N) + B(N),

and every selected path below N, including a LOSS remainder, contains at
most B(N) defender placements before its declared A-resolution. For every
still-live role ρ, r_M(ρ) + k ≤ r_N(ρ). Before the attacker enters W,

  E_M^D(W) + k ≤ E_N^D(W),
  cnt_D(W,P_M) + E_M^D(W) ≤ cnt_D(W,P_N) + E_N^D(W).

Also E_N^D(W) ≤ B(N), every exact role rank r*_N(ρ) ≤ B(N), and
Prot(M) ⊆ Prot(N). *Proof.* Sum the D14 and D15 verifier inequalities
along the path; every AND edge contributes one and every OR edge zero. The
AND maximum covers whichever filler child is selected, and the LOSS lower
bound covers its remainder. A window gains at most one D-stone per counted
defender edge, giving the two count inequalities. The D16 recurrence gives
the exposure inequalities until its stated stop and is pointwise bounded by
the D14 recurrence. An exact role deadline occurs no later than its path's
resolution, giving r* ≤ B. (A verifier-supplied rank may conservatively be
larger.) The last assertion is reachable-descendant set inclusion. ∎

For d(c,W) := min_{w∈W} d(c,w), define at every internal AND node N:

  Z_dir(N) = Prot(N) ∩ Legal(P_N);

  Z_seed(N) = ⋃ { Legal(P_N) ∩ B_{8(r_N(y)−1)}({y}) :
                  y ∈ Prot(N) ∖ (Legal(P_N) ∪ Stones(P_N)), r_N(y) ≥ 1 };

  Z_touch(N) = ⋃ { E(W,P_N) : W is D-alive at P_N,
                    cnt_D(W,P_N) ≥ 1,
                    cnt_D(W,P_N) + E_N^D(W) ≥ 6 };

  Z_virgin(N) = { c ∈ Legal(P_N) : some all-empty window W has
                    E_N^D(W) ≥ 6 and d(c,W) ≤ 8(E_N^D(W)−6) }.

Here B_r(U) = {c : d(c,u) ≤ r for some u ∈ U}. Every Z_touch cell is legal:
L1 puts every empty of its touched window within distance at most 5 of an
in-window D-stone.

**D11 (zone-carrying certificate — round-5 form).** A D9 certificate is
*zone-carrying* when, at every internal AND node, S(N) is independently
nonempty and the following verifier clauses hold:

- **(Z2) direct and completion protection:**
  S(N) ⊇ Z_dir(N) ∪ Z_touch(N) ∪ Z_virgin(N).
- **(Z4) WF-legality:** every certificate attacker placement, including
  placements allowed by a leaf contract, is within distance 8 of an
  attacker stone in its predecessor position or a root stone unaffected by
  the coupling. L1 supplies this for WIN/LOSS continuations.
- **(Z5′) ranked obligation seed guard:** S(N) ⊇ Z_seed(N).

This is the verifier-label mapping used below: the live labels are (Z2),
(Z4), and (Z5′). With the uniform choice r_N(y) = B(N), (Z5′) is

  S(N) ⊇ Legal(P_N) ∩ B_{8(B(N)−1)}
    ( Prot(N) ∖ (Legal(P_N) ∪ Stones(P_N)) ).

The former Z1 hitting clause is not a T3/T4 requirement. Current hitting
cells remain a sensible search heuristic, and T6's kernel regime still uses
the current threat family. All four zone components are finite and
verifier-enumerable: Legal(P_N) and Prot(N) are finite; touched windows are
finite; and the virgin query can be inverted from each legal candidate over
the finite radius allowed by E_N^D(W) ≤ B(N).

**L9′ (first protected occupation; replaces former L9). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* The replacement is necessary
because radius 8(B−1) prevents first protected *occupation*, not the stronger
claim that a protected cell never becomes real-legal.

Suppose direct legal protected cells are searched and every ghost-legal
dismissal x at N lies outside radius 8(B(N)−1) of

  Prot(N) ∖ (Legal(P_N) ∪ Stones(P_N)).

Then no defender placement creates a real-only stone in the current
protected set. The ranked form replaces B(N) by the maximum live role rank
r_N(y) for each target y and reaches the same conclusion through that role's
deadline.

*Proof.* Assume the first violation is the defender placement at y. If y is
ghost-legal, then y ∈ Z_dir and (Z2) makes it searched, so it is placed in
both games and is not real-only. Otherwise trace y's real-only legality
witness backward through current real-only defender stones to the first
ghost-legal dismissed seed x₀. If y is the p-th defender placement from x₀,
counting x₀, there are at most p−1 radius-8 links, so

  d(x₀,y) ≤ 8(p−1).

Protection nesting puts y ∈ Prot(N_{x₀}). At the violation y is still
ghost-empty and ghost-illegal; ghost legality for an unoccupied cell is
monotone, so it was ghost-illegal at N_{x₀}, and permanence makes it a
non-stone there. D14 anchor coverage gives p ≤ B(N_{x₀}); in the ranked
form, the still-live role gives p ≤ r_{N_{x₀}}(y). Thus x₀ lay in the
applicable (Z5′) band, contradicting its dismissal. ∎

The uniform radius is sharp at this level: a ghost-legal x₀ may be followed
by B−1 successive distance-8 placements, with the protected target last.

**D12 (coupling; round-5 form).** T3 couples a real position R to a walk on
𝒞 with ghost position G: equal ply, mover, and budget; identical attacker
stones; and canonical defender differences

  X := Stones_D(R) ∖ Stones_D(G),   Y := Stones_D(G) ∖ Stones_D(R).

The maintained invariants are exactly: (i) every x ∈ X entered at a
dismissal at which (Z2)/L9′ or the selected D17 envelope certified x outside
its live obligations; (ii) every y ∈ Y entered as a searched ghost filler;
and (iii) X ∩ Prot(N) = ∅ at every visited node.
L9′ supplies (iii) for a ghost-illegal dismissal; reachable-obligation
nesting preserves it after descent. A history X̂ ⊇ X contains every cell
ever added to X and never deletes cells. For every window W the canonical
identity

  (MI)  cnt_D(W,R) = cnt_D(W,G) + |X ∩ W| − |Y ∩ W|
                  ≤ cnt_D(W,G) + |X̂ ∩ W|

holds without any witness-window agreement assumption.

**L12 (split completion safety). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* In D12's coupling, under D11's
(Z2), (Z4), and (Z5′), before the mapped certificate attacker wins or first
enters a fixed window W, no real defender play completes W.

*Proof.* Suppose W is first completed in the real game before that stop. It
is D-alive throughout the relevant ghost prefix by shared A-stones and L4.
If no W-cell was ever dismissed, X̂ ∩ W = ∅ and (MI) forces the ghost line to
have at least the real D-count. It therefore contains a defender completion,
contrary to D9's exact successors and ban on defender-terminal edges.

Otherwise let x ∈ W be the first ever-dismissed real-only W-fill, at N.
If the ghost already has a D-stone in W, every W-empty is ghost-legal. Before
x, (MI) gives the real W-count at most the ghost count. The real placements
from x through completion are covered by E_N^D(W), so
cnt_D(W,P_N) + E_N^D(W) ≥ 6; Z_touch searched x, a contradiction.

If the ghost W is virgin and x is ghost-legal, all six W-fills remain in the
exposure count. Hence E_N^D(W) ≥ 6 and d(x,W) = 0, so Z_virgin searched x.
If x is ghost-illegal, trace its X-legality chain to the first ghost-legal
dismissed seed x₀. With j radius-8 links before the chain reaches W,

  d(x₀,W) ≤ 8j,                 E_{N_{x₀}}^D(W) ≥ j + 6.

The second inequality counts the seed/chain tempo and all six W-fills before
the exposure stop. Thus x₀ ∈ Z_virgin, again a contradiction. A searched
fill that touches W moves later nodes to the first case; independent seeds,
interleaving, and overlapping candidate windows are covered by the causal
chain for the first dismissed fill of each W. The D10 obligation for the
attacker-in-W move makes the exposure stop valid in the real game.

At a LOSS remainder the coupling stops at leaf entry. If W has an
ever-dismissed fill, the same first-dismissal count remains valid because
D16 includes the leaf's b placements. If it has none, leaf-time (MI) gives
the real count at most the ghost count; the mandatory leaf
`¬own_win_now` check bounds the latter by 3 at b = 2 and 4 at b = 1, so the
remaining b placements reach at most 5. Thus no defender completion was
lost when the coupling stopped. ∎

The virgin radius is sharp at E = 7: a legal seed at distance 8 may be
followed by the first W-fill and the five remaining fills.

---

## 6. The main theorem

**T3 (local-clock, pathwise dismissal soundness). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* Let 𝒞 obey D9–D12 and D14–D16 and be
zone-carrying under (Z2), (Z4), and (Z5′). For every real defender play
against the compiled attacker strategy, either the real attacker wins
strictly earlier than the mapped ghost resolution, or the play maps to a
finite certificate path and A completes by that path's declared resolution.
Consequently A wins by the global maximum T, and every certified dismissal
is sound.

*Proof.* Couple the real position R, initially P₀, to a ghost walk G on 𝒞,
maintaining D12. Process a typed leaf immediately upon reaching it.

**Step O (OR node, designated placement c).** The live placement role puts
c in Prot at every current ancestor. Hence c ∉ X by D12(iii); ghost
emptiness gives c ∉ Y. It is real-empty. Clause (Z4) supplies a shared
attacker/root legality witness, so c is real-legal. Play c in both games;
the A-stone sets remain identical.

At an OR-COMPLETION, the other five cells of its named window are shared
A-stones immediately before c. The ghost window contains no D-stone, hence
no Y-cell, and c is not an X-cell. The same placement therefore completes
the real window at the declared ply. If instead the real placement completes
some other window that the ghost does not, L5(d)/L6(b) make that an immediate
strictly earlier real A-win; exit. Otherwise discharge c's role after its
deadline check and descend to the exact child.

**Step A (AND node N, real defender placement d).** Exactly one case holds:
d is ghost-occupied; d is ghost-empty and searched; or d is ghost-empty and
unsearched.

*Filler subroutine.* Whenever the ghost must consume a defender ply without
copying d, choose any d₀ ∈ S(N), which exists independently by D9. It is
ghost-empty and ghost-legal. If d₀ ∈ X, ghost placement removes it from X;
otherwise it adds d₀ to Y. Recompute the canonical differences. A new
Y-cell is a searched filler and X only shrinks.

*(A1) d ghost-empty and d ∈ S(N).* Both games play d and descend to its
exact child; X and Y do not change.

*(A2) d ghost-occupied.* Since attacker stones are shared and d is
real-empty, d ∈ Y. Real placement removes d from Y. Run the filler
subroutine and descend to the filler child.

*(A3) d ghost-empty and d ∉ S(N).* If d is ghost-legal and protected, then
d ∈ Z_dir ⊆ S(N), contradiction; thus a ghost-legal dismissal is outside
Prot and, because it is outside Z_seed, satisfies the seed hypothesis of
L9′. If d is ghost-illegal, L9′ says that placing it cannot create the first
real-only protected stone; hence it too is outside the current Prot. Real
plays d; add d to X and X̂. Run the filler subroutine and descend. L9′ and
protection nesting preserve D12(iii). This completes the C1/C3 accounting;
L12 supplies C2 by excluding a real defender completion before the mapped
attacker stop.

**WIN leaf.** Let W be a named A-alive witness. Every ghost A-stone of W is
shared. Every other W-cell is in E(W,P_L), hence has a live witness-empty
role and is X-free at leaf entry. A-aliveness gives no ghost D-stone in W,
so W has no Y-cell. Thus the complete real and ghost masks and empty sets
agree although only leaf empties were protected. L1 puts the one or two
empties within distance 2 of shared A-stones. They are real-legal and remain
empty, and A fills them with no intervening D-placement, completing by the
leaf's declared resolution.

**LOSS leaf.** The same compressed-obligation argument makes the real and
ghost masks and empty sets identical for every W ∈ 𝒯 at leaf entry; hence
the real transversal number is the checked value τ(𝒯) > b. A real defender
`own_win_now` is impossible. If its putative window has no ever-dismissed
fill, leaf-time (MI), the shared A-mask, and equal budget transfer it to the
ghost, contradicting the mandatory leaf check. If it has an ever-dismissed
fill, the L12 first-dismissal/exposure argument includes the LOSS leaf's b
placements, so completing within the current D turn is impossible.

Now discharge the leaf roles and let the real defender choose any complete
nonterminal remainder H. Since |H| ≤ b < τ(𝒯), some W_H ∈ 𝒯 has
E(W_H,P_L) ∩ H = ∅. It remains A-alive. Its at most two surviving empties
are within distance 2 of shared permanent A-stones, so A completes W_H in
the following turn, by leaf-ply + b + 2. Protection is intentionally not
needed after leaf entry; the adaptive quantifier over H supplies the
surviving witness.

Every nonterminal step descends the finite certificate. Thus, absent an
earlier real A-win, the coupled play reaches one of the preceding terminal
transfers and resolves by that selected path's declared ply. T is their
maximum. ∎

**T4 (ranked zone reading of T3). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* At an ordinary AND node the
normative verifier zone is

  Z(N) = Z_dir(N) ∪ Z_seed(N) ∪ Z_touch(N) ∪ Z_virgin(N).

Any independently nonempty searched set S(N) ⊇ Z(N), together with (Z4),
is sufficient. If Z(N) is empty, D9 still requires one arbitrary legal
searched fallback. The uniform admissible deadline r_N(y) = B(N) replaces
Z_seed by the radius-8(B(N)−1) band displayed in D11. Exact cell ranks can
only reduce that uniform obligation band; exact window exposures likewise
reduce the corresponding B-clock completion guards. Larger admissible upper
bounds remain sound. Current hitting cells
are optional search candidates, not a T3/T4 term; T6 below uses them only to
define its separate kernel.

**L13 (sparse LOSS witnesses). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* A D9 LOSS witness can be chosen
with |𝒯| ≤ 3 at b = 1 and |𝒯| ≤ 6 at b = 2. Here b counts placements.

*Proof.* Every threat empty-set has size one or two. For b = 1, if the
family contains {a}, choose one set missing a; those two have no one-point
transversal. Otherwise choose E = {a,b}, one set missing a, and one missing
b. Any point hitting E is a or b and misses the corresponding selected set.
The three edges of a triangle show sharpness.

For b = 2, take a maximal pairwise-disjoint subfamily. It cannot have size
one, because its sole set, of size at most two, would hit the whole family.
If it has size at least three, three disjoint sets suffice. Otherwise take
the two disjoint sets E₁,E₂. Their two-point transversals are the at most
four cross-pairs selecting one point from each. For each cross-pair select
one original set that it misses. Together with E₁,E₂, at most six sets
exclude every two-point transversal. The six edges of K₄ show the general
rank-two bound is sharp. Since every real LOSS remainder H has |H| ≤ b, a
selected family with τ > b retains a witness disjoint from H. ∎

**T5 (static-set coverage for short local budgets). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* Suppose B(N) ≤ 3, at most three
remaining attacker placements are threat-creating, all named witness-empty
roles lie in A-touched windows, and Z_seed(N) is empty. Then

  r3 ∪ {empties of A-touched alive windows},

where r3 is the set of legal empties within distance 3 of any stone, covers
Z(N). *Proof.* E_N^D(W) ≤ B(N) ≤ 3, so Z_virgin is empty. A touched window
in Z_touch has cnt_D ≥ 6−E ≥ 3, and L1 puts all its empties within distance
3 of a D-stone. L10 covers the stated attacker roles, and the witness-role
hypothesis covers the rest of Z_dir. The seed term is empty by hypothesis.
∎ Without all joint hypotheses this static set is a heuristic candidate
generator, not a certified zone.

For an A-threat empty-set family ℱ_N, write

  ℱ_N ∖ d := { E ∈ ℱ_N : d ∉ E },
  K_b(N) := { d ∈ Legal(P_N) : τ(ℱ_N ∖ d) ≤ b−1 }.

**T6 (extendable-hit kernel; forced-tree regime). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* A kernel-governed region is entered
at a certificate node with real and ghost positions equal and uses the
following rule until a typed terminal or an mhs > b handoff. At every
internal AND node
in that region require the explicit defender `¬own_win_now` check and
mhs(P_N) = τ(ℱ_N) ≤ b, and search exactly K_b(N). No original
obligation/core term is required. If mhs < b, K_b(N) is all of Legal(P_N)
and there is no pruning. If mhs = b, K_b weakly refines the current
hitting-cell set and can strictly reduce it; equality can occur. A node with
mhs > b ends the kernel region: it must be a valid LOSS leaf or hand off to
an existing nonempty sound subtree, below which the core-free kernel claim
is not asserted without a fresh equal-position entry.

*Proof.* When τ < b, deleting sets hit by any d cannot increase τ, so every
legal d is in K_b. When τ = b, follow a minimum transversal: its first cell
is legal by T1 and lies in K_b, so the kernel is nonempty. Before the first real reply
d ∉ K_b, real and certificate play agree exactly; thus X = Y = ∅. Abandon
the original subtree and use the residual current threat family directly.
For b = 2, the successor has defender budget 1 and residual τ > 1, a valid
adaptive LOSS refutation. For b = 1, a current A-threat survives and A now
has budget 2, giving a WIN refutation. No original future obligation is used.

The defender cannot complete first. The explicit `¬own_win_now` premise
bounds every D-alive window by count 3 at b = 2 and count 4 at b = 1. The
two or one remaining defender placements reach at most 5. For the same-T
claim, a minimum hitting line is followable through K_b: at b = 2 its first
member leaves a family hit by the second; at b = 1 its sole member hits all
current threats. At b = 2 the b = 1 child remains kernel-governed and
searches the second member; at b = 1 the sole common hit suffices. Thus the
two D-stones in the first case, or the one in the second, kill every current
A-threat. Defender stones create no new A-threat. In both budgets the first
following A-placement therefore cannot complete a window—such a window
would have been a current count-5 threat and was killed—so the relevant
original path's clock, and hence global T, reaches the second A-placement
needed by the auxiliary refutation.

Finally, if τ > b and d ∈ K_b, then d plus a residual transversal of size at
most b−1 would hit all of ℱ_N, contradicting τ > b. Hence K_b is empty in
that case and would violate D9's nonempty searched-set rule. This proves the
stated scope restriction. ∎

---

## 7. The n-relevance closure operator

**D13 (ranked relevance closure).** The certificate-relative mandatory
closure is

  R_cert(𝒞,N) := Z_dir(N) ∪ Z_seed(N) ∪ Z_touch(N) ∪ Z_virgin(N).

For search, the solver may use the finite superset

  R_search(𝒞,N) := R_cert(𝒞,N) ∪ hitting(P_N) ∪ 𝒜(P_N) ∪ r3(P_N),

where hitting(P) is the union of the empty-sets of current A-threats,
𝒜(P) is the set of empties in A-touched alive windows, and r3(P) is the
legal radius-3 neighborhood of current stones. The last three terms are
heuristics, not T3 hypotheses. All mandatory terms are finite: obligations
come from a finite certificate; touched windows from the finite position;
and for each finite legal candidate the virgin test examines only windows
within its bounded exposure radius.

**T7 (ranked-closure coverage). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* At every ordinary zone-governed AND
node, any independently nonempty S(N) ⊇ R_cert(𝒞,N) satisfies (Z2) and
(Z5′); dismissal soundness follows from T3 when D9, the D14–D16 clock
checks, and (Z4) also hold. In constructing R_cert, the verifier unions roles
over all reachable descendants, includes every designated A-move including
OR-COMPLETION, treats WIN/LOSS continuation cells as leaf-entry
witness-empty roles, takes the maximum rank over live roles sharing a cell,
and forms no band after the r = 0 deadline check. Defender-completion windows
remain on E^D (or the conservative B) clock. *Proof.* These requirements are
exactly the four components and three labels of D11; apply T4 and T3. ∎

**D17 (transition-inclusive substitution envelope).** An alternative to
the global reachable-descendant union may annotate every ghost-legal
dismissal d at N with a searched substitute

  φ_N(d) = s ∈ S(N),

whose exact child is C_s. A nonempty fallback F(N) ⊆ S(N) is declared
independently of these annotations; in particular, S(N) is not defined
circularly as "only replies with no safe substitute." The selected envelope
has the following verifier data and requirements.

1. Its transition budget is B̂(N,d,s) := 1 + B(C_s), including the current
   real/ghost defender ply and every reachable LOSS remainder.
2. Its obligations are the union of the roles at **all** reachable
   descendants of C_s, not one leaf or one selected continuation.
3. The real cell d itself avoids those obligation cells and every
   transition-dangerous completion empty.
4. For a child obligation y with rank r_{C_s}(y) and
   y ∉ Legal(P_N) ∪ Stones(P_N), the parent seed test is
   d(d,y) > 8r_{C_s}(y), equivalently radius 8((1+r_{C_s}(y))−1). Direct
   occupation d = y is separately forbidden.
5. With B̂, for every D-alive W with cnt_D(W,P_N) ≥ 1, the touched test
   forbids d ∈ E(W,P_N) whenever cnt_D(W,P_N) + 1 + B(C_s) ≥ 6; for every
   all-empty W, the virgin test forbids a seed with d(d,W) ≤ 8(B̂−6) when
   B̂ ≥ 6. The exact exposure form instead sets
   Ê^D(W) := 1 + E_{C_s}^D(W), uses cnt_D(W,P_N) + Ê^D(W) ≥ 6 for a touched
   W-fill, and uses d(d,W) ≤ 8(Ê^D(W)−6) with Ê^D(W) ≥ 6 for an all-empty W.
   The touched and virgin clauses are both mandatory for the selected clock.
6. F(N), and hence S(N), stays nonempty independently of whether any
   dismissal has a safe substitute.
7. For a ghost-legal A3 dismissal the ghost plays φ_N(d). A2 may use any
   searched filler because A2 creates no new X-stone. A ghost-illegal A3
   inherits the envelope of the earlier ghost-legal seed that supports its
   real legality and may use any searched filler inside the current reachable
   subtree; it does not start a child-only clock.
8. The envelope protects every reachable LOSS witness-empty role through
   leaf entry, while B̂/Ê^D count the leaf's b placements for
   defender-completion and defender-own-win exclusion.

The simpler default-child form declares f(N) ∈ F(N) and uses C_{f(N)} for
every ghost-legal dismissal, but it must satisfy the same transition-inclusive
tests; merely replacing a whole-tree obligation union by a child union is
insufficient.

**T9 (branch-indexed substitution soundness). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* T3 remains valid if an AND node's
global D11 dismissal tests are replaced, dismissal by dismissal, by a valid
D17 envelope. The same holds for the repaired default-child form.

*Proof.* Real d and ghost s consume the same single D4 defender ply, so the
clock and mover/budget remain synchronized. Condition 3 supplies C1 for the
new X-stone. Conditions 4 and 5 supply C3 and C2 with the current transition
included. Thereafter all ghost nodes lie among the reachable descendants of
C_s. A1 may take any searched edge inside that set. A2 adds no X-stone and
may take any filler. At a later ghost-legal A3, its new selected descendant
set is nested inside the earlier one; at a ghost-illegal A3, the causal
legality chain returns to an earlier tested ghost-legal seed and inherits
that envelope. Thus every earlier X-stone continues to avoid all current
roles, and L9′ applies with the transition rank. The exposure argument L12
applies with B̂ or Ê^D; condition 8 covers the post-leaf portion. Clause
(Z4), the compressed witness-mask transfer, and all terminal cases of T3 are
unchanged. Hence the coupled real play either wins earlier or resolves on
the selected reachable path. ∎

The `+1` in D17 is mandatory in both channels. For C3, let N have b = 2,
let legal d be distance 8 from a ghost-illegal future obligation y, and let
C_s have one defender placement before a legal, shared A-setup a supplies
(Z4) and makes designated y legal. Then r_{C_s}(y) = 1. A child-only radius
8(r−1) = 0 permits d; real d
legalizes y and the remaining defender placement occupies it. The parent
radius 8r = 8 forbids d. For C2, let a D-alive W have two D-stones at N,
with d ∈ W, s ∉ W, and B(C_s) = 3. The child-only test reads 2+3 = 5, but
real d plus three later W-fills completes W. The transition-inclusive test
reads 2+1+3 = 6 and forbids the dismissal.

**L14 (internal own-win diagnostic is redundant under the completion
zone). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* Under D11's completion-zone
requirement and D9's ban on defender-terminal edges, an internal AND node
cannot have a D-alive count-5 window, nor a count-4 window at b = 2.

*Proof.* A count-5 touched window has E_N^D(W) ≥ 1, so its legal last empty
lies in Z_touch and must be searched; its exact child is defender-terminal,
forbidden by D9. At b = 2, a count-4 touched window has exposure at least
two before A can move, so both empties are searched. After either first
fill, the b = 1 child has count 5. It cannot be a LOSS leaf because the
mandatory leaf check fails; if internal, its last empty is searched and
gives the same forbidden terminal edge. ∎ The internal check remains a
useful diagnostic. It remains logically necessary at LOSS leaves and an
explicit premise of T6, whose kernel has no completion guard.

**D18 (finite acyclic certificate DAG).** A D9 certificate may instead be a
finite rooted acyclic graph provided every shared node has one exact D9
label: the same position, mover/budget, designated OR action or AND
searched-successor map, leaf witness data, and one consistent path clock.
Obligations at a DAG node are unions over all reachable descendants. Local
budgets, ranks, and exposures satisfy their inequalities on every outgoing
edge. Coupling histories X, Y, X̂ remain path-local.

**T10 (DAG unfolding). [PROVEN]**
*(R5-adopted; round-6 confirmation recorded in §13.)* Every D18 DAG satisfying the
ordinary D11 rules or the D17 substitution rules has the corresponding T3
soundness conclusion.

*Proof.* A finite acyclic graph has finitely many root-to-node paths. Unfold
one copy of each shared node along each such path. The exact label and
consistent-clock conditions make the result a finite D9 tree with the same
legal transitions, terminal data, and local-clock inequalities. Reachability
is nested along every edge, so reachable-descendant obligation unions and
protection monotonicity are preserved. Apply T3 (or T9) to the unfolding;
its X/Y/X̂ histories are precisely the required path-local histories. ∎

The solver may therefore search with R_search, a smaller heuristic, or a
D17-aware generator. At verification it must either establish the ordinary
R_cert coverage plus (Z4), or validate every D17 annotation, while retaining
an independently nonempty searched fallback. Any failure yields UNKNOWN.
Search trimming affects completeness only; a verified WIN remains sound.

---

## 8. Impossibility of pairing strategies [PROVEN]

Pairing strategies — perfect local defenses via a precomputed matching —
would collapse defender sets to size ~1. They do not exist for this game:

**T8 (no pairing).** There is no partial matching M on cells (each cell in
at most one pair) such that every window contains both cells of some pair.

*Proof.* Discard from M every pair that is not contained in any window
(non-axis-aligned pairs and axis pairs at distance > 5) — a smaller
matching with the same covering property, WLOG. By F1 every remaining pair
is axis-aligned at axis-distance δ ∈ {1..5}. On a fixed line ℓ, a pair at
positions (p, p+δ) is contained in the windows starting at s ∈ [p+δ−5, p]:
exactly 6 − δ ≤ 5 window starts. To cover all integer starts, consecutive
pairs' left cells must satisfy p_{i+1} − p_i ≤ 6 − δ_{i+1} ≤ 5 (mixed δ's
cannot beat this), so a segment of ℓ with L cells has L − 5 internal window
starts and carries ≥ (L − 5)/5 pairs, i.e. ≥ 2(L − 5)/5 cells of ℓ matched
along ℓ's axis. Each matched cell serves exactly one axis — its own pair's.
Take B_R = {x : d(x, 0) ≤ R}, |B_R| = 3R² + 3R + 1. For each axis, B_R
meets exactly 2R + 1 line segments whose lengths sum to |B_R|; summing the
per-segment bound over segments and over the 3 axes:

  matched-cell incidences ≥ (6/5)|B_R| − 6(2R + 1).

A matching supplies at most |B_R| incidences. Already at R = 20:
(6/5)(1261) − 246 = 1267.2 > 1261. Contradiction. ∎

**Corollary T8.1 (translation-invariant line grids).** For k-in-a-row on a
point lattice with g pairwise non-parallel translation-invariant line
directions (each lattice line partitioned into length-k windows at all
offsets, balls with vanishing boundary-to-volume ratio — the Følner
property of ℤ²), the argument needs per-line matched density 2/(k − 1),
so a pairing requires g · 2/(k − 1) ≤ 1, i.e. **k ≥ 2g + 1**. Consistency:
the square grid (g = 4) gives k ≥ 9, matching the classical existence of
pairing draws for 9-in-a-row; the hex grid (g = 3) gives k ≥ 7, so
6-in-a-row misses by one. (Existence at the threshold requires a
construction and is not claimed; the corollary is stated for lattices as
above, not arbitrary "grids".)

Consequence for the program: the "pairing prize" of §8.2 of the survey doc
is definitively closed for global pairings. Partial pairings on constrained
regions are not ruled out but cannot be position-independent.

---

## 9. What the theorems deliver, quantitatively

- The exact ordinary verifier zone is the finite ranked union
  Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin. It has no mandatory current-hitting
  term. Cell roles use their own deadlines; touched windows have no frontier
  term; virgin windows use radius 8(E^D−6), which is radius 8 at E^D = 7.
- Fully forcing certificates (T6) search the extendable-hit kernel K_b and
  no original obligation term. Kernel-governed internal nodes require
  mhs ≤ b and the explicit `¬own_win_now` premise; mhs < b performs no
  pruning, while mhs = b can strictly reduce the hitting-cell set.
- LOSS leaves need at most 3 witness windows at b = 1 and 6 at b = 2.
  Pathwise clocks and finite DAG sharing do not enlarge a selected path's
  local budget; shared nodes retain one exact label and clock.
- P3 commutation (§11) additionally halves turn-level branching by
  unordered-pair deduplication, orthogonally to the zone.

The predecessor D-based search heuristic measured a mean 147 cells versus
302 legal cells on random midgame positions (survey doc §9). Those numbers
describe the conservative search generator, not the new certificate-relative
ranked verifier zone.

**Historical mechanized validation of the search heuristic, executed**
(script
`scripts/_tss_moveset_zone_experiments.py`, mini-model validated
move-for-move against `hexo_engine`): (i) the closure-restricted solver
does **not** claim the engineered junction position (the known-adversarial
G1 instance): the junction cell is in the closure and the defender survives
(1,396 nodes, uncapped); (ii) random bounded model check: 91 attackable
positions, 52 closure-restricted WIN claims, **0 divergences** against the
full legal move set at matched horizon (139 s); the earlier r=2 experiments
(survey doc §9) provide the negative controls — the same harness *does*
produce false WINs for unsound restrictions. This is search-behavior
evidence for the predecessor heuristic, not a machine proof of revised T7.

---

## 10. Potential-function layer (Erdős–Selfridge adaptation)

*Derived independently by Codex (ultra); full text with complete proofs in*
`docs/proof_parts/ES_POTENTIAL.md`. Summary rewritten against the source
file after review R1 caught the first summary misstating it (base, "iff",
premises — see §13). Statement numbers refer to that file.

- **Potential.** λ = **√3**; Φ(P) = Σ over **attacker-touched** alive
  windows (cnt_A ≥ 1, cnt_D = 0) of λ^{−(#empties)}. All-empty windows are
  deliberately excluded (their inclusion diverges on the infinite board).
  Base √3 is exactly what the (2:2) round inequality (L5 there) supports;
  base 2 fails (three separated count-4 targets have Σ2^{−2} = 3/4 < 1 yet
  the defender kills at most two per turn).
- **Theorem 1 [PROVEN] (fixed-family blocking — sufficiency).** From a
  nonterminal defender-**FirstStone** position, for any finite fixed family
  F of currently attacker-alive windows: Ψ_F < 1 ⇒ sequential greedy
  defense prevents completion of any member of F, forever. Sufficient only
  — necessity is false (one defender stone can kill 18 windows at once).
  The full-turn premise matters: with b = 1 the theorem fails (two
  separated count-4s, Ψ = 2/3 < 1, one kill, attacker completes the other).
- **Theorem 2 [PROVEN] (one-cycle certificate).** Defender at FirstStone
  with Φ < 1 ⇒ the attacker cannot win during his immediately following
  two placements (mid-turn win check included). Not a forever certificate:
  births can push Φ above 1 afterward.
- **Theorem 3 [PROVEN] (cumulative-birth forever certificate).** With the
  file's fixed tie-breaking/filler policy: if a **uniform bound over all
  attacker strategies** on the total enrolled birth mass exists,
  sup_τ Σ_t C_t(τ) ≤ B_∞, and Φ(P₀) + B_∞ < 1, greedy defense blocks
  forever; a finite-T variant certifies the first T attacker pairs. (A
  pathwise-observed Σ C_t is *not* a certificate — the quantifier is over
  strategies.)
- **Theorem 4 [PROVEN] (finite-region forever theorem).** If the
  **attacker** is forever confined to a finite region R (the confinement is
  on the attacker; defender fillers may lie outside R):
  Φ_R(P₀) + B*_R < 1 — or the coarser Φ_R + N_R/9 < 1 — ⇒ region-target
  greedy defense blocks R forever. All positive-reduction defender
  placements lie in attacker-alive windows of the region family.
- **Practical domain [PROVEN, narrow — stated honestly].** Φ < 1 forces the
  attacker's alive stone–window incidences down to ≤ 17 total; with N_A
  attacker stones, all but ≤ 17 of their 18·N_A incidences must already lie
  in defender-killed windows. The theorem's domain is heavily blocked
  endgame-like regions, not general middlegame. Cor. 2 gives an
  integer-only node check: with a = n₁+3n₃+9n₅, b = n₂+3n₄ over the
  count-profile bins, Φ < 1 ⇔ b ≤ 8 ∧ a² < 3(9−b)². Cost: O(#touched
  windows), not O(#threats).
- **Honest boundary [PROVEN counterexamples].** The raw global claim
  "Φ < 1 blocks the unrestricted infinite game forever" is **OPEN** —
  Beck's induction genuinely fails through window births (Cex. 1: two far
  double-placements birth 36 one-stone windows, Φ jumps to 4/√3 > 1);
  "no window with ≤ 2 empties" does not repair it; Φ = 1 does not suffice.
- **Move-set compatibility.** Every positive-reduction greedy placement
  lies in an attacker-touched alive window (L4 there), hence inside D13's
  A-touched heuristic/candidate term. Mandatory placements with no monitored window
  available are **fillers** — one or both placements of a defender turn may
  be fillers (Cex. 2 and the review R1 correction) — and the certified-node
  strategy must permit arbitrary legal fillers. Dismissal soundness (T3) is
  unaffected; this concerns only the potential-certified *strategy*.

---

## 11. Domination-pattern layer

*Derived independently by Codex (ultra); full text with complete proofs in*
`docs/proof_parts/DOMINATION.md`. Summary (statement numbers refer to that
file):

- **Framework.** Stopped-horizon outcomes (D3), n-outcome-domination (D5)
  with the pruning-direction characterization (L3), and *causal alternating
  simulation* certificates (D6, L4–L6) — strategy transfer with the three
  channels (occupancy, window masks, legality frontier) checked at every
  step; the failure modes that killed the naive reordering theorem are
  addressed structurally, not per-case.
- **Lemma 7 [PROVEN] (dead empties have no frontier).** A dead empty cell's
  18 dead windows already contain stones whose radius-8 balls cover the
  cell's own ball — so occupying a dead cell adds **no** new legal cells.
  This closes the legality-frontier caveat that kept dead-cell dismissal
  conditional in earlier drafts.
- **P1 [PROVEN] (dead-cell dismissal).** A defender reply on a dead cell is
  dominated by a certificate-named searched reply *a* provided *a* itself
  wins immediately or passes the frontier-inertness test B₈(a) ⊆ Λ(P).
  L7 makes the **dismissed** dead cell frontier-inert automatically; the
  substitute's test is a separate obligation on *a* and is not automatic
  (review R1 correction of this summary).
- **P2 [PROVEN] (interchangeable hitting cells).** Two empties of the same
  attacker threat window are interchangeable when their *other* incident
  windows are all dead and successor supports match ("dead-spoke"
  conditions); count-profile equality alone is proven insufficient
  (counterexample) — the naive version of this pattern is false, the
  dead-spoke version is proven.
- **P3 [PROVEN] (same-turn commutation).** The two placements of one turn
  commute when both cells are legal at turn start and neither singleton
  prefix wins — so the solver may deduplicate defender (and attacker) turn
  pairs as unordered sets under those tests, roughly halving turn-level
  branching. Immediate-win prefixes and newly-legalized continuations are
  excluded by explicit side conditions.
- **Machine-verification specs** (MV-P1/P2/P3): finite local encodings,
  D6/Burnside-quotiented configuration counts, and checkable predicates —
  specified by the derivation; two have since been **executed at
  randomized-spot-check grade** (full exhaustive quotient enumeration
  remains open): MV-P3: 2,997 random legal same-mover pairs under the
  singleton-nonwin filter — 0 commutation mismatches; MV-L7: 400
  *adversarially engineered* dead-cell configurations (deadening stones
  pushed to maximal offsets) — 0 legality-frontier violations, worst
  min-distance from any 8-ball cell to an existing stone = 6 (margin 2),
  consistent with the lemma's wedge geometry.
- **Nonclaims** are listed explicitly (e.g. stones of one colour are not
  globally monotone-helpful when frontiers differ) — deliberately mirroring
  the refutation history in the survey doc.

---

## 12. Open problems

1. **Sharpened budget (F + H_W).** B and E^D still count every defender
   placement before the relevant local resolution or exposure stop. The
   sharper debit — quiet placements F plus per-window forced-hit capacity
   H_W — requires branchwise worst-case bookkeeping; no theorem here proves
   that compulsory hits are unavailable for filling a chosen W. It remains
   open.
2. **Resolved — frontier tempo accounting.** D15's ranked obligation radius
   8(r−1) (uniformly 8(B−1)) and D16's touched/virgin split, including the
   sharp virgin radius 8(E^D−6), resolve the former band-sharpening problem.
3. **Pairing at the threshold.** T8.1 leaves k = 7, g = 3 open (density
   exactly 1); irrelevant to Hexo (k = 6) but of independent interest.
4. **Formalization.** All objects here are finite combinatorics; a
   Lean/Coq formalization of §§1–8 is feasible and would yield a verified
   reference checker for (Z2), (Z4), (Z5′), the B/E^D/r inequalities,
   transition envelopes, and DAG label/clock consistency.

---

## 13. Adversarial review log

**Round 1 (Codex ultra, hostile, 2026-07-13).** Confirmed: T8 (density
argument, ball averaging, mixed-δ handling — minor wording repairs applied);
DOMINATION Lemma 7 (wedge geometry verified independently); ES Lemma 5,
Theorem 3 accounting, Counterexample 1; L6(a)–(c); Z1 non-redundancy.
Refuted or gapped, all repaired in this revision:
1. **Fatal**: a dismissed stone's frontier extension could legalize a
   future witness-window cell unreachable (hence unsearched) in every
   certificate line → new (Z5) frontier guard; T3 case A3 now derives
   d ∉ Prot for frontier-reached cells.
2. D12 extension order permitted X ∩ Y ≠ ∅ and an under-specified filler
   update → replaced by canonical difference-based coupling with a
   normalizing filler subroutine.
3. Z3 as written demanded infinitely many illegal cells at D ≥ 6, and the
   "forces all legal cells" remark was false (cells in no D-alive window
   are exempt) → (Z2)/Prot now intersect with Legal; the D ≥ 6 boundary is
   stated exactly.
4. λ¹ leaves carried no machine-checkable witness data; attacker-WIN leaves
   were untreated → D9 strengthened (witness windows, continuation
   placements, resolution plies); WIN-leaf transfer added.
5. The pre-anchor "all-searched fills" step was asserted → the (MI) mask
   inclusion now carries it, with the ghost-line no-D-completion argument.
6. L5(c) false as stated (X-cells lose legality), L5(d) self-contradictory
   → restated.
7. §10 summary misstated ES: base 2 vs λ = √3, "iff" vs sufficiency,
   missing FirstStone premise, wrong quantifiers on Theorems 3/4, "one
   filler" and O(#threats) wrong → §10 rewritten from the source file.
8. §11 P1 summary overstated automatism of the substitute's frontier test
   → corrected.
9. D13 was infinite as written (A-alive included virgin windows) and the
   implementation's max(1, 6−D) clamp diverged from it at D ≥ 6 → D13
   restated (A-touched, ∩ Legal, D ≥ 6 fallback); implementation fixed to
   match; the repo script was stale (closure functions lived only in the
   session scratchpad) → synced; validation re-run and reproduced
   (junction: not claimed, 1,396 nodes uncapped; random: 91 positions,
   52 closure-wins, 0 divergences).

**Round 2 (Codex ultra, hostile, focused on the repaired §§4–7).**
Confirmed-repaired: the canonical X/Y coupling algebra (all five case
updates verified), filler subroutine, ply/budget synchronization; the (MI)
anchor accounting (identity and inequality direction verified); the
finite/legal repair of the old Z3; L5 restatement; Z1-retention
consistency. Still refuted, now repaired in this revision:
1. The one-hop (Z5) guard: a multi-hop corridor of dismissed stones — each
   later link ghost-illegal and hence invisible to per-node checks — could
   still reach protected territory → horizon-scaled band 8·D_N with the
   chain-closure lemma L9, whose clearance invariant propagates down chains
   (the contradiction always fires at the first, ghost-legal link).
2. LOSS-leaf continuation: a fixed placement list cannot cover the
   defender's adaptive choice of hits (explicit two-threat counterexample)
   → adaptive contract quantifying over every legal defender reply
   sequence H, with worst-case resolution leaf-ply + b + 2; leaf transfer
   rewritten to argue post-leaf plies directly in the real game.
New gaps, repaired: D9 lacked a syntactic grammar (exact successors, no
defender-terminal edges, typed maximal nodes, nonempty S(N), checked
¬own_win_now at AND nodes and checked own_win_now at WIN leaves); the
anchor's Prot-qualification citation; T5/T7/D13 omitted deep attacker
setup cells (L10 added: A-touched containment holds only for ≤ 3 remaining
attacker placements; core ∩ Legal is now an explicit searched-set term
everywhere); T6's proof replaced (own_win_now-exclusion count bounds:
a D-alive window is ≤ 3/≤ 4 at b = 2/1 nodes and gains ≤ 2 more before any
LOSS-leaf resolution — never 6); notation defined (hitting(P), r3,
R(P, D)), stale (Z3) references removed.

**Round 3 (Codex ultra, hostile, narrow: L9/(Z5), adaptive leaves, D9
grammar, T6 proof, T5/T7/D13 rescope).**
SOUND: L9 and the 8·D_N band — chain arithmetic, strict/non-strict
boundary, last-ply D ≥ 1, and backward monotonicity all verified; the
adaptive LOSS hitting transfer (verified *stronger* than stated: real and
ghost 𝒯-masks are identical at the leaf); T6's count bounds; L10's
counting; T5's joint conditions; T7's containment.
STILL-BROKEN, repaired in this revision with R3's own prescribed
arguments: (1) the post-leaf own_win_now exclusion had cited the anchored
argument past the point where the coupling stops → replaced by the
leaf-time X̂ ∩ W split (empty: leaf-time (MI) transfers own_win_now to the
ghost, contradicting the checked leaf; nonempty: the case-(b) anchor is
coupling-independent pure counting); (2) D9 lacked root binding, a path
clock, the leaf-ply convention, an OR-COMPLETION leaf type, the complete-
turn convention for H, and the explicit LOSS-leaf ¬own_win_now recheck →
all added (R3's one-node fake-root counterexample is excluded by the root
clause); (3) T6 uniformly built a b′ = b − 1 LOSS leaf, impossible at
b = 1 → b = 2 LOSS-leaf / b = 1 attacker-WIN-leaf split, first-dismissal
framing, witness cells covered as hitting cells of the node, and the
same-T argument via the searched minimum-hitting path; (4) L9's legality-
decomposition sentence restricted to ghost-empty cells (Y-cells are
ghost-occupied and irrelevant to its uses); (5) wording: L10
"immediately before placement", core ∩ Legal "redundant, not empty",
D ≥ 6 dismissal conditions necessary-not-sufficient, certified
sufficiency conditional on valid D9 + (Z4), stale (Z1)–(Z4) references
corrected to (Z1), (Z2), (Z4), (Z5).
R3's final-tag ruling: T3/T4/T6 may carry [PROVEN] *after* these repairs,
with the caveat: proven for valid zone-carrying certificates (D9 grammar)
satisfying (Z1), (Z2), (Z4), (Z5) with exact D_N and the full (not
sharpened) defender-placement budget; T7 is a coverage theorem.

**Round 4 (Codex, confirmation pass).** All five R3 repair groups ruled
APPLIED-CORRECTLY; no remaining proof gap, contradictory sentence, or
dangling live reference found in §§5–6; OR-COMPLETION leaves confirmed
non-dangling (handled by Step O); the T6 same-T and no-double-count steps
verified. Final tag ruling adopted verbatim on T3, T4, T5, T6, T7 (each
theorem's caveat is printed at its statement). Remaining Z3 mentions occur
only historically in this log, by design.

**Round 5 (tightenings review, 2026-07-14).** External-model claims in
`docs/_T3_TIGHTENINGS_REVIEW_CLAIMS.md` received an independent Claude
review and a Codex ultra hostile review, recorded in
`docs/_T3_TIGHTENINGS_REVIEW_ROUND1.md`. Final verdicts:
**8 CONFIRMED, 4 CONFIRMED-WITH-REPAIR, 0 REFUTED**; no error was found in
the prior normative statements. This revision adopts all twelve items, with
the report controlling the claims document on disagreement and all four
prescribed repairs installed.

**Round 6 (Codex ultra, hostile confirmation of the round-5 revision,
2026-07-14).** Full report in `docs/_T3PLUS_ROUND6_CONFIRMATION.md`. All
twelve substantive adoptions and all four controlling repair installations
ruled **APPLIED-CORRECTLY**; no mathematical proof repair prescribed.
Untouched-section integrity verified against the pre-revision text (D1–D8,
§8, §§10–11, historical log entries). Two mechanical defects found and
fixed in this revision: a malformed sentence at T4 (replaced with the
review's exact prescribed wording) and line-terminator damage on the
Round-4 log entry (endings renormalized). With those repairs the pass rules
the revised statements confirmed at PROVEN quality; the per-statement
provisional caveats were dropped accordingly.
