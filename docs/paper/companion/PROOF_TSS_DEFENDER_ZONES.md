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
  |𝒯| ≤ 3 for b = 1 and |𝒯| ≤ 5 for b = 2.

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

The distance-chain inequality is sharp for an exact role rank `r`: a
ghost-legal seed may be followed by `r-1` successive distance-eight
Defender placements, with the protected target last. This does not
establish sharpness of the uniform `B`-only wrapper.

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

For one fixed window, the causal counting inequality is attained at
`E^D=7`: a legal seed at distance eight may be followed by the first
`W`-fill and the five remaining fills. This does not pin the full union
`Z_virgin`.

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

**L13 (Hexo-sparse LOSS witnesses). [PROVEN]** A D9 LOSS witness can be
chosen with at most three windows at `b=1` and at most five windows at `b=2`.
Both bounds are sharp among Hexo threat-window families.

*Proof.* Every threat empty set has size one or two.

For `b=1`, the proof in L13 is unchanged.  If the family contains `{a}`, add
one set missing `a`.  Otherwise start with `{a,b}`, add a set missing `a`,
and add a set missing `b`.  At most three sets have transversal number greater
than one.

For `b=2`, first handle a singleton `{a}`. If `{a}` is a member and the
subfamily `H` of members missing `a` had `tau(H)<=1`, then `{a}` together
with an at-most-one-point transversal of `H` would hit the whole family with
at most two points. Thus `tau(H)>1`; the `b=1` selection takes at most three
members of `H`, and adding `{a}` gives at most four, hence at most five.

Assume now that every member has size two.  Choose an inclusion-minimal
subfamily `G` with `tau(G)>2`.  The L13 maximal-disjoint-family proof gives
`|G|<=6`.  Suppose equality holds.  There cannot be three disjoint members,
because those three already have transversal number three.  A maximal
disjoint family cannot have one member, because that two-point member would
hit all of `G`.  Thus it consists of

```text
E_1={a,b},    E_2={c,d}.
```

The four two-point transversals of `E_1,E_2` are `{a,c}`, `{a,d}`, `{b,c}`,
and `{b,d}`.  Each is missed by a member of `G`.  Equality at six requires
four distinct missing members, each missing only its assigned cross-pair;
otherwise the L13 selection uses at most five sets.  A two-set missing
`{a,c}` but meeting the other three cross-pairs must be `{b,d}`.  If it used
only `b` or only `d` from `{a,b,c,d}`, it would miss a second cross-pair; if
it used two outside points, it would be disjoint from both `E_1,E_2`.
Cycling the argument yields `{b,c}`, `{a,d}`, and `{a,c}`.  Therefore a
six-member minimal obstruction is exactly the six edges of `K_4`.

Such a `K_4` is not a Hexo threat-empty family.  A pair that is the empty set
of a threat window is axis-collinear by F1.  Four cells that are pairwise
axis-collinear either lie on one common axis line or are impossible: after
fixing one cell, three different incident axes give coordinates
`(u,0),(0,v),(w,-w)`, and pairwise alignment forces
`u=v=w=-v`, a contradiction.  If three cells already lie on one axis, an
off-line fourth has only two nonparallel axis lines that meet that common
line, so it cannot align with all three distinct cells.  In the common-line
case, order the four empty cells `p_1<p_2<p_3<p_4`.  Any consecutive
length-six window containing `p_1` and `p_3` also contains the intervening
empty `p_2`.  Its empty set therefore cannot be the `K_4` edge
`{p_1,p_3}`.  Contradiction.  Hence `|G|<=5`.  Sharpness is proved in
§§4.2–4.3. ∎

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

## 6a. Forced-hit gate calculus [PROVEN]

The full derivation and coordinate traces are in
`docs/_OPEN_FHW_REPORT.md`. A branch-coherent weakening proved here uses
protected exact-copy forcing gates.

For T11, an ordinary node is a D21-governed internal AND node. It may use a
T11.1/D17 envelope dismissal by dismissal. A D19 gate is not ordinary. A T6
kernel-region node remains governed by T6 and is outside this extension
unless a separate equal-position T6 handoff is declared.


### 6a.1 A reachable disjoint-hit race

Let

```
W  = {(0,r) : 0 <= r <= 5},
T1 = {(q,-4) : 0 <= q <= 5},   a = (5,-4),
T2 = {(q, 8) : -5 <= q <= 0},  b = (-5,8).
```

The following legal history starts with `D` as Player 0.  Each displayed pair
is one two-placement turn.

| plies | mover | placements |
|---:|:---:|:---|
| 0 | D | `(0,0)` (Opening) |
| 1--2 | A | `(0,-4)`, `(1,-4)` |
| 3--4 | D | `(0,1)`, `(-1,-4)` |
| 5--6 | A | `(2,-4)`, `(3,-4)` |
| 7--8 | D | `(0,2)`, `(0,-8)` |
| 9--10 | A | `(4,-4)`, `(-4,8)` |
| 11--12 | D | `(0,3)`, `(1,8)` |
| 13--14 | A | `(-3,8)`, `(-2,8)` |
| 15--16 | D | `(-8,0)`, `(8,-8)` |
| 17--18 | A | `(-1,8)`, `(0,8)` |

Every setup placement is within distance 8 of an existing stone.  The final
setup has no complete window: the only collinear five-stone A runs are the
displayed parts of `T1` and `T2`, and the only four-stone D run is the displayed
part of `W`.  It is now D's FirstStone position, so `b = 2`.

The complete current attacker threat-empty family is

```
F = {{a}, {a,(6,-4)}, {b}, {b,(-6,8)}},        tau(F) = 2.
```

The singleton members force every transversal to contain `a` and `b`, so its
extendable-hit kernel at `b = 2` is exactly `{a,b}`.  This complete available
hitting kernel is disjoint from `W`.  Nevertheless D may ignore the threats
and play

```
u = (0,4), then v = (0,5).
```

Both placements are legal at turn start.  After `u`,
`cnt_D(W) = 5`; after `v`, `cnt_D(W) = 6`, so D wins immediately on the
second placement.  A never receives the turn in which the ignored threats
would be completed.

Translations and axial symmetries give the same local counterexample, and
arbitrarily many isolated legal filler pairs can be added without meeting the
three named windows.  Thus this is an infinite family, not a boundary effect
of the Opening.

This refutes all rules which subtract the two disjoint “forced” hits without
charging the branch on which D refuses them.  The defect is not dual purpose:
neither hit is in `W`.  It is not a failure to split a `b = 2` pair: the two
placements are displayed separately.  It is the terminal ordering in D4.

### 6a.2 Coupling and substitution obstructions

The following are the compressed conclusions of report §§1.2–1.4; that
report contains the full coordinate traces and exact successor checks.

**Coupling divergence.** At an ordinary `D,b=1` node, a dismissed real reply
and searched ghost reply give `X={(5,0)}` and `Y={(0,8)}`. A fixed window
has ghost D-count three and real D-count four. The next `b=2` gate has
`E1={(14,0),(15,0)}` and `E2={(24,0),(25,0)}`, with `tau=2` and
`K=E1 union E2` disjoint from the window. Refusing the gate and playing
`(3,0),(4,0)` completes the real window. The searched exact child instead
reaches a LOSS leaf with the three disjoint pairs
`{(1,20),(2,20)}`, `{(0,21),(0,22)}`, and `{(1,19),(2,18)}`, hence
`tau=3`. A flat debit reads `3+1<6`; the required escape floor reads
`3+1+2=6`.

**Pressure mismatch.** A ghost family of two disjoint threats has
`tau=2`, while a real-only `X`-stone at `(4,0)` kills one component and
leaves real pressure one. D19 checkpoint roles are therefore required to
make every named gate mask identical at entry.

**Substitution corridor.** Ghost hits `(4,0),(34,0)` and substituted real
hits `(5,0),(35,0)` answer the same two threats but create different
`X,Y` sets. The shared attacker turn plays `(33,1)` and `(33,2)`. Both cells are legal at
turn start from the shared stone `(33,0)`; neither prefix is terminal. They
leave the minimum distances from `(5,8)` equal to 8 in the real position and
9 in the ghost position, and leave the pre-relay distance from `zeta=(5,16)`
equal to 9 in both positions. The real placement `(5,8)` then reduces only
the real distance to `zeta` to 8.
Thus a substituted hit retains D17's transition `+1` and full envelope;
only an exact copied gate hit receives the debit.

### 6a.3 Formal exact-copy weakening


#### D19 (protected exact-copy forcing gate)

**D19 (protected exact-copy forcing gate).** At an internal AND node `Q`
with defender budget `b in {1,2}`, a *forcing gate* names a finite family
`H_Q` of current A-threat windows.  Write

```
F_Q = { E(U,P_Q) : U in H_Q }.
```

The verifier checks

```
tau(F_Q) = b,                  (FG1)
not own_win_now(P_Q).          (FG2)
```

For a cell `d`, put

```
F_Q \ d = { E in F_Q : d notin E },
K(Q) = { d in Legal(P_Q) : tau(F_Q \ d) <= b-1 }.
```

For every `U in H_Q` and `e in E(U,P_Q)`, D19 extends D10 by a third role type,
the checkpoint role `(Q,U,e)`.  Its deadline is the gate-entry mask check
immediately before the defender reply.  It is in every strict ancestor's
reachable obligation union.  At `Q`, let `Prot^-(Q)` be the incoming protected
set before the check; it still includes all checkpoint carriers, and D12's
`X intersect Prot^-(Q) = empty` invariant is maintained by the coupling and
proved in L17.  The verifier checks the checkpoint roles' ancestor
coverage/ranks and the named ghost masks; it does not inspect path-local real
`X`.  Immediately after the masks are checked, discharge those roles and call
the resulting set `Prot^+(Q)`.  The gate then evaluates `S(Q) = K(Q)` in this
post-check phase and is not an ordinary D21 zone node.  This two-phase notation
is only needed at a gate; ordinary nodes continue to use `Prot(N)`.

The searched set at the gate is exactly the nonempty set `K(Q)`.  Every
`d in K(Q)` has its exact D4 child `C_d`, and a continuing real reply is copied
on that edge.  No D17 substitution is permitted for a kernel reply.  A real
reply outside `K(Q)` abandons the original subtree and uses the adaptive
escape contract of L15.

Let `p(Q)` be the absolute index of the last placement before entry to `Q`,
using D9's path-derived clock.  At a root gate it is the last placement index
of the history defining the root position (equivalently, one less than the
root's next-ply clock).  The escape deadline is

```
p(Q) + b + 2.
```

Indeed the remaining D placements have indices `p(Q)+1` through `p(Q)+b`,
and the following A placements have indices `p(Q)+b+1` and `p(Q)+b+2`.
The certificate horizon is the maximum of its old declared resolutions and
all reachable escape deadlines.  A verifier wishing to preserve the old
horizon checks each escape deadline against that old maximum instead.

Every `K(Q)` is finite and nonempty.  If `d` hits no member of `F_Q`, then
`F_Q \ d = F_Q` and `d notin K(Q)`, so `K(Q)` is contained in the finite
union of the named empty sets.  If `T` is a size-`b` minimum transversal,
then every `d in T` satisfies that `T \ {d}` hits `F_Q \ d`; hence
`d in K(Q)`.  T1 makes all such cells legal.

For compact data, one threat suffices at `b = 1`.  At `b = 2`, L13's
`b = 1` selection argument applied to a family of transversal number 2 gives
a subfamily of at most three threats still having transversal number 2.

#### D20 (the branch-coherent `F + H_W` clocks)

**D20 (forced-hit-debited role ranks).** For a role `rho` live at `N`, define
`f_N(rho)` in reverse topological order:

```
f_N(rho) = 0
    at rho's deadline or when rho is no longer reachable;

f_N(rho) = f_C(rho)
    across an ordinary OR edge while rho remains live;

f_N(rho) = 1 + max_C f_C(rho)
    at an ordinary AND node;

f_N(rho) = max_{d in K(N)} f_{C_d}(rho)
    at a forcing gate.
```

A child where `rho` is not reachable contributes zero.  As in D15, put

```
f_N(y) = max { f_N(rho) : rho is live at N and is carried by y }.
```

The unit of `f` is an opportunity to add a real-only defender stone.  An
ordinary AND edge costs one.  An exact gate edge costs zero because its
defender stone is shared; an off-kernel reply abandons every old role.

**D20a (forced-hit-debited window exposure).** For every window `W`, define
`Q_N^D(W)` by the D16 leaf and OR clauses and the following AND clauses:

```
Q_N^D(W) = 0
    at WIN and OR-COMPLETION;

Q_N^D(W) = b
    at a LOSS leaf with remaining defender budget b;

Q_N^D(W) = 0
    at an OR whose designated A-placement enters W;

Q_N^D(W) = Q_C^D(W)
    at every other ordinary OR;

Q_N^D(W) = 1 + max_C Q_C^D(W)
    at an ordinary AND;

Q_N^D(W) = max {
                    b,
                    max_{d in K(N)}
                       ( 1[d in W] + Q_{C_d}^D(W) )
                  }
    at a forcing gate with remaining budget b.
```

As in D16, set `Q_N^D(W) = 0` when `W` is already non-D-alive.  The maximum
inside the gate must be taken child by child.  Replacing it by
`max_d Q_{C_d} + 1[K intersect W nonempty]` is admissible but can combine
different branches and overcount by one.

The path interpretation is exact for the abstract gate bookkeeping.  On a
continuing mapped path let `F` count:

1. every ordinary defender opportunity;
2. every remaining defender placement at a LOSS leaf; and
3. all `b` placements of the first off-kernel escape turn.

Let `H_W` count exact copied gate placements whose cell lies in `W`.  Then
`Q_N^D(W)` is the branch-coherent maximum of `F + H_W` generated by these
rules.  A copied hit outside `W` is debited.  A copied hit in `W` is charged
once.  A quiet placement of a `b = 2` split is charged through `F`.  An
ignored gate is charged through the `b` escape floor.

Here `F` means full-cost placements: every ordinary defender opportunity and
every placement in the first LOSS or off-kernel escape remainder. An
ordinary placement which happens to hit a threat is still counted in `F`.
Only an exact copied gate placement receives the forced-hit debit, and it
contributes to `H_W` exactly when its cell lies in `W`.

The verifier uses the exact reverse-topological values in the displayed
recurrences. These are the labels for which L16(3), the comparisons
`f_N(rho)<=r_N(rho)` and `Q_N^D(W)<=E_N^D(W)`, and the `B`-bounded finite
query procedure are asserted.

For D14-D16 and for a full D17 envelope, a D19 gate remains an internal AND
node: `B`, `r`, and `E^D` retain their original `+1` inequalities over every
`K(Q)` child. Only `f` and `Q^D` use the D20 gate clauses. Consequently the
full clocks cover all defender placements in an off-kernel escape, while the
debited clocks measure only their stated hazards.

#### D21 (debited zones)

**D21 (ordinary debited zone).** At an ordinary internal AND node define

```
Z_dir^FH(N) = Prot(N) intersect Legal(P_N),

Z_seed^FH(N) = union {
    Legal(P_N) intersect B_{8(f_N(y)-1)}({y}) :
    y in Prot(N) \ (Legal(P_N) union Stones(P_N)),
    f_N(y) >= 1
},

Z_touch^FH(N) = union {
    E(W,P_N) :
    W is D-alive at P_N,
    cnt_D(W,P_N) >= 1,
    cnt_D(W,P_N) + Q_N^D(W) >= 6
},

Z_virgin^FH(N) = {
    c in Legal(P_N) :
    some all-empty window W has
    Q_N^D(W) >= 6 and d(c,W) <= 8(Q_N^D(W)-6)
}.
```

The node searches an independently nonempty superset of

```
Z_dir^FH union Z_seed^FH union Z_touch^FH union Z_virgin^FH.
```

The obligation set includes all reachable checkpoint roles from D19.  Clause
(Z4) is unchanged.  A forcing gate instead performs its entry checks and
searches exactly `K(Q)`; every off-kernel reply uses the escape contract.

For a fixed augmented certificate,

```
f_N(rho) <= r_N(rho),       Q_N^D(W) <= E_N^D(W).
```

Thus every displayed radius or completion threshold is no larger than its
D15/D16 predecessor.  Adding checkpoint roles can enlarge `Prot` at strict
ancestors; the theorem makes no claim that the total number of searched cells
must decrease in every certificate.

**D13/T7 augmented clause.** At a D21 ordinary node set
`R_cert^FH(𝒸,N)=Z_dir^FH(N) ∪ Z_seed^FH(N) ∪ Z_touch^FH(N) ∪
Z_virgin^FH(N)`. Any independently nonempty `S(N)` containing
`R_cert^FH(𝒸,N)`, together with (Z4) and all reachable D19 checkpoint roles,
is sufficient by T11. The optional solver superset is
`R_search^FH=R_cert^FH ∪ hitting(P_N) ∪ 𝒜(P_N) ∪ r3(P_N)`. At a
D19 gate this clause does not apply: the certified searched-child map is
exactly `K(Q)`, and heuristic terms are not added to `S(Q)`.

### 6a.4 Transfer and soundness



#### L15 (gate transfer and escape)

**L15 (protected gate dichotomy). [PROVEN]** On entry to a D19 gate, every
named threat window has identical real and ghost masks.  For every real legal
defender reply `d`, exactly one of the following holds.

1. `d in K(Q)`.  The cell is shared-empty and ghost-legal.  Both games place
   at `d`, take the exact child, and leave `X` and `Y` unchanged.
2. `d notin K(Q)`.  If D does not win during the remaining current turn, A
   completes a named surviving threat in at most two placements of the
   following turn, by `p(Q)+b+2`.

*Proof.* Let `U in H_Q`.  Its A-stones are shared.  Every ghost empty
`e in E(U,P_Q)` is a checkpoint carrier and hence is not in `X` at the entry
check.  Since `U` is ghost A-alive, no ghost D-stone lies in `U`, so no such
cell is in `Y`.  The complete real and ghost masks of `U` agree.

If `d in K(Q)`, then `d` belongs to a named empty set, is shared-empty, and is
legal by T1.  D19 supplies its exact child, so this is case 1.

Otherwise

```
tau(F_Q \ d) > b-1.
```

Let `H` be the set of the at most `b-1` later defender placements in the
current turn.  It cannot hit every member of `F_Q \ d`.  A named window
therefore avoids both `d` and `H`, remains A-alive, and retains its one or two
initial empties.  L1 puts those empties within distance 2 of permanent shared
A-stones, so they are legal and A completes the window in the next turn.  D4
allows D to terminate first during its current turn; that is the alternative
explicitly retained in the statement and charged by D20a.  The two cases are
exclusive by definition.  ∎

#### L16 (clock bounds and nesting facts)

**L16 (weighted hazard bounds). [PROVEN]** For every D19--D21 certificate:

1. On a continuing mapped path, the number of ordinary real-only frontier
   opportunities before a live role's deadline is at most `f_N(rho)`.
2. For a fixed window `W`, on every continuation before the certificate
   attacker wins or first enters `W`, or before an off-kernel escape resolves,
   count one for each ordinary defender edge, `1[d in W]` for each exact copied
   gate edge, and every remaining defender placement in the first LOSS or
   off-kernel escape remainder. This count is at most `Q_N^D(W)`.
3. `f_N(rho) <= r_N(rho)` and `Q_N^D(W) <= E_N^D(W)`.
4. At a gate, `B(Q) >= b`.  Consequently D14/L11's ancestor budget covers
   every escape remainder even though `B` is not debited.

*Proof.* Items 1 and 2 follow by reverse induction on the finite certificate.
An ordinary AND placement can add one `X`-stone and is charged one by both
recurrences.  An exact gate placement changes neither `X` nor `Y`; it is
charged to a chosen window exactly when its cell lies in that window.  An
off-kernel reply ends the old role contract and permits at most the current
turn's `b` defender placements before L15's A reply.  A LOSS leaf likewise
permits all `b` placements, so its base cannot be debited.  OR stops and
ordinary OR propagation are exactly D16's.

For item 3, induct against D15 and D16.  The role inequality is immediate:
an old AND edge adds one where a gate edge now adds zero.  At a `b = 1` gate,
the old D16 exposure is `1 + max_C E_C`, which dominates both the escape floor
1 and every `1[d in W] + Q_{C_d}`.  At a `b = 2` gate, every exact child is a
nonterminal `D,b = 1` node, hence has old exposure at least 1 for every still
D-alive `W`.  Therefore the old `1 + max_C E_C` is at least 2 and also
dominates every continuing child term.  Leaf and OR comparisons are equal.

For item 4, `K(Q)` is nonempty.  If `b = 1`, the D14 AND inequality gives
`B(Q) >= 1`.  If `b = 2`, every kernel successor is a nonterminal `D,b = 1`
position.  It is either a LOSS leaf, whose budget is at least 1, or another
internal AND node, whose budget is at least 1.  Hence
`B(Q) >= 1 + B(C_d) >= 2`.  D14 nesting then covers an escape from every
ancestor.  ∎

#### L17 (joint protected-occupation and completion safety)

**L17 (debited first-bad-event lemma). [PROVEN]** Under D19--D21:

1. while the real play remains mapped to the original certificate, no defender
   placement creates a real-only stone in the current protected set; the old
   roles are abandoned when an off-kernel escape begins; and
2. before the mapped certificate attacker or a gate escape attacker resolves,
   no real defender play completes a window.

*Proof.* Suppose the first failure of the applicable type occurs.  Every
earlier gate entry on the mapped prefix has exact named masks by the checkpoint
roles and the minimal choice of the failure.

For a protected occupation at carrier `y`, the direct ghost-legal case is in
`Z_dir^FH` and is searched.  Otherwise trace real-only legality backward from
`y` to the last ghost-legal dismissed `X`-seed `x_0`.  A copied gate stone is
present in both games. If it supplies legality for a later real placement at a
ghost-empty cell, that cell is ghost-legal and any dismissal is a newly checked
seed. If the later cell is ghost-occupied, the move is T3 case A2: it cancels a
`Y`-stone and creates no `X`-stone. Hence a copied gate stone cannot be an
internal link of a ghost-illegal real-only chain. Every link of the final
`X`-chain was created at an ordinary AND opportunity.

If the chain has `j` stones,

```
d(x_0,y) <= 8(j-1),       j <= f_{N_0}(rho)
```

for the still-live role `rho` carried by `y`.  Hence `x_0` was in
`Z_seed^FH(N_0)`, contradicting its dismissal.  This proves the checkpoint
invariant as a special case, because a checkpoint role remains live through
its entry check.

Now suppose the first failure is a real D-completion of `W`.  If no W-cell was
ever dismissed, (MI) gives the real D-count at most the ghost D-count.  On a
continuing exact path the ghost has no defender-terminal edge.  At a LOSS
remainder the mandatory ghost `¬own_win_now` check and the fully charged base
`b` give at most five stones.  At an off-kernel gate, the explicit gate check
gives ghost count at most 3 for `b = 2` and at most 4 for `b = 1`; (MI) and the
escape floor `b` again give a real maximum of five.  Thus this case cannot
complete.

If the first real-only W-fill occurs on an off-kernel reply and there was no
earlier real-only W-fill, gate-entry (MI), (FG2), and the fully charged escape
floor bound the final real count by five exactly as in the preceding
no-dismissal case.  It cannot complete.

Here “no `W`-cell was ever dismissed” concerns the mapped prefix. Any first
`W`-fill among the later placements after an off-kernel reply is already
included in the full `b` escape floor.

Therefore every remaining completion case has a first real-only W-fill
anchored at an earlier ordinary node: a continuing gate reply is copied, and
the only non-copied gate reply terminates the mapped line during its charged
escape turn.

Take that first ordinary W-fill and its last ghost-legal dismissed seed.  If
ghost `W` is already touched at the first fill, pre-fill (MI) bounds the real
count by `cnt_D(W,P_N)`.  L16 charges every later ordinary placement, every
copied gate hit in `W`, every LOSS remainder, and any final escape turn.  A
completion would imply

```
cnt_D(W,P_N) + Q_N^D(W) >= 6.
```

The first fill was therefore in `Z_touch^FH(N)`, a contradiction.

If ghost `W` is virgin and the first real-only fill is ghost-legal, completion
requires six charged W-fills, so `Q_N^D(W) >= 6`; distance zero puts the fill in
`Z_virgin^FH(N)`.  If it is ghost-illegal, trace from the last ghost-legal
dismissed seed to `W`.  Let `j` be the number of real-only radius-8 links before
the chain first reaches `W`.  Before a final escape or LOSS remainder, only
ordinary placements can form those links; copied gate placements outside `W`
are shared, while copied gate placements in `W` are charged through `H_W`.
Every approach link or W-fill in a final LOSS remainder or escape turn is
charged by its full `b` base or floor.  The chain and the six real W-fills give

```
Q_{N_0}^D(W) >= j + 6,       d(x_0,W) <= 8j.
```

Thus `x_0` lies in the radius `8(Q_{N_0}^D(W)-6)` virgin term, again
contradicting dismissal.  This also covers a completion during an escape: the
gate's entire remaining turn is in the `b` floor.  ∎

#### T11 (soundness of the exact-copy debit)

**T11 (exact-copy `F + H_W` soundness). [PROVEN]** Let a finite D9 tree or
D18 DAG use D21 at every ordinary AND node and D19 at every forcing gate.
Include every checkpoint role in the reachable obligation unions, retain
(Z4), retain D14's `B`, and include the D19 escape deadlines in the global
horizon.  Then the compiled attacker wins against every real defender play by
that horizon.  The debited `f` and `Q^D` values are sound replacements for
`r` and `E^D` in the D21 seed, touched, and virgin terms.

*Proof.* At an ordinary node run T3's A1--A3 coupling, using D21 and L17 in
place of L9′ and L12.  At a gate, first check and discharge the checkpoint
roles.  L15 transfers the named family.  A real reply in `K(Q)` is shared-empty,
is copied on its exact child, and leaves `X,Y` unchanged.  A reply outside
`K(Q)` abandons the old subtree.  L17 excludes a D-completion in the remaining
turn, and L15 supplies an adaptive surviving threat which A completes by the
declared escape deadline.

The WIN, LOSS, OR-COMPLETION, and ordinary OR transfers are unchanged.  The
LOSS branch remains sound because D20a charges its whole remainder and D10
still protects every witness empty through leaf entry.  A finite tree must
reach a typed terminal or a first escape.  For a DAG, unfold as in T10; the
extra labels and max recurrences are preserved.  ∎

**T11.1 (D17 envelope compatibility). [PROVEN]** T11 remains valid when an
ordinary node's global D21 dismissal tests are replaced by a valid D17
envelope, provided the selected reachable-role union includes every future
checkpoint role and the envelope retains D17's original transition-inclusive
`B-hat`, role ranks, and `E-hat` tests.  No debited D17 envelope is claimed by
this corollary.

The selected role union contains every role live at `C_s` and at every node
reachable from `C_s`, expressly including a checkpoint role whose deadline
is entry to `C_s` itself. D17 condition 3 therefore forbids the transition
cell `d` from occupying such a carrier.

*Proof.* Before the selected child, D17's real `d` and ghost substitute `s`
create the same canonical `X,Y` transition as in T9, so its current `+1` is
mandatory.

D17 conditions 2-8 protect all selected-child roles and all three L3
channels, including LOSS remainders. If the protected-occupation seed or
first real-only window fill was introduced at a D21 dismissal, L17 applies
with `f` and `Q^D`. If it was introduced by a D17 transition, D17 conditions
3-5 apply with the full transition-inclusive role rank and window exposure;
condition 7 carries that envelope through later ghost-illegal descendants,
and condition 8 covers a LOSS remainder. L16(3) gives `f<=r` and `Q^D<=E^D`
for later D21 steps. This joint D17/D21 induction remains valid through a
D19 gate because the full clocks use the old AND inequalities there. An
off-kernel reply abandons the selected subtree and is charged by its gate
floor.

The remainder of T9's nested-envelope proof is unchanged.  ∎

### 6a.5 Sharpness



#### 6a.5.1 A strict exposure debit attained by a real line

Consider a `D,b = 1` position with the following stones.  The two colors have
15 stones each, the position is nonterminal, and the displayed support chain
makes every module radius-8 connected.

```
A = {(q,0) : 0 <= q <= 4}
    union {(-3,20),(-2,20),(-1,20),
           (0,17),(0,18),(0,19),
           (-3,23),(-2,22),(-1,21)}
    union {(0,-7)}.

D = {(-1,0),
     (8,0),(8,8),(8,16),(8,24),(8,32),
     (0,40),(1,40),(2,40),
     (-8,0),(0,-8),(-8,8),(8,-8),(16,-8),(16,0)}.
```

Direct enumeration of the 18 incident windows per stone gives no complete
window and a maximum D-count of 3 in a D-alive window.  Name the singleton
root threat

```
T = {(0,0),(1,0),(2,0),(3,0),(4,0),(5,0)},
h = (5,0).
```

Then `F = {{h}}`, `tau(F) = 1`, and `K = {h}`.  D copies `h`.  A plays
`(3,20)` and then `(0,20)`, reaching a `D,b = 2` LOSS leaf with the three
windows `U1,U2,U3` from the coupling-divergence trace in §6a.2.  Their empty
pairs are pairwise disjoint,
so `tau = 3`, and the D-alive maximum is still 3.

Let

```
W = {(q,40) : 0 <= q <= 5}.
```

Initially `cnt_D(W) = 3`, and `h notin W`.  The old D16 exposure is

```
E_root^D(W) = 1 + 2 = 3,
```

so the old touched test reads `3 + 3 = 6` and searches all W-empties.  The
new branch-coherent value is

```
Q_root^D(W) = max { 1, 0 + 2 } = 2,
```

so the new test reads `3 + 2 = 5` and omits them.

The value 2 is attained.  Follow the copied hit `h`; at the LOSS remainder D
plays `(3,40)` and `(4,40)`.  Exactly two defender placements enter `W`, its
count reaches only 5, one of the pairwise-disjoint LOSS witnesses survives,
and A fills that witness.  Thus this one-unit debit and the LOSS base `b = 2`
are both sharp on a real legal line.

#### 6a.5.2 The dual-purpose coefficient is independently sharp

Use the complete certificate position and line of §6a.5.1, but query the
different D-alive window

```
W' = {(q,0) : 5 <= q <= 10}.
```

Initially it contains the D-stone `(8,0)` and no A-stone.  The copied kernel
hit `h = (5,0)` lies in `W'`.  At the later LOSS remainder let D play `(6,0)`
and `(7,0)`.  The three placements `h,(6,0),(7,0)` are legal, all lie in
`W'`, do not complete it, and are followed by A's certified LOSS-witness win.
The recurrence gives

```
Q_root^D(W') = max { 1, 1[h in W'] + 2 } = 3,
```

exactly the achieved harm.  Deleting the dual-purpose indicator would give
`max {1, 0+2} = 2`, which is false on this line.  Thus `H_W`'s unit
coefficient is independently sharp.  Sections 6a.1 and 6a.2 separately show
that the full `b = 2` escape charge is sharp when the defender splits or
ignores the asserted pressure.

The exact sharpness of every possible `f`-based seed-radius reduction is open.
On all-ordinary paths `f = r`, so L9′'s existing radius sharpness remains
the boundary case.

### 6a.6 Verifier procedure

All checks are finite.

1. Recompute each named current threat window and its one- or two-cell empty
   set.
2. Check `tau(F_Q) = b` by the existing `mhs <= 2` enumeration and check
   `¬own_win_now`.
3. Enumerate `K(Q)` from the finite union of the named empty sets and test
   `tau(F_Q \ d) <= b-1` for each cell.
4. Check `S(Q) = K(Q)`, exact D4 successors, and the absence of a substituted
   kernel edge.
5. Add the checkpoint roles to all strict-ancestor reachable-role unions and
   check their deadline masks.
6. Compute the least `f` values and the queried `Q` values in reverse
   topological order, using the child-coherent gate maximum.  The queried
   windows are finite: touched windows come from the finite position; for the
   virgin term, `Q <= E <= B`; if `B < 6` there is no query, and otherwise for
   each finite legal candidate enumerate only window starts within radius
   `8(B-6)` of that candidate, exactly as in D11's inverted virgin query.
7. At ordinary nodes enumerate the four D21 zone terms from those finite
   queries.  At gates check only the two-phase gate grammar and kernel.
8. Recompute each absolute escape deadline and include it in the horizon.

No step quantifies over an unbounded continuation.  The universal off-kernel
reply set is compressed by the finite transversal inequality in L15, exactly
as a D9 adaptive LOSS contract compresses its defender remainder.

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

For a base D9-D18 certificate, the solver may search with R_search, a smaller
heuristic, or a D17-aware generator. At verification it must either establish
the ordinary R_cert coverage plus (Z4), or validate every D17 annotation,
while retaining an independently nonempty searched fallback. A D19-D21
augmented certificate instead uses the D13/T7 augmented clause at ordinary
nodes, exact `K(Q)` at gates, and T11.1 for any D17 envelope. Any failure
yields UNKNOWN. Search trimming affects completeness only; a verified WIN
remains sound.

---

## 8. Pairing strategies: the exact hex-lattice threshold [PROVEN]

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
6-in-a-row misses by one. T8.2 supplies existence at the hex-lattice
threshold; the corollary is stated for lattices as above, not arbitrary
"grids".

**T8.2 (threshold pairing at k = 7). [PROVEN-MECH]** On the three-axis hex
lattice there is a periodic perfect matching such that every length-seven
window contains exactly one matched pair. It has period-lattice index 12,
which is minimal among periodic constructions.

*Construction.* In axial coordinates `(q,r)`, with axes `(1,0)`, `(0,1)`,
and `(1,-1)`, take

```text
Λ = ⟨(2,2),(0,6)⟩.
```

This lattice has index 12 and fundamental domain
`{0,1} × {0,1,2,3,4,5}`. Take every simultaneous `Λ`-translate of
the six pairs

```text
H: (0,0)-(1,0),  (0,1)-(1,1)
V: (0,3)-(0,4),  (1,3)-(1,4)
D: (1,2)-(2,1),  (1,5)-(2,4).
```

The homomorphism

```text
φ(q,r) = (q mod 2, r-q mod 6) in ℤ₂×ℤ₆
```

has kernel `Λ`. The twelve displayed endpoints map bijectively to the
twelve quotient cells, so their lifts form a perfect matching. Horizontal
pair starts on the two line orbits are `(6n,0)` and `(6n,1)`; vertical
starts are `(0,3+6n)` and `(1,3+6n)`; on both diagonal line orbits starts
have `q=1 mod 6`. Thus each physical line has one selected unit-edge phase
modulo six. The six internal unit edges of every length-seven window contain
exactly one selected pair.

*Rigidity and completeness.* Pass, if necessary, to a period sublattice
locally injective on sets of diameter at most six. If its quotient has `N`
cell orbits and `P` pair orbits, incidence counting gives

```text
3N <= sum_e (7-delta_e) <= 6P <= 3N.
```

Equality throughout forces `P=N/2`, every pair to be a unit pair, every
cell to be matched, every window to contain exactly one pair, and `N/6`
pair orbits on each axis. On one physical line, if `x_s` indicates a pair
starting at `s`, exact coverage gives

```text
sum_{i=0}^5 x_{s+i} = 1.
```

Subtracting consecutive equations yields `x_s=x_{s+6}`, with exactly one
selected residue. Obtaining exact coverage above uses periodicity; once exact
coverage is known, the recurrence is pointwise. The Folner-density argument
alone does not exclude zero-density defects. The six phase variables—two
line orbits for each of three axes—therefore cover every `Λ`-periodic
covering pairing, and the endpoint constraints impose exactly the remaining
matching condition.

*Index-minimality.* For any original period lattice, the order `o` of each
axis step in the quotient is divisible by six, because the line indicator
has least positive period six. Hence `6` divides the quotient index `N`. If
`N<12`, then `N=6`; the abelian quotient is cyclic. Its only elements of
order six are `g` and `-g`, so the difference of two order-six axis steps
has order at most three, whereas the third hex-axis step, their difference,
must also have order six. This contradiction proves index-minimality.

The full proof and hand coverage trace are in
`docs/_OPEN_PAIRING7_REPORT.md`. The exhaustive verifier
`scripts/_pairing7_search.py` checks the 12 quotient endpoints, all 36
window orbits, all `6^6=46,656` phase assignments, and the finite patch; it
finds 120 solutions in exactly 419 Algorithm-X states.

Consequence for the program: the threshold is exactly characterized on the
hex lattice. The necessary inequality `k ≥ 2g+1` is tight: at `g=3`, T8.2
constructs a pairing for `k=7`, while T8 proves none exists for `k=6`.
Thus the "pairing prize" of §8.2 of the survey doc is definitively closed for
Hexo. Partial pairings on constrained regions are not ruled out but cannot be
position-independent.

---

## 9. What the theorems deliver, quantitatively

- For base D9-D18 certificates the ordinary mandatory zone remains
  `Z_dir union Z_seed union Z_touch union Z_virgin`. For a D19-D21 augmented
  certificate, a D21 ordinary node uses `R_cert^FH`, while a D19 gate searches
  exactly `K(Q)`. For a fixed augmented certificate, exact `f<=r` and
  `Q^D<=E^D` shrink the seed, touched, and virgin hazard terms. Checkpoint
  roles can enlarge `Prot`, so no reduction in total searched-set cardinality
  is claimed. T6 remains a distinct equal-position kernel theorem.
- Fully forcing certificates (T6) search the extendable-hit kernel K_b and
  no original obligation term. Kernel-governed internal nodes require
  mhs ≤ b and the explicit `¬own_win_now` premise; mhs < b performs no
  pruning, while mhs = b can strictly reduce the hitting-cell set.
- LOSS leaves need at most 3 witness windows at b = 1 and 5 at b = 2.
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

*Derived independently by Codex (ultra); the foundational fixed-family text
is in* `docs/proof_parts/ES_POTENTIAL.md`. *The unrestricted-boundary
campaign, including full proofs and exact traces, is in*
`docs/proof_parts/ES_GLOBAL_BOUNDARY.md`. Statement numbers in the first six
bullets refer to `ES_POTENTIAL.md`; the strengthened-boundary bullet names
statements from `ES_GLOBAL_BOUNDARY.md`.

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
- **Strengthened global boundary [GREEDY-REFUTED; raw claim OPEN].** Theorem 1
  refutes dynamic touched-window greedy for every tie-breaking: an exhaustive
  exact-surd enumeration, with no randomness, beam, or depth cutoff and no
  omitted exact maximizer along the fixed Attacker continuation, reaches the
  win on every branch; Proposition 1 embeds the same tree in a legal
  39-placement engine-reachable history. This does not refute the raw
  existential claim for a non-greedy defense. Lemma 1 and Corollary 2 give a
  universal clean escape, so no defense can renew `Φ<1` itself: every
  continuing Attacker turn can create 36 distinct count-one labels of mass
  `4/√3`, although residual supports need not be disjoint. Proposition 2
  rules out static pairing even with finitely many exceptional windows, and
  Proposition 3 rules out static two-tier damping with a finite virgin sum,
  terminal weight at least one, and the edgewise factor-three inequalities.
  Theorem 2 guarantees five future Attacker placements from `Φ<1`; Theorem 3
  guarantees three complete Attacker turns at thresholds `1`, `2/3`, and
  `4/9` according as the first pair shares no window, has axis distance two
  through five, or is adjacent. Theorem 4 reduces every fixed horizon to a
  finite ball and uses König's lemma to equate survival at all finite horizons
  with one forever-surviving strategy. The remaining named gaps are
  `GAP-RAW`, `GAP-GLOBAL-RENEWAL`, `GAP-AMORTIZED-ABANDONMENT`,
  `GAP-DYNAMIC-PAIRING`, and `GAP-FINITE-CUTOFF`.
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

1. **Partially resolved -- protected exact-copy `F+H_W`.** D19-D21 and T11
   prove target-specific `f` and `Q^D` debits at protected tight exact-copy
   gates. No theorem debits scalar `B`, grants zero cost to a D17 substitution,
   or proves net searched-set shrinkage; those extensions remain open.
2. **Resolved — exact-rank and fixed-window tempo accounting.** D15's exact
   ranked obligation radius `8(r−1)` and D16's touched/virgin split, including
   the fixed-window virgin radius `8(E^D−6)`, resolve the former exact-rank and
   fixed-window questions. Sharpness of the uniform `B`-only wrapper and the
   full union remains open in items 5 and 6.
3. **Resolved — pairing at the threshold.** T8.2 constructs a periodic
   pairing for `k = 7`, `g = 3`, and proves index 12 minimal among periodic
   constructions.
4. **Formalization.** All objects here are finite combinatorics; a
   Lean/Coq formalization of §§1–8 is feasible and would yield a verified
   reference checker for (Z2), (Z4), (Z5′), the B/E^D/r inequalities,
   transition envelopes, and DAG label/clock consistency.
5. **Uniform obligation wrapper.** Sharpness of the uniform `8(B−1)` seed
   band is open; the exact-rank trace pins only `8(r−1)`.
6. **Full-union virgin radius.** The fixed-window arithmetic attains
   `8(E^D−6)`, but sharpness of the complete union `Z_virgin` remains open.
7. **ES `GAP-RAW`.** It remains open whether every nonterminal
   Defender-`FirstStone` position with `Phi<1` has some non-greedy
   forever-blocking strategy.

---

## 12a. Tightness frontier

This table records the final status of the quantitative and grammar constants.
Full constructions and traces are in `docs/_TIGHTNESS_FRONTIER_REPORT.md`.

| ID | Parameter | Current value | Result | Normative source |
|---|---|---:|---|---|
| R1a | Exact live-role seed band | `8(r-1)` | **PINNED relatively** | D15, L9′ |
| R1b | Uniform live-role seed band | `8(B-1)` | **OPEN** | D11, T4; §12 item 5 |
| R2 | Virgin-window seed radius | `8(E^D-6)` for `E^D>=6` | **OPEN** in general; fixed-window arithmetic attained | D16, L12; §12 item 6 |
| R3 | Touched-window guard | `cnt_D(W)+E^D(W) >= 6` | fixed-window equality attained; full weakened-L12 pin OPEN | D16, L12 |
| R4a | LOSS witness cap, `b=1` | 3 | **PINNED relatively** | L13 |
| R4b | LOSS witness cap, `b=2` | 5 | **IMPROVED** from 6; 5 is **PINNED relatively** | D9, L13 |
| R5a | Internal T6 kernel scope | `mhs<=b` | **PINNED relatively** | T6 |
| R5b | Kernel `not own_win_now` enforcement | T6 premise plus retained D9 diagnostic | PINNED absolutely for combined predicate enforcement; not a single-clause pin | D9, T6 |
| R5c | T6 residual threshold | `tau(F \ d) <= b-1` | **PINNED relatively** | T6 |
| R6 | Combined LOSS survivor contract | `tau(T) > b` plus the universal-survivor clause | PINNED for the combined contract; not the numeric test alone | D9, T3 |
| R7 | D17 transition charge | `1+` child rank/exposure | **PINNED relatively** | D17, T9 |
| R8 | D14 local-budget recurrence | AND `1+max`; LOSS `b` | exact as defined; uniform use is relative | D14, L11 |
| R9 | Legality-chain coefficient | 8 | **PINNED relatively** | D4, L9′, L12 |
| R10 | D15 ranks and deadlines | AND `+1`, OR `+0`, deadline `0`, role maximum; protection through check | **PINNED**; leaf-entry and OR-COMPLETION clauses are absolute | D10, D15 |
| R11 | LOSS deadline | `leaf-ply+b+2` | **PINNED absolutely** | D9, T3 |
| R12 | D16 exposure recurrence | AND `1+max`; LOSS `b`; stop `0` | exact as defined; uniform use is relative | D16, L11 |
| R13 | T5 static-cover cutoff | `B <= 3` with `r3` | local radius arithmetic attained; full T5 pin OPEN | T5 |
| R14 | L10 short-placement cutoff | first three future threat-creating placements | **PINNED relatively** | L10, T5 |
| R15 | Independently nonempty AND fallback | at least one legal searched reply | **Relative/syntactic** | D9, T4, D17 |
| R16 | Forced-hit debit | protected exact-copy `f` and `Q^D`; full `B`, `r`, `E^D` clocks retained | **PARTIALLY RESOLVED**; broader scalar/substitution debit open | D19-D21, T11 |

There are two kinds of pin. An **absolute pin** comes with a weakened
certificate that declares WIN although a legal real line defeats its declared
resolution. A **relative pin** attains a counting bound or breaks the cited
coupling/coverage lemma, but does not exclude a different proof or certificate
format. The distinction is recorded for every pin in the table.

R5b is absolute only if neither the T6 premise nor D9's retained diagnostic
enforces the predicate. R6 is absolute only for weakening the combined numeric
and universal-survivor contract. R15 is a syntactic well-formedness and
coupling-filler requirement: a zero-child AND has no filler and is undefined
by the current D9/D14/D16 grammar, but deleting nonemptiness alone does not
admit the alleged false certificate.

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

**Round 7 (four-report open-problems campaign, 2026-07-14).** The campaign
installed `docs/_OPEN_FHW_REPORT.md`, `docs/_TIGHTNESS_FRONTIER_REPORT.md`,
`docs/_OPEN_PAIRING7_REPORT.md`, and `docs/_OPEN_ES_GLOBAL_REPORT.md`, under
the two hostile reviews `docs/_REVIEW_FHW_ROUND1.md` and
`docs/_REVIEW_TPE_ROUND1.md`. The FHW exact-copy theorem survived as
**INSTALLABLE-WITH-REPAIRS**: D19-D21, L15-L17, T11, and T11.1 are installed
with R1-R9. The tightness, pairing, and ES reports were each
**INSTALLABLE-WITH-REPAIRS**. L13+ improves the LOSS cap from `3/6` to `3/5`;
the index-12 threshold pairing and `GREEDY-REFUTED` boundary survived. R3 and
R13 are arithmetic-attained only; R5b and R6 are absolute only for their
combined contracts. The campaign's sole **REFUTED** classification was R15
as an absolute pin: it is relative/syntactic, while the current nonempty-AND
grammar remains required. The repaired assertion-enabled tightness, pairing,
ES, and FHW review checkers all exit zero.
