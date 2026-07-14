# Soundness of defender move-set restriction in Hexo threat search — formal proofs

> **Provenance.** 2026-07-13. The proof-layer companion to
> `PLAN_TSS_MOVESET_ZONES.md` (results survey) and the intended normative
> reference for the Stage-3 solver/verifier build (`PLAN_TSS_DEEPENING.md` §6).
> Produced by an adversarial workflow: Claude drafted the framework and core
> proofs; Codex `gpt-5.6-sol` ultra passes independently derived the
> potential-function and domination components (§10, §11) and adversarially
> reviewed the whole; the mini-model solver (validated against `hexo_engine`)
> machine-checked the finite claims. Review log in §13.
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

**D9 (certificate, machine-checkable form — strengthened per reviews
R1/R2).** A *certificate* 𝒞 for "A wins from P₀ by horizon T (absolute ply
index)" is a finite tree of nodes, each labelled with a position and
mover/budget, subject to the following **grammar**, every clause of which
the verifier checks syntactically: **the root node's position is P₀ and is
nonterminal** (the certificate binds the position it claims to prove);
every node carries a **path-derived ply clock** (root ply + depth — the
verifier recomputes it; all resolution labels refer to this clock, and
"leaf-ply" means the index of the last placement completed on the path to
the leaf); every node's children are the *exact* D4-successors of its
placements; every placement is legal in its node's position; **no edge
anywhere is defender-terminal** (a searched reply that would complete a
defender window is rejected — such a certificate is invalid); every
maximal node is a typed leaf, where an OR node whose designated placement
completes is itself the typed **OR-COMPLETION leaf**; every internal AND
node has S(N) ≠ ∅; at every AND node — and again, explicitly, at every
LOSS leaf — the verifier confirms ¬own_win_now for the defender (else the
certificate is invalid).
- **OR node** (A to place): one designated placement; if it completes, the
  node names the **witness window** W* and the completion ply; else the
  child is the successor position.
- **AND node** N (D to place, budget b): a *searched set* S(N) of legal
  replies with one child each, plus the dismissal claim for all other legal
  replies.
- **WIN leaf** (attacker to place): a verifier-checked `own_win_now`
  position — an A-alive witness window with count 5 (any b), or count 4
  with b = 2 — so the completion occurs **within the attacker's current
  turn**, with no defender placement interleaved. Names the witness
  window(s) and the resolution ply ≤ T.
- **LOSS leaf** (defender to place, budget b, λ¹-lost): names a **witness
  family** 𝒯 of A-threat windows whose empty-sets have hitting number > b
  (checkable; the verifier re-derives it, including the ¬own_win_now
  check). The continuation contract is **adaptive** (review R2): for
  *every* complete, nonterminal remainder H of the defender's current turn
  — exactly his b placements unless the game ends earlier (review R3
  convention) — some W_H ∈ 𝒯 survives (E(W_H) ∩ H = ∅, guaranteed by
  hitting number > b), and the attacker then completes W_H's ≤ 2 empties
  within his following turn: D places at plies leaf-ply+1 … leaf-ply+b, A
  at leaf-ply+b+1 and (for a count-4 witness) leaf-ply+b+2. The declared
  resolution ply is the worst case, leaf-ply + b + 2, required ≤ T. No
  fixed placement list is named — the contract quantifies over H.
T is the maximum resolution ply over the tree. Every λ¹ fact a leaf asserts
is tied to named windows and a checkable continuation contract — the
transfer arguments below act on those, never on bare Boolean verdicts.

**D10 (core).** core(𝒞, N) = the union, over the subtree of N, of (i) all
cells of every *named* witness window (OR-completions, WIN leaves, LOSS-leaf
families 𝒯), and (ii) every attacker placement cell in the subtree,
including leaf continuation placements.

**D11 (zone-carrying certificate — repaired per review R1).** For an AND
node N with position P_N and D_N := 𝔇(P_N, T), define the **protected set**

  Prot(N) = core(𝒞, N) ∪ ⋃ { empties of windows W alive for D at P_N with
            cnt_D(W, P_N) + D_N ≥ 6 }.

𝒞 is *zone-carrying* iff at every AND node N:
- **(Z1) hitting:** S(N) ⊇ (empties of every A-threat window of P_N) ∩
  Legal(P_N). (Kept independently of Z2 — review R1 confirmed a current
  threat can be absent from every later witness family, and T3's leaf
  transfer does not require searching it; Z1 is retained as belt and braces
  for λ¹-consistency of interior nodes and costs nothing, since hitting
  cells are the solver's first candidates anyway.)
- **(Z2) protection:** S(N) ⊇ Prot(N) ∩ Legal(P_N).
- **(Z5) frontier guard (horizon-scaled — repaired per review R2):**
  S(N) ⊇ Legal(P_N) ∩ B_{8·D_N}( Prot(N) ∖ (Legal(P_N) ∪ Stones(P_N)) ),
  where B_r(U) = { c : ∃ y ∈ U, d(c, y) ≤ r }. In words: any legal cell
  within *chain* range — the defender has at most D_N placements before the
  horizon, each extending the frontier by ≤ 8 — of not-yet-legal protected
  territory must be searched. The one-hop (radius-8) version of R1's repair
  was refuted in R2 by a multi-hop corridor (a chain of dismissed stones
  approaching protected territory in 8-steps whose *later* links are
  ghost-illegal and hence uncheckable at their own nodes); the horizon
  scaling closes it via L9 below, whose invariant propagates down chains
  automatically. The band is empty in the common case
  Prot(N) ⊆ Legal(P_N) ∪ Stones(P_N).
- **(Z4) WF-legality:** every attacker placement in the subtree of N is
  within distance 8 of an attacker stone of its predecessor position or a
  stone of P₀ (automatic for threat-creating generators, T2).

Both Prot-terms are finite and verifier-enumerable when D_N ≤ 5 (the
completion-guard windows need cnt_D ≥ 1, hence are touched windows). For
D_N ≥ 6 every all-empty window qualifies, so Prot ∩ Legal covers every
legal cell lying in *any* D-alive window, and (Z5) additionally covers the
frontier band; the only remaining dismissible cells are deep-interior cells
lying in **no** D-alive window (every window through them is A-touched or
dead — such cells can never contribute to a defender completion, L4/L7).
This is the honest G2 boundary, now stated exactly rather than as "no
pruning".

**Monotonicity of protection [PROVEN].** For M a descendant AND node of N:
Prot(M) ⊆ Prot(N). *Proof.* core is a subtree union, monotone under
descent. For the completion-guard term: cnt_D(W, ·) grows along a play by
exactly the defender placements made, while 𝔇(·, T) shrinks by the same
count or more, so cnt_D + D is nonincreasing; a window qualifying at M
qualified at N, and its empty-set at M is a subset of its empty-set at N. ∎

**L9 (frontier chain closure). [PROVEN]** In T3's coupling, say an X-stone
x introduced at node N_x is *(Z5)-clear* iff d(x, y) > 8·D_{N_x} for every
y ∈ Prot(N_x) ∖ (Legal(P_{N_x}) ∪ Stones(P_{N_x})). Then:
(a) if every ghost-legal dismissed stone is (Z5)-clear (which is exactly
what the (Z5) searched-set condition enforces, since ghost-legal cells
inside the band are searched, not dismissed), then every ghost-*illegal*
dismissed stone is automatically (Z5)-clear as well; and
(b) consequently no cell of Prot(N) ∖ (Legal(P_N) ∪ Stones(P_N)) is ever
legal in the real game while illegal in the ghost.
*Proof.* Any cell that is **ghost-empty**, real-legal, and ghost-illegal
has an X-stone legality witness: its real justification is a real stone
within distance 8, and among real stones only X-stones are absent from the
ghost (shared root/attacker stones and searched placements exist in both;
Y-cells are ghost-*occupied* and are irrelevant here since every L9/A3 use
concerns ghost-empty cells). Inductively, every X-stone's own
real-legality traces back through a chain x₀, x₁, …, x_m of X-stones with
x₀ real-legal via a shared stone (hence ghost-legal) and consecutive links
within distance 8.
(a) Suppose some ghost-illegal x_j (j ≥ 1) violated clearance: some
protected-not-legal y at N_{x_j} with d(x_j, y) ≤ 8·D_{N_{x_j}}. The chain
prefix gives d(x₀, x_j) ≤ 8·k where k = the number of defender placements
from x₀ up to x_j inclusive minus... precisely, x₁, …, x_j are defender
placements after x₀, so j ≤ D_{N_{x₀}} − 1 and d(x₀, x_j) ≤ 8·j. Also
D_{N_{x_j}} = D_{N_{x₀}} − (defender placements from x₀ to x_j exclusive of
x_j) ≤ D_{N_{x₀}} − j. By protection monotonicity and the growth of ghost
legality, y is protected-not-legal at N_{x₀} too. Then
d(x₀, y) ≤ 8·j + 8·(D_{N_{x₀}} − j) = 8·D_{N_{x₀}} — contradicting x₀'s
clearance (x₀ is ghost-legal, so its clearance is enforced by (Z5)).
(b) If protected y becomes real-legal while ghost-illegal, y ∈ B₈(x_m) for
the last chain stone x_m; then d(x_m, y) ≤ 8 ≤ 8·D_{N_{x_m}} (D ≥ 1 for
any defender placement before T) contradicts x_m's clearance from (a). ∎

**D12 (coupling; replaces the extension order — per review R1).** T3's
proof maintains a *coupling* between the real play (position R) and a walk
on 𝒞 (ghost position G): equal ply index, equal mover/budget, identical
attacker stones, and the defender-stone differences defined canonically as

  X := Stones_D(R) ∖ Stones_D(G),   Y := Stones_D(G) ∖ Stones_D(R)

(disjoint by construction), with invariants: (i) every x ∈ X entered at a
dismissal step whose node certified x ∉ Prot; (ii) every y ∈ Y entered as
a ghost *filler* — a searched placement of its node; (iii) X ∩ Prot(N) = ∅
at every visited node N (by (i) + protection monotonicity); (iv) every
x ∈ X is (Z5)-clear in the sense of L9 — enforced by (Z5) for ghost-legal
dismissals, inherited via L9(a) for ghost-illegal ones. A *dismissal
history* set X̂ ⊇ X (cells ever added to X, never removed) is carried for
the anchor argument, which refers to "ever dismissed" rather than current
membership.

---

## 6. The main theorem

**T3 (horizon-parameterized dismissal soundness). [PROVEN]** *(R4-ruled
caveat: proven for valid zone-carrying certificates obeying the D9 grammar
and satisfying (Z1), (Z2), (Z4), (Z5), with exact D_N and the full — not
sharpened — defender-placement budget. Four-round hostile review trail in
§13.)*
Let 𝒞 be a zone-carrying certificate (D9–D12) for "A wins from P₀ by ply
T". Then A wins from P₀ — against *every* defender strategy, over the full
legal move set — by ply ≤ T. Hence every dismissal of 𝒞 is sound.

*Proof.* We construct A's real-game strategy by coupling the real play
(position R, initially P₀) to a walk on 𝒞 (ghost position G, initially P₀),
maintaining D12's invariants: equal ply/mover/budget, identical attacker
stones, X and Y the canonical defender-stone differences, (i)–(iii).
Leaves are processed the moment the walk reaches them (before any further
case analysis). Induction on T − (current ply).

**Step O (ghost at an OR node, designated placement c).**
c is empty in ghost (𝒞 valid). c ∉ X: c ∈ core ⊆ Prot of every ancestor
(protection monotonicity) and X avoids Prot by (iii). c ∉ Y: Y-cells are
ghost D-stones, c is ghost-empty. So c is empty in real. c is legal in
real: by (Z4) c is within distance 8 of an attacker/root stone of its
predecessor position; attacker and root stones are *identical* in R and G,
so the same justification holds in real. A plays c in the real game; ghost
plays it too. Attacker stones stay identical; X, Y unchanged.
*Termination.* If the placement completes the named witness window W* in
ghost: W* ⊆ core, so X ∩ W* = ∅ (iii), and Y ∩ W* = ∅ (W* is A-complete in
ghost, hence D-free there); masks agree (L5(b)), so the real placement
completes W* at the same ply ≤ T — **A wins**. If the real placement
completes some *other* window that ghost does not (possible only on a
window D-blocked in ghost but not in real, i.e. meeting Y — L5(d)/L6(b)):
the real game terminates immediately with **A the winner** at a ply < T;
return success (per review R1 item 9, this branch exits rather than
recursing). Otherwise recurse on the child.

**Step A (ghost at an AND node N, real defender plays a real-legal d).**
The split below is exhaustive over real-legal d: d is ghost-occupied, or
ghost-empty ∧ ∈ S(N), or ghost-empty ∧ ∉ S(N).

*Filler subroutine (used by A2, A3).* Ghost must consume a defender ply:
pick any d₀ ∈ S(N) — S(N) ≠ ∅ since a non-leaf AND node has children, and
every S-cell is ghost-legal hence ghost-empty. Ghost plays d₀. If d₀ was
real-occupied (d₀ ∈ X), the difference sets update to X ∖ {d₀} (ghost has
caught up); otherwise Y ∪= {d₀}. Either way the canonical differences of
D12 are recomputed and the invariants are preserved: a new Y-cell is a
searched placement of its node (ii); X only shrinks here.

*(A1) d ghost-empty and d ∈ S(N).* Both real and ghost play d; descend to
the d-child. Differences unchanged.

*(A2) d ghost-occupied.* Then d is a ghost D-stone (ghost A-stones are
real A-stones, and d is real-empty), i.e. d ∈ Y. Real plays d; Y loses d.
Run the filler subroutine; descend to the d₀-child.

*(A3) d ghost-empty and d ∉ S(N) — a dismissal.* First, d ∉ Prot(N): if
d ∈ Legal(P_N), this is (Z2) (protected legal cells are searched); if
d ∉ Legal(P_N) — real-legal only through the X-expanded frontier — then by
L9(b), d cannot be a protected-not-legal cell, and d cannot be a protected
*legal* cell by assumption, so d ∉ Prot(N). Second, invariant (iv) for d:
if d ∈ Legal(P_N), the (Z5) band condition means a non-clear d would have
been searched; if d ∉ Legal(P_N), clearance is inherited by L9(a). Real
plays d; X ∪= {d}, X̂ ∪= {d} — invariants (i) and (iv) hold as just shown.
Run the filler subroutine; descend to the d₀-child.

**No real defender completion before ply T (anchored argument with mask
inclusion).** Suppose some window W becomes D-complete in the real play at
a ply < T. Along the coupling, real D-stones in W at any moment are
(ghost D-stones in W) ∪ (X ∩ W) minus (Y ∩ W), so

  **(MI)** cnt_D(W, R) = cnt_D(W, G) + |X ∩ W| − |Y ∩ W|
           ≤ cnt_D(W, G) + |X̂ ∩ W|.

*Case (a): no W-fill was ever dismissed* (X̂ ∩ W = ∅ throughout). Then by
(MI), cnt_D(W, G) ≥ 6 at that ply: the *ghost* position — on an actual
certificate line — contains a D-complete window before T. D9's grammar
forbids defender-terminal edges and requires exact successors, so no valid
certificate line contains a D-completion. Contradiction.
*Case (b): some W-fill was dismissed.* Let x* be the first ever
(membership in X̂), at node N*. Step A3 established x* ∉ Prot(N*). W is
D-alive at N* (it is D-completable later, and alive-for-D persists
backward, L4), and x* is one of W's empties. Had W qualified for the
completion-guard term — cnt_D(W, P_{N*}) + D_{N*} ≥ 6 — then W's empties,
including x*, would lie in Prot(N*): contradiction. Hence
cnt_D(W, P_{N*}) + D_{N*} < 6. Before x*, X̂ ∩ W = ∅, so by (MI)
cnt_D(W, R_{N*}) ≤ cnt_D(W, P_{N*}). The plies are synchronized, so all
real defender placements from x* onward and strictly before T number at
most D_{N*}. By L7 anchored at N*, W cannot reach 6 D-stones before ply T.
Contradiction. ∎

**Leaf transfer (rewritten per review R2 — the post-leaf plies are argued
directly in the real game; the coupling is not extended past the leaf).**
*(WIN leaf.)* By D9 the leaf is a checked own_win_now position: the witness
window has count 5, or count 4 with the attacker owning both placements of
the current turn — so completion occurs within attacker-owned plies with
**no defender placement interleaved**. The witness window is core: X-free
(iii), D-free in ghost (hence Y-free), masks equal (L5(b)); its empties are
within distance 2 of attacker stones (L1), hence legal in real (shared
attacker stones); they are real-empty (X avoids core; Y-cells are absent in
real, i.e. *more* empty). The attacker fills them; the real game completes
at the leaf's resolution ply ≤ T.
*(LOSS leaf, adaptive.)* The leaf names 𝒯 with hitting number > b. At the
leaf, the real and ghost masks and empty-sets of every W ∈ 𝒯 are
**identical**: 𝒯-windows are core, so X avoids them (iii); they are
A-alive in ghost, so they contain no ghost D-stones and in particular no
Y-cells; and A-stones are shared. Hence the real hitting number over 𝒯
equals the ghost's, > b.
The real defender's own_win_now at the leaf is excluded by a leaf-time
split (review R3 — the coupling stops at the leaf, so the anchored
argument is applied *at* the leaf, not extended past it). Let W be a
putative real own-win window. If X̂ ∩ W = ∅: leaf-time (MI) gives
cnt_D(W, R) ≤ cnt_D(W, G), with the same A-mask and the same budget, so
real own_win_now would imply ghost own_win_now — contradicting the leaf's
verifier-checked ¬own_win_now (D9). If X̂ ∩ W ≠ ∅: the anchored case (b)
at the first dismissed x* ∈ W is pure counting over *all* real defender
placements before T, independent of whether the coupling continues, and
forbids W reaching count 6 before T; an own-win at the leaf would complete
W within the defender's current turn, at a ply < resolution ≤ T —
contradiction.
Now let the real defender play any complete remainder H of his turn (D9
convention). Since |H| ≤ b < hitting number(𝒯), some W_H ∈ 𝒯 has
E(W_H) ∩ H = ∅; W_H remains A-alive (its cells avoid H by choice and its
leaf masks were exactly the ghost's); its ≤ 2 empties are within distance
2 of *shared, permanent* attacker stones (L1), hence legal and empty in
the real game when the attacker moves. The attacker completes W_H within
his following turn (count-5 → one placement; count-4 → both), finishing by
leaf-ply + b + 2 ≤ T (D9's clock). H may include placements anywhere —
hits on other 𝒯-windows, quiet moves, frontier moves — the argument uses
only |H| ≤ b and permanence. ∎ (T3)

**T4 (zone reading of T3). [PROVEN]** *(Same R4 caveat as T3.)* Corollary:
at an AND node the sound searched set is

  Z(N) = Legal(P_N) ∩ [ hitting(P_N) ∪ Prot(N) ∪
         B_{8·D_N}( Prot(N) ∖ (Legal(P_N) ∪ Stones(P_N)) ) ],

where hitting(P) := the union of the empty-sets of the A-threat windows of
P (the cells of D6's `hit(P)` members) — i.e. hitting cells, protected
cells (core + completion-guard window empties), and the horizon-scaled
(Z5) frontier band. The completion-guard term uses D_N = all defender
plies before the horizon; the *sharpened* F-version counting only non-hit
placements is **not** proven (Open Problems §12.1). The frontier band is
empty whenever Prot(N) ⊆ Legal(P_N) ∪ Stones(P_N) — the typical case,
since witness windows cluster near the action.

**L10 (short-range attacker-core containment). [PROVEN]** In a certificate
whose attacker placements are all threat-creating (each lands in a window
that, immediately before the placement, is A-alive and holds ≥ 3 attacker
stones), an attacker placement cell that is the attacker's k-th future
placement from node N lies, for k ≤ 3, in a window containing
≥ 3 − (k − 1) ≥ 1 *current* attacker stones of P_N — hence inside the
A-touched-window term. (L10 covers attacker-placement core cells; the
named-witness component of core is a separate, certificate-known set.)
*Proof.* Its window holds ≥ 3 attacker stones at placement time, of which
at most k − 1 are future (one per earlier future placement); so ≥ 3−(k−1)
are current. For k ≤ 3 that is ≥ 1. ∎ (From the 4th future placement on,
attacker core cells can lie in currently-virgin windows — review R2's
setup-cell example — and must be supplied to the searched set from the
certificate's core explicitly.)

**T5 (static-set coverage for short horizons — rescoped per review R2).
[PROVEN]** *(R4 caveat: a coverage theorem only, under all joint
hypotheses — D_N ≤ 3, at most three remaining threat-creating attacker
placements, witness-window cells covered by A-touched windows, empty
frontier band; otherwise the static set is only a candidate generator.)*
For D_N ≤ 3, the completion-guard term of Z(N) is contained in radius-3 of
defender stones (L1 with k ≥ 3); hitting ⊆ radius-2 (T1); and, for
certificates with ≤ 3 remaining attacker placements and threat-creating
attacker moves, attacker core cells lie in A-touched windows (L10). Hence
the static set r3 ∪ {empties of A-touched alive windows} — where r3 :=
{ legal empties within distance 3 of any stone } — covers Z(N) whenever
additionally all witness-window cells are A-touched-window cells and the
frontier band is empty. In general the searched set must include
core(𝒞, N) ∩ Legal(P_N) explicitly — it is certificate-known — and the
static set is a **candidate generator**, not a certified zone by itself.

**T6 (the forced-tree regime — proof replaced per reviews R2/R3).
[PROVEN]** *(R4 caveat: proven for valid D9 certificates whose internal
AND nodes have verifier-checked mhs = b and ¬own_win_now; the sufficient
searched set is hitting(P_N) ∪ (core(𝒞,N) ∩ Legal(P_N)), with the
auxiliary refutation measured against the same certified horizon T.)* If at every
AND node of 𝒞 the defender's λ¹ analysis gives mhs = b and (per D9's
grammar) ¬own_win_now for the defender, then S(N) = hitting(P_N) ∪
(core(𝒞,N) ∩ Legal(P_N)) suffices: the completion-guard and frontier terms
are unnecessary.
*Proof (framed at the first dismissed reply, per review R3 — before it,
the real play follows certificate lines exactly).* Let d be the first
dismissed reply, at a node with budget b; d is non-hitting (hitting is
searched). Attach the auxiliary refutation:
- **b = 2:** the successor is a defender node with b' = 1 and the untouched
  threat family still needing mhs = b = 2 hits: mhs > b', a defender LOSS
  leaf with the current threat family as witness family.
- **b = 1:** the successor is the *attacker* to move with budget 2 and at
  least one surviving A-threat: an attacker WIN leaf (count-4 at b = 2, or
  count-5, is own_win_now).
The auxiliary leaf's witness windows are the node's current A-threat
windows. Their cells are covered by the searched set: they are hitting
cells of the node (searched by hypothesis), so the auxiliary branch's
protection needs no appeal to D10-core of the original subtree. Its
resolution fits inside the original T: following instead a searched
minimum hitting reply (pair at b = 2, common hit at b = 1) kills every
current A-threat and defender placements create no new A-threats, so the
original certificate cannot resolve on the first following attacker ply —
its T already reaches the second attacker ply, exactly what the auxiliary
branch needs (leaf-ply + b' + 2 at b = 2; the attacker's own turn at
b = 1).
It remains to exclude a defender completion before the auxiliary
resolution. By D9's grammar ¬own_win_now holds at AND nodes, so any
D-alive window W has cnt_D(W) ≤ 3 at b = 2 nodes and ≤ 4 at b = 1 nodes
(count 4 at b = 2, count 5 anywhere, is own_win_now). From the dismissal
onward the defender places: at b = 2, the dismissal plus the single
placement of the b' = 1 LOSS contract — the same one placement, counted
once — total 2, reaching at most 3 + 2 = 5 < 6; at b = 1, the dismissal
alone before the attacker's turn — at most 4 + 1 = 5 < 6. No defender
window reaches 6 before the attacker's completion. ∎

---

## 7. The n-relevance closure operator

**D13 (closure — repaired per review R1).** For a position P with defender
to move and D = 𝔇(P, T) defender placements before the horizon:

  R(P, D) = Legal(P) ∩ [ hitting(P) ∪ 𝒜(P) ∪ ℬ(P, D) ],  for D ≤ 5;
  R(P, D) = Legal(P),                                     for D ≥ 6,

where 𝒜(P) = { empties of **A-touched** alive windows (cnt_A ≥ 1,
cnt_D = 0) } and ℬ(P, D) = { empties of D-alive windows with
cnt_D ≥ 6 − D } (a *touched-window* query for D ≤ 5, hence finite).
**Scope (per review R2): R is a candidate generator, not by itself a
certified zone.** 𝒜 covers attacker core cells only within the L10 range
(≤ 3 remaining attacker placements); deeper certificates place attacker
setup cells outside every R-term. Certified sufficiency is always
R ∪ (core(𝒞, N) ∩ Legal(P_N)) ∪ the (Z5) band — the two extra terms are
certificate-known and verifier-enumerable. The D ≥ 6 fallback is
conservative (T3 permits dismissing deep-interior cells even at D ≥ 6 when
they lie in no D-alive window AND outside hitting, core, and the (Z5)
band — no-D-alive-window alone is necessary, not sufficient; the
implementation forgoes that optimization). All certified-sufficiency
statements here presuppose a valid D9 certificate and separately verified
(Z4).

**T7 (closure + certificate extras coverage — corollary of T3/T4/L10).
[PROVEN]** *(R4 caveat: a coverage theorem — the displayed set satisfies
(Z1), (Z2), (Z5); certified dismissal soundness additionally presupposes a
valid D9 certificate, separately verified (Z4), exact D_N, and the full
defender-placement budget.)*
Searching, at every AND node, R(P, D) ∪ (core(𝒞, N) ∩ Legal(P_N)) ∪ the
(Z5) band satisfies (Z1), (Z2), (Z5). The two certificate terms are
verifier-enumerable; core ∩ Legal is typically *redundant* (already inside
R, e.g. the next designated attacker move under L10's hypotheses) rather
than empty, and the band is empty when protected territory is already
legal. R alone costs one window-store pass
per node, O(#touched windows).

The solver's practical loop: search with R (a heuristic candidate
generator); when a proof is found, the verifier recomputes **(Z1), (Z2),
(Z4), (Z5)** exactly, per node, against the actual certificate — including
the core and band terms; any violation ⇒ UNKNOWN. T3 then guarantees no
false WIN can survive verification **regardless of how the search
trimmed** — search trimming affects completeness only, never soundness.

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

- The solver's defender generator is R(P, D) (§7): measured mean 147 cells
  vs 302 legal on random midgame positions (survey doc §9), smaller in
  forced regions, provably sufficient per T3/T4 once the verifier checks
  (Z1), (Z2), (Z4), (Z5) on the certificate.
- Fully-forcing certificates (T6): defender sets = hitting cells plus the
  certificate's own core cells, at any depth — the deep-tunnel regime that
  motivated the program.
- D ≥ 6 certificates: the completion guard covers every legal cell in any
  D-alive window, leaving only no-D-alive-window interior cells dismissible
  — the G2 boundary is a theorem, not a caveat.
- P3 commutation (§11) additionally halves turn-level branching by
  unordered-pair deduplication, orthogonally to the zone.

**Mechanized validation of the closure (T7), executed** (script
`scripts/_tss_moveset_zone_experiments.py`, mini-model validated
move-for-move against `hexo_engine`): (i) the closure-restricted solver
does **not** claim the engineered junction position (the known-adversarial
G1 instance): the junction cell is in the closure and the defender survives
(1,396 nodes, uncapped); (ii) random bounded model check: 91 attackable
positions, 52 closure-restricted WIN claims, **0 divergences** against the
full legal move set at matched horizon (139 s); the earlier r=2 experiments
(survey doc §9) provide the negative controls — the same harness *does*
produce false WINs for unsound restrictions.

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
  lies in an attacker-touched alive window (L4 there), hence inside this
  document's closure set. Mandatory placements with no monitored window
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

1. **Sharpened budget (F + H_W).** T4 counts *all* defender placements
   before the horizon. The sharper count — quiet placements F plus
   per-window forced-hit capacity H_W — requires per-branch worst-case
   bookkeeping over the certificate DAG and remains open; T3 is sound
   without it, at the cost of wider zones for deep loose certificates.
2. **Sharper frontier bands.** The (Z5) band radius 8·D_N is worst-case;
   chains cost the defender tempo the completion guard already restricts,
   and a joint tempo-and-distance accounting should shrink the band
   substantially. Unwritten.
3. **Pairing at the threshold.** T8.1 leaves k = 7, g = 3 open (density
   exactly 1); irrelevant to Hexo (k = 6) but of independent interest.
4. **Formalization.** All objects here are finite combinatorics; a
   Lean/Coq formalization of §§1–8 is feasible and would yield a verified
   reference checker for (Z1), (Z2), (Z4), (Z5).

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
