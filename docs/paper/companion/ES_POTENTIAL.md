# Erdős--Selfridge potential blocking for Hexo

> **Status.** The source model, the fixed-family `(2:2)` theorem, the
> one-cycle theorem, the cumulative-activation theorem, and the finite-region
> theorem below are **PROVEN**.  The optimal base within the Beck exponential
> danger comparison is `lambda = sqrt(3)` and, when Defender moves first, the
> threshold is the strict inequality `Phi < 1`.
>
> The superficially stronger statement
>
> `current touched-window Phi < 1 => touched-window greedy blocks forever`
>
> is **NOT PROVEN AND IS NOT A VALID CONSEQUENCE OF
> Erdős--Selfridge--Beck**.  Its proposed invariant is false: previously blank
> windows enter the touched family during Attacker's turn, and one legal clean
> placement can add `2/sqrt(3) > 1` to `Phi`.  Section 5 gives the exact source
> term and the rigorous replacements derived here.  No claim in
> this document assumes the unproved relevance-zone theorem in
> `docs/PLAN_TSS_MOVESET_ZONES.md` Section 8.
>
> A second requested statement also needs a necessary qualification.  Every
> *positive-reduction* greedy placement is an empty of a currently
> Attacker-alive window and is legal.  If no such window remains, the abstract
> blanket `(2:2)` model still requires the rest of Defender's turn.  Those
> placements must be
> fillers and cannot satisfy the literal alive-window condition.  Section 7
> proves this obstruction.

---

## 1. Source-verified rule model

### Definition 1 (ideal Hexo lattice and windows)

The formal board is the axial lattice `Z^2`.  For `x=(q,r)` and `y=(q',r')`,
put

\[
 d(x,y)=\max\{|q-q'|,|r-r'|,|(q-q')+(r-r')|\}.
\]

The three positive axis vectors are

\[
 Q=(1,0),\qquad R=(0,1),\qquad QR=(1,-1).
\]

A *window* is a set

\[
 W(a,v)=\{a,a+v,\ldots,a+5v\},
 \qquad v\in\{Q,R,QR\}.
\]

For a cell `x`, write `Omega(x)` for the windows containing `x`.

### Lemma 1 (window incidence) **[PROVEN]**

Every cell belongs to exactly 18 windows.  A placement can create at most 18
new Attacker-alive windows, and the bound is attained when all 18 incident
windows were previously stone-free.

*Proof.*  On each of the three axes, `x` can occur at any one of the six
offsets `0,...,5`, giving `3*6=18` distinct `(start,axis)` keys.  A window that
becomes Attacker-alive for the first time on a placement at `x` must contain
`x`; hence it is in `Omega(x)`.  If every member of `Omega(x)` was stone-free,
all 18 become Attacker-alive.  QED.

This is exactly the loop and key construction in
`packages/hexo_engine/rust/src/tactics.rs:13-17,21-52,443-486,511-514`.

### Definition 2 (blanket position and turn order)

A nonterminal blanket position `P` is a finite partial colouring of `Z^2` by
Attacker stones `A` and Defender stones `D`, together with the player and
placement phase to move.  Stones are permanent.  A window is

- *Attacker-alive* if it contains at least one `A` and no `D`;
- *stone-free* if it contains neither colour; and
- *dead for Attacker* if it contains a `D`.

For an Attacker-alive window define

\[
 s_P(W)=|W\cap A|,\qquad e_P(W)=6-s_P(W).
\]

Attacker wins immediately after any placement for which some window has
`s_P(W)=6`.  Defender's only objective in the blanket game is to prevent that
event forever.  A normal turn consists of two consecutive placements by one
player.  This document studies epochs at which Defender is at `FirstStone`,
so the cyclic order is

\[
 D_1,D_2,A_1,A_2,D_1,D_2,A_1,A_2,\ldots . \tag{1}
\]

If `A_1` wins, `A_2` is not played.

The engine agrees: `TurnPhase::FirstStone` leaves the same player at
`SecondStone`, a nonwinning second stone changes player, and the win test
precedes either transition
(`packages/hexo_engine/rust/src/state.rs:46-56,283-337`).  The shared threat
code consequently assigns budget `B=2` at `FirstStone` and `B=1` at
`SecondStone`
(`packages/hexo_models/rust/src/threats_shared.rs:45-53,155-171`).  The
one-stone opening is exceptional and is not part of (1).

The actual engine also lets Defender win by making six.  Ignoring that
objective is conservative: an actual Defender win only terminates the game
earlier in Defender's favour.

### Lemma 2 (legality and finiteness) **[PROVEN]**

At every finite, nonterminal position:

1. the set of Attacker-alive windows is finite, of size at most `18|A|`;
2. every empty of an Attacker-alive window is legal under Hexo's radius-8
   rule; and
3. the potential sums defined below are finite, even though the ideal board
   is infinite.

*Proof.*  Every Attacker-alive window contains an Attacker stone, and each
Attacker stone is in exactly 18 windows, proving (1).  If `x` is empty in an
Attacker-alive window `W`, choose an Attacker stone `a` in `W`.  The two cells
are at most five steps apart on one axis, so `d(a,x)<=5<8`; thus `x` is an
empty cell within the legal radius of an existing stone.  This proves (2),
using `LEGAL_RADIUS=8` from
`packages/hexo_engine/rust/src/legal.rs:17-18,114-145` together with actual
validation in `packages/hexo_engine/rust/src/rules.rs:10-44`.  Item (3) follows
from (1).  QED.

An infinite play causes no compactness problem.  Every alleged Attacker win
occurs at a finite placement index, so induction over all finite prefixes is
enough.  The formal theorems use `Z^2`; the current Rust coordinate carrier is
`i16`, so an implementation must separately exclude coordinate overflow.

---

## 2. Potential and exact one-placement calculus

### Definition 3 (residual hypergraph, potential, and danger)

Let `F` be a finite labelled family of Attacker target windows.  At a position
`P`, let

\[
 \mathcal L_F(P)=\{W\in F:W\cap D=\varnothing\}
\]

be the surviving *window labels*.  For each surviving label use the residual
edge

\[
 E_P(W)=W\setminus A.
\]

The fixed-family residual hypergraph is the indexed family

\[
 \mathcal H_F(P)=\bigl(E_P(W)\bigr)_{W\in\mathcal L_F(P)}.
\]

This is a labelled multi-hypergraph: two distinct labels are still counted
twice if their residual sets coincide.

Set

\[
 \lambda=\sqrt3,
 \qquad
 \Psi_F(P)=
 \sum_{W\in\mathcal L_F(P)}\lambda^{-|E_P(W)|}. \tag{2}
\]

For an empty cell `x`, its *danger* is

\[
 d_F(x;P)=
 \sum_{\substack{W\in\mathcal L_F(P)\\x\in E_P(W)}}
 \lambda^{-|E_P(W)|}. \tag{3}
\]

For the dynamic family of all currently Attacker-alive windows, write

\[
 \Phi(P)=
 \sum_{\substack{W:\ W\cap A\ne\varnothing\\W\cap D=\varnothing}}
 \lambda^{-e_P(W)}. \tag{4}
\]

The all-empty windows are deliberately absent from (4).  Including every
all-empty window would add `lambda^-6=1/27` for infinitely many windows and
make the sum diverge.

### Lemma 3 (exact placement changes) **[PROVEN]**

For the fixed residual family immediately before a placement at `x`:

1. a Defender placement decreases `Psi_F` by exactly `d_F(x)`; and
2. an Attacker placement increases `Psi_F` by exactly
   `(lambda-1)d_F(x)`.

Moreover, if the maximum danger before an Attacker placement is at most `S`,
then after that placement every remaining cell has danger at most `lambda S`.

*Proof.*  Defender deletes exactly the surviving labelled edges containing
`x`, whose weights sum to (3).  Attacker removes `x` from each such edge, so
each affected weight changes from `lambda^-e` to
`lambda^{-(e-1)}=lambda*lambda^-e`; its increase is therefore
`(lambda-1)lambda^-e`.  For the last assertion, fix a remaining empty cell
`y`.  Every old edge contributing to the new danger of `y` either keeps its
weight or has its weight multiplied by `lambda`.  Term by term,
`d_new(y)<=lambda*d_old(y)<=lambda S`.  QED.

### Definition 4 (sequential greedy Defender and delayed enrollment)

On each of Defender's two placements, recompute (3) after the preceding
placement and choose an empty cell of maximum danger among the labels being
scored.  Since positive danger is supported on the finite union of residual
edges, a maximum exists.  Fix a deterministic tie-breaking rule once and for
all.

There are two variants:

1. *full-family greedy* scores every surviving label in a fixed family; and
2. *delayed-enrollment greedy* scores a label only once it is
   Attacker-alive.  Dynamic touched-window greedy and region-target greedy use
   this second variant.

Whenever the scored labels are Attacker-alive, every positive-danger
candidate is legal by Lemma 2.  If every scored danger is zero, use the
following deterministic legal *filler*.  Choose an occupied cell with maximum
`q` coordinate (breaking ties by maximum `r`) and place one step from it in
the positive `Q` direction.  The neighbour is empty by maximality and is
legal at distance one.  Recomputing the rule supplies another filler if
needed.  Thus a filler always exists at a finite post-opening position.  All
strategy and reserve quantifiers below include this fixed tie and filler
policy.

### Lemma 4 (locality of alive-only positive greedy placements) **[PROVEN]**

Suppose every label being scored is currently Attacker-alive, as under
delayed enrollment or when every surviving fixed-family label contained an
Attacker stone initially.  Then every positive-danger greedy placement is an
empty cell of a currently Attacker-alive monitored window, and hence is legal.
A Defender placement never increases `Psi_F` or `Phi` at the instant it is
made.

*Proof.*  Positive danger means that some scored residual edge contains the
chosen cell.  By hypothesis its labelled window contains Attacker stones and
no Defender stone, so it is currently Attacker-alive.  Lemma 2 gives
legality.  Defender only deletes alive window terms.  QED.

The alive-only hypothesis is essential: proactive full-family greedy may
score a virgin window.  The word *positive* also cannot be removed; Section 7
gives the exact counterexample.

---

## 3. The exact `(2:2)` Beck inequality

The classical Erdős--Selfridge theorem is the `(1:1)` case.  Beck's biased
criterion uses

\[
 \lambda=(q+1)^{1/p}
\]

for Maker bias `p` and Breaker bias `q`.  Here `p=q=2`, so
`lambda=sqrt(3)`.  The following proof is specialized to Hexo's consecutive
placements and does not quote the unbiased theorem.

### Lemma 5 (one full fixed-family round) **[PROVEN]**

Let Defender make two sequential greedy placements in a fixed residual
family, followed by at most two Attacker placements.  Assume either that all
positive-danger cells are legal, or, as in the applications below, that every
surviving label is Attacker-alive.  Then the fixed-family potential at the
next Defender epoch is no larger than at the preceding Defender epoch.

*Proof.*  Let `delta_1` and `delta_2` be the exact reductions made by the two
greedy Defender placements.  After both, let

\[
 S=\max_x d_F(x)
\]

over the remaining empty cells.  Deleting edges never increases any cell's
danger.  The first greedy maximum and the second greedy maximum are therefore
both at least `S`; hence

\[
 \delta_1+\delta_2\ge2S. \tag{5}
\]

By Lemma 3, Attacker's first placement increases potential by at most
`(lambda-1)S`.  Afterwards every danger is at most `lambda S`, so Attacker's
second placement increases potential by at most
`(lambda-1)lambda S`.  The total increase is at most

\[
 (\lambda-1)(1+\lambda)S
   =(\lambda^2-1)S
   =2S. \tag{6}
\]

Equations (5)--(6) show that the round change is nonpositive.  If Attacker
wins or otherwise stops after the first placement, omitting the second
nonnegative increase only strengthens the inequality up to that placement.
QED.

### Theorem 1 (fixed-family Hexo blocking) **[PROVEN]**

Let `P_0` be a nonterminal Defender-`FirstStone` position, and let `F` be any
finite fixed family of windows that are Attacker-alive at `P_0`.  If

\[
 \boxed{\Psi_F(P_0)<1}, \tag{7}
\]

then sequential greedy Defender prevents Attacker from ever completing any
window in `F`.

*Proof.*  Lemma 5 and induction over Defender epochs give
`Psi_F(P_t)<=Psi_F(P_0)<1` after every completed round.  Defender placements
only decrease potential.  During Attacker's two placements potential is
monotone nondecreasing.  If a monitored window is completed on Attacker's
first or second placement, its residual edge is empty and its single term is
`lambda^0=1`; the total potential at that placement is at least 1.

For a second-placement completion this contradicts the next-epoch bound.  For
a first-placement completion, Lemma 5 with only that one Attacker placement
already bounds the potential by

\[
 \Psi_F(P_t)-2S+(\lambda-1)S<1,
\]

also a contradiction.  Thus no monitored window is ever completed.  Every
finite prefix is covered, so the conclusion holds for the entire infinite
game.  QED.

### Corollary 1 (why the threshold is 1, not `1/3`) **[PROVEN]**

At a full Defender turn the threshold is `Psi<1`.  If Attacker instead gets
the first pair before any Defender placement, the sufficient threshold is

\[
 \Psi<\lambda^{-2}=\frac13, \tag{8}
\]

because an Attacker pair can multiply each term by at most `lambda^2=3`.
If only one Defender placement remains before the next Attacker pair, the
sufficient threshold is `Psi<2/3`.

*Proof.*  Only the last assertion needs calculation.  Let the sole greedy
reduction be `delta`, and let the post-Defender maximum danger be `S`.  Then
`S<=delta` because the old greedy maximum was `delta`, and
`S<=Psi-delta` because a cell danger is at most total residual potential.
The next Attacker pair adds at most `2S`.  Thus

\[
 \Psi'\le \Psi-\delta+2\min\{\delta,\Psi-\delta\}
          \le\frac32\Psi.
\]

So `Psi<2/3` implies `Psi'<1`.  QED.

These phase thresholds are summarized by `(r+1)/3` for `r=0,1,2` Defender
placements remaining before the next Attacker pair.

### Proposition 1 (optimal base within this greedy proof) **[PROVEN]**

Among potentials of the form `sum lambda^-e` proved by the danger comparison
above, `lambda=sqrt(3)` gives the weakest hypotheses and is optimal.

*Proof.*  For general `lambda>1`, (6) is
`(lambda^2-1)S`.  Two Defender placements guarantee only `2S`, so the proof
requires `lambda^2-1<=2`, or `lambda<=sqrt(3)`.  For every unfinished edge,
`lambda^-e` decreases as `lambda` increases.  The largest admissible base,
`sqrt(3)`, therefore minimizes the tested potential and maximizes the set of
positions satisfying (7).  QED.

### Proposition 2 (strictness for static Hex-window families) **[PROVEN]**

The strict inequality in (7) cannot in general be replaced by `<=1`, even for
length-6 Hex windows.

*Proof.*  Choose three length-6 windows so that the distance between every
cell of one and every cell of another exceeds 5.  In particular, no
length-6 window can intersect two targets.  Put four Attacker stones in each
target and leave two empty cells in each.  For every other window containing
one of those Attacker stones, put a Defender stone in a cell outside the three
target windows.  This is possible: only finitely many other windows are
involved; each intersects at most one target; and a distinct length-6 window
intersecting one target has a cell outside it.  The only Attacker-alive windows
are now the three targets, so

\[
 \Phi=3\lambda^{-2}=3\cdot\frac13=1.
\]

Defender's two placements can kill at most two of the pairwise separated
targets.  Attacker places in the two empties of a survivor and wins on the
second placement.  QED.

This is a static blanket-game construction establishing strictness for the
abstract fixed Hex-window hypergraph criterion.  It is not claimed here as a
separately enumerated, material-balanced engine history, so it does not prove
sharpness after restricting the state space to reachable Hexo nodes.

---

## 4. Why current touched windows are not a fixed family

Theorem 1 is a forever theorem only for the labelled family `F`.  It does not
say that a window which was stone-free at `P_0` is blocked.  That distinction
is decisive on the infinite Hexo board.

### Definition 5 (Defender epochs and birth mass)

Let `P_t` be the position at the start of Defender's `t`-th full turn.  During
the following Attacker pair (or its prefix before a terminal placement), let
`N_t` be the windows which

1. were stone-free at `P_t`;
2. were not hit by the intervening Defender pair; and
3. first become Attacker-alive during that pair.

No Defender placement occurs after the pair begins, so these windows remain
alive through its end.  For \(W\in N_t\), let
\(a_t(W)\in\{1,2\}\) be the number of that pair's stones in `W` by the end of
the pair or the terminal prefix.  Define the exact *birth mass*

\[
 C_t=\sum_{W\in N_t}\lambda^{-(6-a_t(W))}. \tag{9}
\]

Equivalently, if `u_{1,t}` windows are born with one stone and `u_{2,t}` with
two, then

\[
 C_t=\frac{u_{1,t}}{9\sqrt3}+\frac{u_{2,t}}9. \tag{10}
\]

If a play terminates, set all later source terms to zero.  This makes the
uniform sums in Theorem 3 well-defined before safety has been proved.  The
recurrence below is invoked only for rounds that actually reach `P_{t+1}`.

### Lemma 6 (exact dynamic recurrence) **[PROVEN]**

If Defender greedily scores all windows alive at `P_t` and the round finishes
without a terminal placement, then

\[
 \boxed{\Phi(P_{t+1})\le\Phi(P_t)+C_t}. \tag{11}
\]

*Proof.*  Label the windows alive at `P_t` as the old cohort.  Lemma 5 says
that the total contribution of surviving labels from this cohort at
`P_{t+1}` is at most `Phi(P_t)`.  A window alive at `P_{t+1}` but absent from
the old cohort cannot have contained a Defender stone or an Attacker stone at
`P_t`, because stones are permanent.  It was therefore stone-free and belongs
to `N_t`; its exact contribution is the corresponding term of (9).  The old
and new cohorts partition the alive windows at `P_{t+1}`, proving (11).  QED.

### Lemma 7 (sharp universal source rate) **[PROVEN]**

For every Attacker pair,

\[
 u_{1,t}+2u_{2,t}\le36,
 \qquad
 C_t\le36\lambda^{-5}=\frac4{\sqrt3}\approx2.309401. \tag{12}
\]

One clean placement alone can contribute

\[
 18\lambda^{-5}=\frac2{\sqrt3}\approx1.154701>1. \tag{13}
\]

Both bounds are geometrically attainable under radius-8 legality.

*Proof.*  Each placement has 18 window incidences.  A one-stone birth uses one
of the 36 incidences and a two-stone birth uses two, proving the first
inequality.  Since `lambda=sqrt(3)<2`,

\[
 C_t=\lambda^{-5}(u_{1,t}+\lambda u_{2,t})
 \le\lambda^{-5}(u_{1,t}+2u_{2,t})
 \le36\lambda^{-5}.
\]

For attainability, let a Defender anchor be at `(-4,-4)`.  The empty cell
`x=(0,0)` is legal because its distance from the anchor is 8, but the anchor
is in no window through `x` because every such window cell is within distance
5 of `x`.  If that star is otherwise empty, placing at `x` births all 18
windows and gives (13).  Now take `y=(4,4)`.  It becomes legal from `x`, also
at distance 8.  The vector from `x` to `y` lies on none of the three axes, so
the two cells share no window.  If the second star is otherwise empty, the
pair births 36 distinct one-stone windows and attains (12).  QED.

If two clean placements share `m` virgin windows, then necessarily `m<=5`
and their exact source is

\[
 (36-2m)\lambda^{-5}+m\lambda^{-4}
 =\lambda^{-5}\bigl(36+(\lambda-2)m\bigr). \tag{14}
\]

The bound `m<=5` follows because two distinct cells share `6-d<=5` windows
when they lie at axis distance `d in {1,...,5}`, and share none otherwise.

### Counterexample 1 (the naive invariant) **[PROVEN]**

Let `P_t` contain only a Defender anchor at `(0,0)`, so `Phi(P_t)=0`.  Under
the fixed zero-danger filler policy of Definition 4, Defender places at
`(1,0)` and `(2,0)`.  Attacker then places at `x=(6,4)` and `y=(10,8)`.
Each displacement from the immediately preceding anchor is `(4,4)`, of hex
distance 8 and on none of the three window axes.  Consequently `x` is legal
from `(2,0)`, `y` is legal from `x`, neither Attacker star contains a Defender
stone, and the two Attacker cells share no window.  The pair births 36
one-stone windows, so

\[
 \Phi(P_{t+1})=36\lambda^{-5}=\frac4{\sqrt3}>1.
\]

Hence neither

\[
 \Phi(P_{t+1})\le\Phi(P_t)
\]

nor invariance of `Phi<1` is true for dynamic touched-window greedy with this
valid deterministic zero-score policy.  More generally, (13) proves that a
per-placement proof cannot omit the source term.  This is a counterexample to
the proposed ES/Beck invariant, not by itself a proof that Attacker wins the
entire blanket game or that every conceivable filler policy has the same
line.

The extra hypothesis “there is currently no Attacker window with at most two
empties” does not repair the forever invariant: it holds vacuously when
`Phi=0`, yet the same births occur.  It only makes the next Attacker pair
nonwinning in those pre-existing windows.

---

## 5. Strongest dynamic and local theorems derived here

### Theorem 2 (global one-cycle certificate) **[PROVEN]**

If Defender is at `FirstStone` and the current dynamic potential satisfies

\[
 \boxed{\Phi(P_t)<1}, \tag{15}
\]

then two sequential touched-window greedy placements prevent Attacker from
winning during the immediately following two-placement turn.

*Proof.*  Theorem 1 applied to the current alive cohort prevents completion
of any window in that cohort during the next Attacker pair.  Every other
potential winning window either already contains a Defender stone, which is
permanent, or is stone-free at `P_t`.  A stone-free length-6 window can receive
at most two Attacker stones in that pair and cannot be completed.  This
argument includes a win check after `A_1`, not merely after `A_2`.  QED.

Theorem 2 is useful but is not a forever certificate: after the pair, (11)
may place `Phi(P_{t+1})` above 1.

### Theorem 3 (cumulative-birth forever certificate) **[PROVEN]**

Fix the tie-breaking and filler policy in Definition 4 and use sequential
touched-window greedy from `P_0`.  Suppose there is a uniform constant
`B_infinity` such that, for every legal Attacker strategy `tau`, the resulting
play satisfies

\[
 \sup_{\tau}\sum_{t=0}^{\infty}C_t(\tau)\le B_\infty,
 \qquad
 \boxed{\Phi(P_0)+B_\infty<1}. \tag{16}
\]

Then Defender prevents Attacker from ever completing a window.  More
generally, if `B_T` is a uniform upper bound on the first `T` source terms,
the condition

\[
 \sup_{\tau}\sum_{t=0}^{T-1}C_t(\tau)\le B_T,
 \qquad
 \Phi(P_0)+B_T<1 \tag{17}
\]

certifies the first `T` Attacker pairs, indexed `0,...,T-1`.

*Proof.*  Induct on `t`.  The empty sum gives `Phi(P_0)<1`; Theorem 2 makes
the first Attacker pair safe.  If all earlier pairs were safe, Lemma 6 gives

\[
 \Phi(P_t)\le\Phi(P_0)+\sum_{i=0}^{t-1}C_i<1.
\]

Theorem 2 therefore makes pair `t` safe, and Lemma 6 advances the bound.  This
proves every finite prefix allowed by (17).  Under (16), the same argument
holds on the play generated against every `tau`, and every finite placement
lies in a certified prefix.  Thus no Attacker strategy wins.  QED.

Condition (16) is a source-charged sufficient repair of the online-family gap.
The term `C_t` is the exact birth mass, but (16) is conservative because (11)
discards any excess negative drift created by greedy defense.  The universal
upper bound (12) is too large to make (16) useful for an unrestricted infinite
continuation; a finite or otherwise summable activation reservoir is needed.

### Lemma 8 (delayed-enrollment target-family recurrence) **[PROVEN]**

Let `H` be a fixed finite labelled family whose members at `P_0` are either
Attacker-alive or stone-free.  At every Defender epoch, let

\[
 \Xi_H(P_t)=
 \sum_{\substack{W\in H:\ W\cap A_t\ne\varnothing\\
                              W\cap D_t=\varnothing}}
 \lambda^{-e_{P_t}(W)}
\]

and use delayed-enrollment greedy: score only the labels occurring in this
sum.  Let `C_t^H` be the birth mass, defined as in (9), of labels in `H` that
were stone-free at `P_t` and become Attacker-alive during pair `t`.  Then

\[
 \Xi_H(P_{t+1})\le\Xi_H(P_t)+C_t^H. \tag{17a}
\]

If a uniform reserve `B_H` satisfies

\[
 \sup_\tau\sum_{t\ge0}C_t^H(\tau)\le B_H,
 \qquad \Xi_H(P_0)+B_H<1,
\]

then delayed-enrollment greedy prevents Attacker from ever completing a
member of `H`.

*Proof.*  At epoch `t`, the labels contributing to `Xi_H` form a fixed old
cohort of Attacker-alive windows.  Lemma 5 makes the surviving old-cohort mass
nonincreasing across the Defender pair and following Attacker pair.  Every
label contributing at `P_{t+1}` but not in that cohort was stone-free at
`P_t`, because a Defender stone is permanent, and contributes exactly its
term in `C_t^H`.  This proves (17a).  A newly enrolled length-6 label receives
at most two Attacker stones in its birth pair and cannot be completed then.
The induction in Theorem 3, with `Xi_H` and `C_t^H` in place of `Phi` and
`C_t`, now proves the final assertion.  QED.

### Definition 6 (finite-region target family)

Fix a Defender epoch `P_0` and a finite set `R` of cells empty at `P_0`.
Assume every future Attacker placement is confined to `R`.  Define

\[
 \begin{aligned}
 \mathcal A_R&=\{W:W\cap D_0=\varnothing,\ W\cap A_0\ne\varnothing,
                       \ W\setminus A_0\subseteq R\},\\
 \mathcal V_R&=\{W:W\text{ is stone-free at }P_0,\ W\subseteq R\}.
 \end{aligned} \tag{18}
\]

The first family consists of currently alive windows that remain completable
under confinement; the second consists of virgin targets that can become
completable.  Put

\[
 \Phi_R(P_0)=\sum_{W\in\mathcal A_R}\lambda^{-e_{P_0}(W)},
 \qquad N_R=|\mathcal V_R|. \tag{19}
\]

Both families are finite.  In particular `N_R<=3|R|`, because counting
cell--window incidences gives `6N_R<=18|R|`.

At later Defender epochs, the *region-target greedy strategy* scores only
currently Attacker-alive members of the fixed target family
\(\mathcal H_R=\mathcal A_R\cup\mathcal V_R\).

### Definition 7 (refined virgin activation reserve)

For a matching `M` of pairwise vertex-disjoint two-cell subsets of `R`, let

\[
 q(M)=\left|\{W\in\mathcal V_R:
             \text{some }\{x,y\}\in M\text{ satisfies }x,y\in W\}\right|,
\]

and put

\[
 P_R=\max_M q(M). \tag{20}
\]

Define

\[
 B_R^*=N_R\lambda^{-5}
        +P_R(\lambda^{-4}-\lambda^{-5}). \tag{21}
\]

A simpler computable upper bound is obtained from

\[
 P_R\le\min\{N_R,5\lfloor |R|/2\rfloor\}. \tag{22}
\]

For certification, (20) must be evaluated exactly or replaced by a proved
*upper* bound such as (22).  A merely exhibited matching gives a lower bound
on `P_R` and is unsafe in (21).

### Lemma 9 (finite-region birth reserve) **[PROVEN]**

Over the whole continuation, the total birth mass of members of
\(\mathcal V_R\) is at most `B_R^*`, and hence at most `N_R/9`.

*Proof.*  A virgin target is enrolled only on its first Attacker placement.
Charge `lambda^-5` to each target that is ever enrolled; there are at most
`N_R`.  It receives the larger birth weight `lambda^-4` exactly when both of
its first Attacker stones are the two stones of one Attacker turn.  The pairs
of cells used on different turns are vertex-disjoint, hence form a matching
in `R`.  Every double-born target contains its birth pair, so the number of
such targets is at most `P_R`; this proves (21).

Two distinct cells lie together in at most five length-6 windows, as proved
after (14).  A matching has at most `floor(|R|/2)` pairs, which proves (22).
Finally, `B_R^*<=N_R lambda^-4=N_R/9`, since every virgin target is born with
at most two stones.  QED.

### Theorem 4 (finite-region forever theorem) **[PROVEN]**

If Attacker is forever confined to finite `R` and

\[
 \boxed{\Phi_R(P_0)+B_R^*<1}, \tag{23}
\]

then region-target greedy prevents Attacker from ever winning.  The simpler
but weaker sufficient condition is

\[
 \boxed{\Phi_R(P_0)+\frac{N_R}{9}<1}. \tag{24}
\]

Every positive-reduction placement made by this strategy is an empty of a
currently Attacker-alive member of \(\mathcal H_R\) and is legal.

*Proof.*  Apply Lemma 8 to the target family \(\mathcal H_R\), using Lemma 9
for its uniform total birth reserve.  It remains only to prove that
\(\mathcal H_R\) contains every window Attacker could complete.  Let `W` be
completed under the confinement.
It contained no Defender stone at `P_0`.  If it already contained an Attacker
stone, every initially empty cell later filled by Attacker belongs to `R`, so
\(W\in\mathcal A_R\).  If it contained no Attacker stone, it was initially
stone-free and all six cells filled later belong to `R`, so
\(W\in\mathcal V_R\).  Thus no winning target was omitted.  The placement
claim is Lemma 4 applied to the delayed-enrollment strategy.  QED.

The confinement premise is substantive.  At an unrestricted solver node it
must come from a separate certificate that excludes every future Attacker
placement outside `R`; Theorem 4 does not itself establish confinement.

Condition (23) has the requested explicit local threshold

\[
 c_R=1-B_R^*.
\]

For (24), even `Phi_R=0` requires `N_R<=8`; thus the coarse local theorem is
useful only in a highly saturated region with very few virgin target windows.
The refined reserve (21) is better when few virgin windows can be double-born.

**Why bare geometric locality is insufficient.**  A connected but infinite
region can expose fresh window stars forever.  The sharp universal per-turn
upper bound `4/sqrt(3)` is positive and nondecaying (although it need not be
attained in every region), so “Attacker stays in one region” gives no summable
bound unless the region has a finite virgin reservoir or another argument
supplies a summable `C_t`.

**Comparison with proactive virgin blocking.**  If all cells of
\(\mathcal H_R\) were
already legal and Defender were allowed to score stone-free windows, one
could include each virgin target from time zero at weight
`lambda^-6=1/27`.  The fixed-family condition would then be
`Phi_R+N_R/27<1`.  This is stronger numerically, but its greedy moves need not
lie in currently Attacker-alive windows and remote cells need not be legal.
This follows from the abstract calculus of Lemma 5 under the stated legality
hypothesis, not from Lemma 4.  It does not prove the restricted-move theorem
requested here.

---

## 6. Exact threshold arithmetic and practical domain

### Definition 8 (alive-window count profile)

For `s=1,...,5`, let `n_s` be the number of currently Attacker-alive windows
with exactly `s` Attacker stones.  At a nonterminal position,

\[
 \Phi=\sum_{s=1}^5 n_s\lambda^{s-6}. \tag{25}
\]

The weights at `lambda=sqrt(3)` are:

| Attacker stones `s` | empties | one-window weight | decimal |
|---:|---:|---:|---:|
| 1 | 5 | `1/(9 sqrt(3)) = sqrt(3)/27` | 0.064150 |
| 2 | 4 | `1/9` | 0.111111 |
| 3 | 3 | `1/(3 sqrt(3)) = sqrt(3)/9` | 0.192450 |
| 4 | 2 | `1/3` | 0.333333 |
| 5 | 1 | `1/sqrt(3) = sqrt(3)/3` | 0.577350 |

Thus the exact global test `Phi<1` is

\[
 n_1+\sqrt3 n_2+3n_3+3\sqrt3 n_4+9n_5<9\sqrt3. \tag{26}
\]

### Proposition 3 (profile limits) **[PROVEN]**

If only one count class is present, the largest allowed numbers of windows
for `s=1,2,3,4,5` are respectively

\[
 15,\ 8,\ 5,\ 2,\ 1. \tag{27}
\]

Ignoring count-1 and count-2 windows, the complete list of arithmetically
admissible `(n_5,n_4,n_3)` profiles is

\[
\begin{array}{c|c|c}
n_5&n_4&n_3\\ \hline
0&0&0,1,2,3,4,5\\
0&1&0,1,2,3\\
0&2&0,1\\
1&0&0,1,2\\
1&1&0
\end{array} \tag{28}
\]

and no others.  For any listed triple, the exact remaining allowance for the
lower counts is

\[
 n_1+\sqrt3 n_2
 <9\sqrt3-3(n_3+\sqrt3 n_4+3n_5). \tag{29}
\]

*Proof.*  Formula (27) is direct division with strict inequality.  With
`n_1=n_2=0`, divide (26) by 3 to obtain

\[
 n_3+\sqrt3 n_4+3n_5<3\sqrt3.
\]

Checking `n_5=0,1` and `n_4=0,1,2` gives exactly (28); larger values already
meet or exceed the right side.  Restoring the omitted terms gives (29).  QED.

Examples of the narrowness are worth stating plainly.  Five count-3 windows
leave too little budget even for one count-1 window.  A single clean isolated
Attacker stone has 18 count-1 windows and contributes

\[
 18\lambda^{-5}=2/\sqrt3>1.
\]

### Proposition 4 (incidence saturation bound) **[PROVEN]**

Let

\[
 I=\sum_{s=1}^5 s n_s
\]

be the number of Attacker-stone/Attacker-alive-window incidences.  Then

\[
 \Phi\ge I/18. \tag{30}
\]

Consequently `Phi<1` forces `I<=17`.

*Proof.*  For every `s=1,...,5`, direct substitution gives

\[
 \frac{\lambda^{s-6}}s\ge\frac1{18},
\]

with equality at `s=2`.  Multiply by `s n_s` and sum.  Since `I` is an
integer, `I<18` means `I<=17`.  QED.

Each Attacker stone has 18 incident window labels.  Therefore, if there are
`N_A` Attacker stones, all but at most 17 of the `18N_A` corresponding
stone--window incidences must already lie in Defender-killed windows before
the global `Phi<1` test can pass.  The domain is consequently a heavily
blocked, highly saturated part of position space, not an ordinary quiet
opening or middlegame domain.

### Corollary 2 (integer-only threshold check) **[PROVEN]**

Put

\[
 a=n_1+3n_3+9n_5,\qquad b=n_2+3n_4.
\]

Then `Phi<1` holds exactly when

\[
 b\le8\quad\text{and}\quad a^2<3(9-b)^2. \tag{31}
\]

*Proof.*  Equation (26) is `a+sqrt(3)b<9sqrt(3)`, or
`a<sqrt(3)(9-b)`.  Its right side must be positive, giving `b<=8`; both sides
are then nonnegative and may be squared without changing the inequality.
QED.

For an exact local test, define the bins `n_s` here using only the members of
\(\mathcal A_R\), so that they sum to `Phi_R(P_0)`.  For the coarse local
condition (24), replace `b` by `b+N_R`.  For the refined reserve (21), replace

\[
 a\text{ by }a+N_R-P_R,
 \qquad b\text{ by }b+P_R, \tag{32}
\]

because multiplication by `9sqrt(3)` changes the reserve into
`(N_R-P_R)+sqrt(3)P_R`.

If the implementation substitutes global alive-window bins instead, the
result remains sufficient because `Phi_global>=Phi_R`, but it is no longer an
exact characterization of the local condition.

---

## 7. Exact move-set statement and its unavoidable exception

### Proposition 5 (strongest valid restricted-move claim) **[PROVEN]**

At each Defender placement for which some monitored alive window remains,
sequential greedy chooses an empty of a currently Attacker-alive monitored
window and the move is legal.  If no monitored alive window remains, every
danger is zero and no placement can satisfy that condition.

*Proof.*  The first sentence is Lemma 4.  If no monitored alive window
remains, the union of their empty sets is empty, so there is no cell of the
required kind.  QED.

### Counterexample 2 (both exact-turn placements cannot always be local)
**[PROVEN]**

Take a blanket position with exactly one Attacker-alive window globally, five
Attacker stones in that window, and one empty.  Such a finite colouring is
obtained by placing a Defender stone in every other window incident to those
Attacker stones, as in Proposition 2.  Its potential is
`1/sqrt(3)<1`.  Defender's first positive greedy placement must take the sole
empty and kills the last alive window.  The blanket `(2:2)` model nevertheless
requires a second placement, but no currently Attacker-alive window exists.
The even simpler case `Phi=0` fails before the first placement.

Therefore the literal requested statement

> every one of Defender's exactly two placements is in an empty of a
> currently Attacker-alive window

is false.  The correct implementation contract is:

1. generate every positive-reduction greedy move from active-window empties;
2. if that set becomes empty before the turn budget is exhausted, use an
   explicitly retained legal filler fallback.

A filler cannot increase current Attacker potential: it only kills windows.
It can, however, expand the shared radius-8 legality frontier and thereby
affect *future* birth terms.  The fixed-family theorem already allows
Attacker every residual target cell, and Theorems 3--4 explicitly budget
future births, so their proofs remain sound.  A solver must not call arbitrary
fillers globally irrelevant outside those hypotheses.

If an abstract Maker--Breaker variant permits passing when all dangers vanish,
then every stone actually placed by the potential strategy does lie in an
active-window empty.  Hexo itself has no such pass.

---

## 8. Solver use and trust boundary

### 8.1 What may be certified at a node

At a Defender `FirstStone` node:

1. `Phi_current<1` certifies **only the next Attacker turn** by Theorem 2.
2. It certifies a forever block for a fixed target family only when the solver
   also proves that every future winning window belongs to that family
   (Theorem 1).
3. A finite-region certificate may use `(R,H_R,P_R)` and (23), or the cheaper
   `(R,H_R,N_R)` and (24), to certify a forever block, but only together with
   an independent proof that all future Attacker placements remain in `R`.
4. A more general certificate may carry a uniform, verifier-checked
   cumulative birth reserve and use (16).  This is future-policy data, not a
   node-local scalar check.
5. A raw current `Phi<1` check must **not** be converted to a blanket
   `DEFENDER_WIN`, draw, or defender-move-set pruning certificate for the
   unrestricted infinite game.

The remaining unrestricted statement would require a new geometric defense
(for example a proved global pairing or another method controlling virgin
activation).  The direct ES/Beck invariant cannot supply it.  This is the
precise obstruction, not a conjectural proof step hidden in the argument.

### 8.2 Complexity

The source threat predicate is active count at least 4; at a nonterminal node
this means count 4 or 5.  The potential also needs counts 1--3.  Therefore a
from-scratch check through `WindowStore::entries()` is

\[
 O(\#\text{all touched window entries}),
\]

not `O(# engine threats)` if that phrase retains its source meaning.
The scan filters Defender-only, mixed, and dead entries before accumulating
the five bins `n_s`.  Because each placement changes exactly 18 entries, an
alive index, the bins, and the scalar potential can instead be maintained in
`O(18)=O(1)` work per placement; (31) is then an integer-only `O(1)` test.

To construct a greedy move from scratch, scan all touched entries and add each
Attacker-alive window's weight to each of its at most five empties in a
cell-score map.  This takes
`O(# all touched entries + sum_alive e(W))` time and
`O(sum_alive e(W))` auxiliary space.  With an already materialized alive list,
this reduces to `O(# alive windows)` up to the factor of five.  Rebuild or
update after Defender's first placement, then select the second maximum.
Tie-breaking is the fixed deterministic policy of Definition 4; the proofs
work for any fixed choice among maxima.

For a local certificate, computing the exact maximum `P_R` in (20) is a
finite maximum-coverage-over-matchings problem and need not be cheap.  A
verifier may instead substitute the proved upper bound (22), or any tighter
certified upper bound, in (21).  It must never substitute the value of one
candidate matching, which is only a lower bound.

The solver's restricted defender universe must include the filler fallback
from Section 7.  Positive greedy cells themselves are automatically in the
active-window term of the Section-10 heuristic in
`docs/PLAN_TSS_MOVESET_ZONES.md` and automatically legal.
The cheap raw `Phi<1` scalar is only the one-pair certificate of Theorem 2;
the forever theorems require the additional fixed-family closure or uniform
activation-reserve data described above.

---

## 9. Result boundary, in one table

| Claim | Status | Exact condition |
|---|---|---|
| Fixed current cohort blocked forever | **PROVEN** | Defender to move, `Psi_F<1` |
| All current alive windows safe for next Attacker pair | **PROVEN** | Defender to move, `Phi<1` |
| Dynamic touched family blocked forever | **PROVEN conditional on a uniform reserve** | (16) |
| Finite-region blanket block | **PROVEN** | `Phi_R+B_R^*<1` (or `Phi_R+N_R/9<1`) |
| Raw current touched `Phi<1` blocks unrestricted infinite game | **OPEN / NOT ESTABLISHED HERE** | Beck proof fails by births (13) |
| Adding “no current window has <=2 empties” repairs forever claim | **FALSE as an invariant repair** | clean births start from `Phi=0` |
| Every positive greedy placement is in a current alive window | **PROVEN** | Lemma 4 |
| Both mandatory Defender placements always are | **FALSE** | Counterexample 2 |
| Equality `Phi=1` suffices for arbitrary static Hex-window families | **FALSE** | three separated count-4 targets |

---

## 10. References and source concordance

1. P. Erdős and J. L. Selfridge, “On a Combinatorial Game,” *Journal of
   Combinatorial Theory, Series A* 14 (1973), 298--301.
   <https://users.renyi.hu/~p_erdos/1973-10.pdf>
2. J. Beck, “Remarks on Positional Games. I,” *Acta Mathematica Academiae
   Scientiarum Hungaricae* 40 (1982), 65--71.  This is the original biased
   generalization.
3. E. L. Sundberg, “Extremal Hypergraphs for the Biased
   Erdős--Selfridge Theorem,” *Electronic Journal of Combinatorics* 20(1)
   (2013), P26, especially Theorem 1 and its sequential-greedy proof.
   <https://www.combinatorics.org/ojs/index.php/eljc/article/download/v20i1p26/pdf/>
4. `packages/hexo_engine/rust/src/tactics.rs:13-17,21-52,128-208,341-392,443-486,511-514`
   -- length 6, axes, alive/win masks, touched-entry iteration, and 18 incident
   windows.
5. `packages/hexo_models/rust/src/threats_shared.rs:45-53,71-89,155-178`
   -- per-placement phase budget and win-now ordering.
6. `packages/hexo_engine/rust/src/legal.rs:17-18,114-145`,
   `packages/hexo_engine/rust/src/rules.rs:10-44`,
   `packages/hexo_engine/rust/src/board.rs:82-105,167-171`, and
   `packages/hexo_engine/rust/src/state.rs:283-337` -- radius-8 storage and
   validation, board update, per-placement win check, and phase transition.
7. `packages/hexo_engine/rust/src/board.rs:1-5` and
   `packages/hexo_engine/rust/src/coord.rs:1-15,35-39,43-61` -- sparse
   unbounded-board intent and the current `i16` coordinate carrier.
