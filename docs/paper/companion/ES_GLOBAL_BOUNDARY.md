# Report: the unrestricted ES-potential claim

VERDICT: GREEDY-REFUTED

Sequential dynamic touched-window greedy is not a sufficient global defense,
regardless of how maximum-danger ties are broken.  The counterexample below
has `Phi < 1`, uses a fixed nonadaptive Attacker continuation, never invokes a
zero-danger filler, and has an explicitly reachable engine enlargement.

This does **not** refute the raw existential claim.  It remains a `GAP`
whether every nonterminal Defender-`FirstStone` position with `Phi < 1` has
some non-greedy forever-blocking strategy.

All calculations use

\[
 \lambda=\sqrt3.
\]

The definitions and the fixed-family results are those of
`docs/proof_parts/ES_POTENTIAL.md`.  They are not reproved here.

---

## 1. Exact arithmetic and the greedy rule

For an Attacker-alive window with `s` Attacker stones, its contribution is
`lambda^{s-6}`.  For a nonterminal profile `(n_1,...,n_5)`,

\[
 27\Phi
 =3n_2+9n_4+(n_1+3n_3+9n_5)\sqrt3. \tag{1}
\]

If completed windows are retained at a terminal placement, add `27n_6` to
the right side.

At each Defender placement, *dynamic touched-window greedy* recomputes

\[
 d(x)=\sum_{\substack{W\text{ Attacker-alive}\\x\in W\setminus A}}
       \lambda^{-e(W)} \tag{2}
\]

and chooses any empty cell attaining its maximum.  Every maximum in the
counterexample is positive.  Therefore the claim covers every possible
tie-breaking rule and is independent of the filler policy.

The verifier `scripts/_es_global_check.py` represents every value as an
integer pair `(a,b)` denoting `(a+b sqrt(3))/27`.  It compares two such values
without floating point: when the coefficients of their difference have
opposite signs, it compares `a^2` with `3b^2`.  Irrationality excludes
equality.  The enumeration has no randomness, beam, or depth cutoff and omits no
exact greedy maximizer along the fixed Attacker continuation.

---

## 2. A position defeating every greedy tie-breaking

### Theorem 1 (all-ties greedy refutation) **[PROVEN]**

Let

\[
 A_0=\{(0,0)\},\qquad D_0=\{(1,0)\}, \tag{3}
\]

with Defender at `FirstStone`.  Then `Phi(P_0)<1`.  Against every sequential
dynamic touched-window-greedy defense, the following fixed Attacker
placements win:

\[
\begin{array}{c|c}
\text{Attacker turn}&\text{placements}\ \\ \hline
1&(2,-4),(2,2)\\
2&(-5,0),(-4,0)\\
3&(-3,0),(-2,0)\\
4&(-1,0)\quad\text{(win on the first placement).}
\end{array} \tag{4}
\]

The completed window is

\[
 W=\{(-5,0),(-4,0),\ldots,(0,0)\}. \tag{5}
\]

*Proof.*  The cell `(1,0)` shares exactly five of the 18 windows through
`(0,0)`.  Those five are dead and the other 13 are count-1 alive windows.
Thus

\[
 (n_1,n_2,n_3,n_4,n_5)=(13,0,0,0,0),
 \qquad
 \Phi(P_0)=\frac{13\sqrt3}{27}
           =\frac{13}{9\sqrt3}<1, \tag{6}
\]

where the last inequality follows from `13^2 < 3*9^2`, namely
`169 < 243`.  Neither player has completed a window.

Every placement in (4) is legal on every branch.  The first two are at
distance four from `(0,0)`; each later placement is within five of an
Attacker stone already on the target line.  The exhaustive greedy calculation
is given next.  Its maximizer unions contain none of the scheduled Attacker
cells before that cell is played.  Every Defender placement is legal, every
maximum is positive, and no Defender branch completes a window.

In the following table all danger entries mean `27d`.  A multiplicity after a
value is the number of input states having that value.  State deduplication
merges only identical coloured boards at the same phase, so it loses no
tie-breaking continuation.

| Defender placement | Input states | Exact maximum danger | Union of all maximizers | Output states |
|---|---:|---|---|---:|
| `D0.1` | 1 | `5sqrt(3)` | `{(-1,1),(0,-1),(0,1),(1,-1)}` | 4 |
| `D0.2` | 4 | `5sqrt(3)` (4) | same four cells | 4 |
| `D1.1` | 4 | `6sqrt(3)` (2); `7sqrt(3)` (2) | `{(2,r):-3<=r<=1}` | 12 |
| `D1.2` | 12 | `5sqrt(3)` (12) | `{(0,-4),(0,-2),(0,2),(0,4),(1,-4),(1,-3),(1,2),(1,3),(2,-5),(2,3),(3,-5),(3,-4),(3,1),(3,2)}` | 124 |
| `D2.1` | 124 | `10sqrt(3)` (124) | `{(-5,1),(-4,-1)}` | 242 |
| `D2.2` | 242 | `9sqrt(3)` (6); `10sqrt(3)` (236) | `{(-5,1),(-4,-1)}` | 124 |
| `D3.1` | 124 | `21+4sqrt(3)` (124) | `{(-6,0)}` | 124 |
| `D3.2` | 124 | `3+9sqrt(3)` (106); `10sqrt(3)` (18) | `{(-3,1),(-2,-1)}` | 124 |

For completeness, the next table gives the exact alive-window profiles after
every nonterminal placement.  A term `m*p` means that profile `p` occurs on
`m` distinct boards.  Profiles are `(n_1,n_2,n_3,n_4,n_5)`; equation (1)
therefore gives the exact potential in every row.

| Stage | Exact profile multiset |
|---|---|
| `P0` | `1*(13,0,0,0,0)` |
| `D0.1` | `4*(8,0,0,0,0)` |
| `D0.2` | `4*(3,0,0,0,0)` |
| `A1` | `4*(21,0,0,0,0)` |
| `A2` | `4*(39,0,0,0,0)` |
| `D1.1` | `2*(32,0,0,0,0); 10*(33,0,0,0,0)` |
| `D1.2` | `20*(27,0,0,0,0); 104*(28,0,0,0,0)` |
| `A3` | `20*(43,1,0,0,0); 104*(44,1,0,0,0)` |
| `A4` | `1*(49,4,1,0,0); 5*(50,4,1,0,0); 19*(51,4,1,0,0); 99*(52,4,1,0,0)` |
| `D2.1` | `1*(39,4,1,0,0); 5*(40,4,1,0,0); 38*(41,4,1,0,0); 198*(42,4,1,0,0)` |
| `D2.2` | `1*(30,4,1,0,0); 24*(31,4,1,0,0); 99*(32,4,1,0,0)` |
| `A5` | `2*(41,1,3,1,0); 11*(42,1,3,1,0); 24*(43,1,3,1,0); 87*(44,1,3,1,0)` |
| `A6` | `2*(49,3,1,2,1); 11*(50,3,1,2,1); 1*(51,1,1,2,1); 19*(51,3,1,2,1); 7*(52,1,1,2,1); 2*(52,2,1,2,1); 62*(52,3,1,2,1); 10*(53,1,1,2,1); 10*(53,2,1,2,1)` |
| `D3.1` | `2*(48,2,0,0,1); 11*(49,2,0,0,1); 1*(50,0,0,0,1); 19*(50,2,0,0,1); 7*(51,0,0,0,1); 2*(51,1,0,0,1); 62*(51,2,0,0,1); 10*(52,0,0,0,1); 10*(52,1,0,0,1)` |
| `D3.2` | `2*(39,1,0,0,1); 1*(40,0,0,0,1); 11*(40,1,0,0,1); 7*(41,0,0,0,1); 19*(41,1,0,0,1); 12*(42,0,0,0,1); 62*(42,1,0,0,1); 10*(43,0,0,0,1)` |

The last greedy turn explains the failure.  Immediately before `D3.1`, the
cell `(-6,0)` belongs to five alive `Q`-windows having Attacker counts
`1,2,3,4,4`.  Hence

\[
\begin{aligned}
 27d(-6,0)
 &=\sqrt3+3+3\sqrt3+9+9\\
 &=21+4\sqrt3. \tag{7}
\end{aligned}
\]

The immediate winning cell `(-1,0)` belongs to a count-4 window and the
count-5 target `W`, so

\[
 27d(-1,0)=9+9\sqrt3. \tag{8}
\]

The difference is `12-5sqrt(3)>0`, since `12^2>3*5^2`.  Thus every greedy
rule uniquely chooses `(-6,0)` first.  That stone deletes the count-4 term at
`(-1,0)`, leaving

\[
 27d(-1,0)=9\sqrt3. \tag{9}
\]

On 106 states the unique second maximum is `(-3,1)` with danger
`3+9sqrt(3)`; on the other 18 it is `(-2,-1)` with danger `10sqrt(3)`.
Both strictly exceed (9).  Consequently no greedy branch occupies
`(-1,0)`.  Attacker places there, completes (5) on the first placement of
turn 4, and wins immediately.

The verifier independently asserts legality, the absence of every earlier
completion by either colour, every exact profile above, every maximum and tie
above, and the terminal window on all 124 branches.  At the terminal
placement every branch has `n_6=1`, hence `Phi>=1`.  Its full exact terminal
potential multiset, also printed by the verifier, consists of

\[
\begin{array}{c|l}
27+b\sqrt3 &
 b=43(1),44(1),47(1),48(1),49(6),50(6),51(1),52(1),53(4),54(4),55(4)\\
30+b\sqrt3 &
 b=41(1),43(7),46(2),47(6),48(5),49(37),50(1),51(6),52(8),53(5),54(16),
\end{array} \tag{10}
\]

where parentheses give multiplicities.  They sum to 124.  This completes the
finite exhaustive proof.  QED.

### Corollary 1 **[PROVEN]**

The proposed forever theorem cannot use dynamic touched-window greedy, even
with adversarially optimal tie-breaking.  Any proof of the raw existential
claim must supply a genuinely non-greedy defense.

*Proof.*  Theorem 1 gives one fixed Attacker continuation winning against
every possible sequence of greedy maxima.  QED.

---

## 3. Engine-reachable enlargement

The compact position (3) already satisfies the blanket-position definition.
The following construction shows that unreachable material counts are not
the cause.

### Proposition 1 (reachable witness) **[PROVEN]**

There is a legal 39-placement engine history ending at a nonterminal
Defender-`FirstStone` position with 20 Attacker stones, 19 Defender stones,
the profile (6), and exactly the translated greedy tree of Theorem 1.

*Proof.*  Translate the compact core and continuation by `(-1,0)`, so the
Defender core stone is the engine-mandated opening `(0,0)` and the Attacker
core stone is `(-1,0)`.  Put

\[
 c=(-11,10),\qquad I=B_2(c),\qquad R=B_3(c)\setminus B_2(c). \tag{11}
\]

The sets have `|I|=19` and `|R|=18`.  Let `g=(-9,8)`, which lies in `I`.
Use this history:

1. Defender opens at `(0,0)`.
2. Attacker places `(-1,0)` and `g`.
3. For nine cycles, Defender places two unused cells of `R`, then Attacker
   places two unused cells of `I\setminus{g}`.

The core move `(-1,0)` is adjacent to the opening, and
`d(g,(-1,0))=8`.  Every other cell of `B_3(c)` is within distance five of
`g`, so every listed move is legal.  No Attacker completion occurs: every
axis-line chord of `B_2(c)` has at most five cells, and the core is more than
five away.  No Defender
completion occurs: every straight run in the radius-3 ring has at most four
cells, and the opening core is more than five away.  These statements hold at
every prefix.

At the end, Attacker owns `I` and the translated core, while Defender owns
`R` and `(0,0)`.  Every length-6 window meeting an Attacker cell of `I` exits
the radius-2 ball through the radius-3 ring, so it contains a Defender stone.
The padding therefore contributes zero to `Phi`.  Its distance from every
cell occurring in the translated exhaustive tactical tree is at least six;
no length-6 window meets both components.  The profile and every danger are
therefore exactly those of Theorem 1 after translation.

The script constructs the history placement by placement, checks radius-8
legality and both win predicates after each placement, and reruns all greedy
branches on the enlarged position.  The two profile, potential, maximum, and
branch tables are identical.  QED.

---

## 4. No defense can preserve the raw sublevel set

The greedy refutation is not merely the old fixed-filler birth example.
Better fillers or a different defense cannot make `Phi<1` an epochwise
invariant.

### Lemma 1 (universal clean escape) **[PROVEN]**

From every finite, nonempty, nonterminal position immediately before an
Attacker turn, Attacker has a legal pair which births exactly 36 distinct
one-stone windows.  Its exact
birth mass is

\[
 C=36\lambda^{-5}=\frac4{\sqrt3}. \tag{12}
\]

*Proof.*  Put `h(q,r)=q+r`.  Choose an occupied cell `z` maximizing `h`, and
write `M=h(z)`.  Attacker places

\[
 x=z+(4,4),\qquad y=x+(4,4). \tag{13}
\]

Both displacements have hex distance eight, so both placements are legal in
sequence.  A `Q`- or `R`-window through `x` has minimum `h` at least
`h(x)-5=M+3`; a `QR`-window through `x` has constant `h=M+8`.  Every one of
the 18 windows through `x` was therefore stone-free.  After placing `x`, a
`Q`- or `R`-window through `y` has minimum `h` at least `M+11`, and a
`QR`-window through `y` has constant `h=M+16`.  Those 18 windows were also
stone-free.  Finally `y-x=(4,4)` is on none of the three window axes, so the
two cells share no window.  This gives 36 distinct count-1 births and (12).
QED.

### Corollary 2 **[PROVEN]**

From the position containing one Defender stone and no Attacker stone,
`Phi=0`.  After every possible first Defender pair, Attacker can force

\[
 \Phi\ge\frac4{\sqrt3}>1 \tag{14}
\]

at the next Defender epoch.

*Proof.*  A Defender pair cannot complete six from one stone.  Apply Lemma 1.
The 36 new terms alone sum to (12), and `4/sqrt(3)>1` follows from `16>3`.
QED.

Repeating Lemma 1 at every continuing Attacker turn creates 36 fresh,
distinct count-one window labels and a source term `4/sqrt(3)` on every
such turn. Labels born on different turns are distinct, although their
residual cell supports need not be disjoint. No clean-escape placement
promotes an earlier label, because every window through that placement was
stone-free immediately beforehand. In the conservative blanket game that
ignores Defender completions, the cumulative birth sum therefore diverges
while every such label either remains at count one or is killed.

- no strategy can prove the raw claim by maintaining `Phi<1` at every epoch;
- radius-8 mobility alone gives no summable birth reserve for Theorem 3; and
- any amortization that irreversibly debits every ES birth against a finite
  reserve, or pays that debit only from same-turn maturity promotions, fails
  on repeated clean escape: the exact positive debit recurs with zero
  promotions.

These are failures of proof routes, not an Attacker win against a non-greedy
defense.

---

## 5. New finite-horizon results

### Theorem 2 (five Attacker placements at the raw threshold) **[PROVEN]**

If Defender is at `FirstStone` and `Phi(P_0)<1`, Defender has a strategy that
prevents an Attacker win during the next five Attacker placements: two full
Attacker turns and the first placement of the third.

*Proof.*  Let `F` be exactly the windows Attacker-alive at `P_0`, and use
fixed-`F` sequential greedy.  Theorem 1 of `ES_POTENTIAL.md` blocks every
member of `F` forever.  A window containing a Defender stone at `P_0` stays
dead.  Every remaining window was stone-free at `P_0`, so it contains at most
five Attacker stones after the first five future Attacker placements.  These
three cases exhaust all windows.  QED.

This strictly extends the existing one-cycle certificate, which covers only
the first two future Attacker placements.

The five-placement count is sharp for this fixed-initial-cohort strategy with
the filler from `ES_POTENTIAL.md`.  Start with only `D=(-4,-4)`, so `F` is
empty and `Phi=0`, and play

\[
\begin{array}{c|c}
D&(-3,-4),(-2,-4)\\
A&(2,0),(2,1)\\
D&(3,1),(4,1)\\
A&(2,2),(2,3)\\
D&(5,1),(6,1)\\
A&(2,4),(2,5).
\end{array} \tag{15}
\]

The Defender cells are exactly the maximum-`q`, then maximum-`r`, positive-`Q`
fillers.  The first Attacker cell is at distance eight from `(-2,-4)` and all
later Attacker cells are adjacent to an Attacker stone.  The sixth Attacker
placement completes `{(2,r):0<=r<=5}`.  This example concerns fixed-cohort
greedy, not the dynamic greedy refuted in Theorem 1.

### Lemma 2 (one-greedy amplification) **[PROVEN]**

Let `F` be a fixed family alive at a Defender epoch and put `X=Psi_F`.  Suppose
a Defender turn consists of one arbitrary legal placement followed by one
`F`-greedy placement, and then up to two Attacker placements.  At each
Attacker prefix, including a terminal placement, the fixed-family potential
is at most

\[
 \Psi_F\le\frac32X. \tag{16}
\]

If both placements are nonterminal, this is the bound at the next Defender
epoch.  If the right side is strictly below one, no member of `F` is
completed at either placement.

*Proof.*  The arbitrary Defender placement leaves `X'<=X`.  Let the greedy
placement reduce the potential by `delta`, and let `S` be the maximum
remaining `F`-danger.  Deletion monotonicity and greediness give `S<=delta`.
Also `S<=X'-delta`, since a cell danger is a subsum of the remaining
potential.  The next Attacker pair adds at most

\[
 (\lambda-1)(1+\lambda)S=2S.
\]

Put `Y=X'-delta`.  If both Attacker placements occur, therefore

\[
 \Psi_F^+\le Y+2S
 \le X'-\delta+2\min\{\delta,X'-\delta\}
 \le\frac32X'\le\frac32X. \tag{17}
\]

After only the first Attacker placement, the increase is at most
`(lambda-1)S<=2S`, so the same bound applies without assuming that the second
placement occurs.  A completed member of `F` contributes one by itself,
contradicting (16) whenever `3X/2<1`.  QED.

### Theorem 3 (three-pair certificate) **[PROVEN]**

Let `x,y` be the first future Attacker pair.  Defender can prevent a win
during the next three complete Attacker turns under the following conditions:

\[
 \Phi(P_0)<
 \begin{cases}
  1,&x,y\text{ lie in no common window},\\
  2/3,&x,y\text{ have common-window axis distance }2,3,4,\text{ or }5,\\
  4/9,&x,y\text{ are axis-adjacent}.
 \end{cases} \tag{18}
\]

In particular, `Phi(P_0)<4/9` unconditionally certifies all three Attacker
pairs.

*Proof.*  Let `F` be the initial alive cohort and `X_t=Psi_F(P_t)`.  On the
first Defender turn use two `F`-greedy placements.  Then

\[
 X_1\le X_0=\Phi(P_0). \tag{19}
\]

An initially virgin window completed by the sixth future Attacker placement
must contain all six future Attacker cells, hence in particular `x,y`.

If `x,y` share no window, such a sixth-placement virgin win is impossible.
Continue with two `F`-greedy placements per Defender turn; the initial cohort
is blocked.

After exchanging `x,y` if necessary, suppose `y=x+dv`, where `v` is one of
the six signed unit axis vectors and `2<=d<=5`.  Every length-6 window
containing `x,y` also contains the intermediate cell `x+v`.  If that
cell was previously occupied, all relevant initially virgin targets are
already absent or dead; use any legal filler as the arbitrary placement.
Otherwise place at `x+v` on the next Defender turn.  Then make one `F`-greedy
placement.  Lemma 2 gives

\[
 X_2\le\frac32X_1<1 \tag{20}
\]

under the second condition in (18).  Resume two `F`-greedy placements on the
following Defender turn.  No common virgin target or initial target can be
completed by pair three.

Finally, after the same choice of orientation, suppose `y=x+v`.  Their five
common windows start at

\[
 x-4v,x-3v,x-2v,x-v,x. \tag{21}
\]

Let `V(x,y)` be the subfamily of these five windows which were stone-free at
`P_0` and remain alive after the first Attacker pair.  Windows outside this
subfamily are either members of `F` or permanently dead.  The cell `x-v`
hits the first four windows.  Place there on the next Defender turn if it is
empty.  If it is occupied, every relevant member among those four either was
not initially virgin or has already been killed; use any legal filler as the
arbitrary placement.  Then make one `F`-greedy placement.  Among `V(x,y)`,
only

\[
 W_*=\{x,x+v,\ldots,x+5v\} \tag{22}
\]

can remain unblocked.  Before the third Attacker pair, if `W_*` belongs to
`V(x,y)` and remains alive, it contains at most four Attacker stones and has
at least two legal empties; place in one of them.  If `W_*` was not initially
virgin or has already died, use any legal filler as the arbitrary placement;
the fixed-family strategy or permanence covers it.  Then make one `F`-greedy
placement.  Two applications of Lemma 2 give

\[
 X_3\le\left(\frac32\right)^2X_1
      \le\frac94X_0<1 \tag{23}
\]

under the last condition in (18).  Every initially dead window remains dead;
every initially alive window is covered by the strict fixed-family bound;
and every initially virgin window capable of a sixth-placement win was among
the targets just killed.  QED.

The adjacent case cannot be absorbed immediately with one Defender cell:
the intersection of the five windows in (21) is exactly the two
Attacker-occupied cells `{x,y}`.

---

## 6. No-go results for two proposed repairs

### Proposition 2 (finite saturation does not enable static pairing)
**[PROVEN]**

No partial matching covers all but finitely many length-6 windows.  Hence no
finite initial position admits a static pairing that covers every initially
virgin target.

*Proof.*  Suppose a matching misses at most `K` windows.  Discard pairs not
contained in a window.  Let `B_R` be the radius-`R` hex ball, with

\[
 |B_R|=3R^2+3R+1. \tag{24}
\]

For each axis, `B_R` meets `2R+1` line segments whose lengths sum to
`|B_R|`.  For `R>=5`, those segments contain

\[
 |B_R|-5(2R+1)
\]

length-6 windows wholly inside `B_R`.  Across three axes, at least
`3|B_R|-15(2R+1)-K` of these internal windows must be covered.  One matched
pair is contained in at most five windows on its axis.  Thus `B_R` contains
at least

\[
 \frac25\bigl(3|B_R|-15(2R+1)-K\bigr) \tag{25}
\]

matched-cell incidences.  A matching supplies at most `|B_R|`, so (25)
implies

\[
 |B_R|\le60R+30+2K, \tag{26}
\]

contradicting the quadratic value (24) for large `R`.

A finite position has at most `18|S|` windows meeting its occupied support
`S`; every other window is initially virgin.  A matching covering every
virgin window would therefore miss only finitely many windows, contrary to
the first assertion.  QED.

This strengthens the existing no-global-pairing result in the exact way
needed here.  It does not exclude a dynamic reassignment strategy.

### Proposition 3 (static two-tier damping no-go) **[PROVEN]**

Suppose a nonnegative static per-window account assigns weight `w_s(W)` when
`W` has `s` Attacker stones, and assume:

1. the virgin sum `sum_W w_0(W)` is finite on the empty infinite board;
2. `w_6(W)>=1` for every completed target; and
3. the two-placement Beck growth condition holds edgewise:

\[
 w_{s+2}(W)\le3w_s(W),\qquad s=0,2,4. \tag{27}
\]

No such account exists.

*Proof.*  For every window,

\[
 1\le w_6(W)\le3w_4(W)\le9w_2(W)\le27w_0(W). \tag{28}
\]

Thus `w_0(W)>=1/27` for each of infinitely many windows, contradicting the
finite virgin sum.  QED.

This excludes static spatial-damping schemes which also retain both the
uniform terminal threshold and the edgewise factor-three condition (27).
The proposition does not exclude static schemes with different comparison
rules, age-dependent weights, moving centers, cross-window cancellation, or
a nonadditive amortized account.  Those cases remain `GAP`.

---

## 7. Exact finite reduction

### Theorem 4 (compact finite-horizon equivalence) **[PROVEN]**

Fix a finite nonempty position `P` with occupied support `S_0`,
`|S_0|=n`, at Defender-`FirstStone`.  For `h>=1`, define

\[
 d(x,S_0)=\min_{s\in S_0}d(x,s),\qquad
 R_h=\{x:d(x,S_0)\le32h\}. \tag{29}
\]

Every move in the first `h` full Defender/Attacker rounds lies in `R_h`, and

\[
 |R_h|\le n(3072h^2+96h+1). \tag{30}
\]

Defender blocks forever from `P` if and only if Defender has a survival
strategy for every finite `h`-round game on `R_h`.

Here `R_h` is a proved move universe for the exact ideal-board game through
that horizon, not an added boundary rule.

*Proof.*  There are at most `4h` placements.  Inductively, the `k`-th future
placement is within distance `8k` of `S_0`: it is within eight of an earlier
occupied cell.  This proves (29).  A radius-`r` hex ball contains
`3r^2+3r+1` cells; applying the union bound to `n` balls with `r=32h` gives
(30).  The exact truncated game tree is therefore finite and is decided by
backward induction.

One direction of the equivalence is immediate.  For the other, form a tree
whose level `h` consists of finite Defender policy tables surviving the
`h`-round game, with an edge given by restriction to the previous level.
Every level is finite, every policy has finitely many extensions, and every
level is nonempty by hypothesis.  Koenig's lemma gives a consistent infinite
chain.  Its union is one Defender strategy surviving every finite prefix.
Conversely, if no forever strategy exists, some finite level is empty, and
finite minimax gives an Attacker strategy forcing completion within that
horizon.  QED.

Thus every refutation of the raw claim has a finite forced-win strategy-tree
certificate.  For a fixed `P`, the raw claim is equivalent to a concrete
sequence of finite games.  The theorem supplies no computable or uniform
cutoff `H(P)` after which survival implies forever safety.

---

## 8. Remaining obstruction

The results leave the following steps explicitly unproved.

### `GAP-RAW`

Theorem 1 refutes all dynamic touched-window-greedy defenses.  It does not
give an Attacker strategy against arbitrary non-greedy play.  No theorem here
proves that such a defense always exists.  The raw existential claim remains
open.

### `GAP-GLOBAL-RENEWAL`

Theorems 2 and 3 extend the certified horizon, but they do not return to a new
Defender epoch satisfying the same raw hypothesis.  Corollary 2 proves that
no defense can use `Phi<1` itself as the renewed invariant, already from
`Phi=0`.

### `GAP-AMORTIZED-ABANDONMENT`

An online account must discount indefinitely many weak, abandoned births yet
remain safe if Attacker later reuses dormant stones in multi-axis forks.
Repeated clean escape makes the exact birth source positive with no maturity
progress.  Proposition 3 excludes only static edgewise factor-three accounts;
it does not supply the needed history-sensitive refund rule.

### `GAP-DYNAMIC-PAIRING`

Proposition 2 excludes static matchings even after allowing finitely many
initial exceptions.  It does not exclude an adaptive response system whose
pairs or ownership assignments change with play.

### `GAP-FINITE-CUTOFF`

Theorem 4 makes every fixed horizon finite and proves that every loss has a
finite witness.  It gives no a priori horizon bound.  Checking all finite
horizons is equivalent to, not a solution of, the original safety problem.

The established boundary is therefore exact: raw current `Phi<1` gives more
than one-cycle safety, but touched-window greedy can be forced to lose; raw
potential renewal, static pairing, and static spatial damping cannot repair
the proof; and the existence of a different global Defender strategy remains
a `GAP`.
