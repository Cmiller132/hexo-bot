# Defender-reply domination in Hexo

> **Status.** This document is self-contained and proof-oriented. Definitions,
> Lemmas 1--8, and Patterns P1--P3 are **PROVEN** in the intended unbounded
> integer-lattice rules. Section 8 specifies finite checkers that have **not yet
> been run**; consequently no result here is labelled `PROVEN-MECH`. The broad
> versions of P1 and P2 that omit the stated frontier/mask hypotheses are not
> claimed. The one prospective generalization is labelled **CONJECTURE** and is
> not used by any proof.
>
> This is the domination-pattern component referenced by
> `docs/PROOF_TSS_DEFENDER_ZONES.md` Section 11. It answers the program in
> `docs/PLAN_TSS_MOVESET_ZONES.md` Sections 6 and 8.1--8.2. In particular, every
> proof below separately discharges early terminality, the radius-8 legality
> frontier, and occupancy/cross-window conflicts.

---

## 1. Source-verified rule model

### Definition 1 (lattice, balls, and windows)

The formal board is the axial lattice `Z^2`. For cells `x=(q,r)` and
`y=(q',r')`, put

\[
 d(x,y)=\max(|q-q'|,|r-r'|,|(q-q')+(r-r')|),
 \qquad B_R(x)=\{z:d(x,z)\le R\}.
\]

The three unoriented axes have positive vectors `(1,0)`, `(0,1)`, and
`(1,-1)`. A window is six consecutive cells on one axis. Write
`Omega(x)` for the 18 windows containing `x`: three axes times the six
possible offsets of `x` in a window. Write

\[
 X(x)=\bigcup_{W\in\Omega(x)}W.
\]

Thus `|X(x)|=31`: on each of three axis lines it contains offsets `-5,...,5`,
and the three lines meet only at `x`.

### Definition 2 (rule position)

A rule position `P` consists of a finite occupancy map

\[
 \sigma_P:Z^2\rightharpoonup\{A,D\},
\]

the current player, the turn phase (`Opening`, `FirstStone`, or
`SecondStone{first}`), the absolute placement counter, and an optional
terminal outcome `(winner, placement_count)`. Unless explicitly
described as a raw local-checker assignment, a rule position is assumed
reachable by the transition rules. In particular, a reachable nonterminal
position contains no completed window. `A` and `D` are fixed strategic roles;
either may be engine `Player0` in a particular game. All patterns below are
post-opening, nonterminal positions with `D` to place.

For a post-opening position define the *legal support*

\[
 \Lambda(P)=\bigcup_{s\in\operatorname{dom}\sigma_P}B_8(s).
\]

In a nonterminal post-opening position its legal placements are exactly

\[
 L(P)=\Lambda(P)\setminus\operatorname{dom}\sigma_P. \tag{1}
\]

For a terminal position, `L(P)` is empty even though the geometric support
set remains defined.

The opening is the exceptional one-cell turn at the origin. Thereafter a
normal turn is `FirstStone` followed, if nonterminal, by `SecondStone` by the
same player. A win is checked immediately after each placement; a winning
first stone ends the game without a second stone.

For a window `W`, let `M_A(W,P)` and `M_D(W,P)` be its labelled six-bit masks.
The window is *dead* when both masks are nonzero, and *alive for X* when the
other mask is zero. A cell `c` empty in `P` is a *dead cell* when every member
of `Omega(c)` is already dead in `P`.

### Source concordance

The formal model above agrees with the rule code as follows.

1. `packages/hexo_engine/rust/src/tactics.rs:13-17,21-52,55-77` defines
   length 6, the three axes, and 18 affected windows per placement.
   `tactics.rs:128-155,171-208` defines masks, active/dead behavior, threats,
   and wins. `tactics.rs:443-486,511-514` updates exactly the 18 incident
   masks.
2. `packages/hexo_engine/rust/src/legal.rs:17-18,60-64,114-145` fixes
   `LEGAL_RADIUS=8`, starts with an empty store, removes the occupied cell, and
   unions the empty cells of its closed radius-8 ball. The board inserts the
   stone and invokes that update against whole-board emptiness on every
   placement (`packages/hexo_engine/rust/src/board.rs:82-95,167-170`).
   Together with `packages/hexo_engine/rust/src/rules.rs:11-45`, this gives
   (1) for forward-built post-opening states.
3. `packages/hexo_engine/rust/src/state.rs:289-357` (needed to make the four
   named sources operationally complete) validates and places one stone,
   checks its window update for a win, and advances phase only if no win
   occurred.
4. `packages/hexo_models/rust/src/threats_shared.rs:45-53,71-89,92-178`
   gives placement budgets `B=2` at `FirstStone` and `B=1` at `Opening` or
   `SecondStone`, the capped hitting-set rule, and own-win priority. Those are
   derived tactical predicates, not additional terminal rules.
5. The opening state constructed by the engine is initially empty; the origin
   becomes permanently occupied only after the unique legal opening move
   (`rules.rs:16-23`). This document concerns positions after that move.

The Rust coordinate carrier is `i16`, although the board and rules are
designed as unbounded. The theorems use `Z^2`; an implementation checker must
use wide arithmetic or impose the ordinary no-overflow precondition.

### Lemma 1 (the three causal channels) **[PROVEN]**

Adding a stone of a fixed owner at an empty cell `c` can affect the formal game
only through:

1. **occupancy:** `c` ceases to be a playable cell and acquires an owner;
2. **window masks:** precisely the 18 masks in `Omega(c)` gain the bit for
   `c`; and
3. **legality frontier:** `B_8(c)` is added to legal support.

There is no fourth *cell-location* channel. Mover, phase, and placement count
also advance according to the common phase machine; their update is
determined by the old phase and whether the changed masks terminate the game.

*Proof.* Equation (1) is a function only of occupied coordinates and the
radius-8 relation. Win, alive/dead, and threat predicates are functions of the
labelled window masks. `own_win_now`, capped hitting sets, and their verdict
are functions of those masks together with the current player and the
phase-derived budget; paired simulations preserve the latter data. A cell
occurs in exactly the 18 masks in `Omega(c)`. Turn phase and placement count
advance solely as functions of the old phase and the placement, with phase
advance suppressed exactly when the changed masks contain a win. This
exhausts the transition code cited above. Ordered history and insertion order
may affect model features or serialization, but not these formal game rules.
QED.

### Lemma 2 (permanence) **[PROVEN]**

Window masks and legal support are monotone under forward play, apart from
removing a newly occupied cell from the legal set. In particular, a dead
window stays dead forever.

*Proof.* Forward play only adds stones and radius-8 support; it never removes
a stone. A nonzero bit in either colour mask therefore never disappears.
QED.

---

## 2. Finite-horizon outcome domination

The quantifier direction matters. It is insufficient to push one convenient
pair of continuations from the discarded branch to the searched branch. To
prune a defender reply, defender policies must transfer from the discarded
reply to the searched reply, while arbitrary attacker challenges transfer in
the opposite direction.

### Definition 3 (stopped horizon outcome)

Let `S` be a possibly terminal successor of a defender reply. Horizon `n`
means at most `n` *further placements* after that reply. The reply itself is at
time `0`. Given pure continuation strategies `sigma_D` and `tau_A`, play stops
at the first terminal placement or after `n` further placements. Its outcome
is one of

\[
 \operatorname{Out}_n(S,\sigma_D,\tau_A)
 \in \{D@t:0\le t\le n\}\cup
       \{A@t:0\le t\le n\}\cup\{?\},
\]

where `?` means that no terminal placement occurred through the horizon.
For defender comparison use

\[
 u_D(D@t)=1,\qquad u_D(?)=0,\qquad u_D(A@t)=-1. \tag{2}
\]

Times and winning windows are retained in the trace but all wins by the same
player have the same qualitative value. Thus an earlier defender win on a
different crossing window is explicitly good, not an illicit assumption that
the original target window eventually completes. The strict cutoff still
matters: an `A` win at placement `n` is `-1`, while one at `n+1` is `?` at this
horizon. There is no same-turn tie because every placement is checked and the
trace stops immediately.

### Definition 4 (continuation strategies)

A pure `X`-strategy from `S` assigns a legal placement to every nonterminal
history of length less than `n` at which `X` is to place. It must also act at
both consecutive histories when `X` owns both stones of a turn. A strategy is
never queried below a terminal history.

The legal tree is finite at every finite horizon: the current support is a
finite union of finite balls, and only finitely many balls can be added in
`n` placements.

### Definition 5 (`n`-outcome-domination)

Let `a` and `b` be legal defender replies at the same nonterminal position
`P`, and let `S_a=P+a`, `S_b=P+b` denote their stopped rule successors. Reply
`b` is *`n`-outcome-dominated by `a` for the defender*, written

\[
 b\preceq_n a,
\]

iff there are strategy-transfer maps

\[
 T_D:\Sigma_D(S_b)\to\Sigma_D(S_a),\qquad
 T_A:\Sigma_D(S_b)\times\Sigma_A(S_a)\to\Sigma_A(S_b)
\]

such that, for every discarded-branch defender strategy `sigma_b` and every
searched-branch attacker challenge `tau_a`,

\[
 u_D\!\left(\operatorname{Out}_n
    (S_a,T_D(\sigma_b),\tau_a)\right)
 \;\ge\;
 u_D\!\left(\operatorname{Out}_n
    (S_b,\sigma_b,T_A(\sigma_b,\tau_a))\right). \tag{3}
\]

Equivalently, spelling out the transfers,

\[
 \forall\sigma_b\;\exists\sigma_a\;
 \forall\tau_a\;\exists\tau_b:\quad
 u_D(\operatorname{Out}_n(S_a,\sigma_a,\tau_a))
 \ge u_D(\operatorname{Out}_n(S_b,\sigma_b,\tau_b)). \tag{4}
\]

This is the requested "for every continuation strategy pair" comparison:
each defender policy after `b` and each attacker challenge after `a` is paired
by a legal transfer. The asymmetric directions are necessary for sound
defender pruning.

### Lemma 3 (value characterization and pruning direction) **[PROVEN]**

Let

\[
 V_D^n(S)=\max_{\sigma_D}\min_{\tau_A}
 u_D(\operatorname{Out}_n(S,\sigma_D,\tau_A)).
\]

Then

\[
 b\preceq_n a \quad\Longleftrightarrow\quad
 V_D^n(S_a)\ge V_D^n(S_b). \tag{5}
\]

Consequently, if the attacker can force a win by the horizon after the
stronger/search reply `a`, it can force a win by the horizon after the
dominated/omitted reply `b`.

*Proof.* Suppose first that `V_D^n(S_a)>=V_D^n(S_b)`. Given `sigma_b`, choose
an optimal `sigma_a`. Because the trees are finite, choose `tau_b` attaining
the minimum against `sigma_b`. For every `tau_a`,

\[
 u_D(S_a,\sigma_a,\tau_a)\ge V_D^n(S_a)
 \ge V_D^n(S_b)\ge u_D(S_b,\sigma_b,\tau_b),
\]

which gives (4). Conversely, use an optimal `sigma_b` in (4). The supplied
`sigma_a` has outcome at least `V_D^n(S_b)` against every `tau_a`, because
every corresponding `tau_b` has outcome at least the minimum against
`sigma_b`. Taking the minimum over `tau_a` and then the maximum over
`sigma_a` proves (5). If the attacker forces a win after `a`, then
`V_D^n(S_a)=-1`; (5) forces `V_D^n(S_b)=-1`. The final statement uses the
standard pure-strategy determinacy of a finite perfect-information game:
backward induction assigns a maximum at `D` nodes and a minimum at `A` nodes,
and the corresponding local choices form uniform causal strategies. Hence
`V_D^n=-1` is equivalent to one attacker strategy forcing `A@t` for some
`t<=n` against every defender strategy. QED.

### Definition 6 (causal alternating-simulation certificate)

Definitions 5 and (5) are semantic. The local patterns below use a stronger,
machine-checkable witness. A depth-`k` *defender-favouring alternating
simulation* relates a searched/better state `G` to a discarded/worse state
`H`, at equal placement depth, as follows.

1. If `G` is terminal with winner `D`, the pair is accepted immediately. If
   `H` is terminal with winner `A`, it is also accepted immediately.
2. If `G` is terminal with winner `A`, the pair is accepted only when `H` is
   already terminal with winner `A`. If `H` is terminal with winner `D`, the
   pair is accepted only when `G` is already terminal with winner `D`.
   No move is generated after either terminal state.
3. If both are nonterminal and `k=0`, the pair is accepted (`?` versus `?`).
4. Otherwise the two states have the same player to move and the same
   remaining placement budget. At a `D` history, **every** legal move in `H`
   is sent causally to a legal move in `G`, and the successors are related at
   depth `k-1`. At an `A` history, **every** legal move in `G` is sent causally
   to a legal move in `H`, and the successors are related at depth `k-1`.

"Causally" means that the move map depends only on the paired history and the
current move, never on a later choice. `FirstStone` must pair with
`FirstStone`, and `SecondStone` with `SecondStone`; when a coordinate
involution is used, the stored `first` coordinates are paired by it as well.

### Lemma 4 (simulation implies domination) **[PROVEN]**

If `(S_a,S_b)` has a depth-`n` causal alternating-simulation certificate, then
\(b\preceq_n a\).

*Proof.* Construct `T_D` recursively: whenever `sigma_b` selects the next
move in `H`, use the certificate's `D` map to select the move in `G`.
Given an arbitrary `tau_a`, construct `T_A` recursively by using the
certificate's `A` map from its move in `G` to a move in `H`. Both maps are
legal and prefix-preserving by Definition 6. Induct on remaining depth. If a
one-sided accepted terminal leaves the other branch live, extend the unused
live-branch policy arbitrarily with legal moves; `G` already has value `+1`
or `H` already has value `-1`, so the comparison is fixed and no paired move
is required. The conservative terminal clauses are sufficient for the
comparisons in (2), and the `k=0` clause compares two nonterminals. This
proves (3). QED.

### Lemma 5 (the channel checklist for a simulation step) **[PROVEN]**

Assume a paired-history invariant has already synchronized the actor, phase
class/placement budget, and every occupancy, mask, and support fact unchanged
by the next mapped placements. The only *new* rule-channel obligations needed
to validate `c_G` versus `c_H` are the following.

1. **Occupancy:** each mapped cell is empty, and attempts to use a cell
   occupied only in the other branch have an explicit image.
2. **Masks and terminality:** compare every labelled mask in
   \(\Omega(c_G)\cup\Omega(c_H)\) after the placement, and apply the terminal
   clauses before phase advance. There are at most 36 such windows.
3. **Frontier:** prove the legal inclusion in the correct alternating
   direction: `D` maps legal moves of `H` to `G`, while `A` maps legal moves
   of `G` to `H`.

*Proof.* These are precisely Lemma 1's three channels. Under the stated
inductive invariant, every datum outside them remains synchronized. The union
of the two 18-window sets covers every mask that either mapped placement can
change. Immediate terminal testing is a function of those changed masks.
QED.

### Lemma 6 (favourable recolouring with inert reverse cells) **[PROVEN]**

Suppose two nonterminal post-opening states have the same occupied-coordinate
set, mover, and phase class/remaining placement budget. (Their stored
`SecondStone{first}` cells may differ only when both are already occupied and
paired by the continuation map.) If `G` is obtained from `H` only by changing
some stones from `A` to `D`, then identity continuation is
defender-favouring at every horizon: an `A` win in `G` is also an `A` win in
`H`, and a `D` win in `H` is also a `D` win in `G`, no later.

The same conclusion holds if additional reverse recolourings `D` to `A` occur
in a set `R`, provided every window meeting `R` has a common `A` witness and a
common `D` witness at cells whose owners agree in `G` and `H`. Such a window
is dead in both states independently of every recoloured cell and remains so.

*Proof.* Legality depends on occupied coordinates, not owners, so the legal
sets agree. In every non-dead window, the `D` mask in `G` contains the
corresponding `D` mask of `H`, and the `A` mask is contained in that of `H`.
Thus `G` cannot create a new `A` completion and cannot destroy a `D`
completion from `H`. Every window meeting `R` instead contains the stipulated
common opposite-colour witnesses, so it is dead in both states and stays dead
by Lemma 2. (Deadness in only one compared state would not suffice: a
hypothetical recolouring could remove its sole blocker.) Identical later
coordinate moves add the same owner's bits and preserve the relation.
Checking after every placement gives the stopped terminal claim. QED.

---

## 3. P1 -- dead-cell dismissal

The fact that a stone updates a radius-8 ball does not imply that it adds any
new legal support. For a genuinely dead cell, the 18 dead-window certificates
already force old stones on all six axial rays, and those stones cover the
entire ball.

### Lemma 7 (a dead empty has no legality frontier) **[PROVEN]**

Let `b` be empty and dead in a nonterminal post-opening position `P`. Then

\[
 B_8(b)\subseteq\Lambda(P). \tag{6}
\]

In particular, `b` is legal and placing at `b` inserts no previously
unsupported cell into the legal store.

*Proof.* Let `u` range over the six directed axial unit vectors. The endpoint
window

\[
 W_u=\{b,b+u,\ldots,b+5u\}
\]

belongs to `Omega(b)`. It is dead while `b` is empty, so among
`b+u,...,b+5u` there is at least one old stone; write one as
`s_u=b+k_u u`, `1<=k_u<=5`.

Take any `z in B_8(b)`. It lies in a closed 60-degree sector between adjacent
directed axes `u,v`, so

\[
 z-b=\alpha u+\beta v,\qquad
 \alpha,\beta\ge0,\qquad \alpha+\beta=d(b,z)\le8.
\]

By a `D6` rotation/reflection take `u=(1,0)`, `v=(0,1)`, and put
`s=s_u=b+ku`, `1<=k<=5`. Then

\[
 d(z,s)=\max(|\alpha-k|,\beta,|\alpha+\beta-k|)\le8,
\]

because each of `alpha`, `beta`, `alpha+beta`, and `k` lies in `[0,8]`.
Thus \(z\in B_8(s)\subseteq\Lambda(P)\). This holds for every `z`, proving
(6). QED.

### Pattern P1 (frontier-certified dead-cell dismissal) **[PROVEN]**

Let `P` be a nonterminal post-opening position with `D` to place. Let `a != b`
be legal empty cells, and assume:

1. **P1-M (dead masks):** every window in `Omega(b)` is dead already in `P`;
2. **P1-LF (frontier equivalence):** unless `P+a` is an immediate `D` win,

   \[
   \Lambda(P)\cup B_8(a)=\Lambda(P)\cup B_8(b). \tag{7}
   \]

Then \(b\preceq_n a\) for every finite `n`. By Lemma 7, condition (7) is
equivalent here to

\[
 B_8(a)\subseteq\Lambda(P). \tag{8}
\]

Therefore a dead reply may be dismissed in favour of **any searched reply
that either wins immediately or is frontier-inert**. Deadness of `b` alone
does not authorize the phrase "any searched reply."

*Proof.* If `P+a` wins immediately for `D`, its horizon outcome is `D@0`, the
maximum in (2), so it dominates every reply. Assume henceforth that neither
initial comparison needs that trivial clause.

Let `G=P+a` and `H=P+b`; `G` is the searched/better branch. Define the fixed
coordinate transposition

\[
 \phi(a)=b,\qquad \phi(b)=a,\qquad \phi(c)=c
 \text{ for }c\notin\{a,b\}. \tag{9}
\]

We construct the alternating simulation of Definition 6. Until the still
empty special cell is used, pair every placement outside `{a,b}` with itself.
At a `D` node, map an `H` move at `a` to `b` in `G`; at an `A` node, map a
`G` move at `b` to `a` in `H`. These maps are causal.

**Occupancy channel.** Before the special move, the branches have identical
outside occupancy, with `a` occupied by `D` only in `G` and `b` occupied by
`D` only in `H`. Thus (9) sends exactly the branch-exclusive empty cell to
the other branch-exclusive empty cell and fixes every common empty. If the
special move belongs to `D`, both states thereafter have `D` at both `a` and
`b`; their coloured boards, masks, and support agree, while a possible
`SecondStone{first}` payload remains paired by (9). If it belongs to `A`, the
states have

\[
 G:\ a=D,b=A,\qquad H:\ a=A,b=D, \tag{10}
\]

with all other stones identical.

**Legality-frontier channel.** At the initial successors, (7) says that the
support sets are identical. Common outside placements add identical balls to
both supports, so equality persists. The only unequal occupied cells are
`a,b`, which (9) exchanges. Hence (9) is a bijection between legal moves at
every paired prefix before the special move. When that move is made, both
occupied-coordinate sets become identical; their legal sets are then
identical as well. If the initial reply was a `FirstStone`, the two stored
`SecondStone{first}` coordinates are also paired by (9); both are already
occupied, and the remaining-placement budget is the same.

**Window-mask channel and early terminality.** Every window through `b` was
dead before either reply and stays dead by Lemma 2. Before the special move,
an `A`-winning window in `G` can contain neither `a` (a `D` stone) nor `b`
(empty), so it excludes both and is identically winning in `H`. A
`D`-winning window in `H` cannot contain `b` (dead) or `a` (empty), so it is
identically winning in `G`. Extra `D` wins through `a` in `G`, or extra `A`
wins through `a` in `H`, only improve the comparison and stop the relevant
trace immediately.

If `D` makes the special move, the two resulting coloured boards, masks, and
support sets are identical. If it was a `FirstStone`, their stored
`SecondStone{first}` payloads remain transposed rather than byte-identical;
because both stored cells are occupied, this has no additional rule effect.
Per-placement checking still needs one sentence: if placing `a` in
`H` would complete a `D` window not containing `b`, that same window already
contained `D` at `a` in `G` and would have ended `G` no later; a window that
also contains `b` was dead. Thus the only possible unequal terminal prefix is
an earlier `D` win in `G`, which is accepted. If both traces remain
nonterminal, their future rule transitions are equivalent under (9).

If `A` makes the special move, (10) applies. All windows through `b`,
including every window that also contains `a`, are permanently dead. On a
window through `a` but not `b`, (10) is exactly the favourable recolouring
`A` to `D` from `H` to `G`. For every window through the reverse-recoloured
cell `b`, P1-M was certified while both `a` and `b` were empty, so its old
`A` and `D` witnesses are common to both branches and lie outside the
recoloured pair. Thus Lemma 6's inert reverse-cell condition holds. An
immediate `A` win caused by the mapped `a` move can therefore occur only in
`H`, which again favours `G`. If neither trace terminates, Lemma 6 proves that
later identity play is defender-favouring.
This accounts for every member of \(\Omega(a)\cup\Omega(b)\), not merely a
chosen threat window or a count profile.

The terminal clauses and the actor-directed move maps now give an all-depth
alternating simulation. Lemma 4 yields \(b\preceq_n a\) for every `n`. QED.

#### P1 failure-mode audit

1. **Early terminality:** the transfer stops at the first terminal prefix.
   A crossing-window `D` win that occurs earlier in `G` is accepted; an `A`
   win in `G` is proved to occur in `H` at the same prefix. No proof assumes
   that a particular window is eventually completed.
2. **Legality frontier:** Lemma 7 proves that `b` adds no support. P1-LF
   separately requires the searched reply `a` to add no support (or requires
   exact successor-support equality in form (7)). The action swap is then a
   legal bijection at every prefix.
3. **Cross-window occupancy:** the proof treats all windows in
   \(\Omega(a)\cup\Omega(b)\). The displaced `b` stone is used only in windows
   already certified dead; every non-dead `a` window changes in the favourable
   owner direction. Attempts to play the branch-exclusive empty `a` or `b`
   are explicitly swapped.

### Lemma 8 (what an unmatched new frontier can contain) **[PROVEN]**

Let `c` be any legal empty reply in `P`, and let
\(z\in B_8(c)\setminus\Lambda(P)\). Every window through `z` contains no
pre-reply stone. After `D` plays `c`, such a window is either still empty or
contains only that one `D` stone.

Consequently, in this fresh region:

1. an `A`-alive window through `z` omits `c`, starts empty, and needs four
   later `A` placements for count 4 and six for a win;
2. a `D` count-4 needs at least three later `D` placements and a `D` win at
   least five when the window contains `c` (otherwise four and six).

*Proof.* Since `z` was unsupported, `d(z,s)>8` for every old stone `s`.
Any two cells of one length-6 window have distance at most 5, so no window
through `z` can contain an old stone. The post-reply counts and the placement
lower bounds follow. QED.

Lemma 8 is the requested completion/tempo accounting, but it is **not** a
substitute for P1-LF. A count-4 or multi-window fork can change forcing
obligations before a six-stone completion; forced hits may feed another
window; and a first unmatched frontier move can legalize further unmatched
moves. Those are precisely the Section 6/8.1 failure modes of the naive
reordering and completion-only arguments.

**CONJECTURE P1-F (not used).** P1-LF may be replaceable, for a fixed
certificate and finite horizon, by a causal boundary-state simulation that
tracks every unmatched frontier chain and every count-4 interaction. Merely
bounding fresh completions by six placements is insufficient. No such
certificate-level simulation is proved here.

---

## 4. P2 -- interchangeable hitting cells

Two empties of one threat window are not interchangeable merely because that
window has the same count whichever one is hit. Ordinarily the other 17
windows through each cell are different. P2 gives a narrow condition under
which all of those differences are permanently inert.

### Pattern P2 (dead-spoke interchangeable hits) **[PROVEN]**

Let `P` be a nonterminal post-opening position with `D` to place. Let `W` be
an `A`-alive window with exactly four `A` stones and exactly two empty cells
`x != y`. Both cells are legal: each shares `W` with an old stone at distance
at most 5. Assume:

1. **P2-M (dead nonshared spokes):** every window in

   \[
   (\Omega(x)\cup\Omega(y))\setminus\{W\}
   \]

   is already dead in `P`;
2. **P2-LF (frontier equivalence):**

   \[
   \Lambda(P)\cup B_8(x)=\Lambda(P)\cup B_8(y). \tag{11}
   \]

Then the defender replies at `x` and `y` are mutually outcome-dominating at
every horizon:

\[
 x\preceq_n y\quad\text{and}\quad y\preceq_n x
 \qquad\text{for every }n. \tag{12}
\]

Thus a solver may retain one canonical member of `{x,y}` at that defender
node.

*Proof.* A `D` stone at either `x` or `y` makes `W` two-coloured. By P2-M and
Lemma 2, after either reply **every** window through **either** special cell is
permanently dead, including `W`; this holds even at the counterpart cell that
is still empty.

Compare `G=P+x` and `H=P+y` and transpose `x<->y`, fixing every other
coordinate. The channel proof is now symmetric.

**Occupancy.** Before the counterpart is used, the transposition exchanges
the only branch-exclusive occupied/empty pair. If the counterpart is later
used, the occupied-coordinate sets become identical. Its owner is the same
current actor in both paired traces.

**Legality frontier.** Equation (11) makes the initial successor supports
identical. Common outside placements add the same balls. A special-cell move
adds the other special ball, after which both supports contain both balls.
Thus the transposition bijects legal moves at every paired prefix, in both
directions and for both actors.

**Windows and terminality.** Every differing bit belongs to a window through
`x` or `y`, and every such window is dead after the initial hit. It can never
become winning, active, or a member of a hitting set. Every window outside
their union has identical labelled masks. Therefore terminal events outside
the dead union occur for the same player at the same placement; none can
occur inside it. This is an outcome-preserving bisimulation, so Lemma 4 works
in both directions at every depth. QED.

#### Why count profiles are insufficient

It is not enough that the other windows through `x` and `y` have equal pairs
`(cnt_A,cnt_D)`. The labelled empty locations and the intersection hypergraph
matter. For example, two count-4 windows can have a common remaining hit in
one neighborhood and disjoint remaining hits in the other; their count
profiles agree while their minimum hitting sets differ. Likewise, two
count-3 windows with the same counts may or may not share a fork-creating
cell. An exact broader equivalence must preserve labelled masks, window
incidence, occupancy, and radius-8 support throughout the `n`-causal cone (or
be a genuine colour-preserving `D6` automorphism of the complete position).
P2-M avoids that unproved classification by making every differing window
permanently dead.

#### P2 failure-mode audit

1. **Early terminality:** neither hit wins through an incident window: `W`
   contains `A`, and every other incident window is already dead. Later
   terminal events occur outside the differing window union and are identical
   at the same placement. The simulation stops immediately there.
2. **Legality frontier:** P2-LF supplies exact successor-support equality;
   the `x<->y` transposition is proved legal at every prefix, not inferred
   from both cells lying in `W`.
3. **Cross-window occupancy:** P2-M quantifies over the full union
   \(\Omega(x)\cup\Omega(y)\), not just `W` and not just counts. A later attempt
   to occupy the branch-exclusive counterpart is explicitly transposed.

---

## 5. P3 -- two placements of one turn commute

This pattern has direct solver value: many directed two-placement turns can
be represented by one unordered pair. The qualification about singleton wins
is essential because a winning first placement suppresses the second.

### Pattern P3 (nonwinning-prefix turn commutation) **[PROVEN]**

Let `P` be a nonterminal `FirstStone` position (for either player `X`). Let
`p != q` satisfy:

1. both `p` and `q` are legal already in `P`; and
2. neither single transition `P+p` nor `P+q` is terminal.

Then both two-placement traces `p;q` and `q;p` are legal. They have the same
qualitative rule outcome after the second placement:

1. if nonterminal, they have identical occupancy, labelled window masks,
   legal support, player to move, and `FirstStone` phase;
2. if terminal, both terminate on the second placement with winner `X` and
   the same absolute placement count.

*Proof.*

**Occupancy and legality.** The cells are distinct. Because `p,q in L(P)`,
placing either one leaves the other empty and it retains its old legality
justification; legal support only grows. Both orders are therefore legal.
After two placements their occupied-coordinate maps are identical.
Moreover

\[
 \Lambda(P)\cup B_8(p)\cup B_8(q)
\]

is independent of order, so their final legal sets are identical.

**All window masks.** A placement at `p` ORs the `X` bit for `p` into every
mask in `Omega(p)`; similarly for `q`. OR at two distinct bit positions
commutes. This remains true for a crossing window containing both cells.
Every window outside \(\Omega(p)\cup\Omega(q)\) is unchanged. Hence the final
labelled masks are identical.

**Per-placement terminality.** By hypothesis neither possible first
placement wins. If the common final board contains an `X`-winning window,
that window must contain both `p` and `q`: a final winning window containing
only `p` would already have won after singleton `p`, and similarly for `q`.
Whichever cell is second therefore completes the window on the second
placement in both orders. If no final window wins, both traces finish the
turn and advance to the other player at `FirstStone`. QED.

The live `HexoState` values need not be field-identical: ordered
`placement_history` and `last_turn` differ. Their exported snapshots and
serialized `Board` representations also retain placement order. If the
second placement wins, phase advance is suppressed, so the live terminal
states retain respectively `SecondStone{first:p}` and
`SecondStone{first:q}`. P3 equates rule outcomes, not order-sensitive model
features, live metadata, or serialized representations.

### Solver corollary

At a `FirstStone` node with initial legal set `L`, define

\[
 L_0=\{c\in L:P+c\text{ is nonterminal}\}.
\]

Every directed trace on distinct `p,q in L_0` may be generated once as an
unordered pair, including a pair that wins jointly on its second placement.
This reduces \(|L_0|(|L_0|-1)\) traces to \(\binom{|L_0|}{2}\). A correct
implementation must retain or separately resolve:

1. every singleton immediate win, because the game stops before a pair (P3
   makes no claim about directed traces involving a cell outside `L_0`);
2. directed pairs `p->q` when `q` was not legal in `P` and becomes legal only
   through `B_8(p)`; and
3. per-placement intermediate nodes whenever the proof/certificate format or
   learned heuristic depends on their ordered history. Deduplication is safe
   at the completed-turn rule-outcome layer, not by deleting an arbitrary
   `FirstStone` child.

#### P3 failure-mode audit

1. **Early terminality:** both possible singleton prefixes are explicitly
   required nonterminal. Any joint win is proved to occur on the second
   placement in both orders. Immediate wins remain singleton actions.
2. **Legality frontier:** both cells must be legal *before* the turn. Old
   legality persists, and the final frontier is the commutative union of the
   same two balls. Newly legalized second cells are excluded from sorting.
3. **Cross-window occupancy:** every mask in
   \(\Omega(p)\cup\Omega(q)\) is covered. In a window through both cells, the
   two owner-bit insertions commute; such a crossing window is also exactly
   where a genuinely joint second-placement win can occur.

---

## 6. Relationship to threat analysis and certificate horizons

These patterns concern exact rule outcomes. The \(\lambda^1\) threat analysis in
`threats_shared.rs` is a sound leaf predicate but is not a terminal transition.
In particular:

1. horizons and terminal comparisons are counted in placements, not turns;
2. `FirstStone` and `SecondStone` have different remaining budgets;
3. an own win is checked before a hitting-set forced loss; and
4. a pattern transfer must preserve any window masks used by a certificate
   leaf, unless those windows are proved permanently dead.

P1 and P2 are all-horizon rule simulations, so any mask-derived leaf used on
the common live windows transfers as well. P3 supports completed-turn
deduplication; it does not assert that the two intermediate `SecondStone`
nodes have identical threat sets or learned values.

---

## 7. Finite local verification methodology

The proofs above are analytic. This section specifies independent exhaustive
checkers. They verify the local channel implications, not a sampled game tree.

### Definition 7 (finite encoding)

For mask checks, a cell has three states: empty, `A`, or `D`. For cells that
can affect only legal support, owner is irrelevant and a binary
empty/occupied bit suffices. Candidate cells and any stated threat stones are
fixed rather than enumerated.

The displayed configuration counts are *spatial channel assignments for a
fixed scalar rule case*. MV-P1 and MV-P2 are instantiated once at
`FirstStone` and once at `SecondStone` (multiply a literal combined-state
bound by 2). MV-P3 fixes the mover to one colour using the independent global
`A<->D` relabelling symmetry of the rules; without that extra colour symmetry,
multiply its literal bound by 2. These scalar cases are not part of `D6`.

To decide whether a point of `B_8(c)` was already supported, every possible
old supporting stone lies within distance 8 of that point, hence within
`B_16(c)`. Thus:

1. radius 5 (the 31-cell star `X(c)`) suffices for incident masks;
2. radius 8 suffices if the old support bitset is supplied as boundary data;
3. radius 16 suffices to derive that bitset from raw occupancy.

The hex-ball sizes used below are

\[
 |B_R|=1+3R(R+1),\qquad |B_8|=217,qquad |B_{16}|=817. \tag{13}
\]

### Definition 8 (`D6` quotient)

Translations anchor the marked focal geometry; this is a local coordinate
normalization, not a claim that the occupied opening origin may be discarded.
The twelve rotations/reflections of `D6` then act on marked configurations
without swapping colours.

For a fixed marked geometry with stabilizer `G <= D6`, let `M` be the free
ternary mask cells and `O` the free occupancy-only cells. The exact number of
unfiltered assignment orbits is given by Burnside's lemma:

\[
 N_{\rm orb}=\frac1{|G|}\sum_{g\in G}
       3^{c_M(g)}2^{c_O(g)}, \tag{14}
\]

where `c_M(g)` and `c_O(g)` are the numbers of cycles on the respective free
cells. Cycles containing differently fixed labels contribute zero and are
discarded. The predicates below then filter these orbits. One must not divide
a raw count by 12: symmetric assignments have nontrivial stabilizers.

All counts stated below quotient overlapping marked geometries by `D6`, and
(14) additionally quotients residual colouring symmetry. Where two marked
foci may be arbitrarily far apart, `D6` alone leaves infinitely many distance
orbits. MV-P1 and MV-P3 explicitly use one further equivalence: disjoint
truncated neighborhoods are factored as one local interaction type after
proving that exact separation occurs in no checked channel predicate.

---

## 8. Machine-verification specifications

### MV-P1 (dead-cell dismissal)

**Neighborhood.** Enumerate

\[
 U=B_{16}(a)\cup B_{16}(b),\qquad
 M=X(a)\cup X(b).
\]

Only cells in \(B_8(a)\mathbin{\triangle}B_8(b)\) need support comparison, but `U`
contains every possible witness. For displacement distance at most 32, the
number of `D6` displacement types is

\[
 \sum_{r=1}^{32}(\lfloor r/2\rfloor+1)=288.
\]

There are infinitely many literal `D6` displacement orbits on the unbounded
board. However, all distances greater than 32 induce the same disjoint local
incidence type: the two radius-16 patches have no shared cell, and their exact
separation is absent from every predicate. Quotienting these locally
isomorphic disjoint cases gives 289 marked local geometries. Since
`|U|<=1634`, `|M|<=62`, and `a,b` are fixed empty, a conservative
channel-space bound is

\[
 289\cdot 3^{60}\cdot 2^{1572}. \tag{15}
\]

This is intentionally an upper bound; (14) gives the exact orbit count for
each displacement representative.

As a smaller regression subcheck, deadness at a single fixed empty `b`
factorizes into three independent 10-cell axis strings. Of `3^10=59,049`
strings, exactly 22,448 make all six through-`b` intervals contain both
colours; 108 of those are fixed by reversal. Hence there are

\[
 22,448^3=11,311,832,379,392
\]

raw dead stars and, by Burnside over `D6`, exactly

\[
 \frac{V^3+2P+2V+P^3+3V^2+3PV}{12}
 =942,779,391,290,\quad V=22,448,\ P=108, \tag{16}
\]

dead-star colorings up to `D6`.

**Filter and predicate.** Take global nonterminality and reachability as input
preconditions/common boundary data. For every orbit representative:

1. fix `a,b` empty and require both candidates legal;
2. check every one of the 18 `b` windows is two-coloured;
3. if `P+a` is an immediate `D` win, verify the `D@0` terminal clause and
   accept without a frontier test; otherwise compute old support from
   occupancy in `U` and check (7);
4. in the non-immediate branch, create the two successor mask tables;
5. in that branch, check the three inductive relation classes used in the proof: counterpart
   still empty, counterpart filled by `D`, and counterpart filled by `A`;
6. in each such class, verify the `a<->b` legal-action bijection, all masks in
   \(\Omega(a)\cup\Omega(b)\), and the terminal acceptance clauses.

Cells and windows outside `U` are a boundary signature common to both
branches. They cannot distinguish the branches by Lemma 1. Enumerating a
superset of locally consistent patterns is sound for validating the local
implication.

### MV-P2 (interchangeable hitting cells)

Normalize `W` to one axis. Its 15 unordered pairs of empty positions have
nine shapes under window reversal. By separation `d=1,...,5`, their
multiplicities are respectively

\[
 3,2,2,1,1. \tag{17}
\]

For centers `x,y` at separation `d` on an axis:

\[
 |X(x)\cup X(y)|=49+d,
\]

\[
 |\Omega(x)\cup\Omega(y)|=30+d,
\]

so P2-M requires exactly `29+d` distinct non-`W` windows to be dead. Raw
support derivation uses

\[
 U=B_{16}(x)\cup B_{16}(y),\qquad |U|=817+33d.
\]

The six cells of `W` are fixed (four `A`, two empty). Thus the free ternary
count is `(49+d)-6=43+d`, and the occupancy-only count is
`(817+33d)-(49+d)=768+32d`. After the nine-shape geometric quotient, the
conservative configuration bound is

\[
 \begin{aligned}
 &3\cdot3^{44}2^{800}
 +2\cdot3^{45}2^{832}
 +2\cdot3^{46}2^{864}\\
 &\qquad+3^{47}2^{896}+3^{48}2^{928}. \tag{18}
 \end{aligned}
\]

Residual reflections are quotiented exactly by (14).

**Filter and predicate.** Take global nonterminality and reachability as input
preconditions/common boundary data. Require `W=A^4 E^2` at the marked cells,
both hits legal, every one of the `29+d` other incident windows two-coloured,
and (11).
Apply one `D` hit in each branch. Verify that all `30+d` incident windows are
dead, that the `x<->y` action involution preserves legality in the three
counterpart-occupancy classes, and that all possible changed masks satisfy
the exact terminal/bisimulation predicate. The check uses labelled masks and
window keys, never counts alone.

### MV-P3 (same-turn commutation)

**Neighborhood.** Use

\[
 U=B_8(p)\cup B_8(q),\qquad M=X(p)\cup X(q).
\]

Radius 8 is sufficient here: the predicate needs only initial legality and
the algebraic equality of the two final support unions, not a comparison with
an alternative old support. There are

\[
 \sum_{r=1}^{16}(\lfloor r/2\rfloor+1)=80
\]

`D6` displacement types with overlapping radius-8 patches. Literal `D6`
orbits at distances greater than 16 are infinite, but they all induce one
disjoint local-incidence type for this two-placement predicate. With that
additional local-isomorphism quotient, there are 81 marked cases. With
`|U|<=434`, `|M|<=62`, and `p,q` fixed empty, a conservative bound is

\[
 81\cdot3^{60}\cdot2^{372}. \tag{19}
\]

Again, (14) removes residual symmetry.

**Filter and predicate.** Take global nonterminality and reachability as input
preconditions/common boundary data. Require the `FirstStone` phase, distinct
initially legal `p,q`, and no winning window after either singleton. Run both
exact two-placement orders. Check:

1. the second placement is legal in both;
2. every final labelled mask and the final legal set agree;
3. either both are nonterminal with the same next player/`FirstStone` phase,
   or both terminate on placement two with the same winner and placement
   count; and
4. do **not** require ordered history or the retained terminal
   `SecondStone{first}` payload to agree.

This checker directly catches each omitted side condition: removing the
singleton-nonwin filter produces early-terminal counterexamples, and removing
initial legality produces one-way frontier extensions.

---

## 9. Proven solver rules and nonclaims

The following rules may be consumed by a proof-carrying solver.

1. **P1 rule:** omit a dead reply `b` only when the certificate names a
   searched reply `a` satisfying P1-LF (or `a` wins immediately). In this
   geometry, P1-LF is the simple frontier-inert test
   \(B_8(a)\subseteq\Lambda(P)\).
2. **P2 rule:** for a count-4 attacker window with empties `x,y`, retain one
   hit only when all `29+d` other incident windows are dead and successor
   supports satisfy (11).
3. **P3 rule:** canonicalize a completed two-stone trace as an unordered pair
   only when both cells were legal at the turn start **and both singleton
   successors were nonterminal**. Keep immediate wins, traces involving an
   immediate-winning cell, and newly-legalized directed continuations
   separate unless another proof rule resolves them.

Not claimed:

1. a dead-cell reply is not proved dominated by an arbitrary
   frontier-extending searched reply;
2. equal window counts do not make hitting cells interchangeable;
3. stones of one colour are not globally monotone-helpful when their legal
   frontier differs, because the opponent may use the newly opened cells;
4. no transferred strategy is continued after a terminal placement; and
5. the prospective enumerations in Section 8 are specifications, not evidence
   already executed.

These limitations are deliberate. They are exactly where the naive
reordering theorem and the first relevance-zone accounting in
`PLAN_TSS_MOVESET_ZONES.md` failed.
