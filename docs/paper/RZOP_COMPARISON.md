# RZOP Comparison and Follow-up Plan

Comparison of **"Certified Defender Move Restriction in Hexo Threat Search"**
(v3 standalone, commit `e1fb863`, `docs/paper/`) against the closest prior
art:

> I-Chen Wu and Ping-Hung Lin, **"Relevance-Zone-Oriented Proof Search for
> Connect6,"** IEEE Transactions on Computational Intelligence and AI in
> Games, vol. 2, no. 3, pp. 191–207, Sept. 2010.
> DOI: 10.1109/TCIAIG.2010.2060262.

Written 2026-07-15. Purpose: identify duplicated ground to defer via
citation, genuinely novel ground to keep and expand, and gaps to fill in a
follow-up revision.

---

## 1. Verdict in one paragraph

The paper is **not mostly duplicated** — the gate calculus, sparse LOSS caps,
pairing threshold, potential boundary, mechanization discipline, and limit
map have no counterpart in Wu & Lin. But the **core problem and the top-level
idea are prior art from 2010**: RZOP search restricts defender replies in a
six-in-a-row, two-placements-per-turn game to zones derived from the
attacker's winning strategy, with an incremental ranked structure. The paper
currently cites neither Wu & Lin nor Thomsen's lambda search, even though §2
names its tactical predicates "λ¹" — terminology that descends from Thomsen
via Wu & Lin. The correct story is stronger than the current from-nothing
framing: *RZOP solved defender restriction for Connect6; Hexo's radius-8
legality rule breaks their zone semantics; this paper repairs it with ranked
seed bands and per-window clocks.*

## 2. The prior paper in brief

Setting: Connect6 = Connect(19,19,6,2,1), also analyzed on infinite boards
as Connect(6,2,1). Square lattice, four line directions, six in a row, two
stones per turn (one on the opening move), **any empty square is legal** —
no locality rule.

Contributions:

1. **Modified lambda search (Λᵃ-trees).** Thomsen's λ-trees adapted to
   p-stones-per-move games via *null moves* (defender places nothing) and
   *seminull moves* (defender places one stone of two). VCDT = Λ¹-strategy,
   VCST = Λ²-strategy, higher orders for nonthreat winning moves. They solve
   up to Λ³.
2. **Relevance zones.** Incremental zone sequences ⟨Z₁,…,Zₙ⟩ with the
   semantics: the attacker's verified win survives any "irrelevant" sequence
   of extra defender stones (the i-th extra stone outside Zᵢ). Zone algebra
   (their Lemmas 1–3): appending, pairwise containment/monotonicity,
   promotion (left-shift when a seminull stone is absorbed), per-square
   deletion.
3. **The RZOP verifier** (Property RZV), recursive on threat count:
   - EP-1 (endgame): empties of defender active segments with ≤ i empties
     enter Zᵢ (defender counter-win / inversion hazards).
   - AT-1 (attacker turn): child zone ∪ attacker move squares.
   - T3-1/2 (≥3 threats): threat empties into all Zᵢ + EP-1-style segments.
   - T2, T1 (two / one threat): critical defenses only (normal + relaxed),
     recursing with zone merging; seminull-move proof search inside the
     blocking square's zone.
   - T0 (no threats): null-move win first → zone Z₁; then seminull search
     for each square of Z₁ → Z₂; etc.
4. **Practical results:** 65 positions solved with Λ²-strategy including 12
   three-move openings (Mickey Mouse opening among them); one Λ³ position;
   NCTU6 + NCTU6-Verifier + human experts + job-level proof-number search.
5. **Appendix:** generalization to all Connect(k,p,q) with a subverifier
   (Property RZS) and macromove-dominated defenses.

Notable weaknesses of the prior paper (relevant when deciding what to defer):
Theorem 1's proof is omitted; several lemma proofs are "similar and
omitted"; the strategy-replay soundness argument is informal; higher-order
zones are truncated in practice ("simply use Z_∞ whenever needed"); no
mechanized checking.

## 3. Game-structure comparison

| | Connect6 | Hexo |
|---|---|---|
| Lattice | Square (19×19 or infinite) | Hex axial, unbounded |
| Line directions | 4 | 3 |
| Win length | 6 | 6 |
| Placements/turn | 2 (opening 1) | 2 (opening 1) |
| Legality | Any empty square | Within distance 8 of any stone |
| Draws | Board fill (finite board) | None (engine) |

Hexo is, structurally, hex-lattice Connect6 plus a locality rule. The
locality rule is what generates most of the paper's hard content; on the
square lattice with radius ∞ the seed and virgin machinery degenerates and
the ranked zone collapses toward RZOP's construction.

## 4. Concept correspondence

| Wu & Lin | This paper | Disposition |
|---|---|---|
| Zone sequence ⟨Z₁,…,Zₙ⟩, i-th extra stone ranked by index | Ranked zone; per-role ranks r_N(y) (D15), per-window exposures E^D (D16) | **Cite as ancestor.** Ours is finer (per-cell deadlines, entry-stopped clocks) and adaptive per node; theirs is a static sequence with a composable promotion algebra. |
| AT-1: attacker future-move squares | Direct obligations Prot(N) / Z_dir (D10, D11) | Equivalent. Cite. |
| T3-1 threat empties; EP-1/T3-2 defender segments with few empties | LOSS witness empties; touched-window guards cnt+E ≥ 6 | Equivalent hazard; ours has an explicit completion-race proof (L12). Cite. |
| Critical defenses only under threats (T1/T2) | Extendable-hit kernel K_b (T6); forcing gates (D19) | Same coarse idea. **Keep and contrast**: our gate adds the checkpoint, debited f/Q^D clocks, and the disjoint-hit race counterexample they never state. |
| Informal "replay the same strategy" soundness | Real–ghost coupling, X/Y sets, filler subroutine, mask inequality (MI), T3 | **Keep.** This is the rigorous version of their omitted/onmitted-proof arguments; say so once. |
| Null / seminull moves as search organization | No analogue (certificate assumed found) | **Gap — solver-side.** See §6.2. |
| Macromove combining of equivalent defenses | D17 substitution envelopes, D18 DAGs (related, not identical) | **Gap — one-remark fix.** See §6.4. |
| Λᵃ-order hierarchy as difficulty taxonomy | Certificate depth / budget only | Minor gap; optional remark. |
| Solver/verifier separation (experts + NCTU6-Verifier) | Search/verification separation (§1, §13) | **Not novel to us.** Reframe as inherited architecture; cite. |
| Zone promotion algebra (Lemmas 1–3) | Nesting/monotonicity facts (L11), DAG label consistency | Mostly equivalent bookkeeping. Cite; no content change needed. |
| Connect(k,p,q) generalization (appendix) | Hexo-specific constants | Gap; future work. See §6.5. |
| 65 solved positions, node counts, Mickey Mouse | Search-harness measurements + negative controls only | **Gap — applied capstone.** See §6.3. |

## 5. Novel ground — keep and, where marked, expand

1. **Legality-frontier machinery** (the paper's core novelty; nothing in
   RZOP corresponds): ranked seed bands B_{8(r−1)}, ghost-illegal legality
   chains, virgin-window reach radii, clause Z4. *Expand:* this is the
   headline; the related-work section should lead with it.
2. **Local clocks** finer than RZOP's single sequence index: per-role
   deadlines, per-window exposure stopping at attacker entry, attained
   fixed-window boundaries. *Keep.*
3. **Forced-hit gate calculus** (§7) with the reachable disjoint-hit race
   example. RZOP restricts to critical defenses without addressing the
   defender-ignores-and-races hazard explicitly (their zones absorb it
   implicitly via EP-1 segments). *Keep; add one contrast sentence.*
4. **Sparse LOSS witnesses** (3 at b=1, 5 at b=2, both attained). No
   analogue. *Keep.*
5. **Pairing threshold** (no k=6 covering matching; index-12 periodic
   perfect matching at k=7). Different tradition (positional games); new
   for the hex lattice. *Keep; add Hales–Jewett / Winning Ways citations
   for pairing strategies generally.*
6. **Potential boundary layer** (√3 base, fixed-family/finite-region
   certificates, 124-branch greedy refutation, clean escape, five GAPs).
   No analogue in the game-solving line; Erdős–Selfridge/Beck already
   cited. *Keep.*
7. **Mechanization + limit-map discipline** (PROVEN vs PROVEN-MECH, the
   validation-boundary section, tightness map). RZOP is experiments-only.
   *Keep.*
8. **Two-placement turn-structure care**: win checked between placements,
   adaptive LOSS remainders, budget-aware transitions. RZOP treats the
   two-stone move atomically. *Keep.*

## 6. Gaps and follow-up actions

### 6.1 Related work (mandatory, cheap)

No game-solving lineage is cited. Add a related-work subsection to
`sections/01-introduction.tex` and entries to `refs.bib`. Framing for each:

- **Wu & Lin 2010 (RZOP)** — direct ancestor; cite at: intro related-work,
  the ranked-zone definition (sequence-of-zones ancestry), the gate section
  (critical-defense restriction), validation (their solved-position
  benchmark), conclusions/future work (their open problems).
- **Thomsen 2000 (lambda search)** — the λ¹ predicate naming in D6 descends
  from this; cite at first use of λ¹.
- **Wu & Huang 2006** (the Connect(m,n,k,p,q) family) and/or
  **Wu, Huang & Chang 2006** (Connect6, ICGA J. 28(4):234–242) — game
  family context.
- **Allis, van den Herik & Huntjens 1996** (Go-Moku solved),
  **Wágner & Virág 2001** (Renju solved) — threat-space solving lineage
  (allis-1994 is already cited for pn-search/TSS; the solved-game results
  deserve their own citations).
- **van den Herik, Uiterwijk & van Rijswijck 2002** — games-solved survey.
- **Cazenave 2001** (abstract proof search), **Cazenave 2003** (generalized
  threats search) — optional, one clause.
- **Wu et al. 2010 (JL-PN)** — job-level proof-number search; optional, in
  the solver-implications discussion.
- **Soeda, Kaneko & Tanaka 2006** (dual lambda search) — optional; their
  open problem 5 asks whether it helps Connect games; relevant to our
  defender-side questions.
- **Hales & Jewett 1963**; **Berlekamp, Conway & Guy, Winning Ways** —
  pairing-strategy tradition, cite in §9.

Bib-ready entries (verify page ranges before committing):

```bibtex
@article{wu-lin-2010-rzop,
  author  = {I-Chen Wu and Ping-Hung Lin},
  title   = {Relevance-Zone-Oriented Proof Search for {Connect6}},
  journal = {IEEE Transactions on Computational Intelligence and AI in Games},
  volume  = {2}, number = {3}, pages = {191--207}, year = {2010},
  doi     = {10.1109/TCIAIG.2010.2060262}
}
@article{thomsen-2000-lambda,
  author  = {Thomas Thomsen},
  title   = {Lambda-Search in Game Trees --- with Application to {Go}},
  journal = {ICGA Journal},
  volume  = {23}, number = {4}, pages = {203--217}, year = {2000}
}
@inproceedings{wu-huang-2006-family,
  author    = {I-Chen Wu and Dei-Yen Huang},
  title     = {A New Family of $k$-in-a-Row Games},
  booktitle = {Advances in Computer Games 11},
  series    = {LNCS}, volume = {4250}, pages = {180--194},
  publisher = {Springer}, year = {2006}
}
@article{allis-1996-gomoku,
  author  = {L. Victor Allis and H. Jaap van den Herik and
             Matty P. H. Huntjens},
  title   = {Go-Moku Solved by New Search Techniques},
  journal = {Computational Intelligence},
  volume  = {12}, pages = {7--23}, year = {1996}
}
@article{wagner-virag-2001-renju,
  author  = {J{\'a}nos W{\'a}gner and Istv{\'a}n Vir{\'a}g},
  title   = {Solving {Renju}},
  journal = {ICGA Journal},
  volume  = {24}, number = {1}, pages = {30--34}, year = {2001}
}
@article{herik-2002-solved,
  author  = {H. Jaap van den Herik and Jos W. H. M. Uiterwijk and
             Jack van Rijswijck},
  title   = {Games Solved: Now and in the Future},
  journal = {Artificial Intelligence},
  volume  = {134}, number = {1--2}, pages = {277--311}, year = {2002}
}
@article{hales-jewett-1963,
  author  = {Alfred W. Hales and Robert I. Jewett},
  title   = {Regularity and Positional Games},
  journal = {Transactions of the American Mathematical Society},
  volume  = {106}, pages = {222--229}, year = {1963}
}
```

A pending task chip ("Add related-work section to TSS zones paper") covers
this edit; this file is its analysis backing.

### 6.2 Null/seminull-move search organization (solver-side; runbook, not paper)

RZOP's operational trick: prove a win against the *null* move first
(cheap), read the zone off that proof, expand only zone replies, recurse
per seminull square. Our framework derives zones from a certificate that
search must already have produced against all zone replies — a
chicken-and-egg the solver resolves iteratively. Importing null-move-first
ordering into the TSS solver (a hypothetical "defender passes" probe used
only for zone discovery, never as a game move) could cut certificate
discovery cost substantially. Candidate home: PLAN_TSS_SOLVER_UPGRADES
ladder, not the paper.

### 6.3 Applied capstone (biggest content gap)

Wu & Lin anchor their paper with 65 solved positions and node counts. Our
§11 honestly records that the mechanized search runs validated the
*predecessor* generator, not the new ranked closure, and no position is
solved end-to-end *by* the new verifier. Follow-up: run one nontrivial
position (runbook rung ladder) through search → certificate compilation →
ranked-zone verification, and report nodes/zone sizes/uniform-vs-exact
clock comparison. This single addition would move the paper from
theory-only to demonstrated.

### 6.4 Macromove combining (one-remark fix)

RZOP combines defensive moves with an identical winning continuation into
one macromove (>p stones), shrinking both tree and zones. D17/D18 cover
adjacent ground (substitution, suffix sharing) but not this optimization.
Add a remark near D17 noting the relation and that a macromove analogue
under the radius-8 rule would need a frontier-equivalence premise.

### 6.5 Parameterized generalization (future work)

Their appendix covers all Connect(k,p,q). Our constants (6, 8, two
placements) are baked in. A (k, p, legality-radius ρ) treatment — with
RZOP as the ρ=∞ square-lattice limit — would position the frameworks
cleanly. Big lift; future-work paragraph only.

### 6.6 Λ-order taxonomy (optional)

Their VCDT/VCST/Λᵃ ladder classifies positions by forcing depth and
supplies open problems (does a Λ⁴ position exist?). We have certificate
depth but no taxonomy. Optional one-paragraph remark in conclusions; also a
natural axis for the capstone experiment (what order does the tested
position need?).

### 6.7 Their open problems worth tracking

From their §VI: (1) more Λ³ positions; (2) existence of Λ⁴; (3) better
pruning; (4) general Connect games; (5) dual lambda search (defender-side
proofs — relevant to our LOSS/defender-survival machinery, which they
never develop; our defender-side results in §8–§10 partially answer the
Hexo analogue and can be framed as such).

## 7. Quality notes from the review (for the revision pass)

- Strengths to preserve: contract-style statements, plainterms glosses,
  threaded running example, reader routing, validation-boundary honesty,
  limit map, PROVEN vs PROVEN-MECH separation.
- The real–ghost coupling is a genuine improvement over the
  replay-and-hope soundness arguments in this literature; one sentence in
  related work may say so (dry tone: "the coupling makes explicit the
  replay argument left informal in [RZOP]").
- Bibliography is thin beyond the game-solving gap (7 entries / 53 pp);
  the pairing and potential sections are properly rooted
  (Erdős–Selfridge, Beck) but pairing needs Hales–Jewett / Winning Ways.
- No change to any theorem is required by the comparison: nothing in
  Wu & Lin invalidates or preempts a specific result in the paper.

## 8. Suggested edit map (when the revision happens)

| File | Edit |
|---|---|
| `sections/01-introduction.tex` | New related-work subsection (RZOP, lambda search, Go-Moku/Renju/survey, pairing tradition); reframe contribution as repairing zone semantics under the legality rule. |
| `sections/02-game.tex` | Footnote or clause at D6: λ¹ naming after Thomsen; Connect6 kinship of the rules. |
| `sections/06-computable-zones.tex` | Remark near D13/T4: ranked zone vs RZOP incremental sequences; remark near D17: macromove combining. |
| `sections/07-forced-hit-gates.tex` | One sentence: kernel vs RZOP critical-defense restriction; the race example is the hazard their restriction leaves implicit. |
| `sections/11-validation.tex` | One sentence contrasting scope with RZOP's 65-position benchmark; capstone run when available. |
| `sections/13-conclusions.tex` | Future work: parameterized (k,p,ρ) framework; null-move-first discovery; Λ-order taxonomy; dual-lambda / defender-side certificates. |
| `refs.bib` | Entries from §6.1. |

Source PDF of the prior paper (uploaded copy, not in repo):
`C:\Users\epicm\.claude\uploads\a2c04c98-f36c-4bcd-9ef1-f67973f155fd\ae73f06f-RelevanceZoneOriented_Proof_Search_for_Connect6.pdf`
