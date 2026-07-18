# RZOP Solver-Optimization Suggestions

Second follow-up document derived from the prior-art comparison against

> I-Chen Wu and Ping-Hung Lin, **"Relevance-Zone-Oriented Proof Search for
> Connect6,"** IEEE Trans. Comput. Intell. AI Games, vol. 2, no. 3,
> pp. 191–207, 2010. DOI: 10.1109/TCIAIG.2010.2060262.

**Sibling to** [`RZOP_COMPARISON.md`](RZOP_COMPARISON.md). That document is
*paper-primary* (citations, related work, positioning). **This one is
solver-primary**: its goal is a maximally optimized Hexo TSS solver that
searches the smallest necessary state space for **both** players. Where the two
overlap, this document cross-references rather than repeats.

Written 2026-07-15. Backing: a six-lens adversarially-verified mining pass over
the paper against our proof (`companion/PROOF_TSS_DEFENDER_ZONES.md`), the
solver plan (`docs/PLAN_TSS_SOLVER_UPGRADES.md`), and the Rust engine
(`packages/hexfield_eq/rust/src/tss_*.rs`). Line numbers are as of the current
worktree HEAD; treat them as hints, not anchors. The central code claim was
re-verified by hand (see §1).

---

## 0. Verdict in one paragraph

**Attacker-width is the binding constraint, not defender branching.** The OR
(attacker) generator gates every candidate on a pre-placement window count ≥ 3
(`tss_solver.rs:1061`), so count-2 pair-builds — two same-turn placements that
together reach a double-four — are *structurally never proposed*. That single
gate is why the external VCF corpus proves 0 of 14 winnable roots (0 false
wins, all honest UNKNOWN). No defender-side change, no caching change, and — the
correction this pass forced — **not the RZOP null-move trick** touches it. An
optimal Hexo solver is therefore: (a) an enriched OR generator that emits
count-2 VCDT pair-builds and, above that, radius-8-bounded quiet setups; (b)
scheduled by **λ-order iterative deepening**, which bounds the exponentially
branching *non-forcing* links while leaving cheap forcing chains unbounded in
depth; (c) with RZOP's **T2 macromove** as the one genuine defender-side fan-out
collapse. The radius-8 legality rule — the source of our whole defender-side
contribution — also makes (a)/(b) *more* tractable than whole-board Connect6 and
*caps* (c). Everything else the mining surfaced is accelerator, telemetry, or
paper framing.

This reorders our intuition. Going in, the null-move "zone discovery" idea (from
`RZOP_COMPARISON.md` §6.2) looked like the headline solver win. The verification
pass demoted it: RZOP finds quiet builds because its λ²/λ³ *generator* proposes
non-immediate-threat moves, **not** because the defender passed. The null move
only deletes a defender subtree; it cannot invent an attacker move the generator
never emits.

---

## 1. Thesis: the OR generator is the bottleneck

`threat_creating_moves` (`tss_solver.rs:1042`) walks every window; for a
claimant-active window it computes `strength = entry.count(claimant)` and then:

```rust
if strength < 3 { continue; }          // tss_solver.rs:1060-1063
```

So the only empties ever proposed are those of windows where the claimant
*already* holds ≥ 3 stones (one more placement → a count-4 threat). A window at
count 2, where **two** same-turn placements would build a fresh double-four
(the VCDT pair-build), is skipped. The function's own comment concedes the
boundary: *"a count-four extension is an immediate lambda-one proof after a
one-stone remainder."* It has no count-2 case. This matches the VCF-corpus
diagnosis on record: the generator is a VCDT-of-existing-threats generator, and
continuous connect-6 wins that build *through* count-2 windows are invisible to
it.

Three things this thesis is careful **not** to blame, because the code already
handles them:

- **The certificate chicken-and-egg is already resolved.** U1's monotone
  closure loop (the `arena_core` fixpoint, ~`tss_solver.rs:793`) unions proven
  child cores bottom-up and re-iterates — zones are built from proofs, not
  presupposed. The "static superset, then only check" story that motivated the
  null-move-first framing contradicts the code.
- **Core-first defender generation is already approximated.** `prove_universal`
  sorts hitting-first and materializes children lazily (~`:751`).
- **Horizon-relative zone reuse is already gated.** The two-stamp import rule +
  `rebase_zone_distances` (~`:906`, `:290`) already compose zone fragments
  across build horizons soundly.

"Optimal search for both players" therefore splits cleanly:

- **Attacker side** — widen the generator *just enough* to reach the forced win
  and no further. That "no further" is what λ-order deepening buys (§2).
- **Defender side** — the sound zones already proven (`Z_dir/Z_seed/Z_touch/
  Z_virgin`, T3) **plus** the one fan-out collapse RZOP has that we don't: the
  T2 macromove (§3).

A note that connects this to the minimality discussion that preceded this pass:
the proof delivers **soundness, not minimality** — several band widths are open
(uniform `8(B−1)`, full-union virgin radius) and most tightness pins are
relative. This document does not close that gap; §5's capstone *measures* how
close to minimal we search, and §2's generator work is about the *attacker*
analogue of minimality (smallest sufficient generator), which the proof never
addressed at all.

---

## 2. Engine change I — attacker width and λ-order completeness (Tier-A)

This is the highest-leverage work and the only change that can move the corpus
off 0/14. U1–U18 are **all** AND/defender-side; these are the plan's first
attacker-side items.

**2.1 Diagnose before committing the generator shape.** Instrument one 0/14 VCF
root and determine whether the blocking first move is a **count-2 same-turn
pair-build** (λ¹/VCDT — no seminull machinery needed) or a **quiet-prep
nonthreat turn** (λ²). The two need different generator regimes; "widen the
generator" is a targeted hypothesis until the blocking move-class is observed.
Ship nothing before this sub-step.

**2.2 The generator fix.** Add a threat-order parameter to
`threat_creating_moves` / `ordered_threat_creating_moves` (`:1042`/`:1101`):

- **order 1** = current behavior (count ≥ 3 extensions);
- **order 2** (VCDT-analogue) additionally emits count-2 window empties as
  `FirstStone` candidates whose same-turn partner reaches count ≥ 4, with the
  reply search kept **zone-restricted** so width does not explode;
- **order 3** = radius-8-bounded nonthreat setups.

Every addition is gated `[H]` — the independent verifier (`tss_verify.rs`)
re-derives every zone, so the generator can be as liberal as needed **without
touching soundness**. Default-off, shadow-rung, exactly as the tss_zone flags
shipped.

**2.3 λ-order iterative deepening.** Today the only search bound is
`semantic_horizon`, an absolute ply cap; there is no live pn/dn and no iterative
deepening (`PLAN` item 1). Make **order the outer loop and ply-horizon the
inner one**: at order *k* the OR generator admits threat-order ≤ *k*
continuations (§2.2) and a proof may use ≤ (*k*−1) non-forcing links *regardless
of depth*; escalate order only on UNKNOWN. RZOP's evidence for why this is the
right axis: λ² proofs of depth 25 with only 13 order-2 links. Ply depth is not
the complexity axis — non-forcing branching is, and order bounds it directly
while leaving forcing chains unbounded in depth. Standalone this lever does
nothing (without §2.2 there are no non-forcing moves to bound); it is the
*scheduling wrapper* for the widened generator.

**2.4 Null-defender discovery probe — accelerator, not the fix.** A
"defender passes" probe (never a real game move) can discover a seed zone and
seed move-ordering. It is **sound** because radius-8 legality is monotone in the
stone set: the pass-frontier is a subset of any real frontier, so a
pass-winning line replays legally. Use it *only* to amortize discovery — run the
widened §2.2 search once against a passing defender (no defender branching),
harvest touched cells as an `[H]` ordering seed and as Z₁ to bound the reply
search (RZOP T0-3). The idea and the chicken-and-egg framing are already in
`RZOP_COMPARISON.md` §6.2; the only new content here is the **monotonicity
soundness lemma** and the seeding link. Do **not** bill it as the 0/14 fix — it
removes defender branching, not attacker blindness.

---

## 3. Engine change II — defender dominance and macromove reduction

The mining conflated two distinct RZOP reductions; the correction matters.

**3.1 What RZS is *not*.** The Appendix subverifier (Property RZS, Lemma 12,
paper lines 1939–2071) verifies a critical-defense core padded with nulls,
dismisses stones **outside** the zone by zone-irrelevance, and recurses
**per-square** inside the zone. The outside-zone dismissal is exactly what our
T3/U3 zones already do — so RZS ordering alone buys little, and its
zone-irrelevance/left-shift soundness step **does not survive the radius-8
rule** (it relies on Connect6 having no locality). RZS is a citable *ancestor*,
not a drop-in.

**3.2 The genuine collapse — port the T2 macromove.** RZOP T2 (Fig. 15/16,
lines 1324–1330) merges *distinct minimal critical defenses that share one
identical winning continuation* into a single macromove under one shared zone.
Our `prove_universal` (~`:701`) builds a flat candidate union with no
family-level collapse. Porting this as a defender-node reduction (plus a
verifier arm in `tss_verify.rs`) is the real fan-out win. `RZOP_COMPARISON.md`
§6.4 under-weights it as a mere paper-remark near D17; it is a solver mechanism.

**3.3 Optional extension — in-zone filler class-partition.** Verify one
representative per frontier-equivalence class of in-zone fillers. This is a
*novel* generalization of our own P2 single-pair `{x,y}` collapse, not a port of
RZS.

**3.4 Frontier-inertness (DOMINATION.md Lemma 7) is both the class predicate and
the honesty guard.** In Hexo, distinct cells usually have distinct radius-8
balls, so a macromove/partition collapse is sound **only** where the absorbed
filler is frontier-inert (dead per Lemma 7, or interior and already 8-supported).
This both *enables* §3.2/§3.3 and *caps* them — which is why their honest impact
is medium/low, not high. Report the frontier-inert fraction per SPARE node as
telemetry.

**3.5 Proof debt.** Cite Wu & Lin Lemma 12 as the proven ancestor for U11's
`[UNPROVEN]` sub-hitting dispatch and reuse its Par-1..Par-6 case-split
*architecture*, but derive a **fresh** Hexo lemma under the radius-8
frontier-equivalence obligation and put it through hostile review first. Keep
the `[UNPROVEN]` label until then; RZS does **not** de-conjecture the hitting-set
algebra.

---

## 4. Zone reuse, caching, and incremental generation

Position: mostly already done, one constant-factor win, one forward flag.

- **4.1 Already built.** The U1 monotone closure loop (~`:793`) resolves the
  construct-vs-check inversion; the defensible increment is only *seeding* that
  loop from `hitting + Prot` rather than the wider count ≥ 1 A-touched term (an
  ordering tweak, not a redesign).
- **4.2 Constant-factor win.** `zone_initial_candidates` (`:1286`) rebuilds from
  scratch every node (full window scan, two hitting passes, sort/dedup/legality
  filter). Maintain generator state incrementally along the DFS path — on
  descending one placement touch only the ≤ 18 incident windows, restore on
  backtrack (the same ±18-window pattern U9 uses). **Node-throughput only** — not
  a state-space shrink and not a completeness fix, so it converts no current
  UNKNOWN to a proof. Caveat surfaced in verification: the generator uses
  *predicates*, not the verifier-side `r_N/E^D` clocks, so L11's shift licenses
  the verifier band, not this predicate recompute.
- **4.3 Horizon-sound promotion (already shipped; paper strength).** The
  two-stamp import gate (~`:906`) + `rebase_zone_distances` (~`:290`) is the
  minimal sound rule for composing horizon-relative zone fragments — a hazard
  *specific to our horizon-relative D*, absent from RZOP's stone-count zones
  (they sidestep it by truncating to Z∞). Frame as our hazard, not a bug in
  RZOP.
- **4.4 Forward flag.** Keep the leaf/zone schema **promotion-composable** so a
  future λ²/VCST solver can shift+merge a completed lower-order subproof's ranked
  zone as a fixed backbone rather than re-derive it. Cheap to flag now,
  expensive to migrate after the schema is frozen single-order.

---

## 5. Empirical capstone — measuring how close to minimal we search

This is the measurement deliverable that closes the loop with §0, and the
concrete form of the "how minimal are we, provably/empirically" question.

- **5.1 The corpus problem.** A continuous-double-four VCF corpus is
  *structurally dispatch-only*: forcing needs ≥ 2 simultaneous threats, so every
  forced VCF node has hitting number 2 = B = dispatch (`k == b`, ~`:709`) and
  **never** exercises the `k < B` `write_legal_moves` branch that U1/U2 zone
  generation targets. The existing corpus validates *none* of the paper's new
  machinery. **Fix:** curate 1–3 positions whose winning first move is
  nonthreat/spare-turn (the Hexo analogue of RZOP's harder λ orders),
  ground-truthed by matched-horizon agreement with `tss_reference.rs`.
- **5.2 The minimality headline.** Report per-AND-node and aggregate
  `|searched| / |Legal|` — RZOP's own proxy ("2656 recursive calls, small vs
  legal"). This is the literal "how close to minimal" number for the defender
  side.
- **5.3 Uniform vs exact.** Recompute each zone under the uniform `8(B−1)` band
  vs the exact per-window `E^D` clock and publish the size delta as the
  quantified minimality gain. RZOP's static zone index *structurally cannot*
  produce this comparison — the contrast is ours.
- **5.4 Acceptance witness.** Add a certificate-derived λ-order proxy column
  (count `k < B` Universal nodes + nonthreat OR edges; max spare-turn nesting =
  forcing depth) to prove the tested position actually needs the new machinery.
- **5.5 Robustness control.** Re-verify the capstone certificate under a
  radius-8 frontier pushed **outward** by a margin (more legal cells for both
  players) and assert the win still certifies — frontier-independence, the Hexo
  analogue of RZOP's infinite-board recheck, doubling as a live stress test of
  the Z4/virgin clauses that carry the paper's novelty. (Do **not** use the
  self-defeating "assert every certificate cell is strictly interior" phrasing —
  virgin windows live *at* the frontier by construction.)

Honest framing throughout: §5 is methodology. It **measures** the gains of
§§2–3; it does not create them.

---

## 6. Architecture — finder/verifier split and the PN loop

- **6.1 We already have the split RZOP is credited for.** `tss_solver`
  constructs certificates (finder); `tss_verify` independently re-checks
  (checker); a single mint via `hard_value_from_verified`.
  `RZOP_COMPARISON.md` §4 already frames the split as inherited — keep that
  honesty.
- **6.2 The asymmetry that lets us widen fearlessly.** Because every zone is
  re-derived by the verifier, we can trust an **uncertified** racer for
  scheduling/width-estimation only — it mints nothing — whereas RZOP consumes
  its fast NCTU6 solver's zones unverified. This is precisely what makes the
  Tier-A OR-widening (§2) safe: all `[H]` additions are verifier-gated.
- **6.3 Scheduling, not DFS ordering.** Fold the uncertified racer and any
  zone-cardinality proof-number into U8's width predictor (async cross-leaf
  scheduling), **not** into `prove_universal` child ordering. The solver is
  single-pass DFS AND/OR, not best-first; there is no pn frontier to order, and
  "expand smallest-zone-first" is backwards for fail-fast AND semantics (a small
  zone is the *easy* child). Drop the unverifiable "round-8 fix A" provenance.
- **6.4 Residual re-attack frontier.** On UNKNOWN, emit the blocking defender
  AND-node reply set (RZOP's reported λ-moves, lines 1683–1689) as a
  routing-only field on `SolveResponse` (`tss_async.rs`, ~`:230`) and re-solve
  only those at deeper caps next pass — turning a monolithic UNKNOWN into a
  targeted work-list. The current structure discards exactly this signal
  (child-poison ~`:789`, silent OR-exhaustion ~`:680`). **Mints nothing**
  (invariant intact); honest caveat: payoff is gated by §2 — reporting a reply
  helps only if the widened generator can then create the needed threat.

---

## 7. Crosswalk to the U1–U18 solver plan

The key structural finding: **U1–U18 are entirely AND/defender-side; the
highest-impact work here is OR/attacker-side and needs new Tier-A items.**

| # | This doc | Plan disposition |
|---|---|---|
| 1 | OR-generator width (§2.2) + λ-order deepening (§2.3) | **NEW Tier-A OR-side items** — the plan's first attacker-side entries; feed `ordered_threat_creating_moves`. Null-probe (§2.4) a sub-item gated on the monotonicity lemma. |
| 2 | T2 macromove + P2-generalizing partition (§3.2/3.3) | **EXTENDS U11** beyond pairwise P1/P2 (~`PLAN:533`). |
| 3 | RZS / Lemma 12 as template (§3.5) | Proof template for U11's `[UNPROVEN]` sub-hitting; **keep the label**, add a fresh-lemma sketch under radius-8. |
| 4 | Incremental generator state (§4.2) | **FOLD into U1** (~`PLAN:95`); shares the ±18-window pattern with U9 (~`PLAN:484`). |
| 5 | Uncertified racer + zone-cardinality PN (§6.3) | **MERGE into U8** width predictor; no new scheduling tier. |
| 6 | Residual re-attack frontier (§6.4) | **NEW Tier-B/C** enabler-grade item, gated by §2. |
| 7 | Composable zone schema (§4.4) | Deferred backlog flag near U11 / `PLAN` item 5. |
| 8 | Capstone measurement (§5) | `PLAN` item 4 SPARE_WIN/DEEP_SPARE bucket seeds + `sections/11`. |

Housekeeping: fix the stale `PLAN` item-1 `:630` reference for
`threat_creating_moves` (correct is `:1042`). Frontier-inertness telemetry →
`TSS_SOLVER_PROFILE` buckets (~`PLAN:574`).

---

## 8. Honest limits — what this document deliberately does not claim

- **8.1 The generator fix is diagnosis-gated.** Until one 0/14 root is
  instrumented we do not *know* the blocking move-class; "this moves the corpus
  off 0/14" is a targeted hypothesis, not a result.
- **8.2 The null probe is an accelerator only.** It removes defender branching,
  not attacker blindness. Multiple lenses initially over-claimed it as the fix;
  corrected.
- **8.3 Several ideas were already ours.** The chicken-and-egg and
  null-move-first are already in `RZOP_COMPARISON.md` §6.2; bottom-up zone
  construction is already built (U1 closure loop). The "static superset, then
  only check" story contradicts the code.
- **8.4 RZS soundness does not survive radius-8.** Its zone-irrelevance/left-
  shift steps rely on Connect6's lack of a locality rule. Template/ancestor,
  not a proven basis; U11 stays `[UNPROVEN]`.
- **8.5 Soundness, not minimality.** Many band widths are open; most tightness
  pins are relative. §5 *measures* how close to minimal we search; it does not
  prove minimality.
- **8.6 Dominance collapse is capped.** Distinct Hexo cells usually have distinct
  radius-8 balls, so the macromove/partition gain is bounded and localized —
  honestly medium/low.
- **8.7 §2.3 and §6.4 are contingent on §2.2.** λ-order deepening has nothing to
  bound and the residual frontier has no findable strategy to re-attack until
  the generator is widened.

---

## 9. Priority-ordered work list

Ranked by solver-optimization impact first (S/M/L = small/medium/large effort):

| Rank | Item | Impact | Effort | Depends on |
|---:|---|---|---|---|
| 1 | OR-generator width enrichment: admit count-2 VCDT pair-builds (§2.2) | **high** | M | — (internal diagnosis sub-step first) |
| 2 | λ-order iterative deepening as the outer loop (§2.3) | medium | M | 1 |
| 3 | T2 macromove defender-node collapse (§3.2) | medium | L | 7 |
| 4 | Null-defender discovery probe + monotonicity lemma (§2.4) | medium | M | 1 |
| 5 | Verifier residual re-attack frontier (§6.4) | medium | M | 1 |
| 6 | RZS/Lemma 12 as template for U11 sub-hitting (§3.5) | medium | L | 3 |
| 7 | Frontier-inertness as class predicate + honesty guard (§3.4) | low | S | — |
| 8 | Incremental zone-generator state along the DFS path (§4.2) | low | M | — |
| 9 | Fold racer + zone-cardinality PN into U8 (§6.3) | low | S | — |
| 10 | Keep zone/leaf schema promotion-composable (§4.4) | low | S | 1, 2 |
| 11 | Capstone corpus: curate spare-turn positions (§5.1) | low | M | — |
| 12 | Capstone measurement: searched/legal, clock delta, λ-order (§5.2–5.4) | low | S | 11 |
| 13 | Legality-frontier robustness control (§5.5) | low | S | 11 |
| 14 | Paper-framing bundle (§4.3, band-vs-Z∞, locality asymmetry) | low | S | — |

**Do items 1–2 first.** They are the only entries that can convert a current
UNKNOWN into a proof; everything below rank 5 is accelerator, telemetry, proof
debt, or paper framing that pays off *after* the generator is widened.

---

*Provenance: six-lens mining pass (`search-org`, `defender-reduction`,
`attacker-width`, `zone-algebra`, `architecture`, `empirical-capstone`), each
adversarially verified against source, then synthesized. 27 findings survived
verification; the OR-gate claim (`tss_solver.rs:1061`) was re-verified by hand.
Downgrades and drops are recorded honestly in §8.*
