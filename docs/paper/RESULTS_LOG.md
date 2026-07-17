# RESULTS LOG — per-result paper-ready notes (companion to SKELETON_SEARCHING_LESS.md)

> Standing obligation (owner ruling 2026-07-16 ~23:00): every closed line —
> proof, refutation, or consumed lever with its measured delta — gets its
> paper note HERE as it lands, not reconstructed at writing time. Each entry:
> what the paper says, which section it feeds, the trust anchor
> (commit + Lean decl / log + command), and the honest caveats. Entries are
> append-only; corrections append, never silently rewrite.

---

## 2026-07-16 — T10: DAG-unfolding soundness (PROVEN-LEAN)

**Anchor:** `TssZones/DAGUnfoldingSoundness.lean`, tss-lean commit `69adffc`
(audit PASS ×2). Decls: `T10_baseDAGSoundness`, `T10_d17DAGSoundness`,
`T10_baseDAG_soundDismissal`, `T10_d17DAG_soundDismissal`, conjunction `T10`.

**What the paper says (§4 headline list, §5 certificate DAG figure, §6 U18
lever):** a DAG-shaped certificate — one stored copy of each shared
sub-proof, referenced from many parents — unfolds to a valid tree
certificate over the ORIGINAL DAG's compilers and horizon, in both the base
and D17 regimes, with exact dismissal corollaries. The load-bearing subtlety
is the merge rule: labels at shared nodes must be **max-dominant bounds**
over all path copies, not per-copy equalities. That rule is exactly the
U18/U22 shared-TT merge semantics; it is now kernel-checked, so the worst
silent failure mode of fragment sharing (unsound merges minting wrong
verdicts) is formally excluded before the engineering lands.

**Solver consequence (prospective, to be measured at G2R9):** attacks the
measured #1 deep-solve bottleneck — TT saturation (round-8b: 512 MiB ≈ 1M
nodes; the 0-loss row closed only under the 2 GiB profile). Also licenses
cross-leaf fragment reuse inside the 256 KiB trainer leaf budget.

**Caveats:** T10 is a license, not a speedup; no measured delta exists until
the U18/U22 build round. Claim map: C5 flips IN-FLIGHT → PROVEN-LEAN.

---

## 2026-07-16 — NQ2: quiet-join locality REFUTED; K_reply kernel salvaged

**Anchor:** branch `hunt/quiet-locality`, commit `016577bb` (proof doc
PROOF_QUIET_LOCALITY.md Q0–Q9 + hostile review sign-off; refutation landed
`833020ed`). Frozen witness pinned by FROZEN_ID in
`tss_quiet_locality_hunt.rs`.

**What the paper says (§8 + §9 "genuinely new facts" + §12 opens):**
1. *Negative fact with a sharp witness:* the conjecture "quiet defender
   replies near the join of two threats suffice" is FALSE. Frozen
   counterexample: a position whose unique winning quiet placement sits at
   stone-distance 6 — outside every proposed join-locality tier — among 538
   legal quiet moves (537 are hard losses). Complete quiet enumeration
   validated the uniqueness. The previously believed "join-adjacency law"
   was a 7-specimen artifact; the paper reports it as a cautionary example
   of measurement-derived laws (feeds the §1 trust story).
2. *Salvage, proof-backed:* at urgent SecondStone nodes the defender's
   reply set collapses 538 → 1 via the **K_reply kernel** (exact 5-clause
   contract: SecondStone{first} only; post-first recomputation; full-legal
   Win1; BlockAll = intersection over all defender count-4/5 windows;
   full-Legal fallback when no threat). Proven + hostile-review confirmed.

**Solver consequence:** K_reply is a direct branching-factor collapse at the
most expensive defender node class; consume round (G2R8) measures the
delta. The refutation KILLED a planned 15× quiet-universe shrink that would
have been unsound — count it as the proof-first bar paying rent.

**Caveats:** kernel applies only under its 5-clause trigger; no claim about
FirstStone or non-urgent nodes.

---

## 2026-07-16 — DTW census two-gap bound (PROVEN-DOC, production spec)

**Anchor:** branch `hunt/dtw-bounds`, commit `273b89f8`
(PROOF_DTW_CENSUS_BOUND.md + review; landed `bbfb34d0`). Regression:
`dtw_hostile_ply_boundaries`.

**What the paper says (§6 leaf-gate lever + §9 candidate fact):** for the
WIN goal, a distance-to-win ≥ census bound holds with gap constant c ≤ 2 in
BOTH phases at horizon 8 — an O(1) precheck that dismisses h=8 leaf solves
whose census already proves the horizon insufficient. Measured skip rate at
the leaf profile: ~49% of h=8 solves. Two traps the proof exposed, which
the paper reports as sharp negatives: (a) the naive SecondStone c=3
increment is FALSE (counterexample frozen); (b) a threat-index census
falsely gates — the sound production spec REQUIRES a full
`WindowStore::entries()` scan (Contract 8.1/8.2, 7 repairs applied).

**Solver consequence:** deploys at Phase-3 as the leaf gate; no production
delta yet. Exactness of the bound was claimed and RETRACTED during review —
the paper only states the ≥ bound.

---

## 2026-07-16 — Dispatch domination B1 + L-DRQ (PROVEN-DOC, consumption-eligible)

**Anchor:** branch `hunt/domination`, commit `17a6c6de`
(PROOF_DISPATCH_DOMINATION_ROUND1.md, hostile round 1 confirmed; provenance
banner pins base `6b853c0e` vs reviewed `7e240388`).

**What the paper says (§9 domination patterns + §5/§6 consumed-lever slot
once wired):** at budget b=1, dispatch-arm consumption is sound at every
horizon with every-legal-reply from horizon 2 (L-DISPATCH-B1), plus the
L-DRQ quiet-reply lemma. A consumer MUST branch on the computed mhs/budget
pair (the b=1 fence is part of the statement, not an implementation hint).

**Solver consequence:** Phase-3 dispatch-arm wiring; b=2 extension is an
OPEN experiment (review repairs 2–6 bind its protocol; d=1,2 are
non-discriminating — covered successors are analytically Unknown through
d=2).

---

## 2026-07-16 — C1: engine unification (MEASURED/AUDITED)

**Anchor:** branch `claude/tss-vcf-width`, commit `3c180c66` (G2R6; identity
harness landed `ace1f5b2`).

**What the paper says (§5 certified-engine story):** the narrow DFS prover
was deleted; `TssSolver::default()` now hosts the narrow contract inside
the wide engine (`WidePnSearch::prove_narrow_compat`) with **byte-identical**
certificates proven by an identity harness before the flip (nodes, cert
bytes, TT behavior, golden digests). One engine means every theory lever
lands once, in one place — this is the architectural premise of §6.

**Caveats:** post-deletion identity evidence = identity run at post-flip
source + behavior-neutral deletion diff (self-comparison caveat recorded in
the commit).

---

## 2026-07-16 — NQ3: certificate support locality — REFUTED as posed; C_rel redesign conjecture

**Anchor:** branch `hunt/cert-support` (base `2430fc47`),
HUNT_REPORT_CERT_SUPPORT.md + `cert_support_hunt.rs` (commit at gate).
Regen: `cert_support_campaign`, `cert_support_far_threat_adversarial`,
`strict_root_binding_is_a_global_obligation` (commands in the report §
Reproduction).

**What the paper says (§6 TT-policy discussion + §12 opens):**
1. *Negative, structural:* today's strict certificates have NO bounded read
   support — by design. The verifier's first check binds the COMPLETE root
   occupancy (`RootBinding` equality, which also asserts absence of stones
   everywhere else) and shared-DAG `ReplayKey`s bind complete positions.
   Unchanged-certificate transfer to out-of-body mutated roots: **0/180 at
   every K ∈ {1,2,4,8}**; support-hashing under the current contract yields
   exactly today's equivalence classes (multiplier 1.000×). "Support-keyed
   TT" is NOT a free lever on the current format.
2. *Positive, measured:* the PROOF BODIES are compact and transferable in
   principle — body footprint median/p90/max = 22/42/53 cells among 34
   human-corpus WIN certs (vs root populations 31/81/149) and 38/54/68 on
   the solved official rows; after a shadow rebind (replace root binding,
   translate absolute clocks) the UNCHANGED proof body verifies at
   **93.9% / 96.1% / 83.3% / 77.8%** for K=1/2/4/8.
3. *Soundness spot-check passed:* an adversarial remote defender count-5
   formation was rejected even after the shadow rebind — the global checks
   (live-threat analysis, universal reply exactness, zone exposure) do
   their job; no soundness finding.
4. *The sharpened open (C_rel):* a relative certificate — proved support
   projection instead of full binding, root-relative clocks, recorded
   support for legality/universal/terminal/threat/WF/zone/commutation/D6
   obligations — transfers to any agreeing root. Eight named proof
   obstacles enumerated in the report §"Sharpened NQ3 conjecture".

**Solver consequence:** re-ranks the efficiency portfolio — cross-position
fragment reuse is gated on a certificate-format redesign (major), not a
keying change (minor); same-position DAG sharing (T10/U18) is unaffected
and stays first.

**Caveats:** shadow-rebind numbers are engineering evidence about a
NON-EXISTENT format, not behavior of any shipped certificate; 2/14 official
WIN rows + double_fork_compact unsolved at the hunt's plain
`vcf_pair_complete` 100k/64MiB profile (double_fork_compact needs the
zone-enabled Group-2 configuration — expected, not a regression).

---

## 2026-07-16 — NQ4: search-space quotients sized; lazy-frontier class discovered

**Anchor:** branch `hunt/turn-quotient` (base `2430fc47`),
HUNT_REPORT_TURN_QUOTIENT.md + `tss_turn_quotient_hunt.rs` (commit at
gate). Regen: `turn_quotient_campaign` (single gated run, PASS,
anomalies=0; telemetry-on/off node identity 2,412/2,263 on `0hz3hty`).

**What the paper says (§6 engineering story + §12 opens):** four levers
sized on 19 forcing roots (10k/100k), double_fork_compact, and 100
human-corpus roots:
1. *D6 at the search TT: exactly ZERO duplicates* on every cohort, at
   6.5–29.3 µs/key canonicalization cost — a clean negative. Within-root
   search from an asymmetric root essentially never revisits D6 images.
   (Cert-layer D6 transport is unaffected.) Kills the "fold the search TT
   under D6" idea before it wasted a build.
2. *Horizon quotient: the engine already implements the strong form* —
   one position-keyed entry reopened in place avoids 62–81% of the entries
   a naive clock-keyed TT would retain. Residual monotone-transfer gain:
   0.26–1.08 pp, relevant only to a future PERSISTENT proof cache (feeds
   the U18 design). The report fixes the exact sound-transfer rules
   (WIN upward; complete restricted refutation downward; UNKNOWN/caps/
   staged DepthCutoff transfer NOWHERE — dn=0 alone is not a disproof).
3. *Consecutive-turn commutation: ≤0.16% broadly* (5.4% on one compact
   witness) and adversarially unsafe without a strategy-preserving
   diamond theorem (quantifier order ∃attacker/∀defender blocks naive
   canonical ordering). Parked with the exact proof obligation stated.
4. *Discovered class — eager frontier admission (the real lever):*
   **62.6–67.3% of retained wide-TT/arena entries are never expanded**
   before proof or cap. Keeping unselected children as edge thunks
   (realized on first selection) is a frontier representation quotient:
   large reduction in key construction, hash insertions, retained arena
   records, and TT pressure — on top of the same position quotient.
   Required theorem: **Lazy-Frontier Refinement Lemma** (thunks preserve
   PN/DN values, selection order, and transposition linking on
   realization; eager and lazy frontiers reach the same PN fixed points
   and certificates), with a cap-aware corollary for admission-timing
   effects.

**Solver consequence:** build order re-ranked — lazy frontier admission
is the top unbuilt lever; it composes with T10/U18 DAG sharing (both
attack TT pressure, multiplicatively).

**Caveats:** the 62–67% figure is retained-entry share, NOT a promised
node reduction; commutation numbers are upper bounds on removable
interiors, not achieved dedup.

---

## 2026-07-17 — NQ6: interior census gating sized (53–88% trace-subtree coverage); stronger bounds and PN seeding triaged

**Anchor:** branch `hunt/pn-init` (base `2430fc47`),
HUNT_REPORT_PN_INIT.md + `tss_pn_init_hunt.rs` (commit at gate). Regen:
`pn_init_campaign` (single gated run PASS; telemetry on/off identity
`0hz3hty` 9,302 nodes / 2,872 hits / 9,301 expansions both ways).

**What the paper says (§6 leaf-gate lever, promoted to interior + §12):**
1. *The proven Contract-8.1 census gate is far bigger inside the tree
   than at leaves.* Applied at every claimant WIN-arm node with
   remaining horizon ≤ 8 (broad solves at requested relative horizon
   16): gates 80–96% of eligible nodes and covers **82.6% / 88.0% /
   53.1%** of ALL attempt expansions (forcing 10k / forcing 100k /
   human roots) via first-ancestry subtrees, at 0.5–1.3 µs per full
   `WindowStore::entries()` scan. The effect is NOT just the h=8
   predicate: the atomic-turn frontier visits h ∈ {0,4,8,12,16}, and at
   h ∈ {0,1,4,5} the exact phase formula also gates c=3 nodes.
   Soundness cross-check: **zero gated positives** over all solved-root
   gate events; every certificate re-verified. double_fork_compact is
   the honest negative control (0 of 221 eligible evaluations gate).
2. *Stronger census bounds are empirically dominated:* the only
   collision-free stronger target, FirstStone (h=9, c≤2), would add
   0.03–0.10%; every c≥3 or h≥12 screen collides with verified positive
   nodes (23–136 collisions) — the counterfactual harness itself
   demonstrates why those theorems are false or worthless. No broad
   proof round scheduled: measured triage, not taste.
3. *PN/DN census seeding is not build-ready:* count≥3 seeding improves
   an outcome-labelled solved-root replay 37–68%, but population
   Spearman rho is slightly NEGATIVE on every cohort — the replay/live
   divergence is named and the only admissible next step is a test-only
   live A/B, not a production seed.

**Solver consequence:** the interior WIN-arm gate is now the largest
proven-and-unbuilt lever; build round follows immediately (WIN-only
integration, live identity + soundness campaign, all-19 gate). Composes
with lazy frontier (fewer entries) since gating kills whole subtrees
before they generate frontiers.

**Caveats:** all percentages are deterministic trace counterfactuals —
transpositions reachable via other parents and changed PN values mean
live savings must be measured, not assumed; SecondStone coverage was too
thin to recommend any SecondStone strengthening (the frozen ply-5 c=3
counterexample stands).
