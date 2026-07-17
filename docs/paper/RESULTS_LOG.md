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

---

## 2026-07-17 — R-LF1: lazy frontier admission BUILT and measured (−62.6…−68.4% TT admissions, −15.8% wall)

**Anchor:** branch `hunt/turn-quotient` (commit at gate),
PROOF_LAZY_FRONTIER.md + HUNT_REPORT_LAZY_FRONTIER.md + implementation
in `tss_solver.rs` (WidePnSearch) behind `TSS_LAZY_FRONTIER=1`
(default off). Regen: `lazy_frontier_equivalence_campaign`, flag-on
`tss_corpus_check`, flag-on `turn_quotient_campaign`.

**What the paper says (§6 engineering story — the NQ4 discovery,
converted):** unselected generated children are kept as key-bearing
edge thunks and acquire an arena/TT entry only on first selection. The
**Lazy-Frontier Refinement Lemma** (proved prose-level against the
code, with an exhaustive audit of every pre-selection read) shows eager
and lazy frontiers have identical reachable PN fixed points and
materializable certificates in the uncapped-index model; the cap-aware
corollary states exactly which counters may diverge after a TT-index
refusal (admission order, tt_hits, UNKNOWN timing) while proof validity
is invariant — no thunk, staged cutoff, or cap exit is ever promoted to
evidence.

**Measured (paired off/on, 59 roots + corpora):**
- indexed/retained entries: −62.6% (forcing 10k), −67.3% (forcing
  100k), −64.8% (human sample) — the NQ4 never-expanded class drops to
  exactly 0;
- peak TT bytes: −64.9% / −68.4% / −65.5%;
- wall: NQ4 campaign 106.5 s → 89.6 s (−15.8%);
- equivalence: exact verdicts, exact certificate bytes, exact
  expanded-node counts across all 59 roots; official all-19 corpus gate
  flag-on green (failures=0); narrow path untouched
  (double_fork_compact byte-identical).

**Solver consequence:** the same TT budget now holds ~3× the effective
proof mass before saturation — this composes with the T10-licensed DAG
sharing (G2R9) and directly benefits the 256 KiB leaf profile, where
admission pressure dominates (flag-on behavior under tiny TT caps is
validity-preserving but not trace-identical; Phase-3 rungs measure it).

**Caveats:** `peak_tt_bytes` excludes edge-owned future keys and the
deferred-key registry — R-LF1 is an admission/arena reduction, not
total frontier-memory elimination; flag stays default-off until a
consume round flips it with its own gate.

---

## 2026-07-17 — R-IG1: interior census gate BUILT and live-measured (79–93% node savings on horizon-bounded solves)

**Anchor:** branch `hunt/pn-init` (commit at gate), BUILD_INTERIOR_GATE.md
+ implementation in `tss_solver.rs` behind `TSS_INTERIOR_CENSUS_GATE=1`
(default off; read once per solve). Regen: `interior_gate_live_campaign`
off/on + flag-on `tss_corpus_check`.

**What the paper says (§6, paired with the NQ6 sizing):** the proven
Contract-8.1 census gate, integrated at interior claimant WIN arms only
(wide + narrow paths; universal defender arms untouched; wide
refutation and narrow `LOCAL_TT_FAILED` semantics preserved), delivers
LIVE at requested horizon 16:
- forcing 10k: 89,405 → 18,909 expansions (**−78.9%** live vs 82.6%
  trace), wall 7.81 s → 1.13 s;
- forcing 100k: 324,163 → 21,302 (**−93.4%** live, BETTER than the
  88.0% trace estimate — gating compounds: killed subtrees stop
  polluting the TT and PN frontier), wall 28.03 s → 1.25 s (**22×**);
- human roots: −41.2% live vs 53.1% trace (honestly below — the
  transposition caveat is real);
- compact witness: 0 dismissals (required negative control), zero cost.
Soundness: zero verdict differences on 139 keyed off/on rows; all 60
returned certificates verifier-accepted; flag-off identity vs the
frozen NQ6 baselines exact; official 2 GiB corpus gate green.

**The honest structural finding:** the official deep-solve profile runs
at `semantic_horizon = u32::MAX`, where `h_rem ≤ 8` never fires — the
gate is **inert on today's unbounded corpus profile** (gate_evals=0 on
every corpus row; the corpus run validates the refactor, not the gate).
The lever's payoff lives where horizons are finite: the Phase-3 256 KiB
leaf solver (h=8 is its native query) and any horizon-laddered scheme.
Since WIN-within-h transfers upward (NQ4's monotone rule), this poses
NQ8: iterative horizon-deepening for deep solves — find shallow WINs at
a fraction of the cost, escalate only genuinely deep rows.

**Caveats:** default-off until a consume round; live human-cohort
saving is materially below trace (41% vs 53%); no LOSS-arm gating
anywhere (out of proven scope).

---

## 2026-07-17 — NQ8: horizon-laddered deep solving REFUTED; bounded-horizon certificate-clock bug found

**Anchor:** branch `hunt/pn-init` (commit at gate),
HUNT_REPORT_HORIZON_LADDER.md + harness extension in
`tss_pn_init_hunt.rs`. Regen: `horizon_ladder_campaign` (+ focused
`TSS_HORIZON_LADDER_ONLY_GROUP=double_fork_compact` repro).

**What the paper says (§6 negative result + §12):**
1. *Ladder economics: DON'T BUILD.* Resolution-depth census over the
   13 measured WIN roots: 0 resolve within h=8, 3/13 within 16, 7/13
   within 24 — deep resolutions dominate, so shallow rungs mostly fail
   and their nodes are wasted. Every schedule loses: best (single h=16
   rung) +4.8–17.6% nodes vs direct; full ladder up to +148.6%. Even
   under deliberately perfect fragment-overlap assumptions the best
   schedule saves 0.1–0.4% — fragment persistence cannot rescue this
   orchestration. All 152 completed forcing comparisons were
   verdict-identical with verified certificates.
2. *STOP finding — a real pre-existing finder bug:* at bounded
   semantic horizon root+16, the solver returned a WIN certificate for
   double_fork_compact that the independent verifier REJECTED
   (`failure=clock, stored_d=8, derived_B=4` at a zoned Universal
   node). Cause: the materializer's `rebase_zone_distances` stamps zone
   clocks from the caller's external semantic horizon, while the
   verifier requires the exact local budget induced by the materialized
   proof DAG — extra horizon slack makes them disagree. The verifier's
   rejection is the single-mint architecture working as designed (no
   wrong verdict can escape), but the finder loses genuinely-won
   positions at finite horizons — exactly the Phase-3 leaf profile.
   Frozen repro committed; fix round follows.

**Solver consequence:** ladder struck from the portfolio; the interior
gate's deep-solve payoff now depends on the leaf profile only; the
clock-rebase fix is REQUIRED before any bounded-horizon deployment
(leaf solver) ships.

**Caveats:** human-cohort economics never ran (mandated STOP);
refutation is for THIS orchestration of the existing solver — a
verifier-closed bounded-WIN contract (post-fix) could reopen the
question, but the census says the ceiling is small regardless.

---

## 2026-07-17 — R-FIX1: bounded-horizon zone-clock defect FIXED (finder now stamps the verifier's exact D14 budgets)

**Anchor:** branch `hunt/pn-init` (commit at gate), FIX_ZONE_CLOCK.md +
`rebase_zone_distances` rewrite in `tss_solver.rs` + permanent
regression `bounded_horizon_compact_win_certificate_verifies`.

**What the paper says (§5 certified-engine story — a worked example of
the single-mint architecture):** the verifier requires every zoned
Universal node to carry the exact D14 local budget of the materialized
proof subtree (bottom-up: Win/OrCompletion=0, Loss=turn remainder,
Choice=pass-through, Universal=1+max over children) and the exact build
horizon. The finder's materializer instead stamped the defender-clock
count to the CALLER'S external deadline — an admissible bound for
searching, but not the evidence label the contract demands. With
horizon slack the two diverge and the verifier rejects a genuinely
winning certificate (the frozen compact h16 case: stored 8 vs derived
4). Two historical accidents hid it: unbounded solves never attach
zones (the 8-placement bail returns None), and G2R3's consume witness
used an EXACT deadline where the wrong formula coincidentally produced
the right numbers. The fix reconstructs the verifier's own recurrence
over the postorder certificate at final materialization and stamps both
fields from the assembled proof — search stays conservative (superset
searching is sound; only evidence labels changed), the verifier is
untouched.

**Evidence:** frozen repro now verifier-ACCEPTED WIN at h16 with
verdict equal to the direct solve; permanent regression test added; NQ8
forcing cohorts unchanged; R-IG1 live campaign numbers frozen-exact
both flag states; PN identity line unchanged; official 2 GiB gate
green; full release suite 98/0.

**Solver consequence:** unblocks every bounded-horizon deployment —
most importantly the Phase-3 256 KiB leaf solver — and makes imported
finite-horizon zone fragments (G2R9 territory) relabel correctly at
assembly. Recovered value today is small (one measured rung
UNKNOWN→verified WIN; NQ8's refutation stands) — the value is the
restored contract, not the recount.

**Caveats:** none beyond scope — the fix makes evidence labels
truthful; it cannot make an unwinnable rung winnable.

---

## 2026-07-17 — G2R9/G2R9b: T10-licensed shared-fragment store BUILT; warm reuse real (−20% expansions), TT-saturation hope NULL

**Anchor:** branch `hunt/turn-quotient` (commit at gate),
BUILD_SHARED_FRAGMENTS.md (design + amended contract) +
HUNT_REPORT_SHARED_FRAGMENTS.md (stop history + completion) +
implementation behind `TSS_SHARED_FRAGMENTS=1` (default off). Regen:
`shared_fragment_soundness_and_warm_campaign`,
`shared_fragment_reduced_tt_campaign`, official gates with
`TSS_CORPUS_EXPECT_*` assertions.

**What the paper says (§5 DAG figure + §6 + §12):**
1. *The store is sound and the contract is the interesting part.* A
   within-process cross-solve fragment store at T10's max-dominant
   merge semantics, verified fragments only, strict verifier still the
   single mint. Its first warm run tripped the campaign's strict
   verdict-identity requirement in the only direction it could: an
   UNKNOWN became a strict-verifier-accepted WIN (2 fragment imports,
   −62% expansions on that root). The session STOPPED rather than
   reinterpret; the orchestrator ruled the **monotone contract** (cold
   identity mandatory; warm changes only UNKNOWN → verifier-accepted
   hard verdict; NO rows never WIN) — UNKNOWN is a resource verdict,
   and verified proof mass at fixed budget is capacity, not risk.
   Under that contract: 139 roots × {eager, lazy-composed} all green,
   mutation control green, both official 2 GiB gates green.
2. *Warm reuse is real but narrow:* repeat-solves −20.5% expansions /
   −16.5% wall; cold overhead +0.8–1.0% wall; hit rate small (199
   hits / 424k lookups). Exactly one monotone improvement in the
   139-root corpus.
3. *The bottleneck hope is NULL at this scope, honestly:* the 0l row
   did NOT close at 512 MiB or 1 GiB with fragments (0 imports cold;
   13 hits / 0 imports progressive-warm through 1M nodes). Cross-solve
   fragments do not break the TT-saturation wall — the sub-proofs a
   deep solve needs at scale are not the ones earlier solves stored.
   The saturation attack therefore rests on lazy frontier (−67%
   admissions) and, prospectively, in-solve DAG sharing — not on
   cross-solve fragment reuse.

**Solver consequence:** deploy target is the Phase-3 leaf solver (many
solves, nearby positions, one process — where warm reuse is the common
case), not the deep corpus profile. Default stays off; the consume
decision moves to the Phase-3 integration round.

**Caveats:** warm numbers are same-root repeat-solve economics; the
20M-node reduced-TT attempt exceeded the time bound and is reported as
incomplete, not as UNKNOWN.

---

## 2026-07-17 — G2R7: K_reply kernel shadow-validated across 220,160 fires (zero counterexamples)

**Anchor:** branch `claude/tss-vcf-width` (commit at gate),
`tss_k_reply_shadow.rs` + cfg(test) shadow instrumentation in
`tss_solver.rs` (production signature unchanged — the extra parameter
is `#[cfg(test)]`-gated at the signature level) +
`.codex-group2/round7-progress.md`.

**What the paper says (§8 + the K_reply consumption story):** the Q8
K_reply kernel (proven, 5-clause contract, NQ2 salvage) was shadowed —
computed at every defender fallback fire and compared against the full
search, never influencing it — across the official 19 rows, the compact
witness, and 200 human roots:
- 220,160 fallback fires, 338 urgent nodes; **zero Q8 counterexamples,
  zero WIN/LOSS disagreements**;
- at urgent nodes the reply set collapses median **940 → 2**
  (p90 3,914 → 2);
- the single urgent fallback WIN edge observed anywhere in the study
  was inside K_reply (1/1) — the soundness-critical event class is
  rare, and the kernel contained it;
- the frozen NQ2 witness reproduces in-engine: urgent, kernel size 1,
  all 537 alternatives eliminated;
- telemetry on/off identity exact; round-5 identity PASS; consume
  witness WIN/409 verifier-accepted.

**Solver consequence:** consumption (G2R8) is now proof-backed AND
shadow-validated. Expected value is concentrated: urgent nodes are
0.07% of fires on the forcing corpus and 5.1% on human roots, so
global node savings will be modest on these corpora but the per-node
collapse is ~470×; the payoff concentrates where urgent SecondStone
defense dominates (deep refutation lines, leaf-style solves).

**Caveats:** shadow ≠ consume — verdict-identity + gate evidence for
the consuming engine is G2R8's burden; urgency rates are
corpus-dependent.

---

## 2026-07-17 — Lean spine: D19–D21 forced-hit layers + Q ≤ E + exact F+H_W maximum (PROVEN-LEAN)

**Anchor:** tss-lean commits `28f37c6` (S9S-15: full D19 gate layer),
`416bbb9` (S9S-16: D20 forcedRank + f ≤ r, D20a fhExposure recurrence,
D21 four FH zone components + mandatoryZoneFH, FH/FHD17 validators),
S9S-17 landing (commit at gate: `fhExposure ≤ defenderExposure`, exact
branch-coherent F+H_W maximum WITH attaining path, T6 structural
reflection + four-regime handoff dispatch). All audit-PASS, sorry-free.

**What the paper says (§4 headline list + §12 opens map):** the
forced-hit gate calculus is now kernel-checked through the validator
layer: gate validity with executable checkers and reflection, exact
escape deadline p(Q)+b+2, kernel nonemptiness from FG1 via minimum
transversals, compact gate-family bounds (one threat at b=1, ≤3 at
b=2), FH-debited ranks and exposure, and the four mandatory FH zone
components. Notably, **the branch-coherent F+H_W maximum — listed as an
open problem in the companion — is now PROVEN-LEAN with an attaining
path**; the §12 opens map must be updated when the flagship is written
(the fully general pathwise interpretation question remains open only
in its residual form). Claim C6 (ForcedHit/T6 statements) is one
increment from closure: T6's capstone `T6_extendableHitKernel` is the
remaining block.

**Caveats:** T6 increments 3–7 and L15–L17 outstanding; two of the
three spine sessions ended at RAM gates (31 GB host shared with engine
lanes), not at proof obstacles.

---

## 2026-07-17 — R-LF2: the memory wall MOVED — full gate green at 1 GiB; 8.4× at 512 MiB

**Anchor:** branch `hunt/turn-quotient` (commit at gate),
HUNT_REPORT_LAZY_MEMORY_WALL.md + per-run logs (`.codex-hunt/lf2-*`).
Pure measurement round — zero code changes (the official gate's
existing `TSS_CORPUS_ID` filter sufficed). Regen commands in the
report.

**What the paper says (§6 headline + the round-8b bottleneck
epilogue):** the round-8b root cause was TT saturation: the 0-loss row
closed only under the 2 GiB profile. With lazy frontier admission ON
(fragments off, isolating the variable):
- the **full 19-row official gate is green at 1 GiB** (491.3 s,
  row-by-row verdicts/expansions/TT-hits identical to 2 GiB lazy-on);
  lazy-off at 1 GiB is also green but takes 836.5 s — lazy nearly
  halves constrained wall;
- at **512 MiB** the historically hardest row (`0l4291i_live`) closes
  WIN in both modes at the 20M rung, but the difference is the
  campaign's biggest measured number: eager thrashes the capped TT to
  **18.13M expansions / 1,660 s** while lazy needs **1.91M / 198 s**
  (−89.4% expansions, 8.4× wall). Cap-pressure regime: a
  capacity/traversal result under the cap-aware corollary, not
  uncapped equivalence — all certificates verified, no verdict
  anomalies;
- at uncapped 2 GiB, lazy costs ~12% wall (492 vs 437 s baseline) —
  the lever pays exactly where memory pressure exists, and slightly
  taxes where it doesn't.

**Recommendation adopted into the program:** official deep-solve
profile drops to **1 GiB with TSS_LAZY_FRONTIER=1 asserted** (the
CORPUS_MODE echo + expectation assertions make the flag state
auditable); 512 MiB stays unofficial pending a full gate there.

**Caveats:** the 256 KiB trainer leaf does NOT currently benefit —
trainer solves use the narrow path and lazy affects wide PN only; a
dedicated leaf campaign (or wide-leaf enablement) is the open follow-up
for that surface.

---

## 2026-07-17 — DOM-B2/B2R2: b=2 domination experiment — zero reversals, but PARKED as computation-limited

**Anchor:** branch `hunt/domination`, commits `a3047de1` (round 1) +
`af6f777c` (adjudication + orchestrator ruling).
EXPERIMENT_DOMINATION_B2.md carries the repaired §7 protocol (review
repairs 2–6 quoted and folded) and both rounds' data; evidence JSONLs
committed alongside.

**What the paper says (§9 domination + §12 opens):** the b=1 dispatch
domination (L-DISPATCH-B1 + L-DRQ) stands proven and
consumption-eligible. The b=2 extension was attacked empirically under
the repaired protocol: 42,664 b=2 parents inventoried in the human
corpus; every completed exact comparison — 11/11 d=3 children, 9/12 F₃
hitter aggregates — came back Unknown with **zero reversals**, and Q0
admitted no b=2 attacker-Loss (3/16 rows qualified, 13 failed closed).
But the discriminating d=4 quiet/frontier class completed only 1/11
children with no directional comparison, even at 45-minute exact-search
budgets: the naive bounded-minimax reference is the bottleneck, not the
conjecture. **Ruling: parked OPEN-COMPUTATION-LIMITED** — favorable
evidence recorded, no proof round launched on unmeasured discrimination.
Reopening condition named: a certified-engine-based exact reference
(verified certificates as ground truth) that makes d≥4 F_d computable.
A methods note worth keeping: the reference solver was upgraded to
honest incompleteness semantics (budget exhaustion returns incomplete,
never a fabricated verdict, and poisons parent aggregation).

**Caveats:** none hidden — the parking is precisely because the crux
measurement doesn't exist yet.

---

## 2026-07-17 — G2R8: K_reply consumption BUILT sound; deep-profile economics NEGATIVE — flag stays off

**Anchor:** branch `claude/tss-vcf-width` (commit at gate),
BUILD_K_REPLY_CONSUME.md + flag-gated consumption in `tss_solver.rs`
(`TSS_K_REPLY_CONSUME=1`, default off) + identity campaigns in
`tss_k_reply_shadow.rs`.

**What the paper says (§8 + the honest-economics thread):** the proven
five-clause kernel was consumed exactly as licensed — the production
trigger checks the precise contract conditions, and outside them (or
flag-off) nothing changes. Soundness is complete: 200/200 human
verdicts identical, 12 completed forcing rows zero differences, compact
WIN with equal certificates, official 2 GiB gate green. But the
economics REFUSE on the deep profile: wall +4.8% (human) and +181%
(completed forcing) at IDENTICAL node counts — the trigger's full-legal
Win1/BlockAll evaluation at every fallback fire (8,858 fires for 432
urgent hits on forcing) costs far more than the rare 940→2 collapses
save. The compact witness gains −3.4%. The shadow's "huge local, modest
global" expectation did not survive contact with wall-clock.

**Decision:** flag stays default-off; NO deep-profile consumption. The
kernel remains a candidate for the leaf surface only (urgent
SecondStone density is corpus-dependent, and a cheaper precomputed
urgency trigger is the named engineering fix if leaf economics
justify it). This is the campaign's second honest null-conversion
(after cross-solve fragments): a theorem can be true, validated, and
still not worth its trigger cost on a given profile.

**Caveats:** 7/19 forcing identity rows incomplete (time-bounded, not
verdict differences); certificate comparison was structural equality,
not serialized bytes.

---

## 2026-07-17 — CONSOLIDATION: one engine, all levers, full battery green at the tip

**Anchor:** branch `claude/tss-vcf-width` (commit at gate),
MERGE_RESOLUTION.md (per-hunk conflict log + battery table with exact
commands).

**What the paper says (§5/§6 — the state of the engine):** the three
lever branches merged into the mainline with zero semantic escalations:
lazy frontier + shared fragments (wide PN), interior census gate +
zone-clock fix (narrow path + materializer), K_reply shadow + consume
(fallback), all telemetry systems, and the strengthened official gate
(mode echo + expectation assertions now covering all four flags).
Battery at the tip: default suite 104/0; legacy 2 GiB gate green; the
NEW recommended profile (1 GiB + lazy) green; fragments+lazy
composition green; every per-lever identity spot-check reproduced its
frozen baseline (including the fragment store's single monotone
improvement and the bounded-horizon zoned-WIN regression). The
always-on production delta of the entire program is exactly one
correctness fix (R-FIX1's verifier-exact zone budgets); every
performance lever is flag-gated with auditable state.

**Corpus-coverage note (owner question, answered honestly):** all
batteries gated on the official puzzle-derived forcing corpus + the
λ² compact witness; the spare-turn/λ² corpus proper
(`tss_spare_corpus_check`) had NOT been in any optimization battery —
it was run at the merged tip (flags-off regression + lazy-on) as part
of this gate and joins the standing battery definition.

**Caveats:** wall timings at the tip differ slightly from historical
runs (machine load); the leaf-surface campaign decides which flags the
Phase-3 leaf asserts.

---

## 2026-07-17 — LEAF-SURFACE CAMPAIGN: Phase-3 config decided — wide + lazy + interior gate DOUBLES leaf verdict rate

**Anchor:** branch `claude/tss-vcf-width` (commit at gate),
HUNT_REPORT_LEAF_SURFACE.md + `tss_leaf_surface_hunt.rs` +
LEAF_SURFACE_RAW.txt. Regen: `leaf_surface_campaign`.

**What the paper says (the Phase-3 deployment section / trainer note):**
six configurations measured on MCTS-realistic batch workloads (50
deterministic human-game segments, persistent solver per batch,
256 KiB TT, caps 500/2k/8k, horizons 8 and 16; 300 solves per cell):
- **Winner: config D — wide PN + TSS_LAZY_FRONTIER +
  TSS_INTERIOR_CENSUS_GATE.** At horizon 16 it roughly DOUBLES the
  narrow baseline's verdict rate (6.33% → 13.33% at cap 2,000) at
  −65.6% wall; at the native h=8 query the interior gate dismissed ALL
  692 evaluated interior nodes — D expanded ~1,852 nodes regardless of
  nominal cap, p90 wall −93.7%.
- The cap-efficiency headline: **D at cap 500 outperforms narrow at
  cap 8,000** (13.00% vs 7.00% verdict rate at h16) — 16× fewer nodes
  for nearly double the verdicts, the owner's verdict-rate-at-fixed-cap
  axis exactly.
- Fragments: 22/875 hits, zero additional verdicts at this profile —
  OFF. K_reply: not routed through wide PN (no-op there); its separate
  probe route improved 15% but remains ~85× slower at median — OFF.
- Soundness: 806/806 hard certificates verifier-accepted; zero
  WIN/LOSS contradictions between configs; monotonicity clean;
  persistent-reuse (the 13ms-cliff guard) PASS in every config.
- Recommended flags recorded verbatim in the report (vcf_pair_complete
  width, lazy=1, gate=1, fragments=0, k_reply=0, WIN goal, relative
  horizon 8, cap 500, 256 KiB).

**Caveats:** absolute verdict rates are workload-dependent (human-game
segments, not MCTS-visit-weighted positions); the horizon-16 arm
suggests the trainer may want a taller-horizon leaf query than today's
h=8 — a Phase-3 integration decision, flagged not assumed.

## 2026-07-17 — FINAL IDEATION GATE: lever exhaustion FAILS — 3 unposed levers + full wall-clock decomposition of the final engine

**Anchor:** branch `claude/tss-vcf-width`, IDEATION_FINAL.md (audited
HEAD 28eb5ac8; all numbers verified against the retained
codex-consolidate.log transcript). Doc-only round; no code changed.

**What the paper says (the "where the time goes" section + future-work
ledger):** with every landed lever on, the official 1 GiB lazy run
(34 solves, 495.94 s) decomposes as: attacker pair generation
**43.60%**, defender-pair enumeration **35.46%** (together 79.06%),
non-expansion residue (descent/TT reads/materialization) 15.06%, stage
refresh 4.75%, TT insertion 0.66%, strict verifier **≤0.07%**. TT
traffic is dead as a cost center (2.27% hit rate, insertion under 1%);
the engine's remaining cost is move *generation*, not table management
or verification — evidence the search itself is near the certificate
format's floor.

**The one measured untapped lever — cap-ladder repetition:** the
official harness re-solves each root fresh at every node-cap rung
(10k→100k→1M→20M). Exact transcript arithmetic: **30.67% of total
solve wall (151.99 s) and 31.02% of expansions (1,398,409) are
repeated lower-rung work**; the hardest row (0l4291i_live) alone
repeats 1,109,997 expansions / 121.86 s before its closing rung
begins. A resumable proof-number session across a monotone cap ladder
(same root, horizon, flags; only the ceiling rises) exposes this;
soundness scope: lower-cap results stay Unknown, hard results still
cross the unchanged strict verifier.

**Other ranked candidates:** (2) prior-scale-aware df-pn threshold
increments — priors span 1..37 but thresholds still advance by +1, the
exact mismatch the proof-number literature names (Kishimoto et al.
survey; 1+ε trick), counter-first A/B specified; (3) opening-root D6
stabilizer orbit pruning — nontrivial stabilizers cover 62.92% of the
human corpus (root-child-removal ceiling 32.39%), distinct from the
dead mid-search TT folding (NQ5), Choice-roots-only soundness scope.
Plus two pre-existing closure debts the register never disposed:
dynamic child reveal (eager pair classification untouched by lazy
frontier) and the live_ge3 PN-seed live A/B.

**Reference capstone honesty note:** the historical idtt/dfpn/pdspn
columns are NOT matched-budget experiments (no pinned binary, host, or
leaf-work accounting). The only material apparent deficit is 94gnnol
(pdspn cited NO in 21 s vs our UNKNOWN after the 1M rung) — a matched
differential on that row is the required experiment before any
reference gap is claimed either way.

**Verdict recorded:** "nothing realistic remains" would be false today;
exhaustion can only be declared after the five lever rounds fail their
≥5% gates and the 94gnnol differential is dispositioned.

## 2026-07-17 — R-CR1: cap-ladder resume BUILT — official profile −29.10% wall (ideation candidate 1 consumed)

**Anchor:** branch `claude/tss-vcf-width` @ e05324f4,
HUNT_REPORT_CAP_RESUME.md + `tss_cap_resume.rs` + retained
CAP_RESUME_*.log raws. Regen: `tss_cap_resume_campaign` +
`tss_corpus_check` with `TSS_CAP_RESUME=1`.

**What the paper says (the harness/call-surface section):** the
official ladder's fresh-per-rung methodology was itself a measured
30.67% tax (ideation §1.2). A cfg(test) resumable session — one root's
arena, position index, deferred keys, pn/dn, commitment, and
staged-depth cursor surviving monotone cap increases — recovers nearly
all of it: **full official 1 GiB lazy+gate wall 495.94s → 351.62s
(−29.10%)**, expansions −34.83%, peak bytes −8.95%, 15 re-entries.
The hardest row's ladder drops ~319s → ~194s (−39%): its closing rung
now continues from the 1M-rung frontier (1,713,725 cumulative
expansions) instead of rebuilding it (1,879,611 fresh).

**Soundness story (paper-grade):** the session binding pins the exact
root, goal, horizon, width, flags, TT cap, and hash mode; only
monotonically larger caps are accepted; lower-cap results remain
Unknown; hard results traverse the ordinary materializer and the
UNCHANGED strict verifier. 27/27 measured identity triples (pn, dn,
status) matched fresh solves; certificate bytes differed only at
pause-boundary legal ties and every variant strict-verified.
Production builds contain none of the machinery (non-test release
build green; default-off subset byte-identical).

**Honest scope notes:** the win is a property of the *cap-ladder call
surface*, not the single-solve engine — single-rung solves are
unchanged; the resumed total landing 5.52% below even the
final-attempt-only baseline is pause-order scheduling luck, not
guaranteed; 94gnnol's hypothetical 20M rung was not measured (45-min
command rule; its official 1M rung matches exactly). Recommended
deployment: harness-only; an exposed in-process session handle for
repeated exact trainer queries is a separate API/soundness proposal.

**Verified:** orchestrator cfg-audit + digit-exact raw-log truth-check
+ independent single-row rerun reproducing WIN/expansions/peak bytes
identically (CORPUS_DONE failures=0).
