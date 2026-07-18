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

## 2026-07-17 — R-CD1: both ideation closure debts disposed as measured NULLs (dynamic child reveal; live_ge3 seed)

**Anchor:** branch `claude/tss-vcf-width` @ b7e9f36c,
HUNT_REPORT_CLOSURE_DEBTS.md + CLOSURE_*.log raws (incl. the
orchestrator-completed CLOSURE_COUNTER_FULL_OFF_RAW.log).

**Debt A — dynamic child reveal (the 43.6%-of-wall pair-generation
attack): NULL, and the number is a paper-grade negative.** The official
1 GiB counter profile measured 1.80B pairs evaluated vs 3.81M expanded —
99.79% of classification work never feeds an expanded child — yet the
*sound* avoidable fraction is only **7.83%** of pair-generation wall
(23.4s of 299.3s), below the 11.5% needed to clear 5% end-to-end. Why
the gap: soundness forbids refuting a Choice node before generator
exhaustion (so refuted/unknown nodes save nothing), and at proven nodes
the winning child sits deep in rank order (bins
[10785, 5686, 11250, 5666, 21508, 14981, 5343, 2460]), so the reveal
prefix covers most of the cost anyway. The "just generate lazily"
intuition dies on the engine's own soundness contract — this closes the
register's oldest optimization debt (the old opt-spec's "lazy ordering"
row) with a measured ceiling.

**Debt B — live_ge3 proof-number seed: NULL by leaf regression.** The
live A/B on the selected Phase-3 cells regressed wall +18.41% (h8) and
+12.49% with one lost verdict (h16); the promotion conjunction failed
outright, honoring NQ6's warning that its replay numbers froze
outcomes. All 109 hard certificates strict-verified; zero
contradictions.

**Method note:** instrumentation fully cfg(test)/default-off; unused
paths reproduced R-CR1 counts exactly; strict verifier untouched. Debt
A's official run completed by the orchestrator post-session with a
documented RAM judgment (standby-cache deficit, not true scarcity).

## 2026-07-17 — C-REL round 1: support-only verification REFUTED by design; salvage = strict-discharge warm-template cache

**Anchor:** branch `hunt/cert-support` @ 408dc5b6, DESIGN_C_REL.md
(1,173 lines, ultra design round). Design-only; no experiments run.

**What the paper says (the certificate-format chapter's closing
argument):** the natural successor to NQ3's refutation — a relative
certificate whose finite local interface, when cheaply matched at a new
position, licenses the verdict — is REJECTED at the design level, not
merely unmeasured. The NQ2 required-remote witness supplies the direct
counterexample shape: a disjoint remote count-five outside any local
interface flips the verdict while the interface matches. Attacker
locality is the wrong soundness boundary for this game; this elevates
the NQ2 witness from sharp example to a design-level impossibility
argument against support-hashed certificate reuse.

**The surviving project (bounded, killable):** rootless certificate
bodies with the interface demoted to a ROUTING HEURISTIC — a warm hit
only ever proposes a candidate strict certificate that the unchanged
strict verifier must replay in full (monotone contract). Soundness
never depends on the interface; economics is the only open question,
and it faces a named squeeze: weak interfaces pay a strict-rejection
tax, strong interfaces approach full-position binding and erase
transfer. 13 obligations dispositioned (6 proof-sketched / 2
refutation-risk / 5 dissolved, none hidden); staged experiment ladder
with pre-registered kill criteria (routing-selectivity experiment;
matched ≥5% gate).

## 2026-07-17 — R-TS1: prior-scale-aware df-pn thresholds CLOSED NULL (+1 increment proven optimal)

**Anchor:** branch `claude/tss-vcf-width` @ d5a2b5fd,
HUNT_REPORT_THRESHOLD_SCALE.md + ten retained raws.

**What was tested:** the last unposed algorithmic lever from the final
ideation gate (candidate 2). The engine initializes proof numbers from
fork degree (1..37) yet advances descent thresholds by +1 — exactly the
mismatch the 1+ε literature names as a re-traversal pathology.

**Phase 1 (counters, official 1 GiB lazy+gate):** all descent/state
time outside expansion = 13.93% of the 495.94 s wall; the
revisit-attributed share is 7.01% (34.79 s). A ceiling argument opened
Phase 2 honestly (avoiding ~71% of revisit traversal would clear the 5%
bar).

**Phase 2 (A/B, one global TSS_THRESHOLD_DELTA):** delta 2 on the
official deep profile: **927.59 s vs 499.85 s baseline — +85.6%**. Root
cause: coarser thresholds trade cheap revisits for catastrophic
frontier misallocation — the hardest row (0l4291i_live) expanded 3.2×
the nodes (6.05M vs 1.88M), saturated the full 1 GiB TT (vs ~549 MiB),
and tripled its row wall. Delta 4 and mean-sibling-prior each LOSE one
hard verdict on the Phase-3 h16 leaf matrix (39→38) and were correctly
disqualified before their deep runs; the ε ladder was never opened. All
228 leaf hard results strict-verified; zero contradictions; flag-off
byte identity and production build green.

**What the paper says:** in a TT-backed iteratively-deepening df-pn,
the +1 threshold increment is not a naive default but the measured
optimum: revisit cost (≤7.01% of wall) is an order of magnitude cheaper
than the best-first arbitration error any coarser schedule introduces
(+85.6% measured at delta 2). The 1+ε recommendation does not transfer
to this engine class. Re-arm only on a format change that makes
per-node state reconstruction expensive — not on bottleneck movement.

**Provenance note:** both Codex sessions died on API capacity outages;
measurement was complete, and the orchestrator authored the report with
every number verified against the retained raw logs.

## 2026-07-17 — R-KT1: forced-reply kernel taxonomy MEASURED NULL — T6 already owns the seam

**Anchor:** branch `hunt/kernel-taxonomy` @ 562b5e51,
TAXONOMY_KERNEL_REPLY.md + HUNT_REPORT_KERNEL_TAXONOMY.md + six raws.
Codex ultra round; 94gnnol excluded per owner ruling.

**What was tested:** whether the proven urgent-SecondStone reply kernel
(538→1) generalizes to a taxonomy of defender-context classes, each a
certified AND-width cut. Shadow classifier (cfg(test), default-off)
traversed the final wide PN DAG post-solve on the 18-position official
1 GiB lazy+gate profile: 2,910,351 classified forced AND contexts,
**zero load-bearing counterexamples, zero traversal errors** (93.3%
of contexts honestly reported semantically inconclusive — shadow
evidence, not proof).

**The structural finding (corrects the agenda's premise):** there is no
≤37→≤k width cut left to take, because **exact T6 consumption already
compressed the live seam**: observed forced reply-cell widths are 2–4
(full legal widths 333–956 are irrelevant post-T6), and defender turns
are already atomic unordered DefenderPair children. The apparent 52.1%
reply-cell collapse of the F2_COVER projections retains ALL 9,740,262
atomic pair children — zero AND-child value. The one genuine new
complete-reply proposal (S1_DEAD_SPOKE_C4, exact 2→1 with a cheap
incremental trigger) fired zero times: complete-turn pair compression
bypasses intermediate SecondStone AND nodes entirely. D19–D21/adaptive-
escape classes are NO-CONJECTURE — their membership requires
certificate metadata that does not exist in position state.

**What the paper says:** the defender-width story is CLOSED, positively:
the zone-theorem substrate (T6/Q8/P2/P3) already extracts the full
width reduction available at the wide-PN seam, and a 2.9M-context
corpus-wide shadow audit found zero violations of its reply machinery.
The remaining defender-enumeration cost (~33% of wall) is per-node
enumeration WORK, not branching width — the correct attack is
incremental enumeration, not new kernels. Lean proof queue from this
round: empty.

**Integrity:** flag-off/on fast-subset identity row-exact; production
release build with the test-only module absent passes; strict verifier
untouched; RAM gates recorded per invocation (relaxed 07-17 protocol).

## 2026-07-17 — R-IE1: incremental defender enumeration — first live lever since cap-resume (−7.6%), held for a memory round

**Anchor:** branch `hunt/incr-enum` @ 2f46925e, HUNT_REPORT_INCR_ENUM.md
+ sixteen retained raws. Follow-up to R-KT1's work-not-width redirect.

**Phase 1 (counters):** the defender planner's input state is almost
perfectly carryable — 99.9904% of 2,910,351 calls have an exact parent
fingerprint, with ZERO parent mismatches and ZERO residual mismatches
across 11,002,776 bounded one-stone local patches. Conservative
carryable time 34.49 s = 9.33% of full wall (24.3% of the defender
bucket). Cost decomposition also priced the non-carryable core
(canonical frame 30%, final-key construction 25%, fork-prior scan 11%
of planner time) — the input data for the pair-classification
micro-round.

**Phase 2 (implementation):** three-mode env (off / shadow / consume);
shadow mode rebuilds the batch plan field-for-field and aborts on first
mismatch — corpus-wide green. Consuming A/B: official 18-position wall
**353.99 s → 326.94 s (−7.64%)** with all 31 rows IDENTICAL on status,
nodes, expansions, TT entries/hits/peak, stage refreshes, and gate/seed
counters (compute-only lever, exact search identity). Phase-3 leaf
cells: h8 −7.34%, h16 −4.88%, no regression. Production release build
contains none of it; strict verifier untouched.

**The honest hold:** inline fixed-capacity parent snapshots add
353.39 MiB of accounted peak heap payload on the deep profile
(TT peak unchanged). PROMOTE bars are met, but production wiring waits
on a compact-sidecar / selected-edge-reconstruction memory round
(R-IE2). Snapshot capacity bounds (8 windows / 8 kernel cells) exceed
all observed maxima (6/4); out-of-bound shapes fall back to exact
batch.

**What the paper says:** post-T6 defender enumeration is not
inherently per-node work — its input state is exactly maintainable
across make/unmake with zero observed divergence at corpus scale, and
consuming it buys ~7.6% of wall with bit-identical search. The residual
defender cost is now dominated by canonical-frame and key-construction
work, which the census/micro-optimization lines attack separately.

## 2026-07-17 — R-CF1: census-deep round — inertness solved exactly; the census idea survives as a DEADLINE family (4 shadow-clean conjectures)

**Anchor:** branch `hunt/census-deep` @ a70e4b37,
HUNT_REPORT_CENSUS_DEEP.md + CENSUS_CANDIDATES.md + three raws +
SHA-256 manifest. Codex ultra round.

**The diagnosis (definitive):** the landed interior census gate is
inert at the official profile for one exact reason — the profile runs
`semantic_horizon = u32::MAX` while the proven single-window DTW family
tops out at a finite lower bound of 12. All 484,270 claimant-owned
interior evaluation points missed by the maximum bucket; census shape
was never the problem (23.2% already had c≤2). Corollary, worth
stating as a theorem-shaped negative: NO raw scalar finite-DTW-bound
extension (higher c, more shapes, better comparisons) can ever fire at
the unbounded contract. That entire direction is closed.

**The pivot that survived:** deadline-shaped semantic families — claims
of the form "no WIN can complete before a proven finite deadline" —
fire heavily at depth with zero counterexamples across the corpus:
defender-restore (224,761 fires), deadline-ES (168,400), ES + ordered
pre-block witness (256,386), ES + disjoint-triple witness (171,408).
Honest boundary carried in the report: none proves permanent no-WIN at
the unbounded contract, so consumption can only be a deadline-aware
defer/reopen scheduler (defer expansion past the proven deadline,
reopen after), and only AFTER Lean proof. Ordered proof queue:
deadline-family completeness + base ES → ordered pre-block → exact
defender restore → disjoint-triple invariant (dependency order, not
fire-rate order).

**Integrity:** all counterfactual instrumentation cfg(test)
default-off; official Phase A 362.15 s / Phase C 418.49 s both
zero-failure with identical 31-row search trajectories; identity subset
exact; non-test release build clean; SHA-256 manifest zero mismatches;
strict verifier untouched. The naive-SS c=3 refutation was honored as a
design constraint (no candidate infers deadness from max-c alone).

## 2026-07-17 — R-IE2: memory round CLEARS every bar — incremental enumeration is production-ready (−8.74%, 0-byte payload)

**Anchor:** branch `hunt/incr-enum` @ bcf2cc70,
HUNT_REPORT_INCR_ENUM_MEM.md + five raws. Completes R-IE1.

**The design:** selected-edge, path-local reconstruction — the tiny
parent family/kernel (means 2.5 windows / 3.8 kernel cells) is
reconstructed only when a selected attacker-pair child is still
unexpanded; nothing is retained in lazy edges, arena nodes, deferred
frontiers, or TT entries. R-IE1's 353.39 MiB snapshot payload drops to
**0 bytes**; peak TT unchanged.

**Results:** same-build official A/B 360.50 s → 328.99 s (**−8.74%**;
−7.06% vs the R-IE1 batch baseline), all 31 rows identical on every
search statistic; shadow field-for-field equality 2,910,349/2,910,349
with exactly 2 safe batch fallbacks (defender expansions without an
active selected-parent frame); leaf cells h8 **−9.04%** / h16
**−8.62%** — better than the inline-snapshot version everywhere.
Production build free of the implementation; strict verifier untouched.

**Status:** every promote bar met with the memory objection dissolved.
Production wiring (TSS_INCR_DEFENDER in the official profile) is an
owner decision; orchestrator recommendation is YES — exact search
identity, zero memory cost, ~8–9% on both deep and leaf surfaces.

## 2026-07-17 — R-PC1: constant-factor round lands −30.21% on the official wall — the hot core was hash/layout-bound

**Anchor:** branch `hunt/incr-enum` @ 0415fcec,
HUNT_REPORT_PAIRCLASS.md + 14 raws (incl. an independent orchestrator
identity rerun).

**The design:** three semantics-identical (mode-a) rewrites of the
measured residual hot core — no search-rule change of any kind. (1)
`AHashMap`/`AHashSet` at hot maps/sets whose iteration order is never
observable (pair gate coordinate→window maps, second-candidate and
unordered-pair dedup sets, planner membership/rank indexes, fork-degree
accumulator). (2) Exact canonical-frame contender pruning: compute all
12 phase keys + minimum transformed stone tuples allocation-free, fully
sort only exact lexicographic contenders; the historical twelve-sort
implementation is retained as a cfg(test) equality oracle with a
per-prefix ×12-transform test. (3) One sorted turn-root occupancy per
defender plan, with each exact pair key produced by merge-encoding two
sorted extras — byte-identical to the historical constructor.

**Results:** official 1 GiB consuming profile 328.99 s → **229.62 s
(−30.21%)** — the second-largest single-round wall gain of the campaign
(after cap-resume). Component attribution: canonical frame −74.54%,
final keys −73.64%, pair bucket −27.66%, fork scan −20.37%; defender
planning inclusive −54.76%. Identity is total: 31/31 official rows,
31/31 component rows, 3/3 fast rows, 2/2 leaf rows identical on every
search statistic vs the R-IE2 raws; counter fingerprint
`9e9bcaa2f1a631ea` with 2,910,351 calls / 11,002,776 residual patches /
0 mismatches unchanged; leaf wall h8 −11.46% / h16 −21.06% with strict
verification of every hard result; peak TT and the 0-byte snapshot
payload unchanged; strict verifier untouched.

**Why it matters for the paper:** after the algorithmic levers closed
(sound-reveal ceiling, threshold scale, kernel taxonomy), the wall
decomposition said the engine was ~80% pair-generation + defender
planning. This round shows a large constant slice of that was
implementation, not search: SipHash defaults and repeated full sorts on
paths called ~3M times. Combined R-IE2+R-PC1 arc: 360.50 s → 229.62 s
(−36.3%) at exact search identity.

## 2026-07-17 — R-G1: canonical Obligation J REFUTED (exact counterexample); classical strategy stealing FAILS in Hexo — two review-confirmed game-theory results

**Anchor:** branch `hunt/gap-raw` @ 0f7e9405 (result 06a1a649 + hostile
review + errata fold). Codex ultra round + ultra hostile review:
BOTH DOCUMENTS SOUND-WITH-ERRATA (12 ACCEPT / 5 editorial errata,
GAP_RAW_REVIEW_ROUND3.md).

**Result 1 — the draw-side potential route is closed as canonically
posed.** Theorem R3.1: an exact normative root P* with
Φ = B₂ = 8/9 < 1 and no imminent threats — eight isolated count-2
gadgets plus three defender-anchored launch rows — where EVERY legal
defender pair admits a legal attacker response forcing B₂ ≥ 11/9. The
mechanism: two defender stones recover ≤ 2/9 of potential and can touch
at most two of three disjoint launch sites; the attacker plays an
adjacent pair at the untouched site, birthing five fresh count-2
windows (+5/9) — which the round proves is the exact fresh-pair ceiling
(L9.8), so the counterexample saturates a hard bound rather than
exploiting bookkeeping. Canonical GAP-GLOBAL-RENEWAL falls with it.
Honest boundary: the attacking response completes nothing (τ=0 after),
so GAP-RAW — perpetual defender survival itself — remains OPEN; what
died is the Θ₂-invariant proof route. Banked for the successor
(GAP-REPLACEMENT-INVARIANT): B₂ ≥ |I|/3 debt bound (sub-unit debt
already implies serviceability), clause-3 redundancy, the exact
one-pair renewal margin criterion, and a two-clause reformulation of J.
Machine corroboration: a test-gated harness case exhaustively checks
all 2,016 quotient defense pairs and the exact worst row (33/27).

**Result 2 — Hexo is a documented counterexample to classical strategy
stealing.** From the production rules (formalized with source
citations, review-verified): deletion-based stealing fails by two
independent obstructions. S3: legality is not deletion-monotone under
the radius-8 growth rule — erasing the compulsory opening stone removes
a later LEGAL opponent frontier move (exact witness (8,0)). S4: the
singleton-then-pairs opening cadence never aligns — the role-swapped
shadow needs counts (1,0) but deletion leaves (2,0), and translation
cannot fix it. Scope kept honest: only the classical
identity/deletion coupling is refuted; opener non-loss stays OPEN with
named repair obligations (GAP-FRONTIER-COUPLING, opening alignment,
a non-loss determinacy bridge).

**Why it matters for the paper:** the first value-side structural
results for the game — a sharp negative boundary on potential-based
draw proofs (with the exact 5/9-vs-slack arithmetic of why), and a
natural positional game where strategy stealing provably breaks for
growth-rule reasons. Both directions continue with named, narrowed
targets.

## 2026-07-17 — R-G2: the promotion-tempo account — exact TEMPO factorization PROVEN; three invariant classes excluded; GAP-RAW narrowed to two named gaps

**Status:** landed on `hunt/gap-raw` (`8261c177` proof round 4, `12980bc8`
hostile review + errata). Hand proofs only (no machine runs this round);
hostile ultra review verdict SOUND-WITH-ERRATA — every numbered theorem
CONFIRMED, one MAJOR prose-scope erratum folded (R4.7.1), one MINOR
framing erratum folded (mode-(b) ingredient, not mode (b)).

**Result 1 — the replacement invariant exists locally and factorizes
exactly (R4.1/R4.2).** Define `tau(P)` = deadline-zero demand that must
be serviced now, and `TEMPO(Q)` = the exact maximum service demand the
next two Attacker placements can create at an Attacker handoff. Then
`TEMPO(Q) = max over legal ordered Attacker pairs b of
tau(Q + A@b1 + A@b2)`, and `TEMPO(Q) <= 2` iff `Q` is unripe. TEMPO is
deliberately NOT additive and does not dominate Θ₂: it charges a
two-trigger tier inside each interaction component plus only the two
largest one-trigger demands — exactly matching the Attacker's
two-placement turn. This formalizes why obligation J's refutation root
is actually safe for the Defender even at Θ₂ ≥ 11/9. The defender-side
form is same-pair sound: one actual ordered reply both services `I(P)`
and hands over an unripe position.

**Result 2 — three whole invariant classes are now theorem-dead
(R4.3/R4.4/R4.7.1).** (a) Remote-third-component necessity: no state
predicate can cover every Φ<1 root, imply τ≤2, and be unable to
distinguish two far-separated component copies from three — killing
max-local, fixed-radius universal/conjunctive, and top-two-only
current-demand invariants at their exact scopes. (b) Zero-grade-contact
strategies lose: an exact Φ = 1/√3 root forces any Defender pair that
kills zero graded mass to concede τ=3 — a forever strategy must
sometimes pre-empt non-imminent stock (the right statistic is
shared-trigger congestion, not count-three mass). (c) Uniform positive
dormant-component charge is impossible: any root-uniform account with
`C ≥ ε·N_dorm` blows through every fixed bound (scope narrowed per
review: selective-type/root-dependent/vanishing charges are NOT
excluded — the reviewer exhibited an escaping `C_select`).

**Result 3 — a bounded strategy ingredient (R4.8 + review corollary).**
The sealed-pencil transverse-seal construction gives a one-cycle
`S_T`-bound: after the two prescribed extensions the stabilizer pair
witnesses `M ≤ 2`, so the minimizing actual pair hands over
`TEMPO ≤ 2`. Reachability and closure of the sealed class remain open.

**Honest boundary:** GAP-RAW (perpetual defender survival) and
GAP-REPLACEMENT-INVARIANT stay OPEN. The program is now narrowed to two
exact obligations: GAP-TEMPO-INITIALIZATION (`M(P0) ≤ 2` on the τ=0
strict-root slice) and GAP-TEMPO-REPAIR (one strategy preserving `M ≤ 2`
after every legal response). Named resume: classify returns through the
six surviving sealed-pencil count-one extremes, then prove or refute the
one-cycle Bellman closure for `M ≤ 2`.

**Why it matters for the paper:** round 3 killed the additive-potential
route; round 4 supplies its replacement — an exact, turn-matched
order-statistic account with a proven local factorization — and fences
the search space with three impossibility theorems that any future
draw-proof attempt must respect.

## 2026-07-17 — R-T1: df-pn re-traversal theory — the formal counterpart of R-TS1 (10 review-confirmed theorems; second+1 justified by an accounting ceiling)

**Status:** landed on `claude/tss-vcf-width` (`28a276e5` theory,
`71fd05e4` hostile review + errata). Doc-only theory round grounded in
R-TS1's exactly measured instance; hostile ultra review verdict
CONFIRMED-WITH-MAJOR-ERRATA — all ten formal results stand (none
refuted, none downgraded), with model-scope errata folded.

**Result 1 — a traversal ceiling with a near-matching family (T1/F1).**
For persistent, progress-certified df-pn on a finite acyclic arena with
exact selected-cutoff deepening, total recursive activations are at most
(d+1)·E with E ≤ 2N−1 — re-traversal is O(N·(d+1)) — and an explicit
unary staged-deepening family achieves Θ(N·d) repeat activations, so the
ceiling is tight up to constants.

**Result 2 — neither delta nor prior scale controls total work
(T2/T2b/C2/T3/T4).** Explicit worst-case families prove: a coarser
additive threshold increment (δ=2) can cost Θ(N) extra expansions that
unit increments avoid — even with all priors equal to 1; and a single
2× non-admissible prior overestimate can starve the cheapest winning
line behind Θ(N) useless expansions. Under unit-calibrated score
response the starvation is bounded by (b−1)(ρP+δ−1) — so certified
admissible floors (agenda 1.3) provably cap starvation, strengthening
that lever's case. Engine transfer ranges are stated exactly
(sentinel-clamping errata folded; T2 transfers for q < 10^9).

**Result 3 — the engine's +1 schedule is justified by an accounting
identity (T6).** Any scheduling change with positive extra non-revisit
cost can win only if the revisit cost it saves exceeds that extra cost;
the revisit-attributed share of baseline wall is an absolute saving
ceiling. Measured: 7.02% revisit share vs the observed +7.54%
non-revisit inflation of the tested δ=2 arm — even PERFECT revisit
elimination could not have paid for it. Retaining `second+1` is now
theorem-backed, not just A/B-backed.

**Result 4 — the δ=2 catastrophe mechanism, honestly bounded.** The
review-confirmed arithmetic shows δ=2 REDUCED revisits per expansion by
26.3% while raising total expansions 3.22× on the hardest row — coarser
windows trade re-entry for over-expansion past the optimal frontier.
The causal mechanism (high-mass score-band crossing + admission
saturation amplification, D1's (k−1)·M duplication bound) remains
labeled CONJECTURE E1 with a named cheap follow-up (R-T1.1
frontier-band census).

**Why it matters for the paper:** the checked literature has no
total-work theorem jointly parameterized by additive increment, prior
scale, depth, and bounded-TT behavior — this round supplies the first
such family of results, anchored to an exactly measured production
instance, and converts two of the engine's empirical tuning choices
(+1 increment, admissible-leaning priors) into theorem-backed ones.

## 2026-07-17 — R-ST2: stealing round 2 — surplus-stone calculus, a fixed-map obstruction class theorem, and the determinacy bridge (NL_F still open)

**Status:** landed on `hunt/gap-raw` (`a85aa311` round 2, `57fcbda8`
hostile review + errata). Hand proofs anchored to engine source
citations; hostile ultra review verdict CONFIRMED-WITH-ERRATA — no
FATAL or MAJOR defect, all theorems stand at their formal scopes.

**Result 1 — the exact surplus-stone calculus (S5-S7).** One-way
legality monotonicity (a still-empty shadow reply stays legal on the
bigger real board) plus the exact frontier-gap formula: the real-only
moves a subset shadow cannot copy are precisely Γ(A,E) =
(N₈(E)\N₈(A))\B. Surplus stones are monotone-helpful in one direction
and monotone-harmful exactly on Γ — this cleanly separates the
"extra stones never hurt" intuition from S3's growth-rule trap, with a
complete characterization (geometric frontier-neutrality) for one
surplus stone.

**Result 2 — a fixed-map obstruction class theorem (S9.1/S11,
review-confirmed).** No synchronous, owner-faithful, no-invention
role-swapped shadow mapping all real stones injectively one-for-one
can be a legal coupling — for EVERY coordinate map, not just
isometries — and the omit-one-stone repair fails for every
no-invention fixed-isometry immediate-copy history map. The
no-invention premise is load-bearing (one invented proxy stone of each
color repairs the raw cadence counts), so the honest survivor class is
exactly: dynamic recoding, virtual/proxy bookkeeping stones, and
strategy-specific invariants. Any future Hexo stealing proof must live
there. Narrow FIFO/preannouncement/terminal-lag obstructions close the
easy variants.

**Result 3 — the determinacy bridge (D1/D2, PROVEN from CITED
Gale-Stewart).** The payoff "S completes six at a finite prefix" is
open, the macro-game encoding is valid (off-path totality erratum
folded), so NL_F ⇔ "S has no winning strategy." Opener non-loss no
longer needs a constructed drawing strategy — refuting a second-player
win suffices.

**Why it matters for the paper:** round 1 showed classical stealing
breaks in Hexo; round 2 turns that into a sharp boundary — an exact
calculus for when surplus stones help, a theorem-fenced survivor class
for coupling arguments, and a determinacy reduction that halves the
logical burden of the opener-non-loss question. Resume:
GAP-PROXY-SHADOW (the invented-proxy repair the review itself surfaced).

## 2026-07-17 — R-ST3: proxy shadows — the cadence obstruction is REPAIRED (two-proxy synchronization theorem); static proxies die in one round; NL_F narrowed to dynamic proxy management

**Status:** landed on `hunt/gap-raw` (`890aa531` round 3, `5023169f`
hostile review + errata). Hand proofs on the engine-cited rules
contract; hostile ultra review verdict CONFIRMED-WITH-ERRATA — ranked
outcomes (b) and (c) both stand; one MAJOR checklist omission repaired
as a new named obligation (P5R).

**Result 1 — the two-proxy synchronization theorem [(c), PROVEN].**
The cadence mismatch S4 — one of round 1's two refutation pillars — is
constructively repairable: two invented stones can be scheduled as
genuine legal Hexo placements (one being an actual prescription of the
allegedly winning shadow strategy), and a D6 isometry can always be
oriented so the real opponent's first-turn stones avoid both proxies.
Result: a legal, strategy-consistent role-swapped shadow exists at the
first synchronized checkpoint, and the cadence alignment is PERMANENT
(parity checked through both phases and mid-turn wins). Scope kept
honest: conditional on future placement transfer — the theorem
relocates the difficulty from the opening to the transfer invariant,
it does not remove it.

**Result 2 — static proxies cannot work [(b), PROVEN for C_static²].**
If the two proxies and the isometry are frozen with exact two-way
copying, at least one proxy has an already-legal real preimage; the
opponent plays it and the collision kills the coupling within one
ordinary round (or the reply lift had already failed). Dynamic proxy
management — retiring, moving, or recoding proxies as
representation-level operations — is the exactly-characterized
survivor class.

**Result 3 — the complete obligation ledger.** Frontier/collision
obligations, proxy terminal-fabrication gadgets (proxies must never
fabricate or block a six for either color), and — added by the hostile
review as obligation P5R — real-S terminal reflection through OLDER
surplus stones: the review exhibited a configuration where every
per-move check passes yet the real board completes a six the shadow
cannot see. Any future NL_F proof via proxy coupling must discharge
P5R explicitly.

**Why it matters for the paper:** round 1 proved classical stealing
fails in Hexo; rounds 2-3 now prove exactly WHY and WHERE the repair
must live — the opening/cadence half of the obstruction is fully
solved (a genuine theorem with explicit constructions), the static
repair is impossible, and the open frontier is a single sharply-posed
invariant-design problem (GAP-PROXY-RETIRE-OR-RECODE + P5R). This is a
publishable arc: a natural game where strategy stealing breaks, plus
the exact boundary of its repairability.

## 2026-07-18 — R-RS1: root-stabilizer orbit consumption CLOSED NULL — the search never leaves the principal orbit

**Status:** landed on `hunt/root-stabilizer` (`c33e6ff0` implementation
+ atlas, `e1b440f8` binding-rung verdict + raws). Implementation and
soundness were clean (strict verifier PASS on every arm, fail-closed
differential 0); the lever itself is economically dead.

**The idea:** 62.92% of top opening families have nontrivial root
stabilizers (D6 census), so orbit-level deduplication of root children
— solve one representative per stabilizer orbit, consume the verdict
for its siblings — looked like a structural win (rank-1 family: 51,752
raw children collapse to 26,030 orbits).

**The measurement:** A/B (baseline vs consumption) across top
stabilizer-rich families × transforms at cap 128 (12 arms, original
round) and cap 1000 (8 arms, binding rung, official deep profile), plus
one 4.5-hour partial 10k arm. Uniform result: the engine's second-best
threshold descent commits to a single principal orbit — at cap 1000,
26,029/26,030 and 24,807/24,808 orbits saw ZERO work, one orbit
absorbed 998/999 expansions, and consumption changed expansions by
exactly 0.0000% (wall deltas −0.17%/+4.22% = contention noise around
identical work). The orbits deduplication would remove are orbits the
search never touches.

**Why it matters for the paper:** a clean negative with a mechanism —
symmetry-based root deduplication is worthless under depth-first
proof-number descent because threshold scheduling concentrates all
work in one orbit long before sibling orbits are opened. The atlas +
consumption harness are retained with a precise reopen condition: any
future root-parallel or portfolio-descent scheme (where sibling orbits
DO get work) can re-measure instantly.

## 2026-07-18 — R-G3: tempo round 5 — sealed-pencil classification, one-cycle M≤2 theorems, and the shared-hub cascade that refutes naive Bellman closure

**Status:** landed on `hunt/gap-raw` (`d93d5768` round 5, `aed0fecb`
hostile review + folded errata). Review verdict SOUND-WITH-ERRATA:
every round-5 theorem CONFIRMED (all coordinates, residuals, and
hitting sets recomputed by hand), zero refutations; one MAJOR scope
downgrade folded.

**The results:** (1) L11.1–L11.4 give the COMPLETE first-return
classification of the exact sealed pencil — 31 alive labels before
sealing, exactly 6 disjoint count-one residual arms + 5 count-two
common labels survive, and every legal Attacker return pair is
exhausted by a 0/1/2 Q-trigger partition with an at-most-two-cell
transversal covering all 45 axial pairs. (2) R5.1/R5.2: one-cycle
value theorems — from an isolated (or radius-21-separated union of)
exact sealed handoff(s), EVERY legal Attacker pair returns an epoch
with M≤2. (3) R5.3/R5.3.1: the round-4 policy S_T provably REJECTS the
transverse seal entrance (strict objective gap 0<2, robust to
tie-break) on a genuine S_T-consistent history — the natural route
into the nice sealed class is closed. (4) L11.6/L11.7: a strict-
potential (Φ<1) shared-hub position whose forced two-cell service
cascades M: 2 → ≥3 — three count-three labels on three axes survive
every mandatory service. (5) R5.4: embedding that hub behind two
remote count-five gadgets produces a position satisfying every side
condition of the statewise equation-(22) closure conjecture whose
successor still has M≥3 — unrestricted Bellman closure of the
two-coordinate tempo state is FALSE.

**Why it matters for the paper:** the tempo program now has a sharp
shape. The two-coordinate state (τ≤2, M≤2) is locally provable
(one-cycle theorems) but NOT statewise-inductive (hub counterexample);
any perpetual-defense proof must be strategy-reachable, not statewise.
The folded MAJOR erratum makes the remaining obligations
quantifier-precise: fixed-S_T hub reachability (refutes that policy
only), strategy-independent hub forcing (∃P₀∀S∃α — the GAP-RAW
counterroute), and the positive universal repair (∀P₀∃S∀α) that must
also exclude every other escape class. GAP-HUB-FANOUT-REACHABILITY is
the named subproblem; GAP-TEMPO-INITIALIZATION and GAP-TEMPO-REPAIR
remain the broader gates.

## 2026-07-18 — R-T1.1: frontier-band census decides E1 = SPLIT — the delta-2 catastrophe is band work first, saturation amplification second

**Status:** landed on `claude/tss-vcf-width` (`224a3682`: cfg(test)
census counters, report, 4 hashed raws, theory-doc E1 upgraded
CONJECTURE → MEASURED). Both instrumented arms reproduced the
historical R-TS1 solves EXACTLY (1,879,611 vs 6,054,588 expansions,
identical peak TT bytes), strict-verified WIN, histogram conservation
exact to the single root node.

**The question:** R-TS1 measured that widening the df-pn threshold
delta from +1 to 2 blew the hardest official row up 3.221× and
saturated the 1 GiB indexed TT. CONJECTURE E1 offered two mechanisms:
(a) the wider window crossed a high-mass competitive score band
(over-expansion at selection boundaries), and/or (b) admission
saturation then amplified work via loss of indexed transposition
reuse. Which is it?

**The verdict — SPLIT, with exact fractions.** Timestamping the first
admission refusal (expansion 3,586,288) decomposes the 4,174,977
excess expansions into 40.878716% BEFORE any saturation and 59.121284%
after. Clause (a) is confirmed as the originating cause: 1.7M excess
expansions exist before the index ever refuses an entry, and 83.18% of
them are charged to selections that began TIED with the second-best
sibling — the tie band carries 82–85% of all classified work in every
segment, direct evidence of a high-mass competitive band. Clause (b)
is temporally supported (the majority of excess is post-saturation)
but the census cannot distinguish true D1 duplicate re-expansion from
work the widened schedule would have done anyway; that causal gloss
stays at SKETCH level. Bonus: the review's missing sentinel-hit
control is closed with nonzero counts — delta 2 actively clamped at
PN_INFINITY (84 inherited + 128,957 increment strict clamps vs zero
under +1), so no delta-2 behavior may be attributed purely to the
`second_best + delta` expression.

**Why it matters for the paper:** this completes the re-traversal
theory arc with a measured mechanism. The theory proved Θ(N) band
families exist (T2/T2b); the census shows the production catastrophe
is exactly that shape — a tie-heavy competitive band that the +1
schedule slices minimally and the +2 schedule re-enters en masse,
with TT saturation as a secondary amplifier. Combined with the T6
identity (revisit share ≤7% caps any possible widening payoff), the
second+1 policy is now backed end-to-end: theorem ceiling + measured
mechanism + sentinel-controlled A/B.

## 2026-07-18 — R-ST4: dynamic proxy rebinding — one reactive escape per turn is provably insufficient, and the exact price of shielding surplus stones

**Status:** landed on `hunt/gap-raw` (`8fb68864` round 4, `67f996d1`
hostile review + folded errata). Review verdict SOUND-WITH-ERRATA:
both ranked theorems CONFIRMED (all coordinates, cadences, windows,
transversals, and hitting sets recomputed by hand), zero refutations;
one MAJOR logical-shape repair to the resume checklist folded.

**The results:** (1) S23, the dynamic impossibility: starting from
round 3's genuine two-proxy synchronization, any coupling in the
zero-lag total-exact owner-faithful family — now allowed arbitrary
pre-turn rebinding (isometry changes, proxy retirement/backing,
fillers) plus ONE coordinate-reactive escape in the tested S turn —
still fails. The engine is the "proxy-support cut": a genuine shadow
history is radius-8 support-connected, so every committed exact
binding exposes a partition edge whose real preimage is empty and
legal; applying the cut, absorbing the single permitted repair, and
applying it again forces a second unrepairable collision in the same
ordered turn. This strictly extends round 3's static impossibility
(the review exhibits a hand witness where one backing/filler repair
legally executes — the relaxation is real, and it still dies). (2)
S28, the positive half: a phase-sensitive "deadline shield" (every
threatened window must keep more empty cells than S has placements
remaining before F acts) is a sound P5R invariant defining a nonempty
conditional class of safe dynamic couplings — with exact service
economics: restoring the shield costs an F-pair transversal of size
at most 2, a legal three-axis fork pushes it to 3, and permanently
fencing all 18 windows through one surplus stone costs exactly 6
blockers (tight both ways). (3) S24: a winning strategy would admit a
strategy-dependent finite rebinding horizon — so no cheap escalation
to "infinitely many rebindings needed" is available, honestly leaving
finite budgets K>=2 open.

**Why it matters for the paper:** the stealing arc now has matching
halves — the impossibility side has climbed from static maps to
budget-1 dynamic repair, and the possibility side has its first real
invariant (the shield) with exact costs. The folded MAJOR erratum
makes the frontier precise: a successor must, per placement, choose
zero-lag repair / admissible lag / same-step terminal certificate AND
discharge service, persistence, and regression obligations — with any
counterexample forced on the candidate's own strategy-consistent
history. NL_F remains open at GAP-ZERO-LAG-WINDOW-RECODE /
P5R-SERVICE.

## 2026-07-18 — R-OS1: the §1.5 ordering premise is REAL — winning children sit at median rank 6, and a zone key moves them to 4

**Status:** landed on `claude/tss-vcf-width` (`bb7efa44`: cfg(test)
offline-rank instrumentation, report, 6 raws). All eight officially
solved WIN rows reproduced their R-CD1 baselines exactly (incl.
`0l4291i_live` at 1,879,611 expansions); verifier untouched;
default-off identity and production build green.

**The measurement:** across 26,710 proven attacker pair nodes, the
current generator order puts the eventually-winning child at rank 1
only 14.96% of the time (median rank 6, top-4 coverage 36.91%) — so
the long-standing suspicion that "winning children sit deep because
of OUR ordering, not the game" is confirmed as a real, quantified
inefficiency. Re-ranking the same children offline by `zone_bound`
(exact max claimant-support distance of the child pair) lifts top-2
coverage +13.45pp and top-4 coverage +21.65pp (58.55%) and cuts the
median to 4, with the signal concentrated at depth 16+ where 98% of
the nodes live. Honest tail warning: the same key worsens mean rank
(8.70→10.51) and triples the rank-33+ tail (3.86%→10.19%), and
per-row breadth is heterogeneous — so the verdict is PROMISING, not
promoted. `d_stone` is the safe secondary (mean −4.3%, tail shrinks,
no top-4 gain).

**Why it matters for the paper:** this re-opens the generator front
that R-CD1 had closed under the old order — the 7.83% sound-reveal
ceiling was an artifact of measuring under the baseline ordering, and
a 2–4-wide reveal prefix now covers 34–59% of winners instead of
20–37%. The prescribed next step is a default-off live A/B wiring
`zone_bound` as a risk-controlled tie/band key (preserving
width/urgency/fork-prior classes) with hard identity gates and
per-row expansion/wall stops on the three tail-risk rows.

## 2026-07-18 — R-OS2: the ordering lever dies honestly — offline rank gains do not convert to a live df-pn win

**Status:** landed on `claude/tss-vcf-width` (`0e4a34e9`: default-off
implementation, report, 10 raws). Three-arm official A/B, all rows
same-verdict same-rung, every certificate strict-verified, flag-off
path byte-identical to pre-change.

**The result — NULL/REGRESSION.** R-OS1's offline promise (winning
children median rank 6 → 4 under zone_bound re-ranking, top-4
+21.65pp) was wired live as a maximally risk-controlled band key
(historical width/urgency/fork-prior classes preserved, PN band ±1,
generation set unchanged, key cost measured at 246 ns/key = 0.31% of
ladder wall). It did NOT convert: zone_bound summed winning-row
expansions +0.31% (hard row +0.53%), wall −0.63% — mixed, no material
win. The d_stone control regressed outright (+5.94% expansions, hard
row +7.04%) despite being the "safer" offline signal. Implementation
retained default-off.

**Why it matters for the paper:** this completes the §1.5 generator-
ordering arc with a mechanism-level conclusion: static generation-
order rank is a misleading proxy for df-pn visit order, because the
dynamic PN/DN threshold descent already re-ranks children by proof
progress — the winners that sit "deep" in generation order are being
found through transpositions and threshold switching at near-zero
marginal cost. Combined with R-T1's second+1 backing and R-RS1's
orbit null, the scheduling/ordering family of levers is now
comprehensively closed: measured, mechanism-understood, and negative.
The narrow residue (a default-off reveal-prefix measurement targeting
classifier work rather than visit order) is queued behind higher-value
cargo work.

## 2026-07-18 — R-G4: the greedy tempo policy S_T is REFUTED — an exact strategy-consistent line assembles the shared hub and forces the loss

**Status:** landed on `hunt/gap-raw` (`110042e5` round 6, `8ac6caae`
hostile review + folded errata). Review verdict
SOUND-WITH-MINOR-ERRATA: R6.1 CONFIRMED (every S_T reply re-derived
from its exact minimize-TEMPO-then-lexicographic definition), R6.2
CONFIRMED-AND-STRENGTHENED — the reviewer computed the exact post-hub
values, M=4 and next τ=4, stronger than the round's ≥3 bound. No
quantifier leak anywhere.

**The result (Q1 of the round-5 decomposition):** from an exact strict
root with potential Φ=0 (three remote anchors), there is a legal
Attacker continuation, consistent with the round-4 defense policy S_T
at every single Defender turn, that (1) walks through the known
axial-cleanup handoff, (2) erects a "four-pencil tempo shield" — a
20-window diamond whose value structure puts every S_T reply on a
plateau ray, (3) builds the five count-two shared-hub focal windows
behind that shield over five stock turns with no hidden count-three
label ever appearing, and then (4) springs the round-5 hub cascade,
which survives the extra alive labels: after mandatory service the
demand jumps to τ=4 and the Attacker wins. S_T — minimize immediate
TEMPO, break ties lexicographically — is therefore refuted as a
perpetual-defense witness.

**The mechanism (why a greedy defense dies):** hub pre-emption was
always legal and would have saved the game — but it TIES the greedy
objective (immediate TEMPO=2 either way) at the critical P_stock
epoch, and the lexicographic tie-break picks the losing ray. The
failure is not myopia about value; it is that a one-step tempo
objective cannot see next-state risk hiding inside a tie. Any repaired
policy must be Bellman-aware exactly there. Bonus theorem L12.6: every
position whose alive windows all have count ≤2 is an initialized M≤2
root (count ≤1 gives M=0) — the first initialization class beyond
individual instances.

**Why it matters for the paper:** this is the tempo program's headline
so far — a complete, hostile-verified, move-by-move refutation of the
natural greedy defense in Hexo, with the failure mechanism isolated to
a tie-breaking blind spot. Together with round 5's Bellman-closure
refutation it fully shapes the remaining perpetual-defense question:
Q2 (can the hub be forced against EVERY defense — ∃P₀∀S∃α) and Q3 (is
there a tie-aware policy holding M≤2 everywhere — ∀P₀∃S∀α) are now the
exact open fronts, with the sharpest next question pinned: at P_stock,
does some hub-pre-empting pair keep max_b M ≤ 2 over every Attacker
response?

## 2026-07-18 — R-ST5: the lifetime two-escape ceiling, a proven rolling-lag service invariant, and the first window-faithful non-isometric representation

**Artifacts:** `STRATEGY_STEALING_ROUND5.md` + hostile ultra review
`STRATEGY_STEALING_REVIEW_ROUND5.md` (verdict SOUND-WITH-MINOR-ERRATA,
all three objectives CONFIRMED, errata folded as §44) on branch
`hunt/gap-raw` at `3000a117`.

**What was proved (three scoped results):**

1. **S32 — the third support cut (lifetime K=2 REFUTED).** The
total-exact isometric branch-(A) candidate class with a fixed
LIFETIME budget of two coordinate-reactive escapes fails: after both
escapes are spent, the candidate's own next σ-consistent pair forces
either an untransferable shadow-F terminal verdict (six shadow stones
vs five real — no physical certificate possible) or a third occupied
target, and the support-cut premises provably survive both escapes
(the review recomputed the full stone-count table at all three cut
checkpoints). Per-pair K=2 remains OPEN and is now the sharpest
branch-(A) question.

2. **S33/S34 — branch (B) gets its first genuine positive theorem.**
A physical rolling one-cell lag queue (debt rotated by actual shadow
appends, fillers persist forever, no undo) provably maintains the
deadline shield δ>m on the exactly-named finite continuation class
A_FS2, with an exact two-stone real-board service rule (minimum
transversal of the urgent windows, τ_E≤2) restoring δ>2 before every
S turn. The class is nonempty at active-service scope (S35). This is
conditional — first-unsafe coordinates, τ_E≥3 forks, and the P3
carrier for arbitrary σ stay outside — but it is the first proven
lag-based defense module after rounds 2-4 killed every unguarded
variant (S13/S14 stay as regressions, now explicitly cross-referenced).

3. **S36-S39 — window-deficit certificates: representation without a
cell map.** First rigorous definition of window-faithful non-isometric
representations: a selector ν assigning each real window obligation a
physical shadow window with no more holes (δ_H(ν(W)) ≤ δ_R(W)). A
concrete reachable example is PROVEN outside every injective point
representation (two distinct real windows share one shadow
certificate — no injection can induce that), with an exact one-append
maintenance algebra and a P5R transfer lemma implementing the S26
same-step terminal route. The deadline-shield obstruction (S39.1)
shows this representation family cannot certify shield-unsafe debt on
a winning-σ history — a scope limit, not a contradiction.

**Why it matters for the paper:** the strategy-stealing program now
has all three branch fronts pinned with exact theorems: branch (A)
repair budgets have a proven lifetime ceiling at 2, branch (B) has a
proven conditional service invariant instead of a graveyard of failed
lags, and the representation space provably extends beyond point maps
— the first hard evidence that NL_F, if provable by coupling at all,
may need window-level rather than stone-level representations. The
review's 12-item obstacle list is the authoritative open state; the
sharpest question is the P3 carrier: can every alleged-winning σ be
causally paired with the canonical service transversal while
surviving S18/S20/S12?

## 2026-07-18 — R-ST6: the P3 event carrier — prescriptions need no coordinate map, and the whole stealing problem compresses to co-terminal alignment

**Artifacts:** `STRATEGY_STEALING_ROUND6.md` + hostile ultra review
`STRATEGY_STEALING_REVIEW_ROUND6.md` (SOUND-WITH-MINOR-ERRATA, all
three objectives CONFIRMED, errata folded as §53) on `hunt/gap-raw`
at `09e27a93`.

**The headline mechanism (S40, PROVEN at scope):** for five rounds,
every strategy-stealing coupling for Hexo died at P3 — turning the
shadow strategy's prescribed moves into legal real moves. Round 6
proves the prescriptions never needed a coordinate map at all: pair
each shadow prescription TEMPORALLY with the canonical real service
placement the defense already plays after the preceding S turn. On
the named terminal-closure class A_FS2^ET, this event carrier is
PROVEN causal (the second prescription is queried only after both
first placements are nonterminal — the review verified no future
information leaks into the pairing rule), immune to the S18
reverse-legality gadget by construction (S41 exhibits a live trace
whose shadow prescription has an ILLEGAL inverse — and the carrier
doesn't care), and safe from S12 pre-announcement because service is
selected after S's completed turn. The entire remaining F-role
obligation compresses into one named residue: co-terminal alignment
(every terminal shadow win needs a same-step physical real
certificate). S42 proves the terminal-blind shortcut dies on its own
history, so that residue is genuinely load-bearing.

**Secondary results:** per-pair K=2 is PARTIAL — S43 proves the
budget reset provably escapes the round-5 lifetime counting (honest:
the lifetime ceiling does not transfer), while S43.1's colored
two-proxy cut adaptively kills the subclass with isometry fixed
within each pair. The S30 three-axis fork got a certificate barrier
(S44: a window-deficit certificate cannot discount a missing physical
blocker at fixed debt) and its exact price: tau_E=5, with both the
five-cell transversal and the four-cell impossibility recomputed by
the review.

**Why it matters for the paper:** this is the first genuinely
positive P3 mechanism in the program's history, and it changes the
shape of the NL_F question. Instead of five open fronts (legality,
causality, collision, reverse maps, terminal fidelity), the
nonterminal coupling is SOLVED at conditional scope and one question
remains: derive co-terminal alignment for every alleged-winning
sigma, or force adaptive S42-style failure on a genuinely winning
strategy's own history. Window-level representations (round 5) plus
event-level carriers (round 6) together mark the representation
space's real boundary: what dies is point maps, what survives is
structure that never inverts coordinates.

## 2026-07-18 — R-G5: the greedy defense is not repairable — P_stock is a universal local stop state, and tie-breaking was never the disease

**Artifacts:** `GAP_RAW_PROOF_ROUND7.md` + hostile ultra review
`GAP_RAW_REVIEW_ROUND7.md` (SOUND-WITH-MINOR-ERRATA, all four results
CONFIRMED, errata folded as §67) on `hunt/gap-raw` at `a8a0b92d`.

**What was proved:** round 6 refuted the greedy tempo policy S_T and
isolated its death to a tie-breaking blind spot — hub pre-emption tied
the objective and lost only on the lexicographic tie-break. The
natural repair program (a smarter tie-break S_T') is now CLOSED:

1. **R7.1 (hub pre-emption never repairs):** at every one of the six
plateau epochs of the refutation line, every legal Defender pair
containing the hub admits a triangle-fan Attacker response forcing
M>=3 — verified by the review with an axis- and D6-complete demand
census and full enumeration of Defender continuations.
2. **R7.2 (P_stock is a universal local stop state):** at the final
plateau, EVERY legal Defender pair — hub or not — has a response with
M>=3 (the review completed the exact label census: 127 count-one, 55
count-two, none higher). Equivalently min_a R_1 >= 3: every possible
tie refinement is excluded at once, including one-ply Bellman
lookahead. One-ply risk detects the lost state; nothing can act on it.
3. **R7.2.1 (the refutation shortens):** the very first plateau ray
reply already admits the triangle fan — the five stock turns of round
6 were diagnosis, not necessity. The review verified this is
independent of all later stock.
4. **L13.6:** the initialization ladder climbs — any tau=0 position
whose count-three residual family has hitting number <=2 is an M<=2
root (two count-three labels now covered; explicitly not renewal).

**The mechanism-level meaning:** the greedy defense does not fail
because of how it breaks ties; it fails because the Attacker can pack
three action-disjoint threat regions (pairwise-disjoint service sets,
review-verified) faster than any two-stone-per-turn defense can
service them once the packing completes. A defense policy for Hexo
must therefore be measured by WHEN it intervenes, not how finely it
scores ties. The sharpest open question is now pinned at the earliest
epoch: does ANY action at the first diamond plateau keep the worst
response at M<=2? Q2 (can a strict root FORCE arrival at P_stock
against every defense) remains the other open front — answering it
would convert this local forced-loss theorem into a global GAP-RAW
attack result.

**Why it matters for the paper:** together with rounds 5-6, the tempo
program now has a complete negative theory of one-ply greedy defense:
Bellman closure fails (R5), the greedy policy fails with an exact
strategy-consistent line (R6), and no tie refinement can fix it
because the loss is a state property, not a policy property (R7).
This is the cleanest characterization yet of why local heuristics are
insufficient for Hexo defense — the game's threat geometry outruns
any fixed-lookahead scoring rule.
