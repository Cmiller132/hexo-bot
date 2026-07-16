# SKELETON — *Searching Less, Provably: Certified Zone Theory and Solver Optimization for Hexo*

> **DRAFT — NOT FINAL. Requires further review.** Owner-reviewed once
> (2026-07-16, "decent enough for now"); every section, the capstone spec,
> and the claim→evidence map WILL be revised as campaign results land.
> Nothing here is a commitment to final structure or content.

Status: light skeleton. No prose beyond scope notes. Full writing starts
after SolverInterface.lean lands.

**Document contract (owner-ruled 07-16):** one unified document — no reader
tracks. Spine = solver-applicable work; game-theory results appear only
where genuinely novel. Every theorem pays rent: plain-language restatement +
sharp board example + measured solver consequence. Proof sketches in prose;
Lean declaration names are the trust anchors (full rigor lives in Lean and
the companions). Every number carries a commit + regenerable command.
Informal venue: shared with Hexo bot devs and theorists, not submitted
anywhere. Tone per the v3 house spec (dry, no LLM tics). Related work leads
with the honest lineage: RZOP solved defender restriction for Connect6;
radius-8 locality breaks their semantics; this work repairs it.

---

## Section plan

1. **What this is, and how to trust it** — one page. The two artifacts
   behind every claim (kernel-checked Lean decl / logged measurement with
   commit + command). Relationship to the zone-theory companion (v3) and
   the Theory of Hexo living doc.
2. **The problem: defender width in Hexo threat search** — why unrestricted
   defender reply sets explode; what RZOP did for Connect6; exactly where
   radius-8 locality breaks RZOP's semantics (the repair is the paper's
   reason to exist). [Citations per RZOP_COMPARISON.md §8.]
3. **Objects in play** — minimal definitions only, threaded through one
   running example position: windows, threats, budgets `b`, hitting sets,
   λ-orders (λ¹ forcing / λ² spare-turn), certificates, finder/verifier.
4. **Zone theory under locality (the headline)** — T3 pathwise soundness,
   the ranked zone `Z_dir ∪ Z_seed ∪ Z_touch ∪ Z_virgin`, local budgets
   D14/L11, kernel T6, sparse LOSS witnesses L13, substitution T9, DAG
   certificates T10. Per-theorem: restatement / example / what the solver
   gets to skip because of it.
5. **The certified engine** — WidePnSearch df-pn architecture;
   finder/verifier split (verifier re-derives everything; nothing uncertified
   ever mints); what "certificate-grade" means operationally; consumed
   theory: τ-init, K_b, L13 3/5, P3 pair canonicalization.
6. **Making it fast without losing the proofs** — rounds 8b → 9 → 9b:
   what each proof-backed deletion bought (stateless second candidates,
   defender plan, canonical frame); TT memory policy and the U18 DAG-sharing
   lever. The engineering story a bot dev can replicate.
7. **The capstone: how close to minimal do we search?** — table spec below.
8. **Beyond forcing: unforced AND nodes** — the spare-turn/λ² corpus (why
   the VCF corpus structurally cannot exercise `k < B`); certified
   ranked-zone integration at unforced AND nodes (Group-2); uniform-vs-exact
   clock deltas (the comparison RZOP structurally cannot produce). [Results
   land here as Group-2 rounds complete.]
9. **Game theory of Hexo: the genuinely new facts** — admission bar:
   novel AND interesting to a theorist, else it stays in the living doc.
   Current candidates: pairing threshold k=7 (EXISTS at index 12), LOSS
   witness caps 3/5, gate-sharpness fixture, potential boundary + greedy
   refutation, domination patterns. Each entry: statement in plain terms,
   the sharp position, why it surprised us.
10. **The certified opening atlas** — snapshot of the living doc: win/loss/
    open verdicts for the standard opening families with machine-checkable
    certificate pointers, grown by depth. Framing beats RZOP's 12 solved
    Connect6 openings on both certification strength and coverage.
11. **Bounded completeness: "never miss a win ≤ L"** — tier statements
    bound to the actual WideTurnGate/SolverInterface candidate set; the
    Rust ⊇ Lean differential fixture as the enforcement mechanism.
12. **Honest limits and open problems** — labeled opens (ES Prop 2,
    Domination MV, Potential tail, F+H_W, band shrinking status); what is
    PROVEN-LEAN vs PROVEN-DOC vs MEASURED (the v3 §11 distinction,
    inherited).
13. **Reproducing everything** — the commit/command table for every number
    in the document.

---

## Capstone table spec (§7) — lock this before Group-2 numbers land

**Rows:** each of the 19 VCF corpus entries (14 WIN + 5 NO), each
spare-turn/λ² corpus entry (WIN_PENDING/NO), aggregate rows per corpus;
atlas openings as a separate spot-check block.

**Engine columns:** old-narrow → round-8b → round-9b → zone-integrated
(Group-2). **Reference columns:** idtt, dfpn, pdspn — full matched-budget
columns on BOTH corpora; spot-checks only on atlas rows (owner ruling
07-16). **Minimality columns:** per-AND-node and aggregate
`|searched| / |Legal|`; λ-order proxy (count of `k < B` Universal nodes,
nonthreat OR edges, max spare-turn nesting); uniform-`8(B−1)` vs exact
per-window `E^D` zone-size delta.

**Cell contents:** status@rung, nodes, wall. **Fixed profile:** documented
2 GiB TT test profile (`TSS_BACKWALK_TT_BYTES=2147483648`), standard rung
ladder 10k/100k/1M/20M, single process. **Provenance rule:** every cell →
log file + exact command + commit hash; no cell without all three.

---

## Claim → evidence map (grows; every row must resolve before prose ships)

| # | Claim | Evidence type | Anchor | Status |
|---|---|---|---|---|
| C1 | T3 pathwise soundness + sound dismissal | Lean decl | `T3_pathwiseSoundness`, `T3_soundDismissal` (tss-lean) | PROVEN-LEAN |
| C2 | Ranked zone = mandatory closure (T4) | Lean decl | `mandatoryZone`, T4 (tss-lean) | PROVEN-LEAN |
| C3 | T5 short-path containment (amended ERRATA-25) | Lean decl | T5 (tss-lean, S9S-11) | PROVEN-LEAN |
| C4 | T9 substitution-envelope lift (.d17) | Lean decl | T9 (tss-lean, S9S-11) | PROVEN-LEAN |
| C5 | T10 DAG unfolding soundness | Lean decl | TssZones/DAGUnfolding (S9S-12) | IN-FLIGHT |
| C6 | ForcedHit / kernel T6 statements | Lean decl | ForcedHit lane (planned D19–D21, L15–L17) | PLANNED |
| C7 | Sparse LOSS witnesses (≤3 / ≤6) | Lean + Rust fixtures | L13 + tss_verify LOSS arm | PROVEN-DOC → Lean lane |
| C8 | Pairing threshold k=7 exists | Lean + witness | index-12 witness (pairing lane) | PROVEN-LEAN |
| C9 | Engine soundness architecture (single mint, verifier re-derives) | code + invariant audit | tss_solver/tss_verify split | MEASURED/AUDITED |
| C10 | 14/14 + 5/5 official gate at 9b tip | measurement | GATE.md + final-matrix-19-9b.log @ dba6111d | MEASURED |
| C11 | 0l full solve 177.7s beats reference pdspn 264s | measurement | gate log + pdspn run log (re-pin commit) | MEASURED (pdspn side needs re-pin) |
| C12 | Round-8b→9b cumulative speedups (~40x matrix etc.) | measurement | .codex-round9/round9-progress.md ledger | MEASURED |
| C13 | VCF corpus never exercises k<B (dispatch-only) | structural argument + counter-check | RZOP_SOLVER_OPTIMIZATION.md §5.1 + spare-corpus witness columns | IN-FLIGHT (G2R1) |
| C14 | Spare-turn corpus ground truth | reference-solver runs | tss_reference matched-horizon logs (G2R1) | IN-FLIGHT |
| C15 | Zone-integrated engine results at unforced AND nodes | measurement | Group-2 rounds ≥2 | PLANNED |
| C16 | Uniform-vs-exact clock size delta | measurement | §5.3 comparison harness | PLANNED |
| C17 | Bounded-completeness tier (a) binding | Lean + Rust differential | SolverInterface + WideTurnGate fixture | PLANNED |
| C18 | Atlas opening verdicts | certified solves | living doc snapshots + certificates | PLANNED |
| C19 | Open problems honestly labeled | doc | §12 map (ES Prop 2, Domination MV, Potential tail, F+H_W) | STANDING |
| C20 | Reference-solver capstone columns (idtt/dfpn/pdspn, both corpora) | measurement | matched-budget runs (commands TBD, commit-pinned) | PLANNED |

Status legend: PROVEN-LEAN (kernel-checked, sorry-free) · PROVEN-DOC
(hostile-round proof doc, not yet mechanized) · MEASURED (logged, commit +
command) · IN-FLIGHT (session running) · PLANNED · STANDING.

---

## Figures & demonstrations (owner-required; work DEFERRED to writing phase)

The document must build visual intuition the way the zone-theory companion
does (hexfig.sty TikZ boards). Candidate list — pick per-section at writing
time, not now:

- §3: one annotated running-example board introducing windows/threats/`b`
  (reused everywhere after).
- §4: the four ranked-zone components colored on the same board (`Z_dir` /
  `Z_seed` / `Z_touch` / `Z_virgin`); a seed-band radius `8(r−1)` diagram;
  the sharp virgin `E = 7` radius-8 position; kernel `K_b` vs full hitting
  universe side-by-side; a substitution-envelope (T9) before/after pair.
- §5: finder/verifier dataflow diagram (single-mint path highlighted);
  certificate tree vs DAG (T10/U18) sharing figure.
- §6: per-round waterfall of the 8b→9b wall-time drops (log scale).
- §7: `|searched|/|Legal|` distribution plot over AND nodes; capstone table
  itself rendered as heat-shaded grid.
- §8: a λ² position with the quiet connector move highlighted vs the two
  threats it joins; forced-vs-unforced AND node comparison on-board.
- §9: pairing k=7 index-12 witness tile figure (exists in v3 — inherit);
  each "genuinely new fact" gets its sharp position drawn.
- §10: atlas coverage map — opening families × depth, cells colored by
  certified verdict.

Rule: every figure must teach one specific claim from the map; no
decoration.

## Deliberately out of scope

Version history of the companion (v1/v2/v3 — the flagship never mentions
them); trainer/NN integration (Phase-3, separate write-up if ever);
anything whose evidence row above cannot resolve.
