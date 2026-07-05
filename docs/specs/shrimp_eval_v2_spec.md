# Shrimp — multi-stage strength evaluation (eval-v2) specification

Status: **design spec for the statistically-sound, multi-stage strength evaluation of the Shrimp
lineage.** It is implemented in `packages/shrimp/python/shrimp/{eval_stats,eval_arena,multistage_eval}.py`;
this document is the rationale and contract. It supersedes a naive per-epoch 16-game vs-prior arena
for the purpose of *measuring* strength. An adversarial statistician reviewed the naive first-cut
design; what follows is the **corrected** design (converged pairing-aware Bradley-Terry, family-wise
error control via a single primary hypothesis, honest significance accounting). The naive version is
recorded only where needed to say why it was rejected.

> **THIS FEATURE IS PURELY EVALUATIVE.** It measures strength and emits a verdict **label**
> (`PROMOTE` / `REGRESS` / `INCONCLUSIVE`). It MUST NEVER gate, promote, halt, or otherwise alter
> the training run. Gating/promotion hooks may *exist* in code, but they default **OFF**
> (`eval_gating_enabled=False`, `eval_promotion_enabled=False`) and are wired to nothing that
> changes the run. Evaluation is purely observational and never touches the run.
> See §8 (the load-bearing "purely-eval" contract).

Working name: **eval-v2** (the strength-measurement harness for Shrimp).

Conventions: a *pair* = two paired games on a shared opening with the seats swapped (common random
numbers; see §4.3). `N_pairs` = number of pairs; a "128-game eval" is `N_pairs = 64`. `r_X` =
Bradley-Terry log-strength of player X (SealBot pinned `r_SealBot = 0`). Elo = `r * 400 / ln 10`.
"Candidate" / "L" = the epoch under evaluation; "B" = the prior champion (the registered primary
comparator). All file:line citations are against the repo at authoring time.

---

## 0. Thesis

The current eval answers the wrong question with too few games. It plays 16 greedy games against
**whatever checkpoint happens to be one epoch back**, reports a raw win rate, and stops. Two
problems compound:

1. **The opponent is a moving target.** "vs immediately-prior" measures the *delta between two
   adjacent, highly-correlated checkpoints*, not progress against a fixed reference. Two adjacent
   epochs of a thin-data RL run are nearly the same player, so the signal is a coin flip by
   construction.
2. **16 games cannot resolve the moves that matter.** Both real production evals straddle 0.5 —
   ep5 vs ep4 = 9-7 (0.5625), ep10 vs ep9 = 6-10 (0.375) (shrimp-goal memory; the run was in
   fact frozen at ep10 by an unrelated optimizer-device bug, but even healthy the 16-game arena is
   uninformative). A 16-game binomial has a 95% Wilson half-width of ~0.24 around 0.5 — it cannot
   distinguish "improved 80 Elo" from "regressed 80 Elo."

eval-v2 fixes both: a **fixed, multi-role opponent set** (a cross-lineage zero-point, permanent
anchors, and a sliding skill bracket), **paired games with shared openings** for variance
reduction, and a **rolling Bradley-Terry rating** that compounds every epoch's games into one
persistent, SealBot-pinned scale. It is a **standalone runner** that never runs inside the
training pipeline, so training is never interrupted.

The single most important discipline in this doc is **not overclaiming significance** (§5).
The PRIMARY candidate-vs-champion verdict is **permanently single-epoch-limited** — a fresh
candidate node enters the pool each epoch, so its rating never compounds — with a single-epoch
**`SE(r_L − r_B) ≈ 120–140 Elo`** (paired → effective N well below decided, plus the two-rating
√2), resolving only ~250–300 Elo: a **gross-regression tripwire, not a fine-edge test**, and it
does NOT tighten with more epochs. The ~15–20 Elo resolution everyone wants is the
**multi-epoch rolling asymptote of the FIXED-anchor DESCRIPTIVE curve** (bc_prefit/ep5/SealBot —
the same labels every epoch, so those edges pool and tighten); it describes the lineage progress
curve, never the single-epoch verdict. SealBot is a down-weighted (0.5) descriptive zero-point,
never the verdict. The code and this doc say so plainly.

---

## 1. The opponent set (three roles)

Every edge played feeds the rating pool (§4.4 / Stage D). But the opponents play **distinct
roles**, and only one of them backs the primary verdict.

### 1.1 SealBot — the cross-lineage zero-point / calibrator

SealBot (the external C++ minimax baseline, `packages/hexo_runner/python/hexo_runner/adapters/sealbot.py`)
is pinned at **0 Elo**. It is a fixed external reference, so it is the **stable anchor** that keeps
the Shrimp Elo scale comparable across runs and over time.

**Critical caveat (anchor drift).** SealBot is a fixed-wall minimax (50 ms / decision,
`DEFAULT_SEALBOT_TIME_LIMIT = 0.05`, sealbot.py:42). Its effective search **depth varies under
machine load** — when the box is busy, 50 ms buys fewer nodes, so SealBot is non-deterministically
*weaker*. This means:

- SealBot is the **zero-point only**. Its edges pin the scale's origin; they MUST NOT enter the
  *difference* inference (candidate-vs-champion) at full weight (§5.5). It is calibration, not a
  yardstick for fine deltas.
- All SealBot win rates are reported **descriptively** with an explicit "depth-varies-under-load;
  non-stationary opponent" footnote. Hexo has **no draws** (binomial base; see §5, item 6), so
  SealBot results are binomial, but the *over-dispersion* from depth variation is acknowledged and
  the edge is down-weighted in the BT likelihood (§4.4).

### 1.2 Permanent anchors — the fixed yardstick (never slide)

Two checkpoints are **permanent** anchors, evaluated **every epoch forever**:

- **BC prefit**: the behavioral-cloning warm-start checkpoint the RL run began from. "Has RL
  improved on the starting point at all?" is answered against this anchor.
- **ep5**: an early (epoch-5) checkpoint of the run — an early-but-post-warm-start fixed reference.
  (The exact early anchor epoch is a §9 decision point; ep5 is the default because it is an early
  epoch the arena evaluated.)

Permanent anchors are **fixed players in the pool**: their rating is pinned by their entire
accumulated history of games, so they give the rolling scale two stable interior reference points
besides SealBot. **They never slide** — that is the whole point. The progress curve "candidate Elo
vs the BC-prefit anchor, over epochs" is the headline plot.

### 1.3 Sliding bracket — nearest neighbours on a fixed log grid

To keep games informative as the candidate gets stronger (an anchor 100 epochs back is a
blowout — near-zero information per game), the candidate also plays the **nearest two checkpoints
below it on a fixed log-spaced grid**:

```
GRID = {5, 10, 20, 40, 80, 160}     # epochs; fixed for the whole run
bracket(epoch) = the two largest grid points strictly below `epoch`
```

Examples: at epoch 33 → bracket = {20, 10}; at epoch 12 → {10, 5}; at epoch 7 → {5} (only one grid
point below; ep5 also happens to be a permanent anchor, so the bracket adds nothing new and is
skipped — no double-count). The grid is **fixed and global**, so a given grid checkpoint (say
ep20) is the *same player* whenever it is in some later epoch's bracket — its games **accumulate**
in the pool across all the epochs that bracket it. This is the corrected design: the bracket is
**nearest grid neighbours**, explicitly **NOT "vs immediately-prior epoch"** (the rejected naive
choice that made the current eval uninformative, §0).

Roles summary:

| Opponent | Role | Slides? | In primary verdict? | Reported |
|---|---|---|---|---|
| SealBot | cross-lineage zero-point | n/a (pinned 0 Elo) | **No** (zero-point only, down-weighted) | descriptive + load caveat |
| BC prefit (`checkpoint_epoch2.pt`) | permanent anchor | never | No (descriptive) | progress curve + Wilson/Elo CI |
| ep5 anchor | permanent anchor | never | No (descriptive) | Wilson/Elo CI |
| prior champion **B** | **primary comparator** | n/a (= last promoted/registered champion) | **YES (the one primary hypothesis)** | BT difference-CI w/ Cov_LB |
| bracket (2 of `{5,10,20,40,80,160}`) | sliding skill bracket | yes | No (descriptive) | Wilson/Elo CI |

Note the **prior champion B** is a distinct registered player (the last epoch that earned the
`PROMOTE` label, or the warm-start at run start), NOT necessarily a member of the grid or the
anchors. It is the *only* opponent whose edge backs a significance verdict (§5.3).

---

## 2. The four stages

Evaluation is staged so cheap checks gate expensive ones (cheap-first), and so the expensive deep
eval only runs when it can be informative. **All four stages are pure measurement** — "gate" below
always means "decide whether the eval harness spends more eval compute," never "gate the training
run."

### Stage A — bridge / smoke

A fast correctness bridge, not a strength test. Confirms the candidate checkpoint loads, the
multi-root session runs, SealBot is reachable, and a handful of paired games complete without
error (no truncation storm, legal moves, records writable). Tiny (e.g. 2 pairs at low visits).
Purpose: fail fast on a broken checkpoint or environment before committing to Stage C. Emits no
verdict.

### Stage B — SPRT screen vs the prior champion (a GROSS-regression triage ONLY)

A sequential probability ratio test (Wald SPRT) of the candidate vs the prior champion **B**,
used **only to triage gross regressions** — it is a cheap "is this obviously broken?" filter, not
a calibrated promotion test.

**Honest accounting (do NOT advertise this as a 5%/5% test).** A standard SPRT with
`alpha = beta = 0.05` and indifference bounds near the elo region of interest has an **expected
sample size of ~285 games near the indifference region** (where the LLR drifts slowly). The runner
caps the SPRT at a small budget (e.g. `sprt_max_pairs`); because the cap is far below ~285, a
candidate that is genuinely near-equal to B will usually **hit the cap and return `escalate`**
rather than `accept`/`reject`. That is *intended*: Stage B is designed to catch the candidate that
is *clearly* much worse (the LLR crosses the lower bound fast) and otherwise hand off to Stage C.
The doc and code must call this an **escalation-biased triage**, never "the calibrated test."

Outcomes:
- `reject` (LLR crosses lower bound): candidate is grossly worse than B → label trends `REGRESS`;
  Stage C may be skipped or run at reduced budget (a §9 decision; default: still run C for the
  rating-pool contribution, since C games are never wasted — they feed the pool).
- `accept` (rare at the small cap): candidate is clearly better than B.
- `escalate` (the common outcome): inconclusive within the cap → Stage C decides.

SPRT statistics use **pairs as the unit** (§4.3), not individual games, so the correlation between
paired games does not inflate the LLR.

### Stage C — the 128-game deep eval (paired, pentanomial, Wilson)

The core measurement: **128 games (64 pairs)** against the full opponent set — SealBot + the two
permanent anchors + the prior champion B + the bracket. Played concurrently (§6). Each opponent
gets its share of the 64 pairs (the per-opponent split is a §9 knob; a reasonable default is to
prioritize B and the anchors, with the bracket and SealBot getting smaller shares since SealBot is
zero-point-only and the bracket is descriptive).

Per opponent, Stage C computes:
- **Pair-level win rate** with a **pair-level standard error** (`N_pairs` units, §4.3.1) — never
  `N=128` independent Bernoulli.
- **Pentanomial scoring** of pairs (§4.3.2): each pair scores in {0, 0.5, 1, 1.5, 2} points; the
  pentanomial variance is the variance-reduced estimator that exploits the shared opening.
- **Wilson CI** on the pair-level rate (the `wilson_ci` helper in
  `shrimp/eval_stats.py`), reported per opponent.

Stage C is where the **128 games/epoch** are produced; they are then folded into Stage D's rolling
pool so they **compound**.

### Stage D — rolling, SealBot-pinned Bradley-Terry rating

Every edge ever played (this epoch's 128 games plus all prior epochs' games, plus the cached
cross-lineage SealBot edges) is pooled into **one Bradley-Terry rating** with SealBot pinned at 0.
The pool is **persisted across epochs** (default `diagnostics/eval_pool.json`) so the per-epoch 128
games **accumulate** — a grid checkpoint that is bracketed by several epochs ends up with hundreds
of games behind its rating, and the candidate-vs-champion difference-CI tightens epoch over epoch
toward the multi-epoch asymptote (§5.4).

The BT fit (§4.4) is a converged, pairing-aware fit — **not** a naive
non-converging fixed-step gradient descent (§4.1). It yields:
- a full rating vector (Elo per player, SealBot = 0),
- **full-Hessian covariance** → credible intervals per rating AND, critically, the **difference**
  credible interval `r_L − r_B` using the off-diagonal `Cov_LB` term (this is the primary-verdict
  statistic, §5.3).

---

## 3. The verdict label (and what it is NOT)

eval-v2 emits exactly one of three labels per epoch:

- **`PROMOTE`** — the candidate L is better than the prior champion B at the registered
  significance level, by the **one pre-registered primary hypothesis**: the BT difference credible
  interval for `r_L − r_B` (with the `Cov_LB` term) lies entirely above 0 (§5.3).
- **`REGRESS`** — the difference CI lies entirely below 0 (or Stage B `reject` fired).
- **`INCONCLUSIVE`** — the difference CI straddles 0 (the common single-epoch outcome, §5.4).

**The label is reported, not enforced.** It is written to diagnostics and shown on the dashboard;
it does **not** move the checkpoint pointer, halt the run, or change any hyperparameter (§8). A
`PROMOTE` label *updates the registered champion B for the next epoch's primary hypothesis* (a
bookkeeping update inside the eval pool, so the next epoch compares against the right reference) —
this is the only state the label changes, and it changes nothing about training.

---

## 4. Corrected statistics (the load-bearing fixes)

These are the fixes the adversarial statistician required. Get them right; they are the reason this
doc exists.

### 4.1 Bradley-Terry must CONVERGE (do not use fixed-step GD)

A naive BT fit by **fixed-step gradient descent** (e.g. 500 iters, step 0.002) **does not
converge** — at the optimum `max|grad|` is ~0.30, nowhere near zero, so the ratings are biased and
any Fisher-information SEs computed from them are meaningless. **Do not do this.**

eval-v2 fits BT by **Newton's method** (or `scipy.optimize.minimize` with a gradient tolerance) on
the pinned-SealBot log-likelihood, and **asserts `max|grad| < 1e-6` before computing covariance**.
If the assertion fails (e.g. a player with all-wins/all-losses → separable likelihood, rating runs
to ±∞), the harness reports the non-convergence explicitly and falls back to a ridge-regularized
fit rather than emitting a bogus CI. The covariance is `H⁻¹` from the **full Hessian** at the
converged point (with SealBot's pinned coordinate removed from the system), not the diagonal-only
Fisher approximation the legacy script used (which wrongly treats anchors as fixed and drops all
the off-diagonal coupling the difference-CI needs).

### 4.2 Likelihood, gradient, Hessian (reference forms)

For an edge between i and j with `w_ij` wins for i and `w_ji` for j, `p_ij = σ(r_i − r_j)`:

```
NLL  = − Σ_edges [ w_ij·ln p_ij + w_ji·ln(1 − p_ij) ]
∂NLL/∂r_i = Σ_j (w_ij + w_ji)·p_ij − w_ij                      # = expected − observed wins
H_ii = Σ_j n_ij · p_ij·(1 − p_ij)                              # n_ij = w_ij + w_ji
H_ij = − n_ij · p_ij·(1 − p_ij)        (i ≠ j)                  # off-diagonal coupling
```

SealBot's row/column is removed before inverting (it is pinned at 0). Newton step:
`r ← r − H⁻¹ g` on the free (non-SealBot) coordinates, iterate to `max|g| < 1e-6`.
**Pairing/over-dispersion enters via the effective counts fed into `n_ij`/`w_ij` — see §4.3.3.**

### 4.3 Pairing is real correlation (the anti-conservative trap)

Paired games (same opening, seats swapped) are **NOT independent Bernoulli**. Treating 128 paired
games as 128 independent `(w, l)` draws **understates the variance** → anti-conservative CIs (too
narrow), which would manufacture false `PROMOTE`/`REGRESS` labels. Three places this is handled:

#### 4.3.1 Pair-level SE for win rates

A win rate over `N_pairs` pairs uses the **pair as the unit**: estimate the per-pair point score
(0/0.5/1 per game → pair total in {0,0.5,1,1.5,2}, i.e. {0,0.25,0.5,0.75,1} normalized) and take
the SE from the **sample variance of the per-pair scores divided by `N_pairs`**, not from
`p(1−p)/128`. With 64 pairs the effective n is 64, not 128 — the CI is wider, honestly.

#### 4.3.2 Pentanomial scoring

Each pair lands in one of five outcomes by (gameA, gameB) result: {LL, LW/WL split…} → normalized
pair score s ∈ {0, 0.25, 0.5, 0.75, 1}. The **pentanomial sample variance** of `s` is the
variance-reduced estimator: shared-opening common-random-numbers cancel the opening's contribution
to variance, so a +/- result on a balanced opening is more informative than two unpaired games.
This is the estimator the win-rate SE (§4.3.1) and the CI report use. (Hexo has no draws, so the
usual draw-bearing pentanomial cells collapse; the five cells come from the two seats' two binary
outcomes, with split pairs {WL, LW} as the informative middle — see §5 item 6 on the "trinomial as
a conservative device" for split pairs.)

#### 4.3.3 Paired/effective counts into the BT likelihood

The BT likelihood (§4.2) must **not** be fed 128 independent `(w, l)`. Feed it either:
- **paired/effective counts**: scale each edge's `n_ij` by the ratio of the pentanomial pair
  variance to the naive binomial variance (an **effective-sample-size deflation**, `n_eff ≤ n`), so
  the Hessian (information) reflects the true, correlation-reduced information; OR
- a **sandwich / over-dispersion correction**: fit BT on raw counts but inflate the covariance by a
  clustered (per-pair) sandwich estimator (`H⁻¹ · B · H⁻¹` with the per-pair score gradients), which
  is robust to the within-pair correlation.

Either way the rule is absolute: **never feed 128 independent (w, l) into the BT fit.** The default
is the effective-count deflation (simpler, keeps the Hessian PSD); the sandwich is the documented
alternative for the SealBot edge (whose over-dispersion is depth-driven, §5.5).

### 4.4 The rolling pool fit

Stage D loads `eval_pool.json` (all historical edges as effective counts), adds this epoch's edges,
refits BT to convergence (§4.1), writes the updated pool back, and reports the rating vector + the
difference-CI for the primary hypothesis. SealBot edges enter **down-weighted** (over-dispersion
factor, §5.5) so their non-stationarity does not dominate the interior geometry. Permanent anchors
and grid checkpoints accumulate games across epochs, so their ratings stabilize and the
candidate-vs-champion difference-CI tightens toward the asymptote (§5.4).

---

## 5. Honest significance (no overclaiming)

This section is reproduced near-verbatim as a code comment block in the runner and is the
discipline the whole feature turns on.

1. **One pre-registered PRIMARY hypothesis per verdict.** The verdict is candidate **L** vs prior
   champion **B**, tested by the BT difference credible interval for `r_L − r_B` **using the
   `Cov_LB` off-diagonal term** (§5.3). Exactly one test → no multiplicity inflation on the
   verdict.

2. **All other opponent edges are DESCRIPTIVE.** SealBot, the permanent anchors, and the bracket
   get reported Wilson/Elo CIs and feed the rating, but **carry NO significance verdict**. We do
   not attach "significant" to any of them. If a future requirement forces several edges to gate,
   apply **Bonferroni** (per-edge `alpha = 0.05 / k`) — but the default is the single primary test.

3. **The candidate-vs-champion verdict is PERMANENTLY single-epoch-limited (~250–300 Elo
   resolution), not ~100–120 and never ~15–20.** A FRESH candidate node (`cand_epN`) enters the
   pool each epoch, so its rating carries only THIS epoch's games — it never accumulates across
   epochs the way a fixed anchor does. The single-epoch standard error of the rating difference is
   **`SE(r_L − r_B) ≈ 120–140 Elo`** (paired → effective N well below decided, plus the two-rating
   √2), so a 95% CI half-width is ~240–270 Elo and the smallest move a single epoch can call
   significant is **~250–300 Elo**. The PRIMARY verdict is a **gross-regression tripwire, not a
   fine-edge test**, and this does **not** improve with more epochs.

4. **~15–20 Elo is the MULTI-EPOCH asymptote of the FIXED-ANCHOR DESCRIPTIVE curve — NOT of the
   verdict.** The bc_prefit / ep5 / SealBot anchors are the SAME labels every epoch, so THEIR edges
   pool across epochs and THEIR ratings tighten roughly as `1/√(epochs)` toward ~15–20 Elo. That
   asymptote describes the LINEAGE progress curve (where each candidate sits on a stable scale),
   **never** the single-epoch candidate-vs-champion verdict (which cannot compound — item 3). **Do
   not report a single epoch's verdict as if it resolved 15–20 Elo, and do not claim the verdict CI
   tightens with epochs — only the descriptive anchor curve does.**

5. **Never write "statistically significant" without the test that licenses it.** Only the primary
   difference-CI excluding 0 licenses a significance claim, and only about L-vs-B. Descriptive
   numbers are reported as point estimates with CIs and nothing more.

6. **SealBot's non-stationary edge stays out of the difference inference at full weight.** SealBot
   depth varies under load (§1.1), so its edge is over-dispersed; it enters the BT fit only
   down-weighted (§5.5) and is **never** the primary comparator. **Hexo has no draws** — the base
   model is binomial. "Trinomial"/pentanomial language appears **only as a conservative device for
   split pairs** (a {WL, LW} pair is the informative middle outcome), not because the game has
   draws.

### 5.3 The primary-hypothesis statistic (precise)

```
Δ      = r_L − r_B                              # BT log-strength difference
Var(Δ) = Cov_LL + Cov_BB − 2·Cov_LB            # from the full-Hessian inverse (§4.1)
CI95   = Δ ± 1.96·√Var(Δ)        (in log-strength; ×400/ln10 for Elo)
PROMOTE  if CI95 lower bound > 0
REGRESS  if CI95 upper bound < 0
INCONCLUSIVE otherwise
```

The `Cov_LB` term is exactly what the legacy diagonal-only SE dropped, and it is essential: L and B
are coupled through their shared opponents in the pool, so `Var(Δ)` is **not** `Var(r_L) + Var(r_B)`.

### 5.4 Why a single epoch is usually INCONCLUSIVE (and that is correct)

A genuine 30–50 Elo per-epoch improvement is **far below** the single-epoch ~250–300 Elo verdict
resolution (item 3), so the honest single-epoch label is `INCONCLUSIVE` — and that is the *correct*
answer, not a failure of the harness, nor something more epochs fix (the candidate node never
compounds). The progress signal is the **rolling FIXED-ANCHOR DESCRIPTIVE curve** — where each
candidate sits on the stable bc_prefit/ep5/SealBot scale — which is exactly why Stage D persists
and pools those anchor edges. Reporting a noisy per-epoch `PROMOTE`/`REGRESS` off one epoch's games
would be the overclaiming this design exists to prevent.

---

## 6. Throughput plan (concurrent multi-root; never interrupts training)

eval-v2 is a **standalone orchestrator** (`shrimp/multistage_eval.py`). It **does not block the
training epoch** — it reads checkpoints read-only out of the run dir and runs on whatever GPU/CPU
budget the operator gives it, off to the side.

Games run **concurrently** via the already-existing multi-root search API — **no Rust change**:

- **The multi-root API already exists.** `ShrimpMctsSession.search` takes
  `game_keys: Vec<u64>` + a list of `states` and searches them all in one call with cross-game
  leaf batching (`packages/shrimp/rust/src/search.rs:512-544`; it already validates
  `roots.len() == game_keys.len()` and caps at `active_root_limit`). The current `_play_pair`
  (evaluation.py:23-66) calls it with single-element `[seed]` / `(state,)` lists — eval-v2 simply
  passes **all in-flight games at once**.

- **shrimp-vs-checkpoint reuses `_play_pair`'s machinery.** `_play_pair` (evaluation.py:23)
  already accepts **per-seat** `divergence_overrides` / `divergence_overrides_b`, so a paired,
  seat-swapped game with the candidate's and opponent's search configs is already expressible. The
  concurrent runner generalizes its single-game loop to a batched loop (mirroring the dense
  concurrent eval's round structure) so all pairs against checkpoint opponents advance together.

- **shrimp-vs-SealBot is a concurrent round loop.** Each round, the positions where the **model**
  is to move are searched in one batched `ShrimpMctsSession.search([...keys], [...states])` call,
  while **SealBot's** moves are drained **serially per game** via the SealBot adapter
  (`SealBotPlayer.decide`, `sealbot.py`; one isolated worker per game). SealBot's fixed 50 ms
  minimax is independent per game and is not the bottleneck (the batched model forward is); §1.1's
  load-caveat is the price of that serial drain.

This gives the 128 games/epoch at the concurrency the GPU can feed, entirely off the training
critical path.

### 6.1 Common random numbers (paired openings)

A *pair* shares one opening line (same seed-derived opening) and plays it **twice with seats
swapped**. The opening is temperature-sampled once per pair (as `_play_pair` already does for its
first `opening_plies`, evaluation.py:46-48) and **reused** for both games of the pair, so the only
difference between the two games is which seat the candidate holds. This is the common-random-number
variance reduction that the pentanomial estimator (§4.3.2) exploits. The per-pair opening seeds are
disjoint across opponents and epochs (a seed-stride scheme).

---

## 7. Implementation file map

All eval-v2 code is **eval-only** (no edits to the training path, no Rust changes, no `.so`
rebuild). It lives in the `shrimp` package:

| File | Contents |
|---|---|
| `docs/specs/shrimp_eval_v2_spec.md` | this spec |
| `packages/shrimp/python/shrimp/eval_stats.py` | pure-numpy statistics core: converged BT (Newton/scipy + `max|grad|<1e-6` assert + full-Hessian covariance), pair-level SE, pentanomial summary, Wilson CI, SPRT(pairs), `var_diff` difference-CI with `Cov_LB`, over-dispersion weighting, effective counts, expected-SE-Elo. **Pure CPU/numpy (+optional scipy) → unit-testable without GPU.** |
| `packages/shrimp/python/shrimp/eval_arena.py` | game-running layer: CRN-paired seat-swapped checkpoint matches and the concurrent multi-root SealBot loop. Returns the arena result-dict shape + a pentanomial block. |
| `packages/shrimp/python/shrimp/multistage_eval.py` | the multi-stage orchestrator (Stages A–D): opponent-set construction (SealBot config, permanent anchors, sliding bracket on `{5,10,20,40,80,160}`, champion registry), the rolling BT pool, and the verdict. |
| `tests/test_shrimp_eval_stats.py`, `test_shrimp_eval_arena.py`, `test_shrimp_eval_orchestrator.py` | numpy-only unit tests: BT convergence (`max|grad|<1e-6` on a known-answer ladder), pairing widens CI vs naive binomial, pentanomial variance ≤ binomial, Wilson parity, SPRT decision boundaries, difference-CI `Cov_LB` sign, no-write purely-eval sentinel. |
| `diagnostics/eval_pool.json` (per run dir) | persisted rolling BT pool (all historical edges as effective counts + champion registry). |

Reuse (read-only):
- `packages/shrimp/python/shrimp/evaluation.py` — the per-pair play + opening-sampling
  discipline.
- `packages/hexo_runner/python/hexo_runner/adapters/sealbot.py` (+ `_sealbot_worker.py`, hexo_runner
  match mode) — the SealBot adapter.
- `packages/shrimp/rust/src/search.rs` — the multi-root `search` API (used, not modified).

---

## 8. PURELY EVAL — the contract (gating/promotion present-but-disabled)

This feature **measures**. It does not control the run. Stated as hard invariants:

1. **No write to the training run.** eval-v2 never moves the checkpoint pointer, never writes a
   `supervisor_halted.flag`, never edits a config, never signals the supervisor. It reads
   checkpoints read-only and writes only its own diagnostics + `eval_pool.json`.

2. **The verdict is a label.** `PROMOTE`/`REGRESS`/`INCONCLUSIVE` is reported to diagnostics/the
   dashboard. The only state a `PROMOTE` updates is the eval pool's *champion registry* (so the
   next epoch's primary hypothesis compares against the right B) — pure eval bookkeeping, invisible
   to training.

3. **Gating/promotion hooks exist but default OFF and are wired to nothing.** Two config flags are
   added to `EvaluationSection` (`packages/shrimp/python/shrimp/config.py:76`,
   the `EvaluationSection` dataclass):

   ```python
   eval_gating_enabled: bool = False      # if True, a REGRESS label COULD halt — but the
                                          # wiring target is intentionally absent; the flag
                                          # only toggles a no-op branch in the standalone runner.
   eval_promotion_enabled: bool = False   # if True, a PROMOTE label COULD advance the pointer —
                                          # likewise wired to nothing in the training path.
   ```

   These are present so the *capability* is designed-in and reviewable, but **both default
   `False`**, and even when flipped they only enable a branch **inside the standalone eval runner**
   that has **no edge to the training pipeline**. Turning them on requires a separate, explicit,
   reviewed wiring change that this feature deliberately does not make.

4. **The pipeline eval is untouched.** `shrimp/evaluation.py:evaluate_epoch` keeps running its
   existing (now-acknowledged-uninformative) 16-game arena for backward dashboard compatibility, OR
   the operator simply runs eval-v2 standalone; either way eval-v2 adds no work to the training
   epoch.

The litmus test for any future change to this feature: *if it can change what the training run
does, it is out of scope.*

---

## 9. Decision points (owner)

Resolved-by-default below; flagged for owner confirmation.

- **9.1 Early permanent anchor epoch.** Default **ep5** (first epoch the current arena evaluated).
  Alternatives: ep2 (== BC warm-start, redundant), ep10. → default ep5.
- **9.2 Stage-C per-opponent pair split** of the 64 pairs. Default: prioritize **B** and the two
  permanent anchors; smaller shares to the bracket and SealBot (SealBot is zero-point-only). Exact
  split is a tuning knob.
- **9.3 SealBot over-dispersion factor** (the down-weight in §5.5). Default: estimate empirically
  from the variance of repeated SealBot evals at fixed strength; conservative placeholder until
  measured.
- **9.4 Stage-B SPRT bounds + cap.** Default `alpha = beta = 0.05`, indifference region pre-set to
  the gross-regression scale (wide), `sprt_max_pairs` small (escalation-biased by design, §2/B).
- **9.5 On Stage-B `reject`, skip Stage C?** Default **no** — still run C, because C games feed the
  rolling pool and are never wasted; the `REGRESS` label is already implied.
- **9.6 Pairing correction method** (§4.3.3). Default **effective-count deflation**; sandwich
  reserved for the SealBot edge.

(References to "§5.5" above denote the SealBot down-weight rule stated inline in §1.1 / §4.4 /
§5 item 6; it is the single over-dispersion factor applied to SealBot edges in the BT likelihood.)

---

## 10. Acceptance (eval-correctness gates)

Pure-CPU/numpy unit tests (`tests/test_shrimp_eval_stats.py`), runnable without GPU:

1. **BT converges:** on a synthetic ladder with a known closed-form answer, the fit reaches
   `max|grad| < 1e-6` and recovers the planted ratings; the legacy GD on the same data is shown to
   stall at `max|grad| ~ 0.3` (regression guard against re-inheriting it).
2. **Pairing widens the CI:** the pair-level/pentanomial SE on correlated paired data is
   **strictly wider** than the naive `p(1−p)/128` binomial SE → no anti-conservatism.
3. **Pentanomial ≤ binomial variance:** the pentanomial pair-variance is ≤ the unpaired binomial
   variance on the same outcomes (variance reduction holds).
4. **Difference-CI uses `Cov_LB`:** `Var(r_L − r_B)` differs from `Var(r_L) + Var(r_B)` by exactly
   `−2·Cov_LB`, and the sign/coupling is correct on a 3-player pool.
5. **Wilson parity:** the `wilson_ci` helper matches a reference Wilson-interval computation to
   machine precision.
6. **SPRT boundaries + escalation bias:** the LLR crosses the lower bound fast for a clearly-worse
   candidate and **hits the cap → `escalate`** near indifference (the documented behavior, not a
   calibrated accept/reject).
7. **Purely-eval invariants:** with `eval_gating_enabled=eval_promotion_enabled=False`, the runner
   performs no write outside its own diagnostics/`eval_pool.json` (asserted via a no-write sentinel
   on the run dir).

GPU game-play (Stages A/C concurrency) is validated by the operator off the training critical path;
the statistics — the part that was wrong before — are fully unit-tested on CPU.

---

## 11. Summary of the corrections (naive → corrected)

| # | Naive design | Corrected design (this spec) |
|---|---|---|
| 1 | "vs immediately-prior epoch" | fixed multi-role set: SealBot zero-point + permanent anchors (BC, ep5) + sliding nearest-2 on `{5,10,20,40,80,160}` (§1) |
| 2 | one 16-game arena, raw win rate | 4 stages: smoke → SPRT-triage → 128-game paired deep eval → rolling BT pool (§2) |
| 3 | BT by fixed-step GD (no convergence, `max|grad|~0.3`) | Newton/scipy to `max|grad|<1e-6`, assert before covariance, full-Hessian `H⁻¹` (§4.1) |
| 4 | 128 independent `(w,l)` into the likelihood/CI | pair-level SE + pentanomial + effective/sandwich counts; never 128 independent (§4.3) |
| 5 | implicit multiple comparisons | one pre-registered primary hypothesis (L vs B, `Cov_LB` difference-CI); all else descriptive; Bonferroni if forced (§5 items 1-2) |
| 6 | "resolves ~15-20 Elo per epoch" | verdict PERMANENTLY single-epoch-limited ~250-300 Elo (`SE(r_L−r_B)≈120-140 Elo`, candidate node never compounds); ~15-20 Elo is the multi-epoch asymptote of the FIXED-ANCHOR DESCRIPTIVE curve only, never the verdict (§5 items 3-4) |
| 7 | SealBot as a yardstick | SealBot zero-point only, down-weighted (depth varies under load); no draws (binomial base) (§1.1, §5 items 5-6) |
| 8 | eval blocks/gates the run | PURELY EVAL: label only; gating/promotion present-but-OFF, wired to nothing (§8) |
| 9 | eval inside the training epoch | standalone runner, concurrent multi-root (no Rust change), never interrupts training (§6) |
