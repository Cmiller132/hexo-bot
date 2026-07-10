# SPEC: Ray-tap convolution + graded multiplicity planes (hexfield_eq) — v3

Status: ACCEPTED — implementation-ready. Targets the next full run. No
compatibility with existing checkpoints is provided; the live main_1 soak
must remain bit-identical (see §9.1 for how that is guaranteed).

Date: 2026-07-09 (v3). Baseline: branch `main_9-fastrow-strip`, package
`packages/hexfield_eq/`, as described in `docs/quotient_reps/CONTEXT.md`.
Where this document and the code disagree, the code wins — report the
discrepancy, do not silently improvise.

This document specifies mechanism only. It makes no claims about expected
strength, quality, or performance benefit.

v3 changes from v2 (final review):
- **K2 added (§2.4, blocking)**: custom-autograd pre-aggregation for the
  training path; naive implementation saves ~7.2 GB of extra activations
  at `both` and would OOM / force a confounding batch cut. New test T8.
- Featurizer plane map moved behind an import-time env gate
  (`HEXFIELD_EQ_FEATURE_VERSION`, default 1) so landing the code cannot
  perturb the live run (§1.1, §4, §9.1).
- §2.5: train-side raylen citation added (no un-gating needed there).
- §6.3: visit-budgeted (not time-budgeted) matches made explicit; arms
  priority-ranked into waves; full arch-env pinning per arm required.
- K1 gate fallback policy pre-agreed (§2.4) so a bench miss cannot stall
  Phase L.
- §9 rewritten as an implementation plan: work items with file targets,
  acceptance gates, sequencing, and process constraints.

v2 changes from v1 (prior review): fused-kernel claim corrected (K1);
tap→raylen-slot LUT + raylen un-gating specified; per-orbit-channel α;
default `both`; live5 planes; subtraction arms A5/A6; discrimination
gate P1′; state-dict threading tightened.

---

## 0. Summary of changes

| # | Change | Surface |
|---|--------|---------|
| 1 | Featurizer: 25 → 46 input planes (10 axis quantities instead of 4; 3 new scalar planes; fork planes re-indexed), gated by `HEXFIELD_EQ_FEATURE_VERSION` | `features.py`, `constants.py`, `equivariant.py`, Rust featurizer, tests |
| 2 | `HexNodeConv` ray-tap mode: direction taps of designated convs consume visibility-masked, per-channel distance-weighted ray aggregates instead of the distance-1 neighbor. Work items K1 (fused kernel) and K2 (training-memory autograd) | `model.py`, `_triton_conv.py` (K1), `inference.py`, `constants.py`, `arch_meta` |
| 3 | (Optional, independently gated) value-head segment-pool read blocks | `model.py` |
| 4 | (Optional, independently gated) mid-trunk head tap | `model.py` |

Changes 3 and 4 are separable from 1 and 2 and from each other. No new
block types are introduced. L blocks, A blocks, the register lane, the
search, and the training loop are unchanged by this spec. No shard-schema
change is required (all new features derive from stored placement
history).

---

## 1. Change 1 — Featurizer

### 1.1 Feature version gate

New import-time env `HEXFIELD_EQ_FEATURE_VERSION` ∈ {1, 2}, default
**1** (= the current 25-plane map, byte-identical behavior). Version 2
selects the 46-plane map below. All plane-index constants,
`NUM_FEATURES`, `N_AXIS_QUANTITIES`, and the typing sets derive from this
env at import time (the established pattern: all shape knobs are
import-time env, `constants.py`). `arch_meta()` records `feature_version`
and checkpoint load hard-asserts on mismatch (same class as the existing
`support_radius` check). Tests requiring v2 set the env before import
(subprocess pattern, `tests/test_hexfield_eq_equivariance.py`
precedent).

### 1.2 Version-2 plane map (NUM_FEATURES = 46)

| idx | name | type | definition | normalization |
|-----|------|------|------------|---------------|
| 0–10 | (unchanged) | scalar | identical to current planes 0–10 (`F_OWN_STONE` … `F_OPP_LAST_TURN`) | unchanged |
| 11–13 | `F_OWN_LINE_{Q,R,QR}` | axis q=0 | unchanged | /`LINE_NORM` (5.0) |
| 14–16 | `F_OPP_LINE_{Q,R,QR}` | axis q=1 | unchanged | /`LINE_NORM` |
| 17–19 | `F_OWN_LIVE_{Q,R,QR}` | axis q=2 | unchanged | /`LIVE_NORM` (6.0) |
| 20–22 | `F_OPP_LIVE_{Q,R,QR}` | axis q=3 | unchanged | /`LIVE_NORM` |
| 23–25 | `F_OWN_LIVE3_{Q,R,QR}` | axis q=4 | see 1.3 | /`LIVE_NORM` |
| 26–28 | `F_OPP_LIVE3_{Q,R,QR}` | axis q=5 | see 1.3 | /`LIVE_NORM` |
| 29–31 | `F_OWN_LIVE4_{Q,R,QR}` | axis q=6 | see 1.3 | /`LIVE_NORM` |
| 32–34 | `F_OPP_LIVE4_{Q,R,QR}` | axis q=7 | see 1.3 | /`LIVE_NORM` |
| 35–37 | `F_OWN_LIVE5_{Q,R,QR}` | axis q=8 | see 1.3 | /`LIVE_NORM` |
| 38–40 | `F_OPP_LIVE5_{Q,R,QR}` | axis q=9 | see 1.3 | /`LIVE_NORM` |
| 41 | `F_OWN_FORK` | scalar | unchanged definition; **index moves from 23** | /`FORK_NORM` (3.0) |
| 42 | `F_OPP_FORK` | scalar | unchanged definition; **index moves from 24** | /`FORK_NORM` |
| 43 | `F_PLY` | scalar | see 1.4 | see 1.4 |
| 44 | `F_DIST_CENTROID` | scalar | see 1.4 | see 1.4 |
| 45 | `F_SPREAD` | scalar | see 1.4 | see 1.4 |

Axis-plane index formula: `plane = AXIS_PLANE_BASE + q*N_AXES + a` with
`AXIS_PLANE_BASE = 11`, `N_AXIS_QUANTITIES = 10` (v1 map: 4),
`N_AXES = 3`.

Typing sets in `equivariant.py` (version-2):
`_SCALAR_PLANES = {0..10, 41..45}` (16 planes, trivial rep);
`_AXIS_PLANES = {11..40}` (30 planes = 10 copies of the 3-slot axis
module; `rho_in(g)` permutes `a → cosp[g][a]` within each contiguous
3-block).

NOTE: the fork planes move (23/24 → 41/42). Every consumer of
`_SCALAR_PLANES`/`_AXIS_PLANES`, the stem lift, and the derivation-doc §8
typing must be regenerated against the new map under version 2. The
existing derivation warning about mis-placed fork planes ("trains fine,
isn't equivariant") applies to this migration.

### 1.3 live3 / live4 / live5 definitions

Same conventions as the existing `own_live`/`opp_live`
(`features.py:window_features_for_cell`): for cell `x`, axis `a`, the 6
length-6 windows through `x` on `a`; a window is clean-for-side iff it
contains zero anti-side stones; absent/off-support cells contribute 0/0
(existing off-board convention, unchanged); computed at every support row
(legal, stones, halo); sides are side-to-move relative.

- `own_liveK[a] = |{W clean-for-own : own_count(W) >= K}| / LIVE_NORM`,
  for K ∈ {3, 4, 5}.
- `opp_liveK`: same with roles swapped.

(In a decision state, count ≥ 5 ⟺ count = 5, so `liveK=5` is per-cell
standing-win multiplicity.)

Implementation: Python oracle in `features.py`; Rust reads the per-window
counts already maintained in the existing 18-window loop
(`features.rs` / `tactics.rs`) — three additional threshold comparisons
per window, no new state. Parity requirement identical to the existing
graded planes (≤ 1e-6).

### 1.4 New scalar planes

All three are broadcast-or-per-node scalars, D6-invariant and
translation-invariant. Exact formulas (required for Rust/Python parity):

- `F_PLY` (broadcast to all rows): `min(placements_made, 96) / 96.0`.
- Stone centroid `c = (mean(q_s), mean(r_s))` over all placed stones,
  float. Fractional hex distance
  `hexd(dq, dr) = (|dq| + |dr| + |dq + dr|) / 2`.
- `spread = max(1.0, max_s hexd(s - c))`.
- `F_DIST_CENTROID` (per node): `min(hexd(node - c) / (2*spread), 1.0)`.
- `F_SPREAD` (broadcast): `min(spread, 16.0) / 16.0`.
- Empty board (no stones): `F_PLY = 0`, `F_DIST_CENTROID = 0`,
  `F_SPREAD = 1/16`.

### 1.5 What does not change

Existing plane definitions and normalizations (0–22 semantics, fork
semantics), the off-board clean-window convention, `F_PLAYER_COLOUR`,
support-set construction, `HEXFIELD_EQ_SUPPORT_RADIUS` handling, raylen
wire semantics, shard schema.

---

## 2. Change 2 — HexNodeConv ray-tap mode

### 2.1 Baseline mechanism (unchanged parts)

`HexNodeConv` computes, per cell `i`:
`out(i) = sum_{t=0..6} W_t · in_t(i)` — 7 taps (t=0 center, t=1..6 the
`DIRECTIONS` order). Two execution paths exist today:

- Reference path: materialize a `(B, Npad, 7·C)` gathered tensor, one
  GEMM (`_triton_conv.py:93-104`).
- Fused serve path: `_hex_conv_kernel` / conv+LN variant gather the 7 tap
  rows **inside the kernel** from `x` + `gather_idx`; the kernel exists
  to avoid the materialized gather, which is "~60% of the conv cost" at
  serve shapes (`_triton_conv.py:1-16`).

Weight structure (`w_base (7, 12, corb_out, corb_in)`, tap tie,
materialization cache, serve fold) is byte-identical under this proposal.
The tap star (center + 6 directions) is unchanged, so the `tapp` tables
and all tying machinery apply without modification.

### 2.2 The ray-tap input (the change)

For a conv in ray-tap mode, the direction-tap inputs are redefined. Let
`x_{i,d,k}` denote the trunk features of the cell at offset
`k · DIRECTIONS[d]` from cell `i` (zero vector if absent from the
support), and let `raylen_s(i, d)` be the existing per-node ray-length
wire value for side `s ∈ {own, opp}` (semantics exactly as
`ray_lengths_for_cell`, `features.py:187-220`: the ray stops at and
includes the first anti-side stone; own-side stones and empties pass
through; the cell's own occupancy is never consulted).

Orbit channels split into two visibility halves (requires `C_ORBIT` even;
see 2.6): `G_own` = orbit channels `[0, C_ORBIT/2)`, `G_opp` =
`[C_ORBIT/2, C_ORBIT)`, within every fiber slot. For orbit channel `c`
with side `s(c)`:

```
in_d(i)[c] = Σ_{k=1..5}  α[k, c] · 1[k <= raylen_{s(c)}(i, d)] · x_{i,d,k}[c]
in_0(i)    = x_i                                   (center tap unchanged)
```

Free parameter per ray-tap conv: `α ∈ R^(5 × C_ORBIT)` — per-distance,
**per-orbit-channel**, shared across all 6 directions (direction-orbit
tying) and tiled over the 12 fiber slots (slot-constant). 80 scalars per
equipped conv at `C_ORBIT = 16`.

Initialization: `α[:, c] = (1, 0, 0, 0, 0)` for every channel. Under this
init the ray-tap conv is functionally identical to the baseline 7-tap
conv (init-equivalence; tests T4/T4b). Init-equivalence relies on the
terminal-blocker convention above (raylen ≥ 1 whenever the distance-1
cell is on the support); T4 pins this.

### 2.3 Equipped set

Env `HEXFIELD_EQ_RAYTAP` ∈ {`0`, `conv2`, `both`}, default `0` in code;
the candidate arch for the next run sets `both`:

- `0` — off; all convs baseline (current behavior, byte-identical).
- `conv2` — the second conv of every C block runs in ray-tap mode; first
  convs baseline. (Retained as attribution arm A2c.)
- `both` — both convs of every C block run in ray-tap mode.

The stem and the tied head convs are always baseline. The mode is
recorded in `arch_meta` (§4).

### 2.4 Serve and training paths — work items K1 and K2

**Reference path (functional definition).** For equipped convs the
gather step is replaced by the masked weighted gather-sum of 2.2. This
path defines numerics for all tests and is the serve fallback (serve
runs it under `no_grad`, so serve memory is unaffected).

**K2 — training-path memory (BLOCKING for any training at `conv2` or
`both`).** A naive implementation (gather `(B, N, 30, C)` → multiply by
α → sum over k) saves the gathered intermediate for backward. At the
established training shapes (B ≤ 48, S ≤ 648, `model.py:215`) that is
≈ 717 MB fp32 per equipped conv vs ≈ 167 MB for the baseline `(B,N,7C)`
gather — ≈ +7.2 GB at `both` (10 equipped convs), an OOM or forced batch
cut on the 12 GB training card that would silently confound arms A2/A5
against A0/A1. Deliverable: pre-aggregation as a custom
`torch.autograd.Function` that saves only `x` (alive anyway), the ray
gather index, the visibility masks (`(B, N, 30)` u8 ≈ 28 MB), and `α`,
recomputing the gather in backward:
`grad_α[k, c] = Σ_{b,i,d} mask·x_gathered·grad_out` (einsum over the
regathered rows); `grad_x` = scatter-add of `α·mask·grad_out` into
source rows. Acceptance: test T8.

**K1 — fused serve kernel (throughput deliverable).** The existing
kernels cannot consume ray-tap input (they gather internally from
`gather_idx`). Deliverable: a variant of `_hex_conv_kernel` /
`_hex_conv_ln_kernel` whose direction taps run an inner k-loop (5 masked
loads + FMA per direction), consuming the ray gather index, `α`, and the
raylen wire in-kernel — no `(B, Npad, 7·C)` materialization. Numerics
class unchanged (fp16 rows × fp16 weights, fp32 accumulation).

- Bench gate: equipped conv+LN wall-clock within ~10% of the baseline
  fused conv+LN at serve shapes (B·Npad ≈ 24k, C = 192). The in-kernel
  load count per query rises 7 → 31 rows; the gate is acknowledged as
  ambitious.
- **Pre-agreed fallback if the gate is missed after reasonable tuning**
  (so a miss cannot stall Phase L): (a) record the achieved figure and
  relax the gate to ≤ 20%, or (b) promote `conv2` (half the equipped
  convs) as the candidate configuration. The choice is recorded in the
  ladder notes; either way Phase L proceeds.
- Until K1 lands, equipped convs at serve run the reference path. Scale
  of the un-fused exposure, from the kernel header's own accounting
  (gather ≈ 60% of reference conv cost): `conv2` un-fuses ~half of the
  70 conv-units, `both` un-fuses all of them. Serve throughput measured
  before K1 must be labeled reference-path.

### 2.5 Wiring

- **Tap→raylen-slot LUT**: conv taps are indexed in `DIRECTIONS` order
  (`constants.py`); raylen slots are indexed `side*6 + axis*2 + dir`
  with axis in [Q, R, QR] and dir in {+ = 0, − = 1}
  (`features.py:194-196`). The mapping `t → (axis, dir)` MUST be
  generated programmatically from `constants.DIRECTIONS` and the axis
  delta table (not hand-coded), with a unit test asserting the bijection
  and sign convention (T7). Mis-wiring is silent (net trains; visibility
  is wrong); T4b is the behavioral guard.
- **Raylen wire un-gating (serve side)**: raylen staging is currently
  keyed on `'L' in layout` at three sites, each of which becomes keyed
  on `('L' in layout) or (HEXFIELD_EQ_RAYTAP != '0')`: the staging
  buffer KEYS (`inference.py:66-68`), the CUDA-graph statics
  (`needs_raylen`, `inference.py:231-260`), and the serve payload path
  (`inference.py:377`; the Rust serve payload includes raylen under the
  same condition).
- **Training side needs no un-gating** (reviewer-verified): the Rust
  expand kernel emits raylen unconditionally (`samples.py:25`, spec
  D-S29) and the trainer forwards `batch.get("raylen")` unconditionally
  (`trainer.py:810`). Arm A5 (L-free layout + ray-tap) therefore trains
  with no extra plumbing.
- **Ray gather index**: the sync-free ray-index build (currently built
  for L blocks) is enabled whenever ray-tap is on, including L-free
  layouts (required by arm A5).

### 2.6 Equivariance requirements

- Tap star unchanged ⇒ existing tap-tie constraint (C1) and `tapp`
  permutation apply as-is.
- `α` shared across the 6 directions ⇒ invariant under the direction
  permutation induced by any `g` (single direction orbit).
- `α` indexed by orbit channel and tiled slot-constant is exactly
  equivariant: the group action permutes fiber slots and never orbit
  channels (same legality argument as `GroupAffineNorm`'s orbit-tied
  affine and the L block's own/opp sub-head split).
- The own/opp visibility-half split rides the orbit index. New
  import-time check: `C_ORBIT % 2 == 0` whenever
  `HEXFIELD_EQ_RAYTAP != '0'` (currently enforced only for layouts
  containing `'L'`), plus a constructor-kwarg twin of the same check.
- `raylen` transforms covariantly under D6 (geometry + stones only); the
  masked aggregate therefore commutes with the group action. Covered by
  T3.

---

## 3. Optional gated additions (separable)

### 3.1 Segment-pooled value reads (`HEXFIELD_EQ_VALUE_SEGPOOL`, default 0)

Two additional read blocks appended to the value head's existing 8:
group-pooled masked means over (a) the stone segment and (b) the legal
segment of the support (segment boundaries from `Support.segments()`).
Same `inv_read` sharing and 64-wide read-block structure as the existing
pooled-cells block. The `value_reduction` input width grows by 2 × 64;
new reduction columns zero-initialized.

### 3.2 Mid-trunk head tap (`HEXFIELD_EQ_HEAD_MIDTAP`, default 0)

The trunk output after the first A block is additionally normalized by a
dedicated `GroupAffineNorm` and fiber-concatenated to the final trunk
output for the policy and value head reads only. Head input widths grow
from `C` to `2C`; added weight columns zero-initialized.

---

## 4. Constants / env / arch_meta summary

| item | value / change |
|---|---|
| `HEXFIELD_EQ_FEATURE_VERSION` | new env: {1, 2}, default 1; selects plane map; v2 map per §1.2 |
| `NUM_FEATURES` | 25 (v1) / 46 (v2), derived from the env at import |
| `N_AXIS_QUANTITIES` | 4 (v1) / 10 (v2) |
| plane index constants | version-dependent per §1.2; fork re-indexed under v2 |
| `_SCALAR_PLANES` / `_AXIS_PLANES` | version-dependent per §1.2 |
| `HEXFIELD_EQ_RAYTAP` | new env: {0, conv2, both}, default 0 |
| `HEXFIELD_EQ_VALUE_SEGPOOL` | new env: {0,1}, default 0 |
| `HEXFIELD_EQ_HEAD_MIDTAP` | new env: {0,1}, default 0 |
| `arch_meta()` | adds `feature_version`, `raytap` (ternary, authoritative), `value_segpool`, `head_midtap` |
| checkpoint load | hard assert on `feature_version` mismatch (same class as the existing `support_radius` mismatch check) |
| state-dict inference | `raytap` read from `arch_meta`; fallback inference distinguishes `conv2`/`both` by presence of `alpha` on first-conv keys (presence of any `alpha` alone is insufficient) |
| import-time checks | `C_ORBIT` even required when `raytap != 0`; constructor-kwarg twin of the same check |
| new parameters | per equipped conv: `alpha (5, C_ORBIT)`; per 3.1/3.2: listed above |

The `raytap` mode and `feature_version` are threaded as constructor
kwargs (env → net kwargs → `arch_meta` →
`infer_net_kwargs_from_state_dict`), per the existing key-set
discipline.

---

## 5. Cost accounting

MAC accounting per cell, units of C² = 36,864 (C = 192):

| component | baseline | with proposal |
|---|---|---|
| stem GEMM (7·C·NF) | 0.91 (NF=25) | 1.68 (NF=46) |
| C-block conv GEMMs | 70 | 70 (unchanged) |
| ray-tap pre-aggregation (reference path) | — | ~0.16/conv MAC-equivalent; ×10 at `both` ≈ 1.6 |
| L / A / lane / heads | ~73 | unchanged |
| optional 3.1 / 3.2 | — | ~0.5 / ~2 |

Bandwidth note (normative): the pre-aggregation is gather-dominated, not
MAC-dominated — up to 30 row-loads per cell per equipped conv vs 6 for
the baseline tap gather (~4–5× tap-gather volume). MAC deltas above are
therefore not a wall-clock prediction for the serve path. Serve cost is
determined by measurement under §6.3, on the fused path once K1 lands;
pre-K1 numbers are reference-path numbers and labeled as such.

Training memory (normative, drives K2): naive ray-tap backward saves
≈ 717 MB fp32 per equipped conv at B=48, S=648 vs ≈ 167 MB baseline;
K2 reduces the additional saved state to the u8 masks (≈ 28 MB) plus
bookkeeping. See §2.4 and T8.

Featurizer-side cost: liveK planes are threshold reads of per-window
counts already computed in the existing window loop; centroid/spread/ply
are O(stones) per position.

---

## 6. Test and validation requirements

### 6.1 Correctness tests

- **T1 — featurizer parity**: Python oracle vs Rust for planes 23–45
  under `FEATURE_VERSION=2`; exact for binary/int-derived planes, ≤1e-6
  for graded/float (existing tolerance model). Includes the empty-board
  cases of 1.4. Also: `FEATURE_VERSION=1` output is byte-identical to
  pre-change output (regression guard).
- **T2 — typing regeneration**: all-12 D6 expand parity of the 46-plane
  input under the v2 `_SCALAR_PLANES`/`_AXIS_PLANES`; stem-lift
  verification per derivation §8 against the new map (fork re-index).
- **T3 — full-net equivariance**: existing harness pattern (subprocess
  env) with `HEXFIELD_EQ_RAYTAP=conv2` and `both`; all 12 group
  elements.
- **T4 — init equivalence (reference path)**: with `α` at init, ray-tap
  conv output equals baseline conv output on random supports, both
  sides, all node kinds (legal/stone/halo). Guards the terminal-blocker
  raylen convention (2.2).
- **T4b — init equivalence (serve wire path)**: full serve stack (fp16,
  weight cache, perm fold, CUDA-graph capture, raylen staging) with
  ray-tap enabled at init-`α` vs the baseline net, on layouts WITH and
  WITHOUT `'L'`. Behavioral guard for the tap→slot LUT (2.5) and the
  raylen un-gating; T4 alone cannot catch either.
- **T5 — serve parity**: as the existing fold-gate tolerance model, with
  ray-tap enabled and trained (non-init) `α`; repeated on the K1 fused
  path when it lands.
- **T6 — state-dict discipline**: `arch_meta` round-trip and
  `infer_net_kwargs_from_state_dict` for all knob combinations,
  including `conv2`-vs-`both` disambiguation and the `feature_version`
  load assert.
- **T7 — LUT bijection**: unit test that the generated tap→(axis, dir)
  mapping is a bijection consistent with `DIRECTIONS` and the axis
  deltas.
- **T8 — training-memory gate (K2 acceptance)**: one full-shape training
  step (B=48, S=648, C=192, layout `CCLACCLACLA`, `RAYTAP=both`,
  `REG_LANE=1`, AMP as in production) on a 12 GB-class device: no OOM,
  no batch-size reduction, gradients for `α` and `x` match a small-shape
  naive-implementation oracle (≤1e-5 rel), and peak CUDA memory delta vs
  the `RAYTAP=0` baseline recorded and ≤ 1.0 GB.

### 6.2 Pre-implementation prototype gate

- **P1′ — discrimination gate (numpy/torch CPU)**: a single ray-tap
  layer plus pointwise head, trained on a synthetic one-axis task whose
  label depends on stone POSITIONS within reach (contiguity-class task,
  e.g. "placing at center yields a run ≥ 4"), not on counts alone; must
  beat a depth-1 baseline 1-ring conv by a margin fixed before running.
  External simulation evidence indicates the mechanism passes this task
  class at depth 1 (baseline ~75%, ray variants ~100%); the gate is
  retained as a reproducible in-tree artifact.

### 6.3 Evaluation procedure (attribution + subtraction matrix)

Prefit-ladder arms (existing autonomous ladder, SealBot-anchored), all
at C=192 unless noted.

Procedural requirements:
- **Search budgets are visit-based, never time-based**, for every
  arm-vs-arm and arm-vs-anchor match (pre-K1 the reference path is
  ~15–25% slower per sim; a time budget would handicap ray-tap arms).
- **Each arm pins the complete arch env** — baseline
  `scripts/prefit_env/hexfield_eq_arm4_raylayout.env` (CHANNELS=192,
  TRUNK, REG_LANE=1, REG_TOK_READ=0, SUPPORT_RADIUS=4) plus only the
  per-arm deltas listed below. A REG_LANE or SUPPORT_RADIUS mismatch
  across arms is a confound.
- Serve throughput (pos/s, full serve stack, fused-path status labeled)
  is recorded next to BT rank for every arm.

| wave | arm | features | raytap | layout | hypothesis tested |
|---|---|---|---|---|---|
| 1 | A0 (control) | v1 (25) | 0 | `CCLACCLACLA` | baseline |
| 1 | A1 | v2 (46) | 0 | `CCLACCLACLA` | features alone |
| 1 | A2 | v2 | `both` | `CCLACCLACLA` | operator + features |
| 1 | **A5** | v2 | `both` | `CCACCACA` (L removed) | does ray-tap subsume L? (load-bearing arm) |
| 2 (conditional) | A2c | v2 | `conv2` | `CCLACCLACLA` | equipped-set attribution |
| 2 (conditional) | A6 | winner | winner | winner − 1 A block | attention budget |
| optional | A3 | v2 | `both` | `CCLACCLACLA` + §3.1 | value segpool |
| optional | A4 | v2 | `both` | `CCLACCLACLA` + §3.2 | mid tap |

Notes: A5 requires the un-gating of 2.5 on the serve side only (training
needs none). `CCACCACA` ends with 'A' and preserves the C/A counts of
the live layout (the subtraction removes exactly the 3 L blocks). The
matrix is read on both axes (strength AND serve throughput):
"equal strength, smaller layout, faster serve" is a recordable outcome.

Recorded metrics: held-out BC policy top-1, soft-policy KL, value loss,
`value_ece`, ladder BT rank, record-only Strix baseline match for the
ladder winner (existing procedure), serve pos/s.

---

## 7. Design notes (factual record of choices)

- **Hex-grid tap geometry**: all six distance-1 neighbors lie ON the
  three win axes; off-axis cells first occur at hex distance 2 and are
  reached only by composing taps across directions. The baseline 1-ring
  conv is therefore the k=1 truncation of the ray tap star, and the
  ray-tap conv at init-`α` is the baseline conv. Off-axis coverage under
  any equipped set is unchanged relative to baseline (it always came
  from composition).
- **Per-channel vs shared α**: shared-per-side α (v1 of this spec) is
  the tied special case of per-orbit-channel α (default since v2). The
  choice is recorded as a choice, not a requirement: external
  small-scale simulation found shared α sufficient on single synthetic
  tasks; per-channel α is equally equivariant, adds 70 scalars per conv
  over v1, and permits distinct per-channel reach profiles within one
  conv.
- **`both` vs `conv2` as default**: with per-channel α and
  init-equivalence, the division of labor between distance-1 and ray
  reach is learnable per channel; `both` makes it fully learned, `conv2`
  hard-codes the first conv to k=1. `conv2` is retained as arm A2c and
  as the pre-agreed K1 fallback configuration.
- **Raylen convention dependency**: 2.2's mask and T4's equivalence rely
  on `ray_lengths_for_cell` including the terminal anti-side stone and
  never consulting the cell's own occupancy (`features.py:198-203`).
  Any future change to that convention invalidates T4/T4b baselines.

---

## 8. Explicitly out of scope

- New block types; changes to L blocks, A blocks, or the register lane.
  (A5/A6 vary the layout string only; they change no block internals.)
- Register-lane removal / consolidation of its counting role into a
  ported global-pooling block (`docs/PLAN_MAIN12_GLOBAL_POOLING.md`):
  identified as a possible follow-up CONDITIONAL on A5/A6 outcomes;
  requires its own spec; not specified here.
- Window or line node segments, line-scan operators, K>1 tap-star
  variants, shared-transform per-edge modulation as a replacement for
  per-direction weight matrices.
- Quotient-representation changes (separate project; the additions here
  introduce no new rep types).
- Changes to: off-board window convention, support radius, shard schema,
  search, selfplay, eval pipeline.

---

## 9. Implementation plan

### 9.1 Process constraints (read first)

- **Live-run isolation.** A live training soak runs from this tree's
  launch scripts. All work happens on a feature branch off
  `main_9-fastrow-strip`; do not restart or redeploy any live service.
  Two independent guards keep the live run safe even across an
  accidental supervisor restart: every new env knob defaults to the
  current behavior (`FEATURE_VERSION=1`, `RAYTAP=0`, optional knobs 0),
  and no default value or existing code path may change observable
  behavior when all knobs are at defaults (T1's byte-identical check,
  T4's init-equivalence).
- **Import-time env discipline.** All shape knobs are read once at
  import (`constants.py`). Tests that need v2 features or ray-tap must
  set env before importing `hexfield_eq` — use the subprocess pattern of
  `tests/test_hexfield_eq_equivariance.py`.
- **Test invocation** (WSL venv):
  `wsl -e bash -c 'cd /mnt/e/Hexo-BotTrainer-hexgt && source /root/.venvs/hexgt-build/bin/activate && export PYTHONPATH=packages/hexfield_eq/python:packages/hexo_engine/python:packages/hexo_utils/python && pytest tests/<file> -q'`
  CPU-only work may also run under Windows Python with the same
  PYTHONPATH.
- **State-dict discipline.** Any new arch knob rides `arch_meta()` and
  `infer_net_kwargs_from_state_dict`; non-persistent buffers keep key
  sets stable. Follow `CONTEXT.md §9`.
- When this spec and the code disagree, the code wins — stop and report
  the discrepancy (with file:line) rather than improvising.

### 9.2 Work items

| id | deliverable | primary files | acceptance |
|---|---|---|---|
| W-P1 | P1′ discrimination-gate script (CPU, seeded, in-tree) | `tests/` or `scripts/` per repo convention | gate margin fixed in the script header; passes |
| W-F1 | `FEATURE_VERSION` env gate; v2 plane constants + typing sets | `constants.py`, `equivariant.py` | T2; v1 path byte-identical (part of T1) |
| W-F2 | Python oracle: liveK planes + 3 scalars (v2 map) | `features.py` | T1 (oracle side) |
| W-F3 | Rust featurizer: liveK thresholds + scalars | `features.rs`, `tactics.rs` (hexo_engine) | T1 (parity) |
| W-F4 | Stem-lift/derivation verification against the v2 map | derivation test | T2 |
| W-R1 | `HexNodeConv` ray-tap mode, reference path: `α` param, mode kwarg, masked gather-sum | `model.py` | T4; T3 |
| W-R2 | **K2** custom-autograd pre-aggregation | `model.py` (or sibling module) | T8, incl. small-shape gradient oracle |
| W-R3 | Wiring: generated tap→slot LUT; serve-side raylen/ray-index un-gating (3 sites, §2.5) | `model.py`, `inference.py`, Rust serve payload | T7, T4b (with- and without-'L' layouts) |
| W-R4 | Env/kwarg/meta threading; import-time + kwarg-twin checks; state-dict inference | `constants.py`, `model.py`, `checkpoints.py`/`prefit.py` loaders | T6 |
| W-R5 | Serve parity at trained α; fold-gate coverage | existing test files' pattern | T5 |
| W-K1 | Fused kernel variant (inner k-loop, in-kernel α/raylen) | `_triton_conv.py` | bench gate §2.4 (fallback policy applies); T5 on fused path |
| W-O1 | (optional) §3.1 segment-pool reads | `model.py` | T3/T6 extended; zero-init verified |
| W-O2 | (optional) §3.2 mid tap | `model.py` | T3/T6 extended; zero-init verified |
| W-L1 | Arm env files (per-arm full env pinning), ladder configuration, wave-1 launch | `scripts/prefit_env/`, ladder config | §6.3 procedural requirements met; metrics recorded incl. pos/s |

### 9.3 Sequencing and gates

1. **W-P1 first** (pre-implementation gate; expected to pass on external
   evidence, must exist as an in-tree artifact).
2. **Phase F** = W-F1 → W-F2 → W-F3 → W-F4 (T1, T2 green). Independent
   of everything else; can land ahead of Phase R.
3. **Phase R** = W-R1 → W-R2 (K2 is blocking before any training run at
   `conv2`/`both`) → W-R3 → W-R4 → W-R5 (T3–T8 green). W-K1 proceeds in
   parallel after W-R3; its bench gate (or pre-agreed fallback) must
   resolve before any throughput-sensitive ladder reading, but wave-1
   strength results (visit-budgeted) do not wait for it.
4. **Phase L** = W-L1: wave 1 (A0, A1, A2, A5) → read → wave 2
   (A2c, A6) and optional A3/A4 as warranted.
5. **Phase S** (conditional consolidation, register lane → gpool): out
   of scope; separate spec if A5/A6 outcomes trigger it.
