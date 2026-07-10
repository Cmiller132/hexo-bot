# Derivation — exact D6-equivariant attention on a regular-representation fiber

Status: DERIVATION / contract for Phase 3b. Date: 2026-07-08. This is Phase 3a of
`docs/PLAN_D6_EQUIVARIANT_REWRITE.md`: the group-theory spike that the Equivariant
Trunk implements. **No `model.py` code is written here** — this doc plus the
verified prototype (`tests/test_hexfield_eq_derivation.py`, and its scratch twin)
are the exit gate.

**Verified.** A standalone numpy prototype builds the full tied block (typed-lift
stem → tied `HexNodeConv` → group-norm → tied Q/K/V/out with a coset-aligned head
split → jointly-tied relative bias → group-pooled heads) and asserts
`f(g·board) == g·f(board)` for **all 12** `g ∈ D6` to **fp64 machine precision**:

```
[JOINT (row,head) bias]  max|f(g.x)-g.f(x)|  trunk=1.95e-14  policy=5.33e-15  value=9.94e-16
conv tap constraint  W[pi_g t]=M(g)W[t]M(g)^-1  dev: 0.00e+00
q_proj commutes with M_reg(g)                    dev: 0.00e+00
tied-conv free blocks reached (== 84 = 7x12)        : 84
NEGATIVE CONTROL [row-orbit tie, head-free]  trunk=1.13e+00   <-- correctly BREAKS
```

The negative control is load-bearing: it demonstrates that Phase 2's board-orbit
bias tie, *on its own*, does **not** give an equivariant trunk — the head axis
must be tied **jointly** with the row orbit (see §5).

---

## 0. Conventions and the group (grounded in the current code)

`D6` = the order-**12** dihedral symmetry of the hex lattice about the origin.
Elements are indexed `0..11` exactly as `geometry.apply_d6` (`geometry.py:61-71`):
indices `0..5` are `rot60^i`; indices `6..11` are `rot60^{i-6} ∘ reflect`. `rot60`
and `reflect` are `geometry.py:53-58`. Composition, inverses, and every table
below are the machine-computed values from the prototype (all reproduced by
`explore_group.py`), so the doc's concrete numbers are the contract.

- **Inverses** (`d6_inverse`, verified): rotations `i ↦ (6-i) mod 6`; every
  reflection `6..11` is an involution.
- **The 6+1 taps** of `HexNodeConv` (`model.py:325-374`): tap 0 = center (offset
  `(0,0)`); taps 1..6 = the fixed direction order `DIRECTIONS`
  (`constants.py:20-27`), the `rot60`-orbit of `(1,0)`.
- **Tap permutation** `π_g` (offset `δ_t ↦ g·δ_t`): rotations cyclically advance
  taps 1..6; **reflections reverse the tap cycle** (`reflect(D[i]) = D[5-i]`):

  | g | π_g on taps `[0,1,2,3,4,5,6]` |
  |---|---|
  | `rot60` (g=1) | `[0, 2,3,4,5,6,1]` |
  | `reflect` (g=6) | `[0, 6,5,4,3,2,1]` |
  | `refl·rot` (g=7) | `[0, 1,6,5,4,3,2]` |

The board acts on a cell coordinate `x` by `x ↦ g·x = apply_d6(g, x)`; the win
axes `{Q,R,QR}` = `{(1,0),(0,1),(1,-1)}` lines carry the **S3 axis-permutation**:
`rot60` 3-cycles `Q→R→QR`, `reflect` transposes `Q↔QR` (fixes `R`)
(`on_win_axis`, `geometry.py:85-91`).

### 0.1 Everything is a *permutation* representation — no signs

The trunk fiber carries `C_orbit` copies of the **regular representation** of
`D6`. In that basis `ρ_reg(g)` is a **0/1 permutation matrix** (§1). The input
axis-module and the head/axis actions are likewise permutation modules. Therefore
**every tied weight below is a pure index-gather with all signs `+1`** — there are
no `−1` factors. (Sign factors appear only if one decomposes into the sign or 2-D
irreps of `D6`; the regular-rep design deliberately avoids that.) When the plan
says the reflection action is "tap-cycle reversal + axis transpose + fiber
relabel", each of those three is a **permutation**, and this doc gives all three
explicitly.

---

## 1. The regular-rep fiber and the field action

Fix `C = 12·C_orbit` (the plan's `HEXFIELD_C_ORBIT`, `HEXFIELD_GROUP_ORDER=12`).
Index a channel by `(h, a)` with **fiber slot** `h ∈ D6` and **orbit channel**
`a ∈ {0,…,C_orbit−1}`. Slot-major layout: `c = h·C_orbit + a`.

The **regular representation** acts on the fiber by *left multiplication of the
slot label*:

```
(ρ_reg(g) v)[h·C_orbit + a] = v[(g^{-1}·h)·C_orbit + a]        # gather form
```

i.e. slot `h`'s content moves to slot `g·h`; `ρ_reg` is a homomorphism
(`ρ_reg(g1)ρ_reg(g2)=ρ_reg(g1 g2)`), orthogonal, and permutation-valued. On the
fiber it acts as `ρ_reg(g) ⊗ I_{C_orbit}`; write `M(g) := ρ_reg(g) ⊗ I_{C_orbit}`.

A **feature field** `f : cells → R^C` (a steerable field) transforms as

```
(T_g f)(x) = M(g) · f(g^{-1}·x)                                  # (★ field action)
```

— permute cells by `g`, and rotate each fiber by `M(g)`. This is the object the
whole network must intertwine. **Equivariance of a layer `L` means `L∘T_g = T_g∘L`
for all `g`.**

---

## 2. Tied `HexNodeConv` (the trunk conv and the stem share this constraint)

The conv (`model.py:344-374`) is `(Cf)(x) = Σ_{t=0}^{6} W_t f(x+δ_t) + b`, with
`W_t ∈ R^{C_out×C_in}` and `δ_0=0`, `δ_{1..6}=DIRECTIONS`.

### 2.1 The constraint

Substituting (★) and re-indexing the tap sum (full derivation reproduced in the
scratch script header) gives, **for all `g` and all taps `t`**:

```
W_{π_g(t)} = M_out(g) · W_t · M_in(g)^{-1}          # (C1) tap-tie
b          = M_out(g) · b                            # (C2) bias invariance
```

`(C2)` forces the bias to be **slot-constant**: `b[h·C_orbit+a] = β[a]`, one
`C_orbit`-vector broadcast over the 12 slots.

`(C1)` couples taps that lie in the same `π`-orbit. There are exactly **two tap
orbits**: `{center}` (a `π`-fixed point) and `{the 6 directions}` (one orbit,
transitive, with `stab(D[0]) = {e, τ}` where `τ = g7` is the reflection fixing
`(1,0)`).

### 2.2 Free-parameter count and the `w_base` storage

Because `M(g)` is a permutation, `(C1)` says: the block `W_t[a,b]` (a
`C_orbit_out×C_orbit_in` block indexed by out-slot `a`, in-slot `b ∈ D6`) equals
the block `W_{π_g(t)}[g·a, g·b]`. So the free blocks are the **orbits of the
diagonal action** `g·(t,a,b) = (π_g(t), g·a, g·b)` on the `7·12·12` triples:

- **center tap:** `M(g)`-conjugation-invariant ⇒ `W_0` is itself `D6`-equivariant
  ⇒ a **group convolution**: `W_0[a,b] = ω_{a^{-1}b}` — **12** free blocks.
- **direction taps:** stabilizer of the representative is `{e,τ}` (order 2), so the
  `144` blocks of `W_{D[0]}` collapse under `(a,b)∼(τa,τb)` to **72** free blocks;
  the other five direction taps are *determined* by `(C1)`.

Total **12 + 72 = 84** free blocks = **`7 × 12`**, which is exactly the plan's
storage `w_base : (7, C_orbit_in, 12, C_orbit_out)`. A clean transversal realizing
this is `{(t, e, s) : t∈taps, s∈D6}` (input slot pinned to the identity, output
slot `s` free) — 7 taps × 12 slots.

### 2.3 The generator (the "precomputed index-gather")

Store `w_base[t, s]` (block for tap `t`, output slot `s`, input slot `e`). Generate
the materialized `(7, C_in, C_out)` weight each forward by the **pure gather**

```
W_t[out=a, in=b]  =  w_base[ π_{a^{-1}}(t) ,  a^{-1}·b ]        # (GEN)
```

where `π_{a^{-1}}(t)` = the tap index of the offset `a^{-1}·δ_t` (a rotation/
reflection of the tap), and `a^{-1}·b` is group multiplication. `(GEN)` is
verified to satisfy `(C1)` exactly (dev `0.00e+00`) and to touch exactly 84
distinct stored blocks. Note the read tap `π_{a^{-1}}(t)` **depends on the output
slot `a`** — this is precisely what mixes the six direction taps (it is *not* a
per-tap group convolution; only the center tap reduces to one).

**Reflection, explicitly.** In `(GEN)` a reflection `g` enters through three
composed permutations, all sign-free: (i) the **tap-cycle reversal** in `π`
(`D[i]↦D[5−i]`), (ii) the **fiber relabel** `M(g)` (left-mult of slots), and (iii)
the **axis transpose** it induces on the head/axis structure (§4–5). Materialize
the full weight **eagerly inside `forward`** (survives `SERVE_HALF` deepcopy and
CUDA-graph capture, plan §3b); cache the dense weight keyed on the base param
`_version`. Recompute fan-in init on the 84-block basis. Reference GEMM
(`model.py:370-374`) and the fp16 Triton conv/conv+LN kernels consume the
materialized weight unchanged. **Drop `HEXFIELD_CONV_FP8` for v1** (its
`id(weight)` cache, `_triton_conv.py:359-374`, misses every forward under eager
regeneration).

### 2.4 The 1×1 case: Q/K/V/out and MLP `fc1/fc2`

`q_proj/k_proj/v_proj/out_proj` (`model.py:451-454`) and `fc1/fc2`
(`model.py:512-513`) are plain `nn.Linear` = 1×1 convs = **the center-tap case**.
Each must therefore be a **group-convolution**: block-structured with `12` free
blocks,

```
W[out=a, in=b] = ω_{a^{-1}b}          # 12 free C_orbit×C_orbit blocks
```

verified: `q_proj` commutes with `M(g)` to `0.00e+00`. Their biases are
slot-constant `(C2)`. For the MLP, the **hidden width must also be a regular-rep
fiber**: `MLP_RATIO·C = 12·(MLP_RATIO·C_orbit)`, i.e. hidden `C_orbit_hidden =
MLP_RATIO·C_orbit`; `fc1` is `ω : C_orbit → C_orbit_hidden` (12 blocks), `fc2` the
reverse. GELU is applied per channel and commutes with the slot permutation, so
the pointwise nonlinearity is automatically equivariant. These are `nn.Linear`
reparameterizations — invisible to the Triton attention kernel.

---

## 3. Group-norm and orbit-tied `LayerScale`

Every `nn.LayerNorm(C)` (`model.py:394-396, 509-511, 607, 623`) and every
`LayerScale.gamma` (`model.py:377-385`) applies a **dense per-channel affine**,
which breaks equivariance because after `M(g)` permutes channels, channel `c` is
scaled by `γ[c]` instead of `γ[g^{-1}c]`.

**Spec (verified equivariant):**

- **Normalize over the full fiber.** LayerNorm's mean/variance are *symmetric*
  reductions over the `C` channels, hence invariant under the slot permutation;
  the normalized vector transforms by exactly the same `M(g)`. (Per-slot GroupNorm
  is an equally valid alternative — either commutes with `M(g)`.)
- **Affine tied per `C_orbit`.** One `(γ[a], β[a])` per orbit channel, **broadcast
  over the 12 slots**: `weight[h·C_orbit+a] = γ[a]`, `bias[…] = β[a]`. This is the
  only change; with it the affine commutes with `M(g)`.
- **`LayerScale`** identically: `gamma` is one `C_orbit`-vector broadcast over the
  12 slots.

**Serve fused conv+LN epilogue (v1, kept).** The fused conv+LN Triton kernel
(`_triton_conv.py::_hex_conv_ln_kernel`) runs LayerNorm-over-`Cout` on the fp32
conv accumulator then a **per-channel** affine `lnw[n]·x̂ + lnb[n]` in its
epilogue. This is *exactly the spec above* provided the affine vector is the
orbit-tied one: `GroupAffineNorm` exposes `.weight = γ.repeat(groups)` and
`.bias = β.repeat(groups)` (the slot-major broadcast `weight[slot·C_orbit+a] =
γ[a]`), and `ConvBlock.forward` passes exactly those `(C,)` vectors into the
kernel. The kernel's symmetric full-fiber mean/variance already commute with
`M(g)`, so the fused epilogue **is** the equivariant group-norm — **no kernel
change is needed and the fusion is retained for v1** (verified by
`test_hexfield_eq_equivariance.py::test_serve_path_equivariance`, which exercises
the compiled fused conv+LN kernel on CUDA under autocast fp16 and asserts D6
equivariance to a fp16 serve tolerance). The earlier `HEXFIELD_CONV_FP8` epilogue
that would have needed its own scale handling is **removed** for v1 (§2.3), so
this plain fp16 fusion is the only serve conv+LN path.

---

## 4. Head split: heads = cosets of an order-`|K|` subgroup

Multi-head attention (`RelPosAttention.forward`, `model.py:457-501`) reshapes
`q,k,v` to `(B,S,heads,head_dim)` and runs one **independent** attention per head.
Under `M(g)` the 12 fiber slots permute, so a head's channels are moved to
*another head's* channels **iff the head partition is a `D6`-block system**, i.e.
the heads are the **left cosets `gK` of a subgroup `K ≤ D6`**. Then:

```
head_dim = |K|·C_orbit ,   heads = [D6:K] = 12/|K| ,   C = heads·head_dim = 12·C_orbit ✔
```

The plan's target width `C_orbit=16, C=192, heads=3, head_dim=64` forces
`|K| = 4`: **`K` is an order-4 (Klein-four) subgroup, 3 heads, `head_dim = 4·16 =
64`.** The natural choice is

```
K = stab(Q-axis) = {0, 3(rot180), 7, 10}          # an order-4 subgroup
heads (left cosets)  =  Q:{0,3,7,10}   R:{1,4,8,11}   QR:{2,5,6,9}
```

so **the 3 heads are the 3 win-axes**: `rot60` 3-cycles the heads
(`[Q,R,QR]→[R,QR,Q]`), and a reflection transposes two (e.g. `g7`:
`[Q,R,QR]→[Q,QR,R]`). **Implementation contract:** order the 12 slots grouped by
left coset so that the `(heads=3, head_dim=64)` reshape lands each head on one
axis-coset's `4·C_orbit` channels. Under `M(g)`,

```
(M(g) q)|_{head h} = P_K(g) · ( q|_{head g^{-1}·h} )            # (H)
```

with `P_K(g)` an orthogonal within-head permutation — the head index shifts by the
coset action, and the intra-head dot product `q^h·k^h` is invariant to `P_K(g)`.
Because the per-head content dot product is invariant while the **head index
shifts**, the bias must shift with it — §5.

---

## 5. The relative-position bias: it MUST tie jointly across (row, head)

This is the specific under-specification the survey flagged. **Answer: yes — the
per-head bias must additionally tie across the head axis, jointly with the board
orbit; Phase-2's `(45, heads)` table with heads left free is NOT equivariant.**

### 5.1 Why

The score for head `h`, query cell `i`, key cell `j` is
`s^h_{ij} = scale·(q^h_i·k^h_j) + β[row(i,j), h]`, where `row(i,j)` is the bias row
of the offset `o_{ij} = coord(j)−coord(i)` (`build_attn_bias`, `model.py:740-767`;
`rel_bias_index`, `geometry.py:117-131`). From (H) the **content** term transforms
as

```
content^h_{ij}(T_g f) = content^{g^{-1}·h}_{g^{-1}i, g^{-1}j}(f)   # head index and cells both shift
```

so equivariance of the full score requires the **bias** to transform the same way:

```
β[ o_{ij}, h ]  =  β[ g^{-1}·o_{ij},  g^{-1}·h ]   for all g          # (B)
```

The offset transforms by `g` (`o_{gi,gj} = g·o_{ij}`) and the head by the coset
action. So `β` is a function on **(offset) × (head)** that must be invariant under
the **diagonal** action `g·(o,h) = (g·o, g·h)`. The free parameters are therefore
the **orbits of that diagonal action**, *not* `(offset-orbits) × (heads-free)`.

- Phase 2's row-orbit tie makes `β` depend only on the offset's `D6`-orbit **but
  leaves the 3 heads independent**. Under (B) the head must shift with `g` while
  the row orbit is fixed, which would force per-head equality — a contradiction
  unless the head axis is tied *in step with* the rows. Hence Phase 2 alone fails
  (**negative control: `1.13e+00`**).
- The correct table is indexed by the **joint (row, head) orbits**. Concretely:
  for each board-offset orbit `o`, its stabilizer `S_o ≤ D6` acts on the 3 heads,
  and the free head-parameters for that orbit = the **`S_o`-orbits on the 3
  heads**.

### 5.2 What the joint tie looks like (head diversity survives)

The tie is finer than "all heads equal" — it does **not** collapse the heads:

- A **generic / off-axis** offset (trivial stabilizer, `S_o = {e}`): all 3 heads
  are one `S_o`-orbit-free → **3 independent head params** (full diversity).
- A **win-axis** offset, e.g. `(2,0)` on the Q-axis (`S_o = {e, g7}`, and `g7`
  acts on heads as the transposition `Q | R↔QR`): heads split into `{Q}` and
  `{R,QR}` → **2 head params** (the Q-head free; the R/QR-heads shared).
- **Token rows** `BIAS_{CELL_TOKEN,TOKEN_CELL,TOKEN_TOKEN}` (`constants.py:123-125`)
  have a *fixed* "offset" (tokens carry no position, §6), so `S_o = D6`, whose
  action is transitive on the 3 heads → **1 head param** (all heads equal).

### 5.3 Construction (drop-in on top of Phase 2)

Precompute, per attention block, a joint LUT `joint_of[row, head] → class` by
union-finding `(row, head)` under `(o,h) ↦ (g·o, g·h)` over all 12 `g` (rows
carry the same `D6`-offset-orbit structure as Phase 2's `orbit_of_row`; heads
carry the coset action of §4). Store the free table `Θ` of shape
`(n_joint_classes, 1)` and build the effective `(BIAS_ROWS, heads)` table as
`bias_table[row, h] = Θ[joint_of[row, h]]` at the top of the bias build. Everything
downstream — `_build_pair`, `build_attn_bias`, `_BiasGather`, the flex carriers,
the Triton attn kernel (`model.py:699-841`, `_triton_attn.py`) — is unchanged; it
still sees a `(BIAS_ROWS, heads)` table. **This supersedes Phase 2's per-head-free
`(45, heads)` table:** the `45` board-orbit classes are refined into the joint
`(row, head)` classes. Keep a name containing `bias_table` for the AdamW / grad-
norm predicates (`plugin.py:38`, `trainer.py:261-266`).

---

## 6. Summary tokens

The `NUM_TOKENS` summary tokens (`model.py:614`) carry the **trivial (invariant)
subrep**: store a learned `(NUM_TOKENS, C_orbit)` and **broadcast over the 12
slots**, so `M(g)·token = token`. Token queries/keys are then head-independent, and
their bias rows are head-constant by §5.2. Value/aux/moves-left heads read tokens,
which are already invariant. (Plan §1.4: prune dead tokens 6 & 7, `NUM_TOKENS`
8→6.)

---

## 7. Heads: covariant reads and invariant reads

- **Per-cell covariant heads** — policy / opp_policy / soft_policy / cell_q
  (`_policy_logits`, `_cell_q_logits`, `model.py:1018-1038`). Each is a tied
  `HexNodeConv(C,C)` (§2) then a **fiber-invariant read**: **group-pool the fiber**
  (mean over the 12 slots → `C_orbit`) and then `Linear(C_orbit, ·)`. Group-pool
  makes the per-cell logit land in the trivial rep, so it is invariant while the
  cell index moves — verified `policy` dev `5.33e-15`.
- **Invariant heads** — value / stvalue_* / moves_left (`_value_input`,
  `model.py:1012-1016`; `_pooled`, `938-942`). **Group-pool (fiber-mean over the
  12 slots) the pooled-cell vector and the token vectors *before* the `3C→C`
  reductions**, so the reductions consume invariant `C_orbit`-vectors — verified
  `value` dev `9.94e-16`. `_pooled`'s cell-mean stays a `C`-fiber (it still
  transforms by `M(g)`); the fiber-mean is what makes it invariant.

---

## 8. Typed-lift stem for the 25 planes

The stem is a tied `HexNodeConv(25, C)` (`model.py:606`) whose **input** carries a
typed permutation rep `ρ_in(g)` (not the regular rep) and whose **output** is the
regular fiber. It satisfies `(C1)` with `M_in := ρ_in(g)`:
`W^stem_{π_g(t)} = M(g)·W^stem_t·ρ_in(g)^{-1}`.

**Input rep (plan §1.2), 25 planes.** The concrete plane indices are the shipped
`constants.py` layout (**not** contiguous by type): the **11 kept scalars occupy
indices 0-10**, the **12 axis planes indices 11-22**, and the **2 fork scalars
indices 23-24** (the fork scalars sit *after* the axis block, not adjacent to the
kept scalars). By D6 type:

- **13 scalar planes** (the 11 kept at 0-10 + the 2 fork at 23-24) = **trivial
  rep**: `ρ_in(g)` acts as the identity on these.
- **12 axis planes** (indices 11-22) = **4 copies of the 3-D axis-permutation (S3)
  module**: layout `plane = 11 + quantity·3 + axis`, `quantity ∈ {own_line,
  opp_line, own_live, opp_live}` (indices 0..3), `axis ∈ {Q,R,QR}` (indices 0..2)
  — i.e. `F_OWN_LINE_Q = 11 … F_OPP_LIVE_QR = 22`. `ρ_in(g)` permutes the axis
  index by the same 3-set action as the heads (`rot60` 3-cycles, `reflect`
  transposes) — the axis module and the head-coset module are the **same**
  `R[D6/K]`. **Do not** type the stem lift from the old `13 + quantity·3 + axis`
  form: that mis-places all 14 non-kept planes (it assumes the fork scalars precede
  the axis block) and yields a "trains fine, isn't equivariant" stem.

**Resulting free structure (verified):**

- **Scalar plane → copy into all 12 slots.** Its stem column is **slot-constant**
  (`M(g)`-fixed), one value per `(orbit channel, tap-orbit)` broadcast over the 12
  slots. Verified: scalar columns are slot-constant to `<1e-9`.
- **Axis planes → the axis module lifts into the regular fiber.** The three axis
  columns `{w_Q, w_R, w_QR}` form a `D6`-**equivariant triple**
  `M(g)·w_a = w_{axisperm(g)(a)}` (verified exactly), and each column lives in the
  `K`-fixed subspace (`w_Q` is constant on the **3 right cosets of `K`**,
  `{0,3,7,10}/{1,4,6,9}/{2,5,8,11}` — a **different** partition from the head/left
  cosets, since `K` is not normal; verified). This is a **3-dimensional** freedom
  per `(quantity, orbit channel, tap-orbit)` (= `dim Hom_{D6}(R[D6/K], R[D6]) = 3`).
- The plan's phrasing "4 copies of the 3-D axis module, lifted by the axis
  3-cycle/transposition" is realized by the **canonical coset-sum embedding** —
  axis `a`'s value copied onto the 4 slots of its **left coset** — which is the
  rank-1 element of that 3-D space and is **verified equivariant to `0.00e+00`**.
  The full tied stem-conv keeps all 3 dimensions (learned mixing); the coset-sum
  is the minimal fixed injection.

Because `ρ_in` is a permutation rep too, the stem weight is again a **pure
index-gather** with no signs, built by the same transversal construction as §2.

---

## 9. The contract (what Phase 3b must build), checklist

1. `w_base : (7, C_orbit_in, 12, C_orbit_out)` (84 blocks) + precomputed gather
   `(GEN)` `W_t[a,b] = w_base[π_{a^{-1}}(t), a^{-1}b]`; eager materialize in
   `forward`; dense cache keyed on base `_version`; fp8 conv off for v1.
2. Q/K/V/out and `fc1/fc2` as 12-block group-conv 1×1s; MLP hidden = regular fiber
   width `MLP_RATIO·C_orbit`; slot-constant biases.
3. Group-norm at all 8+ LN sites + `LayerScale`: full-fiber normalize, affine
   **tied per `C_orbit`** (broadcast over 12 slots).
4. Head split aligned to left cosets of `K` (order 4 ⇒ 3 heads = axes, head_dim =
   `4·C_orbit`); reorder channels so the reshape lands heads on cosets.
5. Relative bias tied over the **joint (row, head) orbits** (§5) — supersedes the
   Phase-2 per-head-free `(45, heads)` table; token rows head-constant.
6. Tokens = learned `(NUM_TOKENS, C_orbit)` broadcast over slots (invariant);
   prune tokens 6 & 7.
7. Heads: per-cell covariant heads group-pool then `Linear(C_orbit,·)`; invariant
   heads group-pool tokens + pooled cells before the `3C→C` reductions.
8. Typed-lift stem: scalars copy-to-12; axis planes lift by the equivariant triple
   / coset-sum; tied stem-conv via the same gather.

**Exit gate — PASSED.** `f(g·board) == g·f(board)` for all 12 `g` to machine
precision (`tests/test_hexfield_eq_derivation.py`), the tied-conv/qkv weight
constraints hold exactly, the free-block count is `84 = 7×12`, and the
row-orbit-only bias (Phase 2 alone) is shown to break equivariance — so the
joint (row, head) tie is mandatory.
