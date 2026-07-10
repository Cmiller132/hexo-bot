"""D6 regular-representation equivariant primitives for the hexfield_eq trunk.

Ports the verified numpy prototype (``tests/test_hexfield_eq_derivation.py``,
the exit gate for docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md) to torch. Every
representation here is a *permutation* rep, so every tied weight is a pure
index-gather with all signs +1 (§0.1 of the derivation). The model stores small
orbit "base" params and materializes the dense ``(7, C_in, C_out)`` conv weight
/ ``(C_out, C_in)`` linear weight each forward via these gathers, so the
reference GEMM and the fp16 Triton conv/attn kernels consume the generated
weight unchanged.

Layout convention (slot-major): a channel is indexed ``c = slot*C_ORBIT + a``
with fiber slot ``slot in 0..11`` (a D6 element) and orbit channel
``a in 0..C_ORBIT-1``. The regular rep acts on the fiber by left-multiplying the
slot label.

This module is imported ONLY when the equivariant build is active
(GROUP_ORDER == 12). It hardcodes the full D6 group of order 12; the 3 heads are
the left cosets of the order-4 subgroup K = stab(Q-axis).
"""

from __future__ import annotations

import functools

import torch

from .constants import (
    AXIS_PLANE_BASE,
    BIAS_DISK_RADIUS,
    BIAS_EXACT_ROWS,
    BIAS_ROWS,
    C_ORBIT,
    CHANNELS,
    DIRECTIONS,
    FEATURE_VERSION,
    N_AXES,
    N_AXIS_QUANTITIES,
    NUM_FEATURES,
)
from .geometry import apply_d6, d6_inverse, disk_offsets, rel_bias_index

GROUP = 12  # full D6

# The input planes carry a typed permutation rep (docs/DERIVATION §8): the
# scalar planes (trivial rep) + the axis planes = N_AXIS_QUANTITIES quantities
# x 3 axes, the axis index carrying the same 3-set (coset) action as the heads.
# In the real feature layout (constants.py) the axis planes are contiguous
# starting at 11; the fork scalars sit AFTER them (23/24 under FEATURE_VERSION
# 1, 41/42 under version 2 — spec §1.2 fork re-index), so the scalar planes are
# NOT contiguous. The plane-map geometry (AXIS_PLANE_BASE, N_AXIS_QUANTITIES,
# N_AXES) is version-dependent and owned by constants.py; the typing sets here
# derive from it so every consumer regenerates against the active map.
_AXIS_PLANES = set(range(AXIS_PLANE_BASE, AXIS_PLANE_BASE + N_AXIS_QUANTITIES * N_AXES))
_SCALAR_PLANES = [p for p in range(NUM_FEATURES) if p not in _AXIS_PLANES]
assert len(_SCALAR_PLANES) == (13 if FEATURE_VERSION == 1 else 16), _SCALAR_PLANES


@functools.lru_cache(maxsize=1)
def build_group() -> dict:
    """Group tables for D6 (order 12), indexed exactly as ``geometry.apply_d6``.

    Returns a dict with:
      mult[a][b]   -- composition index of (a then... ) i.e. a*b
      inv[a]       -- inverse index
      tapp[g][t]   -- tap permutation pi_g on the 7 taps (0=center, 1..6=DIRECTIONS)
      regp[g][k]   -- regular-rep gather: (rho_reg(g) v)[k] = v[regp[g][k]]
      cosets       -- the 3 left cosets of K=stab(Q) (the 3 heads)
      cos_of[x]    -- coset (head) index of element x
      cosp[g][c]   -- coset action: head c -> head cosp[g][c]
    """

    E = list(range(GROUP))
    act = lambda g, c: apply_d6(g, c[0], c[1])
    # Recover the composition table from the faithful action on the two axis
    # generators (matches the prototype's sig trick).
    sig = {(act(g, (1, 0)), act(g, (0, 1))): g for g in E}
    mult = [
        [sig[(act(a, act(b, (1, 0))), act(a, act(b, (0, 1))))] for b in E] for a in E
    ]
    inv = [d6_inverse(a) for a in E]
    taps = [(0, 0)] + list(DIRECTIONS)
    ti = {o: i for i, o in enumerate(taps)}
    tapp = [[ti[act(g, taps[t])] for t in range(7)] for g in E]
    regp = [[mult[inv[g]][k] for k in E] for g in E]  # out[k] = in[regp[g][k]]
    # Left cosets of K = stab(Q-axis) = {e, rot180, g7, g10} (an order-4 Klein
    # subgroup): the 3 cosets are the 3 win-axes (rot60 3-cycles them).
    cosets = [[0, 3, 7, 10], [1, 4, 8, 11], [2, 5, 6, 9]]
    cos_of = [None] * GROUP
    for ci, c in enumerate(cosets):
        for x in c:
            cos_of[x] = ci
    cosp = [[cos_of[mult[g][cosets[c][0]]] for c in range(N_AXES)] for g in E]
    return dict(
        mult=mult, inv=inv, tapp=tapp, regp=regp, cosets=cosets, cos_of=cos_of, cosp=cosp
    )


# --- weight-generation gather indices (built once) ----------------------------


@functools.lru_cache(maxsize=1)
def conv_gather_index() -> torch.Tensor:
    """(7, 12, 12) long flat index into a w_base flattened to (7*12, ...).

    For output slot a, input slot b, tap t the generated dense-conv block equals
    ``w_base[pi_{a^-1}(t), a^-1 * b]`` (docs/DERIVATION (GEN)); this returns
    ``flat[t,a,b] = tapp[inv[a]][t]*12 + mult[inv[a]][b]``.
    """

    G = build_group()
    inv, tapp, mult = G["inv"], G["tapp"], G["mult"]
    idx = torch.empty((7, GROUP, GROUP), dtype=torch.long)
    for t in range(7):
        for a in range(GROUP):
            for b in range(GROUP):
                idx[t, a, b] = tapp[inv[a]][t] * GROUP + mult[inv[a]][b]
    return idx


@functools.lru_cache(maxsize=1)
def linear_gather_index() -> torch.Tensor:
    """(12, 12) long index into a wb of shape (12, ...): the center-tap (1x1)
    group-convolution ``W[out=a, in=b] = wb[a^-1 * b]`` (docs/DERIVATION §2.4)."""

    G = build_group()
    inv, mult = G["inv"], G["mult"]
    idx = torch.empty((GROUP, GROUP), dtype=torch.long)
    for a in range(GROUP):
        for b in range(GROUP):
            idx[a, b] = mult[inv[a]][b]
    return idx


def gen_conv_weight(w_base: torch.Tensor, gather: torch.Tensor) -> torch.Tensor:
    """Materialize the dense conv weight in torch layout (7, C_in, C_out).

    ``w_base`` is (7, 12, C_orbit_out, C_orbit_in) indexed [tap, out_slot,
    orbit_out, orbit_in]; ``gather`` is :func:`conv_gather_index`. The reference
    GEMM computes ``gathered(B,N,7*C_in) @ weight.reshape(7*C_in, C_out)``, so
    weight[t] is (C_in, C_out)."""

    corb_out, corb_in = w_base.shape[2], w_base.shape[3]
    w_flat = w_base.reshape(7 * GROUP, corb_out, corb_in)
    gathered = w_flat[gather]  # (7, a, b, i=orbit_out, j=orbit_in)
    # torch weight[t, in=b*corb_in+j, out=a*corb_out+i]
    return gathered.permute(0, 2, 4, 1, 3).reshape(
        7, GROUP * corb_in, GROUP * corb_out
    )


def gen_linear_weight(wb: torch.Tensor, gather: torch.Tensor) -> torch.Tensor:
    """Materialize the dense nn.Linear weight (C_out, C_in) for a tied 1x1.

    ``wb`` is (12, C_orbit_out, C_orbit_in); ``gather`` is
    :func:`linear_gather_index`. nn.Linear computes ``x @ weight.T`` so the
    weight is (out, in)."""

    corb_out, corb_in = wb.shape[1], wb.shape[2]
    gathered = wb[gather]  # (a, b, i=orbit_out, j=orbit_in)
    # weight[out=a*corb_out+i, in=b*corb_in+j]
    return gathered.permute(0, 2, 1, 3).reshape(GROUP * corb_out, GROUP * corb_in)


# --- stem typed-lift (Reynolds projection onto the equivariant subspace) -------


@functools.lru_cache(maxsize=1)
def _reg_matrix() -> torch.Tensor:
    """(12, C, C) dense M_reg(g) = rho_reg(g) (x) I_{C_ORBIT} permutation matrices."""

    G = build_group()
    regp = G["regp"]
    C = CHANNELS
    M = torch.zeros((GROUP, C, C))
    eye = torch.eye(C_ORBIT)
    for g in range(GROUP):
        for k in range(GROUP):
            src = regp[g][k]
            M[g, k * C_ORBIT:(k + 1) * C_ORBIT, src * C_ORBIT:(src + 1) * C_ORBIT] = eye
    return M


@functools.lru_cache(maxsize=1)
def _in_rep_matrix() -> torch.Tensor:
    """(12, NF, NF) dense rho_in(g): the scalar planes fixed, the axis planes
    permuted by the coset action (plane BASE+q*3+a -> BASE+q*3+cosp[g][a],
    q over the version's N_AXIS_QUANTITIES)."""

    G = build_group()
    cosp = G["cosp"]
    NF = NUM_FEATURES
    M = torch.zeros((GROUP, NF, NF))
    for g in range(GROUP):
        for p in _SCALAR_PLANES:
            M[g, p, p] = 1.0
        for q in range(N_AXIS_QUANTITIES):
            for a in range(N_AXES):
                dst = AXIS_PLANE_BASE + q * N_AXES + cosp[g][a]
                src = AXIS_PLANE_BASE + q * N_AXES + a
                M[g, dst, src] = 1.0
    return M


@functools.lru_cache(maxsize=1)
def _tapp_inv_index() -> list:
    """tapp[inv[g]] as a plain nested list, for the stem Reynolds sum."""

    G = build_group()
    inv, tapp = G["inv"], G["tapp"]
    return [tapp[inv[g]] for g in range(GROUP)]


def gen_stem_weight(w0: torch.Tensor) -> torch.Tensor:
    """Materialize the typed-lift stem weight in torch layout (7, NF, C).

    ``w0`` is (7, C, NF) free params; the returned weight is the Reynolds
    projection onto the equivariant subspace,
    ``Ws[t] = (1/12) sum_g M_reg(g) @ w0[pi_{g^-1}(t)] @ rho_in(g)^-1`` (§8),
    transposed to the reference-GEMM's (7, in=NF, out=C) layout."""

    Mreg = _reg_matrix().to(w0.dtype).to(w0.device)          # (12, C, C)
    Rin = _in_rep_matrix().to(w0.dtype).to(w0.device)        # (12, NF, NF)
    tapp_inv = _tapp_inv_index()
    Ws = torch.zeros_like(w0)  # (7, C, NF)
    for g in range(GROUP):
        wt = w0[tapp_inv[g]]                                  # (7, C, NF)
        # M_reg(g) @ wt[t] @ Rin(g)^-1 ; Rin permutation so inverse == transpose.
        Ws = Ws + torch.einsum("oc,tcn,mn->tom", Mreg[g], wt, Rin[g])
    Ws = Ws / GROUP
    return Ws.transpose(1, 2).contiguous()                    # (7, NF, C)


# --- head coset channel permutation -------------------------------------------


@functools.lru_cache(maxsize=1)
def head_perm() -> torch.Tensor:
    """(C,) long: reorder slot-major channels into coset-grouped order so a
    ``reshape(..., heads=3, head_dim=4*C_ORBIT)`` lands each head on one
    win-axis coset's channels (docs/DERIVATION §4)."""

    G = build_group()
    cosets = G["cosets"]
    order = [
        slot * C_ORBIT + o
        for h in range(N_AXES)
        for slot in cosets[h]
        for o in range(C_ORBIT)
    ]
    return torch.tensor(order, dtype=torch.long)


@functools.lru_cache(maxsize=1)
def head_perm_inv() -> torch.Tensor:
    """Inverse of :func:`head_perm` (coset-grouped -> slot-major)."""

    perm = head_perm()
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    return inv


@functools.lru_cache(maxsize=1)
def head_perm6() -> torch.Tensor:
    """(C,) long: coset-major, then orbit-half channel order so a
    ``reshape(..., heads=6, head_dim=2*C_ORBIT)`` lands head ``2*coset + half``
    on the (4 K-slots) x (C_ORBIT/2 orbit-half) channels of its win-axis coset
    (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md L4).

    The own/opp sub-head split MUST ride the orbit index: ``M(g) = rho_reg(g)
    (x) I_{C_ORBIT}`` left-multiplies the slot and FIXES the orbit index, so any
    orbit partition is a D6-block system (head ``2c+s -> 2*cosp[g](c)+s`` with
    an internal ``P_K(g) (x) I`` permutation). K acting on itself by left
    multiplication is simply transitive, so EVERY 2+2 partition of the 4
    K-slots is broken by some ``P_K(g)`` — a slot split silently mixes the two
    sub-heads under reflection/rotation. Requires C_ORBIT even (validated at
    import for 'L' layouts)."""

    G = build_group()
    cosets = G["cosets"]
    half = C_ORBIT // 2
    order = [
        slot * C_ORBIT + h * half + o
        for c in range(N_AXES)
        for h in range(2)
        for slot in cosets[c]
        for o in range(half)
    ]
    return torch.tensor(order, dtype=torch.long)


@functools.lru_cache(maxsize=1)
def head_perm6_inv() -> torch.Tensor:
    """Inverse of :func:`head_perm6` (coset/half-grouped -> slot-major)."""

    perm = head_perm6()
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    return inv


# --- jointly (row, head)-tied relative-position bias ---------------------------


@functools.lru_cache(maxsize=1)
def joint_bias_lut() -> tuple[torch.Tensor, int]:
    """(joint_of (BIAS_ROWS, 3) long, n_classes int).

    The per-head bias must tie JOINTLY across (offset-row, head) under the
    diagonal action ``(o,h) -> (g.o, g.h)`` (docs/DERIVATION §5): Phase-2's
    board-orbit tie with heads left free is NOT equivariant. Disk rows carry the
    offset action (row -> rel_bias_index(g.offset)); the ring/far/token rows are
    D6-invariant buckets (row fixed) but the head still permutes, so each
    collapses to a single head-constant class."""

    offsets = disk_offsets(BIAS_DISK_RADIUS)  # BIAS_EXACT_ROWS offsets, in row order
    assert len(offsets) == BIAS_EXACT_ROWS
    G = build_group()
    cosp = G["cosp"]

    n_nodes = BIAS_ROWS * N_AXES
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    def nid(row: int, h: int) -> int:
        return row * N_AXES + h

    for row in range(BIAS_ROWS):
        for g in range(GROUP):
            if row < BIAS_EXACT_ROWS:
                oq, orr = offsets[row]
                gq, gr = apply_d6(g, oq, orr)
                row_g = rel_bias_index(gq, gr)  # stays in disk (D6 preserves dist)
            else:
                row_g = row  # ring/far/token buckets are D6-fixed
            for h in range(N_AXES):
                union(nid(row, h), nid(row_g, cosp[g][h]))

    classes: dict[int, int] = {}
    joint = torch.empty((BIAS_ROWS, N_AXES), dtype=torch.long)
    for row in range(BIAS_ROWS):
        for h in range(N_AXES):
            r = find(nid(row, h))
            cls = classes.get(r)
            if cls is None:
                cls = len(classes)
                classes[r] = cls
            joint[row, h] = cls
    return joint, len(classes)


def group_pool(x: torch.Tensor) -> torch.Tensor:
    """Fiber-mean over the 12 slots: (..., k*C) -> (..., k*C_ORBIT). Makes a
    covariant regular fiber invariant (the trivial subrep read for the heads).
    Generic over the orbit width so the widened invariant reads (EquivLinear
    C -> k*C expansions, spec D-S20) pool the same way as the plain fiber."""

    return x.reshape(*x.shape[:-1], GROUP, x.shape[-1] // GROUP).mean(-2)
