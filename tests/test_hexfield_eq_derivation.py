"""Executable contract for the Phase-3a D6-equivariant-attention derivation
(docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md).

A self-contained numpy prototype of the *tied* trunk block (typed-lift stem +
tied HexNodeConv + group-norm + tied Q/K/V/out with a coset-aligned head split
+ a JOINTLY (offset-row, head)-tied relative bias + group-pooled heads), on a
small regular-rep fiber over a G-closed hex disk.  It asserts

    f(g . board) == g . f(board)     for all 12 g in D6

to fp32 tolerance, and asserts the NEGATIVE CONTROL — a bias tied only over
board orbits but left free across heads (Phase-2's table on its own) — BREAKS
equivariance.  No package import (packages/hexfield_eq does not exist yet at
Phase 3a); numpy only.  This is the sole correctness guarantee behind the
augmentation-free expand path, so it lives in the regression suite.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- group D6 ----
DIRECTIONS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


def _rot60(q, r):
    return (-r, q + r)


def _reflect(q, r):
    return (q, -q - r)


def _apply_d6(index, q, r):
    if index >= 6:
        q, r = _reflect(q, r)
        index -= 6
    for _ in range(index):
        q, r = _rot60(q, r)
    return (q, r)


def _build_group():
    E = list(range(12))
    act = lambda g, c: _apply_d6(g, c[0], c[1])
    sig = {(act(g, (1, 0)), act(g, (0, 1))): g for g in E}
    mult = [[sig[(act(a, act(b, (1, 0))), act(a, act(b, (0, 1))))] for b in E] for a in E]
    inv = [next(b for b in E if mult[a][b] == 0) for a in E]
    taps = [(0, 0)] + list(DIRECTIONS)
    ti = {o: i for i, o in enumerate(taps)}
    tapp = [[ti[act(g, taps[t])] for t in range(7)] for g in E]
    regp = [[mult[inv[g]][k] for k in E] for g in E]  # out[k] = in[regp[g][k]]
    cosets = [[0, 3, 7, 10], [1, 4, 8, 11], [2, 5, 6, 9]]  # left cosets of stab(Q) = heads
    cos_of = [None] * 12
    for ci, c in enumerate(cosets):
        for x in c:
            cos_of[x] = ci
    cosp = [[cos_of[mult[g][cosets[c][0]]] for c in range(3)] for g in E]
    return dict(E=E, act=act, mult=mult, inv=inv, tapp=tapp, regp=regp,
                cosets=cosets, cosp=cosp)


G = _build_group()
CORB = 2
C = 12 * CORB
NF = 25
HEADS = 3


def _board(R=2):
    hexdist = lambda q, r: max(abs(q), abs(r), abs(q + r))
    coords = sorted(
        (q, r) for q in range(-R, R + 1) for r in range(-R, R + 1) if hexdist(q, r) <= R
    )
    N = len(coords)
    cidx = {c: i for i, c in enumerate(coords)}
    gather = np.full((N, 7), N, dtype=int)
    for i, c in enumerate(coords):
        gather[i, 0] = i
        for d, off in enumerate(DIRECTIONS):
            nb = (c[0] + off[0], c[1] + off[1])
            if nb in cidx:
                gather[i, d + 1] = cidx[nb]
    sig = [[cidx[G["act"](g, coords[i])] for i in range(N)] for g in G["E"]]
    return coords, N, gather, sig


COORDS, N, GATHER, SIG = _board()


# ------------------------------------------------------------ rep actions ----
def _reg_field(g, field):  # (N,12,CORB) regular-rep -> g.field
    out = np.empty_like(field)
    out[SIG[g]] = field[:, G["regp"][g], :]
    return out


def _scalar_field(g, field):
    out = np.empty_like(field)
    out[SIG[g]] = field
    return out


def _Rin(g):
    M = np.zeros((NF, NF))
    for p in range(13):
        M[p, p] = 1.0
    for q in range(4):
        for a in range(3):
            M[13 + q * 3 + G["cosp"][g][a], 13 + q * 3 + a] = 1.0
    return M


def _in_field(g, field):
    out = np.empty_like(field)
    out[SIG[g]] = field @ _Rin(g).T
    return out


# ------------------------------------------------------------ tied layers ----
def _gen_conv_weight(wbase):  # (7,12,CORB,CORB) -> (7,C,C)
    W = np.zeros((7, C, C))
    for t in range(7):
        for a in range(12):
            ta = G["tapp"][G["inv"][a]][t]
            for b in range(12):
                s = G["mult"][G["inv"][a]][b]
                W[t, a * CORB:(a + 1) * CORB, b * CORB:(b + 1) * CORB] = wbase[ta, s]
    return W


def _conv(W, bias, field):
    ext = np.concatenate([field, np.zeros((1, field.shape[1]))], 0)
    out = np.zeros((N, W.shape[1]))
    for t in range(7):
        out += ext[GATHER[:, t]] @ W[t].T
    return out + bias


def _gen_1x1(wb):  # 12-block group-conv
    W = np.zeros((C, C))
    for a in range(12):
        for b in range(12):
            s = G["mult"][G["inv"][a]][b]
            W[a * CORB:(a + 1) * CORB, b * CORB:(b + 1) * CORB] = wb[s]
    return W


def _groupnorm(x, gamma, beta, eps=1e-5):
    xn = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + eps)
    return xn * np.tile(gamma, 12) + np.tile(beta, 12)


HEADCH = [[sl * CORB + o for sl in G["cosets"][h] for o in range(CORB)] for h in range(3)]
HD = len(HEADCH[0])
SCALE = 1.0 / np.sqrt(HD)


def _attention(x, wq, wk, wv, wo, biasmat):
    q, k, v = x @ _gen_1x1(wq).T, x @ _gen_1x1(wk).T, x @ _gen_1x1(wv).T
    out = np.zeros_like(x)
    for h in range(HEADS):
        ch = HEADCH[h]
        sc = SCALE * (q[:, ch] @ k[:, ch].T) + biasmat[:, :, h]
        sc = sc - sc.max(1, keepdims=True)
        w = np.exp(sc)
        w /= w.sum(1, keepdims=True)
        out[:, ch] = w @ v[:, ch]
    return out @ _gen_1x1(wo).T


# ------------------------------------------------------- bias construction ----
def _offsets():
    offs = set()
    for i in range(N):
        for j in range(N):
            offs.add((COORDS[j][0] - COORDS[i][0], COORDS[j][1] - COORDS[i][1]))
    return offs


def _joint_bias(rng):
    """(offset,head)-orbit tie under the DIAGONAL action  (o,h)->(g.o, g.h)."""
    offs = _offsets()
    nodes = [(o, h) for o in offs for h in range(3)]
    nid = {n: k for k, n in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (o, h) in nodes:
        for g in G["E"]:
            a, b = find(nid[(o, h)]), find(nid[(G["act"](g, o), G["cosp"][g][h])])
            if a != b:
                parent[a] = b
    cls = {}
    for n in nodes:
        cls.setdefault(find(nid[n]), len(cls))
    theta = rng.standard_normal(len(cls))
    B = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(N):
            o = (COORDS[j][0] - COORDS[i][0], COORDS[j][1] - COORDS[i][1])
            for h in range(3):
                B[i, j, h] = theta[cls[find(nid[(o, h)])]]
    return B, len(cls)


def _row_only_bias(rng):
    """NEGATIVE CONTROL: tie over board orbits only, per-head FREE (Phase-2)."""
    offs = list(_offsets())
    oid = {o: k for k, o in enumerate(offs)}
    parent = list(range(len(offs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for o in offs:
        for g in G["E"]:
            a, b = find(oid[o]), find(oid[G["act"](g, o)])
            if a != b:
                parent[a] = b
    cls = {}
    for o in offs:
        cls.setdefault(find(oid[o]), len(cls))
    phi = rng.standard_normal((len(cls), 3))
    B = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(N):
            o = (COORDS[j][0] - COORDS[i][0], COORDS[j][1] - COORDS[i][1])
            for h in range(3):
                B[i, j, h] = phi[cls[find(oid[o])], h]
    return B


# ------------------------------------------------------------- stem lift  ----
def _Mreg(g):
    M = np.zeros((C, C))
    for k in range(12):
        M[k * CORB:(k + 1) * CORB, G["regp"][g][k] * CORB:(G["regp"][g][k] + 1) * CORB] = np.eye(CORB)
    return M


def _stem_project(W0):  # Reynolds projection onto the equivariant subspace
    Ws = np.zeros_like(W0)
    for g in G["E"]:
        Mg, Rg = _Mreg(g), np.linalg.inv(_Rin(g))
        for t in range(7):
            Ws[t] += Mg @ W0[G["tapp"][G["inv"][g]][t]] @ Rg
    return Ws / 12.0


def _build(rng):
    W = {}
    W["stem"] = _stem_project(rng.standard_normal((7, C, NF)))
    W["sbias"] = np.tile(rng.standard_normal(CORB), 12)
    W["conv"] = _gen_conv_weight(rng.standard_normal((7, 12, CORB, CORB)))
    W["cbias"] = np.tile(rng.standard_normal(CORB), 12)
    W["gn_g"], W["gn_b"] = rng.standard_normal(CORB), rng.standard_normal(CORB)
    W["ls"] = rng.standard_normal(CORB) * 0.1
    for nm in ("wq", "wk", "wv", "wo"):
        W[nm] = rng.standard_normal((12, CORB, CORB))
    return W


def _forward(field25, W, biasmat):
    ext = np.concatenate([field25, np.zeros((1, NF))], 0)
    x = np.zeros((N, C))
    for t in range(7):
        x += ext[GATHER[:, t]] @ W["stem"][t].T
    x = np.maximum(x + W["sbias"], 0)
    y = np.maximum(_groupnorm(_conv(W["conv"], W["cbias"], x), W["gn_g"], W["gn_b"]), 0)
    x = x + y * np.tile(W["ls"], 12)
    xn = _groupnorm(x, W["gn_g"], W["gn_b"])
    x = x + _attention(xn, W["wq"], W["wk"], W["wv"], W["wo"], biasmat) * np.tile(W["ls"], 12)
    return x


def _group_pool(x):
    return x.reshape(N, 12, CORB).mean(1)


# --------------------------------------------------------------- the tests ---
TOL = 1e-9


def test_full_block_is_d6_equivariant():
    rng = np.random.default_rng(0)
    W = _build(rng)
    Bj, _ = _joint_bias(rng)
    field = np.random.default_rng(1).standard_normal((N, NF))
    base = _forward(field, W, Bj)
    wp = np.random.default_rng(2).standard_normal((CORB, 1))
    pol = _group_pool(np.maximum(base, 0)) @ wp
    wv = np.random.default_rng(3).standard_normal((CORB, 1))
    val = base.mean(0).reshape(12, CORB).mean(0) @ wv
    worst_t = worst_p = worst_v = 0.0
    for g in G["E"]:
        fg = _forward(_in_field(g, field), W, Bj)  # f(g.x)
        gf = _reg_field(g, base.reshape(N, 12, CORB)).reshape(N, C)  # g.f(x)
        worst_t = max(worst_t, float(np.abs(fg - gf).max()))
        polg = _group_pool(np.maximum(fg, 0)) @ wp
        valg = fg.mean(0).reshape(12, CORB).mean(0) @ wv
        worst_p = max(worst_p, float(np.abs(polg - _scalar_field(g, pol)).max()))
        worst_v = max(worst_v, float(np.abs(valg - val).max()))
    assert worst_t < TOL, worst_t
    assert worst_p < TOL, worst_p
    assert worst_v < TOL, worst_v


def test_row_only_bias_breaks_equivariance():
    """Phase-2's board-orbit tie, left per-head free, is NOT enough: the head
    axis must be tied jointly with the row orbit."""
    rng = np.random.default_rng(0)
    W = _build(rng)
    Bbad = _row_only_bias(rng)
    field = np.random.default_rng(1).standard_normal((N, NF))
    base = _forward(field, W, Bbad)
    worst = 0.0
    for g in G["E"]:
        fg = _forward(_in_field(g, field), W, Bbad)
        gf = _reg_field(g, base.reshape(N, 12, CORB)).reshape(N, C)
        worst = max(worst, float(np.abs(fg - gf).max()))
    assert worst > 1e-3, worst  # must visibly break


def test_tied_conv_weight_constraint_and_block_count():
    rng = np.random.default_rng(0)
    Wconv = _gen_conv_weight(rng.standard_normal((7, 12, CORB, CORB)))
    worst = 0.0
    for g in G["E"]:
        Mg = _Mreg(g)
        Mgi = np.linalg.inv(Mg)
        for t in range(7):
            worst = max(worst, float(np.abs(Wconv[G["tapp"][g][t]] - Mg @ Wconv[t] @ Mgi).max()))
    assert worst < TOL, worst
    reach = {
        (G["tapp"][G["inv"][a]][t], G["mult"][G["inv"][a]][b])
        for t in range(7) for a in range(12) for b in range(12)
    }
    assert len(reach) == 84  # == 7 taps x 12, the w_base storage budget


def test_qkv_projections_commute_with_regular_rep():
    rng = np.random.default_rng(0)
    Wq = _gen_1x1(rng.standard_normal((12, CORB, CORB)))
    worst = max(float(np.abs(Wq @ _Mreg(g) - _Mreg(g) @ Wq).max()) for g in G["E"])
    assert worst < TOL, worst


def test_stem_typed_lift_structure():
    rng = np.random.default_rng(0)
    Wst = _stem_project(rng.standard_normal((7, C, NF)))
    # scalar planes copy into all 12 slots (slot-constant column)
    for p in range(13):
        col = Wst[0, :, p].reshape(12, CORB)
        assert np.abs(col - col.mean(0, keepdims=True)).max() < TOL
    # axis columns form an equivariant triple M_reg(g) w_a == w_{axisperm(g)(a)}
    for q in range(4):
        cols = [Wst[0, :, 13 + q * 3 + a].reshape(12, CORB) for a in range(3)]
        for g in G["E"]:
            for a in range(3):
                assert np.abs(cols[a][G["regp"][g]] - cols[G["cosp"][g][a]]).max() < TOL


# ------------------------------------------------- shipped plane maps (§8) ----
# The tests above use a toy layout (13 scalars contiguous, axis planes after).
# The SHIPPED maps interleave: the axis block starts at 11 and the fork scalars
# sit AFTER it — 23/24 under HEXFIELD_EQ_FEATURE_VERSION=1; under version 2 the
# fork planes RE-INDEX to 41/42 behind the liveK block and 3 global scalars
# append (SPEC_RAYTAP_CONV.md §1.2). The §8 stem-lift structure must hold
# against both real maps; a stale scalar set (forks left at 23/24 under the
# 46-plane map) mistypes real planes — it trains fine but is not equivariant.

REAL_PLANE_MAPS = (
    # (nf, n_axis_quantities, scalar planes)
    (25, 4, tuple(range(11)) + (23, 24)),
    (46, 10, tuple(range(11)) + tuple(range(41, 46))),
)


def _rin_for_map(g, nf, n_axis_q, scalars):
    M = np.zeros((nf, nf))
    for p in scalars:
        M[p, p] = 1.0
    for q in range(n_axis_q):
        for a in range(3):
            M[11 + q * 3 + G["cosp"][g][a], 11 + q * 3 + a] = 1.0
    return M


def _stem_project_for_map(W0, nf, n_axis_q, scalars):
    Ws = np.zeros_like(W0)
    for g in G["E"]:
        Mg = _Mreg(g)
        Rg = np.linalg.inv(_rin_for_map(g, nf, n_axis_q, scalars))
        for t in range(7):
            Ws[t] += Mg @ W0[G["tapp"][G["inv"][g]][t]] @ Rg
    return Ws / 12.0


def test_stem_typed_lift_structure_real_plane_maps():
    for seed, (nf, n_axis_q, scalars) in enumerate(REAL_PLANE_MAPS):
        # the map is a partition of the planes
        assert set(scalars) | {11 + k for k in range(3 * n_axis_q)} == set(range(nf))
        W0 = np.random.default_rng(10 + seed).standard_normal((7, C, nf))
        Ws = _stem_project_for_map(W0, nf, n_axis_q, scalars)
        for p in scalars:
            col = Ws[0, :, p].reshape(12, CORB)
            assert np.abs(col - col.mean(0, keepdims=True)).max() < TOL, (nf, p)
        for q in range(n_axis_q):
            cols = [Ws[0, :, 11 + q * 3 + a].reshape(12, CORB) for a in range(3)]
            for g in G["E"]:
                for a in range(3):
                    assert (
                        np.abs(cols[a][G["regp"][g]] - cols[G["cosp"][g][a]]).max() < TOL
                    ), (nf, q, g, a)
