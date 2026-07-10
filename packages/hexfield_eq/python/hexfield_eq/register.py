"""Register lane (docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase R).

``RegisterRefresh`` is a one-way sigmoid-gated cross-attention refreshing the
summary tokens from the cells at every C-block exit: the aggregation is an
UNNORMALIZED SUM (no softmax normalizer), so a token accumulates "number of
cells matching pattern q" — the counting primitive softmax lacks (plan R1).
``TokenRead`` is the optional cells <- tokens broadcast read at C-block entry
(plan R4). Both are numerical no-ops at step 0 (plan R3; out_proj near-zero at
std 3e-3 so every lane gradient stays live — spec D-S22) and exactly
D6-equivariant in the tied build: the coset head split and channel permutation
mirror RelPosAttention, and the per-token gate threshold is head-constant (the
S_o = D6 case of docs/DERIVATION §5.2/§6).

Imported lazily by model.HexfieldNet only when the lane is on (the module reuses
model's EquivLinear/_make_norm, so a top-level import in both directions would
be a cycle).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .constants import ATTENTION_HEADS, NUM_TOKENS, REG_SUM_SCALE
from .model import EQUIVARIANT, EquivLinear, _make_norm

if EQUIVARIANT:
    from . import equivariant as _eq


class RegisterRefresh(nn.Module):
    """One-way sigmoid-gated cross-attention: tokens read cells, SUM-aggregated
    in fp32. Near-zero-init out_proj (std 3e-3, spec D-S22) => the lane is
    numerically a no-op at step 0 while every lane gradient stays live."""

    def __init__(self, channels: int, heads: int | None = None) -> None:
        super().__init__()
        # heads defaults to the module-global ATTENTION_HEADS; an explicit value
        # builds the refresh at a different head count (foreign-arch loaders),
        # mirroring RelPosAttention.
        self.heads = ATTENTION_HEADS if heads is None else int(heads)
        if channels % self.heads != 0:
            raise ValueError(
                f"RegisterRefresh: channels ({channels}) must be divisible by "
                f"heads ({self.heads}); a foreign-arch rebuild must pass the "
                "checkpoint's own head count (see infer_net_kwargs_from_state_dict)"
            )
        self.head_dim = channels // self.heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.equivariant = EQUIVARIANT
        linear = EquivLinear if EQUIVARIANT else nn.Linear
        self.ln_q = _make_norm(channels)  # pre-norm on tokens (q input)
        self.ln_kv = _make_norm(channels)  # pre-norm on cells (k/v input)
        self.q_proj = linear(channels, channels)
        self.k_proj = linear(channels, channels)
        self.v_proj = linear(channels, channels)
        self.out_proj = linear(channels, channels)
        # Per-token gate threshold added pre-sigmoid, broadcast over heads.
        # Head-constancy is an equivariance requirement: tokens carry no
        # position, so their score rows sit in the S_o = D6 case of the joint
        # (row, head) tie (plan R2). Init -2.5 (background gate ~ 0.076, spec
        # D-S23: the -1.0 original made every token a board-size integrator at
        # init via its 0.27 background gate).
        self.gate_bias = nn.Parameter(torch.full((NUM_TOKENS,), -2.5))
        # Learnable invariant scale on the summed update (spec D-S24), init at
        # the REG_SUM_SCALE constant: blocks can rescale their count magnitudes
        # as board size grows. 0-dim => no-decay by ndim; trunk_reg grad group.
        self.sum_scale = nn.Parameter(torch.tensor(float(REG_SUM_SCALE)))
        if EQUIVARIANT:
            self.register_buffer("_head_perm", _eq.head_perm(), persistent=False)
            self.register_buffer(
                "_head_perm_inv", _eq.head_perm_inv(), persistent=False
            )
            # Fold the §4 coset perm into the projections' serve weight cache
            # (mirrors RelPosAttention). Static geometry, so calling before the
            # weight (re-)init in _init_projections() below is harmless — only the
            # empty fold cache is dropped.
            self.q_proj.set_serve_perms(out_perm=self._head_perm)
            self.k_proj.set_serve_perms(out_perm=self._head_perm)
            self.v_proj.set_serve_perms(out_perm=self._head_perm)
            self.out_proj.set_serve_perms(in_perm=self._head_perm)
        self._init_projections()

    def _init_projections(self) -> None:
        """q/k/v: the trunk Linear init (trunc_normal 0.02, zero bias); out_proj
        NEAR-zero — trunc_normal std 3e-3, zero bias (spec D-S22): still a
        numerical no-op at step 0 but not EXACTLY zero, which would make the
        q/k/v/gate_bias gradients provably zero (every path chains through
        W_out) and stall the lane's grow-in. Self-contained so construction
        order relative to HexfieldNet._init_weights cannot clobber it."""

        for proj in (self.q_proj, self.k_proj, self.v_proj):
            if self.equivariant:
                nn.init.trunc_normal_(proj.wb, std=0.02)
                nn.init.zeros_(proj.bias_base)
            else:
                nn.init.trunc_normal_(proj.weight, std=0.02)
                nn.init.zeros_(proj.bias)
        if self.equivariant:
            nn.init.trunc_normal_(self.out_proj.wb, std=3e-3)
            nn.init.zeros_(self.out_proj.bias_base)
        else:
            nn.init.trunc_normal_(self.out_proj.weight, std=3e-3)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, tokens: torch.Tensor, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """tokens (B, T, C); x (B, N, C) cells; mask (B, N) bool. Returns the
        refreshed tokens (B, T, C)."""

        b, t, c = tokens.shape
        n = x.shape[1]
        h, d = self.heads, self.head_dim
        # D8 (spec D-S27): the projections run in the CELLS' compute dtype (the
        # token stream may be carried fp32 on half-precision serve); only the
        # residual add below runs in the stream's dtype. No-ops on the uniform
        # fp32 train path.
        kv = self.ln_kv(x)
        q = self.q_proj(self.ln_q(tokens.to(x.dtype)))
        k = self.k_proj(kv)
        v = self.v_proj(kv)
        # Serve folds the §4 coset perm into the cached q/k/v/out weights on the
        # SAME ``equivariant and not grad`` gate EquivLinear._materialize uses.
        folded = self.equivariant and not torch.is_grad_enabled()
        if self.equivariant and not folded:
            # Coset channel reorder exactly as RelPosAttention: head hh of the
            # (heads, head_dim) reshape is win-axis coset hh (docs/DERIVATION §4).
            hp = self._head_perm
            q = q[..., hp]
            k = k[..., hp]
            v = v[..., hp]
        q = q.reshape(b, t, h, d).transpose(1, 2)  # (B, h, T, d)
        k = k.reshape(b, n, h, d).transpose(1, 2)  # (B, h, N, d)
        v = v.reshape(b, n, h, d).transpose(1, 2)
        scores = (q @ k.mT) * self.scale + self.gate_bias.view(1, 1, t, 1)
        # fp32 SUM aggregation (plan R1): sigmoid gates, pad-cell keys zeroed
        # MULTIPLICATIVELY (the sum must see an exact 0, not sigmoid(-3e4)).
        gates = torch.sigmoid(scores.float()) * mask[:, None, None, :]
        upd = (gates @ v.float()) * self.sum_scale  # (B, h, T, d) — counts
        upd = upd.transpose(1, 2).reshape(b, t, c)
        if self.equivariant and not folded:
            upd = upd[..., self._head_perm_inv]
        # Raw residual add — no norm on the update (plan R5): the count
        # magnitudes ARE the signal; the next A block's pre-norm re-normalizes
        # the token stream. The add runs in the token stream's dtype (D-S27).
        return tokens + self.out_proj(upd.to(x.dtype)).to(tokens.dtype)


class TokenRead(nn.Module):
    """cells <- tokens broadcast read (plan R4): per-token tied 1x1s, summed,
    zero-init, added to every live cell at C-block entry."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        linear = EquivLinear if EQUIVARIANT else nn.Linear
        self.reads = nn.ModuleList(
            [linear(channels, channels) for _ in range(NUM_TOKENS)]
        )
        for read in self.reads:
            if EQUIVARIANT:
                nn.init.zeros_(read.wb)
                nn.init.zeros_(read.bias_base)
            else:
                nn.init.zeros_(read.weight)
                nn.init.zeros_(read.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, T, C) -> (B, 1, C), broadcast-added to cells by the caller
        (which applies the node mask)."""

        upd = self.reads[0](tokens[:, 0])
        for ti in range(1, NUM_TOKENS):
            upd = upd + self.reads[ti](tokens[:, ti])
        return upd.unsqueeze(1)
