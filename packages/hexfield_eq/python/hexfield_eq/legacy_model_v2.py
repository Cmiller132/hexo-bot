"""hexfield network: trunk, attention/bias, and heads.

Trunk block order is C C C A C C A C A over variable-N node sets: 6
post-activation conv residual blocks with LayerNorm, and 3 pre-norm transformer
blocks over the joint sequence [8 summary tokens ; cells] with one shared
237-row relative-position bias table. HexNodeConv is a gather + one GEMM.

Batch conventions (built by `batching.py`):
- feats  (B, Npad, F) f32; pad rows all-zero
- nbr    (B, Npad, 6) long, row-local; missing/pad -> Npad (the appended
  zero row, giving conv zero-padding semantics)
- mask   (B, Npad) bool, True at real nodes
- coords (B, Npad, 2) long axial (q, r); pad coords arbitrary

Convs and attention re-apply the node mask after every parameter-carrying op,
so pad rows stay zero and a real row's outputs do not depend on how much
padding shares its batch.
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .constants import (
    ATTENTION_HEADS,
    BIAS_CELL_TOKEN_ROW,
    BIAS_FAR_ROW,
    BIAS_OFF_AXIS_BASE,
    BIAS_ON_AXIS_BASE,
    BIAS_RING_MAX,
    BIAS_RING_MIN,
    BIAS_ROWS,
    BIAS_TOKEN_CELL_ROW,
    BIAS_TOKEN_TOKEN_ROW,
    CHANNELS,
    HEAD_DIM,
    MLP_RATIO,
    NUM_FEATURES,
    NUM_TOKENS,
    VALUE_BINS,
)
from .geometry import rel_bias_index

# Additive fill for pad-key attention columns. Finite in fp16.
PAD_KEY_MASK_VALUE = -3.0e4

STV_HORIZONS = (2, 6, 16)

# FlexAttention serve path. Opt-in via HEXFIELD_SERVE_FLEX=1, default off; used
# only on the no-grad serve forward. When enabled, the rel-pos bias is computed
# inside the attention kernel via a score_mod (coords + _cell_bias_lut +
# bias_table gather) with the pad-key mask folded into the score
# (PAD_KEY_MASK_VALUE additive fill), instead of materializing a
# (B, heads, S, S) bias tensor. The training (grad) path uses build_attn_bias.
# The flag is read once at import; the import is guarded so a torch without
# flex_attention still loads.
_SERVE_FLEX = os.environ.get("HEXFIELD_SERVE_FLEX") == "1"
try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    # _flex_call is torch.compiler.disable'd so the flex op compiles in its own
    # inner graph rather than being inlined into the outer dynamic-compile graph
    # of forward_policy_value. The conv trunk, projections, and heads stay in the
    # outer compiled graph; the outer graph breaks at the attention.
    _flex_compiled = torch.compile(_flex_attention, dynamic=False)

    @torch.compiler.disable(recursive=False)
    def _flex_call(q, k, v, score_mod):
        return _flex_compiled(q, k, v, score_mod=score_mod)

except Exception:  # pragma: no cover - older torch without flex
    _flex_attention = None
    _flex_call = None


class _FlexBias:
    """Carrier for the serve-flex attention path. Built once per forward in
    trunk() and passed in place of the materialized attn_bias tensor;
    RelPosAttention.forward detects it and routes to flex_attention.

    Holds the raw inputs the score_mod needs (coords, mask, the fp16 bias table,
    the cell LUT), not a pre-built closure. The score_mod closure is constructed
    in RelPosAttention.forward (same frame as the flex call) and invoked through
    the torch.compiler.disable'd _flex_call. No (B, heads, S, S) tensor is
    materialized."""

    __slots__ = ("coords", "mask", "table", "lut", "m", "w")

    def __init__(self, coords, mask, table, lut, m) -> None:
        self.coords = coords
        self.mask = mask
        self.table = table
        self.lut = lut
        self.m = m
        self.w = 2 * m + 1

    def make_score_mod(self):
        """Build the flex score_mod closure (called inside RelPosAttention.forward,
        the same frame as the flex_attention call). Computes the additive bias
        (coords + _cell_bias_lut + fp16 bias_table gather) plus the pad-key
        additive fill (PAD_KEY_MASK_VALUE) folded in via the bool mask."""

        nt = NUM_TOKENS
        coords = self.coords
        mask = self.mask
        table = self.table
        lut = self.lut
        m = self.m
        w = self.w
        pad_fill = PAD_KEY_MASK_VALUE

        def score_mod(score, b, h, q_idx, kv_idx):
            qc = torch.clamp(q_idx - nt, min=0)
            kc = torch.clamp(kv_idx - nt, min=0)
            dq = coords[b, kc, 0] - coords[b, qc, 0]
            dr = coords[b, kc, 1] - coords[b, qc, 1]
            qi = torch.clamp(dq, -m, m) + m
            ri = torch.clamp(dr, -m, m) + m
            cell_idx = lut[qi * w + ri]
            q_tok = q_idx < nt
            k_tok = kv_idx < nt
            row = torch.where(
                q_tok & k_tok,
                torch.full_like(cell_idx, BIAS_TOKEN_TOKEN_ROW),
                torch.where(
                    q_tok & ~k_tok,
                    torch.full_like(cell_idx, BIAS_TOKEN_CELL_ROW),
                    torch.where(
                        ~q_tok & k_tok,
                        torch.full_like(cell_idx, BIAS_CELL_TOKEN_ROW),
                        cell_idx,
                    ),
                ),
            )
            biased = score + table[row, h].to(score.dtype)
            # pad-key column: a cell key (kv_idx >= nt) whose row's mask is False.
            is_pad_key = (kv_idx >= nt) & ~mask[b, kc]
            return torch.where(is_pad_key, biased + pad_fill, biased)

        return score_mod


class HexNodeConv(nn.Module):
    """Direction-typed 7-tap hex convolution: gather (B,N,7,Cin) -> one GEMM.

    Tap 0 = center; taps 1-6 = the fixed direction order D (the rotate60
    orbit of (1,0)).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(7, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))
        # PyTorch conv default init with fan_in = 7 * C_in.
        fan_in = 7 * in_channels
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, x: torch.Tensor, gather_idx: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """x (B, Npad, Cin); gather_idx (B, Npad, 7) with tap 0 = self and
        missing -> Npad; mask (B, Npad) bool. Returns (B, Npad, Cout) with
        pad rows zeroed by the mask.
        """

        b, n, c = x.shape
        x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)  # zero row at index Npad
        flat = gather_idx.reshape(b, n * 7, 1).expand(-1, -1, c)
        gathered = x_ext.gather(1, flat).reshape(b, n, 7 * c)
        out = gathered @ self.weight.reshape(7 * c, self.out_channels) + self.bias
        return out * mask.unsqueeze(-1)


class ConvBlock(nn.Module):
    """Post-activation residual block with LayerNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = HexNodeConv(channels, channels)
        self.ln1 = nn.LayerNorm(channels)
        self.conv2 = HexNodeConv(channels, channels)
        self.ln2 = nn.LayerNorm(channels)

    def forward(
        self, x: torch.Tensor, gather_idx: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        m = mask.unsqueeze(-1)
        y = F.relu(self.ln1(self.conv1(x, gather_idx, mask))) * m
        y = self.ln2(self.conv2(y, gather_idx, mask)) * m
        return F.relu(x + y)


class RelPosAttention(nn.Module):
    """Multi-head self-attention (ATTENTION_HEADS heads) over the joint
    [tokens ; cells] sequence with the shared bias table gathered as an additive
    mask, via scaled_dot_product_attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = ATTENTION_HEADS
        # Per-instance head_dim = channels // heads, so instances built at a
        # non-default width work. At the default width this equals HEAD_DIM.
        self.head_dim = channels // ATTENTION_HEADS
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, seq: torch.Tensor, attn_bias) -> torch.Tensor:
        b, s, c = seq.shape
        h, d = self.heads, self.head_dim
        q = self.q_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        k = self.k_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        v = self.v_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        # Serve-flex: the rel-pos bias and pad mask are applied inside the kernel
        # via a score_mod (no materialized (B,heads,S,S) tensor). block_mask is
        # None (score_mod only). The score_mod is built here, in the same frame as
        # the flex call (see _FlexBias).
        if isinstance(attn_bias, _FlexBias):
            score_mod = attn_bias.make_score_mod()
            out = _flex_call(q, k, v, score_mod)
            out = out.transpose(1, 2).reshape(b, s, c)
            return self.out_proj(out)
        # Match the bias dtype to q; a dtype mismatch drops sdpa to the math
        # fallback. PAD_KEY_MASK_VALUE (-3.0e4) stays finite in fp16.
        attn_bias = attn_bias.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = out.transpose(1, 2).reshape(b, s, c)
        return self.out_proj(out)


class AttnBlock(nn.Module):
    """Pre-norm transformer block (GELU MLP, hidden ratio MLP_RATIO)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.attn = RelPosAttention(channels)
        self.ln2 = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, MLP_RATIO * channels)
        self.fc2 = nn.Linear(MLP_RATIO * channels, channels)

    def forward(
        self, seq: torch.Tensor, attn_bias: torch.Tensor, seq_mask: torch.Tensor
    ) -> torch.Tensor:
        m = seq_mask.unsqueeze(-1)
        seq = seq + self.attn(self.ln1(seq), attn_bias) * m
        seq = seq + self.fc2(F.gelu(self.fc1(self.ln2(seq)))) * m
        return seq


class HexfieldNet(nn.Module):
    """The full network: stem, C C C A C C A C A, LN_final, heads."""

    def __init__(self, channels: int = CHANNELS) -> None:
        super().__init__()
        # channels defaults to CHANNELS; an explicit value builds the net at a
        # different width.
        c = channels
        self.stem = HexNodeConv(NUM_FEATURES, c)
        self.stem_ln = nn.LayerNorm(c)
        self.conv_blocks = nn.ModuleList([ConvBlock(c) for _ in range(6)])
        self.attn_blocks = nn.ModuleList([AttnBlock(c) for _ in range(3)])
        self.tokens = nn.Parameter(torch.empty(NUM_TOKENS, c))
        self.bias_table = nn.Parameter(torch.zeros(BIAS_ROWS, ATTENTION_HEADS))
        self.ln_final = nn.LayerNorm(c)

        # Heads. Policy heads read cells; value/aux read tokens plus the masked
        # mean-pool of cells.
        self.policy_conv = HexNodeConv(c, c)
        self.policy_head = nn.Linear(c, 1)
        self.opp_policy_conv = HexNodeConv(c, c)
        self.opp_policy_head = nn.Linear(c, 1)
        self.value_reduction = nn.Linear(3 * c, c)
        self.value_head = nn.Linear(c, VALUE_BINS)
        self.aux_reduction = nn.Linear(3 * c, c)
        self.stv_heads = nn.ModuleDict(
            {str(h): nn.Linear(c, VALUE_BINS) for h in STV_HORIZONS}
        )
        self.moves_left_head = nn.Linear(c, VALUE_BINS)

        # Cell-bias LUT over the (dq, dr) offset domain, indexed as
        # row = _cell_bias_lut[(clamp(dq,-M,M)+M)*W + (clamp(dr,-M,M)+M)] with
        # M = BIAS_RING_MAX+1 = 17 and W = 2M+1 = 35. For |dq|,|dr| <= M the entry
        # is rel_bias_index(dq,dr); a clamped offset lands on a |coord|==M cell
        # whose hex-distance > BIAS_RING_MAX, yielding the far row, which is also
        # the far row for the pre-clamp offset. build_attn_bias uses this as one
        # gather in place of per-forward clamp/where selection.
        self._cell_bias_M = BIAS_RING_MAX + 1
        cw = 2 * self._cell_bias_M + 1
        cell_lut = torch.empty(cw * cw, dtype=torch.long)
        for dq in range(-self._cell_bias_M, self._cell_bias_M + 1):
            for dr in range(-self._cell_bias_M, self._cell_bias_M + 1):
                cell_lut[(dq + self._cell_bias_M) * cw + (dr + self._cell_bias_M)] = (
                    rel_bias_index(dq, dr)
                )
        self.register_buffer("_cell_bias_lut", cell_lut, persistent=False)

        # Serve-flex flag, read once. Used only on the no-grad serve path;
        # training always uses the materialized build_attn_bias regardless.
        self._serve_flex = _SERVE_FLEX and _flex_attention is not None

        self._init_weights()

    def _init_weights(self) -> None:
        """Linears: trunc_normal(std=0.02) weights, zero bias. LayerNorm:
        weight 1, bias 0. Convs keep the PyTorch default init from HexNodeConv.
        Residual-closing parameters are zero-initialized (each ConvBlock's ln2
        gain; each AttnBlock's out_proj and fc2 weights), so each residual branch
        starts as the identity. bias_table stays zero; tokens use
        trunc_normal(std=0.02)."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.tokens, std=0.02)
        for block in self.conv_blocks:
            nn.init.zeros_(block.ln2.weight)
        for block in self.attn_blocks:
            nn.init.zeros_(block.attn.out_proj.weight)
            nn.init.zeros_(block.fc2.weight)

    # --- pair index + bias (built once per batch, shared by all 3 attn blocks) ---

    def build_attn_bias(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """(B, heads, S, S) additive bias: shared-table gather + pad-key mask.

        coords (B, Npad, 2) long; mask (B, Npad) bool. S = NUM_TOKENS + Npad.
        Tokens sit at slots 0-7 with no board position; token keys are never
        masked, so every softmax row has at least one live key."""

        b, n, _ = coords.shape
        dq = coords[:, None, :, 0] - coords[:, :, None, 0]  # (B, N, N) key - query
        dr = coords[:, None, :, 1] - coords[:, :, None, 1]
        # One clamp + mul-add + gather through _cell_bias_lut (see its construction
        # note in __init__) maps each (dq, dr) offset to a bias-table row.
        m = self._cell_bias_M
        w = 2 * m + 1
        qi = dq.clamp(-m, m) + m
        ri = dr.clamp(-m, m) + m
        cell_idx = self._cell_bias_lut[(qi * w + ri).reshape(-1)].reshape(b, n, n)

        s = NUM_TOKENS + n
        pair = coords.new_full((b, s, s), BIAS_TOKEN_TOKEN_ROW)
        pair[:, :NUM_TOKENS, NUM_TOKENS:] = BIAS_TOKEN_CELL_ROW
        pair[:, NUM_TOKENS:, :NUM_TOKENS] = BIAS_CELL_TOKEN_ROW
        pair[:, NUM_TOKENS:, NUM_TOKENS:] = cell_idx

        # Pad-cell key columns: additive fill, finite in fp16. Token keys
        # untouched. The fill is added last so the returned attn_mask has
        # stride(-1) == 1 on the key axis; a non-stride-1 attn_mask forces
        # F.scaled_dot_product_attention onto the fp32 math backend instead of the
        # fused fp16 kernel.
        key_pad = torch.cat(
            [mask.new_ones(b, NUM_TOKENS), mask], dim=1
        )  # (B, S) True = live key

        # Build the (B,heads,S,S) bias in fp16 in head-first layout. Indexing the
        # transposed table (heads, ROWS) yields a contiguous (heads, B, Sq, Sk);
        # permute to (B, heads, Sq, Sk) is a stride(-1)==1 view, and the pad-mask
        # add is the single full-tensor materialization.
        bias_t = self.bias_table.to(torch.float16).t().contiguous()  # (heads, ROWS)
        bias = bias_t[:, pair]                       # (heads, B, Sq, Sk) contiguous
        bias = bias.permute(1, 0, 2, 3)              # (B, heads, Sq, Sk) view, stride(-1)=1
        fill = torch.where(key_pad, 0.0, PAD_KEY_MASK_VALUE).to(bias.dtype)
        return bias + fill[:, None, None, :]         # broadcast over key axis (dim 3)

    def _build_flex_bias(
        self, coords: torch.Tensor, mask: torch.Tensor
    ) -> "_FlexBias":
        """Serve-flex (no-grad) counterpart to build_attn_bias. Packages the raw
        tensors the score_mod needs (coords, mask, fp16 bias table, cell LUT) into
        a _FlexBias carrier; the score_mod closure is built in
        RelPosAttention.forward (see _FlexBias). No (B, heads, S, S) tensor is
        materialized; block_mask is None and the pad mask is folded into the score.

        The pad-key boundary is read from the bool mask directly, not mask.sum():
        a reduction inside the score_mod would produce a data-dependent symint the
        dynamic-Npad Inductor lowering cannot bind. The table stays (ROWS, heads)
        fp16 with no transpose; the score_mod indexes table[row, h]."""

        table = self.bias_table.to(torch.float16)
        return _FlexBias(coords, mask, table, self._cell_bias_lut, self._cell_bias_M)

    # --- forward ---------------------------------------------------------------

    def trunk(
        self,
        feats: torch.Tensor,
        nbr: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (cells (B,Npad,C), tokens (B,8,C), gather_idx) after
        LN_final."""

        b, n, _ = feats.shape
        self_idx = torch.arange(n, device=feats.device).reshape(1, n, 1).expand(b, -1, -1)
        gather_idx = torch.cat([self_idx, nbr], dim=2)  # (B, Npad, 7), tap 0 = self

        x = F.relu(self.stem_ln(self.stem(feats, gather_idx, mask))) * mask.unsqueeze(-1)
        # Serve-flex only on the no-grad path; with grad enabled, use the
        # materialized build_attn_bias.
        if self._serve_flex and not torch.is_grad_enabled():
            attn_bias = self._build_flex_bias(coords, mask)
        else:
            attn_bias = self.build_attn_bias(coords, mask)
        seq_mask = torch.cat([mask.new_ones(b, NUM_TOKENS), mask], dim=1)

        tokens = self.tokens.unsqueeze(0).expand(b, -1, -1)
        x = self.conv_blocks[0](x, gather_idx, mask)
        x = self.conv_blocks[1](x, gather_idx, mask)
        x = self.conv_blocks[2](x, gather_idx, mask)
        seq = self.attn_blocks[0](torch.cat([tokens, x], dim=1), attn_bias, seq_mask)
        tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
        x = self.conv_blocks[3](x, gather_idx, mask)
        x = self.conv_blocks[4](x, gather_idx, mask)
        seq = self.attn_blocks[1](torch.cat([tokens, x], dim=1), attn_bias, seq_mask)
        tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
        x = self.conv_blocks[5](x, gather_idx, mask)
        seq = self.attn_blocks[2](torch.cat([tokens, x], dim=1), attn_bias, seq_mask)
        seq = self.ln_final(seq)
        tokens, x = seq[:, :NUM_TOKENS], seq[:, NUM_TOKENS:]
        return x * mask.unsqueeze(-1), tokens, gather_idx

    def _pooled(self, cells: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean of LN_final cell vectors (pad rows excluded)."""

        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(cells.dtype)
        return (cells * mask.unsqueeze(-1)).sum(dim=1) / counts

    def forward_policy_value(
        self,
        feats: torch.Tensor,
        nbr: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        *,
        request_moves_left: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Serve forward. Returns policy and value; adds moves_left (via the aux
        reduction) only when request_moves_left is True. Does not compute
        opp-policy."""

        cells, tokens, gather_idx = self.trunk(feats, nbr, mask, coords)
        pooled = self._pooled(cells, mask)
        out = {
            "policy": self._policy_logits(
                self.policy_conv, self.policy_head, cells, gather_idx, mask
            ),
            "value": self.value_head(
                F.relu(self.value_reduction(self._value_input(tokens, 0, 1, pooled)))
            ),
        }
        if request_moves_left:
            aux = F.relu(self.aux_reduction(self._value_input(tokens, 2, 3, pooled)))
            out["moves_left"] = self.moves_left_head(aux)
        return out

    @staticmethod
    def _value_input(
        tokens: torch.Tensor, a: int, b: int, pooled: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([tokens[:, a], tokens[:, b], pooled], dim=1)

    @staticmethod
    def _policy_logits(
        conv: HexNodeConv,
        head: nn.Linear,
        cells: torch.Tensor,
        gather_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        y = F.relu(conv(cells, gather_idx, mask))
        return head(y).squeeze(-1) * mask  # (B, Npad); pad rows zeroed by mask
