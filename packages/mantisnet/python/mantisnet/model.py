"""MantisNet trunk and policy, action-value, and state-value heads.

The forward consumes a :class:`~mantisnet.builder.Batch` and performs no
data-dependent index discovery: every gather, scatter, and pad slot was
precomputed by the builder. Weights are fp32; the forward is written to run
under bf16 autocast without assuming it (buffers inherit dtype from inputs,
and the scalar value decode is done in fp32).

Linear maps written as bare matrices in the spec (``U``, ``V``, ``P``) are
bias-free — the per-class embedding added alongside them is the additive term.
Attention key projections are bias-free; FFN, MLP, and the other attention
linears keep the framework-default bias (§10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import (
    cell_latents,
    cell_nodes,
    message_passing,
    relay,
    row_encoder,
    window_latents,
    window_pairs,
)
from .attention import fused_attention
from .builder import (
    ACTION_EMPTY_CLASSES,
    TERN_DEC_CLASSES,
    TERN_OCC_CLASSES,
    TERN_PATTERNS,
    TERN_POST1_CLASSES,
    Batch,
)
from .segments import segment_ids, segment_max


@dataclass(frozen=True)
class MantisConfig:
    """The named parameters of MODEL_SPEC §2, at their suggested defaults."""

    h: int = 128  # H: embedding width, everywhere
    blocks: int = 4  # B
    heads: int = 4  # A
    ffn_factor: int = 2  # F
    d_max: int = 12  # D_MAX: hex-distance clamp
    value_queries: int = 4  # Q
    value_bins: int = 65  # K
    policy_hidden: int = 128  # P_H
    value_hidden: int = 128  # V_H
    dropout: float = 0.0
    # Live knob for the §5.1c typed window-pair attention stage.
    window_attention: bool = True
    # §5.1c claim reach: 5 is the donor geometry; 0 restricts the crossing
    # join to pairs sharing an in-span cell (Step 15 arm E, pricing the
    # out-of-span crossing signal). A path selector, not a tunable radius.
    claim_reach: int = 5
    # Step 15 knobs: cell latents replace the §5.1b relay with persistent
    # typed state on the covered legal cells; the line pass runs whole-line
    # window attention in the §5.1c slot.
    cell_latents: bool = False
    line_pass: bool = False
    # Step 13: extend persistent state to every legal cell and let each block
    # read invariant radius-8 stone context. Adjacency is a factored sub-knob.
    cell_nodes: bool = False
    cell_node_scope: str = "all"
    cell_adjacency: bool = False

    def __post_init__(self) -> None:
        if self.h % self.heads != 0:
            raise ValueError(f"H={self.h} must divide into A={self.heads} heads")
        if self.value_bins % 2 == 0:
            raise ValueError(f"K={self.value_bins} must be odd so an exact-zero bin exists")
        if self.claim_reach not in (0, 5):
            raise ValueError(
                f"claim_reach must be one of {{0, 5}}, got {self.claim_reach}"
            )
        if self.claim_reach == 0 and not self.window_attention:
            raise ValueError(
                "claim_reach=0 modifies the §5.1c stage, which "
                "window_attention=False removes; the knob would be inert"
            )
        if self.cell_adjacency and not self.cell_nodes:
            raise ValueError("cell_adjacency requires cell_nodes=True")
        if self.cell_node_scope not in cell_nodes.CELL_NODE_SCOPES:
            raise ValueError(
                "cell_node_scope must be one of "
                f"{cell_nodes.CELL_NODE_SCOPES}, got {self.cell_node_scope!r}"
            )
        if self.cell_node_scope != "all" and not self.cell_nodes:
            raise ValueError(
                f"cell_node_scope={self.cell_node_scope!r} requires cell_nodes=True"
            )

    @property
    def uses_cell_state(self) -> bool:
        return self.cell_latents or self.cell_nodes

    @property
    def window_vocab(self) -> int:
        return TERN_PATTERNS

    @property
    def dec_classes(self) -> int:
        return TERN_DEC_CLASSES

    @property
    def occ_classes(self) -> int:
        return TERN_OCC_CLASSES

    # Bucket indices of the attention bias table (§4.1): distances 1..D_MAX
    # occupy 0..D_MAX-1, then SELF, then TOKEN. TOKEN wins on the token-token
    # pair. PAD is not a parameter row: attention appends its finite sentinel
    # after casting the learned table to the compute dtype.
    @property
    def self_bucket(self) -> int:
        return self.d_max

    @property
    def token_bucket(self) -> int:
        return self.d_max + 1

    @property
    def pad_bucket(self) -> int:
        return self.d_max + 2


@dataclass
class ModelOutput:
    """What one full forward answers (§11 plus the appendix-B Q head)."""

    policy_logits: Tensor  # (N_cells,) raw, engine legal order per position
    q_score: Tensor  # (N_cells,) the score π′ ranks by, same layout
    q_values: Tensor  # (N_cells,) action values, same layout — the KLENT head
    value: Tensor  # (P,) scalar decode, in [-1, 1]
    value_dist: Tensor  # (P, K) softmax over bins, fp32
    value_logits: Tensor  # (P, K) the bins' raw logits — what value_loss trains


# Width of appendix B's categorical action-value readout: the three logits
# [z_pos, z_neg, z_zero]. Every reader takes the checkpoint shape from here.
CRITIC_LOGITS = 3

# The trunk stages were config knobs while they were ablations; checkpoints
# written in that window record the knob fields in their model_config. These
# are the values that describe the architecture this build bakes in — a
# recorded config carrying exactly these is this architecture and loads; any
# other value names a model this build no longer implements.
LEGACY_BAKED_KNOBS: dict[str, object] = {
    "axis_bias": True,
    "off_axis_bias": False,
    "cell_pass": True,
    "cell_pass_from": 0,
    "cell_pass_rounds": 1,
    "joint_incidence": True,
    "mixed_windows": True,
    "action_rows": True,
    "state_latents": 4,
}


def strip_legacy_knobs(recorded: dict) -> dict:
    """Drop legacy knob keys that match the baked architecture; refuse others."""
    out = dict(recorded)
    for key, baked in LEGACY_BAKED_KNOBS.items():
        if key in out:
            if out[key] != baked:
                raise ValueError(
                    f"recorded model config {key}={out[key]!r} describes an "
                    f"architecture this build no longer implements "
                    f"(it bakes in {key}={baked!r})"
                )
            del out[key]
    return out


def return_mass(critic_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Decode ``(..., 3)`` categorical logits as positive/negative mass, fp32.

    The softmax rows are ``(p_pos, p_neg, p_zero)``. At the categorical
    cross-entropy optimum, the returned pair is ``(E[G⁺], E[G⁻])``;
    their sum is ``E|G|`` and is at most one because the omitted zero mass is
    the remainder of the same simplex.
    """
    p_pos, p_neg, _p_zero = critic_logits.float().softmax(dim=-1).unbind(dim=-1)
    return p_pos, p_neg


def compose_q(critic_logits: Tensor) -> Tensor:
    """Compose ``(..., 3)`` categorical logits into action values, fp32.

    ``Q = p_pos - p_neg`` is the quantity the λ-return targets and v̂ averages.
    The categorical simplex makes ``Q`` lie in ``(-1, 1)`` and committed mass
    ``p_pos + p_neg`` lie in ``(0, 1)`` by construction.

    The function is free-standing because acting composes every legal cell
    while fitting composes only the taken action, off the same raw logits.
    """
    p_pos, p_neg = return_mass(critic_logits)
    return p_pos - p_neg


def compose_acting_q(
    critic_logits: Tensor, offsets: Tensor, mass_floor: float
) -> Tensor:
    """Return Q divided by the position's floored maximum committed mass.

    ``M = p_pos + p_neg = 1 - p_zero`` is structurally in ``(0, 1)``. One
    positive divisor is shared by every legal cell in a position, so the score
    preserves Q's order while expressing it in units of the most committed
    action. Since ``|Q| <= M <= max M``, an unfloored score lies in ``(-1, 1)``;
    ``mass_floor`` additionally bounds sharpening when all actions put most
    probability on zero return.
    """
    p_pos, p_neg = return_mass(critic_logits)
    seg = segment_ids(offsets)
    scale = segment_max(p_pos + p_neg, seg, offsets.shape[0] - 1).clamp(
        min=mass_floor
    )
    return (p_pos - p_neg) / scale.index_select(0, seg)


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


class _PairMlp(nn.Module):
    """``MLP([a; b])`` with the concatenation folded away.

    A linear over a concatenation is the sum of two linears. This form preserves
    the parameters and arithmetic without materializing the 2H-wide input.
    """

    def __init__(self, h: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.lin_a = nn.Linear(h, d_hidden)
        self.lin_b = nn.Linear(h, d_hidden, bias=False)
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.out(F.relu(self.lin_a(a) + self.lin_b(b)))


class _Block(nn.Module):
    """One trunk block (§5): window <- stones, stone <- windows, attention."""

    def __init__(self, cfg: MantisConfig) -> None:
        super().__init__()
        h = cfg.h
        self.cfg = cfg
        # §5.1 window <- stones
        self.ln_ws_s = nn.LayerNorm(h)
        self.ln_ws_w = nn.LayerNorm(h)
        self.u = nn.Linear(h, h, bias=False)
        self.e_ws = nn.Embedding(cfg.occ_classes, h)
        self.mlp_w = _PairMlp(h, h, h)
        # §5.1b window <- windows through their shared empty cells: with the
        # cell-latent knob on, the shared cells hold persistent typed state
        # (cells read their windows with class value rows, windows read back
        # bias-typed); off, the transient relay.
        if cfg.uses_cell_state:
            self.ln_cr_c = nn.LayerNorm(h)
            self.ln_cr_w = nn.LayerNorm(h)
            self.cr_wq = nn.Linear(h, h)
            self.cr_wk = nn.Linear(h, h, bias=False)
            self.cr_wv = nn.Linear(h, h)
            self.cr_wo = nn.Linear(h, h)
            self.cr_bias = nn.Parameter(torch.zeros(cfg.heads, cfg.dec_classes))
            self.cr_vclass = nn.Embedding(cfg.dec_classes, h)
            self.ln_wr_w = nn.LayerNorm(h)
            self.ln_wr_c = nn.LayerNorm(h)
            self.wr_wq = nn.Linear(h, h)
            self.wr_wk = nn.Linear(h, h, bias=False)
            self.wr_wv = nn.Linear(h, h)
            self.wr_wo = nn.Linear(h, h)
            self.wr_bias = nn.Parameter(torch.zeros(cfg.heads, cfg.dec_classes))
            if cfg.cell_nodes:
                self.ln_radius_c = nn.LayerNorm(h)
                self.ln_radius_s = nn.LayerNorm(h)
                self.radius_wq = nn.Linear(h, h)
                self.radius_wk = nn.Linear(h, h, bias=False)
                self.radius_wv = nn.Linear(h, h)
                self.radius_wo = nn.Linear(h, h)
                self.radius_bias = nn.Parameter(
                    torch.zeros(cfg.heads, cell_nodes.RADIUS_CLASSES)
                )
                self.radius_vclass = nn.Embedding(cell_nodes.RADIUS_CLASSES, h)
            if cfg.cell_adjacency:
                self.ln_adj_q = nn.LayerNorm(h)
                self.ln_adj_k = nn.LayerNorm(h)
                self.adj_wq = nn.Linear(h, h)
                self.adj_wk = nn.Linear(h, h, bias=False)
                self.adj_wv = nn.Linear(h, h)
                self.adj_wo = nn.Linear(h, h)
                self.adj_bias = nn.Parameter(
                    torch.zeros(cfg.heads, cell_nodes.ADJACENCY_CLASSES)
                )
                self.adj_vclass = nn.Embedding(cell_nodes.ADJACENCY_CLASSES, h)
        else:
            self.ln_cp_in = nn.LayerNorm(h)
            self.u_cp = nn.Linear(h, h, bias=False)
            self.e_cp = nn.Embedding(cfg.dec_classes, h)
            self.ln_cp_w = nn.LayerNorm(h)
            self.mlp_cp = _PairMlp(h, h, h)
        # §5.1c window <- windows through typed pair relations.
        if cfg.window_attention:
            self.ln_wa = nn.LayerNorm(h)
            self.wq_wa = nn.Linear(h, h)
            self.wk_wa = nn.Linear(h, h, bias=False)
            self.wv_wa = nn.Linear(h, h)
            self.wo_wa = nn.Linear(h, h)
            self.wa_bias = nn.Parameter(
                torch.zeros(cfg.heads, window_pairs.WA_CLASSES)
            )
        if cfg.line_pass:
            # Whole-line window attention: the §5.1c colinear vocabulary at
            # unbounded offset, on the same edge-attention kernels.
            self.ln_lp = nn.LayerNorm(h)
            self.lp_wq = nn.Linear(h, h)
            self.lp_wk = nn.Linear(h, h, bias=False)
            self.lp_wv = nn.Linear(h, h)
            self.lp_wo = nn.Linear(h, h)
            self.lp_bias = nn.Parameter(
                torch.zeros(cfg.heads, cell_latents.LINE_CLASSES)
            )
        # Window read, latent self-mix, and window broadcast. Every key
        # projection is bias-free because a shared key bias cancels from
        # its softmax row.
        self.latent_ln_read_q = nn.LayerNorm(h)
        self.latent_ln_read_w = nn.LayerNorm(h)
        self.latent_wq_read = nn.Linear(h, h)
        self.latent_wk_read = nn.Linear(h, h, bias=False)
        self.latent_wv_read = nn.Linear(h, h)
        self.latent_wo_read = nn.Linear(h, h)

        self.latent_ln_mix = nn.LayerNorm(h)
        self.latent_wq_mix = nn.Linear(h, h)
        self.latent_wk_mix = nn.Linear(h, h, bias=False)
        self.latent_wv_mix = nn.Linear(h, h)
        self.latent_wo_mix = nn.Linear(h, h)

        self.latent_ln_bcast_q = nn.LayerNorm(h)
        self.latent_ln_bcast_l = nn.LayerNorm(h)
        self.latent_wq_bcast = nn.Linear(h, h)
        self.latent_wk_bcast = nn.Linear(h, h, bias=False)
        self.latent_wv_bcast = nn.Linear(h, h)
        self.latent_wo_bcast = nn.Linear(h, h)
        # §5.2 stone <- windows
        self.ln_sw_w = nn.LayerNorm(h)
        self.ln_sw_s = nn.LayerNorm(h)
        self.v = nn.Linear(h, h, bias=False)
        self.e_sw = nn.Embedding(cfg.occ_classes, h)
        self.mlp_s = _PairMlp(h, h, h)
        # §5.3 stone self-attention + state latents
        self.ln_attn = nn.LayerNorm(h)
        self.wq = nn.Linear(h, h)
        self.wk = nn.Linear(h, h, bias=False)
        self.wv = nn.Linear(h, h)
        self.wo = nn.Linear(h, h)
        self.dist_bias = nn.Parameter(torch.zeros(cfg.heads, cfg.d_max + 2))
        self.axis_bias = nn.Parameter(torch.zeros(cfg.heads, cfg.d_max))
        self.ln_ffn = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, cfg.ffn_factor * h), nn.ReLU(), nn.Linear(cfg.ffn_factor * h, h)
        )
        self.drop = nn.Dropout(cfg.dropout)

    def _cell_pass(
        self,
        w: Tensor,
        cell_ptr: Tensor,
        edge_window: Tensor,
        edge_class: Tensor,
        win_ptr: Tensor,
        edge_wcell: Tensor,
        cls_ptr: Tensor,
        edge_ccell: Tensor,
    ) -> Tensor:
        """Update windows through the live-window/empty-cell incidence graph."""
        x = self.u_cp(self.ln_cp_in(w))
        agg = relay.cell_pass(
            x,
            self.e_cp.weight,
            cell_ptr,
            edge_window,
            edge_class,
            win_ptr,
            edge_wcell,
            cls_ptr,
            edge_ccell,
        )
        return w + self.drop(self.mlp_cp(self.ln_cp_w(w), agg))

    def _cell_stage(
        self,
        s: Tensor,
        w: Tensor,
        c: Tensor,
        ctab: cell_latents.CellTables,
        radius: cell_latents.CellTables | None,
        adjacency: cell_latents.CellTables | None,
    ) -> tuple[Tensor, Tensor]:
        """§5.1b as persistent state: cells read their at most 18 containing
        windows with per-class score bias and class value rows, then windows
        read back from their at most 5 empty cells bias-typed. A full window
        has no empty cells and reads zero. Both ops run their scores,
        softmax, and weighted sums in fp32 whatever autocast chose for the
        projections."""
        cfg = self.cfg
        heads, hd = cfg.heads, cfg.h // cfg.heads
        n_c, n_w = c.shape[0], w.shape[0]

        zc = self.ln_cr_c(c)
        zw = self.ln_cr_w(w)
        read = cell_latents.cell_read(
            self.cr_wq(zc).view(n_c, heads, hd),
            self.cr_wk(zw).view(n_w, heads, hd),
            self.cr_wv(zw).view(n_w, heads, hd),
            self.cr_bias,
            self.cr_vclass.weight,
            ctab,
        )
        c = c + self.drop(self.cr_wo(read.reshape(n_c, cfg.h).to(zc.dtype)))

        if cfg.cell_nodes:
            if radius is None:
                raise ValueError("cell_nodes is on but radius tables are missing")
            zc = self.ln_radius_c(c)
            zs = self.ln_radius_s(s)
            read = cell_latents.cell_read(
                self.radius_wq(zc).view(n_c, heads, hd),
                self.radius_wk(zs).view(s.shape[0], heads, hd),
                self.radius_wv(zs).view(s.shape[0], heads, hd),
                self.radius_bias,
                self.radius_vclass.weight,
                radius,
            )
            c = c + self.drop(
                self.radius_wo(read.reshape(n_c, cfg.h).to(zc.dtype))
            )
        elif radius is not None:
            raise ValueError("radius tables were passed but cell_nodes is off")

        if cfg.cell_adjacency:
            if adjacency is None:
                raise ValueError("cell_adjacency is on but adjacency tables are missing")
            zq = self.ln_adj_q(c)
            zk = self.ln_adj_k(c)
            read = cell_latents.cell_read(
                self.adj_wq(zq).view(n_c, heads, hd),
                self.adj_wk(zk).view(n_c, heads, hd),
                self.adj_wv(zk).view(n_c, heads, hd),
                self.adj_bias,
                self.adj_vclass.weight,
                adjacency,
            )
            c = c + self.drop(self.adj_wo(read.reshape(n_c, cfg.h).to(zq.dtype)))
        elif adjacency is not None:
            raise ValueError("adjacency tables were passed but cell_adjacency is off")

        zw = self.ln_wr_w(w)
        zc = self.ln_wr_c(c)
        back = cell_latents.window_read(
            self.wr_wq(zw).view(n_w, heads, hd),
            self.wr_wk(zc).view(n_c, heads, hd),
            self.wr_wv(zc).view(n_c, heads, hd),
            self.wr_bias,
            ctab,
        )
        w = w + self.drop(self.wr_wo(back.reshape(n_w, cfg.h).to(zw.dtype)))
        return w, c

    def _line_pass(self, w: Tensor, lines) -> Tensor:
        """Whole-line attention over each window's colinear edges, on the
        §5.1c edge op with the 13-class line vocabulary. ``lines`` is the
        trunk's per-batch ``PairTables``, derived once on device."""
        cfg = self.cfg
        heads, hd = cfg.heads, cfg.h // cfg.heads
        n_w = w.shape[0]

        z = self.ln_lp(w)
        out = window_pairs.edge_attention(
            self.lp_wq(z).view(n_w, heads, hd),
            self.lp_wk(z).view(n_w, heads, hd),
            self.lp_wv(z).view(n_w, heads, hd),
            self.lp_bias,
            *lines,
        )
        return w + self.drop(self.lp_wo(out.reshape(n_w, cfg.h).to(z.dtype)))

    def _window_attention(self, w: Tensor, pairs) -> Tensor:
        """§5.1c: multi-head attention over each window's relation edges.

        The edge op runs scores, softmax, and the weighted sum in fp32
        whatever autocast chose for the projections, saves only the softmax
        stats, and recomputes every per-edge quantity in backward. ``pairs``
        is the trunk's per-batch ``PairTables``, derived once on device.
        """
        cfg = self.cfg
        heads, hd = cfg.heads, cfg.h // cfg.heads
        n_w = w.shape[0]

        z = self.ln_wa(w)
        out = window_pairs.edge_attention(
            self.wq_wa(z).view(n_w, heads, hd),
            self.wk_wa(z).view(n_w, heads, hd),
            self.wv_wa(z).view(n_w, heads, hd),
            self.wa_bias,
            *pairs,
        )
        out = self.wo_wa(out.reshape(n_w, cfg.h).to(z.dtype))
        return w + self.drop(out)

    def _window_latent_cycle(
        self,
        w: Tensor,
        g: Tensor,
        batch: Batch,
        layout: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Latents read flat windows, self-mix, then broadcast to windows."""
        cfg = self.cfg
        p, slots, h = g.shape
        heads, head_dim = cfg.heads, h // cfg.heads
        window_pos, offsets, order = layout

        # The fused op owns gather, fp32 scores, softmax, and weighted sum.
        # An empty position run returns an exact zero read delta.
        q = self.latent_wq_read(self.latent_ln_read_q(g)).view(
            p, slots, heads, head_dim
        )
        window_rows = self.latent_ln_read_w(w)
        k = self.latent_wk_read(window_rows).view(-1, heads, head_dim)
        v = self.latent_wv_read(window_rows).view(-1, heads, head_dim)
        read = window_latents.read_attention(q, k, v, window_pos, offsets, order)
        delta = self.latent_wo_read(read.reshape(p, slots, h).to(g.dtype))
        g = g + self.drop(delta)

        # Mix. All four slots are live, so the dense K-by-K softmax needs no
        # mask and stays fp32 independently of autocast.
        z = self.latent_ln_mix(g)
        q = self.latent_wq_mix(z).view(p, slots, heads, head_dim)
        k = self.latent_wk_mix(z).view(p, slots, heads, head_dim)
        v = self.latent_wv_mix(z).view(p, slots, heads, head_dim)
        scores = torch.einsum("pqhd,pkhd->phqk", q, k) / math.sqrt(head_dim)
        weights = scores.float().softmax(dim=-1)
        mixed = torch.einsum("phqk,pkhd->pqhd", weights, v.float())
        delta = self.latent_wo_mix(mixed.reshape(p, slots, h).to(g.dtype))
        g = g + self.drop(delta)

        # K is fixed at four, so the fused op uses multiply-sum reductions
        # rather than a small-dimension dot and returns flat window rows.
        latent_rows = self.latent_ln_bcast_l(g)
        q = self.latent_wq_bcast(self.latent_ln_bcast_q(w)).view(
            -1, heads, head_dim
        )
        k = self.latent_wk_bcast(latent_rows).view(p, slots, heads, head_dim)
        v = self.latent_wv_bcast(latent_rows).view(p, slots, heads, head_dim)
        broadcast = window_latents.broadcast_attention(
            q, k, v, window_pos, offsets, order
        )
        delta = self.latent_wo_bcast(broadcast.reshape(-1, h).to(w.dtype))
        w = w + self.drop(delta)
        return w, g

    def forward(
        self,
        s: Tensor,
        w: Tensor,
        g: Tensor,
        c: Tensor | None,
        batch: Batch,
        seq_lens: Tensor,
        plan: message_passing.IncidencePlan,
        pairs,
        latent_layout: tuple[Tensor, Tensor, Tensor],
        lines,
        ctab: cell_latents.CellTables | None,
        radius: cell_latents.CellTables | None,
        adjacency: cell_latents.CellTables | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        cfg = self.cfg
        # Sizes come from tensor shapes, not the Batch's ints: under
        # torch.compile they become symbolic, so one graph serves every shape.
        p, max_t = g.shape[0], batch.attn_valid.shape[1]

        # §5.1: windows aggregate their stones. Sum, not mean — the count is
        # signal. The class term of each pass is a dense ``counts @ table``
        # whose gradient is another matmul, not a per-entry gather whose
        # backward scatters into a few rows; the wide ternary
        # vocabularies run-reduce the same table rows (``class_row_sum``)
        # without materializing the per-edge gather.
        x = self.u(self.ln_ws_s(s))
        agg = message_passing.aggregate_to_windows(
            x,
            batch.inc_stone,
            batch.inc_window,
            plan.run_stone,
            plan.run_window,
            w.shape[0],
        )
        agg = agg + message_passing.class_row_sum(
            self.e_ws.weight,
            batch.inc_class,
            batch.inc_window,
            w.shape[0],
            message_passing.WINDOW_RUN,
        ).to(agg.dtype)
        w = w + self.drop(self.mlp_w(self.ln_ws_w(w), agg))

        if cfg.uses_cell_state:
            w, c = self._cell_stage(s, w, c, ctab, radius, adjacency)
        else:
            w = self._cell_pass(
                w,
                batch.relay_cell_ptr,
                batch.relay_window,
                batch.relay_class,
                batch.relay_win_ptr,
                batch.relay_wcell,
                batch.relay_cls_ptr,
                batch.relay_ccell,
            )
        if cfg.window_attention:
            w = self._window_attention(w, pairs)
        if cfg.line_pass:
            w = self._line_pass(w, lines)

        # §5.2: stones aggregate their windows.
        y = self.v(self.ln_sw_w(w))
        agg = message_passing.aggregate_to_stones(
            y,
            batch.inc_stone,
            batch.inc_window,
            plan.run_stone,
            plan.run_window,
            s.shape[0],
        )
        agg = agg + message_passing.class_row_sum(
            self.e_sw.weight,
            plan.run_class,
            plan.run_stone,
            s.shape[0],
            message_passing.STONE_RUN,
        ).to(agg.dtype)
        s = s + self.drop(self.mlp_s(self.ln_sw_s(s), agg))

        # §5.3: attention over [global rows; stones], block-diagonal per position.
        rows = s.new_zeros(p * max_t, cfg.h)
        global_rows = 4
        global_slot = (
            torch.arange(p, device=s.device)[:, None] * max_t
            + torch.arange(global_rows, device=s.device)[None, :]
        ).reshape(-1)
        rows.index_copy_(0, global_slot, g.reshape(-1, cfg.h))
        rows.index_copy_(0, batch.stone_slot, s)
        z = self.ln_attn(rows.view(p, max_t, cfg.h))

        hd = cfg.h // cfg.heads
        q = self.wq(z).view(p, max_t, cfg.heads, hd).transpose(1, 2)
        k = self.wk(z).view(p, max_t, cfg.heads, hd).transpose(1, 2)
        v = self.wv(z).view(p, max_t, cfg.heads, hd).transpose(1, 2)

        # Coordinates become distance buckets inside the attention kernel.
        # Each position's key loop stops at its live prefix instead of doing
        # quadratic work over padding.
        out = fused_attention(
            q,
            k,
            v,
            batch.coords,
            seq_lens,
            self.dist_bias,
            self.axis_bias,
            global_rows,
        )
        out = self.wo(out.transpose(1, 2).reshape(p, max_t, cfg.h)).view(p * max_t, cfg.h)
        s = s + self.drop(out.index_select(0, batch.stone_slot))
        global_delta = out.index_select(0, global_slot)
        g = g + self.drop(global_delta.view(p, global_rows, cfg.h))
        w, g = self._window_latent_cycle(w, g, batch, latent_layout)

        # FFN over the same rows — row-independent, so no padding needed.
        global_flat = g.reshape(-1, cfg.h)
        rows = torch.cat([s, global_flat], dim=0)
        rows = self.drop(self.ffn(self.ln_ffn(rows)))
        s = s + rows[: s.shape[0]]
        global_flat = global_flat + rows[s.shape[0] :]
        g = global_flat.view(p, global_rows, cfg.h)
        return s, w, g, c


class MantisNet(nn.Module):
    """Embeddings, B trunk blocks, policy, action-value, and state-value heads."""

    def __init__(self, cfg: MantisConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or MantisConfig()
        self.cfg = cfg
        h = cfg.h

        # §3: ternary canonical patterns subsume colour and status.
        self.stone_table = nn.Embedding(2, h)  # own / opp
        self.window_table = nn.Embedding(cfg.window_vocab, h)
        self.latent_base = nn.Parameter(torch.empty(4, h))
        self.token_moves = nn.Embedding(2, h)  # moves_remaining in {1, 2}
        if cfg.uses_cell_state:
            # Every covered cell starts from one learned row; identity
            # accrues through the typed reads. Legal cells no live window
            # covers keep this row at the decoder — far-ring semantics.
            self.cell_base = nn.Parameter(torch.empty(h))
        if cfg.cell_nodes:
            self.cell_occupancy_table = nn.Embedding(3, h)
            self.cell_legal_table = nn.Embedding(2, h)
            self.cell_nearest_table = nn.Embedding(cell_nodes.NEAREST_BUCKETS, h)

        self.blocks = nn.ModuleList(_Block(cfg) for _index in range(cfg.blocks))
        self.ln_out = nn.LayerNorm(h)  # shared final LN over S, W, g (§5)

        # §6 policy decoder. MLP_P([h_a; g]) as a _PairMlp, so the g half of
        # its first layer runs per position, not per legal cell.
        self.p = nn.Linear(h, h, bias=False)
        self.e_pw = nn.Embedding(cfg.dec_classes, h)
        self.mlp_p = _PairMlp(h, cfg.policy_hidden, 1)

        # Appendix B action-value decoder: the same shape as §6 with its own
        # parameters everywhere, three outcome logits per legal cell. KLENT's
        # categorical critic head.
        self.q = nn.Linear(h, h, bias=False)
        self.e_qw = nn.Embedding(cfg.dec_classes, h)
        self.mlp_q = _PairMlp(h, cfg.policy_hidden, CRITIC_LOGITS)

        # The shared row encoder supplies every legal action to both heads.
        # Its second layer is folded into each head's extension matrix because
        # summing the hidden rows commutes with that head-specific linear.
        self.act_proj = nn.Linear(h, h)
        self.act_table = nn.Embedding(TERN_POST1_CLASSES, h)
        self.act_empty_base = nn.Parameter(torch.empty(h))
        self.p_act = nn.Linear(h, h, bias=False)
        self.q_act = nn.Linear(h, h, bias=False)
        self.register_buffer(
            "act_empty_classes",
            torch.tensor(ACTION_EMPTY_CLASSES, dtype=torch.long),
            persistent=False,
        )

        # §7 value head.
        self.value_queries = nn.Parameter(torch.empty(cfg.value_queries, h))
        self.ln_value = nn.LayerNorm(h)
        self.mlp_v = _mlp(cfg.value_queries * h, cfg.value_hidden, cfg.value_bins)
        self.register_buffer(
            "bin_centers", torch.linspace(-1.0, 1.0, cfg.value_bins), persistent=False
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # §10: embeddings, the latent base, and the value queries N(0, 0.02);
        # bias tables zero (already); linears framework-default.
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.latent_base, std=0.02)
        if self.cfg.uses_cell_state:
            nn.init.normal_(self.cell_base, std=0.02)
        nn.init.normal_(self.value_queries, std=0.02)
        nn.init.normal_(self.act_empty_base, std=0.02)
        # Both decoder outputs start at zero, so the initial policy logits are
        # constant across legal cells and all outcome logits vanish, which makes
        # the initial action values exactly zero (appendix B).
        for head in (self.mlp_p, self.mlp_q):
            nn.init.zeros_(head.out.weight)
            nn.init.zeros_(head.out.bias)

    def _pair_tables(self, batch: Batch):
        # §5.1c tables are born on the batch's device from the window
        # identities: the int64 edge views cost several times more to ship
        # over PCIe than to derive beside the model, and every block shares
        # one derivation. The op is opaque to the compiler — a graph break
        # here would spill the surrounding message passing to eager. The
        # source view shares the destination view's arrays (reversal
        # closure), reassembled here because ops may not return aliases.
        ptr, src, cls, scls, cptr, cedge = window_pairs.derive_pair_tables(
            batch.window_id, batch.window_slot // batch.max_w, self.cfg.claim_reach
        )
        return window_pairs.PairTables(ptr, src, cls, ptr, src, scls, cptr, cedge)

    def _line_tables(self, batch: Batch) -> window_pairs.PairTables:
        # Born on device for the same reasons as the §5.1c tables. Line
        # classes are reversal-symmetric, so the source view shares the
        # destination view's arrays outright.
        ptr, src, cls, cptr, cedge = cell_latents.derive_line_tables(
            batch.window_id, batch.window_slot // batch.max_w
        )
        return window_pairs.PairTables(ptr, src, cls, ptr, src, cls, cptr, cedge)

    def _cell_tables(self, batch: Batch, n_windows: int) -> cell_latents.CellTables:
        # The decoder incidence resorted into the cell-attention views, with
        # the covered -> legal mapping the heads scatter refined latents by.
        return cell_latents.CellTables(
            *cell_latents.derive_cell_tables(
                batch.dec_cell,
                batch.dec_window,
                batch.dec_class,
                n_windows,
                self.cfg.dec_classes,
                batch.cell_pos.shape[0] if self.cfg.cell_nodes else -1,
            )
        )

    def _radius_tables(
        self, batch: Batch, covered: Tensor, n_stones: int
    ) -> cell_latents.CellTables:
        classes = batch.radius_orbit + cell_nodes.ORBIT48_CLASSES * (
            batch.radius_own + 2 * batch.radius_on_axis
        )
        return cell_nodes.tables_from_op(
            batch.radius_src,
            batch.radius_dst,
            classes,
            covered,
            n_stones,
            batch.cell_pos.shape[0],
            cell_nodes.RADIUS_CLASSES,
            self.cfg.cell_node_scope == "uncovered",
        )

    def _adjacency_tables(
        self, batch: Batch, covered: Tensor
    ) -> cell_latents.CellTables:
        # The structural axis is emitted for the future equivariant route. In
        # the invariant Step 13 channel all three axes share one relation row.
        classes = torch.zeros_like(batch.adjacency_axis)
        return cell_nodes.tables_from_op(
            batch.adjacency_src,
            batch.adjacency_dst,
            classes,
            covered,
            batch.cell_pos.shape[0],
            batch.cell_pos.shape[0],
            cell_nodes.ADJACENCY_CLASSES,
            self.cfg.cell_node_scope == "uncovered",
        )

    def trunk(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Embeddings through the B blocks and the shared final LN (§5).

        The fourth output is the refined cell latents scattered into the
        decoder's legal order — ``(N_cells, H)``, already through the shared
        LN — when the knob is on, else ``None``. Uncovered legal cells carry
        the learned base row.
        """
        cfg = self.cfg
        s = self.stone_table(batch.stone_own)
        w = self.window_table(batch.window_feat)
        moves = self.token_moves(batch.moves_idx)
        g = self.latent_base[None, :, :] + moves[:, None, :]

        plan = message_passing.incidence_plan(
            batch.inc_stone,
            batch.inc_window,
            batch.inc_class,
        )
        pairs = self._pair_tables(batch) if cfg.window_attention else None
        window_pos = batch.window_slot // batch.max_w
        offsets, order = window_latents.window_latent_layout(
            window_pos, g.shape[0]
        )
        latent_layout = (window_pos, offsets, order)
        lines = self._line_tables(batch) if cfg.line_pass else None
        ctab = c = radius = adjacency = None
        if cfg.uses_cell_state:
            ctab = self._cell_tables(batch, w.shape[0])
            if cfg.cell_nodes:
                c = (
                    self.cell_occupancy_table(batch.cell_occupancy)
                    + self.cell_legal_table(batch.cell_is_legal)
                    + self.cell_nearest_table(batch.cell_nearest)
                )
                c = c.clone()
                c.index_copy_(
                    0,
                    ctab.covered,
                    self.cell_base.expand(ctab.covered.shape[0], cfg.h),
                )
                radius = self._radius_tables(batch, ctab.covered, s.shape[0])
                if cfg.cell_adjacency:
                    adjacency = self._adjacency_tables(batch, ctab.covered)
            else:
                c = self.cell_base.expand(ctab.covered.shape[0], cfg.h)
        seq_lens = batch.attn_valid.sum(dim=1, dtype=torch.int32)
        for block in self.blocks:
            s, w, g, c = block(
                s,
                w,
                g,
                c,
                batch,
                seq_lens,
                plan,
                pairs,
                latent_layout,
                lines,
                ctab,
                radius,
                adjacency,
            )
        global_rows = self.ln_out(g).mean(dim=1)
        cells = None
        if cfg.uses_cell_state:
            if cfg.cell_nodes:
                cells = self.ln_out(c)
            else:
                cells = (
                    self.ln_out(self.cell_base)
                    .expand(batch.cell_pos.shape[0], cfg.h)
                    .clone()
                )
                cells.index_copy_(0, ctab.covered, self.ln_out(c).to(cells.dtype))
        return self.ln_out(s), self.ln_out(w), global_rows, cells

    def _decoder_rows(self, w: Tensor, batch: Batch, dtype: torch.dtype) -> Tensor:
        """The pass over the decoder incidence, shared by both cell heads.

        ``dtype`` comes from a head linear the caller has already run, so the
        aggregation is built in whatever precision autocast chose for the head
        GEMMs that consume it, without this forward assuming which that is.

        The 726-class histogram is too wide, so the shared part is only the
        run-reduced window rows and each head adds its own class-row sum in
        ``_cell_scores``."""
        # The batch's decoder order is cell-major; collation supplies the
        # window-major view used by the backward reduction.
        with torch.no_grad():
            rev_gather = batch.dec_cell.index_select(0, batch.act_rev)
            rev_runs = batch.dec_window.index_select(0, batch.act_rev)
        return message_passing.incidence_row_sum(
            w,
            batch.dec_window,
            batch.dec_cell,
            rev_gather,
            rev_runs,
            batch.cell_pos.shape[0],
            message_passing.STONE_RUN,
            message_passing.WINDOW_RUN,
        ).to(dtype)

    def _decoder_input(
        self, w: Tensor, cells: Tensor | None, batch: Batch, dtype: torch.dtype
    ) -> Tensor:
        """The per-legal-cell rows both heads read: the trunk's refined cell
        latents when that knob is on, the one-shot window aggregate
        otherwise. A ``cells`` value that disagrees with the config is a
        caller wiring bug, not a fallback opportunity."""
        if self.cfg.uses_cell_state:
            if cells is None:
                raise ValueError(
                    "cell state is on but the trunk's cell rows were not "
                    "passed to the decoder"
                )
            return cells.to(dtype)
        if cells is not None:
            raise ValueError("cell rows were passed but cell state is off")
        return self._decoder_rows(w, batch, dtype)

    def _action_contribution(self, w: Tensor, batch: Batch) -> Tensor:
        """The summed row-encoder hidden per legal cell, fp32.

        Kept rows run the fused encoder over the decoder-order edge views;
        EMPTY rows are the per-orbit counts times the three shared hidden
        rows, an exact regrouping because the shared base makes every EMPTY
        row of one orbit identical.
        """
        pre = self.act_proj(w)
        acts = row_encoder.encode_rows(
            pre,
            self.act_table.weight,
            batch.dec_window,
            batch.act_class,
            batch.dec_cell,
            batch.act_rev,
            batch.cell_pos.shape[0],
        )
        empty_rows = F.relu(
            self.act_empty_base.float()
            + self.act_table.weight.index_select(0, self.act_empty_classes).float()
        )
        # Broadcast multiply-add rather than a matmul: the EMPTY term belongs
        # to the same fp32 segment sum as the kernel's, and autocast would
        # downcast a matmul's operands.
        counts = batch.act_empty.to(torch.float32)
        return acts + (counts.unsqueeze(-1) * empty_rows.unsqueeze(0)).sum(dim=1)

    def _cell_scores(
        self,
        rows: Tensor,
        acts: Tensor,
        g_half: Tensor,
        batch: Batch,
        lin: nn.Linear,
        e_w: nn.Embedding,
        extend: nn.Linear,
        mlp: _PairMlp,
    ) -> Tensor:
        """One head's ``(N_cells, d_out)`` readout rows, off the shared
        aggregation. The head's projection and both its embedding tables live
        in the matrix that reads an aggregate row; the token half of the MLP
        runs per position.

        The ternary scope composes the quantity without a histogram:
        ``lin_a(proj @ Σw + Σ e_class rows + extension) + bias``, the class
        sum run-reduced fp32. ``extend`` is the action-row extension matrix
        reading the shared encoder sum."""
        extra = message_passing.class_row_sum(
            e_w.weight,
            batch.dec_class,
            batch.dec_cell,
            rows.shape[0],
            message_passing.STONE_RUN,
        )
        base = F.linear(rows, lin.weight) + F.linear(
            acts.to(rows.dtype), extend.weight
        )
        base = base + extra.to(rows.dtype)
        pre = mlp.lin_a(base)
        return mlp.out(F.relu(pre + g_half.index_select(0, batch.cell_pos)))

    def policy_head(
        self, w: Tensor, g: Tensor, cells: Tensor | None, batch: Batch
    ) -> Tensor:
        """§6: one raw policy logit per legal cell, engine legal order."""
        acts = self._action_contribution(w, batch)
        g_half = self.mlp_p.lin_b(g)
        rows = self._decoder_input(w, cells, batch, g_half.dtype)
        return self._cell_scores(
            rows, acts, g_half, batch, self.p, self.e_pw, self.p_act, self.mlp_p
        ).squeeze(-1)

    def cell_head_logits(
        self, w: Tensor, g: Tensor, cells: Tensor | None, batch: Batch
    ) -> tuple[Tensor, Tensor]:
        """§6 policy logits ``(N,)`` and categorical logits ``(N, 3)``.

        Both heads read the same per-cell rows and own separate decoder
        parameters. Fitting takes one categorical score on the selected row;
        acting composes its score and value from the same logits, so one
        pass answers both.
        """
        acts = self._action_contribution(w, batch)
        g_p, g_q = self.mlp_p.lin_b(g), self.mlp_q.lin_b(g)
        rows = self._decoder_input(w, cells, batch, g_p.dtype)
        return (
            self._cell_scores(
                rows, acts, g_p, batch, self.p, self.e_pw, self.p_act, self.mlp_p
            ).squeeze(-1),
            self._cell_scores(
                rows, acts, g_q, batch, self.q, self.e_qw, self.q_act, self.mlp_q
            ),
        )

    def cell_heads(
        self,
        w: Tensor,
        g: Tensor,
        cells: Tensor | None,
        batch: Batch,
        mass_floor: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return policy logits, acting scores, and values in legal order.

        π′ ranks by the score; v̂ and the λ-return use the unscaled value.
        """
        policy_logits, critic_logits = self.cell_head_logits(w, g, cells, batch)
        return (
            policy_logits,
            compose_acting_q(critic_logits, batch.legal_offsets, mass_floor),
            compose_q(critic_logits),
        )

    def value_head(self, w: Tensor, g: Tensor, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        """§7: (value, value_dist, value_logits). Multi-query attention
        readout over [token; windows]. The LN runs on the concatenated rows
        before padding — row-wise, so identical, and the padded copy is
        written once."""
        cfg = self.cfg
        p, max_w = g.shape[0], batch.value_valid.shape[1]
        rows = g.new_zeros(p * max_w, cfg.h)
        token_slot = torch.arange(p, device=g.device) * max_w
        rows.index_copy_(0, token_slot, self.ln_value(g))
        if batch.window_slot.numel():
            rows.index_copy_(0, batch.window_slot, self.ln_value(w).to(rows.dtype))
        kv = rows.view(p, max_w, cfg.h)
        scores = torch.einsum("qh,pth->pqt", self.value_queries, kv) / math.sqrt(cfg.h)
        scores = scores.masked_fill(
            ~batch.value_valid[:, None, :], torch.finfo(scores.dtype).min
        )
        r = torch.einsum("pqt,pth->pqh", scores.softmax(dim=-1), kv)
        v_logits = self.mlp_v(r.reshape(p, cfg.value_queries * cfg.h))

        # Scalar decode in-forward, fp32, so every consumer sees the same value.
        value_dist = v_logits.float().softmax(dim=-1)
        value = value_dist @ self.bin_centers
        return value, value_dist, v_logits

    def forward(self, batch: Batch, mass_floor: float) -> ModelOutput:
        """Every head. KLENT's loop composes `trunk` with the two heads it
        trains instead, skipping the value readout it never reads."""
        s_, w, g, cells = self.trunk(batch)
        value, value_dist, value_logits = self.value_head(w, g, batch)
        policy_logits, q_score, q_values = self.cell_heads(
            w, g, cells, batch, mass_floor
        )
        return ModelOutput(
            policy_logits=policy_logits,
            q_score=q_score,
            q_values=q_values,
            value=value,
            value_dist=value_dist,
            value_logits=value_logits,
        )
