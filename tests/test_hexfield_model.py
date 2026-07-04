"""Tests for the hexfield network.

Covers:
- parameter count (8 conv + 3 attn blocks, per-block bias tables, cell_q head)
- sdpa vs materialized attention equality (fp32)
- HexNodeConv vs a dense 2D grid convolution on embedded supports (fp64)
- padded-batch vs single-row forward identity
- per-row gradient accumulation vs monolithic batch
- train-mode vs eval-mode output parity
- gradients reach every parameter
- pair-index build vs geometry.rel_bias_index
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hexfield_testkit import api, sample_decision_states

from hexfield import constants as C
from hexfield.batching import collate_rows
from hexfield.engine_facts import facts_from_engine
from hexfield.features import build_position
from hexfield.geometry import rel_bias_index
from hexfield.model import HexfieldNet, _BiasGather

# Sum of p.numel() over a fresh HexfieldNet() (8 conv + 3 attn blocks,
# 3 per-block bias tables, per-node policy/opp_policy/soft_policy/cell_q heads,
# plus the value / short-term-value / moves-left readout heads).
EXPECTED_PARAMS = 1_656_453


def _rows(count: int = 3):
    states = sample_decision_states(range(4), (3, 9, 17, 27))
    rows = []
    for state in states[:count]:
        facts = facts_from_engine(api.to_python_state(state))
        rows.append(build_position(facts))
    assert len(rows) == count
    return rows


def _derandomize(model: HexfieldNet, seed: int = 5) -> None:
    """Fill residual-branch params with small randoms so attention/conv
    branches produce nonzero output in numerical tests."""

    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for block in model.conv_blocks:
            block.ln2.weight.copy_(torch.rand(block.ln2.weight.shape, generator=gen) * 0.5 + 0.5)
        for block in model.attn_blocks:
            for p in (block.attn.out_proj.weight, block.fc2.weight):
                p.copy_(torch.randn(p.shape, generator=gen) * 0.05)
        for table in model.bias_tables:
            table.copy_(torch.randn(table.shape, generator=gen) * 0.1)


def test_param_count_matches_spec_section_9() -> None:
    model = HexfieldNet()
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS


def test_sdpa_equals_materialized() -> None:
    torch.manual_seed(0)
    model = HexfieldNet().eval()
    _derandomize(model)
    batch = collate_rows(_rows(3))
    args = (batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    with torch.no_grad():
        model.set_attention_impl("sdpa")
        out_sdpa = model(*args)
        model.set_attention_impl("materialized")
        out_mat = model(*args)
    for key in out_sdpa:
        diff = (out_sdpa[key] - out_mat[key]).abs().max().item()
        assert diff <= 1e-4, f"{key}: sdpa vs materialized diff {diff}"


def test_conv_oracle_against_dense_grid_conv() -> None:
    """HexNodeConv (gather over the 6 axial neighbours + self) equals a dense
    2D 3x3 convolution on the embedded hex grid, tap-for-tap (fp64).

    The reference is an explicit ``F.conv2d`` on a 41x41 grid where each support
    cell is placed at ``(r+20, q+20)``. The direction->kernel mapping is the
    axial-to-offset embedding used throughout hexfield: tap 0 is the centre tap
    ``(1,1)`` and axial direction ``(dq,dr)`` lands at kernel slot
    ``(dr+1, dq+1)``. This pins the same numerical contract the retired
    dense_cnn ``HexConv2d`` oracle used to, using only in-repo pieces.
    """

    import torch.nn.functional as F

    from hexfield.model import HexNodeConv
    from hexfield.support import build_support

    torch.manual_seed(1)
    cin, cout = 8, 8
    stones = [(0, 0), (1, 0), (0, 1), (2, -1), (1, 1), (-1, 2), (3, 0)]
    sup = build_support(stones)
    n = sup.num_nodes
    assert int(np.abs(sup.coords).max()) + 1 < 20  # fits within the 41x41 embed grid

    mine = HexNodeConv(cin, cout).double()

    # Build the dense 3x3 kernel from HexNodeConv's per-direction weights.
    # F.conv2d weight layout is (cout, cin, kh, kw); HexNodeConv.weight[k] is
    # (cin, cout), so transpose each tap into place.
    kernel = torch.zeros(cout, cin, 3, 3, dtype=torch.float64)
    bias = mine.bias.detach().clone()
    with torch.no_grad():
        kernel[:, :, 1, 1] = mine.weight[0].T  # centre / self tap
        for k, (dq, dr) in enumerate(C.DIRECTIONS):
            kernel[:, :, dr + 1, dq + 1] = mine.weight[k + 1].T

    feats = torch.randn(1, n, cin, dtype=torch.float64)
    grid = torch.zeros(1, cin, 41, 41, dtype=torch.float64)
    for row, (q, r) in enumerate(sup.coords.tolist()):
        grid[0, :, r + 20, q + 20] = feats[0, row]

    self_idx = torch.arange(n).reshape(1, n, 1)
    nbr = torch.from_numpy(sup.nbr.astype(np.int64)).unsqueeze(0)
    nbr = torch.where(nbr >= 0, nbr, torch.full_like(nbr, n))
    gather_idx = torch.cat([self_idx, nbr], dim=2)
    mask = torch.ones(1, n, dtype=torch.bool)

    with torch.no_grad():
        out_mine = mine(feats, gather_idx, mask)
        out_dense = F.conv2d(grid, kernel, bias=bias, padding=1)

    for row, (q, r) in enumerate(sup.coords.tolist()):
        expected = out_dense[0, :, r + 20, q + 20]
        assert torch.allclose(out_mine[0, row], expected, atol=1e-12), (
            f"conv mismatch at node {row} ({q},{r})"
        )


def test_padded_batch_equals_single_row() -> None:
    torch.manual_seed(2)
    model = HexfieldNet().eval()
    _derandomize(model)
    rows = _rows(2)
    rows.sort(key=lambda item: item[0].num_nodes)
    small, large = rows
    batch = collate_rows([small, large])  # small row padded up to large's N
    alone = collate_rows([small])
    with torch.no_grad():
        out_batch = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        out_alone = model(alone["feats"], alone["nbr"], alone["mask"], alone["coords"])
    n = small[0].num_nodes
    # Per-node heads (B, Npad[, BINS]): policy, opp_policy, soft_policy, and
    # cell_q are zero on pad rows. Slice to the row's real nodes for the
    # identity check and assert the pad tail is zero.
    per_node_heads = ("policy", "opp_policy", "soft_policy", "cell_q")
    for key in out_alone:
        a = out_alone[key][0]
        b = out_batch[key][0]
        if key in per_node_heads:
            a, b = a[:n], b[:n]
            # pad logits beyond the row's nodes are zero
            assert out_batch[key][0][n:].abs().max().item() == 0.0
        diff = (a - b).abs().max().item()
        assert diff <= 1e-6, f"{key}: padded vs single-row diff {diff}"


def test_per_row_grad_accumulation_equals_monolithic() -> None:
    """LayerNorm uses no cross-row statistics, so summed per-row losses give
    identical gradients whether rows share a batch or not."""

    torch.manual_seed(3)
    rows = _rows(2)

    def to_double(batch):
        return {
            k: v.double() if v.dtype == torch.float32 else v for k, v in batch.items()
        }

    def row_loss(model: HexfieldNet, batch) -> torch.Tensor:
        out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        # Pad logits are zero, so summing over every output including pads is a
        # per-row sum; every head contributes.
        return sum(v.square().sum() for v in out.values())

    # fp64: tolerance covers summation ordering noise only.
    model = HexfieldNet().double()
    _derandomize(model)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    row_loss(model, to_double(collate_rows(rows))).backward()
    mono = {name: p.grad.detach().clone() for name, p in model.named_parameters()}

    model2 = HexfieldNet().double()
    model2.load_state_dict(state)
    row_loss(model2, to_double(collate_rows([rows[0]]))).backward()
    row_loss(model2, to_double(collate_rows([rows[1]]))).backward()

    for name, p in model2.named_parameters():
        scale = 1.0 + mono[name].abs().max().item()
        diff = (p.grad - mono[name]).abs().max().item()
        assert diff <= 1e-10 * scale, (
            f"{name}: accumulated vs monolithic grad diff {diff} (scale {scale})"
        )


def test_train_eval_bit_parity() -> None:
    torch.manual_seed(4)
    model = HexfieldNet()
    _derandomize(model)
    batch = collate_rows(_rows(2))
    args = (batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    with torch.no_grad():
        model.train()
        out_train = model(*args)
        model.eval()
        out_eval = model(*args)
    for key in out_train:
        assert torch.equal(out_train[key], out_eval[key]), key


def test_grads_reach_every_param() -> None:
    torch.manual_seed(5)
    model = HexfieldNet()
    _derandomize(model)
    batch = collate_rows(_rows(2))
    out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    loss = sum(v.square().sum() for v in out.values())
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_pair_index_matches_geometry() -> None:
    model = HexfieldNet()
    with torch.no_grad():
        # Seed each block's head-0 column with its row id, so a gathered bias
        # value reads back the bias-table row class.
        for table in model.bias_tables:
            table.zero_()
            table[:, 0] = torch.arange(C.BIAS_ROWS, dtype=torch.float32)
    batch = collate_rows(_rows(1))
    coords, mask = batch["coords"], batch["mask"]
    # _build_pair computes the block-independent (pair, key_pad); build_attn_bias
    # materializes the (1, heads, S, S) bias for a given attention block.
    pair, key_pad = model._build_pair(coords, mask)
    bias = model.build_attn_bias(pair, key_pad, 0)  # (1, heads, S, S)
    t = C.NUM_TOKENS
    cells = coords[0][mask[0]].tolist()
    # token/token and token/cell classes
    assert bias[0, 0, 0, 1].item() == C.BIAS_TOKEN_TOKEN_ROW
    assert bias[0, 0, 0, t].item() == C.BIAS_TOKEN_CELL_ROW
    assert bias[0, 0, t, 0].item() == C.BIAS_CELL_TOKEN_ROW
    # cell/cell rows equal the geometry function of (key - query) offsets
    idx = np.random.RandomState(0).choice(len(cells), size=min(40, len(cells)), replace=False)
    for i in idx:
        for j in idx[:10]:
            qi, ri = cells[int(i)]
            qj, rj = cells[int(j)]
            expected = rel_bias_index(qj - qi, rj - ri)
            assert bias[0, 0, t + int(i), t + int(j)].item() == expected


def test_bias_gather_backward_equals_generic_indexing() -> None:
    """_BiasGather's bincount backward equals the generic advanced-indexing
    backward (table[pair]) in fp64 and fp32. A small table, random pair matrix,
    and random upstream weight isolate the gradient path with no model state."""

    for dtype in (torch.float64, torch.float32):
        torch.manual_seed(0)
        rows, heads, n = C.BIAS_ROWS, C.ATTENTION_HEADS, 37
        table_ref = torch.randn(rows, heads, dtype=dtype, requires_grad=True)
        table_gather = table_ref.detach().clone().requires_grad_(True)
        pair = torch.randint(0, rows, (n, n), dtype=torch.long)
        upstream = torch.randn(n, n, heads, dtype=dtype)

        # generic indexing backward (table[pair]): reference
        (table_ref[pair] * upstream).sum().backward()
        # _BiasGather backward: compared for exact equality below
        (_BiasGather.apply(table_gather, pair) * upstream).sum().backward()

        assert torch.equal(table_gather.grad, table_ref.grad), (
            f"{dtype}: _BiasGather grad differs from indexing grad "
            f"(max {(table_gather.grad - table_ref.grad).abs().max().item()})"
        )


def test_fresh_model_zero_init_residual_identity() -> None:
    """A fresh HexfieldNet (not _derandomized) has LayerScale gamma == 1e-4 on
    every residual branch (each ConvBlock.ls, each AttnBlock ls_attn/ls_mlp),
    all per-block relative-position bias tables exactly zero, and a finite
    forward pass."""

    torch.manual_seed(7)
    model = HexfieldNet()
    expected_gamma = torch.full_like(model.conv_blocks[0].ls.gamma, 1e-4)
    for i, block in enumerate(model.conv_blocks):
        assert torch.equal(block.ls.gamma, expected_gamma), f"conv_blocks[{i}].ls.gamma"
    for i, block in enumerate(model.attn_blocks):
        assert torch.equal(block.ls_attn.gamma, expected_gamma), f"attn_blocks[{i}].ls_attn.gamma"
        assert torch.equal(block.ls_mlp.gamma, expected_gamma), f"attn_blocks[{i}].ls_mlp.gamma"
    for i, table in enumerate(model.bias_tables):
        assert torch.count_nonzero(table).item() == 0, f"bias_tables[{i}]"

    b, n = 2, 11
    feats = torch.randn(b, n, C.NUM_FEATURES)
    nbr = torch.randint(0, n, (b, n, 6), dtype=torch.long)
    mask = torch.ones(b, n, dtype=torch.bool)
    coords = torch.randint(-8, 9, (b, n, 2), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        out = model(feats, nbr, mask, coords)
    for key, value in out.items():
        assert torch.isfinite(value).all(), f"fresh-model forward non-finite at {key}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 fused SDPA needs CUDA")
def test_sdpa_equals_materialized_fp16_cuda() -> None:
    """sdpa (fused fp16 kernel) vs materialized attention under cuda fp16
    autocast. Tolerance is an fp16-rounding budget."""

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = HexfieldNet().eval().to(device)
    # Derandomize so attention branches produce nonzero output: set out_proj/fc2
    # to small randoms, open the LayerScale gammas to 1.0, and seed each
    # per-block bias table.
    gen = torch.Generator(device=device).manual_seed(11)
    with torch.no_grad():
        for block in model.attn_blocks:
            block.attn.out_proj.weight.copy_(
                torch.randn(block.attn.out_proj.weight.shape, generator=gen, device=device) * 0.05
            )
            block.fc2.weight.copy_(
                torch.randn(block.fc2.weight.shape, generator=gen, device=device) * 0.05
            )
            block.ls_attn.gamma.fill_(1.0)
            block.ls_mlp.gamma.fill_(1.0)
        for table in model.bias_tables:
            table.copy_(torch.randn(table.shape, generator=gen, device=device) * 0.1)

    b, n = 3, 40
    feats = torch.randn(b, n, C.NUM_FEATURES, device=device)
    nbr = torch.randint(0, n, (b, n, 6), dtype=torch.long, device=device)
    mask = torch.ones(b, n, dtype=torch.bool, device=device)
    mask[2, -5:] = False  # padded row exercises the pad-key mask
    coords = torch.randint(-8, 9, (b, n, 2), dtype=torch.long, device=device)
    args = (feats, nbr, mask, coords)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        model.set_attention_impl("sdpa")
        out_sdpa = model(*args)
        model.set_attention_impl("materialized")
        out_mat = model(*args)

    for key in out_sdpa:
        diff = (out_sdpa[key].float() - out_mat[key].float()).abs().max().item()
        assert diff <= 2e-3, f"{key}: fp16 sdpa vs materialized diff {diff}"
