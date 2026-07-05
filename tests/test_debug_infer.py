"""Tests for the dashboard Debug-tab inference library + worker service.

The debug worker serves the **shrimp** lineage. These tests cover:
  * pure helpers (WSL path translation, always run);
  * ``_detect_lineage`` accept/reject (the only lineage is shrimp);
  * the full inference surface (load / analyze / search / search-tree /
    attention) against a *synthetic* smoke-size shrimp checkpoint built from a
    fresh ``ShrimpNet`` — no real run dir or GPU required;
  * the lineage-agnostic recorded-.npz row reader and the server-layer
    imported-position reconstruction.

Everything that needs torch/shrimp is guarded with ``importorskip`` so the
suite skips cleanly where those are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make shrimp importable from the source tree (never pip-installed; see README).
_SHRIMP_SRC = Path(__file__).resolve().parent.parent / "packages" / "shrimp" / "python"
if _SHRIMP_SRC.is_dir() and str(_SHRIMP_SRC) not in sys.path:
    sys.path.insert(0, str(_SHRIMP_SRC))

# Pure helpers (no torch) — always importable.
from hexo_frontend import debug_service


# --------------------------------------------------------------------------
# Pure: WSL path translation (runs everywhere).
# --------------------------------------------------------------------------


def test_to_wsl_translates_windows_drive_paths():
    assert debug_service._to_wsl("E:\\hexo-bot\\runs\\x.pt") == "/mnt/e/hexo-bot/runs/x.pt"
    assert debug_service._to_wsl("C:\\a\\b") == "/mnt/c/a/b"


def test_to_wsl_passes_through_posix_paths():
    assert debug_service._to_wsl("/mnt/e/runs/x.pt") == "/mnt/e/runs/x.pt"
    assert debug_service._to_wsl("relative/path") == "relative/path"


# --------------------------------------------------------------------------
# Inference surface: gated on torch + shrimp.
# --------------------------------------------------------------------------

di = pytest.importorskip("hexo_frontend.debug_infer", reason="needs torch + shrimp")
torch = pytest.importorskip("torch")


# ---- lineage detection (pure, no model) ----------------------------------


def test_detect_lineage_accepts_shrimp_payload():
    payload = {"meta": {"lineage": "shrimp", "epoch": 0}, "model": {"stem.bias": 1}}
    assert di._detect_lineage(payload) == di.SHRIMP


@pytest.mark.parametrize(
    "payload",
    [
        {"model": {"stem.bias": 1}},  # state dict but no meta/lineage
        {"model": "dense_cnn_restnet", "model_state": {}},  # legacy dense tag
        {"model": {"w": 1}, "arch": {"blocks": 3}},  # legacy hexgt shape
        {"meta": {"lineage": "hexgt"}, "model": {"w": 1}},  # wrong lineage
        {"meta": {"lineage": "shrimp"}, "model": "not-a-dict"},  # model not a state dict
    ],
)
def test_detect_lineage_rejects_non_shrimp(payload):
    with pytest.raises(ValueError, match="shrimp"):
        di._detect_lineage(payload)


def test_detect_lineage_rejects_non_dict():
    with pytest.raises(ValueError, match="not a dict"):
        di._detect_lineage(["not", "a", "dict"])


# ---- synthetic checkpoint fixture ----------------------------------------


@pytest.fixture(scope="module")
def shrimp_checkpoint(tmp_path_factory) -> Path:
    """A real (smoke-size, env-default arch) shrimp checkpoint on disk.

    Builds a fresh ``ShrimpNet`` at the default arch (channels/heads/trunk from
    the process env; the default trunk ``CCCACCCACCA`` is a known 8-conv/3-attn
    layout the loader can reconstruct exactly), dumps its ``state_dict`` under the
    shrimp payload shape, and saves it. Random weights are fine — the tests
    assert shapes, ranges, and determinism, not learned behavior."""

    from shrimp.model import ShrimpNet

    model = ShrimpNet()
    payload = {
        "meta": {"lineage": "shrimp", "epoch": 3, "run": "test_synthetic"},
        "model": model.state_dict(),
        "optimizer": None,
    }
    path = tmp_path_factory.mktemp("ckpt") / "epoch_000003.pt"
    torch.save(payload, path)
    return path


@pytest.fixture(scope="module")
def loaded(shrimp_checkpoint):
    lm = di.load_checkpoint(shrimp_checkpoint)
    assert lm.lineage == di.SHRIMP
    return lm


def _legal_actions(n: int = 6) -> list[int]:
    """A short legal action prefix (no recorded game needed)."""

    from hexo_engine.types import unpack_coord_id

    state = di.engine.new_game()
    actions: list[int] = []
    for _ in range(n):
        legal = list(di.engine.legal_action_ids(state))
        if not legal:
            break
        aid = int(legal[0])
        actions.append(aid)
        di.engine.apply_action(state, di.engine.PlacementAction(unpack_coord_id(aid)))
    return actions


# ---- load ----------------------------------------------------------------


def test_load_shrimp_checkpoint(loaded):
    assert loaded.lineage == di.SHRIMP
    assert loaded.rl_epoch == 3
    assert loaded.has_moves_left is True
    assert loaded.stv_horizons  # non-empty; from the model constant
    assert not loaded.model.training  # eval mode
    # Env-default arch is the known (8 conv, 3 attn) / 4-head layout.
    assert loaded.arch["attention_heads"] == 4
    assert loaded.arch["trunk_layout"] == "CCCACCCACCA"


# ---- analyze -------------------------------------------------------------


def test_analyze_position_shapes(loaded):
    actions = _legal_actions(6)
    result = di.analyze_position(loaded, actions)

    assert result["current_player"] in (0, 1)
    assert -1.0 <= result["value"] <= 1.0
    assert len(result["value_dist"]) == len(result["value_bins"])
    assert abs(sum(result["value_dist"]) - 1.0) < 1e-3

    policy = result["policy"]
    assert policy
    probs = [row["p"] for row in policy]
    assert probs == sorted(probs, reverse=True)  # sorted descending
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-2

    # STV heads present, scalar in range, horizons match the model's constant.
    assert set(result["stvalue"]) == {str(h) for h in loaded.stv_horizons}
    for head in result["stvalue"].values():
        assert -1.0 <= head["scalar"] <= 1.0


# ---- search --------------------------------------------------------------


def test_search_position_is_deterministic_and_normalized(loaded):
    actions = _legal_actions(6)
    s1 = di.search_position(loaded, actions, visits=32, seed=7)
    s2 = di.search_position(loaded, actions, visits=32, seed=7)
    pick = lambda s: [(r["action_id"], round(r["p"], 5)) for r in s["visit_policy"]]
    assert pick(s1) == pick(s2)  # deterministic (no root noise)

    assert s1["visits"] >= 1
    assert -1.0 <= s1["root_value"] <= 1.0
    assert s1["visit_policy"], "search returned an empty visit policy"
    assert abs(sum(row["p"] for row in s1["visit_policy"]) - 1.0) < 1e-2
    best = s1["best_action_id"]
    assert any(row["action_id"] == best for row in s1["visit_policy"])


# ---- pure-Python debug PUCT tree (kept live feature) ---------------------


def test_search_tree_is_deterministic(loaded):
    import json

    actions = _legal_actions(6)
    t1 = di.search_tree_position(loaded, actions, visits=48, seed=3)
    t2 = di.search_tree_position(loaded, actions, visits=48, seed=3)
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)
    assert t1["engine"] == "py_debug"
    assert t1["visits"] == 48
    assert t1["pv"] and t1["pv"][0] == t1["best_action_id"]
    assert -1.0 <= t1["root_value"] <= 1.0


def test_search_tree_respects_pruning_and_node_cap(loaded):
    actions = _legal_actions(6)
    tree = di.search_tree_position(loaded, actions, visits=64, top_k=2, min_n=0, max_depth=3)

    def walk(node, depth):
        assert depth <= 3
        assert len(node["children"]) <= 2
        return 1 + sum(walk(child, depth + 1) for child in node["children"])

    assert walk(tree["tree"], 0) == tree["node_count"]
    assert tree["node_count"] <= 4000


# ---- interactive attention map -------------------------------------------


def test_attention_shrimp_token_rows(loaded):
    actions = _legal_actions(6)
    res = di.attention_position(loaded, actions, block=0, head=None, query={"type": "token", "id": 0})

    assert res["found"] is True
    assert res["lineage"] == di.SHRIMP
    # Env-default arch: 3 attn blocks, 4 heads, 8 tokens.
    assert res["num_blocks"] == 3 and res["num_heads"] == 4 and res["num_tokens"] == 8
    n_cells = res["num_cells"]
    assert n_cells == len(res["cells"]) > 0
    for j, cell in enumerate(res["cells"]):
        assert cell["i"] == 8 + j

    assert len(res["token_queries"]) == 8
    for t, row in enumerate(res["token_queries"]):
        assert row["token"] == t
        assert len(row["attn_over_cells"]) == n_cells
        assert len(row["attn_over_tokens"]) == 8
        total = sum(row["attn_over_cells"]) + sum(row["attn_over_tokens"])
        assert abs(total - 1.0) < 1e-3
    assert res["cell_query"] is None
    assert res["ply"] == len(actions)


def test_attention_block_head_clamped_not_errored(loaded):
    actions = _legal_actions(6)
    clamped = di.attention_position(loaded, actions, block=99, head=99, query={"type": "token", "id": 0})
    assert clamped["found"] is True
    assert clamped["block"] == 2 and clamped["head"] == 3  # clamped to last block/head


def test_attention_bad_cell_query(loaded):
    actions = _legal_actions(6)
    res = di.attention_position(loaded, actions, block=0, head=None, query={"type": "cell", "id": -1})
    assert res["found"] is False
    assert res["reason"] == "bad_query"
    assert res["lineage"] == di.SHRIMP
    assert res["cell_query"] is None


# --------------------------------------------------------------------------
# Lineage-agnostic recorded-.npz row reader (never-raises contract).
# --------------------------------------------------------------------------


def test_record_row_missing_shard_reports_not_found():
    result = di.read_record_row("/definitely/not/a/real/shard.npz", 4, 0)
    assert result["found"] is False
    assert result["reason"] == "no_shard"
    assert result["row"] is None


def test_record_row_bad_shard_reports_found_false(tmp_path):
    """A foreign/partial .npz missing expected arrays yields found:false with
    reason 'bad_shard', never a KeyError."""
    np = pytest.importorskip("numpy")

    no_index = tmp_path / "no_index.npz"
    np.savez(no_index, num_rows=np.asarray(1, dtype=np.int64))  # no turn_index at all
    result = di.read_record_row(str(no_index), 0, None)
    assert result["found"] is False and result["reason"] == "bad_shard"
    assert result["row"] is None

    partial = tmp_path / "partial.npz"  # row indexable, decode arrays absent
    np.savez(
        partial,
        num_rows=np.asarray(1, dtype=np.int64),
        turn_index=np.asarray([0], dtype=np.int32),
    )
    result = di.read_record_row(str(partial), 0, None)
    assert result["found"] is False and result["reason"] == "bad_shard"
    assert result["row"] is None
    # An unmatched turn on the same shard is the plain no_row notice.
    assert di.read_record_row(str(partial), 5, None)["reason"] == "no_row"


# --------------------------------------------------------------------------
# Server-layer: imported-position reconstruction (needs engine build).
# --------------------------------------------------------------------------

web = pytest.importorskip("hexo_frontend.web", reason="needs hexo_runner/engine build")


def test_position_from_actions_reconstructs_board():
    actions = _legal_actions(6)
    csv = ", ".join(str(a) for a in actions)
    payload = web._debug_position_from_actions("some_run", csv, 0)
    assert payload["debug"]["imported"] is True
    assert payload["debug"]["total"] == 6
    assert payload["debug"]["ply"] == 6  # ply<=0 defaults to all moves
    assert len(payload.get("placements", [])) == 6


def test_position_from_actions_rejects_garbage():
    with pytest.raises(ValueError):
        web._debug_position_from_actions("run", "12, not_a_number", 0)
    with pytest.raises(ValueError):
        web._debug_position_from_actions("run", "   ", 0)


def test_debug_run_root_override(tmp_path, monkeypatch):
    """HEXO_DEBUG_RUN_ROOT is searched first; unset falls back to the cwd roots."""
    root = tmp_path / "wt"
    (root / "runs").mkdir(parents=True)
    monkeypatch.setenv("HEXO_DEBUG_RUN_ROOT", str(root))
    roots = [p.resolve() for p in web._debug_training_roots()]
    assert (root / "runs").resolve() == roots[0]
    assert len(roots) >= 1

    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    fallback = [p.resolve() for p in web._debug_training_roots()]
    assert (root / "runs").resolve() not in fallback
    assert fallback == [p.resolve() for p in web._training_roots()]
