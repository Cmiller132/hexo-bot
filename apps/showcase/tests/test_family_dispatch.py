"""Catalogue and worker coverage for the model-family adapter boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from showcase.bots import load_bots_toml
from showcase.families import get_family
from showcase.families.hexfield_eq_family import HexfieldEqFamily
from showcase.families.shrimp_family import ShrimpFamily


def _catalogue(path: Path, checkpoint: Path, family_line: str = "") -> Path:
    path.write_text(
        f'''[[checkpoint]]
id = "test"
checkpoint = "{checkpoint.as_posix()}"
label = "Test"
run = "test"
epoch = 1
{family_line}
''',
        encoding="utf-8",
    )
    return path


def test_catalogue_family_defaults_to_shrimp(tiny_checkpoint, tmp_path):
    catalogue = load_bots_toml(_catalogue(tmp_path / "bots.toml", tiny_checkpoint))
    assert catalogue.checkpoints[0].family == "shrimp"
    assert "family" not in catalogue.checkpoints[0].meta


def test_catalogue_parses_family(tiny_checkpoint, tmp_path):
    catalogue = load_bots_toml(
        _catalogue(tmp_path / "bots.toml", tiny_checkpoint, 'family = "hexfield_eq"')
    )
    assert catalogue.checkpoints[0].family == "hexfield_eq"


def test_catalogue_rejects_unknown_family(tiny_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="unknown model family.*future_net"):
        load_bots_toml(
            _catalogue(tmp_path / "bots.toml", tiny_checkpoint, 'family = "future_net"')
        )


def test_family_registry_dispatches_and_rejects_unknown():
    assert isinstance(get_family("shrimp"), ShrimpFamily)
    assert isinstance(get_family("hexfield_eq"), HexfieldEqFamily)
    with pytest.raises(ValueError, match="unknown model family"):
        get_family("nope")


def test_registered_families_expose_analysis_and_lab_hooks():
    hooks = (
        "net_eval", "searched_eval", "summary_row", "summary_eval",
        "lab_eval_payload", "lab_search_payload",
    )
    for name in ("shrimp", "hexfield_eq"):
        family = get_family(name)
        assert all(callable(getattr(family, hook, None)) for hook in hooks)


def test_hexfield_prepare_rejects_conflicting_arches(monkeypatch, tmp_path):
    import torch

    base = {
        "channels": 192,
        "group_order": 12,
        "c_orbit": 16,
        "attention_heads": 3,
        "support_radius": 4,
        "trunk_layout": "CCACCACA",
        "reg_lane": True,
        "reg_tok_read": False,
        "feature_version": 2,
        "raytap": "both",
    }
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    payloads = {
        a: {"meta": base},
        b: {"meta": {**base, "channels": 288, "c_orbit": 24}},
    }
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: payloads[Path(path)])
    specs = [SimpleNamespace(checkpoint=a), SimpleNamespace(checkpoint=b)]
    with pytest.raises(RuntimeError, match="only one hexfield_eq arch; split the catalogue"):
        HexfieldEqFamily().prepare_process(specs)


EP70 = Path(
    "/mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_2/checkpoints/epoch_000070.pt"
)


def _hexfield_serve_available() -> tuple[bool, str]:
    if not EP70.is_file():
        return False, f"ep70 checkpoint unavailable: {EP70}"
    package = Path(__file__).resolve().parents[3] / "packages/hexfield_eq/python/hexfield_eq"
    if not list(package.glob("_rust*.so")) and not list(package.glob("_rust*.pyd")):
        return False, "hexfield_eq native extension unavailable"
    return True, ""


def test_hexfield_eq_ep70_worker_plays_and_analyzes(tmp_path, settings):
    available, reason = _hexfield_serve_available()
    if not available:
        pytest.skip(reason)

    profile = Path(__file__).resolve().parents[3] / "configs" / "hexfield_eq_main_2.toml"
    path = tmp_path / "bots.toml"
    path.write_text(
        f'''sims = [8]
[[checkpoint]]
id = "hexfield-main2-ep70"
family = "hexfield_eq"
checkpoint = "{EP70.as_posix()}"
label = "Hexfield EQ main_2 ep70"
run = "hexfield_eq_main_2"
epoch = 70
search_profile = "{profile.as_posix()}"
''',
        encoding="utf-8",
    )

    # Import only after load_bots_toml: the catalogue path itself must stay
    # torch/model-stack free. Runtime.prepare_process seeds arch env before
    # the first hexfield_eq.model import.
    from showcase.bots import _WorkerRuntime

    catalogue = load_bots_toml(path)
    runtime = _WorkerRuntime(
        list(catalogue.checkpoints), settings, device_override="cpu"
    )
    result = runtime.bot_turn(
        bot_slug="hexfield-main2-ep70", game_key=91, actions=[], seed=7, visits=8
    )
    assert result["actions"]

    from hexo_engine import api

    legal = set(api.legal_action_ids(api.new_game()))
    assert result["actions"][0]["action_id"] in legal
    analysis = runtime.analyze(
        bot_slug="hexfield-main2-ep70", actions=[], want_search=True,
        search_visits=8, seed=7,
    )
    assert -1.0 <= analysis["value"] <= 1.0
    assert analysis["stv"] is not None
    assert set(analysis["stv_horizons"]) == {"2", "6", "16"}
    assert analysis["policy"]
    assert analysis["search"]["visit_policy"]
    assert all(
        (row["q"], row["r"]) == (0, 0)
        for row in analysis["search"]["visit_policy"]
    )

    summary = runtime.summary(bot_slug="hexfield-main2-ep70", actions=[])
    assert len(summary["stv"]) == 1 and summary["stv"][0] is not None
    assert set(summary["stv_horizons"][0]) == {"2", "6", "16"}

    lab_eval = runtime.lab_eval(
        bot_slug="hexfield-main2-ep70", actions=[], stones=None, to_move=None,
        attention_cell=(0, 0), want_activations=True, want_features=True,
    )
    assert lab_eval["mode"] == "sequence"
    assert set(lab_eval["stv"]) == {"2", "6", "16"}
    assert lab_eval["policy"] == [{"q": 0, "r": 0, "p": 1.0}]
    assert len(lab_eval["features"]["names"]) == 46
    assert lab_eval["features"]["names"][-3:] == [
        "ply", "dist_centroid", "spread",
    ]
    assert len(lab_eval["features"]["planes"]) == 46
    assert lab_eval["attention"]["available"] is False
    assert lab_eval["activations"]["available"] is False
    assert lab_eval["attention"]["reason"]

    lab_search = runtime.lab_search(
        bot_slug="hexfield-main2-ep70", actions=[], visits=8, seed=11,
    )
    assert lab_search["visit_policy"]
    assert lab_search["best"] == {"q": 0, "r": 0}
    assert all(
        (row["q"], row["r"]) == (0, 0)
        for row in lab_search["visit_policy"]
    )
