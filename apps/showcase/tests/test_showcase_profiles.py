"""Per-checkpoint search profiles: bots.toml `search_profile` resolution, the
shipped as-trained profile files (PUCT knobs, gumbel-off overrides), profile
routing in the worker runtime, and playing a legacy PUCT bot over the API."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from showcase.bots import PROFILES_DIR, SearchProfile, load_bots_toml

from test_showcase_api import create_game, fresh_ip, poll_until, resign

_REPO_ROOT = Path(__file__).resolve().parents[3]

_GUMBEL_OVERRIDE_KEYS = (
    "gumbel_root",
    "gumbel_sequential_halving",
    "gumbel_nonroot_select",
    "gumbel_target",
)


def _write_bots_toml(path: Path, checkpoint: Path, entries: list[str]) -> Path:
    blocks = []
    for index, extra in enumerate(entries):
        blocks.append(
            f"""[[checkpoint]]
id = "b{index}"
checkpoint = '{checkpoint.as_posix()}'
label = "B{index}"
run = "r"
epoch = {index}
{extra}"""
        )
    path.write_text("sims = [8]\n\n" + "\n".join(blocks))
    return path


# ---------------------------------------------------------------------------
# bots.toml resolution
# ---------------------------------------------------------------------------


def test_search_profile_resolution_order(tiny_checkpoint, tmp_path):
    """Bare name -> built-in profiles dir; relative path -> bots.toml dir;
    absolute path -> as given; absent -> None (global default)."""
    local = tmp_path / "local_profile.toml"
    shutil.copyfile(PROFILES_DIR / "hexfield_main_4.toml", local)
    absolute = (PROFILES_DIR / "hexfield_main_5.toml").resolve()
    catalogue = load_bots_toml(
        _write_bots_toml(
            tmp_path / "bots.toml",
            tiny_checkpoint,
            [
                "",
                'search_profile = "hexfield_main_5"',
                'search_profile = "local_profile.toml"',
                f"search_profile = '{absolute.as_posix()}'",
            ],
        )
    )
    profiles = [spec.search_profile for spec in catalogue.checkpoints]
    assert profiles[0] is None
    assert profiles[1] == PROFILES_DIR / "hexfield_main_5.toml"
    assert profiles[2] == local.resolve()
    assert profiles[3] == absolute
    # search_profile is a server key, never display metadata.
    assert all("search_profile" not in spec.meta for spec in catalogue.checkpoints)


def test_missing_profile_fails_at_catalogue_load(tiny_checkpoint, tmp_path):
    with pytest.raises(FileNotFoundError, match="no_such_profile"):
        load_bots_toml(
            _write_bots_toml(
                tmp_path / "bots.toml",
                tiny_checkpoint,
                ['search_profile = "no_such_profile"'],
            )
        )


def test_group_and_search_keys_are_scalar_meta(tiny_checkpoint, tmp_path):
    catalogue = load_bots_toml(
        _write_bots_toml(
            tmp_path / "bots.toml",
            tiny_checkpoint,
            ['group = "earlier runs"\nsearch = "puct"'],
        )
    )
    assert catalogue.checkpoints[0].meta == {"group": "earlier runs", "search": "puct"}


# ---------------------------------------------------------------------------
# shipped profile files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "search_visits"),
    [("hexfield_main_4", 512), ("hexfield_main_5", 1024)],
)
def test_legacy_profiles_parse_as_trained_puct(name, search_visits):
    """The distilled PUCT profiles carry the as-trained knob values and emit
    gumbel-off divergence overrides (the PUCT root runs)."""
    profile = SearchProfile(PROFILES_DIR / f"{name}.toml")
    for key in _GUMBEL_OVERRIDE_KEYS:
        assert profile.overrides[key] is False, key
    assert profile.overrides["lcb_z"] == 1.6
    assert profile.overrides["nucleus_f64"] is True
    assert profile.overrides["new_child_fpu"] is True
    assert profile.overrides["lazy_widening"] is False
    assert profile.overrides["clean_root_prior_cache"] is True
    assert profile.overrides["ml_two_sided"] is False
    assert profile.overrides["ml_final_pick"] is True
    assert profile.overrides["ml_final_pick_band"] == 0.08

    sp = profile.selfplay
    assert sp.search_visits == search_visits
    assert sp.c_puct == 1.5
    assert sp.fpu_reduction == 0.2
    assert sp.root_fpu_reduction == 0.2
    assert sp.widening_policy_mass == 0.95
    assert sp.widening_max_children == 96
    assert sp.widening_min_children == 2
    assert sp.tss_enabled is True
    assert sp.active_root_limit == 192
    assert sp.max_game_plies == 256
    assert sp.root_policy_temperature == 1.1
    assert sp.root_policy_temperature_early == 1.15
    assert sp.root_policy_temperature_halflife == 19.0

    # Eval-arena protocol knobs.
    assert profile.virtual_batch_size == 32
    assert profile.opening_plies == 8
    assert profile.opening_temperature == 1.0


def test_main7_profile_matches_global_default():
    """profiles/hexfield_main_7.toml and the SHOWCASE_SEARCH_CONFIG default
    (configs/hexfield_main_7.toml) must stay value-identical."""
    from_profile = SearchProfile(PROFILES_DIR / "hexfield_main_7.toml")
    from_config = SearchProfile(_REPO_ROOT / "configs" / "hexfield_main_7.toml")
    assert from_profile.selfplay == from_config.selfplay
    assert from_profile.overrides == from_config.overrides
    assert from_profile.virtual_batch_size == from_config.virtual_batch_size
    assert from_profile.opening_plies == from_config.opening_plies
    assert from_profile.opening_temperature == from_config.opening_temperature
    for key in _GUMBEL_OVERRIDE_KEYS:
        assert from_profile.overrides[key] is True, key


# ---------------------------------------------------------------------------
# worker routing
# ---------------------------------------------------------------------------


def test_worker_runtime_routes_profiles_per_checkpoint(tiny_checkpoint, tmp_path, settings):
    """Each bot searches with its own profile; bots sharing a profile share
    ONE parsed SearchProfile instance per worker."""
    from showcase.bots import _WorkerRuntime

    catalogue = load_bots_toml(
        _write_bots_toml(
            tmp_path / "bots.toml",
            tiny_checkpoint,
            ["", 'search_profile = "hexfield_main_5"', 'search_profile = "hexfield_main_5"'],
        )
    )
    runtime = _WorkerRuntime(list(catalogue.checkpoints), settings)
    default, legacy_a, legacy_b = (runtime.bots[f"b{i}"].profile for i in range(3))
    assert legacy_a is legacy_b  # one parse per unique profile
    assert legacy_a is not default
    assert default.overrides["gumbel_root"] is True  # main_7 global default
    assert legacy_a.overrides["gumbel_root"] is False  # as-trained PUCT


# ---------------------------------------------------------------------------
# end to end: play and analyze against the legacy catalogue entry
# ---------------------------------------------------------------------------


def test_legacy_puct_bot_plays_and_analyzes(client, settings):
    """The tiny-puct entry (main_5 profile from conftest's bots.toml) plays a
    full turn and serves a searched analysis through the worker pool."""
    headers = fresh_ip()
    snap = create_game(client, headers, checkpoint_id="tiny-puct", sims=8)
    game_id = snap["id"]
    assert snap["bot"]["checkpoint_id"] == "tiny-puct"
    client.post(f"/api/game/{game_id}/move", json={"q": 0, "r": 0}, headers=headers)
    snap = poll_until(client, game_id)  # the PUCT-profile search produced a turn
    assert snap["ply"] >= 3 or snap["status"] == "finished"
    resign(client, game_id, headers)

    resp = client.get(
        f"/api/game/{game_id}/analysis",
        params={"ply": 2, "search": "1"},
        headers=fresh_ip(),
    )
    assert resp.status_code == 200, resp.text
    search = resp.json()["search"]
    assert 1 <= search["visits"] <= settings.analysis_search_visit_cap
    assert -1.0 <= search["root_value"] <= 1.0
