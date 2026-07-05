"""Phase 1 self-test (CPU-only): KataGo replay-buffer config fields +
ShrimpTrainState checkpoint save/load wiring.

Covers PLAN §6 (checkpoints.py/config.py) / §9 test 5 (resume failure modes):
  (0) the modules import (config + checkpoints + trainer);
  (0b) the 11 new TrainingSection fields exist with the EXACT plan defaults,
       and the existing 6 are untouched;
  (1) a NON-fresh ShrimpTrainState round-trips through to_dict/from_dict ->
      field-equal;
  (2) ShrimpCheckpointSaver.save embeds train_state into meta; torch.load back
      shows meta["train_state"] present and round-tripping to the same state;
  (3) an OLD-format meta WITHOUT train_state -> from_dict(meta.get(...)) yields a
      FRESH state, no crash (old checkpoints resume cleanly);
  (4) the loader RESUME branch restores the persisted governor, while the
      initialize_from branch does NOT (fresh governor on a warm start).

Run (CPU, no GPU, no live run):
  PYTHONPATH=packages/shrimp/python python -m pytest tests/katago_buffer/test_p1_train_state.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

# (0) import-gate: these must all import cleanly (CPU, no CUDA touched).
import shrimp.config as hxconfig
import shrimp.checkpoints as hxckpt
import shrimp.trainer as hxtrainer
from shrimp.checkpoints import (
    ShrimpCheckpointLoader,
    ShrimpCheckpointSaver,
    save_checkpoint,
)
from shrimp.config import TrainingSection
from shrimp.model import ShrimpNet
from shrimp.train_state import TRAIN_STATE_VERSION, ShrimpTrainState


# --------------------------------------------------------------------------- #
# Lightweight framework stand-ins. We deliberately do NOT spin up the real
# hexo_train pipeline: the saver/loader only touch a handful of attributes, and
# mirroring exactly those keeps the test CPU-only and contract-focused.
# components.model is the ModelComponents-shaped object (so components.model.model
# is the net, components.model.trainer is the ShrimpTrainer) — matching
# components.py:151-168.
# --------------------------------------------------------------------------- #
def _make_components(trainer):
    model = ShrimpNet()  # no-arg ctor; stays on CPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if trainer is not None:
        # The saver reads components.model.model / .optimizer off the trainer's
        # sibling slots; keep the trainer pointed at the same net for realism.
        trainer.model = model
        trainer.optimizer = optimizer
    model_components = SimpleNamespace(model=model, optimizer=optimizer, trainer=trainer)
    return SimpleNamespace(model=model_components)


def _make_ctx(checkpoint_dir: Path, *, resume_from):
    checkpoint = SimpleNamespace(resume_from=resume_from)
    run = SimpleNamespace(name="shrimp_p1_test")
    config = SimpleNamespace(run=run, checkpoint=checkpoint)
    return SimpleNamespace(checkpoint_dir=checkpoint_dir, config=config)


def _non_fresh_state() -> ShrimpTrainState:
    """A state with every field perturbed away from the fresh defaults so the
    round-trip actually exercises each field (a fresh state would pass trivially)."""
    return ShrimpTrainState(
        total_num_data_rows=123_456,
        global_step_samples=7_777,
        window_start_data_row_idx=4_096,
        train_bucket_level=98_765.5,
        train_bucket_level_at_row=120_000,
        train_steps_since_last_reload=42,
        data_files_used={"epoch_000003/game_0000017.npz", "epoch_000004/game_0000099.npz"},
    )


def _assert_states_equal(a: ShrimpTrainState, b: ShrimpTrainState, ctx: str) -> None:
    assert a.total_num_data_rows == b.total_num_data_rows, ctx
    assert a.global_step_samples == b.global_step_samples, ctx
    assert a.window_start_data_row_idx == b.window_start_data_row_idx, ctx
    assert abs(a.train_bucket_level - b.train_bucket_level) < 1e-9, ctx
    assert a.train_bucket_level_at_row == b.train_bucket_level_at_row, ctx
    assert a.train_steps_since_last_reload == b.train_steps_since_last_reload, ctx
    assert a.data_files_used == b.data_files_used, ctx
    assert int(a.version) == int(b.version) == TRAIN_STATE_VERSION, ctx


# --------------------------------------------------------------------------- #
# (0) imports already happened at module top; assert they resolved to modules.
# --------------------------------------------------------------------------- #
def test_imports():
    assert hxconfig is not None and hxckpt is not None and hxtrainer is not None
    # trainer.__init__ must now carry a fresh train_state (instruction b).
    t = hxtrainer.ShrimpTrainer(
        model=ShrimpNet(),
        config=hxconfig.ShrimpConfig(device="cpu"),
        optimizer=None,
    )
    assert isinstance(t.train_state, ShrimpTrainState)
    _assert_states_equal(t.train_state, ShrimpTrainState(), "fresh trainer.train_state")
    print("[0] imports OK; trainer.__init__ carries a fresh ShrimpTrainState")


# --------------------------------------------------------------------------- #
# (0b) the 11 new config fields exist with EXACT plan defaults; the 6 originals
#      are untouched.
# --------------------------------------------------------------------------- #
def test_config_fields():
    ts = TrainingSection()
    # The original 6 — must be preserved.
    assert ts.batch_rows == 32
    assert ts.learning_rate == 1e-3
    assert ts.weight_decay == 1e-4
    assert ts.grad_clip == 1.0
    assert ts.warmup_steps == 0
    assert ts.shuffle_keep_target_rows == 300_000
    # The 11 new fields — exact defaults + types from PLAN §7 / instruction (a).
    expected = {
        "shuffle_min_rows": (20_000, int),
        "shuffle_taper_window_exponent": (0.65, float),
        "shuffle_expand_window_per_row": (0.4, float),
        "shuffle_taper_window_scale": (20_000.0, float),
        "validation_fraction": (0.0, float),
        "train_samples_per_epoch": (100_000, int),
        "max_train_bucket_per_new_data": (8.0, float),
        "max_train_bucket_size": (500_000.0, float),
        "no_repeat_files": (False, bool),
        "expand_backend": ("serial", str),
        "expand_workers": (0, int),
    }
    fields = TrainingSection.__dataclass_fields__
    for name, (default, typ) in expected.items():
        assert name in fields, f"missing TrainingSection field {name!r}"
        got = getattr(ts, name)
        assert got == default, f"{name}: default {got!r} != expected {default!r}"
        # bool must be checked before int (bool is a subclass of int).
        if typ is bool:
            assert isinstance(got, bool), f"{name}: {type(got)} not bool"
        elif typ is int:
            assert isinstance(got, int) and not isinstance(got, bool), f"{name}: {type(got)} not int"
        elif typ is float:
            assert isinstance(got, float), f"{name}: {type(got)} not float"
        else:
            assert isinstance(got, typ), f"{name}: {type(got)} not {typ}"
    # The new fields must flow through the real toml merge path too (no parse
    # change needed — _merge tolerates them). A partial override keeps the rest
    # at their defaults.
    merged = hxconfig._merge(
        TrainingSection,
        {"shuffle_min_rows": 99, "expand_backend": "rust", "no_repeat_files": True},
    )
    assert merged.shuffle_min_rows == 99
    assert merged.expand_backend == "rust"
    assert merged.no_repeat_files is True
    assert merged.max_train_bucket_size == 500_000.0  # untouched -> default
    print("[0b] TrainingSection: 6 originals preserved + 11 new fields exact; _merge accepts them")


# --------------------------------------------------------------------------- #
# (1) non-fresh round-trip equality.
# --------------------------------------------------------------------------- #
def test_round_trip_non_fresh():
    src = _non_fresh_state()
    rebuilt = ShrimpTrainState.from_dict(src.to_dict())
    _assert_states_equal(src, rebuilt, "non-fresh round-trip")
    # And the serialized dict carries the version + sorted file list.
    d = src.to_dict()
    assert d["version"] == TRAIN_STATE_VERSION
    assert d["data_files_used"] == sorted(src.data_files_used)
    print("[1] non-fresh ShrimpTrainState round-trips field-equal through to_dict/from_dict")


# --------------------------------------------------------------------------- #
# (2) saver embeds train_state; torch.load shows it in meta and it round-trips.
# --------------------------------------------------------------------------- #
def test_saver_embeds_train_state():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer = hxtrainer.ShrimpTrainer(
            model=ShrimpNet(),
            config=hxconfig.ShrimpConfig(device="cpu"),
            optimizer=None,
        )
        trainer.train_state = _non_fresh_state()
        components = _make_components(trainer)
        ctx = _make_ctx(ckpt_dir, resume_from=None)

        saver = ShrimpCheckpointSaver()
        path = saver.save(name="epoch_000007", ctx=ctx, components=components)
        assert Path(path).exists()

        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload["meta"]
        assert meta.get("run") == "shrimp_p1_test"
        assert meta.get("epoch") == 7
        assert "train_state" in meta, "saver did not embed train_state into meta"
        rebuilt = ShrimpTrainState.from_dict(meta["train_state"])
        _assert_states_equal(trainer.train_state, rebuilt, "saver->torch.load round-trip")
        print("[2] saver embeds train_state in meta; torch.load round-trips it to the same state")


def test_saver_no_train_state_guard():
    """A trainer without a train_state (or no trainer) must NOT crash the save and
    must NOT write a train_state key (instruction c guard)."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # trainer present but train_state is None.
        trainer = SimpleNamespace(train_state=None)
        components = _make_components(trainer)
        ctx = _make_ctx(ckpt_dir, resume_from=None)
        path = ShrimpCheckpointSaver().save(name="epoch_000001", ctx=ctx, components=components)
        meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
        assert "train_state" not in meta, "guard failed: train_state written for None state"

        # trainer entirely absent (None) -> still no crash, no key.
        components2 = _make_components(None)
        path2 = ShrimpCheckpointSaver().save(name="epoch_000002", ctx=ctx, components=components2)
        meta2 = torch.load(path2, map_location="cpu", weights_only=False)["meta"]
        assert "train_state" not in meta2, "guard failed: train_state written for absent trainer"
        print("[2b] saver guard: missing/None train_state -> no crash, no train_state key")


# --------------------------------------------------------------------------- #
# (3) old-format meta WITHOUT train_state -> fresh state, no crash.
# --------------------------------------------------------------------------- #
def test_old_format_meta_fresh():
    old_meta = {"lineage": "shrimp", "epoch": 4, "run": "ancient_run"}  # no train_state
    state = ShrimpTrainState.from_dict(old_meta.get("train_state"))
    _assert_states_equal(state, ShrimpTrainState(), "old-format -> fresh")
    # also: None / non-mapping / version-mismatch all degrade to fresh.
    for raw in (None, [], "garbage", {"version": 9999, "train_bucket_level": 5.0}):
        _assert_states_equal(ShrimpTrainState.from_dict(raw), ShrimpTrainState(), f"tolerant from_dict {raw!r}")
    print("[3] old-format meta (no train_state) and bad inputs -> FRESH state, no crash")


# --------------------------------------------------------------------------- #
# (4) loader: RESUME restores the governor; initialize_from does NOT.
# --------------------------------------------------------------------------- #
def _write_pipeline_checkpoint(ckpt_dir: Path, *, train_state_dict, epoch: int) -> Path:
    """Write a {meta, model, optimizer}-shaped checkpoint (the pipeline shape the
    loader's `"meta" in payload` branch expects) carrying a given train_state."""
    model = ShrimpNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    extra = {"run": "src_run"}
    if train_state_dict is not None:
        extra["train_state"] = train_state_dict
    return save_checkpoint(
        ckpt_dir / f"epoch_{epoch:06d}.pt",
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        extra=extra,
    )


def test_loader_resume_restores_initialize_does_not():
    persisted = _non_fresh_state()
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = _write_pipeline_checkpoint(ckpt_dir, train_state_dict=persisted.to_dict(), epoch=7)

        # --- RESUME branch: resume_from set -> governor restored from meta. ---
        trainer = hxtrainer.ShrimpTrainer(
            model=ShrimpNet(), config=hxconfig.ShrimpConfig(device="cpu"), optimizer=None
        )
        # Start from a fresh governor so a successful restore is observable.
        _assert_states_equal(trainer.train_state, ShrimpTrainState(), "pre-resume fresh")
        components = _make_components(trainer)
        ctx = _make_ctx(ckpt_dir, resume_from=str(ckpt_path))
        out = ShrimpCheckpointLoader().load(str(ckpt_path), ctx=ctx, components=components)
        assert out["status"] == "loaded", out
        assert out["epoch"] == 7
        _assert_states_equal(
            components.model.trainer.train_state, persisted, "RESUME must restore the governor"
        )

        # --- initialize_from branch: resume_from None -> governor stays FRESH. ---
        trainer2 = hxtrainer.ShrimpTrainer(
            model=ShrimpNet(), config=hxconfig.ShrimpConfig(device="cpu"), optimizer=None
        )
        components2 = _make_components(trainer2)
        ctx2 = _make_ctx(ckpt_dir, resume_from=None)
        out2 = ShrimpCheckpointLoader().load(str(ckpt_path), ctx=ctx2, components=components2)
        assert out2["status"] == "initialized_from", out2
        _assert_states_equal(
            components2.model.trainer.train_state,
            ShrimpTrainState(),
            "initialize_from must NOT inherit the stale governor",
        )
        print("[4] loader: RESUME restores the persisted governor; initialize_from keeps a fresh one")


def test_loader_resume_old_format_no_crash():
    """Resume from a checkpoint LACKING train_state (old format) -> fresh governor,
    no crash (the load-bearing backward-compat gate, PLAN M1)."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = _write_pipeline_checkpoint(ckpt_dir, train_state_dict=None, epoch=3)
        trainer = hxtrainer.ShrimpTrainer(
            model=ShrimpNet(), config=hxconfig.ShrimpConfig(device="cpu"), optimizer=None
        )
        # Seed the in-memory governor non-fresh; an old-format resume must RESET it.
        trainer.train_state = _non_fresh_state()
        components = _make_components(trainer)
        ctx = _make_ctx(ckpt_dir, resume_from=str(ckpt_path))
        out = ShrimpCheckpointLoader().load(str(ckpt_path), ctx=ctx, components=components)
        assert out["status"] == "loaded", out
        _assert_states_equal(
            components.model.trainer.train_state,
            ShrimpTrainState(),
            "old-format resume must reset to a FRESH governor",
        )
        print("[4b] resume from a train_state-less checkpoint -> FRESH governor, no crash")


def main() -> None:
    # Belt-and-suspenders: this test must never touch CUDA.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    test_imports()
    test_config_fields()
    test_round_trip_non_fresh()
    test_saver_embeds_train_state()
    test_saver_no_train_state_guard()
    test_old_format_meta_fresh()
    test_loader_resume_restores_initialize_does_not()
    test_loader_resume_old_format_no_crash()
    print("\nALL PHASE-1 SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
