"""Power-cut durability of the shrimp training-IO paths.

The host takes real power cuts; one already tore files on this run. These tests
cover the three atomic/guarded fixes:

1. ``checkpoints.save_checkpoint`` writes via tmp + ``os.replace`` (never a
   partial target), and cleans a stale ``.pt.tmp`` left by a prior crash.
2. ``shards.write_compact_shard`` round-trips identically through the new
   tmp+fsync+replace npz/sidecar path.
3. ``window.build_window_split`` skips a torn npz (with an intact sidecar) with a
   loud warning and still loads the surviving shards.

Pure IO on tmp dirs — no GPU, no live run touched.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from shrimp import checkpoints
from shrimp.buffer_manifest import ShardEntry
from shrimp.samples import ShrimpSampleData
from shrimp.shards import read_compact_shard, write_compact_shard
from shrimp.window import build_window_split, load_packed_shard


# --- fixtures -----------------------------------------------------------------


def _sample(turn: int) -> ShrimpSampleData:
    """A small but non-trivial row exercising every CSR/scalar column."""
    return ShrimpSampleData(
        game_id="",  # read back as "" — not round-tripped
        turn_index=turn,
        current_player=turn % 2,
        phase="Opening",
        records=((0, 0, 0, 0), (1, -1, 1, 1), (2, 0, 0, 2)),
        first_stone=(1, -1),
        own_hot=((0, 0), (2, 0)),
        opp_hot=((1, -1),),
        own_win=((3, -1),),
        opp_win=(),
        policy=((5, 0.7), (6, 0.3)),
        opp_policy=((7, 1.0),),
        q_policy=((5, 0.1), (6, -0.2)),
        gumbel_policy=((5, 0.6), (6, 0.25), (9, 0.15)),  # superset of pol_act
        prior_logit=((5, 1.2), (6, -0.5)),
        value=0.5 if turn % 2 == 0 else -0.5,
        short_term_value=((2, 0.4), (6, 0.1)),
        moves_left=float(10 - turn),
        policy_surprise=0.05 * turn,
        metadata={"pcr_full": True},
    )


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)


# --- BUG 1: atomic checkpoint save --------------------------------------------


def test_checkpoint_save_is_atomic_no_partial_target(tmp_path, monkeypatch) -> None:
    """A crash mid-write (simulated by raising inside os.replace) must leave NO
    partial target file — the payload only ever lands via tmp -> replace."""
    model = _TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    target = tmp_path / "epoch_000007.pt"

    replace_calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def boom(src, dst):
        replace_calls.append((str(src), str(dst)))
        raise RuntimeError("simulated power cut at rename")

    monkeypatch.setattr(checkpoints.os, "replace", boom)
    with pytest.raises(RuntimeError, match="simulated power cut"):
        checkpoints.save_checkpoint(target, model=model, optimizer=opt, epoch=7)

    # The tmp file was the write target and the replace was attempted onto the
    # real path (tmp-then-replace ordering), and the real target never appeared.
    assert replace_calls == [(str(target) + ".tmp", str(target))]
    assert not target.exists(), "power cut left a partial target checkpoint"

    # A successful save produces the target and no leftover tmp.
    monkeypatch.setattr(checkpoints.os, "replace", real_replace)
    out = checkpoints.save_checkpoint(target, model=model, optimizer=opt, epoch=7)
    assert out == target and target.exists()
    assert not (tmp_path / "epoch_000007.pt.tmp").exists()

    # The saved payload loads back with the expected shape.
    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert payload["meta"]["epoch"] == 7
    assert set(payload["model"].keys()) == set(model.state_dict().keys())


def test_checkpoint_save_cleans_stale_tmp(tmp_path) -> None:
    """A stale ``.pt.tmp`` left by a previous crash is removed on entry, and the
    fresh save still succeeds atomically."""
    model = _TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    target = tmp_path / "epoch_000042.pt"
    stale = tmp_path / "epoch_000042.pt.tmp"
    stale.write_bytes(b"garbage from a previous crash")

    checkpoints.save_checkpoint(target, model=model, optimizer=opt, epoch=42)
    assert target.exists()
    assert not stale.exists(), "stale .pt.tmp was not cleaned up"


def test_checkpoint_tmp_name_off_supervisor_glob() -> None:
    """The tmp name must not match the supervisor's ``epoch_*.pt`` glob (which
    ends at ``.pt``) — else a torn tmp could be resumed from."""
    import fnmatch

    assert not fnmatch.fnmatch("epoch_000007.pt.tmp", "epoch_*.pt")
    assert fnmatch.fnmatch("epoch_000007.pt", "epoch_*.pt")


# --- BUG 2: atomic shard write round-trip -------------------------------------


def test_shard_write_roundtrips_through_atomic_path(tmp_path) -> None:
    """write_compact_shard (tmp+fsync+replace) -> read_compact_shard equality on a
    small multi-row fixture, and both files commit with no leftover tmp."""
    path = tmp_path / "epoch_000001" / "game_1000000.npz"
    samples = [_sample(0), _sample(1), _sample(2)]

    n = write_compact_shard(path, samples, sidecar={"epoch": 1})
    assert n == 3
    assert path.exists()
    assert path.with_suffix(".json").exists()
    # No leftover tmp files from either write.
    assert not (path.parent / (path.name + ".tmp")).exists()
    assert not list(path.parent.glob("*.tmp"))

    back = read_compact_shard(path)
    assert len(back) == len(samples)
    for orig, got in zip(samples, back):
        assert got.turn_index == orig.turn_index
        assert got.current_player == orig.current_player
        assert got.phase == orig.phase
        assert got.records == orig.records
        assert got.first_stone == orig.first_stone
        assert got.own_hot == orig.own_hot
        assert got.opp_hot == orig.opp_hot
        assert got.own_win == orig.own_win
        assert got.opp_win == orig.opp_win
        assert [a for a, _ in got.policy] == [a for a, _ in orig.policy]
        assert [w for _, w in got.policy] == pytest.approx([w for _, w in orig.policy])
        assert [a for a, _ in got.gumbel_policy] == [a for a, _ in orig.gumbel_policy]
        assert [w for _, w in got.gumbel_policy] == pytest.approx(
            [w for _, w in orig.gumbel_policy]
        )
        assert [a for a, _ in got.opp_policy] == [a for a, _ in orig.opp_policy]
        assert got.value == pytest.approx(orig.value)
        assert got.moves_left == pytest.approx(orig.moves_left)
        assert dict(got.short_term_value) == pytest.approx(dict(orig.short_term_value))

    # The sidecar is the commit marker and carries the passed-through key.
    import json

    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["rows"] == 3 and meta["epoch"] == 1 and meta["schema"] == "shrimp_compact_v1"


def test_shard_write_cleans_stale_tmp(tmp_path) -> None:
    """Stale npz/sidecar tmp files from a prior crash are removed on entry."""
    path = tmp_path / "epoch_000001" / "game_1000000.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / (path.name + ".tmp")).write_bytes(b"torn npz tmp")
    sidecar_tmp = path.with_suffix(".json")
    (sidecar_tmp.parent / (sidecar_tmp.name + ".tmp")).write_text("torn sidecar tmp")

    write_compact_shard(path, [_sample(0)], sidecar={"epoch": 1})
    assert not list(path.parent.glob("*.tmp")), "stale tmp files were not cleaned"
    assert read_compact_shard(path)[0].turn_index == 0


# --- BUG 3: build_window_split skips a torn npz -------------------------------


def _write_shard(samples_dir: Path, epoch: int, idx: int, samples) -> ShardEntry:
    game_key = epoch * 1_000_000 + idx
    rel = f"epoch_{epoch:06d}/game_{game_key}.npz"
    path = samples_dir / rel
    write_compact_shard(path, samples, sidecar={"epoch": epoch})
    return ShardEntry(rel_path=rel, rows=len(samples), generation=epoch, game_key=game_key)


def test_build_window_skips_torn_npz_and_loads_rest(tmp_path) -> None:
    """A torn npz (garbage bytes) with an intact sidecar is skipped with a
    RuntimeWarning; the good shards still load and the window row count reflects
    only the survivors."""
    samples_dir = tmp_path / "samples"
    good1 = _write_shard(samples_dir, 1, 0, [_sample(0), _sample(1)])
    bad = _write_shard(samples_dir, 1, 1, [_sample(0), _sample(1), _sample(2)])
    good2 = _write_shard(samples_dir, 1, 2, [_sample(0)])

    # Corrupt the middle shard's npz in place, leaving its sidecar intact.
    (samples_dir / bad.rel_path).write_bytes(b"not a valid npz -- torn by a power cut")
    # Sanity: the corrupt shard really does raise on load.
    with pytest.raises(Exception):
        load_packed_shard(samples_dir / bad.rel_path)

    rng = np.random.default_rng(0)
    with pytest.warns(RuntimeWarning, match="unreadable shard"):
        window = build_window_split(
            [good1, bad, good2], keep_prob=1.0, rng=rng, samples_dir=samples_dir
        )

    # 2 + 1 survivor rows; the torn shard's 3 rows are dropped.
    assert window.n == 3
