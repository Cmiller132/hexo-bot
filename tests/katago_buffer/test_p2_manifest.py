"""Phase 2 self-test — hexfield/buffer_manifest.py (PLAN §5.3, §5.4, §6).

CPU-only, no GPU, no live-run interaction. All inputs are genuine
hexfield_compact_v1 shards synthesized by ``_shard_gen`` (npz + sidecar), so no
private data is required; ``scan_or_update_manifest`` — which PERSISTS
``.buffer_manifest.json`` into its ``samples_dir`` — is only ever pointed at those
synthesized trees under ``_scratch/``.

Asserts:
  * entries sorted by (generation, game_key)
  * total_rows == sum(entry.rows)
  * cumulative_rows_ever monotone across two scans even after deleting a shard
  * a sidecar-less shard (half-written) is SKIPPED
  * a vanished entry is PRUNED on the next scan
  * generation/game_key derived from the filename; sidecar epoch mismatch WARNs
    but key-derived value wins
  * a garbled manifest self-heals (full rebuild)
  * the manifest write is atomic (real file appears; no .tmp left behind)
Prints PASS on success.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import numpy as np

from hexfield.buffer_manifest import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    BufferManifest,
    ShardEntry,
    scan_or_update_manifest,
)

# The conftest inserts this directory on sys.path; the generator writes genuine
# hexfield_compact_v1 shards (npz + sidecar) so these tests need no private data.
from _shard_gen import generate_samples_tree  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / "_scratch"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _fresh_scratch(name: str) -> Path:
    """A clean scratch subtree we are allowed to mutate."""
    root = SCRATCH / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_synth_shard(dst_dir: Path, game_key: int, rows: int, *, sidecar_epoch=None,
                       with_sidecar: bool = True) -> None:
    """Write a minimal but schema-valid synthetic compact shard (npz with
    ``num_rows`` + ``schema_version``) and optionally its sidecar. Lets us build
    trees with controlled (generation, game_key, rows) without the live data."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    npz_path = dst_dir / f"game_{game_key}.npz"
    np.savez_compressed(
        npz_path,
        schema_version=np.asarray(1, dtype=np.int32),
        num_rows=np.asarray(rows, dtype=np.int64),
        horizons=np.asarray([2, 6, 16], dtype=np.int32),
    )
    if with_sidecar:
        meta = {
            "lineage": "hexfield",
            "schema": "hexfield_compact_v1",
            "schema_version": 1,
            "rows": rows,
            "horizons": [2, 6, 16],
            "game_key": game_key,
        }
        if sidecar_epoch is not None:
            meta["epoch"] = sidecar_epoch
        else:
            meta["epoch"] = game_key // 1_000_000
        (dst_dir / f"game_{game_key}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _assert_sorted(man: BufferManifest) -> None:
    keys = [(e.generation, e.game_key) for e in man.entries]
    assert keys == sorted(keys), f"entries not sorted by (generation, game_key): {keys}"


def _assert_total(man: BufferManifest) -> None:
    assert man.total_rows == sum(e.rows for e in man.entries), (
        f"total_rows {man.total_rows} != sum {sum(e.rows for e in man.entries)}"
    )
    assert man.cumulative_rows_ever >= man.total_rows, (
        f"cumulative {man.cumulative_rows_ever} < live total {man.total_rows}"
    )


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------


def test_real_sidecar_schema() -> None:
    """Confirm a genuine hexfield sidecar (written by the real
    ``write_compact_shard`` path via the shard generator) uses the 'rows' key +
    'epoch'/'game_key' cross-check fields the manifest relies on, and the npz
    carries the 'num_rows' array."""
    root = _fresh_scratch("realsidecar")
    samples = root / "samples"
    generate_samples_tree(samples, epochs=1, games_per_epoch=2, base_seed=808)
    sidecars = sorted(samples.glob("epoch_*/game_*.json"))
    assert sidecars, "generator wrote no sidecars"
    meta = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert "rows" in meta, f"sidecar missing 'rows' key: {sorted(meta)}"
    assert "epoch" in meta, f"sidecar missing 'epoch' key: {sorted(meta)}"
    assert "game_key" in meta, f"sidecar missing 'game_key' key: {sorted(meta)}"
    npz = sidecars[0].with_suffix(".npz")
    with np.load(npz) as data:
        assert "num_rows" in data.files, "npz missing 'num_rows' array"
    print(f"  sidecar schema OK: rows={meta['rows']}, epoch={meta['epoch']}, "
          f"game_key={meta['game_key']}")


def test_happy_path_real_copy() -> None:
    """Scan two epochs of genuine synthesized shards; assert ordering + totals,
    and that the manifest file is written atomically (no leftover .tmp)."""
    root = _fresh_scratch("happy")
    samples = root / "samples"
    n = generate_samples_tree(samples, epochs=2, games_per_epoch=3, base_seed=909)
    assert n > 0, "generator wrote no shards"

    man = scan_or_update_manifest(samples)
    assert len(man.entries) == n, f"expected {n} entries, got {len(man.entries)}"
    _assert_sorted(man)
    _assert_total(man)
    assert man.version == MANIFEST_VERSION

    # Atomic write landed; no tmp residue.
    assert (samples / MANIFEST_NAME).exists(), "manifest not persisted"
    leftover = list(samples.glob(f"{MANIFEST_NAME}.*.tmp"))
    assert not leftover, f"leftover tmp files: {leftover}"

    # Persisted file round-trips to the same content.
    raw = json.loads((samples / MANIFEST_NAME).read_text(encoding="utf-8"))
    reloaded = BufferManifest.from_dict(raw)
    assert reloaded.total_rows == man.total_rows
    assert [e.to_dict() for e in reloaded.entries] == [e.to_dict() for e in man.entries]
    print(f"  happy path (synthesized): {n} shards, total_rows={man.total_rows}, "
          f"cumulative={man.cumulative_rows_ever}")


def test_generation_and_game_key_from_name() -> None:
    """generation = game_key // 1_000_000; game_key from the filename stem; the
    ordering interleaves correctly across two generations."""
    root = _fresh_scratch("genkey")
    samples = root / "samples"
    # Intentionally write out of order; rows differ per shard.
    _write_synth_shard(samples / "epoch_000002", 2000001, 10)
    _write_synth_shard(samples / "epoch_000001", 1000000, 20)
    _write_synth_shard(samples / "epoch_000002", 2000000, 30)
    _write_synth_shard(samples / "epoch_000001", 1000002, 40)

    man = scan_or_update_manifest(samples)
    _assert_sorted(man)
    got = [(e.generation, e.game_key, e.rows) for e in man.entries]
    expect = [(1, 1000000, 20), (1, 1000002, 40), (2, 2000000, 30), (2, 2000001, 10)]
    assert got == expect, f"gen/key derivation wrong: {got} != {expect}"
    # rel_path is POSIX, relative to samples_dir.
    assert man.entries[0].rel_path == "epoch_000001/game_1000000.npz", man.entries[0].rel_path
    _assert_total(man)
    print(f"  gen/key derivation OK: {got}")


def test_sidecar_epoch_mismatch_warns_keywins() -> None:
    """A sidecar 'epoch' that disagrees with the key-derived generation WARNs but
    the key-derived value is authoritative."""
    root = _fresh_scratch("mismatch")
    samples = root / "samples"
    # game_key 1000000 -> key-derived gen 1, but sidecar lies epoch=9.
    _write_synth_shard(samples / "epoch_000001", 1000000, 12, sidecar_epoch=9)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        man = scan_or_update_manifest(samples)
    assert man.entries[0].generation == 1, (
        f"key-derived generation must win, got {man.entries[0].generation}"
    )
    assert any("trusting key-derived" in str(w.message) for w in caught), (
        "expected a RuntimeWarning on epoch mismatch"
    )
    print("  sidecar-epoch mismatch: WARNed, key-derived generation (1) won")


def test_sidecarless_shard_skipped() -> None:
    """A shard whose ``.json`` sidecar is absent (half-written by the live writer)
    is SKIPPED — never opened, never an entry."""
    root = _fresh_scratch("nosidecar")
    samples = root / "samples"
    _write_synth_shard(samples / "epoch_000001", 1000000, 25)  # complete
    _write_synth_shard(samples / "epoch_000001", 1000001, 99, with_sidecar=False)  # torn

    man = scan_or_update_manifest(samples)
    keys = {e.game_key for e in man.entries}
    assert keys == {1000000}, f"sidecar-less shard not skipped: {keys}"
    assert man.total_rows == 25, f"total_rows should exclude skipped shard: {man.total_rows}"

    # Once the sidecar lands, the next scan picks it up incrementally.
    _write_synth_shard(samples / "epoch_000001", 1000001, 99, with_sidecar=True)
    man2 = scan_or_update_manifest(samples)
    keys2 = {e.game_key for e in man2.entries}
    assert keys2 == {1000000, 1000001}, f"sidecar landed but not picked up: {keys2}"
    assert man2.total_rows == 25 + 99
    print("  sidecar-less shard skipped, then picked up once sidecar landed")


def test_vanished_entry_pruned_and_cumulative_monotone() -> None:
    """Deleting a shard between two scans PRUNES its entry and DROPS total_rows,
    but cumulative_rows_ever must NOT decrease (PLAN §3.5 / M2)."""
    root = _fresh_scratch("vanish")
    samples = root / "samples"
    _write_synth_shard(samples / "epoch_000001", 1000000, 41)
    _write_synth_shard(samples / "epoch_000001", 1000001, 34)
    _write_synth_shard(samples / "epoch_000002", 2000000, 28)

    man1 = scan_or_update_manifest(samples)
    _assert_sorted(man1)
    _assert_total(man1)
    total1 = man1.total_rows
    cum1 = man1.cumulative_rows_ever
    assert total1 == 41 + 34 + 28 == 103
    assert cum1 == 103

    # Delete the largest shard (both npz + sidecar) and re-scan.
    (samples / "epoch_000001" / "game_1000000.npz").unlink()
    (samples / "epoch_000001" / "game_1000000.json").unlink()

    man2 = scan_or_update_manifest(samples)
    _assert_sorted(man2)
    keys = {e.game_key for e in man2.entries}
    assert 1000000 not in keys, f"vanished entry not pruned: {keys}"
    assert man2.total_rows == 34 + 28 == 62, f"live total wrong after delete: {man2.total_rows}"
    # The load-bearing monotonicity invariant.
    assert man2.cumulative_rows_ever >= cum1, (
        f"cumulative_rows_ever decreased: {man2.cumulative_rows_ever} < {cum1}"
    )
    assert man2.cumulative_rows_ever == 103, (
        f"cumulative should hold at the high-water mark 103, got {man2.cumulative_rows_ever}"
    )

    # Add a new shard; live total climbs again, cumulative advances past 103.
    _write_synth_shard(samples / "epoch_000002", 2000001, 50)
    man3 = scan_or_update_manifest(samples)
    assert man3.total_rows == 34 + 28 + 50 == 112
    assert man3.cumulative_rows_ever == max(103, 112) == 112
    print(f"  vanish: total {total1}->{man2.total_rows}->{man3.total_rows}; "
          f"cumulative monotone {cum1}->{man2.cumulative_rows_ever}->{man3.cumulative_rows_ever}")


def test_incremental_no_rescan_of_existing() -> None:
    """A second scan with NO tree change is a no-op on contents and preserves the
    monotone counter; adding one shard only adds one entry."""
    root = _fresh_scratch("incremental")
    samples = root / "samples"
    _write_synth_shard(samples / "epoch_000001", 1000000, 41)
    man1 = scan_or_update_manifest(samples)
    man2 = scan_or_update_manifest(samples)  # idempotent
    assert [e.to_dict() for e in man1.entries] == [e.to_dict() for e in man2.entries]
    assert man2.cumulative_rows_ever == man1.cumulative_rows_ever

    _write_synth_shard(samples / "epoch_000001", 1000001, 7)
    man3 = scan_or_update_manifest(samples)
    assert len(man3.entries) == 2
    assert man3.total_rows == 48
    print("  incremental scan idempotent; single add -> single new entry")


def test_self_heal_garbled_manifest() -> None:
    """A corrupt/garbled manifest file self-heals via a full rebuild rather than
    crashing; a wrong-version manifest is discarded too."""
    root = _fresh_scratch("selfheal")
    samples = root / "samples"
    _write_synth_shard(samples / "epoch_000001", 1000000, 41)
    samples.mkdir(parents=True, exist_ok=True)

    # Garbage JSON.
    (samples / MANIFEST_NAME).write_text("{not valid json :::", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        man = scan_or_update_manifest(samples)
    assert len(man.entries) == 1, f"garbled manifest should rebuild: {len(man.entries)}"
    assert man.total_rows == 41

    # Wrong version -> discard + rebuild.
    bad = {"version": MANIFEST_VERSION + 99, "entries": [], "total_rows": 0,
           "cumulative_rows_ever": 0}
    (samples / MANIFEST_NAME).write_text(json.dumps(bad), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        man2 = scan_or_update_manifest(samples)
    assert len(man2.entries) == 1, "version-mismatch manifest should rebuild"
    print("  self-heal: garbled + wrong-version manifests both rebuilt")


def test_npz_fallback_when_sidecar_lacks_rows() -> None:
    """If a sidecar exists but carries no row key, the row count falls back to the
    shard's ``num_rows`` array (not a skip)."""
    root = _fresh_scratch("npzfallback")
    samples = root / "samples"
    dst = samples / "epoch_000001"
    dst.mkdir(parents=True, exist_ok=True)
    npz = dst / "game_1000000.npz"
    np.savez_compressed(
        npz,
        schema_version=np.asarray(1, dtype=np.int32),
        num_rows=np.asarray(77, dtype=np.int64),
        horizons=np.asarray([2, 6, 16], dtype=np.int32),
    )
    # Sidecar present but deliberately WITHOUT any row key.
    (dst / "game_1000000.json").write_text(
        json.dumps({"lineage": "hexfield", "schema_version": 1, "epoch": 1, "game_key": 1000000}),
        encoding="utf-8",
    )
    man = scan_or_update_manifest(samples)
    assert len(man.entries) == 1
    assert man.entries[0].rows == 77, f"npz fallback row count wrong: {man.entries[0].rows}"
    print("  npz row-count fallback OK (sidecar present, no row key -> read num_rows=77)")


def main() -> None:
    tests = [
        ("real_sidecar_schema", test_real_sidecar_schema),
        ("happy_path_real_copy", test_happy_path_real_copy),
        ("generation_and_game_key_from_name", test_generation_and_game_key_from_name),
        ("sidecar_epoch_mismatch_warns_keywins", test_sidecar_epoch_mismatch_warns_keywins),
        ("sidecarless_shard_skipped", test_sidecarless_shard_skipped),
        ("vanished_entry_pruned_and_cumulative_monotone",
         test_vanished_entry_pruned_and_cumulative_monotone),
        ("incremental_no_rescan_of_existing", test_incremental_no_rescan_of_existing),
        ("self_heal_garbled_manifest", test_self_heal_garbled_manifest),
        ("npz_fallback_when_sidecar_lacks_rows", test_npz_fallback_when_sidecar_lacks_rows),
    ]
    for name, fn in tests:
        print(f"[ RUN ] {name}")
        fn()
        print(f"[ OK  ] {name}")
    print("\nPASS — all Phase 2 buffer_manifest tests green")


if __name__ == "__main__":
    main()
