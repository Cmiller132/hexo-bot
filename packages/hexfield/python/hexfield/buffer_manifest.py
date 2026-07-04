"""Persisted shard manifest for the replay buffer.

Shards are ordered by ``(generation, game_key)`` and the manifest is persisted to
``<samples_dir>/.buffer_manifest.json``. It is updated incrementally rather than
re-globbing and ``stat()``-sorting each epoch. It does not perform the per-epoch
window decode.

Two row totals are tracked:

* ``total_rows`` — sum of ``rows`` over present entries. Used for window
  selection.
* ``cumulative_rows_ever`` — monotone counter, ``max(prev, total_rows)``, not
  decremented when shards are pruned.

Generation tagging: the key-derived epoch (``game_key // 1_000_000``) is used;
the sidecar ``epoch`` is compared against it and a warning is emitted on
mismatch; the directory name is the fallback when the sidecar is absent.

Behavior:

* Shards lacking a JSON sidecar are skipped and not opened.
* Entries whose ``.npz`` file no longer exists are pruned.
* A missing, unparseable, or version-mismatched manifest triggers a full
  rebuild from the tree instead of raising.
* The manifest is written atomically (tmp file + ``os.replace``).
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# Persisted manifest schema version. A loaded manifest whose version differs is
# discarded and rebuilt from the tree.
MANIFEST_VERSION = 1

# Manifest file name under ``samples_dir``. The leading dot keeps it out of the
# ``game_*.npz`` glob.
MANIFEST_NAME = ".buffer_manifest.json"

# Game keys per epoch. ``game_key = epoch * 1_000_000 + within-epoch index``.
_KEYS_PER_EPOCH = 1_000_000


@dataclass(frozen=True)
class ShardEntry:
    """One self-play shard. ``rel_path`` is the POSIX path relative to
    ``samples_dir`` and is the key the train/val md5 split hashes.
    ``generation`` is the producing epoch; ``game_key`` is the within-run game id
    (``epoch * 1_000_000 + i``)."""

    rel_path: str
    rows: int
    generation: int
    game_key: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel_path": self.rel_path,
            "rows": int(self.rows),
            "generation": int(self.generation),
            "game_key": int(self.game_key),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ShardEntry":
        # Raises KeyError/TypeError/ValueError on a malformed entry.
        return cls(
            rel_path=str(raw["rel_path"]),
            rows=int(raw["rows"]),
            generation=int(raw["generation"]),
            game_key=int(raw["game_key"]),
        )


@dataclass
class BufferManifest:
    version: int = MANIFEST_VERSION
    # Sorted by (generation, game_key).
    entries: list[ShardEntry] = field(default_factory=list)
    # Sum of rows over present entries.
    total_rows: int = 0
    # Monotone; not decremented.
    cumulative_rows_ever: int = 0

    # -- derived helpers -------------------------------------------------

    def resort(self) -> None:
        """Sort ``entries`` by ``(generation, game_key)``."""
        self.entries.sort(key=lambda e: (e.generation, e.game_key))

    def recompute_totals(self) -> None:
        """Set ``total_rows`` to the sum over entries and set
        ``cumulative_rows_ever`` to ``max(cumulative_rows_ever, total_rows)``."""
        self.total_rows = sum(int(e.rows) for e in self.entries)
        self.cumulative_rows_ever = max(int(self.cumulative_rows_ever), int(self.total_rows))

    # -- (de)serialization ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "total_rows": int(self.total_rows),
            "cumulative_rows_ever": int(self.cumulative_rows_ever),
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BufferManifest":
        """Load from a dict. Raises ``ValueError`` on version mismatch or
        malformed structure."""
        if not isinstance(raw, Mapping):
            raise ValueError("manifest is not a JSON object")
        version = int(raw.get("version", 0))
        if version != MANIFEST_VERSION:
            raise ValueError(f"manifest version {version} != {MANIFEST_VERSION}")
        raw_entries = raw.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError("manifest 'entries' is not a list")
        entries = [ShardEntry.from_dict(item) for item in raw_entries]
        man = cls(
            version=version,
            entries=entries,
            total_rows=int(raw.get("total_rows", 0)),
            cumulative_rows_ever=int(raw.get("cumulative_rows_ever", 0)),
        )
        return man


# ----------------------------------------------------------------------
# Shard metadata extraction
# ----------------------------------------------------------------------


def _game_key_and_generation(npz_path: Path, sidecar: Mapping[str, Any] | None) -> tuple[int, int]:
    """Return ``(game_key, generation)`` for a shard.

    ``game_key`` is the integer after the first ``_`` in the stem
    (``game_<key>``) and ``generation = game_key // 1_000_000``. When the
    key-derived generation is available, the sidecar ``epoch`` is compared
    against it and a warning is emitted on mismatch; the key-derived value is
    returned.
    """
    stem = npz_path.stem  # e.g. "game_1000000"
    game_key: int
    try:
        game_key = int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        # Non-conforming name: set key to -1 (sorts before real keys) and fall
        # back to the sidecar/dir for the generation.
        game_key = -1

    key_generation = game_key // _KEYS_PER_EPOCH if game_key >= 0 else None

    side_epoch: int | None = None
    if sidecar is not None and "epoch" in sidecar:
        try:
            side_epoch = int(sidecar["epoch"])
        except (TypeError, ValueError):
            side_epoch = None

    if key_generation is not None:
        if side_epoch is not None and side_epoch != key_generation:
            warnings.warn(
                f"shard {npz_path.name}: sidecar epoch {side_epoch} != key-derived "
                f"generation {key_generation}; trusting key-derived",
                RuntimeWarning,
                stacklevel=2,
            )
        return game_key, key_generation

    # Key-derivation failed. Fall back to the sidecar epoch, then the parent dir
    # "epoch_NNNNNN", then 0.
    if side_epoch is not None:
        return game_key, side_epoch
    parent = npz_path.parent.name
    if parent.startswith("epoch_"):
        try:
            return game_key, int(parent.split("_", 1)[1])
        except (IndexError, ValueError):
            pass
    return game_key, 0


def _rows_from_sidecar(sidecar: Mapping[str, Any]) -> int | None:
    """Row count from a parsed sidecar. Checks ``rows``, then ``num_rows``, then
    ``effective_rows``. Returns ``None`` if none is present and parseable.
    """
    for key in ("rows", "num_rows", "effective_rows"):
        if key in sidecar:
            try:
                return int(sidecar[key])
            except (TypeError, ValueError):
                continue
    return None


def _rows_from_npz(npz_path: Path) -> int:
    """Row count read from the shard's ``num_rows`` array (falling back to
    ``effective_rows``). Returns 0 if neither array is present. Opens the
    ``.npz`` via ``np.load``."""
    import numpy as np  # local import: keep module import light

    with np.load(npz_path) as data:
        if "num_rows" in data.files:
            return int(data["num_rows"])
        if "effective_rows" in data.files:
            return int(data["effective_rows"])
    return 0


def _build_entry(npz_path: Path, samples_dir: Path) -> ShardEntry | None:
    """Build a :class:`ShardEntry` for one ``.npz``. Returns ``None`` if the
    shard has no ``.json`` sidecar or the sidecar is unreadable; in that case
    the ``.npz`` is not opened.
    """
    sidecar_path = npz_path.with_suffix(".json")
    if not sidecar_path.exists():
        return None

    sidecar: Mapping[str, Any] | None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, Mapping):
            sidecar = None
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    rows: int | None = _rows_from_sidecar(sidecar) if sidecar is not None else None
    if rows is None:
        # Sidecar has no usable row key -> read the count from the shard.
        try:
            rows = _rows_from_npz(npz_path)
        except (OSError, ValueError, KeyError):
            return None

    game_key, generation = _game_key_and_generation(npz_path, sidecar)
    rel_path = npz_path.relative_to(samples_dir).as_posix()
    return ShardEntry(
        rel_path=rel_path,
        rows=int(rows),
        generation=int(generation),
        game_key=int(game_key),
    )


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> BufferManifest | None:
    """Load and validate the persisted manifest. Returns ``None`` if the file is
    absent or on any parse/version/structure failure; does not raise for a bad
    file."""
    if not manifest_path.exists():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return BufferManifest.from_dict(raw)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        warnings.warn(
            f"{manifest_path.name}: unreadable/incompatible manifest; rebuilding from tree",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _atomic_write_manifest(manifest_path: Path, manifest: BufferManifest) -> None:
    """Write the manifest by writing a pid-named tmp file in the same directory
    then ``os.replace``-ing it onto ``manifest_path``."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.{os.getpid()}.tmp")
    payload = json.dumps(manifest.to_dict(), indent=2)
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, manifest_path)
    finally:
        # Remove the tmp file if the replace did not consume it.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def _iter_shard_npzs(samples_dir: Path) -> Iterable[Path]:
    """All ``epoch_*/game_*.npz`` paths under ``samples_dir``, sorted."""
    return sorted(samples_dir.glob("epoch_*/game_*.npz"))


def scan_or_update_manifest(samples_dir: str | os.PathLike[str]) -> BufferManifest:
    """Load, incrementally update, persist, and return the buffer manifest.

    Steps:

    1. Load ``.buffer_manifest.json`` if present and valid; otherwise start a
       fresh manifest (full rebuild on parse/version mismatch).
    2. Drop entries whose ``.npz`` file no longer exists.
    3. ``glob`` ``epoch_*/game_*.npz`` and add only shards not already present.
       Skip shards lacking a sidecar. Row count comes from the sidecar
       (``rows``/``num_rows``/``effective_rows``); if the sidecar has no row key,
       it is read from the shard's ``num_rows`` array.
    4. Re-sort by ``(generation, game_key)``.
    5. Set ``total_rows`` to the sum over present entries and set
       ``cumulative_rows_ever = max(prev, total_rows)``.
    6. Persist atomically (tmp + ``os.replace``).

    Does not move files.
    """
    samples_dir = Path(samples_dir)
    manifest_path = samples_dir / MANIFEST_NAME

    manifest = _load_manifest(manifest_path)
    if manifest is None:
        manifest = BufferManifest()

    if not samples_dir.exists():
        # Nothing to scan; clear entries and persist so cumulative_rows_ever is
        # carried forward.
        manifest.entries = []
        manifest.recompute_totals()
        _atomic_write_manifest(manifest_path, manifest)
        return manifest

    # (2) Keep entries whose .npz still exists; index by rel_path.
    present: dict[str, ShardEntry] = {}
    for entry in manifest.entries:
        npz_path = samples_dir / entry.rel_path
        if npz_path.exists():
            present[entry.rel_path] = entry

    # (3) Add shards not already present; sidecar-less/unreadable shards are
    # skipped (returned as None by _build_entry).
    for npz_path in _iter_shard_npzs(samples_dir):
        rel_path = npz_path.relative_to(samples_dir).as_posix()
        if rel_path in present:
            continue
        new_entry = _build_entry(npz_path, samples_dir)
        if new_entry is not None:
            present[rel_path] = new_entry

    manifest.entries = list(present.values())

    # (4) ordering, (5) totals.
    manifest.resort()
    manifest.recompute_totals()
    manifest.version = MANIFEST_VERSION

    # (6) persist.
    _atomic_write_manifest(manifest_path, manifest)
    return manifest
