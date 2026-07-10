"""Blunder-seed mining for KataGo-style off-policy self-play coverage.

A fraction of self-play games can start from a stored mid-game position where
the net was recently "surprised" (high ``policy_surprise``), instead of an
empty board. This module mines those seed positions from the run's own recent
sample shards and hands the driver a deterministic, cheap pool of seed prefixes.

Route (b) — reconstruct the move prefix from a row's OWN history features:
each ``hexfield_compact_v1`` shard stores, per row, the placement history as
``hist_qr`` (q,r) + ``hist_owner`` + ``hist_pidx`` (placement_index) with CSR
``hist_off`` offsets, alongside the per-row ``policy_surprise`` scalar and
``turn_index`` (== number of placements before the decision). The placements
are stored in placement order (``hist_pidx`` == 1..N contiguous, verified
against real main_7 shards), so a row's own records ARE the full ordered move
prefix up to that ply. This makes the ``.hxr`` cross-reference unnecessary:
a single row yields ``(move_prefix, seed_ply, source metadata)`` directly.

The miner is a pure function of (recent shard contents, config). The seeding
DECISION and seed PICK are made in the driver from a deterministic per-game RNG
stream, so the pool itself carries no randomness — it is a sorted, stable list.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BlunderSeed:
    """One mined seed position.

    ``move_prefix`` is the ordered list of ``(q, r)`` placements to replay from
    an empty board to reach the seed position; ``seed_ply`` == ``len(prefix)``
    == the row's ``turn_index``. ``surprise`` is the row's ``policy_surprise``.
    ``source_epoch`` / ``source_game_key`` / ``source_row`` identify the origin
    shard row for telemetry and reproducibility.
    """

    move_prefix: tuple[tuple[int, int], ...]
    seed_ply: int
    surprise: float
    source_epoch: int
    source_game_key: int
    source_row: int

    @property
    def key(self) -> tuple:
        """Stable sort/identity key (epoch, game, row)."""
        return (self.source_epoch, self.source_game_key, self.source_row)


def _epoch_of(shard_dir: Path) -> int:
    """Parse the epoch index from an ``epoch_NNNNNN`` directory name; -1 if the
    name is not in that form."""
    name = shard_dir.name
    if name.startswith("epoch_"):
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1
    return -1


def _game_key_of(npz_path: Path) -> int:
    """Parse the game key from a ``game_<key>.npz`` filename; -1 if malformed."""
    try:
        return int(npz_path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _mine_shard(
    npz_path: Path,
    epoch: int,
    *,
    max_ply: int,
) -> tuple[list[BlunderSeed], list[float]]:
    """Extract candidate seeds and the full-row surprise values from one shard.

    Returns ``(candidates, surprises)`` where ``candidates`` are the accepted
    rows with ``1 <= turn_index <= max_ply`` (every recorded row is a
    full-search turn; fast rows are never written to shards), and ``surprises``
    is the list of surprise values over those same accepted rows (used by the
    caller to compute the epoch-wide quantile BEFORE thresholding). Corrupt
    rows (history length != turn_index, or truncated history arrays) are
    rejected before entering either list, so they cannot skew the quantile
    pool. A malformed / unreadable shard is skipped with a warning and
    contributes nothing.

    The reconstruction reads the row's placement history directly from the
    columnar arrays (no ``read_compact_shard`` round-trip through the full
    sample dataclass — the miner only needs coordinates + order), and sorts by
    ``placement_index`` so the prefix is in true play order regardless of the
    stored record order.
    """

    game_key = _game_key_of(npz_path)
    try:
        with np.load(npz_path) as data:
            # Only the columns the miner needs; a shard missing any of these is
            # treated as malformed (skip + warn).
            required = (
                "num_rows", "turn_index", "policy_surprise",
                "hist_qr", "hist_pidx", "hist_off",
            )
            missing = [k for k in required if k not in data.files]
            if missing:
                warnings.warn(
                    f"blunder_seeds: shard {npz_path.name} missing columns "
                    f"{missing}; skipping",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return [], []
            n = int(data["num_rows"])
            turn_index = np.asarray(data["turn_index"])
            surprise = np.asarray(data["policy_surprise"], dtype=np.float64)
            hist_qr = np.asarray(data["hist_qr"])
            hist_pidx = np.asarray(data["hist_pidx"])
            hist_off = np.asarray(data["hist_off"])
    except (OSError, ValueError, EOFError, KeyError) as exc:
        warnings.warn(
            f"blunder_seeds: cannot read shard {npz_path.name} ({exc!r}); skipping",
            RuntimeWarning,
            stacklevel=2,
        )
        return [], []

    candidates: list[BlunderSeed] = []
    surprises: list[float] = []
    for i in range(n):
        # All recorded rows are full-search (fast rows are never written).
        ply = int(turn_index[i])
        if ply < 1 or ply > int(max_ply):
            continue
        s = float(surprise[i])
        if not np.isfinite(s):
            continue
        # Reconstruct the ordered move prefix from the row's placement history.
        h0, h1 = int(hist_off[i]), int(hist_off[i + 1])
        length = h1 - h0
        # turn_index must equal the number of placements for the seed to be
        # replayable to the recorded ply; a mismatch means a corrupt row.
        if length != ply:
            continue
        recs = []
        ok = True
        for k in range(length):
            base = h0 + k
            try:
                q = int(hist_qr[2 * base])
                r = int(hist_qr[2 * base + 1])
                pidx = int(hist_pidx[base])
            except IndexError:
                ok = False
                break
            recs.append((pidx, q, r))
        if not ok:
            continue
        # Order by placement_index -> true play order.
        recs.sort(key=lambda t: t[0])
        prefix = tuple((q, r) for _pidx, q, r in recs)
        # Row accepted: only now does it enter the quantile pool, so corrupt
        # rows rejected above never skew the surprise threshold.
        surprises.append(s)
        candidates.append(
            BlunderSeed(
                move_prefix=prefix,
                seed_ply=ply,
                surprise=s,
                source_epoch=epoch,
                source_game_key=game_key,
                source_row=i,
            )
        )
    return candidates, surprises


def mine_blunder_seeds(
    samples_dir: Path,
    *,
    current_epoch: int,
    recent_epochs: int,
    max_ply: int,
    surprise_quantile: float,
) -> list[BlunderSeed]:
    """Mine blunder-seed positions from the run's recent sample shards.

    Scans the ``recent_epochs`` epoch directories strictly BELOW
    ``current_epoch`` under ``samples_dir`` (``epoch_NNNNNN/game_*.npz``),
    collects every FULL row with ``1 <= turn_index <= max_ply`` and finite
    ``policy_surprise``, computes the ``surprise_quantile`` threshold over the
    pooled surprise values, and returns the rows at or above that threshold as
    ``BlunderSeed`` entries.

    The result is DETERMINISTIC given the on-disk shards and the arguments: seeds
    are sorted by ``(source_epoch, source_game_key, source_row)`` so the pool
    order does not depend on filesystem iteration order. Missing / malformed
    shards are skipped with a warning. Returns an empty list when no recent data
    exists or no row clears the threshold.
    """

    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        return []
    lo = max(0, int(current_epoch) - int(recent_epochs))
    hi = int(current_epoch)  # strictly below the current epoch
    epoch_dirs = []
    for d in samples_dir.iterdir():
        if not d.is_dir():
            continue
        ep = _epoch_of(d)
        if ep < 0:
            continue
        if lo <= ep < hi:
            epoch_dirs.append((ep, d))
    # Deterministic epoch order.
    epoch_dirs.sort(key=lambda t: t[0])

    all_candidates: list[BlunderSeed] = []
    all_surprises: list[float] = []
    for ep, d in epoch_dirs:
        for npz_path in sorted(d.glob("game_*.npz")):
            cands, surs = _mine_shard(npz_path, ep, max_ply=max_ply)
            all_candidates.extend(cands)
            all_surprises.extend(surs)

    if not all_candidates:
        return []

    q = float(np.clip(surprise_quantile, 0.0, 1.0))
    threshold = float(np.quantile(np.asarray(all_surprises, dtype=np.float64), q))
    seeds = [c for c in all_candidates if c.surprise >= threshold]
    # Stable, filesystem-order-independent pool.
    seeds.sort(key=lambda c: c.key)
    return seeds
