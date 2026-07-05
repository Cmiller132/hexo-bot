"""Packed columnar in-RAM replay window.

The on-disk ``shrimp_compact_v1`` shard (written by
``shards.write_compact_shard``) is a flat columnar layout: per-row scalar arrays
+ ``(n,H)`` blocks + CSR ``data``/``off`` group pairs. This module loads those
columns while keeping them packed; it does not materialize the per-row
:class:`~shrimp.samples.ShrimpSampleData` tuple representation produced by
``shards.read_compact_shard``.

:class:`PackedWindow` holds the compact column set concatenated across shards
(with CSR offsets rebased to one global index) plus per-row ``generation`` and
``row_shard_id`` tags. :class:`PackedRowView` returns zero-copy slices for one
row in the shape one :func:`~shrimp.samples.expand_sample` call consumes.

Notes:
- ``horizons`` on a :class:`PackedWindow` is the union across concatenated
  shards. The stored ``stvalue``/``stvalue_mask`` columns are preserved verbatim.
  :func:`concat_packed` requires identical horizons across parts; a mismatch
  means the ``stvalue`` columns are not comparable and raises.
- :func:`concat_packed` pre-sizes every output array from the per-shard counts,
  fills in place, rebases CSR offsets to ``int64``, and frees each part after its
  copy, so the transient peak is ~1x the final window plus one shard.
"""

from __future__ import annotations

import hashlib
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .shards import _ACCEPTED_SCHEMA_VERSIONS, SCHEMA_VERSION

if TYPE_CHECKING:
    from .buffer_manifest import ShardEntry

# --- column taxonomy (matches shards.write_compact_shard) --------------------

# Per-row scalar columns: indexed directly by the row index ``i``.
SCALAR_COLS: tuple[str, ...] = (
    "turn_index",
    "current_player",
    "phase",
    "value",
    "moves_left",
    "outcome_valid",
    "policy_valid",
    "gumbel_present",  # 1 => row carries a π' target; 0/absent => visit fallback
    "policy_surprise",
    "first_q",
    "first_r",
    "first_present",
)

# Per-row ``(n, H)`` block columns: indexed ``[i, :]``.
BLOCK_COLS: tuple[str, ...] = ("stvalue", "stvalue_mask")

# CSR groups. Each entry is ``(off_col, data_cols, qr_doubled)`` where:
#   - ``off_col``    is the ``int64[n+1]`` offsets array for the group;
#   - ``data_cols``  are the flat data arrays governed by that offsets array;
#   - ``qr_doubled`` is True for the packed-(q,r) int16 arrays, where ``off``
#     counts *pairs* and the flat slice for row ``i`` is
#     ``data[2*off[i] : 2*off[i+1]]`` (shards._unpack_qr semantics). When False
#     (pol/opp/hist_owner/hist_pidx) the slice is ``data[off[i] : off[i+1]]``.
#
# The ``hist`` group: ``hist_qr`` is qr-doubled while ``hist_owner`` and
# ``hist_pidx`` share the same ``hist_off`` but are not doubled. It appears as
# two pseudo-groups sharing one offsets array.
CSR_GROUPS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("hist_off", ("hist_qr",), True),
    ("hist_off", ("hist_owner", "hist_pidx"), False),
    ("pol_off", ("pol_act", "pol_w", "q_pol_q", "prior_logit"), False),
    # π' target on its OWN support (shard schema v3): a superset of pol_act on
    # reused roots. v2 shards stored it aligned to pol_act; the loader converts
    # to CSR at load so downstream sees one shape.
    ("gumbel_off", ("gumbel_act", "gumbel_w"), False),
    ("opp_off", ("opp_act", "opp_w"), False),
    ("own_hot_off", ("own_hot_qr",), True),
    ("opp_hot_off", ("opp_hot_qr",), True),
    ("own_win_off", ("own_win_qr",), True),
    ("opp_win_off", ("opp_win_qr",), True),
)

# Distinct offset arrays (each appears once even though hist_off backs two
# pseudo-groups). Order is fixed for deterministic concat.
OFF_COLS: tuple[str, ...] = (
    "hist_off",
    "pol_off",
    "gumbel_off",
    "opp_off",
    "own_hot_off",
    "opp_hot_off",
    "own_win_off",
    "opp_win_off",
)

# Map an offsets column to the data columns it governs and whether each is
# qr-doubled. Built once from CSR_GROUPS.
_OFF_TO_DATA: dict[str, list[tuple[str, bool]]] = {}
for _off, _datas, _doubled in CSR_GROUPS:
    bucket = _OFF_TO_DATA.setdefault(_off, [])
    for _d in _datas:
        bucket.append((_d, _doubled))


@dataclass
class PackedRowView:
    """Zero-copy slices for one row, in the shape one ``expand_sample`` consumes.

    Every array attribute is a view into the parent :class:`PackedWindow` columns
    (no copy). The qr arrays are flat ``int16`` pair-packed segments
    (``shards._unpack_qr`` reads ``(seg[2k], seg[2k+1])``); the owner/pidx and
    policy/value arrays are the plain per-row segments.
    """

    # scalars (python-native, boxed per row)
    turn_index: int
    current_player: int
    phase: int
    value: float
    moves_left: float
    outcome_valid: int  # 1 completed / 0 truncated (gates value/stvalue/cell_q)
    policy_valid: int  # 1 full / 0 fast (gates policy/opp/soft/cell_q)
    gumbel_present: int  # 1 => π' target present; 0 => visit fallback
    policy_surprise: float
    first_q: int
    first_r: int
    first_present: int
    # blocks
    stvalue: np.ndarray  # (H,) f32 view
    stvalue_mask: np.ndarray  # (H,) f32 view
    # history CSR (owner/pidx aligned with hist_qr pairs)
    hist_qr: np.ndarray  # (2L,) i16 view
    hist_owner: np.ndarray  # (L,) u8 view
    hist_pidx: np.ndarray  # (L,) u16 view
    # policy / opp-policy CSR
    pol_act: np.ndarray  # (P,) u32 view
    pol_w: np.ndarray  # (P,) f32 view
    q_pol_q: np.ndarray  # (P,) f32 view; one child Q per recorded action (== pol_act)
    gumbel_act: np.ndarray  # (G,) u32 view; π' target support (own CSR group)
    gumbel_w: np.ndarray  # (G,) f32 view; π' weight per gumbel_act entry
    prior_logit_arr: np.ndarray  # (P,) f32 view; raw root logit aligned to pol_act
    opp_act: np.ndarray  # (O,) u32 view
    opp_w: np.ndarray  # (O,) f32 view
    # standing-cell qr CSR (flat pair-packed i16 views)
    own_hot_qr: np.ndarray
    opp_hot_qr: np.ndarray
    own_win_qr: np.ndarray
    opp_win_qr: np.ndarray
    # tags
    horizons: tuple[int, ...]
    generation: int
    row_shard_id: int

    def records(self) -> tuple[tuple[int, int, int, int], ...]:
        """``(q, r, owner, placement_index)`` tuples — the ``records`` field."""
        qr = self.hist_qr
        owner = self.hist_owner
        pidx = self.hist_pidx
        return tuple(
            (int(qr[2 * k]), int(qr[2 * k + 1]), int(owner[k]), int(pidx[k]))
            for k in range(owner.shape[0])
        )

    @staticmethod
    def _qr_pairs(flat: np.ndarray) -> tuple[tuple[int, int], ...]:
        m = flat.shape[0] // 2
        return tuple((int(flat[2 * k]), int(flat[2 * k + 1])) for k in range(m))

    def own_hot(self) -> tuple[tuple[int, int], ...]:
        return self._qr_pairs(self.own_hot_qr)

    def opp_hot(self) -> tuple[tuple[int, int], ...]:
        return self._qr_pairs(self.opp_hot_qr)

    def own_win(self) -> tuple[tuple[int, int], ...]:
        return self._qr_pairs(self.own_win_qr)

    def opp_win(self) -> tuple[tuple[int, int], ...]:
        return self._qr_pairs(self.opp_win_qr)

    def policy(self) -> tuple[tuple[int, float], ...]:
        return tuple((int(self.pol_act[k]), float(self.pol_w[k])) for k in range(self.pol_act.shape[0]))

    def q_policy(self) -> tuple[tuple[int, float], ...]:
        return tuple((int(self.pol_act[k]), float(self.q_pol_q[k])) for k in range(self.pol_act.shape[0]))

    def gumbel_policy(self) -> tuple[tuple[int, float], ...]:
        """Improved-policy target π' on its OWN support (``gumbel_act``).

        Empty when the row carries no target (``gumbel_present == 0``). When
        present, the weights are the per-action π' mass; the support can exceed
        ``pol_act`` (inherited edges on reused roots)."""
        if int(self.gumbel_present) == 0:
            return ()
        return tuple(
            (int(self.gumbel_act[k]), float(self.gumbel_w[k]))
            for k in range(self.gumbel_act.shape[0])
        )

    def prior_logit(self) -> tuple[tuple[int, float], ...]:
        """Raw root logits aligned to ``pol_act``. Empty when ``gumbel_present == 0``."""
        if int(self.gumbel_present) == 0:
            return ()
        return tuple(
            (int(self.pol_act[k]), float(self.prior_logit_arr[k]))
            for k in range(self.pol_act.shape[0])
        )

    def opp_policy(self) -> tuple[tuple[int, float], ...]:
        return tuple((int(self.opp_act[k]), float(self.opp_w[k])) for k in range(self.opp_act.shape[0]))

    def first_stone(self) -> tuple[int, int] | None:
        return (int(self.first_q), int(self.first_r)) if int(self.first_present) == 1 else None

    def short_term_value(self) -> tuple[tuple[int, float], ...]:
        mask = self.stvalue_mask
        vals = self.stvalue
        return tuple(
            (int(self.horizons[c]), float(vals[c]))
            for c in range(len(self.horizons))
            if mask[c] > 0.0
        )


@dataclass
class PackedWindow:
    """Concatenated packed columns for a whole replay window.

    ``cols`` holds every ``shrimp_compact_v1`` column kept packed: the per-row
    scalar arrays, the ``(n,H)`` blocks, the flat CSR data arrays, and the
    ``int64[n+1]`` CSR offsets (one global offsets array per group, rebased by
    :func:`concat_packed`). ``generation`` and ``row_shard_id`` are ``int32[n]``
    per-row tags.

    The window exposes neither ``window_size`` nor an ``index`` with
    ``sample_count``, so ``D6SymmetrySelector`` treats it as opaque
    (``_sample_count`` -> 0).
    """

    n: int
    cols: dict[str, np.ndarray]
    horizons: tuple[int, ...]
    generation: np.ndarray  # int32[n]
    row_shard_id: np.ndarray  # int32[n]

    @classmethod
    def empty(cls) -> "PackedWindow":
        """A zero-row window."""
        cols: dict[str, np.ndarray] = {}
        for name in SCALAR_COLS:
            cols[name] = np.empty(0, dtype=_SCALAR_DTYPES[name])
        for name in BLOCK_COLS:
            cols[name] = np.empty((0, 0), dtype=np.float32)
        for off in OFF_COLS:
            cols[off] = np.zeros(1, dtype=np.int64)
        for _off, datas, _doubled in CSR_GROUPS:
            for d in datas:
                cols[d] = np.empty(0, dtype=_CSR_DTYPES[d])
        return cls(
            n=0,
            cols=cols,
            horizons=(),
            generation=np.empty(0, dtype=np.int32),
            row_shard_id=np.empty(0, dtype=np.int32),
        )

    def row_view(self, i: int) -> PackedRowView:
        """Zero-copy slices for row ``i``, in the shape one ``expand_sample`` consumes."""
        if i < 0 or i >= self.n:
            raise IndexError(f"row {i} out of range for PackedWindow(n={self.n})")
        c = self.cols
        h0, h1 = int(c["hist_off"][i]), int(c["hist_off"][i + 1])
        p0, p1 = int(c["pol_off"][i]), int(c["pol_off"][i + 1])
        g0, g1 = int(c["gumbel_off"][i]), int(c["gumbel_off"][i + 1])
        o0, o1 = int(c["opp_off"][i]), int(c["opp_off"][i + 1])

        def qr_slice(key: str) -> np.ndarray:
            off = c[key + "_off"]
            a, b = int(off[i]), int(off[i + 1])
            return c[key + "_qr"][2 * a : 2 * b]

        return PackedRowView(
            turn_index=int(c["turn_index"][i]),
            current_player=int(c["current_player"][i]),
            phase=int(c["phase"][i]),
            value=float(c["value"][i]),
            moves_left=float(c["moves_left"][i]),
            outcome_valid=int(c["outcome_valid"][i]),
            policy_valid=int(c["policy_valid"][i]),
            gumbel_present=int(c["gumbel_present"][i]),
            policy_surprise=float(c["policy_surprise"][i]),
            first_q=int(c["first_q"][i]),
            first_r=int(c["first_r"][i]),
            first_present=int(c["first_present"][i]),
            stvalue=c["stvalue"][i],
            stvalue_mask=c["stvalue_mask"][i],
            hist_qr=c["hist_qr"][2 * h0 : 2 * h1],
            hist_owner=c["hist_owner"][h0:h1],
            hist_pidx=c["hist_pidx"][h0:h1],
            pol_act=c["pol_act"][p0:p1],
            pol_w=c["pol_w"][p0:p1],
            q_pol_q=c["q_pol_q"][p0:p1],
            gumbel_act=c["gumbel_act"][g0:g1],
            gumbel_w=c["gumbel_w"][g0:g1],
            prior_logit_arr=c["prior_logit"][p0:p1],
            opp_act=c["opp_act"][o0:o1],
            opp_w=c["opp_w"][o0:o1],
            own_hot_qr=qr_slice("own_hot"),
            opp_hot_qr=qr_slice("opp_hot"),
            own_win_qr=qr_slice("own_win"),
            opp_win_qr=qr_slice("opp_win"),
            horizons=self.horizons,
            generation=int(self.generation[i]),
            row_shard_id=int(self.row_shard_id[i]),
        )


# Expected dtypes per column, used by empty() and as a load-time dtype guard.
# These match the writer (shards.write_compact_shard).
_SCALAR_DTYPES: dict[str, np.dtype] = {
    "turn_index": np.dtype(np.int32),
    "current_player": np.dtype(np.uint8),
    "phase": np.dtype(np.uint8),
    "value": np.dtype(np.float32),
    "moves_left": np.dtype(np.float32),
    "outcome_valid": np.dtype(np.uint8),
    "policy_valid": np.dtype(np.uint8),
    "gumbel_present": np.dtype(np.uint8),
    "policy_surprise": np.dtype(np.float32),
    "first_q": np.dtype(np.int16),
    "first_r": np.dtype(np.int16),
    "first_present": np.dtype(np.uint8),
}
_CSR_DTYPES: dict[str, np.dtype] = {
    "hist_qr": np.dtype(np.int16),
    "hist_owner": np.dtype(np.uint8),
    "hist_pidx": np.dtype(np.uint16),
    "pol_act": np.dtype(np.uint32),
    "pol_w": np.dtype(np.float32),
    "q_pol_q": np.dtype(np.float32),
    "gumbel_act": np.dtype(np.uint32),
    "gumbel_w": np.dtype(np.float32),
    "prior_logit": np.dtype(np.float32),
    "opp_act": np.dtype(np.uint32),
    "opp_w": np.dtype(np.float32),
    "own_hot_qr": np.dtype(np.int16),
    "opp_hot_qr": np.dtype(np.int16),
    "own_win_qr": np.dtype(np.int16),
    "opp_win_qr": np.dtype(np.int16),
}


def _shard_generation(path: Path, num_rows: int) -> int:
    """Producing epoch for the shard.

    Resolution order: key-derived epoch (``game_key // 1_000_000`` parsed from the
    filename stem) takes precedence; the sidecar ``.json`` ``epoch`` is used only
    to cross-check (warns on mismatch) and as a fallback when the key parse fails;
    a parent directory named ``epoch_NNNNNN`` is the final fallback. Returns 0 if
    none resolve. Does not use mtime.
    """
    stem = path.stem  # e.g. "game_1000000"
    key_epoch: int | None = None
    if "_" in stem:
        try:
            game_key = int(stem.split("_", 1)[1])
            key_epoch = game_key // 1_000_000
        except (ValueError, IndexError):
            key_epoch = None

    sidecar = path.with_suffix(".json")
    side_epoch: int | None = None
    if sidecar.exists():
        try:
            import json

            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if "epoch" in meta:
                side_epoch = int(meta["epoch"])
        except (ValueError, OSError):
            side_epoch = None

    if key_epoch is not None:
        if side_epoch is not None and side_epoch != key_epoch:
            import warnings

            warnings.warn(
                f"shard {path.name}: sidecar epoch {side_epoch} != key-derived "
                f"epoch {key_epoch}; trusting key-derived",
                RuntimeWarning,
                stacklevel=2,
            )
        return key_epoch
    if side_epoch is not None:
        return side_epoch
    # Last resort: parent dir name "epoch_NNNNNN".
    parent = path.parent.name
    if parent.startswith("epoch_"):
        try:
            return int(parent.split("_", 1)[1])
        except (ValueError, IndexError):
            pass
    return 0


def load_packed_shard(path: Path) -> PackedWindow:
    """Load one ``shrimp_compact_v1`` shard with columns kept packed.

    Loads via ``np.load`` and validates ``schema_version`` against
    ``SCHEMA_VERSION`` (raises on mismatch). Columns are not exploded into
    :class:`ShrimpSampleData`. This loader handles only ``shrimp_compact_v1``
    shards; restnet shards are read by ``shards.read_legacy_restnet_shard``.
    """
    path = Path(path)
    with np.load(path) as data:
        files = set(data.files)
        if "schema_version" not in files:
            raise ValueError(f"{path.name}: not a shrimp_compact_v1 shard (no schema_version)")
        version = int(data["schema_version"])
        if version not in _ACCEPTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported shrimp shard schema {version} "
                f"(loader accepts {_ACCEPTED_SCHEMA_VERSIONS})"
            )
        n = int(data["num_rows"])
        horizons = tuple(int(h) for h in data["horizons"])

        cols: dict[str, np.ndarray] = {}
        # Materialize each column into a real array while the npz is open
        # (np.load arrays are lazy and close on exit).
        for name in SCALAR_COLS:
            if name in ("outcome_valid", "policy_valid") and name not in files:
                # Shards lacking the outcome_valid / policy_valid columns default
                # to all-1 (every row treated as completed / full).
                cols[name] = np.ones(n, dtype=_SCALAR_DTYPES[name])
                continue
            if name == "gumbel_present" and name not in files:
                # Shards lacking the gumbel_present column default to all-0
                # (no π' target; the loss falls back to the visit target).
                cols[name] = np.zeros(n, dtype=_SCALAR_DTYPES[name])
                continue
            cols[name] = np.ascontiguousarray(data[name])
        for name in BLOCK_COLS:
            cols[name] = np.ascontiguousarray(data[name])
        for off in OFF_COLS:
            if off == "gumbel_off" and off not in files:
                # v1/v2 shard: the π' CSR group is synthesized below.
                continue
            cols[off] = np.ascontiguousarray(data[off]).astype(np.int64, copy=False)
        for _off, datas, _doubled in CSR_GROUPS:
            for d in datas:
                if d in ("gumbel_act", "gumbel_w") and d not in files:
                    # Synthesized below for v1/v2 shards.
                    continue
                if d == "prior_logit" and d not in files:
                    # Shards lacking the per-action logit column zero-fill
                    # aligned to pol_act (the pol_off group's length) to keep
                    # downstream slicing valid; a gumbel_present=0 row ignores it.
                    pol_total = int(data["pol_act"].shape[0])
                    cols[d] = np.zeros(pol_total, dtype=_CSR_DTYPES[d])
                    continue
                cols[d] = np.ascontiguousarray(data[d])
        if "gumbel_off" not in cols:
            # Legacy π' storage (v2: gumbel_pol_w aligned to pol_act; v1: none).
            # Convert to the v3 CSR shape at load so downstream sees one format.
            # v2's alignment already truncated π' to the visit support, so this
            # conversion preserves exactly what the shard stored (nonzero mass).
            if "gumbel_pol_w" in files and int(data["pol_act"].shape[0]) > 0:
                aligned = np.ascontiguousarray(data["gumbel_pol_w"]).astype(
                    np.float32, copy=False
                )
                pol_act_arr = cols["pol_act"]
                pol_off_arr = cols["pol_off"]
                present = cols["gumbel_present"]
                keep = np.zeros(aligned.shape[0], dtype=bool)
                lens = np.zeros(n, dtype=np.int64)
                for i in range(n):
                    if int(present[i]) == 0:
                        continue
                    a, b = int(pol_off_arr[i]), int(pol_off_arr[i + 1])
                    seg = aligned[a:b] > 0.0
                    keep[a:b] = seg
                    lens[i] = int(seg.sum())
                cols["gumbel_act"] = np.ascontiguousarray(pol_act_arr[keep])
                cols["gumbel_w"] = np.ascontiguousarray(aligned[keep])
                offsets = np.zeros(n + 1, dtype=np.int64)
                np.cumsum(lens, out=offsets[1:])
                cols["gumbel_off"] = offsets
            else:
                cols["gumbel_act"] = np.empty(0, dtype=_CSR_DTYPES["gumbel_act"])
                cols["gumbel_w"] = np.empty(0, dtype=_CSR_DTYPES["gumbel_w"])
                cols["gumbel_off"] = np.zeros(n + 1, dtype=np.int64)

    generation = np.full(n, _shard_generation(path, n), dtype=np.int32)
    row_shard_id = np.zeros(n, dtype=np.int32)
    return PackedWindow(
        n=n,
        cols=cols,
        horizons=horizons,
        generation=generation,
        row_shard_id=row_shard_id,
    )


def concat_packed(parts: Sequence[PackedWindow]) -> PackedWindow:
    """Concatenate packed shards into one window.

    Pre-sizes every output array from the per-shard counts, fills in place,
    rebases each CSR offsets array to one global ``int64`` index, and frees each
    part after its copy, so the transient stays ~1x the final window plus one
    shard.

    Empty parts (n==0) are skipped. ``horizons`` must be identical across
    non-empty parts; a mismatch means the ``stvalue`` block columns are not
    concatenatable and raises.
    """
    parts = [p for p in parts]
    nonempty = [p for p in parts if p.n > 0]
    if not nonempty:
        return PackedWindow.empty()

    # Validate consistent horizons / block widths across parts.
    horizons = nonempty[0].horizons
    h_width = nonempty[0].cols["stvalue"].shape[1]
    for p in nonempty[1:]:
        if p.horizons != horizons:
            raise ValueError(
                f"concat_packed: horizon mismatch {p.horizons} != {horizons}; "
                "stvalue blocks are not concatenatable"
            )
        if p.cols["stvalue"].shape[1] != h_width:
            raise ValueError("concat_packed: stvalue block width mismatch")

    total_n = int(sum(p.n for p in nonempty))

    # --- pre-size outputs from counts ----------------------------------------
    out: dict[str, np.ndarray] = {}
    for name in SCALAR_COLS:
        out[name] = np.empty(total_n, dtype=_SCALAR_DTYPES[name])
    for name in BLOCK_COLS:
        out[name] = np.empty((total_n, h_width), dtype=np.float32)
    # CSR data totals (sum of each data array length across parts).
    data_totals: dict[str, int] = {}
    for _off, datas, _doubled in CSR_GROUPS:
        for d in datas:
            data_totals[d] = int(sum(p.cols[d].shape[0] for p in nonempty))
    for d, tot in data_totals.items():
        out[d] = np.empty(tot, dtype=_CSR_DTYPES[d])
    for off in OFF_COLS:
        out[off] = np.empty(total_n + 1, dtype=np.int64)
        out[off][0] = 0
    out_gen = np.empty(total_n, dtype=np.int32)
    out_sid = np.empty(total_n, dtype=np.int32)

    # --- fill in place, rebasing CSR offsets ---------------------------------
    row_cursor = 0
    data_cursor: dict[str, int] = {d: 0 for d in data_totals}
    off_base: dict[str, int] = {off: 0 for off in OFF_COLS}

    for shard_idx, part in enumerate(nonempty):
        pc = part.cols
        pn = part.n
        r0, r1 = row_cursor, row_cursor + pn

        for name in SCALAR_COLS:
            out[name][r0:r1] = pc[name]
        for name in BLOCK_COLS:
            out[name][r0:r1, :] = pc[name]
        out_gen[r0:r1] = part.generation
        # row_shard_id is set to this part's index within the window (part-level
        # row_shard_id from load is 0).
        out_sid[r0:r1] = np.int32(shard_idx)

        for off in OFF_COLS:
            src_off = pc[off]  # int64[pn+1], starts at 0
            base = off_base[off]
            # Global offsets for this part's rows: src_off[1:] + base.
            out[off][r0 + 1 : r1 + 1] = src_off[1:] + base
            # Advance base by this part's total count for the group.
            off_base[off] = base + int(src_off[pn])
            # Copy each data array governed by this offsets group.
            for d, doubled in _OFF_TO_DATA[off]:
                src = pc[d]
                m = src.shape[0]
                dc = data_cursor[d]
                out[d][dc : dc + m] = src
                data_cursor[d] = dc + m

        row_cursor = r1
        # Free the part's columns after its copy.
        part.cols.clear()
        part.generation = np.empty(0, dtype=np.int32)
        part.row_shard_id = np.empty(0, dtype=np.int32)
        part.n = 0

    # Sanity: every CSR data array filled exactly, and each offsets array ends at
    # the accumulated element/pair count for its group.
    for d, tot in data_totals.items():
        assert data_cursor[d] == tot, f"CSR data {d} fill mismatch {data_cursor[d]} != {tot}"
    for off in OFF_COLS:
        assert int(out[off][total_n]) == off_base[off]

    return PackedWindow(
        n=total_n,
        cols=out,
        horizons=horizons,
        generation=out_gen,
        row_shard_id=out_sid,
    )


# =============================================================================
# Window sizing, md5 train/val split, and overshoot-skip file selection.
# ``ShardEntry`` carries ``.rows`` / ``.generation`` / ``.game_key`` /
# ``.rel_path``. The window is built in RAM as a ``PackedWindow``.
# =============================================================================


def compute_katago_window_rows(
    usable_rows: int,
    *,
    min_rows: int,
    expand_window_per_row: float,
    taper_window_exponent: float,
    taper_window_scale: float | None,
) -> int:
    """Power-law taper window size.

    Result is truncated with ``int()`` (not rounded). As ``usable_rows ->
    min_rows`` the window collapses to ``min_rows``; ``taper_window_exponent < 1``
    gives a sublinear taper. The caller clamps ``max(window, min_rows)``.
    """
    offset = float(taper_window_scale if taper_window_scale is not None else min_rows)
    power_law_x = float(usable_rows) - float(min_rows) + offset
    unscaled = power_law_x ** taper_window_exponent - offset ** taper_window_exponent
    scaled = unscaled / (taper_window_exponent * (offset ** (taper_window_exponent - 1.0)))
    return int(scaled * expand_window_per_row + float(min_rows))


def keep_prob(used_rows: int, keep_target_rows: int) -> float:
    """Uniform-subsample probability toward ``keep_target_rows``.

    Returns ``min(keep_target_rows, used_rows) / used_rows``: ``1.0`` when the
    window is already at or below the target, else the down-sample ratio. Returns
    ``1.0`` when ``used_rows <= 0`` to avoid a zero divide.
    """
    if used_rows <= 0:
        return 1.0
    return min(float(keep_target_rows), float(used_rows)) / float(used_rows)


def select_recent_window(
    entries: Sequence["ShardEntry"], desired_rows: int
) -> tuple[list["ShardEntry"], int]:
    """Newest-to-oldest whole-shard accumulation until ``used_rows >= desired_rows``.

    ``entries`` arrive (generation, game_key)-ascending from the manifest, so
    ``reversed`` walks newest-first. Whole-shard granularity overshoots
    ``desired_rows`` by less than one shard. The selected list is re-sorted
    ascending on return.
    """
    selected: list["ShardEntry"] = []
    used_rows = 0
    for info in reversed(entries):
        selected.append(info)
        used_rows += int(info.rows)
        if used_rows >= desired_rows:
            break
    selected.reverse()
    return selected, used_rows


def _md5_path_fraction(value: str) -> float:
    """Stable [0, 1) fraction from the md5 of a path.

    Returns the first 13 hex digits of ``md5(value)`` as an int divided by
    ``2**52``. A pure function of the path string (seed-independent), so the
    train/val partition and any md5 sub-range filter are stable across epochs and
    runs.
    """
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:13]
    return int("0x" + digest, 16) / float(2**52)


def _split_by_md5(
    selected: Sequence["ShardEntry"],
    *,
    validation_fraction: float,
) -> tuple[list["ShardEntry"], list["ShardEntry"]]:
    """Per-file md5 train/val split, keyed on ``str(entry.rel_path)``.

    ``validation_fraction <= 0`` returns all entries as train with empty val.
    Otherwise a file goes to val iff its md5 fraction is
    ``>= 1 - validation_fraction`` (a fixed, path-stable cut).
    """
    if validation_fraction <= 0.0:
        return list(selected), []
    train_upper = 1.0 - float(validation_fraction)
    train_infos: list["ShardEntry"] = []
    val_infos: list["ShardEntry"] = []
    for info in selected:
        fraction = _md5_path_fraction(str(info.rel_path))
        if fraction < train_upper:
            train_infos.append(info)
        else:
            val_infos.append(info)
    return train_infos, val_infos


def _select_files_for_rows(
    entries: Sequence["ShardEntry"],
    requested_rows: int,
    rng: np.random.Generator,
) -> tuple[list["ShardEntry"], int]:
    """Overshoot-skip single-pass file selection capped near ``requested_rows``.

    Row counts are read from ``ShardEntry.rows``. Shuffles the candidates with
    ``rng`` and greedily accumulates; a shard that would overshoot is
    probabilistically skipped (``skip_prob = overshoot / row_count``) and
    deferred. Deferred shards are added back if the total is still short. All
    draws come from ``rng``.
    """
    candidates: list[tuple["ShardEntry", int]] = [(info, int(info.rows)) for info in entries]
    rng.shuffle(candidates)
    selected: list["ShardEntry"] = []
    deferred: list[tuple["ShardEntry", int]] = []
    rows = 0
    for info, row_count in candidates:
        if rows > 0 and rows + row_count > requested_rows:
            overshoot = rows + row_count - requested_rows
            skip_prob = min(1.0, max(0.0, overshoot / max(1, row_count)))
            if rng.random() < skip_prob:
                deferred.append((info, row_count))
                continue
        selected.append(info)
        rows += row_count
        if rows >= requested_rows:
            return selected, rows
    for info, row_count in deferred:
        selected.append(info)
        rows += row_count
        if rows >= requested_rows:
            break
    return selected, rows


def build_window_split(
    selected: Sequence["ShardEntry"],
    *,
    keep_prob: float,
    rng: np.random.Generator,
    samples_dir: Path,
    diag: dict | None = None,
) -> PackedWindow:
    """Load the selected shards, per-row Bernoulli subsample, and concat into one
    packed in-RAM window.

    The window is kept packed in RAM; there is no ``data*.npz`` write. The permute
    and ``effective_rows`` truncation are done by the consumer.

    The per-row keep is an independent ``Bernoulli(keep_prob)`` drawn from a single
    shared ``rng`` consumed in ``(generation, game_key)`` shard order and within a
    shard in stored row order (``rng.random(shard.n) < keep_prob`` per shard).
    ``keep_prob >= 1.0`` keeps every row with no RNG draw.

    Survivors are concatenated with :func:`concat_packed`.

    Telemetry: when ``diag`` is a dict it is filled in place with load/skip
    accounting (does not alter what is loaded or concatenated):

    * ``shards_selected``   — survivor shard count (``len(ordered)`` minus skips);
    * ``shards_skipped``    — count of shards skipped as unreadable (torn npz);
    * ``skipped_paths``     — the skipped shard paths, capped at 20;
    * ``rows_loaded``       — total rows across survivors BEFORE keep_prob thinning;
    * ``rows_post_thin``    — the concatenated window's ``n`` (post-thin rows).
    """
    # Consume the keep mask in (generation, game_key) order so the shared rng
    # stream is reproducible regardless of the input ordering.
    ordered = sorted(selected, key=lambda e: (int(e.generation), int(e.game_key)))

    survivors: list[PackedWindow] = []
    skipped: list[str] = []
    rows_loaded = 0  # survivor rows before keep_prob thinning (telemetry only)
    for entry in ordered:
        shard_path = samples_dir / entry.rel_path
        # A power cut can leave a shard's npz torn while its commit-marker sidecar
        # survives (or a v-mismatch/short read), so load_packed_shard raises. Such
        # a shard stays in the recent window for dozens of epochs, so an unguarded
        # load crash-loops training EVERY epoch. Skip the bad shard loudly instead
        # (the operator quarantines the file manually; we do not delete/move it).
        try:
            shard = load_packed_shard(shard_path)
        except (zipfile.BadZipFile, ValueError, KeyError, OSError, EOFError) as exc:
            skipped.append(str(shard_path))
            warnings.warn(
                f"build_window_split: skipping unreadable shard {shard_path} "
                f"({type(exc).__name__}: {exc}); window will have fewer rows "
                "(operator: quarantine this file)",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        rows_loaded += int(shard.n)
        if keep_prob >= 1.0:
            survivors.append(shard)
            continue
        # Independent per-row Bernoulli(keep_prob), one vectorized draw per shard
        # in stored row order.
        mask = rng.random(shard.n) < keep_prob
        survivors.append(_subset_packed(shard, mask))

    if skipped:
        # Surface the aggregate skip count in diagnostics (the return type is a
        # bare PackedWindow, so a summary warning is the visible signal). A skip
        # means the built window has fewer rows than the caller's accounting
        # expects; the trainer permutes over actual window.n rows, so this is
        # tolerated (fewer rows, no out-of-range indexing).
        warnings.warn(
            f"build_window_split: {len(skipped)}/{len(ordered)} shards skipped "
            f"as unreadable (torn npz?); {sorted(skipped)}",
            RuntimeWarning,
            stacklevel=2,
        )

    window = concat_packed(survivors)

    if diag is not None:
        # Telemetry out-param: load/skip accounting for the select diagnostic.
        # skipped_paths capped at 20 to bound the diag json size on a bad-disk
        # epoch; shards_skipped carries the true (uncapped) count.
        diag["shards_selected"] = len(ordered) - len(skipped)
        diag["shards_skipped"] = len(skipped)
        diag["skipped_paths"] = sorted(skipped)[:20]
        diag["rows_loaded"] = int(rows_loaded)
        diag["rows_post_thin"] = int(window.n)

    return window


def _subset_packed(window: PackedWindow, mask: np.ndarray) -> PackedWindow:
    """Return a new :class:`PackedWindow` keeping only the rows where ``mask`` is
    True, rebuilding every CSR group's offsets/data for the survivor rows.

    Used by :func:`build_window_split` for the keep_prob subsample. The kept count
    is ``int(mask.sum())``; an all-False mask yields a valid empty window
    (:func:`concat_packed` tolerates and skips it). Block/scalar columns slice
    directly; CSR columns are rebuilt by walking the kept rows and copying each
    row's flat segment (qr groups copy pair-doubled segments and rebuild the
    pair-counting offsets).
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(np.bool_)
    if mask.shape[0] != window.n:
        raise ValueError(f"_subset_packed: mask length {mask.shape[0]} != window.n {window.n}")
    keep_idx = np.nonzero(mask)[0]
    kept = int(keep_idx.shape[0])
    if kept == 0:
        return PackedWindow.empty()
    if kept == window.n:
        return window  # nothing dropped

    c = window.cols
    out: dict[str, np.ndarray] = {}
    # Scalars + blocks: fancy-index the kept rows.
    for name in SCALAR_COLS:
        out[name] = np.ascontiguousarray(c[name][keep_idx])
    for name in BLOCK_COLS:
        out[name] = np.ascontiguousarray(c[name][keep_idx, :])

    # CSR groups: rebuild offsets + data over the kept rows. Each distinct offsets
    # array is rebuilt once; its governed data arrays (qr-doubled or not) are
    # gathered alongside.
    for off in OFF_COLS:
        src_off = c[off]
        datas = _OFF_TO_DATA[off]
        # New offsets: cumulative kept segment lengths (in *group units*, i.e.
        # pairs for qr-doubled groups since src_off already counts pairs).
        seg_lens = (src_off[keep_idx + 1] - src_off[keep_idx]).astype(np.int64)
        new_off = np.empty(kept + 1, dtype=np.int64)
        new_off[0] = 0
        np.cumsum(seg_lens, out=new_off[1:])
        out[off] = new_off
        for d, doubled in datas:
            src = c[d]
            tot = int(new_off[kept])
            elems = 2 * tot if doubled else tot
            dst = np.empty(elems, dtype=_CSR_DTYPES[d])
            wcur = 0
            for row in keep_idx:
                a = int(src_off[row])
                b = int(src_off[row + 1])
                if doubled:
                    seg = src[2 * a : 2 * b]
                else:
                    seg = src[a:b]
                m = seg.shape[0]
                dst[wcur : wcur + m] = seg
                wcur += m
            assert wcur == elems, f"_subset_packed: {d} fill {wcur} != {elems}"
            out[d] = dst

    return PackedWindow(
        n=kept,
        cols=out,
        horizons=window.horizons,
        generation=np.ascontiguousarray(window.generation[keep_idx]),
        row_shard_id=np.ascontiguousarray(window.row_shard_id[keep_idx]),
    )
