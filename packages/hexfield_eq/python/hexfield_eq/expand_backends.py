"""Train-read row expansion backends.

Factors the per-row ``expand_sample`` work (D6 transform + support BFS + feature
build + legal-slot policy projection, all in ``samples.py``) out of
``trainer.train_passes`` and dispatches it across a configurable backend.

Backends (``backend=`` argument / ``config.training.expand_backend`` /
``HEXFIELD_EXPAND`` env):

* ``"serial"`` — one ``expand_sample`` call per window row on the main thread.
  The default.
* ``"pool"`` — a ``ProcessPoolExecutor(mp_context="spawn")``; the unit of work is
  one contiguous chunk of rows shipped to the top-level picklable
  :func:`expand_chunk_to_arrays`; ``workers+2`` inflight; ``wait(FIRST_COMPLETED)``;
  runs serially below :data:`_PARALLEL_MIN_ROWS`.
* ``"rust"`` — the rayon kernel (``replay_expand.rs::expand_shard_train``). One
  parallel call expands the whole window under the pre-drawn D6, returning
  zero-copy buffers reassembled here into the same ``(rows, valid)`` shape as
  serial/pool. ``workers``/``pool`` are ignored (rayon owns its thread pool).

Determinism: all randomness is pre-drawn on the main thread (the per-row ``d6``
vector) and passed positionally into the workers. Workers import only numpy and
the torch-free hexfield expansion chain (``samples`` → ``features`` /
``geometry`` / ``support``), never torch, and make no ``rng`` call. Results are
reassembled in the original row order (each chunk is written back at its start
offset), so the output is independent of worker count and future-completion
order.

Off-legal handling: an off-legal row (``expand_sample`` raises a message
containing "off the legal set") is flagged invalid in the returned ``valid``
mask rather than dropped in-worker. The caller (``train_passes``) does the
survivor filter, permute, and truncate on the main thread. A non-off-legal
``ValueError`` (e.g. zero policy mass) propagates unchanged.

This module is torch-free so a spawn worker re-importing it does not pull torch.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Sequence

import numpy as np

from .constants import LEGAL_RADIUS, NUM_FEATURES, RAYLEN_SLOTS, RAYTAP, TRUNK_LAYOUT
from .samples import STV_HORIZONS, ExpandedRow, HexfieldSampleData, expand_sample
from .shards import _PHASES
from .support import Support
from .window import PackedRowView, PackedWindow

# Columns the Rust kernel reads off the PackedWindow: the scalar, block,
# CSR-data, and CSR-offset arrays of hexfield_compact_v1, passed as raw
# native-endian bytes plus the explicit row count.
_RUST_SCALAR_COLS = (
    "current_player",
    "phase",
    "value",
    "moves_left",
    "policy_surprise",
    "outcome_valid",
    "gumbel_present",
    "first_q",
    "first_r",
    "first_present",
)
_RUST_BLOCK_COLS = ("stvalue", "stvalue_mask")
_RUST_CSR_DATA = (
    "hist_qr",
    "hist_owner",
    "hist_pidx",
    "pol_act",
    "pol_w",
    "q_pol_q",
    "gumbel_act",
    "gumbel_w",
    "prior_logit",
    "opp_act",
    "opp_w",
)
_RUST_OFF_COLS = (
    "hist_off",
    "pol_off",
    "gumbel_off",
    "opp_off",
)

# Substring present in the ValueError message ``expand_sample`` raises for an
# off-legal target.
_OFF_LEGAL_MARKER = "off the legal set"

# Row-count threshold below which the pool backend expands serially, skipping the
# spawn-pool startup and pickling overhead.
_PARALLEL_MIN_ROWS = 2048

# Rows per chunk shipped to a pool worker. Not a determinism knob: any positive
# value yields the same reassembled output (chunks are written back at their
# offset).
_CHUNK_ROWS = 256


# ---------------------------------------------------------------------------
# PackedRowView -> HexfieldSampleData shim. trainer.py re-exports it.
# ---------------------------------------------------------------------------
def _row_view_to_sample(view: PackedRowView) -> HexfieldSampleData:
    """Adapt a zero-copy :class:`~hexfield.window.PackedRowView` into the
    :class:`~hexfield.samples.HexfieldSampleData` that ``expand_sample`` consumes.

    The ``PackedRowView`` accessors return the shapes the dataclass expects
    (``records()`` → ``(q,r,owner,idx)`` tuples, ``policy()`` / ``opp_policy()``
    → ``(action_id, weight)`` tuples, ``first_stone()`` → ``(q,r)|None``,
    ``short_term_value()`` → ``(horizon, value)`` tuples). ``phase`` is stored as a u8 enum index in the
    packed column and is mapped to its string name through ``shards._PHASES`` (as
    ``read_compact_shard`` does), since ``HexfieldSampleData.phase`` and
    ``build_features`` take the string name. ``game_id`` is unused by expansion
    and left empty."""
    return HexfieldSampleData(
        game_id="",
        turn_index=view.turn_index,
        current_player=view.current_player,
        phase=_PHASES[view.phase],
        records=view.records(),
        first_stone=view.first_stone(),
        policy=view.policy(),
        opp_policy=view.opp_policy(),
        q_policy=view.q_policy(),
        gumbel_policy=view.gumbel_policy(),
        prior_logit=view.prior_logit(),
        value=view.value,
        short_term_value=view.short_term_value(),
        moves_left=view.moves_left,
        policy_surprise=view.policy_surprise,
        # Truncated-game flag (outcome_valid==0) carried as metadata; the serial
        # expand path uses it to mask the value/stvalue/cell_q heads. Only set
        # when truncated, so completed-game rows keep an empty metadata dict.
        metadata=(
            {"truncated": True} if int(view.outcome_valid) == 0 else {}
        ),
    )


def _expand_one(
    sample: HexfieldSampleData,
    sym: int,
    horizons: tuple[int, ...],
    tolerate_off_legal: bool,
) -> tuple[ExpandedRow | None, bool]:
    """Expand one row under its pre-drawn symmetry, returning ``(row, valid)``.

    An off-legal target raises ``ValueError``; when ``tolerate_off_legal`` is set
    and the message contains :data:`_OFF_LEGAL_MARKER`, the row is returned as
    ``(None, False)`` instead of raising. Any other ``ValueError`` (e.g. zero
    policy mass) propagates unchanged.
    """
    try:
        return expand_sample(sample, symmetry=int(sym), horizons=horizons), True
    except ValueError as exc:
        if tolerate_off_legal and _OFF_LEGAL_MARKER in str(exc):
            return None, False
        raise


# ---------------------------------------------------------------------------
# Top-level picklable worker unit: one contiguous chunk.
# ---------------------------------------------------------------------------
def expand_chunk_to_arrays(
    samples: list[HexfieldSampleData],
    d6: np.ndarray,
    horizons: tuple[int, ...],
    tolerate_off_legal: bool,
) -> tuple[list[ExpandedRow | None], list[bool]]:
    """Expand a chunk of pre-extracted rows to ``ExpandedRow`` objects.

    The pool's unit of work: top-level and picklable so it runs in a spawn
    worker. Uses numpy and the torch-free hexfield expansion chain only, and
    makes no rng call (every symmetry arrives pre-drawn in ``d6``). Returns the
    per-row expanded rows (``None`` for an off-legal-skipped row) plus the
    aligned ``valid`` mask. The result is a list of ``ExpandedRow`` because the
    per-row support graph is ragged and cannot be stacked into fixed arrays.

    ``samples`` and ``d6`` are positionally aligned 1:1. The caller ships
    contiguous chunks and stitches the results back at their start offset.
    """
    horizons = tuple(int(h) for h in horizons)
    rows: list[ExpandedRow | None] = []
    valid: list[bool] = []
    for sample, sym in zip(samples, d6):
        row, ok = _expand_one(sample, int(sym), horizons, tolerate_off_legal)
        rows.append(row)
        valid.append(ok)
    return rows, valid


def resolve_expand_workers(configured: int | None = None) -> int:
    """Resolve the pool worker count.

    Precedence: ``HEXFIELD_EXPAND_WORKERS`` env > the ``configured`` value
    (``config.training.expand_workers``, ``0`` ⇒ auto) > the auto default
    ``min(8, max(1, cpu_count // 4))``. Always ``>= 1``.
    """
    env = os.environ.get("HEXFIELD_EXPAND_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if configured is not None and int(configured) > 0:
        return max(1, int(configured))
    return min(8, max(1, (os.cpu_count() or 4) // 4))


def _chunk_bounds(n: int, chunk_rows: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` chunk bounds covering ``range(n)``."""
    step = max(1, int(chunk_rows))
    return [(s, min(s + step, n)) for s in range(0, n, step)]


# ---------------------------------------------------------------------------
# Rust backend: the rayon kernel + zero-copy reassembly.
# ---------------------------------------------------------------------------
def _resolve_support_radius() -> int:
    """The support radius (``HEXFIELD_EQ_SUPPORT_RADIUS`` env), passed to the Rust
    kernel. Matches ``support.py``'s import-time read: an integer value is used as
    given; a missing or non-integer value falls back to ``LEGAL_RADIUS`` (8).
    Clamped to [1, HALO_DIST], the same clamp as support.py and the Rust
    ``support_radius()`` (spec D-S26), so the three readers cannot diverge.
    """
    raw = os.environ.get("HEXFIELD_EQ_SUPPORT_RADIUS")
    if raw is None:
        return LEGAL_RADIUS
    try:
        r = int(raw)
    except ValueError:
        return LEGAL_RADIUS
    return min(max(r, 1), LEGAL_RADIUS + 1)


def _window_columns_as_bytes(window: PackedWindow) -> dict[str, bytes]:
    """Pack the PackedWindow columns the Rust kernel needs into a ``{name: bytes}``
    dict. Each array is made C-contiguous in its writer dtype, then ``.tobytes()``
    (one bulk copy per column). Offsets stay ``int64``; the kernel reinterprets
    the bytes.
    """
    c = window.cols
    out: dict[str, bytes] = {}
    for name in _RUST_SCALAR_COLS + _RUST_BLOCK_COLS + _RUST_CSR_DATA + _RUST_OFF_COLS:
        arr = np.ascontiguousarray(c[name])
        out[name] = arr.tobytes()
    return out


def _reassemble_rust_rows(
    result: dict, horizons_len: int
) -> tuple[list[ExpandedRow | None], np.ndarray]:
    """Turn the kernel's zero-copy buffers back into the ``(rows, valid)`` shape the
    serial/pool backends return.

    Each per-row segment is sliced out of the flat ``coords``/``dist``/``nbr``/
    ``feats``/``policy``/``opp_policy`` buffers via the returned CSR offsets
    (``node_off`` over support nodes, ``pol_off`` over the legal prefix). An
    invalid (off-legal-flagged) row has a zero-length segment and maps to
    ``None``.

    The reassembled ``Support`` carries an empty ``index`` dict. Downstream
    consumers (``collate_rows``, which reads ``coords``/``nbr``/``legal_count``/
    ``num_nodes``) do not read ``index``; the serial path uses ``index`` only
    during expansion, which is already complete here.
    """
    r = int(result["num_rows"])
    valid_bytes = bytes(result["valid"])
    valid = np.frombuffer(valid_bytes, dtype=np.uint8, count=r).astype(bool)

    legal_count = np.frombuffer(bytes(result["legal_count"]), dtype=np.int32, count=r)
    stone_count = np.frombuffer(bytes(result["stone_count"]), dtype=np.int32, count=r)
    halo_count = np.frombuffer(bytes(result["halo_count"]), dtype=np.int32, count=r)
    node_off = np.frombuffer(bytes(result["node_off"]), dtype=np.int64, count=r + 1)
    pol_off = np.frombuffer(bytes(result["pol_off"]), dtype=np.int64, count=r + 1)

    total_nodes = int(node_off[r])
    total_legal = int(pol_off[r])
    coords = np.frombuffer(bytes(result["coords"]), dtype=np.int32, count=2 * total_nodes).reshape(-1, 2)
    dist = np.frombuffer(bytes(result["dist"]), dtype=np.int32, count=total_nodes)
    nbr = np.frombuffer(bytes(result["nbr"]), dtype=np.int32, count=6 * total_nodes).reshape(-1, 6)
    # The kernel's feature width must match this build's plane map: a stale .so
    # under HEXFIELD_EQ_FEATURE_VERSION=2 (or a version desync between the env
    # the .so latched and this import) would otherwise mis-slice every plane.
    if int(result["num_features"]) != NUM_FEATURES:
        raise ValueError(
            f"expand kernel emitted {int(result['num_features'])} feature planes "
            f"but this build expects NUM_FEATURES={NUM_FEATURES} — rebuild "
            "hexfield_eq._rust (HEXFIELD_EQ_FEATURE_VERSION desync)"
        )
    feats = np.frombuffer(bytes(result["feats"]), dtype=np.float32, count=NUM_FEATURES * total_nodes).reshape(-1, NUM_FEATURES)
    # Side-relative ray lengths follow the SAME node_off slices as feats. When
    # the kernel omits the buffer (an older .so), rows carry all-zero raylen —
    # tolerable ONLY for C/A layouts (the model ignores it); under an 'L'
    # layout an all-zero raylen is a SILENT semantics flip (every ray dead), so
    # it raises instead (spec D-S31).
    if "raylen" in result:
        raylen = np.frombuffer(
            bytes(result["raylen"]), dtype=np.uint8, count=RAYLEN_SLOTS * total_nodes
        ).reshape(-1, RAYLEN_SLOTS)
    else:
        if "L" in TRUNK_LAYOUT or RAYTAP != "0":
            raise ValueError(
                "expand kernel emitted no raylen buffer but the net consumes "
                f"it (trunk layout {TRUNK_LAYOUT!r} / HEXFIELD_EQ_RAYTAP="
                f"{RAYTAP!r}) — rebuild hexfield_eq._rust (a fabricated "
                "all-zero raylen would kill every ray)"
            )
        raylen = np.zeros((total_nodes, RAYLEN_SLOTS), dtype=np.uint8)
    policy = np.frombuffer(bytes(result["policy"]), dtype=np.float32, count=total_legal)
    opp_policy = np.frombuffer(bytes(result["opp_policy"]), dtype=np.float32, count=total_legal)
    # Per-cell Q target + presence mask follow the SAME pol_off slices as policy.
    cell_q = np.frombuffer(bytes(result["cell_q"]), dtype=np.float32, count=total_legal)
    cell_q_mask = np.frombuffer(bytes(result["cell_q_mask"]), dtype=np.float32, count=total_legal)
    # Dense Gumbel policy target + dense raw logits follow the same pol_off slices
    # as policy; gumbel_policy_valid is per-row. When the kernel omits these keys,
    # gumbel_policy/prior_logit are all-zero and gumbel_policy_valid is 0.0.
    if "gumbel_policy" in result:
        gumbel_policy = np.frombuffer(bytes(result["gumbel_policy"]), dtype=np.float32, count=total_legal)
        prior_logit = np.frombuffer(bytes(result["prior_logit"]), dtype=np.float32, count=total_legal)
        gumbel_policy_valid = np.frombuffer(bytes(result["gumbel_policy_valid"]), dtype=np.float32, count=r)
    else:
        gumbel_policy = np.zeros(total_legal, dtype=np.float32)
        prior_logit = np.zeros(total_legal, dtype=np.float32)
        gumbel_policy_valid = np.zeros(r, dtype=np.float32)
    policy_surprise = np.frombuffer(bytes(result["policy_surprise"]), dtype=np.float32, count=r)
    # opp_coverage is f64 (see replay_expand.rs RowOut::opp_coverage).
    opp_coverage = np.frombuffer(bytes(result["opp_coverage"]), dtype=np.float64, count=r)
    value = np.frombuffer(bytes(result["value"]), dtype=np.float32, count=r)
    moves_left = np.frombuffer(bytes(result["moves_left"]), dtype=np.float32, count=r)
    moves_left_mask = np.frombuffer(bytes(result["moves_left_mask"]), dtype=np.float32, count=r)
    # value_mask gates the value/stvalue/cell_q heads for truncated-game rows
    # (outcome_valid==0). The serial path (samples.expand_sample) derives it from
    # metadata['truncated']; the Rust kernel reads the outcome_valid column and
    # emits value_mask (and zeroes stvalue_mask/cell_q_mask for truncated rows).
    # If the kernel omits the buffer, all rows are treated as completed.
    if "value_mask" in result:
        value_mask = np.frombuffer(bytes(result["value_mask"]), dtype=np.float32, count=r)
    else:
        value_mask = np.ones(r, dtype=np.float32)
    stvalue = np.frombuffer(bytes(result["stvalue"]), dtype=np.float32, count=r * horizons_len).reshape(r, horizons_len)
    stvalue_mask = np.frombuffer(bytes(result["stvalue_mask"]), dtype=np.float32, count=r * horizons_len).reshape(r, horizons_len)

    # Per-row slices are copied so each ExpandedRow owns independent, writable
    # memory (matching the serial path's fresh per-row arrays and avoiding
    # torch.from_numpy's read-only-tensor warning). `np.frombuffer` arrays are
    # read-only, and a contiguous slice stays read-only, so `.copy()` is required
    # rather than np.ascontiguousarray.
    rows: list[ExpandedRow | None] = []
    for k in range(r):
        if not valid[k]:
            rows.append(None)
            continue
        a, b = int(node_off[k]), int(node_off[k + 1])
        pa, pb = int(pol_off[k]), int(pol_off[k + 1])
        lc = int(legal_count[k])
        sup = Support(
            coords=coords[a:b].copy(),
            legal_count=lc,
            stone_count=int(stone_count[k]),
            halo_count=int(halo_count[k]),
            dist=dist[a:b].copy(),
            nbr=nbr[a:b].copy(),
            index={},  # not read on the assembled row; see docstring
        )
        rows.append(
            ExpandedRow(
                support=sup,
                feats=feats[a:b].copy(),
                policy=policy[pa:pb].copy(),
                opp_policy=opp_policy[pa:pb].copy(),
                opp_coverage=float(opp_coverage[k]),
                value=float(value[k]),
                value_mask=float(value_mask[k]),
                stvalue=stvalue[k].copy(),
                stvalue_mask=stvalue_mask[k].copy(),
                moves_left=float(moves_left[k]),
                moves_left_mask=float(moves_left_mask[k]),
                cell_q=cell_q[pa:pb].copy(),
                cell_q_mask=cell_q_mask[pa:pb].copy(),
                policy_surprise=float(policy_surprise[k]),
                gumbel_policy=gumbel_policy[pa:pb].copy(),
                gumbel_policy_valid=float(gumbel_policy_valid[k]),
                prior_logit=prior_logit[pa:pb].copy(),
                raylen=raylen[a:b].copy(),
            )
        )
    return rows, valid


def _expand_rows_rust(
    window: PackedWindow,
    index: list[int],
    d6: np.ndarray,
    horizons: tuple[int, ...],
    tolerate_off_legal: bool,
) -> tuple[list[ExpandedRow | None], np.ndarray]:
    """Dispatch ``index`` of ``window`` to the Rust rayon kernel and reassemble.

    The per-row D6 vector is pre-drawn and passed positionally (no rng in the
    kernel). ``horizons`` is the config horizon set; the kernel copies the stored
    ``stvalue`` columns and uses ``len(horizons)`` to slice the block.
    """
    from . import _rust  # local import: the package imports without the .so

    horizons_len = len(horizons)
    # The kernel copies the stored stvalue block POSITIONALLY (facts.stvalue[..len]),
    # unlike the serial python path which remaps by horizon VALUE (samples.py
    # horizon_index). A width-only check would pass a re-tuned horizon set of the
    # same length (e.g. reordered, or different values) against old shards and
    # silently train the STV heads on the wrong horizon's target. Require exact
    # tuple equality so the positional copy is provably aligned.
    if tuple(window.horizons) != tuple(horizons):
        raise ValueError(
            f"rust backend: window horizons {tuple(window.horizons)} "
            f"!= requested horizons {tuple(horizons)}; the kernel copies stvalue "
            f"positionally and cannot remap by horizon value"
        )
    columns = _window_columns_as_bytes(window)
    row_index = np.asarray(index, dtype=np.int64)
    # d6 may be longer than the expanded set (contract: len(d6) >= len(index)).
    # The kernel requires len(d6) == len(row_index), so slice to the aligned head.
    d6_i32 = np.asarray(d6, dtype=np.int32)[: row_index.shape[0]]
    result = _rust.expand_shard_train(
        columns,
        int(window.n),
        row_index.tolist(),
        d6_i32.tolist(),
        horizons_len,
        int(_resolve_support_radius()),
        bool(tolerate_off_legal),
    )
    return _reassemble_rust_rows(result, horizons_len)


# ---------------------------------------------------------------------------
# Public dispatch entry point.
# ---------------------------------------------------------------------------
def expand_rows(
    window: PackedWindow,
    survivor_index: Sequence[int] | np.ndarray | None,
    d6: np.ndarray,
    horizons: Sequence[int] = STV_HORIZONS,
    *,
    support_radius: int | None = None,
    tolerate_off_legal: bool = False,
    backend: str = "serial",
    workers: int = 0,
    pool: Any | None = None,
) -> tuple[list[ExpandedRow | None], np.ndarray]:
    """Expand a window's rows under their pre-drawn D6 symmetries.

    Parameters
    ----------
    window
        The packed in-RAM replay window.
    survivor_index
        The rows to expand, in expansion order. ``None`` ⇒ all rows
        ``range(window.n)``. When given, ``d6`` is indexed by the same positions
        (``d6[k]`` is the symmetry for ``survivor_index[k]``), so the result is
        positionally aligned to ``survivor_index``.
    d6
        Pre-drawn per-row symmetry vector (drawn on the main thread). Length must
        cover the expanded rows (``window.n`` when ``survivor_index`` is ``None``;
        otherwise ``len(survivor_index)``).
    horizons
        STV horizons passed verbatim to ``expand_sample``.
    support_radius
        Not read here. The support radius is read from the
        ``HEXFIELD_EQ_SUPPORT_RADIUS`` env at ``support`` import time (shared by
        the serial main thread and every spawn worker, since the pool inherits the
        environment). Accepted for signature symmetry with the Rust kernel, which
        takes it explicitly.
    tolerate_off_legal
        When True, an off-legal row is flagged invalid (mask ``False``) instead of
        raising.
    backend
        ``"serial"`` | ``"pool"`` | ``"rust"`` (the rayon kernel).
    workers
        Pool worker count (``0`` ⇒ auto via :func:`resolve_expand_workers`); only
        consulted for ``backend="pool"`` when no ``pool`` is supplied.
    pool
        An optional pre-built persistent ``ProcessPoolExecutor`` (the trainer owns
        one across epochs). When ``None`` and ``backend="pool"`` needs a pool, an
        ephemeral pool is created and torn down within the call.

    Returns
    -------
    ``(rows, valid)``
        ``rows`` is the list of ``ExpandedRow`` (``None`` for an off-legal-skipped
        row), aligned 1:1 to the expansion order; ``valid`` is the ``bool``
        numpy mask of the same length. The caller applies the survivor filter,
        survivor permutation, and ``effective_rows`` truncation.
    """
    _ = support_radius  # not read here; env-sourced at support import (see docstring)
    horizons = tuple(int(h) for h in horizons)

    if survivor_index is None:
        index = list(range(int(window.n)))
    else:
        index = [int(i) for i in survivor_index]
    n = len(index)

    d6 = np.asarray(d6)
    if d6.shape[0] < n:
        raise ValueError(
            f"expand_rows: d6 vector length {d6.shape[0]} < rows to expand {n}"
        )

    if n == 0:
        return [], np.zeros(0, dtype=bool)

    if backend == "rust":
        # One parallel expand_shard_train call; zero-copy buffers reassembled into
        # the same (rows, valid) shape as the serial/pool paths. Off-legal rows are
        # flagged invalid in the mask rather than dropped. `workers` and `pool` are
        # ignored (rayon manages its own thread pool).
        return _expand_rows_rust(window, index, d6, horizons, tolerate_off_legal)
    if backend not in ("serial", "pool"):
        raise ValueError(f"unknown expand_backend {backend!r} (serial|pool|rust)")

    # --- serial path --------------------------------------------------------
    # Also handles the pool backend below _PARALLEL_MIN_ROWS, skipping the spawn
    # and pickle cost.
    if backend == "serial" or n < _PARALLEL_MIN_ROWS:
        rows: list[ExpandedRow | None] = []
        valid = np.zeros(n, dtype=bool)
        for k, row_i in enumerate(index):
            sample = _row_view_to_sample(window.row_view(row_i))
            row, ok = _expand_one(sample, int(d6[k]), horizons, tolerate_off_legal)
            rows.append(row)
            valid[k] = ok
        return rows, valid

    # --- pool: persistent spawn ProcessPool ---------------------------------
    # Build the picklable per-row facts on the main thread using the same
    # row->sample shim as the serial path.
    samples = [_row_view_to_sample(window.row_view(row_i)) for row_i in index]

    owns_pool = pool is None
    if pool is None:
        n_workers = resolve_expand_workers(workers)
        import multiprocessing as mp

        pool = ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp.get_context("spawn")
        )
    else:
        n_workers = getattr(pool, "_max_workers", resolve_expand_workers(workers))

    rows = [None] * n  # type: ignore[assignment]
    valid = np.zeros(n, dtype=bool)
    bounds = _chunk_bounds(n, _CHUNK_ROWS)
    try:
        pending = iter(bounds)
        inflight: dict[Any, tuple[int, int]] = {}

        def _submit_one() -> bool:
            for start, stop in pending:
                fut = pool.submit(
                    expand_chunk_to_arrays,
                    samples[start:stop],
                    np.asarray(d6[start:stop]),
                    horizons,
                    tolerate_off_legal,
                )
                inflight[fut] = (start, stop)
                return True
            return False

        # Prime workers+2 chunks inflight, then refill on each completion. Each
        # chunk's rows are written back at their [start, stop) offset, so the
        # final arrays are in original order regardless of completion order.
        for _ in range(n_workers + 2):
            if not _submit_one():
                break
        while inflight:
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                start, stop = inflight.pop(fut)
                chunk_rows, chunk_valid = fut.result()
                rows[start:stop] = chunk_rows
                valid[start:stop] = chunk_valid
                _submit_one()
    finally:
        if owns_pool:
            pool.shutdown(wait=False, cancel_futures=True)

    return rows, valid
