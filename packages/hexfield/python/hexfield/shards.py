"""hexfield_compact_v1 shard (de)serialization and the restnet adapter.

One columnar ``.npz`` plus JSON sidecar per game; encoders expand columns at
train read. Layout: no legal-id column (legality is derived from stones);
stones and history are unified into one column
``(q i16, r i16, owner u8, placement_index u16)``; ``phase`` is stored as a u8
enum; standing-win cell columns are included.

The restnet adapter reads restnet compact-v1 shards by re-implementing their
column layout here (dense_cnn_restnet is not imported at runtime). It ignores
the stored legal_ids and derives legality from stones; win-now cells are
derived from the stored stones via the window scan.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import window_scan
from .samples import STV_HORIZONS, HexfieldSampleData

SCHEMA = "hexfield_compact_v1"
# v2 added the per-action `gumbel_pol_w` (improved-policy target weight, ALIGNED
# to `pol_act`) and `prior_logit` (raw root logit, aligned to `pol_act`)
# columns, plus a per-row `gumbel_present` flag. v3 replaces the aligned gumbel
# weight with its own CSR columns (`gumbel_act`/`gumbel_w`/`gumbel_off`): the π'
# target support (cumulative root visits >= min_visits) is a SUPERSET of the
# recorded delta-visit support, and the v2 alignment silently dropped π' mass on
# actions outside `pol_act` (e.g. inherited edges on reused roots) — exactly the
# rows where π' disagrees with the visit policy and carries teaching signal.
# The reader accepts all versions; rows without gumbel columns fall back to the
# visit target.
SCHEMA_VERSION = 3
_ACCEPTED_SCHEMA_VERSIONS = (1, 2, 3)
# The restnet compact-v1 schema_version the adapter reads
# (dense_cnn_restnet.compact_io.COMPACT_SCHEMA_VERSION). The adapter accepts a
# shard with no schema_version but raises on a present-but-different version.
LEGACY_RESTNET_SCHEMA_VERSION = 1
_PHASES = ("Opening", "FirstStone", "SecondStone")
_PHASE_INDEX = {name: i for i, name in enumerate(_PHASES)}


def _concat_offsets(lengths: Sequence[int]) -> np.ndarray:
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
    return offsets


def _pack_qr(points: Sequence[tuple[int, int]]) -> np.ndarray:
    if not points:
        return np.empty(0, dtype=np.int16)
    flat = np.empty(2 * len(points), dtype=np.int16)
    flat[0::2] = [int(q) for q, _ in points]
    flat[1::2] = [int(r) for _, r in points]
    return flat


def _unpack_qr(flat: np.ndarray, off: np.ndarray, i: int) -> tuple[tuple[int, int], ...]:
    a, b = int(off[i]), int(off[i + 1])
    seg = flat[2 * a : 2 * b]
    return tuple((int(seg[2 * k]), int(seg[2 * k + 1])) for k in range(b - a))


def write_compact_shard(
    path: Path,
    samples: Sequence[HexfieldSampleData],
    *,
    short_term_value_horizons: Sequence[int] = STV_HORIZONS,
    sidecar: dict | None = None,
) -> int:
    """Serialize rows into one hexfield_compact_v1 ``.npz`` + JSON sidecar."""

    horizons = tuple(int(h) for h in short_term_value_horizons)
    n = len(samples)
    h = len(horizons)
    horizon_index = {hz: i for i, hz in enumerate(horizons)}

    turn_index = np.empty(n, dtype=np.int32)
    current_player = np.empty(n, dtype=np.uint8)
    phase = np.empty(n, dtype=np.uint8)
    value = np.empty(n, dtype=np.float32)
    moves_left = np.full(n, -1.0, dtype=np.float32)
    # outcome_valid[i] == 0 marks a truncated-game row (no engine winner): the
    # value/stvalue/cell_q heads are masked to zero loss at expand time. Defaults
    # to 1 (completed). Shards lacking this column read back as all-1 (see
    # read_compact_shard). Derived from metadata['truncated'].
    outcome_valid = np.ones(n, dtype=np.uint8)
    # policy_valid[i] == 0 marks a value-only (fast) row: policy/opp_policy/
    # soft_policy/cell_q are masked at expand and loss; value/stvalue/moves_left
    # still train. Defaults to 1 (full). Shards lacking it read back as all-1
    # (see read_compact_shard). Derived from metadata['pcr_full'].
    policy_valid = np.ones(n, dtype=np.uint8)
    policy_surprise = np.zeros(n, dtype=np.float32)
    first_q = np.zeros(n, dtype=np.int16)
    first_r = np.zeros(n, dtype=np.int16)
    first_present = np.zeros(n, dtype=np.uint8)
    stvalue = np.zeros((n, h), dtype=np.float32)
    stvalue_mask = np.zeros((n, h), dtype=np.float32)

    hist_qr: list[np.ndarray] = []
    hist_owner: list[np.ndarray] = []
    hist_pidx: list[np.ndarray] = []
    hist_len: list[int] = []
    cell_cols: dict[str, tuple[list[np.ndarray], list[int]]] = {
        key: ([], []) for key in ("own_hot", "opp_hot", "own_win", "opp_win")
    }
    pol_act: list[np.ndarray] = []
    pol_w: list[np.ndarray] = []
    pol_q: list[np.ndarray] = []  # child Q parallel to pol_act (cell_q head target)
    # Improved-policy target (π') CSR columns: OWN action/weight/offset arrays,
    # NOT aligned to pol_act — π''s support is a superset of the delta-visit
    # support, and aligning would drop off-support π' mass (v2's silent-
    # truncation defect). Raw root logit stays aligned to pol_act.
    # gumbel_present marks rows that carry a gumbel target.
    g_act: list[np.ndarray] = []
    g_w: list[np.ndarray] = []
    g_len: list[int] = []
    pol_logit: list[np.ndarray] = []
    gumbel_present = np.zeros(n, dtype=np.uint8)
    pol_len: list[int] = []
    opp_act: list[np.ndarray] = []
    opp_w: list[np.ndarray] = []
    opp_len: list[int] = []

    for i, sample in enumerate(samples):
        turn_index[i] = int(sample.turn_index)
        current_player[i] = int(sample.current_player)
        phase[i] = _PHASE_INDEX[str(sample.phase)]
        value[i] = float(sample.value)
        moves_left[i] = float(sample.moves_left)
        outcome_valid[i] = 0 if bool(sample.metadata.get("truncated", False)) else 1
        policy_valid[i] = 1 if bool(sample.metadata.get("pcr_full", True)) else 0
        policy_surprise[i] = float(sample.policy_surprise)
        if sample.first_stone is not None:
            first_q[i] = int(sample.first_stone[0])
            first_r[i] = int(sample.first_stone[1])
            first_present[i] = 1
        for hz, val in sample.short_term_value:
            col = horizon_index.get(int(hz))
            if col is not None:
                stvalue[i, col] = float(val)
                stvalue_mask[i, col] = 1.0

        qr = _pack_qr([(q, r) for q, r, _o, _p in sample.records])
        hist_qr.append(qr)
        hist_owner.append(np.asarray([o for _q, _r, o, _p in sample.records], dtype=np.uint8))
        hist_pidx.append(np.asarray([p for _q, _r, _o, p in sample.records], dtype=np.uint16))
        hist_len.append(len(sample.records))

        for key, cells in (
            ("own_hot", sample.own_hot),
            ("opp_hot", sample.opp_hot),
            ("own_win", sample.own_win),
            ("opp_win", sample.opp_win),
        ):
            packed = _pack_qr(tuple(cells))
            cell_cols[key][0].append(packed)
            cell_cols[key][1].append(packed.shape[0] // 2)

        pa = np.fromiter((int(a) for a, _ in sample.policy), dtype=np.uint32, count=len(sample.policy))
        pw = np.fromiter((float(w) for _, w in sample.policy), dtype=np.float32, count=len(sample.policy))
        # Child Q for the cell_q head, aligned to the recorded policy action
        # order (0 for actions absent from q_policy).
        qmap = {int(a): float(q) for a, q in sample.q_policy}
        pq = np.fromiter((qmap.get(int(a), 0.0) for a in pa.tolist()), dtype=np.float32, count=pa.shape[0])
        # π' target on its OWN support (CSR). gumbel_present[i] flags rows that
        # carry a target, so the dense reconstruct can distinguish an all-zero
        # target from an absent one. Raw logit stays aligned to pol_act.
        ga = np.fromiter(
            (int(a) for a, _ in sample.gumbel_policy),
            dtype=np.uint32,
            count=len(sample.gumbel_policy),
        )
        gw = np.fromiter(
            (float(w) for _, w in sample.gumbel_policy),
            dtype=np.float32,
            count=len(sample.gumbel_policy),
        )
        lmap = {int(a): float(l) for a, l in sample.prior_logit}
        pl = np.fromiter((lmap.get(int(a), 0.0) for a in pa.tolist()), dtype=np.float32, count=pa.shape[0])
        if sample.gumbel_policy:
            gumbel_present[i] = 1
        pol_act.append(pa)
        pol_w.append(pw)
        pol_q.append(pq)
        g_act.append(ga)
        g_w.append(gw)
        g_len.append(int(ga.shape[0]))
        pol_logit.append(pl)
        pol_len.append(int(pa.shape[0]))
        oa = np.fromiter((int(a) for a, _ in sample.opp_policy), dtype=np.uint32, count=len(sample.opp_policy))
        ow = np.fromiter((float(w) for _, w in sample.opp_policy), dtype=np.float32, count=len(sample.opp_policy))
        opp_act.append(oa)
        opp_w.append(ow)
        opp_len.append(int(oa.shape[0]))

    def _cat(parts: list[np.ndarray], dtype) -> np.ndarray:
        if not parts:
            return np.empty(0, dtype=dtype)
        return np.concatenate(parts).astype(dtype, copy=False)

    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "num_rows": np.asarray(n, dtype=np.int64),
        "horizons": np.asarray(horizons, dtype=np.int32),
        "turn_index": turn_index,
        "current_player": current_player,
        "phase": phase,
        "value": value,
        "moves_left": moves_left,
        "outcome_valid": outcome_valid,
        "policy_valid": policy_valid,
        "first_q": first_q,
        "first_r": first_r,
        "first_present": first_present,
        "stvalue": stvalue,
        "stvalue_mask": stvalue_mask,
        "hist_qr": _cat(hist_qr, np.int16),
        "hist_owner": _cat(hist_owner, np.uint8),
        "hist_pidx": _cat(hist_pidx, np.uint16),
        "hist_off": _concat_offsets(hist_len),
        "pol_act": _cat(pol_act, np.uint32),
        "pol_w": _cat(pol_w, np.float32),
        "q_pol_q": _cat(pol_q, np.float32),
        "gumbel_act": _cat(g_act, np.uint32),
        "gumbel_w": _cat(g_w, np.float32),
        "gumbel_off": _concat_offsets(g_len),
        "prior_logit": _cat(pol_logit, np.float32),
        "gumbel_present": gumbel_present,
        "pol_off": _concat_offsets(pol_len),
        "policy_surprise": policy_surprise,
        "opp_act": _cat(opp_act, np.uint32),
        "opp_w": _cat(opp_w, np.float32),
        "opp_off": _concat_offsets(opp_len),
    }
    for key, (parts, lens) in cell_cols.items():
        arrays[f"{key}_qr"] = _cat(parts, np.int16)
        arrays[f"{key}_off"] = _concat_offsets(lens)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write of BOTH files (the host takes real power cuts). Commit order is
    # npz-first-then-sidecar: the sidecar is the commit marker — a shard is only
    # counted downstream once its sidecar exists (buffer_manifest._build_entry
    # skips a sidecar-less npz). Writing the sidecar last means a cut between the
    # two leaves an orphan npz (harmless, skipped) rather than a sidecar pointing
    # at a torn/missing npz.
    #
    # np.savez_compressed APPENDS `.npz` to a str/Path lacking that suffix, which
    # would defeat the tmp name — so open the tmp file object ourselves and pass
    # THE HANDLE (savez writes to it verbatim, no suffix). The `.npz.tmp` name
    # also stays off the buffer-manifest glob `epoch_*/game_*.npz` (must end
    # `.npz`).
    npz_tmp = path.with_name(path.name + ".tmp")
    try:
        npz_tmp.unlink()  # clean any stale tmp from a previous crash (guarded)
    except OSError:
        pass
    with open(npz_tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(npz_tmp, path)

    meta = {
        "lineage": "hexfield",
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "rows": n,
        "horizons": list(horizons),
        **(sidecar or {}),
    }
    sidecar_path = path.with_suffix(".json") if path.suffix == ".npz" else Path(str(path) + ".json")
    side_tmp = sidecar_path.with_name(sidecar_path.name + ".tmp")
    try:
        side_tmp.unlink()  # clean any stale tmp from a previous crash (guarded)
    except OSError:
        pass
    with open(side_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(side_tmp, sidecar_path)  # commit marker: publishes the shard

    # Best-effort directory fsync so both renames are durable. Guarded: fsync on
    # a directory fd is not portable to Windows test runs.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass
    return n


def read_compact_shard(path: Path) -> list[HexfieldSampleData]:
    """Decode a hexfield_compact_v1 shard back into sample rows."""

    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    if int(arrays["schema_version"]) not in _ACCEPTED_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported hexfield shard schema {int(arrays['schema_version'])}")

    n = int(arrays["num_rows"])
    horizons = [int(h) for h in arrays["horizons"]]
    # Absent for shards without this column; then all rows read as completed.
    outcome_valid = arrays.get("outcome_valid")
    # Absent for shards without this column; then all rows read as full.
    policy_valid = arrays.get("policy_valid")
    out: list[HexfieldSampleData] = []
    for i in range(n):
        h0, h1 = int(arrays["hist_off"][i]), int(arrays["hist_off"][i + 1])
        qr = arrays["hist_qr"][2 * h0 : 2 * h1]
        records = tuple(
            (
                int(qr[2 * k]),
                int(qr[2 * k + 1]),
                int(arrays["hist_owner"][h0 + k]),
                int(arrays["hist_pidx"][h0 + k]),
            )
            for k in range(h1 - h0)
        )
        p0, p1 = int(arrays["pol_off"][i]), int(arrays["pol_off"][i + 1])
        policy = tuple(
            (int(arrays["pol_act"][k]), float(arrays["pol_w"][k])) for k in range(p0, p1)
        )
        o0, o1 = int(arrays["opp_off"][i]), int(arrays["opp_off"][i + 1])
        opp_policy = tuple(
            (int(arrays["opp_act"][k]), float(arrays["opp_w"][k])) for k in range(o0, o1)
        )
        stval = tuple(
            (horizons[c], float(arrays["stvalue"][i, c]))
            for c in range(len(horizons))
            if arrays["stvalue_mask"][i, c] > 0.0
        )
        first = (
            (int(arrays["first_q"][i]), int(arrays["first_r"][i]))
            if int(arrays["first_present"][i]) == 1
            else None
        )
        # Read q_pol_q back into q_policy (parallel to pol_act). Guarded for
        # shards without the q_pol_q column, which decode to an empty q_policy.
        q_policy = (
            tuple((int(arrays["pol_act"][k]), float(arrays["q_pol_q"][k])) for k in range(p0, p1))
            if "q_pol_q" in arrays
            else ()
        )
        # Reconstruct the per-action improved-policy target. v3 shards carry π'
        # on its OWN CSR columns (gumbel_act/gumbel_w/gumbel_off — support can
        # exceed pol_act); v2 shards carried it aligned to pol_act (silently
        # truncated to the visit support — read as stored). gumbel_present marks
        # rows that carried a target; shards lacking these columns decode to
        # empty tuples, and the expand and loss fall back to the visit target.
        present = "gumbel_present" in arrays and int(arrays["gumbel_present"][i]) == 1
        if present and "gumbel_act" in arrays:
            g0, g1 = int(arrays["gumbel_off"][i]), int(arrays["gumbel_off"][i + 1])
            gumbel_policy = tuple(
                (int(arrays["gumbel_act"][k]), float(arrays["gumbel_w"][k]))
                for k in range(g0, g1)
            )
        elif present and "gumbel_pol_w" in arrays:
            gumbel_policy = tuple(
                (int(arrays["pol_act"][k]), float(arrays["gumbel_pol_w"][k]))
                for k in range(p0, p1)
            )
        else:
            gumbel_policy = ()
        prior_logit = (
            tuple(
                (int(arrays["pol_act"][k]), float(arrays["prior_logit"][k]))
                for k in range(p0, p1)
            )
            if present and "prior_logit" in arrays
            else ()
        )
        out.append(
            HexfieldSampleData(
                game_id="",
                turn_index=int(arrays["turn_index"][i]),
                current_player=int(arrays["current_player"][i]),
                phase=_PHASES[int(arrays["phase"][i])],
                records=records,
                first_stone=first,
                own_hot=_unpack_qr(arrays["own_hot_qr"], arrays["own_hot_off"], i),
                opp_hot=_unpack_qr(arrays["opp_hot_qr"], arrays["opp_hot_off"], i),
                own_win=_unpack_qr(arrays["own_win_qr"], arrays["own_win_off"], i),
                opp_win=_unpack_qr(arrays["opp_win_qr"], arrays["opp_win_off"], i),
                policy=policy,
                q_policy=q_policy,
                gumbel_policy=gumbel_policy,
                prior_logit=prior_logit,
                opp_policy=opp_policy,
                value=float(arrays["value"][i]),
                short_term_value=stval,
                moves_left=float(arrays["moves_left"][i]),
                metadata={
                    **(
                        {"truncated": True}
                        if outcome_valid is not None and int(outcome_valid[i]) == 0
                        else {}
                    ),
                    "pcr_full": bool(policy_valid is None or int(policy_valid[i]) != 0),
                },
            )
        )
    return out


def read_legacy_restnet_shard(path: Path) -> list[HexfieldSampleData]:
    """Read a restnet compact-v1 shard as hexfield rows.

    The stored legal_ids and crop center are ignored; legality re-derives from
    stones at expansion. Stored hot lists are read as raw engine coords;
    standing-win cells are derived from the stored stones via the window scan.
    The stored visit policies are used as-is. Rows are tagged
    ``metadata={"source": "legacy_shard"}``.
    """

    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    # Accept a shard with no schema_version; raise on a present-but-different
    # version.
    legacy_version = arrays.get("schema_version")
    if legacy_version is not None and int(legacy_version) != LEGACY_RESTNET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported legacy restnet shard schema {int(legacy_version)} "
            f"(adapter expects {LEGACY_RESTNET_SCHEMA_VERSION})"
        )

    n = int(arrays["num_rows"])
    horizons = [int(h) for h in arrays["horizons"]]
    moves_left = arrays.get("moves_left")
    out: list[HexfieldSampleData] = []
    for i in range(n):
        h0, h1 = int(arrays["hist_off"][i]), int(arrays["hist_off"][i + 1])
        qr = arrays["hist_qr"][2 * h0 : 2 * h1]
        records = tuple(
            (
                int(qr[2 * k]),
                int(qr[2 * k + 1]),
                int(arrays["hist_owner"][h0 + k]),
                int(arrays["hist_idx"][h0 + k]),
            )
            for k in range(h1 - h0)
        )
        s0, s1 = int(arrays["stones_off"][i]), int(arrays["stones_off"][i + 1])
        if (s1 - s0) != len(records):
            raise ValueError(
                f"legacy row {i}: stones ({s1 - s0}) != history ({len(records)}) — "
                "the unified-records assumption does not hold"
            )
        current = int(arrays["current_player"][i])
        own_win, opp_win = window_scan(records, current, len(records))[2:]
        p0, p1 = int(arrays["pol_off"][i]), int(arrays["pol_off"][i + 1])
        policy = tuple(
            (int(arrays["pol_act"][k]), float(arrays["pol_w"][k])) for k in range(p0, p1)
        )
        o0, o1 = int(arrays["opp_off"][i]), int(arrays["opp_off"][i + 1])
        opp_policy = tuple(
            (int(arrays["opp_act"][k]), float(arrays["opp_w"][k])) for k in range(o0, o1)
        )
        stval = tuple(
            (horizons[c], float(arrays["stvalue"][i, c]))
            for c in range(len(horizons))
            if arrays["stvalue_mask"][i, c] > 0.0
        )
        first = (
            (int(arrays["first_q"][i]), int(arrays["first_r"][i]))
            if int(arrays["first_present"][i]) == 1
            else None
        )
        out.append(
            HexfieldSampleData(
                game_id="",
                turn_index=int(arrays["turn_index"][i]),
                current_player=current,
                phase=str(arrays["phase"][i]),
                records=records,
                first_stone=first,
                own_hot=_unpack_qr(arrays["own_hot_qr"], arrays["own_hot_off"], i),
                opp_hot=_unpack_qr(arrays["opp_hot_qr"], arrays["opp_hot_off"], i),
                own_win=own_win,
                opp_win=opp_win,
                policy=policy,
                opp_policy=opp_policy,
                value=float(arrays["value"][i]),
                short_term_value=stval,
                moves_left=float(moves_left[i]) if moves_left is not None else -1.0,
                metadata={"source": "legacy_shard"},
            )
        )
    return out
