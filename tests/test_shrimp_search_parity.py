"""Stub-evaluator self-parity for the shrimp MCTS session.

Originally a differential harness against the (now retired) dense_cnn session;
restructured to a shrimp-vs-shrimp parity net that runs entirely on the
public native module. It pins the search's determinism and reproducibility
contracts: with the same seed stream, PUCT constants, and stub priors, two
independent sessions must yield bit-identical visit counts, chosen moves, root
values, and exported visit-policy targets; and ``search_parity_mode=True`` must
equal an explicit all-divergences-off override set.

The corpus is constrained (and asserted) to positions whose full legal set lies
within a radius-20 crop of the stones' rounded centroid, so the ``ShrimpStub``
evaluator (which keys priors/values by the legal cells' crop flats) is
well-defined; the legal set is asserted ascending in ``_fully_in_crop``.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from shrimp_testkit import api, sample_decision_states

from shrimp.geometry import hex_dist, unpack_action_id

try:
    from shrimp import _rust as shrimp_rust
except ImportError:  # pragma: no cover
    shrimp_rust = None

needs_native = pytest.mark.skipif(
    shrimp_rust is None,
    reason="shrimp native module not built",
)


def _stub_prior_from_flat(flat: int, row_hash: int) -> float:
    return float((flat * 2654435761 + row_hash * 97) % 1000 + 1)


def _stub_value_from_hash(row_hash: int) -> float:
    return float(row_hash % 2001 - 1000) / 1000.0


def _row_hash(flats: list[int]) -> int:
    h = 1469598103
    for flat in flats:
        h = (h ^ flat) * 1099511628211 % (1 << 61)
    return h % 1000003


def _python_round(numerator: int, denominator: int) -> int:
    """Integer division rounding half to even."""

    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled < denominator:
        return quotient
    if doubled > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def _hexd(dq: int, dr: int) -> int:
    return max(abs(dq), abs(dr), abs(dq + dr))


class ShrimpStub:
    """Evaluator over shrimp's CSR payload.

    Derives the crop center from the stones' rounded centroid, then maps each
    legal cell to its radius-20 crop flat. Legal cells outside the radius-20
    hex disk are assigned prior 0.0. Values are keyed by the row hash of the
    in-disk flats. Returns a reply with little-endian float32 ``values_bytes``
    (one per row) and ``priors_bytes`` (one per legal cell, row-concatenated),
    plus ``moves_left_bytes`` when the payload requests it.
    """

    def __call__(self, payload: dict) -> dict:
        b, total = payload["shape"]
        legal_counts = np.frombuffer(payload["legal_counts"], dtype=np.int32)
        offsets = np.asarray(payload["node_row_offsets"], dtype=np.int64)
        qr = np.frombuffer(payload["node_qr"], dtype=np.int16).reshape(total, 2)
        feats = np.frombuffer(payload["node_feats"], dtype=np.float16).reshape(total, 15)
        values = []
        priors: list[float] = []
        for g in range(b):
            o, e = int(offsets[g]), int(offsets[g + 1])
            l = int(legal_counts[g])
            legal = qr[o : o + l]
            seg = feats[o:e]
            stones = qr[o:e][(seg[:, 0] + seg[:, 1]) > 0.5]
            if len(stones):
                cq = _python_round(int(stones[:, 0].astype(np.int64).sum()), len(stones))
                cr = _python_round(int(stones[:, 1].astype(np.int64).sum()), len(stones))
            else:
                cq, cr = 0, 0
            in_disk_flats = []
            row_priors = []
            for q, r in legal:
                dq, dr = int(q) - cq, int(r) - cr
                if _hexd(dq, dr) <= 20:
                    flat = (dr + 20) * 41 + (dq + 20)
                    in_disk_flats.append((flat, len(row_priors)))
                    row_priors.append(None)  # filled below
                else:
                    row_priors.append(0.0)
            rh = _row_hash([f for f, _ in in_disk_flats])
            for flat, idx in in_disk_flats:
                row_priors[idx] = _stub_prior_from_flat(flat, rh)
            values.append(_stub_value_from_hash(rh))
            priors.extend(row_priors)
        reply = {
            "values_bytes": struct.pack(f"<{b}f", *values),
            "priors_bytes": struct.pack(f"<{len(priors)}f", *priors),
        }
        if payload.get("request_moves_left"):
            reply["moves_left_bytes"] = struct.pack(f"<{b}f", *([100.0] * b))
        return reply


def _crop_center(stones: list[tuple[int, int]]) -> tuple[int, int]:
    if not stones:
        return (0, 0)
    q = round(sum(s[0] for s in stones) / len(stones))
    r = round(sum(s[1] for s in stones) / len(stones))
    return int(q), int(r)


def _fully_in_crop(state, margin: int) -> bool:
    """Return True when every legal cell and every stone lies within
    `20 - margin` of the crop center. dense recomputes its centroid crop per
    state, so `margin` bounds how far leaf states reached during search can
    shift the crop while keeping their legal sets in-crop.

    Also asserts the engine's legal action ids are ascending.
    """

    mirror = api.to_python_state(state)
    stones = [(c.q, c.r) for c, _p in mirror.board.stones]
    cq, cr = _crop_center(stones)
    ids = api.legal_action_ids(state)
    if list(ids) != sorted(ids):
        raise AssertionError("engine legal ids not ascending — stub keying invalid")
    limit = 20 - margin
    for aid in ids:
        q, r = unpack_action_id(aid)
        if hex_dist(q - cq, r - cr) > limit:
            return False
    for q, r in stones:
        if hex_dist(q - cq, r - cr) > limit:
            return False
    return True


def _corpus(min_positions: int = 100, margin: int = 9):
    states = sample_decision_states(range(200), (1, 2, 3, 4, 5, 6, 7, 8))
    in_crop = [s for s in states if _fully_in_crop(s, margin)]
    assert len(in_crop) >= min_positions, f"only {len(in_crop)} in-crop positions"
    return in_crop[:min_positions]


_RESULT_INT_FIELDS = ("action_id", "visits", "visit_policy_count")
_RESULT_BYTE_FIELDS = (
    "visit_policy_action_ids_bytes",
    "visit_policy_weights_bytes",
    "root_prior_policy_action_ids_bytes",
    "root_prior_policy_weights_bytes",
)


def _search_records(states, *, visits, seed, temperature, root_temp, tss, virtual_batch,
                    parity_mode=True, divergence_overrides=None):
    """Run one search per state on a fresh session and capture the comparable
    fields of each result (ints, root value, and the exported policy byte
    blobs)."""

    session = shrimp_rust.ShrimpMctsSession(max_states=65536)
    stub = ShrimpStub()
    records = []
    for index, state in enumerate(states):
        key = 10_000 + index
        kwargs = dict(
            visits=visits,
            c_puct=1.5,
            temperature=temperature,
            seed=seed + index * 7919,
            virtual_batch_size=virtual_batch,
            fpu_reduction=0.2,
            virtual_loss=1.0,
            widening_policy_mass=0.95,
            widening_max_children=96,
            widening_min_children=2,
            root_policy_temperature=root_temp,
            tss_enabled=tss,
        )
        if divergence_overrides is not None:
            kwargs["divergence_overrides"] = divergence_overrides
        result = session.search(
            [key], (state,), evaluator=stub, search_parity_mode=parity_mode, **kwargs
        )[0]
        records.append(
            tuple(int(result[f]) for f in _RESULT_INT_FIELDS)
            + (round(float(result["root_value"]), 6),)
            + tuple(bytes(result[f]) for f in _RESULT_BYTE_FIELDS)
        )
        session.discard(key)
    return records


def _assert_records_equal(a, b) -> None:
    assert len(a) == len(b)
    mismatches = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert not mismatches, f"{len(mismatches)} mismatches; first 5: {mismatches[:5]}"


@needs_native
def test_search_is_deterministic_greedy_no_noise() -> None:
    """Two independent sessions with identical seeds/knobs produce bit-identical
    greedy results across the whole corpus — the seed->result contract the
    differential harness relied on for both engines."""

    states = _corpus(100)
    cfg = dict(visits=32, seed=11, temperature=0.0, root_temp=1.0, tss=True, virtual_batch=8)
    _assert_records_equal(_search_records(states, **cfg), _search_records(states, **cfg))


@needs_native
def test_search_is_deterministic_sampling_temperature() -> None:
    """Reproducibility under sampling temperature + the root-temp knob: the
    seeded sampler is deterministic, so two runs still match exactly. (The classic
    noise / forced-playout exploration machinery is gone; Gumbel replaces it.)"""

    states = _corpus(60)
    cfg = dict(visits=48, seed=23, temperature=1.0, root_temp=1.1, tss=True, virtual_batch=16)
    _assert_records_equal(_search_records(states, **cfg), _search_records(states, **cfg))


@needs_native
def test_search_is_deterministic_tss_disabled() -> None:
    states = _corpus(40)
    cfg = dict(visits=32, seed=31, temperature=0.0, root_temp=1.0, tss=False, virtual_batch=8)
    _assert_records_equal(_search_records(states, **cfg), _search_records(states, **cfg))


@needs_native
def test_search_parity_mode_equals_explicit_divergences_off() -> None:
    """``search_parity_mode=True`` is exactly the all-divergences-off
    configuration: passing the divergence flags off explicitly (their parity()
    values) reproduces the same results across the corpus. Pins that the parity
    knob and the explicit overrides describe the same deterministic search."""

    states = _corpus(60)
    cfg = dict(visits=32, seed=17, temperature=0.0, root_temp=1.0, tss=True, virtual_batch=8)
    parity = _search_records(states, parity_mode=True, **cfg)
    explicit = _search_records(
        states,
        parity_mode=False,
        divergence_overrides={
            "lcb_move_selection": False,
            "early_stop": False,
            "moves_left_utility": False,
            "nucleus_f64": False,
            "new_child_fpu": False,
            "lazy_widening": False,
            "clean_root_prior_cache": False,
        },
        **cfg,
    )
    _assert_records_equal(parity, explicit)
