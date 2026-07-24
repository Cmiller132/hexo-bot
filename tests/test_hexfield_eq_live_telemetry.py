"""CPU-only parity gate for opt-in live Gumbel search telemetry."""

from __future__ import annotations

import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    _ROOT / "packages" / "hexfield_eq" / "python",
    _ROOT / "packages" / "hexo_engine" / "python",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction, unpack_coord_id

try:
    from hexfield_eq import _rust
except ImportError:  # pragma: no cover - focused native gate reports a skip
    _rust = None


pytestmark = pytest.mark.skipif(
    _rust is None,
    reason="hexfield_eq._rust not built; rebuild the isolated worktree extension",
)

_VISITS = 96
_GUMBEL_M = 8
_OVERRIDES = {
    "gumbel_target": False,
    "gumbel_root": True,
    "gumbel_sequential_halving": True,
    "gumbel_nonroot_select": False,
    "gumbel_c_visit": 50.0,
    "gumbel_c_scale": 1.0,
    "gumbel_m": _GUMBEL_M,
    "gumbel_draw_temperature": 1.0,
}


def _model_logit(index: int) -> float:
    """Deterministic nonuniform raw logit, rounded onto the evaluator's f32 ABI."""

    value = float((index % 13) - 6) * 0.125
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _apply_coord(state: Any, q: int, r: int) -> None:
    result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    assert not result.terminal, (q, r)


def _compact_state() -> Any:
    """One centered stone: one compact radius-8 legal disk."""

    state = api.new_game()
    _apply_coord(state, 0, 0)
    return state


def _wide_state() -> Any:
    """A legal nonwinning chain of separated stones that widens the legal union."""

    state = _compact_state()
    for _ in range(5):
        candidates = []
        for action_id in api.legal_action_ids(state):
            coord = unpack_coord_id(action_id)
            candidates.append((coord.q, -abs(coord.r), -coord.r, coord))
        coord = max(candidates, key=lambda item: item[:3])[-1]
        _apply_coord(state, coord.q, coord.r)
    return state


class _DeterministicGumbelEvaluator:
    """Torch-free evaluator that also records the exact Rust request schedule."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, payload: dict[str, Any]) -> dict[str, bytes]:
        batch, total_nodes = (int(value) for value in payload["shape"])
        legal_counts_bytes = bytes(payload["legal_counts"])
        legal_counts = struct.unpack(f"<{batch}i", legal_counts_bytes)
        node_row_offsets = tuple(
            int(value) for value in payload["node_row_offsets"]
        )
        self.calls.append(
            (
                (batch, total_nodes),
                bool(payload.get("request_moves_left")),
                bool(payload.get("request_logits")),
                struct.pack(
                    f"<{len(node_row_offsets)}q", *node_row_offsets
                ),
                legal_counts_bytes,
                bytes(payload["node_feats"]),
                bytes(payload["node_qr"]),
                bytes(payload["nbr"]),
                bytes(payload["raylen"]),
            )
        )

        values: list[float] = []
        priors: list[float] = []
        logits: list[float] = []
        for legal_count in legal_counts:
            assert legal_count > 0
            values.append(float((legal_count % 17) - 8) / 20.0)
            row_logits = [_model_logit(index) for index in range(legal_count)]
            row_max = max(row_logits)
            row_exps = [math.exp(logit - row_max) for logit in row_logits]
            row_total = sum(row_exps)
            priors.extend(value / row_total for value in row_exps)
            logits.extend(row_logits)

        reply = {
            "values_bytes": struct.pack(f"<{len(values)}f", *values),
            "priors_bytes": struct.pack(f"<{len(priors)}f", *priors),
        }
        if payload.get("request_moves_left"):
            reply["moves_left_bytes"] = struct.pack(
                f"<{batch}f", *([60.0] * batch)
            )
        if payload.get("request_logits"):
            reply["priors_logits_bytes"] = struct.pack(
                f"<{len(logits)}f", *logits
            )
        return reply


def _run_search(
    state: Any,
    *,
    telemetry: bool,
    callback: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[tuple[Any, ...], ...]]:
    evaluator = _DeterministicGumbelEvaluator()
    events: list[dict[str, Any]] = []

    if callback is None and telemetry:

        def callback(event: dict[str, Any]) -> None:
            events.append(dict(event))

    kwargs: dict[str, Any] = {}
    if telemetry:
        kwargs["telemetry_callback"] = callback

    session = _rust.HexfieldMctsSession(max_states=65_536)
    result = session.search(
        [0xC0FFEE],
        (state,),
        visits=_VISITS,
        c_puct=1.5,
        temperature=1.0,
        seed=0x51DE_CAFE,
        evaluator=evaluator,
        virtual_batch_size=8,
        active_root_limit=1,
        root_policy_temperature=1.05,
        fpu_reduction=0.2,
        virtual_loss=1.0,
        widening_policy_mass=0.95,
        widening_max_children=96,
        widening_min_children=2,
        forced_playout_k=0.0,
        tss_enabled=False,
        search_parity_mode=True,
        divergence_overrides=_OVERRIDES,
        **kwargs,
    )[0]
    return dict(result), events, tuple(evaluator.calls)


def _authoritative_bytes(result: dict[str, Any]) -> tuple[bytes, ...]:
    """The brief's hard byte-identity surface (including f32 root value)."""

    return (
        struct.pack("<I", int(result["action_id"])),
        struct.pack("<f", float(result["root_value"])),
        struct.pack("<I", int(result["visits"])),
        bytes(result["visit_policy_action_ids_bytes"]),
        bytes(result["visit_policy_weights_bytes"]),
    )


def _assert_event_shape(
    events: list[dict[str, Any]],
    state: Any,
    legal_count: int,
    result: dict[str, Any],
) -> None:
    phases = [event["phase"] for event in events]
    assert phases[0] == "start"
    assert phases[-1] == "complete"
    assert "round" in phases

    start = events[0]
    assert start["root_index"] == 0
    assert start["game_key"] == 0xC0FFEE
    assert start["visits"] == 0
    assert start["target_visits"] == _VISITS
    assert start["policy_count"] == legal_count
    assert len(bytes(start["policy_action_ids_bytes"])) == legal_count * 4
    assert len(bytes(start["policy_weights_bytes"])) == legal_count * 4
    assert len(bytes(start["policy_visits_bytes"])) == legal_count * 4
    assert start["survivor_count"] == _GUMBEL_M
    assert len(bytes(start["survivor_action_ids_bytes"])) == _GUMBEL_M * 4

    # START is the model policy from raw logits, not the 1.05-temperature
    # search prior. The evaluator ABI aligns logits with legal coordinates in
    # ascending (q, r) order; telemetry emits action IDs in numeric order.
    model_order = sorted(
        (int(action_id) for action_id in api.legal_action_ids(state)),
        key=lambda action_id: (
            unpack_coord_id(action_id).q,
            unpack_coord_id(action_id).r,
        ),
    )
    logit_by_action = {
        action_id: _model_logit(index)
        for index, action_id in enumerate(model_order)
    }
    start_ids = struct.unpack(
        f"<{legal_count}I", bytes(start["policy_action_ids_bytes"])
    )
    start_weights = struct.unpack(
        f"<{legal_count}f", bytes(start["policy_weights_bytes"])
    )
    expected_logits = [logit_by_action[action_id] for action_id in start_ids]
    expected_max = max(expected_logits)
    expected_exps = [
        math.exp(logit - expected_max) for logit in expected_logits
    ]
    expected_total = sum(expected_exps)
    expected_weights = [value / expected_total for value in expected_exps]
    assert start_weights == pytest.approx(expected_weights, abs=1.0e-6)

    rounds = [event for event in events if event["phase"] == "round"]
    assert [event["round"] for event in rounds] == [0, 1]
    assert [event["rounds"] for event in rounds] == [3, 3]
    assert [event["policy_count"] for event in rounds] == [8, 4]
    assert [event["survivor_count"] for event in rounds] == [4, 2]
    assert [event["visits"] for event in rounds] == sorted(
        event["visits"] for event in rounds
    )
    for event in rounds:
        count = int(event["policy_count"])
        assert len(bytes(event["policy_action_ids_bytes"])) == count * 4
        assert len(bytes(event["policy_weights_bytes"])) == count * 4
        assert len(bytes(event["policy_visits_bytes"])) == count * 4
        weights = struct.unpack(
            f"<{count}f", bytes(event["policy_weights_bytes"])
        )
        assert all(math.isfinite(weight) and weight >= 0.0 for weight in weights)
        assert sum(weights) == pytest.approx(1.0, abs=1.0e-6)
        # Rust emits candidates in the exact SH-score order and normalizes
        # those same scores for visualization; the first heat leader is
        # therefore the leader that actually drives the halving.
        assert weights[0] == max(weights)

    complete = events[-1]
    assert int(complete["action_id"]) == int(result["action_id"])
    assert struct.pack("<f", float(complete["root_value"])) == struct.pack(
        "<f", float(result["root_value"])
    )
    assert int(complete["visits"]) == int(result["visits"])
    assert bytes(complete["visit_policy_action_ids_bytes"]) == bytes(
        result["visit_policy_action_ids_bytes"]
    )
    assert bytes(complete["visit_policy_weights_bytes"]) == bytes(
        result["visit_policy_weights_bytes"]
    )


@pytest.mark.parametrize(
    "state_factory",
    [_compact_state, _wide_state],
    ids=["compact", "wide"],
)
def test_live_telemetry_is_byte_identical_and_schedule_identical(
    state_factory: Any,
) -> None:
    state = state_factory()
    legal_count = len(api.legal_action_ids(state))
    assert legal_count >= _GUMBEL_M

    without, without_events, without_calls = _run_search(state, telemetry=False)
    with_events_result, events, with_calls = _run_search(state, telemetry=True)

    assert without_events == []
    assert set(without) == set(with_events_result)
    assert _authoritative_bytes(without) == _authoritative_bytes(with_events_result)
    assert without_calls == with_calls, "telemetry changed evaluator batching/schedule"
    _assert_event_shape(events, state, legal_count, with_events_result)


def test_wide_fixture_is_materially_wider_than_compact() -> None:
    compact = len(api.legal_action_ids(_compact_state()))
    wide = len(api.legal_action_ids(_wide_state()))
    assert wide > compact + 200, (compact, wide)


def test_callback_exception_is_swallowed_and_search_still_returns() -> None:
    calls: list[str] = []

    def raising_callback(event: dict[str, Any]) -> None:
        calls.append(str(event["phase"]))
        raise RuntimeError("telemetry consumer failed")

    result, events, evaluator_calls = _run_search(
        _compact_state(), telemetry=True, callback=raising_callback
    )
    assert int(result["visits"]) > 0
    assert evaluator_calls
    assert events == []
    assert calls
    assert calls[0] == "start"
    assert "round" in calls
    assert calls[-1] == "complete"
