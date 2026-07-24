"""Bounded in-process fan-out for live search Server-Sent Events.

The bot worker pool owns search execution; this module deliberately owns none
of it.  Publishers are synchronous and non-blocking, subscribers get a small
replay window plus a bounded asyncio queue, and dropping an SSE connection only
removes that subscriber.
"""

from __future__ import annotations

import asyncio
import json
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Any


_CLOSED = object()
_UINT32_SIZE = struct.calcsize("=I")
_FLOAT32_SIZE = struct.calcsize("=f")


class LiveSearchSubscriberLimit(RuntimeError):
    """Raised when one game already has its maximum number of viewers."""


def _wire_values(value: Any, fmt: str, size: int) -> list[int] | list[float]:
    try:
        wire = bytes(value)
        if len(wire) % size:
            return []
        return [item[0] for item in struct.iter_unpack(fmt, wire)]
    except Exception:
        return []


def _coord(action_id: int) -> tuple[int, int]:
    return (
        ((int(action_id) >> 16) & 0xFFFF) - (1 << 15),
        (int(action_id) & 0xFFFF) - (1 << 15),
    )


def _decoded_policy(raw: dict[str, Any]) -> list[dict[str, Any]]:
    ids = _wire_values(
        raw.get("policy_action_ids_bytes", b""), "=I", _UINT32_SIZE
    )
    weights = _wire_values(
        raw.get("policy_weights_bytes", b""), "=f", _FLOAT32_SIZE
    )
    visits = _wire_values(
        raw.get("policy_visits_bytes", b""), "=I", _UINT32_SIZE
    )
    if not ids:
        return []

    result_style = bool(raw.get("_result_policy"))
    total = float(sum(weights))
    if result_style:
        # Preserve the pre-refactor non-telemetry fallback shape.
        denominator = total or 1.0
        rows: list[dict[str, Any]] = []
        for action_id, weight in zip(ids, weights):
            q, r = _coord(int(action_id))
            rows.append(
                {
                    "q": q,
                    "r": r,
                    "weight": round(float(weight), 6),
                    "p": round(float(weight) / denominator, 6),
                }
            )
        rows.sort(key=lambda row: (-row["p"], row["q"], row["r"]))
        return rows

    rows = []
    for index, action_id in enumerate(ids):
        q, r = _coord(int(action_id))
        weight = float(weights[index]) if index < len(weights) else 0.0
        row: dict[str, Any] = {
            "q": q,
            "r": r,
            "p": weight / total if total > 0.0 else 0.0,
        }
        if index < len(visits):
            row["visits"] = int(visits[index])
        rows.append(row)
    return rows


def _decoded_fields(raw: dict[str, Any], phase: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if raw.get("_fallback_start") or raw.get("_post_search"):
        fields["post_search"] = True
    for key in ("round", "rounds", "visits", "target_visits"):
        try:
            if raw.get(key) is not None:
                fields[key] = int(raw[key])
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        if raw.get("root_value") is not None:
            fields["root_value"] = float(raw["root_value"])
    except (TypeError, ValueError, OverflowError):
        pass

    policy = _decoded_policy(raw)
    if policy:
        fields["policy"] = policy
    survivor_ids = _wire_values(
        raw.get("survivor_action_ids_bytes", b""), "=I", _UINT32_SIZE
    )
    if survivor_ids:
        fields["survivors"] = [
            {"q": q, "r": r}
            for action_id in survivor_ids
            for q, r in (_coord(int(action_id)),)
        ]
    # A tactical certificate overrode the search. The played cell can carry
    # zero policy weight (it is appended to the export with weight 0.0), so the
    # viewer needs to be told rather than left to read it as a stray move.
    if str(raw.get("action_selection", "")) == "tss_deep_root_win":
        fields["tss"] = True
    try:
        if raw.get("action_id") is not None:
            action_id = int(raw["action_id"])
            q, r = _coord(action_id)
            fields["action"] = {
                **({"action_id": action_id} if phase == "complete" else {}),
                "q": q,
                "r": r,
            }
    except (TypeError, ValueError, OverflowError):
        pass
    return fields


def expand_worker_event(event: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Expand a raw worker frame into the existing public event schema.

    BotPool invokes its progress callback on the asyncio loop thread, so this
    is intentionally the first place wide policy lists are constructed.
    Unknown/malformed telemetry phases are dropped; ordinary control events
    pass through unchanged.
    """
    if event.get("kind") != "search_telemetry":
        return (event,)
    try:
        phase = str(event.get("phase", "")).lower()
        tags = {
            key: event[key]
            for key in ("stone", "ply")
            if key in event
        }
        fields = _decoded_fields(event, phase)
        if phase == "start":
            bare_fields = dict(fields)
            bare_fields.pop("survivors", None)
            if event.get("_fallback_start"):
                bare_fields.pop("visits", None)
                bare_fields.pop("target_visits", None)

            candidate_fields = dict(fields)
            survivors = candidate_fields.get("survivors")
            policy = candidate_fields.get("policy")
            if isinstance(survivors, list) and isinstance(policy, list):
                keep = {
                    (row.get("q"), row.get("r"))
                    for row in survivors
                    if isinstance(row, dict)
                }
                candidate_fields["policy"] = [
                    row
                    for row in policy
                    if isinstance(row, dict)
                    and (row.get("q"), row.get("r")) in keep
                ]
            return (
                {"kind": "bare_policy", **tags, **bare_fields},
                {"kind": "candidate_set", **tags, **candidate_fields},
            )
        if phase == "round":
            return ({"kind": "search_round", **tags, **fields},)
        if phase == "complete":
            return ({"kind": "search_complete", **tags, **fields},)
    except Exception:
        # Telemetry remains strictly observational.
        pass
    return ()


def encode_sse_event(event: dict[str, Any]) -> str:
    data = json.dumps(
        event, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return f"id: {event['seq']}\nevent: search\ndata: {data}\n\n"


@dataclass
class _Channel:
    replay: deque[dict[str, Any]]
    next_seq: int = 1
    subscribers: set[asyncio.Queue] = field(default_factory=set)


@dataclass
class LiveSearchSubscription:
    """One subscriber's replay snapshot and future-event queue."""

    replay: tuple[dict[str, Any], ...]
    queue: asyncio.Queue
    _hub: "LiveSearchHub"
    _game_id: str
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hub.unsubscribe(self._game_id, self.queue)


class LiveSearchHub:
    """Per-game replay rings with bounded, non-blocking subscriber fan-out."""

    def __init__(
        self, *, replay_size: int = 64, subscriber_queue_size: int = 64,
        max_subscribers: int = 4,
    ) -> None:
        if replay_size < 1 or subscriber_queue_size < 1 or max_subscribers < 1:
            raise ValueError("live-search queue sizes and subscriber cap must be positive")
        self._replay_size = int(replay_size)
        self._subscriber_queue_size = int(subscriber_queue_size)
        self._max_subscribers = int(max_subscribers)
        self._channels: dict[str, _Channel] = {}

    def ensure(self, game_id: str) -> None:
        self._channels.setdefault(
            str(game_id), _Channel(replay=deque(maxlen=self._replay_size))
        )

    def begin_run(self, game_id: str) -> None:
        """Clear stale run replay/queues while preserving monotonic event ids."""
        game_id = str(game_id)
        self.ensure(game_id)
        channel = self._channels[game_id]
        channel.replay.clear()
        for queue in tuple(channel.subscribers):
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def publish(self, game_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append and fan out one event without ever awaiting a subscriber."""
        game_id = str(game_id)
        self.ensure(game_id)
        channel = self._channels[game_id]
        event = {**payload, "seq": channel.next_seq}
        channel.next_seq += 1
        channel.replay.append(event)
        for queue in tuple(channel.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Slow/disconnected viewers never back-pressure a search. Keep
                # the newest state; reconnect can recover the full bounded ring.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass
        return event

    def subscribe(
        self, game_id: str, *, after_seq: int = 0
    ) -> LiveSearchSubscription:
        game_id = str(game_id)
        self.ensure(game_id)
        channel = self._channels[game_id]
        if len(channel.subscribers) >= self._max_subscribers:
            raise LiveSearchSubscriberLimit(
                f"live-search viewer limit reached for game {game_id}"
            )
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._subscriber_queue_size)
        # This method contains no await: replay snapshot + registration is atomic
        # with respect to all loop-thread publishers.
        replay = tuple(event for event in channel.replay if event["seq"] > after_seq)
        channel.subscribers.add(queue)
        return LiveSearchSubscription(replay, queue, self, game_id)

    def unsubscribe(self, game_id: str, queue: asyncio.Queue) -> None:
        channel = self._channels.get(str(game_id))
        if channel is not None:
            channel.subscribers.discard(queue)

    def drop(self, game_id: str) -> None:
        """Forget replay state and wake every subscriber for one evicted game."""
        channel = self._channels.pop(str(game_id), None)
        if channel is None:
            return
        for queue in tuple(channel.subscribers):
            self._put_closed(queue)
        channel.subscribers.clear()

    def close_all(self) -> None:
        for game_id in tuple(self._channels):
            self.drop(game_id)

    @staticmethod
    def _put_closed(queue: asyncio.Queue) -> None:
        try:
            queue.put_nowait(_CLOSED)
            return
        except asyncio.QueueFull:
            pass
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(_CLOSED)
        except asyncio.QueueFull:
            pass


def subscription_closed(item: object) -> bool:
    return item is _CLOSED
