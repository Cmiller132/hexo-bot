"""Training sample schema and record shapes.

Models write these records during self-play. Core game records remain in
`hexo_runner.records` for detached analysis and audit; samples may keep optional
references back to those records, but they are already training-facing data.

The record layer is intentionally data-only. It defines the shared schema
version and the neutral row shapes that model packages may use or extend, while
buffer storage and sampling mechanics live in `buffer.py`.

Status (2026-06): LEGACY scaffolding with `buffer.py` -- no production writer
exists (every model plugin opts out of the shared sample store and persists
its own compact NPZ rows instead, e.g.
`packages/hexfield/python/hexfield/samples.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SAMPLE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SampleSchema:
    """Version metadata attached to sample files and batches.

    `extensions` lets model packages declare their own payload namespaces
    without making the shared utils package understand those payload schemas.
    """

    name: str = "hexo.samples"
    version: int = SAMPLE_SCHEMA_VERSION
    engine_version: str | None = None
    extensions: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyOutputRecord:
    """Common policy output over the legal actions offered by the engine.

    The parent `TrainingSampleRecord.legal_action_ids` defines the order of
    `logits`. Model packages can turn that compact vector into their own target
    tensors. More complex policy heads, pair policies, search traces, or
    architecture-specific data should be stored as model payload records
    instead of being modeled here. Large arrays may be represented by
    references rather than kept resident in RAM.
    """

    game_id: str
    turn_index: int
    model_id: str
    selected_action_id: int | None = None
    logits: object | None = None
    logits_ref: object | None = None
    value: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelSamplePayload:
    """Opaque model-owned payload attached to a training sample.

    The shared samples layer only knows the payload namespace and version.
    The model package owns the payload schema and parsing code. Large extension
    payloads may be stored out-of-line behind `payload_ref`.
    """

    game_id: str
    turn_index: int
    model_id: str
    namespace: str
    schema_version: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    payload_ref: object | None = None


@dataclass(frozen=True, slots=True)
class TrainingSampleRecord:
    """Training-facing record for one sampled position/decision.

    `source_record_ref` may point at the runner's detached core position record
    for debugging or audit. Training should not depend on scanning runner
    records; models write the sample contents they need during self-play.
    """

    game_id: str
    turn_index: int
    legal_action_ids: tuple[int, ...]
    source_record_ref: object | None = None
    policy: PolicyOutputRecord | None = None
    model_payloads: Sequence[ModelSamplePayload] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
