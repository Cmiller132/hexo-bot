"""Shared sample-buffer mechanics.

The sample layer stores model-owned training rows without interpreting model
payloads. Records are written as compressed JSON chunks under a store
directory, with a small manifest used to rebuild indexes and deterministic
training windows.

On-disk layout (spoken only by this module and its callers):
    <store>/manifest.json          - schema + chunk table + sample_count
    <store>/chunks/chunk-NNNNNN.*  - one compressed-JSON payload per append

Status (2026-06): LEGACY scaffolding. The only runtime caller is
`packages/hexo_train/python/hexo_train/epoch/samples.py` (the shared-store
path, skipped because every model plugin sets `uses_shared_sample_store=False`).
Production replay storage is model-owned NPZ (e.g. the hexfield trainer/shards).
"""

from __future__ import annotations

import json
import random
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .records import (
    SAMPLE_SCHEMA_VERSION,
    ModelSamplePayload,
    PolicyOutputRecord,
    SampleSchema,
    TrainingSampleRecord,
)


# --- Format constants -------------------------------------------------------

MANIFEST_FILENAME = "manifest.json"
CHUNKS_DIRNAME = "chunks"
CHUNK_SUFFIX = ".json.zlib"
# zlib is the default; "json" (uncompressed) is what the tests select. The
# zstd/lz4 options are wired but never selected by any config or test in the
# repo (grep 2026-06-12), even though pyproject.toml hard-requires both libs.
_CHUNK_SUFFIXES = {
    "json": ".json",
    "zlib": ".json.zlib",
    "zstd": ".json.zstd",
    "lz4": ".json.lz4",
}
_SAMPLE_TYPE_KEY = "_hexo_sample_type"
_TRAINING_SAMPLE_TYPE = "training_sample"
_POLICY_OUTPUT_TYPE = "policy_output"
_MODEL_PAYLOAD_TYPE = "model_payload"
_MAPPING_TYPE = "mapping"


# --- Frozen transport dataclasses (store / manifest / index / window) -------


@dataclass(frozen=True, slots=True)
class SampleStore:
    """Directory and metadata for trainable sample chunks."""

    path: Path
    mode: str = "append"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_FILENAME

    @property
    def chunks_path(self) -> Path:
        return self.path / CHUNKS_DIRNAME


@dataclass(frozen=True, slots=True)
class SampleChunkInfo:
    """Manifest entry for one compressed sample chunk."""

    chunk_id: str
    path: Path
    start: int
    count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def stop(self) -> int:
        return self.start + self.count


@dataclass(frozen=True, slots=True)
class SampleManifest:
    """Loaded manifest for a sample store."""

    store: SampleStore
    schema: SampleSchema
    sample_count: int
    chunks: tuple[SampleChunkInfo, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleWriteResult:
    """Summary returned after appending finalized records to a store."""

    count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    chunks: tuple[SampleChunkInfo, ...] = ()


@dataclass(frozen=True, slots=True)
class SampleIndexEntry:
    """Pointer to one record inside a compressed chunk."""

    sample_id: int
    chunk_id: str
    chunk_path: Path
    offset: int
    compression: str | None = None


@dataclass(frozen=True, slots=True)
class SampleIndex:
    """Searchable summary over finalized sample chunks."""

    store: SampleStore
    sample_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    manifest: SampleManifest | None = None
    entries: tuple[SampleIndexEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class SampleWindow:
    """A deterministic slice of indexed samples visible to one training pass."""

    index: SampleIndex
    window_size: int | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    entries: tuple[SampleIndexEntry, ...] = ()

    @property
    def sample_count(self) -> int:
        return len(self.entries)


@dataclass(frozen=True, slots=True)
class SampleRequest:
    """Request for sampled records from a reusable sample buffer."""

    count: int
    seed: int | None = None
    required_extensions: Sequence[str] = field(default_factory=tuple)
    filters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleBatch:
    """A sampled set of training rows plus provenance metadata."""

    records: Sequence[object]
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --- Public store API (open / append / index / window / sample / read) ------


def open_sample_store(
    path: str | Path,
    *,
    mode: str = "append",
    metadata: Mapping[str, Any] | None = None,
) -> SampleStore:
    """Open or create the directory that contains sample chunks."""

    store_path = Path(path)
    store_path.mkdir(parents=True, exist_ok=True)
    store = SampleStore(path=store_path, mode=mode, metadata=dict(metadata or {}))
    if mode != "read":
        store.chunks_path.mkdir(parents=True, exist_ok=True)
        if not store.manifest_path.exists():
            _write_manifest(
                store,
                {
                    "schema": _schema_to_json(
                        SampleSchema(extensions=dict(_extensions_from_metadata(store.metadata)))
                    ),
                    "sample_count": 0,
                    "chunks": [],
                    "metadata": _json_ready(store.metadata),
                },
            )
    return store


def append_samples(
    store: SampleStore,
    records: Sequence[object],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> SampleWriteResult:
    """Append finalized sample records to a compressed JSON chunk."""

    if store.mode == "read":
        raise ValueError("cannot append samples to a store opened in read mode")

    store.chunks_path.mkdir(parents=True, exist_ok=True)
    rows = tuple(records)
    write_metadata = dict(metadata or {})
    if not rows:
        return SampleWriteResult(count=0, metadata=write_metadata)

    manifest_data = _load_manifest_data(store)
    start = int(manifest_data.get("sample_count", 0))
    chunk_id = _next_chunk_id(store, manifest_data)
    compression = _compression_from_metadata({**dict(store.metadata), **write_metadata})
    chunk_path = store.chunks_path / f"{chunk_id}{_chunk_suffix(compression)}"
    encoded_records = [_record_to_json(record) for record in rows]
    schema = _schema_from_manifest_data(manifest_data)
    extensions = {
        **dict(schema.extensions),
        **_extensions_from_records(rows),
        **dict(_extensions_from_metadata(store.metadata)),
        **dict(_extensions_from_metadata(write_metadata)),
    }
    schema = SampleSchema(
        name=schema.name,
        version=schema.version,
        engine_version=schema.engine_version,
        extensions=extensions,
    )

    _write_compressed_json(
        chunk_path,
        {
            "schema": _schema_to_json(schema),
            "records": encoded_records,
            "metadata": _json_ready(write_metadata),
        },
        compression=compression,
    )

    raw_chunks = list(manifest_data.get("chunks", ()))
    raw_chunk = {
        "chunk_id": chunk_id,
        "path": f"{CHUNKS_DIRNAME}/{chunk_path.name}",
        "start": start,
        "count": len(rows),
        "metadata": _json_ready({**write_metadata, "compression": compression}),
    }
    raw_chunks.append(raw_chunk)
    manifest_data = {
        "schema": _schema_to_json(schema),
        "sample_count": start + len(rows),
        "chunks": raw_chunks,
        "metadata": _json_ready({**dict(store.metadata), **dict(manifest_data.get("metadata", {}))}),
    }
    _write_manifest(store, manifest_data)

    chunk_info = _chunk_info_from_json(store, raw_chunk)
    return SampleWriteResult(
        count=len(rows),
        chunks=(chunk_info,),
        metadata={
            **write_metadata,
            "chunk_id": chunk_id,
            "chunk_path": str(chunk_path),
            "compression": compression,
            "sample_count": start + len(rows),
        },
    )


def load_sample_manifest(store: SampleStore | str | Path) -> SampleManifest:
    """Load a store manifest, returning an empty manifest when none exists."""

    sample_store = _coerce_store(store)
    manifest_data = _load_manifest_data(sample_store)
    schema = _schema_from_manifest_data(manifest_data)
    chunks = tuple(
        _chunk_info_from_json(sample_store, item)
        for item in manifest_data.get("chunks", ())
    )
    sample_count = int(manifest_data.get("sample_count", sum(chunk.count for chunk in chunks)))
    return SampleManifest(
        store=sample_store,
        schema=schema,
        sample_count=sample_count,
        chunks=chunks,
        metadata=dict(manifest_data.get("metadata", {})),
    )


def refresh_sample_index(store: SampleStore) -> SampleIndex:
    """Refresh and return the searchable index for a sample store."""

    manifest = load_sample_manifest(store)
    entries = tuple(_index_entries(manifest))
    return SampleIndex(
        store=store,
        sample_count=manifest.sample_count,
        manifest=manifest,
        entries=entries,
        metadata={
            "chunk_count": len(manifest.chunks),
            "schema": manifest.schema.name,
            "schema_version": manifest.schema.version,
            "extensions": dict(manifest.schema.extensions),
        },
    )


def build_sample_window(
    index: SampleIndex,
    *,
    window_size: int | None = None,
    seed: int | None = None,
) -> SampleWindow:
    """Build the sample subset used by one epoch's training passes."""

    entries = _select_entries(index.entries, window_size=window_size, seed=seed)
    return SampleWindow(
        index=index,
        window_size=window_size,
        seed=seed,
        entries=entries,
        metadata={
            "requested_size": window_size,
            "selected_count": len(entries),
            "sample_count": index.sample_count,
        },
    )


def sample_training_samples(source: object, request: SampleRequest) -> SampleBatch:
    """Sample trainable records without constructing model tensors."""

    if request.count < 0:
        raise ValueError("sample request count must be non-negative")

    records = tuple(read_sample_records(source))
    records = tuple(record for record in records if _matches_request(record, request))
    selected = tuple(
        _project_record_extensions(record, request.required_extensions)
        for record in _select_records(records, count=request.count, seed=request.seed)
    )
    return SampleBatch(
        records=selected,
        metadata={
            "requested_count": request.count,
            "returned_count": len(selected),
            "source_count": len(records),
            "seed": request.seed,
            "required_extensions": tuple(request.required_extensions),
            "filters": dict(request.filters),
        },
    )


def read_sample_records(source: object) -> tuple[object, ...]:
    """Read all records addressed by a store, index, or window."""

    store, entries = _source_entries(source)
    if not entries:
        return ()

    by_chunk: dict[str, list[SampleIndexEntry]] = {}
    for entry in entries:
        by_chunk.setdefault(entry.chunk_id, []).append(entry)

    records: list[object] = []
    for chunk_id in sorted(by_chunk):
        chunk_entries = sorted(by_chunk[chunk_id], key=lambda item: item.offset)
        chunk_records = _read_chunk_records(
            chunk_entries[0].chunk_path,
            compression=chunk_entries[0].compression,
        )
        for entry in chunk_entries:
            try:
                records.append(chunk_records[entry.offset])
            except IndexError as exc:
                raise ValueError(
                    f"sample index entry {entry.sample_id} points past chunk {entry.chunk_id}"
                ) from exc
    # Vestigial: `store` is resolved by _source_entries for type dispatch but
    # never consumed here (chunk paths in the entries are already absolute).
    _ = store
    return tuple(records)


# UNUSED(2026-06-12): no references found in packages/tests/scripts (only the
# __init__.py re-export); callers use read_sample_records directly.
def iter_sample_records(source: object) -> Iterator[object]:
    """Iterate records addressed by a store, index, or window."""

    yield from read_sample_records(source)


# --- Private helpers: manifest + chunk JSON IO -------------------------------


def _coerce_store(store: SampleStore | str | Path) -> SampleStore:
    if isinstance(store, SampleStore):
        return store
    return SampleStore(path=Path(store), mode="read")


def _load_manifest_data(store: SampleStore) -> dict[str, Any]:
    if not store.manifest_path.exists():
        return {
            "schema": _schema_to_json(SampleSchema()),
            "sample_count": 0,
            "chunks": [],
            "metadata": {},
        }
    with store.manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"sample manifest is not an object: {store.manifest_path}")
    return dict(payload)


def _write_manifest(store: SampleStore, payload: Mapping[str, Any]) -> None:
    _write_json(store.manifest_path, payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    text = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _write_compressed_json(path: Path, payload: Mapping[str, Any], *, compression: str = "zlib") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    text = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    raw = text.encode("utf-8")
    if compression == "json":
        temp_path.write_bytes(raw)
    elif compression == "zlib":
        temp_path.write_bytes(zlib.compress(raw))
    elif compression == "zstd":
        try:
            import zstandard as zstd
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError("zstd sample compression requires the zstandard package") from exc
        temp_path.write_bytes(zstd.ZstdCompressor().compress(raw))
    elif compression == "lz4":
        try:
            import lz4.frame
        except ImportError as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError("lz4 sample compression requires the lz4 package") from exc
        temp_path.write_bytes(lz4.frame.compress(raw))
    else:
        raise ValueError(f"unsupported sample compression: {compression}")
    temp_path.replace(path)


def _read_compressed_json(path: Path, *, compression: str | None = None) -> Mapping[str, Any]:
    compression = _compression_from_path(path, fallback=compression)
    try:
        raw = path.read_bytes()
        if compression == "json":
            data = raw.decode("utf-8")
        elif compression == "zlib":
            data = zlib.decompress(raw).decode("utf-8")
        elif compression == "zstd":
            try:
                import zstandard as zstd
            except ImportError as exc:  # pragma: no cover - optional dependency.
                raise RuntimeError("zstd sample compression requires the zstandard package") from exc
            data = zstd.ZstdDecompressor().decompress(raw).decode("utf-8")
        elif compression == "lz4":
            try:
                import lz4.frame
            except ImportError as exc:  # pragma: no cover - optional dependency.
                raise RuntimeError("lz4 sample compression requires the lz4 package") from exc
            data = lz4.frame.decompress(raw).decode("utf-8")
        else:
            raise ValueError(f"unsupported sample compression for chunk: {path}")
    except zlib.error as exc:
        raise ValueError(f"sample chunk is not valid zlib data: {path}") from exc
    payload = json.loads(data)
    if not isinstance(payload, Mapping):
        raise ValueError(f"sample chunk is not a JSON object: {path}")
    return payload


def _read_chunk_records(path: Path, *, compression: str | None = None) -> tuple[object, ...]:
    payload = _read_compressed_json(path, compression=compression)
    rows = payload.get("records", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError(f"sample chunk records must be a sequence: {path}")
    return tuple(_record_from_json(row) for row in rows)


# --- Private helpers: schema / chunk-table / compression bookkeeping --------


def _schema_from_manifest_data(manifest_data: Mapping[str, Any]) -> SampleSchema:
    schema = manifest_data.get("schema", {})
    if not isinstance(schema, Mapping):
        schema = {}
    return SampleSchema(
        name=str(schema.get("name", "hexo.samples")),
        version=int(schema.get("version", SAMPLE_SCHEMA_VERSION)),
        engine_version=(
            str(schema["engine_version"])
            if schema.get("engine_version") is not None
            else None
        ),
        extensions={
            str(key): int(value)
            for key, value in dict(schema.get("extensions", {})).items()
        },
    )


def _schema_to_json(schema: SampleSchema) -> Mapping[str, Any]:
    return {
        "name": schema.name,
        "version": schema.version,
        "engine_version": schema.engine_version,
        "extensions": dict(schema.extensions),
    }


def _chunk_info_from_json(store: SampleStore, payload: Mapping[str, Any]) -> SampleChunkInfo:
    chunk_path = Path(str(payload["path"]))
    if not chunk_path.is_absolute():
        chunk_path = store.path / chunk_path
    return SampleChunkInfo(
        chunk_id=str(payload["chunk_id"]),
        path=chunk_path,
        start=int(payload.get("start", 0)),
        count=int(payload.get("count", 0)),
        metadata=dict(payload.get("metadata", {})),
    )


def _index_entries(manifest: SampleManifest) -> Iterable[SampleIndexEntry]:
    for chunk in manifest.chunks:
        for offset in range(chunk.count):
            yield SampleIndexEntry(
                sample_id=chunk.start + offset,
                chunk_id=chunk.chunk_id,
                chunk_path=chunk.path,
                offset=offset,
                compression=(
                    str(chunk.metadata["compression"])
                    if chunk.metadata.get("compression") is not None
                    else None
                ),
            )


def _next_chunk_id(store: SampleStore, manifest_data: Mapping[str, Any]) -> str:
    used_ids = {
        str(item.get("chunk_id"))
        for item in manifest_data.get("chunks", ())
        if isinstance(item, Mapping)
    }
    index = len(used_ids)
    while True:
        chunk_id = f"chunk-{index:06d}"
        if chunk_id not in used_ids and not any(
            (store.chunks_path / f"{chunk_id}{suffix}").exists()
            for suffix in _CHUNK_SUFFIXES.values()
        ):
            return chunk_id
        index += 1


def _compression_from_metadata(metadata: Mapping[str, Any]) -> str:
    compression = str(metadata.get("compression", "zlib")).lower()
    if compression in {"none", "raw"}:
        compression = "json"
    if compression not in _CHUNK_SUFFIXES:
        raise ValueError(f"unsupported sample compression: {compression}")
    return compression


def _chunk_suffix(compression: str) -> str:
    return _CHUNK_SUFFIXES[_compression_from_metadata({"compression": compression})]


def _compression_from_path(path: Path, *, fallback: str | None = None) -> str:
    name = path.name
    for compression, suffix in _CHUNK_SUFFIXES.items():
        if name.endswith(suffix):
            return compression
    if fallback is not None:
        return _compression_from_metadata({"compression": fallback})
    return "zlib"


# --- Private helpers: deterministic selection + request filtering -----------


def _select_entries(
    entries: Sequence[SampleIndexEntry],
    *,
    window_size: int | None,
    seed: int | None,
) -> tuple[SampleIndexEntry, ...]:
    available = tuple(entries)
    if window_size is None:
        return available
    size = max(0, min(int(window_size), len(available)))
    if size == len(available):
        return available
    indices = list(range(len(available)))
    if seed is None:
        selected = indices[:size]
    else:
        selected = random.Random(seed).sample(indices, size)
        selected.sort()
    return tuple(available[index] for index in selected)


def _select_records(records: Sequence[object], *, count: int, seed: int | None) -> tuple[object, ...]:
    available = tuple(records)
    size = max(0, min(int(count), len(available)))
    if size == len(available):
        return available
    if seed is None:
        return available[:size]
    selected = random.Random(seed).sample(range(len(available)), size)
    selected.sort()
    return tuple(available[index] for index in selected)


def _source_entries(source: object) -> tuple[SampleStore, tuple[SampleIndexEntry, ...]]:
    if isinstance(source, SampleWindow):
        return source.index.store, tuple(source.entries)
    if isinstance(source, SampleIndex):
        return source.store, tuple(source.entries)
    if isinstance(source, SampleStore):
        index = refresh_sample_index(source)
        return source, tuple(index.entries)
    raise TypeError(
        "sample source must be a SampleStore, SampleIndex, or SampleWindow"
    )


def _matches_request(record: object, request: SampleRequest) -> bool:
    if request.required_extensions and not _has_required_extensions(
        record, request.required_extensions
    ):
        return False
    for key, expected in request.filters.items():
        if _record_value(record, str(key)) != expected:
            return False
    return True


def _has_required_extensions(record: object, namespaces: Sequence[str]) -> bool:
    required = {str(namespace) for namespace in namespaces}
    if not required:
        return True
    payloads = getattr(record, "model_payloads", ())
    present = {
        str(getattr(payload, "namespace", ""))
        for payload in payloads
    }
    return required.issubset(present)


def _project_record_extensions(record: object, namespaces: Sequence[str]) -> object:
    requested = {str(namespace) for namespace in namespaces}
    if not requested or not isinstance(record, TrainingSampleRecord):
        return record
    return replace(
        record,
        model_payloads=tuple(
            payload
            for payload in record.model_payloads
            if payload.namespace in requested
        ),
    )


def _record_value(record: object, key: str) -> object:
    if key.startswith("metadata."):
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, Mapping):
            return metadata.get(key.removeprefix("metadata."))
        return None
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


# --- Private helpers: record (de)serialization to/from chunk JSON -----------


def _record_to_json(record: object) -> Mapping[str, Any]:
    if isinstance(record, TrainingSampleRecord):
        return {
            _SAMPLE_TYPE_KEY: _TRAINING_SAMPLE_TYPE,
            "game_id": record.game_id,
            "turn_index": record.turn_index,
            "legal_action_ids": list(record.legal_action_ids),
            "source_record_ref": _json_ready(record.source_record_ref),
            "policy": (
                _policy_to_json(record.policy)
                if record.policy is not None
                else None
            ),
            "model_payloads": [
                _model_payload_to_json(payload) for payload in record.model_payloads
            ],
            "metadata": _json_ready(record.metadata),
        }
    if isinstance(record, PolicyOutputRecord):
        return {
            _SAMPLE_TYPE_KEY: _POLICY_OUTPUT_TYPE,
            **_policy_to_json(record),
        }
    if isinstance(record, ModelSamplePayload):
        return {
            _SAMPLE_TYPE_KEY: _MODEL_PAYLOAD_TYPE,
            **_model_payload_to_json(record),
        }
    if isinstance(record, Mapping):
        return {
            _SAMPLE_TYPE_KEY: _MAPPING_TYPE,
            "data": _json_ready(dict(record)),
        }
    raise TypeError(
        f"sample records must be TrainingSampleRecord, ModelSamplePayload, "
        f"PolicyOutputRecord, or Mapping objects, got {type(record).__name__}"
    )


def _record_from_json(payload: object) -> object:
    if not isinstance(payload, Mapping):
        raise ValueError("sample record payload must be a JSON object")
    sample_type = payload.get(_SAMPLE_TYPE_KEY)
    if sample_type == _TRAINING_SAMPLE_TYPE:
        policy = payload.get("policy")
        return TrainingSampleRecord(
            game_id=str(payload["game_id"]),
            turn_index=int(payload["turn_index"]),
            legal_action_ids=tuple(int(item) for item in payload.get("legal_action_ids", ())),
            source_record_ref=payload.get("source_record_ref"),
            policy=(
                _policy_from_json(policy)
                if isinstance(policy, Mapping)
                else None
            ),
            model_payloads=tuple(
                _model_payload_from_json(item)
                for item in payload.get("model_payloads", ())
                if isinstance(item, Mapping)
            ),
            metadata=dict(payload.get("metadata", {})),
        )
    if sample_type == _POLICY_OUTPUT_TYPE:
        return _policy_from_json(payload)
    if sample_type == _MODEL_PAYLOAD_TYPE:
        return _model_payload_from_json(payload)
    if sample_type == _MAPPING_TYPE:
        data = payload.get("data", {})
        if not isinstance(data, Mapping):
            raise ValueError("mapping sample data must be a JSON object")
        return dict(data)
    return dict(payload)


def _policy_to_json(record: PolicyOutputRecord) -> Mapping[str, Any]:
    return {
        "game_id": record.game_id,
        "turn_index": record.turn_index,
        "model_id": record.model_id,
        "selected_action_id": record.selected_action_id,
        "logits": _json_ready(record.logits),
        "logits_ref": _json_ready(record.logits_ref),
        "value": record.value,
        "metadata": _json_ready(record.metadata),
    }


def _policy_from_json(payload: Mapping[str, Any]) -> PolicyOutputRecord:
    logits = payload.get("logits")
    if isinstance(logits, list):
        logits = tuple(logits)
    selected_action_id = payload.get("selected_action_id")
    return PolicyOutputRecord(
        game_id=str(payload["game_id"]),
        turn_index=int(payload["turn_index"]),
        model_id=str(payload["model_id"]),
        selected_action_id=(
            int(selected_action_id)
            if selected_action_id is not None
            else None
        ),
        logits=logits,
        logits_ref=payload.get("logits_ref"),
        value=float(payload["value"]) if payload.get("value") is not None else None,
        metadata=dict(payload.get("metadata", {})),
    )


def _model_payload_to_json(record: ModelSamplePayload) -> Mapping[str, Any]:
    return {
        "game_id": record.game_id,
        "turn_index": record.turn_index,
        "model_id": record.model_id,
        "namespace": record.namespace,
        "schema_version": record.schema_version,
        "payload": _json_ready(record.payload),
        "payload_ref": _json_ready(record.payload_ref),
    }


def _model_payload_from_json(payload: Mapping[str, Any]) -> ModelSamplePayload:
    return ModelSamplePayload(
        game_id=str(payload["game_id"]),
        turn_index=int(payload["turn_index"]),
        model_id=str(payload["model_id"]),
        namespace=str(payload["namespace"]),
        schema_version=int(payload["schema_version"]),
        payload=dict(payload.get("payload", {})),
        payload_ref=payload.get("payload_ref"),
    )


def _extensions_from_records(records: Sequence[object]) -> Mapping[str, int]:
    extensions: dict[str, int] = {}
    for record in records:
        payloads = getattr(record, "model_payloads", ())
        if isinstance(record, ModelSamplePayload):
            payloads = (record,)
        for payload in payloads:
            namespace = str(getattr(payload, "namespace", ""))
            if namespace:
                extensions[namespace] = int(getattr(payload, "schema_version", 1))
    return extensions


def _extensions_from_metadata(metadata: Mapping[str, Any]) -> Mapping[str, int]:
    extensions = metadata.get("extensions") if isinstance(metadata, Mapping) else None
    if not isinstance(extensions, Mapping):
        return {}
    return {str(key): int(value) for key, value in extensions.items()}


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    if is_dataclass(value):
        return {
            item.name: _json_ready(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    raise TypeError(f"value is not JSON-serializable by the sample store: {value!r}")
