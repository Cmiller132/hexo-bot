"""Training config loading and normalization.

This file is the boundary between user-authored YAML/TOML and the rest of the
training code. Everything outside this module should read typed config objects
instead of pulling values directly out of nested dictionaries. In practice
every config in configs/ is TOML (e.g. configs/hexfield_main_9.toml, the live
run); the YAML path is advertised by the CLI but has no caller.

The typed sections here cover only the orchestration skeleton. Model-owned
settings ride through opaquely as `ModelConfig.config` ([model.config] in the
TOML) and are parsed by the plugin's own config module (e.g.
packages/hexfield/python/hexfield/config.py); model-neutral
extras like [shared.game] stay reachable via `TrainingConfig.raw` /
`RunContext.section()`.

The normalized config mirrors the fixed self-play training path:

1. `run` names the run and output location.
2. `model` selects the plugin and carries model-owned config.
3. `loop` decides how many self-play epochs to run.
4. `selfplay`, `samples`, and `train` decide how much work happens per epoch.
5. `checkpoint` decides how the run starts and what final checkpoint is named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import tomllib


ConfigMap = Mapping[str, Any]


# --- Typed config sections (one frozen dataclass per top-level TOML table) ---


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model plugin selection plus opaque model-owned config.

    `hexo_train` uses `name`, `module`, and `entry_point` only to find the
    plugin. The nested `config` mapping is passed through to the plugin without
    interpreting model architecture, tensor, or optimizer semantics.
    """

    name: str
    module: str | None = None
    entry_point: str | None = None
    config: ConfigMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Shared run identity and output locations.

    `output_dir` is resolved during normalization so later code can treat it as
    an absolute or config-relative concrete path.
    """

    name: str = "hexo_train_run"
    output_dir: Path = Path("runs/hexo_train_run")
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """Self-play training loop settings.

    Each epoch includes self-play generation and a training pass sequence.
    """

    epochs: int = 1


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """Self-play generation settings for each epoch.

    `games_per_epoch` is passed into the model/runner self-play hook. Pointer
    settings control whether the final checkpoint path is published for future
    self-play jobs.
    """

    games_per_epoch: int = 1
    update_checkpoint_pointer: bool = False
    checkpoint_pointer: Path | None = None


@dataclass(frozen=True, slots=True)
class SamplesConfig:
    """Sample selection settings for each training epoch.

    `train_sample_count=None` means "use whatever the current sample index
    exposes"; an integer requests a bounded training window.
    """

    train_sample_count: int | None = None


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Training-pass settings for each epoch.

    A pass means one model-owned sweep over the selected sample window. The
    trainer decides what a pass means mechanically, such as batches or steps.
    """

    passes_per_epoch: int = 1


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint load/save settings.

    `resume_from` and `initialize_from` describe the initial model state.
    `save_name` names the final checkpoint after all epochs complete.
    """

    resume_from: Path | None = None
    initialize_from: Path | None = None
    save_name: str = "latest"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Fully normalized config consumed by `TrainingPipeline`.

    `raw` is retained only for transitional access to sections that are still
    intentionally flexible, such as model-neutral shared settings.
    """

    model: ModelConfig
    run: RunConfig = field(default_factory=RunConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    samples: SamplesConfig = field(default_factory=SamplesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    shared: ConfigMap = field(default_factory=dict)
    raw: ConfigMap = field(default_factory=dict)


# --- Loading and normalization (file -> raw mapping -> TrainingConfig) ---


def load_training_config(config_path: str | Path) -> TrainingConfig:
    """Load a YAML or TOML file and return a normalized training config."""

    path = Path(config_path)
    raw = _load_raw_config(path)
    return normalize_training_config(raw, base_dir=path.parent)


def normalize_training_config(raw: ConfigMap, *, base_dir: Path) -> TrainingConfig:
    """Convert an untyped config mapping into the self-play config contract.

    The function validates removed fields first, then builds each typed config
    section. Path values are resolved relative to `base_dir`, which is the
    directory containing the config file.
    """

    if "model_specific" in raw:
        raise ValueError(
            "Training config field 'model_specific' was removed; "
            "move model-owned settings under [model.config]."
        )
    if "stages" in raw:
        raise ValueError(
            "Training config field 'stages' was removed; "
            "self-play epoch training is the only supported path."
        )

    model_section = _require_mapping(raw, "model")
    model_name = str(model_section.get("name", "")).strip()
    if not model_name:
        raise ValueError("Training config must define model.name.")

    run_section = dict(raw.get("run", {}))
    run_name = str(run_section.get("name", f"{model_name}_train"))
    output_dir = Path(run_section.get("output_dir", Path("runs") / run_name))
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir = output_dir.resolve()

    model_config = model_section.get("config", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("Training config field [model.config] must be a mapping.")

    model = ModelConfig(
        name=model_name,
        module=_optional_str(model_section.get("module")),
        entry_point=_optional_str(model_section.get("entry_point")),
        config=dict(model_config),
    )
    run = RunConfig(
        name=run_name,
        output_dir=output_dir,
        seed=_optional_int(run_section.get("seed")),
    )
    loop = LoopConfig(
        epochs=_positive_int(_optional_mapping(raw, "loop"), "epochs", default=1),
    )
    selfplay_section = _optional_mapping(raw, "selfplay")
    checkpoint_pointer = _optional_path(
        selfplay_section.get("checkpoint_pointer"),
        base_dir=base_dir,
    )
    selfplay = SelfPlayConfig(
        games_per_epoch=_non_negative_int(
            selfplay_section,
            "games_per_epoch",
            default=1,
        ),
        update_checkpoint_pointer=bool(
            selfplay_section.get("update_checkpoint_pointer", False)
        ),
        checkpoint_pointer=checkpoint_pointer,
    )
    samples = SamplesConfig(
        train_sample_count=_optional_positive_int(
            _optional_mapping(raw, "samples"),
            "train_sample_count",
        ),
    )
    train = TrainConfig(
        passes_per_epoch=_positive_int(
            _optional_mapping(raw, "train"),
            "passes_per_epoch",
            default=1,
        ),
    )
    checkpoint_section = _optional_mapping(raw, "checkpoint")
    checkpoint = CheckpointConfig(
        resume_from=_optional_path(checkpoint_section.get("resume_from"), base_dir=base_dir),
        initialize_from=_optional_path(checkpoint_section.get("initialize_from"), base_dir=base_dir),
        save_name=str(checkpoint_section.get("save_name", "latest")),
    )

    return TrainingConfig(
        model=model,
        run=run,
        loop=loop,
        selfplay=selfplay,
        samples=samples,
        train=train,
        checkpoint=checkpoint,
        shared=dict(raw.get("shared", {})),
        raw=dict(raw),
    )


def _load_raw_config(path: Path) -> ConfigMap:
    """Read the config file format without applying project semantics."""

    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as handle:
            return tomllib.load(handle)
    # UNUSED(2026-06-12): no .yaml/.yml training config exists anywhere in
    # configs/, tests/, or scripts/ (repo-wide grep); every launcher and
    # generated smoke config emits TOML. Kept because the CLI advertises YAML
    # support and PyYAML is declared in pyproject.toml for this branch.
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on environment.
            raise RuntimeError("YAML training configs require PyYAML.") from exc
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("YAML training config must load to a mapping.")
        return loaded
    raise ValueError(f"Unsupported training config format: {path.suffix}")


# --- Scalar/section validation helpers ---


def _require_mapping(raw: ConfigMap, key: str) -> ConfigMap:
    """Return a required section or raise a user-facing validation error."""

    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Training config must define a [{key}] mapping.")
    return value


def _optional_mapping(raw: ConfigMap, key: str) -> ConfigMap:
    """Return an optional mapping section, treating missing/None as empty."""

    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Training config field [{key}] must be a mapping.")
    return value


def _optional_str(value: object) -> str | None:
    """Normalize optional scalar config values into stripped strings."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    """Normalize optional integer values while preserving omitted values."""

    if value is None or value == "":
        return None
    return int(value)


def _positive_int(section: ConfigMap, key: str, *, default: int) -> int:
    """Read an integer config value that must be at least one."""

    value = int(section.get(key, default))
    if value < 1:
        raise ValueError(f"Training config field {key!r} must be >= 1.")
    return value


def _non_negative_int(section: ConfigMap, key: str, *, default: int) -> int:
    """Read an integer config value that must be zero or greater."""

    value = int(section.get(key, default))
    if value < 0:
        raise ValueError(f"Training config field {key!r} must be >= 0.")
    return value


def _optional_positive_int(section: ConfigMap, key: str) -> int | None:
    """Read an optional positive integer, returning None when omitted."""

    value = section.get(key)
    if value is None or value == "":
        return None
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"Training config field {key!r} must be >= 1.")
    return resolved


def _optional_path(value: object, *, base_dir: Path) -> Path | None:
    """Resolve an optional path relative to the config file directory."""

    text = _optional_str(value)
    if text is None:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
