"""Dynamic model plugin loading.

Model packages provide the model-specific pieces: architecture construction,
sample decoding, losses, checkpoint interpretation, and epoch-time behavior.
`hexo_train` only finds the plugin and calls agreed-upon lifecycle methods.

Plugin lookup supports three development/deployment modes:

1. explicit Python module path from config;
2. explicit entry point name from config;
3. model name lookup through the `hexo_train.models` entry point group.

Registered plugins (entry-point group `hexo_train.models`) include
`hexfield_eq` (packages/hexfield_eq/pyproject.toml) alongside the parked
`dense_cnn_restnet`, `dense_cnn`/`hexgt`, and `hexgnn` plugins. In practice
the active bots select their plugin via mode 1 (explicit module path):
`[model].module = "hexfield.plugin"` in configs/hexfield_main_9.toml and
`"hexfield_eq.plugin"` in configs/hexfield_eq_main_1.toml, bypassing entry
points entirely.

Note: `ModelPlugin` covers only the two construction hooks. The full
duck-typed contract a real plugin/trainer implements is wider — optional
`generate_selfplay`, `evaluate_epoch`, `calibrate_performance` hooks plus the
trainer's `select_training_samples`/`train_passes`/`close()` — dispatched by
hasattr checks in pipeline.py and epoch/*.py rather than this Protocol.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from .config import ModelConfig
from .components import ComponentOverrides, DefaultTrainingComponents, SharedComponents


@runtime_checkable
class ModelPlugin(Protocol):
    """Minimum plugin shape expected by the training orchestrator.

    Plugins can expose more optional hooks, such as self-play generation or
    checkpoint IO, but the pipeline can always start from this small contract.
    """

    name: str

    def build_model(self, game_spec: Any, config: Any) -> Any:
        """Build and return the model object."""

    def training_component_overrides(
        self,
        *,
        defaults: DefaultTrainingComponents,
        config: Any,
        shared: SharedComponents,
        model: Any,
    ) -> ComponentOverrides | None:
        """Return only the default components this model wants to replace."""


def load_model_plugin(config: ModelConfig) -> ModelPlugin:
    """Load a model plugin by explicit module, entry point, or plugin name."""

    if config.module:
        return _load_from_module(config.module)
    if config.entry_point:
        return _load_from_entry_point(config.entry_point)
    return _load_by_name(config.name)


def _load_from_module(module_name: str) -> ModelPlugin:
    """Import a development module and read its plugin object."""

    module = import_module(module_name)
    if hasattr(module, "get_plugin"):
        return module.get_plugin()
    if hasattr(module, "plugin"):
        return module.plugin
    raise AttributeError(
        f"Model module {module_name!r} must expose get_plugin() or plugin."
    )


def _load_from_entry_point(entry_point_name: str) -> ModelPlugin:
    """Load a plugin from the supported Python entry point groups."""

    for group in _entry_point_groups():
        for entry_point in entry_points(group=group):
            if entry_point.name == entry_point_name:
                return _coerce_loaded_plugin(entry_point.load())
    raise LookupError(f"No Hexo model entry point named {entry_point_name!r}.")


def _load_by_name(model_name: str) -> ModelPlugin:
    """Resolve a model name through installed entry points."""

    for group in _entry_point_groups():
        for entry_point in entry_points(group=group):
            if entry_point.name == model_name:
                return _coerce_loaded_plugin(entry_point.load())

    raise LookupError(f"No Hexo model entry point named {model_name!r}.")


def _entry_point_groups() -> tuple[str, ...]:
    """Return the entry point groups that may provide Hexo training models."""

    return (
        "hexo_train.models",
    )


def _coerce_loaded_plugin(loaded: Any) -> ModelPlugin:
    """Accept either a plugin instance or a callable that returns one."""

    return loaded() if callable(loaded) else loaded
