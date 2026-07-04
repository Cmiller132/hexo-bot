"""Dynamic model plugin loading.

Model packages provide the model-specific pieces: architecture construction,
sample decoding, losses, checkpoint interpretation, and epoch-time behavior.
`hexo_train` only finds the plugin and calls agreed-upon lifecycle methods.

Plugin lookup uses an explicit Python module path from config: the shipped
configs set `[model].module = "hexfield.plugin"`, and the module exposes either
a `get_plugin()` factory or a `plugin` object.

Note: `ModelPlugin` covers only the two construction hooks. The full
duck-typed contract a real plugin/trainer implements is wider — optional
`generate_selfplay`, `evaluate_epoch`, `calibrate_performance` hooks plus the
trainer's `select_training_samples`/`train_passes`/`close()` — dispatched by
hasattr checks in pipeline.py and epoch/*.py rather than this Protocol.
"""

from __future__ import annotations

from importlib import import_module
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
    """Load a model plugin from its explicit config module path."""

    if not config.module:
        raise ValueError("Training config must define [model].module.")
    return _load_from_module(config.module)


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
