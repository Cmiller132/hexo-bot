"""Training component contracts.

The self-play training pipeline has three layers:

1. `RunContext` owns run identity, directories, diagnostics, and epoch outputs.
2. `SharedComponents` are reusable defaults created by `hexo_train`.
3. `ModelComponents` are supplied or overridden by the selected model plugin.

Epoch helpers receive `RunContext` plus `TrainingComponents`. This keeps the
loop sequence centralized while still letting models own tensors, losses,
decoders, and checkpoint contents.

Read this file as the dependency container for a run. It defines what shared
orchestration code can rely on and what model plugins may replace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .context import RunContext


@dataclass(slots=True)
class DefaultTrainingComponents:
    """Shared defaults a model may accept or override.

    Defaults are deliberately boring and model-neutral: target helpers,
    checkpoint path helpers, diagnostics, and deterministic symmetry selection.
    """

    scalar_value_target: Any | None = None
    legal_policy_target: Any | None = None
    symmetry_selector: Any | None = None
    checkpoint_store: Any | None = None
    diagnostics: Any | None = None


@dataclass(slots=True)
class SharedComponents:
    """Model-neutral handles built by `hexo_train`.

    The pipeline mutates these fields as the run advances. For example, sample
    selection writes `sample_window`, symmetry selection writes
    `sample_symmetries`, and checkpoint saves update `checkpoint_state`.
    """

    defaults: DefaultTrainingComponents
    sample_store: Any | None = None
    sample_index: Any | None = None
    sample_window: Any | None = None
    sample_symmetries: Any | None = None
    checkpoint_state: Any | None = None
    selfplay_result: Any | None = None
    game_spec: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ComponentOverrides:
    """Optional replacements returned by a model training plugin.

    The plugin returns only the pieces it owns or wants to replace. Missing
    fields fall back to shared defaults where a default exists.

    This dataclass is the one symbol model packages import from hexo_train
    (see packages/hexfield/python/hexfield/plugin.py and the other model
    plugin.py twins). The real plugins set `uses_shared_sample_store=False`
    and supply trainer + checkpoint loader/saver, so the shared-store and
    placeholder-checkpoint defaults are bypassed on every production run.
    """

    scalar_value_target: Any | None = None
    legal_policy_target: Any | None = None
    sample_decoder: Any | None = None
    sample_finalizer: Any | None = None
    symmetry_selector: Any | None = None
    trainer: Any | None = None
    optimizer: Any | None = None
    checkpoint_loader: Any | None = None
    checkpoint_saver: Any | None = None
    uses_shared_sample_store: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelComponents:
    """Model-owned training pieces used by the epoch loop.

    These fields are the resolved plugin contract after defaults and overrides
    are merged. Epoch helpers should call these components instead of importing
    concrete model packages.
    """

    plugin: Any
    model: Any | None = None
    optimizer: Any | None = None
    trainer: Any | None = None
    decoder: Any | None = None
    sample_finalizer: Any | None = None
    symmetry_selector: Any | None = None
    checkpoint_loader: Any | None = None
    checkpoint_saver: Any | None = None
    scalar_value_target: Any | None = None
    legal_policy_target: Any | None = None
    uses_shared_sample_store: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingComponents:
    """All components available to one pipeline run."""

    shared: SharedComponents
    model: ModelComponents


def build_model_components(
    *,
    plugin: Any,
    ctx: RunContext,
    shared: SharedComponents,
) -> ModelComponents:
    """Build model components by merging shared defaults with plugin overrides.

    Step by step:

    1. Ask the plugin to build the model object, if it has a builder.
    2. Pass the built model plus shared defaults back to the plugin.
    3. Coerce the plugin response into `ComponentOverrides`.
    4. Return a single `ModelComponents` object for the epoch loop to use.
    """

    defaults = shared.defaults
    overrides = ComponentOverrides()

    model = None
    if hasattr(plugin, "build_model"):
        model = plugin.build_model(shared.game_spec, ctx.config.model.config)

    if hasattr(plugin, "training_component_overrides"):
        raw_overrides = plugin.training_component_overrides(
            defaults=defaults,
            config=ctx.config.model.config,
            shared=shared,
            model=model,
        )
        overrides = _coerce_overrides(raw_overrides)

    return ModelComponents(
        plugin=plugin,
        model=model,
        optimizer=overrides.optimizer,
        trainer=overrides.trainer,
        decoder=overrides.sample_decoder,
        sample_finalizer=overrides.sample_finalizer,
        symmetry_selector=overrides.symmetry_selector
        or defaults.symmetry_selector,
        checkpoint_loader=overrides.checkpoint_loader,
        checkpoint_saver=overrides.checkpoint_saver,
        scalar_value_target=overrides.scalar_value_target
        or defaults.scalar_value_target,
        legal_policy_target=overrides.legal_policy_target
        or defaults.legal_policy_target,
        uses_shared_sample_store=bool(overrides.uses_shared_sample_store),
        extra=overrides.extra,
    )


def _coerce_overrides(raw: Any) -> ComponentOverrides:
    """Accept the supported plugin override shapes and reject ambiguous ones."""

    if raw is None:
        return ComponentOverrides()
    if isinstance(raw, ComponentOverrides):
        return raw
    if isinstance(raw, Mapping):
        return ComponentOverrides(**raw)
    raise TypeError(
        "training_component_overrides() must return ComponentOverrides, "
        "a mapping, or None."
    )
