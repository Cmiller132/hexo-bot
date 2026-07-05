"""CPU-only inference for the dashboard Debug tab (shrimp lineage).

This module is the *inference library* behind the Debug tab. It loads a training
checkpoint, reconstructs a board position from a move sequence, and returns what
the model "thinks" — per-candidate policy prior, the distributional value head
(+ scalar), auxiliary heads (opponent-policy / short-term-value / moves-left),
and (on demand) a fresh CPU MCTS visit distribution.

Checkpoints are shrimp-lineage: the payload carries the model state dict under
``payload["model"]`` and a ``payload["meta"]`` block whose ``lineage`` field is
``"shrimp"`` (see ``_detect_lineage``). The architecture is inferred from the
state dict, the support-set featurizer is used, and a uniform debug-output schema
is returned (heads the checkpoint does not have are returned as ``None`` so the
UI marks them N/A).

Everything here is **CPU-only by construction**: models are built and run on
``torch.device("cpu")`` and the MCTS evaluator is constructed with ``device=
"cpu"`` (no AMP). The worker process that imports this module is also launched
with ``CUDA_VISIBLE_DEVICES=""`` so it can never touch the training GPU.

The heavy shrimp import is lazy (loaded only when a checkpoint is opened) so
the worker's ``ping`` never pays an import it does not need.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

import hexo_engine as engine
from hexo_engine.types import unpack_coord_id

# Lineage tag (the only lineage this module serves).
SHRIMP = "shrimp"


# ---------------------------------------------------------------------------
# Loaded-model container + lineage detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedModel:
    """A CPU model ready for debug inference, plus provenance metadata.

    Some fields (``candidate_radius``/``graft``/``expanded_*``/
    ``zeroed_feature_cols``) are legacy neutral defaults retained so the worker's
    metadata view stays a stable shape; the shrimp loader leaves them at their
    defaults.
    """

    lineage: str
    model: Any
    arch: dict[str, Any]
    rl_epoch: int | None
    step: int | None
    candidate_radius: int | None = None
    graft: str | None = None
    expanded_value: bool = False
    expanded_stv: list[str] = field(default_factory=list)
    zeroed_feature_cols: list[int] = field(default_factory=list)
    load_warnings: list[str] = field(default_factory=list)
    stv_horizons: tuple[int, ...] = ()
    has_moves_left: bool = False
    # True when the checkpoint's state dict carries the train-only per-cell Q
    # head (``cell_q_head.weight``). Detected from the state dict, NOT from the
    # forward output: a bare ``ShrimpNet()`` ALWAYS has the head module, so
    # output presence is not a valid detector.
    has_cell_q: bool = False
    # Run root (the checkpoint's grandparent dir) — the shrimp search reads
    # the run's manifest.json from here to search with the AS-TRAINED profile
    # (gumbel flags, calibrated gumbel_m, divergences).
    run_dir: str | None = None


def _detect_lineage(payload: Any) -> str:
    """Validate a checkpoint payload as the supported shrimp lineage.

    A shrimp checkpoint stores the model state dict under ``payload["model"]``
    and carries a ``payload["meta"]`` block whose ``lineage`` is ``"shrimp"``.
    Anything else raises with a clear message naming the supported lineage.
    """

    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a dict")
    meta = payload.get("meta")
    if isinstance(meta, dict) and meta.get("lineage") == "shrimp" and isinstance(payload.get("model"), dict):
        return SHRIMP
    raise ValueError(
        "unsupported checkpoint: this build only serves the 'shrimp' lineage "
        "(expected payload['meta']['lineage'] == 'shrimp' with a state dict "
        "under payload['model'])"
    )


def load_checkpoint(path: str | Path) -> LoadedModel:
    """Load a shrimp checkpoint onto CPU."""

    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _detect_lineage(payload)
    return _load_shrimp_checkpoint(ckpt_path, payload)


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Shared position reconstruction
# ---------------------------------------------------------------------------


def state_from_actions(action_ids: Sequence[int]):
    """Replay a move sequence into a fresh engine state (CPU, no model)."""

    state = engine.new_game()
    for aid in action_ids:
        engine.apply_action(state, engine.PlacementAction(unpack_coord_id(int(aid))))
    return state


def _coord_of(action_id: int) -> dict[str, int]:
    coord = unpack_coord_id(int(action_id))
    return {"q": int(coord.q), "r": int(coord.r)}


def _policy_pairs_to_rows(pairs: Sequence[tuple[int, float]], *, normalize: bool) -> list[dict[str, Any]]:
    items = [(int(a), float(w)) for a, w in pairs]
    total = sum(w for _, w in items) if normalize else 0.0
    rows = []
    for aid, w in items:
        coord = _coord_of(aid)
        p = (w / total) if (normalize and total > 0) else w
        rows.append({"action_id": aid, "q": coord["q"], "r": coord["r"], "p": round(float(p), 6), "w": round(float(w), 4)})
    rows.sort(key=lambda r: r["w"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Top-level dispatch (called by the worker)
# ---------------------------------------------------------------------------


@torch.no_grad()
def analyze_position(
    loaded: LoadedModel,
    action_ids: Sequence[int],
    *,
    n: int | None = None,
    planes: bool = False,
) -> dict[str, Any]:
    """Full-head readout for the position reached by ``action_ids``.

    ``n``/``planes`` are accepted for call-site compatibility; the shrimp
    readout uses neither."""

    return _analyze_shrimp(loaded, action_ids)


@torch.no_grad()
def search_position(
    loaded: LoadedModel,
    action_ids: Sequence[int],
    *,
    visits: int = 512,
    c_puct: float = 1.5,
    n: int | None = None,
    seed: int = 0,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run a fresh, reproducible CPU MCTS on the position (no root noise).

    ``temperature`` is the in-search move-selection temperature for the
    as-trained (eval-protocol) profile — 0 keeps the greedy debug read; match
    bots pass 1.0 for opening plies. ``n`` is accepted for call-site
    compatibility and unused."""

    return _search_shrimp(
        loaded, action_ids, visits=visits, c_puct=c_puct, seed=seed,
        temperature=temperature,
    )


# ===========================================================================
# shrimp lineage — support-set (variable-N) graph featurizer + node tokens
# ===========================================================================


@lru_cache(maxsize=1)
def _shrimp() -> SimpleNamespace:
    """Lazily import the shrimp package modules (kept out of import-time so the
    worker's ``ping`` never pays the import, and so an env without torch/shrimp
    can still load this module)."""

    # shrimp is NEVER installed into a shared venv (spec §5.1) — it is imported
    # via a source-path shim. The debug worker is spawned with a minimal
    # PYTHONPATH that does not include it, so add packages/shrimp/python here
    # (derived from this file's location) before importing. Without this the
    # worker fails with ModuleNotFoundError: No module named 'shrimp'.
    import sys

    _hf_src = Path(__file__).resolve().parents[3] / "shrimp" / "python"
    if _hf_src.is_dir() and str(_hf_src) not in sys.path:
        sys.path.insert(0, str(_hf_src))

    from shrimp import _rust
    from shrimp.batching import collate_rows
    from shrimp.constants import VALUE_BINS
    from shrimp.engine_facts import facts_from_state
    from shrimp.features import build_features
    from shrimp.geometry import pack_action_id
    from shrimp.losses import decode_binned_value, decode_moves_left, value_bins
    from shrimp.model import STV_HORIZONS, ShrimpNet
    from shrimp.support import build_support

    return SimpleNamespace(
        _rust=_rust,
        collate_rows=collate_rows,
        VALUE_BINS=VALUE_BINS,
        facts_from_state=facts_from_state,
        build_features=build_features,
        pack_action_id=pack_action_id,
        decode_binned_value=decode_binned_value,
        decode_moves_left=decode_moves_left,
        value_bins=value_bins,
        STV_HORIZONS=STV_HORIZONS,
        ShrimpNet=ShrimpNet,
        build_support=build_support,
    )


def _infer_shrimp_channels(state_dict: dict[str, Any]) -> int | None:
    """Read the trunk width (channels) off a shrimp state dict.

    ``stem.bias`` is a length-``channels`` vector and ``tokens`` is
    ``(NUM_TOKENS, channels)``; either pins the width without consulting the
    process-global CHANNELS, so a c=96 worker can still load a c=128 run.
    Returns None when neither key is present (caller falls back to the default).
    """

    for key, axis in (("stem.bias", 0), ("stem_ln.weight", 0), ("tokens", 1)):
        tensor = state_dict.get(key)
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) > axis:
            return int(shape[axis])
    return None


# Trunk layout by (conv_blocks, attn_blocks) count. The counts alone do not pin
# the C/A interleaving, so known layouts are mapped explicitly: (8 conv, 3 attn)
# is CCC A CCC A CC A; (10 conv, 5 attn) is CC A x5. A new layout must be added
# here before its checkpoints can be debugged (the loader raises a clear error
# instead of mis-building the net).
_SHRIMP_TRUNK_LAYOUTS: dict[tuple[int, int], str] = {
    (8, 3): "CCCACCCACCA",
    (10, 5): "CCACCACCACCACCA",
}


def _infer_shrimp_arch(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Infer (channels, attention_heads, trunk_layout) off a shrimp state dict.

    All three are env-driven at training time (SHRIMP_CHANNELS /
    SHRIMP_ATTENTION_HEADS / SHRIMP_TRUNK), so the worker — whose process
    env is NOT the run's env — must reconstruct them from the weights:
    channels from stem.bias, heads from the bias-table column count
    (``bias_tables.0`` is (BIAS_ROWS, heads)), block counts from the parameter
    key indices, and the C/A interleaving from the known-lineage layout map."""

    channels = _infer_shrimp_channels(state_dict)
    heads = None
    bt = state_dict.get("bias_tables.0")
    if bt is not None and len(getattr(bt, "shape", ())) == 2:
        heads = int(bt.shape[1])
    conv_ids = {
        int(k.split(".")[1]) for k in state_dict if k.startswith("conv_blocks.")
    }
    attn_ids = {
        int(k.split(".")[1]) for k in state_dict if k.startswith("attn_blocks.")
    }
    layout = None
    if conv_ids and attn_ids:
        counts = (max(conv_ids) + 1, max(attn_ids) + 1)
        layout = _SHRIMP_TRUNK_LAYOUTS.get(counts)
        # Unknown counts (e.g. legacy v2's 6C/3A): leave layout None so the
        # net builds at the env default and the non-strict load surfaces the
        # drift as load_warnings — same soft contract as before, never a 500.
    return {"channels": channels, "attention_heads": heads, "trunk_layout": layout}


def _load_shrimp_checkpoint(ckpt_path: Path, payload: dict[str, Any]) -> LoadedModel:
    """Load a shrimp checkpoint onto CPU.

    The payload is ``{meta, model (state dict), optimizer}``. The block layout
    and head set are fixed, but the trunk width (channels) is env-driven per run
    (some runs use a wider trunk than the c=96 default), so it is read off the
    weights and the net built at that width before a strict load. The state dict
    is the authoritative head set, so a strict mismatch is surfaced as a warning
    rather than 500-ing (same defensive contract as the other loaders)."""

    hf = _shrimp()
    meta = dict(payload.get("meta", {}))
    state_dict = payload["model"]
    if not isinstance(state_dict, dict):
        raise ValueError(f"{ckpt_path.name}: shrimp checkpoint 'model' is not a state dict")

    # The shrimp width (channels), head count, AND trunk layout are env-driven
    # at training time (e.g. c=128/4-head/CCCACCCACCA and c=192/3-head/
    # CCACCACCACCACCA are both known layouts), so a worker running with the
    # process-default env cannot load foreign runs unless it reconstructs the
    # arch off the weights. ShrimpNet is fully parameterized for exactly this.
    inferred = _infer_shrimp_arch(state_dict)
    kwargs: dict[str, Any] = {}
    if inferred["channels"] is not None:
        kwargs["channels"] = inferred["channels"]
    if inferred["attention_heads"] is not None:
        kwargs["attention_heads"] = inferred["attention_heads"]
    if inferred["trunk_layout"] is not None:
        kwargs["trunk_layout"] = inferred["trunk_layout"]
    model = hf.ShrimpNet(**kwargs)
    warnings: list[str] = []
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            warnings.append(f"missing keys: {list(result.missing_keys)[:8]}")
        if result.unexpected_keys:
            warnings.append(f"unexpected keys: {list(result.unexpected_keys)[:8]}")
        if not result.missing_keys and not result.unexpected_keys:
            warnings.append(f"load mismatch: {exc}")

    model.eval()
    # arch carries the run/lineage meta for the provenance panel (only jsonable
    # scalars survive the worker's _model_meta filter, which is exactly meta's
    # shape) plus the weight-inferred structural fields.
    arch: dict[str, Any] = {str(k): v for k, v in meta.items()}
    arch["moves_left_head"] = True
    arch["channels"] = inferred["channels"]
    arch["attention_heads"] = inferred["attention_heads"]
    arch["trunk_layout"] = inferred["trunk_layout"]
    return LoadedModel(
        lineage=SHRIMP,
        model=model,
        arch=arch,
        rl_epoch=_maybe_int(meta.get("epoch")),
        step=_maybe_int(payload.get("step")),
        candidate_radius=None,  # shrimp has no candidate radius (full legal set)
        graft=None,
        load_warnings=warnings,
        # The network always carries the STV + moves-left heads; expose them so
        # the Debug tab renders the full readout (same panels as the dense
        # lineage). Horizons come from the model constant, not the checkpoint.
        stv_horizons=tuple(int(h) for h in hf.STV_HORIZONS),
        has_moves_left=True,
        # The per-cell Q head is train-only (v3+). Older shrimp lineages lack
        # it; gate the debug Q heatmap / regret UI on this from the state dict.
        has_cell_q=("cell_q_head.weight" in state_dict),
        # Run root — lets the search read the run's manifest for the as-trained
        # (gumbel) search profile. <run>/checkpoints/<file>.pt -> <run>.
        run_dir=str(ckpt_path.resolve().parent.parent),
    )


def _shrimp_inputs(hf: SimpleNamespace, state: Any):
    """Featurize one engine decision state into the model's (1, N, *) batch.

    Returns ``(batch, legal_action_ids)`` where ``batch`` is the collate dict
    (feats/nbr/mask/coords/legal_counts) and ``legal_action_ids`` are the packed
    action ids of the legal prefix in support node order — the policy logits are
    positional over exactly this prefix (support layout ``[legal|stones|halo]``,
    legal nodes are slots ``[0, legal_count)``)."""

    facts = hf.facts_from_state(state)
    sup = hf.build_support(facts.stones())
    feats = hf.build_features(facts, sup)
    batch = hf.collate_rows([(sup, feats)])
    legal = sup.legal_coords()  # (legal_count, 2) axial (q, r)
    legal_action_ids = [hf.pack_action_id(int(q), int(r)) for q, r in legal.tolist()]
    return batch, legal_action_ids


def _shrimp_forward(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Run the full forward on CPU, forcing the fp32 relative-position-bias path.

    ``ShrimpNet.build_attn_bias`` branches on ``torch.is_grad_enabled()``: in
    no-grad it gathers the bias DIRECTLY in fp16 (``bias_table.to(fp16)[pair]``)
    for the GPU serve path. On CPU a bare fp16 gather + add is supported, but to
    be unconditionally safe (the caveat: a CPU fp16 op could error) we run the
    forward under ``torch.enable_grad()`` so the fp32 ``_BiasGather`` master path
    is taken — every op stays fp32 — then immediately detach. This is the
    caveat's recommended fp32-bias fallback; the cost is negligible for a single
    position on CPU. ``analyze_position``/``search_position`` are decorated
    ``@torch.no_grad()``; ``enable_grad`` overrides that enclosing context."""

    with torch.enable_grad():
        out = model.forward(
            batch["feats"], batch["nbr"], batch["mask"], batch["coords"]
        )
    return {k: v.detach() for k, v in out.items()}


def _shrimp_policy_rows(
    logits_row: torch.Tensor, legal_action_ids: list[int]
) -> list[dict[str, Any]]:
    """Softmax the policy over the legal prefix -> per-candidate rows.

    ``logits_row`` is the (Npad,) policy logit row; only the first
    ``len(legal_action_ids)`` slots are legal (the rest are pad/halo, mask-zeroed
    in the model). Softmax over exactly the legal prefix, then map each slot back
    to its packed action id."""

    legal_count = len(legal_action_ids)
    if legal_count == 0:
        return []
    priors = torch.softmax(logits_row[:legal_count].float(), dim=0).cpu().numpy()
    rows = []
    for aid, prob in zip(legal_action_ids, priors.tolist()):
        coord = _coord_of(aid)
        rows.append({"action_id": int(aid), "q": coord["q"], "r": coord["r"], "p": round(float(prob), 6)})
    rows.sort(key=lambda r: r["p"], reverse=True)
    return rows


def _shrimp_cell_q_scalars(
    hf: SimpleNamespace, out: dict[str, Any], legal_count: int
) -> torch.Tensor | None:
    """Decode the per-cell Q head to a (legal_count,) scalar tensor in [-1, 1].

    The ``cell_q`` logits are (Npad, 65) per the support layout; only the first
    ``legal_count`` rows are legal placements (the rest are stones/halo, mask-
    zeroed in the model and would decode to a spurious 0.0). MUST slice the legal
    prefix BEFORE decoding. ``decode_binned_value`` is the same softmax-
    expectation·linspace(-1,1,65)·clamp[-1,1] used for ``value``; Q is from the
    side-to-move's perspective (+1 good for the mover). Returns ``None`` when the
    head is absent (older checkpoint) or there are no legal cells (terminal)."""

    if "cell_q" not in out or legal_count <= 0:
        return None
    cq_logits = out["cell_q"][0][:legal_count].float()  # (legal_count, 65) — PREFIX SLICE FIRST
    return hf.decode_binned_value(cq_logits)  # (legal_count,) in [-1, 1], mover POV


def _shrimp_cell_q_rows(
    hf: SimpleNamespace, out: dict[str, Any], legal_action_ids: list[int]
) -> list[dict[str, Any]] | None:
    """Per-cell decoded-Q rows for the analyze payload, sorted by Q desc.

    Each row is ``{action_id, q, r, qv}`` where ``qv`` is the decoded scalar in
    [-1, 1] (mover POV). ``cell_q[0]`` is therefore the Q-best legal cell. Played
    / Q-best marking is left to the frontend (analyze knows the legal prefix but
    not the move recorded at the swept ply)."""

    legal_count = len(legal_action_ids)
    q_scalar = _shrimp_cell_q_scalars(hf, out, legal_count)
    if q_scalar is None:
        return None
    rows = []
    for aid, qv in zip(legal_action_ids, q_scalar.tolist()):
        coord = _coord_of(aid)
        rows.append(
            {"action_id": int(aid), "q": coord["q"], "r": coord["r"], "qv": round(float(qv), 5)}
        )
    rows.sort(key=lambda r: r["qv"], reverse=True)
    return rows


def _shrimp_dist(hf: SimpleNamespace, logits: torch.Tensor) -> dict[str, Any]:
    """Scalar + 65-bin distribution for one value-style head's (65,) logits."""

    flat = logits.float().reshape(-1)
    scalar = float(hf.decode_binned_value(flat.reshape(1, -1)).reshape(()).item())
    dist = [round(float(x), 5) for x in torch.softmax(flat, dim=0).cpu().numpy()]
    return {"scalar": scalar, "dist": dist}


@torch.no_grad()
def _analyze_shrimp(loaded: LoadedModel, action_ids: Sequence[int]) -> dict[str, Any]:
    hf = _shrimp()
    state = state_from_actions(action_ids)
    batch, legal_action_ids = _shrimp_inputs(hf, state)
    out = _shrimp_forward(loaded.model, batch)

    policy = _shrimp_policy_rows(out["policy"][0], legal_action_ids)
    opp = None
    if "opp_policy" in out:
        opp = _shrimp_policy_rows(out["opp_policy"][0], legal_action_ids)

    value = _shrimp_dist(hf, out["value"][0])

    stv: dict[str, Any] = {}
    for horizon in loaded.stv_horizons:
        key = f"stvalue_{horizon}"
        if key in out:
            stv[str(horizon)] = _shrimp_dist(hf, out[key][0])

    # moves_left: the head emits 65-bin logits decoded the SAME way the dense
    # lineage's is (decode_binned_value -> scalar in [-1, 1]); the UI undoes the
    # affine map to remaining decisions with moves_left_cap (512). NOT
    # decode_moves_left (that returns a raw decisions count the UI would
    # double-scale).
    moves_left = _shrimp_dist(hf, out["moves_left"][0]) if "moves_left" in out else None

    # Per-cell Q head (v3+): decoded scalar per legal cell, mover POV. None for
    # older lineages lacking the head OR terminal/no-legal positions. The bare
    # net always emits ``out["cell_q"]``, so gate on the checkpoint's state dict
    # (``loaded.has_cell_q``) — not on output presence — for the lineage check.
    cell_q = (
        _shrimp_cell_q_rows(hf, out, legal_action_ids) if loaded.has_cell_q else None
    )

    current = engine.current_player(state)
    current_role = getattr(current, "value", str(current))
    current_index = 1 if str(current_role).endswith("1") else 0

    return {
        "current_player": current_index,
        "current_role": str(current_role),
        "candidate_count": len(legal_action_ids),
        "legal_count": int(engine.legal_action_count(state)),
        "value": value["scalar"],
        "cell_q": cell_q,
        # The owner-swap "optimism" probe does not apply here: shrimp encodes
        # side-to-move ownership in its features (own/opp planes), so it is
        # marked N/A (the UI hides the panel when null).
        "value_swapped": None,
        "optimism": None,
        "value_bins": [round(float(x), 5) for x in hf.value_bins().cpu().numpy()],
        "value_dist": value["dist"],
        "policy": policy,
        "opp_policy": opp,
        "stvalue": stv,
        "moves_left": moves_left,
        "input_planes": None,  # support-graph featurized lineage: no dense planes
    }


def _decode_id_weight_pairs(ids_bytes: Any, weights_bytes: Any) -> list[tuple[int, float]]:
    """Decode a Rust search result's (uint32 action ids, float32 weights) byte
    buffers into ``(action_id, weight)`` pairs for ``_policy_pairs_to_rows``."""

    ids = np.frombuffer(bytes(ids_bytes), dtype=np.uint32)
    weights = np.frombuffer(bytes(weights_bytes), dtype=np.float32)
    return [(int(a), float(w)) for a, w in zip(ids.tolist(), weights.tolist())]


@lru_cache(maxsize=16)
def _shrimp_run_config(run_dir: str | None):
    """The run's parsed ShrimpConfig (from ``<run>/manifest.json``), or None.

    This is how the search runs AS-TRAINED: the manifest carries the exact
    ``model.config`` the run trains/evals with — the gumbel levers, the
    budget-calibrated ``gumbel_m``, the divergence set, vbs. A missing/foreign
    manifest falls back to None (caller uses SelfplayConfig defaults)."""

    if not run_dir:
        return None
    try:
        from shrimp.config import parse_shrimp_config

        manifest = Path(run_dir) / "manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        return parse_shrimp_config(data.get("model", {}).get("config", {}))
    except Exception:
        return None


@torch.no_grad()
def _search_shrimp(
    loaded: LoadedModel,
    action_ids: Sequence[int],
    *,
    visits: int,
    c_puct: float,
    seed: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Real CPU MCTS via the Rust ShrimpMctsSession (single-position search).

    Searches with the run's AS-TRAINED profile, exactly like the eval arena
    (eval_arena.play_checkpoint_match): the divergence overrides — including
    the Gumbel levers (root gumbel + sequential halving + non-root select) and
    the budget-calibrated ``gumbel_m`` — come from the run's own manifest via
    ``build_divergence_overrides``, and move selection happens IN-SEARCH via
    ``move_temperatures`` (the eval protocol: sampled opening plies at
    temperature 1, argmax after). ``temperature=0`` (the debug default) keeps
    the clean reproducible read: no Dirichlet noise, returned best == the
    profile's greedy selection. A run without gumbel in its config gets the
    same plain-PUCT profile as before (the overrides mirror whatever the
    config says)."""

    from shrimp.config import SelfplayConfig, build_divergence_overrides
    from shrimp.inference import ShrimpEvaluator

    hf = _shrimp()
    cfg = _shrimp_run_config(loaded.run_dir)
    sp = cfg.selfplay if cfg is not None else SelfplayConfig()
    # Eval-arena vbs: the multistage eval's eval_virtual_batch_size, not the
    # self-play in-flight depth (48) — single-root CPU search wants the former.
    eval_vbs = 32
    if cfg is not None:
        eval_vbs = int(
            getattr(cfg.multi_stage_eval, "eval_virtual_batch_size", 32) or 32
        )
    overrides = build_divergence_overrides(sp)

    state = state_from_actions(action_ids)
    evaluator = ShrimpEvaluator(loaded.model, device="cpu")
    session = hf._rust.ShrimpMctsSession(max_states=65536)

    # Mirrors eval_arena's `common` search kwargs (visits/c_puct come from the
    # request; the rest from the run's selfplay profile) + per-root
    # move_temperatures for in-search selection.
    result = session.search(
        [int(seed)],
        (state,),
        evaluator=evaluator,
        visits=int(visits),
        c_puct=float(c_puct),
        temperature=0.0,
        move_temperatures=[float(temperature)],
        divergence_overrides=overrides,
        seed=int(seed),
        virtual_batch_size=eval_vbs,
        widening_policy_mass=sp.widening_policy_mass,
        widening_max_children=sp.widening_max_children,
        widening_min_children=sp.widening_min_children,
        fpu_reduction=sp.fpu_reduction,
        tss_enabled=sp.tss_enabled,
        search_parity_mode=sp.search_parity_mode,
    )[0]

    visit_pairs = _decode_id_weight_pairs(
        result["visit_policy_action_ids_bytes"], result["visit_policy_weights_bytes"]
    )
    root_prior_pairs = _decode_id_weight_pairs(
        result["root_prior_policy_action_ids_bytes"],
        result["root_prior_policy_weights_bytes"],
    )
    best_action_id = int(result["action_id"])

    return {
        "visits_requested": int(visits),
        "visits": int(result["visits"]),
        "root_value": float(result["root_value"]),
        "best_action_id": best_action_id,
        "best": _coord_of(best_action_id),
        "visit_policy": _policy_pairs_to_rows(visit_pairs, normalize=True),
        "root_prior": _policy_pairs_to_rows(root_prior_pairs, normalize=False),
        # Selection happened INSIDE the search (move_temperatures), under the
        # run's as-trained profile — callers (match bots) should play
        # best_action_id directly instead of re-sampling visit rows.
        "selection_in_search": True,
        "search_profile": {
            "source": "manifest" if cfg is not None else "defaults",
            "temperature": float(temperature),
            "gumbel_root": bool(overrides.get("gumbel_root")),
            "gumbel_sequential_halving": bool(
                overrides.get("gumbel_sequential_halving")
            ),
            "gumbel_m": int(overrides.get("gumbel_m", 0)),
        },
    }


# ---------------------------------------------------------------------------
# shrimp attention map (Model Debug interactive attention view)
# ---------------------------------------------------------------------------

# Skeleton/echo constants. NUM_TOKENS is a true model constant; BLOCKS/HEADS
# are only the non-shrimp n/a-skeleton clamps — for a loaded shrimp model
# the real counts are read off the model (arch-dependent, e.g. 3 blocks x 4
# heads or 5 blocks x 3 heads) in attention_position.
_ATTN_NUM_TOKENS = 8
_ATTN_NUM_BLOCKS = 3
_ATTN_NUM_HEADS = 4


def _attention_na_skeleton(loaded: "LoadedModel", block: int, head: int | None, action_ids: Sequence[int]) -> dict[str, Any]:
    """Empty attention payload for a non-shrimp lineage (mirrors the INPUTS
    tab's ``input_planes: null`` n/a contract — a 200 with ``found: False``)."""

    return {
        "found": False,
        "reason": "lineage_na",
        "lineage": loaded.lineage,
        "block": int(block),
        "head": (None if head is None else (head if head == "max" else int(head))),
        "num_blocks": 0,
        "num_heads": 0,
        "num_tokens": _ATTN_NUM_TOKENS,
        "num_cells": 0,
        "cells": [],
        "token_queries": [],
        "cell_query": None,
        "incoming_token_to_cell": None,
        "ply": len(action_ids),
    }


def attention_position(
    loaded: "LoadedModel",
    action_ids: Sequence[int],
    *,
    block: int = 0,
    head: int | None = None,
    query: dict[str, Any],
    n: int | None = None,
) -> dict[str, Any]:
    """Per-query attention distribution for the shrimp set-transformer.

    Registers a forward hook on ``model.attn_blocks[block].attn`` (a
    ``RelPosAttention``) and recomputes the per-head softmax attention over the
    joint sequence ``[8 tokens ; N cells]`` from the hook's ``(post-LN1 seq,
    attn_bias)`` inputs. Returns ALL 8 token-query rows plus the ONE requested
    cell-query row (when ``query.type == "cell"``), each sliced into its
    token-part (``[:8]``) and cell-part (``[8:]``); never serializes the full
    ``(8+N)^2`` matrix. Floats rounded to 6 dp.

    ``n`` is an opaque passthrough for signature parity with analyze/search and
    is unused here. Non-shrimp lineages return the n/a skeleton (found False).
    """

    if loaded.lineage != SHRIMP:
        block = max(0, min(int(block), _ATTN_NUM_BLOCKS - 1))
        head = None if head is None else ("max" if head == "max" else max(0, min(int(head), _ATTN_NUM_HEADS - 1)))
        return _attention_na_skeleton(loaded, block, head, action_ids)

    # Structural constants come from the LOADED model, not module constants —
    # the attn block count and head count are arch-dependent (e.g. 3 blocks x 4
    # heads, or 5 blocks x 3 heads).
    num_blocks = len(loaded.model.attn_blocks)
    num_heads = int(loaded.model.attn_blocks[0].attn.heads)
    block = max(0, min(int(block), num_blocks - 1))
    head = None if head is None else ("max" if head == "max" else max(0, min(int(head), num_heads - 1)))

    hf = _shrimp()
    state = state_from_actions(action_ids)
    batch, legal_action_ids = _shrimp_inputs(hf, state)
    model = loaded.model

    target = model.attn_blocks[block].attn  # RelPosAttention
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, inputs):
        # Pre-hook: inputs == (seq, attn_bias). seq is the post-LN1 joint
        # sequence (1, S, C); attn_bias is the materialized (1, heads, S, S)
        # additive bias. Recompute q/k -> scores*scale + bias -> softmax exactly
        # as the 'materialized' impl does (numerically identical to sdpa).
        seq, attn_bias = inputs[0], inputs[1]
        b, s, c = seq.shape
        h, d = module.heads, module.head_dim
        q = module.q_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        k = module.k_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * module.scale + attn_bias  # (1, h, S, S)
        attn = torch.softmax(scores, dim=-1)
        captured["attn"] = attn[0].detach()  # (heads, S, S)

    # Force the materialized fp32 bias path: serve-flex must be OFF so the hook's
    # attn_bias is a real (1, heads, S, S) tensor, never a _FlexBias carrier. We
    # run the forward under enable_grad() (same trick as _shrimp_forward) so
    # build_attn_bias takes the master fp32 _BiasGather branch, and temporarily
    # disable any leaked serve-flag, restoring it in finally.
    prev_flex = getattr(model, "_serve_flex", False)
    handle = target.register_forward_pre_hook(_hook)
    try:
        if prev_flex:
            model._serve_flex = False
        with torch.enable_grad():
            model.forward(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    finally:
        handle.remove()
        if prev_flex:
            model._serve_flex = prev_flex

    attn = captured.get("attn")
    if attn is None:  # pragma: no cover - hook always fires for a valid block
        raise RuntimeError("attention hook did not fire")
    # Guard against a leaked _FlexBias slipping through (no materialized bias):
    if attn.dim() != 3:  # pragma: no cover - defensive
        raise RuntimeError("attention hook captured an unexpected tensor shape")

    # Reduce heads -> (S, S). None: mean over heads (rows sum to 1). "max":
    # per-(query,key) max over heads (surfaces a key any single head attends to
    # strongly; rows no longer sum to 1). int: that single head.
    if head is None:
        A = attn.mean(0)
    elif head == "max":
        A = attn.amax(0)
    else:
        A = attn[head]

    coords = batch["coords"][0]  # (N, 2) axial (q, r), model node order
    n_cells = int(coords.shape[0])
    nt = _ATTN_NUM_TOKENS
    legal_count = len(legal_action_ids)

    def _row(i: int) -> tuple[list[float], list[float]]:
        r = A[i]
        over_tokens = [round(float(x), 6) for x in r[:nt].tolist()]
        over_cells = [round(float(x), 6) for x in r[nt:].tolist()]
        return over_cells, over_tokens

    # Stable cell axis: cells[j] describes sequence index (8 + j). Only the legal
    # prefix [0, legal_count) carries a packed action_id; stones/halo -> None.
    coords_list = coords.tolist()
    cells: list[dict[str, Any]] = []
    for j in range(n_cells):
        q_j, r_j = int(coords_list[j][0]), int(coords_list[j][1])
        aid = hf.pack_action_id(q_j, r_j) if j < legal_count else None
        cells.append({"i": nt + j, "q": q_j, "r": r_j, "action_id": aid})

    token_queries: list[dict[str, Any]] = []
    for t in range(nt):
        over_cells, over_tokens = _row(t)
        token_queries.append(
            {"token": t, "attn_over_cells": over_cells, "attn_over_tokens": over_tokens}
        )

    qtype = str(query.get("type", "cell"))
    qid = int(query.get("id", 0))

    cell_query: dict[str, Any] | None = None
    incoming: list[float] | None = None
    if qtype == "cell":
        # Resolve the clicked cell to a support-node index by matching its packed id
        # against EVERY support cell (legal moves AND placed stones AND halo), not
        # just the legal prefix — a stone/halo cell has no legal action_id, so the
        # old legal-only match silently failed (bad_query) on every stone click.
        j = None
        for k in range(n_cells):
            if hf.pack_action_id(int(coords_list[k][0]), int(coords_list[k][1])) == qid:
                j = k
                break
        if j is None:
            out = _attention_na_skeleton(loaded, block, head, action_ids)
            out["reason"] = "bad_query"
            # still echo the real lineage/structural constants for a shrimp miss
            out["lineage"] = loaded.lineage
            out["num_blocks"] = num_blocks
            out["num_heads"] = num_heads
            out["num_cells"] = n_cells
            out["cells"] = cells
            out["token_queries"] = token_queries
            return out
        i = nt + j
        over_cells, over_tokens = _row(i)
        cq = cells[j]
        cell_query = {
            "action_id": qid,
            "i": i,
            "q": cq["q"],
            "r": cq["r"],
            "attn_over_cells": over_cells,
            "attn_over_tokens": over_tokens,
        }
        # Incoming: how much each of the 8 tokens attends TO this cell (one column
        # of the token rows) — pre-sliced so the UI needs no second request.
        incoming = [token_queries[t]["attn_over_cells"][j] for t in range(nt)]

    return {
        "found": True,
        "reason": None,
        "lineage": loaded.lineage,
        "block": int(block),
        "head": (None if head is None else (head if head == "max" else int(head))),
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "num_tokens": nt,
        "num_cells": n_cells,
        "cells": cells,
        "token_queries": token_queries,
        "cell_query": cell_query,
        "incoming_token_to_cell": incoming,
        "ply": len(action_ids),
    }


# ===========================================================================
# Model Debug v2 additions: checkpoint meta plumb-through, pure-Python PUCT
# tree (the "debug search"), recorded .npz row reader, whole-game error sweep.
# ===========================================================================


def moves_left_cap(loaded: LoadedModel) -> int | None:
    """Decode cap for the moves-left head; ``None`` when the model has none.

    The head maps remaining decisions affinely onto the binned support over
    ``[0, MOVES_LEFT_CAP]``; the UI needs the cap to undo that mapping."""

    if not loaded.has_moves_left:
        return None
    from shrimp.constants import MOVES_LEFT_CAP

    return int(MOVES_LEFT_CAP)


def param_count(loaded: LoadedModel) -> int | None:
    """Total parameter count of the loaded model (display-only provenance)."""

    try:
        return int(sum(p.numel() for p in loaded.model.parameters()))
    except Exception:
        return None


def _stm_index(state: Any) -> int:
    """0/1 index of the side to move (replay-derived, never parity-assumed —
    Hexo phases can give one player consecutive placements)."""

    role = getattr(engine.current_player(state), "value", "")
    return 1 if str(role).endswith("1") else 0


def _player_index(player: Any) -> int:
    return 1 if str(getattr(player, "value", player)).endswith("1") else 0


def _tree_evaluator(loaded: LoadedModel, n: int | None):
    """One-position ``state -> (prior_pairs, value_stm)`` closure.

    Reuses the exact featurize+forward wrappers analyze/search already build. One
    net call per position; batch of 1 on CPU — fine at <=20k visits, ~4096 is the
    practical interactive ceiling. ``n`` is accepted for call-site compatibility
    and unused."""

    hf = _shrimp()

    def evaluate(state: Any) -> tuple[list[tuple[int, float]], float]:
        batch, legal_action_ids = _shrimp_inputs(hf, state)
        out = _shrimp_forward(loaded.model, batch)
        value = float(
            hf.decode_binned_value(out["value"][0].float().reshape(1, -1)).reshape(()).item()
        )
        legal_count = len(legal_action_ids)
        if legal_count == 0:
            return [], value
        priors = torch.softmax(out["policy"][0][:legal_count].float(), dim=0)
        return list(zip(legal_action_ids, priors.tolist())), value

    return evaluate


class DebugSearchNode:
    """One node of the pure-Python debug PUCT tree (``search_tree_position``).

    ``w`` accumulates backed-up values expressed from THIS node's side-to-move
    perspective; ``stm`` is discovered on first arrival (replay-derived), and
    ``children is None`` means unexpanded while ``[]`` means terminal."""

    __slots__ = ("action_id", "prior", "n", "w", "v", "stm", "children", "terminal_value")

    def __init__(self, action_id: int | None, prior: float) -> None:
        self.action_id = action_id
        self.prior = float(prior)
        self.n = 0
        self.w = 0.0
        self.v: float | None = None
        self.stm: int | None = None
        self.children: list["DebugSearchNode"] | None = None
        self.terminal_value: float | None = None

    @property
    def qm(self) -> float:
        return self.w / self.n if self.n else 0.0


def _select_puct(node: DebugSearchNode, c_puct: float, rng: random.Random) -> DebugSearchNode:
    """argmax PUCT child; seeded rng breaks (near-)exact score ties only."""

    assert node.children
    sqrt_n = math.sqrt(max(1, node.n))
    best: list[DebugSearchNode] = []
    best_score: float | None = None
    for child in node.children:
        if child.n > 0 and child.stm is not None:
            q = child.qm if child.stm == node.stm else -child.qm
        else:
            q = 0.0  # neutral FPU, matching the noise-free debug search intent
        score = q + c_puct * child.prior * sqrt_n / (1 + child.n)
        if best_score is None or score > best_score + 1e-12:
            best, best_score = [child], score
        elif score >= best_score - 1e-12:
            best.append(child)
    return best[0] if len(best) == 1 else rng.choice(best)


# Hard ceiling on serialized tree nodes per response (spec §3.7).
_TREE_NODE_CAP = 4000


def _serialize_tree(
    root: DebugSearchNode,
    *,
    c_puct: float,
    max_depth: int,
    top_k: int,
    min_n: int,
) -> tuple[dict[str, Any], int, bool]:
    """Prune + serialize the tree; returns (node dict, node count, pruned any)."""

    count = 0
    pruned_any = False

    def ser(node: DebugSearchNode, parent_n: int | None, depth: int) -> dict[str, Any]:
        nonlocal count, pruned_any
        count += 1
        coord = _coord_of(node.action_id) if node.action_id is not None else {"q": None, "r": None}
        kids = node.children or []
        kept: list[DebugSearchNode] = []
        if depth < max_depth:
            eligible = [c for c in kids if c.n >= min_n]
            eligible.sort(key=lambda c: (-c.n, -c.prior, c.action_id))
            kept = eligible[:top_k]
        n_pruned = len(kids) - len(kept)
        if n_pruned > 0:
            pruned_any = True
        u = None
        if parent_n is not None:
            u = c_puct * node.prior * math.sqrt(max(1, parent_n)) / (1 + node.n)
        qm = node.qm
        qm_p0 = qm if node.stm in (None, 0) else -qm
        return {
            "action_id": node.action_id,
            "q": coord["q"],
            "r": coord["r"],
            "n": int(node.n),
            "qm": round(qm, 5),
            "qm_p0": round(qm_p0, 5),
            "p": round(node.prior, 6),
            "u": round(u, 5) if u is not None else None,
            "v": round(node.v, 5) if node.v is not None else None,
            "pruned_children": int(n_pruned),
            "children": [ser(child, node.n, depth + 1) for child in kept],
        }

    tree = ser(root, None, 0)
    return tree, count, pruned_any


@torch.no_grad()
def search_tree_position(
    loaded: LoadedModel,
    action_ids: Sequence[int],
    *,
    visits: int = 512,
    c_puct: float = 1.5,
    seed: int = 0,
    max_depth: int = 12,
    top_k: int = 8,
    min_n: int = 2,
    n: int | None = None,
) -> dict[str, Any]:
    """Pure-Python deterministic PUCT for the Tree Explorer (spec §3.7).

    A NEW debug-only search ("py_debug"): same inference wrappers as the
    production search, no Dirichlet noise, neutral FPU/temperature, seeded RNG
    used ONLY for exact-score tie-breaks — same request, identical JSON. It will
    NOT bit-match the Rust engine's tie-breaking; the UI labels it accordingly."""

    root_state = state_from_actions(action_ids)
    if engine.terminal(root_state) is not None:
        raise ValueError("position is terminal; nothing to search")

    evaluate = _tree_evaluator(loaded, n)
    rng = random.Random(int(seed))
    root = DebugSearchNode(None, 1.0)

    for _ in range(max(1, int(visits))):
        state = engine.clone_state(root_state)
        node = root
        path = [node]
        while True:
            if node.stm is None:
                node.stm = _stm_index(state)
                term = engine.terminal(state)
                if term is not None:
                    node.children = []
                    if term.winner is None:
                        node.terminal_value = 0.0
                    else:
                        node.terminal_value = 1.0 if _player_index(term.winner) == node.stm else -1.0
            if node.terminal_value is not None:
                leaf_value, leaf_stm = node.terminal_value, node.stm
                break
            if node.children is None:
                pairs, value = evaluate(state)
                node.children = [DebugSearchNode(int(a), float(p)) for a, p in pairs]
                node.v = float(value)
                leaf_value, leaf_stm = node.v, node.stm
                break
            if not node.children:  # defensive: no candidates yet not terminal
                leaf_value, leaf_stm = 0.0, node.stm
                break
            child = _select_puct(node, float(c_puct), rng)
            engine.apply_action(state, engine.PlacementAction(unpack_coord_id(int(child.action_id))))
            path.append(child)
            node = child
        for visited in path:
            visited.n += 1
            if visited.stm is not None:
                visited.w += leaf_value if visited.stm == leaf_stm else -leaf_value

    # PV = max-N path over the FULL tree (serialization pruning is separate).
    pv: list[int] = []
    node = root
    while node.children:
        best = max(node.children, key=lambda c: (c.n, c.prior, -c.action_id))
        pv.append(int(best.action_id))
        if best.n <= 0:
            break
        node = best
    if not pv:
        raise ValueError("debug tree produced no root children")
    best_action_id = pv[0]

    # Serialize under the hard node cap, tightening top_k until it fits.
    effective_top_k = max(1, int(top_k))
    forced_prune = False
    while True:
        tree, count, pruned = _serialize_tree(
            root, c_puct=float(c_puct), max_depth=int(max_depth), top_k=effective_top_k, min_n=int(min_n)
        )
        if count <= _TREE_NODE_CAP or effective_top_k <= 1:
            break
        effective_top_k = max(1, effective_top_k // 2)
        forced_prune = True

    return {
        "visits": int(root.n),
        "root_value": round(root.qm, 5),
        "best_action_id": int(best_action_id),
        "pv": pv,
        "node_count": int(count),
        "truncated": bool(pruned or forced_prune),
        "engine": "py_debug",
        "params": {
            "visits": int(visits),
            "c_puct": float(c_puct),
            "seed": int(seed),
            "max_depth": int(max_depth),
            "top_k": int(top_k),
            "min_n": int(min_n),
        },
        "tree": tree,
    }


def _npz_policy_rows(
    arrays: dict[str, Any], act_key: str, w_key: str, off_key: str, i: int
) -> tuple[list[dict[str, Any]], float]:
    """One row's (action_id -> weight) policy as normalized p-desc rows + raw total."""

    off = arrays[off_key]
    a, b = int(off[i]), int(off[i + 1])
    acts = arrays[act_key][a:b]
    weights = arrays[w_key][a:b]
    total = float(weights.sum())
    rows = []
    for aid, w in zip(acts.tolist(), weights.tolist()):
        coord = _coord_of(int(aid))
        rows.append(
            {
                "action_id": int(aid),
                "q": coord["q"],
                "r": coord["r"],
                "p": round(float(w) / total, 6) if total > 0 else 0.0,
            }
        )
    rows.sort(key=lambda r: r["p"], reverse=True)
    return rows, total


def read_record_row(npz_path: str | Path, turn_index: int, expect_player: int | None) -> dict[str, Any]:
    """Decode one recorded training row from a compact self-play shard (§3.9).

    Never raises — returns ``found:false`` with a reason instead, including
    ``"bad_shard"`` for foreign/partial .npz files missing expected arrays.
    The compact shard format intentionally drops some finalize-time facts
    (``value_target_reason`` / ``policy_surprise`` / ``pcr_full`` /
    ``opp_policy_source`` — see compact_io's docstring), so those come back as
    null (never a fabricated neutral); ``search_visits`` is recovered from the
    raw visit mass (0 = unknown, normalized-weight shards) and
    ``frequency_weight`` from surprise-materialized row duplication. The server
    overlays ``truncated``/``value_target_reason`` from the .hxr record."""

    path = Path(npz_path)
    if not path.is_file():
        return {"found": False, "reason": "no_shard", "npz": None, "turn_index": None, "row": None}
    try:
        with np.load(path, allow_pickle=True) as data:
            arrays = {key: data[key] for key in data.files}
    except Exception:
        return {"found": False, "reason": "no_shard", "npz": str(npz_path), "turn_index": None, "row": None}

    miss = {"found": False, "reason": "no_row", "npz": str(npz_path), "turn_index": int(turn_index), "row": None}
    if "num_rows" not in arrays or "turn_index" not in arrays:
        return {**miss, "reason": "bad_shard"}
    try:
        n_rows = int(arrays["num_rows"])
        turns = arrays["turn_index"]
        matches = [i for i in range(n_rows) if int(turns[i]) == int(turn_index)]
        if not matches:
            return miss
        i = matches[0]

        player = int(arrays["current_player"][i])
        if expect_player is not None and player != int(expect_player):
            # M8 misalignment guard: a row whose stored side-to-move disagrees with
            # the replay-derived one must be refused, never silently shown.
            return {**miss, "reason": "row_mismatch"}

        horizons = [int(h) for h in arrays["horizons"]]
        policy, policy_total = _npz_policy_rows(arrays, "pol_act", "pol_w", "pol_off", i)
        opp_policy, _opp_total = _npz_policy_rows(arrays, "opp_act", "opp_w", "opp_off", i)
        stvalue = {
            str(h): {
                "target": round(float(arrays["stvalue"][i, c]), 5),
                "mask": bool(arrays["stvalue_mask"][i, c] > 0.0),
            }
            for c, h in enumerate(horizons)
        }
        moves_left = None
        if "moves_left" in arrays:  # restnet-era shards only; raw decisions remaining, -1 = masked
            raw = float(arrays["moves_left"][i])
            moves_left = {"target": round(raw, 3), "mask": bool(raw >= 0.0)}

        row = {
            "current_player": player,
            "phase": str(arrays["phase"][i]),
            "value_target": round(float(arrays["value"][i]), 5),
            "value_target_reason": None,  # not persisted in compact shards (server overlays)
            "policy": policy,
            "opp_policy": opp_policy or None,
            "opp_policy_source": None,  # not persisted in compact shards
            "stvalue": stvalue,
            "moves_left": moves_left,
            "policy_surprise": None,  # not persisted (baked into row duplication at write)
            # Raw visit mass when the stored weights are counts; runs that store the
            # NORMALIZED visit policy (sum ~1) can't recover it -> 0 means unknown.
            "search_visits": int(round(policy_total)) if policy_total > 1.5 else 0,
            "pcr_full": None,  # not persisted (row existence implies a full search today)
            "frequency_weight": float(len(matches)),  # surprise weighting = in-place duplication
            "truncated": False,  # not persisted (server overlays from the .hxr record)
        }
    except (KeyError, IndexError, ValueError, TypeError):
        # Never-raises contract (§4.1): a shard carrying num_rows/turn_index but
        # missing/garbling other expected arrays is foreign or partially written.
        return {**miss, "reason": "bad_shard"}
    return {"found": True, "reason": None, "npz": str(npz_path), "turn_index": int(turn_index), "row": row}


def _npz_rows_by_turn(npz_path: str | Path) -> dict[int, dict[str, Any]]:
    """turn_index -> {policy pairs, value_target, current_player} for one shard
    (first row wins). ``current_player`` feeds the per-ply M8 misalignment guard
    in ``game_eval_positions``."""

    try:
        with np.load(Path(npz_path), allow_pickle=True) as data:
            arrays = {key: data[key] for key in data.files}
    except Exception:
        return {}
    if "num_rows" not in arrays:
        return {}
    out: dict[int, dict[str, Any]] = {}
    pol_off = arrays["pol_off"]
    for i in range(int(arrays["num_rows"])):
        turn = int(arrays["turn_index"][i])
        if turn in out:
            continue
        a, b = int(pol_off[i]), int(pol_off[i + 1])
        pairs = [
            (int(aid), float(w))
            for aid, w in zip(arrays["pol_act"][a:b].tolist(), arrays["pol_w"][a:b].tolist())
        ]
        out[turn] = {
            "policy": pairs,
            "value_target": float(arrays["value"][i]),
            "current_player": int(arrays["current_player"][i]),
        }
    return out


def _policy_kl(
    recorded_pairs: Sequence[tuple[int, float]], prior_pairs: Sequence[tuple[int, float]]
) -> float | None:
    """KL(recorded ‖ prior) over the recorded support, prior renormalized over
    that support and floored at 1e-9 (spec §4.1 definition)."""

    total = sum(w for _, w in recorded_pairs)
    if total <= 0:
        return None
    prior = {int(a): float(p) for a, p in prior_pairs}
    support = [(int(a), float(w) / total) for a, w in recorded_pairs if w > 0]
    prior_total = sum(prior.get(a, 0.0) for a, _ in support)
    if prior_total <= 0:
        prior_total = 1.0  # all prior mass off-support; the floor dominates
    kl = 0.0
    for aid, t in support:
        p = max(prior.get(aid, 0.0) / prior_total, 1e-9)
        kl += t * (math.log(t) - math.log(p))
    return round(float(kl), 5)


@torch.no_grad()
def game_eval_positions(
    loaded: LoadedModel,
    action_ids: Sequence[int],
    plies: Sequence[int],
    npz_path: str | Path | None = None,
    winner: int | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    """Game Error Sweep chunk (§3.10): one forward per requested ply, joined
    against the game's .npz training rows (shard decoded ONCE per chunk).

    The game is replayed incrementally with one engine walk (plies ascending),
    not from scratch per ply. ``winner`` is the 0/1 winner index or None."""

    action_ids = [int(a) for a in action_ids]
    evaluate = _tree_evaluator(loaded, n)
    wanted = sorted({int(p) for p in plies if 0 <= int(p) <= len(action_ids)})
    rows = _npz_rows_by_turn(npz_path) if npz_path else {}

    # Per-ply Q-regret (shrimp v3+ only): one extra shrimp forward per ply to
    # decode the per-cell Q head, then regret = bestQ - playedQ (mover POV, >=0).
    # Gated on the checkpoint actually carrying cell_q; every other lineage / an
    # older shrimp checkpoint leaves all regret keys None (additive, backward-
    # compatible). Built once; ``_regret_for_ply`` returns the 6-key dict.
    want_regret = loaded.lineage == SHRIMP and loaded.has_cell_q
    hf_regret = _shrimp() if want_regret else None
    _NULL_REGRET = {
        "played_q": None,
        "best_q": None,
        "regret": None,
        "q_best_aid": None,
        "q_best_match": None,
        "missed_near_win": None,
    }

    def _regret_for_ply(state: Any, ply: int) -> dict[str, Any]:
        # No played move at the final position (ply == total) -> no regret.
        if not want_regret or ply >= len(action_ids):
            return dict(_NULL_REGRET)
        batch, legal_action_ids = _shrimp_inputs(hf_regret, state)
        legal_count = len(legal_action_ids)
        if legal_count == 0:  # terminal / no legal cell
            return dict(_NULL_REGRET)
        out = _shrimp_forward(loaded.model, batch)
        q_scalar = _shrimp_cell_q_scalars(hf_regret, out, legal_count)
        if q_scalar is None:
            return dict(_NULL_REGRET)
        played_aid = action_ids[ply]
        try:
            slot = legal_action_ids.index(played_aid)
        except ValueError:
            slot = None  # played move not in legal set (shouldn't happen) -> None
        best_idx = int(torch.argmax(q_scalar))
        best_q = float(q_scalar[best_idx])
        q_best_aid = int(legal_action_ids[best_idx])
        if slot is None:
            return {
                "played_q": None,
                "best_q": round(best_q, 5),
                "regret": None,
                "q_best_aid": q_best_aid,
                "q_best_match": None,
                "missed_near_win": None,
            }
        played_q = float(q_scalar[slot])
        return {
            "played_q": round(played_q, 5),
            "best_q": round(best_q, 5),
            "regret": round(best_q - played_q, 5),  # >= 0 (best is the argmax)
            "q_best_aid": q_best_aid,
            "q_best_match": bool(q_best_aid == played_aid),
            "missed_near_win": bool(
                best_q >= 0.90 and q_best_aid != played_aid and played_q < 0.5
            ),
        }

    out: list[dict[str, Any]] = []
    state = engine.new_game()
    next_action = 0
    for ply in wanted:
        while next_action < ply:
            engine.apply_action(
                state, engine.PlacementAction(unpack_coord_id(action_ids[next_action]))
            )
            next_action += 1
        pairs, value = evaluate(state)
        current = _stm_index(state)
        value_p0 = value if current == 0 else -value

        top1_match = None
        if pairs and ply < len(action_ids):
            best_aid = max(pairs, key=lambda ap: (ap[1], -ap[0]))[0]
            top1_match = bool(int(best_aid) == action_ids[ply])

        kl = None
        value_err_soft = None
        row = rows.get(ply)
        if row is not None and int(row["current_player"]) != current:
            # M8 misalignment guard (mirrors read_record_row's expect_player):
            # a row whose stored side-to-move disagrees with the replay-derived
            # one is a misjoin (wrong game's shard) and must not contribute
            # kl/value_err_soft against the wrong targets.
            row = None
        if row is not None:
            kl = _policy_kl(row["policy"], pairs)
            value_err_soft = round(value - float(row["value_target"]), 5)

        value_err_z = None
        if winner is not None:
            z_p0 = 1.0 if int(winner) == 0 else -1.0
            value_err_z = round(value_p0 - z_p0, 5)

        regret_row = _regret_for_ply(state, ply)

        out.append(
            {
                "ply": int(ply),
                "current_player": int(current),
                "value": round(float(value), 5),
                "value_p0": round(float(value_p0), 5),
                "kl": kl,
                "top1_match": top1_match,
                "value_err_z": value_err_z,
                "value_err_soft": value_err_soft,
                **regret_row,
            }
        )
    # regret_blunder_threshold is echoed every chunk so the frontend can drive the
    # absolute-mode blunder rule without a separate config round-trip.
    return {"plies": out, "regret_blunder_threshold": 0.3}
