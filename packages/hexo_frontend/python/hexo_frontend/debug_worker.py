"""CPU-only inference worker for the dashboard Debug tab.

Runs as a child process of the dashboard server, isolated so it can NEVER touch
the training GPU: the parent launches it with ``CUDA_VISIBLE_DEVICES=""`` and all
torch work here is on ``device("cpu")``. Keeping torch in this separate process
also means the HTTP server itself never imports torch, so the live-status
long-poll is never blocked by a heavy forward/search.

Protocol: newline-delimited JSON over stdin/stdout. The parent writes one request
object per line and reads exactly one response object per line. **stdout carries
ONLY protocol JSON** — every diagnostic/warning goes to stderr — so the parent can
parse it unambiguously.

Request:  {"id": int, "op": "ping|info|analyze|search|reeval|search_tree|attention|record_row|game_eval", ...}
Response: {"id": int, "ok": true, "result": {...}}  |  {"id": int, "ok": false, "error": str}

Ops:
  ping        -> {"pong": true}
  info        {checkpoint}            -> checkpoint provenance (graft, epoch, ...)
                                          (meta.has_cell_q flags the v3 cell_q head)
  analyze     {checkpoint, action_ids, n?, planes?} -> all model heads for the position
                                          (+ cell_q: per-legal-cell decoded Q rows
                                           sorted qv desc, or null; meta.has_cell_q)
  search      {checkpoint, action_ids, visits?, c_puct?, n?, seed?} -> fresh CPU MCTS
  reeval      {checkpoint, sequences:[[aid,...],...], n?} -> scalar value per sequence
                                          (value-trajectory chart; one forward each)
  search_tree {checkpoint, action_ids, visits?, c_puct?, seed?, max_depth?, top_k?,
               min_n?, n?} -> pure-Python deterministic PUCT tree ("py_debug")
  attention   {checkpoint, action_ids, block?, head?, query{type,id}, n?} ->
                                          shrimp per-query attention map (n/a for
                                          non-attention lineages: found=False)
  record_row  {npz, turn_index, expect_player} -> recorded .npz training row
                                          (no checkpoint; skips the model load)
  game_eval   {checkpoint, action_ids, plies:[int], npz?, winner?, n?} ->
                                          per-ply reeval/KL/top-1 sweep chunk
                                          (+ per-ply played_q/best_q/regret/
                                           q_best_aid/q_best_match/missed_near_win
                                           for v3 cell_q lineages, else null; plus
                                           top-level regret_blunder_threshold)

Models are cached LRU by checkpoint path so repeat views skip the load.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from typing import Any

from . import debug_infer as di

_MODEL_CACHE: "OrderedDict[str, di.LoadedModel]" = OrderedDict()
_MAX_MODELS = 3


def _log(msg: str) -> None:
    print(f"[debug_worker] {msg}", file=sys.stderr, flush=True)


def _translate_path(path: str) -> str:
    """Accept a Windows path from the parent and map it to this OS.

    The parent process may be Windows while the worker runs under WSL, so
    ``E:\\hexo-bot\\runs\\x.pt`` must become ``/mnt/e/hexo-bot/runs/x.pt``.
    A path that is already POSIX is returned unchanged.
    """

    if sys.platform != "win32" and len(path) >= 2 and path[1] == ":" and path[0].isalpha():
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/")
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"
    return path


def _get_model(checkpoint: str) -> di.LoadedModel:
    key = checkpoint
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        _MODEL_CACHE.move_to_end(key)
        return cached
    resolved = _translate_path(checkpoint)
    _log(f"loading checkpoint {resolved}")
    loaded = di.load_checkpoint(resolved)
    _MODEL_CACHE[key] = loaded
    _MODEL_CACHE.move_to_end(key)
    while len(_MODEL_CACHE) > _MAX_MODELS:
        evicted, _ = _MODEL_CACHE.popitem(last=False)
        _log(f"evicted checkpoint {evicted}")
    return loaded


def _model_meta(loaded: di.LoadedModel) -> dict[str, Any]:
    meta = {
        "lineage": loaded.lineage,
        "rl_epoch": loaded.rl_epoch,
        "step": loaded.step,
        "graft": loaded.graft,
        "candidate_radius": loaded.candidate_radius,
        "expanded_value": loaded.expanded_value,
        "expanded_stv": loaded.expanded_stv,
        "zeroed_feature_cols": loaded.zeroed_feature_cols,
        "load_warnings": loaded.load_warnings,
        "stv_horizons": list(loaded.stv_horizons),
        "has_moves_left": loaded.has_moves_left,
        "has_cell_q": loaded.has_cell_q,
        "moves_left_cap": di.moves_left_cap(loaded),
        "param_count": di.param_count(loaded),
        "arch": {k: loaded.arch[k] for k in sorted(loaded.arch) if _jsonable(loaded.arch[k])},
    }
    # Active SHRIMP_SUPPORT_RADIUS of THIS worker process (read-once module-
    # global, fixed at spawn by the env the parent set). This is the truth the UI
    # surfaces — what the worker ACTUALLY ran at, not merely what was requested —
    # so the candidate set / policy / cell_q / attention heatmaps are known to be
    # over the radius-restricted support. None for non-shrimp lineages
    # (dense_cnn/hexgt have no support radius) and if the import ever fails.
    if loaded.lineage and "shrimp" in str(loaded.lineage).lower():
        try:
            from shrimp.support import _SUPPORT_RADIUS
            meta["support_radius"] = int(_SUPPORT_RADIUS)
        except Exception:
            meta["support_radius"] = None
    else:
        meta["support_radius"] = None
    return meta


def _jsonable(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool, list, tuple, type(None)))


def _handle(req: dict[str, Any]) -> dict[str, Any]:
    op = req.get("op")
    if op == "ping":
        return {"pong": True}

    if op == "info":
        loaded = _get_model(str(req["checkpoint"]))
        return _model_meta(loaded)

    if op == "analyze":
        loaded = _get_model(str(req["checkpoint"]))
        action_ids = [int(a) for a in req.get("action_ids", [])]
        result = di.analyze_position(
            loaded, action_ids, n=req.get("n"), planes=bool(req.get("planes", False))
        )
        result["meta"] = _model_meta(loaded)
        result["ply"] = len(action_ids)
        return result

    if op == "search":
        loaded = _get_model(str(req["checkpoint"]))
        action_ids = [int(a) for a in req.get("action_ids", [])]
        result = di.search_position(
            loaded,
            action_ids,
            visits=int(req.get("visits", 512)),
            c_puct=float(req.get("c_puct", 1.5)),
            n=req.get("n"),
            seed=int(req.get("seed", 0)),
            temperature=float(req.get("temperature", 0.0)),
        )
        result["ply"] = len(action_ids)
        return result

    if op == "search_tree":
        loaded = _get_model(str(req["checkpoint"]))
        action_ids = [int(a) for a in req.get("action_ids", [])]
        return di.search_tree_position(
            loaded,
            action_ids,
            visits=int(req.get("visits", 512)),
            c_puct=float(req.get("c_puct", 1.5)),
            seed=int(req.get("seed", 0)),
            max_depth=int(req.get("max_depth", 12)),
            top_k=int(req.get("top_k", 8)),
            min_n=int(req.get("min_n", 2)),
            n=req.get("n"),
        )

    if op == "attention":
        loaded = _get_model(str(req["checkpoint"]))
        action_ids = [int(a) for a in req.get("action_ids", [])]
        return di.attention_position(
            loaded,
            action_ids,
            block=int(req.get("block", 0)),
            head=(None if req.get("head") is None else ("max" if req.get("head") == "max" else int(req["head"]))),
            query=dict(req.get("query") or {"type": "cell", "id": 0}),
            n=req.get("n"),
        )

    if op == "record_row":
        # No checkpoint involved: a pure .npz decode, so skip the model load.
        return di.read_record_row(
            _translate_path(str(req["npz"])),
            int(req.get("turn_index", 0)),
            int(req["expect_player"]) if req.get("expect_player") is not None else None,
        )

    if op == "game_eval":
        loaded = _get_model(str(req["checkpoint"]))
        action_ids = [int(a) for a in req.get("action_ids", [])]
        npz = req.get("npz")
        winner = req.get("winner")
        return di.game_eval_positions(
            loaded,
            action_ids,
            [int(p) for p in req.get("plies", [])],
            npz_path=_translate_path(str(npz)) if npz else None,
            winner=int(winner) if winner is not None else None,
            n=req.get("n"),
        )

    if op == "reeval":
        loaded = _get_model(str(req["checkpoint"]))
        sequences = req.get("sequences", [])
        values = []
        for seq in sequences:
            aids = [int(a) for a in seq]
            res = di.analyze_position(loaded, aids, n=req.get("n"))
            values.append({"ply": len(aids), "value": res["value"], "current_player": res["current_player"]})
        return {"values": values}

    raise ValueError(f"unknown op: {op!r}")


def main() -> int:
    _log("started")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            result = _handle(req)
            sys.stdout.write(json.dumps({"id": req_id, "ok": True, "result": result}) + "\n")
        except Exception as exc:  # never let one bad request kill the loop
            _log(f"request failed: {exc!r}")
            sys.stdout.write(json.dumps({"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        sys.stdout.flush()
    _log("stdin closed; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
