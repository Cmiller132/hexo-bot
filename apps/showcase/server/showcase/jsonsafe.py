"""Non-finite float scrubbing for JSON payloads.

Python's `json.dumps` default (`allow_nan=True`) happily writes the bare
literals `NaN`/`Infinity`, which are not JSON: Starlette's response encoder
(`allow_nan=False`) refuses them, so one stray NaN in a payload is a 500 —
and one persisted into the analysis cache is a 500 on EVERY read of that row.
The wire contract for "no data" is null (the frontend already renders null as
a gap), so every payload producer scrubs through here.

Stdlib-only on purpose: imported by the web process (`db`, `app`) and by the
torch-side workers (`analysis`, `lab`) alike.
"""

from __future__ import annotations

import math
from typing import Any


def finite_or_none(x: float) -> float | None:
    """A finite float unchanged; NaN/±Inf -> None (JSON null)."""
    return x if math.isfinite(x) else None


def sanitize_json(obj: Any) -> Any:
    """Deep copy of `obj` with every non-finite float replaced by None.

    Walks dicts, lists, and tuples (tuples come back as lists, matching what
    JSON serialization does anyway); other leaves pass through untouched
    (bool is not float in Python, so flags survive).
    """
    if isinstance(obj, float):
        return finite_or_none(obj)
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]
    return obj
