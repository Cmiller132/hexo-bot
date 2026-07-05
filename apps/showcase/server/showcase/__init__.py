"""Shrimp — a Hexo bot — showcase server.

FastAPI service that lets anyone play Hexo against Shrimp checkpoints at
several strengths, with every finished game persisted to SQLite (moves as a
compact `.hxr` blob) and a post-game analysis surface (net policy/value,
optional small searched eval).

Imports only `hexo_engine`, `shrimp`, and `hexo_utils` from the repo — the
dev dashboard (`hexo_frontend`) and training stack (`hexo_train`) are never
touched. Model inference runs in a small pool of worker processes (`bots.py`);
the web process itself never imports torch.

LOAD-BEARING: `SHRIMP_SUPPORT_RADIUS` is read once at shrimp import time
(python featurizer and Rust featurizer alike). The shipped main_7 weights were
trained at radius 4, so default it here, before any shrimp import can
happen. An explicit env value always wins.
"""

from __future__ import annotations

import os

os.environ.setdefault("SHRIMP_SUPPORT_RADIUS", "4")

__version__ = "0.1.0"
