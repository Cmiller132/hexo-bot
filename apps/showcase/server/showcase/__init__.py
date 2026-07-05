"""Hexo bot showcase server.

FastAPI service that lets anyone play Hexo against hexfield checkpoints at
several strengths, with every finished game persisted to SQLite (moves as a
compact `.hxr` blob) and a post-game analysis surface (net policy/value,
optional small searched eval).

Imports only `hexo_engine`, `hexfield`, and `hexo_utils` from the repo — the
dev dashboard (`hexo_frontend`) and training stack (`hexo_train`) are never
touched. Model inference runs in a small pool of worker processes (`bots.py`);
the web process itself never imports torch.

LOAD-BEARING: `HEXFIELD_SUPPORT_RADIUS` is read once at hexfield import time
(python featurizer and Rust featurizer alike). The shipped main_7 weights were
trained at radius 4, so default it here, before any hexfield import can
happen. An explicit env value always wins.
"""

from __future__ import annotations

import os

os.environ.setdefault("HEXFIELD_SUPPORT_RADIUS", "4")

__version__ = "0.1.0"
