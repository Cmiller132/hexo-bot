"""Shared pytest setup.

The dashboard package lives under packages/hexo_frontend/python and is only
pip-installed in the WSL venv. On interpreters without it (e.g. the Windows
smoke run), expose the in-repo source so the pure, always-run tests collect.
An installed package (WSL venv) keeps precedence.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("hexo_frontend") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "hexo_frontend" / "python"))
