"""Python-facing engine error types.

Raised by `hexo_engine.api`; callers across hexo_runner, hexo_frontend, and
the model packages catch `IllegalActionError` for rejected moves and
`EngineUnavailableError` when the maturin-built `hexo_engine._rust` extension
is missing (e.g. Windows-native Python, where only the WSL Linux .so exists).
"""

from __future__ import annotations


class HexoEngineError(Exception):
    """Base class for Python errors raised by the engine package."""


class EngineUnavailableError(HexoEngineError):
    """Raised when the `hexo_engine._rust` extension module cannot be imported.

    In practice this means the maturin-built .so is absent or was built for a
    different interpreter/platform (the prebuilt artifact in the tree is
    WSL/Linux py3.12). Raised lazily on first API call, not at import time
    (see `api._bridge`).
    """


class IllegalActionError(HexoEngineError):
    """Raised when an action is rejected by the Rust rules authority."""
