"""Browser-facing tools for local Hexo workflows (the training dashboard).

Modules: ``web`` (the stdlib HTTP server + all routes), ``dashboard`` (board
payload shaping), ``debug_service``/``debug_worker``/``debug_infer`` (the
out-of-process CPU inference stack behind the Debug screen), ``static/`` (the
single-page browser bundle). Entry points: ``python -m hexo_frontend.web`` and
the ``hexo-play`` console script."""

__version__ = "0.1.0"
