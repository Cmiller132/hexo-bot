#!/usr/bin/env bash
# Build the three native crates (hexo_engine, hexo_utils, hexfield) with
# maturin develop --release into the active venv ($HEXO_VENV, default .venv at
# repo root). --release is mandatory: a debug featurizer/search crate is ~10x
# slower.
#
# hexo_engine and hexo_utils resolve from the venv site-packages after install.
# hexfield is imported from the source tree via PYTHONPATH (never pip-installed;
# see README), so its compiled extension is mirrored next to the Python package.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEXO_VENV="${HEXO_VENV:-$ROOT/.venv}"

# Prefer the rustup toolchain in ~/.cargo/bin over any distro-packaged cargo on
# PATH (e.g. Debian's /usr/bin/cargo). The workspace Cargo.lock is version 4,
# which needs cargo >= 1.78; an older system cargo fails with
# "lock file version 4 requires -Znext-lockfile-bump".
if [ -x "$HOME/.cargo/bin/cargo" ]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi

# shellcheck disable=SC1091
source "$HEXO_VENV/bin/activate"

for crate in hexo_engine hexo_utils hexfield; do
  echo "=== building $crate ==="
  maturin develop --release -m "$ROOT/packages/$crate/Cargo.toml"
done

# Mirror the hexfield extension into the source tree so PYTHONPATH imports find
# it (maturin installs it into the venv, but hexfield is imported from-tree).
SITE="$("$HEXO_VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
SO=$(ls "$SITE"/hexfield/_rust*.so 2>/dev/null | head -1)
if [ -n "${SO:-}" ]; then
  cp "$SO" "$ROOT/packages/hexfield/python/hexfield/"
  echo "mirrored $(basename "$SO") into packages/hexfield/python/hexfield/"
else
  echo "WARN: could not locate the built hexfield _rust extension in $SITE" >&2
fi
ls -la "$ROOT"/packages/hexfield/python/hexfield/_rust*.so
