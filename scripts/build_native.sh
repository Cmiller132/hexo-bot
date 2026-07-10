#!/usr/bin/env bash
# Build the four native crates (hexo_engine, hexo_utils, shrimp, hexfield_eq)
# with maturin develop --release into the active venv ($HEXO_VENV, default
# .venv at repo root). --release is mandatory: a debug featurizer/search crate
# is ~10x slower.
#
# hexo_engine and hexo_utils resolve from the venv site-packages after install.
# shrimp and hexfield_eq are imported from the source tree via PYTHONPATH
# (never pip-installed; see README), so their compiled extensions are mirrored
# next to the Python packages.
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

for crate in hexo_engine hexo_utils shrimp hexfield_eq; do
  echo "=== building $crate ==="
  maturin develop --release -m "$ROOT/packages/$crate/Cargo.toml"
done

# Mirror the shrimp and hexfield_eq extensions into the source tree so
# PYTHONPATH imports find them (maturin installs them into the venv, but both
# packages are imported from-tree).
SITE="$("$HEXO_VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
for pkg in shrimp hexfield_eq; do
  SO=$(ls "$SITE/$pkg"/_rust*.so 2>/dev/null | head -1)
  if [ -n "${SO:-}" ]; then
    cp "$SO" "$ROOT/packages/$pkg/python/$pkg/"
    echo "mirrored $(basename "$SO") into packages/$pkg/python/$pkg/"
  else
    echo "WARN: could not locate the built $pkg _rust extension in $SITE" >&2
  fi
  ls -la "$ROOT/packages/$pkg/python/$pkg/"_rust*.so || true
done
