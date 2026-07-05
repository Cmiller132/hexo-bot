# syntax=docker/dockerfile:1
# Hexo showcase image — FastAPI game server + static web UI + CPU inference.
#
# BUILD CONTEXT MUST BE THE REPO ROOT (the crates reference the workspace-root
# Cargo.toml/Cargo.lock). docker-compose.yml in this directory sets
# `context: ../..` accordingly; a manual build is:
#
#   docker build -f apps/showcase/Dockerfile .        # from the repo root
#
# Stage 1 builds release wheels for the three pyo3 crates with maturin.
# The crates are NOT abi3, so the wheels are CPython-version-specific
# (cp312) — the builder therefore bases on python:3.12 and installs a pinned
# rust toolchain via rustup (rust:slim-bookworm ships Debian's python 3.11,
# whose wheels would not install into the 3.12 runtime).
#
# Each wheel ships the FULL python package (maturin `python-source = "python"`
# bundles packages/<crate>/python/<pkg>/ plus the compiled `_rust` extension
# inside it), so the runtime stage installs the wheels alone — no package
# source trees are copied.
#
# The image contains NO checkpoints: bots.toml + weights are a read-only
# volume (see docker-compose.yml). HEXFIELD_* arch env is not baked either —
# radius/arch belong with the mounted weights and come from compose.
#
# GPU variant: Dockerfile.xpu (Intel Arc / torch XPU, opt-in via
# docker-compose.xpu.yml). Its builder stage mirrors this one — keep in sync.

##############################################################################
# Stage 1 — builder: rustup (pinned) + maturin, wheels for the three crates
##############################################################################
FROM python:3.12-slim-bookworm AS builder

# rustc needs a C linker; zstd-sys (hexo_utils) compiles C via cc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Pinned rust toolchain (matches the toolchain the wheels were verified with).
ARG RUST_VERSION=1.95.0
RUN curl -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"
ENV PATH="/root/.cargo/bin:${PATH}"

ARG MATURIN_VERSION=1.14.1
RUN pip install --no-cache-dir "maturin==${MATURIN_VERSION}"

WORKDIR /src

# Workspace root + the three member crates (paths per the workspace manifest).
COPY Cargo.toml Cargo.lock ./
COPY packages/hexo_engine packages/hexo_engine
COPY packages/hexo_utils packages/hexo_utils
COPY packages/hexfield packages/hexfield

# --locked: build exactly the dependency set in the committed Cargo.lock.
RUN maturin build --release --locked -o /wheels -i python3.12 \
        -m packages/hexo_engine/Cargo.toml \
    && maturin build --release --locked -o /wheels -i python3.12 \
        -m packages/hexo_utils/Cargo.toml \
    && maturin build --release --locked -o /wheels -i python3.12 \
        -m packages/hexfield/Cargo.toml

##############################################################################
# Stage 2 — runtime: python 3.12 + CPU torch + wheels + the showcase app
##############################################################################
FROM python:3.12-slim-bookworm AS runtime

# curl: container healthcheck. sqlite3: ops runbook (db shell / .backup).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# CPU-only torch from the PyTorch index (the +cpu local version exists only
# there, so the pin is unambiguous); everything else from PyPI. Pins match the
# venv the showcase test suite passed under (see apps/showcase/requirements.txt
# floors: fastapi>=0.110, uvicorn[standard]>=0.29). lz4/zstandard are
# install_requires of the hexo_utils wheel, pinned here for reproducibility.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.1+cpu \
        fastapi==0.139.0 \
        "uvicorn[standard]==0.50.0" \
        numpy==2.5.0 \
        lz4==4.4.5 \
        zstandard==0.25.0 \
        /wheels/*.whl \
    && rm -rf /wheels

WORKDIR /app

# The server package (PYTHONPATH import, not pip-installed), the static web
# UI, the search-profile configs the server reads at startup, and the
# catalogue example for reference. Paths mirror the repo layout because the
# SHOWCASE_* defaults in server/showcase/config.py are repo-root-relative and
# WORKDIR is the in-image repo root.
COPY apps/showcase/server apps/showcase/server
COPY apps/showcase/web apps/showcase/web
COPY apps/showcase/bots.example.toml apps/showcase/bots.example.toml
COPY configs configs

ENV PYTHONPATH=/app/apps/showcase/server \
    SHOWCASE_DB_PATH=/data/showcase.db

# Non-root. /data is the SQLite volume mountpoint; created + chowned here so
# a fresh named volume inherits app-writable ownership.
RUN useradd --uid 10001 --user-group --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown app:app /data
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["curl", "-fsS", "http://127.0.0.1:8000/healthz"]

# Single uvicorn process by design: SHOWCASE_WORKERS controls the model
# worker-process pool (bots.py); the web process itself never imports torch.
CMD ["uvicorn", "showcase.app:app", "--host", "0.0.0.0", "--port", "8000"]
