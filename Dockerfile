# syntax=docker/dockerfile:1.7
#
# BioAgents — multi-agent AI system for computational biology
#
# Multi-stage build:
#   1. `builder` — resolves Python dependencies with uv into /app/.venv
#   2. `runtime` — slim image with the venv, Node.js + rdkit-agent CLI, and the
#                  application source
#
# The resulting image runs the FastAPI web UI on port 8000 with all agent
# tools available out of the box. The in-app code sandbox defaults to the
# built-in LocalPythonExecutor so the container is self-sufficient; mount the
# host Docker socket if you want nested DockerExecutor isolation.
# =============================================================================

ARG PYTHON_VERSION=3.12
ARG NODE_MAJOR=20

# ---------------------------------------------------------------------------
# Stage 1 — builder (resolves Python deps)
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Build toolchain needed for native wheel fallbacks (rdkit, numpy, etc.).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        pkg-config \
        libffi-dev \
        libssl-dev; \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv>=0.5.0"

# Resolve Python deps first (without the local project) for layer caching.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now copy the source tree and finish the install (installs the local project).
COPY bioagents/ ./bioagents/
COPY frontend/ ./frontend/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Chemistry extras — optional at runtime but expected by drug-discovery
# workflows (`rdkit` Python bindings).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-cache-dir rdkit

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG NODE_MAJOR

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:/usr/local/bin:${PATH}" \
    VIRTUAL_ENV="/app/.venv" \
    PYTHONPATH="/app" \
    BIOAGENTS_SANDBOX_DIR="/app/sandbox_workdir" \
    USE_LOCAL_EXECUTOR="true" \
    HOST="0.0.0.0" \
    PORT="8000"

WORKDIR /app

# Runtime OS deps + Node.js + rdkit-agent CLI + docker CLI (for the optional
# nested DockerExecutor flow when the host socket is mounted).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        tini \
        gnupg \
        libgomp1 \
        libstdc++6 \
        libxrender1 \
        libxext6 \
        libsm6 \
        docker.io; \
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -; \
    apt-get install -y --no-install-recommends nodejs; \
    npm install -g --omit=dev rdkit-agent; \
    rdkit-agent --version || true; \
    apt-get purge -y --auto-remove gnupg; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /root/.npm

# Pull in the uv-managed virtualenv built in stage 1.
COPY --from=builder /app/.venv /app/.venv

# Application source (large vendored repos excluded via .dockerignore).
COPY bioagents/ ./bioagents/
COPY frontend/ ./frontend/
COPY use_cases/ ./use_cases/
COPY playbooks/ ./playbooks/
COPY config/ ./config/
COPY data/ ./data/
COPY pyproject.toml uv.lock README.md ./
COPY docker/entrypoint.sh /usr/local/bin/bioagents-entrypoint
RUN chmod +x /usr/local/bin/bioagents-entrypoint

# Runtime working dirs — also declared as volumes so users can persist them.
RUN mkdir -p \
        /app/generated_artifacts \
        /app/uploads \
        /app/experiment_runs \
        /app/sandbox_workdir \
        /app/logs \
        /app/.bioagents_cache

# Drop privileges. The `docker` group membership lets the process reach the
# host Docker socket when it is mounted; harmless when it is not.
RUN groupadd --system --gid 1000 bioagents \
    && useradd --system --uid 1000 --gid 1000 --create-home --home-dir /home/bioagents bioagents \
    && usermod -aG docker bioagents \
    && chown -R bioagents:bioagents /app /home/bioagents
USER bioagents

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

VOLUME ["/app/generated_artifacts", "/app/uploads", "/app/experiment_runs", "/app/sandbox_workdir", "/app/logs"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/bioagents-entrypoint"]
CMD ["serve"]
