# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# System deps:
# - curl + ca-certificates: install uv + HTTPS
# - git: required for Snapshotter (repo clone)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

ENV PATH="/root/.local/bin:${PATH}"

# Copy project metadata + README first (hatchling validates readme during build)
COPY pyproject.toml uv.lock README.md ./

# Copy source (uv sync will build/install the local project)
COPY src ./src

# Install dependencies (runtime only in image)
RUN uv sync --frozen --no-dev

# IMPORTANT: run with the venv interpreter created by uv (otherwise `python` can't import snapshotter)
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

CMD ["python", "-m", "snapshotter.cli"]
