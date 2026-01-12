# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install uv
RUN apt-get update -y && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -y \
    && echo "export PATH=\"$HOME/.local/bin:$PATH\"" >> /etc/profile

ENV PATH="/root/.local/bin:${PATH}"

# Copy metadata and lockfile first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies (runtime only in image)
RUN uv sync --frozen --no-dev

# Copy source code
COPY src ./src

CMD ["python", "-m", "snapshotter.cli"]
