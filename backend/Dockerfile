FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv to manage dependencies cleanly and fast
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml .
RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install --target=/build/deps -r requirements.txt


FROM python:3.12-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace

WORKDIR /workspace

# Install system dependencies (including runtime libpq for postgresql client)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/deps /usr/local/lib/python3.12/site-packages
COPY --from=builder /build/deps/bin /usr/local/bin

COPY . .

# Run application as non-root user for security compliance
RUN useradd -u 1001 appuser && \
    chown -R appuser:appuser /workspace
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
