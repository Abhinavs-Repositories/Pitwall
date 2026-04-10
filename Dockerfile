# Multi-stage build — keeps final image lean
# Stage 1: dependency installer
FROM python:3.11-slim AS deps

WORKDIR /app

# Install build deps (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Create data directory for SQLite cache
RUN mkdir -p /app/data/cache

# Non-root user for security
RUN useradd -m -u 1000 pitwall
RUN chown -R pitwall:pitwall /app
USER pitwall

# Environment defaults (override via docker-compose or -e flags)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    SQLITE_DB_PATH=/app/data/cache.db \
    CACHE_DIR=/app/data/cache

EXPOSE 8000

# Default: run the FastAPI backend
# Override CMD in docker-compose for the Streamlit service
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
