# ChurnOps inference API
# Dependencies are installed before the (frequently changing) code for layer caching.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

# libgomp1: OpenMP runtime required by LightGBM. curl: container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src ./src
COPY app ./app
# Exported champion (produced by `make promote`). Lineage lives in its MLmodel metadata.
COPY artifacts/champion_model ./artifacts/champion_model

# Defaults are safe for a single container; override via env (never bake secrets in).
ENV MODEL_URI=/app/artifacts/champion_model \
    DATABASE_URL=sqlite:////tmp/churnops.db \
    PORT=8000

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# `exec` keeps uvicorn as PID 1 so it receives SIGTERM; ${PORT} lets PaaS hosts pick the port.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
