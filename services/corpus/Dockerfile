FROM python:3.12-slim AS builder
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

FROM python:3.12-slim
WORKDIR /app
RUN adduser --disabled-password --gecos "" appuser
COPY --from=builder /app/.venv /app/.venv
COPY services/corpus/ services/corpus/
COPY shared/ shared/
ENV PYTHONPATH=/app PATH="/app/.venv/bin:$PATH"
USER appuser
EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=5s CMD curl -f http://localhost:8001/healthz || exit 1
CMD ["uvicorn", "services.corpus.src.corpus.main:app", "--host", "0.0.0.0", "--port", "8001"]
