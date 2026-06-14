FROM python:3.12-slim AS builder
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

FROM python:3.12-slim
WORKDIR /app
RUN adduser --disabled-password --gecos "" appuser
COPY --from=builder /app/.venv /app/.venv
COPY services/gateway/ services/gateway/
COPY services/agent/ services/agent/
COPY shared/ shared/
COPY config/ config/
COPY prompts/ prompts/
ENV PYTHONPATH=/app PATH="/app/.venv/bin:$PATH"
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s CMD curl -f http://localhost:8000/healthz || exit 1
CMD ["uvicorn", "services.gateway.src.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
