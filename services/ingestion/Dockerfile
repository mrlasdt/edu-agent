FROM python:3.12-slim AS builder
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project

FROM python:3.12-slim
WORKDIR /app
RUN adduser --disabled-password --gecos "" appuser
COPY --from=builder /app/.venv /app/.venv
COPY services/ingestion/ services/ingestion/
COPY shared/ shared/
ENV PYTHONPATH=/app PATH="/app/.venv/bin:$PATH"
USER appuser
EXPOSE 8003
# The slim base ships no curl; probe /healthz with python (always present).
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8003/healthz').getcode()==200 else 1)" || exit 1
CMD ["uvicorn", "services.ingestion.src.ingestion.main:app", "--host", "0.0.0.0", "--port", "8003"]
