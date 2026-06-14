FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only what's needed for math-verifier
COPY pyproject.toml .
COPY services/math_verifier/ services/math_verifier/
COPY shared/ shared/

# Install deps (only math-verifier relevant ones)
RUN uv sync --no-dev --no-install-project

ENV PYTHONPATH=/app
ENV MATH_VERIFIER_PORT=8090

EXPOSE 8090

CMD ["uv", "run", "python", "-m", "services.math_verifier.src.math_verifier.server"]
