# One image recipe for the five Python services. Build context is the repo root:
#   docker build -f infra/Dockerfile --build-arg SERVICE=<name> .
# SERVICE is gateway, agent, corpus, ingestion or math_verifier. It selects the dependency group
# in pyproject.toml (one lock for all), the source that ships (infra/assemble-service.sh lists it)
# and the start command, so an image holds what its service imports and nothing else.

# The uv image tag is pinned here. Bump it together with the CI uv version in
# .github/actions/setup-python/action.yml; the two are kept equal by hand, in a PR.
ARG UV_VERSION=0.13.6
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim AS builder
ARG SERVICE
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
# Dependencies come first and depend only on pyproject.toml and uv.lock, so a source edit
# reuses this layer. The project itself is not installed (the services run from source).
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    test -n "$SERVICE" || { echo "pass --build-arg SERVICE=<service>" >&2; exit 1; }; \
    uv sync --frozen --no-default-groups --group "$SERVICE" --no-install-project

# Picks this service's source, start command and health probe out of the repo.
FROM python:3.12-slim AS assemble
ARG SERVICE
COPY infra/assemble-service.sh /assemble-service.sh
COPY services/ /repo/services/
COPY shared/ /repo/shared/
COPY config/ /repo/config/
COPY prompts/ /repo/prompts/
RUN sh /assemble-service.sh "$SERVICE" /repo /out

FROM python:3.12-slim
ARG SERVICE
WORKDIR /app
RUN adduser --disabled-password --gecos "" appuser
COPY --from=builder /app/.venv /app/.venv
COPY --from=assemble /out/app/ /app/
COPY --from=assemble /out/bin/ /usr/local/bin/
# The gateway writes these relative to /app (IELTS attempts, the AI spend ledger, the turn log).
# /app is root-owned, so without them the non-root user cannot create the directories and the
# gateway exits at startup. docker-compose bind-mounts host directories over them in dev.
RUN if [ "$SERVICE" = gateway ]; then \
      mkdir ielts-local .eval-ledger .turn-log && chown appuser ielts-local .eval-ledger .turn-log; \
    fi
ENV PYTHONPATH=/app PATH="/app/.venv/bin:$PATH"
USER appuser
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s CMD ["healthcheck"]
CMD ["serve"]
