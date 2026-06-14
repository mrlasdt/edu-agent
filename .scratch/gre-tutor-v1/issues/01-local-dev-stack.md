Status: ready-for-agent

## What to build

Spin up the complete local development environment via Docker Compose. Every service runs on localhost with no cloud dependencies. This is the foundation every other slice depends on.

Services in Docker Compose: Postgres 16, Qdrant, Langfuse (LLM observability), and a BGE-M3 TEI inference server (embedder). Each service has a health check. A root `pyproject.toml` wires Python deps for all backend services. A `.env.example` documents every required env var.

## Acceptance criteria

- [ ] `docker compose up -d` starts all services with no errors
- [ ] `docker compose ps` shows all services healthy
- [ ] Postgres reachable at `localhost:5432`
- [ ] Qdrant reachable at `localhost:6333`
- [ ] Langfuse UI reachable at `localhost:3000`
- [ ] TEI embedder reachable at `localhost:8080`
- [ ] `.env.example` documents OPENAI_API_KEY, DATABASE_URL, QDRANT_URL, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY
- [ ] `uv sync` installs all Python deps without errors
- [ ] `pytest` runs (no tests yet, exit 0 with no test collection errors)

## Blocked by

None — can start immediately.
