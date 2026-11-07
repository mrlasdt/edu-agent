# Microservices architecture (monorepo)

The system is decomposed into five persistent microservices and one CLI tool, all in a single repository. Services communicate over HTTP (FastAPI). The **Gateway** is the only external-facing service; all others are cluster-internal. The **Math Verifier** is a local MCP server (MCP Python SDK) running as a separate process — the Agent Service is its MCP client. The Eval harness is a CLI tool that runs against the live services in CI and on demand; it is not a persistent service.

## Services

| Service | Responsibility | Exposes |
|---|---|---|
| **gateway** | External API (HTTP + SSE), auth, rate-limiting, session management, Orchestrator logic (validate → enrich → classify → route), Admin UI backend | `/chat`, `/upload`, `/admin/*` |
| **agent** | Quant Agent + AW Agent via plugin registry, Skill loading, post-gen critics (style critic, citation checker), streaming to gateway | internal HTTP |
| **corpus** | Qdrant interface, hybrid search (BGE-M3 dense + sparse + RRF), BGE reranker, ACL enforcement, citation metadata | internal HTTP |
| **ingestion** | Document parsing (PDF/DOCX/TXT), chunking, embedding, Qdrant indexing, job management API; large files are indexed by an in-process background task (see "Amendment: ingestion without a queue worker") | internal HTTP |
| **math-verifier** | sympy sandbox, local MCP server; accepts `verify_math(expression, expected)` tool calls | MCP protocol (Unix socket in Phase 1, cluster-internal in Phase 2) |

**Not a service:** the Eval harness (`evals/`) is a CLI tool invoked by CI and nightly cron. It calls the Agent Service and judges directly. Spinning up a persistent eval service would add operational surface for no benefit.

**Admin UI** lives inside the Gateway service as protected routes. It calls the Ingestion Service's job-management API. Too thin to warrant its own service.

## Extensibility contract (IELTS, GMAT, TOEFL, etc.)

The Agent Service uses a **plugin registry** pattern: agents register by `(test_type, section)` key. The Orchestrator routes by looking up `(session.test_type, session.section)` in the registry. Adding a new test type means:

1. Register new agent class(es): `IELTSWritingAgent`, `GMATQuantAgent`, etc.
2. Add prompts under `prompts/{test_type}/{section}/`
3. Add corpus documents to Global corpus with `test_type` metadata (retrieval ACL already filters by payload)
4. Add golden set: `evals/golden/{test_type}_{section}.jsonl`
5. Update the session-start UI to surface the new test type

Gateway, Corpus Service, Ingestion Service, Math Verifier MCP server, and Eval harness are all test-agnostic and require no changes. The `test_type` dimension is a first-class field on session state, corpus documents, and the Admin UI's upload form from v1.

## Status

Accepted. Supersedes the monolithic service implied by the earlier ADRs; all service-boundary decisions are additive to the prior architecture decisions (ADR-0001 through ADR-0010).

## Considered options

- **Monolith with internal modules** — rejected: the math verifier needs process-level isolation; corpus retrieval has a natural scaling unit different from agent turns; ingestion is purely async and doesn't belong in the request path.
- **Math Verifier as HTTP microservice** — rejected in favour of MCP server: loses the MCP demo surface, gains nothing at our scale.
- **Math Verifier as in-process Skill** — rejected: sympy sandbox isolation requires process separation; a crash in the verifier would take down the Agent Service.
- **Message broker (Kafka, RabbitMQ) between request-path services** — rejected: the request path is synchronous (Gateway → Agent → Corpus, Agent → Math Verifier). Message brokers add latency and operational complexity without benefit on synchronous call chains. Async ingestion was meant to use the existing Postgres pgmq queue; see "Amendment: ingestion without a queue worker" for what is built.
- **Separate eval service** — rejected: the eval harness needs to be invocable as a CLI tool from CI. Wrapping it in a persistent service adds an HTTP layer and a deployment dependency for no benefit. The harness calls Agent and Corpus services directly.
- **Admin UI as a separate service** — rejected: too thin (upload form + job table). Belongs in Gateway as protected routes calling Ingestion API.

## Consequences

- **Each service has its own Dockerfile and its own entry in Docker Compose.** (k8s manifests were planned and later removed unapplied, ADR-0009.)
- **Services share one Postgres instance** (different schemas or tables; no cross-schema foreign keys). Alembic migrations are per-service under `services/{name}/migrations/`.
- **Services share one Qdrant instance** (one collection, ACL via payload filters — ADR-0004).
- **The Agent Service is an MCP client.** It connects to the Math Verifier MCP server at startup and holds the connection. If the Math Verifier is unreachable, Quant turns degrade to "unable to verify" (verifier-fail event) rather than failing the whole service.
- **The plugin registry is the v1 extension seam.** It must be designed before writing the first agent class — not retrofitted. The registry interface: `AgentRegistry.register(test_type: str, section: str, agent_cls: type[BaseAgent])`. The Orchestrator calls `AgentRegistry.get(test_type, section)` to resolve the agent for a session. Agents not registered → Orchestrator returns a "section not available" clarification turn.
- **`test_type` is on session state from day one.** v1 supports only `test_type=gre`; v2 adds `ielts`, `gmat`, etc. The field is present in all session state, corpus document metadata, and the Admin UI upload form even though only one value is valid in v1.
- **Inter-service auth is internal network trust in Phase 1** (Docker Compose network). Phase 2 uses mTLS or an internal JWT between services — a k8s secret. Not built in Phase 1.
- **LiteLLM is used only in the Agent Service and Eval harness.** No other service makes LLM calls. This keeps the model config surface area contained.

## Amendment: ingestion without a queue worker (2026-11)

The decision above named a pgmq queue and a worker for large uploads. Neither was ever built: `enqueue_ingestion_job` only wrote a record into a Python dict, so a file of 1 MB or more got `202 queued` and was never indexed, and the test that should have caught it mocked the stub. The code now does the smallest thing that is true:

- **Large files are indexed by a FastAPI `BackgroundTasks` task**, the mechanism the gateway's IELTS grading already uses. `POST /admin/documents` still answers files under 1 MB with `200 completed` after indexing them, and larger ones with `202 queued` at once. The task then parses (off the event loop), chunks, embeds and indexes, and `GET /admin/documents/{job_id}` reports `processing`, then `completed` with `chunks_indexed`, or `failed` with `error_stage` (`parse` or `index`) and `error`.
- **Job records are per process.** They live in a dict in the service's memory: a restart loses them, and a second replica would not see them. The service therefore stays a single replica (one container in `infra/docker-compose.yml`). A task that is running when the process stops is lost with it, and the upload has to be repeated; that is safe, because of the next point.
- **Point ids are deterministic.** A chunk's Qdrant id is `uuid5(namespace, "<tier>:<test_type>:<source_uri>:<chunk_index>")`, so uploading a document again (same filename, tier and test type) replaces its points instead of adding a second copy, and after the upsert every other point of that document (same filename, tier and test type) is deleted: the tail of a longer old version, and any point stored under a random id before ids were deterministic. The seed job relies on this and uploads every note on every start instead of asking Qdrant what is indexed. The namespace in `ingestion/main.py` must never change.

A collection filled before this change (random ids) needs no manual cleanup: the first upload of each document removes its legacy points as part of the replace, so the seed job's first start on an existing volume leaves one copy of each note, not two. Documents that are never uploaded again keep their legacy points.

**Still open.** A durable queue (pgmq or arq) with a worker and a Postgres job table, which would survive restarts and allow more than one replica. Until then the "pgmq" wording above describes the intent, not the code.
