# Microservices architecture (monorepo)

The system is decomposed into five persistent microservices and one CLI tool, all in a single repository. Services communicate over HTTP (FastAPI). The **Gateway** is the only external-facing service; all others are cluster-internal. The **Math Verifier** is a local MCP server (MCP Python SDK) running as a separate process — the Agent Service is its MCP client. The Eval harness is a CLI tool that runs against the live services in CI and on demand; it is not a persistent service.

## Services

| Service | Responsibility | Exposes |
|---|---|---|
| **gateway** | External API (HTTP + SSE), auth, rate-limiting, session management, Orchestrator logic (validate → enrich → classify → route), Admin UI backend | `/chat`, `/upload`, `/admin/*` |
| **agent** | Quant Agent + AW Agent via plugin registry, Skill loading, post-gen critics (style critic, citation checker), streaming to gateway | internal HTTP |
| **corpus** | Qdrant interface, hybrid search (BGE-M3 dense + sparse + RRF), BGE reranker, ACL enforcement, citation metadata | internal HTTP |
| **ingestion** | Async queue worker (pgmq), document parsing (PDF/DOCX/TXT), chunking, embedding, Qdrant indexing, job management API | internal HTTP |
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
- **Message broker (Kafka, RabbitMQ) between request-path services** — rejected: the request path is synchronous (Gateway → Agent → Corpus, Agent → Math Verifier). Message brokers add latency and operational complexity without benefit on synchronous call chains. Async ingestion uses the existing Postgres pgmq queue, which is sufficient.
- **Separate eval service** — rejected: the eval harness needs to be invocable as a CLI tool from CI. Wrapping it in a persistent service adds an HTTP layer and a deployment dependency for no benefit. The harness calls Agent and Corpus services directly.
- **Admin UI as a separate service** — rejected: too thin (upload form + job table). Belongs in Gateway as protected routes calling Ingestion API.

## Consequences

- **Each service has its own Dockerfile and its own entry in Docker Compose and k8s manifests.**
- **Services share one Postgres instance** (different schemas or tables; no cross-schema foreign keys). Alembic migrations are per-service under `services/{name}/migrations/`.
- **Services share one Qdrant instance** (one collection, ACL via payload filters — ADR-0004).
- **The Agent Service is an MCP client.** It connects to the Math Verifier MCP server at startup and holds the connection. If the Math Verifier is unreachable, Quant turns degrade to "unable to verify" (verifier-fail event) rather than failing the whole service.
- **The plugin registry is the v1 extension seam.** It must be designed before writing the first agent class — not retrofitted. The registry interface: `AgentRegistry.register(test_type: str, section: str, agent_cls: type[BaseAgent])`. The Orchestrator calls `AgentRegistry.get(test_type, section)` to resolve the agent for a session. Agents not registered → Orchestrator returns a "section not available" clarification turn.
- **`test_type` is on session state from day one.** v1 supports only `test_type=gre`; v2 adds `ielts`, `gmat`, etc. The field is present in all session state, corpus document metadata, and the Admin UI upload form even though only one value is valid in v1.
- **Inter-service auth is internal network trust in Phase 1** (Docker Compose network). Phase 2 uses mTLS or an internal JWT between services — a k8s secret. Not built in Phase 1.
- **LiteLLM is used only in the Agent Service and Eval harness.** No other service makes LLM calls. This keeps the model config surface area contained.
