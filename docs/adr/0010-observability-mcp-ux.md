# Observability, MCP, latency budget, and web UX

**Observability:** Langfuse for LLM-specific tracing (prompt version, tokens, cost, cache hit/miss, latency per call); Prometheus + Grafana for infra metrics; Sentry for application errors; structured JSON logs with trace IDs; a single eval dashboard (Grafana). Langfuse runs in Docker Compose in Phase 1, managed instance in Phase 2.

**MCP:** two MCP servers ship in v1. The **Candidate MCP server** exposes `search_corpus`, `get_question`, `get_model_essay`. The **Admin MCP server** exposes `ingest_document`, `list_ingestion_jobs`, `get_job_status`. Both are thin wrappers around the same service layer as the FastAPI backend. Primary corpus management is the **Admin UI** (a minimal web form for file upload and job status) — not dependent on MCP or external clients.

**Latency:** all main-agent turns stream first tokens. Orchestrator + retrieval run before the stream; style critic and citation checker run post-generation and gate final emission. Target TTFT: <2s for Quant Tutor, <4s for AW Solve.

**Web UX:** Vite + React + AI SDK. Session starts with a subject picker (Quant / AW). Mode switch is a visible button in the input bar. Citations render as inline `[N]` with a collapsible Sources panel. Upload is a drag-and-drop overlay. Style opt-in is a preferences-panel toggle, visible only after ≥2 essays uploaded.

## Status

Accepted.

## Considered options

- **Langfuse vs Helicone vs DIY LLM logging** — Langfuse chosen: open-source, self-hostable, richer prompt-versioning and A/B comparison UI than Helicone. DIY loses the dashboard.
- **Admin MCP server only (no Admin UI)** — rejected: corpus management shouldn't depend on Claude Desktop availability. Admin UI is independent and more product-realistic.
- **MCP in v2** — rejected: MCP is a named portfolio requirement and a ~2-3 day build. The Candidate MCP server is a meaningful demo (Claude Desktop can search the GRE corpus). In scope for v1.
- **Next.js for frontend** — rejected for v1 (ADR-0009). The UX shape above (single-page streaming chat) doesn't need SSR.

## Consequences

- **Langfuse is a v1 infra dependency.** A `langfuse` service in Docker Compose (Phase 1) and a managed or self-hosted instance (Phase 2). The LiteLLM callback integration takes ~30 lines of code.
- **MCP servers are additive to the service layer**, not standalone services. They import from the same Python modules as FastAPI routes. No duplicate business logic.
- **Admin UI is minimal by design:** file upload form + job-status table + error log. Not a product; a tool. Built with the same React stack as the frontend (shared component library).
- **Thinking states in the UX are non-optional.** Verifier-fail events and retrieval latency are visible to the Candidate as status lines ("Checking your answer...", "Finding references..."). Silent long pauses are worse than named waiting.
- **Citations in AW output are a UI contract, not just a prompt feature.** The backend must return structured citation metadata alongside the text so the frontend can render inline `[N]` links and the Sources panel. This affects the streaming response format (citation metadata emitted at end of stream, not mid-stream).
- **Latency budgets are guiding targets, not hard SLOs in v1.** They become SLOs when the eval pipeline has enough production data to set a baseline. Watch the p95, not just the mean.
- **The style opt-in toggle's "≥2 essays" gate is client-enforced** (toggle hidden) and server-enforced (Personal style Skill no-ops with <2 essays, returns Canonical style). Both guards needed.
