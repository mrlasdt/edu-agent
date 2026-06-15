# GRE Tutor — Agentic AI Engineering Reference Implementation

A production-grade GRE test-prep tutor that demonstrates advanced agentic AI engineering techniques: RAG pipelines, MCP servers, multi-model cost tiering, five-layer guardrails, eval-as-CI, and k8s deployment — all TDD'd from scratch.

## What it does

A Candidate submits a GRE Quant or Analytical Writing question. The system:

- **Tutors by default** — scaffolds toward the answer via Socratic hints, never revealing it
- **Solves on request** — returns a fully worked solution (Quant) or a scored sample essay (AW)
- **Verifies math** — every Quant answer is checked by a sympy sandbox before emission; wrong answers retry → escalate → degrade gracefully
- **Cites sources** — AW responses reference numbered passages from a curated ETS corpus; uncited claims are blocked or stripped
- **Matches your style** — opt-in personal style feature biases essay voice toward the Candidate's own writing while preserving ETS structural rubric

## Architecture

```
                        ┌─────────────────────────────────────┐
  Candidate / Admin     │           Gateway Service            │
  (Web UI / MCP)  ────► │  Validate → Enrich → Classify → Route│
                        └────────┬──────────────┬─────────────┘
                                 │              │
                    ┌────────────▼───┐   ┌──────▼──────────┐
                    │  Quant Agent   │   │    AW Agent      │
                    │  gpt-4o main   │   │  gpt-4o main     │
                    │  o1 escalation │   │  gpt-4o-mini     │
                    └────┬───────────┘   │  style critic    │
                         │               └──────┬───────────┘
                         │                      │
               ┌─────────▼──────┐    ┌──────────▼──────────┐
               │ Math Verifier  │    │   Corpus Service      │
               │  MCP Server    │    │  Qdrant hybrid search │
               │  (sympy)       │    │  BGE-M3 + reranker    │
               └────────────────┘    └──────────┬────────────┘
                                                │
                                     ┌──────────▼────────────┐
                                     │  Ingestion Service     │
                                     │  PDF/DOCX/TXT → chunks │
                                     │  → embed → index       │
                                     └───────────────────────┘
```

### Services

| Service | Port | Role |
|---|---|---|
| `gateway` | 8000 | External API, Orchestrator, guardrails, rate limiting |
| `agent` | 8002 | QuantAgent + AWAgent, Skills, post-gen critics |
| `corpus` | 8001 | Qdrant retrieval (hybrid RRF + rerank), 3-tier ACL |
| `ingestion` | 8003 | Async document pipeline (parse → chunk → embed → index) |
| `math-verifier` | 8090 | FastMCP server wrapping sympy sandbox |
| `tei-embedder` | 8080 | BGE-M3 text embeddings (Hugging Face TEI) |

### Three-tier corpus

Every retrieval enforces ACL:

```
Global corpus    ← Admin uploads (curated ETS materials)    visible to all
School corpus    ← Teacher uploads (per-school)              v2 only
Candidate corpus ← Candidate uploads (private notes)         visible to owner only
```

### Model tiers (configurable via `config/model_config.dev.yaml`)

| Role | Default model | Swap to Anthropic |
|---|---|---|
| Orchestrator (enrich/classify) | `gpt-4o-mini` | `claude-haiku-4-5` |
| Quant / AW agents | `gpt-4o` | `claude-sonnet-4-6` |
| Math escalation | `o1` | `claude-opus-4-7` |
| Style critic, per-PR judge | `gpt-4o-mini` | `claude-haiku-4-5` |

Every role is one config-line change. LiteLLM routes to the correct provider.

## Quickstart (Phase 1 — localhost)

### Prerequisites

- Python 3.12+, `uv`, Node 18+, Docker Desktop

### 1. Clone and install

```bash
git clone <repo>
cd edu-agent

# Python deps
uv sync --extra dev

# Frontend deps
cd web && npm install && cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY at minimum; all other values have localhost defaults
```

### 3. Start infrastructure

```bash
docker compose -f infra/docker-compose.yml up -d
docker compose -f infra/docker-compose.yml ps   # all services should be healthy
```

Services started: Postgres, Qdrant, Langfuse (LLM observability at http://localhost:3000), BGE-M3 TEI embedder, Math Verifier MCP server.

### 4. Run tests

```bash
# Python tests (156)
uv run pytest tests/ -q

# Frontend tests (55)
cd web && npm test

# Eval slice (Quant + AW, 10% sample — requires OPENAI_API_KEY)
uv run python evals/run_eval.py --suite quant --suite aw --sample 0.1
```

### 5. Start services

```bash
# Gateway (terminal 1)
uv run uvicorn services.gateway.src.gateway.main:app --port 8000 --reload

# Agent service (terminal 2)
uv run uvicorn services.agent.src.agent.main:app --port 8002 --reload

# Corpus service (terminal 3)
uv run uvicorn services.corpus.src.corpus.main:app --port 8001 --reload

# Ingestion service (terminal 4)
uv run uvicorn services.ingestion.src.ingestion.main:app --port 8003 --reload

# Web UI (terminal 5)
cd web && npm run dev
```

### 6. Ingest the sample corpus

```bash
# Via Admin MCP server (Claude Desktop) or curl:
curl -F "file=@path/to/ets-guide.pdf" \
     -F "test_type=gre" -F "tier=global" \
     http://localhost:8003/admin/documents
```

## MCP Servers

Two MCP servers are available for Claude Desktop (or any MCP-compatible client):

**Candidate server** (port 8091) — search the corpus, fetch practice questions, get model essays:
```bash
uv run python -m mcp_servers.candidate_server
```

**Admin server** (port 8092) — trigger ingestion, check job status:
```bash
uv run python -m mcp_servers.admin_server
```

Add to Claude Desktop's `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "gre-tutor-candidate": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_servers.candidate_server"]
    }
  }
}
```

## Eval pipeline

```bash
# Full run (requires OPENAI_API_KEY)
uv run python evals/run_eval.py --suite quant --suite aw --sample 1.0

# CI-style 10% sample
uv run python evals/run_eval.py --suite quant --suite aw --sample 0.1
```

- **Quant judge**: deterministic sympy comparison — no LLM call
- **AW judge**: GPT-4o prompted with ETS rubric + 6 anchor essays per prompt; calibrates score drift on every run
- **Regression check**: exits non-zero when any metric drops >5% vs `evals/baseline.json`
- **Human review queue**: boundary AW scores (3.5/4.5) and `verifier_fail` events are flagged; `gotohuman` integration in v2

## Guardrails (5 layers)

| Layer | What it checks |
|---|---|
| Input validation | Schema, size, content safety (OpenAI Moderation API) |
| Enrichment | Haiku-tier LLM detects incomplete questions → `ClarificationTurn` |
| Generation | Tutor-mode answer-leak check (Haiku); citation completeness (deterministic) |
| Output | PII regex redaction; content safety on output text |
| Rate/quota | 60 turns/min and 200 turns/day per Candidate (in-memory; Redis in v2) |

Every check emits a `guardrail.{layer}.{name}.{pass|fail}` structured log event with trace ID.

## Deployment (Phase 2 — k8s)

k8s manifests are in `infra/k8s/` and were validated in CI. To apply to a cluster:

```bash
# Staging (auto-deploys on merge to main via GitHub Actions)
kubectl apply -f infra/k8s/

# Verify
kubectl rollout status deployment/gateway
kubectl rollout status deployment/agent
```

**CI/CD** (`.github/workflows/`):
- Every PR: lint + type check + unit tests + path-gated eval slice (10% sample, only when prompts/agents/evals/config change)
- Merge to main: build images → push to GHCR → deploy to staging
- Tagged release: 5% canary → 30-min monitor → auto-rollback on error spike → promote to 100%

## Extending to new test types (IELTS, GMAT, …)

The `AgentRegistry` is the extension seam. Adding a new test type requires only:

1. Subclass `BaseAgent` and implement `run(session, message)`
2. Register: `registry.register("ielts", "writing", IELTSWritingAgent)`
3. Add prompts under `prompts/ielts/writing/`
4. Add corpus documents (tagged `test_type=ielts`)
5. Add golden set: `evals/golden/ielts_writing.jsonl`

Gateway, Corpus Service, Ingestion Service, Math Verifier, and the eval harness require no changes.

## Repository layout

```
edu-agent/
├── services/
│   ├── gateway/          # FastAPI — external API, Orchestrator, guardrails
│   ├── agent/            # FastAPI — QuantAgent + AWAgent + Skills + critics
│   ├── corpus/           # FastAPI — Qdrant retrieval (RRF + rerank + ACL)
│   ├── ingestion/        # FastAPI — async document pipeline
│   └── math_verifier/    # FastMCP — sympy sandbox MCP server
├── shared/               # Pydantic models, config loader, observability
├── mcp_servers/          # Candidate + Admin MCP servers
├── web/                  # Vite + React — Candidate UI + Admin UI
├── evals/                # Eval harness, golden sets, judges, regression check
├── prompts/              # Versioned system prompts (quant_agent, aw_agent)
├── config/               # model_config.{dev,staging,prod}.yaml
├── infra/
│   ├── docker-compose.yml  # Phase 1: full local stack
│   └── k8s/                # Phase 2: Deployment + Service manifests
├── .github/workflows/    # ci.yml + deploy.yml
├── docs/
│   ├── adr/              # 11 Architectural Decision Records
│   └── design-log/       # Trade-space analysis behind each ADR
└── CONTEXT.md            # Domain glossary (canonical vocabulary)
```

## Key decisions

All major architectural choices are recorded as ADRs in `docs/adr/`. Highlights:

| ADR | Decision |
|---|---|
| 0001 | Three-tier RAG corpus (Global / School / Candidate) with Schools as tenants |
| 0002 | Two-mode agent architecture (Basic subject-router + Advanced planner, v2) |
| 0004 | RAG stack: BGE-M3 + Qdrant + hybrid RRF + BGE-reranker |
| 0005 | Application is GRE test prep (not general homework help) |
| 0007 | LiteLLM abstraction; OpenAI default, Anthropic configurable per role |
| 0009 | Trunk-based development; k8s; Phase 1 Docker Compose |
| 0011 | Microservices architecture (monorepo); `AgentRegistry` extensibility seam |
