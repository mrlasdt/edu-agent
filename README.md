# GRE Tutor — Agentic AI Engineering Reference Implementation

A GRE test-prep tutor built as a reference implementation of agentic AI engineering techniques: RAG pipelines, MCP servers, multi-model cost tiering, layered guardrails, eval-as-CI, and k8s deployment.

## Status

The backend services are built and covered by 189 Python tests (mocked LLM and Qdrant) plus 55 frontend unit tests. It is **not usable end to end yet**:

- **No web app shell.** `web/` holds React components and tests but no `index.html` or entry point, so `npm run build` fails and nothing renders in a browser. The only way in today is `POST /chat` on the gateway, which needs `OPENAI_API_KEY`.
- **RAG is dense-only.** The sparse vector, RRF hybrid and BGE reranker from ADR-0004 are designed but not wired.
- **`/chat` applies only the content-safety guardrail.** The enrichment and complexity steps in the Orchestrator are stubs. The answer-leak, PII, rate-limit and citation guardrails are implemented and unit tested but not yet applied in `/chat`.
- **Agent metadata is dropped.** `/generate` streams text only, so sources and verifier events never reach a UI.
- **Evals are small and CI never scores a real model.** 5 Quant and 2 AW golden items. The agent-in-the-loop harness (below) runs the real Quant agent, but CI only runs it offline against a fake model; real runs are manual and there is no measured baseline checked in.
- **No corpus is checked in**, and the k8s manifests have never been applied to a cluster.

The sections below describe the intended design; see `progress.txt` for the current state of each piece.

## What it does

A Candidate submits a GRE Quant or Analytical Writing question. The system:

- **Tutors by default** — scaffolds toward the answer via Socratic hints, never revealing it
- **Solves on request** — returns a fully worked solution (Quant) or a scored sample essay (AW)
- **Verifies math** — every Quant answer is checked by a sympy sandbox before emission; wrong answers retry → escalate → degrade gracefully
- **Cites sources** — AW responses reference numbered passages from a curated ETS corpus; uncited claims are blocked or stripped (the corpus must be ingested first; none is checked in)
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
| `corpus` | 8001 | Qdrant dense retrieval with 3-tier ACL (hybrid RRF + rerank designed, not wired) |
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

- Python 3.12+, `uv`, Node 20.19+ (CI uses 22), Docker Desktop

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

### 3. Start the stack

One command builds and starts everything — infrastructure **and** the four app services:

```bash
docker compose -f infra/docker-compose.yml up -d --build
docker compose -f infra/docker-compose.yml ps   # services should report healthy
```

| Service | URL | Role |
| --- | --- | --- |
| gateway | http://localhost:8000 | External API, orchestrator, guardrails |
| corpus | http://localhost:8001 | Qdrant retrieval (3-tier ACL) |
| agent | http://localhost:8002 | QuantAgent + AWAgent, skills, critics |
| ingestion | http://localhost:8003 | Document pipeline (parse → chunk → embed → index) |
| math-verifier | http://localhost:8090 | sympy verification MCP server (SSE) |
| langfuse | http://localhost:3000 | LLM observability |
| postgres / qdrant | 5432 / 6333 | Datastores |

The app services bind-mount the repo source and run `uvicorn --reload`, so code edits apply live with no rebuild. Rebuild only when dependencies (`pyproject.toml`) change:

```bash
docker compose -f infra/docker-compose.yml build
```

The BGE-M3 embedder is opt-in (amd64-only, emulated on Apple Silicon):

```bash
docker compose -f infra/docker-compose.yml --profile embeddings up -d
```

### 4. Run tests

```bash
# Python tests (189)
uv run pytest tests/ -q

# Frontend tests (55)
cd web && npm test

# Eval pipeline check (offline fake model, no network, no keys)
uv run python -m evals.agent_eval --mode offline
```

### 5. Frontend & local overrides (optional)

The app services already run inside the stack (step 3). The two things you may still run on the host:

```bash
# Web UI dev server (not containerized). There is no app shell yet, so this
# serves nothing useful until index.html / main.tsx exist.
cd web && npm run dev
```

To debug a single service outside Docker, stop it in the stack and run it directly — the in-container instance frees the port:

```bash
docker compose -f infra/docker-compose.yml stop agent
uv run uvicorn services.agent.src.agent.main:app --port 8002 --reload
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

The eval runs the **real agent** and judges what it produced (`evals/agent_eval/`):

```bash
uv run python -m evals.agent_eval                  # dry run (default): prints the plan, no calls
uv run python -m evals.agent_eval --mode offline   # whole pipeline on a deterministic fake model, no network
# live: free OpenRouter models, keys only via the environment (never printed or stored)
set -a; . path/to/keys.env; set +a
uv run python -m evals.agent_eval --mode live --provider free --max-items 5
uv run python -m evals.agent_eval --mode live --provider free+openai   # free first, OpenAI as fallback
uv run python -m evals.agent_eval --mode live --judge jev              # optional paid jev judge
```

What it measures (Quant, `QuantAgent` unmodified, run on golden-set questions):

- **Answer-leak rate (tutor mode)**: two probes per question, a hint request and an adversarial "just tell me the answer". Reported two ways: a rule detector (every golden answer value appears as a token; an upper bound, since a stray step number can match) and the judge's yes/no verdict.
- **Solve accuracy**: the agent's final answer is normalised per question type (value list, or an A/B/C/D verdict for quantitative comparison, where the golden expression is `Qty A - Qty B`) and graded by the sympy sandbox against the golden *expression*, never against the agent's own claim. Also the product verifier's `verifier_fail` rate.
- **Judge agreement with golden labels**: calibration probes with known labels (the golden answer vs a perturbed wrong one; a hint that states the answer vs one that does not; each probe solution restates the golden expression and shows a computation, so a judge is not asked to grade a bare stub) plus the judge's agreement with the rule/sympy labels on the agent's real outputs. For AW, the judge scores each published anchor essay with that anchor held out of its context and is compared with the anchor's score (within 1 point, and MAE).
- **Cost and latency**: per run and per call, split into agent vs judge calls (tokens, USD, p50/p95 call latency, mean agent turn latency).

What it does **not** measure: AW agent output (the AW agent needs the corpus service; AW is judge calibration only), multi-turn tutoring, retrieval quality, or anything the 5-item Quant slice is too small to support statistically. Offline numbers are plumbing checks, not quality. The free tier is heavily rate-limited: in the one smoke run (2 Quant items, free models, 0 USD) most units were dropped as unavailable and are counted in `units_dropped_model_unavailable`, so treat any free-model number as noisy and check that count. Real calls are made non-streaming and replayed to the agent as a stream, so time-to-first-token is not measured.

Providers and spend: the free chain is `openrouter/free` then pinned free models; `--provider free+openai` appends `openai/gpt-4o-mini` as the fallback for final runs. A persisted ledger (`.eval-ledger/ledger.json`, gitignored; `--ledger` to relocate) caps cumulative spend at 20 USD (`--ceiling`); each call is projected at worst case and aborts the run before it could exceed the cap. Free models count as zero. The jev judge (`typesafe/jev-1.14`, OpenRouter decisions API, typed yes/no and score answers, no explanations) is paid and off by default; its request/response mapping follows OpenRouter's docs but has not been run live. Reports (with model outputs) go to `.eval-runs/` (gitignored); only aggregates are printed.

Run hygiene: a provider's daily free-tier quota 429 (e.g. OpenRouter `free-models-per-day`) is not retried; the model is skipped for the rest of the run, and every failed call is summarised with its redacted reason under `call_failures`. A run where a suite completes fewer than `--min-coverage` (default 0.8) of its units exits 2 with `coverage_gate` set and no regression metrics, so a mostly-dropped run cannot look green or feed a baseline. Successful responses are cached in `.eval-cache/` (gitignored, holds model outputs; `--cache-dir`, `--no-cache`) keyed by model, messages and sampling parameters, so reruns re-pay nothing. litellm's stdout banners are suppressed so the printed summary is parseable JSON.

Known product gap the harness exposes: `QuantAgent._verify` still checks the agent's answer against itself (`expected=claimed_answer`), so it is a well-formedness check, not independent verification. The high `verifier_fail` rate on correct answers came from the handoff (multi-value answers such as `2, 3` and prose such as `The values are 2 and -2` did not parse) and is fixed: answers are extracted and normalised before the handoff and the sandbox accepts value lists. A real "verified math" claim needs an independent expression to check against (for example the solver emitting a separate `Check:` equation); that is a design decision not made here.

- **Regression check**: `--baseline file.json` fails the run (exit 1) when a higher-is-better metric drops more than 5% vs that file; `--write-baseline` records measured metrics. No baseline is checked in because none has been measured yet.
- **Human review queue** (`evals/human_review_queue.py`): exists as a module with unit tests but is not wired into the harness.
- `evals/judges/aw_judge.py` and `quant_judge.py` remain as judge building blocks; `grade_quant_item` requires an externally supplied candidate answer and no longer defaults to the golden answer.

## Guardrails (5 layers)

All five layers are implemented and unit tested, but `/chat` currently applies only the input content-safety check (see Status).

| Layer | What it checks |
|---|---|
| Input validation | Schema, size, content safety (OpenAI Moderation API) |
| Enrichment | Haiku-tier LLM detects incomplete questions → `ClarificationTurn` |
| Generation | Tutor-mode answer-leak check (Haiku); citation completeness (deterministic) |
| Output | PII regex redaction; content safety on output text |
| Rate/quota | 60 turns/min and 200 turns/day per Candidate (in-memory; Redis in v2) |

Every check emits a `guardrail.{layer}.{name}.{pass|fail}` structured log event with trace ID.

## Deployment (Phase 2 — k8s)

k8s manifests are in `infra/k8s/` (their structure is checked by unit tests; they have not been applied to a cluster). To apply to a cluster:

```bash
# Staging (auto-deploys on merge to main via GitHub Actions)
kubectl apply -f infra/k8s/

# Verify
kubectl rollout status deployment/gateway
kubectl rollout status deployment/agent
```

**CI/CD** (`.github/workflows/`):
- Every PR: ruff + mypy + pytest, web typecheck + unit tests, plus a path-gated offline eval pipeline check (fake model, no secrets, non-blocking; it validates the harness, not model quality)
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
├── web/                  # Vite + React components and tests (no app shell yet)
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
