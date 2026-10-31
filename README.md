# GRE Tutor — Agentic AI Engineering Reference Implementation

A GRE test-prep tutor built as a reference implementation of agentic AI engineering techniques: RAG pipelines, MCP servers, multi-model cost tiering, layered guardrails, eval-as-CI, and k8s deployment.

## Status

**Working end to end locally (MVP).** One command starts the whole chat path; open http://localhost:5173, pick Quant, ask a GRE Quant question and get a tutored answer streamed from a real model, grounded in a small seeded corpus:

```bash
cp .env.example .env   # then fill in OPENAI_API_KEY
docker compose -f infra/docker-compose.yml up -d --build
```

That brings up the web app (nginx, port 5173), gateway, agent, corpus, ingestion, math verifier, Qdrant and the BGE-M3 embedder, then ingests nine self-written GRE Quant notes (`seed/gre-quant/`, about 2,200 words) as the global corpus on first start. `up` returns once the seed job has finished. The first start is slow (image builds plus a ~2GB embedder model download, then emulated amd64 embedding on Apple Silicon); later starts reuse the cached model and the seeded Qdrant volume. Every role runs on `gpt-4o-mini` (`config/model_config.dev.yaml`), so a tutoring turn costs a fraction of a cent. The Quant agent retrieves the top 3 notes for each question and the guardrails apply as usual (answer-leak, citation, PII, content safety, rate limit). Backend services are also covered by 629 Python tests (mocked LLM and Qdrant) plus 111 frontend unit tests, and `npm run test:e2e` runs browser smoke tests with `/chat` and the `/ielts` API mocked.

A recorded live check (screenshot and streamed transcript) is in [docs/mvp-demo](docs/mvp-demo/README.md).

What is still missing:

- **The web app is a thin shell.** Subject picker, tutor/solve switch and a chat that streams `POST /chat` into message bubbles. There is no sources panel, feedback, preferences or upload in the UI (components exist but are not wired), so a `[1]` citation marker is plain text, and no conversation persistence. Answers render basic Markdown, and the plain-text math the tutor writes is typeset lightly (`x^2` as a superscript, `=>`, `<=`, `>=`, `!=`, `->`, `sqrt(` as symbols); there is no LaTeX or fraction layout.
- **The corpus is tiny and Quant only.** Nine short notes written for this repo; the AW agent has no corpus (no ETS material is checked in). More documents go in through the ingestion API (see "Ingest more documents").
- **The RAG reranker is optional.** Retrieval is hybrid (dense + a locally hashed sparse vector, fused with RRF; `HYBRID_RETRIEVAL_ENABLED=false` for dense-only). When `TEI_RERANKER_URL` is set, the top 20 fused hits are reranked by BGE-reranker-v2-m3 (leave it unset to turn reranking off); if the reranker is down or slow, retrieval logs a warning and keeps the RRF order. Retrieval quality with the reranker has not been evaluated.
- **The Orchestrator's complexity step is a stub.** `/chat` applies the rate-limit, PII, content-safety, completeness, answer-leak and citation guardrails (see "Guardrails"), but not output content safety.
- **Most agent metadata is dropped.** `/generate` streams text, an `X-Source-Count` header the gateway uses for the citation check, and (in Solve mode) a one-line trailer with the math verifier's verdict that the web app shows under the answer. Sources themselves never reach the UI. The verifier is a well-formedness check of the agent's own final answer, not an independent solve (see "Agent eval harness"), and the UI says so.
- **Evals are small and CI never scores a real model.** 5 Quant and 2 AW golden GRE items, plus 33 SAT Math, 41 IELTS Reading and 8 IELTS Writing prompt items (see "IELTS and SAT golden sets"); the IELTS Writing target needs third-party data downloaded locally and CI has none. The agent-in-the-loop harness (below) runs the real Quant agent, but CI only runs it offline against a fake model; real runs are manual and there is no measured baseline checked in.
- **The question-generation pipeline is a library plus an eval target.** It is not exposed over HTTP, has no UI, and has only been run for real on a handful of items on free models (see "Question-generation pipeline").
- **Local only.** The k8s manifests have never been applied to a cluster, and Langfuse is opt-in (`--profile observability`): leave the `LANGFUSE_*` keys commented out in `.env` unless it is running, because any value (even the `.env.example` placeholders, which are now ignored) turns tracing on.

The sections below describe the intended design; see `progress.txt` for the current state of each piece.

## What it does

A Candidate submits a GRE Quant or Analytical Writing question. The system:

- **Tutors by default** — scaffolds toward the answer via Socratic hints, never revealing it
- **Solves on request** — returns a fully worked solution (Quant) or a scored sample essay (AW)
- **Verifies math** — every Quant answer is checked by a sympy sandbox before emission; wrong answers retry → escalate → degrade gracefully
- **Generates practice questions** - a LangGraph pipeline plans, routes to a generator expert, verifies the key with sympy, critiques the wording and revises within a bounded retry budget (library only, see below)
- **Cites sources** — responses reference numbered passages from the corpus; uncited claims are blocked or stripped (Quant is grounded in the seeded notes; the AW corpus must be ingested first, no ETS material is checked in)
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
                    │  gpt-4o-mini   │   │  gpt-4o-mini     │
                    │  + escalation  │   │  + style critic  │
                    └────┬───────────┘   │                  │
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
| `corpus` | 8001 | Qdrant hybrid (dense + sparse, RRF) retrieval with 3-tier ACL, optional BGE rerank (fails open to RRF order) |
| `ingestion` | 8003 | Async document pipeline (parse → chunk → embed → index) |
| `math-verifier` | 8090 | FastMCP server wrapping sympy sandbox |
| `tei-embedder` | 8080 | BGE-M3 text embeddings (Hugging Face TEI) |
| `web` | 5173 | React app behind nginx, proxies `/chat` to the gateway |
| `tei-reranker` | 8081 | BGE-reranker-v2-m3 `/rerank` (Hugging Face TEI, opt-in `reranker` profile) |

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
| Quant / AW agents | `gpt-4o-mini` | `claude-sonnet-4-6` |
| Math escalation | `gpt-4o-mini` | `claude-opus-4-7` |
| Style critic, per-PR judge | `gpt-4o-mini` | `claude-haiku-4-5` |

Every role defaults to `gpt-4o-mini` to keep spend low; a production setup would tier the main agents (e.g. `gpt-4o`) and the math escalation (e.g. `o1`). Every role is one config-line change. LiteLLM routes to the correct provider.

## Quickstart (Phase 1 — localhost)

### Prerequisites

- Python 3.12+, `uv`, Node 20.19+ (CI uses 22), Docker Desktop (the app must be running before step 3; on macOS start it with `open -a Docker` and wait until `docker info` answers)

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
# Fill in OPENAI_API_KEY at minimum; all other values have localhost defaults.
```

### 3. Start the stack

One command builds and starts everything: infrastructure, the four app services and the web app. It also seeds the corpus on first start and blocks until that is done.

```bash
docker compose -f infra/docker-compose.yml up -d --build
docker compose -f infra/docker-compose.yml ps   # services should report healthy; seed exits 0
```

Then open http://localhost:5173. To start over with an empty corpus, run `docker compose -f infra/docker-compose.yml down -v`.

| Service | URL | Role |
| --- | --- | --- |
| web | http://localhost:5173 | React app behind nginx; proxies `/chat` to the gateway |
| gateway | http://localhost:8000 | External API, orchestrator, guardrails |
| corpus | http://localhost:8001 | Qdrant retrieval (3-tier ACL) |
| agent | http://localhost:8002 | QuantAgent + AWAgent, skills, critics |
| ingestion | http://localhost:8003 | Document pipeline (parse → chunk → embed → index) |
| math-verifier | http://localhost:8090 | sympy verification MCP server (SSE) |
| tei-embedder | http://localhost:8080 | BGE-M3 embeddings (needed by retrieval and the seed job) |
| qdrant | 6333 | Vector store |
| seed | - | One-shot job: ingests any `seed/gre-quant/*.txt` note not yet indexed |

Opt-in extras, each behind a compose profile:

```bash
# Langfuse LLM observability (http://localhost:3000, plus Postgres). Also needs the
# LANGFUSE_* keys in .env.
docker compose -f infra/docker-compose.yml --profile observability up -d

# BGE reranker (also set TEI_RERANKER_URL=http://tei-reranker:8080 in .env)
docker compose -f infra/docker-compose.yml --profile reranker up -d
```

The app services bind-mount the repo source and run `uvicorn --reload`, so code edits apply live with no rebuild. The Python images install from `uv.lock` (`uv sync --frozen`), so after changing dependencies in `pyproject.toml` run `uv lock` and rebuild:

```bash
docker compose -f infra/docker-compose.yml build
```

The web image is a static build, so rebuild it after frontend edits (`docker compose -f infra/docker-compose.yml up -d --build web`), or use the Vite dev server below.

The BGE-M3 embedder and BGE reranker are amd64-only, so on Apple Silicon they run under emulation. On Apple Silicon the emulated reranker is too slow for its 2s timeout (about 20s for the 20-candidate window), so locally it times out and search uses the RRF order. Real rerank latency needs native hardware, for example an arm64 CPU TEI build or a native amd64/GPU host.

Every role runs on `gpt-4o-mini` (`config/model_config.dev.yaml`). To use another model or provider (for example OpenRouter: add `OPENROUTER_API_KEY` to `.env`), edit that file.

### 4. Run tests

```bash
# Python tests (629)
uv run pytest tests/ -q

# Frontend tests (111)
cd web && npm test

# Eval pipeline check (offline fake model, no network, no keys); includes the generation suite
uv run python -m evals.agent_eval --mode offline
```

### 5. Frontend & local overrides (optional)

The app services and the web app already run inside the stack (step 3). For frontend work with hot reload, run the Vite dev server on the host instead:

```bash
# /chat is proxied to the gateway at http://localhost:8000 (override with GATEWAY_URL).
cd web && npm run dev
```

To debug a single service outside Docker, stop it in the stack and run it directly — the in-container instance frees the port:

```bash
docker compose -f infra/docker-compose.yml stop agent
uv run uvicorn services.agent.src.agent.main:app --port 8002 --reload
```

### 6. Ingest more documents

The sample corpus is seeded automatically (step 3). To add your own:

```bash
# Via Admin MCP server (Claude Desktop) or curl:
curl -F "file=@path/to/notes.txt" \
     -F "test_type=gre" -F "tier=global" \
     http://localhost:8003/admin/documents
```

## IELTS Writing mock

A first slice of IELTS preparation for the **General Training** test: take a Writing Task 2 mock under the computer-delivered clock (40 minutes within the 60-minute Writing module, at least 250 words) and get a **mock score** for each of the four criteria with examiner-style feedback. Task 2 is the same essay task in General Training and Academic, so nothing in the mock depends on the variant. Design, calibration gates and limits are in ADR-0014. This is a mock, not an official test, and every score it shows carries the line **"Mock score: an estimate, not an official IELTS result."**

```bash
# put OPENAI_API_KEY=... in .env (see .env.example), then:
docker compose -f infra/docker-compose.yml up --build -d
open http://localhost:5173/ielts-mock                         # or the "IELTS Writing mock" link on the start page
```

Compose creates `./ielts-local` (attempts, SQLite) and `./.eval-ledger` (the spend ledger), both gitignored, on first start. With Docker Desktop that needs nothing from you; with Docker Engine on Linux run `mkdir -p ielts-local .eval-ledger` first so they are owned by you rather than root.

Without Docker, run the gateway (`uv run uvicorn services.gateway.src.gateway.main:app --port 8000`, with `OPENAI_API_KEY` in the environment) and `cd web && npm run dev`, then open http://localhost:5173/ielts-mock. The mock does not need Qdrant, the embedder or the other services.

What happens: pick one of 8 original Task 2 questions, press Start (the server starts the 40-minute clock; refreshing the page does not reset it), write with a live word counter (it autosaves about every 10 seconds and resumes after a refresh), and submit, or let time run out and the essay is submitted for you. Marking takes under a minute: three independent marking passes whose per-criterion median is the score, deterministic checks (length, paragraphs, off-topic or copied text, instructions aimed at the marker), then a written review whose quotes are checked against your essay. The report shows the overall mock score with a range and a confidence, the four criteria each with evidence from your own text and concrete next steps, an error table, a task checklist, the three things to fix first, and the change since your last attempt.

**Spend cap.** AI spend on the IELTS mock and the evals is capped at **5 USD** (`AI_SPEND_CAP_USD`), counted in one ledger (`.eval-ledger/ledger.json`, `AI_LEDGER_PATH`) that they share. The GRE tutor's `/chat` is not metered and does not count toward it. A mock costs about 0.01 USD. Every mock and eval model call is checked against the cap before it is made. Once less than a mock's worth is left, the page shows how much is spent, refuses to start a new mock and says why; if the cap is reached while you write, your essay is kept and marking can be retried after the cap is raised.

**What the mock score can and cannot be trusted for.** It is shown as **"Uncalibrated estimate"** until a labelled set passes the gate in ADR-0014, and that has not happened: there are no examiner-scored essays here. On 19 instructor-graded essays the grader landed within half a band of the instructor 84% of the time with almost no average bias, but it pulls scores toward the middle (calibration slope 0.55; only half of the 7.5+ essays were within half a band, and only two essays were below 5.5), so a very strong essay may read a band or so low. Treat it as a consistent yardstick for tracking your own progress across attempts and for finding weaknesses, not as a prediction of your exam score. Use the criterion feedback and the priorities more than the exact number; a half band either way is normal. The checks cap the score for responses that are too short (under 150 words: Task Response at 4 and every other criterion at 5.5), a single paragraph (Coherence and Cohesion at 5), off topic, copied from the question, repetitive or that try to instruct the marker, and no estimate is ever above 8.5. When a cap overrides the marking passes the confidence is shown as low, because the passes agreeing says little then.

**Blind-scoring essays (the way to calibrate it).** The repository builds a pool of essays for you to score without seeing the grader's output, then measures the grader against your scores. The default pool is 30 essays: 18 drafted by a model at bands from 4.0 to 8.0 and 12 instructor-graded learner essays, shuffled under opaque ids (about 12 minutes each).

```bash
# 1. build the pool (about 0.02 USD; needs OPENAI_API_KEY; writes ielts-local/blind/)
uv run python -m evals.exam_set build-blind-pool
# 2. the public band descriptors, saved locally (gitignored) - score from these, one criterion at a time
mkdir -p ielts-local/descriptors && curl -L -o ielts-local/descriptors/writing-band-descriptors.pdf \
  https://ielts.org/cdn/Guides/ielts-writing-band-descriptors.pdf
# 3. read ielts-local/blind/essays.md and fill ielts-local/blind/scores.csv:
#    one row per essay id, TR CC LR GRA as half bands (e.g. 6.5); leave a row empty to skip it
# 4. import your scores (they become captain_blind golden items, local only), then measure
uv run python -m evals.exam_set import-blind-scores
uv run python -m evals.agent_eval --mode live --suite ielts_writing --ielts-tier captain_blind --max-items 30
```

Do not open `ielts-local/blind/key-do-not-open-until-scored.jsonl` or run the grader on the pool until you have finished scoring: it would remove the blindness. Thirty essays can reject a bad grader but cannot certify a good one; the decision gate in ADR-0014 needs 60, so scoring a second pool later is how the "uncalibrated" label eventually goes. A second default pool would only add its 18 new model drafts (48 in all), because the instructor-graded essays already imported are skipped as duplicates on import; build the second one with more drafts instead (`build-blind-pool --drafts 30 --teacher 0` adds 30, for 60).

Not in this slice: Reading, Listening, Writing Task 1, Speaking, a full multi-section mock, and tutoring-chat changes. There is no login; it is a single-user local tool.

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

The `generation` suite is described under "Question-generation pipeline" below. What it does **not** measure: AW agent output (the AW agent needs the corpus service; AW is judge calibration only), multi-turn tutoring, retrieval quality, or anything the 5-item Quant slice is too small to support statistically. Offline numbers are plumbing checks, not quality. The free tier is heavily rate-limited: in the one smoke run (2 Quant items, free models, 0 USD) most units were dropped as unavailable and are counted in `units_dropped_model_unavailable`, so treat any free-model number as noisy and check that count. Real calls are made non-streaming and replayed to the agent as a stream, so time-to-first-token is not measured.

Providers and spend: the free chain is `openrouter/free` then pinned free models; `--provider free+openai` appends `openai/gpt-4o-mini` as the fallback for final runs. A persisted ledger (`.eval-ledger/ledger.json`, gitignored; `AI_LEDGER_PATH` or `--ledger` to relocate) caps cumulative spend at 5 USD (`AI_SPEND_CAP_USD` or `--ceiling`), the same cap the IELTS mock obeys; each call is projected at worst case and aborts the run before it could exceed the cap. Free models count as zero. The jev judge (`typesafe/jev-1.14`, OpenRouter decisions API, typed yes/no and score answers, no explanations) is paid and off by default; its request/response mapping follows OpenRouter's docs but has not been run live. Reports (with model outputs) go to `.eval-runs/` (gitignored); only aggregates are printed.

Run hygiene: a provider's daily free-tier quota 429 (e.g. OpenRouter `free-models-per-day`) is not retried; the model is skipped for the rest of the run, and every failed call is summarised with its redacted reason under `call_failures`. A run where a suite completes fewer than `--min-coverage` (default 0.8) of its units exits 2 with `coverage_gate` set and no regression metrics, so a mostly-dropped run cannot look green or feed a baseline. Successful responses are cached in `.eval-cache/` (gitignored, holds model outputs; `--cache-dir`, `--no-cache`) keyed by model, messages and sampling parameters, so reruns re-pay nothing. litellm's stdout banners are suppressed so the printed summary is parseable JSON.

Known product gap the harness exposes: `QuantAgent._verify` still checks the agent's answer against itself (`expected=claimed_answer`), so it is a well-formedness check, not independent verification. The high `verifier_fail` rate on correct answers came from the handoff (multi-value answers such as `2, 3` and prose such as `The values are 2 and -2` did not parse) and is fixed: answers are extracted and normalised before the handoff and the sandbox accepts value lists. A real "verified math" claim needs an independent expression to check against (for example the solver emitting a separate `Check:` equation); that is a design decision not made here.

- **Regression check**: `--baseline file.json` fails the run (exit 1) when a higher-is-better metric drops more than 5% vs that file; `--write-baseline` records measured metrics. No baseline is checked in because none has been measured yet.
- **Human review queue** (`evals/human_review_queue.py`): in-memory; the `generation` suite sends unrepairable items to it and reports the count. Nothing else in the harness uses it and it has no store behind it.
- `evals/judges/aw_judge.py` and `quant_judge.py` remain as judge building blocks; `grade_quant_item` requires an externally supplied candidate answer and no longer defaults to the golden answer.

## IELTS and SAT golden sets

`evals/exam_set/` adds IELTS and SAT items next to the GRE files (the GRE files and `load_golden_set` are unchanged; `evals.loader.load_exam_items` loads the new ones). Decisions and trade-offs are in ADR-0013. The rule that matters: **every item has a label tier, and every metric is reported against a tier and a sample size.**

| Tier | Where it comes from | How far to trust it |
|---|---|---|
| `examiner` | band and comments published in the owner's own Cambridge IELTS books | highest; not a re-rating, and the publisher says sample scripts are not definitive examples of a band |
| `teacher_graded` | IELTS Buddy essays with per-criterion bands and comments from one named instructor | single rater, not an official examiner; overall is the band stated in the page heading, else the mean of the four criteria |
| `captain_blind` | project-owner scores under a blind protocol | single rater, not an examiner (none exist yet) |
| `site_or_model_weak` | writing9 site/model scores (Hugging Face) | noisy (about 1.4 bands off an LLM grader per the dataset card); large-n statistics and judge calibration only, never examiner truth |
| `solver` | sympy plus a second independent computation, which must agree | machine-verified (SAT Math) |
| `author_key` | written by the item's author, an agent, not checked by a human | the IELTS Reading keys (Not Given items may be arguable) and the unscored Writing prompts |

**Committed (original items only):** `evals/golden/sat_math.jsonl` (33 items from 11 original templates, 4 content domains; template id and seed are stored and `regenerate` rebuilds each item exactly; every item also passes the product's sympy verifier) `evals/golden/ielts_reading.jsonl` (41 items on 5 original passages: True/False/Not Given, sentence completion within two words, multiple choice and matching headings, every key citing an evidence sentence) and `evals/golden/ielts_writing_prompts.jsonl` (8 original Task 2 prompts across opinion, discuss both views, advantages and disadvantages, and problem and solution; prompt only, no response or labels, so the Writing grader ignores them). The committed set has 49 IELTS items against 33 SAT items. Regenerate with `uv run python -m evals.exam_set generate-sat` / `generate-reading` / `generate-writing`.

**Local only, never committed:** third-party data goes to the gitignored `evals/golden/local/` (override with `--local-dir` or `$EDU_GOLDEN_LOCAL_DIR`), with `manifest.json` listing each source, licence status, label tier, counts and hash.

```bash
uv run python -m evals.exam_set import-writing9 --limit 1200   # HF datasets-server, no key; slow because the API rate-limits
uv run python -m evals.exam_set import-cambridge --dir ~/ielts-books   # text, text PDF or scans; scans need `brew install tesseract`
uv run python -m evals.exam_set import-ieltsbuddy   # instructor-graded essays; polite: robots.txt, rate limited, cached
uv run python -m evals.exam_set manifest   # sources, licence status, tier, counts
uv run python -m evals.exam_set summary    # counts by exam, skill, tier
```

The Cambridge importer extracts Writing tasks with their sample answers, published bands and examiner comments, and writes the extracted text to `evals/golden/local/cambridge_extracted/` so OCR slips can be fixed by hand. **It has only been tested on synthetic text** (no book files were available), and real book layouts vary, so run it on one book and read the extracted text and printed counts before trusting it. Samples with no recoverable band are skipped and counted. Subfolders are searched, and a file that cannot be read (stalled cloud-drive mount, scan without OCR, corrupt PDF) is counted and named in the manifest notes instead of aborting the run. Sources surveyed so far, their tiers and the follow-ups are in `docs/ielts-golden-sources.md`.

**IELTS Writing Task 2 grader target** (`--suite ielts_writing`, opt-in because it needs local data): the production grader from `services/agent/src/agent/ielts/` (ADR-0014) scores each essay three times on Task Response, Coherence and Cohesion, Lexical Resource and Grammatical Range and Accuracy using an original rubric (official descriptors are not reproduced); the median per criterion is taken and the overall band is computed in code from the four. It reports within-0.5 rate, quadratic weighted kappa, exact match, MAE, bias, calibration slope, per-band-bucket agreement, per-criterion agreement and test-retest spread, and prints the development and decision gates from ADR-0014, against the best tier present (`--ielts-tier` forces one), always with `label_tier`, `label_tier_caveat` and `n_items`.

```bash
uv run python -m evals.agent_eval --suite ielts_writing                         # dry run: item count and plan
uv run python -m evals.agent_eval --mode offline --suite ielts_writing           # plumbing only: an offline fake grader, no key
uv run python -m evals.agent_eval --mode offline --suite ielts_adversarial       # 12 injection / off-topic / junk cases: flags raised, no band 9
uv run python -m evals.agent_eval --mode offline --suite ielts_report            # structural checks on the feedback report
uv run python -m evals.agent_eval --mode live --suite ielts_writing --samples 3 --provider openai --judge-model openai/gpt-4.1-mini --ielts-tier teacher_graded
```

**Live results (2026-10-29, grader prompt v1, 3 samples, labels `teacher_graded`, n=19, about 0.07 USD with gpt-4.1-mini):** within-0.5 0.84, bias -0.03, QWK 0.79, but calibration slope 0.55 (it pulls bands toward the middle; the 7.5+ bucket is 0.50). That passes the ADR-0014 *development* gate and fails the *decision* gate, which needs 60 essays labelled by an examiner or the owner blind. The free OpenRouter chain gave within-0.5 0.47 and bias -0.79 on 17 usable essays, so it is for plumbing only. Adversarial: 12 of 12 cases flagged, highest band 5.0. Report structure: 5 of 5 real reports pass. An earlier 6-essay run against weak labels is superseded. No baseline is checked in.

Not done: SAT Reading and Writing, IELTS Speaking, Listening and Task 1 (charts), TOEIC, a captain-scored original essay set, and the EWCCE scripts (CC BY 4.0, checked: 200 scanned handwritten PDFs that need transcription, task type and band spread unknown without downloading them). `chillies/IELTS-writing-task-2-evaluation` was checked and not imported (one free-text band string per row, no per-criterion labels, no stated source or scorer, duplicates).

## Question-generation pipeline

`services/agent/src/agent/pipeline/` is a LangGraph graph that writes new GRE Quant practice questions and refuses to emit ones it cannot verify (ADR-0012):

```
plan -> route -> generator expert -> verify tools -> critic -> accept
          ^                              |             |
          +--------- revise <------------+-------------+   (max 3 attempts)
                                         |
                                      escalate -> human review queue
```

- **Planner and router**: an LLM planner picks the question type and constraints (a type fixed by the request wins; an unusable reply falls back to a keyword rule); the type selects one of three experts (numeric entry, multiple choice, quantitative comparison), each with its own versioned prompt in `prompts/question_pipeline/`.
- **Verifier tools** (deterministic): structure, leak (answer stated or marked in the stem or options), a sympy solver that recomputes the key through the existing verify-math skill, and a distractor check (keyed options are exactly the solution values; a comparison keyed D must really change sign). The generator never grades its own key.
- **Critic**: an LLM wording check for ambiguity and answer leaks, run only on items that passed the tools. It fails closed: an unparseable reply counts as a failed check, so the item is retried and then escalated.
- **Revise loop**: failed checks become feedback to the same expert; after `max_attempts` the item is escalated instead of emitted.
- **Trace**: every result carries a `TraceEvent` per step (plan, routing, each attempt's item, every check result, feedback, model, latency).
- **Provider layer**: roles `question_planner`, `question_generator`, `question_critic` in `config/model_config.dev.yaml`, called through `litellm`, so the eval harness's metered transport (free OpenRouter first, OpenAI fallback, the shared spend ledger checked before every call) applies unchanged.

```python
from services.agent.src.agent.pipeline.factory import build_pipeline
from services.agent.src.agent.pipeline.schema import GenerationRequest

result = await build_pipeline().run(GenerationRequest(request_id="r1", topic="quadratic equations"))
result.status, result.item, result.trace   # accepted | needs_review
```

Items are structured so the key is recomputable (a sympy `expression`, or two quantities for a comparison) and options must be numeric values, so text-option and word-heavy items are out of scope for "verified" in v1.

### Evaluating it

```bash
uv run python -m evals.agent_eval --mode offline --suite generation --max-items 12
uv run python -m evals.agent_eval --mode live --suite generation --provider free --max-items 3
```

The `generation` suite runs the single-call baseline (same model role and item schema, one prompt for all types, output emitted unchecked) and the pipeline on the same requests (`evals/golden/gre_quant_generation.jsonl`, 12 requests), audits every emitted item, and prints both side by side: valid yield per request, shipped-defect rate and which check caught it, escalation rate, unparseable rate, mean LLM calls, an independent judge verdict, and calls/tokens/USD per target and per valid item. Two further metrics: `verifier_reject_rate` is the share of generate attempts (parsed or unparseable, counted per generate step across all requests) whose deterministic verify step failed, read from the per-attempt traces (`null` for the baseline, which never verifies); `cost.usd_per_accepted_item` is the target's USD divided by emitted items (status accepted or unchecked), alongside `emitted_rate` and `cost.usd_per_valid_item` (USD divided by audit-valid items). Per-item traces are saved in the run report under `rows` (gitignored `.eval-runs/`).

**Offline before/after (fake model with seeded flaws, 12 requests; plumbing, not model quality):**

| | baseline | pipeline |
|---|---|---|
| valid yield per request | 25% | 92% |
| shipped defect rate | 70% | 0% |
| unparseable replies | 17% | 0% |
| escalated to review | n/a | 8% (1 request the fake never fixes) |
| mean LLM calls per request | 1.0 | 3.75 |

Read this honestly: the fake injects a wrong key, a leaked answer, a duplicated option or non-JSON on the first attempt and fixes it only when the prompt carries revision feedback, so the table shows that the routing, tools, revise loop and escalation work, and what the loop costs in calls. It says nothing about how often a real model errs. The audit also reuses the pipeline's own tools, so the pipeline's 0% shipped-defect rate is by construction; the offline judge is derived from the audit, so only a live run gives an independent judge verdict.

**Live smoke (2026-10-24, free OpenRouter chain only, 3 requests, 0 USD):** inconclusive. The free tier was mostly unavailable: 5 of the 6 units (3 baseline, 3 pipeline) were dropped because every model in the chain failed, so only one baseline item completed (it shipped a leak defect) and the pipeline completed none. An earlier attempt with a 900-token cap had every reply cut off mid-JSON by the routed model's hidden reasoning, which is why `max_tokens` is now 2500 for these roles. A pipeline item makes up to 7 calls and one failed call drops the whole item, so it is more exposed to a flaky free tier than the baseline; there is no per-call retry inside the pipeline beyond the transport's rate-limit retries and fallback chain. No measured before/after on a real model exists yet; a run on `--provider free+openai` (up to 8 calls per request across both targets, plus judge calls, at gpt-4o-mini prices) is the next step.

Not done: dedup against the bank, difficulty calibration, knowledge-base grounding, an HTTP endpoint on the agent service, persisting traces, a store behind the review queue, and Verbal or AW generation.

## Guardrails (5 layers)

`/chat` applies the rate-limit, input PII, content-safety, completeness (enrichment), answer-leak, citation and output PII layers. The completeness check sees the last 6 history turns so follow-ups are not flagged, and fails open if the check errors. Output content safety is not applied (see Status). Guardrail decisions that change the reply are streamed to the Candidate as bracketed notices. The answer-leak regex stage runs on the stream and stops it on a hit; the LLM stage runs once after the full reply and can only append a notice. Citation checking applies only when the agent reports a source count (the AW agent).

| Layer | What it checks |
|---|---|
| Input validation | Schema, size, content safety (OpenAI Moderation API) |
| Enrichment | Haiku-tier LLM detects incomplete questions → clarification reply |
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
├── web/                  # Vite + React app shell, nginx image, unit tests and a Playwright smoke test
├── seed/                 # Self-written GRE Quant notes ingested on first start
├── evals/                # Eval harness, golden sets, judges, regression check
├── prompts/              # Versioned system prompts (quant_agent, aw_agent)
├── config/               # model_config.{dev,staging,prod}.yaml
├── infra/
│   ├── docker-compose.yml  # Phase 1: full local stack
│   └── k8s/                # Phase 2: Deployment + Service manifests
├── .github/workflows/    # ci.yml + deploy.yml
├── docs/
│   ├── adr/              # 14 Architectural Decision Records
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
| 0013 | IELTS and SAT golden sets: label tiers, third-party data stays local |
| 0014 | IELTS Writing mock: `/ielts` attempt API, server clock, calibration gates, estimate framing |
