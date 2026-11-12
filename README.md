# edu-agent - Multi-Exam Tutor and Agentic AI Engineering Reference Implementation

An exam-prep tutor built as a reference implementation of agentic AI engineering techniques: RAG pipelines, MCP servers, multi-model cost tiering, layered guardrails and eval-as-CI.

What it covers today, by exam:

| Exam | Tutoring chat | Also in the repo |
|---|---|---|
| GRE Quant | Tutor (Socratic hints) and Solve (worked solution, answer checked by sympy for well-formedness, and independently for numeric answers: a model-written equation solved by sympy, shown as agrees / disagrees / not checkable), grounded in a seeded corpus | Failure-seeded eval suite, question-generation pipeline, opt-in math tools |
| GRE Analytical Writing | Tutor and Solve (scored sample essay), needs an ingested corpus | 2 golden items |
| IELTS Writing (Task 2) | Chat coach (`ielts/writing`, tutor only) that coaches from your graded mock report and never writes the essay; the timed mock itself is at `/ielts-mock` with per-criterion mock scores and feedback | Grader eval suites (`ielts_writing`, `ielts_report`, `ielts_adversarial`) |
| IELTS Reading | None | 41 golden items |
| SAT Math | None | 33 golden items from 11 solver-verified templates |

Only `gre/quant`, `gre/aw` and `ielts/writing` are registered agents (`services/agent/src/agent/bootstrap.py`, with the pair list in `shared/src/shared/catalog.py`); any other `test_type` and `section` sent to `/chat` gets a "not set up yet" reply.

## Quickstart

Prerequisites: Python 3.12+, `uv`, Node 20.19+, Docker (running).

```bash
git clone <repo> && cd edu-agent
uv sync                             # Python deps (the dev group: every service plus test tools)
(cd web && npm install)             # frontend deps

cp .env.example .env                # then edit .env (below)
docker compose -f infra/docker-compose.yml up -d --build
docker compose -f infra/docker-compose.yml ps    # services healthy; seed exits 0
```

Open http://localhost:5173, pick Quant, AW or IELTS Writing and ask a question (the IELTS Writing coach needs a marked mock first; `uv run python -m scripts.seed_ielts_attempt` seeds one without a model call). The first start is slow (image builds, a ~2GB embedder model download, emulated amd64 embedding on Apple Silicon); later starts reuse the cached model and Qdrant volume. `down -v` resets to an empty corpus.

### Configure `.env`

- **`INGESTION_ADMIN_KEY` (required).** The ingestion `/admin` routes refuse every request (503) until it is set, then accept only an `X-Admin-Key` header carrying it. The seed job and admin MCP server read the same variable.
  ```bash
  sed -i.bak '/^INGESTION_ADMIN_KEY=/d' .env && rm .env.bak
  echo "INGESTION_ADMIN_KEY=$(openssl rand -hex 24)" >> .env
  ```
- **Models.** Every service reads the config named by `MODEL_CONFIG_PATH` (default `config/model_config.free.yaml`: every role on one free OpenRouter model, needs `OPENROUTER_API_KEY`; `config/model_config.dev.yaml` is the paid OpenAI config, every role on `gpt-4o-mini`, needs `OPENAI_API_KEY`; OpenAI retires `gpt-4o` and `gpt-4.1` and their minis on 2027-01-14). Other roles and providers are one config line each (LiteLLM). Services read it once at startup: after a change run `docker compose -f infra/docker-compose.yml restart gateway agent`.
- **Free-model default.** `config/model_config.free.yaml` puts every role on one OpenRouter `:free` model that supports tool calling. It is the default, so with no OpenAI key:
  ```bash
  # in .env: set OPENROUTER_API_KEY, leave OPENAI_API_KEY unset, and add
  ENV=dev
  ```
  To use the paid OpenAI models instead, set `OPENAI_API_KEY` and `MODEL_CONFIG_PATH=config/model_config.dev.yaml`.
  With `ENV=dev` and no OpenAI key the gateway skips the OpenAI Moderation call and records `moderation_skipped: true` on the turn; in any other `ENV` a missing key makes the turn answer "safety check unavailable". `MODERATION_ON_ERROR=closed|open` sets the policy for a moderation outage (default closed, open in `ENV=dev`). To try the supportive reply for a self-harm message without a moderation key, set `SAFETY_TEST_TRIGGER` (for example `SAFETY_TEST_TRIGGER=__test_self_harm__`) and send a message containing that text; it works only with `ENV=dev`. Free models are rate limited and slow: use them for development, not for judging quality.
- **Math tools.** `MATH_TOOLS_ENABLED=true` lets the Quant agent call `math_evaluate`, `math_solve` and `math_compare` (served by `math-verifier`, at most 4 tool rounds per turn). Default `false`: tool turns are not streamed and cost more tokens.
- **Long sessions.** The Quant agent shows the model the opening message (the problem) and the latest 8 to 11 messages, so a long tutoring session stops growing in cost and the tutor still knows the problem. Older turns are dropped, in steps of 4; `CONTEXT_COMPACTION_ENABLED=false` sends every message. The web app sends up to 20 messages (the whole conversation, or the opening one plus the latest 19). Compare with `--context-budget` in `evals.agent_eval` (item `m005` has 20 turns).
- **Retrieval mode.** `RETRIEVAL_MODE=just_in_time` makes the Quant tutor look notes up itself with a `notes_search(concept)` tool instead of receiving them on every turn. Default `upfront`: just-in-time turns are not streamed and rest on one more model decision. Solve mode always searches up front; the IELTS Writing coach retrieves nothing. Compare the two with `--retrieval` in `evals.agent_eval`.
- **IELTS mock.** Default config: grader and feedback roles on `gemma-4-26b-a4b-it` called on Google AI Studio directly (`GEMINI_API_KEY`; transient Google 500/503 are retried twice), falling back to the same model and then `google/gemma-4-31b-it` on OpenRouter's free tier (`OPENROUTER_API_KEY`) when the key is missing, Google is down, or a call fails, and `OPENAI_API_KEY` only with `config/model_config.dev.yaml` (`gpt-4.1-mini`). Both Gemma models passed the ADR-0014 development gate (within-0.5 at least 0.75 and |bias| at most 0.3) on the 19 `teacher_graded` IELTS Buddy essays, 3 samples each, max 4000 completion tokens, one instructor's labels: OpenRouter free: 26b 0.83 and -0.14, 31b 0.82 and -0.26; Google direct (one run): 26b 0.84 and -0.08, 31b 0.89 and -0.18; against 0.84 and -0.03 for `gpt-4.1-mini`; `recursive/flint-v1.0` (the default for every other role) failed with 0.62 and -0.62. This is the development gate only: the decision gate is still not met, so every mock score stays an uncalibrated estimate. Full table in ADR-0014. The IELTS Writing coach in `/chat` runs the `ielts_coach` role on the same model as the grader (the other chat gates keep their own models). `AI_SPEND_CAP_USD` (default 5) caps AI spend of the mock and the evals in one shared ledger (`AI_LEDGER_PATH`); `/chat` turns, the coach's included, are not capped and are metered per turn. Compose creates `./ielts-local` and `./.eval-ledger`; on Linux run `mkdir -p ielts-local .eval-ledger` first.
- **Observability.** Langfuse is opt-in (`--profile observability`); leave the `LANGFUSE_*` keys commented out unless it is running. Retrieval reranking is opt-in (`--profile reranker` plus `TEI_RERANKER_URL=http://tei-reranker:8080`); a down or slow reranker falls back to the RRF order.

### Add documents

```bash
curl -H "X-Admin-Key: $INGESTION_ADMIN_KEY" \
     -F "file=@path/to/notes.txt" -F "test_type=gre" \
     http://localhost:8003/admin/documents
```

A missing or wrong key returns 401. There is no `tier` field: the server files every upload under the global tier and ignores a client-sent `tier`. Files under 1 MB are ingested before the response (200); larger ones answer 202 `queued` and are indexed by a background task in the same process. Poll `GET /admin/documents/{job_id}` with the same header for `completed` (with `chunks_indexed`) or `failed` (with `error_stage` and `error`); job records are kept in memory, so they are lost on a restart (the service runs as a single replica). Uploading the same filename again replaces its chunks instead of adding a copy. `.txt`, `.pdf` and `.docx` are accepted.

### Test and eval

```bash
uv run pytest tests/ -q                                  # Python tests (mocked LLM and Qdrant)
(cd web && npm test)                                     # frontend unit tests
(cd web && npm run test:e2e)                             # browser smoke tests (Playwright; /chat and /ielts mocked)
uv run python -m evals.agent_eval --mode offline         # eval pipeline on a fake model: no network, no keys
```

Live evals call real models and spend against the ledger; see "Evals" below. CI (`.github/workflows/ci.yml`) runs ruff, mypy, the reference lint, pytest, the web tests, build (which typechecks) and e2e, and the offline evals on every change (no path filter). It also builds every image and starts the services from them (`infra/docker-compose.smoke.yml`). It also runs a dependency audit (`pip-audit` on `uv.lock`, `npm audit` at high severity for `web/`) and a gitleaks secret scan over the full history. It never scores a real model.

### Run a single piece outside Docker

```bash
(cd web && npm run dev)                                  # Vite dev server; /chat proxied to the gateway
docker compose -f infra/docker-compose.yml stop agent
uv run uvicorn services.agent.src.agent.main:app --port 8002 --reload
```

The app services bind-mount the source and run `uvicorn --reload`. The web image is a static build: rebuild it after frontend edits (`up -d --build web`). After changing Python dependencies run `uv lock` and rebuild the images.

## Architecture

```
Candidate / Admin     ┌──────────────────────────────────────┐
(Web UI / MCP) ─────► │ Gateway: validate, enrich, guardrails │
                      │ + /ielts mock API                     │
                      └───────┬──────────────────────────────┘
                              │ POST /generate
                     ┌────────▼────────────┐   ┌──────────────────┐
                     │ Agent: Quant, AW,   │──►│ Math verifier    │
                     │ IELTS coach, gen.   │   │ (sympy, MCP)     │
                     │ pipeline            │   └──────────────────┘
                     └────────┬────────────┘
                              │ POST /corpus/retrieve
                     ┌────────▼────────────┐   ┌──────────────────┐
                     │ Corpus: Qdrant      │◄──│ Ingestion        │
                     │ hybrid + ACL        │   │ parse, chunk,    │
                     └─────────────────────┘   │ embed, index     │
                                               └──────────────────┘
```

| Service | Port | Role |
|---|---|---|
| `web` | 5173 | React app behind nginx. `/` is the tutor chat, `/ielts-mock` the IELTS Writing mock; the report page links to the coach |
| `gateway` | 8000 | `POST /chat`, `POST /feedback`, the `/ielts/*` mock API (grader code in `shared/src/shared/ielts/`), orchestrator, guardrails, rate limit |
| `corpus` | 8001 | `POST /corpus/retrieve`: Qdrant dense + sparse search fused with RRF, ACL filter, optional rerank |
| `agent` | 8002 | `POST /generate`: Quant, AW and IELTS Writing coach agents, capabilities, tools, critics, question pipeline |
| `ingestion` | 8003 | `/admin/documents` (admin key): parse, chunk, embed, index |
| `math-verifier` | 8090 | FastMCP server: sympy checks behind an allow-list parser, one killable child process per check; also `POST /tools/{name}` for the math tools |
| `tei-embedder` | 8080 | BGE-M3 embeddings (Hugging Face TEI) |
| `qdrant` | 6333 | Vector store |
| `seed` | - | One-shot job: uploads every `seed/gre-quant/*.txt` note; ingestion replaces a note already indexed, so a restart never duplicates it |
| `tei-reranker`, `langfuse`, `postgres` | 8081, 3000, 5432 | Opt-in profiles (`reranker`, `observability`) |

### Three-tier corpus

Every retrieval filters by tier: global (admin uploads, visible to all), school (teacher uploads, not yet writable) and candidate (private to its owner). An empty `school_id` or `candidate_id` grants no access to that tier. The filter is `_build_acl_filter` in `services/corpus/src/corpus/retriever.py` and the filter fields carry keyword payload indexes; `tests/test_tenant_isolation.py` checks tenant isolation. Re-ingest a collection created before the indexes existed into a fresh one. `uv run python -m evals.qdrant_filter_check` measures filter recall and latency against a throwaway Qdrant container (needs Docker, no model calls; `--points 20000` for a quick run).

### Guardrails

`/chat` applies, in order: request size caps (message 4,000 chars; history 20 items of 4,000; candidate id 128; over a cap is a 422 before any model call), per-candidate rate limit (60 turns/min, 200/day, in memory), routing (a blank message or an unsupported exam and section is answered here, before any model call), input PII redaction, for IELTS Writing coach turns, a lookup of your graded mock report (the `attempt_id` in the request, or the latest marked one; none found gets a notice that points to `/ielts-mock`), content safety (OpenAI Moderation), a completeness check on the question, a tutor-mode answer-leak check (regex on the stream, LLM after the reply; both are skipped for the coach, which is told never to write the essay but whose replies may quote one corrected sentence), a citation check (only when the agent reports sources) and output PII redaction. Output content safety is not applied. Every check logs `guardrail.{layer}.{name}.{pass|fail}` with the trace id. Token caps on the guardrail model calls are `COMPLETENESS_MAX_TOKENS` and `LEAK_CHECK_MAX_TOKENS`.

### Measuring a turn

Each `/chat` turn is metered (model calls, tokens, estimated USD, time to first token, notes retrieved, prompt version, guardrail outcome) to `.turn-log/turns.jsonl` (`TURN_LOG_PATH`; empty switches it off). `uv run python -m evals.turn_report --turns` summarizes it. Cost is recorded, never capped. Stop in the chat closes the request; the gateway and the agent service then stop their model calls, and the turn is recorded with `outcome=client_cancelled`. If the tutoring service fails before or during a reply, the reply ends with a plain "Tutor unavailable" notice and the turn is recorded with `outcome=agent_unavailable` (see ADR-0007). Regenerate resends the last question with the history before it.

### Feedback

`/chat` returns the turn's trace id in the `X-Trace-Id` header, and the chat page shows thumbs up/down under each finished reply. A thumbs click is `POST /feedback {trace_id, vote: "up"|"down", comment?}` (comment up to 1,000 chars; the gateway rejects an unknown trace id, rate limits per client address and appends the vote to `.turn-log/feedback.jsonl`, `FEEDBACK_LOG_PATH`, empty switches it off). A line holds the trace id, the vote, the optional comment and a timestamp: no message text. With Langfuse on, the vote is also posted as a `user-feedback` score on that trace.

- `uv run python -m evals.turn_report --feedback` adds the thumbs-down rate per agent and prompt version (the latest vote per turn counts) and lists each thumbs-down with its trace id. With one learner a rate is an anecdote, not a statistic.
- `uv run python -m evals.feedback_to_item <trace_id>` writes a draft eval item to `.turn-log/draft-items.jsonl` for the owner to complete (question type, ground truth, tag) before it moves into a golden file.
- The question and the reply are not stored by default, so a draft is a stub (trace id, agent, prompt version, vote) and the question is re-entered by hand. `TURN_CAPTURE_TEXT=1` (default off) makes the gateway also write the question (after PII redaction) and the reply to `.turn-log/capture.jsonl` (`TURN_CAPTURE_PATH`), keyed by trace id; `feedback_to_item` then fills the draft in. The file is local, gitignored and never sent anywhere.

## IELTS Writing mock

A timed Task 2 mock for the General Training test (40 minutes, server-side clock, 250+ words): pick one of 8 original prompts, write with a live word counter and autosave, submit (or time out), and get a mock score for each of the four criteria with evidence from your essay, next steps and an error table. Marking is three independent passes whose per-criterion median is the score, deterministic checks (length, paragraphs, off-topic or copied text, instructions aimed at the marker) and a written review whose quotes are verified against the essay. Every score is labelled an estimate, not an official IELTS result, and "uncalibrated" until the ADR-0014 gate is met. Attempts are stored in SQLite under `./ielts-local`. A marked report has a "Discuss this report" link that opens the chat coach on that attempt.

Without Docker: run the gateway (`uv run uvicorn services.gateway.src.gateway.main:app --port 8000`, `OPENROUTER_API_KEY` in the environment, or `OPENAI_API_KEY` with `MODEL_CONFIG_PATH=config/model_config.dev.yaml`) and `cd web && npm run dev`, then open http://localhost:5173/ielts-mock. The mock does not need Qdrant or the embedder.

To calibrate against your own scores: `uv run python -m evals.exam_set build-blind-pool`, fill `ielts-local/blind/scores.csv`, then `import-blind-scores` and the `ielts_writing` eval below (ADR-0014).

## Evals

`evals/agent_eval/` runs the real agent and grades what it produced.

```bash
uv run python -m evals.agent_eval                          # dry run (default): prints the plan, no calls
uv run python -m evals.agent_eval --mode offline           # deterministic fake model, no network
uv run python -m evals.agent_eval --mode offline --suite quant --samples 3
uv run python -m evals.agent_eval --mode live --suite quant --samples 3 --provider free --max-tokens 3000
uv run python -m evals.agent_eval --mode live --provider free+openai    # free chain, OpenAI fallback
```

| Suite (`--suite`) | What it checks |
|---|---|
| `quant` | 40 failure-seeded GRE Quant items scored by rule graders: answer leak, solve accuracy via sympy, multi-turn tutoring, hostile math strings for the verifier. `--samples k` reports pass rate, pass@k and pass^k with intervals; `--baseline` flags regressions (`evals/baselines/quant.json`) |
| `quant_judge` | LLM-judge calibration on the Quant solve items |
| `aw` | LLM-judge calibration on the 2 AW anchor essays |
| `generation` | question-generation pipeline against a single-call baseline on 12 requests |
| `ielts_writing` | the IELTS grader against labelled essays; needs local third-party data, so it is opt-in |
| `ielts_coach` | the chat coach on a seeded report (rules, no judge; offline 0 USD) |
| `ielts_adversarial`, `ielts_report` | 12 injection and off-topic cases; structural checks on the feedback report |

`uv run python -m evals.tool_use_eval --mode offline|live` is a separate small eval of the candidate MCP server's tools: 10 plain requests (`evals/golden/mcp_tool_use.jsonl`), each scored on whether the first call is to the right tool, carries the right arguments, and how many calls error (errors divided by calls, as in the quant suite). Offline uses a keyword router in place of a model and only proves the wiring; `--mode live` uses the free OpenRouter chain, and `evals/baselines/tool_use.json` holds one such run (10 tasks, so every rate has a wide interval).

`--math-tools on|off` runs the Quant agent with or without the math tools, so both sides can be compared on one model. Live runs use free OpenRouter models first (`--provider free`), keys only from the environment; successful responses are cached in `.eval-cache/` and reports go to `.eval-runs/` (both gitignored). Spend is capped by the shared ledger. A run completing under `--min-coverage` (default 0.8) of its units exits 2. A manual `eval-live` workflow runs live suites; CI runs offline only.

Useful tools: `uv run python -m evals.read_transcripts --worst 5`, `uv run python -m evals.quant_report`.

Retrieval has its own eval, `uv run python -m evals.retrieval_eval --embedder-url http://localhost:8080` (needs Docker for a throwaway Qdrant and a running `tei-embedder`; no language model is called). It indexes `seed/gre-quant/`, replays 46 labelled queries (`evals/golden/gre_retrieval.jsonl`: exact terms, paraphrases, off-topic) through dense-only, hybrid and, with `--reranker-url`, hybrid plus rerank, and prints recall@3, MRR@3, how often an off-topic query still gets a note, and a sweep of the relevance floor. The rerank column is reported as not measurable locally when the reranker misses its 2 s timeout, as the emulated amd64 TEI image does on Apple Silicon (46 of 46 queries in the checked-in run). The Quant agent's note lookups apply the relevance floor (AW's do not: it has no corpus or labelled queries, so a floor there is unmeasured): every hit must clear the dense cosine floor, and a reranked hit must also clear a rerank-score floor, which stays unset until the reranker is measured, so with the reranker on the cosine floor alone keeps an off-topic question note-free. The checked-in numbers are `evals/baselines/retrieval.json` (BGE-M3, small query set: it detects large differences, not small ones).

### Golden sets for IELTS and SAT

`evals/exam_set/` builds exam items (ADR-0013). Every item has a label tier (`examiner`, `teacher_graded`, `captain_blind`, `site_or_model_weak`, `solver`, `author_key`) and every metric is reported against a tier and a sample size. Committed items are original: `sat_math.jsonl` and `ielts_reading.jsonl` in `evals/golden/`, and `ielts_writing_prompts.jsonl`, which the product serves and so keeps under `shared/src/shared/ielts/data/` (evals reads it there). Third-party data is imported only into the gitignored `evals/golden/local/`:

```bash
uv run python -m evals.exam_set generate-sat      # also generate-reading, generate-writing: rebuild the committed sets
uv run python -m evals.exam_set import-writing9 --limit 1200
uv run python -m evals.exam_set import-cambridge --dir ~/ielts-books
uv run python -m evals.exam_set import-ieltsbuddy
uv run python -m evals.exam_set summary
```

Sources and licence notes are in `docs/ielts-golden-sources.md`.

## Question-generation pipeline

`question_pipeline/` (a top-level package, not part of any service image) is a LangGraph graph that writes GRE Quant practice questions and refuses to emit ones it cannot verify (ADR-0012): plan, route to one of three experts (numeric entry, multiple choice, quantitative comparison), deterministic checks (structure, answer leak, sympy key recompute, distractors), an LLM critic that fails closed, and a revise loop of up to 3 attempts before escalating to a review queue. It is a library with an eval target; there is no HTTP endpoint.

```python
from question_pipeline.factory import build_pipeline
from question_pipeline.schema import GenerationRequest

result = await build_pipeline().run(GenerationRequest(request_id="r1", topic="quadratic equations"))
result.status, result.item, result.trace   # accepted | needs_review
```

## MCP servers

For Claude Desktop or any MCP client. Tools are named `<area>_<verb>` so a client that lists many servers can tell them apart.

| Server | Tools |
|---|---|
| Candidate | `corpus_search(query, test_type, response_format)`, `practice_get_question(section, question_type)`, `essay_get_model(prompt_id, score_tier)` |
| Admin | `ingestion_submit_document(file_path, test_type)`, `ingestion_list_jobs(limit)`, `ingestion_get_job(job_id)` (sends `INGESTION_ADMIN_KEY`; the ingestion service answers 503 to every admin call while that key is unset) |

- **Who is asking is configuration, not an argument.** The candidate server searches as `MCP_CANDIDATE_ID` and `MCP_SCHOOL_ID` from its own environment; the model cannot name them. Unset, only the global notes are searched.
- **Practice questions come from files, not from the eval package.** The candidate server reads `gre_quant.jsonl` and `gre_aw.jsonl` from `MCP_PRACTICE_DIR` (default: the repo's `evals/golden/`), so it imports nothing from `evals/`.
- **A practice question never carries its answer.** `practice_get_question` returns the question and its id; `essay_get_model` takes that id and returns the anchor essay at a tier.
- **An error says what would have worked.** A mistyped `question_type`, an unknown `prompt_id` or a tier an essay does not have comes back as an error result listing the valid values.
- **`corpus_search` is concise by default** (source and the first 300 characters of up to 5 notes); `response_format="detailed"` returns each note in full with its source file and score.
- Neither server authenticates its client; run them only on a machine you trust. The tool-use eval (`uv run python -m evals.tool_use_eval`, see "Evals") scores whether a model picks these tools well.

Both speak **stdio** by default, which is how a client such as Claude Desktop launches them:

```json
{
  "mcpServers": {
    "gre-tutor-candidate": {
      "command": "uv",
      "args": ["--directory", "/path/to/edu-agent", "run", "python", "-m", "mcp_servers.candidate_server"],
      "env": { "MCP_CANDIDATE_ID": "your-candidate-id" }
    }
  }
}
```

SSE is opt-in. It binds `127.0.0.1` (candidate on port 8091, admin on 8092); `MCP_HOST` overrides the bind address, so set it to `0.0.0.0` only on a network you trust.

```bash
MCP_TRANSPORT=sse uv run python -m mcp_servers.candidate_server
MCP_TRANSPORT=sse uv run python -m mcp_servers.admin_server
```

The math verifier (port 8090) is a cluster-internal service, not a client-facing MCP server. It also defaults to `127.0.0.1`; its container image sets `MATH_VERIFIER_HOST=0.0.0.0`.

## Models

`config/model_config.free.yaml` (the default) maps roles (orchestrator, `quant_agent`, `aw_agent`, `ielts_coach`, math escalation, critics, graders, question-pipeline roles) to models; swapping a role to another provider is one line. `config/model_config.dev.yaml` is the paid OpenAI variant. Prompts are versioned under `prompts/`; the active version per agent and mode is pinned once for every model config in `config/prompt_pins.yaml`. `prompt_version(agent, mode)` raises for a pair that is not pinned (it never guesses `v1`), and `uv run python scripts/lint_references.py` (a CI step and `tests/test_lint_references.py`) fails when a pin has no prompt file, a prompt file has no pin or is newer than it, a call site names an unpinned pair or an undefined model role, or a configured role is never used (three reserved ones are listed with their reasons in the script).

## Deployment

Docker Compose (`infra/docker-compose.yml`) is the one supported way to run the system. There is no cluster deploy: Kubernetes manifests and a deploy workflow existed but were never applied and disagreed with each other, so they were removed (they are in git history if a cluster deploy is wanted later).

## Adding an exam or section

`AgentRegistry` is the seam: subclass `BaseAgent`, add the agent to `bootstrap.py` and its `(test_type, section)` pair to `shared/src/shared/catalog.py` (the gateway routes on the catalog; a test fails if the two differ), add prompts under `prompts/`, tag corpus documents with the `test_type`, and add a golden set under `evals/golden/`. The gateway, corpus, ingestion and math verifier need no change. The web subject picker is a hardcoded Quant / AW / IELTS Writing set and needs a new button.

## Not built yet

- Tutoring chat for SAT Math, IELTS Reading and the other IELTS sections (only golden sets exist), and SAT Reading and Writing.
- Login and learner identity: school and candidate corpus tiers cannot be written, and the IELTS mock is single-user.
- Web UI beyond the chat shell: no sources panel, preferences, upload or conversation persistence (components exist but are not wired); math is typeset lightly, no LaTeX.
- Most agent metadata is dropped on the way to the UI. The Solve-mode independent check (ADR-0015) covers numeric answers only: quantitative comparisons and other non-numeric answers show "not checkable", and "agrees" does not show the problem was read correctly. It costs one extra short model call per solve turn, and how often a real model writes a faithful equation is measured only by a live run (the offline `wrong_but_verified` uses each item's own expression).
- Orchestrator complexity classification (Basic or Advanced) and complexity-escalation suggestions: no code, prompt or stub exists. Advanced mode (planner, specialist sub-agents, assembler), its `BasicRouter` and `AdvancedRouter` classes and the "coming soon" message are not built either (ADR-0002).
- Topic relevance (an `off_topic` label for questions outside the exam; design-log 0004).
- Output content safety, and a copyright spot-check on AW output.
- Prompt-injection defence on the chat path: retrieved notes and history enter the prompt as plain text, and there is no regex or model check for injected instructions.
- A refusal policy in the system prompts (the prompts carry no refusal rules).
- A soft quota (3 a day) on full-essay Solve turns; the sample-essay disclaimer is built, the quota is not.
- On an answer leak the model check appends a notice; it does not regenerate the reply (ADR-0008).
- A runtime `verifier-fail` event and the 1% sample of those events sent to human review (only the offline eval reports `verifier_fail`).
- A style profile derived asynchronously from a learner's essays (`style_extractor`), and parent section chunks with summaries in the corpus (chunks are sections split into paragraphs).
- Upload scanning: no magic-byte check, antivirus scan, moderation of chunks or PII offer on ingested files (only extension, MIME and size limits and the admin key).
- Nightly eval run, anchor-essay self-check and the three drift alerts (`eval-live.yml` is manual only); the style-adherence judge, citation judge and personal-style similarity judge.
- An A/B traffic split between prompt versions: one version is pinned per agent and mode, and the turn record carries no variant.
- Named waiting states in the UI ("Checking your answer...", "Finding references..."): one fixed "thinking..." shows.
- A locale key for prompts and rubrics (English only; ADR-0003).
- A learner or session key on turn records, so a per-learner score trend (ADR-0005) cannot be computed yet.
- Corpus beyond 55 short GRE Quant note sections (15 files); no AW corpus.
- Question generation over HTTP, dedup against a bank, and difficulty calibration.
- A calibrated IELTS grader (needs an examiner or blind-scored set of 60 essays).
- A durable human review queue (in memory only).
- A cluster deploy (Kubernetes): only Docker Compose is supported.

## Repository layout

```
services/   gateway, agent, corpus, ingestion, math_verifier
shared/     Pydantic models, config loader, vector store, embedder, turn meter
mcp_servers/ candidate and admin MCP servers
web/        Vite + React app, nginx image, unit and Playwright tests
seed/       self-written GRE Quant notes ingested on first start
evals/      eval harness, golden sets, baselines, exam_set tooling
prompts/    versioned system prompts
config/     model configs, prompt pins and IELTS calibration
scripts/    repository checks run in CI (lint_references.py)
infra/      docker-compose.yml
docs/       adr/ (decision records), design-log/, CONTEXT.md at the repo root
```

Architectural decisions are recorded as ADRs in `docs/adr/`; `CONTEXT.md` is the domain glossary.
