Status: ready-for-agent

## What to build

Two MCP servers that expose agent capabilities to external MCP-capable clients (Claude Desktop, other agents). Both are thin wrappers over the same service layer as the FastAPI backend — no duplicated logic.

**Candidate MCP server** (`mcp/candidate_server.py`): tools `search_corpus`, `get_question`, `get_model_essay`. Allows a Claude Desktop session to search the GRE corpus or fetch practice questions.

**Admin MCP server** (`mcp/admin_server.py`): tools `ingest_document`, `list_ingestion_jobs`, `get_job_status`. Allows an Admin to trigger corpus ingestion from a Claude Desktop session.

Both servers built with the MCP Python SDK. Run as separate processes in Docker Compose.

## Acceptance criteria

- [ ] `candidate_server.py` exposes `search_corpus(query, test_type)` → top-5 chunks with citation metadata
- [ ] `candidate_server.py` exposes `get_question(test_type, section, question_type)` → a random practice question from the Global corpus
- [ ] `candidate_server.py` exposes `get_model_essay(prompt, score_tier)` → a sample essay at the given ETS score tier (1–6)
- [ ] `admin_server.py` exposes `ingest_document(file_path, test_type, tier)` → job_id
- [ ] `admin_server.py` exposes `list_ingestion_jobs(limit)` → recent jobs with status
- [ ] `admin_server.py` exposes `get_job_status(job_id)` → current job state
- [ ] Both servers start in Docker Compose; tool schemas visible via MCP inspector
- [ ] Each tool is tested independently against a mocked service layer

## Blocked by

`.scratch/gre-tutor-v1/issues/05-quant-agent-solve-mode.md`
`.scratch/gre-tutor-v1/issues/07-corpus-retrieval-service.md`
