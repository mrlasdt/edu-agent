"""
Tests for the Candidate and Admin MCP servers (issue 15).

Both servers are thin wrappers over the service layer.
Tests call the tool handler functions directly with mocked service clients.

Behaviors under test:
Candidate server:
  1. search_corpus returns chunks with citation metadata
  2. get_question returns a question from the Global corpus
  3. get_model_essay returns an anchor essay at the requested score tier

Admin server:
  4. ingest_document calls ingestion service and returns job_id
  5. list_ingestion_jobs returns recent jobs list
  6. get_job_status returns current job state
  7. get_job_status returns error for unknown job
"""

import pytest
from unittest.mock import AsyncMock, patch

from mcp_servers.candidate_server import (
    search_corpus_tool,
    get_question_tool,
    get_model_essay_tool,
)
from mcp_servers.admin_server import (
    ingest_document_tool,
    list_ingestion_jobs_tool,
    get_job_status_tool,
)


# ── 0. Server modules import cleanly (guards FastMCP instantiation kwargs) ────

def test_all_mcp_server_modules_import():
    """
    Importing each server module instantiates its FastMCP() at module load.
    This guards against constructor-kwarg drift (e.g. description vs instructions)
    that unit tests of the tool functions alone would not catch.
    """
    import importlib

    for module in (
        "mcp_servers.candidate_server",
        "mcp_servers.admin_server",
        "services.math_verifier.src.math_verifier.server",
    ):
        importlib.import_module(module)


def test_mcp_servers_bind_host_and_port_via_constructor():
    """
    host/port must be set on the FastMCP constructor (stored in .settings), not
    passed to mcp.run() — run() rejects those kwargs in this SDK version. This
    guards the SSE launch path that integration-only testing would otherwise miss.
    """
    from mcp_servers.candidate_server import mcp as candidate_mcp
    from mcp_servers.admin_server import mcp as admin_mcp
    from services.math_verifier.src.math_verifier.server import mcp as verifier_mcp

    assert (candidate_mcp.settings.host, candidate_mcp.settings.port) == ("0.0.0.0", 8091)
    assert (admin_mcp.settings.host, admin_mcp.settings.port) == ("0.0.0.0", 8092)
    assert (verifier_mcp.settings.host, verifier_mcp.settings.port) == ("0.0.0.0", 8090)


# ── 1. search_corpus ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_corpus_returns_chunks():
    mock_chunks = [
        {"text": "GRE prep tip.", "tier": "global", "source_uri": "guide.pdf",
         "page_or_section": "Intro", "score": 0.9}
    ]
    with patch("mcp_servers.candidate_server._search_corpus", new_callable=AsyncMock) as mock:
        mock.return_value = mock_chunks
        result = await search_corpus_tool(
            query="quadratic equations", test_type="gre",
            candidate_id="c1", school_id=""
        )
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["tier"] == "global"


@pytest.mark.asyncio
async def test_search_corpus_passes_test_type():
    with patch("mcp_servers.candidate_server._search_corpus", new_callable=AsyncMock) as mock:
        mock.return_value = []
        await search_corpus_tool(
            query="issue essay", test_type="gre", candidate_id="c1", school_id=""
        )
    call_kwargs = mock.call_args.kwargs
    assert call_kwargs.get("test_type") == "gre"


# ── 2. get_question ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_question_returns_question():
    mock_question = {
        "id": "q001", "question_type": "numeric_entry",
        "expression": "x**2 - 5*x + 6 = 0", "ground_truth": "2,3"
    }
    with patch("mcp_servers.candidate_server._get_random_question", new_callable=AsyncMock) as mock:
        mock.return_value = mock_question
        result = await get_question_tool(test_type="gre", section="quant", question_type="numeric_entry")
    assert result["id"] == "q001"
    assert "expression" in result


@pytest.mark.asyncio
async def test_get_question_passes_section_and_type():
    with patch("mcp_servers.candidate_server._get_random_question", new_callable=AsyncMock) as mock:
        mock.return_value = {"id": "q1"}
        await get_question_tool(test_type="gre", section="quant", question_type="quantitative_comparison")
    call_kwargs = mock.call_args.kwargs
    assert call_kwargs.get("section") == "quant"
    assert call_kwargs.get("question_type") == "quantitative_comparison"


# ── 3. get_model_essay ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_model_essay_returns_essay_at_tier():
    mock_essay = {"score_tier": 6, "text": "Outstanding essay at tier 6."}
    with patch("mcp_servers.candidate_server._get_model_essay", new_callable=AsyncMock) as mock:
        mock.return_value = mock_essay
        result = await get_model_essay_tool(prompt="Technology improves lives.", score_tier=6)
    assert result["score_tier"] == 6
    assert "text" in result


@pytest.mark.asyncio
async def test_get_model_essay_passes_score_tier():
    with patch("mcp_servers.candidate_server._get_model_essay", new_callable=AsyncMock) as mock:
        mock.return_value = {"score_tier": 4, "text": "Good essay."}
        await get_model_essay_tool(prompt="Technology.", score_tier=4)
    call_kwargs = mock.call_args.kwargs
    assert call_kwargs.get("score_tier") == 4


# ── 4. ingest_document ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_document_returns_job_id():
    with patch("mcp_servers.admin_server._ingest_document", new_callable=AsyncMock) as mock:
        mock.return_value = "job-abc-123"
        result = await ingest_document_tool(
            file_path="/tmp/guide.pdf", test_type="gre", tier="global"
        )
    assert result["job_id"] == "job-abc-123"
    assert result["status"] == "queued"


@pytest.mark.asyncio
async def test_ingest_document_passes_tier():
    with patch("mcp_servers.admin_server._ingest_document", new_callable=AsyncMock) as mock:
        mock.return_value = "job-xyz"
        await ingest_document_tool(file_path="/tmp/notes.txt", test_type="gre", tier="global")
    call_kwargs = mock.call_args.kwargs
    assert call_kwargs.get("tier") == "global"


# ── 5. list_ingestion_jobs ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_ingestion_jobs_returns_jobs():
    mock_jobs = [
        {"job_id": "j1", "status": "completed", "filename": "a.pdf"},
        {"job_id": "j2", "status": "processing", "filename": "b.docx"},
    ]
    with patch("mcp_servers.admin_server._list_jobs", new_callable=AsyncMock) as mock:
        mock.return_value = mock_jobs
        result = await list_ingestion_jobs_tool(limit=10)
    assert len(result) == 2
    assert result[0]["job_id"] == "j1"


# ── 6. get_job_status — known job ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_job_status_returns_status():
    with patch("mcp_servers.admin_server._get_job", new_callable=AsyncMock) as mock:
        mock.return_value = {"job_id": "j1", "status": "completed", "error_stage": None}
        result = await get_job_status_tool(job_id="j1")
    assert result["status"] == "completed"
    assert result["job_id"] == "j1"


# ── 7. get_job_status — unknown job ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_job_status_returns_error_for_unknown():
    with patch("mcp_servers.admin_server._get_job", new_callable=AsyncMock) as mock:
        mock.return_value = None
        result = await get_job_status_tool(job_id="nonexistent")
    assert "error" in result
    assert "not found" in result["error"].lower()
