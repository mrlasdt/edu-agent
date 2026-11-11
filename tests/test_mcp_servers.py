"""
Tests for the Candidate and Admin MCP servers.

The tool tests go through the server's own `call_tool`, the path a client's request takes, so
argument validation, the tool list and error results are the real ones. Only the HTTP boundary
to the corpus and ingestion services is replaced.

Candidate tools: corpus_search, practice_get_question, essay_get_model.
Admin tools: ingestion_submit_document, ingestion_list_jobs, ingestion_get_job.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from mcp_servers import candidate_server
from mcp_servers.admin_server import mcp as admin_mcp
from mcp_servers.candidate_server import mcp as candidate_mcp
from mcp_servers.settings import McpSettings

REPO_ROOT = Path(__file__).resolve().parent.parent

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
    The default bind is loopback; the math verifier container opts in to 0.0.0.0.
    """
    from mcp_servers.admin_server import mcp as admin_mcp
    from mcp_servers.candidate_server import mcp as candidate_mcp
    from services.math_verifier.src.math_verifier.server import mcp as verifier_mcp

    assert (candidate_mcp.settings.host, candidate_mcp.settings.port) == ("127.0.0.1", 8091)
    assert (admin_mcp.settings.host, admin_mcp.settings.port) == ("127.0.0.1", 8092)
    assert (verifier_mcp.settings.host, verifier_mcp.settings.port) == ("127.0.0.1", 8090)


def test_transport_defaults_to_stdio_and_sse_is_opt_in(monkeypatch):
    from mcp_servers import launch

    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    assert launch.transport() == "stdio"
    monkeypatch.setenv("MCP_TRANSPORT", "SSE")
    assert launch.transport() == "sse"
    monkeypatch.setenv("MCP_TRANSPORT", "carrier-pigeon")
    with pytest.raises(SystemExit):
        launch.transport()


def test_bind_host_defaults_to_loopback(monkeypatch):
    from mcp_servers import launch

    monkeypatch.delenv("MCP_HOST", raising=False)
    assert launch.bind_host() == "127.0.0.1"
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")
    assert launch.bind_host() == "0.0.0.0"


@pytest.mark.parametrize("module", ["candidate_server", "admin_server"])
def test_stdio_launch_answers_initialize_on_stdout(module):
    """
    The README's Claude Desktop config spawns `python -m mcp_servers.<module>` and speaks
    JSON-RPC over stdin/stdout. Before the default became stdio the process started an SSE
    server and wrote nothing to stdout, so the client hung. Closing stdin ends the server.
    """
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "TURN_LOG_PATH": "",
    }
    env.pop("MCP_TRANSPORT", None)
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    }
    proc = subprocess.run(
        [sys.executable, "-m", f"mcp_servers.{module}"],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=60,
    )
    first = json.loads(proc.stdout.splitlines()[0])
    assert first["id"] == 1
    assert first["result"]["serverInfo"]["name"].startswith("gre-tutor-")



# -- helpers -----------------------------------------------------------------------------------


async def call(server: FastMCP, tool: str, **arguments: Any) -> Any:
    """What a client gets back: the tool's JSON result, or ToolError for an error result."""
    result = await server.call_tool(tool, arguments)
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


async def error_of(server: FastMCP, tool: str, **arguments: Any) -> str:
    with pytest.raises(ToolError) as caught:
        await call(server, tool, **arguments)
    return str(caught.value)


def corpus_returning(chunks: list[dict[str, Any]], seen: list[dict[str, Any]] | None = None):
    """An httpx.AsyncClient whose every request is answered with `chunks` (and recorded)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(200, json=chunks)

    real = httpx.AsyncClient

    def factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return real(*args, transport=httpx.MockTransport(handler), **kwargs)

    return patch("mcp_servers.candidate_server.httpx.AsyncClient", factory)


NOTE = {
    "text": "The discriminant b^2 - 4ac tells you how many real roots a quadratic has. " * 8,
    "tier": "global",
    "source_uri": "seed/gre-quant/algebra.txt",
    "page_or_section": "algebra.txt#Discriminant",
    "score": 0.91,
}


# -- the tool list -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_are_named_area_verb_and_describe_their_arguments():
    candidate = {t.name: t for t in await candidate_mcp.list_tools()}
    admin = {t.name: t for t in await admin_mcp.list_tools()}
    assert set(candidate) == {"corpus_search", "practice_get_question", "essay_get_model"}
    assert set(admin) == {"ingestion_submit_document", "ingestion_list_jobs", "ingestion_get_job"}
    for tool in [*candidate.values(), *admin.values()]:
        assert tool.description and "Args:" in tool.description, tool.name
    assert candidate["corpus_search"].inputSchema["properties"]["response_format"]["enum"] == [
        "concise", "detailed"]
    assert candidate["practice_get_question"].inputSchema["properties"]["section"]["enum"] == [
        "quant", "aw"]


@pytest.mark.asyncio
async def test_no_tool_takes_an_identity_argument():
    for server in (candidate_mcp, admin_mcp):
        for tool in await server.list_tools():
            assert {"candidate_id", "school_id"}.isdisjoint(tool.inputSchema["properties"]), (
                tool.name)


# -- corpus_search -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_corpus_search_is_concise_by_default_and_detailed_on_request():
    with corpus_returning([NOTE, NOTE]):
        concise = await call(candidate_mcp, "corpus_search", query="discriminant")
        detailed = await call(
            candidate_mcp, "corpus_search", query="discriminant", response_format="detailed")
    assert concise["count"] == 2
    short = concise["notes"][0]
    assert set(short) == {"source", "tier", "text"}
    assert short["source"] == "algebra.txt#Discriminant"
    assert len(short["text"]) <= 300 and short["text"].endswith("...")
    full = detailed["notes"][0]
    assert full["text"] == NOTE["text"]
    assert full["score"] == 0.91 and full["source_uri"] == "seed/gre-quant/algebra.txt"


@pytest.mark.asyncio
async def test_corpus_search_searches_as_the_configured_candidate_not_as_the_model_asks(
    monkeypatch,
):
    configured = McpSettings(mcp_candidate_id="cand-1", mcp_school_id="school-1")
    monkeypatch.setattr(candidate_server, "get_settings", lambda: configured)
    seen: list[dict[str, Any]] = []
    with corpus_returning([NOTE], seen):
        # a model that tries to read somebody else's notes: the extra arguments change nothing
        await call(candidate_mcp, "corpus_search", query="roots", candidate_id="victim",
                   school_id="other-school")
    assert seen[0]["candidate_id"] == "cand-1" and seen[0]["school_id"] == "school-1"
    assert seen[0]["test_type"] == "gre" and seen[0]["top_k"] == 5


@pytest.mark.asyncio
async def test_corpus_search_without_a_configured_identity_asks_for_the_global_tier_only(
    monkeypatch,
):
    monkeypatch.setattr(candidate_server, "get_settings", lambda: McpSettings())
    seen: list[dict[str, Any]] = []
    with corpus_returning([], seen):
        await call(candidate_mcp, "corpus_search", query="roots")
    assert seen[0]["candidate_id"] == "" and seen[0]["school_id"] == ""


@pytest.mark.asyncio
async def test_corpus_search_says_what_to_do_about_an_empty_query_or_no_match():
    assert "query is empty" in await error_of(candidate_mcp, "corpus_search", query="  ")
    with corpus_returning([]):
        result = await call(candidate_mcp, "corpus_search", query="zzz")
    assert result["count"] == 0 and "Try the standard term" in result["hint"]
    message = await error_of(
        candidate_mcp, "corpus_search", query="x", response_format="verbose")
    assert "'concise' or 'detailed'" in message


# -- practice_get_question ---------------------------------------------------------------------


def _golden_solve_ids() -> set[str]:
    from evals.loader import load_golden_set

    return {i["id"] for i in load_golden_set("quant") if i["kind"] == "solve"}


@pytest.mark.asyncio
async def test_practice_question_never_carries_the_answer_key_or_an_eval_scenario():
    """
    Regression: the tool used to random.choice over every raw golden row, so one call in five
    returned a prompt-injection test string, and every answer carried the answer key
    (ground_truth) and the eval's internal failure label (failure_source).
    """
    solve_ids = _golden_solve_ids()
    assert len(solve_ids) == 16
    seen: set[str] = set()
    for _ in range(200):
        q = await call(candidate_mcp, "practice_get_question", section="quant")
        assert q["id"] in solve_ids
        assert {"ground_truth", "failure_source", "tag", "kind", "hostile", "student"}.isdisjoint(q)
        assert q["question"]
        seen.add(q["id"])
    assert len(seen) > 1


@pytest.mark.asyncio
async def test_practice_question_type_filter_applies_and_a_mistyped_one_lists_the_valid_ones():
    for _ in range(30):
        q = await call(
            candidate_mcp, "practice_get_question", section="quant", question_type="numeric_entry")
        assert q["question_type"] == "numeric_entry"
    message = await error_of(
        candidate_mcp, "practice_get_question", section="quant", question_type="numeric")
    assert "'numeric'" in message
    for valid in ("numeric_entry", "quantitative_comparison", "multiple_choice_single"):
        assert repr(valid) in message
    aw_message = await error_of(
        candidate_mcp, "practice_get_question", section="aw", question_type="essay")
    assert "'issue'" in aw_message and "'argument'" in aw_message


@pytest.mark.asyncio
async def test_practice_question_rejects_an_unknown_section_and_names_the_valid_ones():
    message = await error_of(candidate_mcp, "practice_get_question", section="verbal")
    assert "'quant' or 'aw'" in message


@pytest.mark.asyncio
async def test_practice_question_aw_returns_the_prompt_and_id_without_anchor_essays():
    q = await call(candidate_mcp, "practice_get_question", section="aw")
    assert q["prompt"] and q["task_type"] in {"issue", "argument"} and q["id"].startswith("aw")
    assert "anchor_essays" not in q


# -- essay_get_model ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_essay_get_model_returns_the_essay_for_the_prompt_asked_about():
    from evals.loader import load_golden_set

    items = {i["id"]: i for i in load_golden_set("aw")}
    first = await call(candidate_mcp, "essay_get_model", prompt_id="aw001", score_tier=6)
    second = await call(candidate_mcp, "essay_get_model", prompt_id="aw002", score_tier=6)
    assert first == {
        "prompt_id": "aw001", "score_tier": 6, "text": items["aw001"]["anchor_essays"]["6"]}
    assert second["text"] == items["aw002"]["anchor_essays"]["6"]
    assert first["text"] != second["text"]


@pytest.mark.asyncio
async def test_essay_get_model_errors_list_the_valid_prompt_ids_and_tiers():
    unknown = await error_of(candidate_mcp, "essay_get_model", prompt_id="aw999", score_tier=6)
    assert "'aw001'" in unknown and "'aw002'" in unknown
    assert "practice_get_question" in unknown
    no_tier = await error_of(candidate_mcp, "essay_get_model", prompt_id="aw002", score_tier=3)
    assert "no essay at score tier 3" in no_tier and "available for it: 1, 6" in no_tier


# -- admin: ingestion_submit_document ----------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_document_returns_a_queued_job_id(tmp_path):
    guide = tmp_path / "guide.pdf"
    guide.write_bytes(b"%PDF")
    with patch("mcp_servers.admin_server._ingest_document", new_callable=AsyncMock) as mock:
        mock.return_value = "job-abc-123"
        result = await call(
            admin_mcp, "ingestion_submit_document", file_path=str(guide), test_type="gre")
    assert result == {"job_id": "job-abc-123", "status": "queued"}
    assert mock.call_args.kwargs == {"file_path": str(guide), "test_type": "gre"}


@pytest.mark.asyncio
async def test_submit_document_explains_a_relative_path_and_a_missing_file(tmp_path):
    relative = await error_of(admin_mcp, "ingestion_submit_document", file_path="notes/a.txt")
    assert "not an absolute path" in relative and "a.txt" in relative
    missing = await error_of(
        admin_mcp, "ingestion_submit_document", file_path=str(tmp_path / "nope.txt"))
    assert "no file at" in missing


# -- admin: ingestion_list_jobs and ingestion_get_job ------------------------------------------


@pytest.mark.asyncio
async def test_list_jobs_returns_a_count_and_the_jobs_and_checks_the_limit():
    jobs = [{"job_id": "j1", "status": "done"}, {"job_id": "j2", "status": "failed"}]
    with patch("mcp_servers.admin_server._list_jobs", new_callable=AsyncMock) as mock:
        mock.return_value = jobs
        result = await call(admin_mcp, "ingestion_list_jobs", limit=10)
    assert result == {"count": 2, "jobs": jobs}
    assert mock.call_args.kwargs == {"limit": 10}
    assert "1 to 100" in await error_of(admin_mcp, "ingestion_list_jobs", limit=0)


@pytest.mark.asyncio
async def test_get_job_returns_the_job_and_points_at_the_list_for_an_unknown_id():
    with patch("mcp_servers.admin_server._get_job", new_callable=AsyncMock) as mock:
        mock.return_value = {"job_id": "j1", "status": "completed", "error_stage": None}
        assert (await call(admin_mcp, "ingestion_get_job", job_id="j1"))["status"] == "completed"
        mock.return_value = None
        message = await error_of(admin_mcp, "ingestion_get_job", job_id="nonexistent")
    assert "'nonexistent'" in message and "ingestion_list_jobs" in message


# -- a service that is down or refuses is explained, not leaked as a stack of httpx text ---------


@pytest.mark.asyncio
async def test_corpus_search_explains_an_unreachable_corpus_service():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    real = httpx.AsyncClient
    with patch(
        "mcp_servers.candidate_server.httpx.AsyncClient",
        lambda *a, **k: real(*a, transport=httpx.MockTransport(refuse), **k),
    ):
        message = await error_of(candidate_mcp, "corpus_search", query="roots")
    assert "notes service is not available" in message and "ConnectError" in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [(503, "INGESTION_ADMIN_KEY is unset"), (401, "rejected the admin key"), (500, "HTTP 500")],
)
async def test_admin_tools_explain_why_the_ingestion_service_refused(status, expected):
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "x"})

    real = httpx.AsyncClient
    with patch(
        "mcp_servers.admin_server.httpx.AsyncClient",
        lambda *a, **k: real(*a, transport=httpx.MockTransport(answer), **k),
    ):
        message = await error_of(admin_mcp, "ingestion_list_jobs", limit=5)
    assert expected in message


@pytest.mark.asyncio
async def test_admin_tool_sends_the_admin_key_the_operator_configured(monkeypatch):
    from mcp_servers import admin_server
    from shared.src.shared.config import ADMIN_KEY_HEADER

    monkeypatch.setattr(
        admin_server, "get_settings", lambda: McpSettings(ingestion_admin_key="s3cret")
    )
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[])

    real = httpx.AsyncClient
    with patch(
        "mcp_servers.admin_server.httpx.AsyncClient",
        lambda *a, **k: real(*a, transport=httpx.MockTransport(answer), **k),
    ):
        assert (await call(admin_mcp, "ingestion_list_jobs", limit=3))["count"] == 0
    assert seen[0].headers[ADMIN_KEY_HEADER] == "s3cret"
