"""One corpus client for the agent's capabilities and the MCP server, and the agent's HTTP pool."""
import json
from typing import Any

import httpx
import pytest

from services.agent.src.agent.capabilities.match_student_style import MatchStudentStyle
from services.agent.src.agent.capabilities.retrieve_corpus import RetrieveCorpus
from services.agent.src.agent.clients import pool
from shared.src.shared.corpus_client import CorpusClient


@pytest.fixture(autouse=True)
def no_live_corpus_service():
    """Override the suite-wide stub: these tests exercise the real HTTP call, against a fake."""
    yield


@pytest.fixture
def corpus_requests(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Every request any new httpx.AsyncClient sends, answered by a fake corpus service."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200, json=[{"text": "mine", "tier": "candidate"}, {"text": "other", "tier": "global"}]
        )

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    return seen


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


async def test_the_request_leaves_the_floor_out_unless_the_caller_sets_it(corpus_requests):
    client = CorpusClient("http://corpus")
    await client.retrieve("q", candidate_id="c", school_id="s", test_type="gre", top_k=3)
    await client.retrieve(
        "q", candidate_id="c", school_id="s", test_type="gre", top_k=3, apply_relevance_floor=False
    )
    assert [str(r.url) for r in corpus_requests] == ["http://corpus/corpus/retrieve"] * 2
    assert _body(corpus_requests[0]) == {
        "query": "q", "candidate_id": "c", "school_id": "s", "test_type": "gre", "top_k": 3,
    }
    assert _body(corpus_requests[1])["apply_relevance_floor"] is False


async def test_an_error_status_raises(monkeypatch):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(503)), **kw),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await CorpusClient("http://corpus").retrieve(
            "q", candidate_id="", school_id="", test_type="gre", top_k=1
        )


async def test_the_style_capability_asks_for_five_candidate_chunks_and_keeps_candidate_tier(
    corpus_requests,
):
    rows = await MatchStudentStyle(corpus_url="http://corpus")._fetch_candidate_essays("c1", "gre")
    assert rows == [{"text": "mine", "tier": "candidate"}]
    assert _body(corpus_requests[0]) == {
        "query": "essay", "candidate_id": "c1", "school_id": "", "test_type": "gre", "top_k": 5,
    }


async def test_capabilities_share_the_open_pool_and_fall_back_when_none_is_open(corpus_requests):
    assert pool.current() is None
    async with pool.open_pool() as shared:
        assert pool.current() is shared
        retrieve = RetrieveCorpus(corpus_url="http://corpus")
        await retrieve.retrieve("q", "c", "s", "gre", 2)
        await retrieve.retrieve("q", "c", "s", "gre", 2)
        assert retrieve._corpus._client is shared
    assert pool.current() is None
    assert RetrieveCorpus(corpus_url="http://corpus")._corpus._client is None
    assert len(corpus_requests) == 2


def test_the_agent_app_opens_the_pool_while_it_runs_and_closes_it_after():
    from fastapi.testclient import TestClient

    from services.agent.src.agent.main import app

    with TestClient(app) as running:
        assert running.get("/healthz").json() == {"status": "ok"}
        assert pool.current() is not None
    assert pool.current() is None
