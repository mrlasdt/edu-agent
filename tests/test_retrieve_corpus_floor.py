"""
Tests for which agents ask the corpus service for the relevance floor.

Behaviors under test:
  1. A RetrieveCorpus sends apply_relevance_floor=false unless built with the option on
  2. The Quant agent's default corpus capability asks for the floor
  3. The AW agent's default corpus capability does not (its threshold is unmeasured)
"""
import json
from typing import Any

import httpx
import pytest

from services.agent.src.agent.agents.aw import AWAgent
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.capabilities.retrieve_corpus import RetrieveCorpus


@pytest.fixture(autouse=True)
def no_live_corpus_service():
    """Override the suite-wide stub: these tests exercise the real HTTP call, against a fake."""
    yield


@pytest.fixture
def sent_bodies(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=[])

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    return bodies


async def _ask(corpus: RetrieveCorpus) -> None:
    await corpus.retrieve("q", "c1", "s1", "gre", 3)


async def test_floor_is_off_by_default(sent_bodies):
    await _ask(RetrieveCorpus(corpus_url="http://corpus"))
    assert sent_bodies[0]["apply_relevance_floor"] is False


async def test_floor_can_be_switched_on(sent_bodies):
    await _ask(RetrieveCorpus(corpus_url="http://corpus", apply_relevance_floor=True))
    assert sent_bodies[0]["apply_relevance_floor"] is True


async def test_quant_agent_lookups_ask_for_the_floor(sent_bodies):
    await _ask(QuantAgent()._retrieve_corpus)
    assert sent_bodies[0]["apply_relevance_floor"] is True


async def test_aw_agent_lookups_skip_the_floor(sent_bodies):
    await _ask(AWAgent()._retrieve_corpus)
    assert sent_bodies[0]["apply_relevance_floor"] is False
