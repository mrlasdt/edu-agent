"""
The contract between the gateway and the agent service, checked at the HTTP boundary.

Behaviors under test:
  1. A misspelled request field is rejected with 422, not silently dropped (a typo in
     `personal_style_enabled` once meant "feature off" with no error anywhere)
  2. A history item over the cap is rejected by the agent service too: the gateway and the agent
     agree on one `HistoryItem`, with one `max_length`
  3. What the gateway forwards, redaction growth included, is accepted by the agent service itself
  4. An agent that yields anything but a typed event fails loudly, instead of its metadata being
     ignored because a key was renamed
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.agent.src.agent import main as agent_main
from services.gateway.src.gateway import agent_client
from services.gateway.src.gateway import main as gateway_main
from services.gateway.src.gateway.chat import answer as chat_answer
from services.gateway.src.gateway.chat import gates as chat_gates
from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.guardrails.verdict import Verdict
from shared.src.shared.events import Sources, Token
from shared.src.shared.wire import MAX_FORWARDED_CHARS, MAX_MESSAGE_CHARS
from tests.gateway.gateway_fakes import rate_limiter_of

client = TestClient(agent_main.app)
BODY = {"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "Solve 2x+3=11"}


def test_a_misspelled_request_field_is_rejected_not_dropped() -> None:
    resp = client.post("/generate", json={**BODY, "personal_style_enable": True})

    assert resp.status_code == 422
    assert [e["loc"] for e in resp.json()["detail"]] == [["body", "personal_style_enable"]]


def test_the_agent_rejects_a_history_item_over_the_shared_cap() -> None:
    too_long = {"role": "candidate", "content": "x" * (MAX_FORWARDED_CHARS + 1)}

    resp = client.post("/generate", json={**BODY, "history": [too_long]})

    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "history", 0, "content"]


def test_a_misspelled_event_field_is_rejected_when_the_event_is_built() -> None:
    with pytest.raises(ValidationError):
        Sources(cnt=3)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Token(txt="a")  # type: ignore[call-arg]


def test_the_agent_accepts_what_the_gateway_forwards_after_redaction_grew_it() -> None:
    """/chat through the real AgentClient into the real agent app (no network): a message and a
    history item at the client's cap, full of addresses, come out longer than the cap."""
    emails = "a@b.co " * 571
    assert len(emails) < MAX_MESSAGE_CHARS
    seen: dict[str, str] = {}

    class Echo:
        async def run(self, session: Any, message: str, trace_id: str = ""):
            seen["message"], seen["history"] = message, session.history[0].content
            yield Sources(count=1)
            yield Token(text="ok")

    real_client = httpx.AsyncClient

    def into_agent_app(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.ASGITransport(app=agent_main.app),
                           timeout=kwargs.get("timeout"))

    body = {**BODY, "message": emails, "history": [{"role": "candidate", "content": emails}]}
    with (
        patch.object(agent_main, "_registry", _Registry(Echo)),
        patch.object(agent_client.httpx, "AsyncClient", into_agent_app),
        patch.object(chat_gates, "check_content_safety", AsyncMock(return_value=Verdict.allow())),
        patch.object(chat_gates, "check_completeness", AsyncMock(return_value=Verdict.allow())),
        patch.object(chat_answer, "check_answer_leak", AsyncMock(return_value=Verdict.allow())),
        rate_limiter_of(RateLimiter()),
    ):
        resp = TestClient(gateway_main.app).post("/chat", json=body)

    assert resp.status_code == 200
    assert resp.text.endswith("ok")  # after the notice that something was redacted
    for redacted in seen.values():
        assert redacted.count("[REDACTED]") == 571
        assert MAX_MESSAGE_CHARS < len(redacted) <= MAX_FORWARDED_CHARS


class _Registry:
    def __init__(self, agent: type) -> None:
        self._agent = agent

    def get(self, test_type: str, section: str) -> type:
        return self._agent


@pytest.mark.parametrize(
    "item", [{"sources": 3}, {"sourcess": 3}, {"verifier_fail": False}, "a bare string"]
)
def test_an_agent_that_yields_an_untyped_item_fails_loudly(item: Any) -> None:
    class Untyped:
        async def run(self, session: Any, message: str, trace_id: str = ""):
            yield item
            yield "text"

    with patch.object(agent_main, "_registry", _Registry(Untyped)):
        with pytest.raises(TypeError):
            client.post("/generate", json=BODY)
