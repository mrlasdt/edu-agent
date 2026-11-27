"""
How long the gateway waits on the agent for a turn.

Behaviors under test:
  1. A coached IELTS turn, which can re-mark an essay before the agent sends a byte, is given a
     read timeout longer than the agent's marking call, so the learner is not failed after the
     spend was recorded
  2. Any other turn (GRE) keeps the 120s wait
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from services.agent.src.agent.tools.ielts_tools import MARK_TIMEOUT_S
from services.gateway.src.gateway import agent_client
from shared.src.shared.wire import GenerateRequest

GRE = GenerateRequest(candidate_id="c1", test_type="gre", section="quant", message="2x+3=11")
COACHED = GenerateRequest(
    candidate_id="c1", test_type="ielts", section="writing", message="re-mark this",
    attempt_id="a1",
)


async def _read_timeout_the_agent_call_gets(request: GenerateRequest) -> float:
    seen: dict[str, Any] = {}

    def handler(http_request: httpx.Request) -> httpx.Response:
        seen.update(http_request.extensions["timeout"])
        return httpx.Response(200, content=b"ok")

    real_client = httpx.AsyncClient

    def with_mock_transport(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), timeout=kwargs["timeout"])

    with patch.object(agent_client.httpx, "AsyncClient", with_mock_transport):
        chunks = [c async for c in agent_client.AgentClient().stream(request)]

    assert chunks == [b"ok"]
    return seen["read"]


@pytest.mark.asyncio
async def test_a_coached_ielts_turn_waits_longer_than_the_agents_marking_call() -> None:
    assert await _read_timeout_the_agent_call_gets(COACHED) > MARK_TIMEOUT_S


@pytest.mark.asyncio
async def test_a_gre_turn_keeps_the_120_second_wait() -> None:
    assert await _read_timeout_the_agent_call_gets(GRE) == 120.0
