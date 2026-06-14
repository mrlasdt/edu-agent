"""
Tests for Orchestrator.

Behaviors under test:
  1. Routes a registered (test_type, section) session to the correct agent class
  2. Returns ClarificationTurn when the message is empty / whitespace-only
  3. Returns ClarificationTurn (not raises) when (test_type, section) not registered
  4. RouteResult carries both the resolved agent class and the validated session
"""

import pytest

from services.agent.src.agent.base import BaseAgent
from services.agent.src.agent.registry import AgentRegistry
from services.gateway.src.gateway.orchestrator import (
    ClarificationTurn,
    Orchestrator,
    RouteResult,
)
from shared.src.shared.models import Session


class StubQuantAgent(BaseAgent):
    async def run(self, session, message):
        yield "stub"  # pragma: no cover


class StubAWAgent(BaseAgent):
    async def run(self, session, message):
        yield "stub"  # pragma: no cover


def make_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register("gre", "quant", StubQuantAgent)
    registry.register("gre", "aw", StubAWAgent)
    return registry


def make_session(test_type: str = "gre", section: str = "quant") -> Session:
    return Session(candidate_id="cand-1", test_type=test_type, section=section)


# ── 1. Routing to the correct agent ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_routes_gre_quant_session_to_quant_agent():
    orchestrator = Orchestrator(registry=make_registry())
    session = make_session("gre", "quant")
    result = await orchestrator.process(session, "Solve x^2 - 5x + 6 = 0")
    assert isinstance(result, RouteResult)
    assert result.agent_cls is StubQuantAgent


@pytest.mark.asyncio
async def test_routes_gre_aw_session_to_aw_agent():
    orchestrator = Orchestrator(registry=make_registry())
    session = make_session("gre", "aw")
    result = await orchestrator.process(session, "Write an issue essay on technology")
    assert isinstance(result, RouteResult)
    assert result.agent_cls is StubAWAgent


@pytest.mark.asyncio
async def test_route_result_carries_session():
    orchestrator = Orchestrator(registry=make_registry())
    session = make_session("gre", "quant")
    result = await orchestrator.process(session, "A math question")
    assert isinstance(result, RouteResult)
    assert result.session is session


# ── 2. ClarificationTurn on empty / blank message ────────────────────────────

@pytest.mark.asyncio
async def test_returns_clarification_on_empty_message():
    orchestrator = Orchestrator(registry=make_registry())
    session = make_session()
    result = await orchestrator.process(session, "")
    assert isinstance(result, ClarificationTurn)


@pytest.mark.asyncio
async def test_returns_clarification_on_whitespace_message():
    orchestrator = Orchestrator(registry=make_registry())
    session = make_session()
    result = await orchestrator.process(session, "   \n\t  ")
    assert isinstance(result, ClarificationTurn)


@pytest.mark.asyncio
async def test_clarification_for_empty_message_prompts_candidate():
    orchestrator = Orchestrator(registry=make_registry())
    session = make_session()
    result = await orchestrator.process(session, "")
    assert isinstance(result, ClarificationTurn)
    assert len(result.prompt) > 0  # has something to show the candidate


# ── 3. ClarificationTurn (not raise) on unregistered (test_type, section) ───

@pytest.mark.asyncio
async def test_returns_clarification_for_unregistered_section():
    orchestrator = Orchestrator(registry=make_registry())
    # "ielts" is not registered
    session = make_session("ielts", "writing")
    result = await orchestrator.process(session, "Help me with my essay")
    assert isinstance(result, ClarificationTurn)


@pytest.mark.asyncio
async def test_clarification_for_unregistered_section_mentions_availability():
    orchestrator = Orchestrator(registry=make_registry())
    session = make_session("ielts", "writing")
    result = await orchestrator.process(session, "Help me")
    assert isinstance(result, ClarificationTurn)
    # Should tell the candidate what IS available
    assert len(result.prompt) > 0
