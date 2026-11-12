"""
Tests for Orchestrator.

Behaviors under test:
  1. Accepts a supported (test_type, section) session and returns a RouteResult
  2. Returns ClarificationTurn when the message is empty / whitespace-only
  3. Returns ClarificationTurn (not raises) when (test_type, section) is not supported,
     naming the supported pairs from the shared catalog
"""

from services.gateway.src.gateway.orchestrator import (
    ClarificationTurn,
    Orchestrator,
    RouteResult,
)
from shared.src.shared.models import Session


def make_session(test_type: str = "gre", section: str = "quant") -> Session:
    return Session(candidate_id="cand-1", test_type=test_type, section=section)


# ── 1. Routing to the correct agent ──────────────────────────────────────────

def test_routes_gre_quant_session():
    orchestrator = Orchestrator()
    session = make_session("gre", "quant")
    result = orchestrator.process(session, "Solve x^2 - 5x + 6 = 0")
    assert isinstance(result, RouteResult)


def test_routes_gre_aw_session():
    orchestrator = Orchestrator()
    session = make_session("gre", "aw")
    result = orchestrator.process(session, "Write an issue essay on technology")
    assert isinstance(result, RouteResult)


# ── 2. ClarificationTurn on empty / blank message ────────────────────────────

def test_returns_clarification_on_empty_message():
    orchestrator = Orchestrator()
    session = make_session()
    result = orchestrator.process(session, "")
    assert isinstance(result, ClarificationTurn)


def test_returns_clarification_on_whitespace_message():
    orchestrator = Orchestrator()
    session = make_session()
    result = orchestrator.process(session, "   \n\t  ")
    assert isinstance(result, ClarificationTurn)


def test_clarification_for_empty_message_prompts_candidate():
    orchestrator = Orchestrator()
    session = make_session()
    result = orchestrator.process(session, "")
    assert isinstance(result, ClarificationTurn)
    assert len(result.prompt) > 0  # has something to show the candidate


# ── 3. ClarificationTurn (not raise) on unsupported (test_type, section) ───

def test_returns_clarification_for_unsupported_section():
    orchestrator = Orchestrator()
    # "ielts/speaking" is not a supported chat pair (ielts/writing is the coach)
    session = make_session("ielts", "speaking")
    result = orchestrator.process(session, "Help me with my essay")
    assert isinstance(result, ClarificationTurn)


def test_clarification_for_unsupported_section_mentions_availability():
    orchestrator = Orchestrator()
    session = make_session("ielts", "speaking")
    result = orchestrator.process(session, "Help me")
    assert isinstance(result, ClarificationTurn)
    # Should tell the candidate what IS available
    assert "Currently available: gre/aw, gre/quant, ielts/writing." in result.prompt
    assert result.reason == "agent_not_registered"
