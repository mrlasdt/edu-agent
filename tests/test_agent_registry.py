"""
Tests for AgentRegistry.

Behaviors under test:
  1. Register an agent class and retrieve it by (test_type, section)
  2. Register multiple agents and retrieve them independently
  3. Raise AgentNotRegisteredError with a clear message when (test_type, section) not found
  4. is_registered() returns True/False correctly
"""

import pytest

from services.agent.src.agent.registry import AgentNotRegisteredError, AgentRegistry
from services.agent.src.agent.base import BaseAgent


class StubQuantAgent(BaseAgent):
    pass


class StubAWAgent(BaseAgent):
    pass


class StubIELTSAgent(BaseAgent):
    pass


# ── 1. Register and retrieve ──────────────────────────────────────────────────

def test_can_register_and_retrieve_agent():
    registry = AgentRegistry()
    registry.register("gre", "quant", StubQuantAgent)
    assert registry.get("gre", "quant") is StubQuantAgent


# ── 2. Multiple agents coexist independently ──────────────────────────────────

def test_multiple_agents_are_independent():
    registry = AgentRegistry()
    registry.register("gre", "quant", StubQuantAgent)
    registry.register("gre", "aw", StubAWAgent)
    registry.register("ielts", "writing", StubIELTSAgent)

    assert registry.get("gre", "quant") is StubQuantAgent
    assert registry.get("gre", "aw") is StubAWAgent
    assert registry.get("ielts", "writing") is StubIELTSAgent


# ── 3. AgentNotRegisteredError with informative message ───────────────────────

def test_raises_agent_not_registered_error_for_unknown_pair():
    registry = AgentRegistry()
    registry.register("gre", "quant", StubQuantAgent)

    with pytest.raises(AgentNotRegisteredError) as exc_info:
        registry.get("ielts", "writing")

    error_message = str(exc_info.value)
    assert "ielts" in error_message
    assert "writing" in error_message


def test_error_message_names_known_registrations():
    registry = AgentRegistry()
    registry.register("gre", "quant", StubQuantAgent)

    with pytest.raises(AgentNotRegisteredError) as exc_info:
        registry.get("gmat", "quant")

    # Should hint at what IS registered to help debugging
    assert "gre" in str(exc_info.value)


# ── 4. is_registered ─────────────────────────────────────────────────────────

def test_is_registered_true_for_known_pair():
    registry = AgentRegistry()
    registry.register("gre", "quant", StubQuantAgent)
    assert registry.is_registered("gre", "quant") is True


def test_is_registered_false_for_unknown_pair():
    registry = AgentRegistry()
    assert registry.is_registered("gre", "quant") is False


def test_is_registered_false_after_wrong_section():
    registry = AgentRegistry()
    registry.register("gre", "quant", StubQuantAgent)
    assert registry.is_registered("gre", "aw") is False
