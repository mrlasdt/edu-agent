"""
Tests for the production agent registry bootstrap.

Behaviors under test:
  1. build_registry() binds the GRE Quant + AW agents and the IELTS Writing coach
  2. An unregistered (test_type, section) still raises AgentNotRegisteredError
  3. The registry binds exactly the pairs in shared.catalog.SUPPORTED_PAIRS
"""

import pytest

from services.agent.src.agent import bootstrap
from services.agent.src.agent.agents.aw import AWAgent
from services.agent.src.agent.agents.ielts_writing import IeltsWritingAgent
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.bootstrap import build_registry
from services.agent.src.agent.registry import AgentNotRegisteredError
from shared.src.shared.catalog import SUPPORTED_PAIRS


def test_registry_binds_the_chat_agents():
    registry = build_registry()
    assert registry.get("ielts", "writing") is IeltsWritingAgent
    assert registry.get("gre", "quant") is QuantAgent
    assert registry.get("gre", "aw") is AWAgent


def test_registry_unregistered_pair_raises():
    registry = build_registry()
    with pytest.raises(AgentNotRegisteredError):
        registry.get("gre", "verbal")


def test_registry_binds_exactly_the_catalog_pairs():
    """The gateway routes against the shared catalog; the agent must serve the same pairs."""
    registry = build_registry()
    assert all(registry.is_registered(t, s) for t, s in SUPPORTED_PAIRS)
    assert registry.is_registered("gre", "verbal") is False
    assert len(SUPPORTED_PAIRS) == 3


def test_registry_refuses_to_start_when_it_drifts_from_the_catalog(monkeypatch):
    monkeypatch.setattr(bootstrap, "SUPPORTED_PAIRS", (*SUPPORTED_PAIRS, ("gre", "verbal")))
    with pytest.raises(RuntimeError, match="do not match"):
        bootstrap.build_registry()
