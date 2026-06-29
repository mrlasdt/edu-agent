"""
Tests for the production agent registry bootstrap.

Behaviors under test:
  1. build_registry() binds the GRE Quant + AW agents
  2. An unregistered (test_type, section) still raises AgentNotRegisteredError
"""

import pytest

from services.agent.src.agent.agents.aw import AWAgent
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.bootstrap import build_registry
from services.agent.src.agent.registry import AgentNotRegisteredError


def test_registry_binds_gre_quant_and_aw():
    registry = build_registry()
    assert registry.get("gre", "quant") is QuantAgent
    assert registry.get("gre", "aw") is AWAgent


def test_registry_unregistered_pair_raises():
    registry = build_registry()
    with pytest.raises(AgentNotRegisteredError):
        registry.get("gre", "verbal")
