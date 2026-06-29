"""
Production agent registration.

This is the single place where concrete agents are bound to their
(test_type, section) keys. Adding a new test type (ielts/writing, gmat/quant, …)
means importing its agent and adding one `register(...)` line here — nothing
else in the gateway or agent service changes.
"""
from __future__ import annotations

from services.agent.src.agent.agents.aw import AWAgent
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.registry import AgentRegistry


def build_registry() -> AgentRegistry:
    """Return an AgentRegistry populated with the v1 GRE agents."""
    registry = AgentRegistry()
    registry.register("gre", "quant", QuantAgent)
    registry.register("gre", "aw", AWAgent)
    return registry
