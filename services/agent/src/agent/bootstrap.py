"""
Production agent registration.

This is the single place where concrete agents are bound to their
(test_type, section) keys. Adding a new test type (gmat/quant, …) means importing its agent,
adding one entry below and the pair to `shared/src/shared/catalog.py`, which the gateway
routes against. `build_registry` refuses to start if the two disagree.
"""
from __future__ import annotations

from services.agent.src.agent.agents.aw import AWAgent
from services.agent.src.agent.agents.ielts_writing import IeltsWritingAgent
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.base import BaseAgent
from services.agent.src.agent.registry import AgentRegistry
from shared.src.shared.catalog import SUPPORTED_PAIRS

_AGENTS: dict[tuple[str, str], type[BaseAgent]] = {
    ("gre", "quant"): QuantAgent,
    ("gre", "aw"): AWAgent,
    ("ielts", "writing"): IeltsWritingAgent,
}


def build_registry() -> AgentRegistry:
    """Return an AgentRegistry populated with the chat agents, one per catalog pair."""
    if set(_AGENTS) != set(SUPPORTED_PAIRS):
        raise RuntimeError(
            f"agent registrations {sorted(_AGENTS)} do not match "
            f"shared.catalog.SUPPORTED_PAIRS {sorted(SUPPORTED_PAIRS)}"
        )
    registry = AgentRegistry()
    for (test_type, section), agent_cls in _AGENTS.items():
        registry.register(test_type, section, agent_cls)
    return registry
