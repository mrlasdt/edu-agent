from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.agent.src.agent.base import BaseAgent


class AgentNotRegisteredError(KeyError):
    """
    Raised when the Orchestrator looks up a (test_type, section) pair
    that has no registered agent. The error message names both the missing
    pair and the known registrations so the caller can diagnose the problem.
    """

    def __init__(self, test_type: str, section: str, known: list[tuple[str, str]]) -> None:
        self.test_type = test_type
        self.section = section
        self.known = known
        known_str = ", ".join(f"({t}, {s})" for t, s in sorted(known)) or "none"
        super().__init__(
            f"No agent registered for (test_type={test_type!r}, section={section!r}). "
            f"Registered pairs: {known_str}"
        )


class AgentRegistry:
    """
    Maps (test_type, section) keys to agent classes.

    This is the primary extension seam: registering a QuantAgent for ("gre", "quant")
    and an IELTSWritingAgent for ("ielts", "writing") is the only change needed to
    support a new test type — no modifications to Orchestrator or other services.
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[str, str], type[BaseAgent]] = {}

    def register(self, test_type: str, section: str, agent_cls: type[BaseAgent]) -> None:
        """Register an agent class for a (test_type, section) pair."""
        self._registry[(test_type, section)] = agent_cls

    def get(self, test_type: str, section: str) -> type[BaseAgent]:
        """
        Return the registered agent class.
        Raises AgentNotRegisteredError with full context if not found.
        """
        key = (test_type, section)
        if key not in self._registry:
            raise AgentNotRegisteredError(
                test_type=test_type,
                section=section,
                known=list(self._registry.keys()),
            )
        return self._registry[key]

    def is_registered(self, test_type: str, section: str) -> bool:
        return (test_type, section) in self._registry
