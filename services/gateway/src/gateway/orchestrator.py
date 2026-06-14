from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from services.agent.src.agent.registry import AgentNotRegisteredError, AgentRegistry
from shared.src.shared.models import Session

if TYPE_CHECKING:
    from services.agent.src.agent.base import BaseAgent


@dataclass(frozen=True)
class RouteResult:
    """The Orchestrator resolved a registered agent for this session."""
    agent_cls: type[BaseAgent]
    session: Session


@dataclass(frozen=True)
class ClarificationTurn:
    """
    The Orchestrator could not route to an agent — either the message was
    incomplete, or the (test_type, section) pair is not registered.
    The prompt is shown to the Candidate to guide their next message.
    """
    prompt: str
    reason: str  # internal; used for logging / guardrail events


class Orchestrator:
    """
    Gateway layer between the chat API and the agent service.

    Steps (per turn):
      1. Validate — reject blank messages immediately
      2. Route    — look up (test_type, section) in the registry
                    On miss → ClarificationTurn (not an exception; the agent service is not called)
      3. Return   — RouteResult carries the agent class and validated session

    Steps 2b (enrichment — Haiku LLM detecting incomplete questions) and
    2c (complexity classification) are stub implementations in v1 and will
    be filled in by issue 10.
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    async def process(self, session: Session, message: str) -> RouteResult | ClarificationTurn:
        # Step 1: validate
        if not message or not message.strip():
            return ClarificationTurn(
                prompt="It looks like your message was empty. What would you like to work on?",
                reason="empty_message",
            )

        # Step 2: route
        try:
            agent_cls = self._registry.get(session.test_type, session.section)
        except AgentNotRegisteredError:
            available = [
                f"{t}/{s}" for (t, s) in sorted(self._registry._registry.keys())
            ]
            available_str = ", ".join(available) if available else "none yet"
            return ClarificationTurn(
                prompt=(
                    f"I'm not set up to help with {session.test_type.upper()} "
                    f"{session.section} yet. "
                    f"Currently available: {available_str}. "
                    "Please start a new session with one of these."
                ),
                reason="agent_not_registered",
            )

        # Step 3: return route result
        return RouteResult(agent_cls=agent_cls, session=session)
