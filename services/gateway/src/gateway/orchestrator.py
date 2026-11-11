from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from shared.src.shared.catalog import SUPPORTED_PAIRS, describe_supported, is_supported
from shared.src.shared.models import Session


@dataclass(frozen=True)
class RouteResult:
    """The Orchestrator accepted this session: its (test_type, section) is supported."""
    session: Session


@dataclass(frozen=True)
class ClarificationTurn:
    """
    The Orchestrator could not route to an agent — either the message was
    incomplete, or the (test_type, section) pair is not supported.
    The prompt is shown to the Candidate to guide their next message.
    """
    prompt: str
    reason: str  # internal; used for logging / guardrail events


class Orchestrator:
    """
    Gateway layer between the chat API and the agent service.

    Steps (per turn):
      1. Validate — reject blank messages immediately
      2. Route    — check (test_type, section) against the catalog of supported pairs
                    On miss → ClarificationTurn (not an exception; the agent service is not called)
      3. Return   — RouteResult carries the validated session

    Step 2b (enrichment — a cheap LLM detecting incomplete questions) runs in the
    gateway's /chat handler via check_completeness, after this class has validated and
    routed the turn.
    Not built: complexity classification (Basic vs Advanced), the complexity-escalation
    suggestion and the topic-relevance label. There is no code for them, not even a stub
    (issue 10 and ADR-0002 describe the design).
    """

    def __init__(self, supported: Collection[tuple[str, str]] = SUPPORTED_PAIRS) -> None:
        self._supported = supported

    async def process(self, session: Session, message: str) -> RouteResult | ClarificationTurn:
        # Step 1: validate
        if not message or not message.strip():
            return ClarificationTurn(
                prompt="It looks like your message was empty. What would you like to work on?",
                reason="empty_message",
            )

        # Step 2: route
        if not is_supported(session.test_type, session.section, self._supported):
            return ClarificationTurn(
                prompt=(
                    f"I'm not set up to help with {session.test_type.upper()} "
                    f"{session.section} yet. "
                    f"Currently available: {describe_supported(self._supported)}. "
                    "Please start a new session with one of these."
                ),
                reason="agent_not_registered",
            )

        # Step 3: return route result
        return RouteResult(session=session)
