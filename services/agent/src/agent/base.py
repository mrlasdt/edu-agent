from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from shared.src.shared.events import Event
from shared.src.shared.models import Session


class BaseAgent(ABC):
    """
    Minimal interface every agent must implement.
    Registered in AgentRegistry by (test_type, section).
    Adding a new test type means subclassing BaseAgent and registering —
    no changes to the registry or orchestrator.
    """

    @abstractmethod
    def run(
        self, session: Session, message: str, trace_id: str = ""
    ) -> AsyncGenerator[Event, None]:
        """
        Process a turn and yield its events (see `shared.events`): `Token`s of the reply,
        interleaved with what the agent reports about it (sources, verdicts, ...).
        Session carries mode (tutor/solve) and history.
        """
        ...  # pragma: no cover
