from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

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
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """
        Process a turn and yield response tokens as an async generator,
        optionally followed by a final dict of turn metadata.
        Session carries mode (tutor/solve) and history.
        """
        ...  # pragma: no cover
