from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from shared.src.shared.models import Session


class BaseAgent(ABC):
    """
    Minimal interface every agent must implement.
    Registered in AgentRegistry by (test_type, section).
    Adding a new test type means subclassing BaseAgent and registering —
    no changes to the registry or orchestrator.
    """

    @abstractmethod
    async def run(self, session: Session, message: str) -> AsyncGenerator[str, None]:
        """
        Process a turn and yield response tokens as an async generator.
        Session carries mode (tutor/solve) and history.
        """
        ...  # pragma: no cover
