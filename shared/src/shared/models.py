from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Mode(str, Enum):
    tutor = "tutor"
    solve = "solve"


class Complexity(str, Enum):
    basic = "basic"
    advanced = "advanced"  # architected but not implemented in v1


class MessageRole(str, Enum):
    candidate = "candidate"
    agent = "agent"


class Message(BaseModel):
    role: MessageRole
    content: str


class Session(BaseModel):
    """
    Carries test_type and section as first-class fields so new test types
    (ielts, gmat, …) can be added by registering new agents — no structural change needed here.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    candidate_id: str
    test_type: str  # "gre" in v1; "ielts", "gmat", … in v2
    section: str  # "quant", "aw" for gre; section names vary per test_type
    mode: Mode = Mode.tutor
    complexity: Complexity = Complexity.basic
    history: list[Message] = Field(default_factory=list)
    personal_style_enabled: bool = False

    def switch_mode(self, new_mode: Mode) -> "Session":
        """Return a new Session with mode switched. Immutable update."""
        return self.model_copy(update={"mode": new_mode})

    def reset_mode(self) -> "Session":
        """Reset mode to tutor at the start of a new question."""
        return self.model_copy(update={"mode": Mode.tutor})

    def append_message(self, role: MessageRole, content: str) -> "Session":
        return self.model_copy(
            update={"history": [*self.history, Message(role=role, content=content)]}
        )
