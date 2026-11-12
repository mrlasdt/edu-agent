from __future__ import annotations

from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class Mode(StrEnum):
    tutor = "tutor"
    solve = "solve"


class MessageRole(StrEnum):
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
    history: list[Message] = Field(default_factory=list)
    personal_style_enabled: bool = False
    # What the gateway resolved for this turn and the agent cannot look up itself: the IELTS
    # Writing coach's digest of the learner's graded report (gateway/ielts/digest.py). Empty for
    # the GRE agents.
    report_digest: str = ""

    def append_message(self, role: MessageRole, content: str) -> Session:
        return self.model_copy(
            update={"history": [*self.history, Message(role=role, content=content)]}
        )
