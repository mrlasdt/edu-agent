"""One /chat turn: the request body and the state its stages share."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from services.gateway.src.gateway.agent_client import AgentClient
from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.ielts_coach import COACH_PAIR
from services.gateway.src.gateway.ielts_store import AttemptStore
from services.gateway.src.gateway.orchestrator import Orchestrator
from shared.src.shared.models import Message, Mode, Session
from shared.src.shared.turn_meter import TurnMeter
from shared.src.shared.wire import (
    MAX_CANDIDATE_ID_CHARS,
    MAX_HISTORY_ITEMS,
    MAX_MESSAGE_CHARS,
    ClientHistoryItem,
)


# A request over a size cap (`shared.wire`) is rejected with 422 before any guardrail or model
# call runs.
class ChatRequest(BaseModel):
    candidate_id: str = Field(max_length=MAX_CANDIDATE_ID_CHARS)
    test_type: str
    section: str
    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    mode: Mode = Mode.tutor
    history: list[ClientHistoryItem] = Field(default_factory=list, max_length=MAX_HISTORY_ITEMS)
    personal_style_enabled: bool = False
    # IELTS Writing coach only: the graded attempt to coach from; absent means the latest one.
    attempt_id: str | None = Field(default=None, max_length=64)


@dataclass
class ChatTurn:
    """Everything one /chat turn's stages read: the request as the model will see it, the
    collaborators FastAPI injected, and the turn's measurement record."""

    req: ChatRequest
    trace_id: str
    message: str  # PII-redacted
    input_pii: bool
    history: list[Message]
    session: Session
    meter: TurnMeter
    flags: list[str]
    limiter: RateLimiter
    orchestrator: Orchestrator
    agent: AgentClient
    attempts: Callable[[], AttemptStore]  # read only by the IELTS coach's turns
    report_digest: str = ""  # the coach's graded report, set by `ielts_report`


def is_coach(turn: ChatTurn) -> bool:
    return (turn.req.test_type, turn.req.section) == COACH_PAIR
