"""
The request the gateway sends the agent service's `/generate`, defined once for both sides.

The gateway builds a `GenerateRequest` and posts its JSON; the agent service takes the same
model as its request body. Before, each side kept its own copy of the fields (and of
`HistoryItem`, with a `max_length` on one side only), so a field added to one was silently
dropped by the other.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from shared.src.shared.models import MessageRole, Mode

# What a client may send /chat. It goes to the model (and through the PII and safety checks), so
# an unbounded body is an unbounded bill: the gateway rejects a request over a cap with 422 before
# any guardrail or model call runs.
MAX_MESSAGE_CHARS = 4_000
# The cap bounds the request. The IELTS writing agent still sends the whole history to the model;
# the Quant agent further bounds it with its ContextBudget (services/agent/src/agent/context.py).
MAX_HISTORY_ITEMS = 20
MAX_HISTORY_ITEM_CHARS = 4_000
# candidate_id keys the rate limiter, so it must not be an arbitrarily large string either.
MAX_CANDIDATE_ID_CHARS = 128

# What the gateway may forward. It redacts emails and phone numbers first, and "[REDACTED]" (10
# characters) is longer than the shortest address it replaces ("a@b.co", 6), so redaction can grow
# a text by up to 10/6. The agent service holds the hop to this bound, not to the client's.
MAX_FORWARDED_CHARS = MAX_MESSAGE_CHARS * 10 // 6 + 1


class HistoryItem(BaseModel):
    """One earlier turn, as the gateway sends it to the agent service."""

    # Extra keys are ignored, not forbidden: the browser sends its whole message objects (id,
    # citations, ...) as history, and /chat reads them through `ClientHistoryItem`.
    role: MessageRole
    content: str = Field(max_length=MAX_FORWARDED_CHARS)


class ClientHistoryItem(HistoryItem):
    """One earlier turn as a client may send it: the same item, under the client's cap."""

    content: str = Field(max_length=MAX_HISTORY_ITEM_CHARS)


class GenerateRequest(BaseModel):
    # An unknown field is an error: a misspelled name would otherwise fall back to its default.
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(max_length=MAX_CANDIDATE_ID_CHARS)
    test_type: str
    section: str
    message: str = Field(max_length=MAX_FORWARDED_CHARS)
    mode: Mode = Mode.tutor
    history: list[HistoryItem] = Field(default_factory=list, max_length=MAX_HISTORY_ITEMS)
    personal_style_enabled: bool = False
    trace_id: str = ""
