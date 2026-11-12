"""
What an agent reports during a turn: the one list of event types, shared by the agent service
(which yields and consumes them) and the evals (which read them).

`BaseAgent.run` yields these and nothing else. Before, it yielded `str | dict`, and a dict was
metadata recognised by which key it held, so a renamed key was a silent miss. Each event is a
frozen model that forbids unknown fields, and `Event` is a discriminated union on `kind`, so a
misspelled field fails when the event is built and a consumer's `match` can be checked for
completeness.

The events cross the HTTP hop to the gateway as text, not as events: the answer is the stream,
the source count a response header and the rest the trailer (see `trailer.py`). That encoding is
unchanged; the agent service's `/generate` is the one place that turns events into it.
"""
from __future__ import annotations

from typing import Annotated, Any, Final, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

Verdict = Literal["agrees", "disagrees", "not_checkable"]
VERDICTS: Final = get_args(Verdict)
AGREES: Final = "agrees"
DISAGREES: Final = "disagrees"
NOT_CHECKABLE: Final = "not_checkable"


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Token(_Event):
    """A piece of the reply the candidate reads."""

    kind: Literal["token"] = "token"
    text: str


class Sources(_Event):
    """How many corpus notes the turn retrieved; the gateway checks the reply's citations
    against it. Sent before the first token."""

    kind: Literal["sources"] = "sources"
    count: int = Field(ge=0)


class Verified(_Event):
    """Whether the math verifier accepted the final answer. Solve mode, after the answer text."""

    kind: Literal["verified"] = "verified"
    passed: bool


class IndependentCheck(_Event):
    """What the independent answer check said (see `agent.independent_check`). Solve mode, just
    before `Verified`. `solver`, the value the solver computed, is set only for "disagrees".
    `reason` says why a check was "not_checkable"; it is for the turn record, so the turn's report
    leaves it empty and `public()` never includes it."""

    kind: Literal["independent_check"] = "independent_check"
    verdict: Verdict
    solver: str = ""
    reason: str = ""

    def public(self) -> dict[str, str]:
        """The trailer's entry for it: the verdict, and the solver's value when they disagree."""
        return {"verdict": self.verdict, **({"solver": self.solver} if self.solver else {})}


class SampleEssay(_Event):
    """The reply is a complete sample essay; the web app words the notice for it."""

    kind: Literal["sample_essay"] = "sample_essay"


class CitationsStripped(_Event):
    """Whether citations that did not match a retrieved note were removed from the reply."""

    kind: Literal["citations_stripped"] = "citations_stripped"
    stripped: bool


class ToolCall(_Event):
    """One math tool call the model made, as the eval transcript shows it."""

    name: str
    arguments: str  # the model's JSON, shortened
    ok: bool
    ms: float
    result: str  # the tool's JSON result, shortened
    round: int


class ToolTrace(_Event):
    """Every math tool call of the turn. For evals and debugging; never streamed to the learner."""

    kind: Literal["tool_trace"] = "tool_trace"
    calls: tuple[ToolCall, ...]


Event = Annotated[
    Token | Sources | Verified | IndependentCheck | SampleEssay | CitationsStripped | ToolTrace,
    Field(discriminator="kind"),
]

_EVENTS: TypeAdapter[Event] = TypeAdapter(Event)


def parse_event(data: Any) -> Event:
    """The event a dict describes (`{"kind": "sources", "count": 3}`); raises ValidationError for
    an unknown kind, a missing field or a misspelled one."""
    return _EVENTS.validate_python(data)
