from __future__ import annotations

from collections.abc import AsyncGenerator

from services.agent.src.agent.base import BaseAgent, Turn
from services.agent.src.agent.prompting import build_messages, token_stream
from shared.src.shared.events import Event, Token
from shared.src.shared.models import Mode
from shared.src.shared.prompts import read_prompt

# What the learner sees when the gateway sent no report. The gateway answers this case itself
# before calling the agent, so this only guards a direct call to /generate.
_NO_REPORT = (
    "I can't see a graded IELTS Writing report for you yet. Take a Writing mock at "
    "/ielts-mock, and once it is marked I can coach you from it."
)


def _report_block(digest: str) -> str:
    """The digest as the last block of the learner's message. It is delimited data (see
    shared/ielts/digest.py), so it is never part of the system prompt."""
    return f"\n\n{digest}" if digest else ""


class IeltsWritingAgent(BaseAgent[Turn]):
    """
    IELTS Writing coach: one model call per turn, coaching from the learner's graded Writing
    Task 2 report. The gateway builds the report digest from the attempt store (this service
    cannot read it) and sends it as `session.report_digest`.

    Tutor mode only: no retries, no critics, no tools, no corpus notes. The prompt forbids
    writing the essay and treats the digest as data. The turn is metered like a GRE tutor turn.
    """

    ROLE = "ielts_coach"
    SINGLE_MODE = Mode.tutor  # the gateway answers solve mode itself; the coach has one mode

    async def tutor_turn(self, turn: Turn) -> AsyncGenerator[Event, None]:
        if not turn.session.report_digest:
            yield Token(text=_NO_REPORT)
            return
        messages = build_messages(
            turn.session, turn.message, read_prompt(self.ROLE, turn.mode.value, turn.version),
            _report_block(turn.session.report_digest),
        )
        async for text in token_stream(self.stream(turn, messages)):
            yield Token(text=text)

    solve_turn = tutor_turn
