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
    "I can't see a marked IELTS Reading result for you yet. Sit a Reading mock at "
    "/ielts-mock, and once you have submitted it I can coach you from it."
)


class IeltsReadingAgent(BaseAgent[Turn]):
    """
    IELTS Reading coach: one model call per turn, coaching from the learner's marked Reading
    mock. The marks come from code, not a model; the gateway builds the digest (raw score, band,
    missed question types, and per missed question the evidence sentence) and sends it as
    `session.report_digest`, as the Writing coach's report is sent.

    Tutor mode only, and no tools: there is no essay to re-mark. The digest is delimited data and
    never part of the system prompt. The turn is metered like a GRE tutor turn.
    """

    ROLE = "ielts_reading_coach"
    SINGLE_MODE = Mode.tutor  # the gateway answers solve mode itself; the coach has one mode

    async def tutor_turn(self, turn: Turn) -> AsyncGenerator[Event, None]:
        if not turn.session.report_digest:
            yield Token(text=_NO_REPORT)
            return
        messages = build_messages(
            turn.session, turn.message, read_prompt(self.ROLE, turn.mode.value, turn.version),
            f"\n\n{turn.session.report_digest}",
        )
        async for text in token_stream(self.stream(turn, messages)):
            yield Token(text=text)

    solve_turn = tutor_turn
