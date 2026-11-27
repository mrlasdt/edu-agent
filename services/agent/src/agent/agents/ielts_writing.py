from __future__ import annotations

from collections.abc import AsyncGenerator

from services.agent.src.agent.base import BaseAgent, Turn
from services.agent.src.agent.prompting import build_messages, token_stream
from services.agent.src.agent.tools.ielts_tools import (
    IELTS_TOOL_SPECS,
    RevisionTools,
    offers_remark,
)
from services.agent.src.agent.tools.tool_turn import ToolTurn
from services.agent.src.agent.tools.toolbox import ToolBox
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
    gateway/ielts/digest.py), so it is never part of the system prompt."""
    return f"\n\n{digest}" if digest else ""


class IeltsWritingAgent(BaseAgent[Turn]):
    """
    IELTS Writing coach: one model call per turn, coaching from the learner's graded Writing
    Task 2 report. The gateway builds the report digest from the attempt store (this service
    cannot read it) and sends it as `session.report_digest`.

    Tutor mode only: no retries, no critics, no corpus notes. The prompt forbids writing the
    essay and treats the digest as data. The one tool, `mark_revision`, is offered only when the
    learner's message is essay-sized and an attempt is being coached; it asks the gateway for a
    practice re-mark (see tools/ielts_tools.py). The turn is metered like a GRE tutor turn.
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
        if not offers_remark(turn.message, turn.session.attempt_id):
            async for text in token_stream(self.stream(turn, messages)):
                yield Token(text=text)
            return
        # An essay-sized message on a coached attempt: the model may re-mark it, once. The turn
        # is not streamed (the tool loop returns the finished reply); the re-mark card, written
        # from the gateway's numbers, comes first so the learner never depends on the model's
        # retelling of a band.
        remark = RevisionTools(turn.session.attempt_id)
        tools = ToolTurn(ToolBox((IELTS_TOOL_SPECS, remark.call)), math=False)
        text = await tools.reply(
            turn.meter,
            model=self._model_config.litellm_model(self.ROLE),
            messages=messages,
            temperature=self._model_config.role(self.ROLE).temperature,
            obs_meta=turn.obs_meta,
        )
        tools.annotate(turn.meter)
        if remark.notice:
            yield Token(text=remark.notice + "\n\n")
        yield Token(text=text)
        if tools.calls:
            yield tools.trace()

    solve_turn = tutor_turn
