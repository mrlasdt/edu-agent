from __future__ import annotations

from collections.abc import AsyncGenerator

from services.agent.src.agent.base import BaseAgent
from services.agent.src.agent.prompting import build_messages
from shared.src.shared.config import load_model_config
from shared.src.shared.events import Event, Token
from shared.src.shared.models import Mode, Session
from shared.src.shared.observability import build_llm_metadata
from shared.src.shared.prompts import read_prompt
from shared.src.shared.turn_meter import TurnMeter

_ROLE = "ielts_coach"

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


class IeltsWritingAgent(BaseAgent):
    """
    IELTS Writing coach: one model call per turn, coaching from the learner's graded Writing
    Task 2 report. The gateway builds the report digest from the attempt store (this service
    cannot read it) and sends it as `session.report_digest`.

    Tutor mode only: no retries, no critics, no tools. The prompt forbids writing the essay and
    treats the digest as data. The turn is metered like a GRE tutor turn (`TurnMeter`).
    """

    def __init__(self, model_config_path: str | None = None) -> None:
        self._model_config = load_model_config(model_config_path)

    async def run(
        self, session: Session, message: str, trace_id: str = ""
    ) -> AsyncGenerator[Event, None]:
        mode = Mode.tutor.value  # the gateway answers solve mode itself; the coach has one mode
        version = self._model_config.prompt_version(_ROLE, mode)
        meter = TurnMeter(trace_id, "agent", _ROLE, mode, version)
        try:
            if not session.report_digest:
                yield Token(text=_NO_REPORT)
                return
            messages = build_messages(
                session, message, read_prompt(_ROLE, mode, version),
                _report_block(session.report_digest),
            )
            response = meter.stream(
                model=self._model_config.litellm_model(_ROLE),
                messages=messages,
                temperature=self._model_config.role(_ROLE).temperature,
                **build_llm_metadata(trace_id=trace_id, prompt_version=version),
            )
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield Token(text=content)
        except Exception as exc:
            meter.finish(error=type(exc).__name__)
            raise
        finally:
            meter.finish()
