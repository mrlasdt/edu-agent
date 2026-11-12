"""The checks a /chat turn passes before the agent is called, in the order they run."""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from services.gateway.src.gateway.chat.turn import ChatTurn, is_coach
from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from services.gateway.src.gateway.guardrails.input import (
    ModerationUnavailableError,
    check_completeness,
    check_content_safety,
)
from services.gateway.src.gateway.guardrails.stream import INPUT_PII_NOTICE
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.ielts_coach import (
    NO_REPORT_NOTICE,
    SOLVE_MODE_NOTICE,
    UNKNOWN_ATTEMPT_NOTICE,
    find_report,
)
from services.gateway.src.gateway.orchestrator import ClarificationTurn, RouteResult
from shared.src.shared.models import Mode
from shared.src.shared.turn_outcome import Outcome

_log = logging.getLogger(__name__)

_MODERATION_UNAVAILABLE = "The safety check is unavailable right now. Please try again in a moment."

# A gate runs before the agent is called. Not allowed ends the turn with the verdict's message
# and outcome; allowed goes on to the next gate, after sending the verdict's message if it has one.
Gate = Callable[[ChatTurn], Awaitable[Verdict]]


async def rate_limit(turn: ChatTurn) -> Verdict:
    allowed, message = turn.limiter.check(turn.req.candidate_id)
    if allowed:
        return Verdict.allow()
    emit_guardrail_event("candidate", "rate_limit", "fail", turn.trace_id)
    return Verdict.stop(Outcome.rate_limited, f"{message} Please wait a moment and try again.")


async def content_safety(turn: ChatTurn) -> Verdict:
    try:
        return await check_content_safety(turn.message, turn.trace_id, turn.meter)
    except ModerationUnavailableError:
        return Verdict.stop(Outcome.moderation_unavailable, _MODERATION_UNAVAILABLE)


async def input_pii_notice(turn: ChatTurn) -> Verdict:
    # After the safety check, so a self-harm reply is only the supportive text.
    return Verdict.allow(INPUT_PII_NOTICE if turn.input_pii else "")


async def completeness(turn: ChatTurn) -> Verdict:
    if is_coach(turn):  # the check is about a GRE question; a coach turn is about the report
        return Verdict.allow()
    # Fail open: a broken completeness check must not block a turn that the other
    # guardrails already passed, same as the LLM stage of the answer-leak check.
    try:
        return await check_completeness(turn.message, turn.session, turn.trace_id, turn.meter)
    except Exception:
        _log.warning("completeness check failed; skipping", exc_info=True)
        turn.flags.append("completeness_check_failed")
        return Verdict.allow()


async def route(turn: ChatTurn) -> Verdict:
    result = await turn.orchestrator.process(turn.session, turn.message)
    if isinstance(result, ClarificationTurn):
        return Verdict.stop(Outcome.clarification_requested, result.prompt)
    assert isinstance(result, RouteResult)  # narrows type; route succeeded
    return Verdict.allow()


async def ielts_report(turn: ChatTurn) -> Verdict:
    """The IELTS Writing coach's turn needs a graded report: find it and keep its digest for the
    agent. Free (a local lookup), so it runs before the gates that call a model. Every other
    turn passes."""
    if not is_coach(turn):
        return Verdict.allow()
    if turn.req.mode == Mode.solve:  # "reveal the worked artifact" has no IELTS meaning
        return Verdict.stop(Outcome.clarification_requested, SOLVE_MODE_NOTICE)
    found = find_report(turn.attempts(), turn.req.attempt_id)
    if found is None:
        notice = UNKNOWN_ATTEMPT_NOTICE if turn.req.attempt_id else NO_REPORT_NOTICE
        return Verdict.stop(Outcome.clarification_requested, notice)
    turn.report_digest = found.digest
    turn.meter.annotate(ielts_attempt=found.attempt_id)
    return Verdict.allow()


# In order: the first gate that does not allow the turn ends it. The free checks come before the
# two that call a model, so a blank or unsupported request is answered without paying for either.
GATES: tuple[Gate, ...] = (
    rate_limit, route, ielts_report, content_safety, input_pii_notice, completeness,
)
