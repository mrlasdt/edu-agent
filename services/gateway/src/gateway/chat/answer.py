"""Stream the agent's reply through the reply guard, then the answer-leak check and trailer."""
from __future__ import annotations

import codecs
import logging
from collections.abc import AsyncGenerator
from typing import Any

from services.gateway.src.gateway.chat.turn import ChatTurn, is_coach
from services.gateway.src.gateway.guardrails.output import check_answer_leak
from services.gateway.src.gateway.guardrails.stream import (
    AGENT_UNAVAILABLE_NOTICE,
    LLM_LEAK_NOTICE,
    ReplyGuard,
)
from shared.src.shared.trailer import TrailerSplitter, encode, independent_check_trailer
from shared.src.shared.turn_outcome import Outcome
from shared.src.shared.wire import GenerateRequest, HistoryItem

_log = logging.getLogger(__name__)


async def agent_chunks(
    turn: ChatTurn, payload: GenerateRequest, meta: dict[str, Any], failure: list[Exception]
) -> AsyncGenerator[bytes, None]:
    """The agent's bytes. A failure of the agent call (a 5xx before the first byte, a connection
    that breaks mid-reply, a read timeout) ends the stream and is left in `failure` for `answer`
    to turn into a notice. Only the agent call is guarded: a bug in the guards still raises."""
    try:
        async for chunk in turn.agent.stream(payload, meta):
            yield chunk
    except Exception as exc:  # any failure of the call reads the same to the learner
        failure.append(exc)


async def answer(turn: ChatTurn) -> AsyncGenerator[bytes, None]:
    """Stream the agent's reply through the reply guard (answer leak, citations, PII), then the
    answer-leak model check and the verifier trailer."""
    req, meter, flags = turn.req, turn.meter, turn.flags
    payload = GenerateRequest(
        candidate_id=req.candidate_id,
        test_type=req.test_type,
        section=req.section,
        message=turn.message,
        mode=req.mode,
        history=[HistoryItem(role=m.role, content=m.content) for m in turn.history],
        personal_style_enabled=req.personal_style_enabled,
        report_digest=turn.report_digest,
        trace_id=turn.trace_id,
    )
    meta: dict[str, Any] = {}
    guard: ReplyGuard | None = None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    trailer = TrailerSplitter()  # the agent's metadata is not part of the answer text
    failure: list[Exception] = []
    async for chunk in agent_chunks(turn, payload, meta, failure):
        if guard is None:
            guard = ReplyGuard(
                req.mode, turn.trace_id, meta.get("num_sources"), leak_check=not is_coach(turn)
            )
        text = guard.feed(trailer.feed(decoder.decode(chunk)))
        if text:
            yield text.encode("utf-8")
        if guard.stopped:
            meter.annotate(outcome=Outcome.answer_leak_blocked)
            return
    tail = ""
    if guard is not None:
        tail = guard.feed(trailer.feed(decoder.decode(b"", final=True))) + guard.finish()
        if guard.stopped:
            meter.annotate(outcome=Outcome.answer_leak_blocked)
        if guard.citations_removed:
            flags.append("citations_trimmed")
        if guard.pii_redacted:
            flags.append("output_pii_redacted")
    if failure:
        # Whatever the guard was still holding passed its checks so far: show it, then say why
        # the reply stops. The answer-leak model check and the verifier trailer are skipped,
        # because a partial reply has no verdict to give. A leak the guard caught keeps its outcome.
        stage = "before_first_byte" if guard is None else "midstream"
        meter.annotate(error=type(failure[0]).__name__, degraded_stage=stage)
        if guard is None or not guard.stopped:
            meter.annotate(outcome=Outcome.agent_unavailable)
        _log.warning(
            "agent unavailable (%s): %s", stage, type(failure[0]).__name__, exc_info=failure[0]
        )
        notice = AGENT_UNAVAILABLE_NOTICE if tail.strip() else AGENT_UNAVAILABLE_NOTICE.lstrip()
        yield (tail + notice).encode("utf-8")
        return
    if guard is not None:
        if (
            not is_coach(turn)
            and not guard.stopped
            and guard.text.strip()
            and not (await check_answer_leak(guard.text, req.mode, turn.trace_id, meter)).allowed
        ):
            tail += LLM_LEAK_NOTICE
            flags.append("answer_leak_flagged")
        answer_meta: dict[str, Any] = {}
        verified = trailer.meta.get("math_verified")
        if isinstance(verified, bool):
            answer_meta["math_verified"] = verified
            check = independent_check_trailer(trailer.meta.get("independent_check"))
            if check:
                answer_meta["independent_check"] = check
        if trailer.meta.get("disclaimer") == "essay":
            answer_meta["disclaimer"] = "essay"  # the web app words the notice itself
        if answer_meta:
            tail += encode(answer_meta)
        if tail:
            yield tail.encode("utf-8")
