from __future__ import annotations

import codecs
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from services.gateway.src.gateway.agent_client import AgentClient, get_agent_client
from services.gateway.src.gateway.feedback_api import router as feedback_router
from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from services.gateway.src.gateway.guardrails.input import (
    ModerationUnavailableError,
    check_completeness,
    check_content_safety,
)
from services.gateway.src.gateway.guardrails.output import check_answer_leak
from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.guardrails.stream import (
    INPUT_PII_NOTICE,
    LLM_LEAK_NOTICE,
    ReplyGuard,
    redact_input,
)
from services.gateway.src.gateway.guardrails.verdict import Verdict
from services.gateway.src.gateway.ielts_api import (
    get_store,
    recover_interrupted_attempts,
)
from services.gateway.src.gateway.ielts_api import router as ielts_router
from services.gateway.src.gateway.orchestrator import (
    ClarificationTurn,
    Orchestrator,
    RouteResult,
)
from shared.src.shared.feedback import capture_turn, default_capture_log
from shared.src.shared.models import Message, Mode, Session
from shared.src.shared.observability import setup_observability
from shared.src.shared.trailer import MARK, TrailerSplitter, encode, independent_check_trailer
from shared.src.shared.turn_meter import TurnLog, TurnMeter
from shared.src.shared.turn_outcome import Outcome
from shared.src.shared.wire import (
    MAX_CANDIDATE_ID_CHARS,
    MAX_HISTORY_ITEMS,
    MAX_MESSAGE_CHARS,
    ClientHistoryItem,
    GenerateRequest,
    HistoryItem,
)

_log = logging.getLogger(__name__)

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # honour a test's dependency override, so the sweep sees the store the requests will use
    get_attempt_store = app.dependency_overrides.get(get_store, get_store)
    recover_interrupted_attempts(get_attempt_store())
    # build the orchestrator now, so a broken one fails the start, not the first request
    app.dependency_overrides.get(get_orchestrator, get_orchestrator)()
    yield


app = FastAPI(title="Gateway Service", lifespan=_lifespan)
app.include_router(ielts_router)
app.include_router(feedback_router)

setup_observability()


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """
    FastAPI's default 422 echoes each offending value back as `input`. For a rejected
    5 MB message that would send the 5 MB straight back, so drop the echo and keep the
    rest (type, loc, msg) of every error.
    """
    errors = [{k: v for k, v in e.items() if k not in ("input", "ctx")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@lru_cache
def get_orchestrator() -> Orchestrator:
    """The orchestrator validates + routes against the shared catalog of supported pairs, the same
    list the agent service's registry must match, so routing decisions here match what the agent
    will accept."""
    return Orchestrator()


@lru_cache
def get_rate_limiter() -> RateLimiter:
    return RateLimiter()



# Response header carrying the turn's trace id, so the browser can attach feedback to it.
TRACE_ID_HEADER = "X-Trace-Id"

_MODERATION_UNAVAILABLE = "The safety check is unavailable right now. Please try again in a moment."


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


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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


# A gate runs before the agent is called. Not allowed ends the turn with the verdict's message
# and outcome; allowed goes on to the next gate, after sending the verdict's message if it has one.
Gate = Callable[[ChatTurn], Awaitable[Verdict]]


async def _rate_limit(turn: ChatTurn) -> Verdict:
    allowed, message = turn.limiter.check(turn.req.candidate_id)
    if allowed:
        return Verdict.allow()
    emit_guardrail_event("candidate", "rate_limit", "fail", turn.trace_id)
    return Verdict.stop(Outcome.rate_limited, f"{message} Please wait a moment and try again.")


async def _content_safety(turn: ChatTurn) -> Verdict:
    try:
        return await check_content_safety(turn.message, turn.trace_id, turn.meter)
    except ModerationUnavailableError:
        return Verdict.stop(Outcome.moderation_unavailable, _MODERATION_UNAVAILABLE)


async def _input_pii_notice(turn: ChatTurn) -> Verdict:
    # After the safety check, so a self-harm reply is only the supportive text.
    return Verdict.allow(INPUT_PII_NOTICE if turn.input_pii else "")


async def _completeness(turn: ChatTurn) -> Verdict:
    # Fail open: a broken completeness check must not block a turn that the other
    # guardrails already passed, same as the LLM stage of the answer-leak check.
    try:
        return await check_completeness(turn.message, turn.session, turn.trace_id, turn.meter)
    except Exception:
        _log.warning("completeness check failed; skipping", exc_info=True)
        turn.flags.append("completeness_check_failed")
        return Verdict.allow()


async def _route(turn: ChatTurn) -> Verdict:
    result = await turn.orchestrator.process(turn.session, turn.message)
    if isinstance(result, ClarificationTurn):
        return Verdict.stop(Outcome.clarification_requested, result.prompt)
    assert isinstance(result, RouteResult)  # narrows type; route succeeded
    return Verdict.allow()


# In order: the first gate that does not allow the turn ends it. The free checks come before the
# two that call a model, so a blank or unsupported request is answered without paying for either.
_GATES: tuple[Gate, ...] = (_rate_limit, _route, _content_safety, _input_pii_notice, _completeness)


async def _answer(turn: ChatTurn) -> AsyncGenerator[bytes, None]:
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
        trace_id=turn.trace_id,
    )
    meta: dict[str, Any] = {}
    guard: ReplyGuard | None = None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    trailer = TrailerSplitter()  # the agent's metadata is not part of the answer text
    async for chunk in turn.agent.stream(payload, meta):
        if guard is None:
            guard = ReplyGuard(req.mode, turn.trace_id, meta.get("num_sources"))
        text = guard.feed(trailer.feed(decoder.decode(chunk)))
        if text:
            yield text.encode("utf-8")
        if guard.stopped:
            meter.annotate(outcome=Outcome.answer_leak_blocked)
            return
    if guard is not None:
        tail = guard.feed(trailer.feed(decoder.decode(b"", final=True))) + guard.finish()
        if guard.stopped:
            meter.annotate(outcome=Outcome.answer_leak_blocked)
        if guard.citations_removed:
            flags.append("citations_trimmed")
        if guard.pii_redacted:
            flags.append("output_pii_redacted")
        if (
            not guard.stopped
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


async def _guarded_stream(turn: ChatTurn) -> AsyncGenerator[bytes, None]:
    for gate in _GATES:
        verdict = await gate(turn)
        if not verdict.allowed:
            turn.meter.annotate(outcome=verdict.outcome)
        if verdict.message:
            yield verdict.message.encode("utf-8")
        if not verdict.allowed:
            return
    async for piece in _answer(turn):
        yield piece


async def _recorded_stream(
    turn: ChatTurn, capture_log: TurnLog | None
) -> AsyncGenerator[bytes, None]:
    """The turn's bytes, with its first-byte time, flags, final outcome and optional text capture
    written to the turn record however the stream ends."""
    meter = turn.meter
    started = time.perf_counter()
    first_byte_seen = False
    error: str | None = None
    completed = False
    sent: list[bytes] = []
    try:
        async for piece in _guarded_stream(turn):
            if not first_byte_seen:
                first_byte_seen = True
                meter.annotate(first_byte_ms=(time.perf_counter() - started) * 1000)
            if capture_log is not None:
                sent.append(piece)
            yield piece
        completed = True
    except Exception as exc:
        error = type(exc).__name__
        meter.annotate(outcome=Outcome.error)
        raise
    finally:
        # The server abandons this generator when the client disconnects (Stop, a closed
        # tab): by cancellation, which also closes the agent request, or by closing it.
        if not completed and error is None:
            meter.annotate(outcome=Outcome.client_cancelled)
        meter.annotate(flags=turn.flags)
        meter.finish(error=error)
        if capture_log is not None:
            reply = b"".join(sent).decode("utf-8", errors="replace").partition(MARK)[0]
            capture_turn(capture_log, turn.trace_id, turn.message, reply)


@app.post("/chat")
async def chat(
    req: ChatRequest,
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    orchestrator: Annotated[Orchestrator, Depends(get_orchestrator)],
    agent: Annotated[AgentClient, Depends(get_agent_client)],
) -> StreamingResponse:
    """
    Single turn: rate limit → content-safety guardrail → input PII notice → completeness
    check → orchestrator route (the gates, in `_GATES`) → stream the agent's answer through the
    reply guard (answer leak, citation check, PII redaction). Clarifications, refusals and
    every guardrail decision are streamed as plain text so the response shape is
    uniform regardless of outcome.
    """
    trace_id = str(uuid.uuid4())
    message, input_pii = redact_input(req.message, trace_id)
    history = [Message(role=h.role, content=redact_input(h.content, trace_id)[0])
               for h in req.history]
    session = Session(
        candidate_id=req.candidate_id,
        test_type=req.test_type,
        section=req.section,
        mode=req.mode,
        history=history,
        personal_style_enabled=req.personal_style_enabled,
    )

    # Measures the gateway's share of the turn (guardrails and their model calls, outcome, time
    # to first byte); the agent service writes its own record under the same trace id.
    meter = TurnMeter(trace_id, "gateway", f"{req.test_type}/{req.section}", req.mode.value)
    meter.annotate(outcome=Outcome.ok)
    turn = ChatTurn(
        req=req,
        trace_id=trace_id,
        message=message,
        input_pii=input_pii,
        history=history,
        session=session,
        meter=meter,
        flags=["input_pii_redacted"] if input_pii else [],
        limiter=limiter,
        orchestrator=orchestrator,
        agent=agent,
    )

    # Opt-in (TURN_CAPTURE_TEXT=1): keep the question (as the model saw it, PII redacted) and
    # the reply in a local file keyed by trace id, so a thumbs-down can become a draft eval item.
    capture_log = default_capture_log()

    return StreamingResponse(
        _recorded_stream(turn, capture_log),
        media_type="text/plain; charset=utf-8",
        headers={TRACE_ID_HEADER: trace_id},
    )
