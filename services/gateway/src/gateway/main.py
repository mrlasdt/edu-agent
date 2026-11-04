from __future__ import annotations

import codecs
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from services.agent.src.agent.bootstrap import build_registry
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
from services.gateway.src.gateway.ielts_api import router as ielts_router
from services.gateway.src.gateway.orchestrator import (
    ClarificationTurn,
    Orchestrator,
    RouteResult,
)
from shared.src.shared.config import get_settings
from shared.src.shared.models import Message, MessageRole, Mode, Session
from shared.src.shared.observability import setup_observability
from shared.src.shared.trailer import TrailerSplitter, encode
from shared.src.shared.turn_meter import TurnMeter

_log = logging.getLogger(__name__)

app = FastAPI(title="Gateway Service")
app.include_router(ielts_router)

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

# The orchestrator validates + routes; the registry is the same one the agent
# service uses, so routing decisions here match what /generate will accept.
_orchestrator = Orchestrator(registry=build_registry())
_rate_limiter = RateLimiter()


_DEFAULT_CLARIFICATION = "Could you share the full question so I can help?"
_MODERATION_UNAVAILABLE = "The safety check is unavailable right now. Please try again in a moment."


# Request size caps. /chat sends the message and the whole history to the model (and through
# the PII and safety checks), so an unbounded body is an unbounded bill. A request over a cap is
# rejected with 422 before any guardrail or model call runs.
MAX_MESSAGE_CHARS = 4_000
MAX_HISTORY_ITEMS = 20
MAX_HISTORY_ITEM_CHARS = 4_000
# candidate_id keys the rate limiter, so it must not be an arbitrarily large string either.
MAX_CANDIDATE_ID_CHARS = 128


class HistoryItem(BaseModel):
    role: MessageRole
    content: str = Field(max_length=MAX_HISTORY_ITEM_CHARS)


class ChatRequest(BaseModel):
    candidate_id: str = Field(max_length=MAX_CANDIDATE_ID_CHARS)
    test_type: str
    section: str
    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    mode: Mode = Mode.tutor
    history: list[HistoryItem] = Field(default_factory=list, max_length=MAX_HISTORY_ITEMS)
    personal_style_enabled: bool = False


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _stream_from_agent(
    payload: dict[str, Any], meta: dict[str, Any] | None = None
) -> AsyncGenerator[bytes, None]:
    """
    Proxy a streaming generation from the Agent Service. Isolated as a seam so
    tests can substitute a fake stream without a running agent service.
    When the agent reports how many sources it retrieved (X-Source-Count), the
    count is stored in `meta["num_sources"]` for the citation check.
    """
    settings = get_settings()
    url = f"{settings.agent_service_url}/generate"
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            count = resp.headers.get("x-source-count")
            if meta is not None and count is not None and count.isdigit():
                meta["num_sources"] = int(count)
            async for chunk in resp.aiter_bytes():
                yield chunk


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """
    Single turn: rate limit → PII redaction → content-safety guardrail →
    completeness check → orchestrator route → stream the agent's answer through the reply guard
    (answer leak, citation check, PII redaction). Clarifications, refusals and
    every guardrail decision are streamed as plain text so the response shape is
    uniform regardless of outcome.
    """
    trace_id = str(uuid.uuid4())
    message, input_pii = redact_input(req.message, trace_id)
    history = [HistoryItem(role=h.role, content=redact_input(h.content, trace_id)[0])
               for h in req.history]
    session = Session(
        candidate_id=req.candidate_id,
        test_type=req.test_type,
        section=req.section,
        mode=req.mode,
        history=[Message(role=h.role, content=h.content) for h in history],
        personal_style_enabled=req.personal_style_enabled,
    )

    # Measures the gateway's share of the turn (guardrails and their model calls, outcome, time
    # to first byte); the agent service writes its own record under the same trace id.
    meter = TurnMeter(trace_id, "gateway", f"{req.test_type}/{req.section}", req.mode.value)
    meter.annotate(outcome="ok")
    flags: list[str] = ["input_pii_redacted"] if input_pii else []

    async def guarded_stream() -> AsyncGenerator[bytes, None]:
        allowed, limit_message = _rate_limiter.check(req.candidate_id)
        if not allowed:
            meter.annotate(outcome="rate_limited")
            emit_guardrail_event("candidate", "rate_limit", "fail", trace_id)
            yield f"{limit_message} Please wait a moment and try again.".encode()
            return
        if input_pii:
            yield INPUT_PII_NOTICE.encode()

        try:
            safe, reason = await check_content_safety(message, trace_id, meter)
        except ModerationUnavailableError:
            meter.annotate(outcome="moderation_unavailable")
            yield _MODERATION_UNAVAILABLE.encode()
            return
        if not safe:
            meter.annotate(outcome="content_blocked")
            yield f"I can't help with that request ({reason}).".encode()
            return

        # Fail open: a broken completeness check must not block a turn that the other
        # guardrails already passed, same as the LLM stage of the answer-leak check.
        try:
            complete, clarification = await check_completeness(message, session, trace_id, meter)
        except Exception:
            _log.warning("completeness check failed; skipping", exc_info=True)
            flags.append("completeness_check_failed")
            complete, clarification = True, ""
        if not complete:
            meter.annotate(outcome="clarification_requested")
            yield (clarification or _DEFAULT_CLARIFICATION).encode("utf-8")
            return

        result = await _orchestrator.process(session, message)
        if isinstance(result, ClarificationTurn):
            meter.annotate(outcome="clarification_requested")
            yield result.prompt.encode("utf-8")
            return

        assert isinstance(result, RouteResult)  # narrows type; route succeeded
        payload = {
            "candidate_id": req.candidate_id,
            "test_type": req.test_type,
            "section": req.section,
            "message": message,
            "mode": req.mode.value,
            "history": [h.model_dump(mode="json") for h in history],
            "personal_style_enabled": req.personal_style_enabled,
            "trace_id": trace_id,
        }
        meta: dict[str, Any] = {}
        guard: ReplyGuard | None = None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        trailer = TrailerSplitter()  # the agent's metadata is not part of the answer text
        async for chunk in _stream_from_agent(payload, meta):
            if guard is None:
                guard = ReplyGuard(req.mode, trace_id, meta.get("num_sources"))
            text = guard.feed(trailer.feed(decoder.decode(chunk)))
            if text:
                yield text.encode("utf-8")
            if guard.stopped:
                meter.annotate(outcome="answer_leak_blocked")
                return
        if guard is not None:
            tail = guard.feed(trailer.feed(decoder.decode(b"", final=True))) + guard.finish()
            if guard.stopped:
                meter.annotate(outcome="answer_leak_blocked")
            if guard.citations_removed:
                flags.append("citations_trimmed")
            if guard.pii_redacted:
                flags.append("output_pii_redacted")
            if (
                not guard.stopped
                and guard.text.strip()
                and await check_answer_leak(guard.text, req.mode, trace_id, meter)
            ):
                tail += LLM_LEAK_NOTICE
                flags.append("answer_leak_flagged")
            verified = trailer.meta.get("math_verified")
            if isinstance(verified, bool):
                tail += encode({"math_verified": verified})
            if tail:
                yield tail.encode("utf-8")

    async def response_stream() -> AsyncGenerator[bytes, None]:
        started = time.perf_counter()
        first_byte_seen = False
        error: str | None = None
        try:
            async for piece in guarded_stream():
                if not first_byte_seen:
                    first_byte_seen = True
                    meter.annotate(first_byte_ms=(time.perf_counter() - started) * 1000)
                yield piece
        except Exception as exc:
            error = type(exc).__name__
            meter.annotate(outcome="error")
            raise
        finally:
            meter.annotate(flags=flags)
            meter.finish(error=error)

    return StreamingResponse(response_stream(), media_type="text/plain; charset=utf-8")
