from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from services.gateway.src.gateway.agent_client import AgentClient, get_agent_client
from services.gateway.src.gateway.chat.stream_record import recorded_stream
from services.gateway.src.gateway.chat.turn import ChatRequest, ChatTurn
from services.gateway.src.gateway.feedback_api import router as feedback_router
from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.guardrails.stream import redact_input
from services.gateway.src.gateway.ielts_api import (
    get_store,
    recover_interrupted_attempts,
)
from services.gateway.src.gateway.ielts_api import router as ielts_router
from services.gateway.src.gateway.limits import chat_limiter, feedback_limiter, ielts_limiter
from services.gateway.src.gateway.orchestrator import Orchestrator
from shared.src.shared.feedback import default_capture_log
from shared.src.shared.models import Message, Session
from shared.src.shared.observability import setup_observability
from shared.src.shared.turn_meter import TurnMeter
from shared.src.shared.turn_outcome import Outcome

_log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # honour a test's dependency override, so the sweep sees the store the requests will use
    get_attempt_store = app.dependency_overrides.get(get_store, get_store)
    recover_interrupted_attempts(get_attempt_store())
    # build the orchestrator and the limiters now, so a broken one fails the start, not the
    # first request
    app.dependency_overrides.get(get_orchestrator, get_orchestrator)()
    app.dependency_overrides.get(chat_limiter, chat_limiter)()
    ielts_limiter()
    feedback_limiter()
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


# Response header carrying the turn's trace id, so the browser can attach feedback to it.
TRACE_ID_HEADER = "X-Trace-Id"


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    req: ChatRequest,
    limiter: Annotated[RateLimiter, Depends(chat_limiter)],
    orchestrator: Annotated[Orchestrator, Depends(get_orchestrator)],
    agent: Annotated[AgentClient, Depends(get_agent_client)],
) -> StreamingResponse:
    """
    Single turn: rate limit → orchestrator route → IELTS report lookup (coach turns only) →
    content-safety guardrail → input PII notice → completeness check (the gates, in
    `chat.gates.GATES`) → stream the agent's answer through the reply guard (answer leak,
    citation check, PII redaction). Clarifications, refusals and every guardrail decision are
    streamed as plain text so the response shape is uniform regardless of outcome.
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
        # like the startup sweep, honour a test's override; opened only if the turn needs it
        attempts=app.dependency_overrides.get(get_store, get_store),
    )

    # Opt-in (TURN_CAPTURE_TEXT=1): keep the question (as the model saw it, PII redacted) and
    # the reply in a local file keyed by trace id, so a thumbs-down can become a draft eval item.
    capture_log = default_capture_log()

    return StreamingResponse(
        recorded_stream(turn, capture_log),
        media_type="text/plain; charset=utf-8",
        headers={TRACE_ID_HEADER: trace_id},
    )
