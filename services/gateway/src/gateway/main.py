from __future__ import annotations

import codecs
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.agent.src.agent.bootstrap import build_registry
from services.gateway.src.gateway.guardrails.events import emit_guardrail_event
from services.gateway.src.gateway.guardrails.input import check_content_safety
from services.gateway.src.gateway.guardrails.output import check_answer_leak
from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.guardrails.stream import (
    INPUT_PII_NOTICE,
    LLM_LEAK_NOTICE,
    ReplyGuard,
    redact_input,
)
from services.gateway.src.gateway.orchestrator import (
    ClarificationTurn,
    Orchestrator,
    RouteResult,
)
from shared.src.shared.config import get_settings
from shared.src.shared.models import Message, MessageRole, Mode, Session

app = FastAPI(title="Gateway Service")

# The orchestrator validates + routes; the registry is the same one the agent
# service uses, so routing decisions here match what /generate will accept.
_orchestrator = Orchestrator(registry=build_registry())
_rate_limiter = RateLimiter()


class HistoryItem(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    candidate_id: str
    test_type: str
    section: str
    message: str
    mode: Mode = Mode.tutor
    history: list[HistoryItem] = Field(default_factory=list)
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
    orchestrator route → stream the agent's answer through the reply guard
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

    async def response_stream() -> AsyncGenerator[bytes, None]:
        allowed, limit_message = _rate_limiter.check(req.candidate_id)
        if not allowed:
            emit_guardrail_event("candidate", "rate_limit", "fail", trace_id)
            yield f"{limit_message} Please wait a moment and try again.".encode()
            return
        if input_pii:
            yield INPUT_PII_NOTICE.encode()

        safe, reason = await check_content_safety(message, trace_id)
        if not safe:
            yield f"I can't help with that request ({reason}).".encode()
            return

        result = await _orchestrator.process(session, message)
        if isinstance(result, ClarificationTurn):
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
        async for chunk in _stream_from_agent(payload, meta):
            if guard is None:
                guard = ReplyGuard(req.mode, trace_id, meta.get("num_sources"))
            text = guard.feed(decoder.decode(chunk))
            if text:
                yield text.encode("utf-8")
            if guard.stopped:
                return
        if guard is not None:
            tail = guard.feed(decoder.decode(b"", final=True)) + guard.finish()
            if (
                not guard.stopped
                and guard.text.strip()
                and await check_answer_leak(guard.text, req.mode, trace_id)
            ):
                tail += LLM_LEAK_NOTICE
            if tail:
                yield tail.encode("utf-8")

    return StreamingResponse(response_stream(), media_type="text/plain; charset=utf-8")
