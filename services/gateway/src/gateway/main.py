from __future__ import annotations

import uuid
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.agent.src.agent.bootstrap import build_registry
from services.gateway.src.gateway.guardrails.input import check_content_safety
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
async def health():
    return {"status": "ok"}


async def _stream_from_agent(payload: dict) -> AsyncGenerator[bytes, None]:
    """
    Proxy a streaming generation from the Agent Service. Isolated as a seam so
    tests can substitute a fake stream without a running agent service.
    """
    settings = get_settings()
    url = f"{settings.agent_service_url}/generate"
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Single turn: content-safety guardrail → orchestrator route → stream the
    agent's answer. Clarifications and refusals are streamed as plain text so
    the response shape is uniform regardless of outcome.
    """
    trace_id = str(uuid.uuid4())
    session = Session(
        candidate_id=req.candidate_id,
        test_type=req.test_type,
        section=req.section,
        mode=req.mode,
        history=[Message(role=h.role, content=h.content) for h in req.history],
        personal_style_enabled=req.personal_style_enabled,
    )

    async def response_stream() -> AsyncGenerator[bytes, None]:
        safe, reason = await check_content_safety(req.message, trace_id)
        if not safe:
            yield f"I can't help with that request ({reason}).".encode("utf-8")
            return

        result = await _orchestrator.process(session, req.message)
        if isinstance(result, ClarificationTurn):
            yield result.prompt.encode("utf-8")
            return

        assert isinstance(result, RouteResult)  # narrows type; route succeeded
        payload = {
            "candidate_id": req.candidate_id,
            "test_type": req.test_type,
            "section": req.section,
            "message": req.message,
            "mode": req.mode.value,
            "history": [h.model_dump(mode="json") for h in req.history],
            "personal_style_enabled": req.personal_style_enabled,
            "trace_id": trace_id,
        }
        async for chunk in _stream_from_agent(payload):
            yield chunk

    return StreamingResponse(response_stream(), media_type="text/plain; charset=utf-8")
