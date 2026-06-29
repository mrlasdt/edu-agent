from __future__ import annotations

from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.agent.src.agent.bootstrap import build_registry
from services.agent.src.agent.registry import AgentNotRegisteredError
from shared.src.shared.models import Message, MessageRole, Mode, Session

app = FastAPI(title="Agent Service")

# Built once at startup; the registry is the (test_type, section) → agent map.
_registry = build_registry()


class HistoryItem(BaseModel):
    role: MessageRole
    content: str


class GenerateRequest(BaseModel):
    candidate_id: str
    test_type: str
    section: str
    message: str
    mode: Mode = Mode.tutor
    history: list[HistoryItem] = Field(default_factory=list)
    personal_style_enabled: bool = False
    trace_id: str = ""


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Resolve the agent for (test_type, section) and stream its response tokens
    as plain text. The agents yield `str` tokens interleaved with a final `dict`
    of turn metadata; only the text is streamed in v1.
    """
    try:
        agent_cls = _registry.get(req.test_type, req.section)
    except AgentNotRegisteredError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    agent = agent_cls()
    session = Session(
        candidate_id=req.candidate_id,
        test_type=req.test_type,
        section=req.section,
        mode=req.mode,
        history=[Message(role=h.role, content=h.content) for h in req.history],
        personal_style_enabled=req.personal_style_enabled,
    )

    async def token_stream() -> AsyncGenerator[bytes, None]:
        async for item in agent.run(session, req.message):
            if isinstance(item, str):
                yield item.encode("utf-8")
            # dict metadata (verifier_fail, citation_stripped) is not part of the
            # text stream in v1 — it will move to a trailing SSE event later.

    return StreamingResponse(token_stream(), media_type="text/plain; charset=utf-8")
