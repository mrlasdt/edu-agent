from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.agent.src.agent.bootstrap import build_registry
from services.agent.src.agent.registry import AgentNotRegisteredError
from shared.src.shared.models import Message, MessageRole, Mode, Session
from shared.src.shared.observability import setup_observability
from shared.src.shared.trailer import encode, strip_mark

app = FastAPI(title="Agent Service")

# The agent's own INFO lines (one per model-called math tool) reach stderr; every other logger
# keeps the default WARNING level, so libraries stay quiet. Uvicorn configures only its own
# loggers, so without this the tool-call lines would never be printed.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("services.agent").setLevel(logging.INFO)

# Every LLM call in the agent service (agents, critics, question pipeline) goes
# through litellm, so register the Langfuse callback once at import. No-op
# without Langfuse keys.
setup_observability()

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
async def health() -> dict[str, str]:
    return {"status": "ok"}


_DISCONNECT_POLL_SECONDS = 0.25


async def _first_item(
    items: AsyncIterator[str | dict[str, Any]], request: Request
) -> tuple[str | dict[str, Any] | None, bool]:
    """
    The first item of the agent's output, and whether the client is still connected. The server
    only watches for a disconnect once the response has started, and the response cannot start
    before this item exists (it decides the headers). Without this watch, a Stop pressed while
    the agent is retrieving or waiting for its first token would run the whole call to its end.
    """
    first = asyncio.ensure_future(anext(items, None))
    try:
        while True:
            done, _ = await asyncio.wait({first}, timeout=_DISCONNECT_POLL_SECONDS)
            if done:
                return first.result(), True
            if await request.is_disconnected():
                return None, False
    finally:
        if not first.done():
            first.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await first


@app.post("/generate", response_model=None)
async def generate(req: GenerateRequest, request: Request) -> StreamingResponse | Response:
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

    items = agent.run(session, req.message, trace_id=req.trace_id)
    # Agents that retrieve sources report the count first; surface it as a header so the
    # gateway can check citations against it.
    headers: dict[str, str] = {}
    first, connected = await _first_item(items, request)
    if not connected:
        return Response(status_code=499)  # nobody is listening; the agent call is cancelled
    if isinstance(first, dict) and "sources" in first:
        headers["X-Source-Count"] = str(first["sources"])
        first = None

    async def token_stream() -> AsyncGenerator[bytes, None]:
        if isinstance(first, str):
            yield strip_mark(first).encode("utf-8")
        async for item in items:
            if isinstance(item, str):
                yield strip_mark(item).encode("utf-8")
            elif isinstance(item, dict) and isinstance(item.get("verifier_fail"), bool):
                # The verifier's verdict is only known with the finished answer: it ends the
                # stream as a trailer the gateway passes on. Other metadata is not streamed.
                yield encode({"math_verified": not item["verifier_fail"]}).encode("utf-8")

    return StreamingResponse(
        token_stream(), media_type="text/plain; charset=utf-8", headers=headers
    )
