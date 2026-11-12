from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Never, NoReturn

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from services.agent.src.agent.bootstrap import build_registry
from services.agent.src.agent.registry import AgentNotRegisteredError
from shared.src.shared.events import (
    CitationsStripped,
    Event,
    IndependentCheck,
    SampleEssay,
    Sources,
    Token,
    ToolTrace,
    Verified,
)
from shared.src.shared.models import Message, Session
from shared.src.shared.observability import setup_observability
from shared.src.shared.trailer import encode, strip_mark
from shared.src.shared.wire import GenerateRequest

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


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


_DISCONNECT_POLL_SECONDS = 0.25


async def _first_event(
    items: AsyncIterator[Event], request: Request
) -> tuple[Event | None, bool]:
    """
    The first event of the agent's output, and whether the client is still connected. The server
    only watches for a disconnect once the response has started, and the response cannot start
    before this event exists (it decides the headers). Without this watch, a Stop pressed while
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


def _reject(item: Never) -> NoReturn:
    """An agent yielded something that is not an `Event`. mypy cannot get here: `Never` means every
    event type was handled by the caller's `match`."""
    raise TypeError(f"an agent yielded {type(item).__name__}, which is not an Event")


class _Report:
    """What the agent says about the answer it is writing, which ends the stream as the trailer."""

    def __init__(self) -> None:
        self._verified: bool | None = None
        self._independent: IndependentCheck | None = None
        self._essay = False

    def take(self, event: Event) -> bytes:
        """Fold one event in; returns the reply text it carries (none for metadata)."""
        match event:
            case Token(text=text):
                return strip_mark(text).encode("utf-8")
            case IndependentCheck():
                self._independent = event
            case Verified(passed=passed):
                self._verified = passed
            case SampleEssay():
                self._essay = True
            case Sources() | CitationsStripped() | ToolTrace():
                pass  # not streamed: the count was the header, the others are for evals and logs
            case _:
                _reject(event)
        return b""

    def trailer(self) -> bytes:
        """The trailer the gateway passes on: the math verifier's verdict (with the independent
        check's, when the agent ran one) and the sample-essay flag the web app shows its notice
        for. Empty when there is nothing to report."""
        meta: dict[str, object] = {}
        if self._verified is not None:
            meta["math_verified"] = self._verified
            if self._independent is not None:
                meta["independent_check"] = self._independent.public()
        if self._essay:
            meta["disclaimer"] = "essay"
        return encode(meta).encode("utf-8") if meta else b""


@app.post("/generate", response_model=None)
async def generate(req: GenerateRequest, request: Request) -> StreamingResponse | Response:
    """
    Resolve the agent for (test_type, section) and stream its reply as plain text. The agents
    yield `Event`s: the `Token`s are the text, `Sources` becomes a header, and the verdicts end
    the stream as a trailer.
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
        report_digest=req.report_digest,
    )

    items = agent.run(session, req.message, trace_id=req.trace_id)
    # Agents that retrieve sources report the count first; surface it as a header so the
    # gateway can check citations against it.
    headers: dict[str, str] = {}
    first, connected = await _first_event(items, request)
    if not connected:
        return Response(status_code=499)  # nobody is listening; the agent call is cancelled
    report = _Report()
    match first:
        case Sources(count=count):
            headers["X-Source-Count"] = str(count)
            head = b""
        case None:
            head = b""
        case _:
            head = report.take(first)  # before the response starts, so a bad event is a 500

    async def token_stream() -> AsyncGenerator[bytes, None]:
        if head:
            yield head
        async for item in items:
            if text := report.take(item):
                yield text
        if trailer := report.trailer():
            yield trailer

    return StreamingResponse(
        token_stream(), media_type="text/plain; charset=utf-8", headers=headers
    )
