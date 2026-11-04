"""
A client that disconnects (the Stop button, a closed tab) must stop the work behind it.

Behaviors under test:
  1. The gateway records the turn as `client_cancelled`, and cancels its request to the agent
  2. A turn that finishes keeps its own outcome; an abandoned one is never recorded as `ok`
  3. The agent service stops a call the client left before its first item: the server only
     notices a disconnect once the response starts, which waits for that first item

The ASGI apps are driven directly so the test controls exactly when the client goes away.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from services.agent.src.agent.main import app as agent_app
from services.gateway.src.gateway.main import app as gateway_app

_REAL_ASYNC_CLIENT = httpx.AsyncClient

_BODY = {"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "Solve 2x+3=11"}


async def _drive(app: Any, path: str, body: dict[str, Any], disconnect: asyncio.Event):
    """Run one POST through `app`; the client disconnects when `disconnect` is set."""
    sent: list[dict[str, Any]] = []
    requested = False

    async def receive() -> dict[str, Any]:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
        "scheme": "http",
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    return sent


class _HangingAgentStream(httpx.AsyncByteStream):
    """The agent's response body: one line, then silence, as a model still generating."""

    def __init__(self, on_first_chunk: asyncio.Event) -> None:
        self._on_first_chunk = on_first_chunk
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"First line.\n"
        self._on_first_chunk.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def gateway_model_calls_mocked():
    with (
        patch(
            "services.gateway.src.gateway.main.check_content_safety",
            new_callable=AsyncMock,
            return_value=(True, ""),
        ),
        patch(
            "services.gateway.src.gateway.main.check_completeness",
            new_callable=AsyncMock,
            return_value=(True, ""),
        ),
        patch(
            "services.gateway.src.gateway.guardrails.output._call_leak_checker",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        yield


async def test_disconnect_cancels_the_agent_request_and_records_client_cancelled(turn_log):
    disconnect = asyncio.Event()
    first_chunk = asyncio.Event()
    agent_stream = _HangingAgentStream(first_chunk)

    def agent(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=agent_stream)

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(agent), **kwargs)

    async def stop_after_first_chunk() -> None:
        await first_chunk.wait()
        disconnect.set()

    with patch("services.gateway.src.gateway.main.httpx.AsyncClient", client_factory):
        _, sent = await asyncio.gather(
            stop_after_first_chunk(), _drive(gateway_app, "/chat", _BODY, disconnect)
        )

    assert agent_stream.closed, "the gateway left its agent request open"
    assert [r["outcome"] for r in turn_log.read() if r["source"] == "gateway"] == [
        "client_cancelled"
    ]
    assert not any(m.get("more_body") is False and m["type"] == "http.response.body" for m in sent)


async def test_a_finished_turn_keeps_its_outcome(turn_log):
    async def agent_reply(payload, meta=None):
        yield b"Done.\n"

    with patch("services.gateway.src.gateway.main._stream_from_agent", agent_reply):
        await _drive(gateway_app, "/chat", _BODY, asyncio.Event())

    assert [r["outcome"] for r in turn_log.read()] == ["ok"]


async def test_a_blocked_reply_keeps_its_outcome(turn_log):
    async def agent_reply(payload, meta=None):
        yield b"The answer is x = 4.\n"

    with patch("services.gateway.src.gateway.main._stream_from_agent", agent_reply):
        await _drive(gateway_app, "/chat", _BODY, asyncio.Event())

    assert [r["outcome"] for r in turn_log.read()] == ["answer_leak_blocked"]


async def test_disconnect_before_any_reply_is_recorded_too(turn_log):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def agent_reply(payload, meta=None):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        yield b""

    disconnect = asyncio.Event()

    async def stop_once_waiting() -> None:
        await started.wait()
        disconnect.set()

    with patch("services.gateway.src.gateway.main._stream_from_agent", agent_reply):
        await asyncio.gather(stop_once_waiting(), _drive(gateway_app, "/chat", _BODY, disconnect))

    assert cancelled.is_set()
    assert [r["outcome"] for r in turn_log.read()] == ["client_cancelled"]


async def test_agent_stops_a_call_the_client_left_before_its_first_item():
    waiting = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowAgent:
        async def run(self, session, message, trace_id=""):
            waiting.set()
            try:
                await asyncio.Event().wait()  # retrieval or a model call that has not answered
            finally:
                cancelled.set()
            yield "never"

    class Registry:
        def get(self, test_type, section):
            return SlowAgent

    disconnect = asyncio.Event()

    async def leave_while_waiting() -> None:
        await waiting.wait()
        disconnect.set()

    body = {"candidate_id": "c1", "test_type": "gre", "section": "quant", "message": "x"}
    with patch("services.agent.src.agent.main._registry", Registry()):
        _, sent = await asyncio.gather(
            leave_while_waiting(), _drive(agent_app, "/generate", body, disconnect)
        )

    assert cancelled.is_set(), "the agent kept working for a client that was gone"
    assert sent[0]["status"] == 499
