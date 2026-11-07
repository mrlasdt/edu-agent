"""Fakes for the gateway's /chat collaborators, swapped in with `app.dependency_overrides`."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any

from services.gateway.src.gateway.agent_client import get_agent_client
from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter
from services.gateway.src.gateway.main import app, get_orchestrator, get_rate_limiter
from shared.src.shared.wire import GenerateRequest

AgentStream = Callable[..., AsyncIterator[bytes]]


class FakeAgentClient:
    """Stands in for the Agent Service: `stream` is `fn(request, meta=None)`, an async generator."""

    def __init__(self, fn: AgentStream) -> None:
        self._fn = fn

    def stream(self, request: GenerateRequest, meta: dict[str, Any] | None = None):
        return self._fn(request, meta)


@contextmanager
def override(dependency: Callable[..., Any], value: Any) -> Iterator[None]:
    """Make /chat's `Depends(dependency)` return `value` for the block, then restore."""
    previous = app.dependency_overrides.get(dependency)
    app.dependency_overrides[dependency] = lambda: value
    try:
        yield
    finally:
        if previous is None:
            app.dependency_overrides.pop(dependency, None)
        else:
            app.dependency_overrides[dependency] = previous


def agent_replies(fn: AgentStream):
    """The agent answers with the bytes `fn(request, meta)` yields."""
    return override(get_agent_client, FakeAgentClient(fn))


def rate_limiter_of(limiter: RateLimiter):
    return override(get_rate_limiter, limiter)


def orchestrator_of(orchestrator: Any):
    return override(get_orchestrator, orchestrator)
