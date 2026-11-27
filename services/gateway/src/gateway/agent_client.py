"""The gateway's client for the Agent Service's streaming `/generate`."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx

from services.gateway.src.gateway.settings import get_settings
from shared.src.shared.wire import GenerateRequest

AGENT_TIMEOUT_S = 120.0
# A coached IELTS turn can re-mark an essay: the agent calls the model, waits up to 180s for the
# gateway's marking (the agent's `MARK_TIMEOUT_S`), then calls the model again, and sends nothing
# until all of that is done. The wait for those bytes must outlast the whole turn.
COACH_TURN_READ_TIMEOUT_S = 420.0


class AgentClient:
    """
    Proxies a streaming generation from the Agent Service. /chat gets one through
    `Depends(get_agent_client)`, so a test substitutes a fake with `app.dependency_overrides`
    instead of running an agent service.
    """

    async def stream(
        self, request: GenerateRequest, meta: dict[str, Any] | None = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Yield the agent's reply as it arrives. When the agent reports how many sources it
        retrieved (X-Source-Count), the count is stored in `meta["num_sources"]` for the
        citation check.
        """
        settings = get_settings()
        url = f"{settings.agent_service_url}/generate"
        read_timeout = COACH_TURN_READ_TIMEOUT_S if request.attempt_id else AGENT_TIMEOUT_S
        timeout = httpx.Timeout(AGENT_TIMEOUT_S, read=read_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", url, json=request.model_dump(mode="json")
            ) as resp:
                resp.raise_for_status()
                count = resp.headers.get("x-source-count")
                if meta is not None and count is not None and count.isdigit():
                    meta["num_sources"] = int(count)
                async for chunk in resp.aiter_bytes():
                    yield chunk


def get_agent_client() -> AgentClient:
    return AgentClient()
