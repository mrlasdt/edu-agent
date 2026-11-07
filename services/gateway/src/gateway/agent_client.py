"""The gateway's client for the Agent Service's streaming `/generate`."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx

from shared.src.shared.config import get_settings
from shared.src.shared.wire import GenerateRequest


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
        async with httpx.AsyncClient(timeout=120.0) as client:
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
