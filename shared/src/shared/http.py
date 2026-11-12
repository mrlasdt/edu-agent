"""
The one way a service POSTs JSON to another service and reads the JSON back.

A caller that owns a long-lived `httpx.AsyncClient` (the agent service opens one for its whole
life, see `agent/clients/pool.py`) passes it in, so calls reuse connections. A caller without one
(tests, one-off scripts, an MCP server) gets a client that lives for just that call.
"""
from __future__ import annotations

from typing import Any

import httpx


async def post_json(
    url: str,
    body: dict[str, Any],
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 10.0,
) -> Any:
    """POST `body` as JSON and return the decoded reply; raises `httpx.HTTPError` on failure."""
    if client is not None:
        response = await client.post(url, json=body, timeout=timeout)
    else:
        async with httpx.AsyncClient(timeout=timeout) as one_shot:
            response = await one_shot.post(url, json=body)
    response.raise_for_status()
    return response.json()
