"""
math-tools Skill - the client that carries the Quant agent's model-chosen math tool calls
(math_evaluate, math_solve, math_compare) to the math-verifier service.

The contract the model reads (name, description, argument schema) is not written here: it is
`shared.math_tool_specs`, the one module the math verifier also builds its argument checks
from, so the two cannot drift. The sympy code that answers a call lives in
`math_verifier.tools`. Nothing the model sends is trusted here either: the service parses it
through the allow-list parser, in a child process it can kill.
"""
from __future__ import annotations

import logging
from typing import Any, cast

import httpx

from shared.src.shared.config import get_settings
from shared.src.shared.math_tool_specs import MATH_TOOL_SPECS, TOOL_NAMES

__all__ = ["MATH_TOOL_SPECS", "TOOL_NAMES", "UNAVAILABLE", "MathToolsSkill"]

_log = logging.getLogger(__name__)

UNAVAILABLE = (
    "the math service could not be reached, so nothing was computed. Do not guess a value: "
    "work it out carefully by hand and tell the student that this step was not checked."
)


class MathToolsSkill:
    """
    Carries one model-chosen tool call to the math-verifier service and returns its answer.

    `call` never raises: a service that is down becomes an `{"error": ...}` the model can read
    and work around. Pass `client` to choose the HTTP transport: tests and the evals hand in an
    `httpx.AsyncClient(transport=httpx.ASGITransport(app=...))` so a call reaches the real routes
    in process; by default each call opens its own client against `server_url`.
    """

    def __init__(
        self, server_url: str | None = None, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._server_url = server_url or get_settings().math_verifier_url
        self._client = client

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self._post_tool(name, arguments)
        except Exception:
            _log.warning("math tool %s failed to reach the math service", name, exc_info=True)
            return {"error": UNAVAILABLE}
        return result if isinstance(result, dict) else {"error": UNAVAILABLE}

    async def _post_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._server_url}/tools/{name}"
        if self._client is not None:
            response = await self._client.post(url, json=arguments)
        else:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(url, json=arguments)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())
