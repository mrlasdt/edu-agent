"""
math-tools tool client - the client that carries the Quant agent's model-chosen math tool calls
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

from services.agent.src.agent.clients import pool
from services.agent.src.agent.settings import get_settings
from shared.src.shared.http import post_json
from shared.src.shared.math_tool_specs import MATH_TOOL_SPECS, TOOL_NAMES

__all__ = ["MATH_TOOL_SPECS", "TOOL_NAMES", "UNAVAILABLE", "MathTools"]

_log = logging.getLogger(__name__)

UNAVAILABLE = (
    "the math service could not be reached, so nothing was computed. Do not guess a value: "
    "work it out carefully by hand and tell the student that this step was not checked."
)


class MathTools:
    """
    Carries one model-chosen tool call to the math-verifier service and returns its answer.

    `call` never raises: a service that is down becomes an `{"error": ...}` the model can read
    and work around. Pass `client` to choose the HTTP transport: tests and the evals hand in an
    `httpx.AsyncClient(transport=httpx.ASGITransport(app=...))` so a call reaches the real routes
    in process; by default calls share the agent service's pooled client (`agent/clients/pool.py`),
    or open a client of their own when no pool is open.
    """

    def __init__(
        self, server_url: str | None = None, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self._server_url = server_url or get_settings().math_verifier_url
        self._client = client or pool.current()

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self._post_tool(name, arguments)
        except Exception:
            _log.warning("math tool %s failed to reach the math service", name, exc_info=True)
            return {"error": UNAVAILABLE}
        return result if isinstance(result, dict) else {"error": UNAVAILABLE}

    async def _post_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        reply = await post_json(
            f"{self._server_url}/tools/{name}", arguments, client=self._client, timeout=15.0
        )
        return cast(dict[str, Any], reply)
