"""
verify-math Skill - client for the math-verifier service's verify_math tool.

The Agent Service calls this skill during every Quant Solve turn. It posts to the verifier's
plain-HTTP route (`POST /tools/verify_math`), the REST facade that sits beside the MCP
surface; the MCP surface itself is for outside clients such as Claude Desktop. Tests and
evals inject an `httpx.AsyncClient` (see `client`) instead of patching a method.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import httpx

from shared.src.shared.config import get_settings

_log = logging.getLogger(__name__)

# `VerifyMathResult.error` when the verifier service cannot be reached (connection refused,
# timeout, HTTP error status). The turn degrades to "unable to verify" (ADR-0011) instead of
# failing.
VERIFIER_UNREACHABLE = "verifier unreachable"


@dataclass
class VerifyMathResult:
    verified: bool
    computed: str
    error: str | None = None


class VerifyMathSkill:
    """
    Client skill for the math-verifier service's verify_math route.

    The service runs at MATH_VERIFIER_URL (default localhost:8090). By default each call opens
    its own `httpx.AsyncClient`; pass `client` to choose the transport. Tests and the evals hand
    in an `httpx.AsyncClient(transport=httpx.ASGITransport(app=...))` so a call reaches the
    real route in process.
    """

    def __init__(
        self, server_url: str | None = None, *, client: httpx.AsyncClient | None = None
    ) -> None:
        settings = get_settings()
        self._server_url = server_url or settings.math_verifier_url
        self._client = client

    async def verify(self, expression: str, expected: str) -> VerifyMathResult:
        """
        Verify a mathematical expression against an expected answer.
        Never raises — returns VerifyMathResult with error set on failure. A verifier outage
        (any httpx.HTTPError) comes back as verified=False, error="verifier unreachable", which
        the Quant solve loop treats like any failed verification and degrades.
        """
        try:
            raw = await self._post_tool(
                "verify_math", {"expression": expression, "expected": expected}
            )
        except httpx.HTTPError as exc:
            _log.warning("math verifier unreachable: %s: %s", type(exc).__name__, exc)
            return VerifyMathResult(verified=False, computed="", error=VERIFIER_UNREACHABLE)
        return VerifyMathResult(
            verified=bool(raw.get("verified", False)),
            computed=str(raw.get("computed", "")),
            error=raw.get("error"),
        )

    async def _post_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """POST the arguments to `/tools/{tool_name}` and return the JSON body."""
        url = f"{self._server_url}/tools/{tool_name}"
        if self._client is not None:
            response = await self._client.post(url, json=arguments)
        else:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=arguments)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())
