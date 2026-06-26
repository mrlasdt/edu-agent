"""
verify-math Skill — MCP client wrapper around the math-verifier service.

The Agent Service calls this skill during every Quant Solve turn.
The skill connects to the math-verifier MCP server and invokes the
verify_math tool. In unit tests, _call_mcp_tool is patched.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from shared.src.shared.config import get_settings


@dataclass
class VerifyMathResult:
    verified: bool
    computed: str
    error: str | None = None


class VerifyMathSkill:
    """
    MCP client skill that calls the math-verifier server's verify_math tool.

    The actual MCP wire protocol is abstracted behind _call_mcp_tool so that
    unit tests can patch it without spinning up a real server.

    In production, _call_mcp_tool calls the MCP server via its SSE endpoint.
    The math-verifier server runs at MATH_VERIFIER_URL (default localhost:8090).
    """

    def __init__(self, server_url: str | None = None) -> None:
        settings = get_settings()
        self._server_url = server_url or settings.math_verifier_url

    async def verify(self, expression: str, expected: str) -> VerifyMathResult:
        """
        Verify a mathematical expression against an expected answer.
        Never raises — returns VerifyMathResult with error set on failure.
        """
        raw = await self._call_mcp_tool(
            tool_name="verify_math",
            arguments={"expression": expression, "expected": expected},
        )
        return VerifyMathResult(
            verified=bool(raw.get("verified", False)),
            computed=str(raw.get("computed", "")),
            error=raw.get("error"),
        )

    async def _call_mcp_tool(
        self, tool_name: str, arguments: dict
    ) -> dict:
        """
        Call a tool on the MCP server via its HTTP/SSE endpoint.
        This thin wrapper is the seam for unit test mocking.

        In production: connects to the math-verifier FastMCP SSE endpoint.
        Protocol: POST to /tools/{tool_name} with JSON body (simplified REST facade
        over the MCP SSE protocol for direct tool invocation).
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self._server_url}/tools/{tool_name}",
                json=arguments,
            )
            response.raise_for_status()
            return response.json()
