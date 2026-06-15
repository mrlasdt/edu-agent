"""
Math Verifier MCP server.

Exposes the verify_math tool via FastMCP (SSE transport).
The Agent Service connects to this as an MCP client.
Claude Desktop can also use it directly via the Candidate MCP server.

Run with:
    python -m math_verifier.server
    # or via Docker Compose (see infra/docker-compose.yml)
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from services.math_verifier.src.math_verifier.sandbox import verify_expression

mcp = FastMCP(
    "math-verifier",
    instructions="sympy-backed GRE Quant answer verification with 2s timeout",
    host="0.0.0.0",
    port=int(os.getenv("MATH_VERIFIER_PORT", "8090")),
)


@mcp.tool()
async def verify_math(expression: str, expected: str) -> dict:
    """
    Verify a mathematical expression or equation against an expected answer.

    Args:
        expression: A mathematical expression or equation.
                    Examples: "x**2 - 5*x + 6 = 0", "2 + 3 * 4"
        expected: The expected answer string.
                  Examples: "x=2 or x=3", "2, 3", "14"

    Returns:
        {"verified": bool, "computed": str, "error": str | null}
    """
    result = await verify_expression(expression, expected)
    return {
        "verified": result.verified,
        "computed": result.computed,
        "error": result.error,
    }


if __name__ == "__main__":
    mcp.run(transport="sse")
