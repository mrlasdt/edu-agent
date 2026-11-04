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
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from services.math_verifier.src.math_verifier.sandbox import run_math_tool, verify_expression
from services.math_verifier.src.math_verifier.tools import TOOLS

mcp = FastMCP(
    "math-verifier",
    instructions=(
        "sympy-backed GRE Quant answer verification and the math_evaluate, math_solve and "
        "math_compare tools the Quant agent calls: allow-listed math notation only, "
        "each check in a child process killed after a 5s timeout"
    ),
    host=os.getenv("MATH_VERIFIER_HOST", "127.0.0.1"),
    port=int(os.getenv("MATH_VERIFIER_PORT", "8090")),
)


@mcp.tool()
async def verify_math(expression: str, expected: str) -> dict[str, Any]:
    """
    Verify a mathematical expression or equation against an expected answer.

    Both arguments are untrusted text (often written by a language model). Only allow-listed
    math notation is accepted; anything else returns error "parse error: not a math expression".

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


# FastMCP.custom_route ships without type annotations
@mcp.custom_route("/tools/verify_math", methods=["POST"])  # type: ignore[untyped-decorator]
async def verify_math_http(request: Request) -> JSONResponse:
    """
    Plain-HTTP facade over the verify_math tool, called by the Agent Service's
    verify-math Skill (POST JSON {expression, expected}). The MCP SSE transport
    has no REST tool route, so this exposes the same sympy-backed logic for
    cluster-internal service-to-service calls. Claude Desktop still uses the
    MCP tool above.
    """
    body = await request.json()
    result = await verify_expression(
        str(body.get("expression", "")), str(body.get("expected", ""))
    )
    return JSONResponse(
        {"verified": result.verified, "computed": result.computed, "error": result.error}
    )


@mcp.custom_route("/tools/{name}", methods=["POST"])  # type: ignore[untyped-decorator]
async def math_tool_http(request: Request) -> JSONResponse:
    """
    Plain-HTTP route for the model-called math tools (math_evaluate, math_solve,
    math_compare), called by the Agent Service's tool loop with the JSON arguments the model
    chose. A bad call is not an HTTP error: the answer is always 200 with `{"error": ...}` in
    the body, because the reader is a model that will retry with a corrected call.
    """
    name = request.path_params["name"]
    if name == "verify_math" or name not in TOOLS:
        return JSONResponse(
            {"error": f"unknown tool `{name}`; the tools are: {', '.join(TOOLS)}."},
            status_code=404,
        )
    try:
        arguments = await request.json()
    except ValueError:
        return JSONResponse({"error": "the request body must be a JSON object."})
    return JSONResponse(await run_math_tool(name, arguments))


if __name__ == "__main__":
    mcp.run(transport="sse")
