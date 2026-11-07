"""
Math Verifier MCP server.

Exposes the verify_math tool via FastMCP (SSE transport) for MCP clients such as Claude Desktop
(through the Candidate MCP server). The Agent Service does not speak MCP to it: it calls the
plain-HTTP routes below (`POST /tools/{name}`) with an httpx client. The argument contract of
the model-called tools is `shared.math_tool_specs`, the same module the agent reads.

Run with:
    python -m math_verifier.server
    # or via Docker Compose (see infra/docker-compose.yml)
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from services.math_verifier.src.math_verifier.sandbox import run_math_tool, verify_expression
from services.math_verifier.src.math_verifier.settings import get_settings
from services.math_verifier.src.math_verifier.tools import TOOLS

mcp = FastMCP(
    "math-verifier",
    instructions=(
        "sympy-backed GRE Quant answer verification and the math_evaluate, math_solve and "
        "math_compare tools the Quant agent calls: allow-listed math notation only, "
        "each check in a child process killed after a 5s timeout"
    ),
    host=get_settings().math_verifier_host,
    port=get_settings().math_verifier_port,
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
