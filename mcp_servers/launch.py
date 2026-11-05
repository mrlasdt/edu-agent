"""
Launch settings shared by the candidate and admin MCP servers.

Default is stdio, which is what Claude Desktop and other MCP clients expect when they spawn
the server as a subprocess. SSE is an explicit opt-in (MCP_TRANSPORT=sse) and binds
127.0.0.1 unless MCP_HOST says otherwise. Neither server has authentication, so widening
MCP_HOST is the operator's decision.
"""
from __future__ import annotations

import os
from typing import Literal, cast

from mcp.server.fastmcp import FastMCP

Transport = Literal["stdio", "sse"]

_TRANSPORTS = ("stdio", "sse")
DEFAULT_HOST = "127.0.0.1"


def transport() -> Transport:
    value = os.getenv("MCP_TRANSPORT", "stdio").strip().lower() or "stdio"
    if value not in _TRANSPORTS:
        raise SystemExit(f"MCP_TRANSPORT must be one of {', '.join(_TRANSPORTS)}; got {value!r}")
    return cast(Transport, value)


def bind_host() -> str:
    return os.getenv("MCP_HOST", "").strip() or DEFAULT_HOST


def run(server: FastMCP) -> None:
    server.run(transport=transport())
