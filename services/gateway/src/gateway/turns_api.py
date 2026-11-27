"""
GET /turns/{trace_id}: what one reply cost and did, for the web app's "Show working" panel.

The gateway and the agent each append a record to the shared turn log under the reply's trace id
(`X-Trace-Id` of `/chat`). This joins them (`shared.turn_merge`) and adds what the panel shows
beyond the totals: each model call, the calculator tool calls, and the calculator check verdict.

The gateway writes its record after the reply stream closes, so a request that comes too early
gets 404 and the browser asks again.

Tutor mode: a tool result there would hand over the answer the learner is meant to work toward.
So unless the request carries `?reveal=1` (the learner pressed Reveal), a turn that is not a solve
turn returns the tool calls with their arguments and results blanked, and no calculator verdict.
The check is on the server, not in the browser: the endpoint is no side door to the answer.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from shared.src.shared import turn_meter
from shared.src.shared.turn_merge import merge_turns

router = APIRouter()

SOLVE = "solve"


def _call_view(source: str, call: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source,
        "model": call.get("model"),
        "input_tokens": call.get("input_tokens", 0),
        "output_tokens": call.get("output_tokens", 0),
        "estimated": bool(call.get("estimated")),
        "cost_usd": call.get("cost_usd", 0.0),
        "ms": call.get("ms", 0.0),
    }


def _tool_view(call: dict[str, Any], hidden: bool) -> dict[str, Any]:
    return {
        "name": call.get("name"),
        "ok": bool(call.get("ok")),
        "ms": call.get("ms", 0.0),
        "round": call.get("round", 0),
        "arguments": None if hidden else call.get("arguments"),
        "result": None if hidden else call.get("result"),
    }


@router.get("/turns/{trace_id}")
def get_turn(trace_id: str, reveal: int = 0) -> JSONResponse:
    log = turn_meter.default_turn_log()
    if log is None:
        return JSONResponse(status_code=503, content={"detail": "turn log is switched off"})
    records = [r for r in log.read() if r.get("trace_id") == trace_id]
    gateway = next((r for r in records if r.get("source") == "gateway"), None)
    agent = next((r for r in records if r.get("source") == "agent"), None)
    if gateway is None:
        return JSONResponse(status_code=404, content={"detail": "turn not recorded yet"})

    (turn,) = merge_turns(records)
    hidden = turn["mode"] != SOLVE and not reveal
    tool_calls = (agent or {}).get("tool_calls") or []
    calc_check = None if hidden else (agent or {}).get("calc_check")

    body = {
        "trace_id": trace_id,
        "agent": turn["agent"],
        "mode": turn["mode"],
        "llm_calls": turn["llm_calls"],
        "input_tokens": turn["input_tokens"],
        "output_tokens": turn["output_tokens"],
        "estimated": turn["estimated"],
        "cost_usd": turn["cost_usd"],
        "total_ms": turn["total_ms"],
        "calls": [
            *(_call_view("guardrails", c) for c in gateway.get("calls") or []),
            *(_call_view("agent", c) for c in (agent or {}).get("calls") or []),
        ],
        "math_tools": bool((agent or {}).get("math_tools")),
        "tool_calls": [_tool_view(c, hidden) for c in tool_calls],
        "tools_hidden": hidden and bool(tool_calls),
        "calc_check": calc_check,
        "moderation_skipped": bool(gateway.get("moderation_skipped")),
    }
    return JSONResponse(status_code=200, content=body)
