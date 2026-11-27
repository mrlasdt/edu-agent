"""
Join the gateway's and the agent's per-turn records (`shared.turn_meter`) into one row per turn.

Lives in `shared` because two places need it: the offline report (`evals/reports/turn_report.py`)
and the gateway's `GET /turns/{trace_id}`. The gateway must not import `evals`.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from shared.src.shared.turn_outcome import Outcome


def merge_turns(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per trace id, oldest first. Gateway and agent records are joined."""
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        trace_id = rec.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            by_trace[trace_id].append(rec)
    turns = [_merge(trace_id, recs) for trace_id, recs in by_trace.items()]
    return sorted(turns, key=lambda t: t["ts"])


def _merge(trace_id: str, recs: list[dict[str, Any]]) -> dict[str, Any]:
    gateway = next((r for r in recs if r.get("source") == "gateway"), {})
    agent = next((r for r in recs if r.get("source") == "agent"), {})
    usages = {r.get("usage") for r in recs if r.get("llm_calls")}
    phases = agent.get("phases_ms") or {}
    total_ms = gateway.get("total_ms", agent.get("total_ms"))
    return {
        "trace_id": trace_id,
        "ts": min(r.get("ts", 0.0) for r in recs),
        "agent": agent.get("agent") or gateway.get("agent") or "?",
        "mode": agent.get("mode") or gateway.get("mode") or "?",
        "prompt_version": agent.get("prompt_version") or "",
        "llm_calls": sum(r.get("llm_calls", 0) for r in recs),
        "agent_calls": agent.get("llm_calls", 0),
        "guardrail_calls": gateway.get("llm_calls", 0),
        "input_tokens": sum(r.get("input_tokens", 0) for r in recs),
        "output_tokens": sum(r.get("output_tokens", 0) for r in recs),
        "guardrail_input_tokens": gateway.get("input_tokens", 0),
        "guardrail_output_tokens": gateway.get("output_tokens", 0),
        "cost_usd": sum(r.get("cost_usd", 0.0) for r in recs),
        "guardrail_cost_usd": gateway.get("cost_usd", 0.0),
        "estimated": bool(usages & {"estimate", "mixed"}),
        "ttft_ms": agent.get("ttft_ms"),
        "first_byte_ms": gateway.get("first_byte_ms"),
        "total_ms": total_ms,
        "retrieval_ms": phases.get("retrieval", 0.0),
        "verify_ms": phases.get("verify", 0.0),
        "llm_ms": agent.get("llm_ms", 0.0),
        "guardrail_llm_ms": gateway.get("llm_ms", 0.0),
        "notes_retrieved": agent.get("notes_retrieved"),
        "tool_calls": len(agent.get("tool_calls") or []),
        "tool_errors": agent.get("tool_errors", 0),
        "outcome": gateway.get("outcome", Outcome.no_gateway_record.value),
        "flags": gateway.get("flags") or [],
        "error": agent.get("error") or gateway.get("error"),
    }
