"""
GET /turns/{trace_id}: the data behind the web app's "Show working" panel.

Behaviors under test:
  1. a solve turn returns totals, each model call, the tool calls with their results, and the
     calculator check
  2. the gateway record is written after the stream closes: before it exists the answer is 404
  3. tutor mode hides tool arguments, results and the verdict until `?reveal=1`
  4. an unknown trace id is 404
"""
from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from services.gateway.src.gateway.main import app
from shared.src.shared.turn_meter import TurnLog

client = TestClient(app)

CALL = {
    "model": "m",
    "input_tokens": 10,
    "output_tokens": 5,
    "estimated": False,
    "cost_usd": 0.001,
    "ms": 120.0,
    "ttft_ms": None,
}
TOOL = {
    "name": "math_solve",
    "arguments": '{"equation":"x^2-3x-4=0"}',
    "ok": True,
    "ms": 12.0,
    "result": '{"roots":["-1","4"]}',
    "round": 1,
}
CHECK = {"verdict": "agrees", "answer": "-8%", "tool_value": "0.92", "tool": "math_evaluate"}


def write_turn(
    log: TurnLog,
    trace_id: str,
    mode: str,
    *,
    gateway: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> None:
    base: dict[str, Any] = {"trace_id": trace_id, "ts": 1.0, "mode": mode, "total_ms": 500.0}
    log.append(
        {
            **base,
            "source": "agent",
            "agent": "quant",
            "llm_calls": 2,
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.002,
            "calls": [CALL, CALL],
            "math_tools": True,
            "tool_calls": tools if tools is not None else [TOOL],
            "calc_check": CHECK,
        }
    )
    if gateway:
        log.append(
            {
                **base,
                "source": "gateway",
                "agent": "quant",
                "llm_calls": 1,
                "input_tokens": 5,
                "output_tokens": 1,
                "cost_usd": 0.0005,
                "calls": [CALL],
                "outcome": "ok",
            }
        )


def test_a_solve_turn_returns_totals_calls_tools_and_the_check(turn_log):
    write_turn(turn_log, "t-solve", "solve")
    body = client.get("/turns/t-solve").json()
    assert body["llm_calls"] == 3
    assert body["input_tokens"] == 25 and body["output_tokens"] == 11
    assert [c["source"] for c in body["calls"]] == ["guardrails", "agent", "agent"]
    assert body["math_tools"] is True
    assert body["tool_calls"][0]["result"] == TOOL["result"]
    assert body["tools_hidden"] is False
    assert body["calc_check"] == CHECK


def test_before_the_gateway_record_exists_the_turn_is_not_ready(turn_log):
    write_turn(turn_log, "t-early", "solve", gateway=False)
    assert client.get("/turns/t-early").status_code == 404


def test_an_unknown_trace_id_is_404(turn_log):
    assert client.get("/turns/nope").status_code == 404


def test_tutor_mode_hides_tool_results_until_reveal(turn_log):
    write_turn(turn_log, "t-tutor", "tutor")
    hidden = client.get("/turns/t-tutor").json()
    assert hidden["tools_hidden"] is True
    assert hidden["tool_calls"][0]["name"] == "math_solve"  # the count and name stay
    first = hidden["tool_calls"][0]
    assert first["result"] is None and first["arguments"] is None
    assert hidden["calc_check"] is None
    assert "-1" not in client.get("/turns/t-tutor").text

    shown = client.get("/turns/t-tutor?reveal=1").json()
    assert shown["tools_hidden"] is False
    assert shown["tool_calls"][0]["result"] == TOOL["result"]


def test_a_notes_search_is_not_a_calculator_call(turn_log):
    notes = {**TOOL, "name": "notes_search", "arguments": '{"concept":"quadratics"}'}
    write_turn(turn_log, "t-notes", "tutor", tools=[notes])

    body = client.get("/turns/t-notes").json()
    assert body["tool_calls"] == []
    assert body["tools_hidden"] is False
    assert "notes_search" not in client.get("/turns/t-notes?reveal=1").text


def test_an_unknown_mode_is_treated_like_tutor(turn_log):
    write_turn(turn_log, "t-odd", "?")
    assert client.get("/turns/t-odd").json()["tools_hidden"] is True
