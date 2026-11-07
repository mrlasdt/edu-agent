"""
The bounded tool loop: ask, run the tools the model asks for, feed results back, stop.

The model is scripted (a function from the messages it was sent to the reply it gives), so
every test states what the model "decided" and checks what the loop did about it.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from services.agent.src.agent.tool_loop import (
    MAX_CALLS_PER_ROUND,
    MAX_TOOL_ROUNDS,
    OUT_OF_BUDGET_NOTE,
    run_tool_loop,
)
from shared.src.shared.turn_meter import TurnMeter

TOOLS = [{"type": "function", "function": {"name": "math_evaluate"}}]


def call(call_id: str, name: str = "math_evaluate", arguments: Any = None) -> Any:
    given = arguments or {"expression": "1+1"}
    raw = arguments if isinstance(arguments, str) else json.dumps(given)
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=raw))


def reply(text: str | None = None, calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=text, tool_calls=calls or None)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class ScriptedModel:
    """Replies in order; remembers the messages and options of each request."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0) if self.replies else reply("done")


async def run(model: ScriptedModel, execute: Any, **kwargs: Any) -> Any:
    meter = TurnMeter.detached()
    with patch("litellm.acompletion", new=model):
        outcome = await run_tool_loop(
            meter, model="m", messages=[{"role": "user", "content": "q"}], tools=TOOLS,
            execute=execute, **kwargs,
        )
    return outcome, meter


async def echo(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"value": "2"}


async def test_a_model_that_needs_no_tool_costs_one_call() -> None:
    model = ScriptedModel(reply("Just 4."))
    outcome, meter = await run(model, echo)

    assert (outcome.text, outcome.rounds, outcome.calls) == ("Just 4.", 0, [])
    assert len(model.requests) == 1 and model.requests[0]["tools"] == TOOLS
    assert meter.snapshot()["llm_calls"] == 1


async def test_a_tool_call_is_run_and_its_result_goes_back_to_the_model() -> None:
    model = ScriptedModel(reply(calls=[call("c1")]), reply("The value is 2."))
    outcome, meter = await run(model, echo)

    assert outcome.text == "The value is 2."
    assert (outcome.rounds, len(outcome.calls), outcome.calls[0].ok) == (1, 1, True)
    second = model.requests[1]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["tool_calls"][0]["id"] == "c1"
    assert second[2] == {
        "role": "tool", "tool_call_id": "c1", "name": "math_evaluate", "content": '{"value":"2"}',
    }
    assert meter.snapshot()["llm_calls"] == 2
    assert "tools" in meter.snapshot()["phases_ms"]


async def test_the_calls_of_one_round_run_together_and_each_gets_a_result() -> None:
    running = 0
    peak = 0

    async def slow(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return {"value": arguments["expression"]}

    calls = [call(f"c{i}", arguments={"expression": str(i)}) for i in range(3)]
    model = ScriptedModel(reply(calls=calls), reply("ok"))
    outcome, _ = await run(model, slow)

    assert peak == 3
    results = [m for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["c0", "c1", "c2"]
    assert outcome.rounds == 1 and len(outcome.calls) == 3


async def test_calls_beyond_the_per_round_limit_get_an_error_not_silence() -> None:
    calls = [call(f"c{i}") for i in range(MAX_CALLS_PER_ROUND + 2)]
    ran: list[str] = []

    async def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ran.append(name)
        return {"value": "1"}

    model = ScriptedModel(reply(calls=calls), reply("ok"))
    outcome, _ = await run(model, execute)

    assert len(ran) == MAX_CALLS_PER_ROUND
    results = [m for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert len(results) == MAX_CALLS_PER_ROUND + 2  # the provider rejects an unanswered call
    assert "too many tool calls" in results[-1]["content"]
    assert [c.ok for c in outcome.calls].count(False) == 2


@pytest.mark.parametrize("raw", ["", "not json", "[1, 2]", "null"])
async def test_unreadable_arguments_become_an_error_the_model_can_fix(raw: str) -> None:
    ran = False

    async def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal ran
        ran = True
        return {}

    model = ScriptedModel(reply(calls=[call("c1", arguments=raw)]), reply("ok"))
    outcome, _ = await run(model, execute)

    assert not ran and not outcome.calls[0].ok
    tool_message = model.requests[1]["messages"][-1]
    assert "JSON object" in tool_message["content"]


async def test_a_tool_that_raises_does_not_kill_the_turn() -> None:
    async def boom(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("secret internals")

    model = ScriptedModel(reply(calls=[call("c1")]), reply("ok"))
    outcome, _ = await run(model, boom)

    assert outcome.text == "ok" and not outcome.calls[0].ok
    content = model.requests[1]["messages"][-1]["content"]
    assert "RuntimeError" in content and "secret internals" not in content


async def test_a_model_that_keeps_asking_is_stopped_after_four_rounds_and_must_answer() -> None:
    model = ScriptedModel(*[reply(calls=[call(f"c{i}")]) for i in range(10)])
    # the 5th request is the forced answer: scripted to return text
    model.replies[MAX_TOOL_ROUNDS] = reply("Best I have: about 2 (unchecked).")
    outcome, meter = await run(model, echo)

    assert outcome.rounds == MAX_TOOL_ROUNDS == 4
    assert outcome.out_of_budget is True
    assert outcome.text == "Best I have: about 2 (unchecked)."
    assert len(model.requests) == MAX_TOOL_ROUNDS + 1
    final = model.requests[-1]
    assert final["tool_choice"] == "none"
    assert final["messages"][-1] == {"role": "user", "content": OUT_OF_BUDGET_NOTE}
    assert "tool_choice" not in model.requests[0]
    assert meter.snapshot()["llm_calls"] == 5


async def test_a_spent_time_budget_goes_straight_to_the_forced_answer() -> None:
    model = ScriptedModel(reply("Answer from what I know."))
    outcome, _ = await run(model, echo, deadline_s=0.0)

    assert outcome.out_of_budget and outcome.rounds == 0
    assert model.requests[0]["tool_choice"] == "none"


async def test_a_blank_final_answer_is_returned_as_blank_for_the_caller_to_handle() -> None:
    model = ScriptedModel(reply(None))
    outcome, _ = await run(model, echo)
    assert outcome.text == ""


async def test_model_options_reach_every_request() -> None:
    model = ScriptedModel(reply(calls=[call("c1")]), reply("ok"))
    await run(model, echo, temperature=0.2, metadata={"trace_id": "t"})
    for request in model.requests:
        # the Router adds its own routing keys to `metadata`; ours must still be in it
        assert request["temperature"] == 0.2 and request["metadata"]["trace_id"] == "t"
