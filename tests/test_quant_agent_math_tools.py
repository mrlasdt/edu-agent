"""
The Quant agent with the math tools on: the model computes with sympy instead of in its head.

Behaviors under test:
  1. a tutor turn calls a tool, but the learner's reply holds only the model's final text:
     the tool result stays private to the model
  2. a solve turn with tools is still verified, and reports the tool calls for the evals
  3. the turn record names the calls, rounds and errors (the operator's log), and the prompt
     that ran
  4. a math service that is down costs a tool result, not the turn
  5. with the flag off the agent is the original single streamed call ("the before")
  6. the flag defaults to on in the product (settings), and the tools prompts exist
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.skills.math_tools import (
    MATH_TOOL_SPECS,
    TOOL_NAMES,
    UNAVAILABLE,
    MathToolsSkill,
)
from services.agent.src.agent.skills.verify_math import VerifyMathResult
from shared.src.shared.config import Settings, load_model_config
from shared.src.shared.models import Mode, Session

PERCENT_RESULT = {"value": "0.92", "exact": "23/25"}


def session(mode: Mode = Mode.tutor) -> Session:
    return Session(candidate_id="c1", test_type="gre", section="quant", mode=mode)


def tool_reply(call_id: str = "c1", name: str = "math_evaluate", **arguments: Any) -> Any:
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name, arguments=json.dumps(arguments or {"expression": "(1+15/100)*(1-20/100)"})
        ),
    )
    return text_reply(None, [call])


def text_reply(text: str | None, calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=text, tool_calls=calls)
    usage = SimpleNamespace(prompt_tokens=300, completion_tokens=20)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class Model:
    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


class FakeMath(MathToolsSkill):
    def __init__(self, result: dict[str, Any] | Exception) -> None:
        super().__init__(server_url="http://math.test")
        self.result = result
        self.posts: list[tuple[str, dict[str, Any]]] = []

    async def _post(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.posts.append((name, arguments))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def agent(math: MathToolsSkill, enabled: bool = True) -> QuantAgent:
    return QuantAgent(
        model_config_path="config/model_config.dev.yaml",
        math_tools_skill=math,
        math_tools_enabled=enabled,
        verify_math_skill=AsyncMock(),
    )


async def collect(a: QuantAgent, mode: Mode, text: str, trace_id: str = "t1") -> list[Any]:
    with patch.object(a, "_retrieve", new_callable=AsyncMock, return_value=[]):
        return [item async for item in a.run(session(mode), text, trace_id=trace_id)]


# ── 1. the tutor keeps tool results private ─────────────────────────────────


async def test_tutor_reply_holds_only_the_final_text_never_the_tool_result(turn_log) -> None:
    math = FakeMath(PERCENT_RESULT)
    model = Model(tool_reply(), text_reply("What does a 20% fall do to the new price?"))
    a = agent(math)
    with patch("litellm.acompletion", new=model):
        items = await collect(a, Mode.tutor, "A price rises 15% then falls 20%. Net change?")

    text = "".join(i for i in items if isinstance(i, str))
    assert text == "What does a 20% fall do to the new price?"
    assert "0.92" not in text and "23/25" not in text
    # the model did see the result, in the tool message of its second request
    seen = [m for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert PERCENT_RESULT["value"] in seen[0]["content"]
    assert math.posts == [("math_evaluate", {"expression": "(1+15/100)*(1-20/100)"})]


async def test_the_tutor_prompt_with_tools_keeps_the_never_state_the_answer_rule() -> None:
    model = Model(text_reply("Try setting up the ratio."))
    a = agent(FakeMath(PERCENT_RESULT))
    with patch("litellm.acompletion", new=model):
        await collect(a, Mode.tutor, "Solve x^2-5x+6=0")

    system = model.requests[0]["messages"][0]["content"]
    assert "Never state the final numeric or symbolic answer" in system
    assert "The student cannot see tool calls or results" in system
    assert model.requests[0]["tools"] == MATH_TOOL_SPECS
    assert model.requests[0]["stream"] is False


# ── 2. solve mode ────────────────────────────────────────────────────────────


async def test_solve_with_tools_is_verified_and_reports_the_calls() -> None:
    final = "1. Net factor 1.15 * 0.8 = 0.92 (math_evaluate).\n**Answer: -8**"
    model = Model(tool_reply(), text_reply(final))
    a = agent(FakeMath(PERCENT_RESULT))
    with (
        patch("litellm.acompletion", new=model),
        patch.object(
            a, "_verify", new_callable=AsyncMock,
            return_value=VerifyMathResult(verified=True, computed="-8"),
        ),
    ):
        items = await collect(a, Mode.solve, "A price rises 15% then falls 20%.")

    assert items[0] == final
    assert {"verifier_fail": False} in items
    trace = items[-1]["tool_trace"]
    assert [(c["name"], c["ok"]) for c in trace] == [("math_evaluate", True)]
    assert "0.92" in trace[0]["result"]
    assert "use a tool for every calculation" in model.requests[0]["messages"][0]["content"]


async def test_a_turn_without_tool_calls_yields_no_trace() -> None:
    a = agent(FakeMath(PERCENT_RESULT))
    with (
        patch("litellm.acompletion", new=Model(text_reply("Easy.\n**Answer: 4**"))),
        patch.object(
            a, "_verify", new_callable=AsyncMock,
            return_value=VerifyMathResult(verified=True, computed="4"),
        ),
    ):
        items = await collect(a, Mode.solve, "2+2")
    assert not any(isinstance(i, dict) and "tool_trace" in i for i in items)


# ── 3. the turn record ───────────────────────────────────────────────────────


async def test_the_turn_record_names_the_calls_and_the_prompt(turn_log) -> None:
    model = Model(
        tool_reply("c1"),
        tool_reply("c2", expression="x^"),
        text_reply("Next, what is the second root?"),
    )
    math = FakeMath(PERCENT_RESULT)
    a = agent(math)
    # second call fails inside the service: the result carries an error
    async def post(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"error": "bad"} if arguments["expression"] == "x^" else PERCENT_RESULT

    math._post = post  # type: ignore[method-assign]
    with patch("litellm.acompletion", new=model):
        await collect(a, Mode.tutor, "Solve it", trace_id="trace-tools")

    (record,) = turn_log.read()
    assert record["llm_calls"] == 3 and record["math_tools"] is True
    assert record["tool_rounds"] == 2 and record["tool_errors"] == 1
    assert [(c["name"], c["ok"], c["round"]) for c in record["tool_calls"]] == [
        ("math_evaluate", True, 1),
        ("math_evaluate", False, 2),
    ]
    cfg = load_model_config("config/model_config.dev.yaml")
    version = cfg.prompt_version("quant_agent_math_tools", "tutor")
    assert record["prompt_version"] == f"math_tools-{version}"


# ── 4. the math service is down ──────────────────────────────────────────────


async def test_a_math_service_that_is_down_gives_the_model_an_error_and_the_turn_goes_on() -> None:
    math = FakeMath(ConnectionError("refused"))
    model = Model(tool_reply(), text_reply("Let's find the new price after a 15% rise first."))
    with patch("litellm.acompletion", new=model):
        items = await collect(agent(math), Mode.tutor, "A price rises 15%...")

    assert "".join(i for i in items if isinstance(i, str)).startswith("Let's find")
    tool_message = [m for m in model.requests[1]["messages"] if m["role"] == "tool"][0]
    assert UNAVAILABLE in tool_message["content"]
    assert "Do not guess" in UNAVAILABLE


async def test_a_blank_final_reply_becomes_a_polite_fallback_not_silence() -> None:
    with patch("litellm.acompletion", new=Model(text_reply(""))):
        items = await collect(agent(FakeMath(PERCENT_RESULT)), Mode.tutor, "hi")
    assert "".join(i for i in items if isinstance(i, str)).startswith("I could not finish")


async def test_solve_retries_keep_calling_tools_and_keep_the_calls_of_every_attempt() -> None:
    model = Model(
        tool_reply("c1"), text_reply("**Answer: 1**"),   # attempt 1
        tool_reply("c2"), text_reply("**Answer: 2**"),   # attempt 2
        text_reply("**Answer: 3**"),                      # escalation, no tool
    )
    a = agent(FakeMath(PERCENT_RESULT))
    with (
        patch("litellm.acompletion", new=model),
        patch.object(
            a, "_verify", new_callable=AsyncMock,
            return_value=VerifyMathResult(verified=False, computed=""),
        ),
    ):
        items = await collect(a, Mode.solve, "q")

    assert {"verifier_fail": True} in items
    assert [c["round"] for c in items[-1]["tool_trace"]] == [1, 2]


# ── 5. the flag off is the original path ─────────────────────────────────────


async def test_with_the_flag_off_the_agent_streams_one_call_and_offers_no_tools() -> None:
    async def stream() -> Any:
        for token in ["Think ", "about it."]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=token))])

    seen: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return stream()

    a = agent(FakeMath(PERCENT_RESULT), enabled=False)
    with patch("litellm.acompletion", new=fake):
        items = await collect(a, Mode.tutor, "q")

    assert items == ["Think ", "about it."]
    assert "tools" not in seen and seen["stream"] is True
    assert "Math tools" not in seen["messages"][0]["content"]


# ── 6. defaults and files ────────────────────────────────────────────────────


def test_the_product_default_is_tools_on() -> None:
    assert Settings(_env_file=None).math_tools_enabled is True  # type: ignore[call-arg]


def test_the_tools_prompts_resolve_and_describe_the_tools() -> None:
    from services.agent.src.agent.agents import quant

    cfg = load_model_config("config/model_config.dev.yaml")
    for mode in (Mode.tutor, Mode.solve):
        version = cfg.prompt_version("quant_agent_math_tools", mode.value)
        text = quant._load_prompt(mode, version, "quant_agent_math_tools")
        assert "# Math tools" in text
        assert all(name in text for name in TOOL_NAMES)


def test_every_tool_has_a_schema_the_service_accepts() -> None:
    from services.math_verifier.src.math_verifier.tools import OPTIONAL, REQUIRED

    for spec in MATH_TOOL_SPECS:
        fn = spec["function"]
        props = set(fn["parameters"]["properties"])
        assert set(fn["parameters"]["required"]) == set(REQUIRED[fn["name"]])
        assert props == set(REQUIRED[fn["name"]]) | set(OPTIONAL.get(fn["name"], ()))
        assert len(fn["description"]) > 200  # written to brief a colleague, not to label a field


async def test_the_skill_turns_a_failed_post_into_an_error_result() -> None:
    skill = FakeMath(ConnectionError("down"))
    assert await skill.call("math_evaluate", {"expression": "1"}) == {"error": UNAVAILABLE}
    not_a_dict = FakeMath({})
    not_a_dict.result = ["x"]  # type: ignore[assignment]
    assert await not_a_dict.call("math_evaluate", {"expression": "1"}) == {"error": UNAVAILABLE}


def test_the_agent_service_prints_its_tool_call_lines() -> None:
    import logging

    from services.agent.src.agent import main  # noqa: F401  (import configures logging)

    assert logging.getLogger("services.agent").getEffectiveLevel() == logging.INFO


def test_the_model_config_comes_from_model_config_path_when_no_path_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shared.src.shared.config import get_settings, load_model_config

    routed = tmp_path / "routed.yaml"
    dev = Path("config/model_config.dev.yaml").read_text()
    routed.write_text(dev.replace("gpt-4o-mini", "x-y"))
    monkeypatch.setattr(get_settings(), "model_config_path", str(routed))
    assert load_model_config().role("orchestrator").model == "x-y"
