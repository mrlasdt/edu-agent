"""
The Quant agent under the context budget.

Behaviors under test:
  1. a short conversation reaches the model exactly as before
  2. a long one is sent as: the system prompt with the pinned problem, the last messages
     verbatim, and the new message; no extra model call is made
  3. the setting turns it off; solve mode is covered too
  4. tool results never reach the next turn's history (nothing to clear)
"""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.settings import AgentSettings
from services.agent.src.agent.tools.math_tools import MathTools
from shared.src.shared.models import Message, MessageRole, Mode, Session
from tests.agent.agent_events import reply_of

PROBLEM = "Solve x^2 - 5x + 6 = 0 for all real x."


def history(turns: int) -> list[Message]:
    out = [Message(role=MessageRole.candidate, content=f"{PROBLEM} Where do I start?")]
    out.append(Message(role=MessageRole.agent, content="Try factoring. What multiplies to 6?"))
    for i in range(1, turns):
        out.append(Message(role=MessageRole.candidate, content=f"student step {i}"))
        out.append(Message(role=MessageRole.agent, content=f"tutor hint {i}"))
    return out


def session(turns: int, mode: Mode = Mode.tutor) -> Session:
    return Session(
        candidate_id="c1", test_type="gre", section="quant", mode=mode, history=history(turns)
    )


def completion(text: str | None, calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=text, tool_calls=calls)
    usage = SimpleNamespace(prompt_tokens=300, completion_tokens=40)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


async def chunks(text: str) -> Any:
    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class Model:
    """The tutor call (streamed); `extra` are queued replies for non-streamed (tool loop) calls."""

    def __init__(self, reply: str = "What pairs of numbers multiply to 6?",
                 extra: list[Any] | None = None) -> None:
        self.reply = reply
        self.extra = list(extra or [])
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if kwargs.get("stream"):
            return chunks(self.reply)
        return self.extra.pop(0)

    @property
    def tutor_calls(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r.get("stream")]


def agent(**kwargs: Any) -> QuantAgent:
    kwargs.setdefault("context_compaction", True)
    return QuantAgent(
        model_config_path="config/model_config.dev.yaml", verify_math=AsyncMock(), **kwargs
    )


async def run(a: QuantAgent, s: Session, text: str = "ok what next?") -> list[Any]:
    with patch.object(a, "_retrieve", new_callable=AsyncMock, return_value=[]):
        return [e async for e in a.run(s, text, trace_id="t1")]


async def test_a_short_conversation_reaches_the_model_as_it_always_did(turn_log) -> None:
    model = Model()
    with patch("litellm.acompletion", new=model):
        await run(agent(), session(5))
    (request,) = model.requests
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["system"] + ["user", "assistant"] * 5 + ["user"]
    assert "Session so far" not in request["messages"][0]["content"]
    (record,) = turn_log.read()
    assert record["context_messages_folded"] == 0 and record["llm_calls"] == 1


async def test_a_long_conversation_sends_the_pin_and_the_latest_turns(turn_log) -> None:
    model = Model()
    with patch("litellm.acompletion", new=model):
        items = await run(agent(), session(15))  # 30 messages
    assert reply_of(items) == "What pairs of numbers multiply to 6?"
    (tutor,) = model.requests
    system, *rest = tutor["messages"]
    assert "# Session so far" in system["content"] and PROBLEM in system["content"]
    assert "left out to save space" in system["content"]
    assert len(rest) == 10 + 1  # 20 of the 30 messages folded, 10 verbatim, and the new one
    assert rest[0]["content"] == "student step 10" and rest[-1]["content"] == "ok what next?"
    assert "student step 9" not in json.dumps(rest)
    (record,) = turn_log.read()
    assert (record["context_messages_folded"], record["context_messages_sent"]) == (20, 10)
    assert record["llm_calls"] == 1


async def test_prompt_tokens_stop_growing_once_turns_start_folding() -> None:
    sizes = []
    for turns in (2, 5, 6, 10, 15, 25):
        model = Model()
        with patch("litellm.acompletion", new=model):
            await run(agent(), session(turns))
        (tutor,) = model.tutor_calls
        sizes.append(sum(len(m["content"]) for m in tutor["messages"]))
    short, mid, at_fold, ten, fifteen, twenty_five = sizes
    assert short < mid  # growing while nothing is folded
    assert abs(twenty_five - fifteen) < 0.15 * fifteen  # flat once folding is running
    assert twenty_five < 1.6 * at_fold


async def test_the_setting_turns_it_off_and_nothing_is_folded() -> None:
    model = Model()
    with patch("litellm.acompletion", new=model):
        await run(agent(context_compaction=False), session(15))
    assert len(model.tutor_calls[0]["messages"]) == 1 + 30 + 1


def test_the_default_comes_from_the_environment(monkeypatch) -> None:
    assert AgentSettings(_env_file=None).context_compaction_enabled is True  # type: ignore[call-arg]
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "false")
    assert AgentSettings(_env_file=None).context_compaction_enabled is False  # type: ignore[call-arg]


async def test_solve_mode_gets_the_same_budget() -> None:
    model = Model(reply="Step 1. Factor.\n**Answer: 2, 3**")
    a = agent()
    with patch("litellm.acompletion", new=model), patch.object(
        a, "formalize", new_callable=AsyncMock, return_value=None
    ), patch.object(a, "verify_well_formed", new_callable=AsyncMock) as verified:
        verified.return_value = SimpleNamespace(verified=True, computed="[2, 3]", error=None)
        await run(a, session(15, Mode.solve), "Now solve it")
    assert "# Session so far" in model.tutor_calls[0]["messages"][0]["content"]
    assert len(model.tutor_calls[0]["messages"]) == 1 + 10 + 1


async def test_tool_results_do_not_reach_the_next_turns_history() -> None:
    """A turn's tool calls live in that turn's own message list; only the reply is kept."""
    call = SimpleNamespace(
        id="c1", function=SimpleNamespace(
            name="math_evaluate", arguments=json.dumps({"expression": "(1+15/100)*(1-20/100)"}))
    )
    first = Model(extra=[completion(None, [call]), completion("What does a 20% fall do?")])
    a = agent(math_tools_enabled=True, math_tools=MathTools(server_url="http://math.test"))
    with patch("litellm.acompletion", new=first), patch.object(
        MathTools, "call", new=AsyncMock(return_value={"value": "0.92"})
    ):
        items = await run(a, session(1), "A price rises 15% then falls 20%. Net change?")
    reply = reply_of(items)
    assert any(m["role"] == "tool" for m in first.requests[-1]["messages"])  # seen this turn

    next_turn = Session(
        candidate_id="c1", test_type="gre", section="quant",
        history=[
            Message(role=MessageRole.candidate, content="A price rises 15% then falls 20%."),
            Message(role=MessageRole.agent, content=reply),
        ],
    )
    second = Model(extra=[completion("Is it a loss?")])
    with patch("litellm.acompletion", new=second):
        await run(a, next_turn, "is it a loss?")
    sent = second.requests[0]["messages"]
    assert not any(m["role"] == "tool" or m.get("tool_calls") for m in sent)
    assert "0.92" not in json.dumps(sent)
