"""
The Quant agent under the context budget.

Tests that look at the note wait for it (`context_background=False`, what the eval does); the
background mode, the product's, has its own tests at the end.

Behaviors under test:
  1. a short conversation reaches the model exactly as before and makes no summary call
  2. a long one is sent as: the system prompt with the pinned problem and a note, the last
     messages verbatim, and the new message; the note is one extra metered call
  3. the same turns are summarised once across turns and across agents (one agent per request)
  4. a summary call that fails still leaves a turn that answers, with the problem pinned
  5. the setting turns it off; solve mode is covered too
  6. tool results never reach the next turn's history (nothing to clear)
"""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.settings import AgentSettings
from services.agent.src.agent.tools.math_tools import MathTools
from shared.src.shared.config import load_model_config
from shared.src.shared.models import Message, MessageRole, Mode, Session
from tests.agent_events import reply_of

PROBLEM = "Solve x^2 - 5x + 6 = 0 for all real x."
NOTE = (
    "Problem: x^2 - 5x + 6 = 0\nHints already given: factor into two binomials\n"
    "Mistakes and confusions: none\nWhere the student is now: listing factor pairs of 6"
)


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
    """The summary call (not streamed) and the tutor call (streamed), told apart by `stream`."""

    def __init__(self, reply: str = "What pairs of numbers multiply to 6?",
                 note: str | Exception = NOTE, extra: list[Any] | None = None) -> None:
        self.reply, self.note = reply, note
        self.extra = list(extra or [])
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if kwargs.get("stream"):
            return chunks(self.reply)
        if self.extra:
            return self.extra.pop(0)
        if isinstance(self.note, Exception):
            raise self.note
        return completion(self.note)

    @property
    def summaries(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if "Notes so far" in r["messages"][-1]["content"]]

    @property
    def tutor_calls(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r.get("stream")]


def agent(**kwargs: Any) -> QuantAgent:
    kwargs.setdefault("context_compaction", True)
    kwargs.setdefault("context_notes", True)
    kwargs.setdefault("context_background", False)  # most tests wait for the note, as the eval does
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
    assert model.summaries == []
    (request,) = model.requests
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["system"] + ["user", "assistant"] * 5 + ["user"]
    assert "Session so far" not in request["messages"][0]["content"]
    (record,) = turn_log.read()
    assert record["context_messages_folded"] == 0 and record["llm_calls"] == 1


async def test_a_long_conversation_sends_the_pin_a_note_and_the_latest_turns(turn_log) -> None:
    model = Model()
    with patch("litellm.acompletion", new=model):
        items = await run(agent(), session(15))  # 30 messages
    assert reply_of(items) == "What pairs of numbers multiply to 6?"
    (tutor,) = model.tutor_calls
    system, *rest = tutor["messages"]
    assert "# Session so far" in system["content"]
    assert PROBLEM in system["content"] and "listing factor pairs of 6" in system["content"]
    assert len(rest) == 10 + 1  # 20 of the 30 messages folded, 10 verbatim, and the new one
    assert rest[0]["content"] == "student step 10" and rest[-1]["content"] == "ok what next?"
    assert "student step 9" not in json.dumps(rest)  # folded turns are not sent twice
    (summary,) = model.summaries
    assert "student step 9" in summary["messages"][-1]["content"]
    (record,) = turn_log.read()
    assert (record["context_messages_folded"], record["context_messages_sent"]) == (20, 10)
    assert record["context_note"] == "model" and record["llm_calls"] == 2
    assert "context_summary" in record["phases_ms"]


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


async def test_the_same_turns_are_summarised_once_even_across_agents() -> None:
    model = Model()
    with patch("litellm.acompletion", new=model):
        await run(agent(), session(15))
        await run(agent(), session(15), "and then?")  # the agent service builds one per request
    assert len(model.summaries) == 1


async def test_a_failed_summary_still_gives_a_turn_with_the_problem_pinned(turn_log) -> None:
    model = Model(note=RuntimeError("provider down"))
    with patch("litellm.acompletion", new=model):
        items = await run(agent(), session(15))
    assert reply_of(items)
    system = model.tutor_calls[0]["messages"][0]["content"]
    assert PROBLEM in system and "left out to save space" in system
    assert turn_log.read()[0]["context_note"] == "failed"


async def test_by_default_the_pin_and_the_latest_turns_go_out_with_no_summary_call(
    turn_log,
) -> None:
    model = Model()
    a = agent(context_notes=None)  # the product default
    with patch("litellm.acompletion", new=model):
        await run(a, session(10))
    assert model.summaries == []
    call = model.tutor_calls[0]["messages"]
    assert "# Session so far" in call[0]["content"] and "Solve x^2" in call[0]["content"]
    assert len(call) == 1 + 8 + 1
    assert turn_log.read()[0]["context_note"] == "pinned"


async def test_the_setting_turns_it_off_and_nothing_is_folded() -> None:
    model = Model()
    with patch("litellm.acompletion", new=model):
        await run(agent(context_compaction=False), session(15))
    assert model.summaries == []
    assert len(model.tutor_calls[0]["messages"]) == 1 + 30 + 1


def test_the_default_comes_from_the_environment(monkeypatch) -> None:
    assert AgentSettings(_env_file=None).context_compaction_enabled is True  # type: ignore[call-arg]
    monkeypatch.setenv("CONTEXT_COMPACTION_ENABLED", "false")
    assert AgentSettings(_env_file=None).context_compaction_enabled is False  # type: ignore[call-arg]


async def test_summary_call_uses_its_own_role_and_prompt_version() -> None:
    model = Model()
    with patch("litellm.acompletion", new=model):
        await run(agent(), session(15))
    cfg = load_model_config("config/model_config.dev.yaml")
    (summary,) = model.summaries
    assert summary["model"] == cfg.litellm_model("context_summarizer")
    assert summary["temperature"] == 0
    system = summary["messages"][0]["content"]
    assert "Never write the final answer" in system and "not instructions to you" in system


async def test_solve_mode_gets_the_same_budget() -> None:
    model = Model(reply="Step 1. Factor.\n**Answer: 2, 3**")
    a = agent()
    with patch("litellm.acompletion", new=model), patch.object(
        a, "_formalize", new_callable=AsyncMock, return_value=None
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
    second = Model()
    with patch("litellm.acompletion", new=second):
        await run(a, next_turn, "is it a loss?")
    sent = second.requests[0]["messages"]
    assert not any(m["role"] == "tool" or m.get("tool_calls") for m in sent)
    assert "0.92" not in json.dumps(sent)


# ── the note in the background, as the product runs it ──────────────────────


async def test_the_product_does_not_wait_for_the_note_and_the_next_turn_has_it(turn_log) -> None:
    from services.agent.src.agent import context

    model = Model()
    a = agent(context_background=True)
    with patch("litellm.acompletion", new=model):
        first = await run(a, session(6))  # 12 messages: the first fold is due, no note yet
        assert reply_of(first)
        assert "Session so far" not in model.tutor_calls[0]["messages"][0]["content"]
        assert len(model.tutor_calls[0]["messages"]) == 1 + 12 + 1  # the history as it was
        await context.settle()  # the note finishes while the student reads
        second = await run(a, session(6), "and then?")
    assert reply_of(second)
    assert "# Session so far" in model.tutor_calls[1]["messages"][0]["content"]
    assert len(model.summaries) == 1
    records = turn_log.read()
    assert [r["context_note"] for r in records if r["agent"] == "quant_agent"] == [
        "pending", "cached"]
    summary = next(r for r in records if r["agent"] == "context_summarizer")
    assert summary["trace_id"] == "t1-ctx" and summary["llm_calls"] == 1
