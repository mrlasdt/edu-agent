"""
Just-in-time retrieval: the tutor looks notes up itself instead of having them injected.

Behaviors under test:
  1. notes_search numbers the notes of a turn once (a note found twice keeps its number), asks the
     corpus for the concept the model wrote and not the student's message, and tells the model
     what to change when the concept is missing or too long
  2. a corpus that is down costs the model a tool result, not the turn
  3. ToolBox sends each call to the set that owns the name and lists the valid names otherwise
  4. in just-in-time mode a tutor turn injects no notes and does not search up front; a turn that
     needs nothing makes one model call and reports 0 sources (so any [N] in its reply is
     stripped by the gateway); a turn that searches reports the notes it was shown
  5. the math tools and the notes tool can be offered together
  6. solve mode keeps the up-front search; the default is still up-front
  7. the turn record says which retrieval ran and what prompt version was used
"""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk, RetrieveCorpus
from services.agent.src.agent.settings import AgentSettings
from services.agent.src.agent.tools.math_tools import MATH_TOOL_SPECS
from services.agent.src.agent.tools.notes_tools import (
    MAX_CONCEPT_CHARS,
    NOTES_TOOL_SPECS,
    NotesTools,
)
from services.agent.src.agent.tools.toolbox import ToolBox
from shared.src.shared.config import load_model_config
from shared.src.shared.events import Event, Sources, Token
from shared.src.shared.models import Mode, Session
from tests.agent_events import reply_of

FACTORING = CorpusChunk(
    text="To factor x^2 + bx + c find two numbers that multiply to c and add to b.",
    tier="global", source_uri="seed/gre-quant/algebra.txt", page_or_section="Factoring",
    score=0.8,
)
SQUARES = CorpusChunk(
    text="a^2 - b^2 = (a - b)(a + b).",
    tier="global", source_uri="seed/gre-quant/algebra.txt", page_or_section="Difference of squares",
    score=0.7,
)


class Corpus(RetrieveCorpus):
    """The corpus as a recorded call: each answer is a list of notes or an exception to raise."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.queries: list[dict[str, Any]] = []

    async def retrieve(self, query, candidate_id, school_id, test_type, top_k=5):  # type: ignore[no-untyped-def]
        self.queries.append(
            {"query": query, "candidate_id": candidate_id, "test_type": test_type, "top_k": top_k}
        )
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def session(mode: Mode = Mode.tutor) -> Session:
    return Session(candidate_id="c1", test_type="gre", section="quant", mode=mode)


def text_reply(text: str | None, calls: list[Any] | None = None) -> Any:
    message = SimpleNamespace(content=text, tool_calls=calls)
    usage = SimpleNamespace(prompt_tokens=300, completion_tokens=20)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def tool_reply(name: str, call_id: str = "c1", **arguments: Any) -> Any:
    call = SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )
    return text_reply(None, [call])


class Model:
    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


async def one_chunk(text: str) -> Any:
    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def jit_agent(corpus: RetrieveCorpus, **kwargs: Any) -> QuantAgent:
    return QuantAgent(
        model_config_path="config/model_config.dev.yaml",
        retrieve_corpus=corpus,
        retrieval_mode="just_in_time",
        verify_math=AsyncMock(),
        **kwargs,
    )


async def run(a: QuantAgent, text: str, mode: Mode = Mode.tutor) -> list[Event]:
    return [e async for e in a.run(session(mode), text, trace_id="t1")]


# ── 1. notes_search ──────────────────────────────────────────────────────────


async def test_notes_are_numbered_once_per_turn_and_the_corpus_gets_the_concept() -> None:
    corpus = Corpus([FACTORING, SQUARES], [SQUARES])
    notes = NotesTools(corpus, candidate_id="c1", test_type="gre")
    first = await notes.call("notes_search", {"concept": "factoring a quadratic"})
    second = await notes.call("notes_search", {"concept": "difference of squares"})
    assert [(n["n"], n["section"]) for n in first["notes"]] == [
        (1, "Factoring"), (2, "Difference of squares")]
    assert [n["n"] for n in second["notes"]] == [2]  # found again: same number
    assert notes.chunks == [FACTORING, SQUARES]
    asked = [q["query"] for q in corpus.queries]
    assert asked == ["factoring a quadratic", "difference of squares"]
    assert {q["candidate_id"] for q in corpus.queries} == {"c1"} and corpus.queries[0]["top_k"] == 3


async def test_a_missing_or_overlong_concept_is_an_error_that_says_what_to_send() -> None:
    corpus = Corpus([FACTORING])
    notes = NotesTools(corpus, candidate_id="c1", test_type="gre")
    assert "concept is required" in (await notes.call("notes_search", {}))["error"]
    blank = await notes.call("notes_search", {"concept": "  "})
    assert "concept is required" in blank["error"]
    long = await notes.call("notes_search", {"concept": "x" * (MAX_CONCEPT_CHARS + 1)})
    assert f"limit is {MAX_CONCEPT_CHARS}" in long["error"]
    assert "not the problem text" in long["error"]
    assert corpus.queries == [] and notes.chunks == []


async def test_no_matching_note_is_an_empty_result_not_an_error() -> None:
    notes = NotesTools(Corpus([]), candidate_id="c1", test_type="gre")
    assert await notes.call("notes_search", {"concept": "pasta recipes"}) == {"notes": []}


# ── 2. the corpus is down ────────────────────────────────────────────────────


async def test_a_corpus_that_is_down_gives_the_model_an_error_and_the_turn_goes_on() -> None:
    corpus = Corpus(ConnectionError("refused"))
    model = Model(
        tool_reply("notes_search", concept="factoring a quadratic"),
        text_reply("What two numbers multiply to 6 and add to -5?"),
    )
    with patch("litellm.acompletion", new=model):
        items = await run(jit_agent(corpus), "I'm stuck on x^2 - 5x + 6 = 0")
    assert reply_of(items) == "What two numbers multiply to 6 and add to -5?"
    assert items[0] == Sources(count=0)
    seen = [m for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert "could not be reached" in seen[0]["content"]


# ── 3. ToolBox ───────────────────────────────────────────────────────────────


async def test_toolbox_routes_by_name_and_lists_the_valid_names_for_an_unknown_one() -> None:
    notes = NotesTools(Corpus([FACTORING]), candidate_id="c1", test_type="gre")
    math = AsyncMock(return_value={"value": "2"})
    box = ToolBox((MATH_TOOL_SPECS, math), (NOTES_TOOL_SPECS, notes.call))
    assert box.names == ["math_evaluate", "math_solve", "math_compare", "notes_search"]
    assert (await box.call("math_evaluate", {"expression": "1+1"})) == {"value": "2"}
    assert (await box.call("notes_search", {"concept": "factoring"}))["notes"][0]["n"] == 1
    unknown = await box.call("web_search", {})
    assert "'web_search'" in unknown["error"] and "notes_search" in unknown["error"]
    with pytest.raises(ValueError, match="both define"):
        ToolBox((NOTES_TOOL_SPECS, notes.call), (NOTES_TOOL_SPECS, notes.call))


# ── 4. the tutor turn ────────────────────────────────────────────────────────


async def test_a_tutor_turn_injects_no_notes_and_does_not_search_up_front() -> None:
    corpus = Corpus([FACTORING])
    model = Model(text_reply("Nice. What do you get when you multiply the two numbers?"))
    a = jit_agent(corpus)
    with patch("litellm.acompletion", new=model), patch.object(
        a, "_retrieve", new_callable=AsyncMock
    ) as upfront:
        items = await run(a, "ok so what next?")
    upfront.assert_not_awaited()
    assert corpus.queries == []
    (request,) = model.requests
    assert "Reference Notes" not in request["messages"][-1]["content"]
    assert request["messages"][-1]["content"] == "ok so what next?"
    assert [t["function"]["name"] for t in request["tools"]] == ["notes_search"]
    system = request["messages"][0]["content"]
    assert "notes_search" in system
    assert "Never state the final numeric or symbolic answer" in system
    # a turn that searched nothing reports 0 sources, so the gateway strips any [N] it wrote
    reply = "Nice. What do you get when you multiply the two numbers?"
    assert items == [Sources(count=0), Token(text=reply)]


async def test_a_turn_that_searches_reports_its_notes_and_keeps_their_text_private() -> None:
    corpus = Corpus([FACTORING, SQUARES])
    model = Model(
        tool_reply("notes_search", concept="factoring a quadratic"),
        text_reply("Look for two numbers that multiply to 6 and add to -5 [1]."),
    )
    with patch("litellm.acompletion", new=model):
        items = await run(jit_agent(corpus), "Here is my question: solve x^2 - 5x + 6 = 0. Hint?")
    assert items[0] == Sources(count=2)
    assert isinstance(items[1], Token) and "[1]" in items[1].text
    assert FACTORING.text not in reply_of(items)
    assert [q["query"] for q in corpus.queries] == ["factoring a quadratic"]  # not the message
    result = next(m for m in model.requests[1]["messages"] if m["role"] == "tool")
    assert json.loads(result["content"])["notes"][0]["text"] == FACTORING.text


async def test_the_gateway_strips_a_citation_from_a_turn_that_searched_nothing() -> None:
    from services.gateway.src.gateway.guardrails.stream import ReplyGuard

    guard = ReplyGuard(Mode.tutor, "t1", 0)
    out = guard.feed("Think about factoring [1]. What two numbers multiply to 6?") + guard.finish()
    assert "[1]" not in out and "What two numbers multiply to 6?" in out


# ── 5. both tool sets at once ────────────────────────────────────────────────


async def test_math_tools_and_notes_search_are_offered_together() -> None:
    corpus = Corpus([FACTORING])
    model = Model(text_reply("What do you get for 15% of 80?"))
    a = jit_agent(corpus, math_tools=AsyncMock(), math_tools_enabled=True)
    with patch("litellm.acompletion", new=model):
        await run(a, "thanks, and now?")
    names = [t["function"]["name"] for t in model.requests[0]["tools"]]
    assert names == ["math_evaluate", "math_solve", "math_compare", "notes_search"]
    system = model.requests[0]["messages"][0]["content"]
    assert "# Math tools" in system and "# Reference notes" in system


# ── 6. solve mode and the default ────────────────────────────────────────────


async def test_solve_mode_keeps_the_up_front_search_and_has_no_notes_tool() -> None:
    corpus = Corpus([FACTORING])
    a = jit_agent(corpus)
    streams: list[dict[str, Any]] = []

    def stream(**kwargs: Any) -> Any:
        streams.append(kwargs)
        return one_chunk("1. Factor.\n**Answer: 2, 3**")

    with patch("shared.src.shared.turn_meter.TurnMeter.stream", side_effect=stream), patch.object(
        a, "_formalize", new_callable=AsyncMock, return_value=None
    ), patch.object(a, "verify_well_formed", new_callable=AsyncMock) as verified:
        verified.return_value = SimpleNamespace(verified=True, computed="[2, 3]", error=None)
        items = await run(a, "Solve x^2 - 5x + 6 = 0", Mode.solve)
    assert corpus.queries and corpus.queries[0]["query"] == "Solve x^2 - 5x + 6 = 0"
    assert Sources(count=1) in items
    assert "tools" not in streams[0]
    assert "Reference Notes" in streams[0]["messages"][-1]["content"]


def test_the_default_is_up_front_and_an_unknown_mode_is_refused(monkeypatch) -> None:
    assert AgentSettings(_env_file=None).retrieval_mode == "upfront"  # type: ignore[call-arg]
    monkeypatch.setenv("RETRIEVAL_MODE", "just_in_time")
    assert AgentSettings(_env_file=None).retrieval_mode == "just_in_time"  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="retrieval_mode 'sometimes'"):
        QuantAgent(model_config_path="config/model_config.dev.yaml", retrieval_mode="sometimes")


async def test_the_default_agent_still_searches_up_front_with_the_whole_message() -> None:
    corpus = Corpus([FACTORING])
    a = QuantAgent(model_config_path="config/model_config.dev.yaml", retrieve_corpus=corpus,
                   retrieval_mode="upfront", verify_math=AsyncMock())

    with patch(
        "shared.src.shared.turn_meter.TurnMeter.stream", side_effect=lambda **_: one_chunk("Hint.")
    ):
        items = await run(a, "ok so what next?")
    assert corpus.queries[0]["query"] == "ok so what next?" and corpus.queries[0]["top_k"] == 3
    assert items[0] == Sources(count=1)


# ── 7. the turn record ───────────────────────────────────────────────────────


async def test_the_turn_record_names_the_retrieval_and_the_prompt_that_ran(turn_log) -> None:
    corpus = Corpus([FACTORING])
    model = Model(
        tool_reply("notes_search", concept="factoring"),
        text_reply("Try two numbers that add to -5."),
    )
    with patch("litellm.acompletion", new=model):
        await run(jit_agent(corpus), "stuck")
    (record,) = turn_log.read()
    assert record["retrieval"] == "just_in_time"
    assert record["notes_retrieved"] == 1 and record["notes_searches"] == 1
    assert [(c["name"], c["ok"]) for c in record["tool_calls"]] == [("notes_search", True)]
    cfg = load_model_config("config/model_config.dev.yaml")
    assert record["prompt_version"] == (
        f"{cfg.prompt_version('quant_agent', 'tutor')}+notes_tool-"
        f"{cfg.prompt_version('quant_agent_notes_tool', 'tutor')}"
    )
    assert "math_tools" not in record
