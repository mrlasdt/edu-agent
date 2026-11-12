"""
The turn every agent shares (BaseAgent.run): what a subclass gets without writing it, and what a
turn leaves in its record when it fails.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest

from services.agent.src.agent.agents.ielts_writing import IeltsWritingAgent
from services.agent.src.agent.base import BaseAgent, Turn
from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk
from services.agent.src.agent.prompting import token_stream
from shared.src.shared.events import Event, Sources, Token
from shared.src.shared.models import Mode, Session


class _Agent(BaseAgent[Turn]):
    ROLE = "quant_agent"
    TOP_K = 2
    fail = False

    async def tutor_turn(self, turn: Turn) -> AsyncGenerator[Event, None]:
        yield Token(text=f"tutor:{turn.chunk_context}")
        if self.fail:
            raise RuntimeError("model down")

    async def solve_turn(self, turn: Turn) -> AsyncGenerator[Event, None]:
        yield Token(text="solve")

    @staticmethod
    def format_chunks(chunks: list[CorpusChunk]) -> str:
        return "|".join(c.text for c in chunks)


def _session(mode: Mode = Mode.tutor) -> Session:
    return Session(candidate_id="c1", test_type="gre", section="quant", mode=mode)


def _chunk(text: str) -> CorpusChunk:
    return CorpusChunk(text=text, tier="global", source_uri="u", page_or_section="p", score=0.5)


async def _run(agent: BaseAgent[Turn], mode: Mode = Mode.tutor) -> list[Event]:
    return [e async for e in agent.run(_session(mode), "question", trace_id="t1")]


def _agent(chunks: list[CorpusChunk]) -> _Agent:
    agent = _Agent("config/model_config.dev.yaml")
    agent._retrieve = AsyncMock(return_value=chunks)  # type: ignore[method-assign]
    return agent


@pytest.mark.asyncio
async def test_notes_are_reported_then_handed_to_the_mode_that_runs():
    events = await _run(_agent([_chunk("a"), _chunk("b")]))
    assert events == [Sources(count=2), Token(text="tutor:a|b")]
    assert await _run(_agent([]), Mode.solve) == [Token(text="solve")]  # no notes, no Sources


@pytest.mark.asyncio
async def test_retrieval_asks_the_corpus_for_top_k_and_fails_open():
    agent = _Agent("config/model_config.dev.yaml", retrieve_corpus=AsyncMock())
    agent._retrieve_corpus.retrieve.return_value = [_chunk("a")]  # type: ignore[union-attr]
    await _run(agent)
    assert agent._retrieve_corpus.retrieve.await_args.kwargs["top_k"] == 2  # type: ignore[union-attr]

    agent._retrieve_corpus.retrieve.side_effect = ConnectionError("corpus down")  # type: ignore[union-attr]
    assert await _run(agent) == [Token(text="tutor:")]  # the turn still answers, ungrounded


@pytest.mark.asyncio
async def test_a_failing_turn_raises_and_its_record_names_the_error(turn_log):
    agent = _agent([])
    agent.fail = True
    with pytest.raises(RuntimeError):
        await _run(agent)
    (record,) = turn_log.read()
    assert record["error"] == "RuntimeError"
    assert record["agent"] == "quant_agent"


@pytest.mark.asyncio
async def test_a_single_mode_agent_plays_its_mode_whatever_the_session_asks(turn_log):
    agent = IeltsWritingAgent("config/model_config.dev.yaml")
    events = await _run(agent, Mode.solve)  # no report digest: the coach says so, in tutor mode
    assert len(events) == 1 and isinstance(events[0], Token)
    (record,) = turn_log.read()
    assert record["mode"] == "tutor"


@pytest.mark.asyncio
async def test_token_stream_yields_only_the_chunks_that_carry_text():
    class _Delta:
        def __init__(self, content: str | None) -> None:
            self.choices = [type("C", (), {"delta": type("D", (), {"content": content})()})()]

    async def chunks():
        for content in ("a", None, "", "b"):
            yield _Delta(content)

    assert [t async for t in token_stream(chunks())] == ["a", "b"]
