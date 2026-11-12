"""
Tests for AWAgent Tutor mode.

Behaviors under test (issue 08):
  1. Tutor mode yields a plan, not a full essay (short structured output)
  2. retrieve-corpus capability is called for every AW turn
  3. Retrieved chunks are injected into the LLM message context
  4. Citation checker is called post-generation
  5. On citation failure → regenerate once; if still failing → strip + citation_stripped=True
  6. run() returns an async generator + metadata dict
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.agent.src.agent.agents.aw import AWAgent
from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk
from shared.src.shared.events import CitationsStripped
from shared.src.shared.models import Mode, Session
from tests.agent.agent_events import texts


def make_session(mode: Mode = Mode.tutor) -> Session:
    return Session(candidate_id="c1", test_type="gre", section="aw", mode=mode)


def mock_stream(tokens: list[str]):
    async def _stream():
        for token in tokens:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            yield chunk

    return _stream()


MOCK_CHUNKS = [
    CorpusChunk(
        text="The GRE Issue task requires...",
        source_uri="ets-guide.pdf",
        page_or_section="Introduction",
        tier="global",
        score=0.9,
    ),
    CorpusChunk(
        text="A strong thesis statement...",
        source_uri="ets-guide.pdf",
        page_or_section="Writing Tips",
        tier="global",
        score=0.8,
    ),
]


# ── 1. Tutor returns tokens (plan) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tutor_yields_tokens():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with (
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
        patch.object(agent, "_retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch.object(agent, "_check_citations", return_value=(True, [])),
    ):
        mock_llm.return_value = mock_stream(["Outline:", " 1.", " Introduction"])
        mock_retrieve.return_value = MOCK_CHUNKS
        items = [item async for item in agent.run(make_session(Mode.tutor), "Discuss technology")]

    assert texts(items) == ["Outline:", " 1.", " Introduction"]


# ── 2. retrieve-corpus capability called ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_tutor_calls_retrieve():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with (
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
        patch.object(agent, "_retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch.object(agent, "_check_citations", return_value=(True, [])),
    ):
        mock_llm.return_value = mock_stream(["plan"])
        mock_retrieve.return_value = []
        async for _ in agent.run(make_session(), "Discuss innovation"):
            pass

    mock_retrieve.assert_called_once()


# ── 3. Chunks injected into LLM context ───────────────────────────────────────


@pytest.mark.asyncio
async def test_chunks_injected_into_messages():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with (
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
        patch.object(agent, "_retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch.object(agent, "_check_citations", return_value=(True, [])),
    ):
        mock_llm.return_value = mock_stream(["plan"])
        mock_retrieve.return_value = MOCK_CHUNKS
        async for _ in agent.run(make_session(), "Discuss innovation"):
            pass

    messages = mock_llm.call_args.kwargs["messages"]
    full_context = " ".join(m["content"] for m in messages)
    assert "GRE Issue task" in full_context or "strong thesis" in full_context


# ── 4. Citation checker called post-generation ───────────────────────────────


@pytest.mark.asyncio
async def test_citation_checker_called():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with (
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
        patch.object(agent, "_retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch.object(agent, "_check_citations", return_value=(True, [])) as mock_check,
    ):
        mock_llm.return_value = mock_stream(["plan with [1] citation"])
        mock_retrieve.return_value = MOCK_CHUNKS
        async for _ in agent.run(make_session(), "Discuss technology"):
            pass

    mock_check.assert_called_once()


# ── 5. Citation failure → regenerate once → strip on second failure ───────────


@pytest.mark.asyncio
async def test_citation_failure_triggers_regeneration():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")
    call_count = 0

    async def count_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_stream(["plan"])

    with (
        patch("litellm.acompletion", side_effect=count_llm),
        patch.object(agent, "_retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch.object(agent, "_check_citations", return_value=(False, ["uncited claim"])),
    ):
        mock_retrieve.return_value = MOCK_CHUNKS
        async for _ in agent.run(make_session(), "Discuss technology"):
            pass

    assert call_count >= 2  # initial + at least 1 retry


@pytest.mark.asyncio
async def test_persistent_citation_failure_emits_stripped_metadata():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with (
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
        patch.object(agent, "_retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch.object(agent, "_check_citations", return_value=(False, ["bad claim"])),
    ):
        mock_llm.return_value = mock_stream(["plan"])
        mock_retrieve.return_value = MOCK_CHUNKS
        items = [item async for item in agent.run(make_session(), "Discuss technology")]

    assert CitationsStripped(stripped=True) in items


# ── 6. Metadata event yielded at end ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_citation_metadata_is_emitted_as_the_last_item():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with (
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
        patch.object(agent, "_retrieve", new_callable=AsyncMock) as mock_retrieve,
        patch.object(agent, "_check_citations", return_value=(True, [])),
    ):
        mock_llm.return_value = mock_stream(["plan"])
        mock_retrieve.return_value = []
        items = [item async for item in agent.run(make_session(), "Discuss technology")]

    assert items[-1] == CitationsStripped(stripped=False)
