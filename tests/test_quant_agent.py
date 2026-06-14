"""
Tests for QuantAgent.

Behaviors under test:
  1. run() in Tutor mode returns an async generator that yields strings
  2. LiteLLM is called with the model name from model_config (not hardcoded)
  3. The session's conversation history is included in the LLM messages
  4. run() in Solve mode uses a different system prompt
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.agent.src.agent.agents.quant import QuantAgent
from shared.src.shared.models import Message, MessageRole, Mode, Session


def make_session(mode: Mode = Mode.tutor) -> Session:
    return Session(
        candidate_id="cand-1",
        test_type="gre",
        section="quant",
        mode=mode,
    )


def make_session_with_history() -> Session:
    return Session(
        candidate_id="cand-1",
        test_type="gre",
        section="quant",
        mode=Mode.tutor,
        history=[
            Message(role=MessageRole.candidate, content="What is x if x^2=4?"),
            Message(role=MessageRole.agent, content="Let's think about this step by step..."),
        ],
    )


def mock_litellm_stream(tokens: list[str]):
    """Build an async iterator that mimics litellm streaming chunks."""
    async def _stream():
        for token in tokens:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            yield chunk
    return _stream()


# ── 1. run() returns an async generator ──────────────────────────────────────

@pytest.mark.asyncio
async def test_tutor_run_returns_async_generator():
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_litellm_stream(["Think", " about", " it."])

        tokens = []
        async for token in agent.run(make_session(), "Solve x^2 - 5x + 6 = 0"):
            tokens.append(token)

    assert tokens == ["Think", " about", " it."]


# ── 2. LiteLLM called with configured model (not hardcoded) ──────────────────

@pytest.mark.asyncio
async def test_tutor_calls_litellm_with_configured_model():
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_litellm_stream(["ok"])

        async for _ in agent.run(make_session(), "A problem"):
            pass

    call_kwargs = mock_llm.call_args.kwargs
    # Should use the model from config, not a hardcoded string
    assert call_kwargs["model"] == "openai/gpt-4o"
    assert call_kwargs["stream"] is True


# ── 3. Conversation history included in messages ─────────────────────────────

@pytest.mark.asyncio
async def test_tutor_includes_history_in_llm_messages():
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    session = make_session_with_history()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_litellm_stream(["hint"])

        async for _ in agent.run(session, "New question"):
            pass

    messages = mock_llm.call_args.kwargs["messages"]
    # system prompt + 2 history messages + current user message
    assert len(messages) >= 4
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    assert "user" in roles
    assert "assistant" in roles


# ── 4. Solve mode uses solve system prompt ────────────────────────────────────

@pytest.mark.asyncio
async def test_solve_mode_uses_different_system_prompt():
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")

    tutor_messages = []
    solve_messages = []

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_litellm_stream(["answer"])
        async for _ in agent.run(make_session(Mode.tutor), "A problem"):
            pass
        tutor_messages = mock_llm.call_args.kwargs["messages"]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_litellm_stream(["answer"])
        async for _ in agent.run(make_session(Mode.solve), "A problem"):
            pass
        solve_messages = mock_llm.call_args.kwargs["messages"]

    tutor_system = tutor_messages[0]["content"]
    solve_system = solve_messages[0]["content"]
    assert tutor_system != solve_system, "Tutor and Solve must use different system prompts"
