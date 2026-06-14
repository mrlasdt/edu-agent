"""
Tests for AWAgent Solve mode + Personal style Skill (issue 09).

Behaviors under test:
  1. Solve mode yields essay tokens
  2. Solve mode uses the solve system prompt (different from tutor)
  3. Style critic called post-generation; structural violation triggers one rewrite
  4. Personal style: injected as few-shot when opt-in + >= 2 essays available
  5. Personal style: no-ops silently (canonical style) when < 2 essays
  6. Disclaimer appended to every Solve output
  7. Citation checker runs on Solve output same as Tutor
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.agent.src.agent.agents.aw import AWAgent, ESSAY_DISCLAIMER
from shared.src.shared.models import Mode, Session


def make_session(mode: Mode = Mode.solve, personal_style: bool = False) -> Session:
    return Session(
        candidate_id="c1", test_type="gre", section="aw",
        mode=mode, personal_style_enabled=personal_style,
    )


def mock_stream(tokens: list[str]):
    async def _stream():
        for token in tokens:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            yield chunk
    return _stream()


SAMPLE_ESSAY_TOKENS = [
    "Technology", " has", " transformed", " society.", " This essay", " argues…"
]


# ── 1. Solve yields essay tokens ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_solve_yields_tokens():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_check_citations", return_value=(True, [])), \
         patch.object(agent, "_run_style_critic", new_callable=AsyncMock, return_value=(True, "")), \
         patch.object(agent, "_get_personal_style_exemplars", new_callable=AsyncMock, return_value=[]):
        mock_llm.return_value = mock_stream(SAMPLE_ESSAY_TOKENS)
        tokens = [
            item async for item in agent.run(make_session(Mode.solve), "Discuss technology")
            if isinstance(item, str)
        ]

    assert len(tokens) > 0


# ── 2. Solve uses solve system prompt ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_solve_uses_different_prompt_from_tutor():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    tutor_system = solve_system = None

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_check_citations", return_value=(True, [])), \
         patch.object(agent, "_run_style_critic", new_callable=AsyncMock, return_value=(True, "")), \
         patch.object(agent, "_get_personal_style_exemplars", new_callable=AsyncMock, return_value=[]):
        mock_llm.return_value = mock_stream(["plan"])
        async for _ in agent.run(make_session(Mode.tutor), "Discuss tech"):
            pass
        tutor_system = mock_llm.call_args.kwargs["messages"][0]["content"]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_check_citations", return_value=(True, [])), \
         patch.object(agent, "_run_style_critic", new_callable=AsyncMock, return_value=(True, "")), \
         patch.object(agent, "_get_personal_style_exemplars", new_callable=AsyncMock, return_value=[]):
        mock_llm.return_value = mock_stream(["essay"])
        async for _ in agent.run(make_session(Mode.solve), "Discuss tech"):
            pass
        solve_system = mock_llm.call_args.kwargs["messages"][0]["content"]

    assert tutor_system != solve_system


# ── 3. Style critic triggers rewrite on violation ─────────────────────────────

@pytest.mark.asyncio
async def test_style_critic_triggers_rewrite_on_violation():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")
    call_count = 0

    async def count_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_stream(["essay content"])

    with patch("litellm.acompletion", side_effect=count_llm), \
         patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_check_citations", return_value=(True, [])), \
         patch.object(agent, "_get_personal_style_exemplars", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_run_style_critic", new_callable=AsyncMock) as mock_critic:
        # First call: violation found; second call: passes
        mock_critic.side_effect = [(False, "Missing PEEL structure"), (True, "")]
        async for _ in agent.run(make_session(Mode.solve), "Discuss tech"):
            pass

    assert call_count >= 2  # initial essay + style rewrite


@pytest.mark.asyncio
async def test_style_critic_no_rewrite_when_structure_ok():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")
    call_count = 0

    async def count_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_stream(["perfect essay"])

    with patch("litellm.acompletion", side_effect=count_llm), \
         patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_check_citations", return_value=(True, [])), \
         patch.object(agent, "_get_personal_style_exemplars", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_run_style_critic", new_callable=AsyncMock, return_value=(True, "")):
        async for _ in agent.run(make_session(Mode.solve), "Discuss tech"):
            pass

    assert call_count == 1


# ── 4. Personal style injected when active + >= 2 essays ─────────────────────

@pytest.mark.asyncio
async def test_personal_style_exemplars_injected_in_messages():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")
    exemplars = ["Essay one by candidate.", "Essay two by candidate."]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_check_citations", return_value=(True, [])), \
         patch.object(agent, "_run_style_critic", new_callable=AsyncMock, return_value=(True, "")), \
         patch.object(agent, "_get_personal_style_exemplars", new_callable=AsyncMock, return_value=exemplars):
        mock_llm.return_value = mock_stream(["essay"])
        async for _ in agent.run(make_session(Mode.solve, personal_style=True), "Discuss tech"):
            pass

    messages = mock_llm.call_args.kwargs["messages"]
    full_context = " ".join(m["content"] for m in messages)
    assert "Essay one by candidate" in full_context


# ── 5. Personal style no-ops with < 2 essays ─────────────────────────────────

@pytest.mark.asyncio
async def test_personal_style_noop_with_fewer_than_2_essays():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_check_citations", return_value=(True, [])), \
         patch.object(agent, "_run_style_critic", new_callable=AsyncMock, return_value=(True, "")), \
         patch.object(agent, "_get_personal_style_exemplars", new_callable=AsyncMock, return_value=["only one essay"]):
        mock_llm.return_value = mock_stream(["canonical essay"])
        async for _ in agent.run(make_session(Mode.solve, personal_style=True), "Discuss tech"):
            pass

    messages = mock_llm.call_args.kwargs["messages"]
    full_context = " ".join(m["content"] for m in messages)
    assert "only one essay" not in full_context


# ── 6. Disclaimer appended ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disclaimer_always_appended_to_solve_output():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_check_citations", return_value=(True, [])), \
         patch.object(agent, "_run_style_critic", new_callable=AsyncMock, return_value=(True, "")), \
         patch.object(agent, "_get_personal_style_exemplars", new_callable=AsyncMock, return_value=[]):
        mock_llm.return_value = mock_stream(["essay content here"])
        items = [item async for item in agent.run(make_session(Mode.solve), "Discuss tech")]

    text_items = [i for i in items if isinstance(i, str)]
    full_output = "".join(text_items)
    assert ESSAY_DISCLAIMER in full_output


# ── 7. Citation checker runs on Solve output ─────────────────────────────────

@pytest.mark.asyncio
async def test_citation_checker_runs_on_solve():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_check_citations", return_value=(True, [])) as mock_check, \
         patch.object(agent, "_run_style_critic", new_callable=AsyncMock, return_value=(True, "")), \
         patch.object(agent, "_get_personal_style_exemplars", new_callable=AsyncMock, return_value=[]):
        mock_llm.return_value = mock_stream(["essay"])
        async for _ in agent.run(make_session(Mode.solve), "Discuss tech"):
            pass

    mock_check.assert_called()
