"""
Tests for Langfuse LLM observability wiring (issue 17).

Behaviors under test:
  1. setup_observability() configures LiteLLM callbacks when Langfuse keys present
  2. setup_observability() is a no-op (no error) when keys are absent
  3. build_llm_metadata() includes trace_id
  4. build_llm_metadata() includes prompt_version tag
  5. build_llm_metadata() includes experiment_name when provided
  6. build_llm_metadata() includes cache hit/miss field
  7. QuantAgent.run() calls LiteLLM with metadata containing trace_id
  8. QuantAgent.run() calls LiteLLM with metadata containing prompt_version
  9. AWAgent.run() (tutor and solve) and the style critic pass trace_id and prompt_version
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.agent.src.agent.agents.aw import AWAgent
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.critics.style_critic import run_style_critic
from shared.src.shared.models import Mode, Session
from shared.src.shared.observability import build_llm_metadata, setup_observability


def make_session() -> Session:
    return Session(candidate_id="c1", test_type="gre", section="quant", mode=Mode.tutor)


def mock_stream(tokens=None):
    async def _stream():
        for token in (tokens or ["test"]):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            yield chunk
    return _stream()


# ── 1. setup_observability with keys present ──────────────────────────────────

@pytest.fixture
def clean_callbacks(monkeypatch):
    import litellm

    from shared.src.shared.config import get_settings

    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    get_settings.cache_clear()
    yield litellm
    get_settings.cache_clear()


def test_setup_observability_enables_langfuse_callback(monkeypatch, clean_callbacks):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")

    setup_observability()
    setup_observability()  # idempotent

    assert clean_callbacks.success_callback == ["langfuse"]
    assert clean_callbacks.failure_callback == ["langfuse"]


# ── 2. setup_observability no-op when keys absent ────────────────────────────

def test_setup_observability_noop_without_keys(monkeypatch, clean_callbacks):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

    setup_observability()

    assert clean_callbacks.success_callback == []
    assert clean_callbacks.failure_callback == []


def test_agent_service_registers_callback_at_startup(monkeypatch, clean_callbacks):
    """Importing the agent service app wires Langfuse when keys are configured."""
    import importlib

    import services.agent.src.agent.main as agent_main
    from shared.src.shared.config import get_settings

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    get_settings.cache_clear()
    importlib.reload(agent_main)

    assert "langfuse" in clean_callbacks.success_callback


def test_agent_service_startup_leaves_callbacks_empty_without_keys(monkeypatch, clean_callbacks):
    import importlib

    import services.agent.src.agent.main as agent_main
    from shared.src.shared.config import get_settings

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    get_settings.cache_clear()
    importlib.reload(agent_main)

    assert clean_callbacks.success_callback == []


def test_gateway_service_registers_callback_at_startup(monkeypatch, clean_callbacks):
    """Importing the gateway app wires Langfuse (answer-leak check uses litellm)."""
    import importlib

    import services.gateway.src.gateway.main as gateway_main
    from shared.src.shared.config import get_settings

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    get_settings.cache_clear()
    importlib.reload(gateway_main)

    assert "langfuse" in clean_callbacks.success_callback


# ── 3. build_llm_metadata includes trace_id ──────────────────────────────────

def test_build_llm_metadata_includes_trace_id():
    meta = build_llm_metadata(trace_id="trace-abc", prompt_version="v1")
    assert meta["metadata"]["trace_id"] == "trace-abc"


# ── 4. build_llm_metadata includes prompt_version ────────────────────────────

def test_build_llm_metadata_includes_prompt_version():
    meta = build_llm_metadata(trace_id="trace-1", prompt_version="v3")
    tags = meta.get("metadata", {}).get("tags", [])
    assert any("v3" in str(t) for t in tags) or \
           meta["metadata"].get("prompt_version") == "v3"


# ── 5. build_llm_metadata includes experiment_name when provided ──────────────

def test_build_llm_metadata_includes_experiment_name():
    meta = build_llm_metadata(
        trace_id="trace-1", prompt_version="v2", experiment_name="aw-agent-v2-test"
    )
    meta_str = str(meta)
    assert "aw-agent-v2-test" in meta_str


def test_build_llm_metadata_no_experiment_name_by_default():
    meta = build_llm_metadata(trace_id="trace-1", prompt_version="v1")
    assert meta.get("metadata", {}).get("experiment_name") is None or \
           "experiment_name" not in str(meta)


# ── 6. build_llm_metadata includes cache field ────────────────────────────────

def test_build_llm_metadata_has_session_id_for_cache_grouping():
    meta = build_llm_metadata(trace_id="trace-1", prompt_version="v1", session_id="sess-99")
    assert meta["metadata"].get("session_id") == "sess-99"


# ── 7. QuantAgent passes trace_id in metadata ────────────────────────────────

@pytest.mark.asyncio
async def test_quant_agent_passes_trace_id_to_litellm():
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    session = make_session()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_stream(["hint"])
        async for _ in agent.run(session, "Solve x^2-5x+6=0", trace_id="trace-xyz"):
            pass

    call_kwargs = mock_llm.call_args.kwargs
    metadata = call_kwargs.get("metadata", {})
    assert metadata.get("trace_id") == "trace-xyz"


# ── 8. QuantAgent passes prompt_version in metadata ──────────────────────────

@pytest.mark.asyncio
async def test_quant_agent_passes_prompt_version_to_litellm():
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    session = make_session()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_stream(["hint"])
        async for _ in agent.run(session, "Solve x^2-5x+6=0", trace_id="trace-1"):
            pass

    call_kwargs = mock_llm.call_args.kwargs
    metadata = call_kwargs.get("metadata", {})
    assert "prompt_version" in metadata
    assert metadata["prompt_version"]  # non-empty


# ── 9. AWAgent and the style critic pass trace metadata ──────────────────────

def _aw_session(mode: Mode) -> Session:
    return Session(candidate_id="c1", test_type="gre", section="aw", mode=mode)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [Mode.tutor, Mode.solve])
async def test_aw_agent_passes_trace_metadata_on_every_llm_call(mode):
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")
    version = agent._model_config.prompt_version("aw_agent", mode.value)

    with (
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
        patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]),
        patch.object(agent, "_run_style_critic", new_callable=AsyncMock, return_value=(False, "x")),
        patch.object(agent, "_check_citations", side_effect=[(True, []), (True, [])]),
    ):
        mock_llm.side_effect = lambda **_: mock_stream(["essay"])
        async for _ in agent.run(_aw_session(mode), "Write an essay", trace_id="trace-aw"):
            pass

    assert mock_llm.await_count >= 1
    for call in mock_llm.await_args_list:
        metadata = call.kwargs["metadata"]
        assert metadata["trace_id"] == "trace-aw"
        assert metadata["prompt_version"] == version


@pytest.mark.asyncio
async def test_aw_solve_passes_trace_id_to_the_style_critic():
    agent = AWAgent(model_config_path="config/model_config.dev.yaml")

    with (
        patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
        patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]),
        patch.object(agent, "_check_citations", return_value=(True, [])),
        patch.object(
            agent, "_run_style_critic", new_callable=AsyncMock, return_value=(True, "")
        ) as critic,
    ):
        mock_llm.return_value = mock_stream(["essay"])
        async for _ in agent.run(_aw_session(Mode.solve), "Write an essay", trace_id="trace-aw"):
            pass

    critic.assert_awaited_once_with("essay", "trace-aw")


@pytest.mark.asyncio
async def test_style_critic_llm_call_carries_trace_metadata():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "PASS"

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=response) as mock_llm:
        passed, _ = await run_style_critic(
            "An essay.", "config/model_config.dev.yaml", trace_id="trace-sc"
        )

    assert passed is True
    metadata = mock_llm.call_args.kwargs["metadata"]
    assert metadata["trace_id"] == "trace-sc"
    assert metadata["prompt_version"]
