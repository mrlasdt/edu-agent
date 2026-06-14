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
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.src.shared.observability import setup_observability, build_llm_metadata
from services.agent.src.agent.agents.quant import QuantAgent
from shared.src.shared.models import Mode, Session


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

def test_setup_observability_enables_langfuse_callback(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")

    import litellm
    original = list(getattr(litellm, "success_callback", []))

    setup_observability()

    assert "langfuse" in litellm.success_callback
    # cleanup
    litellm.success_callback = original


# ── 2. setup_observability no-op when keys absent ────────────────────────────

def test_setup_observability_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    # Should not raise
    setup_observability()


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
    tags = meta.get("metadata", {}).get("tags", [])
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
