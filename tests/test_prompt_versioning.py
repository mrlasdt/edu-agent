"""
Tests for config-driven prompt versioning (closes the ADR-0009 gap).

Behaviors under test:
  1. ModelConfig.prompt_version() returns the version configured per (agent, mode)
  2. ModelConfig.prompt_version() defaults to "v1" when unconfigured (back-compat)
  3. QuantAgent loads the prompt file for the configured version
  4. QuantAgent reports the configured version to Langfuse metadata
  5. AWAgent loads the prompt file for the configured version
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shared.src.shared.config import ModelConfig, ModelRoleConfig
from shared.src.shared.models import Mode, Session


def mock_stream(tokens=None):
    async def _stream():
        for token in (tokens or ["t"]):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            yield chunk
    return _stream()


# ── 1 & 2. Config resolution ──────────────────────────────────────────────────

def test_prompt_version_returns_configured_value():
    cfg = ModelConfig(
        roles={"quant_agent": ModelRoleConfig(provider="openai", model="gpt-4o")},
        prompts={"quant_agent": {"tutor": "v3", "solve": "v2"}},
    )
    assert cfg.prompt_version("quant_agent", "tutor") == "v3"
    assert cfg.prompt_version("quant_agent", "solve") == "v2"


def test_prompt_version_defaults_to_v1_when_unconfigured():
    cfg = ModelConfig(
        roles={"quant_agent": ModelRoleConfig(provider="openai", model="gpt-4o")},
    )
    assert cfg.prompt_version("quant_agent", "tutor") == "v1"
    assert cfg.prompt_version("aw_agent", "solve") == "v1"


def test_dev_config_declares_prompt_versions():
    from shared.src.shared.config import load_model_config

    cfg = load_model_config("config/model_config.dev.yaml")
    # dev config must declare an active version for the shipped agents/modes
    assert cfg.prompt_version("quant_agent", "tutor") == "v1"
    assert cfg.prompt_version("aw_agent", "tutor") == "v1"


# ── 3 & 4. QuantAgent threads version into load + Langfuse ────────────────────

@pytest.mark.asyncio
async def test_quant_agent_loads_configured_prompt_version():
    from services.agent.src.agent.agents.quant import QuantAgent

    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    session = Session(candidate_id="c1", test_type="gre", section="quant", mode=Mode.tutor)

    with patch.object(ModelConfig, "prompt_version", return_value="v7"), \
         patch("services.agent.src.agent.agents.quant._load_prompt", return_value="STUB") as mock_load, \
         patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_stream(["hint"])
        async for _ in agent.run(session, "Solve x^2-5x+6=0", trace_id="t1"):
            pass

    mock_load.assert_called_with(Mode.tutor, "v7")


@pytest.mark.asyncio
async def test_quant_agent_reports_configured_version_to_langfuse():
    from services.agent.src.agent.agents.quant import QuantAgent

    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    session = Session(candidate_id="c1", test_type="gre", section="quant", mode=Mode.tutor)

    with patch.object(ModelConfig, "prompt_version", return_value="v7"), \
         patch("services.agent.src.agent.agents.quant._load_prompt", return_value="STUB"), \
         patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_stream(["hint"])
        async for _ in agent.run(session, "Solve x^2-5x+6=0", trace_id="t1"):
            pass

    metadata = mock_llm.call_args.kwargs["metadata"]
    assert metadata["prompt_version"] == "v7"


# ── 5. AWAgent threads version into load ──────────────────────────────────────

@pytest.mark.asyncio
async def test_aw_agent_loads_configured_prompt_version():
    from services.agent.src.agent.agents.aw import AWAgent

    agent = AWAgent(model_config_path="config/model_config.dev.yaml")
    session = Session(candidate_id="c1", test_type="gre", section="aw", mode=Mode.tutor)

    with patch.object(ModelConfig, "prompt_version", return_value="v5"), \
         patch("services.agent.src.agent.agents.aw._load_prompt", return_value="STUB") as mock_load, \
         patch.object(agent, "_retrieve", new_callable=AsyncMock, return_value=[]), \
         patch.object(agent, "_check_citations", return_value=(True, [])), \
         patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_stream(["plan"])
        async for _ in agent.run(session, "Discuss technology"):
            pass

    mock_load.assert_called_with(Mode.tutor, "v5")
