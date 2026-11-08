"""
Tests for QuantAgent Solve mode — verifier integration, retry, escalation, degradation.

Behaviors under test (from issue 05 acceptance criteria):
  1. Solve turn calls verify-math capability; verified answer emitted normally
  2. On verifier disagreement: agent retries (once) with disagreement as context
  3. After 2 retries still failing: escalates to escalation model
  4. After escalation failure: degrades gracefully with a failed Verified event
  5. Mode resets to tutor at start of next question
  6. Tutor mode does NOT call verifier
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.capabilities.verify_math import VerifyMathResult
from shared.src.shared.events import Verified
from shared.src.shared.models import Mode, Session
from tests.agent_events import texts


def make_session(mode: Mode = Mode.solve) -> Session:
    return Session(candidate_id="c1", test_type="gre", section="quant", mode=mode)


def mock_stream(tokens: list[str]):
    """Build an async iterator mimicking litellm streaming with a final answer marker."""
    async def _stream():
        for token in tokens:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            yield chunk
    return _stream()


# ── 1. Solve with verified answer ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_solve_emits_tokens_when_verified():
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent, "extract_final_answer", return_value="x=2 or x=3"), \
         patch.object(agent, "verify_well_formed", new_callable=AsyncMock) as mock_verify:

        mock_llm.return_value = mock_stream(["Step 1", " ... ", "**Answer: x=2 or x=3**"])
        mock_verify.return_value = VerifyMathResult(verified=True, computed="[2,3]")

        items = [i async for i in agent.run(make_session(Mode.solve), "Solve x^2-5x+6=0")]

    assert texts(items)  # tokens were emitted
    assert Verified(passed=False) not in items


# ── 2. Retry on verifier disagreement ────────────────────────────────────────

@pytest.mark.asyncio
async def test_solve_retries_on_verifier_mismatch():
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")

    call_count = 0
    async def llm_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return mock_stream(["Answer: x=1"])  # always wrong

    with patch("litellm.acompletion", side_effect=llm_side_effect), \
         patch.object(agent, "extract_final_answer", return_value="x=1"), \
         patch.object(agent, "verify_well_formed", new_callable=AsyncMock) as mock_verify:

        # First two verifications fail, third (escalation) also fails
        mock_verify.return_value = VerifyMathResult(verified=False, computed="[2,3]")

        async for _ in agent.run(make_session(Mode.solve), "Solve x^2-5x+6=0"):
            pass

    # Should have called LLM at least twice (initial + 1 retry before escalation)
    assert call_count >= 2


# ── 3. Escalation after max retries ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_solve_escalates_to_stronger_model_after_retries(tmp_path):
    # The dev config runs every role on one model; give escalation its own to observe it.
    raw = yaml.safe_load(Path("config/model_config.dev.yaml").read_text())
    raw["roles"]["math_escalation"]["model"] = "o1"
    config_path = tmp_path / "model_config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    agent = QuantAgent(model_config_path=str(config_path))

    models_used = []

    async def llm_side_effect(*args, **kwargs):
        models_used.append(kwargs.get("model", ""))
        return mock_stream(["Answer: x=1"])

    with patch("litellm.acompletion", side_effect=llm_side_effect), \
         patch.object(agent, "extract_final_answer", return_value="x=1"), \
         patch.object(agent, "verify_well_formed", new_callable=AsyncMock) as mock_verify:

        mock_verify.return_value = VerifyMathResult(verified=False, computed="[2,3]")

        async for _ in agent.run(make_session(Mode.solve), "Solve x^2-5x+6=0"):
            pass

    escalation_model = "openai/o1"
    assert any(escalation_model in m for m in models_used), \
        f"Expected escalation to {escalation_model}, models used: {models_used}"


# ── 4. Graceful degradation after escalation failure ─────────────────────────

@pytest.mark.asyncio
async def test_solve_degrades_gracefully_after_full_pipeline_failure():
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent, "extract_final_answer", return_value="x=1"), \
         patch.object(agent, "verify_well_formed", new_callable=AsyncMock) as mock_verify:

        mock_llm.return_value = mock_stream(["My best reasoning..."])
        mock_verify.return_value = VerifyMathResult(verified=False, computed="[2,3]")

        items = [i async for i in agent.run(make_session(Mode.solve), "Solve x^2-5x+6=0")]

    assert Verified(passed=False) in items


# ── 5. Tutor mode does NOT call verifier ─────────────────────────────────────

@pytest.mark.asyncio
async def test_tutor_mode_does_not_call_verifier():
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm, \
         patch.object(agent, "verify_well_formed", new_callable=AsyncMock) as mock_verify:

        mock_llm.return_value = mock_stream(["Think about it..."])

        async for _ in agent.run(make_session(Mode.tutor), "Help me with x^2-5x+6"):
            pass

    mock_verify.assert_not_called()
