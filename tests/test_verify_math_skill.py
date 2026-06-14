"""
Tests for the verify-math Skill in the Agent Service.

The Skill is an MCP client that calls the math-verifier MCP server.
Tests mock the MCP client — no live server needed.

Behaviors under test:
  1. Returns VerificationResult with verified=True on matching answer
  2. Returns VerificationResult with verified=False on wrong answer
  3. Returns VerificationResult with error on server error
  4. Propagates timeout from the server response
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from services.agent.src.agent.skills.verify_math import (
    VerifyMathSkill,
    VerifyMathResult,
)


def make_mcp_tool_result(verified: bool, computed: str, error=None) -> MagicMock:
    """Build a mock MCP tool call result."""
    import json
    mock_result = MagicMock()
    mock_result.content = [MagicMock()]
    mock_result.content[0].text = json.dumps({
        "verified": verified,
        "computed": computed,
        "error": error,
    })
    return mock_result


# ── 1. Verified = True ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_returns_verified_true():
    skill = VerifyMathSkill(server_url="http://localhost:8090")

    with patch.object(skill, "_call_mcp_tool", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"verified": True, "computed": "[2, 3]", "error": None}

        result = await skill.verify("x**2 - 5*x + 6 = 0", "2, 3")

    assert result.verified is True
    assert result.error is None


# ── 2. Verified = False ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_returns_verified_false():
    skill = VerifyMathSkill(server_url="http://localhost:8090")

    with patch.object(skill, "_call_mcp_tool", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"verified": False, "computed": "[2I, -2I]", "error": None}

        result = await skill.verify("x**2 + 1 = 0", "x=1")

    assert result.verified is False
    assert result.error is None


# ── 3. Server error propagated ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_propagates_server_error():
    skill = VerifyMathSkill(server_url="http://localhost:8090")

    with patch.object(skill, "_call_mcp_tool", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {
            "verified": False,
            "computed": "",
            "error": "parse error: invalid syntax",
        }

        result = await skill.verify("not valid $$", "5")

    assert result.verified is False
    assert result.error is not None
    assert "parse error" in result.error


# ── 4. Timeout propagated ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_propagates_timeout():
    skill = VerifyMathSkill(server_url="http://localhost:8090")

    with patch.object(skill, "_call_mcp_tool", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"verified": False, "computed": "", "error": "timeout"}

        result = await skill.verify("very_long_computation(x)", "0")

    assert result.verified is False
    assert result.error == "timeout"


# ── 5. MCP tool called with correct args ─────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_calls_tool_with_correct_args():
    skill = VerifyMathSkill(server_url="http://localhost:8090")

    with patch.object(skill, "_call_mcp_tool", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"verified": True, "computed": "14", "error": None}

        await skill.verify("2 + 3 * 4", "14")

    mock_call.assert_called_once_with(
        tool_name="verify_math",
        arguments={"expression": "2 + 3 * 4", "expected": "14"},
    )
