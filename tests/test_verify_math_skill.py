"""
Tests for the verify-math Skill in the Agent Service.

The Skill posts to the math-verifier service's verify_math route. Tests hand it an httpx client
whose transport answers in process, so the skill's own request code runs and no live server is
needed.

Behaviors under test:
  1. Returns VerifyMathResult with verified=True on matching answer
  2. Returns VerifyMathResult with verified=False on wrong answer
  3. Returns VerifyMathResult with error on server error
  4. Propagates timeout from the server response
  5. Posts the expression and expected answer to /tools/verify_math
  6. A verifier outage degrades instead of raising
"""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from services.agent.src.agent.skills.verify_math import (
    VerifyMathSkill,
)
from shared.src.shared.events import Verified
from tests.agent_events import reply_of


def skill_answering(
    reply: dict[str, Any] | Exception,
) -> tuple[VerifyMathSkill, list[httpx.Request]]:
    """A skill whose service replies `reply` (or raises it), and the requests it received."""
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if isinstance(reply, Exception):
            raise reply
        return httpx.Response(200, json=reply)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return VerifyMathSkill(server_url="http://localhost:8090", client=client), requests


# ── 1. Verified = True ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_returns_verified_true():
    skill, _ = skill_answering({"verified": True, "computed": "[2, 3]", "error": None})

    result = await skill.verify("x**2 - 5*x + 6 = 0", "2, 3")

    assert result.verified is True
    assert result.error is None


# ── 2. Verified = False ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_returns_verified_false():
    skill, _ = skill_answering({"verified": False, "computed": "[2I, -2I]", "error": None})

    result = await skill.verify("x**2 + 1 = 0", "x=1")

    assert result.verified is False
    assert result.error is None


# ── 3. Server error propagated ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_propagates_server_error():
    skill, _ = skill_answering(
        {"verified": False, "computed": "", "error": "parse error: invalid syntax"}
    )

    result = await skill.verify("not valid $$", "5")

    assert result.verified is False
    assert result.error is not None
    assert "parse error" in result.error


# ── 4. Timeout propagated ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_skill_propagates_timeout():
    skill, _ = skill_answering({"verified": False, "computed": "", "error": "timeout"})

    result = await skill.verify("very_long_computation(x)", "0")

    assert result.verified is False
    assert result.error == "timeout"


# ── 5. Tool route called with correct args ───────────────────────────────────

@pytest.mark.asyncio
async def test_skill_calls_tool_with_correct_args():
    skill, requests = skill_answering({"verified": True, "computed": "14", "error": None})

    await skill.verify("2 + 3 * 4", "14")

    (request,) = requests
    assert request.method == "POST"
    assert str(request.url) == "http://localhost:8090/tools/verify_math"
    assert json.loads(request.content) == {"expression": "2 + 3 * 4", "expected": "14"}


# ── 6. Verifier outage degrades instead of raising ────────────────────────────

@pytest.mark.asyncio
async def test_unreachable_verifier_returns_an_error_instead_of_raising():
    # Nothing listens on port 9, so this is a real ConnectError, not a mocked one.
    skill = VerifyMathSkill(server_url="http://127.0.0.1:9")

    result = await skill.verify("2, 3", "2, 3")

    assert result.verified is False
    assert result.error == "verifier unreachable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("All connection attempts failed"),
        httpx.ReadTimeout("timed out"),
        httpx.HTTPStatusError(
            "502", request=httpx.Request("POST", "http://v/tools/x"),
            response=httpx.Response(502),
        ),
    ],
)
async def test_any_http_error_from_the_verifier_is_reported_not_raised(exc):
    skill, _ = skill_answering(exc)

    result = await skill.verify("2 + 2", "4")

    assert result.verified is False
    assert result.error == "verifier unreachable"


@pytest.mark.asyncio
async def test_quant_solve_turn_degrades_when_the_verifier_is_down():
    """ADR-0011: an unreachable verifier degrades to 'unable to verify', the turn still answers."""
    from services.agent.src.agent.agents.quant import QuantAgent
    from shared.src.shared.models import Mode, Session

    reply = "Factor it.\n**Answer: 2, 3**"

    async def respond(*args, **kwargs):
        async def stream():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=reply))])

        return stream()

    agent = QuantAgent(
        model_config_path="config/model_config.dev.yaml",
        verify_math_skill=VerifyMathSkill(server_url="http://127.0.0.1:9"),
    )
    session = Session(candidate_id="c1", test_type="gre", section="quant", mode=Mode.solve)
    with patch("litellm.acompletion", new=AsyncMock(side_effect=respond)):
        items = [i async for i in agent.run(session, "Solve x^2 - 5x + 6 = 0")]

    assert "Factor it." in reply_of(items)
    assert Verified(passed=False) in items
