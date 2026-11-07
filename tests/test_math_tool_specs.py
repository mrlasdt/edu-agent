"""
One source for the math tools' contract, and skills that reach the real routes.

Behaviors under test:
  1. the agent's tool specs and the verifier's tools come from one module, so a changed
     parameter is one edit (the two used to be hand-copied and could drift)
  2. every tool function takes exactly the arguments its spec lists, required or optional
  3. both skills take an `httpx` client, so a test drives the real FastAPI/Starlette routes
     through `httpx.ASGITransport` instead of subclassing a skill and skipping its `__init__`
"""

import inspect

import httpx
import pytest

from services.agent.src.agent.skills import math_tools
from services.agent.src.agent.skills.math_tools import UNAVAILABLE, MathToolsSkill
from services.agent.src.agent.skills.verify_math import VerifyMathSkill
from services.math_verifier.src.math_verifier import tools
from services.math_verifier.src.math_verifier.server import mcp
from shared.src.shared import math_tool_specs


@pytest.fixture
def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mcp.sse_app()), base_url="http://math-verifier"
    )


# ── 1 and 2. one source ──────────────────────────────────────────────────────


def test_the_agent_and_the_verifier_read_the_same_specs() -> None:
    assert math_tools.MATH_TOOL_SPECS is math_tool_specs.MATH_TOOL_SPECS
    assert math_tools.TOOL_NAMES == math_tool_specs.TOOL_NAMES
    assert tuple(tools.TOOLS) == math_tool_specs.TOOL_NAMES


def test_the_verifier_checks_a_call_against_the_spec() -> None:
    for name in math_tool_specs.TOOL_NAMES:
        assert tools.REQUIRED[name] == math_tool_specs.required_arguments(name)
        assert tools.OPTIONAL.get(name, ()) == math_tool_specs.optional_arguments(name)


def test_every_tool_function_takes_the_arguments_its_spec_lists() -> None:
    for spec in math_tool_specs.MATH_TOOL_SPECS:
        function = spec["function"]
        parameters = inspect.signature(tools.TOOLS[function["name"]]).parameters
        assert set(parameters) == set(function["parameters"]["properties"]), function["name"]
        required = {name for name, p in parameters.items() if p.default is inspect.Parameter.empty}
        assert required == set(function["parameters"]["required"]), function["name"]


# ── 3. a client reaches the real routes ──────────────────────────────────────


async def test_the_verify_skill_reaches_the_real_route(client: httpx.AsyncClient) -> None:
    skill = VerifyMathSkill(client=client)
    right = await skill.verify("x**2 - 5*x + 6 = 0", "x=2 or x=3")
    wrong = await skill.verify("x**2 - 5*x + 6 = 0", "x=1")
    assert right.verified is True and right.error is None
    assert wrong.verified is False


async def test_the_math_tools_skill_reaches_the_real_route(client: httpx.AsyncClient) -> None:
    skill = MathToolsSkill(client=client)
    assert (await skill.call("math_solve", {"equation": "2x + 3 = 11"}))["roots"] == ["4"]
    assert (await skill.call("math_compare", {"quantity_a": "3^2", "quantity_b": "2^3"}))[
        "verdict"
    ] == "A"


async def test_a_bad_call_comes_back_as_the_services_error_for_the_model(
    client: httpx.AsyncClient,
) -> None:
    result = await MathToolsSkill(client=client).call("math_evaluate", {"expression": "15% of 80"})
    assert "15/100" in result["error"]


async def test_an_unknown_tool_is_the_unavailable_message(client: httpx.AsyncClient) -> None:
    assert await MathToolsSkill(client=client).call("math_integrate", {}) == {"error": UNAVAILABLE}


async def test_a_down_service_degrades_through_an_injected_client() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    down = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    assert (await VerifyMathSkill(client=down).verify("2", "2")).error == "verifier unreachable"
    assert await MathToolsSkill(client=down).call("math_evaluate", {"expression": "1"}) == {
        "error": UNAVAILABLE
    }
