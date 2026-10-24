"""Final-answer extraction/normalisation and the verifier handoff it feeds."""
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from evals.agent_eval.runner import InProcessVerifySkill
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.answers import extract_final_answer, normalize_answer
from shared.src.shared.models import Mode, Session


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1. Work.\n**Answer: 14**", "14"),
        ("**Answer: 2, 3**", "2, 3"),
        ("steps\nFinal Answer: The values of x that satisfy the equation are 2 and -2.", "2, -2"),
        ("**Final Answer:** x = 2 or x = 3", "x = 2, x = 3"),
        ("Answer: \\frac{1}{2}", "(1)/(2)"),
        ("The answer is 14.", "14"),
        ("so\nQuantity A is greater than Quantity B.", "Quantity A is greater than Quantity B"),
    ],
)
def test_extract_and_normalize(text: str, expected: str) -> None:
    raw = extract_final_answer(text)
    assert raw is not None
    assert normalize_answer(raw) == expected


def test_extract_prefers_last_bold_answer_and_handles_empty() -> None:
    assert extract_final_answer("**Answer: 1**\nrevised\n**Answer: 2**") == "2"
    assert extract_final_answer("  \n") is None


def _stream(text: str) -> Any:
    async def gen() -> AsyncIterator[Any]:
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    return gen()


@pytest.mark.parametrize(
    "reply",
    [
        "1. Factor.\n**Answer: 2, 3**",
        "1. Factor.\nFinal Answer: The values of x are 2 and 3.",
        "1. Factor.\n**Answer: x = 2 or x = 3**",
    ],
)
async def test_correct_multi_value_answer_passes_verifier_first_try(
    monkeypatch: pytest.MonkeyPatch, reply: str
) -> None:
    """Regression: multi-value answers used to fail the verifier, forcing retry + escalation."""
    calls: list[str] = []

    async def fake(**kw: Any) -> Any:
        calls.append(kw["model"])
        return _stream(reply)

    monkeypatch.setattr("litellm.acompletion", fake)
    agent = QuantAgent(verify_math_skill=InProcessVerifySkill())
    session = Session(candidate_id="c", test_type="gre", section="quant", mode=Mode.solve)
    out = [o async for o in agent.run(session, "Solve x^2 - 5x + 6 = 0")]
    assert out[-1] == {"verifier_fail": False}
    assert len(calls) == 1
