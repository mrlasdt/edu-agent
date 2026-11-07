"""
The independent answer check (services/agent/src/agent/independent_check.py) and its place in the
Quant solve turn.

No paid model is called: the model is a stub that answers from a script, and the numbers are
computed by the real math-verifier sandbox, in process.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from evals.agent_eval.runner import in_process_verify_skill
from services.agent.src.agent import independent_check as ic
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.capabilities.verify_math import (
    VERIFIER_UNREACHABLE,
    VerifyMath,
    VerifyMathResult,
)
from shared.src.shared.events import IndependentCheck, Verified
from shared.src.shared.models import Mode, Session
from shared.src.shared.turn_meter import TurnLog
from tests.agent_events import reply_of


@pytest.fixture
def skill() -> VerifyMath:
    return in_process_verify_skill()


# ── reading the model's reply ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("<expr>x^2 - 5x + 6 = 0</expr>", "x^2 - 5x + 6 = 0"),
        ("Sure.\n<EXPR> 3/4 +\n5/6 </EXPR>", "3/4 + 5/6"),
        ("<expr>1+1</expr> then <expr>2+2</expr>", "2+2"),
        ("<expr>NONE</expr>", None),
        ("<expr> none </expr>", None),
        ("<expr></expr>", None),
        ("no tag at all: 14", None),
        ("", None),
    ],
)
def test_parse_expression(reply: str, expected: str | None) -> None:
    assert ic.parse_expression(reply) == expected


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("14", True), ("2, -2", True), ("[1/3, sqrt(2)]", True), ("x = 4", True),
        ("x = 2 or x = 3", True), ("3.14*4", True), ("pi", True), ("2*pi/3", True),
        ("A", False), ("Quantity A is greater", False), ("x", False), ("[]", False),
        ("", False), ("True", False), ("1 + I", False), ("2*x + 1", False),
    ],
)
def test_is_number_list(text: str, ok: bool) -> None:
    assert ic.is_number_list(text) is ok


# ── comparing the claimed answer with the solver's ───────────────────────────


async def test_agrees_when_the_solver_gets_the_claimed_values(skill) -> None:
    check = await ic.compare(skill, "x^2 - 5x + 6 = 0", "3, 2")
    assert check == ic.IndependentCheck(ic.AGREES)


async def test_disagrees_and_names_what_the_solver_computed(skill) -> None:
    check = await ic.compare(skill, "x^2 - 5x + 6 = 0", "2, 4")
    assert (check.verdict, check.solver) == (ic.DISAGREES, "2, 3")
    assert check.event().public() == {"verdict": "disagrees", "solver": "2, 3"}


async def test_a_single_wrong_value_disagrees(skill) -> None:
    check = await ic.compare(skill, "17 * 23 * 19", "7430")
    assert (check.verdict, check.solver) == (ic.DISAGREES, "7429")


async def test_a_decimal_that_is_the_right_value_rounded_agrees(skill) -> None:
    assert (await ic.compare(skill, "10/3", "3.3")).verdict == ic.AGREES
    assert (await ic.compare(skill, "10/3", "3.33")).verdict == ic.AGREES
    assert (await ic.compare(skill, "10/3", "3.4")).verdict == ic.DISAGREES


async def test_a_rounded_decimal_for_an_equation_is_not_given_the_benefit(skill) -> None:
    assert (await ic.compare(skill, "3x = 10", "3.3")).verdict == ic.DISAGREES


@pytest.mark.parametrize("claimed", ["1,200", "1200", "1,200.0"])
async def test_a_thousands_separator_is_not_a_list(skill, claimed: str) -> None:
    assert (await ic.compare(skill, "6*200", claimed)).verdict == ic.AGREES


async def test_a_wrong_answer_with_a_thousands_separator_still_disagrees(skill) -> None:
    check = await ic.compare(skill, "6*200", "1,300")
    assert (check.verdict, check.solver) == (ic.DISAGREES, "1200")


async def test_a_comma_list_without_digit_groups_stays_a_list(skill) -> None:
    assert (await ic.compare(skill, "x^2 - 5x + 6 = 0", "2, 3")).verdict == ic.AGREES
    assert (await ic.compare(skill, "x^2 - 5x + 6 = 0", "2,3")).verdict == ic.AGREES


@pytest.mark.parametrize("claimed", ["2", "-2", "x = 2", "sqrt(4)"])
async def test_one_root_of_several_is_not_checkable_not_a_disagreement(
    skill, claimed: str
) -> None:
    check = await ic.compare(skill, "x^2 = 4", claimed)
    assert check.verdict == ic.NOT_CHECKABLE
    assert check.reason


async def test_a_subset_of_three_roots_is_not_checkable(skill) -> None:
    check = await ic.compare(skill, "(x - 1)*(x - 2)*(x - 3) = 0", "1, 3")
    assert check.verdict == ic.NOT_CHECKABLE


@pytest.mark.parametrize("claimed", ["3", "2, 3", "3, 2, 5", "4, 5"])
async def test_a_claimed_value_outside_the_roots_disagrees(skill, claimed: str) -> None:
    check = await ic.compare(skill, "x^2 = 4", claimed)
    assert (check.verdict, check.solver) == (ic.DISAGREES, "-2, 2")


async def test_a_wrong_single_value_against_a_single_root_disagrees(skill) -> None:
    check = await ic.compare(skill, "3x = 12", "5")
    assert (check.verdict, check.solver) == (ic.DISAGREES, "4")


@pytest.mark.parametrize(
    ("expression", "claimed", "why"),
    [
        (None, "14", "the model wrote NONE"),
        ("3^2 - 2^3", "A", "a comparison letter is not a number"),
        ("3^2 - 2^3", "Quantity A is greater", "prose is not a number"),
        ("x + y = 5", "3", "two unknowns: no single value to compare"),
        ("x^2 + 1 = 0", "1", "no real solution"),
        ("1 +", "2", "does not parse"),
        ("__import__('os').getpid()", "2", "not math at all"),
    ],
)
async def test_not_checkable_is_never_a_verdict_on_the_answer(
    skill, expression: str | None, claimed: str, why: str
) -> None:
    check = await ic.compare(skill, expression, claimed)
    assert check.verdict == ic.NOT_CHECKABLE, why
    assert check.reason


async def test_a_verifier_outage_is_not_checkable_not_a_disagreement() -> None:
    class Down(VerifyMath):
        async def verify(self, expression: str, expected: str) -> VerifyMathResult:
            return VerifyMathResult(verified=False, computed="", error=VERIFIER_UNREACHABLE)

    check = await ic.compare(Down(), "2 + 2", "5")
    assert check.verdict == ic.NOT_CHECKABLE


# ── in the solve turn: a stub model that answers wrong ───────────────────────

PROBLEM = "Solve x^2 - 5x + 6 = 0"
EQUATION = "<expr>x^2 - 5x + 6 = 0</expr>"
WRONG = "1. Guess.\n**Answer: 2, 4**"
RIGHT = "1. Factor.\n**Answer: 2, 3**"


class ScriptedModel:
    """Stands in for litellm.acompletion: solve calls read `solves` in turn, the check `expr`."""

    def __init__(self, solves: list[str], expr: str = EQUATION) -> None:
        self.solves = list(solves)
        self.expr = expr
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kw: Any) -> AsyncIterator[Any]:
        kind = "formalize" if "<expr>" in kw["messages"][0]["content"] else "solve"
        self.calls.append({"kind": kind, "model": kw["model"], "messages": kw["messages"]})
        text = self.expr if kind == "formalize" else self.solves.pop(0)

        async def stream() -> AsyncIterator[Any]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

        return stream()

    @property
    def kinds(self) -> list[str]:
        return [c["kind"] for c in self.calls]


async def _solve(
    monkeypatch: pytest.MonkeyPatch, model: ScriptedModel, tmp_path: Any
) -> tuple[list[Any], dict[str, Any]]:
    log = TurnLog(tmp_path / "turns.jsonl")
    monkeypatch.setattr("litellm.acompletion", model)
    monkeypatch.setattr("shared.src.shared.turn_meter.default_turn_log", lambda: log)
    agent = QuantAgent(
        model_config_path="config/model_config.dev.yaml",
        verify_math_skill=in_process_verify_skill(),
    )
    session = Session(candidate_id="c", test_type="gre", section="quant", mode=Mode.solve)
    out = [o async for o in agent.run(session, PROBLEM, trace_id="t-1")]
    (record,) = log.read()
    return out, record


async def test_a_wrong_model_is_caught_retried_escalated_and_degraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The model always answers 2, 4. The well-formed check passes every time; only the solver
    notices. The existing ladder runs: retry once, escalate once, then degrade."""
    model = ScriptedModel([WRONG, WRONG, WRONG])

    out, record = await _solve(monkeypatch, model, tmp_path)

    # attempt 1, the check's one short call, attempt 2, the escalation attempt: 3 solves + 1
    assert model.kinds == ["solve", "formalize", "solve", "solve"]
    assert out[-2] == IndependentCheck(verdict="disagrees", solver="2, 3")
    assert out[-1] == Verified(passed=False)
    assert record["verified_independent"] == "disagrees"
    assert record["independent_solver"] == "2, 3"
    assert record["llm_calls"] == 4


async def test_the_retry_is_told_what_the_solver_computed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    model = ScriptedModel([WRONG, WRONG, WRONG])
    await _solve(monkeypatch, model, tmp_path)

    retry = model.calls[2]["messages"][-1]
    assert retry["role"] == "assistant"
    assert "independent calculation" in retry["content"] and "2, 3" in retry["content"]


async def test_the_model_that_fixes_its_answer_on_the_retry_agrees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    model = ScriptedModel([WRONG, RIGHT])

    out, record = await _solve(monkeypatch, model, tmp_path)

    assert model.kinds == ["solve", "formalize", "solve"]  # the expression is asked for once
    assert reply_of(out) == RIGHT  # the wrong draft is not shown
    assert out[-2:] == [IndependentCheck(verdict="agrees"), Verified(passed=True)]
    assert record["verified_independent"] == "agrees"


async def test_the_check_never_sees_the_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    model = ScriptedModel([WRONG, WRONG, WRONG])
    await _solve(monkeypatch, model, tmp_path)

    sent = str(model.calls[1]["messages"])
    assert PROBLEM in sent
    assert "2, 4" not in sent and "Answer" not in sent.replace("answer", "")


async def test_a_problem_with_no_numeric_answer_is_shown_not_checkable_and_not_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    model = ScriptedModel(["Compare.\n**Answer: B**"], expr="<expr>NONE</expr>")

    out, record = await _solve(monkeypatch, model, tmp_path)

    assert model.kinds == ["solve", "formalize"]
    assert out[-2:] == [IndependentCheck(verdict="not_checkable"), Verified(passed=True)]
    assert record["verified_independent"] == "not_checkable"


async def test_a_failing_check_call_does_not_fail_the_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    class Flaky(ScriptedModel):
        async def __call__(self, **kw: Any) -> AsyncIterator[Any]:
            if "<expr>" in kw["messages"][0]["content"]:
                raise RuntimeError("provider down")
            return await super().__call__(**kw)

    out, record = await _solve(monkeypatch, Flaky([RIGHT]), tmp_path)

    assert out[-1] == Verified(passed=True)
    assert record["verified_independent"] == "not_checkable"


async def test_an_unparseable_answer_skips_the_extra_model_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    model = ScriptedModel(["x\n**Answer: the values are two and three**"] * 3)

    out, record = await _solve(monkeypatch, model, tmp_path)

    assert "formalize" not in model.kinds  # nothing well-formed to check
    assert out[-1] == Verified(passed=False)
    assert record["llm_calls"] == 3


async def test_a_tutor_turn_makes_no_check_and_records_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    model = ScriptedModel([])
    log = TurnLog(tmp_path / "turns.jsonl")
    monkeypatch.setattr("litellm.acompletion", model)
    monkeypatch.setattr("shared.src.shared.turn_meter.default_turn_log", lambda: log)
    agent = QuantAgent(model_config_path="config/model_config.dev.yaml")
    session = Session(candidate_id="c", test_type="gre", section="quant", mode=Mode.tutor)
    model.solves.append("What is the first step?")
    _ = [o async for o in agent.run(session, PROBLEM, trace_id="t-2")]

    assert model.kinds == ["solve"]
    assert "verified_independent" not in log.read()[0]
