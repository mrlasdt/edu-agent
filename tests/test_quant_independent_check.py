"""
`wrong_but_verified` against the independent answer check.

The Quant solve turn used to compare the model's answer with itself, so a wrong answer that
parses passed (16 of 16 seeded wrong answers did). The independent check asks the model for the
problem as an expression, computes it with the solver and compares. These tests pin the gap
(the well-formed check alone) and the offline measurement of the check that closes it.

No model is called: the answers and expressions are seeded, the key is graded by sympy.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from evals import quant_report
from evals.agent_eval import cli, quant_suite, runner
from evals.agent_eval.results import (
    Control,
    ObservedWrong,
    QuantAttempt,
    QuantBaseline,
    SeededWrongAnswers,
    SelfCompare,
    WrongButVerified,
)
from evals.judges.quant_judge import grade_quant_item
from evals.loader import load_golden_set
from services.agent.src.agent.agents.quant import QuantAgent
from shared.src.shared.events import AGREES, IndependentCheck

SOLVE_ITEMS = [i for i in load_golden_set("quant") if i["kind"] == "solve"]
QUADRATIC = next(i for i in SOLVE_ITEMS if i["id"] == "q001")
COMPARISONS = [i["id"] for i in SOLVE_ITEMS if i["question_type"] == "quantitative_comparison"]


def _agent() -> QuantAgent:
    return QuantAgent(
        model_config_path="config/model_config.dev.yaml",
        verify_math=runner.in_process_verify_math(),
    )


def _row(kind: str, correct: bool, verdict: str | None) -> QuantAttempt:
    return QuantAttempt(
        item_id="x", kind=kind, tag="t", attempt=0, passed=correct, transcript=[], turns=[],
        checks={"correct": correct, "verifier_accepted": True}, independent_check=verdict,
    )


async def test_a_wrong_but_parseable_answer_passes_the_well_formed_check():
    agent = _agent()
    claimed = agent.extract_final_answer("Guess.\n**Answer: x = 5**") or ""

    graded = await grade_quant_item(QUADRATIC, candidate_answer=claimed)
    result = await agent.verify_well_formed("Guess.\n**Answer: x = 5**", claimed)

    assert graded.correct is False
    assert result.verified is True  # the well-formed check alone cannot tell


async def test_the_independent_check_disagrees_with_that_wrong_answer():
    agent = _agent()
    check = await agent.check_independently(quant_suite.stub_expression(QUADRATIC), "x = 5")
    assert check.verdict == "disagrees"
    assert check.solver == "2, 3"


async def test_the_well_formed_check_still_rejects_an_unparseable_answer():
    result = await _agent().verify_well_formed("", "the values are two and three")
    assert result.verified is False


async def test_every_seeded_wrong_answer_is_wrong_by_the_golden_key():
    for item in SOLVE_ITEMS:
        wrong = quant_suite.seeded_wrong_answer(item)
        assert wrong, item["id"]
        assert not (await grade_quant_item(item, candidate_answer=wrong)).correct, item["id"]
        right = await grade_quant_item(item, candidate_answer=item["ground_truth"])
        assert right.correct, item["id"]


async def test_the_suite_measures_no_seeded_wrong_answer_as_independently_verified():
    result = await quant_suite.wrong_but_verified(SOLVE_ITEMS, _agent(), [])

    assert result.seeded.n == len(SOLVE_ITEMS) == 16
    assert result.seeded.count == 0
    assert result.seeded.accepted_items == []
    # numeric answers are disagreed with; a comparison letter has nothing to compute
    assert result.seeded.disagrees == 14
    assert result.seeded.not_checkable == len(COMPARISONS) == 2
    # the "before": the well-formed check alone accepted every one
    assert result.self_compare.count == 16
    assert result.self_compare.accepted_items == sorted(i["id"] for i in SOLVE_ITEMS)


async def test_the_independent_check_does_not_reject_the_golden_key():
    """Control: a check that disagreed with everything would also score 0 on wrong answers."""
    result = await quant_suite.wrong_but_verified(SOLVE_ITEMS, _agent(), [])

    assert result.control.model_dump() == {
        "n": 16, "agrees": 14, "disagrees": 0, "not_checkable": 2,
    }


async def test_the_count_rises_if_the_check_agrees_with_a_wrong_answer():
    agent = _agent()
    agree = IndependentCheck(verdict=AGREES)
    with patch.object(agent, "check_independently", new=AsyncMock(return_value=agree)):
        result = await quant_suite.wrong_but_verified(SOLVE_ITEMS, agent, [])

    assert result.seeded.count == 16


async def test_observed_counts_real_attempts_that_were_wrong_and_agreed():
    rows = [
        _row("solve", correct=False, verdict="agrees"),
        _row("solve", correct=False, verdict="disagrees"),
        _row("solve", correct=False, verdict="not_checkable"),
        _row("solve", correct=True, verdict="agrees"),
        _row("tutor", correct=False, verdict="agrees"),
    ]
    result = await quant_suite.wrong_but_verified(SOLVE_ITEMS, _agent(), rows)

    assert result.observed.count == 1
    assert result.observed.n == 3  # the wrong solve attempts
    assert result.observed.not_checkable == 1
    assert result.observed.solve_attempts == 4


def test_the_offline_cli_run_reports_wrong_but_verified_and_the_report_renders_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], block_network: None
):
    argv = ["--mode", "offline", "--suite", "quant", "--samples", "1",
            "--ledger", str(tmp_path / "l.json"), "--out-dir", str(tmp_path)]
    assert cli.main(argv) == 0
    suite = json.loads(capsys.readouterr().out)["suites"]["quant"]

    assert suite["wrong_but_verified"]["seeded"]["count"] == 0
    assert suite["wrong_but_verified"]["seeded"]["n"] == 16
    assert suite["wrong_but_verified"]["self_compare"]["count"] == 16
    assert suite["wrong_but_verified"]["observed"]["count"] == 0  # the offline model is right
    assert any(c.startswith("wrong_but_verified: 0 of 16") for c in suite["caveats"])
    # one solve call plus the independent check's short call
    assert suite["groups"]["solve"]["cost_latency"]["llm_calls_per_turn_mean"] == 2.0


def test_the_quant_report_has_a_wrong_but_verified_section_when_the_run_measured_it():
    baseline = QuantBaseline.model_validate_json(Path("evals/baselines/quant.json").read_text())
    baseline.summary.wrong_but_verified = None  # an older baseline did not measure it
    assert "Wrong but verified" not in quant_report.render(baseline)

    baseline.summary.wrong_but_verified = WrongButVerified(
        definition="d",
        seeded=SeededWrongAnswers(
            value=0.0, successes=0, n=16, ci95=[0.0, 0.19], count=0, agrees=0, disagrees=14,
            not_checkable=2, accepted_items=[],
        ),
        self_compare=SelfCompare(
            value=1.0, successes=16, n=16, ci95=[0.8, 1.0], count=16, accepted_items=[],
        ),
        control=Control(n=16, agrees=14, disagrees=0, not_checkable=2),
        observed=ObservedWrong(
            value=0.07, successes=2, n=30, ci95=[0.0, 0.2], count=2, solve_attempts=30,
            not_checkable=0,
        ),
    )
    rendered = quant_report.render(baseline)
    assert "## Wrong but verified" in rendered
    assert "0/16" in rendered and "16/16 accepted" in rendered and "2/30" in rendered
