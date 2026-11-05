"""
`wrong_but_verified`: the Quant solve-turn check compares the answer with itself, so it accepts
a wrong answer that parses. These tests pin that gap and the offline measurement of it.

No model is called: the answers are seeded, the key is graded by sympy.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from evals import quant_report
from evals.agent_eval import cli, quant_suite, runner
from evals.judges.quant_judge import grade_quant_item
from evals.loader import load_golden_set
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.skills.verify_math import VerifyMathResult

SOLVE_ITEMS = [i for i in load_golden_set("quant") if i["kind"] == "solve"]
QUADRATIC = next(i for i in SOLVE_ITEMS if i["id"] == "q001")


def _agent() -> QuantAgent:
    return QuantAgent(
        model_config_path="config/model_config.dev.yaml",
        verify_math_skill=runner.InProcessVerifySkill(),
    )


def _row(kind: str, correct: bool, accepted: bool) -> dict:
    return {"kind": kind, "checks": {"correct": correct, "verifier_accepted": accepted}}


async def test_a_wrong_but_parseable_answer_passes_the_product_check():
    agent = _agent()
    claimed = agent._extract_final_answer("Guess.\n**Answer: x = 5**") or ""

    graded = await grade_quant_item(QUADRATIC, candidate_answer=claimed)
    result = await agent._verify("Guess.\n**Answer: x = 5**", claimed)

    assert graded.correct is False
    assert result.verified is True


async def test_the_product_check_still_rejects_an_unparseable_answer():
    result = await _agent()._verify("", "the values are two and three")
    assert result.verified is False


async def test_every_seeded_wrong_answer_is_wrong_by_the_golden_key():
    for item in SOLVE_ITEMS:
        wrong = quant_suite.seeded_wrong_answer(item)
        assert wrong, item["id"]
        assert not (await grade_quant_item(item, candidate_answer=wrong)).correct, item["id"]
        right = await grade_quant_item(item, candidate_answer=item["ground_truth"])
        assert right.correct, item["id"]


async def test_the_suite_measures_every_seeded_wrong_answer_as_verified():
    result = await quant_suite.wrong_but_verified(SOLVE_ITEMS, _agent(), [])

    assert result["seeded"]["n"] == len(SOLVE_ITEMS) == 16
    assert result["seeded"]["count"] == 16
    assert result["seeded"]["value"] == 1.0
    assert result["seeded"]["accepted_items"] == sorted(i["id"] for i in SOLVE_ITEMS)


async def test_the_count_falls_when_the_check_becomes_independent():
    agent = _agent()
    reject = VerifyMathResult(verified=False, computed="", error="does not match")
    with patch.object(agent, "_verify", new=AsyncMock(return_value=reject)):
        result = await quant_suite.wrong_but_verified(SOLVE_ITEMS, agent, [])

    assert result["seeded"]["count"] == 0
    assert result["seeded"]["n"] == 16


async def test_observed_counts_real_attempts_that_were_wrong_and_accepted():
    rows = [
        _row("solve", correct=False, accepted=True),
        _row("solve", correct=False, accepted=False),
        _row("solve", correct=True, accepted=True),
        _row("tutor", correct=False, accepted=True),
    ]
    result = await quant_suite.wrong_but_verified(SOLVE_ITEMS, _agent(), rows)

    assert result["observed"]["count"] == 1
    assert result["observed"]["n"] == 2  # the wrong solve attempts
    assert result["observed"]["solve_attempts"] == 3


def test_the_offline_cli_run_reports_wrong_but_verified_and_the_report_renders_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], block_network: None
):
    argv = ["--mode", "offline", "--suite", "quant", "--samples", "1",
            "--ledger", str(tmp_path / "l.json"), "--out-dir", str(tmp_path)]
    assert cli.main(argv) == 0
    suite = json.loads(capsys.readouterr().out)["suites"]["quant"]

    assert suite["wrong_but_verified"]["seeded"]["count"] == 16
    assert suite["wrong_but_verified"]["seeded"]["n"] == 16
    assert suite["wrong_but_verified"]["observed"]["count"] == 0  # the offline model is right
    assert any(c.startswith("wrong_but_verified: 16 of 16") for c in suite["caveats"])


def test_the_quant_report_has_a_wrong_but_verified_section_when_the_run_measured_it():
    baseline = json.loads(Path("evals/baselines/quant.json").read_text())
    assert "Wrong but verified" not in quant_report.render(baseline)  # an older baseline

    baseline["summary"]["wrong_but_verified"] = {
        "seeded": {"count": 16, "n": 16, "ci95": [0.81, 1.0]},
        "observed": {"count": 2, "solve_attempts": 30},
    }
    rendered = quant_report.render(baseline)
    assert "## Wrong but verified" in rendered
    assert "16/16" in rendered and "2/30" in rendered
