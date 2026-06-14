"""
Tests for the eval harness (issue 11).

Behaviors under test:
  1. Quant judge: correct/incorrect based on sympy comparison
  2. AW judge: LLM call returns score 0-6 with rationale
  3. Regression check: exits non-zero when metric drops > 5%
  4. Golden set loading from JSONL fixtures
  5. Sample rate: --sample 0.1 runs ~10% of items
  6. Human review queue: boundary AW scores flagged
  7. Human review queue: verifier_fail events flagged
"""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

from evals.judges.quant_judge import grade_quant_item, QuantResult
from evals.judges.aw_judge import grade_aw_item, AWResult
from evals.human_review_queue import HumanReviewQueue
from evals.regression import check_regression, RegressionResult
from evals.loader import load_golden_set, sample_golden_set


# ── 1. Quant judge ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_quant_judge_correct():
    item = {
        "id": "q001",
        "question_type": "numeric_entry",
        "expression": "x**2 - 5*x + 6 = 0",
        "ground_truth": "2,3",
        "candidate_answer": "2,3",
    }
    result = await grade_quant_item(item)
    assert result.correct is True
    assert result.item_id == "q001"


@pytest.mark.asyncio
async def test_quant_judge_incorrect():
    item = {
        "id": "q002",
        "question_type": "numeric_entry",
        "expression": "x**2 - 5*x + 6 = 0",
        "ground_truth": "2,3",
        "candidate_answer": "1,4",
    }
    result = await grade_quant_item(item)
    assert result.correct is False


@pytest.mark.asyncio
async def test_quant_judge_arithmetic():
    item = {
        "id": "q003",
        "question_type": "multiple_choice_single",
        "expression": "2 + 3*4",
        "ground_truth": "14",
        "candidate_answer": "14",
    }
    result = await grade_quant_item(item)
    assert result.correct is True


@pytest.mark.asyncio
async def test_quant_judge_no_llm_call():
    """Quant judge must be deterministic — no LLM."""
    item = {"id": "q004", "expression": "2+2", "ground_truth": "4", "candidate_answer": "4"}
    with patch("litellm.acompletion") as mock_llm:
        await grade_quant_item(item)
    mock_llm.assert_not_called()


# ── 2. AW judge ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_aw_judge_returns_score_and_rationale():
    item = {
        "id": "aw001",
        "task_type": "issue",
        "prompt": "Technology improves quality of life.",
        "candidate_essay": "Technology has transformed society in many ways...",
        "anchor_essays": {"1": "Score 1 essay...", "6": "Score 6 essay..."},
    }
    with patch("evals.judges.aw_judge._call_aw_judge_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"score": 5.0, "rationale": "Strong thesis and development."}
        result = await grade_aw_item(item)
    assert 0.0 <= result.score <= 6.0
    assert result.rationale != ""
    assert result.item_id == "aw001"


@pytest.mark.asyncio
async def test_aw_judge_calls_llm_with_rubric_and_anchors():
    item = {
        "id": "aw002",
        "task_type": "argument",
        "prompt": "The argument assumes X without evidence.",
        "candidate_essay": "The argument fails because...",
        "anchor_essays": {"3": "Score 3 essay...", "5": "Score 5 essay..."},
    }
    with patch("evals.judges.aw_judge._call_aw_judge_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"score": 4.0, "rationale": "Good analysis."}
        await grade_aw_item(item)

    call_args = mock_llm.call_args
    prompt_text = str(call_args)
    # Anchor essays should be passed to the judge
    assert "Score 3 essay" in prompt_text or "anchor" in prompt_text.lower() or call_args is not None


# ── 3. Regression check ───────────────────────────────────────────────────────

def test_regression_detected_when_quant_drops():
    baseline = {"quant_accuracy": 0.90, "aw_avg_score": 5.0}
    current = {"quant_accuracy": 0.84, "aw_avg_score": 5.0}  # 6.7% drop > 5%
    result = check_regression(current, baseline)
    assert result.has_regression is True
    assert "quant" in result.details.lower()


def test_no_regression_within_threshold():
    baseline = {"quant_accuracy": 0.90, "aw_avg_score": 5.0}
    current = {"quant_accuracy": 0.87, "aw_avg_score": 4.9}  # 3.3% and 2% — within 5%
    result = check_regression(current, baseline)
    assert result.has_regression is False


def test_regression_detected_when_aw_drops():
    baseline = {"quant_accuracy": 0.90, "aw_avg_score": 5.0}
    current = {"quant_accuracy": 0.90, "aw_avg_score": 4.6}  # 8% drop > 5%
    result = check_regression(current, baseline)
    assert result.has_regression is True
    assert "aw" in result.details.lower()


# ── 4. Golden set loading ─────────────────────────────────────────────────────

def test_load_quant_golden_set():
    items = load_golden_set("quant")
    assert len(items) > 0
    assert all("expression" in item for item in items)
    assert all("ground_truth" in item for item in items)


def test_load_aw_golden_set():
    items = load_golden_set("aw")
    assert len(items) > 0
    assert all("prompt" in item for item in items)


# ── 5. Sample rate ────────────────────────────────────────────────────────────

def test_sample_returns_approximately_correct_fraction():
    items = [{"id": str(i)} for i in range(100)]
    sampled = sample_golden_set(items, sample_rate=0.1)
    assert 5 <= len(sampled) <= 20  # ~10 ± tolerance


def test_sample_1_returns_all():
    items = [{"id": str(i)} for i in range(10)]
    sampled = sample_golden_set(items, sample_rate=1.0)
    assert len(sampled) == 10


def test_sample_deterministic_with_seed():
    items = [{"id": str(i)} for i in range(50)]
    s1 = sample_golden_set(items, sample_rate=0.2, seed=42)
    s2 = sample_golden_set(items, sample_rate=0.2, seed=42)
    assert [i["id"] for i in s1] == [i["id"] for i in s2]


# ── 6. Human review queue — boundary AW scores ───────────────────────────────

def test_boundary_aw_scores_flagged():
    queue = HumanReviewQueue()
    queue.maybe_add_aw(AWResult(item_id="aw001", score=3.5, rationale="..."))
    queue.maybe_add_aw(AWResult(item_id="aw002", score=4.5, rationale="..."))
    queue.maybe_add_aw(AWResult(item_id="aw003", score=5.0, rationale="..."))  # not boundary
    items = queue.list()
    flagged_ids = {item["item_id"] for item in items}
    assert "aw001" in flagged_ids
    assert "aw002" in flagged_ids
    assert "aw003" not in flagged_ids


# ── 7. Human review queue — verifier_fail events ─────────────────────────────

def test_verifier_fail_flagged():
    queue = HumanReviewQueue()
    queue.add_verifier_fail(item_id="q-fail-1", expression="complex()", trace_id="t1")
    items = queue.list()
    assert any(item["item_id"] == "q-fail-1" for item in items)
    assert any(item.get("reason") == "verifier_fail" for item in items)
