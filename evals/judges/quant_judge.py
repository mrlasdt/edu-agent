from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import sympy as sp

from services.math_verifier.src.math_verifier.sandbox import verify_expression


def _qc_verdict(difference: str) -> str | None:
    """A/B/C from the sign of `Qty A - Qty B`. D (cannot be determined) is never computable."""
    try:
        value = sp.sympify(difference)
        return "A" if value > 0 else "B" if value < 0 else "C"
    except (sp.SympifyError, TypeError):
        return None


@dataclass
class QuantResult:
    item_id: str
    correct: bool
    computed: str
    error: str | None = None


async def grade_quant_item(
    item: dict[str, Any], candidate_answer: str | None = None
) -> QuantResult:
    """
    Grade a candidate answer to a Quant golden-set item using the sympy sandbox.
    No LLM call — purely deterministic.

    The candidate must come from outside the golden set (the agent under test, or an
    explicit `candidate_answer` on the item). It is never defaulted to the golden
    `ground_truth`: that would grade the answer key against itself. A missing candidate
    is graded incorrect.
    """
    item_id = item.get("id", "unknown")
    expression = item.get("expression", "")
    candidate = candidate_answer if candidate_answer is not None else item.get("candidate_answer")
    if candidate is None or not str(candidate).strip():
        return QuantResult(
            item_id=item_id, correct=False, computed="", error="no candidate answer"
        )

    if item.get("question_type") == "quantitative_comparison":
        # The expression is `Qty A - Qty B`; the answer is a verdict letter, not a value.
        computed = await verify_expression(expression, "")
        expected = _qc_verdict(computed.computed) or str(item.get("ground_truth", "")).upper()
        given = re.sub(r"[^A-D]", "", str(candidate).strip().upper())
        return QuantResult(
            item_id=item_id,
            correct=len(given) == 1 and given == expected,
            computed=expected,
            error=computed.error,
        )

    result = await verify_expression(expression, str(candidate))
    return QuantResult(
        item_id=item_id,
        correct=result.verified,
        computed=result.computed,
        error=result.error,
    )
