from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.math_verifier.src.math_verifier.sandbox import verify_expression


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

    result = await verify_expression(expression, str(candidate))
    return QuantResult(
        item_id=item_id,
        correct=result.verified,
        computed=result.computed,
        error=result.error,
    )
