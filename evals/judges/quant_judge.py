from __future__ import annotations

from dataclasses import dataclass

from services.math_verifier.src.math_verifier.sandbox import verify_expression


@dataclass
class QuantResult:
    item_id: str
    correct: bool
    computed: str
    error: str | None = None


async def grade_quant_item(item: dict) -> QuantResult:
    """
    Grade a Quant golden-set item using the sympy sandbox.
    No LLM call — purely deterministic.
    """
    item_id = item.get("id", "unknown")
    expression = item.get("expression", "")
    ground_truth = item.get("ground_truth", "")
    candidate_answer = item.get("candidate_answer", ground_truth)

    result = await verify_expression(expression, candidate_answer)
    return QuantResult(
        item_id=item_id,
        correct=result.verified,
        computed=result.computed,
        error=result.error,
    )
