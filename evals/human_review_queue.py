from __future__ import annotations

from typing import Any

from evals.judges.aw_judge import AWResult

_BOUNDARY_SCORES = {3.5, 4.5}


class HumanReviewQueue:
    """
    Phase 1: in-memory queue for items needing human review.
    Phase 2: backed by the `human_review_queue` Postgres table;
             gotohuman API client replaces the in-memory store.

    Trigger conditions:
      - AW score at a boundary tier (3.5, 4.5)
      - verifier_fail events
    """

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def maybe_add_aw(self, result: AWResult) -> None:
        if result.score in _BOUNDARY_SCORES:
            self._items.append({
                "item_id": result.item_id,
                "reason": "boundary_aw_score",
                "score": result.score,
                "rationale": result.rationale,
            })

    def add_verifier_fail(self, item_id: str, expression: str, trace_id: str) -> None:
        self._items.append({
            "item_id": item_id,
            "reason": "verifier_fail",
            "expression": expression,
            "trace_id": trace_id,
        })

    def list(self) -> list[dict[str, Any]]:
        return list(self._items)

    def count(self) -> int:
        return len(self._items)
