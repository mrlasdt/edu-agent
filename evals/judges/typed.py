"""
Judge interface for the agent-in-the-loop eval.

A judge answers *typed* questions rather than free text: `yes_no` is a two-option choice
and `score` is a bounded numeric score.

Implementations:
  LLMJudge       - any litellm chat model, through the Router (so the budget and the ledger apply)
  HeuristicJudge - deterministic, network-free; used by --offline and unit tests
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from evals.agent_eval.transport import tagged
from shared.src.shared.llm_router import get_router


class JudgeError(RuntimeError):
    """The judge returned something that could not be parsed into the typed answer."""


@dataclass
class JudgeQuestion:
    """`instructions` + `material` are what real judges see; `facts` is for offline fakes only."""

    instructions: str
    material: str
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class YesNo:
    answer: bool
    p_yes: float | None = None


class Judge(Protocol):
    name: str

    async def yes_no(self, q: JudgeQuestion) -> YesNo: ...

    async def score(self, q: JudgeQuestion, criteria: list[str]) -> float:
        """Position on `criteria` counted from 1: 1 is the first descriptor, N the last."""
        ...


class LLMJudge:
    def __init__(self, model: str) -> None:
        self.name = f"llm:{model}"
        self._model = model

    async def _ask(self, system: str, q: JudgeQuestion) -> str:
        # models occasionally reply with nothing; one retry recovers it, a second is a real failure
        text = await self._complete(system, q)
        return text if text.strip() else await self._complete(system, q)

    async def _complete(self, system: str, q: JudgeQuestion) -> str:
        with tagged("judge"):
            response = await get_router().acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{q.instructions}\n\n{q.material}"},
                ],
                temperature=0,
                stream=False,
            )
        return str(response.choices[0].message.content or "")

    async def yes_no(self, q: JudgeQuestion) -> YesNo:
        text = await self._ask(
            "You are a strict grader. Answer with exactly one word: YES or NO.", q
        )
        match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
        if not match:
            raise JudgeError(f"unparseable yes/no answer: {text[:60]!r}")
        return YesNo(answer=match.group(1).lower() == "yes")

    async def score(self, q: JudgeQuestion, criteria: list[str]) -> float:
        scale = "\n".join(f"{i}: {c}" for i, c in enumerate(criteria, start=1))
        text = await self._ask(
            f"You are a strict grader. Score on this scale:\n{scale}\n"
            "Reply with only a number (decimals allowed).",
            q,
        )
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            raise JudgeError(f"unparseable score: {text[:60]!r}")
        return min(float(len(criteria)), max(1.0, float(match.group(0))))


class HeuristicJudge:
    """Network-free stand-in: delegates to callables over `JudgeQuestion.facts`."""

    name = "heuristic"

    def __init__(
        self,
        yes_no: Callable[[JudgeQuestion], bool],
        score: Callable[[JudgeQuestion, list[str]], float],
    ) -> None:
        self._yes_no = yes_no
        self._score = score

    async def yes_no(self, q: JudgeQuestion) -> YesNo:
        return YesNo(answer=self._yes_no(q))

    async def score(self, q: JudgeQuestion, criteria: list[str]) -> float:
        return self._score(q, criteria)
