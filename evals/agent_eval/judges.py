"""
Judge interface for the agent-in-the-loop eval.

A judge answers *typed* questions rather than free text, which is what lets a
decision model such as OpenRouter's jev sit behind the same interface as a chat
LLM: `yes_no` is a two-option choice and `score` is a bounded numeric score.

Implementations:
  LLMJudge       - any litellm chat model (free OpenRouter first, OpenAI fallback)
  JevJudge       - typesafe/jev-1.14 via the OpenRouter decisions API (paid, opt-in)
  HeuristicJudge - deterministic, network-free; used by --offline and unit tests
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import litellm

from evals.agent_eval.budget import estimate_cost, estimate_tokens
from evals.agent_eval.cache import cache_key
from evals.agent_eval.transport import MeteredTransport, tagged

JEV_MODEL = "typesafe/jev-1.14"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"


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

    async def score(self, q: JudgeQuestion, criteria: list[str]) -> float: ...


class LLMJudge:
    def __init__(self, model: str) -> None:
        self.name = f"llm:{model}"
        self._model = model

    async def _ask(self, system: str, q: JudgeQuestion) -> str:
        with tagged("judge"):
            response = await litellm.acompletion(
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
        scale = "\n".join(f"{i}: {c}" for i, c in enumerate(criteria))
        text = await self._ask(
            f"You are a strict grader. Score on this scale:\n{scale}\n"
            "Reply with only a number (decimals allowed).",
            q,
        )
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            raise JudgeError(f"unparseable score: {text[:60]!r}")
        return min(float(len(criteria) - 1), max(0.0, float(match.group(0))))


class JevJudge:
    """
    typesafe/jev-1.14 through the OpenRouter decisions API. Paid (input tokens only), off by
    default, and never used offline. yes/no is a `choice` question with options yes/no; score
    is a `score` question whose criteria are the scale descriptors. Jev returns no explanation.
    """

    def __init__(self, api_key: str, transport: MeteredTransport) -> None:
        self.name = f"jev:{JEV_MODEL}"
        self._key = api_key
        self._transport = transport

    async def _decide(self, q: JudgeQuestion, question: dict[str, Any]) -> dict[str, Any]:
        state = q.material
        cache = self._transport.cache
        key = cache_key(model=JEV_MODEL, state=state, question=question)
        cached = cache.get(key) if cache else None
        if cached is not None:
            body: dict[str, Any] = cached
        else:
            # jev bills input tokens only, so the projection is exact up to the token estimate
            self._transport.ledger.ensure_capacity(
                estimate_jev_cost(len(q.instructions) + len(state))
            )
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    JEV_URL,
                    headers={"Authorization": f"Bearer {self._key}"},
                    json={"model": JEV_MODEL, "state": state, "questions": {"q": question}},
                )
                resp.raise_for_status()
                body = resp.json()
            usage = body.get("usage", {})
            self._transport.record_external(
                "judge",
                JEV_MODEL,
                int(usage.get("input_tokens") or estimate_tokens(q.instructions + state)),
                int(usage.get("output_tokens", 0)),
                float(usage.get("cost", 0.0)),
                time.perf_counter() - start,
            )
            if cache:
                cache.put(key, body)
        answers = body.get("answers", body)
        answer = answers.get("q") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise JudgeError("jev response has no answer for question 'q'")
        return answer

    async def yes_no(self, q: JudgeQuestion) -> YesNo:
        answer = await self._decide(
            q,
            {
                "type": "choice",
                "instructions": q.instructions,
                "criteria": {"yes": "The statement is true.", "no": "The statement is false."},
            },
        )
        if answer.get("choice") not in ("yes", "no"):
            raise JudgeError(f"unexpected jev choice: {answer.get('choice')!r}")
        p_yes = answer.get("probabilities", {}).get("yes")
        return YesNo(answer=answer["choice"] == "yes", p_yes=p_yes)

    async def score(self, q: JudgeQuestion, criteria: list[str]) -> float:
        answer = await self._decide(
            q, {"type": "score", "instructions": q.instructions, "criteria": criteria}
        )
        if "score" not in answer:
            raise JudgeError("jev score answer missing 'score'")
        return float(answer["score"])


def estimate_jev_cost(chars: int) -> float:
    return estimate_cost(JEV_MODEL, estimate_tokens("x" * chars), 0)


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
