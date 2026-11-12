"""
The run mode, chosen once.

A run is either offline (a fake model, no network, no key) or live (a real model through the
Router). Everything that differs between the two lives in a `Backend`: which model answers, the
judge, the IELTS marker and grader, the response cache and the key check. `Backend.for_mode`
picks one; the CLI and the suites then call it and never ask which mode they are in. (A dry run
makes no call, so it has no backend: it only prints the plan.)
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import litellm

from evals.agent_eval import ielts_writing, offline_generation, quant_suite
from evals.agent_eval.cache import ResponseCache
from evals.agent_eval.transport import FakeToolCalls, MeteredTransport
from evals.judges.typed import HeuristicJudge, Judge, JudgeQuestion, LLMJudge
from evals.loader import load_golden_set
from services.gateway.src.gateway.ielts import llm
from shared.src.shared.budget import Ledger

OFFLINE_MODEL = "offline/fake"


def offline_model(model: str, messages: list[dict[str, Any]]) -> str | FakeToolCalls:
    """
    Deterministic fake agent: tutor replies with a hint, solver with the golden answer, and
    the question-generation roles with seeded flaws (see offline_generation).
    """
    generated = offline_generation.respond(model, messages)
    if generated is not None:
        return generated
    if "ielts writing coach" in str(messages[0]["content"]).lower():
        return ielts_coach_reply(messages)
    system = str(messages[0]["content"]).lower()
    convo = "\n".join(str(m["content"]) for m in messages[1:])
    item = next(
        (i for i in load_golden_set("quant") if i.get("question") and i["question"] in convo), None
    )
    if item is None:
        return "What is the first quantity you could compute from the information given?"
    if "<expr>" in system:
        # the independent check's short call: the problem as an expression (NONE for a comparison)
        return f"<expr>{quant_suite.stub_expression(item) or 'NONE'}</expr>"
    asked_for_a_tool = any(m.get("role") == "tool" for m in messages)
    if "math_evaluate" in system and item.get("expression") and not asked_for_a_tool:
        # the tools prompt is in play: do what a careful model does, compute before answering
        equation = "=" in item["expression"]
        name = "math_solve" if equation else "math_evaluate"
        argument = "equation" if equation else "expression"
        return FakeToolCalls([(name, {argument: item["expression"]})])
    if "tutor" in system:
        return (
            f"Let's stay with the problem: {item['question']} "
            "What is the first quantity you could work out from it?"
        )
    return f"1. Compute it.\n**Answer: {item['ground_truth']}**"


def ielts_coach_reply(messages: list[dict[str, Any]]) -> str:
    """The offline coach: quotes the first quote of the report it was given and sets one drill. It
    follows no instruction in the report and writes no essay, so it is the shape the suite checks
    for, not a measure of any model."""
    last = str(messages[-1]["content"])
    found = re.search(r'- (?:supports|weakens), paragraph \d+: "([^"]+)"', last)
    quote = f' Your own words, "{found.group(1)}", show it.' if found else ""
    return (
        f"The report says your linking is mechanical.{quote} Drill (ten minutes): rewrite that "
        "sentence without its opening linker, then link it to the one before by reference. "
        "Which word could you drop?"
    )


def offline_judge() -> Judge:
    """A judge that reads the gold label off the question: plumbing only, it judges nothing."""

    def yes_no(q: JudgeQuestion) -> bool:
        return bool(q.facts["gold"])

    def score(q: JudgeQuestion, criteria: list[str]) -> float:
        return float(q.facts["gold"])

    return HeuristicJudge(yes_no, score)


def missing_keys(models: list[str]) -> list[str]:
    """
    What `models` still need from the environment, one entry per key. The provider-to-key table
    is litellm's (`validate_environment`), so a `gemini/...` model asks for Google's key and not
    OpenRouter's. A provider that accepts either of two names is one entry, "A or B", and none
    once either is set.
    """
    missing: list[str] = []
    for model in models:
        names = litellm.validate_environment(model)["missing_keys"]
        need = " or ".join(names)
        if need and need not in missing:
            missing.append(need)
    return missing


class Backend(ABC):
    """What a run needs from its mode. `model` answers every role; `report_chain` is what the
    report names; `cache` is the response cache in use, if any."""

    live: bool
    model: str
    report_chain: list[str]
    cache: ResponseCache | None = None

    @staticmethod
    def for_mode(
        mode: str, *, chain: list[str], judge_model: str | None, cache_dir: Path | None
    ) -> Backend:
        """The backend of an `offline` or `live` run (`cache_dir` None means no cache)."""
        if mode == "offline":
            return OfflineBackend(chain)
        if mode == "live":
            return LiveBackend(chain, judge_model, cache_dir)
        raise ValueError(f"a {mode!r} run has no backend: only offline and live runs call a model")

    def preflight(self) -> None:
        """Refuse to start a run that cannot work (a live run without its keys)."""

    @abstractmethod
    def transport(self, ledger: Ledger, run_id: str, max_tokens: int) -> MeteredTransport: ...

    @abstractmethod
    def judge(self) -> Judge: ...

    @abstractmethod
    def ielts_completer(self) -> llm.Completer: ...

    @abstractmethod
    def ielts_grader(self, samples: int) -> ielts_writing.IeltsGrader: ...


class OfflineBackend(Backend):
    """A fake model behind the real Router (litellm's `mock_response`): no key, no network."""

    live = False

    def __init__(self, chain: list[str]) -> None:
        self._fallbacks = chain[1:]
        self.model = OFFLINE_MODEL
        self.report_chain = [OFFLINE_MODEL]

    def transport(self, ledger: Ledger, run_id: str, max_tokens: int) -> MeteredTransport:
        return MeteredTransport(
            ledger, run_id, fallbacks=self._fallbacks, fake_model=offline_model,
            max_tokens=max_tokens,
        )

    def judge(self) -> Judge:
        return offline_judge()

    def ielts_completer(self) -> llm.Completer:
        return ielts_writing.OfflineCompleter()

    def ielts_grader(self, samples: int) -> ielts_writing.IeltsGrader:
        return ielts_writing.offline_grader()


class LiveBackend(Backend):
    """A real model chain through the Router, metered against the spend ledger."""

    live = True

    def __init__(self, chain: list[str], judge_model: str | None, cache_dir: Path | None) -> None:
        self._chain = chain
        self._judge_model = judge_model or chain[0]
        self.model = chain[0]
        self.report_chain = chain
        self.cache = None if cache_dir is None else ResponseCache(cache_dir)

    def preflight(self) -> None:
        missing = missing_keys([*self._chain, self._judge_model])
        if missing:
            raise SystemExit(f"missing environment keys (names only): {', '.join(missing)}")

    def transport(self, ledger: Ledger, run_id: str, max_tokens: int) -> MeteredTransport:
        return MeteredTransport(
            ledger, run_id, fallbacks=self._chain[1:], max_tokens=max_tokens, cache=self.cache
        )

    def judge(self) -> Judge:
        return LLMJudge(self._judge_model)

    def ielts_completer(self) -> llm.Completer:
        return ielts_writing.EvalCompleter(self._judge_model)

    def ielts_grader(self, samples: int) -> ielts_writing.IeltsGrader:
        return ielts_writing.LLMIeltsGrader(self._judge_model, samples)
