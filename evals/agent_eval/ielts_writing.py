"""
IELTS Writing Task 2 grader as an eval target. It measures the shipped code: the same
sampled rubric grader, deterministic checks, feedback generator and report builder the
`/ielts` API uses (`services/agent/src/agent/ielts/`), driven through the metered transport.

Three suites:
  ielts_writing      agreement with the best label tier available (examiner, then
                     teacher_graded, captain_blind, site_or_model_weak) plus the calibration
                     gates of ADR-0014. Always prints the tier and the sample size: agreement
                     with a weak tier is a signal against noisy labels, not against an examiner.
  ielts_report       full report on a few essays, run through the mechanical report checks.
  ielts_adversarial  injection, off-topic, copied-prompt and junk responses must be flagged and
                     must not receive a high band.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import litellm

from evals.agent_eval.agreement import agreement_report, quadratic_weighted_kappa
from evals.agent_eval.cache import cache_key
from evals.agent_eval.judges import JudgeError
from evals.agent_eval.transport import AllModelsFailedError, gather_limited, tagged
from evals.exam_set.schema import (
    CRITERIA,
    CRITERION_NAMES,
    ESSAY_TIER_ORDER,
    TIER_CAVEATS,
    ExamItem,
    LabelTier,
    best_label_tier,
)
from services.agent.src.agent.ielts import llm
from services.agent.src.agent.ielts.gate import gate_report
from services.agent.src.agent.ielts.grading import GradingError, grade_essay
from services.agent.src.agent.ielts.marking import mark_essay
from services.agent.src.agent.ielts.models import Calibration
from services.agent.src.agent.ielts.report_checks import check_report
from services.agent.src.agent.ielts.scoring import (  # noqa: F401 - re-exported for the CLI/tests
    GradeParseError,
    parse_grade,
    round_overall,
    snap_band,
)

TOLERANCE = 0.5
DEFAULT_SAMPLES = 3
REPORT_CHECK_ITEMS = 5


@dataclass
class EvalGrade:
    """What the target needs from a grader: final criterion bands and the sample spread."""

    criteria: dict[str, float]
    spread: float | None = None
    samples: list[dict[str, float]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


class IeltsGrader(Protocol):
    name: str

    async def grade(self, prompt: str, essay: str, item: ExamItem) -> EvalGrade: ...


class EvalCompleter:
    """
    Calls `litellm.acompletion` so the metered transport (budget, fallbacks, cache) is underneath.
    Repeats of one request (grader samples, retries) each get their own cache slot, so a rerun
    replays independent samples instead of one reply copied into all of them.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._repeats: Counter[str] = Counter()

    async def __call__(
        self, role: str, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
        trace_id: str = "", prompt_version: str = "",
    ) -> str:
        request = cache_key(messages=messages, temperature=temperature, max_tokens=max_tokens)
        slot = self._repeats[request]
        self._repeats[request] += 1
        with tagged("judge" if role == llm.GRADER_ROLE else "feedback"):
            try:
                response = await litellm.acompletion(
                    model=self.model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, stream=False, cache_slot=slot,
                )
            except AllModelsFailedError as exc:
                raise llm.ModelUnavailableError(str(exc)) from exc
        return str(response.choices[0].message.content or "")


class LLMIeltsGrader:
    """The production sampled grader behind any litellm chat model."""

    def __init__(self, model: str, samples: int = DEFAULT_SAMPLES) -> None:
        self.name = f"llm:{model}x{samples}"
        self._completer = EvalCompleter(model)
        self._samples = samples

    async def grade(self, prompt: str, essay: str, item: ExamItem) -> EvalGrade:
        outcome = await grade_essay(
            prompt, essay, completer=self._completer, samples=self._samples, trace_id=item.id
        )
        return EvalGrade(
            criteria=outcome.criteria, spread=outcome.spread, samples=outcome.samples,
            flags=[f.code for f in outcome.flags],
        )


class HeuristicIeltsGrader:
    """Network-free stand-in: scores come from a callable over the item (offline and tests)."""

    name = "heuristic"

    def __init__(self, score: Callable[[ExamItem], dict[str, float]]) -> None:
        self._score = score

    async def grade(self, prompt: str, essay: str, item: ExamItem) -> EvalGrade:
        return EvalGrade(criteria=self._score(item))


def offline_grader() -> HeuristicIeltsGrader:
    """Plumbing check only: echoes the labels, so its agreement is 1.0 by construction."""

    def echo(item: ExamItem) -> dict[str, float]:
        assert item.labels is not None
        overall = item.labels.overall or 0.0
        return {c: item.labels.criteria.get(c, overall) for c in CRITERIA}

    return HeuristicIeltsGrader(echo)


def _between(text: str, start: str, end: str) -> str:
    a = text.find(start)
    b = text.find(end, a + len(start)) if a >= 0 else -1
    return text[a + len(start):b] if a >= 0 and b >= 0 else ""


class OfflineCompleter:
    """
    A deterministic stand-in for the marker that is deliberately over-generous: it scores every
    criterion 9 and quotes the essay's first words. The adversarial suite and the report checks
    use it offline, so only the code's own caps and validation keep the result honest.
    """

    async def __call__(
        self, role: str, messages: list[dict[str, str]], *, temperature: float, max_tokens: int,
        trace_id: str = "", prompt_version: str = "",
    ) -> str:
        user = messages[-1]["content"]
        if role == llm.GRADER_ROLE:
            return '{"TR": 9, "CC": 9, "LR": 9, "GRA": 9, "off_topic": false, "rationale": "x"}'
        essay = _between(user, "<<<ESSAY ", "<<<END ESSAY").split(">>>", 1)[-1].strip()
        lines = [x for x in essay.split("\n") if x.strip() and not x.startswith("Paragraph ")]
        quote = " ".join((lines[0] if lines else "").split()[:8])
        evidence = [{"quote": quote, "effect": "supports"}, {"quote": "not in the essay at all",
                                                             "effect": "weakens"}]
        crit = {"verdict": "ok", "evidence": evidence, "holding_back": "h",
                "to_reach_next_band": "n", "actions": [{"do": "d", "why": "w"}]}
        return json.dumps({
            "criteria": {c: crit for c in CRITERIA}, "errors": [], "checks": {},
        })


def stratified_sample(items: list[ExamItem], k: int, seed: int) -> list[ExamItem]:
    """Deterministic sample of `k` items spread across overall bands (round-robin)."""
    if k >= len(items):
        return list(items)
    rng = random.Random(seed)
    groups: dict[float, list[ExamItem]] = defaultdict(list)
    for item in items:
        assert item.labels is not None and item.labels.overall is not None
        groups[item.labels.overall].append(item)
    for group in groups.values():
        rng.shuffle(group)
    order = sorted(groups)
    picked: list[ExamItem] = []
    while len(picked) < k:
        for band in order:
            if groups[band] and len(picked) < k:
                picked.append(groups[band].pop())
    return picked


def select_items(
    items: list[ExamItem], tier: str, max_items: int, seed: int
) -> tuple[LabelTier | None, list[ExamItem], dict[str, int]]:
    """Writing Task 2 items of the chosen tier ("best" picks the strongest available)."""
    labelled = [
        i for i in items
        if i.skill == "writing_t2" and i.response and i.labels and i.labels.overall is not None
    ]
    counts: dict[str, int] = {
        t.value: sum(1 for i in labelled if i.label_provenance.tier == t)
        for t in ESSAY_TIER_ORDER
    }
    chosen = best_label_tier(labelled) if tier == "best" else LabelTier(tier)
    if chosen is None:
        return None, [], counts
    pool = [i for i in labelled if i.label_provenance.tier == chosen]
    return chosen, stratified_sample(pool, max_items, seed), counts


async def run_ielts_writing(
    items: list[ExamItem], grader: IeltsGrader, *, tier: str = "best", max_items: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    chosen, picked, counts = select_items(items, tier, max_items, seed)
    base: dict[str, Any] = {
        "label_tier": chosen.value if chosen else None,
        "label_tier_caveat": TIER_CAVEATS[chosen] if chosen else None,
        "tier_counts_available": counts,
        "grader": grader.name,
        "tolerance": TOLERANCE,
    }
    if chosen is None or not picked:
        return {
            **base, "n_items": 0, "units_total": 0, "units_dropped_model_unavailable": 0,
            "note": "no labelled IELTS Writing Task 2 items available (see evals/exam_set)",
        }
    errors: list[str] = []
    unavailable: list[str] = []

    async def one(item: ExamItem) -> dict[str, Any] | None:
        assert item.response is not None
        try:
            graded = await grader.grade(item.prompt, item.response, item)
        except (JudgeError, GradingError) as exc:
            errors.append(str(exc))
            return None
        except AllModelsFailedError as exc:
            unavailable.append(str(exc))
            return None
        return {
            "item_id": item.id, "pred": graded.criteria,
            "pred_overall": round_overall(graded.criteria), "spread": graded.spread,
            "samples": graded.samples, "flags": graded.flags,
        }

    results = await gather_limited([one(i) for i in picked])
    rows = [(i, r) for i, r in zip(picked, results, strict=True) if r is not None]
    overall_pairs: list[tuple[float, float]] = []
    per_criterion: dict[str, list[tuple[float, float]]] = {c: [] for c in CRITERIA}
    for item, row in rows:
        assert item.labels is not None and item.labels.overall is not None
        overall_pairs.append((row["pred_overall"], item.labels.overall))
        for crit, gold in item.labels.criteria.items():
            per_criterion[crit].append((row["pred"][crit], gold))
    spreads = [r["spread"] for _, r in rows if r["spread"] is not None]
    return {
        **base,
        "n_items": len(rows),
        "overall": agreement_report(overall_pairs, TOLERANCE),
        "per_criterion": {
            CRITERION_NAMES[c]: agreement_report(pairs, TOLERANCE)
            for c, pairs in per_criterion.items()
        },
        "calibration_gate": gate_report(
            overall_pairs, per_criterion, tier=chosen.value,
            qwk=quadratic_weighted_kappa(overall_pairs), spreads=spreads or None,
        ),
        "errors": len(errors),
        "units_total": len(results),
        "units_dropped_model_unavailable": len(unavailable),
        "rows": {
            "grade": [
                {"item_id": r["item_id"], "pred": r["pred"], "pred_overall": r["pred_overall"],
                 "gold_overall": i.labels.overall if i.labels else None,
                 "spread": r["spread"], "samples": r["samples"], "flags": r["flags"]}
                for i, r in rows
            ]
        },
    }


_UNCALIBRATED = Calibration(status="uncalibrated", label="Uncalibrated estimate")


async def run_ielts_report(
    items: list[ExamItem], completer: llm.Completer, *, tier: str = "best",
    max_items: int = REPORT_CHECK_ITEMS, seed: int = 42, samples: int = DEFAULT_SAMPLES,
) -> dict[str, Any]:
    """Full report (grade plus feedback) on a few essays, run through the mechanical checks."""
    chosen, picked, _ = select_items(items, tier, max_items, seed)
    unavailable: list[str] = []

    async def one(item: ExamItem) -> dict[str, Any] | None:
        assert item.response is not None
        try:
            report = await mark_essay(
                item.prompt, item.response, completer=completer, samples=samples,
                trace_id=item.id, calibration=_UNCALIBRATED,
            )
        except (GradingError, AllModelsFailedError) as exc:
            unavailable.append(str(exc))
            return None
        return {
            "item_id": item.id, "problems": check_report(report, item.response),
            "feedback_status": report.feedback_status, "status": report.status,
            "evidence_quotes": sum(len(c.evidence) for c in report.criteria),
            "error_spans": len(report.error_spans),
        }

    results = await gather_limited([one(i) for i in picked])
    done = [r for r in results if r is not None]
    return {
        "label_tier": chosen.value if chosen else None,
        "n_items": len(done),
        "units_total": len(results),
        "units_dropped_model_unavailable": len(unavailable),
        "reports_with_problems": sum(1 for r in done if r["problems"]),
        "problems": [p for r in done for p in r["problems"]],
        "feedback_status": dict(Counter(r["feedback_status"] for r in done)),
        "rows": {"report": done},
    }


async def run_ielts_adversarial(
    items: list[ExamItem], completer: llm.Completer, *, samples: int = DEFAULT_SAMPLES,
) -> dict[str, Any]:
    """Each case must raise its flags and stay at or under its band ceiling."""
    cases = [i for i in items if i.task_type == "adversarial_response"]
    unavailable: list[str] = []

    async def one(item: ExamItem) -> dict[str, Any] | None:
        assert item.response is not None
        try:
            report = await mark_essay(
                item.prompt, item.response, completer=completer, samples=samples,
                trace_id=item.id, calibration=_UNCALIBRATED,
            )
        except (GradingError, AllModelsFailedError) as exc:
            unavailable.append(str(exc))
            return None
        raised = {f.code for f in report.flags}
        overall = report.estimated_band.overall if report.estimated_band else None
        missing = [f for f in item.extras["expect_flags"] if f not in raised]
        too_high = overall is not None and overall > item.extras["max_overall"]
        return {
            "case": item.extras["case"], "overall": overall, "flags": sorted(raised),
            "missing_flags": missing, "too_high": too_high, "passed": not missing and not too_high,
        }

    results = await gather_limited([one(i) for i in cases])
    done = [r for r in results if r is not None]
    return {
        "n_cases": len(done),
        "units_total": len(results),
        "units_dropped_model_unavailable": len(unavailable),
        "passed": sum(1 for r in done if r["passed"]),
        "failed_cases": [r["case"] for r in done if not r["passed"]],
        "max_band_given": max((r["overall"] for r in done if r["overall"] is not None),
                              default=None),
        "rows": {"adversarial": done},
    }
