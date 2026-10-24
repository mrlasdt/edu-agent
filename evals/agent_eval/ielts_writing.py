"""
IELTS Writing Task 2 rubric grader as an eval target.

A model scores an essay on the four public criteria (Task Response, Coherence and Cohesion,
Lexical Resource, Grammatical Range and Accuracy) with an original rubric written for this
repository (it does not reproduce the official band descriptors). The overall band is
computed in code from the four criterion bands, never taken from the model.

The target reports agreement with the best label tier available among the loaded items
(examiner, then captain_blind, then site_or_model_weak) and always prints that tier and the
sample size. Agreement with a weak tier is a calibration signal against noisy labels, not
agreement with an examiner, and the report says so.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol

import litellm

from evals.agent_eval.agreement import agreement_report
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

TOLERANCE = 0.5
MAX_ESSAY_WORDS = 700

RUBRIC = """You are an experienced IELTS Writing examiner scoring a Task 2 essay.
Score each of four criteria on the IELTS 0-9 scale in half-band steps. Judge only what is on
the page and ignore the essay's length unless it is clearly under the 250-word minimum.

Task Response (TR): does the essay answer every part of the question, take and hold a clear
  position, and support its points with relevant, developed ideas?
  4 = partial or confused answer, ideas thin or repetitive; 5 = covers the task unevenly,
  position unclear in places, ideas underdeveloped; 6 = addresses all parts, position
  present, some ideas weakly supported; 7 = clear position throughout, ideas well extended;
  8 = full, well-developed response with rare lapses; 9 = fully developed and fully relevant.
Coherence and Cohesion (CC): is the essay easy to follow, with logical paragraphs and
  linking that helps rather than gets in the way?
  4 = ideas hard to follow, linking mechanical or faulty; 5 = some organisation, linking
  inaccurate or overused; 6 = clear overall progression, cohesion sometimes faulty or
  mechanical; 7 = logical progression, a range of linking used mostly well; 8 = ideas
  sequenced so the reader notices nothing awkward; 9 = effortless flow.
Lexical Resource (LR): range, precision and naturalness of vocabulary, including spelling
  and word formation.
  4 = very limited, frequent errors that strain meaning; 5 = limited range, noticeable
  errors; 6 = adequate range, some imprecision, errors rarely impede meaning; 7 = good range
  with some less common items, occasional slips; 8 = wide, precise, natural; 9 = full
  flexibility and precision.
Grammatical Range and Accuracy (GRA): variety of sentence structures and how accurate
  they are, including punctuation.
  4 = mostly simple sentences, frequent errors; 5 = limited range of complex forms, errors
  frequent but meaning mostly clear; 6 = mix of simple and complex forms, errors that rarely
  impede communication; 7 = frequent error-free sentences, good control of varied
  structures; 8 = most sentences error-free; 9 = full range, virtually no errors.

Be calibrated: most real candidate essays fall between bands 4.5 and 7.5; use 8 and above
only for essays that are clearly excellent.
Reply with only a JSON object:
{"TR": <band>, "CC": <band>, "LR": <band>, "GRA": <band>, "rationale": "<one or two sentences>"}
"""


class IeltsGrader(Protocol):
    name: str

    async def grade(self, prompt: str, essay: str, item: ExamItem) -> dict[str, float]:
        """Criterion bands keyed TR, CC, LR, GRA."""
        ...


def round_overall(criteria: dict[str, float]) -> float:
    """Mean of the four criteria rounded to the nearest half band (a quarter rounds up)."""
    mean = sum(criteria[c] for c in CRITERIA) / len(CRITERIA)
    return math.floor(mean * 2 + 0.5) / 2


def snap_band(value: float) -> float:
    return min(9.0, max(0.0, math.floor(value * 2 + 0.5) / 2))


def truncate_words(text: str, limit: int = MAX_ESSAY_WORDS) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit])


def parse_grade(text: str) -> dict[str, float]:
    """Pull the four criterion bands out of a model reply; raises JudgeError if any is missing."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise JudgeError(f"no JSON object in grader reply: {text[:60]!r}")
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"grader reply is not valid JSON: {text[:60]!r}") from exc
    out: dict[str, float] = {}
    for crit in CRITERIA:
        try:
            out[crit] = snap_band(float(raw[crit]))
        except (KeyError, TypeError, ValueError) as exc:
            raise JudgeError(f"grader reply missing a numeric {crit}") from exc
    return out


def grade_request(prompt: str, essay: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RUBRIC},
        {
            "role": "user",
            "content": f"## Task prompt\n{prompt}\n\n## Candidate essay\n{truncate_words(essay)}",
        },
    ]


class LLMIeltsGrader:
    """Any litellm chat model (through the eval transport, so budget and fallbacks apply)."""

    def __init__(self, model: str) -> None:
        self.name = f"llm:{model}"
        self._model = model

    async def grade(self, prompt: str, essay: str, item: ExamItem) -> dict[str, float]:
        with tagged("judge"):
            response = await litellm.acompletion(
                model=self._model,
                messages=grade_request(prompt, essay),
                temperature=0,
                stream=False,
            )
        return parse_grade(str(response.choices[0].message.content or ""))


class HeuristicIeltsGrader:
    """Network-free stand-in: scores come from a callable over the item (offline and tests)."""

    name = "heuristic"

    def __init__(self, score: Callable[[ExamItem], dict[str, float]]) -> None:
        self._score = score

    async def grade(self, prompt: str, essay: str, item: ExamItem) -> dict[str, float]:
        return self._score(item)


def offline_grader() -> HeuristicIeltsGrader:
    """Plumbing check only: echoes the labels, so its agreement is 1.0 by construction."""

    def echo(item: ExamItem) -> dict[str, float]:
        assert item.labels is not None
        overall = item.labels.overall or 0.0
        return {c: item.labels.criteria.get(c, overall) for c in CRITERIA}

    return HeuristicIeltsGrader(echo)


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
            scores = await grader.grade(item.prompt, item.response, item)
        except JudgeError as exc:
            errors.append(str(exc))
            return None
        except AllModelsFailedError as exc:
            unavailable.append(str(exc))
            return None
        return {"item_id": item.id, "pred": scores, "pred_overall": round_overall(scores)}

    results = await gather_limited([one(i) for i in picked])
    rows = [(i, r) for i, r in zip(picked, results, strict=True) if r is not None]
    overall_pairs: list[tuple[float, float]] = []
    per_criterion: dict[str, list[tuple[float, float]]] = {c: [] for c in CRITERIA}
    for item, row in rows:
        assert item.labels is not None and item.labels.overall is not None
        overall_pairs.append((row["pred_overall"], item.labels.overall))
        for crit, gold in item.labels.criteria.items():
            per_criterion[crit].append((row["pred"][crit], gold))
    return {
        **base,
        "n_items": len(rows),
        "overall": agreement_report(overall_pairs, TOLERANCE),
        "per_criterion": {
            CRITERION_NAMES[c]: agreement_report(pairs, TOLERANCE)
            for c, pairs in per_criterion.items()
        },
        "errors": len(errors),
        "units_total": len(results),
        "units_dropped_model_unavailable": len(unavailable),
        "rows": {
            "grade": [
                {"item_id": r["item_id"], "pred": r["pred"], "pred_overall": r["pred_overall"],
                 "gold_overall": i.labels.overall if i.labels else None}
                for i, r in rows
            ]
        },
    }
