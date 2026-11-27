"""
Band arithmetic and grader-reply parsing for the IELTS Writing mock.

The overall band is always computed here from the four criterion bands, never taken from a
model, and the median of several samples is used per criterion.
"""
from __future__ import annotations

import json
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from shared.src.shared.stats import round_half_band

# Public IELTS Writing format and scoring pages (ielts.org): the four criteria are weighted
# equally within a task, Task 1 is marked on Task Achievement and Task 2 on Task Response, and
# "Task 2 contributes twice as much as Task 1 to the Writing score".
#   https://ielts.org/take-a-test/test-types/ielts-general-training-test/ielts-general-training-format-writing
#   https://ielts.org/take-a-test/your-results/ielts-scoring-in-detail
TASK_SOURCE_URLS = (
    "https://ielts.org/take-a-test/test-types/ielts-general-training-test/"
    "ielts-general-training-format-writing",
    "https://ielts.org/take-a-test/your-results/ielts-scoring-in-detail",
)
TASK_WEIGHTS_NOTE = (
    "Writing band = (Task 1 band + 2 x Task 2 band) / 3, rounded to the nearest half band: "
    "IELTS states that Task 2 contributes twice as much as Task 1. Each task band is a mock "
    "estimate, so the combined band is one too."
)
TASK_WEIGHTS: dict[str, int] = {"T1": 1, "T2": 2}
CRITERIA_BY_TASK: dict[str, tuple[str, ...]] = {
    "T1": ("TA", "CC", "LR", "GRA"),
    "T2": ("TR", "CC", "LR", "GRA"),
}
CRITERIA: tuple[str, ...] = CRITERIA_BY_TASK["T2"]  # the Task 2 default used across the mock
CRITERION_NAMES = {
    "TA": "Task Achievement",
    "TR": "Task Response",
    "CC": "Coherence and Cohesion",
    "LR": "Lexical Resource",
    "GRA": "Grammatical Range and Accuracy",
}
# Estimates above this are not supported: the top of the scale is where an uncalibrated LLM
# grader is least trustworthy and where prompt injection aims.
MAX_ESTIMATED_BAND = 8.5


class GradeParseError(ValueError):
    """A grader reply could not be turned into four criterion bands."""


@dataclass(frozen=True)
class GradeSample:
    bands: dict[str, float]
    off_topic: bool = False
    addressed_to_marker: bool = False
    rationale: str = ""


def snap_band(value: float) -> float:
    return min(9.0, max(0.0, round_half_band(value)))


def round_overall(criteria: dict[str, float]) -> float:
    """Mean of the four criteria rounded to the nearest half band (a quarter rounds up)."""
    mean = sum(criteria.values()) / len(criteria)
    return round_half_band(mean)


def writing_band(task_bands: dict[str, float]) -> float:
    """
    The combined Writing band: the task bands weighted 1 (Task 1) to 2 (Task 2), rounded to the
    nearest half band. Eg Task 1 6.0 and Task 2 7.0 give (6 + 2 x 7) / 3 = 6.67, so 6.5.
    """
    total = sum(TASK_WEIGHTS[t] for t in task_bands)
    return round_half_band(sum(b * TASK_WEIGHTS[t] for t, b in task_bands.items()) / total)


def median_band(values: Sequence[float]) -> float:
    return snap_band(statistics.median(values))


def extract_json(text: str) -> dict[str, object]:
    """The outermost JSON object in a model reply; raises GradeParseError if there is none."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise GradeParseError(f"no JSON object in reply: {text[:60]!r}")
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise GradeParseError(f"reply is not valid JSON: {text[:60]!r}") from exc
    if not isinstance(raw, dict):
        raise GradeParseError("reply JSON is not an object")
    return raw


def parse_sample(text: str, task: str = "T2") -> GradeSample:
    """Pull the four criterion bands (and the model's flags) out of a grader reply."""
    raw = extract_json(text)
    bands: dict[str, float] = {}
    for crit in CRITERIA_BY_TASK[task]:
        try:
            bands[crit] = snap_band(float(raw[crit]))  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise GradeParseError(f"grader reply missing a numeric {crit}") from exc
    return GradeSample(
        bands=bands,
        off_topic=raw.get("off_topic") is True,
        addressed_to_marker=raw.get("addressed_to_marker") is True,
        rationale=str(raw.get("rationale", ""))[:400],
    )


def parse_grade(text: str, task: str = "T2") -> dict[str, float]:
    return parse_sample(text, task).bands
