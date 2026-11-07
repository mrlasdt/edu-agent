"""
Band arithmetic and grader-reply parsing for the IELTS Writing mock.

The overall band is always computed here from the four criterion bands, never taken from a
model, and the median of several samples is used per criterion.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

CRITERIA: tuple[str, ...] = ("TR", "CC", "LR", "GRA")
CRITERION_NAMES = {
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
    return min(9.0, max(0.0, math.floor(value * 2 + 0.5) / 2))


def round_overall(criteria: dict[str, float]) -> float:
    """Mean of the four criteria rounded to the nearest half band (a quarter rounds up)."""
    mean = sum(criteria[c] for c in CRITERIA) / len(CRITERIA)
    return math.floor(mean * 2 + 0.5) / 2


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


def parse_sample(text: str) -> GradeSample:
    """Pull the four criterion bands (and the model's flags) out of a grader reply."""
    raw = extract_json(text)
    bands: dict[str, float] = {}
    for crit in CRITERIA:
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


def parse_grade(text: str) -> dict[str, float]:
    return parse_sample(text).bands
