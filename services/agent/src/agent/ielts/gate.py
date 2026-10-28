"""
The calibration gates from ADR-0014: when may the mock call its Writing band "calibrated".

Pure functions over (predicted, gold) band pairs. The eval harness computes the metrics and
prints them; the committed `config/ielts_calibration.json` records the result, and the report
shows "uncalibrated estimate" unless that file says the decision gate passed.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from services.agent.src.agent.ielts.models import Calibration
from services.agent.src.agent.ielts.scoring import CRITERIA

Pairs = Sequence[tuple[float, float]]

CALIBRATION_PATH = Path("config/ielts_calibration.json")
DECISION_TIERS = ("examiner", "captain_blind")
DECISION_MIN_N = 60
TOLERANCE = 0.5

# development gate (can reject a bad grader, cannot certify a good one)
DEV_WITHIN = 0.75
DEV_MAX_ABS_BIAS = 0.3

# decision gate
WITHIN = 0.80
WITHIN_LOWER_BOUND = 0.70
QWK_MIN = 0.75
MAX_ABS_BIAS = 0.25
SLOPE_RANGE = (0.8, 1.2)
BUCKET_WITHIN = 0.65
CRITERION_WITHIN = 0.70
RETEST_MAX_SPREAD = 0.5
RETEST_SHARE = 0.90
INTRA_RATER_MIN = 0.8  # 8 of 10 re-scored essays within 0.5

BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("4-5", 0.0, 5.0), ("5.5-6", 5.5, 6.0), ("6.5-7", 6.5, 7.0), ("7.5+", 7.5, 9.0),
)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration_slope(pairs: Pairs) -> float | None:
    """Least-squares slope of predicted on gold. Below 1 means compression toward the middle."""
    n = len(pairs)
    if n < 2:
        return None
    mean_g = sum(g for _, g in pairs) / n
    mean_p = sum(p for p, _ in pairs) / n
    var = sum((g - mean_g) ** 2 for _, g in pairs)
    if var == 0:
        return None
    return sum((g - mean_g) * (p - mean_p) for p, g in pairs) / var


def _within(pairs: Pairs) -> tuple[int, int]:
    return sum(1 for p, g in pairs if abs(p - g) <= TOLERANCE + 1e-9), len(pairs)


def bucket_within(pairs: Pairs) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, lo, hi in BUCKETS:
        part = [(p, g) for p, g in pairs if lo <= g <= hi]
        ok, n = _within(part)
        out[name] = {"n": n, "within_tolerance": (ok / n) if n else None}
    return out


@dataclass
class GateCheck:
    name: str
    value: float | None
    threshold: str
    passed: bool | None  # None = could not be measured


def _check(name: str, value: float | None, threshold: str, ok: bool | None) -> GateCheck:
    return GateCheck(name, value, threshold, None if value is None else bool(ok))


def gate_report(
    overall: Pairs,
    per_criterion: Mapping[str, Pairs],
    *,
    tier: str | None,
    qwk: float | None,
    spreads: Sequence[float] | None = None,
    intra_rater_within: float | None = None,
) -> dict[str, Any]:
    """Both gates over one set of results. `spreads` are the per-essay sample spreads."""
    n = len(overall)
    ok, _ = _within(overall)
    within = ok / n if n else None
    interval = wilson_interval(ok, n)
    bias = sum(p - g for p, g in overall) / n if n else None

    development = [
        _check("overall within 0.5", within, f">= {DEV_WITHIN}",
               within is not None and within >= DEV_WITHIN),
        _check("overall bias", bias, f"|bias| <= {DEV_MAX_ABS_BIAS}",
               bias is not None and abs(bias) <= DEV_MAX_ABS_BIAS),
    ]

    slope = calibration_slope(overall)
    buckets = bucket_within(overall)
    bucket_values = [b["within_tolerance"] for b in buckets.values()]
    worst_bucket = None if any(v is None for v in bucket_values) else min(bucket_values)
    criterion_within = {}
    for crit in CRITERIA:
        c_ok, c_n = _within(per_criterion.get(crit, []))
        criterion_within[crit] = (c_ok / c_n) if c_n else None
    worst_criterion = (
        None if any(v is None for v in criterion_within.values())
        else min(v for v in criterion_within.values() if v is not None)
    )
    retest_share = (
        sum(1 for s in spreads if s <= RETEST_MAX_SPREAD) / len(spreads) if spreads else None
    )
    decision = [
        _check("decision set size", float(n), f">= {DECISION_MIN_N}", n >= DECISION_MIN_N),
        _check("label tier", 1.0 if tier in DECISION_TIERS else 0.0 if tier else None,
               "examiner or captain_blind", tier in DECISION_TIERS),
        _check("overall within 0.5", within, f">= {WITHIN}",
               within is not None and within >= WITHIN),
        _check("overall within 0.5, lower 95% bound", interval[0] if interval else None,
               f">= {WITHIN_LOWER_BOUND}",
               interval is not None and interval[0] >= WITHIN_LOWER_BOUND),
        _check("overall QWK", qwk, f">= {QWK_MIN}", qwk is not None and qwk >= QWK_MIN),
        _check("overall bias", bias, f"|bias| <= {MAX_ABS_BIAS}",
               bias is not None and abs(bias) <= MAX_ABS_BIAS),
        _check("calibration slope", slope, f"{SLOPE_RANGE[0]} to {SLOPE_RANGE[1]}",
               slope is not None and SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1]),
        _check("worst band bucket within 0.5", worst_bucket, f">= {BUCKET_WITHIN} in every bucket",
               worst_bucket is not None and worst_bucket >= BUCKET_WITHIN),
        _check("worst criterion within 0.5", worst_criterion, f">= {CRITERION_WITHIN} each",
               worst_criterion is not None and worst_criterion >= CRITERION_WITHIN),
        _check("test-retest spread <= 0.5", retest_share, f">= {RETEST_SHARE} of essays",
               retest_share is not None and retest_share >= RETEST_SHARE),
        _check("owner intra-rater agreement", intra_rater_within, f">= {INTRA_RATER_MIN}",
               intra_rater_within is not None and intra_rater_within >= INTRA_RATER_MIN),
    ]
    return {
        "n": n,
        "tier": tier,
        "within_tolerance": within,
        "within_tolerance_wilson95": list(interval) if interval else None,
        "bias": bias,
        "qwk": qwk,
        "slope": slope,
        "buckets": buckets,
        "per_criterion_within": criterion_within,
        "development_gate": {
            "passed": all(c.passed for c in development),
            "checks": [asdict(c) for c in development],
        },
        "decision_gate": {
            "passed": all(c.passed for c in decision),
            "unmeasured": [c.name for c in decision if c.passed is None],
            "checks": [asdict(c) for c in decision],
        },
    }


def load_calibration(path: Path = CALIBRATION_PATH) -> Calibration:
    """The committed gate result; anything missing or unreadable means uncalibrated."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    if data.get("decision_gate_passed") is True:
        n, tier, within = data.get("n"), data.get("tier"), data.get("within_half_band")
        return Calibration(
            status="calibrated", n=n, tier=tier, within_half_band=within,
            label=f"Calibrated on {n} essays ({tier}), {within:.0%} within half a band.",
            note=str(data.get("note", "")),
        )
    return Calibration(
        status="uncalibrated", label="Uncalibrated estimate",
        n=data.get("n"), tier=data.get("tier"), within_half_band=data.get("within_half_band"),
        note=str(data.get(
            "note", "No labelled set has passed the calibration gate yet (ADR-0014)."
        )),
    )
