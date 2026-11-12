"""
Small statistics helpers used by the IELTS release gate and the eval reports, kept in one place so
two reports cannot disagree about the same number.

- `wilson`: Wilson score interval for a proportion, at the two-sided 95% level.
- `percentile`: nearest-rank percentile (the smallest value with at least q of the data at or
  below it), so a p95 is always a value that was actually observed.
- `round_half_band` / `to_half_band`: IELTS bands move in steps of 0.5.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Z95 = 1.959964  # two-sided 95% normal quantile


def wilson(successes: float, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a proportion (successes may be fractional); None when n is 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..1); None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def round_half_band(value: float) -> float:
    """Round to the nearest half band; a quarter rounds up."""
    return math.floor(value * 2 + 0.5) / 2


def is_half_band(value: float) -> bool:
    """True for 0, 0.5, ... 9: a band an IELTS marker can award."""
    return 0.0 <= value <= 9.0 and (value * 2) % 1 == 0


def to_half_band(value: Any) -> float | None:
    """The value as a half band 0..9, or None if it is not a number or not a half band."""
    try:
        band = float(value)
    except (TypeError, ValueError):
        return None
    return band if is_half_band(band) else None
