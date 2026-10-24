"""
Agreement metrics for band-scored essays: within-tolerance rate, exact match, MAE, signed
bias and quadratic weighted kappa (QWK) on the half-band scale.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

HALF_BAND_LEVELS = 19  # 0, 0.5, ... 9


def _level(band: float) -> int:
    return round(band * 2)


def within_tolerance(pairs: Sequence[tuple[float, float]], tolerance: float) -> float | None:
    """Share of (predicted, gold) pairs within `tolerance` bands of each other."""
    if not pairs:
        return None
    return sum(1 for p, g in pairs if abs(p - g) <= tolerance + 1e-9) / len(pairs)


def quadratic_weighted_kappa(
    pairs: Sequence[tuple[float, float]], levels: int = HALF_BAND_LEVELS
) -> float | None:
    """
    QWK between predicted and gold half bands (1 is perfect, 0 is chance level). None when it
    is undefined: no pairs, or one of the raters used a single value for every item.
    """
    n = len(pairs)
    if n == 0:
        return None
    observed = [[0.0] * levels for _ in range(levels)]
    pred_hist = [0.0] * levels
    gold_hist = [0.0] * levels
    for pred, gold in pairs:
        p, g = _level(pred), _level(gold)
        observed[p][g] += 1
        pred_hist[p] += 1
        gold_hist[g] += 1
    num = den = 0.0
    for i in range(levels):
        for j in range(levels):
            weight = (i - j) ** 2 / (levels - 1) ** 2
            num += weight * observed[i][j]
            den += weight * pred_hist[i] * gold_hist[j] / n
    if den == 0:
        return None
    return 1.0 - num / den


def agreement_report(
    pairs: Sequence[tuple[float, float]], tolerance: float = 0.5
) -> dict[str, Any]:
    """`pairs` are (predicted, gold). Every figure is computed over exactly these n pairs."""
    n = len(pairs)
    diffs = [p - g for p, g in pairs]
    return {
        "n": n,
        "tolerance": tolerance,
        "within_tolerance": within_tolerance(pairs, tolerance),
        "exact": (sum(1 for d in diffs if abs(d) < 1e-9) / n) if n else None,
        "mae": (sum(abs(d) for d in diffs) / n) if n else None,
        "bias": (sum(diffs) / n) if n else None,
        "qwk": quadratic_weighted_kappa(pairs),
    }
