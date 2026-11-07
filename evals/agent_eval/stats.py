"""
Small-sample statistics for eval reports: what a pass rate over n items can and cannot say.

Three ideas, all standard:

- A pass rate is a proportion, and with 8 items it is a blurry number. The Wilson score
  interval says how blurry (`wilson`). Every rate in a report carries its n and its interval.
- One model call per item hides luck. Run each item k times and report two numbers
  (`group_summary`): pass@k, the share of items where at least one of k attempts passed
  ("can it ever do this?"), and pass^k, the share where all k passed ("will a learner always
  get this?"). pass^k is the one a tutoring product lives or dies by.
- The mean of the per-attempt pass rates is what one learner sees on average. Attempts of the
  same item are not independent, so its interval counts items, not attempts: a Wilson interval
  with n = number of items (`mean_rate_ci`). That is conservative, and unlike a bootstrap it
  stays wide at a handful of items and does not collapse to a point when every attempt passes.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from evals.agent_eval.results import GroupStats, Rate

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


def mean_rate_ci(mean: float, n_items: int) -> tuple[float, float] | None:
    """95% interval of a mean per-item pass rate, treating each item as one observation."""
    return wilson(mean * n_items, n_items)


def rate(successes: int, n: int) -> Rate:
    """A proportion with its n and 95% Wilson interval (value and interval None when n is 0)."""
    ci = wilson(successes, n)
    return Rate(
        value=successes / n if n else None,
        successes=successes,
        n=n,
        ci95=None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
    )


def group_summary(per_item: Mapping[str, Sequence[bool]], k: int) -> GroupStats:
    """
    Summarise one group of items. `per_item` maps an item id to the pass/fail of each completed
    attempt. Items with no completed attempt are left out (and counted by the caller as dropped).
    """
    runs = {i: list(a) for i, a in per_item.items() if a}
    n = len(runs)
    ever = sum(1 for a in runs.values() if any(a))
    always = sum(1 for a in runs.values() if all(a))
    per_item_rate = [sum(a) / len(a) for a in runs.values()]
    mean = sum(per_item_rate) / n if n else None
    ci = None if mean is None else mean_rate_ci(mean, n)
    return GroupStats(
        n_items=n,
        samples=k,
        attempts=sum(len(a) for a in runs.values()),
        mean_pass_rate=None if mean is None else round(mean, 4),
        mean_pass_rate_ci95=None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
        pass_at_k=rate(ever, n),
        pass_hat_k=rate(always, n),
        flaky_items=sorted(i for i, a in runs.items() if any(a) and not all(a)),
        always_failing_items=sorted(i for i, a in runs.items() if not any(a)),
    )


def resolution(summary: GroupStats) -> float | None:
    """Half-width of the widest 95% interval in a group: the smallest change it can resolve."""
    halves = []
    for interval in (summary.pass_at_k.ci95, summary.pass_hat_k.ci95):
        if interval:
            halves.append((interval[1] - interval[0]) / 2)
    ci = summary.mean_pass_rate_ci95
    if ci:
        halves.append((ci[1] - ci[0]) / 2)
    return round(max(halves), 3) if halves else None


def verdict_vs_baseline(value: float, baseline_ci: Sequence[float] | None) -> str:
    """
    "regression" / "improvement" only when `value` falls outside the interval the baseline
    itself carries, otherwise "within noise". With few items most changes are within noise, and
    saying so is the point: the suite detects large regressions, not small ones.
    """
    if baseline_ci is None:
        return "within noise"
    low, high = baseline_ci
    if value < low:
        return "regression"
    if value > high:
        return "improvement"
    return "within noise"
