from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegressionResult:
    has_regression: bool
    details: str


def check_regression(
    current: dict[str, float],
    baseline: dict[str, float],
    threshold: float = 0.05,
) -> RegressionResult:
    """
    Compare current metrics to baseline. Returns RegressionResult.
    A metric regresses when it drops more than `threshold` (relative) from baseline.
    """
    regressions = []
    for key, base_val in baseline.items():
        if key not in current:
            continue
        curr_val = current[key]
        if base_val == 0:
            continue
        drop = (base_val - curr_val) / base_val
        if drop > threshold:
            regressions.append(
                f"{key}: {base_val:.3f} → {curr_val:.3f} ({drop*100:.1f}% drop)"
            )

    if regressions:
        return RegressionResult(
            has_regression=True,
            details="Regression detected: " + "; ".join(regressions),
        )
    return RegressionResult(has_regression=False, details="")
