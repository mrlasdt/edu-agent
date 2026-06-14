"""
Eval harness CLI.

Usage:
  python evals/run_eval.py --suite quant --sample 0.1
  python evals/run_eval.py --suite aw --sample 1.0
  python evals/run_eval.py --suite quant --suite aw --sample 0.1

Exit code: 0 = no regression, 1 = regression detected.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from evals.human_review_queue import HumanReviewQueue
from evals.judges.aw_judge import AWResult, grade_aw_item
from evals.judges.quant_judge import grade_quant_item
from evals.loader import load_golden_set, sample_golden_set
from evals.regression import check_regression

_BASELINE_PATH = Path(__file__).parent / "baseline.json"


async def run_quant(sample_rate: float, queue: HumanReviewQueue) -> dict:
    items = sample_golden_set(load_golden_set("quant"), sample_rate, seed=42)
    results = await asyncio.gather(*[grade_quant_item(item) for item in items])
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / len(results) if results else 0.0
    by_type: dict[str, list[bool]] = {}
    for item, result in zip(items, results):
        qt = item.get("question_type", "unknown")
        by_type.setdefault(qt, []).append(result.correct)
        if not result.correct and result.error:
            queue.add_verifier_fail(result.item_id, item.get("expression", ""), "eval")
    print(f"\n=== Quant eval ({len(results)} items) ===")
    print(f"  Overall accuracy: {accuracy:.1%}")
    for qt, results_list in sorted(by_type.items()):
        pct = sum(results_list) / len(results_list)
        print(f"  {qt}: {pct:.1%} ({len(results_list)} items)")
    return {"quant_accuracy": accuracy}


async def run_aw(sample_rate: float, queue: HumanReviewQueue) -> dict:
    items = sample_golden_set(load_golden_set("aw"), sample_rate, seed=42)
    results = await asyncio.gather(*[grade_aw_item(item) for item in items])
    scores = [r.score for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    for result in results:
        queue.maybe_add_aw(result)
    print(f"\n=== AW eval ({len(results)} items) ===")
    print(f"  Average rubric score: {avg_score:.2f}/6.0")
    print(f"  Score distribution: {sorted(scores)}")
    return {"aw_avg_score": avg_score}


async def main() -> int:
    parser = argparse.ArgumentParser(description="GRE Tutor eval harness")
    parser.add_argument("--suite", action="append", choices=["quant", "aw"], default=[])
    parser.add_argument("--sample", type=float, default=1.0)
    args = parser.parse_args()
    suites = args.suite or ["quant", "aw"]

    queue = HumanReviewQueue()
    metrics: dict = {}

    if "quant" in suites:
        metrics.update(await run_quant(args.sample, queue))
    if "aw" in suites:
        metrics.update(await run_aw(args.sample, queue))

    print(f"\n=== Human review queue: {queue.count()} items flagged ===")

    # Regression check
    if _BASELINE_PATH.exists():
        baseline = json.loads(_BASELINE_PATH.read_text())
        regression = check_regression(metrics, baseline)
        if regression.has_regression:
            print(f"\n❌ REGRESSION: {regression.details}", file=sys.stderr)
            return 1
        print("\n✅ No regression detected.")
    else:
        print("\n(No baseline found — writing current metrics as new baseline)")
        _BASELINE_PATH.write_text(json.dumps(metrics, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
