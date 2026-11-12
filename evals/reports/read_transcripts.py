"""
Read the transcripts: print the worst attempts of an eval run, in full.

    uv run python -m evals transcripts                       # newest run in .eval-runs/
    uv run python -m evals transcripts .eval-runs/RUN.json --worst 5
    uv run python -m evals transcripts --kind tutor_multiturn --tag multi-turn-drift

A pass rate says THAT something failed; only the transcript says WHY. The grader may be wrong
(a rule too strict), the item may be wrong (a bad ground truth), or the agent may really have
failed. Reading the worst few is the cheapest way to find out which, and it is how graders get
fixed. "Worst" = lowest pass rate over the item's samples, then most failed checks.

Run reports hold model output, so they live in the gitignored `.eval-runs/`; do not commit them.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evals.agent_eval.results import EvalReport, QuantAttempt, QuantSuiteResult
from shared.src.shared.config import anchor

DEFAULT_RUNS_DIR = anchor(".eval-runs")
DEFAULT_WORST = 5
_WRAP = 100


class WorstItem(BaseModel):
    """An item that failed at least once, with the attempt that failed the most checks."""

    item_id: str
    kind: str
    tag: str
    passed: int
    attempts: int
    failed_checks: list[str]
    worst: QuantAttempt


def load_attempts(report: EvalReport, suite: str = "quant") -> list[QuantAttempt]:
    found = report.suites.get(suite)
    rows = found.rows if isinstance(found, QuantSuiteResult) else None
    if rows is None:
        raise SystemExit(f"no per-attempt rows for suite {suite!r} in this report")
    return rows.attempts


def rank_worst(
    attempts: list[QuantAttempt], *, kind: str | None = None, tag: str | None = None,
    limit: int = DEFAULT_WORST,
) -> list[WorstItem]:
    """One entry per item that failed at least once, worst first."""
    by_item: dict[str, list[QuantAttempt]] = defaultdict(list)
    for a in attempts:
        if (kind is None or a.kind == kind) and (tag is None or a.tag == tag):
            by_item[a.item_id].append(a)
    ranked = []
    for item_id, runs in by_item.items():
        passed = sum(1 for r in runs if r.passed)
        if passed == len(runs):
            continue
        worst = max(runs, key=lambda r: sum(1 for ok in r.checks.values() if not ok))
        failed = sorted({name for r in runs for name, ok in r.checks.items() if not ok})
        ranked.append(WorstItem(
            item_id=item_id, kind=runs[0].kind, tag=runs[0].tag, passed=passed,
            attempts=len(runs), failed_checks=failed, worst=worst,
        ))
    ranked.sort(key=lambda e: (e.passed / e.attempts, -len(e.failed_checks), e.item_id))
    return ranked[:limit]


def render(entries: list[WorstItem], golden: dict[str, dict[str, Any]] | None = None) -> str:
    if not entries:
        return "No failing attempts. Read a few passing transcripts anyway: rules miss poor hints."
    out: list[str] = []
    for n, e in enumerate(entries, start=1):
        source = (golden or {}).get(e.item_id, {}).get("failure_source", "")
        out.append(
            f"=== {n}. {e.item_id}  [{e.kind} / {e.tag}]  "
            f"passed {e.passed}/{e.attempts}  failed checks: {', '.join(e.failed_checks)}"
        )
        if source:
            out.append(f"    seeded from: {source}")
        failed_here = [name for name, ok in e.worst.checks.items() if not ok]
        out.append(f"    attempt {e.worst.attempt} (failed: {', '.join(failed_here)})")
        for message in e.worst.transcript:
            body = textwrap.fill(
                " ".join(message.content.split()), _WRAP, initial_indent="      ",
                subsequent_indent="      ",
            )
            out.append(f"  {message.role}:\n{body}")
        out.append("")
    return "\n".join(out)


def newest_report(directory: Path = DEFAULT_RUNS_DIR) -> Path:
    reports = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        raise SystemExit(f"no run reports in {directory}/ (run the eval first)")
    return reports[-1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Print the worst transcripts of an eval run")
    p.add_argument("report", nargs="?", help="run report JSON (default: newest in .eval-runs/)")
    p.add_argument("--worst", type=int, default=DEFAULT_WORST, help="how many items to print")
    p.add_argument("--suite", default="quant")
    p.add_argument("--kind", default=None, help="solve, tutor, tutor_multiturn or security")
    p.add_argument("--tag", default=None, help="failure tag, for example answer-leak")
    args = p.parse_args(argv)
    path = Path(args.report) if args.report else newest_report()
    report = EvalReport.model_validate_json(path.read_text())
    from evals.loader import load_golden_set

    golden = {i["id"]: i for i in load_golden_set("quant")}
    entries = rank_worst(
        load_attempts(report, args.suite), kind=args.kind, tag=args.tag, limit=args.worst
    )
    print(f"# worst transcripts in {path.name} (run {report.run_id}, "
          f"model {report.run_meta.model if report.run_meta else '?'})")
    print(render(entries, golden))
    return 0


if __name__ == "__main__":
    sys.exit(main())
