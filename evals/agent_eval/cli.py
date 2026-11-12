"""
Agent-in-the-loop eval CLI.

  python -m evals.agent_eval                       # dry-run: print the plan, no calls
  python -m evals.agent_eval --mode offline        # full pipeline on a fake model, no network
  python -m evals.agent_eval --mode live --suite generation   # LangGraph pipeline vs baseline
  python -m evals.agent_eval --mode live --provider free
  python -m evals.agent_eval --mode offline --suite quant --samples 3   # failure-seeded suite
  python -m evals.agent_eval --mode live --suite quant --samples 3 --model openrouter/<id>:free \
      --write-baseline evals/baselines/quant.json    # one named model; checked-in baseline
  python -m evals.agent_eval --mode live --suite ielts_writing --max-items 6   # needs local data
  python -m evals.agent_eval --mode live --suite ielts_report --suite ielts_adversarial

Live runs need the keys litellm names for each model in the chain and the judge (for example
OPENROUTER_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY); the CLI only checks that they are set
and never prints or stores them. Cumulative spend across runs is
capped by the shared spend ledger (default AI_LEDGER_PATH and AI_SPEND_CAP_USD, 5 USD, the cap
the IELTS mock also obeys) and checked before every call.
Exit code: 0 ok, 1 regression vs --baseline, 2 budget exceeded, run failure or coverage below
--min-coverage (too many units dropped: no regression metrics are produced).
Successful model responses are cached in --cache-dir (gitignored); --no-cache disables it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from evals.agent_eval import ielts_writing, quant_suite, runner, transport
from evals.agent_eval.backend import Backend
from evals.agent_eval.budget import (
    BudgetExceededError,
    Ledger,
    estimate_cost,
    spend_ledger,
)
from evals.agent_eval.cache import DEFAULT_CACHE_DIR
from evals.agent_eval.results import (
    CoverageGate,
    EvalReport,
    QuantBaseline,
    QuantSuiteResult,
    RunMeta,
)
from evals.agent_eval.suites import (
    SEED,
    SUITES,
    PlanContext,
    RunContext,
    default_suite_names,
    regression_metrics,
    selected_suites,
    suite_names,
)
from evals.agent_eval.transport import AllModelsFailedError
from evals.read_transcripts import DEFAULT_RUNS_DIR
from evals.regression import check_regression
from shared.src.shared.config import load_model_config

# Order matters: openrouter/free is OpenRouter's router over whichever free model is up, so it
# is the most reliable first choice; the pinned models are fallbacks for when it errors.
FREE_MODELS = [
    "openrouter/openrouter/free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/poolside/laguna-s-2.1:free",
]
OPENAI_MODEL = "openai/gpt-4o-mini"
DEFAULT_MIN_COVERAGE = 0.8


def free_chain() -> list[str]:
    """The product's default model first (so a baseline is measured on it), then the fallbacks."""
    default = load_model_config().litellm_model("quant_agent")
    if not default.startswith("openrouter/"):
        return list(FREE_MODELS)
    return [default, *(m for m in FREE_MODELS if m != default)]


def provider_chain(provider: str, model: str | None = None) -> list[str]:
    """
    Ordered model chain: free OpenRouter first, OpenAI only where the provider allows it.
    `model` pins one litellm model string and no fallbacks: a baseline is one named model.
    """
    if model:
        return [model]
    if provider == "free":
        return free_chain()
    if provider == "openai":
        return [OPENAI_MODEL]
    return [*free_chain(), OPENAI_MODEL]  # "free+openai": free first, OpenAI as the fallback


def plan(args: argparse.Namespace, chain: list[str], ledger: Ledger) -> dict[str, Any]:
    """What a run would cost at most, from each selected suite's own count. Makes no call."""
    ctx = PlanContext(args)
    plans = {suite.name: suite.plan(ctx) for suite in selected_suites(args)}
    agent_calls = sum(p.agent_calls for p in plans.values())
    judge_calls = sum(p.judge_calls for p in plans.values())
    per_call = estimate_cost(chain[0], 2000, 800)
    return {
        "mode": "dry-run",
        "model_chain": chain,
        "judge": args.judge,
        "items": {s.name: plans[s.name].items if s.name in plans else 0 for s in SUITES},
        "samples_per_quant_item": next(
            (p.samples for p in plans.values() if p.samples is not None), None
        ),
        "max_agent_calls": agent_calls,
        "max_judge_calls": judge_calls,
        "worst_case_usd_first_model": per_call * (agent_calls + judge_calls),
        "ledger_spent_usd": ledger.spent(),
        "ledger_ceiling_usd": ledger.ceiling_usd,
    }


def backend_for(args: argparse.Namespace) -> Backend:
    """The one backend of an offline or live run, chosen from `--mode`."""
    return Backend.for_mode(
        args.mode, chain=provider_chain(args.provider, args.model), judge_model=args.judge_model,
        cache_dir=None if args.no_cache else Path(args.cache_dir),
    )


async def run(
    args: argparse.Namespace, backend: Backend | None = None
) -> tuple[EvalReport, int]:
    """Run the chosen suites (offline on a fake model, or live); never called for a dry run."""
    backend = backend or backend_for(args)
    ledger = spend_ledger(args.ledger, args.ceiling)
    backend.preflight()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    metered = backend.transport(ledger, run_id, args.max_tokens)
    ctx = RunContext(
        args=args, backend=backend, run_id=run_id, transport=metered, judge=backend.judge()
    )
    report = EvalReport(
        run_id=run_id, mode=args.mode, model_chain=backend.report_chain,
        judge=ctx.judge.name, seed=SEED,
        run_meta=run_meta(args, backend.model, backend.report_chain),
    )
    exit_code = 0
    with metered.installed():
        try:
            for suite in selected_suites(args):
                report.suites[suite.name] = await suite.run(ctx)
        except (BudgetExceededError, AllModelsFailedError) as exc:
            report.aborted = f"{type(exc).__name__}: {exc}"
            exit_code = 2
    report.cost_latency = runner.latency_cost_summary(metered, report.agent_latencies())
    report.models_used = dict(Counter(r.model for r in metered.records))
    report.ledger_spent_usd = ledger.spent()
    report.call_failures = metered.failure_summary()
    if backend.cache:
        report.cache = {"hits": backend.cache.hits, "misses": backend.cache.misses}
    low = report.coverage_failures(args.min_coverage)
    if low:
        # Tiny-n metrics from a mostly-dropped run would pass or trip a baseline arbitrarily.
        report.coverage_gate = CoverageGate(passed=False, min=args.min_coverage, coverage=low)
        report.regression_metrics = {}
        return report, 2
    report.regression_metrics = regression_metrics(report)
    return report, exit_code


def run_meta(args: argparse.Namespace, model: str, chain: list[str]) -> RunMeta:
    """Provenance of a run: which model produced the numbers, and whether it is the default."""
    config = load_model_config()
    role = config.role("quant_agent")
    product_default = f"{role.provider}/{role.model}"
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return RunMeta(
        model=model,
        model_chain=chain,
        samples=args.samples,
        max_tokens=args.max_tokens,
        product_default_model=product_default,
        is_product_default_model=model == product_default,
        math_tools=args.math_tools,
        context_budget=args.context_budget,
        prompt_versions={
            mode: config.prompt_version(
                "quant_agent_math_tools" if args.math_tools == "on" else "quant_agent", mode
            )
            for mode in ("tutor", "solve")
        },
        temperature=role.temperature,
        git_commit=commit or None,
        date_utc=time.strftime("%Y-%m-%d", time.gmtime()),
    )


def summary(report: EvalReport) -> dict[str, Any]:
    """Aggregate-only view (no model outputs) for printing and sharing."""
    stripped: dict[str, Any] = _strip_rows(report.model_dump(mode="json"))
    return stripped


def _strip_rows(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_rows(v) for k, v in obj.items() if k != "rows"}
    return obj


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Agent-in-the-loop eval")
    p.add_argument("--mode", choices=["dry-run", "offline", "live"], default="dry-run")
    p.add_argument("--suite", action="append", default=[], choices=suite_names(),
                   help="; ".join(f"{s.name} = {s.description}" for s in SUITES))
    p.add_argument("--math-tools", choices=["on", "off"], default="off",
                   help="the Quant agent with (on) or without (off, the product default and the "
                        "original single-call path) the math_evaluate/math_solve/math_compare "
                        "tools; run both on one model to measure what the tools change")
    p.add_argument("--context-budget", choices=["on", "off"], default="on",
                   help="the Quant agent with (on, the product default) or without (off) dropping "
                        "the older turns of a long session (the opening message stays pinned); "
                        "only conversations over 12 messages differ, so run both on the 20-turn "
                        "item (m005) to compare")
    p.add_argument("--samples", type=int, default=ielts_writing.DEFAULT_SAMPLES,
                   help="attempts per item in the quant suite (pass@k and pass^k are over these); "
                        "grader samples per essay in the IELTS suites (per-criterion median)")
    p.add_argument("--model", default=None,
                   help="pin one litellm model string for every role (no fallbacks), e.g. "
                        "openrouter/<id>:free; overrides --provider")
    p.add_argument("--provider", choices=["free", "free+openai", "openai"], default="free")
    p.add_argument("--judge", choices=["llm"], default="llm")
    p.add_argument("--judge-model", default=None, help="litellm model string for the llm judge")
    p.add_argument("--ielts-tier", default="best",
                   choices=["best", "examiner", "teacher_graded", "captain_blind",
                            "site_or_model_weak"],
                   help="label tier the IELTS Writing grader is scored against")
    p.add_argument("--local-dir", default=None,
                   help="third-party golden data directory (default evals/golden/local)")
    p.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                   help="fail (exit 2) if a suite completes fewer than this share of its units")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--no-cache", action="store_true", help="always call the model")
    p.add_argument("--max-tokens", type=int, default=transport.DEFAULT_MAX_TOKENS,
                   help="completion cap per model call; the product sets none, so raise this "
                        "for a reasoning model whose hidden thinking would otherwise cut replies")
    p.add_argument("--max-items", type=int, default=None,
                   help="items per suite (default 5; the quant suite defaults to all of them)")
    p.add_argument("--ledger", default=None, help="spend ledger (default AI_LEDGER_PATH)")
    p.add_argument("--ceiling", type=float, default=None,
                   help="spend cap in USD (default AI_SPEND_CAP_USD)")
    p.add_argument("--out-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--baseline", default=None, help="JSON of metrics to regression-check against")
    p.add_argument("--write-baseline", default=None, help="write measured metrics to this path")
    args = p.parse_args(argv)
    args.suite = args.suite or default_suite_names()
    args.max_items_given = args.max_items is not None
    args.max_items = args.max_items if args.max_items is not None else 5
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if not Path("prompts/quant_agent").exists():
        raise SystemExit("run from the repository root (prompts/ not found)")

    if args.mode == "dry-run":
        ledger = spend_ledger(args.ledger, args.ceiling)
        print(json.dumps(plan(args, provider_chain(args.provider, args.model), ledger), indent=2))
        return 0
    backend = backend_for(args)
    try:
        report, code = asyncio.run(run(args, backend))
    except BudgetExceededError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = re.sub(r"[^\w-]", "", report.run_id)
    (out_dir / f"{run_id}.json").write_text(report.model_dump_json(indent=2))
    print(json.dumps(summary(report), indent=2))
    if backend.live and code == 0:
        if args.write_baseline:
            write_baseline(args.write_baseline, report)
        if args.baseline and Path(args.baseline).exists():
            baseline = json.loads(Path(args.baseline).read_text())
            if baseline.get("schema") == quant_suite.BASELINE_SCHEMA:
                return compare_quant_baseline(report, QuantBaseline.model_validate(baseline))
            reg = check_regression(report.regression_metrics, baseline)
            if reg.has_regression:
                print(f"REGRESSION: {reg.details}", file=sys.stderr)
                return 1
    return code


def write_baseline(path: str, report: EvalReport) -> None:
    """
    A quant-only run writes the interval-carrying baseline (`evals/baselines/quant.json`);
    any other run writes the flat metric dict the older suites compare against.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if "quant" in report.suites:
        if set(report.suites) != {"quant"}:
            raise SystemExit("write the quant baseline from a run with --suite quant only")
        quant = report.suite("quant", QuantSuiteResult)
        assert quant is not None
        if report.run_meta is None:
            raise SystemExit("this run recorded no provenance to pin a baseline to")
        doc = quant_suite.baseline_document(quant, report.run_meta)
        Path(path).write_text(doc.model_dump_json(indent=2) + "\n")
    else:
        Path(path).write_text(json.dumps(report.regression_metrics, indent=2))


def compare_quant_baseline(report: EvalReport, baseline: QuantBaseline) -> int:
    """Print current vs baseline; exit 1 on a drop outside its interval (any security drop)."""
    suite = report.suite("quant", QuantSuiteResult)
    if suite is None:
        raise SystemExit("this baseline is for the quant suite; run with --suite quant")
    rows = quant_suite.compare_with_baseline(suite, baseline)
    for row in rows:
        print(
            f"{row.metric}: baseline {row.baseline:.3f} (n={row.baseline_n}, "
            f"95% {row.baseline_ci95}) -> now {row.current:.3f}: {row.verdict}",
            file=sys.stderr,
        )
    if any(r.verdict == "regression" for r in rows):
        print("REGRESSION: a metric fell below the interval its baseline carries", file=sys.stderr)
        return 1
    return 0
