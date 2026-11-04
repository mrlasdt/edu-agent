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

Live runs need keys in the environment (OPENROUTER_API_KEY and/or OPENAI_API_KEY); the CLI
only checks that they are set and never prints or stores them. Cumulative spend across runs is
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
import os
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from evals.agent_eval import (
    generation,
    ielts_writing,
    offline_generation,
    quant_suite,
    runner,
    transport,
)
from evals.agent_eval.budget import (
    BudgetExceededError,
    Ledger,
    estimate_cost,
    spend_ledger,
)
from evals.agent_eval.cache import DEFAULT_CACHE_DIR, ResponseCache
from evals.agent_eval.judges import HeuristicJudge, JevJudge, Judge, JudgeQuestion, LLMJudge
from evals.agent_eval.transport import AllModelsFailedError, FakeToolCalls, MeteredTransport
from evals.exam_set.schema import ExamItem
from evals.loader import load_exam_items, load_golden_set, sample_golden_set
from evals.regression import check_regression
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.ielts import llm
from services.agent.src.agent.pipeline.factory import build_baseline, build_pipeline
from services.agent.src.agent.pipeline.schema import GenerationRequest
from shared.src.shared.config import load_model_config

# Order matters: openrouter/free is OpenRouter's router over whichever free model is up, so it
# is the most reliable first choice; the pinned models are fallbacks for when it errors.
FREE_MODELS = [
    "openrouter/openrouter/free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/poolside/laguna-s-2.1:free",
]
OPENAI_MODEL = "openai/gpt-4o-mini"
DEFAULT_RUNS_DIR = Path(".eval-runs")
OFFLINE_MODEL = "offline/fake"
SEED = 42
DEFAULT_MIN_COVERAGE = 0.8


def provider_chain(provider: str, model: str | None = None) -> list[str]:
    """
    Ordered model chain: free OpenRouter first, OpenAI only where the provider allows it.
    `model` pins one litellm model string and no fallbacks: a baseline is one named model.
    """
    if model:
        return [model]
    if provider == "free":
        return list(FREE_MODELS)
    if provider == "openai":
        return [OPENAI_MODEL]
    return [*FREE_MODELS, OPENAI_MODEL]  # "free+openai": free first, OpenAI as the fallback


def required_keys(provider: str, judge: str, model: str | None = None) -> list[str]:
    if model:
        keys = [
            {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}.get(
                model.partition("/")[0], "OPENROUTER_API_KEY"
            )
        ]
    else:
        keys = {"free": ["OPENROUTER_API_KEY"], "openai": ["OPENAI_API_KEY"]}.get(
            provider, ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]
        )
    return keys + (["OPENROUTER_API_KEY"] if judge == "jev" else [])


def _offline_model(model: str, messages: list[dict[str, Any]]) -> str | FakeToolCalls:
    """
    Deterministic fake agent: tutor replies with a hint, solver with the golden answer, and
    the question-generation roles with seeded flaws (see offline_generation).
    """
    generated = offline_generation.respond(model, messages)
    if generated is not None:
        return generated
    system = str(messages[0]["content"]).lower()
    convo = "\n".join(str(m["content"]) for m in messages[1:])
    item = next(
        (i for i in load_golden_set("quant") if i.get("question") and i["question"] in convo), None
    )
    if item is None:
        return "What is the first quantity you could compute from the information given?"
    asked_for_a_tool = any(m.get("role") == "tool" for m in messages)
    if "math_evaluate" in system and item.get("expression") and not asked_for_a_tool:
        # the tools prompt is in play: do what a careful model does, compute before answering
        equation = "=" in item["expression"]
        name = "math_solve" if equation else "math_evaluate"
        argument = "equation" if equation else "expression"
        return FakeToolCalls([(name, {argument: item["expression"]})])
    if "tutor" in system:
        return (
            f"Let's stay with the problem: {item['question']} "
            "What is the first quantity you could work out from it?"
        )
    return f"1. Compute it.\n**Answer: {item['ground_truth']}**"


def _offline_judge() -> Judge:
    def yes_no(q: JudgeQuestion) -> bool:
        return bool(q.facts["gold"])

    def score(q: JudgeQuestion, criteria: list[str]) -> float:
        return float(q.facts["gold"])

    return HeuristicJudge(yes_no, score)


def _ielts_items(args: argparse.Namespace) -> list[ExamItem]:
    return load_exam_items(exam="ielts", include_local=True, local_directory=args.local_dir)


def selected_quant_items(
    args: argparse.Namespace, items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every failure-seeded item, unless --max-items asked for a deterministic subset."""
    if not args.max_items_given:
        return items
    return runner.select_items("quant", args.max_items, SEED)


def quant_item_count(args: argparse.Namespace, items: list[dict[str, Any]]) -> int:
    return len(selected_quant_items(args, items))


def plan(args: argparse.Namespace, chain: list[str], ledger: Ledger) -> dict[str, Any]:
    quant_items = load_golden_set("quant")
    n_quant_cases = quant_item_count(args, quant_items) if "quant" in args.suite else 0
    quant_agent_calls = 0
    if "quant" in args.suite:
        quant_agent_calls = sum(
            (len(i["turns"]) if i["kind"] == "tutor_multiturn" else 1) * args.samples
            for i in selected_quant_items(args, quant_items)
            if i["kind"] in quant_suite.MODEL_KINDS
        )
    solve_items = [i for i in quant_items if i["kind"] == "solve"]
    n_quant = min(args.max_items, len(solve_items)) if "quant_judge" in args.suite else 0
    n_aw = min(args.max_items, len(load_golden_set("aw"))) if "aw" in args.suite else 0
    n_gen = (
        min(args.max_items, len(load_golden_set("quant_generation")))
        if "generation" in args.suite
        else 0
    )
    agent_calls = n_quant * 5 + quant_agent_calls  # judge suite: 2 tutor probes + up to 3 solves
    agent_calls += n_gen * (1 + 7)  # baseline 1; pipeline planner + 3 x (generate + critic)
    n_ielts = n_report = n_adversarial = 0
    if {"ielts_writing", "ielts_report"} & set(args.suite):
        _, picked, _ = ielts_writing.select_items(
            _ielts_items(args), args.ielts_tier, args.max_items, SEED
        )
        n_ielts = len(picked) if "ielts_writing" in args.suite else 0
        if "ielts_report" in args.suite:
            n_report = min(len(picked), ielts_writing.REPORT_CHECK_ITEMS)
    if "ielts_adversarial" in args.suite:
        n_adversarial = sum(
            1 for i in _ielts_items(args) if i.task_type == "adversarial_response"
        )
    # one verdict per emitted item; an IELTS essay costs `samples` grader calls (+1 feedback)
    judge_calls = n_quant * 7 + n_aw * 3 + n_gen * 2 + n_ielts * args.samples
    judge_calls += (n_report + n_adversarial) * (args.samples + 1)
    per_call = estimate_cost(chain[0], 2000, 800)
    return {
        "mode": "dry-run",
        "model_chain": chain,
        "judge": args.judge,
        "items": {
            "quant": n_quant_cases, "quant_judge": n_quant, "aw": n_aw, "generation": n_gen,
            "ielts_writing": n_ielts,
            "ielts_report": n_report, "ielts_adversarial": n_adversarial,
        },
        "samples_per_quant_item": args.samples if "quant" in args.suite else None,
        "max_agent_calls": agent_calls,
        "max_judge_calls": judge_calls,
        "worst_case_usd_first_model": per_call * (agent_calls + judge_calls),
        "ledger_spent_usd": ledger.spent(),
        "ledger_ceiling_usd": ledger.ceiling_usd,
    }


async def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    ledger = spend_ledger(args.ledger, args.ceiling)
    chain = provider_chain(args.provider, args.model)
    if args.mode == "dry-run":
        return plan(args, chain, ledger), 0

    offline = args.mode == "offline"
    if not offline:
        needed = required_keys(args.provider, args.judge, args.model)
        missing = [k for k in needed if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"missing environment keys (names only): {', '.join(missing)}")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    cache = None if offline or args.no_cache else ResponseCache(Path(args.cache_dir))
    transport = MeteredTransport(
        ledger,
        run_id,
        fallbacks=chain[1:],
        fake_model=_offline_model if offline else None,
        max_tokens=args.max_tokens,
        cache=cache,
    )
    model = OFFLINE_MODEL if offline else chain[0]
    judge: Judge
    if offline:
        judge = _offline_judge()
    elif args.judge == "jev":
        judge = JevJudge(os.environ["OPENROUTER_API_KEY"], transport)
    else:
        judge = LLMJudge(args.judge_model or chain[0])

    report: dict[str, Any] = {
        "run_id": run_id, "mode": args.mode, "model_chain": chain if not offline else [model],
        "judge": judge.name, "seed": SEED, "suites": {},
        "run_meta": run_meta(args, model, chain if not offline else [model]),
    }
    exit_code = 0
    with transport.installed():
        try:
            if "quant" in args.suite:
                report["suites"]["quant"] = await _run_quant(
                    args, model, run_id, selected_quant_items(args, load_golden_set("quant"))
                )
            if "quant_judge" in args.suite:
                agent = QuantAgent(
                    model_config_path=runner.write_model_config(model),
                    verify_math_skill=runner.InProcessVerifySkill(),
                    math_tools_skill=runner.InProcessMathTools(),
                    math_tools_enabled=args.math_tools == "on",
                )
                solve_items = [i for i in load_golden_set("quant") if i["kind"] == "solve"]
                rate = min(1.0, args.max_items / len(solve_items))
                items = sample_golden_set(solve_items, rate, seed=SEED)
                report["suites"]["quant_judge"] = await runner.run_quant(items, agent, judge)
            if "aw" in args.suite:
                items = runner.select_items("aw", args.max_items, SEED)
                report["suites"]["aw"] = await runner.run_aw(items, judge)
            if "generation" in args.suite:
                report["suites"]["generation"] = await _run_generation(
                    args.max_items, model, judge, transport
                )
            ielts_model = args.judge_model or chain[0]
            completer: llm.Completer = (
                ielts_writing.OfflineCompleter() if offline
                else ielts_writing.EvalCompleter(ielts_model)
            )
            if "ielts_writing" in args.suite:
                grader: ielts_writing.IeltsGrader = (
                    ielts_writing.offline_grader()
                    if offline
                    else ielts_writing.LLMIeltsGrader(ielts_model, args.samples)
                )
                report["suites"]["ielts_writing"] = await ielts_writing.run_ielts_writing(
                    _ielts_items(args), grader, tier=args.ielts_tier,
                    max_items=args.max_items, seed=SEED,
                )
            if "ielts_report" in args.suite:
                report["suites"]["ielts_report"] = await ielts_writing.run_ielts_report(
                    _ielts_items(args), completer, tier=args.ielts_tier,
                    max_items=min(args.max_items, ielts_writing.REPORT_CHECK_ITEMS), seed=SEED,
                    samples=args.samples,
                )
            if "ielts_adversarial" in args.suite:
                report["suites"]["ielts_adversarial"] = await ielts_writing.run_ielts_adversarial(
                    _ielts_items(args), completer, samples=args.samples
                )
        except (BudgetExceededError, AllModelsFailedError) as exc:
            report["aborted"] = f"{type(exc).__name__}: {exc}"
            exit_code = 2
    rows = [r for s in report["suites"].values() for r in s.get("rows", {}).get("solve", [])]
    rows += [r for s in report["suites"].values() for r in s.get("rows", {}).get("tutor", [])]
    report["cost_latency"] = runner.latency_cost_summary(
        transport, [r["latency_s"] for r in rows]
    )
    report["models_used"] = dict(Counter(r.model for r in transport.records))
    report["ledger_spent_usd"] = ledger.spent()
    report["call_failures"] = transport.failure_summary()
    if cache:
        report["cache"] = {"hits": cache.hits, "misses": cache.misses}
    low = runner.coverage_failures(report, args.min_coverage)
    if low:
        # Tiny-n metrics from a mostly-dropped run would pass or trip a baseline arbitrarily.
        report["coverage_gate"] = {"passed": False, "min": args.min_coverage, "coverage": low}
        report["regression_metrics"] = {}
        return report, 2
    report["regression_metrics"] = regression_metrics(report)
    return report, exit_code


async def _run_quant(
    args: argparse.Namespace, model: str, run_id: str, items: list[dict[str, Any]]
) -> dict[str, Any]:
    """The failure-seeded suite on `model`: real agent, rule graders, `--samples` attempts."""
    config = runner.write_model_config(model)
    skill = runner.InProcessVerifySkill()

    def factory(notes: list[str]) -> QuantAgent:
        return QuantAgent(
            model_config_path=config, verify_math_skill=skill,
            retrieve_corpus_skill=quant_suite.NotesSkill(notes),
            math_tools_skill=runner.InProcessMathTools(),
            math_tools_enabled=args.math_tools == "on",
        )

    with quant_suite.collecting_turns() as collector:
        return await quant_suite.run_quant_suite(
            items, agent_factory=factory, verify_skill=skill, samples=args.samples,
            collector=collector, run_prefix=run_id,
        )


def run_meta(args: argparse.Namespace, model: str, chain: list[str]) -> dict[str, Any]:
    """Provenance of a run: which model produced the numbers, and whether it is the default."""
    config = load_model_config()
    role = config.role("quant_agent")
    product_default = f"{role.provider}/{role.model}"
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {
        "model": model,
        "model_chain": chain,
        "samples": args.samples,
        "max_tokens": args.max_tokens,
        "product_default_model": product_default,
        "is_product_default_model": model == product_default,
        "math_tools": args.math_tools,
        "prompt_versions": {
            mode: config.prompt_version(
                "quant_agent_math_tools" if args.math_tools == "on" else "quant_agent", mode
            )
            for mode in ("tutor", "solve")
        },
        "temperature": role.temperature,
        "git_commit": commit or None,
        "date_utc": time.strftime("%Y-%m-%d", time.gmtime()),
    }


async def _run_generation(
    max_items: int, model: str, judge: Judge, transport: MeteredTransport
) -> dict[str, Any]:
    """Baseline vs LangGraph pipeline, both routed to `model`, on the same requests."""
    config = runner.write_model_config(model)
    skill = runner.InProcessVerifySkill()
    requests = [
        GenerationRequest.model_validate(i)
        for i in runner.select_items("quant_generation", max_items, SEED)
    ]
    targets: dict[str, generation.Target] = {
        "baseline": build_baseline(config),
        "pipeline": build_pipeline(skill, config),
    }
    return await generation.run_generation(requests, targets, judge, skill, transport)


def regression_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Higher-is-better metrics suitable for check_regression."""
    out: dict[str, float] = {}
    quant = report["suites"].get("quant_judge")
    if quant:
        pairs = {
            "quant_solve_accuracy": quant["solve"]["accuracy"],
            "quant_tutor_no_leak_rate": None
            if quant["tutor"]["leak_rate_rule"] is None
            else 1 - quant["tutor"]["leak_rate_rule"],
            "judge_leak_agreement": quant["judge_calibration"]["leak_labels"]["agreement"],
            "judge_correctness_agreement": quant["judge_calibration"]["correctness_labels"][
                "agreement"
            ],
        }
        out.update({k: v for k, v in pairs.items() if v is not None})
    aw = report["suites"].get("aw")
    if aw and aw["within_tolerance"] is not None:
        out["judge_aw_within_tolerance"] = aw["within_tolerance"]
    ielts = report["suites"].get("ielts_writing")
    if ielts and ielts["n_items"] and ielts["overall"]["within_tolerance"] is not None:
        # keyed by tier so a weak-label baseline is never compared with an examiner one
        out[f"ielts_overall_within_half_band__{ielts['label_tier']}"] = ielts["overall"][
            "within_tolerance"
        ]
    adversarial = report["suites"].get("ielts_adversarial")
    if adversarial and adversarial["n_cases"]:
        out["ielts_adversarial_pass_rate"] = adversarial["passed"] / adversarial["n_cases"]
    reports = report["suites"].get("ielts_report")
    if reports and reports["n_items"]:
        out["ielts_report_sound_rate"] = 1 - reports["reports_with_problems"] / reports["n_items"]
    gen = report["suites"].get("generation")
    if gen:
        pipe = gen["targets"].get("pipeline", {})
        if pipe.get("valid_yield") is not None:
            out["gen_pipeline_valid_yield"] = pipe["valid_yield"]
        if pipe.get("shipped_defect_rate") is not None:
            out["gen_pipeline_no_defect_rate"] = 1 - pipe["shipped_defect_rate"]
    return out


def summary(report: dict[str, Any]) -> dict[str, Any]:
    """Aggregate-only view (no model outputs) for printing and sharing."""
    if report.get("mode") == "dry-run":
        return report
    return {k: v for k, v in _strip_rows(report).items()}


def _strip_rows(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_rows(v) for k, v in obj.items() if k != "rows"}
    return obj


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Agent-in-the-loop eval")
    p.add_argument("--mode", choices=["dry-run", "offline", "live"], default="dry-run")
    p.add_argument("--suite", action="append", default=[],
                   choices=["quant", "quant_judge", "aw", "generation", "ielts_writing",
                            "ielts_report", "ielts_adversarial"],
                   help="quant = failure-seeded suite (rule graders, --samples attempts per "
                        "item); quant_judge = LLM-judge calibration on the solve items")
    p.add_argument("--math-tools", choices=["on", "off"], default="off",
                   help="the Quant agent with (on) or without (off, the product default and the "
                        "original single-call path) the math_evaluate/math_solve/math_compare "
                        "tools; run both on one model to measure what the tools change")
    p.add_argument("--samples", type=int, default=ielts_writing.DEFAULT_SAMPLES,
                   help="attempts per item in the quant suite (pass@k and pass^k are over these); "
                        "grader samples per essay in the IELTS suites (per-criterion median)")
    p.add_argument("--model", default=None,
                   help="pin one litellm model string for every role (no fallbacks), e.g. "
                        "openrouter/<id>:free; overrides --provider")
    p.add_argument("--provider", choices=["free", "free+openai", "openai"], default="free")
    p.add_argument("--judge", choices=["llm", "jev"], default="llm")
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
    args.suite = args.suite or ["quant", "quant_judge", "aw", "generation"]
    args.max_items_given = args.max_items is not None
    args.max_items = args.max_items if args.max_items is not None else 5
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if not Path("prompts/quant_agent").exists():
        raise SystemExit("run from the repository root (prompts/ not found)")

    try:
        report, code = asyncio.run(run(args))
    except BudgetExceededError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 2
    if args.mode != "dry-run":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = re.sub(r"[^\w-]", "", report["run_id"])
        (out_dir / f"{run_id}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary(report), indent=2))
    if args.mode == "live" and code == 0:
        if args.write_baseline:
            write_baseline(args.write_baseline, report)
        if args.baseline and Path(args.baseline).exists():
            baseline = json.loads(Path(args.baseline).read_text())
            if baseline.get("schema") == quant_suite.BASELINE_SCHEMA:
                return compare_quant_baseline(report, baseline)
            reg = check_regression(report["regression_metrics"], baseline)
            if reg.has_regression:
                print(f"REGRESSION: {reg.details}", file=sys.stderr)
                return 1
    return code


def write_baseline(path: str, report: dict[str, Any]) -> None:
    """
    A quant-only run writes the interval-carrying baseline (`evals/baselines/quant.json`);
    any other run writes the flat metric dict the older suites compare against.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if "quant" in report["suites"]:
        if set(report["suites"]) != {"quant"}:
            raise SystemExit("write the quant baseline from a run with --suite quant only")
        doc = quant_suite.baseline_document(report["suites"]["quant"], report["run_meta"])
        Path(path).write_text(json.dumps(doc, indent=2) + "\n")
    else:
        Path(path).write_text(json.dumps(report["regression_metrics"], indent=2))


def compare_quant_baseline(report: dict[str, Any], baseline: dict[str, Any]) -> int:
    """Print current vs baseline; exit 1 on a drop outside its interval (any security drop)."""
    suite = report["suites"].get("quant")
    if suite is None:
        raise SystemExit("this baseline is for the quant suite; run with --suite quant")
    rows = quant_suite.compare_with_baseline(suite, baseline)
    for row in rows:
        print(
            f"{row['metric']}: baseline {row['baseline']:.3f} (n={row['baseline_n']}, "
            f"95% {row['baseline_ci95']}) -> now {row['current']:.3f}: {row['verdict']}",
            file=sys.stderr,
        )
    if any(r["verdict"] == "regression" for r in rows):
        print("REGRESSION: a metric fell below the interval its baseline carries", file=sys.stderr)
        return 1
    return 0
