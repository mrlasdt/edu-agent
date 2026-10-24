"""
Agent-in-the-loop eval CLI.

  python -m evals.agent_eval                       # dry-run: print the plan, no calls
  python -m evals.agent_eval --mode offline        # full pipeline on a fake model, no network
  python -m evals.agent_eval --mode live --provider free

Live runs need keys in the environment (OPENROUTER_API_KEY and/or OPENAI_API_KEY); the CLI
only checks that they are set and never prints or stores them. Cumulative spend across runs is
capped by the ledger (default 20 USD) and checked before every call.
Exit code: 0 ok, 1 regression vs --baseline, 2 budget exceeded or run failure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from evals.agent_eval import runner
from evals.agent_eval.budget import (
    DEFAULT_CEILING_USD,
    DEFAULT_LEDGER_PATH,
    BudgetExceededError,
    Ledger,
    estimate_cost,
)
from evals.agent_eval.judges import HeuristicJudge, JevJudge, Judge, JudgeQuestion, LLMJudge
from evals.agent_eval.transport import AllModelsFailedError, MeteredTransport
from evals.loader import load_golden_set
from evals.regression import check_regression
from services.agent.src.agent.agents.quant import QuantAgent

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


def provider_chain(provider: str) -> list[str]:
    """Ordered model chain: free OpenRouter first, OpenAI only where the provider allows it."""
    if provider == "free":
        return list(FREE_MODELS)
    if provider == "openai":
        return [OPENAI_MODEL]
    return [*FREE_MODELS, OPENAI_MODEL]  # "free+openai": free first, OpenAI as the fallback


def required_keys(provider: str, judge: str) -> list[str]:
    keys = {"free": ["OPENROUTER_API_KEY"], "openai": ["OPENAI_API_KEY"]}.get(
        provider, ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]
    )
    return keys + (["OPENROUTER_API_KEY"] if judge == "jev" else [])


def _offline_model(model: str, messages: list[dict[str, Any]]) -> str:
    """Deterministic fake agent: tutor replies with a hint, solver with the golden answer."""
    system = str(messages[0]["content"]).lower()
    convo = "\n".join(str(m["content"]) for m in messages[1:])
    item = next((i for i in load_golden_set("quant") if i["question"] in convo), None)
    if "tutor" in system or item is None:
        return "What is the first quantity you could compute from the information given?"
    return f"1. Compute it.\n**Answer: {item['ground_truth']}**"


def _offline_judge() -> Judge:
    def yes_no(q: JudgeQuestion) -> bool:
        return bool(q.facts["gold"])

    def score(q: JudgeQuestion, criteria: list[str]) -> float:
        return float(q.facts["gold"])

    return HeuristicJudge(yes_no, score)


def plan(args: argparse.Namespace, chain: list[str], ledger: Ledger) -> dict[str, Any]:
    n_quant = min(args.max_items, len(load_golden_set("quant"))) if "quant" in args.suite else 0
    n_aw = min(args.max_items, len(load_golden_set("aw"))) if "aw" in args.suite else 0
    agent_calls = n_quant * 5  # 2 tutor probes + up to 3 solve attempts
    judge_calls = n_quant * 7 + n_aw * 3
    per_call = estimate_cost(chain[0], 2000, 800)
    return {
        "mode": "dry-run",
        "model_chain": chain,
        "judge": args.judge,
        "items": {"quant": n_quant, "aw": n_aw},
        "max_agent_calls": agent_calls,
        "max_judge_calls": judge_calls,
        "worst_case_usd_first_model": per_call * (agent_calls + judge_calls),
        "ledger_spent_usd": ledger.spent(),
        "ledger_ceiling_usd": ledger.ceiling_usd,
    }


async def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    ledger = Ledger(Path(args.ledger), args.ceiling)
    chain = provider_chain(args.provider)
    if args.mode == "dry-run":
        return plan(args, chain, ledger), 0

    offline = args.mode == "offline"
    if not offline:
        missing = [k for k in required_keys(args.provider, args.judge) if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"missing environment keys (names only): {', '.join(missing)}")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    transport = MeteredTransport(
        ledger, run_id, fallbacks=chain[1:], fake_model=_offline_model if offline else None
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
    }
    exit_code = 0
    with transport.installed():
        try:
            if "quant" in args.suite:
                agent = QuantAgent(
                    model_config_path=runner.write_model_config(model),
                    verify_math_skill=runner.InProcessVerifySkill(),
                )
                items = runner.select_items("quant", args.max_items, SEED)
                report["suites"]["quant"] = await runner.run_quant(items, agent, judge)
            if "aw" in args.suite:
                items = runner.select_items("aw", args.max_items, SEED)
                report["suites"]["aw"] = await runner.run_aw(items, judge)
        except (BudgetExceededError, AllModelsFailedError) as exc:
            report["aborted"] = f"{type(exc).__name__}: {exc}"
            exit_code = 2
    rows = [r for s in report["suites"].values() for r in s.get("rows", {}).get("solve", [])]
    rows += [r for s in report["suites"].values() for r in s.get("rows", {}).get("tutor", [])]
    report["cost_latency"] = runner.latency_cost_summary(
        transport, [r["latency_s"] for r in rows]
    )
    report["ledger_spent_usd"] = ledger.spent()
    report["regression_metrics"] = regression_metrics(report)
    return report, exit_code


def regression_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Higher-is-better metrics suitable for check_regression."""
    out: dict[str, float] = {}
    quant = report["suites"].get("quant")
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
    p.add_argument("--suite", action="append", choices=["quant", "aw"], default=[])
    p.add_argument("--provider", choices=["free", "free+openai", "openai"], default="free")
    p.add_argument("--judge", choices=["llm", "jev"], default="llm")
    p.add_argument("--judge-model", default=None, help="litellm model string for the llm judge")
    p.add_argument("--max-items", type=int, default=5)
    p.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    p.add_argument("--ceiling", type=float, default=DEFAULT_CEILING_USD)
    p.add_argument("--out-dir", default=str(DEFAULT_RUNS_DIR))
    p.add_argument("--baseline", default=None, help="JSON of metrics to regression-check against")
    p.add_argument("--write-baseline", default=None, help="write measured metrics to this path")
    args = p.parse_args(argv)
    args.suite = args.suite or ["quant", "aw"]
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
            Path(args.write_baseline).write_text(json.dumps(report["regression_metrics"], indent=2))
        if args.baseline and Path(args.baseline).exists():
            reg = check_regression(
                report["regression_metrics"], json.loads(Path(args.baseline).read_text())
            )
            if reg.has_regression:
                print(f"REGRESSION: {reg.details}", file=sys.stderr)
                return 1
    return code
