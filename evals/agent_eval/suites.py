"""
The suites of the agent-in-the-loop eval, each declared once.

A `Suite` says how many calls it would make (`plan`), how it runs on the chosen backend (`run`)
and which higher-is-better figures it contributes to a regression check (`metrics`). `SUITES` is
the one list: the CLI reads its names for `--suite`, the dry-run plan adds up every selected
suite's calls, `run` walks it in order, and `regression_metrics` merges what each suite reports.
Adding a suite is one class and one entry here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

from evals.agent_eval import generation, ielts_writing, quant_suite, runner
from evals.agent_eval.backend import Backend
from evals.agent_eval.judges import Judge
from evals.agent_eval.results import (
    AwResult,
    EvalReport,
    GenerationResult,
    IeltsAdversarialResult,
    IeltsReportResult,
    IeltsWritingResult,
    QuantJudgeResult,
    QuantSuiteResult,
    SuiteResult,
)
from evals.agent_eval.transport import MeteredTransport
from evals.exam_set.schema import ExamItem
from evals.loader import load_exam_items, load_golden_set, sample_golden_set
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.pipeline.factory import build_baseline, build_pipeline
from services.agent.src.agent.pipeline.schema import GenerationRequest

SEED = 42


@dataclass
class PlanContext:
    """What a suite needs to count its work: the parsed command line."""

    args: argparse.Namespace

    @cached_property
    def ielts_items(self) -> list[ExamItem]:
        return load_exam_items(
            exam="ielts", include_local=True, local_directory=self.args.local_dir
        )


@dataclass
class RunContext(PlanContext):
    """What a suite needs to run: the backend, the run's id, the transport and the judge."""

    backend: Backend
    run_id: str
    transport: MeteredTransport
    judge: Judge


@dataclass(frozen=True)
class SuitePlan:
    """The most a suite would do: its items and the model calls behind them."""

    items: int
    agent_calls: int = 0
    judge_calls: int = 0
    samples: int | None = None  # attempts per item, for a suite that repeats items


class Suite[R: SuiteResult](Protocol):
    name: str  # the `--suite` value and the key of its result in the report
    description: str
    default: bool  # run when no --suite is given
    result_type: type[R]

    def plan(self, ctx: PlanContext) -> SuitePlan: ...

    async def run(self, ctx: RunContext) -> R: ...

    def metrics(self, result: R) -> dict[str, float]:
        """Higher-is-better figures for `check_regression`."""
        ...


def selected_quant_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Every failure-seeded item, unless --max-items asked for a deterministic subset."""
    if not args.max_items_given:
        return load_golden_set("quant")
    return runner.select_items("quant", args.max_items, SEED)


class QuantSuite:
    name = "quant"
    description = "the failure-seeded suite (rule graders, --samples attempts per item)"
    default = True
    result_type = QuantSuiteResult

    def plan(self, ctx: PlanContext) -> SuitePlan:
        items = selected_quant_items(ctx.args)
        calls = sum(
            (len(i["turns"]) if i["kind"] == "tutor_multiturn" else 1) * ctx.args.samples
            for i in items
            if i["kind"] in quant_suite.MODEL_KINDS
        )
        return SuitePlan(items=len(items), agent_calls=calls, samples=ctx.args.samples)

    async def run(self, ctx: RunContext) -> QuantSuiteResult:
        """The suite on the backend's model: real agent, rule graders, `--samples` attempts."""
        args = ctx.args
        config = runner.write_model_config(ctx.backend.model)
        verify_math = runner.in_process_verify_math()

        def factory(notes: list[str]) -> QuantAgent:
            return runner.build_quant_agent(
                model_config_path=config, verify_math=verify_math,
                retrieve_corpus=quant_suite.NotesCorpus(notes),
                math_tools_enabled=args.math_tools == "on",
            )

        with quant_suite.collecting_turns() as collector:
            return await quant_suite.run_quant_suite(
                selected_quant_items(args), agent_factory=factory, verify_math=verify_math,
                samples=args.samples, collector=collector, run_prefix=ctx.run_id,
            )

    def metrics(self, result: QuantSuiteResult) -> dict[str, float]:
        return {}  # judged against its own interval-carrying baseline, not a flat metric


class QuantJudgeSuite:
    name = "quant_judge"
    description = "LLM-judge calibration on the solve items"
    default = True
    result_type = QuantJudgeResult
    AGENT_CALLS_PER_ITEM = 5  # 2 tutor probes + up to 3 solves
    JUDGE_CALLS_PER_ITEM = 7

    @staticmethod
    def _solve_items() -> list[dict[str, Any]]:
        return [i for i in load_golden_set("quant") if i["kind"] == "solve"]

    def plan(self, ctx: PlanContext) -> SuitePlan:
        n = min(ctx.args.max_items, len(self._solve_items()))
        return SuitePlan(
            items=n, agent_calls=n * self.AGENT_CALLS_PER_ITEM,
            judge_calls=n * self.JUDGE_CALLS_PER_ITEM,
        )

    async def run(self, ctx: RunContext) -> QuantJudgeResult:
        agent = runner.build_quant_agent(
            model_config_path=runner.write_model_config(ctx.backend.model),
            retrieve_corpus=quant_suite.NotesCorpus([]),
            math_tools_enabled=ctx.args.math_tools == "on",
        )
        solve_items = self._solve_items()
        rate = min(1.0, ctx.args.max_items / len(solve_items))
        items = sample_golden_set(solve_items, rate, seed=SEED)
        return await runner.run_quant(items, agent, ctx.judge)

    def metrics(self, result: QuantJudgeResult) -> dict[str, float]:
        no_leak = (
            None if result.tutor.leak_rate_rule is None else 1 - result.tutor.leak_rate_rule
        )
        pairs = {
            "quant_solve_accuracy": result.solve.accuracy,
            "quant_tutor_no_leak_rate": no_leak,
            "judge_leak_agreement": result.judge_calibration.leak_labels.agreement,
            "judge_correctness_agreement": result.judge_calibration.correctness_labels.agreement,
        }
        return {k: v for k, v in pairs.items() if v is not None}


class AwSuite:
    name = "aw"
    description = "judge calibration against the GRE AW anchor essays"
    default = True
    result_type = AwResult
    JUDGE_CALLS_PER_ITEM = 3

    def plan(self, ctx: PlanContext) -> SuitePlan:
        n = min(ctx.args.max_items, len(load_golden_set("aw")))
        return SuitePlan(items=n, judge_calls=n * self.JUDGE_CALLS_PER_ITEM)

    async def run(self, ctx: RunContext) -> AwResult:
        items = runner.select_items("aw", ctx.args.max_items, SEED)
        return await runner.run_aw(items, ctx.judge)

    def metrics(self, result: AwResult) -> dict[str, float]:
        if result.within_tolerance is None:
            return {}
        return {"judge_aw_within_tolerance": result.within_tolerance}


class GenerationSuite:
    name = "generation"
    description = "the LangGraph question pipeline against the single-call baseline"
    default = True
    result_type = GenerationResult
    AGENT_CALLS_PER_ITEM = 1 + 7  # baseline 1; pipeline planner + 3 x (generate + critic)
    JUDGE_CALLS_PER_ITEM = 2  # one verdict per emitted item

    def plan(self, ctx: PlanContext) -> SuitePlan:
        n = min(ctx.args.max_items, len(load_golden_set("quant_generation")))
        return SuitePlan(
            items=n, agent_calls=n * self.AGENT_CALLS_PER_ITEM,
            judge_calls=n * self.JUDGE_CALLS_PER_ITEM,
        )

    async def run(self, ctx: RunContext) -> GenerationResult:
        """Baseline vs LangGraph pipeline, both routed to the backend's model, same requests."""
        config = runner.write_model_config(ctx.backend.model)
        verify_math = runner.in_process_verify_math()
        requests = [
            GenerationRequest.model_validate(i)
            for i in runner.select_items("quant_generation", ctx.args.max_items, SEED)
        ]
        targets: dict[str, generation.Target] = {
            "baseline": build_baseline(config),
            "pipeline": build_pipeline(verify_math, config),
        }
        return await generation.run_generation(
            requests, targets, ctx.judge, verify_math, ctx.transport
        )

    def metrics(self, result: GenerationResult) -> dict[str, float]:
        out: dict[str, float] = {}
        pipe = result.targets.get("pipeline")
        if pipe is not None and pipe.valid_yield is not None:
            out["gen_pipeline_valid_yield"] = pipe.valid_yield
        if pipe is not None and pipe.shipped_defect_rate is not None:
            out["gen_pipeline_no_defect_rate"] = 1 - pipe.shipped_defect_rate
        return out


class IeltsWritingSuite:
    name = "ielts_writing"
    description = "the IELTS Writing grader against labelled essays (needs local data)"
    default = False
    result_type = IeltsWritingResult

    def plan(self, ctx: PlanContext) -> SuitePlan:
        args = ctx.args
        _, picked, _ = ielts_writing.select_items(
            ctx.ielts_items, args.ielts_tier, args.max_items, SEED
        )
        # an essay costs `samples` grader calls
        return SuitePlan(items=len(picked), judge_calls=len(picked) * args.samples)

    async def run(self, ctx: RunContext) -> IeltsWritingResult:
        args = ctx.args
        return await ielts_writing.run_ielts_writing(
            ctx.ielts_items, ctx.backend.ielts_grader(args.samples), tier=args.ielts_tier,
            max_items=args.max_items, seed=SEED,
        )

    def metrics(self, result: IeltsWritingResult) -> dict[str, float]:
        if not (
            result.n_items and result.overall is not None
            and result.overall.within_tolerance is not None
        ):
            return {}
        # keyed by tier so a weak-label baseline is never compared with an examiner one
        return {
            f"ielts_overall_within_half_band__{result.label_tier}":
                result.overall.within_tolerance
        }


class IeltsReportSuite:
    name = "ielts_report"
    description = "the IELTS feedback report checks on labelled essays (needs local data)"
    default = False
    result_type = IeltsReportResult

    def plan(self, ctx: PlanContext) -> SuitePlan:
        args = ctx.args
        _, picked, _ = ielts_writing.select_items(
            ctx.ielts_items, args.ielts_tier, args.max_items, SEED
        )
        n = min(len(picked), ielts_writing.REPORT_CHECK_ITEMS)
        return SuitePlan(items=n, judge_calls=n * (args.samples + 1))  # +1 feedback call

    async def run(self, ctx: RunContext) -> IeltsReportResult:
        args = ctx.args
        return await ielts_writing.run_ielts_report(
            ctx.ielts_items, ctx.backend.ielts_completer(), tier=args.ielts_tier,
            max_items=min(args.max_items, ielts_writing.REPORT_CHECK_ITEMS), seed=SEED,
            samples=args.samples,
        )

    def metrics(self, result: IeltsReportResult) -> dict[str, float]:
        if not result.n_items:
            return {}
        return {"ielts_report_sound_rate": 1 - result.reports_with_problems / result.n_items}


class IeltsAdversarialSuite:
    name = "ielts_adversarial"
    description = "the IELTS marker against adversarial essays (injection, off-topic, gibberish)"
    default = False
    result_type = IeltsAdversarialResult

    def plan(self, ctx: PlanContext) -> SuitePlan:
        n = sum(1 for i in ctx.ielts_items if i.task_type == "adversarial_response")
        return SuitePlan(items=n, judge_calls=n * (ctx.args.samples + 1))

    async def run(self, ctx: RunContext) -> IeltsAdversarialResult:
        return await ielts_writing.run_ielts_adversarial(
            ctx.ielts_items, ctx.backend.ielts_completer(), samples=ctx.args.samples
        )

    def metrics(self, result: IeltsAdversarialResult) -> dict[str, float]:
        if not result.n_cases:
            return {}
        return {"ielts_adversarial_pass_rate": result.passed / result.n_cases}


# Run order, plan order and the order of the `--suite` choices.
SUITES: tuple[Suite[Any], ...] = (
    QuantSuite(), QuantJudgeSuite(), AwSuite(), GenerationSuite(),
    IeltsWritingSuite(), IeltsReportSuite(), IeltsAdversarialSuite(),
)


def suite_names() -> list[str]:
    return [s.name for s in SUITES]


def default_suite_names() -> list[str]:
    return [s.name for s in SUITES if s.default]


def selected_suites(args: argparse.Namespace) -> list[Suite[Any]]:
    """The suites `--suite` named, in run order (not in the order they were typed)."""
    return [s for s in SUITES if s.name in args.suite]


def regression_metrics(report: EvalReport) -> dict[str, float]:
    """Higher-is-better metrics suitable for check_regression, from every suite that ran."""
    out: dict[str, float] = {}
    for suite in SUITES:
        result = report.suite(suite.name, suite.result_type)
        if result is not None:
            out.update(suite.metrics(result))
    return out
