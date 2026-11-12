"""
The failure-seeded Quant suite: `python -m evals agent --suite quant`.

Every item in `evals/golden/gre_quant.jsonl` comes from a failure EduAgent really had (its
`failure_source` names the commit, test or prompt line) or, where none is recorded yet, says
so ("none yet") and works as a capability probe. Items have a `kind` that decides how they run:

  solve            the real QuantAgent in solve mode; pass = the final answer is right (sympy,
                   never the agent's own claim) AND the product's checks accepted it (the answer
                   is well-formed, and the independent check did not disagree with it)
  tutor            one real tutor reply; pass = no answer leak, no LaTeX, citations written as
                   [N] (never "your notes"), and a concrete next step (see `graders`)
  tutor_multiturn  a scripted student talks to the real tutor for several turns; pass = every
                   reply passes the tutor checks AND the last reply still mentions the problem
  security         hostile text where a model's final answer goes; pass = the verifier rejects
                   it and computes nothing. No model is called, so it is exact and free.

Each model-driven item runs `samples` times. Results are reported as pass@k, pass^k and the mean
pass rate, always with n and a 95% interval (see `stats`), plus cost and latency per turn read
from the same `TurnMeter` records the live chat path writes. Graders are rules, not a judge.
"""
from __future__ import annotations

import os
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import sympy as sp
from pydantic import BaseModel

from evals.agent_eval import graders, stats
from evals.agent_eval.probes import answer_values, normalize_for_type
from evals.agent_eval.results import (
    BaselineMetric,
    CheckRate,
    Control,
    GroupSummary,
    ObservedWrong,
    QuantAttempt,
    QuantBaseline,
    QuantRows,
    QuantSuiteResult,
    RunMeta,
    SecurityBlock,
    SeededWrongAnswers,
    SelfCompare,
    TranscriptMessage,
    TurnCost,
    TurnUsage,
    VerdictTally,
    WrongButVerified,
)
from evals.agent_eval.transport import AllModelsFailedError, cache_slot, gather_limited, tagged
from evals.judges.quant_judge import grade_quant_item
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.answers import extract_final_answer, normalize_answer
from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk, RetrieveCorpus
from services.agent.src.agent.capabilities.verify_math import VerifyMath
from shared.src.shared.budget import estimate_cost
from shared.src.shared.events import IndependentCheck, Token, ToolTrace, Verified
from shared.src.shared.models import MessageRole, Mode, Session
from shared.src.shared.stats import percentile
from shared.src.shared.turn_meter import TurnLog

PRICED_AS = "openai/gpt-4o-mini"
MODEL_KINDS = ("solve", "tutor", "tutor_multiturn")
BASELINE_SCHEMA = "eval-baseline/1"
AgentFactory = Callable[[list[str]], QuantAgent]


class NotesCorpus(RetrieveCorpus):
    """Serves an item's own reference notes in place of the corpus service (none: no notes)."""

    def __init__(self, notes: list[str]) -> None:
        self._notes = notes

    async def retrieve(
        self, query: str, candidate_id: str, school_id: str, test_type: str, top_k: int = 5
    ) -> list[CorpusChunk]:
        return [
            CorpusChunk(text=text, tier="global", source_uri="eval", page_or_section="note",
                        score=1.0)
            for text in self._notes[:top_k]
        ]


class TurnCollector(TurnLog):
    """An in-memory turn log: the agent's own `TurnMeter` writes here during an eval."""

    def __init__(self) -> None:
        super().__init__(Path(os.devnull))
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def read(self) -> list[dict[str, Any]]:
        return list(self.records)

    def turn(self, trace_id: str) -> TurnUsage:
        """Cost, tokens and time of one turn, from the record its meter wrote."""
        recs = [r for r in self.records if r.get("trace_id") == trace_id]
        return TurnUsage(
            usd=sum(r.get("cost_usd", 0.0) for r in recs),
            input_tokens=sum(r.get("input_tokens", 0) for r in recs),
            output_tokens=sum(r.get("output_tokens", 0) for r in recs),
            llm_calls=sum(r.get("llm_calls", 0) for r in recs),
            tool_calls=sum(len(r.get("tool_calls", [])) for r in recs),
            tool_errors=sum(r.get("tool_errors", 0) for r in recs),
            tool_budget_exhausted=any(r.get("tool_budget_exhausted") for r in recs),
            latency_ms=sum(r.get("total_ms", 0.0) for r in recs),
            estimated=any(r.get("usage") in ("estimate", "mixed") for r in recs),
            notes_retrieved=sum(r.get("notes_retrieved", 0) for r in recs),
        )


@contextmanager
def collecting_turns() -> Iterator[TurnCollector]:
    """Route the agent's per-turn measurement records into a collector for the duration."""
    collector = TurnCollector()
    with patch("shared.src.shared.turn_meter.default_turn_log", return_value=collector):
        yield collector


# ── running one attempt ──────────────────────────────────────────────────────


def first_message(item: dict[str, Any]) -> str:
    student = item["turns"][0] if item["kind"] == "tutor_multiturn" else item.get("student", "")
    return f"Here is my question: {item['question']}\n{student}".strip()


async def _converse(
    factory: AgentFactory,
    item: dict[str, Any],
    messages: list[str],
    mode: Mode,
    trace_prefix: str,
    collector: TurnCollector,
) -> tuple[list[TranscriptMessage], list[TurnUsage], bool | None, str | None]:
    """
    Send `messages` in order to a fresh agent; returns the transcript, per-turn usage, the
    verifier's verdict (`verifier_fail`) and the independent check's verdict of the last turn.
    """
    agent = factory(item.get("notes", []))
    session = Session(candidate_id="eval", test_type="gre", section="quant", mode=mode)
    transcript: list[TranscriptMessage] = []
    usage: list[TurnUsage] = []
    verifier_fail: bool | None = None
    independent: str | None = None
    for n, text in enumerate(messages):
        trace_id = f"{trace_prefix}-t{n}"
        parts: list[str] = []
        tool_lines: list[TranscriptMessage] = []
        with tagged("agent"):
            async for out in agent.run(session, text, trace_id=trace_id):
                match out:
                    case Token(text=piece):
                        parts.append(piece)
                    case Verified(passed=passed):
                        verifier_fail = not passed
                    case IndependentCheck(verdict=verdict):
                        independent = verdict
                    case ToolTrace(calls=calls):
                        tool_lines += [
                            TranscriptMessage(
                                role="tool_call", content=f"{c.name}({c.arguments}) -> {c.result}"
                            )
                            for c in calls
                        ]
                    case _:
                        pass
        reply = "".join(parts)
        transcript += [TranscriptMessage(role="student", content=text), *tool_lines,
                       TranscriptMessage(role="tutor", content=reply)]
        usage.append(collector.turn(trace_id))
        session = session.append_message(MessageRole.candidate, text).append_message(
            MessageRole.agent, reply
        )
    return transcript, usage, verifier_fail, independent


def _row(
    item: dict[str, Any], attempt: int, checks: dict[str, bool],
    transcript: list[TranscriptMessage], usage: list[TurnUsage],
    *, answer: str | None = None, independent_check: str | None = None,
) -> QuantAttempt:
    return QuantAttempt(
        item_id=item["id"], kind=item["kind"], tag=item["tag"], attempt=attempt,
        passed=all(checks.values()), checks=checks, transcript=transcript, turns=usage,
        answer=answer, independent_check=independent_check,
    )


async def run_attempt(
    item: dict[str, Any], attempt: int, factory: AgentFactory, collector: TurnCollector,
    run_prefix: str,
) -> QuantAttempt:
    prefix = f"{run_prefix}-{item['id']}-a{attempt}"
    kind = item["kind"]
    if kind == "solve":
        transcript, usage, verifier_fail, independent = await _converse(
            factory, item, [item["question"]], Mode.solve, prefix, collector
        )
        raw = extract_final_answer(transcript[-1].content)
        answer = normalize_for_type(
            normalize_answer(raw) if raw else "", item["question_type"]
        )
        graded = await grade_quant_item(item, candidate_answer=answer)
        checks = {"correct": graded.correct, "verifier_accepted": verifier_fail is False}
        return _row(
            item, attempt, checks, transcript, usage, answer=answer, independent_check=independent
        )
    messages = (
        [first_message(item), *item["turns"][1:]] if kind == "tutor_multiturn"
        else [first_message(item)]
    )
    transcript, usage, _, _ = await _converse(
        factory, item, messages, Mode.tutor, prefix, collector
    )
    replies = [m.content for m in transcript if m.role == "tutor"]
    per_reply = [graders.grade_tutor_reply(r, item) for r in replies]
    checks = {name: all(c[name] for c in per_reply) for name in per_reply[0]}
    if kind == "tutor_multiturn":
        checks["stays_on_problem"] = graders.stays_on_problem(replies[-1], item["anchors"])
    return _row(item, attempt, checks, transcript, usage)


async def run_security_case(item: dict[str, Any], verify_math: VerifyMath) -> QuantAttempt:
    """Hostile text as a model's final answer, through the verifier the solve turn uses."""
    text = item.get("hostile") or item["hostile_repeat"][0] * item["hostile_repeat"][1]
    result = await verify_math.verify(expression=text, expected="1")
    checks = {
        "rejected": not result.verified and result.computed == "" and bool(result.error),
        "process_id_not_returned": str(os.getpid()) not in result.computed,
    }
    return QuantAttempt(
        item_id=item["id"], kind="security", tag=item["tag"], attempt=0,
        passed=all(checks.values()), checks=checks,
        transcript=[TranscriptMessage(role="hostile_input", content=text[:200]),
                    TranscriptMessage(role="verifier", content=f"error={result.error!r}")],
        turns=[],
    )


# ── wrong_but_verified: does the product's independent check accept a wrong answer? ──


def seeded_wrong_answer(item: dict[str, Any]) -> str | None:
    """
    A deterministic, plausible wrong answer for a solve item: every value of the golden key
    shifted by 1, or a different verdict letter for a quantitative comparison. None when a
    key value cannot be read as a number.
    """
    if item["question_type"] == "quantitative_comparison":
        return "B" if str(item["ground_truth"]).strip().upper() == "A" else "A"
    shifted = []
    for value in answer_values(str(item["ground_truth"])):
        try:
            shifted.append(str(sp.Rational(value.replace(" ", "")) + 1))
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    return ",".join(shifted) or None


def stub_expression(item: dict[str, Any]) -> str | None:
    """
    What the model writes for the independent check, with no model involved: the item's own
    `expression`, or None for a quantitative comparison (its answer is a letter, so the real
    prompt tells the model to write NONE). The offline suite uses it as its fake model's reply,
    so the seeded measurement exercises the product's comparison and nothing else: it says
    nothing about how well a real model writes the equation.
    """
    if item["question_type"] == "quantitative_comparison":
        return None
    return str(item["expression"]) or None


def _is_wrong_but_verified(row: QuantAttempt) -> bool:
    return not row.checks["correct"] and row.independent_check == "agrees"


def _solve_reply(agent: QuantAgent, answer: str) -> tuple[str, str]:
    """A solve reply that ends in `answer`, and the answer the agent reads back out of it."""
    text = f"1. Work.\n**Answer: {answer}**"
    return text, agent.extract_final_answer(text) or ""


async def _verdicts(
    items: list[dict[str, Any]], agent: QuantAgent, answer_of: Callable[[dict[str, Any]], str]
) -> dict[str, str]:
    """The independent check's verdict per item id, for the answer `answer_of(item)`."""
    out: dict[str, str] = {}
    for item in items:
        _, claimed = _solve_reply(agent, answer_of(item))
        out[item["id"]] = (await agent.check_independently(stub_expression(item), claimed)).verdict
    return out


def _tally(verdicts: dict[str, str]) -> VerdictTally:
    return VerdictTally(
        agrees=sum(1 for x in verdicts.values() if x == "agrees"),
        disagrees=sum(1 for x in verdicts.values() if x == "disagrees"),
        not_checkable=sum(1 for x in verdicts.values() if x == "not_checkable"),
    )


async def wrong_but_verified(
    items: list[dict[str, Any]], agent: QuantAgent, rows: list[QuantAttempt]
) -> WrongButVerified:
    """
    How often the product's independent check agrees with an answer the golden key says is wrong.

    `seeded` needs no model: each solve item gets a wrong answer (`seeded_wrong_answer`), which
    sympy confirms against the golden key is wrong, and the answer goes through the same
    extract-then-`check_independently` step a solve turn runs, with `stub_expression` as the
    model's expression. `self_compare` is the same answers through `verify_well_formed`, the
    well-formed check the live path ran alone before the independent check existed (it compares
    an answer with itself, so it was near 100%): the "before" for the same items. `control`
    runs the golden key through the independent check, to show it does not simply reject
    everything.
    `observed` counts real solve attempts in `rows` that were wrong and still got "agrees".
    "not_checkable" is never counted as verified: the learner is told it was not checked.
    """
    seeded_items: list[dict[str, Any]] = []
    wrong_answers: dict[str, str] = {}
    for item in items:
        wrong = seeded_wrong_answer(item) if item["kind"] == "solve" else None
        if wrong is None or (await grade_quant_item(item, candidate_answer=wrong)).correct:
            continue
        seeded_items.append(item)
        wrong_answers[item["id"]] = wrong
    verdicts = await _verdicts(seeded_items, agent, lambda i: wrong_answers[i["id"]])
    well_formed: list[str] = []
    for item in seeded_items:
        text, claimed = _solve_reply(agent, wrong_answers[item["id"]])
        if (await agent.verify_well_formed(text, claimed)).verified:
            well_formed.append(item["id"])
    control = await _verdicts(seeded_items, agent, lambda i: str(i["ground_truth"]))
    accepted = sorted(i for i, v in verdicts.items() if v == "agrees")
    solve_rows = [r for r in rows if r.kind == "solve"]
    wrong_rows = [r for r in solve_rows if not r.checks["correct"]]
    count = sum(_is_wrong_but_verified(r) for r in wrong_rows)
    return WrongButVerified(
        definition="a wrong answer (per the golden key, graded by sympy) that the product's "
                   "independent check agreed with, so the reply says \"Independent check: "
                   "agrees\"",
        seeded=SeededWrongAnswers(
            **stats.rate(len(accepted), len(seeded_items)).model_dump(), count=len(accepted),
            **_tally(verdicts).model_dump(), accepted_items=accepted,
        ),
        self_compare=SelfCompare(
            **stats.rate(len(well_formed), len(seeded_items)).model_dump(),
            count=len(well_formed), accepted_items=sorted(well_formed),
        ),
        control=Control(n=len(control), **_tally(control).model_dump()),
        observed=ObservedWrong(
            **stats.rate(count, len(wrong_rows)).model_dump(), count=count,
            solve_attempts=len(solve_rows),
            not_checkable=sum(r.independent_check == "not_checkable" for r in wrong_rows),
        ),
    )


def wrong_but_verified_caveats(result: WrongButVerified) -> list[str]:
    seeded, observed = result.seeded, result.observed
    return [
        f"wrong_but_verified: {seeded.count} of {seeded.n} seeded wrong answers got "
        f"\"agrees\" from the independent check ({seeded.disagrees} disagreed, "
        f"{seeded.not_checkable} not checkable); the well-formed check alone accepted "
        f"{result.self_compare.count} of {seeded.n}. "
        f"{observed.count} of {observed.solve_attempts} real solve attempts were wrong "
        "and still got \"agrees\". The seeded run uses each item's own expression as the model's "
        "expression, so it tests the comparison, not how well a model writes the equation; the "
        "seeded answers are one perturbation of each key, not every way a model can be wrong. "
        "The check covers numeric answers only (ADR-0012), and a model that misreads a problem "
        "the same way in its working and in its equation still agrees with itself.",
    ]


# ── the suite ────────────────────────────────────────────────────────────────


def _pct(values: list[float], q: float) -> float | None:
    p = percentile(values, q)
    return None if p is None else round(p, 4)


def _cost_block(rows: list[QuantAttempt]) -> TurnCost:
    turns = [t for r in rows for t in r.turns]
    if not turns:
        return TurnCost(turns=0)
    usd = [t.usd for t in turns]
    latency = [t.latency_ms for t in turns]
    tool_calls = sum(t.tool_calls for t in turns)
    return TurnCost(
        turns=len(turns),
        usd_total=round(sum(usd), 6),
        usd_per_turn_mean=round(statistics.fmean(usd), 6),
        usd_per_turn_p95=_pct(usd, 0.95),
        # The tokens priced as the product's default paid model, so a free-model run still
        # gives a cost to compare later slices against (see shared.budget._PRICES).
        usd_per_turn_if_gpt_4o_mini=round(
            statistics.fmean(
                estimate_cost(PRICED_AS, t.input_tokens, t.output_tokens) for t in turns
            ), 6,
        ),
        llm_calls_per_turn_mean=round(statistics.fmean(t.llm_calls for t in turns), 2),
        tool_calls_per_turn_mean=round(statistics.fmean(t.tool_calls for t in turns), 2),
        turns_with_tool_calls=sum(1 for t in turns if t.tool_calls),
        tool_error_rate=(
            round(sum(t.tool_errors for t in turns) / tool_calls, 4) if tool_calls else None
        ),
        turns_tool_budget_exhausted=sum(1 for t in turns if t.tool_budget_exhausted),
        input_tokens_per_turn_mean=round(statistics.fmean(t.input_tokens for t in turns), 1),
        output_tokens_per_turn_mean=round(statistics.fmean(t.output_tokens for t in turns), 1),
        latency_ms_p50=_pct(latency, 0.5),
        latency_ms_p95=_pct(latency, 0.95),
        turns_with_estimated_tokens=sum(1 for t in turns if t.estimated),
        turns_with_notes=sum(1 for t in turns if t.notes_retrieved),
    )


def _check_rates(rows: list[QuantAttempt]) -> dict[str, CheckRate]:
    totals: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for name, ok in r.checks.items():
            totals[name].append(ok)
    return {
        name: CheckRate(passed=sum(v), total=len(v), value=round(sum(v) / len(v), 4))
        for name, v in sorted(totals.items())
    }


def summarise(rows: list[QuantAttempt], samples: int, dropped: int, total: int) -> QuantSuiteResult:
    model_rows = [r for r in rows if r.kind in MODEL_KINDS]
    groups: dict[str, GroupSummary] = {}
    for kind in MODEL_KINDS:
        by_item: dict[str, list[bool]] = defaultdict(list)
        for r in model_rows:
            if r.kind == kind:
                by_item[r.item_id].append(r.passed)
        if by_item:
            kind_rows = [r for r in model_rows if r.kind == kind]
            groups[kind] = GroupSummary(
                **stats.group_summary(by_item, samples).model_dump(),
                checks=_check_rates(kind_rows),
                cost_latency=_cost_block(kind_rows),
            )
    by_tag: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in model_rows:
        by_tag[r.tag][r.item_id].append(r.passed)
    security_rows = [r for r in rows if r.kind == "security"]
    out = QuantSuiteResult(
        samples=samples,
        n_items=len({r.item_id for r in rows}),
        groups=groups,
        by_tag={t: stats.group_summary(v, samples) for t, v in sorted(by_tag.items())},
        all_model_turns=_cost_block(model_rows),
        units_total=total,
        units_dropped=dropped,
        rows=QuantRows(attempts=rows),
    )
    if security_rows:
        out.security = SecurityBlock(
            **stats.rate(sum(r.passed for r in security_rows), len(security_rows)).model_dump(),
            failed_items=sorted(r.item_id for r in security_rows if not r.passed),
        )
    out.caveats = caveats(out)
    return out


def caveats(suite: QuantSuiteResult) -> list[str]:
    """Plain statements of what this run's sample size and method cannot support."""
    notes = [
        "Rates are over items, not over all possible learner questions; the items are "
        "self-written from past failures and are not a random sample.",
        "Cost and latency cover the agent's own model calls only; the gateway's guardrail "
        "calls (completeness check, answer-leak check) are not included.",
        "The leak check is an upper bound: it also counts a step number equal to the answer.",
        "Rule graders catch the failures someone wrote a rule for; a reply can pass every rule "
        "and still be a poor hint (read the transcripts).",
    ]
    for kind, group in suite.groups.items():
        half = stats.resolution(group)
        notes.append(
            f"{kind}: n={group.n_items} items x {group.samples} samples; a change smaller "
            f"than about {0 if half is None else half:.2f} cannot be told apart from noise."
        )
    est = suite.all_model_turns.turns_with_estimated_tokens
    if est:
        notes.append(f"{est} turns had no provider token report; their cost is an estimate.")
    if suite.security is not None and suite.security.n < 30:
        notes.append(
            f"security: {suite.security.n} hostile strings; all passing does not bound "
            "the failure rate on strings nobody has tried."
        )
    return notes


async def run_quant_suite(
    items: list[dict[str, Any]],
    *,
    agent_factory: AgentFactory,
    verify_math: VerifyMath,
    samples: int,
    collector: TurnCollector,
    run_prefix: str = "eval",
) -> QuantSuiteResult:
    """Run every item (model-driven ones `samples` times) and summarise."""
    dropped: list[str] = []

    async def guarded(item: dict[str, Any], attempt: int) -> QuantAttempt | None:
        try:
            with cache_slot(attempt):
                return await run_attempt(item, attempt, agent_factory, collector, run_prefix)
        except AllModelsFailedError as exc:
            dropped.append(str(exc))
            return None

    model_items = [i for i in items if i["kind"] in MODEL_KINDS]
    jobs = [guarded(i, a) for i in model_items for a in range(samples)]
    done = [r for r in await gather_limited(jobs) if r is not None]
    security = [await run_security_case(i, verify_math) for i in items if i["kind"] == "security"]
    total = len(jobs) + len(security)
    suite = summarise([*done, *security], samples, len(dropped), total)
    if any(i["kind"] == "solve" for i in items):
        suite.wrong_but_verified = await wrong_but_verified(items, agent_factory([]), done)
        suite.caveats += wrong_but_verified_caveats(suite.wrong_but_verified)
    return suite


# ── baseline file and comparison ─────────────────────────────────────────────


def headline_metrics(suite: QuantSuiteResult) -> dict[str, BaselineMetric]:
    """The numbers a baseline pins, each with n and its 95% interval. Higher is better."""
    out: dict[str, BaselineMetric] = {}
    for kind, group in suite.groups.items():
        out[f"{kind}.mean_pass_rate"] = BaselineMetric(
            value=group.mean_pass_rate, n=group.n_items,
            ci95=group.mean_pass_rate_ci95, samples=group.samples,
        )
        out[f"{kind}.pass_hat_k"] = BaselineMetric(
            value=group.pass_hat_k.value, n=group.n_items,
            ci95=group.pass_hat_k.ci95, samples=group.samples,
        )
    if suite.security is not None:
        out["security.pass_rate"] = BaselineMetric(
            value=suite.security.value, n=suite.security.n, ci95=suite.security.ci95, samples=1,
        )
    return out


def baseline_document(suite: QuantSuiteResult, meta: RunMeta) -> QuantBaseline:
    """The checked-in baseline: aggregates only (no model text), with the run's provenance."""
    note = (
        "Measured on the model named in meta.model. "
        + ("" if meta.is_product_default_model else
           f"That is NOT the product default ({meta.product_default_model}); "
           "re-run with the default model before comparing a change against these numbers. ")
        + "Every rate carries its n and 95% interval; a difference inside an interval is noise."
    )
    return QuantBaseline(
        schema=BASELINE_SCHEMA, suite="quant", meta=meta, note=note,
        metrics=headline_metrics(suite), summary=suite.model_copy(update={"rows": None}),
    )


DETERMINISTIC_METRICS = frozenset({"security.pass_rate"})


class MetricComparison(BaseModel):
    """One baseline metric against the current run."""

    metric: str
    baseline: float
    baseline_n: int
    baseline_ci95: list[float] | None
    current: float
    current_n: int
    verdict: str


def compare_with_baseline(
    suite: QuantSuiteResult, baseline: QuantBaseline
) -> list[MetricComparison]:
    """
    One row per baseline metric: baseline value, current value, and a verdict that is
    "regression"/"improvement" only when the current value falls outside the interval the
    baseline carries. The security rate involves no model and no sampling, so any drop is a
    regression and there is no noise band. pass^k is skipped when k differs, since it is not
    comparable.
    """
    current = headline_metrics(suite)
    rows = []
    for name, base in baseline.metrics.items():
        now = current.get(name)
        if now is None or now.value is None or base.value is None:
            continue
        if base.samples != now.samples and name.endswith("pass_hat_k"):
            continue
        exact = name in DETERMINISTIC_METRICS
        verdict_ci = [base.value, base.value] if exact else base.ci95
        rows.append(MetricComparison(
            metric=name, baseline=base.value, baseline_n=base.n, baseline_ci95=base.ci95,
            current=now.value, current_n=now.n,
            verdict=stats.verdict_vs_baseline(now.value, verdict_ci),
        ))
    return rows
