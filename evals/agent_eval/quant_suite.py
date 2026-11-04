"""
The failure-seeded Quant suite: `python -m evals.agent_eval --suite quant`.

Every item in `evals/golden/gre_quant.jsonl` comes from a failure EduAgent really had (its
`failure_source` names the commit, test or prompt line) or, where none is recorded yet, says
so ("none yet") and works as a capability probe. Items have a `kind` that decides how they run:

  solve            the real QuantAgent in solve mode; pass = the final answer is right (sympy,
                   never the agent's own claim) AND the product's verifier accepted it
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

from evals.agent_eval import graders, stats
from evals.agent_eval.probes import normalize_for_type
from evals.agent_eval.transport import AllModelsFailedError, cache_slot, gather_limited, tagged
from evals.judges.quant_judge import grade_quant_item
from evals.turn_report import percentile
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.answers import extract_final_answer, normalize_answer
from services.agent.src.agent.skills.retrieve_corpus import CorpusChunk, RetrieveCorpusSkill
from services.agent.src.agent.skills.verify_math import VerifyMathSkill
from shared.src.shared.budget import estimate_cost
from shared.src.shared.models import MessageRole, Mode, Session
from shared.src.shared.turn_meter import TurnLog

PRICED_AS = "openai/gpt-4o-mini"
MODEL_KINDS = ("solve", "tutor", "tutor_multiturn")
BASELINE_SCHEMA = "eval-baseline/1"
AgentFactory = Callable[[list[str]], QuantAgent]


class NotesSkill(RetrieveCorpusSkill):
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

    def turn(self, trace_id: str) -> dict[str, Any]:
        """Cost, tokens and time of one turn, from the record its meter wrote."""
        recs = [r for r in self.records if r.get("trace_id") == trace_id]
        return {
            "usd": sum(r.get("cost_usd", 0.0) for r in recs),
            "input_tokens": sum(r.get("input_tokens", 0) for r in recs),
            "output_tokens": sum(r.get("output_tokens", 0) for r in recs),
            "llm_calls": sum(r.get("llm_calls", 0) for r in recs),
            "tool_calls": sum(len(r.get("tool_calls", [])) for r in recs),
            "tool_errors": sum(r.get("tool_errors", 0) for r in recs),
            "tool_budget_exhausted": any(r.get("tool_budget_exhausted") for r in recs),
            "latency_ms": sum(r.get("total_ms", 0.0) for r in recs),
            "estimated": any(r.get("usage") in ("estimate", "mixed") for r in recs),
        }


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
) -> tuple[list[dict[str, str]], list[dict[str, Any]], bool | None]:
    """Send `messages` in order to a fresh agent; returns transcript, per-turn usage, verifier."""
    agent = factory(item.get("notes", []))
    session = Session(candidate_id="eval", test_type="gre", section="quant", mode=mode)
    transcript: list[dict[str, str]] = []
    usage: list[dict[str, Any]] = []
    verifier_fail: bool | None = None
    for n, text in enumerate(messages):
        trace_id = f"{trace_prefix}-t{n}"
        parts: list[str] = []
        tool_lines: list[dict[str, str]] = []
        with tagged("agent"):
            async for out in agent.run(session, text, trace_id=trace_id):
                if isinstance(out, str):
                    parts.append(out)
                else:
                    verifier_fail = out.get("verifier_fail", verifier_fail)
                    tool_lines += [
                        {"role": "tool_call", "content": f"{c['name']}({c['arguments']}) -> "
                         f"{c['result']}"}
                        for c in out.get("tool_trace", [])
                    ]
        reply = "".join(parts)
        transcript += [{"role": "student", "content": text}, *tool_lines,
                       {"role": "tutor", "content": reply}]
        usage.append(collector.turn(trace_id))
        session = session.append_message(MessageRole.candidate, text).append_message(
            MessageRole.agent, reply
        )
    return transcript, usage, verifier_fail


def _row(
    item: dict[str, Any], attempt: int, checks: dict[str, bool], transcript: list[dict[str, str]],
    usage: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "item_id": item["id"], "kind": item["kind"], "tag": item["tag"], "attempt": attempt,
        "passed": all(checks.values()), "checks": checks, "transcript": transcript,
        "turns": usage,
    }


async def run_attempt(
    item: dict[str, Any], attempt: int, factory: AgentFactory, collector: TurnCollector,
    run_prefix: str,
) -> dict[str, Any]:
    prefix = f"{run_prefix}-{item['id']}-a{attempt}"
    kind = item["kind"]
    if kind == "solve":
        transcript, usage, verifier_fail = await _converse(
            factory, item, [item["question"]], Mode.solve, prefix, collector
        )
        raw = extract_final_answer(transcript[-1]["content"])
        answer = normalize_for_type(
            normalize_answer(raw) if raw else "", item["question_type"]
        )
        graded = await grade_quant_item(item, candidate_answer=answer)
        checks = {"correct": graded.correct, "verifier_accepted": verifier_fail is False}
        row = _row(item, attempt, checks, transcript, usage)
        row["answer"] = answer
        return row
    messages = (
        [first_message(item), *item["turns"][1:]] if kind == "tutor_multiturn"
        else [first_message(item)]
    )
    transcript, usage, _ = await _converse(factory, item, messages, Mode.tutor, prefix, collector)
    replies = [m["content"] for m in transcript if m["role"] == "tutor"]
    per_reply = [graders.grade_tutor_reply(r, item) for r in replies]
    checks = {name: all(c[name] for c in per_reply) for name in per_reply[0]}
    if kind == "tutor_multiturn":
        checks["stays_on_problem"] = graders.stays_on_problem(replies[-1], item["anchors"])
    return _row(item, attempt, checks, transcript, usage)


async def run_security_case(item: dict[str, Any], skill: VerifyMathSkill) -> dict[str, Any]:
    """Hostile text as a model's final answer, through the verifier the solve turn uses."""
    text = item.get("hostile") or item["hostile_repeat"][0] * item["hostile_repeat"][1]
    result = await skill.verify(expression=text, expected="1")
    checks = {
        "rejected": not result.verified and result.computed == "" and bool(result.error),
        "process_id_not_returned": str(os.getpid()) not in result.computed,
    }
    return {
        "item_id": item["id"], "kind": "security", "tag": item["tag"], "attempt": 0,
        "passed": all(checks.values()), "checks": checks,
        "transcript": [{"role": "hostile_input", "content": text[:200]},
                       {"role": "verifier", "content": f"error={result.error!r}"}],
        "turns": [],
    }


# ── the suite ────────────────────────────────────────────────────────────────


def _pct(values: list[float], q: float) -> float | None:
    p = percentile(values, q)
    return None if p is None else round(p, 4)


def _cost_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [t for r in rows for t in r["turns"]]
    if not turns:
        return {"turns": 0}
    usd = [t["usd"] for t in turns]
    latency = [t["latency_ms"] for t in turns]
    return {
        "turns": len(turns),
        "usd_total": round(sum(usd), 6),
        "usd_per_turn_mean": round(statistics.fmean(usd), 6),
        "usd_per_turn_p95": _pct(usd, 0.95),
        # The tokens priced as the product's default paid model, so a free-model run still
        # gives a cost to compare later slices against (see shared.budget._PRICES).
        "usd_per_turn_if_gpt_4o_mini": round(
            statistics.fmean(
                estimate_cost(PRICED_AS, t["input_tokens"], t["output_tokens"]) for t in turns
            ), 6,
        ),
        "llm_calls_per_turn_mean": round(statistics.fmean(t["llm_calls"] for t in turns), 2),
        "tool_calls_per_turn_mean": round(statistics.fmean(t["tool_calls"] for t in turns), 2),
        "turns_with_tool_calls": sum(1 for t in turns if t["tool_calls"]),
        "tool_error_rate": (
            round(sum(t["tool_errors"] for t in turns) / sum(t["tool_calls"] for t in turns), 4)
            if sum(t["tool_calls"] for t in turns) else None
        ),
        "turns_tool_budget_exhausted": sum(1 for t in turns if t["tool_budget_exhausted"]),
        "input_tokens_per_turn_mean": round(statistics.fmean(t["input_tokens"] for t in turns), 1),
        "output_tokens_per_turn_mean": round(
            statistics.fmean(t["output_tokens"] for t in turns), 1
        ),
        "latency_ms_p50": _pct(latency, 0.5),
        "latency_ms_p95": _pct(latency, 0.95),
        "turns_with_estimated_tokens": sum(1 for t in turns if t["estimated"]),
    }


def _check_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for name, ok in r["checks"].items():
            totals[name].append(ok)
    return {
        name: {"passed": sum(v), "total": len(v), "value": round(sum(v) / len(v), 4)}
        for name, v in sorted(totals.items())
    }


def summarise(rows: list[dict[str, Any]], samples: int, dropped: int, total: int) -> dict[str, Any]:
    model_rows = [r for r in rows if r["kind"] in MODEL_KINDS]
    groups: dict[str, Any] = {}
    for kind in MODEL_KINDS:
        by_item: dict[str, list[bool]] = defaultdict(list)
        for r in model_rows:
            if r["kind"] == kind:
                by_item[r["item_id"]].append(r["passed"])
        if by_item:
            groups[kind] = {
                **stats.group_summary(by_item, samples),
                "checks": _check_rates([r for r in model_rows if r["kind"] == kind]),
                "cost_latency": _cost_block([r for r in model_rows if r["kind"] == kind]),
            }
    by_tag: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in model_rows:
        by_tag[r["tag"]][r["item_id"]].append(r["passed"])
    security_rows = [r for r in rows if r["kind"] == "security"]
    out: dict[str, Any] = {
        "samples": samples,
        "n_items": len({r["item_id"] for r in rows}),
        "groups": groups,
        "by_tag": {t: stats.group_summary(v, samples) for t, v in sorted(by_tag.items())},
        "all_model_turns": _cost_block(model_rows),
        "units_total": total,
        "units_dropped_model_unavailable": dropped,
        "rows": {"attempts": rows},
    }
    if security_rows:
        out["security"] = {
            **stats.rate(sum(r["passed"] for r in security_rows), len(security_rows)),
            "failed_items": sorted(r["item_id"] for r in security_rows if not r["passed"]),
        }
    out["caveats"] = caveats(out)
    return out


def caveats(suite: dict[str, Any]) -> list[str]:
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
    for kind, group in suite["groups"].items():
        half = stats.resolution(group)
        notes.append(
            f"{kind}: n={group['n_items']} items x {group['samples']} samples; a change smaller "
            f"than about {0 if half is None else half:.2f} cannot be told apart from noise."
        )
    est = suite["all_model_turns"].get("turns_with_estimated_tokens", 0)
    if est:
        notes.append(f"{est} turns had no provider token report; their cost is an estimate.")
    if "security" in suite and suite["security"]["n"] < 30:
        notes.append(
            f"security: {suite['security']['n']} hostile strings; all passing does not bound "
            "the failure rate on strings nobody has tried."
        )
    return notes


async def run_quant_suite(
    items: list[dict[str, Any]],
    *,
    agent_factory: AgentFactory,
    verify_skill: VerifyMathSkill,
    samples: int,
    collector: TurnCollector,
    run_prefix: str = "eval",
) -> dict[str, Any]:
    """Run every item (model-driven ones `samples` times) and summarise."""
    dropped: list[str] = []

    async def guarded(item: dict[str, Any], attempt: int) -> dict[str, Any] | None:
        try:
            with cache_slot(attempt):
                return await run_attempt(item, attempt, agent_factory, collector, run_prefix)
        except AllModelsFailedError as exc:
            dropped.append(str(exc))
            return None

    model_items = [i for i in items if i["kind"] in MODEL_KINDS]
    jobs = [guarded(i, a) for i in model_items for a in range(samples)]
    done = [r for r in await gather_limited(jobs) if r is not None]
    security = [await run_security_case(i, verify_skill) for i in items if i["kind"] == "security"]
    total = len(jobs) + len(security)
    return summarise([*done, *security], samples, len(dropped), total)


# ── baseline file and comparison ─────────────────────────────────────────────


def headline_metrics(suite: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The numbers a baseline pins, each with n and its 95% interval. Higher is better."""
    out: dict[str, dict[str, Any]] = {}
    for kind, group in suite["groups"].items():
        out[f"{kind}.mean_pass_rate"] = {
            "value": group["mean_pass_rate"], "n": group["n_items"],
            "ci95": group["mean_pass_rate_ci95"], "samples": group["samples"],
        }
        out[f"{kind}.pass_hat_k"] = {
            "value": group["pass_hat_k"]["value"], "n": group["n_items"],
            "ci95": group["pass_hat_k"]["ci95"], "samples": group["samples"],
        }
    if "security" in suite:
        sec = suite["security"]
        out["security.pass_rate"] = {
            "value": sec["value"], "n": sec["n"], "ci95": sec["ci95"], "samples": 1,
        }
    return out


def baseline_document(suite: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    """The checked-in baseline: aggregates only (no model text), with the run's provenance."""
    summary = {k: v for k, v in suite.items() if k != "rows"}
    note = (
        "Measured on the model named in meta.model. "
        + ("" if meta.get("is_product_default_model") else
           f"That is NOT the product default ({meta.get('product_default_model')}); "
           "re-run with the default model before comparing a change against these numbers. ")
        + "Every rate carries its n and 95% interval; a difference inside an interval is noise."
    )
    return {"schema": BASELINE_SCHEMA, "suite": "quant", "meta": meta, "note": note,
            "metrics": headline_metrics(suite), "summary": summary}


DETERMINISTIC_METRICS = frozenset({"security.pass_rate"})


def compare_with_baseline(suite: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """
    One row per baseline metric: baseline value, current value, and a verdict that is
    "regression"/"improvement" only when the current value falls outside the interval the
    baseline carries. The security rate involves no model and no sampling, so any drop is a
    regression and there is no noise band. pass^k is skipped when k differs, since it is not
    comparable.
    """
    current = headline_metrics(suite)
    rows = []
    for name, base in baseline["metrics"].items():
        now = current.get(name)
        if now is None or now["value"] is None or base["value"] is None:
            continue
        if base.get("samples") != now.get("samples") and name.endswith("pass_hat_k"):
            continue
        exact = name in DETERMINISTIC_METRICS
        verdict_ci = [base["value"], base["value"]] if exact else base["ci95"]
        rows.append({
            "metric": name, "baseline": base["value"], "baseline_n": base["n"],
            "baseline_ci95": base["ci95"], "current": now["value"], "current_n": now["n"],
            "verdict": stats.verdict_vs_baseline(now["value"], verdict_ci),
        })
    return rows
