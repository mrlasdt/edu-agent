"""
Question-generation target for the agent-in-the-loop eval.

Runs the single-call baseline and the LangGraph pipeline on the same generation requests and
audits every item each one emits:

  - audit: the deterministic tools (structure, leak, sympy solver, distractors) applied to
    the emitted item. The pipeline gates on these same tools, so its shipped-defect rate is
    low by construction; the comparison that matters is *yield* (valid items per request) and
    what the baseline ships unchecked.
  - judge: an independent yes/no verdict from the harness judge on each emitted item
    (correct key, unambiguous, no leak). It shares no code with the pipeline's checks.
  - cost: calls, tokens and USD per target, per valid item and per emitted item.
  - verifier_reject_rate: share of generate attempts (parsed or unparseable, summed over all
    rows) whose deterministic verify step failed, read from the per-attempt trace. None for a
    target that never verifies (the baseline).
  - emitted_rate: share of requests whose item was emitted (status accepted or unchecked).
  - usd_per_accepted_item: target USD divided by emitted items.

Items the pipeline cannot repair are not emitted: they go to the human review queue.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from evals.agent_eval.judges import Judge, JudgeQuestion
from evals.agent_eval.runner import _guarded, _rate, _yes_no
from evals.agent_eval.transport import MeteredTransport, gather_limited, tagged
from evals.human_review_queue import HumanReviewQueue
from services.agent.src.agent.pipeline.schema import GenerationRequest, PipelineResult, QuestionItem
from services.agent.src.agent.pipeline.tools import run_checks
from services.agent.src.agent.skills.verify_math import VerifyMathSkill

GEN_JUDGE_INSTRUCTIONS = (
    "Is the keyed answer of the GRE question below correct, and is the question unambiguous "
    "and free of any answer leak? Work it out yourself if needed. Answer YES only if all of "
    "that holds."
)


class Target(Protocol):
    async def run(self, request: GenerationRequest) -> PipelineResult: ...


def _item_text(item: QuestionItem) -> str:
    choices = "\n".join(f"- {c}" for c in item.choices)
    return f"{item.stem}\n{choices}\nKeyed answer: {item.answer}".strip()


async def _audit(item: QuestionItem, skill: VerifyMathSkill) -> list[str]:
    return [c.name for c in await run_checks(item, skill) if not c.passed]


async def _run_one(
    name: str,
    target: Target,
    request: GenerationRequest,
    skill: VerifyMathSkill,
    judge: Judge,
    judge_errors: list[str],
) -> dict[str, Any]:
    with tagged(name):
        result = await target.run(request)
    defects: list[str] = []
    judge_ok: bool | None = None
    if result.item is not None:
        defects = await _audit(result.item, skill)
        verdict = await _yes_no(
            judge,
            JudgeQuestion(
                instructions=GEN_JUDGE_INSTRUCTIONS,
                material=_item_text(result.item),
                facts={"kind": "generation", "gold": not defects},
            ),
            judge_errors,
        )
        judge_ok = None if verdict is None else verdict.answer
    return {
        "request_id": request.request_id,
        "status": result.status,
        "attempts": result.attempts,
        "llm_calls": result.llm_calls,
        "defects": defects,
        "valid": result.item is not None and not defects,
        "judge_ok": judge_ok,
        "failures": result.failures,
        "trace_id": result.trace_id,
        "trace": [e.model_dump(mode="json") for e in result.trace],
    }


def _cost(transport: MeteredTransport, tag: str, n_valid: int, n_emitted: int) -> dict[str, Any]:
    recs = [r for r in transport.records if r.tag == tag]
    usd = sum(r.cost_usd for r in recs)
    return {
        "calls": len(recs),
        "usd": usd,
        "prompt_tokens": sum(r.prompt_tokens for r in recs),
        "completion_tokens": sum(r.completion_tokens for r in recs),
        "usd_per_valid_item": usd / n_valid if n_valid else None,
        "usd_per_accepted_item": usd / n_emitted if n_emitted else None,
    }


def _verifier_reject_rate(rows: Sequence[dict[str, Any]]) -> float | None:
    events = [e for r in rows for e in r["trace"]]
    verifies = [e for e in events if e["step"] == "verify"]
    attempts = sum(1 for e in events if e["step"] == "generate")
    return sum(1 for e in verifies if not e["ok"]) / attempts if verifies else None


def _summarise(
    rows: Sequence[dict[str, Any]], n_requests: int, cost: dict[str, Any]
) -> dict[str, Any]:
    emitted = [r for r in rows if r["status"] in ("accepted", "unchecked")]
    valid = [r for r in rows if r["valid"]]
    breakdown: dict[str, int] = {}
    for r in emitted:
        for d in r["defects"]:
            breakdown[d] = breakdown.get(d, 0) + 1
    judged = [r for r in emitted if r["judge_ok"] is not None]
    return {
        "n_requests": n_requests,
        "n_completed": len(rows),
        "emitted_rate": len(emitted) / len(rows) if rows else None,
        "valid_yield": len(valid) / len(rows) if rows else None,
        "verifier_reject_rate": _verifier_reject_rate(rows),
        "shipped_defect_rate": _rate([bool(r["defects"]) for r in emitted]),
        "shipped_defects_by_check": breakdown,
        "escalation_rate": _rate([r["status"] == "needs_review" for r in rows]),
        "unparseable_rate": _rate([r["status"] == "unparseable" for r in rows]),
        "mean_attempts": sum(r["attempts"] for r in rows) / len(rows) if rows else None,
        "mean_llm_calls": sum(r["llm_calls"] for r in rows) / len(rows) if rows else None,
        "judge_accept_rate": _rate([bool(r["judge_ok"]) for r in judged]),
        "judge_vs_audit_agreement": _rate([r["judge_ok"] == (not r["defects"]) for r in judged]),
        "cost": cost,
    }


def _delta(after: float | None, before: float | None) -> float | None:
    return None if after is None or before is None else after - before


async def run_generation(
    requests: list[GenerationRequest],
    targets: dict[str, Target],
    judge: Judge,
    skill: VerifyMathSkill,
    transport: MeteredTransport,
    queue: HumanReviewQueue | None = None,
) -> dict[str, Any]:
    """`targets` maps a name ("baseline", "pipeline") to the generator under test."""
    judge_errors: list[str] = []
    unavailable: list[str] = []
    queue = queue if queue is not None else HumanReviewQueue()
    work = [(n, t, r) for n, t in targets.items() for r in requests]
    out = await gather_limited(
        [
            _guarded(_run_one(n, t, r, skill, judge, judge_errors), unavailable)
            for n, t, r in work
        ]
    )
    rows: dict[str, list[dict[str, Any]]] = {n: [] for n in targets}
    for (name, _, _), row in zip(work, out, strict=True):
        if row is not None:
            rows[name].append(row)
            if row["status"] == "needs_review":
                queue.add_generation_review(row["request_id"], row["failures"], row["trace_id"])

    summaries = {
        name: _summarise(
            rs,
            len(requests),
            _cost(
                transport,
                name,
                sum(1 for r in rs if r["valid"]),
                sum(1 for r in rs if r["status"] in ("accepted", "unchecked")),
            ),
        )
        for name, rs in rows.items()
    }
    comparison: dict[str, Any] = {}
    if "baseline" in summaries and "pipeline" in summaries:
        before, after = summaries["baseline"], summaries["pipeline"]
        comparison = {
            "valid_yield_gain": _delta(after["valid_yield"], before["valid_yield"]),
            "shipped_defect_rate_change": _delta(
                after["shipped_defect_rate"], before["shipped_defect_rate"]
            ),
            "extra_llm_calls_per_request": _delta(
                after["mean_llm_calls"], before["mean_llm_calls"]
            ),
            "note": (
                "audit tools are shared with the pipeline's own gate, so the pipeline's shipped "
                "defect rate is low by construction; judge_accept_rate is the independent check"
            ),
        }
    return {
        "n_requests": len(requests),
        "targets": summaries,
        "comparison": comparison,
        "review_queue": queue.count(),
        "judge_errors": len(judge_errors),
        "units_dropped_model_unavailable": len(unavailable),
        "rows": rows,
    }
