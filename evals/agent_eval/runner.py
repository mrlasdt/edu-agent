"""
Agent-in-the-loop eval runner.

Runs the real QuantAgent (tutor and solve modes) on golden-set questions, then judges
what the agent actually produced:
  - tutor mode: answer-leak rate (rule detector and judge)
  - solve mode: answer accuracy (sympy against the golden expression, never the agent's
    own claim) and the product verifier's fail rate
  - judge quality: agreement with known labels on calibration probes (golden answer vs a
    perturbed wrong one, leaking vs clean hint) and, for AW, agreement with the published
    anchor-essay scores (the anchor under test is held out of the judge's context)
  - cost and latency of every model call
"""
from __future__ import annotations

import statistics
import tempfile
import time
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Any

import yaml

from evals.agent_eval.judges import Judge, JudgeError, JudgeQuestion, YesNo
from evals.agent_eval.probes import (
    TUTOR_PROBES,
    clean_answer,
    correct_question,
    correctness_calibration,
    leak_calibration,
    leak_question,
    leaks_by_rule,
)
from evals.agent_eval.transport import (
    AllModelsFailedError,
    MeteredTransport,
    gather_limited,
    tagged,
)
from evals.judges.quant_judge import grade_quant_item
from evals.loader import load_golden_set, sample_golden_set
from services.agent.src.agent.agents.quant import QuantAgent
from services.agent.src.agent.skills.verify_math import VerifyMathSkill
from services.math_verifier.src.math_verifier.sandbox import verify_expression
from shared.src.shared.models import Mode, Session

AW_SCALE = [
    "1 - fundamentally deficient",
    "2 - seriously flawed",
    "3 - limited",
    "4 - adequate",
    "5 - strong",
    "6 - outstanding",
]
AW_INSTRUCTIONS = (
    "Score the candidate GRE Analytical Writing essay on the ETS 6-point scale, "
    "calibrating against the anchor essays provided. Output the score."
)
AW_AGREE_TOLERANCE = 1.0


class InProcessVerifySkill(VerifyMathSkill):
    """The product's verify-math skill, with the HTTP hop replaced by the same sympy sandbox."""

    def __init__(self) -> None:
        pass

    async def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await verify_expression(arguments["expression"], arguments["expected"])
        return {"verified": result.verified, "computed": result.computed, "error": result.error}


def write_model_config(model: str, base_path: str = "config/model_config.dev.yaml") -> str:
    """A model config identical to the dev one but with every role routed to `model`."""
    raw = yaml.safe_load(Path(base_path).read_text())
    provider, _, name = model.partition("/")
    for role in raw["roles"].values():
        role["provider"], role["model"] = provider, name
    fd = Path(tempfile.mkdtemp(prefix="eval-config-")) / "model_config.yaml"
    fd.write_text(yaml.safe_dump(raw))
    return str(fd)


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _agreement(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    n = len(pairs)
    return {
        "n": n,
        "agreement": (sum(1 for a, b in pairs if a == b) / n) if n else None,
        "false_positive": sum(1 for p, g in pairs if p and not g),
        "false_negative": sum(1 for p, g in pairs if g and not p),
    }


async def _guarded[T](coro: Coroutine[Any, Any, T], failures: list[str]) -> T | None:
    """Isolate one unit of work: a fully unavailable model chain drops that unit, not the run.

    BudgetExceededError is deliberately not caught - it must abort the whole run.
    """
    try:
        return await coro
    except AllModelsFailedError as exc:
        failures.append(str(exc))
        return None


async def _yes_no(judge: Judge, q: JudgeQuestion, errors: list[str]) -> YesNo | None:
    try:
        return await judge.yes_no(q)
    except JudgeError as exc:
        errors.append(str(exc))
        return None


async def _run_agent(
    agent: QuantAgent, item: dict[str, Any], mode: Mode, message: str
) -> tuple[str, bool | None, float]:
    session = Session(candidate_id="eval", test_type="gre", section="quant", mode=mode)
    parts: list[str] = []
    verifier_fail: bool | None = None
    start = time.perf_counter()
    with tagged("agent"):
        async for out in agent.run(session, message):
            if isinstance(out, str):
                parts.append(out)
            else:
                verifier_fail = out.get("verifier_fail", verifier_fail)
    return "".join(parts), verifier_fail, time.perf_counter() - start


async def run_quant(
    items: list[dict[str, Any]], agent: QuantAgent, judge: Judge
) -> dict[str, Any]:
    judge_errors: list[str] = []
    unavailable: list[str] = []

    async def tutor(item: dict[str, Any], probe: str) -> dict[str, Any]:
        reply, _, latency = await _run_agent(
            agent, item, Mode.tutor, TUTOR_PROBES[probe].format(question=item["question"])
        )
        rule = leaks_by_rule(reply, item["ground_truth"])
        verdict = await _yes_no(judge, leak_question(item, reply, rule), judge_errors)
        return {
            "item_id": item["id"], "probe": probe, "latency_s": latency,
            "leak_rule": rule, "leak_judge": None if verdict is None else verdict.answer,
        }

    async def solve(item: dict[str, Any]) -> dict[str, Any]:
        text, verifier_fail, latency = await _run_agent(
            agent, item, Mode.solve, item["question"]
        )
        answer = clean_answer(agent._extract_final_answer(text) or "")
        graded = await grade_quant_item(item, candidate_answer=answer)
        verdict = await _yes_no(judge, correct_question(item, text, graded.correct), judge_errors)
        return {
            "item_id": item["id"], "latency_s": latency, "answer": answer,
            "correct": graded.correct, "verifier_fail": verifier_fail,
            "judge_correct": None if verdict is None else verdict.answer,
        }

    async def calibrate(
        item: dict[str, Any],
    ) -> tuple[list[tuple[bool, bool]], list[tuple[bool, bool]]]:
        leak_pairs: list[tuple[bool, bool]] = []
        correct_pairs: list[tuple[bool, bool]] = []
        for q, gold in leak_calibration(item):
            v = await _yes_no(judge, q, judge_errors)
            if v is not None:
                leak_pairs.append((v.answer, gold))
        for q, gold in correctness_calibration(item):
            v = await _yes_no(judge, q, judge_errors)
            if v is not None:
                correct_pairs.append((v.answer, gold))
        return leak_pairs, correct_pairs

    tutor_out = await gather_limited(
        [_guarded(tutor(i, p), unavailable) for i in items for p in TUTOR_PROBES]
    )
    solve_out = await gather_limited([_guarded(solve(i), unavailable) for i in items])
    cal_out = await gather_limited([_guarded(calibrate(i), unavailable) for i in items])
    tutor_rows = [r for r in tutor_out if r is not None]
    solve_rows = [r for r in solve_out if r is not None]
    calibration = [c for c in cal_out if c is not None]

    leak_cal = [p for lp, _ in calibration for p in lp]
    correct_cal = [p for _, cp in calibration for p in cp]
    judged_leaks = [
        (r["leak_judge"], r["leak_rule"]) for r in tutor_rows if r["leak_judge"] is not None
    ]
    judged_correct = [
        (r["judge_correct"], r["correct"]) for r in solve_rows if r["judge_correct"] is not None
    ]
    return {
        "n_items": len(items),
        "tutor": {
            "leak_rate_rule": _rate([r["leak_rule"] for r in tutor_rows]),
            "leak_rate_judge": _rate(
                [r["leak_judge"] for r in tutor_rows if r["leak_judge"] is not None]
            ),
            "leak_rate_rule_by_probe": {
                p: _rate([r["leak_rule"] for r in tutor_rows if r["probe"] == p])
                for p in TUTOR_PROBES
            },
            "judge_vs_rule_on_agent_output": _agreement(judged_leaks),
        },
        "solve": {
            "accuracy": _rate([r["correct"] for r in solve_rows]),
            "verifier_fail_rate": _rate(
                [bool(r["verifier_fail"]) for r in solve_rows if r["verifier_fail"] is not None]
            ),
            "judge_vs_sympy_on_agent_output": _agreement(judged_correct),
        },
        "judge_calibration": {
            "leak_labels": _agreement(leak_cal),
            "correctness_labels": _agreement(correct_cal),
            "errors": len(judge_errors),
        },
        "units_dropped_model_unavailable": len(unavailable),
        "rows": {"tutor": tutor_rows, "solve": solve_rows},
    }


async def run_aw(items: list[dict[str, Any]], judge: Judge) -> dict[str, Any]:
    """Judge calibration against the golden anchor essays (held out of the judge's context)."""
    errors: list[str] = []
    unavailable: list[str] = []

    async def one(item: dict[str, Any], level: str) -> tuple[float, float] | None:
        anchors: dict[str, str] = item["anchor_essays"]
        others = "\n\n".join(
            f"--- Anchor essay (score {s}) ---\n{t}"
            for s, t in sorted(anchors.items())
            if s != level
        )
        q = JudgeQuestion(
            instructions=AW_INSTRUCTIONS,
            material=(
                f"## Task prompt\n{item['prompt']}\n\n## Anchor essays\n{others}\n\n"
                f"## Candidate essay\n{anchors[level]}"
            ),
            facts={"kind": "aw", "gold": float(level), "essay": anchors[level]},
        )
        try:
            return await judge.score(q, AW_SCALE), float(level)
        except JudgeError as exc:
            errors.append(str(exc))
            return None

    results = await gather_limited(
        [
            _guarded(one(item, lvl), unavailable)
            for item in items
            for lvl in item.get("anchor_essays", {})
        ]
    )
    pairs = [r for r in results if r is not None]
    diffs = [abs(s - g) for s, g in pairs]
    return {
        "n_judgements": len(pairs),
        "within_tolerance": _rate([d <= AW_AGREE_TOLERANCE for d in diffs]),
        "tolerance": AW_AGREE_TOLERANCE,
        "mae": statistics.fmean(diffs) if diffs else None,
        "errors": len(errors),
        "units_dropped_model_unavailable": len(unavailable),
    }


def latency_cost_summary(
    transport: MeteredTransport, agent_item_latencies: list[float]
) -> dict[str, Any]:
    def pct(vals: list[float], q: float) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        return s[min(len(s) - 1, int(q * len(s)))]

    out: dict[str, Any] = {"total_usd": transport.spent_this_run, "calls": len(transport.records)}
    for tag in ("agent", "judge"):
        recs = [r for r in transport.records if r.tag == tag]
        lat = [r.latency_s for r in recs]
        out[tag] = {
            "calls": len(recs),
            "usd": sum(r.cost_usd for r in recs),
            "prompt_tokens": sum(r.prompt_tokens for r in recs),
            "completion_tokens": sum(r.completion_tokens for r in recs),
            "call_latency_p50_s": pct(lat, 0.5),
            "call_latency_p95_s": pct(lat, 0.95),
        }
    out["agent_turn_latency_mean_s"] = (
        statistics.fmean(agent_item_latencies) if agent_item_latencies else None
    )
    return out


def select_items(suite: str, max_items: int, seed: int) -> list[dict[str, Any]]:
    items = load_golden_set(suite)
    rate = min(1.0, max_items / len(items)) if items else 1.0
    return sample_golden_set(items, rate, seed=seed)
