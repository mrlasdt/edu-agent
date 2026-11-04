"""
Report on the per-turn measurement log of the live chat path.

    uv run python -m evals.turn_report                # per-agent summary of .turn-log/turns.jsonl
    uv run python -m evals.turn_report --turns        # plus one line per turn
    uv run python -m evals.turn_report --json         # the same numbers, machine-readable
    uv run python -m evals.turn_report --price-as gpt-4o-mini   # USD at that model's prices
    uv run python -m evals.turn_report --log PATH --last 20

The gateway and the agent each append one record per turn under the same trace id
(`shared.turn_meter`); this joins them into one row per turn and summarises per agent and mode.
Tokens the provider did not report are our estimate and are counted and flagged, never hidden.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shared.src.shared.budget import estimate_cost
from shared.src.shared.config import get_settings
from shared.src.shared.turn_meter import TurnLog


def merge_turns(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per trace id, oldest first. Gateway and agent records are joined."""
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        trace_id = rec.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            by_trace[trace_id].append(rec)
    turns = [_merge(trace_id, recs) for trace_id, recs in by_trace.items()]
    return sorted(turns, key=lambda t: t["ts"])


def _merge(trace_id: str, recs: list[dict[str, Any]]) -> dict[str, Any]:
    gateway = next((r for r in recs if r.get("source") == "gateway"), {})
    agent = next((r for r in recs if r.get("source") == "agent"), {})
    usages = {r.get("usage") for r in recs if r.get("llm_calls")}
    phases = agent.get("phases_ms") or {}
    total_ms = gateway.get("total_ms", agent.get("total_ms"))
    return {
        "trace_id": trace_id,
        "ts": min(r.get("ts", 0.0) for r in recs),
        "agent": agent.get("agent") or gateway.get("agent") or "?",
        "mode": agent.get("mode") or gateway.get("mode") or "?",
        "prompt_version": agent.get("prompt_version") or "",
        "llm_calls": sum(r.get("llm_calls", 0) for r in recs),
        "agent_calls": agent.get("llm_calls", 0),
        "guardrail_calls": gateway.get("llm_calls", 0),
        "input_tokens": sum(r.get("input_tokens", 0) for r in recs),
        "output_tokens": sum(r.get("output_tokens", 0) for r in recs),
        "guardrail_input_tokens": gateway.get("input_tokens", 0),
        "guardrail_output_tokens": gateway.get("output_tokens", 0),
        "cost_usd": sum(r.get("cost_usd", 0.0) for r in recs),
        "guardrail_cost_usd": gateway.get("cost_usd", 0.0),
        "estimated": bool(usages & {"estimate", "mixed"}),
        "ttft_ms": agent.get("ttft_ms"),
        "first_byte_ms": gateway.get("first_byte_ms"),
        "total_ms": total_ms,
        "retrieval_ms": phases.get("retrieval", 0.0),
        "verify_ms": phases.get("verify", 0.0),
        "llm_ms": agent.get("llm_ms", 0.0),
        "guardrail_llm_ms": gateway.get("llm_ms", 0.0),
        "notes_retrieved": agent.get("notes_retrieved"),
        "tool_calls": len(agent.get("tool_calls") or []),
        "tool_errors": agent.get("tool_errors", 0),
        "outcome": gateway.get("outcome", "no gateway record"),
        "flags": gateway.get("flags") or [],
        "error": agent.get("error") or gateway.get("error"),
    }


def reprice(turn: dict[str, Any], model: str) -> dict[str, Any]:
    """The turn with its USD recomputed from its recorded tokens at `model`'s prices."""
    guard = estimate_cost(model, turn["guardrail_input_tokens"], turn["guardrail_output_tokens"])
    total = estimate_cost(model, turn["input_tokens"], turn["output_tokens"])
    return {**turn, "cost_usd": total, "guardrail_cost_usd": guard}


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..1); None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _vals(turns: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [t[key] for t in turns if isinstance(t.get(key), (int, float))]


def summarise(turns: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per (agent, mode) numbers plus overall totals and outcome counts."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        groups[f"{turn['agent']}/{turn['mode']}"].append(turn)
    per_group: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        n = len(rows)
        costs = _vals(rows, "cost_usd")
        total = _vals(rows, "total_ms")
        per_group[name] = {
            "turns": n,
            "llm_calls_mean": sum(t["llm_calls"] for t in rows) / n,
            "input_tokens_mean": sum(t["input_tokens"] for t in rows) / n,
            "output_tokens_mean": sum(t["output_tokens"] for t in rows) / n,
            "cost_usd_mean": sum(costs) / n,
            "cost_usd_p95": percentile(costs, 0.95),
            "guardrail_cost_usd_mean": sum(t["guardrail_cost_usd"] for t in rows) / n,
            "ttft_ms_p50": percentile(_vals(rows, "ttft_ms"), 0.5),
            "ttft_ms_p95": percentile(_vals(rows, "ttft_ms"), 0.95),
            "first_byte_ms_p50": percentile(_vals(rows, "first_byte_ms"), 0.5),
            "total_ms_p50": percentile(total, 0.5),
            "total_ms_p95": percentile(total, 0.95),
            "retrieval_ms_mean": sum(t["retrieval_ms"] for t in rows) / n,
            "model_ms_mean": sum(t["llm_ms"] for t in rows) / n,
            "guardrail_model_ms_mean": sum(t["guardrail_llm_ms"] for t in rows) / n,
            "verify_ms_mean": sum(t["verify_ms"] for t in rows) / n,
            "notes_retrieved_mean": (
                sum(_vals(rows, "notes_retrieved")) / len(_vals(rows, "notes_retrieved"))
                if _vals(rows, "notes_retrieved")
                else None
            ),
            "estimated_usage_turns": sum(1 for t in rows if t["estimated"]),
        }
    return {
        "turns": len(turns),
        "cost_usd_total": sum(t["cost_usd"] for t in turns),
        "outcomes": dict(Counter(t["outcome"] for t in turns)),
        "flags": dict(Counter(f for t in turns for f in t["flags"])),
        "groups": per_group,
    }


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def _usd(value: float | None) -> str:
    return "-" if value is None else f"{value:.5f}"


def render(
    summary: dict[str, Any],
    turns: Sequence[dict[str, Any]],
    *,
    per_turn: bool,
    price_as: str | None = None,
) -> str:
    lines: list[str] = []
    n = summary["turns"]
    est = sum(1 for t in turns if t["estimated"])
    lines.append(f"{n} turn(s), {summary['cost_usd_total']:.5f} USD in total")
    if price_as:
        lines.append(f"  USD recomputed from the recorded tokens at {price_as} prices")
    if est:
        lines.append(f"  {est} turn(s) include ESTIMATED tokens (provider sent no usage)")
    lines.append("")
    header = (
        f"{'agent/mode':<20}{'turns':>6}{'calls':>7}{'in tok':>8}{'out tok':>9}"
        f"{'usd/turn':>11}{'ttft p50':>10}{'ttft p95':>10}{'total p50':>11}{'total p95':>11}"
    )
    lines += [header, "-" * len(header)]
    for name, g in summary["groups"].items():
        lines.append(
            f"{name:<20}{g['turns']:>6}{g['llm_calls_mean']:>7.1f}{g['input_tokens_mean']:>8.0f}"
            f"{g['output_tokens_mean']:>9.0f}{_usd(g['cost_usd_mean']):>11}"
            f"{_ms(g['ttft_ms_p50']):>10}{_ms(g['ttft_ms_p95']):>10}"
            f"{_ms(g['total_ms_p50']):>11}{_ms(g['total_ms_p95']):>11}"
        )
    lines.append("(times in ms; ttft = model time to first token of the first call)")
    lines.append("")
    lines.append("where a turn's time goes (mean ms per turn)")
    for name, g in summary["groups"].items():
        lines.append(
            f"  {name:<20} retrieval {g['retrieval_ms_mean']:>6,.0f}   "
            f"agent model {g['model_ms_mean']:>6,.0f}   "
            f"guardrail model {g['guardrail_model_ms_mean']:>6,.0f}   "
            f"verify {g['verify_ms_mean']:>5,.0f}"
        )
    lines.append("")
    lines.append("guardrail overhead per turn (mean USD)")
    for name, g in summary["groups"].items():
        lines.append(f"  {name:<20} {_usd(g['guardrail_cost_usd_mean'])}")
    lines.append("")
    lines.append(
        "outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(summary["outcomes"].items()))
    )
    if summary["flags"]:
        lines.append(
            "flags:    " + ", ".join(f"{k} {v}" for k, v in sorted(summary["flags"].items()))
        )
    if per_turn:
        lines += ["", "turns (oldest first)"]
        lines.append(
            f"  {'trace':<10}{'agent/mode':<20}{'calls':>6}{'in':>7}{'out':>6}{'usd':>10}"
            f"{'ttft':>7}{'total':>8}{'notes':>6}{'tools':>6}  outcome"
        )
        for t in turns:
            star = "*" if t["estimated"] else " "
            lines.append(
                f"  {t['trace_id'][:8]:<10}{t['agent'] + '/' + t['mode']:<20}{t['llm_calls']:>6}"
                f"{t['input_tokens']:>7}{t['output_tokens']:>6}{_usd(t['cost_usd']):>10}{star}"
                f"{_ms(t['ttft_ms']):>6}{_ms(t['total_ms']):>8}"
                f"{'-' if t['notes_retrieved'] is None else t['notes_retrieved']:>6}"
                f"{t['tool_calls'] if t['tool_calls'] else '-':>6}"
                f"  {t['outcome']}"
            )
        if est:
            lines.append("  * tokens estimated")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.turn_report", description="Report on the per-turn measurement log."
    )
    parser.add_argument(
        "--log", default=None, help="turn log (default: TURN_LOG_PATH, .turn-log/turns.jsonl)"
    )
    parser.add_argument("--last", type=int, default=None, help="only the most recent N turns")
    parser.add_argument("--turns", action="store_true", help="also list every turn")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument(
        "--price-as",
        metavar="MODEL",
        default=None,
        help="recompute USD from the recorded tokens at this model's prices "
        "(e.g. gpt-4o-mini), for a log recorded on a free or different model",
    )
    args = parser.parse_args(argv)

    path = Path(args.log or get_settings().turn_log_path or ".turn-log/turns.jsonl")
    turns = merge_turns(TurnLog(path).read())
    if args.last is not None:
        turns = turns[-args.last :] if args.last > 0 else []
    if not turns:
        print(f"no turns recorded in {path}", file=sys.stderr)
        return 1
    if args.price_as:
        turns = [reprice(t, args.price_as) for t in turns]
    summary = summarise(turns)
    if args.json:
        print(json.dumps({"summary": summary, "turns": turns}, indent=2))
    else:
        print(render(summary, turns, per_turn=args.turns, price_as=args.price_as))
    return 0


if __name__ == "__main__":
    sys.exit(main())
