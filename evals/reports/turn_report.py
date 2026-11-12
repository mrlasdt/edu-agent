"""
Report on the per-turn measurement log of the live chat path.

    uv run python -m evals turn-report                # per-agent summary of .turn-log/turns.jsonl
    uv run python -m evals turn-report --turns        # plus one line per turn
    uv run python -m evals turn-report --json         # the same numbers, machine-readable
    uv run python -m evals turn-report --feedback     # thumbs-down rate per agent, prompt version
    uv run python -m evals turn-report --price-as gpt-4o-mini   # USD at that model's prices
    uv run python -m evals turn-report --log PATH --last 20

The gateway and the agent each append one record per turn under the same trace id
(`shared.turn_meter`); this joins them into one row per turn and summarises per agent and mode.
Tokens the provider did not report are our estimate and are counted and flagged, never hidden.

`--feedback` joins the thumbs from `.turn-log/feedback.jsonl` (POST /feedback) to the turns by
trace id. A learner's latest vote on a turn is the one that counts. With one learner the rate is
an anecdote, not a statistic; the report shows the counts next to it so that stays visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shared.src.shared.budget import estimate_cost
from shared.src.shared.config import (
    DEFAULT_FEEDBACK_LOG_PATH,
    DEFAULT_TURN_LOG_PATH,
    anchor,
    get_feedback_settings,
)
from shared.src.shared.feedback import latest_votes
from shared.src.shared.stats import percentile
from shared.src.shared.turn_meter import TurnLog
from shared.src.shared.turn_outcome import Outcome


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
        "outcome": gateway.get("outcome", Outcome.no_gateway_record.value),
        "flags": gateway.get("flags") or [],
        "error": agent.get("error") or gateway.get("error"),
    }


def reprice(turn: dict[str, Any], model: str) -> dict[str, Any]:
    """The turn with its USD recomputed from its recorded tokens at `model`'s prices."""
    guard = estimate_cost(model, turn["guardrail_input_tokens"], turn["guardrail_output_tokens"])
    total = estimate_cost(model, turn["input_tokens"], turn["output_tokens"])
    return {**turn, "cost_usd": total, "guardrail_cost_usd": guard}


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


def summarise_feedback(
    turns: Sequence[dict[str, Any]], feedback: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """
    Thumbs per (agent, prompt version) over `turns`, newest vote per turn. `down_rate` is
    thumbs-down over rated turns (None when nothing is rated); `turns` shows how few are rated.
    """
    votes = latest_votes(list(feedback))
    by_trace = {t["trace_id"]: t for t in turns}
    groups: dict[str, dict[str, Any]] = {}
    for row in turns:
        key = f"{row['agent']} {row['prompt_version'] or '-'}"
        groups.setdefault(key, {"turns": 0, "up": 0, "down": 0})["turns"] += 1
    down: list[dict[str, Any]] = []
    for trace_id, vote in votes.items():
        turn = by_trace.get(trace_id)
        if turn is None:
            continue
        counts = groups[f"{turn['agent']} {turn['prompt_version'] or '-'}"]
        counts[vote["vote"]] += 1
        if vote["vote"] == "down":
            down.append(
                {
                    "trace_id": trace_id,
                    "agent": turn["agent"],
                    "mode": turn["mode"],
                    "prompt_version": turn["prompt_version"],
                    "ts": vote.get("ts"),
                    "comment": vote.get("comment", ""),
                }
            )
    for group in groups.values():
        rated = group["up"] + group["down"]
        group["rated"] = rated
        group["down_rate"] = group["down"] / rated if rated else None
    return {
        "votes": sum(g["rated"] for g in groups.values()),
        "thumbs_down": len(down),
        "unmatched_votes": sum(1 for t in votes if t not in by_trace),
        "groups": dict(sorted(groups.items())),
        "down": sorted(down, key=lambda d: d["ts"] or 0.0),
    }


def render_feedback(fb: dict[str, Any]) -> str:
    lines = ["", "feedback (latest thumbs per turn)"]
    header = f"{'agent prompt':<28}{'turns':>6}{'rated':>7}{'up':>5}{'down':>6}{'down %':>9}"
    lines += [header, "-" * len(header)]
    for name, g in fb["groups"].items():
        rate = "-" if g["down_rate"] is None else f"{100 * g['down_rate']:.0f}%"
        lines.append(
            f"{name:<28}{g['turns']:>6}{g['rated']:>7}{g['up']:>5}{g['down']:>6}{rate:>9}"
        )
    lines.append("(a handful of votes is an anecdote, not a rate)")
    if fb["unmatched_votes"]:
        lines.append(f"{fb['unmatched_votes']} vote(s) on turns missing from the turn log")
    if fb["down"]:
        lines.append("")
        lines.append("thumbs-down (python -m evals feedback-to-item <trace id> drafts an item)")
        for d in fb["down"]:
            comment = f"  \"{d['comment']}\"" if d["comment"] else ""
            lines.append(
                f"  {d['trace_id']}  {d['agent']}/{d['mode']} "
                f"{d['prompt_version'] or '-'}{comment}"
            )
    return "\n".join(lines)


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
        prog="python -m evals turn-report", description="Report on the per-turn measurement log."
    )
    parser.add_argument(
        "--log", default=None, help="turn log (default: TURN_LOG_PATH, .turn-log/turns.jsonl)"
    )
    parser.add_argument("--last", type=int, default=None, help="only the most recent N turns")
    parser.add_argument("--turns", action="store_true", help="also list every turn")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="also show thumbs-down rate per agent and prompt version, and each thumbs-down",
    )
    parser.add_argument(
        "--feedback-log",
        default=None,
        help="feedback log (default: FEEDBACK_LOG_PATH, .turn-log/feedback.jsonl)",
    )
    parser.add_argument(
        "--price-as",
        metavar="MODEL",
        default=None,
        help="recompute USD from the recorded tokens at this model's prices "
        "(e.g. gpt-4o-mini), for a log recorded on a free or different model",
    )
    args = parser.parse_args(argv)

    path = Path(
        args.log or get_feedback_settings().turn_log_path or anchor(DEFAULT_TURN_LOG_PATH)
    )
    turns = merge_turns(TurnLog(path).read())
    if args.last is not None:
        turns = turns[-args.last :] if args.last > 0 else []
    if not turns:
        print(f"no turns recorded in {path}", file=sys.stderr)
        return 1
    if args.price_as:
        turns = [reprice(t, args.price_as) for t in turns]
    summary = summarise(turns)
    fb: dict[str, Any] | None = None
    if args.feedback:
        fb_path = Path(
            args.feedback_log
            or get_feedback_settings().feedback_log_path
            or anchor(DEFAULT_FEEDBACK_LOG_PATH)
        )
        fb = summarise_feedback(turns, TurnLog(fb_path).read())
        if args.last is not None:
            fb["unmatched_votes"] = 0  # votes on turns older than the window are not orphans
    if args.json:
        out: dict[str, Any] = {"summary": summary, "turns": turns}
        if fb is not None:
            out["feedback"] = fb
        print(json.dumps(out, indent=2))
    else:
        print(render(summary, turns, per_turn=args.turns, price_as=args.price_as))
        if fb is not None:
            print(render_feedback(fb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
