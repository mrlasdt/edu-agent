"""
Render a Quant suite baseline (or any saved quant summary) as a readable report.

    uv run python -m evals.quant_report                          # evals/baselines/quant.json
    uv run python -m evals.quant_report path/to/baseline.json
    uv run python -m evals.quant_report NEW.json --against OLD.json   # did anything move?

Reads a checked-in file, makes no model call, and prints markdown: where the numbers came from,
the headline rates with their n and 95% intervals, which checks fail, which failure types hurt
most, what a turn costs, and what the sample size does not support.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evals.agent_eval.quant_suite import compare_with_baseline
from evals.agent_eval.results import QuantBaseline
from evals.loader import load_golden_set

DEFAULT_BASELINE = Path("evals/baselines/quant.json")
GROUP_LABELS = {
    "solve": "solve (final answer right and well-formed)",
    "tutor": "tutor, one reply",
    "tutor_multiturn": "tutor, 4-turn conversation",
}


def _ci(ci: list[float] | None) -> str:
    return "n/a" if not ci else f"{ci[0]:.2f}-{ci[1]:.2f}"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _num(value: float | None, spec: str) -> str:
    """A figure for a table cell; "n/a" for a group that ran no turns."""
    return "n/a" if value is None else format(value, spec)


def render(doc: QuantBaseline) -> str:
    meta, summary = doc.meta, doc.summary
    golden = {i["id"]: i for i in load_golden_set("quant")}
    default = (
        "the product default" if meta.is_product_default_model
        else f"NOT the product default `{meta.product_default_model}`"
    )
    out = [
        "# Quant failure suite - baseline",
        "",
        f"- model: `{meta.model}` ({default})",
        f"- samples per item: {meta.samples}, completion cap {meta.max_tokens} tokens, "
        f"temperature {meta.temperature}, prompts {meta.prompt_versions}",
        f"- date {meta.date_utc}, commit `{meta.git_commit}`",
        f"- {doc.note}",
        "",
        "## Headline (rate, 95% interval)",
        "",
        "| group | items | one attempt | pass@k | pass^k |",
        "|---|---|---|---|---|",
    ]
    k = meta.samples
    for kind, g in summary.groups.items():
        out.append(
            f"| {GROUP_LABELS[kind]} | {g.n_items} | {_pct(g.mean_pass_rate)} "
            f"({_ci(g.mean_pass_rate_ci95)}) | {_pct(g.pass_at_k.value)} "
            f"({_ci(g.pass_at_k.ci95)}) | {_pct(g.pass_hat_k.value)} "
            f"({_ci(g.pass_hat_k.ci95)}) |"
        )
    sec = summary.security
    if sec is None:
        raise SystemExit("this baseline has no security block; it is not a Quant suite baseline")
    out += [
        "",
        f"pass@{k}: at least one of {k} attempts passed. pass^{k}: all {k} passed (what a "
        "learner who asks the same thing repeatedly experiences). "
        f"Security (no model, n={sec.n} hostile strings): {sec.successes} rejected, "
        f"95% interval {_ci(sec.ci95)}.",
        "",
        "## Which checks fail (share of attempts passing)",
        "",
        "| group | check | passed |",
        "|---|---|---|",
    ]
    for kind, g in summary.groups.items():
        for name, check in g.checks.items():
            out.append(f"| {kind} | {name} | {check.passed}/{check.total} ({check.value:.2f}) |")
    if wbv := summary.wrong_but_verified:
        seeded, observed, control = wbv.seeded, wbv.observed, wbv.control
        out += [
            "", "## Wrong but verified", "",
            f"- seeded wrong answers the independent check agreed with: {seeded.count}/"
            f"{seeded.n} (95% interval {_ci(seeded.ci95)}); {seeded.disagrees} "
            f"disagreed, {seeded.not_checkable} not checkable",
            f"- the same answers through the well-formed check alone (before the independent "
            f"check existed): {wbv.self_compare.count}/{wbv.self_compare.n} accepted",
            f"- control, the golden key through the independent check: {control.agrees}/"
            f"{control.n} agrees, {control.disagrees} disagrees, "
            f"{control.not_checkable} not checkable",
            f"- real solve attempts that were wrong and still got \"agrees\": "
            f"{observed.count}/{observed.solve_attempts} attempts",
            "- the independent check covers numeric answers only; a wrong answer is not "
            "counted as verified when the check could not run",
        ]
    out += [
        "", "## By failure type", "",
        "| type | items | one attempt | pass^k |", "|---|---|---|---|",
    ]
    for tag, stat in sorted(summary.by_tag.items(), key=lambda kv: kv[1].mean_pass_rate or 0):
        out.append(
            f"| {tag} | {stat.n_items} | {_pct(stat.mean_pass_rate)} | "
            f"{_pct(stat.pass_hat_k.value)} |"
        )
    out += [
        "", "## Cost and latency per turn (agent calls only)", "",
        "| group | turns | calls/turn | in tok | out tok | USD/turn as run | "
        "USD/turn at gpt-4o-mini prices | latency p50 / p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = [(kind, g.cost_latency) for kind, g in summary.groups.items()]
    rows.append(("all", summary.all_model_turns))
    for name, c in rows:
        out.append(
            f"| {name} | {c.turns} | {_num(c.llm_calls_per_turn_mean, '')} | "
            f"{_num(c.input_tokens_per_turn_mean, '.0f')} | "
            f"{_num(c.output_tokens_per_turn_mean, '.0f')} | "
            f"{_num(c.usd_per_turn_mean, '.6f')} | {_num(c.usd_per_turn_if_gpt_4o_mini, '.6f')} | "
            f"{_num(c.latency_ms_p50, '.0f')} / {_num(c.latency_ms_p95, '.0f')} |"
        )
    out += ["", "## Items that failed every attempt, and flaky ones", ""]
    for kind, g in summary.groups.items():
        labelled = (("always failed", g.always_failing_items), ("flaky", g.flaky_items))
        for label, ids in labelled:
            for item_id in ids:
                out.append(f"- {item_id} ({kind}, {label}): {golden[item_id]['tag']}")
    out += ["", "## What this sample size does not support", ""]
    out += [f"- {c}" for c in summary.caveats]
    return "\n".join(out) + "\n"


def render_comparison(new: QuantBaseline, old: QuantBaseline) -> str:
    """
    Two baselines side by side. A difference is called a change only when the new value falls
    outside the 95% interval the old baseline carries; anything inside it is noise at this n.
    """
    rows = compare_with_baseline(new.summary, old)
    out = [
        "# Quant failure suite - before and after", "",
        f"- before: `{old.meta.model}`, completion cap {old.meta.max_tokens} "
        f"tokens, {old.meta.date_utc}",
        f"- after: `{new.meta.model}`, completion cap {new.meta.max_tokens} "
        f"tokens, {new.meta.date_utc}",
        "", "| metric | before (n, 95% interval) | after | verdict |", "|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r.metric} | {r.baseline:.2f} (n={r.baseline_n}, "
            f"{_ci(r.baseline_ci95)}) | {r.current:.2f} | {r.verdict} |"
        )
    moved = [r for r in rows if r.verdict != "within noise"]
    out += [
        "",
        f"{len(moved)} of {len(rows)} metrics moved outside the old interval."
        if moved else
        f"None of the {len(rows)} metrics moved outside the old interval: at this sample size "
        "the comparison cannot tell the two runs apart.",
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render a Quant suite baseline as markdown")
    p.add_argument("baseline", nargs="?", default=str(DEFAULT_BASELINE))
    p.add_argument("--against", default=None, help="an older baseline to compare with")
    args = p.parse_args(argv)
    doc = QuantBaseline.model_validate_json(Path(args.baseline).read_text())
    if args.against:
        print(render_comparison(doc, QuantBaseline.model_validate_json(
            Path(args.against).read_text())), end="")
    else:
        print(render(doc), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
