# Eval pipeline

The agent has a three-layer eval pipeline: **offline** (closed golden set + LLM and deterministic judges, run in CI and nightly), **online** (per-turn telemetry, candidate feedback, drift detection on production traffic), and **integration** (eval-as-CI gates, dashboards, a human-in-the-loop hook). The Quant golden set is ~150 publicly-released ETS problems; the AW golden set is 24 ETS prompts, each accompanied by their 6 published anchor essays at scoring tiers 1–6. The AW judge is an LLM (Sonnet nightly, Haiku per-PR for cost) prompted with the rubric *and* the anchor essays in-context, so absolute scores stay calibrated across runs. The Quant judge is deterministic — direct comparison to ground truth via the sympy verifier — no LLM judge.

## Status

Accepted.

## Considered options

- **Per-PR cost optimisations** — the original proposal (20% sample, Sonnet judge on every PR) was reduced to: (a) path-based gating (eval only runs when a PR touches prompts, agents, skills, evals, or model configs); (b) 10% sample (~15 Quant + 3 AW prompts); (c) Haiku judge for PR evals, Sonnet for nightly. Combined target: ~$0.10–$0.30 per eval run, $0 for code-only PRs.
- **Synthetic golden items** — rejected for v1: synthetic problems introduce labelling bias that's hard to detect; public ETS items are the credible baseline.
- **Opus as the AW judge** — rejected: Sonnet with anchor essays is calibrated by data, not by raw model strength, and is ~5× cheaper with no measurable quality drop on the rubric task. Revisit if calibration check fails.
- **Continuous human-graded subset** — rejected for v1 as a workflow; replaced by the Human review queue with `gotohuman` as the v2 integration. v1 ships a stub.
- **Score predictor (real-GRE-score from session history)** — rejected for v1: the most product-attractive feature and the noisiest one without paired data. Revisit when we have it.

## Consequences

- **The eval pipeline is path-gated in CI.** Code-only PRs skip eval entirely. PRs touching prompts/agents/skills/evals/model-configs run the per-PR slice with the Haiku judge. Nightly main runs the full set with the Sonnet judge and a 7-day baseline for regression alerts.
- **Eval cost has a hard cap.** $20 per nightly run; the runner kills itself before exceeding it.
- **The Human review queue is wired in v1.** Trigger conditions: low-confidence judge scores; boundary AW scores (3.5, 4.5); drift-alert investigation samples; 1% sample of `verifier-fail` events. v1 writes flagged items to a `human_review_queue` table; v2 swaps the writer for a `gotohuman` API client. The eval dashboard surfaces the queue.
- **The AW judge has a continuous self-check.** Every nightly run feeds the six anchor essays back through the judge; if any scores drift, alert. This is the only way to detect judge regression without an external grading source.
- **No "human" surface in v1.** No human-grader UI, no labeling pipeline, no rubric-tool product. The Human review queue is the integration point that lets us add those things in v2 without re-architecting eval.
- **Telemetry is structured and trace-IDed from day one.** Per-turn logs capture input, mode, agent invoked, skills invoked, retries, escalations, output, judge scores, latency, cost, and trace ID. Without trace IDs, multi-step turn debugging is impossible.
- **Drift detection runs against three signals**: avg AW rubric score (24h window vs 7-day baseline), `verifier-fail` rate (24h window vs baseline), and per-turn cost (rolling). Alerts fire to the dashboard, sample items get added to the Human review queue.
