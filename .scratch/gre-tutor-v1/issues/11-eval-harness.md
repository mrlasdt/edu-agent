Status: ready-for-agent

## What to build

The eval harness CLI tool and CI gate. `python evals/run_eval.py` runs the golden set against the live Agent service, scores with deterministic (Quant) and LLM (AW) judges, prints a report, and exits non-zero on regression. Path-gated CI step runs 10% sample with Haiku judge; nightly cron runs full set with GPT-4o judge.

Golden set fixtures: `evals/golden/gre_quant.jsonl` (150 ETS-sourced problems, 4 question types) and `evals/golden/gre_aw.jsonl` (24 ETS prompts × 6 anchor essays). Judges: Quant uses sympy ground-truth comparison; AW uses LLM prompted with rubric + anchor essays.

A `human_review_queue` Postgres table is populated for: low-confidence AW scores, boundary scores (3.5/4.5), and 1% of `verifier_fail` events. `gotohuman` hook is a stub (writes to the table; real API integration in v2).

## Acceptance criteria

- [ ] `python evals/run_eval.py --suite quant --sample 0.1` runs the Quant golden set at 10% sample and prints per-question-type accuracy
- [ ] `python evals/run_eval.py --suite aw --sample 0.1` runs the AW golden set and prints avg rubric score + style-adherence rate
- [ ] Quant judge: `sympy.solve` comparison; binary correct/incorrect; no LLM call
- [ ] AW judge: LLM call with rubric + anchor essays in context; returns score 0–6 with rationale
- [ ] Judge calibration self-check: anchor essays re-scored; alert if any drifts ≥1 point
- [ ] Regression check: exits non-zero if Quant accuracy drops >5% or AW avg score drops >5% vs stored baseline
- [ ] `human_review_queue` table populated for boundary AW scores and `verifier_fail` events
- [ ] CI workflow runs 10% sample on PRs touching `prompts/`, `services/agent/`, `services/gateway/`, `evals/`, `config/`
- [ ] Nightly cron runs full set on main

## Blocked by

`.scratch/gre-tutor-v1/issues/05-quant-agent-solve-mode.md`
`.scratch/gre-tutor-v1/issues/09-aw-agent-solve-mode-style.md`
