# Quant failure suite - baseline

- model: `openrouter/cohere/north-mini-code:free` (NOT the product default `openai/gpt-4o-mini`)
- samples per item: 3, completion cap 3000 tokens, temperature 0.2, prompts {'tutor': 'v3', 'solve': 'v2'}
- date 2026-11-03, commit `592069b`
- Measured on the model named in meta.model. That is NOT the product default (openai/gpt-4o-mini); re-run with the default model before comparing a change against these numbers. Every rate carries its n and 95% interval; a difference inside an interval is noise.

## Headline (rate, 95% interval)

| group | items | one attempt | pass@k | pass^k |
|---|---|---|---|---|
| solve (final answer right and verified) | 10 | 0.80 (0.50-1.00) | 0.80 (0.49-0.94) | 0.80 (0.49-0.94) |
| tutor, one reply | 12 | 0.83 (0.67-0.97) | 1.00 (0.76-1.00) | 0.67 (0.39-0.86) |
| tutor, 4-turn conversation | 4 | 0.25 (0.08-0.33) | 0.75 (0.30-0.95) | 0.00 (0.00-0.49) |

pass@3: at least one of 3 attempts passed. pass^3: all 3 passed (what a learner who asks the same thing repeatedly experiences). Security (no model, n=8 hostile strings): 8 rejected, 95% interval 0.68-1.00.

## Which checks fail (share of attempts passing)

| group | check | passed |
|---|---|---|
| solve | correct | 30/30 (1.00) |
| solve | verifier_accepted | 24/30 (0.80) |
| tutor | citations_ok | 34/36 (0.94) |
| tutor | moves_forward | 36/36 (1.00) |
| tutor | no_latex | 32/36 (0.89) |
| tutor | no_leak | 35/36 (0.97) |
| tutor_multiturn | citations_ok | 7/12 (0.58) |
| tutor_multiturn | moves_forward | 12/12 (1.00) |
| tutor_multiturn | no_latex | 12/12 (1.00) |
| tutor_multiturn | no_leak | 9/12 (0.75) |
| tutor_multiturn | stays_on_problem | 10/12 (0.83) |

## By failure type

| type | items | one attempt | pass^k |
|---|---|---|---|
| qc-label | 2 | 0.00 | 0.00 |
| multi-turn-drift | 4 | 0.25 | 0.00 |
| citation-wording | 2 | 0.67 | 0.50 |
| answer-leak | 5 | 0.80 | 0.60 |
| vague-reply | 3 | 0.89 | 0.67 |
| answer-format | 4 | 1.00 | 1.00 |
| arithmetic-slip | 2 | 1.00 | 1.00 |
| capability | 2 | 1.00 | 1.00 |
| plain-text-math | 2 | 1.00 | 1.00 |

## Cost and latency per turn (agent calls only)

| group | turns | calls/turn | in tok | out tok | USD/turn as run | USD/turn at gpt-4o-mini prices | latency p50 / p95 ms |
|---|---|---|---|---|---|---|---|
| solve | 30 | 1.4 | 324 | 636 | 0.000000 | 0.000430 | 5973 / 21630 |
| tutor | 36 | 1.0 | 412 | 528 | 0.000000 | 0.000378 | 6971 / 17321 |
| tutor_multiturn | 48 | 1.0 | 481 | 271 | 0.000000 | 0.000235 | 3093 / 8042 |
| all | 114 | 1.11 | 418 | 448 | 0.000000 | 0.000332 | 5265 / 14983 |

## Items that failed every attempt, and flaky ones

- q004 (solve, always failed): qc-label
- q010 (solve, always failed): qc-label
- t002 (tutor, flaky): answer-leak
- t004 (tutor, flaky): answer-leak
- t007 (tutor, flaky): citation-wording
- t008 (tutor, flaky): vague-reply
- m001 (tutor_multiturn, always failed): multi-turn-drift
- m002 (tutor_multiturn, flaky): multi-turn-drift
- m003 (tutor_multiturn, flaky): multi-turn-drift
- m004 (tutor_multiturn, flaky): multi-turn-drift

## What this sample size does not support

- Rates are over items, not over all possible learner questions; the items are self-written from past failures and are not a random sample.
- Cost and latency cover the agent's own model calls only; the gateway's guardrail calls (completeness check, answer-leak check) are not included.
- The leak check is an upper bound: it also counts a step number equal to the answer.
- Rule graders catch the failures someone wrote a rule for; a reply can pass every rule and still be a poor hint (read the transcripts).
- solve: n=10 items x 3 samples; a change smaller than about 0.25 cannot be told apart from noise.
- tutor: n=12 items x 3 samples; a change smaller than about 0.24 cannot be told apart from noise.
- tutor_multiturn: n=4 items x 3 samples; a change smaller than about 0.33 cannot be told apart from noise.
- security: 8 hostile strings; all passing does not bound the failure rate on strings nobody has tried.
