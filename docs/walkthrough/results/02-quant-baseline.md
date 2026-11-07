# Quant failure suite - baseline

- model: `openrouter/recursive/flint-v1.0:free` (the product default)
- samples per item: 3, completion cap 3000 tokens, temperature 0.2, prompts {'tutor': 'v3', 'solve': 'v2'}
- date 2026-11-06, commit `1a8d155`
- Measured on the model named in meta.model. Every rate carries its n and 95% interval; a difference inside an interval is noise.

## Headline (rate, 95% interval)

| group | items | one attempt | pass@k | pass^k |
|---|---|---|---|---|
| solve (final answer right and well-formed) | 16 | 0.92 (0.69-0.98) | 1.00 (0.81-1.00) | 0.88 (0.64-0.96) |
| tutor, one reply | 12 | 0.97 (0.72-1.00) | 1.00 (0.76-1.00) | 0.92 (0.65-0.99) |
| tutor, 4-turn conversation | 4 | 0.67 (0.25-0.92) | 0.75 (0.30-0.95) | 0.50 (0.15-0.85) |

pass@3: at least one of 3 attempts passed. pass^3: all 3 passed (what a learner who asks the same thing repeatedly experiences). Security (no model, n=8 hostile strings): 8 rejected, 95% interval 0.68-1.00.

## Which checks fail (share of attempts passing)

| group | check | passed |
|---|---|---|
| solve | correct | 47/48 (0.98) |
| solve | verifier_accepted | 45/48 (0.94) |
| tutor | citations_ok | 36/36 (1.00) |
| tutor | moves_forward | 36/36 (1.00) |
| tutor | no_latex | 35/36 (0.97) |
| tutor | no_leak | 36/36 (1.00) |
| tutor_multiturn | citations_ok | 12/12 (1.00) |
| tutor_multiturn | moves_forward | 12/12 (1.00) |
| tutor_multiturn | no_latex | 11/12 (0.92) |
| tutor_multiturn | no_leak | 9/12 (0.75) |
| tutor_multiturn | stays_on_problem | 12/12 (1.00) |

## Wrong but verified

- seeded wrong answers the independent check agreed with: 0/16 (95% interval 0.00-0.19); 14 disagreed, 2 not checkable
- the same answers through the well-formed check alone (before the independent check existed): 16/16 accepted
- control, the golden key through the independent check: 14/16 agrees, 0 disagrees, 2 not checkable
- real solve attempts that were wrong and still got "agrees": 0/48 attempts
- the independent check covers numeric answers only; a wrong answer is not counted as verified when the check could not run

## By failure type

| type | items | one attempt | pass^k |
|---|---|---|---|
| qc-label | 2 | 0.33 | 0.00 |
| multi-turn-drift | 4 | 0.67 | 0.50 |
| answer-leak | 5 | 0.93 | 0.80 |
| answer-format | 4 | 1.00 | 1.00 |
| arithmetic-heavy | 6 | 1.00 | 1.00 |
| arithmetic-slip | 2 | 1.00 | 1.00 |
| capability | 2 | 1.00 | 1.00 |
| citation-wording | 2 | 1.00 | 1.00 |
| plain-text-math | 2 | 1.00 | 1.00 |
| vague-reply | 3 | 1.00 | 1.00 |

## Cost and latency per turn (agent calls only)

| group | turns | calls/turn | in tok | out tok | USD/turn as run | USD/turn at gpt-4o-mini prices | latency p50 / p95 ms |
|---|---|---|---|---|---|---|---|
| solve | 48 | 2.15 | 611 | 697 | 0.000000 | 0.000510 | 9936 / 22827 |
| tutor | 36 | 1.0 | 414 | 472 | 0.000000 | 0.000346 | 4273 / 11065 |
| tutor_multiturn | 48 | 1.0 | 494 | 307 | 0.000000 | 0.000258 | 3374 / 5900 |
| all | 132 | 1.42 | 515 | 494 | 0.000000 | 0.000374 | 5275 / 13388 |

## Items that failed every attempt, and flaky ones

- q004 (solve, flaky): qc-label
- q010 (solve, flaky): qc-label
- t005 (tutor, flaky): answer-leak
- m001 (tutor_multiturn, always failed): multi-turn-drift
- m002 (tutor_multiturn, flaky): multi-turn-drift

## What this sample size does not support

- Rates are over items, not over all possible learner questions; the items are self-written from past failures and are not a random sample.
- Cost and latency cover the agent's own model calls only; the gateway's guardrail calls (completeness check, answer-leak check) are not included.
- The leak check is an upper bound: it also counts a step number equal to the answer.
- Rule graders catch the failures someone wrote a rule for; a reply can pass every rule and still be a poor hint (read the transcripts).
- solve: n=16 items x 3 samples; a change smaller than about 0.16 cannot be told apart from noise.
- tutor: n=12 items x 3 samples; a change smaller than about 0.17 cannot be told apart from noise.
- tutor_multiturn: n=4 items x 3 samples; a change smaller than about 0.35 cannot be told apart from noise.
- security: 8 hostile strings; all passing does not bound the failure rate on strings nobody has tried.
- wrong_but_verified: 0 of 16 seeded wrong answers got "agrees" from the independent check (14 disagreed, 2 not checkable); the well-formed check alone accepted 16 of 16. 0 of 48 real solve attempts were wrong and still got "agrees". The seeded run uses each item's own expression as the model's expression, so it tests the comparison, not how well a model writes the equation; the seeded answers are one perturbation of each key, not every way a model can be wrong. The check covers numeric answers only (ADR-0012), and a model that misreads a problem the same way in its working and in its equation still agrees with itself.
