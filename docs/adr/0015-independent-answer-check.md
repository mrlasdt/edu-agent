# Quant solve turns check the answer independently: the model writes the equation, a solver computes it

A Quant solve turn used to check the final answer by sending it to the math verifier as both the expression and the expected value (`QuantAgent.verify_well_formed`, formerly `_verify`). That shows the answer parses and evaluates. It cannot show the answer is right: the offline eval's `wrong_but_verified` measured it, and 16 of 16 seeded wrong answers passed. The check was relabelled "well-formed" and the turn record said `verified_independent: false`. This ADR is the check that is independent.

## Status

Accepted. Implemented for GRE Quant solve mode. Tutor mode is unchanged (it states no answer to check). AW and IELTS are out of scope.

## Decision

After the model's solve attempt passes the well-formed check, one extra short model call (role `quant_agent`, prompt `prompts/quant_agent/formalize/v1.md`, temperature 0) writes the problem as `<expr>...</expr>`: an expression whose value is the answer, an equation whose solutions are the answer, or `NONE`. It sees the problem and the conversation, never the solve attempt's answer. The math verifier's sandbox (the same allow-list parser and sympy solver as every other math call) computes it, and `services/agent/src/agent/independent_check.py` compares the result with the claimed answer. The verdict is one of:

- `agrees`: the computed values equal the claimed ones (a decimal that is the correct rounding of the computed value counts, `3.3` for `10/3`; digit groups with thousands separators are read as one number, `1,200` is 1200, while `2, 3` stays a list).
- `disagrees`: they differ and the solver computed a plain number. The solver's value is kept and shown.
- `not_checkable`: there was nothing numeric to compare (the model wrote `NONE`, wrote no tag, the claimed answer is a letter or a word, the equation has an unknown left in it or no real solution, the parser rejected the text, the verifier was unreachable, the formalize call failed), or the expression is an equation with several roots and the claimed answer is some but not all of them (`2` against `-2, 2`: the question may ask for the positive root). A claimed value outside the roots stays `disagrees`. A failed check never fails the turn.

`disagrees` is a failed check in the existing ladder: attempt 2 (told what the solver computed), then the `math_escalation` attempt, then degrade with a failed `Verified` event (`passed=False`). `agrees` and `not_checkable` leave the well-formed result in charge, as before. The turn record carries `verified_independent` (the verdict), plus `independent_solver` and `independent_reason` when there is one. The reply's trailer carries `independent_check`, and the web app shows "Independent check: agrees | disagrees (solver: ...) | not checkable".

## Limits

- **ADR-0012's limit applies: numeric, recomputable answers only.** Quantitative comparisons (a letter), "which statement is true" and anything else without a number come out `not_checkable`, and the UI says "covers only questions with a numeric answer". A `not_checkable` answer is never counted as verified.
- **"Agrees" means two routes reached the same number, not that the problem was read correctly.** The expression comes from the same model. A misreading repeated in the working and in the equation agrees with itself. What the check removes is the arithmetic and algebra slip that the old check could not see.
- **A wrong equation can produce a false `disagrees`** (the model writes an equation for the unknown when the question asks for the sum of the solutions, for example). The prompt tells it to write `NONE` for that case; the cost of a miss is a retry, and the learner is shown both numbers. A claimed answer that is only some of an equation's roots is not counted as a disagreement (see `not_checkable` above), so a restricted root does not cost a retry.
- The model's expression is untrusted text: it only ever reaches the sandbox's allow-list parser (ADR-0011's math verifier, `tests/test_math_verifier_untrusted_input.py`).

## Cost

One short extra call per solve turn whose answer is well-formed (an answer that does not parse skips it), roughly 400 input tokens (the 1.4 KB prompt plus the problem) and a few dozen output tokens (estimated from the prompt size, not measured on a live model). The offline Quant suite shows 2.0 model calls per solve turn instead of 1.0. The expression is asked for once per turn, not once per retry. Latency adds one short call plus one sandbox run.

## Considered options

- **A. Ask for the expression inside the solve reply (no extra call).** Cheaper, but the equation is written in the same context as the answer, after the working, so it tends to be the working restated. Rejected: that is the self-compare again.
- **B. A separate short call that never sees the answer (chosen).** Costs a call; the equation comes from the problem alone.
- **C. The math tools (`MATH_TOOLS_ENABLED`).** The model calls the solver during its working. That makes its working better; it does not check it. Left as is.
- **D. A second full solve by another model.** More independent, much more expensive, and still a model grading a model.

## Measurement

`wrong_but_verified` in the quant suite now counts wrong answers the independent check agreed with. Offline, with each item's own `expression` standing in for the model's equation (so it tests the comparison, not how well a model writes the equation): 0 of 16 seeded wrong answers agreed (14 disagreed, 2 not checkable: the quantitative comparisons), against 16 of 16 for the well-formed check alone; the golden keys themselves get 14 agrees, 0 disagrees, 2 not checkable. How often a real model writes a faithful equation, and how often it produces a false `disagrees`, is measured only by a live run (`--mode live --suite quant`), which this change did not run.
