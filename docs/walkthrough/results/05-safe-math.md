# Results: safe parsing of model-supplied math

Same harmless probe before and after. The probe asks the verifier to "verify" the text
`__import__("os").getpid()`. If the verifier computes a process id, the text ran as Python.

## Before (commit `bc9a9b1`)

Command (extracts the old code to a scratch directory, starts the real verifier server on port 18091, and posts to the plain-HTTP route the agent calls):

```
git archive bc9a9b1 services shared | tar -x -C /tmp/old
(cd /tmp/old && MATH_VERIFIER_PORT=18091 PYTHONPATH=/tmp/old \
  python -m services.math_verifier.src.math_verifier.server &)
curl -s -X POST localhost:18091/tools/verify_math -H 'content-type: application/json' \
  -d '{"expression": "__import__(\"os\").getpid()", "expected": "1"}'
```

Output (the number is the server process's own id, confirmed with `lsof -ti tcp:18091 -sTCP:LISTEN`):

```
{"verified":false,"computed":"83685","error":null}
```

## After (this change)

Same request against the new code on port 18092:

```
{"verified":false,"computed":"","error":"parse error: not a math expression"}
```

A real equation on the same server still verifies:

```
{"verified":true,"computed":"[2, 3]","error":null}
```

## The turn still answers

`test_a_solve_turn_whose_final_answer_is_hostile_degrades_and_still_answers` (in
`tests/test_math_verifier_untrusted_input.py`) runs a solve turn whose model reply ends in
`**Answer: __import__("os").getpid()**`. The learner still receives the reply text, with
a failed `Verified` event (`passed=False`), and the process id appears nowhere.

Reproduce the "after" numbers: `uv run pytest tests/test_math_verifier_untrusted_input.py -q`
