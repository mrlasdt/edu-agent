# 05 - Tool inputs are untrusted

## The problem you would feel

In solve mode EduAgent asks the math verifier to check the model's final answer. A learner can
steer what the model writes as that answer, just by how they word the question. Before this
change, a model reply ending in

```
**Answer: __import__("os").getpid()**
```

made the verifier return its own process id. The text was not read as math. It was run as
Python. In the compose stack that is the math-verifier container (which holds no API keys), but
it is still code execution driven by model output, and every later slice that gives the model
more ways to reach the verifier makes it worse.

## The practice

**Treat everything a model sends to a tool as untrusted input**, exactly like text from a web
form. The model is not malicious, but its output is shaped by whatever is in its context: the
learner's question, retrieved documents, tool results. Anything that reaches an interpreter
(`eval`, a shell, SQL, a template engine) needs the same defences you would use for user input:
an allow-list of what is accepted, a hard limit on size and time, and a way to stop the work
that does not depend on the work's cooperation. Keep credentials and identity out of the
model's reach for the same reason. See Anthropic's engineering post
[Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
for the tool-design side of this.

## Why EduAgent needs it here

The verifier's job is to be the part of the system that is *not* the model. If model text can
run code inside it, the independent check is no longer independent, and the one process meant
to stay simple becomes the one attackers (or an unlucky prompt) can steer.

The old code had two weaknesses, and the previous "sandbox" fixed neither:

- `sympy.parse_expr` passes its input to Python's `eval`. Any text became code.
- The "sandbox" was a 2-second timeout around a thread. Python cannot stop a running thread,
  so a runaway computation (try `9**9**9**9`) kept burning CPU after the timeout fired.

## Code tour

Read these in order. They are named by symbol so the tour survives edits.

1. **`services/math_verifier/src/math_verifier/safe_parser.py`** - the only door from model text
   to sympy.
   - `parse_math(text)` is the public function. It returns a sympy expression or raises
     `MathSyntaxError("not a math expression")`.
   - `_tokenize` is the allow-list. It accepts only numbers, `+ - * / ^ ** ! , ( )`, names in
     `FUNCTIONS` or `CONSTANTS`, and variables. A word of only letters that is not listed is
     split into single-letter variables (`xy` becomes `x y`, a product); it is never looked up
     or evaluated as a name. Two such words in a row (`import os`, `the values`) are rejected as
     prose. Underscores, digits inside a word, dots, brackets, and quotes fail
     here, before any parsing. It also caps length at
     `MAX_EXPRESSION_LENGTH` and rejects unbalanced parentheses and top-level commas.
   - `parse_math` rebuilds the string from the validated tokens, so the checker and the parser
     see the same stream, then calls `parse_expr` with an explicit `_NAMESPACE`. That namespace
     contains `"__builtins__": {}` on purpose: `eval` silently injects the real builtins into any
     namespace that lacks the key.
2. **`services/math_verifier/src/math_verifier/sandbox.py`** - where the work runs.
   - `_sync_verify` is the verification logic. Every parse in it goes through `parse_math`
     (`_parse_safe`, `_solve_equation`, the value-list and plain-expression paths).
   - `verify_expression_sync` runs `_sync_verify` in a throwaway child process (`_child_main`) and
     waits on a pipe with `poll(timeout)`. On timeout it kills the child with `kill()` and reaps it.
     `verify_expression` is the async wrapper over a small bounded thread pool, so a burst of
     requests cannot fork without limit.
   - `_child_main` calls `_limit_resources`, a CPU and memory ceiling behind the wall-clock kill
     (best effort: not every OS honours them).
   - The child comes from `_CTX`, a `forkserver` context. A small helper process is started once
     with sympy already imported, and each check is forked from it in milliseconds instead of
     paying for a cold interpreter. See "Gotchas" below.
3. **`services/math_verifier/src/math_verifier/server.py`** - `verify_math` (MCP) and
   `verify_math_http` (the plain-HTTP route the agent calls) both go through `verify_expression`.
   Nothing else changed.

## Before and after

Same probe, real server, both versions (full commands in
[`results/05-safe-math.md`](results/05-safe-math.md)):

```
BEFORE  {"verified":false,"computed":"83685","error":null}
        ^ the verifier's own process id: the text ran as Python

AFTER   {"verified":false,"computed":"","error":"parse error: not a math expression"}
        and a real equation still verifies:
        {"verified":true,"computed":"[2, 3]","error":null}
```

The learner's turn is unaffected: the reply is still streamed, the verifier says "could not
verify", and the existing retry, escalate, degrade path runs (`verifier_fail: true`).

## Trade-offs, and how to extend

**The allow-list rejects some valid notation.** That is the price of the guarantee. Known gaps:

| You write | What happens | What to do |
|---|---|---|
| `xy`, `3xy`, `12 cm` (letters glued together) | read as a product of single-letter variables: `x*y`, `3*x*y`, `12*c*m` | nothing to do; a unit word is not recognised as a unit |
| `1e3` | rejected | write `1000` |
| `√2`, `×`, `÷`, `−` (Unicode) | rejected | normalise in `agent/answers.py` before the handoff |
| `x_1`, `x1` | rejected | use single letters |
| `sec(x)` and other functions not listed | read as the product `s*e*c*x`, not the function | add to `FUNCTIONS` |

To accept a new function, add one line to `FUNCTIONS` in `safe_parser.py`, for example
`"sec": sp.sec`. To accept a new constant, add it to `CONSTANTS`. Nothing outside those two
tables can be named in an expression, so extending the list is a code review of one line.

Two behaviours are intentionally *wider* than before: `^` now means power (it used to mean
bitwise xor, which no learner intends), and `5!` and `2(x+1)` keep working.

**Not a full OS sandbox.** The child process is killed on timeout and has CPU and memory limits,
but it can still read files the verifier's user can read. The primary defence is that no
hostile text reaches `eval` at all. The container boundary and the absence of API keys in the
verifier container are the second layer. If you ever need to run *arbitrary* code, use a real
sandbox (a container or gVisor with no network); an allow-list parser is the right tool only
because we only ever need arithmetic.

**Latency.** A check costs roughly 10 to 50 ms more than the old in-process call
(fork plus a pipe). That is invisible against a model call, but it is why the 2,400-check loop
in `tests/test_exam_set.py` calls `_sync_verify` directly on trusted generated input.

## Gotchas

- `multiprocessing` with `forkserver` re-imports the main script in each child (as
  `__mp_main__`). Scripts that call the verifier should keep their entry point under
  `if __name__ == "__main__":`. A script piped through stdin cannot be re-imported at all.
- The offline network guard in tests (`block_network` in `tests/conftest.py`) allows Unix
  sockets, because the forkserver is reached over one.

## Try it

```
uv run python -c "import asyncio; from services.math_verifier.src.math_verifier.sandbox import verify_expression as v; print(asyncio.run(v('__import__(\"os\").getpid()','1')))"
```

Expected: `VerificationResult(verified=False, computed='', error='parse error: not a math expression')`.
On the commit before this change the same command prints the process id in `computed`.

Run the regression suite: `uv run pytest tests/test_math_verifier_untrusted_input.py tests/test_math_verifier_sandbox.py -q`

## Exercise

1. Add `"sec": sp.sec` to `FUNCTIONS`. Check `verify_expression("sec(0)", "1")` is verified.
2. Confirm `verify_expression("secx", "1")` is still rejected (a glued name is not a listed
   function), and that the 18 strings in `HOSTILE` in the test file still all fail.
3. Add one more hostile string of your own to `HOSTILE` and watch which layer rejects it: the
   tokenizer (a name or character not on the list) or the length cap.
