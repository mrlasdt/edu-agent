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

## Landscape: ways to make untrusted input safe to act on

The right defence depends on what the input is allowed to *do*. Arithmetic needs far less than "run this program". These are the main categories, roughly from lightest to heaviest. Stronger layers cost more and are for stronger threats; the best systems stack a light layer on top of a heavy one.

| Category | What it is | Example | Trade-off | When to use | EduAgent today |
|---|---|---|---|---|---|
| **Deny-list or regex filter** | Reject input that matches known-bad patterns. | Block the strings `import` and `__`. | Easy, but attackers find the spelling you did not list. Fails open. | Never as the only defence. | **Not used.** This is what the old code was closest to, and why it was unsafe. |
| **Allow-list parser** | Accept only a small, named grammar and reject everything else before it reaches an interpreter. | Numbers, `+ - * / ^`, a short list of functions, one-letter variables. | Rejects some valid input; the safest option when the task is narrow, because the unsafe things cannot even be spelled. | The input has a small, known shape (math, dates, IDs, queries). | **Used.** `_tokenize` and `parse_math` in `safe_parser.py`. |
| **AST validation** | Parse the input into a syntax tree with the language's own parser, then walk it and permit only listed node types. | Python's `ast` module: allow `BinOp` and `Constant`, reject `Call` and `Attribute`. | More expressive than a token list, but you must get the walk right for every node type and every language version. | Input is real code in a language you can parse, and you need more than a token list. | **Not used.** The tokenizer is stricter and simpler for arithmetic. |
| **Restricted namespace** | Run `eval` with empty or curated builtins and globals. | `eval(text, {"__builtins__": {}}, safe_names)`. | A speed bump only: Python object introspection has well-known ways around it. | As defence in depth, never alone. | **Used as a second layer.** `_NAMESPACE` sets `"__builtins__": {}`, behind the allow-list. |
| **Subprocess isolation with a kill** | Run the work in a separate process and kill it on a deadline. | A child process, a pipe, `poll(timeout)`, then `kill()`. | Stops runaway work that a thread cannot be stopped from; does not stop file or network access. Adds tens of milliseconds. | Any untrusted computation that might not finish. | **Used.** `verify_expression_sync` and `_child_main` in `sandbox.py`. |
| **Resource limits** | OS limits on CPU time, memory, processes, open files. | `setrlimit(RLIMIT_CPU, ...)`, `RLIMIT_AS`; cgroup limits in containers. | Cheap backstop; enforcement varies by operating system. | Alongside any subprocess. | **Used, best effort.** `_limit_resources` sets CPU and memory limits where the OS honours them. |
| **Container** | A namespaced process with its own filesystem view, optionally no network and no secrets. | Docker with a read-only filesystem, dropped capabilities and no network. | A real boundary, but a shared kernel: a kernel bug can cross it. Per-run containers are slow to start. | Services you deploy, and code you do not fully trust. | **Partly.** The verifier is its own container with no API keys, which limits what an escape would find. The compose file does not set a read-only filesystem or deny network, so treat it as a boundary of convenience, not hardening. |
| **User-space kernel or microVM** | A stronger isolation boundary than a plain container: a user-space kernel or a minimal virtual machine per workload. | gVisor, Firecracker, Kata Containers. | Strong isolation for arbitrary code; more setup, some overhead and compatibility gaps. | Running genuinely arbitrary code, such as user-submitted programs. | **Not used.** The input is arithmetic, not programs. |
| **Provider-hosted code execution** | The model provider runs the code in its own sandbox and returns the result. | Anthropic's code execution tool, OpenAI's code interpreter. | No sandbox to run, but vendor lock-in, per-use cost, and your data leaves your boundary. | You want model-written code run without operating a sandbox. | **Not used.** Our check is deterministic arithmetic, and it must stay independent of the model's provider. |
| **WebAssembly runtime** | Compile or run code inside a WASM VM that has no host access unless you grant it. | Pyodide (Python in WASM), Wasmtime. | Small attack surface and portable; limited library and native-extension support. | Embedding untrusted code in a process where containers are too heavy. | **Not used.** sympy is a large native-dependent library. |
| **Hosted sandbox service** | A service that creates a short-lived isolated machine per run. | E2B, Modal sandboxes. | Strong isolation with no infrastructure of your own; a dependency, network hop, and cost per run. | Agents that write and run real programs. | **Not used.** Same reason as microVMs. |
| **Credential and identity scoping** | Keep secrets, user identity and permissions out of the arguments the model controls. | The server attaches the signed-in user; the model never supplies a user id or token. | Needs the tool server to know who is calling, which means designing that channel. | Every tool that touches user data or paid APIs. | **Partly.** The verifier takes only an expression and holds no keys. A review of the other tools' arguments is still to come. |

Rule of thumb: pick the lightest layer that makes the dangerous thing *impossible to express*, then add the cheap layers (subprocess, limits, container) for the failures you cannot rule out. A heavier sandbox is the answer when you must run code you cannot constrain; an allow-list is the answer when you can constrain it.

Where this sits among the types of agent tools: a **client-defined function** (what the verifier is) runs in your process or service, so you own its input checking. A **provider-hosted** code execution tool moves that burden to the provider's sandbox but also moves your data. Chapter 4 maps all the tool types.

## Open-source options

| Tool | What it does | EduAgent's position |
|---|---|---|
| [SymPy `parse_expr`](https://docs.sympy.org/latest/modules/parsing.html) | Parses a string into an expression; it calls Python `eval` under the hood, and its documentation says not to use it on untrusted input. | **Used behind the allow-list.** Only tokens that passed `_tokenize` and a namespace with no builtins reach it. |
| [RestrictedPython](https://restrictedpython.readthedocs.io/) | Compiles Python with a restricted set of operations and guarded access, as used in the Zope ecosystem. | **Not adopted.** It restricts Python code, but our input is never Python; an allow-list parser removes the need. Reasonable if learners were to submit real code. |
| [E2B](https://e2b.dev/docs) | Open-source sandboxed cloud environments for running AI-generated code. | **Not adopted.** A network service, an account and a cost per run, for a job that is arithmetic. |
| [gVisor](https://gvisor.dev/docs/) and [Firecracker](https://firecracker-microvm.github.io/) | A user-space kernel and a lightweight microVM, the building blocks behind many hosted sandboxes. | **Not adopted.** Infrastructure work with no payoff when the input is constrained. The right tools if you need to run arbitrary code. |
| [Pyodide](https://pyodide.org/en/stable/) and [Wasmtime](https://docs.wasmtime.dev/) | Python and a general runtime for WebAssembly. | **Not adopted.** Not a fit for sympy's native dependencies. |
| [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) and [`resource`](https://docs.python.org/3/library/resource.html) (standard library) | Child processes and OS resource limits. | **Used.** The subprocess runner needs no third-party package. |

No dependency was added for this chapter. Everything in the code is the standard library plus the sympy that was already there.

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
2. Confirm `verify_expression("secx", "1")` is still read as the product `c*e*s*x` (a glued
   name is not a listed function), and that the 18 strings in `HOSTILE` in the test file still all fail.
3. Add one more hostile string of your own to `HOSTILE` and watch which layer rejects it: the
   tokenizer (a name or character not on the list) or the length cap.
