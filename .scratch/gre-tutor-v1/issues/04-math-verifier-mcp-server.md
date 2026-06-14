Status: completed

## What to build

A local MCP server that exposes a `verify_math` tool backed by a sympy sandbox. The Agent service is its MCP client. The server runs as a separate process (Unix socket in Phase 1); the Agent service connects at startup and holds the connection. This is also the math-focused tool exposed by the Candidate MCP server (issue 15).

The sandbox receives a symbolic expression and an expected answer, evaluates the expression with sympy, and returns `{verified: bool, computed: str, error: str | null}`. It enforces a strict timeout (2s) and runs in a subprocess to contain crashes.

## Acceptance criteria

- [ ] MCP server exposes a `verify_math` tool with schema `{expression: str, expected: str}`
- [ ] `verify_math("x**2 - 5*x + 6 = 0", "x=2 or x=3")` returns `{verified: true, computed: "...", error: null}`
- [ ] `verify_math("x**2 + 1 = 0", "x=1")` returns `{verified: false, ...}`
- [ ] A malformed expression returns `{verified: false, error: "parse error: ..."}` without crashing the server
- [ ] Expressions exceeding the 2s timeout return `{verified: false, error: "timeout"}`
- [ ] Server starts and is reachable from the Agent service (Docker Compose service, Unix socket)
- [ ] Agent service's `verify-math` Skill calls this MCP tool and returns a `VerificationResult`
- [ ] Unit tests for the sandbox logic do not start the full MCP server

## Blocked by

`.scratch/gre-tutor-v1/issues/01-local-dev-stack.md`
