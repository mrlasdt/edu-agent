from __future__ import annotations

import asyncio
import multiprocessing
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.connection import Connection

import sympy as sp

from services.math_verifier.src.math_verifier.safe_parser import (
    MAX_EXPRESSION_LENGTH,
    NOT_MATH,
    parse_math,
)

# The budget covers the wait for the child's reply (parsing and solving). Starting the child
# comes first and is not counted.
DEFAULT_TIMEOUT = 5.0
MAX_CONCURRENT_CHECKS = 4
_MEMORY_LIMIT_BYTES = 1 << 30
_CPU_LIMIT_SECONDS = 10

# Each check runs in its own short-lived child forked from a "forkserver": a small helper
# process, started once with this module (and sympy) already imported, so a fork costs
# milliseconds instead of a cold interpreter start.
_CTX = multiprocessing.get_context("forkserver")
_CTX.set_forkserver_preload([__name__])
_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHECKS, thread_name_prefix="math-check")


@dataclass
class VerificationResult:
    verified: bool
    computed: str
    error: str | None = None


# ── internal helpers ──────────────────────────────────────────────────────────


def _parse_safe(s: str) -> sp.Expr | None:
    try:
        return parse_math(s)
    except Exception:
        return None


def _value_list(s: str) -> list[str]:
    """Split a value list ("2, 3", "x=2 or x=3", "[2, 3]") into bare value strings."""
    s = s.strip().lstrip("[{").rstrip("]}]")
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch in ",;" and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    pieces = [q for part in parts for q in re.split(r"\bor\b|\band\b", part, flags=re.IGNORECASE)]
    return [re.sub(r"^\s*\w+\s*=\s*", "", q.strip()) for q in pieces if q.strip()]


def _normalize_expected(expected: str) -> set[sp.Expr]:
    """
    Parse an expected-answer string into a set of sympy values.
    Handles: "2", "2, 3", "x=2 or x=3", "[2, 3]", "{2, 3}"
    """
    result: set[sp.Expr] = set()
    for cleaned in _value_list(expected):
        expr = _parse_safe(cleaned)
        if expr is not None:
            result.add(sp.simplify(expr))
    return result


def _is_value_list(expression: str) -> bool:
    """A list of two or more values ("2, -2"), not an equation to solve."""
    values = _value_list(expression)
    return len(values) > 1 and not any("=" in v for v in values)


def _solve_equation(expression: str) -> tuple[str, set[sp.Expr]]:
    """
    Solve an equation 'lhs = rhs'. Returns (computed_str, solution_set).
    Raises ValueError on parse failure.
    """
    parts = expression.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse as equation: {expression!r}")

    lhs = parse_math(parts[0])
    rhs = parse_math(parts[1])
    diff = lhs - rhs

    free_syms = sorted(diff.free_symbols, key=str)
    if not free_syms:
        # Pure numeric equality
        simplified = sp.simplify(diff)
        val = sp.sympify(simplified == 0)
        return str(val), {val}

    solutions = sp.solve(diff, free_syms[0] if len(free_syms) == 1 else free_syms)
    solution_list = solutions if isinstance(solutions, list) else [solutions]
    solution_set = {sp.simplify(s) for s in solution_list}
    return str(sorted(solution_set, key=str)), solution_set


def _sync_verify(expression: str, expected: str) -> VerificationResult:
    """Run sympy verification synchronously. Runs only inside the worker subprocess."""
    if not expression or not expression.strip():
        return VerificationResult(
            verified=False, computed="", error="parse error: empty expression"
        )
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return VerificationResult(verified=False, computed="", error=f"parse error: {NOT_MATH}")
    try:
        if _is_value_list(expression):
            computed_set = {sp.simplify(parse_math(v)) for v in _value_list(expression)}
            computed_str = str(sorted(computed_set, key=str))
        elif "=" in expression:
            computed_str, computed_set = _solve_equation(expression)
        else:
            expr = parse_math(expression)
            val = sp.simplify(expr)
            computed_str = str(val)
            computed_set = {val}

        expected_set = _normalize_expected(expected)
        verified = bool(computed_set and expected_set and computed_set == expected_set)
        return VerificationResult(verified=verified, computed=computed_str)
    except Exception as exc:
        return VerificationResult(verified=False, computed="", error=f"parse error: {exc}")


# ── public API ────────────────────────────────────────────────────────────────


def _fail(error: str) -> VerificationResult:
    return VerificationResult(verified=False, computed="", error=error)


def _limit_resources() -> None:
    """Backstop behind the parent's wall-clock kill. Best effort: not every OS honours these."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_LIMIT_SECONDS, _CPU_LIMIT_SECONDS))
        resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_LIMIT_BYTES, _MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        pass


def _child_main(conn: Connection, expression: str, expected: str) -> None:
    """Entry point of the throwaway child: verify, send the result, exit."""
    _limit_resources()
    result = _sync_verify(expression, expected)
    conn.send((result.verified, result.computed, result.error))
    conn.close()


def verify_expression_sync(
    expression: str, expected: str, timeout: float = DEFAULT_TIMEOUT
) -> VerificationResult:
    """
    Check the expression in a throwaway child process that is killed on timeout.
    Never raises - returns a VerificationResult with error set on failure.

    A thread cannot be stopped from outside, so a runaway computation would keep burning CPU
    after its timeout. A child process can be killed, and it shares no state with the server.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH or len(expected) > MAX_EXPRESSION_LENGTH:
        return _fail(f"parse error: {NOT_MATH}")
    receive, send = _CTX.Pipe(duplex=False)
    child = _CTX.Process(target=_child_main, args=(send, expression, expected), daemon=True)
    try:
        try:
            child.start()
        finally:
            send.close()
        if not receive.poll(timeout):
            return _fail("timeout")
        verified, computed, error = receive.recv()
        return VerificationResult(verified=verified, computed=computed, error=error)
    except (EOFError, OSError):
        return _fail("verifier worker failed")
    finally:
        if child.pid is not None:  # the child started
            if child.is_alive():
                child.kill()
            child.join()
        receive.close()


async def verify_expression(
    expression: str, expected: str, timeout: float = DEFAULT_TIMEOUT
) -> VerificationResult:
    """Async entry point: `verify_expression_sync` on a small bounded thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _pool, verify_expression_sync, expression, expected, timeout
    )
