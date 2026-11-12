"""
The agent service's one long-lived HTTP client, opened when the app starts and closed when it
stops, so calls to the corpus and math-verifier services reuse connections instead of setting up
a new one (TCP and TLS) per call.

`current()` is None outside a running app (unit tests, evals, scripts): a caller then falls back
to a client that lives for one call, which is how it behaved before the pool existed.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

_pool: httpx.AsyncClient | None = None


def current() -> httpx.AsyncClient | None:
    """The open pool, or None when none is open."""
    return _pool


@asynccontextmanager
async def open_pool() -> AsyncIterator[httpx.AsyncClient]:
    """Open the pool for the life of the `with` block."""
    global _pool
    async with httpx.AsyncClient() as client:
        _pool = client
        try:
            yield client
        finally:
            _pool = None
