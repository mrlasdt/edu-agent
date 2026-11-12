"""
The gateway's three rate limiters, side by side so their budgets can be compared.

Each is built on first use and then shared by the process; `/chat` and `/feedback` receive theirs
with `Depends`, so a test swaps one in with `app.dependency_overrides`, and a test resets the IELTS
one with `ielts_limiter().reset()`. The app's lifespan builds all three at start, so a bad setting
fails the start, not the first request.
"""
from __future__ import annotations

from functools import lru_cache

from services.gateway.src.gateway.guardrails.rate_limit import RateLimiter


@lru_cache
def chat_limiter() -> RateLimiter:
    """/chat: per candidate id, the limiter's default budget (60 a minute, 200 a day)."""
    return RateLimiter()


@lru_cache
def ielts_limiter() -> RateLimiter:
    """The IELTS mock's attempt routes. One local user, and each attempt is a paid model call."""
    return RateLimiter(max_per_minute=10, max_per_day=60)


@lru_cache
def feedback_limiter() -> RateLimiter:
    """
    /feedback votes, keyed by client address (the request carries no candidate id). Behind the web
    app's nginx every browser shares one address, so this is one global bound, not a per-user limit.
    """
    return RateLimiter(max_per_minute=30, max_per_day=500)
