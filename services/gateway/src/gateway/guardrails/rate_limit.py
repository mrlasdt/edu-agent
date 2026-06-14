"""Per-Candidate rate limiting and daily quota enforcement."""
from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    """
    In-memory rate limiter for Phase 1.
    Tracks per-minute turn counts (sliding window) and per-day turn counts.
    Phase 2: replace with Redis-backed implementation.
    """

    def __init__(
        self,
        max_per_minute: int = 60,
        max_per_day: int = 200,
    ) -> None:
        self._max_per_minute = max_per_minute
        self._max_per_day = max_per_day
        # candidate_id → deque of timestamps (floats, last 60s)
        self._minute_window: dict[str, deque] = defaultdict(deque)
        # candidate_id → (day_str, count)
        self._day_counts: dict[str, tuple[str, int]] = {}

    def check(self, candidate_id: str) -> tuple[bool, str]:
        """
        Record a turn attempt and check limits.
        Returns (allowed, message) — message is empty when allowed=True.
        """
        now = time.time()
        day_str = time.strftime("%Y-%m-%d", time.gmtime(now))

        # Per-minute sliding window
        window = self._minute_window[candidate_id]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self._max_per_minute:
            return False, f"Too many requests — rate limit exceeded ({self._max_per_minute}/min)."

        # Per-day counter
        stored = self._day_counts.get(candidate_id)
        if stored is None or stored[0] != day_str:
            day_count = 0
        else:
            day_count = stored[1]

        if day_count >= self._max_per_day:
            return False, (
                f"Daily quota of {self._max_per_day} turns reached. "
                "Your quota resets at midnight UTC."
            )

        # Record the turn
        window.append(now)
        self._day_counts[candidate_id] = (day_str, day_count + 1)
        return True, ""
