"""Per-Candidate rate limiting and daily quota enforcement."""
from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

# Candidates tracked at once. The id comes from the client, so without a bound every new id
# would add an entry that is never freed.
DEFAULT_MAX_CANDIDATES = 10_000


@dataclass
class _CandidateState:
    # timestamps of turns in the last 60 s (sliding window)
    window: deque[float] = field(default_factory=deque)
    # (UTC day, turns counted that day)
    day: tuple[str, int] = ("", 0)


class RateLimiter:
    """
    In-memory rate limiter for Phase 1.
    Tracks per-minute turn counts (sliding window) and per-day turn counts.
    Memory is bounded: at most `max_candidates` ids are kept, and the least recently seen
    id is evicted first (its counters start over if it returns). The id is client-chosen
    until session tokens exist (plan chapter 18), so this bounds memory, not abuse.
    Phase 2: replace with Redis-backed implementation.
    """

    def __init__(
        self,
        max_per_minute: int = 60,
        max_per_day: int = 200,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        self._max_per_minute = max_per_minute
        self._max_per_day = max_per_day
        self._max_candidates = max_candidates
        # candidate_id → state, least recently used first
        self._states: OrderedDict[str, _CandidateState] = OrderedDict()

    def reset(self) -> None:
        """Forget every candidate (tests, and a manual flush)."""
        self._states.clear()

    def _state(self, candidate_id: str) -> _CandidateState:
        state = self._states.get(candidate_id)
        if state is None:
            state = self._states[candidate_id] = _CandidateState()
            while len(self._states) > self._max_candidates:
                self._states.popitem(last=False)
        else:
            self._states.move_to_end(candidate_id)
        return state

    def check(self, candidate_id: str) -> tuple[bool, str]:
        """
        Record a turn attempt and check limits.
        Returns (allowed, message) — message is empty when allowed=True.
        """
        now = time.time()
        day_str = time.strftime("%Y-%m-%d", time.gmtime(now))
        state = self._state(candidate_id)

        # Per-minute sliding window
        window = state.window
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self._max_per_minute:
            return False, f"Too many requests — rate limit exceeded ({self._max_per_minute}/min)."

        # Per-day counter
        day_count = state.day[1] if state.day[0] == day_str else 0

        if day_count >= self._max_per_day:
            return False, (
                f"Daily quota of {self._max_per_day} turns reached. "
                "Your quota resets at midnight UTC."
            )

        # Record the turn
        window.append(now)
        state.day = (day_str, day_count + 1)
        return True, ""
