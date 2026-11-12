"""
The (test_type, section) pairs the product tutors in chat.

One list, imported by both services: the gateway's Orchestrator asks it "is this pair
supported?" before calling the agent, and the agent's `build_registry` must bind exactly these
pairs to agent classes. Neither service imports the other; they share this file instead.

Adding a test type is one entry here plus one `register(...)` line in the agent's `bootstrap.py`
(a test fails if the two drift apart).
"""
from __future__ import annotations

from collections.abc import Collection

SUPPORTED_PAIRS: tuple[tuple[str, str], ...] = (
    ("gre", "aw"),
    ("gre", "quant"),
    ("ielts", "writing"),
)


def is_supported(
    test_type: str, section: str, pairs: Collection[tuple[str, str]] = SUPPORTED_PAIRS
) -> bool:
    return (test_type, section) in pairs


def describe_supported(pairs: Collection[tuple[str, str]] = SUPPORTED_PAIRS) -> str:
    """The pairs as 'gre/aw, gre/quant' for a message to the Candidate ('none yet' when empty)."""
    return ", ".join(f"{t}/{s}" for t, s in sorted(pairs)) or "none yet"
