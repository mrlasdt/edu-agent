"""
Rule graders for tutor replies: free, instant, exact where a rule can be exact.

Each grader answers one yes/no question about a reply and is named after the failure that
made us write it (see `failure_source` on the golden items). A rule grader cannot judge whether
a hint is *good*; it can only catch a known, mechanical failure. That is the trade: no judge
cost or judge disagreement, in exchange for only seeing the failures somebody already wrote
a rule for. `quant_judge` (model-as-judge) covers the rest.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from evals.agent_eval.probes import leaks_by_rule
from services.gateway.src.gateway.guardrails.output import detect_answer_leak

# The v2/v3 prompts say "no LaTeX": the chat window shows backslash markup literally.
_LATEX = re.compile(r"\\[()\[\]]|\\(?:frac|dfrac|sqrt|boxed|times|cdot|left|right|text|pm)\b")
# The tutor once told students to look at "your notes"; they are the app's notes (cite as [N]).
_YOUR_NOTES = re.compile(r"\byour\s+(?:study\s+|own\s+)?notes?\b", re.IGNORECASE)
_NOTE_WORD = re.compile(r"\b(?:reference\s+|study\s+|the\s+)notes?\b", re.IGNORECASE)
_MARKER = re.compile(r"\[(\d+)\]")
# Replies that hand the work back to the student without a next step.
_VAGUE = re.compile(
    r"what\s+method\s+do\s+you\s+want|what\s+(?:do\s+you\s+)?need\s+help\s+with"
    r"|how\s+can\s+i\s+help|what\s+would\s+you\s+like\s+to\s+(?:do|know|work\s+on|try)"
    r"|which\s+(?:method|approach)\s+(?:would|do)\s+you\s+(?:like|want|prefer)"
    r"|what\s+are\s+you\s+(?:stuck|struggling)\s+(?:on|with)\??\s*$",
    re.IGNORECASE,
)
_MIN_REPLY_CHARS = 20


def leaks_answer(reply: str, item: dict[str, Any]) -> bool:
    """
    True when the reply states the final answer. Two detectors, either one counts: the product's
    own regex guardrail (`detect_answer_leak`, catches "Therefore x = 3" and a bare "x = 3"),
    and the exact-value rule from the eval probes (every answer value appears as a token).
    The second over-counts, for example when a step number equals the answer, so a leak rate
    here is an upper bound on what a learner would see.
    """
    return detect_answer_leak(reply) or leaks_by_rule(
        reply, item["ground_truth"], item.get("question_type", "")
    )


def has_latex(reply: str) -> bool:
    return _LATEX.search(reply) is not None


def citations_ok(reply: str, n_notes: int) -> bool:
    """
    Reference notes are cited as [N] with N one of the notes given; the reply never calls them
    "your notes" and never talks about notes without a marker.
    """
    if _YOUR_NOTES.search(reply):
        return False
    markers = [int(m) for m in _MARKER.findall(reply)]
    if any(not 1 <= m <= n_notes for m in markers):
        return False
    return not (_NOTE_WORD.search(reply) and not markers)


def moves_forward(reply: str) -> bool:
    """Not empty and not one of the vague hand-backs ("what method do you want to use?")."""
    text = reply.strip()
    return len(text) >= _MIN_REPLY_CHARS and _VAGUE.search(text) is None


def stays_on_problem(reply: str, anchors: Sequence[str]) -> bool:
    """The reply mentions something specific to the problem (any anchor pattern matches)."""
    return any(re.search(a, reply, re.IGNORECASE) for a in anchors)


UNIVERSAL_TUTOR_CHECKS: dict[str, Callable[[str, dict[str, Any], int], bool]] = {
    "no_leak": lambda reply, item, n_notes: not leaks_answer(reply, item),
    "no_latex": lambda reply, item, n_notes: not has_latex(reply),
    "citations_ok": lambda reply, item, n_notes: citations_ok(reply, n_notes),
    "moves_forward": lambda reply, item, n_notes: moves_forward(reply),
}


def grade_tutor_reply(reply: str, item: dict[str, Any]) -> dict[str, bool]:
    """Every universal tutor check on one reply: name -> passed."""
    n_notes = len(item.get("notes", []))
    return {name: check(reply, item, n_notes) for name, check in UNIVERSAL_TUTOR_CHECKS.items()}
