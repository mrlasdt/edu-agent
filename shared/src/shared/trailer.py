"""
A one-line metadata trailer at the end of a plain-text answer stream.

The agent service and the gateway stream the answer as `text/plain`, and a reply's metadata
(whether the math verifier accepted the final answer, what the independent check of it said, and
whether it is a sample essay the web app must label) is only known once it is written, too late
for a response header. It travels as the last bytes of the stream instead: the record separator
character, then a JSON object. Each hop peels it off before the text reaches a guardrail and the
gateway re-emits it, validated, for the web app.
"""
from __future__ import annotations

import json
from typing import Any

from shared.src.shared.events import VERDICTS

MARK = "\x1e"  # ASCII record separator: never part of an answer


def encode(meta: dict[str, Any]) -> str:
    return MARK + json.dumps(meta, separators=(",", ":"))


def strip_mark(text: str) -> str:
    """Model output with any record separator removed, so it cannot forge a trailer."""
    return text.replace(MARK, "")


class TrailerSplitter:
    """Feed a stream's text chunk by chunk; `feed` returns the part before the trailer."""

    def __init__(self) -> None:
        self._tail: list[str] = []
        self._seen = False

    def feed(self, text: str) -> str:
        if self._seen:
            self._tail.append(text)
            return ""
        head, mark, rest = text.partition(MARK)
        if mark:
            self._seen = True
            self._tail.append(rest)
        return head

    @property
    def meta(self) -> dict[str, Any]:
        """The trailer's JSON object, or {} when there was none or it is not one."""
        try:
            value = json.loads("".join(self._tail))
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}


_SOLVER_MAX_CHARS = 80


def independent_check_trailer(value: object) -> dict[str, str]:
    """
    The independent answer check's entry for the trailer, or {} when `value` is not one.

    The gateway re-emits it for the web app, so it is checked on the way: one of three verdicts,
    and a `solver` text (the value the solver computed) only for "disagrees", short and printable.
    """
    if not isinstance(value, dict) or value.get("verdict") not in VERDICTS:
        return {}
    out = {"verdict": str(value["verdict"])}
    solver = value.get("solver")
    if out["verdict"] == "disagrees" and isinstance(solver, str):
        text = "".join(ch for ch in solver if ch.isprintable())[:_SOLVER_MAX_CHARS].strip()
        if text:
            out["solver"] = text
    return out
