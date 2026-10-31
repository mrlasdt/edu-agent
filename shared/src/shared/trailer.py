"""
A one-line metadata trailer at the end of a plain-text answer stream.

The agent service and the gateway stream the answer as `text/plain`, and a reply's metadata (so
far, whether the math verifier accepted the final answer) is only known once it is written, too
late for a response header. It travels as the last bytes of the stream instead: the record
separator character, then a JSON object. Each hop peels it off before the text reaches a
guardrail and the gateway re-emits it, validated, for the web app.
"""
from __future__ import annotations

import json
from typing import Any

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
