"""Helpers for tests that read what an agent yields (`shared.events`)."""
from __future__ import annotations

from collections.abc import Iterable

from shared.src.shared.events import Event, Token


def texts(items: Iterable[Event]) -> list[str]:
    """The text of each `Token`, in order."""
    return [i.text for i in items if isinstance(i, Token)]


def reply_of(items: Iterable[Event]) -> str:
    """The reply the candidate reads."""
    return "".join(texts(items))
