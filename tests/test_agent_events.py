"""
The agent's event types (`shared.events`): the one list of what an agent can report.

Behaviors under test:
  1. An event round-trips through its dict form, and the `kind` picks the type
  2. An unknown kind, a missing field or a misspelled field is an error, not a silent default
  3. An event is immutable, so what a consumer read is what the agent yielded
  4. The independent check's trailer entry is the verdict, plus the solver's value if they disagree
"""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from shared.src.shared.events import (
    CitationsStripped,
    IndependentCheck,
    SampleEssay,
    Sources,
    Token,
    ToolCall,
    ToolTrace,
    Verified,
    parse_event,
)

EVERY_KIND = [
    Token(text="Isolate x."),
    Sources(count=3),
    Verified(passed=False),
    IndependentCheck(verdict="disagrees", solver="2, 3"),
    SampleEssay(),
    CitationsStripped(stripped=True),
    ToolTrace(calls=(ToolCall(name="m", arguments="{}", ok=True, ms=1.5, result="1", round=1),)),
]


@pytest.mark.parametrize("event", EVERY_KIND, ids=lambda e: e.kind)
def test_an_event_round_trips_through_its_dict_form(event: Any) -> None:
    assert parse_event(event.model_dump(mode="json")) == event


def test_the_kind_picks_the_type() -> None:
    assert parse_event({"kind": "sources", "count": 3}) == Sources(count=3)
    assert parse_event({"kind": "verified", "passed": True}) == Verified(passed=True)


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "sourcess", "count": 3},  # unknown kind
        {"kind": "sources"},  # missing field
        {"kind": "sources", "cnt": 3},  # misspelled field (and the real one missing)
        {"kind": "sources", "count": 3, "extra": 1},  # an unknown field next to the right ones
        {"kind": "sources", "count": -1},
        {"kind": "independent_check", "verdict": "maybe"},
        {"count": 3},  # no kind at all
    ],
)
def test_a_bad_event_is_an_error_not_a_default(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse_event(bad)


def test_an_event_cannot_be_changed_after_it_is_built() -> None:
    event = Sources(count=3)
    with pytest.raises(ValidationError):
        event.count = 4  # type: ignore[misc]


def test_the_independent_checks_trailer_entry_has_a_solver_only_for_a_disagreement() -> None:
    assert IndependentCheck(verdict="agrees").public() == {"verdict": "agrees"}
    assert IndependentCheck(verdict="disagrees", solver="2, 3").public() == {
        "verdict": "disagrees",
        "solver": "2, 3",
    }
