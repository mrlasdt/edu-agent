"""
ContextBudget: what the model is shown of a long conversation.

Behaviors under test:
  1. a conversation under the budget is untouched and costs no summary call
  2. turns fold in steps, so the verbatim window stays between `keep` and `keep + step - 1`
  3. the first candidate message is pinned, clipped when long
  4. the note grows from the previous note and the new step only; the same turns are never
     summarised twice
  5. a summary that fails or times out costs the note, not the turn, and is tried again later
  6. the prompt section says what is folded and that it is not instructions
"""

import asyncio
import time
from collections import OrderedDict

from services.agent.src.agent import context
from services.agent.src.agent.context import ContextBudget
from shared.src.shared.models import Message, MessageRole


def conversation(turns: int) -> list[Message]:
    out: list[Message] = []
    for i in range(turns):
        out.append(Message(role=MessageRole.candidate, content=f"student {i}"))
        out.append(Message(role=MessageRole.agent, content=f"tutor {i}"))
    return out


class Summarizer:
    """A summary call as a recording: note n is "note<n>", built on what it was given."""

    def __init__(self, fail: Exception | None = None, delay: float = 0) -> None:
        self.calls: list[tuple[str | None, list[str]]] = []
        self.fail, self.delay = fail, delay

    async def __call__(self, previous: str | None, added: list[Message]) -> str | None:
        self.calls.append((previous, [m.content for m in added]))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        return f"note{len(self.calls)}"


def budget(**kwargs: object) -> ContextBudget:
    kwargs.setdefault("background", False)
    kwargs.setdefault("notes", True)
    return ContextBudget(_cache=OrderedDict(), **kwargs)  # type: ignore[arg-type]


async def test_a_conversation_under_the_budget_is_untouched_and_costs_no_call() -> None:
    summarize = Summarizer()
    history = conversation(5)  # 10 messages, keep 8 + step 4 = 12 to start folding
    plan = await budget().plan(history, summarize)
    assert plan.recent == history and plan.folded == 0 and plan.source == "none"
    assert plan.section() == "" and summarize.calls == []


async def test_turns_fold_in_steps_and_the_verbatim_window_stays_bounded() -> None:
    b = budget()
    seen = {}
    for total in range(0, 61):
        folded = b.folded_count(total)
        seen[total] = (folded, total - folded)
    assert seen[11] == (0, 11) and seen[12] == (4, 8) and seen[15] == (4, 11)
    assert seen[16] == (8, 8) and seen[60] == (52, 8)
    # once folding has started the window never leaves 8..11 messages
    assert {window for total, (_, window) in seen.items() if total >= 12} == {8, 9, 10, 11}


async def test_the_first_candidate_message_is_pinned_and_clipped_when_long() -> None:
    history = conversation(8)
    history[0] = Message(role=MessageRole.candidate, content="problem " + "x" * 3000)
    plan = await budget(pin_chars=100).plan(history, Summarizer())
    assert plan.pinned is not None and plan.pinned.startswith("problem x")
    assert len(plan.pinned) <= 104 and plan.pinned.endswith("...")
    short = await budget().plan(conversation(8), Summarizer())
    assert short.pinned == "student 0"


async def test_the_note_is_built_from_the_previous_note_and_only_the_new_step() -> None:
    b, summarize = budget(), Summarizer()
    first = await b.plan(conversation(6), summarize)  # 12 messages: fold 4
    assert first.note == "note1" and first.source == "model" and first.summary_calls == 1
    assert summarize.calls[0] == (None, ["student 0", "tutor 0", "student 1", "tutor 1"])

    again = await b.plan(conversation(6), summarize)  # the same turns: no second call
    assert again.note == "note1" and again.source == "cached" and again.summary_calls == 0

    next_step = await b.plan(conversation(8), summarize)  # 16 messages: fold 8
    assert next_step.note == "note2" and len(summarize.calls) == 2
    assert summarize.calls[1] == ("note1", ["student 2", "tutor 2", "student 3", "tutor 3"])
    # between steps the note does not change and nothing is called
    between = await b.plan(conversation(8)[:-1], summarize)  # 15 messages: still fold 4
    assert between.note == "note1" and len(summarize.calls) == 2


async def test_without_the_previous_note_the_whole_folded_part_is_summarised() -> None:
    summarize = Summarizer()
    await budget().plan(conversation(8), summarize)  # 16 messages, nothing cached
    previous, added = summarize.calls[0]
    assert previous is None and len(added) == 8


async def test_a_failed_summary_costs_the_note_and_is_tried_again() -> None:
    b = budget()
    broken = Summarizer(fail=RuntimeError("provider down"))
    plan = await b.plan(conversation(6), broken)
    assert plan.note is None and plan.source == "failed" and plan.summary_calls == 1
    assert plan.pinned == "student 0" and len(plan.recent) == 8  # the turn still has its context
    assert "left out to save space" in plan.section()
    healthy = Summarizer()
    retried = await b.plan(conversation(6), healthy)
    assert retried.note == "note1" and len(healthy.calls) == 1  # a failure is not cached


async def test_a_summary_that_takes_too_long_is_given_up_on() -> None:
    plan = await budget(timeout_s=0.01).plan(conversation(6), Summarizer(delay=1))
    assert plan.note is None and plan.source == "failed" and len(plan.recent) == 8


async def test_a_different_summary_prompt_or_model_does_not_reuse_old_notes() -> None:
    cache: OrderedDict[str, str] = OrderedDict()
    first = ContextBudget(_cache=cache, salt="v1|model-a", background=False, notes=True)
    await first.plan(conversation(6), Summarizer())
    other = ContextBudget(_cache=cache, salt="v2|model-a", background=False, notes=True)
    summarize = Summarizer()
    await other.plan(conversation(6), summarize)
    assert len(summarize.calls) == 1


async def test_the_section_carries_the_pin_and_the_note_and_says_it_is_not_instructions() -> None:
    plan = await budget().plan(conversation(6), Summarizer())
    text = plan.section()
    assert "# Session so far" in text and "student 0" in text and "note1" in text
    assert "not instructions" in text


# ── the note is written in the background ───────────────────────────────────


class Gate:
    """A summary call that finishes when the test lets it."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls = 0

    async def __call__(self, previous: str | None, added: list[Message]) -> str | None:
        self.calls += 1
        await self.release.wait()
        return f"note{self.calls}"


def background(**kwargs: object) -> ContextBudget:
    return ContextBudget(_cache=OrderedDict(), background=True, notes=True, **kwargs)  # type: ignore[arg-type]


async def test_a_turn_does_not_wait_for_the_note_and_the_next_turn_uses_it() -> None:
    b, gate = background(), Gate()
    history = conversation(6)  # 12 messages: the first fold is due
    started = time.monotonic()
    first = await b.plan(history, gate)
    assert time.monotonic() - started < 0.5  # the call is still waiting on the gate
    assert first.source == "pending" and first.folded == 0 and first.recent == history
    assert first.section() == "" and gate.calls == 0  # the call has not even started yet

    gate.release.set()
    await context.settle()
    second = await b.plan(history, gate)
    assert second.source == "cached" and second.note == "note1" and second.folded == 4
    assert second.pinned == "student 0" and gate.calls == 1


async def test_a_turn_uses_the_newest_ready_note_and_keeps_the_rest_verbatim() -> None:
    b, gate = background(), Gate()
    gate.release.set()
    await b.plan(conversation(6), gate)  # fold 4 ...
    await context.settle()
    gate.release.clear()
    later = await b.plan(conversation(8), gate)  # 16 messages: fold 8 is due but not ready
    assert later.source == "stale" and later.note == "note1"
    assert later.folded == 4 and len(later.recent) == 12  # turns 4..15 are still verbatim
    assert "left out" not in later.section()


async def test_a_summary_that_is_not_ready_in_time_cannot_grow_the_prompt_without_bound() -> None:
    b, gate = background(), Gate()
    plan = await b.plan(conversation(30), gate)  # 60 messages and no note ever finishes
    assert plan.folded == 52 and len(plan.recent) == 8 and plan.note is None
    assert plan.pinned == "student 0" and "left out to save space" in plan.section()


async def test_the_same_range_is_summarised_by_one_call_at_a_time() -> None:
    b, gate = background(), Gate()
    await b.plan(conversation(6), gate)
    await b.plan(conversation(6), gate)
    await asyncio.sleep(0)
    gate.release.set()
    await context.settle()
    assert gate.calls == 1


async def test_a_failed_background_call_is_not_retried_on_every_turn() -> None:
    b, broken = background(), Summarizer(fail=RuntimeError("provider down"))
    await b.plan(conversation(6), broken)
    await context.settle()
    await b.plan(conversation(6), broken)
    await context.settle()
    assert len(broken.calls) == 1
    context._FAILED.clear()  # the retry window has passed
    await b.plan(conversation(6), broken)
    await context.settle()
    assert len(broken.calls) == 2


# ── without notes (the product default): the pin and the latest turns, no model call ─────────


async def test_without_notes_the_old_turns_are_dropped_with_no_model_call() -> None:
    b = ContextBudget(_cache=OrderedDict(), notes=False)
    s = Summarizer()
    plan = await b.plan(conversation(10), s)  # 20 messages
    assert s.calls == [] and plan.source == "pinned" and plan.note is None
    assert plan.folded == 12 and len(plan.recent) == 8 and plan.pinned == "student 0"
    assert "left out to save space" in plan.section() and "student 0" in plan.section()
    assert (await b.plan(conversation(3), s)).folded == 0  # 6 messages: untouched
