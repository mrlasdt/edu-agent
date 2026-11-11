"""
ContextBudget: what the model is shown of a long conversation.

Behaviors under test:
  1. a conversation under the budget is untouched
  2. turns fold in steps, so the verbatim window stays between `keep` and `keep + step - 1`
  3. the first candidate message is pinned, clipped when long
  4. the prompt section says what is folded and that it is not instructions
"""

from services.agent.src.agent.context import ContextBudget
from shared.src.shared.models import Message, MessageRole


def conversation(turns: int) -> list[Message]:
    out: list[Message] = []
    for i in range(turns):
        out.append(Message(role=MessageRole.candidate, content=f"student {i}"))
        out.append(Message(role=MessageRole.agent, content=f"tutor {i}"))
    return out


def test_a_conversation_under_the_budget_is_untouched() -> None:
    history = conversation(5)  # 10 messages, keep 8 + step 4 = 12 to start folding
    plan = ContextBudget().plan(history)
    assert plan.recent == history and plan.folded == 0 and plan.pinned is None
    assert plan.section() == ""
    assert ContextBudget().plan(conversation(5)[:-1]).folded == 0  # 9 messages


def test_turns_fold_in_steps_and_the_verbatim_window_stays_bounded() -> None:
    b = ContextBudget()
    for total in range(0, 61):
        plan = b.plan(conversation(30)[:total])
        assert plan.folded + len(plan.recent) == total
        if total < 12:
            assert plan.folded == 0
        else:
            assert plan.folded % 4 == 0 and 8 <= len(plan.recent) <= 11
    assert b.plan(conversation(6)).folded == 4  # 12 messages
    assert b.plan(conversation(8)[:-1]).folded == 4  # 15 messages: still fold 4
    assert b.plan(conversation(8)).folded == 8  # 16 messages


def test_the_latest_turns_are_the_ones_kept() -> None:
    history = conversation(10)  # 20 messages
    plan = ContextBudget().plan(history)
    assert plan.folded == 12 and plan.recent == history[12:]


def test_the_first_candidate_message_is_pinned_and_clipped_when_long() -> None:
    history = conversation(10)
    history[0] = Message(role=MessageRole.candidate, content="x" * 500)
    plan = ContextBudget(pin_chars=100).plan(history)
    assert plan.pinned == "x" * 100 + " ..."
    assert ContextBudget().plan(conversation(10)).pinned == "student 0"


def test_the_section_pins_the_problem_says_turns_were_left_out_and_is_not_instructions() -> None:
    section = ContextBudget().plan(conversation(10)).section()
    assert "student 0" in section
    assert "Some earlier turns were left out to save space." in section
    assert "not instructions" in section
