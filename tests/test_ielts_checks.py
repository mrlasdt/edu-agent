"""Deterministic essay checks and band arithmetic for the IELTS Writing mock."""
import pytest

from evals.exam_set import ielts_writing_adversarial as adversarial
from services.agent.src.agent.ielts.checks import (
    check_essay,
    paragraph_spans,
    prompt_copy_share,
    prompt_overlap,
)
from services.agent.src.agent.ielts.scoring import (
    GradeParseError,
    median_band,
    parse_sample,
)

PROMPT = adversarial.PROMPT
PLAIN = adversarial._PLAIN
CASES = {i.extras["case"]: i for i in adversarial.generate_all()}


def long_essay(words: int) -> str:
    base = PLAIN.split()
    return " ".join((base * (words // len(base) + 1))[:words])


def test_median_band_and_parse_sample():
    assert median_band([5.0, 6.5, 6.0]) == 6.0
    assert median_band([6.0, 6.5]) == 6.5  # a 6.25 midpoint rounds up
    sample = parse_sample('{"TR": 6, "CC": 6.5, "LR": 5.5, "GRA": 6, "off_topic": true}')
    assert sample.bands["CC"] == 6.5 and sample.off_topic and not sample.addressed_to_marker
    with pytest.raises(GradeParseError):
        parse_sample('{"TR": 6}')


def test_a_normal_short_essay_is_flagged_for_length_not_blocked():
    checks = check_essay(PROMPT, PLAIN)
    assert checks.blocked is None and checks.word_count < 150
    assert checks.has("under_length") and checks.caps["TR"] == 4.0
    assert checks.prompt_overlap >= 0.5


@pytest.mark.parametrize(
    ("words", "cap"), [(160, 5.5), (240, None), (260, None)],
)
def test_length_caps_task_response(words: int, cap: float | None):
    checks = check_essay(PROMPT, long_essay(words))
    assert checks.caps.get("TR") == cap


def test_paragraph_spans_split_on_blank_lines_then_single_lines():
    text = "One idea.\n\nSecond idea here.\n\n  Third.  "
    assert [text[a:b] for a, b in paragraph_spans(text)] == [
        "One idea.", "Second idea here.", "Third.",
    ]
    assert len(paragraph_spans("a\nb\nc")) == 3
    assert paragraph_spans("   ") == []


def test_prompt_overlap_and_copy_share():
    assert prompt_overlap(PROMPT, PLAIN) > 0.4
    assert prompt_overlap(PROMPT, CASES["off_topic_fluent"].response or "") < 0.2
    assert prompt_copy_share(PROMPT, PROMPT * 2) > 0.9
    assert prompt_copy_share(PROMPT, PLAIN) == 0.0


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_adversarial_case_raises_its_expected_flags(name: str):
    item = CASES[name]
    assert item.response is not None
    checks = check_essay(item.prompt, item.response)
    for flag in item.extras["expect_flags"]:
        assert checks.has(flag), f"{name}: {flag} not raised, got {[f.code for f in checks.flags]}"


def test_the_plain_essay_alone_raises_no_manipulation_flag():
    codes = {f.code for f in check_essay(PROMPT, PLAIN).flags}
    assert not codes & {"addressed_to_marker", "possibly_off_topic", "copies_prompt"}


def test_a_response_far_under_length_holds_every_criterion_down():
    checks = check_essay(PROMPT, PLAIN)
    assert checks.word_count < 150
    assert checks.caps == {"TR": 4.0, "CC": 5.0, "LR": 5.5, "GRA": 5.5}  # PLAIN is one paragraph


def test_one_paragraph_caps_coherence_but_two_or_three_only_note_it():
    body = long_essay(260)
    one = check_essay(PROMPT, body)
    assert one.has("one_paragraph") and not one.has("few_paragraphs")
    assert one.caps == {"CC": 5.0}
    words = body.split()
    three = check_essay(PROMPT, "\n\n".join(" ".join(words[i::3]) for i in range(3)))
    assert three.has("few_paragraphs") and not three.has("one_paragraph")
    assert "CC" not in three.caps
    four = check_essay(PROMPT, "\n\n".join(" ".join(words[i::4]) for i in range(4)))
    assert not four.has("few_paragraphs") and not four.has("one_paragraph")
