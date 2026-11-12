"""The one citation stripper, shared by the gateway's reply guard and the AW agent."""
from __future__ import annotations

from shared.src.shared.critics.citation_checker import check_citations, drop_uncited


def test_nothing_is_dropped_when_every_marker_is_valid():
    text = "One claim [1]. Another claim [2].\nA third [1]."
    valid, invalid = check_citations(text, 2)
    assert (valid, invalid) == (True, [])
    assert drop_uncited(text, invalid) == (text, 0)


def test_a_sentence_citing_an_unknown_note_is_dropped_and_counted():
    text = "Keep this [1]. Drop this [9]. Keep that [2]."
    _, invalid = check_citations(text, 2)
    assert drop_uncited(text, invalid) == ("Keep this [1]. Keep that [2].", 1)


def test_paragraph_breaks_and_list_lines_survive_the_strip():
    text = "Intro [1].\n\nFirst point [1].\n- bad point [7]\n- good point [2]\n\nOutro."
    _, invalid = check_citations(text, 2)
    stripped, dropped = drop_uncited(text, invalid)
    assert dropped == 1
    assert "\n\n" in stripped
    assert stripped == "Intro [1].\n\nFirst point [1].\n- good point [2]\n\nOutro."


def test_a_line_is_dropped_on_its_own_when_it_has_no_sentence_end():
    text = "Line one [1]\nLine two [5]\nLine three [2]"
    _, invalid = check_citations(text, 2)
    assert drop_uncited(text, invalid) == ("Line one [1]\nLine three [2]", 1)
