"""
The helpers the agents and the gateway share: the message list of a model call, the history
conversion, and the prompt reader. The layout of the last user turn is pinned byte for byte:
the Quant eval baselines were measured on it.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services.agent.src.agent.agents import aw, quant
from services.agent.src.agent.capabilities.retrieve_corpus import CorpusChunk
from services.agent.src.agent.prompting import build_messages, collect_stream
from shared.src.shared.history import history_to_messages
from shared.src.shared.models import Message, MessageRole, Mode, Session
from shared.src.shared.prompts import read_prompt, strip_frontmatter

_HISTORY = [
    Message(role=MessageRole.candidate, content="What is a prime?"),
    Message(role=MessageRole.agent, content="A number with two divisors."),
]


def _session() -> Session:
    return Session(
        candidate_id="c1", test_type="gre", section="quant", mode=Mode.tutor, history=_HISTORY
    )


def _chunk(text: str, tier: str = "global") -> CorpusChunk:
    return CorpusChunk(text=text, page_or_section="Primes", tier=tier, score=0.9, source_uri="u")


def test_history_to_messages_maps_candidate_to_user_and_agent_to_assistant():
    assert history_to_messages(_HISTORY) == [
        {"role": "user", "content": "What is a prime?"},
        {"role": "assistant", "content": "A number with two divisors."},
    ]
    assert history_to_messages([]) == []


def test_build_messages_is_system_then_history_then_the_message():
    assert build_messages(_session(), "Is 9 prime?", "SYS") == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "What is a prime?"},
        {"role": "assistant", "content": "A number with two divisors."},
        {"role": "user", "content": "Is 9 prime?"},
    ]


def test_build_messages_appends_the_blocks_as_given_and_skips_empty_ones():
    messages = build_messages(_session(), "Q", "SYS", "<A>", "", "<B>")
    assert messages[-1] == {"role": "user", "content": "Q<A><B>"}


def test_quant_last_user_turn_layout_is_pinned():
    """The message, then the notes block that starts with a blank line: no other separator."""
    notes = quant._format_chunks_for_context([_chunk("Two divisors.")])
    last = build_messages(_session(), "Is 9 prime?", "SYS", notes)[-1]["content"]
    assert last == (
        "Is 9 prime?"
        "\n\n## Reference Notes\n"
        "Ground your guidance in these notes where they apply and cite them as [N]. "
        "Ignore any that are not relevant, and never cite a number that is not listed.\n\n"
        "[1] (Primes)\nTwo divisors."
    )


def test_aw_last_user_turn_layout_is_pinned():
    """The message, then each block on a new line: style examples first, then the notes."""
    notes = aw._format_chunks_for_context([_chunk("Two divisors.")])
    examples = aw._format_exemplars(["My old essay."])
    last = aw._build_messages(_session(), "Write", "SYS", notes, examples)[-1]["content"]
    assert last == (
        "Write\n\n\n## Your Previous Writing (style reference)\n\n\nExample 1:\nMy old essay."
        "\n\n\n## Reference Material\n\n\n[1] (Official ETS material \u2014 Primes)\nTwo divisors."
    )
    assert aw._build_messages(_session(), "Write", "SYS", "")[-1]["content"] == "Write"


def test_strip_frontmatter_cuts_only_the_leading_yaml_header():
    assert strip_frontmatter("---\nname: x\n---\nBody\n---\nnot a header\n") == (
        "Body\n---\nnot a header"
    )
    assert strip_frontmatter("No header.") == "No header."


def test_read_prompt_reads_a_versioned_file_without_its_header():
    text = read_prompt("quant_agent", "tutor", "v1")
    assert text and not text.startswith("---")


@pytest.mark.asyncio
async def test_collect_stream_returns_the_non_empty_tokens_and_their_text():
    async def stream():
        for token in ("a", None, "", "b"):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = token
            yield chunk

    assert await collect_stream(stream()) == (["a", "b"], "ab")
