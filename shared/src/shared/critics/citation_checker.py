from __future__ import annotations

import re


def check_citations(text: str, num_chunks: int) -> tuple[bool, list[str]]:
    """
    Verify that every [N] citation marker in text refers to a valid chunk index.

    Args:
        text: The generated text to check.
        num_chunks: Number of available chunks (valid markers: [1] … [num_chunks]).

    Returns:
        (valid, invalid_citations) where valid=True means all markers are resolvable.
        invalid_citations lists the marker strings that could not be resolved.
    """
    markers = re.findall(r"\[(\d+)\]", text)
    if not markers:
        return True, []

    invalid = []
    for m in markers:
        n = int(m)
        if n < 1 or n > num_chunks:
            invalid.append(f"[{m}]")

    return len(invalid) == 0, invalid


# A sentence ends at . ! or ? followed by blanks, and a line ends at a newline. The separator is
# kept with the text it follows so the reply keeps its paragraphs and lists.
_SENTENCE_SPLIT = re.compile(r"((?<=[.!?])[ \t]+|\s*\n\s*)")


def drop_uncited(text: str, invalid_citations: list[str]) -> tuple[str, int]:
    """
    Remove the sentences or lines that cite an invalid marker, keeping every separator.
    Returns (the remaining text, how many sentences or lines were dropped). Best-effort: the split
    is rough, so a sentence that holds an abbreviation may be cut in two.
    """
    if not invalid_citations:
        return text, 0
    parts = _SENTENCE_SPLIT.split(text)
    out: list[str] = []
    dropped = 0
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        if any(marker in sentence for marker in invalid_citations):
            dropped += 1
            if out and "\n" in sep and "\n" not in out[-1]:
                out[-1] = sep
            continue
        out += [sentence, sep]
    return "".join(out), dropped
