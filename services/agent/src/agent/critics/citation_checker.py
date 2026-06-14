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


def strip_uncited_claims(text: str, invalid_citations: list[str]) -> str:
    """
    Remove sentences containing invalid citation markers.
    Best-effort: splits on sentence boundaries and drops sentences with bad markers.
    """
    if not invalid_citations:
        return text

    invalid_set = set(invalid_citations)
    # Split into sentences (rough approximation)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean = [
        s for s in sentences
        if not any(marker in s for marker in invalid_set)
    ]
    return " ".join(clean)
