from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    section_title: str | None
    chunk_index: int
    char_start: int
    char_end: int


def _approx_tokens(text: str) -> int:
    """Word-count approximation: words × 1.3 ≈ tokens."""
    return int(len(text.split()) * 1.3)


_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+(.+)|([A-Z][A-Z\s]{3,40}[A-Z]))\s*$", re.MULTILINE
)


def _extract_heading(line: str) -> str | None:
    """Return the heading text if this line is a Markdown heading or ALL-CAPS title."""
    m = _HEADING_RE.match(line.strip())
    if not m:
        return None
    return (m.group(1) or m.group(2)).strip()


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Split text into (section_title, content) pairs.
    Headings become section titles; content between headings is grouped under them.
    """
    lines = text.splitlines()
    sections: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in lines:
        heading = _extract_heading(line)
        if heading:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = heading
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return [(t, c) for t, c in sections if c]


def _split_section_into_chunks(
    section_text: str,
    section_title: str | None,
    target_tokens: int,
    overlap_ratio: float,
    start_index: int,
    char_offset: int,
) -> list[Chunk]:
    """Split a single section into overlapping token-sized chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", section_text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    current_paras: list[str] = []
    current_tokens = 0
    overlap_tokens = int(target_tokens * overlap_ratio)
    chunk_idx = start_index

    # Track char positions within the section
    para_char_starts: list[int] = []
    pos = 0
    for para in paragraphs:
        idx = section_text.find(para, pos)
        para_char_starts.append(char_offset + idx)
        pos = idx + len(para)

    para_ptr = 0

    while para_ptr < len(paragraphs):
        current_paras = []
        current_tokens = 0
        chunk_start_para = para_ptr

        while para_ptr < len(paragraphs):
            para = paragraphs[para_ptr]
            tokens = _approx_tokens(para)
            if current_tokens + tokens > target_tokens and current_paras:
                break
            current_paras.append(para)
            current_tokens += tokens
            para_ptr += 1

        if not current_paras:
            # Single paragraph larger than target; include it as-is
            current_paras = [paragraphs[para_ptr]]
            para_ptr += 1

        text = "\n\n".join(current_paras)
        char_start = para_char_starts[chunk_start_para]
        last_para_idx = chunk_start_para + len(current_paras) - 1
        last_para_text = paragraphs[last_para_idx]
        char_end = para_char_starts[last_para_idx] + len(last_para_text)

        chunks.append(
            Chunk(
                text=text,
                section_title=section_title,
                chunk_index=chunk_idx,
                char_start=char_start,
                char_end=char_end,
            )
        )
        chunk_idx += 1

        # Apply overlap: step back by overlap_ratio paragraphs
        if para_ptr < len(paragraphs):
            overlap_para_count = max(1, int(len(current_paras) * overlap_ratio))
            para_ptr = max(chunk_start_para + 1, para_ptr - overlap_para_count)

    return chunks


def chunk_text(
    text: str,
    target_tokens: int = 400,
    overlap_ratio: float = 0.1,
) -> list[Chunk]:
    """
    Split text into structured, overlapping chunks.

    Headings are detected and carried as section_title metadata.
    Child chunks are ~target_tokens tokens with ~overlap_ratio overlap.
    Returns an empty list for empty or whitespace-only input.
    """
    if not text or not text.strip():
        return []

    sections = _split_into_sections(text)
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for title, content in sections:
        # Find the char offset of this section's content in the original text
        offset = text.find(content[:50]) if content else 0
        if offset == -1:
            offset = 0
        new_chunks = _split_section_into_chunks(
            content, title, target_tokens, overlap_ratio, chunk_index, offset
        )
        all_chunks.extend(new_chunks)
        chunk_index += len(new_chunks)

    return all_chunks
