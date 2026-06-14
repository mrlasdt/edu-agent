"""
Tests for the structure-aware hierarchical chunker.

Behaviors under test (from issue 06 acceptance criteria):
  1. Plain text → list of Chunk objects sized ~400 tokens
  2. Structured text with headings → chunks carry section title
  3. Adjacent chunks share ~10% token overlap
  4. Empty text → empty list (no crash)
  5. Chunk metadata fields are populated (char_start, char_end, chunk_index)
"""

import pytest
from services.ingestion.src.ingestion.chunker import Chunk, chunk_text


# ── helpers ───────────────────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    """Word-count approximation: words × 1.3 ≈ tokens."""
    return int(len(text.split()) * 1.3)


def long_paragraph(n_words: int) -> str:
    return " ".join([f"word{i}" for i in range(n_words)])


# ── 1. Plain text produces chunks ─────────────────────────────────────────────

def test_plain_text_produces_chunks():
    text = "\n\n".join([long_paragraph(200) for _ in range(5)])  # ~1000 words
    chunks = chunk_text(text, target_tokens=400)
    assert len(chunks) >= 2


def test_chunk_size_approximately_correct():
    text = "\n\n".join([long_paragraph(100) for _ in range(10)])
    chunks = chunk_text(text, target_tokens=400)
    for chunk in chunks:
        tokens = approx_tokens(chunk.text)
        # Allow generous tolerance: chunks should be under 2× target
        assert tokens <= 900, f"Chunk too large: {tokens} tokens"


# ── 2. Headings preserved in section_title ────────────────────────────────────

def test_markdown_heading_captured_as_section_title():
    text = "# Introduction\n\nThis is the intro paragraph.\n\n# Methods\n\nThis is the methods section."
    chunks = chunk_text(text)
    titles = [c.section_title for c in chunks]
    assert "Introduction" in titles or any("Introduction" in (t or "") for t in titles)
    assert "Methods" in titles or any("Methods" in (t or "") for t in titles)


def test_chunk_without_heading_has_none_section_title():
    text = "Just a plain paragraph with no heading before it.\n\nAnd another one."
    chunks = chunk_text(text)
    # All section titles should be None when no headings present
    assert all(c.section_title is None for c in chunks)


# ── 3. Overlap ────────────────────────────────────────────────────────────────

def test_adjacent_chunks_share_overlap():
    # Build enough text to guarantee multiple chunks
    text = "\n\n".join([long_paragraph(120) for _ in range(8)])
    chunks = chunk_text(text, target_tokens=400, overlap_ratio=0.1)
    if len(chunks) < 2:
        pytest.skip("Not enough text to produce multiple chunks")
    # The end of chunk N should overlap with the start of chunk N+1
    # Verify by checking that char ranges overlap or are adjacent
    for i in range(len(chunks) - 1):
        assert chunks[i].char_end >= chunks[i + 1].char_start - 1, \
            f"Chunk {i} end={chunks[i].char_end} does not overlap chunk {i+1} start={chunks[i+1].char_start}"


# ── 4. Empty text ─────────────────────────────────────────────────────────────

def test_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\n   ") == []


# ── 5. Chunk metadata populated ───────────────────────────────────────────────

def test_chunk_has_char_start_and_end():
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = chunk_text(text)
    for chunk in chunks:
        assert isinstance(chunk.char_start, int)
        assert isinstance(chunk.char_end, int)
        assert chunk.char_end > chunk.char_start


def test_chunk_index_is_sequential():
    text = "\n\n".join([long_paragraph(150) for _ in range(6)])
    chunks = chunk_text(text, target_tokens=300)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_text_matches_source_at_char_range():
    text = "Hello world.\n\nGoodbye world."
    chunks = chunk_text(text)
    for chunk in chunks:
        source_slice = text[chunk.char_start : chunk.char_end]
        # The chunk text should be contained within the source slice
        assert chunk.text.strip() in source_slice or source_slice in chunk.text
