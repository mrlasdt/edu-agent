"""
Tests for the first-start corpus seed job.

Behaviors under test:
  1. Every shipped seed note is an allowed, non-empty upload with a heading
  2. An already-seeded corpus is left alone (idempotent restart)
  3. An empty corpus gets every note uploaded as the global tier
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from services.ingestion.src.ingestion import seed
from services.ingestion.src.ingestion.chunker import chunk_text
from services.ingestion.src.ingestion.parser import is_allowed, parse_bytes

SEED_NOTES = sorted(Path("seed/gre-quant").glob("*.txt"))


def test_seed_notes_are_ingestible():
    assert SEED_NOTES
    for note in SEED_NOTES:
        assert is_allowed(note.name, "text/plain")
        chunks = chunk_text(parse_bytes(note.read_bytes(), note.name))
        assert chunks, note.name
        assert all(c.section_title for c in chunks), note.name


@pytest.mark.asyncio
async def test_seeded_corpus_is_not_reuploaded():
    with patch.object(seed, "corpus_is_seeded", AsyncMock(return_value=True)), \
         patch.object(seed, "upload", AsyncMock()) as mock_upload:
        assert await seed.main() == 0
    mock_upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_corpus_gets_every_note_as_global_tier():
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("utf-8", errors="replace")
        name = next(n.name for n in SEED_NOTES if n.name in body)
        seen.append((name, "global" if 'name="tier"\r\n\r\nglobal' in body else "other"))
        return httpx.Response(200, json={"status": "completed"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        for note in SEED_NOTES:
            await seed.upload(client, "http://ingestion:8003", note)

    assert [name for name, _ in seen] == [n.name for n in SEED_NOTES]
    assert {tier for _, tier in seen} == {"global"}
