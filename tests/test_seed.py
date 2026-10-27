"""
Tests for the first-start corpus seed job.

Behaviors under test:
  1. Every shipped seed note is an allowed, non-empty upload with a heading
  2. An empty corpus gets every note uploaded as the global tier
  3. A fully seeded corpus is left alone (idempotent restart)
  4. An interrupted seed is finished on the next run without re-uploading indexed notes
  5. A 4xx from ingestion fails at once; a 5xx is retried
"""

import uuid
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, models

from services.ingestion.src.ingestion import seed
from services.ingestion.src.ingestion.chunker import chunk_text
from services.ingestion.src.ingestion.parser import is_allowed, parse_bytes
from shared.src.shared.vectorstore import COLLECTION, DENSE_VECTOR, ensure_collection

SEED_NOTES = sorted(Path("seed/gre-quant").glob("*.txt"))


def test_seed_notes_are_ingestible():
    assert SEED_NOTES
    for note in SEED_NOTES:
        assert is_allowed(note.name, "text/plain")
        chunks = chunk_text(parse_bytes(note.read_bytes(), note.name))
        assert chunks, note.name
        assert all(c.section_title for c in chunks), note.name


class _Corpus:
    """An in-memory Qdrant that survives the seed job closing its client."""

    def __init__(self) -> None:
        self.client = AsyncQdrantClient(location=":memory:")

    def get_client(self) -> AsyncQdrantClient:
        return self.client

    async def index(self, source_uri: str, tier: str = "global") -> None:
        await ensure_collection(self.client, dim=4)
        await self.client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={DENSE_VECTOR: [0.1, 0.2, 0.3, 0.4]},
                    payload={"source_uri": source_uri, "tier": tier, "test_type": "gre"},
                )
            ],
        )


@pytest.fixture
def corpus():
    c = _Corpus()
    with patch.object(seed, "get_qdrant_client", c.get_client), \
         patch.object(c.client, "close", AsyncMock()):
        yield c


def _ingestion(responses: Callable[[str], httpx.Response] | None = None):
    """MockTransport standing in for the ingestion service; records (filename, tier)."""
    posted: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("utf-8", errors="replace")
        name = next(n.name for n in SEED_NOTES if f'filename="{n.name}"' in body)
        posted.append((name, "global" if 'name="tier"\r\n\r\nglobal' in body else "other"))
        if responses:
            return responses(name)
        return httpx.Response(200, json={"status": "completed"})

    return httpx.MockTransport(handler), posted


async def _run_main(transport: httpx.MockTransport) -> int:
    real_client = httpx.AsyncClient
    with patch.object(
        seed.httpx, "AsyncClient", lambda **kw: real_client(transport=transport, **kw)
    ):
        return await seed.main()


@pytest.mark.asyncio
async def test_empty_corpus_gets_every_note_as_global_tier(corpus):
    transport, posted = _ingestion()

    assert await _run_main(transport) == 0

    assert [name for name, _ in posted] == [n.name for n in SEED_NOTES]
    assert {tier for _, tier in posted} == {"global"}


@pytest.mark.asyncio
async def test_seeded_corpus_is_not_reuploaded(corpus):
    for note in SEED_NOTES:
        await corpus.index(note.name)
    transport, posted = _ingestion()

    assert await _run_main(transport) == 0

    assert posted == []


@pytest.mark.asyncio
async def test_interrupted_seed_uploads_only_the_missing_notes(corpus):
    done, rest = SEED_NOTES[:4], SEED_NOTES[4:]
    for note in done:
        await corpus.index(note.name)
    # Unrelated documents (another tier, or another source) do not count as seeded.
    await corpus.index(rest[0].name, tier="school")
    await corpus.index("someone-elses-notes.txt")
    transport, posted = _ingestion()

    assert await _run_main(transport) == 0

    assert [name for name, _ in posted] == [n.name for n in rest]


@pytest.mark.asyncio
async def test_client_error_fails_fast_without_retrying(corpus):
    transport, posted = _ingestion(lambda _: httpx.Response(415, text="unsupported"))

    with patch.object(seed, "_READY_TIMEOUT_S", 3600.0), patch.object(seed, "_RETRY_DELAY_S", 0):
        assert await _run_main(transport) == 1

    assert len(posted) == 1


@pytest.mark.asyncio
async def test_server_error_is_retried(corpus):
    attempts: dict[str, int] = {}

    def flaky(name: str) -> httpx.Response:
        attempts[name] = attempts.get(name, 0) + 1
        if attempts[name] == 1:
            return httpx.Response(503, text="embedder loading")
        return httpx.Response(200, json={"status": "completed"})

    transport, posted = _ingestion(flaky)

    with patch.object(seed, "_RETRY_DELAY_S", 0):
        assert await _run_main(transport) == 0

    assert [name for name, _ in posted] == [n.name for n in SEED_NOTES for _ in range(2)]
