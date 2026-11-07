"""
Tests for the first-start corpus seed job.

Behaviors under test:
  1. Every shipped seed note is an allowed, non-empty upload with a heading
  2. An empty corpus gets every note uploaded with the admin key and no tier field (the
     ingestion service files it under the global tier); a missing key fails before any upload
  3. A restart uploads every note again: the ingestion service replaces a note's chunks in place
     (see tests/test_ingestion_api.py), so the seed no longer asks Qdrant what is indexed
  4. A 4xx from ingestion fails at once; a 5xx is retried
"""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from services.ingestion.src.ingestion import seed
from services.ingestion.src.ingestion.chunker import chunk_text
from services.ingestion.src.ingestion.parser import is_allowed, parse_bytes
from services.ingestion.src.ingestion.settings import get_settings
from shared.src.shared.config import ADMIN_KEY_HEADER

ADMIN_KEY = "seed-test-key"


@pytest.fixture(autouse=True)
def admin_key(monkeypatch):
    monkeypatch.setenv("INGESTION_ADMIN_KEY", ADMIN_KEY)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

SEED_NOTES = sorted(Path("seed/gre-quant").glob("*.txt"))


def test_seed_notes_are_ingestible():
    assert SEED_NOTES
    for note in SEED_NOTES:
        assert is_allowed(note.name, "text/plain")
        chunks = chunk_text(parse_bytes(note.read_bytes(), note.name))
        assert chunks, note.name
        assert all(c.section_title for c in chunks), note.name


def _ingestion(responses: Callable[[str], httpx.Response] | None = None):
    """MockTransport standing in for the ingestion service; records (filename, admin key sent)."""
    posted: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("utf-8", errors="replace")
        name = next(n.name for n in SEED_NOTES if f'filename="{n.name}"' in body)
        assert 'name="tier"' not in body  # the server picks the tier, never the client
        posted.append((name, request.headers.get(ADMIN_KEY_HEADER, "")))
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
async def test_empty_corpus_gets_every_note_uploaded_with_the_admin_key():
    transport, posted = _ingestion()

    assert await _run_main(transport) == 0

    assert [name for name, _ in posted] == [n.name for n in SEED_NOTES]
    assert {key for _, key in posted} == {ADMIN_KEY}


@pytest.mark.asyncio
async def test_missing_admin_key_fails_before_any_upload(monkeypatch):
    monkeypatch.setenv("INGESTION_ADMIN_KEY", "")
    get_settings.cache_clear()
    transport, posted = _ingestion()

    assert await _run_main(transport) == 1

    assert posted == []


@pytest.mark.asyncio
async def test_a_restart_uploads_every_note_again():
    transport, posted = _ingestion()

    assert await _run_main(transport) == 0
    assert await _run_main(transport) == 0

    assert [name for name, _ in posted] == [n.name for n in SEED_NOTES] * 2


@pytest.mark.asyncio
async def test_client_error_fails_fast_without_retrying():
    transport, posted = _ingestion(lambda _: httpx.Response(415, text="unsupported"))

    with patch.object(seed, "_READY_TIMEOUT_S", 3600.0), patch.object(seed, "_RETRY_DELAY_S", 0):
        assert await _run_main(transport) == 1

    assert len(posted) == 1


@pytest.mark.asyncio
async def test_server_error_is_retried():
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
