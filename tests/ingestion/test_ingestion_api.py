"""
Tests for the Ingestion Service FastAPI endpoints.

Behaviors under test (from issue 06 acceptance criteria):
  1. Small TXT upload (<1MB) → processed synchronously, status=completed
  2. Large file (>=1MB) → 202 queued, then really indexed in the background; the job-status
     route reports completed (with the chunk count) or failed (with the stage and the error)
  3. Unsupported file type → 415
  4. GET /admin/documents/{job_id} → job status
  5. Chunks stored with correct metadata fields
  6. Re-uploading a document replaces its points instead of adding a second copy (deterministic
     point ids), small or large, and a shorter new version leaves no stale tail
"""

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from services.ingestion.src.ingestion import main as ingestion_main
from services.ingestion.src.ingestion import pipeline as ingestion_pipeline
from services.ingestion.src.ingestion.main import app
from services.ingestion.src.ingestion.settings import get_settings
from shared.src.shared.config import ADMIN_KEY_HEADER
from shared.src.shared.vectorstore import COLLECTION

ADMIN_KEY = "test-admin-key"


@pytest.fixture
def client(monkeypatch):
    # The /admin routes need INGESTION_ADMIN_KEY; auth itself is covered by test_ingestion_auth.py.
    monkeypatch.setenv("INGESTION_ADMIN_KEY", ADMIN_KEY)
    get_settings.cache_clear()
    yield TestClient(app, headers={ADMIN_KEY_HEADER: ADMIN_KEY})
    get_settings.cache_clear()


def make_upload_file(content: bytes, filename: str, content_type: str):
    return ("file", (filename, io.BytesIO(content), content_type))


# ── 1. Small TXT upload → sync completed ─────────────────────────────────────


def test_small_txt_upload_returns_completed(client):
    content = b"This is a test document.\n\nSecond paragraph here."

    with patch(
        "services.ingestion.src.ingestion.pipeline.run_ingestion_pipeline", new_callable=AsyncMock
    ) as mock_pipeline:
        mock_pipeline.return_value = {"chunks_indexed": 2}
        response = client.post(
            "/admin/documents",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
            data={"test_type": "gre"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert "job_id" in body


# ── 2. Large file → queued, then indexed in the background ───────────────────


class _Corpus:
    """An in-memory Qdrant and a fake embedder behind the real ingestion pipeline."""

    def __init__(self) -> None:
        self.client = AsyncQdrantClient(location=":memory:")
        self.embed_error: Exception | None = None

    async def embed(self, texts: list[str], *, url: str = "") -> list[list[float]]:
        if self.embed_error:
            raise self.embed_error
        return [[1.0, float(len(t) % 7), 0.5, 0.25] for t in texts]

    async def points(self, source_uri: str | None = None):
        if not await self.client.collection_exists(COLLECTION):
            return []
        found, _ = await self.client.scroll(COLLECTION, limit=10_000, with_payload=True)
        return [p for p in found if source_uri is None or p.payload["source_uri"] == source_uri]


@pytest.fixture
def corpus(client):
    c = _Corpus()
    with patch.object(ingestion_pipeline, "get_qdrant_client", lambda url: c.client), \
         patch.object(ingestion_pipeline, "embed_texts", c.embed), \
         patch.object(c.client, "close", AsyncMock()):
        yield c


def _doc(paragraphs: int, size_kb: int = 0) -> bytes:
    """A text document of `paragraphs` blank-line separated paragraphs, padded to `size_kb`."""
    body = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(paragraphs))
    return (body + "\n\n" + "pad " * (size_kb * 256)).encode()


def _post(client, content: bytes, name: str = "notes.txt", test_type: str = "gre"):
    return client.post(
        "/admin/documents",
        files={"file": (name, io.BytesIO(content), "text/plain")},
        data={"test_type": test_type},
    )


def test_large_file_is_queued_then_really_indexed(client, corpus):
    large = _doc(paragraphs=40, size_kb=1100)  # a bit over 1 MB
    assert len(large) >= ingestion_main._SYNC_SIZE_LIMIT

    response = _post(client, large, "large.txt")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    job = client.get(f"/admin/documents/{body['job_id']}").json()
    assert job["status"] == "completed"
    assert job["error_stage"] is None
    stored = asyncio_run(corpus.points("large.txt"))
    assert stored and job["chunks_indexed"] == len(stored)
    assert {p.payload["tier"] for p in stored} == {"global"}


def test_large_file_that_cannot_be_parsed_is_reported_failed(client, corpus):
    garbage = b"%PDF-1.4 this is not a pdf " * 60_000

    response = _post_pdf(client, garbage)

    assert response.status_code == 202
    job = client.get(f"/admin/documents/{response.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert job["error_stage"] == "parse"
    assert job["error"]
    assert asyncio_run(corpus.points()) == []


def test_large_file_whose_embedding_fails_is_reported_failed(client, corpus):
    corpus.embed_error = RuntimeError("embedder down")

    response = _post(client, _doc(paragraphs=40, size_kb=1100), "large.txt")

    assert response.status_code == 202
    job = client.get(f"/admin/documents/{response.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert job["error_stage"] == "index"
    assert "embedder down" in job["error"]
    assert asyncio_run(corpus.points()) == []


def _post_pdf(client, content: bytes):
    return client.post(
        "/admin/documents",
        files={"file": ("broken.pdf", io.BytesIO(content), "application/pdf")},
        data={"test_type": "gre"},
    )


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


# ── 3. Unsupported file type → 415 ───────────────────────────────────────────


def test_unsupported_file_type_returns_415(client):
    response = client.post(
        "/admin/documents",
        files={"file": ("image.jpg", io.BytesIO(b"fake image"), "image/jpeg")},
        data={"test_type": "gre"},
    )
    assert response.status_code == 415


def test_exe_file_returns_415(client):
    response = client.post(
        "/admin/documents",
        files={"file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
        data={"test_type": "gre"},
    )
    assert response.status_code == 415


# ── 4. Job status endpoint ────────────────────────────────────────────────────


def test_get_job_status_returns_the_synchronous_job(client, corpus):
    job_id = _post(client, _doc(paragraphs=3)).json()["job_id"]

    response = client.get(f"/admin/documents/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "completed"
    assert body["filename"] == "notes.txt"
    assert body["test_type"] == "gre"
    assert body["chunks_indexed"] == len(asyncio_run(corpus.points()))


def test_get_job_status_404_for_unknown_job(client):
    response = client.get("/admin/documents/nonexistent")

    assert response.status_code == 404


# ── 5. Chunk metadata has required fields ─────────────────────────────────────


def test_uploaded_chunks_have_correct_metadata(client):
    content = (
        b"# Introduction\n\nThis is a GRE practice passage.\n\nIt tests reading comprehension."
    )

    captured_chunks = []

    async def capture_pipeline(chunks, metadata, **kwargs):
        captured_chunks.extend(chunks)
        return {"chunks_indexed": len(chunks)}

    with patch(
        "services.ingestion.src.ingestion.pipeline.run_ingestion_pipeline",
        side_effect=capture_pipeline,
    ):
        client.post(
            "/admin/documents",
            files={"file": ("passage.txt", io.BytesIO(content), "text/plain")},
            data={"test_type": "gre"},
        )

    assert len(captured_chunks) > 0
    for chunk in captured_chunks:
        assert "tier" in chunk
        assert "test_type" in chunk
        assert chunk["tier"] == "global"
        assert chunk["test_type"] == "gre"


# ── 6. Re-upload replaces, never duplicates ──────────────────────────────────


def test_reuploading_a_small_document_does_not_duplicate_its_points(client, corpus):
    doc = _doc(paragraphs=12)

    assert _post(client, doc).status_code == 200
    first = asyncio_run(corpus.points())
    assert _post(client, doc).status_code == 200
    second = asyncio_run(corpus.points())

    assert len(first) > 1
    assert sorted(p.id for p in second) == sorted(p.id for p in first)


def test_reuploading_a_large_document_does_not_duplicate_its_points(client, corpus):
    large = _doc(paragraphs=40, size_kb=1100)

    assert _post(client, large, "large.txt").status_code == 202
    first = asyncio_run(corpus.points())
    assert _post(client, large, "large.txt").status_code == 202
    second = asyncio_run(corpus.points())

    assert first
    assert sorted(p.id for p in second) == sorted(p.id for p in first)


def test_a_shorter_new_version_leaves_no_stale_chunks(client, corpus):
    _post(client, _doc(paragraphs=30))
    before = len(asyncio_run(corpus.points()))
    _post(client, _doc(paragraphs=6))
    after = asyncio_run(corpus.points())

    assert 0 < len(after) < before
    assert sorted(p.payload["chunk_index"] for p in after) == list(range(len(after)))


def test_the_same_filename_in_another_test_type_is_a_different_document(client, corpus):
    doc = _doc(paragraphs=6)

    _post(client, doc, test_type="gre")
    _post(client, doc, test_type="ielts")

    stored = asyncio_run(corpus.points())
    assert {p.payload["test_type"] for p in stored} == {"gre", "ielts"}
    assert len(stored) % 2 == 0 and len(stored) == len({p.id for p in stored})


def test_reuploading_a_document_removes_its_legacy_random_id_points(client, corpus):
    # A collection filled before ids were deterministic holds each chunk under a random uuid4.
    doc = _doc(paragraphs=12)
    assert _post(client, doc).status_code == 200
    current = asyncio_run(corpus.points())
    asyncio_run(_replace_with_legacy_ids(corpus, current))
    legacy_ids = {p.id for p in asyncio_run(corpus.points())}
    other = asyncio_run(_add_point(corpus, "other.txt"))
    assert legacy_ids and not legacy_ids & {p.id for p in current}

    assert _post(client, doc).status_code == 200

    stored = asyncio_run(corpus.points())
    notes = [p for p in stored if p.payload["source_uri"] == "notes.txt"]
    assert sorted(p.id for p in notes) == sorted(p.id for p in current)
    assert not legacy_ids & {p.id for p in stored}
    assert other in {p.id for p in stored}  # another document is left alone


async def _replace_with_legacy_ids(corpus: _Corpus, points) -> None:
    import uuid

    from qdrant_client import models

    full = await corpus.client.scroll(COLLECTION, limit=10_000, with_vectors=True)
    await corpus.client.delete(
        COLLECTION, points_selector=models.PointIdsList(points=[p.id for p in points])
    )
    await corpus.client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(id=str(uuid.uuid4()), vector=p.vector, payload=p.payload)
            for p in full[0]
        ],
    )


async def _add_point(corpus: _Corpus, source_uri: str) -> str:
    import uuid

    from qdrant_client import models

    point_id = str(uuid.uuid4())
    await corpus.client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector={"dense": [1.0, 0.0, 0.5, 0.25]},
                payload={
                    "source_uri": source_uri, "tier": "global", "test_type": "gre",
                    "chunk_index": 0, "text": "x",
                },
            )
        ],
    )
    return point_id
