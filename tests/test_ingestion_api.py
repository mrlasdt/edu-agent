"""
Tests for the Ingestion Service FastAPI endpoints.

Behaviors under test (from issue 06 acceptance criteria):
  1. Small TXT upload (<1MB) → processed synchronously, status=completed
  2. Large file (>1MB) → queued, returns job_id
  3. Unsupported file type → 415
  4. GET /admin/documents/{job_id} → job status
  5. Chunks stored with correct metadata fields
"""

import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from services.ingestion.src.ingestion.main import app


@pytest.fixture
def client():
    return TestClient(app)


def make_upload_file(content: bytes, filename: str, content_type: str):
    return ("file", (filename, io.BytesIO(content), content_type))


# ── 1. Small TXT upload → sync completed ─────────────────────────────────────

def test_small_txt_upload_returns_completed(client):
    content = b"This is a test document.\n\nSecond paragraph here."

    with patch("services.ingestion.src.ingestion.main.run_ingestion_pipeline", new_callable=AsyncMock) as mock_pipeline:
        mock_pipeline.return_value = {"chunks_indexed": 2}
        response = client.post(
            "/admin/documents",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
            data={"test_type": "gre", "tier": "global"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert "job_id" in body


# ── 2. Large file → queued ────────────────────────────────────────────────────

def test_large_file_upload_returns_queued(client):
    large_content = b"word " * 300_000  # ~1.5 MB

    with patch("services.ingestion.src.ingestion.main.enqueue_ingestion_job", new_callable=AsyncMock) as mock_enqueue:
        mock_enqueue.return_value = "job-abc-123"
        response = client.post(
            "/admin/documents",
            files={"file": ("large.txt", io.BytesIO(large_content), "text/plain")},
            data={"test_type": "gre", "tier": "global"},
        )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"] == "job-abc-123"


# ── 3. Unsupported file type → 415 ───────────────────────────────────────────

def test_unsupported_file_type_returns_415(client):
    response = client.post(
        "/admin/documents",
        files={"file": ("image.jpg", io.BytesIO(b"fake image"), "image/jpeg")},
        data={"test_type": "gre", "tier": "global"},
    )
    assert response.status_code == 415


def test_exe_file_returns_415(client):
    response = client.post(
        "/admin/documents",
        files={"file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")},
        data={"test_type": "gre", "tier": "global"},
    )
    assert response.status_code == 415


# ── 4. Job status endpoint ────────────────────────────────────────────────────

def test_get_job_status_returns_job(client):
    with patch("services.ingestion.src.ingestion.main.get_job", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {
            "job_id": "job-xyz",
            "status": "processing",
            "filename": "test.txt",
            "test_type": "gre",
            "error_stage": None,
        }
        response = client.get("/admin/documents/job-xyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "processing"
    assert body["job_id"] == "job-xyz"


def test_get_job_status_404_for_unknown_job(client):
    with patch("services.ingestion.src.ingestion.main.get_job", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        response = client.get("/admin/documents/nonexistent")

    assert response.status_code == 404


# ── 5. Chunk metadata has required fields ─────────────────────────────────────

def test_uploaded_chunks_have_correct_metadata(client):
    content = b"# Introduction\n\nThis is a GRE practice passage.\n\nIt tests reading comprehension."

    captured_chunks = []

    async def capture_pipeline(chunks, metadata, **kwargs):
        captured_chunks.extend(chunks)
        return {"chunks_indexed": len(chunks)}

    with patch("services.ingestion.src.ingestion.main.run_ingestion_pipeline", side_effect=capture_pipeline):
        client.post(
            "/admin/documents",
            files={"file": ("passage.txt", io.BytesIO(content), "text/plain")},
            data={"test_type": "gre", "tier": "global"},
        )

    assert len(captured_chunks) > 0
    for chunk in captured_chunks:
        assert "tier" in chunk
        assert "test_type" in chunk
        assert chunk["tier"] == "global"
        assert chunk["test_type"] == "gre"
