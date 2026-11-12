from __future__ import annotations

import logging
import secrets
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import JSONResponse

from services.ingestion.src.ingestion.parser import is_allowed
from services.ingestion.src.ingestion.pipeline import IngestionError, ingest
from services.ingestion.src.ingestion.settings import get_settings
from shared.src.shared.config import ADMIN_KEY_HEADER

app = FastAPI(title="Ingestion Service")

_log = logging.getLogger(__name__)

# Job records live in this process's memory: they are lost on a restart and invisible to a second
# replica, so the ingestion service runs as a single replica (one container in
# infra/docker-compose.yml). A durable job table with a real queue is the Phase 2 follow-up
# (ADR-0011).
_jobs: dict[str, dict[str, Any]] = {}

# Files under this threshold are indexed inside the upload request; larger ones are indexed by a
# FastAPI background task after the 202 is sent, and the job-status route reports how it ended.
_SYNC_SIZE_LIMIT = 1 * 1024 * 1024  # 1 MB


def require_admin_key(
    supplied: str | None = Header(default=None, alias=ADMIN_KEY_HEADER),
) -> None:
    """
    Gate for the /admin routes: the request must carry INGESTION_ADMIN_KEY in the
    X-Admin-Key header. An unset key refuses everything (503) rather than leaving the
    routes open, so a missing configuration is loud, not a silent hole.
    """
    expected = get_settings().ingestion_admin_key
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Ingestion is locked: INGESTION_ADMIN_KEY is not set on the server.",
        )
    if supplied is None or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid admin key.")


admin = APIRouter(prefix="/admin", dependencies=[Depends(require_admin_key)])


# ── endpoints ─────────────────────────────────────────────────────────────────


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _ingest_in_background(
    job_id: str, content: bytes, filename: str, test_type: str, source_uri: str
) -> None:
    """The background task of a large upload: runs the job and records how it ended."""
    job = _jobs[job_id]
    job["status"] = "processing"
    try:
        job["chunks_indexed"] = await ingest(content, filename, test_type, source_uri)
    except IngestionError as exc:
        _log.warning("ingestion job %s failed at %s: %s", job_id, exc.stage, exc, exc_info=True)
        job.update(status="failed", error_stage=exc.stage, error=str(exc))
        return
    job["status"] = "completed"


@admin.post("/documents")
async def upload_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    test_type: str = Form(...),
) -> JSONResponse:
    """
    Upload a document for ingestion into the corpus (global tier; see UPLOAD_TIER in pipeline.py).
    Requires the admin key.
    Files under 1 MB are indexed before the response (200 + completed).
    Larger files are answered at once (202 + queued) and indexed by a background task; poll
    GET /admin/documents/{job_id} for completed or failed. Job records are per process.
    Uploading a document again (same filename, tier and test type) replaces its chunks.
    Unsupported types return 415.
    """
    if not is_allowed(file.filename or "", file.content_type or ""):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    content = await file.read()
    job_id = str(uuid.uuid4())
    filename = file.filename or "upload.txt"
    source_uri = file.filename or job_id
    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "test_type": test_type,
        "chunks_indexed": None,
        "error_stage": None,
        "error": None,
    }
    _jobs[job_id] = job

    if len(content) < _SYNC_SIZE_LIMIT:
        job["chunks_indexed"] = await ingest(content, filename, test_type, source_uri)
        job["status"] = "completed"
        return JSONResponse(status_code=200, content={"job_id": job_id, "status": "completed"})

    background.add_task(_ingest_in_background, job_id, content, filename, test_type, source_uri)
    return JSONResponse(status_code=202, content={"job_id": job_id, "status": "queued"})


@admin.get("/documents/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job


app.include_router(admin)
