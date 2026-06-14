from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.ingestion.src.ingestion.chunker import chunk_text
from services.ingestion.src.ingestion.parser import is_allowed, parse_bytes

app = FastAPI(title="Ingestion Service")

# In-memory job store for Phase 1 (replaced by Postgres in Phase 2)
_jobs: dict[str, dict] = {}

# Files under this threshold are processed synchronously
_SYNC_SIZE_LIMIT = 1 * 1024 * 1024  # 1 MB


# ── endpoints ─────────────────────────────────────────────────────────────────


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.post("/admin/documents")
async def upload_document(
    file: UploadFile = File(...),
    test_type: str = Form(...),
    tier: str = Form("global"),
):
    """
    Upload a document for ingestion into the corpus.
    Files under 1 MB are processed synchronously (200 + completed).
    Larger files are queued (202 + queued).
    Unsupported types return 415.
    """
    if not is_allowed(file.filename or "", file.content_type or ""):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    content = await file.read()
    job_id = str(uuid.uuid4())
    doc_metadata = {
        "tier": tier,
        "test_type": test_type,
        "source_uri": file.filename or job_id,
        "school_id": None,
        "candidate_id": None,
    }

    if len(content) < _SYNC_SIZE_LIMIT:
        # Synchronous fast path
        text = parse_bytes(content, file.filename or "upload.txt")
        chunks_raw = chunk_text(text)
        chunk_dicts = [
            {
                **doc_metadata,
                "text": c.text,
                "section_title": c.section_title,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "page_or_section": c.section_title or f"chunk-{c.chunk_index}",
            }
            for c in chunks_raw
        ]
        result = await run_ingestion_pipeline(chunk_dicts, doc_metadata)
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "completed",
            "filename": file.filename,
            "test_type": test_type,
            "chunks_indexed": result.get("chunks_indexed", len(chunk_dicts)),
            "error_stage": None,
        }
        return JSONResponse(status_code=200, content={"job_id": job_id, "status": "completed"})

    # Async path: enqueue
    queued_job_id = await enqueue_ingestion_job(
        content=content,
        filename=file.filename or "upload.bin",
        metadata=doc_metadata,
    )
    _jobs[queued_job_id] = {
        "job_id": queued_job_id,
        "status": "queued",
        "filename": file.filename,
        "test_type": test_type,
        "error_stage": None,
    }
    return JSONResponse(status_code=202, content={"job_id": queued_job_id, "status": "queued"})


@app.get("/admin/documents/{job_id}")
async def get_job_status(job_id: str):
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job


# ── seams (patched in tests / replaced by real impls in integration) ──────────


async def run_ingestion_pipeline(
    chunks: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    """
    Embed chunks via TEI server and index into Qdrant.
    Stub in unit tests; real implementation wired in integration tests.
    """
    # Phase 1 stub: just return a count
    return {"chunks_indexed": len(chunks)}


async def enqueue_ingestion_job(
    content: bytes, filename: str, metadata: dict[str, Any]
) -> str:
    """
    Push a large-file ingestion job to the pgmq queue.
    Stub in unit tests; real implementation uses pgmq in Phase 2.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": filename,
        "test_type": metadata.get("test_type"),
        "error_stage": None,
    }
    return job_id


async def get_job(job_id: str) -> dict | None:
    """Retrieve a job by ID. Stub backed by in-memory store."""
    return _jobs.get(job_id)
