from __future__ import annotations

import asyncio
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
from qdrant_client import AsyncQdrantClient, models

from services.ingestion.src.ingestion.chunker import chunk_text
from services.ingestion.src.ingestion.parser import is_allowed, parse_bytes
from shared.src.shared.config import ADMIN_KEY_HEADER, get_settings
from shared.src.shared.corpus import ChunkPayload, Tier
from shared.src.shared.embedder import embed_texts
from shared.src.shared.sparse import encode_document
from shared.src.shared.vectorstore import (
    COLLECTION,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    ensure_collection,
    get_qdrant_client,
)

app = FastAPI(title="Ingestion Service")

_log = logging.getLogger(__name__)

# Job records live in this process's memory: they are lost on a restart and invisible to a second
# replica, so the ingestion service runs as a single replica (infra/k8s/ingestion.yaml). A
# durable job table with a real queue is the Phase 2 follow-up (ADR-0011).
_jobs: dict[str, dict[str, Any]] = {}

# Files under this threshold are indexed inside the upload request; larger ones are indexed by a
# FastAPI background task after the 202 is sent, and the job-status route reports how it ended.
_SYNC_SIZE_LIMIT = 1 * 1024 * 1024  # 1 MB

# Namespace of the Qdrant point ids. It must never change: every id already stored was derived
# from it, and a new namespace would make a re-upload add a second copy of every chunk.
_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://edu-agent/ingestion/chunk-points")

# The server, not the uploader, decides which tier a document lands in. Only the shared
# global tier is writable for now: school and candidate tiers need an authenticated owner
# id to scope them, which arrives with session tokens (plan chapter 18, slice E2).
_UPLOAD_TIER: Tier = "global"


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


class IngestionError(Exception):
    """A document could not be ingested; `stage` is "parse" or "index"."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.stage = stage


def chunk_point_id(chunk: dict[str, Any]) -> str:
    """
    The Qdrant point id of a chunk: the same document (tier, test type, source) and chunk
    position always give the same id, so upserting a document again replaces its points instead
    of adding a second copy.
    """
    key = f"{chunk['tier']}:{chunk['test_type']}:{chunk['source_uri']}:{chunk['chunk_index']}"
    return str(uuid.uuid5(_POINT_NAMESPACE, key))


def _build_chunks(
    content: bytes, filename: str, test_type: str, source_uri: str
) -> list[dict[str, Any]]:
    text = parse_bytes(content, filename)
    return [
        ChunkPayload(
            text=c.text,
            tier=_UPLOAD_TIER,
            test_type=test_type,
            source_uri=source_uri,
            page_or_section=c.section_title or f"chunk-{c.chunk_index}",
            section_title=c.section_title,
            chunk_index=c.chunk_index,
            char_start=c.char_start,
            char_end=c.char_end,
            school_id=None,
            candidate_id=None,
        ).model_dump()
        for c in chunk_text(text)
    ]


async def _ingest(content: bytes, filename: str, test_type: str, source_uri: str) -> int:
    """Parse, chunk and index one document; returns the number of chunks indexed."""
    try:
        # Parsing and chunking a large PDF or DOCX is CPU work: keep it off the event loop.
        chunks = await asyncio.to_thread(_build_chunks, content, filename, test_type, source_uri)
    except Exception as exc:
        raise IngestionError("parse", exc) from exc
    metadata = {
        "tier": _UPLOAD_TIER,
        "test_type": test_type,
        "source_uri": source_uri,
        "school_id": None,
        "candidate_id": None,
    }
    try:
        result = await run_ingestion_pipeline(chunks, metadata)
    except Exception as exc:
        raise IngestionError("index", exc) from exc
    return int(result.get("chunks_indexed", len(chunks)))


async def _ingest_in_background(
    job_id: str, content: bytes, filename: str, test_type: str, source_uri: str
) -> None:
    """The background task of a large upload: runs the job and records how it ended."""
    job = _jobs[job_id]
    job["status"] = "processing"
    try:
        job["chunks_indexed"] = await _ingest(content, filename, test_type, source_uri)
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
    Upload a document for ingestion into the corpus (global tier; see _UPLOAD_TIER).
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
        job["chunks_indexed"] = await _ingest(content, filename, test_type, source_uri)
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


# ── pipeline (patched in tests that do not need a real index) ─────────────────


async def run_ingestion_pipeline(
    chunks: list[dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    """
    Embed each chunk's text via TEI (BGE-M3 dense), add its lexical sparse
    vector, and upsert both as named vectors into Qdrant with the ChunkPayload
    as the point payload. The collection is created on first
    write using the embedding's actual dimension, so it can never mismatch the
    model. Raises on embedder/Qdrant failure so the upload surfaces the error.
    """
    if not chunks:
        return {"chunks_indexed": 0}

    vectors = await embed_texts([c["text"] for c in chunks])
    client = get_qdrant_client()
    try:
        await ensure_collection(client, dim=len(vectors[0]))
        points = [
            models.PointStruct(
                id=chunk_point_id(chunk),
                vector={DENSE_VECTOR: vector, SPARSE_VECTOR: encode_document(chunk["text"])},
                payload=chunk,
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        await client.upsert(collection_name=COLLECTION, points=points)
        # The upload replaces the whole document: delete every other point of it, which is the
        # tail of a longer old version and any point stored under a random id before ids were
        # deterministic. This runs after the upsert, so a failure leaves old plus new chunks,
        # never none.
        await _delete_replaced_points(client, chunks, [p.id for p in points])
    finally:
        await client.close()
    return {"chunks_indexed": len(points)}


async def _delete_replaced_points(
    client: AsyncQdrantClient, chunks: list[dict[str, Any]], keep_ids: list[Any]
) -> None:
    """Delete the points of this document (same source, tier and test type) not in `keep_ids`."""
    first = chunks[0]
    await client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_uri", match=models.MatchValue(value=first["source_uri"])
                    ),
                    models.FieldCondition(key="tier", match=models.MatchValue(value=first["tier"])),
                    models.FieldCondition(
                        key="test_type", match=models.MatchValue(value=first["test_type"])
                    ),
                ],
                must_not=[models.HasIdCondition(has_id=keep_ids)],
            )
        ),
    )
