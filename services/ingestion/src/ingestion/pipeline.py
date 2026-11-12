"""
The ingestion work itself, apart from the HTTP routes in `main.py`: parse a document, cut it into
chunks, embed them and write them to Qdrant. The evals call `build_chunks` and
`run_ingestion_pipeline` directly, so importing this module does not start the web app.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from services.ingestion.src.ingestion.chunker import chunk_text
from services.ingestion.src.ingestion.parser import parse_bytes
from services.ingestion.src.ingestion.settings import get_settings
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

# Namespace of the Qdrant point ids. It must never change: every id already stored was derived
# from it, and a new namespace would make a re-upload add a second copy of every chunk.
_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://edu-agent/ingestion/chunk-points")

# The server, not the uploader, decides which tier a document lands in. Only the shared
# global tier is writable for now: school and candidate tiers need an authenticated owner
# id to scope them, which arrives with session tokens (plan chapter 18, slice E2).
UPLOAD_TIER: Tier = "global"


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


def build_chunks(
    content: bytes, filename: str, test_type: str, source_uri: str
) -> list[dict[str, Any]]:
    text = parse_bytes(content, filename)
    return [
        ChunkPayload(
            text=c.text,
            tier=UPLOAD_TIER,
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


async def ingest(content: bytes, filename: str, test_type: str, source_uri: str) -> int:
    """Parse, chunk and index one document; returns the number of chunks indexed."""
    try:
        # Parsing and chunking a large PDF or DOCX is CPU work: keep it off the event loop.
        chunks = await asyncio.to_thread(build_chunks, content, filename, test_type, source_uri)
    except Exception as exc:
        raise IngestionError("parse", exc) from exc
    metadata = {
        "tier": UPLOAD_TIER,
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

    settings = get_settings()
    vectors = await embed_texts([c["text"] for c in chunks], url=settings.tei_embedder_url)
    client = get_qdrant_client(settings.qdrant_url)
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
