from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from services.corpus.src.corpus.search import ChunkResult, is_transient, search_corpus
from services.corpus.src.corpus.settings import get_settings
from shared.src.shared.vectorstore import (
    COLLECTION,
    check_collection_schema,
    ensure_payload_indexes,
    get_qdrant_client,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """
    Fail startup if an existing collection predates the named dense + sparse
    layout (CollectionSchemaError). Qdrant being down or the collection not
    existing yet is fine: retrieval degrades until ingestion creates it.

    An existing collection also gets the keyword payload indexes on the filter
    fields (a no-op once present). That fixes filter latency, but not recall on
    a selective filter: Qdrant builds payload-aware graph links when points are
    indexed, so a collection loaded before the indexes existed must be
    re-ingested into a fresh collection to recover recall.
    """
    client = get_qdrant_client(get_settings().qdrant_url)
    try:
        if await client.collection_exists(COLLECTION):
            await check_collection_schema(client)
            await ensure_payload_indexes(client)
    except Exception as exc:
        if not is_transient(exc):
            raise
        logger.warning("qdrant unavailable at startup; skipping schema check", exc_info=True)
    finally:
        await client.close()
    yield


app = FastAPI(title="Corpus Service", lifespan=lifespan)


class RetrieveRequest(BaseModel):
    query: str
    candidate_id: str
    school_id: str
    test_type: str
    top_k: int = 5
    # Drop chunks below the relevance floor (see passes_relevance_floor) so an off-topic
    # question gets no notes. Off by default: callers that ask for a candidate's own essays
    # with a generic query ("essay") want every chunk, not only the on-topic ones.
    apply_relevance_floor: bool = False


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/corpus/retrieve", response_model=list[ChunkResult])
async def retrieve(request: RetrieveRequest) -> list[dict[str, Any]]:
    results = await search_corpus(
        query=request.query,
        candidate_id=request.candidate_id,
        school_id=request.school_id,
        test_type=request.test_type,
        top_k=request.top_k,
        apply_relevance_floor=request.apply_relevance_floor,
    )
    return results


