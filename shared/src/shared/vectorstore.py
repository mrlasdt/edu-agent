"""
Qdrant access shared across the boundary.

Both ingestion (write) and corpus (read) target the same collection with the
same vector config; centralising the name/distance here keeps the two sides
from drifting (a mismatched collection name or distance metric is a silent
"retrieval returns nothing" bug). The actual upsert/query calls live in each
service — only the shared invariants live here.
"""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from shared.src.shared.config import get_settings
from shared.src.shared.embedder import EMBED_DIM

# Single collection; tier and test_type are payload fields filtered at query
# time (ADR-0001/0004: one source of truth, no per-tier/-test indexes to drift).
COLLECTION = "corpus"
DISTANCE = models.Distance.COSINE


def get_qdrant_client(url: str | None = None) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=url or get_settings().qdrant_url)


async def ensure_collection(
    client: AsyncQdrantClient, *, collection: str = COLLECTION, dim: int = EMBED_DIM
) -> None:
    """Create the collection if it does not yet exist (idempotent)."""
    if not await client.collection_exists(collection):
        await client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=dim, distance=DISTANCE),
        )
