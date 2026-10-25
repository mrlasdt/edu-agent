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

# Named vectors: the collection holds a dense (BGE-M3 via TEI) and a sparse
# (lexical, see shared.sparse) vector per chunk; hybrid retrieval fuses the two.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


class CollectionSchemaError(RuntimeError):
    """The existing collection does not have the named dense + sparse layout."""


def get_qdrant_client(url: str | None = None) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=url or get_settings().qdrant_url)


async def check_collection_schema(
    client: AsyncQdrantClient, *, collection: str = COLLECTION
) -> None:
    """
    Raise CollectionSchemaError if `collection` exists without the named dense
    and sparse vectors (e.g. a pre-hybrid collection with one unnamed vector).
    Queries and upserts against such a collection are rejected by Qdrant, so
    fail loudly rather than let retrieval silently return nothing.
    """
    params = (await client.get_collection(collection)).config.params
    vectors = params.vectors
    sparse = params.sparse_vectors or {}
    if isinstance(vectors, dict) and DENSE_VECTOR in vectors and SPARSE_VECTOR in sparse:
        return
    raise CollectionSchemaError(
        f"Qdrant collection '{collection}' uses the legacy single-unnamed-vector layout; "
        f"it needs named '{DENSE_VECTOR}' + '{SPARSE_VECTOR}' vectors. Recreate the "
        "collection (delete it in dev) and re-ingest the corpus."
    )


async def ensure_collection(
    client: AsyncQdrantClient, *, collection: str = COLLECTION, dim: int = EMBED_DIM
) -> None:
    """
    Create the collection if it does not yet exist (idempotent); if it exists,
    verify it has the named-vector layout (see check_collection_schema).
    """
    if await client.collection_exists(collection):
        await check_collection_schema(client, collection=collection)
    else:
        await client.create_collection(
            collection_name=collection,
            vectors_config={DENSE_VECTOR: models.VectorParams(size=dim, distance=DISTANCE)},
            sparse_vectors_config={
                # IDF is computed by Qdrant over the collection, so the sparse
                # vectors only carry term frequencies.
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
