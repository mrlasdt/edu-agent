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

# Payload fields the access filter (corpus.retriever.build_search_filter) matches on.
PAYLOAD_INDEX_FIELDS = ("test_type", "tier", "school_id", "candidate_id")


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


async def ensure_payload_indexes(
    client: AsyncQdrantClient, *, collection: str = COLLECTION
) -> None:
    """
    Make sure every field the access filter matches on has a keyword payload
    index. Without one Qdrant cannot size the filter up front, so it either
    reads every point's payload (a slow scan) or walks the graph and returns
    too few, mostly wrong rows; with one it picks graph search or a scan by
    the filter's size. Fields already indexed are skipped, so this is safe to
    run at every start. Adding indexes to an already-loaded collection speeds
    up filtering but does not restore recall on a selective filter (the graph
    links are built at load time); re-ingest into a fresh collection for that.
    """
    existing = (await client.get_collection(collection)).payload_schema or {}
    for field in PAYLOAD_INDEX_FIELDS:
        if field not in existing:
            await client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )


async def ensure_collection(
    client: AsyncQdrantClient, *, collection: str = COLLECTION, dim: int = EMBED_DIM
) -> None:
    """
    Create the collection if it does not yet exist (idempotent); if it exists,
    verify it has the named-vector layout (see check_collection_schema). Either
    way, make sure the filter fields are indexed (see ensure_payload_indexes).
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
    await ensure_payload_indexes(client, collection=collection)
