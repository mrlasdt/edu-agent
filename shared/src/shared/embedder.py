"""
BGE-M3 embedding client (dense vectors via the TEI server).

Shared by both sides of Qdrant — ingestion (embed documents) and corpus (embed
queries) — because they MUST embed with the identical model/endpoint or the
query and document vectors live in different spaces and retrieval silently
returns garbage. Same single-source-of-truth rationale as ChunkPayload.

Note: this returns DENSE embeddings only. ADR-0004's hybrid (dense + sparse)
and the BGE reranker are architected but not wired in this slice — see
search_corpus for where they slot in.
"""
from __future__ import annotations

import httpx

from shared.src.shared.config import get_settings

# BGE-M3 dense dimensionality. Used as a fallback; ingestion derives the real
# dimension from the first embedding so the collection can never mismatch.
EMBED_DIM = 1024

# TEI accepts modest client batches; keep well under its limits.
_BATCH = 32


async def embed_texts(
    texts: list[str], *, url: str | None = None, timeout: float = 60.0
) -> list[list[float]]:
    """
    Embed a list of texts via the TEI server's /embed endpoint.
    Returns one dense vector per input, in order. Empty input → empty list.
    """
    if not texts:
        return []
    base = (url or get_settings().tei_embedder_url).rstrip("/")
    vectors: list[list[float]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for start in range(0, len(texts), _BATCH):
            batch = texts[start : start + _BATCH]
            resp = await client.post(f"{base}/embed", json={"inputs": batch})
            resp.raise_for_status()
            vectors.extend(resp.json())
    return vectors


async def embed_query(query: str, *, url: str | None = None) -> list[float]:
    """Embed a single query string; returns its dense vector."""
    vectors = await embed_texts([query], url=url)
    return vectors[0] if vectors else []
