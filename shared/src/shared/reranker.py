"""
BGE-reranker-v2-m3 client (cross-encoder scores via the TEI server).

Used by the corpus service to rerank the RRF-fused top candidates against the
query (ADR-0004). Served the same way as the embedder: a Hugging Face TEI
container, here exposing /rerank.
"""
from __future__ import annotations

import httpx

from shared.src.shared.config import get_settings


async def rerank_texts(
    query: str, texts: list[str], *, url: str | None = None, timeout: float = 2.0
) -> list[float]:
    """
    Score each text against the query via the TEI server's /rerank endpoint.
    Returns one relevance score per input, in input order (TEI sorts its response
    by score, so scores are mapped back through each result's `index`). Empty
    input → empty list. Raises httpx.HTTPError on transport/status failures and
    ValueError/KeyError on a malformed or incomplete body. The short default
    timeout keeps the corpus fail-open well inside callers' /corpus/retrieve budget.
    """
    if not texts:
        return []
    base = (url or get_settings().tei_reranker_url).rstrip("/")
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{base}/rerank", json={"query": query, "texts": texts})
        resp.raise_for_status()
        results = resp.json()
    scores: list[float | None] = [None] * len(texts)
    for item in results:
        scores[int(item["index"])] = float(item["score"])
    if any(s is None for s in scores):
        raise ValueError("reranker response did not score every text")
    return [s for s in scores if s is not None]
