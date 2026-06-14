from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from services.corpus.src.corpus.retriever import (
    _build_acl_filter,
    _rrf_merge,
    _tier_tiebreak,
)

app = FastAPI(title="Corpus Service")


class RetrieveRequest(BaseModel):
    query: str
    candidate_id: str
    school_id: str
    test_type: str
    top_k: int = 5


class ChunkResult(BaseModel):
    text: str
    tier: str
    source_uri: str
    page_or_section: str
    score: float


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.post("/corpus/retrieve", response_model=list[ChunkResult])
async def retrieve(request: RetrieveRequest) -> list[dict]:
    results = await search_corpus(
        query=request.query,
        candidate_id=request.candidate_id,
        school_id=request.school_id,
        test_type=request.test_type,
        top_k=request.top_k,
    )
    return results


# ── seam (patched in tests; real Qdrant + TEI + reranker in integration) ──────


async def search_corpus(
    query: str,
    candidate_id: str,
    school_id: str,
    test_type: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Full retrieval pipeline:
      1. Embed query via TEI server (BGE-M3 dense + sparse)
      2. Query Qdrant with ACL payload filter and test_type filter
      3. RRF merge dense + sparse results
      4. Rerank top-20 with BGE-reranker
      5. Tier tiebreak
      6. Return top_k with citation metadata

    Phase 1 stub: returns empty list. Wired to real Qdrant + TEI in integration.
    """
    acl_filter = _build_acl_filter(candidate_id=candidate_id, school_id=school_id)
    # Real implementation will:
    #   dense_results = await qdrant.search(query_vector=dense_vec, filter=acl_filter, test_type=test_type)
    #   sparse_results = await qdrant.search(query_vector=sparse_vec, filter=acl_filter, test_type=test_type)
    #   merged = _rrf_merge(dense_results, sparse_results)[:20]
    #   reranked = await reranker.rerank(query, merged)
    #   final = _tier_tiebreak(reranked)[:top_k]
    #   return [_to_chunk_result(r) for r in final]
    return []
