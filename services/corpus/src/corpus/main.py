from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from services.corpus.src.corpus.retriever import _tier_tiebreak, build_search_filter
from shared.src.shared.corpus import ChunkPayload
from shared.src.shared.embedder import embed_query
from shared.src.shared.vectorstore import COLLECTION, get_qdrant_client

logger = logging.getLogger(__name__)

app = FastAPI(title="Corpus Service")


class RetrieveRequest(BaseModel):
    query: str
    candidate_id: str
    school_id: str
    test_type: str
    top_k: int = 5


class ChunkResult(BaseModel):
    """
    Citation-facing projection of ChunkPayload returned to the agent.
    Deliberately omits ACL ids (school_id / candidate_id) so they never leak
    into a retrieval response; adds the retrieval score.
    """

    text: str
    tier: str
    source_uri: str
    page_or_section: str
    score: float


def to_chunk_result(payload: dict, score: float) -> ChunkResult:
    """
    Parse a stored Qdrant payload through the shared ChunkPayload contract and
    project it into the citation-facing ChunkResult. Validating here means a
    write-side schema drift is caught at read time rather than producing a
    malformed citation.
    """
    parsed = ChunkPayload.model_validate(payload)
    return ChunkResult(
        text=parsed.text,
        tier=parsed.tier,
        source_uri=parsed.source_uri,
        page_or_section=parsed.page_or_section,
        score=score,
    )


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
    Dense retrieval against Qdrant:
      1. Embed the query via TEI (BGE-M3 dense)
      2. ANN search with the three-tier ACL + test_type payload filter
      3. Tier tiebreak (candidate > school > global on near-ties)
      4. Project the top_k into citation-facing ChunkResults

    Degrades to [] (no context, no citations) if the corpus is empty or the
    embedder/Qdrant is unreachable — the agent still answers, just ungrounded.

    Architected but not in this slice (ADR-0004): the sparse vector + _rrf_merge
    hybrid and the BGE-reranker over the top-20. They slot in between steps 2–3.
    """
    try:
        query_vec = await embed_query(query)
        if not query_vec:
            return []
        search_filter = build_search_filter(candidate_id, school_id, test_type)
        client = get_qdrant_client()
        try:
            if not await client.collection_exists(COLLECTION):
                return []
            response = await client.query_points(
                collection_name=COLLECTION,
                query=query_vec,
                query_filter=search_filter,
                limit=max(top_k, 20),  # over-fetch so the tiebreak has room
                with_payload=True,
            )
        finally:
            await client.close()
    except Exception:  # embedder down, Qdrant down, etc. — degrade, don't fail the turn
        logger.warning("corpus retrieval unavailable; returning no chunks", exc_info=True)
        return []

    hits = [
        {"id": str(p.id), "score": float(p.score or 0.0), "payload": p.payload or {}}
        for p in response.points
    ]
    ranked = _tier_tiebreak(hits)[:top_k]
    return [to_chunk_result(h["payload"], h["score"]).model_dump() for h in ranked]
