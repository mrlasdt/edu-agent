from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from services.corpus.src.corpus.retriever import (
    RRF_TIE_THRESHOLD,
    _rrf_merge,
    _tier_tiebreak,
    build_search_filter,
)
from shared.src.shared.config import get_settings
from shared.src.shared.corpus import ChunkPayload
from shared.src.shared.embedder import embed_query
from shared.src.shared.reranker import rerank_texts
from shared.src.shared.sparse import encode_query
from shared.src.shared.vectorstore import (
    COLLECTION,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    check_collection_schema,
    get_qdrant_client,
)

logger = logging.getLogger(__name__)

# Fused hits reranked by the cross-encoder (ADR-0004: top-20 -> top-5); also the
# per-channel Qdrant over-fetch, so the window is always filled when hits exist.
RERANK_CANDIDATES = 20


def _is_transient(exc: Exception) -> bool:
    """
    Infrastructure trouble that degrades to "no chunks": unreachable embedder or
    Qdrant, or Qdrant 5xx. Anything else, notably a 4xx schema mismatch rejected
    by Qdrant, must surface.
    """
    if isinstance(exc, UnexpectedResponse):
        return (exc.status_code or 0) >= 500
    return isinstance(exc, httpx.HTTPError | ResponseHandlingException)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """
    Fail startup if an existing collection predates the named dense + sparse
    layout (CollectionSchemaError). Qdrant being down or the collection not
    existing yet is fine: retrieval degrades until ingestion creates it.
    """
    client = get_qdrant_client()
    try:
        if await client.collection_exists(COLLECTION):
            await check_collection_schema(client)
    except Exception as exc:
        if not _is_transient(exc):
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


class ChunkResult(BaseModel):
    """
    Citation-facing projection of ChunkPayload returned to the agent.
    Deliberately omits ACL ids (school_id / candidate_id) so they never leak
    into a retrieval response; adds the retrieval score.

    `score` is only meaningful for ordering, and its scale depends on the
    request: the reranker's relevance score after a successful rerank, otherwise
    the RRF fused score (about 0.03 at best) in hybrid mode, or cosine
    similarity when dense-only or when the sparse query fell back to dense.
    """

    text: str
    tier: str
    source_uri: str
    page_or_section: str
    score: float


def to_chunk_result(payload: dict[str, Any], score: float) -> ChunkResult:
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
    )
    return results


async def _rerank(query: str, ranked: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """
    Reorder the top candidates by cross-encoder score; hits beyond the candidate
    window keep their place after them. Fails open to the incoming order.
    """
    window = ranked[: max(top_k, RERANK_CANDIDATES)]
    if not window:
        return ranked
    try:
        scores = await rerank_texts(query, [h["payload"].get("text", "") for h in window])
        if len(scores) != len(window):
            raise ValueError(f"reranker returned {len(scores)} scores for {len(window)} texts")
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        logger.warning("reranker unavailable; keeping pre-rerank order", exc_info=True)
        return ranked
    reranked = [
        {**hit, "score": score}
        for hit, score in sorted(zip(window, scores, strict=True), key=lambda x: -x[1])
    ]
    return _tier_tiebreak(reranked, threshold=RRF_TIE_THRESHOLD) + ranked[len(window) :]


# ── seam (patched in tests; real Qdrant + TEI + reranker in integration) ──────


async def search_corpus(
    query: str,
    candidate_id: str,
    school_id: str,
    test_type: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval against Qdrant (ADR-0004):
      1. Embed the query via TEI (BGE-M3 dense) and encode its sparse vector
      2. Dense and sparse ANN searches, each with the three-tier ACL +
         test_type payload filter, fused with Reciprocal Rank Fusion
      3. Tier tiebreak (candidate > school > global on near-ties: within 5% of
         the cosine score dense-only, exact RRF-score ties when fused)
      4. Cross-encoder rerank (BGE-reranker-v2-m3 via TEI) of the top
         RERANK_CANDIDATES (at least top_k), when TEI_RERANKER_URL is set; exact
         reranker-score ties again prefer the more specific tier
      5. Project the top_k into citation-facing ChunkResults

    With HYBRID_RETRIEVAL_ENABLED=false only the dense search runs. If the
    sparse search fails the dense results are used on their own.

    Degrades to [] (no context, no citations) if the corpus is empty or the
    embedder/Qdrant is unavailable — the agent still answers, just ungrounded.
    Other failures (e.g. a 4xx schema rejection) propagate.

    The rerank step fails open: if the reranker is unreachable or returns a bad
    response, the pre-rerank order (and scores) are used and a warning is logged.
    After a successful rerank, `score` is the reranker's relevance score.
    """
    settings = get_settings()
    hybrid = settings.hybrid_retrieval_enabled
    try:
        query_vec = await embed_query(query)
    except (httpx.HTTPError, ValueError, KeyError):  # embedder down or malformed body
        logger.warning("embedder unavailable; returning no chunks", exc_info=True)
        return []
    if not query_vec:
        return []
    dense_response: Any
    sparse_response: Any
    try:
        search_filter = build_search_filter(candidate_id, school_id, test_type)
        limit = max(top_k, RERANK_CANDIDATES)  # over-fetch for fusion, tiebreak and rerank
        client = get_qdrant_client()
        try:
            if not await client.collection_exists(COLLECTION):
                return []
            dense_search = client.query_points(
                collection_name=COLLECTION,
                query=query_vec,
                using=DENSE_VECTOR,
                query_filter=search_filter,
                limit=limit,
                with_payload=True,
            )
            if hybrid:
                sparse_search = client.query_points(
                    collection_name=COLLECTION,
                    query=encode_query(query),
                    using=SPARSE_VECTOR,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True,
                )
                dense_response, sparse_response = await asyncio.gather(
                    dense_search, sparse_search, return_exceptions=True
                )
                if isinstance(dense_response, BaseException):
                    raise dense_response
                if isinstance(sparse_response, BaseException):
                    if not isinstance(sparse_response, Exception):
                        raise sparse_response
                    logger.warning(
                        "sparse search failed; using dense results only",
                        exc_info=sparse_response,
                    )
                    sparse_response = None
            else:
                dense_response, sparse_response = await dense_search, None
        finally:
            await client.close()
    except Exception as exc:  # Qdrant down — degrade, don't fail the turn
        if not _is_transient(exc):
            raise
        logger.warning("corpus retrieval unavailable; returning no chunks", exc_info=True)
        return []

    def _hits(response: Any) -> list[dict[str, Any]]:
        return [
            {"id": str(p.id), "score": float(p.score or 0.0), "payload": p.payload or {}}
            for p in response.points
        ]

    if sparse_response is None:
        ranked = _tier_tiebreak(_hits(dense_response))
    else:
        fused = _rrf_merge(_hits(dense_response), _hits(sparse_response))
        ranked = _tier_tiebreak(fused, threshold=RRF_TIE_THRESHOLD)
    if settings.tei_reranker_url:
        ranked = await _rerank(query, ranked, top_k)
    ranked = ranked[:top_k]
    return [to_chunk_result(h["payload"], h["score"]).model_dump() for h in ranked]
