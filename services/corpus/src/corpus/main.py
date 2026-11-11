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
from services.corpus.src.corpus.settings import CorpusSettings, get_settings
from shared.src.shared.corpus import ChunkPayload
from shared.src.shared.embedder import embed_query
from shared.src.shared.reranker import rerank_texts
from shared.src.shared.sparse import encode_query
from shared.src.shared.vectorstore import (
    COLLECTION,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    check_collection_schema,
    ensure_payload_indexes,
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
    # Drop chunks below the relevance floor (see passes_relevance_floor) so an off-topic
    # question gets no notes. Off by default: callers that ask for a candidate's own essays
    # with a generic query ("essay") want every chunk, not only the on-topic ones.
    apply_relevance_floor: bool = False


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
        apply_relevance_floor=request.apply_relevance_floor,
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
        scores = await rerank_texts(
            query,
            [h["payload"].get("text", "") for h in window],
            url=get_settings().tei_reranker_url,
        )
        if len(scores) != len(window):
            raise ValueError(f"reranker returned {len(scores)} scores for {len(window)} texts")
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        logger.warning("reranker unavailable; keeping pre-rerank order", exc_info=True)
        return ranked
    reranked = [
        {**hit, "score": score, "reranked": True}
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
    apply_relevance_floor: bool = False,
) -> list[dict[str, Any]]:
    """
    The top_k citation-facing chunks for a query (see `search_candidates` for how they are
    found and ranked). With `apply_relevance_floor`, chunks whose relevance signal is below the
    configured floor are dropped (see `passes_relevance_floor`), so an off-topic message gets
    no notes at all instead of the three least-bad ones.
    """
    settings = get_settings()
    ranked = await search_candidates(
        query, candidate_id, school_id, test_type, top_k, settings=settings
    )
    if apply_relevance_floor:
        ranked = [h for h in ranked if passes_relevance_floor(h, settings)]
    return [to_chunk_result(h["payload"], h["score"]).model_dump() for h in ranked]


def passes_relevance_floor(hit: dict[str, Any], settings: CorpusSettings) -> bool:
    """
    Is this ranked hit relevant enough to show the model?

    A fused RRF score only says "rank 1 in both lists", whatever the query was, so it cannot
    tell an on-topic hit from the least-bad hit of an off-topic query. The floor therefore
    reads signals that keep their magnitude. Every hit must clear the dense cosine floor (hits
    only the sparse channel found carry no cosine and do not pass); a hit the cross-encoder
    reranked must also clear the rerank floor. The two scales differ, so each has its own floor
    in CorpusSettings. The rerank floor is 0 until it is measured, so the cosine floor is what
    keeps an off-topic query note-free while the reranker is on.
    """
    dense = hit.get("dense_score")
    if dense is None or float(dense) < settings.dense_relevance_floor:
        return False
    if hit.get("reranked"):
        return float(hit["score"]) >= settings.rerank_relevance_floor
    return True


async def search_candidates(
    query: str,
    candidate_id: str,
    school_id: str,
    test_type: str,
    top_k: int = 5,
    *,
    settings: CorpusSettings | None = None,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval against Qdrant (ADR-0004), returning the ranked top_k hits as
    {"id", "score", "payload", "dense_score"?, "reranked"?} dicts:
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

    `settings` defaults to the process settings; the retrieval eval passes its own so it can
    compare configurations without touching the environment.
    """
    settings = settings or get_settings()
    hybrid = settings.hybrid_retrieval_enabled
    try:
        query_vec = await embed_query(query, url=settings.tei_embedder_url)
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
        client = get_qdrant_client(settings.qdrant_url)
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

    dense_hits = _hits(dense_response)
    if sparse_response is None:
        ranked = _tier_tiebreak(dense_hits)
    else:
        fused = _rrf_merge(dense_hits, _hits(sparse_response))
        ranked = _tier_tiebreak(fused, threshold=RRF_TIE_THRESHOLD)
    # Keep the cosine similarity next to the fused or reranked score: it is the signal the
    # relevance floor can read when no reranker ran (see passes_relevance_floor).
    dense_scores = {h["id"]: h["score"] for h in dense_hits}
    sparse_scores = (
        {h["id"]: h["score"] for h in _hits(sparse_response)} if sparse_response is not None else {}
    )
    ranked = [
        {**h, "dense_score": dense_scores.get(h["id"]), "sparse_score": sparse_scores.get(h["id"])}
        for h in ranked
    ]
    if settings.tei_reranker_url:
        ranked = await _rerank(query, ranked, top_k)
    return ranked[:top_k]
