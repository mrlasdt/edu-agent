"""The corpus service's settings: only what retrieval reads."""
from __future__ import annotations

from functools import lru_cache

from shared.src.shared.config import ServiceSettings


class CorpusSettings(ServiceSettings):
    qdrant_url: str = "http://localhost:6333"
    tei_embedder_url: str = "http://localhost:8080"
    # Dense + sparse fused by RRF (ADR-0004). false = dense-only queries. Not a
    # rollback to the old collection layout: both modes need the named
    # dense/sparse vectors, so a legacy collection must be recreated and re-ingested.
    hybrid_retrieval_enabled: bool = True
    # BGE-reranker-v2-m3 cross-encoder over the RRF top candidates (ADR-0004).
    # Empty URL = no rerank.
    tei_reranker_url: str = ""
    # Relevance floors for requests with apply_relevance_floor
    # (corpus.search.passes_relevance_floor).
    # The two signals live on different scales, so each has its own floor. The dense floor was
    # read off the sweep table of `python -m evals retrieval` (BGE-M3 cosine, 55 seed
    # chunks, 46 queries): at 0.45 11 of 12 off-topic queries get no note and 2 of 34 on-topic
    # queries lose theirs (hybrid numbers, scored on the same queries that set the floor, so
    # in-sample). The rerank floor is 0 (no floor) until the reranker has been measured on hardware
    # where it answers inside its timeout; until then a reranked hit must also clear the dense
    # floor.
    dense_relevance_floor: float = 0.45
    rerank_relevance_floor: float = 0.0


@lru_cache
def get_settings() -> CorpusSettings:
    return CorpusSettings()
