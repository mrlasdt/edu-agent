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


@lru_cache
def get_settings() -> CorpusSettings:
    return CorpusSettings()
