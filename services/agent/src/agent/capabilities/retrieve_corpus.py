from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.agent.src.agent.clients import pool
from services.agent.src.agent.settings import get_settings
from shared.src.shared.corpus_client import CorpusClient


@dataclass
class CorpusChunk:
    text: str
    tier: str
    source_uri: str
    page_or_section: str
    score: float


class RetrieveCorpus:
    """
    Calls the Corpus Service to retrieve relevant chunks for a query.
    The _call_corpus seam is patched in unit tests.
    """

    def __init__(
        self, corpus_url: str | None = None, *, apply_relevance_floor: bool = False
    ) -> None:
        settings = get_settings()
        self._corpus = CorpusClient(
            corpus_url or settings.corpus_service_url, client=pool.current()
        )
        self._apply_relevance_floor = apply_relevance_floor

    async def retrieve(
        self,
        query: str,
        candidate_id: str,
        school_id: str,
        test_type: str,
        top_k: int = 5,
    ) -> list[CorpusChunk]:
        raw = await self._call_corpus(
            query=query,
            candidate_id=candidate_id,
            school_id=school_id,
            test_type=test_type,
            top_k=top_k,
        )
        return [
            CorpusChunk(
                text=r.get("text", ""),
                tier=r.get("tier", "global"),
                source_uri=r.get("source_uri", ""),
                page_or_section=r.get("page_or_section", ""),
                score=float(r.get("score", 0.0)),
            )
            for r in raw
        ]

    async def _call_corpus(
        self,
        query: str,
        candidate_id: str,
        school_id: str,
        test_type: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        return await self._corpus.retrieve(
            query,
            candidate_id=candidate_id,
            school_id=school_id,
            test_type=test_type,
            top_k=top_k,
            apply_relevance_floor=self._apply_relevance_floor,
        )
