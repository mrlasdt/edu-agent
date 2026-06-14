from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from shared.src.shared.config import get_settings


@dataclass
class CorpusChunk:
    text: str
    tier: str
    source_uri: str
    page_or_section: str
    score: float


class RetrieveCorpusSkill:
    """
    Calls the Corpus Service to retrieve relevant chunks for a query.
    The _call_corpus seam is patched in unit tests.
    """

    def __init__(self, corpus_url: str | None = None) -> None:
        settings = get_settings()
        self._corpus_url = corpus_url or settings.corpus_service_url

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
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self._corpus_url}/corpus/retrieve",
                json={
                    "query": query,
                    "candidate_id": candidate_id,
                    "school_id": school_id,
                    "test_type": test_type,
                    "top_k": top_k,
                },
            )
            response.raise_for_status()
            return response.json()
