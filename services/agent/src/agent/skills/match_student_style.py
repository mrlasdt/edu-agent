"""
Personal style Skill — retrieves Candidate's own essays from their Candidate corpus
and returns up to 2 as few-shot exemplars for style injection.

Returns [] (silent fallback to Canonical style) when fewer than 2 essays are available.
"""
from __future__ import annotations

import httpx

from shared.src.shared.config import get_settings

_MIN_ESSAYS = 2


class MatchStudentStyleSkill:
    def __init__(self, corpus_url: str | None = None) -> None:
        settings = get_settings()
        self._corpus_url = corpus_url or settings.corpus_service_url

    async def get_exemplars(self, candidate_id: str, test_type: str) -> list[str]:
        """
        Fetch up to _MIN_ESSAYS from the Candidate corpus.
        Returns [] if fewer than _MIN_ESSAYS are available (cold-start fallback).
        """
        raw = await self._fetch_candidate_essays(candidate_id, test_type)
        if len(raw) < _MIN_ESSAYS:
            return []
        return [r.get("text", "") for r in raw[:_MIN_ESSAYS]]

    async def _fetch_candidate_essays(self, candidate_id: str, test_type: str) -> list[dict]:
        """Retrieve Candidate-tier chunks from the Corpus Service."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self._corpus_url}/corpus/retrieve",
                json={
                    "query": "essay",
                    "candidate_id": candidate_id,
                    "school_id": "",
                    "test_type": test_type,
                    "top_k": 5,
                },
            )
            response.raise_for_status()
            results = response.json()
            # Filter to candidate-tier only
            return [r for r in results if r.get("tier") == "candidate"]
