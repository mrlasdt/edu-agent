"""
Personal style capability — retrieves Candidate's own essays from their Candidate corpus
and returns up to 2 as few-shot exemplars for style injection.

Returns [] (silent fallback to Canonical style) when fewer than 2 essays are available.
"""

from __future__ import annotations

from typing import Any

from services.agent.src.agent.clients import pool
from services.agent.src.agent.settings import get_settings
from shared.src.shared.corpus_client import CorpusClient

_MIN_ESSAYS = 2


class MatchStudentStyle:
    def __init__(self, corpus_url: str | None = None) -> None:
        settings = get_settings()
        self._corpus = CorpusClient(
            corpus_url or settings.corpus_service_url, client=pool.current()
        )

    async def get_exemplars(self, candidate_id: str, test_type: str) -> list[str]:
        """
        Fetch up to _MIN_ESSAYS from the Candidate corpus.
        Returns [] if fewer than _MIN_ESSAYS are available (cold-start fallback).
        """
        raw = await self._fetch_candidate_essays(candidate_id, test_type)
        if len(raw) < _MIN_ESSAYS:
            return []
        return [r.get("text", "") for r in raw[:_MIN_ESSAYS]]

    async def _fetch_candidate_essays(
        self, candidate_id: str, test_type: str
    ) -> list[dict[str, Any]]:
        """Retrieve Candidate-tier chunks from the Corpus Service."""
        results = await self._corpus.retrieve(
            "essay", candidate_id=candidate_id, school_id="", test_type=test_type, top_k=5
        )
        # Filter to candidate-tier only
        return [r for r in results if r.get("tier") == "candidate"]
