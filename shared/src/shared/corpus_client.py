"""
The client for the corpus service's `POST /corpus/retrieve`, used by the agent's capabilities and
by the candidate MCP server, so the request shape and the timeout live in one place.
"""
from __future__ import annotations

from typing import Any, cast

import httpx

from shared.src.shared.http import post_json

RETRIEVE_TIMEOUT_SECONDS = 10.0


class CorpusClient:
    """Asks the corpus service for the chunks that answer a query, as the service returned them."""

    def __init__(self, base_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url
        self._client = client

    async def retrieve(
        self,
        query: str,
        *,
        candidate_id: str,
        school_id: str,
        test_type: str,
        top_k: int,
        apply_relevance_floor: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        The ranked chunks (a list of dicts). `apply_relevance_floor` is sent only when given, so a
        caller that does not set it gets the service's own default.
        """
        body: dict[str, Any] = {
            "query": query,
            "candidate_id": candidate_id,
            "school_id": school_id,
            "test_type": test_type,
            "top_k": top_k,
        }
        if apply_relevance_floor is not None:
            body["apply_relevance_floor"] = apply_relevance_floor
        reply = await post_json(
            f"{self._base_url}/corpus/retrieve",
            body,
            client=self._client,
            timeout=RETRIEVE_TIMEOUT_SECONDS,
        )
        return cast(list[dict[str, Any]], reply)
