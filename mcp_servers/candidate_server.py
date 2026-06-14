"""
Candidate MCP server.

Exposes GRE corpus capabilities to external MCP-capable clients (Claude Desktop, etc.).
All tools are thin wrappers over the service layer — no business logic here.

Tools:
  - search_corpus(query, test_type, candidate_id, school_id) → list of chunks
  - get_question(test_type, section, question_type) → a random practice question
  - get_model_essay(prompt, score_tier) → an ETS anchor essay at the given scoring tier

Run with:
    python -m mcp_servers.candidate_server
"""
from __future__ import annotations

import random
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from evals.loader import load_golden_set
from shared.src.shared.config import get_settings

mcp = FastMCP("gre-tutor-candidate", instructions="GRE corpus search and practice material")


@mcp.tool()
async def search_corpus_tool(
    query: str,
    test_type: str,
    candidate_id: str,
    school_id: str = "",
) -> list[dict]:
    """
    Search the GRE corpus for relevant passages.

    Args:
        query: Search query (e.g. "quadratic equations", "issue essay structure")
        test_type: GRE section type (e.g. "gre")
        candidate_id: Candidate identifier for ACL-scoped retrieval
        school_id: School identifier (empty string for no school tier)

    Returns:
        List of chunks with text, tier, source_uri, page_or_section, score
    """
    return await _search_corpus(
        query=query, test_type=test_type,
        candidate_id=candidate_id, school_id=school_id,
    )


@mcp.tool()
async def get_question_tool(
    test_type: str,
    section: str,
    question_type: str = "",
) -> dict:
    """
    Get a random practice question from the Global corpus.

    Args:
        test_type: "gre"
        section: "quant" or "aw"
        question_type: Optional filter (e.g. "numeric_entry", "issue", "argument")

    Returns:
        A question dict with id, question_type, expression/prompt, and ground_truth
    """
    return await _get_random_question(
        test_type=test_type, section=section, question_type=question_type
    )


@mcp.tool()
async def get_model_essay_tool(prompt: str, score_tier: int) -> dict:
    """
    Get a sample ETS-style essay at the given scoring tier (1-6).

    Args:
        prompt: The AW prompt text
        score_tier: Desired score tier (1 = weak, 6 = outstanding)

    Returns:
        Dict with score_tier and text
    """
    return await _get_model_essay(prompt=prompt, score_tier=score_tier)


# ── service seams (patched in tests) ─────────────────────────────────────────


async def _search_corpus(
    query: str, test_type: str, candidate_id: str, school_id: str
) -> list[dict[str, Any]]:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{settings.corpus_service_url}/corpus/retrieve",
            json={"query": query, "test_type": test_type,
                  "candidate_id": candidate_id, "school_id": school_id, "top_k": 5},
        )
        r.raise_for_status()
        return r.json()


async def _get_random_question(
    test_type: str, section: str, question_type: str
) -> dict[str, Any]:
    suite = "quant" if section == "quant" else "aw"
    items = load_golden_set(suite)
    if question_type:
        items = [i for i in items if i.get("question_type") == question_type or
                 i.get("task_type") == question_type]
    if not items:
        return {"error": f"No questions found for {section}/{question_type}"}
    return random.choice(items)


async def _get_model_essay(prompt: str, score_tier: int) -> dict[str, Any]:
    items = load_golden_set("aw")
    for item in items:
        anchors = item.get("anchor_essays", {})
        tier_key = str(score_tier)
        if tier_key in anchors:
            return {"score_tier": score_tier, "text": anchors[tier_key]}
    return {"error": f"No anchor essay found at tier {score_tier}"}


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8091)
