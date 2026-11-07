"""
Candidate MCP server.

Exposes GRE corpus capabilities to external MCP-capable clients (Claude Desktop, etc.).
All tools are thin wrappers over the service layer — no business logic here.

Tools:
  - search_corpus(query, test_type, candidate_id, school_id) → list of chunks
  - get_question(test_type, section, question_type) → a random practice question
  - get_model_essay(prompt, score_tier) → an ETS anchor essay at the given scoring tier

Run with (stdio by default; MCP_TRANSPORT=sse serves SSE on 127.0.0.1:8091):
    python -m mcp_servers.candidate_server
"""
from __future__ import annotations

import random
from typing import Any, cast

import httpx
from mcp.server.fastmcp import FastMCP

from evals.loader import load_golden_set
from mcp_servers.launch import bind_host, run
from mcp_servers.settings import get_settings

mcp = FastMCP(
    "gre-tutor-candidate",
    instructions="GRE corpus search and practice material",
    host=bind_host(),
    port=8091,
)


@mcp.tool()
async def search_corpus_tool(
    query: str,
    test_type: str,
    candidate_id: str,
    school_id: str = "",
) -> list[dict[str, Any]]:
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
) -> dict[str, Any]:
    """
    Get a random practice question from the Global corpus.

    Args:
        test_type: "gre"
        section: "quant" or "aw"
        question_type: Optional filter (e.g. "numeric_entry", "issue", "argument")

    Returns:
        A question dict with id, question_type and the question text (no answer key)
    """
    return await _get_random_question(
        test_type=test_type, section=section, question_type=question_type
    )


@mcp.tool()
async def get_model_essay_tool(prompt: str, score_tier: int) -> dict[str, Any]:
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
        return cast(list[dict[str, Any]], r.json())


# What a candidate may see of a practice question. The golden rows are eval fixtures: they carry
# the answer key (ground_truth), the eval's internal failure label (failure_source) and tags, so
# the tool projects onto these fields instead of returning rows as they are.
_QUANT_FIELDS = ("id", "question_type", "question", "expression")
_AW_FIELDS = ("id", "task_type", "prompt")


async def _get_random_question(
    test_type: str, section: str, question_type: str
) -> dict[str, Any]:
    if section == "quant":
        # Only "solve" rows are practice questions; tutor, tutor_multiturn and security rows
        # are eval scenarios (a security row is a prompt-injection test string).
        rows = [i for i in load_golden_set("quant") if i.get("kind") == "solve"]
        fields: tuple[str, ...] = _QUANT_FIELDS
    else:
        rows = load_golden_set("aw")
        fields = _AW_FIELDS
    if question_type:
        rows = [i for i in rows if i.get("question_type") == question_type or
                i.get("task_type") == question_type]
    if not rows:
        return {"error": f"No questions found for {section}/{question_type}"}
    chosen = random.choice(rows)
    return {k: chosen[k] for k in fields if k in chosen}


async def _get_model_essay(prompt: str, score_tier: int) -> dict[str, Any]:
    items = load_golden_set("aw")
    for item in items:
        anchors = item.get("anchor_essays", {})
        tier_key = str(score_tier)
        if tier_key in anchors:
            return {"score_tier": score_tier, "text": anchors[tier_key]}
    return {"error": f"No anchor essay found at tier {score_tier}"}


if __name__ == "__main__":
    run(mcp)
