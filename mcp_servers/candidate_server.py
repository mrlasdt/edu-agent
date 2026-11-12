"""
Candidate MCP server.

Exposes GRE corpus capabilities to external MCP-capable clients (Claude Desktop, etc.).
All tools are thin wrappers over the service layer - no business logic here.

Tools (named <area>_<verb>, so a client that lists many servers can tell them apart):
  - corpus_search(query, test_type, response_format) -> reference notes for a topic
  - practice_get_question(section, question_type) -> one practice question, never its answer
  - essay_get_model(prompt_id, score_tier) -> an anchor essay for an AW prompt at a score tier

Who is asking (candidate and school) comes from this server's environment (MCP_CANDIDATE_ID,
MCP_SCHOOL_ID), never from a tool argument: the model cannot choose whose notes it reads.

A call that cannot be answered raises ToolError, which the client sees as an error result
(isError) whose message says what was wrong and which values would have worked.

Run with (stdio by default; MCP_TRANSPORT=sse serves SSE on 127.0.0.1:8091):
    python -m mcp_servers.candidate_server
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from mcp_servers.launch import bind_host, run
from mcp_servers.settings import get_settings
from shared.src.shared.corpus_client import CorpusClient

mcp = FastMCP(
    "gre-tutor-candidate",
    instructions=(
        "GRE study material: search the reference notes (corpus_search), get a practice "
        "question (practice_get_question), and read a model essay for an AW prompt "
        "(essay_get_model, using a prompt_id from practice_get_question)."
    ),
    host=bind_host(),
    port=8091,
)

ResponseFormat = Literal["concise", "detailed"]
Section = Literal["quant", "aw"]

# A concise search result keeps the first part of each note: enough to decide whether it is the
# right one, small enough that five of them stay a few hundred tokens.
_CONCISE_CHARS = 300
_TOP_K = 5


@mcp.tool()
async def corpus_search(
    query: str,
    test_type: str = "gre",
    response_format: ResponseFormat = "concise",
) -> dict[str, Any]:
    """
    Search the reference notes for a topic, a term or a worked idea (for example "difference of
    squares", "discriminant", "issue essay structure"). Use it to look something up before
    explaining it; it does not give practice questions (practice_get_question) or essays
    (essay_get_model). Searches the global notes plus the notes of the configured candidate.

    Args:
        query: What to look up, in a few words or one sentence.
        test_type: The exam; "gre" is the only one with notes today.
        response_format: "concise" (default) returns the source and the first ~300 characters of
            each of up to 5 notes; "detailed" returns each note in full with its source file and
            retrieval score. Search concise first and ask for detailed only to quote a note.

    Returns:
        {"count": n, "notes": [...]}; an empty list comes with a "hint" on how to widen the query.
    """
    if not query.strip():
        raise ToolError(
            "query is empty. Say what to look up, e.g. query='difference of squares'."
        )
    chunks = await _search_corpus(query=query, test_type=test_type)
    if response_format == "concise":
        notes = [
            {
                "source": c.get("page_or_section", ""),
                "tier": c.get("tier", ""),
                "text": _shorten(str(c.get("text", "")), _CONCISE_CHARS),
            }
            for c in chunks
        ]
    else:
        notes = [
            {k: c[k] for k in ("text", "tier", "source_uri", "page_or_section", "score") if k in c}
            for c in chunks
        ]
    result: dict[str, Any] = {"count": len(notes), "notes": notes}
    if not notes:
        result["hint"] = (
            "No note matched. Try the standard term for the idea (e.g. 'quadratic formula' "
            "instead of 'that formula with the square root') or a broader topic."
        )
    return result


@mcp.tool()
async def practice_get_question(
    section: Section,
    question_type: str = "",
) -> dict[str, Any]:
    """
    Get one practice question to work on. It never includes the answer. Use it to give a student
    something to try; for background on a topic use corpus_search instead.

    Args:
        section: "quant" or "aw" (the analytical writing section).
        question_type: Optional filter. Quant: the question's type, e.g. "numeric_entry",
            "quantitative_comparison", "multiple_choice". AW: "issue" or "argument". Leave empty
            for any type; a value that does not exist returns the list that does.

    Returns:
        Quant: {"id", "question_type", "question", "expression"}. AW: {"id", "task_type",
        "prompt"}; pass that id to essay_get_model as prompt_id to read a model essay for it.
    """
    return await _get_random_question(section=section, question_type=question_type)


@mcp.tool()
async def essay_get_model(prompt_id: str, score_tier: int) -> dict[str, Any]:
    """
    Get a scored anchor essay for one AW prompt, to show a student what a given score looks like.
    Get the prompt_id from practice_get_question(section="aw"). Not for Quant.

    Args:
        prompt_id: The id of an AW prompt, e.g. "aw001".
        score_tier: The score of the essay wanted, 1 (weak) to 6 (outstanding). Only some tiers
            exist per prompt; one that does not returns the tiers that do.

    Returns:
        {"prompt_id", "score_tier", "text"}.
    """
    return await _get_model_essay(prompt_id=prompt_id, score_tier=score_tier)


# -- service seams (patched in tests) ----------------------------------------------------------


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


async def _search_corpus(query: str, test_type: str) -> list[dict[str, Any]]:
    settings = get_settings()
    try:
        return await CorpusClient(settings.corpus_service_url).retrieve(
            query,
            candidate_id=settings.mcp_candidate_id,
            school_id=settings.mcp_school_id,
            test_type=test_type,
            top_k=_TOP_K,
        )
    except httpx.HTTPError as exc:
        raise ToolError(
            f"the notes service is not available right now ({type(exc).__name__}). Try again "
            "in a moment, or carry on without the notes."
        ) from exc


# What a candidate may see of a practice question. The golden rows are eval fixtures: they carry
# the answer key (ground_truth), the eval's internal failure label (failure_source) and tags, so
# the tool projects onto these fields instead of returning rows as they are.
_QUANT_FIELDS = ("id", "question_type", "question", "expression")
_AW_FIELDS = ("id", "task_type", "prompt")


def _load_practice_rows(section: str) -> list[dict[str, Any]]:
    """Read the practice file for `section` from the directory in MCP_PRACTICE_DIR."""
    path = Path(get_settings().mcp_practice_dir) / f"gre_{section}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _practice_rows(section: str) -> tuple[list[dict[str, Any]], tuple[str, ...], str]:
    """The rows a candidate can be given for `section`, the fields to show, the type key."""
    if section == "quant":
        # Only "solve" rows are practice questions; tutor, tutor_multiturn and security rows
        # are eval scenarios (a security row is a prompt-injection test string).
        rows = [i for i in _load_practice_rows("quant") if i.get("kind") == "solve"]
        return rows, _QUANT_FIELDS, "question_type"
    if section == "aw":
        return _load_practice_rows("aw"), _AW_FIELDS, "task_type"
    raise ToolError(f"section {section!r} is not available. Valid sections: 'quant', 'aw'.")


async def _get_random_question(section: str, question_type: str) -> dict[str, Any]:
    rows, fields, type_key = _practice_rows(section)
    if question_type:
        valid = sorted({str(r[type_key]) for r in rows if type_key in r})
        rows = [r for r in rows if r.get(type_key) == question_type]
        if not rows:
            raise ToolError(
                f"no {section} question has question_type {question_type!r}. "
                f"Valid question types for {section}: {', '.join(repr(v) for v in valid)}. "
                "Leave question_type empty for any type."
            )
    chosen = random.choice(rows)
    return {k: chosen[k] for k in fields if k in chosen}


async def _get_model_essay(prompt_id: str, score_tier: int) -> dict[str, Any]:
    items = {str(i["id"]): i for i in _load_practice_rows("aw") if "anchor_essays" in i}
    item = items.get(prompt_id)
    if item is None:
        raise ToolError(
            f"no AW prompt has id {prompt_id!r}. Valid prompt ids: "
            f"{', '.join(repr(k) for k in sorted(items))}. "
            "Get one from practice_get_question(section='aw')."
        )
    anchors: dict[str, str] = item["anchor_essays"]
    text = anchors.get(str(score_tier))
    if text is None:
        tiers = sorted(int(t) for t in anchors)
        raise ToolError(
            f"prompt {prompt_id!r} has no essay at score tier {score_tier}. "
            f"Tiers available for it: {', '.join(str(t) for t in tiers)}."
        )
    return {"prompt_id": prompt_id, "score_tier": score_tier, "text": text}


if __name__ == "__main__":
    run(mcp)
