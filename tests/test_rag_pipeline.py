"""
Tests for the real RAG wiring (dense slice): the search-filter builder, the
corpus search_corpus orchestration, and the ingestion run_ingestion_pipeline.

Qdrant and TEI are mocked here; live end-to-end is verified against the stack.

Behaviors under test:
  1. build_search_filter: test_type AND three-tier ACL, candidate id scoped
  2. search_corpus: embeds, queries, tiebreaks, projects (ACL ids stripped)
  3. search_corpus: degrades to [] on missing collection / embedder failure
  4. run_ingestion_pipeline: embeds + upserts one point per chunk with payload
  5. run_ingestion_pipeline: empty chunk list short-circuits
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from qdrant_client import models

from services.corpus.src.corpus.main import search_corpus
from services.corpus.src.corpus.retriever import build_search_filter
from services.ingestion.src.ingestion.main import run_ingestion_pipeline


# ── 1. filter builder ─────────────────────────────────────────────────────────

def test_build_search_filter_combines_test_type_and_acl():
    f = build_search_filter(candidate_id="c1", school_id="s1", test_type="gre")
    assert isinstance(f, models.Filter)
    # must[0] = test_type match; must[1] = nested should with the 3 ACL branches
    assert f.must[0].key == "test_type"
    assert f.must[0].match.value == "gre"
    acl = f.must[1]
    assert len(acl.should) == 3
    # the candidate branch carries the requesting candidate's id
    cand_branch = acl.should[2]
    assert any(getattr(c, "key", None) == "candidate_id" and c.match.value == "c1" for c in cand_branch.must)


# ── 2. search_corpus happy path ───────────────────────────────────────────────

def _point(pid, score, payload):
    p = MagicMock()
    p.id, p.score, p.payload = pid, score, payload
    return p


@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_projects_and_strips_acl_ids(mock_embed, mock_factory):
    mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]
    client = AsyncMock()
    client.collection_exists.return_value = True
    client.query_points.return_value = MagicMock(
        points=[
            _point("p1", 0.92, {
                "text": "GRE content", "tier": "global", "test_type": "gre",
                "source_uri": "ets.pdf", "page_or_section": "Intro",
                "candidate_id": "secret", "school_id": "secret",
            })
        ]
    )
    mock_factory.return_value = client

    out = await search_corpus("quadratics", "c1", "s1", "gre", top_k=5)

    assert len(out) == 1
    assert out[0]["text"] == "GRE content"
    assert out[0]["score"] == 0.92
    # ACL ids must never leak into a citation-facing result
    assert "candidate_id" not in out[0]
    assert "school_id" not in out[0]
    client.query_points.assert_awaited_once()
    client.close.assert_awaited_once()


# ── 3. graceful degradation ───────────────────────────────────────────────────

@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_empty_when_collection_missing(mock_embed, mock_factory):
    mock_embed.return_value = [0.1, 0.2]
    client = AsyncMock()
    client.collection_exists.return_value = False
    mock_factory.return_value = client
    assert await search_corpus("q", "c1", "s1", "gre") == []
    client.query_points.assert_not_awaited()


@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_degrades_on_embedder_failure(mock_embed):
    mock_embed.side_effect = httpx.ConnectError("tei down")
    assert await search_corpus("q", "c1", "s1", "gre") == []


# ── 4 & 5. ingestion pipeline ─────────────────────────────────────────────────

@patch("services.ingestion.src.ingestion.main.get_qdrant_client")
@patch("services.ingestion.src.ingestion.main.ensure_collection", new_callable=AsyncMock)
@patch("services.ingestion.src.ingestion.main.embed_texts", new_callable=AsyncMock)
async def test_run_ingestion_pipeline_upserts_one_point_per_chunk(
    mock_embed, mock_ensure, mock_factory
):
    mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
    client = AsyncMock()
    mock_factory.return_value = client
    chunks = [
        {"text": "alpha", "tier": "global", "test_type": "gre"},
        {"text": "beta", "tier": "global", "test_type": "gre"},
    ]

    result = await run_ingestion_pipeline(chunks, {"tier": "global"})

    assert result["chunks_indexed"] == 2
    mock_ensure.assert_awaited_once()
    client.upsert.assert_awaited_once()
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 2
    assert {p.payload["text"] for p in points} == {"alpha", "beta"}


@patch("services.ingestion.src.ingestion.main.embed_texts", new_callable=AsyncMock)
async def test_run_ingestion_pipeline_empty_short_circuits(mock_embed):
    assert await run_ingestion_pipeline([], {}) == {"chunks_indexed": 0}
    mock_embed.assert_not_awaited()
