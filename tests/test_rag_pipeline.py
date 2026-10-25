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
  6. search_corpus hybrid: dense + sparse searches share the ACL filter, RRF fuses,
     tier tiebreak still applies
  7. search_corpus dense-only fallback via HYBRID_RETRIEVAL_ENABLED=false
  8. run_ingestion_pipeline writes named dense + sparse vectors
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from qdrant_client import models

from services.corpus.src.corpus.main import search_corpus
from services.corpus.src.corpus.retriever import build_search_filter
from services.ingestion.src.ingestion.main import run_ingestion_pipeline
from shared.src.shared.config import Settings

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
    assert any(
        getattr(c, "key", None) == "candidate_id" and c.match.value == "c1"
        for c in cand_branch.must
    )


# ── 2. search_corpus happy path ───────────────────────────────────────────────


def _point(pid, score, payload):
    p = MagicMock()
    p.id, p.score, p.payload = pid, score, payload
    return p


def _settings(hybrid: bool) -> Settings:
    return Settings(hybrid_retrieval_enabled=hybrid)


@patch("services.corpus.src.corpus.main.get_settings", return_value=_settings(False))
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_projects_and_strips_acl_ids(mock_embed, mock_factory, _settings):
    mock_embed.return_value = [0.1, 0.2, 0.3, 0.4]
    client = AsyncMock()
    client.collection_exists.return_value = True
    client.query_points.return_value = MagicMock(
        points=[
            _point(
                "p1",
                0.92,
                {
                    "text": "GRE content",
                    "tier": "global",
                    "test_type": "gre",
                    "source_uri": "ets.pdf",
                    "page_or_section": "Intro",
                    "candidate_id": "secret",
                    "school_id": "secret",
                },
            )
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
    assert client.query_points.call_args.kwargs["using"] == "dense"
    client.close.assert_awaited_once()


# ── 6 & 7. hybrid retrieval ───────────────────────────────────────────────────


def _payload(text, tier="global"):
    return {
        "text": text,
        "tier": tier,
        "test_type": "gre",
        "source_uri": "doc.txt",
        "page_or_section": "s1",
    }


@patch("services.corpus.src.corpus.main.get_settings", return_value=_settings(True))
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_hybrid_fuses_dense_and_sparse_with_same_acl_filter(
    mock_embed, mock_factory, _settings
):
    mock_embed.return_value = [0.1, 0.2]
    dense = MagicMock(points=[_point("a", 0.9, _payload("A")), _point("b", 0.8, _payload("B"))])
    sparse = MagicMock(points=[_point("b", 5.0, _payload("B")), _point("c", 4.0, _payload("C"))])

    async def query_points(**kwargs):
        return dense if kwargs["using"] == "dense" else sparse

    client = AsyncMock()
    client.collection_exists.return_value = True
    client.query_points.side_effect = query_points
    mock_factory.return_value = client

    out = await search_corpus("rare term 42", "c1", "s1", "gre", top_k=3)

    # "b" is ranked by both channels, so RRF puts it first
    assert [r["text"] for r in out] == ["B", "A", "C"]
    calls = client.query_points.call_args_list
    assert {c.kwargs["using"] for c in calls} == {"dense", "sparse"}
    filters = [c.kwargs["query_filter"] for c in calls]
    assert filters[0] == filters[1] == build_search_filter("c1", "s1", "gre")
    sparse_query = next(c.kwargs["query"] for c in calls if c.kwargs["using"] == "sparse")
    assert isinstance(sparse_query, models.SparseVector)
    assert len(sparse_query.indices) == 3  # rare, term, 42
    client.close.assert_awaited_once()


@patch("services.corpus.src.corpus.main.get_settings", return_value=_settings(True))
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_hybrid_keeps_tier_tiebreak(mock_embed, mock_factory, _settings):
    mock_embed.return_value = [0.1]
    # Identical ranks in both channels give near-tied RRF scores; candidate tier wins.
    dense = MagicMock(
        points=[_point("g", 0.9, _payload("G")), _point("k", 0.8, _payload("K", "candidate"))]
    )
    sparse = MagicMock(
        points=[_point("g", 3.0, _payload("G")), _point("k", 2.0, _payload("K", "candidate"))]
    )

    async def query_points(**kwargs):
        return dense if kwargs["using"] == "dense" else sparse

    client = AsyncMock()
    client.collection_exists.return_value = True
    client.query_points.side_effect = query_points
    mock_factory.return_value = client

    out = await search_corpus("q", "c1", "s1", "gre", top_k=2)
    # G leads on RRF score (2/61 vs 2/62) but the 1.6% gap is inside the tiebreak threshold
    assert [r["text"] for r in out] == ["K", "G"]


@patch("services.corpus.src.corpus.main.get_settings", return_value=_settings(False))
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_dense_only_when_hybrid_disabled(mock_embed, mock_factory, _settings):
    mock_embed.return_value = [0.1]
    client = AsyncMock()
    client.collection_exists.return_value = True
    client.query_points.return_value = MagicMock(points=[_point("a", 0.9, _payload("A"))])
    mock_factory.return_value = client

    out = await search_corpus("q", "c1", "s1", "gre")

    assert [r["text"] for r in out] == ["A"]
    client.query_points.assert_awaited_once()
    assert client.query_points.call_args.kwargs["using"] == "dense"


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
    # 8. both named vectors are written
    for p in points:
        assert set(p.vector) == {"dense", "sparse"}
        assert isinstance(p.vector["sparse"], models.SparseVector)


@patch("services.ingestion.src.ingestion.main.embed_texts", new_callable=AsyncMock)
async def test_run_ingestion_pipeline_empty_short_circuits(mock_embed):
    assert await run_ingestion_pipeline([], {}) == {"chunks_indexed": 0}
    mock_embed.assert_not_awaited()
