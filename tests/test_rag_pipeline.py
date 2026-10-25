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
     tier only breaks exact RRF ties, sparse failure falls back to dense
  7. search_corpus dense-only fallback via HYBRID_RETRIEVAL_ENABLED=false
  8. run_ingestion_pipeline writes named dense + sparse vectors
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from services.corpus.src.corpus.main import app, lifespan, search_corpus
from services.corpus.src.corpus.retriever import build_search_filter
from services.ingestion.src.ingestion.main import run_ingestion_pipeline
from shared.src.shared.config import Settings
from shared.src.shared.vectorstore import CollectionSchemaError, ensure_collection

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


def _hybrid_client(dense_points, sparse_points):
    dense = MagicMock(points=dense_points)
    sparse = MagicMock(points=sparse_points)

    async def query_points(**kwargs):
        return dense if kwargs["using"] == "dense" else sparse

    client = AsyncMock()
    client.collection_exists.return_value = True
    client.query_points.side_effect = query_points
    return client


@patch("services.corpus.src.corpus.main.get_settings", return_value=_settings(True))
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_hybrid_does_not_demote_higher_ranked_chunk_by_tier(
    mock_embed, mock_factory, _settings
):
    mock_embed.return_value = [0.1]
    # G outranks K in both channels (2/61 vs 2/62): tier must not reorder them.
    mock_factory.return_value = _hybrid_client(
        [_point("g", 0.9, _payload("G")), _point("k", 0.8, _payload("K", "candidate"))],
        [_point("g", 3.0, _payload("G")), _point("k", 2.0, _payload("K", "candidate"))],
    )

    out = await search_corpus("q", "c1", "s1", "gre", top_k=2)

    assert [r["text"] for r in out] == ["G", "K"]


@patch("services.corpus.src.corpus.main.get_settings", return_value=_settings(True))
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_hybrid_tier_breaks_exact_rrf_ties(mock_embed, mock_factory, _settings):
    mock_embed.return_value = [0.1]
    # Ranks swapped between channels give identical RRF scores; candidate tier wins.
    mock_factory.return_value = _hybrid_client(
        [_point("g", 0.9, _payload("G")), _point("k", 0.8, _payload("K", "candidate"))],
        [_point("k", 3.0, _payload("K", "candidate")), _point("g", 2.0, _payload("G"))],
    )

    out = await search_corpus("q", "c1", "s1", "gre", top_k=2)

    assert [r["text"] for r in out] == ["K", "G"]


@patch("services.corpus.src.corpus.main.get_settings", return_value=_settings(True))
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_hybrid_falls_back_to_dense_when_sparse_fails(
    mock_embed, mock_factory, _settings
):
    mock_embed.return_value = [0.1]
    dense = MagicMock(points=[_point("a", 0.9, _payload("A")), _point("b", 0.8, _payload("B"))])

    async def query_points(**kwargs):
        if kwargs["using"] == "sparse":
            raise UnexpectedResponse(400, "Bad Request", b"sparse boom", {})
        return dense

    client = AsyncMock()
    client.collection_exists.return_value = True
    client.query_points.side_effect = query_points
    mock_factory.return_value = client

    out = await search_corpus("q", "c1", "s1", "gre", top_k=2)

    assert [r["text"] for r in out] == ["A", "B"]
    client.close.assert_awaited_once()


@pytest.mark.parametrize("hybrid", [True, False])
async def test_search_corpus_propagates_schema_rejection_instead_of_empty(hybrid):
    with (
        patch("services.corpus.src.corpus.main.get_settings", return_value=_settings(hybrid)),
        patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock) as embed,
        patch("services.corpus.src.corpus.main.get_qdrant_client") as factory,
    ):
        embed.return_value = [0.1]
        client = AsyncMock()
        client.collection_exists.return_value = True
        client.query_points.side_effect = UnexpectedResponse(
            400, "Bad Request", b"Not existing vector name error: dense", {}
        )
        factory.return_value = client

        with pytest.raises(UnexpectedResponse):
            await search_corpus("q", "c1", "s1", "gre")
        client.close.assert_awaited_once()


@pytest.mark.parametrize(
    "error",
    [
        UnexpectedResponse(503, "Service Unavailable", b"overloaded", {}),
        ResponseHandlingException(httpx.ConnectError("down")),
    ],
)
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_degrades_on_transient_qdrant_failure(mock_embed, mock_factory, error):
    mock_embed.return_value = [0.1]
    client = AsyncMock()
    client.collection_exists.side_effect = error
    mock_factory.return_value = client

    assert await search_corpus("q", "c1", "s1", "gre") == []
    client.close.assert_awaited_once()


@pytest.mark.parametrize("error", [ValueError("bad json"), KeyError("embeddings")])
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_degrades_on_malformed_embedder_response(mock_embed, error):
    mock_embed.side_effect = error
    assert await search_corpus("q", "c1", "s1", "gre") == []


@pytest.mark.parametrize("error", [ValueError("bad input"), KeyError("k")])
@patch("services.corpus.src.corpus.main.get_settings", return_value=_settings(True))
@patch("services.corpus.src.corpus.main.encode_query")
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_search_corpus_propagates_value_and_key_errors_after_embedding(
    mock_embed, mock_factory, mock_encode, _settings, error
):
    mock_embed.return_value = [0.1]
    mock_encode.side_effect = error
    client = AsyncMock()
    client.collection_exists.return_value = True
    mock_factory.return_value = client

    with pytest.raises(type(error)):
        await search_corpus("q", "c1", "s1", "gre")
    client.close.assert_awaited_once()


@patch("services.corpus.src.corpus.main.get_qdrant_client")
async def test_corpus_startup_does_not_swallow_value_errors(mock_factory):
    client = AsyncMock()
    client.collection_exists.side_effect = ValueError("bug")
    mock_factory.return_value = client
    with pytest.raises(ValueError):
        async with lifespan(app):
            pass


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


# ── 9. collection schema guard (legacy single-unnamed-vector collection) ──────


def _collection_info(vectors, sparse_vectors=None):
    params = SimpleNamespace(vectors=vectors, sparse_vectors=sparse_vectors)
    return SimpleNamespace(config=SimpleNamespace(params=params))


_LEGACY = _collection_info(models.VectorParams(size=4, distance=models.Distance.COSINE))
_CURRENT = _collection_info(
    {"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
    {"sparse": models.SparseVectorParams()},
)


async def test_ensure_collection_rejects_legacy_layout_with_reingest_hint():
    client = AsyncMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = _LEGACY

    with pytest.raises(CollectionSchemaError, match="re-ingest"):
        await ensure_collection(client)
    client.create_collection.assert_not_awaited()


async def test_corpus_startup_fails_on_legacy_collection():
    with patch("services.corpus.src.corpus.main.get_qdrant_client") as factory:
        client = AsyncMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = _LEGACY
        factory.return_value = client

        with pytest.raises(CollectionSchemaError):
            async with lifespan(app):
                pass
        client.close.assert_awaited_once()


async def test_corpus_startup_tolerates_missing_collection():
    with patch("services.corpus.src.corpus.main.get_qdrant_client") as factory:
        client = AsyncMock()
        client.collection_exists.return_value = False
        factory.return_value = client
        async with lifespan(app):
            pass


@pytest.mark.parametrize(
    "error",
    [
        ResponseHandlingException(httpx.ConnectError("down")),
        UnexpectedResponse(503, "Service Unavailable", b"overloaded", {}),
    ],
)
async def test_corpus_startup_tolerates_transient_qdrant_failure(error):
    with patch("services.corpus.src.corpus.main.get_qdrant_client") as factory:
        client = AsyncMock()
        client.collection_exists.side_effect = error
        factory.return_value = client
        async with lifespan(app):
            pass
        client.close.assert_awaited_once()


async def test_corpus_startup_fails_on_qdrant_client_error():
    with patch("services.corpus.src.corpus.main.get_qdrant_client") as factory:
        client = AsyncMock()
        client.collection_exists.side_effect = UnexpectedResponse(400, "Bad Request", b"bad", {})
        factory.return_value = client
        with pytest.raises(UnexpectedResponse):
            async with lifespan(app):
                pass


async def test_corpus_startup_accepts_named_layout():
    with patch("services.corpus.src.corpus.main.get_qdrant_client") as factory:
        client = AsyncMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = _CURRENT
        factory.return_value = client
        async with lifespan(app):
            pass
