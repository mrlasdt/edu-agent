"""
Tests for the cross-encoder rerank step in search_corpus (ADR-0004).

Behaviors under test:
  1. With rerank enabled, the fused top candidates are reordered by reranker score
  2. Only the top RERANK_CANDIDATES fused hits are sent to the reranker
  3. Fail open: a reranker error returns the RRF order and logs a warning
  4. Rerank is off when no reranker URL is configured, and RERANK_ENABLED=false wins
  5. ACL ids never leak from reranked results; exact reranker ties still prefer the
     more specific tier
  6. The TEI /rerank client maps scores back to input order
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from services.corpus.src.corpus.main import search_corpus
from shared.src.shared.config import Settings
from shared.src.shared.reranker import rerank_texts

RERANKER_URL = "http://tei-reranker:8080"


def _settings(**overrides) -> Settings:
    base = {"hybrid_retrieval_enabled": True, "tei_reranker_url": RERANKER_URL}
    return Settings(**{**base, **overrides})


def _point(pid, score, payload):
    p = MagicMock()
    p.id, p.score, p.payload = pid, score, payload
    return p


def _payload(text, tier="global"):
    return {
        "text": text,
        "tier": tier,
        "test_type": "gre",
        "source_uri": "doc.txt",
        "page_or_section": "s1",
        "candidate_id": "secret",
        "school_id": "secret",
    }


def _client(dense_points, sparse_points):
    dense = MagicMock(points=dense_points)
    sparse = MagicMock(points=sparse_points)

    async def query_points(**kwargs):
        return dense if kwargs["using"] == "dense" else sparse

    client = AsyncMock()
    client.collection_exists.return_value = True
    client.query_points.side_effect = query_points
    return client


def _abc_client():
    # RRF order: B (both channels), then A, then C
    return _client(
        [_point("a", 0.9, _payload("A")), _point("b", 0.8, _payload("B"))],
        [_point("b", 5.0, _payload("B")), _point("c", 4.0, _payload("C"))],
    )


@patch("services.corpus.src.corpus.main.rerank_texts", new_callable=AsyncMock)
@patch("services.corpus.src.corpus.main.get_settings")
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_rerank_reorders_fused_candidates(mock_embed, mock_factory, mock_settings, mock_rr):
    mock_embed.return_value = [0.1]
    mock_settings.return_value = _settings()
    mock_factory.return_value = _abc_client()
    # fused order is B, A, C; the cross-encoder prefers C, then A, then B
    mock_rr.return_value = [0.1, 0.5, 0.9]

    out = await search_corpus("q", "c1", "s1", "gre", top_k=2)

    assert [r["text"] for r in out] == ["C", "A"]
    assert [r["score"] for r in out] == [0.9, 0.5]
    assert mock_rr.await_args.args[0] == "q"
    assert mock_rr.await_args.args[1] == ["B", "A", "C"]
    assert "candidate_id" not in out[0]
    assert "school_id" not in out[0]


@patch("services.corpus.src.corpus.main.rerank_texts", new_callable=AsyncMock)
@patch("services.corpus.src.corpus.main.get_settings")
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_rerank_only_sends_top_candidates(mock_embed, mock_factory, mock_settings, mock_rr):
    mock_embed.return_value = [0.1]
    mock_settings.return_value = _settings(rerank_candidates=2)
    mock_factory.return_value = _abc_client()
    mock_rr.return_value = [0.2, 0.8]

    out = await search_corpus("q", "c1", "s1", "gre", top_k=1)

    assert mock_rr.await_args.args[1] == ["B", "A"]
    assert [r["text"] for r in out] == ["A"]


@patch("services.corpus.src.corpus.main.rerank_texts", new_callable=AsyncMock)
@patch("services.corpus.src.corpus.main.get_settings")
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_rerank_candidates_never_below_top_k(
    mock_embed, mock_factory, mock_settings, mock_rr
):
    mock_embed.return_value = [0.1]
    mock_settings.return_value = _settings(rerank_candidates=1)
    mock_factory.return_value = _abc_client()
    mock_rr.return_value = [0.1, 0.9]

    out = await search_corpus("q", "c1", "s1", "gre", top_k=2)

    assert mock_rr.await_args.args[1] == ["B", "A"]
    assert [r["text"] for r in out] == ["A", "B"]


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("reranker down"),
        httpx.HTTPStatusError("boom", request=MagicMock(), response=MagicMock()),
        ValueError("malformed body"),
        KeyError("score"),
    ],
)
@patch("services.corpus.src.corpus.main.rerank_texts", new_callable=AsyncMock)
@patch("services.corpus.src.corpus.main.get_settings")
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_rerank_fails_open_to_rrf_order(
    mock_embed, mock_factory, mock_settings, mock_rr, error, caplog
):
    mock_embed.return_value = [0.1]
    mock_settings.return_value = _settings()
    mock_factory.return_value = _abc_client()
    mock_rr.side_effect = error

    with caplog.at_level(logging.WARNING):
        out = await search_corpus("q", "c1", "s1", "gre", top_k=3)

    assert [r["text"] for r in out] == ["B", "A", "C"]
    assert any("rerank" in r.message and r.levelno == logging.WARNING for r in caplog.records)


@patch("services.corpus.src.corpus.main.rerank_texts", new_callable=AsyncMock)
@patch("services.corpus.src.corpus.main.get_settings")
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_rerank_fails_open_on_wrong_score_count(
    mock_embed, mock_factory, mock_settings, mock_rr
):
    mock_embed.return_value = [0.1]
    mock_settings.return_value = _settings()
    mock_factory.return_value = _abc_client()
    mock_rr.return_value = [0.9]

    out = await search_corpus("q", "c1", "s1", "gre", top_k=3)

    assert [r["text"] for r in out] == ["B", "A", "C"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"tei_reranker_url": ""},
        {"rerank_enabled": False},
    ],
)
@patch("services.corpus.src.corpus.main.rerank_texts", new_callable=AsyncMock)
@patch("services.corpus.src.corpus.main.get_settings")
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_rerank_skipped_when_unconfigured_or_disabled(
    mock_embed, mock_factory, mock_settings, mock_rr, overrides
):
    mock_embed.return_value = [0.1]
    mock_settings.return_value = _settings(**overrides)
    mock_factory.return_value = _abc_client()

    out = await search_corpus("q", "c1", "s1", "gre", top_k=3)

    mock_rr.assert_not_awaited()
    assert [r["text"] for r in out] == ["B", "A", "C"]


def test_rerank_default_follows_reranker_url():
    assert Settings(tei_reranker_url="").rerank_active is False
    assert Settings(tei_reranker_url=RERANKER_URL).rerank_active is True
    assert Settings(tei_reranker_url=RERANKER_URL, rerank_enabled=False).rerank_active is False
    # explicitly enabled but no service configured: nothing to call
    assert Settings(tei_reranker_url="", rerank_enabled=True).rerank_active is False


@patch("services.corpus.src.corpus.main.rerank_texts", new_callable=AsyncMock)
@patch("services.corpus.src.corpus.main.get_settings")
@patch("services.corpus.src.corpus.main.get_qdrant_client")
@patch("services.corpus.src.corpus.main.embed_query", new_callable=AsyncMock)
async def test_rerank_exact_ties_prefer_more_specific_tier(
    mock_embed, mock_factory, mock_settings, mock_rr
):
    mock_embed.return_value = [0.1]
    mock_settings.return_value = _settings()
    mock_factory.return_value = _client(
        [_point("g", 0.9, _payload("G")), _point("k", 0.8, _payload("K", "candidate"))],
        [_point("g", 3.0, _payload("G")), _point("k", 2.0, _payload("K", "candidate"))],
    )
    mock_rr.return_value = [0.7, 0.7]

    out = await search_corpus("q", "c1", "s1", "gre", top_k=2)

    assert [r["text"] for r in out] == ["K", "G"]


# ── TEI /rerank client ────────────────────────────────────────────────────────


@respx.mock
async def test_rerank_texts_maps_scores_back_to_input_order():
    def reply(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["query"] == "q"
        assert body["texts"] == ["t0", "t1", "t2"]
        # TEI returns results sorted by score, each carrying its input index
        return httpx.Response(
            200,
            json=[
                {"index": 2, "score": 0.9},
                {"index": 0, "score": 0.4},
                {"index": 1, "score": 0.1},
            ],
        )

    respx.post("http://tei/rerank").mock(side_effect=reply)
    assert await rerank_texts("q", ["t0", "t1", "t2"], url="http://tei") == [0.4, 0.1, 0.9]


async def test_rerank_texts_empty_input_makes_no_call():
    assert await rerank_texts("q", []) == []


@respx.mock
async def test_rerank_texts_raises_on_http_error():
    respx.post("http://tei/rerank").mock(return_value=httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        await rerank_texts("q", ["t"], url="http://tei")
