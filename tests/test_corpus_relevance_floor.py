"""
Tests for the relevance floor on corpus retrieval.

Behaviors under test:
  1. Without apply_relevance_floor nothing changes: every top_k hit comes back
  2. With it, dense-only and hybrid retrieval drop hits whose cosine similarity is below
     the configured floor, so an off-topic message gets no notes at all
  3. A fused RRF score is never the floor's signal: a rank-1 hit with a low cosine is dropped
  4. A reranked hit must clear both the reranker floor and the cosine floor; an off-topic
     query with the reranker on still gets no notes while the rerank floor is unset
  5. A hit only the sparse channel found has no cosine and does not pass
  6. ACL ids stay out of floored results, and the HTTP request defaults the flag off
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.corpus.src.corpus.main import (
    RetrieveRequest,
    passes_relevance_floor,
    search_candidates,
    search_corpus,
)
from services.corpus.src.corpus.settings import CorpusSettings

RERANKER_URL = "http://tei-reranker:8080"


def _settings(**overrides) -> CorpusSettings:
    base = {
        "hybrid_retrieval_enabled": True,
        "dense_relevance_floor": 0.5,
        "rerank_relevance_floor": 0.4,
    }
    return CorpusSettings(**{**base, **overrides})


def _point(pid, score, text):
    p = MagicMock()
    p.id, p.score = pid, score
    p.payload = {
        "text": text,
        "tier": "global",
        "test_type": "gre",
        "source_uri": "doc.txt",
        "page_or_section": text,
        "candidate_id": "secret",
        "school_id": "secret",
    }
    return p


def _client(dense_points, sparse_points):
    dense, sparse = MagicMock(points=dense_points), MagicMock(points=sparse_points)

    async def query_points(**kwargs):
        return dense if kwargs["using"] == "dense" else sparse

    client = AsyncMock()
    client.collection_exists.return_value = True
    client.query_points.side_effect = query_points
    return client


async def _search(settings, client, *, floor, top_k=3, reranker=None):
    """`reranker` is the scores the fake cross-encoder returns, or an exception it raises."""
    failing = isinstance(reranker, Exception)
    with (
        patch("services.corpus.src.corpus.main.get_settings", return_value=settings),
        patch("services.corpus.src.corpus.main.get_qdrant_client", return_value=client),
        patch(
            "services.corpus.src.corpus.main.embed_query",
            new_callable=AsyncMock,
            return_value=[0.1],
        ),
        patch(
            "services.corpus.src.corpus.main.rerank_texts",
            new_callable=AsyncMock,
            side_effect=reranker if failing else None,
            return_value=None if failing else reranker,
        ),
    ):
        return await search_corpus("q", "c1", "s1", "gre", top_k, apply_relevance_floor=floor)


async def test_floor_is_off_unless_requested():
    client = _client([_point("a", 0.2, "A"), _point("b", 0.1, "B")], [])
    out = await _search(_settings(), client, floor=False)
    assert [r["text"] for r in out] == ["A", "B"]


async def test_dense_only_floor_drops_hits_below_the_cosine_floor():
    client = _client([_point("a", 0.62, "A"), _point("b", 0.48, "B")], [])
    out = await _search(_settings(hybrid_retrieval_enabled=False), client, floor=True)
    assert [r["text"] for r in out] == ["A"]


async def test_off_topic_message_gets_no_notes():
    client = _client([_point("a", 0.41, "A"), _point("b", 0.39, "B")], [])
    out = await _search(_settings(), client, floor=True)
    assert out == []


async def test_a_rank_one_fused_hit_with_a_low_cosine_is_still_dropped():
    # A is first in both channels, so its RRF score is the best possible, yet its cosine is
    # low: the floor must read the cosine, not the rank-derived fused score.
    client = _client(
        [_point("a", 0.44, "A"), _point("b", 0.43, "B")],
        [_point("a", 9.0, "A"), _point("b", 5.0, "B")],
    )
    out = await _search(_settings(), client, floor=True)
    assert out == []


async def test_hybrid_floor_keeps_relevant_hits_in_fused_order():
    client = _client(
        [_point("a", 0.7, "A"), _point("b", 0.55, "B"), _point("c", 0.3, "C")],
        [_point("b", 5.0, "B"), _point("a", 4.0, "A"), _point("c", 3.0, "C")],
    )
    out = await _search(_settings(), client, floor=True)
    assert [r["text"] for r in out] == ["A", "B"]
    assert all("candidate_id" not in r and "school_id" not in r for r in out)


async def test_sparse_only_hit_has_no_cosine_and_does_not_pass():
    client = _client([_point("a", 0.7, "A")], [_point("z", 9.0, "Z"), _point("a", 1.0, "A")])
    out = await _search(_settings(), client, floor=True)
    assert [r["text"] for r in out] == ["A"]


async def test_floor_reads_the_reranker_score_after_a_rerank():
    client = _client([_point("a", 0.9, "A"), _point("b", 0.8, "B")], [])
    settings = _settings(tei_reranker_url=RERANKER_URL)
    # high cosines, but the cross-encoder says B is irrelevant
    out = await _search(settings, client, floor=True, reranker=[0.9, 0.1])
    assert [r["text"] for r in out] == ["A"]
    assert out[0]["score"] == 0.9


async def test_off_topic_message_gets_no_notes_with_the_reranker_on():
    client = _client([_point("a", 0.41, "A"), _point("b", 0.39, "B")], [])
    settings = _settings(tei_reranker_url=RERANKER_URL, rerank_relevance_floor=0.0)
    out = await _search(settings, client, floor=True, reranker=[0.8, 0.7])
    assert out == []


async def test_failed_rerank_falls_back_to_the_cosine_floor():
    client = _client([_point("a", 0.9, "A"), _point("b", 0.3, "B")], [])
    settings = _settings(tei_reranker_url=RERANKER_URL)
    out = await _search(settings, client, floor=True, reranker=ValueError("bad body"))
    assert [r["text"] for r in out] == ["A"]


@pytest.mark.parametrize(
    ("hit", "expected"),
    [
        ({"score": 0.03, "dense_score": 0.5}, True),
        ({"score": 0.03, "dense_score": 0.49}, False),
        ({"score": 0.03, "dense_score": None}, False),
        ({"score": 0.4, "reranked": True, "dense_score": 0.5}, True),
        ({"score": 0.39, "reranked": True, "dense_score": 0.9}, False),
        ({"score": 0.9, "reranked": True, "dense_score": 0.1}, False),
        ({"score": 0.9, "reranked": True, "dense_score": None}, False),
    ],
)
def test_passes_relevance_floor_signal_choice(hit, expected):
    assert passes_relevance_floor(hit, _settings()) is expected


def test_request_defaults_the_floor_off():
    request = RetrieveRequest(query="essay", candidate_id="c", school_id="", test_type="gre")
    assert request.apply_relevance_floor is False


async def test_the_injected_settings_decide_the_reranker_url():
    client = _client([_point("a", 0.9, "A"), _point("b", 0.8, "B")], [])
    rerank = AsyncMock(return_value=[0.2, 0.9])
    with (
        patch(
            "services.corpus.src.corpus.main.get_settings",
            return_value=_settings(tei_reranker_url=""),
        ),
        patch("services.corpus.src.corpus.main.get_qdrant_client", return_value=client),
        patch(
            "services.corpus.src.corpus.main.embed_query",
            new_callable=AsyncMock,
            return_value=[0.1],
        ),
        patch("services.corpus.src.corpus.main.rerank_texts", rerank),
    ):
        ranked = await search_candidates(
            "q", "c1", "s1", "gre", 3, settings=_settings(tei_reranker_url=RERANKER_URL)
        )
    assert rerank.await_args.kwargs["url"] == RERANKER_URL
    assert [h["id"] for h in ranked] == ["b", "a"]
    assert all(h["reranked"] for h in ranked)
