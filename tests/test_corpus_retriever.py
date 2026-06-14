"""
Tests for the Corpus retrieval service.

Behaviors under test (from issue 07 acceptance criteria):
  1. RRF merge: two ranked lists → correct RRF-scored merged list
  2. ACL filter construction: builds correct three-tier Qdrant filter
  3. Tier-priority tiebreak: within-5%-threshold, prefer Student > School > Global
  4. POST /corpus/retrieve → chunks with required fields (text, tier, source_uri, page_or_section, score)
  5. Candidate cannot retrieve another candidate's chunks (ACL)
  6. test_type filter applied
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from services.corpus.src.corpus.retriever import (
    _build_acl_filter,
    _rrf_merge,
    _tier_tiebreak,
)
from services.corpus.src.corpus.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ── 1. RRF merge ──────────────────────────────────────────────────────────────

def test_rrf_merge_combines_two_lists():
    dense = [
        {"id": "a", "score": 0.9, "payload": {"text": "A", "tier": "global"}},
        {"id": "b", "score": 0.8, "payload": {"text": "B", "tier": "global"}},
    ]
    sparse = [
        {"id": "b", "score": 0.95, "payload": {"text": "B", "tier": "global"}},
        {"id": "c", "score": 0.7, "payload": {"text": "C", "tier": "global"}},
    ]
    merged = _rrf_merge(dense, sparse)
    ids = [r["id"] for r in merged]
    # "b" appears in both lists → should rank highest
    assert ids[0] == "b"


def test_rrf_merge_unique_ids_only():
    dense = [{"id": "x", "score": 0.9, "payload": {}}]
    sparse = [{"id": "x", "score": 0.8, "payload": {}}]
    merged = _rrf_merge(dense, sparse)
    assert len(merged) == 1


def test_rrf_merge_empty_inputs():
    assert _rrf_merge([], []) == []


def test_rrf_merge_single_list():
    dense = [
        {"id": "a", "score": 0.9, "payload": {"text": "A"}},
        {"id": "b", "score": 0.5, "payload": {"text": "B"}},
    ]
    merged = _rrf_merge(dense, [])
    assert [r["id"] for r in merged] == ["a", "b"]


# ── 2. ACL filter ─────────────────────────────────────────────────────────────

def test_acl_filter_includes_global_tier():
    f = _build_acl_filter(candidate_id="c1", school_id="s1")
    conditions = f["should"]
    global_cond = next(
        (c for c in conditions if c.get("key") == "tier" and c["match"]["value"] == "global"),
        None,
    )
    assert global_cond is not None


def test_acl_filter_includes_school_tier_with_school_id():
    f = _build_acl_filter(candidate_id="c1", school_id="s1")
    conditions = f["should"]
    school_cond = next(
        (
            c for c in conditions
            if isinstance(c, dict) and "must" in c
            and any(m.get("match", {}).get("value") == "school" for m in c["must"])
        ),
        None,
    )
    assert school_cond is not None
    school_id_cond = next(
        m for m in school_cond["must"] if m.get("key") == "school_id"
    )
    assert school_id_cond["match"]["value"] == "s1"


def test_acl_filter_includes_candidate_tier_with_candidate_id():
    f = _build_acl_filter(candidate_id="c1", school_id="s1")
    conditions = f["should"]
    cand_cond = next(
        (
            c for c in conditions
            if isinstance(c, dict) and "must" in c
            and any(m.get("match", {}).get("value") == "candidate" for m in c["must"])
        ),
        None,
    )
    assert cand_cond is not None
    cand_id_cond = next(m for m in cand_cond["must"] if m.get("key") == "candidate_id")
    assert cand_id_cond["match"]["value"] == "c1"


# ── 3. Tier-priority tiebreak ─────────────────────────────────────────────────

def make_result(doc_id, score, tier):
    return {"id": doc_id, "score": score, "payload": {"tier": tier}}


def test_tiebreak_promotes_candidate_over_global_when_close():
    results = [
        make_result("global-doc", 1.000, "global"),
        make_result("cand-doc", 0.998, "candidate"),  # within 5%
    ]
    reranked = _tier_tiebreak(results, threshold=0.05)
    assert reranked[0]["id"] == "cand-doc"


def test_tiebreak_no_swap_when_scores_differ_significantly():
    results = [
        make_result("global-doc", 1.000, "global"),
        make_result("cand-doc", 0.50, "candidate"),  # >5% gap
    ]
    reranked = _tier_tiebreak(results, threshold=0.05)
    assert reranked[0]["id"] == "global-doc"


def test_tiebreak_preserves_order_when_same_tier():
    results = [
        make_result("doc-a", 1.000, "global"),
        make_result("doc-b", 0.999, "global"),
    ]
    reranked = _tier_tiebreak(results, threshold=0.05)
    assert reranked[0]["id"] == "doc-a"


def test_tiebreak_single_result_unchanged():
    results = [make_result("a", 1.0, "global")]
    assert _tier_tiebreak(results) == results


# ── 4. HTTP endpoint ──────────────────────────────────────────────────────────

def test_retrieve_endpoint_returns_chunks_with_required_fields(client):
    mock_chunks = [
        {
            "text": "GRE prep content",
            "tier": "global",
            "source_uri": "ets-guide.pdf",
            "page_or_section": "Introduction",
            "score": 0.92,
        }
    ]
    with patch("services.corpus.src.corpus.main.search_corpus", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_chunks
        response = client.post(
            "/corpus/retrieve",
            json={
                "query": "quadratic equations",
                "candidate_id": "c1",
                "school_id": "s1",
                "test_type": "gre",
                "top_k": 5,
            },
        )

    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) == 1
    chunk = chunks[0]
    assert "text" in chunk
    assert "tier" in chunk
    assert "source_uri" in chunk
    assert "page_or_section" in chunk
    assert "score" in chunk


def test_retrieve_passes_test_type_to_search(client):
    with patch("services.corpus.src.corpus.main.search_corpus", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        client.post(
            "/corpus/retrieve",
            json={
                "query": "quadratic equations",
                "candidate_id": "c1",
                "school_id": "s1",
                "test_type": "gre",
                "top_k": 5,
            },
        )
    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs.get("test_type") == "gre"


def test_retrieve_passes_acl_ids_to_search(client):
    with patch("services.corpus.src.corpus.main.search_corpus", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        client.post(
            "/corpus/retrieve",
            json={
                "query": "test",
                "candidate_id": "cand-99",
                "school_id": "school-42",
                "test_type": "gre",
                "top_k": 5,
            },
        )
    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs.get("candidate_id") == "cand-99"
    assert call_kwargs.get("school_id") == "school-42"
