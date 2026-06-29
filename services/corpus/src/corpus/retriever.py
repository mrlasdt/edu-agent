from __future__ import annotations

from typing import Any

from qdrant_client import models

# Tier priority for tiebreak: higher = more specific = preferred
TIER_PRIORITY = {"candidate": 3, "school": 2, "global": 1}

RRF_K = 60  # Standard RRF constant


def _rrf_merge(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.

    Each result dict must have: {"id": str, "score": float, "payload": dict}
    Returns a merged list sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        payloads[doc_id] = result["payload"]

    for rank, result in enumerate(sparse_results):
        doc_id = result["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        if doc_id not in payloads:
            payloads[doc_id] = result["payload"]

    return [
        {"id": doc_id, "score": score, "payload": payloads[doc_id]}
        for doc_id, score in sorted(scores.items(), key=lambda x: -x[1])
    ]


def _tier_tiebreak(
    results: list[dict[str, Any]],
    threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """
    When two adjacent results have scores within `threshold` (relative), prefer
    the more specific tier: candidate > school > global.

    Uses a single pass of adjacent swaps — sufficient for the common case.
    """
    if len(results) <= 1:
        return results

    result = list(results)
    for i in range(len(result) - 1):
        a_score = result[i]["score"]
        b_score = result[i + 1]["score"]
        if a_score == 0:
            continue
        if abs(a_score - b_score) / a_score > threshold:
            continue
        a_tier = TIER_PRIORITY.get(result[i]["payload"].get("tier", "global"), 1)
        b_tier = TIER_PRIORITY.get(result[i + 1]["payload"].get("tier", "global"), 1)
        if b_tier > a_tier:
            result[i], result[i + 1] = result[i + 1], result[i]

    return result


def _build_acl_filter(candidate_id: str, school_id: str) -> dict[str, Any]:
    """
    Build a Qdrant payload filter enforcing the three-tier ACL:
      tier=global  OR  (tier=school AND school_id=:s)  OR  (tier=candidate AND candidate_id=:u)
    """
    return {
        "should": [
            {"key": "tier", "match": {"value": "global"}},
            {
                "must": [
                    {"key": "tier", "match": {"value": "school"}},
                    {"key": "school_id", "match": {"value": school_id}},
                ]
            },
            {
                "must": [
                    {"key": "tier", "match": {"value": "candidate"}},
                    {"key": "candidate_id", "match": {"value": candidate_id}},
                ]
            },
        ]
    }


def build_search_filter(candidate_id: str, school_id: str, test_type: str) -> models.Filter:
    """
    Realise the logical filter for a query as a Qdrant models.Filter:
        test_type = :t  AND  (<three-tier ACL>)

    The ACL branch reuses _build_acl_filter (the unit-tested spec), nested under
    a top-level `must` alongside the test_type match so a candidate can never
    retrieve another candidate's — or the wrong test's — chunks.
    """
    acl = _build_acl_filter(candidate_id=candidate_id, school_id=school_id)
    return models.Filter.model_validate(
        {
            "must": [
                {"key": "test_type", "match": {"value": test_type}},
                {"should": acl["should"]},
            ]
        }
    )
