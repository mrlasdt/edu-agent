"""
Tenant-isolation proof: the three-tier ACL holds against real Qdrant filter semantics.

`tests/test_corpus_retriever.py` only checks the *shape* of the filter dict. This
module runs the real `build_search_filter` against an in-memory Qdrant
(`AsyncQdrantClient(":memory:")`, the engine's local mode) holding the real
collection layout (`ensure_collection`: named dense vector + named sparse vector
with the IDF modifier), so it tests what the engine does with the filter.

Setup: 3 candidates across 2 schools, global notes, and "orphan" chunks that lack
an owner id. Every chunk carries a unique canary token in its text.

Cases:
  1. 200 random and adversarial queries per candidate, on both the dense and the
     sparse channel, return only chunks that candidate may see: no other
     candidate's chunk, no other school's chunk (and, because the visible set is
     small and `limit` exceeds it, nothing the candidate may see goes missing).
  2. A candidate with an empty `school_id` does not see school-tier chunks that
     lack a school id (null or absent), nor candidate-tier chunks that lack an id.
  3. Mutation test: the same check fails when the candidate or school clause is
     dropped from `build_acl_filter`, so the test can detect the bug it guards.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient, models

from services.corpus.src.corpus import retriever
from services.corpus.src.corpus.retriever import build_search_filter
from shared.src.shared.sparse import encode_document, encode_query
from shared.src.shared.vectorstore import (
    COLLECTION,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    ensure_collection,
)

DIM = 16  # tiny fake embeddings: the filter, not the model, is under test
TEST_TYPE = "gre-quant"
QUERIES_PER_CANDIDATE = 200
LIMIT = 100  # larger than the corpus, so every visible chunk is returned


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    school_id: str


ALICE = Candidate("cand-alice", "school-a")
BOB = Candidate("cand-bob", "school-a")  # same school as Alice, different candidate
CARA = Candidate("cand-cara", "school-b")
CANDIDATES = [ALICE, BOB, CARA]


def _canary(*parts: str) -> str:
    return "canary" + "".join(p.replace("-", "") for p in parts)


def _fake_dense(text: str) -> list[float]:
    """Deterministic pseudo-embedding: a unit-scale vector seeded by the text."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(DIM)]


_ABSENT: Any = object()  # sentinel: the payload key is not written at all


def _chunk(
    canary: str,
    tier: str,
    *,
    school_id: str | None = _ABSENT,
    candidate_id: str | None = _ABSENT,
    test_type: str = TEST_TYPE,
) -> dict[str, Any]:
    """A payload as ingestion writes it; _ABSENT leaves the key out altogether."""
    payload: dict[str, Any] = {
        "text": f"discriminant of a quadratic {canary} nCr factor",
        "tier": tier,
        "test_type": test_type,
        "source_uri": f"{canary}.md",
        "page_or_section": "s",
    }
    if school_id is not _ABSENT:
        payload["school_id"] = school_id
    if candidate_id is not _ABSENT:
        payload["candidate_id"] = candidate_id
    return payload


def _seed_chunks() -> list[dict[str, Any]]:
    chunks = [
        _chunk(_canary("global", "one"), "global"),
        _chunk(_canary("global", "two"), "global"),
        _chunk(_canary("school", "a", "one"), "school", school_id="school-a"),
        _chunk(_canary("school", "a", "two"), "school", school_id="school-a"),
        _chunk(_canary("school", "b", "one"), "school", school_id="school-b"),
        _chunk(_canary("school", "b", "two"), "school", school_id="school-b"),
        # Orphans: tier says school/candidate but the owner id is null or absent.
        _chunk(_canary("orphan", "school", "null"), "school", school_id=None),
        _chunk(_canary("orphan", "school", "absent"), "school"),
        _chunk(_canary("orphan", "school", "empty"), "school", school_id=""),
        _chunk(_canary("orphan", "candidate", "null"), "candidate", candidate_id=None),
        _chunk(_canary("orphan", "candidate", "absent"), "candidate"),
        _chunk(_canary("orphan", "candidate", "empty"), "candidate", candidate_id=""),
        # Another test type: never visible to a gre-quant query at any tier.
        _chunk(_canary("global", "othertest"), "global", test_type="ielts"),
    ]
    for cand in CANDIDATES:
        for n in ("one", "two"):
            chunks.append(
                _chunk(
                    _canary("cand", cand.candidate_id, n),
                    "candidate",
                    candidate_id=cand.candidate_id,
                    school_id=cand.school_id,
                )
            )
    return chunks


def _allowed(payload: dict[str, Any], who: Candidate) -> bool:
    """The ACL spec, written independently of the filter under test."""
    if payload["test_type"] != TEST_TYPE:
        return False
    tier = payload["tier"]
    if tier == "global":
        return True
    if tier == "school":
        return payload.get("school_id") == who.school_id and bool(who.school_id)
    if tier == "candidate":
        return payload.get("candidate_id") == who.candidate_id and bool(who.candidate_id)
    return False


def _canary_of(payload: dict[str, Any]) -> str:
    return str(payload["source_uri"]).removesuffix(".md")


@pytest.fixture
async def store() -> AsyncIterator[tuple[AsyncQdrantClient, list[dict[str, Any]]]]:
    client = AsyncQdrantClient(":memory:")
    await ensure_collection(client, dim=DIM)
    chunks = _seed_chunks()
    await client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=i,
                vector={
                    DENSE_VECTOR: _fake_dense(c["text"]),
                    SPARSE_VECTOR: encode_document(c["text"]),
                },
                payload=c,
            )
            for i, c in enumerate(chunks, start=1)
        ],
        wait=True,
    )
    try:
        yield client, chunks
    finally:
        await client.close()


def _queries(who: Candidate, chunks: list[dict[str, Any]], n: int, seed: int) -> list[str]:
    """Random text plus adversarial queries that name other tenants' canaries and ids."""
    rng = random.Random(seed)
    canaries = [_canary_of(c) for c in chunks]
    foreign = [_canary_of(c) for c in chunks if not _allowed(c, who)]
    ids = [c.candidate_id for c in CANDIDATES] + [c.school_id for c in CANDIDATES]
    vocab = [
        "discriminant", "quadratic", "nCr", "factor", "ratio", "median", "probability",
        "school", "candidate", "tier", "global", "everything", "other", "student",
    ]  # fmt: skip
    queries: list[str] = []
    for i in range(n):
        kind = i % 5
        if kind == 0:  # random vocabulary
            q = " ".join(rng.choices(vocab, k=rng.randint(1, 6)))
        elif kind == 1:  # the exact canary of a chunk this candidate must not see
            q = rng.choice(foreign)
        elif kind == 2:  # every canary at once, so lexical search matches the whole corpus
            q = " ".join(canaries)
        elif kind == 3:  # another tenant's ids, plus filter-injection looking text
            q = f'{rng.choice(ids)} {rng.choice(ids)} tier:school" OR "1"="1 {rng.choice(foreign)}'
        else:  # empty and whitespace-only
            q = rng.choice(["", " ", "   \n"])
        queries.append(q)
    return queries


async def _leaks(
    client: AsyncQdrantClient, chunks: list[dict[str, Any]], who: Candidate, *, school_id: str
) -> list[str]:
    """Run every query on both channels; return human-readable violations."""
    search_filter = build_search_filter(who.candidate_id, school_id, TEST_TYPE)
    visible = {_canary_of(c) for c in chunks if _allowed(c, who)}
    problems: list[str] = []
    seed = int.from_bytes(hashlib.sha256(who.candidate_id.encode()).digest()[:4], "big")
    for query in _queries(who, chunks, QUERIES_PER_CANDIDATE, seed):
        channels: dict[str, Any] = {
            DENSE_VECTOR: _fake_dense(query),
            SPARSE_VECTOR: encode_query(query),
        }
        for channel, vector in channels.items():
            if channel == SPARSE_VECTOR and not vector.indices:
                continue  # an empty sparse query is not a valid search
            response = await client.query_points(
                collection_name=COLLECTION,
                query=vector,
                using=channel,
                query_filter=search_filter,
                limit=LIMIT,
                with_payload=True,
            )
            got = {_canary_of(p.payload or {}) for p in response.points}
            for point in response.points:
                payload = point.payload or {}
                if not _allowed(payload, who):
                    problems.append(
                        f"{who.candidate_id} {channel} q={query!r}: leaked {_canary_of(payload)}"
                    )
            if channel == DENSE_VECTOR and got != visible:
                # Dense search ranks every point, so with limit > corpus it must return
                # exactly the visible set: also catches a filter that over-blocks.
                problems.append(
                    f"{who.candidate_id} dense q={query!r}: visible set mismatch "
                    f"(missing={sorted(visible - got)}, extra={sorted(got - visible)})"
                )
    return problems


@pytest.mark.parametrize("who", CANDIDATES, ids=lambda c: c.candidate_id)
async def test_no_foreign_chunks_for_random_and_adversarial_queries(
    store: tuple[AsyncQdrantClient, list[dict[str, Any]]], who: Candidate
) -> None:
    client, chunks = store
    problems = await _leaks(client, chunks, who, school_id=who.school_id)
    assert problems == []


@pytest.mark.parametrize("who", CANDIDATES, ids=lambda c: c.candidate_id)
async def test_empty_school_id_sees_no_school_tier_chunks(
    store: tuple[AsyncQdrantClient, list[dict[str, Any]]], who: Candidate
) -> None:
    """The agents call retrieval with school_id="" today (quant, aw): that must see no school."""
    client, chunks = store
    no_school = Candidate(who.candidate_id, "")
    problems = await _leaks(client, chunks, no_school, school_id="")
    assert problems == []

    response = await client.query_points(
        collection_name=COLLECTION,
        query=_fake_dense("discriminant"),
        using=DENSE_VECTOR,
        query_filter=build_search_filter(who.candidate_id, "", TEST_TYPE),
        limit=LIMIT,
        with_payload=True,
    )
    tiers = {(p.payload or {})["tier"] for p in response.points}
    assert tiers == {"global", "candidate"}  # global + own notes; no school-tier chunk at all


async def test_empty_candidate_id_sees_no_candidate_tier_chunks(
    store: tuple[AsyncQdrantClient, list[dict[str, Any]]],
) -> None:
    client, chunks = store
    nobody = Candidate("", ALICE.school_id)
    assert await _leaks(client, chunks, nobody, school_id=nobody.school_id) == []
    response = await client.query_points(
        collection_name=COLLECTION,
        query=_fake_dense("discriminant"),
        using=DENSE_VECTOR,
        query_filter=build_search_filter("", ALICE.school_id, TEST_TYPE),
        limit=LIMIT,
        with_payload=True,
    )
    assert {(p.payload or {})["tier"] for p in response.points} == {"global", "school"}


# ── mutation: the test must be able to see the bug it guards ─────────────────


def _drop_clause(tier: str) -> Any:
    real = retriever.build_acl_filter

    def mutated(candidate_id: str, school_id: str) -> dict[str, Any]:
        acl = real(candidate_id, school_id)
        acl["should"] = [
            clause
            for clause in acl["should"]
            if tier not in {m["match"]["value"] for m in clause.get("must", [])}
        ]
        return acl

    return mutated


@pytest.mark.parametrize("dropped_tier", ["candidate", "school"])
async def test_mutation_dropping_an_acl_clause_is_detected(
    store: tuple[AsyncQdrantClient, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    dropped_tier: str,
) -> None:
    """Drop the candidate (or school) clause: the isolation check must report a problem.

    Dropping a clause makes the filter stricter, not looser, so the proof here is the
    "visible set mismatch" arm (the engine no longer returns what the ACL allows).
    The leak arm is proven by the loosened mutation below.
    """
    client, chunks = store
    monkeypatch.setattr(retriever, "build_acl_filter", _drop_clause(dropped_tier))
    problems = await _leaks(client, chunks, ALICE, school_id=ALICE.school_id)
    assert problems, f"dropping the {dropped_tier} clause went undetected"


async def test_mutation_unscoped_candidate_clause_leaks_and_is_detected(
    store: tuple[AsyncQdrantClient, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The realistic bug: the candidate clause loses its `candidate_id` condition."""
    real = retriever.build_acl_filter

    def mutated(candidate_id: str, school_id: str) -> dict[str, Any]:
        acl = real(candidate_id, school_id)
        for clause in acl["should"]:
            clause_must = clause.get("must")
            if clause_must and clause_must[0]["match"]["value"] == "candidate":
                clause["must"] = clause_must[:1]  # tier=candidate, no candidate_id check
        return acl

    monkeypatch.setattr(retriever, "build_acl_filter", mutated)
    client, chunks = store
    problems = await _leaks(client, chunks, ALICE, school_id=ALICE.school_id)
    leaked = [p for p in problems if "leaked" in p]
    assert leaked, "an unscoped candidate clause leaked chunks but the test did not notice"
    assert any("cand-bob" in p or "cand-cara" in p for p in leaked)
