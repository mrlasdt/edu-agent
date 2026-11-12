"""The filter-recall check's own logic: corpus shape, ground truth, report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from qdrant_client import QdrantClient, models

from evals.qdrant_filter_check import (
    PROFILES,
    QDRANT_IMAGE,
    TEST_TYPES,
    Caller,
    Measurement,
    caller_filter,
    caller_mask,
    exact_top_k,
    format_table,
    make_corpus,
    sample_callers,
)
from shared.src.shared.stats import percentile
from shared.src.shared.vectorstore import DENSE_VECTOR

DIM = 8


def test_corpus_is_deterministic_for_a_seed():
    a = make_corpus(500, DIM, "global-heavy", seed=3)
    b = make_corpus(500, DIM, "global-heavy", seed=3)
    assert np.array_equal(a.vectors, b.vectors)
    assert np.array_equal(a.tier, b.tier)
    assert not np.array_equal(a.tier, make_corpus(500, DIM, "global-heavy", seed=4).tier)


@pytest.mark.parametrize("profile", list(PROFILES))
def test_corpus_follows_the_profile(profile):
    corpus = make_corpus(20_000, DIM, profile, seed=1)
    p = PROFILES[profile]
    for tier, name in enumerate(("global", "school", "candidate")):
        assert abs((corpus.tier == tier).mean() - p[name]) < 0.02
    assert np.allclose(np.linalg.norm(corpus.vectors, axis=1), 1.0, atol=1e-4)
    # owner ids exist exactly where the tier needs them
    assert ((corpus.school >= 0) == (corpus.tier == 1)).all()
    assert ((corpus.candidate >= 0) == (corpus.tier == 2)).all()


def test_callers_own_a_private_chunk_and_some_have_no_school():
    corpus = make_corpus(5_000, DIM, "tenant-heavy", seed=2)
    callers = sample_callers(corpus, 50, seed=2)
    assert len({c.candidate for c in callers}) == len(callers)
    assert all((corpus.candidate == c.candidate).any() for c in callers)
    assert any(c.school == -1 for c in callers)


def test_mask_matches_the_real_filter_in_qdrant_semantics():
    """Plain-numpy ACL == build_search_filter evaluated by the engine (local mode)."""
    corpus = make_corpus(2_000, DIM, "tenant-heavy", seed=5)
    client = QdrantClient(":memory:")
    client.create_collection(
        "c",
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=DIM, distance=models.Distance.COSINE)
        },
    )
    client.upload_collection(
        "c", vectors={DENSE_VECTOR: corpus.vectors}, payload=corpus.payloads(), ids=range(2_000)
    )
    unit = np.ones(DIM, dtype=np.float32)
    for caller in sample_callers(corpus, 20, seed=5):
        mask = caller_mask(corpus, caller)
        hits = client.query_points(
            "c",
            query=unit.tolist(),
            using=DENSE_VECTOR,
            query_filter=caller_filter(caller),
            limit=2_000,
        ).points
        assert {int(p.id) for p in hits} == set(np.flatnonzero(mask).tolist())


def test_caller_without_school_sees_no_school_tier():
    corpus = make_corpus(2_000, DIM, "tenant-heavy", seed=6)
    mask = caller_mask(corpus, Caller(candidate=0, school=-1))
    assert not (mask & (corpus.tier == 1)).any()
    ielts = caller_mask(corpus, Caller(0, 0, test_type=TEST_TYPES[1]))
    assert not (ielts & (corpus.test_type == 0)).any()


def test_exact_top_k_is_the_k_most_similar_among_allowed():
    corpus = make_corpus(300, DIM, "global-heavy", seed=7)
    mask = corpus.tier == 0
    query = corpus.vectors[10]
    got = exact_top_k(corpus, mask, query, 5)
    allowed = np.flatnonzero(mask)
    expected = allowed[np.argsort(-(corpus.vectors[allowed] @ query))[:5]]
    assert got == set(expected.tolist())


def test_percentile_and_table():
    assert percentile([1.0, 2.0, 3.0, 4.0, 100.0], 0.5) == 3.0
    row = Measurement(
        variant="no-index", profile="global-heavy", points=10, indexed_fields=[], queries=2,
        matched_share_median=0.5, matched_share_min=0.4, recall_at_k=1.0, recall_min=1.0,
        rows_returned_mean=10.0, p50_ms=1.234, p95_ms=2.5, build_s=0.1, note="default hnsw_ef",
    )
    table = format_table([row])
    assert "| global-heavy | no-index (default hnsw_ef) | none |" in table
    assert "1.23" in table


def test_the_compose_qdrant_image_is_the_one_the_filter_check_runs():
    compose_file = Path(__file__).resolve().parents[2] / "infra/docker-compose.yml"
    compose = yaml.safe_load(compose_file.read_text())
    assert compose["services"]["qdrant"]["image"] == QDRANT_IMAGE
