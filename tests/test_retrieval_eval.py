"""
Tests for the retrieval eval's data and arithmetic (the live run needs Docker and the embedder).

Behaviors under test:
  1. Every expected note in gre_retrieval.jsonl exists in the seed notes, as the chunker labels it
  2. The query set has all three groups; off-topic queries expect nothing, the others expect a note
  3. recall@3, MRR@3 and the off-topic "notes shown" rate are computed from ranked hits
  4. A floor drops hits by the corpus rule (cosine floor for all hits, reranker floor on top for
     reranked ones) and the sweep reports what each floor costs on-topic and buys off-topic
"""
from collections import Counter

import pytest

from evals.retrieval_eval import (
    Hit,
    Query,
    QueryRun,
    best_floor,
    first_rank,
    load_queries,
    quality,
    seed_labels,
    sweep,
)
from shared.src.shared.config import anchor


def test_every_expected_note_exists_in_the_seed_notes():
    labels = seed_labels(anchor("seed/gre-quant"))
    missing = [(q.id, e) for q in load_queries() for e in q.expect if e not in labels]
    assert missing == []


def test_query_set_groups_and_expectations():
    queries = load_queries()
    assert len({q.id for q in queries}) == len(queries)
    counts = Counter(q.group for q in queries)
    assert set(counts) == {"exact", "paraphrase", "off_topic"}
    assert all(n >= 10 for n in counts.values())
    assert all(not q.expect for q in queries if q.group == "off_topic")
    assert all(q.expect for q in queries if q.group != "off_topic")


def _hit(label, score, dense=None, reranked=False):
    return Hit(label, score, dense if dense is not None else score, None, reranked)


def _run(group, expect, hits):
    return QueryRun(Query("q", group, "text", tuple(expect)), tuple(hits), 1.0)


def test_first_rank():
    assert first_rank(["a"], ["x", "a", "y"]) == 2
    assert first_rank(["a", "b"], ["b", "a"]) == 1
    assert first_rank(["a"], ["x"]) is None


def test_quality_recall_mrr_and_notes_shown():
    runs = [
        _run("exact", ["a"], [_hit("a", 0.9)]),
        _run("exact", ["b"], [_hit("x", 0.8), _hit("b", 0.7)]),
        _run("exact", ["c"], [_hit("x", 0.8)]),
        _run("off_topic", [], [_hit("x", 0.4)]),
        _run("off_topic", [], []),
    ]
    q = quality(runs)
    assert q["exact"]["recall_at_3"] == pytest.approx(2 / 3)
    assert q["exact"]["mrr_at_3"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert q["off_topic"]["notes_shown"] == 0.5


def test_cosine_sweep_reads_the_cosine_not_the_fused_score():
    runs = [
        _run("exact", ["a"], [_hit("a", 0.03, dense=0.6)]),
        _run("off_topic", [], [_hit("x", 0.03, dense=0.45)]),
    ]
    rows = {r["floor"]: r for r in sweep(runs, [0.3, 0.5, 0.7])}
    assert rows[0.3]["on_topic_recall_at_3"] == 1.0
    assert rows[0.3]["off_topic_notes_shown"] == 1.0
    assert rows[0.5]["on_topic_recall_at_3"] == 1.0
    assert rows[0.5]["off_topic_notes_shown"] == 0.0
    assert rows[0.7]["on_topic_recall_at_3"] == 0.0


def test_rerank_sweep_applies_the_rerank_floor_on_top_of_the_cosine_floor():
    runs = [
        _run("exact", ["a"], [_hit("a", 0.6, dense=0.9, reranked=True)]),
        _run("paraphrase", ["b"], [_hit("b", 0.2, dense=0.9, reranked=True)]),
        _run("exact", ["c"], [_hit("c", 0.9, dense=0.30, reranked=True)]),  # production drops it
        _run("off_topic", [], [_hit("x", 0.9, dense=0.30, reranked=True)]),
    ]
    rows = {r["floor"]: r for r in sweep(runs, [0.0, 0.5], rerank=True)}
    assert rows[0.0]["on_topic_recall_at_3"] == pytest.approx(2 / 3)
    assert rows[0.0]["off_topic_notes_shown"] == 0.0
    assert rows[0.5]["on_topic_recall_at_3"] == pytest.approx(1 / 3)


def test_best_floor_keeps_most_recall_among_floors_quiet_enough():
    rows = [
        {"floor": 0.3, "on_topic_recall_at_3": 1.0, "off_topic_notes_shown": 1.0},
        {"floor": 0.4, "on_topic_recall_at_3": 0.9, "off_topic_notes_shown": 0.08},
        {"floor": 0.5, "on_topic_recall_at_3": 0.9, "off_topic_notes_shown": 0.0},
        {"floor": 0.6, "on_topic_recall_at_3": 0.4, "off_topic_notes_shown": 0.0},
    ]
    assert best_floor(rows)["floor"] == 0.4  # ties on recall go to the lower floor
    assert best_floor(rows[:1])["floor"] == 0.3  # none quiet enough: the highest floor there is
