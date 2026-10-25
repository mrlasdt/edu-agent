"""Tests for the local lexical sparse encoder and the sparse-aware collection config."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from qdrant_client import models

from shared.src.shared.sparse import encode_document, encode_query
from shared.src.shared.vectorstore import ensure_collection


def test_document_and_query_share_indices_for_same_term():
    doc = encode_document("The discriminant of a quadratic")
    query = encode_query("discriminant")
    assert set(query.indices) <= set(doc.indices)


def test_encoding_is_case_insensitive_and_deterministic():
    assert encode_query("Quadratic") == encode_query("quadratic")
    assert encode_document("a b a") == encode_document("a b a")


def test_repeated_terms_get_sublinear_weight_and_indices_are_unique():
    vec = encode_document("x x x x y")
    assert len(vec.indices) == len(set(vec.indices)) == 2
    assert vec.indices == sorted(vec.indices)
    assert max(vec.values) < 4.0  # damped, not raw tf
    assert min(vec.values) == 1.0


def test_empty_text_yields_empty_vector():
    assert encode_query("").indices == []
    assert encode_document("  ").values == []


async def test_ensure_collection_creates_named_dense_and_idf_sparse_vectors():
    client = AsyncMock()
    client.collection_exists.return_value = False

    await ensure_collection(client, dim=8)

    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["vectors_config"]["dense"].size == 8
    assert kwargs["sparse_vectors_config"]["sparse"].modifier == models.Modifier.IDF


async def test_ensure_collection_is_idempotent():
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=True)
    client.create_collection = AsyncMock()
    params = SimpleNamespace(
        vectors={"dense": models.VectorParams(size=8, distance=models.Distance.COSINE)},
        sparse_vectors={"sparse": models.SparseVectorParams()},
    )
    info = SimpleNamespace(config=SimpleNamespace(params=params))
    client.get_collection = AsyncMock(return_value=info)
    await ensure_collection(client, dim=8)
    client.create_collection.assert_not_awaited()
