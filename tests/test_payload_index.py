"""
Payload indexes on the access-filter fields.

Without a keyword index Qdrant reads each point's payload to evaluate the tenant
filter, so latency grows with the collection (measured by
`python -m evals.qdrant_filter_check`). `ensure_collection` creates the indexes,
idempotently, for new and for already-existing collections.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from qdrant_client import models

from services.corpus.src.corpus.main import app, lifespan
from services.corpus.src.corpus.retriever import build_search_filter
from shared.src.shared.vectorstore import (
    COLLECTION,
    DENSE_VECTOR,
    PAYLOAD_INDEX_FIELDS,
    SPARSE_VECTOR,
    ensure_collection,
    ensure_payload_indexes,
)


class FakeQdrant:
    """
    Just enough of AsyncQdrantClient to track collections and payload indexes.
    Qdrant's in-memory mode ignores payload indexes ("no effect in the local
    Qdrant"), so it cannot show whether one was created.
    """

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, models.PayloadSchemaType]] = {}
        self.index_calls: list[str] = []

    async def collection_exists(self, name: str) -> bool:
        return name in self.collections

    async def create_collection(self, collection_name: str, **_: object) -> None:
        self.collections[collection_name] = {}

    async def get_collection(self, name: str) -> SimpleNamespace:
        params = SimpleNamespace(
            vectors={DENSE_VECTOR: object()}, sparse_vectors={SPARSE_VECTOR: object()}
        )
        schema = {f: SimpleNamespace(data_type=t) for f, t in self.collections[name].items()}
        return SimpleNamespace(config=SimpleNamespace(params=params), payload_schema=schema)

    async def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: models.PayloadSchemaType
    ) -> None:
        self.index_calls.append(field_name)
        self.collections[collection_name][field_name] = field_schema

    async def delete_payload_index(self, collection_name: str, field_name: str) -> None:
        del self.collections[collection_name][field_name]


@pytest.fixture
def client() -> FakeQdrant:
    return FakeQdrant()


def _indexes(client: FakeQdrant) -> dict[str, models.PayloadSchemaType]:
    return dict(client.collections[COLLECTION])


def _filter_keys(node: object) -> set[str]:
    """Every payload key a Qdrant filter matches on, however deeply nested."""
    if isinstance(node, models.FieldCondition):
        return {node.key}
    if isinstance(node, models.Filter):
        keys: set[str] = set()
        for group in (node.must, node.should, node.must_not):
            conditions = group if isinstance(group, list) else [group] if group else []
            for condition in conditions:
                keys |= _filter_keys(condition)
        return keys
    raise AssertionError(f"unexpected condition type {type(node)}")


def test_indexed_fields_cover_every_field_the_access_filter_uses():
    used = _filter_keys(build_search_filter("c1", "s1", "gre"))
    assert used == set(PAYLOAD_INDEX_FIELDS)


async def test_new_collection_gets_a_keyword_index_per_filter_field(client):
    await ensure_collection(client, dim=4)  # type: ignore[arg-type]

    assert _indexes(client) == {
        field: models.PayloadSchemaType.KEYWORD for field in PAYLOAD_INDEX_FIELDS
    }


async def test_existing_collection_without_indexes_gains_them(client):
    await client.create_collection(COLLECTION)  # as created before the indexes existed
    assert _indexes(client) == {}

    await ensure_collection(client, dim=4)  # type: ignore[arg-type]

    assert set(_indexes(client)) == set(PAYLOAD_INDEX_FIELDS)


async def test_ensure_collection_is_idempotent_for_indexes(client):
    await ensure_collection(client, dim=4)  # type: ignore[arg-type]
    client.index_calls.clear()

    await ensure_collection(client, dim=4)  # type: ignore[arg-type]
    await ensure_payload_indexes(client)  # type: ignore[arg-type]

    assert client.index_calls == []
    assert set(_indexes(client)) == set(PAYLOAD_INDEX_FIELDS)


async def test_only_missing_indexes_are_created(client):
    await ensure_collection(client, dim=4)  # type: ignore[arg-type]
    await client.delete_payload_index(COLLECTION, "tier")
    client.index_calls.clear()

    await ensure_payload_indexes(client)  # type: ignore[arg-type]

    assert client.index_calls == ["tier"]


async def test_corpus_startup_adds_indexes_to_an_existing_collection():
    with patch("services.corpus.src.corpus.main.get_qdrant_client") as factory:
        qdrant = AsyncMock()
        qdrant.collection_exists.return_value = True
        qdrant.get_collection.return_value.config.params.vectors = {DENSE_VECTOR: object()}
        qdrant.get_collection.return_value.config.params.sparse_vectors = {SPARSE_VECTOR: object()}
        qdrant.get_collection.return_value.payload_schema = {}
        factory.return_value = qdrant

        async with lifespan(app):
            pass

    created = [c.kwargs["field_name"] for c in qdrant.create_payload_index.await_args_list]
    assert created == list(PAYLOAD_INDEX_FIELDS)
