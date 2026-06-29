"""
Tests for the shared BGE-M3 embedding client.

Behaviors under test:
  1. Posts to TEI /embed and returns vectors in order
  2. Batches inputs above the per-request cap
  3. Empty input short-circuits (no HTTP call)
"""

import json

import httpx
import respx

from shared.src.shared.embedder import embed_texts


@respx.mock
async def test_embed_texts_returns_vectors():
    respx.post("http://tei/embed").mock(
        return_value=httpx.Response(200, json=[[1.0, 2.0, 3.0]])
    )
    out = await embed_texts(["a query"], url="http://tei")
    assert out == [[1.0, 2.0, 3.0]]


@respx.mock
async def test_embed_texts_batches_over_the_cap():
    def echo(request: httpx.Request) -> httpx.Response:
        n = len(json.loads(request.content)["inputs"])
        return httpx.Response(200, json=[[0.0]] * n)

    route = respx.post("http://tei/embed").mock(side_effect=echo)
    out = await embed_texts(["t"] * 40, url="http://tei")  # _BATCH = 32 → 2 calls
    assert len(out) == 40
    assert route.call_count == 2


async def test_embed_texts_empty_input_makes_no_call():
    assert await embed_texts([]) == []
