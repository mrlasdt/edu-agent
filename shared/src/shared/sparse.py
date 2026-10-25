"""
Local lexical sparse encoder for the hybrid (dense + sparse) retrieval channel.

BGE-M3's learned sparse head is not exposed by the TEI container used for dense
embeddings, and reproducing it locally needs torch plus the model's
sparse_linear weights. Until that dependency is accepted, the sparse channel is
a hashed term-frequency vector: Qdrant applies the IDF modifier configured on
the collection (see vectorstore.ensure_collection), so the dot product is a
TF-IDF lexical match. It complements dense exactly where dense is weak (exact
terms, numbers, rare tokens).

Shared by both sides of Qdrant, like the dense embedder: ingestion and query
MUST tokenise and hash identically or the indices never line up. Swapping in a
learned sparse model only means replacing these two functions.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter

from qdrant_client import models

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _index(token: str) -> int:
    # crc32 is stable across processes (unlike hash()) and fits Qdrant's u32 index.
    return zlib.crc32(token.encode("utf-8"))


def encode_document(text: str) -> models.SparseVector:
    """Sub-linear term-frequency weights (1 + ln tf) per hashed token."""
    counts = Counter(_index(t) for t in _tokens(text))
    indices = sorted(counts)
    return models.SparseVector(
        indices=indices, values=[1.0 + math.log(counts[i]) for i in indices]
    )


def encode_query(text: str) -> models.SparseVector:
    """Unique query terms, weight 1.0 (IDF is applied server-side)."""
    indices = sorted({_index(t) for t in _tokens(text)})
    return models.SparseVector(indices=indices, values=[1.0] * len(indices))
