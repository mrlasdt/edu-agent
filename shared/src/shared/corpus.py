from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Tier = Literal["global", "school", "candidate"]


class ChunkPayload(BaseModel):
    """
    The Qdrant payload schema — the single source of truth for what is stored
    on each vector.

    Shared by the two services that sit either side of Qdrant:
      - ingestion (write): builds a ChunkPayload per chunk and stores model_dump()
      - corpus (read): parses the stored payload back through this model

    Because both services construct/parse through this one model, renaming a
    field is a single edit that surfaces as a type/validation error at both call
    sites — not a silent KeyError at runtime across the network boundary.
    """

    text: str
    tier: Tier
    test_type: str
    source_uri: str
    page_or_section: str
    section_title: str | None = None
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    # ACL scoping — null for the global tier
    school_id: str | None = None
    candidate_id: str | None = None
