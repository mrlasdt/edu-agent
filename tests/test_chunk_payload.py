"""
Tests for the shared ChunkPayload contract (shared/corpus.py).

ChunkPayload is the single source of truth for the Qdrant payload schema,
shared by the ingestion (write) and corpus (read) services. These tests pin
the contract so a rename surfaces here and at both call sites, not silently
at runtime.

Behaviors under test:
  1. ChunkPayload round-trips build → model_dump → all fields present
  2. tier is constrained to the three valid tiers
  3. to_chunk_result projects a payload dict into the citation-facing result
  4. to_chunk_result excludes ACL ids (candidate_id/school_id) from the result
  5. The corpus ACL filter only references valid ChunkPayload fields (rename guard)
"""

import pytest
from pydantic import ValidationError

from shared.src.shared.corpus import ChunkPayload


# ── 1. Round-trip ─────────────────────────────────────────────────────────────

def test_chunk_payload_round_trips():
    payload = ChunkPayload(
        text="The GRE tests reasoning.",
        tier="global",
        test_type="gre",
        source_uri="ets-guide.pdf",
        page_or_section="Introduction",
        section_title="Introduction",
        chunk_index=0,
        char_start=0,
        char_end=24,
    )
    dumped = payload.model_dump()
    for field in ("text", "tier", "test_type", "source_uri", "page_or_section"):
        assert field in dumped
    assert dumped["tier"] == "global"


def test_chunk_payload_defaults_acl_ids_to_none():
    payload = ChunkPayload(
        text="x", tier="global", test_type="gre",
        source_uri="f.pdf", page_or_section="c-0",
    )
    assert payload.school_id is None
    assert payload.candidate_id is None


# ── 2. Tier constraint ────────────────────────────────────────────────────────

def test_valid_tiers_accepted():
    for tier in ("global", "school", "candidate"):
        ChunkPayload(text="x", tier=tier, test_type="gre",
                     source_uri="f", page_or_section="c-0")


def test_invalid_tier_rejected():
    with pytest.raises(ValidationError):
        ChunkPayload(text="x", tier="bogus", test_type="gre",
                     source_uri="f", page_or_section="c-0")


# ── 3 & 4. to_chunk_result projection ────────────────────────────────────────

def test_to_chunk_result_includes_citation_fields_and_score():
    from services.corpus.src.corpus.main import to_chunk_result

    payload = {
        "text": "GRE prep content", "tier": "global", "test_type": "gre",
        "source_uri": "ets-guide.pdf", "page_or_section": "Intro",
        "section_title": "Intro", "chunk_index": 0,
        "char_start": 0, "char_end": 16,
        "school_id": "s1", "candidate_id": "c1",
    }
    result = to_chunk_result(payload, score=0.92)
    assert result.text == "GRE prep content"
    assert result.tier == "global"
    assert result.source_uri == "ets-guide.pdf"
    assert result.page_or_section == "Intro"
    assert result.score == 0.92


def test_to_chunk_result_excludes_acl_ids():
    from services.corpus.src.corpus.main import to_chunk_result, ChunkResult

    payload = {
        "text": "x", "tier": "candidate", "test_type": "gre",
        "source_uri": "notes.txt", "page_or_section": "c-0",
        "school_id": "s1", "candidate_id": "secret-candidate-99",
    }
    result = to_chunk_result(payload, score=0.5)
    # ACL ids must not leak into the citation-facing result returned to the agent
    assert "candidate_id" not in result.model_dump()
    assert "school_id" not in result.model_dump()


# ── 5. ACL filter rename guard ────────────────────────────────────────────────

def test_acl_filter_keys_are_valid_chunk_payload_fields():
    from services.corpus.src.corpus.retriever import _build_acl_filter

    f = _build_acl_filter(candidate_id="c1", school_id="s1")
    valid_fields = set(ChunkPayload.model_fields.keys())

    def collect_keys(node):
        keys = []
        if isinstance(node, dict):
            if "key" in node:
                keys.append(node["key"])
            for v in node.values():
                keys.extend(collect_keys(v))
        elif isinstance(node, list):
            for item in node:
                keys.extend(collect_keys(item))
        return keys

    referenced = collect_keys(f)
    assert referenced, "ACL filter referenced no payload fields"
    for key in referenced:
        assert key in valid_fields, f"ACL filter references unknown payload field {key!r}"
