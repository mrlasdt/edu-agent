Status: ready-for-agent

## What to build

The Corpus service: a FastAPI service that wraps Qdrant with hybrid search (BGE-M3 dense + sparse via RRF), BGE-reranker reranking, tier ACL enforcement, and citation metadata. The Agent service calls `POST /corpus/retrieve` with a query, `(candidate_id, school_id, test_type)`, and receives the top-5 ranked chunks with full citation metadata.

ACL is enforced at query time via Qdrant payload filter: `tier=global OR (tier=school AND school_id=:s) OR (tier=candidate AND candidate_id=:u)`. The tier-priority tiebreak (Student > School > Global for close scores) is applied post-rerank.

## Acceptance criteria

- [ ] `POST /corpus/retrieve` accepts `{query: str, candidate_id: str, school_id: str, test_type: str, top_k: int}` and returns chunks with `{text, tier, source_uri, page_or_section, score}`
- [ ] Dense + sparse scores merged via RRF before reranking
- [ ] BGE-reranker applied to top-20 results; top-5 returned
- [ ] Qdrant payload filter enforces ACL: a Candidate cannot retrieve another Candidate's chunks
- [ ] Tier-priority tiebreak: when top-2 scores are within 5% of each other, prefer the more specific tier
- [ ] `test_type` filter applied: only chunks for the correct test returned
- [ ] Response latency under 500ms for a 10k-chunk corpus (measured in integration test)
- [ ] Unit tests mock Qdrant and TEI server; integration test uses local Docker services

## Blocked by

`.scratch/gre-tutor-v1/issues/06-corpus-ingestion-pipeline.md`
