# RAG pipeline stack

The literature side of the agent retrieves from the three-tier corpus (ADR-0001) using a deliberate, opinionated stack: **BGE-M3** as the embedding model (dense + sparse outputs in one model, multilingual-capable per ADR-0003), **Qdrant** as the vector store with native payload filtering for tier ACL, **hybrid retrieval** (dense + sparse merged via Reciprocal Rank Fusion), **BGE-reranker-v2-m3** on the top-20 to return top-5 chunks, and **document-structure-aware hierarchical chunking** (section → paragraph). Ingestion is async via a Postgres-backed job queue; the v1 ingestion path accepts digital PDFs, DOCX, and plain text only.

## Status

Accepted. The full trade-space (alternatives considered for every sub-decision) is captured in `docs/design-log/0001-rag-pipeline-trade-space.md` for future deep-dive.

## Considered options (high-level only — see design log for detail)

- **pgvector vs Qdrant** — Qdrant chosen for the higher ceiling on corpus growth, native payload filtering, and idiomatic vector-store ergonomics.
- **Dense-only vs hybrid retrieval** — hybrid chosen: dense-only consistently underperforms on noun-rich content (proper nouns, technical terms, citation lookups).
- **Reranking on/off** — on: the gap between "good retrieval" and "grounded generation" usually lives in the reranker.
- **OCR in v1** — deferred: keeps v1 ingestion simple. Scanned-content support is v2 and will likely use vision-LLM OCR rather than Tesseract.

## Consequences

- **Tier ACL is enforced at query time** via Qdrant payload filtering: each query carries `(student_id, school_id)` and filters to `tier=global OR (tier=school AND school_id=:s) OR (tier=student AND student_id=:u)`. Single source of truth, no per-tier indexes that can drift.
- **Citations are first-class.** Every chunk stores `(tier, source_uri, page_or_section, char_range)` metadata. The post-generation citation checker fails the turn on any literature claim that doesn't trace back to a retrieved chunk.
- **Multilingual is free for the cost of v1.** BGE-M3 is multilingual; the choice was made under ADR-0003. Adding a language in v2 doesn't require re-indexing.
- **No OCR in v1** means scanned PDFs, photos of handwriting, and image uploads are explicitly rejected at the ingestion API with a clear error. Surface this in the UI so users don't waste time uploading.
- **Async ingestion has a sync fast-path** for student uploads under 1MB so the student-upload UX feels responsive. Larger uploads go to the queue and a "processing" state.
- **Reranker hop adds ~100-200ms** at retrieval time. Acceptable inside a turn already costing ~2s of LLM time. Watch the budget if Sonnet turns get faster.
