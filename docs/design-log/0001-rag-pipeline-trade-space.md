# RAG pipeline trade-space

Long-form deep-dive of the sub-decisions captured concisely in ADR-0004. Each row in the table records the chosen option, the rejected alternatives, and the rationale, so future sessions can re-open any single sub-decision without re-deriving the analysis.

This file is documentation, not a spec. ADR-0004 is the authority on what was decided; this is the authority on *why*.

## Sub-decisions

### Input formats

**Chosen:** PDF (digital), DOCX, plain text. Web URLs as a stretch goal.

**Rejected:**
- *Scanned PDFs / image uploads / handwriting photos* — deferred to v2 with OCR.
- *Audio (lecture recordings)* — different pipeline (transcription → text), defer.
- *Video* — same as audio plus a frame-extraction problem, defer.

**Rationale:** covers ~95% of teacher/admin uploads and the majority of student-upload use cases that don't involve photos of homework. Skipping OCR removes a meaningful operational and quality burden from v1.

### OCR

**Chosen:** none in v1.

**Rejected alternatives if OCR were in scope:**
- *Tesseract + LLM cleanup* — cheap but noisy, especially on non-English text and complex layouts.
- *Cloud OCR (Google Document AI, AWS Textract, Azure Form Recognizer)* — strong on structured forms, less so on prose; ToS lock-in.
- *Vision-LLM OCR (Claude / GPT-4V / Gemini vision)* — best quality on prose; more expensive per page; this is the v2 default if OCR is added.

**Rationale:** skipping OCR removes a whole class of failure modes (low-DPI scans, skewed pages, mixed handwritten/printed content) from v1. When OCR is added in v2, vision-LLM is the right choice.

### Chunking strategy

**Chosen:** document-structure-aware hierarchical chunking. Parse headings and sections; emit parent chunks (whole section, summarised) plus child chunks (paragraph-level, ~300-500 tokens with ~10% overlap).

**Rejected:**
- *Fixed-size chunking (every N tokens)* — loses structure; poor recall on questions that depend on which section a passage came from.
- *Sentence-level chunking* — too granular; loses context.
- *Semantic chunking (boundary detection via embeddings)* — gains nothing over structure-aware on documents with explicit structure (textbooks, essays), and adds a fragile step.

**Rationale:** literary and curricular content has explicit structure. Use it. Hierarchical lets the retriever surface a child chunk for precision and the parent for context — meaningfully better than either alone.

### Embedding model

**Chosen:** BGE-M3 (multilingual, dense + sparse + ColBERT outputs in one model).

**Rejected:**
- *OpenAI text-embedding-3-large* — strong but per-token cost; vendor lock; closed weights mean no ability to self-host or fine-tune.
- *Language-specific embedders (e.g. PhoBERT variants for Vietnamese)* — incompatible with ADR-0003's multilingual-extensibility constraint.
- *Cohere embed-v3 multilingual* — competitive but paid; BGE-M3 matches it within margin and is open.
- *Smaller open embedders (e5-small, GTE-base, etc.)* — measurable quality drop on literature-style retrieval; not worth the latency saving for our scale.

**Rationale:** BGE-M3 is the open-source state of the art for multilingual dense+sparse and gives hybrid retrieval for free (one model produces both vector types).

### Vector store

**Chosen:** Qdrant.

**Rejected:**
- *pgvector* — fine up to ~1M vectors and simpler operationally (same DB as app data), but ceiling is lower and payload filtering is awkward at multi-tier scale.
- *Pinecone* — managed, no infra burden, but per-vector cost and vendor lock.
- *Weaviate* — feature-rich but heavier to operate; payload filtering more verbose than Qdrant.
- *Milvus* — high-scale powerhouse, overkill for our scale.

**Rationale:** Qdrant balances scale ceiling, native payload filtering (clean ACL story), open-source self-hostability, and operational simplicity. Idiomatic for vector workloads.

### Retrieval

**Chosen:** hybrid (dense + sparse) merged via Reciprocal Rank Fusion (RRF).

**Rejected:**
- *Dense-only* — underperforms on noun-rich domains (character names, technical terms, citation lookups).
- *Sparse-only (BM25)* — fine for exact-match but misses paraphrased queries.
- *Weighted linear combination of dense and sparse scores* — sensitive to score normalisation; RRF is rank-based and robust.

**Rationale:** RRF is the simplest hybrid technique that consistently beats either single mode on heterogeneous queries.

### Reranking

**Chosen:** BGE-reranker-v2-m3, applied to the top-20 hybrid results, returning top-5 to the LLM.

**Rejected:**
- *No reranker* — measurable quality drop; the gap between "good retrieval" and "grounded generation" lives in the reranker.
- *Cohere reranker* — competitive but paid.
- *Cross-encoder reranker fine-tuned in-house* — way too much infrastructure for v1.

**Rationale:** BGE-reranker-v2-m3 is open, multilingual, and same-family as the embedder. ~100-200ms latency cost is acceptable inside a 2-second LLM turn.

### Tier-priority weighting

**Chosen:** when relevance scores are close (within a configurable threshold after rerank), prefer Student > School > Global.

**Rejected:**
- *Always prefer the more-specific tier regardless of relevance* — risks surfacing low-quality personal notes over high-quality textbook content.
- *Pure score-based merge with no tier weighting* — misses the product intent (the student's own notes are more meaningful to them when they apply).

**Rationale:** lets relevance dominate when the gap is meaningful, lets tier specificity tie-break when scores are close. Threshold is tunable from eval data.

### ACL enforcement

**Chosen:** query-time filtering via Qdrant payload filter: `tier=global OR (tier=school AND school_id=:s) OR (tier=student AND student_id=:u)`.

**Rejected:**
- *Separate Qdrant collections per tier* — drift risk (different index settings, different embedding versions across tiers).
- *Application-side filtering after retrieval* — leaky; one bug exposes cross-tenant data.
- *Per-school separate collections* — operationally painful at scale (thousands of small collections).

**Rationale:** single collection, single ACL rule, testable in one unit test. The right level of abstraction.

### Pipeline orchestration

**Chosen:** async job queue (Postgres-backed, e.g. pgmq or a small Celery setup if Python). Sync fast-path for student uploads <1MB.

**Rejected:**
- *Temporal / Airflow* — heavy, multi-component, overkill for our scale.
- *Pure synchronous ingestion* — fine for small docs, fails for the large-document case we explicitly need to support.
- *Cloud-only (Lambda, Cloud Functions)* — vendor lock and harder local dev.

**Rationale:** Postgres-queue gives us one less infra dependency and is sufficient for v1 throughput. Move to a heavier orchestrator only on hard evidence.

### Failure handling

**Chosen:** dead-letter queue with per-stage error categorisation (parse fail, chunk fail, embed fail, index fail). Admin UI surfaces failures.

**Rejected:**
- *Silent retries forever* — pipelines stall invisibly; no diagnosis path.
- *Generic "failed" status without categorisation* — engineer has to dig per-failure.

**Rationale:** visibility is the difference between "it broke" and "we fix it." Cheap to implement, expensive to retrofit.

### Citation enforcement

**Chosen:** every chunk carries `(tier, source_uri, page_or_section, char_range)` metadata. Side-car citation checker validates that every literature claim has a citation traceable to a retrieved chunk; uncited claims block emission.

**Rejected:**
- *Citation as a soft prompt instruction only* — empirically unreliable; LLMs cite when convenient and omit when not.
- *Post-hoc citation generation (claim → search for source)* — fragile; can fabricate plausible-looking citations.

**Rationale:** citations are a hard product constraint ("must reference lessons"). Treating them as a guardrail rather than a nice-to-have makes the constraint enforceable.
