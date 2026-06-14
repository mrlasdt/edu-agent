Status: completed

## What to build

The async ingestion pipeline that converts uploaded documents into searchable chunks in Qdrant. An Admin uploads a PDF, DOCX, or TXT file via `POST /admin/documents`; the Ingestion service queues the job (pgmq), a worker processes it (parse → chunk → embed → index), and the job status is queryable at `GET /admin/documents/{job_id}`.

Documents are stored in the three-tier corpus model (Global / School / Candidate) keyed by `(tier, school_id, candidate_id)`. Every chunk in Qdrant carries `(tier, source_uri, page_or_section, test_type)` metadata for ACL filtering and citation.

Sync fast-path for uploads under 1 MB (process inline, return job result immediately). Async path for larger files.

## Acceptance criteria

- [ ] `POST /admin/documents` accepts PDF, DOCX, TXT; rejects other types with 415
- [ ] Files under 1 MB are processed synchronously (status `completed` in response)
- [ ] Files over 1 MB return `status: "queued"` with a `job_id`; worker processes asynchronously
- [ ] `GET /admin/documents/{job_id}` returns current job status (`queued | processing | completed | failed`)
- [ ] Processed chunks appear in Qdrant with correct metadata fields: `tier`, `source_uri`, `page_or_section`, `test_type`, `school_id`, `candidate_id`
- [ ] Chunk size: ~400 tokens, ~10% overlap, structure-aware (headings preserved)
- [ ] Failed jobs appear in a `dead_letter_queue` table with per-stage error category
- [ ] Unit tests: parser, chunker, and embedder tested independently with fixture documents; no real Qdrant in unit tests

## Blocked by

`.scratch/gre-tutor-v1/issues/01-local-dev-stack.md`
