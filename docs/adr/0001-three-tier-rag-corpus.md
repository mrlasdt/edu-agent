# Three-tier RAG corpus with Schools as tenants

The literature side of the agent must "reference lessons from the student's school" and support both student uploads and a large-scale teacher/admin ingestion pipeline. We adopt a three-tier corpus — **Global** (Admin-curated MOET base), **School** (Teacher uploads, scoped by `school_id`), and **Student** (private per-student uploads) — with the School being the tenant boundary. Retrieval merges all three tiers with provenance preserved through to citation, so every quoted passage is labelled by tier ("from the textbook" / "from your teacher's notes" / "from your own notes").

## Status

Accepted.

## Considered options

- **A. Flat corpus (no tenancy)** — rejected: a single bad upload from any contributor pollutes every student's experience, and there is no ACL story.
- **B. Schools as tenants (chosen)** — gives real multi-tenant ACL on retrieval without requiring a full classroom-management product.
- **C. Classes as tenants** — rejected for the MVP: requires enrolment, roster management, semester lifecycle, and teacher-to-class assignment. A non-breaking upgrade path from B; revisit when there is a teacher pilot.
- **D. Explicit publish workflow** — rejected for the MVP: solvable later by adding `published_at` and `published_scope` columns. No moderator role exists yet.

## Consequences

- v1 ships user-facing **Global + Student** tiers only. The School tier exists in the data model and the retrieval merge logic, but the Teacher-facing UX and onboarding flow are deferred to v2.
- The ingestion pipeline is built once and shared by Global (Admin writes) and, in v2, School (Teacher writes) — same parser, chunker, embedder, indexer, with a different scope value on the ACL hook. So Admin ingestion in v1 exercises the multi-tenant pipeline end-to-end.
- Citations must surface the source tier. Conflating tiers ("from the textbook" when the source was actually a teacher's note) is a guardrail failure and a tracked eval signal.
- B → C is an additive migration (add `class_id` alongside `school_id`); no rewrite required.
