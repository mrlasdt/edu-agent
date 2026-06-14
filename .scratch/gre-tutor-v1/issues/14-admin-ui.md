Status: completed

## What to build

A minimal Admin UI (protected routes inside the Gateway service's React bundle) for managing the Global corpus. An Admin can upload documents (PDF, DOCX, TXT) specifying `test_type` and `tier=global`, view ingestion job status (queued / processing / completed / failed), and see failed jobs with their per-stage error category. No user management, no content editing — just upload and observe.

## Acceptance criteria

- [ ] `/admin` route is protected; unauthenticated requests redirect to login
- [ ] File upload form: file picker, `test_type` dropdown (gre; more added later), submit button
- [ ] Upload calls `POST /admin/documents`; shows progress, then job status on completion
- [ ] Jobs table shows all jobs with `id`, `filename`, `test_type`, `status`, `created_at`, `error_stage` (if failed)
- [ ] Jobs table polls `GET /admin/documents` every 5s while any job is in `queued` or `processing` state; stops when all done
- [ ] Failed jobs show the error stage (parse / chunk / embed / index) and a truncated error message
- [ ] Dead-letter queue count shown as a badge; clicking navigates to a simple DLQ list view
- [ ] Admin auth is HTTP Basic (username/password from env) in Phase 1; replaceable in Phase 2

## Blocked by

`.scratch/gre-tutor-v1/issues/06-corpus-ingestion-pipeline.md`
`.scratch/gre-tutor-v1/issues/12-web-ui-chat-core.md`
