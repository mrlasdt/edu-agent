Status: ready-for-agent

## What to build

Candidate-facing upload overlay and Personal style opt-in preference panel. The upload overlay (paperclip icon in the chat input bar) supports drag-and-drop of PDF, DOCX, and TXT files. Client-side validation fires immediately (file type, size). A PII warning banner appears if the server detects PII in the uploaded content. The style opt-in toggle lives in a slide-out preferences panel, visible only after the Candidate has ≥2 essays in their Candidate corpus.

## Acceptance criteria

- [ ] Paperclip icon opens a drag-and-drop overlay; accepted types shown (PDF, DOCX, TXT)
- [ ] Dropping an unsupported file type shows an inline error immediately (no server call)
- [ ] Files over 10 MB are rejected client-side with a clear error
- [ ] Successful upload shows a progress indicator; on completion shows filename and "ready for search"
- [ ] If the server returns a PII warning, a banner asks the Candidate to redact before proceeding
- [ ] Preferences panel accessible via a gear icon; contains "Match my writing style" toggle
- [ ] Style toggle is hidden (not just disabled) until the server confirms ≥2 essays in Candidate corpus
- [ ] Toggling the style opt-in sends `PATCH /candidate/preferences {personal_style_enabled: bool}`
- [ ] All interactions tested with Vitest + React Testing Library

## Blocked by

`.scratch/gre-tutor-v1/issues/09-aw-agent-solve-mode-style.md`
`.scratch/gre-tutor-v1/issues/12-web-ui-chat-core.md`
