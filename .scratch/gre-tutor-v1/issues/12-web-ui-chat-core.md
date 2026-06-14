Status: completed

## What to build

The core Candidate-facing React web UI: session-start subject picker (Quant / AW), streaming chat view, mode-switch button, inline citation rendering with collapsible Sources panel, and thumbs-up/down feedback per agent turn. Built with Vite + React + Tailwind + Vercel AI SDK `useChat`.

The chat view streams tokens as they arrive. Status lines ("Checking your answer…", "Finding references…") appear above the streaming response while verifier/retrieval are running. Agent turns in AW show `[N]` inline citation links that open the Sources panel.

## Acceptance criteria

- [ ] Session-start modal shows Quant / AW picker; selection sets `test_type` and `section` on the session
- [ ] `useChat` (or equivalent SSE hook) streams tokens from `POST /chat/turn` into the message bubble in real time
- [ ] "Show me the answer" button appears below the input; clicking it sends `mode: "solve"` on the next turn
- [ ] Button label changes to "Back to hints" in Solve mode; another click sends `mode: "tutor"`
- [ ] Status lines shown during verifier ("Checking your answer…") and retrieval ("Finding references…") phases via SSE metadata events
- [ ] AW agent turns render `[N]` inline citations; clicking opens a Sources panel listing `source_uri`, `tier`, `page_or_section`
- [ ] Thumbs-up / thumbs-down icon on every agent turn; click sends `POST /feedback`
- [ ] Full-essay Solve output renders the non-removable disclaimer in a visually distinct style
- [ ] All interactions tested with Vitest + React Testing Library; no real API calls in component tests

## Blocked by

`.scratch/gre-tutor-v1/issues/03-quant-agent-tutor-mode.md`
