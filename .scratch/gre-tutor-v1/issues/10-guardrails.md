Status: completed

## What to build

The five-layer guardrail pipeline wired end-to-end: input validation, Orchestrator enrichment (Haiku-tier LLM detects incomplete questions, emits `ClarificationTurn`), Orchestrator complexity classification (Haiku-tier LLM detects Basic/Advanced mismatch, emits `ComplexityEscalation` suggestion), output content safety, PII echo redaction, and per-Candidate rate + quota counters.

Input content safety uses the OpenAI Moderation API. Tutor-mode answer-leak detection uses a post-generation Haiku check. Every guardrail emits a `guardrail.{layer}.{name}.{pass|fail}` structured log event with the turn's trace ID.

## Acceptance criteria

- [ ] Empty or whitespace-only messages return a `ClarificationTurn` (already in issue 02; this wires the Haiku enrichment step)
- [ ] An incomplete Quant question ("solve this") triggers a `ClarificationTurn` asking for the missing problem
- [ ] OpenAI Moderation API called on every input; unsafe content returns a neutral refusal with no LLM invocation
- [ ] Tutor-mode answer leak: post-generation Haiku check; if answer leaked, regenerates once; logs `guardrail.generation.tutor_mode_answer_leak.fail`
- [ ] Output PII echo: regex detects email/phone patterns in output; redacts before emission
- [ ] Per-Candidate rate limit: >60 turns/min returns HTTP 429
- [ ] Per-Candidate daily quota: >200 turns/day returns HTTP 429 with quota message
- [ ] Every guardrail check emits a structured log event with trace ID (verified in test by inspecting captured logs)
- [ ] Unit tests cover each guardrail independently; integration test runs a full guarded turn

## Blocked by

`.scratch/gre-tutor-v1/issues/03-quant-agent-tutor-mode.md`
