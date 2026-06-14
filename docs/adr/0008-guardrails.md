# Guardrails

The agent's safety, quality, and policy guarantees are enforced by a layered set of guardrails, each at a specific point in the request pipeline: **Input** (Orchestrator pre-routing), **Generation** (wrapping the agent turn), **Output** (final pre-emission check), **Upload** (when Candidate or Admin writes to the corpus), and **Per-Candidate / per-session** (rate, quota, cost). Each guardrail is a discrete check with a pass/fail signal, a defined fail action, and an observable event (`guardrail.{layer}.{name}.{pass|fail}` with trace ID). v1 uses OpenAI's Moderation API for content safety, regex + Haiku-tier LLM checks for prompt-injection detection and answer-leak detection, deterministic string/metadata matching for citation completeness, the sympy verifier for math correctness, and rate/quota counters for abuse protection.

## Status

Accepted.

## Considered options

- **Self-hosted Llama Guard or similar for content safety** — rejected for v1: adds infrastructure for marginal quality gain over OpenAI Moderation, which is free and zero-friction.
- **Sophisticated prompt-injection defence (dedicated model, instruction-hierarchy enforcement)** — rejected for v1: real engineering effort with diminishing returns at our scale. Light regex + Haiku spot-check is calibrated to v1 risk.
- **Hard "no full essays" policy** — rejected: full-essay Solve is a real Candidate use case (reviewing model essays at each scoring tier). v1 ships with a non-removable disclaimer and a soft daily quota instead.
- **Refuse PII uploads outright** — rejected: offered-redaction UX is friendlier and preserves the Candidate's intent to upload their own work for style extraction.

## Consequences

- **Guardrails are first-class observability surfaces.** Every check emits a `guardrail.*` event so drift detection (ADR-0006) can watch rates. If `tutor_mode_answer_leak` starts triggering at 5%, that's a real prompt regression.
- **No silent degradation.** When a guardrail forces a degraded output (e.g. strips uncited claims, falls back to canonical style), the trace flags it. Eval and dashboards see degraded outputs distinctly from healthy ones.
- **The agent's system prompt carries the refusal policy.** Wired into the base prompt, not enforced by a separate runtime check: refuse to take a real GRE test on someone's behalf; refuse generation explicitly framed for submission under test conditions; redirect non-GRE academic requests; refuse "cheat-the-test" requests that violate ETS rules. Cheaper than runtime gates for the things the agent should natively refuse.
- **Full-essay Solve outputs carry a non-removable disclaimer.** A template append, not a runtime guardrail. Soft daily quota of 3 full-essay solves per Candidate per day; subsequent requests suggest Tutor-mode or section-by-section work instead.
- **Cost ceilings double as safety guardrails.** Per-Candidate-per-day caps prevent runaway use and abuse; per-system-per-hour caps prevent runaway spend. Both alert before they block.
- **Citation completeness is enforced, not requested.** Every AW claim that references corpus content must have a citation that matches a retrieved chunk's metadata. Uncited claims block emission; on retry failure they're stripped before emit.
- **The Math verifier and Citation checker are also guardrails** (their roles in ADRs 0004 and 0006 are about quality; here they're framed as policy gates). Single physical check, two roles: quality signal in eval, hard gate at runtime.
- **PII handling is opt-in redaction at upload.** Detected PII (regex first, Haiku review on uncertain matches) is surfaced to the Candidate before storage. The Candidate decides; nothing redacted silently.
