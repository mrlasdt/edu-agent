# Guardrails trade-space

Long-form deep-dive of the sub-decisions captured in ADR-0008. Organised by pipeline layer.

## Input layer

### Content safety on input

**Chosen:** OpenAI Moderation API.
**Rejected:**
- *Llama Guard self-hosted* — adds infra (GPU or hosted inference); marginal quality difference at our scale.
- *Haiku-based safety prompt* — flexible but pays per call.
- *No moderation* — irresponsible for any user-facing surface.
**Rationale:** free, ~50ms latency, calibrated against a broad safety taxonomy. Right tradeoff at v1.

### Topic relevance

**Chosen:** folded into the Orchestrator's classify step (Haiku, structured output: `subject ∈ {quant, aw, off_topic}`).
**Rejected:**
- *Dedicated classifier* — duplicates an LLM call.
- *No topic check* — Candidates would derail sessions into non-GRE territory unnoticed.
**Rationale:** the Orchestrator is already classifying, so one extra label adds zero latency.

### Prompt-injection defence

**Chosen:** regex first pass for known injection patterns ("ignore previous", "system:", role tokens) + Haiku spot-check on suspicious turns.
**Rejected:**
- *Dedicated injection-detection model* — real engineering for marginal v1 risk reduction.
- *Strict input sanitisation (escape all special chars)* — false-positive heavy; degrades real user inputs.
- *No injection defence* — naive at any meaningful scale.
**Rationale:** layered cheap checks catch the common 90%; sophisticated defence is v2 if attack data demands it.

## Generation layer

### Tutor-mode answer-leak detection

**Chosen:** post-generation regex (for known leak patterns like "the answer is") + Haiku binary check ("does this output reveal the final answer?").
**Rejected:**
- *System-prompt-only enforcement* — empirically unreliable; models forget mid-turn.
- *Tool-call-only output (the agent can't produce the answer at all)* — too restrictive; hints sometimes need to be near the answer to be useful.
**Rationale:** prompt + post-check is the belt-and-braces approach. Track failure rate as a regression signal.

### Math verifier agreement

**Chosen:** sympy sandbox match (ADR-0004); retry → escalate → degrade.
**Rejected:** see ADR-0004.

### Citation completeness

**Chosen:** deterministic — every `[cite]` resolves to a retrieved chunk's metadata.
**Rejected:**
- *LLM judge for citation correctness* — overkill; string match is reliable.
- *Citation as soft prompt-instruction only* — unreliable.
**Rationale:** citations are a guardrail. Don't use an LLM where a string match works.

### Style adherence

**Chosen:** Haiku-tier style critic; single rewrite pass for structural violations only.
**Rejected:** see ADR-0004's style design.

## Output layer

### Content safety on output

**Chosen:** OpenAI Moderation API on output text.
**Rejected:** *Skip output check (trust the generator)* — incidents will come from generation, not just input.
**Rationale:** belt-and-braces.

### PII echo

**Chosen:** regex on output text; redact and emit.
**Rejected:**
- *Block on PII echo* — degrades UX for legitimate cases (Candidate's own essay has their name in it).
**Rationale:** redact-and-emit is the safer default; the Candidate can re-add explicit identifiers if they want.

### Copyright spot-check on AW output

**Chosen:** sliding-window hash match against Global corpus; block if N consecutive words verbatim.
**Rejected:**
- *Embedding-similarity check* — false positives on common phrasing.
- *No copyright check* — risks the agent ghost-writing verbatim ETS sample essays.
**Rationale:** verbatim reproduction is the bright line; paraphrase is acceptable. Hash match catches the bright-line violations.

### Full-essay Solve policy

**Chosen:** non-removable disclaimer + soft daily quota (3/day).
**Rejected:**
- *No full essays ever* — kills a real Candidate use case.
- *Watermarking the essays* — known to be removable; security theatre.
- *Block copy-paste of essay text* — futile; users have screenshots, OCR, etc.
**Rationale:** can't prevent misuse mechanically. Disclaimer + quota signals intent; logs the rate for product-level visibility.

## Upload layer

### File type / size / malware / safety

**Chosen:** allow-list MIME, magic-bytes check, ClamAV scan, OpenAI Moderation on sampled chunks. Reject on any fail.
**Rejected:** *Trust user uploads* — obvious abuse surface.

### PII on upload

**Chosen:** detect → surface to Candidate → offer redaction.
**Rejected:** *Auto-redact silently* — degrades documents in surprising ways. *Refuse PII uploads outright* — friendly user experience demands offering the redact path.
**Rationale:** the Candidate is the authority on their own content; we provide the warning and the tool.

## Per-Candidate / per-session layer

### Rate / quota / cost limits

**Chosen:** per-turn rate (60/min), per-day session quota (200 turns), per-day Candidate cost cap (alert at 80%, block at 100%), full-essay-Solve daily quota (3/day), per-hour system spend cap (page on alert).
**Rejected:** *No limits* — runaway cost is a portfolio-embarrassing failure mode. *Hard per-turn cost cap* — degrades legitimate variability (long essay generation costs more, that's fine).
**Rationale:** four layers of cost protection — per-turn rate, daily quota, daily cost, hourly system spend — catches different abuse / outage patterns.

## Refusal policy (system prompt, not runtime)

**Chosen:** wire refusal categories into the agent's base system prompt: real-test taking on behalf of, content explicitly for test-condition submission, non-GRE academic work, cheat-the-test requests.
**Rejected:**
- *Runtime intent classifier* — adds an LLM hop and is less reliable than a prompt directive backed by the model's own RLHF.
**Rationale:** these are things the agent should natively refuse; system-prompt directives use the model's existing safety training rather than fighting it.
