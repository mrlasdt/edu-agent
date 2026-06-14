# Eval pipeline trade-space

Long-form deep-dive of the sub-decisions captured in ADR-0006. Sub-decisions are grouped by the three eval layers (offline, online, integration) plus the meta-decision (where to draw the line on eval engineering for v1).

## Sub-decisions

### Quant golden set

**Chosen:** ~150 problems from publicly released ETS materials, proportional across the four Quant question types in v1 (QC, MC-single, MC-multi, Numeric Entry).

**Rejected:**
- *Synthetic problems (LLM-generated)* — labelling bias is hard to detect; fragility on edge cases that the LLM doesn't naturally produce.
- *Larger set (500+)* — diminishing returns past ~30/slice; cost scales linearly.
- *Private/paid GRE problems* — copyright and ToS risk.

**Rationale:** 150 ETS items gives ≥30 problems per slice for stable percentages, fits a ~$5–10 nightly budget, and is uncontroversially the right ground truth.

### AW golden set

**Chosen:** 24 prompts (12 Issue + 12 Argument), each with the 6 anchor essays.

**Rejected:**
- *Larger AW set (60+ prompts)* — each prompt costs meaningfully more to eval than a Quant problem (long output, LLM judge), and the marginal information drops quickly.
- *Hand-graded AW set from internal graders* — not v1; would be needed to validate the judge against external truth but adds a labeling product.

**Rationale:** 24 prompts × 6 anchors = 144 anchor essays as calibration baseline. Distributional metrics stable at this size.

### Quant judge

**Chosen:** Deterministic comparison to ground truth via the sympy verifier (same verifier used in the agent's verify-math Skill).

**Rejected:**
- *LLM judge for Quant* — pure overhead. Quant correctness is symbolic; LLM adds noise.
- *Step-by-step grading* — interesting for proof problems, irrelevant for GRE Quant (final-answer matters).

**Rationale:** the right judge is the cheapest correct judge.

### AW judge

**Chosen:** LLM judge (Sonnet nightly, Haiku per-PR), prompted with the ETS rubric *and* the 6 anchor essays for the prompt being scored.

**Rejected:**
- *Opus judge* — too expensive for the marginal gain; Sonnet with anchors is calibrated.
- *Sonnet judge without anchors* — drifts in absolute scoring across runs; baselines become unreliable.
- *Cross-check ensemble (e.g. Sonnet + Gemini + GPT-4)* — interesting but adds API surface, secrets, and latency for limited benefit at this stage.
- *Rule-based rubric scorer* — under-fits the rubric. AW scoring requires holistic judgment.

**Rationale:** anchor-essay prompting is the cheapest known technique that gets the judge into the same calibration zone across runs. It's not perfect — that's what the Human review queue is for.

### Judge calibration self-check

**Chosen:** every nightly run feeds the six anchor essays per prompt back through the judge; if any anchor mis-scores by ≥1 point, alert.

**Rejected:**
- *Periodic human re-grading* — would be more rigorous; not v1. Use the Human review queue instead.
- *Cross-model agreement* — see above; adds API surface.

**Rationale:** if the judge can't score the very essays it's been anchored on, the whole AW signal is broken. Cheap continuous self-check is the right level of paranoia.

### Style adherence judge

**Chosen:** separate LLM-judge prompt focused on the canonical structural rubric (PEEL/MEAL paragraphs, thesis presence, evidence-analysis-link pattern). Binary pass/fail per essay.

**Rejected:**
- *Folded into the rubric score* — loses the signal. A 4-rubric essay can be structurally weak (fixable) or intrinsically thin (not fixable); we want to know which.
- *Rule-based parser (e.g. sentence-pattern detection)* — fragile on real text.

**Rationale:** splitting "score" from "structure" is the single most useful eval split because it tells us where to invest engineering effort.

### Citation judge

**Chosen:** deterministic post-processor. Every claim with a `[cite]` tag must resolve to a retrieved chunk's source metadata. Per-claim pass/fail; per-essay completeness rate.

**Rejected:**
- *LLM judge for citation correctness* — overkill. String/metadata match is reliable here.

**Rationale:** citations are a guardrail, not a heuristic. Don't use an LLM where a string-match works.

### Personal-style adherence

**Chosen:** cosine similarity between style profile of generated essay and the Candidate's existing style profile (only when Personal style is on).

**Rejected:**
- *LLM judge for style match* — unreliable; LLMs don't have a great sense of writing-voice similarity in a single comparison.
- *Stylometric distance metrics (Burrows' Delta, etc.)* — would work but heavier than necessary for the signal value.

**Rationale:** cosine on the existing style-profile vector is a free, monotone, ablatable signal.

### Per-PR eval cost optimisation

**Chosen:** path-based CI gating (eval runs only on PRs that touch prompts/agents/skills/evals/model-configs) + 10% sample (~15 Quant + 3 AW) + Haiku judge for the PR run.

**Rejected:**
- *Run full eval on every PR* — wasteful; most PRs don't change behaviour-relevant code.
- *Run no eval on PRs, only nightly* — too slow a feedback loop for prompt iteration.
- *Same Sonnet judge as nightly* — Haiku is sufficient for catching big regressions; Sonnet is overkill for PR-level sampling.

**Rationale:** the goal of per-PR eval is to catch obvious regressions before merge, not to replace nightly. Cheaper judge + smaller sample + skip-when-irrelevant is the right balance.

### Online telemetry

**Chosen:** structured per-turn logs with trace IDs. Fields: input, mode, agent, skills invoked, retry count, escalation events, output, judge scores (if applicable), latency, cost.

**Rejected:**
- *Free-text logging* — useless for analytics.
- *No telemetry until product launch* — debugging the agent without traces is impractical.

**Rationale:** trace IDs are non-negotiable. Without them, multi-step turn debugging falls apart.

### Candidate feedback

**Chosen:** thumbs-up / thumbs-down per response; optional structured form on AW essays.

**Rejected:**
- *Rubric-style feedback from Candidates* — they're not graders; data quality would be poor.
- *Forced feedback before next turn* — friction kills usage.

**Rationale:** low-friction signal beats high-friction. Aggregated thumbs are surprisingly informative on prompt regressions.

### Drift detection

**Chosen:** three alerts — avg AW rubric drop >5% (24h window vs 7-day baseline); `verifier-fail` rate >2× baseline; per-turn cost >150% of budget.

**Rejected:**
- *Many alerts on many signals* — alert fatigue; ignored after a week.
- *No drift detection in v1* — too easy to miss a model-version regression in production.

**Rationale:** three alerts cover the three distinct failure modes (quality / correctness / cost). Add more only on a real incident.

### Human review queue (gotohuman placeholder)

**Chosen:** wire the hook in v1 as a stub that writes to a `human_review_queue` table. v2 swaps the writer for the `gotohuman` API client.

**Triggers** (v1 and v2):
- Judge confidence below threshold
- Boundary AW scores (3.5, 4.5)
- Drift-alert investigation samples
- 1% sample of `verifier-fail` events

**Rejected:**
- *Build a human-grading UI in v1* — own product; defer.
- *Skip the hook entirely* — locks us out of human-in-the-loop without re-architecture later.
- *Wire gotohuman API directly in v1* — premature integration; the stub serves the same architectural purpose at zero external dependency.

**Rationale:** the integration point matters more than the integration itself. The stub gives us the architectural seam without taking on `gotohuman` as a v1 dependency, and the queue is useful for manual review even without the external service.

### What we explicitly chose NOT to build in v1

- No eval orchestration UI (CLI is fine)
- No labeling pipeline / human grader product
- No score-predictor model
- No synthetic golden items
- No ensemble cross-model judging
- No A/B framework infrastructure (Q-deployment-grill will discuss)

**Rationale:** each is its own product. Defer until eval data shows we need it. Building them now is vanity surface area.
