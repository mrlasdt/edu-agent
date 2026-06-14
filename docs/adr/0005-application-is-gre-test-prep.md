# Application is GRE test prep

The agent's application domain is GRE test preparation — helping Candidates raise their score on the GRE General Test — not generalised high-school homework help. This pivot leaves the architecture (ADRs 0001, 0002, 0004) untouched: corpus tiers, two-mode agent shape, Orchestrator gateway, Skill registry, math verifier, and RAG stack all map cleanly onto GRE. What changes is content scope and vocabulary.

## Status

Accepted. Supersedes the implicit high-school / per-curriculum scoping of CONTEXT.md prior to this ADR.

## Considered options

- **Generalised high-school homework helper** — original framing. Rejected: content scope is unbounded (every grade, every subject, every country); demo would feel shallow.
- **Per-country high-school helper (US 10th-grade)** — earlier proposal. Rejected: country-locks the product and forces curriculum-specific content choices that don't earn their place.
- **GRE test prep (chosen)** — a single, well-defined, standardised exam with publicly published rubrics, sample materials, and a global candidate base.

## Why GRE specifically maps so cleanly onto the existing architecture

- **Quant** plays the role the Math content was playing. The sympy verifier and the verify-math Skill apply directly. Quant question types (QC / MC / Numeric Entry / Data Interp) are slightly more varied than the earlier algebra/trig scope, but they're all within sympy's reach.
- **AW** plays the role the Literature content was playing. The two task types (Issue / Argument) replace the "two literary works" plus "one essay form" structure. The published ETS rubric replaces the canonical-style rubric we'd otherwise have to define ourselves.
- **The Personal style skill** unchanged in semantics — biases voice toward Candidate's own essay drafts while keeping the high-rubric structure intact.
- **The Global corpus has a sharper definition** — official ETS publicly-released materials. Curation effort drops because the canon is bounded and copyright-clear.
- **The three-tier corpus design** continues to make sense: Global (ETS public materials) + School (prep-course tenant — a real commercial customer) + Candidate (personal drafts and error logs).

## Consequences

- **v1 sections**: scope to be confirmed in a follow-up grill question. Default proposal: **Quant + AW**, with **Verbal deferred to v2** (Verbal requires a different problem shape — vocabulary lookups and reading-comprehension multi-step reasoning — and is the weakest fit for the existing architecture).
- **Multilingual extensibility (ADR-0003) re-anchored.** GRE is English-only, so v2 multilingual extension means *other tests in other languages* (e.g. GMAT, TOEFL, IELTS, JLPT-style test prep), not "the same GRE agent in another language."
- **Score is the product north star.** Eval signals must include not just per-turn correctness/rubric scores but also (eventually) score-delta proxies — does the Candidate get better over time? Even though v1 won't ship a score predictor, the data model must capture per-turn performance so v2 can.
- **Time-pressure and adaptive features explicitly deferred.** Timed practice, adaptive difficulty, and diagnostic-and-targeted-practice flows are real GRE-prep features but are post-v1.
- **Candidate (not Student) is the user term** throughout the codebase, docs, and prompts. CONTEXT.md is the source of truth on vocabulary.
