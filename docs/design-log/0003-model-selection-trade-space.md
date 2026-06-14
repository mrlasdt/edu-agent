# Model selection trade-space

Long-form deep-dive of the sub-decisions captured in ADR-0007.

## Sub-decisions

### Provider abstraction layer

**Chosen:** LiteLLM.

**Rejected:**
- *Direct provider SDKs (OpenAI Python, Anthropic Python)* — forecloses cross-provider work without a rewrite; ergonomic for one provider only.
- *Hand-rolled wrapper* — reimplements LiteLLM badly; doesn't pay rent.
- *LangChain* — heavy framework; opinionated about chains and agents in ways that conflict with our explicit Skill + Orchestrator design.
- *Pydantic AI* — newer, lighter framework, attractive; rejected because LiteLLM is single-purpose and our agent shape is already opinionated by ADR-0002. Pydantic AI's value is in the agent abstraction we already have.

**Rationale:** LiteLLM does one thing — unify provider APIs — and does it well. Lets us swap providers in YAML without code changes.

### Primary provider default

**Chosen:** OpenAI.

**Rejected:**
- *Anthropic primary* — was the prior recommendation; explicit user pivot to OpenAI as default. Anthropic remains a first-class alternative via config.

**Rationale:** the choice is configuration, not architecture. The abstraction layer makes the default reversible per environment.

### Per-role model picks (OpenAI defaults)

#### Orchestrator — `gpt-4o-mini`

**Rejected:** `gpt-4o` (overkill — task is short classification), `o1-mini` (too slow for a pre-turn step).
**Rationale:** sub-second latency target; ~$0.0005/call; sufficient quality on enrich + classify with structured outputs.

#### Quant Agent / AW Agent — `gpt-4o`

**Rejected:** `gpt-4o-mini` (measurable quality drop on multi-step reasoning), `o1` (slow, overspec for primary turns, can't stream well).
**Rationale:** the right cost/quality tier for primary turns; streams well; strong structured-output support for tool use.

#### Math verifier escalation — `o1`

**Rejected:** `gpt-4o` (already tried via retry path), `o3-mini` if available (depends on roadmap; could swap), `claude-opus-4-7` (strong alternative — keep as one-line swap).
**Rationale:** when Sonnet/gpt-4o + 2 retries can't agree with the verifier, the problem genuinely needs reasoning depth. `o1` is the reasoning-tier model. Bounded to ~1% of Quant turns by the retry policy, so per-turn cost is not a budget killer.

#### Style critic / per-PR judge / style extractor — `gpt-4o-mini`

**Rejected:** `gpt-4o` (overkill for narrow tasks).
**Rationale:** all three tasks are calibrated against a narrow rubric or are background. Mini is enough.

#### AW judge nightly — `gpt-4o`

**Rejected:** `o1` (slow; the AW judge runs in a high-throughput nightly pipeline), `gpt-4o-mini` (less calibrated; baseline judge needs to be the authority).
**Rationale:** nightly eval is the authority on AW quality; use the calibrated tier.

### Embedder / Reranker

**Unchanged:** BGE-M3 / BGE-reranker-v2-m3 (ADR-0004).
**Rationale:** open-source, multilingual, provider-independent.

### Caching strategy

**Chosen:** put stable content first in prompts so both OpenAI (automatic prefix caching) and Anthropic (explicit `cache_control`) benefit.

**Rejected:**
- *Provider-specific prompt templates* — duplicates work; bug-prone.
- *No caching* — leaves 50–90% input-cost savings on the table.

**Rationale:** one prompt template that benefits from both caching mechanisms is the right level of abstraction. LiteLLM hides the wire-format detail.

### Provider failover

**Chosen:** none in v1. Single provider per role; LiteLLM retry only on transient errors (429/5xx) within the configured provider.

**Rejected:**
- *Automatic provider failover (LiteLLM fallback chain)* — adds operational complexity: dual secrets, divergent rate limits, divergent caching semantics, divergent structured-output behaviour. Promotes silent quality regression (e.g. an Anthropic fallback under load gives subtly different outputs).
- *Manual failover playbook* — implicit; an engineer-edits-config is fine as the v1 incident response.

**Rationale:** the abstraction supports failover, but the operational discipline for it isn't worth v1 effort. v2 reliability ADR if production demands it.

### Streaming

**Chosen:** stream on main-agent user-facing turns (Quant, AW); don't stream on Orchestrator, critics, judges, extractors.

**Rejected:**
- *Stream everywhere* — wastes effort on structured outputs that the system consumes whole.
- *Stream nowhere* — degrades user-facing UX (longer perceived latency on essay generation).

**Rationale:** stream where users see the output; don't where the system does.

### What's NOT in this layer

- *Fine-tuning* — would marginally improve Orchestrator latency/cost; doesn't pay rent at v1 scale.
- *Speculative decoding* — provider-side optimisation; not exposed via LiteLLM in a useful way.
- *Routing by content (e.g. classify question difficulty → choose model)* — interesting; the Orchestrator already does this for complexity (Basic vs Advanced); per-model dynamic routing is overfitting.
- *Multi-model ensembling* — adds API surface and latency for marginal gain.
