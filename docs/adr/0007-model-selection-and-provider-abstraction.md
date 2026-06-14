# Model selection and provider abstraction

The agent's generation layer is accessed through a single abstraction (**LiteLLM**) so any role can swap providers via config without code change. The v1 default configuration uses **OpenAI** for all generation roles: `gpt-4o` for primary agent turns (Quant Agent, AW Agent, nightly AW judge), `gpt-4o-mini` for low-cost roles (Orchestrator, style critic, per-PR judge, async style extractor), and `o1` for the math-escalation tier (rare; bounded to ~1% of Quant turns). Anthropic models are first-class alternatives — a one-line config change in `roles.{role}.provider` and `roles.{role}.model` swaps any role to `claude-sonnet-4-6`, `claude-haiku-4-5`, or `claude-opus-4-7`. Embedding (BGE-M3) and reranking (BGE-reranker-v2-m3) are provider-independent and unchanged.

## Status

Accepted.

## Considered options

- **Direct OpenAI SDK (no abstraction)** — rejected: forecloses Anthropic / other providers without a rewrite.
- **Hand-rolled provider wrapper** — rejected: re-implements LiteLLM badly. Real engineering effort for marginal benefit over LiteLLM.
- **LangChain / Pydantic AI / similar full framework** — rejected: too much surface area and lock-in for the value we need (just provider abstraction). LiteLLM is the focused tool.
- **Multi-provider runtime failover in v1** — rejected: LiteLLM supports it (fallback chains), but the operational complexity (multi-secret management, per-provider rate-limit handling, divergent prompt-cache mechanics) doesn't earn its place in v1. Single provider per role; no automatic failover. v2 reliability ADR.
- **All-Anthropic primary** — rejected as the default after the explicit user pivot to OpenAI. Anthropic remains a per-role configurable swap, including a likely-popular hybrid where `math_escalation` uses `claude-opus-4-7` against OpenAI defaults elsewhere.

## Consequences

- **Config is the source of truth for model picks.** A `model_config.yaml` per environment lists every role with its provider, model, and params. Code never hard-codes a model name. Tests pin a fixture config.
- **Provider differences are absorbed by the abstraction.**
  - *Caching mechanics* — OpenAI does prefix-prefix caching automatically (≥1024 token prefix, 50% hit discount, short TTL); Anthropic uses explicit `cache_control` breakpoints (90% hit discount, up to 1h TTL). The prompt assembly puts stable content first regardless, so both providers benefit; LiteLLM handles the wire-format detail.
  - *Structured outputs* — OpenAI's `response_format` with strict JSON schema is used for Orchestrator classification, verify-math tool, and AW judge. LiteLLM passes the schema through to whichever provider supports it.
  - *Streaming* — main-agent turns stream (UX); orchestrator/critics/judges do not (structured outputs needed whole). Same behaviour across providers.
- **Cost ceiling per turn is calibrated under OpenAI defaults.** Blended average ~$0.012/turn with caching working; ~$0.04 without. Hard ceiling on escalation cost: o1 caps escalations at ~$0.25/turn. Per-session 30-turn budget: $0.30–$1.50.
- **Streaming, retry/backoff, and rate-limit handling are LiteLLM-native** so the same policy applies across providers without per-provider code.
- **The escalation tier is a known cost outlier.** `o1` is expensive per token; the escalation policy (Quant verifier-fail after 2 retries) keeps the rate near 1% of Quant turns. Watch the rate as a cost-alert signal.
- **Anthropic-hybrid is one config line.** If `claude-opus-4-7` outperforms `o1` on Quant escalation in our eval, swap `math_escalation.provider: anthropic, model: claude-opus-4-7`. No code change.
