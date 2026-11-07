# Model selection and provider abstraction

The agent's generation layer is accessed through a single abstraction (**LiteLLM**) so any role can swap providers via config without code change. The v1 default configuration uses **OpenAI** for all generation roles: `gpt-4o` for primary agent turns (Quant Agent, AW Agent, nightly AW judge), `gpt-4o-mini` for low-cost roles (Orchestrator, style critic, per-PR judge, async style extractor), and `o1` for the math-escalation tier (rare; bounded to ~1% of Quant turns). Anthropic models are first-class alternatives — a one-line config change in `roles.{role}.provider` and `roles.{role}.model` swaps any role to `claude-sonnet-4-6`, `claude-haiku-4-5`, or `claude-opus-4-7`. Embedding (BGE-M3) and reranking (BGE-reranker-v2-m3) are provider-independent and unchanged.

## Status

Accepted.

## Considered options

- **Direct OpenAI SDK (no abstraction)** — rejected: forecloses Anthropic / other providers without a rewrite.
- **Hand-rolled provider wrapper** — rejected: re-implements LiteLLM badly. Real engineering effort for marginal benefit over LiteLLM.
- **LangChain / Pydantic AI / similar full framework** — rejected: too much surface area and lock-in for the value we need (just provider abstraction). LiteLLM is the focused tool.
- **Multi-provider runtime failover in v1** — rejected: LiteLLM supports it (fallback chains), but the operational complexity (multi-secret management, per-provider rate-limit handling, divergent prompt-cache mechanics) doesn't earn its place in v1. Single provider per role; no automatic failover. v2 reliability ADR. (Superseded in part by "Amendment: resilience is LiteLLM Router-native": a role may now list `fallbacks`.)
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

## Amendment: free-model default (2026-11)

The default model config is now `config/model_config.free.yaml` (every role on one free OpenRouter model), because OpenAI retires `gpt-4o`, `gpt-4.1` and their minis on 2027-01-14. `config/model_config.dev.yaml` stays as the selectable paid config (`MODEL_CONFIG_PATH`). The two IELTS mock roles (`ielts_grader`, `ielts_feedback`) are the exception: they run on `gemini/gemma-4-26b-a4b-it` (Google AI Studio, `GEMINI_API_KEY`) with the OpenRouter free 26b and 31b Gemma models as an ordered per-role `fallbacks` entry, because only the Gemma models passed the ADR-0014 development gate. The tests pin the paid one. The next amendment describes that paid config.

## Amendment: all-mini dev and local default

The shipped dev/local configuration (`config/model_config.dev.yaml`, also used by the docker compose MVP stack) runs every role, including `math_escalation`, on `gpt-4o-mini` to stay within the project's spend budget. The `gpt-4o` / `o1` tiering described above remains the intended production shape; the per-turn cost figures above assume that tiering, not the dev default.

## Amendment: resilience is LiteLLM Router-native (2026-11)

The consequences above say streaming, retry/backoff and rate-limit handling are LiteLLM-native, and "Multi-provider runtime failover" was deferred to a v2 reliability ADR. In practice two hand-written loops had grown instead: the IELTS completer (one try per model, plus a Google 500/503 retry) and the eval transport (3-retry linear backoff, a daily-quota regex, an `exhausted` set that acted as a cooldown), while the live `/chat` path had no retry, fallback or timeout at all. That was the "hand-rolled wrapper" this ADR rejected, grown back one incident at a time.

**Decision.** Every model call goes through one `litellm.Router`, built by `shared/src/shared/llm_router.py` and fetched with `get_router()`. The YAML shape is unchanged: each model string a role names (its own model and its `fallbacks:`) becomes a Router deployment, and a role's `fallbacks:` list becomes that model's fallback chain, in order. The policy is one `RouterPolicy`:

- `num_retries` and `retry_policy`: a timeout, 429, 500 or 503 is retried on the same model twice (three times with a 15 s wait in the evals, where free-tier quotas clear per minute); a 400 or 401 is never retried, but still moves on to the next fallback, so a missing Google key hands the call to OpenRouter at once.
- `timeout`: 120 s per attempt, on streaming and non-streaming calls alike.
- `allowed_fails` and `cooldown_time`: a model that fails more than 5 attempts inside 30 s is skipped for 30 s. This replaces the eval transport's process-lifetime `exhausted` set. Cooldown applies only to a model that has a fallback chain; a model with nothing to fall back to (a lone `/chat` model, the `*` pass-through) is never cooled down, because skipping it would fail every call for 30 s instead of trying it.
- Streaming falls back as well, to the next model when the stream fails before its first token, and litellm resumes a stream that breaks midway on the fallback.

`/chat` (`TurnMeter.stream` and `complete`), the IELTS mock, the question pipeline, the judges and the evals all use that policy. The eval harness wraps its own Router in `MeteredTransport` and installs it with `use_router`; its offline mode serves the fake model's reply as litellm's `mock_response`, so offline runs exercise the Router too.

**What stays outside the Router.** The spend cap. A call is checked against the ledger before it is made, at the dearest model in the call's fallback chain, so a fallback can never take a call past the cap and `BudgetExceededError` is never followed by an attempt on a pricier model. The ledger and the per-call record use the model that actually answered (`answered_model`). One thing the Router does not do is treat an empty reply as a failure; the IELTS completer continues down the chain itself in that case, checking the cap again before each further call.

**Not changed.** Provider and model choices, and the rule that a role lists its backups in config. The Router makes failover a runtime default for roles that list `fallbacks`; the single-provider default for every other role is unchanged.
