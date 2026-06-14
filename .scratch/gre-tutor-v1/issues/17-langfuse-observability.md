Status: ready-for-agent

## What to build

Wire Langfuse LLM observability into the Agent service via the LiteLLM callback integration. Every LLM call (Quant Agent, AW Agent, style critic, AW judge, Orchestrator Haiku steps) emits a trace to Langfuse with: model name, prompt version, input/output tokens, cost, latency, cache hit/miss, and the turn's trace ID. Prompt version is logged so A/B comparison is possible in the Langfuse UI.

Langfuse runs locally in Docker Compose (Phase 1) and on a managed or self-hosted instance (Phase 2). The integration is ~30 lines added to the LiteLLM config.

## Acceptance criteria

- [ ] LiteLLM configured with Langfuse callback (`LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY` from env)
- [ ] Every LLM call appears in the Langfuse UI as a trace with correct model, tokens, cost, and latency
- [ ] Prompt version (from `model_config.yaml` active version) is logged as a trace tag
- [ ] Trace ID from the Gateway session flows through to the Langfuse trace (parent trace linking)
- [ ] Cache hit/miss visible per call (LiteLLM reports this via the callback)
- [ ] Langfuse service included in `docker-compose.yml` (already in issue 01; this wires the client)
- [ ] A/B experiment name logged as a trace tag when a non-default prompt version is active
- [ ] Integration test: run one full Quant Tutor turn against local services; verify trace appears in local Langfuse

## Blocked by

`.scratch/gre-tutor-v1/issues/05-quant-agent-solve-mode.md`
`.scratch/gre-tutor-v1/issues/09-aw-agent-solve-mode-style.md`
