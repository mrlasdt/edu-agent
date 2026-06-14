Status: ready-for-agent

## What to build

End-to-end streaming turn for the Quant Agent in Tutor mode. A Candidate submits a GRE Quant problem; the agent scaffolds toward the answer via Socratic hints without revealing it. The Gateway receives the turn, the Orchestrator routes it to `QuantAgent`, and the response streams back token-by-token via Server-Sent Events (SSE).

The `QuantAgent` loads its system prompt from `prompts/quant_agent/tutor/v1.md`, calls the configured LLM via LiteLLM, and yields tokens as an async generator. The Gateway service exposes `POST /chat/turn` which proxies the stream to the client.

## Acceptance criteria

- [ ] `POST /chat/turn` with a GRE Quant session returns an SSE stream
- [ ] First token arrives in under 2 seconds against the real OpenAI API (integration test, skipped in CI without API key)
- [ ] `QuantAgent.run(session, message)` returns an `AsyncGenerator[str, None]`
- [ ] LiteLLM is called with the model from `model_config.dev.yaml` (not hardcoded)
- [ ] The tutor system prompt is loaded from `prompts/quant_agent/tutor/v1.md`
- [ ] Conversation history from `session.history` is included in the LLM call
- [ ] Unit tests mock `litellm.acompletion`; no real API calls

## Blocked by

`.scratch/gre-tutor-v1/issues/02-agent-registry-session-orchestrator-skeleton.md`
