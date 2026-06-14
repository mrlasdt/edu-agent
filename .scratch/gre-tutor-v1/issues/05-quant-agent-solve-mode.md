Status: completed

## What to build

Quant Agent Solve mode: the Candidate explicitly switches to Solve mode and receives a fully worked solution (step-by-step algebraic explanation). The `verify-math` Skill (MCP client to issue 04's server) validates the final answer before emission. If the verifier disagrees, the agent retries up to 2 times; on persistent failure it escalates to the stronger model (configured under `math_escalation` in `model_config`); on continued failure it emits a graceful `verifier-fail` degradation.

Mode switch is a deliberate Candidate action (`mode: "solve"` in the request body). Session mode resets to `"tutor"` at the start of each new question.

## Acceptance criteria

- [ ] `POST /chat/turn` with `mode: "solve"` returns a worked solution with step-by-step reasoning
- [ ] The solve system prompt is loaded from `prompts/quant_agent/solve/v1.md`
- [ ] `verify-math` Skill is called on every Solve turn; the answer is only emitted if verified
- [ ] On verifier disagreement: agent retries (same model, temperature unchanged, disagreement fed as context)
- [ ] After 2 retries still failing: LiteLLM called with the escalation model from config
- [ ] After escalation failure: response degrades to "I'm not confident in the final answer — here's my best reasoning" with `verifier_fail: true` in the SSE metadata
- [ ] `verifier_fail` events are logged with trace ID
- [ ] Mode resets to `"tutor"` on the next turn (new question)
- [ ] Unit tests cover: success path, single-retry success, escalation path, degradation path — all with mocked LiteLLM and mocked MCP client

## Blocked by

`.scratch/gre-tutor-v1/issues/03-quant-agent-tutor-mode.md`
`.scratch/gre-tutor-v1/issues/04-math-verifier-mcp-server.md`
