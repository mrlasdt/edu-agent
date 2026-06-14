Status: ready-for-agent

## What to build

The foundational extension seam for the system. An `AgentRegistry` that maps `(test_type, section)` pairs to agent classes, a `Session` model that carries `test_type` and `section` as first-class fields (enabling IELTS/GMAT extension later with no structural changes), and a minimal `Orchestrator` inside the Gateway service that resolves the correct agent from the registry and returns either a `RouteResult` or a `ClarificationTurn`.

The registry pattern is the v1 extensibility contract: registering a new agent for a new test type (`ielts`, `gmat`) is the only required change to add that test.

## Acceptance criteria

- [ ] `AgentRegistry.register(test_type, section, agent_cls)` stores the mapping
- [ ] `AgentRegistry.get(test_type, section)` returns the registered class
- [ ] `AgentRegistry.get` raises `AgentNotRegisteredError` with a message naming the unregistered `(test_type, section)` when no match found
- [ ] `Session` carries `id`, `test_type`, `section`, `mode` (default `"tutor"`), `candidate_id`, `history`
- [ ] `Session.mode` can be switched to `"solve"` and back to `"tutor"`
- [ ] `Orchestrator.process(session, message)` returns `RouteResult` (with resolved agent class) for a registered session
- [ ] `Orchestrator.process` returns `ClarificationTurn` when message is empty or whitespace-only
- [ ] `Orchestrator.process` returns `ClarificationTurn` when `(test_type, section)` not registered, rather than raising
- [ ] All behaviors covered by unit tests; no real LLM calls in tests

## Blocked by

`.scratch/gre-tutor-v1/issues/01-local-dev-stack.md`
