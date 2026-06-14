# Two-mode agent architecture with Orchestrator gateway

The agent operates in two **complexity modes**. **Basic** — the v1-shipped default for everyday questions — uses a per-subject specialist pattern: an Orchestrator validates and routes each turn to either Math Agent or Lit Agent, each with its own system prompt and Skill set, with side-car critics (math verifier, citation checker, style critic) running post-generation. **Advanced** — for multi-section reports and mini-projects — uses a planner → specialist sub-agents → assembler pipeline; it is architected in v1 (interfaces and stubs in code) but not implemented, deferring to v2 once real Advanced use-cases inform the planner design. The Orchestrator is more than a router: it validates input, enriches incomplete questions (emitting Clarification turns), suggests complexity escalation when a turn is mismatched with the session mode, and dispatches.

## Status

Accepted.

## Considered options

- **A. Single agent with conditionally-loaded Skills** — closest competitor; rejected because it loses the explicit code-level separation that makes the two specialist surfaces easy to evolve and test independently.
- **B. Subject-routed specialist agents (chosen for Basic)** — selected with the qualification that subject is set at session start by the Student, so the per-turn "routing" is effectively a session-level lookup, not a classifier.
- **C. Planner → executor → critic pipeline (chosen for Advanced)** — selected for the Advanced surface because multi-section reports actually decompose into multiple planned steps. Rejected for Basic where turns are single-step and planning overhead would multiply latency without quality gain.
- **D. Full orchestrator with N specialist sub-agents** — rejected as the v1 shape; reserved as the Advanced-mode shape because that is where parallel sub-tasks (research, drafting, citation, style) are real.

## Consequences

- v1 ships Basic fully and Advanced as architected stubs. The orchestrator's `BasicRouter` returns Math Agent or Lit Agent; the `AdvancedRouter` throws `NotImplemented` and surfaces a friendly "coming soon" message to the Student.
- The Orchestrator is itself an LLM-using component (Haiku-tier) for the enrich and complexity-classify steps. One Haiku call precedes every main-agent turn, adding ~300-500ms latency in exchange for incomplete-input handling and complexity-mismatch detection.
- Tutor mode has different semantics by complexity: in Basic it scaffolds the answer; in Advanced it produces a plan rather than drafted prose. Solve mode produces the full artifact in both.
- Complexity escalation is suggested, not auto-applied. Students stay in control of complexity, matching the student-controlled mode-switch policy.
- The Skill registry and sub-agent registration interface are defined and exercised by Basic in v1; Advanced sub-agents register against the same interface in v2 with no contract change.
