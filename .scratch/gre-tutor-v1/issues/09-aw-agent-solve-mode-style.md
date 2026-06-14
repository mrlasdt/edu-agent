Status: completed

## What to build

AW Agent Solve mode and the Personal style Skill. Solve mode generates a full scored essay (targeting a 5–6 rubric score). The style critic (Haiku-tier model) performs a single post-generation rewrite pass fixing structural violations only (PEEL/MEAL paragraph pattern, thesis presence). The Personal style Skill is activated when the Candidate has opted in AND has ≥2 essays in their Candidate corpus; it injects 2 of the Candidate's own essays as few-shot exemplars and adds voice-bias instructions — but never overrides the structural rubric.

A non-removable disclaimer is appended to every full-essay Solve output.

## Acceptance criteria

- [ ] Solve mode returns a complete essay, not a plan
- [ ] Solve system prompt loaded from `prompts/aw_agent/solve/v1.md`
- [ ] Style critic (Haiku model from config) runs after generation; structural violations trigger one rewrite pass
- [ ] If Personal style is active: 2 Candidate corpus essays retrieved and injected as few-shot exemplars
- [ ] Personal style Skill no-ops silently (falls back to Canonical style) when Candidate has < 2 corpus essays
- [ ] Style opt-in flag is stored on the Candidate's session preferences (persisted to Postgres)
- [ ] Every Solve essay output ends with the non-removable disclaimer (verified via string check in tests)
- [ ] Citation checker runs on full-essay output same as Tutor mode
- [ ] Unit tests: solve path, style critic rewrite triggered, Personal style active, Personal style cold-start fallback

## Blocked by

`.scratch/gre-tutor-v1/issues/08-aw-agent-tutor-mode.md`
