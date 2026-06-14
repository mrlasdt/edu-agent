Status: completed

## What to build

End-to-end AW Agent Tutor mode turn. A Candidate submits an AW prompt (Issue or Argument task); the agent emits a **plan** — outline, key points to address, suggested evidence — without drafting the essay. The `retrieve-corpus` Skill fetches relevant chunks from the Corpus service; the `cite-source` Skill validates that every claim about the corpus has a traceable citation; the citation checker post-gen critic blocks emission of any uncited claim.

The AW Agent determines which task type (Issue or Argument) from the prompt and selects the matching tutor system prompt.

## Acceptance criteria

- [ ] `POST /chat/turn` with an AW session in Tutor mode returns an essay plan (not a full essay)
- [ ] Tutor system prompt is loaded from `prompts/aw_agent/tutor/v1.md`
- [ ] `retrieve-corpus` Skill is called; retrieved chunks are injected into the LLM context
- [ ] Every corpus claim in the output has a `[N]` citation that resolves to a retrieved chunk's `source_uri`
- [ ] The citation checker blocks emission of any claim with an unresolvable citation; on block, regenerates once; if still failing, strips the uncited claim and emits with a `citation_stripped: true` flag in SSE metadata
- [ ] Issue task and Argument task are detected from the prompt and handled distinctly (different examples in context)
- [ ] Unit tests mock LiteLLM, Corpus service HTTP call, and citation checker; no real API calls

## Blocked by

`.scratch/gre-tutor-v1/issues/02-agent-registry-session-orchestrator-skeleton.md`
`.scratch/gre-tutor-v1/issues/07-corpus-retrieval-service.md`
