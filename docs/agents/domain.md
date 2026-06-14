# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the canonical glossary for all domain terms. Use its vocabulary in issue titles, refactor proposals, test names, and all agent output.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in. Every significant architectural and technology decision is recorded here.
- **`docs/design-log/`** — long-form trade-space analysis behind each ADR. Read these when you need to understand *why* a decision was made, not just *what* was decided.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront.

## File structure

```
/
├── CONTEXT.md                          ← domain glossary (single context)
├── docs/
│   ├── adr/                            ← architectural decisions
│   │   ├── 0001-three-tier-rag-corpus.md
│   │   ├── 0002-two-mode-agent-architecture.md
│   │   ├── 0003-language-scope-english-first-multilingual-extensible.md
│   │   ├── 0004-rag-pipeline-stack.md
│   │   ├── 0005-application-is-gre-test-prep.md
│   │   ├── 0006-eval-pipeline.md
│   │   ├── 0007-model-selection-and-provider-abstraction.md
│   │   ├── 0008-guardrails.md
│   │   ├── 0009-deployment-pipeline.md
│   │   └── 0010-observability-mcp-ux.md
│   └── design-log/                     ← trade-space deep-dives per ADR
│       ├── 0001-rag-pipeline-trade-space.md
│       ├── 0002-eval-pipeline-trade-space.md
│       ├── 0003-model-selection-trade-space.md
│       ├── 0004-guardrails-trade-space.md
│       └── 0005-deployment-trade-space.md
└── src/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly marks as _Avoid_.

Key term pairs to watch:
- **Candidate** not Student/User/Learner
- **Quant** not Math/Maths/Quantitative
- **AW** not Writing/Essay/Analytical Writing (full form)
- **Tutor mode / Solve mode** not Hint mode / Answer mode
- **Basic complexity / Advanced complexity** not Standard/Expert mode
- **Global corpus / School corpus / Candidate corpus** not Base/Public/Personal corpus
- **Canonical style / Personal style** not Default/Custom style
- **Mode switch** not Toggle/Override
- **Clarification turn** not Follow-up/Retry
- **Verifier-fail** not Failed turn/Math fail

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0004 (RAG pipeline stack) — proposes pgvector but we chose Qdrant because…_

## Design log policy

The design logs are read-only historical records. Don't edit them to reflect new decisions — write a new ADR instead.
