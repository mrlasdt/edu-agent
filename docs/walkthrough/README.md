# EduAgent walkthrough

A guide for a person who wants to learn how to apply industrial LLM engineering practices by reading a small, working system. EduAgent is a GRE tutor, but the product is the excuse: each chapter takes one practice, shows the problem you would feel without it, points at the exact code that applies it, and shows a number from this repo that proves what it did.

It is a guide, not a spec. The decisions live in [docs/adr](../adr/); this explains how to read the code and why it looks the way it does. The guide grows one chapter per change that adds a practice, so a chapter always matches the code it describes and carries the numbers measured when that code landed.

## Chapter template

Every concept chapter follows the same template, so you always know where to look. It is the standard later chapters must meet. The parts, in order:

1. **Problem** - what you would notice as a user or an operator without the practice.
2. **The practice** - one paragraph, with the source post linked.
3. **Landscape** - a table of the *categories* of the practice, not just the one EduAgent picked. Columns: **What it is**, **Example**, **Trade-off**, **When to use**, **EduAgent today** (used here, or not used and why). A learner should be able to choose a different category for a different product after reading it.
4. **Open-source options** - a table of the real tools in this space (with links): what each does and why EduAgent does or does not adopt it. Adopt in code only when the tool teaches better than the hand-rolled version and does not add a running service or a heavy dependency; otherwise say why not.
5. **Why here** - the specific place the practice bites in EduAgent.
6. **Code tour** - files and symbol names (not line numbers, so the tour survives edits).
7. **Before and after** - a result file under [results/](results/), produced by a command the chapter names.
8. **Trade-offs** - cost, risk, and when not to use it.
9. **Try it** - a command to run yourself, then an **exercise**: a small change with a result to predict.

Rules for the content: it must be general public knowledge with public links, never the names or internals of a private system, and every "EduAgent today" cell must be true of the code at the commit that adds it. Chapter 0 is orientation and is exempt from the landscape tables.

## Roadmap

This is the plan for the whole guide, kept in the repo so it is not lost. Every concept chapter meets the [chapter template](#chapter-template): a landscape table of the categories of the practice, an open-source options table, then the code. Status: **merged** (written and in the repo), **in progress** (being written in another change), **planned**. Chapters are in learning order, not build order; a chapter without a link has not been written yet. An appendix (glossary, results index, how to rerun everything) is planned after chapter 12.

| Ch | Concept | Status | Landscape categories it covers | Open-source candidates |
|---|---|---|---|---|
| 0 | How to use this guide | merged | orientation only, no landscape table | - |
| 1 | [Measure first](01-measure-first.md) | merged | per-call usage metering; per-task records; structured logs; request tracing (spans per model and tool call); metrics and dashboards; LLM tracing platforms; cost attribution; sampling and retention | Langfuse, OpenTelemetry with OpenLLMetry, Arize Phoenix, Helicone, LiteLLM callbacks |
| 2 | [Evals from real failures](02-evals-from-real-failures.md) | merged | rule or code graders; model-as-judge graders; human review; capability versus regression suites; offline versus online evals; single-turn versus multi-turn and trajectory evals; pass@k versus pass^k; golden sets versus synthetic sets | Ragas, promptfoo, DeepEval, Langfuse evals, OpenAI Evals, Inspect |
| 3 | [Workflows before agents](03-workflows-before-agents.md) | merged | prompt chaining; routing; parallelization (sectioning, voting); orchestrator-workers; evaluator-optimizer; autonomous agent loop; when a fixed workflow beats an agent | LangGraph, LangChain LCEL, LlamaIndex Workflows, CrewAI, plain Python |
| 4 | Tool design: compute, don't guess | planned | client-defined functions (function calling); provider-hosted server tools (web search, code execution); code-execution sandbox; computer use and browser; bash and text editor; memory tool; tool search and deferred loading; MCP servers versus in-process tools; tool description and error design; bounded tool loops | MCP Python SDK, LangChain tools, SymPy (already used), Instructor or Pydantic for schemas |
| 5 | [Tool inputs are untrusted](05-tool-inputs-are-untrusted.md) | merged | allow-list parsers; AST validation; subprocess isolation with hard kill; containers; gVisor and microVM sandboxes; WebAssembly; provider-hosted code execution; credential isolation; prompt injection through tool output | RestrictedPython, E2B, Pyodide, nsjail, Firejail, Docker |
| 6 | Hybrid retrieval and relevance thresholds | planned | keyword (BM25); dense vectors; hybrid with rank fusion; rerankers (cross-encoders); query rewriting and HyDE; chunking strategies; metadata filters; relevance thresholds and "no answer"; retrieval evals (recall@k, MRR, nDCG, faithfulness) | Ragas, rank-bm25, sentence-transformers, pgvector, Qdrant, Chroma, LlamaIndex, LangChain retrievers |
| 7 | Just-in-time retrieval | planned | always-retrieve RAG; tool-called retrieval (agentic RAG); pre-fetch by a router; memory and notes files; caching; when the model should decide to look things up | LangChain or LlamaIndex agents, LangGraph, MCP resources |
| 8 | Context is a budget | planned | sliding window; running summary; tool-result clearing; structured note-taking and memory; sub-agents with clean context; prompt caching; token counting and budgets | tiktoken or tokenizers, LangChain memory classes, mem0, Letta |
| 9 | Every loop needs an exit | planned | max-turn and max-tool caps; timeouts; retries with backoff and jitter; circuit breakers; fallbacks to cheaper or different models; graceful degradation; idempotency; spend caps | tenacity, LiteLLM router fallbacks, pybreaker |
| 10 | Make silent failures loud | planned | lint and static checks on references and prompts; contract tests; schema validation of model output; canaries and alerts; failure budgets | Pydantic, Guardrails AI, pytest plugins, pre-commit |
| 11 | Guardrails and the spend cap | planned | input moderation; output filters (answer leak, PII); policy classifiers; structured-output validation; rate limits; per-user and global spend caps; human escalation | NeMo Guardrails, Guardrails AI, LLM Guard, Presidio, OpenAI moderation API |
| 12 | What we deliberately skipped, and why | planned | fine-tuning versus prompting; multi-agent frameworks; heavy platforms; when not to adopt | names each skipped option and the reason |

Rules the roadmap runs on:

- **Open-source adoption.** Adopt a library in code only when the concept genuinely needs it and it teaches better than the hand-rolled version. A tool that adds a new running service (for example a self-hosted tracing server) or a heavy dependency needs the owner's explicit yes before it is built. Every chapter still names the options, adopted or not, and says why.
- **Live runs.** Live proofs and baselines use a free OpenRouter model until the owner says otherwise. Each results file names the model and says it is not the product default; baselines are re-run when the paid model is used.
- **Content.** General public knowledge with public links only. No names or internals of any private system: this repository is public.

## The source posts

The practices come from Anthropic's engineering posts, all under <https://www.anthropic.com/engineering/>: [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents), [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents), [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) and [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). The chapters paraphrase them and link the post for the full argument.
