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

## Table of contents

Chapters are in learning order, not build order. A chapter without a link has not been written yet; it lands with the change that adds its practice.

| # | Chapter | Status |
|---|---|---|
| 0 | [How to use this guide](00-how-to-use-this-guide.md): service map and the five-minute demo | written |
| 1 | [Measure first](01-measure-first.md): what one tutoring turn costs and where its time goes | written |
| 2 | Evals from real failures, reading transcripts, regression versus capability | planned |
| 3 | [Workflows before agents](03-workflows-before-agents.md): routing, the question pipeline, evaluator-optimizer, 3-pass median | written |
| 4 | Tool design: compute, don't guess (will carry the landscape of tool types: client functions, provider-hosted tools, code sandboxes, computer use, bash and editor, memory, tool search) | planned |
| 5 | [Tool inputs are untrusted](05-tool-inputs-are-untrusted.md): allow-lists, subprocess isolation, and when you need a real sandbox | written |
| 6 | Hybrid retrieval and relevance thresholds | planned |
| 7 | Just-in-time retrieval: the model decides when to look something up | planned |
| 8 | Context is a budget | planned |
| 9 | Every loop needs an exit: stopping conditions and resilience | planned |
| 10 | Make silent failures loud: reference lint | planned |
| 11 | Guardrails and the spend cap | planned |
| 12 | What we deliberately skipped, and why | planned |
| A | Appendix: glossary, results index, how to rerun everything | planned |

## The source posts

The practices come from Anthropic's engineering posts, all under <https://www.anthropic.com/engineering/>: [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents), [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents), [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) and [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). The chapters paraphrase them and link the post for the full argument.
