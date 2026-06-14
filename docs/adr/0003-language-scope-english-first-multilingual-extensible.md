# Language scope: English-first, multilingual-extensible

The agent serves English-language students in v1. Multilingual support (additional languages including but not limited to Vietnamese) is a v2 commitment, and v1 design decisions must not foreclose it. Technology choices that touch language — embeddings, chunking, tokenization, prompt assembly, style guides, eval rubrics — must be language-agnostic enough that adding a new language is additive, not a rewrite.

## Status

Accepted. Supersedes the implicit Vietnamese-first scope of earlier glossary entries; CONTEXT.md updated accordingly.

## Considered options

- **Vietnamese-first** — rejected: ties scope and content choices to a specific national curriculum (MOET 2018), with corpus-curation effort and political nuance that doesn't earn its place in a portfolio MVP.
- **Multilingual from day one** — rejected for the MVP: doubles the corpus-curation cost, multiplies eval-rubric design, and the language-routing logic is not interesting until at least one language ships well.
- **English-first, multilingual-extensible (chosen)** — keeps corpus work tractable for v1 and forces the architecture to be honest about i18n from the start.

## Consequences

- The embedding model must be multilingual-capable even though only English content lands in v1 (so v2 doesn't require a re-embed of the corpus). See ADR-0004.
- Prompt templates, system prompts, and style rubrics live behind a locale key from the start. v1 has one locale (`en`); v2 adds more.
- The Personal style profile (see CONTEXT.md) is language-agnostic — stylometric features are language-neutral by construction.
- Math correctness logic is language-independent (sympy operates on symbols, not natural language). Word-problem parsing is language-specific and must be locale-aware.
- The eval golden set is currently English-only; the v2 milestone for any new language is "a parallel golden set exists for that language."
