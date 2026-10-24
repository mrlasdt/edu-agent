# Question-generation pipeline: LangGraph graph with verifier tools and a bounded revise loop

Generating practice questions is a different job from tutoring one: a wrong key or an ambiguous stem ships to many Candidates, and nobody is in the loop to catch it. We generate items with a LangGraph `StateGraph` (`services/agent/src/agent/pipeline/`): an LLM **planner** picks the question type and constraints, a deterministic **router** sends the plan to one of three **generator experts** (numeric entry, multiple choice, quantitative comparison), **verifier tools** check the item, an LLM **critic** reviews wording, and failures loop back through **revise** up to a fixed number of attempts. Items that still fail are not emitted; they are escalated to the Human review queue. Every step appends a `TraceEvent`, so each item carries its own audit trail (plan, routing, every attempt's item, every check result, feedback, model, latency).

## Status

Accepted. Implemented for GRE Quant; no HTTP endpoint or UI yet.

## Considered options

- **A. One prompt, emit whatever comes back (the baseline).** Cheapest, and kept in code (`SingleCallGenerator`) as the "before" for evals. Rejected as the product path: it ships wrong keys, leaked answers and unparseable replies unchecked.
- **B. Single call plus deterministic verification, no loop.** Catches defects but turns every defect into a lost item.
- **C. Planner, expert routing, tools and a bounded revise loop (chosen).** Feeding the failed checks back to the generator recovers most defective first attempts at the cost of extra calls. Expert prompts are narrower than one all-types prompt, and the planner can pick the type when the request does not.
- **D. LLM-only verification (judge the key).** Rejected as the gate: a model grading its own key is the failure mode we want to remove. The solver recomputes the key with sympy through the existing verify-math skill; the LLM critic only judges wording, and only after the tools pass.
- **E. Escalate to a stronger model on the last attempt (as `QuantAgent` does).** Deferred. It needs a priced escalation role and a measured benefit; the bounded loop plus review queue is the v1 failure path.

## Consequences

- Items are structured so the key is recomputable: `expression` (a sympy expression or equation) for numeric and multiple-choice items, `quantity_a` / `quantity_b` for comparisons. Choices must be numeric values. This excludes items whose options are text or expressions we cannot evaluate; that is a deliberate limit of "verified" in v1.
- The tools are deterministic and independent of the generator: structure (option count, duplicates, select-all wording), leak (answer stated or marked in stem or options, every keyed value already in the stem), solver (recomputed key must equal the keyed answer; a comparison keyed D must genuinely change sign over an integer grid), and distractors (keyed options are exactly the solution values and no distractor is also correct). The critic is advisory about wording: an unparseable critic reply does not block an item that passed the tools.
- Retries are bounded (`max_attempts`, default 3). Worst case is 1 planner call + 3 generations + 3 critic calls. The extra calls over the baseline are reported by the eval as `extra_llm_calls_per_request` and `usd_per_valid_item`.
- All model traffic goes through `litellm.acompletion` with roles in `config/model_config.dev.yaml` (`question_planner`, `question_generator`, `question_critic`) and versioned prompts under `prompts/question_pipeline/`. The eval harness's metered transport therefore applies unchanged: free OpenRouter first, OpenAI fallback, and every real call checked against the budget ledger before it is made.
- The pipeline is evaluated by `evals/agent_eval` (suite `generation`): baseline and pipeline on the same requests, audited by the same tools plus an independent judge verdict. The audit shares the pipeline's tools, so the pipeline's shipped-defect rate is low by construction; the informative numbers are yield per request, the baseline's shipped defects, the judge's acceptance rate and the cost per valid item. Offline runs use a fake model with seeded flaws and prove the wiring, not model quality.
- Not done: dedup against the bank, difficulty calibration, a knowledge-base grounding check, persisting traces, an HTTP endpoint on the agent service, wiring the review queue to a store, and Verbal or AW generation.
