# Chapter 3 - Workflows before agents

This chapter documents code that already exists. Nothing in it is new; the point is to learn to see the patterns that are already there.

## 1. The problem

You open the code expecting "an AI agent" and find a tutor that makes one model call, a solver that loops at most three times, a question generator drawn as a graph, and an essay grader that asks the same question three times. It is fair to wonder why it is not one smart loop that decides everything. The answer is a design rule: use the simplest structure that does the job, and give a model control over the flow only when fixed steps cannot do it.

## 2. The practice

Anthropic's post [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) separates two things. A **workflow** is a system where your code decides the order of the steps and the model fills in each step. An **agent** is a system where the model decides the next step itself, in a loop, using tools. Workflows are predictable, cheaper and easier to debug; agents are for open-ended problems where you cannot write the steps in advance. The post lists a small set of workflow shapes that cover most needs: prompt chaining, routing, parallelization (including voting), orchestrator-workers, and evaluator-optimizer. Start with a single call, add structure only when a measurement says the single call is not good enough, and bound every loop.

## 3. Landscape: the shapes an LLM system can take

Between "one model call" and "a model that runs itself" there is a ladder. Climb it one rung at a time, and only when a measurement (chapter 1) says the current rung fails. The first six rows are the workflow shapes from the source post; the rest are what sits above and around them.

| Category | What it is | Example | Trade-off | When to use | EduAgent today |
|---|---|---|---|---|---|
| **Single call (the augmented LLM)** | One model call, possibly with retrieved context or a tool. | Answer a question using retrieved notes. | Cheapest and fastest; no structure to catch a bad answer. | Always start here. | **Used.** Tutor mode (`QuantAgent._tutor_turn`) is one call. |
| **Prompt chaining** | A fixed sequence where each call's output feeds the next, with optional checks between steps. | Outline, then check the outline, then write. | More calls and latency; each step is simpler and checkable. | A task that splits cleanly into steps. | **Partly.** The question pipeline runs plan, generate, verify, critique in order. |
| **Routing** | Classify the input and send it to a specialised path. The classifier can be plain code or a model. | Pick the agent by `(test_type, section)`. | Code routing is free and exact but only handles inputs you can name; model routing handles fuzzy inputs and adds a call and an error source. | Distinct input kinds that need different prompts or tools. | **Used.** `AgentRegistry` routes on a key in code. The question pipeline's `plan` step is a small model-routed choice. |
| **Parallelization: sectioning** | Run independent subtasks at the same time and combine them. | Run a guardrail alongside the main answer. | Lower latency, same cost; the subtasks must really be independent. | Latency-bound work with independent parts. | **Not used.** Guardrail calls run in sequence around the stream. |
| **Parallelization: voting** | Run the same task several times and combine the answers. | Median of three essay gradings. | Costs N times the calls; reduces noise and one-off failures. | A noisy judge where a wrong answer is expensive. | **Used.** `grade_essay` takes the per-criterion median of 3 samples. |
| **Orchestrator-workers** | A model decides at run time what subtasks exist, hands them to workers, and merges the results. | A coding task that touches an unknown set of files. | Handles open-ended decomposition; unpredictable cost and depth. | The subtasks cannot be listed in advance. | **Not used.** A GRE question has a known shape. |
| **Evaluator-optimizer** | One call produces, another judges, and the producer retries with the feedback, within a bound. | Essay, style critic, rewrite once. | Better quality for extra calls; the judge is only as good as its prompt unless it is deterministic. | A clear quality bar that a judge can check. | **Used.** `AWAgent._solve_turn`, `QuantAgent._solve_turn` (sympy as the deterministic judge) and `QuestionPipeline`. |
| **Autonomous agent loop** | The model chooses the next action each step, with tools, until it decides it is done or a limit stops it. | A research or coding agent. | The most flexible and the hardest to predict, cost and debug; needs stopping conditions. | Open-ended problems where steps cannot be written down. | **Not yet.** Chapter 4 adds model-facing tools in a bounded loop, with an explicit round cap. |
| **Multi-agent handoff** | Several model-driven agents with their own prompts and tools pass the task between them. | A triage agent handing to a billing agent. | Clean separation of concerns; more state to pass and more places to fail. | Distinct domains that each need their own tools and instructions. | **Not used.** Two agents exist, but the registry, not a model, chooses between them. |
| **Durable workflow engine** | Infrastructure that persists each step so a long job survives crashes, retries and waits. | A job that runs for hours and waits for a human. | Strong guarantees, but a service to run and a programming model to learn. | Long-running or human-in-the-loop jobs. | **Not used.** A turn takes seconds, and an escalated item is handed to a review queue instead of a job pausing to wait for a person. |

## 4. Open-source options

| Tool | What it does | EduAgent's position |
|---|---|---|
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Express a workflow as a graph of nodes and edges with shared state, conditional routing and loops. | **Adopted, narrowly.** `QuestionPipeline` is a `StateGraph` because that flow has real branches (route to one of three experts, loop back to revise, escalate). The graph makes the retry budget and the exits visible. It is not used for `/chat`: the rest of the system is a lookup and a bounded loop, where a graph would add a layer without clarifying anything. |
| [LangChain](https://python.langchain.com/docs/introduction/) | A large library of model, retriever and tool abstractions and ready-made chains. | **Not adopted.** `litellm` already covers provider abstraction, and the call sites are short enough to read directly. Adding LangChain's abstractions would hide the prompts and call counts this guide wants you to see. Reasonable if you need its many integrations. |
| [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) | Event-driven workflow steps, strong on retrieval-centred apps. | **Not adopted.** Overlaps LangGraph for what we need. |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) and other provider agent SDKs | A small framework for model-driven loops with tools, handoffs and guardrails. | **Not adopted.** EduAgent stays provider-neutral through `litellm` and keeps its own bounded loop so you can read it. Worth a look if you are committed to one provider. |
| [Pydantic AI](https://ai.pydantic.dev/) | Typed agents and structured outputs on top of Pydantic. | **Not adopted.** Our structured replies are small. A good fit if typed outputs become the main pain. |
| [Temporal](https://docs.temporal.io/) | A durable execution engine for long-running workflows. | **Not adopted.** No job outlives a request; it would add a server and a worker fleet. |
| [CrewAI](https://docs.crewai.com/) and [AutoGen](https://microsoft.github.io/autogen/) | Multi-agent frameworks where agents with roles converse to complete a task. | **Not adopted.** The post's advice is to find the simplest solution first; a fixed workflow already meets every requirement here. |

Net: LangGraph is the one framework in the code, used where a graph earns it. Everything else on this list solves a problem EduAgent does not have yet.

## 5. Why EduAgent needs it here

Every learner turn costs money and a learner waits for it. A fixed shape lets us say exactly how many model calls a turn can make (so cost and latency have a ceiling), where a wrong answer can be caught, and what happens when it cannot be fixed. Open-ended agent loops have none of those guarantees, and nothing in tutoring a quadratic needs one.

## 6. Code tour

Four of the five workflow shapes already exist. Search for the symbol names.

**Routing - `AgentRegistry`, `build_registry`, `Orchestrator.process`.** A turn's `(test_type, section)` pair picks the agent: `build_registry` in `services/agent/src/agent/bootstrap.py` registers `QuantAgent` for `("gre", "quant")` and `AWAgent` for `("gre", "aw")`, and `Orchestrator.process` in `services/gateway/src/gateway/orchestrator.py` checks the pair against the catalog of supported pairs in `shared/src/shared/catalog.py` (the gateway never imports agent code). The model is not involved in routing. A miss does not raise; it returns a `ClarificationTurn`, a polite reply, and the agent service is never called. Adding a test type is one `register(...)` entry and one catalog entry.

**A bounded retry loop with a deterministic judge - `QuantAgent._solve_turn`.** Solve mode generates, asks the sympy verifier (`verify_well_formed`, through `VerifyMathSkill`) whether the final answer is well-formed (it parses and evaluates; `verify_well_formed` compares the answer with itself, so a wrong but parseable answer passes), then runs the independent check (ADR-0015: one extra short model call writes the problem as an equation, the solver computes it, and a disagreement counts as a failed check; it costs one more call per solve turn whose answer is well-formed), and walks a fixed ladder: attempt 1, attempt 2 with the failure as context (`_MAX_RETRIES = 2`), then one attempt on the stronger `math_escalation` model, then degrade by emitting the best text with `verifier_fail: true`. Three solve calls (plus the independent check's one short call) is the ceiling and the exit is explicit. Tutor mode (`QuantAgent._tutor_turn`) has no loop at all: one call, because a hint does not need checking the way an answer does.

**Evaluator-optimizer - `AWAgent._solve_turn`.** The essay is generated, then `run_style_critic` (a second, cheaper model) judges its structure, and on a failure the essay is rewritten once with the critic's feedback. A citation check (`check_citations`) can force one more regeneration, and if it still fails `strip_uncited_claims` removes the bad sentences rather than looping again. Generator, evaluator, a bounded number of rounds.

**A graph of fixed steps - `QuestionPipeline`.** In `services/agent/src/agent/pipeline/graph.py`, `_build` wires a LangGraph `StateGraph`: `plan` (an LLM chooses the question type), a router `_route` that sends the plan to one of three expert generators, `verify` (deterministic tools, including sympy recomputing the key), `critic` (an LLM reviews wording, only after the tools pass), then `accept`, or `revise` back to the expert, or `escalate` to the human review queue after `max_attempts` (`DEFAULT_MAX_ATTEMPTS = 3`). This combines routing with evaluator-optimizer. ADR [0012](../adr/0012-question-generation-pipeline.md) records why the key is recomputed by tools instead of judged by the model that wrote it.

**Parallel voting - `grade_essay`.** In `shared/src/shared/ielts/grading.py`, the IELTS marker runs the same grading request `samples` times (default 3, `IELTS_GRADER_SAMPLES`) at once with `asyncio.gather`, and takes the per-criterion `median_band` of the samples that parsed. One noisy sample cannot move the score far, and a sample that fails to parse is dropped instead of failing the mark. A separate pass of deterministic checks can then cap the score.

What is not here on purpose: no model decides which tool to call next and no loop runs until the model says it is done. Chapter 4 adds model-facing tools, with a bounded loop, because computing an answer is a job a fixed step cannot do.

## 7. Before and after

The question pipeline has a single-call baseline kept in the code (`SingleCallGenerator`), so the cost of the workflow can be read against the plain call on the same requests. The result file comes from the offline harness, which uses a fake model that injects known flaws (a wrong key, a leaked answer, a duplicated option, non-JSON) on the first attempt:

```bash
uv run python -m evals.agent_eval --mode offline --suite generation --max-items 12 2>/dev/null \
  | jq '.suites.generation' > docs/walkthrough/results/03-offline-generation.json
```

Result: [results/03-offline-generation.json](results/03-offline-generation.json).

| 12 requests | single call (baseline) | workflow (pipeline) |
|---|---|---|
| valid items per request | 25% | 92% |
| shipped defects among emitted items | 70% | 0% |
| unparseable replies | 17% | 0% |
| escalated to human review | not applicable | 8% (1 request) |
| model calls per request | 1.0 | 3.75 |

Read it honestly. The fake model only fixes a flaw when the prompt carries revision feedback, so the table proves the wiring (routing, tools, revise loop, escalation) and shows what the loop costs in calls. It says nothing about how often a real model errs. The audit also reuses the pipeline's own checks, so the 0% is true by construction; `judge_accept_rate` (30% against 100%) is the independent number, and in offline mode the judge is derived from the audit. The honest takeaway is the last row: the structure buys safety at 2.75 extra calls per item. Chapter 1 gives you the tool to price that on the live path.

## 8. Trade-offs

- **More calls, more latency, more money.** The pipeline above makes 3.75 calls where a single call makes 1. Each added step must earn its cost against a measured defect.
- **Fixed shapes are rigid.** If a learner needs something the workflow did not anticipate, the code does not adapt. Routing returns a clarification instead of improvising.
- **A model judging a model is weak.** The strongest evaluator in this repo is deterministic (sympy recomputing a key). The style critic and wording critic are model judges and are only as good as their prompts.
- **When not to use a workflow.** If you cannot write the steps down in advance (open-ended research, debugging an unknown system), a bounded agent loop is the right tool. Even then, bound the loop and measure it.

## 9. Try it

```bash
uv run python -m evals.agent_eval --mode offline --suite generation --max-items 12 2>/dev/null | jq '.suites.generation.comparison'
uv run pytest tests/test_quant_agent_solve.py -q          # the retry, escalate, degrade ladder
```

## 10. Exercise

Predict, then check. In `services/agent/src/agent/agents/quant.py`, `_MAX_RETRIES = 2` controls how many attempts the solver makes on the first model before it escalates. If the well-formedness check rejects every answer, how many model calls does a Solve turn make, and which of them uses the `math_escalation` role?

Expected: 3 calls (two on `quant_agent`, one on `math_escalation`), then the turn degrades with `verifier_fail: true`. `test_solve_degrades_gracefully_after_full_pipeline_failure` in `tests/test_quant_agent_solve.py` exercises exactly this path. After chapter 1, you can see the same count on a live turn in the `llm_calls` column of `python -m evals.turn_report --turns`.
