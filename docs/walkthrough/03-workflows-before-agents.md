# Chapter 3 - Workflows before agents

This chapter documents code that already exists. Nothing in it is new; the point is to learn to see the patterns that are already there.

## 1. The problem

You open the code expecting "an AI agent" and find a tutor that makes one model call, a solver that loops at most three times, a question generator drawn as a graph, and an essay grader that asks the same question three times. It is fair to wonder why it is not one smart loop that decides everything. The answer is a design rule: use the simplest structure that does the job, and give a model control over the flow only when fixed steps cannot do it.

## 2. The practice

Anthropic's post [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) separates two things. A **workflow** is a system where your code decides the order of the steps and the model fills in each step. An **agent** is a system where the model decides the next step itself, in a loop, using tools. Workflows are predictable, cheaper and easier to debug; agents are for open-ended problems where you cannot write the steps in advance. The post lists a small set of workflow shapes that cover most needs: prompt chaining, routing, parallelization (including voting), orchestrator-workers, and evaluator-optimizer. Start with a single call, add structure only when a measurement says the single call is not good enough, and bound every loop.

## 3. Why EduAgent needs it

Every learner turn costs money and a learner waits for it. A fixed shape lets us say exactly how many model calls a turn can make (so cost and latency have a ceiling), where a wrong answer can be caught, and what happens when it cannot be fixed. Open-ended agent loops have none of those guarantees, and nothing in tutoring a quadratic needs one.

## 4. Code tour

Four of the five workflow shapes already exist. Search for the symbol names.

**Routing - `AgentRegistry`, `build_registry`, `Orchestrator.process`.** A turn's `(test_type, section)` pair picks the agent: `build_registry` in `services/agent/src/agent/bootstrap.py` registers `QuantAgent` for `("gre", "quant")` and `AWAgent` for `("gre", "aw")`, and `Orchestrator.process` in `services/gateway/src/gateway/orchestrator.py` looks the pair up. The model is not involved in routing. A miss does not raise; it returns a `ClarificationTurn`, a polite reply, and the agent service is never called. Adding a test type is one `register(...)` line.

**A bounded retry loop with a deterministic judge - `QuantAgent._solve_turn`.** Solve mode generates, asks the sympy verifier (`_verify`, through `VerifyMathSkill`) whether the final answer holds, and walks a fixed ladder: attempt 1, attempt 2 with the failure as context (`_MAX_RETRIES = 2`), then one attempt on the stronger `math_escalation` model, then degrade by emitting the best text with `verifier_fail: true`. Three model calls is the ceiling and the exit is explicit. Tutor mode (`QuantAgent._tutor_turn`) has no loop at all: one call, because a hint does not need checking the way an answer does.

**Evaluator-optimizer - `AWAgent._solve_turn`.** The essay is generated, then `run_style_critic` (a second, cheaper model) judges its structure, and on a failure the essay is rewritten once with the critic's feedback. A citation check (`check_citations`) can force one more regeneration, and if it still fails `strip_uncited_claims` removes the bad sentences rather than looping again. Generator, evaluator, a bounded number of rounds.

**A graph of fixed steps - `QuestionPipeline`.** In `services/agent/src/agent/pipeline/graph.py`, `_build` wires a LangGraph `StateGraph`: `plan` (an LLM chooses the question type), a router `_route` that sends the plan to one of three expert generators, `verify` (deterministic tools, including sympy recomputing the key), `critic` (an LLM reviews wording, only after the tools pass), then `accept`, or `revise` back to the expert, or `escalate` to the human review queue after `max_attempts` (`DEFAULT_MAX_ATTEMPTS = 3`). This combines routing with evaluator-optimizer. ADR [0012](../adr/0012-question-generation-pipeline.md) records why the key is recomputed by tools instead of judged by the model that wrote it.

**Parallel voting - `grade_essay`.** In `services/agent/src/agent/ielts/grading.py`, the IELTS marker runs the same grading request `samples` times (default 3, `IELTS_GRADER_SAMPLES`) at once with `asyncio.gather`, and takes the per-criterion `median_band` of the samples that parsed. One noisy sample cannot move the score far, and a sample that fails to parse is dropped instead of failing the mark. A separate pass of deterministic checks can then cap the score.

What is not here on purpose: no model decides which tool to call next and no loop runs until the model says it is done. Chapter 4 adds model-facing tools, with a bounded loop, because computing an answer is a job a fixed step cannot do.

## 5. Before and after

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

## 6. Trade-offs

- **More calls, more latency, more money.** The pipeline above makes 3.75 calls where a single call makes 1. Each added step must earn its cost against a measured defect.
- **Fixed shapes are rigid.** If a learner needs something the workflow did not anticipate, the code does not adapt. Routing returns a clarification instead of improvising.
- **A model judging a model is weak.** The strongest evaluator in this repo is deterministic (sympy recomputing a key). The style critic and wording critic are model judges and are only as good as their prompts.
- **When not to use a workflow.** If you cannot write the steps down in advance (open-ended research, debugging an unknown system), a bounded agent loop is the right tool. Even then, bound the loop and measure it.

## 7. Try it

```bash
uv run python -m evals.agent_eval --mode offline --suite generation --max-items 12 2>/dev/null | jq '.suites.generation.comparison'
uv run pytest tests/test_quant_agent_solve.py -q          # the retry, escalate, degrade ladder
```

## 8. Exercise

Predict, then check. In `services/agent/src/agent/agents/quant.py`, `_MAX_RETRIES = 2` controls how many attempts the solver makes on the first model before it escalates. If the verifier rejects every answer, how many model calls does a Solve turn make, and which of them uses the `math_escalation` role?

Expected: 3 calls (two on `quant_agent`, one on `math_escalation`), then the turn degrades with `verifier_fail: true`. `test_solve_degrades_gracefully_after_full_pipeline_failure` in `tests/test_quant_agent_solve.py` exercises exactly this path. After chapter 1, you can see the same count on a live turn in the `llm_calls` column of `python -m evals.turn_report --turns`.
