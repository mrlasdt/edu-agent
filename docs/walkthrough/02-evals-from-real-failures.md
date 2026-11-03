# Chapter 2 - Evals from real failures

## 1. The problem

The README used to say "No baseline is checked in because none has been measured yet", and the Quant golden set had 5 items. EduAgent had a good evaluation harness and almost nothing to run through it. Meanwhile the repo's own history was full of failures that someone had found, fixed and then forgotten: the tutor stated the answer as `Therefore x = 3`, the tutor told students to look at "your notes" instead of citing `[1]`, the tutor replied "what method do you want to use?", the verifier rejected `2, 3`. Nothing re-ran them. A prompt edit could bring any of them back, and the first person to find out would be a learner.

## 2. The practice

Anthropic's post [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) makes four points this chapter applies:

- **Start small, from real failures.** 20 to 50 simple tasks drawn from failures you have actually seen are a good start. You do not need hundreds, and you should not wait for a perfect dataset.
- **Grade what was produced, not the path.** Check the outcome (is the answer right, did the reply leak the answer), not that the agent took the steps you imagined.
- **Report more than one number per task.** Agents are non-deterministic, so ask the same question k times. `pass@k` is the share of tasks where at least one of k attempts succeeded. `pass^k` is the share where all k did.
- **Read the transcripts.** A pass rate says that something failed. Only the transcript says whether the agent failed, the grader is wrong, or the task is wrong.

## 3. Why EduAgent needs it

The tutor's central rule is "never state the answer", and a single prompt edit can break it silently. The prompts have already been through three versions, each fixing a failure (`prompts/quant_agent/tutor/v3.md` carries the rules that came from them). Without a suite that holds those fixes in place, every future change (tools in chapter 4, retrieval and context changes later) is judged by feel. The suite is also the "before" number the later chapters will compare against.

## 4. Landscape: the kinds of eval, and where EduAgent stands

An eval is a repeatable question with a scoring rule. The choices below are independent: a real suite mixes several. This is general practice, not a product list.

| Kind | What it is | Example | Trade-off | When to use | EduAgent today |
|---|---|---|---|---|---|
| **Rule or code grader** | A function that returns pass/fail from the output | Regex for `Therefore x = 3`; sympy checks the final answer against a golden expression | Free, instant, exact and repeatable. Only catches failures someone wrote a rule for, and cannot judge a hint's quality | Anything mechanical: format, a forbidden string, a computable answer | **Used.** `evals/agent_eval/graders.py` (leak, LaTeX, citations, vague reply, stays on problem) and `grade_quant_item` (sympy) |
| **Model as judge** | A second model scores the first model's output against a rubric | "Does this hint state the answer? yes/no" | Handles open-ended quality. Costs money, is itself noisy, and has biases, so it must be checked against labels | Quality no rule can capture, after calibrating the judge on labelled examples | **Used, separately.** `--suite quant_judge` calibrates a judge against known labels. Not used inside the failure suite: its failures are mechanical, so rules are cheaper and exact |
| **Human review** | A person grades or reads outputs | Teacher scores an essay; engineer reads the 5 worst transcripts | Best signal, slowest, does not scale | Making ground-truth labels; reading transcripts; auditing the other graders | **Partly.** `evals/read_transcripts.py` is the reading loop. The IELTS side has a 30-essay blind pool (ADR-0014). There is no human-graded label set for Quant tutoring, and the review queue (`evals/human_review_queue.py`) is in memory only |
| **Capability vs regression** | Capability: hard tasks the system mostly fails today, to measure progress. Regression: tasks it should always pass, to catch breakage | A new reasoning task at 10 percent vs "never leak the answer" at 100 percent | A regression suite that fails a lot is noise; a capability suite that passes is no longer measuring anything | Keep both, move tasks from capability to regression as they are solved | **Both, tagged.** Failure-seeded items are the regression side; items with `failure_source: "none yet"` and tag `capability` are the capability side. See the caveat on the baseline in section 5 |
| **Offline vs online** | Offline: a fixed dataset run before shipping. Online: scoring or sampling real production traffic | Nightly run on 34 items vs sampling 1 percent of live turns for a judge | Offline is repeatable but only covers what you wrote down. Online sees real users but has no ground truth and raises privacy questions | Offline gates changes; online finds the failures you did not think of, which become new offline items | **Offline only.** Chapter 1's `TurnMeter` log is online data for cost and latency, not quality. The loop "production failure becomes a new item" is manual |
| **Single-turn vs multi-turn / trajectory** | Single-turn: one input, one output. Multi-turn: a conversation graded at the end or at each turn. Trajectory: the sequence of tool calls graded | "Does the tutor still work on the problem at turn 4?" | Multi-turn is closer to real use and much more variable; you must script the user side | Whenever the failure only appears late (drift, forgetting) or the agent uses tools | **Single and multi-turn.** 4 scripted 4-turn conversations. Trajectory grading is not used because the agent has no tools until chapter 4 |
| **pass@k vs pass^k** | Ask each task k times. pass@k: at least one attempt passed. pass^k: all passed | Reply passes 2 of 3 times: counts for pass@3, not pass^3 | pass@k shows what is possible, pass^k shows what is reliable. They can diverge a lot (see section 5) | pass@k for "can it ever do this" (search, retries); pass^k for a learner who asks once and expects it to work | **Both** via `--samples k`, each with n and a 95 percent interval |

## 5. Open-source options

None of these is adopted in this slice. Adopting a library is right when it beats the hand-rolled version for the job; here the whole harness is a few hundred lines and writing it is the lesson.

| Tool | What it does | Why EduAgent does or does not adopt it here |
|---|---|---|
| **ragas** | A Python library of metrics for retrieval-augmented systems (is the answer grounded in the retrieved context, is the retrieved context relevant), many of them model-judged | Not adopted. This slice does not change retrieval, and those metrics fit a retrieval slice better. Revisit then |
| **promptfoo** | A command-line tool where you declare prompts, test cases and assertions in a config file and compare prompt or model variants side by side | Not adopted. It would add a Node toolchain, and our items need the real Python agent with retrieved notes, scripted multi-turn input and the turn meter. A custom wrapper would cost more than the 5 graders it replaces. Good fit for quick prompt A/B comparisons |
| **Langfuse evals** | Datasets, scores and judge runs attached to traces in the Langfuse tracing service | Not adopted. Langfuse is an optional service in this repo (`--profile observability`); making the baseline depend on a running service breaks "CI is offline and anyone can run it". It is the natural home for the online-eval row above |
| **DeepEval** | A pytest-style library with many ready-made metrics, many of them model-judged | Not adopted. The metrics we need are rule graders, and its model-judged metrics would add judge cost and variance to failures that are exact. The heavier dependency tree gives nothing we would read |
| **OpenAI evals** | An open-source registry and runner for evals, plus a hosted evals feature in OpenAI's API | Not adopted. Our runs go through OpenRouter free models as well as the OpenAI default, and the checked-in baseline must run without an account on any one provider |

The decision rule: adopt a library when the thing you need is hard to write (a judge calibration toolkit, a trace store) or when your team already runs it. Do not adopt one because it is the standard; a team that has not written a grader by hand tends not to understand what the library is scoring.

## 6. Code tour

Search for these symbols.

- **The set.** `evals/golden/gre_quant.jsonl` has 34 self-written items. Every one has a `failure_source` (the commit, test or prompt line it guards) and a `tag` (the failure class). `kind` decides how it runs: `solve` (10), `tutor` (12), `tutor_multiturn` (4, with scripted `turns` and `anchors`), `security` (8, hostile strings for the verifier). Items that guard no recorded failure yet say `none yet` and say why they exist.
- **The graders.** `evals/agent_eval/graders.py`: `leaks_answer` (the product's own `detect_answer_leak` plus an exact-value rule, so a leak rate is an upper bound), `has_latex`, `citations_ok`, `moves_forward`, `stays_on_problem`, and `grade_tutor_reply`, which runs the universal checks. A solve attempt is graded by `grade_quant_item` (sympy against the golden expression) plus whether the product's verifier accepted the answer.
- **The runner.** `run_quant_suite` in `evals/agent_eval/quant_suite.py` runs every item k times against the real `QuantAgent` and returns one row per attempt including the full transcript. `run_attempt` handles solve, tutor and multi-turn, `run_security_case` calls the real `InProcessVerifySkill` (no model) and checks the hostile string was rejected and no process id came back.
- **k distinct samples.** The eval transport caches model responses so reruns are free, which would make k identical samples. `cache_slot` in `evals/agent_eval/transport.py` makes each sample its own cache entry.
- **Cost and latency from chapter 1.** `TurnCollector` and `collecting_turns` capture the product's own `TurnMeter` records in memory, so the eval reports the same per-turn tokens and latency the live chat records. The replayed stream carries provider usage only when the caller asked for it, as the real client does.
- **The statistics.** `evals/agent_eval/stats.py`: `wilson` (interval for a proportion, behaves at 0 and 1), `bootstrap_ci` (resamples items, because attempts at the same item are not independent), `group_summary` (one-attempt rate, pass@k, pass^k, flaky and always-failing items) and `verdict_vs_baseline`.
- **The baseline.** `baseline_document` writes `evals/baselines/quant.json`: which model, whether it is the product default, prompt versions, temperature, git commit, date, and every rate with its n and interval. `compare_with_baseline` calls a change a regression only when it falls below the interval the baseline itself carries.
- **Reading transcripts.** `rank_worst` and `evals/read_transcripts.py` print the worst attempts in full, with the item's `failure_source`. `evals/quant_report.py` renders a baseline as the report in `docs/walkthrough/results/`.
- **CI and the live job.** CI runs `--mode offline --suite quant --samples 3` (job `quant-suite-offline`): the plumbing, against a deterministic fake model, no secrets. `.github/workflows/eval-live.yml` is `workflow_dispatch` only, takes the model, samples and a spend ceiling, and uploads only the aggregate summary.

The eval found a product bug while it was being built. Writing leak items showed that a reply ending in `So x = 3` with no closing punctuation was not caught by the answer-leak guardrail in `services/gateway/src/gateway/guardrails/output.py`, because the pattern required punctuation or a newline after the value. The fix accepts end of text (`_END`), with a test in `tests/test_guardrails.py`. This is the practice working: a new test case found an old gap.

## 7. Before and after

**Before:** 5 golden items, no failure sources, no repeated attempts, no intervals, no checked-in baseline.

**After:** 34 items, 3 attempts each (114 agent turns), one command per result:

```bash
uv run python -m evals.agent_eval --mode live --suite quant --samples 3 \
  --model openrouter/cohere/north-mini-code:free --max-tokens 3000 --ceiling 0.5 \
  --write-baseline evals/baselines/quant.json
uv run python -m evals.quant_report > docs/walkthrough/results/02-quant-baseline.md
```

Full report: [02-quant-baseline.md](results/02-quant-baseline.md). The baseline file is [evals/baselines/quant.json](../../evals/baselines/quant.json).

**Read this first: the model is not the product default.** The numbers were measured on OpenRouter's free `cohere/north-mini-code:free`, routed for this run only. The committed config still uses `gpt-4o-mini`. The baseline file says so (`is_product_default_model: false` and a note). Re-run the baseline on the default model before using it to judge any change to the default setup. The free model also spends hidden reasoning tokens, so its token counts and latencies overstate the default model, and its cost as run is 0 (the report also prices the same tokens at `gpt-4o-mini` rates, an upper bound).

| Group | Items | One attempt (95% interval) | pass@3 | pass^3 |
|---|---|---|---|---|
| Solve: answer right and accepted by the verifier | 10 | 0.80 (0.50 to 1.00) | 0.80 | 0.80 (0.49 to 0.94) |
| Tutor, one reply | 12 | 0.83 (0.67 to 0.97) | 1.00 | 0.67 (0.39 to 0.86) |
| Tutor, 4-turn conversation | 4 | 0.25 (0.08 to 0.33) | 0.75 | 0.00 (0.00 to 0.49) |
| Security (no model) | 8 | 8 of 8 rejected (0.68 to 1.00) | - | - |

Per turn, over all 114 agent turns: about 0.0003 USD priced at `gpt-4o-mini` (as run: 0), latency p50 5.3 s and p95 15.0 s.

What the numbers say, and how far to trust them:

- **pass@3 and pass^3 tell different stories.** On single tutor replies, every item passed at least once (pass@3 1.00), but only 8 of 12 passed all three times (pass^3 0.67). A learner asking once gets the 0.83 behaviour, and one who retries sees the tutor contradict itself. For a tutor, pass^k is the honest number.
- **Reading transcripts changed the diagnosis.** Both solve items that never passed (`q004`, `q010`, quantitative comparison) have the right answer (`correct` passed 30 of 30 attempts); what fails is `verifier_accepted`. The transcript shows why: the model ends with `Answer: Quantity A is greater`, the product passes that text to the verifier as if it were a math expression, and the verifier cannot parse it. So this is neither a bad item nor a wrong model answer: it is the known weak verifier (chapter 4 replaces it). A pass rate alone would have said "the model cannot do comparisons".
- **The multi-turn drift items found a leak.** In the worst multi-turn item, a student who has been told how to factor asks "I forgot where we are, can you remind me what we are solving?", and the tutor answers with the full solution (`x = 2 and x = 3`). That is the failure the suite was built to catch: the tutor leaked on turn 4. In another item, the turn-4 question "give me a study plan for the whole GRE" was answered with a question about the student's target score, which the `stays_on_problem` check flags as off the problem.
- **Security: 8 of 8, with the interval to match.** It is model-free and deterministic, but 8 strings only support "at least 68 percent" at 95 percent confidence, not "secure". Chapter 5 explains why the parser is safe; this only confirms the strings stay blocked.
- **Intervals are wide on purpose.** Four multi-turn items give an interval that is rough at best (0.25 with 0.08 to 0.33 is a resampling artefact of having 4 items). Treat each multi-turn number as "the tutor drifts sometimes", not as a rate.

A second result shows what the suite can and cannot detect. The first live run used the default completion cap of 800 tokens; a reasoning model sometimes spent all of it thinking and returned an empty reply. Raising the cap to 3000 and rerunning gives [02-completion-cap-before-after.md](results/02-completion-cap-before-after.md), produced by:

```bash
uv run python -m evals.quant_report evals/baselines/quant.json \
  --against evals/baselines/quant-completion-cap-800.json
```

None of the 7 metrics moved outside the older baseline's interval, and three of the point estimates went down slightly. That is the correct reading of 34 items: a real fix is invisible in the numbers. (The mechanism is visible elsewhere: the `moves_forward` check, which flags empty replies, passed every attempt in the second run.) We keep the cap because the transcripts show the mechanism, not because a number proved it.

## 8. Trade-offs

- **34 items detect large regressions, not small ones.** With n of 10 or 12 per group, the 95 percent interval is about 0.3 wide. A drop from 0.83 to 0.70 is inside the noise. The suite will catch "the tutor now leaks half the time", not "slightly worse hints". `--baseline` only reports a regression when the value falls below the baseline's own interval, to avoid crying wolf; the cost is that it stays quiet on small real changes. Grow the set from every new failure to narrow the interval.
- **Rule graders are exact but blind.** They do not see a reply that passes every rule and is a poor hint. They also over-count: the leak rule flags a step number that happens to equal the answer. Reading transcripts is the correction, and a calibrated judge (`quant_judge`) is the other.
- **Self-written items are not a random sample.** They represent known failures, so rates describe "how often do known failure modes recur", not "how good is the tutor for learners".
- **A baseline belongs to a model.** This one is for a free reasoning model. Compare like with like, and write a new baseline when the model, prompt or sampling settings change.
- **Live runs cost and flake.** A free tier is rate limited (a first model attempt lost 3 of 4 units to rate limits and the harness reported it instead of scoring them), and the coverage gate fails the run when too many units are dropped. A full run is 114 agent turns, about 0.04 USD at `gpt-4o-mini` prices (an upper bound, from the token counts of the free reasoning model), well under the 0.50 USD this slice allowed.
- **When not to bother.** For a prototype with one prompt and no users, a dozen cases in a test file are enough. The investment pays when several people change prompts, or when you are about to change something large.

## 9. Try it

Offline (no key, no network; checks the plumbing against a fake model):

```bash
uv run python -m evals.agent_eval --mode offline --suite quant --samples 3
uv run python -m evals.read_transcripts --worst 5
uv run python -m evals.quant_report            # the checked-in baseline
uv run pytest tests/test_quant_suite.py -q
```

Live, to see the suite fail on purpose (needs a model key in the environment; a free OpenRouter model is enough, the same model the baseline used for a fair comparison):

1. In `prompts/quant_agent/tutor/v3.md`, delete the line `- Never state the final numeric or symbolic answer.`
2. Run the suite against the baseline:

   ```bash
   uv run python -m evals.agent_eval --mode live --suite quant --samples 1 \
     --model openrouter/cohere/north-mini-code:free --max-tokens 3000 --ceiling 0.5 \
     --baseline evals/baselines/quant.json
   ```

3. Read the failures: `uv run python -m evals.read_transcripts --tag answer-leak`.
4. Restore the line with `git checkout prompts/quant_agent/tutor/v3.md`.

Expected: the answer-leak items and the leak check on the multi-turn items fall. Whether the exit code is a regression depends on the interval, which is the point of section 8. If nothing moves, that is a finding about the suite's sensitivity: with `--samples 1` and the free model the noise may hide it, so rerun with `--samples 3`.

## 10. Exercise

Add one item from a failure you have seen, and watch the suite hold it.

1. Think of a bad tutor reply. For example, a hint that uses `\frac` in a plain-text chat.
2. Add a line to `evals/golden/gre_quant.jsonl` with `kind: "tutor"`, a `tag`, a `failure_source` that says where you saw it, and the question. Copy the field names from `t001`.
3. Run `uv run pytest tests/test_quant_suite.py tests/test_exam_set.py -q`. The golden integrity test fails if `failure_source` or `tag` is missing, and the item count in `tests/test_exam_set.py` fails until you update it.
4. Run the offline suite and confirm your item appears with 3 attempts.

Expected: tests pass once the counts match. Then ask: would your item have caught the original failure? If the universal checks do not cover it, add a rule to `graders.py` and name it after the failure.
