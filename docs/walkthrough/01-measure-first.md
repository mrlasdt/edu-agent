# Chapter 1 - Measure first

## 1. The problem

You ask the tutor a question and it feels slow. Or someone asks "what does one tutoring turn cost?" and the honest answer is "a fraction of a cent, probably". Before this chapter's code, nobody could do better than that: the live chat made model calls that nothing counted. Only the evals and the IELTS mock recorded usage. You could not say how many model calls a turn makes, how many tokens it uses, or which step is slow. Every later claim in this guide ("this change makes answers better for X cents more") needs those numbers, so the first practice is to get them.

## 2. The practice

Anthropic's post [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) recommends measuring each task you run: its total runtime, how many tool calls it made, how many tokens it consumed, and how many calls errored, and then reading the transcripts to see why. The idea generalises to any LLM feature. Keep one record per task, not just an average, because averages hide the turns that cost ten times the others. And label what you measured versus what you guessed.

## 3. Landscape: the ways to measure an LLM feature

"Measure it" is not one technique. These are the main categories, from cheapest to heaviest. They stack: most real systems use several.

| Category | What it is | Example | Trade-off | When to use | EduAgent today |
|---|---|---|---|---|---|
| **Per-call usage metering** | Read the token counts the provider returns with each response and price them. | `usage.prompt_tokens` and `usage.completion_tokens` on a chat completion; the OpenAI `stream_options={"include_usage": True}` flag. | Exact for tokens, but one call is not one user task. Providers differ on whether streamed responses report usage. | Always. It is the raw material for everything else. | **Used.** `TurnMeter.stream` and `TurnMeter.complete` read usage; `estimate_tokens` is the labelled fallback. |
| **Per-task record** | One row per unit of user work (a turn, a job), joining every call, tool and step inside it. | A line such as `turn=t-9f2 llm_calls=3 in=1,840 out=212 cost=$0.0004 ttft=0.9s`. | You must define the task boundary and carry an id across services. Averages over tasks hide the expensive tail, so keep the rows. | As soon as one task makes more than one call. | **Used.** One JSON line per service per turn in `.turn-log/turns.jsonl`, joined by trace id in `evals/turn_report.py`. |
| **Structured logs** | Log events as key-value or JSON records with stable field names and a correlation id, not free text. | `{"event": "guardrail", "name": "pii", "result": "fail", "trace_id": "..."}`. | Cheap and universal, but you must design the fields and build the queries yourself. | Any service. The correlation id is the part that matters. | **Partly.** Guardrail events carry `trace_id` (`emit_guardrail_event`), and turn records are JSON; general application logs are plain text. |
| **Distributed tracing** | A tree of timed spans for one request across services, with propagated context. | An OpenTelemetry trace: `POST /chat` span with child spans for retrieval and each model call. | Needs instrumentation in every service plus a collector and a viewer. Excellent for "where did the time go". | Several services, or deep call trees you cannot hold in your head. | **Not used.** The gateway and the agent share a hand-passed `trace_id` and a join, which is enough for two services and one flat list of calls. |
| **LLM-specific tracing platform** | A tool that captures prompts, completions, token counts, cost and latency per call, with a UI to browse, filter and compare them. | Langfuse, LangSmith, Arize Phoenix, Helicone. | Rich UI and prompt/version views for little code, but it stores prompt text (a privacy question) and is one more service to run or pay for. | When people need to read individual transcripts, compare prompt versions or review sessions. | **Used, opt-in.** `setup_observability` turns on the LiteLLM-to-Langfuse callback when real keys are set; Langfuse runs under the `observability` compose profile. |
| **Metrics and dashboards** | Aggregated numeric time series (rates, percentiles) scraped and charted over time, with alerts. | Prometheus counters and histograms charted in Grafana: p95 latency, error rate, spend per hour. | Great for trends and alerting, poor for "why was this one turn slow". Needs a metrics store and dashboards to maintain. | A deployed service with traffic and an on-call. | **Not used.** [ADR-0010](../adr/0010-observability-mcp-ux.md) names Prometheus and Grafana, but the Kubernetes manifests were never applied, so nothing scrapes them. The report CLI prints p50/p95 instead. |
| **Cost attribution** | Splitting spend by feature, prompt version, tenant or user so you know what drives it. | Tagging each call with an agent name and prompt version, then grouping. | Needs tags on every call; per-user attribution also means storing an identity next to usage. | When more than one feature shares a bill, or you price per customer. | **Partly.** The record is grouped by agent and mode and carries the prompt version. There is no per-user attribution. |
| **Sampling** | Recording only some traffic (a fixed fraction, or only slow or failed requests) to bound storage and cost. | Head sampling (decide at the start) or tail sampling (keep the slow and failed traces). | Cheaper, but you can miss the rare turn you wanted to study. | High volume, or traces that carry large payloads. | **Not used.** Every turn is recorded because volume is one development machine and each record is a short line without message text. |
| **Provider invoice and usage dashboards** | The provider's own billing and usage pages. | OpenAI or OpenRouter usage dashboards. | The source of truth for money, but they cannot split cost by your feature or turn. | Reconciliation: check your own estimate against the bill. | **Not integrated.** `estimate_cost` is a price table, so treat it as a comparison tool and the invoice as accounting (see Trade-offs). |

## 4. Open-source options

Adopt a tool when it teaches or does something the hand-rolled version cannot. The per-turn record is deliberately hand-rolled (`shared/src/shared/turn_meter.py` is about 300 lines) because the point of this chapter is to see exactly what is measured. The tools below are what you would reach for next.

| Tool | What it does | EduAgent's position |
|---|---|---|
| [Langfuse](https://langfuse.com/docs) | Open-source LLM tracing and prompt-management platform; self-hostable; records every call with tokens, cost, latency and metadata. | **Adopted, optional.** Wired through the LiteLLM callback in `shared/src/shared/observability.py`, so it adds no code at the call sites. Off unless real keys are set, because it needs Postgres and a server (the `observability` compose profile). It complements the turn log: Langfuse answers "show me that transcript", the turn log answers "what does a turn cost on average". |
| [OpenTelemetry](https://opentelemetry.io/docs/specs/semconv/gen-ai/) plus [OpenLLMetry](https://github.com/traceloop/openllmetry) | The vendor-neutral tracing standard, with semantic conventions for generative AI calls, and a library that auto-instruments LLM SDKs. | **Not adopted.** Two services and a flat list of calls do not need span trees, and a collector plus a backend is a new running service. Adopt it when a turn spans more services or you must send traces to an existing observability stack. |
| [LangSmith](https://docs.smith.langchain.com) | Hosted tracing, datasets and evaluation from the LangChain team; works with or without LangChain code. | **Not adopted.** It is a hosted, closed-source service, and Langfuse already fills the same role in a self-hostable form. |
| [Arize Phoenix](https://github.com/Arize-ai/phoenix) | Open-source tracing and evaluation UI built on OpenTelemetry conventions. | **Not adopted.** Overlaps Langfuse. A reasonable alternative if you standardise on OpenTelemetry. |
| [Helicone](https://www.helicone.ai) | Proxy-style logging: send model traffic through it and get usage and cost for free. | **Not adopted.** [ADR-0010](../adr/0010-observability-mcp-ux.md) compared it with Langfuse and chose Langfuse for prompt versioning and comparison. A proxy also puts a new hop in the request path. |
| [LiteLLM proxy](https://docs.litellm.ai/docs/proxy/cost_tracking) | A gateway that adds spend tracking, budgets and keys per team in front of many providers. | **Not adopted.** EduAgent uses the `litellm` library only. The proxy is the next step if several apps or people share keys and need budgets enforced centrally. |
| [Prometheus](https://prometheus.io/docs/introduction/overview/) and [Grafana](https://grafana.com/docs/grafana/latest/) | Metrics storage with alerting, and dashboards. | **Not adopted.** Needs a deployed environment to scrape; local-only today. |

No new dependency was added for this chapter: the optional adoption already exists, and nothing else on the list would teach more than the 300 lines it would replace.

## 5. Why EduAgent needs it here

A single "turn" (one learner message) crosses two services and can make several model calls: a completeness check, the tutor itself (or a solver with retries), and a leak check. You cannot see that from the outside. Chapters 3 to 9 add retries, tools and retrieval decisions, each of which changes the number of calls per turn; without this record you would be arguing from impressions.

## 6. Code tour

Search for these symbols.

- **`TurnMeter`** (`shared/src/shared/turn_meter.py`). One per service per turn. `TurnMeter.stream` wraps a streaming `litellm.acompletion`: it asks the provider for usage with `stream_options={"include_usage": True}`, swallows the usage-only closing chunk so callers are unchanged, and records time to first token. `TurnMeter.complete` does the same for a non-streaming call. `TurnMeter.phase` times a non-model step such as retrieval. `TurnMeter.annotate` attaches facts (notes retrieved, guardrail outcome). `TurnMeter.finish` appends one JSON line to the log and never raises: measuring must not break a turn.
- **Estimate fallback.** When a provider sends no usage, `_add_call` falls back to `estimate_tokens` (about four characters per token) and marks the call `estimated`. A record's `usage` field is `provider`, `estimate` or `mixed`, so a guess is never passed off as a measurement. A call that raised before any response (key rejected, connection error) records 0 tokens and 0 USD, keeping the call count and its `error`, so a failure is never billed as spend. USD comes from `estimate_cost` in `shared/src/shared/budget.py`, the same price table the IELTS spend cap uses.
- **The call sites.** `QuantAgent.run` creates the meter, times retrieval with `meter.phase("retrieval")`, and passes it to `_tutor_turn` and `_solve_turn`, where `meter.stream(...)` replaces the bare `litellm.acompletion(...)`. `AWAgent` does the same, and `run_style_critic` takes an optional meter. Solve's verifier call is timed as the `verify` phase.
- **The gateway half.** `chat` in `services/gateway/src/gateway/main.py` creates a gateway-side meter, passes it to `check_completeness` and `check_answer_leak` (the two guardrails that call a model), and records the turn's `outcome` (`ok`, `rate_limited`, `content_blocked`, `supportive_reply`, `clarification_requested`, `answer_leak_blocked`, `error`), non-blocking `flags` (`input_pii_redacted`, `citations_trimmed`, `answer_leak_flagged`, ...) and `first_byte_ms`, what the learner actually waits for.
- **One line per service, joined by trace id.** The gateway and the agent are separate processes, so each appends its own line to `.turn-log/turns.jsonl` under the gateway's `trace_id`. `TurnLog` appends and reads; `merge_turns` in `evals/turn_report.py` joins the two lines into one row per turn. Both containers mount the same `.turn-log` directory in `infra/docker-compose.yml`.
- **The report.** `python -m evals.turn_report` prints per-agent means and p50/p95 (nearest-rank percentiles via `percentile`), where a turn's time goes, guardrail overhead, outcomes and flags. `--price-as MODEL` recomputes USD from the recorded tokens at another model's prices.
- **The guard test.** `tests/test_turn_meter.py` runs the real agents and the real `/chat` endpoint against a fake model that reports known usage, and asserts the recorded calls, tokens and cost equal what the fake said (for example, a Solve turn where the verifier rejects everything must record 3 calls and exactly 3 times the fake's tokens).

Cost is recorded and not capped. The 5 USD cap (`Ledger`) still covers only the IELTS mock and the evals; a hard cap on `/chat` would stop the tutor mid-lesson.

## 7. Before and after

**Before:** nothing recorded. There was no way to answer "how many calls, tokens or seconds?" for a live turn.

**After:** the stack from the README was run and the README's example question (`Solve x^2 - 5x + 6 = 0`) was asked through `POST /chat`, the same request the web app sends. Three rounds of: a Tutor first turn, a Tutor follow-up with history, and a Solve turn; 9 turns in all. Then:

```bash
uv run python -m evals.turn_report --log docs/walkthrough/results/01-turns.jsonl --turns
```

Results: [01-turn-report.txt](results/01-turn-report.txt) (as recorded), [01-turn-report-priced-as-gpt-4o-mini.txt](results/01-turn-report-priced-as-gpt-4o-mini.txt) (same tokens priced as the default paid model), and the raw log [01-turns.jsonl](results/01-turns.jsonl) so you can rerun the report yourself. The log holds counts and timings only, never message text.

```
agent/mode           turns  calls  in tok  out tok   usd/turn  ttft p50  ttft p95  total p50  total p95
quant_agent/solve        3    2.0     769      795    0.00059     1,830     1,993      5,856      6,007
quant_agent/tutor        6    3.0    1027     2851    0.00186     4,952     8,482     21,376    150,013
```
(USD here is the priced-as-gpt-4o-mini version; times in ms.)

What the numbers say, and how far to trust them:

- **A Tutor turn makes 3 model calls, a Solve turn 2.** That is the most solid finding because it does not depend on the model. Only one of them is the tutor itself: the others are the gateway's completeness check (every turn) and its answer-leak check (Tutor mode only). Two thirds of a Tutor turn's calls are guardrails.
- **A turn costs a few hundredths of a cent on the default model** (priced: 0.0019 USD per Tutor turn, 0.0006 per Solve turn), and in this run most of a Tutor turn's cost and time was the guardrail calls, not the tutor (0.0014 of 0.0019 USD; 37.5 s of 43.2 s of model time per turn, in means). Treat that split as specific to this run; see the next point.
- **These figures come from a free reasoning model, not the default.** The live proof ran with the owner's OpenAI key rejected, so every role was pointed at OpenRouter's free `cohere/north-mini-code:free` for this run only (nothing in the committed config changed), and the OpenAI moderation call (which is not a model-token call) was skipped. The model spends many hidden reasoning tokens (a Tutor turn used up to 5,800 output tokens for a few sentences of reply) and the free tier is slow, so the token counts and latencies overstate what `gpt-4o-mini` would show, and the dollar cost as recorded is 0 because the model is free. The `--price-as gpt-4o-mini` file shows how the paid default would be computed: `(input_tokens x 0.15 + output_tokens x 0.60) / 1,000,000` USD from the price table in `shared/src/shared/budget.py`, applied to the recorded tokens. Because those tokens are inflated by reasoning, read the priced figure as an upper bound, and rerun on the real default to get the real one.
- **One turn is flagged as estimated.** A guardrail call in that turn came back after about 139 seconds without a usage report, so the meter fell back to the character-based estimate and the report marks the turn `*`. The turn's 150 s total is also partly the load script's own 150 s timeout. This is the fallback path working as designed, not a result to quote.

## 8. Trade-offs

- **Estimates are not measurements.** The fallback is about four characters per token and is wrong for code, math and non-English text. It is labelled for that reason; if estimated turns become common, fix the provider setup instead of trusting them.
- **Pricing is a table, not a bill.** `estimate_cost` can drift from real prices and falls back to a deliberately pessimistic price for unknown models. Use it to compare changes, and use the provider's invoice for accounting.
- **A file log is a development tool.** It suits one machine with a shared directory. A multi-replica deployment should emit the same record to a log pipeline or tracing system instead (Langfuse is already an optional sink for traces). The record shape is the part worth keeping.
- **Two writers, one join.** Joining by trace id means a turn that dies between services has only one half. The report shows such a turn with its outcome as `no gateway record` rather than hiding it.
- **Logging cost.** One short line per service per turn, no message text. Rotate or delete `.turn-log/` like any log, and do not log content here without thinking about privacy.
- **When not to bother.** A one-off script that makes a single call does not need a per-task record. A feature that makes several calls per task and serves real users does.

## 9. Try it

With the stack running and a few turns asked in the web app:

```bash
uv run python -m evals.turn_report --turns
uv run python -m evals.turn_report --json | jq '.summary.groups'
uv run pytest tests/test_turn_meter.py -q     # the known-usage guard tests
```

## 10. Exercise

Predict, then run. The checked-in log was recorded on a free model, so ask what the same tokens would cost on `gpt-4.1-mini`, whose prices are 0.40 USD per million input tokens and 1.60 per million output (see `_PRICES` in `shared/src/shared/budget.py`):

```bash
uv run python -m evals.turn_report --log docs/walkthrough/results/01-turns.jsonl --price-as gpt-4.1-mini
```

Expected: both prices are exactly 8/3 of `gpt-4o-mini`'s, so every USD figure is 2.67 times the priced-as-`gpt-4o-mini` report: 0.00497 USD per Tutor turn and 0.00158 per Solve turn. Then change one number in `_PRICES` and see the report move: the price table is the only place a dollar figure comes from.
