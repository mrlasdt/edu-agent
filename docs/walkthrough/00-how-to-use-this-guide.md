# Chapter 0 - How to use this guide

Read this once. It gives you the map of the system, the five-minute demo that makes every later chapter concrete, and the habits that make the guide useful.

## How to read a chapter

- **Run before you read.** Every chapter has a "Try it" command. Run it first, then read the code tour with the output next to you.
- **Follow symbols, not line numbers.** The tour names files and symbols (`TurnMeter`, `QuantAgent._solve_turn`). Search for the name; the line will have moved.
- **Trust the result files, not the prose.** Numbers in a chapter come from a file under [results/](results/) that a named command wrote. Rerun the command to see whether it still says the same thing.
- **Do the exercise.** Each is a few lines of change with a result you can predict. Predict first.
- **Read the Landscape table before the code.** Each concept chapter opens with a table of all the ways to apply its practice, with a column saying whether EduAgent uses each one and why not. The code tour then shows the one that was chosen. The follow-on table of open-source options says which libraries were adopted, which were left out, and why. The full chapter template is in the [README](README.md#chapter-template).

## Map of the services

A learner types a message in the web app. One message is one **turn**. Here is everything a turn can touch:

```
 browser (web, :5173)
    |  POST /chat  (message, mode, history)
    v
 gateway (:8000)  --- the front door and the guardrails
    |  1. rate limit, redact PII in the message
    |  2. content safety          (OpenAI moderation API, no model tokens)
    |  3. completeness check      (one small model call)
    |  4. route by (test_type, section) with the agent registry
    |  POST /generate
    v
 agent (:8002)  --- QuantAgent or AWAgent
    |  a. retrieve notes  -------------->  corpus (:8001) --> qdrant, tei-embedder
    |  b. one or more model calls (the tutor, or the solver with retries)
    |  c. Solve mode: check the final answer with  ---->  math-verifier (:8090, sympy)
    v
 gateway again: scan the streamed reply (answer leak, citations, PII),
                one more small model call for the leak check in tutor mode
    v
 browser: the streamed answer
```

| Service | Directory | Role |
|---|---|---|
| web | `web/` | React app behind nginx; streams `POST /chat` into message bubbles |
| gateway | `services/gateway/` | `/chat`, the `Orchestrator`, all guardrails (`guardrails/`) |
| agent | `services/agent/` | `QuantAgent` and `AWAgent` (`agents/`), skills (`skills/`), critics, the question pipeline |
| corpus | `services/corpus/` | hybrid retrieval over the notes, with a three-tier access filter |
| ingestion | `services/ingestion/` | parse, chunk, embed, index documents; the `seed` job loads the demo notes |
| math-verifier | `services/math_verifier/` | sympy sandbox behind a small MCP server |
| shared | `shared/src/shared/` | config, model roles, budget and pricing, observability, per-turn measurement |
| evals | `evals/` | the offline and live evaluation harness, and the turn report |

Two things to notice, because later chapters lean on them:

- **A turn crosses two Python services.** The gateway owns the guardrails; the agent owns the tutor. Anything that wants "one number per turn" has to join their views (chapter 1 does this with a trace id).
- **Model calls are cheap here, and that hides cost.** Every role defaults to `gpt-4o-mini`, so a turn costs a fraction of a cent. The cost is still real once a turn makes several calls, and you only know how many it makes if you measure.

## The five-minute demo

You need Docker and an OpenAI API key (see the README quickstart for the one-time setup).

```bash
cp .env.example .env            # put OPENAI_API_KEY in it
docker compose -f infra/docker-compose.yml up -d --build
open http://localhost:5173      # pick Quant, type: Solve x^2 - 5x + 6 = 0
```

Ask it in Tutor mode and read the reply: it should give a hint and not the answer. Switch to Solve mode and ask again: it should give the full worked solution. Then see what those two turns did behind the scenes:

```bash
uv run python -m evals.turn_report --turns
```

That table is the first thing chapter 1 explains. Everything else in the guide is easier to read once you know how many model calls, tokens, dollars and milliseconds a turn really takes.

## Where to go next

Chapter 1 ([Measure first](01-measure-first.md)) first, because every later chapter reports its effect with that measurement. Chapter 3 ([Workflows before agents](03-workflows-before-agents.md)) explains why the code is shaped the way it is, using only code that already exists.
