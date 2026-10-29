# IELTS Writing mock: architecture, calibration gates and estimate framing

The owner wants to sit a full IELTS mock with the agent and get a score with detailed feedback to improve. ADR-0013 gave the project an IELTS Writing Task 2 grader target and small golden sets, but nothing a learner could use: the grader lived only in the eval harness, there was no timed attempt, and the one live run on weak labels compressed bands to 5.5-6.5 with a +0.75 bias. This ADR records the first user-facing slice (Writing Task 2) and, more importantly, when its band may be called calibrated.

## Status

Accepted. Implemented for Writing Task 2 only. Reading, Listening, Task 1, Speaking and the full-mock orchestration are later slices. The decision gate below has **not** passed, so the UI says "Uncalibrated estimate".

## Decision

1. **Estimate framing.** The product says "IELTS mock" and every score is a "mock score" (an estimate, never an official band). Every score carries the fixed line "Mock score: an estimate, not an official IELTS result." in the UI and public docs. The band is shown with a plausible range (plus or minus half a band around the overall) and a confidence note from the agreement between marking passes. Timing follows the computer-delivered test (40 minutes for Task 2, minimum 250 words). The owner sits **General Training**, so that is the variant the copy names. Task 2 is the same task and the same four criteria in both variants, so there is no variant switch and the grader prompt is shared; the variant matters only for the later Task 1 (a letter) and for Reading. The prompts are original to the repo and no Cambridge material is imported (the owner chose to skip the Cambridge books entirely).
2. **A separate `/ielts` API, not `/chat`.** A mock is an attempt with a lifecycle (created, started, submitted, marked), not a conversation turn. The gateway exposes `POST/GET /ielts/attempts`, `/start`, `PUT /draft` (autosave), `/submit`, `/regrade` and `GET /report`, plus `GET /ielts/prompts`. Attempts live in a local SQLite file under `IELTS_DATA_DIR` (default `ielts-local/`, gitignored, a bind mount in Compose). There is no auth because the mock is single-user and local; a hosted version needs auth and per-user storage first.
3. **The server owns the clock.** `start` stamps `started_at` and `deadline_at = started_at + 40 min`. Refreshing the page resumes the same attempt with the remaining time computed by the server. A submission after the deadline plus a 30 s grace is stored with `late = 1`, the text is the last autosave at the deadline, and the report says so. The browser countdown is a display only; when it reaches zero the client submits.
4. **The grader moved into production code** (`services/agent/src/agent/ielts/`) and the eval harness imports it, so the thing measured is the thing shipped. A `Completer` seam (an async function from messages to text) lets tests, the offline fake and the live runs inject the model. The production grader:
   - takes the **median of three samples** per criterion (`IELTS_GRADER_SAMPLES`), snaps to half bands and computes the overall in code with IELTS rounding;
   - runs **deterministic checks** first (word count, paragraph count, prompt overlap, copied-prompt share, repetition, non-English text, text addressed to the marker) and caps criteria for failures that a fluent essay can hide;
   - passes the essay as **delimited data** between markers derived from a hash of the essay (so an essay cannot guess its own marker) and tells the model it is untrusted;
   - never silently truncates: an essay over `DRAFT_MAX_CHARS` is rejected by the API as invalid, not cut;
   - returns an explicit `ungraded` result with a reason when the model fails or all samples are unusable, never a default band.
5. **The feedback generator** writes the report (criterion rationales, what holds the band back, what would reach the next half band, evidence quotes, error type counts, paragraph map, ranked actions) from a separate prompt, and the report is **mechanically checked**: every quote must be a verbatim span of one paragraph of the essay, error quotes longer than 10 words are dropped, verdicts may not be bare numbers, and a report with unverifiable parts is marked `feedback_status: partial` and the UI says it is incomplete. A feedback failure never removes the band.
6. **Heuristic caps are a policy, not a measurement.** Under 150 words caps TR at 4, under 250 words caps it at 5.5; off-topic, copied-prompt and repetitive responses cap TR at 4.0 and the other criteria at 5.5 (LR at 4.0 for repetition); text addressed to the marker caps TR at 5. Fewer than four paragraphs only raises an informational flag. No estimate is ever above 8.5 (`MAX_ESTIMATED_BAND`), because the model cannot be trusted to award the top of the scale. The thresholds are in `checks.py`. They were chosen by reading the descriptors, not fitted to labels, and they are documented as uncalibrated.
7. **Calibration gates.** There are two, and the difference matters.
   - **Development gate** (can reject a grader, cannot certify one): on the labelled set at hand, within-0.5 of the label at least 0.75 and |bias| at most 0.3. It decides whether prompt and model iteration is worth continuing.
   - **Decision gate** (the only thing that lets the UI drop "uncalibrated"): at least 60 essays; label tier `examiner` or `captain_blind`; overall within-0.5 at least 0.80 with a Wilson 95% lower bound at least 0.70; quadratic weighted kappa at least 0.75; |bias| at most 0.25; calibration slope between 0.8 and 1.2; within-0.5 at least 0.65 in every band bucket; within-0.5 at least 0.70 for each criterion; test-retest spread at most 0.5 band for at least 90% of essays; and the owner's own intra-rater agreement at least 0.8 (the owner re-scores a subset, so label noise is known). Implemented in `gate.py` and printed by `python -m evals.agent_eval --suite ielts_writing`.
   - `config/ielts_calibration.json` records the result. The report reads it; if `decision_gate_passed` is not true, the band is shown as "Uncalibrated estimate" with the measured numbers in the note.
8. **A local blind pool is how the decision gate gets its labels.** `python -m evals.exam_set build-blind-pool` writes `ielts-local/blind/` (gitignored): 41 model-drafted essays at target bands from 4.0 to 8.0 plus the 19 instructor-graded essays, shuffled under opaque ids, with the key in a separate file the owner must not open until finished. The owner scores `essays.md` into `scores.csv`; `import-blind-scores` writes `captain_blind` items into `evals/golden/local/`. Each pool's ids come from a random salt recorded in its key file, so pools never share ids, and `import-blind-scores` merges into the imported file by id instead of replacing it. Every pool carries the instructor-graded essays, so the import also de-duplicates on normalised essay text against the items already imported: the first imported score wins and later copies are skipped (the import prints how many). Rebuilding is refused once `scores.csv` holds any score, because a rebuild would overwrite scores not yet imported. Single-rater, not an examiner, which is why the gate also needs the intra-rater check.

## Owner decisions of 2026-10-29

- **Variant: General Training.** Wording above. **Cambridge: skipped**; all content is original. **Mock, not official:** the product says "mock" and "mock score". **Storage: local SQLite.** **Timings: computer-delivered** (40 minutes for Task 2 inside the 60-minute Writing module; Task 1 is a later slice).
- **Speaking (a later slice)** shows a partial score with the unassessed criterion clearly marked partial.
- **Blind scoring: 30 essays**, not 60. Thirty can reject a grader and measure how the owner compares with the instructor's scores, but the decision gate keeps its n of 60: a gate that a smaller set could pass would not be a gate. The default pool is therefore 30 essays (18 model drafts across bands 4.0 to 8.0 and 12 instructor-graded essays spread over their bands) and a second pool of 30 later gets to 60.
- **AI spend cap: 5 USD total.** The existing budget ledger (`shared/src/shared/budget.py`, formerly under `evals/`) now defaults to 5 USD and is shared by the evals and the product. The production completer checks every call against the cap before it is made (the whole prompt plus `max_tokens` of output, plus the calls already in flight, since the three samples run together) and records the reported usage after it. Free models cost nothing. When less than 0.05 USD is left (about two mocks at the default model), `POST /ielts/attempts` and `/regrade` answer 402 with a message saying the cap is used up and how to raise it (`AI_SPEND_CAP_USD`), and `GET /ielts/budget` tells the page how much is spent so it can say so up front. If the cap is reached while the candidate is writing, the essay is kept and the attempt is `ungraded` and retryable with the same message. A budget stop is never reported as a model outage.
- **Public band descriptors** are downloaded to the gitignored `ielts-local/descriptors/` for the owner to score from (not committed, not injected into the prompt: the rubric prompt stays an original paraphrase). The tests assert that the local-only directories and the ledger are ignored and hold nothing tracked.

## Measured results

All on 2026-10-29, grader prompt `ielts_grader/writing_t2/v1`, three samples per essay, tuned on nothing (one prompt version, no iteration against these labels).

| Run | Labels | n | within 0.5 | bias | QWK | slope | worst bucket | Spend |
|---|---|---|---|---|---|---|---|---|
| Free chain (OpenRouter free router) | `teacher_graded` | 17 of 19 (2 unusable replies) | 0.47 | -0.79 | 0.53 | 0.59 | 0.00 (7.5+) | 0 |
| gpt-4.1-mini | `teacher_graded` | 19 | **0.84** | **-0.03** | **0.79** | 0.55 | 0.50 (7.5+) | about 0.07 USD |

- The **development gate passes** with gpt-4.1-mini (0.84 >= 0.75, |-0.03| <= 0.3) and fails with the free chain. This is why the dev config uses gpt-4.1-mini for both roles.
- The **decision gate fails** on 19 `teacher_graded` essays: n is below 60, the tier is not examiner or captain_blind, the Wilson lower bound is 0.62, the slope is 0.55 (the grader pulls scores toward the middle: the 7.5+ bucket is 0.50), per-criterion within-0.5 is 0.53 to 0.63, and the intra-rater check is unmeasured. Test-retest passes (spread at most 0.5 for every essay). A set of 19 cannot certify anything and the instructor's scores are one person's, so a pass here would not have changed the UI.
- **Adversarial set** (12 cases: five injection variants, off-topic, copied prompt, memorised template, too short, word salad, gibberish, non-English): 12 of 12 raise the expected flags and the highest overall given is 5.0, so none reaches 9.
- **Report structure** on 5 real essays: 5 of 5 pass `check_report` with `feedback_status: complete`.
- Live spend for the slice, including drafting the blind pool (about 0.02 USD) and development runs: about 0.15 USD (gpt-4.1-mini about 0.11, the rest earlier smoke runs), recorded in `.eval-ledger/ledger.json` (ceiling 20 USD). One mock costs about 0.01 USD with gpt-4.1-mini.

## Consequences

- The band is a consistent yardstick across attempts and a good pointer to weaknesses. It is not a prediction of an exam score: it compresses the top and bottom of the scale, and nothing here was checked against examiner marks. The PR and README say this plainly.
- Moving the grader into `services/agent` means the eval harness, the API and tests share one implementation; a prompt change invalidates the recorded result and the gate file must be re-measured.
- Free models are fine for development of the plumbing and unusable for the band: reasoning models need a large completion cap (a 300-token cap returned empty replies) and the free router returned unusable replies for 2 of 19 essays.
- gpt-4o-mini was tried first for the feedback writer and flagged correct text as errors; the verbatim-span checks catch invented quotes but not wrong judgements, so the writer is gpt-4.1-mini and every error quote is capped at 10 words.
- Runtime spend is metered (see "AI spend cap" below). The ledger is a local JSON file for one user; a hosted version needs a shared store and per-user caps.
- Test-retest is measured on three samples of one model, so it shows stability, not accuracy.
- The prompts are original to the repo and the rubric paraphrases the public criteria; it does not reproduce the official band descriptors.

## Next slices

1. Owner scores the 30-essay blind pool, the grader is measured on it as a development check, and a second pool brings the decision gate to n of 60 (`captain_blind`); if the slope stays below 0.8, add a calibration map fitted on the dev half and checked on the test half, or try a stronger model.
2. Task 1 (charts and letters), then Reading and Listening (objective, machine-keyed), then Speaking (audio transcription and fluency features).
3. Full-mock orchestration across sections with a combined overall band, and tutoring chat grounded in the attempt report.
4. Auth and per-user storage and spend caps before any hosted deployment.
