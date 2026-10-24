# IELTS and SAT golden sets: label tiers and local-only third-party data

The HeyHi targets are IELTS, TOEIC and SAT, and the golden set only had 7 GRE items. Extending it raises two problems the GRE files never had: where the labels come from (nobody on the project is an examiner and nobody has time to hand-label essays), and what may be committed (edu-agent is a public repo, and the useful essay data is scraped or copyrighted).

## Status

Accepted. Implemented for IELTS Writing Task 2 (grader target and importers), SAT Math (templates), IELTS Reading (five original passages) and a prompt-only set of IELTS Writing Task 2 prompts. The committed set has more IELTS items (49) than SAT items (33). SAT Reading and Writing, IELTS Speaking and Listening, Task 1 charts and TOEIC are not done.

## Decision

1. **Every item carries a label tier**, and every metric prints the tier and the sample size it was computed against. Tiers: `examiner` (a band published by the exam owner), `captain_blind` (project owner, blind protocol, not an examiner), `site_or_model_weak` (scraped site or model scores), `solver` (sympy plus an independent second computation) and `author_key` (written by the item's author, an agent unless a human checked it). The schema is `evals/exam_set/schema.py`; the GRE files keep their own shape and are untouched.
2. **Third-party text is never committed.** Scraped or copyrighted items live only in the gitignored `evals/golden/local/`, with `manifest.json` recording each source's licence status, label tier, counts, band histogram and file hash. Committed code is importers, schema, templates, tests and tiny synthetic fixtures. Tests check that the directory is ignored, that nothing under it is tracked and that every committed golden file holds only `original` or `cc_by` items.
3. **Essay labels come from three places, strongest first, and the grader target uses the best tier present:**
   - the project owner's own Cambridge IELTS books (`examiner` tier, the book's published band and examiner comments, local OCR, built and tested on synthetic fixtures because no book files exist yet);
   - the writing9 corpus from Hugging Face (`site_or_model_weak`, downloaded at run time, large n, used for statistics and judge calibration only);
   - later, `captain_blind` scores of original essays if the owner chooses to label some.
4. **Objective items are machine-labelled.** SAT Math items are original templates; the key is computed by sympy and re-computed by a different method, and generation fails if they disagree. Distractors come from coded error models. The template id and seed are stored so each item regenerates bit for bit. Reading items are original passages with author keys that cite their evidence, and Writing Task 2 prompts are committed prompt-only (no response, no labels, so the grader ignores them).
5. **The IELTS Writing target** scores the four public criteria with an original rubric (the official descriptors are not reproduced). The overall band is computed from the four criterion bands in code. It reports within-0.5 rate, quadratic weighted kappa, exact match, MAE, bias and per-criterion agreement, against the best available tier.

## Consequences

- Agreement with `site_or_model_weak` labels is a noisy calibration signal, not agreement with an examiner. The dataset card reports about 1.4 bands of disagreement between two LLM graders and the site labels, so a low score may be label noise as much as grader error. Regression metrics are keyed by tier (`ielts_overall_within_half_band__<tier>`) so a weak-label baseline is never compared with an examiner one.
- The published sample scripts in Cambridge books are examiner-banded but the publisher warns they are not definitive examples of a band, and the book layouts vary; the parser is heuristic and must be checked on one real book before its output is trusted.
- chillies/IELTS-writing-task-2-evaluation was checked and not imported (one free-text band string per row, no per-criterion labels, no stated source or scorer, duplicates). The CC BY 4.0 EWCCE scripts were checked and deferred: the data is 200 scanned handwritten PDFs, so it needs handwriting transcription, and its task type and band spread are not published without downloading the files.
- Templated SAT items test the pipeline and the solver more than the nuance of real SAT items, and they are not a substitute for College Board material, which is not used. Reading keys are unchecked by a human, and the Not Given items in particular may be arguable.
- Licence: the owner decided licence is not a concern for personal use. The local-only rule still holds because the repo is public and anything committed is redistributed.
