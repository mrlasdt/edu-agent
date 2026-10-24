# IELTS golden set: source survey

This file records which public sources were surveyed for graded IELTS material, the label tier of each, and what was done. It contains no third-party text, essays or scraped content. The committed repo contains no third-party data; everything imported lives in the gitignored `evals/golden/local/`, and `manifest.json` there records each source's tier, counts and hashes.

## Tier ranking

examiner > teacher_graded > captain_blind > site_or_model_weak

See `docs/adr/0013-ielts-sat-golden-set-label-tiers.md` and `evals/exam_set/schema.py`.

## Sources

| Source | Content | Tier | Outcome |
| --- | --- | --- | --- |
| Official IELTS sample scripts (mirror PDFs) | Writing scripts with examiner bands | examiner | Not imported. Candidate scripts are handwritten image fragments (some JBIG2) and Task 1 prompts are charts; needs handwriting OCR. |
| IELTS Buddy essay samples, bands 4-8 | Task 2 essays with per-criterion bands | teacher_graded | Imported, 19 essays (`importers/ieltsbuddy.py`). |
| writing9 on Hugging Face | Task 2 essays with site scores | site_or_model_weak | Imported, 1200 sampled (PR 5). |
| nlpatunt/D_Ielts_Writing_Task_2_Dataset | LLM-written evaluations | site_or_model_weak | Skipped. Same family as the previously rejected chillies dataset. |
| ieltspodcast.com | Sample essays | unknown | Skipped. Returns 403. |
| essaysmith.com | Sample essays | unknown | Skipped. Band ranges only, no exact bands. |
| Project owner's real Cambridge IELTS books (Google Drive) | Examiner-banded samples and comments | examiner | Not imported. Reads hang on the Drive mount. |

## Status against the 30-item target

Met locally across tiers: 19 teacher_graded plus 1200 weak-label writing9 items, and 0 examiner items. Only the weak and teacher tiers are populated, so examiner-tier agreement cannot yet be reported.

## Follow-ups (not part of this PR)

- Task 1 sources.
- Reading and Listening answer-key importers.
- Official examiner scripts (needs handwriting OCR).
- The real Cambridge books, once readable.
