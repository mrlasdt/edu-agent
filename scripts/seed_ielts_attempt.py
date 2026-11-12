"""
Write one graded IELTS Writing attempt with a canned report, so the chat coach can be tried
without sitting a 40-minute mock. No model call and no spend.

    uv run python -m scripts.seed_ielts_attempt

The attempt goes into the gateway's attempt store (`ielts-local/attempts.sqlite3`, or
`IELTS_DATA_DIR`), so it also shows in the history at /ielts-mock. The essay is a made-up
mid-band one whose weakness is mechanical linking: the report's Coherence verdict quotes it.
"""
from __future__ import annotations

from pathlib import Path

from services.gateway.src.gateway.ielts_store import ALLOWED_S, Attempt, AttemptStore
from services.gateway.src.gateway.settings import get_settings
from shared.src.shared.ielts.checks import words
from shared.src.shared.ielts.models import (
    Action,
    Calibration,
    CriterionReport,
    ErrorSpan,
    EstimatedBand,
    Evidence,
    Priority,
    Report,
    TaskCheck,
    Timing,
)
from shared.src.shared.ielts.scoring import CRITERION_NAMES

PROMPT_ID = "seed-1"
PROMPT = (
    "Some people think that children should start learning a foreign language at primary "
    "school. Others think it is better to wait until secondary school. Discuss both views "
    "and give your own opinion."
)

ESSAY = (
    "Nowadays, many parents debate when children should begin to study a foreign language. "
    "In this essay I will discuss both views and explain why I believe an early start is "
    "better.\n\n"
    "Firstly, some people think primary school is too early. They say young children are still "
    "learning to read and write in their own language, and a second language could confuse "
    "them. Also, primary schools often have few teachers who speak a foreign language well. "
    "Furthermore, a child who is made to study too much can lose interest in school. "
    "Moreover, the time could be used for maths and science. For example, a six-year-old "
    "may still be learning letters in their own language, and a second alphabet or new sounds "
    "could slow that down.\n\n"
    "Secondly, other people think that children learn languages more easily when they are "
    "young. Moreover, they have no fear of making mistakes, so they speak freely. In addition, "
    "a child who starts early will hear the sounds of the language and copy them naturally. "
    "Furthermore, they will have many more years of practice before they leave school. "
    "Also, many jobs in the future will need a second language, and speaking one well is a "
    "skill that children can keep for life.\n\n"
    "In my opinion, children should start a foreign language at primary school, but only for "
    "short and enjoyable lessons such as songs and games. Therefore, they will not be "
    "overloaded and they will still gain the advantages of an early start. In conclusion, I "
    "agree with the second view, because the benefits are bigger than the problems."
)


def _evidence(quote: str, effect: str) -> Evidence:
    start = ESSAY.index(quote)
    para = ESSAY.count("\n\n", 0, start) + 1
    return Evidence(
        quote=quote, start=start, end=start + len(quote), para=para,
        effect="weakens" if effect == "weakens" else "supports",
    )


_also = ESSAY.index("Also, primary schools")


def build_report() -> Report:
    """A graded Task 2 report whose quotes are exact spans of `ESSAY` (it passes `check_report`)."""
    criteria = [
        CriterionReport(
            id="TR", name=CRITERION_NAMES["TR"], band=6.0, sample_bands=[6.0, 6.0, 6.0],
            verdict="Both views are covered and the opinion is clear, but the ideas stay general.",
            evidence=[
                _evidence("In my opinion, children should start a foreign language at primary "
                          "school", "supports"),
                _evidence("the time could be used for maths and science", "weakens"),
            ],
            holding_back="Points are stated but not developed with an example or a reason.",
            to_reach_next_band="Develop one point per paragraph with a specific example.",
            actions=[Action(do="Add one example to the 'few teachers' point.",
                            why="Examples turn a claim into support.")],
        ),
        CriterionReport(
            id="CC", name=CRITERION_NAMES["CC"], band=5.5, sample_bands=[5.5, 5.5, 6.0],
            verdict="Paragraphing is clear, but linking words are used mechanically.",
            evidence=[
                _evidence("Moreover, they have no fear of making mistakes", "weakens"),
                _evidence("Furthermore, a child who is made to study too much", "weakens"),
                _evidence("Firstly, some people think primary school is too early", "supports"),
            ],
            holding_back="Almost every sentence opens with an additive linker (Also, Furthermore, "
                         "Moreover), so linking is mechanical and the ideas do not build.",
            to_reach_next_band="Link by reference and logic (this, such, as a result) and let "
                               "some sentences follow without a linker.",
            actions=[
                Action(do="Rewrite paragraph 3 using at most two linking words.",
                       why="Fewer, better-placed links read as a stronger flow."),
                Action(do="Replace 'Furthermore' and 'Moreover' with a sentence that refers back.",
                       why="Reference links show how one idea follows another."),
            ],
        ),
        CriterionReport(
            id="LR", name=CRITERION_NAMES["LR"], band=6.0, sample_bands=[6.0, 6.0, 6.0],
            verdict="Adequate range with some repetition of 'children' and 'language'.",
            evidence=[_evidence("they will have many more years of practice", "supports")],
            holding_back="Topic words are repeated rather than varied.",
            to_reach_next_band="Vary the topic vocabulary (learners, pupils, second language).",
            actions=[Action(do="Swap two repeated 'children' for another word.",
                            why="Variety shows range.")],
        ),
        CriterionReport(
            id="GRA", name=CRITERION_NAMES["GRA"], band=6.5, sample_bands=[6.5, 6.5, 6.5],
            verdict="A mix of simple and complex sentences with few errors.",
            evidence=[_evidence("a child who starts early will hear the sounds of the language",
                                "supports")],
            holding_back="Complex sentences are mostly relative clauses.",
            to_reach_next_band="Add conditionals and passive forms where they fit.",
            actions=[Action(do="Write one conditional sentence per paragraph.",
                            why="It widens range without adding errors.")],
            error_table={"article": 1, "punctuation": 2, "word_form": 1},
        ),
    ]
    return Report(
        status="graded", word_count=len(words(ESSAY)),
        timing=Timing(allowed_s=ALLOWED_S, used_s=ALLOWED_S - 4 * 60, late=False),
        estimated_band=EstimatedBand(
            overall=6.0, range=(5.5, 6.5), confidence="medium", samples=3, sample_spread=0.5,
            calibration=Calibration(
                status="uncalibrated", label="Uncalibrated estimate",
                note="Seeded example: a canned report, not marked by a model.",
            ),
        ),
        criteria=criteria,
        error_spans=[ErrorSpan(
            start=_also, end=_also + 4,
            quote="Also", type="linking", note="Starts a sentence with an additive linker.",
            suggestion="Use a reference link instead.",
        )],
        task_checks=[
            TaskCheck(check="Answers every part of the question", passed=True),
            TaskCheck(check="Develops ideas with examples", passed=False),
        ],
        top_priorities=[
            Priority(priority=1, criterion="CC",
                     do="Keep at most two additive linkers (Also, Furthermore) a paragraph.",
                     why="They are the main thing holding Coherence at 5.5."),
            Priority(priority=2, criterion="TR", do="Give one example for each main point.",
                     why="Underdeveloped ideas limit Task Response."),
            Priority(priority=3, criterion="LR", do="Vary the repeated topic words.",
                     why="Repetition caps the range."),
        ],
        feedback_status="complete",
    )


def seed(store: AttemptStore) -> Attempt:
    """Store one graded attempt (essay and canned report) and return it."""
    attempt = store.create(PROMPT_ID, PROMPT)
    store.start(attempt.id)
    store.submit(attempt.id, ESSAY)
    store.begin_grading(attempt.id)
    return store.finish(attempt.id, build_report().model_dump_json(), graded=True)


def main() -> None:
    store = AttemptStore(Path(get_settings().ielts_data_dir) / "attempts.sqlite3")
    attempt = seed(store)
    print(f"Seeded graded attempt {attempt.id}.")
    print("Open http://localhost:5173/, choose IELTS Writing and ask about your report.")


if __name__ == "__main__":
    main()
