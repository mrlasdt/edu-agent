"""
Original IELTS Writing Task 2 prompts, prompt only.

Written for this repository, not taken from any exam or book. Items carry no response and no
labels, so they carry no score: the Writing grader selects only items with a response and a
band, and ignores these. They exist to give the product and future essay collection a
committed, licence-clean set of prompts across the four common Task 2 essay types.

Regenerate the committed file with `python -m evals.exam_set generate-writing`.
"""
from __future__ import annotations

from evals.exam_set.schema import Design, ExamItem, LabelProvenance, LabelTier, Source

FILE_NAME = "ielts_writing_prompts.jsonl"

# (essay type, prompt)
_PROMPTS: tuple[tuple[str, str], ...] = (
    ("opinion",
     "Some people say that all students should learn to cook and manage a household budget "
     "at school. To what extent do you agree or disagree?"),
    ("opinion",
     "Many cities are replacing car parking spaces with cycle lanes and public gardens. "
     "Do you think this is a positive development? Give reasons for your answer."),
    ("discuss_both_views",
     "Some people believe that university students should choose the subjects they enjoy, "
     "while others think they should study subjects that lead to well-paid jobs. Discuss "
     "both views and give your own opinion."),
    ("discuss_both_views",
     "Some people think that the best way to reduce crime is to give longer prison "
     "sentences. Others believe that education and job training are more effective. "
     "Discuss both views and give your own opinion."),
    ("advantages_disadvantages",
     "More and more people now work from home for part or all of the week. What are the "
     "advantages and disadvantages of this trend for workers and for their employers?"),
    ("advantages_disadvantages",
     "Tourism brings money to small towns but also changes the way local people live. "
     "Do the advantages of tourism outweigh the disadvantages?"),
    ("problem_solution",
     "In many countries, a growing number of young people find it difficult to buy or rent "
     "a home. What problems does this cause, and what can governments do to help?"),
    ("problem_solution",
     "Rubbish from packaging is becoming a serious problem in large cities. What are the "
     "causes of this problem, and what measures could be taken to reduce it?"),
)


def generate_all() -> list[ExamItem]:
    return [
        ExamItem(
            id=f"ielts_writing_prompt_{n}",
            exam="ielts",
            skill="writing_t2",
            task_type="essay_prompt",
            prompt=prompt,
            label_provenance=LabelProvenance(
                tier=LabelTier.author_key,
                date="2026-10-24",
                note="prompt only; carries no score",
            ),
            source=Source(kind="original", license="original to this repository (MIT)"),
            design=Design(generator="llm_drafted"),
            extras={"essay_type": essay_type},
        )
        for n, (essay_type, prompt) in enumerate(_PROMPTS, start=1)
    ]
