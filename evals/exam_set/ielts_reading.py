"""
Original IELTS-style Reading items on five original passages.

The passages are written for this repository (not derived from any official test and not copied
from any source), so they are committed. Keys are author keys: written by the implementing agent
and NOT checked by a human, so the True/False/Not Given items in particular may be arguable.
Every key cites its supporting sentence, and tests check mechanically that the evidence occurs
in the passage and that each completion answer occurs in its evidence within the word limit;
multiple-choice and matching-heading keys must be one of the listed choices.

Regenerate the committed file with `python -m evals.exam_set generate-reading`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evals.exam_set.schema import Design, ExamItem, LabelProvenance, LabelTier, Source

FILE_NAME = "ielts_reading.jsonl"
WORD_LIMIT = 2


@dataclass(frozen=True)
class Row:
    suffix: str
    task_type: str
    prompt: str
    key: str
    evidence: str | None
    note: str = ""
    choices: dict[str, str] | None = None
    paragraph: int | None = None  # 1-based paragraph the evidence sits in (matching headings)


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    rows: tuple[Row, ...]


GREEN_ROOFS = (
    "Cities that grow upward often run short of ground-level space for parks, so some "
    "planners have started to look at rooftops instead. A green roof is a roof covered with "
    "soil and plants, and the idea is much older than most people assume: families in "
    "northern Europe laid turf over their wooden houses centuries ago to keep out the cold. "
    "Modern versions are thinner and lighter, built in layers that include a waterproof "
    "membrane, a drainage sheet and a shallow growing medium.\n\n"
    "The benefits are easiest to see during summer. A conventional dark roof can become far "
    "hotter than the surrounding air, whereas a planted roof stays cooler because the plants "
    "shade the surface and release water vapour. Buildings below therefore need less air "
    "conditioning, and the roof membrane is protected from sunlight, which can double its "
    "working life. Green roofs also hold back rainwater. In one study of a mid-sized city, "
    "planted roofs delayed the peak of runoff after a storm by about twenty minutes, giving "
    "drains more time to cope.\n\n"
    "There are drawbacks. A green roof costs more to install than a standard one, and the "
    "structure beneath must be strong enough to carry the extra weight of wet soil. "
    "Maintenance is also needed, since weeds and dry spells can damage thin plant layers. "
    "Supporters argue that these costs are recovered through lower energy bills, although the "
    "researchers who carried out the city study did not measure savings over more than five "
    "years. Critics point out that a roof garden open to the public raises insurance "
    "questions that a sealed roof does not."
)

TFNG_INSTRUCTION = (
    "Do the following statements agree with the information in the passage? Answer TRUE if "
    "the statement agrees, FALSE if it contradicts the passage, NOT GIVEN if there is no "
    "information on this."
)
COMPLETION_INSTRUCTION = "Complete the sentence using NO MORE THAN TWO WORDS from the passage."
MCQ_INSTRUCTION = "Choose the correct letter, A, B, C or D."
MATCHING_INSTRUCTION = (
    "The passage has three paragraphs. Choose the correct heading for the paragraph named in "
    "the question from the list of headings."
)
INSTRUCTIONS = {
    "tfng": TFNG_INSTRUCTION,
    "sentence_completion": COMPLETION_INSTRUCTION,
    "multiple_choice_single": MCQ_INSTRUCTION,
    "matching_headings": MATCHING_INSTRUCTION,
}


CITY_BEES = (
    "Beekeeping has moved from the countryside to the city in the last two "
    "decades. Hives now sit on office roofs, in school gardens and beside "
    "railway lines. Keepers say that cities offer a surprisingly good home "
    "because gardens, parks and street trees flower at different times of "
    "year, giving bees food from early spring to late autumn.\n\n"
    "Urban honey is not always better than rural honey, however. A survey "
    "of hives in one large city found that colonies in dense districts "
    "produced less honey per hive than those on the edge of town, because "
    "rooftop forage was patchy. The same survey recorded that the honey "
    "itself contained no more pollutants than honey from farmland. Some "
    "ecologists worry that the number of hives has grown faster than the "
    "supply of flowers, so that honeybees compete with wild bees for "
    "nectar.\n\n"
    "Several cities have therefore begun to limit hive numbers in busy "
    "parks. Wild bee specialists recommend that anyone who wants to help "
    "should plant a variety of flowers rather than keep a hive. Beekeeping "
    "groups reply that hives are valuable for education, since many new "
    "keepers become more interested in protecting nature in general."
)

SCHOOL_START = (
    "In many countries the school day begins before eight o'clock, even "
    "though teenagers tend to fall asleep later than children or adults. "
    "Sleep researchers explain that the body clock shifts during "
    "adolescence, so a fifteen-year-old who is asked to wake at six may be "
    "getting up at the equivalent of a much earlier hour for an adult. "
    "Surveys of students show that many of them sleep for fewer than eight "
    "hours on school nights.\n\n"
    "Several school districts have tested later start times. In one trial "
    "involving nine schools, classes began about an hour later than "
    "before. Attendance rose slightly, and teachers reported that fewer "
    "students fell asleep in the first lesson. The trial did not, however, "
    "produce a clear change in examination results, and the authors warned "
    "that two years was too short a period for judging academic effects.\n\n"
    "Not everyone supports the change. Parents who work early shifts say "
    "that a later start leaves their children unsupervised in the morning, "
    "while sports coaches point out that afternoon training sessions would "
    "end after dark. Bus companies add that rearranging routes costs "
    "money. Supporters reply that these problems can be managed, and that "
    "the health benefits justify the effort."
)

WIND_SHIPS = (
    "Most of the goods people buy travel at least part of their journey by "
    "sea, and the ships that carry them burn large amounts of fuel. A "
    "growing number of shipping companies are therefore testing a "
    "technology that seems to belong to the past: sails. The modern "
    "versions are not canvas but tall rigid wings or spinning cylinders "
    "that use the wind to push the ship forward.\n\n"
    "Engineers say that the main advantage of these devices is that they "
    "can be added to existing vessels. On a long route across an ocean, a "
    "ship fitted with rotating cylinders used about one tenth less fuel "
    "than a similar ship without them. The saving depends heavily on the "
    "route, since a ship sailing against steady winds gains little. "
    "Designers have also had to solve the problem of bridges and ports, "
    "because tall rigs must fold down or tilt to pass underneath.\n\n"
    "Cost remains a barrier. Fuel is cheap compared with the price of a "
    "new rig, and many owners wait to see how long the equipment lasts at "
    "sea. Some analysts believe that stricter rules on emissions will "
    "change this calculation within a decade. Others doubt that sails "
    "alone will be enough, and argue that they should be combined with "
    "cleaner fuels."
)

LANGUAGE_LEARNING = (
    "Adults who start learning a new language often assume that children "
    "have an unbeatable advantage. Research gives a more mixed picture. "
    "Young children do tend to acquire a native-like accent more easily, "
    "but older learners can progress faster in the early stages because "
    "they understand grammar rules and can study in a planned way.\n\n"
    "A study of two hundred immigrants found that the best predictor of "
    "fluency after five years was not the age of arrival but the amount of "
    "daily conversation in the new language. Those who spent at least two "
    "hours a day speaking with local people outperformed those who relied "
    "mainly on classes. The researchers suggested that classes work best "
    "when they are combined with real conversation.\n\n"
    "Technology has made conversation practice easier to find. Language "
    "exchange websites connect learners with partners abroad, and some "
    "apps offer short speaking tasks. Critics, however, observe that many "
    "app users stop within a month, and that games which reward streaks "
    "may encourage frequent but shallow practice."
)

PASSAGES: tuple[Passage, ...] = (
    Passage(
        "green_roofs", GREEN_ROOFS,
        (
            Row('tfng_1', 'tfng',
                (
                    "In the past, some families in northern Europe covered their houses "
                    "with turf for protection against the cold."
                ),
                'TRUE',
                (
                    "families in northern Europe laid turf over their wooden houses "
                    "centuries ago to keep out the cold"
                )),
            Row('tfng_2', 'tfng',
                (
                    "A planted roof is always cooler than a conventional roof in winter."
                ),
                'NOT GIVEN',
                None,
                note=(
                    "the passage discusses summer cooling only; nothing is said about "
                    "winter"
                )),
            Row('tfng_3', 'tfng',
                (
                    "The city study found that planted roofs made the peak of storm runoff "
                    "arrive earlier."
                ),
                'FALSE',
                (
                    "planted roofs delayed the peak of runoff after a storm by about "
                    "twenty minutes"
                )),
            Row('fill_1', 'sentence_completion',
                (
                    "Modern green roofs are built in layers that include a waterproof "
                    "membrane, a drainage sheet and a shallow ______."
                ),
                'growing medium',
                (
                    "a waterproof membrane, a drainage sheet and a shallow growing medium"
                )),
            Row('fill_2', 'sentence_completion',
                (
                    "Protection from sunlight can ______ the working life of the roof "
                    "membrane."
                ),
                'double',
                (
                    "the roof membrane is protected from sunlight, which can double its "
                    "working life"
                )),
            Row('fill_3', 'sentence_completion',
                (
                    "The structure beneath a green roof must be able to carry the weight "
                    "of wet ______."
                ),
                'soil',
                (
                    "the structure beneath must be strong enough to carry the extra weight "
                    "of wet soil"
                )),
            Row('mcq_1', 'multiple_choice_single',
                (
                    "Which problem does the writer say affects a roof garden open to the "
                    "public?"
                ),
                'B',
                (
                    "a roof garden open to the public raises insurance questions that a "
                    "sealed roof does not"
                ),
                choices={
                    'A': (
                    "It needs a stronger drainage sheet."
                ),
                    'B': (
                    "It raises insurance questions."
                ),
                    'C': (
                    "It is cooler than other roofs in summer."
                ),
                    'D': (
                    "It cannot be planted with grass."
                ),
                }),
            Row('match_1', 'matching_headings',
                (
                    "Paragraph 2"
                ),
                'ii',
                (
                    "The benefits are easiest to see during summer"
                ),
                choices={
                    'i': (
                    "Costs and difficulties"
                ),
                    'ii': (
                    "Benefits in hot weather"
                ),
                    'iii': (
                    "The origins of the idea"
                ),
                    'iv': (
                    "Parks in crowded cities"
                ),
                    'v': (
                    "Measuring energy bills"
                ),
                },
                paragraph=2),
            Row('match_2', 'matching_headings',
                (
                    "Paragraph 3"
                ),
                'i',
                (
                    "There are drawbacks"
                ),
                choices={
                    'i': (
                    "Costs and difficulties"
                ),
                    'ii': (
                    "Benefits in hot weather"
                ),
                    'iii': (
                    "The origins of the idea"
                ),
                    'iv': (
                    "Parks in crowded cities"
                ),
                    'v': (
                    "Measuring energy bills"
                ),
                },
                paragraph=3),
        ),
    ),
    Passage(
        "city_bees", CITY_BEES,
        (
            Row('tfng_1', 'tfng',
                (
                    "Keepers believe that city plants flower at varied times, which helps "
                    "bees to find food for much of the year."
                ),
                'TRUE',
                (
                    "gardens, parks and street trees flower at different times of year, "
                    "giving bees food from early spring to late autumn"
                )),
            Row('tfng_2', 'tfng',
                (
                    "The survey found that city honey contained more pollutants than honey "
                    "from farmland."
                ),
                'FALSE',
                (
                    "the honey itself contained no more pollutants than honey from farmland"
                )),
            Row('tfng_3', 'tfng',
                (
                    "Beekeepers in cities earn more money per hive than keepers in the "
                    "countryside."
                ),
                'NOT GIVEN',
                None,
                note=(
                    "the passage never discusses what keepers earn"
                )),
            Row('fill_1', 'sentence_completion',
                (
                    "Hives in dense districts gave less honey because rooftop ______ was "
                    "patchy."
                ),
                'forage',
                (
                    "because rooftop forage was patchy"
                )),
            Row('fill_2', 'sentence_completion',
                (
                    "Some ecologists fear that honeybees compete with wild bees for ______."
                ),
                'nectar',
                (
                    "honeybees compete with wild bees for nectar"
                )),
            Row('mcq_1', 'multiple_choice_single',
                (
                    "What do wild bee specialists recommend for people who want to help "
                    "bees?"
                ),
                'B',
                (
                    "recommend that anyone who wants to help should plant a variety of "
                    "flowers rather than keep a hive"
                ),
                choices={
                    'A': (
                    "Keep several hives."
                ),
                    'B': (
                    "Plant a variety of flowers."
                ),
                    'C': (
                    "Avoid busy parks."
                ),
                    'D': (
                    "Give honey to schools."
                ),
                }),
            Row('match_1', 'matching_headings',
                (
                    "Paragraph 2"
                ),
                'i',
                (
                    "Urban honey is not always better than rural honey, however"
                ),
                choices={
                    'i': (
                    "The limits of urban honey production"
                ),
                    'ii': (
                    "Why cities suit bees"
                ),
                    'iii': (
                    "Advice and rules for protecting wild bees"
                ),
                    'iv': (
                    "The history of beekeeping"
                ),
                    'v': (
                    "How honey is tested"
                ),
                },
                paragraph=2),
            Row('match_2', 'matching_headings',
                (
                    "Paragraph 3"
                ),
                'iii',
                (
                    "Wild bee specialists recommend that anyone who wants to help should "
                    "plant a variety of flowers rather than keep a hive"
                ),
                choices={
                    'i': (
                    "The limits of urban honey production"
                ),
                    'ii': (
                    "Why cities suit bees"
                ),
                    'iii': (
                    "Advice and rules for protecting wild bees"
                ),
                    'iv': (
                    "The history of beekeeping"
                ),
                    'v': (
                    "How honey is tested"
                ),
                },
                paragraph=3),
        ),
    ),
    Passage(
        "school_start", SCHOOL_START,
        (
            Row('tfng_1', 'tfng',
                (
                    "Teenagers usually go to sleep later than younger children."
                ),
                'TRUE',
                (
                    "teenagers tend to fall asleep later than children or adults"
                )),
            Row('tfng_2', 'tfng',
                (
                    "In the nine-school trial, attendance fell."
                ),
                'FALSE',
                (
                    "Attendance rose slightly"
                )),
            Row('tfng_3', 'tfng',
                (
                    "Students in the trial said that they felt happier at school."
                ),
                'NOT GIVEN',
                None,
                note=(
                    "the passage reports attendance, alertness and results, not happiness"
                )),
            Row('fill_1', 'sentence_completion',
                (
                    "The authors warned that ______ was too short a period for judging "
                    "academic effects."
                ),
                'two years',
                (
                    "two years was too short a period for judging academic effects"
                )),
            Row('fill_2', 'sentence_completion',
                (
                    "Sports coaches point out that afternoon training sessions would end "
                    "after ______."
                ),
                'dark',
                (
                    "afternoon training sessions would end after dark"
                )),
            Row('mcq_1', 'multiple_choice_single',
                (
                    "What do bus companies say about a later school start?"
                ),
                'B',
                (
                    "Bus companies add that rearranging routes costs money"
                ),
                choices={
                    'A': (
                    "It would reduce the fares they charge."
                ),
                    'B': (
                    "It would require route changes that cost money."
                ),
                    'C': (
                    "It would make buses more punctual."
                ),
                    'D': (
                    "It would shorten most journeys."
                ),
                }),
            Row('match_1', 'matching_headings',
                (
                    "Paragraph 2"
                ),
                'iii',
                (
                    "Several school districts have tested later start times"
                ),
                choices={
                    'i': (
                    "Objections from families and others"
                ),
                    'ii': (
                    "Why teenagers sleep late"
                ),
                    'iii': (
                    "Evidence from a school trial"
                ),
                    'iv': (
                    "Rising demand for sports"
                ),
                    'v': (
                    "Sleep needs of adults"
                ),
                },
                paragraph=2),
            Row('match_2', 'matching_headings',
                (
                    "Paragraph 3"
                ),
                'i',
                (
                    "Not everyone supports the change"
                ),
                choices={
                    'i': (
                    "Objections from families and others"
                ),
                    'ii': (
                    "Why teenagers sleep late"
                ),
                    'iii': (
                    "Evidence from a school trial"
                ),
                    'iv': (
                    "Rising demand for sports"
                ),
                    'v': (
                    "Sleep needs of adults"
                ),
                },
                paragraph=3),
        ),
    ),
    Passage(
        "wind_ships", WIND_SHIPS,
        (
            Row('tfng_1', 'tfng',
                (
                    "Modern ship sails differ in form from traditional canvas sails."
                ),
                'TRUE',
                (
                    "The modern versions are not canvas but tall rigid wings or spinning "
                    "cylinders"
                )),
            Row('tfng_2', 'tfng',
                (
                    "Wind rigs can be fitted only to newly built ships."
                ),
                'FALSE',
                (
                    "they can be added to existing vessels"
                )),
            Row('tfng_3', 'tfng',
                (
                    "Ships with wind rigs travel faster than ships with ordinary engines."
                ),
                'NOT GIVEN',
                None,
                note=(
                    "the passage discusses fuel use, not speed"
                )),
            Row('fill_1', 'sentence_completion',
                (
                    "Tall rigs must fold down or ______ to pass under bridges."
                ),
                'tilt',
                (
                    "tall rigs must fold down or tilt to pass underneath"
                )),
            Row('fill_2', 'sentence_completion',
                (
                    "Some analysts expect stricter rules on ______ to change the cost "
                    "calculation."
                ),
                'emissions',
                (
                    "stricter rules on emissions will change this calculation within a "
                    "decade"
                )),
            Row('mcq_1', 'multiple_choice_single',
                (
                    "According to the passage, a ship sailing against steady winds"
                ),
                'C',
                (
                    "a ship sailing against steady winds gains little"
                ),
                choices={
                    'A': (
                    "saves a great deal of fuel."
                ),
                    'B': (
                    "must use rotating cylinders."
                ),
                    'C': (
                    "gains little from the devices."
                ),
                    'D': (
                    "needs a new engine."
                ),
                }),
            Row('match_1', 'matching_headings',
                (
                    "Paragraph 2"
                ),
                'ii',
                (
                    "Engineers say that the main advantage of these devices is that they "
                    "can be added to existing vessels"
                ),
                choices={
                    'i': (
                    "Doubts over the price of the technology"
                ),
                    'ii': (
                    "Practical benefits and design problems"
                ),
                    'iii': (
                    "A return to an old idea"
                ),
                    'iv': (
                    "The history of ocean trade"
                ),
                    'v': (
                    "Cleaner fuels in other industries"
                ),
                },
                paragraph=2),
            Row('match_2', 'matching_headings',
                (
                    "Paragraph 3"
                ),
                'i',
                (
                    "Cost remains a barrier"
                ),
                choices={
                    'i': (
                    "Doubts over the price of the technology"
                ),
                    'ii': (
                    "Practical benefits and design problems"
                ),
                    'iii': (
                    "A return to an old idea"
                ),
                    'iv': (
                    "The history of ocean trade"
                ),
                    'v': (
                    "Cleaner fuels in other industries"
                ),
                },
                paragraph=3),
        ),
    ),
    Passage(
        "language_learning", LANGUAGE_LEARNING,
        (
            Row('tfng_1', 'tfng',
                (
                    "Older learners can sometimes make faster early progress than children."
                ),
                'TRUE',
                (
                    "older learners can progress faster in the early stages"
                )),
            Row('tfng_2', 'tfng',
                (
                    "The study found that age of arrival was the best predictor of fluency."
                ),
                'FALSE',
                (
                    "the best predictor of fluency after five years was not the age of "
                    "arrival"
                )),
            Row('tfng_3', 'tfng',
                (
                    "Classes cost more than language exchange websites."
                ),
                'NOT GIVEN',
                None,
                note=(
                    "the passage never compares costs"
                )),
            Row('fill_1', 'sentence_completion',
                (
                    "Older learners can study in a planned way because they understand "
                    "grammar ______."
                ),
                'rules',
                (
                    "they understand grammar rules and can study in a planned way"
                )),
            Row('fill_2', 'sentence_completion',
                (
                    "Games that reward ______ may encourage frequent but shallow practice."
                ),
                'streaks',
                (
                    "games which reward streaks may encourage frequent but shallow practice"
                )),
            Row('mcq_1', 'multiple_choice_single',
                (
                    "What did the researchers suggest about classes?"
                ),
                'B',
                (
                    "classes work best when they are combined with real conversation"
                ),
                choices={
                    'A': (
                    "They should be replaced by apps."
                ),
                    'B': (
                    "They work best alongside real conversation."
                ),
                    'C': (
                    "They suit only young learners."
                ),
                    'D': (
                    "They should last two hours a day."
                ),
                }),
            Row('match_1', 'matching_headings',
                (
                    "Paragraph 2"
                ),
                'ii',
                (
                    "A study of two hundred immigrants"
                ),
                choices={
                    'i': (
                    "Technology and its limits"
                ),
                    'ii': (
                    "What predicts success"
                ),
                    'iii': (
                    "Children and adults compared"
                ),
                    'iv': (
                    "The history of grammar teaching"
                ),
                    'v': (
                    "Choosing a language to learn"
                ),
                },
                paragraph=2),
            Row('match_2', 'matching_headings',
                (
                    "Paragraph 3"
                ),
                'i',
                (
                    "Technology has made conversation practice easier to find"
                ),
                choices={
                    'i': (
                    "Technology and its limits"
                ),
                    'ii': (
                    "What predicts success"
                ),
                    'iii': (
                    "Children and adults compared"
                ),
                    'iv': (
                    "The history of grammar teaching"
                ),
                    'v': (
                    "Choosing a language to learn"
                ),
                },
                paragraph=3),
        ),
    ),
)


def generate_all() -> list[ExamItem]:
    items: list[ExamItem] = []
    for passage in PASSAGES:
        for row in passage.rows:
            extras: dict[str, Any] = {
                "passage_id": passage.id,
                "instruction": INSTRUCTIONS[row.task_type],
            }
            if row.note:
                extras["key_note"] = row.note
            if row.paragraph is not None:
                extras["paragraph"] = row.paragraph
            items.append(
                ExamItem(
                    id=f"ielts_reading_{passage.id}_{row.suffix}",
                    exam="ielts",
                    skill="reading",
                    task_type=row.task_type,
                    prompt=row.prompt,
                    passage=passage.text,
                    ground_truth=row.key,
                    choices=row.choices,
                    answer_choice=row.key if row.choices else None,
                    evidence=row.evidence,
                    label_provenance=LabelProvenance(
                        tier=LabelTier.author_key,
                        date="2026-10-24",
                        note="written by an AI agent; not checked by a human",
                    ),
                    source=Source(kind="original", license="original to this repository (MIT)"),
                    design=Design(generator="llm_drafted"),
                    extras=extras,
                )
            )
    return items
