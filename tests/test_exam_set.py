"""
Schema, loader, local-only third-party data, SAT Math templates and IELTS Reading items.
No network and no model calls; every third-party-shaped fixture here is tiny and synthetic.
"""
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evals.agent_eval.ielts_writing import select_items
from evals.exam_set import ielts_reading, ielts_writing_prompts, sat_math
from evals.exam_set.local_store import (
    band_histogram,
    load_manifest,
    local_dir,
    read_items,
    register_source,
)
from evals.exam_set.schema import (
    Design,
    ExamItem,
    LabelProvenance,
    Labels,
    LabelTier,
    Source,
    best_label_tier,
)
from evals.loader import _GOLDEN_DIR, EXAM_FAMILIES, load_exam_items, load_golden_set
from services.math_verifier.src.math_verifier.sandbox import verify_expression

REPO = Path(__file__).resolve().parents[1]


def essay_item(
    id: str = "syn1", overall: float = 6.0, tier: LabelTier = LabelTier.site_or_model_weak,
    kind: str = "third_party_local",
) -> ExamItem:
    return ExamItem(
        id=id, exam="ielts", skill="writing_t2", task_type="essay_score",
        prompt="Synthetic prompt.", response="Synthetic essay text.",
        labels=Labels(overall=overall, criteria={"TR": 6, "CC": 6, "LR": 6, "GRA": 6}),
        label_provenance=LabelProvenance(tier=tier),
        source=Source(kind=kind, license="synthetic"),  # type: ignore[arg-type]
        design=Design(),
    )


# ── schema ───────────────────────────────────────────────────────────────────


def test_labels_reject_non_half_bands_and_unknown_criteria():
    Labels(overall=6.5, criteria={"TR": 7.0})
    with pytest.raises(ValidationError):
        Labels(overall=6.3)
    with pytest.raises(ValidationError):
        Labels(overall=9.5)
    with pytest.raises(ValidationError):
        Labels(criteria={"XX": 6.0})


def test_item_schema_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ExamItem.model_validate({**json.loads(essay_item().model_dump_json()), "surprise": 1})


def test_best_label_tier_prefers_examiner_then_captain_then_weak():
    weak = essay_item("a")
    captain = essay_item("b", tier=LabelTier.captain_blind)
    examiner = essay_item("c", tier=LabelTier.examiner)
    assert best_label_tier([weak]) == LabelTier.site_or_model_weak
    assert best_label_tier([weak, captain]) == LabelTier.captain_blind
    assert best_label_tier([weak, captain, examiner]) == LabelTier.examiner
    teacher = essay_item("d", tier=LabelTier.teacher_graded)
    assert best_label_tier([weak, captain, teacher]) == LabelTier.teacher_graded
    assert best_label_tier([weak, teacher, examiner]) == LabelTier.examiner
    assert best_label_tier([]) is None


# ── loader ───────────────────────────────────────────────────────────────────


def test_gre_loader_is_unchanged():
    assert len(load_golden_set("quant")) == 5
    assert len(load_golden_set("aw")) == 2


def test_committed_exam_families_load_and_are_committable():
    items = load_exam_items()
    assert {i.skill for i in items} == {"sat_math", "reading", "writing_t2"}
    assert items and all(i.committable() for i in items)
    for family in EXAM_FAMILIES:
        assert (_GOLDEN_DIR / f"{family}.jsonl").exists()


def test_local_items_load_only_when_asked(tmp_path: Path):
    register_source(
        tmp_path, source_id="syn", name="synthetic", url="n/a", licence_status="synthetic",
        items=[essay_item()], file_name="syn.jsonl",
    )
    assert not [i for i in load_exam_items(local_directory=tmp_path) if i.id == "syn1"]
    loaded = load_exam_items(include_local=True, local_directory=tmp_path, skill="writing_t2")
    assert [i.id for i in loaded if i.source.kind == "third_party_local"] == ["syn1"]
    missing = tmp_path / "missing"
    assert load_exam_items(include_local=True, local_directory=missing, exam="gre") == []


def test_local_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("EDU_GOLDEN_LOCAL_DIR", str(tmp_path))
    assert local_dir() == tmp_path
    assert local_dir(tmp_path / "x") == tmp_path / "x"


# ── local-only third-party data ──────────────────────────────────────────────


def test_local_directory_is_gitignored():
    probe = "evals/golden/local/anything.jsonl"
    done = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", probe], check=False
    )
    assert done.returncode == 0, "evals/golden/local/ must be gitignored"


# Third-party essays, the mock's attempts, blind pools and downloaded band descriptors; and the
# spend ledger and run reports, which hold costs and model outputs.
LOCAL_ONLY_DIRS = (
    "evals/golden/local", "ielts-local", "ielts-local/blind", "ielts-local/descriptors",
    ".eval-ledger", ".eval-runs", ".eval-cache",
)


@pytest.mark.parametrize("directory", LOCAL_ONLY_DIRS)
def test_local_only_directories_are_ignored_and_hold_nothing_tracked(directory: str):
    """Nothing under a local-only directory may be tracked, and each must be gitignored."""
    ignored = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", f"{directory}/anything.jsonl"],
        check=False,
    )
    assert ignored.returncode == 0, f"{directory}/ must be gitignored"
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", directory],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert not tracked, f"files under {directory}/ are tracked: {tracked}"


def test_no_third_party_text_in_committed_golden_files():
    committed = [p for p in _GOLDEN_DIR.glob("*.jsonl") if p.stem not in {
        "gre_quant", "gre_aw", "gre_quant_generation"}]
    assert committed
    for path in committed:
        for item in read_items(path):
            assert item.source.kind in {"original", "cc_by"}, f"{path.name}:{item.id}"
    assert not (_GOLDEN_DIR / "local").exists() or not subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "evals/golden/local"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_manifest_records_source_tier_counts_and_hash(tmp_path: Path):
    items = [essay_item("a", 5.0), essay_item("b", 6.5), essay_item("c", 6.5)]
    record = register_source(
        tmp_path, source_id="syn", name="Synthetic", url="https://example.invalid",
        licence_status="synthetic only", items=items, file_name="syn.jsonl", caveat="weak",
    )
    assert record.count == 3 and record.label_tier == "site_or_model_weak"
    assert record.counts_by_overall_band == {"5": 1, "6.5": 2}
    manifest = load_manifest(tmp_path)
    (entry,) = manifest["sources"]
    assert entry["licence_status"] == "synthetic only" and entry["sha256"] == record.sha256
    # re-registering replaces the entry rather than duplicating it
    register_source(
        tmp_path, source_id="syn", name="Synthetic", url="u", licence_status="l",
        items=items[:1], file_name="syn.jsonl",
    )
    assert [s["count"] for s in load_manifest(tmp_path)["sources"]] == [1]
    assert band_histogram(items) == {"5": 1, "6.5": 2}


def test_items_with_unicode_line_separators_round_trip(tmp_path: Path):
    item = essay_item()
    item = item.model_copy(update={"response": "line one\u2028line two\x85three\u00a0end"})
    register_source(
        tmp_path, source_id="syn", name="n", url="u", licence_status="l", items=[item, item],
        file_name="syn.jsonl",
    )
    loaded = read_items(tmp_path / "syn.jsonl")
    assert [i.response for i in loaded] == [item.response, item.response]
    assert not list(tmp_path.glob("*.tmp"))


# ── SAT Math templates ───────────────────────────────────────────────────────


def test_sat_generation_is_deterministic():
    assert sat_math.generate("sat_linear_equation", 101) == sat_math.generate(
        "sat_linear_equation", 101
    )
    assert sat_math.generate("sat_linear_equation", 101) != sat_math.generate(
        "sat_linear_equation", 202
    )


def test_committed_sat_file_matches_regeneration_from_stored_seed():
    items = [i for i in load_exam_items() if i.skill == "sat_math"]
    assert len(items) >= 30
    assert [i.model_dump() for i in items] == [i.model_dump() for i in sat_math.generate_all()]
    for item in items:
        assert sat_math.regenerate(item) == item
        assert item.design.seed is not None and item.design.generator.startswith("template:")
        assert len(item.design.checks) == 2  # sympy plus an independent computation


@pytest.mark.parametrize("item", sat_math.generate_all(), ids=lambda i: i.id)
def test_sat_item_agrees_with_the_product_verifier(item: ExamItem):
    assert item.expression is not None and item.ground_truth is not None
    result = asyncio.run(verify_expression(item.expression, item.ground_truth))
    assert result.verified, (item.id, item.expression, item.ground_truth, result)


def test_sat_multiple_choice_items_have_one_key_and_three_distinct_distractors():
    mc = [i for i in sat_math.generate_all() if i.task_type == "multiple_choice_single"]
    assert mc
    for item in mc:
        assert item.choices is not None and item.answer_choice is not None
        assert sorted(item.choices) == ["A", "B", "C", "D"]
        assert len(set(item.choices.values())) == 4
        assert item.choices[item.answer_choice].replace("π", "*pi") == item.ground_truth


def test_sat_numeric_items_have_no_choices():
    numeric = [i for i in sat_math.generate_all() if i.task_type == "numeric_entry"]
    assert numeric and all(i.choices is None and i.answer_choice is None for i in numeric)


def test_sat_generation_refuses_disagreeing_computations(monkeypatch: pytest.MonkeyPatch):
    original = sat_math.t_hypotenuse

    def broken(rng: Any) -> sat_math.Draft:
        draft = original(rng)
        draft.secondary = draft.secondary + 1  # type: ignore[operator]
        return draft

    spec = sat_math._BY_ID["sat_hypotenuse"]
    monkeypatch.setitem(
        sat_math._BY_ID, "sat_hypotenuse",
        sat_math.TemplateSpec("sat_hypotenuse", broken, spec.kind),
    )
    with pytest.raises(sat_math.VerificationDisagreementError):
        sat_math.generate("sat_hypotenuse", 101)


def test_sat_covers_the_four_public_domains():
    domains = {i.extras["domain"] for i in sat_math.generate_all()}
    assert domains == {
        "Algebra", "Advanced Math", "Problem-Solving and Data Analysis",
        "Geometry and Trigonometry",
    }


def test_sat_templates_never_emit_malformed_signs_and_agree_with_the_verifier():
    for spec in sat_math.TEMPLATES:
        for seed in range(200):
            item = sat_math.generate(spec.id, seed)
            assert "+ -" not in item.prompt and "- -" not in item.prompt, (spec.id, seed)
            assert item.expression is not None and item.ground_truth is not None
            result = asyncio.run(verify_expression(item.expression, item.ground_truth))
            assert result.verified, (spec.id, seed, item.expression, item.ground_truth)


# ── IELTS Reading ────────────────────────────────────────────────────────────


def reading_items() -> list[ExamItem]:
    return [i for i in load_exam_items() if i.skill == "reading"]


def test_committed_reading_matches_generator_and_covers_five_passages():
    items = reading_items()
    assert [i.model_dump() for i in items] == [i.model_dump() for i in ielts_reading.generate_all()]
    assert len({i.extras["passage_id"] for i in items}) >= 5
    assert {i.task_type for i in items} == {
        "tfng", "sentence_completion", "multiple_choice_single", "matching_headings",
    }


def test_reading_evidence_is_in_the_passage_and_answers_in_evidence():
    for item in reading_items():
        assert item.passage is not None and item.ground_truth is not None
        assert item.label_provenance.tier == LabelTier.author_key
        if item.task_type == "tfng":
            assert item.ground_truth in {"TRUE", "FALSE", "NOT GIVEN"}
            if item.ground_truth != "NOT GIVEN":
                assert item.evidence is not None and item.evidence in item.passage
            else:
                assert item.evidence is None and item.extras["key_note"]
        else:
            assert item.evidence is not None and item.evidence in item.passage
        if item.task_type == "sentence_completion":
            assert item.ground_truth in item.evidence  # type: ignore[operator]
            assert len(item.ground_truth.split()) <= ielts_reading.WORD_LIMIT


def test_reading_choice_items_have_a_key_among_the_choices():
    for item in reading_items():
        if item.task_type == "multiple_choice_single":
            assert item.choices is not None and sorted(item.choices) == ["A", "B", "C", "D"]
        if item.task_type == "matching_headings":
            assert item.choices is not None and len(item.choices) >= 4
            assert item.passage is not None and item.evidence is not None
            paragraph = item.passage.split("\n\n")[item.extras["paragraph"] - 1]
            assert item.evidence in paragraph
        if item.choices is not None:
            assert item.answer_choice == item.ground_truth in item.choices


def test_writing_prompts_are_unscored_and_ignored_by_the_writing_grader():
    prompts = [i for i in load_exam_items() if i.task_type == "essay_prompt"]
    assert [i.model_dump() for i in prompts] == [
        i.model_dump() for i in ielts_writing_prompts.generate_all()
    ]
    assert {i.extras["essay_type"] for i in prompts} == {
        "opinion", "discuss_both_views", "advantages_disadvantages", "problem_solution",
    }
    for item in prompts:
        assert item.response is None and item.labels is None and item.committable()
        assert item.label_provenance.tier == LabelTier.author_key
    _, picked, _ = select_items(prompts, "best", 100, 0)
    assert picked == []


def test_committed_ielts_items_outnumber_sat_items_with_unique_ids():
    items = load_exam_items()
    ielts = [i for i in items if i.exam == "ielts"]
    sat = [i for i in items if i.exam == "sat"]
    assert len(ielts) >= len(sat) and len(ielts) >= 36
    assert len({i.id for i in items}) == len(items)
    assert all(i.committable() for i in items)
