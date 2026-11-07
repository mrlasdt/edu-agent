"""The IELTS writing prompts are product data: the product owns the file, evals reads it."""
from __future__ import annotations

import re
from pathlib import Path

from evals.exam_set import ielts_writing_prompts
from evals.loader import load_exam_items
from services.agent.src.agent.ielts import prompts as product_prompts

REPO = Path(__file__).resolve().parent.parent
PRODUCT_DIRS = ("services", "shared")
EVALS_REFERENCE = re.compile(r"(^\s*(from|import)\s+evals\b)|evals/golden", re.MULTILINE)


def test_prompts_file_lives_with_the_product_and_loads_from_any_working_directory(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    path = product_prompts.PROMPTS_PATH
    assert path.is_absolute() and path.is_file()
    assert (REPO / "services" / "agent") in path.parents
    prompts = product_prompts.load_prompts()
    assert len(prompts) == len(ielts_writing_prompts.generate_all()) > 0
    assert {p.essay_type for p in prompts} == set(product_prompts.ESSAY_TYPE_LABELS)


def test_the_old_eval_copy_is_gone_and_evals_reads_the_product_file():
    assert not (REPO / "evals" / "golden" / "ielts_writing_prompts.jsonl").exists()
    items = [i for i in load_exam_items() if i.task_type == "essay_prompt"]
    assert [i.id for i in items] == [p.id for p in product_prompts.load_prompts()]


def test_product_code_and_deploy_files_do_not_reach_into_evals():
    offenders: list[str] = []
    files = [
        p for d in PRODUCT_DIRS for p in (REPO / d).rglob("*")
        if p.is_file() and (p.suffix in {".py", ".toml", ".yml", ".yaml"} or p.name == "Dockerfile")
    ]
    files += [REPO / "infra" / "docker-compose.yml"]
    for path in files:
        if "__pycache__" in path.parts or "node_modules" in path.parts:
            continue
        if EVALS_REFERENCE.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, offenders
