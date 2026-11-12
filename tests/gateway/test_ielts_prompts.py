"""The IELTS writing prompts are product data: the product owns the file, evals reads it."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from evals.exam_set import ielts_writing_prompts
from evals.loader import load_exam_items
from services.gateway.src.gateway.ielts import prompts as product_prompts

REPO = Path(__file__).resolve().parents[2]


def test_prompts_file_lives_with_the_product_and_loads_from_any_working_directory(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    path = product_prompts.PROMPTS_PATH
    assert path.is_absolute() and path.is_file()
    assert (REPO / "services" / "gateway") in path.parents
    prompts = product_prompts.load_prompts()
    assert len(prompts) == len(ielts_writing_prompts.generate_all()) > 0
    assert {p.essay_type for p in prompts} == set(product_prompts.ESSAY_TYPE_LABELS)


def test_the_old_eval_copy_is_gone_and_evals_reads_the_product_file():
    assert not (REPO / "evals" / "golden" / "ielts_writing_prompts.jsonl").exists()
    items = [i for i in load_exam_items() if i.task_type == "essay_prompt"]
    assert [i.id for i in items] == [p.id for p in product_prompts.load_prompts()]


def test_product_loader_works_with_evals_unimportable():
    """The product must serve its prompts from a deployment that has no evals/ directory."""
    code = (
        f"import sys; sys.path.insert(0, {str(REPO)!r}); sys.modules['evals'] = None\n"
        "from services.gateway.src.gateway.ielts.prompts import load_prompts\n"
        "print(len(load_prompts()))\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=REPO,
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert int(done.stdout) == len(product_prompts.load_prompts())
