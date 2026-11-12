"""The one JSON-lines reader."""
from __future__ import annotations

from pathlib import Path

from shared.src.shared.jsonl import read_jsonl


def test_reads_one_object_per_line_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n\n  \n{"id": 2}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"id": 1}, {"id": 2}]


def test_a_row_with_a_raw_line_separator_inside_a_string_stays_one_row(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"text": "a b"}\n{"text": "c"}', encoding="utf-8")
    assert read_jsonl(path) == [{"text": "a b"}, {"text": "c"}]


def test_reads_utf8_whatever_the_locale(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes('{"text": "naïve – café"}\n'.encode())
    assert read_jsonl(path) == [{"text": "naïve – café"}]
