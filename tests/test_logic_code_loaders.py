"""Tests for logic/code dataset loaders and graders."""

from __future__ import annotations

from pathlib import Path

import pytest

from panda.datasets.loaders import load_tokur_jsonl_rows
from panda.datasets.registry import get_dataset_spec
from panda.grading.answer_canonicalizer import grade_answer
from panda.grading.code_grader import extract_code_block, normalize_code
from panda.grading.logic_grader import extract_logic_answer, logic_equal

FIXTURES = Path(__file__).parent / "fixtures"


def test_leg_counting_loader():
    spec = get_dataset_spec("leg_counting")
    rows = load_tokur_jsonl_rows(FIXTURES / "leg-counting.jsonl", spec)
    assert len(rows) == 2
    assert rows[0]["id"] == "leg_0"
    assert "cats" in rows[0]["question"].lower()
    assert rows[0]["reference"] == "10"


def test_zebra_puzzles_compose_question():
    spec = get_dataset_spec("zebra_puzzles")
    rows = load_tokur_jsonl_rows(FIXTURES / "zebra_puzzles.jsonl", spec, seed=0)
    by_id = {r["id"]: r for r in rows}
    assert "houses" in by_id["zp_0"]["question"].lower()
    assert by_id["zp_1"]["reference"] == "alice"


def test_humaneval_loader():
    spec = get_dataset_spec("humaneval")
    rows = load_tokur_jsonl_rows(FIXTURES / "humaneval.jsonl", spec)
    assert rows[0]["id"] == "HumanEval_0"
    assert "def add" in rows[0]["question"]
    assert rows[0]["entry_point"] == "add"


def test_logic_answer_extraction():
    text = "Step 1: count\nThe final answer is 10."
    assert extract_logic_answer(text) == "10"


def test_logic_grade_dataset():
    g = grade_answer("The final answer is 10", "10", dataset="leg_counting")
    assert g["is_correct_clean"] is True


def test_code_extract_and_grade():
    pred = "```python\ndef add(a, b):\n    return a + b\n```"
    ref = "    return a + b\n"
    assert "return a + b" in normalize_code(pred)
    g = grade_answer(pred, ref, dataset="humaneval")
    assert g["is_correct_clean"] is True


def test_logic_equal_numeric():
    assert logic_equal("10 legs", "10") is True
