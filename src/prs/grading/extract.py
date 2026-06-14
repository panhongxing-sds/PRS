"""Dataset-aware answer extraction dispatcher.

Generation must extract the final answer with the extractor that matches the
dataset's grading mode. Using the math extractor on logic datasets (e.g. a plain
color word like "indigo") yields an empty string and silently fails all grading.
"""

from __future__ import annotations

from prs.datasets.registry import get_dataset_spec, normalize_dataset_id
from prs.grading.code_grader import extract_code_block
from prs.grading.logic_grader import extract_logic_answer
from prs.grading.math_grader import extract_math_answer


def extract_answer_for_dataset(text: str, dataset: str | None) -> str:
    """Extract the final answer using the grader appropriate for `dataset`.

    Falls back to the math extractor (then raw stripped text) when the dataset is
    unknown, preserving prior behaviour for math benchmarks.
    """
    if not text:
        return ""
    mode = "math"
    norm_id = None
    if dataset:
        try:
            norm_id = normalize_dataset_id(dataset)
            mode = get_dataset_spec(norm_id).grading
        except (ValueError, KeyError):
            mode = "math"
    # Zebra solutions are full grids; grading parses the assignment out of the whole
    # response, so keep the full conclusion text rather than a single final line.
    if norm_id == "zebra_puzzles":
        return text.strip()
    if mode == "string":
        return extract_logic_answer(text)
    if mode == "code":
        return extract_code_block(text) or text.strip()
    return extract_math_answer(text)
