"""答案抽取与判分（依赖 PANDA math_grader）。"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_PRS = Path(os.environ.get("PANDA_ROOT", "/root/PANDA")) / "src"
if str(_PRS) not in sys.path:
    sys.path.insert(0, str(_PRS))

from panda.grading.math_grader import math_equal, extract_math_answer

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_MCQ_RE = re.compile(r"(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)", re.IGNORECASE)
_MCQ_TAIL_RE = re.compile(
    r"(?:answer|choice|option)\s*(?:is)?\s*[:：]?\s*\(?([A-D])\)?",
    re.IGNORECASE,
)


def extract_answer(text: str, grading: str, dataset: str = "") -> str:
    if not text:
        return ""
    if grading == "mcq":
        return extract_mcq_letter(text)
    boxed = _BOXED_RE.search(text)
    if boxed:
        inner = boxed.group(1).strip()
        nums = re.findall(r"-?\d+", inner)
        if nums and dataset.startswith(("aime", "aimo")):
            return nums[-1]
        return inner
    ans = extract_math_answer(text)
    if ans:
        return ans
    if dataset.startswith(("aime", "aimo")):
        nums = re.findall(r"\b(\d{1,3})\b", text)
        if nums:
            return nums[-1]
    return ans or ""


def extract_mcq_letter(text: str) -> str:
    m = _MCQ_TAIL_RE.search(text)
    if m:
        return m.group(1).upper()
    for m in reversed(list(_MCQ_RE.finditer(text))):
        return m.group(1).upper()
    return ""


def is_correct(pred: str, gold: str, grading: str) -> bool:
    if not pred or not gold:
        return False
    if grading == "mcq":
        return pred.strip().upper() == gold.strip().upper()
    return math_equal(pred, gold)
