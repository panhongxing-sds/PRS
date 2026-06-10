"""Answer extraction and grading for logic benchmarks (string match)."""

from __future__ import annotations

import json
import re


_FINAL_ANSWER_RE = re.compile(
    r"(?:the\s+)?final\s+answer\s+is\s*[:\-]?\s*(.+?)(?:\.|$)",
    re.IGNORECASE | re.DOTALL,
)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_logic_answer(text: str) -> str:
    if not text:
        return ""
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().strip("\"'")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def normalize_logic_answer(text: str) -> str:
    s = extract_logic_answer(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def logic_equal(pred: str, ref: str) -> bool:
    if not pred or not ref:
        return False
    p = normalize_logic_answer(pred)
    r = normalize_logic_answer(ref)
    if p == r:
        return True
    # Zebra puzzle JSON solutions
    if ref.startswith("{") or ref.startswith("["):
        try:
            ref_obj = json.loads(ref)
            try:
                pred_obj = json.loads(pred)
                return pred_obj == ref_obj
            except json.JSONDecodeError:
                pass
        except json.JSONDecodeError:
            pass
    # Numeric leg-counting
    p_num = re.search(r"-?\d+(?:\.\d+)?", p)
    r_num = re.search(r"-?\d+(?:\.\d+)?", r)
    if p_num and r_num:
        try:
            return int(float(p_num.group())) == int(float(r_num.group()))
        except ValueError:
            pass
    return False
