"""Math answer grading (TokUR-compatible when available)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from panda.paths import TOKUR_ROOT

_TOKUR_ROOT = TOKUR_ROOT
_USE_TOKUR = False
if _TOKUR_ROOT.exists():
    root = str(_TOKUR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from run.utils.qwen_math_parser import extract_answer, strip_string  # noqa: F401

        _USE_TOKUR = True
    except Exception:
        _USE_TOKUR = False


def _extract_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    idx = text.find(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def extract_math_answer(text: str) -> str:
    if _USE_TOKUR:
        try:
            return strip_string(extract_answer(text, "math"))
        except Exception:
            pass
    boxed = _extract_boxed(text)
    if boxed:
        return boxed
    m = re.search(r"####\s*([^\n]+)", text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def math_equal(pred: str, ref: str) -> bool:
    pred = extract_math_answer(pred).strip()
    ref = ref.strip()
    if pred == ref:
        return True
    try:
        from sympy import simplify
        from sympy.parsing.latex import parse_latex

        a = simplify(parse_latex(pred) - parse_latex(ref))
        return abs(float(a)) < 1e-6
    except Exception:
        pass
    pred_n = re.sub(r"\s+", "", pred.lower())
    ref_n = re.sub(r"\s+", "", ref.lower())
    if pred_n == ref_n:
        return True
    try:
        return abs(float(pred_n) - float(ref_n)) < 1e-3
    except ValueError:
        return pred_n in ref_n or ref_n in pred_n