"""HumanEval-style answer extraction (functional grading optional)."""

from __future__ import annotations

import re


def extract_code_block(text: str) -> str:
    if not text:
        return ""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    if "def " in text:
        idx = text.rfind("def ")
        return text[idx:].strip()
    return text.strip()


def normalize_code(text: str) -> str:
    code = extract_code_block(text)
    lines = [ln.rstrip() for ln in code.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip())


def code_equal(pred: str, ref: str) -> bool:
    """Lightweight check: normalized body match (full pass@1 needs test execution)."""
    p = normalize_code(pred)
    r = normalize_code(ref)
    if not p or not r:
        return False
    if p == r:
        return True

    def _compact(s: str) -> str:
        return re.sub(r"\s+", "", s)

    pc, rc = _compact(p), _compact(r)
    if pc == rc:
        return True
    # Reference may be body-only while pred is full function
    return rc in pc or pc in rc
