"""Strict, conservative offline grader for math answers.

Motivation: the `label_wrong_clean` relabel path (answer_canonicalizer) collapses
scientific notation to its exponent only (e.g. ``1.01e6`` -> ``6``), so any two
answers sharing an exponent were judged equal. This module re-grades each sample
*strictly* from the stored ``a0`` (model answer) and ``reference`` (gold), and
returns a three-way verdict:

    "correct" | "wrong" | "drop"

"drop" means the grader cannot make a confident decision (unparseable / mixed
symbolic forms it cannot verify). Those samples are excluded from metric
computation rather than risk a misjudgment.

No model calls, no GPU: operates purely on the strings already in summary.jsonl.
"""

from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# relative / absolute tolerance for numeric equality
_REL_TOL = 1e-3
_ABS_TOL = 1e-9


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\boxed\s*\{", "", s)
    s = s.replace("$", "")
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("\\%", "").replace("%", "")
    s = s.strip()
    # drop a single pair of outer braces
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    return s


def _to_number(raw: str) -> float | None:
    """Parse a scalar number, fully handling scientific notation.

    Handles: ``1.01e6``, ``1.01E6``, ``1.01\\times10^{6}``, ``1.01\\times10^6``,
    ``1.01 x 10^6``, ``3\\cdot10^8``, plain ints/floats, simple fractions ``a/b``,
    leading ``+``, commas as thousands separators.
    """
    if raw is None:
        return None
    s = _strip_wrappers(str(raw))
    if s == "":
        return None
    s = s.replace(" ", "")
    # thousands separators: 1,234,567 (only when grouped in 3s)
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    # scientific notation: mantissa * 10^exp  (latex \times / \cdot / x)
    m = re.fullmatch(
        r"([-+]?\d*\.?\d+)\s*(?:\\times|\\cdot|x|\*)?\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?",
        s,
        flags=re.I,
    )
    if m:
        try:
            return float(m.group(1)) * (10.0 ** int(m.group(2)))
        except ValueError:
            return None
    # python-style sci notation 1.01e6
    m = re.fullmatch(r"([-+]?\d*\.?\d+)[eE]([-+]?\d+)", s)
    if m:
        try:
            return float(s)
        except ValueError:
            return None
    # simple fraction a/b
    m = re.fullmatch(r"([-+]?\d*\.?\d+)/([-+]?\d*\.?\d+)", s)
    if m:
        try:
            denom = float(m.group(2))
            return float(m.group(1)) / denom if denom != 0 else None
        except (ValueError, ZeroDivisionError):
            return None
    # plain number
    try:
        return float(s)
    except ValueError:
        return None


def _num_equal(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b), _ABS_TOL)
    return abs(a - b) <= max(_REL_TOL * scale, _ABS_TOL)


def _norm_struct(s: str) -> str:
    """Normalize a structured answer (tuple/set/interval/list) to a canonical
    string for exact comparison: strip wrappers, unify separators/spaces."""
    s = _strip_wrappers(s).lower()
    s = s.replace("\\cup", "u").replace("\\in", "")
    s = s.replace("\\le", "<=").replace("\\ge", ">=")
    s = s.replace("\\leq", "<=").replace("\\geq", ">=")
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\", "")
    return s


_STRUCT_CHARS = set("(),[];")


def _looks_structured(s: str) -> bool:
    return any(c in s for c in _STRUCT_CHARS) or "\\cup" in s or "\\in" in s


def _sympy_value(s: str):
    """Try to parse to a sympy expression; return expr or None."""
    s = _strip_wrappers(s)
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\d+)", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\^\{([^}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^(\w+)", r"**(\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    if not s or any(ch.isalpha() for ch in re.sub(r"(sqrt|pi|exp|log|sin|cos|tan)", "", s)):
        # contains free symbols/letters we don't want to guess on
        return None
    try:
        expr = parse_expr(s, transformations=_TRANSFORMATIONS, evaluate=True)
        if expr.free_symbols:
            return None
        return expr
    except Exception:
        return None


def strict_grade(a0: str, ref: str) -> str:
    """Return 'correct' | 'wrong' | 'drop'."""
    if a0 is None or ref is None:
        return "drop"
    a0s, refs = str(a0).strip(), str(ref).strip()
    if a0s == "" or refs == "":
        return "drop"

    # 1) exact normalized string match (covers identical answers, format diffs)
    if _norm_struct(a0s) == _norm_struct(refs):
        return "correct"

    # 2) scalar numeric comparison (fully parses scientific notation)
    na, nb = _to_number(a0s), _to_number(refs)
    if na is not None and nb is not None:
        return "correct" if _num_equal(na, nb) else "wrong"

    # 3) one side numeric, the other not parseable as number → structured vs scalar
    if (na is None) != (nb is None):
        # mismatched kinds; only safe verdict if structured strings differ cleanly
        if _looks_structured(a0s) or _looks_structured(refs):
            return "drop"
        return "drop"

    # 4) structured answers (tuples/sets/intervals): exact normalized compare only
    if _looks_structured(a0s) or _looks_structured(refs):
        # already failed exact match in step 1 → can't confidently call it wrong
        # unless both are clearly structured of same shape; be conservative → drop
        return "drop"

    # 5) symbolic numeric expressions via sympy (sqrt, pi, fractions ...)
    ea, eb = _sympy_value(a0s), _sympy_value(refs)
    if ea is not None and eb is not None:
        try:
            diff = sp.simplify(ea - eb)
            if diff == 0:
                return "correct"
            dv = abs(float(sp.N(diff)))
            scale = max(abs(float(sp.N(ea))), abs(float(sp.N(eb))), _ABS_TOL)
            return "correct" if dv <= max(_REL_TOL * scale, _ABS_TOL) else "wrong"
        except Exception:
            return "drop"

    return "drop"
