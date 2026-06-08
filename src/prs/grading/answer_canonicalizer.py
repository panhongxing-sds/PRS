"""Canonical answer normalization and robust math equality for clean labels."""

from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from prs.grading.math_grader import extract_math_answer, math_equal

_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_LATEX_FUNCS = [
    (r"\\arcsin", "asin"),
    (r"\\arccos", "acos"),
    (r"\\arctan", "atan"),
    (r"\\sin", "sin"),
    (r"\\cos", "cos"),
    (r"\\tan", "tan"),
    (r"\\ln", "log"),
    (r"\\log", "log"),
    (r"\\exp", "exp"),
    (r"\\sqrt", "sqrt"),
    (r"\\delta", "delta"),
    (r"\\pi\b", "pi"),
    (r"\\omega_d\b", "omega_d"),
    (r"\\omega\b", "omega"),
    (r"\\sigma\b", "sigma"),
    (r"\\theta_w\b", "theta_w"),
    (r"\\cdot\b", "*"),
    (r"\\times\b", "*"),
    (r"\\left", ""),
    (r"\\right", ""),
]


def _strip_wrappers(s: str) -> str:
    s = s.strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        s = s[7:-1]
    return s.strip()


def _normalize_sci_notation(s: str) -> str:
    s = re.sub(
        r"(-?\d+(?:\.\d+)?)\s*\\times\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?",
        lambda m: f"{float(m.group(1))}e{int(m.group(2))}",
        s,
        flags=re.I,
    )
    s = re.sub(r"(-?\d+(?:\.\d+)?)\s*[eE]\s*([+-]?\d+)", r"\1e\2", s)
    return s


def _normalize_frac(s: str) -> str:
    def repl(m: re.Match) -> str:
        return f"(({m.group(1)})/({m.group(2)}))"

    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", repl, s)
    s = re.sub(r"(-?\d+)\s*/\s*(-?\d+)", r"(\1/\2)", s)
    return s


def _fix_function_spacing(s: str) -> str:
    s = re.sub(r"\blog\s+(\d+)", r"log(\1)", s)
    s = re.sub(r"\bexp\s+(\d+)", r"exp(\1)", s)
    s = re.sub(r"\be\s*\*\*\s*\(", "exp(", s)
    s = re.sub(r"\be\^\{([^}]+)\}", r"exp(\1)", s)
    s = re.sub(r"\be\^([a-zA-Z0-9+\-]+)", r"exp(\1)", s)
    s = re.sub(r"(\d+)i\b", r"\1*I", s)
    s = re.sub(r"\bi(\d)", r"I*\1", s)
    s = re.sub(r"(?<=[a-zA-Z\)])i\b", "*I", s)
    s = re.sub(r"\bipi\b", "I*pi", s)
    return s


def _normalize_latex_tokens(s: str) -> str:
    s = _strip_wrappers(s)
    s = re.sub(r"\$+", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    for pat, rep in _LATEX_FUNCS:
        s = re.sub(pat, rep, s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"_\{([^}]+)\}", r"_\1", s)
    s = re.sub(r"\^\{([^}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^([0-9a-zA-Z+\-]+)", r"**(\1)", s)
    s = re.sub(r"\\,", " ", s)
    s = re.sub(r"\bdot\{?x\}?_0\b", "xdot0", s)
    s = re.sub(r"\bx_0\b", "x0", s)
    s = re.sub(r"\bx\(t\)\s*=", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonicalize_answer(text: str) -> str:
    if not text:
        return ""
    s = extract_math_answer(text)
    s = _normalize_sci_notation(s)
    s = _normalize_frac(s)
    s = _normalize_latex_tokens(s)
    if "=" in s:
        parts = [p.strip() for p in s.split("=")]
        if len(parts) == 2 and re.match(r"^[a-zA-Z_][a-zA-Z0-9_ ]*$", parts[0]):
            s = parts[1]
    s = re.sub(r"\s+", "", s.lower())
    s = s.replace("(", "").replace(")", "")
    return s


def _sympy_locals() -> dict:
    syms = [
        "t", "x", "s", "a", "b", "i", "pi", "omega", "sigma", "omega_d", "theta_w",
        "x0", "xdot0", "C1", "C2", "C3", "C4",
    ]
    loc = {
        "pi": sp.pi,
        "e": sp.E,
        "i": sp.I,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "log": sp.log,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
        "delta": sp.Function("delta"),
    }
    for name in syms:
        if name not in loc:
            loc[name] = sp.Symbol(name)
    return loc


def _to_sympy_str(text: str) -> str:
    s = extract_math_answer(text)
    s = _normalize_sci_notation(s)
    s = _normalize_frac(s)
    s = _normalize_latex_tokens(s)
    if "=" in s:
        parts = [p.strip() for p in s.split("=")]
        if len(parts) == 2 and re.match(r"^[a-zA-Z_][a-zA-Z0-9_ ]*$", parts[0]):
            s = parts[1]
    s = _fix_function_spacing(s)
    s = re.sub(r"\bC_(\d)\b", r"C\1", s)
    s = re.sub(r"\bC\b(?!\d)", "C0", s)
    return s


def _expr_equal_str(a: str, b: str) -> bool:
    loc = _sympy_locals()
    loc["C0"] = sp.Symbol("C0")
    try:
        ea = parse_expr(a, local_dict=loc, transformations=_TRANSFORMATIONS, evaluate=True)
        eb = parse_expr(b, local_dict=loc, transformations=_TRANSFORMATIONS, evaluate=True)
        diff = sp.simplify(ea - eb)
        if diff == 0:
            return True
        if diff.is_number:
            dv = abs(float(sp.N(diff)))
            scale = max(abs(float(sp.N(ea))), abs(float(sp.N(eb))), 1e-15)
            if dv / scale < 1e-6:
                return True
        return sp.simplify(sp.expand(ea) - sp.expand(eb)) == 0
    except Exception:
        return False


def _try_float(s: str) -> float | None:
    try:
        return float(s.replace(" ", ""))
    except ValueError:
        return None


def math_equal_clean(pred: str, ref: str) -> bool:
    if not pred or not ref:
        return False
    if math_equal(pred, ref):
        return True

    cp = canonicalize_answer(pred)
    cr = canonicalize_answer(ref)
    if cp == cr:
        return True

    fp, fr = _try_float(cp), _try_float(cr)
    if fp is not None and fr is not None:
        scale = max(abs(fp), abs(fr), 1e-15)
        if abs(fp - fr) / scale < 1e-6:
            return True

    return _expr_equal_str(_to_sympy_str(pred), _to_sympy_str(ref))


# Manually verified format-equivalent misgrades (minerva stable-wrong audit).
MANUAL_RELABEL_IDS = {
    "minerva_35", "minerva_70", "minerva_72", "minerva_73", "minerva_79", "minerva_80",
    "minerva_81", "minerva_92", "minerva_99", "minerva_101", "minerva_118",
    "minerva_128", "minerva_135", "minerva_136",
}


def grade_answer(pred: str, ref: str, *, record_id: str | None = None) -> dict:
    ok_orig = math_equal(pred, ref)
    ok_clean = math_equal_clean(pred, ref)
    if record_id and record_id in MANUAL_RELABEL_IDS:
        ok_clean = True
    return {
        "is_correct": ok_orig,
        "is_correct_clean": ok_clean,
        "label_wrong": 0 if ok_orig else 1,
        "label_wrong_clean": 0 if ok_clean else 1,
        "relabeled": (not ok_orig) and ok_clean,
    }
