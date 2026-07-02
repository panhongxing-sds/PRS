"""Structured reasoning-token features (beyond entropy aggregation)."""

from __future__ import annotations

import math
import re

import numpy as np

from panda.core.numeric_trajectory import extract_numbers
from panda.grading.answer_canonicalizer import math_equal_clean
from panda.grading.math_grader import math_equal

_NUM_TEXT = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_SYM_TEXT = re.compile(r"^[+\-*/=^\\]$|^\*\*$|^\\[a-zA-Z]+")
_UNIT = re.compile(r"^(m|s|kg|N|J|Hz|rad|mol|K|A|V|W|Pa)$", re.I)


def _span_slice(trace: list[dict], span: dict | None) -> list[dict]:
    if not trace:
        return []
    span = span or {}
    s = max(0, min(int(span.get("start_token", 0)), len(trace) - 1))
    e = max(s, min(int(span.get("end_token", len(trace) - 1)), len(trace) - 1))
    return trace[s : e + 1]


def _reasoning_text(run: dict) -> str:
    resp = run.get("full_response") or ""
    trace = run.get("token_trace") or []
    span = run.get("answer_span") or {}
    if trace and span.get("start_token") is not None:
        s = int(span["start_token"])
        pre = trace[: max(0, s)]
        return "".join(t.get("token", "") for t in pre)
    # fallback: all but last line
    lines = resp.splitlines()
    return "\n".join(lines[:-1]) if len(lines) > 1 else resp[: max(0, len(resp) // 2)]


def _classify_token(tok: str) -> str:
    t = tok.strip()
    if not t or t.isspace():
        return "other"
    if _NUM_TEXT.match(t.replace(" ", "")):
        return "numeric"
    if _UNIT.match(t):
        return "unit"
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", t):
        return "variable"
    if _SYM_TEXT.match(t) or t in "+-*/=^":
        return "symbol"
    if re.search(r"\\frac|\\sqrt|\\pi|\\delta|\\sin|\\cos|\\omega|\\theta", t):
        return "symbol"
    if re.search(r"\d", t):
        return "numeric"
    return "other"


def _math_tokens(trace: list[dict]) -> list[tuple[int, str, str]]:
    out = []
    for i, t in enumerate(trace):
        tok = t.get("token", "")
        kind = _classify_token(tok)
        if kind != "other":
            out.append((i, tok, kind))
    return out


def _extract_last_equation(text: str) -> str:
    if not text:
        return ""
    # last line with = before boxed/final
    candidates = []
    for line in text.splitlines():
        line = line.strip()
        if "=" in line and len(line) > 2:
            candidates.append(line.split("=")[-1].strip())
    if candidates:
        return candidates[-1]
    nums = extract_numbers(text)
    return str(nums[-1]) if nums else ""


def _topk_probs(topk: list) -> np.ndarray:
    if not topk:
        return np.array([])
    lps = np.array([float(x[1]) for x in topk[:10]], dtype=float)
    p = np.exp(lps - lps.max())
    p = p / p.sum()
    return p


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if len(p) == 0 or len(q) == 0:
        return float("nan")
    n = min(len(p), len(q))
    p, q = p[:n], q[:n]
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log((a + 1e-12) / (b + 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------------------------------------------------------------------
# Per-run features
# ---------------------------------------------------------------------------


def final_answer_support(run: dict) -> dict:
    """A: final answer numbers/symbols supported by preceding reasoning."""
    ans = run.get("answer_normalized") or ""
    reasoning = _reasoning_text(run)
    ans_nums = extract_numbers(str(ans))
    reason_nums = extract_numbers(reasoning)
    num_support = 0.0
    if ans_nums:
        num_support = float(
            np.mean([any(math_equal(str(a), str(r)) for r in reason_nums) for a in ans_nums])
        )
    # symbol tokens in answer span
    ans_trace = _span_slice(run.get("token_trace") or [], run.get("answer_span"))
    sym_tokens = [
        t.get("token", "").strip()
        for t in ans_trace
        if _classify_token(t.get("token", "")) in ("symbol", "variable")
    ]
    sym_tokens = [t for t in sym_tokens if t and len(t) <= 20]
    sym_support = 0.0
    if sym_tokens:
        sym_support = float(np.mean([t.lower() in reasoning.lower() for t in sym_tokens]))
    score = 0.5 * num_support + 0.5 * sym_support if sym_tokens else num_support
    return {
        "final_answer_support": score,
        "final_num_support": num_support,
        "final_sym_support": sym_support,
    }


def last_equation_consistency(run: dict) -> dict:
    """D: last equation/expression vs final answer."""
    reasoning = _reasoning_text(run)
    last_eq = _extract_last_equation(reasoning)
    final = run.get("answer_normalized") or ""
    if not last_eq or not final:
        return {"last_eq_match_final": float("nan"), "last_eq_present": 0.0}
    match = float(math_equal_clean(last_eq, final) or math_equal(last_eq, final))
    return {"last_eq_match_final": match, "last_eq_present": 1.0}


def per_run_structured(run: dict) -> dict:
    out = {}
    out.update(final_answer_support(run))
    out.update(last_equation_consistency(run))
    return out


# ---------------------------------------------------------------------------
# Cross-run (branch-level)
# ---------------------------------------------------------------------------


def math_token_flip_rate(runs: list[dict], prefix: str) -> dict:
    """B: flip rate on numeric/symbol/variable tokens in answer span."""
    buckets: dict[float, list[tuple[str, str]]] = {}
    for run in runs:
        ans = _span_slice(run.get("token_trace") or [], run.get("answer_span"))
        n = len(ans)
        if n == 0:
            continue
        for i, tok in enumerate(ans):
            kind = _classify_token(tok.get("token", ""))
            if kind == "other":
                continue
            rp = round(i / max(n - 1, 1), 2)
            buckets.setdefault(rp, []).append((kind, tok.get("token", "").strip()))
    flips, numeric_flips, all_rates = [], [], []
    for items in buckets.values():
        if len(items) < 2:
            continue
        texts = [x[1] for x in items]
        kinds = [x[0] for x in items]
        uniq = len(set(texts))
        rate = 1.0 - 1.0 / uniq if uniq > 1 else 0.0
        all_rates.append(rate)
        num_texts = [t for t, k in zip(texts, kinds) if k == "numeric"]
        if num_texts:
            numeric_flips.append(1.0 if len(set(num_texts)) > 1 else 0.0)
    nan = float("nan")
    return {
        f"{prefix}_math_token_flip_mean": float(np.mean(all_rates)) if all_rates else nan,
        f"{prefix}_math_token_flip_max": float(np.max(all_rates)) if all_rates else nan,
        f"{prefix}_numeric_token_flip_rate": float(np.mean(numeric_flips)) if numeric_flips else nan,
    }


def answer_span_topk_stability(runs: list[dict], prefix: str) -> dict:
    """C: JS divergence of top-k at aligned answer-span positions."""
    bucket_topk: dict[float, list[np.ndarray]] = {}
    for run in runs:
        ans = _span_slice(run.get("token_trace") or [], run.get("answer_span"))
        n = len(ans)
        if n == 0:
            continue
        for i, tok in enumerate(ans):
            rp = round(i / max(n - 1, 1), 2)
            p = _topk_probs(tok.get("topk") or [])
            if len(p):
                bucket_topk.setdefault(rp, []).append(p)
    js_vals = []
    for dists in bucket_topk.values():
        if len(dists) < 2:
            continue
        for i in range(len(dists)):
            for j in range(i + 1, len(dists)):
                js_vals.append(_js_divergence(dists[i], dists[j]))
    nan = float("nan")
    return {
        f"{prefix}_ans_topk_js_mean": float(np.mean(js_vals)) if js_vals else nan,
        f"{prefix}_ans_topk_js_max": float(np.max(js_vals)) if js_vals else nan,
    }


def base_answer_topk_recall(base: dict, runs: list[dict], prefix: str) -> dict:
    """C: fraction of base answer-span tokens whose text appears in perturb top-k."""
    base_ans = _span_slice(base.get("token_trace") or [], base.get("answer_span"))
    if not base_ans or not runs:
        return {f"{prefix}_base_ans_topk_recall": float("nan")}
    recalls = []
    for run in runs:
        pert = _span_slice(run.get("token_trace") or [], run.get("answer_span"))
        n = min(len(base_ans), len(pert))
        if n == 0:
            continue
        hits = 0
        for i in range(n):
            base_tok = base_ans[i].get("token", "").strip()
            topk_texts = [x[0].strip() for x in (pert[i].get("topk") or [])]
            if base_tok and base_tok in topk_texts:
                hits += 1
        recalls.append(hits / n)
    return {
        f"{prefix}_base_ans_topk_recall": float(np.mean(recalls)) if recalls else float("nan"),
    }


def merge_reasoning_token_metrics(base: dict, text_runs: list[dict], weight_runs: list[dict]) -> dict:
    out: dict[str, float] = {}
    for prefix, runs in (("T", text_runs), ("W", weight_runs)):
        out.update(math_token_flip_rate(runs, prefix))
        out.update(answer_span_topk_stability(runs, prefix))
        out.update(base_answer_topk_recall(base, runs, prefix))
    # per-run support from base
    base_st = per_run_structured(base)
    out["base_final_answer_support"] = base_st["final_answer_support"]
    out["base_last_eq_match_final"] = base_st["last_eq_match_final"]
    out["base_final_num_support"] = base_st["final_num_support"]
    # branch avg support
    for prefix, runs in (("T", text_runs), ("W", weight_runs)):
        sup = [per_run_structured(r)["final_answer_support"] for r in runs]
        le = [per_run_structured(r)["last_eq_match_final"] for r in runs if r]
        out[f"{prefix}_final_answer_support_avg"] = float(np.mean(sup)) if sup else float("nan")
        valid_le = [x for x in le if x == x]
        out[f"{prefix}_last_eq_match_final_avg"] = float(np.mean(valid_le)) if valid_le else float("nan")
    return out
