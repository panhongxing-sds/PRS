"""Prefix re-query stage signals (CONST-style procedural volatility).

T_s = local token spread in reasoning segment s (T_math = math-relevant only)
B_entropy = prefix-internal answer entropy H(p_{i,s}) — often degenerates at low T
B_shift = cross-prefix answer drift JSD(p_{i,s}, p_{i,s-1}) averaged over runs
B_flip = cross-prefix majority-answer flip 1[argmax p_s != argmax p_{s-1}]
A_s = hard response fragmentation (entropy of K run majority answers at stage s)
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

import numpy as np

from prs.ase.altmass_decomposition import _local_topk_spread
from prs.ase.reasoning_token_features import _classify_token
from prs.ase.semantic_entropy import compute_ase

MATH_KINDS = frozenset({"numeric", "symbol", "variable", "unit"})

REQUERY_USER_SUFFIX = (
    "\n\nReasoning so far:\n{prefix}\n\n"
    "Given the problem and the reasoning prefix so far, output the final answer directly. "
    "Do not continue the reasoning. Only output the answer within \\boxed{{}}."
)


def build_prefix_requery_prompt(question: str, prefix_text: str, tokenizer, model_path: str = "") -> str:
    from prs.grading.tokur_records import build_prompt_tfb

    user = question.strip() + REQUERY_USER_SUFFIX.format(prefix=prefix_text.strip())
    return build_prompt_tfb(user, tokenizer, model_path)


def reasoning_trace(run: dict) -> list[dict]:
    trace = run.get("token_trace") or []
    span = run.get("answer_span") or {}
    s = int(span.get("start_token", len(trace)))
    return trace[: max(0, s)]


def prefix_text_from_trace(trace: list[dict], end: int) -> str:
    return "".join(t.get("token", "") for t in trace[: max(0, end)])


def stage_ranges(L: int, S: int) -> list[tuple[int, int]]:
    """Stage s: segment (seg_start, prefix_end) token indices, 0-indexed."""
    if L <= 0 or S <= 0:
        return []
    out: list[tuple[int, int]] = []
    prev = 0
    for s in range(1, S + 1):
        end = min(L, max(prev + 1, int(math.ceil(s * L / S))))
        out.append((prev, end))
        prev = end
        if end >= L:
            while len(out) < S:
                out.append((L, L))
            break
    while len(out) < S:
        out.append((L, L))
    return out[:S]


def _is_equation_ctx(trace: list[dict], i: int) -> bool:
    tok = trace[i].get("token", "").strip()
    if tok in ("=", "≈"):
        return True
    for j in range(max(0, i - 2), min(len(trace), i + 3)):
        if trace[j].get("token", "").strip() in ("=", "≈"):
            return True
    return False


def segment_local_spread(
    runs: list[dict],
    seg_start: int,
    seg_end: int,
    *,
    math_only: bool = False,
    generic_only: bool = False,
) -> float:
    vals: list[float] = []
    for run in runs:
        trace = reasoning_trace(run)
        for i in range(seg_start, min(seg_end, len(trace))):
            t = trace[i]
            topk = t.get("topk") or []
            m = _local_topk_spread(topk, t.get("token", ""))
            if m is None:
                continue
            kind = _classify_token(t.get("token", ""))
            is_math = kind in MATH_KINDS or _is_equation_ctx(trace, i)
            if generic_only:
                if kind != "other":
                    continue
            elif math_only:
                if not is_math:
                    continue
            vals.append(m)
    return float(np.mean(vals)) if vals else float("nan")


def answer_distribution(answers: list[str]) -> dict[str, float]:
    xs = [a.strip() for a in answers if a and str(a).strip()]
    if not xs:
        return {}
    c = Counter(xs)
    m = len(xs)
    return {k: v / m for k, v in c.items()}


def answer_distribution_entropy(answers: list[str]) -> float:
    """Normalized entropy H/log(M) for M re-query answers (empty stripped)."""
    xs = [a.strip() for a in answers if a and str(a).strip()]
    if not xs:
        return float("nan")
    c = Counter(xs)
    m = len(xs)
    h = 0.0
    for cnt in c.values():
        p = cnt / m
        if p > 0:
            h -= p * math.log(p)
    denom = math.log(m) if m > 1 else 1.0
    return h / denom


def distribution_jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Symmetric JSD in [0, 1] (log base 2)."""
    if not p or not q:
        return float("nan")
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        s = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            bk = b.get(k, 0.0)
            if ak > 0 and bk > 0:
                s += ak * math.log2(ak / bk)
        return s

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def prefix_drift_jsd(answers_prev: list[str], answers_curr: list[str]) -> float:
    return distribution_jsd(answer_distribution(answers_prev), answer_distribution(answers_curr))


def prefix_drift_flip(answers_prev: list[str], answers_curr: list[str]) -> float:
    m0 = majority_answer(answers_prev)
    m1 = majority_answer(answers_curr)
    if not m0 or not m1:
        return float("nan")
    return 1.0 if m0 != m1 else 0.0


def soft_drift_B(requery_by_run: list[list[str]]) -> float:
    """Mean normalized entropy across K runs at one stage."""
    ent = [answer_distribution_entropy(ans) for ans in requery_by_run]
    ent = [e for e in ent if e == e]
    return float(np.mean(ent)) if ent else float("nan")


def cross_prefix_drift(
    requery: dict[int, dict[int, list[str]]],
    K: int,
    stage: int,
) -> tuple[float, float]:
    """Mean JSD and flip rate across runs between stage and stage-1."""
    if stage <= 0:
        return 0.0, 0.0
    jsds: list[float] = []
    flips: list[float] = []
    for i in range(K):
        prev = requery.get(i, {}).get(stage - 1, [])
        curr = requery.get(i, {}).get(stage, [])
        j = prefix_drift_jsd(prev, curr)
        f = prefix_drift_flip(prev, curr)
        if j == j:
            jsds.append(j)
        if f == f:
            flips.append(f)
    b_jsd = float(np.mean(jsds)) if jsds else float("nan")
    b_flip = float(np.mean(flips)) if flips else float("nan")
    return b_jsd, b_flip


def hard_fragmentation_A(majority_answers: list[str]) -> float:
    """ASE H_norm across K run-prefix majority answers."""
    xs = [a for a in majority_answers if a is not None and str(a).strip()]
    if len(xs) < 2:
        return 0.0
    ase = compute_ase(xs)
    return float(ase.get("H_norm", float("nan")))


def majority_answer(answers: list[str]) -> str:
    xs = [a.strip() for a in answers if a and str(a).strip()]
    if not xs:
        return ""
    return Counter(xs).most_common(1)[0][0]


def stage_signals(
    runs: list[dict],
    requery: dict[int, dict[int, list[str]]],
    S: int,
) -> dict[str, list[float]]:
    """requery[run_idx][stage_idx] = list of M answers."""
    K = len(runs)
    T: list[float] = []
    T_math: list[float] = []
    T_gen: list[float] = []
    B: list[float] = []
    B_shift: list[float] = []
    B_flip: list[float] = []
    A: list[float] = []
    L_ref = len(reasoning_trace(runs[0])) if runs else 0
    ranges = stage_ranges(L_ref, S)

    for s, (seg_start, seg_end) in enumerate(ranges):
        T.append(segment_local_spread(runs, seg_start, seg_end))
        T_math.append(segment_local_spread(runs, seg_start, seg_end, math_only=True))
        T_gen.append(segment_local_spread(runs, seg_start, seg_end, generic_only=True))

        by_run = [requery.get(i, {}).get(s, []) for i in range(K)]
        B.append(soft_drift_B(by_run))
        jsd_s, flip_s = cross_prefix_drift(requery, K, s)
        B_shift.append(jsd_s)
        B_flip.append(flip_s)
        majors = [majority_answer(by_run[i]) for i in range(K)]
        A.append(hard_fragmentation_A(majors))

    return {
        "T": T,
        "T_math": T_math,
        "T_gen": T_gen,
        "B": B,
        "B_entropy": B,
        "B_shift": B_shift,
        "B_flip": B_flip,
        "A": A,
        "ranges": ranges,
    }


def first_above(series: list[float], thresh: float, *, inclusive: bool = False) -> int | None:
    for i, v in enumerate(series):
        if v != v:
            continue
        if inclusive:
            if v >= thresh:
                return i
        elif v > thresh:
            return i
    return None


def event_times(
    signals: dict[str, list[float]],
    q_T: float,
    q_B: float,
    q_A: float,
    *,
    t_key: str = "T",
    b_key: str = "B",
    b_inclusive: bool = False,
) -> dict[str, int | None]:
    t_series = signals.get(t_key, signals["T"])
    b_series = signals.get(b_key, signals["B"])
    return {
        "tau_T": first_above(t_series, q_T),
        "tau_B": first_above(b_series, q_B, inclusive=b_inclusive),
        "tau_A": first_above(signals["A"], q_A),
        "tau_T_math": first_above(signals["T_math"], q_T),
        "tau_T_gen": first_above(signals["T_gen"], q_T),
    }


def order_stats(events: list[dict[str, int | None]]) -> dict[str, float]:
    n = len(events)
    if n == 0:
        return {}
    def rate(key: str) -> float:
        ok = 0
        for e in events:
            a, b, c = e.get("tau_T"), e.get("tau_B"), e.get("tau_A")
            if key == "T<B<A" and a is not None and b is not None and c is not None and a < b < c:
                ok += 1
            elif key == "T<B" and a is not None and b is not None and a < b:
                ok += 1
            elif key == "B<A" and b is not None and c is not None and b < c:
                ok += 1
            elif key == "T<A" and a is not None and c is not None and a < c:
                ok += 1
        return ok / n

    return {
        "P_T_lt_B_lt_A": rate("T<B<A"),
        "P_T_lt_B": rate("T<B"),
        "P_B_lt_A": rate("B<A"),
        "P_T_lt_A": rate("T<A"),
        "n": float(n),
    }


def shuffle_null_t(events: list[dict], signals_list: list[dict], q_T: float, rng: random.Random) -> dict[str, float]:
    """Shuffle T_s stage order within each question."""
    shuffled: list[dict[str, int | None]] = []
    for sig in signals_list:
        T = list(sig["T"])
        rng.shuffle(T)
        fake = {"T": T, "T_math": sig["T_math"], "T_gen": sig["T_gen"], "B": sig["B"], "A": sig["A"]}
        shuffled.append(event_times(fake, q_T, first_above(sig["B"], q_T) or 0, first_above(sig["A"], q_T) or 0))
    return order_stats(shuffled)


def dataset_thresholds(
    all_signals: list[dict[str, list[float]]],
    q: float = 0.75,
    *,
    t_key: str = "T",
    b_key: str = "B",
    b_stage_min: int = 0,
) -> tuple[float, float, float]:
    flat_T = [v for s in all_signals for v in s.get(t_key, s["T"]) if v == v]
    flat_B = [
        v
        for s in all_signals
        for i, v in enumerate(s.get(b_key, s["B"]))
        if v == v and i >= b_stage_min
    ]
    flat_A = [v for s in all_signals for v in s["A"] if v == v]
    qT = float(np.quantile(flat_T, q)) if flat_T else 0.3
    qB = float(np.quantile(flat_B, q)) if flat_B else 0.3
    qA = float(np.quantile(flat_A, q)) if flat_A else 0.3
    return qT, qB, qA


def propagation_ladder_key(t: int | None, b: int | None, a: int | None, *, b_label: str = "B") -> str:
    if t is None and b is None and a is None:
        return "none"
    if t is not None and b is None and a is None:
        return "T-only"
    if t is not None and b is not None and a is None and t < b:
        return f"T→{b_label}"
    if b is not None and a is not None and t is None and b < a:
        return f"{b_label}→A"
    if t is not None and b is not None and a is not None and t < b < a:
        return f"T→{b_label}→A"
    if t is not None and a is not None and b is None and t < a:
        return "T→A"
    if a is not None and t is None and b is None:
        return "A-only"
    return "other"
