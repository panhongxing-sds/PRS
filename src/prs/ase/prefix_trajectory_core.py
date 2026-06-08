"""Shared prefix trajectory: A_k (provisional ASE) and T_k (step AltMass)."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from prs.ase.cluster import cluster_answers
from prs.ase.numeric_trajectory import extract_numbers
from prs.ase.reasoning_token_features import _classify_token
from prs.ase.semantic_entropy import compute_ase

SPIKE_THRESH = 0.30
FRAG_RISE_THRESH = 0.05
PRE_FRAG_HORIZON = 3
PERSIST_RUN = 2


def reasoning_prefix_trace(run: dict) -> list[dict]:
    trace = run.get("token_trace") or []
    span = run.get("answer_span") or {}
    s = int(span.get("start_token", len(trace)))
    return trace[: max(0, s)]


def _step_end_indices(trace: list[dict], answer_start: int) -> list[int]:
    pref = trace[: max(0, answer_start)]
    ends: list[int] = []
    for i, tok in enumerate(pref):
        if "\n" in tok.get("token", ""):
            ends.append(i + 1)
    if pref and (not ends or ends[-1] < len(pref)):
        ends.append(len(pref))
    return ends


def _prefix_text(trace: list[dict], end: int) -> str:
    return "".join(t.get("token", "") for t in trace[:end])


def provisional_answer(trace: list[dict], end: int) -> str:
    text = _prefix_text(trace, end)
    nums = extract_numbers(text)
    if nums:
        return str(nums[-1])
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "=" in ln:
            rhs = ln.split("=")[-1].strip()
            if rhs:
                return rhs
    return ""


def token_altmass(topk: list, other_tokens: set[str]) -> float | None:
    if not topk or not other_tokens:
        return None
    total_p = 0.0
    alt_p = 0.0
    for tok_s, lp in topk:
        p = math.exp(float(lp))
        total_p += p
        if tok_s.strip() in other_tokens:
            alt_p += p
    if total_p <= 0:
        return None
    return alt_p / total_p


def step_altmass_by_type(
    runs: list[dict],
    labels: list[int],
    step_start: int,
    step_end: int,
) -> dict[str, float]:
    """AltMass split by token type in step."""
    cluster_toks: dict[int, set[str]] = defaultdict(set)
    for run, lab in zip(runs, labels):
        trace = reasoning_prefix_trace(run)
        for t in trace[step_start:step_end]:
            tok = t.get("token", "").strip()
            if tok and _classify_token(tok) != "other":
                cluster_toks[lab].add(tok)

    type_masses: dict[str, list[float]] = defaultdict(list)
    for run, lab in zip(runs, labels):
        trace = reasoning_prefix_trace(run)
        other: set[str] = set()
        for c, toks in cluster_toks.items():
            if c != lab:
                other |= toks
        for i, t in enumerate(trace[step_start:step_end]):
            tok = t.get("token", "").strip()
            kind = _classify_token(tok)
            if kind == "other":
                continue
            m = token_altmass(t.get("topk") or [], other)
            if m is None:
                continue
            abs_i = step_start + i
            trace_full = reasoning_prefix_trace(run)
            is_eq = tok in ("=", "≈") or any(
                trace_full[j].get("token", "").strip() in ("=", "≈")
                for j in range(max(0, abs_i - 2), min(len(trace_full), abs_i + 3))
            )
            if is_eq:
                bucket = "equation"
            elif kind == "numeric":
                bucket = "numeric"
            elif kind in ("symbol", "variable"):
                bucket = "operator"
            else:
                bucket = "other_math"
            type_masses[bucket].append(m)

    return {k: float(np.mean(v)) if v else float("nan") for k, v in type_masses.items()}


def step_altmass(
    runs: list[dict],
    labels: list[int],
    step_start: int,
    step_end: int,
) -> float:
    by_type = step_altmass_by_type(runs, labels, step_start, step_end)
    vals = [v for v in by_type.values() if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def align_steps(runs: list[dict]) -> tuple[list[tuple[int, int]], int]:
    base = runs[0]
    trace = base.get("token_trace") or []
    ans_start = int((base.get("answer_span") or {}).get("start_token", len(trace)))
    ends = _step_end_indices(trace, ans_start)
    if not ends:
        return [], ans_start
    ranges: list[tuple[int, int]] = []
    prev = 0
    for e in ends:
        ranges.append((prev, e))
        prev = e
    return ranges, ans_start


def run_aligned_step_end(run: dict, k: int, ans_start: int) -> int:
    trace = run.get("token_trace") or []
    ends = _step_end_indices(trace, ans_start)
    if not ends:
        return 0
    return ends[k] if k < len(ends) else ends[-1]


def trajectory_for_sample(wruns: list[dict]) -> dict | None:
    if len(wruns) < 2:
        return None
    ref_ranges, ans_start = align_steps(wruns)
    if not ref_ranges:
        return None

    final_answers = [r.get("answer_normalized", "") for r in wruns]
    labels, _ = cluster_answers(final_answers)

    K = len(ref_ranges)
    A: list[float] = []
    nclust: list[int] = []
    T: list[float] = []
    T_by_type: list[dict[str, float]] = []

    for k in range(K):
        provs = []
        for run in wruns:
            end = run_aligned_step_end(run, k, ans_start)
            provs.append(provisional_answer(run.get("token_trace") or [], end))
        ase_k = compute_ase(provs)
        A.append(float(ase_k["U"]))
        nclust.append(int(ase_k["num_clusters"]))

        st, en = ref_ranges[k]
        T.append(step_altmass(wruns, labels, st, en))
        T_by_type.append(step_altmass_by_type(wruns, labels, st, en))

    dT = [float("nan")] + [T[k] - T[k - 1] if T[k] == T[k] and T[k - 1] == T[k - 1] else float("nan") for k in range(1, K)]
    dA = [float("nan")] + [A[k] - A[k - 1] if A[k] == A[k] and A[k - 1] == A[k - 1] else float("nan") for k in range(1, K)]

    return {
        "A": A,
        "T": T,
        "dT": dT,
        "dA": dA,
        "nclust": nclust,
        "K": K,
        "T_by_type": T_by_type,
        "ranges": ref_ranges,
    }


def first_a_event(A: list[float], nc: list[int]) -> int | None:
    """First step k where A spikes or clusters increase."""
    K = len(A)
    for k in range(K):
        if A[k] == A[k] and A[k] >= SPIKE_THRESH:
            return k
        if k + 1 < K and nc[k + 1] > nc[k]:
            return k + 1
        if k + 1 < K and A[k] == A[k] and A[k + 1] == A[k + 1]:
            if A[k + 1] - A[k] > FRAG_RISE_THRESH:
                return k + 1
    return None


def first_t_spike(T: list[float], thresh: float = SPIKE_THRESH) -> int | None:
    for k, v in enumerate(T):
        if v == v and v >= thresh:
            return k
    return None


def spike_location_label(T_by_type: dict[str, float]) -> str:
    buckets = {k: v for k, v in T_by_type.items() if v == v}
    if not buckets:
        return "none"
    return max(buckets, key=buckets.get)


def relative_bin(k: int, K: int) -> str:
    if K <= 1:
        return "early"
    r = k / (K - 1)
    if r <= 1 / 3:
        return "early"
    if r <= 2 / 3:
        return "mid"
    return "late"
