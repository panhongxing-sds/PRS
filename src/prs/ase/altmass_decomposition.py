"""Decomposed AltMass: final answer commitment vs reasoning-internal competition."""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from prs.ase.cluster import cluster_answers
from prs.ase.reasoning_token_features import _classify_token, _span_slice


def _reasoning_trace(run: dict) -> list[dict]:
    trace = run.get("token_trace") or []
    span = run.get("answer_span") or {}
    if span.get("start_token") is not None:
        return trace[: max(0, int(span["start_token"]))]
    return trace


def _token_altmass(topk: list, other_tokens: set[str]) -> float | None:
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


def _local_topk_spread(topk: list, chosen: str) -> float | None:
    """Mass on top-k alternatives excluding the realized token."""
    if not topk:
        return None
    total_p = 0.0
    alt_p = 0.0
    chosen = chosen.strip()
    for tok_s, lp in topk:
        p = math.exp(float(lp))
        total_p += p
        if tok_s.strip() != chosen:
            alt_p += p
    return (alt_p / total_p) if total_p > 0 else None


def _step_end_indices(trace: list[dict]) -> set[int]:
    """Indices of last 2 math tokens before each newline in reasoning trace."""
    idxs: set[int] = set()
    line_math: list[int] = []
    for i, t in enumerate(trace):
        tok = t.get("token", "")
        if "\n" in tok:
            if line_math:
                for j in line_math[-2:]:
                    idxs.add(j)
            line_math = []
            continue
        if _classify_token(tok) != "other":
            line_math.append(i)
    if line_math:
        for j in line_math[-2:]:
            idxs.add(j)
    return idxs


def _equation_indices(trace: list[dict]) -> set[int]:
    """Tokens at '=' and adjacent numeric/symbol tokens."""
    idxs: set[int] = set()
    for i, t in enumerate(trace):
        tok = t.get("token", "").strip()
        if tok == "=" or tok == "≈":
            for j in range(max(0, i - 2), min(len(trace), i + 3)):
                idxs.add(j)
    return idxs


def _late_indices(trace: list[dict], frac: float = 1 / 3) -> set[int]:
    n = len(trace)
    if n == 0:
        return set()
    start = int(n * (1 - frac))
    return set(range(start, n))


def _cluster_other_token_sets(
    runs: list[dict],
    labels: list[int],
    *,
    region: str,
) -> dict[int, set[str]]:
    """Per-cluster token string sets from answer span or reasoning."""
    out: dict[int, set[str]] = defaultdict(set)
    for run, lab in zip(runs, labels):
        if region == "final":
            toks = _span_slice(run.get("token_trace") or [], run.get("answer_span"))
        else:
            toks = _reasoning_trace(run)
        for t in toks:
            s = t.get("token", "").strip()
            if not s:
                continue
            if region == "numeric" and _classify_token(s) != "numeric":
                continue
            if region == "reason_all":
                if _classify_token(s) == "other":
                    continue
            out[lab].add(s)
    return out


def _aggregate_altmass(
    runs: list[dict],
    labels: list[int],
    *,
    region: str,
    position_filter: set[int] | None = None,
    use_local_spread: bool = False,
) -> float:
    """Cross-cluster AltMass on selected token positions."""
    if len(runs) < 2:
        return float("nan")
    _, sizes = cluster_answers([r.get("answer_normalized", "") for r in runs])
    if len(sizes) < 2 and not use_local_spread:
        return 0.0

    cluster_toks = _cluster_other_token_sets(runs, labels, region="final" if region == "final" else "reason_all")
    if region == "numeric":
        cluster_toks = _cluster_other_token_sets(runs, labels, region="numeric")

    per_run = []
    for run, lab in zip(runs, labels):
        if region == "final":
            trace = _span_slice(run.get("token_trace") or [], run.get("answer_span"))
        else:
            trace = _reasoning_trace(run)

        if region == "step_end":
            pos_ok = _step_end_indices(trace)
        elif region == "equation":
            pos_ok = _equation_indices(trace)
        elif region == "late_reason":
            pos_ok = _late_indices(trace)
        elif region == "numeric":
            pos_ok = {i for i, t in enumerate(trace) if _classify_token(t.get("token", "")) == "numeric"}
        elif region == "reason_all":
            pos_ok = {i for i, t in enumerate(trace) if _classify_token(t.get("token", "")) != "other"}
        else:
            pos_ok = set(range(len(trace)))

        if position_filter is not None:
            pos_ok &= position_filter

        other = set()
        for c, toks in cluster_toks.items():
            if c != lab:
                other |= toks

        masses = []
        for i, t in enumerate(trace):
            if i not in pos_ok:
                continue
            topk = t.get("topk") or []
            if use_local_spread:
                m = _local_topk_spread(topk, t.get("token", ""))
            else:
                m = _token_altmass(topk, other)
            if m is not None:
                masses.append(m)
        if masses:
            per_run.append(float(np.mean(masses)))
    return float(np.mean(per_run)) if per_run else float("nan")


def altmass_variants_weight_branch(weight_runs: list[dict]) -> dict[str, float]:
    """Compute decomposed AltMass features on weight-perturbed runs."""
    nan = float("nan")
    if len(weight_runs) < 2:
        return {k: nan for k in (
            "AltMass_final", "AltMass_reason_all", "AltMass_numeric",
            "AltMass_equation", "AltMass_step_end", "AltMass_late_reason",
            "AltMass_local_spread_reason",
        )}
    answers = [r.get("answer_normalized", "") for r in weight_runs]
    labels, sizes = cluster_answers(answers)

    return {
        "AltMass_final": _aggregate_altmass(weight_runs, labels, region="final"),
        "AltMass_reason_all": _aggregate_altmass(weight_runs, labels, region="reason_all"),
        "AltMass_numeric": _aggregate_altmass(weight_runs, labels, region="numeric"),
        "AltMass_equation": _aggregate_altmass(weight_runs, labels, region="equation"),
        "AltMass_step_end": _aggregate_altmass(weight_runs, labels, region="step_end"),
        "AltMass_late_reason": _aggregate_altmass(weight_runs, labels, region="late_reason"),
        "AltMass_local_spread_reason": _aggregate_altmass(
            weight_runs, labels, region="reason_all", use_local_spread=True
        ),
    }
