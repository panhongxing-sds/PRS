"""Decomposed AltMass: final answer commitment vs reasoning-internal competition."""

from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from panda.core.cluster import cluster_answers
from panda.core.reasoning_token_features import _classify_token, _span_slice


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


# Domain-aware content-token selection for the A (per-domain) ablation of S_tr.
# Logic tasks: keep content words, drop English stopwords/pure punctuation.
# Code tasks: keep identifiers, keywords, operators, brackets, literals.
_LOGIC_STOPWORDS = frozenset(
    "the a an of to in on at and or but if then so is are was were be been being "
    "we i it he she they you that this these those there here as for with from by "
    "will would can could may might must shall should do does did has have had".split()
)
_CODE_KEEP = re.compile(r"[A-Za-z_]\w*|[(){}\[\];:,.]|[-+*/%=<>!&|^~]+|\d+")


def _domain_content_indices(trace: list[dict], domain: str) -> set[int]:
    """Indices of 'decision-bearing' tokens per domain (A ablation)."""
    if domain == "math":
        return {i for i, t in enumerate(trace) if _classify_token(t.get("token", "")) != "other"}
    if domain == "code":
        idxs = set()
        for i, t in enumerate(trace):
            s = t.get("token", "").strip()
            if s and _CODE_KEEP.search(s):
                idxs.add(i)
        return idxs
    if domain == "logic":
        idxs = set()
        for i, t in enumerate(trace):
            s = t.get("token", "").strip()
            if not s or s.isspace():
                continue
            if s.lower() in _LOGIC_STOPWORDS:
                continue
            if re.search(r"[A-Za-z0-9]", s):
                idxs.add(i)
        return idxs
    # unknown domain → all reasoning tokens
    return set(range(len(trace)))


def _dataset_domain(dataset: str | None) -> str:
    ds = (dataset or "").lower()
    if ds in {"humaneval", "mbpp"} or "code" in ds:
        return "code"
    if ds in {"leg_counting", "zebra_puzzles", "color_cube"} or "logic" in ds:
        return "logic"
    return "math"


def _local_spread_topk(
    runs: list[dict],
    *,
    top_pct: float = 0.10,
    indices_fn=None,
) -> float:
    """
    Domain-agnostic trace drift (B): mean of the top-k% highest local-spread
    reasoning tokens per run, averaged across runs.

    ``indices_fn(trace) -> set[int]`` optionally restricts positions (A ablation).
    """
    if not runs:
        return float("nan")
    per_run: list[float] = []
    for run in runs:
        trace = _reasoning_trace(run)
        if not trace:
            # Short-answer tasks (e.g. color_cube) have no separate reasoning span;
            # fall back to the full token trace so the local-spread signal (answer-token
            # alternative mass under perturbation) is still defined instead of nan.
            trace = run.get("token_trace") or []
        if not trace:
            continue
        pos_ok = indices_fn(trace) if indices_fn is not None else set(range(len(trace)))
        spreads: list[float] = []
        for i, t in enumerate(trace):
            if i not in pos_ok:
                continue
            m = _local_topk_spread(t.get("topk") or [], t.get("token", ""))
            if m is not None:
                spreads.append(m)
        if not spreads:
            continue
        spreads.sort(reverse=True)
        k = max(1, int(math.ceil(len(spreads) * top_pct)))
        per_run.append(float(np.mean(spreads[:k])))
    return float(np.mean(per_run)) if per_run else float("nan")


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


def drift_form_variants(
    runs: list[dict],
    *,
    top_pct: float = 0.10,
    prefix: str = "",
) -> dict[str, float]:
    """Alternative mathematical forms of D_reason / D_ans for later form-selection
    analysis. Same raw quantities as the primary metrics, different aggregations.

    D_reason family (per-run local-spread sequence over the reasoning trace):
      Dr_q90        — mean over runs of the 90th-percentile spread
      Dr_max        — mean of per-run max spread
      Dr_lse        — mean of per-run soft-max (log-sum-exp, tau=0.1)
      Dr_late_w     — position-weighted mean (w ∝ (i/n)^2; late drift counts more)
      Dr_fork_density — fraction of tokens with spread > 0.3 (decision-fork rate)
      Dr_run_std    — std across runs of per-run top-k%% mean (drift instability)

    D_ans family (final answer span / answer clusters):
      Da_final_max  — mean of per-run max token alt-mass inside the final span
      Da_margin     — 1 - (top1 - top2 cluster mass): close-competition form
      Da_minority   — mean spread over minority-cluster runs only
    """
    nan = float("nan")
    keys = ("Dr_q90", "Dr_max", "Dr_lse", "Dr_late_w", "Dr_fork_density", "Dr_run_std",
            "Da_final_max", "Da_margin", "Da_minority")
    if len(runs) < 2:
        return {prefix + k: nan for k in keys}

    answers = [r.get("answer_normalized", "") for r in runs]
    labels, sizes = cluster_answers(answers)
    n_runs = len(runs)

    per_run_spreads: list[list[float]] = []
    per_run_topk_mean: list[float] = []
    q90s, maxs, lses, late_ws, forks = [], [], [], [], []
    for run in runs:
        trace = _reasoning_trace(run)
        spreads, pos = [], []
        for i, t in enumerate(trace):
            m = _local_topk_spread(t.get("topk") or [], t.get("token", ""))
            if m is not None:
                spreads.append(m)
                pos.append(i)
        if not spreads:
            continue
        arr = np.asarray(spreads)
        per_run_spreads.append(spreads)
        srt = np.sort(arr)[::-1]
        k = max(1, int(math.ceil(len(arr) * top_pct)))
        per_run_topk_mean.append(float(srt[:k].mean()))
        q90s.append(float(np.quantile(arr, 0.9)))
        maxs.append(float(arr.max()))
        tau = 0.1
        lses.append(float(tau * np.log(np.mean(np.exp(arr / tau)))))
        w = (np.asarray(pos, float) / max(1, len(trace))) ** 2
        late_ws.append(float((arr * w).sum() / (w.sum() + 1e-12)))
        forks.append(float((arr > 0.3).mean()))

    # D_ans family
    final_maxs = []
    minority_spreads = []
    majority_label = max(sizes, key=sizes.get)
    for run, lab in zip(runs, labels):
        span_toks = _span_slice(run.get("token_trace") or [], run.get("answer_span"))
        ms = [m for t in span_toks
              if (m := _local_topk_spread(t.get("topk") or [], t.get("token", ""))) is not None]
        if ms:
            final_maxs.append(float(max(ms)))
            if lab != majority_label:
                minority_spreads.append(float(np.mean(ms)))
    masses = sorted((c / n_runs for c in sizes.values()), reverse=True)
    margin = 1.0 - (masses[0] - (masses[1] if len(masses) > 1 else 0.0))

    def _m(x):
        return float(np.mean(x)) if x else nan

    return {
        prefix + "Dr_q90": _m(q90s),
        prefix + "Dr_max": _m(maxs),
        prefix + "Dr_lse": _m(lses),
        prefix + "Dr_late_w": _m(late_ws),
        prefix + "Dr_fork_density": _m(forks),
        prefix + "Dr_run_std": float(np.std(per_run_topk_mean)) if len(per_run_topk_mean) > 1 else nan,
        prefix + "Da_final_max": _m(final_maxs),
        prefix + "Da_margin": float(margin),
        prefix + "Da_minority": _m(minority_spreads) if minority_spreads else 0.0,
    }


def altmass_variants_weight_branch(
    weight_runs: list[dict],
    dataset: str | None = None,
    *,
    top_pct: float = 0.10,
) -> dict[str, float]:
    """Compute decomposed AltMass features on weight-perturbed runs.

    Trace-drift (S_tr) keys:
      - ``AltMass_local_spread_topk``    — B (primary): domain-agnostic top-k%%.
      - ``AltMass_local_spread_content`` — A (ablation): per-domain content tokens.
      - ``AltMass_local_spread_reason``  — legacy math-token mean (backward compat).
    """
    nan = float("nan")
    keys = (
        "AltMass_final", "AltMass_reason_all", "AltMass_numeric",
        "AltMass_equation", "AltMass_step_end", "AltMass_late_reason",
        "AltMass_local_spread_reason", "AltMass_local_spread_topk",
        "AltMass_local_spread_content",
    )
    if len(weight_runs) < 2:
        return {k: nan for k in keys}
    answers = [r.get("answer_normalized", "") for r in weight_runs]
    labels, sizes = cluster_answers(answers)

    domain = _dataset_domain(dataset)
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
        # B (primary): domain-agnostic, top-k%% highest local spread
        "AltMass_local_spread_topk": _local_spread_topk(weight_runs, top_pct=top_pct),
        # A (ablation): per-domain content tokens, top-k%% within them
        "AltMass_local_spread_content": _local_spread_topk(
            weight_runs,
            top_pct=top_pct,
            indices_fn=lambda tr: _domain_content_indices(tr, domain),
        ),
    }
