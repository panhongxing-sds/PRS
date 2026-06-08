"""Advanced token metrics: answer-span ATU, cross-run disagreement, base deltas."""

from __future__ import annotations

import math

import numpy as np


def _span_slice(trace: list[dict], span: dict | None) -> list[dict]:
    if not trace:
        return []
    span = span or {}
    s = int(span.get("start_token", 0))
    e = int(span.get("end_token", len(trace) - 1))
    s = max(0, min(s, len(trace) - 1))
    e = max(s, min(e, len(trace) - 1))
    return trace[s : e + 1]


def _trace_arrays(trace: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ent = np.array([t.get("entropy", 0.0) for t in trace], dtype=float)
    mar = np.array([t.get("margin_top2", 0.0) for t in trace], dtype=float)
    lp = np.array([t.get("logprob", 0.0) for t in trace], dtype=float)
    rk = np.array([t.get("rank", 1) for t in trace], dtype=float)
    return ent, mar, lp, rk


def per_generation_answer_stats(run: dict, top_pct: float = 0.10) -> dict:
    """B: ATU restricted to answer_span tokens only."""
    trace = run.get("token_trace") or []
    ans = _span_slice(trace, run.get("answer_span"))
    if not ans:
        nan = float("nan")
        return {
            "ans_n_tokens": 0,
            "ans_ent_sum": nan,
            "ans_ent_mean": nan,
            "ans_ent_max": nan,
            "ans_ent_top10_sum": nan,
            "ans_ent_top10_mean": nan,
            "ans_mar_mean": nan,
            "ans_logprob_min": nan,
            "ans_rank_max": nan,
        }
    ent, mar, lp, rk = _trace_arrays(ans)
    k = max(1, int(math.ceil(len(ent) * top_pct)))
    top_ent = np.partition(ent, -k)[-k:]
    return {
        "ans_n_tokens": len(ans),
        "ans_ent_sum": float(ent.sum()),
        "ans_ent_mean": float(ent.mean()),
        "ans_ent_max": float(ent.max()),
        "ans_ent_top10_sum": float(top_ent.sum()),
        "ans_ent_top10_mean": float(top_ent.mean()),
        "ans_mar_mean": float(mar.mean()),
        "ans_logprob_min": float(lp.min()),
        "ans_rank_max": float(rk.max()),
    }


def cross_run_token_disagreement(runs: list[dict]) -> dict:
    """A: token_id flip rate and cross-run entropy variance in answer span."""
    buckets: dict[float, list[int]] = {}
    ent_by_pos: dict[float, list[float]] = {}
    for run in runs:
        trace = run.get("token_trace") or []
        ans = _span_slice(trace, run.get("answer_span"))
        n = len(ans)
        if n == 0:
            continue
        for i, tok in enumerate(ans):
            rp = round(i / max(n - 1, 1), 2)
            buckets.setdefault(rp, []).append(int(tok.get("token_id", 0)))
            ent_by_pos.setdefault(rp, []).append(float(tok.get("entropy", 0.0)))
    disagree, ent_vars = [], []
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        bc = np.bincount(np.array(ids, dtype=int))
        disagree.append(1.0 - bc.max() / len(ids))
    for ents in ent_by_pos.values():
        if len(ents) >= 2:
            ent_vars.append(float(np.var(ents)))
    nan = float("nan")
    if not disagree:
        return {
            "tok_disagree_mean": nan,
            "tok_disagree_max": nan,
            "tok_ent_xrun_var_mean": nan,
            "tok_ent_xrun_var_max": nan,
        }
    return {
        "tok_disagree_mean": float(np.mean(disagree)),
        "tok_disagree_max": float(np.max(disagree)),
        "tok_ent_xrun_var_mean": float(np.mean(ent_vars)) if ent_vars else nan,
        "tok_ent_xrun_var_max": float(np.max(ent_vars)) if ent_vars else nan,
    }


def base_perturb_delta(base: dict, runs: list[dict]) -> dict:
    """C: |entropy_base - entropy_perturb| aggregated over answer span."""
    base_st = per_generation_answer_stats(base)
    run_st = [per_generation_answer_stats(r) for r in runs]
    run_st = [s for s in run_st if s.get("ans_n_tokens", 0) > 0]
    nan = float("nan")
    if not run_st or base_st.get("ans_n_tokens", 0) == 0:
        return {
            "delta_ans_ent_mean_mean": nan,
            "delta_ans_ent_mean_max": nan,
            "delta_ans_ent_max_mean": nan,
            "delta_ans_ent_max_max": nan,
        }
    d_mean, d_max = [], []
    for s in run_st:
        d_mean.append(abs(s["ans_ent_mean"] - base_st["ans_ent_mean"]))
        d_max.append(abs(s["ans_ent_max"] - base_st["ans_ent_max"]))
    return {
        "delta_ans_ent_mean_mean": float(np.mean(d_mean)),
        "delta_ans_ent_mean_max": float(np.max(d_mean)),
        "delta_ans_ent_max_mean": float(np.mean(d_max)),
        "delta_ans_ent_max_max": float(np.max(d_max)),
    }


def base_stable_features(base: dict) -> dict:
    """D: base-only signals for stable-wrong detection."""
    st = per_generation_answer_stats(base)
    trace = base.get("token_trace") or []
    full_ent, _, full_lp, full_rk = _trace_arrays(trace) if trace else (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )
    nan = float("nan")
    out = {f"base_{k}": v for k, v in st.items()}
    if len(full_ent):
        out["base_full_ent_mean"] = float(full_ent.mean())
        out["base_full_ent_max"] = float(full_ent.max())
        out["base_full_logprob_min"] = float(full_lp.min())
        out["base_full_rank_gt1_frac"] = float(np.mean(full_rk > 1))
    else:
        out["base_full_ent_mean"] = nan
        out["base_full_ent_max"] = nan
        out["base_full_logprob_min"] = nan
        out["base_full_rank_gt1_frac"] = nan
    return out


def _pool_stats(stats: list[dict], prefix: str, keys: list[str]) -> dict:
    valid = [s for s in stats if s.get("ans_n_tokens", 0) > 0]
    out: dict[str, float] = {}
    nan = float("nan")
    if not valid:
        for key in keys:
            out[f"{prefix}_{key}_avg"] = nan
            out[f"{prefix}_{key}_max"] = nan
        return out
    for key in keys:
        vals = [s[key] for s in valid]
        out[f"{prefix}_{key}_avg"] = float(np.mean(vals))
        out[f"{prefix}_{key}_max"] = float(np.max(vals))
    return out


def merge_advanced_branch(
    base: dict,
    runs: list[dict],
    prefix: str,
    *,
    top_pct: float = 0.10,
) -> dict:
    """Aggregate A/B/C for one branch (T or W)."""
    ans_stats = [per_generation_answer_stats(r, top_pct=top_pct) for r in runs]
    out = _pool_stats(
        ans_stats,
        prefix,
        [
            "ans_n_tokens",
            "ans_ent_sum",
            "ans_ent_mean",
            "ans_ent_max",
            "ans_ent_top10_mean",
            "ans_logprob_min",
        ],
    )
    out.update({f"{prefix}_{k}": v for k, v in cross_run_token_disagreement(runs).items()})
    out.update({f"{prefix}_{k}": v for k, v in base_perturb_delta(base, runs).items()})
    return out


def merge_advanced_metrics(
    base: dict,
    text_runs: list[dict],
    weight_runs: list[dict],
    *,
    top_pct: float = 0.10,
) -> dict:
    out = base_stable_features(base)
    out.update(merge_advanced_branch(base, text_runs, "T", top_pct=top_pct))
    out.update(merge_advanced_branch(base, weight_runs, "W", top_pct=top_pct))
    return out
