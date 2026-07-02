"""Token-trace baselines: PE, LL, Self-Certainty, DeepConf, SAR (approx)."""

from __future__ import annotations

import math
import re

import numpy as np

_CONTENT_RE = re.compile(r"[\w\\{}\d]")


def _trace_slice(trace: list[dict], span: dict | None) -> list[dict]:
    if not trace:
        return []
    span = span or {}
    s = int(span.get("start_token", 0))
    e = int(span.get("end_token", len(trace) - 1))
    s = max(0, min(s, len(trace) - 1))
    e = max(s, min(e, len(trace) - 1))
    return trace[s : e + 1]


def predictive_entropy(trace: list[dict], span: dict | None = None) -> dict[str, float]:
    """Mean / sum / max predictive entropy (PE)."""
    toks = _trace_slice(trace, span)
    if not toks:
        nan = float("nan")
        return {"baseline_PE_mean": nan, "baseline_PE_sum": nan, "baseline_PE_max": nan}
    ent = np.array([float(t.get("entropy", 0.0)) for t in toks], dtype=float)
    return {
        "baseline_PE_mean": float(ent.mean()),
        "baseline_PE_sum": float(ent.sum()),
        "baseline_PE_max": float(ent.max()),
    }


def log_likelihood_nll(trace: list[dict], span: dict | None = None) -> dict[str, float]:
    """Log-likelihood baselines: higher NLL => more uncertain => more likely wrong."""
    toks = _trace_slice(trace, span)
    if not toks:
        nan = float("nan")
        return {"baseline_LL_nll": nan, "baseline_LL_mean_logprob": nan}
    lp = np.array([float(t.get("logprob", 0.0)) for t in toks], dtype=float)
    return {
        "baseline_LL_nll": float(-lp.mean()),
        "baseline_LL_mean_logprob": float(lp.mean()),
    }


def _topk_probs(token: dict, k: int) -> np.ndarray:
    pairs = token.get("topk") or []
    if not pairs:
        tid_lp = float(token.get("logprob", 0.0))
        return np.array([math.exp(tid_lp)], dtype=float)
    lps = np.array([float(p[1]) for p in pairs[:k]], dtype=float)
    probs = np.exp(lps - lps.max())
    s = probs.sum()
    if s <= 0:
        return np.ones(1) / 1.0
    return probs / s


def deepconf_mean(trace: list[dict], span: dict | None = None, k: int = 40) -> dict[str, float]:
    """
    DeepConf token confidence (TokUR ``calculate_token_confidence``).

    C_i = -mean(log p_j) over top-k renormed probs at position i.
  Higher C => more confident => invert for wrong-detection.
    """
    toks = _trace_slice(trace, span)
    if not toks:
        nan = float("nan")
        return {"baseline_DC_mean": nan, "baseline_DC_min": nan}
    vals = []
    for t in toks:
        p = _topk_probs(t, k)
        p = np.clip(p, 1e-12, 1.0)
        vals.append(float(-np.log(p).mean()))
    arr = np.array(vals, dtype=float)
    return {"baseline_DC_mean": float(arr.mean()), "baseline_DC_min": float(arr.min())}


def self_certainty_mean(
    trace: list[dict],
    span: dict | None = None,
    *,
    vocab_size: int = 151_936,
) -> dict[str, float]:
    """
    Approximate Self-Certainty from saved top-k (full-vocab SC needs logits).

    Uses TokUR-style normalization: SC ≈ -H_topk / vocab_size per token.
    """
    toks = _trace_slice(trace, span)
    if not toks:
        nan = float("nan")
        return {"baseline_SC_mean": nan, "baseline_SC_min": nan}
    vals = []
    log_v = math.log(max(vocab_size, 2))
    for t in toks:
        p = _topk_probs(t, 20)
        p = np.clip(p, 1e-12, 1.0)
        h = float(-(p * np.log(p)).sum())
        vals.append(-h / vocab_size)
    arr = np.array(vals, dtype=float)
    return {"baseline_SC_mean": float(arr.mean()), "baseline_SC_min": float(arr.min())}


def sar_token_approx(trace: list[dict], span: dict | None = None) -> dict[str, float]:
    """
    Approximate Token-SAR: relevance-weighted entropy.

    Relevance heuristic: math/content tokens weight 1.0, whitespace/punct 0.1.
    Official SAR uses embedding-based semantic shift when removing tokens.
    """
    toks = _trace_slice(trace, span)
    if not toks:
        return {"baseline_SAR": float("nan")}
    weights, ents = [], []
    for t in toks:
        tok = str(t.get("token", ""))
        rel = 1.0 if _CONTENT_RE.search(tok) else 0.1
        weights.append(rel)
        ents.append(float(t.get("entropy", 0.0)))
    w = np.array(weights, dtype=float)
    e = np.array(ents, dtype=float)
    if w.sum() <= 0:
        return {"baseline_SAR": float(e.mean()) if len(e) else float("nan")}
    return {"baseline_SAR": float((w * e).sum() / w.sum())}


def token_baselines_from_generation(gen: dict) -> dict[str, float]:
    """All token-trace baselines for one generation dict."""
    trace = gen.get("token_trace") or []
    span = gen.get("answer_span")
    out: dict[str, float] = {}
    out.update(predictive_entropy(trace, span))
    out.update(log_likelihood_nll(trace, span))
    out.update(deepconf_mean(trace, span))
    out.update(self_certainty_mean(trace, span))
    out.update(sar_token_approx(trace, span))
    return out
