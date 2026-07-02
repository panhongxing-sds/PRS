"""Candidate answer distribution statistics over rephrases."""

from __future__ import annotations

import itertools

import numpy as np


def softmax_scores(scores: np.ndarray, tau: float = 1.0) -> np.ndarray:
    s = scores.astype(float) / max(tau, 1e-6)
    s = s - np.max(s)
    exp_s = np.exp(s)
    z = exp_s.sum()
    if z <= 0:
        return np.ones_like(s) / len(s)
    return exp_s / z


def entropy(probs: np.ndarray) -> float:
    p = np.clip(probs.astype(float), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-12, 1.0)
    q = np.clip(q.astype(float), 1e-12, 1.0)
    m = 0.5 * (p + q)
    kl_pm = float((p * np.log(p / m)).sum())
    kl_qm = float((q * np.log(q / m)).sum())
    return 0.5 * (kl_pm + kl_qm)


def pairwise_js(distributions: list[np.ndarray]) -> list[float]:
    out: list[float] = []
    for i, j in itertools.combinations(range(len(distributions)), 2):
        out.append(js_divergence(distributions[i], distributions[j]))
    return out
