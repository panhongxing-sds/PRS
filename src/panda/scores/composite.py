"""Composite token score S_t from z-scored features."""

from __future__ import annotations

import numpy as np

from panda.types import TokenFeatures

# First-version ATokUR: EU^W + AD^E + AD^H + MC^H (see README)
FEATURE_KEYS = ("eu_weight", "ad_embedding", "ad_hidden", "mc_hidden")
OPTIONAL_KEYS = ("mc_embedding", "nll")


def zscore(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros_like(x)
    std = x.std()
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def compute_token_scores(
    tokens: list[TokenFeatures],
    zscore_within_response: bool = True,
    weights: dict[str, float] | None = None,
) -> list[TokenFeatures]:
    """
    S_t = sum_i w_i * z(feature_i), default equal weights on EU^W, AD^E, AD^H, MC^H (+ optional MC^E, NLL).
    """
    w = weights or {k: 1.0 for k in FEATURE_KEYS}
    n = len(tokens)
    if n == 0:
        return tokens

    arrays = {k: np.array([getattr(t, k) for t in tokens], dtype=float) for k in FEATURE_KEYS}
    if zscore_within_response:
        arrays = {k: zscore(v) for k, v in arrays.items()}

    for i, t in enumerate(tokens):
        t.score = float(sum(w.get(k, 0.0) * arrays[k][i] for k in FEATURE_KEYS))
    return tokens
