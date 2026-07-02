"""Response-level aggregation of token scores."""

from __future__ import annotations

import numpy as np

from panda.types import TokenFeatures


def aggregate_response_score(
    tokens: list[TokenFeatures],
    mode: str = "mean",
    topk: int = 5,
) -> float:
    if not tokens:
        return 0.0
    scores = np.array([t.score for t in tokens])
    if mode == "topk":
        k = min(topk, len(scores))
        return float(np.sort(scores)[-k:].mean())
    return float(scores.mean())
