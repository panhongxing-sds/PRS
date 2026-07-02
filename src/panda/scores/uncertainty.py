"""TokUR-style total / aleatoric / epistemic uncertainty from weight-perturbed distributions."""

from __future__ import annotations

import numpy as np


def entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def epistemic_uncertainty(distributions: list[np.ndarray]) -> float:
    """EU = H(mean p) - mean H(p)."""
    if len(distributions) < 2:
        return 0.0
    stack = np.stack(distributions, axis=0)
    mean_p = stack.mean(axis=0)
    tu = entropy(mean_p)
    au = float(np.mean([entropy(p) for p in stack]))
    return max(0.0, tu - au)


def epistemic_per_position(
    distributions_per_pos: list[list[np.ndarray]],
) -> list[float]:
    """One EU value per response token position."""
    return [epistemic_uncertainty(dists) for dists in distributions_per_pos]
