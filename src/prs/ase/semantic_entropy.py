"""Semantic-level ASE scores from answer clusters."""

from __future__ import annotations

import math

from prs.ase.cluster import cluster_answers


def compute_ase(answers: list[str]) -> dict[str, float | int | list[float]]:
    """
    Compute Adversarial Semantic Entropy metrics.

    Primary score: U = 1 - max cluster mass.
    """
    n = len(answers)
    if n == 0:
        return {
            "U": float("nan"),
            "H": float("nan"),
            "H_norm": float("nan"),
            "num_clusters": 0,
            "max_mass": float("nan"),
            "cluster_masses": [],
        }

    _, sizes = cluster_answers(answers)
    masses = sorted((c / n for c in sizes.values()), reverse=True)
    max_mass = masses[0]
    h = 0.0
    for p in masses:
        if p > 0:
            h -= p * math.log(p)
    k = len(masses)
    h_norm = h / math.log(k) if k > 1 else 0.0

    return {
        "U": 1.0 - max_mass,
        "H": h,
        "H_norm": h_norm,
        "num_clusters": k,
        "max_mass": max_mass,
        "cluster_masses": masses,
    }
