"""Semantic answer cache for re-clustering with different thresholds."""

from __future__ import annotations

from panda.core.cluster import cluster_answers
from panda.grading.math_grader import extract_math_answer, math_equal


def build_semantic_cache(
    base_answer: str,
    text_answers: list[str],
    weight_answers: list[str],
    *,
    cluster_method: str = "math_equal_v1",
) -> dict:
    all_raw = [base_answer] + list(text_answers) + list(weight_answers)
    all_norm = [extract_math_answer(a) for a in all_raw]
    labels, sizes = cluster_answers(all_norm)

    # pairwise equivalence (compact for n<=17)
    n = len(all_norm)
    pairwise_eq = [[False] * n for _ in range(n)]
    for i in range(n):
        pairwise_eq[i][i] = True
        for j in range(i + 1, n):
            eq = math_equal(all_norm[i], all_norm[j])
            pairwise_eq[i][j] = eq
            pairwise_eq[j][i] = eq

    return {
        "all_answers_raw": all_raw,
        "all_answers_normalized": all_norm,
        "parse_success": [bool(a.strip()) for a in all_norm],
        "cluster_assignments": {
            "method": cluster_method,
            "labels": labels,
            "cluster_sizes": sizes,
            "num_clusters": len(sizes),
        },
        "pairwise_equivalent": pairwise_eq,
    }
