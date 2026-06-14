"""Sample-based baselines: SE, U_Ecc, U_Deg from multiple generations."""

from __future__ import annotations

import math
import os

import numpy as np

from prs.ase.cluster import cluster_answers
from prs.grading.math_grader import extract_math_answer, math_equal


def _normalize_answer(text: str) -> str:
    return extract_math_answer(text).strip()


def _se_cluster_mode() -> str:
    """math_equal (generation) or nli (official post-hoc via --recompute-se)."""
    return os.environ.get("PRS_SE_CLUSTER", "math_equal").strip().lower()


def _cluster_for_se(
    sequences: list[str], *, cluster_mode: str | None = None
) -> tuple[dict[int, int], str]:
    mode = (cluster_mode or _se_cluster_mode()).strip().lower()
    if mode == "math_equal":
        _, sizes = cluster_answers(sequences)
        return sizes, "math_equal"
    try:
        from prs.baselines.nli_entailment import cluster_sequences_nli

        _, sizes = cluster_sequences_nli(sequences)
        return sizes, "nli"
    except Exception:
        # NLI model missing/offline → fall back so metrics pipeline does not crash.
        _, sizes = cluster_answers(sequences)
        return sizes, "math_equal_nli_fallback"


def semantic_entropy_h(
    sequences: list[str], *, cluster_mode: str | None = None
) -> dict[str, float | str | int]:
    """
    Semantic entropy H over clusters (Kuhn et al.).

    Generation default: math_equal on extracted answers (fast).
    Official table SE: pass cluster_mode=\"nli\" or PRS_SE_CLUSTER=nli (post-hoc).
    """
    mode_label = (cluster_mode or _se_cluster_mode()).strip().lower()
    n = len(sequences)
    if n == 0:
        nan = float("nan")
        return {
            "baseline_SE_H": nan,
            "baseline_SE_H_norm": nan,
            "baseline_SE_num_clusters": 0,
            "baseline_SE_cluster_mode": mode_label,
        }
    sizes, mode = _cluster_for_se(sequences, cluster_mode=cluster_mode)
    masses = [c / n for c in sizes.values()]
    h = 0.0
    for p in masses:
        if p > 0:
            h -= p * math.log(p)
    k = len(masses)
    h_norm = h / math.log(k) if k > 1 else 0.0
    return {
        "baseline_SE_H": h,
        "baseline_SE_H_norm": h_norm,
        "baseline_SE_num_clusters": k,
        "baseline_SE_cluster_mode": mode,
    }


def build_similarity_matrix(answers: list[str]) -> np.ndarray:
    """Pairwise similarity W_ij in [0,1] via math_equal on normalized answers."""
    n = len(answers)
    normed = [_normalize_answer(a) for a in answers]
    w = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            sim = 0.0
            if normed[i] and normed[j] and math_equal(normed[i], normed[j]):
                sim = 1.0
            w[i, j] = w[j, i] = sim
    return w


def u_deg(W: np.ndarray) -> float:
    """UDeg(x) = trace(mI - D) / m^2  (Lin et al., UQ-NLG)."""
    m = W.shape[0]
    if m == 0:
        return float("nan")
    d = np.diag(W.sum(axis=1))
    return float(np.trace(m * np.eye(m) - d) / (m**2))


def u_ecc(W: np.ndarray, k: int | None = None) -> float:
    """UEcc(x) = ||V'||_F^2 from smallest k Laplacian eigenvectors."""
    m = W.shape[0]
    if m == 0:
        return float("nan")
    if m == 1:
        return 0.0
    deg = W.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(m) - d_inv_sqrt @ W @ d_inv_sqrt
    _, eigvecs = np.linalg.eigh(L)
    kk = k if k is not None else max(1, m - 1)
    kk = min(kk, m)
    U = eigvecs[:, :kk]
    Vp = U - U.mean(axis=1, keepdims=True)
    return float(np.linalg.norm(Vp, ord="fro") ** 2)


def graph_uncertainty(answers: list[str]) -> dict[str, float]:
    """U_Ecc and U_Deg from sample similarity graph."""
    if not answers:
        nan = float("nan")
        return {"baseline_U_Ecc": nan, "baseline_U_Deg": nan}
    w = build_similarity_matrix(answers)
    return {"baseline_U_Ecc": u_ecc(w), "baseline_U_Deg": u_deg(w)}


def sample_baselines_from_answers(answers: list[str]) -> dict[str, float]:
    """Legacy helper: SE + graph baselines from one answer pool (tests only)."""
    out: dict[str, float] = {}
    out.update(semantic_entropy_h(answers))
    out.update(graph_uncertainty(answers))
    return out
