"""TokUR-style metrics: AUROC, AUPRC, ACC* (Top-50% accuracy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class TokURMetrics:
    auroc: float
    auprc: float
    acc_star: float  # Top-50% accuracy (TokUR Table ACC*)
    aurc: float  # Area under risk-coverage curve (lower = better abstention)
    n: int


def top_p_accuracy(
    labels: list[bool],
    scores: list[float],
    p: float = 0.5,
    *,
    higher_is_uncertain: bool = True,
) -> float:
    """
    TokUR Top-p% ACC* (``eval_detect_multi_seed.get_top_p_acc``).

    Official TokUR stores **confidence** scores (``-EU``, ``-TU``, …) and sorts
    descending to keep the most confident top-p fraction. When ``higher_is_uncertain``
    is True (default), ``scores`` are raw uncertainty (higher = more uncertain) and we
    take the lowest-uncertainty top-p% — equivalent to sorting negated scores descending.

    ``labels[i]`` is answer correctness (True = correct).
    """
    df = pd.DataFrame({"label": labels, "score": scores})
    ascending = higher_is_uncertain
    df = df.sort_values("score", ascending=ascending)
    k = max(1, int(len(df) * p))
    return float(df.iloc[:k]["label"].mean())


def detection_aurc(labels_correct: list[bool], scores: list[float]) -> float:
    """
    Selective-abstention AURC: abstain highest scores first, integrate error rate vs coverage.
    Lower is better (less risk at each coverage level).
    """
    y_wrong = np.array([not c for c in labels_correct], dtype=int)
    s = np.array(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y_wrong[mask], s[mask]
    if len(y) < 2:
        return float("nan")
    order = np.argsort(-s)
    n = len(y)
    coverages: list[float] = []
    risks: list[float] = []
    for k in range(n + 1):
        kept = order[k:]
        if not len(kept):
            break
        coverages.append(len(kept) / n)
        risks.append(float(y[kept].mean()))
    if len(coverages) < 2:
        return float("nan")
    idx = np.argsort(coverages)
    c = np.array(coverages)[idx]
    r = np.array(risks)[idx]
    return float(np.trapz(r, c))


def compute_detection_metrics(
    labels_correct: list[bool],
    scores: list[float],
) -> TokURMetrics:
    """
    Hallucination detection: y=1 if incorrect (not correct).
    Higher score => more likely incorrect.
    """
    y = np.array([not c for c in labels_correct], dtype=int)
    s = np.array(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return TokURMetrics(float("nan"), float("nan"), float("nan"), float("nan"), len(y))
    labels_kept = [bool(c) for c, m in zip(labels_correct, mask) if m]
    return TokURMetrics(
        auroc=float(roc_auc_score(y, s)),
        auprc=float(average_precision_score(y, s)),
        acc_star=top_p_accuracy(labels_kept, s.tolist(), p=0.5),
        aurc=detection_aurc(labels_kept, s.tolist()),
        n=len(y),
    )


def bootstrap_auroc(
    labels_correct: list[bool],
    scores: list[float],
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float]:
    """Bootstrap CI for AUROC (test-set robustness check)."""
    y = np.array([not c for c in labels_correct], dtype=int)
    s = np.array(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n = len(y)
    if n < 10 or len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    point = float(roc_auc_score(y, s))
    rng = np.random.RandomState(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(float(roc_auc_score(y[idx], s[idx])))
    if not boots:
        return {"auroc": point, "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    alpha = (1.0 - ci) / 2.0
    return {
        "auroc": point,
        "ci_low": float(np.quantile(boots, alpha)),
        "ci_high": float(np.quantile(boots, 1.0 - alpha)),
        "n": n,
        "n_boot_effective": len(boots),
    }
