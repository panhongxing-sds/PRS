"""Evaluation: AUROC, AUPRC, risk-coverage, LR fusion, bootstrap CI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from panda.token_qaac.features import FEATURE_GROUPS, PRIMARY_UNIVARIATE


@dataclass
class EvalMetrics:
    auroc: float
    auprc: float
    risk_coverage_auc: float
    n: int
    n_wrong: int
    status: str  # ok | unstable | insufficient | small_test


@dataclass
class LRResult:
    probs: np.ndarray
    coefs: dict[str, float]
    intercept: float
    feature_names: list[str]


def _orient_score(name: str, value: float) -> float:
    sign = PRIMARY_UNIVARIATE.get(name, 1)
    if not np.isfinite(value):
        return float("nan")
    return float(sign * value)


def vectorize(features: dict[str, float], names: Iterable[str]) -> np.ndarray:
    return np.array([_orient_score(n, features.get(n, float("nan"))) for n in names], dtype=float)


def dataset_status(n_wrong: int) -> str:
    if n_wrong < 5:
        return "insufficient"
    if n_wrong < 20:
        return "unstable"
    return "ok"


def subset_status(n_wrong: int, n_total: int) -> str:
    n_correct = n_total - n_wrong
    if n_total < 2 or n_wrong == 0 or n_correct == 0:
        return "insufficient"
    if n_wrong < 3 or n_correct < 3:
        return "small_test"
    if n_wrong < 5 or n_correct < 5:
        return "unstable"
    return "ok"


def risk_coverage_curve(labels_wrong: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    mask = np.isfinite(scores)
    y = labels_wrong[mask].astype(int)
    s = scores[mask]
    if len(y) < 2:
        return []
    order = np.argsort(-s)
    n = len(y)
    points: list[dict[str, float]] = []
    for k in range(n + 1):
        kept = order[k:]
        if not len(kept):
            break
        coverage = len(kept) / n
        risk = float(y[kept].mean())
        acc = 1.0 - risk
        points.append({"coverage": coverage, "accuracy": acc, "risk": risk, "n_kept": float(len(kept))})
    return sorted(points, key=lambda p: p["coverage"], reverse=True)


def risk_coverage_auc(labels_wrong: np.ndarray, scores: np.ndarray) -> float:
    pts = risk_coverage_curve(labels_wrong, scores)
    if len(pts) < 2:
        return float("nan")
    xs = [p["coverage"] for p in pts]
    ys = [p["accuracy"] for p in pts]
    return float(np.trapz(ys, xs))


def eval_binary(
    labels_wrong: list[int],
    scores: list[float],
    *,
    use_dataset_threshold: bool = False,
) -> EvalMetrics:
    y = np.array(labels_wrong, dtype=int)
    s = np.array(scores, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_wrong = int(y.sum())
    if use_dataset_threshold:
        status = dataset_status(n_wrong)
    else:
        status = subset_status(n_wrong, len(y))
    if status == "insufficient" or len(y) < 2 or len(np.unique(y)) < 2:
        return EvalMetrics(float("nan"), float("nan"), float("nan"), int(len(y)), n_wrong, status)
    return EvalMetrics(
        auroc=float(roc_auc_score(y, s)),
        auprc=float(average_precision_score(y, s)),
        risk_coverage_auc=risk_coverage_auc(y, s),
        n=int(len(y)),
        n_wrong=n_wrong,
        status=status,
    )


def bootstrap_auroc_ci(
    labels_wrong: np.ndarray,
    scores: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    y = labels_wrong.astype(int)
    s = scores.astype(float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot_valid": 0}

    rng = np.random.RandomState(seed)
    n = len(y)
    aurocs: list[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aurocs.append(float(roc_auc_score(y[idx], s[idx])))
    if not aurocs:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_boot_valid": 0}
    alpha = (1 - ci) / 2
    return {
        "auroc": float(np.mean(aurocs)),
        "ci_low": float(np.percentile(aurocs, 100 * alpha)),
        "ci_high": float(np.percentile(aurocs, 100 * (1 - alpha))),
        "n_boot_valid": len(aurocs),
    }


def split_indices(n: int, train_frac: float, dev_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = max(1, int(n * train_frac))
    n_dev = max(1, int(n * dev_frac))
    train = idx[:n_train]
    dev = idx[n_train : n_train + n_dev]
    test = idx[n_train + n_dev :]
    if len(test) == 0:
        test = dev
    return train, dev, test


def split_overlap(a: np.ndarray, b: np.ndarray) -> int:
    return int(len(set(a.tolist()) & set(b.tolist())))


def train_lr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
) -> LRResult:
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_te = scaler.transform(x_test)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_tr, y_train)
    coefs = {name: float(c) for name, c in zip(feature_names, clf.coef_[0])}
    return LRResult(
        probs=clf.predict_proba(x_te)[:, 1],
        coefs=coefs,
        intercept=float(clf.intercept_[0]),
        feature_names=feature_names,
    )


def average_lr_coefs(coef_list: list[dict[str, float]]) -> dict[str, float]:
    if not coef_list:
        return {}
    keys = coef_list[0].keys()
    return {k: float(np.mean([c[k] for c in coef_list])) for k in keys}


def eval_univariate(rows: list[dict], labels: np.ndarray, *, n_boot: int = 2000) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for feat, direction in PRIMARY_UNIVARIATE.items():
        vals = np.array([direction * r["features"].get(feat, float("nan")) for r in rows])
        m = eval_binary(labels.tolist(), vals.tolist(), use_dataset_threshold=True)
        boot = bootstrap_auroc_ci(labels, vals, n_boot=n_boot)
        auroc_oriented = m.auroc
        auroc_best = max(auroc_oriented, 1 - auroc_oriented) if np.isfinite(auroc_oriented) else float("nan")
        display = f"-{feat}" if direction == -1 else feat
        out[display] = {
            "feature": feat,
            "direction": direction,
            "auroc_oriented": auroc_oriented,
            "auroc": auroc_best,
            "auprc": m.auprc,
            "rc_auc": m.risk_coverage_auc,
            "status": m.status,
            "bootstrap_ci_low": boot["ci_low"],
            "bootstrap_ci_high": boot["ci_high"],
            "rc_curve": risk_coverage_curve(labels, vals),
        }
    return out


def eval_feature_groups(
    rows: list[dict],
    *,
    seeds: list[int] | None = None,
    train_frac: float = 0.6,
    dev_frac: float = 0.2,
    n_boot: int = 2000,
) -> dict[str, dict]:
    seeds = seeds or [42, 43, 44]
    labels = np.array([int(r["label_wrong"]) for r in rows], dtype=int)
    n_wrong = int(labels.sum())
    ds_status = dataset_status(n_wrong)

    uni = eval_univariate(rows, labels, n_boot=n_boot)
    best_feat = max(uni.items(), key=lambda x: x[1].get("auroc") or -1)
    best_m = best_feat[1]
    # Re-bootstrap best feature with direction that maximizes AUROC
    best_raw = np.array([best_m["direction"] * r["features"].get(best_m["feature"], float("nan")) for r in rows])
    if best_m.get("auroc_oriented", 0) is not None and best_m.get("auroc_oriented", 0.5) < 0.5:
        best_raw = -best_raw
    best_boot = bootstrap_auroc_ci(labels, best_raw, n_boot=n_boot)

    results: dict[str, dict] = {
        "_meta": {
            "n": len(rows),
            "n_wrong": n_wrong,
            "n_correct": int(len(rows) - n_wrong),
            "status": ds_status,
        },
        "_univariate": uni,
        "_univariate_best": {
            "feature": best_feat[0],
            "auroc": best_m.get("auroc"),
            "auprc": best_m.get("auprc"),
            "bootstrap_ci_low": best_boot["ci_low"],
            "bootstrap_ci_high": best_boot["ci_high"],
        },
        "_splits": {"seeds": {}, "train_frac": train_frac, "dev_frac": dev_frac},
    }

    if ds_status == "insufficient":
        return results

    test_by_seed: dict[int, np.ndarray] = {}
    for seed in seeds:
        tr, dev, te = split_indices(len(rows), train_frac, dev_frac, seed)
        test_by_seed[seed] = te
        results["_splits"]["seeds"][str(seed)] = {
            "train_n": int(len(tr)),
            "dev_n": int(len(dev)),
            "test_n": int(len(te)),
            "test_wrong": int(labels[te].sum()),
            "test_correct": int(len(te) - labels[te].sum()),
            "test_indices": te.tolist(),
        }

    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1 :]:
            key = f"overlap_test_{s1}_{s2}"
            results["_splits"][key] = split_overlap(test_by_seed[s1], test_by_seed[s2])

    for group, names in FEATURE_GROUPS.items():
        x = np.vstack([vectorize(r["features"], names) for r in rows])
        seed_metrics: list[EvalMetrics] = []
        seed_aurocs: list[float] = []
        seed_details: list[dict] = []
        coef_accum: list[dict[str, float]] = []
        boot_aurocs: list[float] = []

        for seed in seeds:
            tr, _, te = split_indices(len(rows), train_frac, dev_frac, seed)
            y_tr, y_te = labels[tr], labels[te]
            if len(np.unique(y_tr)) < 2:
                continue
            lr = train_lr(x[tr], y_tr, x[te], names)
            coef_accum.append(lr.coefs)
            m = eval_binary(y_te.tolist(), lr.probs.tolist(), use_dataset_threshold=False)
            seed_metrics.append(m)
            seed_details.append(
                {
                    "seed": seed,
                    "auroc": m.auroc,
                    "auprc": m.auprc,
                    "rc_auc": m.risk_coverage_auc,
                    "test_n": m.n,
                    "test_wrong": m.n_wrong,
                    "status": m.status,
                }
            )
            if np.isfinite(m.auroc):
                seed_aurocs.append(m.auroc)

        # bootstrap LR: OOB evaluation
        rng = np.random.RandomState(0)
        n = len(rows)
        all_idx = np.arange(n)
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            inbag = set(idx.tolist())
            oob = np.array([i for i in all_idx if i not in inbag], dtype=int)
            if len(oob) < 5 or len(np.unique(labels[idx])) < 2 or len(np.unique(labels[oob])) < 2:
                continue
            lr = train_lr(x[idx], labels[idx], x[oob], names)
            m = eval_binary(labels[oob].tolist(), lr.probs.tolist(), use_dataset_threshold=False)
            if np.isfinite(m.auroc):
                boot_aurocs.append(m.auroc)

        avg_coefs = average_lr_coefs(coef_accum)
        intercepts = []
        full_lr = None
        if len(np.unique(labels)) >= 2:
            full_lr = train_lr(x, labels, x, names)
            intercepts = [full_lr.intercept]

        rc_curve: list[dict] = []
        rc_auc_full = float("nan")
        if full_lr is not None:
            rc_curve = risk_coverage_curve(labels, full_lr.probs)
            rc_auc_full = risk_coverage_auc(labels, full_lr.probs)

        results[group] = {
            "auroc_mean": float(np.nanmean([m.auroc for m in seed_metrics])) if seed_metrics else float("nan"),
            "auprc_mean": float(np.nanmean([m.auprc for m in seed_metrics])) if seed_metrics else float("nan"),
            "rc_auc_mean": float(np.nanmean([m.risk_coverage_auc for m in seed_metrics])) if seed_metrics else float("nan"),
            "rc_auc_full": rc_auc_full,
            "auroc_std": float(np.std(seed_aurocs)) if len(seed_aurocs) > 1 else 0.0,
            "n_seeds_valid": len(seed_aurocs),
            "n_seeds_total": len(seed_details),
            "status": ds_status,
            "per_seed": seed_details,
            "lr_coefs_mean": avg_coefs,
            "lr_coefs_full": full_lr.coefs if full_lr else {},
            "lr_intercept_full": full_lr.intercept if full_lr else float("nan"),
            "bootstrap_auroc_ci_low": float(np.percentile(boot_aurocs, 2.5)) if boot_aurocs else float("nan"),
            "bootstrap_auroc_ci_high": float(np.percentile(boot_aurocs, 97.5)) if boot_aurocs else float("nan"),
            "bootstrap_n_valid": len(boot_aurocs),
            "rc_curve": rc_curve,
        }

    return results
