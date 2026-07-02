"""Complementarity analysis: range_norm vs weight attack vs TokUR."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from panda.token_qaac.eval import bootstrap_auroc_ci, risk_coverage_auc, subset_status
from panda.token_qaac.range_only import attach_range_features, compute_range_features

METHODS = [
    "TokUR",
    "RangeOnly",
    "WeightOnly",
    "Range+Weight",
    "TokUR+Range+Weight",
]


def split_train_dev_test(n: int, seed: int, train_frac: float = 0.6, dev_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = max(1, int(n * train_frac))
    n_dev = max(1, int(n * dev_frac))
    train = idx[:n_train]
    dev = idx[n_train : n_train + n_dev]
    test = idx[n_train + n_dev :]
    if len(test) == 0:
        test = dev.copy()
    return train, dev, test


def pick_sign(raw: np.ndarray, labels: np.ndarray, dev_idx: np.ndarray) -> int:
    """Pick +1 or -1 on dev to maximize AUROC (no test leakage)."""
    y = labels[dev_idx]
    x = raw[dev_idx]
    mask = np.isfinite(x)
    y, x = y[mask], x[mask]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return 1
    a_pos = roc_auc_score(y, x)
    a_neg = roc_auc_score(y, -x)
    return 1 if a_pos >= a_neg else -1


def recall_at_top_frac(labels: np.ndarray, scores: np.ndarray, frac: float = 0.2) -> float:
    mask = np.isfinite(scores)
    y, s = labels[mask].astype(int), scores[mask]
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(y) * frac)))
    top = np.argsort(-s)[:k]
    return float(y[top].sum() / y.sum())


def _eval_scores(y: np.ndarray, s: np.ndarray) -> dict:
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    st = subset_status(int(y.sum()), len(y))
    if st == "insufficient" or len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "rc_auc": float("nan"), "recall_top20": float("nan"), "status": st}
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "rc_auc": risk_coverage_auc(y, s),
        "recall_top20": recall_at_top_frac(y, s, 0.2),
        "status": st,
    }


def _train_lr_probs(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    mask = np.all(np.isfinite(x_tr), axis=1)
    x_tr, y_tr = x_tr[mask], y_tr[mask]
    if len(y_tr) < 2 or len(np.unique(y_tr)) < 2:
        return np.full(len(x_te), float("nan"))
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(x_tr_s, y_tr)
    mask_te = np.all(np.isfinite(x_te), axis=1)
    x_te_s = scaler.transform(np.nan_to_num(x_te, nan=0.0))
    if not mask_te.any():
        return np.full(len(x_te), float("nan"))
    probs = clf.predict_proba(x_te_s)[:, 1]
    out = np.full(len(x_te), float("nan"))
    out[mask_te] = probs[mask_te]
    return out


def build_rows(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        rf = compute_range_features(rec)
        out.append(
            {
                "id": rec["id"],
                "label_wrong": int(rec["label_wrong"]),
                "range_norm_raw": rf["range_norm"],
                "a0_num_tokens": rf["a0_num_tokens"],
                "clean_logprob_norm": rf["clean_logprob_norm"],
                "weight_raw": float(rec.get("weight_ad_mean", float("nan"))),
                "tokur_raw": float(rec.get("tokur_eu", float("nan"))),
            }
        )
    return out


def _finite_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if all(np.isfinite(r[k]) for k in ("range_norm_raw", "weight_raw", "tokur_raw")):
            out.append(r)
    return out


def analyze_complementarity(records: list[dict], *, seeds: list[int] | None = None, n_boot: int = 2000) -> dict:
    seeds = seeds or [42, 43, 44]
    rows_all = build_rows(records)
    rows = _finite_rows(rows_all)
    labels = np.array([r["label_wrong"] for r in rows], dtype=int)
    n_wrong = int(labels.sum())
    n = len(rows)

    range_raw = np.array([r["range_norm_raw"] for r in rows])
    weight_raw = np.array([r["weight_raw"] for r in rows])
    tokur_raw = np.array([r["tokur_raw"] for r in rows])

    # Spearman on full data (exploratory)
    pairs = [
        ("range_norm vs weight", range_raw, weight_raw),
        ("range_norm vs TokUR", range_raw, tokur_raw),
        ("weight vs TokUR", weight_raw, tokur_raw),
    ]
    correlations: dict[str, float] = {}
    for name, a, b in pairs:
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            correlations[name] = float("nan")
        else:
            correlations[name], _ = spearmanr(a[mask], b[mask])

    seed_results: dict[str, list[dict]] = {m: [] for m in METHODS}
    signs_by_seed: dict[str, dict] = {}

    for seed in seeds:
        tr, dev, te = split_train_dev_test(n, seed)
        signs = {
            "range": pick_sign(range_raw, labels, dev),
            "weight": pick_sign(weight_raw, labels, dev),
            "tokur": pick_sign(tokur_raw, labels, dev),
        }
        signs_by_seed[str(seed)] = signs

        s_range = signs["range"] * range_raw
        s_weight = signs["weight"] * weight_raw
        s_tokur = signs["tokur"] * tokur_raw

        y_te = labels[te]
        for method, scores in [
            ("TokUR", s_tokur[te]),
            ("RangeOnly", s_range[te]),
            ("WeightOnly", s_weight[te]),
        ]:
            seed_results[method].append(_eval_scores(y_te, scores))

        # LR on train, eval test
        x_rw = np.column_stack([s_range, s_weight])
        x_trw = np.column_stack([s_tokur, s_range, s_weight])
        probs_rw = _train_lr_probs(x_rw[tr], labels[tr], x_rw[te])
        probs_trw = _train_lr_probs(x_trw[tr], labels[tr], x_trw[te])
        seed_results["Range+Weight"].append(_eval_scores(y_te, probs_rw))
        seed_results["TokUR+Range+Weight"].append(_eval_scores(y_te, probs_trw))

    # Aggregate across seeds
    methods_out: dict[str, dict] = {}
    for method in METHODS:
        vals = seed_results[method]
        aurocs = [v["auroc"] for v in vals if np.isfinite(v["auroc"])]
        methods_out[method] = {
            "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
            "auroc_std": float(np.std(aurocs)) if len(aurocs) > 1 else 0.0,
            "auprc_mean": float(np.nanmean([v["auprc"] for v in vals])),
            "rc_auc_mean": float(np.nanmean([v["rc_auc"] for v in vals])),
            "recall_top20_mean": float(np.nanmean([v["recall_top20"] for v in vals])),
            "per_seed": vals,
            "n_seeds_valid": len(aurocs),
        }

    # Top-20% overlap on full data (exploratory, fixed sign from seed 42 dev)
    s42 = signs_by_seed.get("42", {"range": 1, "weight": 1, "tokur": 1})
    sr = s42["range"] * range_raw
    sw = s42["weight"] * weight_raw
    st = s42["tokur"] * tokur_raw
    k = max(1, int(np.ceil(n * 0.2)))
    wrong_idx = set(np.where(labels == 1)[0].tolist())

    def top_set(scores: np.ndarray) -> set[int]:
        mask = np.isfinite(scores)
        idx = np.where(mask)[0]
        order = idx[np.argsort(-scores[idx])[:k]]
        return set(order.tolist())

    top_r, top_w, top_t = top_set(sr), top_set(sw), top_set(st)
    caught_r = len(top_r & wrong_idx)
    caught_w = len(top_w & wrong_idx)
    caught_t = len(top_t & wrong_idx)

    overlap = {
        "top20_wrong_total": n_wrong,
        "top20_k": k,
        "range_caught_wrong": caught_r,
        "weight_caught_wrong": caught_w,
        "tokur_caught_wrong": caught_t,
        "range_recall_top20": caught_r / n_wrong if n_wrong else float("nan"),
        "weight_recall_top20": caught_w / n_wrong if n_wrong else float("nan"),
        "tokur_recall_top20": caught_t / n_wrong if n_wrong else float("nan"),
        "range_weight_wrong_overlap": len(top_r & top_w & wrong_idx),
        "range_tokur_wrong_overlap": len(top_r & top_t & wrong_idx),
        "weight_tokur_wrong_overlap": len(top_w & top_t & wrong_idx),
        "range_only_not_tokur": len((top_r & wrong_idx) - top_t),
        "weight_only_not_tokur": len((top_w & wrong_idx) - top_t),
        "spearman": correlations,
        "signs_seed42_dev": s42,
    }

    # Bootstrap CI for RangeOnly on full data with dev sign (seed 42)
    boot_range = bootstrap_auroc_ci(labels, s42["range"] * range_raw, n_boot=n_boot)

    return {
        "_meta": {"n": n, "n_all": len(rows_all), "n_valid": n, "n_wrong": n_wrong, "mode": "complementarity"},
        "_correlations": correlations,
        "_overlap": overlap,
        "_signs_by_seed": signs_by_seed,
        "_methods": methods_out,
        "_range_bootstrap": boot_range,
    }
