"""Evaluation for range-only UQ ablation."""

from __future__ import annotations

import numpy as np

from panda.token_qaac.eval import (
    bootstrap_auroc_ci,
    dataset_status,
    eval_binary,
    split_indices,
    split_overlap,
    train_lr,
)
from panda.token_qaac.range_only import (
    RANGE_ONLY_DISPLAY,
    RANGE_ONLY_FEATURES,
    RANGE_ONLY_LR_FEATURES,
    RANGE_ONLY_LR_GROUP,
    attach_range_features,
)


def _orient(name: str, value: float) -> float:
    sign = RANGE_ONLY_FEATURES.get(name, 1)
    if not np.isfinite(value):
        return float("nan")
    return float(sign * value)


def eval_range_only(
    rows: list[dict],
    *,
    seeds: list[int] | None = None,
    n_boot: int = 2000,
) -> dict:
    seeds = seeds or [42, 43, 44]
    enriched = [attach_range_features(r) for r in rows]
    labels = np.array([int(r["label_wrong"]) for r in enriched], dtype=int)
    n_wrong = int(labels.sum())
    ds_status = dataset_status(n_wrong)

    uni: dict[str, dict] = {}
    for feat, direction in RANGE_ONLY_FEATURES.items():
        vals = np.array([direction * r["range_features"][feat] for r in enriched])
        m = eval_binary(labels.tolist(), vals.tolist(), use_dataset_threshold=True)
        boot = bootstrap_auroc_ci(labels, vals, n_boot=n_boot)
        auroc_o = m.auroc
        auroc_best = max(auroc_o, 1 - auroc_o) if np.isfinite(auroc_o) else float("nan")
        boot_vals = vals if (np.isfinite(auroc_o) and auroc_o >= 0.5) else -vals
        best_boot = bootstrap_auroc_ci(labels, boot_vals, n_boot=n_boot)
        display = RANGE_ONLY_DISPLAY[feat]
        uni[display] = {
            "feature": feat,
            "direction": direction,
            "auroc": auroc_best,
            "auroc_oriented": auroc_o,
            "auprc": m.auprc,
            "status": m.status,
            "bootstrap_ci_low": best_boot["ci_low"],
            "bootstrap_ci_high": best_boot["ci_high"],
            "random_auprc_baseline": n_wrong / len(labels) if len(labels) else float("nan"),
        }

    best = max(uni.items(), key=lambda x: x[1].get("auroc") or -1)

    results: dict = {
        "_meta": {
            "n": len(enriched),
            "n_wrong": n_wrong,
            "n_correct": len(enriched) - n_wrong,
            "status": ds_status,
            "mode": "range-only",
        },
        "_univariate": uni,
        "_univariate_best": {
            "feature": best[0],
            "auroc": best[1].get("auroc"),
            "auprc": best[1].get("auprc"),
            "bootstrap_ci_low": best[1].get("bootstrap_ci_low"),
            "bootstrap_ci_high": best[1].get("bootstrap_ci_high"),
        },
        "_splits": {"seeds": {}},
    }

    if ds_status == "insufficient":
        return results

    test_by_seed: dict[int, np.ndarray] = {}
    for seed in seeds:
        tr, dev, te = split_indices(len(enriched), 0.6, 0.2, seed)
        test_by_seed[seed] = te
        results["_splits"]["seeds"][str(seed)] = {
            "test_n": int(len(te)),
            "test_wrong": int(labels[te].sum()),
            "test_indices": te.tolist(),
        }
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1 :]:
            results["_splits"][f"overlap_test_{s1}_{s2}"] = split_overlap(
                test_by_seed[s1], test_by_seed[s2]
            )

    # Single LR: range_norm + a0_num_tokens
    names = RANGE_ONLY_LR_FEATURES
    x = np.vstack([[_orient(n, r["range_features"][n]) for n in names] for r in enriched])
    seed_aurocs: list[float] = []
    per_seed: list[dict] = []
    for seed in seeds:
        tr, _, te = split_indices(len(enriched), 0.6, 0.2, seed)
        if len(np.unique(labels[tr])) < 2:
            continue
        lr = train_lr(x[tr], labels[tr], x[te], names)
        m = eval_binary(labels[te].tolist(), lr.probs.tolist(), use_dataset_threshold=False)
        per_seed.append({"seed": seed, "auroc": m.auroc, "auprc": m.auprc, "status": m.status})
        if np.isfinite(m.auroc):
            seed_aurocs.append(m.auroc)

    boot_aurocs: list[float] = []
    rng = np.random.RandomState(0)
    n = len(enriched)
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

    full_lr = train_lr(x, labels, x, names) if len(np.unique(labels)) >= 2 else None
    lr_auroc_o = float("nan")
    lr_auprc = float("nan")
    if full_lr is not None:
        fm = eval_binary(labels.tolist(), full_lr.probs.tolist(), use_dataset_threshold=True)
        lr_auroc_o = fm.auroc
        lr_auprc = fm.auprc

    results[RANGE_ONLY_LR_GROUP] = {
        "features": names,
        "auroc_mean": float(np.mean(seed_aurocs)) if seed_aurocs else float("nan"),
        "auroc_std": float(np.std(seed_aurocs)) if len(seed_aurocs) > 1 else 0.0,
        "auroc_max": max(lr_auroc_o, 1 - lr_auroc_o) if np.isfinite(lr_auroc_o) else float("nan"),
        "auprc_mean": float(np.mean([p["auprc"] for p in per_seed if np.isfinite(p["auprc"])]))
        if per_seed
        else float("nan"),
        "per_seed": per_seed,
        "lr_coefs": full_lr.coefs if full_lr else {},
        "lr_intercept": full_lr.intercept if full_lr else float("nan"),
        "bootstrap_ci_low": float(np.percentile(boot_aurocs, 2.5)) if boot_aurocs else float("nan"),
        "bootstrap_ci_high": float(np.percentile(boot_aurocs, 97.5)) if boot_aurocs else float("nan"),
    }

    results["_verdict"] = _verdict(uni, results.get(RANGE_ONLY_LR_GROUP, {}))
    return results


def _verdict(uni: dict[str, dict], lr: dict) -> dict[str, str]:
    def _g(name: str) -> float:
        return uni.get(name, {}).get("auroc") or float("nan")

    clean = _g("-clean_logprob_norm")
    length = _g("a0_num_tokens")
    raw = _g("range_raw")
    norm = _g("range_norm")
    lr_auc = lr.get("auroc_max") or float("nan")

    if not np.isfinite(norm):
        return {"case": "?", "summary": "insufficient data"}

    if norm > clean and norm > length and (np.isfinite(lr_auc) and lr_auc > length):
        return {
            "case": "A",
            "summary": "range_norm beats clean and length; rephrase instability is real UQ beyond length",
        }
    if raw > 0.55 and norm < 0.52 and length > 0.55:
        return {
            "case": "B",
            "summary": "raw range and length strong, normalized weak → signal mostly answer complexity",
        }
    if norm > 0.55 and length < 0.52:
        return {
            "case": "C",
            "summary": "normalized range strong, length weak → per-token rephrase instability predicts errors",
        }
    if norm > 0.5 and length > 0.5 and np.isfinite(lr_auc) and lr_auc >= max(norm, length) - 0.02:
        return {
            "case": "D",
            "summary": "both signals present; range+length LR captures complementary UQ",
        }
    return {
        "case": "mixed",
        "summary": f"no clear winner (norm={norm:.3f}, clean={clean:.3f}, length={length:.3f}, lr={lr_auc:.3f})",
    }
