#!/usr/bin/env python3
"""Fast token feature ranking — summary.jsonl + cached bd."""
from __future__ import annotations

import glob
import json
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from panda.grading.math_grader import math_equal  # noqa: E402

OUT = Path("/root/autodl-tmp/panda-outputs")
MODEL_DIR = "maintable_qwen25_3b"
MATH_DS = ("minerva", "math500", "gsm8k")
SEEDS = (41, 42, 43)
BD_CACHE = OUT / ".qwen25_bd_cache.pkl"

TOKEN_PAT = re.compile(
    r"(tok|ans_|baseline_|base_ans|ATU|margin|logprob|ent_|mar_|flip|js|cluster|"
    r"DC_|LL_|PE_|SC_|SAR|disagree|delta_ans|topk|rank|numeric_token|operator|"
    r"formula_skeleton|alternative_answer|confident_fragment)",
    re.I,
)


def _bd_from_path(rp: str) -> tuple | None:
    raw = json.loads(Path(rp).read_text())
    sm = raw.get("summary_metrics") or {}
    if sm.get("label_drop"):
        return None
    base_g = raw.get("base_generation") or {}
    text = list(raw.get("text_rephrase_runs") or [])
    weight = list(raw.get("weight_perturb_runs") or [])
    a0 = str(sm.get("a0") or base_g.get("answer_normalized", "")).strip()
    pert = [str(g.get("answer_normalized", "")).strip() for g in text + weight]
    if not a0 or not pert:
        return None
    p = Path(rp)
    return (p.stem, int(p.parts[-4].replace("seed", "")), p.parts[-3],
            sum(1 for x in pert if not math_equal(a0, x)) / len(pert))


def ensure_bd_cache() -> dict:
    if BD_CACHE.exists():
        return pickle.loads(BD_CACHE.read_bytes())
    jobs = [
        rp for seed in SEEDS for ds in MATH_DS
        for rp in glob.glob(str(OUT / MODEL_DIR / f"seed{seed}" / ds / "raw_runs" / "*.json"))
        if "partial" not in rp and "error" not in rp
    ]
    bd_map = {}
    with ProcessPoolExecutor(max_workers=16) as ex:
        for res in ex.map(_bd_from_path, jobs, chunksize=64):
            if res:
                bd_map[(res[0], res[1], res[2])] = res[3]
    BD_CACHE.write_bytes(pickle.dumps(bd_map))
    return bd_map


def load_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    bd_map = ensure_bd_cache()
    ys, ds_i, seed_i, bd, rows = [], [], [], [], []
    cands: list[str] | None = None
    ds_map = {d: i for i, d in enumerate(MATH_DS)}

    for seed in SEEDS:
        for ds in MATH_DS:
            sj = OUT / MODEL_DIR / f"seed{seed}" / ds / "summary.jsonl"
            for ln in sj.read_text().splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if r.get("label_drop") or r.get("label_wrong_clean") is None:
                    continue
                b = bd_map.get((r["id"], seed, ds))
                if b is None:
                    continue
                if cands is None:
                    cands = sorted(
                        k for k, v in r.items()
                        if isinstance(v, (int, float)) and TOKEN_PAT.search(k)
                    )
                ys.append(int(r["label_wrong_clean"]))
                ds_i.append(ds_map[ds])
                seed_i.append(seed)
                bd.append(b)
                rows.append([float(r.get(k, np.nan)) for k in cands])

    assert cands is not None
    return (
        np.array(ys), np.array(ds_i), np.array(seed_i), np.array(bd),
        np.array(rows, dtype=np.float64), cands,
    )


def pooled_auroc(y: np.ndarray, s: np.ndarray) -> float:
    m = np.isfinite(s)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return np.nan
    yy, ss = y[m], s[m]
    a = roc_auc_score(yy, ss)
    return max(a, 1 - a)


def lodo_macro(y, ds_i, seed_i, *cols: np.ndarray) -> float:
    vals = []
    for test_ds in range(3):
        tr = ds_i != test_ds
        scs = []
        for seed in SEEDS:
            te = (ds_i == test_ds) & (seed_i == seed)
            if te.sum() < 10:
                continue
            Xtr = np.column_stack([c[tr] for c in cols])
            Xte = np.column_stack([c[te] for c in cols])
            ytr, yte = y[tr], y[te]
            if len(np.unique(ytr)) < 2:
                continue
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(max_iter=1500)
            clf.fit((Xtr - mu) / sd, ytr)
            scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
        if scs:
            vals.append(np.mean(scs))
    return float(np.mean(vals)) if vals else np.nan


def lodo_per_ds(y, ds_i, seed_i, test_ds: int, *cols: np.ndarray) -> float:
    tr = ds_i != test_ds
    scs = []
    for seed in SEEDS:
        te = (ds_i == test_ds) & (seed_i == seed)
        if te.sum() < 10:
            continue
        Xtr = np.column_stack([c[tr] for c in cols])
        Xte = np.column_stack([c[te] for c in cols])
        ytr, yte = y[tr], y[te]
        if len(np.unique(ytr)) < 2:
            continue
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=1500)
        clf.fit((Xtr - mu) / sd, ytr)
        scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
    return float(np.mean(scs)) if scs else np.nan


def main() -> None:
    t0 = time.time()
    y, ds_i, seed_i, bd, X, names = load_table()
    print(f"N={len(y)} features={len(names)} loaded in {time.time()-t0:.1f}s", flush=True)

    pooled = []
    for j, name in enumerate(names):
        col = X[:, j]
        au, au_neg = pooled_auroc(y, col), pooled_auroc(y, -col)
        pooled.append((name, -1 if au_neg > au else 1, max(au, au_neg)))
    pooled.sort(key=lambda x: -x[2])

    bd_macro = lodo_macro(y, ds_i, seed_i, bd)
    print(f"\nbd-only LODO macro: {bd_macro:.3f}\n", flush=True)

    print("=== Pooled top-12 (cheap screen) ===", flush=True)
    for name, sign, au in pooled[:12]:
        print(f"  {name:<42} dir={sign:>2} pooled={au:.3f}", flush=True)

    print("\n=== LODO top-20 bd+token ===", flush=True)
    results = []
    for name, sign, _ in pooled[:20]:
        j = names.index(name)
        col = sign * X[:, j]
        results.append((name, sign, lodo_macro(y, ds_i, seed_i, col),
                        lodo_macro(y, ds_i, seed_i, bd, col)))
    results.sort(key=lambda x: -x[3])

    print(f"{'feature':<42} {'dir':>4} {'tok':>7} {'bd+tok':>7}", flush=True)
    for row in results:
        print(f"{row[0]:<42} {row[1]:>4} {row[2]:>7.3f} {row[3]:>7.3f}", flush=True)

    name, sign, _, fus_m = results[0]
    col = sign * X[:, names.index(name)]
    print(f"\n*** BEST: {name} (dir={sign}) bd+tok macro={fus_m:.3f} ***", flush=True)
    for ds_name, ds_idx in zip(MATH_DS, range(3)):
        print(f"  {ds_name}: {lodo_per_ds(y, ds_i, seed_i, ds_idx, bd, col)*100:.2f}%", flush=True)

    print("\n=== Reference ===", flush=True)
    for ref in ["T_tok_disagree_mean", "baseline_DC_min", "baseline_DC_mean",
                "base_ans_mar_mean", "T_ans_logprob_min_avg", "T_math_token_flip_mean"]:
        if ref not in names:
            continue
        j = names.index(ref)
        f1 = lodo_macro(y, ds_i, seed_i, bd, X[:, j])
        f2 = lodo_macro(y, ds_i, seed_i, bd, -X[:, j])
        print(f"  {ref:<35} bd+tok={max(f1,f2):.3f}", flush=True)

    print(f"\ntotal {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
