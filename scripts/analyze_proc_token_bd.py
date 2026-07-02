#!/usr/bin/env python3
"""Process-level 'token bd': base vs perturb reasoning token differences."""
from __future__ import annotations

import glob
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panda.core.reasoning_token_features import _classify_token  # noqa: E402
from panda.grading.math_grader import math_equal  # noqa: E402

MATH_DS = ("minerva", "math500", "gsm8k")
SEEDS = (41, 42, 43)
MODEL = "maintable_qwen25_3b"
CACHE = Path("/root/autodl-tmp/panda-outputs/.proc_token_bd_cache.pkl")
CALC_KINDS = frozenset({"numeric", "symbol", "variable"})


def reasoning_trace(run: dict) -> list[dict]:
    trace = run.get("token_trace") or []
    a = (run.get("answer_span") or {}).get("start_token")
    if a is None:
        return []
    return trace[: int(a)]


def aligned_flip_rate(base_proc: list[dict], pert_proc: list[dict], *, calc_only: bool) -> float:
    nb, np_ = len(base_proc), len(pert_proc)
    if nb == 0 or np_ == 0:
        return float("nan")
    flips, counted = 0, 0
    for i, bt in enumerate(base_proc):
        if calc_only and _classify_token(bt.get("token", "")) not in CALC_KINDS:
            continue
        rp = i / max(nb - 1, 1)
        j = min(int(round(rp * max(np_ - 1, 1))), np_ - 1)
        pt = pert_proc[j]
        if calc_only and _classify_token(pt.get("token", "")) not in CALC_KINDS:
            continue
        counted += 1
        if int(bt.get("token_id", -1)) != int(pt.get("token_id", -2)):
            flips += 1
    return flips / counted if counted else float("nan")


def aligned_delta_mean(
    base_proc: list[dict], pert_proc: list[dict], key: str, *, calc_only: bool
) -> float:
    nb, np_ = len(base_proc), len(pert_proc)
    if nb == 0 or np_ == 0:
        return float("nan")
    deltas, counted = [], 0
    for i, bt in enumerate(base_proc):
        if calc_only and _classify_token(bt.get("token", "")) not in CALC_KINDS:
            continue
        rp = i / max(nb - 1, 1)
        j = min(int(round(rp * max(np_ - 1, 1))), np_ - 1)
        pt = pert_proc[j]
        if calc_only and _classify_token(pt.get("token", "")) not in CALC_KINDS:
            continue
        bv, pv = bt.get(key), pt.get(key)
        if bv is None or pv is None:
            continue
        deltas.append(abs(float(bv) - float(pv)))
        counted += 1
    return float(np.mean(deltas)) if counted else float("nan")


def proc_bd_features(base: dict, pert_runs: list[dict]) -> dict[str, float]:
    """Token-level bd analogues on reasoning (process) region."""
    base_proc = reasoning_trace(base)
    out: dict[str, list[float]] = {
        "proc_flip": [],
        "proc_calc_flip": [],
        "proc_margin_delta": [],
        "proc_calc_margin_delta": [],
        "proc_ent_delta": [],
        "proc_calc_ent_delta": [],
    }
    for pr in pert_runs:
        pert_proc = reasoning_trace(pr)
        for key, calc_only in [
            ("proc_flip", False),
            ("proc_calc_flip", True),
        ]:
            v = aligned_flip_rate(base_proc, pert_proc, calc_only=calc_only)
            if np.isfinite(v):
                out[key].append(v)
        for key, field, calc_only in [
            ("proc_margin_delta", "margin_top2", False),
            ("proc_calc_margin_delta", "margin_top2", True),
            ("proc_ent_delta", "entropy", False),
            ("proc_calc_ent_delta", "entropy", True),
        ]:
            v = aligned_delta_mean(base_proc, pert_proc, field, calc_only=calc_only)
            if np.isfinite(v):
                out[key].append(v)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}


def _parse_one(args: tuple) -> dict | None:
    rp, label, seed, ds = args
    try:
        raw = json.loads(Path(rp).read_text())
    except Exception:
        return None
    sm = raw.get("summary_metrics") or {}
    if sm.get("label_drop"):
        return None
    base = raw.get("base_generation") or {}
    text = list(raw.get("text_rephrase_runs") or [])
    weight = list(raw.get("weight_perturb_runs") or [])
    pert = text + weight
    a0 = str(sm.get("a0") or base.get("answer_normalized", "")).strip()
    pert_ans = [str(g.get("answer_normalized", "")).strip() for g in pert]
    if not a0 or not pert_ans:
        return None
    bd = sum(1 for x in pert_ans if not math_equal(a0, x)) / len(pert_ans)
    pf = proc_bd_features(base, pert)
    if not any(np.isfinite(v) for v in pf.values()):
        return None
    return {"seed": seed, "ds": ds, "y": int(label), "bd": bd, **pf}


def collect_jobs(out_root: Path) -> list[tuple]:
    jobs = []
    base = out_root / MODEL
    for seed in SEEDS:
        for ds in MATH_DS:
            sj = base / f"seed{seed}" / ds / "summary.jsonl"
            if not sj.exists():
                continue
            labels = {}
            for ln in sj.read_text().splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if r.get("label_drop") or r.get("label_wrong_clean") is None:
                    continue
                labels[r["id"]] = int(r["label_wrong_clean"])
            for rp in glob.glob(str(base / f"seed{seed}" / ds / "raw_runs" / "*.json")):
                if "partial" in rp or "error" in rp:
                    continue
                rid = Path(rp).stem
                if rid in labels:
                    jobs.append((rp, labels[rid], seed, ds))
    return jobs


def load_rows(out_root: Path, workers: int) -> list[dict]:
    jobs = collect_jobs(out_root)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_parse_one, j) for j in jobs]
        for fu in as_completed(futs):
            r = fu.result()
            if r:
                rows.append(r)
    return rows


def lodo_macro(rows: list[dict], cols: list[str]) -> float:
    vals = []
    for test_ds in MATH_DS:
        tr = [r for r in rows if r["ds"] != test_ds]
        scs = []
        for seed in SEEDS:
            te = [r for r in rows if r["ds"] == test_ds and r["seed"] == seed]
            if len(te) < 10:
                continue
            Xtr = np.array([[r[c] for c in cols] for r in tr], float)
            Xte = np.array([[r[c] for c in cols] for r in te], float)
            ytr = np.array([r["y"] for r in tr])
            yte = np.array([r["y"] for r in te])
            if len(np.unique(ytr)) < 2:
                continue
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(max_iter=2000)
            clf.fit((Xtr - mu) / sd, ytr)
            scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
        if scs:
            vals.append(float(np.mean(scs)))
    return float(np.mean(vals)) if vals else float("nan")


def auroc_auto(y: np.ndarray, s: np.ndarray) -> float:
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return float("nan")
    return float(max(roc_auc_score(y, s), roc_auc_score(y, -s)))


def main() -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/panda-outputs"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--use-cache", action="store_true")
    args = ap.parse_args()

    if args.use_cache and CACHE.exists():
        rows = pickle.loads(CACHE.read_bytes())
    else:
        t0 = time.time()
        rows = load_rows(args.out_root, args.workers)
        CACHE.write_bytes(pickle.dumps(rows))
        print(f"built N={len(rows)} in {time.time()-t0:.1f}s")

    bd_m = lodo_macro(rows, ["bd"])
    print(f"bd-only macro: {bd_m:.3f}\n")
    print(f"{'feature':<26} {'pooled':>7} {'tok':>7} {'bd+T':>7} {'Δ':>7} {'drop':>7}")
    y = np.array([r["y"] for r in rows])
    keys = list(rows[0].keys())
    feat_keys = [k for k in keys if k.startswith("proc_")]
    for k in feat_keys:
        s = np.array([r[k] for r in rows])
        col = k
        if auroc_auto(y, -s) > auroc_auto(y, s):
            col = f"__neg_{k}"
            for r in rows:
                r[col] = -r[k]
        else:
            for r in rows:
                r[col] = r[k]
        po = max(auroc_auto(y, s), auroc_auto(y, -s))
        tok = lodo_macro(rows, [col])
        fus = lodo_macro(rows, ["bd", col])
        print(f"{k:<26} {po:>7.3f} {tok:>7.3f} {fus:>7.3f} {fus-bd_m:>+7.3f} {bd_m-fus:>+7.3f}")

    # refs
    print("\nrefs:")
    for ref in ["T_tok_disagree_mean", "TW_ent_sum_total"]:
        pass  # from summary if needed


if __name__ == "__main__":
    main()
