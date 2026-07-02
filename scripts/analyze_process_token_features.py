#!/usr/bin/env python3
"""Process-token features: T_calc_margin/entropy, T_tail_margin/entropy.

  python3 scripts/analyze_process_token_features.py
  python3 scripts/analyze_process_token_features.py --use-cache
"""
from __future__ import annotations

import argparse
import glob
import json
import math
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
CACHE = Path("/root/autodl-tmp/panda-outputs/.process_token_feature_cache.pkl")

CALC_KINDS = frozenset({"numeric", "symbol", "variable"})
KAPPA = 0.10
TAIL_L = 16


def topmean_kappa(values: list[float], kappa: float = KAPPA) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    k = max(1, int(math.ceil(len(arr) * kappa)))
    part = np.partition(arr, -k)[-k:]
    return float(np.mean(part))


def per_run_process_stats(run: dict, *, tail_l: int, kappa: float) -> dict[str, float]:
    trace = run.get("token_trace") or []
    a = (run.get("answer_span") or {}).get("start_token")
    if a is None or a <= 0:
        nan = float("nan")
        return {k: nan for k in ("calc_mar_u", "calc_ent", "tail_mar_u", "tail_ent")}

    calc_mar, calc_ent, tail_mar, tail_ent = [], [], [], []
    tail_start = max(0, int(a) - tail_l)

    for i, tok in enumerate(trace[: int(a)]):
        kind = _classify_token(tok.get("token", ""))
        mar = tok.get("margin_top2")
        ent = tok.get("entropy")
        if mar is None or ent is None:
            continue
        mar, ent = float(mar), float(ent)
        if kind in CALC_KINDS:
            calc_mar.append(-mar)  # higher => more uncertain
            calc_ent.append(ent)
        if i >= tail_start:
            tail_mar.append(-mar)
            tail_ent.append(ent)

    return {
        "calc_mar_u": topmean_kappa(calc_mar, kappa),
        "calc_ent": topmean_kappa(calc_ent, kappa),
        "tail_mar_u": topmean_kappa(tail_mar, kappa),
        "tail_ent": topmean_kappa(tail_ent, kappa),
    }


def aggregate_runs(runs: list[dict], *, tail_l: int, kappa: float) -> dict[str, float]:
    keys = ("calc_mar_u", "calc_ent", "tail_mar_u", "tail_ent")
    buckets: dict[str, list[float]] = {k: [] for k in keys}
    for run in runs:
        st = per_run_process_stats(run, tail_l=tail_l, kappa=kappa)
        for k in keys:
            v = st[k]
            if np.isfinite(v):
                buckets[k].append(v)
    out = {}
    for k in keys:
        out[k] = float(np.mean(buckets[k])) if buckets[k] else float("nan")
    return out


def _parse_one(args: tuple) -> dict | None:
    rp, label, seed, ds, tail_l, kappa = args
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
    a0 = str(sm.get("a0") or base.get("answer_normalized", "")).strip()
    pert = [str(g.get("answer_normalized", "")).strip() for g in text + weight]
    if not a0 or not pert:
        return None
    bd = sum(1 for x in pert if not math_equal(a0, x)) / len(pert)
    runs = [base] + text + weight
    proc = aggregate_runs(runs, tail_l=tail_l, kappa=kappa)
    if not any(np.isfinite(proc[k]) for k in proc):
        return None
    return {
        "seed": seed,
        "ds": ds,
        "y": int(label),
        "bd": bd,
        **proc,
    }


def collect_jobs(out_root: Path, tail_l: int, kappa: float) -> list[tuple]:
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
                    jobs.append((rp, labels[rid], seed, ds, tail_l, kappa))
    return jobs


def load_rows(out_root: Path, workers: int, tail_l: int, kappa: float) -> list[dict]:
    jobs = collect_jobs(out_root, tail_l, kappa)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_parse_one, j) for j in jobs]
        for fu in as_completed(futs):
            r = fu.result()
            if r:
                rows.append(r)
    return rows


def auroc_auto(y: np.ndarray, s: np.ndarray) -> float:
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return float("nan")
    return float(max(roc_auc_score(y, s), roc_auc_score(y, -s)))


def lodo_per_ds(rows: list[dict], cols: list[str], test_ds: str) -> float:
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
        clf = LogisticRegression(max_iter=3000)
        clf.fit((Xtr - mu) / sd, ytr)
        scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
    return float(np.mean(scs)) if scs else float("nan")


def lodo_macro(rows: list[dict], cols: list[str]) -> float:
    vals = [lodo_per_ds(rows, cols, ds) for ds in MATH_DS]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def mean_std_pct(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    a = np.array(vals) * 100
    return float(a.mean()), float(a.std())


def eval_feature(rows: list[dict], key: str) -> dict:
    # pick direction by pooled auto-invert
    y = np.array([r["y"] for r in rows])
    s = np.array([r[key] for r in rows])
    au_pos = auroc_auto(y, s)
    au_neg = auroc_auto(y, -s)
    col = key if au_pos >= au_neg else f"__neg_{key}"
    if col != key:
        for r in rows:
            v = r[key]
            r[col] = -v if np.isfinite(v) else float("nan")

    tok_ds = {ds: auroc_auto(np.array([r["y"] for r in rows if r["ds"] == ds]),
                             np.array([r[col] for r in rows if r["ds"] == ds]))
              for ds in MATH_DS}
    tok_macro = lodo_macro(rows, [col])
    fus_macro = lodo_macro(rows, ["bd", col])
    bd_macro = lodo_macro(rows, ["bd"])
    drop = lodo_macro(rows, ["bd"])  # -tok ablation = bd only per ds

    per_ds_fus = {ds: lodo_per_ds(rows, ["bd", col], ds) for ds in MATH_DS}
    per_ds_bd = {ds: lodo_per_ds(rows, ["bd"], ds) for ds in MATH_DS}
    per_ds_drop = {ds: per_ds_bd[ds] for ds in MATH_DS}

    return {
        "key": key,
        "col": col,
        "pooled_auroc": max(au_pos, au_neg),
        "tok_lodo_macro": tok_macro,
        "bd_lodo_macro": bd_macro,
        "fus_lodo_macro": fus_macro,
        "delta_fus_vs_bd": fus_macro - bd_macro,
        "tok_ds": tok_ds,
        "fus_ds": per_ds_fus,
        "drop_ds": per_ds_drop,
        "drop_delta": {ds: per_ds_drop[ds] - per_ds_fus[ds] for ds in MATH_DS},
    }


def report(rows: list[dict], md_out: Path, *, tail_l: int, kappa: float) -> None:
    feats = [
        ("T_calc_margin", "calc_mar_u"),
        ("T_calc_entropy", "calc_ent"),
        ("T_tail_margin", "tail_mar_u"),
        ("T_tail_entropy", "tail_ent"),
    ]
    results = [eval_feature(rows, k) for _, k in feats]
    bd_macro = lodo_macro(rows, ["bd"])

    lines = [
        "# Process-token 特征诊断（calc / tail × margin / entropy）",
        "",
        f"> Qwen2.5-3B, N={len(rows)}, κ={kappa}, tail L={tail_l}, K=9 runs TopMean_κ 后均值。",
        "",
        f"**bd-only LODO macro**: {bd_macro:.3f}",
        "",
        "## 1. 单特征 pooled AUROC（auto-invert）",
        "",
        "| 特征 | pooled AUROC | >0.6? |",
        "|------|------:|:---:|",
    ]
    for (name, _), res in zip(feats, results):
        ok = "✓" if res["pooled_auroc"] >= 0.6 else "✗"
        lines.append(f"| {name} | {res['pooled_auroc']:.3f} | {ok} |")

    lines += [
        "",
        "## 2. LODO macro（tok-only / bd+tok / Δ vs bd）",
        "",
        "| 特征 | tok-only | bd+tok | Δ vs bd | drop tok Δ |",
        "|------|------:|------:|------:|------:|",
    ]
    for (name, _), res in zip(feats, results):
        drop_macro = float(np.mean([res["drop_delta"][ds] for ds in MATH_DS]))
        lines.append(
            f"| {name} | {res['tok_lodo_macro']:.3f} | {res['fus_lodo_macro']:.3f} | "
            f"{res['delta_fus_vs_bd']:+.3f} | {drop_macro:+.3f} |"
        )

    lines += ["", "## 3. 分数据集 bd+tok vs bd-only", ""]
    for (name, _), res in zip(feats, results):
        lines.append(f"### {name}")
        lines.append("| ds | tok pooled | bd+tok LODO | bd-only | Δ fusion | drop tok |")
        lines.append("|----|------:|------:|------:|------:|------:|")
        for ds in MATH_DS:
            lines.append(
                f"| {ds} | {res['tok_ds'][ds]:.3f} | {res['fus_ds'][ds]:.3f} | "
                f"{res['drop_ds'][ds]:.3f} | {res['fus_ds'][ds]-res['drop_ds'][ds]:+.3f} | "
                f"{res['drop_delta'][ds]:+.3f} |"
            )
        lines.append("")

    best = max(results, key=lambda r: r["fus_lodo_macro"])
    lines += [
        "## 4. 结论",
        "",
        f"- **最佳 bd+tok**：{best['key']} macro={best['fus_lodo_macro']:.3f} "
        f"(Δ={best['delta_fus_vs_bd']:+.3f} vs bd-only {bd_macro:.3f})",
        "- cliff 已弃用；本表不含 cliff。",
    ]
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/panda-outputs"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--tail-l", type=int, default=TAIL_L)
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--md-out", type=Path, default=ROOT / "paper/tables/table_process_token.md")
    args = ap.parse_args()
    tail_l, kappa = args.tail_l, args.kappa

    if args.use_cache and CACHE.exists():
        rows = pickle.loads(CACHE.read_bytes())
        print(f"cache hit N={len(rows)}")
    else:
        import time
        t0 = time.time()
        rows = load_rows(args.out_root, args.workers, tail_l, kappa)
        CACHE.write_bytes(pickle.dumps(rows))
        print(f"built cache N={len(rows)} in {time.time()-t0:.1f}s")

    report(rows, args.md_out, tail_l=tail_l, kappa=kappa)
    bd_macro = lodo_macro(rows, ["bd"])
    print(f"\nbd-only LODO macro: {bd_macro:.3f}\n")
    for label, key in [
        ("T_calc_margin", "calc_mar_u"),
        ("T_calc_entropy", "calc_ent"),
        ("T_tail_margin", "tail_mar_u"),
        ("T_tail_entropy", "tail_ent"),
    ]:
        res = eval_feature(rows, key)
        print(
            f"{label:16s} pooled={res['pooled_auroc']:.3f}  "
            f"tok={res['tok_lodo_macro']:.3f}  bd+tok={res['fus_lodo_macro']:.3f}  "
            f"Δ={res['delta_fus_vs_bd']:+.3f}"
        )
    print(f"\nWrote {args.md_out}")


if __name__ == "__main__":
    main()
