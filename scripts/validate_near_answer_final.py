#!/usr/bin/env python3
"""Final near-answer token variants: late-gap, slope, ratio, text/weight split, BD interaction."""
from __future__ import annotations

import glob
import json
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prs.grading.math_grader import math_equal  # noqa: E402

MATH_DS = ("minerva", "math500", "gsm8k")
SEEDS = (41, 42, 43)
MODELS = {
    "qwen25_3b": "maintable_qwen25_3b",
    "llama32_1b": "maintable_llama32_1b",
    "llama31_8b": "maintable_llama31_8b",
    "qwen3_8b": "maintable_qwen3_8b",
}
CACHE = Path("/root/autodl-tmp/prs-outputs/.proc_near_answer_final_cache.pkl")

CONFIGS: list[tuple[str, list[str]]] = [
    ("1 prox_lin", ["bd", "T_ent_prox_lin"]),
    ("2 prox_exp_tau16", ["bd", "T_ent_prox_exp_tau16"]),
    ("3 prox_exp_tau32", ["bd", "T_ent_prox_exp_tau32"]),
    ("4 late20-early50", ["bd", "T_ent_late20_minus_early50"]),
    ("5 ent_slope", ["bd", "T_ent_slope"]),
    ("6 late20_ratio", ["bd", "T_ent_late20_ratio"]),
    ("7 prox_text+weight", ["bd", "T_ent_prox_text", "T_ent_prox_weight"]),
    ("8 prox+BD×prox", ["bd", "T_ent_prox_lin", "I_bd_x_prox_lin"]),
]


def _pre_answer(run: dict) -> list[dict]:
    trace = run.get("token_trace") or []
    a = (run.get("answer_span") or {}).get("start_token")
    return trace[: int(a)] if a is not None and int(a) > 0 else []


def _prox_weight(i: int, n: int, mode: str = "linear", tau: float = 16.0) -> float:
    if n <= 1:
        return 1.0
    u = (i + 1) / n
    if mode == "linear":
        return u
    if mode == "exp":
        return float(math.exp(tau * u))
    return u


def _mean_entropy(tokens: list[dict]) -> float:
    hs = [float(t["entropy"]) for t in tokens if t.get("entropy") is not None]
    return float(np.mean(hs)) if hs else float("nan")


def prox_weighted_entropy(pre: list[dict], mode: str = "linear", tau: float = 16.0) -> float:
    n = len(pre)
    if n == 0:
        return float("nan")
    s, wsum = 0.0, 0.0
    for i, t in enumerate(pre):
        h = t.get("entropy")
        if h is None:
            continue
        w = _prox_weight(i, n, mode, tau)
        s += w * float(h)
        wsum += w
    return s / wsum if wsum > 0 else float("nan")


def ent_late_minus_early(pre: list[dict], late_frac: float = 0.2, early_frac: float = 0.5) -> float:
    n = len(pre)
    if n == 0:
        return float("nan")
    early_end = max(1, int(math.floor(n * early_frac)))
    late_start = min(n - 1, int(math.floor(n * (1.0 - late_frac))))
    if late_start >= n:
        late_start = n - 1
    late = _mean_entropy(pre[late_start:])
    early = _mean_entropy(pre[:early_end])
    if not (np.isfinite(late) and np.isfinite(early)):
        return float("nan")
    return late - early


def ent_slope(pre: list[dict]) -> float:
    n = len(pre)
    if n < 3:
        return float("nan")
    rs, hs = [], []
    for i, t in enumerate(pre):
        h = t.get("entropy")
        if h is None:
            continue
        rs.append((i + 1) / n)
        hs.append(float(h))
    if len(rs) < 3:
        return float("nan")
    r = np.asarray(rs, float)
    h = np.asarray(hs, float)
    vr = r.var()
    if vr < 1e-12:
        return float("nan")
    return float(np.cov(r, h)[0, 1] / vr)


def ent_late_ratio(pre: list[dict], late_frac: float = 0.2) -> float:
    n = len(pre)
    if n == 0:
        return float("nan")
    late_start = max(0, int(math.floor(n * (1.0 - late_frac))))
    all_sum = 0.0
    late_sum = 0.0
    for i, t in enumerate(pre):
        h = t.get("entropy")
        if h is None:
            continue
        v = float(h)
        all_sum += v
        if i >= late_start:
            late_sum += v
    if all_sum <= 0:
        return float("nan")
    return late_sum / (all_sum + 1e-9)


def _mean_feat(runs: list[dict], fn) -> float:
    vals = [fn(_pre_answer(r)) for r in runs]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def extract_features(text_runs: list[dict], weight_runs: list[dict]) -> dict[str, float]:
    pert = text_runs + weight_runs

    def over(fn):
        return _mean_feat(pert, fn)

    return {
        "T_ent_prox_lin": over(lambda pre: prox_weighted_entropy(pre, "linear")),
        "T_ent_prox_exp_tau16": over(lambda pre: prox_weighted_entropy(pre, "exp", 16.0)),
        "T_ent_prox_exp_tau32": over(lambda pre: prox_weighted_entropy(pre, "exp", 32.0)),
        "T_ent_late20_minus_early50": over(lambda pre: ent_late_minus_early(pre, 0.2, 0.5)),
        "T_ent_slope": over(ent_slope),
        "T_ent_late20_ratio": over(lambda pre: ent_late_ratio(pre, 0.2)),
        "T_ent_prox_text": _mean_feat(text_runs, lambda pre: prox_weighted_entropy(pre, "linear")),
        "T_ent_prox_weight": _mean_feat(weight_runs, lambda pre: prox_weighted_entropy(pre, "linear")),
    }


def _parse_one(args: tuple) -> dict | None:
    rp, label, seed, ds, model_key = args
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
    feats = extract_features(text, weight)
    tw = float(sm.get("TW_ent_sum_total", float("nan")))
    row = {
        "model": model_key,
        "seed": seed,
        "ds": ds,
        "y": int(label),
        "bd": bd,
        "TW_ent_sum": tw,
        **feats,
    }
    row["I_bd_x_prox_lin"] = bd * feats["T_ent_prox_lin"]
    return row


def collect_jobs(out_root: Path) -> list[tuple]:
    jobs = []
    for mk, dname in MODELS.items():
        base = out_root / dname
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
                    stem = Path(rp).stem
                    if stem in labels:
                        jobs.append((rp, labels[stem], seed, ds, mk))
    return jobs


def load_rows(out_root: Path, workers: int) -> list[dict]:
    jobs = collect_jobs(out_root)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_parse_one, jobs, chunksize=32):
            if r:
                rows.append(r)
    return rows


def _impute_fit(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(Xtr, axis=0)
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    for j in range(Xtr.shape[1]):
        Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
        Xte[~np.isfinite(Xte[:, j]), j] = med[j]
    return Xtr, Xte


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
        Xtr, Xte = _impute_fit(Xtr, Xte)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=3000)
        clf.fit((Xtr - mu) / sd, ytr)
        scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
    return float(np.mean(scs)) if scs else float("nan")


def lodo_macro(rows: list[dict], cols: list[str]) -> float:
    vals = [lodo_per_ds(rows, cols, ds) for ds in MATH_DS]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def pooled_auroc(rows: list[dict], key: str) -> float:
    y = np.array([r["y"] for r in rows])
    s = np.array([r[key] for r in rows], float)
    m = np.isfinite(s)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return float("nan")
    a = roc_auc_score(y[m], s[m])
    return float(max(a, 1 - a))


def evaluate(rows: list[dict]) -> list[dict]:
    out = []
    for name, cols in CONFIGS:
        macro = lodo_macro(rows, cols)
        per_model = {mk: lodo_macro([r for r in rows if r["model"] == mk], cols) for mk in MODELS}
        worse = 0
        for mk in MODELS:
            sub = [r for r in rows if r["model"] == mk]
            for ds in MATH_DS:
                full = lodo_per_ds(sub, cols, ds)
                bd_only = lodo_per_ds(sub, ["bd"], ds)
                if full < bd_only - 1e-6:
                    worse += 1
        out.append({"name": name, "cols": cols, "macro": macro, "per_model": per_model, "worse_cells": worse})
    return out


def report(rows: list[dict], results: list[dict], md_out: Path) -> None:
    bd = lodo_macro(rows, ["bd"])
    ref_tw = lodo_macro(rows, ["bd", "TW_ent_sum"])
    ref1 = next(r for r in results if r["name"].startswith("1 "))
    best = max(results, key=lambda x: x["macro"])

    lines = [
        "# 近答案 token 指标 — 最后一轮（8 配置）",
        "",
        f"> N={len(rows)}，LODO OOF LR。",
        "",
        f"**bd-only**: {bd:.3f} | **bd+TW_ent_sum**: {ref_tw:.3f} | **bd+T_ent_prox_lin (ref)**: {ref1['macro']:.3f}",
        "",
        "## macro LODO",
        "",
        "| # | 配置 | macro | Δ vs bd | Qwen2.5 | Llama3.2 | Llama3.1 | Qwen3 | pooled T |",
        "|---|------|------:|---:|---:|---:|---:|---:|---:|",
    ]
    t_key = {
        "1 prox_lin": "T_ent_prox_lin",
        "2 prox_exp_tau16": "T_ent_prox_exp_tau16",
        "3 prox_exp_tau32": "T_ent_prox_exp_tau32",
        "4 late20-early50": "T_ent_late20_minus_early50",
        "5 ent_slope": "T_ent_slope",
        "6 late20_ratio": "T_ent_late20_ratio",
        "7 prox_text+weight": "T_ent_prox_text",
        "8 prox+BD×prox": "T_ent_prox_lin",
    }
    for r in results:
        pm = r["per_model"]
        tk = t_key.get(r["name"], "")
        po = pooled_auroc(rows, tk) if tk else float("nan")
        lines.append(
            f"| {r['name']} | `{', '.join(c for c in r['cols'] if c != 'bd')}` | "
            f"{r['macro']:.3f} | {r['macro'] - bd:+.3f} | "
            f"{pm.get('qwen25_3b', float('nan')):.3f} | {pm.get('llama32_1b', float('nan')):.3f} | "
            f"{pm.get('llama31_8b', float('nan')):.3f} | {pm.get('qwen3_8b', float('nan')):.3f} | "
            f"{po:.3f} |"
        )

    lines += ["", "## 12 格 drop-one（fusion vs bd-only，负=融合更差）", ""]
    for mk in MODELS:
        sub = [r for r in rows if r["model"] == mk]
        lines.append(f"### {mk}")
        lines.append("| ds | bd | #1 prox | #5 slope | #7 T+W | best |")
        lines.append("|----|---:|---:|---:|---:|---|")
        r1 = next(r for r in results if r["name"].startswith("1 "))
        r5 = next(r for r in results if r["name"].startswith("5 "))
        r7 = next(r for r in results if r["name"].startswith("7 "))
        for ds in MATH_DS:
            b = lodo_per_ds(sub, ["bd"], ds)
            v1 = lodo_per_ds(sub, r1["cols"], ds)
            v5 = lodo_per_ds(sub, r5["cols"], ds)
            v7 = lodo_per_ds(sub, r7["cols"], ds)
            best_v, best_n = b, "bd"
            for r in results:
                v = lodo_per_ds(sub, r["cols"], ds)
                if v > best_v:
                    best_v, best_n = v, r["name"]
            lines.append(
                f"| {ds} | {b * 100:.2f} | {v1 * 100:.2f} | {v5 * 100:.2f} | {v7 * 100:.2f} | {best_n} |"
            )
        lines.append("")

    llama31 = best["per_model"].get("llama31_8b", 0)
    llama31_bd = lodo_macro([r for r in rows if r["model"] == "llama31_8b"], ["bd"])
    qwen = best["per_model"].get("qwen25_3b", 0)
    qwen_ref = ref1["per_model"].get("qwen25_3b", 0)
    gain_vs_ref1 = best["macro"] - ref1["macro"]

    lines += [
        "## 判定",
        "",
        f"- **最佳**: {best['name']} macro={best['macro']:.3f} (Δ vs bd {best['macro'] - bd:+.3f})",
        f"- vs #1 prox_lin: {gain_vs_ref1:+.3f} macro",
        f"- 12格 fusion≥bd: {12 - best['worse_cells']}/12",
        f"- Llama3.1: {llama31:.3f} vs bd {llama31_bd:.3f} ({llama31 - llama31_bd:+.3f})",
        f"- Qwen2.5: {qwen:.3f} vs prox_lin {qwen_ref:.3f}",
        "",
    ]
    if gain_vs_ref1 < 0.002:
        lines.append(
            f"**定稿建议**: `PRS = LODO(bd, T_ent_prox_lin)` — 新 variant 未超过 +0.002，保留 linear。"
        )
    else:
        lines.append(f"**定稿建议**: `{best['name']}` — `{', '.join(best['cols'])}`")

    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/prs-outputs"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--md-out", type=Path, default=ROOT / "paper/tables/table_near_answer_final.md")
    args = ap.parse_args()

    if args.use_cache and CACHE.exists():
        rows = pickle.loads(CACHE.read_bytes())
        print(f"loaded cache N={len(rows)}")
    else:
        t0 = time.time()
        rows = load_rows(args.out_root, args.workers)
        CACHE.write_bytes(pickle.dumps(rows))
        print(f"built N={len(rows)} in {time.time() - t0:.1f}s")

    bd = lodo_macro(rows, ["bd"])
    ref_tw = lodo_macro(rows, ["bd", "TW_ent_sum"])
    print(f"bd-only={bd:.3f}  bd+TW_ent={ref_tw:.3f}\n")
    print(f"{'config':<22} {'macro':>7} {'Δbd':>7} {'Q25':>7} {'L31':>7} {'12ok':>5}")
    results = evaluate(rows)
    for r in sorted(results, key=lambda x: -x["macro"]):
        pm = r["per_model"]
        print(
            f"{r['name']:<22} {r['macro']:>7.3f} {r['macro']-bd:>+7.3f} "
            f"{pm['qwen25_3b']:>7.3f} {pm['llama31_8b']:>7.3f} {12-r['worse_cells']:>4}/12"
        )

    report(rows, results, args.md_out)
    print(f"\nWrote {args.md_out}")


if __name__ == "__main__":
    main()
