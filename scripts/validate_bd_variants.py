#!/usr/bin/env python3
"""Final BD decomposition / weighting / interaction sweep (post-hoc only)."""
from __future__ import annotations

import glob
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prs.ase.cluster import cluster_answers  # noqa: E402
from prs.grading.math_grader import math_equal  # noqa: E402

MATH_DS = ("minerva", "math500", "gsm8k")
SEEDS = (41, 42, 43)
MODELS = {
    "qwen25_3b": "maintable_qwen25_3b",
    "llama32_1b": "maintable_llama32_1b",
    "llama31_8b": "maintable_llama31_8b",
    "qwen3_8b": "maintable_qwen3_8b",
}
CACHE = Path("/root/autodl-tmp/prs-outputs/.bd_variant_cache.pkl")

CONFIGS = [
    ("1 D_base", ["D_base"]),
    ("2 D_text+D_weight", ["D_text", "D_weight"]),
    ("3 D_text+D_weight+D_min", ["D_text", "D_weight", "D_min"]),
    ("4 D_base+TW_ent", ["D_base", "TW_ent_sum"]),
    ("5 D_base+D_conf", ["D_base", "D_conf"]),
    ("6 D_base+D_unc", ["D_base", "D_unc"]),
    ("7 D_base+TW+inter", ["D_base", "TW_ent_sum", "I_bd_ent"]),
    ("8 D_tw+D_min+TW", ["D_text", "D_weight", "D_min", "TW_ent_sum"]),
]


def run_entropy(run: dict) -> float:
    tt = run.get("token_trace") or []
    if tt:
        return float(sum(t.get("entropy", 0.0) for t in tt))
    ents = run.get("token_entropies") or []
    return float(sum(ents)) if ents else float("nan")


def disagree_rate(a0: str, answers: list[str]) -> float:
    if not a0 or not answers:
        return float("nan")
    return float(np.mean([0.0 if math_equal(a0, x) else 1.0 for x in answers]))


def weighted_disagree(a0: str, runs: list[dict], weight_fn) -> float:
    num, den = 0.0, 0.0
    for run in runs:
        ans = str(run.get("answer_normalized", "")).strip()
        if not ans:
            continue
        h = run_entropy(run)
        if not np.isfinite(h):
            continue
        w = weight_fn(h)
        if w <= 0 or not np.isfinite(w):
            continue
        dis = 0.0 if math_equal(a0, ans) else 1.0
        num += w * dis
        den += w
    return num / den if den > 0 else float("nan")


def mode_support(answers: list[str]) -> float:
    if not answers:
        return float("nan")
    _, sizes = cluster_answers(answers)
    if not sizes:
        return float("nan")
    return float(max(sizes.values()) / len(answers))


def _parse_one(args: tuple) -> dict | None:
    rp, model_key, tw_ent = args
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
    text_ans = [str(g.get("answer_normalized", "")).strip() for g in text]
    weight_ans = [str(g.get("answer_normalized", "")).strip() for g in weight]
    pert_ans = text_ans + weight_ans
    if not a0 or len(text_ans) < 1 or len(weight_ans) < 1:
        return None

    p = Path(rp)
    seed = int(p.parts[-4].replace("seed", ""))
    ds = p.parts[-3]

    d_text = disagree_rate(a0, text_ans)
    d_weight = disagree_rate(a0, weight_ans)
    d_base = disagree_rate(a0, pert_ans)
    s_base = 1.0 - d_base if np.isfinite(d_base) else float("nan")
    s_mode = mode_support(pert_ans)
    g_mode = s_mode - s_base if np.isfinite(s_mode) and np.isfinite(s_base) else float("nan")

    all_runs = text + weight
    d_conf = weighted_disagree(a0, all_runs, lambda h: float(np.exp(-h)))
    d_unc = weighted_disagree(a0, all_runs, lambda h: h)

    tw = float(tw_ent) if tw_ent is not None and tw_ent == tw_ent else float("nan")
    d_min = min(d_text, d_weight)
    d_max = max(d_text, d_weight)

    row = {
        "model": model_key,
        "seed": seed,
        "ds": ds,
        "y": int(sm.get("label_wrong_clean", 0)),
        "D_text": d_text,
        "D_weight": d_weight,
        "D_base": d_base,
        "D_min": d_min,
        "D_max": d_max,
        "D_conf": d_conf,
        "D_unc": d_unc,
        "s_base": s_base,
        "s_mode": s_mode,
        "G_mode_base": g_mode,
        "TW_ent_sum": tw,
    }
    row["I_bd_ent"] = row["D_base"] * row["TW_ent_sum"] if np.isfinite(tw) else float("nan")
    row["I_mid_ent"] = row["D_base"] * (1 - row["D_base"]) * row["TW_ent_sum"] if np.isfinite(tw) else float("nan")
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
                tw_map = {}
                for ln in sj.read_text().splitlines():
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    if r.get("label_drop"):
                        continue
                    tw_map[r["id"]] = r.get("TW_ent_sum_total")
                for rp in glob.glob(str(base / f"seed{seed}" / ds / "raw_runs" / "*.json")):
                    if "partial" in rp or "error" in rp:
                        continue
                    rid = Path(rp).stem
                    if rid in tw_map:
                        jobs.append((rp, mk, tw_map[rid]))
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


def _impute_fit_transform(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.array([np.nanmedian(Xtr[:, j]) if np.any(np.isfinite(Xtr[:, j])) else 0.0 for j in range(Xtr.shape[1])])
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
        Xtr, Xte = _impute_fit_transform(Xtr, Xte)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=3000)
        clf.fit((Xtr - mu) / sd, ytr)
        scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
    return float(np.mean(scs)) if scs else float("nan")


def lodo_macro(rows: list[dict], cols: list[str]) -> float:
    vals = [lodo_per_ds(rows, cols, ds) for ds in MATH_DS]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def drop_one_score(rows: list[dict], full_cols: list[str], drop_col: str) -> float:
    cols = [c for c in full_cols if c != drop_col]
    if not cols:
        return lodo_macro(rows, ["D_base"])
    return lodo_macro(rows, cols)


def evaluate(rows: list[dict]) -> list[dict]:
    out = []
    for name, cols in CONFIGS:
        macro = lodo_macro(rows, cols)
        per_model = {mk: lodo_macro([r for r in rows if r["model"] == mk], cols) for mk in MODELS}
        per_ds_drop = 0
        for mk in MODELS:
            sub = [r for r in rows if r["model"] == mk]
            for ds in MATH_DS:
                full = lodo_per_ds(sub, cols, ds)
                # drop all non-D_base if only bd variant; for drop-one use remove last added feature
                bd_only = lodo_per_ds(sub, ["D_base"], ds)
                if full < bd_only - 1e-6:
                    per_ds_drop += 1
        out.append({"name": name, "cols": cols, "macro": macro, "per_model": per_model, "worse_than_bd_cells": per_ds_drop})
    return out


def report(rows: list[dict], results: list[dict], md_out: Path) -> None:
    bd_macro = lodo_macro(rows, ["D_base"])
    ref4 = next(r for r in results if r["name"].startswith("4 "))
    best = max(results, key=lambda x: x["macro"])

    lines = [
        "# BD 拆分 / 加权 / 交互 — 最后一轮验证",
        "",
        f"> N={len(rows)}，4 模型，LODO OOF LR，后处理 only。",
        "",
        f"**D_base macro**: {bd_macro:.3f}  |  **当前 ref D_base+TW_ent**: {ref4['macro']:.3f}",
        "",
        "## 8 组配置 macro LODO",
        "",
        "| # | 配置 | macro | Δ vs D_base | Qwen2.5 | Llama3.2 | Llama3.1 | Qwen3 |",
        "|---|------|------:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        pm = r["per_model"]
        lines.append(
            f"| {r['name']} | {', '.join(r['cols'])} | {r['macro']:.3f} | {r['macro']-bd_macro:+.3f} | "
            f"{pm.get('qwen25_3b', float('nan')):.3f} | {pm.get('llama32_1b', float('nan')):.3f} | "
            f"{pm.get('llama31_8b', float('nan')):.3f} | {pm.get('qwen3_8b', float('nan')):.3f} |"
        )

    lines += ["", "## 分数据集 drop-one（full vs D_base-only，负=融合更好）", ""]
    for mk in MODELS:
        sub = [r for r in rows if r["model"] == mk]
        lines.append(f"### {mk}")
        lines.append("| ds | D_base | best cfg | best | Δ | #4 TW |")
        lines.append("|----|---:|---|---:|---:|---:|")
        for ds in MATH_DS:
            b = lodo_per_ds(sub, ["D_base"], ds)
            best_v, best_n = b, "D_base"
            for r in results:
                v = lodo_per_ds(sub, r["cols"], ds)
                if v > best_v:
                    best_v, best_n = v, r["name"]
            v4 = lodo_per_ds(sub, ref4["cols"], ds)
            lines.append(f"| {ds} | {b*100:.2f} | {best_n} | {best_v*100:.2f} | {(best_v-b)*100:+.2f}pp | {v4*100:.2f} |")
        lines.append("")

    # criteria for best config
    b = best
    n_model_beat = sum(1 for mk in MODELS if b["per_model"].get(mk, 0) >= lodo_macro([r for r in rows if r["model"] == mk], ["D_base"]) - 1e-6)
    n_drop = 0
    for mk in MODELS:
        sub = [r for r in rows if r["model"] == mk]
        for ds in MATH_DS:
            if lodo_per_ds(sub, b["cols"], ds) >= lodo_per_ds(sub, ["D_base"], ds) - 1e-6:
                n_drop += 1
    llama31 = b["per_model"].get("llama31_8b", 0)
    llama31_bd = lodo_macro([r for r in rows if r["model"] == "llama31_8b"], ["D_base"])
    qwen = b["per_model"].get("qwen25_3b", 0)

    lines += [
        "## 判定",
        "",
        f"- **最佳配置**: {best['name']} macro={best['macro']:.3f} (Δ={best['macro']-bd_macro:+.3f})",
        f"- 4模型 macro 超过 D_base: {best['macro'] > bd_macro}",
        f"- 4模型中 ≥D_base: {n_model_beat}/4",
        f"- 12格 fusion≥bd-only: {12 - b['worse_than_bd_cells']}/12",
        f"- Llama3.1-8B: {llama31:.3f} vs bd {llama31_bd:.3f} ({llama31-llama31_bd:+.3f})",
        f"- Qwen2.5 vs ref#4: {qwen:.3f} vs {ref4['per_model']['qwen25_3b']:.3f}",
        "",
    ]
    if best["macro"] <= ref4["macro"] + 0.001:
        lines.append("**建议定稿**: `PRS = LODO(D_base, TW_ent_sum)` — 本轮未显著超过 #4。")
    else:
        lines.append(f"**建议定稿**: `{best['name']}` — {', '.join(best['cols'])}")

    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/prs-outputs"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--md-out", type=Path, default=ROOT / "paper/tables/table_bd_variants_final.md")
    args = ap.parse_args()

    if args.use_cache and CACHE.exists():
        rows = pickle.loads(CACHE.read_bytes())
        print(f"cache N={len(rows)}")
    else:
        t0 = time.time()
        rows = load_rows(args.out_root, args.workers)
        CACHE.write_bytes(pickle.dumps(rows))
        print(f"built N={len(rows)} in {time.time()-t0:.1f}s")

    results = evaluate(rows)
    report(rows, results, args.md_out)

    bd = lodo_macro(rows, ["D_base"])
    print(f"\nD_base macro: {bd:.3f}\n")
    for r in results:
        print(f"{r['name']:<28} macro={r['macro']:.3f} Δ={r['macro']-bd:+.3f}  qwen25={r['per_model'].get('qwen25_3b',0):.3f}")
    print(f"\nWrote {args.md_out}")


if __name__ == "__main__":
    main()
