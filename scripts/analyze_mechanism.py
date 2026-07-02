#!/usr/bin/env python3
"""Mechanism analysis: token-level (T0, DeepConf) vs answer-level (bd, F_resp).

CPU-only; reads existing raw_runs + summary.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from panda.grading.math_grader import math_equal  # noqa: E402
from panda.baselines.token_scores import deepconf_mean  # noqa: E402

MATH_DS = ("minerva", "math500", "gsm8k")
SEEDS = (41, 42, 43)
MODELS = {
    "qwen25_3b": "maintable_qwen25_3b",
    "llama32_1b": "maintable_llama32_1b",
    "llama31_8b": "maintable_llama31_8b",
    "qwen3_8b": "maintable_qwen3_8b",
}
DS_LABEL = {"math500": "MATH-500", "gsm8k": "GSM8K", "minerva": "Minerva"}
EQ: dict[tuple[str, str], bool] = {}


def eq(a: str, b: str) -> bool:
    if a == b:
        return True
    k = (a, b) if a < b else (b, a)
    if k in EQ:
        return EQ[k]
    try:
        v = bool(math_equal(a, b))
    except Exception:
        v = False
    EQ[k] = v
    return v


def cliff(run: dict, w: int = 2) -> float:
    tt = run.get("token_trace") or []
    a = (run.get("answer_span") or {}).get("start_token")
    if a is None or a < w or a >= len(tt):
        return float("nan")
    rl = [t.get("logprob") for t in tt[a - w : a] if t.get("logprob") is not None]
    al = [t.get("logprob") for t in tt[a : min(len(tt), a + w)] if t.get("logprob") is not None]
    if not rl or not al:
        return float("nan")
    return float(np.mean(rl) - np.mean(al))


def bd_rate(a0: str, answers: list[str]) -> float:
    if not a0 or not answers:
        return 0.0
    return float(np.mean([0.0 if eq(a0, x) else 1.0 for x in answers]))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    mask = np.isfinite(s)
    y, s = y[mask].astype(int), s[mask].astype(float)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return float("nan")
    try:
        a_pos = roc_auc_score(y, s)
        a_neg = roc_auc_score(y, -s)
        return float(max(a_pos, a_neg))
    except Exception:
        return float("nan")


def lodo_lr_auroc(rows: list[dict], feat_keys: list[str], *, invert: dict[str, bool] | None = None) -> dict[str, float]:
    """Leave-one-dataset-out LR AUROC averaged over 3 math folds."""
    invert = invert or {}
    by_ds: dict[str, list] = {d: [] for d in MATH_DS}
    for r in rows:
        by_ds[r["ds"]].append(r)
    scores = []
    for test_ds in MATH_DS:
        train = [r for d in MATH_DS if d != test_ds for r in by_ds[d]]
        test = by_ds[test_ds]
        if len(train) < 50 or len(test) < 20:
            continue
        y_tr = np.array([r["y"] for r in train], dtype=int)
        y_te = np.array([r["y"] for r in test], dtype=int)
        x_tr = np.array([[(-1 if invert.get(k) else 1) * r[k] for k in feat_keys] for r in train], dtype=float)
        x_te = np.array([[(-1 if invert.get(k) else 1) * r[k] for k in feat_keys] for r in test], dtype=float)
        mask_tr = np.all(np.isfinite(x_tr), axis=1)
        mask_te = np.all(np.isfinite(x_te), axis=1)
        if mask_tr.sum() < 30 or mask_te.sum() < 10:
            continue
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr[mask_tr])
        x_te_s = scaler.transform(x_te[mask_te])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(x_tr_s, y_tr[mask_tr])
        probs = clf.predict_proba(x_te_s)[:, 1]
        scores.append(auroc(y_te[mask_te], probs))
    return float(np.nanmean(scores)) if scores else float("nan")


def parse_record(raw_path: Path, summ: dict) -> dict | None:
    try:
        r = json.loads(raw_path.read_text())
    except Exception:
        return None
    sm = r.get("summary_metrics") or {}
    if sm.get("label_drop"):
        return None
    rid = r.get("id")
    if rid not in summ:
        return None
    s = summ[rid]
    y = s.get("label_wrong_clean")
    if y is None:
        return None

    base = r.get("base_generation") or {}
    text = list(r.get("text_rephrase_runs") or [])
    weight = list(r.get("weight_perturb_runs") or [])
    a0 = str(sm.get("a0") or base.get("answer_normalized", "")).strip()
    text_ans = [str(g.get("answer_normalized", "")).strip() for g in text]
    weight_ans = [str(g.get("answer_normalized", "")).strip() for g in weight]
    pert = text_ans + weight_ans

    t0 = cliff(base)
    dc = deepconf_mean(base.get("token_trace") or [], base.get("answer_span"))
    dc_min = float(dc.get("baseline_DC_min", float("nan")))

    run_panel = []
    for g in text + weight:
        ci = cliff(g)
        ai = str(g.get("answer_normalized", "")).strip()
        run_panel.append({"cliff": ci, "flip": 0.0 if eq(a0, ai) else 1.0})

    return {
        "id": rid,
        "model": None,
        "seed": None,
        "ds": r.get("dataset"),
        "y": int(y),
        "F": float(sm.get("F_resp") or sm.get("TW_ASE") or 0),
        "bd": bd_rate(a0, pert),
        "bd_text": bd_rate(a0, text_ans),
        "bd_weight": bd_rate(a0, weight_ans),
        "T0": t0,
        "T_full": float(np.nanmean([cliff(g) for g in [base] + text + weight])),
        "dc_min": dc_min,
        "split": 1.0 if bd_rate(a0, pert) > 0 else 0.0,
        "run_panel": run_panel,
    }


def load_all(out_root: Path) -> list[dict]:
    rows = []
    for model_key, dir_name in MODELS.items():
        base = out_root / dir_name
        for seed in SEEDS:
            for ds in MATH_DS:
                summ_path = base / f"seed{seed}" / ds / "summary.jsonl"
                if not summ_path.exists():
                    continue
                summ = {}
                for ln in summ_path.read_text().splitlines():
                    if not ln.strip():
                        continue
                    rec = json.loads(ln)
                    if rec.get("label_drop", 0):
                        continue
                    summ[rec["id"]] = rec
                for rp in glob.glob(str(base / f"seed{seed}" / ds / "raw_runs" / "*.json")):
                    if "partial" in rp or "error" in rp:
                        continue
                    row = parse_record(Path(rp), summ)
                    if row is None:
                        continue
                    row["model"] = model_key
                    row["seed"] = seed
                    rows.append(row)
    return rows


def median_split(vals: np.ndarray) -> float:
    v = vals[np.isfinite(vals)]
    return float(np.median(v)) if len(v) else 0.0


def enrichment(rows: list[dict], *, t_key: str = "T0", bd_thr: float = 0.0) -> dict:
    """P(T low & bd>thr) for wrong vs correct, per dataset."""
    out = {}
    for ds in MATH_DS:
        sub = [r for r in rows if r["ds"] == ds]
        if not sub:
            continue
        t_med = median_split(np.array([r[t_key] for r in sub]))
        def rate(label_y: int) -> float:
            ss = [r for r in sub if r["y"] == label_y]
            if not ss:
                return float("nan")
            hit = sum(1 for r in ss if np.isfinite(r[t_key]) and r[t_key] < t_med and r["bd"] > bd_thr)
            valid = sum(1 for r in ss if np.isfinite(r[t_key]))
            return hit / valid if valid else float("nan")
        out[ds] = {
            "t_median": t_med,
            "p_pattern_wrong": rate(1),
            "p_pattern_correct": rate(0),
            "n_wrong": sum(1 for r in sub if r["y"] == 1),
            "n_correct": sum(1 for r in sub if r["y"] == 0),
        }
        w, c = out[ds]["p_pattern_wrong"], out[ds]["p_pattern_correct"]
        out[ds]["enrichment"] = (w / c) if c and np.isfinite(w) and np.isfinite(c) and c > 0 else float("nan")
    return out


def run_level_flip_curve(rows: list[dict], n_bins: int = 5) -> list[dict]:
    """P(flip | cliff bin) from 8 perturb runs per question."""
    cliffs, flips = [], []
    for r in rows:
        for p in r["run_panel"]:
            if np.isfinite(p["cliff"]):
                cliffs.append(p["cliff"])
                flips.append(p["flip"])
    if len(cliffs) < 50:
        return []
    cliffs = np.array(cliffs)
    flips = np.array(flips)
    qs = np.quantile(cliffs, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        if i == n_bins - 1:
            mask = (cliffs >= lo) & (cliffs <= hi)
        else:
            mask = (cliffs >= lo) & (cliffs < hi)
        if mask.sum() < 10:
            continue
        out.append({
            "bin": i,
            "cliff_lo": float(lo),
            "cliff_hi": float(hi),
            "cliff_mid": float(np.mean(cliffs[mask])),
            "p_flip": float(flips[mask].mean()),
            "n": int(mask.sum()),
        })
    return out


def conditional_auroc(rows: list[dict]) -> dict:
    subsets = {
        "all": lambda r: True,
        "bd_eq_0": lambda r: r["bd"] == 0,
        "bd_gt_0": lambda r: r["bd"] > 0,
        "bd0_wrong": lambda r: r["bd"] == 0 and r["y"] == 1,
        "bd0_correct": lambda r: r["bd"] == 0 and r["y"] == 0,
    }
    predictors = {
        "T0": ("T0", False),
        "bd": ("bd", False),
        "F": ("F", False),
        "dc_min": ("dc_min", True),
        "T_full": ("T_full", False),
    }
    out = {}
    for sname, fn in subsets.items():
        sub = [r for r in rows if fn(r)]
        out[sname] = {"n": len(sub), "wrong_rate": float(np.mean([r["y"] for r in sub])) if sub else float("nan")}
        for pname, (key, inv) in predictors.items():
            y = np.array([r["y"] for r in sub], dtype=int)
            s = np.array([(-1 if inv else 1) * r[key] for r in sub], dtype=float)
            out[sname][f"auroc_{pname}"] = auroc(y, s)
    return out


def residual_auroc(rows: list[dict]) -> dict:
    """Orthogonal residual AUROC: bd ~ T0 and T0 ~ bd."""
    out = {}
    for pred_key, resp_key in [("bd", "T0"), ("T0", "bd")]:
        x = np.array([r[resp_key] for r in rows], dtype=float)
        y = np.array([r[pred_key] for r in rows], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            out[f"residual_{pred_key}_given_{resp_key}"] = float("nan")
            continue
        coef = np.polyfit(x[mask], y[mask], 1)
        resid = y[mask] - np.polyval(coef, x[mask])
        labels = np.array([r["y"] for r in rows])[mask]
        out[f"residual_{pred_key}_given_{resp_key}"] = auroc(labels, resid)
    return out


def ladder(rows: list[dict]) -> dict:
    inv = {"dc_min": True}
    return {
        "L0_dc_min": lodo_lr_auroc(rows, ["dc_min"], invert=inv),
        "L1_dc_T0": lodo_lr_auroc(rows, ["dc_min", "T0"], invert=inv),
        "L2_dc_T0_bd": lodo_lr_auroc(rows, ["dc_min", "T0", "bd"], invert=inv),
        "L3_dc_T0_bd_F": lodo_lr_auroc(rows, ["dc_min", "T0", "bd", "F"], invert=inv),
        "PANDA_full": lodo_lr_auroc(rows, ["F", "bd", "T_full"], invert={}),
    }


def spearman_block(rows: list[dict]) -> dict[str, float]:
    keys = ["T0", "T_full", "bd", "F", "dc_min", "y"]
    out = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            va = np.array([r[a] for r in rows], dtype=float)
            vb = np.array([r[b] for r in rows], dtype=float)
            mask = np.isfinite(va) & np.isfinite(vb)
            if mask.sum() < 30:
                rho = float("nan")
            else:
                rho, _ = spearmanr(va[mask], vb[mask])
            out[f"{a}~{b}"] = float(rho)
    return out


def predict_split(rows: list[dict]) -> dict:
    y = np.array([r["split"] for r in rows], dtype=float)
    out = {}
    for key, inv in [("T0", False), ("F", False), ("dc_min", True)]:
        s = np.array([(-1 if inv else 1) * r[key] for r in rows], dtype=float)
        out[f"auroc_{key}_to_split"] = auroc(y.astype(int), s)
    return out


def perturb_heterogeneity(rows: list[dict]) -> dict:
    y_text = np.array([r["bd_text"] for r in rows])
    y_weight = np.array([r["bd_weight"] for r in rows])
    t0 = np.array([r["T0"] for r in rows])
    mask = np.isfinite(t0)
    out = {
        "mean_bd_text": float(np.mean(y_text)),
        "mean_bd_weight": float(np.mean(y_weight)),
        "spearman_T0_bd_text": float(spearmanr(t0[mask], y_text[mask])[0]) if mask.sum() > 30 else float("nan"),
        "spearman_T0_bd_weight": float(spearmanr(t0[mask], y_weight[mask])[0]) if mask.sum() > 30 else float("nan"),
    }
    return out


def quadrant_wrong_rate(rows: list[dict]) -> list[dict]:
    t_med = median_split(np.array([r["T0"] for r in rows]))
    bd_med = median_split(np.array([r["bd"] for r in rows]))
    quads = [
        ("T_low_bd_low", lambda r: r["T0"] < t_med and r["bd"] <= bd_med),
        ("T_low_bd_high", lambda r: r["T0"] < t_med and r["bd"] > bd_med),
        ("T_high_bd_low", lambda r: r["T0"] >= t_med and r["bd"] <= bd_med),
        ("T_high_bd_high", lambda r: r["T0"] >= t_med and r["bd"] > bd_med),
    ]
    out = []
    for name, fn in quads:
        sub = [r for r in rows if np.isfinite(r["T0"]) and fn(r)]
        if not sub:
            continue
        out.append({
            "quadrant": name,
            "n": len(sub),
            "wrong_rate": float(np.mean([r["y"] for r in sub])),
            "mean_bd": float(np.mean([r["bd"] for r in sub])),
            "mean_T0": float(np.mean([r["T0"] for r in sub])),
        })
    return out


def fmt(x: float, pct: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    if pct:
        return f"{x*100:.1f}%"
    return f"{x:.3f}"


def write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# Mechanism Analysis: Token (T₀, DeepConf) vs Answer (bd, F_resp)",
        "",
        f"> N={results['n_samples']} samples, {results['n_models']} models × 3 math datasets × 3 seeds.",
        "> T₀ = cliff on **base run only**; bd = 8-vote disagree rate; split = 1[bd>0].",
        "",
        "## I. Token → Answer split (predictive)",
        "",
        "| Predictor | AUROC → split (bd>0) |",
        "|-----------|---------------------:|",
    ]
    for k, v in results["predict_split"].items():
        lines.append(f"| {k.replace('auroc_', '').replace('_to_split', '')} | {fmt(v)} |")

    lines += ["", "## II. Pattern enrichment P(T₀ low ∧ bd>0)", ""]
    lines.append("| Dataset | P(wrong) | P(correct) | Enrichment |")
    lines.append("|---------|----------:|-----------:|-----------:|")
    for ds, e in results["enrichment"].items():
        lines.append(
            f"| {DS_LABEL.get(ds, ds)} | {fmt(e['p_pattern_wrong'], True)} | "
            f"{fmt(e['p_pattern_correct'], True)} | {fmt(e['enrichment'])}× |"
        )

    lines += ["", "## III. Run-level: P(flip | cliff bin)", ""]
    lines.append("| cliff bin mid | P(flip) | n |")
    lines.append("|--------------:|--------:|--:|")
    for b in results["flip_curve"]:
        lines.append(f"| {fmt(b['cliff_mid'])} | {fmt(b['p_flip'], True)} | {b['n']} |")

    lines += ["", "## IV. Conditional AUROC (wrong detection)", ""]
    cond = results["conditional"]
    preds = ["T0", "bd", "F", "dc_min"]
    lines.append("| Subset | n | wrong% | " + " | ".join(preds) + " |")
    lines.append("|--------|--:|-------:|" + "|".join(["--:"] * len(preds)) + "|")
    for sname, d in cond.items():
        cells = [fmt(d.get(f"auroc_{p}")) for p in preds]
        lines.append(
            f"| {sname} | {d['n']} | {fmt(d['wrong_rate'], True)} | " + " | ".join(cells) + " |"
        )

    lines += ["", "## V. Complementarity ladder (LODO LR AUROC)", ""]
    for k, v in results["ladder"].items():
        lines.append(f"- **{k}**: {fmt(v)}")

    lines += ["", "## VI. Residual AUROC (orthogonal component)", ""]
    for k, v in results["residual"].items():
        lines.append(f"- {k}: {fmt(v)}")

    lines += ["", "## VII. Quadrant wrong rates (T₀ × bd)", ""]
    lines.append("| Quadrant | n | wrong% | mean bd | mean T₀ |")
    lines.append("|----------|--:|-------:|--------:|--------:|")
    for q in results["quadrants"]:
        lines.append(
            f"| {q['quadrant']} | {q['n']} | {fmt(q['wrong_rate'], True)} | "
            f"{fmt(q['mean_bd'])} | {fmt(q['mean_T0'])} |"
        )

    lines += ["", "## VIII. Perturbation heterogeneity", ""]
    h = results["heterogeneity"]
    lines.append(f"- mean bd_text={fmt(h['mean_bd_text'])}  mean bd_weight={fmt(h['mean_bd_weight'])}")
    lines.append(f"- ρ(T₀, bd_text)={fmt(h['spearman_T0_bd_text'])}  ρ(T₀, bd_weight)={fmt(h['spearman_T0_bd_weight'])}")

    lines += ["", "## Key Spearman correlations", ""]
    for k, v in sorted(results["spearman"].items()):
        if any(x in k for x in ["T0~bd", "T0~F", "bd~F", "T0~dc", "bd~y", "T0~y"]):
            lines.append(f"- {k}: {fmt(v)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/panda-outputs"))
    ap.add_argument("--json-out", type=Path, default=ROOT / "paper/tables/mechanism_results.json")
    ap.add_argument("--md-out", type=Path, default=ROOT / "paper/tables/table_mechanism.md")
    args = ap.parse_args()

    print("Loading rows...", flush=True)
    rows = load_all(args.out_root)
    print(f"Loaded {len(rows)} samples", flush=True)

    results = {
        "n_samples": len(rows),
        "n_models": len(set(r["model"] for r in rows)),
        "predict_split": predict_split(rows),
        "enrichment": enrichment(rows),
        "flip_curve": run_level_flip_curve(rows),
        "conditional": conditional_auroc(rows),
        "residual": residual_auroc(rows),
        "ladder": ladder(rows),
        "spearman": spearman_block(rows),
        "heterogeneity": perturb_heterogeneity(rows),
        "quadrants": quadrant_wrong_rate(rows),
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    # strip run_panel for json size
    json.dump(results, args.json_out.open("w"), indent=2)
    write_markdown(results, args.md_out)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
