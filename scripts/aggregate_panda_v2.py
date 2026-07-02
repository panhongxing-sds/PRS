#!/usr/bin/env python3
"""Aggregate main-table metrics for PANDA (D_base/bd + T_ent_prox_lin, LODO LR).

PANDA = LODO-LR(bd, T_ent_prox_lin).  F_resp kept as diagnostic only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from panda.grading.math_grader import math_equal  # noqa: E402

MATH_DS = ("minerva", "math500", "gsm8k")
LOGIC_DS = ("leg_counting", "color_cube")
SEEDS = (41, 42, 43)

MODELS = {
    "qwen25_3b": "maintable_qwen25_3b",
    "llama32_1b": "maintable_llama32_1b",
    "llama31_8b": "maintable_llama31_8b",
    "qwen3_8b": "maintable_qwen3_8b",
}

DS_LABEL = {
    "math500": "MATH-500",
    "gsm8k": "GSM8K",
    "minerva": "Minerva",
    "leg_counting": "Leg Counting",
    "color_cube": "Color Cube",
}

BASELINE_METHODS = [
    ("SAR", "baseline_SAR", False),
    ("PE", "baseline_PE_mean", False),
    ("LL", "baseline_LL_nll", False),
    ("Self-Certainty", "baseline_SC_mean", True),
    ("DeepConf", "baseline_DC_min", True),
]

PANDA_METHODS = [
    ("PANDA (Ours)", "panda_full"),
    ("F_resp", "F"),
    ("bd", "bd"),
    ("T_ent_prox_lin", "T_ent_prox_lin"),
]

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


def _pre_answer(run: dict) -> list[dict]:
    trace = run.get("token_trace") or []
    a = (run.get("answer_span") or {}).get("start_token")
    return trace[: int(a)] if a is not None and int(a) > 0 else []


def prox_weighted_entropy(pre: list[dict]) -> float:
    n = len(pre)
    if n == 0:
        return float("nan")
    s, wsum = 0.0, 0.0
    for i, t in enumerate(pre):
        h = t.get("entropy")
        if h is None:
            continue
        w = (i + 1) / n
        s += w * float(h)
        wsum += w
    return s / wsum if wsum > 0 else float("nan")


def uniform_pre_answer_entropy(text_runs: list[dict], weight_runs: list[dict]) -> float:
    vals = []
    for run in text_runs + weight_runs:
        pre = _pre_answer(run)
        if not pre:
            continue
        hs = [float(t["entropy"]) for t in pre if t.get("entropy") is not None]
        if hs:
            vals.append(float(np.mean(hs)))
    return float(np.mean(vals)) if vals else float("nan")


def t_ent_prox_lin(text_runs: list[dict], weight_runs: list[dict]) -> float:
    vals = []
    for run in text_runs + weight_runs:
        v = prox_weighted_entropy(_pre_answer(run))
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def bd_vote(a0: str, pert: list[str]) -> float:
    if not a0 or not pert:
        return 0.0
    return float(np.mean([0.0 if eq(a0, x) else 1.0 for x in pert]))


def extract_features(raw_path: Path) -> dict | None:
    try:
        r = json.loads(raw_path.read_text())
    except Exception:
        return None
    sm = r.get("summary_metrics") or {}
    if sm.get("label_drop"):
        return None
    base = r.get("base_generation") or {}
    text = list(r.get("text_rephrase_runs") or [])
    weight = list(r.get("weight_perturb_runs") or [])
    a0 = str(sm.get("a0") or base.get("answer_normalized", "")).strip()
    text_ans = [str(g.get("answer_normalized", "")).strip() for g in text]
    weight_ans = [str(g.get("answer_normalized", "")).strip() for g in weight]
    pert = text_ans + weight_ans
    tw = sm.get("TW_ent_sum_total")
    return {
        "id": r.get("id"),
        "F": float(sm.get("F_resp") or sm.get("TW_ASE") or 0),
        "bd": bd_vote(a0, pert),
        "bd_text": bd_vote(a0, text_ans),
        "bd_weight": bd_vote(a0, weight_ans),
        "T_ent_prox_lin": t_ent_prox_lin(text, weight),
        "T_ent_prox_text": t_ent_prox_lin(text, []),
        "T_ent_prox_weight": t_ent_prox_lin([], weight),
        "T_ent_uniform": uniform_pre_answer_entropy(text, weight),
        "TW_ent_sum": float(tw) if tw is not None and math.isfinite(float(tw)) else float("nan"),
    }


def _parse_job(args: tuple) -> dict | None:
    rp, rid, seed, ds, model_key, summ_row = args
    feat = extract_features(Path(rp))
    if feat is None or feat["id"] != rid:
        return None
    s = summ_row
    return {
        "id": rid,
        "seed": seed,
        "ds": ds,
        "y": int(s["label_wrong_clean"]),
        "F": feat["F"],
        "bd": feat["bd"],
        "bd_text": feat["bd_text"],
        "bd_weight": feat["bd_weight"],
        "T_ent_prox_lin": feat["T_ent_prox_lin"],
        "T_ent_prox_text": feat["T_ent_prox_text"],
        "T_ent_prox_weight": feat["T_ent_prox_weight"],
        "T_ent_uniform": feat["T_ent_uniform"],
        "TW_ent_sum": feat["TW_ent_sum"],
        "model": model_key,
        **{k: s.get(k) for _, k, _ in BASELINE_METHODS},
    }


def load_summary(base: Path, seed: int, ds: str) -> dict[str, dict]:
    p = base / f"seed{seed}" / ds / "summary.jsonl"
    out = {}
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("label_drop", 0):
            continue
        lw = r.get("label_wrong_clean")
        if lw is None:
            continue
        out[r["id"]] = r
    return out


def load_model_rows(out_root: Path, model_key: str, datasets: tuple[str, ...], workers: int) -> list[dict]:
    base = out_root / MODELS[model_key]
    jobs = []
    for seed in SEEDS:
        for ds in datasets:
            summ = load_summary(base, seed, ds)
            raw_glob = base / f"seed{seed}" / ds / "raw_runs" / "*.json"
            for rp in glob.glob(str(raw_glob)):
                if "partial" in rp or "error" in rp:
                    continue
                rid = Path(rp).stem
                if rid in summ:
                    jobs.append((rp, rid, seed, ds, model_key, summ[rid]))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_parse_job, jobs, chunksize=32):
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


def fit_predict(Xtr, ytr, Xte):
    Xtr, Xte = _impute_fit(np.asarray(Xtr, float), np.asarray(Xte, float))
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=5000).fit((Xtr - mu) / sd, ytr)
    return clf.predict_proba((Xte - mu) / sd)[:, 1]


def lodo_scores(rows: list[dict], cols: list[str], datasets: tuple[str, ...]) -> dict[tuple[int, str], np.ndarray]:
    out: dict[tuple[int, str], np.ndarray] = {}
    for test_ds in datasets:
        for seed in SEEDS:
            te = [r for r in rows if r["ds"] == test_ds and r["seed"] == seed]
            if not te:
                continue
            tr = [r for r in rows if r["ds"] != test_ds]
            if len(set(r["y"] for r in tr)) < 2:
                continue
            Xtr = np.array([[r[c] for c in cols] for r in tr])
            ytr = np.array([r["y"] for r in tr])
            Xte = np.array([[r[c] for c in cols] for r in te])
            out[(seed, test_ds)] = fit_predict(Xtr, ytr, Xte)
    return out


def compute_metrics(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float] | None:
    if len(set(y)) < 2:
        return None
    m = np.isfinite(s)
    if m.sum() < 2 or len(set(y[m])) < 2:
        return None
    order = np.argsort(s[m])
    k = m.sum() // 2
    acc_star = float(np.mean(y[m][order[:k]] == 0))
    return (
        float(roc_auc_score(y[m], s[m])),
        float(average_precision_score(y[m], s[m])),
        acc_star,
    )


def _equal_weight_score(Xtr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    Xtr, Xte = _impute_fit(np.asarray(Xtr, float), np.asarray(Xte, float))
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-9
    return ((Xte - mu) / sd).sum(1)


def lodo_macro(
    rows: list[dict],
    cols: list[str],
    datasets: tuple[str, ...],
    *,
    metric: str = "auroc",
) -> float:
    vals = []
    for test_ds in datasets:
        scs = []
        for seed in SEEDS:
            te = [r for r in rows if r["ds"] == test_ds and r["seed"] == seed]
            tr = [r for r in rows if r["ds"] != test_ds]
            if len(te) < 10 or len(set(r["y"] for r in tr)) < 2:
                continue
            Xtr = np.array([[r[c] for c in cols] for r in tr], float)
            Xte = np.array([[r[c] for c in cols] for r in te], float)
            ytr = np.array([r["y"] for r in tr])
            yte = np.array([r["y"] for r in te])
            if len(set(ytr)) < 2:
                continue
            if metric == "fusion":
                pred = _equal_weight_score(Xtr, Xte)
            else:
                pred = fit_predict(Xtr, ytr, Xte)
            if metric == "auprc":
                scs.append(float(average_precision_score(yte, pred)))
            else:
                scs.append(float(roc_auc_score(yte, pred)))
        if scs:
            vals.append(float(np.mean(scs)))
    return float(np.mean(vals)) if vals else float("nan")


def compute_ablation_macro(rows: list[dict], datasets: tuple[str, ...]) -> dict[str, dict[str, float]]:
    full_cols = ["bd", "T_ent_prox_lin"]
    text_cols = ["bd_text", "T_ent_prox_text"]
    weight_cols = ["bd_weight", "T_ent_prox_weight"]
    variants = {
        "DH-Score (full)": full_cols,
        "w/o Dissent": ["T_ent_prox_lin"],
        "w/o Hesitation": ["bd"],
        "w/o Text Pert.": weight_cols,
        "w/o Weight Pert.": text_cols,
        "w/o Fusion": full_cols,
        "w/o Prox.": ["bd", "T_ent_uniform"],
    }
    out: dict[str, dict[str, float]] = {}
    for name, cols in variants.items():
        fusion = name == "w/o Fusion"
        out[name] = {
            "auroc": lodo_macro(rows, cols, datasets, metric="fusion" if fusion else "auroc"),
            "auprc": lodo_macro(rows, cols, datasets, metric="fusion" if fusion else "auprc"),
        }
    return out


def aggregate_model(rows: list[dict], model_key: str, datasets: tuple[str, ...]) -> dict:
    rows = [r for r in rows if r["model"] == model_key]
    if not rows:
        return {"rows": 0}

    panda_cols = {
        "panda_full": ["bd", "T_ent_prox_lin"],
        "panda_no_bd": ["T_ent_prox_lin"],
        "panda_no_T": ["bd"],
        "panda_F_bd_T": ["F", "bd", "T_ent_prox_lin"],
        "panda_F_bd": ["F", "bd"],
    }
    preds = {k: lodo_scores(rows, cols, datasets) for k, cols in panda_cols.items()}

    results = {"model": model_key, "n": len(rows), "datasets": {}, "ablation": {}}
    for ds in datasets:
        results["datasets"][ds] = {}
        for seed in SEEDS:
            sub = [r for r in rows if r["seed"] == seed and r["ds"] == ds]
            if not sub:
                continue
            y = np.array([r["y"] for r in sub])
            cell = {}

            for name, key, inv in BASELINE_METHODS:
                ss = []
                for r in sub:
                    v = r.get(key)
                    if v is None or not math.isfinite(float(v)):
                        continue
                    ss.append((r["y"], -float(v) if inv else float(v)))
                if len(ss) >= 2 and len(set(t[0] for t in ss)) == 2:
                    yy = np.array([t[0] for t in ss])
                    sc = np.array([t[1] for t in ss])
                    cell[name] = compute_metrics(yy, sc)

            for name, col in PANDA_METHODS:
                if col == "panda_full":
                    sc = preds["panda_full"].get((seed, ds))
                elif col in ("F", "bd", "T_ent_prox_lin"):
                    sc = np.array([r[col] for r in sub], float)
                else:
                    sc = None
                if sc is not None and len(sc) == len(y):
                    cell[name] = compute_metrics(y, sc)

            for ab_name, ab_key in [
                ("full", "panda_full"),
                ("no_bd", "panda_no_bd"),
                ("no_T", "panda_no_T"),
                ("legacy_F", "panda_F_bd_T"),
            ]:
                sc = preds[ab_key].get((seed, ds))
                if sc is not None:
                    cell[f"ab_{ab_name}"] = compute_metrics(y, sc)

            results["datasets"][ds][str(seed)] = cell
    return results


def fmt_ms(vals: list[float]) -> str:
    if not vals:
        return "—"
    return f"{np.mean(vals)*100:.2f} ± {np.std(vals)*100:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/panda-outputs"))
    ap.add_argument("--json-out", type=Path, default=ROOT / "paper" / "maintable" / "panda_v2_results.json")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    all_rows: list[dict] = []
    all_res = {}
    for mk in MODELS:
        print(f"Processing {mk}...")
        rows = load_model_rows(args.out_root, mk, MATH_DS, args.workers)
        all_rows.extend(rows)
        all_res[mk] = aggregate_model(rows, mk, MATH_DS)
        print(f"  n={all_res[mk]['n']}")

    bd_m = lodo_macro(all_rows, ["bd"], MATH_DS)
    panda_m = lodo_macro(all_rows, ["bd", "T_ent_prox_lin"], MATH_DS)
    ablation_macro = compute_ablation_macro(all_rows, MATH_DS)
    macro_summary = {
        "N": len(all_rows),
        "bd_only": bd_m,
        "panda_full": panda_m,
        "delta": panda_m - bd_m,
        "ablation_macro": ablation_macro,
    }

    payload = {"macro_summary": macro_summary, "models": all_res}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.json_out}")

    bd_m = lodo_macro(all_rows, ["bd"], MATH_DS)
    panda_m = lodo_macro(all_rows, ["bd", "T_ent_prox_lin"], MATH_DS)
    print(f"\n=== Cross-model macro LODO (N={len(all_rows)}) ===")
    print(f"  bd-only:       {bd_m:.3f}")
    print(f"  PANDA (bd+prox): {panda_m:.3f}  Δ={(panda_m-bd_m):+.3f}")

    print("\n=== Ablation macro (N=8121 pooled LODO) ===")
    for name, m in ablation_macro.items():
        print(f"  {name:22s}  AUROC={m['auroc']:.3f}  AUPRC={m['auprc']:.3f}")

    print("\n=== Table 3: PANDA AUROC (math, 3-seed mean±std) ===")
    print("| Model | MATH-500 | GSM8K | Minerva |")
    print("|-------|--:|--:|--:|")
    labels = {
        "qwen25_3b": "Qwen2.5-3B",
        "llama32_1b": "Llama-3.2-1B",
        "llama31_8b": "Llama-3.1-8B",
        "qwen3_8b": "Qwen3-8B",
    }
    for mk, lab in labels.items():
        r = all_res[mk]
        cols = []
        for ds in MATH_DS:
            vs = []
            for seed in SEEDS:
                c = r.get("datasets", {}).get(ds, {}).get(str(seed), {})
                m = c.get("PANDA (Ours)")
                if m:
                    vs.append(m[0])
            cols.append(fmt_ms(vs) if vs else "—")
        print(f"| {lab} | {' | '.join(cols)} |")

    print("\n=== Drop-one ablation Qwen2.5-3B (Δ AUROC pp vs full) ===")
    r = all_res.get("qwen25_3b", {})
    for ds in MATH_DS:
        full_v = [
            r.get("datasets", {}).get(ds, {}).get(str(seed), {}).get("PANDA (Ours)", [float("nan")])[0]
            for seed in SEEDS
        ]
        full_m = float(np.nanmean(full_v))
        print(f"  {DS_LABEL[ds]} full={full_m*100:.2f}%")
        for ab_name, ab_key in [("−D_base", "ab_no_bd"), ("−T", "ab_no_T"), ("+F legacy", "ab_legacy_F")]:
            vs = [
                r.get("datasets", {}).get(ds, {}).get(str(seed), {}).get(ab_key, [float("nan")])[0]
                for seed in SEEDS
            ]
            mu = float(np.nanmean(vs))
            print(f"    {ab_name}: {mu*100:.2f}%  Δ={(mu-full_m)*100:+.2f}pp")


if __name__ == "__main__":
    main()
