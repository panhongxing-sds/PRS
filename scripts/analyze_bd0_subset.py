#!/usr/bin/env python3
"""Deep-dive bd=0 subset analysis for PANDA (N=8121, cache-only)."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate_panda_v2 as agg  # noqa: E402
from mine_process_signals import (  # noqa: E402
    ANSWER_DISPERSION,
    INVERT,
    NEAR_CACHE,
    BD_CACHE,
    is_process_candidate,
    is_process_field,
    load_rows,
    orient_score,
)

OUT_JSON = ROOT / "paper" / "analysis" / "bd0_subset_analysis.json"
OUT_MD = ROOT / "paper" / "analysis" / "bd0_subset_analysis.md"

# Signals explicitly requested + top mining candidates
FOCUS_SIGNALS = [
    "bd",
    "T_ent_prox_lin",
    "T_ent_prox_exp_tau16",
    "T_ent_prox_exp_tau32",
    "T_ent_prox_text",
    "T_ent_prox_weight",
    "T_formula_skeleton_entropy",
    "W_operator_flip_rate",
    "T_earliest_cluster_branch_pos",
    "T_earliest_cluster_branch_ratio",
    "W_n_tokens_avg",
    "T_n_tokens_avg",
    "baseline_SE_H",
    "baseline_SE_H_norm",
    "baseline_DC_min",
    "baseline_U_Deg",
    "TW_ASE_H_norm",
]


def _fast_bd(a0: str, answers: list[str]) -> float:
    a0 = str(a0).strip()
    if not answers:
        return 0.0
    return sum(1 for a in answers if str(a).strip() != a0) / len(answers)


def attach_bd_variants(rows: list[dict]) -> None:
    bdrows = pickle.loads(BD_CACHE.read_bytes())
    bd_by = {(r["model"], r["seed"], r["ds"], r["y"], r["D_base"]): r for r in bdrows}
    for r in rows:
        b = bd_by.get((r["model"], r["seed"], r["ds"], r["y"], r["bd"]))
        if b:
            r["bd_text"] = float(b["D_text"])
            r["bd_weight"] = float(b["D_weight"])


def attach_summary_meta(rows: list[dict], out_root: Path) -> None:
    pool: dict[tuple, dict[float, dict]] = defaultdict(dict)
    for mk, folder in agg.MODELS.items():
        for seed in agg.SEEDS:
            for ds in agg.MATH_DS:
                p = out_root / folder / f"seed{seed}" / ds / "summary.jsonl"
                if not p.exists():
                    continue
                for ln in p.read_text().splitlines():
                    if not ln.strip():
                        continue
                    s = json.loads(ln)
                    if s.get("label_drop") or s.get("label_wrong_clean") is None:
                        continue
                    a0 = str(s.get("a0", ""))
                    ta = [str(x) for x in (s.get("text_answers") or [])]
                    wt = [str(x) for x in (s.get("weight_answers") or [])]
                    ans = ta + wt
                    bd = round(_fast_bd(a0, ans), 4)
                    ta_same = len(set(x.strip() for x in ta)) <= 1 if ta else True
                    tw0 = float(s.get("TW_ASE_H_norm") or 1) == 0.0
                    pool[(mk, seed, ds, int(s["label_wrong_clean"]))][bd] = {
                        "TW_ASE_H_norm": float(s.get("TW_ASE_H_norm") or float("nan")),
                        "text_agree": ta_same,
                        "n_text": len(ta),
                        "n_weight": len(wt),
                    }
    for r in rows:
        g = pool.get((r["model"], r["seed"], r["ds"], r["y"]), {})
        bd = round(r["bd"], 4)
        extra = g.get(bd)
        if extra is None and g:
            extra = g[min(g, key=lambda b: abs(b - bd))]
        if extra:
            r.update(extra)


def lodo_subset(rows: list[dict], cols: list[str], *, metric: str = "auroc") -> float:
    return agg.lodo_macro(rows, cols, agg.MATH_DS, metric=metric)


def eval_method_bd0(rows: list[dict], name: str, cols: list[str]) -> dict:
    sub = [r for r in rows if r["bd"] == 0]
    if len(sub) < 20 or len(set(r["y"] for r in sub)) < 2:
        return {"name": name, "n": len(sub), "skip": True}
    return {
        "name": name,
        "n": len(sub),
        "wrong_rate": float(np.mean([r["y"] for r in sub])),
        "lodo_auroc": lodo_subset(sub, cols),
        "lodo_auprc": lodo_subset(sub, cols, metric="auprc"),
        "cols": cols,
    }


def prepare_orient_col(rows: list[dict], key: str) -> str:
    col = f"__o_{key}"
    vals = np.array([float(r.get(key, float("nan"))) for r in rows])
    oriented = orient_score(key, vals)
    for r, v in zip(rows, oriented):
        r[col] = v
    return col


def single_auroc(y: np.ndarray, s: np.ndarray) -> float:
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) < 10 or len(set(y)) < 2:
        return float("nan")
    try:
        return float(max(roc_auc_score(y, s), roc_auc_score(y, -s)))
    except Exception:
        return float("nan")


def signal_separation(rows: list[dict], key: str) -> dict | None:
    sub = [r for r in rows if r["bd"] == 0 and np.isfinite(r.get(key, float("nan")))]
    if len(sub) < 20:
        return None
    y = np.array([r["y"] for r in sub])
    raw = np.array([float(r[key]) for r in sub])
    oriented = orient_score(key, raw)
    wrong = y == 1
    correct = y == 0
    if wrong.sum() < 2 or correct.sum() < 2:
        pb = float("nan")
    else:
        try:
            pb, _ = pointbiserialr(y, oriented)
            pb = float(abs(pb))
        except Exception:
            pb = float("nan")
    w_mean = float(np.mean(oriented[wrong])) if wrong.any() else float("nan")
    c_mean = float(np.mean(oriented[correct])) if correct.any() else float("nan")
    w_std = float(np.std(oriented[wrong])) if wrong.any() else float("nan")
    c_std = float(np.std(oriented[correct])) if correct.any() else float("nan")
    pooled_std = float(np.std(oriented))
    cohen_d = (w_mean - c_mean) / (pooled_std + 1e-9) if pooled_std > 1e-9 else float("nan")
    return {
        "key": key,
        "n": len(sub),
        "n_wrong": int(wrong.sum()),
        "n_correct": int(correct.sum()),
        "variance": float(np.var(oriented)),
        "std": float(np.std(oriented)),
        "auroc_oriented": single_auroc(y, oriented),
        "point_biserial_abs": pb,
        "mean_wrong": w_mean,
        "mean_correct": c_mean,
        "std_wrong": w_std,
        "std_correct": c_std,
        "mean_diff": w_mean - c_mean,
        "cohen_d": cohen_d,
        "spearman_bd": float(spearmanr([r["bd"] for r in rows if np.isfinite(r.get(key, float("nan")))],
                                       [orient_score(key, np.array([r.get(key)]))[0]
                                        for r in rows if np.isfinite(r.get(key, float("nan")))])[0])
        if len([r for r in rows if np.isfinite(r.get(key, float("nan")))]) > 10 else float("nan"),
    }


def lodo_conditional_panda_else_proc(
    rows: list[dict],
    proc_key: str,
    *,
    metric: str = "auroc",
) -> float:
    """LODO: bd>0 → logistic(bd, hes); bd=0 → logistic(proc) trained on bd=0 train only."""
    hes_col = prepare_orient_col(rows, "T_ent_prox_lin")
    proc_col = prepare_orient_col(rows, proc_key)
    vals = []
    for test_ds in agg.MATH_DS:
        scs = []
        for seed in agg.SEEDS:
            te = [r for r in rows if r["ds"] == test_ds and r["seed"] == seed]
            tr = [r for r in rows if r["ds"] != test_ds]
            if len(te) < 10 or len(set(r["y"] for r in tr)) < 2:
                continue
            tr0 = [r for r in tr if r["bd"] == 0]
            if len(tr0) < 20 or len(set(r["y"] for r in tr0)) < 2:
                continue
            Xp_tr = np.array([[r["bd"], r[hes_col]] for r in tr], float)
            Xp_te = np.array([[r["bd"], r[hes_col]] for r in te], float)
            X0_tr = np.array([[r[proc_col]] for r in tr0], float)
            ytr = np.array([r["y"] for r in tr])
            yte = np.array([r["y"] for r in te])
            ytr0 = np.array([r["y"] for r in tr0])
            if len(set(ytr)) < 2 or len(set(ytr0)) < 2:
                continue
            pred_panda = agg.fit_predict(Xp_tr, ytr, Xp_te)
            pred = pred_panda.copy()
            te0_idx = [i for i, r in enumerate(te) if r["bd"] == 0]
            if te0_idx:
                X0_te = np.array([[te[i][proc_col]] for i in te0_idx], float)
                pred0 = agg.fit_predict(X0_tr, ytr0, X0_te)
                for j, i in enumerate(te0_idx):
                    pred[i] = pred0[j]
            if metric == "auprc":
                scs.append(float(average_precision_score(yte, pred)))
            else:
                scs.append(float(roc_auc_score(yte, pred)))
        if scs:
            vals.append(float(np.mean(scs)))
    return float(np.mean(vals)) if vals else float("nan")


def collapse_narrative(rows: list[dict], out_root: Path) -> dict:
    """SC-proxy collapse: text-agree vs full bd=0, before/after weight perturb."""
    attach_bd_variants(rows)

    # Cache-level
    all_n = len(rows)
    bd0 = [r for r in rows if r["bd"] == 0]
    bd0_wrong = [r for r in bd0 if r["y"] == 1]
    bd_text0 = [r for r in rows if r.get("bd_text", 1) == 0]
    bd_text0_wrong = [r for r in bd_text0 if r["y"] == 1]
    bd_text0_only = [r for r in rows if r.get("bd_text", 1) == 0 and r.get("bd_weight", 0) > 0]
    bd_text0_only_wrong = [r for r in bd_text0_only if r["y"] == 1]

    # Summary scan for text_agree proxy
    text_agree_wrong = 0
    text_agree_all = 0
    tw0_wrong = 0
    for mk, folder in agg.MODELS.items():
        for seed in agg.SEEDS:
            for ds in agg.MATH_DS:
                p = out_root / folder / f"seed{seed}" / ds / "summary.jsonl"
                if not p.exists():
                    continue
                for ln in p.read_text().splitlines():
                    if not ln.strip():
                        continue
                    s = json.loads(ln)
                    if s.get("label_drop"):
                        continue
                    y = int(s.get("label_wrong_clean", 0))
                    a0 = str(s.get("a0", ""))
                    ta = [str(x) for x in (s.get("text_answers") or [])]
                    if not ta:
                        continue
                    ta_same = len(set(x.strip() for x in ta)) <= 1
                    if ta_same:
                        text_agree_all += 1
                        if y == 1:
                            text_agree_wrong += 1
                    if y == 1 and float(s.get("TW_ASE_H_norm") or 1) == 0:
                        tw0_wrong += 1

    # Wrong rescued by weight perturb (text agree but full bd>0)
    rescued = [r for r in rows if r.get("bd_text", 1) == 0 and r["bd"] > 0 and r["y"] == 1]
    rescued_total = [r for r in rows if r.get("bd_text", 1) == 0 and r["bd"] > 0]

    return {
        "all_n": all_n,
        "bd0_n": len(bd0),
        "bd0_wrong_n": len(bd0_wrong),
        "bd0_wrong_rate": len(bd0_wrong) / len(bd0) if bd0 else float("nan"),
        "bd_text0_n": len(bd_text0),
        "bd_text0_wrong_n": len(bd_text0_wrong),
        "bd_text0_wrong_rate": len(bd_text0_wrong) / len(bd_text0) if bd_text0 else float("nan"),
        "bd_text0_weight_break_n": len(bd_text0_only),
        "bd_text0_weight_break_wrong_n": len(bd_text0_only_wrong),
        "text_agree_all_n": text_agree_all,
        "text_agree_wrong_n": text_agree_wrong,
        "text_agree_wrong_rate": text_agree_wrong / text_agree_all if text_agree_all else float("nan"),
        "tw0_wrong_n": tw0_wrong,
        "weight_rescue_wrong_n": len(rescued),
        "weight_rescue_total_n": len(rescued_total),
        "weight_rescue_rate": len(rescued) / len(rescued_total) if rescued_total else float("nan"),
        "delta_wrong_n_text_to_full": len(bd_text0_wrong) - len(bd0_wrong),
        "hypothetical_text_only_remaining_wrong": len(bd_text0_wrong),
        "perturbation_reduction_wrong": len(bd_text0_wrong) - len(bd0_wrong),
        "perturbation_reduction_pct": (1 - len(bd0_wrong) / len(bd_text0_wrong)) * 100
        if bd_text0_wrong else float("nan"),
    }


def bd0_wrong_profile(rows: list[dict], separations: list[dict]) -> dict:
    bw = [r for r in rows if r["bd"] == 0 and r["y"] == 1]
    bc = [r for r in rows if r["bd"] == 0 and r["y"] == 0]
    by_model = Counter(r["model"] for r in bw)
    by_ds = Counter(r["ds"] for r in bw)
    # signals with meaningful variance within bd0_wrong
    var_rank = sorted(
        [s for s in separations if s and s["variance"] > 1e-12 and not s["key"].startswith("__")],
        key=lambda x: (-x.get("point_biserial_abs", 0) if np.isfinite(x.get("point_biserial_abs", float("nan")))
                     else -x.get("auroc_oriented", 0), -abs(x.get("cohen_d", 0))),
    )
    top_vary = []
    for s in var_rank[:15]:
        bw_vals = [float(r[s["key"]]) for r in bw if np.isfinite(r.get(s["key"], float("nan")))]
        if len(bw_vals) >= 5:
            top_vary.append({
                "key": s["key"],
                "n_finite_in_wrong": len(bw_vals),
                "std_in_wrong": float(np.std(bw_vals)),
                "range_in_wrong": [float(min(bw_vals)), float(max(bw_vals))],
                "point_biserial_abs": s["point_biserial_abs"],
                "auroc_bd0": s["auroc_oriented"],
            })
    return {
        "n": len(bw),
        "n_correct_bd0": len(bc),
        "by_model": dict(by_model),
        "by_dataset": dict(by_ds),
        "mean_hes_wrong": float(np.mean([r["T_ent_prox_lin"] for r in bw])),
        "mean_hes_correct": float(np.mean([r["T_ent_prox_lin"] for r in bc])),
        "tw0_rate_in_wrong": float(np.mean([
            float(r.get("TW_ASE_H_norm", 1)) == 0 for r in bw
            if np.isfinite(r.get("TW_ASE_H_norm", float("nan")))
        ])) if bw else float("nan"),
        "text_agree_rate_in_wrong": float(np.mean([r.get("text_agree", False) for r in bw])) if bw else float("nan"),
        "top_varying_signals": top_vary,
    }


def discover_signal_keys(rows: list[dict]) -> list[str]:
    keys = set(FOCUS_SIGNALS)
    for r in rows:
        keys.update(
            k for k in r
            if is_process_field(k) and not k.startswith("__")
            and isinstance(r.get(k), (int, float))
        )
    return sorted(k for k in keys if k not in ("y", "bd"))


def run_analysis(out_root: Path) -> dict:
    rows = load_rows(out_root)
    attach_bd_variants(rows)
    attach_summary_meta(rows, out_root)
    print(f"Loaded N={len(rows)}")

    bd0 = [r for r in rows if r["bd"] == 0]
    assert len(bd0) == 2960, f"expected 2960 bd=0, got {len(bd0)}"

    # --- LODO on bd=0 subset ---
    lodo_methods = [
        eval_method_bd0(rows, "bd_only", ["bd"]),
        eval_method_bd0(rows, "hesitation_T_ent_prox_lin", [prepare_orient_col(rows, "T_ent_prox_lin")]),
        eval_method_bd0(rows, "panda_full_bd_hes", ["bd", prepare_orient_col(rows, "T_ent_prox_lin")]),
        eval_method_bd0(rows, "panda_wo_hesitation", ["bd"]),
    ]

    for sig in FOCUS_SIGNALS:
        if sig == "bd":
            continue
        if not any(np.isfinite(r.get(sig, float("nan"))) for r in bd0):
            continue
        col = prepare_orient_col(rows, sig)
        lodo_methods.append(eval_method_bd0(rows, f"single_{sig}", [col]))

    # Fusions within bd=0
    skel_col = prepare_orient_col(rows, "T_formula_skeleton_entropy")
    hes_col = prepare_orient_col(rows, "T_ent_prox_lin")
    lodo_methods.append(eval_method_bd0(rows, "fusion_hes_skeleton", [hes_col, skel_col]))
    tau_col = prepare_orient_col(rows, "T_ent_prox_exp_tau16")
    lodo_methods.append(eval_method_bd0(rows, "fusion_hes_tau16", [hes_col, tau_col]))
    flip_col = prepare_orient_col(rows, "W_operator_flip_rate")
    lodo_methods.append(eval_method_bd0(rows, "fusion_hes_flip", [hes_col, flip_col]))

    # Full-dataset conditional mixes (separate LODO models per regime)
    lodo_full_cond = [
        {
            "name": "panda_full_reference",
            "n": len(rows),
            "lodo_auroc": agg.lodo_macro(rows, ["bd", hes_col], agg.MATH_DS),
            "lodo_auprc": agg.lodo_macro(rows, ["bd", hes_col], agg.MATH_DS, metric="auprc"),
            "scope": "full_N8121",
        },
        {
            "name": "cond_bd0_hes_else_panda",
            "n": len(rows),
            "lodo_auroc": lodo_conditional_panda_else_proc(rows, "T_ent_prox_lin"),
            "lodo_auprc": lodo_conditional_panda_else_proc(rows, "T_ent_prox_lin", metric="auprc"),
            "scope": "full_N8121",
            "note": "bd=0→hes (same as PANDA on bd=0 remainder)",
        },
    ]
    for proc, label in [
        ("T_formula_skeleton_entropy", "cond_bd0_skeleton_else_panda"),
        ("T_ent_prox_exp_tau16", "cond_bd0_tau16_else_panda"),
        ("W_operator_flip_rate", "cond_bd0_flip_else_panda"),
    ]:
        lodo_full_cond.append({
            "name": label,
            "n": len(rows),
            "lodo_auroc": lodo_conditional_panda_else_proc(rows, proc),
            "lodo_auprc": lodo_conditional_panda_else_proc(rows, proc, metric="auprc"),
            "scope": "full_N8121",
            "proc": proc,
        })

    lodo_bd0 = sorted(
        [m for m in lodo_methods if not m.get("skip")],
        key=lambda x: -x.get("lodo_auroc", 0),
    )

    # --- Separation analysis within bd=0 ---
    all_keys = discover_signal_keys(rows)
    separations = []
    for k in all_keys:
        if k.startswith("__") or k in ANSWER_DISPERSION or "ASE" in k:
            continue
        s = signal_separation(rows, k)
        if s and s["n"] >= 500:  # require broad coverage within bd=0
            separations.append(s)
    separations.sort(
        key=lambda x: (
            -x["point_biserial_abs"] if np.isfinite(x["point_biserial_abs"]) else 0,
            -x["auroc_oriented"] if np.isfinite(x["auroc_oriented"]) else 0,
            -abs(x["cohen_d"]) if np.isfinite(x["cohen_d"]) else 0,
        ),
    )

    # --- Collapse narrative ---
    collapse = collapse_narrative(rows, out_root)
    profile = bd0_wrong_profile(rows, separations)

    # Reference full-dataset from mining
    refs_full = {
        "panda_full": {"lodo_auroc": agg.lodo_macro(rows, ["bd", hes_col], agg.MATH_DS), "n": len(rows)},
        "bd_only": {"lodo_auroc": agg.lodo_macro(rows, ["bd"], agg.MATH_DS), "n": len(rows)},
        "hes_only": {"lodo_auroc": agg.lodo_macro(rows, [hes_col], agg.MATH_DS), "n": len(rows)},
    }

    # Best bd=0 single
    bd0_singles = [m for m in lodo_bd0 if m["name"].startswith("single_")]
    best_single = bd0_singles[0] if bd0_singles else None

    narrative = {
        "perturbation_reduces_collapse": collapse["perturbation_reduction_wrong"] > 0,
        "text_only_wrong_n": collapse["hypothetical_text_only_remaining_wrong"],
        "full_bd0_wrong_n": collapse["bd0_wrong_n"],
        "wrong_reduced_by_weight": collapse["perturbation_reduction_wrong"],
        "bd0_wrong_rate_pct": collapse["bd0_wrong_rate"] * 100,
        "text_agree_wrong_rate_pct": collapse["text_agree_wrong_rate"] * 100,
        "hes_increment_on_bd0": next(
            (m["lodo_auroc"] for m in lodo_bd0 if m["name"] == "hesitation_T_ent_prox_lin"), float("nan")
        ) - 0.5,
        "best_bd0_signal": best_single["name"].replace("single_", "") if best_single else None,
        "best_bd0_auroc": best_single["lodo_auroc"] if best_single else float("nan"),
        "skeleton_vs_hes_on_bd0": {
            "skeleton": next((m["lodo_auroc"] for m in lodo_bd0 if "skeleton" in m["name"] and "fusion" not in m["name"]), float("nan")),
            "hes": next((m["lodo_auroc"] for m in lodo_bd0 if m["name"] == "hesitation_T_ent_prox_lin"), float("nan")),
        },
    }

    return {
        "N": len(rows),
        "subset_counts": {
            "all": len(rows),
            "bd0": len(bd0),
            "bd0_wrong": sum(1 for r in rows if r["bd"] == 0 and r["y"] == 1),
            "bd0_correct": sum(1 for r in rows if r["bd"] == 0 and r["y"] == 0),
            "bd_gt0": sum(1 for r in rows if r["bd"] > 0),
        },
        "lodo_bd0_methods": lodo_bd0,
        "lodo_full_conditional": lodo_full_cond,
        "references_full_dataset": refs_full,
        "separation_bd0_top30": separations[:30],
        "separation_bd0_all_count": len(separations),
        "bd0_wrong_profile": profile,
        "collapse_perturbation_narrative": collapse,
        "narrative_support": narrative,
        "spearman_bd_hes": float(spearmanr([r["bd"] for r in rows], [r["T_ent_prox_lin"] for r in rows])[0]),
    }


def write_markdown(payload: dict) -> None:
    sc = payload["subset_counts"]
    collapse = payload["collapse_perturbation_narrative"]
    narr = payload["narrative_support"]
    prof = payload["bd0_wrong_profile"]

    lines = [
        "# bd=0 子集深度分析 (PANDA N=8121)",
        "",
        "> 由 `scripts/analyze_bd0_subset.py` 生成，基于 cache + summary，无 GPU。",
        "",
        "## 1. 子集规模",
        "",
        f"| 子集 | n | wrong | wrong_rate |",
        f"|------|--:|------:|-----------:|",
        f"| bd=0 | {sc['bd0']} | {sc['bd0_wrong']} | {sc['bd0_wrong']/sc['bd0']:.2%} |",
        f"| bd=0∧wrong | {sc['bd0_wrong']} | {sc['bd0_wrong']} | 100% (AUROC不可定义) |",
        f"| bd>0 | {sc['bd_gt0']} | — | — |",
        "",
        "## 2. bd=0 子集 LODO AUROC/AUPRC",
        "",
        "bd 在 bd=0 子集上恒为 0 → AUROC 退化为 0.5。",
        "",
        "| 方法 | LODO AUROC | LODO AUPRC |",
        "|------|----------:|----------:|",
    ]
    for m in payload["lodo_bd0_methods"]:
        auprc = m.get("lodo_auprc", float("nan"))
        auprc_s = f"{auprc:.3f}" if np.isfinite(auprc) else "—"
        lines.append(f"| {m['name']} | {m['lodo_auroc']:.3f} | {auprc_s} |")

    lines += [
        "",
        "## 3. 全量条件混合 (bd>0→PANDA, bd=0→过程信号)",
        "",
        "| 策略 | LODO AUROC | LODO AUPRC |",
        "|------|----------:|----------:|",
    ]
    for m in payload["lodo_full_conditional"]:
        lines.append(f"| {m['name']} | {m['lodo_auroc']:.3f} | {m.get('lodo_auprc', float('nan')):.3f} |")

    lines += [
        "",
        "## 4. Perturbation 已缓解 collapse",
        "",
        f"| 代理 | n | wrong | wrong_rate |",
        f"|------|--:|------:|-----------:|",
        f"| text-only agree (bd_text=0) | {collapse['bd_text0_n']} | {collapse['bd_text0_wrong_n']} | {collapse['bd_text0_wrong_rate']:.2%} |",
        f"| **full bd=0** (text+weight agree) | {collapse['bd0_n']} | {collapse['bd0_wrong_n']} | {collapse['bd0_wrong_rate']:.2%} |",
        f"| text_agree proxy (summary) | {collapse['text_agree_all_n']} | {collapse['text_agree_wrong_n']} | {collapse['text_agree_wrong_rate']:.2%} |",
        "",
        f"- 仅 text rephrase（无 weight）时剩余 wrong ≈ **{collapse['hypothetical_text_only_remaining_wrong']}**",
        f"- 加入 weight perturb 后 bd=0 wrong 降至 **{collapse['bd0_wrong_n']}**（减少 {collapse['perturbation_reduction_wrong']}，{collapse['perturbation_reduction_pct']:.1f}%）",
        f"- weight 打破 text 一致且仍 wrong 的样本：{collapse['weight_rescue_wrong_n']}/{collapse['weight_rescue_total_n']}",
        "",
        "### SC-proxy collapse 对照",
        "",
        "| 代理 (text 全同 ≈ SC collapse) | n | wrong_rate | 相对 bd=0 |",
        "|--------------------------------|--:|-----------:|----------:|",
        f"| text_agree (summary scan) | {collapse['text_agree_all_n']} | {collapse['text_agree_wrong_rate']:.2%} | 基线 collapse |",
        f"| bd_text=0 (仅 rephrase) | {collapse['bd_text0_n']} | {collapse['bd_text0_wrong_rate']:.2%} | −0.69pp |",
        f"| **bd=0 (rephrase+weight)** | {collapse['bd0_n']} | {collapse['bd0_wrong_rate']:.2%} | **−4.85pp vs text_agree** |",
        "",
        "→ rephrase+weight 将 collapse 子集 wrong_rate 从 ~8% 压到 3.2%；剩余 96 例为「高共识仍错」硬案例。",
        "→ SC DeepScaleR (K=64) AUROC(1−p_top)=0.774 vs PANDA full=0.904（不相交数据集，见 collapse_gain.json）。",
        "",
        "## 5. bd=0∧wrong (n=96) 特征",
        "",
        f"- 按模型: {prof['by_model']}",
        f"- 按数据集: {prof['by_dataset']}",
        f"- hesitation 均值 wrong={prof['mean_hes_wrong']:.4f} vs correct={prof['mean_hes_correct']:.4f}",
        f"- TW_ASE_H_norm=0 占比 (wrong): {prof['tw0_rate_in_wrong']:.1%}",
        "",
        "### 子集内有方差的过程信号 (Top)",
        "",
        "| 信号 | std(wrong) | |r_pb| | bd=0 AUROC |",
        "|------|----------:|-----:|-----------:|",
    ]
    for v in [x for x in prof["top_varying_signals"] if not x["key"].startswith("__")][:10]:
        pb = v["point_biserial_abs"]
        pb_s = f"{pb:.3f}" if np.isfinite(pb) else "—"
        au = v["auroc_bd0"]
        au_s = f"{au:.3f}" if np.isfinite(au) else "—"
        lines.append(f"| `{v['key']}` | {v['std_in_wrong']:.4f} | {pb_s} | {au_s} |")

    lines += [
        "",
        "## 6. bd=0 内 CORRECT vs WRONG 分离度 Top-15",
        "",
        "| 信号 | AUROC | |r_pb| | Cohen d | mean_diff | ρ(bd) |",
        "|------|------:|-----:|--------:|----------:|------:|",
    ]
    for s in payload["separation_bd0_top30"][:15]:
        pb = s["point_biserial_abs"]
        pb_s = f"{pb:.3f}" if np.isfinite(pb) else "—"
        lines.append(
            f"| `{s['key']}` | {s['auroc_oriented']:.3f} | {pb_s} | "
            f"{s['cohen_d']:.3f} | {s['mean_diff']:.4f} | {s['spearman_bd']:.3f} |"
        )

    lines += [
        "",
        "## 7. 叙事支持摘要",
        "",
        f"1. **Perturbation 已大幅削减 confident-wrong**：text-only wrong n={narr['text_only_wrong_n']} → bd=0 wrong n={narr['full_bd0_wrong_n']}（−{narr['wrong_reduced_by_weight']}）。",
        f"2. **bd=0 上 hesitation 是增量**：bd AUROC=0.5，hes LODO={narr['hes_increment_on_bd0']+0.5:.3f}（+{narr['hes_increment_on_bd0']:.3f} vs bd）。",
        f"3. **bd=0 最佳单信号**：`{narr['best_bd0_signal']}` AUROC={narr['best_bd0_auroc']:.3f}；skeleton={narr['skeleton_vs_hes_on_bd0']['skeleton']:.3f} vs hes={narr['skeleton_vs_hes_on_bd0']['hes']:.3f}。",
        f"4. bd=0∧wrong n=96 全为 wrong → 无法算 AUROC，但过程信号仍有方差可用于 rank / case study。",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, default=Path("/root/autodl-tmp/panda-outputs"))
    args = ap.parse_args()

    payload = run_analysis(args.outputs_root)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    write_markdown(payload)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
