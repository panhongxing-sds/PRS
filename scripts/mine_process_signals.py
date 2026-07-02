#!/usr/bin/env python3
"""Mine process-level signals from cached PANDA outputs (N=8121 math samples)."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate_panda_v2 as agg  # noqa: E402

NEAR_CACHE = Path("/root/autodl-tmp/panda-outputs/.proc_near_answer_final_cache.pkl")
BD_CACHE = Path("/root/autodl-tmp/panda-outputs/.bd_variant_cache.pkl")
OUT_JSON = ROOT / "paper" / "analysis" / "process_signal_mining.json"
OUT_MD = ROOT / "paper" / "analysis" / "process_signal_mining.md"

# Answer-dispersion / outcome features — exclude from process-level mining
EXCLUDE_PREFIXES = (
    "text_answers", "weight_answers", "a0", "reference", "id", "dataset",
)
EXCLUDE_EXACT = {
    "bd", "y", "F_resp", "PANDA", "TW_ASE", "TW_ASE_H_norm", "TW_num_clusters",
    "T_ASE", "T_ASE_H_norm", "T_num_clusters", "W_ASE", "W_ASE_H_norm", "W_num_clusters",
    "D_ans", "D_reason", "answer_drift", "reasoning_drift",
    "is_correct", "is_correct_clean", "label_wrong", "label_wrong_clean", "label_drop",
    "relabeled", "n_rephrases", "n_weight_perturb", "lambda_a", "lambda_r",
    "cot_greedy_acc",
    # bd 代理 / 交互项（非独立过程信号）
    "s_base", "s_mode", "D_conf", "D_unc", "I_bd_ent", "I_mid_ent", "I_bd_x_prox_lin",
}
# 答案分散基线：保留用于对照，但不计入过程候选排名
ANSWER_DISPERSION = {
    "baseline_SE_H", "baseline_SE_H_norm", "baseline_SE_cluster_mode",
    "baseline_SE_num_clusters", "baseline_SE_status", "F_resp",
}
# Answer-path / cluster dispersion under perturbation
EXCLUDE_SUBSTR = (
    "num_unique_finals", "num_edit_dist", "num_clusters", "num_count_avg",
    "final_answer_support", "alternative_answer_mass", "base_answer_mass",
    "cluster_conditioned_flip", "cluster_token_js", "cluster_math_token_js",
    "last_eq_match_final", "final_in_reasoning_rate", "base_ans_topk_recall",
    "Da_final", "Da_margin", "Da_minority", "tw8_Da_",
)

# Known baselines: higher raw value => more confident (invert for wrong detection)
INVERT = {
    "baseline_DC_min": True, "baseline_DC_mean": True,
    "baseline_LL_nll": True, "baseline_LL_mean_logprob": False,
    "baseline_PE_mean": True, "baseline_PE_max": True, "baseline_PE_sum": True,
    "baseline_SAR": True, "baseline_SC_mean": True, "baseline_SC_min": True,
    "base_ans_mar_mean": True, "base_ans_logprob_min": False,
    "bpanda_full_logprob_min": False,
    "T_mar_sum_total": True, "T_mar_top10_sum_total": True,
    "W_mar_sum_total": True, "W_mar_top10_sum_total": True,
    "TW_mar_sum_total": True, "TW_mar_top10_sum_total": True,
    "T_ATU_margin_top10": True, "W_ATU_margin_top10": True, "TW_ATU_margin_top10": True,
}


def _fast_bd(a0: str, answers: list[str]) -> float:
    a0 = str(a0).strip()
    if not answers:
        return 0.0
    return sum(1 for a in answers if str(a).strip() != a0) / len(answers)


def is_process_field(key: str) -> bool:
    if key in EXCLUDE_EXACT:
        return False
    if key.startswith(EXCLUDE_PREFIXES):
        return False
    for s in EXCLUDE_SUBSTR:
        if s in key:
            return False
    if key.startswith(("Da_", "tw8_Da_")):
        return False
    if "ASE" in key and "baseline" not in key:
        return False
    if key.startswith("TW9_"):  # answer-cluster graph features
        return False
    return True


def is_process_candidate(key: str) -> bool:
    return is_process_field(key) and key not in ANSWER_DISPERSION


def load_rows(out_root: Path) -> list[dict]:
    near = pickle.loads(NEAR_CACHE.read_bytes())
    bdrows = pickle.loads(BD_CACHE.read_bytes())
    bd_by = {(r["model"], r["seed"], r["ds"], r["y"], r["D_base"]): r for r in bdrows}

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
                    ans = [str(x) for x in (s.get("text_answers") or []) + (s.get("weight_answers") or [])]
                    bd = round(_fast_bd(a0, ans), 4)
                    feats = {}
                    for k, v in s.items():
                        if not is_process_field(k):
                            continue
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            feats[k] = float(v)
                    pool[(mk, seed, ds, int(s["label_wrong_clean"]))][bd] = feats

    rows = []
    for r in near:
        b = bd_by.get((r["model"], r["seed"], r["ds"], r["y"], r["bd"]))
        if not b:
            continue
        row = {
            "model": r["model"], "seed": r["seed"], "ds": r["ds"], "y": int(r["y"]),
            "bd": float(r["bd"]),
            "T_ent_prox_lin": float(r["T_ent_prox_lin"]),
            "baseline_SE_H": float("nan"),
        }
        for k in ("T_ent_prox_exp_tau16", "T_ent_prox_exp_tau32", "T_ent_late20_minus_early50",
                  "T_ent_slope", "T_ent_late20_ratio", "T_ent_prox_text", "T_ent_prox_weight"):
            if k in r:
                row[k] = float(r[k])
        for k in ("I_bd_ent", "I_mid_ent", "D_conf", "D_unc", "s_base", "s_mode"):
            if k in b:
                row[k] = float(b[k])
        g = pool.get((r["model"], r["seed"], r["ds"], r["y"]), {})
        bd = round(r["bd"], 4)
        extra = g.get(bd)
        if extra is None and g:
            extra = g[min(g, key=lambda x: abs(x - bd))]
        if extra:
            row.update(extra)
            if "baseline_SE_H" in extra:
                row["baseline_SE_H"] = extra["baseline_SE_H"]
        rows.append(row)
    return rows


def orient_score(key: str, vals: np.ndarray) -> np.ndarray:
    v = vals.astype(float).copy()
    if INVERT.get(key, False):
        v = -v
    return v


def single_auroc(y: np.ndarray, s: np.ndarray) -> float:
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) < 10 or len(set(y)) < 2:
        return float("nan")
    try:
        a1 = roc_auc_score(y, s)
        a2 = roc_auc_score(y, -s)
        return float(max(a1, a2))
    except Exception:
        return float("nan")


def lodo_macro(rows: list[dict], cols: list[str], *, metric: str = "auroc") -> float:
    return agg.lodo_macro(rows, cols, agg.MATH_DS, metric=metric)


def eval_signal(rows: list[dict], key: str) -> dict:
    vals = np.array([float(r.get(key, float("nan"))) for r in rows])
    finite = np.isfinite(vals)
    if finite.sum() < 100:
        return {"key": key, "n_finite": int(finite.sum()), "skip": True}
    oriented = orient_score(key, vals)
    for r, v in zip(rows, oriented):
        r[f"__sig_{key}"] = v
    col = f"__sig_{key}"
    out = {
        "key": key,
        "n_finite": int(finite.sum()),
        "missing_rate": float(1 - finite.mean()),
        "invert": INVERT.get(key, False),
        "spearman_bd": float(spearmanr([r["bd"] for r in rows if np.isfinite(r.get(key, float("nan")))],
                                       [orient_score(key, np.array([r.get(key)]))[0]
                                        for r in rows if np.isfinite(r.get(key, float("nan")))])[0]),
        "lodo_auroc": lodo_macro(rows, [col]),
        "lodo_auprc": lodo_macro(rows, [col], metric="auprc"),
    }
    for name, fn in [
        ("all", lambda r: True),
        ("bd0", lambda r: r["bd"] == 0),
        ("bd0_wrong", lambda r: r["bd"] == 0 and r["y"] == 1),
        ("wrong_only", lambda r: r["y"] == 1),
        ("bd_gt0", lambda r: r["bd"] > 0),
    ]:
        sub = [r for r in rows if fn(r) and np.isfinite(r.get(key, float("nan")))]
        if len(sub) < 20:
            out[f"subset_{name}_n"] = len(sub)
            continue
        y = np.array([r["y"] for r in sub])
        s = orient_score(key, np.array([r[key] for r in sub]))
        out[f"subset_{name}_n"] = len(sub)
        out[f"subset_{name}_auroc"] = single_auroc(y, s)
    return out


def eval_fusion(rows: list[dict], proc_key: str) -> dict:
    sig_col = f"__sig_{proc_key}"
    if sig_col not in rows[0]:
        vals = np.array([float(r.get(proc_key, float("nan"))) for r in rows])
        oriented = orient_score(proc_key, vals)
        for r, v in zip(rows, oriented):
            r[sig_col] = v
    bd_col = "__bd_orient"
    for r in rows:
        r[bd_col] = r["bd"]
    return {
        "proc": proc_key,
        "lodo_auroc_bd_plus_proc": lodo_macro(rows, [bd_col, sig_col]),
        "lodo_auprc_bd_plus_proc": lodo_macro(rows, [bd_col, sig_col], metric="auprc"),
        "delta_auroc_vs_bd_only": lodo_macro(rows, [bd_col, sig_col]) - lodo_macro(rows, [bd_col]),
        "delta_auroc_vs_panda": lodo_macro(rows, [bd_col, sig_col]) - lodo_macro(rows, ["bd", "T_ent_prox_lin"]),
    }


def rank_candidates(results: list[dict]) -> list[dict]:
    proc = [r for r in results if not r.get("skip") and np.isfinite(r.get("lodo_auroc", float("nan")))]
    proc.sort(key=lambda x: (-x["lodo_auroc"], abs(x.get("spearman_bd", 0))))
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/panda-outputs"))
    args = ap.parse_args()

    rows = load_rows(args.out_root)
    print(f"Loaded N={len(rows)}")

    # Reference baselines
    refs = {}
    for name, cols in [
        ("bd", ["bd"]),
        ("hesitation_T_ent_prox_lin", ["T_ent_prox_lin"]),
        ("panda_bd_hes", ["bd", "T_ent_prox_lin"]),
    ]:
        refs[name] = {
            "lodo_auroc": lodo_macro(rows, cols),
            "lodo_auprc": lodo_macro(rows, cols, metric="auprc"),
            "n": len(rows),
        }
    se_rows = [r for r in rows if np.isfinite(r.get("baseline_SE_H", float("nan")))]
    if se_rows:
        refs["SE_baseline_SE_H"] = {
            "lodo_auroc": lodo_macro(se_rows, ["baseline_SE_H"]),
            "lodo_auprc": lodo_macro(se_rows, ["baseline_SE_H"], metric="auprc"),
            "n": len(se_rows),
        }
    dc_rows = [r for r in rows if np.isfinite(r.get("baseline_DC_min", float("nan")))]
    if dc_rows:
        dc_orient = []
        for r in dc_rows:
            r["__dc"] = -float(r["baseline_DC_min"])
            dc_orient.append(r)
        refs["DeepConf_baseline_DC_min"] = {
            "lodo_auroc": lodo_macro(dc_orient, ["__dc"]),
            "lodo_auprc": lodo_macro(dc_orient, ["__dc"], metric="auprc"),
            "n": len(dc_rows),
        }

    # Discover candidate keys
    keys = set()
    for r in rows:
        keys.update(k for k in r if is_process_field(k) and isinstance(r.get(k), (int, float)))
    keys.update(["T_ent_prox_lin", "T_ent_prox_exp_tau16", "T_ent_prox_exp_tau32",
                 "T_ent_late20_minus_early50", "T_ent_slope", "T_ent_late20_ratio",
                 "T_ent_prox_text", "T_ent_prox_weight", "I_bd_ent", "I_mid_ent"])
    keys.discard("bd")

    results = []
    refs_extra = {}
    for k in sorted(keys):
        res = eval_signal(rows, k)
        if res.get("skip"):
            continue
        if k in ANSWER_DISPERSION:
            refs_extra[k] = res
        elif is_process_candidate(k):
            results.append(res)

    ranked = rank_candidates(results)

    # Top fusion candidates: low |rho_bd| & strong bd0 subset
    fusion_pool = [
        r for r in ranked
        if abs(r.get("spearman_bd", 1)) < 0.35
        and r.get("subset_bd0_auroc", 0) >= 0.60
        and r["key"] != "T_ent_prox_lin"
    ]
    fusion_pool.sort(key=lambda x: (-x.get("subset_bd0_auroc", 0), abs(x.get("spearman_bd", 0))))
    fusions = [eval_fusion(rows, r["key"]) for r in fusion_pool[:15]]

    # Top 3 by composite: process-level, low bd corr, strong bd0
    composite = []
    for r in ranked:
        if r["key"] in ("T_ent_prox_lin",):
            continue
        bd0 = r.get("subset_bd0_auroc", float("nan"))
        if not np.isfinite(bd0):
            continue
        score = 0.6 * bd0 + 0.3 * r["lodo_auroc"] + 0.1 * (1 - min(abs(r["spearman_bd"]), 1))
        composite.append({**r, "composite_score": score})
    composite.sort(key=lambda x: -x["composite_score"])
    top3 = composite[:3]

    # hesitation rank among process candidates
    hes_rank = next((i + 1 for i, r in enumerate(ranked) if r["key"] == "T_ent_prox_lin"), None)
    beats_hes = [r for r in ranked if r["lodo_auroc"] > refs["hesitation_T_ent_prox_lin"]["lodo_auroc"]]

    payload = {
        "N": len(rows),
        "references": refs,
        "answer_dispersion_refs": refs_extra,
        "n_candidates_evaluated": len(ranked),
        "hesitation_rank_among_process": hes_rank,
        "n_process_beats_hesitation_full": len(beats_hes),
        "top20_by_lodo_auroc": ranked[:20],
        "top20_low_corr_strong_bd0": fusion_pool[:20],
        "top3_composite": top3,
        "fusion_top15": fusions,
        "beats_hesitation_full_top10": beats_hes[:10],
        "subset_counts": {
            "all": len(rows),
            "bd0": sum(1 for r in rows if r["bd"] == 0),
            "bd0_wrong": sum(1 for r in rows if r["bd"] == 0 and r["y"] == 1),
            "wrong_only": sum(1 for r in rows if r["y"] == 1),
            "bd_gt0": sum(1 for r in rows if r["bd"] > 0),
        },
        "spearman_bd_hes": float(spearmanr([r["bd"] for r in rows], [r["T_ent_prox_lin"] for r in rows])[0]),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))

    # Markdown report
    lines = [
        "# 过程级信号挖掘报告 (N=8121)",
        "",
        "## 参考基线 (macro LODO)",
        "",
        "| 方法 | AUROC | AUPRC | n |",
        "|------|------:|------:|--:|",
    ]
    for k, v in refs.items():
        lines.append(f"| {k} | {v['lodo_auroc']:.3f} | {v['lodo_auprc']:.3f} | {v['n']} |")

    lines += [
        "",
        f"Hesitation 在过程信号中排名: #{payload.get('hesitation_rank_among_process', '?')} / {payload['n_candidates_evaluated']}",
        f"全量 LODO AUROC 超过 hesitation 的过程信号数: {payload.get('n_process_beats_hesitation_full', 0)}",
        "",
        "## 子集规模",
        "",
    ]
    for k, v in payload["subset_counts"].items():
        lines.append(f"- {k}: n={v}")

    if payload.get("answer_dispersion_refs"):
        lines += ["", "## 答案分散对照 (非过程候选)", ""]
        for k, r in payload["answer_dispersion_refs"].items():
            lines.append(f"- `{k}`: LODO AUROC={r['lodo_auroc']:.3f}")

    lines += [
        "",
        f"Spearman(bd, hesitation) = {payload['spearman_bd_hes']:.3f}",
    ]

    lines += [
        "",
        "## Top-20 过程信号 (macro LODO AUROC)",
        "",
        "| 信号 | LODO AUROC | LODO AUPRC | ρ(bd) | bd=0 AUROC | bd0_wrong n |",
        "|------|----------:|----------:|------:|-----------:|------------:|",
    ]
    for r in ranked[:20]:
        bd0w_n = r.get("subset_bd0_wrong_n", 0)
        bd0w = r.get("subset_bd0_wrong_auroc", float("nan"))
        bd0w_s = f"{bd0w:.3f}" if np.isfinite(bd0w) else "—"
        lines.append(
            f"| `{r['key']}` | {r['lodo_auroc']:.3f} | {r['lodo_auprc']:.3f} | "
            f"{r['spearman_bd']:.3f} | {r.get('subset_bd0_auroc', float('nan')):.3f} | {bd0w_n} ({bd0w_s}) |"
        )

    lines += [
        "",
        "## Top-3 推荐候选 (低 bd 相关 + bd=0 强)",
        "",
    ]
    for i, r in enumerate(top3, 1):
        fus = next((f for f in fusions if f["proc"] == r["key"]), None)
        lines.append(f"### {i}. `{r['key']}`")
        lines.append(f"- LODO AUROC={r['lodo_auroc']:.3f}, bd=0 AUROC={r.get('subset_bd0_auroc', float('nan')):.3f}, ρ(bd)={r['spearman_bd']:.3f}")
        if fus:
            lines.append(f"- bd+signal LODO={fus['lodo_auroc_bd_plus_proc']:.3f} (Δ vs bd={fus['delta_auroc_vs_bd_only']:+.3f}, Δ vs PANDA={fus['delta_auroc_vs_panda']:+.3f})")
        lines.append("")

    lines += [
        "## bd 互补融合 (|ρ|<0.35, bd=0 AUROC≥0.60)",
        "",
        "| proc + bd | LODO AUROC | Δ vs bd | Δ vs PANDA |",
        "|-----------|----------:|--------:|-----------:|",
    ]
    for f in sorted(fusions, key=lambda x: -x["lodo_auroc_bd_plus_proc"])[:10]:
        lines.append(
            f"| `{f['proc']}` | {f['lodo_auroc_bd_plus_proc']:.3f} | "
            f"{f['delta_auroc_vs_bd_only']:+.3f} | {f['delta_auroc_vs_panda']:+.3f} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
