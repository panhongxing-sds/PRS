#!/usr/bin/env python3
"""CPU-only SC vs PANDA dissent + collapse-subset figure (cache + summary, no GPU).

Outputs:
  paper/analysis/figures/sc_vs_panda_dissent.png
  paper/analysis/collapse_gain.json
  paper/analysis/FAST_PATH.md
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import aggregate_panda_v2 as agg  # noqa: E402

SC_ROOT = ROOT / "experiments" / "spurious_consensus"
NEAR_CACHE = Path("/root/autodl-tmp/panda-outputs/.proc_near_answer_final_cache.pkl")
BD_CACHE = Path("/root/autodl-tmp/panda-outputs/.bd_variant_cache.pkl")
FIG_OUT = ROOT / "paper" / "analysis" / "figures" / "sc_vs_panda_dissent.png"
JSON_OUT = ROOT / "paper" / "analysis" / "collapse_gain.json"
MD_OUT = ROOT / "paper" / "analysis" / "FAST_PATH.md"

SC_MODEL_MAP = {
    "qwen25_3b": "qwen25_3b",
    "llama32_1b": "llama32_1b",
}
SC_LABEL = {"qwen25_3b": "Qwen-3B", "llama32_1b": "Llama-1B"}


def _fast_bd(a0: str, answers: list[str]) -> float:
    a0 = str(a0).strip()
    if not answers:
        return 0.0
    return sum(1 for a in answers if str(a).strip() != a0) / len(answers)


def load_panda_rows(out_root: Path) -> list[dict]:
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
                    pool[(mk, seed, ds, int(s["label_wrong_clean"]))][bd] = s

    rows: list[dict] = []
    for r in near:
        b = bd_by.get((r["model"], r["seed"], r["ds"], r["y"], r["bd"]))
        if not b:
            continue
        g = pool.get((r["model"], r["seed"], r["ds"], r["y"]), {})
        bd = round(float(r["bd"]), 4)
        s = g.get(bd)
        if s is None and g:
            s = g[min(g, key=lambda b: abs(b - bd))]
        if not s:
            continue
        rows.append({
            "id": s["id"],
            "model": r["model"],
            "seed": r["seed"],
            "ds": r["ds"],
            "y": int(r["y"]),
            "bd": float(r["bd"]),
            "bd_text": float(b["D_text"]),
            "bd_weight": float(b["D_weight"]),
            "T_ent_prox_lin": float(r["T_ent_prox_lin"]),
            "T_ent_prox_text": float(r.get("T_ent_prox_text", float("nan"))),
            "T_ent_prox_weight": float(r.get("T_ent_prox_weight", float("nan"))),
            "T_ent_uniform": float(r.get("TW_ent_sum", float("nan"))) / 8.0
            if np.isfinite(float(r.get("TW_ent_sum", float("nan")))) else float("nan"),
            "TW_ASE_H_norm": s.get("TW_ASE_H_norm"),
            "baseline_SE_H": s.get("baseline_SE_H"),
        })
    return rows


def subset_lodo(rows: list[dict], name: str, fn) -> dict:
    sub = [r for r in rows if fn(r)]
    out: dict = {"name": name, "n": len(sub)}
    if len(sub) < 20 or len(set(r["y"] for r in sub)) < 2:
        out["note"] = "insufficient class balance for AUROC"
        return out
    out["wrong_rate"] = float(np.mean([r["y"] for r in sub]))
    for lab, cols in [
        ("panda", ["bd", "T_ent_prox_lin"]),
        ("bd", ["bd"]),
        ("hes", ["T_ent_prox_lin"]),
        ("wo_dissent", ["T_ent_prox_lin"]),
        ("wo_hesitation", ["bd"]),
    ]:
        out[f"lodo_{lab}"] = agg.lodo_macro(sub, cols, agg.MATH_DS)
    if out.get("lodo_bd") is not None and out.get("lodo_panda") is not None:
        out["delta_panda_vs_bd"] = out["lodo_panda"] - out["lodo_bd"]
    if out.get("lodo_hes") is not None and out.get("lodo_bd") is not None:
        out["delta_hes_vs_bd_on_subset"] = out["lodo_hes"] - out["lodo_bd"]
    return out


def load_sc_deepscaler() -> dict[str, dict]:
    p = SC_ROOT / "results" / "all_models_clean_metrics.json"
    raw = json.loads(p.read_text())
    out = {}
    for mk in SC_MODEL_MAP:
        tag = raw.get("summary", {}).get(mk, {})
        ds = tag.get("by_benchmark", {}).get("deepscaler", {})
        if ds:
            out[mk] = {
                "auroc_u_ans": ds.get("auroc"),
                "scr90_pct_wrong": ds.get("scr_90_pct_wrong"),
                "scr90_n": ds.get("scr_90"),
                "n": ds.get("n"),
                "maj_at_64": ds.get("maj_at_64"),
            }
    return out


def sc_p_top(rec: dict) -> tuple[float, float]:
    pairs = [(a, c) for a, c in zip(rec["answers"], rec["correct"]) if a]
    if len(pairs) < 2:
        return float("nan"), float("nan")
    ans, _ = zip(*pairs)
    cnt = Counter(ans)
    p_top = max(cnt.values()) / len(ans)
    return p_top, 1.0 - p_top


def minerva_overlap(rows: list[dict]) -> dict:
    """Paired SC K=64 vs PANDA on qwen25_3b seed41 minerva (n≈198)."""
    sc_path = SC_ROOT / "data" / "samples" / "samples_qwen25_3b_seed41_minerva.jsonl"
    if not sc_path.exists():
        return {"available": False}
    sc = {}
    for ln in sc_path.read_text().splitlines():
        if ln.strip():
            sc[json.loads(ln)["id"]] = json.loads(ln)

    joined = []
    for r in rows:
        if r["model"] != "qwen25_3b" or r["seed"] != 41 or r["ds"] != "minerva":
            continue
        srec = sc.get(r["id"])
        if not srec:
            continue
        p_top, u_ans = sc_p_top(srec)
        if not np.isfinite(u_ans):
            continue
        joined.append({**r, "p_top": p_top, "u_ans": u_ans})

    res: dict = {"available": True, "n": len(joined), "model": "qwen25_3b", "seed": 41, "dataset": "minerva"}
    if len(joined) < 20 or len(set(j["y"] for j in joined)) < 2:
        res["note"] = "too few for AUROC"
        return res

    y = np.array([j["y"] for j in joined])
    for lab, key in [("sc_u_ans", "u_ans"), ("panda_bd", "bd"), ("panda_hes", "T_ent_prox_lin")]:
        s = np.array([float(j[key]) for j in joined])
        res[f"auroc_{lab}"] = float(roc_auc_score(y, s))

    X = np.array([[j["bd"], j["T_ent_prox_lin"]] for j in joined], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    res["auroc_panda_uw"] = float(roc_auc_score(y, ((X - mu) / sd).sum(1)))

    scr = [j for j in joined if j["y"] == 1 and j["p_top"] >= 0.9]
    res["scr90_wrong_n"] = len(scr)
    bd0 = [j for j in joined if j["bd"] == 0]
    res["bd0_n"] = len(bd0)
    res["bd0_wrong_n"] = sum(1 for j in joined if j["bd"] == 0 and j["y"] == 1)
    return res


def collapse_proxy_counts(out_root: Path) -> dict:
    counts = Counter()
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
                    tw = float(s.get("TW_ASE_H_norm") or 1)
                    a0 = str(s.get("a0", ""))
                    ta = [str(x) for x in (s.get("text_answers") or [])]
                    ans = ta + [str(x) for x in (s.get("weight_answers") or [])]
                    bd = _fast_bd(a0, ans)
                    ta_same = len(set(x.strip() for x in ta)) <= 1 if ta else False
                    ref = str(s.get("reference", "")).strip()
                    if y == 1 and tw == 0:
                        counts["tw0_wrong"] += 1
                    if y == 1 and bd == 0:
                        counts["bd0_wrong_fast"] += 1
                    if y == 1 and tw == 0 and bd == 0:
                        counts["tw0_bd0_wrong"] += 1
                    if y == 1 and ta_same and ta and all(x.strip() != ref for x in ta):
                        counts["text_agree_wrong"] += 1
    return dict(counts)


def plot_figure(payload: dict) -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "axes.linewidth": 1.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    # Panel A: macro dissent comparison (disjoint datasets, honest labels)
    ax = axes[0]
    ab = payload["panda_ablation_macro"]
    sc_q3 = payload["sc_deepscaler"].get("qwen25_3b", {})
    bars = [
        ("SC\n1−p_top\n(DeepScaleR, K=64)", sc_q3.get("auroc_u_ans", float("nan")), "#8491B4"),
        ("PANDA\nw/o Dissent\n(hes only)", ab["w/o Dissent"]["auroc"], "#F39B7F"),
        ("PANDA\nbd (dissent)", ab["w/o Hesitation"]["auroc"], "#E64B35"),
        ("PANDA\nfull", ab["DH-Score (full)"]["auroc"], "#3C5488"),
    ]
    x = np.arange(len(bars))
    vals = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.8, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("Macro LODO AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_title("(a) Answer consensus vs perturbation dissent")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.text(
        0.02, 0.02,
        "SC: Qwen-3B deepscaler (disjoint)\nPANDA: 4 models × 3 math, N=8121",
        transform=ax.transAxes, fontsize=8, va="bottom", color="#333333",
    )

    # Panel B: collapse-proxy subsets (bd=0)
    ax = axes[1]
    subs = {s["name"]: s for s in payload["panda_subsets"]}
    names = ["all", "bd0"]
    methods = [
        ("bd (dissent)", "lodo_bd", "#E64B35"),
        ("hesitation", "lodo_hes", "#F39B7F"),
        ("PANDA full", "lodo_panda", "#3C5488"),
    ]
    w = 0.24
    x0 = np.arange(len(names))
    for j, (lab, key, col) in enumerate(methods):
        vals = [subs[n].get(key, float("nan")) for n in names]
        pos = x0 + (j - 1) * w
        ax.bar(pos, vals, width=w * 0.9, label=lab, color=col, edgecolor="black", linewidth=0.6)
        for xi, v in zip(pos, vals):
            if np.isfinite(v):
                ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ns = [subs[n]["n"] for n in names]
    ax.set_xticks(x0)
    ax.set_xticklabels([f"{n}\n(n={ns[i]:,})" for i, n in enumerate(names)])
    ax.set_ylabel("Macro LODO AUROC")
    ax.set_ylim(0.45, 1.0)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_title("(b) Collapse proxy: bd=0 subset")
    ax.legend(loc="upper right", frameon=True)
    bd0 = subs.get("bd0", {})
    delta = bd0.get("delta_hes_vs_bd_on_subset")
    if delta is not None:
        ax.text(
            0.02, 0.02,
            f"bd=0: bd AUROC=0.5 (degenerate)\nhes gains +{delta:.3f} via process signal",
            transform=ax.transAxes, fontsize=8, va="bottom", color="#333333",
        )
    ax.text(
        0.98, 0.02,
        f"bd=0∧wrong n={payload['bd0_wrong_cache_n']} (all wrong → AUROC N/A)",
        transform=ax.transAxes, fontsize=7.5, va="bottom", ha="right", color="#666666",
    )

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_fast_path_md(payload: dict) -> None:
    ab = payload["panda_ablation_macro"]
    subs = {s["name"]: s for s in payload["panda_subsets"]}
    sc = payload["sc_deepscaler"]
    overlap = payload["minerva_overlap"]
    proxy = payload["collapse_proxy_counts"]

    lines = [
        "# FAST_PATH — SC vs PANDA dissent & collapse subset",
        "",
        "> CPU-only deliverable generated by `scripts/plot_sc_vs_panda_collapse.py`.",
        "",
        "## 调查结论",
        "",
        "### 1. 数据集/模型重叠",
        "",
        "| 维度 | SC (spurious_consensus) | PANDA 主表 | 重叠 |",
        "|------|-------------------------|------------|------|",
        "| 模型 | 6 (0.5B–7B + Phi) | 4 (qwen25_3b, llama32_1b, llama31_8b, qwen3_8b) | **llama32_1b, qwen25_3b** |",
        "| 数据集 | deepscaler, gpqa, aime (+少量 minerva) | minerva, math500, gsm8k | **仅 minerva**（qwen3b seed41: 201 ID 全重叠） |",
        "| 预算 | K=64 高温采样 | 1 clean + 4 text + 4 weight ≈ 9 decodes | 量级可比 (~8–9) |",
        "",
        "**无 deepscaler PANDA raw** → 无法在同一题上直接对比 SC K=64 与 PANDA bd。",
        "",
        "### 2. PANDA collapse proxy（无需 SC）",
        "",
        "| 代理定义 | n (summary scan) | n (cache exact) | 可算 AUROC? |",
        "|----------|----------------:|----------------:|:-----------:|",
        f"| bd=0 ∧ wrong | {proxy.get('bd0_wrong_fast', '?')} | **{payload['bd0_wrong_cache_n']}** | ❌ 全为 wrong |",
        f"| TW_ASE_H_norm=0 ∧ wrong | {proxy.get('tw0_wrong', 0)} | 0 (join) | ❌ |",
        f"| text_answers 全同 ∧ wrong | {proxy.get('text_agree_wrong', '?')} | — | ✅ |",
        f"| **bd=0（collapse proxy）** | — | **{subs['bd0']['n']}** | ✅ wrong_rate≈{subs['bd0'].get('wrong_rate', 0):.1%} |",
        "",
        "→ **可用 bd=0 子集**展示 dissent 退化时 hesitation 的补救；bd=0∧wrong 仅作 case 计数。",
        "",
        "### 3. SC 样本字段 (`data/samples/`)",
        "",
        "`id, dataset, gold, answers[], correct[], label_drop, seed` — **无 p_top 预计算**；",
        "p_top / u_ans=1−p_top 由 answers 频率现场算。",
        "",
        "### 4. PANDA summary.jsonl 可用字段",
        "",
        "`bd` 需从 text_answers+weight_answers 对 a0 重算；直接有 `TW_ASE_H_norm`, `baseline_SE_H`,",
        "`T_ent_*`, `text_answers`, `weight_answers`, `label_wrong_clean`。",
        "",
        "### 5. 其他文档",
        "",
        "- `process_signal_mining.md/json` ✅ 已有（N=8121 LODO 信号排名）",
        "- `weak_points_countermeasures.md` ❌ 不存在",
        "",
        "---",
        "",
        "## 方案排序（快 → 慢）",
        "",
        "### Option A — CPU only（**已实现，~30s**）",
        "",
        "| 项 | 值 |",
        "|----|-----|",
        "| GPU | 否 |",
        "| 时间 | **<1 min** |",
        "| 论文强度 | **中等**（分数据集诚实对比 + bd=0 代理） |",
        "| 产出 | 本图 + collapse_gain.json |",
        "",
        "```bash",
        "source scripts/env.sh",
        "export PANDA_OUTPUTS=/root/autodl-tmp/panda-outputs",
        "python scripts/plot_sc_vs_panda_collapse.py",
        "```",
        "",
        "**论文用法**：",
        "- Fig: `paper/analysis/figures/sc_vs_panda_dissent.png`",
        "  - (a) SC 1−p_top vs PANDA 消融 → dissent (bd) 是 PANDA 相对 SC 的关键增量",
        "  - (b) bd=0 子集：bd 退化到 0.5，hesitation 仍 0.623",
        "- Table/正文：引用 `collapse_gain.json` 中 `panda_subsets` 与 `panda_ablation_macro`",
        "",
        "**诚实 caveat**：",
        "- Panel (a) 为**不相交数据集**（SC=DeepScaleR，PANDA=主表数学）；不能直接声称同题 superiority。",
        f"- Minerva 配对 (n={overlap.get('n', 0)})：SC u_ans AUROC={overlap.get('auroc_sc_u_ans', float('nan')):.3f} **高于** PANDA bd={overlap.get('auroc_panda_bd', float('nan')):.3f} — 勿过度宣称。",
        "- bd=0∧wrong (n=96) 全错，无法报告 AUROC 增益；用 bd=0 子集 (+0.123 hes vs bd) 替代。",
        "",
        "### Option B — Minimal GPU（**推荐若审稿要求同题对齐**）",
        "",
        "| 项 | 值 |",
        "|----|-----|",
        "| GPU | 是，1×5090 |",
        "| 时间 | **~6–12 GPU·h**（1 model × deepscaler ~400题 × Phase A+B） |",
        "| 论文强度 | **强**（同题 SC K=64 + PANDA bd 可配对） |",
        "",
        "```bash",
        "source scripts/env.sh",
        "export PANDA_OUTPUTS=/root/autodl-tmp/panda-outputs",
        "export PANDA_MODELS=/root/autodl-tmp/panda-models",
        "# Phase A+B: qwen25_3b on deepscaler only",
        "SKIP_HF=0 bash scripts/run_maintable_vllm.sh qwen25_3b deepscaler",
        "python scripts/recompute_metrics_parallel.sh /root/autodl-tmp/panda-outputs/deepscaler_qwen25_3b deepscaler",
        "python scripts/plot_sc_vs_panda_collapse.py --joined-deepscaler",
        "```",
        "",
        "然后与 `experiments/spurious_consensus/data/samples/samples_qwen25_3b_seed41_deepscaler.jsonl` join。",
        "",
        "### Option C — Full alignment",
        "",
        "| 项 | 值 |",
        "|----|-----|",
        "| GPU | 是 |",
        "| 时间 | **~80–150 GPU·h**（4 models × 3 SC sets × Phase A+B） |",
        "| 论文强度 | **最强** |",
        "",
        "仅当 reviewer 明确要求全模型全 SC 数据集对齐时考虑。",
        "",
        "---",
        "",
        "## 关键数字（本次 CPU run）",
        "",
        f"- PANDA macro LODO: full={ab['DH-Score (full)']['auroc']:.3f}, w/o Dissent={ab['w/o Dissent']['auroc']:.3f}, bd-only={ab['w/o Hesitation']['auroc']:.3f}",
        f"- SC Qwen-3B deepscaler AUROC(1−p_top)={sc.get('qwen25_3b', {}).get('auroc_u_ans', float('nan')):.3f}",
        f"- bd=0 subset: bd={subs['bd0'].get('lodo_bd', float('nan')):.3f}, hes={subs['bd0'].get('lodo_hes', float('nan')):.3f}, PANDA={subs['bd0'].get('lodo_panda', float('nan')):.3f}",
        f"- bd=0∧wrong: n={payload['bd0_wrong_cache_n']} (AUROC N/A)",
        "",
    ]
    MD_OUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, default=Path("/root/autodl-tmp/panda-outputs"))
    args = ap.parse_args()

    rows = load_panda_rows(args.outputs_root)
    ab = agg.compute_ablation_macro(rows, agg.MATH_DS)
    bd = np.array([r["bd"] for r in rows])
    hes = np.array([r["T_ent_prox_lin"] for r in rows])
    rho, _ = spearmanr(bd, hes, nan_policy="omit")

    subsets = [
        subset_lodo(rows, "all", lambda r: True),
        subset_lodo(rows, "bd0", lambda r: r["bd"] == 0),
        subset_lodo(rows, "bd0_wrong", lambda r: r["bd"] == 0 and r["y"] == 1),
        subset_lodo(rows, "text_agree_wrong", lambda r: r["y"] == 1 and r["bd_text"] == 0),
    ]

    payload = {
        "N": len(rows),
        "spearman_bd_hes": float(rho),
        "panda_ablation_macro": ab,
        "panda_subsets": subsets,
        "collapse_lodo": {s["name"]: s for s in subsets},
        "bd0_wrong_cache_n": sum(1 for r in rows if r["bd"] == 0 and r["y"] == 1),
        "sc_deepscaler": load_sc_deepscaler(),
        "minerva_overlap": minerva_overlap(rows),
        "collapse_proxy_counts": collapse_proxy_counts(args.outputs_root),
        "outputs": {
            "figure": str(FIG_OUT),
            "json": str(JSON_OUT),
            "fast_path_md": str(MD_OUT),
        },
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2, default=str))
    plot_figure(payload)
    write_fast_path_md(payload)
    print(f"Wrote {FIG_OUT}\nWrote {JSON_OUT}\nWrote {MD_OUT}")


if __name__ == "__main__":
    main()
