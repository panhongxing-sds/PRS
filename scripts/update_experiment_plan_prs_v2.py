#!/usr/bin/env python3
"""Refresh EXPERIMENT_PLAN.md table sections from prs_v2_results.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "paper/maintable/prs_v2_results.json"
PLAN_PATH = ROOT / "paper/EXPERIMENT_PLAN.md"

DS_ORDER = ["math500", "gsm8k", "minerva"]
SEEDS = [41, 42, 43]
ROWS = [
    ("SAR", "SAR"),
    ("PE", "PE"),
    ("LL", "LL"),
    ("Self-Certainty", "Self-Certainty"),
    ("DeepConf", "DeepConf"),
    ("**PRS (Ours)**", "PRS (Ours)"),
    ("F_resp", "F_resp"),
    ("bd", "bd"),
    ("T_ent_prox_lin", "T_ent_prox_lin"),
]
DS_LAB = {"math500": "MATH-500", "gsm8k": "GSM8K", "minerva": "Minerva"}
LABELS = {
    "qwen25_3b": ("1a", "Qwen2.5-3B"),
    "llama32_1b": ("1b", "Llama-3.2-1B"),
    "llama31_8b": ("1c", "Llama-3.1-8B"),
    "qwen3_8b": ("1d", "Qwen3-8B"),
}


def get_m(r, ds, seed, key):
    c = r["datasets"].get(ds, {}).get(str(seed), {})
    return c.get(key)


def fmt_ms(vals):
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]*100:.2f}"
    return f"{np.mean(vals)*100:.2f} ± {np.std(vals)*100:.2f}"


def fmt_v(v):
    return f"{v*100:.2f}"


def table(model_key, data, seed=None):
    r = data[model_key]
    parts = []
    for ds in DS_ORDER:
        parts += [f"{DS_LAB[ds]} AUROC", "AUPRC", "ACC*"]
    lines = [
        "| Method | " + " | ".join(parts) + " |",
        "|--------|" + "|".join(["--:"] * len(parts)) + "|",
    ]
    for row_label, key in ROWS:
        cells = []
        for ds in DS_ORDER:
            if seed is not None:
                m = get_m(r, ds, seed, key)
                cells += [fmt_v(m[i]) if m else "—" for i in range(3)] if m else ["—", "—", "—"]
            else:
                for mi in range(3):
                    vs = []
                    for s in SEEDS:
                        m = get_m(r, ds, s, key)
                        if m:
                            vs.append(m[mi])
                    cells.append(fmt_ms(vs))
        lines.append("| " + row_label + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_section2(data: dict) -> str:
    out = [
        "## 2. 主表（E1：数学三数据集）",
        "",
        "每个数据集三列 **AUROC | AUPRC | ACC***。Baselines 不变；**PRS 及两分量（bd / T_ent_prox_lin）为 LODO OOF 结果**。",
        "",
        f"> **自动更新**：`update_experiment_plan_prs_v2.py`（数据源 `{JSON_PATH.name}`）",
        "",
    ]
    for mk, (tid, name) in LABELS.items():
        n = data[mk]["n"]
        out.append(f"### 表 {tid}：{name}")
        out.append("")
        out.append(
            f"> **数据状态**：PRS = LODO OOF LR(`D_base`/bd, `T_ent_prox_lin`)。n={n}。"
            f" 复现：`python PRS/scripts/aggregate_prs_v2.py`"
        )
        out.append("")
        out.append("#### 3-seed 平均 (strict label, LODO OOF)")
        out.append("")
        out.append(table(mk, data))
        out.append("")
        for seed in SEEDS:
            out.append(f"#### seed{seed}")
            out.append("")
            out.append(table(mk, data, seed))
            out.append("")
    return "\n".join(out)


def build_table3(data: dict) -> str:
    lines = [
        "### 表 3：PRS AUROC × 模型 × 数据集",
        "",
        "| Model | MATH-500 | GSM8K | Minerva |",
        "|-------|--:|--:|--:|",
    ]
    names = {
        "qwen25_3b": "**Qwen2.5-3B**",
        "llama32_1b": "Llama-3.2-1B",
        "llama31_8b": "Llama-3.1-8B",
        "qwen3_8b": "Qwen3-8B",
    }
    for mk, lab in names.items():
        cols = []
        for ds in DS_ORDER:
            vs = []
            for s in SEEDS:
                m = get_m(data[mk], ds, s, "PRS (Ours)")
                if m:
                    vs.append(m[0])
            cols.append(fmt_ms(vs))
        lines.append(f"| {lab} | {' | '.join(cols)} |")
    lines.append("")
    lines.append("> 3-seed mean±std，strict label，PRS = LODO(bd, T_ent_prox_lin)。")
    lines.append("")
    return "\n".join(lines)


def build_table4(data: dict) -> str:
    r = data["qwen25_3b"]
    full = {ds: np.mean([get_m(r, ds, s, "PRS (Ours)")[0] for s in SEEDS]) for ds in DS_ORDER}
    lines = [
        "### 表 4：组件消融（Qwen2.5-3B，LODO OOF）",
        "",
        "| Variant | MATH-500 AUROC | Δ | GSM8K AUROC | Δ | Minerva AUROC | Δ |",
        "|---------|--:|--:|--:|--:|--:|--:|",
    ]
    for vname, key, is_full in [
        ("**PRS (full)**", "PRS (Ours)", True),
        ("− D_base → T", "ab_no_bd", False),
        ("− T → D_base", "ab_no_T", False),
        ("+ F legacy (附录)", "ab_legacy_F", False),
    ]:
        cells = []
        for ds in DS_ORDER:
            vs = [get_m(r, ds, s, key)[0] for s in SEEDS]
            mu = np.mean(vs)
            if is_full:
                cells += [fmt_ms(vs), "0"]
            else:
                cells += [fmt_v(mu), f"{(mu - full[ds]) * 100:+.2f}"]
        lines.append("| " + vname + " | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### 表 4b：逐步消融（Qwen2.5-3B，LODO OOF AUROC）")
    lines.append("")
    lines.append("| 配置 | MATH-500 | GSM8K | Minerva |")
    lines.append("|------|--:|--:|--:|")
    for label, key in [
        ("D_base（raw AUROC）", "bd"),
        ("T_ent_prox_lin（raw）", "T_ent_prox_lin"),
        ("**PRS = LODO(D_base, T_ent_prox_lin)**", "PRS (Ours)"),
    ]:
        cols = []
        for ds in DS_ORDER:
            vs = [get_m(r, ds, s, key)[0] for s in SEEDS]
            cols.append(fmt_ms(vs))
        lines.append("| " + label + " | " + " | ".join(cols) + " |")
    lines.append("> **Drop-one 消融**：相对 full，Δ<0 表示去掉该分量后下降。")
    lines.append("")
    return "\n".join(lines)


def build_section0c(data: dict, macro_bd: float, macro_prs: float) -> str:
    n = sum(data[mk]["n"] for mk in data)
    return "\n".join(
        [
            "## 0c. 定稿验证摘要（自动更新）",
            "",
            f"> N={n}（4 模型 × 3 seeds × 3 数学集）。复现：`python PRS/scripts/aggregate_prs_v2.py`",
            "",
            "| 配置 | macro LODO AUROC | Δ vs bd-only |",
            "|------|:--:|:--:|",
            f"| bd-only | {macro_bd:.3f} | — |",
            f"| **PRS = bd + T_ent_prox_lin** | **{macro_prs:.3f}** | **{(macro_prs - macro_bd):+.3f}** |",
            "",
        ]
    )


def compute_macro_from_json(data: dict) -> tuple[float, float]:
    return float("nan"), float("nan")


def main():
    payload = json.loads(JSON_PATH.read_text())
    if "models" in payload:
        macro = payload.get("macro_summary", {})
        data = payload["models"]
    else:
        data = payload
        macro = {}

    plan = PLAN_PATH.read_text()

    sec2 = build_section2(data)
    t3 = build_table3(data)
    t4 = build_table4(data)

    if macro:
        sec0c = build_section0c(data, macro["bd_only"], macro["prs_full"])
        plan = re.sub(
            r"## 0c\. 定稿验证摘要.*?(\n---\n+\n## 1\. 实验总览)",
            sec0c + r"\1",
            plan,
            count=1,
            flags=re.DOTALL,
        )

    plan = re.sub(
        r"## 2\. 主表（E1：数学三数据集）.*?(\n## 3\. 主表（E2：逻辑）)",
        sec2 + r"\1",
        plan,
        count=1,
        flags=re.DOTALL,
    )
    plan = re.sub(
        r"### 表 3：PRS AUROC × 模型 × 数据集.*?(\n---\n\n## 5\. 组件消融)",
        t3 + r"\1",
        plan,
        count=1,
        flags=re.DOTALL,
    )
    plan = re.sub(
        r"### 表 4：组件消融.*?(?=\n## 6\. 扰动预算|\n---\n\n## 6\.)",
        t4,
        plan,
        count=1,
        flags=re.DOTALL,
    )

    PLAN_PATH.write_text(plan)
    print(f"Updated {PLAN_PATH}")


if __name__ == "__main__":
    main()
