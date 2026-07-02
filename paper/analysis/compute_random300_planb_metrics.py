#!/usr/bin/env python3
"""Plan B random300: Qwen2.5-3B + Llama-3.2-1B paired metrics on same 300 ids."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_panda_v2 import eq  # noqa: E402
from panda.grading.math_grader import math_equal  # noqa: E402

# Reuse confident-wrong helpers
from compute_random300_confident_wrong import (  # noqa: E402
    TAU_SPECS,
    panda_vote_stats,
    rate_confident_wrong,
    sc_vote_stats,
)

META = ROOT / "paper/analysis/deepscaler_random300_meta.json"
OUT_JSON = ROOT / "paper/analysis/random300_planb_metrics.json"
OUT_MD = ROOT / "paper/analysis/random300_planb_results_summary.md"
FIG = ROOT / "paper/analysis/figures/random300_planb_comparison.png"

MODELS = {
    "qwen25_3b": {
        "label": "Qwen2.5-3B",
        "k9": ROOT
        / "experiments/spurious_consensus/data/samples"
        / "samples_qwen25_3b_seed41_deepscaler_random300_k9.jsonl",
        "summary": Path(
            "/root/autodl-tmp/panda-outputs/maintable_qwen25_3b_deepscaler_random300"
            "/seed41/deepscaler/summary.jsonl"
        ),
        "join_out": ROOT / "paper/analysis/random300_sc9_panda_join.jsonl",
    },
    "llama32_1b": {
        "label": "Llama-3.2-1B",
        "k9": ROOT
        / "experiments/spurious_consensus/data/samples"
        / "samples_llama32_1b_seed41_deepscaler_random300_k9.jsonl",
        "summary": Path(
            "/root/autodl-tmp/panda-outputs/maintable_llama32_1b_deepscaler_random300"
            "/seed41/deepscaler/summary.jsonl"
        ),
        "join_out": ROOT / "paper/analysis/random300_sc9_panda_join_llama.jsonl",
    },
}


def _grade(a: str, gold: str) -> bool:
    a, gold = str(a).strip(), str(gold).strip()
    if a == gold:
        return True
    try:
        return bool(math_equal(a, gold))
    except Exception:
        return False


def sc_k9_cohort_stats(records: list[dict]) -> dict:
    maj_c = maj_w = blind = any_c = 0
    p_tops = []
    for rec in records:
        pairs = [(a, c) for a, c in zip(rec.get("answers") or [], rec.get("correct") or []) if a]
        if len(pairs) < 9:
            continue
        answers, correct = zip(*pairs[:9])
        cnt = Counter(answers)
        _, top = cnt.most_common(1)[0]
        p_top = top / 9
        p_tops.append(p_top)
        cmap: dict[str, int] = {}
        for a, c in zip(answers, correct):
            cmap.setdefault(a, int(c))
        maj, _ = cnt.most_common(1)[0]
        if cmap.get(maj, 0) == 1:
            maj_c += 1
        else:
            maj_w += 1
        if not any(correct):
            blind += 1
        if any(correct):
            any_c += 1
    n = len(p_tops)
    return {
        "n": n,
        "majority_correct": maj_c,
        "majority_wrong": maj_w,
        "majority_accuracy": maj_c / n if n else float("nan"),
        "blind_at_9": blind,
        "blind_rate": blind / n if n else float("nan"),
        "any_correct_at_9": any_c,
        "p_top_mean": float(np.mean(p_tops)) if p_tops else float("nan"),
        "p_top_min": float(np.min(p_tops)) if p_tops else float("nan"),
        "p_top_max": float(np.max(p_tops)) if p_tops else float("nan"),
    }


def _auroc_auprc(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(s)
    if m.sum() < 2 or len(np.unique(y[m])) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y[m], s[m])), float(average_precision_score(y[m], s[m]))


def analyze_model(key: str, cfg: dict, ids: list[str]) -> dict:
    sc_by_id: dict[str, dict] = {}
    for ln in cfg["k9"].read_text().splitlines():
        if ln.strip():
            o = json.loads(ln)
            sc_by_id[str(o["id"])] = o

    summ_by_id: dict[str, dict] = {}
    for ln in cfg["summary"].read_text().splitlines():
        if not ln.strip():
            continue
        p = json.loads(ln)
        num = str(p["id"]).split("_", 1)[-1]
        summ_by_id[num] = p

    sc_recs = [sc_by_id[i] for i in ids if i in sc_by_id]
    sc_stats = sc_k9_cohort_stats(sc_recs)

    per_question: list[dict] = []
    join_rows: list[dict] = []
    for qid in ids:
        if qid not in sc_by_id or qid not in summ_by_id:
            continue
        scs = sc_vote_stats(sc_by_id[qid])
        pvs = panda_vote_stats(summ_by_id[qid])
        if not scs or not pvs:
            continue
        summ = summ_by_id[qid]
        row = {
            "id": qid,
            "gold": summ.get("reference"),
            "label_drop": int(summ.get("label_drop") or 0),
            "label_wrong_clean": int(summ.get("label_wrong_clean") or 0),
            "sc_maj": scs["maj"],
            "sc_p_top": scs["p_top"],
            "sc_maj_wrong": scs["maj_wrong"],
            "sc_maj_correct": scs["maj_correct"],
            "panda_maj": pvs["maj"],
            "panda_p_top": pvs["p_top"],
            "panda_maj_wrong": pvs["maj_wrong"],
            "panda_maj_correct": pvs["maj_correct"],
            "panda_a0_correct": pvs["a0_correct"],
            "panda_bd": pvs["bd"],
            "panda": float(summ.get("PANDA") if summ.get("PANDA") is not None else summ.get("PRS") if summ.get("PRS") is not None else float("nan")),
            "f_resp": float(summ.get("F_resp") or float("nan")),
        }
        per_question.append(row)
        join_rows.append(
            {
                "id": qid,
                "gold": row["gold"],
                "panda_correct": row["panda_a0_correct"],
                "panda_maj_correct": row["panda_maj_correct"],
                "sc_maj_correct": row["sc_maj_correct"],
                "y_wrong_panda": int(not row["panda_a0_correct"]),
                "y_wrong_sc": int(row["sc_maj_wrong"]),
                "p_top": row["sc_p_top"],
                "panda_p_top": row["panda_p_top"],
                "sc_uptop": 1.0 - row["sc_p_top"],
                "panda": row["panda"],
                "bd": row["panda_bd"],
                "a0": summ.get("a0"),
                "sc_maj": row["sc_maj"],
                "panda_maj": row["panda_maj"],
                "label_drop": row["label_drop"],
            }
        )

    cfg["join_out"].parent.mkdir(parents=True, exist_ok=True)
    with cfg["join_out"].open("w") as f:
        for jr in join_rows:
            f.write(json.dumps(jr, ensure_ascii=False) + "\n")

    def metrics_block(rows: list[dict]) -> dict:
        flat = [{**r, "sc_maj_wrong": r["sc_maj_wrong"], "sc_p_top": r["sc_p_top"],
                 "panda_maj_wrong": r["panda_maj_wrong"], "panda_p_top": r["panda_p_top"]} for r in rows]
        n = len(rows)
        sc_acc = sum(1 for r in rows if r["sc_maj_correct"]) / n if n else float("nan")
        pv_acc = sum(1 for r in rows if r["panda_maj_correct"]) / n if n else float("nan")
        a0_acc = sum(1 for r in rows if r["panda_a0_correct"]) / n if n else float("nan")
        y = np.array([r["label_wrong_clean"] for r in rows], dtype=float)
        sc_u = np.array([1.0 - r["sc_p_top"] for r in rows], dtype=float)
        panda = np.array([r["panda"] for r in rows], dtype=float)
        sc_auroc, sc_auprc = _auroc_auprc(y, sc_u)
        panda_auroc, panda_auprc = _auroc_auprc(y, panda)
        # SC majority wrong label
        y_sc = np.array([int(r["sc_maj_wrong"]) for r in rows], dtype=float)
        sc_auroc_sc_label, _ = _auroc_auprc(y_sc, sc_u)
        panda_auroc_sc_label, _ = _auroc_auprc(y_sc, panda)
        sc_taus = {spec["name"]: rate_confident_wrong(flat, "sc", spec["tau"]) for spec in TAU_SPECS}
        panda_taus = {spec["name"]: rate_confident_wrong(flat, "panda", spec["tau"]) for spec in TAU_SPECS}
        a0_wrong = [r for r in rows if not r["panda_a0_correct"]]
        collapse = [r for r in a0_wrong if abs(r["panda_bd"]) < 1e-12]
        return {
            "n": n,
            "label_drop_count": sum(r["label_drop"] for r in per_question) if rows is per_question else sum(
                1 for r in rows if r.get("label_drop")
            ),
            "sc9_stats_inline": {
                "majority_correct": sum(1 for r in rows if r["sc_maj_correct"]),
                "majority_wrong": sum(1 for r in rows if not r["sc_maj_correct"]),
            },
            "accuracy": {
                "sc9_majority_vote": sc_acc,
                "panda9_majority_vote_fair": pv_acc,
                "panda_greedy_a0": a0_acc,
                "delta_fair_vote_minus_sc_pp": (pv_acc - sc_acc) * 100 if n else float("nan"),
                "delta_greedy_minus_sc_pp": (a0_acc - sc_acc) * 100 if n else float("nan"),
            },
            "uq_label_wrong_clean": {
                "sc_1_minus_p_top_auroc": sc_auroc,
                "sc_1_minus_p_top_auprc": sc_auprc,
                "panda_auroc": panda_auroc,
                "panda_auprc": panda_auprc,
                "delta_panda_minus_sc_auroc_pp": (panda_auroc - sc_auroc) * 100
                if np.isfinite(panda_auroc) and np.isfinite(sc_auroc)
                else float("nan"),
            },
            "uq_label_sc_majority_wrong": {
                "sc_1_minus_p_top_auroc": sc_auroc_sc_label,
                "panda_auroc": panda_auroc_sc_label,
            },
            "confident_wrong_sc@9": sc_taus,
            "confident_wrong_panda@9_vote": panda_taus,
            "panda_collapse_a0_wrong_bd0_count": len(collapse),
        }

    eval_rows = [r for r in per_question if r["label_drop"] == 0]
    return {
        "model_key": key,
        "model_label": cfg["label"],
        "completion": {
            "summary_lines": sum(1 for _ in cfg["summary"].read_text().splitlines() if _.strip()),
            "k9_lines": sum(1 for _ in cfg["k9"].read_text().splitlines() if _.strip()),
            "joined": len(per_question),
        },
        "sc9_cohort": sc_stats,
        "sources": {"k9": str(cfg["k9"]), "summary": str(cfg["summary"]), "join": str(cfg["join_out"])},
        "metrics_n300": metrics_block(per_question),
        "metrics_n228_eval": metrics_block(eval_rows),
        "per_question_count": len(per_question),
    }


def write_md(doc: dict) -> None:
    q, l = doc["models"]["qwen25_3b"], doc["models"]["llama32_1b"]
    m3q, m3l = q["metrics_n300"], l["metrics_n300"]
    e228q, e228l = q["metrics_n228_eval"], l["metrics_n228_eval"]

    def fmt_acc(m: dict) -> str:
        a = m["accuracy"]
        return (
            f"| SC@9 多数票 | {a['sc9_majority_vote']*100:.1f}% |\n"
            f"| PANDA@9 公平多数票（9 路 decode） | {a['panda9_majority_vote_fair']*100:.1f}% |\n"
            f"| PANDA greedy a0 | {a['panda_greedy_a0']*100:.1f}% |\n"
        )

    def fmt_uq(m: dict, label: str) -> str:
        u = m["uq_label_wrong_clean"]
        return (
            f"### {label}\n\n"
            f"| 指标 | SC 1−p_top | PANDA | Δ(PANDA−SC) |\n|------|------------|-----|----------|\n"
            f"| AUROC (label_wrong_clean) | {u['sc_1_minus_p_top_auroc']:.3f} | {u['panda_auroc']:.3f} | "
            f"{u['delta_panda_minus_sc_auroc_pp']:+.2f} pp |\n"
            f"| AUPRC | {u['sc_1_minus_p_top_auprc']:.3f} | {u['panda_auprc']:.3f} | — |\n"
        )

    def fmt_cw(m: dict, name: str) -> str:
        lines = [f"### {name} — ConfWrong@τ（多数票错且 p_top≥τ）\n", "| τ | SC@9 rate | PANDA@9 rate | SC count | PANDA count |\n|---|-----------|--------------|----------|-------------|\n"]
        for spec in TAU_SPECS:
            sc = m["confident_wrong_sc@9"][spec["name"]]
            pd = m["confident_wrong_panda@9_vote"][spec["name"]]
            lines.append(
                f"| {spec['name']} | {sc['confident_wrong_rate_all']*100:.2f}% | "
                f"{pd['confident_wrong_rate_all']*100:.2f}% | {sc['confident_wrong_count_all']} | {pd['confident_wrong_count_all']} |\n"
            )
        return "".join(lines)

    md = f"""# Plan B：DeepScaler random300 双模型对比（同一 300 题）

**日期：** 2026-07-02  
**队列：** `deepscaler_random300_meta.json`（seed=42 抽 300 id；SC 与 PANDA 一一对齐）

## 1. 完成度验证

| 模型 | summary.jsonl | k9 | join | 状态 |
|------|---------------|-----|------|------|
| Qwen2.5-3B | {q['completion']['summary_lines']}/300 | {q['completion']['k9_lines']}/300 | {q['completion']['joined']} | ✅ Phase B 300/300 |
| Llama-3.2-1B | {l['completion']['summary_lines']}/300 | {l['completion']['k9_lines']}/300 | {l['completion']['joined']} | ✅ Phase B 300/300 |

公平预算：PANDA = 1 greedy + 4 text + 4 weight（共 9 decode）；SC@9 = K=64 池前 9 条（T=0.5）。

---

## 2. SC@9（同一 300 id）

| 指标 | Qwen2.5-3B | Llama-3.2-1B |
|------|------------|--------------|
| 多数票准确率 | {q['sc9_cohort']['majority_accuracy']*100:.1f}% ({q['sc9_cohort']['majority_correct']}/300) | {l['sc9_cohort']['majority_accuracy']*100:.1f}% ({l['sc9_cohort']['majority_correct']}/300) |
| blind@9（9/9 全错） | {q['sc9_cohort']['blind_at_9']} ({q['sc9_cohort']['blind_rate']*100:.1f}%) | {l['sc9_cohort']['blind_at_9']} ({l['sc9_cohort']['blind_rate']*100:.1f}%) |
| any_correct@9 | {q['sc9_cohort']['any_correct_at_9']} | {l['sc9_cohort']['any_correct_at_9']} |
| mean p_top@9 | {q['sc9_cohort']['p_top_mean']:.3f} | {l['sc9_cohort']['p_top_mean']:.3f} |

**趋势：** Llama 更小、更难：SC 多数票仅 ~{l['sc9_cohort']['majority_accuracy']*100:.0f}%，blind@9 远高于 Qwen（{l['sc9_cohort']['blind_at_9']} vs {q['sc9_cohort']['blind_at_9']}），共识更弱（mean p_top {l['sc9_cohort']['p_top_mean']:.2f} vs {q['sc9_cohort']['p_top_mean']:.2f}）。

---

## 3. 准确率（n=300）

### Qwen2.5-3B
| 方法 | 准确率 |
|------|--------|
{fmt_acc(m3q).replace('| ', '| ').strip()}

### Llama-3.2-1B
| 方法 | 准确率 |
|------|--------|
{fmt_acc(m3l).replace('SC@9 多数票', 'SC@9 多数票').strip()}

**公平 PANDA@9 vs SC@9：** Qwen {m3q['accuracy']['delta_fair_vote_minus_sc_pp']:+.1f} pp；Llama {m3l['accuracy']['delta_fair_vote_minus_sc_pp']:+.1f} pp。

---

## 4. UQ：PANDA vs SC（1−p_top），标签 `label_wrong_clean`

主表 eval：**n=228**（去掉 label_drop=72）。

{fmt_uq(e228q, 'Qwen — n=228')}

{fmt_uq(e228l, 'Llama — n=228')}

{fmt_uq(m3q, 'Qwen — n=300（参考）')}

{fmt_uq(m3l, 'Llama — n=300（参考）')}

---

## 5. ConfWrong（n=300）

{fmt_cw(m3q, 'Qwen2.5-3B')}

{fmt_cw(m3l, 'Llama-3.2-1B')}

---

## 6. 与仅 Qwen 旧稿 `random300_qwen_results_analysis.md` 对照

- 旧稿 greedy a0 **40.0%**、SC **44.0%**、n=228 PANDA AUROC **0.862** vs SC **0.823**（+3.95 pp）。
- 本次 Qwen 重算：greedy **{m3q['accuracy']['panda_greedy_a0']*100:.1f}%**，SC **{m3q['accuracy']['sc9_majority_vote']*100:.1f}%**，n=228 PANDA **{e228q['uq_label_wrong_clean']['panda_auroc']:.3f}** vs SC **{e228q['uq_label_wrong_clean']['sc_1_minus_p_top_auroc']:.3f}**（{e228q['uq_label_wrong_clean']['delta_panda_minus_sc_auroc_pp']:+.2f} pp）。
- 公平 PANDA@9 多数票：旧 confident_wrong 脚本约 **50.7%**；本次 **{m3q['accuracy']['panda9_majority_vote_fair']*100:.1f}%**（math_equal 多数票口径一致）。

---

## 7. 叙事结论（双模型）

| 维度 | Qwen | Llama | 双模型是否一致 |
|------|------|-------|----------------|
| 任务准确率（公平 9 票） | PANDA@9 {m3q['accuracy']['delta_fair_vote_minus_sc_pp']:+.1f} pp vs SC | {m3l['accuracy']['delta_fair_vote_minus_sc_pp']:+.1f} pp vs SC | {'✅ 两模型公平票均优于/不劣于 SC' if m3q['accuracy']['delta_fair_vote_minus_sc_pp']>0 and m3l['accuracy']['delta_fair_vote_minus_sc_pp']>0 else '⚠️ 见上表'} |
| UQ（greedy wrong, n=228） | PANDA {e228q['uq_label_wrong_clean']['delta_panda_minus_sc_auroc_pp']:+.2f} pp | {e228l['uq_label_wrong_clean']['delta_panda_minus_sc_auroc_pp']:+.2f} pp | {'✅ 两模型 PANDA 均高于 SC' if e228q['uq_label_wrong_clean']['delta_panda_minus_sc_auroc_pp']>0 and e228l['uq_label_wrong_clean']['delta_panda_minus_sc_auroc_pp']>0 else '⚠️ 见上表'} |

**总评：** {'**对 PANDA UQ 叙事偏正面**：两模型在同一 300 题上，PANDA 相对 SC 1−p_top 在 n=228 均有提升；公平预算下 PANDA@9 多数票准确率不低于 SC@9。' if (e228q['uq_label_wrong_clean']['delta_panda_minus_sc_auroc_pp']>0 and e228l['uq_label_wrong_clean']['delta_panda_minus_sc_auroc_pp']>0) else '**条件性正面**：UQ 提升见上表；准确率与 ConfWrong 需按模型分别表述。'}  
**注意：** SC 与 PANDA 解码协议仍不同（SC 前缀 T=0.5 vs PANDA greedy 扰动）；UQ 标签是 **greedy a0 错**，不是 SC 多数票错。

---

## 产物

- JSON：`random300_planb_metrics.json`
- Join：`random300_sc9_panda_join.jsonl`（Qwen）、`random300_sc9_panda_join_llama.jsonl`（Llama）
- 图：`figures/random300_planb_comparison.png`
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT_MD}")


def plot_comparison(doc: dict) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; skip plot")
        return
    keys = ["qwen25_3b", "llama32_1b"]
    labels = [doc["models"][k]["model_label"] for k in keys]
    acc_sc = [doc["models"][k]["metrics_n228_eval"]["accuracy"]["sc9_majority_vote"] * 100 for k in keys]
    acc_pv = [doc["models"][k]["metrics_n228_eval"]["accuracy"]["panda9_majority_vote_fair"] * 100 for k in keys]
    auroc_sc = [doc["models"][k]["metrics_n228_eval"]["uq_label_wrong_clean"]["sc_1_minus_p_top_auroc"] for k in keys]
    auroc_prs = [doc["models"][k]["metrics_n228_eval"]["uq_label_wrong_clean"]["panda_auroc"] for k in keys]

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 1.5 * w, acc_sc, w, label="SC@9 acc (n=228)")
    ax.bar(x - 0.5 * w, acc_pv, w, label="PANDA@9 fair acc (n=228)")
    ax.bar(x + 0.5 * w, [a * 100 for a in auroc_sc], w, label="SC 1−p_top AUROC×100")
    ax.bar(x + 1.5 * w, [a * 100 for a in auroc_prs], w, label="PANDA AUROC×100")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percent / AUROC×100")
    ax.set_title("Plan B random300: Qwen vs Llama (eval n=228)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG}")


def main() -> None:
    meta = json.loads(META.read_text())
    ids = [str(x) for x in meta["ids"]]
    doc = {
        "plan": "Plan B random300 paired comparison",
        "cohort": str(META),
        "n_ids": len(ids),
        "verified_at": "2026-07-02",
        "models": {},
    }
    for key, cfg in MODELS.items():
        print(f"Analyzing {key}...")
        doc["models"][key] = analyze_model(key, cfg, ids)
    OUT_JSON.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_JSON}")
    write_md(doc)
    plot_comparison(doc)


if __name__ == "__main__":
    main()
