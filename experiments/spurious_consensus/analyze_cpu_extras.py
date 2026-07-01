#!/usr/bin/env python3
"""Extra CPU-only analyses on existing K=64 samples (no GPU)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from analyze_all_models import MODELS, MAIN_BENCHES, load_model, rows_for_ids

ROOT = Path(__file__).resolve().parent
N_GRID = [2, 4, 8, 16, 32, 64]


def majority_stats(row: dict) -> tuple[str, float, bool]:
    pairs = [(a, c) for a, c in zip(row["answers"], row["correct"]) if a]
    if len(pairs) < 2:
        return "", 0.0, False
    answers, correct = zip(*pairs)
    cnt = Counter(answers)
    maj, top = cnt.most_common(1)[0]
    p_top = top / len(answers)
    cmap = {}
    for a, c in zip(answers, correct):
        cmap.setdefault(a, int(c))
    wrong = cmap.get(maj, 0) != 1
    return maj, p_top, wrong


def cross_model_scr_overlap(all_data: dict[str, dict], common: set[str], tau: float = 0.9) -> dict:
    scr_sets: dict[str, set[str]] = {}
    maj_ans: dict[str, dict[str, str]] = {}
    for tag, _ in MODELS:
        scr_sets[tag] = set()
        maj_ans[tag] = {}
        for qid in common:
            r = all_data[tag].get(qid)
            if not r or int(r.get("label_drop", 0)) == 1:
                continue
            maj, p_top, wrong = majority_stats(r)
            if wrong and p_top >= tau:
                scr_sets[tag].add(qid)
                maj_ans[tag][qid] = maj

    pairs_out = []
    tags = [t for t, _ in MODELS]
    for i, t1 in enumerate(tags):
        for t2 in tags[i + 1 :]:
            inter = scr_sets[t1] & scr_sets[t2]
            same_ans = sum(1 for q in inter if maj_ans[t1].get(q) == maj_ans[t2].get(q))
            pairs_out.append({
                "model_a": t1,
                "model_b": t2,
                "scr_both": len(inter),
                "same_wrong_answer": same_ans,
                "scr_a": len(scr_sets[t1]),
                "scr_b": len(scr_sets[t2]),
            })
    return {"tau": tau, "pairwise": pairs_out, "scr_counts": {t: len(scr_sets[t]) for t in tags}}


def exogenous_difficulty(all_data: dict[str, dict], common: set[str]) -> dict:
    """Per-question LOO difficulty + SCR vs dispersed wrong comparison."""
    qids = sorted(common)
    per_model_acc = {}
    for tag, _ in MODELS:
        acc = []
        for qid in qids:
            r = all_data[tag][qid]
            pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a]
            if not pairs:
                acc.append(np.nan)
                continue
            acc.append(np.mean([c for _, c in pairs]))
        per_model_acc[tag] = acc

    out = {}
    for tag, lab in MODELS:
        scr_exo, disp_exo = [], []
        for qi, qid in enumerate(qids):
            r = all_data[tag][qid]
            if int(r.get("label_drop", 0)) == 1:
                continue
            maj, p_top, wrong = majority_stats(r)
            if not wrong:
                continue
            others = [per_model_acc[t][qi] for t, _ in MODELS if t != tag]
            exo = float(np.nanmean(others))
            if p_top >= 0.9:
                scr_exo.append(exo)
            elif p_top < 0.5:
                disp_exo.append(exo)
        out[tag] = {
            "label": lab,
            "scr_wrong_exo_mean": float(np.mean(scr_exo)) if scr_exo else None,
            "scr_wrong_n": len(scr_exo),
            "dispersed_wrong_exo_mean": float(np.mean(disp_exo)) if disp_exo else None,
            "dispersed_wrong_n": len(disp_exo),
        }
    return out


def bootstrap_n_sweep(rows: list[dict], n: int, B: int, rng: np.random.Generator) -> dict:
    """Bootstrap subsample n from K=64 for AUROC and SCR@0.9."""
    Q = []
    for r in rows:
        pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a]
        if len(pairs) < max(n, 2):
            continue
        ans, cor = zip(*pairs)
        Q.append({"answers": list(ans), "correct": list(cor)})

    aurocs, scrs = [], []
    for _ in range(B):
        labels, scores, wrong_flags, p_tops = [], [], [], []
        for q in Q:
            idx = rng.choice(len(q["answers"]), size=n, replace=True)
            sub = [q["answers"][i] for i in idx]
            cmap = {}
            for i in idx:
                a, c = q["answers"][i], q["correct"][i]
                cmap.setdefault(a, c)
            cnt = Counter(sub)
            maj, top = cnt.most_common(1)[0]
            p_top = top / n
            wrong = cmap.get(maj, 0) != 1
            labels.append(int(wrong))
            scores.append(1 - p_top)
            wrong_flags.append(wrong)
            p_tops.append(p_top)
        if 0 < sum(labels) < len(labels):
            aurocs.append(float(roc_auc_score(labels, scores)))
        wr = [w for w, p in zip(wrong_flags, p_tops) if w]
        scrs.append(sum(1 for p in wr if p >= 0.9) / len(wr) if wr else 0.0)

    return {
        "n": n,
        "n_questions": len(Q),
        "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "auroc_std": float(np.std(aurocs)) if aurocs else None,
        "scr09_mean": float(np.mean(scrs)) if scrs else None,
        "scr09_std": float(np.std(scrs)) if scrs else None,
    }


def matched_budget_table(all_data: dict[str, dict], common: set[str], cfg_an: dict) -> dict:
    rng = np.random.default_rng(0)
    B = min(cfg_an.get("bootstrap_draws", 200), 100)  # faster for extras
    out = {}
    for tag, lab in MODELS:
        rows = rows_for_ids(all_data[tag], common)
        out[tag] = {
            "label": lab,
            "n8": bootstrap_n_sweep(rows, 8, B, rng),
            "n64": bootstrap_n_sweep(rows, 64, B, rng),
        }
    return out


def cost_table() -> dict:
    return {
        "protocol": {"K": 64, "temperature": 0.5, "top_p": 0.95, "seed": 41},
        "per_question_decodes": 64,
        "common_questions": 2228,
        "per_model_total_decodes": 64 * 2228,
        "six_models_total_decodes": 64 * 2228 * 6,
        "scr_reasoning_resample": {
            "model": "qwen25_7b",
            "subset": "SCR@1.0",
            "n_questions": 42,
            "K": 64,
            "temperature": 1.0,
            "extra_decodes": 42 * 64,
        },
        "note": "Matched-budget PANDA uses 1+8=9 decodes/question; SC@8 would use 8 decodes/question.",
    }


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    cfg_an = cfg["analysis"]
    seed = 41

    all_data = {tag: load_model(tag, seed) for tag, _ in MODELS}
    common = set.intersection(*[set(d) for d in all_data.values()])

    payload = {
        "common_n": len(common),
        "cross_model_scr": cross_model_scr_overlap(all_data, common, tau=0.9),
        "cross_model_scr_100": cross_model_scr_overlap(all_data, common, tau=1.0),
        "exogenous_difficulty": exogenous_difficulty(all_data, common),
        "matched_budget_n8_vs_n64": matched_budget_table(all_data, common, cfg_an),
        "cost": cost_table(),
    }

    out_json = ROOT / "results" / "cpu_extras_analysis.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {out_json}")


if __name__ == "__main__":
    main()
