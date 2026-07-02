#!/usr/bin/env python3
"""Generate Table 1/2/3: ASE main + token auxiliary + minimal fusion (clean labels)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from panda.core.altmass_decomposition import altmass_variants_weight_branch
from panda.core.cluster_token_trace import merge_cluster_token_trace
from panda.core.panda import enrich_row_with_panda
from panda.baselines.from_record import enrich_row_with_baselines
from panda.baselines.registry import BASELINE_INVERT_KEYS, BASELINE_TABLE_ROWS
from panda.core.reasoning_token_features import merge_reasoning_token_metrics
from panda.grading.answer_canonicalizer import grade_answer
from panda.metrics_tokur import compute_detection_metrics
from panda.paths import DEFAULT_OUT

DEFAULT_OUT = DEFAULT_OUT

TABLE1_BASELINE = [
    ("TokUR EU", "tokur_eu_sum", "teacher-forced $a_0$ + weight epistemic uncertainty (baseline)"),
    *BASELINE_TABLE_ROWS,
]

TABLE1 = [
    ("PANDA", "PANDA", "PANDA (F_resp + 0.05·D_ans + 0.03·D_reason)"),
    ("F_resp", "F_resp", "response fragmentation (4T+4W ASE)"),
    ("D_ans", "D_ans", "answer drift at commitment"),
    ("D_reason", "D_reason", "reasoning drift (local spread)"),
    ("T-ASE", "T_ASE", "text rephrase answer fragmentation"),
    ("W-ASE", "W_ASE", "weight perturbation answer fragmentation"),
    ("TW-ASE", "TW_ASE", "joint text-weight answer fragmentation (≈ F_resp)"),
    ("T_num_clusters", "T_num_clusters", "text cluster count"),
    ("W_num_clusters", "W_num_clusters", "weight cluster count"),
    ("TW_num_clusters", "TW_num_clusters", "joint cluster count"),
]

TABLE2 = [
    ("W_math_token_flip_max", "W_math_token_flip_max", "math-token instability"),
    ("W_alternative_answer_mass_topk", "W_alternative_answer_mass_topk", "answer-cluster competition (≈ D_ans)"),
    ("AltMass_final", "AltMass_final", "commitment-boundary alt mass (≈ D_ans)"),
    ("AltMass_local_spread_reason", "AltMass_local_spread_reason", "local reasoning spread (≈ D_reason)"),
    ("W_confident_fragmentation", "W_confident_fragmentation", "confident path switching"),
    ("W_formula_skeleton_entropy", "W_formula_skeleton_entropy", "formula-level instability"),
    ("final_answer_equiv_last_equation", "final_answer_equiv_last_equation", "process consistency"),
]

TABLE2_INVERT = {"final_answer_equiv_last_equation"}


def relabel(rows: list[dict]) -> list[dict]:
    for r in rows:
        g = grade_answer(
            r.get("a0", ""),
            r.get("reference", ""),
            record_id=r.get("id"),
            dataset=r.get("dataset"),
        )
        r["is_correct_clean"] = g["is_correct_clean"]
        r["label_wrong_clean"] = g["label_wrong_clean"]
        r["relabeled"] = g["relabeled"]
    return rows


def merge_tokur_scores(out_dir: Path, dataset: str, rows: list[dict]) -> list[dict]:
    tok_path = out_dir / dataset / "tokur_baseline.jsonl"
    if not tok_path.exists():
        return rows
    by_id: dict[str, float] = {}
    for line in tok_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("error"):
            by_id[rec["id"]] = float(rec.get("tokur_eu_sum", float("nan")))
    for r in rows:
        if r["id"] in by_id:
            r["tokur_eu_sum"] = by_id[r["id"]]
    return rows


def load_features_table(out_dir: Path, dataset: str) -> list[dict]:
    """Fast path for paper tables: features.jsonl + clean labels only (no raw_runs scan)."""
    feat_path = out_dir / dataset / "features.jsonl"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}; run recompute_metrics first.")
    rows = [json.loads(ln) for ln in feat_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = relabel(rows)
    return merge_tokur_scores(out_dir, dataset, rows)


def load_enriched(out_dir: Path, dataset: str) -> list[dict]:
    feat_path = out_dir / dataset / "features.jsonl"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}; run recompute_metrics first.")
    rows = [json.loads(l) for l in feat_path.read_text().splitlines() if l.strip()]
    rows = relabel(rows)
    rows = merge_tokur_scores(out_dir, dataset, rows)
    raw_dir = out_dir / dataset / "raw_runs"
    by_id = {r["id"]: r for r in rows}
    enriched = []
    for p in sorted(raw_dir.glob("*.json")):
        try:
            if p.stat().st_size < 1000 or p.name.endswith((".error.json", ".partial.json")):
                continue
        except FileNotFoundError:
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["id"] not in by_id:
            continue
        row = dict(by_id[rec["id"]])
        base = rec.get("base_generation") or {}
        text = rec.get("text_rephrase_runs") or []
        weight = rec.get("weight_perturb_runs") or []
        row.update(merge_reasoning_token_metrics(base, text, weight))
        row.update(merge_cluster_token_trace(base, text, weight))
        row.update(altmass_variants_weight_branch(weight))
        enrich_row_with_panda(row, write_legacy=False)
        enrich_row_with_baselines(row, rec)
        enriched.append(row)
    return enriched


def auroc_clean(rows: list[dict], key: str, invert: bool = False) -> float:
    labels = [r["label_wrong_clean"] == 0 for r in rows]
    scores = [float(r.get(key, float("nan"))) for r in rows]
    if invert:
        scores = [-s if s == s else float("nan") for s in scores]
    if all(math.isnan(s) for s in scores):
        return float("nan")
    return compute_detection_metrics(labels, scores).auroc


def lr_auroc(rows: list[dict], feats: list[str]) -> float:
    y = np.array([r["label_wrong_clean"] for r in rows], dtype=int)
    X = np.array([[float(r.get(k, 0.0)) for k in feats] for r in rows], dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(Xs, y)
    return float(roc_auc_score(y, clf.predict_proba(Xs)[:, 1]))


def build_tables(rows: list[dict], dataset: str) -> str:
    n = len(rows)
    wrong = sum(r["label_wrong_clean"] for r in rows)
    relabeled = sum(r.get("relabeled", False) for r in rows)
    lines = [
        f"# {dataset} 主表（Clean labels）",
        "",
        f"N={n}, wrong={wrong}, acc={1-wrong/n:.1%}, relabeled={relabeled}",
        "",
        "## 表 1：Answer-level fragmentation 主结果",
        "",
        "| Method | Clean AUROC | 解释 |",
        "|--------|------------:|------|",
    ]
    if any(not math.isnan(float(r.get("tokur_eu_sum", float("nan")))) for r in rows):
        for name, key, desc in TABLE1_BASELINE:
            a = auroc_clean(rows, key, invert=(key in BASELINE_INVERT_KEYS))
            lines.append(f"| {name} | {a:.4f} | {desc} |")
        lines.append("| | | |")
    for name, key, desc in TABLE1:
        a = auroc_clean(rows, key, invert=(key in BASELINE_INVERT_KEYS))
        lines.append(f"| {name} | {a:.4f} | {desc} |")

    lines.extend(
        [
            "",
            "## 表 2：Token/process 辅助",
            "",
            "| Method | Clean AUROC | 解释 |",
            "|--------|------------:|------|",
        ]
    )
    for name, key, desc in TABLE2:
        a = auroc_clean(rows, key, invert=(key in TABLE2_INVERT))
        lines.append(f"| {name} | {a:.4f} | {desc} |")

    lines.extend(
        [
            "",
            "> Token/process 有信号，但低于 answer-level ASE。",
            "",
            "## 表 3：Minimal add-on",
            "",
            "| Model | AUROC | 目的 |",
            "|-------|------:|------|",
        ]
    )
    fusions = [
        ("PANDA (linear)", ["PANDA"], "主方法（线性组合）"),
        ("TW-ASE", ["TW_ASE"], "legacy 主分量 F_resp"),
        ("TW-ASE + alternative_answer_mass", ["TW_ASE", "W_alternative_answer_mass_topk"], "token auxiliary gain"),
        ("TW-ASE + final_equiv_last_eq", ["TW_ASE", "final_answer_equiv_last_equation"], "process diagnostic gain"),
        (
            "TW-ASE + alt_mass + final_eq",
            ["TW_ASE", "W_alternative_answer_mass_topk", "final_answer_equiv_last_equation"],
            "token + process",
        ),
    ]
    for name, feats, purpose in fusions:
        if all(f in rows[0] for f in feats):
            # final_equiv: invert for LR (low equiv = wrong); use (1-x) for fusion
            X_rows = []
            for r in rows:
                vals = []
                for f in feats:
                    v = float(r.get(f, 0.0))
                    if f == "final_answer_equiv_last_equation":
                        v = 1.0 - v  # higher = more likely wrong
                    vals.append(v)
                X_rows.append(vals)
            y = np.array([r["label_wrong_clean"] for r in rows], dtype=int)
            if len(np.unique(y)) >= 2:
                Xs = StandardScaler().fit_transform(np.array(X_rows))
                clf = LogisticRegression(max_iter=2000, random_state=42)
                clf.fit(Xs, y)
                a = float(roc_auc_score(y, clf.predict_proba(Xs)[:, 1]))
            else:
                a = float("nan")
            lines.append(f"| {name} | {a:.4f} | {purpose} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--score-tokur", action="store_true", help="Run TokUR EU baseline before tables")
    ap.add_argument("--tokur-device", default="cuda:0")
    args = ap.parse_args()

    if args.score_tokur:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "panda.core.score_tokur_baseline",
                "--out-dir",
                str(args.out_dir),
                "--dataset",
                args.dataset,
                "--device",
                args.tokur_device,
                "--resume",
            ],
            env={**dict(__import__("os").environ), "PYTHONPATH": "src"},
            cwd=Path(__file__).resolve().parents[2],
        )

    feat_path = args.out_dir / args.dataset / "features.jsonl"
    if args.recompute or not feat_path.exists():
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "panda.core.recompute_metrics",
                "--out-dir",
                str(args.out_dir),
                "--datasets",
                args.dataset,
            ],
            env={**dict(__import__("os").environ), "PYTHONPATH": "src"},
            cwd=Path(__file__).resolve().parents[2],
        )

    rows = load_enriched(args.out_dir, args.dataset)
    clean_path = args.out_dir / args.dataset / "features_clean.jsonl"
    with clean_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = build_tables(rows, args.dataset)
    out = args.out_dir / f"TABLES_{args.dataset}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {out}")
    print(f"Wrote {clean_path}")


if __name__ == "__main__":
    main()