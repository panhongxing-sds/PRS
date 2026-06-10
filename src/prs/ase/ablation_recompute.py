#!/usr/bin/env python3
"""CPU-only PRS ablation: subset perturbations + component / lambda variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from prs.ase.metrics import metrics_from_record
from prs.ase.prs import LAMBDA_A, LAMBDA_R, compute_prs
from prs.grading.answer_canonicalizer import grade_answer
from prs.metrics_tokur import compute_detection_metrics
from prs.paths import DEFAULT_OUT


def _subset_record(record: dict, *, n_text: int | None, n_weight: int | None) -> dict:
    rec = dict(record)
    text = list(record.get("text_rephrase_runs") or [])
    weight = list(record.get("weight_perturb_runs") or [])
    if n_text is not None:
        text = text[:n_text]
    if n_weight is not None:
        weight = weight[:n_weight]
    rec["text_rephrase_runs"] = text
    rec["weight_perturb_runs"] = weight
    return rec


def _auroc(rows: list[dict], key: str, *, invert: bool = False) -> float:
    labels = [r["label_wrong_clean"] == 0 for r in rows]
    scores = [float(r.get(key, float("nan"))) for r in rows]
    if invert:
        scores = [-s if math.isfinite(s) else float("nan") for s in scores]
    if all(math.isnan(s) for s in scores):
        return float("nan")
    return compute_detection_metrics(labels, scores).auroc


def _add_ablation_scores(row: dict, *, lambda_a: float, lambda_r: float) -> dict[str, float]:
    """Fixed-lambda PRS variants (reference only).

    NOTE: with default lambda_a=0.05, lambda_r=0.03 these tiny terms barely move
    the score, so ``abl_wo_D_ans``/``abl_wo_D_reason`` are numerically ~= full PRS.
    They are kept only to *demonstrate* why naive zeroing is uninformative; the
    real ablation AUROC table is produced by ``summarize_ablation`` (re-fit LR).
    """
    f = float(row.get("F_resp", row.get("TW_ASE", float("nan"))))
    a = float(row.get("D_ans", row.get("AltMass_final", float("nan"))))
    r = float(row.get("D_reason", row.get("AltMass_local_spread_reason", float("nan"))))
    t = float(row.get("T_ASE", float("nan")))
    w = float(row.get("W_ASE", float("nan")))
    return {
        "abl_PRS_full": compute_prs(f, a, r, lambda_a=lambda_a, lambda_r=lambda_r),
        "abl_wo_F_resp": compute_prs(0.0, a, r, lambda_a=lambda_a, lambda_r=lambda_r),
        "abl_wo_D_ans": compute_prs(f, 0.0, r, lambda_a=lambda_a, lambda_r=lambda_r),
        "abl_wo_D_reason": compute_prs(f, a, 0.0, lambda_a=lambda_a, lambda_r=lambda_r),
        "abl_F_resp_T": compute_prs(t, a, r, lambda_a=lambda_a, lambda_r=lambda_r),
        "abl_F_resp_W": compute_prs(w, a, r, lambda_a=lambda_a, lambda_r=lambda_r),
    }


# ---------------------------------------------------------------------------
# Real ablation: full PRS minus exactly one component, re-fit logistic weights.
#
# Why re-fit instead of zeroing a lambda term: default lambda_a/lambda_r are tiny,
# so F_resp dominates and "remove D_ans" via lambda=0 is numerically ~= full PRS
# (delta ~ 0). Re-fitting a logistic model on the *remaining* components lets each
# surviving component drive the score, so AUROC reflects its true contribution.
# Single-feature LR is a monotonic transform -> equals that component's raw AUROC.
# ---------------------------------------------------------------------------
ABLATION_FEATURES: list[tuple[str, list[str]]] = [
    ("PRS (full)", ["F_resp", "D_ans", "D_reason"]),
    ("-F_resp", ["D_ans", "D_reason"]),
    ("-D_ans", ["F_resp", "D_reason"]),
    ("-D_reason", ["F_resp", "D_ans"]),
    ("F_resp=T-ASE", ["T_ASE", "D_ans", "D_reason"]),
    ("F_resp=W-ASE", ["W_ASE", "D_ans", "D_reason"]),
    # S_tr definition ablation: B (domain-agnostic top-k%, primary) vs
    # A (per-domain content tokens) vs legacy math-token mean.
    ("S_tr=topk (B)", ["F_resp", "D_ans", "AltMass_local_spread_topk"]),
    ("S_tr=content (A)", ["F_resp", "D_ans", "AltMass_local_spread_content"]),
    ("S_tr=math (legacy)", ["F_resp", "D_ans", "AltMass_local_spread_reason"]),
]


def _lr_auroc(rows: list[dict], feats: list[str], *, seed: int = 42) -> float:
    """In-sample logistic-regression AUROC over the given component features."""
    y = np.array([int(r["label_wrong_clean"]) for r in rows], dtype=int)
    X = np.array([[float(r.get(k, float("nan"))) for k in feats] for r in rows], dtype=float)
    finite = np.all(np.isfinite(X), axis=1)
    X, y = X[finite], y[finite]
    if len(np.unique(y)) < 2:
        return float("nan")
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xs, y)
    return float(roc_auc_score(y, clf.predict_proba(Xs)[:, 1]))


def summarize_ablation(rows: list[dict], *, seed: int = 42) -> dict[str, dict[str, float]]:
    """AUROC + ΔAUROC (vs full PRS) for each one-component ablation."""
    aurocs = {name: _lr_auroc(rows, feats, seed=seed) for name, feats in ABLATION_FEATURES}
    full = aurocs.get("PRS (full)", float("nan"))
    out: dict[str, dict[str, float]] = {}
    for name, a in aurocs.items():
        delta = (a - full) if (a == a and full == full) else float("nan")
        out[name] = {"auroc": a, "delta": delta}
    return out


def recompute_dataset(
    out_dir: Path,
    dataset: str,
    *,
    n_text: int | None = None,
    n_weight: int | None = None,
    lambda_a: float = LAMBDA_A,
    lambda_r: float = LAMBDA_R,
    atu_top_pct: float = 0.10,
) -> list[dict]:
    raw_dir = out_dir / dataset / "raw_runs"
    tok_path = out_dir / dataset / "tokur_baseline.jsonl"
    tok_by_id: dict[str, float] = {}
    if tok_path.exists():
        for line in tok_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if not rec.get("error"):
                    tok_by_id[rec["id"]] = float(rec.get("tokur_eu_sum", float("nan")))

    rows: list[dict] = []
    for p in sorted(raw_dir.glob("*.json")):
        if p.name.endswith((".error.json", ".partial.json")):
            continue
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rec = _subset_record(record, n_text=n_text, n_weight=n_weight)
        row = metrics_from_record(rec, top_pct=atu_top_pct)
        g = grade_answer(row.get("a0", ""), row.get("reference", ""), record_id=row.get("id"))
        row["label_wrong_clean"] = g["label_wrong_clean"]
        row["is_correct_clean"] = g["is_correct_clean"]
        if row["id"] in tok_by_id:
            row["tokur_eu_sum"] = tok_by_id[row["id"]]
        row.update(_add_ablation_scores(row, lambda_a=lambda_a, lambda_r=lambda_r))
        row["abl_n_text"] = n_text if n_text is not None else len(rec.get("text_rephrase_runs") or [])
        row["abl_n_weight"] = n_weight if n_weight is not None else len(rec.get("weight_perturb_runs") or [])
        row["abl_lambda_a"] = lambda_a
        row["abl_lambda_r"] = lambda_r
        rows.append(row)
    return rows


def build_markdown_table(results: dict[str, dict[str, dict[str, float]]], datasets: list[str]) -> str:
    lines = [
        "# PRS 组件消融（Clean AUROC，re-fit logistic，CPU recompute）",
        "",
        "消融 = 完整 PRS 去掉恰好一个组件后**重新拟合** logistic 权重；ΔAUROC 相对 full PRS。",
        "",
    ]
    header = ["Variant"]
    for ds in datasets:
        header += [f"{ds} AUROC", f"{ds} ΔAUROC"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---:"] * len(header)) + "|")
    for name, _ in ABLATION_FEATURES:
        cells = [name]
        for ds in datasets:
            m = results.get(ds, {}).get(name, {})
            a = m.get("auroc", float("nan"))
            d = m.get("delta", float("nan"))
            cells.append(f"{a:.4f}" if a == a else "—")
            cells.append(f"{d:+.4f}" if d == d else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU PRS ablation from raw_runs")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--datasets", default="minerva,math500,gsm8k,deepscaler")
    ap.add_argument("--n-text", type=int, default=None, help="Subset text rephrase runs (default: all)")
    ap.add_argument("--n-weight", type=int, default=None, help="Subset weight perturb runs (default: all)")
    ap.add_argument("--lambda-a", type=float, default=LAMBDA_A)
    ap.add_argument("--lambda-r", type=float, default=LAMBDA_R)
    ap.add_argument("--atu-top-pct", type=float, default=0.10)
    ap.add_argument("--budget-sweep", action="store_true", help="Sweep n_text/n_weight grid for Table B")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    results: dict[str, dict[str, float]] = {}

    if args.budget_sweep:
        sweep: dict[str, dict[str, dict[str, float]]] = {}
        for ds in datasets:
            sweep[ds] = {}
            for n in (0, 2, 4, 8):
                rows = recompute_dataset(
                    args.out_dir, ds, n_text=n, n_weight=8, lambda_a=args.lambda_a, lambda_r=args.lambda_r
                )
                sweep[ds][f"N{n}"] = {"PRS": _auroc(rows, "abl_PRS_full"), "TW": _auroc(rows, "F_resp")}
            for m in (0, 2, 4, 8):
                rows = recompute_dataset(
                    args.out_dir, ds, n_text=8, n_weight=m, lambda_a=args.lambda_a, lambda_r=args.lambda_r
                )
                sweep[ds][f"M{m}"] = {"PRS": _auroc(rows, "abl_PRS_full"), "TW": _auroc(rows, "F_resp")}
        out = args.output or (args.out_dir / "ABLATION_budget_sweep.json")
        out.write_text(json.dumps(sweep, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out}")
        return

    for ds in datasets:
        raw_dir = args.out_dir / ds / "raw_runs"
        if not raw_dir.exists():
            print(f"skip {ds}: no {raw_dir}")
            continue
        rows = recompute_dataset(
            args.out_dir,
            ds,
            n_text=args.n_text,
            n_weight=args.n_weight,
            lambda_a=args.lambda_a,
            lambda_r=args.lambda_r,
            atu_top_pct=args.atu_top_pct,
        )
        results[ds] = summarize_ablation(rows)
        abl_path = args.out_dir / ds / "ablation_rows.jsonl"
        with abl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{ds}: n={len(rows)} → {abl_path}")

    md = build_markdown_table(results, list(results))
    out_md = args.output or (args.out_dir / "ABLATION_component.md")
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
