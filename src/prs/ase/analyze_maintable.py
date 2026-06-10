#!/usr/bin/env python3
"""Aggregate 3-seed ASE results into paper main tables (AUROC / AUPRC / ACC*)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from prs.ase.analyze_main_tables import TABLE2_INVERT, load_enriched, load_features_table
from prs.baselines.registry import BASELINE_INVERT_KEYS, BASELINE_REGISTRY
from prs.datasets.registry import (
    DATASET_IDS,
    DEFAULT_EXPERIMENT_SEEDS,
    get_dataset_spec,
)
from prs.metrics_tokur import compute_detection_metrics

# Main-table rows: TokUR EU + ISO baselines + core PRS
MAINTABLE_METHODS: list[tuple[str, str, str, bool]] = [
    ("TokUR EU", "tokur_eu_sum", "teacher-forced EU sum", False),
    *[(b.name, b.key, b.description, b.invert) for b in BASELINE_REGISTRY],
    ("PRS", "PRS", "Perturbation Reliability Score", False),
    ("F_resp", "F_resp", "response fragmentation", False),
    ("D_ans", "D_ans", "answer drift", False),
    ("D_reason", "D_reason", "reasoning drift", False),
]

COT_KEY = "cot_greedy_acc"


def seed_out_dir(base: Path, seed: int) -> Path:
    return base / f"seed{seed}"


def eval_clean(rows: list[dict], key: str, invert: bool = False) -> dict[str, float]:
    labels = [r["label_wrong_clean"] == 0 for r in rows]
    scores = [float(r.get(key, float("nan"))) for r in rows]
    if invert:
        scores = [-s if s == s else float("nan") for s in scores]
    if all(math.isnan(s) for s in scores):
        return {"auroc": float("nan"), "auprc": float("nan"), "acc_star": float("nan"), "n": len(rows)}
    m = compute_detection_metrics(labels, scores)
    return {"auroc": m.auroc, "auprc": m.auprc, "acc_star": m.acc_star, "n": m.n}


def cot_acc_star(rows: list[dict]) -> float:
    """CoT lower-bound: greedy accuracy only (no uncertainty ranking)."""
    if not rows:
        return float("nan")
    correct = sum(1 for r in rows if r.get("label_wrong_clean", 1) == 0)
    return correct / len(rows)


def aggregate_seed_metrics(values: list[float]) -> dict[str, float]:
    arr = np.array([v for v in values if v == v], dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": len(arr)}


def fmt_mean_std(mean: float, std: float, *, as_pct: bool = True, digits: int = 2) -> str:
    if mean != mean:
        return "—"
    if as_pct:
        mean, std = mean * 100.0, std * 100.0
    if std == std and std > 0:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"{mean:.{digits}f}"


def collect_per_seed(
    base_out: Path,
    dataset: str,
    seeds: list[int],
    *,
    features_only: bool,
) -> dict[int, list[dict]]:
    by_seed: dict[int, list[dict]] = {}
    for seed in seeds:
        out = seed_out_dir(base_out, seed)
        try:
            rows = load_features_table(out, dataset) if features_only else load_enriched(out, dataset)
            by_seed[seed] = rows
        except FileNotFoundError:
            continue
    return by_seed


def build_results(
    base_out: Path,
    datasets: list[str],
    seeds: list[int],
    *,
    features_only: bool,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """dataset -> method -> metric -> {mean, std}."""
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for ds in datasets:
        per_seed_rows = collect_per_seed(base_out, ds, seeds, features_only=features_only)
        if not per_seed_rows:
            continue
        results[ds] = {}
        sample_rows = next(iter(per_seed_rows.values()))
        for name, key, _desc, invert in MAINTABLE_METHODS:
            if key == COT_KEY:
                accs = [cot_acc_star(rows) for rows in per_seed_rows.values()]
                agg = aggregate_seed_metrics(accs)
                results[ds][name] = {
                    "auroc": {"mean": float("nan"), "std": float("nan")},
                    "auprc": {"mean": float("nan"), "std": float("nan")},
                    "acc_star": agg,
                }
                continue
            if key not in sample_rows[0] and key != "tokur_eu_sum":
                continue
            if key == "tokur_eu_sum" and not any("tokur_eu_sum" in r for r in sample_rows):
                continue
            inv = invert or key in BASELINE_INVERT_KEYS or key in TABLE2_INVERT
            aurocs, auprcs, accs = [], [], []
            for rows in per_seed_rows.values():
                m = eval_clean(rows, key, invert=inv)
                aurocs.append(m["auroc"])
                auprcs.append(m["auprc"])
                accs.append(m["acc_star"])
            results[ds][name] = {
                "auroc": aggregate_seed_metrics(aurocs),
                "auprc": aggregate_seed_metrics(auprcs),
                "acc_star": aggregate_seed_metrics(accs),
            }
    return results


def build_markdown(
    results: dict[str, dict[str, dict[str, dict[str, float]]]],
    datasets: list[str],
    *,
    model_label: str,
    as_pct: bool,
    include_iso_col: bool,
) -> str:
    ds_labels = {get_dataset_spec(ds).id: get_dataset_spec(ds).label for ds in datasets}
    header = ["Method"]
    if include_iso_col:
        header.append("ISO?")
    for ds in datasets:
        lbl = ds_labels.get(ds, ds)
        header.extend([f"{lbl} AUROC", f"{lbl} AUPRC", f"{lbl} ACC*"])

    lines = [
        f"# Main table — {model_label}",
        "",
        f"Metrics: mean ± std over seeds {DEFAULT_EXPERIMENT_SEEDS}.",
        "ACC* = Top-50% accuracy (TokUR protocol). CoT row: ACC* only.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + (["---:"] if include_iso_col else []) + ["---:"] * (3 * len(datasets))) + " |",
    ]

    iso_by_name = {b.name: ("✓" if b.official else "≈") for b in BASELINE_REGISTRY}
    iso_by_name["TokUR EU"] = "✓"

    for name, key, _desc, _inv in MAINTABLE_METHODS:
        shown = False
        for ds in datasets:
            if name in results.get(ds, {}):
                shown = True
                break
        if not shown:
            continue
        row = [name]
        if include_iso_col:
            row.append(iso_by_name.get(name, ""))
        for ds in datasets:
            m = results.get(ds, {}).get(name, {})
            if key == COT_KEY:
                row.extend(["—", "—", fmt_mean_std(m.get("acc_star", {}).get("mean", float("nan")),
                                                   m.get("acc_star", {}).get("std", float("nan")), as_pct=as_pct)])
            else:
                for metric in ("auroc", "auprc", "acc_star"):
                    agg = m.get(metric, {})
                    row.append(fmt_mean_std(agg.get("mean", float("nan")), agg.get("std", float("nan")), as_pct=as_pct))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def build_latex(
    results: dict[str, dict[str, dict[str, dict[str, float]]]],
    datasets: list[str],
    *,
    model_label: str,
    as_pct: bool,
    caption: str,
    label: str,
) -> str:
    ds_labels = {get_dataset_spec(ds).id: get_dataset_spec(ds).label for ds in datasets}
    col_spec = "l|" + "|".join(["ccc"] * len(datasets))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(["Method"] + [f"\\multicolumn{{3}}{{c}}{{{ds_labels.get(ds, ds)}}}" for ds in datasets]) + " \\\\",
    ]
    start = 2
    for _ in datasets:
        lines.append(f"\\cmidrule(lr){{{start}-{start + 2}}}")
        start += 3
    lines.append(" & ".join(["Method"] + ["AUROC", "AUPRC", "ACC*"] * len(datasets)) + " \\\\")
    lines.append("\\midrule")

    for name, key, _desc, _inv in MAINTABLE_METHODS:
        if not any(name in results.get(ds, {}) for ds in datasets):
            continue
        cells = [name.replace("_", "\\_")]
        for ds in datasets:
            m = results.get(ds, {}).get(name, {})
            if key == COT_KEY:
                cells.extend(["—", "—", fmt_mean_std(m.get("acc_star", {}).get("mean", float("nan")),
                                                    m.get("acc_star", {}).get("std", float("nan")), as_pct=as_pct)])
            else:
                for metric in ("auroc", "auprc", "acc_star"):
                    agg = m.get(metric, {})
                    cells.append(fmt_mean_std(agg.get("mean", float("nan")), agg.get("std", float("nan")), as_pct=as_pct))
        line = " & ".join(cells) + " \\\\"
        if name == "TokUR EU":
            line += " \\midrule"
        lines.append(line)

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="3-seed main table aggregation")
    ap.add_argument("--out-dir", type=Path, required=True, help="Model output root (contains seed41/, seed42/, ...)")
    ap.add_argument("--paper-dir", type=Path, default=None)
    ap.add_argument("--model-label", default="Model")
    ap.add_argument("--datasets", default=",".join(DATASET_IDS))
    ap.add_argument("--seeds", default=",".join(map(str, DEFAULT_EXPERIMENT_SEEDS)))
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--no-pct", action="store_true", help="Report 0-1 scale instead of 0-100")
    ap.add_argument("--iso-col", action="store_true", help="Add ISO? column for baselines")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    as_pct = not args.no_pct

    results = build_results(args.out_dir, datasets, seeds, features_only=args.features_only)
    paper = args.paper_dir or (args.out_dir / "maintable")
    paper.mkdir(parents=True, exist_ok=True)

    md = build_markdown(results, datasets, model_label=args.model_label, as_pct=as_pct, include_iso_col=args.iso_col)
    tex = build_latex(
        results,
        datasets,
        model_label=args.model_label,
        as_pct=as_pct,
        caption=f"{args.model_label}: detection metrics (mean$\\pm$std over {len(seeds)} seeds).",
        label=f"tab:maintable_{args.model_label.replace('.', '').replace('-', '_')}",
    )

    (paper / "maintable.md").write_text(md, encoding="utf-8")
    (paper / "maintable.tex").write_text(tex, encoding="utf-8")
    (paper / "maintable_results.json").write_text(
        json.dumps({"seeds": seeds, "datasets": datasets, "metrics": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(md)
    print(f"Wrote {paper}/maintable.{{md,tex,results.json}}")


if __name__ == "__main__":
    main()
