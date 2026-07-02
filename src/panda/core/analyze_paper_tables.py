#!/usr/bin/env python3
"""Generate paper-ready tables: AUROC / AUPRC / ACC* across datasets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from panda.core.analyze_main_tables import (
    TABLE1,
    TABLE1_BASELINE,
    TABLE2,
    TABLE2_INVERT,
    load_enriched,
    load_features_table,
)
from panda.baselines.registry import BASELINE_INVERT_KEYS
from panda.metrics_tokur import compute_detection_metrics
from panda.paths import DEFAULT_OUT, PAPER

DEFAULT_OUT = DEFAULT_OUT
DEFAULT_PAPER = Path("/home/phx/PANDA/paper")

PAPER_MAIN = TABLE1_BASELINE + TABLE1[:3]  # TokUR + T/W/TW-ASE
PAPER_EXTENDED = TABLE1_BASELINE + TABLE1


def eval_clean(rows: list[dict], key: str, invert: bool = False) -> dict[str, float]:
    labels = [r["label_wrong_clean"] == 0 for r in rows]
    scores = [float(r.get(key, float("nan"))) for r in rows]
    if invert:
        scores = [-s if s == s else float("nan") for s in scores]
    if all(math.isnan(s) for s in scores):
        return {"auroc": float("nan"), "auprc": float("nan"), "acc_star": float("nan"), "n": len(rows)}
    m = compute_detection_metrics(labels, scores)
    return {"auroc": m.auroc, "auprc": m.auprc, "acc_star": m.acc_star, "n": m.n}


def _fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}" if x == x else "—"


def build_markdown(
    results: dict[str, dict[str, dict[str, float]]],
    methods: list[tuple[str, str, str]],
    datasets: list[str],
) -> str:
    ds_labels = {
        "minerva": "Minerva",
        "math500": "MATH-500",
        "gsm8k": "GSM8K",
        "deepscaler": "DeepScaler",
    }
    header = ["Method"]
    sep = ["---"]
    for ds in datasets:
        label = ds_labels.get(ds, ds)
        header.extend([f"{label} AUROC", f"{label} AUPRC", f"{label} ACC*"])
        sep.extend(["---:", "---:", "---:"])

    lines = [
        "# 主结果表（Clean labels：AUROC / AUPRC / ACC*）",
        "",
        "ACC* = Top-50% accuracy（TokUR 协议：按**置信度**取前 50% 最自信样本的正确率，等价于对 EU/TU 等取负后降序）。",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for name, key, _ in methods:
        cells = [name]
        for ds in datasets:
            m = results.get(ds, {}).get(name, {})
            cells.extend([_fmt(m.get("auroc", float("nan"))), _fmt(m.get("auprc", float("nan"))), _fmt(m.get("acc_star", float("nan")))])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_latex(
    results: dict[str, dict[str, dict[str, float]]],
    methods: list[tuple[str, str, str]],
    datasets: list[str],
    caption: str,
    label: str,
) -> str:
    ds_labels = {
        "minerva": "Minerva",
        "math500": "MATH-500",
        "gsm8k": "GSM8K",
        "deepscaler": "DeepScaler",
    }
    ncol = 1 + 3 * len(datasets)
    col_spec = "l|" + "|".join(["ccc"] * len(datasets))

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]
    hdr1 = ["", *sum([[f"\\multicolumn{{3}}{{c}}{{{ds_labels.get(ds, ds)}}}"] for ds in datasets], [])]
    lines.append(" & ".join(hdr1) + " \\\\")
    cmid = []
    start = 2
    for _ in datasets:
        cmid.append(f"\\cmidrule(lr){{{start}-{start+2}}}")
        start += 3
    lines.append(" ".join(cmid))
    hdr2 = ["Method"] + ["AUROC", "AUPRC", "ACC*"] * len(datasets)
    lines.append(" & ".join(hdr2) + " \\\\")
    lines.append("\\midrule")

    for i, (name, key, _) in enumerate(methods):
        row = [name.replace("_", "\\_")]
        for ds in datasets:
            m = results.get(ds, {}).get(name, {})
            row.extend([_fmt(m.get("auroc", float("nan"))), _fmt(m.get("auprc", float("nan"))), _fmt(m.get("acc_star", float("nan")))])
        line = " & ".join(row) + " \\\\"
        if name == "TokUR EU":
            line += " \\midrule"
        lines.append(line)

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER)
    ap.add_argument("--datasets", default="minerva,math500,gsm8k,deepscaler")
    ap.add_argument(
        "--features-only",
        action="store_true",
        help="Use features.jsonl only (fast; skips raw_runs token enrichment)",
    )
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    def _load(ds: str) -> list[dict]:
        if args.features_only:
            return load_features_table(args.out_dir, ds)
        return load_enriched(args.out_dir, ds)
    paper = args.paper_dir
    tables_dir = paper / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, dict[str, float]]] = {}
    stats: dict[str, dict] = {}
    for ds in datasets:
        rows = _load(ds)
        results[ds] = {}
        n = len(rows)
        wrong = sum(r["label_wrong_clean"] for r in rows)
        stats[ds] = {"n": n, "wrong": wrong, "acc": 1 - wrong / n, "relabeled": sum(r.get("relabeled", False) for r in rows)}
        for name, key, _ in PAPER_EXTENDED:
            invert = key in TABLE2_INVERT or key in BASELINE_INVERT_KEYS
            if key not in rows[0] and key != "tokur_eu_sum":
                continue
            if key == "tokur_eu_sum" and not any("tokur_eu_sum" in r for r in rows):
                continue
            results[ds][name] = eval_clean(rows, key, invert=invert)

    # Token table (AUROC only in extended md)
    token_results = {ds: {} for ds in datasets}
    for ds in datasets:
        rows = _load(ds)
        for name, key, _ in TABLE2:
            token_results[ds][name] = eval_clean(rows, key, invert=(key in TABLE2_INVERT))

    main_md = build_markdown(results, PAPER_MAIN, datasets)
    ext_md = build_markdown(results, PAPER_EXTENDED, datasets)
    main_tex = build_latex(
        results,
        PAPER_MAIN,
        datasets,
        caption="Clean-label detection on three math benchmarks (AUROC / AUPRC / ACC*). ACC* denotes Top-50\\% accuracy.",
        label="tab:main",
    )
    ext_tex = build_latex(
        results,
        PAPER_EXTENDED,
        datasets,
        caption="Extended answer-level fragmentation metrics (clean labels).",
        label="tab:extended",
    )

    (tables_dir / "table_main.md").write_text(main_md, encoding="utf-8")
    (tables_dir / "table_extended.md").write_text(ext_md, encoding="utf-8")
    (tables_dir / "table_main.tex").write_text(main_tex, encoding="utf-8")
    (tables_dir / "table_extended.tex").write_text(ext_tex, encoding="utf-8")
    (tables_dir / "results.json").write_text(
        json.dumps({"datasets": stats, "metrics": results, "token_auroc": token_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(main_md)
    print(f"Wrote {tables_dir}/table_main.{{md,tex}}")
    print(f"Wrote {tables_dir}/table_extended.{{md,tex}}")


if __name__ == "__main__":
    main()