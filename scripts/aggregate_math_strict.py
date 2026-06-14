#!/usr/bin/env python
"""Strict offline aggregation of math main-table metrics.

Reads the strict labels already stored in summary.jsonl (written by the pipeline
via `prs.grading.grade_answer`, which uses the strict three-way grader):
  - `label_drop == 1`  -> sample could not be confidently graded; EXCLUDED.
  - `label_wrong_clean` -> 0 correct / 1 wrong on the kept subset.
Computes AUROC / AUPRC / ACC* on the kept subset (y = 1 means WRONG, i.e. the
positive class for error detection).

If summary.jsonl predates the strict pipeline (no `label_drop` field), run
`recompute_metrics --from-cache` first to refresh labels.

Excluded by request: SE, U_Ecc, U_Deg (non-standard implementations) and CoT.

Outputs a markdown file with one table per seed plus a 3-seed mean +/- std table.
No GPU / model calls -- operates purely on summary.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# (display name, summary.jsonl key, invert)
# invert=True: higher score = more confident/correct, so negate for error detection.
METHODS = [
    ("SAR", "baseline_SAR", False),
    ("PE", "baseline_PE_mean", False),
    ("LL", "baseline_LL_nll", False),
    ("Self-Certainty", "baseline_SC_mean", True),
    ("DeepConf", "baseline_DC_min", True),
    ("PRS (Ours)", "PRS", False),
    ("F_resp", "F_resp", False),
    ("D_ans", "D_ans", False),
    ("D_reason", "D_reason", False),
]

DS_LABEL = {
    "math500": "MATH-500", "gsm8k": "GSM8K", "minerva": "Minerva",
    "color_cube": "Color Cube", "leg_counting": "Leg Counting", "zebra_puzzles": "Zebra",
}


def _ds_label(d: str) -> str:
    return DS_LABEL.get(d, d)


def load(base: Path, seed: int, ds: str) -> list[dict]:
    p = base / f"seed{seed}" / ds / "summary.jsonl"
    rows = []
    if not p.exists():
        return rows
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def graded_rows(rows: list[dict]) -> list[dict]:
    """Keep samples the strict grader could confidently judge (label_drop != 1);
    use the stored strict clean label (label_wrong_clean)."""
    kept = []
    for r in rows:
        if r.get("label_drop", 0):
            continue
        lw = r.get("label_wrong_clean")
        if lw is None:
            continue
        r = dict(r)
        r["_y_wrong"] = 1 if int(lw) == 1 else 0
        kept.append(r)
    return kept


def metric(rows: list[dict], key: str, invert: bool):
    y, s = [], []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        y.append(r["_y_wrong"])
        s.append(-v if invert else v)
    if len(set(y)) < 2:
        return None
    y = np.array(y)
    s = np.array(s)
    order = np.argsort(s)  # ascending score = most confident first
    k = len(s) // 2
    sel = order[:k]
    acc_star = float(np.mean(y[sel] == 0))
    return (
        float(roc_auc_score(y, s)),
        float(average_precision_score(y, s)),
        acc_star,
    )


def fmt(x: float) -> str:
    return f"{x*100:.2f}"


def fmt_ms(xs: list[float]) -> str:
    return f"{np.mean(xs)*100:.2f} ± {np.std(xs)*100:.2f}"


def build(base: Path, seeds: list[int], ds_list: list[str]) -> str:
    lines: list[str] = []

    # cache graded rows per (seed, ds)
    cache: dict[tuple[int, str], list[dict]] = {}
    stats: dict[tuple[int, str], tuple[int, int]] = {}  # (n_total, n_kept)
    for seed in seeds:
        for ds in ds_list:
            raw = load(base, seed, ds)
            g = graded_rows(raw)
            cache[(seed, ds)] = g
            stats[(seed, ds)] = (len(raw), len(g))

    def header() -> list[str]:
        h = "| Method | " + " | ".join(
            f"{_ds_label(d)} AUROC | AUPRC | ACC*" for d in ds_list) + " |"
        sep = "|--------|" + "|".join(["--:|--:|--:"] * len(ds_list)) + "|"
        return [h, sep]

    # ---- drop-rate summary ----
    lines.append("#### 样本剔除统计（strict_grade=drop 的题目被排除）")
    lines.append("")
    lines.append("| seed | " + " | ".join(f"{_ds_label(d)} 保留/总数 (剔除%)" for d in ds_list) + " |")
    lines.append("|--|" + "|".join(["--:"] * len(ds_list)) + "|")
    for seed in seeds:
        cells = []
        for ds in ds_list:
            tot, kept = stats[(seed, ds)]
            drp = (tot - kept) / tot * 100 if tot else 0
            cells.append(f"{kept}/{tot} ({drp:.1f}%)")
        lines.append(f"| seed{seed} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- per-seed metric tables ----
    for seed in seeds:
        lines.append(f"#### seed{seed}（strict label, 子集）")
        lines.append("")
        lines += header()
        for name, key, inv in METHODS:
            cells = []
            for ds in ds_list:
                m = metric(cache[(seed, ds)], key, inv)
                cells.append(f"{fmt(m[0])} | {fmt(m[1])} | {fmt(m[2])}" if m else "— | — | —")
            bold = "**" if name.startswith("PRS") else ""
            lines.append(f"| {bold}{name}{bold} | " + " | ".join(cells) + " |")
        lines.append("")

    # ---- 3-seed mean table ----
    lines.append("#### 3-seed 平均 (mean ± std, strict label, 子集)")
    lines.append("")
    lines += header()
    for name, key, inv in METHODS:
        cells = []
        for ds in ds_list:
            a, p, c = [], [], []
            for seed in seeds:
                m = metric(cache[(seed, ds)], key, inv)
                if m:
                    a.append(m[0])
                    p.append(m[1])
                    c.append(m[2])
            cells.append(f"{fmt_ms(a)} | {fmt_ms(p)} | {fmt_ms(c)}" if a else "— | — | —")
        bold = "**" if name.startswith("PRS") else ""
        lines.append(f"| {bold}{name}{bold} | " + " | ".join(cells) + " |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/prs-outputs/maintable_qwen25_3b")
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--datasets", default="math500,gsm8k,minerva")
    ap.add_argument("--out", default="/root/autodl-tmp/_math_tables_strict.md")
    args = ap.parse_args()

    base = Path(args.base)
    seeds = [int(x) for x in args.seeds.split(",")]
    ds_list = [x.strip() for x in args.datasets.split(",")]

    md = build(base, seeds, ds_list)
    Path(args.out).write_text(md)
    print(md)
    print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
