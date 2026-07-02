#!/usr/bin/env python3
"""Merge per-model results.json into one four-model AUROC/AUPRC/ACC* table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_MODELS = ROOT / "paper" / "models"
OUT_TABLE = ROOT / "paper" / "tables" / "four_models_main.md"

MODEL_ORDER = ("qwen25_3b", "llama32_1b", "llama31_8b", "qwen3_8b")
MODEL_LABEL = {
    "qwen25_3b": "Qwen2.5-3B",
    "llama32_1b": "Llama-3.2-1B",
    "llama31_8b": "Llama-3.1-8B",
    "qwen3_8b": "Qwen3-8B",
}
DATASETS = ("minerva", "math500", "gsm8k", "deepscaler")
DS_LABEL = {
    "minerva": "Minerva",
    "math500": "MATH-500",
    "gsm8k": "GSM8K",
    "deepscaler": "DeepScaler",
}
# TokUR + top ASE methods for cross-model summary
METHODS = ("TokUR EU", "PANDA", "TW-ASE", "F_resp")


def _fmt(x: float) -> str:
    return f"{x:.3f}" if x == x else "—"


def load_model_results(tag: str) -> dict | None:
    p = PAPER_MODELS / tag / "tables" / "results.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_combined_md(by_model: dict[str, dict]) -> str:
    lines = [
        "# Four-model main results (AUROC / AUPRC / ACC*)",
        "",
        "ACC* = Top-50% accuracy (TokUR protocol).",
        "Partial datasets are reported as-is (available raw/features, no 400/272 gate).",
        "",
    ]
    for method in METHODS:
        lines.append(f"## {method}")
        lines.append("")
        header = ["Model"]
        sep = ["---"]
        for ds in DATASETS:
            lbl = DS_LABEL[ds]
            header.extend([f"{lbl} AUROC", f"{lbl} AUPRC", f"{lbl} ACC*"])
            sep.extend(["---:", "---:", "---:"])
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(sep) + " |")
        for tag in MODEL_ORDER:
            data = by_model.get(tag)
            if not data:
                continue
            row = [MODEL_LABEL.get(tag, tag)]
            metrics = data.get("metrics", {})
            for ds in DATASETS:
                m = metrics.get(ds, {}).get(method, {})
                row.extend([
                    _fmt(m.get("auroc", float("nan"))),
                    _fmt(m.get("auprc", float("nan"))),
                    _fmt(m.get("acc_star", float("nan"))),
                ])
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Write combined four-model table")
    ap.add_argument("--single", default="", help="Model tag (log only)")
    args = ap.parse_args()

    by_model: dict[str, dict] = {}
    for tag in MODEL_ORDER:
        data = load_model_results(tag)
        if data:
            by_model[tag] = data
            print(f"loaded {tag} n_datasets={len(data.get('metrics', {}))}")
        else:
            print(f"missing {tag}")

    if args.all or not args.single:
        OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
        md = build_combined_md(by_model)
        OUT_TABLE.write_text(md, encoding="utf-8")
        print(f"Wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
