#!/usr/bin/env python3
"""Motivation figure: SC@9 vs PANDA@9 confident-wrong rates (fair 9-decode vote)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "paper/analysis/random300_confident_wrong.json"
DEFAULT_PNG = ROOT / "paper/analysis/figures/random300_confident_wrong.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_PNG)
    ap.add_argument("--subset", choices=("n300", "n228_eval_no_label_drop"), default="n300")
    args = ap.parse_args()

    doc = json.loads(args.json.read_text())
    m = doc["metrics"][args.subset]
    tau_order = ["tau_0.9", "tau_8_over_9", "tau_7_over_9"]
    tau_labels = ["$p_{\\top}\\geq 0.9$", "$\\geq 8/9$", "$\\geq 7/9$"]

    sc_rates = [m["confident_wrong_sc@9"][k]["confident_wrong_rate_all"] * 100 for k in tau_order]
    panda_rates = [
        m["confident_wrong_panda@9_vote"][k]["confident_wrong_rate_all"] * 100 for k in tau_order
    ]
    collapse = m["panda_collapse_a0_wrong_bd0"]["rate_all"] * 100

    x = np.arange(len(tau_order))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=150)
    b1 = ax.bar(x - w / 2, sc_rates, w, label="SC@9 (majority vote)", color="#8491B4")
    b2 = ax.bar(x + w / 2, panda_rates, w, label="PANDA@9 (majority vote)", color="#E64B35")
    ax.axhline(
        collapse,
        color="#4DBBD5",
        linestyle="--",
        linewidth=1.5,
        label=f"PANDA $a_0$ wrong, $bd{{=}}0$ ({collapse:.1f}%)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tau_labels)
    ax.set_ylabel("Confident-wrong rate (% of all questions)")
    ax.set_xlabel("Consensus threshold on 9 decodes")
    sc_acc = m["accuracy"]["sc9_majority_vote"] * 100
    pv_acc = m["accuracy"]["panda9_majority_vote"] * 100
    ax.set_title(
        f"DeepScaler random300 (Qwen2.5-3B): "
        f"maj@9 acc SC {sc_acc:.1f}% vs PANDA {pv_acc:.1f}%"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(sc_rates + panda_rates + [collapse]) * 1.25 + 2)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.4,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
