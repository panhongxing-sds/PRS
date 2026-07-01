"""主图：六模型 n-sweep 三联图 — AUROC / CW / 能力–CW。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

from analyze import analyze

ROOT = Path(__file__).resolve().parent
MAIN_BENCHES = ["deepscaler", "gpqa_diamond", "aime_2024"]
MODELS = [
    ("qwen25_05b", "Qwen-0.5B", "Qwen2.5-0.5B"),
    ("llama32_1b", "Llama-1B", "Llama-3.2-1B"),
    ("qwen25_15b", "Qwen-1.5B", "Qwen2.5-1.5B"),
    ("phi4_mini", "Phi-3.8B", "Phi-4-mini"),
    ("qwen25_3b", "Qwen-3B", "Qwen2.5-3B"),
    ("qwen25_7b", "Qwen-7B", "Qwen2.5-7B"),
]
NS = [2, 4, 8, 16, 32, 64]
CACHE_PATH = ROOT / "results" / "nsweep_6models_plot.json"

# 按能力从弱到强
MODEL_COLORS = {
    "Qwen-0.5B": "#4DBBD5",
    "Llama-1B": "#00A087",
    "Qwen-1.5B": "#3C5488",
    "Phi-3.8B": "#8491B4",
    "Qwen-3B": "#F39B7F",
    "Qwen-7B": "#E64B35",
}

PLOT_CONFIG = {
    "labelsize": 20,
    "ticksize": 18,
    "titlesize": 20,
    "legendsize": 18,
    "linewidth": 2.5,
    "markersize": 10,
    "scatter_size": 480,
    "annotsize": 22,
    "y1_range": (0.64, 0.88),
    "y2_range": (0.0, 0.35),
}

CAP_SHORT_LABELS = {
    "Qwen-0.5B": "0.5B",
    "Llama-1B": "1B",
    "Qwen-1.5B": "1.5B",
    "Phi-3.8B": "3.8B",
    "Qwen-3B": "3B",
    "Qwen-7B": "7B",
}
# panel (c) 标签：相对散点的像素偏移 (ox, oy)
CAP_ANNOT_OFFSETS = {
    "Qwen-0.5B": (0, 18, "center", "bottom"),
    "Llama-1B": (16, 2, "left", "center"),
    "Qwen-1.5B": (-18, 0, "right", "center"),
    "Phi-3.8B": (16, 0, "left", "center"),
    "Qwen-3B": (16, 0, "left", "center"),
    "Qwen-7B": (-18, 0, "right", "center"),
}


def set_style() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["axes.linewidth"] = 2.5
    mpl.rcParams["axes.edgecolor"] = "black"
    mpl.rcParams["axes.labelsize"] = PLOT_CONFIG["labelsize"]
    mpl.rcParams["xtick.labelsize"] = PLOT_CONFIG["ticksize"]
    mpl.rcParams["ytick.labelsize"] = PLOT_CONFIG["ticksize"]


def load_model_data() -> tuple[dict[str, dict[str, dict]], set[str]]:
    all_data: dict[str, dict[str, dict]] = {}
    for tag, _, _ in MODELS:
        d: dict[str, dict] = {}
        for f in sorted((ROOT / "data" / "samples").glob(f"samples_{tag}_seed41_*.jsonl")):
            if str(f).endswith(".bak") or ".dryrun" in str(f) or ".preshard" in str(f):
                continue
            bench = f.name.replace(f"samples_{tag}_seed41_", "").replace(".jsonl", "")
            bench = bench.split(".shard")[0].split(".s3_")[0]
            if bench not in MAIN_BENCHES:
                continue
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        d[r["id"]] = r
        all_data[tag] = d
    common = set.intersection(*[set(d) for d in all_data.values()])
    return all_data, common


def rows_for_ids(data: dict[str, dict], qids: set[str]) -> list[dict]:
    rows = []
    for qid in qids:
        r = data[qid]
        if int(r.get("label_drop", 0)) == 1:
            continue
        pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a]
        if len(pairs) < 2:
            continue
        a, c = zip(*pairs)
        rows.append({
            "id": qid,
            "dataset": r.get("benchmark") or r.get("dataset", ""),
            "seed": r.get("seed", 41),
            "answers": list(a),
            "correct": list(c),
        })
    return rows


def compute_nsweep_curves(cfg_an: dict, *, use_cache: bool = True) -> dict[str, dict]:
    if use_cache and CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    all_data, common = load_model_data()
    print(f"computing n-sweep for 6 models, common N={len(common)} ...", flush=True)
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {"common_n": len(common), "models": {}}
    for tag, lab, _ in MODELS:
        rows = rows_for_ids(all_data[tag], common)
        metrics = analyze(rows, cfg_an, rng)
        out["models"][lab] = {
            "auroc": [metrics["per_n"][str(n)]["auroc_uptop"][0] for n in NS],
            "cw": [metrics["per_n"][str(n)]["spurious_consensus_rate"]["0.9"][0] for n in NS],
            "n_questions": metrics["n_questions"],
        }
        print(f"  {lab}: n={metrics['n_questions']}", flush=True)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(CACHE_PATH, "w", encoding="utf-8"), indent=2)
    return out


def load_capability_points() -> tuple[list[float], list[float], list[str]]:
    all_data, common = load_model_data()
    xs, ys, labs = [], [], []
    for tag, lab, _ in MODELS:
        items = []
        for qid in common:
            r = all_data[tag][qid]
            if int(r.get("label_drop", 0)) == 1:
                continue
            pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a]
            if len(pairs) < 2:
                continue
            a, c = zip(*pairs)
            cnt = Counter(a)
            maj, top = cnt.most_common(1)[0]
            p_top = top / len(a)
            cmap = {ai: ci for ai, ci in zip(a, c)}
            wrong = cmap.get(maj, 0) == 0
            items.append({"wrong": wrong, "p_top": p_top})
        if not items:
            continue
        wrongs = [x for x in items if x["wrong"]]
        cw = sum(1 for x in wrongs if x["p_top"] >= 0.9) / max(len(wrongs), 1)
        xs.append((1 - np.mean([x["wrong"] for x in items])) * 100)
        ys.append(cw * 100)
        labs.append(lab)
    return xs, ys, labs


def _panel_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(
        text,
        loc="center",
        pad=12,
        fontsize=PLOT_CONFIG["titlesize"],
        fontfamily="serif",
    )


def _style_line_axis(ax: plt.Axes, x: np.ndarray, ns: list[int]) -> None:
    ax.set_xlim(-0.15, len(ns) - 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.grid(ls="--", color="gray", alpha=0.4)
    ax.tick_params(bottom=True, top=False, left=True, right=False)
    ax.set_xlabel("Number of Samples", labelpad=10)


def _draw_multi_lines(
    ax: plt.Axes,
    x: np.ndarray,
    curves: dict[str, dict],
    key: str,
) -> None:
    for tag, lab, legend_lab in MODELS:
        y = curves["models"][lab][key]
        color = MODEL_COLORS[lab]
        ax.plot(
            x, y, "-o", color=color, label=legend_lab,
            linewidth=PLOT_CONFIG["linewidth"],
            markersize=PLOT_CONFIG["markersize"],
            clip_on=True,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )


def _draw_capability(ax: plt.Axes, xs: list[float], ys: list[float], labs: list[str]) -> None:
    for x, y, lab in zip(xs, ys, labs):
        ax.scatter(
            x, y,
            s=PLOT_CONFIG["scatter_size"],
            c=MODEL_COLORS[lab],
            zorder=3,
            edgecolors="white",
            linewidths=1.5,
        )
        short = CAP_SHORT_LABELS.get(lab, lab)
        ox, oy, ha, va = CAP_ANNOT_OFFSETS.get(lab, (14, 0, "left", "center"))
        ax.annotate(
            short,
            (x, y),
            textcoords="offset points",
            xytext=(ox, oy),
            fontsize=PLOT_CONFIG["annotsize"],
            fontfamily="serif",
            ha=ha,
            va=va,
            annotation_clip=True,
        )
    ax.set_xlabel("majority@64 accuracy (%)", labelpad=10)
    ax.set_ylabel("CW@0.9", labelpad=10)
    ax.set_xlim(23, 53)
    ax.set_ylim(-0.5, 12.5)
    ax.set_xticks([25, 30, 35, 40, 45, 50])
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.grid(ls="--", color="gray", alpha=0.4)
    ax.tick_params(bottom=True, top=False, left=True, right=False)


def create_plot(
    output: Path = Path("figures/fig_main.pdf"),
    *,
    use_cache: bool = True,
    bootstrap: int | None = None,
) -> None:
    set_style()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    cfg_an = cfg["analysis"]
    if bootstrap is not None:
        cfg_an = {**cfg_an, "bootstrap_draws": bootstrap}

    curves = compute_nsweep_curves(cfg_an, use_cache=use_cache and bootstrap is None)
    cap_x, cap_y, cap_labs = load_capability_points()
    x = np.arange(len(NS), dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.2), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.26, top=0.84, wspace=0.32)

    ax_a, ax_b, ax_c = axes

    _draw_multi_lines(ax_a, x, curves, "auroc")
    ax_a.set_ylabel("AUROC", labelpad=10)
    ax_a.set_ylim(PLOT_CONFIG["y1_range"])
    ax_a.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    _style_line_axis(ax_a, x, NS)
    _panel_title(ax_a, "(a) Answer-level UQ saturates\nwith sampling")

    _draw_multi_lines(ax_b, x, curves, "cw")
    ax_b.set_ylabel("CW@0.9", labelpad=10)
    ax_b.set_ylim(PLOT_CONFIG["y2_range"])
    ax_b.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    _style_line_axis(ax_b, x, NS)
    _panel_title(ax_b, "(b) More samples do not\neliminate confident wrongs")

    _draw_capability(ax_c, cap_x, cap_y, cap_labs)
    _panel_title(ax_c, "(c) Stronger models are\nmore confidently wrong")

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=6,
        fontsize=PLOT_CONFIG["legendsize"],
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        markerscale=1.2,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(output.parent / "fig2_panels.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(output.parent / "fig_main_triple.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"common N={curves['common_n']}")
    print(f"saved {output}, {output.with_suffix('.png')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/fig_main.pdf")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args()
    create_plot(
        output=Path(args.out),
        use_cache=not args.no_cache,
        bootstrap=args.bootstrap,
    )


if __name__ == "__main__":
    main()
