#!/usr/bin/env python3
"""六模型统一 cleaned/canonicalized 指标 + n-sweep + confidence collapse + 出图。"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from analyze import analyze

ROOT = Path(__file__).resolve().parent
MAIN_BENCHES = ["deepscaler", "gpqa_diamond", "aime_2024"]
MODELS = [
    ("qwen25_05b", "Qwen-0.5B"),
    ("llama32_1b", "Llama-1B"),
    ("qwen25_15b", "Qwen-1.5B"),
    ("phi4_mini", "Phi-3.8B"),
    ("qwen25_3b", "Qwen-3B"),
    ("qwen25_7b", "Qwen-7B"),
]
SCR_TAUS = [0.9, 0.95, 1.0]
EXO_BINS = [
    ("极难 (<10%)", 0.0, 0.10),
    ("难 (10–25%)", 0.10, 0.25),
    ("中 (25–45%)", 0.25, 0.45),
    ("易 (>45%)", 0.45, 1.01),
]


def load_model(tag: str, seed: int = 41) -> dict[str, dict]:
    d: dict[str, dict] = {}
    for f in sorted((ROOT / "data" / "samples").glob(f"samples_{tag}_seed{seed}_*.jsonl")):
        if str(f).endswith(".bak") or ".dryrun" in str(f) or ".preshard" in str(f):
            continue
        bench = f.name.replace(f"samples_{tag}_seed{seed}_", "").replace(".jsonl", "")
        bench = bench.split(".shard")[0].split(".s3_")[0]
        if bench not in MAIN_BENCHES:
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    d[r["id"]] = r
    return d


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
            "benchmark": r.get("benchmark") or r.get("dataset", ""),
            "seed": r.get("seed", 41),
            "answers": list(a),
            "correct": list(c),
            "gold": r["gold"],
            "grading": r.get("grading", "math"),
        })
    return rows


def pooled_stats(rows: list[dict]) -> dict:
    items = []
    for r in rows:
        cnt = Counter(r["answers"])
        maj, top = cnt.most_common(1)[0]
        p_top = top / len(r["answers"])
        cmap = {a: c for a, c in zip(r["answers"], r["correct"])}
        wrong = cmap.get(maj, 0) == 0
        items.append({"wrong": wrong, "p_top": p_top, "bench": r["benchmark"]})

    if not items:
        return {}

    labels = [int(x["wrong"]) for x in items]
    wrongs = [x for x in items if x["wrong"]]
    scores = [1 - x["p_top"] for x in items]
    auroc = float(roc_auc_score(labels, scores))
    conf = [x for x in items if x["p_top"] >= 0.9]

    def scr(tau: float) -> int:
        return sum(1 for x in wrongs if x["p_top"] >= tau)

    out = {
        "n": len(items),
        "maj_at_64": 1 - np.mean(labels),
        "auroc": auroc,
        "wrong_n": len(wrongs),
        "cov_p09": len(conf) / len(items),
        "sel_acc_p09": 1 - np.mean([x["wrong"] for x in conf]) if conf else float("nan"),
    }
    for tau in SCR_TAUS:
        s = scr(tau)
        out[f"scr_{int(tau*100)}"] = s
        out[f"scr_{int(tau*100)}_pct_wrong"] = s / max(len(wrongs), 1)
    return out


def exogenous_rates(all_data: dict[str, dict[str, dict]], qids: set[str]) -> dict[str, float]:
    """每题：所有模型逐样本正确率均值（非 leave-one-out，用于统一难度锚）。"""
    acc = defaultdict(list)
    for tag in all_data:
        for qid in qids:
            r = all_data[tag].get(qid)
            if not r or int(r.get("label_drop", 0)) == 1:
                continue
            pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a]
            if not pairs:
                continue
            _, c = zip(*pairs)
            acc[qid].append(sum(c) / len(c))
    return {qid: statistics.mean(v) for qid, v in acc.items() if v}


def exogenous_loo(all_data: dict[str, dict[str, dict]], target: str, qids: set[str]) -> dict[str, float]:
    out = {}
    others = [t for t in all_data if t != target]
    for qid in qids:
        rates = []
        for tag in others:
            r = all_data[tag].get(qid)
            if not r or int(r.get("label_drop", 0)) == 1:
                continue
            pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a]
            if pairs:
                _, c = zip(*pairs)
                rates.append(sum(c) / len(c))
        if rates:
            out[qid] = statistics.mean(rates)
    return out


def confidence_collapse(rows: list[dict], exo: dict[str, float]) -> list[dict]:
    bins_out = []
    for label, lo, hi in EXO_BINS:
        subset = []
        for r in rows:
            qid = r["id"]
            if qid not in exo or not (lo <= exo[qid] < hi):
                continue
            cnt = Counter(r["answers"])
            maj, top = cnt.most_common(1)[0]
            p_top = top / len(r["answers"])
            if p_top < 0.9:
                continue
            cmap = {a: c for a, c in zip(r["answers"], r["correct"])}
            subset.append(cmap.get(maj, 0) == 1)
        bins_out.append({
            "bin": label,
            "n": len(subset),
            "reliability": float(np.mean(subset)) if subset else float("nan"),
        })
    return bins_out


def plot_nsweep(nsweep: dict, out: Path) -> None:
    ns = [2, 4, 8, 16, 32, 64]
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, (tag, lab) in enumerate(MODELS):
        pn = nsweep[tag]["per_n"]
        auroc = [pn[str(n)]["auroc_uptop"][0] for n in ns]
        scr = [pn[str(n)]["spurious_consensus_rate"]["0.9"][0] for n in ns]
        axes[0].plot(ns, auroc, "-o", label=lab, color=colors[i], linewidth=2, markersize=5)
        axes[1].plot(ns, scr, "-o", label=lab, color=colors[i], linewidth=2, markersize=5)
    for ax, title, ylab in [
        (axes[0], "(a) AUROC(1−p_top) saturates", "AUROC"),
        (axes[1], "(b) CW@0.9 persists", "CW@0.9 (fraction of wrong@n)"),
    ]:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("sampling budget n")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Six models, cleaned/canonicalized scoring, common N=2228", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_capability(summary: dict, out: Path) -> None:
    xs, ys, labs = [], [], []
    for tag, lab in MODELS:
        s = summary[tag]["common"]
        xs.append(s["maj_at_64"] * 100)
        ys.append(s["scr_90_pct_wrong"] * 100)
        labs.append(lab)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=120, c="#2166ac", zorder=3)
    for x, y, lab in zip(xs, ys, labs):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("majority@64 accuracy (%)")
    ax.set_ylabel("CW@0.9 (% of wrong questions)")
    ax.set_title("Capability ↑ → Confident Wrong ↑ (cleaned, common set)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_confidence_collapse(collapse: dict, out: Path) -> None:
    bin_labels = [b[0] for b in EXO_BINS]
    x = np.arange(len(bin_labels))
    width = 0.12
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (tag, lab) in enumerate(MODELS):
        ys = []
        ns = []
        for b in collapse[tag]:
            ys.append(b["reliability"] * 100 if b["n"] else float("nan"))
            ns.append(b["n"])
        offset = (i - len(MODELS) / 2) * width + width / 2
        bars = ax.bar(x + offset, ys, width, label=lab, alpha=0.85)
        for bar, n in zip(bars, ns):
            if n:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        str(n), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylabel("Actual majority accuracy among p_top≥0.9 (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Confidence collapse by exogenous difficulty (LOO, cleaned)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def render_report_tables(payload: dict) -> str:
    lines = [
        "<!-- AUTO-GENERATED by analyze_all_models.py -->",
        "",
        "### 六模型主表（cleaned，共同题 N=2228，剔除 label_drop）",
        "",
        "| 模型 | maj@64 | AUROC | SCR@0.9 | SCR@0.95 | SCR@1.0 | p≥0.9 覆盖 | p≥0.9 可靠 |",
        "|------|-------:|------:|--------:|---------:|--------:|-----------:|-----------:|",
    ]
    for tag, lab in MODELS:
        s = payload["summary"][tag]["common"]
        lines.append(
            f"| {lab} | {s['maj_at_64']*100:.1f}% | {s['auroc']:.3f} "
            f"| {s['scr_90']} ({s['scr_90_pct_wrong']*100:.1f}%错) "
            f"| {s['scr_95']} ({s['scr_95_pct_wrong']*100:.1f}%错) "
            f"| {s['scr_100']} ({s['scr_100_pct_wrong']*100:.1f}%错) "
            f"| {s['cov_p09']*100:.1f}% | {s['sel_acc_p09']*100:.1f}% |"
        )

    lines += ["", "### n-sweep @64（cleaned，共同题）", ""]
    lines.append("| 模型 | maj@64 | AUROC | SCR@0.9 |")
    lines.append("|------|-------:|------:|--------:|")
    for tag, lab in MODELS:
        pn = payload["nsweep"][tag]["per_n"]["64"]
        maj = pn["majority_acc"][0]
        auroc = pn["auroc_uptop"][0]
        scr = pn["spurious_consensus_rate"]["0.9"][0]
        lines.append(f"| {lab} | {maj*100:.1f}% | {auroc:.3f} | {scr*100:.1f}% |")

    lines += ["", "### benchmark 分解（cleaned，各模型全量有效题）", ""]
    lines.append("| 模型 | benchmark | N | maj@64 | AUROC | SCR@0.9占错 | SCR@1.0占错 |")
    lines.append("|------|-----------|--:|-------:|------:|------------:|------------:|")
    for tag, lab in MODELS:
        for bench in MAIN_BENCHES:
            s = payload["summary"][tag]["by_benchmark"].get(bench)
            if not s:
                continue
            lines.append(
                f"| {lab} | {bench} | {s['n']} | {s['maj_at_64']*100:.1f}% | {s['auroc']:.3f} "
                f"| {s['scr_90_pct_wrong']*100:.1f}% | {s['scr_100_pct_wrong']*100:.1f}% |"
            )

    lines += ["", "### confidence collapse（p_top≥0.9 子集，外生难度 LOO）", ""]
    hdr = "| 模型 | " + " | ".join(b[0] for b in EXO_BINS) + " |"
    sep = "|------|" + "|".join(["---:"] * len(EXO_BINS)) + "|"
    lines += [hdr, sep]
    for tag, lab in MODELS:
        cells = []
        for b in payload["confidence_collapse"][tag]:
            if b["n"]:
                cells.append(f"{b['reliability']*100:.0f}% (n={b['n']})")
            else:
                cells.append("—")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--skip-nsweep", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    cfg_an = cfg["analysis"]
    rng = np.random.default_rng(0)

    all_data = {tag: load_model(tag, args.seed) for tag, _ in MODELS}
    common = set.intersection(*[set(d) for d in all_data.values()])
    print(f"共同题 N={len(common)}")

    summary = {}
    nsweep_out = {}
    collapse_out = {}

    for tag, lab in MODELS:
        rows_common = rows_for_ids(all_data[tag], common)
        rows_all = rows_for_ids(all_data[tag], set(all_data[tag]) - {
            q for q, r in all_data[tag].items() if int(r.get("label_drop", 0)) == 1
        })
        by_bench = {}
        for bench in MAIN_BENCHES:
            br = [r for r in rows_all if r["benchmark"] == bench]
            if br:
                by_bench[bench] = pooled_stats(br)
        summary[tag] = {
            "label": lab,
            "common": pooled_stats(rows_common),
            "by_benchmark": by_bench,
        }
        exo_loo = exogenous_loo(all_data, tag, common)
        collapse_out[tag] = confidence_collapse(rows_common, exo_loo)

        if not args.skip_nsweep:
            loaded = rows_for_ids(all_data[tag], common)
            nsweep_out[tag] = analyze(loaded, cfg_an, rng)
            print(f"  n-sweep {lab}: n={nsweep_out[tag]['n_questions']}")

    out_dir = ROOT / "results"
    fig_dir = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    payload = {
        "protocol": "cleaned/canonicalized via clean_samples.py; drop_label_drop=true",
        "common_n": len(common),
        "models": [lab for _, lab in MODELS],
        "summary": summary,
        "nsweep": nsweep_out,
        "confidence_collapse": collapse_out,
    }
    json.dump(payload, open(out_dir / "all_models_clean_metrics.json", "w"),
              ensure_ascii=False, indent=2, default=float)

    if nsweep_out:
        plot_nsweep(nsweep_out, fig_dir / "nsweep_irreducible.png")
        json.dump(nsweep_out, open(out_dir / "all_models_nsweep.json", "w"),
                  ensure_ascii=False, indent=2, default=float)
    plot_capability(summary, fig_dir / "capability_vs_scr.png")
    plot_confidence_collapse(collapse_out, fig_dir / "confidence_collapse.png")

    md = render_report_tables(payload)
    (out_dir / "all_models_clean_tables.md").write_text(md, encoding="utf-8")
    print(f"\n→ {out_dir / 'all_models_clean_metrics.json'}")
    print(f"→ {out_dir / 'all_models_clean_tables.md'}")
    print(f"→ {fig_dir / 'nsweep_irreducible.png'}")
    print(f"→ {fig_dir / 'capability_vs_scr.png'}")
    print(f"→ {fig_dir / 'confidence_collapse.png'}")


if __name__ == "__main__":
    main()
