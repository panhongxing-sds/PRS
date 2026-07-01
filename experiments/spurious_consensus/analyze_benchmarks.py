"""跨 benchmark 汇总：AUROC、错题分型、spurious consensus。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from glob import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from grading import is_correct

ROOT = Path(__file__).resolve().parent


def load_all(pattern: str) -> dict[str, list[dict]]:
    by_bench: dict[str, list[dict]] = {}
    for path in sorted(glob(str(ROOT / pattern))):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if int(r.get("label_drop", 0)) == 1:
                continue
            key = r.get("benchmark") or r.get("dataset")
            by_bench.setdefault(key, []).append(r)
    return by_bench


def stats_one(rows: list[dict]) -> dict:
    items = []
    for r in rows:
        pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a != ""]
        if len(pairs) < 2:
            continue
        a, _ = zip(*pairs)
        grading = r.get("grading", "math")
        gold = r["gold"]
        cnt = Counter(a)
        maj, top = cnt.most_common(1)[0]
        p_top = top / len(a)
        wrong = not is_correct(maj, gold, grading)
        items.append({"p_top": p_top, "wrong": wrong, "id": r["id"]})

    if not items:
        return {}
    labels = [int(x["wrong"]) for x in items]
    scores = [1 - x["p_top"] for x in items]
    auroc = float(roc_auc_score(labels, scores)) if 0 < sum(labels) < len(labels) else float("nan")
    wrongs = [x for x in items if x["wrong"]]
    blind = [x for x in wrongs if x["p_top"] >= 0.9]
    moderate = [x for x in wrongs if 0.3 <= x["p_top"] < 0.9]
    split = [x for x in wrongs if x["p_top"] < 0.3]
    sub90 = [x for x in items if x["p_top"] >= 0.9]
    sel_acc = 1 - np.mean([x["wrong"] for x in sub90]) if sub90 else float("nan")

    return {
        "n": len(items),
        "wrong_rate": float(np.mean(labels)),
        "auroc_uptop": auroc,
        "selective_acc_p09": float(sel_acc),
        "cov_p09": len(sub90) / len(items),
        "wrong_n": len(wrongs),
        "blind_p09": len(blind),
        "moderate_wrong": len(moderate),
        "split_wrong": len(split),
        "mean_p_top_on_wrong": float(np.mean([x["p_top"] for x in wrongs])) if wrongs else float("nan"),
        "blind_ids": [x["id"] for x in blind],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/samples/samples_*_seed41_*.jsonl")
    ap.add_argument("--out", default="results/benchmark_comparison.json")
    args = ap.parse_args()

    all_data = load_all(args.samples)
    results = {name: stats_one(rows) for name, rows in sorted(all_data.items())}

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)

    print(f"{'benchmark':<28} {'n':>5} {'wrong%':>7} {'AUROC':>6} {'blind':>5}")
    for name in sorted(results, key=lambda k: -results[k].get("wrong_rate", 0)):
        s = results[name]
        if not s:
            continue
        print(f"{name:<28} {s['n']:5d} {s['wrong_rate']*100:6.1f}% {s['auroc_uptop']:6.3f} {s['blind_p09']:5d}")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
