#!/usr/bin/env python3
"""Offline ablation: compare S_ans / S_tr drift variants under different run sets.

For each completed dataset (reads raw_runs + features.jsonl, NO GPU, NO pipeline
recompute), report AUROC/AUPRC/ACC* for:

  S_tr (reasoning drift) definitions:
    B       = AltMass_local_spread_topk     (domain-agnostic top-k%, current default)
    A       = AltMass_local_spread_content  (per-domain content/math-key-step tokens)
    legacy  = AltMass_local_spread_reason   (math/reason-token mean)
  S_ans (answer drift):
    final   = AltMass_final

each computed over three run sets:
    W4   = 4 weight-perturb runs (current pipeline default)
    T4   = 4 text-rephrase runs
    TW8  = all 8 runs (text + weight)

Clean labels come from load_features_table (same grading as the paper main table).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from panda.core.altmass_decomposition import altmass_variants_weight_branch
from panda.metrics_tokur import compute_detection_metrics

VARIANTS = {
    "S_tr/B_topk": "AltMass_local_spread_topk",
    "S_tr/A_content": "AltMass_local_spread_content",
    "S_tr/legacy_reason": "AltMass_local_spread_reason",
    "S_ans/final": "AltMass_final",
}


def _load_features(out_dir: Path, ds: str) -> dict[str, dict]:
    """Read features.jsonl directly (fast, no re-grading)."""
    fp = out_dir / ds / "features.jsonl"
    by_id: dict[str, dict] = {}
    if not fp.exists():
        return by_id
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        by_id[r["id"]] = r
    return by_id


def _auroc(label_by_id: dict[str, int], scores_by_id: dict[str, float]) -> dict[str, float]:
    labels, scores = [], []
    for sid, lw in label_by_id.items():
        if sid not in scores_by_id:
            continue
        s = scores_by_id[sid]
        if s is None or s != s:
            continue
        labels.append(lw == 0)  # correct = positive (matches eval_clean)
        scores.append(float(s))
    if not scores or len(set(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "acc_star": float("nan"), "n": len(scores)}
    m = compute_detection_metrics(labels, scores)
    return {"auroc": m.auroc, "auprc": m.auprc, "acc_star": m.acc_star, "n": m.n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True, help="seed dir, e.g. .../maintable_qwen25_3b/seed41")
    ap.add_argument("--datasets", default="minerva")
    ap.add_argument("--top-pct", type=float, default=0.10)
    args = ap.parse_args()

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        raw_dir = args.out_dir / ds / "raw_runs"
        if not raw_dir.exists():
            print(f"skip {ds}: no raw_runs")
            continue
        feats = _load_features(args.out_dir, ds)  # W4 variants + label_wrong (no re-grade)
        label_by_id = {i: int(r.get("label_wrong", 1)) for i, r in feats.items()}
        # Only TW8 needs recompute from raw token traces (W4 already cached in features).
        tw8: dict[str, dict[str, float]] = {}
        for p in sorted(raw_dir.glob("*.json")):
            if p.name.endswith((".error.json", ".partial.json")):
                continue
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            wruns = rec.get("weight_perturb_runs") or []
            truns = rec.get("text_rephrase_runs") or []
            tw8[rec["id"]] = altmass_variants_weight_branch(truns + wruns, ds, top_pct=args.top_pct)

        print(f"\n=== {ds}  (n={len(label_by_id)}, top_pct={args.top_pct}, label=label_wrong) — AUROC|AUPRC|ACC* ===")
        print(f"{'variant':<24}{'W4 (default)':<24}{'TW8 (text+weight)':<24}")
        for label, key in VARIANTS.items():
            w4 = _auroc(label_by_id, {i: r.get(key) for i, r in feats.items()})
            t8 = _auroc(label_by_id, {i: v.get(key) for i, v in tw8.items()})
            def fmt(m):
                return f"{m['auroc']*100:.2f}|{m['auprc']*100:.2f}|{m['acc_star']*100:.2f}"
            print(f"{label:<24}{fmt(w4):<24}{fmt(t8):<24}")


if __name__ == "__main__":
    main()
