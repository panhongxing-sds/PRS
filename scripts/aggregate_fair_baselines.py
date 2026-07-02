#!/usr/bin/env python3
"""Fair-budget baselines: aggregate token scores over all 9 decodes (1 base + 8 perturb).

Main table PE/LL/SC/DeepConf/SAR use the base trace only (n=1).
This script recomputes the same metrics with max/mean over 9 runs so comparison
with DH-Score uses matched generation budget.

Also reports F_resp (8 perturb answers) and SE (8 high-temp samples when present).

Usage:
  python scripts/aggregate_fair_baselines.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panda.baselines.from_record import official_nli_sample_uq_from_record  # noqa: E402
from panda.baselines.token_scores import token_baselines_from_generation  # noqa: E402

# Reuse loaders from aggregate_panda_v2
from aggregate_panda_v2 import (  # noqa: E402
    MODELS,
    MATH_DS,
    SEEDS,
    bd_vote,
    load_summary,
)

TOKEN_KEYS = [
    ("PE@9-max", "baseline_PE_mean", True),
    ("LL@9-max", "baseline_LL_nll", True),
    ("SC@9-max", "baseline_SC_mean", True),
    ("DeepConf@9-max", "baseline_DC_min", True),
    ("SAR@9-max", "baseline_SAR", True),
    ("PE@9-mean", "baseline_PE_mean", False),
]


def _runs_from_raw(raw: dict) -> list[dict]:
    base = raw.get("base_generation") or {}
    text = list(raw.get("text_rephrase_runs") or [])
    weight = list(raw.get("weight_perturb_runs") or [])
    runs = []
    if base.get("token_trace"):
        runs.append(base)
    runs.extend(text)
    runs.extend(weight)
    return runs


def _aggregate_token_scores(raw: dict) -> dict[str, float]:
    runs = _runs_from_raw(raw)
    if not runs:
        return {}
    per_run = [token_baselines_from_generation(r) for r in runs]
    out: dict[str, float] = {}
    for label, key, use_max in TOKEN_KEYS:
        vals = [float(d.get(key, float("nan"))) for d in per_run]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            out[label] = float("nan")
        elif use_max:
            out[label] = float(np.max(vals))
        else:
            out[label] = float(np.mean(vals))
    se = official_nli_sample_uq_from_record(raw)
    out["SE (8-sample)"] = float(se.get("baseline_SE_H", float("nan")))
    feat = extract_features_from_raw(raw)
    if feat:
        out["Dissent"] = feat["bd"]
        out["F_resp (8-pert)"] = feat["F"]
    return out


def extract_features_from_raw(r: dict) -> dict | None:
    sm = r.get("summary_metrics") or {}
    if sm.get("label_drop"):
        return None
    base = r.get("base_generation") or {}
    text = list(r.get("text_rephrase_runs") or [])
    weight = list(r.get("weight_perturb_runs") or [])
    a0 = str(sm.get("a0") or base.get("answer_normalized", "")).strip()
    pert = [str(g.get("answer_normalized", "")).strip() for g in text + weight]
    return {
        "bd": bd_vote(a0, pert),
        "F": float(sm.get("F_resp") or sm.get("TW_ASE") or 0),
    }


def _job(args: tuple) -> dict | None:
    rp, rid, seed, ds, model_key, summ_row = args
    try:
        raw = json.loads(Path(rp).read_text())
    except Exception:
        return None
    if raw.get("id") != rid:
        return None
    sm = raw.get("summary_metrics") or {}
    if sm.get("label_drop"):
        return None
    scores = _aggregate_token_scores(raw)
    if not scores:
        return None
    return {
        "id": rid,
        "seed": seed,
        "ds": ds,
        "model": model_key,
        "y": int(summ_row["label_wrong_clean"]),
        **scores,
    }


def load_fair_rows(out_root: Path, model_key: str, workers: int) -> list[dict]:
    base = out_root / MODELS[model_key]
    jobs = []
    for seed in SEEDS:
        for ds in MATH_DS:
            summ = load_summary(base, seed, ds)
            raw_glob = base / f"seed{seed}" / ds / "raw_runs" / "*.json"
            for rp in raw_glob.parent.glob("*.json"):
                if "partial" in rp.name or "error" in rp.name:
                    continue
                rid = rp.stem
                if rid in summ:
                    jobs.append((str(rp), rid, seed, ds, model_key, summ[rid]))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_job, jobs, chunksize=32):
            if r:
                rows.append(r)
    return rows


def pooled_auroc(rows: list[dict], key: str, invert: bool = False) -> float:
    y = np.array([r["y"] for r in rows])
    s = np.array([r[key] for r in rows], dtype=float)
    if invert:
        s = -s
    m = np.isfinite(s)
    if m.sum() < 10 or len(set(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "paper/maintable/fair_baseline_results.json")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for mk in MODELS:
        rows = load_fair_rows(args.outputs_root, mk, args.workers)
        print(f"{mk}: n={len(rows)}")
        all_rows.extend(rows)

    keys = [k for k, _, _ in TOKEN_KEYS] + ["SE (8-sample)", "Dissent", "F_resp (8-pert)"]
    summary = {"N": len(all_rows), "methods": {}}
    for k in keys:
        inv = k.startswith(("SC", "DeepConf", "SAR")) or "DeepConf" in k
        summary["methods"][k] = pooled_auroc(all_rows, k, invert=inv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
