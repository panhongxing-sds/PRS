#!/usr/bin/env python3
"""Sweep token-level features: LODO macro AUROC (Qwen2.5-3B)."""
from __future__ import annotations

import glob
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from panda.grading.math_grader import math_equal  # noqa: E402

OUT = Path("/root/autodl-tmp/panda-outputs")
MODEL = "maintable_qwen25_3b"
MATH_DS = ("minerva", "math500", "gsm8k")
SEEDS = (41, 42, 43)
TOKEN_PAT = re.compile(
    r"(tok|ans_|baseline_|base_ans|ATU|margin|logprob|ent_|mar_|flip|js|cluster|"
    r"DC_|LL_|PE_|SC_|SAR|disagree|delta_ans|topk|rank|numeric_token|operator|"
    r"formula_skeleton|alternative_answer|confident_fragment)",
    re.I,
)


def _bd_from_path(rp: str) -> tuple | None:
    raw = json.loads(Path(rp).read_text())
    sm = raw.get("summary_metrics") or {}
    if sm.get("label_drop"):
        return None
    base_g = raw.get("base_generation") or {}
    text = list(raw.get("text_rephrase_runs") or [])
    weight = list(raw.get("weight_perturb_runs") or [])
    a0 = str(sm.get("a0") or base_g.get("answer_normalized", "")).strip()
    pert = [str(g.get("answer_normalized", "")).strip() for g in text + weight]
    if not a0 or not pert:
        return None
    rid = Path(rp).stem
    parts = Path(rp).parts
    ds = parts[-3]
    seed = int(parts[-4].replace("seed", ""))
    return (rid, seed, ds, sum(1 for x in pert if not math_equal(a0, x)) / len(pert))


def load_rows() -> tuple[list[dict], list[str]]:
    base = OUT / MODEL
    rows: list[dict] = []
    for seed in SEEDS:
        for ds in MATH_DS:
            sj = base / f"seed{seed}" / ds / "summary.jsonl"
            for ln in sj.read_text().splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if r.get("label_drop") or r.get("label_wrong_clean") is None:
                    continue
                rows.append({"id": r["id"], "seed": seed, "ds": ds, "y": int(r["label_wrong_clean"]), "feat": r})

    # bd from raw (parallel in main via pickle cache if exists)
    cache_p = OUT / ".qwen25_bd_cache.pkl"
    if cache_p.exists():
        bd_map = pickle.loads(cache_p.read_bytes())
    else:
        from concurrent.futures import ThreadPoolExecutor

        jobs = []
        for seed in SEEDS:
            for ds in MATH_DS:
                jobs.extend(
                    rp
                    for rp in glob.glob(str(base / f"seed{seed}" / ds / "raw_runs" / "*.json"))
                    if "partial" not in rp and "error" not in rp
                )
        bd_map = {}
        with ThreadPoolExecutor(16) as ex:
            for res in ex.map(_bd_from_path, jobs, chunksize=64):
                if res:
                    bd_map[(res[0], res[1], res[2])] = res[3]
        cache_p.write_bytes(pickle.dumps(bd_map))

    sample = rows[0]["feat"]
    cands = sorted(
        k
        for k, v in sample.items()
        if isinstance(v, (int, float)) and TOKEN_PAT.search(k)
    )
    flat = []
    for r in rows:
        d = {"y": r["y"], "ds": r["ds"], "seed": r["seed"], "bd": bd_map.get((r["id"], r["seed"], r["ds"]), np.nan)}
        for k in cands:
            v = r["feat"].get(k)
            d[k] = float(v) if isinstance(v, (int, float)) else np.nan
        flat.append(d)
    return flat, cands


def lodo_macro(rows: list[dict], cols: list[str]) -> float:
    vals = []
    for test_ds in MATH_DS:
        tr = [r for r in rows if r["ds"] != test_ds]
        scs = []
        for seed in SEEDS:
            te = [r for r in rows if r["ds"] == test_ds and r["seed"] == seed]
            if len(te) < 10:
                continue
            Xtr = np.array([[r[c] for c in cols] for r in tr], float)
            Xte = np.array([[r[c] for c in cols] for r in te], float)
            ytr = np.array([r["y"] for r in tr])
            yte = np.array([r["y"] for r in te])
            if len(np.unique(ytr)) < 2:
                continue
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(max_iter=2000)
            clf.fit((Xtr - mu) / sd, ytr)
            scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
        if scs:
            vals.append(float(np.mean(scs)))
    return float(np.mean(vals)) if vals else float("nan")


def lodo_per_ds(rows: list[dict], cols: list[str], test_ds: str) -> float:
    tr = [r for r in rows if r["ds"] != test_ds]
    scs = []
    for seed in SEEDS:
        te = [r for r in rows if r["ds"] == test_ds and r["seed"] == seed]
        if len(te) < 10:
            continue
        Xtr = np.array([[r[c] for c in cols] for r in tr], float)
        Xte = np.array([[r[c] for c in cols] for r in te], float)
        ytr = np.array([r["y"] for r in tr])
        yte = np.array([r["y"] for r in te])
        if len(np.unique(ytr)) < 2:
            continue
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=2000)
        clf.fit((Xtr - mu) / sd, ytr)
        scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
    return float(np.mean(scs)) if scs else float("nan")


def main() -> None:
    rows, cands = load_rows()
    print(f"N={len(rows)} candidates={len(cands)}", flush=True)

    bd_macro = lodo_macro(rows, ["bd"])
    print(f"bd only macro LODO: {bd_macro:.3f}", flush=True)

    best: dict[str, tuple[int, float, float]] = {}
    for k in cands:
        for sign in (1, -1):
            sk = f"__{k}"
            for r in rows:
                v = r.get(k, np.nan)
                r[sk] = sign * v if np.isfinite(v) else np.nan
            tok = lodo_macro(rows, [sk])
            fus = lodo_macro(rows, ["bd", sk])
            if k not in best or fus > best[k][2]:
                best[k] = (sign, tok, fus)

    rank_f = sorted(best.items(), key=lambda x: -x[1][2])
    rank_t = sorted(best.items(), key=lambda x: -x[1][1])

    print("\n=== TOP 12 bd + token (LODO macro) ===", flush=True)
    print(f"{'feature':<42} {'dir':>4} {'tok':>7} {'bd+tok':>7}", flush=True)
    for k, (sign, tok, fus) in rank_f[:12]:
        print(f"{k:<42} {sign:>4} {tok:>7.3f} {fus:>7.3f}", flush=True)

    print("\n=== TOP 8 token-only LODO ===", flush=True)
    for k, (sign, tok, fus) in rank_t[:8]:
        print(f"  {k:<42} dir={sign:>2} tok={tok:.3f} bd+tok={fus:.3f}", flush=True)

    top = rank_f[0]
    k, (sign, tok, fus) = top
    sk = f"__{k}"
    for r in rows:
        v = r.get(k, np.nan)
        r[sk] = sign * v if np.isfinite(v) else np.nan
    print(f"\n=== BEST: {k} (dir={sign}) macro={fus:.3f} ===", flush=True)
    for ds in MATH_DS:
        print(f"  {ds}: {lodo_per_ds(rows, ['bd', sk], ds)*100:.2f}%", flush=True)

    # combos of top-3 if different families
    tops = rank_f[:5]
    print("\n=== bd + top1 + top2 (if gains) ===", flush=True)
    k1, (s1, _, f1) = tops[0]
    k2, (s2, _, f2) = tops[1]
    for r in rows:
        for kk, ss in [(k1, s1), (k2, s2)]:
            v = r.get(kk, np.nan)
            r[f"__{kk}"] = ss * v if np.isfinite(v) else np.nan
    tri = lodo_macro(rows, ["bd", f"__{k1}", f"__{k2}"])
    print(f"  bd+{k1}+{k2}: {tri:.3f}  (vs bd+{k1}: {f1:.3f})", flush=True)


if __name__ == "__main__":
    main()
