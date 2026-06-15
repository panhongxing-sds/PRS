#!/usr/bin/env python3
"""Broad validation: proc_calc_flip vs bd / tok_disagree / TW_ent (4 models)."""
from __future__ import annotations

import glob
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prs.ase.reasoning_token_features import _classify_token  # noqa: E402
from prs.grading.math_grader import math_equal  # noqa: E402

MATH_DS = ("minerva", "math500", "gsm8k")
SEEDS = (41, 42, 43)
MODELS = {
    "qwen25_3b": "maintable_qwen25_3b",
    "llama32_1b": "maintable_llama32_1b",
    "llama31_8b": "maintable_llama31_8b",
    "qwen3_8b": "maintable_qwen3_8b",
}
CACHE = Path("/root/autodl-tmp/prs-outputs/.proc_calc_flip_all_models.pkl")
CALC_KINDS = frozenset({"numeric", "symbol", "variable"})

SUMMARY_REFS = {
    "T_tok_disagree": "T_tok_disagree_mean",
    "TW_ent_sum": "TW_ent_sum_total",
}


def reasoning_trace(run: dict) -> list[dict]:
    trace = run.get("token_trace") or []
    a = (run.get("answer_span") or {}).get("start_token")
    return trace[: int(a)] if a is not None else []


def proc_calc_flip(base: dict, pert_runs: list[dict]) -> float:
    base_proc = reasoning_trace(base)
    nb = len(base_proc)
    if nb == 0:
        return float("nan")
    rates = []
    for pr in pert_runs:
        pert_proc = reasoning_trace(pr)
        np_ = len(pert_proc)
        if np_ == 0:
            continue
        flips, counted = 0, 0
        for i, bt in enumerate(base_proc):
            if _classify_token(bt.get("token", "")) not in CALC_KINDS:
                continue
            rp = i / max(nb - 1, 1)
            j = min(int(round(rp * max(np_ - 1, 1))), np_ - 1)
            pt = pert_proc[j]
            if _classify_token(pt.get("token", "")) not in CALC_KINDS:
                continue
            counted += 1
            if int(bt.get("token_id", -1)) != int(pt.get("token_id", -2)):
                flips += 1
        if counted:
            rates.append(flips / counted)
    return float(np.mean(rates)) if rates else float("nan")


def _parse_one(args: tuple) -> dict | None:
    rp, label, seed, ds, model_key, summary_feats = args
    try:
        raw = json.loads(Path(rp).read_text())
    except Exception:
        return None
    sm = raw.get("summary_metrics") or {}
    if sm.get("label_drop"):
        return None
    base = raw.get("base_generation") or {}
    text = list(raw.get("text_rephrase_runs") or [])
    weight = list(raw.get("weight_perturb_runs") or [])
    pert = text + weight
    a0 = str(sm.get("a0") or base.get("answer_normalized", "")).strip()
    pert_ans = [str(g.get("answer_normalized", "")).strip() for g in pert]
    if not a0 or not pert_ans:
        return None
    bd = sum(1 for x in pert_ans if not math_equal(a0, x)) / len(pert_ans)
    pcf = proc_calc_flip(base, pert)
    if not np.isfinite(pcf):
        return None
    row = {
        "model": model_key,
        "seed": seed,
        "ds": ds,
        "y": int(label),
        "bd": bd,
        "proc_calc_flip": pcf,
    }
    for alias, key in SUMMARY_REFS.items():
        v = summary_feats.get(key)
        row[alias] = float(v) if v is not None and v == v else float("nan")
    return row


def collect_all(out_root: Path) -> list[tuple]:
    jobs = []
    for mk, dname in MODELS.items():
        base = out_root / dname
        for seed in SEEDS:
            for ds in MATH_DS:
                sj = base / f"seed{seed}" / ds / "summary.jsonl"
                if not sj.exists():
                    continue
                summaries = {}
                for ln in sj.read_text().splitlines():
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    if r.get("label_drop") or r.get("label_wrong_clean") is None:
                        continue
                    summaries[r["id"]] = {
                        "y": int(r["label_wrong_clean"]),
                        "feat": {k: r.get(k) for k in SUMMARY_REFS.values()},
                    }
                for rp in glob.glob(str(base / f"seed{seed}" / ds / "raw_runs" / "*.json")):
                    if "partial" in rp or "error" in rp:
                        continue
                    rid = Path(rp).stem
                    if rid in summaries:
                        s = summaries[rid]
                        jobs.append((rp, s["y"], seed, ds, mk, s["feat"]))
    return jobs


def load_rows(out_root: Path, workers: int) -> list[dict]:
    jobs = collect_all(out_root)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_parse_one, jobs, chunksize=16):
            if r:
                rows.append(r)
    return rows


def best_col(rows: list[dict], key: str) -> str:
    y = np.array([r["y"] for r in rows])
    s = np.array([r[key] for r in rows])
    m = np.isfinite(s)
    if m.sum() < 20:
        return key
    yy, ss = y[m], s[m]
    a = roc_auc_score(yy, ss)
    if max(a, 1 - a) == (1 - a):
        ck = f"__neg_{key}"
        for r in rows:
            v = r.get(key, np.nan)
            r[ck] = -v if np.isfinite(v) else np.nan
        return ck
    return key


def _fit_impute(rows: list[dict], cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array([[r[c] for c in cols] for r in rows], float)
    med = np.array([np.nanmedian(X[:, j]) if np.any(np.isfinite(X[:, j])) else 0.0 for j in range(X.shape[1])])
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        X[bad, j] = med[j]
    return X, med, X.std(0) + 1e-9


def lodo_per_ds(rows: list[dict], cols: list[str], test_ds: str) -> float:
    tr = [r for r in rows if r["ds"] != test_ds]
    scs = []
    for seed in SEEDS:
        te = [r for r in rows if r["ds"] == test_ds and r["seed"] == seed]
        if len(te) < 10:
            continue
        Xtr = np.array([[r[c] for c in cols] for r in tr], float)
        med = np.array([np.nanmedian(Xtr[:, j]) if np.any(np.isfinite(Xtr[:, j])) else 0.0 for j in range(Xtr.shape[1])])
        for j in range(Xtr.shape[1]):
            for M in (Xtr,):
                bad = ~np.isfinite(M[:, j])
                M[bad, j] = med[j]
        Xte = np.array([[r[c] for c in cols] for r in te], float)
        for j in range(Xte.shape[1]):
            bad = ~np.isfinite(Xte[:, j])
            Xte[bad, j] = med[j]
        ytr = np.array([r["y"] for r in tr])
        yte = np.array([r["y"] for r in te])
        if len(np.unique(ytr)) < 2:
            continue
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=2000)
        clf.fit((Xtr - mu) / sd, ytr)
        scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
    return float(np.mean(scs)) if scs else float("nan")


def lodo_macro(rows: list[dict], cols: list[str]) -> float:
    vals = [lodo_per_ds(rows, cols, ds) for ds in MATH_DS]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def pooled_auroc(rows: list[dict], key: str) -> float:
    y = np.array([r["y"] for r in rows])
    s = np.array([r[key] for r in rows])
    m = np.isfinite(s)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return float("nan")
    yy, ss = y[m], s[m]
    a = roc_auc_score(yy, ss)
    return float(max(a, 1 - a))


def eval_variant(rows: list[dict], tok_key: str) -> dict:
    if tok_key not in rows[0]:
        return {}
    col = best_col(rows, tok_key)
    bd = lodo_macro(rows, ["bd"])
    fus = lodo_macro(rows, ["bd", col])
    tok = lodo_macro(rows, [col])
    per_ds = {}
    for ds in MATH_DS:
        f = lodo_per_ds(rows, ["bd", col], ds)
        b = lodo_per_ds(rows, ["bd"], ds)
        per_ds[ds] = {"fus": f, "bd": b, "drop": b - f}
    return {
        "pooled": pooled_auroc(rows, tok_key),
        "tok_lodo": tok,
        "bd_lodo": bd,
        "fus_lodo": fus,
        "delta": fus - bd,
        "per_ds": per_ds,
    }


def report(rows: list[dict], md_out: Path) -> None:
    lines = [
        "# proc_calc_flip 广泛验证（4 模型 × 3 数据集）",
        "",
        f"> N={len(rows)}，LODO OOF LR。对比 bd-only / bd+proc_calc_flip / bd+T_tok / bd+TW_ent。",
        "",
        "## 1. 各模型 LODO macro",
        "",
        "| Model | bd | +proc_calc_flip | Δ | +T_tok | Δ | +TW_ent | Δ |",
        "|-------|---:|---:|---:|---:|---:|---:|---:|",
    ]
    variants = ["proc_calc_flip", "T_tok_disagree", "TW_ent_sum"]
    model_results = {}
    for mk in MODELS:
        sub = [r for r in rows if r["model"] == mk]
        if not sub:
            continue
        ev = {v: eval_variant(sub, v) for v in variants}
        model_results[mk] = ev
        bd = ev["proc_calc_flip"]["bd_lodo"]
        line = f"| {mk} | {bd:.3f}"
        for v in variants:
            line += f" | {ev[v]['fus_lodo']:.3f} | {ev[v]['delta']:+.3f}"
        lines.append(line)

    # pooled all models
    lines += ["", "## 2. 四模型 pooled", ""]
    ev_all = {v: eval_variant(rows, v) for v in variants}
    lines.append("| Variant | pooled AUROC | tok LODO | bd+tok | Δ vs bd |")
    lines.append("|---------|------:|------:|------:|------:|")
    for v in variants:
        e = ev_all[v]
        lines.append(
            f"| {v} | {e['pooled']:.3f} | {e['tok_lodo']:.3f} | {e['fus_lodo']:.3f} | {e['delta']:+.3f} |"
        )
    lines.append(f"| bd-only | — | — | {ev_all['proc_calc_flip']['bd_lodo']:.3f} | 0 |")

    lines += ["", "## 3. 分数据集 drop-one（bd+T → bd，负=去掉 T 后下降）", ""]
    for mk, ev in model_results.items():
        lines.append(f"### {mk}")
        lines.append("| ds | bd | +proc_calc | drop | +T_tok drop | +TW_ent drop |")
        lines.append("|----|---:|---:|---:|---:|---:|")
        for ds in MATH_DS:
            p = ev["proc_calc_flip"]["per_ds"][ds]
            t = ev["T_tok_disagree"]["per_ds"][ds]
            w = ev["TW_ent_sum"]["per_ds"][ds]
            lines.append(
                f"| {ds} | {p['bd']*100:.2f} | {p['fus']*100:.2f} | {p['drop']*100:+.2f}pp | "
                f"{t['drop']*100:+.2f}pp | {w['drop']*100:+.2f}pp |"
            )
        lines.append("")

    # pass criteria
    pcf = ev_all["proc_calc_flip"]
    n_pos = sum(1 for mk, ev in model_results.items() if ev["proc_calc_flip"]["delta"] > 0)
    n_drop = sum(
        1 for mk, ev in model_results.items()
        for ds in MATH_DS
        if ev["proc_calc_flip"]["per_ds"][ds]["drop"] < 0
    )
    lines += [
        "## 4. 判定",
        "",
        f"- proc_calc_flip pooled AUROC: **{pcf['pooled']:.3f}** ({'>' if pcf['pooled']>=0.6 else '<'}0.6)",
        f"- 4模型中 bd+proc_calc > bd-only: **{n_pos}/4**",
        f"- 12格（4模型×3ds）drop-one 为负（去掉 T 伤害）: **{n_drop}/12**",
        f"- macro Δ vs bd: **{pcf['delta']:+.3f}**（TW_ent: {ev_all['TW_ent_sum']['delta']:+.3f}，T_tok: {ev_all['T_tok_disagree']['delta']:+.3f}）",
    ]
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/prs-outputs"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--md-out", type=Path, default=ROOT / "paper/tables/table_proc_calc_flip_validation.md")
    args = ap.parse_args()

    if args.use_cache and CACHE.exists():
        rows = pickle.loads(CACHE.read_bytes())
        print(f"cache N={len(rows)}")
    else:
        t0 = time.time()
        rows = load_rows(args.out_root, args.workers)
        CACHE.write_bytes(pickle.dumps(rows))
        print(f"built N={len(rows)} in {time.time()-t0:.1f}s")

    report(rows, args.md_out)

    print("\n=== per-model macro ===")
    for mk in MODELS:
        sub = [r for r in rows if r["model"] == mk]
        if not sub:
            print(f"{mk}: no data")
            continue
        for v in ["proc_calc_flip", "T_tok_disagree", "TW_ent_sum"]:
            e = eval_variant(sub, v)
            print(f"  {mk} {v:20s} bd={e['bd_lodo']:.3f} fus={e['fus_lodo']:.3f} Δ={e['delta']:+.3f}")
    print(f"\nWrote {args.md_out}")


if __name__ == "__main__":
    main()
