#!/usr/bin/env python3
"""TW_ent-like token features with sharper narrative (post-hoc only)."""
from __future__ import annotations

import glob
import json
import math
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
CACHE = Path("/root/autodl-tmp/prs-outputs/.tw_ent_insight_cache.pkl")
CALC = frozenset({"numeric", "symbol", "variable"})
TAIL_L = 16
TOP_PCT = 0.10


def _pre_answer(run: dict) -> list[dict]:
    trace = run.get("token_trace") or []
    a = (run.get("answer_span") or {}).get("start_token")
    return trace[: int(a)] if a is not None else []


def _ents(tokens: list[dict], *, calc_only: bool) -> list[float]:
    out = []
    for t in tokens:
        if calc_only and _classify_token(t.get("token", "")) not in CALC:
            continue
        e = t.get("entropy")
        if e is not None:
            out.append(float(e))
    return out


def _ent_sum(tokens: list[dict], *, calc_only: bool, top_pct: float | None = None) -> float:
    ents = _ents(tokens, calc_only=calc_only)
    if not ents:
        return float("nan")
    if top_pct is not None:
        k = max(1, int(math.ceil(len(ents) * top_pct)))
        ents = list(np.partition(ents, -k)[-k:])
    return float(sum(ents))


def _tail_tokens(run: dict, tail_l: int) -> list[dict]:
    pre = _pre_answer(run)
    return pre[-tail_l:] if len(pre) > tail_l else pre


def extract_tw_features(base: dict, pert_runs: list[dict], a0: str) -> dict[str, float]:
    """TW_* aggregates: sum over 8 perturb runs (same budget as TW_ent_sum)."""
    runs = list(pert_runs)
    base_pre = _pre_answer(base)

    full_sums, calc_sums, tail_sums, calc_top_sums = [], [], [], []
    reject_full, reject_calc = [], []
    delta_full, delta_calc = [], []

    base_full = _ent_sum(base.get("token_trace") or [], calc_only=False)
    base_calc = _ent_sum(base_pre, calc_only=True)

    for run in runs:
        tt = run.get("token_trace") or []
        pre = _pre_answer(run)
        ans = str(run.get("answer_normalized", "")).strip()
        dis = not math_equal(a0, ans) if a0 and ans else False

        hf = _ent_sum(tt, calc_only=False)
        hc = _ent_sum(pre, calc_only=True)
        ht = _ent_sum(_tail_tokens(run, TAIL_L), calc_only=False)
        hct = _ent_sum(pre, calc_only=True, top_pct=TOP_PCT)

        if np.isfinite(hf):
            full_sums.append(hf)
            if dis:
                reject_full.append(hf)
            if np.isfinite(base_full):
                delta_full.append(abs(hf - base_full))
        if np.isfinite(hc):
            calc_sums.append(hc)
            if dis:
                reject_calc.append(hc)
            if np.isfinite(base_calc):
                delta_calc.append(abs(hc - base_calc))
        if np.isfinite(ht):
            tail_sums.append(ht)
        if np.isfinite(hct):
            calc_top_sums.append(hct)

    # cross-run calc entropy variance at aligned relative positions (base-indexed)
    nb = len(base_pre)
    calc_vars = []
    if nb > 0:
        for i, bt in enumerate(base_pre):
            if _classify_token(bt.get("token", "")) not in CALC:
                continue
            rp = i / max(nb - 1, 1)
            vals = [float(bt.get("entropy", 0.0))]
            for run in runs:
                pre = _pre_answer(run)
                np_ = len(pre)
                if np_ == 0:
                    continue
                j = min(int(round(rp * max(np_ - 1, 1))), np_ - 1)
                vals.append(float(pre[j].get("entropy", 0.0)))
            if len(vals) >= 2:
                calc_vars.append(float(np.var(vals)))
    tw_calc_xvar = float(sum(calc_vars)) if calc_vars else float("nan")

    def _sum(xs: list[float]) -> float:
        return float(sum(xs)) if xs else float("nan")

    return {
        "TW_ent_ref": _sum(full_sums),  # replicate TW_ent_sum on pert only (no base in sum)
        "TW_calc_ent": _sum(calc_sums),
        "TW_tail_ent": _sum(tail_sums),
        "TW_calc_top10": _sum(calc_top_sums),
        "TW_ent_delta": _sum(delta_full),
        "TW_calc_ent_delta": _sum(delta_calc),
        "TW_reject_ent": _sum(reject_full),
        "TW_reject_calc_ent": _sum(reject_calc),
        "TW_calc_xvar_sum": tw_calc_xvar,
    }


def _parse_one(args: tuple) -> dict | None:
    rp, label, seed, ds, model_key, tw_ref = args
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
    feats = extract_tw_features(base, pert, a0)
    if not any(np.isfinite(v) for v in feats.values()):
        return None
    row = {"model": model_key, "seed": seed, "ds": ds, "y": int(label), "bd": bd,
           "TW_ent_sum": float(tw_ref) if tw_ref is not None and tw_ref == tw_ref else feats["TW_ent_ref"]}
    row.update(feats)
    return row


def collect_jobs(out_root: Path) -> list[tuple]:
    jobs = []
    for mk, dname in MODELS.items():
        base = out_root / dname
        for seed in SEEDS:
            for ds in MATH_DS:
                sj = base / f"seed{seed}" / ds / "summary.jsonl"
                if not sj.exists():
                    continue
                tw_map = {}
                for ln in sj.read_text().splitlines():
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    if r.get("label_drop") or r.get("label_wrong_clean") is None:
                        continue
                    tw_map[r["id"]] = {
                        "tw": r.get("TW_ent_sum_total"),
                        "y": int(r["label_wrong_clean"]),
                    }
                for rp in glob.glob(str(base / f"seed{seed}" / ds / "raw_runs" / "*.json")):
                    if "partial" in rp or "error" in rp:
                        continue
                    rid = Path(rp).stem
                    if rid in tw_map:
                        info = tw_map[rid]
                        jobs.append((rp, info["y"], seed, ds, mk, info["tw"]))
    return jobs


def load_rows(out_root: Path, workers: int) -> list[dict]:
    jobs = collect_jobs(out_root)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_parse_one, jobs, chunksize=16):
            if r:
                rows.append(r)
    return rows


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
            med = np.nanmedian(Xtr, axis=0)
            for j in range(Xtr.shape[1]):
                Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
                Xte[~np.isfinite(Xte[:, j]), j] = med[j]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(max_iter=2000)
            clf.fit((Xtr - mu) / sd, ytr)
            scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
        if scs:
            vals.append(float(np.mean(scs)))
    return float(np.mean(vals)) if vals else float("nan")


def pooled_auroc(rows: list[dict], key: str) -> float:
    y = np.array([r["y"] for r in rows])
    s = np.array([r[key] for r in rows], float)
    m = np.isfinite(s)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return float("nan")
    a = roc_auc_score(y[m], s[m])
    return float(max(a, 1 - a))


NARRATIVES = {
    "TW_ent_sum": "基线：8 run 全 trace entropy 总和",
    "TW_calc_ent": "仅 reasoning 内计算 token（数/符/变量）entropy 和",
    "TW_tail_ent": f"答案前 {TAIL_L} 个 process token 的 entropy 和（commit 前犹豫）",
    "TW_calc_top10": "计算 token 中 top10% 高 entropy 之和（最犹豫的计算步）",
    "TW_ent_delta": "相对 base，8 run 全 trace |Δentropy| 累计（扰动致不确定漂移）",
    "TW_calc_ent_delta": "相对 base，计算区 |Δentropy| 累计",
    "TW_reject_ent": "仅「否决 base 答案」的 run 的 entropy 和（否决 run 有多飘）",
    "TW_reject_calc_ent": "否决 run 上计算 token entropy 和",
    "TW_calc_xvar_sum": "计算 token 对齐位置的跨 run entropy 方差之和（计算步 epistemic 分歧）",
}


def report(rows: list[dict], md_out: Path) -> None:
    bd = lodo_macro(rows, ["bd"])
    keys = [k for k in NARRATIVES if k in rows[0]]
    ranked = []
    for k in keys:
        fus = lodo_macro(rows, ["bd", k])
        ranked.append((k, fus, fus - bd, pooled_auroc(rows, k)))
    ranked.sort(key=lambda x: -x[1])

    lines = [
        "# TW_ent 叙事变体扫描",
        "",
        f"> N={len(rows)}，4 模型 LODO。对比 bd + 各 TW 变体 vs bd + TW_ent_sum。",
        "",
        f"**bd-only macro**: {bd:.3f}  |  **bd+TW_ent_sum**: {lodo_macro(rows, ['bd','TW_ent_sum']):.3f}",
        "",
        "## 排名（bd+T macro）",
        "",
        "| 特征 | bd+T | Δ vs bd | pooled AUROC | 叙事 |",
        "|------|------:|------:|------:|------|",
    ]
    ref = lodo_macro(rows, ["bd", "TW_ent_sum"])
    for k, fus, d, po in ranked:
        mark = " **←best**" if k == ranked[0][0] else (" ≈ref" if k == "TW_ent_sum" else "")
        lines.append(f"| `{k}` | {fus:.3f} | {d:+.3f} | {po:.3f} | {NARRATIVES[k]}{mark} |")

    lines += ["", "## Qwen2.5-3B", ""]
    sub = [r for r in rows if r["model"] == "qwen25_3b"]
    lines.append("| 特征 | bd+T | Δ vs bd+TW_ent |")
    lines.append("|------|------:|------:|")
    for k, fus, d, _ in ranked:
        fk = lodo_macro(sub, ["bd", k])
        lines.append(f"| `{k}` | {fk:.3f} | {fk - lodo_macro(sub, ['bd','TW_ent_sum']):+.3f} |")

    best = ranked[0][0]
    lines += [
        "",
        "## 建议",
        "",
        f"- **数字最好**: `{best}`",
        f"- **相对 TW_ent_sum**: macro {ranked[0][1]:.3f} vs ref {ref:.3f} ({ranked[0][1]-ref:+.3f})",
    ]
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/prs-outputs"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--md-out", type=Path, default=ROOT / "paper/tables/table_tw_ent_insight.md")
    args = ap.parse_args()

    if args.use_cache and CACHE.exists():
        rows = pickle.loads(CACHE.read_bytes())
    else:
        t0 = time.time()
        rows = load_rows(args.out_root, args.workers)
        CACHE.write_bytes(pickle.dumps(rows))
        print(f"built N={len(rows)} in {time.time()-t0:.1f}s")

    report(rows, args.md_out)
    bd = lodo_macro(rows, ["bd"])
    ref = lodo_macro(rows, ["bd", "TW_ent_sum"])
    print(f"bd={bd:.3f}  bd+TW_ent_sum={ref:.3f}\n")
    for k in NARRATIVES:
        if k not in rows[0]:
            continue
        fus = lodo_macro(rows, ["bd", k])
        q = lodo_macro([r for r in rows if r["model"] == "qwen25_3b"], ["bd", k])
        print(f"{k:<22} macro={fus:.3f} Δ={fus-bd:+.3f}  qwen25={q:.3f}  pooled={pooled_auroc(rows,k):.3f}")
    print(f"\nWrote {args.md_out}")


if __name__ == "__main__":
    main()
