#!/usr/bin/env python3
"""Fast T_commit diagnosis: cliff vs split pre/post (multiprocess + cache).

  python3 scripts/analyze_t_commit_split.py              # build cache + report
  python3 scripts/analyze_t_commit_split.py --use-cache  # seconds
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prs.grading.math_grader import math_equal  # noqa: E402

MATH_DS = ("minerva", "math500", "gsm8k")
SEEDS = (41, 42, 43)
MODELS = {
    "qwen25_3b": "maintable_qwen25_3b",
    "llama32_1b": "maintable_llama32_1b",
    "llama31_8b": "maintable_llama31_8b",
    "qwen3_8b": "maintable_qwen3_8b",
}
CACHE = Path("/root/autodl-tmp/prs-outputs/.t_commit_feature_cache.pkl")
W = 2


def _eq_cache():
    return {}


def cliff_parts(run: dict) -> tuple[float, float, float]:
    tt = run.get("token_trace") or []
    a = (run.get("answer_span") or {}).get("start_token")
    if a is None or a < W or a >= len(tt):
        return float("nan"), float("nan"), float("nan")
    rl = [t.get("logprob") for t in tt[a - W : a] if t.get("logprob") is not None]
    al = [t.get("logprob") for t in tt[a : min(len(tt), a + W)] if t.get("logprob") is not None]
    if not rl or not al:
        return float("nan"), float("nan"), float("nan")
    rb, ra = float(np.mean(rl)), float(np.mean(al))
    return rb - ra, rb, ra


def _parse_one(args: tuple) -> dict | None:
    rp, label, model, seed, ds = args
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
    a0 = str(sm.get("a0") or base.get("answer_normalized", "")).strip()
    pert = [str(g.get("answer_normalized", "")).strip() for g in text + weight]
    if not a0 or not pert:
        return None
    n_dis = sum(1 for x in pert if not math_equal(a0, x))
    bd = n_dis / len(pert)
    cliffs, before, after = [], [], []
    for g in [base] + text + weight:
        c, rb, ra = cliff_parts(g)
        if np.isfinite(c):
            cliffs.append(c)
            before.append(rb)
            after.append(ra)
    if not cliffs:
        return None
    return {
        "model": model,
        "seed": seed,
        "ds": ds,
        "y": int(label),
        "bd": bd,
        "T_cliff_u": -float(np.mean(cliffs)),
        "T_pre_u": -float(np.mean(before)),
        "T_post_u": -float(np.mean(after)),
    }


def collect_jobs(out_root: Path) -> list[tuple]:
    jobs = []
    for mk, dname in MODELS.items():
        base = out_root / dname
        for seed in SEEDS:
            for ds in MATH_DS:
                sj = base / f"seed{seed}" / ds / "summary.jsonl"
                if not sj.exists():
                    continue
                labels = {}
                for ln in sj.read_text().splitlines():
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    if r.get("label_drop") or r.get("label_wrong_clean") is None:
                        continue
                    labels[r["id"]] = int(r["label_wrong_clean"])
                for rp in glob.glob(str(base / f"seed{seed}" / ds / "raw_runs" / "*.json")):
                    if "partial" in rp or "error" in rp:
                        continue
                    rid = Path(rp).stem
                    if rid in labels:
                        jobs.append((rp, labels[rid], mk, seed, ds))
    return jobs


def load_features(out_root: Path, workers: int) -> list[dict]:
    jobs = collect_jobs(out_root)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_parse_one, j) for j in jobs]
        for fu in as_completed(futs):
            r = fu.result()
            if r:
                rows.append(r)
    return rows


def auroc_auto(y: np.ndarray, s: np.ndarray) -> float:
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return float("nan")
    return float(max(roc_auc_score(y, s), roc_auc_score(y, -s)))


def lodo(rows: list[dict], cols: list[str]) -> float:
    scores = []
    for test_ds in MATH_DS:
        tr = [r for r in rows if r["ds"] != test_ds]
        for seed in SEEDS:
            te = [r for r in rows if r["ds"] == test_ds and r["seed"] == seed]
            if len(te) < 10:
                continue
            Xtr = np.array([[r[c] for c in cols] for r in tr])
            ytr = np.array([r["y"] for r in tr])
            Xte = np.array([[r[c] for c in cols] for r in te])
            yte = np.array([r["y"] for r in te])
            if len(np.unique(ytr)) < 2:
                continue
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=3000)
            clf.fit(sc.fit_transform(Xtr), ytr)
            p = clf.predict_proba(sc.transform(Xte))[:, 1]
            scores.append(roc_auc_score(yte, p))
    return float(np.mean(scores)) if scores else float("nan")


def report(rows: list[dict], md_out: Path) -> None:
    y = np.array([r["y"] for r in rows])
    singles = ["bd", "T_cliff_u", "T_pre_u", "T_post_u"]
    lines = [
        "# T_commit 诊断（cliff vs 边界前/后拆分）",
        "",
        f"> N={len(rows)}，multiprocess CPU（JSON 解析无 GPU 加速意义）。",
        "",
        "## 1. 四模型 pooled 单特征 AUROC（auto-invert）",
        "",
        "| 特征 | AUROC | 说明 |",
        "|------|------:|------|",
    ]
    desc = {
        "bd": "8 票否决",
        "T_cliff_u": "−(lp_before−lp_after) 当前 T_commit",
        "T_pre_u": "−lp_before 边界前",
        "T_post_u": "−lp_after 答案 token",
    }
    for k in singles:
        a = auroc_auto(y, np.array([r[k] for r in rows]))
        lines.append(f"| {k} | {a:.3f} | {desc[k]} |")

    lines += ["", "## 2. LODO 融合", "", "| 配置 | AUROC |", "|------|------:|"]
    for name, cols in [
        ("bd + T_cliff **(当前)**", ["bd", "T_cliff_u"]),
        ("bd + T_pre", ["bd", "T_pre_u"]),
        ("bd + T_post", ["bd", "T_post_u"]),
        ("bd + T_pre + T_post **(拆开)**", ["bd", "T_pre_u", "T_post_u"]),
    ]:
        lines.append(f"| {name} | {lodo(rows, cols):.3f} |")

    lines += ["", "## 3. Qwen2.5-3B 分数据集", "", "| ds | bd | T_cliff | T_pre | T_post |", "|----|---:|--------:|------:|-------:|"]
    q = [r for r in rows if r["model"] == "qwen25_3b"]
    for ds in MATH_DS:
        sub = [r for r in q if r["ds"] == ds]
        yy = np.array([r["y"] for r in sub])
        lines.append(
            "| "
            + " | ".join(
                [ds]
                + [f"{auroc_auto(yy, np.array([r[k] for r in sub])):.3f}" for k in singles]
            )
            + " |"
        )

    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/prs-outputs"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--md-out", type=Path, default=ROOT / "paper/tables/table_t_commit_split.md")
    args = ap.parse_args()

    if args.use_cache and CACHE.exists():
        rows = pickle.loads(CACHE.read_bytes())
        print(f"cache hit N={len(rows)}")
    else:
        import time

        t0 = time.time()
        rows = load_features(args.out_root, args.workers)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps(rows))
        print(f"built cache N={len(rows)} in {time.time()-t0:.1f}s → {CACHE}")

    report(rows, args.md_out)

    y = np.array([r["y"] for r in rows])
    print("\n--- pooled single AUROC ---")
    for k in ["bd", "T_cliff_u", "T_pre_u", "T_post_u"]:
        print(f"  {k}: {auroc_auto(y, np.array([r[k] for r in rows])):.3f}")
    print("\n--- LODO ---")
    for name, cols in [
        ("bd+T_cliff", ["bd", "T_cliff_u"]),
        ("bd+T_pre", ["bd", "T_pre_u"]),
        ("bd+pre+post", ["bd", "T_pre_u", "T_post_u"]),
    ]:
        print(f"  {name}: {lodo(rows, cols):.3f}")
    print(f"\nWrote {args.md_out}")


if __name__ == "__main__":
    main()
