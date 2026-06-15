#!/usr/bin/env python3
"""Near-answer weighted process token features (later reasoning tokens matter more)."""
from __future__ import annotations

import glob
import json
import math
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prs.ase.numeric_trajectory import extract_numbers  # noqa: E402
from prs.ase.reasoning_token_features import _classify_token, _reasoning_text  # noqa: E402
from prs.grading.math_grader import math_equal  # noqa: E402

MATH_DS = ("minerva", "math500", "gsm8k")
SEEDS = (41, 42, 43)
MODELS = {
    "qwen25_3b": "maintable_qwen25_3b",
    "llama32_1b": "maintable_llama32_1b",
    "llama31_8b": "maintable_llama31_8b",
    "qwen3_8b": "maintable_qwen3_8b",
}
CACHE = Path("/root/autodl-tmp/prs-outputs/.proc_near_answer_cache.pkl")
CALC = frozenset({"numeric", "symbol", "variable"})


def _pre_answer(run: dict) -> list[dict]:
    trace = run.get("token_trace") or []
    a = (run.get("answer_span") or {}).get("start_token")
    return trace[: int(a)] if a is not None and int(a) > 0 else []


def _prox_weight(i: int, n: int, mode: str = "linear") -> float:
    """Weight increases toward answer: i=0 far, i=n-1 closest."""
    if n <= 1:
        return 1.0
    u = (i + 1) / n  # (0,1], later tokens larger
    if mode == "linear":
        return u
    if mode == "quad":
        return u * u
    if mode == "exp":
        return float(math.expm1(2.0 * u) / math.expm1(2.0))
    return u


def _lev(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if not n:
        return m
    if not m:
        return n
    prev = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[m]


def _norm_edit(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 0.0
    return _lev(a, b) / max(len(a), len(b), 1)


def _calc_tok_text(t: dict) -> str | None:
    raw = (t.get("token") or "").strip()
    if not raw:
        return None
    if _classify_token(raw) in CALC:
        return re.sub(r"\s+", "", raw)
    return None


def calc_seq_tail(pre: list[dict], frac: float = 0.25) -> list[str]:
    """Calc tokens in last `frac` of reasoning (near answer)."""
    if not pre:
        return []
    start = max(0, int(math.floor(len(pre) * (1.0 - frac))))
    seq = []
    for t in pre[start:]:
        x = _calc_tok_text(t)
        if x:
            seq.append(x)
    return seq


def prox_weighted_entropy(pre: list[dict], mode: str = "linear") -> float:
    n = len(pre)
    if n == 0:
        return float("nan")
    s = 0.0
    wsum = 0.0
    for i, t in enumerate(pre):
        w = _prox_weight(i, n, mode)
        h = t.get("entropy")
        if h is None:
            continue
        s += w * float(h)
        wsum += w
    return s / wsum if wsum > 0 else float("nan")


def prox_weighted_margin_unc(pre: list[dict], mode: str = "linear") -> float:
    """Higher = more uncertain (low margin near answer)."""
    n = len(pre)
    if n == 0:
        return float("nan")
    s, wsum = 0.0, 0.0
    for i, t in enumerate(pre):
        w = _prox_weight(i, n, mode)
        m = t.get("margin_top2")
        if m is None:
            continue
        s += w * (-float(m))
        wsum += w
    return s / wsum if wsum > 0 else float("nan")


def answer_missing_in_tail(run: dict, tail_frac: float = 0.25) -> float:
    """1 if answer not supported in last tail_frac of reasoning text."""
    ans = str(run.get("answer_normalized") or "").strip()
    if not ans:
        return float("nan")
    pre = _pre_answer(run)
    if not pre:
        return float("nan")
    start = max(0, int(math.floor(len(pre) * (1.0 - tail_frac))))
    tail_text = "".join(t.get("token", "") for t in pre[start:])
    ans_nums = extract_numbers(ans)
    if ans_nums:
        tail_nums = extract_numbers(tail_text)
        if tail_nums:
            hit = any(any(math_equal(str(a), str(b)) for b in tail_nums) for a in ans_nums)
            return 0.0 if hit else 1.0
    if ans.lower() in tail_text.lower():
        return 0.0
    return 1.0


def extract_features(base: dict, pert_runs: list[dict]) -> dict[str, float]:
    base_pre = _pre_answer(base)
    base_tail = calc_seq_tail(base_pre, 0.25)

    ent_lin, ent_quad, mar_lin = [], [], []
    edit_tail, edit_tail_pair = [], []
    support_tail_miss = []

    for run in pert_runs:
        pre = _pre_answer(run)
        for arr, val, mode in [
            (ent_lin, prox_weighted_entropy(pre, "linear"), None),
            (ent_quad, prox_weighted_entropy(pre, "quad"), None),
            (mar_lin, prox_weighted_margin_unc(pre, "linear"), None),
        ]:
            if np.isfinite(val):
                arr.append(val)
        et = calc_seq_tail(pre, 0.25)
        edit_tail.append(_norm_edit(et, base_tail))
        sm = answer_missing_in_tail(run, 0.25)
        if np.isfinite(sm):
            support_tail_miss.append(sm)

    # pairwise tail calc edit among all runs
    all_tail = [base_tail] + [calc_seq_tail(_pre_answer(r), 0.25) for r in pert_runs]
    pairs = []
    for i in range(len(all_tail)):
        for j in range(i + 1, len(all_tail)):
            pairs.append(_norm_edit(all_tail[i], all_tail[j]))

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "T_ent_prox_lin": mean(ent_lin),
        "T_ent_prox_quad": mean(ent_quad),
        "T_margin_prox_lin": mean(mar_lin),
        "T_calc_edit_tail": mean(edit_tail),
        "T_calc_edit_tail_pair": mean(pairs),
        "T_support_tail_miss": mean(support_tail_miss),
    }


def _parse_one(args: tuple) -> dict | None:
    rp, label, seed, ds, model_key = args
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
    feats = extract_features(base, pert)
    tw = float(sm.get("TW_ent_sum_total", float("nan")))
    row = {"model": model_key, "seed": seed, "ds": ds, "y": int(label), "bd": bd, "TW_ent_sum": tw, **feats}
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
                    if Path(rp).stem in labels:
                        jobs.append((rp, labels[Path(rp).stem], seed, ds, mk))
    return jobs


def load_rows(out_root: Path, workers: int) -> list[dict]:
    jobs = collect_jobs(out_root)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_parse_one, jobs, chunksize=32):
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


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/prs-outputs"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--md-out", type=Path, default=ROOT / "paper/tables/table_near_answer_token.md")
    args = ap.parse_args()

    if args.use_cache and CACHE.exists():
        rows = pickle.loads(CACHE.read_bytes())
    else:
        t0 = time.time()
        rows = load_rows(args.out_root, args.workers)
        CACHE.write_bytes(pickle.dumps(rows))
        print(f"built N={len(rows)} in {time.time()-t0:.1f}s")

    bd = lodo_macro(rows, ["bd"])
    ref = lodo_macro(rows, ["bd", "TW_ent_sum"])
    keys = [k for k in rows[0] if k.startswith("T_")]

    print(f"bd-only={bd:.3f}  bd+TW_ent_sum={ref:.3f}\n")
    print(f"{'feature':<28} {'pooled':>7} {'bd+T':>7} {'Δ':>7} {'Q25':>7}")
    ranked = []
    for k in keys:
        fus = lodo_macro(rows, ["bd", k])
        q25 = lodo_macro([r for r in rows if r["model"] == "qwen25_3b"], ["bd", k])
        ranked.append((k, fus, fus - bd, pooled_auroc(rows, k), q25))
    ranked.sort(key=lambda x: -x[1])
    for k, fus, d, po, q25 in ranked:
        print(f"{k:<28} {po:>7.3f} {fus:>7.3f} {d:>+7.3f} {q25:>7.3f}")

    lines = [
        "# 近答案加权 process-token 指标",
        "",
        f"> N={len(rows)}。权重 $w_t\\propto$ 距 answer 越近越大（linear / quad）。",
        "",
        f"bd-only={bd:.3f}, bd+TW_ent_sum={ref:.3f}",
        "",
        "| 特征 | bd+T | Δ | Qwen2.5 | pooled | 说明 |",
        "|------|------:|---:|---:|---:|------|",
    ]
    desc = {
        "T_ent_prox_lin": "近答案加权 mean entropy（linear）",
        "T_ent_prox_quad": "近答案加权 mean entropy（quad，更强调尾部）",
        "T_margin_prox_lin": "近答案加权 −margin（尾部犹豫）",
        "T_calc_edit_tail": "最后25% reasoning 计算序列 vs base edit",
        "T_calc_edit_tail_pair": "尾部计算序列 pairwise edit",
        "T_support_tail_miss": "答案是否缺失于最后25% reasoning",
    }
    for k, fus, d, po, q25 in ranked:
        lines.append(f"| `{k}` | {fus:.3f} | {d:+.3f} | {q25:.3f} | {po:.3f} | {desc.get(k,k)} |")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.md_out}")


if __name__ == "__main__":
    main()
