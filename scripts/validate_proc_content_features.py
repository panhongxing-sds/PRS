#!/usr/bin/env python3
"""Process-token CONTENT features (not probability): edit/support/last/repair."""
from __future__ import annotations

import glob
import json
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

from panda.core.numeric_trajectory import extract_numbers  # noqa: E402
from panda.core.reasoning_token_features import (  # noqa: E402
    _classify_token,
    _reasoning_text,
)
from panda.grading.answer_canonicalizer import math_equal_clean  # noqa: E402
from panda.grading.math_grader import math_equal  # noqa: E402

MATH_DS = ("minerva", "math500", "gsm8k")
SEEDS = (41, 42, 43)
MODELS = {
    "qwen25_3b": "maintable_qwen25_3b",
    "llama32_1b": "maintable_llama32_1b",
    "llama31_8b": "maintable_llama31_8b",
    "qwen3_8b": "maintable_qwen3_8b",
}
CACHE = Path("/root/autodl-tmp/panda-outputs/.proc_content_feature_cache.pkl")

CALC_KINDS = frozenset({"numeric", "symbol", "variable"})
BRACKET_CHARS = frozenset("()[]{}")
REPAIR_PAT = re.compile(
    r"\b(wait|actually|however|mistake|instead|recalculate|re-check|recheck|"
    r"let me check|let's check|i made an error|my mistake|correction|retry)\b",
    re.I,
)

FEAT_KEYS = (
    "proc_calc_edit_base",
    "proc_calc_edit_pair",
    "proc_answer_support_missing",
    "proc_last_value_mismatch",
    "proc_repair_marker_rate",
)

NARR = {
    "proc_calc_edit_base": "base vs perturb 计算 token 序列归一化 edit distance",
    "proc_calc_edit_pair": "perturb 间计算序列 pairwise edit distance（process self-consistency）",
    "proc_answer_support_missing": "1 − mean(答案是否在 reasoning 中出现)",
    "proc_last_value_mismatch": "mean(最后推理数字 ≠ 最终答案)",
    "proc_repair_marker_rate": "repair/backtracking 词在 process 文本中的比例",
}


def _levenshtein(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
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
    d = _levenshtein(a, b)
    return d / max(len(a), len(b), 1)


def calc_seq(run: dict) -> list[str]:
    """Calculation token sequence before answer (content, not probability)."""
    trace = run.get("token_trace") or []
    a = (run.get("answer_span") or {}).get("start_token")
    pre = trace[: int(a)] if a is not None else trace
    seq: list[str] = []
    for t in pre:
        raw = (t.get("token") or "").strip()
        if not raw:
            continue
        if raw in BRACKET_CHARS:
            seq.append(raw)
            continue
        kind = _classify_token(raw)
        if kind in CALC_KINDS:
            seq.append(re.sub(r"\s+", "", raw))
    return seq


def answer_in_reasoning(run: dict) -> float:
    """1 if normalized answer appears in reasoning text (numbers/symbols)."""
    ans = str(run.get("answer_normalized") or "").strip()
    if not ans:
        return float("nan")
    reasoning = _reasoning_text(run).lower()
    ans_nums = extract_numbers(ans)
    if ans_nums:
        hits = [any(math_equal(str(n), str(r)) for r in extract_numbers(reasoning)) for n in ans_nums]
        if hits:
            return float(np.mean(hits))
    # string fallback
    ans_clean = re.sub(r"\s+", "", ans.lower())
    if ans_clean and ans_clean in re.sub(r"\s+", "", reasoning):
        return 1.0
    if ans.lower() in reasoning:
        return 1.0
    return 0.0


def last_number_mismatch(run: dict) -> float:
    """1 if last number in reasoning != final answer."""
    reasoning = _reasoning_text(run)
    ans = str(run.get("answer_normalized") or "").strip()
    nums = extract_numbers(reasoning)
    if not nums or not ans:
        return float("nan")
    z = nums[-1]
    ans_nums = extract_numbers(ans)
    if not ans_nums:
        return float(math_equal(str(z), ans) is False)
    return float(not any(math_equal(str(z), str(a)) for a in ans_nums))


def repair_rate(run: dict) -> float:
    text = _reasoning_text(run).lower()
    if not text:
        return float("nan")
    hits = len(REPAIR_PAT.findall(text))
    # rate per 100 chars to scale reasonably
    return float(hits / max(len(text) / 100.0, 1.0))


def extract_features(base: dict, pert_runs: list[dict]) -> dict[str, float]:
    base_seq = calc_seq(base)
    edits_base, supports, last_mm, repairs = [], [], [], []
    pert_seqs = [calc_seq(r) for r in pert_runs]

    for run, pseq in zip(pert_runs, pert_seqs):
        edits_base.append(_norm_edit(pseq, base_seq))
        s = answer_in_reasoning(run)
        if np.isfinite(s):
            supports.append(s)
        lm = last_number_mismatch(run)
        if np.isfinite(lm):
            last_mm.append(lm)
        rr = repair_rate(run)
        if np.isfinite(rr):
            repairs.append(rr)

    pair_edits = []
    all_runs = [base] + list(pert_runs)
    all_seqs = [base_seq] + pert_seqs
    for i in range(len(all_seqs)):
        for j in range(i + 1, len(all_seqs)):
            pair_edits.append(_norm_edit(all_seqs[i], all_seqs[j]))

    nan = float("nan")
    return {
        "proc_calc_edit_base": float(np.mean(edits_base)) if edits_base else nan,
        "proc_calc_edit_pair": float(np.mean(pair_edits)) if pair_edits else nan,
        "proc_answer_support_missing": 1.0 - float(np.mean(supports)) if supports else nan,
        "proc_last_value_mismatch": float(np.mean(last_mm)) if last_mm else nan,
        "proc_repair_marker_rate": float(np.mean(repairs)) if repairs else nan,
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
    if not any(np.isfinite(v) for v in feats.values()):
        return None
    return {"model": model_key, "seed": seed, "ds": ds, "y": int(label), "bd": bd, **feats}


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
                        jobs.append((rp, labels[rid], seed, ds, mk))
    return jobs


def load_rows(out_root: Path, workers: int) -> list[dict]:
    jobs = collect_jobs(out_root)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_parse_one, j) for j in jobs]
        for fu in as_completed(futs):
            r = fu.result()
            if r:
                rows.append(r)
    return rows


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
        med = np.nanmedian(Xtr, axis=0)
        for j in range(Xtr.shape[1]):
            Xtr[~np.isfinite(Xtr[:, j]), j] = med[j]
            Xte[~np.isfinite(Xte[:, j]), j] = med[j]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=3000)
        clf.fit((Xtr - mu) / sd, ytr)
        scs.append(roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
    return float(np.mean(scs)) if scs else float("nan")


def lodo_macro(rows: list[dict], cols: list[str]) -> float:
    vals = [lodo_per_ds(rows, cols, ds) for ds in MATH_DS]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def pooled_auroc(rows: list[dict], key: str) -> float:
    y = np.array([r["y"] for r in rows])
    s = np.array([r[key] for r in rows], float)
    m = np.isfinite(s)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return float("nan")
    a = roc_auc_score(y[m], s[m])
    return float(max(a, 1 - a))


def drop_wins(rows: list[dict], cols: list[str]) -> int:
    """Count cells (4 models × 3 ds) where fusion >= bd-only."""
    wins = 0
    for mk in MODELS:
        sub = [r for r in rows if r["model"] == mk]
        for ds in MATH_DS:
            if lodo_per_ds(sub, cols, ds) >= lodo_per_ds(sub, ["bd"], ds) - 1e-6:
                wins += 1
    return wins


def report(rows: list[dict], md_out: Path) -> None:
    bd_m = lodo_macro(rows, ["bd"])
    tw_ref = lodo_macro(rows, ["bd"])  # placeholder if no TW in cache

    configs = [(k, ["bd", k]) for k in FEAT_KEYS]
    configs.append(
        ("proc_edit+support_missing", ["bd", "proc_calc_edit_base", "proc_answer_support_missing"])
    )

    ranked = []
    for name, cols in configs:
        fus = lodo_macro(rows, cols)
        ranked.append((name, cols, fus, fus - bd_m, drop_wins(rows, cols)))

    ranked.sort(key=lambda x: -x[2])

    lines = [
        "# Process-token CONTENT 特征验证",
        "",
        f"> N={len(rows)}，4 模型 LODO。非 probability，看计算序列/支持/最后值/repair。",
        "",
        f"**bd-only macro**: {bd_m:.3f}",
        "",
        "## 结果",
        "",
        "| 配置 | bd+T macro | Δ vs bd | win/12 | pooled T | 叙事 |",
        "|------|------:|------:|------:|------:|------|",
    ]
    for name, cols, fus, d, w in ranked:
        tk = cols[-1] if len(cols) == 2 else name
        po = pooled_auroc(rows, cols[1]) if len(cols) == 2 else float("nan")
        narr = NARR.get(cols[1], "组合") if len(cols) == 2 else "edit + support_missing"
        lines.append(f"| {name} | {fus:.3f} | {d:+.3f} | {w}/12 | {po:.3f} | {narr} |")

    lines += ["", "## Qwen2.5-3B", ""]
    sub = [r for r in rows if r["model"] == "qwen25_3b"]
    lines.append("| 配置 | bd+T | tok-only |")
    lines.append("|------|------:|------:|")
    for name, cols, fus, _, _ in ranked:
        tok = lodo_macro(sub, [cols[1]]) if len(cols) == 2 else float("nan")
        lines.append(f"| {name} | {lodo_macro(sub, cols):.3f} | {tok:.3f} |")

    best = ranked[0]
    lines += [
        "",
        "## 判定",
        "",
        f"- 最佳: **{best[0]}** macro={best[2]:.3f} (Δ={best[3]:+.3f}, win={best[4]}/12)",
    ]
    if best[2] <= bd_m + 0.003:
        lines.append("- 未显著超过 bd-only → 建议停搜，定稿 `LODO(bd, TW_ent_sum)` 或 bd-only。")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/panda-outputs"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--md-out", type=Path, default=ROOT / "paper/tables/table_proc_content_features.md")
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
    bd = lodo_macro(rows, ["bd"])
    print(f"\nbd-only macro: {bd:.3f}\n")
    for k in FEAT_KEYS:
        fus = lodo_macro(rows, ["bd", k])
        q = lodo_macro([r for r in rows if r["model"] == "qwen25_3b"], ["bd", k])
        print(f"{k:<32} macro={fus:.3f} Δ={fus-bd:+.3f} qwen25={q:.3f} pooled={pooled_auroc(rows,k):.3f} win={drop_wins(rows,['bd',k])}/12")
    c6 = ["bd", "proc_calc_edit_base", "proc_answer_support_missing"]
    fus6 = lodo_macro(rows, c6)
    print(f"\nedit+support: macro={fus6:.3f} Δ={fus6-bd:+.3f} win={drop_wins(rows,c6)}/12")
    print(f"\nWrote {args.md_out}")


if __name__ == "__main__":
    main()
