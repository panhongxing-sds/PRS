"""Step 2 [CPU]: answer-level UQ 极限/盲区分析 + 出图。

输入: sample_answers.py 产出的 data/samples_*.jsonl
输出: results/metrics.json + figures/fig1_panels.png + figures/fig2_dualaxis.png

实现要点:
- n-sweep 用 bootstrap 子采样（对每题从 K 个样本有放回抽 n 个），免 GPU、自带误差带；
- 盲区率分母固定为 majority@Kmax 判错的题；
- 核心新指标: 残余错误的平均 p_top 随 n 的变化（预期上升）；
- 多 seed 时计算 stable-wrong（跨 seed 同一错误答案）作为不可约盲区证据。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import yaml

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------- 数据加载 -----------------------------

def load_samples(globs: list[str], drop_label_drop: bool) -> list[dict]:
    rows: list[dict] = []
    for g in globs:
        for path in sorted(glob(g)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if drop_label_drop and int(r.get("label_drop", 0)) == 1:
                        continue
                    answers = r.get("answers", [])
                    correct = r.get("correct", [])
                    # 过滤空答案样本（解析失败）；保留至少 2 个有效样本的题
                    pairs = [(a, c) for a, c in zip(answers, correct) if a != ""]
                    if len(pairs) < 2:
                        continue
                    a, c = zip(*pairs)
                    rows.append({
                        "id": r["id"],
                        "dataset": r["dataset"],
                        "seed": r.get("seed"),
                        "answers": list(a),
                        "correct": list(c),
                    })
    return rows


# ----------------------------- 单题工具 -----------------------------

def answer_correct_map(answers: list[str], correct: list[int]) -> dict[str, int]:
    m: dict[str, int] = {}
    for a, c in zip(answers, correct):
        m.setdefault(a, int(c))
    return m


def dist_stats(sub_answers: list[str]) -> tuple[str, float, float]:
    """返回 (majority_answer, p_top, entropy)。"""
    n = len(sub_answers)
    cnt = Counter(sub_answers)
    majority, top = cnt.most_common(1)[0]
    p_top = top / n
    ent = 0.0
    for _, c in cnt.items():
        p = c / n
        ent -= p * np.log(p)
    return majority, p_top, ent


# ----------------------------- 主分析 -----------------------------

def analyze(rows: list[dict], cfg_an: dict, rng: np.random.Generator) -> dict:
    n_grid = cfg_an["n_grid"]
    B = cfg_an["bootstrap_draws"]
    taus = cfg_an["blindspot_taus"]

    # 预处理：每题 -> answers, correct, correct_map, Kmax majority 判错标志
    Q = []
    for r in rows:
        ans = r["answers"]
        cor = r["correct"]
        cmap = answer_correct_map(ans, cor)
        maj_full, _, _ = dist_stats(ans)
        Q.append({
            "id": r["id"], "dataset": r["dataset"], "seed": r["seed"],
            "answers": ans, "K": len(ans), "cmap": cmap,
            "maj_full": maj_full,
            "wrong_full": 0 if cmap.get(maj_full, 0) == 1 else 1,
        })

    W_idx = [i for i, q in enumerate(Q) if q["wrong_full"] == 1]  # 固定分母

    per_n = {}
    wrong_freq_per_n = {}  # n -> 每题 majority-wrong 的频率（用于预算稳健盲区）
    for n in n_grid:
        auroc_uptop, auroc_ent, accs, resid_conf = [], [], [], []
        scr_dyn = {tau: [] for tau in taus}  # |wrong@n & p_top>=τ| / |wrong@n|
        # 每题在该 n 下的平均 p_top（用于盲区率），跨 draw 平均
        ptop_accum = np.zeros(len(Q))
        ptop_count = np.zeros(len(Q))
        wrong_accum = np.zeros(len(Q))
        for _ in range(B):
            scores_uptop = np.empty(len(Q))
            scores_ent = np.empty(len(Q))
            label_wrong = np.empty(len(Q), dtype=int)
            maj_correct = np.empty(len(Q), dtype=int)
            ptop_arr = np.empty(len(Q))
            for i, q in enumerate(Q):
                K = q["K"]
                idx = rng.integers(0, K, size=min(n, K)) if n <= K else rng.integers(0, K, size=n)
                sub = [q["answers"][j] for j in idx]
                maj, p_top, ent = dist_stats(sub)
                scores_uptop[i] = 1.0 - p_top
                scores_ent[i] = ent
                wrong = 0 if q["cmap"].get(maj, 0) == 1 else 1
                label_wrong[i] = wrong
                maj_correct[i] = 1 - wrong
                ptop_arr[i] = p_top
            ptop_accum += ptop_arr
            ptop_count += 1
            wrong_accum += label_wrong
            accs.append(maj_correct.mean())
            # 残余错误自信度: E[p_top | majority wrong]
            if label_wrong.sum() > 0:
                resid_conf.append(ptop_arr[label_wrong == 1].mean())
                wp = ptop_arr[label_wrong == 1]
                for tau in taus:
                    scr_dyn[tau].append(float((wp >= tau).mean()))
            # AUROC（需要正负样本都存在）
            if roc_auc_score is not None and 0 < label_wrong.sum() < len(label_wrong):
                auroc_uptop.append(roc_auc_score(label_wrong, scores_uptop))
                auroc_ent.append(roc_auc_score(label_wrong, scores_ent))

        mean_ptop_per_q = ptop_accum / np.maximum(ptop_count, 1)
        wrong_freq_per_n[n] = wrong_accum / max(B, 1)
        # 盲区率: 在固定分母 W 上，mean p_top >= tau 的比例
        bsr = {}
        if W_idx:
            wp = mean_ptop_per_q[W_idx]
            for tau in taus:
                bsr[str(tau)] = float((wp >= tau).mean())
        else:
            bsr = {str(t): float("nan") for t in taus}

        def ci(a):
            a = np.asarray(a, dtype=float)
            a = a[~np.isnan(a)]
            if a.size == 0:
                return [float("nan")] * 3
            return [float(a.mean()),
                    float(np.percentile(a, 2.5)),
                    float(np.percentile(a, 97.5))]

        per_n[n] = {
            "majority_acc": ci(accs),
            "auroc_uptop": ci(auroc_uptop),
            "auroc_entropy": ci(auroc_ent),
            "residual_error_conf": ci(resid_conf),  # 核心新指标
            "blindspot_rate": bsr,
            "spurious_consensus_rate": {str(t): ci(scr_dyn[t]) for t in taus},
        }

    # stable-wrong：跨 seed 同一错误答案
    stable = stable_wrong(Q)
    # 预算稳健盲区（单 seed 也可算）：majority 在所有 n 下都判错的题，占 wrong@Kmax 的比例
    if W_idx:
        persist = np.ones(len(Q), dtype=bool)
        for n in n_grid:
            persist &= (wrong_freq_per_n[n] >= 0.5)
        budget_stable = int(persist[W_idx].sum())
        stable["budget_stable_wrong"] = budget_stable
        stable["budget_stable_wrong_frac"] = budget_stable / len(W_idx)
        stable["budget_stable_note"] = "majority 在所有 n∈grid 下都判错 / wrong@Kmax"

    return {
        "n_questions": len(Q),
        "n_wrong_full": len(W_idx),
        "wrong_rate_full": len(W_idx) / max(len(Q), 1),
        "per_n": {str(k): v for k, v in per_n.items()},
        "stable_wrong": stable,
    }


def stable_wrong(Q: list[dict]) -> dict:
    """跨 seed 都判错且 majority 同一错误答案的题占（并集错误）的比例。"""
    by_key = defaultdict(dict)  # (dataset,id) -> {seed: (wrong_full, maj_full)}
    for q in Q:
        by_key[(q["dataset"], q["id"])][q["seed"]] = (q["wrong_full"], q["maj_full"])
    seeds = sorted({q["seed"] for q in Q})
    union_wrong, stable_cnt = 0, 0
    for _, sd in by_key.items():
        wrongs = [v[0] for v in sd.values()]
        majs = [v[1] for v in sd.values()]
        if any(w == 1 for w in wrongs):
            union_wrong += 1
        if len(sd) >= 2:
            if all(w == 1 for w in wrongs) and len(set(majs)) == 1:
                stable_cnt += 1
    return {
        "n_seeds": len(seeds),
        "union_wrong": union_wrong,
        "stable_wrong": stable_cnt,
        "stable_wrong_frac": (stable_cnt / union_wrong) if union_wrong else float("nan"),
        "note": ("跨 seed 一致错误" if len(seeds) >= 2
                 else "单 seed：stable_wrong 不可算，需多 seed"),
    }


# ----------------------------- 出图 -----------------------------

def make_figures(metrics: dict, taus: list[float], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    ns = sorted(int(k) for k in metrics["per_n"].keys())
    pn = metrics["per_n"]

    def band(key):
        m = [pn[str(n)][key][0] for n in ns]
        lo = [pn[str(n)][key][1] for n in ns]
        hi = [pn[str(n)][key][2] for n in ns]
        return np.array(m), np.array(lo), np.array(hi)

    # Fig 1: 2x2 多面板
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        ("majority_acc", "(a) Majority-vote accuracy", "accuracy"),
        ("auroc_uptop", "(b) Answer-level UQ AUROC (1 - p_top)", "AUROC"),
        (None, "(c) Blind-spot rate (denom = wrong@Kmax)", "blind-spot rate"),
        ("residual_error_conf", "(d) Residual error confidence  E[p_top | wrong]", "mean p_top"),
    ]
    for ax, (key, title, ylab) in zip(axes.flat, specs):
        if key is not None:
            m, lo, hi = band(key)
            ax.plot(ns, m, "-o", color="tab:blue")
            ax.fill_between(ns, lo, hi, alpha=0.2, color="tab:blue")
        else:
            for tau in taus:
                ys = [pn[str(n)]["blindspot_rate"][str(tau)] for n in ns]
                ax.plot(ns, ys, "-s", label=f"τ={tau}")
            ax.legend(fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("sampling budget n")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Answer-level UQ improves with budget but retains a persistent blind spot",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_dir / "fig1_panels.png", dpi=160)
    plt.close(fig)

    # Fig 2: 主图双纵轴（n≥2；右轴 = Confident Wrong@0.9，动态分母 wrong@n）
    ns_main = [n for n in ns if n >= 2]
    tau = 0.9
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    m = [pn[str(n)]["auroc_uptop"][0] for n in ns_main]
    lo = [pn[str(n)]["auroc_uptop"][1] for n in ns_main]
    hi = [pn[str(n)]["auroc_uptop"][2] for n in ns_main]
    ax1.plot(ns_main, m, "-o", color="#2166ac", linewidth=2, markersize=6, label="AUROC (1 − p_top)")
    ax1.fill_between(ns_main, lo, hi, alpha=0.2, color="#2166ac")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ns_main)
    ax1.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel("number of samples per question (n)")
    ax1.set_ylabel("AUROC of 1 − p_top", color="#2166ac")
    ax1.tick_params(axis="y", labelcolor="#2166ac")
    ax1.set_ylim(0.75, 0.90)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    cw_key = "spurious_consensus_rate"
    cw_m = [pn[str(n)][cw_key][str(tau)][0] for n in ns_main]
    cw_lo = [pn[str(n)][cw_key][str(tau)][1] for n in ns_main]
    cw_hi = [pn[str(n)][cw_key][str(tau)][2] for n in ns_main]
    ax2.plot(ns_main, cw_m, "--s", color="#b2182b", linewidth=2, markersize=6,
             label=f"Confident Wrong@{tau}")
    ax2.fill_between(ns_main, cw_lo, cw_hi, alpha=0.15, color="#b2182b")
    ax2.set_ylabel(f"Confident Wrong@{tau}", color="#b2182b")
    ax2.tick_params(axis="y", labelcolor="#b2182b")
    ax2.set_ylim(0.0, 0.25)

    n_q = metrics.get("n_questions", "?")
    ax1.set_title(
        "More Samples Improve Answer-Level UQ but Do Not Eliminate Confident Wrongs\n"
        f"(pooled, N={n_q} questions)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_dualaxis.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--samples", nargs="+", required=True,
                    help="glob(s)，如 'data/samples/samples_*_seed41_*.jsonl'")
    ap.add_argument("--out", default="results/metrics.json")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg_an = cfg["analysis"]
    rng = np.random.default_rng(0)

    rows = load_samples(args.samples, cfg_an.get("drop_label_drop", True))
    if not rows:
        raise SystemExit("没有可用样本，检查 --samples glob 与 data/ 产物。")
    print(f"载入 {len(rows)} 题（{len({(r['dataset'], r['seed']) for r in rows})} 个 dataset×seed）")

    metrics = analyze(rows, cfg_an, rng)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    make_figures(metrics, cfg_an["blindspot_taus"], Path(args.fig_dir))

    print(f"\n题数={metrics['n_questions']}  错误率@Kmax={metrics['wrong_rate_full']:.3f}")
    print(f"stable-wrong: {metrics['stable_wrong']}")
    print(f"指标 → {args.out}\n图 → {args.fig_dir}/fig1_panels.png, fig2_dualaxis.png")


if __name__ == "__main__":
    main()
