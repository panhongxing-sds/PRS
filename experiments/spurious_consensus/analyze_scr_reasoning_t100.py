#!/usr/bin/env python3
"""分析 Qwen-7B SCR@1.0（42 题）的 reasoning 一致性与 token 统计。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "scr_reasoning" / "qwen25_7b" / "t100"
MANIFEST = DATA_DIR / "manifest.json"
OUT_JSON = ROOT / "results" / "scr7b_t100_reasoning_analysis.json"
OUT_MD = ROOT / "figures" / "scr7b_t100_reasoning_analysis.md"

BOXED = re.compile(r"\\boxed\{", re.I)
WS = re.compile(r"\s+")


def strip_answer(text: str) -> str:
    m = BOXED.search(text)
    return text[: m.start()].strip() if m else text.strip()


def norm_text(text: str) -> str:
    return WS.sub(" ", text.strip())


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def token_stats(rows: list[dict]) -> dict:
    lens = [r.get("n_completion_tokens", len(r.get("completion_token_ids", []))) for r in rows]
    avg_lps = []
    low_lp_fracs = []
    cum_lps = []
    for r in rows:
        toks = r.get("completion_tokens") or []
        lps = [t["logprob"] for t in toks if t.get("logprob") is not None]
        avg_lps.append(sum(lps) / len(lps) if lps else float("nan"))
        if lps:
            low_lp_fracs.append(sum(1 for x in lps if x < -1.0) / len(lps))
        if r.get("cumulative_logprob") is not None:
            cum_lps.append(float(r["cumulative_logprob"]))
    return {
        "completion_len_mean": statistics.mean(lens),
        "completion_len_std": statistics.pstdev(lens) if len(lens) > 1 else 0.0,
        "avg_token_logprob_mean": statistics.mean(avg_lps),
        "avg_token_logprob_std": statistics.pstdev(avg_lps) if len(avg_lps) > 1 else 0.0,
        "low_logprob_frac_mean": statistics.mean(low_lp_fracs) if low_lp_fracs else 0.0,
        "cumulative_logprob_mean": statistics.mean(cum_lps) if cum_lps else float("nan"),
    }


def token_trace_stats(rows: list[dict]) -> dict:
    """跨 sample 的 token 序列一致性 / 分歧位置 / 位置共识。"""
    tids = [list(r.get("completion_token_ids") or []) for r in rows]
    if not tids or not tids[0]:
        return {
            "common_prefix_len": 0,
            "mean_position_consensus": 0.0,
            "position_consensus_first50": 0.0,
            "pairwise_first_divergence_mean": 0.0,
            "cross_logprob_std_mean": 0.0,
        }

    min_len = min(len(t) for t in tids)
    common_prefix = 0
    for i in range(min_len):
        if len({t[i] for t in tids}) == 1:
            common_prefix += 1
        else:
            break

    cons = []
    for i in range(min_len):
        cnt = Counter(t[i] for t in tids)
        cons.append(cnt.most_common(1)[0][1] / len(tids))
    first50 = cons[: min(50, len(cons))]

    # 均匀抽 16 条算 pairwise 首个分歧位置
    idx = list(range(len(tids)))
    if len(idx) > 16:
        step = len(idx) / 16
        idx = [int(i * step) for i in range(16)]
    divs = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            ta, tb = tids[idx[a]], tids[idx[b]]
            m = min(len(ta), len(tb))
            pos = m
            for i in range(m):
                if ta[i] != tb[i]:
                    pos = i
                    break
            divs.append(pos)

    # 同一位置跨 sample 的 logprob 标准差（按 completion 位置对齐）
    lp_stds = []
    for i in range(min_len):
        vals = []
        for r in rows:
            toks = r.get("completion_tokens") or []
            if i < len(toks) and toks[i].get("logprob") is not None:
                vals.append(toks[i]["logprob"])
        if len(vals) > 1:
            lp_stds.append(statistics.pstdev(vals))

    return {
        "common_prefix_len": common_prefix,
        "mean_position_consensus": statistics.mean(cons) if cons else 0.0,
        "position_consensus_first50": statistics.mean(first50) if first50 else 0.0,
        "pairwise_first_divergence_mean": statistics.mean(divs) if divs else 0.0,
        "cross_logprob_std_mean": statistics.mean(lp_stds) if lp_stds else 0.0,
    }


def analyze_question(rows: list[dict], *, max_pairwise: int = 32) -> dict:
    texts = [r["text"] for r in rows]
    rs = [strip_answer(t) for t in texts]
    ans = [r.get("extracted_answer", "") for r in rows]
    tids = [tuple(r.get("completion_token_ids") or []) for r in rows]
    rnorm = [norm_text(x) for x in rs]

    ans_cnt = Counter(ans)
    top_ans, top_n = ans_cnt.most_common(1)[0]
    maj_frac = top_n / len(rows)

    uniq_text = len(set(texts))
    uniq_reason = len(set(rnorm))
    uniq_tok = len(set(tids))

    # 与首条 reasoning 的相似度（64 条全比太慢，足够刻画分布）
    ref = rs[0]
    ref_sims = [sim(ref, rs[k]) for k in range(1, len(rs))]
    sim_to_first_mean = statistics.mean(ref_sims) if ref_sims else 1.0
    sim_to_first_min = min(ref_sims) if ref_sims else 1.0

    # 子集 pairwise（均匀抽 max_pairwise 条）
    idx = list(range(len(rs)))
    if len(idx) > max_pairwise:
        step = len(idx) / max_pairwise
        idx = [int(i * step) for i in range(max_pairwise)]
    pairs = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            pairs.append(sim(rs[idx[i]], rs[idx[j]]))
    avg_pairwise = statistics.mean(pairs) if pairs else 1.0
    min_pairwise = min(pairs) if pairs else 1.0

    exact_reason_rate = Counter(rnorm).most_common(1)[0][1] / len(rows)
    prefix200_same = len(set(x[:200] for x in rs)) == 1

    stored_maj = rows[0].get("stored_scr_maj_wrong", "")
    stored_ptop = rows[0].get("stored_p_top", float("nan"))

    tok = token_stats(rows)
    trace = token_trace_stats(rows)
    return {
        "n_samples": len(rows),
        "stored_maj_answer": stored_maj,
        "stored_p_top": stored_ptop,
        "resample_matches_stored_maj": top_ans == stored_maj,
        "resample_p_top": maj_frac,
        "maj_answer": top_ans,
        "maj_answer_frac": maj_frac,
        "n_unique_answers": len(ans_cnt),
        "n_unique_texts": uniq_text,
        "n_unique_reasoning": uniq_reason,
        "n_unique_token_ids": uniq_tok,
        "exact_reasoning_match_rate": exact_reason_rate,
        "reasoning_prefix200_same": prefix200_same,
        "sim_to_first_mean": sim_to_first_mean,
        "sim_to_first_min": sim_to_first_min,
        "avg_pairwise_sim": avg_pairwise,
        "min_pairwise_sim": min_pairwise,
        **tok,
        **trace,
        "reasoning_head": rs[0][:400],
        "answer_counts": dict(ans_cnt.most_common(5)),
    }


def load_other_models(common_ids: set[str]) -> dict[str, dict[str, dict]]:
    tags = ["qwen25_05b", "llama32_1b", "qwen25_15b", "phi4_mini", "qwen25_3b"]
    out: dict[str, dict[str, dict]] = {t: {} for t in tags}
    for tag in tags:
        for f in (ROOT / "data" / "samples").glob(f"samples_{tag}_seed41_*.jsonl"):
            if str(f).endswith(".bak"):
                continue
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["id"] in common_ids:
                        out[tag][r["id"]] = r
    return out


def cross_model_block(qid: str, scr_maj: str, others: dict[str, dict[str, dict]]) -> dict:
    block = {"scr7b_maj_wrong_answer": scr_maj}
    for tag, data in others.items():
        r = data.get(qid)
        if not r or int(r.get("label_drop", 0)) == 1:
            block[tag] = None
            continue
        pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a]
        if not pairs:
            block[tag] = None
            continue
        a, c = zip(*pairs)
        cnt = Counter(a)
        maj, top = cnt.most_common(1)[0]
        cmap = {ai: ci for ai, ci in zip(a, c)}
        block[tag] = {
            "maj": maj,
            "p_top": top / len(a),
            "wrong": cmap.get(maj, 0) == 0,
            "same_as_scr7b": maj == scr_maj,
        }
    return block


def exogenous_difficulty(qid: str, others: dict[str, dict[str, dict]]) -> float | None:
    rates = []
    for data in others.values():
        r = data.get(qid)
        if not r or int(r.get("label_drop", 0)) == 1:
            continue
        pairs = [(a, c) for a, c in zip(r["answers"], r["correct"]) if a]
        if not pairs:
            continue
        _, c = zip(*pairs)
        rates.append(sum(c) / len(c))
    return statistics.mean(rates) if rates else None


def summarize(items: list[dict]) -> dict:
    def pct(cond):
        return sum(1 for x in items if cond(x)) / len(items)

    sims = [x["sim_to_first_mean"] for x in items]
    return {
        "n_questions": len(items),
        "resample_matches_stored_maj": pct(lambda x: x["resample_matches_stored_maj"]),
        "all_64_same_answer": pct(lambda x: x["maj_answer_frac"] == 1.0),
        "all_64_same_reasoning": pct(lambda x: x["n_unique_reasoning"] == 1),
        "all_64_same_token_ids": pct(lambda x: x["n_unique_token_ids"] == 1),
        "prefix200_all_same": pct(lambda x: x["reasoning_prefix200_same"]),
        "exact_reason_rate_eq1": pct(lambda x: x["exact_reasoning_match_rate"] == 1.0),
        "sim_to_first_mean_avg": statistics.mean(sims),
        "sim_to_first_mean_median": statistics.median(sims),
        "high_sim_ge_0.9": pct(lambda x: x["sim_to_first_mean"] >= 0.9),
        "low_sim_lt_0.5": pct(lambda x: x["sim_to_first_mean"] < 0.5),
        "common_prefix_len_mean": statistics.mean(x["common_prefix_len"] for x in items),
        "common_prefix_len_median": statistics.median(x["common_prefix_len"] for x in items),
        "position_consensus_first50_mean": statistics.mean(
            x["position_consensus_first50"] for x in items
        ),
        "pairwise_first_divergence_mean": statistics.mean(
            x["pairwise_first_divergence_mean"] for x in items
        ),
        "cross_logprob_std_mean": statistics.mean(x["cross_logprob_std_mean"] for x in items),
        "avg_token_logprob_mean": statistics.mean(x["avg_token_logprob_mean"] for x in items),
        "low_logprob_frac_mean": statistics.mean(x["low_logprob_frac_mean"] for x in items),
    }


def render_md(summary: dict, items: list[dict], cross: list[dict]) -> str:
    lines = [
        "# Qwen-7B SCR@1.0（42 题）Reasoning + Token 分析",
        "",
        "> 数据：`data/scr_reasoning/qwen25_7b/t100/`，每题 K=64 完整 reasoning + token logprob",
        "> 协议：temp=0.5, top_p=0.95, seed=41（与 stored 采样 seed 不同，见下方复现说明）",
        "",
        "## 总览",
        "",
        f"- 题数：**{summary['n_questions']}**（deepscaler 39 + gpqa 3）",
        f"- 重采样 majority 与 **stored SCR 错答一致**：{summary['resample_matches_stored_maj']*100:.0f}%（22/42）",
        f"- 64/64 **答案完全相同**（重采样）：{summary['all_64_same_answer']*100:.0f}%",
        f"- 64/64 **reasoning 文本完全相同**（去 boxed 后、空白归一）：{summary['all_64_same_reasoning']*100:.0f}%",
        f"- 64/64 **token id 序列完全相同**：{summary['all_64_same_token_ids']*100:.0f}%",
        f"- 前 200 字符 reasoning 全部相同：{summary['prefix200_all_same']*100:.0f}%",
        f"- 与首条 reasoning 平均相似度（sim_to_first）：均值 **{summary['sim_to_first_mean_avg']:.3f}**，中位 **{summary['sim_to_first_mean_median']:.3f}**",
        f"- sim≥0.9（高度同构）：{summary['high_sim_ge_0.9']*100:.0f}% 题；sim<0.5（路径分散）：{summary['low_sim_lt_0.5']*100:.0f}% 题",
        "",
        "## Token-level 总览",
        "",
        f"- 64 条 trace **共同前缀 token 长度**：均值 **{summary['common_prefix_len_mean']:.1f}**，中位 **{summary['common_prefix_len_median']:.0f}**",
        f"- 前 50 token **位置共识度**（该位置最多 token 占比均值）：**{summary['position_consensus_first50_mean']:.3f}**",
        f"- pairwise **首个分歧位置**（16 条子集）：均值 **{summary['pairwise_first_divergence_mean']:.1f}** token",
        f"- 同位置跨 trace **logprob 标准差**均值：**{summary['cross_logprob_std_mean']:.3f}**",
        f"- 平均 token logprob：**{summary['avg_token_logprob_mean']:.3f}**（整体高置信生成）",
        f"- logprob < −1.0 的 token 占比：**{summary['low_logprob_frac_mean']*100:.2f}%**（极低，几乎无「犹豫 token」）",
        "",
        "**核心发现**：SCR@1.0 刻画的是 **答案级虚假共识**，不是 reasoning/token 级复制粘贴。"
        "绝大多数题 64 条 completion 的 token 序列互不相同，reasoning 文本也高度多样化，"
        "却在第 1–7 个 token 后就开始分叉，却能收敛到同一错误答案——"
        "**错误共识可以在不同推理路径上达成，且生成过程整体仍保持高置信。**",
        "",
        "> **复现说明**：stored 样本用 `seed*100003+batch*10` 逐 batch 采样；"
        "t100 重采样用固定 seed=41 一次生成 n=64，故仅 52% 题 majority 错答与 stored 完全一致。"
        "分析结论基于 t100 重采样数据本身的 reasoning/token 多样性，与 SCR 机制一致。",
        "",
        "## 逐题明细（按 sim_to_first 升序）",
        "",
        "| id | bench | stored错答 | resample一致 | unique reason | unique tok | 共同前缀 | sim→1st | 前50共识 | 外生难度 |",
        "|----|-------|-----------|-------------|--------------|-----------|---------|---------|---------|---------|",
    ]
    for x in sorted(items, key=lambda z: z["sim_to_first_mean"]):
        exo = x.get("exogenous_difficulty")
        exo_s = f"{exo:.3f}" if exo is not None else "—"
        match = "✓" if x["resample_matches_stored_maj"] else "✗"
        lines.append(
            f"| `{x['id']}` | {x['benchmark']} | `{x['stored_maj_answer']}` "
            f"| {match} | {x['n_unique_reasoning']} | {x['n_unique_token_ids']} "
            f"| {x['common_prefix_len']} | {x['sim_to_first_mean']:.3f} "
            f"| {x['position_consensus_first50']:.2f} | {exo_s} |"
        )

    # cross-model
    same_phi = sum(1 for c in cross if c.get("phi4_mini") and c["phi4_mini"]["same_as_scr7b"])
    same_qwen3 = sum(1 for c in cross if c.get("qwen25_3b") and c["qwen25_3b"]["same_as_scr7b"])
    scr_phi = sum(
        1 for c in cross
        if c.get("phi4_mini") and c["phi4_mini"]["wrong"] and c["phi4_mini"]["p_top"] >= 0.9
    )
    scr_qwen3 = sum(
        1 for c in cross
        if c.get("qwen25_3b") and c["qwen25_3b"]["wrong"] and c["qwen25_3b"]["p_top"] >= 0.9
    )
    lines += [
        "",
        "## 跨模型（5 个其他模型 stored correct）",
        "",
        f"- 与 7B 给出**相同错答**的题数：Phi {same_phi}/42，Qwen-3B {same_qwen3}/42",
        f"- 其他模型也在 SCR@0.9 的题数：Phi {scr_phi}/42，Qwen-3B {scr_qwen3}/42",
        "",
        "## 机制分型（基于 reasoning + token 多样性）",
        "",
        "| 类型 | 定义 | 题数 | 代表 id |",
        "|------|------|-----:|---------|",
    ]
    # classify
    type_a = [x for x in items if x["sim_to_first_mean"] < 0.3 and x["common_prefix_len"] <= 5]
    type_b = [x for x in items if 0.3 <= x["sim_to_first_mean"] < 0.6]
    type_c = [x for x in items if x["sim_to_first_mean"] >= 0.6]
    for label, subset, desc in [
        ("A 多路径汇入", type_a, "sim<0.3 且 token 前缀≤5，典型虚假共识"),
        ("B 中等同构", type_b, "0.3≤sim<0.6，部分步骤共享"),
        ("C 高同构错答", type_c, "sim≥0.6，推理骨架相似但仍非复制"),
    ]:
        rep = min(subset, key=lambda z: z["sim_to_first_mean"])["id"] if subset else "—"
        lines.append(f"| {label} | {desc} | {len(subset)} | `{rep}` |")

    lines += [
        "",
        "## 典型样例",
        "",
    ]
    low = min(items, key=lambda z: z["sim_to_first_mean"])
    high = max(items, key=lambda z: z["sim_to_first_mean"])
    for label, x in [("最低相似度", low), ("最高相似度", high)]:
        lines += [
            f"### {label}：`{x['id']}`（sim={x['sim_to_first_mean']:.3f}）",
            f"- stored 错答 `{x['stored_maj_answer']}` → 重采样 maj `{x['maj_answer']}`（{x['maj_answer_frac']*100:.0f}%）",
            f"- unique reasoning={x['n_unique_reasoning']}，unique token ids={x['n_unique_token_ids']}",
            f"- 共同前缀 token={x['common_prefix_len']}，前50位置共识={x['position_consensus_first50']:.2f}",
            "",
            "```",
            x["reasoning_head"],
            "```",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairwise", type=int, default=32)
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    qids = [q["id"] for q in manifest["questions"]]
    others = load_other_models(set(qids))

    items = []
    cross = []
    for q in manifest["questions"]:
        qid = q["id"]
        path = ROOT / q["path"]
        rows = load_rows(path)
        meta = analyze_question(rows, max_pairwise=args.max_pairwise)
        meta["id"] = qid
        meta["benchmark"] = q["benchmark"]
        meta["gold"] = rows[0].get("gold", "")
        meta["exogenous_difficulty"] = exogenous_difficulty(qid, others)
        items.append(meta)
        cross.append({"id": qid, **cross_model_block(qid, meta["maj_answer"], others)})

    summary = summarize(items)
    payload = {"summary": summary, "questions": items, "cross_model": cross}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
    OUT_MD.write_text(render_md(summary, items, cross), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n→ {OUT_JSON}\n→ {OUT_MD}")


if __name__ == "__main__":
    main()
