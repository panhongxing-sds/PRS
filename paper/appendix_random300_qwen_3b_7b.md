# 附录：DeepScaleR random300 · Qwen2.5-3B vs 7B 公平对比

**更新日期：** 2026-07-02  
**Cohort：** `paper/analysis/deepscaler_random300_meta.json`（**n=300**，`seed=42`，自 2000 题随机抽样）  
**LaTeX（论文正文）：** `paper/iclr2026/panda_iclr2026.tex` → Appendix~\ref{app:random300-fair}（\S\ref{app:random300-fair}: Paired Fair Comparison on DeepScaleR (Random300)）；表 \ref{tab:random300-primary}、\ref{tab:random300-risk}（**附录以高密度表格为主，不引用三联图**；`figures/random300_planb_comparison.png` 与 `random300_confident_wrong.png` 仅作本地审阅）。

**公平解码预算：** `SE_SAMPLES=0`，`N_REPHRASES=4`，`WEIGHT_SEEDS=42,43,44,45` → **每题 9 次 decode**（1 greedy + 4×2 rephrase 权重路径）；PANDA@9 公平多数票与 SC@9 使用相同 9 路预算。

---

## 呈现方式（表格优先）

附录 LaTeX 使用 **两张综合表** 替代三联图，便于审计与并排比较 3B/7B：

| LaTeX 标签 | 内容 |
|------------|------|
| `tab:random300-primary` | 准确率（SC@9 / PANDA@9 / greedy a0）、$\Delta_{\mathrm{vote}}$、UQ（AUROC/AUPRC）、eval 子集 |
| `tab:random300-risk` | ConfWrong@$\tau\in\{0.9,8/9,7/9\}$、SC 队列统计、PANDA collapse 计数 |

7B SC **严格 n=246** 敏感性（无 `answers_orig` fallback）见 `random300_planb_metrics.json` → `metrics_strict_sc_vote_n246`，正文附录以段落脚注形式收录。

---

## 实验协议备注

| 项 | 说明 |
|----|------|
| 数据集 | DeepScaleR（random300 子集） |
| SC@9 | K=9 样本文件多数票；`p_top` = 最高票占比 / 9 |
| PANDA@9 公平票 | 与 Plan B 脚本一致：9 路 PANDA 解码上的多数票 |
| PANDA greedy a0 | 单次 greedy 答案准确率 |
| UQ 标签 | `label_wrong_clean`（greedy a0 是否错误）；**论文主表 eval** 为去掉 `label_drop` 的子集（3B **n=228**，7B **n=239**）；**全量 n=300** 亦报告 |
| SC UQ 分数 | `1 - p_top`（SC@9 共识置信的反号） |
| PANDA UQ 分数 | summary 中 `PANDA`（7B）；3B 历史 summary 字段为 `PRS`，指标脚本已 **PRS→PANDA 回退**（与旧 `random300_fair_baselines.json` 一致） |
| 7B SC 修复 | **106/300** 行 `answers[]` 为空；SC 取 `answers` 与 **`answers_orig`（K=64 前缀）** 中前 9 个非空答案（见 `random300_qwen25_7b_planb_metrics.json` → `sc_k9_note`） |

---

## 表 1（LaTeX `tab:random300-primary`）· 准确率 + UQ

| 方法 | Qwen2.5-3B | Qwen2.5-7B | Δ(7B−3B) |
|------|------------|------------|----------|
| SC@9 多数票 | 44.0% (132/300) | 50.7% (152/300) | +6.7 pp |
| PANDA@9 公平多数票 | 50.7% (152/300) | 58.3% (175/300) | +7.7 pp |
| PANDA greedy a0 | 42.0% (126/300) | 55.3% (166/300) | +13.3 pp |
| **Δ(PANDA@9 − SC)** | **+6.7 pp** | **+7.7 pp** | — |
| Δ(PANDA greedy − SC) | -2.0 pp | +4.7 pp | — |

**论文 eval 子集（无 label_drop）：**

| 方法 | 3B (n=228) | 7B (n=239) |
|------|------------------|------------------|
| SC@9 | 46.5% | 56.5% |
| PANDA@9 | 50.4% | 60.7% |
| Δ vote | +3.95 pp | +4.18 pp |

---


（UQ 与准确率已合并进 LaTeX 主表 `tab:random300-primary`；`label_drop`：3B **72**，7B **61**。）

---

## 表 2（LaTeX `tab:random300-risk`）· ConfWrong + SC 队列（n=300）

τ=**0.9**、**8/9**、**7/9**；列为 **全体样本率 (计数)** / **给定多数票错误条件下的率**（见 `random300_confident_wrong.json` 与各 `*_planb_metrics.json`）。

| τ | 方法 | Qwen2.5-3B | Qwen2.5-7B |
|---|------|------------|------------|
| 0.9 | SC@9 | 3.7% (11) / 6.5% | 6.7% (20) / 13.5% |
| 0.9 | PANDA@9 | 3.7% (11) / 7.4% | 4.3% (13) / 10.4% |
| 8/9 | SC@9 | 5.3% (16) / 9.5% | 8.7% (26) / 17.6% |
| 8/9 | PANDA@9 | 4.7% (14) / 9.5% | 6.3% (19) / 15.2% |
| 7/9 | SC@9 | 7.0% (21) / 12.5% | 11.3% (34) / 23.0% |
| 7/9 | PANDA@9 | 6.7% (20) / 13.5% | 8.7% (26) / 20.8% |

**SC@9 队列：** blind@9 109 (36.3%) / 94 (31.3%)；mean `p_top` 0.545 / 0.646。PANDA a0 错误且 `bd≈0`：3B **40**，7B **42**。

---

## 跨尺度要点（3B → 7B）

- **SC@9** 自 44.0% 升至 50.7%（+6.7 pp）；**PANDA@9** 自 50.7% 升至 58.3%。
- **PANDA 相对 SC 的投票增益** 稳定在约 **+6.7～+7.7 pp**（n=300）。
- **UQ（eval 子集）** PANDA 相对 SC 约 **+3.1～+4.0 pp AUROC**（3B n=228，7B n=239），量级一致。
- 7B **blind@9** 更低、**p_top 均值**更高，反映更强共识；ConfWrong@τ 在 7B 上 SC 侧略高，PANDA 投票仍低于或接近 3B PANDA 水平。

---

## 可复现产物（JSON / join）

| 文件 | 内容 |
|------|------|
| `paper/analysis/random300_qwen25_3b_planb_metrics.json` | 3B 全量指标（n=300 + n=228 eval） |
| `paper/analysis/random300_qwen25_7b_planb_metrics.json` | 7B 全量指标（n=300 + n=239 eval） |
| `paper/analysis/random300_planb_metrics.json` | Plan B 汇总（3B + Llama；3B 与上表一致） |
| `paper/analysis/random300_fair_baselines.json` | 3B 公平基线（UQ 多信号对照） |
| `paper/analysis/random300_sc9_panda_join.jsonl` | 3B 逐题 join（300 行） |
| `paper/analysis/random300_sc9_panda_join_qwen25_7b.jsonl` | 7B 逐题 join（300 行） |
| `paper/analysis/deepscaler_random300_meta.json` | 300 题 ID 列表与 seed |
| `paper/analysis/compute_random300_planb_metrics.py` | 指标重算脚本（含 3B `PRS` 分数回退） |

摘要文档：`paper/analysis/random300_planb_results_summary.md`（3B/Llama）、`paper/analysis/random300_qwen25_7b_results_summary.md`（7B）。

---

## 输出路径（本地跑数，不入库）

- 3B：`panda-outputs/maintable_qwen25_3b_deepscaler_random300/seed41/deepscaler/`
- 7B：`panda-outputs/maintable_qwen25_7b_deepscaler_random300/seed41/deepscaler/`
- 7B SC k9：`experiments/spurious_consensus/data/samples/samples_qwen25_7b_seed41_deepscaler_random300_k9.jsonl`
