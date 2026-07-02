# 附录：DeepScaleR random300 · Qwen2.5-3B vs 7B 公平对比

**更新日期：** 2026-07-02  
**Cohort：** `paper/analysis/deepscaler_random300_meta.json`（**n=300**，`seed=42`，自 2000 题随机抽样）  
**公平解码预算：** `SE_SAMPLES=0`，`N_REPHRASES=4`，`WEIGHT_SEEDS=42,43,44,45` → **每题 9 次 decode**（1 greedy + 4×2 rephrase 权重路径）；PANDA@9 公平多数票与 SC@9 使用相同 9 路预算。

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

## 表 1 · 准确率（n=300）

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

## 表 2 · 不确定性（UQ，label_wrong_clean）

### 全量 n=300

| 指标 | Qwen2.5-3B | Qwen2.5-7B | Δ(PANDA−SC) 3B | Δ(PANDA−SC) 7B |
|------|------------|------------|----------------|----------------|
| SC `1−p_top` AUROC | 0.772 | 0.773 | — | — |
| PANDA AUROC | 0.803 | 0.784 | +3.12 pp | +1.14 pp |
| SC AUPRC | 0.801 | 0.741 | — | — |
| PANDA AUPRC | 0.821 | 0.711 | — | — |

### 论文 eval（无 label_drop）

| 指标 | 3B (n=228) | 7B (n=239) |
|------|------------------|------------------|
| SC AUROC | 0.823 | 0.789 |
| PANDA AUROC | 0.862 | 0.825 |
| **Δ(PANDA−SC)** | **+3.95 pp** | **+3.60 pp** |
| SC AUPRC | 0.847 | 0.747 |
| PANDA AUPRC | 0.902 | 0.769 |

`label_drop` 计数：3B **72**，7B **61**。

---

## 表 3 · ConfWrong@τ（n=300）

τ=**0.9** 与 τ=**8/9**（0.888…）；列为 **全体样本率** / **给定多数票错误条件下的率**（括号内为计数，见 JSON）。

### τ = 0.9

| 方法 | Qwen2.5-3B | Qwen2.5-7B |
|------|------------|------------|
| SC@9 | 3.7% (11) / 6.5% | 6.7% (20) / 13.5% |
| PANDA@9 投票 | 3.7% (11) / 7.4% | 4.3% (13) / 10.4% |

### τ = 8/9

| 方法 | Qwen2.5-3B | Qwen2.5-7B |
|------|------------|------------|
| SC@9 | 5.3% (16) / 9.5% | 8.7% (26) / 17.6% |
| PANDA@9 投票 | 4.7% (14) / 9.5% | 6.3% (19) / 15.2% |

PANDA a0 错误且 `bd≈0`（collapse）计数：3B **40**，7B **42**。

---

## 表 4 · SC@9 队列统计（n=300）

| 统计量 | Qwen2.5-3B | Qwen2.5-7B |
|--------|------------|------------|
| 多数票正确 | 132 | 152 |
| 多数票错误 | 168 | 148 |
| 多数票准确率 | 44.0% | 50.7% |
| blind@9（9 路全错） | 109 (36.3%) | 94 (31.3%) |
| 至少一路正确@9 | 191 | 206 |
| `p_top` 均值 | 0.545 | 0.646 |
| `p_top` min / max | 0.111 / 1.000 | 0.111 / 1.000 |

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
