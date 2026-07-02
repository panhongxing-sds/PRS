# Plan B：DeepScaler random300 双模型对比（同一 300 题）

**日期：** 2026-07-02  
**队列：** `deepscaler_random300_meta.json`（seed=42 抽 300 id；SC 与 PANDA 一一对齐）

## 1. 完成度验证

| 模型 | summary.jsonl | k9 | join | 状态 |
|------|---------------|-----|------|------|
| Qwen2.5-3B | 300/300 | 300/300 | 300 | ✅ Phase B 300/300 |
| Llama-3.2-1B | 300/300 | 300/300 | 300 | ✅ Phase B 300/300 |

公平预算：PANDA = 1 greedy + 4 text + 4 weight（共 9 decode）；SC@9 = K=64 池前 9 条（T=0.5）。

---

## 2. SC@9（同一 300 id）

| 指标 | Qwen2.5-3B | Llama-3.2-1B |
|------|------------|--------------|
| 多数票准确率 | 44.0% (132/300) | 27.0% (81/300) |
| blind@9（9/9 全错） | 109 (36.3%) | 148 (49.3%) |
| any_correct@9 | 191 | 152 |
| mean p_top@9 | 0.545 | 0.289 |

**趋势：** Llama 更小、更难：SC 多数票仅 ~27%，blind@9 远高于 Qwen（148 vs 109），共识更弱（mean p_top 0.29 vs 0.54）。

---

## 3. 准确率（n=300）

### Qwen2.5-3B
| 方法 | 准确率 |
|------|--------|
| SC@9 多数票 | 44.0% |
| PANDA@9 公平多数票（9 路 decode） | 50.7% |
| PANDA greedy a0 | 42.0% |

### Llama-3.2-1B
| 方法 | 准确率 |
|------|--------|
| SC@9 多数票 | 27.0% |
| PANDA@9 公平多数票（9 路 decode） | 28.7% |
| PANDA greedy a0 | 28.3% |

**公平 PANDA@9 vs SC@9：** Qwen +6.7 pp；Llama +1.7 pp。

---

## 4. UQ：PANDA vs SC（1−p_top），标签 `label_wrong_clean`

主表 eval：**n=228**（去掉 label_drop=72）。

### Qwen — n=228

| 指标 | SC 1−p_top | PANDA | Δ(PANDA−SC) |
|------|------------|-----|----------|
| AUROC (label_wrong_clean) | 0.823 | 0.862 | +3.95 pp |
| AUPRC | 0.847 | 0.902 | — |


### Llama — n=228

| 指标 | SC 1−p_top | PANDA | Δ(PANDA−SC) |
|------|------------|-----|----------|
| AUROC (label_wrong_clean) | 0.829 | 0.769 | -6.07 pp |
| AUPRC | 0.916 | 0.906 | — |


### Qwen — n=300（参考）

| 指标 | SC 1−p_top | PANDA | Δ(PANDA−SC) |
|------|------------|-----|----------|
| AUROC (label_wrong_clean) | 0.772 | 0.803 | +3.12 pp |
| AUPRC | 0.801 | 0.821 | — |


### Llama — n=300（参考）

| 指标 | SC 1−p_top | PANDA | Δ(PANDA−SC) |
|------|------------|-----|----------|
| AUROC (label_wrong_clean) | 0.721 | 0.706 | -1.47 pp |
| AUPRC | 0.840 | 0.854 | — |


---

## 5. ConfWrong（n=300）

### Qwen2.5-3B — ConfWrong@τ（多数票错且 p_top≥τ）
| τ | SC@9 rate | PANDA@9 rate | SC count | PANDA count |
|---|-----------|--------------|----------|-------------|
| tau_0.9 | 3.67% | 3.67% | 11 | 11 |
| tau_8_over_9 | 5.33% | 4.67% | 16 | 14 |
| tau_7_over_9 | 7.00% | 6.67% | 21 | 20 |


### Llama-3.2-1B — ConfWrong@τ（多数票错且 p_top≥τ）
| τ | SC@9 rate | PANDA@9 rate | SC count | PANDA count |
|---|-----------|--------------|----------|-------------|
| tau_0.9 | 0.00% | 4.33% | 0 | 13 |
| tau_8_over_9 | 1.00% | 6.00% | 3 | 18 |
| tau_7_over_9 | 1.33% | 7.67% | 4 | 23 |


---

## 6. 与仅 Qwen 旧稿 `random300_qwen_results_analysis.md` 对照

- 旧稿 greedy a0 **40.0%**、SC **44.0%**、n=228 PANDA AUROC **0.862** vs SC **0.823**（+3.95 pp）。
- 本次 Qwen 重算：greedy **42.0%**，SC **44.0%**，n=228 PANDA **0.862** vs SC **0.823**（+3.95 pp）。
- 公平 PANDA@9 多数票：旧 confident_wrong 脚本约 **50.7%**；本次 **50.7%**（math_equal 多数票口径一致）。

---

## 7. 叙事结论（双模型）

| 维度 | Qwen | Llama | 双模型是否一致 |
|------|------|-------|----------------|
| 任务准确率（公平 9 票） | PANDA@9 +6.7 pp vs SC | +1.7 pp vs SC | ✅ 两模型公平票均优于/不劣于 SC |
| UQ（greedy wrong, n=228） | PANDA +3.95 pp | -6.07 pp | ⚠️ 见上表 |

**总评：** **条件性正面**：UQ 提升见上表；准确率与 ConfWrong 需按模型分别表述。  
**注意：** SC 与 PANDA 解码协议仍不同（SC 前缀 T=0.5 vs PANDA greedy 扰动）；UQ 标签是 **greedy a0 错**，不是 SC 多数票错。

---

## 产物

- JSON：`random300_planb_metrics.json`
- Join：`random300_sc9_panda_join.jsonl`（Qwen）、`random300_sc9_panda_join_llama.jsonl`（Llama）
- 图：`figures/random300_planb_comparison.png`
