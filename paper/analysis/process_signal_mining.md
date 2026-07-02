# 过程级信号挖掘报告 (N=8121)

**方法**：与 `aggregate_panda_v2.py` 一致的 macro LODO（3 数据集 leave-one-out + 3 seed 均值）；单信号用 LODO-LR；融合用 `bd + proc` 的 LODO-LR。数据来自 `.proc_near_answer_final_cache.pkl` + `summary.jsonl` 全字段扫描（175 个过程候选，排除 bd 代理/答案分散特征）。

---

## 1. 参考基线

| 方法 | 全量 LODO AUROC | AUPRC | 说明 |
|------|----------------:|------:|------|
| **bd (Dissent)** | 0.897 | 0.850 | 主驱动 |
| **PANDA (bd + T_ent_prox_lin)** | **0.904** | 0.875 | 当前方法 |
| **Hesitation (T_ent_prox_lin)** | 0.704 | 0.622 | 过程信号 #46/175 |
| SE (baseline_SE_H) | 0.890* | 0.865 | *n=4261，仅多簇样本 |
| DeepConf (DC_min) | 0.701 | 0.667 | token 置信度基线 |

- Spearman(bd, hesitation) = **0.429**（部分互补，非正交）
- w/o Hesitation 消融：0.904→0.897（−0.007）；w/o Dissent：0.904→0.704
- **45 个过程信号**全量 LODO AUROC 超过 hesitation，但多数与 bd 高度相关（>0.7），不能视为独立互补

---

## 2. 子集规模

| 子集 | n | 备注 |
|------|--:|------|
| all | 8121 | |
| bd=0 | 2960 | dissent 坍缩 |
| bd=0 & wrong | 96 | 全为 wrong，AUROC 不可定义 |
| wrong-only | 3300 | |
| bd>0 | 5161 | |

---

## 3. Top 过程候选（按场景）

### 3a. 全量 LODO 最强（过程定义，排除 bd 代理）

| 信号 | LODO AUROC | ρ(bd) | bd=0 AUROC | 解读 |
|------|----------:|------:|-----------:|------|
| response_fragmentation | 0.850 | 0.847 | 0.506 | 与 bd 几乎共线，**非互补** |
| baseline_U_Deg | 0.803 | 0.834 | 0.653 | 图不确定性；bd=0 可用 |
| AltMass_reason_all_tw8 | 0.802 | 0.801 | 0.506 | 备选答案质量；bd=0 无效 |
| T_math_token_flip_mean | 0.774 | 0.731 | 0.513 | 扰动下算子/token 翻转 |
| W_tok_disagree_mean | 0.759 | 0.720 | 0.528 | 跨 run token 不一致 |
| **T_ent_prox_lin (hesitation)** | **0.704** | 0.429 | **0.626** | 当前 hesitation |

### 3b. **bd=0 坍缩子集**（最关键：dissent 失效）

| 信号 | bd=0 raw AUROC | bd=0 LODO | ρ(bd) | vs hesitation (0.623) |
|------|---------------:|----------:|------:|------------------------|
| **T_formula_skeleton_entropy** | **0.814** | **0.860** | 0.286 | **+0.24** |
| T_earliest_cluster_branch_pos† | 0.750 | 0.508 | −0.10 | +0.09（†bd=0 仅 n=29 有值，不可靠） |
| W_operator_flip_rate | 0.683 | **0.715** | −0.31 | +0.09 |
| T_ent_prox_exp_tau16 | 0.688 | 0.623 | 0.547 | +0.06 |
| baseline_U_Deg | 0.653 | — | 0.834 | +0.03 |
| SE (答案分散) | — | 0.708 | — | +0.09 |
| hesitation | 0.626 | 0.623 | 0.429 | — |

**结论**：在 bd=0 上，**T_formula_skeleton_entropy**（推理公式骨架熵）是覆盖最全、最强的过程信号，显著优于 hesitation 和 SE。

### 3c. 与 bd 低相关 (|ρ|<0.35) 且 bd=0 可用

| 信号 | 全量 LODO | bd=0 AUROC | ρ(bd) |
|------|----------:|-----------:|------:|
| T_formula_skeleton_entropy | 0.536 | 0.814 | 0.286 |
| W_operator_flip_rate | 0.567 | 0.683 | −0.309 |
| baseline_LL_nll | 0.561 | 0.633 | −0.200 |
| baseline_SAR | 0.581 | 0.597 | −0.222 |
| T_mar_sum_total | 0.521 | 0.709 | 0.165 |

---

## 4. 与 bd 融合（能否替代 hesitation 改进 PANDA？）

| 融合 (LODO-LR) | 全量 AUROC | Δ vs bd | Δ vs PANDA | bd=0 LODO |
|----------------|----------:|--------:|-----------:|----------:|
| PANDA (bd + hes) | 0.904 | +0.007 | — | 0.623 |
| bd + W_n_tokens_avg | **0.907** | +0.010 | **+0.003** | 0.513 |
| bd + T_mar_sum_total | 0.906 | +0.008 | +0.001 | — |
| bd + W_operator_flip_rate | 0.902 | +0.005 | −0.002 | **0.715** |
| bd + T_formula_skeleton | 0.894 | −0.004 | −0.011 | **0.860** |
| bd + T_earliest_cluster_branch | 0.900 | +0.002 | −0.005 | 0.487 |

- **全量最优替换**：`bd + W_n_tokens_avg`（+0.003），但 token 数更像长度/复杂度代理，叙事上弱
- **bd=0 最优单信号**：`T_formula_skeleton_entropy`（0.860），但单独全量仅 0.536，与 bd 融合反而略降
- **平衡互补**：`bd + W_operator_flip_rate` 在 bd=0 上 0.715（优于 hes 0.623），全量 0.902（略低于 PANDA）

**没有任何单一过程信号在全量上既 beat hesitation 又保持低 bd 相关**；高 AUROC 的过程信号（fragmentation、AltMass）与 bd 共线。

---

## 5. 信号类别盘点（175 候选）

| 类别 | 代表字段 | 全量最强 | bd=0 表现 | 与 bd 互补性 |
|------|----------|----------|-----------|-------------|
| 近答案熵 (proximity) | T_ent_prox_lin, exp_tau16/32 | prox_lin 0.704 | exp_tau16 0.688 > lin 0.626 | 中等 (ρ≈0.43–0.55) |
| 时序熵 | T_ent_slope, late20_minus_early50 | 0.458 | ~0.50 | **极弱**，T0 类机制确认 |
| 公式骨架 | T_formula_skeleton_entropy | 0.536 | **0.814** | **最佳 bd=0 过程信号** |
| 扰动 token 动态 | W_operator_flip_rate, T_math_token_flip | 0.57–0.77 | flip 0.683 | 负相关，bd=0 有效 |
| AltMass 备选质量 | AltMass_reason_all_tw8 | 0.802 | ~0.506 | 高 bd 相关，bd=0 无效 |
| 图/fragmentation | response_fragmentation, U_Deg | 0.80–0.85 | ~0.506–0.653 | 高 bd 相关 |
| Margin / NLL | T_mar_sum_total, baseline_LL_nll | 0.52–0.56 | 0.63–0.71 | 低–中相关，fusion 微增益 |
| DeepConf / SAR | baseline_DC_min, SAR | 0.58–0.70 | 0.55–0.60 | 弱于 prox 变体 |

---

## 6. 叙事建议

### Hesitation (T_ent_prox_lin)：建议 **降级 (demote)**，保留为辅助消融
- 全量贡献仅 +0.007 AUROC；排名 #46/175
- bd=0 上弱于 SE (0.708)、远弱于 skeleton entropy (0.814)
- 时序变体 (slope, late−early) 几乎无预测力（与 mechanism T0≈0.50 一致）
- **不建议**与 Dissent 等权重写进标题；可改称 "proximity-weighted process entropy"

### 值得新增消融/讨论的过程信号
1. **T_formula_skeleton_entropy** — bd=0 坍缩场景的"过程级安全网"；建议 `bd=0` 条件 AUROC 表 + 案例
2. **W_operator_flip_rate** — 权重扰动下算子翻转，ρ(bd)=−0.31，bd=0 LODO 0.715
3. **T_ent_prox_exp_tau16** — 同族熵信号，bd=0 优于 linear prox（0.688 vs 0.626）

### 是否替换 PANDA 第二特征？
- 若追求 **全量数字**：可试 `bd + T_mar_sum_total` 或 `bd + W_n_tokens_avg`（+0.001–0.003），但叙事需解释
- 若追求 **bd=0 叙事**：应用 skeleton entropy 作条件分析/混合专家（bd=0 时用 skeleton，bd>0 时用 bd），而非全局替换
- **不推荐**用 skeleton 全局替换 hesitation（全量 −0.011）

### bd0_wrong (n=96)
- 标签全为 1，AUROC 不可计算；仅能作描述统计（hesitation/bd 常数 0）

---

## 7. 输出文件

- 详细 JSON：`paper/analysis/process_signal_mining.json`
- 本报告：`paper/analysis/process_signal_mining.md`
- 复现脚本：`scripts/mine_process_signals.py`
