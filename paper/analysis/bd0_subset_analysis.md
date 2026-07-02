# bd=0 子集深度分析 (PANDA N=8121)

> 由 `scripts/analyze_bd0_subset.py` 生成，基于 cache + summary，无 GPU。

## 1. 子集规模

| 子集 | n | wrong | wrong_rate |
|------|--:|------:|-----------:|
| bd=0 | 2960 | 96 | 3.24% |
| bd=0∧wrong | 96 | 96 | 100% (AUROC不可定义) |
| bd>0 | 5161 | — | — |

## 2. bd=0 子集 LODO AUROC/AUPRC

bd 在 bd=0 子集上恒为 0 → AUROC 退化为 0.5。

| 方法 | LODO AUROC | LODO AUPRC |
|------|----------:|----------:|
| fusion_hes_skeleton | 0.884 | 0.474 |
| single_T_formula_skeleton_entropy | 0.860 | 0.545 |
| fusion_hes_flip | 0.769 | 0.329 |
| single_T_n_tokens_avg | 0.764 | 0.531 |
| single_baseline_DC_min | 0.721 | 0.428 |
| single_W_operator_flip_rate | 0.715 | 0.208 |
| single_baseline_SE_H_norm | 0.697 | 0.409 |
| single_baseline_SE_H | 0.673 | 0.357 |
| single_baseline_U_Deg | 0.635 | 0.320 |
| single_T_ent_prox_text | 0.627 | 0.170 |
| single_T_ent_prox_exp_tau16 | 0.623 | 0.144 |
| hesitation_T_ent_prox_lin | 0.623 | 0.160 |
| panda_full_bd_hes | 0.623 | 0.160 |
| single_T_ent_prox_lin | 0.623 | 0.160 |
| single_T_ent_prox_exp_tau32 | 0.620 | 0.156 |
| single_T_ent_prox_weight | 0.610 | 0.143 |
| fusion_hes_tau16 | 0.603 | 0.155 |
| single_TW_ASE_H_norm | 0.518 | 0.088 |
| single_W_n_tokens_avg | 0.513 | 0.269 |
| single_T_earliest_cluster_branch_pos | 0.508 | 0.079 |
| single_T_earliest_cluster_branch_ratio | 0.508 | 0.079 |
| bd_only | 0.500 | 0.065 |
| panda_wo_hesitation | 0.500 | 0.065 |

## 3. 全量条件混合 (bd>0→PANDA, bd=0→过程信号)

| 策略 | LODO AUROC | LODO AUPRC |
|------|----------:|----------:|
| panda_full_reference | 0.904 | 0.875 |
| cond_bd0_hes_else_panda | 0.904 | 0.875 |
| cond_bd0_skeleton_else_panda | 0.879 | 0.750 |
| cond_bd0_tau16_else_panda | 0.905 | 0.875 |
| cond_bd0_flip_else_panda | 0.906 | 0.873 |

## 4. Perturbation 已缓解 collapse

| 代理 | n | wrong | wrong_rate |
|------|--:|------:|-----------:|
| text-only agree (bd_text=0) | 3205 | 126 | 3.93% |
| **full bd=0** (text+weight agree) | 2960 | 96 | 3.24% |
| text_agree proxy (summary) | 3177 | 257 | 8.09% |

- 仅 text rephrase（无 weight）时剩余 wrong ≈ **126**
- 加入 weight perturb 后 bd=0 wrong 降至 **96**（减少 30，23.8%）
- weight 打破 text 一致且仍 wrong 的样本：30/245

### SC-proxy collapse 对照

| 代理 (text 全同 ≈ SC collapse) | n | wrong_rate | 相对 bd=0 |
|--------------------------------|--:|-----------:|----------:|
| text_agree (summary scan) | 3177 | 8.09% | 基线 collapse |
| bd_text=0 (仅 rephrase) | 3205 | 3.93% | −0.69pp |
| **bd=0 (rephrase+weight)** | 2960 | 3.24% | **−4.85pp vs text_agree** |

→ rephrase+weight 将 collapse 子集 wrong_rate 从 ~8% 压到 3.2%；剩余 96 例为「高共识仍错」硬案例。
→ SC DeepScaleR (K=64) AUROC(1−p_top)=0.774 vs PANDA full=0.904（不相交数据集，见 collapse_gain.json）。

## 5. bd=0∧wrong (n=96) 特征

- 按模型: {'qwen25_3b': 38, 'llama32_1b': 28, 'llama31_8b': 24, 'qwen3_8b': 6}
- 按数据集: {'minerva': 23, 'math500': 34, 'gsm8k': 39}
- hesitation 均值 wrong=0.1925 vs correct=0.1517
- TW_ASE_H_norm=0 占比 (wrong): 0.0%

### 子集内有方差的过程信号 (Top)

| 信号 | std(wrong) | |r_pb| | bd=0 AUROC |
|------|----------:|-----:|-----------:|
| `final_number_newly_introduced` | 0.4134 | 0.462 | 0.609 |
| `T_formula_skeleton_entropy` | 0.4376 | 0.215 | 0.814 |
| `base_final_num_support` | 0.4134 | 0.186 | 0.595 |
| `baseline_U_Deg` | 0.2021 | 0.161 | 0.653 |
| `T_ent_prox_exp_tau16` | 0.1274 | 0.156 | 0.688 |
| `text_agree` | 0.1428 | 0.142 | 0.510 |
| `T_ent_prox_exp_tau32` | 0.1210 | 0.133 | 0.665 |
| `T_operator_flip_rate` | 0.1527 | 0.131 | 0.710 |
| `T_numeric_token_flip_rate` | 0.1090 | 0.127 | 0.511 |
| `T_ent_epi_H` | 0.0366 | 0.126 | 0.540 |

## 6. bd=0 内 CORRECT vs WRONG 分离度 Top-15

| 信号 | AUROC | |r_pb| | Cohen d | mean_diff | ρ(bd) |
|------|------:|-----:|--------:|----------:|------:|
| `final_number_newly_introduced` | 0.609 | 0.462 | 2.606 | 0.2188 | 0.238 |
| `T_formula_skeleton_entropy` | 0.814 | 0.215 | -1.216 | -0.4863 | 0.286 |
| `base_final_num_support` | 0.595 | 0.186 | -1.051 | -0.1908 | -0.165 |
| `baseline_U_Deg` | 0.653 | 0.161 | 0.844 | 0.1162 | 0.834 |
| `T_ent_prox_exp_tau16` | 0.688 | 0.156 | 0.878 | 0.0541 | 0.547 |
| `text_agree` | 0.510 | 0.142 | -0.802 | -0.0208 | -0.816 |
| `T_ent_prox_exp_tau32` | 0.665 | 0.133 | 0.749 | 0.0508 | 0.492 |
| `T_operator_flip_rate` | 0.710 | 0.131 | 0.739 | 0.2146 | -0.306 |
| `T_numeric_token_flip_rate` | 0.511 | 0.127 | 0.739 | 0.0142 | 0.571 |
| `T_ent_epi_H` | 0.540 | 0.126 | -0.709 | -0.0105 | -0.007 |
| `W_ans_ent_top10_mean_avg` | 0.626 | 0.125 | -0.706 | -0.1516 | 0.334 |
| `W_ans_ent_max_avg` | 0.626 | 0.125 | -0.704 | -0.1547 | 0.350 |
| `final_answer_equiv_last_equation` | 0.596 | 0.123 | -0.693 | -0.1926 | -0.284 |
| `final_answer_in_last_k_equations` | 0.596 | 0.123 | -0.693 | -0.1926 | -0.284 |
| `W_ans_ent_mean_avg` | 0.627 | 0.120 | -0.678 | -0.0207 | 0.423 |

## 7. 叙事支持摘要

1. **Perturbation 已大幅削减 confident-wrong**：text-only wrong n=126 → bd=0 wrong n=96（−30）。
2. **bd=0 上 hesitation 是增量**：bd AUROC=0.5，hes LODO=0.623（+0.123 vs bd）。
3. **bd=0 最佳单信号**：`T_formula_skeleton_entropy` AUROC=0.860；skeleton=0.860 vs hes=0.623。
4. bd=0∧wrong n=96 全为 wrong → 无法算 AUROC，但过程信号仍有方差可用于 rank / case study。

