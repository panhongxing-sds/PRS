# proc_calc_flip 广泛验证（4 模型 × 3 数据集）

> N=8121，LODO OOF LR。对比 bd-only / bd+proc_calc_flip / bd+T_tok / bd+TW_ent。

## 1. 各模型 LODO macro

| Model | bd | +proc_calc_flip | Δ | +T_tok | Δ | +TW_ent | Δ |
|-------|---:|---:|---:|---:|---:|---:|---:|
| qwen25_3b | 0.858 | 0.871 | +0.013 | 0.867 | +0.009 | 0.873 | +0.015
| llama32_1b | 0.831 | 0.831 | +0.000 | 0.831 | +0.000 | 0.836 | +0.005
| llama31_8b | 0.858 | 0.855 | -0.003 | 0.852 | -0.006 | 0.854 | -0.003
| qwen3_8b | 0.942 | 0.943 | +0.001 | 0.944 | +0.002 | 0.945 | +0.003

## 2. 四模型 pooled

| Variant | pooled AUROC | tok LODO | bd+tok | Δ vs bd |
|---------|------:|------:|------:|------:|
| proc_calc_flip | 0.656 | 0.590 | 0.901 | +0.003 |
| T_tok_disagree | 0.770 | 0.704 | 0.897 | -0.000 |
| TW_ent_sum | 0.789 | 0.710 | 0.896 | -0.002 |
| bd-only | — | — | 0.897 | 0 |

## 3. 分数据集 drop-one（bd+T → bd，负=去掉 T 后下降）

### qwen25_3b
| ds | bd | +proc_calc | drop | +T_tok drop | +TW_ent drop |
|----|---:|---:|---:|---:|---:|
| minerva | 84.63 | 85.14 | -0.51pp | -0.34pp | -1.80pp |
| math500 | 92.73 | 92.94 | -0.21pp | -0.72pp | -0.18pp |
| gsm8k | 80.07 | 83.24 | -3.18pp | -1.61pp | -2.40pp |

### llama32_1b
| ds | bd | +proc_calc | drop | +T_tok drop | +TW_ent drop |
|----|---:|---:|---:|---:|---:|
| minerva | 75.51 | 75.08 | +0.43pp | -0.06pp | -0.15pp |
| math500 | 85.04 | 84.95 | +0.09pp | -0.46pp | -0.82pp |
| gsm8k | 88.83 | 89.38 | -0.55pp | +0.50pp | -0.42pp |

### llama31_8b
| ds | bd | +proc_calc | drop | +T_tok drop | +TW_ent drop |
|----|---:|---:|---:|---:|---:|
| minerva | 77.50 | 78.05 | -0.55pp | +0.54pp | +0.37pp |
| math500 | 90.11 | 89.65 | +0.46pp | +0.87pp | -0.07pp |
| gsm8k | 89.73 | 88.87 | +0.86pp | +0.26pp | +0.72pp |

### qwen3_8b
| ds | bd | +proc_calc | drop | +T_tok drop | +TW_ent drop |
|----|---:|---:|---:|---:|---:|
| minerva | 92.90 | 93.30 | -0.39pp | -0.15pp | +0.20pp |
| math500 | 97.48 | 97.94 | -0.47pp | -0.34pp | -0.03pp |
| gsm8k | 92.14 | 91.71 | +0.43pp | -0.07pp | -1.10pp |

## 4. 判定

- proc_calc_flip pooled AUROC: **0.656** (>0.6)
- 4模型中 bd+proc_calc > bd-only: **3/4**
- 12格（4模型×3ds）drop-one 为负（去掉 T 伤害）: **7/12**
- macro Δ vs bd: **+0.003**（TW_ent: -0.002，T_tok: -0.000）
